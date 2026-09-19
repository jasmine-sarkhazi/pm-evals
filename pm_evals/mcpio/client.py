"""Thin async wrapper over the MCP Python SDK client.

Supports streamable-http (with custom headers), SSE and stdio transports, plus
in-process servers (used by tests and the bundled demo).
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Optional

from mcp.client import Client
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from ..models import MCPServerConfig


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]
    server: str
    annotations: dict[str, Any] = field(default_factory=dict)
    output_schema: Optional[dict[str, Any]] = None
    title: Optional[str] = None

    def fingerprint(self) -> str:
        payload = json.dumps(
            {"name": self.name, "description": self.description, "input_schema": self.input_schema},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "server": self.server,
            "annotations": self.annotations,
            "output_schema": self.output_schema,
            "title": self.title,
            "fingerprint": self.fingerprint(),
        }


@dataclass
class ToolOutcome:
    text: str
    structured: Any = None
    is_error: bool = False
    raw: Any = None
    duration_ms: float = 0.0
    malformed: list[str] = field(default_factory=list)
    """Protocol-level problems noticed on the response (used by conformance)."""


def _content_to_text(content: Any) -> str:
    parts: list[str] = []
    for block in content or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append(getattr(block, "text", ""))
        elif btype == "image":
            parts.append("[image]")
        elif btype == "resource":
            res = getattr(block, "resource", None)
            parts.append(getattr(res, "text", None) or "[resource]")
        elif btype == "resource_link":
            parts.append(f"[resource_link {getattr(block, 'uri', '')}]")
        else:
            parts.append(str(block))
    return "\n".join(p for p in parts if p is not None)


class MCPConnection:
    """One live connection to one MCP server."""

    def __init__(self, config: MCPServerConfig, in_process_server: Any = None):
        self.config = config
        self.name = config.server_name
        self._in_process = in_process_server
        self._stack: Optional[AsyncExitStack] = None
        self.client: Optional[Client] = None
        self.handshake: dict[str, Any] = {}
        self._tools_cache: Optional[list[ToolDef]] = None

    # -- lifecycle ---------------------------------------------------------
    async def __aenter__(self) -> "MCPConnection":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        cfg = self.config
        started = time.perf_counter()
        try:
            if self._in_process is not None:
                self.client = Client(self._in_process, mode="legacy")
            elif cfg.transport == "streamable-http":
                if not cfg.url:
                    raise ValueError(f"server '{cfg.server_name}': url is required for streamable-http")
                http_client = create_mcp_http_client(headers=cfg.headers or None)
                await self._stack.enter_async_context(http_client)
                transport = streamable_http_client(cfg.url, http_client=http_client)
                self.client = Client(transport)
            elif cfg.transport == "sse":
                if not cfg.url:
                    raise ValueError(f"server '{cfg.server_name}': url is required for sse")
                self.client = Client(sse_client(cfg.url, headers=cfg.headers or None))
            elif cfg.transport == "stdio":
                if not cfg.command:
                    raise ValueError(f"server '{cfg.server_name}': command is required for stdio")
                import os

                params = StdioServerParameters(command=cfg.command, args=list(cfg.args), env={**os.environ, **cfg.env})
                self.client = Client(params)
            else:  # pragma: no cover - validated by pydantic
                raise ValueError(f"unknown transport {cfg.transport}")
            await self._stack.enter_async_context(self.client)
        except Exception:
            await self._stack.aclose()
            self._stack = None
            raise
        elapsed = (time.perf_counter() - started) * 1000
        info = self.client.server_info
        caps = self.client.server_capabilities
        self.handshake = {
            "ok": True,
            "connect_ms": round(elapsed, 1),
            "protocol_version": getattr(self.client, "protocol_version", None),
            "server_info": info.model_dump(mode="json") if info is not None else None,
            "capabilities": caps.model_dump(mode="json", exclude_none=True) if caps is not None else None,
            "instructions": self.client.instructions,
        }
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                pass
            self._stack = None
            self.client = None

    # -- operations --------------------------------------------------------
    async def list_tools(self, refresh: bool = False) -> list[ToolDef]:
        assert self.client is not None, "connection not open"
        if self._tools_cache is not None and not refresh:
            return self._tools_cache
        result = await self.client.list_tools(cache_mode="bypass" if refresh else "use")
        tools: list[ToolDef] = []
        for t in result.tools:
            ann = t.annotations.model_dump(mode="json", exclude_none=True) if t.annotations else {}
            tools.append(
                ToolDef(
                    name=t.name,
                    description=t.description or "",
                    input_schema=dict(t.input_schema or {"type": "object", "properties": {}}),
                    server=self.name,
                    annotations=ann,
                    output_schema=dict(t.output_schema) if t.output_schema else None,
                    title=t.title,
                )
            )
        self._tools_cache = tools
        return tools

    async def raw_list_tools(self) -> Any:
        assert self.client is not None
        return await self.client.list_tools(cache_mode="bypass")

    async def call_tool(self, name: str, args: dict[str, Any], timeout: Optional[float] = 120.0) -> ToolOutcome:
        assert self.client is not None, "connection not open"
        started = time.perf_counter()
        malformed: list[str] = []
        try:
            result = await self.client.call_tool(name, args or {}, read_timeout_seconds=timeout)
        except Exception as exc:  # transport / protocol error
            return ToolOutcome(
                text=f"ERROR: {type(exc).__name__}: {exc}",
                is_error=True,
                raw=None,
                duration_ms=(time.perf_counter() - started) * 1000,
                malformed=[f"exception:{type(exc).__name__}"],
            )
        elapsed = (time.perf_counter() - started) * 1000
        content = getattr(result, "content", None)
        if content is None:
            malformed.append("missing content array")
        text = _content_to_text(content)
        structured = getattr(result, "structured_content", None)
        if structured is None and text:
            try:
                structured = json.loads(text)
            except Exception:
                structured = None
        try:
            raw = result.model_dump(mode="json", exclude_none=True)
        except Exception:
            raw = str(result)
        return ToolOutcome(
            text=text,
            structured=structured,
            is_error=bool(getattr(result, "is_error", False)),
            raw=raw,
            duration_ms=elapsed,
            malformed=malformed,
        )

    async def ping(self) -> bool:
        assert self.client is not None
        try:
            await self.client.send_ping()
            return True
        except Exception:
            return False


async def probe_server(config: MCPServerConfig, in_process_server: Any = None) -> dict[str, Any]:
    """Connect, list tools and return a JSON-friendly description. Used by the
    UI "Test connection" button."""
    async with MCPConnection(config, in_process_server=in_process_server) as conn:
        tools = await conn.list_tools()
        return {
            "server_name": config.server_name,
            "handshake": conn.handshake,
            "tools": [t.to_dict() for t in tools],
        }
