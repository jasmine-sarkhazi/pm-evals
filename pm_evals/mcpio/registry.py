"""Tool registry: merges tools from every configured server plus distractors
into one list the model sees, and routes calls back to the right server."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..models import ActualToolCall, DistractorTool, MCPServerConfig
from .client import MCPConnection, ToolDef, ToolOutcome


@dataclass
class RegisteredTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    server: Optional[str] = None
    distractor: bool = False
    allowed: bool = True
    """False when the server's ``available_tools`` allow-list excludes it. The
    tool is still shown to the model (so we can detect permission escapes) but
    calls are refused."""
    definition: Optional[ToolDef] = None


@dataclass
class ToolRegistry:
    tools: dict[str, RegisteredTool] = field(default_factory=dict)
    connections: dict[str, MCPConnection] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    @classmethod
    async def build(
        cls,
        connections: list[MCPConnection],
        distractors: Optional[list[DistractorTool]] = None,
        hide_disallowed: bool = False,
        shuffle_seed: Optional[int] = None,
    ) -> "ToolRegistry":
        reg = cls()
        for conn in connections:
            reg.connections[conn.name] = conn
            allow = conn.config.available_tools
            for t in await conn.list_tools():
                name = t.name
                if name in reg.tools:  # collision across servers -> prefix
                    name = f"{conn.name}__{t.name}"
                allowed = allow is None or t.name in allow
                if hide_disallowed and not allowed:
                    continue
                reg.tools[name] = RegisteredTool(
                    name=name,
                    description=t.description,
                    input_schema=t.input_schema,
                    server=conn.name,
                    allowed=allowed,
                    definition=t,
                )
                reg.order.append(name)
        for d in distractors or []:
            if d.name in reg.tools:
                continue
            reg.tools[d.name] = RegisteredTool(
                name=d.name, description=d.description, input_schema=d.input_schema, distractor=True
            )
            reg.order.append(d.name)
        if shuffle_seed is not None:
            import random

            random.Random(shuffle_seed).shuffle(reg.order)
        return reg

    # -- views -------------------------------------------------------------
    def llm_tools(self) -> list[dict[str, Any]]:
        """Provider-neutral tool definitions in the order the model sees them."""
        out = []
        for name in self.order:
            t = self.tools[name]
            schema = dict(t.input_schema or {})
            schema.setdefault("type", "object")
            schema.setdefault("properties", {})
            out.append({"name": t.name, "description": t.description or "", "input_schema": schema})
        return out

    def real_tool_names(self) -> list[str]:
        return [n for n in self.order if not self.tools[n].distractor]

    def distractor_names(self) -> list[str]:
        return [n for n in self.order if self.tools[n].distractor]

    def schema_for(self, name: str) -> Optional[dict[str, Any]]:
        t = self.tools.get(name)
        return t.input_schema if t else None

    def get(self, name: str) -> Optional[RegisteredTool]:
        return self.tools.get(name)

    # -- execution ---------------------------------------------------------
    async def execute(self, name: str, args: dict[str, Any], order: int) -> tuple[ActualToolCall, Optional[ToolOutcome]]:
        t = self.tools.get(name)
        if t is None:
            call = ActualToolCall(
                name=name,
                args=args,
                result="ERROR: unknown tool. This tool does not exist.",
                is_error=True,
                order=order,
                exists=False,
            )
            return call, None
        if t.distractor:
            call = ActualToolCall(
                name=name,
                args=args,
                result="ERROR: this tool is not available in this environment.",
                is_error=True,
                order=order,
                exists=True,
                distractor=True,
            )
            return call, None
        if not t.allowed:
            call = ActualToolCall(
                name=name,
                args=args,
                result="ERROR: permission denied. This tool is not allowed for this task.",
                is_error=True,
                order=order,
                server=t.server,
            )
            return call, None
        conn = self.connections[t.server]  # type: ignore[index]
        real_name = t.definition.name if t.definition else name
        outcome = await conn.call_tool(real_name, args)
        call = ActualToolCall(
            name=name,
            args=args,
            result=outcome.text,
            structured_result=outcome.structured,
            is_error=outcome.is_error,
            order=order,
            server=t.server,
            duration_ms=round(outcome.duration_ms, 1),
        )
        return call, outcome


async def open_connections(servers: list[MCPServerConfig], in_process: Optional[dict[str, Any]] = None) -> list[MCPConnection]:
    """Open every server; caller closes them via :func:`close_connections`."""
    conns: list[MCPConnection] = []
    try:
        for cfg in servers:
            conn = MCPConnection(cfg, in_process_server=(in_process or {}).get(cfg.server_name))
            await conn.__aenter__()
            conns.append(conn)
    except Exception:
        await close_connections(conns)
        raise
    return conns


async def close_connections(conns: list[MCPConnection]) -> None:
    for conn in reversed(conns):
        await conn.__aexit__(None, None, None)
