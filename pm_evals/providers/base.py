"""Provider-neutral message format and interface.

History is a list of dicts:

* ``{"role": "user", "content": str}``
* ``{"role": "assistant", "content": str | None, "tool_calls": [ToolCallRequest...]}``
* ``{"role": "tool", "tool_call_id": str, "name": str, "content": str, "is_error": bool}``

Each provider converts to/from its native shape. Tool definitions use the
MCP-style ``{"name", "description", "input_schema"}`` dict.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCallRequest:
    id: str
    name: str
    args: dict[str, Any]
    raw_args: Optional[str] = None
    """The raw argument string when the provider returned invalid JSON."""


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCallRequest]
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None
    thinking: Optional[str] = None


class ProviderError(RuntimeError):
    pass


class LLMProvider:
    """Abstract provider. Concrete classes implement :meth:`complete` and
    :meth:`json_completion`."""

    name = "base"

    def __init__(self, model: str):
        self.model = model

    async def complete(
        self,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> Turn:
        raise NotImplementedError

    async def json_completion(self, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 4096) -> dict[str, Any]:
        """Return a JSON object matching ``schema``. Default implementation
        asks for JSON in text and parses it; providers with structured output
        support override this."""
        turn = await self.complete(
            system + "\n\nRespond with a single JSON object only. No prose, no markdown fences.",
            [{"role": "user", "content": prompt + f"\n\nJSON schema:\n{json.dumps(schema)}"}],
            tools=[],
            max_tokens=max_tokens,
        )
        return parse_json_object(turn.text)

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ProviderError("empty response where JSON was expected")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = _FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    raise ProviderError(f"could not parse JSON from model response: {text[:200]}")
