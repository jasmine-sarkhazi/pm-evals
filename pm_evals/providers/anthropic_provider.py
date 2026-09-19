"""Claude models via the official Anthropic SDK."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from .base import LLMProvider, ProviderError, ToolCallRequest, Turn, parse_json_object


def _to_anthropic_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal pending_results
        if pending_results:
            messages.append({"role": "user", "content": pending_results})
            pending_results = []

    for msg in history:
        role = msg["role"]
        if role == "user":
            flush()
            messages.append({"role": "user", "content": msg["content"]})
        elif role == "assistant":
            flush()
            raw_blocks = msg.get("raw_blocks")
            if raw_blocks:
                # Replay the assistant turn exactly as the API returned it so
                # thinking blocks (required when thinking is on) stay intact.
                messages.append({"role": "assistant", "content": raw_blocks})
                continue
            blocks: list[dict[str, Any]] = []
            if msg.get("content"):
                blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg.get("tool_calls", []):
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args})
            if not blocks:
                blocks.append({"type": "text", "text": "(no output)"})
            messages.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg.get("content") or "",
                    "is_error": bool(msg.get("is_error")),
                }
            )
    flush()
    return messages


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(model)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("pip install anthropic") from exc
        kwargs: dict[str, Any] = {}
        if api_key or os.environ.get("ANTHROPIC_API_KEY"):
            kwargs["api_key"] = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if base_url:
            kwargs["base_url"] = base_url
        self._anthropic = anthropic
        self.client = anthropic.AsyncAnthropic(**kwargs)

    async def complete(self, system: str, history: list[dict[str, Any]], tools: list[dict[str, Any]], max_tokens: int = 4096) -> Turn:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": _to_anthropic_messages(history),
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [
                {"name": t["name"], "description": t.get("description") or "", "input_schema": t["input_schema"]} for t in tools
            ]
        try:
            response = await self.client.messages.create(**params)
        except self._anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise ProviderError(f"Anthropic connection error: {exc}") from exc

        text_parts: list[str] = []
        calls: list[ToolCallRequest] = []
        thinking: list[str] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                calls.append(ToolCallRequest(id=block.id, name=block.name, args=args))
            elif block.type == "thinking" and getattr(block, "thinking", None):
                thinking.append(block.thinking)
        usage = {}
        if response.usage:
            usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        stop = response.stop_reason or "end_turn"
        if stop == "refusal":
            text_parts.append("[model refused the request]")
        return Turn(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            stop_reason=stop,
            usage=usage,
            raw=response.model_dump(mode="json"),
            thinking="\n".join(thinking) or None,
            raw_blocks=[b.model_dump(mode="json", exclude_none=True) for b in response.content],
        )

    async def json_completion(self, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 4096) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        try:
            response = await self.client.messages.create(**params)
        except self._anthropic.BadRequestError:
            # Older models / proxies without structured outputs: fall back to text parsing.
            return await super().json_completion(system, prompt, schema, max_tokens)
        except self._anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except Exception:
            return parse_json_object(text)

    async def aclose(self) -> None:
        await self.client.close()
