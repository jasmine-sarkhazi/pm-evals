"""GPT models via the official OpenAI SDK, and any OpenAI-compatible endpoint
(Gemini's compatibility layer, Groq, Mistral, Ollama, vLLM ...)."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from .base import LLMProvider, ProviderError, ToolCallRequest, Turn, parse_json_object


def _to_openai_messages(system: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    for msg in history:
        role = msg["role"]
        if role == "user":
            messages.append({"role": "user", "content": msg["content"]})
        elif role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or None}
            if msg.get("tool_calls"):
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.raw_args or json.dumps(tc.args)},
                    }
                    for tc in msg["tool_calls"]
                ]
            messages.append(entry)
        elif role == "tool":
            messages.append({"role": "tool", "tool_call_id": msg["tool_call_id"], "content": msg.get("content") or ""})
    return messages


def _sanitize_schema(schema: Any) -> Any:
    """Drop pydantic ``title`` annotations (some compatible endpoints reject
    them) without touching property *names* such as a parameter called title."""
    if isinstance(schema, dict):
        out: dict[str, Any] = {}
        for k, v in schema.items():
            if k == "title" and not isinstance(v, dict):
                continue
            if k in ("properties", "$defs", "definitions", "patternProperties") and isinstance(v, dict):
                out[k] = {pk: _sanitize_schema(pv) for pk, pv in v.items()}
            else:
                out[k] = _sanitize_schema(v)
        return out
    if isinstance(schema, list):
        return [_sanitize_schema(v) for v in schema]
    return schema


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, label: str = "openai"):
        super().__init__(model)
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("pip install openai") from exc
        self._openai = openai
        self.name = label
        kwargs: dict[str, Any] = {}
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if base_url:
            kwargs["base_url"] = base_url
            key = key or "not-needed"
        if key:
            kwargs["api_key"] = key
        self.client = openai.AsyncOpenAI(**kwargs)

    def _tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description") or "",
                    "parameters": _sanitize_schema(t["input_schema"]),
                },
            }
            for t in tools
        ]

    async def complete(self, system: str, history: list[dict[str, Any]], tools: list[dict[str, Any]], max_tokens: int = 4096) -> Turn:
        params: dict[str, Any] = {"model": self.model, "messages": _to_openai_messages(system, history)}
        if tools:
            params["tools"] = self._tools(tools)
        # Newer OpenAI models reject max_tokens in favour of max_completion_tokens.
        if self.name == "openai":
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        try:
            response = await self.client.chat.completions.create(**params)
        except self._openai.APIStatusError as exc:
            raise ProviderError(f"{self.name} API error {exc.status_code}: {exc.message}") from exc
        except self._openai.APIConnectionError as exc:
            raise ProviderError(f"{self.name} connection error: {exc}") from exc
        choice = response.choices[0]
        msg = choice.message
        calls: list[ToolCallRequest] = []
        for tc in msg.tool_calls or []:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            raw = fn.arguments or "{}"
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    args = {"_value": args}
                calls.append(ToolCallRequest(id=tc.id, name=fn.name, args=args))
            except Exception:
                calls.append(ToolCallRequest(id=tc.id, name=fn.name, args={}, raw_args=raw))
        usage = {}
        if response.usage:
            usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
        stop = choice.finish_reason or "stop"
        return Turn(
            text=(msg.content or "").strip(),
            tool_calls=calls,
            stop_reason="tool_use" if calls else stop,
            usage=usage,
            raw=response.model_dump(mode="json"),
        )

    async def json_completion(self, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 4096) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "result", "schema": _sanitize_schema(schema)}},
        }
        if self.name == "openai":
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        try:
            response = await self.client.chat.completions.create(**params)
        except self._openai.BadRequestError:
            return await super().json_completion(system, prompt, schema, max_tokens)
        except self._openai.APIStatusError as exc:
            raise ProviderError(f"{self.name} API error {exc.status_code}: {exc.message}") from exc
        text = response.choices[0].message.content or ""
        try:
            return json.loads(text)
        except Exception:
            return parse_json_object(text)

    async def aclose(self) -> None:
        await self.client.close()
