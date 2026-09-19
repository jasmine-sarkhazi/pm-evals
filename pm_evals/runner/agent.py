"""Run one eval case through the "api" harness: the model under test drives
the MCP tools in a loop and we record the full trajectory."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..mcpio.registry import ToolRegistry
from ..models import ActualToolCall, CleanupConfig, CreatedEntity, EvalCase, RunConfig
from ..providers.base import LLMProvider, ProviderError

_CREATE_STEMS = ("create", "add", "new", "insert", "register", "make", "build", "clone", "copy", "upload", "schedule", "publish", "generate")


def system_prompt_for(case: EvalCase, cfg: RunConfig) -> str:
    base = case.system_prompt or (
        "You are an assistant that completes the user's request by calling the available tools. "
        "Use tools rather than describing what you would do. When the request is fully handled, "
        "reply with a short summary of what you did and the results. If something failed, say so plainly."
    )
    if cfg.cleanup.enabled and cfg.cleanup.entity_prefix:
        base += (
            f"\n\nThis is a test environment. Every entity you create (segments, campaigns, records, etc.) "
            f"must have a name that starts with the prefix \"{cfg.cleanup.entity_prefix}\"."
        )
    return base


@dataclass
class AgentTrace:
    calls: list[ActualToolCall] = field(default_factory=list)
    output: str = ""
    trajectory: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    turns: int = 0
    stop_reason: str = ""


def _looks_like_id(key: str, value: Any, id_fields: list[str]) -> bool:
    if not isinstance(value, (str, int)):
        return False
    k = key.lower()
    return k in {f.lower() for f in id_fields} or k.endswith("_id") or k == "id"


def extract_created_entities(call: ActualToolCall, cleanup: CleanupConfig) -> list[CreatedEntity]:
    """Best-effort: a successful call to a create-like tool whose result contains
    an id-like field is a created entity we may need to clean up."""
    base = call.name.split("__")[-1].lower()
    if call.is_error or not call.exists or call.distractor:
        return []
    if not base.startswith(_CREATE_STEMS) and not any(f"_{s}_" in f"_{base}_" for s in _CREATE_STEMS):
        return []
    data = call.structured_result
    if data is None and call.result:
        try:
            data = json.loads(call.result)
        except Exception:
            data = None
    entities: list[CreatedEntity] = []
    name = call.args.get("name") or call.args.get("title") or call.args.get("label")

    def collect(obj: Any) -> None:
        if isinstance(obj, dict):
            eid = None
            for k, v in obj.items():
                if _looks_like_id(k, v, cleanup.id_fields) and k.lower() in {f.lower() for f in cleanup.id_fields}:
                    eid = str(v)
                    break
            if eid is None:
                for k, v in obj.items():
                    if k.lower() == "id" and isinstance(v, (str, int)):
                        eid = str(v)
                        break
            if eid is not None:
                entities.append(CreatedEntity(server=call.server, tool=call.name, entity_id=eid, name=obj.get("name") or name, raw={k: obj[k] for k in list(obj)[:8]}))
                return
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    collect(v)
        elif isinstance(obj, list):
            for v in obj[:5]:
                collect(v)

    collect(data)
    if not entities and call.result:
        m = re.search(r"\b(id|ID|Id)\s*[:=]\s*['\"]?([A-Za-z0-9_-]{3,})", call.result)
        if m:
            entities.append(CreatedEntity(server=call.server, tool=call.name, entity_id=m.group(2), name=name, raw=call.result[:200]))
    return entities


async def run_agent(case: EvalCase, cfg: RunConfig, provider: LLMProvider, registry: ToolRegistry) -> AgentTrace:
    trace = AgentTrace()
    tools = registry.llm_tools()
    history: list[dict[str, Any]] = [{"role": "user", "content": case.input}]
    system = system_prompt_for(case, cfg)
    max_turns = case.max_turns if case.max_turns is not None else cfg.max_turns
    order = 0
    usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    started = time.perf_counter()
    if hasattr(provider, "load_case"):
        provider.load_case(case, cfg.cleanup.entity_prefix)  # mock provider
    for turn in range(max_turns):
        trace.turns = turn + 1
        try:
            result = await provider.complete(system, history, tools, max_tokens=16000)
        except ProviderError as exc:
            trace.error = str(exc)
            trace.trajectory.append(f"provider error: {exc}")
            break
        for k in usage:
            usage[k] += int(result.usage.get(k, 0) or 0)
        trace.stop_reason = result.stop_reason
        if result.thinking:
            trace.trajectory.append(f"thinking: {result.thinking[:500]}")
        if result.text:
            trace.trajectory.append(f"assistant: {result.text[:1000]}")
        history.append({"role": "assistant", "content": result.text, "tool_calls": result.tool_calls, "raw_blocks": result.raw_blocks})
        if not result.tool_calls:
            trace.output = result.text
            break
        for tc in result.tool_calls:
            order += 1
            if tc.raw_args is not None:
                call = ActualToolCall(name=tc.name, args={}, result=f"ERROR: tool arguments were not valid JSON: {tc.raw_args[:200]}", is_error=True, order=order, exists=tc.name in registry.tools)
                trace.trajectory.append(f"call {tc.name} with INVALID JSON args")
            else:
                call, _ = await registry.execute(tc.name, tc.args, order)
                trace.trajectory.append(f"call {tc.name}({json.dumps(tc.args, default=str)[:400]}) -> {'ERROR ' if call.is_error else ''}{(call.result or '')[:300]}")
            trace.calls.append(call)
            history.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": (call.result or "")[:12000], "is_error": call.is_error})
    else:
        trace.trajectory.append(f"stopped: reached max turns ({max_turns})")
        trace.output = next((m["content"] for m in reversed(history) if m["role"] == "assistant" and m.get("content")), "")
    trace.usage = usage
    trace.usage["duration_ms"] = round((time.perf_counter() - started) * 1000)
    return trace
