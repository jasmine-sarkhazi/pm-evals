"""Harness adapters for running evals outside our own API loop.

* ``transcript``  - the PM runs the prompt in Claude.ai chat, ChatGPT, Claude
  Code, Cursor, Copilot... and drops the exported conversation in. We parse
  tool calls + final answer out of the export and score them with the same
  checks.
* ``claude-code`` - shells out to the ``claude`` CLI in print mode with the
  case's MCP servers wired in, and parses the JSON transcript.

Supported transcript formats (auto-detected):

1. pm-evals generic: ``{"actual_output": str, "actual_tool_calls": [{"name", "args", "result", "is_error"}]}``
2. Claude.ai data export ``conversations.json`` (one conversation or the list;
   ``chat_messages[].content[]`` with ``tool_use`` / ``tool_result`` blocks)
3. ChatGPT data export ``conversations.json`` (``mapping`` tree with
   ``author.role`` = assistant/tool and ``recipient`` = connector tool name)
4. Anthropic Messages API response / message list (``content[]`` blocks)
5. OpenAI chat completion message list (``tool_calls`` / ``role: tool``)
6. Claude Code ``--output-format json`` / ``stream-json`` output
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from typing import Any, Optional

from ..models import ActualToolCall, EvalCase, MCPServerConfig, RunConfig


class ParsedTranscript(dict):
    """{"actual_output": str, "actual_tool_calls": list[ActualToolCall], "format": str}"""


def _mk(name: str, args: Any, result: Any = None, is_error: bool = False, order: int = 0) -> ActualToolCall:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {"_value": args}
    if isinstance(result, (dict, list)):
        result_text = json.dumps(result, ensure_ascii=False)
    else:
        result_text = None if result is None else str(result)
    return ActualToolCall(name=name.split(".")[-1] if "." in name and " " not in name else name, args=args, result=result_text, is_error=is_error, order=order)


def _blocks_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif isinstance(b, str):
            parts.append(b)
    return "\n".join(parts).strip()


def _from_anthropic_messages(messages: list[dict[str, Any]]) -> ParsedTranscript:
    calls: list[ActualToolCall] = []
    pending: dict[str, ActualToolCall] = {}
    output = ""
    order = 0
    for m in messages:
        role = m.get("role") or m.get("sender")
        content = m.get("content")
        if isinstance(content, str):
            if role == "assistant":
                output = content
            continue
        for b in content or []:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "tool_use":
                order += 1
                c = _mk(b.get("name", ""), b.get("input", {}), order=order)
                pending[b.get("id", str(order))] = c
                calls.append(c)
            elif t == "tool_result":
                c = pending.get(b.get("tool_use_id"))
                res = b.get("content")
                if c is not None:
                    c.result = _blocks_text(res) if isinstance(res, list) else (json.dumps(res) if isinstance(res, dict) else str(res or ""))
                    c.is_error = bool(b.get("is_error"))
            elif t == "text" and role == "assistant":
                output = b.get("text", "") or output
    return ParsedTranscript(actual_output=output, actual_tool_calls=calls, format="anthropic_messages")


def _from_claude_ai_export(conv: dict[str, Any]) -> ParsedTranscript:
    msgs = []
    for m in conv.get("chat_messages", []):
        role = "assistant" if m.get("sender") == "assistant" else "user"
        content = m.get("content") or [{"type": "text", "text": m.get("text", "")}]
        msgs.append({"role": role, "content": content})
    parsed = _from_anthropic_messages(msgs)
    parsed["format"] = "claude_ai_export"
    return parsed


def _from_openai_messages(messages: list[dict[str, Any]]) -> ParsedTranscript:
    calls: list[ActualToolCall] = []
    pending: dict[str, ActualToolCall] = {}
    output = ""
    order = 0
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                order += 1
                c = _mk(fn.get("name", ""), fn.get("arguments", "{}"), order=order)
                pending[tc.get("id", str(order))] = c
                calls.append(c)
            if m.get("content"):
                output = m["content"] if isinstance(m["content"], str) else _blocks_text(m["content"])
        elif role == "tool":
            c = pending.get(m.get("tool_call_id"))
            if c is not None:
                c.result = m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
                c.is_error = "error" in (c.result or "").lower()[:40]
    return ParsedTranscript(actual_output=output, actual_tool_calls=calls, format="openai_messages")


def _from_chatgpt_export(conv: dict[str, Any]) -> ParsedTranscript:
    mapping = conv.get("mapping", {})
    # order nodes by create_time, falling back to parent chain
    nodes = [n for n in mapping.values() if n.get("message")]
    nodes.sort(key=lambda n: (n["message"].get("create_time") or 0))
    calls: list[ActualToolCall] = []
    output = ""
    order = 0
    last_call: Optional[ActualToolCall] = None
    for n in nodes:
        msg = n["message"]
        role = (msg.get("author") or {}).get("role")
        content = msg.get("content") or {}
        ctype = content.get("content_type")
        recipient = msg.get("recipient") or "all"
        if role == "assistant" and recipient != "all":
            order += 1
            text = content.get("text") or "".join(p for p in content.get("parts", []) if isinstance(p, str))
            last_call = _mk(recipient, text or "{}", order=order)
            calls.append(last_call)
        elif role == "tool" and last_call is not None and last_call.result is None:
            text = content.get("text") or "".join(p for p in content.get("parts", []) if isinstance(p, str))
            last_call.result = text
            last_call.is_error = "error" in (text or "").lower()[:60]
        elif role == "assistant" and ctype == "text":
            parts = [p for p in content.get("parts", []) if isinstance(p, str)]
            if parts and parts[-1].strip():
                output = parts[-1]
    return ParsedTranscript(actual_output=output, actual_tool_calls=calls, format="chatgpt_export")


def _from_claude_code(data: Any) -> ParsedTranscript:
    events = data if isinstance(data, list) else [data]
    msgs: list[dict[str, Any]] = []
    result_text = ""
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("type") in ("assistant", "user") and isinstance(ev.get("message"), dict):
            msgs.append(ev["message"])
        elif ev.get("type") == "result":
            result_text = ev.get("result") or result_text
    parsed = _from_anthropic_messages(msgs)
    if result_text:
        parsed["actual_output"] = result_text
    parsed["format"] = "claude_code"
    return parsed


def parse_transcript(data: Any, case_input: Optional[str] = None) -> ParsedTranscript:
    """Auto-detect the transcript format and normalise it."""
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict):
        if "actual_tool_calls" in data or "actual_output" in data:
            calls = []
            for i, c in enumerate(data.get("actual_tool_calls") or [], 1):
                if isinstance(c, dict):
                    calls.append(_mk(c.get("name", ""), c.get("args") or c.get("arguments") or {}, c.get("result"), bool(c.get("is_error")), c.get("order") or i))
            return ParsedTranscript(actual_output=data.get("actual_output") or "", actual_tool_calls=calls, format="generic")
        if "chat_messages" in data:
            return _from_claude_ai_export(data)
        if "mapping" in data:
            return _from_chatgpt_export(data)
        if "messages" in data and isinstance(data["messages"], list):
            return parse_transcript(data["messages"], case_input)
        if data.get("type") in ("message",) and "content" in data:
            return _from_anthropic_messages([data])
        if "choices" in data:
            msg = (data["choices"][0] or {}).get("message", {})
            return _from_openai_messages([{"role": "assistant", **msg}])
    if isinstance(data, list):
        if not data:
            return ParsedTranscript(actual_output="", actual_tool_calls=[], format="empty")
        first = data[0]
        if isinstance(first, dict) and ("chat_messages" in first or "mapping" in first):
            # A full export with many conversations: pick the one whose first
            # human message matches the case input, else the first.
            chosen = first
            if case_input:
                for conv in data:
                    text = json.dumps(conv)[:20000]
                    if case_input[:60] in text:
                        chosen = conv
                        break
            return parse_transcript(chosen, case_input)
        if isinstance(first, dict) and first.get("type") in ("system", "assistant", "user", "result"):
            return _from_claude_code(data)
        if isinstance(first, dict) and "role" in first:
            if any(isinstance(m.get("content"), list) and any(isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result") for b in m["content"]) for m in data if isinstance(m, dict)):
                return _from_anthropic_messages(data)
            return _from_openai_messages(data)
    raise ValueError("Unrecognised transcript format. Use the generic pm-evals format, a Claude.ai / ChatGPT export, or an API message list.")


# ---------------------------------------------------------------------------
# Harness pack: what to paste into a chat harness
# ---------------------------------------------------------------------------


def harness_pack(cases: list[EvalCase], cfg: RunConfig) -> str:
    lines = [
        "# pm-evals harness pack",
        "",
        "Run each prompt below in the harness you are testing (Claude.ai, ChatGPT, Claude Code, Cursor ...) with the MCP server connected.",
        "Export the conversation afterwards (Claude.ai: Settings > Privacy > Export data; ChatGPT: Settings > Data controls > Export) "
        "or paste the tool calls into the generic JSON form in the Transcripts tab.",
        "",
        f"Entity prefix to use for anything you create: `{cfg.cleanup.entity_prefix}`",
        "",
    ]
    for c in cases:
        lines += [f"## {c.id}", "", "```", c.input, "```", ""]
        if cfg.cleanup.enabled:
            lines += [f"(Append if the harness lets you set a system prompt: *Name every created entity with the prefix {cfg.cleanup.entity_prefix}.*)", ""]
    lines += [
        "## Generic transcript format (one JSON file per case)",
        "",
        "```json",
        json.dumps({"case_id": "<case id>", "actual_output": "<final assistant answer>", "actual_tool_calls": [{"name": "create_segment", "args": {"name": "EVAL_x"}, "result": "{...}", "is_error": False}]}, indent=2),
        "```",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude Code harness
# ---------------------------------------------------------------------------


def _mcp_config(servers: list[MCPServerConfig]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in servers:
        if s.transport == "stdio":
            out[s.server_name] = {"type": "stdio", "command": s.command, "args": s.args, "env": s.env}
        else:
            entry: dict[str, Any] = {"type": "http" if s.transport == "streamable-http" else "sse", "url": s.url}
            if s.headers:
                entry["headers"] = s.headers
            out[s.server_name] = entry
    return {"mcpServers": out}


async def run_claude_code(case: EvalCase, cfg: RunConfig, servers: list[MCPServerConfig], system_prompt: str, timeout: float = 600) -> ParsedTranscript:
    binary = shutil.which("claude")
    if not binary:
        raise RuntimeError("The 'claude' CLI is not installed or not on PATH. Install Claude Code to use this harness.")
    with tempfile.TemporaryDirectory() as tmp:
        mcp_path = os.path.join(tmp, "mcp.json")
        with open(mcp_path, "w", encoding="utf-8") as fh:
            json.dump(_mcp_config(servers), fh)
        allowed = ",".join(f"mcp__{s.server_name}__*" for s in servers) or "mcp__*"
        cmd = [
            binary, "-p", case.input,
            "--output-format", "json",
            "--mcp-config", mcp_path,
            "--strict-mcp-config",
            "--allowedTools", allowed,
            "--append-system-prompt", system_prompt,
            "--max-turns", str(case.max_turns or cfg.max_turns),
        ]
        if cfg.model and not cfg.model.startswith("mock"):
            cmd += ["--model", cfg.model]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=tmp)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("claude CLI timed out")
        text = out.decode("utf-8", "replace").strip()
        if proc.returncode != 0 and not text:
            raise RuntimeError(f"claude CLI failed: {err.decode('utf-8', 'replace')[:500]}")
        try:
            data = json.loads(text)
        except Exception:
            data = [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]
        parsed = _from_claude_code(data)
        # Tool names arrive as mcp__<server>__<tool>; normalise.
        for c in parsed["actual_tool_calls"]:
            if c.name.startswith("mcp__"):
                parts = c.name.split("__", 2)
                c.server = parts[1] if len(parts) > 2 else None
                c.name = parts[-1]
        return parsed
