import json

from pm_evals.providers.anthropic_provider import _to_anthropic_messages
from pm_evals.providers.base import ToolCallRequest, parse_json_object
from pm_evals.providers.openai_provider import _to_openai_messages
from pm_evals.providers.registry import provider_for


def test_provider_routing():
    assert provider_for("claude-opus-5") == "anthropic"
    assert provider_for("gpt-5") == "openai"
    assert provider_for("o3-mini") == "openai"
    assert provider_for("gemini-2.5-pro") == "gemini"
    assert provider_for("ollama:llama3.1") == "ollama"
    assert provider_for("groq:llama-3.3-70b") == "groq"
    assert provider_for("openai:gpt-4.1") == "openai"
    assert provider_for("mock:phantom") == "mock"


def test_anthropic_history_replays_raw_blocks_with_thinking():
    raw = [
        {"type": "thinking", "thinking": "", "signature": "sig123"},
        {"type": "text", "text": "Creating it."},
        {"type": "tool_use", "id": "tu1", "name": "create_segment", "input": {"name": "EVAL_x"}},
    ]
    history = [
        {"role": "user", "content": "Create a segment"},
        {"role": "assistant", "content": "Creating it.", "tool_calls": [ToolCallRequest(id="tu1", name="create_segment", args={"name": "EVAL_x"})], "raw_blocks": raw},
        {"role": "tool", "tool_call_id": "tu1", "name": "create_segment", "content": "{\"id\": 1}", "is_error": False},
        {"role": "tool", "tool_call_id": "tu2", "name": "other", "content": "boom", "is_error": True},
    ]
    msgs = _to_anthropic_messages(history)
    assert msgs[1] == {"role": "assistant", "content": raw}  # verbatim, thinking block kept
    assert msgs[2]["role"] == "user"
    results = msgs[2]["content"]
    assert [r["type"] for r in results] == ["tool_result", "tool_result"]  # grouped in one user message
    assert results[1]["is_error"] is True and results[0]["tool_use_id"] == "tu1"


def test_anthropic_history_without_raw_blocks_is_rebuilt():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [ToolCallRequest(id="a", name="get_schema", args={})]},
        {"role": "tool", "tool_call_id": "a", "name": "get_schema", "content": "{}", "is_error": False},
        {"role": "assistant", "content": "", "tool_calls": []},
    ]
    msgs = _to_anthropic_messages(history)
    assert msgs[1]["content"] == [{"type": "tool_use", "id": "a", "name": "get_schema", "input": {}}]
    assert msgs[3]["content"] == [{"type": "text", "text": "(no output)"}]


def test_openai_history_conversion_keeps_raw_args_and_tool_role():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [ToolCallRequest(id="c1", name="list_segments", args={}, raw_args="{not json")]},
        {"role": "tool", "tool_call_id": "c1", "name": "list_segments", "content": "ERROR", "is_error": True},
    ]
    msgs = _to_openai_messages("sys", history)
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == "{not json"
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "ERROR"}


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here it is: {"score": 4, "reason": "ok"} hope that helps') == {"score": 4, "reason": "ok"}
