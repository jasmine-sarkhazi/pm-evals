import json

from pm_evals.importers.sheet import google_sheet_csv_url, parse_csv_bytes, rows_to_cases, template_csv
from pm_evals.runner.harness import parse_transcript


def test_template_round_trip():
    rows = parse_csv_bytes(template_csv().encode())
    cases, warnings = rows_to_cases(rows)
    assert not warnings
    ids = [c.id for c in cases]
    assert ids == ["ajo_create_segment_001", "ajo_lookup_schema_002", "safety_tenant_003", "judge_task_completion_004"]
    multi = cases[1]
    assert [e.name for e in multi.expected_tool_calls] == ["get_schema", "create_segment"]
    assert multi.expected_tool_calls[1].order == 2
    assert cases[0].state_checks[0].op == "len_gte"
    assert cases[2].tenant.forbidden_values == ["tenant_b"]
    judge = cases[3]
    assert judge.category == "llm_judge" and [r.criterion for r in judge.rubric] == ["tool_selection_appropriate", "no_redundant_calls", "final_state_matches_intent"]
    assert abs(sum(r.weight for r in judge.rubric) - 1.0) < 1e-9


def test_loose_column_names():
    rows = [{"Test ID": "t1", "Prompt": "do x", "Tool": "create_segment", "Arguments": '{"a": 1}', "Checks": "tool_correctness", "Type": "Hallucination"}]
    cases, _ = rows_to_cases(rows)
    assert cases[0].category == "hallucination" and cases[0].expected_tool_calls[0].args == {"a": 1} and cases[0].metrics == ["tool_correctness"]


def test_google_sheet_url():
    assert google_sheet_csv_url("https://docs.google.com/spreadsheets/d/abc123/edit#gid=42") == "https://docs.google.com/spreadsheets/d/abc123/export?format=csv&gid=42"


def test_parse_generic_transcript():
    p = parse_transcript({"actual_output": "done", "actual_tool_calls": [{"name": "create_segment", "args": {"name": "EVAL_x"}, "result": "{\"id\": 1}"}]})
    assert p["format"] == "generic" and p["actual_tool_calls"][0].name == "create_segment" and p["actual_output"] == "done"


def test_parse_anthropic_messages():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "create_segment", "input": {"name": "EVAL_a"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Created it."}]},
    ]
    p = parse_transcript(msgs)
    assert p["format"] == "anthropic_messages" and p["actual_tool_calls"][0].result == "ok" and p["actual_output"] == "Created it."


def test_parse_openai_messages():
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "list_segments", "arguments": "{\"name_prefix\": \"EVAL_\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "[]"},
        {"role": "assistant", "content": "No segments."},
    ]
    p = parse_transcript(msgs)
    assert p["format"] == "openai_messages" and p["actual_tool_calls"][0].args == {"name_prefix": "EVAL_"} and p["actual_output"] == "No segments."


def test_parse_chatgpt_export():
    conv = {"title": "x", "mapping": {
        "a": {"message": {"author": {"role": "user"}, "create_time": 1, "content": {"content_type": "text", "parts": ["Create a segment"]}, "recipient": "all"}},
        "b": {"message": {"author": {"role": "assistant"}, "create_time": 2, "content": {"content_type": "code", "text": "{\"name\": \"EVAL_x\"}"}, "recipient": "ajo.create_segment"}},
        "c": {"message": {"author": {"role": "tool"}, "create_time": 3, "content": {"content_type": "text", "parts": ["{\"id\": \"seg_1\"}"]}, "recipient": "all"}},
        "d": {"message": {"author": {"role": "assistant"}, "create_time": 4, "content": {"content_type": "text", "parts": ["Done, created seg_1."]}, "recipient": "all"}},
    }}
    p = parse_transcript([conv], "Create a segment")
    assert p["format"] == "chatgpt_export"
    assert p["actual_tool_calls"][0].name == "create_segment" and p["actual_tool_calls"][0].result == "{\"id\": \"seg_1\"}"
    assert p["actual_output"] == "Done, created seg_1."


def test_parse_claude_ai_export():
    conv = {"chat_messages": [
        {"sender": "human", "content": [{"type": "text", "text": "Create a segment"}]},
        {"sender": "assistant", "content": [{"type": "tool_use", "id": "u1", "name": "create_segment", "input": {"name": "EVAL_y"}}, {"type": "tool_result", "tool_use_id": "u1", "content": [{"type": "text", "text": "{\"id\": 2}"}]}, {"type": "text", "text": "All set."}]},
    ]}
    p = parse_transcript(conv)
    assert p["format"] == "claude_ai_export" and p["actual_tool_calls"][0].result == "{\"id\": 2}" and p["actual_output"] == "All set."
