"""Regression tests for issues found in code review."""

import pytest

from pm_evals.checks.base import CheckContext, load_all
from pm_evals.checks.deterministic import call_order, tool_correctness
from pm_evals.checks.safety import sandbox_escape
from pm_evals.models import ActualToolCall, CleanupConfig, EvalCase, ExpectedToolCall, RunConfig
from pm_evals.providers.openai_provider import _sanitize_schema
from pm_evals.runner.agent import extract_created_entities
from pm_evals.storage import Workspace, slug

load_all()


def test_slug_never_escapes_workspace(tmp_path):
    for bad in (".", "..", "...", "../x", "", "_", "-."):
        assert slug(bad) not in (".", "..", "") and "/" not in slug(bad)
    ws = Workspace(tmp_path / "ws")
    (ws.root / "reports" / "keep.json").write_text("{}")
    assert ws.dataset_dir("..").parent == (ws.root / "datasets").resolve()
    assert ws.delete_dataset("..") is False
    assert (ws.root / "reports" / "keep.json").exists()


def test_sanitize_schema_keeps_property_named_title():
    schema = {"type": "object", "title": "Args", "properties": {"title": {"type": "string", "title": "Title"}, "body": {"type": "string"}}, "required": ["title", "body"]}
    out = _sanitize_schema(schema)
    assert "title" not in out and set(out["properties"]) == {"title", "body"}
    assert "title" not in out["properties"]["title"] and out["properties"]["title"]["type"] == "string"


async def test_calls_with_zero_order_are_renumbered_by_runner(tmp_path):
    from pm_evals.mcpio.demo_server import build_server
    from pm_evals.runner.orchestrator import Runner
    from tests.conftest import SERVER

    ws = Workspace(tmp_path / "ws")
    ws.save_server(SERVER)
    case = EvalCase(
        id="embedded", input="create a segment then a campaign", mcp_servers=[SERVER.model_copy(update={"available_tools": None})],
        expected_tool_calls=[ExpectedToolCall(name="create_segment", order=1), ExpectedToolCall(name="create_campaign", order=2)],
        actual_tool_calls=[ActualToolCall(name="create_segment", args={"name": "EVAL_a", "filter": {}}, result="{\"id\": \"seg_1\"}"), ActualToolCall(name="create_campaign", args={"name": "EVAL_b", "segment_id": "seg_1"}, result="{\"id\": \"cmp_1\"}")],
        actual_output="Done.", metrics=["tool_correctness", "call_order"],
    )
    ws.save_cases("emb", [case])
    cfg = RunConfig(dataset="emb", harness="transcript", judge_model="mock")
    cfg.cleanup.enabled = False
    report = await Runner(ws, in_process={"AJO-MCP": build_server()}).run(cfg)
    m = {x.metric: x for x in report.results[0].metrics}
    assert m["tool_correctness"].score == 1.0 and m["call_order"].score == 1.0
    assert [c.order for c in report.results[0].actual_tool_calls] == [1, 2]


def test_blank_cleanup_prefixes_are_rejected():
    with pytest.raises(ValueError):
        CleanupConfig(entity_prefix="")
    with pytest.raises(ValueError):
        CleanupConfig(delete_tool_prefix="   ")


def test_entity_extraction_prefers_own_id_over_foreign_key():
    cfg = CleanupConfig()
    call = ActualToolCall(name="create_campaign", args={"name": "EVAL_c"}, structured_result={"segment_id": "seg_0001", "id": "cmp_0002", "name": "EVAL_c"}, order=1)
    ents = extract_created_entities(call, cfg)
    assert [e.entity_id for e in ents] == ["cmp_0002"]
    call2 = ActualToolCall(name="create_segment", args={"name": "EVAL_s"}, structured_result={"segment_id": "seg_9", "tenant_id": "t"}, order=1)
    assert [e.entity_id for e in extract_created_entities(call2, cfg)] == ["seg_9"]


async def test_destructive_hint_snake_case_is_detected():
    case = EvalCase(id="d", input="x", expected_tool_calls=[ExpectedToolCall(name="list_segments")])
    defs = {"purge_all": {"description": "", "input_schema": {"type": "object", "properties": {}}, "distractor": False, "allowed": True, "annotations": {"destructive_hint": True}}}
    ctx = CheckContext(case=case, config=RunConfig(dataset="t"), calls=[ActualToolCall(name="purge_all", order=1)], output="", trajectory=[], tool_defs=defs)
    r = await sandbox_escape(ctx)
    assert r.score < 1.0 and any(v["type"] == "unexpected_destructive_tool" for v in r.details["violations"])
