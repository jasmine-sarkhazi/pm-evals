"""Tests for the TypeSafe System One (Jev) judge, run fully offline with the
mock System One client — no SDK, key or network required."""

import pytest

from pm_evals.checks.base import CheckContext, load_all
from pm_evals.checks.judge import task_completion, trajectory_quality
from pm_evals.models import EvalCase, ExpectedToolCall, RubricItem, RunConfig
from pm_evals.providers.registry import is_system_one_model, make_judge
from pm_evals.providers.typesafe_provider import (
    MockSystemOneClient,
    Noul,
    Score,
    TypeSafeJevJudge,
    position_1_based,
    position_to_1_5,
)

load_all()

TOOL_DEFS = {
    "create_segment": {"description": "create", "input_schema": {"type": "object"}, "distractor": False, "allowed": True, "annotations": {}},
    "create_audience": {"description": "distractor", "input_schema": {"type": "object"}, "distractor": True, "allowed": True, "annotations": {}},
}


def _case():
    return EvalCase(
        id="jev_case",
        category="llm_judge",
        input="Create a segment of users who opened an email in the last 7 days.",
        description="Agent should create the segment with the right filter.",
        expected_tool_calls=[ExpectedToolCall(name="create_segment", args={"filter": {"event": "email_open"}})],
        rubric=[
            RubricItem(criterion="final_state_matches_intent", weight=0.6, anchors={"1": "nothing done", "5": "exactly right"}),
            RubricItem(criterion="tool_selection_appropriate", weight=0.4, anchors={"1": "wrong tool", "5": "best tool"}),
        ],
        threshold=0.7,
    )


def _ctx(judge, calls=None, output="Done."):
    from tests.conftest import call

    calls = calls if calls is not None else [call("create_segment", {"filter": {"event": "email_open"}}, result='{"id": "seg_1"}')]
    return CheckContext(case=_case(), config=RunConfig(dataset="t", judge_model="jev:mock"), calls=calls, output=output, trajectory=[], tool_defs=TOOL_DEFS, judge=judge)


# -- routing ----------------------------------------------------------------


def test_is_system_one_model():
    for m in ("jev", "JEV", "typesafe", "system-one", "jev:mock", "typesafe:prod"):
        assert is_system_one_model(m)
    for m in ("claude-opus-5", "gpt-5", "mock", "gemini-2.5-pro", "jevons"):
        assert not is_system_one_model(m)


def test_make_judge_routes_to_system_one():
    j = make_judge("jev:mock")
    assert isinstance(j, TypeSafeJevJudge) and j.kind == "system_one"
    assert not isinstance(make_judge("mock"), TypeSafeJevJudge)


# -- score normalisation ----------------------------------------------------


def test_position_from_probabilities_is_weighted_mean():
    class S:
        probabilities = {"4": 0.5, "5": 0.5}
        score = 99  # ignored when probabilities are present

    assert position_1_based(S(), 5) == pytest.approx(4.5)


def test_position_falls_back_to_score_and_handles_zero_based():
    class OneBased:
        probabilities = None
        score = 3.0

    class ZeroBased:
        probabilities = None
        score = 0.0  # 0-based bottom level -> level 1

    assert position_1_based(OneBased(), 5) == pytest.approx(3.0)
    assert position_1_based(ZeroBased(), 5) == pytest.approx(1.0)


def test_position_to_1_5_identity_for_five_levels():
    assert position_to_1_5(1.0, 5) == pytest.approx(1.0)
    assert position_to_1_5(5.0, 5) == pytest.approx(5.0)
    assert position_to_1_5(3.0, 5) == pytest.approx(3.0)


# -- grade() ----------------------------------------------------------------


async def test_grade_builds_typed_questions_and_parses():
    captured = {}

    class Recorder(MockSystemOneClient):
        async def system_one(self, state, questions):
            captured["state"] = state
            captured["questions"] = questions
            return await super().system_one(state, questions)

    judge = TypeSafeJevJudge(model="jev", client=Recorder(behaviour="good"))
    case = _case()
    data = await judge.grade({"task_given_to_the_agent": case.input}, case.rubric, "Did the tool perform correctly?")

    q = captured["questions"]
    # one Score per rubric criterion + one overall Noul
    assert sum(isinstance(v, Score) for v in q.values()) == 2
    assert isinstance(q["overall_correct"], Noul)
    assert all(len(v.criteria) == 5 for v in q.values() if isinstance(v, Score))

    assert len(data["criteria"]) == 2
    assert {c["criterion"] for c in data["criteria"]} == {"final_state_matches_intent", "tool_selection_appropriate"}
    assert all(1.0 <= c["score"] <= 5.0 for c in data["criteria"])
    assert data["correct"] is True and data["correct_probability"] > 0.5


async def test_task_completion_check_good_and_bad():
    good = await task_completion(_ctx(TypeSafeJevJudge(model="jev", client=MockSystemOneClient("good"))))
    assert good.category == "llm_judge" and good.details["judge_method"] == "system_one"
    assert good.details["correct"] is True and good.score >= 0.7 and good.passed
    assert good.details["criteria"] and good.details["judge_model"] == "jev"

    bad = await task_completion(_ctx(TypeSafeJevJudge(model="jev", client=MockSystemOneClient("bad"))))
    assert bad.details["correct"] is False and bad.score < 0.5 and not bad.passed


async def test_trajectory_quality_uses_system_one_path():
    r = await trajectory_quality(_ctx(TypeSafeJevJudge(model="jev", client=MockSystemOneClient("good"))))
    assert r.details["judge_method"] == "system_one" and 0.0 <= r.score <= 1.0


async def test_grade_failure_is_reported_not_raised():
    class Boom(MockSystemOneClient):
        async def system_one(self, state, questions):
            raise RuntimeError("api down")

    r = await task_completion(_ctx(TypeSafeJevJudge(model="jev", client=Boom())))
    assert r.score == 0.0 and "System One judge call failed" in r.reason


async def test_orchestrator_end_to_end_with_jev_mock_judge(ws, demo):
    from pm_evals.mcpio.client import MCPConnection  # noqa: F401
    from pm_evals.models import MCPServerConfig
    from pm_evals.runner.orchestrator import Runner
    from tests.conftest import SERVER

    ws.save_server(SERVER)
    case = EvalCase(
        id="jev_judge_e2e",
        category="llm_judge",
        input="Create a segment of users who opened an email in the last 7 days.",
        mcp_servers=[MCPServerConfig(server_name="AJO-MCP", transport="stdio")],
        expected_tool_calls=[ExpectedToolCall(name="create_segment", args={"filter": {"event": "email_open", "window_days": 7}})],
        rubric=[RubricItem(criterion="final_state_matches_intent", weight=1.0, anchors={"1": "nothing", "5": "exact"})],
        threshold=0.7,
    )
    ws.save_cases("jev", [case])
    cfg = RunConfig(dataset="jev", harness="api", model="mock", judge_model="jev:mock")
    report = await Runner(ws, in_process=demo).run(cfg)
    res = report.results[0]
    tc = {m.metric: m for m in res.metrics}["task_completion"]
    assert tc.details["judge_method"] == "system_one" and tc.details["judge_model"] == "jev:mock"
    assert tc.details["correct"] is True and res.passed
