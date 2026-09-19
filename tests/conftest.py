import pytest

from pm_evals.mcpio.demo_server import build_server
from pm_evals.models import ActualToolCall, EvalCase, ExpectedToolCall, MCPServerConfig, RunConfig
from pm_evals.storage import Workspace

SERVER = MCPServerConfig(server_name="AJO-MCP", transport="stdio", command="python", args=["-m", "pm_evals.mcpio.demo_server"])


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws")


@pytest.fixture
def demo():
    return {"AJO-MCP": build_server()}


@pytest.fixture
def case_create():
    return EvalCase(
        id="ajo_create_segment_001",
        category="deterministic",
        description="Agent should call create_segment with correctly derived filter criteria",
        input="Create a segment of users who opened an email in the last 7 days",
        mcp_servers=[MCPServerConfig(server_name="AJO-MCP", transport="stdio", available_tools=["create_segment", "list_segments", "get_schema", "eval_delete_segment"])],
        expected_tool_calls=[ExpectedToolCall(name="create_segment", args={"filter": {"event": "email_open", "window_days": 7}}, required=True, order=1)],
        metrics=["tool_correctness", "argument_correctness", "hallucination_check"],
        threshold=0.8,
        tags=["ajo", "segmentation"],
    )


def call(name, args=None, result="{}", is_error=False, order=1, exists=True, distractor=False):
    return ActualToolCall(name=name, args=args or {}, result=result, is_error=is_error, order=order, exists=exists, distractor=distractor)


@pytest.fixture
def mk_call():
    return call


@pytest.fixture
def run_cfg():
    return RunConfig(dataset="x")
