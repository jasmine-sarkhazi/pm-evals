"""Pydantic data models shared across the platform.

The JSON shape of :class:`EvalCase` follows the sample format PMs author
(see README "Case format"). Everything else is internal but serialised into
reports so that reports are self-describing.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Category = Literal["deterministic", "hallucination", "safety", "llm_judge"]
Transport = Literal["streamable-http", "sse", "stdio"]


# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------


class MCPServerConfig(BaseModel):
    """How to reach one MCP server."""

    server_name: str
    transport: Transport = "streamable-http"
    url: Optional[str] = None
    headers: dict[str, str] = Field(default_factory=dict)
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    available_tools: Optional[list[str]] = None
    """Allow-list of tools the agent may use. ``None`` means every tool the
    server exposes. Calling a tool outside this list is a permission escape."""

    @field_validator("server_name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("server_name must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------


class ExpectedToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    required: bool = True
    order: Optional[int] = None
    arg_match: Literal["subset", "exact"] = "subset"
    """``subset``: every expected key must match, extra keys are allowed.
    ``exact``: actual args must contain no keys beyond the expected ones."""


class RubricItem(BaseModel):
    criterion: str
    weight: float = 1.0
    description: Optional[str] = None
    anchors: dict[str, str] = Field(default_factory=dict)
    """Calibration anchors keyed by score, e.g. ``{"1": "...", "5": "..."}``."""


class StateCheck(BaseModel):
    """Verify post-run state by calling a read tool and inspecting the result."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    path: Optional[str] = None
    """Dotted path into the structured/JSON result, e.g. ``items.0.name``.
    ``None`` inspects the whole result text."""
    op: Literal["eq", "contains", "not_contains", "exists", "gte", "lte", "len_gte"] = "contains"
    value: Any = None
    description: Optional[str] = None


class TenantPolicy(BaseModel):
    id_fields: list[str] = Field(default_factory=lambda: ["tenant_id", "org_id", "account_id", "workspace_id", "sandbox"])
    allowed_values: list[str] = Field(default_factory=list)
    forbidden_values: list[str] = Field(default_factory=list)


class DistractorTool(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ActualToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Optional[str] = None
    structured_result: Any = None
    is_error: bool = False
    order: int = 0
    server: Optional[str] = None
    duration_ms: Optional[float] = None
    exists: bool = True
    """False when the model invented a tool that no server exposes."""
    distractor: bool = False


class EvalCase(BaseModel):
    id: str
    category: Category = "deterministic"
    description: str = ""
    input: str
    expected_output: Optional[str] = None
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    expected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    actual_tool_calls: list[ActualToolCall] = Field(default_factory=list)
    actual_output: Optional[str] = None
    trajectory: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    threshold: float = 0.8
    tags: list[str] = Field(default_factory=list)
    rubric: list[RubricItem] = Field(default_factory=list)
    judge_model: Optional[str] = None
    output_format: dict[str, Any] = Field(default_factory=lambda: {"score": "1-5", "reason": "string"})
    system_prompt: Optional[str] = None
    allowed_extra_tools: list[str] = Field(default_factory=list)
    """Tools the agent may call in addition to the expected ones without
    being penalised (read-only helpers are auto-allowed, see checks)."""
    forbidden_tools: list[str] = Field(default_factory=list)
    state_checks: list[StateCheck] = Field(default_factory=list)
    tenant: Optional[TenantPolicy] = None
    distractor_tools: list[DistractorTool] = Field(default_factory=list)
    calibration_examples: list[dict[str, Any]] = Field(default_factory=list)
    """Few-shot anchors for the judge: ``{"trajectory": ..., "score": 4, "reason": ...}``."""
    max_turns: Optional[int] = None
    """Overrides the run-level max_turns when set."""

    @field_validator("threshold")
    @classmethod
    def _threshold_range(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("threshold must be between 0 and 1")
        return v


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


class DistractorConfig(BaseModel):
    mode: Literal["none", "manual", "auto", "both"] = "none"
    tools: list[DistractorTool] = Field(default_factory=list)
    """Manually supplied distractor tools (names + optional descriptions)."""
    count: int = 20
    """How many auto-generated distractors to add."""
    near_miss: bool = True
    """Also add look-alike variants of the real tools (hardest distractors)."""
    seed: int = 7


class CleanupConfig(BaseModel):
    enabled: bool = True
    entity_prefix: str = "EVAL_"
    """Every entity the agent creates must carry this prefix in its name."""
    delete_tool_prefix: str = "eval_delete"
    """Only delete tools whose name starts with this prefix are ever called."""
    id_fields: list[str] = Field(default_factory=lambda: ["id", "segment_id", "entity_id", "uuid", "_id"])
    dry_run: bool = False


class RunConfig(BaseModel):
    dataset: str
    model: str = "claude-opus-5"
    """Model under test. Prefix decides the provider (claude-, gpt-, gemini-, mock ...)."""
    judge_model: str = "claude-opus-5"
    harness: Literal["api", "transcript", "claude-code", "dry-run"] = "api"
    servers: list[MCPServerConfig] = Field(default_factory=list)
    """Overrides the servers on every case when non-empty."""
    distractors: DistractorConfig = Field(default_factory=DistractorConfig)
    cleanup: CleanupConfig = Field(default_factory=CleanupConfig)
    case_ids: list[str] = Field(default_factory=list)
    transcripts_dir: Optional[str] = None
    max_turns: int = 12
    concurrency: int = 1
    label: Optional[str] = None
    compare_to: Optional[str] = None
    """Report id (or path) of a previous report to compare against."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class MetricResult(BaseModel):
    metric: str
    category: str
    score: float
    passed: bool
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    skipped: bool = False


class CreatedEntity(BaseModel):
    server: Optional[str]
    tool: str
    entity_id: Optional[str] = None
    name: Optional[str] = None
    raw: Any = None
    deleted: Optional[bool] = None
    delete_tool: Optional[str] = None
    cleanup_note: str = ""


class CaseResult(BaseModel):
    case_id: str
    category: str
    description: str = ""
    input: str = ""
    score: float = 0.0
    passed: bool = False
    threshold: float = 0.8
    metrics: list[MetricResult] = Field(default_factory=list)
    actual_tool_calls: list[ActualToolCall] = Field(default_factory=list)
    actual_output: Optional[str] = None
    trajectory: list[str] = Field(default_factory=list)
    created_entities: list[CreatedEntity] = Field(default_factory=list)
    error: Optional[str] = None
    duration_ms: float = 0.0
    tags: list[str] = Field(default_factory=list)
    distractors_presented: int = 0
    usage: dict[str, Any] = Field(default_factory=dict)


class ServerFinding(BaseModel):
    server_name: str
    metric: str
    score: float
    passed: bool
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class Report(BaseModel):
    id: str
    created_at: dt.datetime
    date: str
    dataset: str
    label: Optional[str] = None
    config: RunConfig
    summary: dict[str, Any] = Field(default_factory=dict)
    results: list[CaseResult] = Field(default_factory=list)
    server_findings: list[ServerFinding] = Field(default_factory=list)
    cleanup: dict[str, Any] = Field(default_factory=dict)
    comparison: Optional[dict[str, Any]] = None
    tool_inventory: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
