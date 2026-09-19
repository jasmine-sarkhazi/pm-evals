"""LLM-as-judge checks with calibrated, task-specific rubrics.

Every judged metric returns a 1-5 score per rubric criterion plus a reason,
weighted into a 0-1 metric score. Calibration comes from three places:

1. rubric ``anchors`` describing what a 1 and a 5 look like for that criterion,
2. ``calibration_examples`` on the case (few-shot graded trajectories),
3. a fixed grading protocol in the system prompt (score the evidence, not the
   tone; missing evidence scores low; never reward claims without tool calls).
"""

from __future__ import annotations

import json
from typing import Any

from ..models import EvalCase, MetricResult, RubricItem
from .base import CheckContext, make_result, register

DEFAULT_RUBRICS: dict[str, list[RubricItem]] = {
    "task_completion": [
        RubricItem(criterion="final_state_matches_intent", weight=0.5, description="The end state produced by the tool calls is what the user asked for.", anchors={"1": "Nothing relevant was done, or the wrong thing was done.", "3": "Partially done or done with notable inaccuracies.", "5": "Exactly what was asked, verified by tool results."}),
        RubricItem(criterion="answer_is_truthful", weight=0.3, description="The final answer accurately reports what happened, including failures.", anchors={"1": "Claims success that tool results do not support.", "5": "Every claim is backed by a tool result."}),
        RubricItem(criterion="constraints_respected", weight=0.2, description="Explicit constraints in the request (filters, windows, names, limits) were honoured.", anchors={"1": "Key constraints ignored.", "5": "All constraints honoured."}),
    ],
    "trajectory_quality": [
        RubricItem(criterion="tool_selection_appropriate", weight=0.4, description="Chose the most fitting tools; did not reach for irrelevant or look-alike tools.", anchors={"1": "Wrong or irrelevant tools.", "5": "Best possible tool for every step."}),
        RubricItem(criterion="no_redundant_calls", weight=0.3, description="No repeated, unnecessary or exploratory calls that did not contribute.", anchors={"1": "Many wasted calls or loops.", "5": "Minimal, purposeful sequence."}),
        RubricItem(criterion="sensible_ordering_and_recovery", weight=0.3, description="Steps in a logical order; errors handled and recovered from sensibly.", anchors={"1": "Illogical order, ignored errors.", "5": "Logical order, graceful recovery."}),
    ],
}

_SYSTEM = """You are a strict, calibrated evaluator of AI agents that use tools (MCP servers).
You grade the EVIDENCE in the transcript, never the agent's tone or confidence.

Grading protocol:
- Score each rubric criterion from 1 (worst) to 5 (best) using the anchors when given.
- A claim in the final answer that is not backed by a successful tool call counts as false.
- Errors from tools that the agent ignored or misreported are severe.
- Do not reward verbosity. Do not penalise brevity.
- If the expected tool calls are provided, treat them as the golden path but accept
  equivalent alternatives that achieve the same end state.
- Be consistent: the same behaviour must always get the same score.
Return only the JSON object requested."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "reason": {"type": "string"},
                },
                "required": ["criterion", "score", "reason"],
                "additionalProperties": False,
            },
        },
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
    },
    "required": ["criteria", "score", "reason"],
    "additionalProperties": False,
}


def _format_trajectory(ctx: CheckContext) -> str:
    lines = []
    for c in ctx.calls:
        flag = " [NON-EXISTENT TOOL]" if not c.exists else (" [DISTRACTOR TOOL]" if c.distractor else "")
        status = "ERROR" if c.is_error else "ok"
        lines.append(f"{c.order}. call {c.name}{flag} args={json.dumps(c.args, default=str)[:600]}\n   -> {status}: {(c.result or '')[:600]}")
    if not lines:
        lines.append("(no tool calls)")
    return "\n".join(lines)


def _rubric_text(items: list[RubricItem]) -> str:
    out = []
    for r in items:
        line = f"- {r.criterion} (weight {r.weight})"
        if r.description:
            line += f": {r.description}"
        if r.anchors:
            line += " Anchors: " + "; ".join(f"{k} = {v}" for k, v in sorted(r.anchors.items()))
        out.append(line)
    return "\n".join(out)


def build_prompt(case: EvalCase, ctx: CheckContext, metric: str, rubric: list[RubricItem]) -> str:
    parts = [
        f"## Task given to the agent\n{case.input}",
    ]
    if case.description:
        parts.append(f"## What a correct run looks like (from the test author)\n{case.description}")
    if case.expected_output:
        parts.append(f"## Expected final answer (reference)\n{case.expected_output}")
    if case.expected_tool_calls:
        exp = "\n".join(f"- {e.name} {json.dumps(e.args, default=str)}{' (optional)' if not e.required else ''}" for e in case.expected_tool_calls)
        parts.append(f"## Golden tool calls\n{exp}")
    tools = ", ".join(sorted(n for n, d in ctx.tool_defs.items() if not d.get("distractor")))
    parts.append(f"## Tools that were available\n{tools or '(none)'}")
    parts.append(f"## Agent trajectory (tool calls and results)\n{_format_trajectory(ctx)}")
    parts.append(f"## Agent final answer\n{ctx.output or '(empty)'}")
    if case.calibration_examples:
        ex = "\n".join(f"- score {e.get('score')}: {e.get('reason', '')} | trajectory: {str(e.get('trajectory', ''))[:300]}" for e in case.calibration_examples)
        parts.append(f"## Calibration examples (previously graded)\n{ex}")
    focus = "whether the task was completed" if metric == "task_completion" else "the quality of the path the agent took (tool choice, efficiency, ordering, recovery)"
    parts.append(f"## Rubric for '{metric}' (grade {focus})\n{_rubric_text(rubric)}")
    parts.append(
        "Return JSON: {\"criteria\": [{\"criterion\", \"score\" (1-5), \"reason\"}...], \"score\": overall 1-5, \"reason\": one paragraph}. "
        "Include every rubric criterion exactly once."
    )
    return "\n\n".join(parts)


def _weighted(rubric: list[RubricItem], criteria: list[dict[str, Any]], fallback: float) -> float:
    by_name = {c.get("criterion"): c for c in criteria if isinstance(c, dict)}
    total_w, acc = 0.0, 0.0
    for r in rubric:
        c = by_name.get(r.criterion)
        if not c:
            continue
        try:
            s = float(c.get("score", 0))
        except (TypeError, ValueError):
            continue
        s = max(1.0, min(5.0, s))
        acc += r.weight * (s - 1) / 4
        total_w += r.weight
    if total_w == 0:
        return max(0.0, min(1.0, (fallback - 1) / 4))
    return acc / total_w


async def _judge(ctx: CheckContext, metric: str) -> MetricResult:
    if ctx.judge is None:
        return make_result(metric, 1.0, "No judge model configured.", skipped=True)
    case = ctx.case
    rubric = case.rubric if (case.rubric and (metric == "task_completion" or case.category == "llm_judge")) else DEFAULT_RUBRICS[metric]
    if metric == "trajectory_quality" and case.rubric and case.category == "llm_judge":
        # A user rubric on an llm_judge case usually mixes both concerns: use
        # the trajectory-flavoured criteria for this metric when identifiable.
        traj = [r for r in case.rubric if any(k in r.criterion for k in ("tool", "redundant", "order", "efficien", "trajectory", "step"))]
        rubric = traj or DEFAULT_RUBRICS[metric]
    prompt = build_prompt(case, ctx, metric, rubric)
    try:
        data = await ctx.judge.json_completion(_SYSTEM, prompt, _SCHEMA, max_tokens=2048)
    except Exception as exc:
        return make_result(metric, 0.0, f"judge call failed: {exc}", {"error": str(exc)})
    criteria = data.get("criteria") or []
    overall = data.get("score", 0)
    try:
        overall_f = float(overall)
    except (TypeError, ValueError):
        overall_f = 1.0
    score = _weighted(rubric, criteria, overall_f)
    reason = str(data.get("reason") or "").strip()
    return make_result(
        metric,
        score,
        reason or f"judge score {overall}/5",
        {
            "judge_model": getattr(ctx.judge, "model", None),
            "overall_1_to_5": overall,
            "criteria": criteria,
            "rubric": [r.model_dump(mode="json") for r in rubric],
            "output_format": case.output_format,
        },
    )


@register("task_completion")
async def task_completion(ctx: CheckContext) -> MetricResult:
    return await _judge(ctx, "task_completion")


@register("trajectory_quality")
async def trajectory_quality(ctx: CheckContext) -> MetricResult:
    return await _judge(ctx, "trajectory_quality")


# ---------------------------------------------------------------------------
# Rubric drafting helper (used by the UI "Suggest rubric" button)
# ---------------------------------------------------------------------------

_RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "rubric": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "weight": {"type": "number"},
                    "description": {"type": "string"},
                    "anchors": {
                        "type": "object",
                        "properties": {"1": {"type": "string"}, "3": {"type": "string"}, "5": {"type": "string"}},
                        "required": ["1", "3", "5"],
                        "additionalProperties": False,
                    },
                },
                "required": ["criterion", "weight", "description", "anchors"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rubric"],
    "additionalProperties": False,
}


async def suggest_rubric(judge: Any, case: EvalCase, tool_names: list[str]) -> list[RubricItem]:
    prompt = (
        "Draft a task-specific grading rubric (3-5 criteria, weights summing to 1, snake_case criterion names, "
        "with anchors for scores 1, 3 and 5) for evaluating an AI agent on this task.\n\n"
        f"Task: {case.input}\nAuthor notes: {case.description or '-'}\nExpected tools: "
        f"{[e.name for e in case.expected_tool_calls] or '-'}\nAvailable tools: {tool_names}"
    )
    data = await judge.json_completion("You design calibrated evaluation rubrics for tool-using agents.", prompt, _RUBRIC_SCHEMA, max_tokens=1500)
    items = []
    for r in data.get("rubric", []):
        try:
            items.append(RubricItem(**r))
        except Exception:
            continue
    total = sum(i.weight for i in items) or 1.0
    for i in items:
        i.weight = round(i.weight / total, 3)
    return items
