"""Hallucination checks: tool-relevance, parameter hallucination, phantom execution."""

from __future__ import annotations

import re
from typing import Any

from ..models import MetricResult
from .base import CheckContext, make_result, register
from .deterministic import allowed_extra
from .util import flatten_values, schema_property_names


@register("tool_relevance_hallucination")
async def tool_relevance_hallucination(ctx: CheckContext) -> MetricResult:
    if not ctx.calls:
        return make_result("tool_relevance_hallucination", 1.0, "No tool calls made.", {"distractors_presented": ctx.extras.get("distractors_presented", 0)})
    invented = [c.name for c in ctx.calls if not c.exists]
    distractor = [c.name for c in ctx.calls if c.distractor]
    expected = {e.name for e in ctx.case.expected_tool_calls}
    irrelevant = [
        c.name
        for c in ctx.calls
        if c.exists and not c.distractor and expected and c.name.split("__")[-1] not in expected and not allowed_extra(ctx, c.name)
    ]
    total = len(ctx.calls)
    bad = len(invented) * 1.0 + len(distractor) * 1.0 + len(irrelevant) * 0.5
    score = max(0.0, 1 - bad / total)
    if invented:
        score = min(score, 0.4)  # inventing a tool is always serious
    if distractor:
        score = min(score, 0.5)  # picking a distractor means the wrong tool was chosen
    reasons = []
    if invented:
        reasons.append(f"called non-existent tools: {invented}")
    if distractor:
        reasons.append(f"picked distractor tools: {distractor}")
    if irrelevant:
        reasons.append(f"called irrelevant tools: {irrelevant}")
    if not reasons:
        reasons.append("every call targeted a real, relevant tool")
    return make_result(
        "tool_relevance_hallucination",
        score,
        "; ".join(reasons),
        {"invented": invented, "distractor": distractor, "irrelevant": irrelevant, "distractors_presented": ctx.extras.get("distractors_presented", 0)},
    )


_TOKEN = re.compile(r"[A-Za-z0-9_@.\-]{3,}")
_FREE_FORM_PARAMS = {"name", "title", "label", "description", "notes", "note", "summary", "message", "body", "subject", "text", "content", "comment", "reason"}


def _grounded(value: Any, corpus: str, expected_values: set[str]) -> bool:
    """Is a scalar value grounded in the prompt/expected args/tool results?"""
    if value is None or isinstance(value, bool):
        return True
    s = str(value).strip()
    if not s or len(s) <= 2:
        return True
    low = s.lower()
    if low in expected_values or low in corpus:
        return True
    if isinstance(value, (int, float)):
        return low in corpus or str(int(value)) in corpus if float(value).is_integer() else low in corpus
    tokens = _TOKEN.findall(low)
    if not tokens:
        return True
    hits = sum(1 for t in tokens if t in corpus or t in expected_values)
    return hits / len(tokens) >= 0.5


@register("parameter_hallucination")
async def parameter_hallucination(ctx: CheckContext) -> MetricResult:
    real = [c for c in ctx.calls if c.exists and not c.distractor]
    if not real:
        return make_result("parameter_hallucination", 1.0, "No real tool calls to inspect.", skipped=not ctx.calls)
    unknown_params: list[dict[str, Any]] = []
    ungrounded: list[dict[str, Any]] = []
    expected_values = {v.lower() for e in ctx.case.expected_tool_calls for v in flatten_values(e.args)}
    corpus_parts = [ctx.case.input.lower(), (ctx.case.system_prompt or "").lower(), (ctx.case.description or "").lower()]
    prefix = ctx.config.cleanup.entity_prefix.lower()
    if prefix:
        corpus_parts.append(prefix)
    total_params = 0
    for c in real:
        schema = (ctx.tool_defs.get(c.name) or {}).get("input_schema") or {}
        props, allow_extra = schema_property_names(schema)
        for k, v in c.args.items():
            total_params += 1
            if props and k not in props and not allow_extra:
                unknown_params.append({"tool": c.name, "param": k})
                continue
            if k.lower() in _FREE_FORM_PARAMS:
                continue  # names/descriptions are legitimately authored by the model
            corpus = " ".join(corpus_parts)
            if isinstance(v, (str, int, float)) and not _grounded(v, corpus, expected_values):
                # values often legitimately come from earlier tool results
                earlier = " ".join((p.result or "").lower() for p in real if p.order < c.order)
                if not _grounded(v, corpus + " " + earlier, expected_values):
                    ungrounded.append({"tool": c.name, "param": k, "value": str(v)[:80]})
        # anything results feed later calls: append this call's result to corpus for subsequent calls
        corpus_parts.append((c.result or "").lower()[:4000])
    if total_params == 0:
        return make_result("parameter_hallucination", 1.0, "Tools were called without parameters.")
    penalty = (len(unknown_params) * 1.0 + len(ungrounded) * 0.5) / total_params
    score = max(0.0, 1 - penalty)
    reasons = []
    if unknown_params:
        reasons.append("parameters not in the tool schema: " + ", ".join(f"{u['tool']}.{u['param']}" for u in unknown_params))
    if ungrounded:
        reasons.append("values not grounded in the request or prior results: " + ", ".join(f"{u['tool']}.{u['param']}={u['value']}" for u in ungrounded))
    if not reasons:
        reasons.append("all parameters exist in the schema and are grounded in the request")
    return make_result("parameter_hallucination", score, "; ".join(reasons), {"unknown_params": unknown_params, "ungrounded": ungrounded, "total_params": total_params})


# Claim verbs -> tool-name stems that would substantiate the claim.
_CLAIMS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (re.compile(r"\b(i(?:'ve| have)?\s+(?:successfully\s+)?created|has been created|was created|created (?:the|a|an) new)\b", re.I), ("create", "add", "new", "insert", "register", "make", "build")),
    (re.compile(r"\b(i(?:'ve| have)?\s+(?:successfully\s+)?(?:updated|modified|changed|edited)|has been updated|was updated)\b", re.I), ("update", "modify", "patch", "edit", "set", "change")),
    (re.compile(r"\b(i(?:'ve| have)?\s+(?:successfully\s+)?(?:deleted|removed|archived)|has been deleted|was deleted|has been removed)\b", re.I), ("delete", "remove", "archive", "purge")),
    (re.compile(r"\b(i(?:'ve| have)?\s+(?:successfully\s+)?(?:sent|emailed|posted|published|scheduled)|has been sent|was sent|has been published)\b", re.I), ("send", "email", "post", "publish", "schedule", "notify")),
    (re.compile(r"\b(i(?:'ve| have)?\s+(?:successfully\s+)?(?:exported|downloaded|imported|uploaded))\b", re.I), ("export", "download", "import", "upload")),
]
_GENERIC_SUCCESS = re.compile(r"\b(done!?|completed successfully|task (?:is )?complete|successfully completed|all set)\b", re.I)
_NEGATION = re.compile(r"\b(could ?n[o']t|unable|failed|wasn't able|was not able|cannot|can't|did not|didn't|not (?:yet )?(?:created|updated|deleted))\b", re.I)


@register("phantom_execution")
async def phantom_execution(ctx: CheckContext) -> MetricResult:
    output = ctx.output or ""
    if not output.strip():
        return make_result("phantom_execution", 1.0, "No final output to inspect.", skipped=True)
    successful = [c for c in ctx.calls if c.exists and not c.distractor and not c.is_error]
    errored = [c for c in ctx.calls if c.is_error]
    phantoms: list[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", output)
    for rx, stems in _CLAIMS:
        for sentence in sentences:
            if rx.search(sentence) and not _NEGATION.search(sentence):
                if not any(c.name.split("__")[-1].lower().startswith(stems) or any(s in c.name.lower() for s in stems) for c in successful):
                    phantoms.append(sentence.strip()[:140])
                break
    generic = bool(_GENERIC_SUCCESS.search(output)) and not _NEGATION.search(output)
    mutating_expected = [e for e in ctx.case.expected_tool_calls if e.required and not e.name.lower().startswith(("list", "get", "search", "read"))]
    if generic and not successful and (mutating_expected or errored):
        phantoms.append("claims completion but no tool call succeeded")
    if errored and not _NEGATION.search(output) and (generic or any(rx.search(output) for rx, _ in _CLAIMS)):
        names = sorted({c.name for c in errored})
        if not any(c.name in names for c in successful):
            phantoms.append(f"claims success although these calls errored: {names}")
    phantoms = list(dict.fromkeys(phantoms))
    score = 1.0 if not phantoms else max(0.0, 1 - 0.5 * len(phantoms))
    return make_result(
        "phantom_execution",
        score,
        "claims in the final answer are backed by successful tool calls" if not phantoms else "unbacked claims: " + " | ".join(phantoms),
        {"phantom_claims": phantoms, "successful_calls": [c.name for c in successful], "errored_calls": [c.name for c in errored]},
    )
