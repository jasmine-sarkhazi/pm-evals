"""Deterministic checks: tool-call correctness, argument correctness,
call order, post-run state checks, and server-level protocol conformance."""

from __future__ import annotations

import re
from typing import Any, Optional

from ..models import ActualToolCall, ExpectedToolCall, MetricResult, ServerFinding
from .base import CheckContext, make_result, register
from .util import compare_args, get_path, is_read_only, value_matches


def _successful(calls: list[ActualToolCall]) -> list[ActualToolCall]:
    return [c for c in calls if c.exists and not c.distractor and not c.is_error]


def _match_expected(exp: ExpectedToolCall, calls: list[ActualToolCall], used: set[int]) -> Optional[ActualToolCall]:
    """Pick the best unused actual call for an expected call (same name, most
    matching args)."""
    best, best_score = None, -1.0
    for c in calls:
        if c.order in used or c.name.split("__")[-1] != exp.name.split("__")[-1]:
            continue
        s = compare_args(exp.args, c.args, exp.arg_match)["score"] if exp.args else 1.0
        if s > best_score:
            best, best_score = c, s
    if best is not None:
        used.add(best.order)
    return best


def allowed_extra(ctx: CheckContext, name: str) -> bool:
    base = name.split("__")[-1]
    if base in ctx.case.allowed_extra_tools or name in ctx.case.allowed_extra_tools:
        return True
    return is_read_only(name, ctx.tool_defs.get(name))


@register("tool_correctness")
async def tool_correctness(ctx: CheckContext) -> MetricResult:
    expected = ctx.case.expected_tool_calls
    if not expected:
        # No golden calls: pass if the model did not call anything it shouldn't.
        bad = [c.name for c in ctx.calls if not c.exists or c.distractor]
        return make_result("tool_correctness", 1.0 if not bad else 0.0, "No expected tool calls defined; checked only for invented/distractor tools." + (f" Offenders: {bad}" if bad else ""), {"unexpected": bad}, skipped=not bad and not ctx.calls)
    required = [e for e in expected if e.required]
    optional = [e for e in expected if not e.required]
    success = _successful(ctx.calls)
    all_real = [c for c in ctx.calls if c.exists and not c.distractor]
    used: set[int] = set()
    hit, missed, errored = [], [], []
    for e in required:
        m = _match_expected(e, success, used)
        if m is not None:
            hit.append(e.name)
        else:
            attempted = _match_expected(e, all_real, set())
            (errored if attempted is not None and attempted.is_error else missed).append(e.name)
    for e in optional:
        _match_expected(e, success, used)
    expected_names = {e.name for e in expected}
    unexpected = [c.name for c in ctx.calls if c.name.split("__")[-1] not in expected_names and not allowed_extra(ctx, c.name)]
    recall = len(hit) / len(required) if required else 1.0
    penalty = min(0.75, 0.25 * len(unexpected))
    score = max(0.0, recall - penalty)
    reasons = []
    if missed:
        reasons.append(f"missing required calls: {missed}")
    if errored:
        reasons.append(f"required calls attempted but errored: {errored}")
    if unexpected:
        reasons.append(f"unexpected calls: {unexpected}")
    if not reasons:
        reasons.append(f"all {len(required)} required tools called")
    return make_result(
        "tool_correctness",
        score,
        "; ".join(reasons),
        {"hit": hit, "missed": missed, "errored": errored, "unexpected": unexpected, "recall": round(recall, 3)},
    )


@register("argument_correctness")
async def argument_correctness(ctx: CheckContext) -> MetricResult:
    expected = [e for e in ctx.case.expected_tool_calls if e.args]
    if not expected:
        return make_result("argument_correctness", 1.0, "No golden arguments to compare.", skipped=True)
    real = [c for c in ctx.calls if c.exists and not c.distractor]
    used: set[int] = set()
    per_call = []
    scores = []
    for e in expected:
        m = _match_expected(e, real, used)
        if m is None:
            if e.required:
                per_call.append({"tool": e.name, "score": 0.0, "note": "tool not called"})
                scores.append(0.0)
            continue
        cmp = compare_args(e.args, m.args, e.arg_match)
        per_call.append({"tool": e.name, "score": round(cmp["score"], 3), **{k: cmp[k] for k in ("matched", "mismatched", "missing", "extra")}})
        scores.append(cmp["score"])
    score = sum(scores) / len(scores) if scores else 1.0
    problems = [p for p in per_call if p["score"] < 1.0]
    reason = "all arguments match the golden values" if not problems else "; ".join(
        f"{p['tool']}: " + ", ".join(
            [f"missing {p.get('missing')}" if p.get("missing") else ""]
            + [f"mismatched {[m['key'] for m in p.get('mismatched', [])]}" if p.get("mismatched") else ""]
            + [p.get("note", "")]
        ).strip(", ")
        for p in problems
    )
    return make_result("argument_correctness", score, reason, {"per_call": per_call})


@register("call_order")
async def call_order(ctx: CheckContext) -> MetricResult:
    ordered = sorted([e for e in ctx.case.expected_tool_calls if e.order is not None], key=lambda e: e.order or 0)
    if len(ordered) < 2:
        return make_result("call_order", 1.0, "Fewer than two ordered expectations; nothing to check.", skipped=True)
    first_seen: dict[str, int] = {}
    for c in ctx.calls:
        base = c.name.split("__")[-1]
        if c.exists and not c.distractor and base not in first_seen:
            first_seen[base] = c.order
    pairs_ok, pairs_total, violations = 0, 0, []
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            if a.order == b.order:
                continue
            pairs_total += 1
            if a.name in first_seen and b.name in first_seen:
                if first_seen[a.name] < first_seen[b.name]:
                    pairs_ok += 1
                else:
                    violations.append(f"{b.name} was called before {a.name}")
            elif not a.required or not b.required:
                pairs_ok += 1  # optional tool missing: not an ordering violation
            else:
                violations.append(f"cannot check order: {a.name if a.name not in first_seen else b.name} was not called")
    score = pairs_ok / pairs_total if pairs_total else 1.0
    return make_result("call_order", score, "order respected" if not violations else "; ".join(violations), {"first_seen": first_seen, "violations": violations})


_OPS = {
    "eq": lambda actual, v: value_matches(v, actual),
    "contains": lambda actual, v: (str(v).lower() in str(actual).lower()) if not isinstance(actual, (list, dict)) else any(value_matches(v, x) for x in (actual if isinstance(actual, list) else actual.values())),
    "not_contains": lambda actual, v: not _OPS["contains"](actual, v),
    "exists": lambda actual, v: actual is not None,
    "gte": lambda actual, v: float(actual) >= float(v),
    "lte": lambda actual, v: float(actual) <= float(v),
    "len_gte": lambda actual, v: len(actual) >= int(v),
}


@register("state_check")
async def state_check(ctx: CheckContext) -> MetricResult:
    checks = ctx.case.state_checks
    if not checks:
        return make_result("state_check", 1.0, "No state checks defined.", skipped=True)
    if ctx.call_tool is None:
        return make_result("state_check", 1.0, "State checks need a live MCP connection (not available for transcript harnesses).", skipped=True)
    results = []
    passed = 0
    for chk in checks:
        outcome = await ctx.call_tool(chk.tool, chk.args)
        if outcome is None or outcome.is_error:
            results.append({"tool": chk.tool, "ok": False, "note": "tool call failed", "result": getattr(outcome, "text", None)})
            continue
        data = outcome.structured if outcome.structured is not None else outcome.text
        actual = get_path(data, chk.path)
        try:
            ok = bool(_OPS[chk.op](actual, chk.value))
        except Exception as exc:
            ok = False
            results.append({"tool": chk.tool, "ok": False, "note": f"{chk.op} failed: {exc}", "actual": actual})
            continue
        passed += int(ok)
        results.append({"tool": chk.tool, "path": chk.path, "op": chk.op, "expected": chk.value, "actual": actual if not isinstance(actual, (dict, list)) else str(actual)[:300], "ok": ok})
    score = passed / len(checks)
    failed = [r for r in results if not r["ok"]]
    return make_result("state_check", score, "state matches intent" if not failed else f"{len(failed)} of {len(checks)} state checks failed", {"checks": results})


# ---------------------------------------------------------------------------
# Server-level: protocol conformance
# ---------------------------------------------------------------------------

_IDENT = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


async def protocol_conformance(conn: Any, tools: list[Any]) -> ServerFinding:
    """Run once per server. ``conn`` is an :class:`MCPConnection`."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, note: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "note": note})

    hs = conn.handshake or {}
    add("initialize handshake", hs.get("ok", False), f"protocol {hs.get('protocol_version')}")
    add("server_info present", bool((hs.get("server_info") or {}).get("name")), str((hs.get("server_info") or {}).get("name")))
    caps = hs.get("capabilities") or {}
    add("tools capability advertised", "tools" in caps, str(list(caps.keys())))
    add("tools/list returned tools", len(tools) > 0, f"{len(tools)} tools")
    names = [t.name for t in tools]
    add("tool names unique", len(names) == len(set(names)))
    add("tool names valid identifiers", all(_IDENT.match(n) for n in names), str([n for n in names if not _IDENT.match(n)]))
    no_desc = [t.name for t in tools if not (t.description or "").strip()]
    add("every tool has a description", not no_desc, f"missing: {no_desc}" if no_desc else "")
    short_desc = [t.name for t in tools if 0 < len((t.description or "").strip()) < 15]
    add("descriptions are meaningful (>=15 chars)", not short_desc, f"too short: {short_desc}" if short_desc else "")
    bad_schema: list[str] = []
    try:
        import jsonschema

        for t in tools:
            schema = t.input_schema or {}
            try:
                jsonschema.Draft202012Validator.check_schema(schema)
                if schema.get("type", "object") != "object":
                    bad_schema.append(f"{t.name}: type must be object")
            except Exception as exc:
                bad_schema.append(f"{t.name}: {str(exc)[:80]}")
    except ImportError:  # pragma: no cover
        for t in tools:
            if not isinstance(t.input_schema, dict) or t.input_schema.get("type", "object") != "object":
                bad_schema.append(t.name)
    add("input schemas are valid JSON Schema objects", not bad_schema, "; ".join(bad_schema))
    undocumented: list[str] = []
    for t in tools:
        props = (t.input_schema or {}).get("properties", {}) or {}
        for pname, pdef in props.items():
            if isinstance(pdef, dict) and not (pdef.get("description") or pdef.get("title")) and "type" not in pdef and "$ref" not in pdef and "anyOf" not in pdef:
                undocumented.append(f"{t.name}.{pname}")
    add("parameters have types or descriptions", not undocumented, ", ".join(undocumented[:10]))
    required_missing: list[str] = []
    for t in tools:
        props = (t.input_schema or {}).get("properties", {}) or {}
        for r in (t.input_schema or {}).get("required", []) or []:
            if r not in props:
                required_missing.append(f"{t.name}.{r}")
    add("required params exist in properties", not required_missing, ", ".join(required_missing))
    # Behavioural probes
    unknown = await conn.call_tool("__pm_evals_nonexistent_tool__", {})
    add("unknown tool returns an error (not a crash)", unknown.is_error and not unknown.malformed, unknown.text[:120])
    if tools:
        target = next((t for t in tools if (t.input_schema or {}).get("required")), None)
        if target is not None:
            missing = await conn.call_tool(target.name, {})
            add("missing required args are rejected", missing.is_error, f"{target.name}: {missing.text[:100]}")
    try:
        second = await conn.raw_list_tools()
        add("tools/list is stable across calls", sorted(t.name for t in second.tools) == sorted(names))
    except Exception as exc:
        add("tools/list is stable across calls", False, str(exc)[:100])
    passed = sum(1 for c in checks if c["ok"])
    score = passed / len(checks) if checks else 0.0
    failing = [c["check"] for c in checks if not c["ok"]]
    return ServerFinding(
        server_name=conn.name,
        metric="protocol_conformance",
        score=round(score, 3),
        passed=score >= 0.85,
        reason="conforms to MCP protocol" if not failing else f"failed: {', '.join(failing)}",
        details={"checks": checks},
    )
