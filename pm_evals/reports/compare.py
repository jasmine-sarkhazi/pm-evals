"""Compare two reports: per-case and per-metric deltas, regressions, fixes."""

from __future__ import annotations

from typing import Any

from ..models import Report


def _metric_map(report: Report) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in report.results:
        out[r.case_id] = {m.metric: m.score for m in r.metrics if not m.skipped}
    return out


def compare_reports(current: Report, previous: Report) -> dict[str, Any]:
    cur_cases = {r.case_id: r for r in current.results}
    prev_cases = {r.case_id: r for r in previous.results}
    cur_m, prev_m = _metric_map(current), _metric_map(previous)
    regressions, improvements, unchanged, new_cases, removed_cases = [], [], [], [], []
    for cid, r in cur_cases.items():
        p = prev_cases.get(cid)
        if p is None:
            new_cases.append(cid)
            continue
        delta = round(r.score - p.score, 4)
        entry = {"case_id": cid, "before": p.score, "after": r.score, "delta": delta, "passed_before": p.passed, "passed_after": r.passed}
        metric_deltas = []
        for m, s in cur_m.get(cid, {}).items():
            if m in prev_m.get(cid, {}):
                d = round(s - prev_m[cid][m], 4)
                if abs(d) >= 0.05:
                    metric_deltas.append({"metric": m, "before": prev_m[cid][m], "after": s, "delta": d})
        entry["metric_deltas"] = metric_deltas
        if (p.passed and not r.passed) or delta <= -0.1:
            regressions.append(entry)
        elif (not p.passed and r.passed) or delta >= 0.1:
            improvements.append(entry)
        else:
            unchanged.append(entry)
    for cid in prev_cases:
        if cid not in cur_cases:
            removed_cases.append(cid)
    cur_s, prev_s = current.summary, previous.summary
    metric_avg_delta = {}
    for m, v in (cur_s.get("metric_averages") or {}).items():
        pv = (prev_s.get("metric_averages") or {}).get(m)
        if pv is not None:
            metric_avg_delta[m] = {"before": pv, "after": v, "delta": round(v - pv, 4)}
    category_delta = {}
    for c, v in (cur_s.get("by_category") or {}).items():
        pv = (prev_s.get("by_category") or {}).get(c)
        if pv:
            category_delta[c] = {"before": pv.get("pass_rate"), "after": v.get("pass_rate"), "delta": round((v.get("pass_rate") or 0) - (pv.get("pass_rate") or 0), 4)}
    server_delta = []
    prev_sf = {(f.server_name, f.metric): f for f in previous.server_findings}
    for f in current.server_findings:
        pf = prev_sf.get((f.server_name, f.metric))
        if pf is not None and (pf.passed != f.passed or abs(pf.score - f.score) >= 0.05):
            server_delta.append({"server": f.server_name, "metric": f.metric, "before": pf.score, "after": f.score, "passed_before": pf.passed, "passed_after": f.passed})
    return {
        "previous_report_id": previous.id,
        "previous_date": previous.date,
        "previous_model": previous.config.model,
        "previous_label": previous.label,
        "pass_rate": {"before": prev_s.get("pass_rate"), "after": cur_s.get("pass_rate"), "delta": round((cur_s.get("pass_rate") or 0) - (prev_s.get("pass_rate") or 0), 4)},
        "avg_score": {"before": prev_s.get("avg_score"), "after": cur_s.get("avg_score"), "delta": round((cur_s.get("avg_score") or 0) - (prev_s.get("avg_score") or 0), 4)},
        "regressions": sorted(regressions, key=lambda e: e["delta"]),
        "improvements": sorted(improvements, key=lambda e: -e["delta"]),
        "unchanged_count": len(unchanged),
        "new_cases": new_cases,
        "removed_cases": removed_cases,
        "metric_averages": metric_avg_delta,
        "by_category": category_delta,
        "server_findings": server_delta,
        "verdict": (
            "regression" if regressions and len(regressions) >= len(improvements) else ("improvement" if improvements else "no significant change")
        ),
    }
