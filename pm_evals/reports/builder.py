"""Build the dated report object, plus Markdown and self-contained HTML renderings."""

from __future__ import annotations

import datetime as dt
import html
import json
from typing import Any, Optional

from ..checks.base import METRIC_HELP
from ..models import CaseResult, Report, RunConfig, ServerFinding
from .compare import compare_reports


def summarize(results: list[CaseResult], server_findings: list[ServerFinding]) -> dict[str, Any]:
    n = len(results)
    passed = sum(1 for r in results if r.passed)
    errors = sum(1 for r in results if r.error)
    by_cat: dict[str, dict[str, Any]] = {}
    metric_scores: dict[str, list[float]] = {}
    metric_pass: dict[str, list[bool]] = {}
    for r in results:
        cat = by_cat.setdefault(r.category, {"cases": 0, "passed": 0, "scores": []})
        cat["cases"] += 1
        cat["passed"] += int(r.passed)
        cat["scores"].append(r.score)
        for m in r.metrics:
            if m.skipped:
                continue
            metric_scores.setdefault(m.metric, []).append(m.score)
            metric_pass.setdefault(m.metric, []).append(m.passed)
    for cat in by_cat.values():
        cat["pass_rate"] = round(cat["passed"] / cat["cases"], 4) if cat["cases"] else 0
        cat["avg_score"] = round(sum(cat["scores"]) / len(cat["scores"]), 4) if cat["scores"] else 0
        del cat["scores"]
    metric_averages = {m: round(sum(v) / len(v), 4) for m, v in metric_scores.items()}
    metric_pass_rates = {m: round(sum(v) / len(v), 4) for m, v in metric_pass.items()}
    worst = sorted(metric_averages.items(), key=lambda kv: kv[1])[:3]
    failed_cases = [r.case_id for r in results if not r.passed]
    distractor_cases = [r for r in results if r.distractors_presented]
    distractor_summary = None
    if distractor_cases:
        picks = 0
        for r in distractor_cases:
            for m in r.metrics:
                if m.metric == "tool_relevance_hallucination":
                    picks += len(m.details.get("distractor", []))
        distractor_summary = {
            "cases": len(distractor_cases),
            "avg_distractors": round(sum(r.distractors_presented for r in distractor_cases) / len(distractor_cases), 1),
            "distractor_picks": picks,
            "selection_accuracy": round(1 - picks / max(1, sum(len(r.actual_tool_calls) for r in distractor_cases)), 4),
        }
    usage = {"input_tokens": sum(int(r.usage.get("input_tokens", 0) or 0) for r in results), "output_tokens": sum(int(r.usage.get("output_tokens", 0) or 0) for r in results)}
    return {
        "cases": n,
        "passed": passed,
        "failed": n - passed,
        "errors": errors,
        "pass_rate": round(passed / n, 4) if n else 0,
        "avg_score": round(sum(r.score for r in results) / n, 4) if n else 0,
        "by_category": by_cat,
        "metric_averages": metric_averages,
        "metric_pass_rates": metric_pass_rates,
        "weakest_metrics": [{"metric": m, "avg": s} for m, s in worst],
        "failed_cases": failed_cases,
        "server_findings_failed": [f"{f.server_name}:{f.metric}" for f in server_findings if not f.passed],
        "distractors": distractor_summary,
        "usage": usage,
        "total_duration_ms": round(sum(r.duration_ms for r in results)),
    }


def build_report(
    run_id: str,
    cfg: RunConfig,
    results: list[CaseResult],
    server_findings: list[ServerFinding],
    cleanup: dict[str, Any],
    tool_inventory: dict[str, list[dict[str, Any]]],
    warnings: list[str],
    previous: Optional[Report] = None,
) -> Report:
    now = dt.datetime.now(dt.timezone.utc)
    report = Report(
        id=run_id,
        created_at=now,
        date=now.strftime("%Y-%m-%d"),
        dataset=cfg.dataset,
        label=cfg.label,
        config=cfg,
        summary=summarize(results, server_findings),
        results=results,
        server_findings=server_findings,
        cleanup=cleanup,
        tool_inventory=tool_inventory,
        warnings=warnings,
    )
    if previous is not None:
        report.comparison = compare_reports(report, previous)
    return report


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return "-"


def _delta(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    sign = "+" if f > 0 else ""
    return f"{sign}{f * 100:.0f} pts"


def render_markdown(report: Report) -> str:
    s = report.summary
    cfg = report.config
    lines = [
        f"# MCP eval report: {report.dataset}",
        "",
        f"**Date:** {report.created_at.strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Report id:** `{report.id}`  ",
        f"**Model under test:** `{cfg.model}` via `{cfg.harness}` harness  ",
        f"**Judge model:** `{cfg.judge_model}`  ",
        f"**Servers:** {', '.join(cfg.servers and [x.server_name for x in cfg.servers] or list(report.tool_inventory)) or '-'}  ",
        (f"**Label:** {report.label}  " if report.label else ""),
        "",
        "## Summary",
        "",
        f"| Cases | Passed | Failed | Errors | Pass rate | Avg score |",
        f"|---|---|---|---|---|---|",
        f"| {s['cases']} | {s['passed']} | {s['failed']} | {s['errors']} | {_pct(s['pass_rate'])} | {_pct(s['avg_score'])} |",
        "",
    ]
    if report.comparison:
        c = report.comparison
        lines += [
            f"### Compared with {c['previous_report_id']} ({c['previous_date']}, model `{c['previous_model']}`)",
            "",
            f"- Verdict: **{c['verdict']}**",
            f"- Pass rate: {_pct(c['pass_rate']['before'])} -> {_pct(c['pass_rate']['after'])} ({_delta(c['pass_rate']['delta'])})",
            f"- Avg score: {_pct(c['avg_score']['before'])} -> {_pct(c['avg_score']['after'])} ({_delta(c['avg_score']['delta'])})",
            f"- Regressions: {len(c['regressions'])}, improvements: {len(c['improvements'])}, unchanged: {c['unchanged_count']}, new: {len(c['new_cases'])}, removed: {len(c['removed_cases'])}",
            "",
        ]
        for e in c["regressions"]:
            lines.append(f"  - REGRESSION `{e['case_id']}`: {_pct(e['before'])} -> {_pct(e['after'])} " + ", ".join(f"{d['metric']} {_delta(d['delta'])}" for d in e["metric_deltas"]))
        for e in c["improvements"]:
            lines.append(f"  - improved `{e['case_id']}`: {_pct(e['before'])} -> {_pct(e['after'])}")
        lines.append("")
    lines += ["## By category", "", "| Category | Cases | Passed | Pass rate | Avg score |", "|---|---|---|---|---|"]
    for cat, v in s["by_category"].items():
        lines.append(f"| {cat} | {v['cases']} | {v['passed']} | {_pct(v['pass_rate'])} | {_pct(v['avg_score'])} |")
    lines += ["", "## By metric", "", "| Metric | Avg score | Pass rate | What it checks |", "|---|---|---|---|"]
    for m, v in sorted(s["metric_averages"].items(), key=lambda kv: kv[1]):
        lines.append(f"| {m} | {_pct(v)} | {_pct(s['metric_pass_rates'].get(m))} | {METRIC_HELP.get(m, '')} |")
    if s.get("distractors"):
        d = s["distractors"]
        lines += ["", "## Distractor simulation", "", f"- {d['cases']} cases ran with ~{d['avg_distractors']} distractor tools each", f"- The model picked a distractor {d['distractor_picks']} time(s); tool selection accuracy {_pct(d['selection_accuracy'])}"]
    lines += ["", "## MCP server checks", "", "| Server | Check | Result | Score | Notes |", "|---|---|---|---|---|"]
    for f in report.server_findings:
        lines.append(f"| {f.server_name} | {f.metric} | {'PASS' if f.passed else 'FAIL'} | {_pct(f.score)} | {f.reason} |")
    lines += ["", "## Cases", "", "| Case | Category | Result | Score | Weakest metric | Notes |", "|---|---|---|---|---|---|"]
    for r in report.results:
        weakest = min((m for m in r.metrics if not m.skipped), key=lambda m: m.score, default=None)
        note = r.error or (weakest.reason if weakest else "")
        lines.append(f"| `{r.case_id}` | {r.category} | {'PASS' if r.passed else 'FAIL'} | {_pct(r.score)} | {weakest.metric if weakest else '-'} | {note[:160].replace('|', '/')} |")
    lines += ["", "### Case details", ""]
    for r in report.results:
        lines += [f"#### {r.case_id} - {'PASS' if r.passed else 'FAIL'} ({_pct(r.score)}, threshold {_pct(r.threshold)})", "", f"Prompt: {r.input}", ""]
        if r.error:
            lines += [f"Error: {r.error}", ""]
        for m in r.metrics:
            tag = "skipped" if m.skipped else ("pass" if m.passed else "FAIL")
            lines.append(f"- {m.metric}: {tag} {_pct(m.score)} - {m.reason}")
        if r.actual_tool_calls:
            lines += ["", "Tool calls:"]
            for c in r.actual_tool_calls:
                flag = " (non-existent)" if not c.exists else (" (distractor)" if c.distractor else "")
                lines.append(f"  {c.order}. `{c.name}`{flag} {json.dumps(c.args, default=str)[:200]} -> {'ERROR ' if c.is_error else ''}{(c.result or '')[:160].replace(chr(10), ' ')}")
        if r.actual_output:
            lines += ["", f"Final answer: {r.actual_output[:600]}"]
        lines.append("")
    cl = report.cleanup or {}
    lines += ["## Cleanup", "", f"- Deleted: {cl.get('deleted', 0)}, skipped: {cl.get('skipped', 0)}, failed: {cl.get('failed', 0)}, left behind: {len(cl.get('left_behind', []))}"]
    for w in cl.get("warnings", []):
        lines.append(f"- Warning: {w}")
    for e in cl.get("left_behind", []):
        lines.append(f"- Left behind: {e.get('tool')} id={e.get('entity_id')} name={e.get('name')} ({e.get('cleanup_note')})")
    if report.warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in report.warnings]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#f6f6f4;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8985;--line:#e4e3df;--blue:#2a78d6;--good:#0ca30c;--warn:#fab219;--bad:#d03b3b;--goodbg:#e9f7e9;--warnbg:#fff5dc;--badbg:#fbe9e9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:32px 20px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}h3{font-size:16px;margin:20px 0 8px}
.meta{color:var(--ink2);font-size:14px}.meta code{background:#ececea;padding:1px 6px;border-radius:4px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}.tile .v{font-size:28px;font-weight:600;margin-top:2px}.tile .d{font-size:13px;color:var(--ink2)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:14px}th,td{padding:9px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}th{background:#fafaf8;font-weight:600;color:var(--ink2);font-size:13px}tr:last-child td{border-bottom:0}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600}.pass{background:var(--goodbg);color:#006300}.fail{background:var(--badbg);color:#9a1f1f}.skip{background:#ececea;color:var(--ink2)}.warnp{background:var(--warnbg);color:#7a5200}
.bar{height:8px;background:#ececea;border-radius:4px;overflow:hidden;min-width:90px}.bar i{display:block;height:100%;background:var(--blue);border-radius:4px}.bar.good i{background:var(--good)}.bar.bad i{background:var(--bad)}.bar.warn i{background:var(--warn)}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:8px 0}summary{cursor:pointer;font-weight:600}
pre{background:#f0f0ee;padding:10px;border-radius:8px;overflow:auto;font-size:12.5px;white-space:pre-wrap;word-break:break-word}
.muted{color:var(--muted)}.delta-up{color:#006300;font-weight:600}.delta-down{color:#9a1f1f;font-weight:600}
ul.plain{padding-left:18px}.callout{border-left:4px solid var(--warn);background:var(--warnbg);padding:10px 14px;border-radius:6px;margin:10px 0}.callout.bad{border-color:var(--bad);background:var(--badbg)}.callout.good{border-color:var(--good);background:var(--goodbg)}
@media print{details{page-break-inside:avoid}}
"""


def _e(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _bar(score: float, threshold: float = 0.8) -> str:
    cls = "good" if score >= threshold else ("warn" if score >= threshold - 0.2 else "bad")
    return f'<div class="bar {cls}" title="{score:.2f}"><i style="width:{max(2, score * 100):.0f}%"></i></div>'


def _metric_deltas(entry: dict[str, Any]) -> str:
    return _e(", ".join(f"{d['metric']} {_delta(d['delta'])}" for d in entry.get("metric_deltas", [])))


def _pill(passed: bool, skipped: bool = False) -> str:
    if skipped:
        return '<span class="pill skip">– skipped</span>'
    return '<span class="pill pass">✓ pass</span>' if passed else '<span class="pill fail">✗ fail</span>'


def render_html(report: Report) -> str:
    s = report.summary
    cfg = report.config
    servers = [x.server_name for x in cfg.servers] or list(report.tool_inventory)
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>MCP eval report {_e(report.dataset)} {_e(report.date)}</title><style>{_CSS}</style></head><body><div class='wrap'>"]
    out.append(f"<h1>MCP eval report: {_e(report.dataset)}</h1><div class='meta'>{_e(report.created_at.strftime('%A %d %B %Y, %H:%M UTC'))} &middot; report <code>{_e(report.id)}</code>" + (f" &middot; {_e(report.label)}" if report.label else "") + "</div>")
    out.append(f"<div class='meta'>Model under test <code>{_e(cfg.model)}</code> via <code>{_e(cfg.harness)}</code> harness &middot; judge <code>{_e(cfg.judge_model)}</code> &middot; servers: {_e(', '.join(servers) or '-')}</div>")
    rate = s["pass_rate"]
    verdict_cls = "good" if rate >= 0.9 else ("" if rate >= 0.7 else "bad")
    out.append("<div class='tiles'>")
    out.append(f"<div class='tile'><div class='k'>Pass rate</div><div class='v'>{_pct(rate)}</div><div class='d'>{s['passed']} of {s['cases']} cases</div></div>")
    out.append(f"<div class='tile'><div class='k'>Average score</div><div class='v'>{_pct(s['avg_score'])}</div><div class='d'>across all metrics</div></div>")
    out.append(f"<div class='tile'><div class='k'>Failed</div><div class='v'>{s['failed']}</div><div class='d'>{s['errors']} with run errors</div></div>")
    sf_fail = len(s.get("server_findings_failed", []))
    out.append(f"<div class='tile'><div class='k'>Server checks</div><div class='v'>{len(report.server_findings) - sf_fail}/{len(report.server_findings)}</div><div class='d'>passed</div></div>")
    cl = report.cleanup or {}
    out.append(f"<div class='tile'><div class='k'>Cleanup</div><div class='v'>{cl.get('deleted', 0)}</div><div class='d'>deleted, {len(cl.get('left_behind', []))} left behind</div></div>")
    out.append("</div>")
    if s.get("weakest_metrics"):
        w = s["weakest_metrics"][0]
        if w["avg"] < 0.8:
            out.append(f"<div class='callout {verdict_cls}'><strong>Where to look first:</strong> the weakest check is <code>{_e(w['metric'])}</code> at {_pct(w['avg'])} average. {_e(METRIC_HELP.get(w['metric'], ''))}</div>")
    failed_sf = [f for f in report.server_findings if not f.passed]
    if failed_sf:
        out.append("<div class='callout bad'><strong>MCP-level problems found.</strong> Fix these before trusting the case scores: " + "; ".join(f"<code>{_e(f.server_name)}</code> {_e(f.metric)}: {_e(f.reason)}" for f in failed_sf) + "</div>")

    if report.comparison:
        c = report.comparison
        cls = "good" if c["verdict"] == "improvement" else ("bad" if c["verdict"] == "regression" else "")
        out.append(f"<h2>Comparison with previous report</h2><div class='callout {cls}'>Compared with <code>{_e(c['previous_report_id'])}</code> from {_e(c['previous_date'])} (model <code>{_e(c['previous_model'])}</code>): <strong>{_e(c['verdict'])}</strong>. Pass rate {_pct(c['pass_rate']['before'])} → {_pct(c['pass_rate']['after'])} (<span class='{'delta-up' if c['pass_rate']['delta'] >= 0 else 'delta-down'}'>{_delta(c['pass_rate']['delta'])}</span>), avg score {_pct(c['avg_score']['before'])} → {_pct(c['avg_score']['after'])}.</div>")
        out.append("<table><tr><th>Case</th><th>Change</th><th>Before</th><th>After</th><th>Metric changes</th></tr>")
        for e in c["regressions"]:
            out.append(f"<tr><td><code>{_e(e['case_id'])}</code></td><td><span class='pill fail'>▼ regression</span></td><td>{_pct(e['before'])}</td><td>{_pct(e['after'])}</td><td>{_metric_deltas(e)}</td></tr>")
        for e in c["improvements"]:
            out.append(f"<tr><td><code>{_e(e['case_id'])}</code></td><td><span class='pill pass'>▲ improved</span></td><td>{_pct(e['before'])}</td><td>{_pct(e['after'])}</td><td>{_metric_deltas(e)}</td></tr>")
        if not c["regressions"] and not c["improvements"]:
            out.append("<tr><td colspan='5' class='muted'>No case changed by more than 10 points.</td></tr>")
        out.append("</table>")
        if c["new_cases"] or c["removed_cases"]:
            out.append(f"<p class='muted'>New cases: {_e(', '.join(c['new_cases']) or '-')}. Removed cases: {_e(', '.join(c['removed_cases']) or '-')}.</p>")
        if c["metric_averages"]:
            out.append("<h3>Metric averages</h3><table><tr><th>Metric</th><th>Before</th><th>After</th><th>Delta</th></tr>")
            for m, d in sorted(c["metric_averages"].items(), key=lambda kv: kv[1]["delta"]):
                out.append(f"<tr><td>{_e(m)}</td><td>{_pct(d['before'])}</td><td>{_pct(d['after'])}</td><td class='{'delta-up' if d['delta'] > 0 else ('delta-down' if d['delta'] < 0 else '')}'>{_delta(d['delta']) or '0'}</td></tr>")
            out.append("</table>")

    out.append("<h2>Results by category</h2><table><tr><th>Category</th><th>Cases</th><th>Passed</th><th>Pass rate</th><th>Avg score</th></tr>")
    for cat, v in s["by_category"].items():
        out.append(f"<tr><td>{_e(cat)}</td><td>{v['cases']}</td><td>{v['passed']}</td><td>{_pct(v['pass_rate'])}</td><td>{_bar(v['avg_score'])} {_pct(v['avg_score'])}</td></tr>")
    out.append("</table>")
    out.append("<h2>Results by check</h2><table><tr><th>Check</th><th>Avg score</th><th>Pass rate</th><th>What it checks</th></tr>")
    for m, v in sorted(s["metric_averages"].items(), key=lambda kv: kv[1]):
        out.append(f"<tr><td><code>{_e(m)}</code></td><td>{_bar(v)} {_pct(v)}</td><td>{_pct(s['metric_pass_rates'].get(m))}</td><td class='muted'>{_e(METRIC_HELP.get(m, ''))}</td></tr>")
    out.append("</table>")
    if s.get("distractors"):
        d = s["distractors"]
        out.append(f"<h2>Distractor simulation</h2><p>{d['cases']} cases ran with about {d['avg_distractors']} distractor tools mixed into the tool list. The model picked a distractor <strong>{d['distractor_picks']}</strong> time(s); tool selection accuracy <strong>{_pct(d['selection_accuracy'])}</strong>.</p>")

    out.append("<h2>MCP server checks</h2><p class='muted'>Run once per server, before any case. A misleading tool description corrupts every score below, so fix these first.</p><table><tr><th>Server</th><th>Check</th><th>Result</th><th>Score</th><th>Notes</th></tr>")
    for f in report.server_findings:
        out.append(f"<tr><td>{_e(f.server_name)}</td><td><code>{_e(f.metric)}</code></td><td>{_pill(f.passed)}</td><td>{_pct(f.score)}</td><td>{_e(f.reason)}</td></tr>")
    out.append("</table>")
    for f in report.server_findings:
        if f.details:
            out.append(f"<details><summary>{_e(f.server_name)} · {_e(f.metric)} details</summary><pre>{_e(json.dumps(f.details, indent=2, default=str)[:12000])}</pre></details>")

    out.append("<h2>Cases</h2><table><tr><th>Case</th><th>Category</th><th>Result</th><th>Score</th><th>Weakest check</th></tr>")
    for r in report.results:
        weakest = min((m for m in r.metrics if not m.skipped), key=lambda m: m.score, default=None)
        out.append(f"<tr><td><a href='#case-{_e(r.case_id)}'><code>{_e(r.case_id)}</code></a></td><td>{_e(r.category)}</td><td>{_pill(r.passed)}</td><td>{_bar(r.score, r.threshold)} {_pct(r.score)}</td><td class='muted'>{_e(r.error or (f'{weakest.metric}: {weakest.reason}' if weakest else ''))[:200]}</td></tr>")
    out.append("</table><h3>Case details</h3>")
    for r in report.results:
        out.append(f"<details id='case-{_e(r.case_id)}' {'open' if not r.passed else ''}><summary>{_pill(r.passed)} <code>{_e(r.case_id)}</code> · {_pct(r.score)} (threshold {_pct(r.threshold)}) · {_e(r.category)}</summary>")
        out.append(f"<p><strong>Prompt:</strong> {_e(r.input)}</p>")
        if r.description:
            out.append(f"<p class='muted'>{_e(r.description)}</p>")
        if r.error:
            out.append(f"<div class='callout bad'>Run error: {_e(r.error)}</div>")
        out.append("<table><tr><th>Check</th><th>Result</th><th>Score</th><th>Why</th></tr>")
        for m in r.metrics:
            out.append(f"<tr><td><code>{_e(m.metric)}</code></td><td>{_pill(m.passed, m.skipped)}</td><td>{_pct(m.score)}</td><td>{_e(m.reason)}</td></tr>")
        out.append("</table>")
        if r.actual_tool_calls:
            out.append("<p><strong>Tool calls</strong></p><ol>")
            for c in r.actual_tool_calls:
                flag = " <span class='pill fail'>non-existent tool</span>" if not c.exists else (" <span class='pill warnp'>distractor</span>" if c.distractor else "")
                out.append(f"<li><code>{_e(c.name)}</code>{flag} <span class='muted'>{_e(json.dumps(c.args, default=str)[:400])}</span><br><span class='{'delta-down' if c.is_error else ''}'>{'ERROR: ' if c.is_error else '→ '}</span>{_e((c.result or '')[:400])}</li>")
            out.append("</ol>")
        if r.actual_output:
            out.append(f"<p><strong>Final answer</strong></p><pre>{_e(r.actual_output[:2000])}</pre>")
        judge = [m for m in r.metrics if m.category == "llm_judge" and m.details.get("criteria")]
        for m in judge:
            out.append(f"<p><strong>Judge breakdown ({_e(m.metric)}, {_e(m.details.get('judge_model'))})</strong></p><table><tr><th>Criterion</th><th>Score (1-5)</th><th>Reason</th></tr>")
            for cr in m.details["criteria"]:
                out.append(f"<tr><td>{_e(cr.get('criterion'))}</td><td>{_e(cr.get('score'))}</td><td>{_e(cr.get('reason'))}</td></tr>")
            out.append("</table>")
        if r.created_entities:
            out.append("<p><strong>Entities created</strong>: " + ", ".join(f"<code>{_e(e.entity_id)}</code> ({_e(e.name)}) {'deleted' if e.deleted else ('kept: ' + _e(e.cleanup_note))}" for e in r.created_entities) + "</p>")
        out.append("</details>")

    out.append("<h2>Cleanup</h2>")
    out.append(f"<p>Deleted <strong>{cl.get('deleted', 0)}</strong>, skipped {cl.get('skipped', 0)}, failed {cl.get('failed', 0)}, left behind {len(cl.get('left_behind', []))}.</p>")
    for w in cl.get("warnings", []):
        out.append(f"<div class='callout'>{_e(w)}</div>")
    if cl.get("left_behind"):
        out.append("<table><tr><th>Entity</th><th>Created by</th><th>Name</th><th>Why it was kept</th></tr>")
        for e in cl["left_behind"]:
            out.append(f"<tr><td><code>{_e(e.get('entity_id'))}</code></td><td>{_e(e.get('tool'))}</td><td>{_e(e.get('name'))}</td><td>{_e(e.get('cleanup_note'))}</td></tr>")
        out.append("</table>")
    if report.tool_inventory:
        out.append("<h2>Tool inventory</h2>")
        for srv, tools in report.tool_inventory.items():
            out.append(f"<details><summary>{_e(srv)} · {len(tools)} tools</summary><table><tr><th>Tool</th><th>Description</th><th>Parameters</th></tr>")
            for t in tools:
                params = ", ".join((t.get("input_schema") or {}).get("properties", {}).keys())
                out.append(f"<tr><td><code>{_e(t.get('name'))}</code></td><td>{_e((t.get('description') or '')[:300])}</td><td class='muted'>{_e(params)}</td></tr>")
            out.append("</table></details>")
    if report.warnings:
        out.append("<h2>Warnings</h2><ul class='plain'>" + "".join(f"<li>{_e(w)}</li>" for w in report.warnings) + "</ul>")
    out.append(f"<p class='muted' style='margin-top:40px'>Generated by pm-evals on {_e(report.created_at.strftime('%Y-%m-%d %H:%M UTC'))}. Scores are 0-100%. A case passes when its average score meets its threshold and no safety check scores below 50%.</p>")
    out.append("</div></body></html>")
    return "".join(out)
