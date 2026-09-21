"""Run a whole dataset and produce a report."""

from __future__ import annotations

import asyncio
import datetime as dt
import time
import traceback
import uuid
from typing import Any, Awaitable, Callable, Optional

from ..checks.base import CheckContext, get_check, load_all, resolve_metrics
from ..checks.deterministic import protocol_conformance
from ..checks.safety import description_injection, rug_pull
from ..mcpio.client import MCPConnection
from ..mcpio.distractors import build_distractors
from ..mcpio.registry import ToolRegistry, close_connections, open_connections
from ..models import ActualToolCall, CaseResult, CreatedEntity, EvalCase, MCPServerConfig, MetricResult, Report, RunConfig, ServerFinding
from ..providers.base import LLMProvider
from ..providers.registry import make_provider
from ..reports.builder import build_report, render_html, render_markdown
from ..storage import Workspace
from .agent import extract_created_entities, run_agent, system_prompt_for
from .cleanup import cleanup_entities
from .harness import parse_transcript, run_claude_code

ProgressFn = Callable[[dict[str, Any]], Awaitable[None] | None]


def aggregate(metrics: list[MetricResult], threshold: float) -> tuple[float, bool]:
    scored = [m for m in metrics if not m.skipped]
    if not scored:
        return 1.0, True
    score = sum(m.score for m in scored) / len(scored)
    safety_floor = all(m.score >= 0.5 for m in scored if m.category == "safety")
    return round(score, 4), score >= threshold and safety_floor


def _servers_for_run(cfg: RunConfig, cases: list[EvalCase], ws: Workspace) -> list[MCPServerConfig]:
    if cfg.servers:
        return cfg.servers
    seen: dict[str, MCPServerConfig] = {}
    for c in cases:
        for s in c.mcp_servers:
            if s.server_name not in seen:
                # cases carry the server name + allow-list; connection details live in the workspace
                try:
                    stored = ws.load_server(s.server_name)
                    merged = stored.model_copy(update={"available_tools": s.available_tools if s.available_tools is not None else stored.available_tools})
                    seen[s.server_name] = merged
                except FileNotFoundError:
                    seen[s.server_name] = s
    return list(seen.values())


class Runner:
    def __init__(self, ws: Workspace, in_process: Optional[dict[str, Any]] = None):
        self.ws = ws
        self.in_process = in_process or {}
        load_all()

    async def _emit(self, progress: Optional[ProgressFn], run_id: str, status: dict[str, Any]) -> None:
        status["run_id"] = run_id
        status["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self.ws.save_run(run_id, status)
        if progress:
            res = progress(status)
            if asyncio.iscoroutine(res):
                await res

    async def server_checks(self, conns: list[MCPConnection], accept_baseline: bool = False) -> tuple[list[ServerFinding], dict[str, list[dict[str, Any]]]]:
        findings: list[ServerFinding] = []
        inventory: dict[str, list[dict[str, Any]]] = {}
        for conn in conns:
            tools = await conn.list_tools()
            inventory[conn.name] = [t.to_dict() for t in tools]
            findings.append(await protocol_conformance(conn, tools))
            findings.append(description_injection(conn.name, tools))
            baseline = self.ws.load_snapshot(conn.name)
            finding, snapshot = rug_pull(conn.name, tools, baseline)
            if baseline is None or accept_baseline:
                self.ws.save_snapshot(conn.name, snapshot)
            findings.append(finding)
        return findings, inventory

    async def run(self, cfg: RunConfig, progress: Optional[ProgressFn] = None, run_id: Optional[str] = None) -> Report:
        run_id = run_id or f"{cfg.dataset}_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        status: dict[str, Any] = {"state": "starting", "dataset": cfg.dataset, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "done": 0, "total": 0, "cases": [], "log": []}
        await self._emit(progress, run_id, status)
        warnings: list[str] = []
        cases = self.ws.load_cases(cfg.dataset, cfg.case_ids)
        if not cases:
            raise ValueError(f"dataset '{cfg.dataset}' has no cases")
        status["total"] = len(cases)
        servers = _servers_for_run(cfg, cases, self.ws)
        needs_connection = cfg.harness in ("api", "dry-run", "claude-code") or any(c.state_checks for c in cases) or cfg.cleanup.enabled
        conns: list[MCPConnection] = []
        server_findings: list[ServerFinding] = []
        inventory: dict[str, list[dict[str, Any]]] = {}
        provider: Optional[LLMProvider] = None
        judge: Optional[LLMProvider] = None
        results: list[CaseResult] = []
        all_entities: list[CreatedEntity] = []
        cleanup_summary: dict[str, Any] = {"enabled": cfg.cleanup.enabled, "deleted": 0, "skipped": 0, "failed": 0, "left_behind": [], "actions": [], "warnings": []}
        try:
            if servers and needs_connection:
                status["state"] = "connecting"
                status["log"].append(f"connecting to {len(servers)} MCP server(s)")
                await self._emit(progress, run_id, status)
                try:
                    conns = await open_connections(servers, self.in_process)
                except Exception as exc:
                    if cfg.harness == "transcript":
                        warnings.append(f"could not connect to MCP servers ({exc}); state checks and cleanup skipped")
                    else:
                        raise
                if conns:
                    status["state"] = "server checks"
                    await self._emit(progress, run_id, status)
                    server_findings, inventory = await self.server_checks(conns)
                    status["server_findings"] = [f.model_dump(mode="json") for f in server_findings]
                    for f in server_findings:
                        status["log"].append(f"{f.server_name} {f.metric}: {'pass' if f.passed else 'FAIL'} - {f.reason}")
            elif not servers and cfg.harness != "transcript":
                raise ValueError("No MCP server configured. Add a server in step 1 or attach one to the cases.")

            if cfg.harness == "dry-run":
                provider = make_provider("mock")
                judge = make_provider("mock")
            elif cfg.harness == "api":
                provider = make_provider(cfg.model)
            needs_judge = any(c.category == "llm_judge" or c.rubric or any(m in ("task_completion", "trajectory_quality", "llm_judge", "judge") for m in c.metrics) for c in cases)
            if judge is None and needs_judge:
                judge = make_provider(cfg.judge_model)
            judges: dict[str, LLMProvider] = {}

            status["state"] = "running cases"
            await self._emit(progress, run_id, status)
            conn_map = {c.name: c for c in conns}
            for idx, case in enumerate(cases):
                started = time.perf_counter()
                res = CaseResult(case_id=case.id, category=case.category, description=case.description, input=case.input, threshold=case.threshold, tags=case.tags)
                try:
                    case_judge = judge
                    if case.judge_model and case.judge_model != cfg.judge_model and cfg.harness != "dry-run":
                        if case.judge_model not in judges:
                            judges[case.judge_model] = make_provider(case.judge_model)
                        case_judge = judges[case.judge_model]
                    distractors = build_distractors(cfg.distractors, [t.name for c in conns for t in await c.list_tools()])
                    distractors += [d for d in case.distractor_tools if d.name not in {x.name for x in distractors}]
                    registry = await ToolRegistry.build(conns, distractors, shuffle_seed=cfg.distractors.seed + idx if distractors else None) if conns else ToolRegistry()
                    res.distractors_presented = len(registry.distractor_names())
                    calls: list[ActualToolCall] = []
                    output = ""
                    trajectory: list[str] = []
                    if cfg.harness in ("api", "dry-run"):
                        trace = await run_agent(case, cfg, provider, registry)  # type: ignore[arg-type]
                        calls, output, trajectory = trace.calls, trace.output, trace.trajectory
                        res.usage = trace.usage
                        if trace.error:
                            res.error = trace.error
                    elif cfg.harness == "claude-code":
                        parsed = await run_claude_code(case, cfg, servers, system_prompt_for(case, cfg))
                        calls, output = parsed["actual_tool_calls"], parsed["actual_output"]
                        trajectory = [f"claude-code transcript ({parsed['format']})"]
                    else:  # transcript
                        data = self.ws.load_transcript(cfg.dataset, case.id)
                        if data is None and case.actual_tool_calls:
                            calls, output = list(case.actual_tool_calls), case.actual_output or ""
                        elif data is None:
                            raise RuntimeError("no transcript uploaded for this case (Transcripts tab)")
                        else:
                            parsed = parse_transcript(data, case.input)
                            calls, output = parsed["actual_tool_calls"], parsed["actual_output"]
                            trajectory = [f"imported transcript ({parsed['format']})"]
                        if any(c.order <= 0 for c in calls) or any(b.order <= a.order for a, b in zip(calls, calls[1:])):
                            for i, c in enumerate(calls, 1):
                                c.order = i
                        for c in calls:
                            if registry.tools:
                                t = registry.get(c.name)
                                c.exists = t is not None
                                c.distractor = bool(t and t.distractor)
                                c.server = t.server if t else None
                    tool_defs = {
                        n: {"description": t.description, "input_schema": t.input_schema, "distractor": t.distractor, "server": t.server, "allowed": t.allowed, "annotations": (t.definition.annotations if t.definition else {})}
                        for n, t in registry.tools.items()
                    }
                    for c in calls:
                        res.created_entities.extend(extract_created_entities(c, cfg.cleanup))

                    async def _call(name: str, args: dict[str, Any], _reg: ToolRegistry = registry) -> Any:
                        t = _reg.get(name)
                        if t is None or t.server is None:
                            return None
                        return await conn_map[t.server].call_tool(t.definition.name if t.definition else name, args)

                    ctx = CheckContext(case=case, config=cfg, calls=calls, output=output, trajectory=trajectory, tool_defs=tool_defs, judge=case_judge, call_tool=_call if conns else None, extras={"distractors_presented": res.distractors_presented})
                    for metric in resolve_metrics(case):
                        fn = get_check(metric)
                        if fn is None:
                            warnings.append(f"{case.id}: unknown metric '{metric}' ignored")
                            continue
                        try:
                            res.metrics.append(await fn(ctx))
                        except Exception as exc:  # a broken check must not kill the run
                            res.metrics.append(MetricResult(metric=metric, category="other", score=0.0, passed=False, reason=f"check crashed: {exc}"))
                    res.actual_tool_calls, res.actual_output, res.trajectory = calls, output, trajectory
                    res.score, res.passed = aggregate(res.metrics, case.threshold)
                    if res.error:
                        res.passed = False
                except Exception as exc:
                    res.error = f"{type(exc).__name__}: {exc}"
                    res.trajectory.append(traceback.format_exc()[-1500:])
                    res.passed = False
                res.duration_ms = round((time.perf_counter() - started) * 1000)
                results.append(res)
                all_entities.extend(res.created_entities)
                status["done"] = idx + 1
                status["cases"].append({"case_id": res.case_id, "passed": res.passed, "score": res.score, "error": res.error})
                status["log"].append(f"{res.case_id}: {'pass' if res.passed else 'FAIL'} {res.score:.2f}" + (f" ({res.error})" if res.error else ""))
                await self._emit(progress, run_id, status)

            if conns and cfg.cleanup.enabled:
                status["state"] = "cleaning up"
                await self._emit(progress, run_id, status)
                cleanup_summary = await cleanup_entities(all_entities, conn_map, cfg.cleanup)
                status["log"].append(f"cleanup: deleted {cleanup_summary['deleted']}, left behind {len(cleanup_summary['left_behind'])}")
            elif all_entities and not conns:
                cleanup_summary["warnings"].append("no live MCP connection; created entities could not be cleaned up")
                cleanup_summary["left_behind"] = [e.model_dump(mode="json") for e in all_entities]
        finally:
            await close_connections(conns)
            for p in [provider, judge, *judges.values()] if "judges" in locals() else [provider, judge]:
                if p is not None:
                    try:
                        await p.aclose()
                    except Exception:
                        pass

        previous: Optional[Report] = None
        if cfg.compare_to:
            try:
                previous = self.ws.load_report(cfg.compare_to)
            except Exception as exc:
                warnings.append(f"could not load previous report '{cfg.compare_to}': {exc}")
        report = build_report(run_id, cfg, results, server_findings, cleanup_summary, inventory, warnings, previous)
        self.ws.save_report(report, html=render_html(report), markdown=render_markdown(report))
        status["state"] = "done"
        status["report_id"] = report.id
        status["summary"] = report.summary
        await self._emit(progress, run_id, status)
        return report
