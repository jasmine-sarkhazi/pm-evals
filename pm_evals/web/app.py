"""FastAPI application: JSON API + the single-page PM UI."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from .. import __version__
from ..checks.base import ALIASES, DEFAULT_METRICS_BY_CATEGORY, METRIC_HELP, SERVER_LEVEL_METRICS, load_all
from ..checks.deterministic import protocol_conformance
from ..checks.judge import suggest_rubric
from ..checks.safety import description_injection, rug_pull
from ..importers.sheet import TEMPLATE_COLUMNS, fetch_google_sheet, parse_upload, rows_to_cases_or_json, template_csv
from ..mcpio.client import MCPConnection
from ..models import EvalCase, MCPServerConfig, Report, RunConfig
from ..providers.registry import configured_providers, is_system_one_model, make_provider
from ..reports.builder import render_html, render_markdown
from ..reports.compare import compare_reports
from ..runner.harness import harness_pack, parse_transcript
from ..runner.orchestrator import Runner
from ..storage import Workspace

STATIC = Path(__file__).parent / "static"


def create_app(workspace: Optional[Workspace] = None, in_process: Optional[dict[str, Any]] = None) -> FastAPI:
    ws = workspace or Workspace()
    app = FastAPI(title="pm-evals", version=__version__)
    load_all()

    @app.exception_handler(FileNotFoundError)
    async def _not_found(_request: Any, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"not found: {exc}"})

    @app.exception_handler(ValueError)
    async def _bad_request(_request: Any, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    runs: dict[str, dict[str, Any]] = {}
    tasks: dict[str, asyncio.Task[Any]] = {}

    def _err(exc: Exception, code: int = 400) -> HTTPException:
        return HTTPException(status_code=code, detail=f"{type(exc).__name__}: {exc}")

    # -- static -------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def index() -> Any:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return {
            "version": __version__,
            "workspace": str(ws.root),
            "providers": configured_providers(),
            "metrics": METRIC_HELP,
            "aliases": ALIASES,
            "defaults": DEFAULT_METRICS_BY_CATEGORY,
            "server_metrics": list(SERVER_LEVEL_METRICS),
            "claude_cli": bool(__import__("shutil").which("claude")),
        }

    # -- servers ------------------------------------------------------------
    @app.get("/api/servers")
    async def list_servers() -> list[dict[str, Any]]:
        return [s.model_dump(mode="json") for s in ws.list_servers()]

    @app.post("/api/servers")
    async def save_server(cfg: MCPServerConfig) -> dict[str, Any]:
        ws.save_server(cfg)
        return cfg.model_dump(mode="json")

    @app.delete("/api/servers/{name}")
    async def delete_server(name: str) -> dict[str, Any]:
        return {"deleted": ws.delete_server(name)}

    @app.post("/api/servers/test")
    async def test_server(cfg: MCPServerConfig) -> dict[str, Any]:
        try:
            async with MCPConnection(cfg, in_process_server=(in_process or {}).get(cfg.server_name)) as conn:
                tools = await conn.list_tools()
                conformance = await protocol_conformance(conn, tools)
                injection = description_injection(cfg.server_name, tools)
                baseline = ws.load_snapshot(cfg.server_name)
                rug, snapshot = rug_pull(cfg.server_name, tools, baseline)
                if baseline is None:
                    ws.save_snapshot(cfg.server_name, snapshot)
                delete_prefix = "eval_delete"
                delete_tools = [t.name for t in tools if t.name.lower().startswith(delete_prefix)]
                return {
                    "ok": True,
                    "handshake": conn.handshake,
                    "tools": [t.to_dict() for t in tools],
                    "findings": [f.model_dump(mode="json") for f in (conformance, injection, rug)],
                    "delete_tools": delete_tools,
                    "has_delete_tools": bool(delete_tools),
                }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    @app.post("/api/servers/{name}/baseline")
    async def accept_baseline(name: str) -> dict[str, Any]:
        cfg = ws.load_server(name)
        async with MCPConnection(cfg, in_process_server=(in_process or {}).get(name)) as conn:
            tools = await conn.list_tools()
            _, snapshot = rug_pull(name, tools, None)
            ws.save_snapshot(name, snapshot)
            return {"ok": True, "tools": len(tools), "captured_at": snapshot["captured_at"]}

    # -- datasets -----------------------------------------------------------
    @app.get("/api/template.csv")
    async def template() -> Response:
        return Response(template_csv(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=pm-evals-golden-dataset-template.csv"})

    @app.get("/api/template/columns")
    async def template_columns() -> dict[str, Any]:
        return {"columns": TEMPLATE_COLUMNS}

    @app.get("/api/datasets")
    async def list_datasets() -> list[dict[str, Any]]:
        return ws.list_datasets()

    @app.post("/api/datasets/import")
    async def import_dataset(
        dataset: str = Form(...),
        servers: str = Form("[]"),
        sheet_url: str = Form(""),
        default_category: str = Form("deterministic"),
        replace: bool = Form(False),
        file: Optional[UploadFile] = File(None),
    ) -> dict[str, Any]:
        try:
            server_names = json.loads(servers or "[]")
            server_cfgs = []
            for n in server_names:
                try:
                    stored = ws.load_server(n)
                    server_cfgs.append(MCPServerConfig(server_name=n, transport=stored.transport, available_tools=stored.available_tools))
                except FileNotFoundError:
                    server_cfgs.append(MCPServerConfig(server_name=n))
            if file is not None and file.filename:
                rows = parse_upload(file.filename, await file.read())
                source = file.filename
            elif sheet_url.strip():
                rows = await fetch_google_sheet(sheet_url.strip())
                source = sheet_url.strip()
            else:
                raise ValueError("Upload a CSV/XLSX/JSON file or paste a Google Sheets link.")
            cases, warnings = rows_to_cases_or_json(rows, server_cfgs, default_category)
            if not cases:
                raise ValueError("No cases found in the sheet. " + "; ".join(warnings or ["Check that it has an 'input' (prompt) column and at least one row."]))
            if replace:
                ws.delete_dataset(dataset)
            n = ws.save_cases(dataset, cases, {"source": source, "imported_at": dt.datetime.now(dt.timezone.utc).isoformat(), "servers": server_names})
            return {"dataset": dataset, "imported": n, "warnings": warnings, "cases": [c.model_dump(mode="json") for c in cases]}
        except HTTPException:
            raise
        except Exception as exc:
            raise _err(exc)

    @app.get("/api/datasets/{name}/cases")
    async def get_cases(name: str) -> list[dict[str, Any]]:
        try:
            return [c.model_dump(mode="json") for c in ws.load_cases(name)]
        except FileNotFoundError as exc:
            raise _err(exc, 404)

    @app.put("/api/datasets/{name}/cases/{case_id}")
    async def put_case(name: str, case_id: str, case: EvalCase) -> dict[str, Any]:
        if case.id != case_id:
            ws.delete_case(name, case_id)
        ws.save_cases(name, [case])
        return case.model_dump(mode="json")

    @app.post("/api/datasets/{name}/cases")
    async def add_case(name: str, case: EvalCase) -> dict[str, Any]:
        ws.save_cases(name, [case])
        return case.model_dump(mode="json")

    @app.delete("/api/datasets/{name}/cases/{case_id}")
    async def delete_case(name: str, case_id: str) -> dict[str, Any]:
        return {"deleted": ws.delete_case(name, case_id)}

    @app.delete("/api/datasets/{name}")
    async def delete_dataset(name: str) -> dict[str, Any]:
        return {"deleted": ws.delete_dataset(name)}

    @app.get("/api/datasets/{name}/export.json")
    async def export_dataset(name: str) -> Response:
        cases = [c.model_dump(mode="json") for c in ws.load_cases(name)]
        return Response(json.dumps(cases, indent=2), media_type="application/json", headers={"Content-Disposition": f"attachment; filename={name}-cases.json"})

    class RubricRequest(BaseModel):
        dataset: str
        case_id: str
        judge_model: str = "claude-opus-5"

    @app.post("/api/rubric/suggest")
    async def rubric_suggest(req: RubricRequest) -> dict[str, Any]:
        cases = {c.id: c for c in ws.load_cases(req.dataset)}
        case = cases.get(req.case_id)
        if case is None:
            raise HTTPException(404, "case not found")
        if is_system_one_model(req.judge_model):
            raise HTTPException(
                422,
                "TypeSafe System One (Jev) returns typed judgments and does not draft rubrics. "
                "Draft with an LLM judge model (e.g. claude-opus-5), then run scoring with Jev.",
            )
        tool_names: list[str] = []
        for s in case.mcp_servers:
            snap = ws.load_snapshot(s.server_name)
            if snap:
                tool_names += list(snap.get("tools", {}).keys())
        judge = make_provider(req.judge_model)
        try:
            items = await suggest_rubric(judge, case, tool_names)
        except Exception as exc:
            raise _err(exc)
        finally:
            await judge.aclose()
        return {"rubric": [i.model_dump(mode="json") for i in items]}

    # -- transcripts (other harnesses) -------------------------------------
    @app.get("/api/datasets/{name}/harness-pack.md")
    async def get_harness_pack(name: str, entity_prefix: str = "EVAL_") -> Response:
        cases = ws.load_cases(name)
        cfg = RunConfig(dataset=name)
        cfg.cleanup.entity_prefix = entity_prefix
        return PlainTextResponse(harness_pack(cases, cfg), headers={"Content-Disposition": f"attachment; filename={name}-harness-pack.md"})

    @app.get("/api/datasets/{name}/transcripts")
    async def list_transcripts(name: str) -> dict[str, Any]:
        out = {}
        for c in ws.load_cases(name):
            t = ws.load_transcript(name, c.id)
            out[c.id] = None if t is None else {"uploaded": True, "format": t.get("_format"), "calls": t.get("_calls"), "output": (t.get("_output") or "")[:120]}
        return out

    @app.post("/api/datasets/{name}/transcripts/{case_id}")
    async def upload_transcript(name: str, case_id: str, file: Optional[UploadFile] = File(None), text: str = Form("")) -> dict[str, Any]:
        try:
            raw = (await file.read()).decode("utf-8") if file is not None and file.filename else text
            if not raw.strip():
                raise ValueError("empty transcript")
            data = json.loads(raw)
            cases = {c.id: c for c in ws.load_cases(name)}
            case = cases.get(case_id)
            parsed = parse_transcript(data, case.input if case else None)
            stored = data if isinstance(data, dict) else {"messages": data}
            if isinstance(data, list) and data and isinstance(data[0], dict) and ("chat_messages" in data[0] or "mapping" in data[0] or data[0].get("type") in ("system", "assistant", "user", "result")):
                stored = {"export": data}
            stored = dict(stored)
            stored["_format"] = parsed["format"]
            stored["_calls"] = len(parsed["actual_tool_calls"])
            stored["_output"] = parsed["actual_output"]
            ws.save_transcript(name, case_id, stored)
            return {"ok": True, "format": parsed["format"], "tool_calls": [c.model_dump(mode="json") for c in parsed["actual_tool_calls"]], "actual_output": parsed["actual_output"]}
        except Exception as exc:
            raise _err(exc)

    # -- runs ---------------------------------------------------------------
    @app.post("/api/runs")
    async def start_run(cfg: RunConfig) -> dict[str, Any]:
        try:
            ws.load_cases(cfg.dataset)
        except FileNotFoundError as exc:
            raise _err(exc, 404)
        run_id = f"{cfg.dataset}_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        runs[run_id] = {"state": "queued", "run_id": run_id, "dataset": cfg.dataset, "log": [], "done": 0, "total": 0, "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        runner = Runner(ws, in_process=in_process)

        async def progress(st: dict[str, Any]) -> None:
            runs[run_id] = dict(st)

        async def go() -> None:
            try:
                await runner.run(cfg, progress, run_id=run_id)
            except Exception as exc:
                st = dict(runs.get(run_id, {}))
                st.update({"state": "failed", "error": f"{type(exc).__name__}: {exc}", "run_id": run_id})
                st.setdefault("log", []).append(st["error"])
                runs[run_id] = st
                ws.save_run(run_id, st)

        task = asyncio.create_task(go())
        tasks[run_id] = task
        task.add_done_callback(lambda _t, rid=run_id: tasks.pop(rid, None))
        return {"run_id": run_id}

    @app.get("/api/runs")
    async def list_runs() -> list[dict[str, Any]]:
        merged = {r["run_id"]: r for r in ws.list_runs() if r.get("run_id")}
        merged.update(runs)
        out = sorted(merged.values(), key=lambda r: r.get("started_at") or "", reverse=True)
        return [{k: v for k, v in r.items() if k != "log"} for r in out[:50]]

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        st = runs.get(run_id) or ws.load_run(run_id)
        if st is None:
            raise HTTPException(404, "run not found")
        return st

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> dict[str, Any]:
        t = tasks.get(run_id)
        if t and not t.done():
            t.cancel()
            st = dict(runs.get(run_id, {}))
            st.update({"state": "cancelled"})
            runs[run_id] = st
            ws.save_run(run_id, st)
            return {"cancelled": True}
        return {"cancelled": False}

    # -- reports ------------------------------------------------------------
    @app.get("/api/reports")
    async def list_reports() -> list[dict[str, Any]]:
        return ws.list_reports()

    @app.get("/api/reports/{report_id}")
    async def get_report(report_id: str) -> dict[str, Any]:
        try:
            return ws.load_report(report_id).model_dump(mode="json")
        except Exception as exc:
            raise _err(exc, 404)

    @app.get("/api/reports/{report_id}/html", response_class=HTMLResponse)
    async def report_html(report_id: str, download: bool = False) -> Any:
        p = ws.report_path(report_id, "html")
        if not p.exists():
            rep = ws.load_report(report_id)
            p.write_text(render_html(rep), encoding="utf-8")
        headers = {"Content-Disposition": f"attachment; filename=mcp-eval-report-{report_id}.html"} if download else {}
        return HTMLResponse(p.read_text(encoding="utf-8"), headers=headers)

    @app.get("/api/reports/{report_id}/md")
    async def report_md(report_id: str) -> Response:
        p = ws.report_path(report_id, "md")
        text = p.read_text(encoding="utf-8") if p.exists() else render_markdown(ws.load_report(report_id))
        return PlainTextResponse(text, headers={"Content-Disposition": f"attachment; filename=mcp-eval-report-{report_id}.md"})

    @app.get("/api/reports/{report_id}/json")
    async def report_json_download(report_id: str) -> Response:
        rep = ws.load_report(report_id)
        return Response(rep.model_dump_json(indent=2), media_type="application/json", headers={"Content-Disposition": f"attachment; filename=mcp-eval-report-{report_id}.json"})

    @app.post("/api/reports/{report_id}/compare")
    async def compare(report_id: str, previous_id: str = Form(""), file: Optional[UploadFile] = File(None)) -> dict[str, Any]:
        try:
            current = ws.load_report(report_id)
            if file is not None and file.filename:
                previous = Report.model_validate(json.loads((await file.read()).decode("utf-8")))
            elif previous_id:
                previous = ws.load_report(previous_id)
            else:
                raise ValueError("Pick a previous report or upload one.")
            current.comparison = compare_reports(current, previous)
            current.config.compare_to = previous.id
            ws.save_report(current, html=render_html(current), markdown=render_markdown(current))
            return current.comparison
        except HTTPException:
            raise
        except Exception as exc:
            raise _err(exc)

    @app.delete("/api/reports/{report_id}")
    async def delete_report(report_id: str) -> dict[str, Any]:
        n = 0
        for ext in ("json", "html", "md"):
            p = ws.report_path(report_id, ext)
            if p.exists():
                p.unlink()
                n += 1
        return {"deleted": n > 0}

    return app


app = create_app()
