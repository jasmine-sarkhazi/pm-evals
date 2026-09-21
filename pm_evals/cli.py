"""Command line interface.

    pm-evals serve [--port 8080]                 start the web UI
    pm-evals demo                                start the UI with the bundled demo server + dataset
    pm-evals import <dataset> <file-or-sheet-url> [--server NAME ...]
    pm-evals servers add <name> --transport streamable-http --url URL [--header K=V ...]
    pm-evals servers test <name>
    pm-evals run <dataset> [--model claude-opus-5] [--judge ...] [--harness api|dry-run|transcript|claude-code]
                           [--distractors none|auto|manual|both] [--distractor NAME ...] [--compare REPORT_ID]
    pm-evals report <report_id> [--format html|md|json]
    pm-evals compare <report_id> <previous_report_id_or_path>
    pm-evals template > golden.csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Optional

from .models import DistractorConfig, DistractorTool, MCPServerConfig, RunConfig
from .storage import Workspace


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .web.app import create_app

    ws = Workspace(args.workspace)
    in_process = None
    if getattr(args, "demo", False):
        from .mcpio.demo_server import build_server

        in_process = {"AJO-MCP": build_server()}
        _seed_demo(ws)
    app = create_app(ws, in_process=in_process)
    print(f"pm-evals UI: http://{args.host}:{args.port}   (workspace: {ws.root})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def _seed_demo(ws: Workspace) -> None:
    from .importers.sheet import parse_csv_bytes, rows_to_cases, template_csv

    ws.save_server(MCPServerConfig(server_name="AJO-MCP", transport="stdio", command=sys.executable, args=["-m", "pm_evals.mcpio.demo_server"], available_tools=None))
    if not any(d["name"] == "demo" for d in ws.list_datasets()):
        cases, _ = rows_to_cases(parse_csv_bytes(template_csv().encode()), [MCPServerConfig(server_name="AJO-MCP", transport="stdio")])
        ws.save_cases("demo", cases, {"source": "bundled template", "servers": ["AJO-MCP"]})
    print("Demo ready: server 'AJO-MCP' (in-process) and dataset 'demo'. Use harness 'dry-run' to run without API keys.")


def cmd_template(args: argparse.Namespace) -> None:
    from .importers.sheet import template_csv

    sys.stdout.write(template_csv())


def cmd_import(args: argparse.Namespace) -> None:
    from .importers.sheet import fetch_google_sheet, parse_upload, rows_to_cases

    ws = Workspace(args.workspace)
    if args.source.startswith("http"):
        rows = asyncio.run(fetch_google_sheet(args.source))
    else:
        with open(args.source, "rb") as fh:
            rows = parse_upload(args.source, fh.read())
    servers = [MCPServerConfig(server_name=n) for n in (args.server or [])]
    cases, warnings = rows_to_cases(rows, servers)
    n = ws.save_cases(args.dataset, cases, {"source": args.source, "servers": args.server or []})
    print(f"imported {n} cases into dataset '{args.dataset}'")
    for w in warnings:
        print(f"  warning: {w}")
    if args.show:
        for c in cases:
            _print(c.model_dump(mode="json"))


def cmd_servers(args: argparse.Namespace) -> None:
    ws = Workspace(args.workspace)
    if args.servers_cmd == "list":
        _print([s.model_dump(mode="json") for s in ws.list_servers()])
    elif args.servers_cmd == "add":
        headers = dict(h.split("=", 1) for h in (args.header or []))
        env = dict(e.split("=", 1) for e in (args.env or []))
        cfg = MCPServerConfig(
            server_name=args.name, transport=args.transport, url=args.url, headers=headers, command=args.command, args=args.args or [], env=env,
            available_tools=args.tools or None,
        )
        ws.save_server(cfg)
        print(f"saved server '{args.name}'")
    elif args.servers_cmd == "test":
        from .checks.deterministic import protocol_conformance
        from .checks.safety import description_injection, rug_pull
        from .mcpio.client import MCPConnection

        cfg = ws.load_server(args.name)

        async def go() -> None:
            async with MCPConnection(cfg) as conn:
                tools = await conn.list_tools()
                print(f"connected: {conn.handshake.get('server_info', {}).get('name')} protocol {conn.handshake.get('protocol_version')} ({len(tools)} tools)")
                for t in tools:
                    print(f"  - {t.name}: {t.description[:90]}")
                for f in (await protocol_conformance(conn, tools), description_injection(cfg.server_name, tools), rug_pull(cfg.server_name, tools, ws.load_snapshot(cfg.server_name))[0]):
                    print(f"{'PASS' if f.passed else 'FAIL'} {f.metric} ({f.score:.2f}): {f.reason}")

        asyncio.run(go())
    elif args.servers_cmd == "remove":
        print("removed" if ws.delete_server(args.name) else "not found")


def cmd_run(args: argparse.Namespace) -> None:
    from .runner.orchestrator import Runner

    ws = Workspace(args.workspace)
    distractors = DistractorConfig(mode=args.distractors, count=args.distractor_count, tools=[DistractorTool(name=n) for n in (args.distractor or [])])
    cfg = RunConfig(
        dataset=args.dataset, model=args.model, judge_model=args.judge or args.model, harness=args.harness, distractors=distractors,
        case_ids=args.case or [], label=args.label, compare_to=args.compare, max_turns=args.max_turns,
    )
    cfg.cleanup.enabled = not args.no_cleanup
    cfg.cleanup.entity_prefix = args.entity_prefix
    cfg.cleanup.delete_tool_prefix = args.delete_tool_prefix
    if args.server:
        cfg.servers = [ws.load_server(n) for n in args.server]
    in_process = None
    if args.demo:
        from .mcpio.demo_server import build_server

        in_process = {"AJO-MCP": build_server()}
        _seed_demo(ws)

    def progress(st: dict[str, Any]) -> None:
        if st.get("log"):
            print(f"[{st['state']}] {st['log'][-1]}")

    report = asyncio.run(Runner(ws, in_process=in_process).run(cfg, progress))
    s = report.summary
    print(f"\nReport {report.id} ({report.date}): {s['passed']}/{s['cases']} passed, avg score {s['avg_score']:.2f}")
    print(f"  html: {ws.report_path(report.id, 'html')}\n  md:   {ws.report_path(report.id, 'md')}")
    if report.comparison:
        c = report.comparison
        print(f"  vs {c['previous_report_id']}: {c['verdict']} ({len(c['regressions'])} regressions, {len(c['improvements'])} improvements)")


def cmd_report(args: argparse.Namespace) -> None:
    from .reports.builder import render_html, render_markdown

    ws = Workspace(args.workspace)
    if args.report_id == "list":
        _print(ws.list_reports())
        return
    rep = ws.load_report(args.report_id)
    if args.format == "json":
        print(rep.model_dump_json(indent=2))
    elif args.format == "html":
        print(render_html(rep))
    else:
        print(render_markdown(rep))


def cmd_compare(args: argparse.Namespace) -> None:
    from .reports.builder import render_html, render_markdown
    from .reports.compare import compare_reports

    ws = Workspace(args.workspace)
    cur, prev = ws.load_report(args.report_id), ws.load_report(args.previous)
    cur.comparison = compare_reports(cur, prev)
    cur.config.compare_to = prev.id
    ws.save_report(cur, html=render_html(cur), markdown=render_markdown(cur))
    c = cur.comparison
    print(f"{c['verdict']}: pass rate {c['pass_rate']['before']} -> {c['pass_rate']['after']}")
    for e in c["regressions"]:
        print(f"  REGRESSION {e['case_id']}: {e['before']} -> {e['after']}")
    for e in c["improvements"]:
        print(f"  improved   {e['case_id']}: {e['before']} -> {e['after']}")
    print(f"updated report: {ws.report_path(cur.id, 'html')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pm-evals", description="Eval platform for MCP server tools and skills")
    p.add_argument("--workspace", default=None, help="workspace directory (default: $PM_EVALS_HOME or ./workspace)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(fn=cmd_serve, demo=False)

    d = sub.add_parser("demo", help="start the web UI with the bundled demo MCP server and dataset")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8080)
    d.set_defaults(fn=cmd_serve, demo=True)

    t = sub.add_parser("template", help="print the golden dataset CSV template")
    t.set_defaults(fn=cmd_template)

    i = sub.add_parser("import", help="import a CSV/XLSX/JSON file or Google Sheets link as a dataset")
    i.add_argument("dataset")
    i.add_argument("source")
    i.add_argument("--server", action="append", help="MCP server name(s) to attach to every case")
    i.add_argument("--show", action="store_true")
    i.set_defaults(fn=cmd_import)

    sv = sub.add_parser("servers", help="manage MCP servers")
    svs = sv.add_subparsers(dest="servers_cmd", required=True)
    svs.add_parser("list")
    a = svs.add_parser("add")
    a.add_argument("name")
    a.add_argument("--transport", default="streamable-http", choices=["streamable-http", "sse", "stdio"])
    a.add_argument("--url")
    a.add_argument("--header", action="append", help="K=V")
    a.add_argument("--command")
    a.add_argument("--args", nargs="*")
    a.add_argument("--env", action="append", help="K=V")
    a.add_argument("--tools", nargs="*", help="allow-list of tools the agent may use")
    te = svs.add_parser("test")
    te.add_argument("name")
    rm = svs.add_parser("remove")
    rm.add_argument("name")
    sv.set_defaults(fn=cmd_servers)

    r = sub.add_parser("run", help="run a dataset and write a report")
    r.add_argument("dataset")
    r.add_argument("--model", default="claude-opus-5")
    r.add_argument("--judge", default=None, help="judge model (default: same as --model); use 'jev' for the TypeSafe System One judge, 'jev:mock' offline")
    r.add_argument("--harness", default="api", choices=["api", "dry-run", "transcript", "claude-code"])
    r.add_argument("--server", action="append")
    r.add_argument("--case", action="append")
    r.add_argument("--distractors", default="none", choices=["none", "auto", "manual", "both"])
    r.add_argument("--distractor", action="append", help="extra tool name to present as a distractor")
    r.add_argument("--distractor-count", type=int, default=20)
    r.add_argument("--no-cleanup", action="store_true")
    r.add_argument("--entity-prefix", default="EVAL_")
    r.add_argument("--delete-tool-prefix", default="eval_delete")
    r.add_argument("--max-turns", type=int, default=12)
    r.add_argument("--label")
    r.add_argument("--compare", help="previous report id to compare against")
    r.add_argument("--demo", action="store_true", help="use the bundled in-process demo server")
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("report", help="print a report (or 'list')")
    rp.add_argument("report_id")
    rp.add_argument("--format", default="md", choices=["md", "html", "json"])
    rp.set_defaults(fn=cmd_report)

    c = sub.add_parser("compare", help="compare a report with a previous one")
    c.add_argument("report_id")
    c.add_argument("previous")
    c.set_defaults(fn=cmd_compare)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
