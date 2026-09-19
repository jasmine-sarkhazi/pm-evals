import asyncio
import io
import json

import httpx
import pytest

from pm_evals.importers.sheet import template_csv
from pm_evals.web.app import create_app

from .conftest import SERVER


@pytest.fixture
async def client(ws, demo):
    app = create_app(ws, in_process=demo)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_full_flow_through_api(client):
    assert (await client.get("/")).status_code == 200
    st = (await client.get("/api/status")).json()
    assert "tool_correctness" in st["metrics"]

    r = await client.post("/api/servers", json=SERVER.model_dump(mode="json"))
    assert r.status_code == 200
    t = (await client.post("/api/servers/test", json=SERVER.model_dump(mode="json"))).json()
    assert t["ok"] and t["has_delete_tools"] and len(t["findings"]) == 3

    files = {"file": ("golden.csv", template_csv().encode(), "text/csv")}
    r = await client.post("/api/datasets/import", data={"dataset": "ajo", "servers": json.dumps(["AJO-MCP"])}, files=files)
    assert r.status_code == 200, r.text
    assert r.json()["imported"] == 4
    cases = (await client.get("/api/datasets/ajo/cases")).json()
    assert cases[0]["mcp_servers"][0]["server_name"] == "AJO-MCP"

    tmpl = await client.get("/api/template.csv")
    assert tmpl.status_code == 200 and b"expected_tool" in tmpl.content
    pack = await client.get("/api/datasets/ajo/harness-pack.md")
    assert b"ajo_create_segment_001" in pack.content

    r = await client.post("/api/runs", json={"dataset": "ajo", "harness": "dry-run", "distractors": {"mode": "auto", "count": 5}})
    run_id = r.json()["run_id"]
    for _ in range(100):
        st = (await client.get(f"/api/runs/{run_id}")).json()
        if st["state"] in ("done", "failed"):
            break
        await asyncio.sleep(0.2)
    assert st["state"] == "done", st
    rep = (await client.get(f"/api/reports/{st['report_id']}")).json()
    assert rep["summary"]["cases"] == 4 and rep["cleanup"]["deleted"] == 2
    html = await client.get(f"/api/reports/{st['report_id']}/html")
    assert html.status_code == 200 and b"MCP eval report" in html.content
    md = await client.get(f"/api/reports/{st['report_id']}/md")
    assert b"## Summary" in md.content

    # compare with itself via upload
    prev = (await client.get(f"/api/reports/{st['report_id']}/json")).content
    r = await client.post(f"/api/reports/{st['report_id']}/compare", files={"file": ("prev.json", prev, "application/json")})
    assert r.status_code == 200 and r.json()["verdict"] == "no significant change"

    # transcript upload + parse
    r = await client.post("/api/datasets/ajo/transcripts/ajo_create_segment_001", data={"text": json.dumps({"actual_output": "done", "actual_tool_calls": [{"name": "create_segment", "args": {"name": "EVAL_x"}}]})})
    assert r.status_code == 200 and r.json()["format"] == "generic"
    tl = (await client.get("/api/datasets/ajo/transcripts")).json()
    assert tl["ajo_create_segment_001"]["calls"] == 1
