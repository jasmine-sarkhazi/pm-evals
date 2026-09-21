"""Filesystem-backed workspace.

Layout (``PM_EVALS_HOME`` or ``./workspace``)::

    datasets/<dataset>/cases/<case_id>.json
    datasets/<dataset>/meta.json
    servers/<server_name>.json
    snapshots/<server_name>.json         # rug-pull baselines
    runs/<run_id>.json                   # live run status
    reports/<report_id>.json|.html|.md
    transcripts/<dataset>/<case_id>.json # imported harness transcripts
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from .models import EvalCase, MCPServerConfig, Report

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def slug(value: str) -> str:
    """Filesystem-safe name. Never yields '', '.' or '..' (or anything that
    would escape the workspace)."""
    value = _SAFE.sub("_", (value or "").strip())
    value = value.strip("_.")
    if not value or set(value) <= {".", "_", "-"}:
        return "item"
    return value


class Workspace:
    def __init__(self, root: Optional[str | Path] = None):
        self.root = Path(root or os.environ.get("PM_EVALS_HOME") or "workspace").resolve()
        for sub in ("datasets", "servers", "snapshots", "runs", "reports", "transcripts"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # -- generic helpers ---------------------------------------------------
    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
        tmp.replace(path)

    # -- datasets ----------------------------------------------------------
    def _inside(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError(f"refusing path outside the workspace: {path}")
        return resolved

    def dataset_dir(self, dataset: str) -> Path:
        return self._inside(self.root / "datasets" / slug(dataset))

    def list_datasets(self) -> list[dict[str, Any]]:
        out = []
        for d in sorted((self.root / "datasets").iterdir()):
            if d.is_dir():
                cases = list((d / "cases").glob("*.json")) if (d / "cases").exists() else []
                meta = self._read_json(d / "meta.json") if (d / "meta.json").exists() else {}
                out.append({"name": d.name, "cases": len(cases), **meta})
        return out

    def save_cases(self, dataset: str, cases: Iterable[EvalCase], meta: Optional[dict[str, Any]] = None) -> int:
        n = 0
        for case in cases:
            self._write_json(self.dataset_dir(dataset) / "cases" / f"{slug(case.id)}.json", case.model_dump(mode="json"))
            n += 1
        if meta is not None:
            existing = {}
            mp = self.dataset_dir(dataset) / "meta.json"
            if mp.exists():
                existing = self._read_json(mp)
            existing.update(meta)
            self._write_json(mp, existing)
        return n

    def load_cases(self, dataset: str, case_ids: Optional[list[str]] = None) -> list[EvalCase]:
        ddir = self.dataset_dir(dataset)
        cdir = ddir / "cases"
        if not ddir.exists():
            raise FileNotFoundError(f"dataset '{dataset}' not found in {self.root}")
        cases = [EvalCase.model_validate(self._read_json(p)) for p in sorted(cdir.glob("*.json"))] if cdir.exists() else []
        if case_ids:
            wanted = set(case_ids)
            cases = [c for c in cases if c.id in wanted]
        return cases

    def delete_case(self, dataset: str, case_id: str) -> bool:
        p = self.dataset_dir(dataset) / "cases" / f"{slug(case_id)}.json"
        if p.exists():
            p.unlink()
            return True
        return False

    def delete_dataset(self, dataset: str) -> bool:
        import shutil

        d = self.dataset_dir(dataset)
        if d.exists():
            shutil.rmtree(d)
            return True
        return False

    def dataset_meta(self, dataset: str) -> dict[str, Any]:
        mp = self.dataset_dir(dataset) / "meta.json"
        return self._read_json(mp) if mp.exists() else {}

    # -- servers -----------------------------------------------------------
    def save_server(self, cfg: MCPServerConfig) -> None:
        self._write_json(self.root / "servers" / f"{slug(cfg.server_name)}.json", cfg.model_dump(mode="json"))

    def load_server(self, name: str) -> MCPServerConfig:
        return MCPServerConfig.model_validate(self._read_json(self.root / "servers" / f"{slug(name)}.json"))

    def list_servers(self) -> list[MCPServerConfig]:
        return [MCPServerConfig.model_validate(self._read_json(p)) for p in sorted((self.root / "servers").glob("*.json"))]

    def delete_server(self, name: str) -> bool:
        p = self.root / "servers" / f"{slug(name)}.json"
        if p.exists():
            p.unlink()
            return True
        return False

    # -- snapshots (rug-pull baselines) ------------------------------------
    def snapshot_path(self, server_name: str) -> Path:
        return self.root / "snapshots" / f"{slug(server_name)}.json"

    def load_snapshot(self, server_name: str) -> Optional[dict[str, Any]]:
        p = self.snapshot_path(server_name)
        return self._read_json(p) if p.exists() else None

    def save_snapshot(self, server_name: str, snapshot: dict[str, Any]) -> None:
        self._write_json(self.snapshot_path(server_name), snapshot)

    # -- runs --------------------------------------------------------------
    def save_run(self, run_id: str, status: dict[str, Any]) -> None:
        self._write_json(self.root / "runs" / f"{slug(run_id)}.json", status)

    def load_run(self, run_id: str) -> Optional[dict[str, Any]]:
        p = self.root / "runs" / f"{slug(run_id)}.json"
        return self._read_json(p) if p.exists() else None

    def list_runs(self) -> list[dict[str, Any]]:
        runs = [self._read_json(p) for p in (self.root / "runs").glob("*.json")]
        runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        return runs

    # -- reports -----------------------------------------------------------
    def report_path(self, report_id: str, ext: str = "json") -> Path:
        return self.root / "reports" / f"{slug(report_id)}.{ext}"

    def save_report(self, report: Report, html: Optional[str] = None, markdown: Optional[str] = None) -> Path:
        path = self.report_path(report.id)
        self._write_json(path, report.model_dump(mode="json"))
        if html is not None:
            self.report_path(report.id, "html").write_text(html, encoding="utf-8")
        if markdown is not None:
            self.report_path(report.id, "md").write_text(markdown, encoding="utf-8")
        return path

    def load_report(self, report_id_or_path: str) -> Report:
        p = Path(report_id_or_path)
        if not p.exists():
            p = self.report_path(report_id_or_path)
        return Report.model_validate(self._read_json(p))

    def list_reports(self) -> list[dict[str, Any]]:
        out = []
        for p in (self.root / "reports").glob("*.json"):
            try:
                data = self._read_json(p)
            except Exception:
                continue
            out.append(
                {
                    "id": data.get("id"),
                    "date": data.get("date"),
                    "created_at": data.get("created_at"),
                    "dataset": data.get("dataset"),
                    "label": data.get("label"),
                    "model": (data.get("config") or {}).get("model"),
                    "pass_rate": (data.get("summary") or {}).get("pass_rate"),
                    "cases": len(data.get("results") or []),
                }
            )
        out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return out

    # -- transcripts -------------------------------------------------------
    def transcript_dir(self, dataset: str) -> Path:
        d = self._inside(self.root / "transcripts" / slug(dataset))
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_transcript(self, dataset: str, case_id: str, data: dict[str, Any]) -> Path:
        p = self.transcript_dir(dataset) / f"{slug(case_id)}.json"
        self._write_json(p, data)
        return p

    def load_transcript(self, dataset: str, case_id: str) -> Optional[dict[str, Any]]:
        p = self.transcript_dir(dataset) / f"{slug(case_id)}.json"
        return self._read_json(p) if p.exists() else None
