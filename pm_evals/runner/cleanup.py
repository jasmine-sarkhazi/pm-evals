"""Delete entities created during a run.

Safety rules (deliberately conservative):

1. Only tools whose name starts with ``cleanup.delete_tool_prefix`` (default
   ``eval_delete``) are ever called. A plain ``delete_segment`` is never used.
2. Only entities whose recorded name carries ``cleanup.entity_prefix``
   (default ``EVAL_``) are deleted. Entities without a known name are deleted
   only if the id itself carries the prefix; otherwise they are reported as
   "left behind, needs manual review".
3. The delete tool receives the entity id in its first required parameter
   (or the first parameter ending in ``_id``), plus ``entity_prefix`` if the
   tool declares that parameter, so the server can double-check.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ..mcpio.client import MCPConnection, ToolDef
from ..models import CleanupConfig, CreatedEntity


def find_delete_tools(tools: list[ToolDef], cfg: CleanupConfig) -> list[ToolDef]:
    prefix = cfg.delete_tool_prefix.lower()
    return [t for t in tools if t.name.lower().startswith(prefix)]


def _entity_type(tool_name: str) -> str:
    base = tool_name.split("__")[-1].lower()
    base = re.sub(r"^(create|add|new|insert|register|make|build|clone|copy|upload|schedule|publish|generate)_?", "", base)
    return base.rstrip("s")


def match_delete_tool(entity: CreatedEntity, delete_tools: list[ToolDef], cfg: CleanupConfig) -> Optional[ToolDef]:
    etype = _entity_type(entity.tool)
    prefix = cfg.delete_tool_prefix.lower()
    best, best_score = None, 0
    for t in delete_tools:
        rest = t.name.lower()[len(prefix) :].strip("_")
        rest = rest.rstrip("s")
        score = 0
        if rest == etype:
            score = 3
        elif etype and (etype in rest or rest in etype):
            score = 2
        elif not etype:
            score = 1
        if score > best_score:
            best, best_score = t, score
    return best


def _id_param(tool: ToolDef) -> Optional[str]:
    schema = tool.input_schema or {}
    props = list((schema.get("properties") or {}).keys())
    required = schema.get("required") or []
    for p in required:
        if p.lower().endswith("_id") or p.lower() == "id":
            return p
    for p in props:
        if p.lower().endswith("_id") or p.lower() == "id":
            return p
    return required[0] if required else (props[0] if props else None)


async def cleanup_entities(
    entities: list[CreatedEntity],
    connections: dict[str, MCPConnection],
    cfg: CleanupConfig,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"enabled": cfg.enabled, "deleted": 0, "skipped": 0, "failed": 0, "left_behind": [], "actions": [], "warnings": []}
    if not cfg.enabled:
        summary["warnings"].append("cleanup disabled; created entities were left in place")
        summary["left_behind"] = [e.model_dump(mode="json") for e in entities]
        return summary
    if not entities:
        return summary
    if not cfg.entity_prefix.strip() or not cfg.delete_tool_prefix.strip():
        summary["warnings"].append("cleanup skipped: entity prefix and delete-tool prefix must both be set")
        summary["left_behind"] = [e.model_dump(mode="json") for e in entities]
        return summary
    tools_by_server: dict[str, list[ToolDef]] = {}
    for name, conn in connections.items():
        try:
            tools_by_server[name] = find_delete_tools(await conn.list_tools(), cfg)
        except Exception as exc:
            summary["warnings"].append(f"{name}: could not list tools for cleanup: {exc}")
            tools_by_server[name] = []
    for name, dts in tools_by_server.items():
        if not dts:
            summary["warnings"].append(
                f"server '{name}' exposes no delete tool starting with '{cfg.delete_tool_prefix}'. "
                f"Add one (for example '{cfg.delete_tool_prefix}_<entity>') that only deletes entities whose name starts with '{cfg.entity_prefix}'."
            )
    seen: set[tuple[Optional[str], str]] = set()
    for e in entities:
        key = (e.server, e.entity_id or "")
        if not e.entity_id or key in seen:
            e.deleted = None
            e.cleanup_note = "no id captured" if not e.entity_id else "duplicate"
            continue
        seen.add(key)
        prefixed = (e.name or "").startswith(cfg.entity_prefix) or (e.entity_id or "").startswith(cfg.entity_prefix)
        if not prefixed:
            e.deleted = False
            e.cleanup_note = f"name does not start with '{cfg.entity_prefix}' - not deleting (manual review)"
            summary["skipped"] += 1
            summary["left_behind"].append(e.model_dump(mode="json"))
            continue
        conn = connections.get(e.server or "")
        dts = tools_by_server.get(e.server or "", [])
        tool = match_delete_tool(e, dts, cfg) if dts else None
        if conn is None or tool is None:
            e.deleted = False
            e.cleanup_note = "no matching prefixed delete tool"
            summary["failed"] += 1
            summary["left_behind"].append(e.model_dump(mode="json"))
            continue
        param = _id_param(tool)
        if not param:
            e.deleted = False
            e.cleanup_note = f"delete tool {tool.name} has no id parameter"
            summary["failed"] += 1
            summary["left_behind"].append(e.model_dump(mode="json"))
            continue
        args: dict[str, Any] = {param: e.entity_id}
        if "entity_prefix" in (tool.input_schema.get("properties") or {}):
            args["entity_prefix"] = cfg.entity_prefix
        e.delete_tool = tool.name
        if cfg.dry_run:
            e.deleted = None
            e.cleanup_note = f"dry run: would call {tool.name}({args})"
            summary["actions"].append({"entity": e.entity_id, "tool": tool.name, "args": args, "dry_run": True})
            continue
        outcome = await conn.call_tool(tool.name, args)
        ok = not outcome.is_error
        if ok and isinstance(outcome.structured, dict) and outcome.structured.get("deleted") is False:
            ok = False
        e.deleted = ok
        e.cleanup_note = (outcome.text or "")[:200]
        summary["actions"].append({"entity": e.entity_id, "name": e.name, "tool": tool.name, "args": args, "ok": ok, "result": (outcome.text or "")[:200]})
        if ok:
            summary["deleted"] += 1
        else:
            summary["failed"] += 1
            summary["left_behind"].append(e.model_dump(mode="json"))
    return summary
