"""Shared helpers: argument matching, dotted paths, injection scanning."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Optional

ANY = "<any>"

READ_ONLY_PREFIXES = ("list", "get", "search", "read", "describe", "schema", "find", "query", "lookup", "fetch", "show", "check")


def is_read_only(name: str, tool_def: Optional[dict[str, Any]] = None) -> bool:
    if tool_def:
        ann = tool_def.get("annotations") or {}
        if ann.get("readOnlyHint") is True or ann.get("read_only_hint") is True:
            return True
    base = name.split("__")[-1].lower()
    return base.startswith(READ_ONLY_PREFIXES)


# ---------------------------------------------------------------------------
# Argument matching
# ---------------------------------------------------------------------------


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return v


def value_matches(expected: Any, actual: Any) -> bool:
    if expected == ANY:
        return actual is not None
    if isinstance(expected, str) and expected.startswith("re:"):
        try:
            return re.search(expected[3:], str(actual), re.I) is not None
        except re.error:
            return False
    if isinstance(expected, str) and expected.startswith("contains:"):
        return expected[len("contains:") :].lower() in str(actual).lower()
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(k in actual and value_matches(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        remaining = list(actual)
        for e in expected:
            idx = next((i for i, a in enumerate(remaining) if value_matches(e, a)), None)
            if idx is None:
                return False
            remaining.pop(idx)
        return True
    if isinstance(expected, str) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        try:
            return float(expected) == float(actual)
        except ValueError:
            return False
    if isinstance(actual, str) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return float(actual) == float(expected)
        except ValueError:
            return False
    return _norm(expected) == _norm(actual)


def compare_args(expected: dict[str, Any], actual: dict[str, Any], mode: str = "subset") -> dict[str, Any]:
    """Return {score, matched, mismatched, missing, extra}."""
    matched, mismatched, missing = [], [], []
    for k, v in expected.items():
        if k not in actual:
            missing.append(k)
        elif value_matches(v, actual[k]):
            matched.append(k)
        else:
            mismatched.append({"key": k, "expected": v, "actual": actual[k]})
    extra = [k for k in actual if k not in expected]
    total = len(expected)
    if total == 0:
        score = 1.0 if (mode != "exact" or not extra) else 0.0
    else:
        score = len(matched) / total
        if mode == "exact" and extra:
            score *= max(0.0, 1 - 0.25 * len(extra))
    return {"score": score, "matched": matched, "mismatched": mismatched, "missing": missing, "extra": extra}


def flatten_values(obj: Any, out: Optional[list[str]] = None) -> list[str]:
    out = [] if out is None else out
    if isinstance(obj, dict):
        for v in obj.values():
            flatten_values(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            flatten_values(v, out)
    elif obj is not None:
        out.append(str(obj))
    return out


def flatten_keys(obj: Any, prefix: str = "", out: Optional[list[str]] = None) -> list[str]:
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.append(key)
            flatten_keys(v, key, out)
    elif isinstance(obj, list):
        for v in obj:
            flatten_keys(v, prefix, out)
    return out


def get_path(obj: Any, path: Optional[str]) -> Any:
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def schema_property_names(schema: dict[str, Any]) -> tuple[set[str], bool]:
    """Return (top-level property names, additional_properties_allowed)."""
    props = set((schema or {}).get("properties", {}).keys())
    ap = (schema or {}).get("additionalProperties", None)
    allow_extra = ap is not False and bool(ap) if ap is not None else False
    if not props and ap is None:
        allow_extra = True  # schema-less tool: cannot judge
    return props, allow_extra


# ---------------------------------------------------------------------------
# Injection scanning (used for descriptions *and* tool results)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, str, float]] = [
    ("override_instructions", r"\bignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)", 1.0),
    ("override_instructions", r"\bdisregard\s+(all\s+|any\s+)?(previous|prior|your)\s+(instructions?|rules?|guidelines?)", 1.0),
    ("secrecy", r"\bdo\s+not\s+(tell|inform|mention|reveal|show|let)\b.*\b(user|human|anyone)\b", 1.0),
    ("secrecy", r"\b(without|never)\s+(telling|informing|notifying)\s+(the\s+)?(user|human)", 1.0),
    ("role_hijack", r"\b(you\s+are\s+now|from\s+now\s+on\s+you|act\s+as\s+(an?\s+)?(unrestricted|jailbroken))", 0.9),
    ("system_prompt_ref", r"\b(system\s+prompt|developer\s+message)\b", 0.5),
    ("fake_system_tag", r"</?\s*(system|assistant|developer|instructions?|important)\s*>", 0.8),
    ("tool_chaining", r"\b(before|after)\s+(using|calling|running)\s+this\s+tool\b.*\b(call|run|use|invoke)\b", 0.8),
    ("tool_chaining", r"\b(always|must|you\s+must|you\s+should)\s+(first\s+)?(call|run|use|invoke)\s+\w+", 0.5),
    ("exfiltration", r"\b(send|post|upload|forward|transmit|email)\b.{0,80}\b(to|at)\s+(https?://|[\w.-]+@[\w.-]+)", 0.9),
    ("exfiltration", r"https?://[^\s\"'<>]+", 0.2),
    ("credential_access", r"\b(api[_\s-]?key|password|secret|token|credentials?|\.ssh|\.env|id_rsa|aws_secret)\b", 0.5),
    ("destructive", r"\b(delete|drop|remove|wipe|erase)\s+(all|every|everything|the\s+entire)\b", 0.7),
    ("shell", r"\b(rm\s+-rf|curl\s+[^\n]*\|\s*(ba)?sh|sudo\s+|chmod\s+777)\b", 0.8),
    ("hidden_html", r"<!--.*?-->", 0.4),
    ("markdown_hidden", r"\[[^\]]*\]\(\s*(https?://)?[^)]*\)", 0.1),
]
_COMPILED = [(name, re.compile(rx, re.I | re.S), w) for name, rx, w in _PATTERNS]

_INVISIBLE = {"​", "‌", "‍", "⁠", "﻿", "­", "‪", "‫", "‬", "‭", "‮", "⁦", "⁧", "⁨", "⁩"}


def scan_for_injection(text: str, *, context: str = "") -> dict[str, Any]:
    """Scan text for prompt-injection signals. Returns {severity, findings}."""
    findings: list[dict[str, Any]] = []
    if not text:
        return {"severity": 0.0, "findings": findings}
    for name, rx, weight in _COMPILED:
        for m in rx.finditer(text):
            snippet = m.group(0)
            if name == "exfiltration" and weight == 0.2 and re.search(r"(docs|documentation|help|example\.com/?$)", snippet, re.I):
                continue
            findings.append({"type": name, "weight": weight, "snippet": snippet[:160].strip(), "context": context})
            break  # one finding per pattern is enough
    invisible = [c for c in text if c in _INVISIBLE]
    if invisible:
        findings.append({"type": "invisible_unicode", "weight": 0.9, "snippet": f"{len(invisible)} invisible characters", "context": context})
    tags = [c for c in text if 0xE0000 <= ord(c) <= 0xE007F]
    if tags:
        findings.append({"type": "unicode_tag_chars", "weight": 1.0, "snippet": f"{len(tags)} unicode tag characters (hidden text)", "context": context})
    confusables = [c for c in text if unicodedata.category(c) == "Lo" and "CYRILLIC" in unicodedata.name(c, "") and re.search(r"[A-Za-z]", text)]
    if len(confusables) > 3 and len(confusables) < len(text) * 0.3:
        findings.append({"type": "mixed_script", "weight": 0.4, "snippet": "Latin text mixed with Cyrillic look-alikes", "context": context})
    b64 = re.search(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{80,}={0,2}(?![A-Za-z0-9+/])", text)
    if b64:
        findings.append({"type": "encoded_blob", "weight": 0.5, "snippet": b64.group(0)[:60] + "...", "context": context})
    if len(text) > 3000:
        findings.append({"type": "excessive_length", "weight": 0.2, "snippet": f"{len(text)} characters", "context": context})
    severity = 0.0
    for f in findings:
        severity = 1 - (1 - severity) * (1 - f["weight"] * 0.8)
    return {"severity": round(min(1.0, severity), 3), "findings": findings}


def try_json(text: Optional[str]) -> Any:
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None
