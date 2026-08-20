from __future__ import annotations

"""Lightweight page-run provenance shared by every transfer route.

Mode renderers remain independent.  The lifecycle writes one common receipt only
*after* a renderer returns successfully.  It records which route ran and which
mode-owned files exist so stale/cross-mode state is diagnosable without coupling
pixel code together.
"""

from pathlib import Path
from typing import Any

from .io_utils import save_json
from .mode_contracts import artifact_ownership_snapshot

_SCHEMA = "folirina.page_run_receipt.v1"
_COMMON_OUTPUTS = (
    "final.png", "final_reviewed.png", "review_preview.png", "text_layer.png",
    "chinese_transfer_layer.png", "target_clear_mask.png", "clear_mask.png",
)


def write_run_receipt(page_root: str | Path, project: Any, *, requested_mode: str, selected_strategy: str, run_id: str) -> dict:
    root = Path(page_root)
    meta = getattr(project, "meta", {}) or {}
    artifacts = getattr(project, "artifacts", {}) or {}
    names: set[str] = set()
    for value in artifacts.values():
        try:
            p = Path(str(value))
        except Exception:
            continue
        if p.parent == root and p.exists() and p.is_file():
            names.add(p.name)
    for name in _COMMON_OUTPUTS:
        if (root / name).is_file():
            names.add(name)
    rows = []
    for name in sorted(names):
        p = root / name
        try:
            st = p.stat()
            rows.append({"name": name, "bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
        except OSError:
            continue
    payload = {
        "schema": _SCHEMA,
        "run_id": str(run_id),
        "requested_mode": str(requested_mode or "").strip().lower(),
        "project_mode": str(meta.get("transfer_mode") or "").strip().lower(),
        "selected_strategy": str(selected_strategy or "").strip().lower(),
        "mode_execution": dict(meta.get("mode_execution") or {}),
        "artifact_owners": artifact_ownership_snapshot(root),
        "artifacts": rows,
    }
    save_json(root / "run_receipt.json", payload)
    return payload


def validate_run_receipt(page_root: str | Path, *, requested_mode: str, selected_strategy: str = "") -> list[dict]:
    import json
    root = Path(page_root)
    p = root / "run_receipt.json"
    if not p.exists():
        return [{"code": "missing_run_receipt", "path": "run_receipt.json"}]
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return [{"code": "invalid_run_receipt", "path": "run_receipt.json"}]
    issues: list[dict] = []
    if str(obj.get("schema") or "") != _SCHEMA:
        issues.append({"code": "run_receipt_schema", "value": obj.get("schema")})
    requested = str(requested_mode or "").strip().lower()
    if str(obj.get("requested_mode") or "").strip().lower() != requested:
        issues.append({"code": "run_receipt_mode_mismatch", "expected": requested, "actual": obj.get("requested_mode")})
    project_mode = str(obj.get("project_mode") or "").strip().lower()
    if project_mode and project_mode != requested:
        issues.append({"code": "run_receipt_project_mode_mismatch", "expected": requested, "actual": project_mode})
    expected_strategy = str(selected_strategy or "").strip().lower()
    actual_strategy = str(obj.get("selected_strategy") or "").strip().lower()
    if expected_strategy and actual_strategy and expected_strategy != actual_strategy:
        issues.append({"code": "run_receipt_strategy_mismatch", "expected": expected_strategy, "actual": actual_strategy})
    execution = obj.get("mode_execution")
    if isinstance(execution, dict) and execution.get("pass") is False:
        issues.append({"code": "run_receipt_mode_execution_failed", "violations": list(execution.get("violations") or [])})
    return issues

__all__ = ["write_run_receipt", "validate_run_receipt"]
