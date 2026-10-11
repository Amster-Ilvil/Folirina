from __future__ import annotations

"""Deterministic failed/review-only replay planning.

The plan does not change renderer semantics.  It turns QA/review evidence into a
stable region worklist so GUI/automation can retry only affected regions and can
invalidate stale asynchronous work by comparing the generation token.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .io_utils import save_json

_SCHEMA = "folirina.selective_replay.v1"


def _stage_for_issue(code: str) -> str:
    code = str(code or "").lower()
    if "registration" in code or "same_page" in code or "pairing" in code:
        return "registration"
    if "ocr" in code or "text" in code and "content" not in code:
        return "text_ocr"
    if "match" in code or "ambiguous" in code:
        return "matching"
    if "content" in code or "mask" in code or "direct_patch" in code or "transfer" in code:
        return "transfer"
    if "integrity" in code or "mode_" in code:
        return "full_page"
    return "review"


def _stable_issue_row(issue: Any) -> dict:
    meta = dict(getattr(issue, "meta", {}) or {})
    unit_id = str(getattr(issue, "unit_id", "") or "")
    target_id = str(meta.get("target") or meta.get("target_unit_id") or meta.get("target_bubble_id") or "")
    return {
        "code": str(getattr(issue, "code", "") or ""),
        "severity": str(getattr(issue, "severity", "") or "").lower(),
        "unit_id": unit_id,
        "target_id": target_id,
        "stage_start": _stage_for_issue(getattr(issue, "code", "")),
    }


def build_selective_replay_plan(project: Any) -> dict:
    meta = dict(getattr(project, "meta", {}) or {})
    mode = str(meta.get("transfer_mode") or "").strip().lower()
    issues = list(getattr(project, "qa", []) or [])
    candidates = []
    seen: set[tuple[str, str, str, str]] = set()
    for issue in issues:
        row = _stable_issue_row(issue)
        if row["severity"] not in {"warning", "error"}:
            continue
        key = (row["code"], row["severity"], row["unit_id"], row["target_id"])
        if key in seen:
            continue
        seen.add(key)
        row["priority"] = "must_retry" if row["severity"] == "error" else "review_retry"
        row["region_key"] = row["unit_id"] or row["target_id"] or "page"
        candidates.append(row)
    candidates.sort(key=lambda r: (0 if r["severity"] == "error" else 1, r["stage_start"], r["region_key"], r["code"]))
    canonical = {
        "mode": mode,
        "page_id": str(getattr(project, "page_id", "") or ""),
        "candidates": candidates,
    }
    generation = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return {
        "schema": _SCHEMA,
        "generation": generation,
        "mode": mode,
        "page_id": canonical["page_id"],
        "retry_policy": "errors_and_warnings_only",
        "stale_work_guard": "generation_must_match",
        "candidate_count": len(candidates),
        "error_count": sum(1 for r in candidates if r["severity"] == "error"),
        "warning_count": sum(1 for r in candidates if r["severity"] == "warning"),
        "candidates": candidates,
    }


def replay_region_ids(plan: dict, *, include_warnings: bool = True) -> list[str]:
    allowed = {"error", "warning"} if include_warnings else {"error"}
    return sorted({
        str(row.get("region_key") or "")
        for row in list(plan.get("candidates") or [])
        if str(row.get("severity") or "") in allowed and str(row.get("region_key") or "")
    })


def write_selective_replay_plan(page_root: str | Path, project: Any) -> tuple[Path, dict]:
    root = Path(page_root)
    payload = build_selective_replay_plan(project)
    path = root / "replay_plan.json"
    save_json(path, payload)
    return path, payload


def generation_matches(plan: dict, generation: str | None) -> bool:
    """Cheap stale-task cancellation guard inspired by browser generation IDs."""
    return bool(generation) and str(plan.get("generation") or "") == str(generation)


__all__ = [
    "build_selective_replay_plan",
    "replay_region_ids",
    "write_selective_replay_plan",
    "generation_matches",
]
