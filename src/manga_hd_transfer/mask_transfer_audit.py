from __future__ import annotations

"""Pure aggregation/serialization helpers for transfer audit data.

No renderer, Pipeline or Qt import is allowed here.  This keeps audit/reporting
logic independently testable and prevents metadata refactors from changing pixel
rendering behavior.
"""

from collections.abc import Iterable
from typing import Any


def transfer_records_to_dict(records: Iterable[Any]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        to_dict = getattr(record, "to_dict", None)
        if callable(to_dict):
            rows.append(dict(to_dict()))
        elif isinstance(record, dict):
            rows.append(dict(record))
        else:
            rows.append(dict(vars(record)))
    return rows


def transfer_reason_counts(records: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        reason = str(getattr(record, "reason", "") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def summarize_transfer_records(
    records: Iterable[Any],
    *,
    verification_scope: str,
    clear_pixels: int = 0,
    write_pixels: int = 0,
    reason_counts: dict[str, int] | None = None,
) -> dict:
    rows = list(records)
    checked = [r for r in rows if str(getattr(r, "content_check", "")).startswith("checked")]
    applied = [r for r in rows if bool(getattr(r, "applied", False))]
    counts = dict(reason_counts) if reason_counts is not None else transfer_reason_counts(rows)
    return {
        "records": len(rows),
        # Geometry write success and semantic/raster content success are
        # deliberately separate. Do not equate ``applied`` with a
        # publication-complete translation.
        "applied": len(applied),
        "geometry_applied": len(applied),
        "rejected": len(rows) - len(applied),
        "content_checked": len(checked),
        "content_complete": sum(1 for r in rows if bool(getattr(r, "content_complete", False))),
        "content_incomplete": sum(1 for r in checked if not bool(getattr(r, "content_complete", False))),
        "content_unverified": sum(1 for r in applied if not str(getattr(r, "content_check", "")).startswith("checked")),
        "auto_repair_attempted": sum(1 for r in rows if bool(getattr(r, "repair_attempted", False))),
        "auto_repair_succeeded": sum(1 for r in rows if bool(getattr(r, "repair_succeeded", False))),
        "triage_safe": sum(1 for r in rows if str(getattr(r, "triage_state", "")) == "SAFE"),
        "triage_review": sum(1 for r in rows if str(getattr(r, "triage_state", "")) == "REVIEW"),
        "triage_reject": sum(1 for r in rows if str(getattr(r, "triage_state", "")) == "REJECT"),
        "min_source_ink_coverage": min([float(getattr(r, "source_ink_coverage", 0.0)) for r in checked] or [0.0]),
        "max_target_residual_ratio": max([float(getattr(r, "target_residual_ratio", 0.0)) for r in checked] or [0.0]),
        "verification_scope": str(verification_scope),
        "review_required": sum(1 for r in rows if bool(getattr(r, "review_required", False))),
        "low_confidence_candidates": sum(1 for r in rows if bool(getattr(r, "candidate", False))),
        "ocr_guided_records": sum(1 for r in rows if str(getattr(r, "geometry_mode", "")) == "ocr_guided_components"),
        "complex_text_records": sum(1 for r in rows if str(getattr(r, "geometry_mode", "")) == "complex_text"),
        "clear_pixels": int(clear_pixels),
        "write_pixels": int(write_pixels),
        "reason_counts": counts,
    }


def manual_reletter_required_rows(records: Iterable[Any]) -> list[dict]:
    return [
        {
            "source_bubble_id": r.source_bubble_id,
            "target_bubble_id": r.target_bubble_id,
            "source_bbox": list(r.source_bbox),
            "target_bbox": list(r.target_bbox),
            "source_edge_sides": r.source_edge_sides,
            "reason": getattr(r, "review_reason", "") or r.reason,
            "candidate_applied": bool(getattr(r, "candidate", False)),
            "clarity_mode": getattr(r, "clarity_mode", "pixels"),
            "restorable": bool(getattr(r, "restorable", False)),
            "editable": bool(getattr(r, "editable", False)),
        }
        for r in records
        if bool(getattr(r, "review_required", False))
    ]


__all__ = [
    "transfer_records_to_dict",
    "transfer_reason_counts",
    "summarize_transfer_records",
    "manual_reletter_required_rows",
]
