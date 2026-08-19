from __future__ import annotations

"""Post-render transfer composition for the main page pipeline.

This module deliberately starts *after* Direct/Mask/Reletter pixel execution.
It owns audit/review/editable metadata projection and completion-display merging,
not renderer algorithms or file persistence.
"""

from dataclasses import dataclass
from typing import Any

import cv2

from .io_utils import stem_id
from .mask_transfer_audit import (
    summarize_transfer_records,
    transfer_reason_counts,
)
from .qa import qa_summary


@dataclass
class TransferComposition:
    page_id: str
    transfer_audit: dict[str, Any]
    reletter_editable_regions: list[dict[str, Any]]
    active_records: list[Any]
    active_matches: list[Any]
    direct_manual_effect_candidates: list[dict[str, Any]]
    active_review_regions: list[dict[str, Any]]
    mask_manual_reletter: list[dict[str, Any]]
    source_bubbles: list[Any]
    target_bubbles: list[Any]


def compose_transfer_state(
    *,
    pair,
    registration,
    pair_check,
    decision,
    mode: str,
    config,
    cache_stats: dict,
    source_blocks: list,
    target_blocks: list,
    source_units: list,
    target_units: list,
    matches: list,
    accepted: list,
    match_result,
    paired_diff,
    mask_transfer,
    direct_container_fast: bool,
    direct_container_plan,
    unseeded_white_pair_count: int,
    fallback_matches: list,
    lettering: list,
    source_bubbles: list,
    target_bubbles: list,
    completion_display_source: list,
    completion_display_target: list,
    qa: list,
) -> TransferComposition:
    """Project renderer outputs into stable audit/review metadata.

    No renderer is imported or invoked here.  The passed transfer/reletter results
    are treated as immutable evidence except for copying their serialized fields.
    """
    page_id = stem_id(pair.target_path)
    pd_records = list(paired_diff.records) if paired_diff is not None else []
    pd_supp = getattr(paired_diff, "supplemental", None) if paired_diff is not None else None
    supp_records = list(getattr(pd_supp, "records", []) or [])
    transfer_records = list(mask_transfer.records) if mask_transfer is not None else []
    clear_pixels = int(cv2.countNonZero(mask_transfer.clear_mask)) if (mask_transfer is not None and mask_transfer.clear_mask is not None) else 0
    write_pixels = int(cv2.countNonZero(mask_transfer.composite_mask)) if mask_transfer is not None else 0
    reason_counts = transfer_reason_counts(transfer_records)
    kind_counts = {"bubble": 0, "free_text": 0, "complex_text": 0}
    for rec in pd_records + supp_records:
        kind = str(getattr(rec, "region_kind", "bubble") or "bubble")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    transfer_audit = {
        "schema": "manga_hd_translation_transfer.transfer_audit.v2",
        "page_id": page_id,
        "registration": {
            "method": registration.method,
            "confidence": float(registration.confidence),
            "accepted_for_structural": bool(registration.confidence >= float(getattr(config.mask_replace, "paired_diff_structural_min_registration_confidence", 0.62))),
            "route": registration.diagnostics.get("route", registration.method),
        },
        "page_pairing_check": pair_check.to_dict(),
        "planner": decision.to_dict(),
        "candidate_detection": {
            "direct_patch_used": bool(direct_container_fast),
            "source_direct_container_used": bool(direct_container_fast),
            "mask_route_used": bool(not direct_container_fast and mode in {"auto", "mask_replace", "hybrid"}),
            "source_direct_container_diagnostics": dict(direct_container_plan.diagnostics) if direct_container_plan is not None else {},
            "paired_diff_used": bool(paired_diff is not None),
            "paired_diff_method": paired_diff.method if paired_diff is not None else None,
            "primary_regions": len(pd_records),
            "supplemental_regions": len(supp_records),
            "regions_by_kind": kind_counts,
            "unseeded_white_container_pairs": int(unseeded_white_pair_count),
            "raw_diagnostics": dict(paired_diff.diagnostics) if paired_diff is not None else {},
        },
        "ocr_evidence": {
            "source_route": str(cache_stats.get("ocr_source", "")),
            "target_route": str(cache_stats.get("ocr_target", "")),
            "source_blocks": len(source_blocks),
            "target_blocks": len(target_blocks),
            "source_nonempty_blocks": len([b for b in source_blocks if str(getattr(b, "text", "")).strip()]),
            "source_units": len(source_units),
            "target_units": len(target_units),
            "unit_matches": len(matches),
            "accepted_unit_matches": len(accepted),
            "ambiguous_source_units": list(match_result.ambiguous_source),
        },
        "transfer": {
            "mode": mode,
            **summarize_transfer_records(
                transfer_records,
                verification_scope=("detected_regions_plus_ocr" if (source_blocks and target_blocks) else "detected_regions_only_no_ocr"),
                clear_pixels=clear_pixels,
                write_pixels=write_pixels,
                reason_counts=reason_counts,
            ),
        },
        "qa": {"summary": qa_summary(qa), "issue_codes": [x.code for x in qa]},
    }

    # Stable Reletter region identity for GUI edits.  This is metadata projection
    # only; the renderer has already finished by the time this function runs.
    reletter_editable_regions: list[dict[str, Any]] = []
    if mode == "reletter" and target_units:
        source_unit_by_id = {u.id: u for u in source_units}
        target_block_by_id_edit = {b.id: b for b in target_blocks}
        match_by_target_edit = {m.target_unit_id: m for m in fallback_matches}
        lettering_by_target_edit = {x.unit_id: x for x in lettering}
        for dst in target_units:
            lr = lettering_by_target_edit.get(dst.id)
            match = match_by_target_edit.get(dst.id)
            if lr is None or match is None:
                continue
            src = source_unit_by_id.get(match.source_unit_id)
            if src is None:
                continue
            block_meta = {}
            if dst.block_ids:
                block = target_block_by_id_edit.get(dst.block_ids[0])
                if block is not None:
                    block_meta = dict(block.meta or {})
            region_id = str((dst.meta or {}).get("reletter_region_id") or block_meta.get("reletter_region_id") or dst.id)
            reletter_editable_regions.append({
                "review_kind": "reletter_auto",
                "target_region_id": region_id,
                "target_unit_id": dst.id,
                "source_unit_id": src.id,
                "target_bubble_id": str(dst.bubble_id or ""),
                "target_bbox": [float(v) for v in dst.bbox],
                "target_polygon": [[float(x), float(y)] for x, y in dst.polygon],
                "text": str(src.text or ""),
                "auto_text": str(src.text or ""),
                "orientation": str(getattr(lr, "orientation", "auto") or "auto"),
                "auto_orientation": str(getattr(lr, "orientation", "auto") or "auto"),
                "font_path": str(getattr(lr, "font_path", "") or ""),
                "auto_font_path": str(getattr(lr, "font_path", "") or ""),
                "font_size": int(getattr(lr, "font_size", 0) or 0),
                "auto_font_size": int(getattr(lr, "font_size", 0) or 0),
                "columns": int(len(getattr(lr, "lines", []) or [])) if str(getattr(lr, "orientation", "")) == "vertical" else 0,
                "line_break_mode": str(getattr(config.lettering, "line_break_mode", "smart")),
                "layout_mode": str(getattr(config.lettering, "layout_mode", "smart_scaling")),
                "line_spacing_ratio": float(getattr(config.lettering, "line_spacing_ratio", 0.16)),
                "auto_lines": list(getattr(lr, "lines", []) or []),
                "auto_bbox": [int(v) for v in getattr(lr, "bbox", (0, 0, 0, 0))],
                "success": bool(getattr(lr, "success", False)),
                "reason": str(getattr(lr, "reason", "") or ""),
                "region_diagnostics": dict(block_meta.get("region_diagnostics") or {}),
            })

    active_records = list(mask_transfer.records) if mask_transfer is not None else []
    active_matches = list(mask_transfer.matches) if mask_transfer is not None else []
    direct_manual_effect_candidates = list(((direct_container_plan.diagnostics if direct_container_plan is not None else {}) or {}).get("manual_effect_candidates", []) or [])
    active_review_regions = [
        {
            "source_bubble_id": r.source_bubble_id,
            "target_bubble_id": r.target_bubble_id,
            "source_bbox": list(r.source_bbox),
            "target_bbox": list(r.target_bbox),
            "source_edge_sides": r.source_edge_sides,
            "reason": (getattr(r, "review_reason", "") or r.reason) if bool(getattr(r, "review_required", False)) else "photographed_text_without_ocr_reletter",
            "candidate_applied": bool(getattr(r, "candidate", False)),
            "clarity_mode": getattr(r, "clarity_mode", "pixels"),
            "restorable": True,
            "editable": True,
            "review_level": "required" if bool(getattr(r, "review_required", False)) else "recommended",
        }
        for r in active_records
        if (
            bool(getattr(r, "review_required", False))
            or (not direct_container_fast and not source_blocks and bool(getattr(r, "applied", False))
                and str(getattr(r, "clarity_mode", "")).startswith("photo-"))
        )
    ]
    mask_manual_reletter = [
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
        for r in active_records
        if (not direct_container_fast and bool(getattr(r, "review_required", False)))
    ]

    display_source_bubbles = list(source_bubbles)
    display_target_bubbles = list(target_bubbles)
    if completion_display_source:
        known = {str(b.id) for b in display_source_bubbles}
        display_source_bubbles.extend([b for b in completion_display_source if str(b.id) not in known])
    if completion_display_target:
        known = {str(b.id) for b in display_target_bubbles}
        display_target_bubbles.extend([b for b in completion_display_target if str(b.id) not in known])

    return TransferComposition(
        page_id=page_id,
        transfer_audit=transfer_audit,
        reletter_editable_regions=reletter_editable_regions,
        active_records=active_records,
        active_matches=active_matches,
        direct_manual_effect_candidates=direct_manual_effect_candidates,
        active_review_regions=active_review_regions,
        mask_manual_reletter=mask_manual_reletter,
        source_bubbles=display_source_bubbles,
        target_bubbles=display_target_bubbles,
    )


__all__ = ["TransferComposition", "compose_transfer_state"]
