from __future__ import annotations

"""PageProject/QA assembly after transfer rendering.

This module owns metadata schema construction and final QA/mode-contract
persistence.  It does not import or invoke Direct, Mask, or Reletter renderers.
"""

from pathlib import Path

from .detector_policy import auxiliary_detectors, detector_strategy, primary_detector
from .cache import page_job_fingerprint
from .io_utils import save_json
from .mask_transfer_audit import transfer_records_to_dict
from .mode_contracts import (
    mode_artifact_violations,
    mode_execution_violations,
    mode_ocr_execution_violations,
    ocr_route_executed,
)
from .models import PageProject, QAItem
from .qa import qa_summary
from .runtime import runtime_summary
from .transfer_policy import _replace_translation_regions
from .workspace_cleanup import cleanup_page_workspace


def assemble_page_project(
    *,
    config,
    pair,
    mark,
    registration,
    source_blocks,
    target_blocks,
    source_units,
    target_units,
    matches,
    lettering,
    qa,
    mode: str,
    mode_contract,
    stale_mode_cleanup: dict,
    review_mode_archive: dict,
    fallback_matches,
    mask_transfer,
    direct_container_fast: bool,
    direct_container_plan,
    use_paired_diff: bool,
    paired_diff,
    target_driven_reletter_regions,
    target_driven_reletter_diagnostics,
    constrained_layout_units: int,
    target_layout_hint_units: int,
    inpaint_result,
    mask_result,
    match_result,
    decision,
    pair_check,
    cache_stats: dict,
    authority_source_path,
    source_path_local,
    selected_source_kind: str,
    secondary_source_available: bool,
    selected_secondary_source: bool,
    dual_source_arbitration,
    selected_arbitration_evidence,
    replace_source_specs,
    composition,
) -> PageProject:
    active_records = composition.active_records
    active_matches = composition.active_matches
    direct_manual_effect_candidates = composition.direct_manual_effect_candidates
    active_review_regions = composition.active_review_regions
    mask_manual_reletter = composition.mask_manual_reletter

    return PageProject(
        page_id=composition.page_id,
        pair=pair,
        registration=registration,
        source_blocks=source_blocks,
        target_blocks=target_blocks,
        source_bubbles=composition.source_bubbles,
        target_bubbles=composition.target_bubbles,
        source_units=source_units,
        target_units=target_units,
        matches=matches,
        lettering=lettering,
        qa=qa,
        meta={
            "page_management": mark.to_dict(),
            "transfer_audit": composition.transfer_audit,
            "mode_contract": mode_contract.to_dict(),
            "mode_workspace_cleanup": dict(stale_mode_cleanup),
            "review_mode_archive": dict(review_mode_archive),
            "auto_applied_match_ids": [f"{m.source_unit_id}->{m.target_unit_id}" for m in fallback_matches],
            "auto_applied_count": len(fallback_matches) + (mask_transfer.applied_count if mask_transfer is not None else 0),
            "reletter_applied_count": len(fallback_matches),
            "reletter": {
                "requested": bool(mode == "reletter"),
                "paired_geometry_used": bool(mode == "reletter" and use_paired_diff),
                "paired_identity_binding_used": bool(mode == "reletter" and (getattr(match_result, "diagnostics", {}) or {}).get("route") == "paired_id_binding"),
                "target_driven_regions_used": bool(mode == "reletter" and target_driven_reletter_regions),
                "target_driven_region_diagnostics": dict(target_driven_reletter_diagnostics or {}),
                "textbox_safe_region_used": int(constrained_layout_units),
                "target_layout_hint_units": int(target_layout_hint_units),
                "ocr_source_route": str(cache_stats.get("ocr_source", "")),
                "ocr_target_route": str(cache_stats.get("ocr_target", "")),
                "configured_font_path": str(config.lettering.font_path or ""),
                "layout_mode": str(getattr(config.lettering, "layout_mode", "smart_scaling")),
                "resolved_font_path": str(next((x.font_path for x in lettering if getattr(x, "font_path", "")), "")),
                "successful_regions": int(sum(1 for x in lettering if bool(getattr(x, "success", False)))),
                "failed_regions": int(sum(1 for x in lettering if not bool(getattr(x, "success", False)))),
                "failed_reasons": [str(getattr(x, "reason", "") or "unknown") for x in lettering if not bool(getattr(x, "success", False))],
                "editable_regions": list(composition.reletter_editable_regions),
            },
            "inpainting": {"method": inpaint_result.method, **inpaint_result.diagnostics},
            "mask_clipped_ratio": mask_result.clipped_ratio,
            "qa_summary": qa_summary(qa),
            "unmatched_source_units": match_result.unmatched_source,
            "unmatched_target_units": match_result.unmatched_target,
            "ambiguous_source_units": match_result.ambiguous_source,
            "matching_diagnostics": dict(getattr(match_result, "diagnostics", {}) or {}),
            "transfer_mode": mode,
            "transfer_planner": decision.to_dict(),
            "page_pairing_check": pair_check.to_dict(),
            "job_fingerprint": page_job_fingerprint(pair, config),
            "cache": cache_stats,
            "layout_evidence": {
                "engine": primary_detector(config.bubbles),
                "preferred": True,
                "authority_policy": detector_strategy(config.bubbles),
                "auxiliary_detectors": auxiliary_detectors(config.bubbles),
                "legacy_preference_value": bool(getattr(config.bubbles, "prefer_koharu_layout", True)),
                "cache_enabled": bool(config.cache.bubbles) and bool(getattr(config.bubbles, "koharu_layout_cache_enabled", True)),
                "prefetch": str(cache_stats.get("layout_prefetch", "")),
                "source_cache": str(cache_stats.get("layout_source_cache", "")),
                "target_cache": str(cache_stats.get("layout_target_cache", "")),
                "source_available": "source" in str(cache_stats.get("layout_prefetch_available", "")).split(","),
                "target_available": "target" in str(cache_stats.get("layout_prefetch_available", "")).split(","),
                "ocr_independent": True,
            },
            "runtime": runtime_summary(config.runtime.device),
            "registration_route": registration.diagnostics.get("route", registration.method),
            "direct_patch": {
                "used": bool(direct_container_fast),
                "requested": bool(mode == "direct_patch"),
                "diagnostics": dict(direct_container_plan.diagnostics) if direct_container_plan is not None else {},
                "contract": "borderless_source_overlay_target_underlay",
                "applied_count": mask_transfer.applied_count if (direct_container_fast and mask_transfer is not None) else 0,
                "records": transfer_records_to_dict(active_records) if direct_container_fast else [],
                "bubble_matches": [m.to_dict() for m in active_matches] if direct_container_fast else [],
                "review_regions": active_review_regions if direct_container_fast else [],
                # Keep Direct safety-review candidates even when Auto falls through
                # to Mask so manual effect omissions remain actionable in the GUI.
                "manual_effect_candidates": direct_manual_effect_candidates,
            },
            # Backward-compatible alias for v0.8.33 project readers.
            "source_direct_container": {
                "used": bool(direct_container_fast),
                "diagnostics": dict(direct_container_plan.diagnostics) if direct_container_plan is not None else {},
            },
            "paired_diff": {
                "used": bool(use_paired_diff),
                "method": paired_diff.method if paired_diff is not None else None,
                "safe_to_skip_ocr": bool(paired_diff.safe_to_skip_ocr) if paired_diff is not None else False,
                "threshold": float(paired_diff.threshold) if paired_diff is not None else None,
                "noise_floor": float(paired_diff.noise_floor) if paired_diff is not None else None,
                "diagnostics": dict(paired_diff.diagnostics) if paired_diff is not None else {},
                "records": transfer_records_to_dict(paired_diff.records) if paired_diff is not None else [],
                "supplemental": {
                    "used": bool(getattr(paired_diff, "supplemental", None)),
                    "method": getattr(getattr(paired_diff, "supplemental", None), "method", None),
                    "records": transfer_records_to_dict(getattr(getattr(paired_diff, "supplemental", None), "records", [])),
                    "diagnostics": dict(getattr(getattr(paired_diff, "supplemental", None), "diagnostics", {}) or {}),
                } if paired_diff is not None else {},
            },
            "replace_translation": {
                "schema": "manga-hd-transfer/replace_translation/v1",
                "compatible_with": "manga-translator-ui/replace_translation",
                "authority_source_path": str(authority_source_path),
                "selected_source_path": str(source_path_local),
                "selected_source_kind": str(selected_source_kind),
                "secondary_source_available": bool(secondary_source_available),
                "secondary_source_selected": bool(selected_secondary_source),
                "arbitration": list(dual_source_arbitration),
                "selected_arbitration_evidence": dict(selected_arbitration_evidence or {}),
                "source_candidates": [{"path": pair.source_path, "kind": "primary"}] + [dict(x) for x in replace_source_specs],
                "regions": _replace_translation_regions(
                    source_units, target_units, matches,
                    overlap_threshold=float(getattr(config.matching, "replace_translation_overlap_gate", 0.30)),
                ),
                "unmatched_source": list(match_result.unmatched_source),
                "unmatched_target": list(match_result.unmatched_target),
                "ambiguous_source": list(match_result.ambiguous_source),
                "matching_diagnostics": dict(getattr(match_result, "diagnostics", {}) or {}),
                "force_actions": list(getattr(match_result, "diagnostics", {}).get("force_actions", [])),
            },
            "mask_replace": {
                "used": bool(mask_transfer is not None and not direct_container_fast and mode in {"auto", "mask_replace", "hybrid"}),
                "strict_no_ocr_reletter": bool(mode == "mask_replace"),
                "mode_contract_no_text_renderer": bool(mode == "mask_replace"),
                "applied_count": mask_transfer.applied_count if (mask_transfer is not None and not direct_container_fast) else 0,
                "records": transfer_records_to_dict(active_records) if not direct_container_fast else [],
                "bubble_matches": [m.to_dict() for m in active_matches] if not direct_container_fast else [],
                "ocr_reletter_preferred_count": len([m for m in fallback_matches if mode == "mask_replace" and paired_diff is not None and paired_diff.method == "photo_pair"]),
                "manual_reletter_required": mask_manual_reletter,
                "review_regions": active_review_regions if not direct_container_fast else [],
            },
            "hybrid": {
                "requested": bool(mode == "hybrid"),
                "used": bool(mode == "hybrid" and mask_transfer is not None),
                "contract": "hybrid_mask_first_then_ocr_reletter_fallback",
                "mask_first": bool(mode == "hybrid" and getattr(config.hybrid, "mask_first", True)),
                "mask_stage_applied_count": int(mask_transfer.applied_count if (mode == "hybrid" and mask_transfer is not None) else 0),
                "reletter_fallback_requested_count": int(len(fallback_matches) if mode == "hybrid" else 0),
                "reletter_fallback_success_count": int(sum(1 for x in lettering if bool(getattr(x, "success", False))) if mode == "hybrid" else 0),
                "mask_incomplete_regions_eligible_for_ocr": int(sum(
                    1 for r in active_records
                    if mode == "hybrid"
                    and bool(getattr(r, "applied", False))
                    and str(getattr(r, "content_check", "") or "").startswith("checked")
                    and not bool(getattr(r, "content_complete", False))
                    and str(getattr(r, "reason", "") or "") not in {"source_text_region_clipped_at_page_edge", "source_open_text_clipped_at_page_edge"}
                    and str(getattr(r, "review_reason", "") or "") not in {"source_text_region_clipped_at_page_edge", "source_open_text_clipped_at_page_edge"}
                )),
                "ocr_source_route": str(cache_stats.get("ocr_source", "")) if mode == "hybrid" else "",
                "ocr_target_route": str(cache_stats.get("ocr_target", "")) if mode == "hybrid" else "",
                "ocr_fallback_available": bool(mode == "hybrid" and source_blocks and target_blocks),
                "ocr_fallback_status": (
                    "used" if mode == "hybrid" and any(bool(getattr(x, "success", False)) for x in lettering)
                    else "ready_no_fallback_needed" if mode == "hybrid" and source_blocks and target_blocks
                    else "backend_unavailable_or_no_usable_text" if mode == "hybrid"
                    else ""
                ),
                "admission": str(cache_stats.get("hybrid_admission", "")) if mode == "hybrid" else "",
                "admission_method": str(cache_stats.get("hybrid_admission_method", "")) if mode == "hybrid" else "",
                "integrity_blocked_regions": [
                    row for row in mask_manual_reletter
                    if str(row.get("reason") or "") in {"source_text_region_clipped_at_page_edge", "source_open_text_clipped_at_page_edge"}
                ] if mode == "hybrid" else [],
                "review_regions": active_review_regions if mode == "hybrid" else [],
            },
        },
    )


def finalize_page_project(
    *,
    config,
    project: PageProject,
    page_root: str | Path,
    mode: str,
    decision,
    direct_container_fast: bool,
    mask_transfer,
    lettering,
) -> PageProject:
    """Apply mode-contract QA, persist project/QA, and compact stale diagnostics."""
    page_root = Path(page_root)
    qa = project.qa
    execution_violations = mode_execution_violations(
        mode,
        direct_used=bool(direct_container_fast),
        mask_used=bool(mask_transfer is not None and not direct_container_fast),
        reletter_used=bool(lettering),
    )
    cache_meta = (project.meta.get("cache") or {}) if isinstance(project.meta, dict) else {}
    ocr_execution_violations = mode_ocr_execution_violations(mode, cache_meta)
    all_execution_violations = list(execution_violations) + list(ocr_execution_violations)
    isolation_violations = mode_artifact_violations(mode, page_root, selected_strategy=decision.strategy)
    project.meta["mode_execution"] = {
        "pass": not bool(all_execution_violations),
        "violations": all_execution_violations,
        "direct_used": bool(direct_container_fast),
        "mask_used": bool(mask_transfer is not None and not direct_container_fast),
        "reletter_used": bool(lettering),
        "ocr_used": bool(
            ocr_route_executed(cache_meta.get("ocr_source"))
            or ocr_route_executed(cache_meta.get("ocr_target"))
        ),
        "ocr_source_route": str(cache_meta.get("ocr_source", "")),
        "ocr_target_route": str(cache_meta.get("ocr_target", "")),
    }
    project.meta["mode_isolation"] = {
        "pass": not bool(isolation_violations),
        "violations": list(isolation_violations),
        "requested_mode": mode,
        "selected_strategy": str(decision.strategy),
    }
    if all_execution_violations:
        qa.append(QAItem(
            "mode_execution_leak", "error",
            "A subsystem outside the selected transfer-mode contract executed on this page.",
            meta={"violations": all_execution_violations, "requested_mode": mode},
        ))
    if isolation_violations:
        qa.append(QAItem(
            "mode_artifact_leak", "error",
            "Renderer artifacts from another transfer mode were found in the current page workspace.",
            meta={"violations": isolation_violations, "requested_mode": mode, "selected_strategy": str(decision.strategy)},
        ))
    project.qa = qa
    project.meta["qa_summary"] = qa_summary(qa)
    save_json(page_root / "qa.json", {"summary": qa_summary(qa), "issues": [x.to_dict() for x in qa]})
    save_json(page_root / "project.json", project.to_dict())

    # Remove leftovers from older full-diagnostic runs when the current compact
    # export settings no longer request them. Manual/review artifacts are never
    # touched by this conservative cleanup.
    if (not config.export.save_debug and not config.export.layer_bundle
            and not config.export.save_inpainted
            and not bool(getattr(config.export, "save_component_masks", False))):
        cleanup_page_workspace(page_root, keep_review_preview=True, keep_authority_alias=True)
    return project


__all__ = ["assemble_page_project", "finalize_page_project"]
