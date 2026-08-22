from __future__ import annotations

"""Persistence boundary for explicit/auto Reveal routes.

Pixel planners/executors stay file-free.  This module owns the route-specific
artifacts, project metadata, QA summaries, and mode-isolation checks without
importing the main pipeline orchestrator.
"""

import os
from pathlib import Path

import cv2
import numpy as np

from .modes.aligned_overlay_reveal.core import AlignedOverlayResult
from .transparent_bubble_reveal import TransparentBubbleResult
from .layout_evidence import collect_koharu_layout_evidence_cached
from .debug import mask_overlay
from .storage_clone import publish_independent_png
from .io_utils import save_json, stem_id, write_image
from .export import write_rgba
from .models import PagePair, PageProject, QAItem, RegistrationResult
from .page_management import PageMark
from .page_pairing import PagePairingCheck
from .result_state import commit_automatic_result
from .detector_policy import auxiliary_detectors, detector_strategy, primary_detector
from .cache import page_job_fingerprint
from .runtime import runtime_summary
from .qa import qa_summary
from .mode_contracts import get_mode_contract, mode_artifact_violations, mode_execution_violations


def _replace_with_hardlink(alias: Path, target: Path) -> bool:
    try:
        alias.unlink(missing_ok=True)
        os.link(target, alias)
        return True
    except OSError:
        return False



def _draw_layout_items_overlay(image: np.ndarray, items, *, labels: tuple[str, ...], title: str = "") -> np.ndarray:
    out = image.copy()
    wanted = {str(x).strip().lower() for x in labels}
    for row in items:
        if str(getattr(row, "label", "")).strip().lower() not in wanted:
            continue
        poly = np.asarray(getattr(row, "polygon", []), dtype=np.int32)
        if poly.ndim == 2 and poly.shape[0] >= 3:
            cv2.polylines(out, [poly.reshape((-1, 1, 2))], True, (255, 0, 255), 2, cv2.LINE_AA)
        x0, y0, x1, y1 = [int(v) for v in getattr(row, "box", (0, 0, 0, 0))]
        if x1 > x0 and y1 > y0:
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 255), 1, cv2.LINE_AA)
        label = f"{getattr(row, 'label', '')}:{float(getattr(row, 'confidence', 0.0)):.2f}"
        if x1 > x0 and y1 > y0:
            cv2.putText(out, label, (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, label, (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
    if title:
        cv2.putText(out, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(out, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (32,32,32), 1, cv2.LINE_AA)
    return out


def _collect_text_only_erase_mask(result: TransparentBubbleResult) -> np.ndarray:
    shape = result.clear_mask.shape[:2]
    out = np.zeros(shape, np.uint8)
    for region in result.plan.applied_regions:
        if str(getattr(region, 'clear_mode', '')).strip().lower() != 'text_only':
            continue
        if region.clear_mask.shape != out.shape:
            mask = cv2.resize(region.clear_mask, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask = region.clear_mask
        out = np.maximum(out, mask)
    return out


def _write_transparent_reveal_debug_artifacts(page_root: Path, target: np.ndarray, result: TransparentBubbleResult, bubble_cfg) -> dict[str, str]:
    layout = collect_koharu_layout_evidence_cached(target, bubble_cfg, role='transparent_debug_target', allow_missing=True)
    shape = target.shape[:2]
    if bool(getattr(layout, 'available', False)):
        bubble_mask = layout.combined_mask(('bubble',), dilate_px=0)
        text_sfx_mask = layout.combined_mask(('text', 'sfx'), dilate_px=0)
        bubble_overlay = _draw_layout_items_overlay(target, layout.items, labels=('bubble',), title='Koharu bubble')
        text_sfx_overlay = _draw_layout_items_overlay(target, layout.items, labels=('text','sfx'), title='Koharu text / sfx')
    else:
        bubble_mask = np.zeros(shape, np.uint8)
        text_sfx_mask = np.zeros(shape, np.uint8)
        bubble_overlay = target.copy()
        text_sfx_overlay = target.copy()
        note = 'Koharu unavailable'
        for img in (bubble_overlay, text_sfx_overlay):
            cv2.putText(img, note, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(img, note, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)

    refined_text_erase = _collect_text_only_erase_mask(result)

    bubble_mask_path = page_root / 'koharu_bubble_mask.png'
    text_sfx_mask_path = page_root / 'koharu_text_sfx_mask.png'
    bubble_overlay_path = page_root / 'koharu_bubble_overlay.png'
    text_sfx_overlay_path = page_root / 'koharu_text_sfx_overlay.png'
    refined_text_erase_path = page_root / 'refined_jp_text_erase_mask.png'

    write_image(bubble_mask_path, bubble_mask)
    write_image(text_sfx_mask_path, text_sfx_mask)
    write_image(bubble_overlay_path, bubble_overlay)
    write_image(text_sfx_overlay_path, text_sfx_overlay)
    write_image(refined_text_erase_path, refined_text_erase)

    return {
        'koharu_bubble_mask': str(bubble_mask_path),
        'koharu_text_sfx_mask': str(text_sfx_mask_path),
        'koharu_bubble_overlay': str(bubble_overlay_path),
        'koharu_text_sfx_overlay': str(text_sfx_overlay_path),
        'refined_jp_text_erase_mask': str(refined_text_erase_path),
        'koharu_available': bool(getattr(layout, 'available', False)),
        'koharu_items': int(len(getattr(layout, 'items', []))),
        'koharu_bubble_pixels': int(cv2.countNonZero(bubble_mask)),
        'koharu_text_sfx_pixels': int(cv2.countNonZero(text_sfx_mask)),
        'refined_jp_text_erase_pixels': int(cv2.countNonZero(refined_text_erase)),
    }

def _write_semantic_artifacts(page_root: Path, target: np.ndarray, result: TransparentBubbleResult, semantic_cfg) -> dict[str, object]:
    payload = dict((result.plan.diagnostics or {}).get("semantic_layout") or {})
    if not payload or not bool(payload.get("available", False)):
        return {"available": False, "provider": str(payload.get("provider", "unavailable"))}
    artifacts: dict[str, object] = {
        "available": True,
        "provider": str(payload.get("provider", "unknown")),
        "block_count": int(len(payload.get("blocks", []) or [])),
    }
    if bool(getattr(semantic_cfg, "save_json", True)):
        semantic_json = page_root / "semantic_layout.json"
        save_json(semantic_json, payload)
        artifacts["semantic_layout_json"] = str(semantic_json)
    if bool(getattr(semantic_cfg, "save_overlay", False)):
        overlay = target.copy()
        for idx, row in enumerate(payload.get("blocks", []) or []):
            bbox = row.get("bbox", []) if isinstance(row, dict) else []
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [int(v) for v in bbox]
            action = str(row.get("action", "REVIEW"))
            color = (40,180,40) if action == "PROCESS" else ((40,40,230) if action == "IGNORE" else (0,180,230))
            cv2.rectangle(overlay, (x0,y0), (x1,y1), color, 2, cv2.LINE_AA)
            label = f"{idx}:{action} {row.get('raw_label','unknown')} {float(row.get('confidence',0.0)):.2f}"
            ty = max(16, y0 - 4)
            cv2.putText(overlay, label, (x0,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(overlay, label, (x0,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
        overlay_path = page_root / "semantic_layout_overlay.png"
        write_image(overlay_path, overlay)
        artifacts["semantic_layout_overlay"] = str(overlay_path)
    return artifacts


def emit_aligned_overlay_page(
    config,
    pair: PagePair,
    page_root: str | Path,
    final_path: str | Path | None,
    mark: PageMark,
    *,
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    pair_check: PagePairingCheck,
    result: AlignedOverlayResult,
    requested_mode: str,
    planner_decision,
    cache_stats: dict[str, str],
) -> PageProject:
    """Persist one aligned-overlay candidate without invoking OCR/Mask.

    The pixel module owns no files.  This pipeline boundary and result_state
    own automatic artifacts/final output so manual Reveal can safely layer on
    top later.
    """
    page_root = Path(page_root)
    page_root.mkdir(parents=True, exist_ok=True)
    local_final, book_final = commit_automatic_result(page_root, result.image, final_path)

    source_original = page_root / "source_original.png"
    target_original = page_root / "target_original.png"
    layer_path = page_root / "aligned_overlay_reveal_layer.png"
    mask_path = page_root / "aligned_overlay_reveal_mask.png"
    source_mask_path = page_root / "aligned_overlay_reveal_source_ink.png"
    hole_mask_path = page_root / "aligned_overlay_reveal_hole_mask.png"
    erase_mask_path = page_root / "aligned_overlay_reveal_erase_mask.png"
    regions_path = page_root / "aligned_overlay_reveal_regions.png"
    meta_path = page_root / "aligned_overlay_reveal.json"
    review_preview = page_root / "review_preview.png"
    target_clear_mask = page_root / "target_clear_mask.png"

    persistent_level = int(max(0, min(9, getattr(config.export, "persistent_png_compression", 4))))
    sparse_level = int(max(0, min(9, getattr(config.export, "sparse_png_compression", 9))))
    persistent_png = [cv2.IMWRITE_PNG_COMPRESSION, persistent_level]
    sparse_png = [cv2.IMWRITE_PNG_COMPRESSION, sparse_level]
    source_method = publish_independent_png(pair.source_path, source_original) if bool(getattr(config.export, "prefer_input_reflink", True)) else None
    target_method = publish_independent_png(pair.target_path, target_original) if bool(getattr(config.export, "prefer_input_reflink", True)) else None
    if source_method is None:
        write_image(source_original, source, params=persistent_png)
    if target_method is None:
        write_image(target_original, target, params=persistent_png)
    write_rgba(layer_path, result.layer_rgba, params=sparse_png)
    write_image(mask_path, result.erase_mask, params=sparse_png)
    write_image(source_mask_path, result.source_ink_mask, params=sparse_png)
    write_image(hole_mask_path, result.plan.full_raster_mask, params=sparse_png)
    write_image(erase_mask_path, result.erase_mask, params=sparse_png)
    write_image(regions_path, result.regions_overlay)
    write_image(target_clear_mask, result.erase_mask, params=sparse_png)

    # Aligned whole-page mode owns only its dedicated diagnostics.  Do not borrow
    # Transparent Reveal debug writers/artifacts.
    debug_artifacts: dict[str, str] = {}

    regions = [r.to_dict() for r in result.plan.regions]
    manual_candidates = list(result.diagnostics.get("manual_effect_candidates", []) or [])
    review_regions = [
        {
            "id": r.get("id", ""),
            "target_bbox": list(r.get("target_bbox", [])),
            "source_bbox": list(r.get("source_bbox", [])),
            "reason": r.get("reason", ""),
            "triage": r.get("triage", "REVIEW"),
            "review_level": "required" if r.get("triage") == "REJECT" else "recommended",
            "restorable": True,
            "editable": True,
        }
        for r in regions if r.get("triage") in {"REVIEW", "REJECT"}
    ]
    write_image(review_preview, result.regions_overlay if review_regions else result.image)

    qa: list[QAItem] = []
    if not result.plan.accepted:
        qa.append(QAItem(
            "aligned_overlay_reveal_rejected", "warning",
            "Experimental aligned erase-to-reveal did not pass its strict gates; TARGET was kept unchanged.",
            meta={"reason": result.plan.reason, **dict(result.diagnostics)},
        ))
    elif result.plan.page_triage == "REVIEW":
        qa.append(QAItem(
            "aligned_overlay_reveal_review", "warning",
            "Experimental aligned erase-to-reveal produced reviewable regions; inspect the diagnostic overlay or finish with manual Reveal.",
            meta={"applied_count": result.applied_count},
        ))
    elif result.plan.page_triage == "SAFE":
        qa.append(QAItem(
            "aligned_overlay_reveal_safe", "info",
            "Experimental aligned erase-to-reveal passed strict white-region and registration gates.",
            meta={"applied_count": result.applied_count},
        ))

    payload = {
        "schema": "manga_hd_translation_transfer.aligned_overlay_reveal.v2",
        "requested_mode": requested_mode,
        "used": bool(result.applied_count > 0),
        "accepted": bool(result.plan.accepted),
        "reason": result.plan.reason,
        "page_triage": result.plan.page_triage,
        "contract": str(result.diagnostics.get("contract", "target_upper_layer__bubble_holes__registered_cn_lower_layer")),
        "diagnostics": dict(result.diagnostics),
        "debug_artifacts": debug_artifacts,
        "regions": regions,
        "review_regions": review_regions,
        "manual_effect_candidates": manual_candidates,
        "planner": planner_decision.to_dict(),
        "page_pairing_check": pair_check.to_dict(),
    }
    save_json(meta_path, payload)

    artifacts = {
        "source_original": str(source_original),
        "target_original": str(target_original),
        "target_clear_mask": str(target_clear_mask),
        "aligned_overlay_reveal_layer": str(layer_path),
        "aligned_overlay_reveal_mask": str(mask_path),
        "aligned_overlay_reveal_source_ink": str(source_mask_path),
        "aligned_overlay_reveal_hole_mask": str(hole_mask_path),
        "aligned_overlay_reveal_erase_mask": str(erase_mask_path),
        "aligned_overlay_reveal_regions": str(regions_path),
        "aligned_overlay_reveal_json": str(meta_path),
        "review_preview": str(review_preview),
        "final": str(local_final),
        "book_final": str(book_final) if book_final is not None else str(local_final),
    }
    project = PageProject(
        page_id=stem_id(pair.target_path), pair=pair, registration=registration,
        source_blocks=[], target_blocks=[], source_bubbles=[], target_bubbles=[],
        source_units=[], target_units=[], matches=[], lettering=[], qa=qa,
        artifacts=artifacts,
        meta={
            "page_management": mark.to_dict(),
            "transfer_mode": requested_mode,
            "mode_contract": get_mode_contract(requested_mode).to_dict(),
            "transfer_planner": planner_decision.to_dict(),
            "page_pairing_check": pair_check.to_dict(),
            "aligned_overlay_reveal": payload,
            "direct_patch": {"used": False, "manual_effect_candidates": []},
            "mask_replace": {"used": False, "review_regions": []},
            "auto_applied_count": int(result.applied_count),
            "job_fingerprint": page_job_fingerprint(pair, config),
            "cache": dict(cache_stats),
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
                "ocr_independent": True,
            },
            "runtime": runtime_summary(config.runtime.device),
            "registration_route": registration.diagnostics.get("route", registration.method),
            "qa_summary": qa_summary(qa),
        },
    )
    aligned_exec_violations = mode_execution_violations(requested_mode, aligned_used=True)
    project.meta["mode_execution"] = {
        "pass": not bool(aligned_exec_violations), "violations": aligned_exec_violations,
        "direct_used": False, "mask_used": False, "reletter_used": False,
        "transparent_used": False, "aligned_used": True,
        "ocr_used": False,
        "ocr_source_route": "skipped_explicit_reveal_contract",
        "ocr_target_route": "skipped_explicit_reveal_contract",
    }
    aligned_violations = mode_artifact_violations(requested_mode, page_root, selected_strategy=planner_decision.strategy)
    project.meta["mode_isolation"] = {
        "pass": not bool(aligned_violations), "violations": aligned_violations,
        "requested_mode": requested_mode, "selected_strategy": str(planner_decision.strategy),
    }
    if aligned_violations:
        qa.append(QAItem("mode_artifact_leak", "error", "Aligned reveal workspace contains artifacts owned by another transfer mode.", meta={"violations": aligned_violations}))
        project.qa = qa
    project.meta["qa_summary"] = qa_summary(qa)
    save_json(page_root / "qa.json", {"summary": qa_summary(qa), "issues": [x.to_dict() for x in qa]})
    if config.export.save_project_json:
        save_json(page_root / "project.json", project.to_dict())
    return project

def emit_transparent_bubble_page(
    config,
    pair: PagePair,
    page_root: str | Path,
    final_path: str | Path | None,
    mark: PageMark,
    *,
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    pair_check: PagePairingCheck,
    result: TransparentBubbleResult,
    planner_decision,
    cache_stats: dict[str, str],
) -> PageProject:
    """Persist the explicit TARGET-alpha reveal route.

    The flattened final.png is the compatibility preview/publication image.
    final_rgba.png and jp_layer_rgba.png preserve the transparent TARGET top
    layer so a layer-aware viewer/editor can place cn_layer_rgb.png beneath it.
    """
    page_root = Path(page_root)
    page_root.mkdir(parents=True, exist_ok=True)
    local_final, book_final = commit_automatic_result(page_root, result.image_rgb, final_path)

    source_original = page_root / "source_original.png"
    target_original = page_root / "target_original.png"
    final_rgba = page_root / "final_rgba.png"
    jp_layer = page_root / "jp_layer_rgba.png"
    cn_layer = page_root / "cn_layer_rgb.png"
    clear_mask = page_root / "clear_mask.png"
    target_clear_mask = page_root / "target_clear_mask.png"
    meta_path = page_root / "transparent_bubble_reveal.json"
    review_preview = page_root / "review_preview.png"

    persistent_level = int(max(0, min(9, getattr(config.export, "persistent_png_compression", 4))))
    sparse_level = int(max(0, min(9, getattr(config.export, "sparse_png_compression", 9))))
    persistent_png = [cv2.IMWRITE_PNG_COMPRESSION, persistent_level]
    sparse_png = [cv2.IMWRITE_PNG_COMPRESSION, sparse_level]
    source_method = publish_independent_png(pair.source_path, source_original) if bool(getattr(config.export, "prefer_input_reflink", True)) else None
    target_method = publish_independent_png(pair.target_path, target_original) if bool(getattr(config.export, "prefer_input_reflink", True)) else None
    if source_method is None:
        write_image(source_original, source, params=persistent_png)
    if target_method is None:
        write_image(target_original, target, params=persistent_png)
    write_rgba(final_rgba, result.image_rgba, params=sparse_png)
    if not _replace_with_hardlink(jp_layer, final_rgba):
        write_rgba(jp_layer, result.jp_layer_rgba, params=sparse_png)
    write_image(cn_layer, result.cn_layer_rgb)
    write_image(clear_mask, result.clear_mask, params=sparse_png)
    if not _replace_with_hardlink(target_clear_mask, clear_mask):
        write_image(target_clear_mask, result.clear_mask, params=sparse_png)
    write_image(review_preview, mask_overlay(target, result.clear_mask) if cv2.countNonZero(result.clear_mask) else target)
    debug_artifacts = _write_transparent_reveal_debug_artifacts(page_root, target, result, config.bubbles)
    semantic_artifacts = _write_semantic_artifacts(page_root, target, result, config.semantic)

    regions = [r.to_dict() for r in result.plan.regions]
    review_regions = [
        {
            "id": row.get("id", ""),
            "target_bbox": list(row.get("target_bbox", [])),
            "reason": row.get("reason", ""),
            "triage": row.get("triage", "REVIEW"),
            "applied": bool(row.get("applied", False)),
            "editable": True,
            "restorable": True,
        }
        for row in regions if row.get("triage") in {"REVIEW", "REJECT"}
    ]
    qa: list[QAItem] = []
    if not result.plan.accepted:
        qa.append(QAItem(
            "transparent_bubble_reveal_rejected", "warning",
            "Transparent bubble reveal was rejected; TARGET was kept unchanged.",
            meta={"reason": result.plan.reason, **dict(result.diagnostics)},
        ))
    elif result.plan.page_triage == "REVIEW":
        qa.append(QAItem(
            "transparent_bubble_reveal_review", "warning",
            "Transparent bubble reveal completed with regions that require review.",
            meta={"applied_count": result.applied_count, **dict(result.diagnostics)},
        ))
    else:
        qa.append(QAItem(
            "transparent_bubble_reveal_safe", "info",
            "Transparent bubble reveal passed registration and page-area gates.",
            meta={"applied_count": result.applied_count},
        ))

    payload = {
        "schema": "manga_hd_translation_transfer.transparent_bubble_reveal.v1",
        "requested_mode": "transparent_bubble_reveal",
        "used": bool(result.plan.accepted and result.applied_count > 0),
        "accepted": bool(result.plan.accepted),
        "reason": result.plan.reason,
        "page_triage": result.plan.page_triage,
        "contract": "registered_cn_lower_layer__transparent_target_bubble_holes",
        "source_detection_used": False,
        "target_only_detection": True,
        "diagnostics": dict(result.diagnostics),
        "regions": regions,
        "review_regions": review_regions,
        "planner": planner_decision.to_dict(),
        "page_pairing_check": pair_check.to_dict(),
    }
    save_json(meta_path, payload)

    artifacts = {
        "source_original": str(source_original),
        "target_original": str(target_original),
        "final": str(local_final),
        "book_final": str(book_final) if book_final is not None else str(local_final),
        "final_rgba": str(final_rgba),
        "jp_layer_rgba": str(jp_layer),
        "cn_layer_rgb": str(cn_layer),
        "clear_mask": str(clear_mask),
        "target_clear_mask": str(target_clear_mask),
        "transparent_bubble_reveal_json": str(meta_path),
        "review_preview": str(review_preview),
        "koharu_bubble_mask": str(debug_artifacts["koharu_bubble_mask"]),
        "koharu_text_sfx_mask": str(debug_artifacts["koharu_text_sfx_mask"]),
        "koharu_bubble_overlay": str(debug_artifacts["koharu_bubble_overlay"]),
        "koharu_text_sfx_overlay": str(debug_artifacts["koharu_text_sfx_overlay"]),
        "refined_jp_text_erase_mask": str(debug_artifacts["refined_jp_text_erase_mask"]),
    }
    if semantic_artifacts.get("semantic_layout_json"):
        artifacts["semantic_layout_json"] = str(semantic_artifacts["semantic_layout_json"])
    if semantic_artifacts.get("semantic_layout_overlay"):
        artifacts["semantic_layout_overlay"] = str(semantic_artifacts["semantic_layout_overlay"])
    project = PageProject(
        page_id=stem_id(pair.target_path), pair=pair, registration=registration,
        source_blocks=[], target_blocks=[], source_bubbles=[], target_bubbles=[],
        source_units=[], target_units=[], matches=[], lettering=[], qa=qa, artifacts=artifacts,
        meta={
            "page_management": mark.to_dict(),
            "transfer_mode": "transparent_bubble_reveal",
            "mode_contract": get_mode_contract("transparent_bubble_reveal").to_dict(),
            "transfer_planner": planner_decision.to_dict(),
            "page_pairing_check": pair_check.to_dict(),
            "transparent_bubble_reveal": payload,
            "transparent_reveal_debug": debug_artifacts,
            "semantic_layout": semantic_artifacts,
            "direct_patch": {"used": False, "manual_effect_candidates": []},
            "mask_replace": {"used": False, "review_regions": []},
            "auto_applied_count": int(result.applied_count),
            "job_fingerprint": page_job_fingerprint(pair, config),
            "cache": dict(cache_stats),
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
                "ocr_independent": True,
            },
            "runtime": runtime_summary(config.runtime.device),
            "registration_route": registration.diagnostics.get("route", registration.method),
            "qa_summary": qa_summary(qa),
        },
    )
    transparent_exec_violations = mode_execution_violations("transparent_bubble_reveal", transparent_used=True)
    project.meta["mode_execution"] = {
        "pass": not bool(transparent_exec_violations), "violations": transparent_exec_violations,
        "direct_used": False, "mask_used": False, "reletter_used": False,
        "transparent_used": True, "aligned_used": False,
        "ocr_used": False,
        "ocr_source_route": "skipped_explicit_reveal_contract",
        "ocr_target_route": "skipped_explicit_reveal_contract",
    }
    transparent_violations = mode_artifact_violations("transparent_bubble_reveal", page_root, selected_strategy=planner_decision.strategy)
    project.meta["mode_isolation"] = {
        "pass": not bool(transparent_violations), "violations": transparent_violations,
        "requested_mode": "transparent_bubble_reveal", "selected_strategy": str(planner_decision.strategy),
    }
    if transparent_violations:
        qa.append(QAItem("mode_artifact_leak", "error", "Transparent reveal workspace contains artifacts owned by another transfer mode.", meta={"violations": transparent_violations}))
        project.qa = qa
    project.meta["qa_summary"] = qa_summary(qa)
    save_json(page_root / "qa.json", {"summary": qa_summary(qa), "issues": [x.to_dict() for x in qa]})
    if config.export.save_project_json:
        save_json(page_root / "project.json", project.to_dict())
    return project

__all__ = ["emit_aligned_overlay_page", "emit_transparent_bubble_page"]
