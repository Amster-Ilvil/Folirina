from __future__ import annotations

"""Lossless page artifact export after renderer completion.

Direct/Mask/Reletter produce pixels before this boundary.  This module owns
artifact naming, layer composition, debug/review exports, editable bundles and
replace-translation sidecars without importing renderer implementations.
"""

import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from .debug import mask_overlay, matching_overlay, registration_overlay, structure_overlay
from .export import export_openraster, export_psd_imagemagick, make_text_layer_rgba, write_rgba
from .io_utils import save_json, write_image
from .mask_transfer_audit import manual_reletter_required_rows, transfer_records_to_dict
from .transfer_policy import (
    _replace_translation_regions,
    _review_candidate_overlay,
    _write_replace_translation_bundle,
)


def _replace_with_hardlink(alias: Path, target: Path) -> bool:
    try:
        alias.unlink(missing_ok=True)
        os.link(target, alias)
        return True
    except OSError:
        return False


def export_page_artifacts(
    *,
    config,
    project,
    pair,
    page_root: str | Path,
    final_path: str | Path | None,
    source,
    authority_source,
    target,
    rendered,
    inpaint_result,
    mask_result,
    lettering_masks,
    mask_transfer,
    direct_container_fast: bool,
    transfer_rgba,
    transfer_audit: dict,
    target_bubbles,
    source_units,
    target_units,
    matches,
    source_blocks,
    target_blocks,
    registration,
    paired_diff,
    decision,
    pair_check,
    direct_container_plan,
    authority_source_path,
    source_path_local,
    selected_source_kind: str,
    secondary_source_available: bool,
    selected_secondary_source: bool,
    dual_source_arbitration,
    selected_arbitration_evidence,
    target_path_local,
    match_result,
) -> None:
    page_root = Path(page_root)

    # Artifacts are intentionally explicit and lossless.
    source_original_path = page_root / "source_original.png"
    authority_source_path_artifact = page_root / "source_authority_original.png"
    original_path = page_root / "target_original.png"
    final_local = page_root / "final.png"
    inpainted_path = page_root / "inpainted.png"
    clear_mask_path = page_root / "clear_mask.png"
    target_clear_mask_path = page_root / "target_clear_mask.png"
    text_layer_path = page_root / "text_layer.png"
    transfer_layer_path = page_root / "mask_transfer_layer.png"
    direct_layer_path = page_root / "direct_patch_layer.png"
    chinese_layer_path = page_root / "chinese_transfer_layer.png"
    transfer_mask_path = page_root / "mask_transfer_mask.png"
    direct_region_path = page_root / "direct_patch_regions.png"
    transfer_audit_path = page_root / "transfer_audit.json"

    write_image(source_original_path, source)
    # The primary/authority source equals the selected source on the common path.
    # Do not store the same lossless page twice. Secondary-source arbitration
    # still keeps the separate authority original.
    authority_artifact_path = authority_source_path_artifact
    if bool(selected_secondary_source) or str(source_path_local) != str(authority_source_path):
        write_image(authority_source_path_artifact, authority_source)
    else:
        if not _replace_with_hardlink(authority_source_path_artifact, source_original_path):
            authority_artifact_path = source_original_path

    write_image(original_path, target)
    write_image(final_local, rendered)

    active_review_meta = project.meta.get("direct_patch", {}) if direct_container_fast else project.meta.get("mask_replace", {})
    # Manual-effect candidates originate from Direct safety analysis even when
    # Auto falls through to Mask. Always include them in review preview/queue.
    direct_review_meta = project.meta.get("direct_patch", {}) if isinstance(project.meta.get("direct_patch", {}), dict) else {}
    review_queue_for_preview = (
        list(active_review_meta.get("review_regions", []) or [])
        + list(direct_review_meta.get("manual_effect_candidates", []) or [])
    )
    review_preview_path = page_root / "review_preview.png"
    review_effective_mask = (
        mask_transfer.clear_mask
        if mask_transfer is not None and mask_transfer.clear_mask is not None
        else mask_result.mask
    )
    write_review_preview = bool(review_queue_for_preview) or bool(getattr(config.export, "save_review_preview_always", False))
    if write_review_preview:
        write_image(
            review_preview_path,
            _review_candidate_overlay(rendered, review_queue_for_preview, effective_mask=review_effective_mask) if review_queue_for_preview else rendered,
        )
    else:
        # Preserve the long-standing review_preview artifact without another
        # full-resolution copy. It is exactly final when there is no queue.
        if not _replace_with_hardlink(review_preview_path, final_local):
            review_preview_path.unlink(missing_ok=True)

    exact_clear = review_effective_mask
    write_image(target_clear_mask_path, exact_clear)
    save_json(transfer_audit_path, transfer_audit)

    if config.export.save_inpainted:
        write_image(inpainted_path, inpaint_result.image)
    elif inpainted_path.exists():
        inpainted_path.unlink(missing_ok=True)

    if config.export.save_masks:
        write_image(clear_mask_path, mask_result.mask)
        if bool(getattr(config.export, "save_component_masks", False)):
            for unit_id, mask in mask_result.per_unit.items():
                write_image(page_root / "masks" / f"{unit_id}.png", mask)
            for bubble in target_bubbles:
                if bubble.mask is not None:
                    write_image(page_root / "bubbles" / f"{bubble.id}.png", bubble.mask)
                if bubble.safe_mask is not None:
                    write_image(page_root / "bubbles" / f"{bubble.id}_safe.png", bubble.safe_mask)
        else:
            for d in (page_root / "masks", page_root / "bubbles"):
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
    else:
        clear_mask_path.unlink(missing_ok=True)

    # Post-render layer composition only: renderer pixels are already final.
    text_rgba = make_text_layer_rgba(target.shape[:2], lettering_masks, color=config.lettering.fill)
    write_rgba(text_layer_path, text_rgba)
    chinese_rgba = np.zeros_like(text_rgba)
    if mask_transfer is not None:
        if direct_container_fast:
            # Direct has its own artifacts. Do not also write mask_transfer_* aliases.
            write_rgba(direct_layer_path, transfer_rgba)
            write_image(direct_region_path, mask_transfer.composite_mask)
        else:
            write_rgba(transfer_layer_path, transfer_rgba)
            write_image(transfer_mask_path, mask_transfer.composite_mask)
        chinese_rgba = transfer_rgba.copy()
    text_use = text_rgba[..., 3] > 0
    chinese_rgba[text_use, :3] = text_rgba[text_use, :3]
    chinese_rgba[..., 3] = np.maximum(chinese_rgba[..., 3], text_rgba[..., 3])
    write_rgba(chinese_layer_path, chinese_rgba)

    if config.export.layer_bundle:
        ora_path = page_root / "editable.ora"
        export_openraster(ora_path, target, inpaint_result.image, text_rgba, transfer_rgba if mask_transfer is not None else None)
        psd_path = page_root / "editable.psd"
        active_transfer_layer_path = direct_layer_path if direct_container_fast else transfer_layer_path
        # PSD needs an on-disk inpainted base; create it lazily only for bundles.
        if not inpainted_path.exists():
            write_image(inpainted_path, inpaint_result.image)
        psd_ok = export_psd_imagemagick(
            psd_path, original_path, inpainted_path, text_layer_path,
            active_transfer_layer_path if mask_transfer is not None and active_transfer_layer_path.exists() else None,
        )
        project.meta["psd_exported"] = psd_ok
        if not config.export.save_inpainted:
            inpainted_path.unlink(missing_ok=True)
    else:
        for stale in (page_root / "editable.ora", page_root / "editable.psd"):
            stale.unlink(missing_ok=True)

    if config.export.save_debug:
        write_image(page_root / "debug_registration.png", registration_overlay(source, target, registration))
        write_image(page_root / "debug_structure.png", structure_overlay(target, target_units, target_bubbles))
        write_image(page_root / "debug_matching.png", matching_overlay(target, source_units, target_units, matches, registration))
        write_image(page_root / "debug_clear_mask.png", mask_overlay(target, mask_result.mask))
        if mask_transfer is not None:
            debug_name = "debug_direct_patch.png" if direct_container_fast else "debug_mask_replace.png"
            write_image(page_root / debug_name, mask_overlay(target, mask_transfer.composite_mask))
        if paired_diff is not None:
            write_image(page_root / "debug_paired_diff.png", mask_overlay(target, paired_diff.change_mask))
            if paired_diff.aligned_source is not None and paired_diff.method == "structural_v08":
                write_image(page_root / "debug_paired_aligned_source.png", paired_diff.aligned_source)
    else:
        for stale in page_root.glob("debug_*.png"):
            stale.unlink(missing_ok=True)

    if mask_transfer is not None and not direct_container_fast and config.mask_replace.save_patch_artifacts:
        save_json(
            page_root / "mask_transfer.json",
            {
                "mode": str(project.meta.get("transfer_mode", "")),
                "applied_count": mask_transfer.applied_count,
                "matches": [m.to_dict() for m in mask_transfer.matches],
                "records": transfer_records_to_dict(mask_transfer.records),
                "manual_reletter_required": manual_reletter_required_rows(mask_transfer.records),
            },
        )
    if direct_container_fast and mask_transfer is not None:
        save_json(
            page_root / "direct_patch.json",
            {
                "schema": "manga_hd_translation_transfer.direct_patch.v1",
                "mode": str(project.meta.get("transfer_mode", "")),
                "contract": "borderless_source_overlay_target_underlay",
                "planner": decision.to_dict(),
                "page_pairing_check": pair_check.to_dict(),
                "diagnostics": dict(direct_container_plan.diagnostics) if direct_container_plan is not None else {},
                "applied_count": mask_transfer.applied_count,
                "records": transfer_records_to_dict(mask_transfer.records),
            },
        )

    if final_path is not None:
        write_image(final_path, rendered)
        project.artifacts["book_final"] = str(Path(final_path))
    project.artifacts.update(
        {
            "source_original": str(source_original_path),
            "source_authority_original": str(authority_artifact_path),
            "target_original": str(original_path),
            "target_clear_mask": str(target_clear_mask_path),
            "chinese_transfer_layer": str(chinese_layer_path),
            "transfer_audit": str(transfer_audit_path),
            "final": str(final_local),
            "review_preview": str(review_preview_path) if review_preview_path.exists() else str(final_local),
            "inpainted": str(inpainted_path) if inpainted_path.exists() else "",
            "clear_mask": str(clear_mask_path) if clear_mask_path.exists() else "",
            "text_layer": str(text_layer_path),
            "mask_transfer_layer": str(transfer_layer_path) if transfer_layer_path.exists() else "",
            "mask_transfer_mask": str(transfer_mask_path) if transfer_mask_path.exists() else "",
            "mask_transfer_json": str(page_root / "mask_transfer.json") if (page_root / "mask_transfer.json").exists() else "",
            "direct_patch_layer": str(direct_layer_path) if direct_layer_path.exists() else "",
            "direct_patch_regions": str(direct_region_path) if direct_region_path.exists() else "",
            "direct_patch_json": str(page_root / "direct_patch.json") if (page_root / "direct_patch.json").exists() else "",
            "openraster": str(page_root / "editable.ora") if (page_root / "editable.ora").exists() else "",
            "psd": str(page_root / "editable.psd") if (page_root / "editable.psd").exists() else "",
        }
    )

    manual_queue = project.meta.get("mask_replace", {}).get("manual_reletter_required", []) if (not direct_container_fast and isinstance(project.meta.get("mask_replace"), dict)) else []
    if manual_queue:
        template = {
            "status": "needs_manual_reletter",
            "notes": "Fill text for clipped or otherwise incomplete source bubbles, then run review apply.",
            "manual_reletter": [
                {
                    "target_bubble_id": row.get("target_bubble_id", ""),
                    "target_bbox": list(row.get("target_bbox", [])),
                    "text": "",
                    "orientation": "auto",
                    "reason": row.get("reason", ""),
                    "source_edge_sides": row.get("source_edge_sides", ""),
                }
                for row in manual_queue
            ],
        }
        template_path = page_root / "review_overrides.template.json"
        save_json(template_path, template)
        project.artifacts["review_template"] = str(template_path)

    rt_summary = {
        "schema": "manga-hd-transfer/replace_translation/v1",
        "compatible_with": "manga-translator-ui/replace_translation",
        "authority_source_path": str(authority_source_path),
        "selected_source_path": str(source_path_local),
        "selected_source_kind": str(selected_source_kind),
        "secondary_source_available": bool(secondary_source_available),
        "secondary_source_selected": bool(selected_secondary_source),
        "arbitration": list(dual_source_arbitration),
        "selected_arbitration_evidence": dict(selected_arbitration_evidence or {}),
        "source_path": str(pair.source_path),
        "target_path": str(target_path_local),
        "regions": _replace_translation_regions(
            source_units, target_units, matches,
            overlap_threshold=float(getattr(config.matching, "replace_translation_overlap_gate", 0.30)),
        ),
        "unmatched_source": list(match_result.unmatched_source),
        "unmatched_target": list(match_result.unmatched_target),
        "ambiguous_source": list(match_result.ambiguous_source),
        "matching_diagnostics": dict(getattr(match_result, "diagnostics", {}) or {}),
        "force_actions": list(getattr(match_result, "diagnostics", {}).get("force_actions", [])),
        "match_stats": {
            "total": len(matches),
            "one_to_one": len([m for m in matches if m.relation == "one_to_one"]),
            "many_to_one": len([m for m in matches if m.relation == "many_to_one"]),
            "one_to_many": len([m for m in matches if m.relation == "one_to_many"]),
        },
    }
    rt_artifacts = _write_replace_translation_bundle(page_root, config.replace_translation, source_blocks, target_blocks, matches, rt_summary)
    if rt_artifacts:
        project.artifacts.update({f"replace_translation_{k}": v for k, v in rt_artifacts.items()})
        if isinstance(project.meta.get("replace_translation"), dict):
            project.meta["replace_translation"]["artifacts"] = rt_artifacts


__all__ = ["export_page_artifacts"]
