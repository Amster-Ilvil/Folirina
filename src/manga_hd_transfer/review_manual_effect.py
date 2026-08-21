from __future__ import annotations

"""Manual-effect review service.

Owns open-text/effect/reveal compositing while keeping the main review dispatcher
focused on orchestration.
"""

from pathlib import Path
import cv2
import numpy as np

from .config import PipelineConfig
from .export import export_openraster, export_psd_imagemagick, make_text_layer_rgba, write_rgba
from .inpainting import inpaint_image
from .io_utils import read_image, save_json, write_image
from . import manual_effect as _legacy_manual_effect
from .result_state import ensure_manual_baseline, manual_baseline_path, commit_reviewed_result
from .schema_compat import as_list
from .text_only_transfer import clear_to_target_paper, target_text_mask_in_container
from .review_target_layer import _apply_target_layer_erase_to_rendered, _apply_target_layer_restore_to_rendered
from .review_common import _dict_rows, _source_for_review, _write_bgra, _alpha_over_bgra
from .review_manual_force import _apply_manual_force_transfer_mask
from .region_composite import is_region_mode, apply_region_action
from .review_artifacts import safe_page_artifact_path


def _manual_effect_ops_for_project(project: dict):
    """Return the mode-owned manual pixel engine for the current page.

    Mask Replace and Hybrid intentionally use separate source files so fixes to
    one mode cannot change the other mode's open-text behavior. Other legacy
    routes keep the historical shared helper for backward compatibility.
    """
    meta = project.get("meta", {}) if isinstance(project, dict) else {}
    mode = str(meta.get("transfer_mode", "") or "").strip().lower() if isinstance(meta, dict) else ""
    if mode == "mask_replace":
        from .modes.mask_replace import open_text_manual as ops
        return ops
    if mode == "hybrid":
        from .modes.hybrid import open_text_manual as ops
        return ops
    return _legacy_manual_effect

def _load_reveal_commit_patch(page_dir: Path, row: dict, shape: tuple[int, int]) -> np.ndarray | None:
    """Load an exact sparse preview patch saved by the Qt Reveal editor.

    Older projects do not have this artifact and transparently fall back to
    recomputing the effect from SOURCE/TARGET masks.
    """
    name = str(row.get("reveal_patch_file", "") or "").strip()
    if not name:
        return None
    path = safe_page_artifact_path(page_dir, name)
    if path is None or not path.exists():
        return None
    patch = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if patch is None or patch.ndim != 3 or patch.shape[2] != 4 or patch.shape[:2] != shape:
        return None
    return patch

def _apply_manual_effect_regions(
    rendered: np.ndarray,
    target: np.ndarray,
    page_dir: Path,
    project: dict,
    overrides: dict,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Apply arbitrary detector-independent open/SFX recovery regions.

    This path intentionally does not depend on OCR or a detected speech bubble.
    A human supplies the small target rectangle; paired-image evidence then
    separates SOURCE-supported Chinese strokes from TARGET-supported Japanese
    strokes.  The target-only strokes are inpainted and the Chinese source raster
    is composited afterwards.  ``full_patch`` is also available for genuinely
    safe white/paper regions.
    """
    rows = [dict(x) for x in _dict_rows(overrides.get("manual_effect_regions")) if bool(x.get("enabled", True))]
    h, w = target.shape[:2]
    empty_layer = np.zeros((h, w, 4), np.uint8)
    empty_mask = np.zeros((h, w), np.uint8)
    if not rows:
        return rendered, empty_layer, empty_mask, []

    source = _source_for_review(page_dir, project)
    ops = _manual_effect_ops_for_project(project)
    build_manual_effect_masks = ops.build_manual_effect_masks
    apply_reveal_window = ops.apply_reveal_window
    estimate_source_background = ops.estimate_source_background
    composite_source_text_delta = ops.composite_source_text_delta
    strip_border_ring_components = ops.strip_border_ring_components
    clean_manual_target_text = ops.clean_manual_target_text
    white_container_safe_mask = ops.white_container_safe_mask
    project_mode = str((project.get("meta", {}) or {}).get("transfer_mode", "") or "").strip().lower()
    out = rendered.copy()
    effect_layer = np.zeros((h, w, 4), np.uint8)
    all_clear = np.zeros((h, w), np.uint8)
    applied: list[dict] = []
    for index, row in enumerate(rows):
        row_owner = str(row.get("owner_transfer_mode", "") or "").strip().lower()
        if row_owner and row_owner != project_mode:
            applied.append({
                "id": str(row.get("id", f"manual-effect-{index:03d}")),
                "success": False,
                "reason": "manual_region_owned_by_other_mode",
                "owner_transfer_mode": row_owner,
                "project_transfer_mode": project_mode,
                "skipped": True,
            })
            continue
        mode = str(row.get("mode", "effect_text") or "effect_text")
        if is_region_mode(mode):
            try:
                out, region_layer, region_clear, rec = apply_region_action(out, target, source, project, row, cfg, page_dir=page_dir)
            except Exception as exc:
                applied.append({
                    "id": str(row.get("id", f"manual-effect-{index:03d}")),
                    "success": False, "reason": str(exc), "mode": mode,
                    "region_composite": True,
                })
                continue
            effect_layer = _alpha_over_bgra(effect_layer, region_layer)
            if region_clear.shape == target.shape[:2]:
                all_clear = np.maximum(all_clear, region_clear)
            applied.append(rec)
            continue
        if mode == "open_text_box" and project_mode not in {"mask_replace", "hybrid"}:
            applied.append({
                "id": str(row.get("id", f"manual-effect-{index:03d}")),
                "success": False,
                "reason": "open_text_box_requires_mask_or_hybrid",
                "project_transfer_mode": project_mode,
                "skipped": True,
            })
            continue
        if mode == "open_text_box":
            try:
                manual = ops.render_open_text_box(source, target, project, row, cfg)
            except Exception as exc:
                applied.append({
                    "id": str(row.get("id", f"manual-effect-{index:03d}")),
                    "success": False, "reason": str(exc), "mode": mode,
                    "owner_transfer_mode": row_owner or project_mode,
                })
                continue
            write_mask = np.asarray(manual.get("write_mask"), np.uint8)
            source_mask = np.asarray(manual.get("source_mask"), np.uint8)
            clear_mask = np.asarray(manual.get("target_clear_mask"), np.uint8)
            rendered_region = np.asarray(manual.get("rendered"), np.uint8)
            aligned_source = np.asarray(manual.get("aligned_source"), np.uint8)
            if write_mask.shape != target.shape[:2] or rendered_region.shape != target.shape:
                applied.append({
                    "id": str(row.get("id", f"manual-effect-{index:03d}")),
                    "success": False, "reason": "open_text_box_result_shape_mismatch", "mode": mode,
                })
                continue
            sel = write_mask > 0
            if not np.any(sel):
                applied.append({
                    "id": str(row.get("id", f"manual-effect-{index:03d}")),
                    "success": False, "reason": "open_text_box_empty_write", "mode": mode,
                    "diagnostics": dict(manual.get("diagnostics") or {}),
                })
                continue
            out[sel] = rendered_region[sel]
            if clear_mask.shape == target.shape[:2]:
                all_clear = np.maximum(all_clear, clear_mask)
            if source_mask.shape == target.shape[:2] and aligned_source.shape == target.shape:
                top=np.zeros_like(effect_layer)
                sm=source_mask>0
                top[sm,:3]=aligned_source[sm]
                top[sm,3]=source_mask[sm]
                effect_layer=_alpha_over_bgra(effect_layer,top)
            applied.append({
                "id": str(row.get("id", f"manual-effect-{index:03d}")),
                "success": True, "mode": mode,
                "owner_transfer_mode": row_owner or project_mode,
                "target_bbox": as_list(row.get("target_bbox")),
                "source_pixels": int(cv2.countNonZero(source_mask)) if source_mask.shape == target.shape[:2] else 0,
                "target_clear_pixels": int(cv2.countNonZero(clear_mask)) if clear_mask.shape == target.shape[:2] else 0,
                "write_pixels": int(cv2.countNonZero(write_mask)),
                "ocr_used": False,
                "diagnostics": dict(manual.get("diagnostics") or {}),
            })
            continue
        try:
            masks = build_manual_effect_masks(source, target, project, row, cfg)
        except Exception as exc:
            applied.append({"id": str(row.get("id", f"manual-effect-{index:03d}")), "success": False, "reason": str(exc)})
            continue
        full_source_mask = masks.source_mask.copy()
        source_mask = full_source_mask.copy()
        clear_mask = masks.target_clear_mask.copy()
        mode = str(row.get("mode", "effect_text") or "effect_text")
        reveal_patch = None
        if mode == "reveal_text":
            mask_name = str(row.get("reveal_mask_file", "") or "")
            mask_path = safe_page_artifact_path(page_dir, mask_name) if mask_name else None
            reveal = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path is not None and mask_path.exists() else None
            if reveal is None or reveal.shape != target.shape[:2]:
                applied.append({
                    "id": str(row.get("id", f"manual-effect-{index:03d}")),
                    "success": False,
                    "reason": "reveal mask missing or size mismatch",
                    "mode": mode,
                })
                continue
            source_mask, clear_mask = apply_reveal_window(source_mask, clear_mask, reveal)
            reveal_patch = _load_reveal_commit_patch(page_dir, row, target.shape[:2])
            if reveal_patch is not None:
                # The sparse preview patch is authoritative. Empty patches are a
                # no-op: never clear TARGET just because an old/stale reveal mask
                # exists. This preserves the transactional "save only if Chinese
                # pixels were actually produced" contract.
                patch_alpha = np.minimum(reveal_patch[:, :, 3], reveal).astype(np.uint8)
                patch_sel = patch_alpha > 0
                if np.any(patch_sel) and cv2.countNonZero(clear_mask) > 0:
                    # Clear TARGET Japanese first, then place the sparse Chinese
                    # Reveal patch. Flat colour bubbles use TARGET fill
                    # reconstruction instead of a blurry full-ROI inpaint.
                    cleaned, clean_diag = clean_manual_target_text(
                        target, clear_mask, bbox=as_list(row.get("target_bbox"))
                    )
                    clear_sel = clear_mask > 0
                    out[clear_sel] = cleaned[clear_sel]
                    masks.diagnostics.setdefault("manual_target_cleanup", {}).update(clean_diag)
                if np.any(patch_sel):
                    out[patch_sel] = reveal_patch[:, :, :3][patch_sel]
                    top = reveal_patch.copy(); top[:, :, 3] = patch_alpha
                    effect_layer = _alpha_over_bgra(effect_layer, top)
                all_clear = np.maximum(all_clear, clear_mask)
                diag = dict(masks.diagnostics)
                applied.append({
                    "id": str(row.get("id", f"manual-effect-{index:03d}")),
                    "success": bool(np.any(patch_sel)),
                    "mode": mode,
                    "target_bbox": as_list(row.get("target_bbox")),
                    "source_pixels": int(cv2.countNonZero(source_mask)),
                    "target_clear_pixels": int(cv2.countNonZero(clear_mask)),
                    "preview_patch_applied": True,
                    "preview_patch_pixels": int(cv2.countNonZero(patch_alpha)),
                    "preview_patch_exact": bool(np.array_equal(out[patch_sel], reveal_patch[:, :, :3][patch_sel])) if np.any(patch_sel) else False,
                    "diagnostics": diag,
                })
                continue
        if mode in {"full_patch", "white_bubble_text"}:
            # Manual white-bubble correction is replacement, not additive.  The
            # current base may already contain an automatically transferred CN
            # layer, so clear both TARGET JP text and any existing rendered text
            # inside the confirmed white container before drawing the nudged CN.
            region = np.zeros((h, w), np.uint8)
            safe = np.zeros((h, w), np.uint8)
            bx = as_list(row.get("target_bbox"))
            inset = 4
            if len(bx) == 4:
                rx0, ry0, rx1, ry1 = [int(v) for v in bx]
                rx0 = max(0, min(w, rx0)); rx1 = max(0, min(w, rx1))
                ry0 = max(0, min(h, ry0)); ry1 = max(0, min(h, ry1))
                if rx1 > rx0 and ry1 > ry0:
                    region[ry0:ry1, rx0:rx1] = 255
                    lo = max(0, int(getattr(cfg.mask_replace, "white_container_manual_inset_min_px", 1)))
                    hi = max(lo, int(getattr(cfg.mask_replace, "white_container_manual_inset_max_px", 4)))
                    ratio = max(0.0, float(getattr(cfg.mask_replace, "white_container_manual_inset_ratio", 0.02)))
                    safe, safe_diag = white_container_safe_mask(
                        target, region,
                        inset_min_px=lo,
                        inset_max_px=hi,
                        inset_ratio=ratio,
                    )
                    inset = int(safe_diag.get("container_border_inset_px", 0) or 0)
                    masks.diagnostics.setdefault("white_container_safe_mask", safe_diag)
            current_text = target_text_mask_in_container(out, safe) if cv2.countNonZero(safe) else np.zeros((h, w), np.uint8)
            authority = cv2.bitwise_or(clear_mask, source_mask)
            if cv2.countNonZero(current_text):
                current_text, current_diag = strip_border_ring_components(current_text, safe)
                masks.diagnostics.setdefault("current_text_border_ring_removed", current_diag)
            white_clear = cv2.bitwise_or(authority, current_text)
            if cv2.countNonZero(white_clear):
                white_clear = cv2.dilate(white_clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
                white_clear[safe == 0] = 0
                if cv2.countNonZero(safe):
                    white_clear, white_diag = strip_border_ring_components(white_clear, safe)
                    masks.diagnostics.setdefault("white_clear_border_ring_removed", white_diag)
                out = clear_to_target_paper(out, target, white_clear)
                all_clear = np.maximum(all_clear, white_clear)
        elif cv2.countNonZero(clear_mask) > 0:
            cleaned, clean_diag = clean_manual_target_text(
                target, clear_mask, bbox=as_list(row.get("target_bbox"))
            )
            sel = clear_mask > 0
            out[sel] = cleaned[sel]
            all_clear = np.maximum(all_clear, clear_mask)
            masks.diagnostics.setdefault("manual_target_cleanup", {}).update(clean_diag)

        alpha = source_mask.astype(np.float32) / 255.0
        feather = max(0, min(4, int(row.get("feather_px", 0) or 0)))
        if feather > 0 and cv2.countNonZero(source_mask) > 0 and mode != "full_patch":
            sigma = max(0.35, feather * 0.55)
            alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
            alpha[source_mask > 0] = np.maximum(alpha[source_mask > 0], 0.92)
            alpha = np.clip(alpha, 0.0, 1.0)
        if np.any(alpha > 0):
            source_bg = estimate_source_background(masks.aligned_source, full_source_mask)
            out, delta_diag = composite_source_text_delta(
                out,
                masks.aligned_source,
                source_mask,
                source_background=source_bg,
                alpha=alpha,
            )
            # Keep an approximate editable layer for inspection/export.  The
            # final published render is the delta composite above; this layer is
            # only a reviewer aid and therefore may not fully reproduce the
            # darkening/lightening blend by itself.
            top = np.zeros_like(effect_layer)
            top[:, :, :3] = masks.aligned_source
            top[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
            effect_layer = _alpha_over_bgra(effect_layer, top)
            masks.diagnostics.setdefault("delta_composite", {}).update(delta_diag)

        diag = dict(masks.diagnostics)
        applied.append({
            "id": str(row.get("id", f"manual-effect-{index:03d}")),
            "success": bool(cv2.countNonZero(source_mask) > 0 or cv2.countNonZero(clear_mask) > 0),
            "mode": mode,
            "target_bbox": as_list(row.get("target_bbox")),
            "source_pixels": int(cv2.countNonZero(source_mask)),
            "target_clear_pixels": int(cv2.countNonZero(clear_mask)),
            "diagnostics": diag,
        })

    write_image(page_dir / "manual_effect_transfer_mask.png", effect_layer[:, :, 3])
    write_image(page_dir / "manual_effect_clear_mask.png", all_clear)
    _write_bgra(page_dir / "manual_effect_transfer_layer.png", effect_layer)
    return out, effect_layer, all_clear, applied

def _ensure_manual_effect_stable_base(page_dir: Path) -> Path:
    """Compatibility wrapper around the shared result-state contract."""
    return ensure_manual_baseline(page_dir)

def _manual_effect_overlay_base_path(page_dir: Path) -> Path:
    """Compatibility wrapper retained for older tests/plugins."""
    return manual_baseline_path(page_dir)

def _commit_reviewed_result(page_dir: Path, final_path: Path) -> Path:
    """Compatibility wrapper; all result synchronization is centralized."""
    return commit_reviewed_result(page_dir, final_path)

def _manual_effect_can_overlay_final(page_dir: Path, overrides: dict) -> bool:
    """True when manual omission repair is the only visual review operation.

    In this common workflow the already-rendered automatic ``final.png`` is the
    authoritative base.  Reconstructing the whole page from one transfer layer
    can drop successful replacements produced by another route/supplement.
    """
    if not _dict_rows(overrides.get("manual_effect_regions")):
        return False
    visual_keys = (
        "restore_target_bubbles", "manual_reletter", "text_overrides",
        "match_overrides", "unit_actions",
    )
    if any(bool(overrides.get(key)) for key in visual_keys):
        return False
    if (page_dir / "manual_clear_mask.png").exists():
        return False
    if (page_dir / "manual_transfer_mask.png").exists() or (page_dir / "manual_direct_patch_regions.png").exists():
        return False
    return True

def _apply_manual_effect_only_review(page_dir: Path, project: dict, overrides: dict, cfg: PipelineConfig) -> Path:
    """Manual recovery also works on pages where the automatic route passed through.

    This matters for a page containing only open/SFX text: the detector may emit
    zero speech bubbles, yet the reviewer can still box the missed text and finish
    the page without re-enabling OCR or rerunning the page pipeline.
    """
    target = read_image(page_dir / "target_original.png")
    base_path = _manual_effect_overlay_base_path(page_dir)
    base = read_image(base_path) if base_path.exists() else target.copy()
    rendered = base.copy()
    rendered, layer, clear_mask, applied = _apply_manual_effect_regions(rendered, target, page_dir, project, overrides, cfg)
    rendered, force_layer, force_clear_mask, force_diag = _apply_manual_force_transfer_mask(page_dir, rendered, target, project, cfg)
    erase_extra = [layer[:, :, 3]] if layer is not None and layer.ndim == 3 and layer.shape[2] >= 4 else []
    if force_layer is not None and force_layer.ndim == 3 and force_layer.shape[2] >= 4:
        erase_extra.append(force_layer[:, :, 3])
    rendered, target_erase_diag = _apply_target_layer_erase_to_rendered(
        page_dir, rendered, target, cfg, refresh_base=True, extra_protect_masks=erase_extra
    )
    rendered, target_restore_diag = _apply_target_layer_restore_to_rendered(
        page_dir, rendered, target, refresh_base=True
    )
    final_path = page_dir / "final_reviewed.png"
    write_image(final_path, rendered)
    # Preserve every pre-existing successful replacement in the flattened base.
    # Only the newly requested TARGET text-clear pixels are updated here.
    clean_base = base.copy()
    if cv2.countNonZero(clear_mask) > 0:
        cleaned = inpaint_image(target, clear_mask, cfg.inpainting).image
        clean_base[clear_mask > 0] = cleaned[clear_mask > 0]
    clean_path = page_dir / "review_base.png"
    write_image(clean_path, clean_base)
    transfer_path = page_dir / "manual_effect_transfer_layer.png"
    empty_text = make_text_layer_rgba(target.shape[:2], [], color=cfg.lettering.fill)
    text_path = page_dir / "text_layer_reviewed.png"
    write_rgba(text_path, empty_text)
    export_openraster(page_dir / "editable_reviewed.ora", target, clean_base, empty_text, cv2.cvtColor(layer, cv2.COLOR_BGRA2RGBA))
    psd_ok = export_psd_imagemagick(page_dir / "editable_reviewed.psd", page_dir / "target_original.png", clean_path, text_path, transfer_path)
    save_json(page_dir / "review_applied.json", {
        "mode": "manual_effect_only",
        "status": overrides.get("status", "reviewed_with_manual_effect"),
        "manual_effect_applied": applied,
        "manual_effect_clear_pixels": int(cv2.countNonZero(clear_mask)),
        "manual_force_transfer": force_diag,
        "manual_force_clear_pixels": int(cv2.countNonZero(force_clear_mask)),
        "target_layer_erase": target_erase_diag,
        "target_layer_restore": target_restore_diag,
        "manual_effect_preview_patch_verified": bool(all(
            (not bool(x.get("preview_patch_applied"))) or bool(x.get("preview_patch_exact"))
            for x in applied if bool(x.get("success"))
        )),
        "manual_effect_base": str(base_path),
        "psd_exported": psd_ok,
        "final": str(final_path),
    })
    return final_path

__all__ = ['_load_reveal_commit_patch', '_apply_manual_effect_regions', '_ensure_manual_effect_stable_base', '_manual_effect_overlay_base_path', '_commit_reviewed_result', '_manual_effect_can_overlay_final', '_apply_manual_effect_only_review']
