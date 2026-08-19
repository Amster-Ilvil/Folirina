from __future__ import annotations
from .workspace_guard import guarded_page_write


from dataclasses import replace
from pathlib import Path
import shutil

import cv2
import numpy as np

from .config import PipelineConfig
from .export import export_openraster, export_psd_imagemagick, make_text_layer_rgba, write_rgba
from .inpainting import inpaint_image
from .geometry import rasterize_polygon
from .io_utils import load_json, read_image, save_json, write_image
from .lettering import composite_text, fit_text, polygon_safe_mask, textbox_safe_mask
from .masking import build_clear_mask
from .manual_effect import build_manual_effect_masks, apply_reveal_window, estimate_source_background, composite_source_text_delta, strip_border_ring_components, clean_manual_target_text, white_container_safe_mask
from .models import BubbleInstance, PagePair, TextBlock, TextUnit, UnitMatch
from .text_only_transfer import clear_to_target_paper, target_text_mask_in_container
from .schema_compat import as_dict, as_dict_rows, as_list, normalize_project, normalize_overrides, normalize_route_meta
from .result_state import ensure_manual_baseline, manual_baseline_path, commit_reviewed_result, resolve_result_state
from .mode_contracts import get_mode_contract, review_owner_compatible
from .review_target_layer import (
    _read_layer_alpha, _protected_chinese_mask, _clean_target_under_erase_mask,
    _apply_target_layer_erase_to_rendered, _apply_target_layer_restore_to_rendered,
    apply_target_layer_erase_review, reset_target_layer_erase_review,
    apply_target_layer_restore_review, reset_target_layer_restore_review,
)
from .review_common import (
    _dict_or_empty, _route_meta, _dict_rows, _text_block, _text_unit, _load_target_bubbles,
    _rect_mask, _clear_region_to_paper, _source_for_review, _polygon_mask,
    _project_text_ink_mask, _write_bgra, _alpha_over_bgra,
)
from .review_manual_force import (
    manual_force_auto_evidence_masks, _auto_evidence_touching, _manual_force_settings,
    _manual_force_mask_regions, _apply_manual_force_transfer_mask,
    apply_manual_force_transfer_review, reset_manual_force_transfer_review,
)
from .review_manual_effect import (
    _load_reveal_commit_patch, _apply_manual_effect_regions, _ensure_manual_effect_stable_base,
    _manual_effect_overlay_base_path, _commit_reviewed_result,
    _manual_effect_can_overlay_final, _apply_manual_effect_only_review,
)


















def _load_effective_clear_mask(page_dir: Path, shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    """Load base clear mask plus the additive Japanese-cleanup brush.

    ``manual_clear_mask.png`` is the legacy authoritative full-mask editor.  The
    v1.3 ``manual_japanese_clear_mask.png`` is deliberately additive so a reviewer
    can brush missed Japanese without replacing the automatic detector mask.
    """
    base = np.zeros(shape, np.uint8)
    source = "none"
    candidates = [
        # The force editor can directly revise the blue OCR/automatic layer.
        # When present, that page-local override is the newest explicit clear
        # decision and must win over the original automatic detector artifact.
        (page_dir / "manual_force_auto_target_override.png", "manual_force_auto_target_override"),
        (page_dir / "manual_clear_mask.png", "manual_clear_mask"),
        (page_dir / "target_clear_mask.png", "target_clear_mask"),
        (page_dir / "clear_mask.png", "clear_mask"),
    ]
    for path, label in candidates:
        if not path.exists():
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != shape:
            if label in {"manual_clear_mask", "manual_force_auto_target_override"}:
                raise ValueError(f"{path.name} has invalid dimensions")
            continue
        base = (mask > 0).astype(np.uint8) * 255
        source = label
        break

    additive = page_dir / "manual_japanese_clear_mask.png"
    if additive.exists():
        extra = cv2.imread(str(additive), cv2.IMREAD_GRAYSCALE)
        if extra is None or extra.shape != shape:
            raise ValueError("manual_japanese_clear_mask.png has invalid dimensions")
        base = np.maximum(base, (extra > 0).astype(np.uint8) * 255)
        source = f"{source}+manual_japanese_clear_mask" if source != "none" else "manual_japanese_clear_mask"
    return base, source


def _load_clear_brush_settings(page_dir: Path) -> dict:
    """Read page-local Japanese-cleanup behavior without mutating user config."""
    path = page_dir / "clear_brush_settings.json"
    data = load_json(path) if path.exists() else {}
    mode = str(data.get("mode", "strict")).strip().lower()
    if mode not in {"strict", "aggressive"}:
        mode = "strict"
    default_dilate = 1 if mode == "strict" else 3
    try:
        dilate = int(data.get("dilate_px", default_dilate))
    except Exception:
        dilate = default_dilate
    dilate = max(0, min(8, dilate))
    return {"mode": mode, "dilate_px": dilate}


def _apply_manual_brush_expansion(page_dir: Path, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Expand only the additive brush; never shrink/crop explicit reviewer intent."""
    settings = _load_clear_brush_settings(page_dir)
    additive_path = page_dir / "manual_japanese_clear_mask.png"
    if not additive_path.exists():
        return mask.copy(), {**settings, "manual_pixels": 0, "expanded_pixels": 0}
    extra = cv2.imread(str(additive_path), cv2.IMREAD_GRAYSCALE)
    if extra is None or extra.shape != mask.shape:
        raise ValueError("manual_japanese_clear_mask.png has invalid dimensions")
    raw = (extra > 0).astype(np.uint8) * 255
    expanded = raw.copy()
    r = int(settings["dilate_px"])
    if r > 0 and cv2.countNonZero(expanded) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
        expanded = cv2.dilate(expanded, k, iterations=1)
    out = np.maximum(mask, expanded)
    return out, {
        **settings,
        "manual_pixels": int(cv2.countNonZero(raw)),
        "expanded_pixels": int(cv2.countNonZero(expanded)),
        "explicit_mask_never_clipped": True,
    }


def _residual_dark_heatmap(target: np.ndarray, cleaned: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Visualize dark TARGET pixels that remain dark after the remove-text stage."""
    if target.shape != cleaned.shape or target.shape[:2] != mask.shape:
        raise ValueError("residual heatmap inputs must share shape")
    tgray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    cgray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    use = mask > 0
    target_dark = use & (tgray <= 205)
    residual = target_dark & (cgray <= 205)
    denom = max(1, int(np.count_nonzero(target_dark)))
    ratio = float(np.count_nonzero(residual) / denom)
    heat = target.copy()
    if np.any(residual):
        overlay = heat.copy()
        overlay[residual] = (0, 0, 255)
        heat = cv2.addWeighted(heat, 0.48, overlay, 0.52, 0.0)
    return heat, {
        "target_dark_pixels": int(np.count_nonzero(target_dark)),
        "residual_dark_pixels": int(np.count_nonzero(residual)),
        "residual_dark_ratio": ratio,
    }



















@guarded_page_write("remove_text_preview")
def generate_remove_text_preview(page_dir: str | Path, config: PipelineConfig | None = None) -> Path:
    """Run only the target-text removal stage for the current page.

    This is deliberately independent from Chinese raster transfer. It lets the
    user inspect/repair the clear mask without rerunning registration, detection,
    OCR or glyph placement. The result is safe to regenerate repeatedly.
    """
    page_dir = Path(page_dir)
    cfg = config or PipelineConfig()
    target = read_image(page_dir / "target_original.png")
    raw_mask, source = _load_effective_clear_mask(page_dir, target.shape[:2])
    mask, brush_diag = _apply_manual_brush_expansion(page_dir, raw_mask)
    # Persist exactly what the remove/apply stage will use. This is the
    # reviewer-facing "真实生效 mask", independent from the brush source file.
    effective_path = page_dir / "effective_clear_mask.png"
    write_image(effective_path, mask)
    if cv2.countNonZero(mask) == 0:
        preview = target.copy()
        backend = "none"
    else:
        result = inpaint_image(target, mask, cfg.inpainting)
        preview = result.image
        backend = str(getattr(result, "method", getattr(cfg.inpainting, "backend", "auto")))
    out = page_dir / "removed_text_preview.png"
    write_image(out, preview)
    heat, residual_diag = _residual_dark_heatmap(target, preview, mask)
    heat_path = page_dir / "japanese_residual_heatmap.png"
    write_image(heat_path, heat)
    save_json(page_dir / "remove_text_stage.json", {
        "schema": "manga_hd_translation_transfer.remove_text_stage.v2",
        "mask_source": source,
        "raw_mask_pixels": int(cv2.countNonZero(raw_mask)),
        "mask_pixels": int(cv2.countNonZero(mask)),
        "effective_mask": str(effective_path),
        "brush": brush_diag,
        "inpainting_backend": backend,
        **residual_diag,
        "residual_review_recommended": bool(residual_diag["target_dark_pixels"] >= 8 and residual_diag["residual_dark_ratio"] > 0.08),
        "residual_heatmap": str(heat_path),
        "output": str(out),
    })
    return out










































def _apply_manual_reletters(rendered: np.ndarray, target: np.ndarray, page_dir: Path, project: dict, overrides: dict, cfg: PipelineConfig) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    rows = _dict_rows(overrides.get("manual_reletter"))
    if not rows:
        return rendered, [], []
    target_bubbles = _load_target_bubbles(page_dir, project.get("target_bubbles", []))
    bubbles_by_id = {b.id: b for b in target_bubbles}
    meta = _dict_or_empty(project.get("meta"))
    direct_meta = _route_meta(meta, "direct_patch")
    active_meta = direct_meta if bool(direct_meta.get("used")) else _route_meta(meta, "mask_replace")
    manual_queue = _dict_rows(active_meta.get("manual_reletter_required"))
    queue_by_target = {str(x.get("target_bubble_id", "")): x for x in manual_queue if x.get("target_bubble_id")}
    out = rendered.copy()
    masks: list[np.ndarray] = []
    applied: list[dict] = []
    for i, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        bubble_id = str(row.get("target_bubble_id", "") or "")
        bubble = bubbles_by_id.get(bubble_id)
        safe = None
        polygon = None
        if bubble is not None:
            bubble_id = bubble.id
            polygon = bubble.polygon
            safe = bubble.safe_mask if bubble.safe_mask is not None and cv2.countNonZero(bubble.safe_mask) > 0 else bubble.mask
        bbox = row.get("target_bbox")
        if (not bbox) and bubble_id and bubble_id in queue_by_target:
            bbox = queue_by_target[bubble_id].get("target_bbox")
        if (safe is None or cv2.countNonZero(safe) == 0) and bbox:
            safe = _rect_mask(target.shape[:2], bbox, inset=3)
            x0, y0, x1, y1 = map(int, bbox)
            polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if safe is None or cv2.countNonZero(safe) == 0 or not polygon:
            continue
        out = _clear_region_to_paper(out, target, safe)
        orientation = str(row.get("orientation", "auto") or "auto")
        lcfg = cfg.lettering.model_copy(deep=True) if hasattr(cfg.lettering, 'model_copy') else cfg.lettering.copy(deep=True)
        lcfg.orientation = orientation
        font_path = str(row.get("font_path") or "").strip()
        if font_path:
            lcfg.font_path = font_path
        manual_size = int(row.get("font_size") or 0)
        if manual_size > 0:
            lcfg.min_font_size = manual_size; lcfg.max_font_size = manual_size; lcfg.preferred_font_size = manual_size
        manual_columns = int(row.get("columns") or 0)
        if manual_columns > 0:
            lcfg.preferred_columns = manual_columns
        manual_break = str(row.get("line_break_mode") or "").strip().lower()
        if manual_break in {"smart", "balanced", "source"}:
            lcfg.line_break_mode = manual_break
        manual_layout = str(row.get("layout_mode") or "").strip().lower()
        if manual_layout in {"strict", "smart_scaling", "balloon_fill"}:
            lcfg.layout_mode = manual_layout
        manual_spacing = row.get("line_spacing_ratio")
        if manual_spacing is not None:
            try: lcfg.line_spacing_ratio = float(np.clip(float(manual_spacing), 0.0, 0.6))
            except Exception: pass
        unit = TextUnit(
            id=f"manual-reletter-{i:03d}",
            polygon=polygon,
            block_ids=[],
            text=text,
            confidence=1.0,
            kind=str(row.get("kind", "speech") or "speech"),
            reading_order=i,
            bubble_id=bubble_id,
            meta={"review_manual": True},
        )
        lr = fit_text(target.shape[:2], safe, unit, text, lcfg)
        if not lr.success or lr.text_mask is None:
            continue
        out = composite_text(out, lr, lcfg)
        masks.append(lr.text_mask)
        applied.append({
            "target_bubble_id": bubble_id or "",
            "text": text,
            "orientation": lr.orientation,
            "font_path": lr.font_path,
            "font_size": int(lr.font_size),
            "columns": int(manual_columns),
            "line_break_mode": str(getattr(lcfg, "line_break_mode", "smart")),
            "layout_mode": str(getattr(lcfg, "layout_mode", "smart_scaling")),
            "line_spacing_ratio": float(getattr(lcfg, "line_spacing_ratio", 0.16)),
            "lines": list(lr.lines),
            "bbox": list(lr.bbox),
        })
    return out, masks, applied


def _apply_mask_replace_review(page_dir: Path, project: dict, cfg: PipelineConfig) -> Path:
    target = read_image(page_dir / "target_original.png")
    overrides_probe = normalize_overrides(load_json(page_dir / "review_overrides.json") if (page_dir / "review_overrides.json").exists() else {})
    manual_force_present = (page_dir / "manual_force_transfer_mask.png").exists()
    if manual_force_present:
        ensure_manual_baseline(page_dir)
    automatic_path = _manual_effect_overlay_base_path(page_dir) if (_dict_rows(overrides_probe.get("manual_effect_regions")) or manual_force_present) else page_dir / "final.png"
    automatic = read_image(automatic_path) if automatic_path.exists() else None
    if automatic is not None and automatic.shape != target.shape:
        automatic = None
    review_change_mask = np.zeros(target.shape[:2], np.uint8)
    meta = _dict_or_empty(project.get("meta"))
    direct_meta = _route_meta(meta, "direct_patch")
    direct_used = bool(direct_meta.get("used"))
    route = "direct_patch" if direct_used else "mask_replace"
    layer_path = page_dir / ("direct_patch_layer.png" if direct_used else "mask_transfer_layer.png")
    if not layer_path.exists():
        raise FileNotFoundError(f"{layer_path.name} is missing")
    bgra = cv2.imread(str(layer_path), cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError("mask_transfer_layer.png must be RGBA")
    if bgra.shape[:2] != target.shape[:2]:
        raise ValueError("mask transfer layer size mismatch")

    overrides_path = page_dir / "review_overrides.json"
    overrides = normalize_overrides(load_json(overrides_path) if overrides_path.exists() else {})
    transfer_meta = _route_meta(meta, route)
    review_queue = _dict_rows(transfer_meta.get("review_regions") or transfer_meta.get("manual_reletter_required"))
    queue_by_target = {str(x.get("target_bubble_id", "")): x for x in review_queue if x.get("target_bubble_id")}
    restore_ids = set(map(str, overrides.get("restore_target_bubbles", []) or []))
    accept_ids = set(map(str, overrides.get("accept_candidate_targets", []) or []))
    manual_rows = [x for x in _dict_rows(overrides.get("manual_reletter")) if str(x.get("text", "")).strip()]
    edit_ids = {str(x.get("target_bubble_id", "")) for x in manual_rows if x.get("target_bubble_id")}

    patch_bgr = bgra[:, :, :3]
    original_alpha = bgra[:, :, 3]
    manual = page_dir / ("manual_direct_patch_regions.png" if direct_used else "manual_transfer_mask.png")
    if manual.exists():
        m = cv2.imread(str(manual), cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != target.shape[:2]:
            raise ValueError("manual_transfer_mask.png has invalid dimensions")
        alpha = np.minimum(original_alpha, m)
        review_change_mask[alpha != original_alpha] = 255
    else:
        alpha = original_alpha.copy()

    # v0.8.25: the clear mask is an independent editable overlay. When present,
    # run only inpainting on that mask first; the Chinese transfer layer is
    # composited afterwards. This mirrors the detector -> mask -> remove -> write
    # separation used by mature comic-translation editors.
    effective_clear_raw, clear_source = _load_effective_clear_mask(page_dir, target.shape[:2])
    effective_clear, brush_diag = _apply_manual_brush_expansion(page_dir, effective_clear_raw)
    write_image(page_dir / "effective_clear_mask.png", effective_clear)
    manual_clear = page_dir / "manual_clear_mask.png"
    manual_japanese_clear = page_dir / "manual_japanese_clear_mask.png"
    manual_force_auto_override = page_dir / "manual_force_auto_target_override.png"
    manual_clear_present = manual_clear.exists() or manual_japanese_clear.exists() or manual_force_auto_override.exists()
    if manual_clear_present:
        auto_clear = np.zeros(target.shape[:2], np.uint8)
        for candidate in (page_dir / "target_clear_mask.png", page_dir / "clear_mask.png"):
            if not candidate.exists():
                continue
            probe = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if probe is not None and probe.shape == target.shape[:2]:
                auto_clear = (probe > 0).astype(np.uint8) * 255
                break
        clear_delta = cv2.absdiff((effective_clear > 0).astype(np.uint8) * 255, auto_clear)
        if cv2.countNonZero(clear_delta) > 0:
            # Small halo covers antialiased glyph edges affected by local inpaint.
            clear_delta = cv2.dilate(clear_delta, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
            review_change_mask = np.maximum(review_change_mask, clear_delta)
    residual_diag = {"target_dark_pixels": 0, "residual_dark_pixels": 0, "residual_dark_ratio": 0.0}
    if manual_clear_present and cv2.countNonZero(effective_clear) > 0:
        base = inpaint_image(target, effective_clear, cfg.inpainting).image
        heat, residual_diag = _residual_dark_heatmap(target, base, effective_clear)
        write_image(page_dir / "japanese_residual_heatmap.png", heat)
        # The original transfer layer alpha contains both Chinese glyphs and the
        # automatically cleared paper. When the user erases part of the clear
        # overlay, keep only real dark Chinese raster there; otherwise the old
        # white clear patch would silently override the manual erase.
        pgray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        glyph_alpha = np.where((original_alpha > 0) & (pgray <= 232), original_alpha, 0).astype(np.uint8)
        alpha = np.maximum(glyph_alpha, np.minimum(alpha, effective_clear))
    else:
        base = target.copy()

    # Restore means the candidate disappears and the exact untouched Japanese
    # master becomes visible again. Edit also removes the candidate, but clears the
    # old Japanese text on a review base before drawing new HD Chinese text.
    for tid in sorted(restore_ids | edit_ids):
        row = queue_by_target.get(tid, {})
        box = row.get("target_bbox") or []
        if len(box) != 4:
            continue
        x0, y0, x1, y1 = map(int, box)
        x0=max(0,x0); y0=max(0,y0); x1=min(target.shape[1],x1); y1=min(target.shape[0],y1)
        if x1 <= x0 or y1 <= y0:
            continue
        review_change_mask[y0:y1, x0:x1] = 255
        alpha[y0:y1, x0:x1] = 0
        if tid in restore_ids:
            base[y0:y1, x0:x1] = target[y0:y1, x0:x1]
        elif tid in edit_ids:
            clear = _rect_mask(target.shape[:2], (x0,y0,x1,y1), inset=2)
            base = _clear_region_to_paper(base, target, clear)

    a = (alpha.astype(np.float32) / 255.0)[..., None]
    rendered = np.clip(patch_bgr.astype(np.float32) * a + base.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)
    if automatic is not None:
        # The automatic final may contain successful replacements from several
        # routes (Direct + supplemental Mask + safe lettering).  A review of one
        # local region must not flatten the whole page back to TARGET merely
        # because the chosen editable layer represents only one of those routes.
        untouched = review_change_mask == 0
        rendered[untouched] = automatic[untouched]
    final_path = page_dir / "final_reviewed.png"
    transfer_reviewed = page_dir / ("direct_patch_layer_reviewed.png" if direct_used else "mask_transfer_layer_reviewed.png")
    reviewed_base_path = page_dir / "review_base.png"
    write_image(reviewed_base_path, base)

    reviewed_bgra = bgra.copy(); reviewed_bgra[:, :, 3] = alpha
    ok, data = cv2.imencode(".png", reviewed_bgra)
    if not ok: raise ValueError("could not encode reviewed transfer layer")
    data.tofile(transfer_reviewed)

    rendered, manual_masks, manual_applied = _apply_manual_reletters(rendered, target, page_dir, project, overrides, cfg)
    rendered, effect_layer, effect_clear_mask, effect_applied = _apply_manual_effect_regions(rendered, target, page_dir, project, overrides, cfg)
    rendered, force_layer, force_clear_mask, force_diag = _apply_manual_force_transfer_mask(page_dir, rendered, target, project, cfg)
    erase_extra = list(manual_masks)
    if effect_layer is not None and effect_layer.ndim == 3 and effect_layer.shape[2] >= 4:
        erase_extra.append(effect_layer[:, :, 3])
    if force_layer is not None and force_layer.ndim == 3 and force_layer.shape[2] >= 4:
        erase_extra.append(force_layer[:, :, 3])
    rendered, target_erase_diag = _apply_target_layer_erase_to_rendered(
        page_dir, rendered, target, cfg, refresh_base=True, extra_protect_masks=erase_extra
    )
    rendered, target_restore_diag = _apply_target_layer_restore_to_rendered(
        page_dir, rendered, target, refresh_base=True
    )
    write_image(final_path, rendered)

    # Keep editable exports faithful to the reviewed result: target-only Japanese
    # strokes removed by a manual open-text region also disappear from the base,
    # while the recovered Chinese raster is merged into the transfer layer.
    if cv2.countNonZero(effect_clear_mask) > 0:
        effect_clean = inpaint_image(target, effect_clear_mask, cfg.inpainting).image
        base[effect_clear_mask > 0] = effect_clean[effect_clear_mask > 0]
        write_image(reviewed_base_path, base)
    merged_bgra = _alpha_over_bgra(reviewed_bgra, effect_layer) if np.any(effect_layer[:, :, 3] > 0) else reviewed_bgra
    if force_layer is not None and np.any(force_layer[:, :, 3] > 0):
        merged_bgra = _alpha_over_bgra(merged_bgra, force_layer)
    _write_bgra(transfer_reviewed, merged_bgra)
    transfer_rgba = cv2.cvtColor(merged_bgra, cv2.COLOR_BGRA2RGBA)
    text_rgba = make_text_layer_rgba(target.shape[:2], manual_masks, color=cfg.lettering.fill)
    text_path = page_dir / "text_layer_reviewed.png"
    write_rgba(text_path, text_rgba)
    export_openraster(page_dir / "editable_reviewed.ora", target, base, text_rgba, transfer_rgba)
    psd_ok = export_psd_imagemagick(page_dir / "editable_reviewed.psd", page_dir / "target_original.png", reviewed_base_path, text_path, transfer_reviewed)

    unresolved = [
        x for x in review_queue
        if str(x.get("target_bubble_id", "")) not in (restore_ids | accept_ids | edit_ids)
        and str(x.get("review_level", "required")) == "required"
    ]
    unreviewed_recommended = [
        x for x in review_queue
        if str(x.get("target_bubble_id", "")) not in (restore_ids | accept_ids | edit_ids)
        and str(x.get("review_level", "required")) != "required"
    ]
    save_json(page_dir / "review_applied.json", {
        "mode": route,
        "status": overrides.get("status", "reviewed"),
        "notes": overrides.get("notes", ""),
        "manual_transfer_mask": manual.exists(),
        "manual_clear_mask": manual_clear.exists(),
        "manual_japanese_clear_mask": manual_japanese_clear.exists(),
        "manual_force_auto_target_override": manual_force_auto_override.exists(),
        "clear_mask_source": clear_source,
        "clear_brush": brush_diag,
        "target_layer_erase": target_erase_diag,
        "target_layer_restore": target_restore_diag,
        "effective_clear_pixels": int(cv2.countNonZero(effective_clear)),
        "residual_dark_pixels": int(residual_diag.get("residual_dark_pixels", 0)),
        "residual_dark_ratio": float(residual_diag.get("residual_dark_ratio", 0.0)),
        "residual_review_recommended": bool(int(residual_diag.get("target_dark_pixels", 0)) >= 8 and float(residual_diag.get("residual_dark_ratio", 0.0)) > 0.08),
        "review_change_pixels": int(cv2.countNonZero(review_change_mask)),
        "automatic_final_preserved_outside_review": bool(automatic is not None),
        "manual_reletter_applied": manual_applied,
        "manual_effect_applied": effect_applied,
        "manual_effect_clear_pixels": int(cv2.countNonZero(effect_clear_mask)),
        "manual_force_transfer": force_diag,
        "manual_force_clear_pixels": int(cv2.countNonZero(force_clear_mask)),
        "manual_effect_preview_patch_verified": bool(all(
            (not bool(x.get("preview_patch_applied"))) or bool(x.get("preview_patch_exact"))
            for x in effect_applied if bool(x.get("success"))
        )),
        "restored_targets": sorted(restore_ids),
        "accepted_candidate_targets": sorted(accept_ids),
        "unresolved_candidates": unresolved,
        "unreviewed_recommended": unreviewed_recommended,
        "psd_exported": psd_ok,
        "final": str(final_path),
    })
    return final_path


def _apply_reletter_review(page_dir: Path, project: dict, overrides: dict, cfg: PipelineConfig) -> Path:
    """Apply manual edits to stable TARGET-driven reletter regions only.

    The automatic reletter result remains the pixel authority everywhere outside
    the edited region envelopes.  Each edited region is locally rebuilt from the
    pristine TARGET page, its Japanese text is cleared again, and only then is the
    user-supplied Chinese text rendered with per-region typography overrides.
    Detector/OCR/matching are never rerun here.
    """
    target = read_image(page_dir / "target_original.png")
    # Freeze the pre-review automatic result once. commit_reviewed_result mirrors
    # reviewed pixels to final.png for compatibility, so using final.png directly
    # on a second edit/reset would stack edits and make “恢复自动重排” impossible.
    automatic_path = ensure_manual_baseline(page_dir, preferred_source=page_dir / "final.png")
    if not automatic_path.exists():
        raise FileNotFoundError("automatic reletter baseline is missing")
    automatic = read_image(automatic_path)
    if automatic.shape != target.shape:
        raise ValueError("automatic final size mismatch")

    meta = _dict_or_empty(project.get("meta"))
    rmeta = _dict_or_empty(meta.get("reletter"))
    editable = _dict_rows(rmeta.get("editable_regions"))
    by_region = {str(x.get("target_region_id") or x.get("target_unit_id") or ""): x for x in editable}
    by_unit_region = {str(x.get("target_unit_id") or ""): x for x in editable if x.get("target_unit_id")}

    rows = [x for x in _dict_rows(overrides.get("manual_reletter")) if str(x.get("text", "")).strip()]
    target_units = [_text_unit(x) for x in project.get("target_units", [])]
    target_blocks = [_text_block(x) for x in project.get("target_blocks", [])]
    target_bubbles = _load_target_bubbles(page_dir, project.get("target_bubbles", []))
    target_by_id = {u.id: u for u in target_units}
    bubbles_by_id = {b.id: b for b in target_bubbles}

    out = automatic.copy()
    manual_masks: list[np.ndarray] = []
    applied: list[dict] = []
    failed: list[dict] = []
    changed_mask = np.zeros(target.shape[:2], np.uint8)

    for index, row in enumerate(rows):
        region_key = str(row.get("target_region_id") or row.get("target_unit_id") or row.get("target_bubble_id") or "")
        region = by_region.get(region_key) or by_unit_region.get(str(row.get("target_unit_id") or ""))
        if not region:
            failed.append({"target_region_id": region_key, "reason": "unknown_reletter_region"})
            continue
        unit_id = str(region.get("target_unit_id") or row.get("target_unit_id") or "")
        dst = target_by_id.get(unit_id)
        if dst is None:
            failed.append({"target_region_id": region_key, "reason": "missing_target_unit"})
            continue

        bubble = bubbles_by_id.get(str(dst.bubble_id or ""))
        bubble_safe = None
        if bubble is not None:
            bubble_safe = bubble.safe_mask if bubble.safe_mask is not None and cv2.countNonZero(bubble.safe_mask) > 0 else bubble.mask
        region_limit = rasterize_polygon(dst.polygon, target.shape[:2])
        rb = dst.bbox
        pad = max(3, int(round(min(max(1.0, rb[2]-rb[0]), max(1.0, rb[3]-rb[1])) * 0.18)))
        if cv2.countNonZero(region_limit) > 0 and pad > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
            region_limit = cv2.dilate(region_limit, k, iterations=1)
        if bubble_safe is not None and cv2.countNonZero(bubble_safe) > 0:
            base_safe = cv2.bitwise_and((bubble_safe > 0).astype(np.uint8) * 255, region_limit)
        else:
            base_safe = region_limit
        if cv2.countNonZero(base_safe) == 0:
            base_safe = polygon_safe_mask(dst, target.shape[:2], margin=2)
        if cv2.countNonZero(base_safe) == 0:
            failed.append({"target_region_id": region_key, "reason": "empty_region_safe_mask"})
            continue

        clear_result = build_clear_mask(
            target.shape[:2], target_blocks, target_units, target_bubbles,
            [UnitMatch("manual-source", dst.id, 1.0, 0.0, "one_to_one", ["manual_reletter_region"])],
            cfg.masking, min_match_confidence=0.0, allow_relations={"one_to_one"}, target_image=target,
        )
        clear = clear_result.per_unit.get(dst.id)
        if clear is None or cv2.countNonZero(clear) == 0:
            clear = rasterize_polygon(dst.polygon, target.shape[:2])
            clear = cv2.bitwise_and(clear, base_safe)

        orientation = str(row.get("orientation") or region.get("auto_orientation") or "auto")
        safe = textbox_safe_mask(base_safe, clear, orientation=orientation)
        if safe is None or cv2.countNonZero(safe) == 0:
            safe = base_safe

        text_value = str(row.get("text", "")).strip()
        lcfg = cfg.lettering.model_copy(deep=True) if hasattr(cfg.lettering, "model_copy") else cfg.lettering.copy(deep=True)
        lcfg.orientation = orientation
        font_path = str(row.get("font_path") or "").strip()
        if font_path:
            lcfg.font_path = font_path
        font_size = int(row.get("font_size") or 0)
        if font_size > 0:
            # A positive manual size is an explicit user choice, not a hint.
            lcfg.min_font_size = font_size
            lcfg.max_font_size = font_size
            lcfg.preferred_font_size = font_size
        else:
            # “自动” in the Region editor means re-fit around the automatic
            # typography, not restart from the global maximum size.
            auto_size = int(region.get("auto_font_size") or 0)
            if auto_size > 0:
                lcfg.preferred_font_size = auto_size
                lcfg.preferred_font_tolerance_ratio = max(0.12, min(0.30, float(getattr(lcfg, "preferred_font_tolerance_ratio", 0.22))))
        columns = int(row.get("columns") or 0)
        if columns > 0:
            lcfg.preferred_columns = columns
        elif int(region.get("columns") or 0) > 0:
            lcfg.preferred_columns = int(region.get("columns") or 0)
        line_break_mode = str(row.get("line_break_mode") or "").strip().lower()
        if line_break_mode in {"smart", "balanced", "source"}:
            lcfg.line_break_mode = line_break_mode
        layout_mode = str(row.get("layout_mode") or "").strip().lower()
        if layout_mode in {"strict", "smart_scaling", "balloon_fill"}:
            lcfg.layout_mode = layout_mode
        manual_spacing = row.get("line_spacing_ratio")
        if manual_spacing is not None:
            try: lcfg.line_spacing_ratio = float(np.clip(float(manual_spacing), 0.0, 0.6))
            except Exception: pass
        # Keep manual edits anchored to the original automatic TARGET text island.
        sb = cv2.boundingRect((safe > 0).astype(np.uint8))
        sx, sy, sw, sh = sb
        if sw > 0 and sh > 0:
            dx0, dy0, dx1, dy1 = dst.bbox
            lcfg.anchor_x_ratio = float(np.clip((((dx0+dx1)*0.5)-sx)/sw, 0.05, 0.95))
            lcfg.anchor_y_ratio = float(np.clip((((dy0+dy1)*0.5)-sy)/sh, 0.05, 0.95))
            lcfg.preferred_bbox_width_ratio = float(np.clip((dx1-dx0)/max(1.0,sw), 0.08, 1.0))
            lcfg.preferred_bbox_height_ratio = float(np.clip((dy1-dy0)/max(1.0,sh), 0.08, 1.0))

        edit_unit = TextUnit(
            id=dst.id,
            polygon=list(dst.polygon),
            block_ids=list(dst.block_ids),
            text=text_value,
            confidence=1.0,
            kind=dst.kind,
            reading_order=dst.reading_order,
            bubble_id=dst.bubble_id,
            meta={**dict(dst.meta or {}), "review_manual_reletter": True},
        )
        lr = fit_text(target.shape[:2], safe, edit_unit, text_value, lcfg)
        if not lr.success or lr.text_mask is None:
            # Keep the automatic region intact if a requested manual typography
            # setting cannot fit. Never replace a good automatic result with blank.
            failed.append({
                "target_region_id": str(region.get("target_region_id") or region_key),
                "target_unit_id": dst.id,
                "text": text_value,
                "reason": str(lr.reason or "manual_layout_failed"),
            })
            continue

        # Rebuild just this region from TARGET so old automatic Chinese glyphs do
        # not ghost underneath the edited text. Outside base_safe remains bit-exact.
        local_clean = inpaint_image(target, clear, cfg.inpainting).image
        use = base_safe > 0
        out[use] = target[use]
        cuse = clear > 0
        out[cuse] = local_clean[cuse]
        out = composite_text(out, lr, lcfg)
        changed_mask[use] = 255
        manual_masks.append(lr.text_mask)
        applied.append({
            "target_region_id": str(region.get("target_region_id") or region_key),
            "target_unit_id": dst.id,
            "target_bubble_id": str(dst.bubble_id or ""),
            "text": text_value,
            "orientation": lr.orientation,
            "font_path": lr.font_path,
            "font_size": int(lr.font_size),
            "columns": int(columns),
            "line_break_mode": str(getattr(lcfg, "line_break_mode", "smart")),
            "layout_mode": str(getattr(lcfg, "layout_mode", "smart_scaling")),
            "line_spacing_ratio": float(getattr(lcfg, "line_spacing_ratio", 0.16)),
            "lines": list(lr.lines),
            "bbox": list(lr.bbox),
            "coverage_inside_safe": float(lr.coverage_inside_safe),
        })

    # Preserve the rest of the review toolchain without involving another transfer
    # mode. These are additive/local review operations only.
    out, effect_layer, effect_clear_mask, effect_applied = _apply_manual_effect_regions(out, target, page_dir, project, overrides, cfg)
    out, force_layer, force_clear_mask, force_diag = _apply_manual_force_transfer_mask(page_dir, out, target, project, cfg)
    protect = list(manual_masks)
    if effect_layer is not None and effect_layer.ndim == 3 and effect_layer.shape[2] >= 4:
        protect.append(effect_layer[:, :, 3])
    if force_layer is not None and force_layer.ndim == 3 and force_layer.shape[2] >= 4:
        protect.append(force_layer[:, :, 3])
    out, target_erase_diag = _apply_target_layer_erase_to_rendered(page_dir, out, target, cfg, refresh_base=True, extra_protect_masks=protect)
    out, target_restore_diag = _apply_target_layer_restore_to_rendered(page_dir, out, target, refresh_base=True)

    final_path = page_dir / "final_reviewed.png"
    write_image(final_path, out)
    if manual_masks:
        text_rgba = make_text_layer_rgba(target.shape[:2], manual_masks, color=cfg.lettering.fill)
        write_rgba(page_dir / "text_layer_reviewed.png", text_rgba)

    save_json(page_dir / "review_applied.json", {
        "mode": "reletter_region_review",
        "status": overrides.get("status", "reviewed_with_manual_reletter"),
        "manual_reletter_applied": applied,
        "manual_reletter_failed": failed,
        "edited_region_count": len(applied),
        "edited_region_ids": [x["target_region_id"] for x in applied],
        "changed_region_pixels": int(cv2.countNonZero(changed_mask)),
        "automatic_final_preserved_outside_review": True,
        "manual_effect_applied": effect_applied,
        "manual_effect_clear_pixels": int(cv2.countNonZero(effect_clear_mask)),
        "manual_force_transfer": force_diag,
        "manual_force_clear_pixels": int(cv2.countNonZero(force_clear_mask)),
        "target_layer_erase": target_erase_diag,
        "target_layer_restore": target_restore_diag,
        "final": str(final_path),
    })
    return final_path


@guarded_page_write("force_rerun")
def rerun_page_with_force(page_dir: str | Path, mode: str, config: PipelineConfig | None = None) -> Path:
    """Re-run one page through an explicitly selected Direct/Mask route."""
    page_dir = Path(page_dir)
    if mode not in {"direct_patch", "mask_replace"}:
        raise ValueError(f"unsupported force mode: {mode}")
    project = normalize_project(load_json(page_dir / "project.json"))
    pair = PagePair(**project["pair"])
    cfg = (config or PipelineConfig()).model_copy(deep=True)
    cfg.transfer.mode = mode
    # Remove stale route-specific products before regenerating. This prevents a
    # failed Direct force from leaving an old Mask layer that Review could mistake
    # for the new result.
    for name in (
        "direct_patch_layer.png", "direct_patch_regions.png", "direct_patch.json",
        "mask_transfer_layer.png", "mask_transfer_mask.png", "mask_transfer.json",
        "direct_patch_layer_reviewed.png", "mask_transfer_layer_reviewed.png",
        "final_reviewed.png",
    ):
        path = page_dir / name
        if path.exists():
            path.unlink()
    final_path = None
    book_final = str(as_dict(project.get("artifacts")).get("book_final", "") or "").strip()
    if book_final:
        final_path = Path(book_final)
    from .pipeline import TransferPipeline
    regenerated = TransferPipeline(cfg).process_page(pair, page_dir, final_path=final_path)
    result = page_dir / "final.png"
    save_json(page_dir / "force_action_result.json", {
        "schema": "manga_hd_translation_transfer.force_action.v1",
        "action": f"force_{mode}",
        "passthrough": bool((regenerated.meta or {}).get("passthrough")),
        "planner": (regenerated.meta or {}).get("transfer_planner", {}),
        "final": str(result),
    })
    return result


@guarded_page_write("review_apply")
def apply_review_page(page_dir: str | Path, config: PipelineConfig | None = None) -> Path:
    page_dir = Path(page_dir)
    cfg = config or PipelineConfig()
    override_path = page_dir / "review_overrides.json"
    overrides = normalize_overrides(load_json(override_path) if override_path.exists() else {})
    force_action = str(overrides.get("page_force_action", "") or "")
    if force_action in {"force_direct_patch", "force_mask_replace"}:
        forced_mode = "direct_patch" if force_action == "force_direct_patch" else "mask_replace"
        forced_final = rerun_page_with_force(page_dir, forced_mode, cfg)
        overrides["page_force_action_consumed"] = force_action
        overrides["page_force_action"] = ""
        overrides["owner_transfer_mode"] = forced_mode
        save_json(override_path, overrides)
        project = normalize_project(load_json(page_dir / "project.json"))
        if bool(_dict_or_empty(project.get("meta")).get("passthrough")):
            return forced_final
    else:
        project = normalize_project(load_json(page_dir / "project.json"))
    meta = _dict_or_empty(project.get("meta"))
    project_mode = str(meta.get("transfer_mode", "") or "").strip().lower()
    # Validate mode up-front so review dispatch cannot silently invent a route.
    get_mode_contract(project_mode)
    review_owner = str(overrides.get("owner_transfer_mode", "") or "").strip().lower()
    if review_owner and not review_owner_compatible(review_owner, project_mode):
        raise ValueError(
            f"Review state belongs to transfer mode '{review_owner}', but current page mode is '{project_mode}'. "
            "Recreate or explicitly convert the review state instead of silently mixing modes."
        )
    manual_effect_rows = _dict_rows(overrides.get("manual_effect_regions"))
    manual_force_present = (page_dir / "manual_force_transfer_mask.png").exists()
    if manual_effect_rows or manual_force_present:
        _ensure_manual_effect_stable_base(page_dir)
    # Reletter has its own stable Region editor. It must never be hijacked by a
    # leftover standalone manual-force mask before the reletter dispatcher runs.
    if project_mode == "reletter":
        unit_actions_probe = {str(k): str(v) for k, v in dict(overrides.get("unit_actions", {}) or {}).items()}
        reletter_unit_override = bool(dict(overrides.get("text_overrides", {}) or {})) or bool(dict(overrides.get("match_overrides", {}) or {})) or bool(list(overrides.get("accepted_source_units", []) or [])) or any(
            action in {"force_match", "skip_unit"} for action in unit_actions_probe.values()
        )
        if not reletter_unit_override:
            return _commit_reviewed_result(page_dir, _apply_reletter_review(page_dir, project, overrides, cfg))
    if manual_force_present and project_mode in {"auto", "direct_patch", "mask_replace", "hybrid"} and not ((page_dir / "direct_patch_layer.png").exists() or (page_dir / "mask_transfer_layer.png").exists()):
        return apply_manual_force_transfer_review(page_dir, cfg)
    # Manual omission repair is an additive overlay.  When it is the only
    # visual review operation, never rebuild the whole page from a single
    # Direct/Mask layer; use the already-good automatic final as the base.
    if manual_effect_rows and (
        _manual_effect_can_overlay_final(page_dir, overrides)
        or bool(meta.get("passthrough"))
        or not ((page_dir / "direct_patch_layer.png").exists() or (page_dir / "mask_transfer_layer.png").exists())
    ):
        return _commit_reviewed_result(page_dir, _apply_manual_effect_only_review(page_dir, project, overrides, cfg))
    # Raster-review is the fast/default route for Auto/Mask/Direct pages, but an
    # actual unit-level text/match override means the reviewer explicitly wants
    # regenerated lettering. normalize_overrides() always inserts empty keys, so
    # route on *non-empty values* rather than key presence.
    unit_actions_probe = {str(k): str(v) for k, v in dict(overrides.get("unit_actions", {}) or {}).items()}
    unit_level_override = bool(dict(overrides.get("text_overrides", {}) or {})) or bool(dict(overrides.get("match_overrides", {}) or {})) or bool(list(overrides.get("accepted_source_units", []) or [])) or any(
        action in {"force_match", "skip_unit"} for action in unit_actions_probe.values()
    )
    if meta.get("transfer_mode") in {"mask_replace", "direct_patch", "auto"} and not unit_level_override:
        return _commit_reviewed_result(page_dir, _apply_mask_replace_review(page_dir, project, cfg))

    source_units = [_text_unit(x) for x in project.get("source_units", [])]
    target_units = [_text_unit(x) for x in project.get("target_units", [])]
    target_blocks = [_text_block(x) for x in project.get("target_blocks", [])]
    target_bubbles = _load_target_bubbles(page_dir, project.get("target_bubbles", []))
    target = read_image(page_dir / "target_original.png")

    source_by_id = {u.id: u for u in source_units}
    target_by_id = {u.id: u for u in target_units}
    bubbles_by_id = {b.id: b for b in target_bubbles}

    for source_id, text in dict(overrides.get("text_overrides", {})).items():
        if source_id in source_by_id:
            source_by_id[source_id].text = str(text)

    existing = {}
    for row in project.get("matches", []):
        if row.get("relation") == "one_to_one":
            existing[row["source_unit_id"]] = row["target_unit_id"]
    existing.update({str(k): str(v) for k, v in dict(overrides.get("match_overrides", {})).items()})

    if "accepted_source_units" in overrides:
        accepted_ids = set(map(str, overrides.get("accepted_source_units", [])))
    else:
        accepted_ids = {
            x.split("->", 1)[0]
            for x in _dict_or_empty(project.get("meta")).get("auto_applied_match_ids", [])
            if "->" in x
        }
    unit_actions = {str(k): str(v) for k, v in dict(overrides.get("unit_actions", {}) or {}).items()}
    for source_id, action in unit_actions.items():
        if action == "force_match":
            accepted_ids.add(source_id)
        elif action == "skip_unit":
            accepted_ids.discard(source_id)
    matches: list[UnitMatch] = []
    for source_id in accepted_ids:
        target_id = existing.get(source_id)
        if source_id in source_by_id and target_id in target_by_id:
            matches.append(UnitMatch(source_id, target_id, 1.0, 0.0, "one_to_one", ["review_accepted"]))

    manual_mask = page_dir / "manual_clear_mask.png"
    additive_mask = page_dir / "manual_japanese_clear_mask.png"
    if manual_mask.exists() or additive_mask.exists():
        mask, _clear_source = _load_effective_clear_mask(page_dir, target.shape[:2])
        mask, _brush_diag = _apply_manual_brush_expansion(page_dir, mask)
        write_image(page_dir / "effective_clear_mask.png", mask)
        from .masking import MaskBuildResult
        mask_result = MaskBuildResult((mask > 0).astype("uint8") * 255, {}, 0, int(cv2.countNonZero(mask)))
    else:
        mask_result = build_clear_mask(
            target.shape[:2], target_blocks, target_units, target_bubbles, matches, cfg.masking,
            min_match_confidence=0.0, allow_relations={"one_to_one"}, target_image=target
        )

    inpainted = inpaint_image(target, mask_result.mask, cfg.inpainting).image
    rendered = inpainted.copy()
    lettering = []
    masks = []
    for match in matches:
        src = source_by_id[match.source_unit_id]
        dst = target_by_id[match.target_unit_id]
        safe = bubbles_by_id.get(dst.bubble_id).safe_mask if dst.bubble_id and dst.bubble_id in bubbles_by_id else None
        if safe is None or cv2.countNonZero(safe) == 0:
            safe = polygon_safe_mask(dst, target.shape[:2], margin=max(2, cfg.bubbles.safe_margin_px // 2))
        lr = fit_text(target.shape[:2], safe, dst, src.text, cfg.lettering)
        lettering.append(lr)
        if lr.success and lr.text_mask is not None:
            rendered = composite_text(rendered, lr, cfg.lettering)
            masks.append(lr.text_mask)

    rendered, effect_layer, effect_clear_mask, effect_applied = _apply_manual_effect_regions(rendered, target, page_dir, project, overrides, cfg)
    if cv2.countNonZero(effect_clear_mask) > 0:
        effect_clean = inpaint_image(target, effect_clear_mask, cfg.inpainting).image
        inpainted[effect_clear_mask > 0] = effect_clean[effect_clear_mask > 0]
    rendered, force_layer, force_clear_mask, force_diag = _apply_manual_force_transfer_mask(page_dir, rendered, target, project, cfg)
    erase_extra = list(masks)
    if effect_layer is not None and effect_layer.ndim == 3 and effect_layer.shape[2] >= 4:
        erase_extra.append(effect_layer[:, :, 3])
    if force_layer is not None and force_layer.ndim == 3 and force_layer.shape[2] >= 4:
        erase_extra.append(force_layer[:, :, 3])
    rendered, target_erase_diag = _apply_target_layer_erase_to_rendered(
        page_dir, rendered, target, cfg, refresh_base=True, extra_protect_masks=erase_extra
    )
    rendered, target_restore_diag = _apply_target_layer_restore_to_rendered(
        page_dir, rendered, target, refresh_base=True
    )
    final_path = page_dir / "final_reviewed.png"
    inpainted_path = page_dir / "inpainted_reviewed.png"
    text_path = page_dir / "text_layer_reviewed.png"
    write_image(final_path, rendered)
    write_image(inpainted_path, inpainted)
    text_rgba = make_text_layer_rgba(target.shape[:2], masks, color=cfg.lettering.fill)
    write_rgba(text_path, text_rgba)
    transfer_rgba = cv2.cvtColor(effect_layer, cv2.COLOR_BGRA2RGBA) if np.any(effect_layer[:, :, 3] > 0) else None
    export_openraster(page_dir / "editable_reviewed.ora", target, inpainted, text_rgba, transfer_rgba)
    transfer_path = page_dir / "manual_effect_transfer_layer.png" if transfer_rgba is not None else None
    psd_ok = export_psd_imagemagick(page_dir / "editable_reviewed.psd", page_dir / "target_original.png", inpainted_path, text_path, transfer_path)
    save_json(
        page_dir / "review_applied.json",
        {
            "status": overrides.get("status", "reviewed"),
            "notes": overrides.get("notes", ""),
            "accepted_source_units": sorted(accepted_ids),
            "matches": [m.to_dict() for m in matches],
            "lettering": [x.to_dict() for x in lettering],
            "manual_mask": manual_mask.exists(),
            "manual_effect_applied": effect_applied,
            "manual_effect_clear_pixels": int(cv2.countNonZero(effect_clear_mask)),
            "manual_force_transfer": force_diag,
            "manual_force_clear_pixels": int(cv2.countNonZero(force_clear_mask)),
            "target_layer_erase": target_erase_diag,
        "target_layer_restore": target_restore_diag,
            "psd_exported": psd_ok,
            "final": str(final_path),
        },
    )
    return _commit_reviewed_result(page_dir, final_path)
