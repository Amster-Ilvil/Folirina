from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import cv2
import numpy as np

from .config import PipelineConfig
from .export import export_openraster, export_psd_imagemagick, make_text_layer_rgba, write_rgba
from .inpainting import inpaint_image
from .io_utils import load_json, read_image, save_json, write_image
from .lettering import composite_text, fit_text, polygon_safe_mask
from .masking import build_clear_mask
from .manual_effect import build_manual_effect_masks, apply_reveal_window, estimate_source_background, composite_source_text_delta
from .models import BubbleInstance, PagePair, TextBlock, TextUnit, UnitMatch
from .text_only_transfer import clear_to_target_paper, target_text_mask_in_container
from .schema_compat import as_dict, as_dict_rows, as_list, normalize_project, normalize_overrides, normalize_route_meta
from .result_state import ensure_manual_baseline, manual_baseline_path, commit_reviewed_result


def _dict_or_empty(value):
    """Return a plain dict for mixed/legacy JSON schema values."""
    return as_dict(value)


def _route_meta(meta, key: str) -> dict:
    return normalize_route_meta(_dict_or_empty(meta).get(key))


def _dict_rows(value) -> list[dict]:
    """Normalize a stale list-like review field to dictionary rows only."""
    return as_dict_rows(value)


def _text_block(row: dict) -> TextBlock:
    return TextBlock(**row)


def _text_unit(row: dict) -> TextUnit:
    return TextUnit(**row)


def _load_target_bubbles(page_dir: Path, rows: list[dict]) -> list[BubbleInstance]:
    out = []
    for row in rows:
        b = BubbleInstance(
            id=row["id"],
            polygon=row["polygon"],
            confidence=row.get("confidence", 1.0),
            kind=row.get("kind", "speech"),
            block_ids=list(row.get("block_ids", [])),
            meta=as_dict(row.get("meta")),
        )
        mp = page_dir / "bubbles" / f"{b.id}.png"
        sp = page_dir / "bubbles" / f"{b.id}_safe.png"
        if mp.exists():
            b.mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if sp.exists():
            b.safe_mask = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
        out.append(b)
    return out


def _rect_mask(shape: tuple[int, int], bbox: list[int] | tuple[int, int, int, int], inset: int = 2) -> np.ndarray:
    x0, y0, x1, y1 = map(int, bbox)
    x0 = max(0, min(shape[1], x0 + inset)); y0 = max(0, min(shape[0], y0 + inset))
    x1 = max(0, min(shape[1], x1 - inset)); y1 = max(0, min(shape[0], y1 - inset))
    mask = np.zeros(shape, np.uint8)
    if x1 > x0 and y1 > y0:
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask


def _clear_region_to_paper(rendered: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rendered.copy()
    if mask is None or cv2.countNonZero(mask) == 0:
        return out
    sel = mask > 0
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    bright = sel & (gray >= 205)
    if np.count_nonzero(bright) >= 20:
        paper = np.median(target[bright], axis=0).astype(np.uint8)
    else:
        paper = np.array([255, 255, 255], np.uint8)
    out[sel] = paper
    return out


def _load_effective_clear_mask(page_dir: Path, shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    """Load the page clear-mask overlay used by independent remove-text review.

    A manually edited mask is authoritative. Otherwise use the exact transfer
    clear mask emitted by the pipeline, falling back to the legacy clear mask.
    """
    candidates = [
        (page_dir / "manual_clear_mask.png", "manual_clear_mask"),
        (page_dir / "target_clear_mask.png", "target_clear_mask"),
        (page_dir / "clear_mask.png", "clear_mask"),
    ]
    for path, label in candidates:
        if not path.exists():
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != shape:
            if label == "manual_clear_mask":
                raise ValueError("manual_clear_mask.png has invalid dimensions")
            continue
        return (mask > 0).astype(np.uint8) * 255, label
    return np.zeros(shape, np.uint8), "none"


def generate_remove_text_preview(page_dir: str | Path, config: PipelineConfig | None = None) -> Path:
    """Run only the target-text removal stage for the current page.

    This is deliberately independent from Chinese raster transfer. It lets the
    user inspect/repair the clear mask without rerunning registration, detection,
    OCR or glyph placement. The result is safe to regenerate repeatedly.
    """
    page_dir = Path(page_dir)
    cfg = config or PipelineConfig()
    target = read_image(page_dir / "target_original.png")
    mask, source = _load_effective_clear_mask(page_dir, target.shape[:2])
    if cv2.countNonZero(mask) == 0:
        preview = target.copy()
        backend = "none"
    else:
        result = inpaint_image(target, mask, cfg.inpainting)
        preview = result.image
        backend = str(getattr(result, "method", getattr(cfg.inpainting, "backend", "auto")))
    out = page_dir / "removed_text_preview.png"
    write_image(out, preview)
    save_json(page_dir / "remove_text_stage.json", {
        "schema": "manga_hd_translation_transfer.remove_text_stage.v1",
        "mask_source": source,
        "mask_pixels": int(cv2.countNonZero(mask)),
        "inpainting_backend": backend,
        "output": str(out),
    })
    return out



def _source_for_review(page_dir: Path, project: dict) -> np.ndarray:
    local = page_dir / "source_original.png"
    if local.exists():
        return read_image(local)
    pair = dict(project.get("pair", {}) or {})
    source_path = str(pair.get("source_path", "") or "")
    if not source_path:
        raise FileNotFoundError("manual effect transfer needs source_original.png or pair.source_path")
    return read_image(source_path)


def _write_bgra(path: Path, bgra: np.ndarray) -> None:
    ok, data = cv2.imencode(".png", bgra)
    if not ok:
        raise ValueError(f"could not encode {path.name}")
    data.tofile(path)


def _alpha_over_bgra(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Alpha-composite one BGRA manual layer over an existing BGRA layer."""
    if bottom.shape != top.shape:
        raise ValueError("manual effect layer size mismatch")
    ba = bottom[:, :, 3].astype(np.float32) / 255.0
    ta = top[:, :, 3].astype(np.float32) / 255.0
    out_a = ta + ba * (1.0 - ta)
    out = np.zeros_like(bottom)
    denom = np.maximum(out_a, 1e-6)
    for c in range(3):
        bc = bottom[:, :, c].astype(np.float32)
        tc = top[:, :, c].astype(np.float32)
        out[:, :, c] = np.clip((tc * ta + bc * ba * (1.0 - ta)) / denom, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)
    return out


def _load_reveal_commit_patch(page_dir: Path, row: dict, shape: tuple[int, int]) -> np.ndarray | None:
    """Load an exact sparse preview patch saved by the Qt Reveal editor.

    Older projects do not have this artifact and transparently fall back to
    recomputing the effect from SOURCE/TARGET masks.
    """
    name = str(row.get("reveal_patch_file", "") or "").strip()
    if not name:
        return None
    path = page_dir / name
    if not path.exists():
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
    out = rendered.copy()
    effect_layer = np.zeros((h, w, 4), np.uint8)
    all_clear = np.zeros((h, w), np.uint8)
    applied: list[dict] = []
    for index, row in enumerate(rows):
        try:
            masks = build_manual_effect_masks(source, target, project, row)
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
            mask_path = page_dir / mask_name if mask_name else None
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
                # The sparse preview patch is authoritative. Its alpha is further
                # gated by the accepted Reveal mask so manually erased areas can
                # never reappear from a stale patch artifact.
                # For monochrome SOURCE -> colour TARGET pages, however, an old
                # patch may have been rendered with a broad colour-difference
                # erase mask. Rebuild only its RGB values from the current,
                # colour-protected masks while retaining the user's brush alpha.
                # This makes an existing saved review self-heal after the mask
                # policy is tightened, without requiring the user to repaint.
                if bool(masks.diagnostics.get("color_container_protection")):
                    rebuilt = target.copy()
                    if cv2.countNonZero(clear_mask) > 0:
                        cleaned = inpaint_image(target, clear_mask, cfg.inpainting).image
                        rebuilt[clear_mask > 0] = cleaned[clear_mask > 0]
                    source_bg = estimate_source_background(masks.aligned_source, full_source_mask)
                    rebuilt, _ = composite_source_text_delta(
                        rebuilt, masks.aligned_source, source_mask,
                        source_background=source_bg,
                    )
                    reveal_patch = reveal_patch.copy()
                    reveal_patch[:, :, :3] = rebuilt
                    diag_color_rebuilt = True
                else:
                    diag_color_rebuilt = False
                # The reveal patch contains the SOURCE text layer, while the
                # accepted clear mask is the TARGET Japanese-removal layer.
                # Apply the latter explicitly before compositing the patch;
                # otherwise Japanese ink remains visible underneath Chinese
                # whenever the two masks do not have identical alpha coverage.
                patch_alpha = np.minimum(reveal_patch[:, :, 3], reveal).astype(np.uint8)
                reveal_clear = (clear_mask > 0) & (reveal > 0)
                if np.any(reveal_clear) and np.any(patch_alpha > 0):
                    cleared_target = inpaint_image(target, clear_mask, cfg.inpainting).image
                    out[reveal_clear] = cleared_target[reveal_clear]
                patch_sel = patch_alpha > 0
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
                    "color_protected_patch_rebuilt": bool(diag_color_rebuilt),
                    "preview_patch_exact": bool(np.array_equal(out[patch_sel], reveal_patch[:, :, :3][patch_sel])) if np.any(patch_sel) else False,
                    "diagnostics": diag,
                })
                continue
        if mode in {"full_patch", "white_bubble_text"}:
            # Manual white-bubble correction is replacement, not additive.  The
            # current base may already contain an automatically transferred CN
            # layer, so clear both TARGET JP text and any existing rendered text
            # inside the confirmed white container before drawing the nudged CN.
            region=np.zeros((h,w),np.uint8)
            bx=as_list(row.get("target_bbox"))
            if len(bx)==4:
                rx0,ry0,rx1,ry1=[int(v) for v in bx]
                rx0=max(0,min(w,rx0));rx1=max(0,min(w,rx1));ry0=max(0,min(h,ry0));ry1=max(0,min(h,ry1))
                if rx1>rx0 and ry1>ry0: region[ry0:ry1,rx0:rx1]=255
            current_text=target_text_mask_in_container(out,region) if cv2.countNonZero(region) else np.zeros((h,w),np.uint8)
            white_clear=cv2.bitwise_or(clear_mask,source_mask)
            white_clear=cv2.bitwise_or(white_clear,current_text)
            if cv2.countNonZero(white_clear):
                white_clear=cv2.dilate(white_clear,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
                white_clear[region==0]=0
                out=clear_to_target_paper(out,target,white_clear)
                all_clear=np.maximum(all_clear,white_clear)
        elif cv2.countNonZero(clear_mask) > 0:
            cleaned = inpaint_image(target, clear_mask, cfg.inpainting).image
            sel = clear_mask > 0
            out[sel] = cleaned[sel]
            all_clear = np.maximum(all_clear, clear_mask)

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
        "manual_effect_preview_patch_verified": bool(all(
            (not bool(x.get("preview_patch_applied"))) or bool(x.get("preview_patch_exact"))
            for x in applied if bool(x.get("success"))
        )),
        "manual_effect_base": str(base_path),
        "psd_exported": psd_ok,
        "final": str(final_path),
    })
    return final_path


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
            "orientation": orientation,
            "bbox": list(lr.bbox),
        })
    return out, masks, applied


def _apply_mask_replace_review(page_dir: Path, project: dict, cfg: PipelineConfig) -> Path:
    target = read_image(page_dir / "target_original.png")
    overrides_probe = normalize_overrides(load_json(page_dir / "review_overrides.json") if (page_dir / "review_overrides.json").exists() else {})
    automatic_path = _manual_effect_overlay_base_path(page_dir) if _dict_rows(overrides_probe.get("manual_effect_regions")) else page_dir / "final.png"
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
    effective_clear, clear_source = _load_effective_clear_mask(page_dir, target.shape[:2])
    manual_clear = page_dir / "manual_clear_mask.png"
    if manual_clear.exists():
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
    if manual_clear.exists() and cv2.countNonZero(effective_clear) > 0:
        base = inpaint_image(target, effective_clear, cfg.inpainting).image
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
    write_image(final_path, rendered)

    # Keep editable exports faithful to the reviewed result: target-only Japanese
    # strokes removed by a manual open-text region also disappear from the base,
    # while the recovered Chinese raster is merged into the transfer layer.
    if cv2.countNonZero(effect_clear_mask) > 0:
        effect_clean = inpaint_image(target, effect_clear_mask, cfg.inpainting).image
        base[effect_clear_mask > 0] = effect_clean[effect_clear_mask > 0]
        write_image(reviewed_base_path, base)
    merged_bgra = _alpha_over_bgra(reviewed_bgra, effect_layer) if np.any(effect_layer[:, :, 3] > 0) else reviewed_bgra
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
        "clear_mask_source": clear_source,
        "review_change_pixels": int(cv2.countNonZero(review_change_mask)),
        "automatic_final_preserved_outside_review": bool(automatic is not None),
        "manual_reletter_applied": manual_applied,
        "manual_effect_applied": effect_applied,
        "manual_effect_clear_pixels": int(cv2.countNonZero(effect_clear_mask)),
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
        save_json(override_path, overrides)
        project = normalize_project(load_json(page_dir / "project.json"))
        if bool(_dict_or_empty(project.get("meta")).get("passthrough")):
            return forced_final
    else:
        project = normalize_project(load_json(page_dir / "project.json"))
    manual_effect_rows = _dict_rows(overrides.get("manual_effect_regions"))
    if manual_effect_rows:
        _ensure_manual_effect_stable_base(page_dir)
    meta = _dict_or_empty(project.get("meta"))
    # Manual omission repair is an additive overlay.  When it is the only
    # visual review operation, never rebuild the whole page from a single
    # Direct/Mask layer; use the already-good automatic final as the base.
    if manual_effect_rows and (
        _manual_effect_can_overlay_final(page_dir, overrides)
        or bool(meta.get("passthrough"))
        or not ((page_dir / "direct_patch_layer.png").exists() or (page_dir / "mask_transfer_layer.png").exists())
    ):
        return _commit_reviewed_result(page_dir, _apply_manual_effect_only_review(page_dir, project, overrides, cfg))
    if meta.get("transfer_mode") in {"mask_replace", "direct_patch", "auto"}:
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
    if manual_mask.exists():
        mask = cv2.imread(str(manual_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != target.shape[:2]:
            raise ValueError("manual_clear_mask.png has invalid dimensions")
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
            "psd_exported": psd_ok,
            "final": str(final_path),
        },
    )
    return _commit_reviewed_result(page_dir, final_path)
