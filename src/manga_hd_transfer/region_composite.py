from __future__ import annotations

"""Region-level compositing adapters for the review workbench.

These actions are explicitly manual. They do not change or dispatch the page's
automatic transfer mode. Each action is clipped by a TARGET-space selection mask
and can therefore be stacked with actions from other algorithms in review order.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import TextUnit
from .region_selection import selection_mask_from_row, bbox_from_mask
from . import manual_effect as legacy_ops
from .modes.mask_replace import open_text_manual as mask_ops
from .modes.mask_replace import transfer_ops as mask_transfer_ops
from .modes.hybrid.lettering_ops import fit_text, composite_text
from .review_artifacts import safe_page_artifact_path

REGION_MODES = {
    "region_direct_patch",
    "region_precise_mask",
    "region_hole_reveal",
    "region_transparent",
    "region_ocr",
    "region_brush_reveal",
}


def is_region_mode(mode: str) -> bool:
    return str(mode or "").strip().lower() in REGION_MODES


def _layer_from_rgb(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    out = np.zeros((rgb.shape[0], rgb.shape[1], 4), np.uint8)
    out[:, :, :3] = rgb
    out[:, :, 3] = np.asarray(alpha, np.uint8)
    return out


def _alpha_composite(base: np.ndarray, top_rgb: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha_u8, np.float32)[:, :, None] / 255.0
    return np.clip(base.astype(np.float32) * (1.0 - a) + top_rgb.astype(np.float32) * a, 0, 255).astype(np.uint8)


def _feather(mask: np.ndarray, amount: int) -> np.ndarray:
    px = max(0, min(8, int(amount)))
    if px <= 0:
        return np.asarray(mask, np.uint8)
    alpha = np.asarray(mask, np.float32) / 255.0
    sigma = max(0.45, px * 0.65)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
    alpha[mask > 0] = np.maximum(alpha[mask > 0], 0.96)
    # Feather inward only: the manual selection is a hard authority boundary.
    alpha[mask == 0] = 0.0
    return np.clip(alpha * 255.0, 0, 255).astype(np.uint8)


def _polygon_mask(shape: tuple[int, int], polygons: list[Any]) -> np.ndarray:
    h, w = shape; out = np.zeros((h, w), np.uint8)
    for poly in polygons or []:
        pts = np.asarray(poly, np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, max(0, w - 1)); pts[:, 1] = np.clip(pts[:, 1], 0, max(0, h - 1))
        cv2.fillPoly(out, [np.round(pts).astype(np.int32)], 255)
    if cv2.countNonZero(out):
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return out


def _roi_bounds(mask: np.ndarray, *, halo: int = 0) -> tuple[int, int, int, int]:
    box = bbox_from_mask(mask)
    if len(box) != 4:
        raise ValueError("区域工具没有有效选区")
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    pad = max(0, int(halo))
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def _aligned_source_roi(source: np.ndarray, target_shape: tuple[int, int], project: dict[str, Any],
                        roi: tuple[int, int, int, int], *, source_offset_x: int = 0,
                        source_offset_y: int = 0) -> tuple[np.ndarray, bool]:
    """Warp SOURCE directly into a TARGET-space ROI instead of a full page."""
    th, tw = map(int, target_shape)
    xa, ya, xb, yb = [int(v) for v in roi]
    rw, rh = max(1, xb - xa), max(1, yb - ya)
    H = mask_ops.registration_homography(project).copy()
    dx, dy = int(source_offset_x), int(source_offset_y)
    identity = bool(
        source.shape[:2] == (th, tw) and not dx and not dy
        and np.max(np.abs(np.asarray(H, np.float64) - np.eye(3))) <= 1e-7
    )
    if identity:
        return source[ya:yb, xa:xb].copy(), True
    if dx or dy:
        T = np.asarray([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]], np.float64)
        H = T @ H
    C = np.asarray([[1.0, 0.0, -float(xa)], [0.0, 1.0, -float(ya)], [0.0, 0.0, 1.0]], np.float64)
    crop = cv2.warpPerspective(
        source, C @ H, (rw, rh), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, False


def _precise_mask_roi_halo(selection: np.ndarray, cfg: Any) -> int:
    box = bbox_from_mask(selection)
    if len(box) != 4:
        return 32
    x0, y0, x1, y1 = box
    mc = getattr(cfg, "mask_replace", cfg)
    base = max(1, int(max(x1 - x0, y1 - y0)))
    inner = max(6, int(round(base * float(getattr(mc, "paired_diff_complex_region_pad_ratio", 0.16)))))
    tol = max(1, int(getattr(mc, "paired_diff_ink_tolerance_px", 2)))
    gap = max(1, int(getattr(mc, "paired_diff_complex_group_gap_px", 5)))
    clear = max(1, int(getattr(mc, "paired_diff_complex_clear_dilate_px", 2)))
    return max(224, inner + tol * 3 + gap * 2 + clear * 2 + 14)


def _expand_roi_mask(value: Any, shape: tuple[int, int], roi: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape; xa, ya, xb, yb = roi
    out = np.zeros((h, w), np.uint8)
    if isinstance(value, np.ndarray) and value.shape[:2] == (yb - ya, xb - xa):
        out[ya:yb, xa:xb] = np.asarray(value, np.uint8)
    return out


def _fallback_text_mask(target: np.ndarray, safe: np.ndarray) -> np.ndarray:
    ys, xs = np.where(safe > 0)
    out = np.zeros(safe.shape, np.uint8)
    if xs.size == 0:
        return out
    x0, x1 = int(xs.min()), int(xs.max()) + 1; y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = target[y0:y1, x0:x1]; gate = safe[y0:y1, x0:x1] > 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    values = gray[gate]
    if values.size == 0:
        return out
    threshold = min(180, int(np.percentile(values, 35)) + 24)
    cand = ((gray < threshold) & gate).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    keep = np.zeros_like(cand); area_total = max(1, int(np.count_nonzero(gate)))
    for lab in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area < 2 or area > area_total * 0.10:
            continue
        if bw > crop.shape[1] * 0.78 or bh > crop.shape[0] * 0.78:
            continue
        keep[labels == lab] = 1
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    out[y0:y1, x0:x1] = keep * 255; out[safe == 0] = 0
    return out


def apply_region_action(current: np.ndarray, target: np.ndarray, source: np.ndarray,
                        project: dict[str, Any], row: dict[str, Any], cfg: Any,
                        *, page_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mode = str(row.get("mode") or "").strip().lower()
    if mode not in REGION_MODES:
        raise ValueError(f"unknown region action: {mode}")
    h, w = target.shape[:2]
    out = current.copy(); clear = np.zeros((h, w), np.uint8); layer = np.zeros((h, w, 4), np.uint8)

    # Brush reveal commits are sparse exact-preview patches.  They deliberately
    # do not require a rectangle selection: the painted alpha itself is the
    # authority mask and may span arbitrary disconnected parts of the page.
    if mode == "region_brush_reveal":
        root = Path(page_dir) if page_dir is not None else None
        name = str(row.get("reveal_patch_file") or "").strip()
        box = [int(v) for v in list(row.get("reveal_patch_bbox") or row.get("target_bbox") or [])]
        if root is None or not name or len(box) != 4:
            raise ValueError("涂抹揭示补丁缺少文件或范围")
        x0, y0, x1, y1 = box
        x0=max(0,min(w,x0)); x1=max(0,min(w,x1)); y0=max(0,min(h,y0)); y1=max(0,min(h,y1))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("涂抹揭示补丁范围无效")
        patch_path = safe_page_artifact_path(root, name)
        if patch_path is None:
            raise ValueError("涂抹揭示补丁路径无效")
        patch = cv2.imread(str(patch_path), cv2.IMREAD_UNCHANGED)
        if patch is None or patch.ndim != 3 or patch.shape[2] != 4 or patch.shape[:2] != (y1-y0, x1-x0):
            raise ValueError("涂抹揭示补丁缺失或尺寸不一致")
        alpha = np.asarray(patch[:, :, 3], np.uint8).copy()
        mask_name = str(row.get("reveal_mask_file") or "").strip()
        if mask_name:
            mask_path = safe_page_artifact_path(root, mask_name)
            if mask_path is None:
                raise ValueError("涂抹揭示 authority mask 路径无效")
            authority = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if authority is None or authority.shape != alpha.shape:
                raise ValueError("涂抹揭示 authority mask 缺失或尺寸不一致")
            alpha[authority == 0] = 0
            patch = patch.copy(); patch[:, :, 3] = alpha
        if cv2.countNonZero(alpha) <= 0:
            raise ValueError("涂抹揭示补丁为空")
        base_roi = out[y0:y1, x0:x1].copy()
        out[y0:y1, x0:x1] = _alpha_composite(base_roi, patch[:, :, :3], alpha)
        layer[y0:y1, x0:x1] = patch
        # Do not report the brush alpha as a TARGET inpaint mask.  The sparse
        # patch already represents the exact top-layer transparency/cut-out;
        # feeding this alpha into review_base inpainting would unnecessarily
        # alter manga background underneath a reveal stroke.
        changed = np.any(out[y0:y1, x0:x1] != base_roi, axis=2)
        rec = {
            "id": str(row.get("id") or "region-brush-reveal"),
            "success": True,
            "mode": mode,
            "region_composite": True,
            "brush_reveal": True,
            "target_bbox": [x0,y0,x1,y1],
            "write_pixels": int(cv2.countNonZero(alpha)),
            "changed_pixels": int(np.count_nonzero(changed)),
            "transparent_pixels": int(row.get("transparent_pixels", 0) or 0),
            "hole_pixels": int(row.get("hole_pixels", 0) or 0),
            "preview_patch_replayed": True,
            "diagnostics": {"sparse_patch": True, "patch_shape": [int(y1-y0), int(x1-x0)]},
        }
        return out, layer, clear, rec

    selection = selection_mask_from_row(row, (h, w))
    selected_px = int(cv2.countNonZero(selection))
    if selected_px <= 0:
        raise ValueError("区域工具没有有效选区")
    bbox = bbox_from_mask(selection)
    diag: dict[str, Any] = {"selection_pixels": selected_px, "selection_bbox": bbox}

    if mode in {"region_direct_patch", "region_hole_reveal"}:
        roi = _roi_bounds(selection, halo=max(10, int(row.get("feather_px", 0) or 0) * 4))
        xa, ya, xb, yb = roi
        aligned_crop, identity = _aligned_source_roi(
            source, target.shape[:2], project, roi,
            source_offset_x=int(row.get("source_offset_x", 0) or 0),
            source_offset_y=int(row.get("source_offset_y", 0) or 0),
        )
        effective = selection[ya:yb, xa:xb].copy()
        if mode == "region_hole_reveal":
            inset = max(0, min(12, int(row.get("inset_px", 1) or 0)))
            if inset:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1))
                effective = cv2.erode(effective, k, iterations=1)
        alpha_crop = _feather(effective, int(row.get("feather_px", 0 if mode == "region_direct_patch" else 1) or 0))
        current_crop = out[ya:yb, xa:xb].copy()
        out[ya:yb, xa:xb] = _alpha_composite(current_crop, aligned_crop, alpha_crop)
        layer_crop = _layer_from_rgb(aligned_crop, alpha_crop); layer[ya:yb, xa:xb] = layer_crop
        diag.update({
            "identity_pixel_lock": bool(identity), "write_pixels": int(cv2.countNonZero(alpha_crop)),
            "inset_px": int(row.get("inset_px", 0) or 0), "roi_fast_path": True,
            "roi_bbox": [xa, ya, xb, yb], "roi_fraction": float(((xb-xa)*(yb-ya))/max(1,h*w)),
        })

    elif mode == "region_precise_mask":
        halo = _precise_mask_roi_halo(selection, cfg)
        roi = _roi_bounds(selection, halo=halo); xa, ya, xb, yb = roi
        aligned_crop, identity = _aligned_source_roi(
            source, target.shape[:2], project, roi,
            source_offset_x=int(row.get("source_offset_x", 0) or 0),
            source_offset_y=int(row.get("source_offset_y", 0) or 0),
        )
        target_crop = target[ya:yb, xa:xb]
        selection_crop = selection[ya:yb, xa:xb]
        cfg_obj = getattr(cfg, "mask_replace", cfg)
        rendered_crop, write_crop, src_crop, raw_diag = mask_transfer_ops._transfer_open_complex_text_region_full(
            aligned_crop, target_crop, selection_crop, cfg_obj
        )
        if rendered_crop is None:
            raise ValueError(str((raw_diag or {}).get("reason") or "精准蒙版区域处理失败"))
        write_crop = cv2.bitwise_and(np.asarray(write_crop, np.uint8), selection_crop)
        src_crop = cv2.bitwise_and(np.asarray(src_crop, np.uint8), selection_crop)
        clear_crop = cv2.bitwise_and(
            np.asarray((raw_diag or {}).get("clear_mask", np.zeros_like(selection_crop)), np.uint8), selection_crop
        )
        use = write_crop > 0
        out_roi = out[ya:yb, xa:xb].copy(); out_roi[use] = rendered_crop[use]; out[ya:yb, xa:xb] = out_roi
        clear[ya:yb, xa:xb] = clear_crop
        layer_crop = _layer_from_rgb(aligned_crop, src_crop); layer[ya:yb, xa:xb] = layer_crop
        clean_diag = {k:v for k,v in dict(raw_diag or {}).items() if not isinstance(v, np.ndarray)}
        diag.update(clean_diag)
        diag.update({
            "manual_open_text_box": True, "identity_pixel_lock": bool(identity),
            "write_pixels": int(cv2.countNonZero(write_crop)), "ocr_used": False,
            "roi_fast_path": True, "roi_bbox": [xa, ya, xb, yb], "roi_halo": int(halo),
            "roi_fraction": float(((xb-xa)*(yb-ya))/max(1,h*w)),
        })

    elif mode == "region_transparent":
        request = deepcopy(row); request["mode"] = "effect_text"
        masks = mask_ops.build_manual_effect_masks(source, target, project, request, cfg)
        src_mask = cv2.bitwise_and(masks.source_mask, selection)
        clear = cv2.bitwise_and(masks.target_clear_mask, selection)
        if cv2.countNonZero(clear):
            cleaned, clean_diag = mask_ops.clean_manual_target_text(target, clear, bbox=bbox)
            use = clear > 0; out[use] = cleaned[use]; diag["target_cleanup"] = clean_diag
        if cv2.countNonZero(src_mask):
            bg = mask_ops.estimate_source_background(masks.aligned_source, masks.source_mask)
            alpha = _feather(src_mask, int(row.get("feather_px", 0) or 0)).astype(np.float32) / 255.0
            out, delta_diag = mask_ops.composite_source_text_delta(out, masks.aligned_source, src_mask, source_background=bg, alpha=alpha)
            layer = _layer_from_rgb(masks.aligned_source, np.clip(alpha * 255.0, 0, 255).astype(np.uint8))
            diag["delta_composite"] = delta_diag
        diag.update({"source_pixels": int(cv2.countNonZero(src_mask)), "target_clear_pixels": int(cv2.countNonZero(clear)), "ocr_used": False})

    elif mode == "region_ocr":
        text = str(row.get("render_text") or row.get("ocr_text") or "").strip()
        if not text:
            raise ValueError("OCR 区域没有可排版的中文文本")
        clear = cv2.bitwise_and(_polygon_mask((h, w), list(row.get("target_ocr_polygons") or [])), selection)
        if cv2.countNonZero(clear) == 0:
            clear = _fallback_text_mask(target, selection)
        if cv2.countNonZero(clear):
            cleaned, clean_diag = mask_ops.clean_manual_target_text(target, clear, bbox=bbox)
            use = clear > 0; out[use] = cleaned[use]; diag["target_cleanup"] = clean_diag
        lcfg = cfg.lettering.model_copy(deep=True)
        lcfg.orientation = str(row.get("orientation") or "auto")
        font_path = str(row.get("font_path") or "").strip()
        if font_path:
            lcfg.font_path = font_path
        font_size = int(row.get("font_size") or 0)
        if font_size > 0:
            lcfg.min_font_size = font_size; lcfg.max_font_size = font_size; lcfg.preferred_font_size = font_size
        columns = int(row.get("columns") or 0)
        if columns > 0:
            lcfg.preferred_columns = columns
        if str(row.get("line_break_mode") or "smart") in {"smart", "balanced", "source"}:
            lcfg.line_break_mode = str(row.get("line_break_mode") or "smart")
        if str(row.get("layout_mode") or "smart_scaling") in {"strict", "smart_scaling", "balloon_fill"}:
            lcfg.layout_mode = str(row.get("layout_mode") or "smart_scaling")
        x0, y0, x1, y1 = bbox
        unit = TextUnit(
            id=str(row.get("id") or "region-ocr"), polygon=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)],
            block_ids=[], text=text, confidence=float(row.get("confidence") or 1.0), kind="speech", reading_order=0,
            bubble_id=None, meta={"manual_region_ocr": True, "box_locked": True},
        )
        lr = fit_text(target.shape[:2], selection, unit, text, lcfg)
        if not lr.success or lr.text_mask is None:
            raise ValueError(str(lr.reason or "OCR 区域排版失败"))
        before = out.copy(); out = composite_text(out, lr, lcfg)
        text_mask = cv2.bitwise_and(np.asarray(lr.text_mask, np.uint8), selection)
        # composite_text is constrained by fit mask, but enforce selection again
        # so a future renderer change cannot leak outside the explicit manual ROI.
        outside = selection == 0; out[outside] = before[outside]
        layer = np.zeros((h, w, 4), np.uint8); use = text_mask > 0; layer[use, :3] = out[use]; layer[use, 3] = text_mask[use]
        diag.update({"text": text, "text_pixels": int(cv2.countNonZero(text_mask)), "font_path": lr.font_path, "font_size": int(lr.font_size), "orientation": lr.orientation, "ocr_used": True})

    changed = np.any(out != current, axis=2)
    outside_changed = int(np.count_nonzero(changed & (selection == 0)))
    if outside_changed:
        raise RuntimeError(f"区域工具越界写入 {outside_changed} px")
    audit = {
        "id": str(row.get("id") or ""), "success": bool(np.any(changed)), "mode": mode,
        "target_bbox": bbox, "selection_kind": str((row.get("selection_spec") or {}).get("kind") or "rect"),
        "changed_pixels": int(np.count_nonzero(changed)), "outside_selection_changed_pixels": outside_changed,
        "target_clear_pixels": int(cv2.countNonZero(clear)), "diagnostics": diag,
    }
    return out, layer, clear, audit


__all__ = ["REGION_MODES", "is_region_mode", "apply_region_action"]
