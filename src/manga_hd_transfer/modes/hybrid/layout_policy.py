from __future__ import annotations

import cv2
import numpy as np

from ...models import TextBlock, TextUnit
from .binding import mask_bbox

def apply_target_layout_hints(lcfg, dst: TextUnit, safe_mask: np.ndarray | None) -> None:
    if safe_mask is None or cv2.countNonZero(safe_mask) == 0:
        return
    sb = mask_bbox(safe_mask)
    if sb is None:
        return
    sx0, sy0, sx1, sy1 = [float(v) for v in sb]
    dx0, dy0, dx1, dy1 = [float(v) for v in dst.bbox]
    sw = max(1.0, sx1-sx0); sh = max(1.0, sy1-sy0)
    dcx = (dx0+dx1)*0.5; dcy = (dy0+dy1)*0.5
    lcfg.anchor_x_ratio = float(np.clip((dcx-sx0)/sw, 0.05, 0.95))
    lcfg.anchor_y_ratio = float(np.clip((dcy-sy0)/sh, 0.05, 0.95))
    lcfg.preferred_bbox_width_ratio = float(np.clip((dx1-dx0)/sw, 0.08, 1.0))
    lcfg.preferred_bbox_height_ratio = float(np.clip((dy1-dy0)/sh, 0.08, 1.0))


def project_source_profile_mask(
    source_unit: TextUnit,
    target_unit: TextUnit,
    blocks_by_id: dict[str, TextBlock],
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    """Project the source text box to the target bubble as a layout fallback.

    Paired-region OCR stores a source-side ink bbox inside each recognized source
    bubble crop. When synthetic target text-mask detection is weak or absent,
    reuse that measured textbox placement instead of drifting across the entire
    target balloon. The result is only a layout hint; the Japanese clear-mask
    remains authoritative when available.
    """
    src_box = source_unit.bbox
    dst_box = target_unit.bbox
    sx0, sy0, sx1, sy1 = [float(v) for v in src_box]
    tx0, ty0, tx1, ty1 = [float(v) for v in dst_box]
    sw = max(1.0, sx1 - sx0)
    sh = max(1.0, sy1 - sy0)
    tw = max(1.0, tx1 - tx0)
    th = max(1.0, ty1 - ty0)
    h, w = int(target_shape[0]), int(target_shape[1])

    for bid in source_unit.block_ids:
        block = blocks_by_id.get(bid)
        if block is None:
            continue
        meta = getattr(block, "meta", {}) or {}
        profile = meta.get("source_layout_profile") or {}
        ink_box = profile.get("ink_bbox") if isinstance(profile, dict) else None
        crop_box = meta.get("ocr_region_bbox")
        if not ink_box or not crop_box or len(ink_box) != 4 or len(crop_box) != 4:
            continue
        cx0, cy0, cx1, cy1 = [float(v) for v in crop_box]
        ix0, iy0, ix1, iy1 = [float(v) for v in ink_box]
        abs_x0 = cx0 + ix0; abs_y0 = cy0 + iy0
        abs_x1 = cx0 + ix1; abs_y1 = cy0 + iy1
        rx0 = (abs_x0 - sx0) / sw; ry0 = (abs_y0 - sy0) / sh
        rx1 = (abs_x1 - sx0) / sw; ry1 = (abs_y1 - sy0) / sh
        rx0 = float(np.clip(rx0, 0.0, 1.0)); ry0 = float(np.clip(ry0, 0.0, 1.0))
        rx1 = float(np.clip(rx1, 0.0, 1.0)); ry1 = float(np.clip(ry1, 0.0, 1.0))
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        px0 = int(np.floor(tx0 + rx0 * tw))
        py0 = int(np.floor(ty0 + ry0 * th))
        px1 = int(np.ceil(tx0 + rx1 * tw))
        py1 = int(np.ceil(ty0 + ry1 * th))
        if px1 - px0 < 2 or py1 - py0 < 2:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        px0 = max(0, min(w, px0)); px1 = max(0, min(w, px1))
        py0 = max(0, min(h, py0)); py1 = max(0, min(h, py1))
        if px1 - px0 < 2 or py1 - py0 < 2:
            continue
        mask[py0:py1, px0:px1] = 255
        return mask
    return None


def preserved_layout_looks_complete(source_profile: dict, target_profile: dict) -> bool:
    """Reject apparently clipped sharp transfers before skipping OCR re-lettering.

    v0.8.11 proved that a sharp-but-partial transferred patch can still pass the
    generic mask/coverage gates, causing missing leading columns like “小丽差不多”.
    Compare source-vs-target layout recovery in normalized bubble space: if a crisp
    preserved transfer collapses columns/rows or loses too much ink extent, force
    OCR re-letter fallback instead of publishing incomplete text.
    """
    if not source_profile or not target_profile:
        return False
    orientation = str(source_profile.get("orientation") or target_profile.get("orientation") or "")
    sw, sh = [max(1.0, float(v)) for v in (source_profile.get("container_size") or [1, 1])]
    tw, th = [max(1.0, float(v)) for v in (target_profile.get("container_size") or [1, 1])]
    siw, sih = [max(1.0, float(v)) for v in (source_profile.get("ink_bbox_size") or [1, 1])]
    tiw, tih = [max(1.0, float(v)) for v in (target_profile.get("ink_bbox_size") or [1, 1])]
    src_fill_x, src_fill_y = siw / sw, sih / sh
    tgt_fill_x, tgt_fill_y = tiw / tw, tih / th
    src_cols = max(1, int(source_profile.get("columns") or 1))
    tgt_cols = max(1, int(target_profile.get("columns") or 1))
    src_rows = max(1, int(source_profile.get("rows") or 1))
    tgt_rows = max(1, int(target_profile.get("rows") or 1))
    src_pitch = float(source_profile.get("glyph_pitch_px") or 0.0)
    tgt_pitch = float(target_profile.get("glyph_pitch_px") or 0.0)

    if orientation == "vertical":
        if src_cols >= 2 and tgt_cols < src_cols:
            return False
        if src_rows >= 4 and tgt_rows + 1 < src_rows:
            return False
        if src_cols >= 2 and tgt_fill_x < src_fill_x * 0.72:
            return False
        if tgt_pitch > 0 and src_pitch > 0 and tgt_pitch < src_pitch * 0.58:
            return False
        return True

    if src_rows >= 2 and tgt_rows < src_rows:
        return False
    if src_cols >= 4 and tgt_cols + 1 < src_cols:
        return False
    if src_rows >= 2 and tgt_fill_y < src_fill_y * 0.72:
        return False
    if tgt_pitch > 0 and src_pitch > 0 and tgt_pitch < src_pitch * 0.58:
        return False
    return True


def reletter_orientation(base_orientation: str, unit: TextUnit, blocks_by_id: dict) -> str:
    """Infer manga text direction, honoring source-image hints first."""
    if base_orientation in {"horizontal", "vertical"}:
        return base_orientation
    ratios = []
    for bid in unit.block_ids:
        b = blocks_by_id.get(bid)
        if b is None:
            continue
        hinted = str(b.meta.get("orientation_hint") or "").lower()
        if hinted in {"horizontal", "vertical"}:
            return hinted
        x0, y0, x1, y1 = b.bbox
        bw = max(1.0, x1 - x0); bh = max(1.0, y1 - y0)
        ratios.append(bh / bw)
    if ratios:
        med = float(np.median(ratios))
        if med >= 1.30:
            return "vertical"
        if med <= 0.78:
            return "horizontal"
    x0, y0, x1, y1 = unit.bbox
    return "vertical" if (y1 - y0) / max(1.0, x1 - x0) >= 1.45 else "horizontal"

__all__ = [
    "apply_target_layout_hints", "project_source_profile_mask",
    "preserved_layout_looks_complete", "reletter_orientation",
]
