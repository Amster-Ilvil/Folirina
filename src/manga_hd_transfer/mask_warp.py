from __future__ import annotations

"""Geometric warp/refinement helpers for Mask transfer.

This module changes coordinates/sampling only. It does not choose transfer
regions or composite final pixels, so it can be regression-tested independently.
"""

import cv2
import numpy as np

from .config import MaskReplaceConfig
from .geometry import transform_to_homography
from .mask_geometry import _bbox_from_mask, _target_coverage, _mask_iou
from .mask_quality import _superresolve_patch

def _soft_mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    af = np.clip(a.astype(np.float32) / 255.0, 0.0, 1.0)
    bf = np.clip(b.astype(np.float32) / 255.0, 0.0, 1.0)
    inter = float(np.minimum(af, bf).sum())
    union = float(np.maximum(af, bf).sum())
    return inter / max(1e-6, union)


def _shift(image: np.ndarray, dx: float, dy: float, nearest: bool = False) -> np.ndarray:
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LANCZOS4
    border = 0 if image.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _subpixel_translation_refine(
    warped_mask: np.ndarray,
    target_mask: np.ndarray,
    dx: float,
    dy: float,
    cfg: MaskReplaceConfig,
) -> tuple[float, float, float, dict]:
    """Refine local translation in bounded fractional-pixel steps.

    This uses only container geometry and never deforms the source raster.  It is
    deliberately a tiny search around the ECC solution, so the global/local
    registration remains the authority and text cannot drift to chase different
    Chinese/Japanese glyph shapes.
    """
    base_shifted = _shift(warped_mask, dx, dy, nearest=False)
    base_score = _soft_mask_iou(base_shifted, target_mask)
    if not bool(getattr(cfg, "local_subpixel_refine_enabled", True)):
        return dx, dy, base_score, {"enabled": False, "before": base_score, "after": base_score}
    step = max(0.1, float(getattr(cfg, "local_subpixel_step", 0.5)))
    radius = max(0.0, float(getattr(cfg, "local_subpixel_radius_px", 1.0)))
    if radius < step * 0.5:
        return dx, dy, base_score, {"enabled": True, "before": base_score, "after": base_score, "tested": 1}
    offsets = np.arange(-radius, radius + step * 0.25, step, dtype=np.float32)
    best_dx, best_dy, best = float(dx), float(dy), float(base_score)
    tested = 0
    for oy in offsets:
        for ox in offsets:
            cdx, cdy = float(dx + ox), float(dy + oy)
            moved = _shift(warped_mask, cdx, cdy, nearest=False)
            score = _soft_mask_iou(moved, target_mask)
            tested += 1
            if score > best + 1e-9:
                best_dx, best_dy, best = cdx, cdy, score
    min_gain = float(getattr(cfg, "local_subpixel_min_iou_gain", 0.0015))
    if best < base_score + min_gain:
        best_dx, best_dy, best = float(dx), float(dy), float(base_score)
    return best_dx, best_dy, best, {
        "enabled": True, "before": float(base_score), "after": float(best),
        "tested": int(tested), "dx": float(best_dx), "dy": float(best_dy),
        "gain": float(best - base_score),
    }


def _warp_source_patch(
    source: np.ndarray,
    source_mask: np.ndarray,
    H: np.ndarray,
    target_shape: tuple[int, int],
    target_bbox: tuple[int, int, int, int],
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    box = _bbox_from_mask(source_mask)
    if not box:
        h, w = target_shape
        return np.zeros((h, w, 3), np.uint8), np.zeros((h, w), np.uint8), "off", 1.0
    x0, y0, x1, y1 = box
    pad = max(3, cfg.source_mask_expand_px + 2)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(source.shape[1], x1 + pad), min(source.shape[0], y1 + pad)
    crop = source[y0:y1, x0:x1]
    cmask = source_mask[y0:y1, x0:x1]
    tbw = max(1, target_bbox[2] - target_bbox[0]); tbh = max(1, target_bbox[3] - target_bbox[1])
    desired = max(tbw / max(1, x1 - x0), tbh / max(1, y1 - y0))
    crop_sr, backend, actual_scale = _superresolve_patch(crop, desired, cfg)
    mask_sr = cv2.resize(cmask, (crop_sr.shape[1], crop_sr.shape[0]), interpolation=cv2.INTER_NEAREST)

    # SR changes only sampling density. This matrix maps SR patch coordinates back
    # to original source coordinates before the geometrical source->target warp.
    sx = (x1 - x0) / max(1, crop_sr.shape[1])
    sy = (y1 - y0) / max(1, crop_sr.shape[0])
    sr_to_source = np.array([[sx, 0.0, x0], [0.0, sy, y0], [0.0, 0.0, 1.0]], np.float64)
    Hpatch = transform_to_homography(H) @ sr_to_source
    th, tw = target_shape
    warped_img = cv2.warpPerspective(crop_sr, Hpatch, (tw, th), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    warped_mask = cv2.warpPerspective(mask_sr, Hpatch, (tw, th), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_img, warped_mask, backend, actual_scale


def _photo_pair_salvage_warp(
    source: np.ndarray,
    base_source_mask: np.ndarray,
    warped_img: np.ndarray,
    warped_mask: np.ndarray,
    H: np.ndarray,
    target_shape: tuple[int, int],
    target_bbox: tuple[int, int, int, int],
    target_mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Try a tiny extra source-mask expansion for photographed pairs.

    Phone-shot pages sometimes under-segment the source bubble by a few pixels
    because of glare or clipped outlines. When coverage is only slightly below
    the safe threshold, expanding the *source* mask 1-3 px is often enough to
    recover the full target interior without changing the geometric transform.
    """
    best_img, best_mask = warped_img, warped_mask
    best_backend, best_scale = "off", 1.0
    best_cov, best_spill = _target_coverage(warped_mask, target_mask)
    best_iou = _mask_iou(warped_mask, target_mask)
    if cfg.photo_pair_salvage_max_expand_px <= 0:
        return best_img, best_mask, best_backend, best_scale
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grown = base_source_mask.copy()
    for _ in range(int(cfg.photo_pair_salvage_max_expand_px)):
        grown = cv2.dilate(grown, k, iterations=1)
        cand_img, cand_mask, cand_backend, cand_scale = _warp_source_patch(source, grown, H, target_shape, target_bbox, cfg)
        cov, spill = _target_coverage(cand_mask, target_mask)
        iou = _mask_iou(cand_mask, target_mask)
        better = (
            cov > best_cov + 1e-6
            or (abs(cov - best_cov) <= 1e-6 and spill < best_spill - 1e-6)
            or (abs(cov - best_cov) <= 1e-6 and abs(spill - best_spill) <= 1e-6 and iou > best_iou + 1e-6)
        )
        if better:
            best_img, best_mask = cand_img, cand_mask
            best_backend, best_scale = cand_backend, cand_scale
            best_cov, best_spill, best_iou = cov, spill, iou
        if cov >= cfg.photo_pair_min_transfer_coverage and spill <= cfg.photo_pair_max_spill_ratio and iou >= cfg.photo_pair_min_transfer_iou:
            break
    return best_img, best_mask, best_backend, best_scale


__all__ = ['_soft_mask_iou', '_shift', '_subpixel_translation_refine', '_warp_source_patch', '_photo_pair_salvage_warp']
