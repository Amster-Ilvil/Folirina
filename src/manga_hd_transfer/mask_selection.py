from __future__ import annotations

"""Renderer-independent mask route/candidate selection.

Appearance statistics and eligibility decisions live here so the 4k-line raster
renderer does not also own the policy that decides which renderer is suitable.
"""

import math

import cv2
import numpy as np

from .config import MaskReplaceConfig
from .mask_geometry import _bbox_from_mask
from .text_only_transfer import changed_text_masks

def _publication_safety_enabled(cfg) -> bool:
    """Legacy compatibility shim: publication blocking was removed in v1.0.6."""
    return False

def _rigid_container_stats(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    gray: np.ndarray | None = None,
    hsv: np.ndarray | None = None,
) -> dict[str, float]:
    """Cheap appearance/geometry guard for full-container raster transfer.

    v0.8.28 accepts precomputed page colour spaces so a page with many bubbles
    does not repeatedly convert the same multi-megapixel source/target image.
    """
    sel = mask > 0
    box = _bbox_from_mask(mask)
    if box is None or not np.any(sel):
        return {}
    x0, y0, x1, y1 = box
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hsv is None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    area = int(np.count_nonzero(sel))
    return {
        "area": float(area),
        "fill": float(area / max(1, (x1 - x0) * (y1 - y0))),
        "white_ratio": float(np.mean(gray[sel] >= 220)),
        "dark_ratio": float(np.mean(gray[sel] <= 180)),
        "sat_median": float(np.median(hsv[..., 1][sel])),
        "sat_p90": float(np.percentile(hsv[..., 1][sel], 90.0)),
        "width": float(x1 - x0),
        "height": float(y1 - y0),
    }

def _rigid_container_pair_eligible(
    source: np.ndarray,
    target_reference: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    source_gray: np.ndarray | None = None,
    source_hsv: np.ndarray | None = None,
    target_gray: np.ndarray | None = None,
    target_hsv: np.ndarray | None = None,
) -> tuple[bool, dict[str, float | str]]:
    """Return whether a pair is safe for a locked whole-raster transfer.

    The important contract is *not* that the page registration is rigid.  Page
    registration may use affine/homography to discover the corresponding region.
    Once a speech/narration container is paired, however, the source raster is
    rendered with one scalar scale only.  This decoupling prevents CJK glyphs from
    inheriting anisotropic scan/camera correction.
    """
    ss = _rigid_container_stats(source, source_mask, gray=source_gray, hsv=source_hsv)
    ts = _rigid_container_stats(target_reference, target_mask, gray=target_gray, hsv=target_hsv)
    if not ss or not ts:
        return False, {"reason": "empty_mask"}
    base_min_fill = float(getattr(cfg, "rigid_container_min_fill_ratio", 0.55))
    spiky_min_fill = float(getattr(cfg, "rigid_container_spiky_min_fill_ratio", 0.30))
    spiky_min_white = float(getattr(cfg, "rigid_container_spiky_min_white_ratio", 0.78))
    spiky_max_aspect = float(getattr(cfg, "rigid_container_spiky_max_aspect", 3.5))
    saspect = max(ss["width"] / max(1.0, ss["height"]), ss["height"] / max(1.0, ss["width"]))
    taspect = max(ts["width"] / max(1.0, ts["height"]), ts["height"] / max(1.0, ts["width"]))
    spiky_ok = bool(getattr(cfg, "rigid_container_spiky_white_enabled", True)) and (
        ss["fill"] >= spiky_min_fill and ts["fill"] >= spiky_min_fill
        and ss["white_ratio"] >= spiky_min_white and ts["white_ratio"] >= spiky_min_white
        and saspect <= spiky_max_aspect and taspect <= spiky_max_aspect
    )
    if not _publication_safety_enabled(cfg):
        # Safety-off does not mean using the wrong renderer.  A rigid whole-raster
        # patch is technically suitable only for paper/white target containers;
        # coloured or textured targets must fall through to target-aware Mask
        # transfer so their HD fill is preserved.  This is route selection, not a
        # publication rejection.
        if ts["white_ratio"] < 0.55 or ts["sat_median"] > 55.0:
            d = {"reason": "requires_target_aware_colored_transfer"}
            d.update({f"source_{k}": v for k, v in ss.items()}); d.update({f"target_{k}": v for k, v in ts.items()})
            return False, d
        sar = ss["width"] / max(1.0, ss["height"])
        tar = ts["width"] / max(1.0, ts["height"])
        aspect_delta = abs(math.log(max(1e-6, sar / max(1e-6, tar))))
        scale = math.sqrt(ts["area"] / max(1.0, ss["area"]))
        if not (0.20 <= scale <= 4.0):
            return False, {"reason": "uniform_scale_unusable", "uniform_scale": scale}
        d: dict[str, float | str] = {"reason": "eligible_aggressive_white", "uniform_scale": scale, "aspect_log_delta": aspect_delta}
        d.update({f"source_{k}": v for k, v in ss.items()}); d.update({f"target_{k}": v for k, v in ts.items()})
        return True, d
    checks = (
        (ss["white_ratio"] >= float(getattr(cfg, "rigid_container_min_source_white_ratio", 0.78)), "source_not_white_container"),
        (ts["white_ratio"] >= float(getattr(cfg, "rigid_container_min_target_white_ratio", 0.75)), "target_not_white_container"),
        (ss["fill"] >= base_min_fill or spiky_ok, "source_mask_too_sparse"),
        (ts["fill"] >= base_min_fill or spiky_ok, "target_mask_too_sparse"),
        (ss["dark_ratio"] >= float(getattr(cfg, "rigid_container_min_source_dark_ratio", 0.020)), "source_has_too_little_ink"),
        (ts["dark_ratio"] >= float(getattr(cfg, "rigid_container_min_target_dark_ratio", 0.015)), "target_has_too_little_ink"),
        (ss["dark_ratio"] <= float(getattr(cfg, "rigid_container_max_dark_ratio", 0.30)), "source_too_art_like"),
        (ts["dark_ratio"] <= float(getattr(cfg, "rigid_container_max_dark_ratio", 0.30)), "target_too_art_like"),
        (ss["sat_p90"] <= float(getattr(cfg, "rigid_container_max_source_saturation_p90", 28.0)), "source_not_monochrome_paper"),
        (ts["sat_median"] <= float(getattr(cfg, "rigid_container_max_target_saturation_median", 36.0)), "target_not_white_paper"),
    )
    for ok, reason in checks:
        if not ok:
            d = {"reason": reason}; d.update({f"source_{k}": v for k, v in ss.items()}); d.update({f"target_{k}": v for k, v in ts.items()})
            return False, d
    sar = ss["width"] / max(1.0, ss["height"])
    tar = ts["width"] / max(1.0, ts["height"])
    aspect_delta = abs(math.log(max(1e-6, sar / max(1e-6, tar))))
    if aspect_delta > float(getattr(cfg, "rigid_container_max_aspect_log_delta", 0.16)):
        return False, {"reason": "container_aspect_mismatch", "aspect_log_delta": aspect_delta}
    scale = math.sqrt(ts["area"] / max(1.0, ss["area"]))
    if not (float(getattr(cfg, "rigid_container_min_uniform_scale", 0.35)) <= scale <= float(getattr(cfg, "rigid_container_max_uniform_scale", 1.85))):
        return False, {"reason": "uniform_scale_out_of_range", "uniform_scale": scale}
    d: dict[str, float | str] = {"reason": "eligible_spiky_white" if spiky_ok and (ss["fill"] < base_min_fill or ts["fill"] < base_min_fill) else "eligible", "uniform_scale": scale, "aspect_log_delta": aspect_delta}
    d.update({f"source_{k}": v for k, v in ss.items()}); d.update({f"target_{k}": v for k, v in ts.items()})
    return True, d




def _support_mask_textlike(
    support_mask: np.ndarray,
    region_mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[bool, dict[str, float | int | str | bool]]:
    """Reject art-like support masks that are not plausible translated text.

    ``changed_text_masks`` is intentionally permissive: on noisy scans it may keep
    edition-exclusive artwork edges (helmet rings, glasses, buttons) because they
    also differ between SOURCE and TARGET.  A rigid container is allowed only
    when the surviving support still looks glyph-like.  Large hollow loops that
    dominate the support mask are therefore treated as art, not text.
    """
    region_box = _bbox_from_mask(region_mask)
    support_pixels = int(cv2.countNonZero(support_mask))
    if support_pixels <= 0 or region_box is None:
        return False, {
            "valid": False,
            "reason": "empty_support_mask",
            "support_pixels": int(support_pixels),
        }
    x0, y0, x1, y1 = [int(v) for v in region_box]
    region_w = max(1, x1 - x0)
    region_h = max(1, y1 - y0)
    region_area = max(1, int(cv2.countNonZero(region_mask)))

    labels_n, labels, stats, _ = cv2.connectedComponentsWithStats((support_mask > 0).astype(np.uint8), 8)
    compact_components = 0
    compact_pixels = 0
    suspicious_large_hollow = 0
    largest_component_pixels = 0
    largest_component_hole_ratio = 0.0
    largest_component_fraction = 0.0
    max_hole_ratio = 0.0

    hollow_enabled = bool(getattr(cfg, "rigid_container_support_large_hollow_reject_enabled", True))
    hollow_min_pixels = max(24, int(getattr(cfg, "rigid_container_support_large_hollow_min_pixels", 240)))
    hollow_min_fraction = float(getattr(cfg, "rigid_container_support_large_hollow_min_fraction", 0.45))
    hollow_max_ratio = float(getattr(cfg, "rigid_container_support_large_hollow_max_ratio", 0.90))
    hollow_min_extent = float(getattr(cfg, "rigid_container_support_large_hollow_min_extent", 0.30))

    compact_max_area = max(600, int(0.035 * region_area))
    compact_max_w = max(10.0, 0.34 * float(region_w))
    compact_max_h = max(10.0, 0.34 * float(region_h))
    compact_max_aspect = 10.0

    for lab in range(1, labels_n):
        comp = (labels == lab).astype(np.uint8) * 255
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        lx = int(stats[lab, cv2.CC_STAT_LEFT]); ly = int(stats[lab, cv2.CC_STAT_TOP])
        lw = int(stats[lab, cv2.CC_STAT_WIDTH]); lh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        bbox_area = max(1, lw * lh)
        aspect = max(lw / max(1.0, lh), lh / max(1.0, lw))
        extent = float(area / bbox_area)
        fraction = float(area / max(1, support_pixels))
        largest_component_pixels = max(largest_component_pixels, area)
        if area == largest_component_pixels:
            largest_component_fraction = fraction

        contours, hierarchy = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hole_area = 0.0
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for idx, node in enumerate(hierarchy):
                if int(node[3]) >= 0:
                    hole_area += abs(float(cv2.contourArea(contours[idx])))
        hole_ratio = float(hole_area / max(1.0, area))
        max_hole_ratio = max(max_hole_ratio, hole_ratio)
        if area == largest_component_pixels:
            largest_component_hole_ratio = hole_ratio

        is_compact = (
            2 <= area <= compact_max_area
            and lw <= compact_max_w
            and lh <= compact_max_h
            and aspect <= compact_max_aspect
        )
        if is_compact:
            compact_components += 1
            compact_pixels += area

        if hollow_enabled and area >= hollow_min_pixels and fraction >= hollow_min_fraction and extent >= hollow_min_extent and hole_ratio >= hollow_max_ratio:
            suspicious_large_hollow += 1

    compact_ratio = float(compact_pixels / max(1, support_pixels))
    ok = suspicious_large_hollow == 0
    reason = "ok" if ok else "art_like_support_component"
    return ok, {
        "valid": True,
        "reason": reason,
        "support_pixels": int(support_pixels),
        "compact_components": int(compact_components),
        "compact_pixels": int(compact_pixels),
        "compact_ratio": float(compact_ratio),
        "largest_component_pixels": int(largest_component_pixels),
        "largest_component_fraction": float(largest_component_fraction),
        "largest_component_hole_ratio": float(largest_component_hole_ratio),
        "max_component_hole_ratio": float(max_hole_ratio),
        "suspicious_large_hollow_components": int(suspicious_large_hollow),
        "threshold_large_hollow_min_pixels": int(hollow_min_pixels),
        "threshold_large_hollow_min_fraction": float(hollow_min_fraction),
        "threshold_large_hollow_max_ratio": float(hollow_max_ratio),
        "threshold_large_hollow_min_extent": float(hollow_min_extent),
    }
def _rigid_source_text_support(
    source: np.ndarray,
    target_reference: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[bool, dict[str, float | int | str | bool]]:
    """Prove that a rigid container candidate actually contains translated text.

    A pair can look like a clean white container statistically while still being
    a face highlight, shirt, or other illustration region.  Rigid text transfer
    must therefore verify that the SOURCE crop contains *edition-exclusive*
    compact ink after being aligned to the TARGET crop.  If no such support
    exists, the pair is not a text container and must be rejected.
    """
    if not bool(getattr(cfg, "rigid_container_source_text_support_enabled", True)):
        return True, {"enabled": False, "reason": "disabled"}
    sbox = _bbox_from_mask(source_mask)
    tbox = _bbox_from_mask(target_mask)
    if sbox is None or tbox is None:
        return False, {"enabled": True, "reason": "empty_mask"}
    sx0, sy0, sx1, sy1 = [int(v) for v in sbox]
    tx0, ty0, tx1, ty1 = [int(v) for v in tbox]
    if sx1 <= sx0 or sy1 <= sy0 or tx1 <= tx0 or ty1 <= ty0:
        return False, {"enabled": True, "reason": "degenerate_bbox"}

    source_crop = source[sy0:sy1, sx0:sx1]
    target_crop = target_reference[ty0:ty1, tx0:tx1]
    source_mask_crop = source_mask[sy0:sy1, sx0:sx1]
    target_mask_crop = target_mask[ty0:ty1, tx0:tx1]
    if source_crop.size == 0 or target_crop.size == 0:
        return False, {"enabled": True, "reason": "empty_crop"}

    target_h, target_w = target_crop.shape[:2]
    source_to_target = cv2.resize(source_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    source_mask_to_target = cv2.resize(source_mask_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    region_mask = np.maximum((source_mask_to_target > 0).astype(np.uint8), (target_mask_crop > 0).astype(np.uint8)) * 255
    if cv2.countNonZero(region_mask) == 0:
        region_mask[:, :] = 255

    source_text, target_text, diff_diag = changed_text_masks(
        source_to_target,
        target_crop,
        region_mask,
        tolerance_px=max(1, int(getattr(cfg, "rigid_container_source_text_tolerance_px", 2))),
        min_unique_ratio=float(getattr(cfg, "rigid_container_source_text_unique_ratio", 0.055)),
        max_component_fraction=float(getattr(cfg, "rigid_container_source_text_max_component_fraction", 0.10)),
    )
    region_pixels = max(1, int(cv2.countNonZero(region_mask)))
    source_pixels = int(cv2.countNonZero(source_text))
    target_pixels = int(cv2.countNonZero(target_text))
    source_ratio = float(source_pixels / region_pixels)
    target_ratio = float(target_pixels / region_pixels)
    min_pixels = max(1, int(getattr(cfg, "rigid_container_min_source_text_pixels", 24)))
    min_ratio = float(getattr(cfg, "rigid_container_min_source_text_ratio", 0.0025))
    ok = bool(source_pixels >= min_pixels or source_ratio >= min_ratio)
    diag: dict[str, float | int | str | bool] = {
        "enabled": True,
        "reason": "ok" if ok else "missing_source_text_support",
        "region_pixels": int(region_pixels),
        "source_text_pixels": int(source_pixels),
        "target_text_pixels": int(target_pixels),
        "source_text_ratio": float(source_ratio),
        "target_text_ratio": float(target_ratio),
        "min_source_text_pixels": int(min_pixels),
        "min_source_text_ratio": float(min_ratio),
    }
    diag.update({f"text_{k}": v for k, v in diff_diag.items()})
    if ok:
        source_shape_ok, source_shape_diag = _support_mask_textlike(source_text, region_mask, cfg)
        target_shape_ok, target_shape_diag = _support_mask_textlike(target_text, region_mask, cfg)
        diag["source_text_shape"] = source_shape_diag
        diag["target_text_shape"] = target_shape_diag
        # SOURCE is the only raster authority in rigid transfer. If SOURCE's
        # changed support is art-like, a text-like TARGET cannot rescue it: that
        # exact asymmetry is how face/helmet structure used to enter the Chinese
        # migration layer. TARGET shape remains diagnostic evidence only.
        if not source_shape_ok:
            ok = False
            diag["reason"] = "art_like_source_text_support"
    return ok, diag

__all__ = [
    "_publication_safety_enabled",
    "_rigid_container_stats",
    "_rigid_container_pair_eligible",
    "_support_mask_textlike",
    "_rigid_source_text_support",
]
