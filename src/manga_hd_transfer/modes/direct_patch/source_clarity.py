from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.astype(np.uint8)


def _percentile(values: np.ndarray, q: float, default: float) -> float:
    if values.size <= 0:
        return float(default)
    return float(np.percentile(values, q))


def _paper_and_text_support(
    source_region: np.ndarray,
    region_u8: np.ndarray,
    source_text_mask: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate source paper and recover a topology-safe Chinese ink support.

    v2.3.30 used a fairly strict text mask as the final rendering authority. That
    was safe against scan noise, but it could drop an entire antialiased edge or a
    small glyph component.  Here the supplied mask is a *seed*, not a clipping
    mask: compact dark components on the same neutral paper are recovered while
    long boundary rules are rejected.
    """
    use = region_u8 > 0
    gray = _gray(source_region).astype(np.float32)
    seed = ((source_text_mask > 0) & use).astype(np.uint8)
    seed_halo = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1) > 0

    if source_region.ndim == 3:
        sat = cv2.cvtColor(source_region, cv2.COLOR_BGR2HSV)[..., 1]
        neutral = sat <= 72
    else:
        neutral = np.ones_like(use, dtype=bool)
    paper_candidates = use & (~seed_halo) & neutral & (gray >= 178.0)
    if int(np.count_nonzero(paper_candidates)) >= 32:
        paper_gray = _percentile(gray[paper_candidates], 72.0, 255.0)
        paper_rgb = np.median(source_region[paper_candidates], axis=0) if source_region.ndim == 3 else np.array([paper_gray] * 3)
    else:
        vals = gray[use]
        paper_gray = _percentile(vals, 84.0, 255.0)
        paper_rgb = np.array([paper_gray] * 3)

    # Detect all meaningful source ink on the white paper, then use the original
    # seed only to bias ambiguous components.  This recovers punctuation and thin
    # antialiased CJK strokes that strict component selection can miss.
    contrast_floor = max(5.0, min(12.0, paper_gray * 0.035))
    raw_ink = (use & ((paper_gray - gray) >= contrast_floor)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw_ink, 8)
    ys, xs = np.where(use)
    rw = max(1, int(xs.max() - xs.min() + 1)) if xs.size else 1
    rh = max(1, int(ys.max() - ys.min() + 1)) if ys.size else 1
    region_area = max(1, int(np.count_nonzero(use)))
    er = cv2.erode(use.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    boundary_ring = use & (er == 0)
    seed_near = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1) > 0

    core = np.zeros_like(raw_ink)
    recovered_components = 0
    rejected_structural = 0
    for lab in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area < 2 or area > max(320, int(round(region_area * 0.14))):
            continue
        comp = labels == lab
        span_x = bw / max(1.0, float(rw))
        span_y = bh / max(1.0, float(rh))
        fill = area / max(1.0, float(bw * bh))
        touches_boundary = bool(np.any(comp & boundary_ring))
        long_rule = (
            (span_x >= 0.34 and bh <= max(8, int(round(rh * 0.11))))
            or (span_y >= 0.34 and bw <= max(8, int(round(rw * 0.11))))
        )
        outline_like = span_x >= 0.58 and span_y >= 0.42 and fill <= 0.30
        seed_supported = bool(np.any(comp & seed_near))
        if touches_boundary and (long_rule or outline_like) and not seed_supported:
            rejected_structural += 1
            continue
        # Isolated 1-4px scan dust must not become "Chinese punctuation" merely
        # because it is dark on paper. True punctuation missed by the strict seed
        # is normally larger than this or sits beside accepted text.
        if not seed_supported and area < 6:
            continue
        # Compact glyphs/punctuation are accepted even if the strict seed missed
        # them; very large unsupported artwork-like blobs are rejected above.
        core[comp] = 1
        if not seed_supported:
            recovered_components += 1

    # Preserve the native gray antialias fringe around accepted core pixels.
    support = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1) > 0
    support &= use
    return paper_gray, np.asarray(paper_rgb, np.float32), core > 0, support, {
        "seed_text_pixels": int(np.count_nonzero(seed)),
        "recovered_core_pixels": int(np.count_nonzero(core)),
        "support_pixels": int(np.count_nonzero(support)),
        "recovered_components": int(recovered_components),
        "rejected_structural_components": int(rejected_structural),
        "contrast_floor": float(contrast_floor),
    }


def enhance_white_source_patch(
    source_region: np.ndarray,
    region_mask: np.ndarray,
    source_text_mask: np.ndarray,
    *,
    enabled: bool = True,
    alpha_gamma: float = 1.0,
    black_boost: int = 0,
    pure_white_floor: int = 248,
    min_text_pixels: int = 18,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clean SOURCE paper while preserving the complete original Chinese raster.

    v2.3.30/v2.3.31-pre used a derived text support mask as the final drawing
    authority. On real manga this could drop complete vertical CJK columns when
    the strict/changed-text evidence disagreed with the source typesetting.

    The safe contract is now simpler: the text mask proves that this white patch
    contains lettering, but it never clips the published glyph. Once authorized,
    every SOURCE pixel inside the already accepted borderless region is mapped by
    one continuous paper-normalisation curve. Therefore all Chinese strokes,
    punctuation and native antialias shades survive. The surrounding Direct/Mask
    geometry remains responsible for excluding the balloon outline/artwork.
    """
    region_u8 = (region_mask > 0).astype(np.uint8) * 255
    use = region_u8 > 0
    region_pixels = int(np.count_nonzero(use))
    seed_text = ((source_text_mask > 0) & use).astype(np.uint8) * 255
    seed_pixels = int(cv2.countNonZero(seed_text))
    diag: dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "region_pixels": region_pixels,
        "text_pixels": seed_pixels,
        "seed_text_pixels": seed_pixels,
        "paper_pixels": 0,
        "paper_ratio": 0.0,
        "paper_gray": 255.0,
        "paper_rgb": [255, 255, 255],
        "background_whitened_pixels": 0,
        "gray_background_pixels_before": 0,
        "gray_background_pixels_after": 0,
        "removed_gray_pixels": 0,
        "alpha_gamma": float(alpha_gamma),
        "black_boost": int(black_boost),
        "pure_white_floor": int(pure_white_floor),
        "reason": "ok",
        "text_render_mode": "source_raster_continuous_paper_normalization",
        "preserve_source_antialias": True,
        "source_text_mask_is_gate_only": True,
        "source_raster_fidelity_lock": True,
    }
    if not bool(enabled):
        diag["reason"] = "disabled"
        return source_region.copy(), diag
    if region_pixels <= 0:
        diag["reason"] = "empty_region"
        return source_region.copy(), diag
    if seed_pixels < max(1, int(min_text_pixels)):
        diag["reason"] = "too_little_text"
        return source_region.copy(), diag

    src_gray = _gray(source_region).astype(np.float32)
    paper_gray, paper_rgb, _core, _support, support_diag = _paper_and_text_support(
        source_region, region_u8, seed_text
    )
    if not np.isfinite(paper_gray) or paper_gray < 120.0:
        diag.update(support_diag)
        diag["reason"] = "invalid_paper_estimate"
        return source_region.copy(), diag

    # Estimate background only for diagnostics. It is not used to decide which
    # glyph pixels survive.
    seed_halo = cv2.dilate(seed_text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1) > 0
    bg_only = use & (~seed_halo)
    if int(np.count_nonzero(bg_only)) < 16:
        bg_only = use & (src_gray >= max(170.0, paper_gray - 28.0))
    paper_pixels = int(np.count_nonzero(bg_only & (src_gray >= max(170.0, paper_gray - 28.0))))
    diag["paper_pixels"] = paper_pixels
    diag["paper_ratio"] = float(paper_pixels / max(1, region_pixels))

    base_level = int(np.clip(max(float(pure_white_floor), paper_gray), 0.0, 255.0))
    paper_rgb_u8 = np.clip(np.maximum(paper_rgb, base_level), 0, 255).astype(np.uint8)
    diag["paper_gray"] = float(paper_gray)
    diag["paper_rgb"] = [int(x) for x in paper_rgb_u8.tolist()]
    diag.update(support_diag)

    before_gray_bg = int(np.count_nonzero(bg_only & (src_gray < max(238.0, paper_gray - 2.0))))

    # Map source paper to clean white with one monotonic continuous transform.
    # Default gamma=1 / boost=0 is topology-preserving: no threshold, binary
    # silhouette, hard-core forcing or mask clipping is involved.
    paper_scale = 255.0 / max(1.0, float(paper_gray))
    normalized = np.clip(src_gray * paper_scale, 0.0, 255.0)
    coverage = np.clip((255.0 - normalized) / 255.0, 0.0, 1.0)
    gamma = max(0.40, float(alpha_gamma))
    if abs(gamma - 1.0) > 1e-6:
        coverage[use] = np.power(coverage[use], gamma)
    boost = max(0, int(black_boost))
    out_gray = np.full(src_gray.shape, 255.0, np.float32)
    out_gray[use] = 255.0 - coverage[use] * 255.0
    if boost > 0:
        ink_like = use & (coverage > 0.015)
        out_gray[ink_like] = np.clip(out_gray[ink_like] - float(boost), 0.0, 255.0)

    # Suppress only sub-3-level paper wobble. This is deliberately much weaker
    # than a text threshold so faint antialias fringe and punctuation remain.
    near_paper = use & ((255.0 - out_gray) <= 3.0)
    out_gray[near_paper] = 255.0

    # Remove only isolated <=4px dark scan specks that are not supported by the
    # original text evidence. Larger unsupported components are deliberately kept:
    # on real CJK pages they can be complete columns/punctuation missed by a mask.
    dust_binary = (use & ((255.0 - out_gray) >= 18.0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dust_binary, 8)
    seed_near = cv2.dilate(seed_text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1) > 0
    dust_removed = 0
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area <= 0 or area > 4:
            continue
        comp = labels == lab
        if np.any(comp & seed_near):
            continue
        dust_removed += int(np.count_nonzero(comp))
        out_gray[comp] = 255.0

    clean = source_region.copy()
    gray_u8 = np.clip(out_gray, 0.0, 255.0).astype(np.uint8)
    if clean.ndim == 3:
        rgb = np.repeat(gray_u8[..., None], 3, axis=2)
        clean[use] = rgb[use]
    else:
        clean[use] = gray_u8[use]

    after_gray = _gray(clean).astype(np.float32)
    after_gray_bg = int(np.count_nonzero(bg_only & (after_gray < 244.0)))
    ink_like = use & ((255.0 - out_gray) > 3.0)
    diag["text_pixels"] = int(np.count_nonzero(ink_like))
    diag["background_whitened_pixels"] = int(np.count_nonzero(bg_only))
    diag["gray_background_pixels_before"] = before_gray_bg
    diag["gray_background_pixels_after"] = after_gray_bg
    diag["removed_gray_pixels"] = int(max(0, before_gray_bg - after_gray_bg))
    diag["paper_normalization_scale"] = float(paper_scale)
    diag["continuous_coverage_pixels"] = int(np.count_nonzero(ink_like))
    diag["published_raster_pixels"] = int(np.count_nonzero(use))
    diag["dust_pixels_removed"] = int(dust_removed)
    diag["applied"] = True
    return clean, diag


__all__ = ["enhance_white_source_patch"]
