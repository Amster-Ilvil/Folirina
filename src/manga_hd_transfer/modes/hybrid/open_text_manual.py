from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .text_transfer import changed_text_masks, prune_source_text_mask, target_text_mask_in_container
from ...schema_compat import as_dict, as_list


@dataclass(slots=True)
class ManualEffectMasks:
    aligned_source: np.ndarray
    source_mask: np.ndarray
    target_clear_mask: np.ndarray
    diagnostics: dict[str, Any]


def registration_homography(project: dict[str, Any]) -> np.ndarray:
    """Read the persisted SOURCE->TARGET page transform from ``project.json``."""
    root = as_dict(project)
    row = as_dict(root.get("registration"))
    raw = np.asarray(row.get("matrix", np.eye(3)), dtype=np.float64)
    if raw.shape == (2, 3):
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = raw
        raw = H
    if raw.shape != (3, 3) or not np.all(np.isfinite(raw)):
        return np.eye(3, dtype=np.float64)
    if abs(float(raw[2, 2])) > 1e-12:
        raw = raw / float(raw[2, 2])
    return raw


def map_target_bbox_to_source(project: dict[str, Any], bbox: list[int] | tuple[int, int, int, int]) -> list[int]:
    """Map one target-space rectangle back to SOURCE for UI guidance only."""
    x0, y0, x1, y1 = map(float, bbox)
    H = registration_homography(project)
    try:
        inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        inv = np.eye(3, dtype=np.float64)
    pts = np.asarray([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.float32)
    mapped = cv2.perspectiveTransform(pts, inv.astype(np.float64))[0]
    sx0 = int(np.floor(float(np.min(mapped[:, 0]))))
    sy0 = int(np.floor(float(np.min(mapped[:, 1]))))
    sx1 = int(np.ceil(float(np.max(mapped[:, 0]))))
    sy1 = int(np.ceil(float(np.max(mapped[:, 1]))))
    return [sx0, sy0, sx1, sy1]


def _is_raster_identity(H: np.ndarray, source_shape: tuple[int, int], target_shape: tuple[int, int], dx: int, dy: int) -> bool:
    if source_shape != target_shape or dx or dy:
        return False
    return bool(np.max(np.abs(np.asarray(H, np.float64) - np.eye(3))) <= 1e-7)


def align_source_to_target(
    source: np.ndarray,
    target_shape: tuple[int, int],
    project: dict[str, Any],
    *,
    source_offset_x: int = 0,
    source_offset_y: int = 0,
) -> tuple[np.ndarray, bool]:
    """Align SOURCE to target space for a *manual* region.

    Registration is used only to locate the old Chinese raster.  If the persisted
    transform is exactly identity and the canvases already match, this returns an
    untouched copy so manual recovery cannot blur source glyphs unnecessarily.
    """
    th, tw = map(int, target_shape)
    H = registration_homography(project).copy()
    dx = int(source_offset_x)
    dy = int(source_offset_y)
    if _is_raster_identity(H, source.shape[:2], (th, tw), dx, dy):
        return source.copy(), True
    if dx or dy:
        T = np.asarray([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]], np.float64)
        H = T @ H
    aligned = cv2.warpPerspective(
        source,
        H,
        (tw, th),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, False


def _gradient(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return cv2.GaussianBlur(mag, (0, 0), 0.55)


def _cleanup_components(mask: np.ndarray, min_area: int = 2, max_fraction: float = 0.55) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    total = float(binary.size)
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        if total > 0 and area / total > float(max_fraction):
            continue
        out[labels == i] = 1
    return out * 255


def _filter_text_like_components(mask: np.ndarray, crop_shape: tuple[int, int]) -> np.ndarray:
    """Reject obvious non-text islands such as large burst borders.

    Manual regions often include a whole coloured burst box.  After the paired
    edge-difference step the desired Chinese glyphs are usually compact internal
    components, while the spiky border touches the crop boundary and spans a much
    larger bounding box.  Keep the heuristic deliberately conservative: only
    components that look like plausible text strokes survive.
    """
    binary = (mask > 0).astype(np.uint8)
    if int(np.count_nonzero(binary)) == 0:
        return binary * 255
    h, w = map(int, crop_shape)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    area_total = float(max(1, h * w))
    for i in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < 2:
            continue
        region = labels[y:y + bh, x:x + bw] == i
        border_hits = int(np.count_nonzero(region[0, :])) + int(np.count_nonzero(region[-1, :])) + int(np.count_nonzero(region[:, 0])) + int(np.count_nonzero(region[:, -1]))
        touches_crop_border = bool(x <= 1 or y <= 1 or (x + bw) >= w - 1 or (y + bh) >= h - 1)
        fill = float(area / max(1, bw * bh))
        long_ratio = float(max(bw, bh) / max(1, min(bw, bh)))
        area_frac = float(area / area_total)
        # Reject huge/border-anchored streaks; preserve compact glyph blocks.
        if touches_crop_border and (bw >= int(round(w * 0.34)) or bh >= int(round(h * 0.34))) and (border_hits >= 6 or long_ratio >= 4.5):
            continue
        if area_frac > 0.22 and fill < 0.14:
            continue
        out[labels == i] = 1
    return out * 255



def _grow_seed_into_local_strokes(gray: np.ndarray, seed: np.ndarray, *, kernel_px: int = 11, contrast_threshold: int = 8) -> np.ndarray:
    """Fill glyph interiors from paired-difference edge seeds without flooding artwork.

    Both dark-on-light and light-on-dark text are supported.  Local black-hat and
    top-hat responses provide candidate stroke pixels; only connected components
    touched by a structural-difference seed are retained.  This turns edge-only
    masks into complete glyph masks while rejecting most unchanged burst rays and
    panel artwork.
    """
    if gray.shape != seed.shape:
        raise ValueError("stroke-grow inputs must share shape")
    if cv2.countNonZero((seed > 0).astype(np.uint8)) == 0:
        return np.zeros_like(gray, np.uint8)
    ksize = max(5, int(kernel_px) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    blackhat = cv2.subtract(closed, gray)
    tophat = cv2.subtract(gray, opened)
    thr = max(3, int(contrast_threshold))
    candidate = ((blackhat >= thr) | (tophat >= thr)).astype(np.uint8)
    # Preserve very dark/light antialiased stroke cores with weak local contrast.
    candidate |= (((gray <= 80) & (blackhat >= 3)) | ((gray >= 238) & (tophat >= 3))).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, _stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    seed_support = cv2.dilate((seed > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out = np.zeros_like(candidate)
    for i in range(1, count):
        comp = labels == i
        if np.any(comp & seed_support):
            out[comp] = 1
    out = cv2.dilate(out, np.ones((3, 3), np.uint8), iterations=1)
    return out * 255



def _manual_text_corridor(source_mask: np.ndarray, target_mask: np.ndarray, region: np.ndarray, *, radius: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Constrain a broad manual ROI to text-associated pixels only.

    A reviewer may draw a generous box around an open SFX.  That box is a search
    area, never a write mask.  SOURCE Chinese text is the authority: TARGET clear
    pixels must remain within a moderate corridor around confirmed SOURCE glyphs.
    If no SOURCE glyph exists, TARGET clearing is suppressed rather than allowing
    a destructive "clear Japanese only" commit.
    """
    src = ((source_mask > 0) & (region > 0)).astype(np.uint8) * 255
    tgt = ((target_mask > 0) & (region > 0)).astype(np.uint8) * 255
    src = prune_source_text_mask(src, region)
    src_px = int(cv2.countNonZero(src))
    tgt_before = int(cv2.countNonZero(tgt))
    if src_px == 0:
        return src, np.zeros_like(tgt), {
            "source_text_required": True,
            "clear_suppressed_without_source": bool(tgt_before > 0),
            "target_pixels_before_corridor": tgt_before,
            "target_pixels_after_corridor": 0,
        }
    rad = max(8, min(48, int(radius)))
    corridor = cv2.dilate((src > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad * 2 + 1, rad * 2 + 1))) > 0
    tgt = ((tgt > 0) & corridor).astype(np.uint8) * 255
    return src, tgt, {
        "source_text_required": True,
        "clear_suppressed_without_source": False,
        "text_corridor_radius": rad,
        "target_pixels_before_corridor": tgt_before,
        "target_pixels_after_corridor": int(cv2.countNonZero(tgt)),
    }


def strip_border_ring_components(mask: np.ndarray, safe_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove bubble-border / selection-ring like components from a text mask.

    Manual white-bubble review uses a generous rectangular ROI as a *search area*.
    Even after eroding the rectangle, the container outline can still appear as
    thin connected components touching the safe boundary ring.  Those geometry
    fragments are never transferable text and should not participate in either
    SOURCE text transfer or TARGET clear-mask estimation.
    """
    if mask.shape != safe_mask.shape:
        raise ValueError('mask/safe_mask shape mismatch')
    binary = ((mask > 0) & (safe_mask > 0)).astype(np.uint8)
    if int(np.count_nonzero(binary)) == 0:
        return np.zeros_like(mask), {
            'removed_components': 0,
            'removed_pixels': 0,
            'kept_components': 0,
            'kept_pixels': 0,
        }
    safe_bin = (safe_mask > 0).astype(np.uint8)
    eroded = cv2.erode(safe_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    boundary_ring = (safe_bin > 0) & (eroded == 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    removed_components = removed_pixels = kept_components = kept_pixels = 0
    for lab in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        comp = labels == lab
        touches_ring = bool(np.any(comp & boundary_ring))
        fill = float(area / max(1, bw * bh))
        aspect = float(max(bw, bh) / max(1.0, min(bw, bh)))
        thin_dim = min(bw, bh)
        span_fraction = max(bw / max(1.0, safe_mask.shape[1]), bh / max(1.0, safe_mask.shape[0]))
        border_like = touches_ring and (
            thin_dim <= 3
            or fill <= 0.34
            or aspect >= 4.5
            or (fill <= 0.52 and span_fraction >= 0.14)
        )
        if border_like:
            removed_components += 1
            removed_pixels += area
            continue
        out[comp] = 1
        kept_components += 1
        kept_pixels += area
    return out.astype(np.uint8) * 255, {
        'removed_components': int(removed_components),
        'removed_pixels': int(removed_pixels),
        'kept_components': int(kept_components),
        'kept_pixels': int(kept_pixels),
    }


def white_container_safe_mask(
    target: np.ndarray,
    region: np.ndarray,
    *,
    inset_min_px: int = 1,
    inset_max_px: int = 4,
    inset_ratio: float = 0.02,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve the actual bright/neutral balloon interior inside a manual ROI.

    The user's rectangle is only a *search window*.  It may include clothes,
    hair, panel borders or the balloon outline.  Build a trusted interior from
    TARGET white-paper evidence, fill text holes, then erode a small safety ring.
    If the white component is too weak/fragmented, fall back to the historical
    eroded ROI so old projects remain usable.
    """
    if target.shape[:2] != region.shape:
        raise ValueError("white-container target/region shape mismatch")
    use = region > 0
    out = np.zeros(region.shape, np.uint8)
    ys, xs = np.where(use)
    if xs.size == 0:
        return out, {"white_container_detected": False, "reason": "empty_region"}
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    rw, rh = max(1, x1 - x0), max(1, y1 - y0)
    crop = target[y0:y1, x0:x1]
    local_region = use[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # White manga balloon paper is bright and nearly neutral.  Keep the gate
    # tolerant enough for JPEG/scan tint, then close small lettering holes.
    paper = ((gray >= 200) & (hsv[..., 1] <= 62) & local_region).astype(np.uint8) * 255
    kclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kclose, iterations=2)
    count, labels, stats, cents = cv2.connectedComponentsWithStats((paper > 0).astype(np.uint8), 8)
    region_area = max(1, int(np.count_nonzero(local_region)))
    cx_roi = (rw - 1) * 0.5; cy_roi = (rh - 1) * 0.5
    candidates: list[tuple[float, int, int]] = []
    for lab in range(1, count):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < max(30, int(region_area * 0.035)):
            continue
        cx, cy = map(float, cents[lab])
        dist = ((cx - cx_roi) ** 2 + (cy - cy_roi) ** 2) ** 0.5 / max(1.0, (rw * rw + rh * rh) ** 0.5)
        # Prefer a large white component near the user-selected area center.
        score = float(area / region_area) - 0.22 * float(dist)
        candidates.append((score, area, lab))

    detected = False
    detected_fraction = 0.0
    safe_local = np.zeros((rh, rw), np.uint8)
    if candidates:
        _score, area, chosen = max(candidates)
        comp = (labels == chosen).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(comp)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
        else:
            filled = comp
        filled[~local_region] = 0
        detected_fraction = float(cv2.countNonZero(filled) / region_area)
        detected = detected_fraction >= 0.12
        if detected:
            safe_local = filled

    lo = max(0, int(inset_min_px)); hi = max(lo, int(inset_max_px))
    inset = max(lo, min(hi, int(round(min(rw, rh) * max(0.0, float(inset_ratio))))))
    if not detected:
        safe_local = local_region.astype(np.uint8) * 255
    if inset > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1))
        safe_local = cv2.erode(safe_local, k, iterations=1)
    safe_local[~local_region] = 0
    out[y0:y1, x0:x1] = safe_local
    return out, {
        "white_container_detected": bool(detected),
        "white_container_fraction": float(detected_fraction),
        "container_border_inset_px": int(inset),
        "safe_area": int(cv2.countNonZero(out)),
        "roi_area": int(region_area),
        "background_policy": "target_only",
    }


def _white_container_text_masks(
    aligned_source: np.ndarray, target: np.ndarray, region: np.ndarray,
    *, inset_min_px: int = 1, inset_max_px: int = 4, inset_ratio: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract only compact text from a manually confirmed white container.

    The rectangle/bubble outline is geometry, never text.  Erode a small border
    ring before component extraction so SOURCE/TARGET balloon edges cannot enter
    the transferable glyph mask or distort manual X/Y nudge behavior.
    """
    safe, safe_diag = white_container_safe_mask(
        target, region,
        inset_min_px=inset_min_px,
        inset_max_px=inset_max_px,
        inset_ratio=inset_ratio,
    )
    inset = int(safe_diag.get("container_border_inset_px", 0) or 0)
    src = target_text_mask_in_container(aligned_source, safe)
    tgt = target_text_mask_in_container(target, safe)
    src = prune_source_text_mask(src, safe)
    src, src_diag = strip_border_ring_components(src, safe)
    tgt, tgt_diag = strip_border_ring_components(tgt, safe)
    return src, tgt, {
        'white_container_text_only': True,
        **safe_diag,
        'container_border_inset_px': int(inset),
        'source_pixels': int(cv2.countNonZero(src)),
        'target_clear_pixels': int(cv2.countNonZero(tgt)),
        'background_policy': 'target_only',
        'border_ring_removed_source': src_diag,
        'border_ring_removed_target': tgt_diag,
    }

def estimate_open_text_masks(
    aligned_source: np.ndarray,
    target: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
    *,
    diff_threshold: int = 24,
    edge_threshold: float = 52.0,
    expand_px: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Separate SOURCE Chinese glyphs from TARGET Japanese glyphs in a manual ROI.

    v1.0 uses paired structural difference only as a *seed*.  The seed is then
    grown through local stroke evidence so whole glyph interiors are recovered.
    For monochrome SOURCE -> colour TARGET pairs, a wider edge correspondence
    radius is used to tolerate colourization/scan jitter without treating burst
    rays and artwork outlines as translated text.
    """
    h, w = target.shape[:2]
    x0, y0, x1, y1 = map(int, bbox)
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    src_mask = np.zeros((h, w), np.uint8)
    tgt_mask = np.zeros((h, w), np.uint8)
    if x1 <= x0 or y1 <= y0:
        return src_mask, tgt_mask, {"reason": "empty_bbox", "bbox": [x0, y0, x1, y1]}

    s = aligned_source[y0:y1, x0:x1]
    t = target[y0:y1, x0:x1]
    diff = np.max(np.abs(s.astype(np.int16) - t.astype(np.int16)), axis=2).astype(np.uint8)
    sg = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    tg = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    dthr = max(6, int(diff_threshold))
    ethr = max(8.0, float(edge_threshold))
    low = max(24, int(round(ethr * 0.70)))
    high = max(low + 16, int(round(ethr * 1.65)))
    source_edges = cv2.Canny(sg, low, high, L2gradient=True) > 0
    target_edges = cv2.Canny(tg, low, high, L2gradient=True) > 0

    source_sat = float(np.percentile(cv2.cvtColor(s, cv2.COLOR_BGR2HSV)[..., 1], 90.0))
    target_sat = float(np.percentile(cv2.cvtColor(t, cv2.COLOR_BGR2HSV)[..., 1], 90.0))
    cross_rendition = source_sat < 24.0 and target_sat >= 24.0
    edge_radius = max(1, min(2, int(expand_px)))
    if cross_rendition:
        edge_radius = max(edge_radius, 4)
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_radius + 1, 2 * edge_radius + 1))
    source_near_target = cv2.dilate(target_edges.astype(np.uint8), ek) > 0
    target_near_source = cv2.dilate(source_edges.astype(np.uint8), ek) > 0
    source_seed = ((diff >= dthr) & source_edges & (~source_near_target)).astype(np.uint8) * 255
    target_seed = ((diff >= dthr) & target_edges & (~target_near_source)).astype(np.uint8) * 255

    # First recover full stroke interiors from local light/dark contrast, then
    # remove border-spanning / artwork-like components.
    stroke_kernel = max(9, 2 * max(1, int(expand_px)) + 7)
    src_local = _grow_seed_into_local_strokes(sg, source_seed, kernel_px=stroke_kernel, contrast_threshold=8)
    tgt_local = _grow_seed_into_local_strokes(tg, target_seed, kernel_px=stroke_kernel, contrast_threshold=8)
    src_local = _cleanup_components(src_local, min_area=2, max_fraction=0.50)
    tgt_local = _cleanup_components(tgt_local, min_area=2, max_fraction=0.50)
    src_local = _filter_text_like_components(src_local, (y1 - y0, x1 - x0))
    tgt_local = _filter_text_like_components(tgt_local, (y1 - y0, x1 - x0))
    if cv2.countNonZero(tgt_local):
        # One-pixel AA safety ring is enough after interior growth; the old two
        # large dilations were responsible for purple burst-background damage.
        tgt_local = cv2.dilate(tgt_local, np.ones((3, 3), np.uint8), iterations=1)

    # v1.0.7: promote changed *whole glyph components* rather than leaving the
    # reviewer with edge fragments.  The shared text-only extractor excludes
    # SOURCE paper/background and common artwork by construction, so even a
    # generously drawn manual box remains a lettering-only edit.
    region = np.zeros((h, w), np.uint8)
    region[y0:y1, x0:x1] = 255
    try:
        comp_src, comp_tgt, comp_diag = changed_text_masks(
            aligned_source, target, region, tolerance_px=max(1, min(3, int(expand_px)))
        )
        # Complex colour pages sometimes leave only a few SOURCE-exclusive edge
        # pixels under the normal gate.  Retry with a relaxed uniqueness ratio,
        # but keep the same compact-component/artwork filters.
        min_expected = max(10, int(round((x1 - x0) * (y1 - y0) * 0.00035)))
        if cv2.countNonZero(comp_src) < min_expected:
            relaxed_src, relaxed_tgt, relaxed_diag = changed_text_masks(
                aligned_source, target, region,
                tolerance_px=max(1, min(4, int(expand_px) + 1)),
                min_unique_ratio=0.018,
                max_component_fraction=0.065,
            )
            if cv2.countNonZero(relaxed_src) > cv2.countNonZero(comp_src):
                comp_src, comp_tgt = relaxed_src, relaxed_tgt
                comp_diag = {**dict(comp_diag), "relaxed_fallback": True, "relaxed": relaxed_diag}
    except Exception as exc:
        comp_src = np.zeros_like(src_mask); comp_tgt = np.zeros_like(tgt_mask)
        comp_diag = {"reason": f"component_extract_failed:{exc}"}
    if cv2.countNonZero(comp_src) > 0:
        src_mask = comp_src
    else:
        src_mask[y0:y1, x0:x1] = src_local
    if cv2.countNonZero(comp_tgt) > 0:
        tgt_mask = comp_tgt
        # The component extractor is intentionally conservative.  For a manual
        # Reveal ROI, recover additional antialias/outline pixels from the older
        # structural target mask only when they are spatially attached to the
        # confirmed SOURCE Chinese text. This removes visible Japanese ghosts
        # without touching burst rays/background elsewhere in the box.
        if cv2.countNonZero(comp_src) > 0 and cv2.countNonZero(tgt_local) > 0:
            edge_full = np.zeros_like(tgt_mask)
            edge_full[y0:y1, x0:x1] = tgt_local
            rad = max(8, min(20, int(round(max(x1 - x0, y1 - y0) * 0.08))))
            assoc = cv2.dilate(
                (comp_src > 0).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad * 2 + 1, rad * 2 + 1)),
            ) > 0
            extra = (edge_full > 0) & assoc
            tgt_mask[extra] = 255
    else:
        tgt_mask[y0:y1, x0:x1] = tgt_local
    corridor_radius = max(24, min(60, int(round(min(x1 - x0, y1 - y0) * 0.60))))
    src_mask, tgt_mask, corridor_diag = _manual_text_corridor(src_mask, tgt_mask, region, radius=corridor_radius)
    area = max(1, (x1 - x0) * (y1 - y0))
    diagnostics = {
        "bbox": [x0, y0, x1, y1],
        "diff_threshold": dthr,
        "edge_threshold": ethr,
        "edge_match_radius": int(edge_radius),
        "cross_rendition": bool(cross_rendition),
        "source_saturation_p90": source_sat,
        "target_saturation_p90": target_sat,
        "source_pixels": int(cv2.countNonZero(src_local)),
        "target_clear_pixels": int(cv2.countNonZero(tgt_local)),
        "source_fraction": float(cv2.countNonZero(src_local) / area),
        "target_clear_fraction": float(cv2.countNonZero(tgt_local) / area),
        "mean_pair_diff": float(np.mean(diff)),
        "text_only_components": comp_diag,
        "background_policy": "target_only",
        "manual_text_corridor": corridor_diag,
    }
    return src_mask, tgt_mask, diagnostics



def _dominant_colored_container_safe_mask(
    target: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
    *,
    min_saturation: int = 70,
    inset_px: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a conservative interior mask for a saturated manual bubble.

    The ROI may also contain coloured artwork. We therefore estimate the fill's
    dominant HSV hue from the central ROI and require hue proximity in addition
    to saturation/value. This keeps a yellow/red burst isolated even when nearby
    roofs, clothes or signs are also saturated.
    """
    h,w=target.shape[:2]; x0,y0,x1,y1=map(int,bbox)
    x0=max(0,min(w,x0)); x1=max(0,min(w,x1)); y0=max(0,min(h,y0)); y1=max(0,min(h,y1))
    full=np.zeros((h,w),np.uint8)
    if x1<=x0 or y1<=y0:
        return full,{"reason":"empty_bbox","colored_container":False}
    roi=target[y0:y1,x0:x1]; hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    hue,sat,val=hsv[...,0],hsv[...,1],hsv[...,2]
    sat_floor=max(int(min_saturation),min(135,int(round(float(np.percentile(sat,65.0))))))
    rh,rw=sat.shape
    cy0,cy1=int(rh*.22),max(int(rh*.78),int(rh*.22)+1); cx0,cx1=int(rw*.22),max(int(rw*.78),int(rw*.22)+1)
    center_use=(sat[cy0:cy1,cx0:cx1]>=max(55,sat_floor-15))&(val[cy0:cy1,cx0:cx1]>=70)
    center_h=hue[cy0:cy1,cx0:cx1][center_use]
    if center_h.size<20:
        all_use=(sat>=max(55,sat_floor-15))&(val>=70); center_h=hue[all_use]
    if center_h.size<20:
        return full,{"reason":"no_saturated_component","colored_container":False,"sat_floor":sat_floor}
    hist=np.bincount(center_h.astype(np.int32),minlength=180); dominant_h=int(np.argmax(hist))
    # Bright burst fills are normally close to the upper value range.  Requiring
    # value proximity prevents similarly-hued brown/red artwork outside the
    # bubble from joining the same saturation component.
    center_val=val[cy0:cy1,cx0:cx1][center_use]
    dominant_v=float(np.percentile(center_val,72.0)) if center_val.size else 255.0
    value_floor=max(70,int(round(dominant_v-48.0)))
    hd=np.abs(hue.astype(np.int16)-dominant_h); hd=np.minimum(hd,180-hd)
    hue_tol=14
    raw_seed=((sat>=sat_floor)&(val>=value_floor)&(hd<=hue_tol)).astype(np.uint8)*255
    raw=cv2.morphologyEx(raw_seed,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),iterations=2)
    count,labels,stats,_=cv2.connectedComponentsWithStats((raw>0).astype(np.uint8),8)
    if count<=1:
        return full,{"reason":"no_hue_component","colored_container":False,"sat_floor":sat_floor,"dominant_hue":dominant_h}
    central=np.zeros_like(raw,dtype=bool); central[cy0:cy1,cx0:cx1]=True
    candidates=[]; roi_area=max(1,rh*rw)
    for lab in range(1,count):
        area=int(stats[lab,cv2.CC_STAT_AREA])
        if area<max(40,int(roi_area*.02)): continue
        overlap=int(np.count_nonzero((labels==lab)&central))
        candidates.append((overlap,area,lab))
    if not candidates:
        return full,{"reason":"component_too_small","colored_container":False,"sat_floor":sat_floor,"dominant_hue":dominant_h}
    _,_,chosen=max(candidates)

    # A gradient burst is frequently split into two same-hue islands: a
    # saturated cap above the almost-white centre and a second cap below it.
    # Picking only the largest connected component truncates the bubble exactly
    # through the text area.  Recover nearby central islands and fill their
    # convex envelope, but only when the envelope remains a local bounded
    # object.  This is intentionally conservative and applies only inside a
    # user/local detector bbox; unrelated coloured artwork should therefore not
    # be bridged across the page.
    chosen_stat=stats[chosen]
    cbx0=int(chosen_stat[cv2.CC_STAT_LEFT]); cby0=int(chosen_stat[cv2.CC_STAT_TOP])
    cbx1=cbx0+int(chosen_stat[cv2.CC_STAT_WIDTH]); cby1=cby0+int(chosen_stat[cv2.CC_STAT_HEIGHT])
    ex=max(8,int(round(rw*.12))); ey=max(8,int(round(rh*.12)))
    ex0=max(0,cbx0-ex); ex1=min(rw,cbx1+ex); ey0=max(0,cby0-ey); ey1=min(rh,cby1+ey)
    central_ex=np.zeros_like(raw,dtype=bool)
    cex0=max(0,cx0-int(round(rw*.10))); cex1=min(rw,cx1+int(round(rw*.10)))
    cey0=max(0,cy0-int(round(rh*.10))); cey1=min(rh,cy1+int(round(rh*.10)))
    central_ex[cey0:cey1,cex0:cex1]=True
    merge_labels=[int(chosen)]
    for overlap,area,lab in candidates:
        lab=int(lab)
        if lab==chosen:
            continue
        sx=int(stats[lab,cv2.CC_STAT_LEFT]); sy=int(stats[lab,cv2.CC_STAT_TOP])
        sw=int(stats[lab,cv2.CC_STAT_WIDTH]); sh=int(stats[lab,cv2.CC_STAT_HEIGHT])
        sx1=sx+sw; sy1=sy+sh
        near = not (sx1 < ex0 or sx > ex1 or sy1 < ey0 or sy > ey1)
        central_near = int(np.count_nonzero((labels==lab)&central_ex)) > 0
        if near and central_near and area>=max(40,int(roi_area*.015)):
            merge_labels.append(lab)

    comp=np.isin(labels,merge_labels).astype(np.uint8)*255
    contours,_=cv2.findContours(comp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    filled=np.zeros_like(comp)
    hull_recovered=False
    hull_fraction=0.0
    if len(contours)>1 and len(merge_labels)>1:
        points=np.vstack([c.reshape(-1,2) for c in contours if c.size]).astype(np.int32)
        if points.size:
            hull=cv2.convexHull(points.reshape(-1,1,2))
            hull_mask=np.zeros_like(comp)
            cv2.drawContours(hull_mask,[hull],-1,255,thickness=cv2.FILLED)
            hull_fraction=float(cv2.countNonZero(hull_mask)/max(1,roi_area))
            # A real speech/effect container can occupy most of a tight local
            # box, but a hull that consumes virtually the whole ROI is too risky.
            if 0.0 < hull_fraction <= 0.86:
                filled=hull_mask
                hull_recovered=True
    if cv2.countNonZero(filled)==0:
        if contours: cv2.drawContours(filled,contours,-1,255,thickness=cv2.FILLED)
        else: filled=comp
    comp_pixels=int(cv2.countNonZero(comp)); comp_fraction=float(comp_pixels/max(1,roi_area))
    # Judge whether this is a bounded colour container from the *pre-close*
    # colour evidence. Morphological closing deliberately fills text holes and
    # can expand a legitimate burst by a few pixels until it touches the manual
    # ROI boundary; using that expanded contour for the full-frame guard would
    # incorrectly reject a tightly boxed red/yellow caption.
    seed_comp=((comp>0)&(raw_seed>0)).astype(np.uint8)*255
    ys,xs=np.where(seed_comp>0)
    if xs.size==0:
        ys,xs=np.where(comp>0)
    if xs.size==0:
        return full,{"reason":"empty_component","colored_container":False,"sat_floor":sat_floor,"dominant_hue":dominant_h}
    bx0,bx1=int(xs.min()),int(xs.max())+1; by0,by1=int(ys.min()),int(ys.max())+1
    bw=max(1,bx1-bx0); bh=max(1,by1-by0)
    seed_pixels=int(cv2.countNonZero(seed_comp)); seed_fraction=float(seed_pixels/max(1,roi_area))
    touch_left=bx0<=1; touch_top=by0<=1; touch_right=bx1>=rw-1; touch_bottom=by1>=rh-1
    touch_sides=int(touch_left)+int(touch_top)+int(touch_right)+int(touch_bottom)
    # A tight manual box can legitimately be almost entirely filled by one
    # coloured burst.  Distinguish that from selecting a patch of a much larger
    # same-colour background by looking immediately *outside* the ROI.  If the
    # same saturated hue continues across multiple sides, it is a background
    # field rather than a bounded caption/container.
    suspicious_full = bool(seed_fraction >= 0.94 or touch_sides >= 3 or (bw >= int(rw*0.97) and bh >= int(rh*0.97)))
    outside_match_sides = 0
    outside_checked = 0
    if suspicious_full:
        full_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        band = max(2, min(8, int(round(min(rw, rh) * 0.035))))
        strips = []
        if x0 > 0: strips.append(full_hsv[y0:y1, max(0,x0-band):x0])
        if x1 < w: strips.append(full_hsv[y0:y1, x1:min(w,x1+band)])
        if y0 > 0: strips.append(full_hsv[max(0,y0-band):y0, x0:x1])
        if y1 < h: strips.append(full_hsv[y1:min(h,y1+band), x0:x1])
        for strip in strips:
            if strip.size == 0: continue
            outside_checked += 1
            sh, ss, sv = strip[...,0], strip[...,1], strip[...,2]
            dh = np.abs(sh.astype(np.int16)-dominant_h); dh=np.minimum(dh,180-dh)
            same = (ss >= max(45, sat_floor-15)) & (sv >= max(55, value_floor-20)) & (dh <= hue_tol+4)
            if float(np.mean(same)) >= 0.55:
                outside_match_sides += 1
        roi_page_fraction=float((rw*rh)/max(1,w*h))
        reject_field = outside_match_sides >= 2 or (outside_checked == 0 and roi_page_fraction >= 0.45)
        if reject_field:
            return full,{"reason":"dominant_full_frame_color_field","colored_container":False,"sat_floor":sat_floor,"dominant_hue":dominant_h,
                         "component_fraction":comp_fraction,"seed_fraction":seed_fraction,"touch_sides":touch_sides,
                         "outside_match_sides":outside_match_sides,"outside_checked":outside_checked,"component_bbox":[bx0,by0,bx1,by1]}
    inset=max(1,int(inset_px)); safe=cv2.erode(filled,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*inset+1,2*inset+1)),iterations=1)
    full[y0:y1,x0:x1]=safe
    return full,{
        "colored_container":True,"sat_floor":int(sat_floor),"dominant_hue":int(dominant_h),"hue_tolerance":int(hue_tol),"dominant_value":float(dominant_v),"value_floor":int(value_floor),
        "component_area":comp_pixels,"component_fraction":comp_fraction,"seed_fraction":seed_fraction,"safe_area":int(cv2.countNonZero(safe)),"bbox":[x0,y0,x1,y1],"component_bbox":[bx0,by0,bx1,by1],"touch_sides":int(touch_sides),"outside_match_sides":int(outside_match_sides),"outside_checked":int(outside_checked),
        "merged_component_labels":[int(x) for x in merge_labels],"merged_component_count":int(len(merge_labels)),"gradient_hull_recovered":bool(hull_recovered),"gradient_hull_fraction":float(hull_fraction),
    }

def _augment_colored_manual_text_masks(
    aligned_source: np.ndarray,
    target: np.ndarray,
    bbox: list[int] | tuple[int,int,int,int],
    source_mask: np.ndarray,
    target_clear_mask: np.ndarray,
) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    """Recover CJK masks inside a manually confirmed saturated container.

    This is a fallback/augmentation, not a replacement for paired-difference
    masks.  It fixes the common case where the red/yellow fill dominates the
    SOURCE/TARGET difference and the old algorithm returns an empty Reveal seed.
    """
    safe,cdiag=_dominant_colored_container_safe_mask(target,bbox)
    if cv2.countNonZero(safe)==0:
        return source_mask,target_clear_mask,cdiag
    # SOURCE is often monochrome and its Chinese glyphs are cleanly separable
    # once the TARGET colour container gives us the correct safe interior.
    src_extra=target_text_mask_in_container(aligned_source,safe)
    src_extra=prune_source_text_mask(src_extra,safe)
    src_extra,_=strip_border_ring_components(src_extra,safe)

    # For coloured TARGET fills, grayscale thresholds are unreliable (pure red
    # is dark in luminance). Detect Japanese by a value drop relative to the
    # high-value local fill, then keep compact components only.
    hsv=cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
    use=safe>0
    vals=hsv[...,2][use]
    fill_v=float(np.percentile(vals,78.0)) if vals.size else 255.0
    dark=((hsv[...,2].astype(np.float32) <= max(35.0,fill_v-42.0)) & use).astype(np.uint8)*255
    # Include truly black/neutral strokes even if JPEG colour fringes raise V.
    bgr_max=np.max(target,axis=2)
    dark[((bgr_max<150)&use)]=255
    dark=cv2.morphologyEx(dark,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    tgt_extra,_=strip_border_ring_components(dark,safe)
    if cv2.countNonZero(tgt_extra):
        tgt_extra=cv2.dilate(tgt_extra,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
        tgt_extra[safe==0]=0

    # Preserve helpful baseline target-clear evidence only when it is spatially
    # associated with the translated Chinese text.  This recovers antialias/
    # fringe pixels the colour-only extractor can miss, while the saturated
    # container mask prevents the whole burst/background from joining the clear.
    baseline_src=int(cv2.countNonZero(source_mask)); baseline_tgt=int(cv2.countNonZero(target_clear_mask))
    text_authority = np.maximum(src_extra, source_mask)
    merged_tgt = tgt_extra.copy()
    corridor_radius = 0
    if cv2.countNonZero(text_authority) > 0 and cv2.countNonZero(target_clear_mask) > 0:
        x0, y0, x1, y1 = map(int, bbox)
        bw = max(1, x1 - x0); bh = max(1, y1 - y0)
        corridor_radius = max(10, min(26, int(round(min(bw, bh) * 0.09))))
        near = cv2.dilate(
            (text_authority > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (corridor_radius * 2 + 1, corridor_radius * 2 + 1)),
        ) > 0
        recovered = ((target_clear_mask > 0) & (safe > 0) & near).astype(np.uint8) * 255
        merged_tgt = np.maximum(merged_tgt, recovered)
    if cv2.countNonZero(merged_tgt):
        merged_tgt,_=strip_border_ring_components(merged_tgt,safe)
        merged_tgt=cv2.dilate(merged_tgt,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
        merged_tgt[safe==0]=0

    # IMPORTANT: this routine is an *augmentation*.  Older builds accidentally
    # replaced a good paired-difference SOURCE mask whenever the colour-specific
    # extractor found as few as eight pixels.  On red/pink gradient bursts the
    # safe colour component can cover only one cap, so a complete 10k+ pixel
    # Chinese mask was replaced by a few hundred pixels and the UI appeared to
    # "recognise one bubble but not the other".  Never downgrade existing text
    # evidence: union the two conservative text masks and let the normal pruning
    # stage below keep only compact glyph components inside the requested ROI.
    src_out=np.maximum(source_mask,src_extra)
    tgt_out=np.maximum(target_clear_mask,merged_tgt)
    source_extra_px=int(cv2.countNonZero(src_extra))
    merged_target_px=int(cv2.countNonZero(merged_tgt))
    source_out_px=int(cv2.countNonZero(src_out))
    target_out_px=int(cv2.countNonZero(tgt_out))
    baseline_preserved=bool(baseline_src>0 and source_out_px>=baseline_src)
    source_policy="union_preserve_baseline" if baseline_src>0 and source_extra_px>0 else ("colored_fallback" if source_extra_px>0 else "baseline_only")
    cdiag.update({
        "source_extra_pixels":source_extra_px,
        "target_extra_pixels":int(cv2.countNonZero(tgt_extra)),
        "baseline_source_pixels":baseline_src,
        "baseline_target_pixels":baseline_tgt,
        "merged_target_pixels":merged_target_px,
        "output_source_pixels":source_out_px,
        "output_target_pixels":target_out_px,
        "source_mask_policy":source_policy,
        "baseline_source_preserved":baseline_preserved,
        "colored_target_corridor_radius":int(corridor_radius),
        "authoritative_colored_text_masks":bool(source_extra_px>=8 and merged_target_px>=8),
        "fill_value_p78":float(fill_v),
    })
    return src_out,tgt_out,cdiag


def clean_manual_target_text(
    target: np.ndarray,
    clear_mask: np.ndarray,
    *,
    bbox: list[int] | tuple[int,int,int,int] | None = None,
    honor_mask_outside_colored_safe: bool = False,
) -> tuple[np.ndarray,dict[str,Any]]:
    """Clear manual TARGET text without smearing flat coloured bubbles.

    Root cause of the old coloured-page failure: the yellow/red burst interior
    is not actually a flat solid.  Treating it as "complex art" and inpainting
    each glyph component separately produced blurry seams and residual strokes.
    The new path still keeps the TARGET background authoritative, but when a
    dominant coloured container is detected it repairs text against the whole
    safe interior first, falling back to a true flat-colour fill only when the
    container really is flat.
    """
    mask=(np.asarray(clear_mask)>0).astype(np.uint8)*255
    out=target.copy()
    if cv2.countNonZero(mask)==0:
        return out,{"mode":"none","pixels":0}

    def _repair_local_components(dst: np.ndarray, work_mask: np.ndarray) -> tuple[np.ndarray,int]:
        """Inpaint only explicitly requested components, never a whole ROI."""
        work=(np.asarray(work_mask)>0).astype(np.uint8)*255
        count,labels,stats,_=cv2.connectedComponentsWithStats((work>0).astype(np.uint8),8)
        blocks=0
        for lab in range(1,count):
            x,y,bw,bh,area=[int(v) for v in stats[lab]]
            if area<=0: continue
            pad=max(8,min(32,int(round(max(bw,bh)*0.75))))
            x0=max(0,x-pad); y0=max(0,y-pad); x1=min(target.shape[1],x+bw+pad); y1=min(target.shape[0],y+bh+pad)
            crop=target[y0:y1,x0:x1]
            cm=np.where(labels[y0:y1,x0:x1]==lab,255,0).astype(np.uint8)
            repaired=cv2.inpaint(crop,cm,3.0,cv2.INPAINT_TELEA)
            sel=cm>0
            view=dst[y0:y1,x0:x1]
            view[sel]=repaired[sel]
            blocks+=1
        return dst,blocks

    safe=None; cdiag={}
    if bbox is not None and len(bbox)==4:
        safe,cdiag=_dominant_colored_container_safe_mask(target,bbox)
    if safe is not None and cv2.countNonZero(safe)>0:
        use=(safe>0)
        safe_mask=(safe>0).astype(np.uint8)*255
        outside_mask=np.zeros_like(mask)
        if honor_mask_outside_colored_safe:
            outside_mask[(mask>0)&(~use)]=255
        outside_pixels=int(cv2.countNonZero(outside_mask))
        # Recover antialias fringes that the conservative clear mask may miss,
        # but keep the expansion strictly inside the coloured container.
        repair_mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
        repair_mask[safe==0]=0
        exclude=cv2.dilate(repair_mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),iterations=1)>0
        samples=target[use & (~exclude)]
        if samples.shape[0]>=40:
            med=np.median(samples,axis=0)
            mad=np.median(np.abs(samples.astype(np.float32)-med.astype(np.float32)),axis=0)
            # Do not decide "flat colour" from the pixels left after excluding
            # the clear mask: a broad reviewer stroke can remove the pale centre
            # of a red/pink gradient and leave only a uniform red rim, which made
            # the old code fill the whole speech bubble solid red.  Estimate the
            # underlying field from a strongly median-blurred copy of the *whole
            # safe interior* so thin Japanese glyphs disappear while real radial
            # gradients remain measurable.
            smooth=cv2.medianBlur(target,11)
            field=smooth[use]
            if field.shape[0]>=40:
                p10=np.percentile(field.astype(np.float32),10.0,axis=0)
                p90=np.percentile(field.astype(np.float32),90.0,axis=0)
                field_span=float(np.max(p90-p10))
                field_mad=float(np.max(np.median(np.abs(field.astype(np.float32)-np.median(field.astype(np.float32),axis=0)),axis=0)))
            else:
                field_span=255.0; field_mad=255.0
            # Robust spread is tolerant of JPEG grain but rejects radial/linear
            # gradients and artwork. Both local sample spread and full-field
            # spread must agree that the fill is genuinely flat.
            if float(np.max(mad))<=18.0 and field_span<=30.0 and field_mad<=16.0:
                sel=(repair_mask>0)&use
                out[sel]=np.clip(np.round(med),0,255).astype(np.uint8)
                outside_blocks=0
                if outside_pixels:
                    out,outside_blocks=_repair_local_components(out,outside_mask)
                return out,{"mode":"target_flat_color","pixels":int(np.count_nonzero(sel))+outside_pixels,"median_bgr":[float(x) for x in med],"mad_max":float(np.max(mad)),"field_span_p10_p90":field_span,"field_mad_max":field_mad,"container":cdiag,"authoritative_outside_pixels":outside_pixels,"outside_cleanup_blocks":int(outside_blocks)}
        ys, xs = np.where(repair_mask > 0)
        if ys.size:
            pad = 10
            x0=max(0,int(xs.min())-pad); y0=max(0,int(ys.min())-pad)
            x1=min(target.shape[1],int(xs.max())+pad+1); y1=min(target.shape[0],int(ys.max())+pad+1)
            crop=target[y0:y1,x0:x1].copy()
            cm=repair_mask[y0:y1,x0:x1].copy()
            # Use the full coloured interior as context so gradients survive.
            repaired=cv2.inpaint(crop,cm,4.0,cv2.INPAINT_TELEA)
            sel=cm>0
            view=out[y0:y1,x0:x1]
            view[sel]=repaired[sel]
            outside_blocks=0
            if outside_pixels:
                out,outside_blocks=_repair_local_components(out,outside_mask)
            return out,{"mode":"target_colored_container_inpaint","pixels":int(np.count_nonzero(sel))+outside_pixels,"bbox":[x0,y0,x1,y1],"container":cdiag,"authoritative_outside_pixels":outside_pixels,"outside_cleanup_blocks":int(outside_blocks)}
    # Complex fallback: inpaint each connected text component in a small local
    # crop and paste only its mask. Avoids whole-ROI softening.
    out,blocks=_repair_local_components(out,mask)
    return out,{"mode":"local_component_inpaint","pixels":int(cv2.countNonZero(mask)),"blocks":int(blocks),"container":cdiag}

def build_reveal_seed_mask(source_mask: np.ndarray, target_clear_mask: np.ndarray, *, padding_px: int = 5) -> np.ndarray:
    """Build a broad editable reveal window from paired text evidence.

    The seed deliberately covers both the SOURCE Chinese glyph alpha and the
    TARGET Japanese clear mask.  The reviewer can then add/remove coverage with
    a brush instead of having to trace every antialiased stroke from scratch.
    """
    if source_mask.shape != target_clear_mask.shape:
        raise ValueError("source/target reveal masks must share the same shape")
    seed = np.maximum((source_mask > 0).astype(np.uint8), (target_clear_mask > 0).astype(np.uint8)) * 255
    if cv2.countNonZero(seed) == 0:
        return seed
    pad = max(0, int(padding_px))
    if pad > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        seed = cv2.dilate(seed, k, iterations=1)
    return seed


def apply_reveal_window(
    source_mask: np.ndarray,
    target_clear_mask: np.ndarray,
    reveal_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gate Chinese transfer and Japanese removal with one reviewer reveal mask."""
    if source_mask.shape != target_clear_mask.shape or source_mask.shape != reveal_mask.shape:
        raise ValueError("reveal masks must share the same shape")
    gate = (reveal_mask > 0).astype(np.uint8) * 255
    return cv2.bitwise_and(source_mask, gate), cv2.bitwise_and(target_clear_mask, gate)


def estimate_source_background(aligned_source: np.ndarray, source_mask: np.ndarray, *, radius: float = 3.0) -> np.ndarray:
    """Estimate source background under the Chinese glyph mask.

    This is the key for colour pages: the translated source may be monochrome or
    low-quality, but we only want the *text contribution* relative to its local
    background, not the old page background itself.
    """
    if aligned_source.shape[:2] != source_mask.shape:
        raise ValueError("source/background mask shape mismatch")
    mask = (source_mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(mask) == 0:
        return aligned_source.copy()
    return cv2.inpaint(aligned_source, mask, max(1.0, float(radius)), cv2.INPAINT_TELEA)


def composite_source_text_delta(
    base: np.ndarray,
    aligned_source: np.ndarray,
    source_mask: np.ndarray,
    *,
    source_background: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transfer text-only colour/luminance contribution onto the TARGET background.

    Each connected text component is polarity-aware.  Predominantly black text
    may only darken the target; predominantly white text may only lighten it.
    This removes the white/purple halos produced by expanded antialias/background
    pixels while still supporting genuine white or outlined SFX lettering.
    """
    if base.shape != aligned_source.shape or base.shape[:2] != source_mask.shape:
        raise ValueError("delta composite inputs must share canvas size")
    if source_background is None:
        source_background = estimate_source_background(aligned_source, source_mask)
    if source_background.shape != aligned_source.shape:
        raise ValueError("source background shape mismatch")
    gate = (source_mask > 0).astype(np.float32)
    if alpha is None:
        alpha_f = gate
    else:
        alpha_f = np.asarray(alpha, dtype=np.float32)
        if alpha_f.shape != source_mask.shape:
            raise ValueError("alpha shape mismatch")
        alpha_f = np.clip(alpha_f, 0.0, 1.0) * gate

    raw_delta = aligned_source.astype(np.float32) - source_background.astype(np.float32)
    sg = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.cvtColor(source_background, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lum_delta = sg - bg
    filtered = raw_delta.copy()
    labels_count, labels, _stats, _ = cv2.connectedComponentsWithStats((source_mask > 0).astype(np.uint8), 8)
    dark_components = light_components = mixed_components = 0
    for i in range(1, labels_count):
        comp = labels == i
        strong = comp & (np.abs(lum_delta) >= 3.0)
        if not np.any(strong):
            filtered[comp] = 0.0
            continue
        neg_frac = float(np.mean(lum_delta[strong] < -3.0))
        pos_frac = float(np.mean(lum_delta[strong] > 3.0))
        if neg_frac >= 0.60 and pos_frac < 0.25:
            keep = comp & (lum_delta < -1.0)
            filtered[comp & ~keep] = 0.0
            dark_components += 1
        elif pos_frac >= 0.60 and neg_frac < 0.25:
            keep = comp & (lum_delta > 1.0)
            filtered[comp & ~keep] = 0.0
            light_components += 1
        else:
            mixed_components += 1

    out = base.astype(np.float32) + filtered * alpha_f[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    mag = np.max(np.abs(filtered), axis=2)
    strong = (alpha_f > 0.0) & (mag >= 4.0)
    return out, {
        "delta_pixels": int(np.count_nonzero(strong)),
        "max_delta": float(np.max(mag)) if mag.size else 0.0,
        "mean_delta": float(np.mean(mag[strong])) if np.any(strong) else 0.0,
        "dark_components": int(dark_components),
        "light_components": int(light_components),
        "mixed_components": int(mixed_components),
    }


def build_manual_effect_masks(
    source: np.ndarray,
    target: np.ndarray,
    project: dict[str, Any],
    row: dict[str, Any],
    config: Any | None = None,
) -> ManualEffectMasks:
    bbox = as_list(row.get("target_bbox"))
    if len(bbox) != 4:
        raise ValueError("manual effect region requires target_bbox=[x0,y0,x1,y1]")
    # Extract text at the registered geometry first.  Manual X/Y nudge is an
    # explicit post-registration layer translation; applying it before text
    # extraction made the selected components change unpredictably as the image
    # moved under the ROI, so a requested +3/-2 px could produce a different
    # apparent displacement.
    aligned, identity = align_source_to_target(
        source,
        target.shape[:2],
        project,
        source_offset_x=0,
        source_offset_y=0,
    )
    manual_dx=int(row.get("source_offset_x", 0) or 0)
    manual_dy=int(row.get("source_offset_y", 0) or 0)
    mode = str(row.get("mode", "effect_text") or "effect_text")
    h, w = target.shape[:2]
    x0, y0, x1, y1 = map(int, bbox)
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    source_mask = np.zeros((h, w), np.uint8)
    clear_mask = np.zeros((h, w), np.uint8)
    diag: dict[str, Any]
    if mode in {"full_patch", "white_bubble_text"}:
        region = np.zeros((h, w), np.uint8)
        if x1 > x0 and y1 > y0:
            region[y0:y1, x0:x1] = 255
        cfg_obj = getattr(config, "mask_replace", config) if config is not None else None
        source_mask, clear_mask, diag = _white_container_text_masks(
            aligned, target, region,
            inset_min_px=int(getattr(cfg_obj, "white_container_manual_inset_min_px", 1)) if cfg_obj is not None else 1,
            inset_max_px=int(getattr(cfg_obj, "white_container_manual_inset_max_px", 4)) if cfg_obj is not None else 4,
            inset_ratio=float(getattr(cfg_obj, "white_container_manual_inset_ratio", 0.02)) if cfg_obj is not None else 0.02,
        )
        diag.update({"bbox": [x0, y0, x1, y1], "mode": "white_bubble_text", "legacy_full_patch_requested": mode == "full_patch"})
    else:
        source_mask, clear_mask, diag = estimate_open_text_masks(
            aligned,
            target,
            [x0, y0, x1, y1],
            diff_threshold=int(row.get("diff_threshold", 24) or 24),
            edge_threshold=float(row.get("edge_threshold", 52.0) or 52.0),
            expand_px=int(row.get("expand_px", 2) or 2),
        )
        if mode in {"reveal_text", "effect_text", "open_text_box"}:
            source_mask, clear_mask, color_diag = _augment_colored_manual_text_masks(
                aligned, target, [x0, y0, x1, y1], source_mask, clear_mask
            )
            diag["colored_manual_fallback"] = color_diag
            if mode in {"effect_text", "open_text_box"} and bool(color_diag.get("authoritative_colored_text_masks")):
                diag["effect_text_colored_upgrade"] = True
        diag["mode"] = mode
    if cv2.countNonZero(source_mask) > 0:
        region = np.zeros((h, w), np.uint8)
        if x1 > x0 and y1 > y0:
            region[y0:y1, x0:x1] = 255
        source_mask = prune_source_text_mask(source_mask, region)
        diag["source_pixels_pruned"] = int(cv2.countNonZero(source_mask))
    if manual_dx or manual_dy:
        M=np.asarray([[1.0,0.0,float(manual_dx)],[0.0,1.0,float(manual_dy)]],np.float32)
        aligned=cv2.warpAffine(aligned,M,(w,h),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
        source_mask=cv2.warpAffine(source_mask,M,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        source_mask[0:y0,:]=0; source_mask[y1:h,:]=0; source_mask[:,0:x0]=0; source_mask[:,x1:w]=0
        identity=False
    if not bool(row.get("auto_clear_target", True)):
        clear_mask[:] = 0
        diag["target_clear_pixels"] = 0
    diag["identity_pixel_lock"] = bool(identity)
    diag["source_offset"] = [manual_dx, manual_dy]
    return ManualEffectMasks(aligned, source_mask, clear_mask, diag)


def render_open_text_box(
    source: np.ndarray,
    target: np.ndarray,
    project: dict[str, Any],
    row: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Render one manually boxed open-text region with this mode's own renderer.

    The user rectangle is only a semantic gate.  Actual writes remain limited to
    changed SOURCE/TARGET glyph components selected by the mode-private
    open/complex-text transfer path. No OCR or rectangular SOURCE background
    copy is performed.
    """
    bbox = as_list(row.get("target_bbox"))
    if len(bbox) != 4:
        raise ValueError("open text box requires target_bbox=[x0,y0,x1,y1]")
    h, w = target.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0=max(0,min(w,x0)); x1=max(0,min(w,x1)); y0=max(0,min(h,y0)); y1=max(0,min(h,y1))
    if x1-x0 < 3 or y1-y0 < 3:
        raise ValueError("开放文字框选区域太小")
    aligned, identity = align_source_to_target(
        source, target.shape[:2], project,
        source_offset_x=int(row.get("source_offset_x",0) or 0),
        source_offset_y=int(row.get("source_offset_y",0) or 0),
    )
    region=np.zeros((h,w),np.uint8); region[y0:y1,x0:x1]=255
    from . import transfer_ops as _mode_transfer_ops
    cfg_obj=getattr(getattr(config, "hybrid", None), "mask", config)
    rendered, write_mask, source_mask, diag = _mode_transfer_ops._transfer_open_complex_text_region(
        aligned, target, region, cfg_obj
    )
    if rendered is None:
        raise ValueError(str((diag or {}).get("reason") or "open_text_component_transfer_failed"))
    clear_mask=np.asarray((diag or {}).get("clear_mask", np.zeros((h,w),np.uint8)),np.uint8)
    if clear_mask.shape != (h,w):
        clear_mask=np.zeros((h,w),np.uint8)
    # A manual box is a hard authority boundary. The internal open-text renderer
    # may inspect a small padded neighborhood to classify connected glyphs, but
    # it may never write, clear or restore pixels outside the user's rectangle.
    write_mask=cv2.bitwise_and(np.asarray(write_mask,np.uint8),region)
    source_mask=cv2.bitwise_and(np.asarray(source_mask,np.uint8),region)
    clear_mask=cv2.bitwise_and(clear_mask,region)
    clipped=target.copy()
    clipped[write_mask>0]=rendered[write_mask>0]
    return {
        "aligned_source": aligned,
        "rendered": clipped,
        "write_mask": write_mask,
        "source_mask": source_mask,
        "target_clear_mask": clear_mask,
        "diagnostics": {**dict(diag or {}), "manual_open_text_box": True, "identity_pixel_lock": bool(identity), "bbox":[x0,y0,x1,y1], "ocr_used": False},
    }
