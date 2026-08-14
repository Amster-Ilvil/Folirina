from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .text_only_transfer import changed_text_masks, prune_source_text_mask, target_text_mask_in_container
from .schema_compat import as_dict, as_list


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


def _white_container_text_masks(aligned_source: np.ndarray, target: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract only compact text from a manually confirmed white container.

    The rectangle/bubble outline is geometry, never text.  Erode a small border
    ring before component extraction so SOURCE/TARGET balloon edges cannot enter
    the transferable glyph mask or distort manual X/Y nudge behavior.
    """
    ys,xs=np.where(region>0)
    if xs.size:
        rw=int(xs.max()-xs.min()+1); rh=int(ys.max()-ys.min()+1)
        inset=max(3,min(10,int(round(min(rw,rh)*0.045))))
    else:
        inset=3
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(inset*2+1,inset*2+1))
    safe=cv2.erode((region>0).astype(np.uint8)*255,k,iterations=1)
    src = target_text_mask_in_container(aligned_source, safe)
    tgt = target_text_mask_in_container(target, safe)
    src = prune_source_text_mask(src, safe)
    # A white-bubble selection is a search window, never a raster patch. Keep
    # only compact text-like components so a connected bubble outline or a
    # large dark Japanese layer cannot become the SOURCE transfer layer.
    src = _filter_text_like_components(src, (rh, rw))
    tgt = _filter_text_like_components(tgt, (rh, rw))
    return src, tgt, {
        "white_container_text_only": True,
        "container_border_inset_px": int(inset),
        "source_pixels": int(cv2.countNonZero(src)),
        "target_clear_pixels": int(cv2.countNonZero(tgt)),
        "background_policy": "target_only",
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
    # In a monochrome SOURCE -> colour TARGET pair, a wide corridor lets
    # colourization differences (bubble fill, halftone and burst rays) enter
    # the TARGET erase mask.  That is especially damaging for yellow/purple
    # balloons: inpainting then reconstructs the fill from nearby artwork.
    # Keep the editable ROI broad, but make the actual TARGET erase corridor
    # local to the aligned SOURCE glyphs.  The source/target registration
    # already tolerates a few pixels of scan drift; 16 px is enough for the
    # largest manually selected lettering while preventing container bleed.
    if cross_rendition:
        corridor_radius = min(corridor_radius, 16)
    src_mask, tgt_mask, corridor_diag = _manual_text_corridor(src_mask, tgt_mask, region, radius=corridor_radius)
    # Very saturated, compact manual boxes are usually coloured balloons rather
    # than open artwork. In that case the paired edge mask can still miss dark
    # Japanese glyph interiors after colourization. Recover only compact TARGET
    # lettering components inside the small confirmed box; never do this for a
    # page-sized box, where artwork lines would be legitimate candidates.
    roi_area = max(1, (x1 - x0) * (y1 - y0))
    page_area = max(1, h * w)
    if cross_rendition and target_sat >= 180.0 and (roi_area / page_area) <= 0.08:
        target_container = target_text_mask_in_container(target, region)
        if cv2.countNonZero(target_container) > 0:
            tgt_mask = cv2.bitwise_or(tgt_mask, target_container)
            corridor_diag["color_bubble_target_text_recovery"] = True
            corridor_diag["target_pixels_after_color_recovery"] = int(cv2.countNonZero(tgt_mask))
    else:
        corridor_diag["color_bubble_target_text_recovery"] = False
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
        "color_container_protection": bool(cross_rendition),
        "manual_text_corridor": corridor_diag,
    }
    return src_mask, tgt_mask, diagnostics


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
        source_mask, clear_mask, diag = _white_container_text_masks(aligned, target, region)
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
