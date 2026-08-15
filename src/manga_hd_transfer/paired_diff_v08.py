from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import cv2
import numpy as np

from .config import MaskReplaceConfig
from .geometry import mask_to_largest_polygon, transform_to_homography
from .models import BubbleInstance, RegistrationResult


@dataclass(slots=True)
class DiffBubbleRecord:
    source_id: str
    target_id: str
    change_density: float
    mask_iou: float
    confidence: float
    bbox_target: tuple[int, int, int, int]
    region_kind: str = "bubble"
    changed_pixels: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class PairedDiffResult:
    source_bubbles: list[BubbleInstance]
    target_bubbles: list[BubbleInstance]
    change_mask: np.ndarray
    records: list[DiffBubbleRecord]
    threshold: float
    noise_floor: float
    # v0.8: global SIFT/RANSAC registration is refined only for paired transfer.
    # The target-sized source is deliberately returned so compositing can be
    # target-driven and no longer fail on tiny photographed-page mask mismatch.
    aligned_source: np.ndarray | None = None
    alignment_diagnostics: dict = field(default_factory=dict)


def _warp_source(source: np.ndarray, registration: RegistrationResult, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    H = transform_to_homography(registration.matrix)
    return cv2.warpPerspective(
        source,
        H,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _transform_xy(x: float, y: float, H: np.ndarray) -> tuple[float, float]:
    p = H @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-9:
        return x, y
    return float(p[0] / p[2]), float(p[1] / p[2])


def _nearest_seed(binary: np.ndarray, x: float, y: float, radius: int) -> tuple[int, int] | None:
    h, w = binary.shape
    ix = int(np.clip(round(x), 0, w - 1))
    iy = int(np.clip(round(y), 0, h - 1))
    if binary[iy, ix] > 0:
        return ix, iy
    for r in range(1, max(1, radius) + 1):
        x0, x1 = max(0, ix - r), min(w - 1, ix + r)
        y0, y1 = max(0, iy - r), min(h - 1, iy + r)
        candidates: list[tuple[int, int]] = []
        for xx in range(x0, x1 + 1):
            if binary[y0, xx]:
                candidates.append((xx, y0))
            if y1 != y0 and binary[y1, xx]:
                candidates.append((xx, y1))
        for yy in range(y0 + 1, y1):
            if binary[yy, x0]:
                candidates.append((x0, yy))
            if x1 != x0 and binary[yy, x1]:
                candidates.append((x1, yy))
        if candidates:
            return min(candidates, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
    return None


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a > 0, b > 0
    inter = int(np.count_nonzero(aa & bb))
    union = int(np.count_nonzero(aa | bb))
    return inter / union if union else 0.0


def _classify_kind(mask: np.ndarray) -> str:
    box = _bbox(mask)
    if box is None:
        return "speech"
    x0, y0, x1, y1 = box
    area = max(1, cv2.countNonZero(mask))
    rect_area = max(1, (x1 - x0) * (y1 - y0))
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "speech"
    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 0.02 * max(1, (x1 - x0) + (y1 - y0)), True)
    rect_fill = area / rect_area
    return "narration" if len(approx) <= 6 and rect_fill > 0.78 else "speech"


def _safe_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        return mask.copy()
    x0, y0, x1, y1 = box
    margin = max(3, int(0.025 * min(x1 - x0, y1 - y0)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
    safe = cv2.erode(mask, k)
    return safe if cv2.countNonZero(safe) else mask.copy()


def _warp_mask(mask: np.ndarray, H: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _flatten_gray(image: np.ndarray, sigma: float = 15.0, scale: float = 245.0) -> np.ndarray:
    """Suppress smooth camera illumination while retaining ink/outline contrast."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    bg = cv2.GaussianBlur(gray, (0, 0), max(2.0, float(sigma)))
    bg = np.maximum(bg, 1)
    return cv2.divide(gray, bg, scale=float(scale))


def _ink_map(image: np.ndarray, cfg: MaskReplaceConfig) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # CLAHE + adaptive threshold is intentionally detector-like rather than a raw
    # pixel difference. It tolerates scanner tone, JPEG phase and smooth glare.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    block = 31
    ink = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 10)
    # Keep very dark strokes that adaptive threshold may split under a bright halo.
    dark = (eq <= 150).astype(np.uint8) * 255
    ink = cv2.bitwise_or(ink, dark)
    return ink


def _is_near_identity(registration: RegistrationResult) -> bool:
    if registration.source_size != registration.target_size:
        return False
    H = transform_to_homography(registration.matrix)
    return bool(
        np.max(np.abs(H - np.eye(3, dtype=np.float64))) < 0.01
        and abs(float(H[0, 2])) < 1.0
        and abs(float(H[1, 2])) < 1.0
    )


def _local_align_source(
    source_warped: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray, dict]:
    """Refine global page registration with conservative dense DIS flow.

    Flow is estimated only on heavily blurred grayscale pages, so lettering itself
    cannot drag glyphs into the Japanese text. The output remains in target space.
    """
    if not cfg.paired_diff_local_flow_enabled or _is_near_identity(registration):
        return source_warped, {"method": "global-only", "flow_used": False}

    h, w = target.shape[:2]
    max_side = max(h, w)
    scale = min(1.0, max(0.20, cfg.paired_diff_flow_max_side / max(1.0, float(max_side))))
    sw, sh = max(64, int(round(w * scale))), max(64, int(round(h * scale)))

    tg = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    sg = cv2.cvtColor(source_warped, cv2.COLOR_BGR2GRAY)
    sigma = max(1.0, float(cfg.paired_diff_flow_blur_sigma))
    tg = cv2.GaussianBlur(tg, (0, 0), sigma)
    sg = cv2.GaussianBlur(sg, (0, 0), sigma)
    a = cv2.resize(tg, (sw, sh), interpolation=cv2.INTER_AREA)
    b = cv2.resize(sg, (sw, sh), interpolation=cv2.INTER_AREA)

    try:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setFinestScale(2)
        flow = dis.calc(a, b, None)
    except (cv2.error, AttributeError):
        return source_warped, {"method": "global-only", "flow_used": False, "flow_error": "DIS unavailable"}

    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / max(scale, 1e-6)
    # A small spatial smooth prevents local text strokes from becoming a flow field.
    flow[..., 0] = cv2.GaussianBlur(flow[..., 0], (0, 0), 2.0)
    flow[..., 1] = cv2.GaussianBlur(flow[..., 1], (0, 0), 2.0)
    mag = np.linalg.norm(flow, axis=2)
    limit = max(1.0, float(cfg.paired_diff_flow_max_shift_px))
    over = mag > limit
    if np.any(over):
        factor = limit / np.maximum(mag[over], 1e-6)
        flow[over, 0] *= factor
        flow[over, 1] *= factor
        mag = np.linalg.norm(flow, axis=2)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    aligned = cv2.remap(
        source_warped,
        xx + flow[..., 0].astype(np.float32),
        yy + flow[..., 1].astype(np.float32),
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return aligned, {
        "method": "global+dis-dense",
        "flow_used": True,
        "flow_scale": float(scale),
        "flow_median_px": float(np.median(mag)),
        "flow_p95_px": float(np.percentile(mag, 95.0)),
        "flow_max_px": float(np.max(mag)),
    }


def _structural_change_map(
    source_aligned: np.ndarray,
    target: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Build a structural mismatch map without assuming a white text background.

    v0.8.20 discarded every changed stroke whose target neighbourhood was darker
    than ``paired_diff_local_mean_threshold``.  That was safe for white balloons,
    but it also made yellow burst balloons, coloured captions and text printed on
    artwork invisible.  v0.8.21 keeps the old bright-paper gate *and* adds a
    conservative two-sided ink-density gate: both editions must contain a compact
    amount of line/text ink in the same neighbourhood and that neighbourhood must
    contain enough structural disagreement.  Candidate-level component filtering
    below still decides whether the island is text-like.
    """
    si = _ink_map(source_aligned, cfg) > 0
    ti = _ink_map(target, cfg) > 0
    tol = max(0, int(cfg.paired_diff_ink_tolerance_px))
    if tol:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))
        sd = cv2.dilate(si.astype(np.uint8), k) > 0
        td = cv2.dilate(ti.astype(np.uint8), k) > 0
        mismatch = (si & ~td) | (ti & ~sd)
    else:
        mismatch = si ^ ti

    tg = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    local_mean = cv2.GaussianBlur(tg, (0, 0), 4.0)
    bright_gate = local_mean >= int(cfg.paired_diff_local_mean_threshold)

    if bool(getattr(cfg, "paired_diff_complex_text_enabled", True)):
        win = max(7, int(getattr(cfg, "paired_diff_complex_local_window", 31)) | 1)
        sf = si.astype(np.float32); tf = ti.astype(np.float32); mf = mismatch.astype(np.float32)
        sdens = cv2.boxFilter(sf, cv2.CV_32F, (win, win), normalize=True)
        tdens = cv2.boxFilter(tf, cv2.CV_32F, (win, win), normalize=True)
        mdens = cv2.boxFilter(mf, cv2.CV_32F, (win, win), normalize=True)
        lo = float(getattr(cfg, "paired_diff_complex_min_ink_density", 0.014))
        hi = float(getattr(cfg, "paired_diff_complex_max_ink_density", 0.42))
        mlo = float(getattr(cfg, "paired_diff_complex_min_change_density", 0.014))
        complex_gate = (sdens >= lo) & (sdens <= hi) & (tdens >= lo) & (tdens <= hi) & (mdens >= mlo)
        mismatch &= (bright_gate | complex_gate)
    else:
        mismatch &= bright_gate

    raw = mismatch.astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    regions = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    r = max(0, int(cfg.paired_diff_seed_dilate_px))
    if r:
        regions = cv2.dilate(regions, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1)))

    # Retain comparable diagnostics with v0.7, but calculate the photometric noise
    # on bright paper instead of allowing page art to dominate the 95th percentile.
    sg = cv2.cvtColor(source_aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tgf = tg.astype(np.float32)
    bright = (sg >= 180) & (tgf >= 180)
    diff = np.abs(sg - tgf)
    vals = diff[bright]
    noise_floor = float(np.percentile(vals, 95.0)) if vals.size else float(np.percentile(diff, 95.0))
    threshold = float(max(cfg.paired_diff_pixel_threshold, noise_floor + cfg.paired_diff_noise_margin))
    return raw, threshold, noise_floor, regions


def _compact_ink_component_metrics(ink_roi: np.ndarray) -> tuple[int, int]:
    """Count compact glyph-like components and their pixels in a binary ROI.

    This is deliberately language agnostic.  It excludes long panel/balloon lines
    while allowing multi-piece CJK glyphs, punctuation and bold burst lettering.
    """
    if ink_roi.size == 0:
        return 0, 0
    binary = (ink_roi > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    h, w = binary.shape
    roi_area = max(1, h * w)
    count = 0; pixels = 0
    for i in range(1, n):
        _x, _y, cw, ch, area = [int(v) for v in stats[i]]
        if area < 2 or area > max(36, int(0.10 * roi_area)):
            continue
        if cw > 0.72 * w or ch > 0.72 * h:
            continue
        aspect = max(cw / max(1.0, ch), ch / max(1.0, cw))
        if aspect > 9.0:
            continue
        count += 1; pixels += area
    return count, pixels

def _legacy_change_map(source_warped: np.ndarray, target: np.ndarray, cfg: MaskReplaceConfig) -> tuple[np.ndarray, float, float, np.ndarray]:
    sg = cv2.cvtColor(source_warped, cv2.COLOR_BGR2GRAY)
    tg = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    sb = cv2.GaussianBlur(sg, (3, 3), 0.55).astype(np.float32)
    tb = cv2.GaussianBlur(tg, (3, 3), 0.55).astype(np.float32)
    diff = np.abs(sb - tb)
    noise_floor = float(np.percentile(diff, 95.0))
    threshold = float(max(cfg.paired_diff_pixel_threshold, noise_floor + cfg.paired_diff_noise_margin))
    raw = (diff >= threshold).astype(np.uint8) * 255
    if cfg.paired_diff_close_px > 1:
        k = max(3, cfg.paired_diff_close_px | 1)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    if cfg.paired_diff_dilate_px > 0:
        r = max(1, cfg.paired_diff_dilate_px)
        regions = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1)))
    else:
        regions = raw.copy()
    return raw, threshold, noise_floor, regions


def _change_map(source_warped: np.ndarray, target: np.ndarray, cfg: MaskReplaceConfig) -> tuple[np.ndarray, float, float, np.ndarray]:
    if cfg.paired_diff_structural:
        return _structural_change_map(source_warped, target, cfg)
    return _legacy_change_map(source_warped, target, cfg)


def _barrier_component_from_seed(
    image: np.ndarray,
    x: float,
    y: float,
    cfg: MaskReplaceConfig,
    dark_threshold: int | None = None,
    dilate_px: int | None = None,
    *,
    cache: dict | None = None,
) -> np.ndarray | None:
    """Flood an enclosed light region using dark line art as the barrier.

    v1.3.7 performance note: one page may probe hundreds of changed-text seeds.
    The old implementation recomputed the same illumination flattening and the
    same connected-component label map for every seed.  ``cache`` is deliberately
    request-local and stores only deterministic intermediates for the current
    SOURCE/TARGET arrays, so reuse is pixel-identical and cannot leak across pages.
    """
    thr = int(cfg.paired_diff_barrier_dark_threshold if dark_threshold is None else dark_threshold)
    dp = int(cfg.paired_diff_barrier_dilate_px if dilate_px is None else dilate_px)
    cache = cache if cache is not None else {}
    image_key = (id(image), image.shape, image.strides)
    norm_key = ("norm", image_key)
    norm = cache.get(norm_key)
    if norm is None:
        norm = _flatten_gray(image, sigma=15.0, scale=245.0)
        cache[norm_key] = norm
    label_key = ("barrier_labels", image_key, int(thr), int(dp))
    row = cache.get(label_key)
    if row is None:
        barrier = (norm <= thr).astype(np.uint8) * 255
        barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if dp > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dp * 2 + 1, dp * 2 + 1))
            barrier = cv2.dilate(barrier, k)
        free = (barrier == 0).astype(np.uint8) * 255
        n, labels, _, _ = cv2.connectedComponentsWithStats((free > 0).astype(np.uint8), 8)
        row = (free, int(n), labels)
        cache[label_key] = row
    free, n, labels = row
    seed = _nearest_seed(free, x, y, cfg.paired_diff_search_radius)
    if seed is None:
        return None
    label = int(labels[seed[1], seed[0]])
    if label <= 0 or label >= n:
        return None
    raw = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    outer = max(contours, key=cv2.contourArea)
    component = np.zeros_like(raw)
    cv2.drawContours(component, [outer], -1, 255, cv2.FILLED)
    return component


def _white_component_from_seed(image: np.ndarray, x: float, y: float, threshold: int, search_radius: int) -> np.ndarray | None:
    # Legacy fallback is useful for synthetic pages and unusual border styles.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    white = (gray >= threshold).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    seed = _nearest_seed(white, x, y, search_radius)
    if seed is None:
        return None
    n, labels, _, _ = cv2.connectedComponentsWithStats((white > 0).astype(np.uint8), 8)
    label = int(labels[seed[1], seed[0]])
    if label <= 0 or label >= n:
        return None
    raw = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    component = np.zeros_like(raw)
    cv2.drawContours(component, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return component


def _target_component_score(mask: np.ndarray, target: np.ndarray, raw_change: np.ndarray, cfg: MaskReplaceConfig) -> tuple[bool, dict]:
    box = _bbox(mask)
    if box is None:
        return False, {}
    x0, y0, x1, y1 = box
    area = cv2.countNonZero(mask)
    page_area = mask.shape[0] * mask.shape[1]
    rect_area = max(1, (x1 - x0) * (y1 - y0))
    rect_fill = area / rect_area
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    inside = mask > 0
    white_ratio = float(np.mean(gray[inside] >= 230)) if np.any(inside) else 0.0
    changed = int(np.count_nonzero((raw_change > 0) & inside))
    density = changed / max(1, area)
    ok = (
        area >= max(cfg.paired_diff_min_component_area, cfg.paired_diff_min_enclosed_area)
        and area / max(1, page_area) <= cfg.paired_diff_max_region_ratio
        and rect_fill >= cfg.paired_diff_min_rect_fill
        and white_ratio >= cfg.paired_diff_min_white_ratio
        and changed >= cfg.paired_diff_min_changed_pixels
        and density >= cfg.paired_diff_min_enclosed_change_density
    )
    return ok, {
        "area": int(area), "rect_fill": float(rect_fill), "white_ratio": white_ratio,
        "changed_pixels": changed, "change_density": float(density),
    }


def _source_component_for_target(
    source: np.ndarray,
    target_mask: np.ndarray,
    target_centroid: tuple[float, float],
    H: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    barrier_cache: dict | None = None,
) -> tuple[np.ndarray, float]:
    Hinv = np.linalg.inv(H)
    sx, sy = _transform_xy(target_centroid[0], target_centroid[1], Hinv)
    best: tuple[np.ndarray | None, float] = (None, -1.0)
    # Camera exposure and old print screening can move the best dark-line threshold.
    base = int(cfg.paired_diff_barrier_dark_threshold)
    for thr in (base - 20, base - 10, base, base + 10, base + 20):
        for dp in sorted({1, max(1, int(cfg.paired_diff_barrier_dilate_px)), 2}):
            sm = _barrier_component_from_seed(source, sx, sy, cfg, int(np.clip(thr, 130, 225)), dp, cache=barrier_cache)
            if sm is None or cv2.countNonZero(sm) < cfg.paired_diff_min_component_area:
                continue
            warped = _warp_mask(sm, H, target_mask.shape)
            score = _iou(warped, target_mask)
            if score > best[1]:
                best = (sm, score)
    if best[0] is None:
        sm = _white_component_from_seed(source, sx, sy, cfg.paired_diff_white_threshold, cfg.paired_diff_search_radius)
        if sm is not None:
            return sm, _iou(_warp_mask(sm, H, target_mask.shape), target_mask)
        # Last-resort geometric proxy; this mask is debug/QA only in target-driven mode.
        sm = _warp_mask(target_mask, Hinv, source.shape[:2])
        return sm, _iou(_warp_mask(sm, H, target_mask.shape), target_mask)
    return best[0], float(best[1])


def _find_enclosed_candidates(
    target: np.ndarray,
    raw_change: np.ndarray,
    regions: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    barrier_cache: dict | None = None,
) -> list[tuple[np.ndarray, dict, tuple[int, int, int, int]]]:
    n, labels, stats, cents = cv2.connectedComponentsWithStats((regions > 0).astype(np.uint8), 8)
    found: list[tuple[np.ndarray, dict, tuple[int, int, int, int]]] = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < cfg.paired_diff_seed_min_area:
            continue
        cx, cy = map(float, cents[label])
        tm = _barrier_component_from_seed(target, cx, cy, cfg, cache=barrier_cache)
        if tm is None:
            tm = _white_component_from_seed(target, cx, cy, cfg.paired_diff_white_threshold, cfg.paired_diff_search_radius)
        if tm is None:
            continue
        ok, metrics = _target_component_score(tm, target, raw_change, cfg)
        if not ok:
            continue
        box = _bbox(tm)
        if box is None:
            continue
        found.append((tm, metrics, box))

    # Several changed glyph clusters point to the same balloon. Keep one mask.
    found.sort(key=lambda r: (r[1]["change_density"], r[1]["changed_pixels"]), reverse=True)
    kept: list[tuple[np.ndarray, dict, tuple[int, int, int, int]]] = []
    for row in found:
        if any(_iou(row[0], k[0]) > 0.82 for k in kept):
            continue
        kept.append(row)
    return kept


def _free_text_candidates(
    source_aligned: np.ndarray,
    target: np.ndarray,
    raw_change: np.ndarray,
    bubble_masks: list[np.ndarray],
    cfg: MaskReplaceConfig,
) -> list[tuple[np.ndarray, dict, tuple[int, int, int, int]]]:
    """Find open/free/complex text-change islands outside closed balloons.

    Bright paper continues to use the v0.8 rules.  Coloured or textured regions
    are admitted only when *both* aligned editions contain a text-like amount of
    compact ink and enough changed strokes.  The returned mask is a region gate,
    not a rectangular replacement mask; the compositor later clears/draws only
    actual target/source glyph components.
    """
    if not cfg.paired_diff_free_text_enabled:
        return []
    h, w = raw_change.shape
    union = np.zeros((h, w), np.uint8)
    for mask in bubble_masks:
        union = np.maximum(union, mask)
    exclude = union.copy()
    r = max(0, int(cfg.paired_diff_free_exclude_bubble_px))
    if r and cv2.countNonZero(exclude):
        exclude = cv2.dilate(exclude, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1)))

    free = (raw_change > 0).astype(np.float32)
    free[exclude > 0] = 0.0
    win = max(5, int(cfg.paired_diff_free_density_window) | 1)
    density = cv2.boxFilter(free, cv2.CV_32F, (win, win), normalize=True)
    seed = (density >= float(cfg.paired_diff_free_density_threshold)).astype(np.uint8) * 255
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((seed > 0).astype(np.uint8), 8)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    src_ink = _ink_map(source_aligned, cfg)
    tgt_ink = _ink_map(target, cfg)
    out: list[tuple[np.ndarray, dict, tuple[int, int, int, int]]] = []
    page_area = h * w
    complex_enabled = bool(getattr(cfg, "paired_diff_complex_text_enabled", True))
    complex_max_ratio = float(getattr(cfg, "paired_diff_complex_max_region_ratio", 0.065))
    min_changed_complex = int(getattr(cfg, "paired_diff_complex_min_changed_pixels", 70))
    min_compact = int(getattr(cfg, "paired_diff_complex_min_compact_components", 3))
    lo_ink = float(getattr(cfg, "paired_diff_complex_min_ink_density", 0.014))
    hi_ink = float(getattr(cfg, "paired_diff_complex_max_ink_density", 0.42))

    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < cfg.paired_diff_free_min_component_area:
            continue
        if x < 6 or y < 6 or x + bw > w - 6 or y + bh > h - 6:
            continue
        ratio = area / max(1, page_area)
        if ratio > max(0.035, complex_max_ratio):
            continue
        changed = int(np.count_nonzero(free[y:y + bh, x:x + bw] > 0))
        if changed < min(cfg.paired_diff_free_min_changed_pixels, min_changed_complex):
            continue
        local_mean = float(np.mean(gray[y:y + bh, x:x + bw]))

        sroi = src_ink[y:y + bh, x:x + bw]
        troi = tgt_ink[y:y + bh, x:x + bw]
        sden = float(np.count_nonzero(sroi) / max(1, sroi.size))
        tden = float(np.count_nonzero(troi) / max(1, troi.size))
        scomp, spix = _compact_ink_component_metrics(sroi)
        tcomp, tpix = _compact_ink_component_metrics(troi)
        change_density = changed / max(1, area)

        bright_ok = (
            changed >= cfg.paired_diff_free_min_changed_pixels
            and local_mean >= cfg.paired_diff_free_min_local_mean
            and ratio <= 0.035
        )
        complex_ok = (
            complex_enabled
            and changed >= min_changed_complex
            and change_density >= float(getattr(cfg, "paired_diff_complex_min_change_density", 0.014))
            and lo_ink <= sden <= hi_ink
            and lo_ink <= tden <= hi_ink
            and min(scomp, tcomp) >= min_compact
            and min(spix, tpix) >= 10
            and ratio <= complex_max_ratio
        )
        if not bright_ok and not complex_ok:
            continue

        comp = (labels == i).astype(np.uint8) * 255
        comp[exclude > 0] = 0
        d = max(0, int(cfg.paired_diff_free_mask_dilate_px))
        if d:
            comp = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1)))
        comp[union > 0] = 0
        box = _bbox(comp)
        if box is None or cv2.countNonZero(comp) < 20:
            continue
        mask_changed = int(np.count_nonzero((raw_change > 0) & (comp > 0)))
        vals = gray[comp > 0]
        region_kind = "complex_text" if (complex_ok and not bright_ok) else "free_text"
        out.append((comp, {
            "area": int(cv2.countNonZero(comp)),
            "rect_fill": float(cv2.countNonZero(comp) / max(1, (box[2] - box[0]) * (box[3] - box[1]))),
            "white_ratio": float(np.mean(vals >= 230)) if vals.size else 0.0,
            "changed_pixels": mask_changed,
            "change_density": mask_changed / max(1, cv2.countNonZero(comp)),
            "local_mean": local_mean,
            "source_ink_density": sden,
            "target_ink_density": tden,
            "source_compact_components": scomp,
            "target_compact_components": tcomp,
            "region_kind": region_kind,
            "complex_text": bool(complex_ok),
        }, box))

    # Density islands can overlap; keep the strongest distinct island.
    out.sort(key=lambda r: (r[1]["changed_pixels"], r[1].get("complex_text", False)), reverse=True)
    kept: list[tuple[np.ndarray, dict, tuple[int, int, int, int]]] = []
    for row in out:
        if any(_iou(row[0], k[0]) > 0.35 for k in kept):
            continue
        kept.append(row)
    return kept

def extract_paired_diff_bubbles(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    cfg: MaskReplaceConfig | None = None,
) -> PairedDiffResult:
    """Locate translated regions in an old edition and map them to a clean target.

    v0.8 deliberately separates three problems:
      1. global page registration (existing SIFT/RANSAC path),
      2. dense low-frequency local alignment for photographed page curl,
      3. structural text-change detection + target-side region masks.

    Speech/narration regions are found by enclosed dark-line barriers; changed SFX
    or free text are retained as independent masks instead of being discarded.
    """
    cfg = cfg or MaskReplaceConfig()
    target_shape = target.shape[:2]
    source_global = _warp_source(source, registration, target_shape)
    source_aligned, align_diag = _local_align_source(source_global, target, registration, cfg)
    raw_change, threshold, noise_floor, regions = _change_map(source_aligned, target, cfg)

    H = transform_to_homography(registration.matrix)
    target_barrier_cache: dict = {}
    source_barrier_cache: dict = {}
    enclosed = _find_enclosed_candidates(target, raw_change, regions, cfg, barrier_cache=target_barrier_cache)
    free_text = _free_text_candidates(source_aligned, target, raw_change, [r[0] for r in enclosed], cfg)

    rows: list[tuple[np.ndarray, np.ndarray, dict, tuple[int, int, int, int], str, float]] = []
    for tm, metrics, box in enclosed:
        moments = cv2.moments((tm > 0).astype(np.uint8))
        if moments["m00"]:
            cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        else:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        sm, mask_iou = _source_component_for_target(source, tm, (cx, cy), H, cfg, barrier_cache=source_barrier_cache)
        rows.append((sm, tm, metrics, box, "bubble", mask_iou))

    Hinv = np.linalg.inv(H)
    for tm, metrics, box in free_text:
        # Open/complex text has no enclosing source component. Its source mask is
        # only a bookkeeping proxy; transfer uses aligned source glyph components.
        sm = _warp_mask(tm, Hinv, source.shape[:2])
        mask_iou = _iou(_warp_mask(sm, H, target_shape), tm)
        rows.append((sm, tm, metrics, box, str(metrics.get("region_kind") or "free_text"), mask_iou))

    rows.sort(key=lambda r: (r[3][1], r[3][0]))
    source_bubbles: list[BubbleInstance] = []
    target_bubbles: list[BubbleInstance] = []
    records: list[DiffBubbleRecord] = []
    for idx, (sm, tm, metrics, box, region_kind, mask_iou) in enumerate(rows):
        sp = mask_to_largest_polygon(sm)
        tp = mask_to_largest_polygon(tm)
        if len(sp) < 3 or len(tp) < 3:
            continue
        density = float(metrics.get("change_density", 0.0))
        # In target-driven mode source-mask IoU is diagnostic, not a rejection gate.
        iou_term = float(np.clip(mask_iou, 0.0, 1.0))
        conf = float(np.clip(
            0.42 * registration.confidence
            + 0.22 * iou_term
            + 0.22 * min(1.0, density / 0.10)
            + 0.14 * (1.0 if region_kind == "bubble" else 0.90),
            0.0, 0.995,
        ))
        kind = _classify_kind(tm) if region_kind == "bubble" else "speech"
        sid, tid = f"diff-src-{idx:03d}", f"diff-dst-{idx:03d}"
        common_meta = {
            "backend": "paired_diff_v08",
            "mask_is_interior": region_kind == "bubble",
            "paired_region_kind": region_kind,
            "change_density": density,
            "changed_pixels": int(metrics.get("changed_pixels", 0)),
            "paired_mask_iou": float(mask_iou),
            "diff_threshold": threshold,
            "noise_floor": noise_floor,
            "target_white_ratio": float(metrics.get("white_ratio", 0.0)),
            "target_rect_fill": float(metrics.get("rect_fill", 0.0)),
            "source_ink_density": float(metrics.get("source_ink_density", 0.0)),
            "target_ink_density": float(metrics.get("target_ink_density", 0.0)),
            "source_compact_components": int(metrics.get("source_compact_components", 0)),
            "target_compact_components": int(metrics.get("target_compact_components", 0)),
        }
        source_bubbles.append(BubbleInstance(sid, sp, conf, kind, [f"paired-diff-{idx:03d}"], sm, _safe_mask(sm), dict(common_meta)))
        target_bubbles.append(BubbleInstance(tid, tp, conf, kind, [], tm, _safe_mask(tm), dict(common_meta)))
        records.append(DiffBubbleRecord(
            sid, tid, density, float(mask_iou), conf, box,
            region_kind=region_kind,
            changed_pixels=int(metrics.get("changed_pixels", 0)),
        ))

    align_diag = dict(align_diag)
    align_diag.update({
        "detector": "structural-ink+barrier" if cfg.paired_diff_structural else "photometric-legacy",
        "bubble_regions": sum(1 for r in records if r.region_kind == "bubble"),
        "free_text_regions": sum(1 for r in records if r.region_kind == "free_text"),
        "complex_text_regions": sum(1 for r in records if r.region_kind == "complex_text"),
        "total_regions": len(records),
        "changed_pixels": int(cv2.countNonZero(raw_change)),
        "barrier_cache_target_entries": int(len(target_barrier_cache)),
        "barrier_cache_source_entries": int(len(source_barrier_cache)),
    })
    return PairedDiffResult(
        source_bubbles, target_bubbles, raw_change, records, threshold, noise_floor,
        aligned_source=source_aligned,
        alignment_diagnostics=align_diag,
    )
