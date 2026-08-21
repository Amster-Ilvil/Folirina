from __future__ import annotations

"""Pure mask geometry, coverage scoring, and correspondence helpers.

This module is intentionally renderer-independent.  It owns only deterministic
mask geometry/candidate correspondence operations used by Mask Replace and
related raster-transfer paths.  It must not import the monolithic pipeline or
GUI modules.
"""

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    def linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        costs = np.asarray(cost_matrix, dtype=float)
        pairs = sorted((float(costs[i, j]), i, j) for i in range(costs.shape[0]) for j in range(costs.shape[1]))
        used_rows: set[int] = set(); used_cols: set[int] = set(); selected: list[tuple[int, int]] = []
        for _, i, j in pairs:
            if i not in used_rows and j not in used_cols:
                selected.append((i, j)); used_rows.add(i); used_cols.add(j)
        selected.sort()
        return np.array([i for i, _ in selected], dtype=int), np.array([j for _, j in selected], dtype=int)

from ...config import MaskReplaceConfig
from ...geometry import polygon_bbox, transform_points, transform_to_homography
from ...models import BubbleInstance, RegistrationResult, TextUnit


@dataclass(slots=True)
class BubblePatchMatch:
    source_bubble_id: str
    target_bubble_id: str
    confidence: float
    cost: float
    global_overlap: float
    centroid_distance: float
    shape_score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

def _identity_like(registration: RegistrationResult, cfg: MaskReplaceConfig) -> bool:
    if registration.source_size != registration.target_size:
        return False
    H = transform_to_homography(registration.matrix)
    if abs(float(H[2, 0])) > 1e-7 or abs(float(H[2, 1])) > 1e-7:
        return False
    sx = math.hypot(float(H[0, 0]), float(H[1, 0]))
    sy = math.hypot(float(H[0, 1]), float(H[1, 1]))
    scale_err = max(abs(sx - 1.0), abs(sy - 1.0))
    shift = max(abs(float(H[0, 2])), abs(float(H[1, 2])))
    rot_shear = max(abs(float(H[0, 1])), abs(float(H[1, 0])))
    return (
        scale_err <= cfg.exact_identity_scale_error
        and shift <= cfg.exact_identity_translation_px
        and rot_shear <= cfg.exact_identity_scale_error
    )

def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Fast non-zero bounding box without materialising two full coordinate arrays."""
    arr = np.asarray(mask)
    binary = arr if arr.dtype == np.uint8 else (arr > 0).astype(np.uint8)
    nz = cv2.findNonZero(binary)
    if nz is None:
        return None
    x, y, w, h = cv2.boundingRect(nz)
    return int(x), int(y), int(x + w), int(y + h)

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a > 0, b > 0
    inter = int(np.count_nonzero(aa & bb))
    union = int(np.count_nonzero(aa | bb))
    return inter / union if union else 0.0

def _target_coverage(src: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    ss, tt = src > 0, target > 0
    inter = int(np.count_nonzero(ss & tt))
    t_area = int(np.count_nonzero(tt))
    s_area = int(np.count_nonzero(ss))
    coverage = inter / t_area if t_area else 0.0
    spill = max(0, s_area - inter) / s_area if s_area else 1.0
    return coverage, spill

def _edge_touch_sides(box: tuple[int, int, int, int] | None, shape: tuple[int, int], margin: int = 0) -> tuple[str, ...]:
    """Return image edges touched by a mask bbox.

    For photographed source pages this is an integrity signal rather than a
    geometry error: if a translated bubble runs into the camera frame, pixels
    outside that frame do not exist and must not be silently invented.
    """
    if box is None:
        return ()
    h, w = shape
    x0, y0, x1, y1 = box
    m = max(0, int(margin))
    sides: list[str] = []
    if x0 <= m:
        sides.append("left")
    if y0 <= m:
        sides.append("top")
    if x1 >= w - m:
        sides.append("right")
    if y1 >= h - m:
        sides.append("bottom")
    return tuple(sides)

def _centroid(mask: np.ndarray) -> tuple[float, float]:
    m = cv2.moments((mask > 0).astype(np.uint8))
    if abs(m["m00"]) < 1e-8:
        box = _bbox_from_mask(mask)
        if not box:
            return 0.0, 0.0
        x0, y0, x1, y1 = box
        return (x0 + x1) / 2, (y0 + y1) / 2
    return m["m10"] / m["m00"], m["m01"] / m["m00"]

def _ordered_offsets(max_shift: int) -> list[tuple[int, int]]:
    """Search small nudges in a quality-first order.

    Container-fit placement should try the centroid-aligned position first,
    then expand in Manhattan-distance rings. This keeps v0.8.24 faster than a
    naive full grid while still finding the small nudge needed by slightly
    different bubble interiors.
    """
    if max_shift <= 0:
        return [(0, 0)]
    pts: list[tuple[int, int]] = [(0, 0)]
    seen = {(0, 0)}
    for dist in range(1, max_shift + 1):
        ring: list[tuple[int, int]] = []
        for dx in range(-dist, dist + 1):
            dy = dist - abs(dx)
            for sy in (-1, 1):
                oy = sy * dy
                pt = (dx, oy)
                if pt not in seen:
                    seen.add(pt)
                    ring.append(pt)
        ring.sort(key=lambda p: (abs(p[0]) + abs(p[1]), abs(p[1]), abs(p[0]), p[1], p[0]))
        pts.extend(ring)
    return pts

def _solidify_container_mask(mask: np.ndarray, cfg: MaskReplaceConfig) -> np.ndarray:
    """Reconstruct a solid white-container interior from a detector mask.

    Bright-region detectors can encode Japanese/Chinese glyphs that touch the
    container boundary as black notches. If that raw mask is reused for clearing,
    the original glyph survives by construction; if reused for source clipping,
    Chinese strokes can be cut away. v0.8.26 closes narrow boundary notches and
    fills enclosed holes *inside the original container bbox only*.
    """
    m = (mask > 0).astype(np.uint8) * 255
    if not bool(getattr(cfg, "rigid_container_solidify_enabled", True)):
        return m
    box = _bbox_from_mask(m)
    if box is None:
        return m
    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    base = m[y0:y1, x0:x1].copy()
    base_area = max(1, cv2.countNonZero(base))
    ratio = float(np.clip(getattr(cfg, "rigid_container_solidify_radius_ratio", 0.065), 0.0, 0.25))
    rmin = max(1, int(getattr(cfg, "rigid_container_solidify_min_radius_px", 3)))
    rmax = max(rmin, int(getattr(cfg, "rigid_container_solidify_max_radius_px", 12)))
    radius = int(np.clip(round(min(bw, bh) * ratio), rmin, rmax))
    contours0, _ = cv2.findContours(base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 1.0
    if contours0:
        outer0 = max(contours0, key=cv2.contourArea)
        hull0 = cv2.convexHull(outer0)
        hull_area0 = max(1.0, float(cv2.contourArea(hull0)))
        solidity = float(cv2.contourArea(outer0) / hull_area0)
    spiky_threshold = float(getattr(cfg, "rigid_container_spiky_solidity_threshold", 0.76))
    spiky_shape = bool(getattr(cfg, "rigid_container_spiky_white_enabled", True)) and solidity < spiky_threshold
    if spiky_shape:
        # A jagged burst/star balloon is intentionally concave. Large closing
        # kernels would round off its spikes and turn coloured background inside
        # the convex hull into fake source ink. For these shapes, filling enclosed
        # text holes is enough; preserve the external star geometry byte-for-byte.
        closed = base.copy()
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        closed = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel)

    # Fill enclosed holes without allowing flood-fill to escape through a bbox
    # edge touched by the foreground: pad with a one-pixel black exterior first.
    padded = np.zeros((closed.shape[0] + 2, closed.shape[1] + 2), np.uint8)
    padded[1:-1, 1:-1] = closed
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, holes)[1:-1, 1:-1]

    # Closing can also smooth the *real* black bubble outline. Only admit new
    # pixels that sit safely inside the raw mask's convex envelope. This keeps a
    # thin protected boundary around circular/burst outlines while still filling
    # text-shaped notches that cut inward from a straight/curved edge.
    contours, _ = cv2.findContours(base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if spiky_shape:
        # Only enclosed holes were added above; no convex-envelope clipping is
        # needed, and avoiding it preserves every concave spike/valley.
        candidate = filled
    elif contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        hull_mask = np.zeros_like(base); cv2.fillConvexPoly(hull_mask, hull, 255)
        guard = max(1, int(getattr(cfg, "rigid_container_solidify_boundary_guard_px", 2)))
        hk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard * 2 + 1, guard * 2 + 1))
        hull_inner = cv2.erode(hull_mask, hk)
        added_mask = cv2.bitwise_and(filled, cv2.bitwise_not(base))
        added_mask = cv2.bitwise_and(added_mask, hull_inner)
        candidate = cv2.bitwise_or(base, added_mask)
    else:
        candidate = base

    added = int(cv2.countNonZero(cv2.bitwise_and(candidate, cv2.bitwise_not(base))))
    max_added = float(np.clip(getattr(cfg, "rigid_container_solidify_max_added_ratio", 0.24), 0.02, 0.80))
    if added > base_area * max_added:
        candidate = base

    out = m.copy()
    out[y0:y1, x0:x1] = candidate
    return out

def _bubble_mask(bubble: BubbleInstance, shape: tuple[int, int]) -> np.ndarray:
    if bubble.mask is not None and bubble.mask.shape == shape:
        return (bubble.mask > 0).astype(np.uint8) * 255
    mask = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(bubble.polygon, np.float32)).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 255)
    return mask

def _warp_mask(mask: np.ndarray, matrix: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    H = transform_to_homography(matrix)
    return cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def _shape_score(source_mask: np.ndarray, target_mask: np.ndarray) -> float:
    sc, _ = cv2.findContours((source_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tc, _ = cv2.findContours((target_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not sc or not tc:
        return 0.0
    d = cv2.matchShapes(max(sc, key=cv2.contourArea), max(tc, key=cv2.contourArea), cv2.CONTOURS_MATCH_I1, 0.0)
    return float(math.exp(-2.8 * min(3.0, max(0.0, d))))

def match_bubbles(
    source_bubbles: list[BubbleInstance],
    target_bubbles: list[BubbleInstance],
    registration: RegistrationResult,
    target_shape: tuple[int, int],
    config: MaskReplaceConfig,
) -> list[BubblePatchMatch]:
    kinds = set(config.enabled_kinds)
    src = [b for b in source_bubbles if b.kind in kinds and (b.block_ids or not config.require_source_text)]
    # Target OCR text is not required for direct mask transfer when a dedicated
    # bubble instance segmenter (MangaLens/sidecar) is available.
    dst = [b for b in target_bubbles if b.kind in kinds]
    if not src or not dst:
        return []

    # Paired-diff extraction already established source<->target identity. Keep
    # that deterministic association instead of solving a second Hungarian match
    # that can swap adjacent photographed bubbles after perspective distortion.
    dst_by_id = {b.id: b for b in dst}
    if any(b.meta.get("paired_target_id") for b in src):
        direct: list[BubblePatchMatch] = []
        sh, sw = registration.source_size[1], registration.source_size[0]
        th, tw = target_shape
        diag = max(1.0, math.hypot(tw, th))
        for sb in src:
            tid = sb.meta.get("paired_target_id")
            tb = dst_by_id.get(tid)
            if tb is None:
                continue
            sm = _bubble_mask(sb, (sh, sw)); tm = _bubble_mask(tb, target_shape)
            warped = _warp_mask(sm, registration.matrix, target_shape)
            overlap = _mask_iou(warped, tm)
            scx, scy = _centroid(warped); tcx, tcy = _centroid(tm)
            dist = math.hypot(scx - tcx, scy - tcy) / diag
            shape = _shape_score(warped, tm)
            confidence = float(np.clip(min(sb.confidence, tb.confidence), 0.0, 0.995))
            direct.append(BubblePatchMatch(
                sb.id, tb.id, confidence, 1.0 - confidence, overlap, dist, shape,
                ["paired_diff_identity", f"global_iou={overlap:.3f}", f"centroid={dist:.4f}", f"shape={shape:.3f}"],
            ))
        if direct:
            return direct

    th, tw = target_shape
    diag = max(1.0, math.hypot(tw, th))
    global_masks: dict[str, np.ndarray] = {}
    costs = np.full((len(src), len(dst)), 2.0, np.float64)
    details: dict[tuple[int, int], tuple[float, float, float, list[str]]] = {}

    sh, sw = registration.source_size[1], registration.source_size[0]
    for i, sb in enumerate(src):
        sm = _bubble_mask(sb, (sh, sw))
        warped = _warp_mask(sm, registration.matrix, target_shape)
        global_masks[sb.id] = warped
        scx, scy = _centroid(warped)
        for j, tb in enumerate(dst):
            tm = _bubble_mask(tb, target_shape)
            tcx, tcy = _centroid(tm)
            dist = math.hypot(scx - tcx, scy - tcy) / diag
            overlap = _mask_iou(warped, tm)
            shape = _shape_score(warped, tm)
            kind_penalty = 0.0 if sb.kind == tb.kind else 0.10
            # Geometry dominates. Shape is deliberately secondary because old scans
            # may have clipped/retouched bubble outlines.
            cost = 0.54 * (1.0 - overlap) + 0.27 * min(1.0, dist * 6.0) + 0.13 * (1.0 - shape) + kind_penalty
            reasons = [f"global_iou={overlap:.3f}", f"centroid={dist:.4f}", f"shape={shape:.3f}"]
            costs[i, j] = cost
            details[(i, j)] = (overlap, dist, shape, reasons)

    rows, cols = linear_sum_assignment(costs)
    out: list[BubblePatchMatch] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        cost = float(costs[i, j])
        overlap, dist, shape, reasons = details[(i, j)]
        confidence = float(np.clip(1.0 - cost, 0.0, 1.0))
        out.append(
            BubblePatchMatch(
                src[i].id,
                dst[j].id,
                confidence,
                cost,
                overlap,
                dist,
                shape,
                reasons,
            )
        )
    return out

def _bbox_fit_matrix(source_bbox: tuple[float, float, float, float], target_bbox: tuple[float, float, float, float], H: np.ndarray) -> np.ndarray:
    mapped = transform_points(
        [
            (source_bbox[0], source_bbox[1]),
            (source_bbox[2], source_bbox[1]),
            (source_bbox[2], source_bbox[3]),
            (source_bbox[0], source_bbox[3]),
        ],
        H,
    )
    mx0, my0, mx1, my1 = polygon_bbox(mapped)
    tx0, ty0, tx1, ty1 = target_bbox
    mw, mh = max(1e-6, mx1 - mx0), max(1e-6, my1 - my0)
    tw, th = max(1e-6, tx1 - tx0), max(1e-6, ty1 - ty0)
    sx, sy = tw / mw, th / mh
    F = np.array([[sx, 0.0, tx0 - sx * mx0], [0.0, sy, ty0 - sy * my0], [0.0, 0.0, 1.0]], np.float64)
    return F @ transform_to_homography(H)

def _bbox_uniform_fit_matrix(source_bbox: tuple[float, float, float, float], target_bbox: tuple[float, float, float, float], H: np.ndarray) -> np.ndarray:
    """Local photo fit that cannot squeeze/stretch glyphs independently by axis."""
    mapped = transform_points(
        [
            (source_bbox[0], source_bbox[1]),
            (source_bbox[2], source_bbox[1]),
            (source_bbox[2], source_bbox[3]),
            (source_bbox[0], source_bbox[3]),
        ],
        H,
    )
    mx0, my0, mx1, my1 = polygon_bbox(mapped)
    tx0, ty0, tx1, ty1 = target_bbox
    mw, mh = max(1e-6, mx1 - mx0), max(1e-6, my1 - my0)
    tw, th = max(1e-6, tx1 - tx0), max(1e-6, ty1 - ty0)
    sx, sy = tw / mw, th / mh
    # Geometric mean is symmetric and preserves shape. Translation aligns centers.
    scale = float(math.sqrt(max(1e-8, sx * sy)))
    mcx, mcy = (mx0 + mx1) * 0.5, (my0 + my1) * 0.5
    tcx, tcy = (tx0 + tx1) * 0.5, (ty0 + ty1) * 0.5
    F = np.array([[scale, 0.0, tcx - scale * mcx], [0.0, scale, tcy - scale * mcy], [0.0, 0.0, 1.0]], np.float64)
    return F @ transform_to_homography(H)

def _local_translation_ecc(warped_mask: np.ndarray, target_mask: np.ndarray, cfg: MaskReplaceConfig) -> tuple[float, float, float]:
    box = _bbox_from_mask(target_mask)
    if not box:
        return 0.0, 0.0, _mask_iou(warped_mask, target_mask)
    x0, y0, x1, y1 = box
    pad = max(8, int(0.08 * max(x1 - x0, y1 - y0)))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(target_mask.shape[1], x1 + pad), min(target_mask.shape[0], y1 + pad)
    template = (target_mask[y0:y1, x0:x1] > 0).astype(np.float32)
    inp = (warped_mask[y0:y1, x0:x1] > 0).astype(np.float32)
    if template.size == 0 or inp.size == 0 or template.sum() == 0 or inp.sum() == 0:
        return 0.0, 0.0, _mask_iou(warped_mask, target_mask)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.ecc_iterations, cfg.ecc_epsilon)
    try:
        cv2.findTransformECC(template, inp, warp, cv2.MOTION_TRANSLATION, criteria)
    except cv2.error:
        return 0.0, 0.0, _mask_iou(warped_mask, target_mask)

    # OpenCV ECC convention is easy to misuse. Test both signs and retain only an
    # actual IoU improvement. This also prevents local refinement from drifting.
    base = _mask_iou(warped_mask, target_mask)
    candidates = [(0.0, 0.0, base)]
    for sign in (1.0, -1.0):
        dx, dy = float(warp[0, 2] * sign), float(warp[1, 2] * sign)
        max_shift = cfg.max_local_translation_ratio * max(1, max(x1 - x0, y1 - y0))
        if abs(dx) > max_shift or abs(dy) > max_shift:
            continue
        M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
        moved = cv2.warpAffine(warped_mask, M, (warped_mask.shape[1], warped_mask.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
        candidates.append((dx, dy, _mask_iou(moved, target_mask)))
    return max(candidates, key=lambda x: x[2])

def _ocr_guided_region_gate(
    source_unit: TextUnit,
    target_unit: TextUnit,
    registration: RegistrationResult,
    shape: tuple[int, int],
    cfg: MaskReplaceConfig,
) -> np.ndarray | None:
    """Build a target-space *gate* around matched OCR geometry.

    OCR is used only to say "these two text regions correspond".  The gate is
    never pasted or filled.  The compositor below still selects/erases concrete
    Japanese glyph components and copies concrete registered source raster ink.
    """
    h, w = shape
    try:
        projected = transform_points(source_unit.polygon, registration.matrix)
    except Exception:
        return None
    pts = np.asarray(list(projected) + list(target_unit.polygon), dtype=np.float32)
    if pts.shape[0] < 3 or not np.all(np.isfinite(pts)):
        return None
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, w - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, h - 1))
    hull = cv2.convexHull(np.round(pts).astype(np.int32))
    gate = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(gate, hull, 255)
    box = _bbox_from_mask(gate)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    pad = max(4, int(round(max(x1 - x0, y1 - y0) * float(getattr(cfg, "ocr_guided_region_pad_ratio", 0.14)))))
    pad = min(pad, max(8, int(round(min(h, w) * 0.045))))
    if pad > 0:
        gate = cv2.dilate(gate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)))
    ratio = cv2.countNonZero(gate) / max(1, h * w)
    if ratio > float(getattr(cfg, "ocr_guided_max_region_ratio", 0.08)):
        return None
    return gate

__all__ = [
    "BubblePatchMatch",
    "_identity_like",
    "_bbox_from_mask",
    "_mask_iou",
    "_target_coverage",
    "_edge_touch_sides",
    "_centroid",
    "_ordered_offsets",
    "_solidify_container_mask",
    "_bubble_mask",
    "_warp_mask",
    "_shape_score",
    "match_bubbles",
    "_bbox_fit_matrix",
    "_bbox_uniform_fit_matrix",
    "_local_translation_ecc",
    "_ocr_guided_region_gate",
]
