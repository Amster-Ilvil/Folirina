from __future__ import annotations

import math
import copy
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

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

from .text_only_transfer import target_text_mask_in_container, cleanup_target_residual_specks, clear_uniform_white_container_interior, white_container_paper_mask, white_container_write_envelope, target_container_border_mask, remove_container_boundary_line_components
from .config import MaskReplaceConfig
from .geometry import polygon_bbox, transform_points, transform_to_homography
from .io_utils import read_image, write_image
from .models import BubbleInstance, RegistrationResult, TextUnit, UnitMatch


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


@dataclass(slots=True)
class MaskTransferRecord:
    source_bubble_id: str
    target_bubble_id: str
    confidence: float
    applied: bool
    reason: str
    sr_backend: str = "off"
    sr_scale: float = 1.0
    mask_iou: float = 0.0
    target_coverage: float = 0.0
    spill_ratio: float = 0.0
    local_dx: float = 0.0
    local_dy: float = 0.0
    sharpness: float = 0.0
    target_sharpness: float = 0.0
    relative_sharpness: float = 0.0
    clarity_mode: str = "pixels"
    geometry_mode: str = "standard"
    ink_ratio: float = 0.0
    source_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    target_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    source_edge_clipped: bool = False
    source_edge_sides: str = ""
    candidate: bool = False
    review_required: bool = False
    review_reason: str = ""
    restorable: bool = False
    editable: bool = False
    # v0.8.22: geometry/raster application and content verification are separate.
    # ``applied`` only means pixels were written.  These fields answer whether the
    # expected source ink is present and target-language ink was actually removed.
    content_check: str = "not_checked"
    source_ink_coverage: float = 0.0
    target_residual_ratio: float = 0.0
    content_complete: bool = False
    repair_attempted: bool = False
    repair_succeeded: bool = False
    triage_state: str = "UNSET"  # SAFE|REVIEW|REJECT|UNSET
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class MaskTransferResult:
    image: np.ndarray
    layer_rgba: np.ndarray
    composite_mask: np.ndarray
    matches: list[BubblePatchMatch]
    records: list[MaskTransferRecord]
    clear_mask: np.ndarray | None = None

    @property
    def applied_count(self) -> int:
        return sum(1 for x in self.records if x.applied)




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
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


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


def _publication_safety_enabled(cfg) -> bool:
    """Legacy compatibility shim: publication blocking was removed in v1.0.6."""
    return False


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


def _soft_mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    af = np.clip(a.astype(np.float32) / 255.0, 0.0, 1.0)
    bf = np.clip(b.astype(np.float32) / 255.0, 0.0, 1.0)
    inter = float(np.minimum(af, bf).sum())
    union = float(np.maximum(af, bf).sum())
    return inter / max(1e-6, union)


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


def _pixel_enhance_text_raster(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray | None, dict]:
    """Sharpen low-resolution SOURCE text without OCR, reflow or glyph rebuilding."""
    if not bool(getattr(cfg, "pixel_enhance_enabled", True)) or cv2.countNonZero(mask) < 24:
        return None, {"enabled": False}
    use = mask > 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stroke = use & (gray < 235)
    if int(np.count_nonzero(stroke)) < int(getattr(cfg, "content_completeness_min_ink_pixels", 18)):
        return None, {"enabled": True, "reason": "too_little_ink"}
    scale = max(1.0, float(getattr(cfg, "pixel_enhance_upscale", 2.0)))
    h, w = image.shape[:2]
    if scale > 1.01:
        up = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_LANCZOS4)
        work = cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)
    else:
        work = image.copy()
    sigma = max(0.2, float(getattr(cfg, "pixel_enhance_unsharp_sigma", 0.85)))
    amount = max(0.0, float(getattr(cfg, "pixel_enhance_unsharp_amount", 0.55)))
    blur = cv2.GaussianBlur(work, (0, 0), sigma)
    sharp = cv2.addWeighted(work, 1.0 + amount, blur, -amount, 0)
    # Keep paper/background byte-stable. Only glyph-like pixels can change, and
    # cap darkening so JPEG halos are not converted into heavy fake strokes.
    out = image.copy()
    max_dark = max(0, int(getattr(cfg, "pixel_enhance_max_darkening", 28)))
    lo = np.maximum(image.astype(np.int16) - max_dark, 0).astype(np.uint8)
    candidate = np.maximum(sharp, lo)
    out[stroke] = candidate[stroke]
    before = _masked_sharpness(image, mask)
    after = _masked_sharpness(out, mask)
    if after <= before * 1.015:
        return None, {"enabled": True, "before_sharpness": float(before), "after_sharpness": float(after), "reason": "no_material_gain"}
    return out, {
        "enabled": True, "before_sharpness": float(before), "after_sharpness": float(after),
        "scale": float(scale), "amount": float(amount), "stroke_pixels": int(np.count_nonzero(stroke)),
    }


def _superresolve_patch(patch: np.ndarray, desired_scale: float, cfg: MaskReplaceConfig) -> tuple[np.ndarray, str, float]:
    if cfg.sr_backend == "off" or desired_scale < cfg.sr_min_trigger:
        return patch, "off", 1.0
    desired_scale = float(np.clip(desired_scale, 1.0, cfg.sr_max_scale))
    backend = cfg.sr_backend
    if backend == "auto":
        if cfg.sr_model_path:
            try:
                import spandrel  # noqa: F401
                import torch  # noqa: F401
                backend = "torch"
            except Exception:
                backend = "external" if cfg.sr_command else "lanczos"
        else:
            backend = "external" if cfg.sr_command else "lanczos"
    if backend == "torch":
        if not cfg.sr_model_path:
            if cfg.sr_backend == "torch":
                raise ValueError("mask_replace.sr_backend='torch' requires mask_replace.sr_model_path")
            backend = "lanczos"
        else:
            try:
                from .superres import upscale_patch
                result, actual = upscale_patch(
                    patch, desired_scale, model_path=cfg.sr_model_path,
                    device_preference=cfg.sr_device, precision=cfg.sr_precision,
                    tile_size=cfg.sr_tile_size, tile_overlap=cfg.sr_tile_overlap,
                    fallback_cpu=cfg.sr_fallback_cpu,
                )
                return result, "torch", float(actual)
            except Exception:
                if cfg.sr_backend == "torch":
                    raise
                backend = "lanczos"
    if backend == "external":
        if not cfg.sr_command:
            if cfg.sr_backend == "external":
                raise ValueError("mask_replace.sr_backend='external' requires mask_replace.sr_command")
            backend = "lanczos"
        else:
            with tempfile.TemporaryDirectory(prefix="mhd-sr-") as td:
                root = Path(td)
                inp, out = root / "input.png", root / "output.png"
                write_image(inp, patch)
                scale_int = 4 if desired_scale > 2.4 else 2
                rendered = cfg.sr_command.format(input=shlex.quote(str(inp)), output=shlex.quote(str(out)), scale=scale_int)
                proc = subprocess.run(rendered, shell=True, capture_output=True, text=True, timeout=cfg.sr_timeout_seconds)
                if proc.returncode == 0 and out.exists():
                    result = read_image(out)
                    actual = result.shape[1] / max(1, patch.shape[1])
                    return result, "external", float(actual)
                backend = "lanczos"
    if backend == "lanczos":
        nw = max(1, int(round(patch.shape[1] * desired_scale)))
        nh = max(1, int(round(patch.shape[0] * desired_scale)))
        out = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        if cfg.sharpen_amount > 0:
            blur = cv2.GaussianBlur(out, (0, 0), 0.8)
            out = cv2.addWeighted(out, 1.0 + cfg.sharpen_amount, blur, -cfg.sharpen_amount, 0)
        return out, "lanczos", nw / max(1, patch.shape[1])
    raise ValueError(f"Unknown mask_replace.sr_backend: {cfg.sr_backend}")


def _normalize_bubble_background(patch: np.ndarray, patch_mask: np.ndarray, target: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    src_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    src_sel = (patch_mask > 0) & (src_gray > 155)
    dst_sel = (target_mask > 0) & (dst_gray > 155)
    if np.count_nonzero(src_sel) < 20 or np.count_nonzero(dst_sel) < 20:
        return patch
    src_bg = np.median(patch[src_sel], axis=0).astype(np.float32)
    dst_bg = np.median(target[dst_sel], axis=0).astype(np.float32)
    gain = np.clip(dst_bg / np.maximum(src_bg, 20.0), 0.88, 1.18)
    out = patch.astype(np.float32) * gain.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


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


def _shift(image: np.ndarray, dx: float, dy: float, nearest: bool = False) -> np.ndarray:
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LANCZOS4
    border = 0 if image.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=border)



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


def _reconstruct_photo_crisp_layer(
    warped_img: np.ndarray,
    target: np.ndarray,
    paste_mask: np.ndarray,
    dest_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    allow_nonwhite_target: bool = False,
) -> tuple[np.ndarray | None, float]:
    """Recover photographed Chinese lettering as clean antialiased target-space ink.

    This is deliberately not OCR and does not invent glyphs. It estimates the
    source paper illumination, converts only locally-dark source detail into a
    soft ink alpha, removes tiny camera/JPEG speckles, guards a few pixels near
    the balloon boundary, clears the Japanese text on the clean target paper,
    then composites neutral black ink. Compared with direct photo-pixel pasting,
    it avoids blur, glare and duplicated balloon outlines; compared with hard
    binarization, it keeps antialiasing and does not thicken adjacent CJK strokes.
    """
    box = _bbox_from_mask(dest_mask)
    if box is None:
        return None, 0.0
    x0, y0, x1, y1 = box
    dmask = dest_mask[y0:y1, x0:x1] > 0
    pmask = paste_mask[y0:y1, x0:x1] > 0
    if np.count_nonzero(dmask) < 80 or np.count_nonzero(pmask) < 35:
        return None, 0.0

    # Never derive "text" from the photographed balloon border. A small erosion
    # keeps genuine lettering while dropping the doubled-outline artifact seen in
    # phone-shot editions. Keep the full destination mask for clearing Japanese.
    guard = max(0, int(cfg.photo_pair_crisp_border_guard_px))
    if guard > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard * 2 + 1, guard * 2 + 1))
        core = cv2.erode((pmask.astype(np.uint8) * 255), k) > 0
        # Very tiny balloons may collapse under erosion; retain the original mask.
        if np.count_nonzero(core) >= max(30, int(np.count_nonzero(pmask) * 0.55)):
            pmask = core

    tgt = target[y0:y1, x0:x1]
    target_mask_u8 = dmask.astype(np.uint8) * 255
    if (not allow_nonwhite_target
            and _target_white_ratio(tgt, target_mask_u8, cfg.ink_target_white_threshold) < max(0.70, cfg.ink_target_white_ratio)):
        return None, 0.0

    src = warped_img[y0:y1, x0:x1]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    work = gray.copy()
    paper_seed = int(np.percentile(gray[pmask], 90)) if np.any(pmask) else 245
    work[~pmask] = np.uint8(np.clip(paper_seed, 180, 255))

    side = max(9, min(x1 - x0, y1 - y0))
    ksize = int(np.clip(round(side * 0.20), 15, 55)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (0, 0), max(1.0, ksize / 11.0))
    detail = np.maximum(bg.astype(np.float32) - gray.astype(np.float32), 0.0)

    # A gentle local-contrast branch recovers strokes affected by curved-page
    # illumination without turning broad gray shadows into ink.
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6, 6))
    eq = clahe.apply(work)
    eq_bg = cv2.morphologyEx(eq, cv2.MORPH_CLOSE, kernel)
    eq_bg = cv2.GaussianBlur(eq_bg, (0, 0), max(1.0, ksize / 11.0))
    eq_detail = np.maximum(eq_bg.astype(np.float32) - eq.astype(np.float32), 0.0)
    detail = np.maximum(detail, eq_detail * 0.92)
    detail[~pmask] = 0.0

    floor = float(max(0.0, cfg.photo_pair_crisp_detail_floor))
    positive = detail[pmask & (detail > floor)]
    if positive.size < 16:
        return None, 0.0
    # Soft thresholds are the key difference from the older binary recovery.
    # Low contrast becomes antialiasing, strong strokes become black, and nearby
    # strokes do not get merged by morphological closing.
    lo = max(floor, float(np.percentile(positive, 18)) * 0.70)
    hi = max(lo + 8.0, float(np.percentile(positive, 88)) * 0.92)
    alpha = np.clip((detail - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    gamma = float(np.clip(cfg.photo_pair_crisp_alpha_gamma, 0.45, 1.40))
    alpha = np.power(alpha, gamma)

    amount = float(np.clip(cfg.photo_pair_crisp_unsharp_amount, 0.0, 1.5))
    if amount > 0:
        blur = cv2.GaussianBlur(alpha, (0, 0), 0.70)
        alpha = np.clip(alpha + amount * (alpha - blur), 0.0, 1.0)
    alpha[~pmask] = 0.0

    # Component cleanup is driven by a permissive seed, while the final edges
    # remain soft. This removes isolated photo grain without redrawing glyphs.
    seed = (alpha >= 0.16).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    keep = np.zeros_like(seed, np.uint8)
    min_area = max(1, int(cfg.photo_pair_crisp_min_component_area))
    max_area = max(12, int(np.count_nonzero(pmask) * cfg.photo_pair_crisp_max_ink_ratio))
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            keep[labels == i] = 1
    alpha[keep == 0] = 0.0
    ink_ratio = float(np.count_nonzero(alpha >= 0.16) / max(1, np.count_nonzero(pmask)))
    if ink_ratio < cfg.ink_min_ratio or ink_ratio > cfg.photo_pair_crisp_max_ink_ratio:
        return None, ink_ratio

    clear_gray = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
    bright = dmask & (clear_gray >= cfg.ink_target_white_threshold)
    if np.count_nonzero(bright) >= 20:
        paper_bgr = np.median(tgt[bright], axis=0).astype(np.float32)
    else:
        paper_bgr = np.array([252.0, 252.0, 252.0], np.float32)

    out = target.copy()
    roi = out[y0:y1, x0:x1].astype(np.float32)
    roi[dmask] = paper_bgr
    a = alpha[..., None]
    black = np.zeros_like(roi, np.float32)
    roi = roi * (1.0 - a) + black * a
    out[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
    return out, ink_ratio


def _normalize_photo_text_pixels(
    warped_img: np.ndarray,
    target: np.ndarray,
    paste_mask: np.ndarray,
    dest_mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> np.ndarray | None:
    """Flatten photographed bubble illumination while preserving real glyph pixels.

    Phone photos can be perfectly registered but still contain a smooth gray/blue
    glare field. Copying those pixels makes the clean target bubble look dirty,
    while hard thresholding destroys antialiasing and can merge small CJK strokes.
    This deterministic transform estimates the local paper illumination by a
    grayscale morphological closing, transfers only the *dark detail relative to
    that paper field*, and paints it over the clean target paper colour.

    No glyph is generated or inferred here: every dark detail comes from the
    registered source photograph. If the target is not a mostly-white text area,
    return None so the caller can use the normal pixel/ink/reletter policies.
    """
    box = _bbox_from_mask(dest_mask)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    dmask = dest_mask[y0:y1, x0:x1] > 0
    pmask = paste_mask[y0:y1, x0:x1] > 0
    if np.count_nonzero(dmask) < 80 or np.count_nonzero(pmask) < 40:
        return None

    tgt = target[y0:y1, x0:x1]
    target_mask_u8 = dmask.astype(np.uint8) * 255
    if _target_white_ratio(tgt, target_mask_u8, cfg.ink_target_white_threshold) < max(0.72, cfg.ink_target_white_ratio):
        return None

    src = warped_img[y0:y1, x0:x1]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Outside the valid source region must not influence the estimated light field.
    work = gray.copy()
    if np.any(pmask):
        paper_seed = int(np.percentile(gray[pmask], 88))
    else:
        paper_seed = 245
    work[~pmask] = np.uint8(np.clip(paper_seed, 180, 255))

    side = max(9, min(x1 - x0, y1 - y0))
    # Kernel should be wider than typical glyph strokes/characters but remain
    # local enough to follow page glare and curved-book illumination gradients.
    k = int(np.clip(round(side * 0.22), 17, 61)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), max(1.2, k / 10.0))

    detail = np.maximum(background.astype(np.float32) - gray.astype(np.float32), 0.0)
    # Specular glare can turn one side of a glyph from black into mid-gray while
    # leaving the local paper even brighter. A CLAHE branch recovers that *local*
    # contrast without changing character topology; take the stronger evidence
    # from the raw and equalized views.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    eq = clahe.apply(work)
    eq_bg = cv2.morphologyEx(eq, cv2.MORPH_CLOSE, kernel)
    eq_bg = cv2.GaussianBlur(eq_bg, (0, 0), max(1.2, k / 10.0))
    eq_detail = np.maximum(eq_bg.astype(np.float32) - eq.astype(np.float32), 0.0)
    detail = np.maximum(detail, eq_detail * 1.12)
    # Ignore tiny photographic texture/compression residue so a clean Japanese
    # bubble does not inherit gray smudges from the phone photo.
    floor = float(max(0.0, cfg.photo_pair_normalize_detail_floor))
    if floor > 0:
        detail = np.maximum(detail - floor, 0.0)
    detail[~pmask] = 0.0
    positive = detail[pmask & (detail > 2.0)]
    if positive.size < 20:
        return None

    # Normalize contrast from the photographed page to a clean manga-paper range.
    # A percentile scale keeps antialiasing while preventing a few black cores from
    # making every other stroke too pale.
    p92 = float(np.percentile(positive, 92))
    scale = float(np.clip((225.0 / max(45.0, p92)) * cfg.photo_pair_normalize_contrast_gain, 1.0, 2.7))

    clear_gray = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
    bright = dmask & (clear_gray >= cfg.ink_target_white_threshold)
    if np.count_nonzero(bright) >= 20:
        paper_bgr = np.median(tgt[bright], axis=0).astype(np.float32)
    else:
        paper_bgr = np.array([250.0, 250.0, 250.0], np.float32)
    paper_luma = float(np.mean(paper_bgr))

    corrected = np.clip(paper_luma - detail * scale, 0, 255).astype(np.uint8)
    amount = float(np.clip(cfg.photo_pair_normalize_unsharp_amount, 0.0, 1.0))
    if amount > 0:
        blur = cv2.GaussianBlur(corrected, (0, 0), 0.75)
        corrected = cv2.addWeighted(corrected, 1.0 + amount, blur, -amount, 0)

    out = target.copy()
    # Clear the entire clean target interior first so Japanese glyphs disappear,
    # then place normalized source detail only where valid source pixels exist.
    roi = out[y0:y1, x0:x1]
    roi[dmask] = np.clip(paper_bgr, 0, 255).astype(np.uint8)
    # Keep neutral antialiasing; source photo chroma/glare is intentionally dropped.
    norm_bgr = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
    roi[pmask] = norm_bgr[pmask]
    out[y0:y1, x0:x1] = roi
    return out

def _masked_sharpness(image: np.ndarray, mask: np.ndarray) -> float:
    """Text-oriented sharpness, not whole-white-bubble sharpness.

    Measuring the entire bubble is easily fooled by JPEG grain or a crisp box
    border. Prefer dark glyph neighbourhoods; fall back to the full interior only
    when no text-like pixels are present.
    """
    if cv2.countNonZero(mask) == 0:
        return 0.0
    box = _bbox_from_mask(mask)
    if not box:
        return 0.0
    x0, y0, x1, y1 = box
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    m = mask[y0:y1, x0:x1] > 0
    if gray.size == 0 or np.count_nonzero(m) < 20:
        return 0.0
    dark = m & (gray < 190)
    if np.count_nonzero(dark) >= 12:
        dark = cv2.dilate(dark.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
        sel = m & dark
    else:
        sel = m
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    vals = lap[sel]
    return float(np.var(vals)) if vals.size else 0.0


def _target_white_ratio(image: np.ndarray, mask: np.ndarray, threshold: int) -> float:
    if cv2.countNonZero(mask) == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sel = mask > 0
    return float(np.mean(gray[sel] >= int(threshold))) if np.any(sel) else 0.0


def _reconstruct_ink_layer(
    warped_img: np.ndarray,
    target: np.ndarray,
    paste_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    clear_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    """Recover crisp black glyphs from a soft photographed source bubble.

    This path never invents characters. It only binarizes the already-existing
    source ink after geometric alignment, cleans tiny camera/JPEG speckles,
    clears the target bubble interior to its local bright paper colour, and
    paints the recovered ink deterministically. It is intentionally limited to
    mostly-white speech/narration interiors.
    """
    box = _bbox_from_mask(paste_mask)
    if not box:
        return None, 0.0
    x0, y0, x1, y1 = box
    pm = paste_mask[y0:y1, x0:x1] > 0
    if np.count_nonzero(pm) < 40:
        return None, 0.0
    tgt = target[y0:y1, x0:x1]
    if _target_white_ratio(tgt, (pm.astype(np.uint8) * 255), cfg.ink_target_white_threshold) < cfg.ink_target_white_ratio:
        return None, 0.0

    src = warped_img[y0:y1, x0:x1]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Photographed pages often have smooth illumination gradients. CLAHE makes
    # dark strokes locally separable without hallucinating lost details.
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    block = max(15, int(cfg.ink_adaptive_block_size) | 1)
    adaptive = cv2.adaptiveThreshold(
        eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block, int(cfg.ink_adaptive_c),
    )
    # A conservative global threshold catches thick dark strokes that adaptive
    # thresholding can split under glare. Keep only pixels inside the target.
    vals = eq[pm]
    if vals.size:
        # Preserve antialiased/photographed glyph strokes, not only the darkest
        # core pixels. A local paper-relative threshold plus masked Otsu is much
        # more stable for small Chinese characters photographed slightly out of
        # focus (where the old threshold could collapse a glyph into black bars).
        q = float(np.percentile(vals, 32))
        global_ink = (eq <= min(185.0, q + 10.0)).astype(np.uint8) * 255
        paper_level = float(np.percentile(vals, 86))
        relative_thr = float(np.clip(paper_level - 26.0, 105.0, 210.0))
        relative_ink = (eq <= relative_thr).astype(np.uint8) * 255
        try:
            otsu_thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_thr = float(np.clip(otsu_thr + 10.0, 100.0, 205.0))
            otsu_ink = (eq <= otsu_thr).astype(np.uint8) * 255
        except cv2.error:
            otsu_ink = np.zeros_like(eq, np.uint8)
        ink = cv2.bitwise_or(adaptive, global_ink)
        ink = cv2.bitwise_or(ink, relative_ink)
        ink = cv2.bitwise_or(ink, otsu_ink)
    else:
        ink = adaptive
    ink[~pm] = 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    cleaned = np.zeros_like(ink)
    max_area = max(8, int(np.count_nonzero(pm) * cfg.ink_max_component_area_ratio))
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if cfg.ink_min_component_area <= area <= max_area:
            cleaned[labels == i] = 255
    # One tiny close reconnects antialiased strokes after photographic blur.
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    cleaned[~pm] = 0
    ratio = cv2.countNonZero(cleaned) / max(1, int(np.count_nonzero(pm)))
    if ratio < cfg.ink_min_ratio or ratio > cfg.ink_max_ratio:
        return None, float(ratio)

    out = target.copy()
    # Estimate the clean target paper colour from the brightest pixels in the
    # destination interior, then remove Japanese glyphs deterministically.
    # For photographed pairs ``clear_mask`` may be larger than the valid source
    # mask: clear the whole clean target balloon, but only derive Chinese ink from
    # source pixels that actually belong to the matched source balloon.
    full_clear = clear_mask if clear_mask is not None else paste_mask
    cb = _bbox_from_mask(full_clear)
    if cb is None:
        return None, float(ratio)
    cx0, cy0, cx1, cy1 = cb
    clear_local = full_clear[cy0:cy1, cx0:cx1] > 0
    target_clear_crop = target[cy0:cy1, cx0:cx1]
    clear_gray = cv2.cvtColor(target_clear_crop, cv2.COLOR_BGR2GRAY)
    bright = clear_local & (clear_gray >= cfg.ink_target_white_threshold)
    if np.count_nonzero(bright) >= 20:
        paper = np.median(target_clear_crop[bright], axis=0).astype(np.uint8)
    else:
        paper = np.array([255, 255, 255], np.uint8)
    clear_roi = out[cy0:cy1, cx0:cx1]
    clear_roi[clear_local] = paper
    out[cy0:cy1, cx0:cx1] = clear_roi
    roi = out[y0:y1, x0:x1]
    roi[cleaned > 0] = (0, 0, 0)
    out[y0:y1, x0:x1] = roi
    return out, float(ratio)


def _alpha_from_mask(mask: np.ndarray, feather_px: int) -> np.ndarray:
    alpha = (mask > 0).astype(np.float32)
    if feather_px > 0:
        k = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), max(0.5, feather_px * 0.55))
        alpha = np.clip(alpha, 0.0, 1.0)
    return alpha


def _expand_target_clear_mask_with_text_components(target: np.ndarray, target_mask: np.ndarray, cfg=None) -> np.ndarray:
    """Add compact target text components just outside an imperfect balloon mask.

    Bright-region balloon detection can stop at Japanese glyphs that nearly bridge
    the interior, especially in tiny balloons. The missing glyph tops then survive
    replacement and visually mix with Chinese. Add only compact dark components
    touching a small dilation of the target interior; reject long outline/panel
    components so the balloon border remains intact.
    """
    box = _bbox_from_mask(target_mask)
    if box is None:
        return target_mask.copy()
    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    pad = max(5, int(round(0.10 * max(bw, bh))))
    near = cv2.dilate(
        target_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)),
    ) > 0
    dark = ((gray < 190) & near).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    out = target_mask.copy()
    mask_area = max(1, cv2.countNonZero(target_mask))
    for i in range(1, n):
        _, _, cw, ch, area = map(int, stats[i])
        if area < 2 or area > 0.14 * mask_area:
            continue
        # v0.8.21: vertical Japanese columns can legitimately span most of a
        # balloon's height, and horizontal captions can span most of its width.
        # The old independent 42%/48% limits rejected exactly those glyph groups.
        # Keep long components only when the orthogonal dimension remains
        # text-column-like; true balloon/panel outlines are long in both axes or
        # become very large connected components and are still rejected above.
        long_vertical = ch <= 0.88 * bh and cw <= 0.28 * bw
        long_horizontal = cw <= 0.88 * bw and ch <= 0.30 * bh
        compact = cw <= 0.48 * bw and ch <= 0.54 * bh
        # Tiny target text boxes often have a final Japanese column pressed
        # against the edge. It is thin enough to be mistaken for the box border;
        # accept it only inside the confirmed text-box bbox and only when it is
        # clearly text-column-like, never as a general edge expansion.
        thin_edge_text = (
            bool(getattr(cfg, "paired_diff_clear_thin_edge_text", True))
            and cw <= float(getattr(cfg, "paired_diff_clear_thin_edge_text_max_width_ratio", 0.10)) * bw
            and ch <= float(getattr(cfg, "paired_diff_clear_thin_edge_text_max_height_ratio", 0.82)) * bh
        )
        if not (compact or long_vertical or long_horizontal or thin_edge_text):
            continue
        comp = (labels == i).astype(np.uint8) * 255
        # Include antialiased fringe around the old Japanese glyph so review-first
        # replacement does not leave a faint gray ghost after the black core is
        # cleared. One pixel is enough and still far smaller than balloon outlines.
        comp = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        out = np.maximum(out, comp)
    return out



def _extract_photo_text_cluster(
    aligned_source: np.ndarray,
    target_mask: np.ndarray,
    *,
    pad_ratio: float,
    dark_threshold: int,
    cluster_gap_ratio: float,
) -> tuple[np.ndarray, tuple[int, int, int, int], float, float] | None:
    """Extract the dominant compact source-ink cluster around a target bubble.

    The input page is already in target coordinates.  We intentionally do not use
    OCR or character boxes: compact dark connected components are grouped by local
    proximity, while long balloon/panel/art lines are discarded.  Returning the
    original raster footprint (plus a soft alpha later) lets mask replacement keep
    punctuation, glyph shapes, column layout and relative spacing verbatim.
    """
    box = _bbox_from_mask(target_mask)
    if box is None:
        return None
    h, w = aligned_source.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    pad = max(8, int(round(max(bw, bh) * float(np.clip(pad_ratio, 0.08, 0.55)))))
    xa, ya = max(0, x0 - pad), max(0, y0 - pad)
    xb, yb = min(w, x1 + pad), min(h, y1 + pad)
    roi = aligned_source[ya:yb, xa:xb]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gate_full = cv2.dilate(
        target_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)),
    )
    gate = gate_full[ya:yb, xa:xb] > 0
    vals = gray[gate]
    if vals.size < 80:
        return None
    paper = float(np.percentile(vals, 88.0))
    thr = float(np.clip(min(float(dark_threshold), paper - 34.0), 105.0, 190.0))
    ink = ((gray <= thr) & gate).astype(np.uint8) * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    clean = np.zeros_like(ink)
    gate_area = max(1, int(np.count_nonzero(gate)))
    rw, rh = ink.shape[1], ink.shape[0]
    for i in range(1, n):
        _, _, cw, ch, area = map(int, stats[i])
        if area < 2 or area > 0.12 * gate_area:
            continue
        # Long components are almost always balloon outlines, panel borders,
        # hair/furniture edges or hatch lines rather than a CJK glyph component.
        if cw > 0.45 * rw or ch > 0.45 * rh:
            continue
        clean[labels == i] = 255
    if cv2.countNonZero(clean) < 12:
        return None

    # Join adjacent glyph pieces/characters into text blocks.  This is the key
    # outlier guard missing from the old recenter path: isolated speckles or art
    # fragments no longer enlarge the text bbox and force destructive shrinking.
    gap = int(np.clip(round(min(bw, bh) * float(cluster_gap_ratio)), 3, 14))
    grouped = cv2.dilate(
        clean,
        cv2.getStructuringElement(cv2.MORPH_RECT, (gap * 2 + 1, gap * 2 + 1)),
    )
    gn, glabels, _, _ = cv2.connectedComponentsWithStats((grouped > 0).astype(np.uint8), 8)
    tm_roi = target_mask[ya:yb, xa:xb] > 0
    tcx = (x0 + x1) * 0.5 - xa
    tcy = (y0 + y1) * 0.5 - ya
    best: tuple[float, np.ndarray] | None = None
    for i in range(1, gn):
        region = glabels == i
        original = (clean > 0) & region
        area = int(np.count_nonzero(original))
        if area < 12:
            continue
        ys, xs = np.where(original)
        bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        bbox_area = max(1, (bx1 - bx0) * (by1 - by0))
        fill = area / bbox_area
        overlap = float(np.count_nonzero(original & tm_roi) / max(1, area))
        cx, cy = float(xs.mean()), float(ys.mean())
        dist = math.hypot((cx - tcx) / bw, (cy - tcy) / bh)
        # A real dialogue block is compact, mostly intersects its target bubble,
        # and sits near that bubble.  Area stays dominant so punctuation columns
        # and multi-column CJK layouts remain together.
        score = (
            area
            * (0.62 + 0.38 * min(1.0, fill / 0.10))
            * (0.70 + 0.30 * overlap)
            / (1.0 + 0.22 * dist)
        )
        if best is None or score > best[0]:
            best = (score, original)
    if best is None:
        return None

    selected = best[1].astype(np.uint8) * 255
    ys, xs = np.where(selected > 0)
    if len(xs) == 0:
        return None
    ibox_local = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    ibox = (
        ibox_local[0] + xa,
        ibox_local[1] + ya,
        ibox_local[2] + xa,
        ibox_local[3] + ya,
    )

    # Re-introduce the antialiased fringe around the accepted components, but do
    # not resurrect long components that were deliberately classified as art.
    fringe = cv2.dilate(selected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    denom = max(35.0, paper - 82.0)
    alpha = np.clip((paper - gray.astype(np.float32)) / denom, 0.0, 1.0)
    alpha = np.power(alpha, 0.92)
    alpha[~fringe] = 0.0
    alpha[selected > 0] = np.maximum(alpha[selected > 0], 0.72)

    alpha_global = np.zeros(target_mask.shape, np.float32)
    alpha_global[ya:yb, xa:xb] = alpha
    raw_outside_ratio = float(
        np.count_nonzero((selected > 0) & (~tm_roi)) / max(1, cv2.countNonZero(selected))
    )
    ink_ratio = float(cv2.countNonZero(selected) / max(1, cv2.countNonZero(target_mask)))
    return alpha_global, ibox, raw_outside_ratio, ink_ratio


def _fit_alpha_into_target_mask(
    alpha: np.ndarray,
    target_mask: np.ndarray,
    text_bbox: tuple[int, int, int, int],
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray | None, float, float, float, float]:
    """Move/shrink one raster glyph block minimally until all strokes fit.

    Search starts at scale 1.0 and only reduces scale if translation alone cannot
    reach the configured containment target.  X/Y are never scaled independently,
    so glyph proportions and the source typesetting remain unchanged.
    """
    h, w = target_mask.shape
    x0, y0, x1, y1 = text_bbox
    if x1 <= x0 or y1 <= y0:
        return None, 0.0, 0.0, 1.0, 0.0
    crop = alpha[y0:y1, x0:x1]
    if crop.size == 0 or float(crop.max()) <= 0:
        return None, 0.0, 0.0, 1.0, 0.0

    tbox = _bbox_from_mask(target_mask)
    if tbox is None:
        return None, 0.0, 0.0, 1.0, 0.0
    tbw, tbh = max(1, tbox[2] - tbox[0]), max(1, tbox[3] - tbox[1])
    inset = max(0, int(getattr(cfg, "photo_pair_glyph_rescue_safe_inset_px", 1)))
    safe = target_mask.copy()
    if inset > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1))
        eroded = cv2.erode(safe, k)
        if cv2.countNonZero(eroded) > max(20, int(cv2.countNonZero(safe) * 0.70)):
            safe = eroded
    safe_bool = safe > 0

    min_scale = float(np.clip(getattr(cfg, "photo_pair_glyph_rescue_min_scale", 0.86), 0.65, 1.0))
    step = float(np.clip(getattr(cfg, "photo_pair_glyph_rescue_scale_step", 0.02), 0.01, 0.10))
    scales = [1.0]
    s = 1.0 - step
    while s >= min_scale - 1e-9:
        scales.append(round(s, 4))
        s -= step
    required = float(np.clip(getattr(cfg, "photo_pair_glyph_rescue_min_coverage", 0.995), 0.94, 1.0))
    max_shift = max(3, int(round(max(tbw, tbh) * float(np.clip(
        getattr(cfg, "photo_pair_glyph_rescue_max_shift_ratio", 0.14), 0.03, 0.30
    )))))

    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    best_any: tuple[float, float, int, int, float, np.ndarray, int, int] | None = None
    for scale in scales:
        nw = max(1, int(round(crop.shape[1] * scale)))
        nh = max(1, int(round(crop.shape[0] * scale)))
        interp = cv2.INTER_AREA if scale < 0.999 else cv2.INTER_LINEAR
        scaled = cv2.resize(crop, (nw, nh), interpolation=interp)
        binary = scaled >= 0.14
        yy0, xx0 = np.where(binary)
        if len(xx0) < 8:
            continue
        base_x = int(round(cx - nw * 0.5))
        base_y = int(round(cy - nh * 0.5))

        local_best: tuple[float, int, int] | None = None
        # Manhattan shells make the first full-coverage hit the least invasive.
        for radius in range(0, max_shift + 1):
            shell: list[tuple[int, int]] = []
            if radius == 0:
                shell = [(0, 0)]
            else:
                for dx in range(-radius, radius + 1):
                    dy = radius - abs(dx)
                    shell.append((dx, dy))
                    if dy:
                        shell.append((dx, -dy))
            for dx, dy in shell:
                gx = xx0 + base_x + dx
                gy = yy0 + base_y + dy
                valid = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
                if not np.any(valid):
                    coverage = 0.0
                else:
                    vv = valid.copy()
                    inside = np.zeros_like(vv)
                    inside[vv] = safe_bool[gy[vv], gx[vv]]
                    coverage = float(np.count_nonzero(inside) / len(xx0))
                if local_best is None or coverage > local_best[0] + 1e-9:
                    local_best = (coverage, dx, dy)
                if coverage >= required:
                    out = np.zeros_like(alpha, np.float32)
                    px, py = base_x + dx, base_y + dy
                    sx0, sy0 = max(0, -px), max(0, -py)
                    dx0, dy0 = max(0, px), max(0, py)
                    cw = min(nw - sx0, w - dx0); ch = min(nh - sy0, h - dy0)
                    if cw <= 0 or ch <= 0:
                        continue
                    out[dy0:dy0 + ch, dx0:dx0 + cw] = scaled[sy0:sy0 + ch, sx0:sx0 + cw]
                    return out, float(dx), float(dy), float(scale), coverage
        if local_best is not None:
            coverage, dx, dy = local_best
            key = (coverage, scale, -(abs(dx) + abs(dy)))
            if best_any is None or key > (best_any[0], best_any[1], -best_any[2]):
                best_any = (coverage, scale, abs(dx) + abs(dy), dx, dy, scaled, base_x, base_y)

    # Do not publish a partially clipped "rescue". The ordinary crisp path below
    # is safer if the complete raster block cannot be contained with small motion.
    return None, 0.0, 0.0, 1.0, (best_any[0] if best_any is not None else 0.0)


def _reconstruct_photo_glyph_footprint_layer(
    aligned_source: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    clear_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float, dict[str, float]]:
    """Rescue complete source glyphs that straddle an imperfect target mask.

    This is a pure mask/pixel path: no OCR, no transcript, no font rendering.  It
    only activates when the dominant translated source-ink cluster demonstrably
    has pixels outside the target bubble mask.  The whole raster text block is
    then translated (and, only if necessary, uniformly shrunk) to achieve almost
    full containment.  Relative layout, punctuation and glyph shapes are preserved.
    """
    if not bool(getattr(cfg, "photo_pair_glyph_rescue_enabled", True)):
        return None, 0.0, {}
    box = _bbox_from_mask(target_mask)
    if box is None:
        return None, 0.0, {}
    h, w = target.shape[:2]
    area_ratio = cv2.countNonZero(target_mask) / max(1, h * w)
    if area_ratio > float(getattr(cfg, "photo_pair_glyph_rescue_max_area_ratio", 0.028)):
        return None, 0.0, {}
    if _target_white_ratio(target, target_mask, cfg.ink_target_white_threshold) < 0.62:
        return None, 0.0, {}

    extracted = _extract_photo_text_cluster(
        aligned_source,
        target_mask,
        pad_ratio=float(getattr(cfg, "photo_pair_glyph_rescue_pad_ratio", 0.30)),
        dark_threshold=int(getattr(cfg, "photo_pair_glyph_rescue_dark_threshold", 175)),
        cluster_gap_ratio=float(getattr(cfg, "photo_pair_glyph_rescue_cluster_gap_ratio", 0.07)),
    )
    if extracted is None:
        return None, 0.0, {}
    alpha, ibox, outside_ratio, ink_ratio = extracted
    trigger = float(getattr(cfg, "photo_pair_glyph_rescue_min_outside_ink_ratio", 0.003))
    if outside_ratio < trigger:
        return None, ink_ratio, {"outside_ratio": outside_ratio}

    fitted, dx, dy, scale, coverage = _fit_alpha_into_target_mask(alpha, target_mask, ibox, cfg)
    if fitted is None:
        return None, ink_ratio, {"outside_ratio": outside_ratio, "coverage": coverage}

    out = target.copy()
    clear = clear_mask if clear_mask is not None else target_mask
    cb = _bbox_from_mask(clear)
    if cb is None:
        return None, ink_ratio, {}
    cx0, cy0, cx1, cy1 = cb
    clear_local = clear[cy0:cy1, cx0:cx1] > 0
    tgt_crop = target[cy0:cy1, cx0:cx1]
    tgray = cv2.cvtColor(tgt_crop, cv2.COLOR_BGR2GRAY)
    bright = clear_local & (tgray >= cfg.ink_target_white_threshold)
    if np.count_nonzero(bright) >= 20:
        paper_bgr = np.median(tgt_crop[bright], axis=0).astype(np.float32)
    else:
        paper_bgr = np.array([255.0, 255.0, 255.0], np.float32)
    clear_roi = out[cy0:cy1, cx0:cx1]
    clear_roi[clear_local] = np.clip(paper_bgr, 0, 255).astype(np.uint8)
    out[cy0:cy1, cx0:cx1] = clear_roi

    a = np.clip(fitted[..., None], 0.0, 1.0)
    out = np.clip(out.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)
    return out, ink_ratio, {
        "outside_ratio": outside_ratio,
        "coverage": coverage,
        "dx": dx,
        "dy": dy,
        "scale": scale,
    }


def _reconstruct_photo_recentered_ink_layer(
    aligned_source: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    clear_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    """Recover a complete small translated glyph block and re-center it.

    Cross-edition pages often keep the same speech balloon while the translated
    Chinese columns occupy slightly different positions from the Japanese text.
    The v0.8.16 implementation shares the clustered source-ink extractor with the
    boundary rescue path, so isolated JPEG/art speckles cannot enlarge the text
    bbox and force the entire block to shrink. No OCR or glyph generation occurs.
    """
    box = _bbox_from_mask(target_mask)
    if box is None:
        return None, 0.0
    h, w = target.shape[:2]
    x0, y0, x1, y1 = box
    area_ratio = cv2.countNonZero(target_mask) / max(1, h * w)
    if area_ratio > float(getattr(cfg, "photo_pair_recenter_max_area_ratio", 0.006)):
        return None, 0.0
    bw, bh = x1 - x0, y1 - y0
    if min(bw, bh) < 24:
        return None, 0.0
    if _target_white_ratio(target, target_mask, cfg.ink_target_white_threshold) < 0.62:
        return None, 0.0

    extracted = _extract_photo_text_cluster(
        aligned_source,
        target_mask,
        pad_ratio=float(getattr(cfg, "photo_pair_recenter_pad_ratio", 0.30)),
        dark_threshold=int(getattr(cfg, "photo_pair_recenter_dark_threshold", 175)),
        cluster_gap_ratio=float(getattr(cfg, "photo_pair_glyph_rescue_cluster_gap_ratio", 0.07)),
    )
    if extracted is None:
        return None, 0.0
    alpha_global, ibox, _, _ = extracted
    ix0, iy0, ix1, iy1 = ibox
    if ix1 - ix0 < 5 or iy1 - iy0 < 5:
        return None, 0.0
    crop = alpha_global[iy0:iy1, ix0:ix1]
    if crop.size == 0 or float(crop.max()) <= 0:
        return None, 0.0

    # Derive a safe destination interior from the target balloon and fit with one
    # scale factor only—never squeeze X/Y independently.
    safe = cv2.erode(target_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    sbox = _bbox_from_mask(safe) or box
    sx0, sy0, sx1, sy1 = sbox
    fit_ratio = float(np.clip(getattr(cfg, "photo_pair_recenter_fit_ratio", 0.88), 0.60, 0.96))
    avail_w = max(1.0, (sx1 - sx0) * fit_ratio)
    avail_h = max(1.0, (sy1 - sy0) * fit_ratio)
    scale = min(avail_w / max(1, crop.shape[1]), avail_h / max(1, crop.shape[0]))
    if not np.isfinite(scale) or scale <= 0:
        return None, 0.0
    nw = max(1, int(round(crop.shape[1] * scale)))
    nh = max(1, int(round(crop.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    alpha_small = np.clip(cv2.resize(crop, (nw, nh), interpolation=interp), 0.0, 1.0)

    cx, cy = (sx0 + sx1) * 0.5, (sy0 + sy1) * 0.5
    dx = int(round(cx - nw * 0.5)); dy = int(round(cy - nh * 0.5))
    dx = max(0, min(w - nw, dx)); dy = max(0, min(h - nh, dy))

    out = target.copy()
    tgray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    clear = clear_mask if clear_mask is not None else target_mask
    sel = clear > 0
    bright = (target_mask > 0) & (tgray >= cfg.ink_target_white_threshold)
    if np.count_nonzero(bright) >= 20:
        paper_bgr = np.median(target[bright], axis=0).astype(np.float32)
    else:
        paper_bgr = np.array([255.0, 255.0, 255.0], np.float32)
    out[sel] = np.clip(paper_bgr, 0, 255).astype(np.uint8)
    roi_out = out[dy:dy + nh, dx:dx + nw].astype(np.float32)
    a = alpha_small[..., None]
    roi_out = roi_out * (1.0 - a)
    out[dy:dy + nh, dx:dx + nw] = np.clip(roi_out, 0, 255).astype(np.uint8)
    ink_ratio = float(np.count_nonzero(alpha_small > 0.10) / max(1, cv2.countNonZero(target_mask)))
    if ink_ratio < cfg.ink_min_ratio or ink_ratio > 0.36:
        return None, ink_ratio
    return out, ink_ratio


def _complex_text_ink_map(image: np.ndarray) -> np.ndarray:
    """Language-agnostic compact ink map for open/coloured text transfer."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    ink = cv2.adaptiveThreshold(
        eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10,
    )
    ink = cv2.bitwise_or(ink, (eq <= 150).astype(np.uint8) * 255)
    return ink


def _select_changed_text_components(
    ink: np.ndarray,
    unique_seed: np.ndarray,
    gate: np.ndarray,
    gap_px: int,
) -> np.ndarray:
    """Keep whole compact ink groups that contain language-change evidence.

    ``unique_seed`` contains strokes present only in this edition after a small
    tolerance.  We use it only to *select* a group, then restore all compact ink
    components belonging to that group so overlapping Chinese/Japanese strokes do
    not create broken glyphs.  Long panel/balloon/hair lines are discarded before
    grouping.
    """
    masked = ((ink > 0) & (gate > 0)).astype(np.uint8)
    if not np.any(masked):
        return np.zeros_like(ink, np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(masked, 8)
    ys, xs = np.where(gate > 0)
    if len(xs) == 0:
        return np.zeros_like(ink, np.uint8)
    gw = max(1, int(xs.max() - xs.min() + 1)); gh = max(1, int(ys.max() - ys.min() + 1))
    gate_area = max(1, int(np.count_nonzero(gate)))
    clean = np.zeros_like(ink, np.uint8)
    for i in range(1, n):
        _x, _y, cw, ch, area = [int(v) for v in stats[i]]
        if area < 2 or area > max(40, int(0.12 * gate_area)):
            continue
        if cw > 0.64 * gw or ch > 0.64 * gh:
            # Allow a tall narrow text column / wide short caption, but not long
            # line-art structures spanning the candidate in both directions.
            if not ((ch <= 0.92 * gh and cw <= 0.22 * gw) or (cw <= 0.92 * gw and ch <= 0.24 * gh)):
                continue
        aspect = max(cw / max(1.0, ch), ch / max(1.0, cw))
        if aspect > 12.0:
            continue
        clean[labels == i] = 255
    if cv2.countNonZero(clean) == 0:
        return clean

    gap = max(2, int(gap_px))
    grouped = cv2.dilate(
        clean,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap * 2 + 1, gap * 2 + 1)),
    )
    gn, glabels, _, _ = cv2.connectedComponentsWithStats((grouped > 0).astype(np.uint8), 8)
    out = np.zeros_like(clean)
    seed = (unique_seed > 0) & (gate > 0)
    for i in range(1, gn):
        region = glabels == i
        original = (clean > 0) & region
        pixels = int(np.count_nonzero(original))
        if pixels < 8:
            continue
        evidence = int(np.count_nonzero(seed & region))
        if evidence < max(2, int(round(0.015 * pixels))):
            continue
        out[original] = 255
    return out


def _soft_ink_alpha(image: np.ndarray, ink_mask: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Recover antialiased glyph opacity without copying the source background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gvals = gray[gate > 0]
    ivals = gray[ink_mask > 0]
    if gvals.size == 0 or ivals.size == 0:
        return (ink_mask > 0).astype(np.float32)
    paper = float(np.percentile(gvals, 82.0))
    dark = float(np.percentile(ivals, 18.0))
    denom = max(20.0, paper - dark)
    tone = np.clip((paper - gray) / denom, 0.0, 1.0)
    support = cv2.GaussianBlur((ink_mask > 0).astype(np.float32), (0, 0), 0.65)
    return np.clip(np.maximum(tone * (ink_mask > 0), support * 0.72), 0.0, 1.0)


def _dominant_saturated_container_mask(
    target: np.ndarray,
    region_mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> np.ndarray | None:
    """Recover the flat-colour interior of a burst balloon near ``region_mask``.

    This route is deliberately colour-geometry only.  It is used for saturated
    yellow/red burst balloons where the ordinary white-container detector cannot
    provide an interior mask.  Text holes are closed by filling the selected
    colour component's *external* contour; panel/SFX colours with a different hue
    therefore stay outside the writable area.
    """
    if not bool(getattr(cfg, "paired_diff_saturated_container_enabled", True)):
        return None
    box = _bbox_from_mask(region_mask)
    if box is None:
        return None
    h, w = region_mask.shape
    x0, y0, x1, y1 = box
    pad = max(12, int(round(max(x1 - x0, y1 - y0) * 0.10)))
    xa, ya = max(0, x0 - pad), max(0, y0 - pad)
    xb, yb = min(w, x1 + pad), min(h, y1 + pad)
    local = np.zeros((h, w), np.uint8)
    local[ya:yb, xa:xb] = 255

    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    hue = hsv[..., 0]
    min_sat = int(getattr(cfg, "paired_diff_saturated_min_saturation", 72))
    min_val = int(getattr(cfg, "paired_diff_saturated_min_value", 160))
    near = cv2.dilate(region_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    seed = (local > 0) & near & (sat >= min_sat) & (val >= min_val)
    hs = hue[seed]
    min_pixels = int(getattr(cfg, "paired_diff_saturated_min_pixels", 180))
    if hs.size < min_pixels:
        return None

    # Quantise hue into 30 bins (6 OpenCV hue units ~= 12 degrees) to survive
    # JPEG/halftone variation without accidentally merging a purple panel with a
    # yellow burst.  Hue distance is circular on [0, 180).
    hist = np.bincount((hs.astype(np.int32) // 6), minlength=30)
    best_bin = int(np.argmax(hist))
    dominant_fraction = float(hist[best_bin] / max(1, hs.size))
    if dominant_fraction < float(getattr(cfg, "paired_diff_saturated_min_dominant_fraction", 0.34)):
        return None
    center = best_bin * 6 + 3
    d1 = (hue.astype(np.int16) - center) % 180
    d2 = (center - hue.astype(np.int16)) % 180
    hue_dist = np.minimum(d1, d2)
    tol = int(getattr(cfg, "paired_diff_saturated_hue_tolerance", 10))
    colour = (
        (local > 0)
        & (hue_dist <= tol)
        & (sat >= max(48, min_sat - 12))
        & (val >= max(130, min_val - 10))
    ).astype(np.uint8) * 255
    close_px = max(1, int(getattr(cfg, "paired_diff_saturated_component_close_px", 4)))
    colour = cv2.morphologyEx(
        colour, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)),
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats((colour > 0).astype(np.uint8), 8)
    rr = region_mask > 0
    best: tuple[float, np.ndarray] | None = None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        comp = labels == i
        overlap = int(np.count_nonzero(comp & rr))
        region_area = max(1, int(np.count_nonzero(rr)))
        overlap_ratio = float(overlap / region_area)
        if overlap < max(12, int(0.015 * area)):
            continue
        if overlap_ratio < float(getattr(cfg, "paired_diff_saturated_min_region_overlap_ratio", 0.15)):
            continue
        score = float(overlap + 0.15 * area)
        if best is None or score > best[0]:
            best = (score, comp)
    if best is None:
        return None
    raw = best[1].astype(np.uint8) * 255
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    filled = np.zeros_like(raw)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    if cv2.countNonZero(filled) < min_pixels:
        return None
    return filled


def _compact_container_ink(
    image: np.ndarray,
    gate: np.ndarray,
    threshold: int,
    cfg: MaskReplaceConfig,
    *,
    gray: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return compact glyph-like dark components inside a coloured container."""
    core = gate.copy()
    erode_px = max(0, int(getattr(cfg, "paired_diff_saturated_core_erode_px", 5)))
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        e = cv2.erode(core, k)
        if cv2.countNonZero(e) >= max(100, int(cv2.countNonZero(core) * 0.58)):
            core = e
    box = _bbox_from_mask(core)
    if box is None:
        return np.zeros_like(core), core
    x0, y0, x1, y1 = box
    gw, gh = max(1, x1 - x0), max(1, y1 - y0)
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    raw = ((gray <= int(threshold)) & (core > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    out = np.zeros_like(raw)
    gate_area = max(1, cv2.countNonZero(core))
    for i in range(1, n):
        _x, _y, cw, ch, area = [int(v) for v in stats[i]]
        if area < 2 or area > max(600, int(0.025 * gate_area)):
            continue
        # Reject star outlines, panel rules and speed lines while keeping CJK
        # radicals/strokes.  Long components are accepted only when their other
        # axis remains narrow enough to be character-like.
        if (cw > 0.32 * gw and ch < 0.045 * gh) or (ch > 0.32 * gh and cw < 0.045 * gw):
            continue
        if cw > 0.45 * gw or ch > 0.45 * gh:
            if not ((ch < 0.75 * gh and cw < 0.12 * gw) or (cw < 0.75 * gw and ch < 0.12 * gh)):
                continue
        aspect = max(cw / max(1.0, ch), ch / max(1.0, cw))
        if aspect > 18.0:
            continue
        out[labels == i] = 255
    return out, core


def _target_edge_distance(gray: np.ndarray, edge_threshold: int = 175) -> np.ndarray:
    edges = cv2.Canny(gray, max(0, edge_threshold // 2), max(1, edge_threshold))
    return cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)


def _expand_safe_write_mask(
    base_mask: np.ndarray,
    safe_envelope: np.ndarray,
    source_image: np.ndarray,
    target_image: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    max_px: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Grow a bubble write mask a few pixels inside a safe target envelope.

    The target-driven path may intentionally inset the write mask to protect the
    outline. When SOURCE/TARGET container interiors differ slightly, that inset can
    clip Chinese edge strokes or leave a narrow ring where Japanese remnants stay
    visible. Grow only a few pixels, only inside ``safe_envelope``, and only away
    from strong target edges.
    """
    m = (base_mask > 0).astype(np.uint8)
    env = (safe_envelope > 0).astype(np.uint8)
    if not bool(getattr(cfg, 'mask_write_gap_fill_enabled', True)):
        return m * 255, {'enabled': False, 'iterations': 0, 'added_pixels': 0}
    if m.shape != env.shape or source_image.shape[:2] != m.shape or target_image.shape[:2] != m.shape:
        return (m * 255), {'enabled': True, 'shape_mismatch': True, 'iterations': 0, 'added_pixels': 0}
    max_px = max(0, int(max_px if max_px is not None else getattr(cfg, 'mask_write_gap_fill_max_px', 3)))
    if max_px <= 0:
        return m * 255, {'enabled': True, 'iterations': 0, 'added_pixels': 0}
    source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY) if source_image.ndim == 3 else source_image.astype(np.uint8)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY) if target_image.ndim == 3 else target_image.astype(np.uint8)
    edge_dist = _target_edge_distance(target_gray)
    if source_image.ndim == 3:
        raw_support = (np.max(source_image, axis=2) > 4).astype(np.uint8)
        # Include black glyph cores that are surrounded by valid white/gray patch
        # pixels, but do not let zero-filled canvas outside a placed patch become
        # eligible merely because target Japanese ink is dark there.
        source_support = cv2.dilate(raw_support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    else:
        source_support = np.ones_like(m, dtype=bool)
    src_white_thr = int(getattr(cfg, 'mask_write_gap_fill_source_white_threshold', 238))
    src_dark_thr = int(getattr(cfg, 'mask_write_gap_fill_source_dark_threshold', 205))
    tgt_dark_thr = int(getattr(cfg, 'mask_write_gap_fill_target_dark_threshold', 185))
    edge_floor = float(getattr(cfg, 'mask_write_gap_fill_target_edge_distance_px', 1.35))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = m.copy()
    before = int(np.count_nonzero(out))
    iterations = 0
    for _ in range(max_px):
        dil = cv2.dilate(out, kernel)
        ring = (dil > 0) & (out == 0) & (env > 0)
        if not np.any(ring):
            break
        edge_safe = (edge_dist >= edge_floor) | ((target_gray <= tgt_dark_thr) & (env > 0))
        allow = ring & source_support & edge_safe & (
            (source_gray >= src_white_thr) | (source_gray <= src_dark_thr) | (target_gray <= tgt_dark_thr)
        )
        added = int(np.count_nonzero(allow))
        if added == 0:
            break
        out[allow] = 1
        iterations += 1
    return (out * 255), {
        'enabled': True,
        'iterations': int(iterations),
        'added_pixels': int(max(0, int(np.count_nonzero(out)) - before)),
        'edge_floor_px': float(edge_floor),
    }


def _evaluate_content_completeness(
    rec: MaskTransferRecord,
    source_ink: np.ndarray | None,
    target_ink: np.ndarray | None,
    final_image: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    tolerance_px: int | None = None,
    min_source_coverage: float | None = None,
    max_target_residual: float | None = None,
) -> None:
    """Verify content independently from the fact that a raster write occurred.

    The check is deliberately language/OCR independent.  It asks two local
    questions: (1) did the expected registered source ink survive in the final
    pixels, and (2) did target-only ink disappear?  Overlapping source/target
    strokes are excluded from the residual test so legitimate Chinese ink is not
    mistaken for leftover Japanese.
    """
    if not bool(getattr(cfg, "content_completeness_enabled", True)):
        rec.content_check = "disabled"
        return
    if source_ink is None or source_ink.shape != final_image.shape[:2]:
        rec.content_check = "insufficient_source_ink_evidence"
        return
    src = (source_ink > 0).astype(np.uint8) * 255
    src_count = int(cv2.countNonZero(src))
    min_ink = int(getattr(cfg, "content_completeness_min_ink_pixels", 18))
    if src_count < min_ink:
        rec.content_check = "insufficient_source_ink_evidence"
        return
    tol = max(1, int(tolerance_px if tolerance_px is not None else getattr(cfg, "content_completeness_tolerance_px", 2)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))
    fgray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    final_ink = (fgray <= 220).astype(np.uint8) * 255
    final_near = cv2.dilate(final_ink, k)
    coverage = float(np.count_nonzero((src > 0) & (final_near > 0)) / max(1, src_count))

    residual = 0.0
    tgt_count = 0
    if target_ink is not None and target_ink.shape == src.shape:
        tgt = (target_ink > 0).astype(np.uint8) * 255
        tgt_count = int(cv2.countNonZero(tgt))
        if tgt_count >= min_ink:
            src_near = cv2.dilate(src, k)
            target_only = (tgt > 0) & (src_near == 0)
            target_only_count = int(np.count_nonzero(target_only))
            if target_only_count >= max(4, min_ink // 3):
                # A slightly stricter darkness threshold catches real leftover
                # glyph cores while ignoring mild inpaint/halftone texture.
                residual = float(np.count_nonzero(target_only & (fgray <= 205)) / target_only_count)
            else:
                residual = 0.0

    rec.source_ink_coverage = coverage
    rec.target_residual_ratio = residual
    rec.content_check = "checked" if tgt_count >= min_ink else "checked_source_only"
    min_cov = float(min_source_coverage if min_source_coverage is not None else getattr(cfg, "content_completeness_min_source_coverage", 0.90))
    max_res = float(max_target_residual if max_target_residual is not None else getattr(cfg, "content_completeness_max_target_residual", 0.10))
    rec.content_complete = bool(coverage >= min_cov and residual <= max_res)


def _repair_content_region(
    rec: MaskTransferRecord,
    rendered: np.ndarray,
    source_image: np.ndarray,
    target_original: np.ndarray,
    current_write_mask: np.ndarray,
    safe_envelope: np.ndarray,
    source_ink: np.ndarray | None,
    target_ink: np.ndarray | None,
    cfg: MaskReplaceConfig,
    *,
    tolerance_px: int | None = None,
    min_source_coverage: float | None = None,
    max_target_residual: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Attempt one bounded OCR-free completeness repair inside a trusted region.

    The repair is intentionally conservative: it can only grow inside the
    existing safe target envelope, it refuses strong target edges, and it clears
    only compact target-only ink that is not near expected source ink. The result
    is re-audited before it can be considered successful.
    """
    if not bool(getattr(cfg, "content_auto_repair_enabled", True)):
        return rendered, current_write_mask, {"enabled": False}
    rec.repair_attempted = True
    before_cov = float(getattr(rec, "source_ink_coverage", 0.0))
    before_res = float(getattr(rec, "target_residual_ratio", 0.0))
    before_check = str(getattr(rec, "content_check", "not_checked"))
    before_complete = bool(getattr(rec, "content_complete", False))
    out = rendered.copy()
    repair_mask, growth = _expand_safe_write_mask(
        current_write_mask, safe_envelope, source_image, target_original, cfg,
        max_px=int(getattr(cfg, "content_auto_repair_max_growth_px", 5)),
    )
    new_pixels = (repair_mask > 0) & (current_write_mask == 0)
    if np.any(new_pixels):
        out[new_pixels] = source_image[new_pixels]

    residual_clear = np.zeros_like(repair_mask)
    if source_ink is not None and target_ink is not None and source_ink.shape == repair_mask.shape and target_ink.shape == repair_mask.shape:
        tol = max(1, int(tolerance_px if tolerance_px is not None else getattr(cfg, "content_completeness_tolerance_px", 2)))
        src_near = cv2.dilate((source_ink > 0).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))) > 0
        target_only = (target_ink > 0) & (~src_near) & (safe_envelope > 0)
        if np.any(target_only):
            # ``target_ink`` is already compact-component filtered and the safe
            # envelope excludes the bubble border. Do not reject it for being a
            # strong edge: Japanese glyph cores are strong edges by definition.
            residual_clear[target_only] = 255
            grow = max(0, int(getattr(cfg, "content_auto_repair_residual_dilate_px", 1)))
            if grow > 0 and cv2.countNonZero(residual_clear) > 0:
                residual_clear = cv2.dilate(residual_clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1)))
                residual_clear[safe_envelope == 0] = 0
            if cv2.countNonZero(residual_clear) > 0:
                out = cv2.inpaint(out, residual_clear, float(getattr(cfg, "content_auto_repair_inpaint_radius", 2.5)), cv2.INPAINT_TELEA)
                # Restore expected source content wherever the expanded write mask allows it.
                repaint = (repair_mask > 0) & (source_ink > 0)
                if np.any(repaint):
                    out[repaint] = source_image[repaint]

    _evaluate_content_completeness(
        rec, source_ink, target_ink, out, cfg,
        tolerance_px=tolerance_px,
        min_source_coverage=min_source_coverage,
        max_target_residual=max_target_residual,
    )
    if cv2.countNonZero(residual_clear) > 0:
        repair_mask = np.maximum(repair_mask, residual_clear)
    after_cov = float(getattr(rec, "source_ink_coverage", 0.0))
    after_res = float(getattr(rec, "target_residual_ratio", 0.0))
    gain = (after_cov - before_cov) + (before_res - after_res)
    min_gain = float(getattr(cfg, "content_auto_repair_min_gain", 0.01))
    improved = bool(rec.content_complete or gain >= min_gain)
    rec.repair_succeeded = bool(improved)
    if not improved:
        rec.source_ink_coverage = before_cov
        rec.target_residual_ratio = before_res
        rec.content_check = before_check
        rec.content_complete = before_complete
    return out if improved else rendered, repair_mask if improved else current_write_mask, {
        "enabled": True,
        "growth": growth,
        "residual_clear_pixels": int(cv2.countNonZero(residual_clear)),
        "before_source_coverage": before_cov,
        "after_source_coverage": after_cov,
        "before_target_residual": before_res,
        "after_target_residual": after_res,
        "gain": float(gain),
        "improved": bool(improved),
        "content_complete": bool(rec.content_complete),
    }


def finalize_transfer_records(records: list[MaskTransferRecord], cfg: MaskReplaceConfig) -> None:
    """Assign one auditable SAFE/REVIEW/REJECT state to every transfer record."""
    safe_conf = float(getattr(cfg, "triage_safe_confidence", 0.82))
    reject_conf = float(getattr(cfg, "triage_reject_confidence", 0.55))
    for rec in records:
        if not bool(getattr(rec, "applied", False)):
            rec.triage_state = "REJECT"
            continue
        if not _publication_safety_enabled(cfg):
            # Diagnostics remain on the record, but they no longer block or
            # downgrade a successfully written region.
            rec.triage_state = "SAFE"
            continue
        if float(getattr(rec, "confidence", 0.0)) < reject_conf:
            rec.triage_state = "REJECT"
            continue
        check = str(getattr(rec, "content_check", "not_checked") or "not_checked")
        verified = check.startswith("checked") and bool(getattr(rec, "content_complete", False))
        if verified and float(getattr(rec, "confidence", 0.0)) >= safe_conf and not bool(getattr(rec, "review_required", False)):
            rec.triage_state = "SAFE"
        else:
            rec.triage_state = "REVIEW"


def _transfer_saturated_text_container(
    aligned_source: np.ndarray,
    target: np.ndarray,
    colour_gate: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict]:
    """Clean a flat-colour burst and rebuild only the source Chinese glyph ink."""
    shape = target.shape[:2]
    empty = np.zeros(shape, np.uint8)
    thr = int(getattr(cfg, "paired_diff_saturated_text_dark_threshold", 182))
    src_sel, core = _compact_container_ink(aligned_source, colour_gate, thr, cfg)
    tgt_sel, _ = _compact_container_ink(target, colour_gate, thr, cfg)
    src_count = int(cv2.countNonZero(src_sel))
    tgt_count = int(cv2.countNonZero(tgt_sel))
    if src_count < int(getattr(cfg, "paired_diff_complex_min_source_ink_pixels", 16)):
        return None, empty, empty, {"reason": "insufficient_saturated_source_ink", "source_ink_pixels": src_count, "target_ink_pixels": tgt_count}
    if tgt_count < int(getattr(cfg, "paired_diff_complex_min_target_ink_pixels", 12)):
        return None, empty, empty, {"reason": "insufficient_saturated_target_ink", "source_ink_pixels": src_count, "target_ink_pixels": tgt_count}

    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    clear = tgt_sel.copy()
    # v1.3: the compact target mask identifies glyph cores, but JPEG/print
    # antialiasing on saturated fills can be much brighter than the core threshold.
    # Admit only low-saturation darker pixels immediately around verified cores.
    aa_added = 0
    aa_r = max(0, int(getattr(cfg, "paired_diff_saturated_antialias_expand_px", 2)))
    if aa_r > 0 and cv2.countNonZero(clear) > 0:
        halo = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (aa_r * 2 + 1, aa_r * 2 + 1))) > 0
        bg_probe = (core > 0) & (clear == 0) & (hsv[..., 1] >= max(40, int(getattr(cfg, "paired_diff_saturated_min_saturation", 72)) - 18))
        bg_gray = float(np.median(gray[bg_probe])) if int(np.count_nonzero(bg_probe)) >= 32 else float(np.median(gray[core > 0]))
        contrast = max(4, int(getattr(cfg, "paired_diff_saturated_antialias_contrast", 8)))
        sat_max = int(getattr(cfg, "paired_diff_saturated_antialias_max_saturation", 96))
        fringe = halo & (core > 0) & (clear == 0) & (gray <= bg_gray - contrast) & (hsv[..., 1] <= sat_max)
        aa_added = int(np.count_nonzero(fringe))
        clear[fringe] = 255
    d = max(1, int(getattr(cfg, "paired_diff_saturated_clear_dilate_px", 2)))
    clear = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1)))
    clear = cv2.bitwise_and(clear, core)

    bg_sel = (
        (colour_gate > 0)
        & (hsv[..., 1] >= max(40, int(getattr(cfg, "paired_diff_saturated_min_saturation", 72)) - 18))
        & (hsv[..., 2] >= int(getattr(cfg, "paired_diff_saturated_min_value", 160)))
        & (gray > thr + 18)
    )
    if np.count_nonzero(bg_sel) >= 80:
        paper_bgr = np.median(target[bg_sel], axis=0).astype(np.float32)
    else:
        paper_bgr = np.median(target[colour_gate > 0], axis=0).astype(np.float32)

    out = target.copy()
    # Flat bursts can be restored exactly with the median fill. Gradients/halftone
    # need local structure, so prefer Telea there instead of painting a flat patch.
    bg_pixels = target[bg_sel] if np.any(bg_sel) else target[colour_gate > 0]
    bg_std = float(np.mean(np.std(bg_pixels.astype(np.float32), axis=0))) if bg_pixels.size else 999.0
    flat_limit = float(getattr(cfg, "paired_diff_saturated_flat_std_threshold", 10.0))
    if bg_std <= flat_limit:
        out[clear > 0] = np.clip(paper_bgr, 0, 255).astype(np.uint8)
        clear_backend = "flat-median"
    else:
        out = cv2.inpaint(target, clear, float(getattr(cfg, "paired_diff_saturated_inpaint_radius", 3.0)), cv2.INPAINT_TELEA)
        clear_backend = "opencv-telea"
    alpha = _soft_ink_alpha(aligned_source, src_sel, core)
    # Saturated-container source pages in this route are monochrome scans.  Use
    # the source raster *opacity/topology* but neutral black ink; never composite
    # the source white background/halftone into the colour master.
    a3 = alpha[..., None]
    out = np.clip(out.astype(np.float32) * (1.0 - a3), 0, 255).astype(np.uint8)
    write = np.maximum(clear, (alpha * 255).astype(np.uint8))
    changed = np.any(out != target, axis=2)
    write[changed] = 255
    return out, write, src_sel, {
        "reason": "ok_saturated_container",
        "source_ink_pixels": src_count,
        "target_ink_pixels": tgt_count,
        "clear_pixels": int(cv2.countNonZero(clear)),
        "antialias_added_pixels": int(aa_added),
        "clear_backend": clear_backend,
        "background_std": bg_std,
        "boundary_touch": False,
        "clear_mask": clear,
        "target_ink_mask": tgt_sel,
        "container_mask": colour_gate,
    }


def _transfer_open_complex_text_region(
    aligned_source: np.ndarray,
    target: np.ndarray,
    region_mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict]:
    """Erase only Japanese glyph strokes and composite the registered Chinese ink.

    No rectangular crop, OCR re-typesetting or whole coloured-background fill is
    used.  The source/target ink maps select changed glyph groups, Japanese groups
    are inpainted with a 1-2px fringe, then the source raster glyphs are composited
    at their registered location.  This is the safe path for burst balloons, open
    captions and text on artwork.
    """
    shape = target.shape[:2]
    empty = np.zeros(shape, np.uint8)
    box = _bbox_from_mask(region_mask)
    if box is None:
        return None, empty, empty, {"reason": "empty_region"}
    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    pad = max(6, int(round(max(bw, bh) * float(getattr(cfg, "paired_diff_complex_region_pad_ratio", 0.16)))))
    gate = cv2.dilate(
        region_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)),
    )
    # Keep the dilation bounded around the actual candidate; very large free-text
    # islands otherwise risk touching unrelated speech bubbles across a gutter.
    rect = np.zeros(shape, np.uint8)
    xa, ya = max(0, x0 - pad), max(0, y0 - pad)
    xb, yb = min(shape[1], x1 + pad), min(shape[0], y1 + pad)
    rect[ya:yb, xa:xb] = 255
    gate = cv2.bitwise_and(gate, rect)

    src_ink = _complex_text_ink_map(aligned_source)
    tgt_ink = _complex_text_ink_map(target)
    tol = max(1, int(getattr(cfg, "paired_diff_ink_tolerance_px", 2)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))
    src_near = cv2.dilate(src_ink, k)
    tgt_near = cv2.dilate(tgt_ink, k)
    src_unique = cv2.bitwise_and(src_ink, cv2.bitwise_not(tgt_near))
    tgt_unique = cv2.bitwise_and(tgt_ink, cv2.bitwise_not(src_near))
    gap = int(getattr(cfg, "paired_diff_complex_group_gap_px", 5))
    src_sel = _select_changed_text_components(src_ink, src_unique, gate, gap)
    tgt_sel = _select_changed_text_components(tgt_ink, tgt_unique, gate, gap)
    src_count = int(cv2.countNonZero(src_sel)); tgt_count = int(cv2.countNonZero(tgt_sel))
    # Cross-rendition scans can move/blur a glyph enough that the strict
    # edition-exclusive seed disappears, even though compact source/target
    # lettering is visibly present inside the trusted local region. Recover
    # those groups from the local ink map as a bounded fallback; never expand
    # beyond ``gate`` or copy a rectangular coloured background.
    if src_count < int(getattr(cfg, "paired_diff_complex_min_source_ink_pixels", 16)):
        src_sel = _select_changed_text_components(src_ink, src_ink, gate, gap)
        src_count = int(cv2.countNonZero(src_sel))
    if tgt_count < int(getattr(cfg, "paired_diff_complex_min_target_ink_pixels", 12)):
        tgt_sel = _select_changed_text_components(tgt_ink, tgt_ink, gate, gap)
        tgt_count = int(cv2.countNonZero(tgt_sel))
    if src_count < int(getattr(cfg, "paired_diff_complex_min_source_ink_pixels", 16)):
        return None, empty, empty, {"reason": "insufficient_source_ink", "source_ink_pixels": src_count, "target_ink_pixels": tgt_count}
    # A trusted structural region may contain Chinese source lettering where
    # the HD target has no separable Japanese component (for example a glyph
    # lost in a bright/halftone edge). In that case there is nothing to erase;
    # the bounded source ink can still be written safely.
    target_missing_but_source_present = (
        tgt_count < int(getattr(cfg, "paired_diff_complex_min_target_ink_pixels", 12))
        and src_count >= int(getattr(cfg, "paired_diff_complex_min_source_ink_pixels", 16))
    )
    if tgt_count < int(getattr(cfg, "paired_diff_complex_min_target_ink_pixels", 12)) and not target_missing_but_source_present:
        return None, empty, empty, {"reason": "insufficient_target_ink", "source_ink_pixels": src_count, "target_ink_pixels": tgt_count}

    clear = tgt_sel.copy()
    d = max(1, int(getattr(cfg, "paired_diff_complex_clear_dilate_px", 2)))
    clear = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1)))
    clear = cv2.bitwise_and(clear, gate)
    out = cv2.inpaint(target, clear, 2.5, cv2.INPAINT_TELEA)

    alpha = _soft_ink_alpha(aligned_source, src_sel, gate)
    a3 = alpha[..., None]
    out = np.clip(
        aligned_source.astype(np.float32) * a3 + out.astype(np.float32) * (1.0 - a3),
        0, 255,
    ).astype(np.uint8)
    write = np.maximum(clear, (alpha * 255).astype(np.uint8))
    # Float alpha can change a pixel by one code value even when uint8(alpha*255)
    # rounds to zero.  Audit/export masks must describe the *exact* write footprint
    # so pixels outside it are guaranteed byte-identical to the target.
    changed = np.any(out != target, axis=2)
    write[changed] = 255

    # Candidate-boundary contact is not a hard drop: keep a reversible preview so
    # the page cannot silently retain Japanese, but make the review requirement
    # explicit. The expanded gate normally prevents this in complete source pages.
    edge_band = np.zeros(shape, np.uint8)
    edge_band[ya:yb, xa:xb] = 255
    inner = cv2.erode(edge_band, np.ones((3, 3), np.uint8))
    boundary_touch = bool(np.count_nonzero((src_sel > 0) & (edge_band > inner)))
    return out, write, src_sel, {
        "reason": "ok",
        "source_ink_pixels": src_count,
        "target_ink_pixels": tgt_count,
        "clear_pixels": int(cv2.countNonZero(clear)),
        "boundary_touch": boundary_touch,
        "clear_mask": clear,
        "target_ink_mask": tgt_sel,
    }


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


def transfer_ocr_guided_text_units(
    aligned_source: np.ndarray,
    target: np.ndarray,
    source_units: list[TextUnit],
    target_units: list[TextUnit],
    matches: list[UnitMatch],
    registration: RegistrationResult,
    cfg: MaskReplaceConfig | None = None,
    *,
    exclude_mask: np.ndarray | None = None,
) -> MaskTransferResult:
    """Recover OCR-confirmed text regions without OCR re-lettering.

    This is a completeness fallback for regions that closed-balloon/paired-diff
    geometry missed (structured boxes, burst balloons and open text).  OCR only
    provides the correspondence and polygon gate.  Final pixels come exclusively
    from the registered source raster; target Japanese glyph components are
    inpainted individually.  Low-confidence matches remain reversible review
    candidates instead of being silently discarded.
    """
    cfg = cfg or MaskReplaceConfig()
    shape = target.shape[:2]
    if aligned_source.shape[:2] != shape:
        raise ValueError("aligned_source must be in target coordinates")
    if exclude_mask is None or exclude_mask.shape != shape:
        exclude_mask = np.zeros(shape, np.uint8)

    rendered = target.copy()
    layer = np.zeros((shape[0], shape[1], 4), np.uint8)
    composite = np.zeros(shape, np.uint8)
    clear_all = np.zeros(shape, np.uint8)
    records: list[MaskTransferRecord] = []
    patch_matches: list[BubblePatchMatch] = []
    src = {u.id: u for u in source_units}
    dst = {u.id: u for u in target_units}
    candidate_floor = float(getattr(cfg, "ocr_guided_candidate_min_match_confidence", 0.42))
    auto_floor = float(getattr(cfg, "ocr_guided_auto_apply_min_match_confidence", 0.64))
    ocr_floor = float(getattr(cfg, "ocr_guided_min_ocr_confidence", 0.45))
    max_overlap = float(getattr(cfg, "ocr_guided_max_existing_overlap", 0.18))

    for match in matches:
        if match.relation != "one_to_one" or float(match.confidence) < candidate_floor:
            continue
        su = src.get(match.source_unit_id); tu = dst.get(match.target_unit_id)
        if su is None or tu is None or not str(su.text or "").strip():
            continue
        if min(float(su.confidence), float(tu.confidence)) < ocr_floor:
            continue
        gate = _ocr_guided_region_gate(su, tu, registration, shape, cfg)
        if gate is None or cv2.countNonZero(gate) == 0:
            continue
        overlap = int(np.count_nonzero((gate > 0) & (exclude_mask > 0))) / max(1, cv2.countNonZero(gate))
        if overlap > max_overlap:
            continue

        source_id = str(su.bubble_id or su.id)
        target_id = str(tu.bubble_id or tu.id)
        confidence = float(min(match.confidence, su.confidence, tu.confidence))
        rec = MaskTransferRecord(source_id, target_id, confidence, False, "not_applied")
        rec.geometry_mode = "ocr_guided_components"
        rec.source_bbox = tuple(int(round(v)) for v in su.bbox)
        rec.target_bbox = tuple(int(round(v)) for v in tu.bbox)
        patch_matches.append(BubblePatchMatch(
            source_id, target_id, confidence, float(match.cost), 0.0, 0.0, 1.0,
            ["ocr-geometry-only", "registered-raster-ink"],
        ))

        text_img, write_mask, source_ink_mask, diag = _transfer_open_complex_text_region(
            aligned_source, rendered, gate, cfg
        )
        rec.sr_backend = "ocr-guided-components"
        rec.sr_scale = 1.0
        rec.mask_iou = 1.0
        rec.spill_ratio = 0.0
        rec.ink_ratio = float(cv2.countNonZero(source_ink_mask) / max(1, cv2.countNonZero(gate)))
        if text_img is None:
            # A geometrically credible match with insufficient raster evidence is
            # intentionally visible to QA instead of disappearing from the run.
            rec.reason = str(diag.get("reason") or "ocr_guided_component_transfer_failed")
            rec.review_required = True
            rec.review_reason = rec.reason
            rec.restorable = True
            rec.editable = True
            records.append(rec)
            continue

        rendered = text_img
        composite = np.maximum(composite, write_mask)
        diag_clear = diag.get("clear_mask")
        if isinstance(diag_clear, np.ndarray) and diag_clear.shape == shape:
            clear_all = np.maximum(clear_all, diag_clear)
        use = write_mask > 0
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        layer[use, :3] = rgb[use]
        layer[..., 3] = np.maximum(layer[..., 3], write_mask)
        rec.applied = True
        rec.target_coverage = 1.0
        rec.clarity_mode = "ocr-guided-ink-transfer"
        _evaluate_content_completeness(
            rec, source_ink_mask, diag.get("target_ink_mask"), rendered, cfg
        )
        low = confidence < auto_floor
        boundary_touch = bool(diag.get("boundary_touch", False))
        if low or boundary_touch:
            rec.reason = "applied_ocr_guided_review_candidate"
            rec.candidate = True
            rec.review_required = True
            rec.review_reason = (
                "source_text_cluster_touches_candidate_boundary" if boundary_touch
                else "low_confidence_ocr_geometry_match"
            )
            rec.restorable = True
            rec.editable = True
        else:
            rec.reason = "applied_ocr_guided_registered_components"
        records.append(rec)
        exclude_mask = np.maximum(exclude_mask, write_mask)

    return MaskTransferResult(rendered, layer, composite, patch_matches, records, clear_all)


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


def _rigid_source_raster(
    source: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    target_shape: tuple[int, int],
    base_scale: float,
    cfg: MaskReplaceConfig,
    *,
    source_gray: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float] | None:
    """Place the complete source lettering raster with uniform scale only.

    Returns ``(soft_alpha, ink_mask, scale, dx, dy, ink_coverage, mask_containment)``
    in target coordinates.  ``soft_alpha`` is derived from the original source
    grayscale raster as one field; characters are never split/reassembled.
    """
    sbox = _bbox_from_mask(source_mask); tbox = _bbox_from_mask(target_mask)
    if sbox is None or tbox is None:
        return None
    sx0, sy0, sx1, sy1 = sbox
    src_mask = source_mask.copy()
    sinset = max(0, int(getattr(cfg, "rigid_container_source_inset_px", 1)))
    if sinset:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sinset * 2 + 1, sinset * 2 + 1))
        er = cv2.erode(src_mask, k)
        if cv2.countNonZero(er) > 0:
            src_mask = er
    target_inner = target_mask.copy()
    tinset = max(0, int(getattr(cfg, "rigid_container_target_inset_px", 1)))
    if tinset:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tinset * 2 + 1, tinset * 2 + 1))
        er = cv2.erode(target_inner, k)
        if cv2.countNonZero(er) > 0:
            target_inner = er

    gray = source_gray if source_gray is not None else cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)
    crop_gray = gray[sy0:sy1, sx0:sx1]
    crop_mask = src_mask[sy0:sy1, sx0:sx1]
    vals = crop_gray[crop_mask > 0]
    if vals.size < 8:
        return None
    paper = float(np.percentile(vals, float(getattr(cfg, "rigid_container_paper_percentile", 90.0))))
    if paper < 180.0:
        return None
    # Normalise scan paper to white and derive one continuous opacity field.  On
    # white this is exactly equivalent to the normalised source grayscale raster,
    # including antialiasing/halftone edge values; no glyph component logic occurs.
    alpha = np.clip((paper - crop_gray) / max(1.0, paper), 0.0, 1.0).astype(np.float32)
    floor = float(np.clip(getattr(cfg, "rigid_container_alpha_floor", 0.055), 0.0, 0.30))
    alpha[alpha < floor] = 0.0
    alpha *= (crop_mask.astype(np.float32) / 255.0)
    if np.count_nonzero(alpha >= max(0.08, floor)) < int(getattr(cfg, "content_completeness_min_ink_pixels", 18)):
        return None

    scx, scy = _centroid(src_mask); tcx, tcy = _centroid(target_inner)
    rel_cx, rel_cy = scx - sx0, scy - sy0
    min_factor = float(np.clip(getattr(cfg, "rigid_container_min_scale_factor", 0.94), 0.80, 1.0))
    step = float(np.clip(getattr(cfg, "rigid_container_scale_step", 0.01), 0.005, 0.05))
    factors = []
    f = 1.0
    while f >= min_factor - 1e-8:
        factors.append(f); f -= step
    max_shift = max(0, int(getattr(cfg, "rigid_container_max_shift_px", 7)))
    min_ink_cov = float(np.clip(getattr(cfg, "rigid_container_min_ink_coverage", 0.985), 0.90, 1.0))
    min_mask_contain = float(np.clip(getattr(cfg, "rigid_container_min_mask_containment", 0.955), 0.90, 1.0))
    offset_patience = max(6, int(getattr(cfg, "rigid_container_offset_patience", 24)))
    offsets = _ordered_offsets(max_shift)
    th, tw = target_shape
    best = None
    for factor in factors:
        scale = float(base_scale * factor)
        nw = max(1, int(round((sx1 - sx0) * scale))); nh = max(1, int(round((sy1 - sy0) * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        ra = cv2.resize(alpha, (nw, nh), interpolation=interp)
        rm = cv2.resize(crop_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        bx = int(round(tcx - rel_cx * scale)); by = int(round(tcy - rel_cy * scale))
        tested = 0
        for dx, dy in offsets:
            tested += 1
            px, py = bx + dx, by + dy
            xa, ya = max(0, px), max(0, py); xb, yb = min(tw, px + nw), min(th, py + nh)
            if xb <= xa or yb <= ya:
                continue
            aa = ra[ya - py:yb - py, xa - px:xb - px]
            mm = rm[ya - py:yb - py, xa - px:xb - px] > 0
            ink = aa >= max(0.08, floor)
            ink_n = int(np.count_nonzero(ink)); mask_n = int(np.count_nonzero(mm))
            if ink_n < int(getattr(cfg, "content_completeness_min_ink_pixels", 18)) or mask_n <= 0:
                continue
            tg = target_inner[ya:yb, xa:xb] > 0
            outer = target_mask[ya:yb, xa:xb] > 0
            ink_cov = float(np.count_nonzero(ink & tg) / ink_n)
            mask_cov = float(np.count_nonzero(mm & outer) / mask_n)
            # Prefer complete ink first, then source-container containment,
            # then the smallest possible nudge/shrink from paired geometry.
            score = 4.0 * ink_cov + mask_cov - 0.002 * (abs(dx) + abs(dy)) - 0.08 * (1.0 - factor)
            if best is None or score > best[0]:
                best = (score, ink_cov, mask_cov, scale, px, py, ra)
            if ink_cov >= min_ink_cov and mask_cov >= min_mask_contain and tested >= min(4, offset_patience):
                break
            if tested >= offset_patience and best is not None and best[1] >= min_ink_cov and best[2] >= min_mask_contain:
                break
        if best is not None and best[1] >= min_ink_cov and best[2] >= min_mask_contain:
            break
    if best is None or best[1] < min_ink_cov or best[2] < min_mask_contain:
        return None
    _, ink_cov, mask_cov, scale, px, py, ra = best
    full_alpha = np.zeros((th, tw), np.float32)
    nh, nw = ra.shape
    xa, ya = max(0, px), max(0, py); xb, yb = min(tw, px + nw), min(th, py + nh)
    if xb <= xa or yb <= ya:
        return None
    local = ra[ya - py:yb - py, xa - px:xb - px].copy()
    local *= (target_mask[ya:yb, xa:xb].astype(np.float32) / 255.0)
    full_alpha[ya:yb, xa:xb] = np.maximum(full_alpha[ya:yb, xa:xb], local)
    ink_mask = (full_alpha >= max(0.08, floor)).astype(np.uint8) * 255
    # dx/dy are reported relative to the pure centroid placement at the selected scale.
    base_x = tcx - rel_cx * scale; base_y = tcy - rel_cy * scale
    return full_alpha, ink_mask, scale, float(px - base_x), float(py - base_y), float(ink_cov), float(mask_cov)


def _rigid_target_write_envelope(target_mask: np.ndarray, cfg: MaskReplaceConfig) -> np.ndarray:
    """Return the only writable part of a rigid target container.

    The container outline belongs to TARGET geometry.  Earlier versions eroded
    the patch once, but the later gap-fill stage used the full target mask as its
    safe envelope and could grow back over the protected outline.  Keep one
    canonical envelope so clear, patch, and gap-fill all obey the same border
    contract.
    """
    env = (target_mask > 0).astype(np.uint8) * 255
    if not bool(getattr(cfg, "rigid_container_full_patch_preserve_target_border", True)):
        return env
    inset = max(
        0,
        int(getattr(cfg, "rigid_container_target_inset_px", 1)),
        int(getattr(cfg, "rigid_container_full_patch_target_inset_px", 1)),
    )
    if inset <= 0:
        return env
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1))
    er = cv2.erode(env, k)
    return er if cv2.countNonZero(er) > 0 else env


def _rigid_container_full_patch(
    source: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    target_shape: tuple[int, int],
    scale: float,
    dx: float,
    dy: float,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Render the whole source container interior into target coordinates.

    Unlike alpha-only lettering compositing, this copies the source white paper
    + Chinese text as one locked patch.  For same-layout editions this is the
    most faithful route: if the container shapes match, Japanese cannot leak and
    Chinese glyphs cannot be partially clipped by target-side text geometry.
    """
    sbox = _bbox_from_mask(source_mask)
    if sbox is None:
        return None
    sx0, sy0, sx1, sy1 = sbox
    crop_img = source[sy0:sy1, sx0:sx1]
    crop_mask = source_mask[sy0:sy1, sx0:sx1]
    if crop_img.size == 0 or cv2.countNonZero(crop_mask) <= 0:
        return None

    scx, scy = _centroid(source_mask)
    tcx, tcy = _centroid(target_mask)
    rel_cx, rel_cy = scx - sx0, scy - sy0
    px = int(round((tcx - rel_cx * scale) + dx))
    py = int(round((tcy - rel_cy * scale) + dy))
    nw = max(1, int(round((sx1 - sx0) * scale)))
    nh = max(1, int(round((sy1 - sy0) * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized_img = cv2.resize(crop_img, (nw, nh), interpolation=interp)
    resized_mask = cv2.resize(crop_mask, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # One canonical border-safe envelope is shared by patch placement and the
    # later gap-fill stage.  This prevents source box outlines from darkening a
    # preserved TARGET outline after the initial inset.
    write_mask = _rigid_target_write_envelope(target_mask, cfg)

    th, tw = target_shape
    full_rgb = np.zeros((th, tw, 3), np.uint8)
    full_alpha = np.zeros((th, tw), np.float32)
    xa, ya = max(0, px), max(0, py)
    xb, yb = min(tw, px + nw), min(th, py + nh)
    if xb <= xa or yb <= ya:
        return None
    local_img = resized_img[ya - py:yb - py, xa - px:xb - px]
    local_mask = resized_mask[ya - py:yb - py, xa - px:xb - px].astype(np.float32) / 255.0
    local_mask *= (write_mask[ya:yb, xa:xb].astype(np.float32) / 255.0)
    blur_px = max(0, int(getattr(cfg, 'rigid_container_full_patch_mask_blur_px', 1)))
    if blur_px > 0:
        k = blur_px * 2 + 1
        local_mask = cv2.GaussianBlur(local_mask, (k, k), 0)
        local_mask = np.clip(local_mask, 0.0, 1.0)
        local_mask *= (write_mask[ya:yb, xa:xb].astype(np.float32) / 255.0)
    full_rgb[ya:yb, xa:xb] = local_img
    full_alpha[ya:yb, xa:xb] = local_mask
    return full_rgb, (full_alpha * 255.0).astype(np.uint8)


def _compact_target_glyph_fringe(
    target: np.ndarray,
    target_mask: np.ndarray,
    *,
    target_gray: np.ndarray | None = None,
) -> np.ndarray:
    """Recover only character-sized dark components straddling a partial mask.

    Structural changed-text masks sometimes cut through the last Japanese column.
    Expanding the whole white region is unsafe (tails can connect to panel paper),
    so this helper adds only compact dark components that physically touch the
    trusted mask. Long balloon outlines, gutters and panel rules are rejected.
    """
    box = _bbox_from_mask(target_mask)
    out = np.zeros_like(target_mask)
    if box is None:
        return out
    x0, y0, x1, y1 = box; bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    gray = target_gray if target_gray is not None else cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    pad = max(4, int(round(0.12 * max(bw, bh))))
    near = cv2.dilate(target_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))) > 0
    dark = ((gray < 190) & near).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    touch = cv2.dilate(target_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    mask_area = max(1, cv2.countNonZero(target_mask))
    max_w = max(8, int(round(0.20 * bw))); max_h = max(8, int(round(0.20 * bh)))
    max_area = max(80, int(round(0.035 * mask_area)))
    for i in range(1, n):
        _, _, cw, ch, area = map(int, stats[i])
        if area < 4 or area > max_area or cw > max_w or ch > max_h:
            continue
        comp = labels == i
        if not np.any(comp & touch):
            continue
        # Only a straddling/near-edge component is useful here. Components fully
        # inside target_mask are already erased by the whole-container clear.
        outside = comp & (target_mask == 0)
        if not np.any(outside):
            continue
        cm = comp.astype(np.uint8) * 255
        cm = cv2.dilate(cm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        out = np.maximum(out, cm)
    return out


def transfer_rigid_container_rasters(
    source: np.ndarray,
    target_reference: np.ndarray,
    base_image: np.ndarray,
    source_bubbles: list[BubbleInstance],
    target_bubbles: list[BubbleInstance],
    cfg: MaskReplaceConfig | None = None,
) -> MaskTransferResult:
    """Transfer safe white speech/text containers without distorting glyphs.

    This is the preferred path for same-layout B/W Chinese -> colour Japanese
    pairs.  It deliberately works from the *original* source page rather than an
    affine/dense-flow warped source.  Pair geometry supplies one uniform local
    scale and translation; only source lettering opacity is moved. TARGET remains
    the sole background/colour authority.
    """
    cfg = cfg or MaskReplaceConfig()
    shape = target_reference.shape[:2]
    empty_layer = np.zeros((shape[0], shape[1], 4), np.uint8)
    empty_mask = np.zeros(shape, np.uint8)
    if not bool(getattr(cfg, "rigid_container_transfer_enabled", True)):
        return MaskTransferResult(base_image.copy(), empty_layer, empty_mask.copy(), [], [], empty_mask.copy())
    if base_image.shape[:2] != shape:
        raise ValueError("base_image and target_reference must share target coordinates")

    # v0.8.28 performance: these multi-megapixel colour-space conversions are
    # page invariants. Compute them once instead of once per bubble/container.
    source_gray_u8 = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    source_gray_f32 = source_gray_u8.astype(np.float32)
    source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    target_gray = cv2.cvtColor(target_reference, cv2.COLOR_BGR2GRAY)
    target_hsv = cv2.cvtColor(target_reference, cv2.COLOR_BGR2HSV)

    target_by_id = {b.id: b for b in target_bubbles}
    target_by_source = {
        str(b.meta.get("paired_source_id")): b
        for b in target_bubbles
        if str(b.meta.get("paired_source_id") or "")
    }
    pairs: list[tuple[BubbleInstance, BubbleInstance]] = []
    used_target_ids: set[str] = set()
    for index, sb in enumerate(source_bubbles):
        tid = str(sb.meta.get("paired_target_id") or "")
        tb = target_by_id.get(tid) if tid else None
        # Target-driven OCR-free completion stores the relation on TARGET only.
        # Older code silently dropped these pairs whenever at least one ordinary
        # paired_target_id existed in the same batch.
        if tb is None:
            tb = target_by_source.get(str(sb.id))
        if tb is None and len(source_bubbles) == len(target_bubbles) and index < len(target_bubbles):
            candidate = target_bubbles[index]
            if candidate.id not in used_target_ids:
                tb = candidate
        if tb is not None and tb.id not in used_target_ids:
            pairs.append((sb, tb))
            used_target_ids.add(tb.id)

    rendered = base_image.copy(); layer = empty_layer.copy(); composite = empty_mask.copy(); clear_all = empty_mask.copy()
    records: list[MaskTransferRecord] = []; matches: list[BubblePatchMatch] = []
    for sb, tb in pairs:
        sm_raw = _bubble_mask(sb, source.shape[:2]); tm_raw = _bubble_mask(tb, shape)
        # v0.8.26: detector geometry and true writable container interior are
        # different layers. Solidify both masks before any clear/clip operation.
        # This prevents target Japanese glyphs from surviving in mask notches and
        # prevents source Chinese strokes from being clipped by source-side text.
        sm = _solidify_container_mask(sm_raw, cfg); tm = _solidify_container_mask(tm_raw, cfg)
        sbox = _bbox_from_mask(sm); tbox = _bbox_from_mask(tm)
        conf = float(min(sb.confidence, tb.confidence))
        # Target-driven recovery is used when the source page is much larger
        # than the HD target (common for the Chinese scan).  Its page scale is
        # intentionally below the ordinary rigid-container range; geometry is
        # still locked by the target mask and the source raster is resized with
        # one scalar only.  Keep the stricter range for normal candidates.
        eligibility_cfg = cfg
        placement_cfg = cfg
        if bool(sb.meta.get("target_driven_recovery")) or str(sb.meta.get("backend", "")) == "unseeded_white":
            if hasattr(cfg, "model_copy"):
                eligibility_cfg = cfg.model_copy(deep=True)
            else:
                eligibility_cfg = copy.copy(cfg)
            object.__setattr__(eligibility_cfg, "rigid_container_min_uniform_scale", min(
                0.20, float(getattr(cfg, "rigid_container_min_uniform_scale", 0.35))
            ))
            # OCR-free completion candidates already passed a target-space white-container
            # detector plus a registered ink-change gate. Their SOURCE crop can still
            # look less white than the TARGET because Chinese glyphs are denser or the
            # scan is darker/noisier. Relax the source/shape thresholds so missed open
            # balloons and starbursts can upgrade from warped text transfer to the rigid
            # original-source patch path.
            object.__setattr__(eligibility_cfg, "rigid_container_min_source_white_ratio", min(
                0.48, float(getattr(cfg, "rigid_container_min_source_white_ratio", 0.78))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_min_target_white_ratio", min(
                0.55, float(getattr(cfg, "rigid_container_min_target_white_ratio", 0.75))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_min_fill_ratio", min(
                0.18, float(getattr(cfg, "rigid_container_min_fill_ratio", 0.55))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_spiky_min_fill_ratio", min(
                0.14, float(getattr(cfg, "rigid_container_spiky_min_fill_ratio", 0.30))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_spiky_min_white_ratio", min(
                0.55, float(getattr(cfg, "rigid_container_spiky_min_white_ratio", 0.78))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_min_ink_coverage", min(
                0.88, float(getattr(cfg, "rigid_container_min_ink_coverage", 0.985))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_min_mask_containment", min(
                0.88, float(getattr(cfg, "rigid_container_min_mask_containment", 0.955))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_min_scale_factor", min(
                0.88, float(getattr(cfg, "rigid_container_min_scale_factor", 0.94))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_max_shift_px", max(
                12, int(getattr(cfg, "rigid_container_max_shift_px", 7))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_offset_patience", max(
                48, int(getattr(cfg, "rigid_container_offset_patience", 24))
            ))
            placement_cfg = eligibility_cfg
            if bool(sb.meta.get("target_driven_colored")):
                # A coloured burst is intentionally not white on the target;
                # its geometry is still trusted, while source pixels provide
                # the Chinese raster and the whole target shape is cleared.
                object.__setattr__(eligibility_cfg, "rigid_container_min_target_white_ratio", 0.0)
                object.__setattr__(eligibility_cfg, "rigid_container_max_target_saturation_median", 255.0)
                object.__setattr__(eligibility_cfg, "rigid_container_min_target_dark_ratio", 0.002)
                object.__setattr__(eligibility_cfg, "rigid_container_min_fill_ratio", 0.30)
                object.__setattr__(eligibility_cfg, "rigid_container_spiky_min_fill_ratio", 0.20)
                object.__setattr__(eligibility_cfg, "rigid_container_max_dark_ratio", 1.0)
        eligible, diag = _rigid_container_pair_eligible(
            source, target_reference, sm, tm, eligibility_cfg,
            source_gray=source_gray_u8, source_hsv=source_hsv,
            target_gray=target_gray, target_hsv=target_hsv,
        )
        if not eligible:
            continue  # caller sends unhandled regions down the legacy/saturated paths
        base_scale = float(diag.get("uniform_scale", 1.0))
        placed = _rigid_source_raster(source, sm, tm, shape, base_scale, placement_cfg, source_gray=source_gray_f32)
        if placed is None:
            continue
        alpha, source_ink_mask, scale, dx, dy, ink_cov, mask_cov = placed
        source_ink_mask, source_boundary_removed = remove_container_boundary_line_components(source_ink_mask, tm)
        if source_boundary_removed > 0:
            alpha[source_ink_mask == 0] = 0.0
            diag["source_boundary_line_pixels_removed"] = int(source_boundary_removed)

        # v1.0.7: the target container is geometry truth, but TARGET artwork/
        # colour is also the only background truth. Clear only compact Japanese
        # lettering, never the whole interior, and draw only SOURCE ink opacity as
        # neutral black. SOURCE paper RGB is forbidden from this path.
        border_safe_envelope = _rigid_target_write_envelope(tm, placement_cfg)
        clear = np.zeros(shape, np.uint8)
        full_clear_diag = {
            "white_full_clear_applied": False,
            "white_full_clear_reason": "target_colored_or_disabled",
            "white_full_clear_pixels": 0,
        }
        target_colored = bool(sb.meta.get("target_driven_colored"))
        full_clear_envelope = border_safe_envelope
        if (
            not target_colored
            and bool(getattr(cfg, "white_container_full_clear_enabled", True))
        ):
            paper_mask = white_container_paper_mask(target_reference, tm, source_ink_mask)
            candidate_clear_env, clear_env_diag = white_container_write_envelope(
                target_reference, tm, paper_mask,
                inset_px=max(0, int(getattr(cfg, "white_container_clear_inset_px", 0))),
                border_guard_px=max(0, int(getattr(cfg, "white_container_clear_border_guard_px", 0))),
            )
            if cv2.countNonZero(candidate_clear_env) > 0:
                full_clear_envelope = candidate_clear_env
            full_out, full_mask, full_clear_diag = clear_uniform_white_container_interior(
                rendered, target_reference, full_clear_envelope,
                min_paper_ratio=float(getattr(cfg, "white_container_full_clear_min_paper_ratio", 0.68)),
                max_robust_spread=float(getattr(cfg, "white_container_full_clear_max_robust_spread", 14.0)),
            )
            full_clear_diag["clear_envelope"] = clear_env_diag
            if bool(full_clear_diag.get("white_full_clear_applied", False)):
                rendered = full_out
                clear = full_mask

        if not bool(full_clear_diag.get("white_full_clear_applied", False)):
            clear = target_text_mask_in_container(target_reference, border_safe_envelope)
            if cv2.countNonZero(clear) > 0:
                d = max(1, int(getattr(cfg, "paired_diff_complex_clear_dilate_px", 2)))
                clear = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1)))
                clear = cv2.bitwise_and(clear, border_safe_envelope)
                cleaned = cv2.inpaint(rendered, clear, 2.5, cv2.INPAINT_TELEA)
                rendered[clear > 0] = cleaned[clear > 0]

        write_alpha8: np.ndarray
        backend = "rigid-container-text-only"
        clarity = "target-background-source-ink-only"
        match_notes = ["rigid_container_text_only", "target_background_preserved", f"uniform_scale={scale:.6f}", f"ink_coverage={ink_cov:.5f}", f"mask_containment={mask_cov:.5f}"]
        patch = None
        gap_fill_diag = {"added_pixels": 0, "enabled": False}
        a = np.clip(alpha, 0.0, 1.0)
        a *= (border_safe_envelope.astype(np.float32) / 255.0)
        # Neutral black preserves the original Chinese raster opacity/topology but
        # cannot carry white/gray SOURCE paper or scan colour into TARGET.
        rendered = np.clip(rendered.astype(np.float32) * (1.0 - a[..., None]), 0, 255).astype(np.uint8)
        # A conservative clear can leave isolated old-Japanese punctuation dots
        # near the inset envelope. Remove only TARGET-origin tiny components that
        # have no SOURCE Chinese support; real Chinese punctuation is protected.
        rendered, speck_mask, speck_diag = cleanup_target_residual_specks(
            rendered, target_reference, border_safe_envelope, source_ink_mask, clear,
            white_container=True, inpaint_radius=2.0,
        )
        if int(speck_diag.get("residual_specks_removed", 0)) > 0:
            match_notes.append(f"residual_specks_removed={int(speck_diag.get('residual_specks_removed', 0))}")
        clear = np.maximum(clear, speck_mask)
        write_alpha8 = np.maximum(clear, (a * 255.0).astype(np.uint8))

        border_diag = {
            "enabled": bool(getattr(cfg, "rigid_container_full_patch_preserve_target_border", True)),
            "protected_pixels": 0,
            "changed_before_restore": 0,
            "changed_after_restore": 0,
        }
        if bool(getattr(placement_cfg, "rigid_container_full_patch_preserve_target_border", True)):
            if bool(full_clear_diag.get("white_full_clear_applied", False)):
                # Restore only the actual HD outline/rules. The old implementation
                # restored the entire inset ring and could resurrect Japanese glyph
                # fragments sitting near a narration-box edge.
                protected_ring = target_container_border_mask(target_reference, tm, band_px=4)
            else:
                protected_ring = cv2.bitwise_and(tm, cv2.bitwise_not(border_safe_envelope))
            ring_sel = protected_ring > 0
            border_diag["protected_pixels"] = int(np.count_nonzero(ring_sel))
            if np.any(ring_sel):
                before_delta = np.max(
                    np.abs(rendered.astype(np.int16) - target_reference.astype(np.int16)), axis=2
                )
                border_diag["changed_before_restore"] = int(np.count_nonzero(ring_sel & (before_delta > 0)))
                if bool(getattr(placement_cfg, "rigid_container_exact_target_border_restore", True)):
                    rendered[ring_sel] = target_reference[ring_sel]
                    write_alpha8[ring_sel] = 0
                    clear[ring_sel] = 0
                after_delta = np.max(
                    np.abs(rendered.astype(np.int16) - target_reference.astype(np.int16)), axis=2
                )
                border_diag["changed_after_restore"] = int(np.count_nonzero(ring_sel & (after_delta > 0)))

        composite = np.maximum(composite, write_alpha8); clear_all = np.maximum(clear_all, clear)
        use = write_alpha8 > 0
        if np.any(use):
            layer[use, :3] = rendered[use][:, ::-1]
        layer[..., 3] = np.maximum(layer[..., 3], write_alpha8)

        matches.append(BubblePatchMatch(
            sb.id, tb.id, conf, 1.0 - conf, mask_cov, 0.0, 1.0,
            match_notes,
        ))
        rec = MaskTransferRecord(sb.id, tb.id, conf, True, "applied_rigid_container_raster")
        rec.sr_backend = backend; rec.sr_scale = scale
        rec.geometry_mode = "rigid_uniform_container"; rec.clarity_mode = clarity
        rec.mask_iou = mask_cov; rec.target_coverage = ink_cov; rec.spill_ratio = max(0.0, 1.0 - ink_cov)
        rec.local_dx = dx; rec.local_dy = dy; rec.ink_ratio = float(cv2.countNonZero(source_ink_mask) / max(1, cv2.countNonZero(tm)))
        if sbox is not None: rec.source_bbox = sbox
        if tbox is not None: rec.target_bbox = tbox
        if patch is not None:
            rec.meta["mask_write_gap_fill"] = gap_fill_diag
        rec.meta["target_border_preservation"] = border_diag
        rec.meta["white_container_full_clear"] = full_clear_diag
        if int(border_diag.get("changed_after_restore", 0)) > 0:
            rec.review_required = True
            rec.review_reason = "protected_target_border_changed"
            rec.restorable = True
            rec.editable = True
        target_ink, _ = _compact_container_ink(target_reference, clear, 190, cfg, gray=target_gray)
        min_cov = float(getattr(cfg, "rigid_container_acceptance_min_source_coverage", 0.985))
        max_res = float(getattr(cfg, "rigid_container_acceptance_max_target_residual", 0.02))
        _evaluate_content_completeness(
            rec,
            source_ink_mask,
            target_ink,
            rendered,
            cfg,
            tolerance_px=3,
            min_source_coverage=min_cov,
            max_target_residual=max_res,
        )
        if not rec.content_complete and patch is not None:
            repaired, repaired_mask, repair_diag = _repair_content_region(
                rec, rendered, patch_rgb, target_reference, write_alpha8, tm,
                source_ink_mask, target_ink, cfg,
                tolerance_px=3, min_source_coverage=min_cov, max_target_residual=max_res,
            )
            rec.meta["content_auto_repair"] = repair_diag
            if bool(repair_diag.get("improved", False)):
                rendered = repaired
                write_alpha8 = np.maximum(write_alpha8, repaired_mask)
                clear = np.maximum(clear, repaired_mask)
                composite = np.maximum(composite, write_alpha8)
                clear_all = np.maximum(clear_all, clear)
                use = write_alpha8 > 0
                if np.any(use):
                    layer[use, :3] = rendered[use][:, ::-1]
                layer[..., 3] = np.maximum(layer[..., 3], write_alpha8)
        # A full-container clear + near-total source-raster containment is a stronger
        # success criterion than legacy "pixels were written".
        if not rec.content_complete:
            rec.review_required = True; rec.review_reason = "rigid_container_content_check_failed"; rec.restorable = True; rec.editable = True
        elif rec.repair_succeeded and rec.review_reason == "rigid_container_content_check_failed":
            rec.review_required = False; rec.review_reason = ""
        records.append(rec)
    return MaskTransferResult(rendered, layer, composite, matches, records, clear_all)


def _fast_dark_pixel_clear(
    image: np.ndarray,
    clear_envelope: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray | None, np.ndarray, dict]:
    """Fast white-container clear that removes only dark target glyph pixels.

    Returns ``(image_or_none, actual_clear_mask, diagnostics)``. ``None`` means
    the local region is not paper-like enough and callers should use the normal
    component/inpaint path instead.
    """
    shape = image.shape[:2]
    empty = np.zeros(shape, np.uint8)
    if not bool(getattr(cfg, "fast_dark_pixel_clear_enabled", True)):
        return None, empty, {"enabled": False, "reason": "disabled"}
    env = (clear_envelope > 0)
    n = int(np.count_nonzero(env))
    if n < 16:
        return None, empty, {"enabled": True, "reason": "empty_envelope"}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_ratio = float(np.mean(gray[env] >= 220))
    min_white = float(getattr(cfg, "fast_dark_pixel_clear_min_white_ratio", 0.72))
    if white_ratio < min_white:
        return None, empty, {
            "enabled": True, "reason": "not_white_container",
            "white_ratio": white_ratio, "min_white_ratio": min_white,
        }
    threshold = int(getattr(cfg, "fast_dark_pixel_clear_threshold", 185))
    dark = ((gray <= threshold) & env).astype(np.uint8) * 255
    if cv2.countNonZero(dark) == 0:
        return image.copy(), dark, {
            "enabled": True, "reason": "no_dark_pixels", "white_ratio": white_ratio,
            "cleared_pixels": 0,
        }
    # Keep the operation glyph-local. A tiny dilation catches antialiasing without
    # turning the whole balloon into an inpaint request.
    dark = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    dark[~env] = 0
    bright = env & (gray >= 220)
    paper = np.median(image[bright], axis=0).astype(np.uint8) if int(np.count_nonzero(bright)) >= 12 else np.array([255, 255, 255], np.uint8)
    base = image.copy()
    base[dark > 0] = paper
    radius = float(getattr(cfg, "fast_dark_pixel_clear_inpaint_radius", 1.5))
    if radius > 0 and cv2.countNonZero(dark) > 0:
        refined = cv2.inpaint(base, dark, radius, cv2.INPAINT_TELEA)
        base[dark > 0] = refined[dark > 0]
    return base, dark, {
        "enabled": True,
        "reason": "applied",
        "white_ratio": white_ratio,
        "threshold": threshold,
        "cleared_pixels": int(cv2.countNonZero(dark)),
        "paper_bgr": paper.tolist(),
        "inpaint_radius": radius,
    }


def transfer_paired_diff_regions(
    aligned_source: np.ndarray,
    target: np.ndarray,
    source_bubbles: list[BubbleInstance],
    target_bubbles: list[BubbleInstance],
    cfg: MaskReplaceConfig | None = None,
) -> MaskTransferResult:
    """Composite paired-diff regions directly in target coordinates.

    ``aligned_source`` has already passed global registration and the optional
    low-frequency dense alignment in :mod:`paired_diff`. The target mask therefore
    defines the only writable pixels. This avoids rejecting photographed editions
    merely because their source balloon outline is 1-3 pixels different after warp.
    """
    cfg = cfg or MaskReplaceConfig()
    shape = target.shape[:2]
    if aligned_source.ndim == 3 and aligned_source.shape[2] >= 3:
        source_sat_p90 = float(np.percentile(cv2.cvtColor(aligned_source, cv2.COLOR_BGR2HSV)[..., 1], 90.0))
    else:
        source_sat_p90 = 0.0
    if aligned_source.shape[:2] != shape:
        raise ValueError("aligned_source must be in target coordinates")

    source_by_id = {b.id: b for b in source_bubbles}
    target_by_id = {b.id: b for b in target_bubbles}
    # Paired-diff ids share the same numeric suffix. Keep explicit one-to-one
    # matches for project/debug compatibility without re-solving Hungarian geometry.
    pairs: list[tuple[BubbleInstance, BubbleInstance]] = []
    for sb in source_bubbles:
        suffix = sb.id.rsplit("-", 1)[-1]
        tb = target_by_id.get(f"diff-dst-{suffix}")
        if tb is not None:
            pairs.append((sb, tb))
    if not pairs and len(source_bubbles) == len(target_bubbles):
        pairs = list(zip(source_bubbles, target_bubbles))

    matches: list[BubblePatchMatch] = []
    records: list[MaskTransferRecord] = []
    rendered = target.copy()
    layer = np.zeros((shape[0], shape[1], 4), np.uint8)
    composite_mask = np.zeros(shape, np.uint8)
    clear_mask_all = np.zeros(shape, np.uint8)
    saturated_seen = np.zeros(shape, np.uint8)

    def _is_page_furniture(box: tuple[int, int, int, int] | None, kind: str) -> bool:
        """Reject tiny edge text that is page furniture, not translated content."""
        if not bool(getattr(cfg, "paired_diff_protect_page_furniture", True)):
            return False
        if box is None or kind not in {"free_text", "complex_text"}:
            return False
        x0, y0, x1, y1 = box
        h, w = shape
        bw = max(1, x1 - x0); bh = max(1, y1 - y0)
        narrow = bw <= w * float(getattr(cfg, "paired_diff_page_furniture_max_width_ratio", 0.18))
        short = bh <= h * float(getattr(cfg, "paired_diff_page_furniture_max_height_ratio", 0.12))
        edge = (
            y1 <= h * float(getattr(cfg, "paired_diff_page_furniture_top_ratio", 0.10))
            or y0 >= h * float(getattr(cfg, "paired_diff_page_furniture_bottom_ratio", 0.965))
        )
        return bool(edge and narrow and short)

    for sb, tb in pairs:
        tm = _bubble_mask(tb, shape)
        sm = _bubble_mask(sb, sb.mask.shape if sb.mask is not None else shape)
        tbox = _bbox_from_mask(tm)
        sbox = _bbox_from_mask(sm)
        conf = float(min(sb.confidence, tb.confidence))
        match = BubblePatchMatch(
            sb.id, tb.id, conf, 1.0 - conf, 1.0, 0.0, 1.0,
            ["paired-target-space", f"source_mask_iou={tb.meta.get('paired_mask_iou', 0.0):.3f}"],
        )
        matches.append(match)
        rec = MaskTransferRecord(sb.id, tb.id, conf, False, "not_applied")
        photo_source = bool(tb.meta.get("photo_source")) or str(tb.meta.get("paired_diff_method", "")).startswith("photo_")
        rec.geometry_mode = "photo_pair" if photo_source else str(tb.meta.get("paired_diff_method") or "standard")
        if sbox:
            rec.source_bbox = sbox
        if tbox:
            rec.target_bbox = tbox
        if photo_source and sbox is not None:
            edge_sides = _edge_touch_sides(
                sbox, sm.shape[:2], int(getattr(cfg, "photo_pair_edge_clip_margin_px", 2))
            )
            rec.source_edge_clipped = bool(edge_sides)
            rec.source_edge_sides = ",".join(edge_sides)
        if tbox is None or cv2.countNonZero(tm) == 0:
            rec.reason = "empty_target_mask"
            records.append(rec)
            continue

        region_kind = str(tb.meta.get("paired_region_kind", "bubble"))
        if _is_page_furniture(tbox, region_kind):
            # This is intentionally not a review candidate: the safest output is
            # the untouched HD target, because the region is outside the dialogue
            # replacement scope. Do not create a clear mask or transfer record.
            matches.pop()
            continue
        paste_mask = tm.copy()
        # Enclosed bubble masks include the complete interior. A tiny inset keeps
        # the clean HD target outline/tail untouched. Free SFX masks are already
        # local density masks, so they are not eroded.
        if (
            region_kind == "bubble"
            and cfg.preserve_target_border
            and cfg.paired_diff_target_border_inset_px > 0
        ):
            r = int(cfg.paired_diff_target_border_inset_px)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
            eroded = cv2.erode(paste_mask, k)
            if cv2.countNonZero(eroded) > 0:
                paste_mask = eroded

        if cv2.countNonZero(paste_mask) == 0:
            rec.reason = "empty_paste_mask"
            records.append(rec)
            continue

        # v0.8.33: classify the TARGET container before choosing the compositor.
        # A paired-diff candidate may be labelled ``bubble`` even when its mask
        # contains a coloured burst (or a piece of artwork). If that happens, the
        # ordinary whole-raster branch would paste the monochrome SOURCE paper
        # over the TARGET colour. The target-aware compositor must win first for
        # every region kind, not only for pre-labelled free_text/complex_text.
        saturated_gate = None
        saturated_route = False
        if region_kind == "bubble":
            candidate_gate = _dominant_saturated_container_mask(target, tm, cfg)
            if candidate_gate is not None:
                gate_area = max(1, cv2.countNonZero(candidate_gate))
                overlap = float(np.count_nonzero((candidate_gate > 0) & (tm > 0)) / gate_area)
                src_gray_for_sat = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2GRAY)
                sat_sel = candidate_gate > 0
                source_bright_ratio = float(np.mean(src_gray_for_sat[sat_sel] >= 220)) if np.any(sat_sel) else 0.0
                min_overlap = float(getattr(cfg, "paired_diff_saturated_min_region_overlap_ratio", 0.15))
                min_bright = float(getattr(cfg, "paired_diff_saturated_min_source_bright_ratio", 0.75))
                if overlap >= min_overlap and source_bright_ratio >= min_bright:
                    saturated_gate = candidate_gate
                    saturated_route = True

        # v0.8.21: open/complex text must never replace a rectangular coloured
        # background. Clear only Japanese glyph components and composite only the
        # registered Chinese raster ink. This path also covers burst balloons and
        # captions printed directly on artwork.
        if region_kind in {"free_text", "complex_text"} or saturated_route:
            # A legitimate cross-rendition burst has a bright paper-like source
            # container under the registered Chinese glyphs.  If the recovered
            # saturated colour belongs to nearby artwork/signage instead, the
            # aligned source at that location is usually not bright.  Reject that
            # colour route and fall back to bounded component transfer.
            if saturated_gate is None:
                saturated_gate = _dominant_saturated_container_mask(target, tm, cfg)
            if saturated_gate is not None:
                src_gray_for_sat = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2GRAY)
                sat_sel = saturated_gate > 0
                source_bright_ratio = float(np.mean(src_gray_for_sat[sat_sel] >= 220)) if np.any(sat_sel) else 0.0
                if source_bright_ratio < float(getattr(cfg, "paired_diff_saturated_min_source_bright_ratio", 0.75)):
                    saturated_gate = None
            saturated_route = saturated_gate is not None
            if saturated_gate is not None:
                sat_area = max(1, cv2.countNonZero(saturated_gate))
                overlap = int(np.count_nonzero((saturated_gate > 0) & (saturated_seen > 0))) / sat_area
                if overlap >= float(getattr(cfg, "paired_diff_saturated_duplicate_overlap", 0.82)):
                    # Structural splitting can emit two changed-text islands from
                    # the same burst balloon.  The first route already rebuilt the
                    # complete flat-colour container; do not process a subset a
                    # second time or turn it into a fake extra "applied" record.
                    matches.pop()
                    continue
                text_img, write_mask, source_ink_mask, text_diag = _transfer_saturated_text_container(
                    aligned_source, rendered, saturated_gate, cfg
                )
            else:
                text_img, write_mask, source_ink_mask, text_diag = _transfer_open_complex_text_region(
                    aligned_source, rendered, tm, cfg
                )
            rec.geometry_mode = region_kind
            rec.sr_backend = "paired-saturated-container" if saturated_route else "paired-text-components"
            rec.sr_scale = 1.0
            rec.mask_iou = 1.0
            rec.target_coverage = 1.0 if text_img is not None else 0.0
            rec.spill_ratio = 0.0
            ink_gate = saturated_gate if saturated_gate is not None else tm
            rec.ink_ratio = float(cv2.countNonZero(source_ink_mask) / max(1, cv2.countNonZero(ink_gate)))
            if text_img is None:
                rec.reason = str(text_diag.get("reason") or "open_text_component_transfer_failed")
                rec.review_required = True
                rec.review_reason = rec.reason
                rec.restorable = True
                rec.editable = True
                records.append(rec)
                continue
            # Content completeness is a review signal, not a reason to restore
            # Japanese.  If a usable Chinese raster candidate was produced,
            # publish it even when the audit finds missing strokes or residual
            # target ink.  The result remains explicitly reversible/editable.
            if saturated_route:
                rec.clarity_mode = "saturated-container-ink-transfer"
            else:
                rec.clarity_mode = "complex-text-ink-transfer" if region_kind == "complex_text" else "open-text-ink-transfer"
            _evaluate_content_completeness(
                rec, source_ink_mask, text_diag.get("target_ink_mask"), text_img, cfg
            )

            rendered = text_img
            alpha8 = write_mask.astype(np.uint8)
            composite_mask = np.maximum(composite_mask, alpha8)
            diag_clear = text_diag.get("clear_mask")
            if isinstance(diag_clear, np.ndarray) and diag_clear.shape == shape:
                clear_mask_all = np.maximum(clear_mask_all, diag_clear)
            if saturated_gate is not None:
                saturated_seen = np.maximum(saturated_seen, saturated_gate)
            use = alpha8 > 0
            layer[use, :3] = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)[use]
            layer[..., 3] = np.maximum(layer[..., 3], alpha8)
            rec.applied = True
            if not bool(getattr(rec, "content_complete", False)):
                rec.reason = "applied_incomplete_review_candidate"
                rec.candidate = True
                rec.review_required = True
                rec.review_reason = "content_incomplete_published_per_user_policy"
                rec.restorable = True
                rec.editable = True
                records.append(rec)
                continue
            low_conf = conf < float(getattr(cfg, "paired_diff_low_confidence_candidate_threshold", 0.64))
            boundary_touch = bool(text_diag.get("boundary_touch", False))
            if low_conf or boundary_touch:
                rec.reason = "applied_low_confidence_text_candidate"
                rec.candidate = True
                rec.review_required = True
                rec.review_reason = (
                    "source_text_cluster_touches_candidate_boundary" if boundary_touch
                    else "low_confidence_open_or_complex_text_region"
                )
                rec.restorable = True
                rec.editable = True
            else:
                rec.reason = "applied_registered_text_components"
            records.append(rec)
            continue

        warped_img = aligned_source.copy()
        target_clear_mask = (
            _expand_target_clear_mask_with_text_components(target, tm, cfg)
            if region_kind == "bubble" else tm
        )
        # The geometric write mask may be intentionally inset from the balloon
        # edge.  Japanese glyph components can bridge that inset and survive as
        # fragments.  Clear only those verified target text components first; do
        # not widen the source patch or copy surrounding artwork.
        preclear_extra = cv2.bitwise_and(target_clear_mask, cv2.bitwise_not(paste_mask))
        if region_kind == "bubble" and cv2.countNonZero(preclear_extra) > 0:
            fast_cleared, fast_clear_mask, fast_clear_diag = _fast_dark_pixel_clear(rendered, preclear_extra, cfg)
            if fast_cleared is not None:
                rendered = fast_cleared
                preclear_extra = fast_clear_mask
                rec.meta["fast_dark_pixel_clear"] = fast_clear_diag
            else:
                rendered = cv2.inpaint(rendered, preclear_extra, 2.5, cv2.INPAINT_TELEA)
                rec.meta["fast_dark_pixel_clear"] = fast_clear_diag
        output_mask = paste_mask.copy()
        if cfg.normalize_background:
            warped_img = _normalize_bubble_background(warped_img, tm, target, tm)

        rec.sr_backend = "paired-dense-align"
        rec.sr_scale = 1.0
        # Geometry QA is defined against the target-driven write mask. Preserve the
        # lower source-outline IoU separately in paired-diff metadata/diagnostics.
        rec.mask_iou = 1.0
        rec.target_coverage = 1.0
        rec.spill_ratio = 0.0
        rec.sharpness = _masked_sharpness(warped_img, paste_mask)
        rec.target_sharpness = _masked_sharpness(target, paste_mask)
        rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0

        fidelity = (cfg.text_fidelity_mode or "auto").lower().strip()
        # v0.8.34.4 Pixel Enhance: preserve the original source glyph raster and
        # layout, but strengthen soft antialiased edges before falling all the way
        # to binary ink reconstruction/OCR. This is most useful for clean low-res
        # scans mapped onto a higher-resolution target.
        if (fidelity in {"auto", "pixels"}
                and rec.clarity_mode in {"pixels", "paired-aligned-pixels"}
                and bool(getattr(cfg, "pixel_enhance_enabled", True))
                and (rec.sharpness < float(getattr(cfg, "pixel_enhance_sharpness_trigger", 58.0))
                     or rec.relative_sharpness < float(getattr(cfg, "pixel_enhance_relative_trigger", 0.58)))):
            enhanced, enhance_diag = _pixel_enhance_text_raster(warped_img, paste_mask, cfg)
            rec.meta["pixel_enhance"] = enhance_diag
            if enhanced is not None:
                warped_img = enhanced
                rec.clarity_mode = "pixel-enhance"
                rec.sharpness = _masked_sharpness(warped_img, paste_mask)
                rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0

        too_soft = (
            rec.sharpness < cfg.min_pixel_text_sharpness
            or rec.relative_sharpness < cfg.min_relative_text_sharpness
        )
        # v0.8.5 structural supplements still originate from a photographed page.
        # Rebuild their local Chinese ink over the clean target instead of copying
        # camera pixels. Unlike a whole-bubble clear, the supplemental target mask
        # is compact, so this is also safe for open burst bubbles/free text.
        if photo_source and cfg.photo_pair_crisp_text_enabled:
            reconstructed = None
            ink_ratio = 0.0
            # v0.8.16: if source glyph pixels cross the target mask boundary, rescue
            # the complete *raster* glyph footprint first. This is deliberately
            # mask-only: it never reads/reflows OCR text and only moves the whole
            # source text block by the minimum amount required for containment.
            if (region_kind == "bubble"
                    and source_sat_p90 <= float(getattr(cfg, "photo_pair_recenter_max_source_saturation_p90", 24.0))):
                reconstructed, ink_ratio, rescue_meta = _reconstruct_photo_glyph_footprint_layer(
                    aligned_source, target, tm, cfg, clear_mask=target_clear_mask
                )
                if reconstructed is not None:
                    rec.clarity_mode = "photo-glyph-footprint-rescue"
                    rec.local_dx = float(rescue_meta.get("dx", 0.0))
                    rec.local_dy = float(rescue_meta.get("dy", 0.0))
                    rec.sr_scale = float(rescue_meta.get("scale", 1.0))
            # v0.8.7 small-balloon fallback remains for cases where the source text
            # is globally displaced but does not actually straddle the target mask.
            if (reconstructed is None
                    and region_kind == "bubble"
                    and getattr(cfg, "photo_pair_recenter_small_text_enabled", True)
                    and source_sat_p90 <= float(getattr(cfg, "photo_pair_recenter_max_source_saturation_p90", 24.0))):
                reconstructed, ink_ratio = _reconstruct_photo_recentered_ink_layer(
                    aligned_source, target, tm, cfg, clear_mask=target_clear_mask
                )
                if reconstructed is not None:
                    rec.clarity_mode = "photo-recentered-ink"
            if reconstructed is None:
                reconstructed, ink_ratio = _reconstruct_photo_crisp_layer(
                    warped_img, target, paste_mask, target_clear_mask, cfg
                )
                if reconstructed is not None:
                    rec.clarity_mode = "photo-crisp-ink"
            rec.ink_ratio = ink_ratio
            if reconstructed is not None:
                warped_img = reconstructed
                # The reconstructed image clears the complete clean target bubble
                # before painting source Chinese ink. Blend that complete interior
                # back, otherwise Japanese glyphs near the balloon edge survive the
                # smaller border-inset sampling mask and mix with Chinese.
                if region_kind == "bubble":
                    output_mask = target_clear_mask.copy()
                rec.sharpness = _masked_sharpness(warped_img, paste_mask)
                rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0
            else:
                rec.clarity_mode = "paired-aligned-pixels"
        # Paired transfer already knows where the translated pixels live. For
        # non-photo structural regions, use deterministic ink reconstruction only
        # when the aligned pixels are too soft.
        elif region_kind == "bubble" and (fidelity == "ink" or (fidelity == "auto" and too_soft)):
            reconstructed = None
            ink_ratio = 0.0
            if cfg.ink_reconstruction_enabled:
                reconstructed, ink_ratio = _reconstruct_ink_layer(warped_img, target, paste_mask, cfg)
            rec.ink_ratio = ink_ratio
            if reconstructed is not None:
                warped_img = reconstructed
                rec.clarity_mode = "ink-reconstruction"
                rec.sharpness = _masked_sharpness(warped_img, paste_mask)
                rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0
            else:
                rec.clarity_mode = "paired-aligned-pixels"
        else:
            rec.clarity_mode = "paired-aligned-pixels"

        gap_fill_diag = {"enabled": False, "iterations": 0, "added_pixels": 0}
        if region_kind == "bubble":
            output_mask, gap_fill_diag = _expand_safe_write_mask(output_mask, target_clear_mask, warped_img, target, cfg)
            rec.meta["mask_write_gap_fill"] = gap_fill_diag
        alpha = _alpha_from_mask(output_mask, cfg.feather_px)
        a3 = alpha[..., None]
        rendered = np.clip(
            warped_img.astype(np.float32) * a3 + rendered.astype(np.float32) * (1.0 - a3),
            0, 255,
        ).astype(np.uint8)
        alpha8 = (alpha * 255).astype(np.uint8)
        write_mask = np.maximum(alpha8, preclear_extra)
        composite_mask = np.maximum(composite_mask, write_mask)
        clear_mask_all = np.maximum(clear_mask_all, target_clear_mask)
        use = write_mask > 0
        layer[use, :3] = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)[use]
        layer[..., 3] = np.maximum(layer[..., 3], write_mask)
        rec.applied = True
        if photo_source and rec.source_edge_clipped:
            # Review-first policy: target-driven transfer may still recover useful
            # Chinese from an edge-clipped source. Publish it as an explicitly
            # reversible/editable candidate instead of reverting to Japanese.
            rec.reason = "applied_low_confidence_candidate"
            rec.candidate = True
            rec.review_required = True
            rec.review_reason = "source_text_region_clipped_at_page_edge"
            rec.restorable = True
            rec.editable = True
        else:
            rec.reason = "applied_paired_target_driven"
        # White speech balloons get the same second-stage content audit.  Use a
        # wider tolerance for photographed/global alignment because a valid glyph
        # rescue may intentionally translate the whole source text block a few px.
        src_audit, _ = _compact_container_ink(aligned_source, tm, 190, cfg)
        tgt_audit, _ = _compact_container_ink(target, tm, 190, cfg)
        audit_tol = max(int(getattr(cfg, "content_completeness_tolerance_px", 2)), 7 if photo_source else 3)
        _evaluate_content_completeness(
            rec, src_audit, tgt_audit, rendered, cfg,
            tolerance_px=audit_tol,
        )
        if region_kind == "bubble" and not rec.content_complete:
            repaired, repaired_mask, repair_diag = _repair_content_region(
                rec, rendered, warped_img, target, write_mask, target_clear_mask,
                src_audit, tgt_audit, cfg, tolerance_px=audit_tol,
            )
            rec.meta["content_auto_repair"] = repair_diag
            if bool(repair_diag.get("improved", False)):
                rendered = repaired
                write_mask = np.maximum(write_mask, repaired_mask)
                composite_mask = np.maximum(composite_mask, write_mask)
                clear_mask_all = np.maximum(clear_mask_all, repaired_mask)
                use = write_mask > 0
                if np.any(use):
                    layer[use, :3] = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)[use]
                layer[..., 3] = np.maximum(layer[..., 3], write_mask)
        records.append(rec)

    return MaskTransferResult(rendered, layer, composite_mask, matches, records, clear_mask_all)

def transfer_photo_color_sfx(
    aligned_source: np.ndarray,
    target: np.ndarray,
    cfg: MaskReplaceConfig | None = None,
) -> MaskTransferResult:
    """Transfer large vivid-red translated SFX from a photographed page.

    This intentionally targets only high-saturation red glyph groups that exist
    in both editions but have substantially different shapes. Stable red artwork
    (clothes, hats, scenery) overlaps after registration and therefore fails the
    change gate. The clean target SFX is locally inpainted, then source red fill
    and its immediately adjacent dark outline are rebuilt with crisp masks.
    """
    cfg = cfg or MaskReplaceConfig()
    shape = target.shape[:2]
    if aligned_source.shape[:2] != shape:
        return MaskTransferResult(target.copy(), np.zeros((shape[0], shape[1], 4), np.uint8), np.zeros(shape, np.uint8), [], [])

    def vivid_red(image: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(image.astype(np.float32))
        return ((r > 150.0) & (r > g * 1.55) & (r > b * 1.45)).astype(np.uint8) * 255

    sm = vivid_red(aligned_source); tm = vivid_red(target)
    if cv2.countNonZero(sm) < 300 or cv2.countNonZero(tm) < 300:
        return MaskTransferResult(target.copy(), np.zeros((shape[0], shape[1], 4), np.uint8), np.zeros(shape, np.uint8), [], [])
    group_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    union = cv2.dilate(cv2.bitwise_or(sm, tm), group_k)
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 9)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((union > 0).astype(np.uint8), 8)
    td = cv2.dilate(tm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    sd = cv2.dilate(sm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    page_area = shape[0] * shape[1]
    candidates = []
    for i in range(1, n):
        x, y, bw, bh, area = map(int, stats[i])
        if area < 500 or area / max(1, page_area) > 0.035:
            continue
        region = labels == i
        sr = int(np.count_nonzero((sm > 0) & region)); tr = int(np.count_nonzero((tm > 0) & region))
        if sr < 500 or tr < 500:
            continue
        matched = int(np.count_nonzero((sm > 0) & td & region) + np.count_nonzero((tm > 0) & sd & region))
        change = float(np.clip(1.0 - matched / max(1, sr + tr), 0.0, 1.0))
        if change < 0.22:
            continue
        candidates.append((i, (x, y, x + bw, y + bh), change, sr, tr))

    if not candidates:
        return MaskTransferResult(target.copy(), np.zeros((shape[0], shape[1], 4), np.uint8), np.zeros(shape, np.uint8), [], [])

    rendered = target.copy()
    layer = np.zeros((shape[0], shape[1], 4), np.uint8)
    composite = np.zeros(shape, np.uint8)
    clear_all = np.zeros(shape, np.uint8)
    records: list[MaskTransferRecord] = []
    matches: list[BubblePatchMatch] = []
    src_gray = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2GRAY)
    for idx, (lab, box, change, sr, tr) in enumerate(candidates):
        region = labels == lab
        src_red = ((sm > 0) & region).astype(np.uint8) * 255
        tgt_red = ((tm > 0) & region).astype(np.uint8) * 255
        # Include immediately adjacent dark outline/shadow, but not distant panel art.
        near_src = cv2.dilate(src_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        near_tgt = cv2.dilate(tgt_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        src_outline = ((src_gray < 70) & near_src & region).astype(np.uint8) * 255
        tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        tgt_outline = ((tgt_gray < 70) & near_tgt & region).astype(np.uint8) * 255
        clear = cv2.bitwise_or(tgt_red, tgt_outline)
        clear = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if cv2.countNonZero(clear) == 0:
            continue
        # Inpaint only the old SFX glyph pixels, preserving the rest of the panel.
        rendered = cv2.inpaint(rendered, clear, 3.0, cv2.INPAINT_TELEA)
        clear_all = np.maximum(clear_all, clear)
        red_pixels = target[tgt_red > 0]
        if len(red_pixels):
            red_color = np.median(red_pixels, axis=0).astype(np.uint8)
        else:
            red_color = np.array([35, 35, 205], np.uint8)
        rendered[src_red > 0] = red_color
        rendered[src_outline > 0] = (0, 0, 0)
        write_mask = cv2.bitwise_or(clear, cv2.bitwise_or(src_red, src_outline))
        composite = np.maximum(composite, write_mask)
        use = write_mask > 0
        layer[use, :3] = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)[use]
        layer[..., 3] = np.maximum(layer[..., 3], write_mask)
        sid, tid = f"color-sfx-src-{idx:03d}", f"color-sfx-dst-{idx:03d}"
        matches.append(BubblePatchMatch(sid, tid, 0.90, 0.10, 1.0, 0.0, 1.0, [f"red_shape_change={change:.3f}"]))
        rec = MaskTransferRecord(sid, tid, 0.90, True, "applied_color_sfx_rebuild")
        rec.geometry_mode = "photo_pair"
        rec.clarity_mode = "color-sfx-rebuild"
        rec.mask_iou = 1.0; rec.target_coverage = 1.0; rec.spill_ratio = 0.0
        rec.source_bbox = box; rec.target_bbox = box
        rec.ink_ratio = float(sr / max(1, (box[2]-box[0]) * (box[3]-box[1])))
        records.append(rec)
    return MaskTransferResult(rendered, layer, composite, matches, records, clear_all)


def transfer_bubble_patches(
    source: np.ndarray,
    target: np.ndarray,
    source_bubbles: list[BubbleInstance],
    target_bubbles: list[BubbleInstance],
    registration: RegistrationResult,
    cfg: MaskReplaceConfig | None = None,
) -> MaskTransferResult:
    cfg = cfg or MaskReplaceConfig()
    shape = target.shape[:2]
    matches = match_bubbles(source_bubbles, target_bubbles, registration, shape, cfg)
    source_by_id = {b.id: b for b in source_bubbles}
    target_by_id = {b.id: b for b in target_bubbles}
    rendered = target.copy()
    layer = np.zeros((shape[0], shape[1], 4), np.uint8)
    composite_mask = np.zeros(shape, np.uint8)
    clear_mask_all = np.zeros(shape, np.uint8)
    records: list[MaskTransferRecord] = []

    for match in matches:
        sb, tb = source_by_id[match.source_bubble_id], target_by_id[match.target_bubble_id]
        sm = _bubble_mask(sb, source.shape[:2])
        tm = _bubble_mask(tb, shape)
        sbox = _bbox_from_mask(sm); tbox = _bbox_from_mask(tm)
        rec = MaskTransferRecord(sb.id, tb.id, match.confidence, False, "not_applied")
        is_photo_pair = tb.meta.get("paired_diff_method") == "photo_pair"
        rec.geometry_mode = "photo_pair" if is_photo_pair else str(tb.meta.get("paired_diff_method") or "standard")
        if sbox: rec.source_bbox = sbox
        if tbox: rec.target_bbox = tbox
        source_edge_sides = _edge_touch_sides(
            sbox, source.shape[:2], getattr(cfg, "photo_pair_edge_clip_margin_px", 0)
        ) if is_photo_pair else ()
        rec.source_edge_clipped = bool(source_edge_sides)
        rec.source_edge_sides = ",".join(source_edge_sides)
        if _publication_safety_enabled(cfg) and match.confidence < cfg.min_match_confidence:
            rec.reason = "bubble_match_low_confidence"; records.append(rec); continue
        if not sbox or not tbox:
            rec.reason = "empty_bubble_mask"; records.append(rec); continue
        sw, sh = max(1, sbox[2]-sbox[0]), max(1, sbox[3]-sbox[1])
        tw, th = max(1, tbox[2]-tbox[0]), max(1, tbox[3]-tbox[1])
        small_text_photo_pair = bool(is_photo_pair and min(tw, th) < cfg.photo_pair_min_direct_side_px)
        # Compare target size against the *registered* source geometry, not raw
        # source pixels. A 2400px phone photo mapped onto an 850px clean master
        # is expected to have a much larger raw bbox; rejecting that scale change
        # made photographed editions impossible to transfer.
        H_base = transform_to_homography(registration.matrix)
        mapped_box = transform_points(
            [(sbox[0], sbox[1]), (sbox[2], sbox[1]), (sbox[2], sbox[3]), (sbox[0], sbox[3])], H_base
        )
        mx0, my0, mx1, my1 = polygon_bbox(mapped_box)
        mapped_w, mapped_h = max(1e-6, mx1 - mx0), max(1e-6, my1 - my0)
        if _publication_safety_enabled(cfg) and min(tw / mapped_w, th / mapped_h) < cfg.reject_if_target_smaller_ratio:
            rec.reason = "target_bubble_much_smaller_after_registration"; records.append(rec); continue

        # Same-source editions often differ only by translated lettering. When
        # page registration is effectively identity and paired-diff masks agree,
        # copy the translated interior pixel-for-pixel. This avoids needless
        # resampling/feathering and gives a measurable exactness fast path while
        # keeping the Japanese HD page untouched outside the destination mask.
        if cfg.exact_identity_copy and _identity_like(registration, cfg):
            warped_sm = _warp_mask(sm, registration.matrix, shape)
            exact_iou = _mask_iou(warped_sm, tm)
            coverage, spill = _target_coverage(warped_sm, tm)
            if exact_iou >= cfg.exact_identity_mask_iou:
                dest_mask = tm.copy()
                # Paired-diff / white-component masks already represent the interior
                # bounded by the black balloon outline. Do not erode them again.
                interior_mask = bool(tb.meta.get("mask_is_interior"))
                if cfg.preserve_target_border and cfg.border_inset_px > 0 and not interior_mask:
                    ksize = cfg.border_inset_px * 2 + 1
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                    dest_mask = cv2.erode(dest_mask, k, iterations=1)
                paste_mask = dest_mask
                if cfg.exact_identity_changed_fringe_px > 0:
                    # Translation editors often touch a few antialiased pixels just
                    # outside the white interior (ruby cleanup / box-edge AA). In the
                    # strict same-source fast path we can detect those pixels directly
                    # from the paired pages and include only the actually changed fringe.
                    r = int(cfg.exact_identity_changed_fringe_px)
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
                    near = cv2.dilate((dest_mask > 0).astype(np.uint8) * 255, k) > 0
                    pair_diff = np.mean(np.abs(source.astype(np.int16) - target.astype(np.int16)), axis=2)
                    changed = pair_diff >= float(cfg.paired_diff_pixel_threshold)
                    fringe = near & (dest_mask == 0) & changed
                    if np.any(fringe):
                        paste_mask = np.maximum(paste_mask, fringe.astype(np.uint8) * 255)
                if cv2.countNonZero(paste_mask):
                    use = paste_mask > 0
                    rendered[use] = source[use]
                    composite_mask = np.maximum(composite_mask, paste_mask)
                    layer[use, :3] = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)[use]
                    layer[..., 3] = np.maximum(layer[..., 3], paste_mask)
                    rec.sr_backend, rec.sr_scale = "pixel-exact", 1.0
                    rec.mask_iou, rec.target_coverage, rec.spill_ratio = exact_iou, coverage, spill
                    rec.sharpness = _masked_sharpness(source, paste_mask)
                    rec.applied = True
                    rec.reason = "applied_exact_identity"
                    src_audit, _ = _compact_container_ink(source, dest_mask, 190, cfg)
                    tgt_audit, _ = _compact_container_ink(target, dest_mask, 190, cfg)
                    _evaluate_content_completeness(rec, src_audit, tgt_audit, rendered, cfg, tolerance_px=3)
                    records.append(rec)
                    continue

        H = H_base.copy()
        if cfg.local_fit in {"bbox", "ecc"}:
            mapped = mapped_box
            mx0,my0,mx1,my1 = polygon_bbox(mapped)
            corr_x = tw / max(1e-6, mx1-mx0); corr_y = th / max(1e-6, my1-my0)
            max_scale_change = cfg.photo_pair_max_local_scale_change if is_photo_pair else cfg.max_local_scale_change
            correction_ok = abs(corr_x-1.0) <= max_scale_change and abs(corr_y-1.0) <= max_scale_change
            if correction_ok:
                if is_photo_pair and getattr(cfg, "photo_pair_uniform_local_fit", True):
                    # Never independently resize X/Y for photographed CJK text.
                    # That produced the visibly squeezed/twisted glyphs reported on
                    # real 007/009 pages. Use one local scale and center alignment.
                    axis_delta = abs(float(corr_x - corr_y))
                    if axis_delta <= float(getattr(cfg, "photo_pair_max_axis_scale_delta", 0.10)):
                        fit_H = _bbox_uniform_fit_matrix(
                            (sbox[0], sbox[1], sbox[2], sbox[3]),
                            (tbox[0], tbox[1], tbox[2], tbox[3]), H,
                        )
                    else:
                        fit_H = H.copy()
                else:
                    fit_H = _bbox_fit_matrix((sbox[0], sbox[1], sbox[2], sbox[3]), (tbox[0], tbox[1], tbox[2], tbox[3]), H)
                # Never accept a local correction merely because it exists. Curved
                # pages may already be better under the global registration.
                global_preview = _warp_mask(sm, H, shape)
                fit_preview = _warp_mask(sm, fit_H, shape)
                if _mask_iou(fit_preview, tm) > _mask_iou(global_preview, tm) + 0.005:
                    H = fit_H
            elif not is_photo_pair and _publication_safety_enabled(cfg):
                rec.reason = "local_scale_correction_too_large"; records.append(rec); continue
            # Photo pairs safely retain the global transform when local scaling is
            # implausible; the target mask remains the publication boundary.

        transfer_sm = sm
        if cfg.source_mask_expand_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            transfer_sm = cv2.dilate(transfer_sm, k, iterations=cfg.source_mask_expand_px)
        warped_img, warped_mask, sr_backend, sr_scale = _warp_source_patch(source, transfer_sm, H, shape, tbox, cfg)
        rec.sr_backend, rec.sr_scale = sr_backend, sr_scale
        if cfg.local_fit == "ecc":
            dx, dy, _ = _local_translation_ecc(warped_mask, tm, cfg)
            dx, dy, _subpix_score, subpix_diag = _subpixel_translation_refine(warped_mask, tm, dx, dy, cfg)
            if dx or dy:
                warped_img = _shift(warped_img, dx, dy, nearest=False)
                warped_mask = _shift(warped_mask, dx, dy, nearest=True)
            rec.local_dx, rec.local_dy = dx, dy
            rec.meta["local_subpixel_refine"] = subpix_diag

        rec.mask_iou = _mask_iou(warped_mask, tm)
        rec.target_coverage, rec.spill_ratio = _target_coverage(warped_mask, tm)
        min_iou = cfg.photo_pair_min_transfer_iou if is_photo_pair else cfg.min_mask_iou
        min_coverage = cfg.photo_pair_min_transfer_coverage if is_photo_pair else cfg.min_target_coverage
        max_spill = cfg.photo_pair_max_spill_ratio if is_photo_pair else cfg.max_spill_ratio
        if (
            is_photo_pair
            and rec.target_coverage < min_coverage
            and rec.target_coverage >= max(0.0, min_coverage - cfg.photo_pair_salvage_coverage_margin)
            and rec.spill_ratio <= (max_spill + cfg.photo_pair_salvage_spill_extra)
        ):
            salvaged_img, salvaged_mask, salvaged_backend, salvaged_scale = _photo_pair_salvage_warp(
                source, transfer_sm, warped_img, warped_mask, H, shape, tbox, tm, cfg,
            )
            salvage_iou = _mask_iou(salvaged_mask, tm)
            salvage_cov, salvage_spill = _target_coverage(salvaged_mask, tm)
            if (
                salvage_cov > rec.target_coverage + 1e-6
                or (abs(salvage_cov - rec.target_coverage) <= 1e-6 and salvage_iou > rec.mask_iou + 1e-6)
            ):
                warped_img, warped_mask = salvaged_img, salvaged_mask
                rec.sr_backend = salvaged_backend if salvaged_backend != "off" else rec.sr_backend
                rec.sr_scale = salvaged_scale if salvaged_scale != 1.0 else rec.sr_scale
                rec.mask_iou = salvage_iou
                rec.target_coverage, rec.spill_ratio = salvage_cov, salvage_spill

        # A camera-edge-clipped source is fundamentally different from ordinary
        # under-segmentation. Dilating/locally fitting the mask can improve the
        # geometry, but it cannot recover Chinese glyph pixels that were never
        # captured. v0.8.2 incorrectly accepted the real 009 top-right bubble at
        # ~86% coverage and then cleared the whole Japanese target, publishing a
        # visibly truncated translation. Require near-complete coverage whenever
        # the source bubble itself reaches the photo boundary. Bubbles that only
        # lose a little outline but retain their text still pass this stricter gate.
        if (
            is_photo_pair
            and getattr(cfg, "photo_pair_edge_clip_guard_enabled", True)
            and source_edge_sides
            and rec.target_coverage < max(
                min_coverage, float(getattr(cfg, "photo_pair_edge_clip_min_target_coverage", 0.94))
            )
        ):
            # v0.8.6 review-first policy: do not silently leave Japanese. When a
            # meaningful portion of the translated source still exists, publish a
            # recoverable Chinese *candidate* in the automatic preview, clearly
            # flag it for review, and preserve one-click restore/manual reletter.
            candidate_enabled = bool(getattr(cfg, "photo_pair_low_confidence_candidate_enabled", True))
            candidate_min_cov = float(getattr(cfg, "photo_pair_candidate_min_coverage", 0.55))
            if candidate_enabled and rec.target_coverage >= candidate_min_cov:
                dest_mask = tm.copy()
                interior_mask = bool(tb.meta.get("mask_is_interior"))
                if cfg.preserve_target_border and cfg.border_inset_px > 0 and not interior_mask:
                    ksize = cfg.border_inset_px * 2 + 1
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                    eroded = cv2.erode(dest_mask, k, iterations=1)
                    if cv2.countNonZero(eroded) > 0:
                        dest_mask = eroded
                partial_mask = cv2.bitwise_and(dest_mask, warped_mask)
                candidate_img = None
                candidate_ratio = 0.0
                if cv2.countNonZero(partial_mask) > 0:
                    if cfg.normalize_background:
                        warped_img = _normalize_bubble_background(warped_img, warped_mask, target, tm)
                    if cfg.photo_pair_crisp_text_enabled:
                        candidate_img, candidate_ratio = _reconstruct_photo_crisp_layer(
                            warped_img, target, partial_mask, dest_mask, cfg,
                            allow_nonwhite_target=True,
                        )
                    if candidate_img is None and cfg.photo_pair_normalize_text_pixels:
                        candidate_img = _normalize_photo_text_pixels(
                            warped_img, target, partial_mask, dest_mask, cfg,
                        )
                if candidate_img is not None:
                    use = dest_mask > 0
                    rendered[use] = candidate_img[use]
                    composite_mask = np.maximum(composite_mask, dest_mask)
                    rgb_candidate = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2RGB)
                    layer[use, :3] = rgb_candidate[use]
                    layer[..., 3] = np.maximum(layer[..., 3], dest_mask)
                    rec.ink_ratio = float(candidate_ratio)
                    rec.clarity_mode = "photo-crisp-ink-candidate" if candidate_ratio > 0 else "photo-normalized-candidate"
                    rec.sharpness = _masked_sharpness(candidate_img, dest_mask)
                    rec.target_sharpness = _masked_sharpness(target, dest_mask)
                    rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0
                    rec.applied = True
                    rec.reason = "applied_low_confidence_candidate"
                    rec.candidate = True
                    rec.review_required = True
                    rec.review_reason = "source_text_region_clipped_at_page_edge"
                    rec.restorable = True
                    rec.editable = True
                    src_audit, _ = _compact_container_ink(warped_img, dest_mask, 190, cfg)
                    tgt_audit, _ = _compact_container_ink(target, dest_mask, 190, cfg)
                    _evaluate_content_completeness(rec, src_audit, tgt_audit, rendered, cfg, tolerance_px=5)
                    records.append(rec)
                    continue
            rec.reason = "source_text_region_clipped_at_page_edge"
            rec.review_required = True
            rec.review_reason = rec.reason
            rec.restorable = True
            rec.editable = True
            records.append(rec)
            continue
        if _publication_safety_enabled(cfg):
            if rec.mask_iou < min_iou:
                rec.reason = "mask_iou_below_threshold"; records.append(rec); continue
            if rec.target_coverage < min_coverage:
                rec.reason = "target_coverage_below_threshold"; records.append(rec); continue
            if rec.spill_ratio > max_spill:
                rec.reason = "source_mask_spills_outside_target"; records.append(rec); continue

        dest_mask = tm.copy()
        if cfg.target_mask_expand_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dest_mask = cv2.dilate(dest_mask, k, iterations=cfg.target_mask_expand_px)
        interior_mask = bool(tb.meta.get("mask_is_interior"))
        if cfg.preserve_target_border and cfg.border_inset_px > 0 and not interior_mask:
            ksize = cfg.border_inset_px * 2 + 1
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            dest_mask = cv2.erode(dest_mask, k, iterations=1)

        # Only composite where source bubble pixels are valid. Gate above ensures this
        # still covers essentially the whole target interior, so Japanese text cannot
        # silently remain in uncovered holes.
        paste_mask = cv2.bitwise_and(dest_mask, warped_mask)
        if cv2.countNonZero(paste_mask) == 0:
            rec.reason = "empty_paste_mask"; records.append(rec); continue

        if cfg.normalize_background:
            warped_img = _normalize_bubble_background(warped_img, warped_mask, target, tm)
        rec.sharpness = _masked_sharpness(warped_img, paste_mask)
        rec.target_sharpness = _masked_sharpness(target, paste_mask)
        rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0

        fidelity = (cfg.text_fidelity_mode or "auto").lower().strip()
        if is_photo_pair and cfg.photo_pair_force_ink_reconstruction and fidelity == "auto":
            fidelity = "ink"
        if is_photo_pair and small_text_photo_pair and fidelity == "auto":
            fidelity = "ink"

        # Preferred v0.8.3 path: extract only registered Chinese ink and rebuild it
        # on clean target paper. This removes camera blur/glare and duplicated
        # source balloon outlines without requiring OCR or a font.
        if is_photo_pair and cfg.photo_pair_crisp_text_enabled and fidelity != "reject":
            crisp, crisp_ratio = _reconstruct_photo_crisp_layer(
                warped_img, target, paste_mask, dest_mask, cfg,
            )
            rec.ink_ratio = crisp_ratio
            if crisp is not None:
                warped_img = crisp
                paste_mask = dest_mask
                rec.clarity_mode = "photo-crisp-ink"
                rec.sharpness = _masked_sharpness(warped_img, dest_mask)
                rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0
                fidelity = "pixels"  # already rebuilt; skip binary reconstruction

        # Photograph-specific fidelity ladder. First remove smooth glare / page
        # shading while retaining real antialiased glyph pixels. Only if the
        # normalized result remains too soft do we use deterministic ink recovery;
        # very soft/tiny text is refused and left for OCR relettering.
        if (is_photo_pair and rec.clarity_mode != "photo-crisp-ink"
                and fidelity in {"auto", "pixels"} and cfg.photo_pair_normalize_text_pixels):
            normalized = _normalize_photo_text_pixels(warped_img, target, paste_mask, dest_mask, cfg)
            if normalized is not None:
                norm_sharp = _masked_sharpness(normalized, dest_mask)
                norm_rel = norm_sharp / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0
                # Explicit pixels means "preserve glyph pixels", but still allows
                # deterministic illumination normalization because it does not
                # redraw/infer characters.
                if fidelity == "pixels" or norm_rel >= cfg.photo_pair_normalize_min_relative_sharpness:
                    warped_img = normalized
                    paste_mask = dest_mask
                    rec.clarity_mode = "photo-normalized-pixels"
                    rec.sharpness = norm_sharp
                    rec.relative_sharpness = norm_rel
                    fidelity = "pixels"
                elif norm_rel < cfg.photo_pair_reletter_below_relative_sharpness:
                    # Do not hard-reject yet. Very soft photographed text can
                    # still be recovered deterministically as crisp black ink on
                    # clean target paper, especially for small bubbles.
                    warped_img = normalized
                    paste_mask = dest_mask
                    rec.sharpness = norm_sharp
                    rec.relative_sharpness = norm_rel
                    fidelity = "ink"
                else:
                    # Keep the normalized pixels as the input to ink recovery. It
                    # has no camera background gradient, making thresholding safer.
                    warped_img = normalized
                    paste_mask = dest_mask
                    rec.sharpness = norm_sharp
                    rec.relative_sharpness = norm_rel
                    fidelity = "ink"

        too_soft = (
            rec.sharpness < cfg.min_pixel_text_sharpness
            or rec.relative_sharpness < cfg.min_relative_text_sharpness
        )
        if fidelity == "reject":
            rec.clarity_mode = "rejected"
            rec.reason = "source_text_fidelity_rejected"
            records.append(rec)
            continue

        photo_prefers_ink = bool(
            is_photo_pair and rec.clarity_mode != "photo-crisp-ink" and (
                small_text_photo_pair
                or fidelity == "ink"
                or rec.relative_sharpness < cfg.photo_pair_prefer_ink_below_relative_sharpness
            )
        )
        should_try_ink = bool(
            fidelity == "ink"
            or (fidelity == "auto" and too_soft)
            or photo_prefers_ink
        )
        if should_try_ink:
            reconstructed = None
            ink_ratio = 0.0
            reconstruction_mask = paste_mask
            if cfg.ink_reconstruction_enabled:
                reconstructed, ink_ratio = _reconstruct_ink_layer(
                    warped_img, target, reconstruction_mask, cfg,
                    clear_mask=dest_mask if is_photo_pair else None,
                )
            rec.ink_ratio = ink_ratio
            used_reconstruction = False
            if reconstructed is not None:
                candidate_mask = dest_mask if is_photo_pair else paste_mask
                recon_sharpness = _masked_sharpness(reconstructed, candidate_mask)
                recon_relative = recon_sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0
                if (
                    fidelity == "ink"
                    or small_text_photo_pair
                    or rec.clarity_mode == "reletter-required"
                    or recon_sharpness >= rec.sharpness * cfg.photo_pair_prefer_ink_min_gain
                    or recon_relative >= rec.relative_sharpness + 0.06
                ):
                    warped_img = reconstructed
                    if is_photo_pair:
                        # Japanese glyphs were cleared across the clean target mask,
                        # but Chinese ink was extracted only from valid source pixels.
                        paste_mask = dest_mask
                    rec.clarity_mode = "ink-reconstruction"
                    rec.sharpness = recon_sharpness
                    rec.relative_sharpness = recon_relative
                    used_reconstruction = True
            if not used_reconstruction and (fidelity == "ink" or small_text_photo_pair or (fidelity == "auto" and too_soft)):
                if cfg.reject_blurry_source:
                    rec.clarity_mode = "reletter-required"
                    rec.reason = "source_text_too_blurry_for_pixel_transfer"
                    records.append(rec)
                    continue
        elif not rec.clarity_mode:
            rec.clarity_mode = "pixels"

        alpha = _alpha_from_mask(paste_mask, cfg.feather_px)
        a3 = alpha[..., None]
        rendered = np.clip(warped_img.astype(np.float32) * a3 + rendered.astype(np.float32) * (1.0 - a3), 0, 255).astype(np.uint8)
        composite_mask = np.maximum(composite_mask, (alpha * 255).astype(np.uint8))
        # RGBA layer is the exact transferred patch, useful for ORA/PSD review.
        use = alpha > 0
        layer[use, :3] = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)[use]
        layer[..., 3] = np.maximum(layer[..., 3], (alpha * 255).astype(np.uint8))
        rec.applied = True
        rec.reason = "applied"
        clear_mask_all = np.maximum(clear_mask_all, dest_mask if is_photo_pair else paste_mask)
        audit_gate = dest_mask if is_photo_pair else paste_mask
        src_audit, _ = _compact_container_ink(warped_img, audit_gate, 190, cfg)
        tgt_audit, _ = _compact_container_ink(target, audit_gate, 190, cfg)
        _evaluate_content_completeness(
            rec, src_audit, tgt_audit, rendered, cfg,
            tolerance_px=max(3, 5 if is_photo_pair else 3),
        )
        records.append(rec)

    return MaskTransferResult(rendered, layer, composite_mask, matches, records, clear_mask_all)
