from __future__ import annotations

"""Fast source-driven whole-container replacement for same-layout editions.

The translated source page is the only content source.  We discover closed white
speech/text containers on that page, use page registration + the target's dark
outline *only* to place them, and then copy the source interior as one raster.
The outline itself is never copied and no debug/alignment box is composited.

This route intentionally avoids the older source-bubble -> target-bubble detect /
match / resize loop.  If the pages are pixel-identical in size/geometry, the
source pixels are copied at the same coordinates.  If scan resolution, crop, or slight scan geometry differs, page registration may
use affine/homography coordinates to *locate* the corresponding area. Final CJK
pixels are still rendered only with a local similarity transform (uniform scale +
rotation + translation), with the target outline used only for fine alignment.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from ...config import DirectPatchConfig, MaskReplaceConfig
from ...coordinate_space import SourceCoordinateSpace
from ...geometry import mask_to_largest_polygon, transform_to_homography
from .geometry_ops import BubblePatchMatch
from .transfer_models import MaskTransferRecord, MaskTransferResult
from ...models import BubbleInstance, RegistrationResult
from ...plugins import REGISTRY as PROVIDER_REGISTRY
from .text_transfer import transfer_text_only, target_container_border_mask, target_text_mask_in_container
from .overlay import compose_direct_overlay
from .content_audit import _evaluate_content_completeness


@dataclass(slots=True)
class DirectContainerPlan:
    result: MaskTransferResult
    source_bubbles: list[BubbleInstance]
    target_bubbles: list[BubbleInstance]
    diagnostics: dict
    safe_to_skip_other_paths: bool


def _mapped_bbox(coord_space: SourceCoordinateSpace, bbox: tuple[int, int, int, int], target_shape: tuple[int, int], pad: int = 6) -> list[int]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    mapped = []
    for px, py in pts:
        try:
            mapped.append(coord_space.map_point(px, py))
        except ValueError:
            continue
    if not mapped:
        return []
    th, tw = target_shape
    xs = [p[0] for p in mapped]; ys = [p[1] for p in mapped]
    rx0 = max(0, int(math.floor(min(xs))) - int(pad))
    ry0 = max(0, int(math.floor(min(ys))) - int(pad))
    rx1 = min(int(tw), int(math.ceil(max(xs))) + int(pad))
    ry1 = min(int(th), int(math.ceil(max(ys))) + int(pad))
    return [rx0, ry0, rx1, ry1]


def _append_manual_effect_candidate(rows: list[dict], *, source_bbox: tuple[int, int, int, int], target_bbox: list[int], reason: str, hint_backend: str = '', source_hint: bool = False, compact_components: int = 0, compact_ratio: float = 0.0, spiky: bool = False) -> None:
    if len(target_bbox) != 4:
        return
    x0, y0, x1, y1 = [int(v) for v in target_bbox]
    if x1 <= x0 or y1 <= y0:
        return
    new_area = max(1, (x1 - x0) * (y1 - y0))
    replacement_index = -1
    for i, row in enumerate(rows):
        box = list(row.get('target_bbox') or [])
        if len(box) != 4:
            continue
        bx0, by0, bx1, by1 = [int(v) for v in box]
        ix0, iy0 = max(x0, bx0), max(y0, by0)
        ix1, iy1 = min(x1, bx1), min(y1, by1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        old_area = max(1, (bx1 - bx0) * (by1 - by0))
        iou = inter / max(1, new_area + old_area - inter)
        cover = inter / max(1, min(new_area, old_area))
        if iou >= 0.52 or cover >= 0.72:
            # Same physical omission discovered by contour + detector hint.
            # Keep the tighter region, and prefer text-seeded evidence when the
            # areas are comparable. This prevents the GUI from asking the user
            # to repair the same purple/pink effect twice.
            old_hint = bool(row.get('source_hint'))
            prefer_new = (new_area < old_area * 0.92) or (bool(source_hint) and not old_hint and new_area <= old_area * 1.15)
            if not prefer_new:
                return
            replacement_index = i
            break
    payload = {
        'source_bbox': [int(v) for v in source_bbox],
        'target_bbox': [x0, y0, x1, y1],
        'reason': reason,
        'review_level': 'recommended',
        'editable': True,
        'restorable': False,
        'workflow': 'manual_effect',
        'suggested_manual_mode': 'reveal_text',
        'source_hint': bool(source_hint),
        'source_hint_backend': str(hint_backend or ''),
        'compact_components': int(compact_components),
        'compact_ratio': float(compact_ratio),
        'spiky': bool(spiky),
        # Only high-confidence text-seeded coloured rejects are safe to prefill
        # automatically in the editor. Raw contour/artwork rejects remain in
        # diagnostics for transparency but do not become clickable regions.
        'auto_actionable': bool(
            str(reason).startswith('colored_')
            and bool(source_hint)
            and int(compact_components) >= 5
            and float(compact_ratio) >= 0.030
        ),
    }
    if replacement_index >= 0:
        rows[replacement_index] = payload
    else:
        rows.append(payload)




def _bbox_iou(a: list[int] | tuple[int, int, int, int], b: list[int] | tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = [int(v) for v in a]
    bx0, by0, bx1, by1 = [int(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    area_a = max(1.0, float((ax1 - ax0) * (ay1 - ay0)))
    area_b = max(1.0, float((bx1 - bx0) * (by1 - by0)))
    return inter / max(1.0, area_a + area_b - inter)


def _append_manual_effect_suggestion(
    rows: list[dict],
    *,
    target_bbox: list[int] | tuple[int, int, int, int],
    source_bbox: list[int] | tuple[int, int, int, int],
    reason: str,
    detail: str,
    colored: bool,
    source_hint: bool,
    confidence: float = 0.0,
) -> None:
    box = [int(v) for v in target_bbox]
    src_box = [int(v) for v in source_bbox]
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        return
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area < 100:
        return
    for existing in rows:
        prev = list(existing.get("target_bbox", []) or [])
        if len(prev) == 4 and _bbox_iou(prev, box) >= 0.72:
            return
    rows.append({
        "id": f"auto-manual-effect-{len(rows):03d}",
        "target_bbox": box,
        "source_bbox": src_box,
        "suggested_mode": "reveal_text",
        "diff_threshold": 24,
        "expand_px": 2,
        "feather_px": 0,
        "auto_clear_target": True,
        "reason": str(reason),
        "detail": str(detail),
        "candidate_kind": "colored_open_text" if colored else "manual_review",
        "source_hint": bool(source_hint),
        "confidence": float(confidence),
        "ui_label": "彩底/开放式文字建议走『擦除显字』" if colored else "建议人工补漏复核",
        "origin": "direct_patch_auto_suggestion",
    })

def _uniform_page_scale(H: np.ndarray) -> tuple[float, float, float, float]:
    A = np.asarray(H[:2, :2], dtype=np.float64)
    # Singular values expose anisotropic scan stretch without being confused by
    # a small rotation.  The geometric mean preserves area with one scalar.
    sv = np.linalg.svd(A, compute_uv=False)
    s_hi, s_lo = float(max(sv)), float(min(sv))
    scale = math.sqrt(max(1e-12, abs(float(np.linalg.det(A)))))
    anisotropy = abs(s_hi - s_lo) / max(1e-9, 0.5 * (s_hi + s_lo))
    angle = math.degrees(math.atan2(float(A[1, 0]), float(A[0, 0])))
    perspective = max(abs(float(H[2, 0])), abs(float(H[2, 1])))
    return scale, anisotropy, angle, perspective



def _homography_local_similarity(H: np.ndarray, x: float, y: float) -> tuple[float, float, float]:
    """Return raster-safe local scale/rotation from a page mapping.

    The full registration may be affine or homographic because that is useful for
    *locating* corresponding points on scans with crop/aspect/perspective drift.
    Final Chinese pixels must never inherit that anisotropic/sheared deformation,
    so we project the local Jacobian to its nearest orientation-preserving
    similarity transform (one scalar scale + rotation).
    """
    h = np.asarray(H, dtype=np.float64)
    den = float(h[2, 0] * x + h[2, 1] * y + h[2, 2])
    if abs(den) < 1e-12:
        return 1.0, 0.0, 1.0
    nu = float(h[0, 0] * x + h[0, 1] * y + h[0, 2])
    nv = float(h[1, 0] * x + h[1, 1] * y + h[1, 2])
    den2 = den * den
    J = np.array([
        [(float(h[0, 0]) * den - float(h[2, 0]) * nu) / den2,
         (float(h[0, 1]) * den - float(h[2, 1]) * nu) / den2],
        [(float(h[1, 0]) * den - float(h[2, 0]) * nv) / den2,
         (float(h[1, 1]) * den - float(h[2, 1]) * nv) / den2],
    ], dtype=np.float64)
    try:
        U, singular, Vt = np.linalg.svd(J)
    except np.linalg.LinAlgError:
        return 1.0, 0.0, 1.0
    R = U @ Vt
    if float(np.linalg.det(R)) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    s_hi, s_lo = float(max(singular)), float(min(singular))
    scale = math.sqrt(max(1e-12, abs(float(np.linalg.det(J)))))
    anisotropy = abs(s_hi - s_lo) / max(1e-9, 0.5 * (s_hi + s_lo))
    angle = math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))
    return float(scale), float(angle), float(anisotropy)


def _warp_similarity_patch(
    crop: np.ndarray,
    crop_gray: np.ndarray,
    crop_mask: np.ndarray,
    crop_boundary: np.ndarray,
    crop_origin: tuple[int, int],
    source_center: tuple[float, float],
    target_center: tuple[float, float],
    scale: float,
    angle_deg: float,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Warp one source container by a shape-preserving local similarity only."""
    sx0, sy0 = crop_origin
    cx, cy = source_center
    tcx, tcy = target_center
    ch, cw = crop.shape[:2]
    # A pure integer translation needs no interpolation at all. This matters for
    # small CJK strokes: even Lanczos on an otherwise identity transform can add a
    # faint halo when registration carries a fractional offset. Preserve original
    # source pixels whenever the requested transform is raster-exact.
    if abs(float(scale) - 1.0) <= 1e-9 and abs(float(angle_deg)) <= 1e-9:
        dx = float(target_center[0] - source_center[0])
        dy = float(target_center[1] - source_center[1])
        rdx, rdy = int(round(dx)), int(round(dy))
        if abs(dx - rdx) <= 1e-9 and abs(dy - rdy) <= 1e-9:
            return (
                int(sx0 + rdx), int(sy0 + rdy),
                crop.copy(), crop_gray.copy(), crop_mask.copy(), crop_boundary.copy(),
            )
    rad = math.radians(float(angle_deg))
    c, s = math.cos(rad), math.sin(rad)
    A = float(scale) * np.array([[c, -s], [s, c]], dtype=np.float64)
    local_center = np.array([float(cx - sx0), float(cy - sy0)], dtype=np.float64)
    corners = np.array([[0.0, 0.0], [float(cw), 0.0], [float(cw), float(ch)], [0.0, float(ch)]], dtype=np.float64)
    mapped = (corners - local_center) @ A.T + np.array([tcx, tcy], dtype=np.float64)
    base_x = int(math.floor(float(mapped[:, 0].min())))
    base_y = int(math.floor(float(mapped[:, 1].min())))
    max_x = int(math.ceil(float(mapped[:, 0].max())))
    max_y = int(math.ceil(float(mapped[:, 1].max())))
    out_w = max(1, max_x - base_x + 1)
    out_h = max(1, max_y - base_y + 1)
    translate = np.array([tcx - base_x, tcy - base_y], dtype=np.float64) - A @ local_center
    M = np.array([
        [A[0, 0], A[0, 1], translate[0]],
        [A[1, 0], A[1, 1], translate[1]],
    ], dtype=np.float64)
    interp = cv2.INTER_AREA if scale < 0.92 and abs(angle_deg) < 1e-6 else cv2.INTER_LANCZOS4
    warped_crop = cv2.warpAffine(
        crop, M, (out_w, out_h), flags=interp,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    warped_gray = cv2.warpAffine(
        crop_gray, M, (out_w, out_h), flags=interp,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    warped_mask = cv2.warpAffine(
        crop_mask, M, (out_w, out_h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_boundary = cv2.warpAffine(
        crop_boundary, M, (out_w, out_h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return base_x, base_y, warped_crop, warped_gray, warped_mask, warped_boundary


def _boundary_alignment_score(
    boundary: np.ndarray,
    base_x: int,
    base_y: int,
    target_edge_distance: np.ndarray,
    search: int,
    *,
    coarse_step: int = 1,
    around: tuple[int, int] | None = None,
    around_radius: int | None = None,
) -> tuple[float, int, int]:
    """Find translation that best overlays a source boundary on target dark lines."""
    th, tw = target_edge_distance.shape[:2]
    bys, bxs = np.where(boundary > 0)
    if len(bxs) > 1200:
        take = np.linspace(0, len(bxs) - 1, 1200).astype(np.int32)
        bxs = bxs[take]
        bys = bys[take]
    if len(bxs) < 24:
        return 0.0, 0, 0
    step = max(1, int(coarse_step))
    if around is None:
        xs = range(-search, search + 1, step)
        ys = range(-search, search + 1, step)
    else:
        rr = max(1, int(around_radius if around_radius is not None else step))
        xs = range(max(-search, int(around[0]) - rr), min(search, int(around[0]) + rr) + 1)
        ys = range(max(-search, int(around[1]) - rr), min(search, int(around[1]) + rr) + 1)
    best_score = float("inf")
    best_dx = best_dy = 0
    for dy in ys:
        for dx in xs:
            xx = bxs + base_x + dx
            yy = bys + base_y + dy
            ok = (xx >= 0) & (xx < tw) & (yy >= 0) & (yy < th)
            if int(np.count_nonzero(ok)) < 24:
                continue
            score = float(np.mean(np.minimum(target_edge_distance[yy[ok], xx[ok]], 12.0)))
            if score < best_score:
                best_score = score
                best_dx, best_dy = int(dx), int(dy)
    return best_score, best_dx, best_dy

def _compact_ink_stats(gray: np.ndarray, inner: np.ndarray, bw: int, bh: int) -> tuple[int, int, float]:
    """Character-like ink statistics, cropped to the candidate instead of full page.

    v0.8.30 ran connectedComponentsWithStats on a page-sized bitmap once per
    contour. Cropping first is equivalent for component geometry but makes the
    source-direct route substantially cheaper on 2K/4K pages.
    """
    ys, xs = np.where(inner > 0)
    if len(xs) == 0:
        return 0, 0, 0.0
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    local_inner = inner[y0:y1, x0:x1] > 0
    local_gray = gray[y0:y1, x0:x1]
    ink = ((local_gray < 185) & local_inner).astype(np.uint8)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    total = max(1, int(np.count_nonzero(local_inner)))
    compact = 0
    compact_area = 0
    for j in range(1, count):
        _x, _y, ww, hh, aa = [int(v) for v in stats[j]]
        if aa < 2 or aa > max(500, int(total * 0.035)):
            continue
        if ww > 0.38 * bw or hh > 0.38 * bh:
            continue
        if max(ww / max(1.0, hh), hh / max(1.0, ww)) > 9.0:
            continue
        compact += 1
        compact_area += aa
    return compact, compact_area, float(compact_area / total)


def _estimate_outline_thickness(
    gray: np.ndarray, filled: np.ndarray, *, dark_threshold: int, max_px: int, dark_ratio_floor: float
) -> tuple[int, list[float]]:
    """Estimate source outline thickness on a tight ROI, not the whole page."""
    pts = cv2.findNonZero((filled > 0).astype(np.uint8))
    if pts is None:
        return 1, []
    x, y, w, h = cv2.boundingRect(pts)
    pad = max(2, int(max_px) + 2)
    x0=max(0,x-pad); y0=max(0,y-pad); x1=min(filled.shape[1],x+w+pad); y1=min(filled.shape[0],y+h+pad)
    local_filled=(filled[y0:y1,x0:x1] > 0).astype(np.uint8)
    local_gray=gray[y0:y1,x0:x1]
    dist=cv2.distanceTransform(local_filled, cv2.DIST_L2, 5)
    dark=local_gray < int(dark_threshold)
    ratios: list[float]=[]
    thickness=1
    for d in range(1,max(1,int(max_px))+1):
        ring=(dist >= d-0.45) & (dist < d+0.55)
        n=int(np.count_nonzero(ring))
        if n < 16:
            break
        ratio=float(np.mean(dark[ring]))
        ratios.append(ratio)
        if ratio >= float(dark_ratio_floor):
            thickness=d
        elif d > 1:
            prev=ratios[-2] if len(ratios) >= 2 else ratio
            if prev < float(dark_ratio_floor):
                break
    return max(1,int(thickness)), ratios


def _progressive_transfer_mask(
    gray: np.ndarray, filled: np.ndarray, base_inset: int, cfg
) -> tuple[np.ndarray, int, dict]:
    """Pick the largest safe interior using ROI-local progressive erosion.

    The fitting policy is PanelCleaner-inspired, while the implementation is
    independent and optimized for source-direct transfer: expensive morphology
    and distance transforms run only around one container, never on the 2K/4K
    page-sized bitmap for every contour.
    """
    dynamic=bool(getattr(cfg,"source_direct_dynamic_border_enabled",True))
    if dynamic:
        border,ring_ratios=_estimate_outline_thickness(
            gray,filled,
            dark_threshold=int(getattr(cfg,"source_direct_outline_dark_threshold",160)),
            max_px=int(getattr(cfg,"source_direct_dynamic_border_max_px",10)),
            dark_ratio_floor=float(getattr(cfg,"source_direct_dynamic_border_dark_ratio",0.22)),
        )
        start=max(1,int(base_inset),int(border)+1)
    else:
        border,ring_ratios=max(1,int(base_inset)-1),[]
        start=max(1,int(base_inset))

    steps=max(1,int(getattr(cfg,"source_direct_progressive_inset_steps",4)))
    max_outer_dark=float(getattr(cfg,"source_direct_progressive_max_outer_dark_ratio",0.22))
    pts=cv2.findNonZero((filled > 0).astype(np.uint8))
    if pts is None:
        return np.zeros_like(filled),start,{"estimated_border_px":int(border),"ring_dark_ratios":ring_ratios,"trials":[]}
    x,y,w,h=cv2.boundingRect(pts)
    max_inset=start+steps+2
    x0=max(0,x-max_inset); y0=max(0,y-max_inset); x1=min(filled.shape[1],x+w+max_inset); y1=min(filled.shape[0],y+h+max_inset)
    local_filled=filled[y0:y1,x0:x1]
    local_gray=gray[y0:y1,x0:x1]

    best: tuple[np.ndarray,int,float,float] | None=None
    trials=[]
    for inset in range(start,start+steps):
        ksize=inset*2+1
        local_mask=cv2.erode(local_filled,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize)))
        count=int(cv2.countNonZero(local_mask))
        if count < 150:
            continue
        inner_dist=cv2.distanceTransform((local_mask > 0).astype(np.uint8),cv2.DIST_L2,3)
        outer_band=(local_mask > 0) & (inner_dist <= 2.0)
        outer_dark=float(np.mean(local_gray[outer_band] < 180)) if np.any(outer_band) else 1.0
        white=float(np.mean(local_gray[local_mask > 0] > 225))
        trials.append({"inset":int(inset),"outer_dark":outer_dark,"white":white,"pixels":count})
        if outer_dark <= max_outer_dark:
            full=np.zeros_like(filled); full[y0:y1,x0:x1]=local_mask
            return full,int(inset),{
                "estimated_border_px":int(border),"ring_dark_ratios":ring_ratios,
                "trials":trials,"selected_outer_dark":outer_dark,"selected_white":white,
                "roi":[int(x0),int(y0),int(x1),int(y1)],
            }
        score=outer_dark-0.15*white
        if best is None or score < best[2]:
            best=(local_mask.copy(),int(inset),score,outer_dark)
    if best is None:
        return np.zeros_like(filled),int(start),{"estimated_border_px":int(border),"ring_dark_ratios":ring_ratios,"trials":trials}
    full=np.zeros_like(filled); full[y0:y1,x0:x1]=best[0]
    return full,best[1],{
        "estimated_border_px":int(border),"ring_dark_ratios":ring_ratios,
        "trials":trials,"selected_outer_dark":float(best[3]),"fallback_selection":True,
        "roi":[int(x0),int(y0),int(x1),int(y1)],
    }


# Expose the active PanelCleaner-inspired progressive border fitter as a real
# provider. The source-direct path resolves it through the registry below.
PROVIDER_REGISTRY.register("mask_refiner", "progressive_border", _progressive_transfer_mask, replace=True)


def _colored_text_component_refiner(raw_clear: np.ndarray, use_mask: np.ndarray, cfg) -> tuple[np.ndarray, dict]:
    """Refine a dark-pixel clear mask using local connected components.

    Adapted to the source-direct use case from Cotrans' component-to-text mask
    refinement idea: we do not need OCR boxes because the enclosing container is
    already known. Very large dark components are treated as decoration/outline,
    compact components are retained, and dilation is applied only afterwards.
    """
    use = use_mask > 0
    raw = ((raw_clear > 0) & use).astype(np.uint8) * 255
    if not bool(getattr(cfg, "source_direct_colored_component_refine_enabled", True)):
        return raw, {"component_refine": False, "components_kept": None, "components_rejected": None}
    n, labels, stats, _ = cv2.connectedComponentsWithStats((raw > 0).astype(np.uint8), 8)
    container_area = max(1, int(np.count_nonzero(use)))
    min_area = max(1, int(getattr(cfg, "source_direct_colored_component_min_area_px", 2)))
    max_area = max(min_area, int(round(container_area * float(getattr(cfg, "source_direct_colored_component_max_area_ratio", 0.12)))))
    refined = np.zeros_like(raw)
    kept = rejected = 0
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            refined[labels == lab] = 255
            kept += 1
        else:
            rejected += 1
    # Never turn a valid non-empty clear mask into an empty mask solely because
    # JPEG/print strokes happened to connect. Fall back conservatively.
    if kept == 0 and cv2.countNonZero(raw) > 0:
        refined = raw
    refined[~use] = 0
    return refined, {
        "component_refine": True,
        "components_kept": int(kept),
        "components_rejected": int(rejected),
        "max_component_area_px": int(max_area),
    }


PROVIDER_REGISTRY.register("mask_refiner", "colored_text_components", _colored_text_component_refiner, replace=True)


def _colored_transfer_preserving_fill(
    target_region: np.ndarray, source_region: np.ndarray, source_gray: np.ndarray, mask: np.ndarray, cfg
) -> tuple[np.ndarray, dict]:
    """Clear target lettering and lay source Chinese ink over the original colour fill.

    No OCR/text recognition is involved.  The container interior itself is the
    safety envelope; dark target glyph pixels are inpainted only inside it, then
    source ink alpha derived from the already-registered translated scan is
    composited.  The target outline and saturated background therefore stay HD.
    """
    use = mask > 0
    out = target_region.copy()
    if not np.any(use):
        return out, {"cleared_target_pixels": 0, "source_ink_pixels": 0}

    target_hsv = cv2.cvtColor(target_region, cv2.COLOR_BGR2HSV)
    target_value = target_hsv[..., 2]
    # Use HSV value rather than grayscale luminance: a saturated red burst has
    # low grayscale intensity but high value and must never be mistaken for black
    # Japanese lettering.
    clear = ((target_value < int(getattr(cfg, "source_direct_colored_clear_dark_threshold", 185))) & use).astype(np.uint8) * 255
    component_refiner = PROVIDER_REGISTRY.get("mask_refiner", "colored_text_components") or _colored_text_component_refiner
    clear, component_diag = component_refiner(clear, mask, cfg)

    # v1.3: verified dark components are only the glyph cores. Anti-aliased
    # Japanese edges on vivid fills often have high value and survived the old
    # ``value < threshold`` test. Grow only around those verified components and
    # only into low-saturation pixels that are darker than the local fill.
    antialias_added = 0
    aa_r = max(0, int(getattr(cfg, "source_direct_colored_antialias_expand_px", 2)))
    if aa_r > 0 and cv2.countNonZero(clear) > 0:
        halo = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (aa_r * 2 + 1, aa_r * 2 + 1))) > 0
        sat = target_hsv[..., 1]
        bg = use & (clear == 0) & (target_value >= int(getattr(cfg, "source_direct_colored_clear_dark_threshold", 185)))
        bg_value = float(np.median(target_value[bg])) if int(np.count_nonzero(bg)) >= 16 else float(np.median(target_value[use]))
        margin = max(4, int(getattr(cfg, "source_direct_colored_antialias_value_margin", 10)))
        sat_max = int(getattr(cfg, "source_direct_colored_antialias_max_saturation", 92))
        fringe = halo & use & (clear == 0) & (target_value <= max(0.0, bg_value - margin)) & (sat <= sat_max)
        antialias_added = int(np.count_nonzero(fringe))
        clear[fringe] = 255

    grow = max(0, int(getattr(cfg, "source_direct_colored_clear_dilate_px", 2)))
    if grow > 0:
        clear = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1)))
        clear[~use] = 0
    cleared = int(cv2.countNonZero(clear))
    if cleared > 0:
        out = cv2.inpaint(out, clear, float(getattr(cfg, "source_direct_colored_inpaint_radius", 3.0)), cv2.INPAINT_TELEA)
        # Telea can pull a few black outline pixels inward near sharp spikes.
        # Coloured dialogue/SFX containers are predominantly solid fill, so use
        # the original high-value interior as a conservative fallback for only
        # those still-dark cleared pixels. This never touches the protected border.
        bg_use = use & (target_value >= max(200, int(getattr(cfg, "source_direct_colored_clear_dark_threshold", 185)) + 10))
        if int(np.count_nonzero(bg_use)) >= 16:
            bg = np.median(target_region[bg_use].astype(np.float32), axis=0)
            out_value = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[..., 2]
            bad = (clear > 0) & (out_value < int(getattr(cfg, "source_direct_colored_clear_dark_threshold", 185)))
            out[bad] = np.clip(bg, 0, 255).astype(np.uint8)

    ink_threshold = int(getattr(cfg, "source_direct_source_ink_threshold", 215))
    hard_ink = ((source_gray < ink_threshold) & use).astype(np.uint8) * 255
    ink_grow = max(0, int(getattr(cfg, "source_direct_source_ink_dilate_px", 1)))
    # Continuous alpha preserves antialiasing/JPEG edge shades. Newly grown edge
    # pixels borrow the strongest neighbouring alpha rather than becoming opaque.
    paper = float(np.percentile(source_gray[use], 82.0)) if np.any(use) else 255.0
    denom = max(40.0, paper - 25.0)
    alpha = np.clip((paper - source_gray.astype(np.float32)) / denom, 0.0, 1.0)
    alpha *= (hard_ink > 0).astype(np.float32)
    if ink_grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ink_grow * 2 + 1, ink_grow * 2 + 1))
        alpha = cv2.dilate(alpha, k)
    gamma = max(0.2, float(getattr(cfg, "source_direct_source_ink_alpha_gamma", 0.92)))
    alpha = np.power(np.clip(alpha, 0.0, 1.0), gamma)
    alpha[~use] = 0.0
    a3 = alpha[..., None]
    out = np.clip(source_region.astype(np.float32) * a3 + out.astype(np.float32) * (1.0 - a3), 0, 255).astype(np.uint8)
    final_hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    final_value = final_hsv[..., 2]
    residual_probe = (clear > 0) & (alpha < 0.08)
    residual_dark = int(np.count_nonzero(residual_probe & (final_value < int(getattr(cfg, "source_direct_colored_clear_dark_threshold", 185)))))
    residual_ratio = float(residual_dark / max(1, int(np.count_nonzero(residual_probe))))

    # A second, tightly bounded sweep catches fringe that sat just outside the
    # first clear mask. It is explicitly forbidden from touching SOURCE Chinese
    # alpha support, so manual/automatic cleanup cannot erase translated ink.
    residual_cleanup = np.zeros_like(clear)
    if bool(getattr(cfg, "source_direct_colored_residual_cleanup_enabled", True)) and cv2.countNonZero(clear) > 0:
        rr = max(1, int(getattr(cfg, "source_direct_colored_residual_expand_px", 2)))
        near = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rr * 2 + 1, rr * 2 + 1))) > 0
        sat = target_hsv[..., 1]
        bg = use & (clear == 0) & (target_value >= int(getattr(cfg, "source_direct_colored_clear_dark_threshold", 185)))
        bg_value = float(np.median(target_value[bg])) if int(np.count_nonzero(bg)) >= 16 else float(np.median(target_value[use]))
        margin = max(4, int(getattr(cfg, "source_direct_colored_antialias_value_margin", 10)))
        sat_max = int(getattr(cfg, "source_direct_colored_antialias_max_saturation", 92))
        residual = near & use & (alpha < 0.08) & (target_value <= max(0.0, bg_value - margin)) & (sat <= sat_max) & (final_value <= max(0.0, bg_value - margin * 0.5))
        residual_cleanup[residual] = 255
        if cv2.countNonZero(residual_cleanup) > 0:
            out = cv2.inpaint(out, residual_cleanup, float(getattr(cfg, "source_direct_colored_inpaint_radius", 3.0)), cv2.INPAINT_TELEA)
            final_value = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[..., 2]
            residual_dark = int(np.count_nonzero(residual_probe & (final_value < int(getattr(cfg, "source_direct_colored_clear_dark_threshold", 185)))))
            residual_ratio = float(residual_dark / max(1, int(np.count_nonzero(residual_probe))))
    return out, {
        "cleared_target_pixels": cleared,
        "source_ink_pixels": int(np.count_nonzero(alpha > 0.05)),
        "source_paper_level": paper,
        "target_residual_ratio": residual_ratio,
        "colored_antialias_added_pixels": int(antialias_added),
        "colored_residual_cleanup_pixels": int(cv2.countNonZero(residual_cleanup)),
        "colored_fill_preserved": True,
        "mask_refiner": "colored_text_components",
        **component_diag,
    }


def _ink_change_score(source_gray: np.ndarray, target_gray: np.ndarray, mask: np.ndarray) -> float:
    use = mask > 0
    if not np.any(use):
        return 0.0
    source_ink = ((source_gray < 185) & use).astype(np.uint8) * 255
    target_ink = ((target_gray < 185) & use).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    source_d = cv2.dilate(source_ink, kernel) > 0
    target_d = cv2.dilate(target_ink, kernel) > 0
    sn = int(cv2.countNonZero(source_ink)); tn = int(cv2.countNonZero(target_ink))
    if sn + tn == 0:
        return 0.0
    matched = (
        int(np.count_nonzero((source_ink > 0) & target_d))
        + int(np.count_nonzero((target_ink > 0) & source_d))
    )
    return float(np.clip(1.0 - matched / max(1, sn + tn), 0.0, 1.0))


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand_white_gap_mask(
    local_mask: np.ndarray,
    source_gray: np.ndarray,
    target_gray: np.ndarray,
    target_edge_distance: np.ndarray,
    cfg,
) -> tuple[np.ndarray, dict]:
    """Slightly enlarge a white-container direct mask to swallow target-only residue.

    The direct path already excludes the bubble outline via the source-side border
    inset.  Real editions can still differ by 1-3 pixels in the interior size,
    which leaves a narrow untouched ring where Japanese ink survives.  Grow only
    into pixels that look like source paper and are not sitting on a strong target
    edge.  Dark target pixels are allowed here on purpose: they are often the
    residual Japanese strokes we are trying to clear with source white.
    """
    m = (local_mask > 0).astype(np.uint8)
    if not bool(getattr(cfg, "source_direct_white_gap_fill_enabled", True)):
        return m, {"enabled": False, "iterations": 0, "added_pixels": 0}
    if m.shape != source_gray.shape or m.shape != target_gray.shape or m.shape != target_edge_distance.shape:
        return m, {"enabled": True, "shape_mismatch": True, "iterations": 0, "added_pixels": 0}
    max_px = max(0, int(getattr(cfg, "source_direct_white_gap_fill_max_px", 3)))
    if max_px <= 0:
        return m, {"enabled": True, "iterations": 0, "added_pixels": 0}
    source_white_thr = int(getattr(cfg, "source_direct_white_gap_fill_source_white_threshold", 238))
    edge_floor = float(getattr(cfg, "source_direct_white_gap_fill_target_edge_distance_px", 1.35))
    target_value_min = int(getattr(cfg, "source_direct_white_gap_fill_target_value_min", 120))
    target_black_thr = int(getattr(cfg, "source_direct_white_gap_fill_target_black_threshold", 170))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = m.copy()
    before = int(np.count_nonzero(out))
    iterations = 0
    for _ in range(max_px):
        dil = cv2.dilate(out, kernel)
        ring = (dil > 0) & (out == 0)
        if not np.any(ring):
            break
        allowed = (
            ring
            & (source_gray >= source_white_thr)
            & (target_edge_distance >= edge_floor)
            & ((target_gray >= target_value_min) | (target_gray <= target_black_thr))
        )
        added = int(np.count_nonzero(allowed))
        if added == 0:
            break
        out[allowed] = 1
        iterations += 1
    after = int(np.count_nonzero(out))
    return out, {
        "enabled": True,
        "iterations": int(iterations),
        "added_pixels": int(max(0, after - before)),
        "source_white_threshold": int(source_white_thr),
        "target_edge_distance_floor": float(edge_floor),
    }



def _protect_registered_geometry_boundary(
    geometry: np.ndarray,
    target_edge_distance: np.ndarray,
    cfg,
) -> tuple[np.ndarray, dict]:
    """Protect only *container-border* target lines during geometry snap.

    The old implementation removed every pixel sitting on a dark TARGET edge from
    the write geometry.  That also removed Japanese glyph strokes inside a white
    balloon, so Direct Patch could leave Japanese text underneath the Chinese.
    Restrict the strong-edge guard to a narrow band around the proposed container
    boundary; internal dark lettering remains writable and is correctly cleared.
    """
    geom = geometry.astype(bool).copy()
    if geom.shape != target_edge_distance.shape or not np.any(geom):
        return geom, {"protected_boundary_pixels": 0, "internal_dark_writable_pixels": 0}
    edge_floor = float(getattr(cfg, "source_direct_geometry_snap_edge_distance_px", 0.75))
    band_px = max(1, int(getattr(cfg, "source_direct_geometry_snap_boundary_guard_px", 3)))
    g8 = geom.astype(np.uint8) * 255
    boundary = cv2.morphologyEx(g8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if band_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
        boundary = cv2.dilate(boundary.astype(np.uint8), k) > 0
    strong_target = target_edge_distance < edge_floor
    protected = geom & boundary & strong_target
    internal_dark = geom & (~boundary) & strong_target
    geom[protected] = False
    return geom, {
        "protected_boundary_pixels": int(np.count_nonzero(protected)),
        "internal_dark_writable_pixels": int(np.count_nonzero(internal_dark)),
        "boundary_guard_px": int(band_px),
        "edge_distance_px": float(edge_floor),
    }



def _constrain_direct_write_to_target_border(
    local_mask: np.ndarray,
    registered_geometry: np.ndarray,
    target_edge_distance: np.ndarray,
    cfg,
    target_region: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Clamp Direct writes inside the TARGET container and protect its outline.

    SOURCE border fitting and TARGET border preservation solve different failure
    modes.  The source-side progressive inset stops an old scan outline from
    being interpreted as translated ink.  This final target-side guard then
    shrinks the actual writable envelope a small amount on all sides and removes
    strong TARGET line pixels near the registered container boundary.

    Returns ``(safe_mask, protected_mask, diagnostics)``.  ``protected_mask`` is
    the byte-exact restore ring used after text transfer as a final invariant.
    Internal Japanese glyph strokes remain writable because strong-edge
    protection is restricted to the container-boundary band.
    """
    use = (local_mask > 0).astype(np.uint8)
    geom = (registered_geometry > 0).astype(np.uint8)
    if use.shape != geom.shape or use.shape != target_edge_distance.shape:
        return use.astype(bool), np.zeros_like(use, dtype=bool), {
            "enabled": bool(getattr(cfg, "source_direct_target_border_guard_enabled", True)),
            "shape_mismatch": True,
            "input_pixels": int(np.count_nonzero(use)),
            "output_pixels": int(np.count_nonzero(use)),
            "protected_pixels": 0,
        }
    if not bool(getattr(cfg, "source_direct_target_border_guard_enabled", True)) or not np.any(use):
        return use.astype(bool), np.zeros_like(use, dtype=bool), {
            "enabled": False,
            "input_pixels": int(np.count_nonzero(use)),
            "output_pixels": int(np.count_nonzero(use)),
            "protected_pixels": 0,
        }

    # If registration geometry is unavailable/empty, the already-fitted Direct
    # mask is the best available container envelope.  Never expand beyond it.
    if not np.any(geom):
        geom = use.copy()

    inset = max(0, int(getattr(cfg, "source_direct_target_border_inset_px", 2)))
    safe_geom = geom.copy()
    if inset > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1))
        eroded = cv2.erode(geom * 255, k) > 0
        # Tiny containers should not vanish because of a fixed inset. Fall back
        # to the un-eroded geometry if erosion leaves too little useful area.
        if int(np.count_nonzero(eroded)) >= max(80, int(np.count_nonzero(use) * 0.35)):
            safe_geom = eroded.astype(np.uint8)

    guard_px = max(1, int(getattr(cfg, "source_direct_target_border_guard_px", 3)))
    edge_floor = float(getattr(cfg, "source_direct_target_border_edge_distance_px", 1.0))
    boundary = cv2.morphologyEx(geom * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if guard_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * guard_px + 1, 2 * guard_px + 1))
        boundary_band = cv2.dilate(boundary.astype(np.uint8), k) > 0
    else:
        boundary_band = boundary
    strong_target_line = target_edge_distance < edge_floor
    # Edge distance alone also fires on Japanese glyphs touching the container
    # margin.  Protect/restore only *structural* long TARGET rules/outlines; short
    # glyph fragments must remain writable so full-clear can actually remove them.
    if target_region is not None and target_region.shape[:2] == use.shape:
        geom_for_border = cv2.dilate(
            geom.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, guard_px) + 1, 2 * max(1, guard_px) + 1)),
            iterations=1,
        )
        structural_line = target_container_border_mask(
            target_region, geom_for_border, band_px=max(3, guard_px)
        ) > 0
        line_protected = boundary_band & strong_target_line & structural_line
    else:
        # Backward-compatible/synthetic fallback when raw TARGET pixels are not
        # available to the helper. Production calls always pass target_region.
        structural_line = boundary_band & strong_target_line
        line_protected = structural_line.copy()

    # Actual replacement mask is always inside the inset target geometry.  This
    # is the deterministic "shrink from all four sides" rule for rectangular
    # narration boxes and the equivalent inward shrink for curved balloons.
    safe = (use > 0) & (safe_geom > 0)
    safe[line_protected] = False

    # Only actual TARGET line pixels are byte-exact restore authority. The generic
    # inset ring is a *write* restriction, not a restore mask: restoring the whole
    # ring can resurrect edge-adjacent Japanese punctuation after a full clear.
    protected = line_protected.copy()
    return safe, protected, {
        "enabled": True,
        "input_pixels": int(np.count_nonzero(use)),
        "output_pixels": int(np.count_nonzero(safe)),
        "removed_pixels": int(np.count_nonzero((use > 0) & ~safe)),
        "protected_pixels": int(np.count_nonzero(protected)),
        "line_protected_pixels": int(np.count_nonzero(line_protected)),
        "structural_line_pixels": int(np.count_nonzero(structural_line)),
        "target_inset_px": int(inset),
        "boundary_guard_px": int(guard_px),
        "edge_distance_px": float(edge_floor),
    }


def _restore_damaged_target_structural_lines(
    candidate: np.ndarray,
    target_before: np.ndarray,
    clear_geometry: np.ndarray,
    cfg,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Restore TARGET balloon/rule pixels that a white-container clear damaged.

    The ordinary pre-write border detector is intentionally conservative so it
    does not mistake edge-adjacent Japanese punctuation for a balloon outline.
    That can miss a broken/partial oval segment.  This post-pass has a stronger
    fact available: it only inspects TARGET dark pixels that the candidate has
    *actually lightened*.  Within a narrow outer strip of the registered clear
    geometry it restores long/thin structural components byte-exact from TARGET.

    Central Japanese text is excluded by the outer-strip gate, while a missing
    left/right/top/bottom bubble edge remains recoverable even when the source
    edition does not contain that exact line segment.
    """
    geom = (np.asarray(clear_geometry) > 0).astype(np.uint8)
    empty = np.zeros(geom.shape, np.uint8)
    diag = {
        "enabled": bool(getattr(cfg, "direct_post_structural_restore_enabled", True)),
        "candidate_lost_dark_pixels": 0,
        "restored_components": 0,
        "restored_core_pixels": 0,
        "restored_pixels": 0,
        "reason": "ok",
    }
    if not diag["enabled"]:
        diag["reason"] = "disabled"
        return candidate, empty, diag
    if candidate.shape != target_before.shape or candidate.shape[:2] != geom.shape or not np.any(geom):
        diag["reason"] = "shape_or_geometry_invalid"
        return candidate, empty, diag

    tg = cv2.cvtColor(target_before, cv2.COLOR_BGR2GRAY) if target_before.ndim == 3 else target_before.astype(np.uint8)
    cg = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY) if candidate.ndim == 3 else candidate.astype(np.uint8)
    changed = np.any(candidate != target_before, axis=2) if candidate.ndim == 3 else (candidate != target_before)
    dark_max = int(getattr(cfg, "direct_post_structural_restore_target_dark_max", 205))
    lighten = int(getattr(cfg, "direct_post_structural_restore_min_lighten", 18))
    lost_dark = changed & (tg <= dark_max) & ((cg.astype(np.int16) - tg.astype(np.int16)) >= lighten)
    diag["candidate_lost_dark_pixels"] = int(np.count_nonzero(lost_dark))
    if not np.any(lost_dark):
        diag["reason"] = "no_lost_target_dark_pixels"
        return candidate, empty, diag

    nz = cv2.findNonZero(geom * 255)
    if nz is None:
        diag["reason"] = "empty_geometry"
        return candidate, empty, diag
    gx, gy, gw, gh = [int(v) for v in cv2.boundingRect(nz)]
    # Outer strips are defined in the *registered container* coordinate frame,
    # not the image crop frame. This is what rejects central Japanese lettering.
    x_margin = max(6, int(round(gw * float(getattr(cfg, "direct_post_structural_restore_edge_ratio_x", 0.16)))))
    y_margin = max(6, int(round(gh * float(getattr(cfg, "direct_post_structural_restore_edge_ratio_y", 0.12)))))
    yy, xx = np.indices(geom.shape)
    edge_strip = geom.astype(bool) & (
        (xx <= gx + x_margin)
        | (xx >= gx + gw - 1 - x_margin)
        | (yy <= gy + y_margin)
        | (yy >= gy + gh - 1 - y_margin)
    )

    # Also stay near the geometry boundary. Broken oval segments can sit a few
    # pixels inward because the clear mask describes the paper interior rather
    # than the line centre itself.
    boundary = cv2.morphologyEx(geom * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    band_px = max(2, int(getattr(cfg, "direct_post_structural_restore_boundary_band_px", 6)))
    band = cv2.dilate(
        boundary.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1)),
        iterations=1,
    ) > 0
    core_candidates = lost_dark & edge_strip & band
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core_candidates.astype(np.uint8), 8)
    restore_core = np.zeros(geom.shape, np.uint8)
    min_area = max(3, int(getattr(cfg, "direct_post_structural_restore_min_area_px", 10)))
    for lab in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area < min_area:
            continue
        comp = labels == lab
        span_x = bw / max(1.0, float(gw))
        span_y = bh / max(1.0, float(gh))
        fill = area / max(1.0, float(bw * bh))
        near_lr = x <= gx + x_margin or (x + bw - 1) >= gx + gw - 1 - x_margin
        near_tb = y <= gy + y_margin or (y + bh - 1) >= gy + gh - 1 - y_margin
        vertical_line = near_lr and bh >= max(10, int(round(bw * 1.8))) and span_y >= 0.12
        horizontal_line = near_tb and bw >= max(10, int(round(bh * 1.8))) and span_x >= 0.12
        curved_outline = (near_lr or near_tb) and (span_x >= 0.16 or span_y >= 0.16) and fill <= 0.46
        if vertical_line or horizontal_line or curved_outline:
            restore_core[comp] = 255
            diag["restored_components"] += 1

    core_n = int(cv2.countNonZero(restore_core))
    diag["restored_core_pixels"] = core_n
    if core_n <= 0:
        diag["reason"] = "no_structural_component"
        return candidate, empty, diag

    # Restore the anti-aliased fringe of accepted line cores as well, but only
    # pixels that truly changed and were non-paper in TARGET.
    fringe_px = max(0, int(getattr(cfg, "direct_post_structural_restore_fringe_px", 1)))
    restore = restore_core > 0
    if fringe_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * fringe_px + 1, 2 * fringe_px + 1))
        halo = cv2.dilate(restore_core, k, iterations=1) > 0
        restore |= halo & changed & (tg <= int(getattr(cfg, "direct_post_structural_restore_fringe_gray_max", 238)))
    restore_u8 = restore.astype(np.uint8) * 255
    out = candidate.copy()
    out[restore] = target_before[restore]
    diag["restored_pixels"] = int(cv2.countNonZero(restore_u8))
    return out, restore_u8, diag


def _publication_safety_enabled(cfg) -> bool:
    """Legacy compatibility shim: publication blocking was removed in v1.0.6.

    The field may still exist in old config files, but it no longer changes the
    write path.  Only basic geometry/validity checks remain.
    """
    return False


def _source_direct_registration_gate(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    cfg: MaskReplaceConfig | DirectPatchConfig,
) -> tuple[bool, bool]:
    """Return (accepted, used_relaxed_cross_rendition_gate)."""
    if not _publication_safety_enabled(cfg):
        # Basic geometry sanity only.  This is intentionally much looser than the
        # former publication gate; a zero/failed registration is still unusable.
        return bool(float(registration.confidence) >= 0.30), False
    min_reg = float(getattr(cfg, "source_direct_min_registration_confidence", 0.82))
    if float(registration.confidence) >= min_reg:
        return True, False
    if not bool(getattr(cfg, "source_direct_cross_rendition_relaxed_gate_enabled", True)):
        return False, False
    source_sat = float(np.percentile(cv2.cvtColor(source, cv2.COLOR_BGR2HSV)[..., 1], 90.0))
    target_sat = float(np.percentile(cv2.cvtColor(target, cv2.COLOR_BGR2HSV)[..., 1], 90.0))
    cross_rendition = source_sat < 24.0 and target_sat >= 24.0
    accepted = bool(
        cross_rendition
        and float(registration.confidence) >= float(getattr(cfg, "source_direct_cross_rendition_min_registration_confidence", 0.78))
        and float(getattr(registration, "inlier_ratio", 0.0)) >= float(getattr(cfg, "source_direct_cross_rendition_min_inlier_ratio", 0.68))
        and float(getattr(registration, "reprojection_error", 999.0)) <= float(getattr(cfg, "source_direct_cross_rendition_max_reprojection_error", 1.40))
        and float(getattr(registration, "spatial_coverage", 0.0)) >= float(getattr(cfg, "source_direct_cross_rendition_min_spatial_coverage", 0.55))
    )
    return accepted, accepted

def _refine_direct_source_hint_mask(
    source_gray: np.ndarray,
    filled: np.ndarray,
    inset: int,
    cfg,
    *,
    is_source_hint: bool,
    hint_backend: str,
) -> tuple[np.ndarray, int, dict]:
    """Return the Direct transfer interior for one accepted SOURCE candidate.

    A pseudo-text-barrier hint is already a reconstructed semantic *interior*;
    applying the generic progressive border erosion again clips boundary-adjacent
    translated columns. All other providers keep the v2.3.17 refiner unchanged.
    """
    pseudo_semantic_interior=bool(is_source_hint and str(hint_backend)=="pseudo_text_barrier")
    if pseudo_semantic_interior:
        return filled.copy(),0,{
            "provider_semantic_interior":True,
            "estimated_border_px":0,
            "selected_inset_px":0,
        }
    mask_refiner=PROVIDER_REGISTRY.get("mask_refiner","progressive_border") or _progressive_transfer_mask
    return mask_refiner(source_gray,filled,inset,cfg)


def build_source_direct_container_plan(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    cfg: MaskReplaceConfig | DirectPatchConfig | None = None,
    source_hint_bubbles: list[BubbleInstance] | None = None,
) -> DirectContainerPlan | None:
    """Build and render a fast whole-container plan from the translated source.

    Safety model:
      * source must be predominantly monochrome white-container manga;
      * registration must be strong; affine/homography is location-only and never
        applied directly to the final CJK raster;
      * candidate source interior must be white/paper + compact ink;
      * mapped target area must look like either a white text container or a
        saturated coloured burst container;
      * source/target lettering must actually differ;
      * target outline is used only for local similarity/translation refinement
        and is never written to the output.
    """
    cfg = cfg or MaskReplaceConfig()
    if not bool(getattr(cfg, "source_direct_container_enabled", True)):
        return None
    if source.ndim != 3 or target.ndim != 3:
        return None
    reg_ok, relaxed_cross_rendition_gate = _source_direct_registration_gate(source, target, registration, cfg)
    if not reg_ok:
        return None

    H = transform_to_homography(registration.matrix)
    coord_space = SourceCoordinateSpace.from_registration(registration)
    scale, anisotropy, angle, perspective = _uniform_page_scale(H)
    if not (float(getattr(cfg, "source_direct_min_uniform_scale", 0.25)) <= scale <= float(getattr(cfg, "source_direct_max_uniform_scale", 2.5))):
        return None
    if anisotropy > float(getattr(cfg, "source_direct_max_axis_scale_delta", 0.065)):
        return None
    if abs(angle) > float(getattr(cfg, "source_direct_max_rotation_deg", 1.5)):
        return None
    if perspective > float(getattr(cfg, "source_direct_max_perspective", 2.5e-5)):
        return None

    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    # This route is designed for the user's B/W translated scan -> colour/HD
    # master workflow.  It must not accidentally copy a coloured source artwork
    # patch as a "container".
    source_sat_p90 = float(np.percentile(source_hsv[..., 1], 90.0))
    if _publication_safety_enabled(cfg) and source_sat_p90 > float(getattr(cfg, "source_direct_max_source_saturation_p90", 35.0)):
        return None

    sh, sw = source_gray.shape; th, tw = target_gray.shape
    page_area = max(1, sh * sw)

    dark_thr = int(getattr(cfg, "source_direct_outline_dark_threshold", 160))
    source_dark = (source_gray < dark_thr).astype(np.uint8) * 255
    source_dark = cv2.morphologyEx(source_dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _hier = cv2.findContours(source_dark, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    raw_contour_entries: list[tuple[np.ndarray, bool, str]] = [(c, False, "contour") for c in contours]
    hint_contour_entries: list[tuple[np.ndarray, bool, str]] = []
    source_detector_hint_count = 0
    # Direct v2.3.13: SOURCE bubble/text-box masks are an authority boundary, not
    # merely an extra candidate source.  The old v2.3.4 code appended detector
    # hints to *all* dark contours.  With publication safety relaxed this allowed
    # a face/hair/panel contour to become an independent Direct candidate.
    # When semantic hints exist, use them exclusively.  Raw contour discovery is
    # retained only as a compatibility fallback for pages with zero usable hints.
    if source_hint_bubbles:
        for bubble in source_hint_bubbles:
            mask = bubble.mask
            if mask is None or mask.shape[:2] != source_gray.shape[:2]:
                continue
            hint_u8 = (mask > 0).astype(np.uint8) * 255
            hint_contours, _ = cv2.findContours(hint_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not hint_contours:
                continue
            # A detector mask can contain disconnected dust. Keep every meaningful
            # component so no bubble/text box is silently lost, but never synthesize
            # a candidate outside the supplied semantic mask.
            for hc in hint_contours:
                if cv2.contourArea(hc) < 24.0:
                    continue
                hint_contour_entries.append((hc, True, str(bubble.meta.get("backend") or "source_hint")))
            source_detector_hint_count += 1
    semantic_hint_lock_enabled = bool(hint_contour_entries)
    # Strict Direct no longer promotes raw artwork contours into writable masks.
    # If no bubble/text-box hint exists, fail closed and let the caller keep TARGET
    # unchanged (or use another mode) instead of guessing a destructive region.
    raw_contour_suppressed_count = len(raw_contour_entries)
    if not semantic_hint_lock_enabled:
        return None
    contour_entries = hint_contour_entries

    # Distance to target dark lines.  Only this scalar score is used for border
    # alignment.  The target/source border pixels themselves are never pasted.
    target_dark = (target_gray < int(getattr(cfg, "source_direct_target_edge_threshold", 175))).astype(np.uint8) * 255
    target_edge_distance = cv2.distanceTransform((target_dark == 0).astype(np.uint8), cv2.DIST_L2, 3)

    rendered = target.copy()
    layer = np.zeros((th, tw, 4), np.uint8)
    composite = np.zeros((th, tw), np.uint8)
    clear_mask = np.zeros((th, tw), np.uint8)
    source_bubbles: list[BubbleInstance] = []
    target_bubbles: list[BubbleInstance] = []
    matches: list[BubblePatchMatch] = []
    records: list[MaskTransferRecord] = []

    candidate_count = 0
    rejected_container_like = 0
    accepted_white = 0
    accepted_colored = 0
    rejected_by_artwork = 0
    accepted_boundary_scores: list[float] = []
    accepted_local_scales: list[float] = []
    accepted_local_angles: list[float] = []
    accepted_local_anisotropy: list[float] = []
    accepted_border_insets: list[int] = []
    accepted_border_estimates: list[int] = []
    accepted_refined = 0
    review_candidates = 0
    rejected_alignment = 0
    manual_effect_candidates: list[dict] = []
    colored_cleared_target_pixels = 0
    colored_source_ink_pixels = 0
    residual_specks_removed = 0
    variant_probe_count = 0
    identity_locked_regions = 0
    geometry_snapped_regions = 0
    geometry_snap_gap_pixels = 0
    target_border_protected_pixels = 0
    target_border_changed_before_restore = 0
    target_border_changed_after_restore = 0
    min_area_ratio = float(getattr(cfg, "source_direct_min_area_ratio", 0.00045))
    max_area_ratio = float(getattr(cfg, "source_direct_max_area_ratio", 0.085))
    min_side = int(getattr(cfg, "source_direct_min_side_px", 30))
    max_aspect = float(getattr(cfg, "source_direct_max_aspect", 4.0))
    min_src_white = float(getattr(cfg, "source_direct_min_source_white_ratio", 0.78))
    min_src_dark = float(getattr(cfg, "source_direct_min_source_dark_ratio", 0.008))
    max_src_dark = float(getattr(cfg, "source_direct_max_source_dark_ratio", 0.20))
    min_fill = float(getattr(cfg, "source_direct_min_contour_fill", 0.20))
    inset = max(1, int(getattr(cfg, "source_direct_border_inset_px", 3)))
    search = max(0, int(getattr(cfg, "source_direct_alignment_search_px", 8)))
    max_edge_distance = float(getattr(cfg, "source_direct_max_boundary_distance", 5.2))
    aggressive = not _publication_safety_enabled(cfg)
    expand_candidate_range = bool(getattr(cfg, "source_direct_expand_candidate_range", False))
    if aggressive:
        # Detection still needs a plausible connected region, but former
        # publication-only thresholds must not silently discard translations.
        max_area_ratio = max(max_area_ratio, 0.40)
        max_aspect = max(max_aspect, 8.0)
        min_fill = min(min_fill, 0.05)
        min_src_white = min(min_src_white, 0.30)
        min_src_dark = min(min_src_dark, 0.001)
        max_src_dark = max(max_src_dark, 0.65)
        max_edge_distance = max(max_edge_distance, 18.0)
    if expand_candidate_range:
        # Explicit recovery mode: admit weaker/smaller candidate geometry but
        # retain registered same-page and TARGET-background-only contracts.
        min_area_ratio = min(min_area_ratio, 0.00012)
        max_area_ratio = max(max_area_ratio, 0.24)
        min_side = min(min_side, 18)
        max_aspect = max(max_aspect, 10.0)
        min_fill = min(min_fill, 0.035)
        min_src_white = min(min_src_white, 0.18)
        min_src_dark = min(min_src_dark, 0.0004)
        max_src_dark = max(max_src_dark, 0.82)
        max_edge_distance = max(max_edge_distance, 30.0)

    # When the decoded pages are truly the same dimensions and registration is
    # essentially identity, keep exact source coordinates and skip all resizing.
    exact_same_canvas = (
        source.shape[:2] == target.shape[:2]
        and abs(scale - 1.0) <= 0.008
        and anisotropy <= 0.008
        and abs(angle) <= 0.25
        and max(abs(float(H[0, 2])), abs(float(H[1, 2]))) <= 2.0
    )
    method_lower = str(registration.method).lower()
    if exact_same_canvas:
        auto_alignment_mode = "A0_direct_1to1"
    elif perspective > 1e-8 or "homography" in method_lower:
        auto_alignment_mode = "A3_homography_location_local_similarity_raster"
    elif anisotropy > 0.010 or "affine" in method_lower:
        auto_alignment_mode = "A2_affine_location_local_similarity_raster"
    else:
        auto_alignment_mode = "A1_global_similarity"

    for contour, is_source_hint, hint_backend in contour_entries:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        area_ratio = area / page_area
        x, y, bw, bh = [int(v) for v in cv2.boundingRect(contour)]
        aspect = max(bw / max(1.0, bh), bh / max(1.0, bw))
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue
        if min(bw, bh) < min_side or aspect > max_aspect:
            continue
        fill = area / max(1, bw * bh)
        if fill < min_fill:
            continue
        # Cross-rendition (monochrome translated scan -> colour master) is the
        # highest-risk case for a false white-region Direct candidate: censored
        # skin/background or a whole panel can look like a giant white container.
        # Large regions are routed to target-aware Mask/manual review instead of
        # destructively copying monochrome artwork into the colour page.
        if (
            _publication_safety_enabled(cfg)
            and relaxed_cross_rendition_gate
            and area_ratio > float(getattr(cfg, "source_direct_cross_rendition_max_auto_area_ratio", 0.040))
        ):
            rejected_by_artwork += 1
            continue

        filled = np.zeros_like(source_gray)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
        # pseudo_text_barrier already returns a recovered *interior* semantic
        # container.  Re-running the generic progressive border erosion on it
        # double-insets the authority and can clip the first/last translated
        # column.  Keep the v2.3.17 Direct renderer unchanged and skip only this
        # provider-specific second erosion; TARGET border protection and the
        # renderer's own border guard still apply later in the Direct pipeline.
        inner, selected_inset, mask_fit_diag = _refine_direct_source_hint_mask(
            source_gray,filled,inset,cfg,is_source_hint=is_source_hint,hint_backend=hint_backend,
        )
        inner_px = int(cv2.countNonZero(inner))
        if inner_px < 150:
            continue
        vals = source_gray[inner > 0]
        white_ratio = float(np.mean(vals > 225))
        dark_ratio = float(np.mean(vals < 180))
        if _publication_safety_enabled(cfg):
            if white_ratio < min_src_white or not (min_src_dark <= dark_ratio <= max_src_dark):
                continue
        elif dark_ratio < 0.001:
            continue

        hull = cv2.convexHull(contour)
        solidity = area / max(1.0, float(cv2.contourArea(hull)))
        perimeter = float(cv2.arcLength(contour, True))
        approx_count = len(cv2.approxPolyDP(contour, 0.01 * perimeter, True))
        compact, _compact_area, compact_ratio = _compact_ink_stats(source_gray, inner, bw, bh)
        spiky = (
            solidity < float(getattr(cfg, "source_direct_spiky_solidity", 0.84))
            and approx_count >= int(getattr(cfg, "source_direct_spiky_min_vertices", 10))
            and dark_ratio >= float(getattr(cfg, "source_direct_spiky_min_dark_ratio", 0.025))
        )
        if aggressive and not expand_candidate_range and not is_source_hint and compact < 2 and compact_ratio < 0.006:
            # No publication threshold here: simply no credible text seed exists.
            # Skipping such a contour prevents smoke rings, hair islands and
            # clothing folds from becoming replacements when safety is disabled.
            continue
        if (
            _publication_safety_enabled(cfg)
            and not is_source_hint
            and area_ratio < float(getattr(cfg, "source_direct_small_unhinted_area_ratio", 0.0020))
            and compact < int(getattr(cfg, "source_direct_small_unhinted_min_compact_components", 6))
        ):
            rejected_by_artwork += 1
            continue
        if (compact < 1 or compact_ratio < (0.001 if expand_candidate_range else 0.003)) and not spiky:
            continue
        if (
            _publication_safety_enabled(cfg)
            and not spiky
            and solidity < float(getattr(cfg, "source_direct_white_min_solidity", 0.90))
            and compact_ratio < float(getattr(cfg, "source_direct_low_solidity_min_compact_ratio", 0.030))
        ):
            # Closed artwork regions (a white shirt is a common example) can look
            # like white bubbles.  They are not safe for direct overwrite; keep
            # the fast path conservative and let downstream/manual recovery handle
            # them rather than painting artwork.
            rejected_by_artwork += 1
            continue
        candidate_count += 1

        moments = cv2.moments(filled, True)
        if moments["m00"] <= 0:
            continue
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        try:
            tcx, tcy = coord_space.map_point(cx, cy)
        except ValueError:
            continue
        mapped_source_bbox = _mapped_bbox(coord_space, (x, y, x + bw, y + bh), (th, tw), pad=max(6, selected_inset + 2))

        pad = max(3, selected_inset + 1)
        sx0, sy0 = max(0, x - pad), max(0, y - pad)
        sx1, sy1 = min(sw, x + bw + pad), min(sh, y + bh + pad)
        crop = source[sy0:sy1, sx0:sx1]
        crop_gray = source_gray[sy0:sy1, sx0:sx1]
        crop_mask = inner[sy0:sy1, sx0:sx1]
        # Boundary is a guide only; it is not part of crop_mask.
        crop_boundary = cv2.morphologyEx(
            filled[sy0:sy1, sx0:sx1], cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        )
        if crop.size == 0:
            continue

        # Same-canvas pixel lock. The registration matrix is estimated from page
        # imagery whose lettering intentionally differs, so a tiny false shift is
        # common. If the source outline is already sitting on the target outline at
        # the original coordinates, the only correct Direct operation is a 1:1
        # source raster copy: no warpAffine, no fractional translation, no local
        # border optimizer.
        identity_lock = False
        identity_raw_boundary_score: float | None = None
        if (
            exact_same_canvas
            and bool(getattr(cfg, "exact_identity_copy", True))
            and bool(getattr(cfg, "source_direct_identity_lock_enabled", True))
        ):
            identity_raw_boundary_score, _idx, _idy = _boundary_alignment_score(
                crop_boundary, sx0, sy0, target_edge_distance, 0, coarse_step=1
            )
            identity_lock = bool(
                np.isfinite(identity_raw_boundary_score)
                and identity_raw_boundary_score <= float(getattr(cfg, "source_direct_identity_lock_boundary_distance", 0.85))
            )

        # Geometry-only target envelope. Full affine/homography may safely move a
        # binary container mask even though it must never squeeze the Chinese raster.
        # On non-identical scans this lets the white background reach the registered
        # target border while the glyph patch below stays similarity-only.
        if identity_lock:
            registered_inner = inner.copy()
        else:
            registered_inner = cv2.warpPerspective(
                inner, H, (tw, th), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

        # Location may come from affine/homography registration, but final CJK
        # raster geometry is always a local similarity transform. This lets the
        # page mapping absorb crop/aspect/perspective drift without ever squeezing
        # or shearing Chinese glyphs.
        _local = coord_space.local_similarity(cx, cy)
        local_scale, local_angle, local_anisotropy = _local.scale, _local.rotation_deg, _local.anisotropy
        if identity_lock:
            tcx, tcy = cx, cy
            local_scale, local_angle, local_anisotropy = 1.0, 0.0, 0.0
        elif exact_same_canvas:
            local_scale, local_angle, local_anisotropy = 1.0, 0.0, 0.0
        if bool(getattr(cfg, "source_direct_axis_lock_rotation", False)):
            local_angle = 0.0
        mapping_local_scale = float(local_scale)
        mapping_local_angle = float(local_angle)
        mapping_local_anisotropy = float(local_anisotropy)
        if not (float(getattr(cfg, "source_direct_min_uniform_scale", 0.25)) <= local_scale <= float(getattr(cfg, "source_direct_max_uniform_scale", 2.5))):
            continue
        if local_anisotropy > float(getattr(cfg, "source_direct_max_local_mapping_anisotropy", 0.16)):
            continue

        refine = bool(getattr(cfg, "source_direct_local_similarity_refine", True)) and not exact_same_canvas and not identity_lock
        scale_delta = max(0.0, float(getattr(cfg, "source_direct_local_scale_refine_ratio", 0.025))) if refine else 0.0
        angle_delta = max(0.0, float(getattr(cfg, "source_direct_local_angle_refine_deg", 0.35))) if refine else 0.0
        extra_variants: list[tuple[float, float]] = []
        if scale_delta > 1e-6:
            extra_variants += [(1.0 - scale_delta, 0.0), (1.0 + scale_delta, 0.0)]
        if angle_delta > 1e-6:
            extra_variants += [(1.0, -angle_delta), (1.0, angle_delta)]

        coarse_step = max(1, int(getattr(cfg, "source_direct_alignment_coarse_step", 2)))
        variant_candidates: list[tuple[float, bool, int, int, tuple]] = []

        def probe_variant(sf: float, ad: float) -> None:
            nonlocal variant_probe_count
            candidate_scale = local_scale * float(sf)
            candidate_angle = local_angle + float(ad)
            warped = _warp_similarity_patch(
                crop, crop_gray, crop_mask, crop_boundary,
                (sx0, sy0), (cx, cy), (tcx, tcy), candidate_scale, candidate_angle,
            )
            bx, by, wcrop, wgray, wmask, wboundary = warped
            score, dx0, dy0 = _boundary_alignment_score(
                wboundary, bx, by, target_edge_distance, search, coarse_step=coarse_step,
            )
            variant_probe_count += 1

            # Do not let a slightly prettier border fit win by shrinking/sliding
            # the raster until the actual text-container content no longer maps.
            # This is especially important for saturated starbursts. Rank viable
            # content mappings ahead of geometry-only variants.
            px0, py0 = bx + dx0, by + dy0
            vh, vw = wmask.shape[:2]
            txa, tya = max(0, px0), max(0, py0)
            txb, tyb = min(tw, px0 + vw), min(th, py0 + vh)
            viable = False
            if txb > txa and tyb > tya:
                ox0v, oy0v = txa - px0, tya - py0
                ox1v, oy1v = ox0v + (txb - txa), oy0v + (tyb - tya)
                vm = wmask[oy0v:oy1v, ox0v:ox1v] > 0
                if int(np.count_nonzero(vm)) >= 100:
                    sg = wgray[oy0v:oy1v, ox0v:ox1v]
                    tg = target_gray[tya:tyb, txa:txb]
                    hsvv = target_hsv[tya:tyb, txa:txb]
                    srcw = float(np.mean(sg[vm] > 225)); srci = float(np.mean(sg[vm] < 185))
                    tgtw = float(np.mean(tg[vm] > 225)); tgti = float(np.mean(tg[vm] < 185))
                    satv = hsvv[..., 1][vm]
                    satmed = float(np.median(satv)); highsat = float(np.mean(satv > int(getattr(cfg, "source_direct_colored_sat_pixel_threshold", 80))))
                    white_ok = (
                        tgtw >= float(getattr(cfg, "source_direct_target_min_white_ratio", 0.58))
                        and float(getattr(cfg, "source_direct_target_min_dark_ratio", 0.004)) <= tgti <= float(getattr(cfg, "source_direct_target_max_dark_ratio", 0.24))
                        and highsat < float(getattr(cfg, "source_direct_white_max_high_sat_ratio", 0.35))
                    )
                    color_ok = (
                        spiky
                        and highsat >= float(getattr(cfg, "source_direct_colored_min_high_sat_ratio", 0.65))
                        and satmed >= float(getattr(cfg, "source_direct_colored_min_saturation_median", 65.0))
                    )
                    if (white_ok or color_ok) and srcw >= min_src_white and srci >= min_src_dark:
                        ch = _ink_change_score(sg, tg, vm.astype(np.uint8) * 255)
                        floor = float(getattr(cfg, "source_direct_white_min_ink_change", 0.07)) if white_ok else float(getattr(cfg, "source_direct_colored_min_ink_change", 0.18))
                        viable = ch > floor
            variant_candidates.append((float(score), bool(viable), int(dx0), int(dy0), (candidate_scale, candidate_angle, bx, by, wcrop, wgray, wmask, wboundary)))

        # The page registration already supplies the most likely local scale and
        # rotation. Pixel-locked same-canvas regions bypass the warp/refine path
        # entirely. Otherwise probe the page-derived similarity first and open the
        # extra variants only when its border fit is uncertain.
        if identity_lock:
            variant_candidates.append((
                float(identity_raw_boundary_score if identity_raw_boundary_score is not None else 0.0),
                True, 0, 0,
                (1.0, 0.0, sx0, sy0, crop.copy(), crop_gray.copy(), crop_mask.copy(), crop_boundary.copy()),
            ))
        else:
            probe_variant(1.0, 0.0)
            adaptive = bool(getattr(cfg, "source_direct_adaptive_variant_search", True))
            probe_floor = float(getattr(cfg, "source_direct_variant_probe_boundary_distance", 0.25))
            base_score = variant_candidates[0][0] if variant_candidates else float("inf")
            if (not adaptive) or base_score > probe_floor:
                for sf, ad in extra_variants:
                    probe_variant(sf, ad)
        if not variant_candidates:
            continue
        viable_variants = [v for v in variant_candidates if v[1]]
        chosen = min(viable_variants or variant_candidates, key=lambda v: v[0])
        best_score, _best_viable, best_dx, best_dy, best_variant = chosen

        # One exact translation refinement around the best coarse solution.
        chosen_scale, chosen_angle, base_x, base_y, resized_crop, resized_gray, resized_mask, resized_boundary = best_variant
        if identity_lock:
            fine_score, fine_dx, fine_dy = float(best_score), 0, 0
            best_dx = best_dy = 0
        else:
            fine_score, fine_dx, fine_dy = _boundary_alignment_score(
                resized_boundary, base_x, base_y, target_edge_distance, search,
                coarse_step=1, around=(best_dx, best_dy), around_radius=max(1, coarse_step),
            )
            if fine_score <= best_score:
                best_score, best_dx, best_dy = fine_score, fine_dx, fine_dy
        local_scale = float(chosen_scale)
        local_angle = float(chosen_angle)
        reject_distance = min(max_edge_distance, float(getattr(cfg, "source_direct_reject_boundary_distance", max_edge_distance)))
        review_distance = min(reject_distance, float(getattr(cfg, "source_direct_review_boundary_distance", 3.2)))
        if _publication_safety_enabled(cfg) and best_score > reject_distance:
            rejected_alignment += 1
            continue
        if _publication_safety_enabled(cfg) and best_score > review_distance:
            # Conservative PanelCleaner-style policy: uncertain geometry is never
            # auto-painted. Report a concrete target rectangle so the GUI can
            # route the user straight into the non-destructive omission editor.
            review_candidates += 1
            _bx = int(base_x + best_dx); _by = int(base_y + best_dy)
            _append_manual_effect_candidate(
                manual_effect_candidates,
                source_bbox=(sx0, sy0, sx1, sy1),
                target_bbox=[_bx, _by, _bx + int(resized_mask.shape[1]), _by + int(resized_mask.shape[0])],
                reason="uncertain_container_alignment",
                hint_backend=hint_backend,
                source_hint=is_source_hint,
                compact_components=compact,
                compact_ratio=compact_ratio,
                spiky=spiky,
            )
            continue
        nh, nw = resized_mask.shape[:2]

        px = base_x + best_dx; py = base_y + best_dy
        tx0, ty0 = max(0, px), max(0, py)
        tx1, ty1 = min(tw, px + nw), min(th, py + nh)
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        ox0, oy0 = tx0 - px, ty0 - py
        ox1, oy1 = ox0 + (tx1 - tx0), oy0 + (ty1 - ty0)
        local_mask = resized_mask[oy0:oy1, ox0:ox1] > 0
        if int(np.count_nonzero(local_mask)) < 100:
            continue
        original_local_raster_mask = local_mask.copy()
        local_src = resized_crop[oy0:oy1, ox0:ox1]
        local_src_gray = resized_gray[oy0:oy1, ox0:ox1]
        local_tgt_gray = target_gray[ty0:ty1, tx0:tx1]
        local_tgt_hsv = target_hsv[ty0:ty1, tx0:tx1]
        local_registered_geometry = registered_inner[ty0:ty1, tx0:tx1] > 0

        src_white = float(np.mean(local_src_gray[local_mask] > 225))
        src_ink = float(np.mean(local_src_gray[local_mask] < 185))
        tgt_white = float(np.mean(local_tgt_gray[local_mask] > 225))
        tgt_ink = float(np.mean(local_tgt_gray[local_mask] < 185))
        tgt_sat_values = local_tgt_hsv[..., 1][local_mask]
        tgt_sat_median = float(np.median(tgt_sat_values))
        tgt_high_sat = float(np.mean(tgt_sat_values > int(getattr(cfg, "source_direct_colored_sat_pixel_threshold", 80))))

        white_mode = (
            tgt_white >= float(getattr(cfg, "source_direct_target_min_white_ratio", 0.58))
            and float(getattr(cfg, "source_direct_target_min_dark_ratio", 0.004)) <= tgt_ink <= float(getattr(cfg, "source_direct_target_max_dark_ratio", 0.24))
            and tgt_high_sat < float(getattr(cfg, "source_direct_white_max_high_sat_ratio", 0.35))
        )
        colored_mode = (
            spiky
            and tgt_high_sat >= float(getattr(cfg, "source_direct_colored_min_high_sat_ratio", 0.65))
            and tgt_sat_median >= float(getattr(cfg, "source_direct_colored_min_saturation_median", 65.0))
        )
        if aggressive and colored_mode and not is_source_hint:
            # Removing publication safety must not turn arbitrary character/hair
            # contours into text containers.  This is detector validity rather
            # than a publication gate: unhinted coloured regions still need a
            # compact text-like cluster and must be local rather than an entire
            # figure/panel. SOURCE detector hints are trusted directly.
            colored_mode = bool(
                compact >= 3
                and compact_ratio >= 0.012
                and area_ratio <= 0.035
            )
        if not (white_mode or colored_mode):
            rejected_by_artwork += 1
            # Spiky SOURCE containers that land on a partially saturated TARGET
            # are exactly the purple/pink open-effect class Direct must not paint
            # wholesale. Preserve the rejection, but expose it as an actionable
            # reviewer candidate instead of silently losing the translation.
            if spiky and tgt_high_sat >= 0.20:
                review_candidates += 1
                _append_manual_effect_candidate(
                    manual_effect_candidates,
                    source_bbox=(sx0, sy0, sx1, sy1),
                    target_bbox=[tx0, ty0, tx1, ty1],
                    reason="colored_complex_region_requires_reveal",
                    hint_backend=hint_backend,
                    source_hint=is_source_hint,
                    compact_components=compact,
                    compact_ratio=compact_ratio,
                    spiky=spiky,
                )
            continue
        # Direct Patch is intentionally not a disguised mask compositor. A
        # coloured/textured target requires clearing Japanese ink while preserving
        # target fill, which is Mask Transfer semantics.  Explicit Direct mode
        # therefore rejects it; Auto may fall through to the mask route. Legacy
        # MaskReplaceConfig callers keep the old target-aware behavior.
        if _publication_safety_enabled(cfg) and colored_mode and not bool(getattr(cfg, "allow_target_aware_colored_composite", True)):
            review_candidates += 1
            _append_manual_effect_candidate(
                manual_effect_candidates,
                source_bbox=(sx0, sy0, sx1, sy1),
                target_bbox=[tx0, ty0, tx1, ty1],
                reason="colored_target_requires_reveal",
                hint_backend=hint_backend,
                source_hint=is_source_hint,
                compact_components=compact,
                compact_ratio=compact_ratio,
                spiky=spiky,
            )
            continue

        local_gap_fill_diag = {"enabled": False, "iterations": 0, "added_pixels": 0}
        geometry_snap_diag = {"enabled": False, "accepted": False, "gap_pixels": 0}
        local_target_edge_distance = target_edge_distance[ty0:ty1, tx0:tx1]
        if white_mode:
            expanded_mask, local_gap_fill_diag = _expand_white_gap_mask(
                local_mask.astype(np.uint8), local_src_gray, local_tgt_gray, local_target_edge_distance, cfg,
            )
            local_mask = expanded_mask > 0

            if (
                not identity_lock
                and bool(getattr(cfg, "source_direct_geometry_snap_enabled", True))
                and local_registered_geometry.shape == local_mask.shape
            ):
                geom, boundary_guard_diag = _protect_registered_geometry_boundary(
                    local_registered_geometry, local_target_edge_distance, cfg
                )
                la = max(1, int(np.count_nonzero(local_mask)))
                ga = max(1, int(np.count_nonzero(geom)))
                inter = int(np.count_nonzero(local_mask & geom))
                overlap = inter / max(1, min(la, ga))
                area_ratio = ga / max(1, la)
                accepted_geom = (
                    overlap >= float(getattr(cfg, "source_direct_geometry_snap_min_overlap", 0.78))
                    and float(getattr(cfg, "source_direct_geometry_snap_min_area_ratio", 0.72)) <= area_ratio <= float(getattr(cfg, "source_direct_geometry_snap_max_area_ratio", 1.38))
                )
                geometry_snap_diag = {
                    "enabled": True, "accepted": bool(accepted_geom),
                    "overlap": float(overlap), "area_ratio": float(area_ratio),
                    "gap_pixels": 0, **boundary_guard_diag,
                }
                if accepted_geom:
                    gap = geom & ~local_mask
                    geometry_snap_diag["gap_pixels"] = int(np.count_nonzero(gap))
                    # Use the geometry envelope for white background completion,
                    # but do not clip dark source glyph pixels merely because the
                    # page H has a tiny local shape mismatch.
                    ink_thr = int(getattr(cfg, "source_direct_geometry_snap_source_ink_threshold", 220))
                    source_ink = local_mask & (local_src_gray < ink_thr)
                    safe_outside_ink = source_ink & ~geom & (local_target_edge_distance >= float(getattr(cfg, "source_direct_geometry_snap_edge_distance_px", 0.75)))
                    local_mask = geom | safe_outside_ink

        # Keep TARGET clear geometry independent from SOURCE write geometry.
        # The clear path may reach edge-adjacent JP inside the registered box;
        # SOURCE ink still uses the stricter inset write mask.
        local_target_clear_geometry = (local_registered_geometry > 0) if white_mode and np.any(local_registered_geometry) else local_mask.copy()
        target_border_diag = {"enabled": False, "protected_pixels": 0}
        local_target_border_protected = np.zeros_like(local_mask, dtype=bool)
        if white_mode:
            local_mask, local_target_border_protected, target_border_diag = _constrain_direct_write_to_target_border(
                local_mask, local_registered_geometry, local_target_edge_distance, cfg,
                target_region=target[ty0:ty1, tx0:tx1],
            )
            if int(np.count_nonzero(local_mask)) < 100:
                continue
        local_mask_u8 = local_mask.astype(np.uint8) * 255
        change = _ink_change_score(local_src_gray, local_tgt_gray, local_mask_u8)
        change_floor = (
            float(getattr(cfg, "source_direct_white_min_ink_change", 0.07))
            if white_mode else float(getattr(cfg, "source_direct_colored_min_ink_change", 0.18))
        )
        if change <= change_floor:
            # Same artwork / same lettering: it does not need a replacement.
            # This is not treated as an ambiguous failure.
            continue
        if _publication_safety_enabled(cfg) and (src_white < min_src_white or src_ink < min_src_dark):
            rejected_container_like += 1
            continue
        # Complex low-solidity closed artwork (faces/hair/garments) can satisfy
        # the old 'spiky' geometric definition. Text-seeded recovered containers
        # are exempt; raw contours must show a substantially different ink layout
        # before any destructive overwrite is allowed.
        if (
            _publication_safety_enabled(cfg)
            and spiky and not is_source_hint
            and solidity < float(getattr(cfg, "source_direct_white_min_solidity", 0.90))
            and change < float(getattr(cfg, "source_direct_artwork_low_solidity_min_ink_change", 0.24))
        ):
            _append_manual_effect_candidate(
                manual_effect_candidates,
                source_bbox=(x, y, x + bw, y + bh),
                target_bbox=mapped_source_bbox,
                reason="spiky_text_like_region_needs_manual_reveal",
                hint_backend=hint_backend,
                source_hint=is_source_hint,
                compact_components=compact,
                compact_ratio=compact_ratio,
                spiky=spiky,
            )
            rejected_by_artwork += 1
            continue

        target_full_mask = np.zeros((th, tw), np.uint8)
        target_full_mask[ty0:ty1, tx0:tx1][local_mask] = 255
        # RETR_TREE exposes both sides of a thick bubble outline.  They can yield
        # near-identical nested interiors. Keep the first/larger accepted region
        # and never paint the same container twice.
        duplicate = False
        ta = max(1, int(cv2.countNonZero(target_full_mask)))
        for existing in target_bubbles:
            em = existing.mask
            if em is None or em.shape != target_full_mask.shape:
                continue
            ea = max(1, int(cv2.countNonZero(em)))
            inter = int(cv2.countNonZero(cv2.bitwise_and(target_full_mask, em)))
            if inter / max(1, min(ta, ea)) >= 0.78:
                duplicate = True
                break
        if duplicate:
            continue

        # v1.3.10: local border alignment is geometry-only.  A narration-box
        # outline can legitimately prefer a +2/+3px border fit because editions
        # have different rule thickness/crops; dragging Chinese by that amount
        # is a visible typesetting error.  Keep the TARGET/container geometry on
        # ``best_dx/best_dy`` but cap the SOURCE text raster correction around the
        # page-registration position.
        text_raster_dx = int(best_dx)
        text_raster_dy = int(best_dy)
        text_raster_shift_diag = {
            "enabled": False,
            "geometry_dx": int(best_dx), "geometry_dy": int(best_dy),
            "text_dx": int(best_dx), "text_dy": int(best_dy),
            "correction_x": 0, "correction_y": 0,
        }
        if white_mode and bool(getattr(cfg, "source_direct_text_raster_shift_limit_enabled", True)) and not identity_lock:
            cap = max(0, int(getattr(cfg, "source_direct_text_raster_max_local_shift_px", 1)))
            text_raster_dx = max(-cap, min(cap, int(best_dx)))
            text_raster_dy = max(-cap, min(cap, int(best_dy)))
            corr_x = int(text_raster_dx - int(best_dx))
            corr_y = int(text_raster_dy - int(best_dy))
            text_raster_shift_diag = {
                "enabled": True,
                "max_local_shift_px": int(cap),
                "geometry_dx": int(best_dx), "geometry_dy": int(best_dy),
                "text_dx": int(text_raster_dx), "text_dy": int(text_raster_dy),
                "correction_x": int(corr_x), "correction_y": int(corr_y),
            }
            if corr_x or corr_y:
                M_text = np.asarray([[1.0, 0.0, float(corr_x)], [0.0, 1.0, float(corr_y)]], np.float32)
                local_src = cv2.warpAffine(
                    local_src, M_text, (local_src.shape[1], local_src.shape[0]),
                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
                )
                local_src_gray = cv2.warpAffine(
                    local_src_gray, M_text, (local_src_gray.shape[1], local_src_gray.shape[0]),
                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
                )

        # v1.0.7: Direct is now a *text-only* transfer contract on colour masters.
        # SOURCE background RGB is never permitted to enter TARGET, even for a
        # white speech balloon.  This prevents a falsely-large white candidate
        # from bleaching skin, clothes, halftone or coloured effects.  We clear
        # only TARGET-only lettering components and draw only SOURCE-only Chinese
        # ink over the original TARGET artwork.
        dst_region = rendered[ty0:ty1, tx0:tx1]
        target_region_before = dst_region.copy()
        region_u8 = local_mask.astype(np.uint8) * 255
        candidate_region_ratio = float(np.count_nonzero(region_u8) / max(1, th * tw))
        complete_white_ink = bool(
            white_mode
            and (is_source_hint or candidate_region_ratio <= float(getattr(cfg, "source_direct_text_only_complete_white_max_region_ratio", 0.035)))
        )
        if bool(getattr(cfg, "direct_borderless_overlay_enabled", False)):
            candidate_region, local_write_mask, local_source_text_mask, colored_transfer_diag = compose_direct_overlay(
                dst_region.copy(), local_src, region_u8,
                white_mode=bool(complete_white_ink),
                support_mask=original_local_raster_mask.astype(np.uint8) * 255,
                border_guard_px=max(0, int(getattr(cfg, "direct_white_overlay_border_guard_px", 2))),
                clear_target_text=bool(getattr(cfg, "direct_clear_target_text_before_overlay", True)),
                clear_dilate_px=max(0, int(getattr(cfg, "source_direct_colored_clear_dilate_px", 1))),
                inpaint_radius=float(getattr(cfg, "source_direct_colored_inpaint_radius", 2.5)),
                target_clear_region_mask=local_target_clear_geometry.astype(np.uint8) * 255 if white_mode else None,
                white_source_clarity_enabled=bool(getattr(cfg, "direct_white_clarity_enhance_enabled", True)),
                white_source_clarity_alpha_gamma=float(getattr(cfg, "direct_white_clarity_alpha_gamma", 1.0)),
                white_source_clarity_black_boost=int(getattr(cfg, "direct_white_clarity_black_boost", 0)),
                white_source_clarity_pure_white_floor=int(getattr(cfg, "direct_white_clarity_pure_white_floor", 248)),
                white_source_clarity_min_text_pixels=int(getattr(cfg, "direct_white_clarity_min_text_pixels", 18)),
                post_structural_restore_enabled=bool(getattr(cfg, "direct_post_structural_restore_enabled", True)),
                post_structural_restore_target_dark_max=int(getattr(cfg, "direct_post_structural_restore_target_dark_max", 205)),
                post_structural_restore_min_lighten=int(getattr(cfg, "direct_post_structural_restore_min_lighten", 18)),
                post_structural_restore_edge_ratio_x=float(getattr(cfg, "direct_post_structural_restore_edge_ratio_x", 0.16)),
                post_structural_restore_edge_ratio_y=float(getattr(cfg, "direct_post_structural_restore_edge_ratio_y", 0.12)),
                post_structural_restore_boundary_band_px=int(getattr(cfg, "direct_post_structural_restore_boundary_band_px", 6)),
                post_structural_restore_min_area_px=int(getattr(cfg, "direct_post_structural_restore_min_area_px", 10)),
                post_structural_restore_fringe_px=int(getattr(cfg, "direct_post_structural_restore_fringe_px", 1)),
                post_structural_restore_fringe_gray_max=int(getattr(cfg, "direct_post_structural_restore_fringe_gray_max", 238)),
            )
        else:
            candidate_region, local_write_mask, local_source_text_mask, colored_transfer_diag = transfer_text_only(
                dst_region.copy(), local_src, region_u8,
                tolerance_px=max(1, int(getattr(cfg, "source_direct_text_only_tolerance_px", 2))),
                clear_dilate_px=max(0, int(getattr(cfg, "source_direct_colored_clear_dilate_px", 1))),
                inpaint_radius=float(getattr(cfg, "source_direct_colored_inpaint_radius", 2.5)),
                white_container=complete_white_ink,
                localized_white_text=bool(white_mode and not complete_white_ink),
                white_full_clear_enabled=bool(getattr(cfg, "white_container_full_clear_enabled", True)),
                white_full_clear_min_paper_ratio=float(getattr(cfg, "white_container_full_clear_min_paper_ratio", 0.68)),
                white_full_clear_max_robust_spread=float(getattr(cfg, "white_container_full_clear_max_robust_spread", 14.0)),
                white_write_inset_px=int(getattr(cfg, "white_container_write_inset_px", 1)),
                white_write_border_guard_px=int(getattr(cfg, "white_container_write_border_guard_px", 1)),
                white_clear_inset_px=int(getattr(cfg, "white_container_clear_inset_px", 0)),
                white_clear_border_guard_px=int(getattr(cfg, "white_container_clear_border_guard_px", 0)),
                target_clear_region_mask=local_target_clear_geometry.astype(np.uint8) * 255 if white_mode else None,
            )
        colored_transfer_diag["complete_white_ink"] = bool(complete_white_ink)
        colored_transfer_diag["candidate_region_ratio"] = float(candidate_region_ratio)
        colored_transfer_diag["text_raster_shift"] = text_raster_shift_diag

        # A face/shirt can accidentally satisfy a loose low-saturation "white"
        # test.  For an unhinted compact white candidate, require the same
        # uniform TARGET-paper proof that authorizes full-container blanking.
        # If it fails, do not perform a selective destructive Direct edit; route
        # it to Review instead.
        if (
            white_mode and complete_white_ink and not is_source_hint
            and bool(getattr(cfg, "source_direct_unhinted_white_requires_full_clear", True))
            and not bool(colored_transfer_diag.get("white_full_clear_applied", False))
        ):
            _append_manual_effect_candidate(
                manual_effect_candidates,
                source_bbox=(x, y, x + bw, y + bh),
                target_bbox=[int(tx0), int(ty0), int(tx1), int(ty1)],
                reason="unhinted_white_candidate_not_uniform_paper",
                hint_backend=hint_backend, source_hint=is_source_hint,
                compact_components=compact, compact_ratio=compact_ratio, spiky=spiky,
            )
            rejected_by_artwork += 1
            continue

        changed_before_restore = 0
        changed_after_restore = 0
        if white_mode and np.any(local_target_border_protected):
            changed_before_restore = int(np.count_nonzero(
                np.any(candidate_region[local_target_border_protected] != target_region_before[local_target_border_protected], axis=1)
            ))
            if bool(getattr(cfg, "source_direct_exact_target_border_restore", True)):
                candidate_region[local_target_border_protected] = target_region_before[local_target_border_protected]
                local_source_text_mask[local_target_border_protected] = 0
                changed_now = np.any(candidate_region != target_region_before, axis=2)
                local_write_mask = changed_now.astype(np.uint8) * 255
            changed_after_restore = int(np.count_nonzero(
                np.any(candidate_region[local_target_border_protected] != target_region_before[local_target_border_protected], axis=1)
            ))
        post_structural_restore_diag = dict(colored_transfer_diag.get("post_structural_restore") or {"enabled": False, "restored_pixels": 0})
        # The borderless Direct overlay owns the post-clear structural restore so
        # it can update its own clear/source-text masks atomically.  Retain this
        # local fallback only for the legacy non-borderless transfer path; running
        # both passes needlessly repeats connected-components work and can make
        # diagnostics ambiguous.
        if (
            not bool(getattr(cfg, "direct_borderless_overlay_enabled", False))
            and white_mode
            and bool(colored_transfer_diag.get("white_full_clear_applied", False))
        ):
            candidate_region, post_restore_mask, post_structural_restore_diag = _restore_damaged_target_structural_lines(
                candidate_region, target_region_before, local_target_clear_geometry, cfg
            )
            if int(cv2.countNonZero(post_restore_mask)) > 0:
                local_source_text_mask[post_restore_mask > 0] = 0
                changed_now = np.any(candidate_region != target_region_before, axis=2)
                local_write_mask = changed_now.astype(np.uint8) * 255
        target_border_diag.update({
            "exact_restore_enabled": bool(getattr(cfg, "source_direct_exact_target_border_restore", True)),
            "changed_before_restore": int(changed_before_restore),
            "changed_after_restore": int(changed_after_restore),
            "post_structural_restore": post_structural_restore_diag,
        })
        colored_transfer_diag["source_ink_pixels"] = int(cv2.countNonZero(local_source_text_mask))
        colored_transfer_diag["write_pixels"] = int(cv2.countNonZero(local_write_mask))
        colored_transfer_diag["target_border_preservation"] = target_border_diag
        ink_pixels = int(colored_transfer_diag.get("source_ink_pixels", 0))
        evidence_source_pixels = int(colored_transfer_diag.get("changed_source_text_pixels", colored_transfer_diag.get("source_text_pixels", 0)))
        evidence_target_pixels = int(colored_transfer_diag.get("changed_target_text_pixels", colored_transfer_diag.get("target_text_pixels", 0)))
        region_pixels = max(1, int(colored_transfer_diag.get("region_pixels", int(np.count_nonzero(region_u8)))))
        write_pixels = int(colored_transfer_diag.get("write_pixels", 0))
        source_text_density = float(evidence_source_pixels / region_pixels)
        target_text_density = float(evidence_target_pixels / region_pixels)
        if write_pixels == 0 or ink_pixels == 0:
            # A container can be geometrically valid yet contain no edition-
            # changed lettering.  Do not copy paper/background just to force an
            # apparent replacement.
            continue
        # This is a text-validity gate, not publication safety.  A huge panel or
        # body contour can look like a white container after safety is disabled;
        # if changed SOURCE ink is vanishingly sparse, or TARGET "text" is mostly
        # unmatched artwork, Direct must not erase it.  The OCR-free white-bubble
        # completion pass can still recover the true nested balloon afterwards.
        localized_write_ratio = float(write_pixels / region_pixels)
        localized_artwork_overreach = bool(
            white_mode and not complete_white_ink and localized_write_ratio > 0.16
        )
        text_evidence_bad = (
            source_text_density < 0.010
            or (target_text_density > 0.20 and evidence_target_pixels > max(24, evidence_source_pixels * 3))
            or localized_artwork_overreach
        )
        if text_evidence_bad:
            _append_manual_effect_candidate(
                manual_effect_candidates,
                source_bbox=(x, y, x + bw, y + bh),
                target_bbox=[int(tx0), int(ty0), int(tx1), int(ty1)],
                reason="non_text_artwork_candidate_skipped",
                hint_backend=hint_backend,
                source_hint=is_source_hint,
                compact_components=compact,
                compact_ratio=compact_ratio,
                spiky=spiky,
            )
            rejected_by_artwork += 1
            continue
        dst_region = candidate_region
        rendered[ty0:ty1, tx0:tx1] = dst_region
        colored_cleared_target_pixels += int(colored_transfer_diag.get("cleared_target_pixels", 0))
        colored_source_ink_pixels += ink_pixels
        residual_specks_removed += int(colored_transfer_diag.get("residual_specks_removed", 0) or 0)
        target_write_mask = np.zeros((th, tw), np.uint8)
        target_write_mask[ty0:ty1, tx0:tx1] = local_write_mask

        source_full_mask = inner.copy()
        sid = f"direct-src-{len(source_bubbles):04d}"
        tid = f"direct-dst-{len(target_bubbles):04d}"
        sp = mask_to_largest_polygon(source_full_mask)
        tp = mask_to_largest_polygon(target_full_mask)
        if len(sp) < 3 or len(tp) < 3:
            continue
        kind = "complex_text" if colored_mode else "speech"
        conf = float(np.clip(
            0.76 + 0.10 * min(1.0, change / 0.6) + 0.08 * min(1.0, max(0.0, 1.0 - best_score / max(1e-6, max_edge_distance))),
            0.0, 0.97,
        ))
        sb = BubbleInstance(
            id=sid, polygon=sp, confidence=conf, kind=kind,
            block_ids=[f"direct-raster-{len(source_bubbles):04d}"],
            mask=source_full_mask, safe_mask=source_full_mask.copy(),
            meta={
                "backend": "source_direct_container", "paired_target_id": tid,
                "alignment_only_border": True, "uniform_page_scale": float(local_scale),
                "local_rotation_deg": float(local_angle),
                "mapping_local_anisotropy": float(mapping_local_anisotropy),
                "ink_change": float(change), "colored_target": bool(colored_mode),
                "source_hint": bool(is_source_hint), "source_hint_backend": hint_backend,
                "source_bbox_original": [int(x), int(y), int(x + bw), int(y + bh)],
                "transfer_border_inset_px": int(selected_inset),
                "estimated_source_border_px": int(mask_fit_diag.get("estimated_border_px", max(1, selected_inset - 1))),
                "progressive_mask_fit": mask_fit_diag,
                "white_gap_fill": local_gap_fill_diag,
                "identity_pixel_lock": bool(identity_lock),
                "identity_raw_boundary_score": identity_raw_boundary_score,
                "geometry_snap": geometry_snap_diag,
                "target_border_preservation": target_border_diag,
                "colored_fill_preserved": True,
                "background_policy": "target_underlay_source_on_top",
                "text_only_transfer": False,
            },
        )
        tb = BubbleInstance(
            id=tid, polygon=tp, confidence=conf, kind=kind,
            block_ids=[], mask=target_full_mask, safe_mask=target_full_mask.copy(),
            meta={
                "backend": "source_direct_container", "paired_source_id": sid,
                "alignment_only_border": True, "uniform_page_scale": float(local_scale),
                "local_rotation_deg": float(local_angle),
                "mapping_local_anisotropy": float(mapping_local_anisotropy),
                "alignment_dx": int(best_dx), "alignment_dy": int(best_dy),
                "text_raster_dx": int(text_raster_dx), "text_raster_dy": int(text_raster_dy),
                "text_raster_shift": text_raster_shift_diag,
                "boundary_distance": float(best_score), "ink_change": float(change),
                "source_hint": bool(is_source_hint), "source_hint_backend": hint_backend,
                "colored_target": bool(colored_mode),
                "transfer_border_inset_px": int(selected_inset),
                "estimated_source_border_px": int(mask_fit_diag.get("estimated_border_px", max(1, selected_inset - 1))),
                "white_gap_fill": local_gap_fill_diag,
                "identity_pixel_lock": bool(identity_lock),
                "identity_raw_boundary_score": identity_raw_boundary_score,
                "geometry_snap": geometry_snap_diag,
                "target_border_preservation": target_border_diag,
                "colored_fill_preserved": True,
                "background_policy": "target_underlay_source_on_top",
                "text_only_transfer": False,
                "colored_transfer": colored_transfer_diag,
            },
        )
        source_bubbles.append(sb); target_bubbles.append(tb)

        composite = np.maximum(composite, target_write_mask)
        clear_mask = np.maximum(clear_mask, target_write_mask)
        use = target_write_mask > 0
        layer[use, :3] = rendered[use][:, ::-1]
        layer[..., 3] = np.maximum(layer[..., 3], target_write_mask)

        source_ink_pixels = int(np.count_nonzero((local_src_gray < 185) & local_mask))
        target_mask_pixels = max(1, int(np.count_nonzero(local_mask)))
        match = BubblePatchMatch(
            sid, tid, conf, 1.0 - conf, 1.0, 0.0, 1.0,
            [
                "source_direct_whole_container", "border_alignment_only",
                "pixel_exact_identity_lock" if identity_lock else ("same_canvas_similarity" if exact_same_canvas else "local_similarity_raster"),
                auto_alignment_mode, f"local_rotation={local_angle:.4f}",
                f"boundary_distance={best_score:.3f}", f"ink_change={change:.4f}",
            ],
        )
        rec = MaskTransferRecord(sid, tid, conf, True, "applied_source_direct_container")
        rec.sr_backend = "source-direct-container"
        rec.sr_scale = float(local_scale)
        rec.mask_iou = 1.0
        rec.target_coverage = 1.0
        rec.spill_ratio = 0.0
        rec.local_dx = float(text_raster_dx); rec.local_dy = float(text_raster_dy)
        rec.meta["geometry_local_shift"] = {"dx": int(best_dx), "dy": int(best_dy)}
        rec.meta["text_raster_local_shift"] = dict(text_raster_shift_diag)
        rec.meta["text_only_transfer"] = dict(colored_transfer_diag)
        rec.meta["direct_transform_contract"] = {
            "source_on_top": bool(getattr(cfg, "direct_source_on_top", True)),
            "target_underlay": True,
            "source_border_removed": bool(getattr(cfg, "direct_remove_source_border_lines", True)),
            "axis_locked": bool(getattr(cfg, "source_direct_axis_lock_rotation", False)),
            "rotation_deg": float(local_angle),
            "uniform_scale": float(local_scale),
            "text_shift_dx": int(text_raster_shift_diag.get("dx", 0) or 0),
            "text_shift_dy": int(text_raster_shift_diag.get("dy", 0) or 0),
            "geometry_refine_dx": int(best_dx),
            "geometry_refine_dy": int(best_dy),
        }
        rec.geometry_mode = "source_direct_container"
        rec.clarity_mode = "direct-source-container-patch"
        rec.ink_ratio = float(source_ink_pixels / target_mask_pixels)
        rec.source_bbox = _mask_bbox(source_full_mask)
        rec.target_bbox = _mask_bbox(target_full_mask)
        # Independent content verification.  ``applied`` only means pixels were
        # written; it must never be treated as proof that all Chinese survived or
        # all Japanese disappeared.  Audit the actual final local raster against
        # complete SOURCE/TARGET compact ink evidence.
        target_audit_region = (local_target_clear_geometry.astype(np.uint8) * 255)
        # Completeness QA must judge lettering, not the balloon outline/tail.
        # Direct's target-clear geometry deliberately reaches the container edge
        # so it can erase Japanese glyphs, but compact dark edge fragments can
        # otherwise be misclassified as residual text.  Audit only a shallow
        # interior of the same trusted container; rendering/clearing itself is
        # unchanged.
        audit_guard_px = 3
        audit_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (audit_guard_px * 2 + 1, audit_guard_px * 2 + 1)
        )
        target_audit_region = cv2.erode(target_audit_region, audit_kernel, iterations=1)
        target_audit_mask = target_text_mask_in_container(target_region_before, target_audit_region)
        clarity_audit_diag = colored_transfer_diag.get("white_source_clarity") or {}
        source_fidelity_audit = bool(
            clarity_audit_diag.get("source_raster_fidelity_lock", False)
            or clarity_audit_diag.get("source_text_mask_is_gate_only", False)
        )
        # The source-faithful route publishes the continuous SOURCE raster, while
        # the audit mask is a thresholded compact-ink approximation. Subpixel
        # resampling can therefore score ~0.93-0.94 despite every glyph component
        # being visibly present. Keep the normal 0.94 gate elsewhere, but use a
        # still-conservative 0.92 floor for this explicitly fidelity-locked path.
        audit_min_source_coverage = max(
            0.92 if source_fidelity_audit else 0.94,
            float(getattr(cfg, "content_completeness_min_source_coverage", 0.90)),
        )
        _evaluate_content_completeness(
            rec, local_source_text_mask, target_audit_mask, dst_region, cfg,
            tolerance_px=max(2, int(getattr(cfg, "content_completeness_tolerance_px", 2))),
            min_source_coverage=audit_min_source_coverage,
            max_target_residual=min(0.06, float(getattr(cfg, "content_completeness_max_target_residual", 0.10))),
        )
        rec.meta["content_audit_min_source_coverage"] = float(audit_min_source_coverage)
        rec.meta["source_raster_fidelity_audit"] = bool(source_fidelity_audit)
        if not bool(rec.content_complete):
            rec.review_required = True
            rec.review_reason = "direct_content_incomplete_after_render"
            rec.restorable = True
            rec.editable = True
        rec.clarity_mode = "direct-borderless-source-overlay"
        matches.append(match); records.append(rec)
        accepted_boundary_scores.append(float(best_score))
        accepted_local_scales.append(float(local_scale))
        accepted_local_angles.append(float(local_angle))
        accepted_local_anisotropy.append(float(mapping_local_anisotropy))
        accepted_border_insets.append(int(selected_inset))
        accepted_border_estimates.append(int(mask_fit_diag.get("estimated_border_px", max(1, selected_inset - 1))))
        if identity_lock:
            identity_locked_regions += 1
        if bool(geometry_snap_diag.get("accepted")):
            geometry_snapped_regions += 1
            geometry_snap_gap_pixels += int(geometry_snap_diag.get("gap_pixels", 0))
        target_border_protected_pixels += int(target_border_diag.get("protected_pixels", 0) or 0)
        target_border_changed_before_restore += int(target_border_diag.get("changed_before_restore", 0) or 0)
        target_border_changed_after_restore += int(target_border_diag.get("changed_after_restore", 0) or 0)
        if abs(local_scale - mapping_local_scale) > 1e-7 or abs(local_angle - mapping_local_angle) > 1e-7 or best_dx or best_dy:
            accepted_refined += 1
        if colored_mode:
            accepted_colored += 1
        else:
            accepted_white += 1

    if not records:
        return None

    diagnostics = {
        "method": "direct_patch",
        "strategy_contract": "source_raster_patch_not_mask_composition",
        "registration_confidence": float(registration.confidence),
        "relaxed_cross_rendition_registration_gate": bool(relaxed_cross_rendition_gate),
        "uniform_page_scale": float(scale),
        "axis_anisotropy": float(anisotropy),
        "rotation_deg": float(angle),
        "perspective": float(perspective),
        "exact_same_canvas": bool(exact_same_canvas),
        "auto_alignment_mode": auto_alignment_mode,
        "shape_preserving_raster": True,
        "source_on_top": bool(getattr(cfg, "direct_source_on_top", True)),
        "target_underlay": True,
        "source_border_removed": bool(getattr(cfg, "direct_remove_source_border_lines", True)),
        "canonical_coordinate_space": "source_original_pixels",
        "mapping_can_be_affine_or_homography": True,
        "final_raster_transform": "local_similarity_only_axis_locked",
        "source_saturation_p90": float(source_sat_p90),
        "candidate_count": int(candidate_count),
        "source_detector_hint_count": int(source_detector_hint_count),
        "semantic_hint_lock_enabled": bool(semantic_hint_lock_enabled),
        "raw_contour_candidates_total": int(len(raw_contour_entries)),
        "raw_contour_candidates_suppressed": int(raw_contour_suppressed_count),
        "candidate_authority": "source_bubble_or_textbox_hints_only",
        "strict_direct_support_guard": "bounded_semantic_support_tolerance",
        "accepted": len(records),
        "accepted_white": int(accepted_white),
        "accepted_colored_spiky": int(accepted_colored),
        "rejected_artwork_like": int(rejected_by_artwork),
        "rejected_container_like": int(rejected_container_like),
        "review_candidates_skipped": int(review_candidates),
        "manual_effect_candidates": manual_effect_candidates,
        "manual_effect_candidate_count": int(len(manual_effect_candidates)),
        "rejected_alignment": int(rejected_alignment),
        "local_refined_regions": int(accepted_refined),
        "variant_probe_count": int(variant_probe_count),
        "identity_pixel_lock_enabled": bool(getattr(cfg, "source_direct_identity_lock_enabled", True)),
        "identity_pixel_locked_regions": int(identity_locked_regions),
        "geometry_snap_enabled": bool(getattr(cfg, "source_direct_geometry_snap_enabled", True)),
        "geometry_snapped_regions": int(geometry_snapped_regions),
        "geometry_snap_gap_pixels": int(geometry_snap_gap_pixels),
        "target_border_guard_enabled": bool(getattr(cfg, "source_direct_target_border_guard_enabled", True)),
        "target_border_inset_px": int(getattr(cfg, "source_direct_target_border_inset_px", 2)),
        "target_border_guard_px": int(getattr(cfg, "source_direct_target_border_guard_px", 3)),
        "target_border_protected_pixels": int(target_border_protected_pixels),
        "target_border_changed_before_restore": int(target_border_changed_before_restore),
        "target_border_changed_after_restore": int(target_border_changed_after_restore),
        "median_boundary_distance": float(np.median(accepted_boundary_scores)) if accepted_boundary_scores else None,
        "max_boundary_distance_accepted": float(max(accepted_boundary_scores)) if accepted_boundary_scores else None,
        "median_local_scale": float(np.median(accepted_local_scales)) if accepted_local_scales else None,
        "median_local_rotation_deg": float(np.median(accepted_local_angles)) if accepted_local_angles else None,
        "max_local_mapping_anisotropy": float(max(accepted_local_anisotropy)) if accepted_local_anisotropy else None,
        "median_transfer_border_inset_px": float(np.median(accepted_border_insets)) if accepted_border_insets else None,
        "median_estimated_source_border_px": float(np.median(accepted_border_estimates)) if accepted_border_estimates else None,
        "progressive_dynamic_border": bool(getattr(cfg, "source_direct_dynamic_border_enabled", True)),
        "white_gap_fill_enabled": bool(getattr(cfg, "source_direct_white_gap_fill_enabled", True)),
        "white_gap_fill_max_px": int(getattr(cfg, "source_direct_white_gap_fill_max_px", 3)),
        "colored_fill_preserved": bool(getattr(cfg, "source_direct_colored_preserve_target_fill", True)),
        "colored_cleared_target_pixels": int(colored_cleared_target_pixels),
        "colored_source_ink_pixels": int(colored_source_ink_pixels),
        "residual_specks_removed": int(residual_specks_removed),
        "triage_policy": "safe_review_rejected" if _publication_safety_enabled(cfg) else "aggressive_apply_with_diagnostics",
        "publication_safety_enabled": bool(_publication_safety_enabled(cfg)),
        "border_pixels_written": 0,
        "ocr_used": False,
        "target_bubble_matching_used": False,
    }
    result = MaskTransferResult(rendered, layer, composite, matches, records, clear_mask)
    # Every accepted region passed independent source-paper, target-container,
    # ink-change and outline-alignment gates.  No unresolved container-like region
    # is known, so precise-mask mode may skip the expensive detector/OCR stack.
    artwork_blocks_fast_path = bool(
        getattr(cfg, "source_direct_fail_on_artwork_rejections", True)
        and rejected_by_artwork > 0
    )
    diagnostics["safe_blocked_by_artwork_rejections"] = artwork_blocks_fast_path
    safe = (
        len(records) > 0
        if not _publication_safety_enabled(cfg)
        else (
            rejected_container_like == 0
            and review_candidates == 0
            and len(records) > 0
            and not artwork_blocks_fast_path
        )
    )
    return DirectContainerPlan(result, source_bubbles, target_bubbles, diagnostics, safe)
