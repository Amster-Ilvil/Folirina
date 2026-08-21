from __future__ import annotations

"""Selection-scoped manga topology for closed-region recognition.

This is an independent Folirina implementation inspired by Colortina's
line-boundary/region-map/structural-line pipeline.  The important contract is
that a rough user selection is only authority/scope; recognition chooses one
closed paintable component inside that scope instead of trying to warp the
rectangle itself toward arbitrary nearby ink.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RegionBoundaryMap:
    labels: np.ndarray
    fillable: np.ndarray
    barrier: np.ndarray
    confidence: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.labels.shape[:2]

    def region_at(self, x: int, y: int, *, search_radius: int = 28) -> int:
        h, w = self.shape
        if not (0 <= int(x) < w and 0 <= int(y) < h):
            return 0
        x = int(x); y = int(y)
        rid = int(self.labels[y, x])
        if rid > 0:
            return rid
        radius = max(1, int(search_radius))
        x1, x2 = max(0, x-radius), min(w, x+radius+1)
        y1, y2 = max(0, y-radius), min(h, y+radius+1)
        local = self.labels[y1:y2, x1:x2]
        ys, xs = np.nonzero(local > 0)
        if xs.size == 0:
            return 0
        d2 = (xs-(x-x1))**2 + (ys-(y-y1))**2
        ids = local[ys, xs].astype(np.int32)
        best: dict[int, float] = {}
        for label, dist in zip(ids.tolist(), d2.tolist()):
            if label <= 0:
                continue
            best[label] = min(float(dist), best.get(label, float("inf")))
        if not best:
            return 0
        nearest = min(best.values())
        near = [label for label, dist in best.items() if dist <= nearest + 4.0]
        areas = np.bincount(self.labels.ravel())
        return int(min(near, key=lambda label: (int(areas[label]) if label < len(areas) else 2**31, label)))


def _normalise_percentile(values: np.ndarray, percentile: float) -> np.ndarray:
    arr = np.asarray(values, np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, np.float32)
    scale = float(np.percentile(arr, percentile))
    if scale <= 1e-6:
        return np.zeros_like(arr, np.float32)
    return np.clip(arr / scale, 0.0, 1.0).astype(np.float32)


def _line_confidence(gray: np.ndarray, *, line_low: int = 88) -> np.ndarray:
    g = gray.astype(np.float32)
    dark = np.clip((float(line_low + 58) - g) / 82.0, 0.0, 1.0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    ridge = _normalise_percentile(blackhat, 98.0)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0); gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    edge = _normalise_percentile(cv2.magnitude(gx, gy), 97.5)
    local_mean = cv2.GaussianBlur(g, (0, 0), 2.0)
    edge *= np.clip((246.0 - np.minimum(g, local_mean)) / 80.0, 0.0, 1.0)
    confidence = np.maximum(dark, np.maximum(ridge * 0.82, edge * 0.66))
    confidence[gray <= int(line_low)] = 1.0
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _remove_texture_dots(mask: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary * 255
    keep = np.zeros(count, dtype=bool)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA]); width = int(stats[label, cv2.CC_STAT_WIDTH]); height = int(stats[label, cv2.CC_STAT_HEIGHT])
        span = max(width, height); short = max(1, min(width, height)); aspect = span / float(short)
        pixels = labels == label
        mean_conf = float(confidence[pixels].mean()) if np.any(pixels) else 0.0
        keep[label] = bool(
            (span >= 5 and aspect >= 1.55)
            or area >= 14 or span >= 9
            or (mean_conf >= 0.84 and span >= 4 and aspect >= 1.25)
        )
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _direction_kernels(length: int) -> list[np.ndarray]:
    length = max(3, int(length)); length += 0 if length % 2 else 1
    diag = np.eye(length, dtype=np.uint8)
    return [np.ones((1, length), np.uint8), np.ones((length, 1), np.uint8), diag, np.fliplr(diag).copy()]


def _repair_short_gaps(base_ink: np.ndarray, gray: np.ndarray, max_gap: int) -> np.ndarray:
    """Bridge only small plausible line endpoint gaps; never globally thicken ink."""
    max_gap = max(0, int(max_gap))
    if max_gap <= 0:
        return np.zeros_like(base_ink, np.uint8)
    base = (base_ink > 0).astype(np.uint8)
    accepted = np.zeros_like(base)
    _, base_labels, base_stats, _ = cv2.connectedComponentsWithStats(base, 8)
    budget = max(24, int(base.size * min(0.012, 0.0014 * max_gap)))
    accepted_count = 0
    lengths = list(range(3, max_gap + 2, 2)) or [3]
    for length in lengths:
        max_area = max(4, int(length * 2.8)); max_span = length + 2
        for kernel in _direction_kernels(length):
            closed = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel, iterations=1)
            proposed = ((closed > 0) & (base == 0) & (accepted == 0)).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(proposed, 8)
            for idx in range(1, n):
                x = int(stats[idx, cv2.CC_STAT_LEFT]); y = int(stats[idx, cv2.CC_STAT_TOP])
                w = int(stats[idx, cv2.CC_STAT_WIDTH]); h = int(stats[idx, cv2.CC_STAT_HEIGHT]); area = int(stats[idx, cv2.CC_STAT_AREA])
                if area > max_area or max(w, h) > max_span or min(w, h) > max(3, int(length * 0.45) + 1):
                    continue
                local = labels[y:y+h, x:x+w] == idx
                x1, y1 = max(0, x-1), max(0, y-1); x2, y2 = min(base.shape[1], x+w+1), min(base.shape[0], y+h+1)
                padded = np.zeros((y2-y1, x2-x1), np.uint8)
                padded[y-y1:y-y1+h, x-x1:x-x1+w] = local.astype(np.uint8)
                contact = (cv2.dilate(padded, np.ones((3,3), np.uint8)) > 0) & (base[y1:y2, x1:x2] > 0)
                cy, cx = np.nonzero(contact)
                if cx.size < 2:
                    continue
                span = max(float(np.ptp(cx)), float(np.ptp(cy)))
                if span < max(2.0, min(max(w, h), length) * 0.45):
                    continue
                touching = np.unique(base_labels[y1:y2, x1:x2][contact]); touching = touching[touching > 0]
                if len(touching) > 2:
                    continue
                if len(touching) == 2:
                    plausible = True
                    for lid in touching:
                        st = base_stats[int(lid)]; ca = int(st[cv2.CC_STAT_AREA]); cw = int(st[cv2.CC_STAT_WIDTH]); ch = int(st[cv2.CC_STAT_HEIGHT])
                        major = max(cw, ch); aspect = major / max(1.0, float(min(cw, ch)))
                        if major < 5 or (ca < 18 and aspect < 1.65): plausible = False; break
                    if not plausible:
                        continue
                ys, xs = np.nonzero(local); sample = gray[y+ys, x+xs]
                if sample.size and float(np.percentile(sample, 35)) < 118.0:
                    continue
                target = accepted[y:y+h, x:x+w]; new = local & (target == 0); target[local] = 1
                accepted_count += int(np.count_nonzero(new))
                if accepted_count >= budget:
                    return accepted * 255
    return accepted * 255


def build_region_boundary_map(image_bgr: np.ndarray, *, gap_close: int = 5) -> RegionBoundaryMap:
    if image_bgr.ndim == 2:
        gray = image_bgr.astype(np.uint8, copy=False)
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    native = _line_confidence(gray, line_low=88)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6,6)).apply(gray)
    contrast = _line_confidence(clahe, line_low=102)
    g = gray.astype(np.float32); local_bg = cv2.GaussianBlur(g, (0,0), 2.3)
    dark_ridge = _normalise_percentile(np.maximum(local_bg-g, 0.0), 98.0)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0); gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    gradient = _normalise_percentile(cv2.magnitude(gx, gy), 97.5)
    structural_edge = gradient * np.clip((250.0 - np.minimum(g, local_bg)) / 92.0, 0.0, 1.0)
    block = int(round(min(gray.shape[:2]) * 0.09)); block = max(15, min(61, block)); block += 0 if block % 2 else 1
    adaptive = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,block,7)
    adaptive_score = (adaptive.astype(np.float32)/255.0) * np.maximum(dark_ridge, structural_edge*0.82)
    canny = cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),18,54,L2gradient=True)
    confidence = np.maximum.reduce([native, contrast*0.82, dark_ridge*0.90, structural_edge*0.78, adaptive_score*0.86]).astype(np.float32)
    base = ((confidence >= 0.30) | (gray <= 92) | (canny > 0)).astype(np.uint8) * 255
    base = _remove_texture_dots(base, confidence)
    repaired = _repair_short_gaps(base, gray, max_gap=max(1, int(gap_close)))
    barrier = np.where((base > 0) | (repaired > 0), 255, 0).astype(np.uint8)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, cross, iterations=1)
    # A single cross dilation closes pale anti-aliased shoulders while adding
    # much less false wall area than square/global dilation.
    barrier = cv2.dilate(barrier, cross, iterations=1)
    fillable = np.where(barrier > 0, 0, 255).astype(np.uint8)
    _, labels = cv2.connectedComponents(fillable, connectivity=4)
    return RegionBoundaryMap(labels.astype(np.int32), fillable, barrier, confidence)


def _seed_from_selection(raw: np.ndarray) -> tuple[int, int]:
    binary = (raw > 0).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if float(distance.max()) > 0:
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        return int(x), int(y)
    x, y, w, h = cv2.boundingRect(binary)
    return int(x + max(0, w-1)/2), int(y + max(0, h-1)/2)


def closed_region_from_selection(
    image_bgr: np.ndarray,
    raw_mask: np.ndarray,
    *,
    gap_close: int = 5,
    max_expand_px: int = 12,
    max_component_ratio: float = 0.96,
) -> tuple[np.ndarray, dict[str, object]]:
    """Extract the best truly closed line-art component inside a rough selection.

    This intentionally follows the topology contract used by Colortina's closed
    selection tool: the user's rectangle/lasso is *authority*, not a contour to
    warp.  Structural ink divides only that authority mask into paintable
    connected components.  Any component that can reach the one-pixel selection
    perimeter is open/background and is rejected.  The remaining enclosed
    components are scored for bubble-like coverage and centrality.

    ``max_expand_px`` is retained for schema/API compatibility.  Closed-region
    recognition itself never expands outside the user's selection; expansion is
    therefore zero-authority by construction and safer for manga panels.
    """
    if image_bgr is None or image_bgr.size == 0:
        raw = np.asarray(raw_mask, np.uint8)
        return raw, {"used_fallback": True, "reason": "empty_image", "method": "closed_region"}
    h, w = image_bgr.shape[:2]
    raw = np.asarray(raw_mask, np.uint8)
    if raw.shape != (h, w):
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    raw = np.where(raw > 0, 255, 0).astype(np.uint8)
    raw_area = int(cv2.countNonZero(raw))
    x, y, bw, bh = cv2.boundingRect(raw)
    if raw_area <= 0 or bw < 3 or bh < 3:
        return raw, {"used_fallback": True, "reason": "empty_selection", "method": "closed_region", "raw_area": raw_area}

    # Analyze a small context halo because a border stroke can sit exactly on
    # the rough selection boundary.  Topology itself is still clipped to the
    # exact user mask below, so no pixel outside authority can be selected.
    halo = max(18, int(round(max(bw, bh) * 0.08)), int(gap_close) * 3 + 8)
    xa, ya = max(0, x-halo), max(0, y-halo)
    xb, yb = min(w, x+bw+halo), min(h, y+bh+halo)
    crop = image_bgr[ya:yb, xa:xb]
    local_sel = np.where(raw[ya:yb, xa:xb] > 0, 1, 0).astype(np.uint8)
    topology = build_region_boundary_map(crop, gap_close=gap_close)

    # Colortina-style closed topology: structural lines are walls, while the
    # selection perimeter is the escape boundary.  Background/foliage that can
    # escape to the perimeter is open; a speech-balloon interior cannot.
    paintable = ((local_sel > 0) & (topology.barrier == 0)).astype(np.uint8)
    if not np.any(paintable):
        return raw, {"used_fallback": True, "reason": "no_paintable_region", "method": "closed_region", "raw_area": raw_area}
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(paintable, connectivity=4)
    if count <= 1:
        return raw, {"used_fallback": True, "reason": "no_closed_region", "method": "closed_region", "raw_area": raw_area}

    eroded = cv2.erode(local_sel, np.ones((3, 3), np.uint8), iterations=1)
    perimeter = (local_sel > 0) & (eroded == 0)
    open_labels = {
        int(v) for v in np.unique(labels[perimeter & (paintable > 0)])
        if int(v) > 0
    }

    # Prefer a substantial enclosed block near the authority centre.  Tiny
    # holes inside glyphs/screentone are intentionally filtered by both area and
    # coverage.  A closed bubble normally dominates the useful interior while
    # open foliage/panel background has already been removed above.
    sel_ys, sel_xs = np.nonzero(local_sel)
    cx = float(sel_xs.mean()); cy = float(sel_ys.mean())
    min_area = max(24, int(round(raw_area * 0.012)))
    candidates: list[tuple[float, int, int, float]] = []
    for rid in range(1, count):
        if rid in open_labels:
            continue
        area = int(stats[rid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp_w = int(stats[rid, cv2.CC_STAT_WIDTH]); comp_h = int(stats[rid, cv2.CC_STAT_HEIGHT])
        major = max(comp_w, comp_h); minor = max(1, min(comp_w, comp_h))
        aspect = major / float(minor)
        # Double edges around one long stroke can manufacture a thin enclosed
        # strip.  It is topologically closed but not a usable manga bubble.
        if aspect >= 4.5 and minor <= max(12, int(round(min(bw, bh) * 0.14))):
            continue
        ratio = area / float(max(1, raw_area))
        if ratio > float(np.clip(max_component_ratio, 0.25, 0.995)):
            continue
        rx = float(centroids[rid][0]); ry = float(centroids[rid][1])
        dx = (rx-cx) / max(1.0, float(bw)); dy = (ry-cy) / max(1.0, float(bh))
        distance = float((dx*dx + dy*dy) ** 0.5)
        # Area is strongest; centrality breaks ties and prevents selecting a
        # small decorative enclosed patch near a corner of a generous rough box.
        score = ratio * 4.0 - distance * 0.85
        candidates.append((score, rid, area, distance))

    if not candidates:
        return raw, {
            "used_fallback": True, "reason": "no_closed_region", "method": "closed_region",
            "raw_area": raw_area, "open_components": len(open_labels),
            "component_count": max(0, count-1),
            "roi_bbox": [int(xa), int(ya), int(xb), int(yb)],
            "roi_fraction": float(((xb-xa)*(yb-ya))/max(1,h*w)),
        }

    candidates.sort(reverse=True, key=lambda row: row[0])
    score, rid, cand_area, distance = candidates[0]
    candidate = np.where(labels == int(rid), 255, 0).astype(np.uint8)

    # Restore a tiny amount of the line shoulder *inside* the selection.  The
    # connected component stops on the detected black contour; a one-pixel
    # geodesic shoulder makes the visible selection read as a closed bubble
    # without ever crossing user authority or swallowing neighbouring artwork.
    if int(max_expand_px) > 0:
        allowed = local_sel > 0
        grown = cv2.dilate((candidate > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)), iterations=1) > 0
        candidate = np.where((candidate > 0) | (grown & allowed & (topology.confidence < 0.92)), 255, 0).astype(np.uint8)

    out = np.zeros_like(raw)
    out[ya:yb, xa:xb] = candidate
    # Hard authority invariant: recognition can only contract the user's rough
    # selection.  This makes accidental panel/background flood impossible.
    out[raw == 0] = 0
    snapped_area = int(cv2.countNonZero(out))
    if snapped_area <= 0:
        return raw, {"used_fallback": True, "reason": "empty_closed_region", "method": "closed_region", "raw_area": raw_area}

    changed = int(np.count_nonzero((out > 0) != (raw > 0)))
    return out, {
        "used_fallback": False,
        "method": "closed_region",
        "raw_area": raw_area,
        "snapped_area": snapped_area,
        "area_ratio": snapped_area / float(max(1, raw_area)),
        "changed_pixels": changed,
        "open_components": len(open_labels),
        "component_count": max(0, count-1),
        "closed_candidates": len(candidates),
        "selected_component": int(rid),
        "selected_score": float(score),
        "selected_center_distance": float(distance),
        "roi_bbox": [int(xa), int(ya), int(xb), int(yb)],
        "roi_fraction": float(((xb-xa)*(yb-ya))/max(1,h*w)),
        "gap_close": int(gap_close),
        "authority_contract": "inside_selection_only",
    }


__all__ = ["RegionBoundaryMap", "build_region_boundary_map", "closed_region_from_selection"]
