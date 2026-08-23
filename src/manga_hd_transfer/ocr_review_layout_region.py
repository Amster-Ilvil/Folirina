from __future__ import annotations

"""Review-overlay geometry recovery for OCR lettering containers.

The manual OCR rectangle identifies the source text island; it is not necessarily
where translated text should be typeset.  Recover a closed speech-bubble or
caption interior around that island and return an eroded safe mask.  If no closed
container can be proven, keep the user rectangle as an open-text layout envelope.

This deliberately uses only TARGET pixels and local geometry. It never invokes an
automatic transfer renderer.
"""

from typing import Any

import cv2
import numpy as np


def _clamp_bbox(shape: tuple[int, int], bbox: list[int] | tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    h, w = int(shape[0]), int(shape[1])
    if len(bbox) != 4:
        return None
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
    y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return x0, y0, x1, y1


def _rect_mask(shape: tuple[int, int], box: tuple[int, int, int, int], inset: int = 0) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    x0, y0, x1, y1 = box
    x0 = max(0, min(w, x0 + inset)); x1 = max(0, min(w, x1 - inset))
    y0 = max(0, min(h, y0 + inset)); y1 = max(0, min(h, y1 - inset))
    out = np.zeros((h, w), np.uint8)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = 255
    return out


def _touches_boundary(mask: np.ndarray, band: int = 2) -> float:
    if mask.size == 0 or cv2.countNonZero(mask) == 0:
        return 1.0
    band = max(1, min(int(band), max(1, min(mask.shape[:2]) // 5)))
    border = np.zeros_like(mask, np.uint8)
    border[:band, :] = 255; border[-band:, :] = 255
    border[:, :band] = 255; border[:, -band:] = 255
    touch = cv2.countNonZero(cv2.bitwise_and(mask, border))
    perimeter = max(1, 2 * (mask.shape[0] + mask.shape[1]) * band)
    return float(touch / perimeter)


def _seed_in_selection(free: np.ndarray, selection_local: tuple[int, int, int, int]) -> tuple[int, int] | None:
    x0, y0, x1, y1 = selection_local
    if x1 <= x0 or y1 <= y0:
        return None
    # Pick the point with the greatest clearance from Canny barriers. This avoids
    # landing directly on a Japanese glyph at the rectangle centre.
    dist = cv2.distanceTransform((free > 0).astype(np.uint8), cv2.DIST_L2, 5)
    roi = dist[y0:y1, x0:x1]
    if roi.size and float(roi.max()) > 0:
        iy, ix = np.unravel_index(int(np.argmax(roi)), roi.shape)
        return int(x0 + ix), int(y0 + iy)
    ys, xs = np.where(free[y0:y1, x0:x1] > 0)
    if len(xs):
        mid = len(xs) // 2
        return int(x0 + xs[mid]), int(y0 + ys[mid])
    return None


def _component_from_seed(binary: np.ndarray, seed: tuple[int, int]) -> np.ndarray | None:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    sx, sy = seed
    if not (0 <= sx < binary.shape[1] and 0 <= sy < binary.shape[0]):
        return None
    lab = int(labels[sy, sx])
    if lab <= 0 or lab >= n:
        return None
    out = np.zeros_like(binary, np.uint8)
    out[labels == lab] = 255
    return out


def _best_component_in_selection(
    binary: np.ndarray,
    selection_local: tuple[int, int, int, int],
) -> tuple[np.ndarray | None, tuple[int, int] | None, dict[str, Any]]:
    """Choose a likely *closed interior* component overlapping the OCR box.

    A largest-distance seed can land outside the balloon when the user's OCR
    rectangle extends beyond a curved border.  Enumerate every connected free
    component that actually overlaps the selection and prefer components that
    do not touch the analysis-crop boundary.  This keeps the selection a search
    hint while letting a smaller enclosed interior win over the huge page/panel
    background.
    """
    x0, y0, x1, y1 = selection_local
    if x1 <= x0 or y1 <= y0:
        return None, None, {"component_candidates": 0}
    n, labels, stats, centroids = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    sel_labels, sel_counts = np.unique(labels[y0:y1, x0:x1], return_counts=True)
    sel_area = max(1, (x1 - x0) * (y1 - y0))
    rows: list[tuple[float, np.ndarray, tuple[int, int], dict[str, Any]]] = []
    for lab_raw, overlap_raw in zip(sel_labels.tolist(), sel_counts.tolist()):
        lab = int(lab_raw); overlap = int(overlap_raw)
        if lab <= 0 or overlap <= 0 or lab >= n:
            continue
        comp = np.zeros_like(binary, np.uint8); comp[labels == lab] = 255
        filled = _fill_external_shape(comp)
        area = max(1, int(cv2.countNonZero(filled)))
        touch = _touches_boundary(filled)
        overlap_ratio = float(overlap / sel_area)
        candidate_overlap = float(overlap / area)
        # Closure dominates. Among closed candidates, prefer strong selection
        # overlap and modest component size. External page space normally has a
        # boundary touch close to 1 and is therefore heavily penalized.
        score = overlap_ratio + 0.32 * min(1.0, candidate_overlap) - 4.5 * touch
        cx, cy = centroids[lab]
        seed = (int(round(float(cx))), int(round(float(cy))))
        rows.append((score, filled, seed, {
            "component_label": lab,
            "component_overlap_pixels": overlap,
            "component_overlap_ratio": overlap_ratio,
            "component_selection_share": candidate_overlap,
            "component_boundary_touch": float(touch),
        }))
    if not rows:
        return None, None, {"component_candidates": 0}
    rows.sort(key=lambda row: row[0], reverse=True)
    score, comp, seed, diag = rows[0]
    return comp, seed, {"component_candidates": len(rows), "component_score": float(score), **diag}


def _fill_external_shape(component: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours((component > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return component
    contour = max(contours, key=cv2.contourArea)
    out = np.zeros_like(component, np.uint8)
    cv2.drawContours(out, [contour], -1, 255, thickness=cv2.FILLED)
    return out


def _edge_container(crop: np.ndarray, selection_local: tuple[int, int, int, int]) -> tuple[np.ndarray | None, dict[str, Any]]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Median-derived thresholds remain stable for white manga paper and tolerate
    # lightly coloured bubbles/captions.
    med = float(np.median(gray))
    low = int(np.clip(med * 0.22, 28, 82))
    high = int(np.clip(max(low + 35, med * 0.52), 82, 170))
    edges = cv2.Canny(gray, low, high, L2gradient=True)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, k3, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k3, iterations=1)
    free = np.where(edges > 0, 0, 255).astype(np.uint8)
    filled, seed, component_diag = _best_component_in_selection(free, selection_local)
    if filled is None or seed is None:
        return None, {"source": "edge", "reason": "no_selection_component", **component_diag}
    return filled, {
        "source": "edge", "reason": "candidate", "seed": list(seed),
        "edge_low": low, "edge_high": high, "boundary_touch": _touches_boundary(filled),
        **component_diag,
    }


def _photometric_container(crop: np.ndarray, selection_local: tuple[int, int, int, int]) -> tuple[np.ndarray | None, dict[str, Any]]:
    # A second path covers low-contrast / coloured outlines. Estimate the local
    # paper colour from the brightest low-saturation pixels inside the selection,
    # then connect nearby pixels in Lab colour space.
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    x0, y0, x1, y1 = selection_local
    sub = crop[y0:y1, x0:x1]
    sub_hsv = hsv[y0:y1, x0:x1]
    if sub.size == 0:
        return None, {"source": "photo", "reason": "empty_selection"}
    score = sub_hsv[..., 2].astype(np.float32) - 0.45 * sub_hsv[..., 1].astype(np.float32)
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    sx, sy = int(x0 + ix), int(y0 + iy)
    seed_lab = lab[sy, sx].astype(np.int16)
    diff = lab.astype(np.int16) - seed_lab[None, None, :]
    # L is allowed to vary more than chroma so black source glyphs remain holes
    # while shaded/cream paper remains connected.
    d = np.sqrt((diff[..., 0] * 0.60) ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2)
    similar = (d <= 34.0).astype(np.uint8) * 255
    similar = cv2.morphologyEx(similar, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    filled, seed, component_diag = _best_component_in_selection(similar, selection_local)
    if filled is None or seed is None:
        return None, {"source": "photo", "reason": "no_selection_component", **component_diag}
    return filled, {
        "source": "photo", "reason": "candidate", "seed": list(seed),
        "boundary_touch": _touches_boundary(filled),
        **component_diag,
    }


def _candidate_quality(mask: np.ndarray, selection_local: tuple[int, int, int, int]) -> tuple[bool, dict[str, Any]]:
    area = int(cv2.countNonZero(mask))
    h, w = mask.shape[:2]
    roi_area = max(1, h * w)
    x0, y0, x1, y1 = selection_local
    sel_area = max(1, (x1 - x0) * (y1 - y0))
    overlap = int(cv2.countNonZero(mask[y0:y1, x0:x1]))
    overlap_ratio = float(overlap / sel_area)
    overlap_candidate_ratio = float(overlap / max(1, area))
    bx, by, bw, bh = cv2.boundingRect((mask > 0).astype(np.uint8)) if area else (0, 0, 0, 0)
    bbox_area = max(1, bw * bh)
    fill_ratio = float(area / bbox_area)
    boundary_touch = _touches_boundary(mask)
    area_ratio_selection = float(area / sel_area)
    area_ratio_roi = float(area / roi_area)
    sel_w, sel_h = max(1, x1 - x0), max(1, y1 - y0)
    bbox_width_ratio = float(bw / sel_w)
    bbox_height_ratio = float(bh / sel_h)
    # The manual OCR rectangle is a *search/clear* rectangle, not a balloon bbox.
    # Users often draw it slightly outside a speech bubble, so a valid closed
    # interior may legitimately be smaller than the selection.  Accept either
    # direction of strong containment, but reject panel-sized free components.
    # This mirrors the containment logic used by mature manga lettering tools:
    # prove enclosure first, then typeset against the recovered shape.
    strong_containment = bool(
        overlap_ratio >= 0.68 or overlap_candidate_ratio >= 0.72
    )
    panel_like_expansion = bool(
        area_ratio_selection > 7.5
        or bbox_width_ratio > 3.4
        or bbox_height_ratio > 3.4
    )
    valid = bool(
        area > 0
        and strong_containment
        and overlap_candidate_ratio >= 0.58
        and area_ratio_selection >= 0.38
        and area_ratio_roi <= 0.90
        and boundary_touch <= 0.035
        and not panel_like_expansion
        and bw >= max(6, int(sel_w * 0.68))
        and bh >= max(6, int(sel_h * 0.68))
    )
    return valid, {
        "area": area, "overlap_ratio": overlap_ratio,
        "overlap_candidate_ratio": overlap_candidate_ratio,
        "area_ratio_selection": area_ratio_selection, "area_ratio_roi": area_ratio_roi,
        "bbox_width_ratio": bbox_width_ratio, "bbox_height_ratio": bbox_height_ratio,
        "fill_ratio": fill_ratio, "boundary_touch": boundary_touch,
        "bbox": [int(bx), int(by), int(bx + bw), int(by + bh)],
    }


def _safe_erode(mask: np.ndarray, kind: str) -> tuple[np.ndarray, int]:
    if cv2.countNonZero(mask) == 0:
        return mask, 0
    x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8))
    short = max(1, min(w, h))
    ratio = 0.026 if kind == "textbox" else 0.038
    margin = max(2, int(round(short * ratio)))
    # Do not erase small bubbles entirely.
    margin = min(margin, max(2, short // 9))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    eroded = cv2.erode(mask, k, iterations=1)
    if cv2.countNonZero(eroded) < cv2.countNonZero(mask) * 0.38:
        margin = max(1, margin // 2)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        eroded = cv2.erode(mask, k, iterations=1)
    return (eroded if cv2.countNonZero(eroded) else mask), margin


def recover_layout_region(
    target: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
    *,
    open_inset: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a page-sized safe mask and diagnostics for OCR typesetting.

    Closed speech balloons and caption boxes use their recovered interior.
    Unproven/open text keeps the user's selected rectangle as the authority.
    """
    shape = target.shape[:2]
    box = _clamp_bbox(shape, bbox)
    if box is None:
        return np.zeros(shape, np.uint8), {"layout_kind": "invalid", "reason": "invalid_bbox"}
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    # Retry with a generous local crop. A closed bubble should fit completely
    # inside it; an open/panel region naturally reaches the crop boundary.
    padx = max(24, int(round(bw * 1.15)))
    pady = max(24, int(round(bh * 1.15)))
    rx0 = max(0, x0 - padx); ry0 = max(0, y0 - pady)
    rx1 = min(shape[1], x1 + padx); ry1 = min(shape[0], y1 + pady)
    crop = target[ry0:ry1, rx0:rx1]
    local_sel = (x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0)

    candidates: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    for builder in (_edge_container, _photometric_container):
        cand, diag = builder(crop, local_sel)
        if cand is None:
            continue
        valid, q = _candidate_quality(cand, local_sel)
        diag = {**diag, **q, "valid": valid}
        if valid:
            # Prefer high selection coverage, modest area expansion, and closure.
            score = q["overlap_ratio"] + min(1.0, q["area_ratio_selection"] / 4.0) - q["boundary_touch"] * 5.0
            candidates.append((float(score), cand, diag))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        _score, local, diag = candidates[0]
        area = int(cv2.countNonZero(local))
        bx, by, ww, hh = cv2.boundingRect((local > 0).astype(np.uint8))
        fill_ratio = float(area / max(1, ww * hh))
        contours, _ = cv2.findContours((local > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        approx_vertices = 0
        if contour is not None:
            peri = max(1.0, cv2.arcLength(contour, True))
            approx_vertices = len(cv2.approxPolyDP(contour, 0.025 * peri, True))
        # Rectangular caption/text boxes have high rectangular occupancy and a
        # small polygon vertex count. Everything else is treated as a bubble.
        kind = "textbox" if fill_ratio >= 0.86 and 4 <= approx_vertices <= 8 else "bubble"
        local_safe, margin = _safe_erode(local, kind)
        page = np.zeros(shape, np.uint8)
        page[ry0:ry1, rx0:rx1] = local_safe
        return page, {
            "layout_kind": kind,
            "layout_source": str(diag.get("source") or "container"),
            "container_bbox": [int(rx0 + bx), int(ry0 + by), int(rx0 + bx + ww), int(ry0 + by + hh)],
            "container_fill_ratio": fill_ratio,
            "approx_vertices": int(approx_vertices),
            "safe_margin_px": int(margin),
            "safe_pixels": int(cv2.countNonZero(page)),
            "selection_bbox": [x0, y0, x1, y1],
            "candidate": diag,
        }

    # Open lettering has no proven enclosing border. Do not invent one and do
    # not spill into neighbouring artwork: the manual selection remains the hard
    # layout fence. This also covers SFX and free-floating captions.
    page = _rect_mask(shape, box, inset=max(0, int(open_inset)))
    return page, {
        "layout_kind": "open",
        "layout_source": "manual_selection",
        "selection_bbox": [x0, y0, x1, y1],
        "safe_margin_px": int(max(0, open_inset)),
        "safe_pixels": int(cv2.countNonZero(page)),
    }


def infer_text_orientation(
    target: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
) -> tuple[str, dict[str, Any]]:
    """Estimate horizontal/vertical flow from TARGET glyph geometry.

    OCR engines frequently return one coarse polygon for a whole Japanese text
    block, so polygon aspect alone is unreliable.  Count compact dark components
    and compare how many stable x/y bands their centroids form.  Vertical manga
    text has few x columns and many y rows; horizontal text shows the inverse.
    Ambiguous regions fall back to the selection aspect rather than guessing.
    """
    box = _clamp_bbox(target.shape[:2], bbox)
    if box is None:
        return "horizontal", {"source": "invalid_bbox", "confidence": 0.0}
    x0, y0, x1, y1 = box
    crop = target[y0:y1, x0:x1]
    if crop.size == 0:
        return "horizontal", {"source": "empty_crop", "confidence": 0.0}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    # Keep true ink while avoiding light halftone. Otsu alone can become too
    # permissive on shaded panels, so cap it by a conservative percentile rule.
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = int(min(188, max(92, min(float(otsu), float(np.percentile(gray, 38)) + 42.0))))
    dark = (gray < threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    ch, cw = gray.shape[:2]
    area_total = max(1, ch * cw)
    components: list[tuple[float, float, int, int]] = []
    edge = max(1, int(round(min(ch, cw) * 0.012)))
    for lab in range(1, count):
        x, y, ww, hh, area = [int(v) for v in stats[lab]]
        if area < 3 or area > area_total * 0.055:
            continue
        if x <= edge or y <= edge or x + ww >= cw - edge or y + hh >= ch - edge:
            continue
        if ww > cw * 0.34 or hh > ch * 0.34:
            continue
        cx, cy = centroids[lab]
        components.append((float(cx), float(cy), ww, hh))
    if len(components) < 5:
        aspect = ch / max(1.0, float(cw))
        orientation = "vertical" if aspect >= 1.72 else "horizontal"
        return orientation, {
            "source": "aspect_fallback", "confidence": 0.25,
            "component_count": len(components), "aspect": float(aspect),
        }

    med_w = float(np.median([c[2] for c in components]))
    med_h = float(np.median([c[3] for c in components]))

    def _cluster_count(values: list[float], tolerance: float) -> int:
        values = sorted(values)
        if not values:
            return 0
        groups = 1
        prev = values[0]
        for value in values[1:]:
            if value - prev > tolerance:
                groups += 1
            prev = value
        return groups

    x_bands = _cluster_count([c[0] for c in components], max(4.0, med_w * 0.82))
    y_bands = _cluster_count([c[1] for c in components], max(4.0, med_h * 0.82))
    denom = max(1.0, float(max(x_bands, y_bands)))
    confidence = abs(float(y_bands - x_bands)) / denom
    if y_bands >= x_bands + 2 and y_bands >= x_bands * 1.22:
        orientation = "vertical"
        source = "glyph_bands"
    elif x_bands >= y_bands + 2 and x_bands >= y_bands * 1.22:
        orientation = "horizontal"
        source = "glyph_bands"
    else:
        aspect = ch / max(1.0, float(cw))
        orientation = "vertical" if aspect >= 1.72 else "horizontal"
        source = "aspect_ambiguous"
        confidence *= 0.55
    return orientation, {
        "source": source, "confidence": float(confidence),
        "component_count": len(components), "x_bands": int(x_bands),
        "y_bands": int(y_bands), "median_component_w": med_w,
        "median_component_h": med_h,
    }


__all__ = ["recover_layout_region", "infer_text_orientation"]
