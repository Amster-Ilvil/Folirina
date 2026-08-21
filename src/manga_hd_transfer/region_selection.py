from __future__ import annotations

"""Serializable region-selection geometry for the review workbench.

The GUI is allowed to use convenient coarse shapes, but every renderer receives a
full-resolution TARGET-space uint8 mask.  This keeps selection semantics separate
from Direct/Mask/Reveal/OCR algorithms and makes it possible to stack tools on the
same page without changing the page's automatic transfer mode.
"""

from typing import Any

import cv2
import numpy as np

from .schema_compat import as_dict, as_list
from .region_boundary import closed_region_from_selection

SELECTION_SCHEMA = "folirina.region_selection.v1"
SELECTION_KINDS = {"rect", "ellipse", "freehand", "smart"}


def _clamp_point(point: Any, w: int, h: int) -> tuple[int, int] | None:
    try:
        x, y = point
        return max(0, min(w - 1, int(round(float(x))))), max(0, min(h - 1, int(round(float(y)))))
    except Exception:
        return None


def _normalise_bbox(raw: Any, w: int, h: int) -> list[int]:
    vals = as_list(raw)
    if len(vals) != 4:
        return []
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in vals]
    except Exception:
        return []
    x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
    y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
    return [x0, y0, x1, y1] if x1 - x0 >= 2 and y1 - y0 >= 2 else []


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    """Return the tight non-zero bbox without materialising full-page x/y grids.

    ``np.where`` on a 4K manga mask can allocate tens of megabytes merely to
    compute a box during mouse interaction.  OpenCV's image ``boundingRect``
    scans the same pixels in native code and returns the rectangle directly.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        return []
    binary = arr if arr.dtype == np.uint8 else np.where(arr > 0, 255, 0).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(binary)
    if w <= 0 or h <= 0:
        return []
    return [int(x), int(y), int(x + w), int(y + h)]


def selection_bbox_from_spec(spec: dict[str, Any] | None, shape: tuple[int, int]) -> list[int]:
    """Compute a selection bbox from serialized geometry without raster scanning."""
    h, w = map(int, shape)
    data = as_dict(spec)
    kind = str(data.get("kind") or "rect").strip().lower()
    raw_points = as_list(data.get("points"))
    points = [p for p in (_clamp_point(v, w, h) for v in raw_points) if p is not None]
    if kind in {"freehand", "smart"} and len(points) >= 3:
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs) + 1, max(ys) + 1
        return _normalise_bbox([x0, y0, x1, y1], w, h)
    return _normalise_bbox(data.get("bbox"), w, h)


def _largest_contour_points(mask: np.ndarray, max_points: int = 180) -> list[list[int]]:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    peri = max(1.0, cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, max(0.7, peri * 0.0035), True).reshape(-1, 2)
    if len(approx) > max_points:
        step = max(1, int(np.ceil(len(approx) / max_points)))
        approx = approx[::step]
    return [[int(x), int(y)] for x, y in approx]


def selection_mask_from_spec(
    spec: dict[str, Any] | None,
    shape: tuple[int, int],
    *,
    out: np.ndarray | None = None,
    clear_bbox: list[int] | tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Rasterise a selection, optionally reusing an existing full-page buffer.

    The reusable path is primarily for Qt mouse-drag interaction.  When
    ``clear_bbox`` is supplied only the previous selection rectangle is cleared
    rather than zeroing a 20-30 megapixel page on every mouse move.
    """
    h, w = map(int, shape)
    if isinstance(out, np.ndarray) and out.shape == (h, w) and out.dtype == np.uint8:
        mask = out
        old = _normalise_bbox(clear_bbox, w, h) if clear_bbox else []
        if old:
            ox0, oy0, ox1, oy1 = old; mask[oy0:oy1, ox0:ox1] = 0
        else:
            mask.fill(0)
    else:
        mask = np.zeros((h, w), np.uint8)
    data = as_dict(spec)
    kind = str(data.get("kind") or "rect").strip().lower()
    if kind not in SELECTION_KINDS:
        kind = "rect"
    bbox = _normalise_bbox(data.get("bbox"), w, h)
    raw_points = as_list(data.get("points"))
    points = [p for p in (_clamp_point(v, w, h) for v in raw_points) if p is not None]

    if kind in {"freehand", "smart"} and len(points) >= 3:
        cv2.fillPoly(mask, [np.asarray(points, np.int32)], 255)
    elif bbox:
        x0, y0, x1, y1 = bbox
        if kind == "ellipse":
            # Use half-pixel fixed-point coordinates so an ellipse with bbox
            # [x0,y0,x1,y1] never paints the x1/y1 pixel.  The previous rounded
            # centre/radius could spill one pixel beyond even-sized bboxes,
            # leaving a stale edge when the reusable drag buffer cleared only
            # the cached old bbox.
            cx2 = x0 + x1 - 1; cy2 = y0 + y1 - 1
            ax2 = max(1, x1 - x0 - 1); ay2 = max(1, y1 - y0 - 1)
            cv2.ellipse(mask, (cx2, cy2), (ax2, ay2), 0.0, 0.0, 360.0, 255, -1, lineType=cv2.LINE_8, shift=1)
        else:
            mask[y0:y1, x0:x1] = 255
    return mask


def warp_selection_mask(
    mask: np.ndarray,
    source_to_target: np.ndarray,
    output_shape: tuple[int, int],
    *,
    target_to_source: bool = True,
) -> np.ndarray:
    """Project one binary selection mask between TARGET and SOURCE coordinates."""
    out_h, out_w = map(int, output_shape)
    H = np.asarray(source_to_target, dtype=np.float64)
    if H.shape == (2, 3):
        tmp = np.eye(3, dtype=np.float64); tmp[:2, :] = H; H = tmp
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        H = np.eye(3, dtype=np.float64)
    if target_to_source:
        try:
            H = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H = np.eye(3, dtype=np.float64)
    src = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    return cv2.warpPerspective(
        src, H, (out_w, out_h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def selection_mask_from_row(row: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    data = as_dict(row)
    spec = as_dict(data.get("selection_spec"))
    if spec:
        mask = selection_mask_from_spec(spec, shape)
        if cv2.countNonZero(mask) > 0:
            return mask
    h, w = map(int, shape)
    bbox = _normalise_bbox(data.get("target_bbox"), w, h)
    fallback = np.zeros((h, w), np.uint8)
    if bbox:
        x0, y0, x1, y1 = bbox
        fallback[y0:y1, x0:x1] = 255
    return fallback


def spec_from_mask(mask: np.ndarray, *, kind: str = "smart", snapped: bool = False,
                   diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    bbox = bbox_from_mask(binary)
    return {
        "schema": SELECTION_SCHEMA,
        "kind": kind if kind in SELECTION_KINDS else "smart",
        "bbox": bbox,
        "points": _largest_contour_points(binary),
        "snapped": bool(snapped),
        "diagnostics": dict(diagnostics or {}),
    }



def project_selection_spec(
    spec: dict[str, Any] | None,
    source_to_target: np.ndarray,
    output_shape: tuple[int, int],
    *,
    target_to_source: bool = True,
) -> dict[str, Any]:
    """Project selection boundary points without warping a full-resolution mask.

    This is used only for the visual SOURCE reference.  Rectangles become a
    quadrilateral, ellipses are sampled around their perimeter, and free/smart
    polygons retain their vertices.  The committed TARGET selection itself is
    never changed by this helper.
    """
    data = as_dict(spec)
    out_h, out_w = map(int, output_shape)
    if out_h <= 0 or out_w <= 0:
        return {}
    kind = str(data.get("kind") or "rect").strip().lower()
    bbox = _normalise_bbox(data.get("bbox"), 1_000_000_000, 1_000_000_000)
    raw_points = as_list(data.get("points"))
    points: list[tuple[float, float]] = []
    if kind in {"freehand", "smart"} and len(raw_points) >= 3:
        for raw in raw_points:
            try:
                x, y = raw; points.append((float(x), float(y)))
            except (TypeError, ValueError):
                continue
    elif bbox:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        if kind == "ellipse":
            cx = (x0 + x1 - 1.0) * 0.5; cy = (y0 + y1 - 1.0) * 0.5
            rx = max(1.0, (x1 - x0) * 0.5); ry = max(1.0, (y1 - y0) * 0.5)
            for theta in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
                points.append((cx + rx * float(np.cos(theta)), cy + ry * float(np.sin(theta))))
        else:
            points = [(x0, y0), (x1 - 1.0, y0), (x1 - 1.0, y1 - 1.0), (x0, y1 - 1.0)]
    if len(points) < 3:
        return {}
    H = np.asarray(source_to_target, dtype=np.float64)
    if H.shape == (2, 3):
        tmp = np.eye(3, dtype=np.float64); tmp[:2, :] = H; H = tmp
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        H = np.eye(3, dtype=np.float64)
    if target_to_source:
        try:
            H = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H = np.eye(3, dtype=np.float64)
    pts = np.asarray([points], np.float32)
    mapped = cv2.perspectiveTransform(pts, H.astype(np.float64))[0]
    out_points: list[list[int]] = []
    for x, y in mapped:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        out_points.append([max(0, min(out_w - 1, int(round(float(x))))), max(0, min(out_h - 1, int(round(float(y)))) )])
    if len(out_points) < 3:
        return {}
    xs = [p[0] for p in out_points]; ys = [p[1] for p in out_points]
    bbox_out = _normalise_bbox([min(xs), min(ys), max(xs) + 1, max(ys) + 1], out_w, out_h)
    if not bbox_out:
        return {}
    return {
        "schema": SELECTION_SCHEMA,
        "kind": "smart" if kind == "smart" else "freehand",
        "bbox": bbox_out,
        "points": out_points,
        "snapped": bool(data.get("snapped", False)),
        "reference_projection": True,
    }

def _snap_selection_mask_to_lineart_full(image_bgr: np.ndarray, raw_mask: np.ndarray, *,
                                   max_distance: int = 10,
                                   max_growth_ratio: float = 1.55) -> tuple[np.ndarray, dict[str, Any]]:
    """Snap only the boundary band of a coarse selection toward nearby manga ink.

    Marker-controlled watershed is deliberately bounded by an eroded/dilated
    band.  A weak line signal, excessive area drift or no measurable boundary
    improvement falls back to the user's exact selection.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.asarray(raw_mask, np.uint8), {"used_fallback": True, "reason": "empty_image"}
    h, w = image_bgr.shape[:2]
    raw = np.asarray(raw_mask, np.uint8)
    if raw.shape != (h, w):
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    raw = np.where(raw > 0, 255, 0).astype(np.uint8)
    raw_area = int(cv2.countNonZero(raw))
    distance = max(0, int(max_distance))
    if raw_area <= 0 or distance <= 0:
        return raw, {"used_fallback": False, "raw_area": raw_area, "snapped_area": raw_area, "max_distance": distance}

    radius = max(1, distance)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    binary = (raw > 0).astype(np.uint8)
    outer = cv2.dilate(binary, kernel, iterations=1)
    inner = cv2.erode(binary, kernel, iterations=1)
    core_radius = max(1, min(8, int(round(distance * 0.55))))
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_radius * 2 + 1, core_radius * 2 + 1))
    core = cv2.erode(binary, core_kernel, iterations=1)
    if not np.any(core):
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        if float(dist.max()) <= 0:
            return raw, {"used_fallback": True, "reason": "no_seed", "raw_area": raw_area}
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        core = np.zeros_like(binary); cv2.circle(core, (int(x), int(y)), 2, 1, -1)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr.astype(np.uint8)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    grad = cv2.magnitude(gx, gy)
    if float(grad.max()) > 1e-6:
        grad *= 255.0 / float(grad.max())
    # Dark ink and gradients both create watershed ridges. Otsu is only used to
    # locate structural ink; it never directly becomes a selection mask.
    _thr, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elevation = np.maximum(np.clip(grad, 0, 255).astype(np.uint8), dark)
    elevation = cv2.GaussianBlur(elevation, (3, 3), 0)
    band = (outer > 0) & (inner == 0)
    if int(np.count_nonzero((dark > 0) & band)) < max(3, distance // 2):
        return raw, {"used_fallback": True, "reason": "no_nearby_ink", "raw_area": raw_area}

    markers = np.zeros((h, w), np.int32)
    markers[outer == 0] = 1
    markers[core > 0] = 2
    cv2.watershed(cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR), markers)
    candidate = ((markers == 2) & (outer > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if count > 1:
        overlaps = np.bincount(labels[core > 0].ravel(), minlength=count)
        ids = [i for i in range(1, count) if int(overlaps[i]) > 0]
        if ids:
            keep = max(ids, key=lambda i: (int(overlaps[i]), int(stats[i, cv2.CC_STAT_AREA])))
            candidate = (labels == keep).astype(np.uint8)
    out = candidate * 255
    area = int(cv2.countNonZero(out))
    if area <= 0:
        return raw, {"used_fallback": True, "reason": "empty_watershed", "raw_area": raw_area}
    ratio = area / float(max(1, raw_area))
    max_ratio = max(1.05, float(max_growth_ratio)); min_ratio = max(0.28, 1.0 / (max_ratio * 1.55))
    if ratio < min_ratio or ratio > max_ratio:
        return raw, {"used_fallback": True, "reason": "area_drift", "raw_area": raw_area, "candidate_area": area, "area_ratio": ratio}

    line = (dark > 0).astype(np.uint8)
    line_distance = cv2.distanceTransform(np.where(line > 0, 0, 1).astype(np.uint8), cv2.DIST_L2, 5)
    edge_kernel = np.ones((3, 3), np.uint8)
    raw_edge = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, edge_kernel) > 0
    out_edge = cv2.morphologyEx(candidate, cv2.MORPH_GRADIENT, edge_kernel) > 0
    raw_values = line_distance[raw_edge]; out_values = line_distance[out_edge]
    raw_mean = float(np.mean(np.minimum(raw_values, distance + 2))) if raw_values.size else float(distance + 2)
    out_mean = float(np.mean(np.minimum(out_values, distance + 2))) if out_values.size else float(distance + 2)
    raw_near = float(np.mean(raw_values <= 2.25)) if raw_values.size else 0.0
    out_near = float(np.mean(out_values <= 2.25)) if out_values.size else 0.0
    changed = int(np.count_nonzero((out > 0) != (raw > 0)))
    improved = out_mean <= raw_mean - 0.20 or out_near >= raw_near + 0.035 or (out_near >= 0.30 and out_mean <= raw_mean + 0.10)
    if not improved or changed < max(6, int(round(np.sqrt(raw_area) * 0.35))):
        return raw, {"used_fallback": True, "reason": "no_boundary_improvement", "raw_area": raw_area, "raw_mean": raw_mean, "candidate_mean": out_mean}
    return out, {
        "used_fallback": False, "raw_area": raw_area, "snapped_area": area,
        "max_distance": distance, "area_ratio": ratio, "changed_pixels": changed,
        "raw_boundary_distance": raw_mean, "snapped_boundary_distance": out_mean,
        "raw_near_fraction": raw_near, "snapped_near_fraction": out_near,
    }


def snap_selection_mask_to_lineart(image_bgr: np.ndarray, raw_mask: np.ndarray, *,
                                   max_distance: int = 10,
                                   max_growth_ratio: float = 1.55) -> tuple[np.ndarray, dict[str, Any]]:
    """Snap a coarse selection using only a local boundary neighbourhood.

    v2.3.41 ran Sobel/Otsu/watershed on the whole manga page for every completed
    selection. On 4K pages this made the selection system feel unresponsive.
    The exact same bounded snap algorithm now runs on an ROI that contains the
    selection plus a generous halo, then is pasted back to TARGET coordinates.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.asarray(raw_mask, np.uint8), {"used_fallback": True, "reason": "empty_image"}
    h, w = image_bgr.shape[:2]
    raw = np.asarray(raw_mask, np.uint8)
    if raw.shape != (h, w):
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    raw = np.where(raw > 0, 255, 0).astype(np.uint8)
    box = bbox_from_mask(raw)
    distance = max(0, int(max_distance))
    if len(box) != 4 or distance <= 0:
        area = int(cv2.countNonZero(raw))
        return raw, {"used_fallback": False, "raw_area": area, "snapped_area": area, "max_distance": distance, "roi_fast_path": True}
    x0, y0, x1, y1 = box
    # Watershed may move only max_distance from the raw boundary. Extra halo
    # keeps gradient / threshold operations independent from the crop border.
    halo = max(24, distance * 3 + 12)
    xa, ya = max(0, x0 - halo), max(0, y0 - halo)
    xb, yb = min(w, x1 + halo), min(h, y1 + halo)
    roi_pixels = max(1, (xb - xa) * (yb - ya)); full_pixels = max(1, h * w)
    if roi_pixels >= int(full_pixels * 0.82):
        out, diag = _snap_selection_mask_to_lineart_full(
            image_bgr, raw, max_distance=distance, max_growth_ratio=max_growth_ratio
        )
        diag = dict(diag or {}); diag.update({"roi_fast_path": False, "roi_bbox": [0,0,w,h], "roi_fraction": 1.0})
        return out, diag
    crop_img = image_bgr[ya:yb, xa:xb]
    crop_raw = raw[ya:yb, xa:xb]
    crop_out, diag = _snap_selection_mask_to_lineart_full(
        crop_img, crop_raw, max_distance=distance, max_growth_ratio=max_growth_ratio
    )
    out = raw.copy(); out[ya:yb, xa:xb] = crop_out
    diag = dict(diag or {})
    diag.update({
        "roi_fast_path": True, "roi_bbox": [int(xa),int(ya),int(xb),int(yb)],
        "roi_pixels": int(roi_pixels), "full_page_pixels": int(full_pixels),
        "roi_fraction": float(roi_pixels / full_pixels), "roi_halo": int(halo),
    })
    return out, diag


def recognize_closed_region_from_selection(
    image_bgr: np.ndarray,
    raw_mask: np.ndarray,
    *,
    gap_close: int = 5,
    max_expand_px: int = 12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recognize one closed manga region from a coarse user authority mask.

    This is intentionally different from ``snap_selection_mask_to_lineart``:
    boundary snap adjusts an already-good lasso, while closed-region recognition
    finds the actual connected interior of a speech/effect bubble.  The latter
    mirrors Colortina's topology-first workflow and is what the UI's explicit
    "识别当前选框的闭合区域" action needs.
    """
    return closed_region_from_selection(
        image_bgr, raw_mask, gap_close=gap_close, max_expand_px=max_expand_px
    )


__all__ = [
    "SELECTION_SCHEMA", "SELECTION_KINDS", "selection_mask_from_spec",
    "selection_mask_from_row", "selection_bbox_from_spec", "bbox_from_mask", "spec_from_mask",
    "warp_selection_mask", "project_selection_spec", "snap_selection_mask_to_lineart",
    "recognize_closed_region_from_selection",
]
