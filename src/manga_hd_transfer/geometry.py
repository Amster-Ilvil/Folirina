from __future__ import annotations

from math import hypot
from typing import Iterable, Sequence

import cv2
import numpy as np

Point = tuple[float, float]
Polygon = list[Point]


def as_points(poly: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(poly, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"polygon must have shape (N,2), got {arr.shape}")
    return arr


def polygon_bbox(poly: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    pts = as_points(poly)
    x0, y0 = np.min(pts, axis=0)
    x1, y1 = np.max(pts, axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def bbox_polygon(box: Sequence[float]) -> Polygon:
    x0, y0, x1, y1 = map(float, box)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def polygon_centroid(poly: Sequence[Sequence[float]]) -> Point:
    pts = as_points(poly)
    if len(pts) < 3:
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    moments = cv2.moments(pts)
    if abs(moments["m00"]) < 1e-8:
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def polygon_area(poly: Sequence[Sequence[float]]) -> float:
    pts = as_points(poly)
    if len(pts) < 3:
        return 0.0
    return float(abs(cv2.contourArea(pts)))


def bbox_area(box: Sequence[float]) -> float:
    x0, y0, x1, y1 = map(float, box)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = map(float, a)
    bx0, by0, bx1, by1 = map(float, b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def overlap_over_smaller(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = map(float, a)
    bx0, by0, bx1, by1 = map(float, b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    denom = min(bbox_area(a), bbox_area(b))
    return inter / denom if denom > 0 else 0.0


def normalized_centroid_distance(
    a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], image_size: tuple[int, int]
) -> float:
    ax, ay = polygon_centroid(a)
    bx, by = polygon_centroid(b)
    w, h = image_size
    diag = hypot(max(w, 1), max(h, 1))
    return hypot(ax - bx, ay - by) / diag


def transform_points(poly: Sequence[Sequence[float]], matrix: np.ndarray) -> Polygon:
    pts = as_points(poly).reshape(1, -1, 2)
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape == (2, 3):
        out = cv2.transform(pts, m)[0]
    elif m.shape == (3, 3):
        out = cv2.perspectiveTransform(pts, m)[0]
    else:
        raise ValueError(f"matrix must be 2x3 or 3x3, got {m.shape}")
    return [(float(x), float(y)) for x, y in out]


def transform_to_homography(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape == (3, 3):
        return m
    if m.shape == (2, 3):
        return np.vstack([m, [0.0, 0.0, 1.0]])
    raise ValueError(f"matrix must be 2x3 or 3x3, got {m.shape}")


def clip_polygon(poly: Sequence[Sequence[float]], width: int, height: int) -> Polygon:
    pts = as_points(poly)
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, width - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, height - 1))
    return [(float(x), float(y)) for x, y in pts]


def union_bbox(polygons: Iterable[Sequence[Sequence[float]]]) -> tuple[float, float, float, float]:
    boxes = [polygon_bbox(p) for p in polygons]
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def bbox_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = map(float, a)
    bx0, by0, bx1, by1 = map(float, b)
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return hypot(dx, dy)


def mask_to_largest_polygon(mask: np.ndarray) -> Polygon:
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    eps = max(1.0, 0.003 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in approx]


def rasterize_polygon(poly: Sequence[Sequence[float]], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(poly) >= 3:
        pts = np.round(as_points(poly)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask
