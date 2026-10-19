from __future__ import annotations
"""Lightweight SOURCE->TARGET ROI alignment used by both review preview and commit.

Keeping the shared alignment separate from region_composite prevents the FIRST
brush stroke from cold-importing every Precise/Hybrid rendering backend.
"""
from typing import Any
import cv2
import numpy as np
from .registration_geometry import registration_homography

def aligned_source_roi(source: np.ndarray, target_shape: tuple[int, int], project: dict[str, Any],
                        roi: tuple[int, int, int, int], *, source_offset_x: int = 0,
                        source_offset_y: int = 0) -> tuple[np.ndarray, bool]:
    """Warp SOURCE directly into a TARGET-space ROI instead of a full page."""
    th, tw = map(int, target_shape)
    xa, ya, xb, yb = [int(v) for v in roi]
    rw, rh = max(1, xb - xa), max(1, yb - ya)
    H = registration_homography(project).copy()
    dx, dy = int(source_offset_x), int(source_offset_y)
    identity = bool(
        source.shape[:2] == (th, tw) and not dx and not dy
        and np.max(np.abs(np.asarray(H, np.float64) - np.eye(3))) <= 1e-7
    )
    if identity:
        return source[ya:yb, xa:xb].copy(), True
    if dx or dy:
        T = np.asarray([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]], np.float64)
        H = T @ H
    C = np.asarray([[1.0, 0.0, -float(xa)], [0.0, 1.0, -float(ya)], [0.0, 0.0, 1.0]], np.float64)
    crop = cv2.warpPerspective(
        source, C @ H, (rw, rh), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, False

