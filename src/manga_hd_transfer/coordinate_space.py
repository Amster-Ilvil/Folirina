from __future__ import annotations

"""Canonical SOURCE-original coordinate space for all precise-transfer regions.

All detector masks remain in the translated SOURCE page's original pixels.  A
single registration mapping converts positions into TARGET space.  Even when the
page transform is affine/homographic, raster content is projected to the nearest
local similarity so Chinese glyphs are never non-uniformly squeezed or sheared.
"""

from dataclasses import dataclass
import math
import numpy as np

from .geometry import transform_to_homography
from .models import RegistrationResult


@dataclass(slots=True)
class LocalSimilarity:
    scale: float
    rotation_deg: float
    anisotropy: float
    target_x: float
    target_y: float


@dataclass(slots=True)
class SourceCoordinateSpace:
    H: np.ndarray
    source_size: tuple[int, int]
    target_size: tuple[int, int]

    @classmethod
    def from_registration(cls, registration: RegistrationResult) -> "SourceCoordinateSpace":
        return cls(
            transform_to_homography(registration.matrix).astype(np.float64),
            tuple(registration.source_size),
            tuple(registration.target_size),
        )

    def map_point(self, x: float, y: float) -> tuple[float, float]:
        q = self.H @ np.array([float(x), float(y), 1.0], dtype=np.float64)
        if abs(float(q[2])) < 1e-12:
            raise ValueError("Registration maps point to infinity")
        return float(q[0] / q[2]), float(q[1] / q[2])

    def local_similarity(self, x: float, y: float) -> LocalSimilarity:
        h = self.H
        x = float(x); y = float(y)
        den = float(h[2, 0] * x + h[2, 1] * y + h[2, 2])
        if abs(den) < 1e-12:
            tx, ty = self.map_point(x, y)
            return LocalSimilarity(1.0, 0.0, 1.0, tx, ty)
        nu = float(h[0, 0] * x + h[0, 1] * y + h[0, 2])
        nv = float(h[1, 0] * x + h[1, 1] * y + h[1, 2])
        den2 = den * den
        J = np.array([
            [(h[0, 0] * den - nu * h[2, 0]) / den2,
             (h[0, 1] * den - nu * h[2, 1]) / den2],
            [(h[1, 0] * den - nv * h[2, 0]) / den2,
             (h[1, 1] * den - nv * h[2, 1]) / den2],
        ], dtype=np.float64)
        U, s, Vt = np.linalg.svd(J)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        scale = float(math.sqrt(max(1e-12, abs(float(np.linalg.det(J))))))
        anisotropy = float(abs(float(s[0]) - float(s[1])) / max(1e-9, 0.5 * (float(s[0]) + float(s[1]))))
        angle = math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))
        tx, ty = self.map_point(x, y)
        return LocalSimilarity(scale, angle, anisotropy, tx, ty)
