from __future__ import annotations

"""Optional clean-room visual verifier for manga remake page pairing.

This module intentionally contains no renderer logic and no dependency on external
models.  It provides a conservative second opinion for already-selected smart
pairs using OpenCV AKAZE descriptors plus RANSAC geometry.  A failed or
inconclusive verification is never sufficient on its own to delete a pair; callers
may use only strong positive evidence to raise pairing confidence.

The implementation is original Folirina code.  It is inspired by the general
workflow of projects that use a semantic/cheap first-stage matcher followed by a
feature/RANSAC geometric check, without copying their GPL implementation.
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class RemakePairEvidence:
    confirmed: bool
    confidence: float
    good_matches: int
    inliers: int
    inlier_ratio: float
    spatial_coverage: float
    median_reprojection_error: float
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed": bool(self.confirmed),
            "confidence": float(self.confidence),
            "good_matches": int(self.good_matches),
            "inliers": int(self.inliers),
            "inlier_ratio": float(self.inlier_ratio),
            "spatial_coverage": float(self.spatial_coverage),
            "median_reprojection_error": float(self.median_reprojection_error),
            "diagnostics": dict(self.diagnostics),
        }


def _empty(reason: str, **extra: Any) -> RemakePairEvidence:
    return RemakePairEvidence(
        False, 0.0, 0, 0, 0.0, 0.0, float("inf"),
        {"reason": reason, **extra},
    )


def _prepare_gray(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("unsupported image shape")
    h, w = gray.shape[:2]
    scale = min(1.0, float(max_side) / max(1.0, float(max(h, w))))
    if scale < 0.999:
        gray = cv2.resize(
            gray,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    # Mild blur suppresses JPEG grain and some glyph-level differences while
    # preserving panel borders, faces, clothing edges and background structure.
    sigma = max(0.6, max(gray.shape[:2]) / 900.0)
    smooth = cv2.GaussianBlur(gray, (0, 0), sigma)
    # Local contrast normalization makes scans with different exposure useful to
    # the same binary descriptor without trying to transfer tone or colour.
    smooth = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(smooth)
    return smooth, scale


def _ratio_matches(des_a: np.ndarray, des_b: np.ndarray, ratio: float) -> list[cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    rows = matcher.knnMatch(des_a, des_b, k=2)
    good: list[cv2.DMatch] = []
    for row in rows:
        if len(row) < 2:
            continue
        first, second = row[0], row[1]
        if first.distance < float(ratio) * max(float(second.distance), 1e-6):
            good.append(first)
    return good


def _hull_coverage(points: np.ndarray, width: int, height: int) -> float:
    if points.shape[0] < 3 or width <= 0 or height <= 0:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
    area = float(abs(cv2.contourArea(hull)))
    return float(np.clip(area / max(1.0, float(width * height)), 0.0, 1.0))


def verify_remake_pair(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_side: int = 1000,
    ratio_test: float = 0.76,
    min_good_matches: int = 18,
    min_inlier_ratio: float = 0.45,
    min_spatial_coverage: float = 0.08,
    max_median_error: float = 4.5,
) -> RemakePairEvidence:
    """Return strong positive geometric evidence for an already-selected pair.

    The verifier is deliberately asymmetric in policy but not in geometry: it can
    confirm a pair, or be inconclusive.  Callers should not treat an inconclusive
    result as proof that two pages differ because translations, crops, scans and
    heavy cleanup can all reduce local feature counts.
    """
    if source is None or target is None or source.size == 0 or target.size == 0:
        return _empty("empty_image")
    try:
        src_gray, src_scale = _prepare_gray(source, int(max_side))
        tgt_gray, tgt_scale = _prepare_gray(target, int(max_side))
    except (ValueError, cv2.error) as exc:
        return _empty("preprocess_failed", error=str(exc))

    detector = cv2.AKAZE_create()
    kp_s, des_s = detector.detectAndCompute(src_gray, None)
    kp_t, des_t = detector.detectAndCompute(tgt_gray, None)
    if des_s is None or des_t is None or len(kp_s) < 8 or len(kp_t) < 8:
        return _empty(
            "insufficient_features", source_keypoints=len(kp_s or []), target_keypoints=len(kp_t or []),
            source_scale=float(src_scale), target_scale=float(tgt_scale),
        )

    try:
        good = _ratio_matches(des_s, des_t, float(ratio_test))
    except cv2.error as exc:
        return _empty("matching_failed", error=str(exc))
    if len(good) < max(4, int(min_good_matches)):
        return RemakePairEvidence(
            False, 0.0, len(good), 0, 0.0, 0.0, float("inf"),
            {
                "reason": "insufficient_good_matches",
                "source_keypoints": len(kp_s), "target_keypoints": len(kp_t),
                "min_good_matches": int(min_good_matches),
                "source_scale": float(src_scale), "target_scale": float(tgt_scale),
            },
        )

    src_pts = np.float32([kp_s[m.queryIdx].pt for m in good])
    tgt_pts = np.float32([kp_t[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src_pts, tgt_pts, cv2.RANSAC, 4.0)
    if H is None or mask is None or H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return RemakePairEvidence(
            False, 0.0, len(good), 0, 0.0, 0.0, float("inf"),
            {"reason": "homography_failed", "source_keypoints": len(kp_s), "target_keypoints": len(kp_t)},
        )

    inlier_mask = mask.reshape(-1).astype(bool)
    inliers = int(np.count_nonzero(inlier_mask))
    inlier_ratio = float(inliers / max(1, len(good)))
    if inliers >= 4:
        src_in = src_pts[inlier_mask]
        tgt_in = tgt_pts[inlier_mask]
        projected = cv2.perspectiveTransform(src_in.reshape(-1, 1, 2), H).reshape(-1, 2)
        errors = np.linalg.norm(projected - tgt_in, axis=1)
        median_error = float(np.median(errors)) if errors.size else float("inf")
        src_cov = _hull_coverage(src_in, src_gray.shape[1], src_gray.shape[0])
        tgt_cov = _hull_coverage(tgt_in, tgt_gray.shape[1], tgt_gray.shape[0])
        coverage = float(min(src_cov, tgt_cov))
    else:
        median_error = float("inf")
        coverage = 0.0

    match_score = float(np.clip(len(good) / max(float(min_good_matches) * 2.5, 1.0), 0.0, 1.0))
    inlier_score = float(np.clip((inlier_ratio - 0.20) / 0.65, 0.0, 1.0))
    coverage_score = float(np.clip(coverage / max(float(min_spatial_coverage) * 2.5, 0.12), 0.0, 1.0))
    error_score = float(np.clip(1.0 - median_error / max(float(max_median_error) * 1.8, 1e-6), 0.0, 1.0))
    confidence = float(
        np.clip(
            (max(match_score, 1e-6) ** 0.18)
            * (max(inlier_score, 1e-6) ** 0.42)
            * (max(coverage_score, 1e-6) ** 0.24)
            * (max(error_score, 1e-6) ** 0.16),
            0.0,
            1.0,
        )
    )
    confirmed = bool(
        len(good) >= int(min_good_matches)
        and inlier_ratio >= float(min_inlier_ratio)
        and coverage >= float(min_spatial_coverage)
        and median_error <= float(max_median_error)
    )
    return RemakePairEvidence(
        confirmed=confirmed,
        confidence=confidence if confirmed else min(confidence, 0.49),
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        spatial_coverage=coverage,
        median_reprojection_error=median_error,
        diagnostics={
            "reason": "confirmed" if confirmed else "geometry_not_strong_enough",
            "source_keypoints": len(kp_s),
            "target_keypoints": len(kp_t),
            "source_scale": float(src_scale),
            "target_scale": float(tgt_scale),
            "ratio_test": float(ratio_test),
            "min_good_matches": int(min_good_matches),
            "min_inlier_ratio": float(min_inlier_ratio),
            "min_spatial_coverage": float(min_spatial_coverage),
            "max_median_error": float(max_median_error),
        },
    )


__all__ = ["RemakePairEvidence", "verify_remake_pair"]
