from __future__ import annotations

"""Structure-only registration refinement for cross-rendition manga pages.

This module deliberately ignores colour and relies on stable page structure after
an initial feature registration.  It is meant for pairs such as a monochrome
Chinese scan and a colour Japanese master where text and palette differ but panel
rules, character contours and scenery remain shared.

The refinement is residual only: SIFT/ORB/RANSAC still establishes the page
correspondence.  A small ECC Euclidean correction can then reduce sub-pixel
translation / tiny rotation drift.  The result is gated against the original
feature inliers so it cannot silently trade geometric correctness for a prettier
pixel correlation.
"""

from dataclasses import replace
import math

import cv2
import numpy as np

from .models import RegistrationResult


def build_structure_map(image: np.ndarray) -> np.ndarray:
    """Return Structure Map v2: colour-insensitive, multi-channel page geometry.

    Palette/text differences between a translated monochrome scan and a coloured
    Japanese master can dominate a single edge map.  v2 fuses robust Sobel
    magnitude, binary Canny support, panel/long-line evidence and a low-frequency
    luminance-detail channel.  No OCR is involved.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.7)

    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    scale = float(np.percentile(mag, 96.0)) if mag.size else 1.0
    sobel = np.clip(mag / max(scale, 1e-6), 0.0, 1.0)

    med = float(np.median(smooth))
    lo = int(max(20, 0.55 * med)); hi = int(min(235, max(lo + 20, 1.25 * med)))
    canny = (cv2.Canny(smooth, lo, hi).astype(np.float32) / 255.0)
    canny = cv2.GaussianBlur(canny, (3, 3), 0.55)

    # Long horizontal/vertical strokes strongly represent panel borders and
    # architecture, while being comparatively insensitive to translated text.
    edge8 = (canny > 0.18).astype(np.uint8) * 255
    hlen = max(9, int(round(gray.shape[1] * 0.018)))
    vlen = max(9, int(round(gray.shape[0] * 0.018)))
    hline = cv2.morphologyEx(edge8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1)))
    vline = cv2.morphologyEx(edge8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen)))
    lines = cv2.bitwise_or(hline, vline).astype(np.float32) / 255.0
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))

    low = cv2.GaussianBlur(smooth, (0, 0), 5.0)
    detail = cv2.absdiff(smooth, low).astype(np.float32)
    dscale = float(np.percentile(detail, 95.0)) if detail.size else 1.0
    detail = np.clip(detail / max(dscale, 1e-6), 0.0, 1.0)

    structure = 0.52 * sobel + 0.20 * canny + 0.18 * lines + 0.10 * detail
    structure = np.clip(structure, 0.0, 1.0).astype(np.float32)
    return cv2.GaussianBlur(structure, (3, 3), 0.42)


def _masked_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    use = mask > 0
    if int(np.count_nonzero(use)) < 256:
        return 0.0
    av = a[use].astype(np.float64)
    bv = b[use].astype(np.float64)
    sa = float(av.std()); sb = float(bv.std())
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.mean((av - av.mean()) * (bv - bv.mean())) / (sa * sb))


def _sample_reprojection_error(result: RegistrationResult, H: np.ndarray) -> float | None:
    sp = result.diagnostics.get("sample_source_points")
    tp = result.diagnostics.get("sample_target_points")
    if not sp or not tp:
        return None
    try:
        src = np.asarray(sp, np.float32).reshape(-1, 1, 2)
        dst = np.asarray(tp, np.float32).reshape(-1, 2)
        pred = cv2.perspectiveTransform(src, np.asarray(H, np.float64)).reshape(-1, 2)
        if len(pred) != len(dst) or len(pred) == 0:
            return None
        return float(np.median(np.linalg.norm(pred - dst, axis=1)))
    except Exception:
        return None


def refine_registration_with_structure(
    source: np.ndarray,
    target: np.ndarray,
    result: RegistrationResult,
    cfg,
) -> RegistrationResult:
    """Apply a tiny, strictly-gated structure-only ECC residual correction."""
    if not bool(getattr(cfg, "structure_refine_enabled", True)):
        return result
    if float(result.confidence) < float(getattr(cfg, "structure_refine_min_confidence", 0.72)):
        return result
    if "identity" in str(result.method).lower() and source.shape[:2] == target.shape[:2]:
        return result

    H = np.asarray(result.matrix, np.float64)
    try:
        # ECC is a residual refinement, not the primary registration.  Running it
        # at full manga resolution adds little geometric information but costs a
        # lot of CPU/memory.  Refine on independent source/target thumbnails and
        # convert the residual back to full target coordinates.
        max_side = max(320, int(getattr(cfg, "structure_refine_max_side", 900)))
        sh0, sw0 = source.shape[:2]
        th0, tw0 = target.shape[:2]
        ss = min(1.0, float(max_side) / float(max(sh0, sw0)))
        ts = min(1.0, float(max_side) / float(max(th0, tw0)))
        if ss < 0.999:
            source_small = cv2.resize(source, (max(1, int(round(sw0 * ss))), max(1, int(round(sh0 * ss)))), interpolation=cv2.INTER_AREA)
        else:
            source_small = source
        if ts < 0.999:
            target_small = cv2.resize(target, (max(1, int(round(tw0 * ts))), max(1, int(round(th0 * ts)))), interpolation=cv2.INTER_AREA)
        else:
            target_small = target
        source_map = build_structure_map(source_small)
        target_map = build_structure_map(target_small)
        th, tw = target_map.shape[:2]
        Ss = np.array([[ss, 0.0, 0.0], [0.0, ss, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        St = np.array([[ts, 0.0, 0.0], [0.0, ts, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        H_small = St @ H @ np.linalg.inv(Ss)
        warped = cv2.warpPerspective(
            source_map, H_small, (tw, th), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        valid = cv2.warpPerspective(
            np.full(source_map.shape, 255, np.uint8), H_small, (tw, th),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        # Exclude a small edge band where warp padding can bias ECC.
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
        if int(cv2.countNonZero(valid)) < int(0.30 * tw * th):
            return result

        before = _masked_corr(warped, target_map, valid)
        W = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(getattr(cfg, "structure_refine_ecc_iterations", 60)),
            float(getattr(cfg, "structure_refine_ecc_epsilon", 1e-5)),
        )
        cc, W = cv2.findTransformECC(
            target_map, warped, W, cv2.MOTION_EUCLIDEAN, criteria,
            inputMask=valid,
            gaussFiltSize=int(getattr(cfg, "structure_refine_gauss_size", 5)),
        )
        if not np.isfinite(float(cc)) or float(cc) < float(getattr(cfg, "structure_refine_min_ecc", 0.70)):
            return result
        dx, dy = float(W[0, 2]), float(W[1, 2])
        angle = math.degrees(math.atan2(float(W[1, 0]), float(W[0, 0])))
        if max(abs(dx), abs(dy)) > float(getattr(cfg, "structure_refine_max_shift_px", 3.0)):
            return result
        if abs(angle) > float(getattr(cfg, "structure_refine_max_rotation_deg", 0.35)):
            return result

        D_small = np.array([
            [float(W[0, 0]), float(W[0, 1]), dx],
            [float(W[1, 0]), float(W[1, 1]), dy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        # OpenCV ECC returns the inverse-map-style residual. Convert that
        # residual from target-thumbnail coordinates back to full target pixels,
        # then invert it before composing into our source->target homography.
        D_full = np.linalg.inv(St) @ D_small @ St
        refined_H = np.linalg.inv(D_full) @ H
        refined_H_small = St @ refined_H @ np.linalg.inv(Ss)
        refined_warp = cv2.warpPerspective(
            source_map, refined_H_small, (tw, th), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        after = _masked_corr(refined_warp, target_map, valid)
        min_gain = float(getattr(cfg, "structure_refine_min_correlation_gain", 0.003))
        if after + 1e-9 < before + min_gain:
            return result

        old_sample = _sample_reprojection_error(result, H)
        new_sample = _sample_reprojection_error(result, refined_H)
        max_worsen = float(getattr(cfg, "structure_refine_max_feature_error_worsen_px", 0.20))
        if old_sample is not None and new_sample is not None and new_sample > old_sample + max_worsen:
            return result

        diagnostics = dict(result.diagnostics)
        diagnostics.update({
            "structure_refine_applied": True,
            "structure_ecc": float(cc),
            "structure_corr_before": float(before),
            "structure_corr_after": float(after),
            "structure_residual_dx": float(D_full[0, 2]),
            "structure_residual_dy": float(D_full[1, 2]),
            "structure_residual_rotation_deg": float(angle),
            "structure_refine_max_side": int(max_side),
            "structure_source_scale": float(ss),
            "structure_target_scale": float(ts),
            "structure_feature_median_before": old_sample,
            "structure_feature_median_after": new_sample,
        })
        return replace(
            result,
            matrix=refined_H,
            method=f"{result.method}+structure-ecc",
            diagnostics=diagnostics,
        )
    except cv2.error as exc:
        diagnostics = dict(result.diagnostics)
        diagnostics["structure_refine_error"] = f"cv2:{exc}"
        return replace(result, diagnostics=diagnostics)
    except Exception as exc:
        diagnostics = dict(result.diagnostics)
        diagnostics["structure_refine_error"] = f"{type(exc).__name__}:{exc}"
        return replace(result, diagnostics=diagnostics)


# Provider registration is intentionally at module tail so the callable is fully defined.
from .plugins import REGISTRY
REGISTRY.register("registration_refiner", "structure_ecc", refine_registration_with_structure, replace=True)
