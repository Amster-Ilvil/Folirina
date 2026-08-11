from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable

import cv2
import numpy as np

from .config import RegistrationConfig
from .geometry import transform_points, transform_to_homography
from .models import RegistrationResult


@dataclass(slots=True)
class FeatureMatches:
    source_points: np.ndarray
    target_points: np.ndarray
    method: str
    diagnostics: dict


def _gray_for_features(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.GaussianBlur(gray, (3, 3), 0.6)


def _resize_for_features(gray: np.ndarray, max_side: int = 1800) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 0.999:
        return gray, 1.0
    return cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA), scale


def _opencv_matches(source: np.ndarray, target: np.ndarray, cfg: RegistrationConfig) -> FeatureMatches:
    sgray, ss = _resize_for_features(_gray_for_features(source))
    tgray, ts = _resize_for_features(_gray_for_features(target))

    use_sift = cfg.feature.lower() == "sift" and hasattr(cv2, "SIFT_create")
    if use_sift:
        detector = cv2.SIFT_create(nfeatures=cfg.max_features, contrastThreshold=0.018, edgeThreshold=14)
        norm = cv2.NORM_L2
        method = "opencv-sift"
    else:
        detector = cv2.ORB_create(nfeatures=cfg.max_features, fastThreshold=7, edgeThreshold=15)
        norm = cv2.NORM_HAMMING
        method = "opencv-orb"

    skp, sdesc = detector.detectAndCompute(sgray, None)
    tkp, tdesc = detector.detectAndCompute(tgray, None)
    if sdesc is None or tdesc is None or len(skp) < 4 or len(tkp) < 4:
        return FeatureMatches(np.empty((0, 2)), np.empty((0, 2)), method, {"reason": "insufficient_features"})

    matcher = cv2.BFMatcher(norm)
    raw = matcher.knnMatch(sdesc, tdesc, k=2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        a, b = pair
        if a.distance < cfg.ratio_test * b.distance:
            good.append(a)

    # Enforce a weak one-to-one mapping by target descriptor to reduce text-glyph duplicates.
    best_by_train = {}
    for m in sorted(good, key=lambda x: x.distance):
        best_by_train.setdefault(m.trainIdx, m)
    good = list(best_by_train.values())

    sp = np.float32([skp[m.queryIdx].pt for m in good]) / ss
    tp = np.float32([tkp[m.trainIdx].pt for m in good]) / ts
    return FeatureMatches(
        sp,
        tp,
        method,
        {
            "source_features": len(skp),
            "target_features": len(tkp),
            "ratio_matches": len(good),
            "source_scale": ss,
            "target_scale": ts,
        },
    )


def _lightglue_matches(source: np.ndarray, target: np.ndarray, cfg: RegistrationConfig) -> FeatureMatches:
    try:
        import torch
        from lightglue import ALIKED, DISK, SIFT, LightGlue
        from lightglue.utils import rbd
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError("LightGlue backend is not installed") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    feature = cfg.feature.lower()
    if feature == "aliked":
        extractor = ALIKED(max_num_keypoints=cfg.max_features).eval().to(device)
    elif feature == "disk":
        extractor = DISK(max_num_keypoints=cfg.max_features).eval().to(device)
    else:
        feature = "sift"
        extractor = SIFT(max_num_keypoints=cfg.max_features).eval().to(device)
    matcher = LightGlue(features=feature).eval().to(device)

    def to_tensor(img: np.ndarray):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ten = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        return ten.to(device)

    with torch.inference_mode():
        i0, i1 = to_tensor(source), to_tensor(target)
        f0, f1 = extractor.extract(i0), extractor.extract(i1)
        m01 = matcher({"image0": f0, "image1": f1})
        f0, f1, m01 = [rbd(x) for x in (f0, f1, m01)]
        matches = m01["matches"].detach().cpu().numpy()
        k0 = f0["keypoints"].detach().cpu().numpy()
        k1 = f1["keypoints"].detach().cpu().numpy()
    if len(matches) == 0:
        sp = tp = np.empty((0, 2), np.float32)
    else:
        sp = k0[matches[:, 0]].astype(np.float32)
        tp = k1[matches[:, 1]].astype(np.float32)
    return FeatureMatches(sp, tp, f"lightglue-{feature}", {"device": str(device), "matches": len(matches)})


def _loftr_matches(source: np.ndarray, target: np.ndarray, cfg: RegistrationConfig) -> FeatureMatches:
    try:
        import torch
        import kornia as K
        import kornia.feature as KF
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError("LoFTR backend requires torch and kornia") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    matcher = KF.LoFTR(pretrained="outdoor").eval().to(device)

    def prep(img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = min(1.0, 1280 / max(h, w))
        if scale < 1:
            gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        ten = torch.from_numpy(gray).float()[None, None] / 255.0
        return ten.to(device), scale

    a, sa = prep(source)
    b, sb = prep(target)
    with torch.inference_mode():
        pred = matcher({"image0": a, "image1": b})
    sp = pred["keypoints0"].detach().cpu().numpy().astype(np.float32) / sa
    tp = pred["keypoints1"].detach().cpu().numpy().astype(np.float32) / sb
    conf = pred.get("confidence")
    if conf is not None and len(conf):
        c = conf.detach().cpu().numpy()
        keep = c >= max(0.2, float(np.quantile(c, 0.25)))
        sp, tp = sp[keep], tp[keep]
    return FeatureMatches(sp, tp, "loftr", {"device": str(device), "matches": len(sp)})


def _coverage(points: np.ndarray, size: tuple[int, int]) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    area = abs(float(cv2.contourArea(hull)))
    w, h = size
    return float(np.clip(area / max(1.0, w * h), 0.0, 1.0))


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.copy()
    pts = points.reshape(1, -1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, transform_to_homography(matrix))[0]


def _evaluate_candidate(
    name: str,
    matrix: np.ndarray | None,
    mask: np.ndarray | None,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_size: tuple[int, int],
    cfg: RegistrationConfig,
) -> dict | None:
    if matrix is None or mask is None:
        return None
    hmat = transform_to_homography(matrix)
    if not np.all(np.isfinite(hmat)):
        return None
    # Reject reflections unless explicitly requested.
    det = np.linalg.det(hmat[:2, :2])
    if det < 0 and not cfg.allow_reflection:
        return None
    inliers = mask.reshape(-1).astype(bool)
    if np.count_nonzero(inliers) < 4:
        return None
    projected = _project(source_points[inliers], hmat)
    errors = np.linalg.norm(projected - target_points[inliers], axis=1)
    med = float(np.median(errors))
    ratio = float(np.mean(inliers))
    cov = _coverage(source_points[inliers], source_size)
    complexity_penalty = {"similarity": 0.00, "affine": 0.025, "homography": 0.055}.get(name, 0.03)
    quality = (
        0.50 * ratio
        + 0.28 * exp(-med / 5.0)
        + 0.22 * min(1.0, cov / 0.35)
        - complexity_penalty
    )
    if ratio < cfg.min_inlier_ratio:
        quality *= max(0.25, ratio / max(cfg.min_inlier_ratio, 1e-6))
    if med > cfg.max_median_error:
        quality *= max(0.25, cfg.max_median_error / med)
    if cov < cfg.min_spatial_coverage:
        quality *= max(0.35, cov / max(cfg.min_spatial_coverage, 1e-6))
    return {
        "name": name,
        "matrix": hmat,
        "mask": inliers,
        "inlier_ratio": ratio,
        "median_error": med,
        "coverage": cov,
        "quality": float(np.clip(quality, 0.0, 1.0)),
    }


def _estimate_transform(matches: FeatureMatches, source: np.ndarray, target: np.ndarray, cfg: RegistrationConfig) -> RegistrationResult:
    sp, tp = matches.source_points, matches.target_points
    sw, sh = source.shape[1], source.shape[0]
    tw, th = target.shape[1], target.shape[0]
    if len(sp) < max(4, cfg.min_matches):
        return _fallback_resize_translation(source, target, matches.method, diagnostics={**matches.diagnostics, "reason": "insufficient_matches"})

    candidates: list[dict] = []
    if "similarity" in cfg.model_preference:
        mat, mask = cv2.estimateAffinePartial2D(
            sp, tp, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_threshold, maxIters=5000, confidence=0.999, refineIters=15
        )
        c = _evaluate_candidate("similarity", mat, mask, sp, tp, (sw, sh), cfg)
        if c:
            candidates.append(c)
    if "affine" in cfg.model_preference:
        mat, mask = cv2.estimateAffine2D(
            sp, tp, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_threshold, maxIters=5000, confidence=0.999, refineIters=15
        )
        c = _evaluate_candidate("affine", mat, mask, sp, tp, (sw, sh), cfg)
        if c:
            candidates.append(c)
    if "homography" in cfg.model_preference and len(sp) >= 6:
        mat, mask = cv2.findHomography(sp, tp, cv2.RANSAC, cfg.ransac_threshold, maxIters=6000, confidence=0.999)
        c = _evaluate_candidate("homography", mat, mask, sp, tp, (sw, sh), cfg)
        if c:
            candidates.append(c)

    if not candidates:
        return _fallback_resize_translation(source, target, matches.method, diagnostics={**matches.diagnostics, "reason": "model_estimation_failed"})

    best = max(candidates, key=lambda c: c["quality"])
    diagnostics = dict(matches.diagnostics)
    diagnostics["candidates"] = [
        {k: v for k, v in c.items() if k not in {"matrix", "mask"}} for c in candidates
    ]
    inliers = best["mask"]
    diagnostics["sample_source_points"] = sp[inliers][:100].round(2).tolist()
    diagnostics["sample_target_points"] = tp[inliers][:100].round(2).tolist()
    return RegistrationResult(
        matrix=best["matrix"],
        method=f"{matches.method}+{best['name']}",
        confidence=best["quality"],
        inlier_ratio=best["inlier_ratio"],
        reprojection_error=best["median_error"],
        spatial_coverage=best["coverage"],
        num_matches=int(np.count_nonzero(inliers)),
        source_size=(sw, sh),
        target_size=(tw, th),
        diagnostics=diagnostics,
    )


def _fallback_resize_translation(
    source: np.ndarray, target: np.ndarray, feature_method: str, diagnostics: dict | None = None
) -> RegistrationResult:
    """Low-confidence fallback: anisotropic scale plus phase-correlation translation.

    It is deliberately marked low-confidence so it cannot silently pass publication QA.
    """
    sh, sw = source.shape[:2]
    th, tw = target.shape[:2]
    sx, sy = tw / max(sw, 1), th / max(sh, 1)
    scaled = cv2.resize(_gray_for_features(source), (tw, th), interpolation=cv2.INTER_AREA if sx < 1 or sy < 1 else cv2.INTER_LINEAR)
    tgt = _gray_for_features(target)
    try:
        shift, response = cv2.phaseCorrelate(np.float32(scaled), np.float32(tgt))
        dx, dy = shift
        if not np.isfinite(dx + dy):
            dx = dy = 0.0
            response = 0.0
    except cv2.error:
        dx = dy = 0.0
        response = 0.0
    matrix = np.array([[sx, 0.0, dx], [0.0, sy, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
    conf = float(np.clip(0.18 + 0.22 * max(0.0, response), 0.05, 0.40))
    return RegistrationResult(
        matrix=matrix,
        method=f"{feature_method}+resize-phase-fallback",
        confidence=conf,
        inlier_ratio=0.0,
        reprojection_error=999.0,
        spatial_coverage=0.0,
        num_matches=0,
        source_size=(sw, sh),
        target_size=(tw, th),
        diagnostics={**(diagnostics or {}), "phase_response": float(response), "fallback": True},
    )


def register_images(source: np.ndarray, target: np.ndarray, config: RegistrationConfig | None = None) -> RegistrationResult:
    cfg = config or RegistrationConfig()
    backend = cfg.backend.lower()
    errors: list[str] = []

    if backend in {"lightglue", "auto"}:
        try:
            if backend == "lightglue":
                return _estimate_transform(_lightglue_matches(source, target, cfg), source, target, cfg)
            # In auto mode, only use LightGlue when it is already importable; failures do not block OpenCV.
            import importlib.util
            if importlib.util.find_spec("lightglue") is not None:
                lg = _estimate_transform(_lightglue_matches(source, target, cfg), source, target, cfg)
                if lg.confidence >= cfg.review_confidence:
                    return lg
        except Exception as e:  # pragma: no cover - optional dependency
            errors.append(f"lightglue:{type(e).__name__}:{e}")

    if backend in {"opencv", "auto"}:
        ocv = _estimate_transform(_opencv_matches(source, target, cfg), source, target, cfg)
        if errors:
            ocv.diagnostics["backend_errors"] = errors
        if backend == "opencv" or ocv.confidence >= cfg.review_confidence:
            return ocv
        errors.append(f"opencv_low_confidence:{ocv.confidence:.3f}")

    if backend in {"loftr", "auto"}:
        try:
            if backend == "auto":
                import importlib.util
                if importlib.util.find_spec("kornia") is None:
                    if 'ocv' in locals():
                        ocv.diagnostics["backend_errors"] = errors + ["loftr:not_installed"]
                        return ocv
                    return _fallback_resize_translation(source, target, "auto", {"backend_errors": errors})
            result = _estimate_transform(_loftr_matches(source, target, cfg), source, target, cfg)
            if errors:
                result.diagnostics["backend_errors"] = errors
            return result
        except Exception as e:  # pragma: no cover - optional dependency
            errors.append(f"loftr:{type(e).__name__}:{e}")
            if 'ocv' in locals():
                ocv.diagnostics["backend_errors"] = errors
                return ocv
            return _fallback_resize_translation(source, target, backend, {"backend_errors": errors})

    raise ValueError(f"Unknown registration backend: {cfg.backend}")


def warp_source_to_target(source: np.ndarray, registration: RegistrationResult) -> np.ndarray:
    tw, th = registration.target_size
    return cv2.warpPerspective(source, registration.matrix, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
