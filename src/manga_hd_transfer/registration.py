from __future__ import annotations

from dataclasses import dataclass
from math import exp
import math
from typing import Iterable
import threading
import os
from pathlib import Path

import cv2
import numpy as np

from .config import RegistrationConfig
from .geometry import transform_points, transform_to_homography
from .models import RegistrationResult
from .runtime import accelerator_lock, select_device
from .structure_registration import refine_registration_with_structure
from .plugins import REGISTRY as PROVIDER_REGISTRY

def _run_registration_refiner(source: np.ndarray, target: np.ndarray, base: RegistrationResult, cfg: RegistrationConfig) -> RegistrationResult:
    provider=PROVIDER_REGISTRY.get("registration_refiner","structure_ecc")
    return provider(source,target,base,cfg) if provider is not None else refine_registration_with_structure(source,target,base,cfg)


_DEEP_MODEL_CACHE: dict[tuple, object] = {}
_DEEP_MODEL_LOCK = threading.RLock()


def _torch_checkpoint_dir() -> Path:
    root = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
    return root / "hub" / "checkpoints"


def _deep_weights_ready(kind: str, cfg: RegistrationConfig) -> bool:
    if cfg.allow_model_downloads:
        return True
    root = _torch_checkpoint_dir()
    if not root.exists():
        return False
    try:
        names = {p.name for p in root.iterdir() if p.is_file()}
    except OSError:
        return False
    if kind == "lightglue":
        feature = cfg.feature.lower() if cfg.feature.lower() in {"sift","disk","aliked"} else "sift"
        # Learned DISK/ALIKED extractors may have their own weights; without
        # explicit download permission, auto escalation only trusts SIFT.
        if feature != "sift":
            return False
        return any(name.startswith("sift_lightglue_") and name.endswith(".pth") for name in names)
    if kind == "loftr":
        return any("loftr_outdoor" in name and name.endswith((".ckpt", ".pth")) for name in names)
    return False


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


def _fast_identity_registration(source: np.ndarray, target: np.ndarray, cfg: RegistrationConfig) -> RegistrationResult | None:
    """Cheap same/near-source preflight before feature extraction.

    Heavy Gaussian blur suppresses translated glyph differences while retaining
    panels, screentones and character structure.  Phase correlation estimates a
    tiny translation; normalized correlation then decides whether the page is
    safe for the fast path.  Different source dimensions are allowed only as an
    anisotropic resize prior, and the result is still gated by strict correlation.
    """
    if not cfg.fast_identity:
        return None
    sh, sw = source.shape[:2]; th, tw = target.shape[:2]
    if sw < 32 or sh < 32 or tw < 32 or th < 32:
        return None
    aspect_s = sw / max(1, sh); aspect_t = tw / max(1, th)
    if abs(aspect_s - aspect_t) / max(aspect_t, 1e-6) > 0.01:
        return None

    max_side = max(128, int(cfg.fast_identity_max_side))
    scale = min(1.0, max_side / max(tw, th))
    ww, hh = max(16, round(tw * scale)), max(16, round(th * scale))
    sg = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source
    tg = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    # Normalize source to target thumbnail dimensions; the final matrix preserves
    # the real source->target scale, so this does not blur final rendering.
    sg = cv2.resize(sg, (ww, hh), interpolation=cv2.INTER_AREA if sw >= ww and sh >= hh else cv2.INTER_LINEAR)
    tg = cv2.resize(tg, (ww, hh), interpolation=cv2.INTER_AREA if tw >= ww and th >= hh else cv2.INTER_LINEAR)
    sigma = max(1.0, float(cfg.fast_identity_blur_sigma))
    sg = cv2.GaussianBlur(sg, (0, 0), sigma).astype(np.float32)
    tg = cv2.GaussianBlur(tg, (0, 0), sigma).astype(np.float32)
    sg = (sg - float(sg.mean())) / max(float(sg.std()), 1e-6)
    tg = (tg - float(tg.mean())) / max(float(tg.std()), 1e-6)
    try:
        (dx_t, dy_t), response = cv2.phaseCorrelate(sg, tg)
    except cv2.error:
        return None
    if not np.isfinite(dx_t + dy_t + response):
        return None
    M = np.array([[1.0, 0.0, dx_t], [0.0, 1.0, dy_t]], np.float32)
    moved = cv2.warpAffine(sg, M, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(np.ones((hh, ww), np.uint8), M, (ww, hh), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    if np.count_nonzero(valid) < ww * hh * 0.75:
        return None
    a, b = moved[valid].astype(np.float64), tg[valid].astype(np.float64)
    denom = max(float(a.std() * b.std()), 1e-8)
    corr = float(np.mean((a - a.mean()) * (b - b.mean())) / denom)
    dx = dx_t / max(scale, 1e-9); dy = dy_t / max(scale, 1e-9)
    shift_mag = max(abs(dx), abs(dy))
    normal_ok = (
        response >= cfg.fast_identity_min_phase_response
        and corr >= cfg.fast_identity_min_correlation
        and shift_mag <= float(cfg.fast_identity_max_shift_px)
    )
    large_shift_ok = (
        response >= cfg.fast_identity_large_shift_min_phase_response
        and corr >= cfg.fast_identity_large_shift_min_correlation
        and shift_mag <= float(cfg.fast_identity_large_shift_px)
    )
    if not (normal_ok or large_shift_ok):
        return None
    sx, sy = tw / max(sw, 1), th / max(sh, 1)
    H = np.array([[sx, 0.0, dx], [0.0, sy, dy], [0.0, 0.0, 1.0]], np.float64)
    confidence = float(np.clip(0.55 * corr + 0.45 * response, 0.0, 0.9999))
    scale_like_identity = abs(sx - 1.0) < 0.0025 and abs(sy - 1.0) < 0.0025
    method = "fast-phase-identity" if scale_like_identity else "fast-resize-phase"
    return RegistrationResult(
        matrix=H, method=method, confidence=confidence,
        inlier_ratio=1.0, reprojection_error=float(math.hypot(dx, dy)),
        spatial_coverage=1.0, num_matches=0, source_size=(sw, sh), target_size=(tw, th),
        diagnostics={
            "route": "fast_identity", "phase_response": float(response),
            "blurred_correlation": corr, "thumb_shift": [float(dx_t), float(dy_t)],
            "full_shift": [float(dx), float(dy)], "source_scale": [sx, sy],
        },
    )


def _opencv_matches(source: np.ndarray, target: np.ndarray, cfg: RegistrationConfig) -> FeatureMatches:
    sgray, ss = _resize_for_features(_gray_for_features(source), cfg.deep_max_side)
    tgray, ts = _resize_for_features(_gray_for_features(target), cfg.deep_max_side)

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

    device_name = select_device(cfg.device)
    device = torch.device(device_name)
    feature = cfg.feature.lower()
    if feature not in {"aliked", "disk", "sift"}:
        feature = "sift"
    key = ("lightglue", feature, int(cfg.max_features), device_name)
    with _DEEP_MODEL_LOCK:
        pair = _DEEP_MODEL_CACHE.get(key)
        if pair is None:
            if feature == "aliked":
                extractor = ALIKED(max_num_keypoints=cfg.max_features).eval().to(device)
            elif feature == "disk":
                extractor = DISK(max_num_keypoints=cfg.max_features).eval().to(device)
            else:
                extractor = SIFT(max_num_keypoints=cfg.max_features).eval().to(device)
            matcher = LightGlue(features=feature).eval().to(device)
            pair = (extractor, matcher)
            _DEEP_MODEL_CACHE[key] = pair
    extractor, matcher = pair

    def to_tensor(img: np.ndarray):
        gray = _gray_for_features(img)
        small, scale = _resize_for_features(gray, cfg.deep_max_side)
        rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
        ten = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        return ten.to(device), scale

    i0, s0 = to_tensor(source)
    i1, s1 = to_tensor(target)
    with accelerator_lock(), torch.inference_mode():
        f0, f1 = extractor.extract(i0), extractor.extract(i1)
        m01 = matcher({"image0": f0, "image1": f1})
        f0, f1, m01 = [rbd(x) for x in (f0, f1, m01)]
        matches = m01["matches"].detach().cpu().numpy()
        k0 = f0["keypoints"].detach().cpu().numpy()
        k1 = f1["keypoints"].detach().cpu().numpy()
    if len(matches) == 0:
        sp = tp = np.empty((0, 2), np.float32)
    else:
        sp = k0[matches[:, 0]].astype(np.float32) / max(s0, 1e-9)
        tp = k1[matches[:, 1]].astype(np.float32) / max(s1, 1e-9)
    return FeatureMatches(sp, tp, f"lightglue-{feature}", {
        "device": device_name, "matches": len(matches), "model_cache": True,
        "source_scale": s0, "target_scale": s1,
    })


def _loftr_matches(source: np.ndarray, target: np.ndarray, cfg: RegistrationConfig) -> FeatureMatches:
    try:
        import torch
        import kornia.feature as KF
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError("LoFTR backend requires torch and kornia") from e

    device_name = select_device(cfg.device)
    device = torch.device(device_name)
    key = ("loftr", "outdoor", device_name)
    with _DEEP_MODEL_LOCK:
        matcher = _DEEP_MODEL_CACHE.get(key)
        if matcher is None:
            matcher = KF.LoFTR(pretrained="outdoor").eval().to(device)
            _DEEP_MODEL_CACHE[key] = matcher

    def prep(img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        small, scale = _resize_for_features(gray, min(cfg.deep_max_side, 1280))
        ten = torch.from_numpy(small).float()[None, None] / 255.0
        return ten.to(device), scale

    a, sa = prep(source)
    b, sb = prep(target)
    with accelerator_lock(), torch.inference_mode():
        pred = matcher({"image0": a, "image1": b})
    sp = pred["keypoints0"].detach().cpu().numpy().astype(np.float32) / max(sa, 1e-9)
    tp = pred["keypoints1"].detach().cpu().numpy().astype(np.float32) / max(sb, 1e-9)
    conf = pred.get("confidence")
    if conf is not None and len(sp):
        conf_np = conf.detach().cpu().numpy()
        keep = conf_np >= 0.25
        sp, tp = sp[keep], tp[keep]
    return FeatureMatches(sp, tp, "loftr", {
        "device": device_name, "matches": len(sp), "model_cache": True,
        "source_scale": sa, "target_scale": sb,
    })


def _lightglue_matches_resilient(source: np.ndarray, target: np.ndarray, cfg: RegistrationConfig) -> FeatureMatches:
    try:
        return _lightglue_matches(source, target, cfg)
    except Exception as exc:
        if select_device(cfg.device) != "mps":
            raise
        cpu_cfg = cfg.model_copy(deep=True)
        cpu_cfg.device = "cpu"
        result = _lightglue_matches(source, target, cpu_cfg)
        result.diagnostics["device_fallback"] = f"mps->cpu:{type(exc).__name__}"
        return result


def _loftr_matches_resilient(source: np.ndarray, target: np.ndarray, cfg: RegistrationConfig) -> FeatureMatches:
    try:
        return _loftr_matches(source, target, cfg)
    except Exception as exc:
        if select_device(cfg.device) != "mps":
            raise
        cpu_cfg = cfg.model_copy(deep=True)
        cpu_cfg.device = "cpu"
        result = _loftr_matches(source, target, cpu_cfg)
        result.diagnostics["device_fallback"] = f"mps->cpu:{type(exc).__name__}"
        return result

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
    """Register pages with a cost-aware escalation policy.

    Auto route: cheap paired-page preflight -> OpenCV SIFT/ORB -> LightGlue ->
    LoFTR. Deep models are only touched when cheaper stages are uncertain.
    """
    cfg = config or RegistrationConfig()
    backend = cfg.backend.lower()
    errors: list[str] = []

    if backend == "auto":
        fast = _fast_identity_registration(source, target, cfg)
        if fast is not None:
            return fast

        candidates: list[RegistrationResult] = []
        # v0.8.30: most same-layout manga pairs do not need 1800px/6000-feature
        # extraction. Try a bounded OpenCV pass first; accept it only when both
        # geometric quality and spatial coverage are strong. Full OpenCV/deep
        # escalation remains unchanged for uncertain or genuinely different pages.
        quick_used = bool(getattr(cfg, "quick_opencv", True))
        if quick_used:
            quick_cfg = cfg.model_copy(deep=True)
            quick_cfg.deep_max_side = min(int(cfg.deep_max_side), int(getattr(cfg, "quick_opencv_max_side", 1000)))
            quick_cfg.max_features = min(int(cfg.max_features), int(getattr(cfg, "quick_opencv_max_features", 2800)))
            quick = _estimate_transform(_opencv_matches(source, target, quick_cfg), source, target, quick_cfg)
            quick.diagnostics["route"] = "opencv_quick"
            quick.diagnostics["quick_max_side"] = int(quick_cfg.deep_max_side)
            quick.diagnostics["quick_max_features"] = int(quick_cfg.max_features)
            candidates.append(quick)
            quick_ok = (
                quick.confidence >= max(float(cfg.review_confidence), float(getattr(cfg, "quick_opencv_accept_confidence", 0.72)))
                and quick.reprojection_error <= float(getattr(cfg, "quick_opencv_max_median_error", 3.5))
                and quick.spatial_coverage >= float(getattr(cfg, "quick_opencv_min_spatial_coverage", 0.18))
                and quick.num_matches >= max(4, int(cfg.min_matches))
            )
            if quick_ok:
                return _run_registration_refiner(source, target, quick, cfg)
            errors.append(f"opencv_quick_low_confidence:{quick.confidence:.3f}")

        need_full = (
            not quick_used
            or int(getattr(cfg, "quick_opencv_max_side", 1000)) < int(cfg.deep_max_side)
            or int(getattr(cfg, "quick_opencv_max_features", 2800)) < int(cfg.max_features)
        )
        if need_full:
            ocv = _estimate_transform(_opencv_matches(source, target, cfg), source, target, cfg)
            ocv.diagnostics["route"] = "opencv"
            candidates.append(ocv)
        else:
            ocv = candidates[-1]
        if ocv.confidence >= cfg.review_confidence:
            return _run_registration_refiner(source, target, ocv, cfg)
        errors.append(f"opencv_low_confidence:{ocv.confidence:.3f}")

        import importlib.util
        if importlib.util.find_spec("lightglue") is not None and _deep_weights_ready("lightglue", cfg):
            try:
                lg = _estimate_transform(_lightglue_matches_resilient(source, target, cfg), source, target, cfg)
                lg.diagnostics["route"] = "lightglue_escalation"
                candidates.append(lg)
                if lg.confidence >= cfg.review_confidence:
                    lg.diagnostics["backend_errors"] = list(errors)
                    return _run_registration_refiner(source, target, lg, cfg)
                errors.append(f"lightglue_low_confidence:{lg.confidence:.3f}")
            except Exception as e:  # pragma: no cover - optional dependency
                errors.append(f"lightglue:{type(e).__name__}:{e}")

        if importlib.util.find_spec("kornia") is not None and _deep_weights_ready("loftr", cfg):
            try:
                lf = _estimate_transform(_loftr_matches_resilient(source, target, cfg), source, target, cfg)
                lf.diagnostics["route"] = "loftr_escalation"
                candidates.append(lf)
                if lf.confidence >= cfg.review_confidence:
                    lf.diagnostics["backend_errors"] = list(errors)
                    return _run_registration_refiner(source, target, lf, cfg)
                errors.append(f"loftr_low_confidence:{lf.confidence:.3f}")
            except Exception as e:  # pragma: no cover - optional dependency
                errors.append(f"loftr:{type(e).__name__}:{e}")

        best = max(candidates, key=lambda r: r.confidence)
        best.diagnostics["backend_errors"] = errors
        best.diagnostics["route"] = best.diagnostics.get("route", "best_low_confidence")
        return best

    if backend == "opencv":
        base = _estimate_transform(_opencv_matches(source, target, cfg), source, target, cfg)
        return _run_registration_refiner(source, target, base, cfg)
    if backend == "lightglue":
        base = _estimate_transform(_lightglue_matches_resilient(source, target, cfg), source, target, cfg)
        return _run_registration_refiner(source, target, base, cfg)
    if backend == "loftr":
        base = _estimate_transform(_loftr_matches_resilient(source, target, cfg), source, target, cfg)
        return _run_registration_refiner(source, target, base, cfg)
    raise ValueError(f"Unknown registration backend: {cfg.backend}")

def warp_source_to_target(source: np.ndarray, registration: RegistrationResult) -> np.ndarray:
    tw, th = registration.target_size
    return cv2.warpPerspective(source, registration.matrix, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
