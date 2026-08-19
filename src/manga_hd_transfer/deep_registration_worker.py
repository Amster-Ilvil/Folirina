from __future__ import annotations

"""JSONL worker for isolated LightGlue / LoFTR page matching."""

import json
import os
import sys
import traceback
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch

_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}


def _gray_for_features(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.GaussianBlur(gray, (3, 3), 0.6)


def _resize_for_features(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale >= 0.999:
        return gray, 1.0
    return cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA), scale


def _select_device(preferred: str) -> str:
    preferred = str(preferred or "auto").lower().strip()
    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if preferred == "mps":
        return "mps" if mps else "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if mps:
        return "mps"
    return "cpu"


def _lightglue(source: np.ndarray, target: np.ndarray, req: dict[str, Any], device_name: str):
    from lightglue import ALIKED, DISK, SIFT, LightGlue
    from lightglue.utils import rbd

    feature = str(req.get("feature", "sift")).lower()
    if feature not in {"aliked", "disk", "sift"}:
        feature = "sift"
    max_features = int(req.get("max_features", 4096))
    max_side = int(req.get("deep_max_side", 1800))
    device = torch.device(device_name)
    key = ("lightglue", feature, max_features, device_name)
    pair = _MODEL_CACHE.get(key)
    if pair is None:
        if feature == "aliked":
            extractor = ALIKED(max_num_keypoints=max_features).eval().to(device)
        elif feature == "disk":
            extractor = DISK(max_num_keypoints=max_features).eval().to(device)
        else:
            extractor = SIFT(max_num_keypoints=max_features).eval().to(device)
        matcher = LightGlue(features=feature).eval().to(device)
        pair = (extractor, matcher); _MODEL_CACHE[key] = pair
    extractor, matcher = pair

    def to_tensor(img: np.ndarray):
        gray = _gray_for_features(img)
        small, scale = _resize_for_features(gray, max_side)
        rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
        ten = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        return ten.to(device), scale

    i0, s0 = to_tensor(source); i1, s1 = to_tensor(target)
    with torch.inference_mode():
        f0, f1 = extractor.extract(i0), extractor.extract(i1)
        m01 = matcher({"image0": f0, "image1": f1})
        f0, f1, m01 = [rbd(x) for x in (f0, f1, m01)]
        matches = m01["matches"].detach().cpu().numpy()
        k0 = f0["keypoints"].detach().cpu().numpy(); k1 = f1["keypoints"].detach().cpu().numpy()
    if len(matches) == 0:
        sp = tp = np.empty((0, 2), np.float32)
    else:
        sp = k0[matches[:, 0]].astype(np.float32) / max(s0, 1e-9)
        tp = k1[matches[:, 1]].astype(np.float32) / max(s1, 1e-9)
    return sp, tp, f"lightglue-{feature}", {"device": device_name, "matches": int(len(matches)), "model_cache": True, "source_scale": float(s0), "target_scale": float(s1), "isolated_runtime": True}


def _loftr(source: np.ndarray, target: np.ndarray, req: dict[str, Any], device_name: str):
    import kornia.feature as KF
    device = torch.device(device_name)
    key = ("loftr", "outdoor", device_name)
    matcher = _MODEL_CACHE.get(key)
    if matcher is None:
        matcher = KF.LoFTR(pretrained="outdoor").eval().to(device)
        _MODEL_CACHE[key] = matcher
    max_side = min(int(req.get("deep_max_side", 1800)), 1280)

    def prep(img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        small, scale = _resize_for_features(gray, max_side)
        ten = torch.from_numpy(small).float()[None, None] / 255.0
        return ten.to(device), scale

    a, sa = prep(source); b, sb = prep(target)
    with torch.inference_mode():
        pred = matcher({"image0": a, "image1": b})
    sp = pred["keypoints0"].detach().cpu().numpy().astype(np.float32) / max(sa, 1e-9)
    tp = pred["keypoints1"].detach().cpu().numpy().astype(np.float32) / max(sb, 1e-9)
    conf = pred.get("confidence")
    if conf is not None and len(sp):
        conf_np = conf.detach().cpu().numpy(); keep = conf_np >= 0.25
        sp, tp = sp[keep], tp[keep]
    return sp, tp, "loftr", {"device": device_name, "matches": int(len(sp)), "model_cache": True, "source_scale": float(sa), "target_scale": float(sb), "isolated_runtime": True}


def _match(req: dict[str, Any]) -> dict[str, Any]:
    source = np.load(str(req["source"]), allow_pickle=False)
    target = np.load(str(req["target"]), allow_pickle=False)
    kind = str(req.get("kind"))
    requested = str(req.get("device", "auto"))
    device = _select_device(requested)
    fn = _lightglue if kind == "lightglue" else _loftr
    try:
        sp, tp, method, diag = fn(source, target, req, device)
    except Exception as first:
        if device == "cpu":
            raise
        # MPS/CUDA kernels in third-party matchers may occasionally hit an
        # unsupported op.  Retry in this isolated process on CPU; the GUI never
        # sees a partial Torch import or corrupted module state.
        sp, tp, method, diag = fn(source, target, req, "cpu")
        diag["device_fallback"] = f"{device}->cpu:{type(first).__name__}"
    return {"source_points": sp.tolist(), "target_points": tp.tolist(), "method": method, "diagnostics": diag}


def main() -> int:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        request_id = -1
        try:
            req = json.loads(raw); request_id = int(req.get("request_id", -1))
            if req.get("op") == "shutdown":
                return 0
            if req.get("op") != "match":
                raise ValueError("unsupported op")
            payload = _match(req)
            row = {"ok": True, "request_id": request_id, **payload}
        except Exception as exc:
            row = {"ok": False, "request_id": request_id, "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-6000:]}"}
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
