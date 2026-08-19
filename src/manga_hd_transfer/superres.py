from __future__ import annotations

"""Optional local super-resolution adapter with Apple MPS support.

No model is downloaded by this module.  A local model path must be supplied.
Spandrel is used only when installed, allowing common ESRGAN/RCAN-style weights
without hard-coding one model architecture.
"""

import math
import threading
from pathlib import Path

import cv2
import numpy as np

from .runtime import accelerator_lock, empty_accelerator_cache, select_device

_MODEL_CACHE: dict[tuple[str, int, int, str], object] = {}
_MODEL_LOCK = threading.RLock()


def _model_cache_key(path: str, device: str) -> tuple[str, int, int, str]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"超分模型不存在: {p}")
    st = p.stat()
    return (str(p), int(st.st_size), int(st.st_mtime_ns), str(device))


def _load_model(path: str, device: str):
    key = _model_cache_key(path, device)
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        try:
            import torch
            from spandrel import ModelLoader
        except Exception as e:
            raise RuntimeError("Torch 超分需要可选依赖 torch + spandrel") from e
        p = Path(key[0])
        model = ModelLoader().load_from_file(str(p))
        model = model.eval().to(torch.device(device))
        # If the user replaced weights at the same path during a long session,
        # discard the stale object instead of silently reusing old parameters.
        stale = [k for k in _MODEL_CACHE if k[0] == key[0] and k[3] == key[3] and k != key]
        for old_key in stale:
            _MODEL_CACHE.pop(old_key, None)
        _MODEL_CACHE[key] = model
        return model


def clear_model_cache() -> None:
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()
    empty_accelerator_cache("auto")


def _infer_tile(model, tile_bgr: np.ndarray, device: str, precision: str) -> np.ndarray:
    import torch
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    ten = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    ten = ten.to(torch.device(device))
    use_half = precision == "fp16" and device in {"mps", "cuda"}
    if use_half:
        ten = ten.half()
        try:
            model = model.half()
        except Exception:
            pass
    else:
        ten = ten.float()
        try:
            model = model.float()
        except Exception:
            pass
    with torch.inference_mode():
        out = model(ten)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = next(iter(out.values()))
    arr = out.detach().float().clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0)
    return cv2.cvtColor(np.round(arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def upscale_patch(
    patch: np.ndarray,
    desired_scale: float,
    *,
    model_path: str,
    device_preference: str = "auto",
    precision: str = "fp32",
    tile_size: int = 512,
    tile_overlap: int = 24,
    fallback_cpu: bool = True,
) -> tuple[np.ndarray, str, float]:
    device = select_device(device_preference)
    model = _load_model(model_path, device)
    h, w = patch.shape[:2]

    def run_full(dev: str):
        mdl = _load_model(model_path, dev)
        # Small speech-bubble patches are normally cheaper and cleaner as one tile.
        if max(h, w) <= max(64, tile_size):
            return _infer_tile(mdl, patch, dev, precision)

        # Generic overlap tiling.  Infer one tile to discover model scale.
        first = patch[: min(tile_size, h), : min(tile_size, w)]
        probe = _infer_tile(mdl, first, dev, precision)
        native_scale = probe.shape[1] / max(1, first.shape[1])
        out_h, out_w = max(1, round(h * native_scale)), max(1, round(w * native_scale))
        accum = np.zeros((out_h, out_w, 3), np.float32)
        weight = np.zeros((out_h, out_w, 1), np.float32)
        step = max(32, tile_size - max(0, tile_overlap))
        ys = list(range(0, max(1, h - tile_size + 1), step))
        xs = list(range(0, max(1, w - tile_size + 1), step))
        if not ys or ys[-1] != max(0, h - tile_size): ys.append(max(0, h - tile_size))
        if not xs or xs[-1] != max(0, w - tile_size): xs.append(max(0, w - tile_size))
        for y in sorted(set(ys)):
            for x in sorted(set(xs)):
                tile = patch[y:min(h, y+tile_size), x:min(w, x+tile_size)]
                up = _infer_tile(mdl, tile, dev, precision)
                oy, ox = round(y * native_scale), round(x * native_scale)
                y1, x1 = min(out_h, oy + up.shape[0]), min(out_w, ox + up.shape[1])
                up = up[:y1-oy, :x1-ox]
                # Feather overlaps, but never blur the final target geometry; this is
                # only model-tile blending inside the source patch.
                wh, ww = up.shape[:2]
                wy = np.ones(wh, np.float32); wx = np.ones(ww, np.float32)
                edge = max(1, round(tile_overlap * native_scale))
                if oy > 0: wy[:min(edge,wh)] = np.linspace(0.05,1,min(edge,wh))
                if y1 < out_h: wy[-min(edge,wh):] = np.minimum(wy[-min(edge,wh):], np.linspace(1,0.05,min(edge,wh)))
                if ox > 0: wx[:min(edge,ww)] = np.linspace(0.05,1,min(edge,ww))
                if x1 < out_w: wx[-min(edge,ww):] = np.minimum(wx[-min(edge,ww):], np.linspace(1,0.05,min(edge,ww)))
                wwgt = (wy[:,None] * wx[None,:])[:,:,None]
                accum[oy:y1,ox:x1] += up.astype(np.float32) * wwgt
                weight[oy:y1,ox:x1] += wwgt
        return np.clip(accum / np.maximum(weight, 1e-6), 0, 255).astype(np.uint8)

    try:
        with accelerator_lock():
            out = run_full(device)
    except Exception:
        if not fallback_cpu or device == "cpu":
            raise
        # MPS operator coverage differs across models.  A CPU fallback keeps batch
        # jobs moving without silently changing geometry or downloading anything.
        with accelerator_lock():
            out = run_full("cpu")
        device = "cpu-fallback"

    native = out.shape[1] / max(1, w)
    desired_scale = max(1.0, float(desired_scale))
    if abs(native - desired_scale) > 0.03:
        nw, nh = max(1, round(w * desired_scale)), max(1, round(h * desired_scale))
        interpolation = cv2.INTER_AREA if desired_scale < native else cv2.INTER_LANCZOS4
        out = cv2.resize(out, (nw, nh), interpolation=interpolation)
    actual = out.shape[1] / max(1, w)
    return out, f"torch-{device}", float(actual)
