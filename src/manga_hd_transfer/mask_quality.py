from __future__ import annotations

"""Mask transfer quality/SR helpers extracted from the renderer.

These functions may improve sampling/sharpness but never decide which region is
writable.  Keeping them outside ``mask_transfer`` makes SR policy independently
testable without importing the full renderer.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from .config import MaskReplaceConfig
from .external_command import run_external_command
from .io_utils import read_image, write_image
from .mask_geometry import _bbox_from_mask

def _masked_sharpness(image: np.ndarray, mask: np.ndarray) -> float:
    """Text-oriented sharpness, not whole-white-bubble sharpness.

    Measuring the entire bubble is easily fooled by JPEG grain or a crisp box
    border. Prefer dark glyph neighbourhoods; fall back to the full interior only
    when no text-like pixels are present.
    """
    if cv2.countNonZero(mask) == 0:
        return 0.0
    box = _bbox_from_mask(mask)
    if not box:
        return 0.0
    x0, y0, x1, y1 = box
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    m = mask[y0:y1, x0:x1] > 0
    if gray.size == 0 or np.count_nonzero(m) < 20:
        return 0.0
    dark = m & (gray < 190)
    if np.count_nonzero(dark) >= 12:
        dark = cv2.dilate(dark.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
        sel = m & dark
    else:
        sel = m
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    vals = lap[sel]
    return float(np.var(vals)) if vals.size else 0.0


def _pixel_enhance_text_raster(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray | None, dict]:
    """Sharpen low-resolution SOURCE text without OCR, reflow or glyph rebuilding."""
    if not bool(getattr(cfg, "pixel_enhance_enabled", True)) or cv2.countNonZero(mask) < 24:
        return None, {"enabled": False}
    use = mask > 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stroke = use & (gray < 235)
    if int(np.count_nonzero(stroke)) < int(getattr(cfg, "content_completeness_min_ink_pixels", 18)):
        return None, {"enabled": True, "reason": "too_little_ink"}
    scale = max(1.0, float(getattr(cfg, "pixel_enhance_upscale", 2.0)))
    h, w = image.shape[:2]
    if scale > 1.01:
        up = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_LANCZOS4)
        work = cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)
    else:
        work = image.copy()
    sigma = max(0.2, float(getattr(cfg, "pixel_enhance_unsharp_sigma", 0.85)))
    amount = max(0.0, float(getattr(cfg, "pixel_enhance_unsharp_amount", 0.55)))
    blur = cv2.GaussianBlur(work, (0, 0), sigma)
    sharp = cv2.addWeighted(work, 1.0 + amount, blur, -amount, 0)
    # Keep paper/background byte-stable. Only glyph-like pixels can change, and
    # cap darkening so JPEG halos are not converted into heavy fake strokes.
    out = image.copy()
    max_dark = max(0, int(getattr(cfg, "pixel_enhance_max_darkening", 28)))
    lo = np.maximum(image.astype(np.int16) - max_dark, 0).astype(np.uint8)
    candidate = np.maximum(sharp, lo)
    out[stroke] = candidate[stroke]
    before = _masked_sharpness(image, mask)
    after = _masked_sharpness(out, mask)
    if after <= before * 1.015:
        return None, {"enabled": True, "before_sharpness": float(before), "after_sharpness": float(after), "reason": "no_material_gain"}
    return out, {
        "enabled": True, "before_sharpness": float(before), "after_sharpness": float(after),
        "scale": float(scale), "amount": float(amount), "stroke_pixels": int(np.count_nonzero(stroke)),
    }


def _superresolve_patch(patch: np.ndarray, desired_scale: float, cfg: MaskReplaceConfig) -> tuple[np.ndarray, str, float]:
    if cfg.sr_backend == "off" or desired_scale < cfg.sr_min_trigger:
        return patch, "off", 1.0
    desired_scale = float(np.clip(desired_scale, 1.0, cfg.sr_max_scale))
    backend = cfg.sr_backend
    if backend == "auto":
        if cfg.sr_model_path:
            try:
                import spandrel  # noqa: F401
                import torch  # noqa: F401
                backend = "torch"
            except Exception:
                backend = "external" if cfg.sr_command else "lanczos"
        else:
            backend = "external" if cfg.sr_command else "lanczos"
    if backend == "torch":
        if not cfg.sr_model_path:
            if cfg.sr_backend == "torch":
                raise ValueError("mask_replace.sr_backend='torch' requires mask_replace.sr_model_path")
            backend = "lanczos"
        else:
            try:
                from .superres import upscale_patch
                result, actual = upscale_patch(
                    patch, desired_scale, model_path=cfg.sr_model_path,
                    device_preference=cfg.sr_device, precision=cfg.sr_precision,
                    tile_size=cfg.sr_tile_size, tile_overlap=cfg.sr_tile_overlap,
                    fallback_cpu=cfg.sr_fallback_cpu,
                )
                return result, "torch", float(actual)
            except Exception:
                if cfg.sr_backend == "torch":
                    raise
                backend = "lanczos"
    if backend == "external":
        if not cfg.sr_command:
            if cfg.sr_backend == "external":
                raise ValueError("mask_replace.sr_backend='external' requires mask_replace.sr_command")
            backend = "lanczos"
        else:
            with tempfile.TemporaryDirectory(prefix="mhd-sr-") as td:
                root = Path(td)
                inp, out = root / "input.png", root / "output.png"
                write_image(inp, patch)
                scale_int = 4 if desired_scale > 2.4 else 2
                proc = run_external_command(
                    cfg.sr_command, {"input": inp, "output": out, "scale": scale_int},
                    timeout=cfg.sr_timeout_seconds,
                    allow_shell=bool(getattr(cfg, "sr_allow_shell", False)),
                )
                if proc.returncode == 0 and out.exists():
                    result = read_image(out)
                    actual = result.shape[1] / max(1, patch.shape[1])
                    return result, "external", float(actual)
                backend = "lanczos"
    if backend == "lanczos":
        nw = max(1, int(round(patch.shape[1] * desired_scale)))
        nh = max(1, int(round(patch.shape[0] * desired_scale)))
        out = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        if cfg.sharpen_amount > 0:
            blur = cv2.GaussianBlur(out, (0, 0), 0.8)
            out = cv2.addWeighted(out, 1.0 + cfg.sharpen_amount, blur, -cfg.sharpen_amount, 0)
        return out, "lanczos", nw / max(1, patch.shape[1])
    raise ValueError(f"Unknown mask_replace.sr_backend: {cfg.sr_backend}")


def _normalize_bubble_background(patch: np.ndarray, patch_mask: np.ndarray, target: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    src_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    src_sel = (patch_mask > 0) & (src_gray > 155)
    dst_sel = (target_mask > 0) & (dst_gray > 155)
    if np.count_nonzero(src_sel) < 20 or np.count_nonzero(dst_sel) < 20:
        return patch
    src_bg = np.median(patch[src_sel], axis=0).astype(np.float32)
    dst_bg = np.median(target[dst_sel], axis=0).astype(np.float32)
    gain = np.clip(dst_bg / np.maximum(src_bg, 20.0), 0.88, 1.18)
    out = patch.astype(np.float32) * gain.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


__all__ = ['_masked_sharpness', '_pixel_enhance_text_raster', '_superresolve_patch', '_normalize_bubble_background']
