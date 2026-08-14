from __future__ import annotations

import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import InpaintingConfig
from .io_utils import read_image, write_image


@dataclass(slots=True)
class InpaintResult:
    image: np.ndarray
    method: str
    diagnostics: dict


def _ring(mask: np.ndarray, radius: int = 7) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated = cv2.dilate((mask > 0).astype(np.uint8) * 255, k)
    return cv2.bitwise_and(dilated, cv2.bitwise_not((mask > 0).astype(np.uint8) * 255))


def _background_stats(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, int]:
    ring = _ring(mask)
    pixels = image[ring > 0]
    if len(pixels) == 0:
        pixels = image[mask == 0]
    if len(pixels) == 0:
        return np.array([255, 255, 255], dtype=np.uint8), 9999.0, 0
    # Robust median resists nearby black bubble outlines.
    median = np.median(pixels, axis=0).astype(np.uint8)
    gray = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).reshape(-1)
    lo, hi = np.quantile(gray, [0.15, 0.85]) if len(gray) > 10 else (gray.min(), gray.max())
    trimmed = gray[(gray >= lo) & (gray <= hi)]
    variance = float(np.var(trimmed)) if len(trimmed) else float(np.var(gray))
    return median, variance, len(pixels)


def solid_fill(image: np.ndarray, mask: np.ndarray) -> InpaintResult:
    result = image.copy()
    color, variance, samples = _background_stats(image, mask)
    result[mask > 0] = color
    return InpaintResult(result, "solid", {"fill_bgr": color.tolist(), "ring_variance": variance, "samples": samples})




def threshold_clear(image: np.ndarray, mask: np.ndarray, cfg: InpaintingConfig) -> InpaintResult:
    result = image.copy()
    color, variance, samples = _background_stats(image, mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ring = _ring(mask)
    vals = gray[ring > 0]
    paper = float(np.percentile(vals, 80)) if vals.size else 245.0
    threshold = int(np.clip(paper - int(getattr(cfg, "threshold_clear_dark_offset", 52)), 105, 220))
    dark = ((mask > 0) & (gray <= threshold))
    dark_ratio = float(np.count_nonzero(dark) / max(1, int(np.count_nonzero(mask))))
    if dark_ratio >= float(getattr(cfg, "threshold_clear_min_dark_ratio", 0.008)):
        result[dark] = color
    return InpaintResult(result, "threshold_clear", {
        "fill_bgr": color.tolist(),
        "ring_variance": variance,
        "samples": samples,
        "paper_gray": paper,
        "threshold": threshold,
        "dark_ratio": dark_ratio,
    })


def opencv_inpaint(image: np.ndarray, mask: np.ndarray, radius: float = 3.0) -> InpaintResult:
    result = cv2.inpaint(image, (mask > 0).astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    color, variance, samples = _background_stats(image, mask)
    return InpaintResult(result, "opencv-telea", {"ring_variance": variance, "samples": samples, "radius": radius})


def lama_subprocess(image: np.ndarray, mask: np.ndarray, command: str, timeout: int = 120) -> InpaintResult:
    """Run an external LaMa-compatible command without vendoring model code.

    The configured command may use placeholders: {input}, {mask}, {output}.
    It must write the final image to {output}. Example wrapper:
      python lama_wrapper.py --input {input} --mask {mask} --output {output}
    """
    with tempfile.TemporaryDirectory(prefix="mhd-lama-") as td:
        root = Path(td)
        inp, msk, out = root / "input.png", root / "mask.png", root / "output.png"
        write_image(inp, image)
        write_image(msk, (mask > 0).astype(np.uint8) * 255)
        rendered = command.format(input=shlex.quote(str(inp)), mask=shlex.quote(str(msk)), output=shlex.quote(str(out)))
        proc = subprocess.run(rendered, shell=True, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"LaMa command failed ({proc.returncode}): {proc.stderr[-2000:]}")
        result = read_image(out)
        if result.shape[:2] != image.shape[:2]:
            raise RuntimeError(f"LaMa output shape mismatch: {result.shape} vs {image.shape}")
        return InpaintResult(result, "lama-external", {"stdout": proc.stdout[-1000:], "command_exit": proc.returncode})


def inpaint_image(image: np.ndarray, mask: np.ndarray, config: InpaintingConfig | None = None) -> InpaintResult:
    cfg = config or InpaintingConfig()
    if mask is None or cv2.countNonZero(mask) == 0:
        return InpaintResult(image.copy(), "none", {"reason": "empty_mask"})
    backend = cfg.backend.lower()
    if backend == "solid":
        return solid_fill(image, mask)
    if backend == "opencv":
        return opencv_inpaint(image, mask, cfg.opencv_radius)
    if backend == "threshold_clear":
        return threshold_clear(image, mask, cfg)
    if backend == "lama":
        if not cfg.lama_command:
            raise ValueError("inpainting.backend='lama' requires inpainting.lama_command")
        return lama_subprocess(image, mask, cfg.lama_command, cfg.lama_timeout_seconds)
    if backend != "auto":
        raise ValueError(f"Unknown inpainting backend: {cfg.backend}")

    _, variance, _ = _background_stats(image, mask)
    if bool(getattr(cfg, "prefer_threshold_clear_for_white", True)) and variance <= float(getattr(cfg, "threshold_clear_max_variance", 140.0)):
        result = threshold_clear(image, mask, cfg)
        result.diagnostics["auto_selected"] = True
        return result
    if variance <= cfg.solid_variance_threshold:
        result = solid_fill(image, mask)
        result.diagnostics["auto_selected"] = True
        return result
    if cfg.lama_command:
        try:
            result = lama_subprocess(image, mask, cfg.lama_command, cfg.lama_timeout_seconds)
            result.diagnostics["auto_selected"] = True
            return result
        except Exception as e:
            # Publication QA sees the fallback in diagnostics instead of hiding it.
            fallback = opencv_inpaint(image, mask, cfg.opencv_radius)
            fallback.diagnostics.update({"auto_selected": True, "lama_error": str(e)})
            return fallback
    result = opencv_inpaint(image, mask, cfg.opencv_radius)
    result.diagnostics["auto_selected"] = True
    return result
