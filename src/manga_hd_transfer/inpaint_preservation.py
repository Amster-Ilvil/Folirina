"""Pixel authority and conservative local flat-background proofs for inpainting."""
from __future__ import annotations

import cv2
import numpy as np


def binary_text_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('inpainting expects a uint8 HxWx3 image')
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != image.shape[:2]:
        raise ValueError('inpainting mask dimensions must match the image')
    if not np.all(np.isfinite(mask)):
        raise ValueError('inpainting mask must contain finite values')
    return (mask > 0).astype(np.uint8) * 255


def commit_mask_only(image: np.ndarray, repaired: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not isinstance(repaired, np.ndarray) or repaired.shape != image.shape or repaired.dtype != np.uint8:
        raise ValueError('inpainting output must be a matching uint8 image')
    result = image.copy()
    result[mask > 0] = repaired[mask > 0]
    return result


def preclear_flat_components(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fill each admitted mask island only when its own visible ring is flat.

    Never derive a colour from a different balloon or from another masked glyph.
    Textured/gradient rings remain model-owned; no mask expansion is performed.
    """
    mask = binary_text_mask(image, mask)
    result, remaining = image.copy(), mask.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    h, w = mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    rows = []
    for label in range(1, count):
        x, y, bw, bh, area = map(int, stats[label])
        xa, ya, xb, yb = max(0,x-8), max(0,y-8), min(w,x+bw+8), min(h,y+bh+8)
        local = (labels[ya:yb,xa:xb] == label).astype(np.uint8)
        ring = (cv2.dilate(local, kernel) > 0) & (mask[ya:yb,xa:xb] == 0)
        pixels = image[ya:yb,xa:xb][ring]
        if len(pixels) < 32:
            continue
        colour = np.median(pixels, axis=0)
        deviation = np.max(np.abs(pixels.astype(np.float32)-colour),axis=1)
        # Both quantiles and almost-all-sample agreement must hold. A robust
        # median alone can mistake paper around a thin art line for flatness.
        spread = np.max(np.percentile(pixels,95,axis=0)-np.percentile(pixels,5,axis=0))
        if spread > 8 or np.mean(deviation <= 6) < .99:
            continue
        use = local > 0
        result[ya:yb,xa:xb][use] = np.rint(colour).astype(np.uint8)
        remaining[ya:yb,xa:xb][use] = 0
        rows.append({'label':label,'pixels':area,'fill_bgr':np.rint(colour).astype(int).tolist()})
    return result, remaining, {
        'flat_components':len(rows), 'flat_pixels':int(np.count_nonzero(mask)-np.count_nonzero(remaining)),
        'remaining_inpaint_pixels':int(np.count_nonzero(remaining)), 'local_fills':rows,
    }
