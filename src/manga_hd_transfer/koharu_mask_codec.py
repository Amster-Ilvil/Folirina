"""Lossless compact transport for disconnected Koharu foreground masks."""
import base64

import cv2
import numpy as np


def refine_text_mask(mask: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Koharu-style dilation/closing, bounded by the admitted local region."""
    radius = max(1, int(round(max(mask.shape) / 1024.0 * 6.0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    expanded = cv2.dilate((mask > 0).astype(np.uint8)*255, kernel)
    closed = cv2.morphologyEx(expanded, cv2.MORPH_CLOSE, kernel)
    return cv2.bitwise_and(closed, (gate > 0).astype(np.uint8)*255)


def encode_mask(mask: np.ndarray) -> dict | None:
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if binary.ndim != 2:
        raise ValueError("Koharu mask must be two-dimensional")
    ys, xs = np.nonzero(binary)
    if not len(xs):
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)
    ok, png = cv2.imencode(".png", binary[y0:y1, x0:x1])
    if not ok:
        raise ValueError("Could not encode Koharu mask")
    return {"canvas": list(binary.shape), "box": [x0,y0,x1,y1],
            "png": base64.b64encode(png.tobytes()).decode("ascii")}


def decode_mask(payload: dict, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, payload["canvas"])
    if h <= 0 or w <= 0 or h*w > 16777216:
        raise ValueError("Invalid Koharu native mask size")
    x0,y0,x1,y1 = map(int, payload["box"])
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Invalid Koharu native mask crop")
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(payload["png"], validate=True), np.uint8), cv2.IMREAD_GRAYSCALE)
    if crop is None or crop.shape != (y1-y0,x1-x0):
        raise ValueError("Koharu mask crop dimensions differ")
    native = np.zeros((h,w), np.uint8)
    native[y0:y1,x0:x1] = (crop > 0).astype(np.uint8)*255
    return cv2.resize(native, (int(shape[1]),int(shape[0])), interpolation=cv2.INTER_NEAREST)
