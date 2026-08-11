from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name.casefold())
    return tuple(int(p) if p.isdigit() else p for p in parts)


def list_images(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(root)
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS]
    return sorted(files, key=natural_key)


def read_image(path: str | Path) -> np.ndarray:
    p = Path(path)
    data = np.fromfile(p, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {p}")
    return img


def write_image(path: str | Path, image: np.ndarray, params: list[int] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image, params or [])
    if not ok:
        raise ValueError(f"Could not encode image as {ext}: {p}")
    encoded.tofile(p)


def save_json(path: str | Path, payload: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_bgr(image: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return arr.copy()


def stem_id(path: str | Path) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", Path(path).stem).strip("_") or "page"
