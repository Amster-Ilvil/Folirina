from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
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
    """Atomically replace a JSON state file.

    Page Manager state and review overrides can be written while the user is
    rapidly changing pages.  Writing to a sibling temporary file and replacing
    the destination prevents a crash/forced quit from leaving half-written JSON.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(p.parent),
            prefix=f".{p.name}.", suffix=".tmp", delete=False,
        ) as fh:
            tmp_name = fh.name
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, p)
        tmp_name = None
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


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
    # Keep Unicode letters/numbers (Japanese/Chinese/Korean included) so books
    # whose pages are named only with CJK characters do not all collapse to the
    # same ``page`` cache/output directory.  Replace only filesystem-unfriendly
    # punctuation/spacing while preserving the historical ASCII behaviour.
    stem = unicodedata.normalize("NFKC", Path(path).stem)
    return re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("_") or "page"
