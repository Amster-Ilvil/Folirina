from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import cv2
import numpy as np

from .io_utils import save_json, write_image


def make_text_layer_rgba(shape: tuple[int, int], masks: list[np.ndarray], color=(0, 0, 0)) -> np.ndarray:
    h, w = shape
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    for mask in masks:
        if mask is None:
            continue
        layer[mask > 0, :3] = color
        layer[mask > 0, 3] = np.maximum(layer[mask > 0, 3], mask[mask > 0])
    return layer


def write_rgba(path: str | Path, rgba: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    ok, data = cv2.imencode(p.suffix or ".png", bgra)
    if not ok:
        raise ValueError(f"Could not encode {p}")
    data.tofile(p)


def export_openraster(
    path: str | Path,
    original: np.ndarray,
    inpainted: np.ndarray,
    text_rgba: np.ndarray,
    transfer_rgba: np.ndarray | None = None,
) -> None:
    """Write an OpenRaster layered file (supported by Krita/GIMP and lossless)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    h, w = original.shape[:2]
    tmp = p.parent / f".{p.stem}_ora_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "data").mkdir(parents=True)
    write_image(tmp / "data" / "original.png", original)
    write_image(tmp / "data" / "inpainted.png", inpainted)
    write_rgba(tmp / "data" / "text.png", text_rgba)
    if transfer_rgba is not None:
        write_rgba(tmp / "data" / "mask_transfer.png", transfer_rgba)

    image = Element("image", {"version": "0.0.1", "w": str(w), "h": str(h), "name": p.stem})
    stack = SubElement(image, "stack", {"name": "root"})
    # ORA order is top to bottom.
    SubElement(stack, "layer", {"name": "Chinese text", "src": "data/text.png"})
    if transfer_rgba is not None:
        SubElement(stack, "layer", {"name": "Chinese bubble patch transfer", "src": "data/mask_transfer.png"})
    SubElement(stack, "layer", {"name": "Inpainted / HD base", "src": "data/inpainted.png"})
    SubElement(stack, "layer", {"name": "Original HD Japanese", "src": "data/original.png"})
    (tmp / "stack.xml").write_bytes(tostring(image, encoding="utf-8", xml_declaration=True))
    (tmp / "mimetype").write_text("image/openraster", encoding="ascii")

    with zipfile.ZipFile(p, "w") as zf:
        zf.write(tmp / "mimetype", "mimetype", compress_type=zipfile.ZIP_STORED)
        zf.write(tmp / "stack.xml", "stack.xml", compress_type=zipfile.ZIP_DEFLATED)
        names = ["original.png", "inpainted.png", "text.png"]
        if transfer_rgba is not None:
            names.append("mask_transfer.png")
        for name in names:
            zf.write(tmp / "data" / name, f"data/{name}", compress_type=zipfile.ZIP_DEFLATED)
    shutil.rmtree(tmp)


def export_psd_imagemagick(
    path: str | Path,
    original_path: str | Path,
    inpainted_path: str | Path,
    text_path: str | Path,
    transfer_path: str | Path | None = None,
) -> bool:
    """Best-effort layered PSD export via ImageMagick when available.

    The engine always writes lossless layer PNGs and ORA; PSD is additionally emitted when
    ImageMagick is installed, avoiding a hard dependency on a PSD writer.
    """
    magick = shutil.which("magick")
    if not magick:
        return False
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        magick,
        str(original_path), "-set", "label", "Original HD Japanese",
        str(inpainted_path), "-set", "label", "Inpainted / HD base",
    ]
    if transfer_path:
        cmd += [str(transfer_path), "-set", "label", "Chinese bubble patch transfer"]
    cmd += [str(text_path), "-set", "label", "Chinese text", str(p)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and p.exists() and p.stat().st_size > 0
