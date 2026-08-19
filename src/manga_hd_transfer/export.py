from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import cv2
import numpy as np

from .io_utils import write_image


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
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    write_image(p, bgra)


def export_openraster(
    path: str | Path,
    original: np.ndarray,
    inpainted: np.ndarray,
    text_rgba: np.ndarray,
    transfer_rgba: np.ndarray | None = None,
) -> None:
    """Write an OpenRaster layered file (supported by Krita/GIMP and lossless).

    Temporary layer files and the final ZIP are unique per invocation.  The ORA
    itself is replaced atomically only after the archive has been fully written,
    so a cancelled review/export cannot destroy a previously valid editable file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    h, w = original.shape[:2]
    tmp_zip: str | None = None
    with tempfile.TemporaryDirectory(prefix=f".{p.stem}_ora_", dir=str(p.parent)) as td:
        tmp = Path(td)
        (tmp / "data").mkdir(parents=True)
        write_image(tmp / "data" / "original.png", original)
        write_image(tmp / "data" / "inpainted.png", inpainted)
        write_rgba(tmp / "data" / "text.png", text_rgba)
        if transfer_rgba is not None:
            write_rgba(tmp / "data" / "mask_transfer.png", transfer_rgba)

        image = Element("image", {"version": "0.0.1", "w": str(w), "h": str(h), "name": p.stem})
        stack = SubElement(image, "stack", {"name": "root"})
        SubElement(stack, "layer", {"name": "Chinese text", "src": "data/text.png"})
        if transfer_rgba is not None:
            SubElement(stack, "layer", {"name": "Chinese bubble patch transfer", "src": "data/mask_transfer.png"})
        SubElement(stack, "layer", {"name": "Inpainted / HD base", "src": "data/inpainted.png"})
        SubElement(stack, "layer", {"name": "Original HD Japanese", "src": "data/original.png"})
        (tmp / "stack.xml").write_bytes(tostring(image, encoding="utf-8", xml_declaration=True))
        (tmp / "mimetype").write_text("image/openraster", encoding="ascii")

        fd, tmp_zip = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
        os.close(fd)
        try:
            with zipfile.ZipFile(tmp_zip, "w") as zf:
                zf.write(tmp / "mimetype", "mimetype", compress_type=zipfile.ZIP_STORED)
                zf.write(tmp / "stack.xml", "stack.xml", compress_type=zipfile.ZIP_DEFLATED)
                names = ["original.png", "inpainted.png", "text.png"]
                if transfer_rgba is not None:
                    names.append("mask_transfer.png")
                for name in names:
                    zf.write(tmp / "data" / name, f"data/{name}", compress_type=zipfile.ZIP_DEFLATED)
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                bad = zf.testzip()
                if bad:
                    raise RuntimeError(f"OpenRaster archive integrity failure at {bad}")
            os.replace(tmp_zip, p)
            tmp_zip = None
        finally:
            if tmp_zip:
                try:
                    Path(tmp_zip).unlink(missing_ok=True)
                except OSError:
                    pass

def export_psd_imagemagick(
    path: str | Path,
    original_path: str | Path,
    inpainted_path: str | Path,
    text_path: str | Path,
    transfer_path: str | Path | None = None,
    *,
    timeout_seconds: float = 60.0,
) -> bool:
    """Best-effort layered PSD export via ImageMagick when available.

    PSD is optional and must never hang page review. The export is written to a
    unique sibling ``.psd`` staging file and replaces an existing valid PSD only
    after ImageMagick succeeds, so timeout/plugin failures cannot destroy the
    previous editable artifact.
    """
    magick = shutil.which("magick")
    if not magick:
        return False
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.stem}.", suffix=".psd", dir=str(p.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        # ImageMagick expects to create/replace the output itself. Remove the
        # zero-byte mkstemp placeholder while keeping the unique path reserved by
        # name in our process.
        tmp.unlink(missing_ok=True)
        cmd = [
            magick,
            str(original_path), "-set", "label", "Original HD Japanese",
            str(inpainted_path), "-set", "label", "Inpainted / HD base",
        ]
        if transfer_path:
            cmd += [str(transfer_path), "-set", "label", "Chinese bubble patch transfer"]
        cmd += [str(text_path), "-set", "label", "Chinese text", str(tmp)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=max(1.0, float(timeout_seconds)),
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 0:
            return False
        os.replace(tmp, p)
        return True
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

