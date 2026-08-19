from __future__ import annotations

"""Shared non-UI review helpers.

These functions carry schema/image primitives only and intentionally do not own
manual-force/manual-effect workflows.
"""

from pathlib import Path
import cv2
import numpy as np

from .io_utils import read_image
from .models import BubbleInstance, TextBlock, TextUnit
from .schema_compat import as_dict, as_dict_rows, normalize_route_meta
from .text_only_transfer import target_text_mask_in_container

def _dict_or_empty(value):
    """Return a plain dict for mixed/legacy JSON schema values."""
    return as_dict(value)

def _route_meta(meta, key: str) -> dict:
    return normalize_route_meta(_dict_or_empty(meta).get(key))

def _dict_rows(value) -> list[dict]:
    """Normalize a stale list-like review field to dictionary rows only."""
    return as_dict_rows(value)

def _text_block(row: dict) -> TextBlock:
    return TextBlock(**row)

def _text_unit(row: dict) -> TextUnit:
    return TextUnit(**row)

def _load_target_bubbles(page_dir: Path, rows: list[dict]) -> list[BubbleInstance]:
    out = []
    for row in rows:
        b = BubbleInstance(
            id=row["id"],
            polygon=row["polygon"],
            confidence=row.get("confidence", 1.0),
            kind=row.get("kind", "speech"),
            block_ids=list(row.get("block_ids", [])),
            meta=as_dict(row.get("meta")),
        )
        mp = page_dir / "bubbles" / f"{b.id}.png"
        sp = page_dir / "bubbles" / f"{b.id}_safe.png"
        if mp.exists():
            b.mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if sp.exists():
            b.safe_mask = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
        out.append(b)
    return out

def _rect_mask(shape: tuple[int, int], bbox: list[int] | tuple[int, int, int, int], inset: int = 2) -> np.ndarray:
    x0, y0, x1, y1 = map(int, bbox)
    x0 = max(0, min(shape[1], x0 + inset)); y0 = max(0, min(shape[0], y0 + inset))
    x1 = max(0, min(shape[1], x1 - inset)); y1 = max(0, min(shape[0], y1 - inset))
    mask = np.zeros(shape, np.uint8)
    if x1 > x0 and y1 > y0:
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask

def _clear_region_to_paper(rendered: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rendered.copy()
    if mask is None or cv2.countNonZero(mask) == 0:
        return out
    sel = mask > 0
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    bright = sel & (gray >= 205)
    if np.count_nonzero(bright) >= 20:
        paper = np.median(target[bright], axis=0).astype(np.uint8)
    else:
        paper = np.array([255, 255, 255], np.uint8)
    out[sel] = paper
    return out

def _source_for_review(page_dir: Path, project: dict) -> np.ndarray:
    local = page_dir / "source_original.png"
    if local.exists():
        return read_image(local)
    pair = dict(project.get("pair", {}) or {})
    source_path = str(pair.get("source_path", "") or "")
    if not source_path:
        raise FileNotFoundError("manual effect transfer needs source_original.png or pair.source_path")
    return read_image(source_path)

def _polygon_mask(shape: tuple[int, int], polygon) -> np.ndarray:
    """Rasterize a project OCR/detector polygon without assuming a rectangle."""
    out = np.zeros(shape, np.uint8)
    try:
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return out
    if pts.shape[0] < 3:
        return out
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, shape[1] - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, shape[0] - 1))
    cv2.fillPoly(out, [np.rint(pts).astype(np.int32)], 255)
    return out

def _project_text_ink_mask(image: np.ndarray, rows) -> tuple[np.ndarray, int]:
    """Turn OCR/detector text polygons into compact ink evidence.

    Project polygons are used only as trusted ROIs.  We still select actual ink
    inside them so an OCR box can never become a broad white/background erase.
    """
    shape = image.shape[:2]
    out = np.zeros(shape, np.uint8)
    used = 0
    for row in _dict_rows(rows):
        poly = row.get("polygon") or []
        roi = _polygon_mask(shape, poly)
        if cv2.countNonZero(roi) == 0:
            continue
        ink = target_text_mask_in_container(image, roi)
        if cv2.countNonZero(ink) <= 0:
            continue
        out = np.maximum(out, ink)
        used += 1
    return out, used

def _write_bgra(path: Path, bgra: np.ndarray) -> None:
    ok, data = cv2.imencode(".png", bgra)
    if not ok:
        raise ValueError(f"could not encode {path.name}")
    data.tofile(path)

def _alpha_over_bgra(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Alpha-composite one BGRA manual layer over an existing BGRA layer."""
    if bottom.shape != top.shape:
        raise ValueError("manual effect layer size mismatch")
    ba = bottom[:, :, 3].astype(np.float32) / 255.0
    ta = top[:, :, 3].astype(np.float32) / 255.0
    out_a = ta + ba * (1.0 - ta)
    out = np.zeros_like(bottom)
    denom = np.maximum(out_a, 1e-6)
    for c in range(3):
        bc = bottom[:, :, c].astype(np.float32)
        tc = top[:, :, c].astype(np.float32)
        out[:, :, c] = np.clip((tc * ta + bc * ba * (1.0 - ta)) / denom, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)
    return out

__all__ = ['_dict_or_empty', '_route_meta', '_dict_rows', '_text_block', '_text_unit', '_load_target_bubbles', '_rect_mask', '_clear_region_to_paper', '_source_for_review', '_polygon_mask', '_project_text_ink_mask', '_write_bgra', '_alpha_over_bgra']
