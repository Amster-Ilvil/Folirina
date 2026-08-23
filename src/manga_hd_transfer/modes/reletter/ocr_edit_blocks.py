from __future__ import annotations

"""Mode-scoped manual OCR blocks for Mask+OCR and OCR Reletter.

This module is intentionally isolated from Direct / pure Mask / Reveal.  The
legacy runtime keys ``hybrid`` and ``reletter`` are retained for saved-project
compatibility, but editor artifacts live under the product-family names
``mask_ocr`` and ``ocr_reletter``.
"""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

import cv2

from ...io_utils import load_json, save_json
from .manual_effect_ops import map_target_bbox_to_source
from ...ocr import build_backend
from ...schema_compat import as_dict, as_dict_rows
from ...ocr_block_conflicts import canonicalize_ocr_blocks, conflicting_block_ids

OCR_EDIT_MODE_SCOPE = {
    "hybrid": "mask_ocr",
    "reletter": "ocr_reletter",
}


def is_ocr_edit_mode(mode: str | None) -> bool:
    return str(mode or "").strip().lower() in OCR_EDIT_MODE_SCOPE


def ocr_edit_scope(mode: str | None) -> str:
    key = str(mode or "").strip().lower()
    if key not in OCR_EDIT_MODE_SCOPE:
        raise ValueError(f"OCR block editor is unavailable for transfer mode: {mode}")
    return OCR_EDIT_MODE_SCOPE[key]


def ocr_edit_dir(page_dir: str | Path, mode: str | None) -> Path:
    return Path(page_dir) / "ocr_edit" / ocr_edit_scope(mode)


def ocr_blocks_path(page_dir: str | Path, mode: str | None) -> Path:
    return ocr_edit_dir(page_dir, mode) / "blocks.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_raw_ocr_blocks(page_dir: str | Path, mode: str | None) -> list[dict[str, Any]]:
    path = ocr_blocks_path(page_dir, mode)
    if not path.exists():
        return []
    try:
        payload = load_json(path)
    except Exception:
        return []
    rows = payload.get("blocks", []) if isinstance(payload, dict) else []
    return [dict(x) for x in as_dict_rows(rows)]


def load_ocr_blocks(page_dir: str | Path, mode: str | None) -> list[dict[str, Any]]:
    # Legacy v2.3.83 and earlier projects could accumulate several manual OCR
    # blocks over the same ROI.  Only one text authority may own one locator.
    # Canonicalize on every read so old projects stop double/triple-rendering
    # immediately, even before their JSON is rewritten by the next save/apply.
    rows, _ = canonicalize_ocr_blocks(_load_raw_ocr_blocks(page_dir, mode))
    return rows




def _json_safe(value):
    """Convert small numpy/scalar containers from image geometry to JSON values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return value

def save_ocr_blocks(page_dir: str | Path, mode: str | None, blocks: list[dict[str, Any]]) -> Path:
    scope = ocr_edit_scope(mode)
    path = ocr_blocks_path(page_dir, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical, suppressed = canonicalize_ocr_blocks(blocks)
    save_json(path, {
        "schema": "folirina.ocr_edit.blocks.v1",
        "scope": scope,
        "runtime_mode": str(mode or "").strip().lower(),
        "updated_at": _now(),
        "conflict_policy": "single_overlapping_authority_v1",
        "suppressed_legacy_conflicts": [_json_safe(x) for x in suppressed],
        "blocks": [_json_safe(dict(x)) for x in canonical],
    })
    return path


def upsert_ocr_block(page_dir: str | Path, mode: str | None, block: dict[str, Any]) -> dict[str, Any]:
    rows = load_ocr_blocks(page_dir, mode)
    item = deepcopy(as_dict(block))
    block_id = str(item.get("id") or f"ocr-block-{uuid.uuid4().hex[:10]}")
    item["id"] = block_id
    item.setdefault("source", "manual")
    item.setdefault("manual_override", True)
    item.setdefault("box_locked", True)
    item.setdefault("created_at", _now())
    item["updated_at"] = _now()

    # Saving a newly drawn ROI is an explicit user decision.  If it targets the
    # same area as an existing manual OCR block it *replaces* that authority; it
    # must never append a second full text layer over the first one.
    replaced = conflicting_block_ids(rows, item)
    if replaced:
        item["replaces_overlapping_ids"] = list(replaced)
    rows = [
        row for row in rows
        if str(row.get("id") or "") != block_id
        and str(row.get("id") or "") not in set(replaced)
    ]
    rows.append(item)
    save_ocr_blocks(page_dir, mode, rows)
    return item


def delete_ocr_block(page_dir: str | Path, mode: str | None, block_id: str) -> bool:
    rows = load_ocr_blocks(page_dir, mode)
    new_rows = [row for row in rows if str(row.get("id") or "") != str(block_id)]
    changed = len(new_rows) != len(rows)
    if changed:
        save_ocr_blocks(page_dir, mode, new_rows)
    return changed


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    if len(bbox) != 4:
        return []
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
    y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
    return [x0, y0, x1, y1] if x1 - x0 >= 2 and y1 - y0 >= 2 else []


def _offset_polygons(blocks: list, dx: int, dy: int) -> list[list[list[float]]]:
    out: list[list[list[float]]] = []
    for block in blocks:
        poly = getattr(block, "polygon", None) or []
        if len(poly) < 3:
            continue
        out.append([[float(x) + dx, float(y) + dy] for x, y in poly])
    return out


def _join_text(blocks: list) -> tuple[str, float]:
    valid = [b for b in blocks if str(getattr(b, "text", "") or "").strip()]
    valid.sort(key=lambda b: int(getattr(b, "reading_order", 0) or 0))
    text = "\n".join(str(getattr(b, "text", "") or "").strip() for b in valid).strip()
    conf = min((float(getattr(b, "confidence", 0.0) or 0.0) for b in valid), default=0.0)
    return text, conf


def _recognize_region(config, image, bbox: list[int], *, role: str) -> tuple[list, str, str | None]:
    ocr_cfg = config.ocr
    backend_name = str((ocr_cfg.source_backend if role == "source" else ocr_cfg.target_backend) or ocr_cfg.backend or "paddle")
    lang = str(ocr_cfg.source_lang if role == "source" else ocr_cfg.target_lang)
    backend = None
    try:
        backend = build_backend(ocr_cfg, lang, backend_name, role=role)
        if not bool(getattr(backend, "supports_crop_recognition", True)):
            raise RuntimeError(f"{backend_name} 不支持人工局部 OCR，请切换到 Paddle / Manga OCR / Apple OCR。")
        blocks = backend.recognize_region(image, tuple(bbox), image_path=None)
        return blocks, backend_name, None
    except Exception as exc:
        return [], backend_name, str(exc)
    finally:
        close = getattr(backend, "close", None) if backend is not None else None
        if callable(close):
            try:
                close()
            except Exception:
                pass


def recognize_manual_ocr_block(
    project: dict[str, Any],
    source_path: str | Path,
    target_path: str | Path,
    target_bbox: list[int],
    config,
    *,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """OCR one manually selected TARGET rectangle.

    Chinese content is recognized from the SOURCE crop mapped by persisted page
    registration. TARGET OCR is used only as a Japanese-clear geometry hint.
    Typography from an existing block is preserved across re-OCR.
    """
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if source is None or target is None:
        raise FileNotFoundError("人工 OCR 无法读取 SOURCE / TARGET 图片。")
    th, tw = target.shape[:2]
    sh, sw = source.shape[:2]
    tb = _clamp_bbox(list(target_bbox), tw, th)
    if not tb:
        raise ValueError("人工 OCR 选框太小或超出图片范围。")
    sb = _clamp_bbox(map_target_bbox_to_source(project, tb), sw, sh)
    if not sb:
        raise ValueError("无法把人工 OCR 选框映射回 SOURCE。")

    source_blocks, source_backend, source_error = _recognize_region(config, source, sb, role="source")
    target_blocks, target_backend, target_error = _recognize_region(config, target, tb, role="target")
    text, confidence = _join_text(source_blocks)
    target_polys = _offset_polygons(target_blocks, tb[0], tb[1])

    old = deepcopy(as_dict(existing))
    item = {
        "id": str(old.get("id") or f"ocr-block-{uuid.uuid4().hex[:10]}"),
        "source": "manual",
        "target_bbox": tb,
        "source_bbox": sb,
        "ocr_text": text,
        "render_text": text if text else str(old.get("render_text") or old.get("ocr_text") or ""),
        "confidence": confidence,
        "source_backend": source_backend,
        "target_backend": target_backend,
        "source_ocr_error": source_error,
        "target_ocr_error": target_error,
        "target_ocr_polygons": target_polys,
        "orientation": str(old.get("orientation") or "auto"),
        "line_break_mode": str(old.get("line_break_mode") or "smart"),
        "layout_mode": str(old.get("layout_mode") or "smart_scaling"),
        "font_path": str(old.get("font_path") or ""),
        "font_size": int(old.get("font_size") or 0),
        "columns": int(old.get("columns") or 0),
        "line_spacing_ratio": old.get("line_spacing_ratio", None),
        "letter_spacing_ratio": old.get("letter_spacing_ratio", None),
        "column_spacing_ratio": old.get("column_spacing_ratio", None),
        "text_alignment": str(old.get("text_alignment") or "center"),
        "layout_bbox": list(old.get("layout_bbox") or []),
        "layout_box_mode": str(old.get("layout_box_mode") or "auto"),
        "rotation_degrees": float(old.get("rotation_degrees") or 0.0),
        "fill_color": old.get("fill_color", None),
        "box_locked": True,
        "manual_override": True,
        "created_at": str(old.get("created_at") or _now()),
        "updated_at": _now(),
    }
    return item


__all__ = [
    "OCR_EDIT_MODE_SCOPE", "is_ocr_edit_mode", "ocr_edit_scope", "ocr_edit_dir",
    "ocr_blocks_path", "load_ocr_blocks", "save_ocr_blocks", "upsert_ocr_block",
    "delete_ocr_block", "recognize_manual_ocr_block",
]
