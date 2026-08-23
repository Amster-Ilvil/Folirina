from __future__ import annotations

"""Neutral conflict policy for durable manual OCR blocks.

A manual OCR rectangle is a locator for one text authority.  Re-drawing the
same balloon/textbox must update/replace that authority, never stack another
full text rendering on top of it.  This module deliberately contains no mode
renderer imports so Direct/Mask/Hybrid/Reletter/Reveal remain isolated.
"""

from datetime import datetime
from typing import Any, Callable, Iterable


_CLOSED_LAYOUT_KINDS = {"bubble", "textbox"}


def _bbox(value: Any) -> list[int]:
    try:
        vals = [int(round(float(v))) for v in list(value or [])]
    except Exception:
        return []
    if len(vals) != 4:
        return []
    x0, y0, x1, y1 = vals
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
    if x1 <= x0 or y1 <= y0:
        return []
    return [x0, y0, x1, y1]


def bbox_overlap_metrics(a: Any, b: Any) -> tuple[float, float]:
    """Return (IoU, intersection/min-area overlap)."""
    aa = _bbox(a); bb = _bbox(b)
    if not aa or not bb:
        return 0.0, 0.0
    ax0, ay0, ax1, ay1 = aa; bx0, by0, bx1, by1 = bb
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0); area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    iou = float(inter / union) if union > 0 else 0.0
    overlap_min = float(inter / min(area_a, area_b)) if min(area_a, area_b) > 0 else 0.0
    return iou, overlap_min


def ocr_bbox_conflict(a: dict[str, Any], b: dict[str, Any], *, iou_threshold: float = 0.52,
                      containment_threshold: float = 0.80) -> bool:
    iou, overlap_min = bbox_overlap_metrics(a.get("target_bbox"), b.get("target_bbox"))
    return bool(iou >= iou_threshold or overlap_min >= containment_threshold)


def _normalized_text(value: Any) -> str:
    return "".join(str(value or "").split())


def has_explicit_text_edit(row: dict[str, Any]) -> bool:
    render = _normalized_text(row.get("render_text"))
    ocr = _normalized_text(row.get("ocr_text"))
    return bool(render and ocr and render != ocr)


def _timestamp(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _rank(row: dict[str, Any], index: int) -> tuple[int, float, int]:
    # Legacy repair prefers a block whose OCR text was explicitly corrected by
    # the user.  Among equal authorities the latest update wins.  New explicit
    # saves bypass this ranking by removing conflicts before append.
    return (
        1 if has_explicit_text_edit(row) else 0,
        max(_timestamp(row.get("updated_at")), _timestamp(row.get("created_at"))),
        int(index),
    )


def canonicalize_ocr_blocks(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse legacy near-duplicate OCR ROIs into one active authority.

    Conflict clusters are resolved deterministically.  A manually corrected
    text block outranks an unedited OCR retry; otherwise the newest block wins.
    Non-overlapping blocks are preserved.
    """
    items = [dict(row) for row in rows if isinstance(row, dict)]
    order = sorted(range(len(items)), key=lambda i: _rank(items[i], i), reverse=True)
    kept: list[int] = []
    suppressed: list[dict[str, Any]] = []
    for idx in order:
        row = items[idx]
        winner_idx = next((k for k in kept if ocr_bbox_conflict(row, items[k])), None)
        if winner_idx is None:
            kept.append(idx)
            continue
        suppressed.append({
            "id": str(row.get("id") or ""),
            "winner_id": str(items[winner_idx].get("id") or ""),
            "reason": "overlapping_manual_ocr_roi",
        })
    active = [items[i] for i in sorted(kept)]
    return active, suppressed


def conflicting_block_ids(rows: Iterable[dict[str, Any]], item: dict[str, Any]) -> list[str]:
    out: list[str] = []
    item_id = str(item.get("id") or "")
    for row in rows:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id") or "")
        if rid and rid == item_id:
            continue
        if ocr_bbox_conflict(row, item):
            out.append(rid)
    return out


def resolve_render_container_conflicts(
    rows: Iterable[dict[str, Any]],
    target: Any,
    recover_layout_region: Callable[..., tuple[Any, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Second-line renderer guard for multiple ROIs targeting one closed container.

    Two locator rectangles may not strongly overlap yet both recover to the same
    speech balloon/textbox.  Such rows cannot independently own the full
    container layout.  Keep a single authority using the same legacy ranking.
    Open regions remain independent unless their locator rectangles themselves
    conflict.
    """
    items = [dict(row) for row in rows if isinstance(row, dict)]
    layouts: list[dict[str, Any]] = []
    for row in items:
        box = _bbox(row.get("target_bbox"))
        diag: dict[str, Any] = {}
        if box:
            try:
                _, raw = recover_layout_region(target, box, open_inset=2)
                if isinstance(raw, dict):
                    diag = dict(raw)
            except Exception:
                diag = {}
        layouts.append(diag)

    def conflict(i: int, j: int) -> bool:
        if ocr_bbox_conflict(items[i], items[j]):
            return True
        li, lj = layouts[i], layouts[j]
        ki = str(li.get("layout_kind") or ""); kj = str(lj.get("layout_kind") or "")
        if ki not in _CLOSED_LAYOUT_KINDS or kj not in _CLOSED_LAYOUT_KINDS:
            return False
        iou, overlap_min = bbox_overlap_metrics(li.get("container_bbox"), lj.get("container_bbox"))
        return bool(iou >= 0.72 or overlap_min >= 0.90)

    order = sorted(range(len(items)), key=lambda i: _rank(items[i], i), reverse=True)
    kept: list[int] = []
    suppressed: list[dict[str, Any]] = []
    for idx in order:
        winner_idx = next((k for k in kept if conflict(idx, k)), None)
        if winner_idx is None:
            kept.append(idx)
            continue
        suppressed.append({
            "id": str(items[idx].get("id") or ""),
            "winner_id": str(items[winner_idx].get("id") or ""),
            "reason": "same_ocr_layout_container",
            "layout_kind": str(layouts[winner_idx].get("layout_kind") or ""),
            "container_bbox": list(layouts[winner_idx].get("container_bbox") or []),
        })
    active = [items[i] for i in sorted(kept)]
    return active, suppressed


__all__ = [
    "bbox_overlap_metrics", "ocr_bbox_conflict", "has_explicit_text_edit",
    "canonicalize_ocr_blocks", "conflicting_block_ids",
    "resolve_render_container_conflicts",
]
