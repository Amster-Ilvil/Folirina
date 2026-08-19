from __future__ import annotations

import numpy as np

def remaining_paired_bubbles(source_bubbles, target_bubbles, handled_target_ids: set[str]):
    """Preserve explicit paired identity while excluding already-rendered regions."""
    target_by_id = {b.id: b for b in target_bubbles}
    out_s, out_t = [], []
    for sb in source_bubbles:
        tid = str(sb.meta.get("paired_target_id") or "")
        tb = target_by_id.get(tid) if tid else None
        if tb is not None and tb.id not in handled_target_ids:
            out_s.append(sb); out_t.append(tb)
    if not out_s and not handled_target_ids and len(source_bubbles) == len(target_bubbles):
        return list(source_bubbles), list(target_bubbles)
    return out_s, out_t


def mask_transfer_completion_needed(mask_transfer) -> bool:
    records = list(getattr(mask_transfer, "records", []) or []) if mask_transfer is not None else []
    if not records:
        return True
    return any((not bool(getattr(r, "applied", False))) or bool(getattr(r, "review_required", False)) for r in records)


def completion_existing_target_bubbles(mask_transfer, *candidate_groups):
    records = list(getattr(mask_transfer, "records", []) or []) if mask_transfer is not None else []
    handled_ids = {
        str(getattr(r, "target_bubble_id", ""))
        for r in records
        if bool(getattr(r, "applied", False))
        and not bool(getattr(r, "review_required", False))
        and bool(getattr(r, "content_complete", True))
        and str(getattr(r, "target_bubble_id", ""))
    }
    if not handled_ids:
        return []
    existing = []
    seen = set()
    for group in candidate_groups:
        for bubble in list(group or []):
            bid = str(getattr(bubble, "id", ""))
            if not bid or bid not in handled_ids or bid in seen:
                continue
            existing.append(bubble)
            seen.add(bid)
    return existing


def completion_review_regions(mask_transfer):
    records = list(getattr(mask_transfer, "records", []) or []) if mask_transfer is not None else []
    boxes: list[tuple[int, int, int, int]] = []
    for rec in records:
        box = getattr(rec, "target_bbox", None)
        if not box or len(box) != 4:
            continue
        if bool(getattr(rec, "review_required", False)) or not bool(getattr(rec, "content_complete", True)) or not bool(getattr(rec, "applied", False)):
            boxes.append(tuple(int(v) for v in box))
    return boxes


def bbox_tuple_from_bubble(bubble):
    mask = getattr(bubble, "mask", None)
    if mask is not None and getattr(mask, "size", 0):
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    poly = getattr(bubble, "polygon", None) or []
    if poly:
        xs = [int(p[0]) for p in poly]; ys = [int(p[1]) for p in poly]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def bbox_intersection_over_smaller(a, b) -> float:
    ax0, ay0, ax1, ay1 = map(int, a); bx0, by0, bx1, by1 = map(int, b)
    ix0=max(ax0,bx0); iy0=max(ay0,by0); ix1=min(ax1,bx1); iy1=min(ay1,by1)
    inter=max(0,ix1-ix0)*max(0,iy1-iy0)
    aa=max(1,(ax1-ax0)*(ay1-ay0)); ba=max(1,(bx1-bx0)*(by1-by0))
    return float(inter / max(1, min(aa, ba)))


def filter_uncovered_white_completion_pairs(source_bubbles, target_bubbles, existing_boxes, config=None, *, overlap_threshold: float = 0.28):
    """Keep ordinary unseeded white containers that are not already represented.

    v1.0.6 removes the old publication-only 0.84/0.16 paper thresholds.  Those
    values silently discarded perfectly valid small speech balloons with dense
    CJK glyphs (the real page-45 round balloon measures about 0.81 white ratio).
    Detection/pairing already proved white-container geometry and registered ink
    change, so this stage now uses the detector's own permissive limits and only
    keeps the duplicate/coloured-region exclusions.
    """
    kept_s, kept_t = [], []
    existing = [tuple(map(int, b)) for b in list(existing_boxes or []) if b and len(b) == 4]
    min_white = float(getattr(config, "rigid_container_unseeded_min_white_ratio", 0.55)) if config is not None else 0.55
    max_dark = float(getattr(config, "rigid_container_unseeded_max_dark_ratio", 0.28)) if config is not None else 0.28
    # Dense translated CJK can legitimately lower paper ratio; never make this
    # completion gate stricter than the detector that produced the candidate.
    min_white = min(min_white, 0.55)
    max_dark = max(max_dark, 0.28)
    for sb, tb in zip(list(source_bubbles or []), list(target_bubbles or [])):
        meta = dict(getattr(tb, "meta", {}) or {})
        if str(meta.get("backend", "")) != "unseeded_white":
            continue
        if bool(meta.get("target_colored_recovery", False)) or bool(meta.get("target_driven_colored", False)):
            continue
        white = float(meta.get("white_ratio", 1.0) or 0.0)
        dark = float(meta.get("dark_ratio", 0.0) or 0.0)
        sat_median = float(meta.get("saturation_median", 0.0) or 0.0)
        sat_p75 = float(meta.get("saturation_p75", 0.0) or 0.0)
        if white < min_white or dark > max_dark:
            continue
        # This is route classification, not publication gating: a light purple/
        # pink burst may contain a large high-value component that the white
        # detector sees as paper.  Rigid white completion would paste a white
        # island over the coloured TARGET.  Keep only genuinely neutral paper;
        # coloured regions stay on the target-aware component/Reveal route.
        if sat_median > 12.0 or sat_p75 > 24.0:
            continue
        box = bbox_tuple_from_bubble(tb)
        if box is None:
            continue
        if any(bbox_intersection_over_smaller(box, old) >= float(overlap_threshold) for old in existing):
            continue
        kept_s.append(sb); kept_t.append(tb)
    return kept_s, kept_t


def completion_filter_pairs_to_review_regions(source_bubbles, target_bubbles, review_boxes, config=None):
    if not source_bubbles or not target_bubbles:
        return source_bubbles, target_bubbles
    kept_src = []
    kept_dst = []
    max_aspect = float(getattr(config, "rigid_container_unseeded_completion_max_aspect", 5.0)) if config is not None else 5.0
    for sb, tb in zip(source_bubbles, target_bubbles):
        tbox = None
        tmask = getattr(tb, "mask", None)
        if tmask is not None and getattr(tmask, 'size', 0):
            ys, xs = np.where(tmask > 0)
            if len(xs) > 0:
                tbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if tbox is None:
            poly = getattr(tb, "polygon", None) or []
            if poly:
                xs = [int(p[0]) for p in poly]; ys = [int(p[1]) for p in poly]
                tbox = (min(xs), min(ys), max(xs), max(ys))
        if tbox is None:
            continue
        tx0, ty0, tx1, ty1 = tbox
        tw = max(1, tx1 - tx0); th = max(1, ty1 - ty0)
        aspect = max(tw / max(1.0, th), th / max(1.0, tw))
        keep = aspect <= max_aspect
        if review_boxes:
            for rx0, ry0, rx1, ry1 in review_boxes:
                ix0 = max(tx0, rx0); iy0 = max(ty0, ry0); ix1 = min(tx1, rx1); iy1 = min(ty1, ry1)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                if inter > 0:
                    keep = True
                    break
                cx = (rx0 + rx1) // 2; cy = (ry0 + ry1) // 2
                if tx0 <= cx <= tx1 and ty0 <= cy <= ty1:
                    keep = True
                    break
        if keep:
            kept_src.append(sb)
            kept_dst.append(tb)
    return kept_src, kept_dst

__all__ = [
    "remaining_paired_bubbles", "mask_transfer_completion_needed",
    "completion_existing_target_bubbles", "completion_review_regions",
    "bbox_tuple_from_bubble", "bbox_intersection_over_smaller",
    "filter_uncovered_white_completion_pairs", "completion_filter_pairs_to_review_regions",
]
