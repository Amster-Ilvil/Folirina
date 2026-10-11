from __future__ import annotations

"""Semantic authority boundary for destructive transfer completion.

This module owns only evidence collection/classification.  It never renders
pixels.  That makes the selected primary detector's authority explicit and
prevents auxiliary detectors from silently gaining destructive ownership.
"""

from dataclasses import dataclass
from typing import Any

from .detector_policy import koharu_is_primary, primary_detector
from .layout_evidence import collect_koharu_layout_evidence_cached, classify_layout_authority
from .layout_evidence_models import LayoutEvidence, LayoutEvidenceItem
from .geometry import rasterize_polygon
from .models import BubbleInstance

import cv2
import numpy as np


def _safe_rect_mask(shape: tuple[int, int], box: Any) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return mask
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    except Exception:
        return mask
    h, w = shape
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return mask
    mask[y0:y1, x0:x1] = 255
    return mask


def _translated_target_text_preserve_mask(result, target: np.ndarray) -> np.ndarray:
    """Return TARGET text pixels that must survive the final PROTECT guard.

    Direct/semantic transfer may intentionally replace Japanese inside a valid
    translation container.  The page-level Koharu PROTECT guard still needs to
    restore surrounding artwork, but it must not resurrect TARGET text inside
    those already accepted translation regions.  Build a compact TARGET-text mask
    from each applied record's target region only; border/rule structure remains
    restorable outside that text-shaped exclusion.
    """
    shape = target.shape[:2]
    out = np.zeros(shape, np.uint8)
    records = list(getattr(result, 'records', []) or [])
    if not records:
        return out
    try:
        from .modes.direct_patch.text_transfer import (
            target_text_mask_in_container,
            target_white_container_text_mask,
        )
    except Exception:
        return out
    for row in records:
        if not bool(getattr(row, 'applied', False)):
            continue
        region = _safe_rect_mask(shape, getattr(row, 'target_bbox', None))
        if cv2.countNonZero(region) <= 0:
            continue
        try:
            compact = target_text_mask_in_container(target, region)
            white = target_white_container_text_mask(target, region)
            text_mask = cv2.bitwise_or(compact, white)
        except Exception:
            continue
        if text_mask.shape != shape:
            try:
                text_mask = cv2.resize(text_mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            except Exception:
                continue
        text_mask = cv2.bitwise_and(text_mask.astype(np.uint8), region)
        if cv2.countNonZero(text_mask) <= 0:
            continue
        out = np.maximum(out, text_mask)
    return out


def _translated_source_ink_preserve_mask(result, target: np.ndarray) -> np.ndarray:
    """Return newly written SOURCE ink that TARGET PROTECT must never erase.

    ``_translated_target_text_preserve_mask`` protects the *cleared* TARGET glyph
    positions, but translated glyphs rarely occupy exactly the same pixels.  A
    coarse PROTECT map can therefore restore white TARGET paper over the new CN
    strokes even though the Direct record itself was accepted (Vol.04 p0048).

    Admit only pixels inside an applied record's target bbox where the rendered
    candidate is materially darker than TARGET.  Existing dark TARGET structure
    is deliberately excluded, so panel/border/art restoration keeps ownership.
    """
    shape = target.shape[:2]
    out = np.zeros(shape, np.uint8)
    image = getattr(result, "image", None)
    if not isinstance(image, np.ndarray) or image.shape != target.shape:
        return out
    records = list(getattr(result, "records", []) or [])
    if not records:
        return out
    if image.ndim == 3:
        cand = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tgt = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    else:
        cand = image.astype(np.uint8)
        tgt = target.astype(np.uint8)
    cand_i = cand.astype(np.int16)
    tgt_i = tgt.astype(np.int16)
    darker = tgt_i - cand_i
    for row in records:
        if not bool(getattr(row, "applied", False)):
            continue
        region = _safe_rect_mask(shape, getattr(row, "target_bbox", None)) > 0
        if not np.any(region):
            continue
        # Strong core: genuinely new dark ink on a lighter TARGET pixel.
        core = region & (cand_i <= 210) & (darker >= 12)
        if not np.any(core):
            continue
        core_u8 = core.astype(np.uint8) * 255
        support = cv2.dilate(
            core_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
        ) > 0
        # Keep anti-aliased stroke support, but never claim a pixel whose TARGET
        # was already equally dark: that remains structural PROTECT authority.
        support &= region & (cand_i <= 238) & (darker >= 3)
        out[support] = 255
    return out


def _box_overlap_over_min(a, b) -> float:
    if not isinstance(a, (list, tuple)) or len(a) != 4 or not isinstance(b, (list, tuple)) or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = max(1.0, max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0))
    bb = max(1.0, max(0.0, bx1 - bx0) * max(0.0, by1 - by0))
    return float(inter / max(1.0, min(aa, bb)))


def _project_polygon_to_target(polygon, registration, target_shape):
    if registration is None or not polygon:
        return None
    try:
        matrix = np.asarray(getattr(registration, "matrix", None), dtype=np.float64)
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
        if matrix.shape == (3, 3):
            mapped = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
        elif matrix.shape == (2, 3):
            mapped = cv2.transform(pts, matrix).reshape(-1, 2)
        else:
            return None
        if mapped.size == 0 or not np.all(np.isfinite(mapped)):
            return None
        h, w = target_shape
        poly = [(float(x), float(y)) for x, y in mapped]
        x0, y0 = mapped.min(axis=0); x1, y1 = mapped.max(axis=0)
        # Reject projections that are almost entirely off-page.
        cx0, cy0 = max(0.0, float(x0)), max(0.0, float(y0))
        cx1, cy1 = min(float(w), float(x1)), min(float(h), float(y1))
        full = max(1.0, float(x1 - x0) * float(y1 - y0))
        visible = max(0.0, cx1 - cx0) * max(0.0, cy1 - cy0)
        if visible / full < 0.70:
            return None
        mask = rasterize_polygon(poly, target_shape)
        if cv2.countNonZero(mask) <= 0:
            return None
        ys, xs = np.where(mask > 0)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        return poly, mask, box
    except Exception:
        return None


def _source_text_children(source_evidence, bubble_row, *, text_min: float):
    out = []
    bm = getattr(bubble_row, "mask", None)
    if not isinstance(bm, np.ndarray) or cv2.countNonZero(bm) <= 0:
        return out
    for row in list(getattr(source_evidence, "items", []) or []):
        if str(getattr(row, "label", "")).strip().lower() != "text":
            continue
        if float(getattr(row, "confidence", 0.0)) < float(text_min):
            continue
        tm = getattr(row, "mask", None)
        if not isinstance(tm, np.ndarray) or tm.shape != bm.shape:
            continue
        area = max(1, int(cv2.countNonZero(tm)))
        overlap = float(np.count_nonzero((tm > 0) & (bm > 0)) / area)
        if overlap >= 0.35:
            out.append(row)
    return out


def augment_target_authority_from_registered_source_gaps(
    source_evidence, target_evidence, registration, target_shape, cfg, *, stats: dict | None = None,
):
    """Recover TARGET-side Koharu misses from strong registered SOURCE evidence.

    This is intentionally narrow: only SOURCE bubbles that contain SOURCE text
    may synthesize TARGET authority, and only when no TARGET bubble overlaps the
    registered location.  SFX-only regions are never promoted.  The renderer
    still performs its normal changed-ink/content checks before writing pixels.
    """
    if (source_evidence is None or target_evidence is None
            or not bool(getattr(source_evidence, "available", False))
            or not bool(getattr(target_evidence, "available", False))
            or registration is None):
        return target_evidence
    reg_conf = float(getattr(registration, "confidence", 0.0) or 0.0)
    if reg_conf < 0.72:
        return target_evidence
    bubble_min = float(getattr(cfg, "koharu_layout_bubble_threshold", 0.50))
    text_min = float(getattr(cfg, "koharu_layout_text_threshold", 0.25))
    existing = list(getattr(target_evidence, "items", []) or [])
    target_bubbles = [x for x in existing if str(getattr(x, "label", "")) == "bubble"]
    target_texts = [x for x in existing if str(getattr(x, "label", "")) == "text"]
    added: list[LayoutEvidenceItem] = []
    recovered_bubbles = 0
    recovered_texts = 0
    for src in list(getattr(source_evidence, "items", []) or []):
        if str(getattr(src, "label", "")) != "bubble" or float(getattr(src, "confidence", 0.0)) < bubble_min:
            continue
        children = _source_text_children(source_evidence, src, text_min=text_min)
        if not children:
            continue
        projected = _project_polygon_to_target(getattr(src, "polygon", None), registration, target_shape)
        if projected is None:
            continue
        poly, mask, box = projected
        if any(_box_overlap_over_min(box, getattr(t, "box", None)) >= 0.35 for t in target_bubbles):
            continue
        idx = len(existing) + len(added)
        conf = min(float(getattr(src, "confidence", 0.0)), reg_conf)
        meta = dict(getattr(src, "meta", {}) or {})
        meta.update({
            "item_index": idx,
            "projected_from_source": True,
            "authority_source": "registered_source_gap_recovery",
            "source_item_index": (getattr(src, "meta", {}) or {}).get("item_index"),
        })
        synthetic = LayoutEvidenceItem("bubble", conf, poly, mask, box, meta)
        added.append(synthetic); target_bubbles.append(synthetic); recovered_bubbles += 1
        # Carry only SOURCE text children of the recovered bubble.  This gives
        # semantic completion a text-bearing proof without promoting SFX.
        for child in children:
            pchild = _project_polygon_to_target(getattr(child, "polygon", None), registration, target_shape)
            if pchild is None:
                continue
            cpoly, cmask, cbox = pchild
            if any(_box_overlap_over_min(cbox, getattr(t, "box", None)) >= 0.35 for t in target_texts):
                continue
            cidx = len(existing) + len(added)
            cmeta = dict(getattr(child, "meta", {}) or {})
            cmeta.update({
                "item_index": cidx,
                "projected_from_source": True,
                "authority_source": "registered_source_gap_recovery",
                "source_item_index": (getattr(child, "meta", {}) or {}).get("item_index"),
            })
            cconf = min(float(getattr(child, "confidence", 0.0)), reg_conf)
            synth_text = LayoutEvidenceItem("text", cconf, cpoly, cmask, cbox, cmeta)
            added.append(synth_text); target_texts.append(synth_text); recovered_texts += 1
    if not added:
        return target_evidence
    diagnostics = dict(getattr(target_evidence, "diagnostics", {}) or {})
    diagnostics["registered_source_gap_recovery"] = {
        "enabled": True, "bubbles": recovered_bubbles, "texts": recovered_texts,
        "registration_confidence": round(reg_conf, 4),
    }
    if stats is not None:
        stats["koharu_projected_source_gap_bubbles"] = str(recovered_bubbles)
        stats["koharu_projected_source_gap_texts"] = str(recovered_texts)
    return LayoutEvidence(True, str(getattr(target_evidence, "backend", "koharu_layout")), existing + added, diagnostics)


@dataclass(frozen=True)
class TransferLayoutAuthorityState:
    source: Any | None
    target: Any | None
    status: str


def collect_transfer_layout_authority(
    *, config, source, target, stage_cache=None, cache_stats: dict | None = None,
    source_path: str | None = None, target_path: str | None = None, registration=None,
) -> TransferLayoutAuthorityState:
    """Collect two-sided Koharu authority only when Koharu is the selected primary.

    Missing evidence stays fail-open for completion candidates, but an auxiliary
    Koharu detector never outranks the user's selected primary detector.
    """
    if not koharu_is_primary(config.bubbles):
        status = f"skipped_primary:{primary_detector(config.bubbles)}"
        if cache_stats is not None:
            cache_stats["transfer_layout_authority"] = status
        return TransferLayoutAuthorityState(None, None, status)

    layout_cache_enabled = bool(getattr(config.cache, "bubbles", True)) and bool(
        getattr(config.bubbles, "koharu_layout_cache_enabled", True)
    )
    source_evidence = collect_koharu_layout_evidence_cached(
        source, config.bubbles, role="transfer_authority_source", image_path=source_path,
        cache=stage_cache, cache_enabled=layout_cache_enabled, stats=cache_stats, allow_missing=True,
    )
    target_evidence = collect_koharu_layout_evidence_cached(
        target, config.bubbles, role="transfer_authority_target", image_path=target_path,
        cache=stage_cache, cache_enabled=layout_cache_enabled, stats=cache_stats, allow_missing=True,
    )
    target_evidence = augment_target_authority_from_registered_source_gaps(
        source_evidence, target_evidence, registration, target.shape[:2], config.bubbles, stats=cache_stats,
    )
    status = (
        "two_sided" if source_evidence.available and target_evidence.available
        else "partial_fail_open" if source_evidence.available or target_evidence.available
        else "unavailable_fail_open"
    )
    if cache_stats is not None:
        cache_stats["transfer_layout_authority"] = status
    return TransferLayoutAuthorityState(source_evidence, target_evidence, status)


def filter_completion_pairs_by_layout_authority(
    source_rows: list[BubbleInstance],
    target_rows: list[BubbleInstance],
    *,
    source_evidence,
    target_evidence,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    cfg,
) -> tuple[list[BubbleInstance], list[BubbleInstance], list[dict]]:
    """Reject completion pairs positively classified as panel/artwork.

    UNKNOWN remains eligible because completion candidates are already
    conservative; PROTECT never reaches a destructive renderer.
    """
    kept_s: list[BubbleInstance] = []
    kept_t: list[BubbleInstance] = []
    audit: list[dict] = []
    for sb, tb in zip(source_rows or [], target_rows or []):
        sd = classify_layout_authority(source_evidence, sb, source_shape, region_kind="bubble", cfg=cfg)
        td = classify_layout_authority(target_evidence, tb, target_shape, region_kind="bubble", cfg=cfg)
        passed = sd.state != "PROTECT" and td.state != "PROTECT"
        decision = {
            "source_id": str(getattr(sb, "id", "")),
            "target_id": str(getattr(tb, "id", "")),
            "passed": bool(passed),
            "source": sd.to_dict(),
            "target": td.to_dict(),
            "reason": "koharu_authority_completion_allowed" if passed else "koharu_layout_panel_only_artwork",
        }
        for row, side in ((sb, sd), (tb, td)):
            meta = dict(getattr(row, "meta", {}) or {})
            meta["koharu_layout_authority"] = side.to_dict()
            row.meta = meta
        audit.append(decision)
        if passed:
            kept_s.append(sb)
            kept_t.append(tb)
    return kept_s, kept_t, audit


def apply_target_koharu_protect_guard(
    result,
    target: np.ndarray,
    evidence,
    *,
    allow_dilate_px: int = 12,
) -> dict[str, int | bool]:
    """Hard-restore TARGET pixels in Koharu PROTECT areas.

    Koharu is the first semantic authority when selected as the primary detector.
    Auxiliary geometry and raster heuristics may operate in UNKNOWN, but they are
    never allowed to overwrite pixels that Koharu classified as panel/artwork-only.
    ALLOW (bubble/text/SFX) wins over panel protection through ``authority_map``.

    This guard is deliberately applied *after* pixel rendering. It therefore also
    catches write-mask spill, dilation and local alignment side-effects that passed
    the earlier candidate-level authority checks.
    """
    diag: dict[str, int | bool] = {
        "enabled": bool(evidence is not None and getattr(evidence, "available", False)),
        "protect_pixels": 0,
        "restored_pixels": 0,
        "mask_pixels_removed": 0,
    }
    if not diag["enabled"] or result is None:
        return diag
    try:
        amap = evidence.authority_map(allow_dilate_px=max(0, int(allow_dilate_px)))
    except Exception:
        return diag
    if amap.shape != target.shape[:2]:
        amap = cv2.resize(amap.astype(np.uint8), (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST)
    protect = amap == 2
    explicit_allow = getattr(result, "authority_allow_mask", None)
    if isinstance(explicit_allow, np.ndarray) and explicit_allow.shape == protect.shape:
        allow_pixels = explicit_allow > 0
        diag["explicit_semantic_allow_pixels"] = int(np.count_nonzero(allow_pixels))
        protect = protect & (~allow_pixels)
    else:
        diag["explicit_semantic_allow_pixels"] = 0

    translated_text = _translated_target_text_preserve_mask(result, target)
    if translated_text.shape == protect.shape and cv2.countNonZero(translated_text) > 0:
        diag["translated_target_text_excluded_pixels"] = int(cv2.countNonZero(translated_text))
        protect = protect & (~(translated_text > 0))
    else:
        diag["translated_target_text_excluded_pixels"] = 0

    translated_source_ink = _translated_source_ink_preserve_mask(result, target)
    if translated_source_ink.shape == protect.shape and cv2.countNonZero(translated_source_ink) > 0:
        diag["translated_source_ink_excluded_pixels"] = int(cv2.countNonZero(translated_source_ink))
        protect = protect & (~(translated_source_ink > 0))
    else:
        diag["translated_source_ink_excluded_pixels"] = 0

    # Page-furniture text is semantically ALLOW in the generic authority map, but
    # it is not a translation container.  A low-confidence running title/page
    # number can otherwise be pulled into paired-diff masks and rewritten even
    # though every dialogue bubble was detected correctly.  Protect standalone
    # text near the page edges when it does not belong to any Koharu bubble.
    furniture = np.zeros(target.shape[:2], np.uint8)
    try:
        bubble_mask = evidence.combined_mask(("bubble",), dilate_px=2)
        h, w = target.shape[:2]
        for row in list(getattr(evidence, "items", []) or []):
            if str(getattr(row, "label", "")).strip().lower() != "text":
                continue
            box = getattr(row, "box", None)
            mask = getattr(row, "mask", None)
            if not isinstance(box, (list, tuple)) or len(box) != 4 or not isinstance(mask, np.ndarray):
                continue
            x0, y0, x1, y1 = [int(round(float(v))) for v in box]
            near_edge = bool(y1 <= int(round(h * 0.08)) or y0 >= int(round(h * 0.94)))
            if not near_edge:
                continue
            use = mask if mask.shape == furniture.shape else cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            area = max(1, int(cv2.countNonZero(use)))
            bubble_overlap = float(np.count_nonzero((use > 0) & (bubble_mask > 0)) / area)
            if bubble_overlap >= 0.25:
                continue
            furniture = np.maximum(furniture, use)
        if cv2.countNonZero(furniture) > 0:
            furniture = cv2.dilate(furniture, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
            protect = protect | (furniture > 0)
            diag["page_furniture_protect_pixels"] = int(cv2.countNonZero(furniture))
        else:
            diag["page_furniture_protect_pixels"] = 0
    except Exception:
        diag["page_furniture_protect_pixels"] = 0

    diag["protect_pixels"] = int(np.count_nonzero(protect))
    if not np.any(protect):
        return diag

    image = getattr(result, "image", None)
    if isinstance(image, np.ndarray) and image.shape == target.shape:
        before = np.any(image != target, axis=2) if image.ndim == 3 else (image != target)
        diag["restored_pixels"] = int(np.count_nonzero(before & protect))
        image[protect] = target[protect]

    for name in ("composite_mask", "clear_mask"):
        mask = getattr(result, name, None)
        if isinstance(mask, np.ndarray) and mask.shape[:2] == protect.shape:
            before_n = int(np.count_nonzero(mask[protect]))
            if before_n:
                mask[protect] = 0
                diag["mask_pixels_removed"] = int(diag["mask_pixels_removed"]) + before_n

    layer = getattr(result, "layer_rgba", None)
    if isinstance(layer, np.ndarray) and layer.shape[:2] == protect.shape:
        if layer.ndim == 3 and layer.shape[2] >= 4:
            layer[protect, :] = 0
        else:
            layer[protect] = 0

    for row in list(getattr(result, "records", []) or []):
        try:
            meta = dict(getattr(row, "meta", {}) or {})
            meta["koharu_target_protect_guard"] = dict(diag)
            row.meta = meta
        except Exception:
            pass
    return diag


def apply_target_koharu_page_furniture_guard(
    result,
    target: np.ndarray,
    evidence,
    *,
    top_ratio: float = 0.10,
    bottom_ratio: float = 0.965,
    max_bubble_overlap: float = 0.18,
    max_panel_overlap: float = 0.30,
    dilate_px: int = 2,
) -> dict[str, int | bool]:
    """Restore Koharu text/SFX that is clearly page furniture, not dialogue.

    Koharu correctly labels running titles/page numbers as ``text``.  That label
    is useful detection evidence, but it must not automatically grant destructive
    dialogue authority.  A text/SFX item near the top/bottom page margin is
    protected when it has little overlap with any Koharu bubble.  This keeps the
    detector semantic-first while preventing bubble-completion masks from
    accidentally eating headers such as ``桜通信3``.
    """
    diag: dict[str, int | bool] = {
        "enabled": bool(evidence is not None and getattr(evidence, "available", False)),
        "candidate_items": 0,
        "protected_items": 0,
        "protect_pixels": 0,
        "restored_pixels": 0,
        "mask_pixels_removed": 0,
    }
    if not diag["enabled"] or result is None:
        return diag
    h, w = target.shape[:2]
    try:
        bubbles = evidence.combined_mask(("bubble",), dilate_px=2) > 0
        panels = evidence.combined_mask(("panel",), dilate_px=2) > 0
        rows = list(evidence.by_label("text", "sfx"))
    except Exception:
        return diag
    protect = np.zeros((h, w), np.uint8)
    explicit_allow = getattr(result, "authority_allow_mask", None)
    if isinstance(explicit_allow, np.ndarray) and explicit_allow.shape == (h, w):
        explicit_allow = explicit_allow > 0
    else:
        explicit_allow = None
    translated_text = _translated_target_text_preserve_mask(result, target)
    translated_text_sel = translated_text > 0 if isinstance(translated_text, np.ndarray) and translated_text.shape == (h, w) else None
    translated_source_ink = _translated_source_ink_preserve_mask(result, target)
    translated_source_ink_sel = translated_source_ink > 0 if isinstance(translated_source_ink, np.ndarray) and translated_source_ink.shape == (h, w) else None
    # The page margin itself is structural furniture when Koharu says it is
    # outside both panels and bubbles.  This catches running titles/page numbers
    # even when the layout model does not emit a dedicated text instance for
    # them, while still allowing a dialogue bubble or panel that legitimately
    # touches the top/bottom edge.
    margin = np.zeros((h, w), np.bool_)
    margin[: max(0, min(h, int(round(h * float(top_ratio))))), :] = True
    margin[max(0, min(h, int(round(h * float(bottom_ratio))))):, :] = True
    structural_margin = margin & (~bubbles) & (~panels)
    protect[structural_margin] = 255
    for row in rows:
        mask = np.asarray(getattr(row, "mask", np.zeros((h, w), np.uint8)))
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mb = mask > 0
        area = int(np.count_nonzero(mb))
        if area <= 0:
            continue
        diag["candidate_items"] = int(diag["candidate_items"]) + 1
        box = getattr(row, "box", None)
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            ys, xs = np.where(mb)
            if xs.size == 0:
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        else:
            x0, y0, x1, y1 = [int(v) for v in box]
        edge_item = bool(y1 <= h * float(top_ratio) or y0 >= h * float(bottom_ratio))
        if not edge_item:
            continue
        overlap = float(np.count_nonzero(mb & bubbles)) / max(1, area)
        if overlap > float(max_bubble_overlap):
            continue
        panel_overlap = float(np.count_nonzero(mb & panels)) / max(1, area)
        if panel_overlap > float(max_panel_overlap):
            continue
        protect[mb] = 255
        diag["protected_items"] = int(diag["protected_items"]) + 1
    radius = max(0, int(dilate_px))
    if radius > 0 and cv2.countNonZero(protect) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        protect = cv2.dilate(protect, k, iterations=1)
    if explicit_allow is not None:
        protect[explicit_allow] = 0
        diag["explicit_semantic_allow_pixels"] = int(np.count_nonzero(explicit_allow))
    else:
        diag["explicit_semantic_allow_pixels"] = 0
    if translated_text_sel is not None and np.any(translated_text_sel):
        protect[translated_text_sel] = 0
        diag["translated_target_text_excluded_pixels"] = int(np.count_nonzero(translated_text_sel))
    else:
        diag["translated_target_text_excluded_pixels"] = 0
    if translated_source_ink_sel is not None and np.any(translated_source_ink_sel):
        protect[translated_source_ink_sel] = 0
        diag["translated_source_ink_excluded_pixels"] = int(np.count_nonzero(translated_source_ink_sel))
    else:
        diag["translated_source_ink_excluded_pixels"] = 0
    diag["protect_pixels"] = int(cv2.countNonZero(protect))
    if diag["protect_pixels"] <= 0:
        return diag
    sel = protect > 0
    image = getattr(result, "image", None)
    if isinstance(image, np.ndarray) and image.shape == target.shape:
        before = np.any(image != target, axis=2) if image.ndim == 3 else (image != target)
        diag["restored_pixels"] = int(np.count_nonzero(before & sel))
        image[sel] = target[sel]
    for name in ("composite_mask", "clear_mask"):
        mask = getattr(result, name, None)
        if isinstance(mask, np.ndarray) and mask.shape[:2] == sel.shape:
            n = int(np.count_nonzero(mask[sel]))
            if n:
                mask[sel] = 0
                diag["mask_pixels_removed"] = int(diag["mask_pixels_removed"]) + n
    layer = getattr(result, "layer_rgba", None)
    if isinstance(layer, np.ndarray) and layer.shape[:2] == sel.shape:
        layer[sel] = 0
    for row in list(getattr(result, "records", []) or []):
        try:
            meta = dict(getattr(row, "meta", {}) or {})
            meta["koharu_page_furniture_guard"] = dict(diag)
            row.meta = meta
        except Exception:
            pass
    return diag


# Compatibility name used by older tests/plugins.
_filter_completion_pairs_by_koharu_authority = filter_completion_pairs_by_layout_authority

__all__ = [
    "TransferLayoutAuthorityState",
    "collect_transfer_layout_authority",
    "filter_completion_pairs_by_layout_authority",
    "apply_target_koharu_protect_guard",
    "apply_target_koharu_page_furniture_guard",
    "_filter_completion_pairs_by_koharu_authority",
]
