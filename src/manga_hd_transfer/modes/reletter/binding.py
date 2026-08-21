from __future__ import annotations

import re

import cv2
import numpy as np

from ...matching import linear_sum_assignment
from ...models import BubbleInstance, TextBlock

def contains_cjk(text: str) -> bool:
    return any(
        "\u3400" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff"
        for ch in str(text or "")
    )


def infer_region_orientation(crop: np.ndarray, text: str, kind: str = "speech") -> tuple[str, dict]:
    """Infer vertical/horizontal layout from the *source glyph image*.

    VisionKit Live Text/Shortcuts return transcript only, so their synthetic
    TextBlock polygon is the whole balloon and cannot describe text direction.
    Use dark connected components in the already-isolated source balloon instead.
    Strong image evidence wins; ambiguous CJK speech defaults to vertical, which
    matches the dominant manga dialogue convention while still allowing obvious
    horizontal captions/dialogue to remain horizontal.
    """
    if crop.size == 0:
        return "vertical" if contains_cjk(text) and kind in {"speech", "narration", "unknown"} else "horizontal", {"reason": "empty_crop"}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    # Keep printed ink while ignoring pale paper/antialias noise.
    ink = (gray < 185).astype(np.uint8)
    h, w = ink.shape[:2]
    # Remove a tiny border where balloon/panel edges can survive the safe mask.
    border = max(1, int(round(min(h, w) * 0.025)))
    if h > border * 2 and w > border * 2:
        ink[:border, :] = 0; ink[-border:, :] = 0
        ink[:, :border] = 0; ink[:, -border:] = 0
    ys, xs = np.where(ink > 0)
    if len(xs) < 8:
        fallback = "vertical" if contains_cjk(text) and kind in {"speech", "narration", "unknown"} else "horizontal"
        return fallback, {"reason": "too_little_ink", "ink_pixels": int(len(xs))}

    ink_w = max(1, int(xs.max() - xs.min() + 1)); ink_h = max(1, int(ys.max() - ys.min() + 1))
    ink_aspect = float(ink_h / ink_w)

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(ink, 8)
    comps = []
    page_area = float(max(1, h * w))
    for idx in range(1, count):
        x, y, cw, ch, area = [int(v) for v in stats[idx]]
        # Reject dust and large artwork/borders. Individual CJK glyph components
        # are small relative to an isolated balloon crop.
        if area < max(3, int(page_area * 0.00008)):
            continue
        if area > page_area * 0.09 or cw > w * 0.55 or ch > h * 0.55:
            continue
        comps.append((float(centroids[idx][0]), float(centroids[idx][1]), cw, ch, area))

    vertical_neighbors = 0; horizontal_neighbors = 0
    if len(comps) >= 2:
        for i, (cx, cy, cw, ch, _area) in enumerate(comps):
            best = None
            for j, (dx0, dy0, dw, dh, _a2) in enumerate(comps):
                if i == j:
                    continue
                # Normalize displacement by local glyph scale so punctuation and
                # mixed-size CJK characters do not dominate the vote.
                sx = max(2.0, (cw + dw) * 0.5); sy = max(2.0, (ch + dh) * 0.5)
                dx = abs(dx0 - cx) / sx; dy = abs(dy0 - cy) / sy
                dist = dx * dx + dy * dy
                if best is None or dist < best[0]:
                    best = (dist, dx, dy)
            if best is None:
                continue
            _, dx, dy = best
            if dy > dx * 1.20:
                vertical_neighbors += 1
            elif dx > dy * 1.30:
                horizontal_neighbors += 1

    # Strong overall shape is the most stable clue. The neighbor vote resolves
    # roughly square multi-column bubbles (common in manga).
    if ink_aspect >= 1.18:
        orientation, reason = "vertical", "ink_bbox_tall"
    elif ink_aspect <= 0.66:
        orientation, reason = "horizontal", "ink_bbox_wide"
    elif vertical_neighbors >= max(2, int(horizontal_neighbors * 1.25 + 0.5)):
        orientation, reason = "vertical", "component_flow_vertical"
    elif horizontal_neighbors >= max(2, int(vertical_neighbors * 1.40 + 0.5)):
        orientation, reason = "horizontal", "component_flow_horizontal"
    elif contains_cjk(text) and kind in {"speech", "narration", "unknown"}:
        orientation, reason = "vertical", "ambiguous_cjk_manga_default"
    else:
        orientation, reason = ("vertical", "bubble_crop_tall") if h / max(1.0, w) >= 1.35 else ("horizontal", "bubble_crop_wide")
    return orientation, {
        "reason": reason,
        "ink_bbox_aspect": round(ink_aspect, 4),
        "component_count": len(comps),
        "vertical_neighbor_votes": vertical_neighbors,
        "horizontal_neighbor_votes": horizontal_neighbors,
    }


def source_layout_profile(crop: np.ndarray, text: str, orientation: str) -> dict:
    """Estimate source typography from isolated translated ink.

    The key invariant is that OCR contributes Unicode only.  Column count and glyph
    pitch come from the source image.  We intentionally avoid connected-component
    counts because CJK glyphs split into many stroke components; instead solve the
    approximate grid from ink-bbox aspect ratio and character count.
    """
    if crop.size == 0 or not str(text or "").strip():
        return {}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    h, w = gray.shape[:2]
    raw = (gray < 188).astype(np.uint8) * 255
    # Remove dust while retaining punctuation and thin antialiased strokes.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    kept = np.zeros_like(raw)
    area_total = float(max(1, h*w))
    for i in range(1, count):
        x, y, cw, ch, area = [int(v) for v in stats[i]]
        if area < max(2, int(area_total * 0.000015)):
            continue
        if area > area_total * 0.12 or cw > w * 0.72 or ch > h * 0.72:
            continue
        kept[labels == i] = 255
    ys, xs = np.where(kept > 0)
    if len(xs) < 10:
        return {}
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    ink_w, ink_h = max(1, x1-x0), max(1, y1-y0)
    chars = [ch for ch in str(text) if not ch.isspace()]
    n = max(1, len(chars))
    common = {
        "ink_bbox": [x0, y0, x1, y1],
        "ink_bbox_size": [ink_w, ink_h],
        "container_size": [int(w), int(h)],
        "fill_ratio": round(float((ink_w * ink_h) / max(1.0, w * h)), 4),
    }

    if orientation == "vertical":
        # For a roughly square glyph grid: ink_w/ink_h ≈ columns/rows and
        # rows ≈ n/columns, hence columns ≈ sqrt(n * aspect).
        aspect = float(ink_w / max(1.0, ink_h))
        columns = int(np.clip(round(np.sqrt(max(0.05, aspect) * n)), 1, 8))
        rows = max(1, int(np.ceil(n / columns)))
        pitch_h = ink_h / rows
        pitch_w = ink_w / columns
        pitch = float(min(pitch_h, pitch_w) * 0.96)
        return {
            "orientation": "vertical", "columns": columns, "rows": rows,
            "glyph_pitch_px": round(max(4.0, pitch), 3),
            **common,
        }

    aspect = float(ink_h / max(1.0, ink_w))
    rows = int(np.clip(round(np.sqrt(max(0.05, aspect) * n)), 1, 8))
    cols = max(1, int(np.ceil(n / rows)))
    pitch_h = ink_h / rows
    pitch_w = ink_w / cols
    pitch = float(min(pitch_h, pitch_w) * 0.96)
    return {
        "orientation": "horizontal", "rows": rows, "columns": cols,
        "glyph_pitch_px": round(max(4.0, pitch), 3),
        **common,
    }


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Local non-zero bbox helper for pipeline masks.

    Do not depend on lettering.py's private _safe_bbox; v0.8.13 accidentally
    referenced that private helper without importing it, causing a NameError on
    the new layout-integrity path on macOS.
    """
    if mask is None or getattr(mask, "size", 0) == 0:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def masked_layout_profile(image: np.ndarray, mask: np.ndarray, text: str, orientation: str) -> dict:
    if image.size == 0 or mask.size == 0:
        return {}
    box = mask_bbox(mask)
    if box is None:
        return {}
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1].copy()
    local = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return {}
    if crop.ndim == 3:
        crop[local == 0] = 255
    else:
        crop[local == 0] = 255
    return source_layout_profile(crop, text, orientation)


def region_center_norm(bbox: tuple[int, int, int, int] | list[int], bubble_bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    bx0, by0, bx1, by1 = [float(v) for v in bubble_bbox]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bw = max(1.0, bx1 - bx0); bh = max(1.0, by1 - by0)
    return ((x0 + x1) * 0.5 - bx0) / bw, ((y0 + y1) * 0.5 - by0) / bh


def pair_reletter_bubbles(
    source_bubbles: list[BubbleInstance],
    target_bubbles: list[BubbleInstance],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[dict[str, BubbleInstance], dict]:
    """Build a conservative one-to-one SOURCE→TARGET bubble map for reletter.

    Paired-diff IDs are authoritative.  Older workspaces/caches may miss one side
    of the reciprocal metadata, so unresolved bubbles are recovered geometrically
    in normalized page coordinates.  List order is never used as a pairing signal:
    that was the main route by which text from one balloon could be rendered into
    a neighbouring balloon when detector ordering changed.
    """
    sh, sw = source_shape[:2]; th, tw = target_shape[:2]
    source_by_id = {str(b.id): b for b in source_bubbles}
    target_by_id = {str(b.id): b for b in target_bubbles}
    mapping: dict[str, BubbleInstance] = {}
    used_source: set[str] = set()
    rows: list[dict] = []

    # Target-side explicit identity.
    for tb in target_bubbles:
        sid = str((tb.meta or {}).get('paired_source_id') or '')
        sb = source_by_id.get(sid)
        if sb is not None and sid not in used_source:
            mapping[str(tb.id)] = sb; used_source.add(sid)
            rows.append({'target_bubble_id': str(tb.id), 'source_bubble_id': sid, 'route': 'target_explicit', 'cost': 0.0})

    # Reciprocal source-side identity can repair a missing target-side field.
    for sb in source_bubbles:
        if str(sb.id) in used_source:
            continue
        tid = str((sb.meta or {}).get('paired_target_id') or '')
        tb = target_by_id.get(tid)
        if tb is not None and tid not in mapping:
            mapping[tid] = sb; used_source.add(str(sb.id))
            rows.append({'target_bubble_id': tid, 'source_bubble_id': str(sb.id), 'route': 'source_reciprocal', 'cost': 0.0})

    unresolved_t = [tb for tb in target_bubbles if str(tb.id) not in mapping]
    unresolved_s = [sb for sb in source_bubbles if str(sb.id) not in used_source]
    if unresolved_t and unresolved_s:
        cost = np.full((len(unresolved_t), len(unresolved_s)), 9.0, np.float32)
        for ti, tb in enumerate(unresolved_t):
            tx0,ty0,tx1,ty1 = [float(v) for v in tb.bbox]
            tcx=((tx0+tx1)*0.5)/max(1.0,float(tw)); tcy=((ty0+ty1)*0.5)/max(1.0,float(th))
            taw=max(1.0,tx1-tx0)/max(1.0,float(tw)); tah=max(1.0,ty1-ty0)/max(1.0,float(th))
            for si, sb in enumerate(unresolved_s):
                sx0,sy0,sx1,sy1 = [float(v) for v in sb.bbox]
                scx=((sx0+sx1)*0.5)/max(1.0,float(sw)); scy=((sy0+sy1)*0.5)/max(1.0,float(sh))
                saw=max(1.0,sx1-sx0)/max(1.0,float(sw)); sah=max(1.0,sy1-sy0)/max(1.0,float(sh))
                center=float(np.hypot(tcx-scx,tcy-scy))
                size=min(1.5,abs(np.log(max(1e-5,saw/max(1e-5,taw))))+abs(np.log(max(1e-5,sah/max(1e-5,tah)))))
                kind=0.0 if str(tb.kind)==str(sb.kind) else 0.18
                cost[ti,si]=center*4.8+size*0.20+kind
        rr,cc=linear_sum_assignment(cost)
        for ti,si in zip(rr.tolist(),cc.tolist()):
            c=float(cost[ti,si]); tb=unresolved_t[ti]; sb=unresolved_s[si]
            # Publication-safe gate: uncertain bubbles remain untranslated rather
            # than borrowing text from the wrong neighbour.
            row_sorted=np.sort(cost[ti]); margin=float(row_sorted[1]-row_sorted[0]) if len(row_sorted)>1 else 1.0
            if c <= 0.72 and (margin >= 0.08 or c <= 0.28):
                mapping[str(tb.id)] = sb; used_source.add(str(sb.id))
                rows.append({'target_bubble_id': str(tb.id), 'source_bubble_id': str(sb.id), 'route': 'normalized_geometry', 'cost': round(c,4), 'margin': round(margin,4)})
            else:
                rows.append({'target_bubble_id': str(tb.id), 'source_bubble_id': str(sb.id), 'route': 'rejected_ambiguous_geometry', 'cost': round(c,4), 'margin': round(margin,4)})
    return mapping, {
        'route': 'explicit_then_normalized_geometry',
        'mapped_count': len(mapping),
        'source_count': len(source_bubbles),
        'target_count': len(target_bubbles),
        'pairs': rows,
    }


def filter_region_query_blocks(
    blocks: list[TextBlock],
    bbox: tuple[int,int,int,int],
    full_mask: np.ndarray | None = None,
) -> tuple[list[TextBlock], dict]:
    """Keep only external/coordinate OCR blocks owned by this source region.

    A broad bbox overlap alone can pull a neighbouring balloon's block into the
    query.  Prefer centroid-in-region and, when available, the actual detected
    source text-island mask.
    """
    x0,y0,x1,y1=[int(v) for v in bbox]
    accepted=[]; rejected=[]
    for b in blocks:
        cx,cy=b.centroid
        center_inside=(x0 <= cx <= x1 and y0 <= cy <= y1)
        mask_inside=False
        if full_mask is not None and full_mask.size:
            ix=int(round(cx)); iy=int(round(cy))
            if 0 <= iy < full_mask.shape[0] and 0 <= ix < full_mask.shape[1]:
                mask_inside=bool(full_mask[iy,ix] > 0)
        bx0,by0,bx1,by1=b.bbox
        ix0,iy0=max(float(x0),bx0),max(float(y0),by0); ix1,iy1=min(float(x1),bx1),min(float(y1),by1)
        inter=max(0.0,ix1-ix0)*max(0.0,iy1-iy0)
        barea=max(1.0,(bx1-bx0)*(by1-by0))
        overlap=inter/barea
        if full_mask is not None and getattr(full_mask, "size", 0):
            # When we have an actual source text-island mask it is authoritative:
            # broad rectangle overlap must not pull a neighbour into this Region.
            ok = mask_inside or overlap >= 0.90
        else:
            ok = (center_inside and overlap >= 0.45) or overlap >= 0.82
        (accepted if ok else rejected).append((b, overlap, center_inside, mask_inside))
    accepted.sort(key=lambda row:(int(getattr(row[0],'reading_order',0)),row[0].bbox[1],row[0].bbox[0]))
    return [row[0] for row in accepted], {
        'input_blocks': len(blocks), 'accepted_blocks': len(accepted), 'rejected_blocks': len(rejected),
        'accepted_ids': [str(row[0].id) for row in accepted],
    }


def match_paired_bubble_regions(
    source_regions: list,
    target_regions: list,
    source_bubble_bbox: tuple[float, float, float, float],
    target_bubble_bbox: tuple[float, float, float, float],
) -> tuple[dict[str, object], dict]:
    """Match detected source subregions to target subregions inside one paired bubble.

    The earlier reletter path projected each target text island directly through
    the two bubble bounding boxes. On photographed SOURCE pages, that rectangle can
    cut across neighbouring source text islands within the same compound balloon,
    causing mixed OCR transcripts and severely wrong fill. Here we detect source
    subregions too and bind them by relative position/orientation before OCR.
    """
    if not source_regions or not target_regions:
        return {}, {"route": "fallback_no_source_subregions", "source_count": len(source_regions), "target_count": len(target_regions)}
    cost = np.zeros((len(target_regions), len(source_regions)), dtype=np.float32)
    pair_rows = []
    for ti, tr in enumerate(target_regions):
        tcx, tcy = region_center_norm(tr.bbox, target_bubble_bbox)
        tw = max(1.0, tr.bbox[2] - tr.bbox[0]); th = max(1.0, tr.bbox[3] - tr.bbox[1])
        tasp = tw / th
        for si, sr in enumerate(source_regions):
            scx, scy = region_center_norm(sr.bbox, source_bubble_bbox)
            sw = max(1.0, sr.bbox[2] - sr.bbox[0]); sh = max(1.0, sr.bbox[3] - sr.bbox[1])
            sasp = sw / sh
            d = float(np.hypot(tcx - scx, tcy - scy))
            aspect = float(min(1.0, abs(np.log(max(1e-4, tasp / max(1e-4, sasp)))) / 1.8))
            orient_pen = 0.0 if str(tr.orientation) == str(sr.orientation) else 0.28
            # Slightly prefer larger source component groups when otherwise tied.
            comp_pen = abs(int(getattr(tr, 'component_count', 1)) - int(getattr(sr, 'component_count', 1))) * 0.012
            cost[ti, si] = d * 1.9 + aspect * 0.35 + orient_pen + comp_pen
    rows, cols = linear_sum_assignment(cost)
    mapping: dict[str, object] = {}
    accepted_costs: list[float] = []
    for ti, si in zip(rows.tolist(), cols.tolist()):
        c = float(cost[ti, si])
        tr = target_regions[ti]; sr = source_regions[si]
        row_sorted = np.sort(cost[ti])
        row_margin = float(row_sorted[1] - row_sorted[0]) if len(row_sorted) > 1 else 1.0
        col_sorted = np.sort(cost[:, si])
        col_margin = float(col_sorted[1] - col_sorted[0]) if len(col_sorted) > 1 else 1.0
        tcx,tcy=region_center_norm(tr.bbox,target_bubble_bbox)
        scx,scy=region_center_norm(sr.bbox,source_bubble_bbox)
        center_delta=float(np.hypot(tcx-scx,tcy-scy))
        accepted = bool(c <= 0.92 and center_delta <= 0.34 and ((row_margin >= 0.10 and col_margin >= 0.08) or center_delta <= 0.12))
        pair_rows.append({
            'target_region_id': str(getattr(tr, 'id', ti)),
            'source_region_id': str(getattr(sr, 'id', si)),
            'cost': round(c, 4),
            'center_delta': round(center_delta,4),
            'row_margin': round(row_margin,4),
            'col_margin': round(col_margin,4),
            'accepted': accepted,
        })
        # Ambiguous subregion matches are intentionally rejected. Falling back to
        # the normalized target projection is safer than swapping text islands.
        if accepted:
            mapping[str(getattr(tr, 'id', ti))] = sr
            accepted_costs.append(c)
    diag = {
        'route': 'paired_source_subregions',
        'source_count': len(source_regions),
        'target_count': len(target_regions),
        'matched_count': len(mapping),
        'mean_cost': round(float(np.mean(accepted_costs)), 4) if accepted_costs else None,
        'pairs': pair_rows,
    }
    return mapping, diag


def normalize_region_ocr_text(blocks: list[TextBlock], orientation: str) -> tuple[str, dict]:
    """Conservatively normalize OCR output from one already-bound source Region.

    OCR engines often return the same short line twice or insert arbitrary line
    breaks inside vertical manga text. Region identity is already fixed, so this
    stage only cleans transcript noise; it never changes which target Region owns
    the text.
    """
    rows = [b for b in blocks if str(getattr(b, "text", "") or "").strip()]
    rows.sort(key=lambda b: (int(getattr(b, "reading_order", 0)), b.bbox[1], b.bbox[0]))
    seen: set[str] = set()
    texts: list[str] = []
    dropped_dup = 0
    for b in rows:
        value = str(b.text).replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"[ \t\u3000]+", "", value).strip()
        compact = re.sub(r"\s+", "", value)
        if not compact:
            continue
        if compact in seen:
            dropped_dup += 1
            continue
        seen.add(compact)
        texts.append(value)
    if not texts:
        return "", {"input_blocks": len(rows), "deduped_blocks": 0, "dropped_duplicates": dropped_dup}
    if str(orientation) == "vertical":
        joined = "".join(t.replace("\n", "") for t in texts)
    else:
        joined = "\n".join(texts)
    joined = re.sub(r"([，。！？：；、])\1{1,}", r"\1", joined)
    return joined.strip(), {
        "input_blocks": len(rows),
        "deduped_blocks": len(texts),
        "dropped_duplicates": dropped_dup,
        "orientation": str(orientation),
    }

__all__ = [
    "contains_cjk", "infer_region_orientation", "source_layout_profile", "mask_bbox",
    "masked_layout_profile", "region_center_norm", "pair_reletter_bubbles",
    "filter_region_query_blocks", "match_paired_bubble_regions", "normalize_region_ocr_text",
]
