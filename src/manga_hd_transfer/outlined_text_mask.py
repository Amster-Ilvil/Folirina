"""Bounded evidence for dark TARGET glyphs printed with a white outline.

This is not a white-pixel eraser. Only compact white components enclosing dark
glyph evidence on a coloured field qualify. Callers still own the write scope.
"""
from __future__ import annotations

import cv2
import numpy as np


def trusted_project_text_mask(project: dict, shape: tuple[int,int]) -> np.ndarray:
    """Recover detector-owned ink, never OCR rectangles, from saved projects."""
    from .koharu_mask_codec import decode_mask
    mask=np.zeros(shape,np.uint8)
    for row in project.get('target_blocks',[]) or []:
        meta=row.get('meta',{}) if isinstance(row,dict) else {}
        if (meta.get('backend')!='koharu_layout' or not meta.get('layout_evidence')
                or meta.get('layout_label') not in {'text','sfx'}
                or float(row.get('confidence',0))<.8):
            continue
        payload=meta.get('foreground_mask')
        if not isinstance(payload,dict):
            continue
        try:
            mask|=decode_mask(payload,shape)
        except (ValueError,TypeError,KeyError):
            continue
    return mask


def _scoped_neutral_glyph_cores(target, gray, scope, text_seed):
    """Find compact black cores even when their white rim joins page paper."""
    out = np.zeros(gray.shape,np.uint8)
    if scope is None and text_seed is None:
        return out
    authority = scope if scope is not None else cv2.dilate((text_seed>0).astype(np.uint8),np.ones((9,9),np.uint8))
    # Some printed outlines have AA gaps linking a black glyph to the black
    # illustration. A local closing isolates thin ink from that large field;
    # global dark-component labelling would discard both as one art component.
    local_paper=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    cores=(gray<190)&(local_paper.astype(np.int16)-gray.astype(np.int16)>32)
    n,labels,stats,_ = cv2.connectedComponentsWithStats(cores.astype(np.uint8),8)
    candidates=[]
    h,w=gray.shape
    for i in range(1,n):
        x,y,bw,bh,area=map(int,stats[i])
        if area<3 or max(bw,bh)>96 or bw*bh>4000 or max(bw,bh)>max(1,min(bw,bh))*16:
            continue
        core=labels[y:y+bh,x:x+bw]==i
        if np.mean(authority[y:y+bh,x:x+bw][core]>0)<.98:
            continue
        xa,ya,xb,yb=max(0,x-10),max(0,y-10),min(w,x+bw+10),min(h,y+bh+10)
        local=(labels[ya:yb,xa:xb]==i).astype(np.uint8)
        near=(cv2.dilate(local,np.ones((5,5),np.uint8))>0)&(local==0)
        far=(cv2.dilate(local,np.ones((21,21),np.uint8))>0)&(cv2.dilate(local,np.ones((9,9),np.uint8))==0)
        if np.mean(gray[ya:yb,xa:xb][near]>=220)<.60 or not np.any(far):
            continue
        surround=target[ya:yb,xa:xb][far]
        neutral_dark=(surround.max(1)<170)&(np.ptp(surround,axis=1)<=24)
        if np.mean(neutral_dark)<.20:
            continue
        seeded=text_seed is not None and np.any(text_seed[y:y+bh,x:x+bw][core]>0)
        candidates.append((x,y,bw,bh,i,seeded))
    # Explicit semantic glyph seeds are sufficient. Manual rectangular/stroke
    # scopes need a repeated text run, so isolated eyes/highlights stay intact.
    for x,y,bw,bh,i,seeded in candidates:
        aligned=[]
        for ox,oy,ow,oh,j,_ in candidates:
            if max(bw,ow)/max(1,min(bw,ow))>3.5 or max(bh,oh)/max(1,min(bh,oh))>3.5:
                continue
            dx,dy=abs(x+bw/2-ox-ow/2),abs(y+bh/2-oy-oh/2)
            if (dx<=max(bw,ow)*.8 and dy<=4*max(bh,oh)) or (dy<=max(bh,oh)*.8 and dx<=4*max(bw,ow)):
                aligned.append(j)
        if seeded or len(aligned)>=3:
            out[y:y+bh,x:x+bw][labels[y:y+bh,x:x+bw]==i]=255
    return out


def outlined_target_text_mask(target: np.ndarray, *, scope: np.ndarray | None = None,
                              text_seed: np.ndarray | None = None) -> np.ndarray:
    if target.ndim != 3 or target.shape[2] != 3 or target.dtype != np.uint8:
        raise ValueError("outlined text detection expects uint8 BGR")
    h, w = target.shape[:2]
    for mask in (scope, text_seed):
        if mask is not None and mask.shape != (h,w):
            raise ValueError('outlined text evidence mask dimensions differ')
    if text_seed is not None and not np.any(text_seed):
        text_seed = None
    out = np.zeros((h, w), np.uint8)
    neutral_candidates = []
    bright = ((target.min(2) >= 235) & (np.ptp(target, axis=2) <= 24)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    for label in range(1, n):
        x, y, bw, bh, area = map(int, stats[label])
        if area < 8 or max(bw, bh) > 192 or bw * bh > 12000:
            continue
        if x == 0 or y == 0 or x + bw == w or y + bh == h:
            continue
        component = (labels[y:y+bh, x:x+bw] == label).astype(np.uint8)
        holes = cv2.copyMakeBorder(1-component, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
        cv2.floodFill(holes, None, (0, 0), 0)
        holes = holes[1:-1, 1:-1] > 0
        dark = holes & (gray[y:y+bh, x:x+bw] < 150)
        dark_pixels = int(np.count_nonzero(dark))
        filled = (component > 0) | holes
        filled_area = int(np.count_nonzero(filled))
        if dark_pixels < 3 or dark_pixels / max(1, filled_area) < .015:
            continue
        # A broad white speech balloon/paper label is not a glyph outline.
        # Most admitted white pixels must sit close to its enclosed dark ink.
        distance = cv2.distanceTransform((~dark).astype(np.uint8),cv2.DIST_L2,3)
        if np.percentile(distance[component>0],95) > 12:
            continue
        # White balloons/paper and broad page furniture never become this tool's
        # authority. A coloured local surround is required independently.
        pad = 12
        xa, ya, xb, yb = max(0,x-pad), max(0,y-pad), min(w,x+bw+pad), min(h,y+bh+pad)
        context = target[ya:yb, xa:xb]
        context_labels = labels[ya:yb, xa:xb]
        surround = context[context_labels != label]
        if len(surround) < 24:
            continue
        colour = (np.ptp(surround, axis=1) >= 30) & (surround.max(1) >= 120)
        if np.count_nonzero(colour) < max(24, int(len(surround) * .30)):
            # A white outline on black/gray manga is just as real as one on
            # coloured paper. Require semantic ink evidence, or a manually
            # scoped run of at least three similar aligned glyphs. An eye or
            # isolated highlight never gains page-wide erasing authority.
            neutral = np.ptp(surround,axis=1) <= 24
            if np.mean(neutral) < .80 or np.median(surround) > 170:
                continue
            if text_seed is not None and np.any(dark & (text_seed[y:y+bh,x:x+bw] > 0)):
                out[y:y+bh,x:x+bw][filled] = 255
            elif scope is not None and np.mean(scope[y:y+bh,x:x+bw][filled] > 0) >= .98:
                neutral_candidates.append((x,y,bw,bh,filled))
            continue
        out[y:y+bh, x:x+bw][filled] = 255
    neighbours_by_id = [set() for _ in neutral_candidates]
    for i,(x,y,bw,bh,_filled) in enumerate(neutral_candidates):
        for j in range(i):
            ox,oy,ow,oh,_ = neutral_candidates[j]
            if max(bw,ow)/max(1,min(bw,ow)) > 2 or max(bh,oh)/max(1,min(bh,oh)) > 2:
                continue
            dx,dy = abs((x+bw/2)-(ox+ow/2)),abs((y+bh/2)-(oy+oh/2))
            aligned = ((dy <= max(bh,oh)*.5 and .4*min(bw,ow) < dx <= 2.5*max(bw,ow))
                       or (dx <= max(bw,ow)*.5 and .4*min(bh,oh) < dy <= 2.5*max(bh,oh)))
            if aligned:
                neighbours_by_id[i].add(j); neighbours_by_id[j].add(i)
    visited = set()
    for i in range(len(neutral_candidates)):
        if i in visited:
            continue
        group, todo = set(), [i]
        while todo:
            node = todo.pop()
            if node in group:
                continue
            group.add(node); todo.extend(neighbours_by_id[node]-group)
        visited.update(group)
        if len(group) >= 3:
            for node in group:
                x,y,bw,bh,filled = neutral_candidates[node]
                out[y:y+bh,x:x+bw][filled] = 255
    # Some open kana strokes have an AA gap in their white contour, so their
    # black centre is not topologically enclosed (the left stroke of い).
    # Recruit only compact white fragments next to an independently proven
    # glyph and directly touching dark ink. This cannot flood distant highlights.
    neighbours = cv2.dilate(out, np.ones((9, 9), np.uint8)) > 0
    near_dark = cv2.dilate((gray < 150).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    for label in range(1, n):
        x, y, bw, bh, area = map(int, stats[label])
        if area < 8 or max(bw,bh) > 64 or bw*bh > 3500:
            continue
        if x == 0 or y == 0 or x+bw == w or y+bh == h:
            continue
        component = labels[y:y+bh,x:x+bw] == label
        if not np.any(component & neighbours[y:y+bh,x:x+bw]):
            continue
        if np.count_nonzero(component & near_dark[y:y+bh,x:x+bw]) < 3:
            continue
        out[y:y+bh,x:x+bw][component] = 255
    # Printed white AA can extend farther than the solid white component. A
    # bounded four-pixel fringe removes that conditioning signal before LaMa.
    out |= _scoped_neutral_glyph_cores(target, gray, scope, text_seed)
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if scope is not None:
        out[scope == 0] = 0
    if text_seed is not None:
        close = cv2.dilate((text_seed > 0).astype(np.uint8), np.ones((17,17),np.uint8))
        out[close == 0] = 0
    return out


def refine_smooth_inpainted_background(target: np.ndarray, mask: np.ndarray,
                                      repaired: np.ndarray) -> tuple[np.ndarray, dict]:
    """Use a verified local colour plane where a model invents texture.

    Smooth skin/balloon gradients do not need hallucinated marks. Every local
    fit must explain *all* unmasked samples (including dots/hatching); textured
    or multi-colour neighbourhoods stay model-owned. Writes remain mask-only.
    """
    if target.shape != repaired.shape or target.shape[:2] != mask.shape:
        raise ValueError("smooth repair inputs must match")
    out = repaired.copy()
    active = mask > 0
    # Do not classify a reconstruction as a hallucination merely because it is
    # darker than Telea or a page-average brightness. That erased legitimate
    # reconstructed black outlines/halftone (image_53's dragon wing). Only the
    # verified local colour-plane fits below may replace model-owned pixels.
    dark_artifact_pixels = 0
    exclude = cv2.dilate(active.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    changed = 0
    h, w = mask.shape
    for y in range(0, h, 16):
        for x in range(0, w, 16):
            yb, xb = min(h,y+16), min(w,x+16)
            if not np.any(active[y:yb,x:xb]):
                continue
            xa, ya, xe, ye = max(0,x-20), max(0,y-20), min(w,xb+20), min(h,yb+20)
            sy, sx = np.nonzero(~exclude[ya:ye,xa:xe])
            if len(sx) < 64:
                continue
            values = target[sy+ya,sx+xa].astype(np.float64)
            median = np.median(values,axis=0)
            if median.max() < 120 or np.ptp(median) < 30:
                continue
            coords = np.column_stack((sx/max(1,xe-xa),sy/max(1,ye-ya),np.ones(len(sx))))
            fit = np.linalg.lstsq(coords,values,rcond=None)[0]
            residual = np.max(np.abs(coords@fit-values),axis=1)
            keep = residual <= 6
            if np.mean(keep) < .95:
                continue
            fit = np.linalg.lstsq(coords[keep],values[keep],rcond=None)[0]
            residual = np.max(np.abs(coords@fit-values),axis=1)
            if np.percentile(residual,95) > 6:
                continue
            ty, tx = np.nonzero(active[y:yb,x:xb])
            gy, gx = ty+y, tx+x
            at = np.column_stack(((gx-xa)/max(1,xe-xa),(gy-ya)/max(1,ye-ya),np.ones(len(gx))))
            predicted=np.clip(np.rint(at@fit),0,255).astype(np.uint8)
            dark_artifact_pixels += int(np.count_nonzero(np.mean(predicted.astype(float)-repaired[gy,gx].astype(float),axis=1)>24))
            out[gy,gx] = predicted
            changed += len(gx)
    out[~active] = target[~active]
    return out, {"verified_smooth_field_pixels": int(changed),
                 "model_dark_artifact_pixels": dark_artifact_pixels}
