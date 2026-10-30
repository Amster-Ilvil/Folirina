"""Bounded evidence for dark TARGET glyphs printed with a white outline.

This is not a white-pixel eraser. Only compact white components enclosing dark
glyph evidence on a coloured field qualify. Callers still own the write scope.
"""
from __future__ import annotations

import cv2
import numpy as np


def outlined_target_text_mask(target: np.ndarray) -> np.ndarray:
    if target.ndim != 3 or target.shape[2] != 3 or target.dtype != np.uint8:
        raise ValueError("outlined text detection expects uint8 BGR")
    h, w = target.shape[:2]
    out = np.zeros((h, w), np.uint8)
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
            continue
        out[y:y+bh, x:x+bw][filled] = 255
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
    return cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


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
    # A generative fill may invent a coherent dark mark where a neutral local
    # inpaint predicts only background. Compare low-frequency fields so real
    # fine screentone dots do not trigger this guard; keep the model texture
    # elsewhere and never write beyond the admitted mask.
    fallback = cv2.inpaint(target, (active.astype(np.uint8)*255), 3, cv2.INPAINT_TELEA)
    model_field = cv2.GaussianBlur(cv2.cvtColor(repaired,cv2.COLOR_BGR2GRAY),(0,0),3)
    local_field = cv2.GaussianBlur(cv2.cvtColor(fallback,cv2.COLOR_BGR2GRAY),(0,0),3)
    valid=(cv2.dilate(active.astype(np.uint8),np.ones((7,7),np.uint8))==0).astype(np.float32)
    weight=cv2.GaussianBlur(valid,(0,0),16)
    observed=cv2.GaussianBlur(cv2.cvtColor(target,cv2.COLOR_BGR2GRAY).astype(np.float32)*valid,(0,0),16)
    expected=np.divide(observed,weight,out=np.zeros_like(observed),where=weight>.01)
    suspicious = (active & (weight>.01)
                  & (local_field.astype(np.int16)-model_field.astype(np.int16) > 24)
                  & (expected-model_field.astype(np.float32)>24))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(suspicious.astype(np.uint8),8)
    keep = np.zeros(n,np.uint8)
    keep[1:] = (stats[1:,cv2.CC_STAT_AREA] >= 12).astype(np.uint8)
    guard = cv2.dilate(keep[labels], np.ones((5,5),np.uint8)) > 0
    guard &= active
    out[guard] = fallback[guard]
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
            out[gy,gx] = np.clip(np.rint(at@fit),0,255).astype(np.uint8)
            changed += len(gx)
    out[~active] = target[~active]
    return out, {"verified_smooth_field_pixels": int(changed),
                 "model_dark_artifact_pixels": int(np.count_nonzero(guard))}
