from __future__ import annotations

"""Aligned-hole private TARGET container completion.

This module deliberately belongs to aligned_overlay_reveal.  It uses no OCR and
never imports another transfer mode.  Its only job is to recover locally enclosed
white/burst containers around compact printed-text geometry when the selected
primary bubble detector misses them.
"""

from typing import Sequence

import cv2
import numpy as np

from ...geometry import mask_to_largest_polygon
from ...models import BubbleInstance


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero((a > 0) & (b > 0)))
    union = int(np.count_nonzero((a > 0) | (b > 0)))
    return inter / max(1, union)


def _flatten_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (0, 0), 15.0)
    norm = np.clip(gray.astype(np.float32) * (245.0 / np.maximum(32.0, blur.astype(np.float32))), 0, 255)
    return norm.astype(np.uint8)


def _nearest_seed(free: np.ndarray, x: float, y: float, radius: int = 56) -> tuple[int, int] | None:
    h, w = free.shape
    xi = int(np.clip(round(x), 0, w - 1)); yi = int(np.clip(round(y), 0, h - 1))
    if free[yi, xi] > 0:
        return xi, yi
    for r in range(1, max(1, int(radius)) + 1):
        x0=max(0,xi-r); x1=min(w,xi+r+1); y0=max(0,yi-r); y1=min(h,yi+r+1)
        yy, xx = np.where(free[y0:y1, x0:x1] > 0)
        if xx.size:
            d=(xx+x0-xi)**2+(yy+y0-yi)**2
            k=int(np.argmin(d))
            return int(xx[k]+x0), int(yy[k]+y0)
    return None


def _compact_character_components(gray: np.ndarray) -> list[tuple[int,int,int,int,int,tuple[float,float]]]:
    dark=(gray < 175).astype(np.uint8)*255
    joined=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
    joined=cv2.dilate(joined,np.ones((2,2),np.uint8))
    n,_lab,stats,cents=cv2.connectedComponentsWithStats(joined,8)
    out=[]
    for i in range(1,n):
        x,y,w,h,area=[int(v) for v in stats[i]]
        if not (15 <= area <= 4000 and 4 <= w <= 80 and 5 <= h <= 90):
            continue
        if max(w/max(1,h),h/max(1,w)) > 5.2:
            continue
        if area/max(1,w*h) < 0.08:
            continue
        out.append((x,y,w,h,area,(float(cents[i][0]),float(cents[i][1]))))
    return out


def _cluster_text_components(gray: np.ndarray, comps, min_components: int) -> list[dict]:
    if not comps:
        return []
    glyph=np.zeros_like(gray,np.uint8)
    for x,y,w,h,_a,_c in comps:
        glyph[y:y+h,x:x+w]=255
    widths=[c[2] for c in comps]; heights=[c[3] for c in comps]
    mw=float(np.median(widths)); mh=float(np.median(heights))
    kernels=[
        cv2.getStructuringElement(cv2.MORPH_RECT,(max(9,int(mw*1.35)),max(17,int(mh*2.2)))),
        cv2.getStructuringElement(cv2.MORPH_RECT,(max(17,int(mw*2.2)),max(9,int(mh*1.35)))),
    ]
    merged=np.zeros_like(glyph)
    for k in kernels:
        merged=cv2.bitwise_or(merged,cv2.dilate(glyph,k))
    n,_labels,stats,_cents=cv2.connectedComponentsWithStats(merged,8)
    groups=[]
    for i in range(1,n):
        x,y,w,h,_area=[int(v) for v in stats[i]]
        if w>int(gray.shape[1]*0.55) or h>int(gray.shape[0]*0.38):
            continue
        inside=[c for c in comps if x <= c[5][0] < x+w and y <= c[5][1] < y+h]
        if len(inside) < min_components:
            continue
        xs=[c[0] for c in inside]; ys=[c[1] for c in inside]
        x2=[c[0]+c[2] for c in inside]; y2=[c[1]+c[3] for c in inside]
        tb=(min(xs),min(ys),max(x2),max(y2))
        groups.append({"count":len(inside),"bbox":tb,"centroid":(0.5*(tb[0]+tb[2]),0.5*(tb[1]+tb[3]))})
    groups.sort(key=lambda g:g["count"],reverse=True)
    kept=[]
    for g in groups:
        gx0,gy0,gx1,gy1=g["bbox"]
        if any(abs(gx0-k["bbox"][0])<12 and abs(gy0-k["bbox"][1])<12 and abs(gx1-k["bbox"][2])<12 and abs(gy1-k["bbox"][3])<12 for k in kept):
            continue
        kept.append(g)
    return kept


def _barrier_label_map(norm: np.ndarray, threshold: int, dilate_px: int):
    barrier=(norm <= int(threshold)).astype(np.uint8)*255
    barrier=cv2.morphologyEx(barrier,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    if dilate_px>0:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilate_px*2+1,dilate_px*2+1))
        barrier=cv2.dilate(barrier,k)
    free=(barrier==0).astype(np.uint8)*255
    n,labels,_stats,_=cv2.connectedComponentsWithStats((free>0).astype(np.uint8),8)
    return free,n,labels


def _component_from_label_map(free: np.ndarray, n: int, labels: np.ndarray, x: float, y: float) -> np.ndarray | None:
    seed=_nearest_seed(free,x,y,56)
    if seed is None:
        return None
    lab=int(labels[seed[1],seed[0]])
    if lab<=0 or lab>=n:
        return None
    raw=(labels==lab).astype(np.uint8)*255
    contours,_=cv2.findContours(raw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    out=np.zeros_like(raw)
    cv2.drawContours(out,[max(contours,key=cv2.contourArea)],-1,255,cv2.FILLED)
    return out


def _semanticize(mask: np.ndarray, text_bbox, container_bbox, barrier_dilate_px: int) -> np.ndarray:
    m=(mask>0).astype(np.uint8)*255
    h,w=m.shape[:2]
    bx0,by0,bx1,by1=[int(v) for v in container_bbox]
    tx0,ty0,tx1,ty1=[int(v) for v in text_bbox]
    seed_pad=max(2,int(barrier_dilate_px)+2)
    sx0=max(bx0,tx0-seed_pad); sy0=max(by0,ty0-seed_pad)
    sx1=min(bx1,tx1+seed_pad); sy1=min(by1,ty1+seed_pad)
    repaired=m.copy()
    if sx1>sx0 and sy1>sy0:
        repaired[sy0:sy1,sx0:sx1]=255
    poly=mask_to_largest_polygon(repaired)
    if len(poly)<3:
        return np.zeros_like(m)
    semantic=np.zeros_like(m)
    cv2.fillPoly(semantic,[np.asarray(poly,dtype=np.int32).reshape(-1,1,2)],255)
    guard=np.zeros_like(m); guard[max(0,by0):min(h,by1),max(0,bx0):min(w,bx1)]=255
    return cv2.bitwise_and(semantic,guard)


def detect_text_barrier_containers(target: np.ndarray, cfg, existing: Sequence[BubbleInstance] | None = None) -> list[BubbleInstance]:
    """Recover white/burst containers from TARGET printed-text geometry only."""
    if not bool(getattr(cfg,"target_text_barrier_enabled",True)):
        return []
    gray=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY) if target.ndim==3 else target
    h,w=gray.shape; page=max(1,h*w)
    comps=_compact_character_components(gray)
    groups=_cluster_text_components(gray,comps,int(getattr(cfg,"target_text_barrier_min_components",4)))
    max_groups=int(getattr(cfg,"target_text_barrier_max_candidates",48))
    norm=_flatten_gray(target)
    label_cache={}
    existing_masks=[b.mask for b in (existing or []) if b.mask is not None and b.mask.shape==gray.shape]
    out=[]
    variants=((205,4),(185,4),(165,3),(145,3),(175,2))
    for g in groups[:max_groups]:
        tx0,ty0,tx1,ty1=g["bbox"]; text_area=max(1,(tx1-tx0)*(ty1-ty0)); cx,cy=g["centroid"]
        if any(0 <= int(round(cy)) < h and 0 <= int(round(cx)) < w and m[int(round(cy)),int(round(cx))] > 0 for m in existing_masks):
            continue
        best=None
        for thr,dp in variants:
            key=(thr,dp)
            if key not in label_cache:
                label_cache[key]=_barrier_label_map(norm,thr,dp)
            m=_component_from_label_map(*label_cache[key],cx,cy)
            if m is None:
                continue
            area=int(cv2.countNonZero(m)); ar=area/page
            if area < 300 or ar > float(getattr(cfg,"target_text_barrier_max_area_ratio",0.055)):
                continue
            bb=_bbox(m)
            if bb is None:
                continue
            bx0,by0,bx1,by1=bb
            if not (bx0 <= tx0 and by0 <= ty0 and bx1 >= tx1 and by1 >= ty1):
                continue
            ratio=area/max(1,text_area)
            if ratio > float(getattr(cfg,"target_text_barrier_max_area_to_text_ratio",45.0)):
                continue
            vals=gray[m>0]
            white=float(np.mean(vals>225)); dark=float(np.mean(vals<180))
            if white < float(getattr(cfg,"target_text_barrier_min_white_ratio",0.72)):
                continue
            if dark > float(getattr(cfg,"target_text_barrier_max_dark_ratio",0.24)) or dark < float(getattr(cfg,"target_text_barrier_min_dark_ratio",0.02)):
                continue
            best=(m,bb,thr,dp,white,dark,ratio)
            break
        if best is None:
            continue
        m,bb,thr,dp,white,dark,ratio=best
        if any(_iou(m,e)>0.70 for e in existing_masks):
            continue
        if any(_iou(m,b.mask)>0.70 for b in out if b.mask is not None):
            continue
        semantic=_semanticize(m,g["bbox"],bb,dp)
        if cv2.countNonZero(semantic)<300:
            continue
        poly=mask_to_largest_polygon(semantic)
        if len(poly)<3:
            continue
        out.append(BubbleInstance(
            id=f"aligned-text-barrier-{len(out):04d}", polygon=poly,
            confidence=float(np.clip(0.78 + min(0.15,g["count"]*0.004),0,0.94)),
            kind="speech", mask=semantic, safe_mask=semantic.copy(), block_ids=[],
            meta={
                "backend":"aligned_text_barrier","target_only":True,
                "text_component_count":int(g["count"]),"text_bbox":[int(v) for v in g["bbox"]],
                "barrier_threshold":int(thr),"barrier_dilate_px":int(dp),
                "white_ratio":float(white),"dark_ratio":float(dark),"area_to_text_ratio":float(ratio),
                "container_bbox":[int(v) for v in bb],
            },
        ))
    return out


__all__ = ["detect_text_barrier_containers"]
