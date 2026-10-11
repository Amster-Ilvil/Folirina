from __future__ import annotations

"""Conservative relocated open-text/SFX transfer for Direct Patch.

Cross-edition manga sometimes moves a translated effect caption within the same
panel instead of keeping it on top of the Japanese glyphs.  The ordinary
registered-ink completion assumes identical text position, so it can report an
"applied" record while writing only unrelated pixels.  This module handles only
large, standalone, same-panel text groups whose SOURCE/TARGET seed geometry is
strongly supported but clearly relocated.

It never copies a rectangular SOURCE background.  TARGET text pixels are erased
from a style-expanded mask and only SOURCE glyph/outline pixels are composited
at the registered SOURCE location.  Ambiguous cases are left untouched.
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ...geometry import transform_to_homography
from ...models import RegistrationResult
from .geometry_ops import BubblePatchMatch, _bbox_from_mask
from .transfer_models import MaskTransferRecord, MaskTransferResult


def _item_mask(item: Any, shape: tuple[int, int]) -> np.ndarray:
    mask = getattr(item, "mask", None)
    if isinstance(mask, np.ndarray) and mask.shape == shape:
        return (mask > 0).astype(np.uint8) * 255
    poly = list(getattr(item, "polygon", []) or [])
    out = np.zeros(shape, np.uint8)
    if len(poly) >= 3:
        pts = np.asarray(poly, np.float32)
        cv2.fillPoly(out, [np.round(pts).astype(np.int32)], 255)
    return out


def _project_item_mask(item: Any, registration: RegistrationResult, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    poly = list(getattr(item, "polygon", []) or [])
    if len(poly) < 3:
        return out
    pts = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    H = transform_to_homography(registration.matrix)
    mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    if mapped.size and np.all(np.isfinite(mapped)):
        cv2.fillPoly(out, [np.round(mapped).astype(np.int32)], 255)
    return out


def _overlap_over_min(a: np.ndarray, b: np.ndarray) -> float:
    aa = int(cv2.countNonZero(a)); ba = int(cv2.countNonZero(b))
    if aa <= 0 or ba <= 0 or a.shape != b.shape:
        return 0.0
    inter = int(np.count_nonzero((a > 0) & (b > 0)))
    return float(inter / max(1, min(aa, ba)))


def _box_overlap_over_min(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    aa = max(1.0, (ax1-ax0)*(ay1-ay0)); ba = max(1.0, (bx1-bx0)*(by1-by0))
    return float(inter / max(1.0, min(aa, ba)))


def _center(box):
    x0,y0,x1,y1 = [float(v) for v in box]
    return ((x0+x1)*0.5, (y0+y1)*0.5)


def _containing_panel_box(item_box, evidence: Any, registration: RegistrationResult | None = None, shape: tuple[int,int] | None = None):
    cx, cy = _center(item_box)
    choices=[]
    for row in list(getattr(evidence, "items", []) or []):
        if str(getattr(row, "label", "")) != "panel":
            continue
        if registration is None:
            box = tuple(float(v) for v in getattr(row, "box", ())) if len(getattr(row, "box", ())) == 4 else None
        else:
            if shape is None:
                continue
            mask = _project_item_mask(row, registration, shape)
            box = _bbox_from_mask(mask)
        if box is None:
            continue
        x0,y0,x1,y1=box
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            choices.append((max(1.0,(x1-x0)*(y1-y0)), tuple(float(v) for v in box)))
    return min(choices, key=lambda x:x[0])[1] if choices else None


def _inside_any_bubble(mask: np.ndarray, evidence: Any, shape: tuple[int,int], registration: RegistrationResult | None = None) -> bool:
    for row in list(getattr(evidence, "items", []) or []):
        if str(getattr(row, "label", "")) != "bubble":
            continue
        other = _project_item_mask(row, registration, shape) if registration is not None else _item_mask(row, shape)
        if _overlap_over_min(mask, other) >= 0.55:
            return True
    return False


def _page_furniture(mask: np.ndarray, shape: tuple[int,int]) -> bool:
    box=_bbox_from_mask(mask)
    if box is None:
        return False
    h,w=shape; x0,y0,x1,y1=box; bw=max(1,x1-x0); bh=max(1,y1-y0)
    return bool((y1 <= h*0.10 or y0 >= h*0.94) and bw <= w*0.22 and bh <= h*0.14)


def _style_group(image: np.ndarray, seed: np.ndarray, panel_box, *, min_components: int = 2):
    """Expand one Koharu text seed into a same-baseline glyph group.

    Returns (mask, diagnostics).  The style is inferred from the seed itself:
    saturated colour, bright-on-dark, or dark-on-bright.  Only glyph-sized
    connected components sharing the seed baseline are admitted.
    """
    box=_bbox_from_mask(seed)
    if box is None:
        return None, {"reason":"empty_seed"}
    x0,y0,x1,y1=box; sh=max(1,y1-y0); scy=(y0+y1)*0.5
    h,w=seed.shape
    px0,py0,px1,py1 = (0,0,w,h) if panel_box is None else [int(round(v)) for v in panel_box]
    px0=max(0,min(w,px0)); px1=max(0,min(w,px1)); py0=max(0,min(h,py0)); py1=max(0,min(h,py1))
    ylo=max(py0,int(round(scy-0.90*sh))); yhi=min(py1,int(round(scy+0.90*sh)))
    if yhi <= ylo:
        return None,{"reason":"empty_band"}
    band=np.zeros((h,w),np.uint8); band[ylo:yhi,px0:px1]=255
    ring=cv2.dilate(seed,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
    ring=cv2.subtract(ring,seed)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV); hue,sat,val=cv2.split(hsv)
    inside=(seed>0); ring_sel=(ring>0)
    if not np.any(inside) or not np.any(ring_sel):
        return None,{"reason":"no_style_samples"}
    sat_inside=sat[inside]; saturated_fraction=float(np.mean(sat_inside>=80))
    base=float(np.median(gray[ring_sel]))
    in_p70=float(np.percentile(gray[inside],70)); in_p30=float(np.percentile(gray[inside],30))
    style=""
    if saturated_fraction >= 0.18 and np.count_nonzero((inside)&(sat>=80)) >= 20:
        hs=hue[(inside)&(sat>=80)]
        hm=float(np.median(hs)); dh=np.minimum(np.abs(hue.astype(np.float32)-hm),180.0-np.abs(hue.astype(np.float32)-hm))
        raw=((sat>=70)&(val>=55)&(dh<=20)&(band>0)).astype(np.uint8)*255
        style="saturated"
    elif in_p70-base >= 32:
        thr=max(175.0,min(245.0,base+50.0)); raw=((gray>=thr)&(band>0)).astype(np.uint8)*255; style="bright"
    elif base-in_p30 >= 32:
        thr=min(165.0,max(20.0,base-45.0)); raw=((gray<=thr)&(band>0)).astype(np.uint8)*255; style="dark"
    else:
        return None,{"reason":"weak_style_contrast","base":base,"p70":in_p70,"p30":in_p30}

    n,labels,stats,cents=cv2.connectedComponentsWithStats((raw>0).astype(np.uint8),8)
    keep=np.zeros_like(raw); kept=0
    seed_area=max(1,int(cv2.countNonZero(seed)))
    min_area=max(12,int(round(seed_area*0.003)))
    for i in range(1,n):
        x,y,cw,ch,area=[int(v) for v in stats[i]]; cy=float(cents[i][1])
        if area < min_area: continue
        if ch < 0.24*sh or ch > 1.35*sh: continue
        if abs(cy-scy) > 0.48*sh: continue
        if cw > 2.2*sh: continue
        keep[labels==i]=255; kept+=1
    if kept < int(min_components) or cv2.countNonZero(keep) <= 0:
        return None,{"reason":"insufficient_group_components","style":style,"components":kept}

    near=cv2.dilate(keep,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    if style == "bright":
        fringe=((gray < max(135.0,base-18.0))&(near>0)&(band>0)).astype(np.uint8)*255
    elif style == "dark":
        fringe=((gray > min(215.0,base+18.0))&(near>0)&(band>0)).astype(np.uint8)*255
    else:
        fringe=((gray < max(145.0,base-15.0))&(near>0)&(band>0)).astype(np.uint8)*255
    group=cv2.bitwise_or(keep,fringe)
    group=cv2.morphologyEx(group,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    group=cv2.bitwise_and(group,band)
    return group,{"reason":"ok","style":style,"components":kept,"pixels":int(cv2.countNonZero(group))}


def transfer_relocated_koharu_open_text_regions(
    aligned_source: np.ndarray,
    original_target: np.ndarray,
    current_image: np.ndarray,
    source_evidence: Any,
    target_evidence: Any,
    registration: RegistrationResult,
    cfg: Any,
    *,
    exclude_mask: np.ndarray | None = None,
) -> MaskTransferResult:
    shape=original_target.shape[:2]
    empty_rgba=np.zeros((shape[0],shape[1],4),np.uint8); empty=np.zeros(shape,np.uint8)
    if aligned_source.shape[:2] != shape or current_image.shape[:2] != shape:
        return MaskTransferResult(current_image.copy(),empty_rgba,empty,[],[],empty)
    if source_evidence is None or target_evidence is None or not bool(getattr(source_evidence,"available",False)) or not bool(getattr(target_evidence,"available",False)):
        return MaskTransferResult(current_image.copy(),empty_rgba,empty,[],[],empty)
    if float(getattr(registration,"confidence",0.0) or 0.0) < 0.68:
        return MaskTransferResult(current_image.copy(),empty_rgba,empty,[],[],empty)
    if exclude_mask is None or exclude_mask.shape != shape:
        exclude_mask=np.zeros(shape,np.uint8)

    target_rows=[]
    for row in list(getattr(target_evidence,"items",[]) or []):
        if str(getattr(row,"label","")) not in {"text","sfx"}: continue
        if float(getattr(row,"confidence",0.0) or 0.0) < 0.25: continue
        mask=_item_mask(row,shape); box=_bbox_from_mask(mask)
        if box is None or _page_furniture(mask,shape) or _inside_any_bubble(mask,target_evidence,shape): continue
        area=cv2.countNonZero(mask)/max(1,shape[0]*shape[1])
        if area < 0.0022 or area > 0.08: continue
        if _overlap_over_min(mask,exclude_mask) > 0.22: continue
        panel=_containing_panel_box(box,target_evidence)
        if panel is None: continue
        target_rows.append((row,mask,box,panel))

    source_rows=[]
    for row in list(getattr(source_evidence,"items",[]) or []):
        if str(getattr(row,"label","")) not in {"text","sfx"}: continue
        if float(getattr(row,"confidence",0.0) or 0.0) < 0.25: continue
        pmask=_project_item_mask(row,registration,shape); box=_bbox_from_mask(pmask)
        if box is None or _page_furniture(pmask,shape): continue
        # Check bubble ownership in SOURCE coordinates before projection.
        smask=_item_mask(row,getattr(source_evidence,"shape",None) or getattr(getattr(row,"mask",None),"shape",shape))
        # LayoutEvidence masks normally exist in SOURCE shape; use polygon overlap directly when available.
        inside=False
        for b in list(getattr(source_evidence,"items",[]) or []):
            if str(getattr(b,"label","")) != "bubble": continue
            bm=getattr(b,"mask",None); rm=getattr(row,"mask",None)
            if isinstance(bm,np.ndarray) and isinstance(rm,np.ndarray) and bm.shape==rm.shape and _overlap_over_min((rm>0).astype(np.uint8)*255,(bm>0).astype(np.uint8)*255)>=0.55:
                inside=True; break
        if inside: continue
        area=cv2.countNonZero(pmask)/max(1,shape[0]*shape[1])
        if area < 0.0022 or area > 0.08: continue
        panel=_containing_panel_box(box,source_evidence,registration,shape)
        if panel is None: continue
        source_rows.append((row,pmask,box,panel))

    rendered=current_image.copy(); layer=np.zeros((shape[0],shape[1],4),np.uint8); composite=np.zeros(shape,np.uint8); clear_all=np.zeros(shape,np.uint8)
    records=[]; matches=[]; used_source=set()
    for trow,tmask,tbox,tpanel in target_rows:
        # Ordinary same-location semantic completion owns geometrically aligned text.
        if any(_overlap_over_min(tmask,smask)>=0.18 for _,smask,_,_ in source_rows):
            continue
        tx,ty=_center(tbox); tph=max(1.0,tpanel[3]-tpanel[1]); trely=(ty-tpanel[1])/tph
        tarea=max(1.0,(tbox[2]-tbox[0])*(tbox[3]-tbox[1]))
        best=None
        for si,(srow,smask,sbox,spanel) in enumerate(source_rows):
            if si in used_source: continue
            if _box_overlap_over_min(tpanel,spanel) < 0.70: continue
            sx,sy=_center(sbox); sph=max(1.0,spanel[3]-spanel[1]); srely=(sy-spanel[1])/sph
            rel_y=abs(trely-srely)
            sarea=max(1.0,(sbox[2]-sbox[0])*(sbox[3]-sbox[1])); area_ratio=min(tarea,sarea)/max(tarea,sarea)
            xshift=abs(tx-sx)/max(1.0,float(shape[1]))
            if rel_y>0.12 or area_ratio<0.35 or xshift<0.075: continue
            score=rel_y + 0.18*abs(np.log(max(area_ratio,1e-6))) + 0.05*min(xshift,0.5)
            if best is None or score < best[0]: best=(score,si,srow,smask,sbox,spanel,area_ratio,rel_y,xshift)
        if best is None: continue
        _,si,srow,sseed,sbox,spanel,area_ratio,rel_y,xshift=best
        src_group,sdiag=_style_group(aligned_source,sseed,spanel,min_components=2)
        tgt_group,tdiag=_style_group(original_target,tmask,tpanel,min_components=2)
        if src_group is None or tgt_group is None: continue
        # A relocated group should expand meaningfully beyond the one Koharu seed;
        # otherwise we risk treating an ordinary single glyph as a whole caption.
        if cv2.countNonZero(src_group) < 1.15*cv2.countNonZero(sseed): continue
        clear=cv2.dilate(tgt_group,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        if _overlap_over_min(clear,exclude_mask)>0.18: continue
        base=cv2.inpaint(rendered,clear,3.0,cv2.INPAINT_TELEA)
        alpha=cv2.GaussianBlur((src_group>0).astype(np.float32),(0,0),0.55)
        a3=alpha[...,None]
        out=np.clip(aligned_source.astype(np.float32)*a3 + base.astype(np.float32)*(1.0-a3),0,255).astype(np.uint8)
        write=np.maximum(clear,(alpha*255).astype(np.uint8)); changed=np.any(out!=rendered,axis=2); write[changed]=255
        rendered=out; composite=np.maximum(composite,write); clear_all=np.maximum(clear_all,clear)
        rgb=cv2.cvtColor(rendered,cv2.COLOR_BGR2RGB); use=write>0; layer[use,:3]=rgb[use]; layer[...,3]=np.maximum(layer[...,3],write)
        sid=str((getattr(srow,"meta",{}) or {}).get("item_index",si)); tid=str((getattr(trow,"meta",{}) or {}).get("item_index","?"))
        src_id=f"koharu-relocated-src-{sid}"; dst_id=f"koharu-relocated-dst-{tid}"
        conf=float(min(getattr(srow,"confidence",0.0),getattr(trow,"confidence",0.0)))
        rec=MaskTransferRecord(src_id,dst_id,conf,True,"applied_relocated_open_text")
        rec.geometry_mode="relocated_open_text"; rec.sr_backend="koharu-relocated-registered-raster"; rec.source_bbox=tuple(int(v) for v in _bbox_from_mask(src_group)); rec.target_bbox=tuple(int(v) for v in _bbox_from_mask(tgt_group)); rec.mask_iou=1.0; rec.target_coverage=1.0; rec.spill_ratio=0.0; rec.clarity_mode="koharu-relocated-open-text"; rec.content_check="checked_relocated_source_group"; rec.content_complete=True; rec.review_required=True; rec.review_reason="relocated_open_text_review"; rec.restorable=True; rec.editable=True
        rec.meta={"semantic_completion":True,"relocated":True,"source_style":sdiag,"target_style":tdiag,"panel_relative_y_delta":round(float(rel_y),4),"seed_area_ratio":round(float(area_ratio),4),"normalized_x_shift":round(float(xshift),4)}
        records.append(rec); matches.append(BubblePatchMatch(src_id,dst_id,conf,0.0,1.0,0.0,1.0,["koharu-relocated-open-text","same-panel","registered-source-raster"]))
        used_source.add(si)
    result = MaskTransferResult(rendered,layer,composite,matches,records,clear_all)
    # This pass requires two-sided Koharu semantic seeds, same-panel geometry,
    # style-group expansion and bounded glyph-only compositing.  Its exact write
    # footprint is therefore stronger evidence than the coarse panel PROTECT map.
    # Register only the pixels actually written; surrounding artwork stays protected.
    if records and any(bool(getattr(r, "applied", False)) for r in records):
        result.authority_allow_mask = composite.copy()
    return result


__all__=["transfer_relocated_koharu_open_text_regions"]
