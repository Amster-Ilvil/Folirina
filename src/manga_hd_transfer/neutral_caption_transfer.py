"""Paired raster captions on neutral smooth fields, including halftone TARGETs.

No OCR, page identifiers, fixed coordinates or generated glyphs are used.
"""
from __future__ import annotations
import cv2
import numpy as np


def _box(row, matrix, shape):
    raw = (getattr(row, 'meta', {}) or {}).get('raw_box')
    if not raw or len(raw) != 4:
        return None
    a,b,c,d = map(float,raw)
    points = np.array([[[a,b],[c,b],[c,d],[a,d]]],np.float32)
    if matrix is not None:
        points = cv2.perspectiveTransform(points,np.asarray(matrix,np.float64))
    x,y = points.reshape(-1,2).min(axis=0); X,Y = points.reshape(-1,2).max(axis=0)
    return (max(0,int(np.floor(x))-3),max(0,int(np.floor(y))-3),min(shape[1],int(np.ceil(X))+3),min(shape[0],int(np.ceil(Y))+3))


def caption_ink(image, box, *, halftone=False):
    """Prove a smooth paper field and recover dark lettering with continuous alpha."""
    x,y,X,Y=box
    if X-x < 16 or Y-y < 16:
        return None
    crop=image[y:Y,x:X]
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth=cv2.GaussianBlur(gray,(0,0),1.25) if halftone else gray
    paper=float(np.percentile(smooth,80))
    if not 145 <= paper <= 245:
        return None
    # Exclude dark glyphs and bright UI decoration before testing paper.
    field=smooth[(smooth>paper-35)&(smooth<paper+25)]
    # Bold single-line headings can legitimately occupy almost half the box.
    minimum_field = .50 if X-x > 3*(Y-y) else .65
    if field.size < smooth.size*minimum_field or float(np.std(field))>13:
        return None
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    if float(np.percentile(hsv[:,:,1],80))>65:
        return None
    # Thin Japanese strokes become gray after suppressing the screentone.
    # A fixed dark cutoff drops them entirely and leaves ghost lettering.
    cutoff = paper-32 if halftone else min(125,paper-60)
    core=(smooth < cutoff).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(core,8)
    support=np.zeros_like(core)
    accepted=0
    for i in range(1,n):
        a,b,w,h,area=map(int,stats[i])
        if area<4:continue
        # Reject lines, borders, large artwork and page furniture.
        if area>core.size*.22 or w>max(18,h*12) or h>max(18,w*12):
            return None
        support[labels==i]=1;accepted+=1
    ratio=np.mean(support>0)
    if accepted<3 or not .025<=ratio<=.40:
        return None
    near=cv2.dilate(support,np.ones((5,5),np.uint8))>0
    # Keep SOURCE antialiasing; never copy its paper/texture into TARGET.
    alpha=np.clip((paper-gray)/max(40,paper-25),0,1)
    alpha[~near]=0
    alpha[alpha<.08]=0
    return {'alpha':alpha,'core':support,'paper':paper,'components':accepted,'ink_ratio':float(ratio)}


def recover_neutral_captions(source,target,source_rows,target_rows,registration,exclude):
    from .lama_manga_runtime import status,inpaint_koharu_text
    if registration is None or float(registration.confidence)<.75 or not status().ready:
        return []
    shape=target.shape[:2]; proposals=[]; claimed=[]
    for t in target_rows:
        if t.confidence < .35:continue
        tb=_box(t,None,shape)
        if tb is None:continue
        tx,ty,tX,tY=tb;ta=(tX-tx)*(tY-ty)
        if ta<400 or ta>np.prod(shape)*.08:continue
        if any(
            max(0,min(tX,p['target_box'][2])-max(tx,p['target_box'][0]))
            *max(0,min(tY,p['target_box'][3])-max(ty,p['target_box'][1]))
            > .25*min(ta,(p['target_box'][2]-p['target_box'][0])*(p['target_box'][3]-p['target_box'][1]))
            for p in proposals
        ):continue
        if np.mean(exclude[ty:tY,tx:tX]>0)>.20:continue
        target_ink=caption_ink(target,tb,halftone=True)
        if target_ink is None:continue
        best=None
        for s in source_rows:
            if s.confidence < .30:continue
            sb=_box(s,registration.matrix,shape)
            if sb is None:continue
            sx,sy,sX,sY=sb;sa=(sX-sx)*(sY-sy)
            inter=max(0,min(sX,tX)-max(sx,tx))*max(0,min(sY,tY)-max(sy,ty))
            overlap=inter/max(1,min(sa,ta));sim=min(sa,ta)/max(1,max(sa,ta))
            if overlap<.75 or sim<.40:continue
            ink=caption_ink(source,sb)
            if ink is None:continue
            score=overlap*np.sqrt(sim)
            if best is None or score>best[0]:best=(score,s,sb,ink)
        if best is None:continue
        _,s,sb,ink=best
        if any(s is prev for prev in claimed):continue
        sx,sy,sX,sY=sb
        src=np.zeros(shape,np.float32);src[sy:sY,sx:sX]=ink['alpha']
        tgt=np.zeros(shape,np.uint8);tgt[ty:tY,tx:tX]=target_ink['core']*255
        # Identical source/target lettering is not a translation candidate.
        a=cv2.dilate((src>.35).astype(np.uint8),np.ones((5,5),np.uint8))>0
        b=cv2.dilate((tgt>0).astype(np.uint8),np.ones((5,5),np.uint8))>0
        difference=np.count_nonzero(a^b)/max(1,np.count_nonzero(a|b))
        if difference<.35:continue
        clear=cv2.dilate(tgt,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
        scope=np.maximum(clear,(src>0).astype(np.uint8)*255)
        if np.any((scope>0)&(exclude>0)):continue
        proposals.append({'source':s,'target':t,'source_box':sb,'target_box':tb,'alpha':src,'clear':clear,'difference':float(difference)})
        claimed.append(s)
    if not proposals:return []
    # One model load per page; compose original SOURCE opacity after cleaning.
    union=np.maximum.reduce([p['clear'] for p in proposals])
    cleaned,details=inpaint_koharu_text(target,union)
    for p in proposals:
        alpha=p['alpha'];out=target.copy();m=p['clear']>0;out[m]=cleaned[m]
        # Dark ink is applied as opacity; SOURCE background is never published.
        out=np.rint(out.astype(np.float32)*(1-alpha[:,:,None])).clip(0,255).astype(np.uint8)
        p['image']=out;p['write']=np.maximum(p['clear'],(alpha>0).astype(np.uint8)*255)
        p['model']=details
    return proposals
