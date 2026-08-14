from __future__ import annotations

"""SOURCE-only container detector providers for precise raster transfer.

No provider in this module is allowed to inspect TARGET text or create a
source/target bubble matching problem.  Providers only add container hypotheses
in canonical SOURCE-original coordinates.
"""

from pathlib import Path
import logging
import os
import cv2
import numpy as np

from .config import BubbleConfig, MaskReplaceConfig
from .geometry import mask_to_largest_polygon
from .models import BubbleInstance
from .plugins import REGISTRY, register_provider

logger = logging.getLogger(__name__)

_RTDETR_RUNTIME: dict[tuple[str, str, int], tuple[object, object]] = {}
_SAM2_RUNTIME: dict[tuple[str, str, str], object] = {}



def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero((a > 0) & (b > 0)))
    union = int(np.count_nonzero((a > 0) | (b > 0)))
    return inter / max(1, union)


def _nearest_seed(free: np.ndarray, x: float, y: float, radius: int = 48) -> tuple[int, int] | None:
    h, w = free.shape
    xi = int(np.clip(round(x), 0, w - 1)); yi = int(np.clip(round(y), 0, h - 1))
    if free[yi, xi] > 0:
        return xi, yi
    rmax = max(1, int(radius))
    for r in range(1, rmax + 1):
        x0=max(0,xi-r); x1=min(w,xi+r+1); y0=max(0,yi-r); y1=min(h,yi+r+1)
        yy, xx = np.where(free[y0:y1, x0:x1] > 0)
        if len(xx):
            d=(xx+x0-xi)**2+(yy+y0-yi)**2
            k=int(np.argmin(d))
            return int(xx[k]+x0), int(yy[k]+y0)
    return None


def _flatten_gray(image: np.ndarray) -> np.ndarray:
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    # Local illumination normalisation is intentionally mild: preserve printed
    # outlines while making photographed/off-white paper comparable.
    blur=cv2.GaussianBlur(gray,(0,0),15.0)
    norm=np.clip(gray.astype(np.float32) * (245.0 / np.maximum(32.0, blur.astype(np.float32))),0,255)
    return norm.astype(np.uint8)


def _barrier_component(image: np.ndarray, x: float, y: float, threshold: int, dilate_px: int) -> np.ndarray | None:
    norm=_flatten_gray(image)
    barrier=(norm <= int(threshold)).astype(np.uint8)*255
    barrier=cv2.morphologyEx(barrier,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    if dilate_px>0:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilate_px*2+1,dilate_px*2+1))
        barrier=cv2.dilate(barrier,k)
    free=(barrier==0).astype(np.uint8)*255
    seed=_nearest_seed(free,x,y,56)
    if seed is None:
        return None
    n,labels,stats,_=cv2.connectedComponentsWithStats((free>0).astype(np.uint8),8)
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




def _barrier_label_map(norm: np.ndarray, threshold: int, dilate_px: int):
    barrier=(norm <= int(threshold)).astype(np.uint8)*255
    barrier=cv2.morphologyEx(barrier,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    if dilate_px>0:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilate_px*2+1,dilate_px*2+1))
        barrier=cv2.dilate(barrier,k)
    free=(barrier==0).astype(np.uint8)*255
    n,labels,stats,_=cv2.connectedComponentsWithStats((free>0).astype(np.uint8),8)
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

def _compact_character_components(gray: np.ndarray) -> list[tuple[int,int,int,int,int,tuple[float,float]]]:
    # Join strokes just enough to approximate printed glyphs. Long panel/grid
    # lines are rejected by size/aspect/fill constraints.
    dark=(gray < 175).astype(np.uint8)*255
    joined=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
    joined=cv2.dilate(joined,np.ones((2,2),np.uint8))
    n,lab,stats,cents=cv2.connectedComponentsWithStats(joined,8)
    out=[]
    for i in range(1,n):
        x,y,w,h,area=[int(v) for v in stats[i]]
        if not (15 <= area <= 4000 and 4 <= w <= 80 and 5 <= h <= 90):
            continue
        if max(w/max(1,h),h/max(1,w)) > 5.2:
            continue
        fill=area/max(1,w*h)
        if fill < 0.08:
            continue
        out.append((x,y,w,h,area,(float(cents[i][0]),float(cents[i][1]))))
    return out


def _cluster_text_components(gray: np.ndarray, comps: list[tuple[int,int,int,int,int,tuple[float,float]]], min_components: int) -> list[dict]:
    if not comps:
        return []
    # Build a glyph map and directional unions. Vertical CJK columns and short
    # horizontal captions both become seed clusters, without requiring OCR.
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
    n,labels,stats,cents=cv2.connectedComponentsWithStats(merged,8)
    groups=[]
    for i in range(1,n):
        x,y,w,h,area=[int(v) for v in stats[i]]
        if w>int(gray.shape[1]*0.55) or h>int(gray.shape[0]*0.38):
            continue
        inside=[c for c in comps if x <= c[5][0] < x+w and y <= c[5][1] < y+h]
        if len(inside) < min_components:
            continue
        xs=[c[0] for c in inside]; ys=[c[1] for c in inside]
        x2=[c[0]+c[2] for c in inside]; y2=[c[1]+c[3] for c in inside]
        tb=(min(xs),min(ys),max(x2),max(y2))
        tc=(0.5*(tb[0]+tb[2]),0.5*(tb[1]+tb[3]))
        groups.append({"count":len(inside),"bbox":tb,"centroid":tc,"dilated_bbox":(x,y,x+w,y+h)})
    groups.sort(key=lambda g:g["count"],reverse=True)
    # remove near-identical/nested clusters
    kept=[]
    for g in groups:
        gx0,gy0,gx1,gy1=g["bbox"]
        if any(abs(gx0-k["bbox"][0])<12 and abs(gy0-k["bbox"][1])<12 and abs(gx1-k["bbox"][2])<12 and abs(gy1-k["bbox"][3])<12 for k in kept):
            continue
        kept.append(g)
    return kept


@register_provider("source_detector","pseudo_text_barrier")
def detect_pseudo_text_barrier(
    source: np.ndarray,
    mask_cfg: MaskReplaceConfig,
    bubble_cfg: BubbleConfig | None = None,
    existing: list[BubbleInstance] | None = None,
) -> list[BubbleInstance]:
    """Recover locally broken white/burst containers from SOURCE text seeds.

    The text seed is morphology only (no OCR/model). Dark line art is treated as
    a topological barrier. Multiple thresholds are tried and the smallest clean
    enclosed component is selected. It is especially useful for starbursts whose
    outer edge is connected to screentone and is missed by RETR_TREE contours.
    """
    if not bool(getattr(mask_cfg,"source_direct_text_seed_completion_enabled",True)):
        return []
    gray=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape; page=max(1,h*w)
    comps=_compact_character_components(gray)
    groups=_cluster_text_components(gray,comps,int(getattr(mask_cfg,"source_direct_text_seed_min_components",4)))
    max_groups=int(getattr(mask_cfg,"source_direct_text_seed_max_candidates",48))
    norm=_flatten_gray(source)
    label_cache={}
    existing_masks=[b.mask for b in (existing or []) if b.mask is not None and b.mask.shape==gray.shape]
    out=[]
    for gi,g in enumerate(groups[:max_groups]):
        tx0,ty0,tx1,ty1=g["bbox"]; text_area=max(1,(tx1-tx0)*(ty1-ty0))
        cx,cy=g["centroid"]
        # If this text is already well inside an accepted source container, no
        # completion work is needed.
        if any(0 <= int(round(cy)) < h and 0 <= int(round(cx)) < w and m[int(round(cy)),int(round(cx))] > 0 for m in existing_masks):
            continue
        best=None
        # High threshold + strong barrier closes almost every printed bubble in
        # one connected-components pass. Only escalate if that preferred variant
        # cannot yield a plausible text-bearing container. This is the cheap-first
        # equivalent of PanelCleaner's progressive fitting.
        for thr,dp in ((205,4),(185,4),(165,3),(145,3),(175,2)):
            key=(thr,dp)
            if key not in label_cache:
                label_cache[key]=_barrier_label_map(norm,thr,dp)
            m=_component_from_label_map(*label_cache[key],cx,cy)
            if m is None:
                continue
            area=int(cv2.countNonZero(m)); ar=area/page
            if area < 300 or ar > float(getattr(mask_cfg,"source_direct_text_seed_barrier_max_area_ratio",0.11)):
                continue
            bb=_bbox(m)
            if bb is None:
                continue
            bx0,by0,bx1,by1=bb
            if not (bx0 <= tx0 and by0 <= ty0 and bx1 >= tx1 and by1 >= ty1):
                continue
            ratio=area/max(1,text_area)
            if ratio > float(getattr(mask_cfg,"source_direct_text_seed_max_area_to_text_ratio",45.0)):
                continue
            vals=gray[m>0]
            white=float(np.mean(vals>225)); dark=float(np.mean(vals<180))
            if white < float(getattr(mask_cfg,"source_direct_text_seed_min_white_ratio",0.72)):
                continue
            if dark > float(getattr(mask_cfg,"source_direct_text_seed_max_dark_ratio",0.24)):
                continue
            # The seed must still contain printed ink. This rejects bright artwork
            # voids around faces/garments before they can become source hints.
            if dark < float(getattr(mask_cfg,"source_direct_text_seed_min_dark_ratio",0.02)):
                continue
            score=ratio + 4.0*(1.0-white) + 2.0*dark + 0.03*dp
            best=(score,m,bb,thr,dp,white,dark,ratio)
            break
        if best is None:
            continue
        _score,m,bb,thr,dp,white,dark,ratio=best
        # De-duplicate recovered masks and existing containers by overlap.
        if any(_iou(m,e)>0.70 for e in existing_masks):
            continue
        if any(_iou(m,b.mask)>0.70 for b in out if b.mask is not None):
            continue
        poly=mask_to_largest_polygon(m)
        if len(poly)<3:
            continue
        out.append(BubbleInstance(
            id=f"pseudo-text-barrier-{len(out):04d}", polygon=poly,
            confidence=float(np.clip(0.78 + min(0.15,g["count"]*0.004),0,0.94)),
            kind="speech", mask=m, safe_mask=m.copy(), block_ids=[],
            meta={
                "backend":"pseudo_text_barrier","source_only":True,"text_component_count":int(g["count"]),
                "text_bbox":[int(tx0),int(ty0),int(tx1),int(ty1)],"barrier_threshold":int(thr),
                "barrier_dilate_px":int(dp),"white_ratio":float(white),"dark_ratio":float(dark),
                "area_to_text_ratio":float(ratio),"container_bbox":[int(v) for v in bb],
            },
        ))
    return out


@register_provider("source_detector","sidecar")
def detect_source_sidecar(source: np.ndarray, mask_cfg: MaskReplaceConfig, bubble_cfg: BubbleConfig, existing=None, *, source_path: str | None = None) -> list[BubbleInstance]:
    if not source_path:
        return []
    from .bubbles import load_bubble_sidecar
    try:
        return load_bubble_sidecar(source, source_path, [], bubble_cfg)
    except Exception:
        return []


@register_provider("source_detector","ctd_sidecar")
def detect_source_ctd_sidecar(source: np.ndarray, mask_cfg: MaskReplaceConfig, bubble_cfg: BubbleConfig, existing=None, *, source_path: str | None = None, **kwargs) -> list[BubbleInstance]:
    """Explicit comic-text-detector/other external pixel-mask adapter alias."""
    rows = detect_source_sidecar(source, mask_cfg, bubble_cfg, existing=existing, source_path=source_path)
    for row in rows:
        row.meta = dict(row.meta or {})
        row.meta["provider_alias"] = "ctd_sidecar"
        row.meta["source_only"] = True
    return rows


@register_provider("source_detector","debubble_white")
def detect_source_debubble_white(source: np.ndarray, mask_cfg: MaskReplaceConfig, bubble_cfg: BubbleConfig, existing=None, **kwargs) -> list[BubbleInstance]:
    """Lightweight DebubbleBot-style white-contour SOURCE provider.

    Kept optional because the main direct-container finder is stricter. It exists
    as a pluggable low-cost fallback/debug provider and returns editable masks,
    never painted overlays or TARGET detections.
    """
    gray=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
    _, white=cv2.threshold(gray,250,255,cv2.THRESH_BINARY)
    contours,_=cv2.findContours(white,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    h,w=gray.shape; page=max(1,h*w); out=[]
    for c in contours:
        if len(c)<5: continue
        area=float(cv2.contourArea(c)); ar=area/page
        if ar<float(getattr(mask_cfg,"source_direct_min_area_ratio",0.0012)) or ar>float(getattr(mask_cfg,"source_direct_max_area_ratio",0.085)): continue
        try:
            _,(major,minor),_=cv2.fitEllipse(c)
        except cv2.error:
            continue
        ellipse_area=float(np.pi*major*minor/4.0)
        if ellipse_area<=0 or area/max(1.0,ellipse_area)<0.35: continue
        m=np.zeros((h,w),np.uint8); cv2.drawContours(m,[c],-1,255,cv2.FILLED)
        vals=gray[m>0]
        if vals.size<100 or float(np.mean(vals>225))<0.70: continue
        if any(e.mask is not None and e.mask.shape==m.shape and _iou(e.mask,m)>0.72 for e in (existing or [])): continue
        poly=mask_to_largest_polygon(m)
        if len(poly)<3: continue
        out.append(BubbleInstance(id=f"debubble-source-{len(out):04d}",polygon=poly,confidence=0.72,kind="speech",mask=m,safe_mask=m.copy(),block_ids=[],meta={"backend":"debubble_white","source_only":True,"editable_overlay":True}))
    return out


@register_provider("source_detector","mangalens")
def detect_source_mangalens(source: np.ndarray, mask_cfg: MaskReplaceConfig, bubble_cfg: BubbleConfig, existing=None, **kwargs) -> list[BubbleInstance]:
    if not getattr(bubble_cfg,"mangalens_model_path",None):
        return []
    from .bubbles import detect_mangalens_bubbles
    return detect_mangalens_bubbles(source, [], bubble_cfg)


@register_provider("source_detector","rtdetr_v2")
def detect_source_rtdetr_v2(source: np.ndarray, mask_cfg: MaskReplaceConfig, bubble_cfg: BubbleConfig, existing=None, **kwargs) -> list[BubbleInstance]:
    """Optional Apache-2.0 Comic Translate RT-DETR-v2 SOURCE adapter.

    The adapter is lazy, cached across pages, and refuses network downloads unless
    explicitly enabled. It returns bubble box hints in SOURCE-original pixels;
    TARGET is never sent to the model.
    """
    model_path=getattr(bubble_cfg,"rtdetr_model_path",None)
    allow=bool(getattr(bubble_cfg,"rtdetr_allow_model_downloads",False))
    repo=str(getattr(bubble_cfg,"rtdetr_repo_name","ogkalu/comic-text-and-bubble-detector"))
    if not model_path and not allow:
        return []
    try:
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
    except Exception as exc:
        logger.warning("RT-DETR-v2 source detector unavailable: %s",exc)
        return []
    source_ref=str(Path(model_path).expanduser().resolve()) if model_path else repo
    local_only=not allow
    device=str(getattr(bubble_cfg,"device","auto")).lower()
    if device=="auto":
        device="mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    imgsz=max(256,int(getattr(bubble_cfg,"rtdetr_imgsz",640)))
    key=(source_ref,device,imgsz)
    runtime=_RTDETR_RUNTIME.get(key)
    if runtime is None:
        try:
            proc=RTDetrImageProcessor.from_pretrained(
                source_ref, local_files_only=local_only,
                size={"width":imgsz,"height":imgsz},
            )
            model=RTDetrV2ForObjectDetection.from_pretrained(source_ref,local_files_only=local_only)
            model=model.to(device).eval()
            _RTDETR_RUNTIME[key]=(proc,model)
        except Exception as exc:
            logger.warning("RT-DETR-v2 model unavailable: %s",exc)
            return []
    else:
        proc,model=runtime
    rgb=cv2.cvtColor(source,cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil=Image.fromarray(rgb)
    inputs=proc(images=pil,return_tensors="pt")
    inputs={k:v.to(device) if hasattr(v,"to") else v for k,v in inputs.items()}
    with torch.inference_mode():
        outputs=model(**inputs)
    sizes=torch.tensor([pil.size[::-1]],device=device)
    result=proc.post_process_object_detection(outputs,target_sizes=sizes,threshold=float(getattr(bubble_cfg,"rtdetr_confidence",0.30)))[0]
    h,w=source.shape[:2]; out=[]
    for box,score,label in zip(result["boxes"],result["scores"],result["labels"]):
        # Upstream Comic Translate class 0 is bubble. Text classes are ignored;
        # they are never used to reconstruct a TARGET-side matching problem.
        if int(label.item()) != 0:
            continue
        x1,y1,x2,y2=[int(round(v)) for v in box.tolist()]
        x1=max(0,min(w-1,x1)); y1=max(0,min(h-1,y1)); x2=max(x1+1,min(w,x2)); y2=max(y1+1,min(h,y2))
        m=np.zeros((h,w),np.uint8); m[y1:y2,x1:x2]=255
        poly=mask_to_largest_polygon(m)
        if len(poly)<3: continue
        out.append(BubbleInstance(
            id=f"rtdetr-source-{len(out):04d}",polygon=poly,confidence=float(score.item()),kind="speech",
            mask=m,safe_mask=m.copy(),block_ids=[],meta={"backend":"rtdetr_v2","source_only":True,"bbox_only_hint":True,"runtime_cached":True},
        ))
    return out


@register_provider("source_detector","sam2")
def detect_source_sam2(source: np.ndarray, mask_cfg: MaskReplaceConfig, bubble_cfg: BubbleConfig, existing=None, **kwargs) -> list[BubbleInstance]:
    """Optional SAM 2 / 2.1 SOURCE-only geometry refiner.

    Compact SOURCE glyph groups provide positive point + box prompts. Returned
    masks must contain the seed text and pass the same paper/ink/size gates as the
    zero-model completion detector. This is an expensive *fallback only* and is
    never used on TARGET. No model is downloaded unless explicitly allowed.
    """
    checkpoint=getattr(bubble_cfg,"sam2_checkpoint",None)
    allow=bool(getattr(bubble_cfg,"sam2_allow_model_downloads",False))
    model_id=str(getattr(bubble_cfg,"sam2_model_id","facebook/sam2.1-hiera-tiny"))
    config_file=str(getattr(bubble_cfg,"sam2_config","configs/sam2.1/sam2.1_hiera_t.yaml"))
    if not checkpoint and not allow:
        return []
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        logger.warning("SAM2 source detector unavailable: %s",exc)
        return []
    device=str(getattr(bubble_cfg,"device","auto")).lower()
    if device=="auto":
        device="mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint:
        ckpt=str(Path(checkpoint).expanduser().resolve())
        key=(ckpt,config_file,device)
        predictor=_SAM2_RUNTIME.get(key)
        if predictor is None:
            try:
                from sam2.build_sam import build_sam2
                model=build_sam2(config_file,ckpt_path=ckpt,device=device,mode="eval")
                predictor=SAM2ImagePredictor(model)
                _SAM2_RUNTIME[key]=predictor
            except Exception as exc:
                logger.warning("SAM2 local checkpoint unavailable: %s",exc)
                return []
    else:
        # Official SAM2ImagePredictor.from_pretrained uses Hugging Face. We only
        # enter this branch when the user explicitly opted into model downloads.
        key=(model_id,"hf",device)
        predictor=_SAM2_RUNTIME.get(key)
        if predictor is None:
            try:
                predictor=SAM2ImagePredictor.from_pretrained(model_id,device=device)
                _SAM2_RUNTIME[key]=predictor
            except TypeError:
                # Older/current upstream variants may not expose ``device`` in
                # from_pretrained; move the underlying model after construction.
                try:
                    predictor=SAM2ImagePredictor.from_pretrained(model_id)
                    if hasattr(predictor,"model"):
                        predictor.model=predictor.model.to(device)
                    _SAM2_RUNTIME[key]=predictor
                except Exception as exc:
                    logger.warning("SAM2 HF model unavailable: %s",exc)
                    return []
            except Exception as exc:
                logger.warning("SAM2 HF model unavailable: %s",exc)
                return []
    gray=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape; page=max(1,h*w)
    groups=_cluster_text_components(gray,_compact_character_components(gray),int(getattr(mask_cfg,"source_direct_text_seed_min_components",6)))
    existing_masks=[b.mask for b in (existing or []) if b.mask is not None and b.mask.shape==gray.shape]
    rgb=cv2.cvtColor(source,cv2.COLOR_BGR2RGB)
    try:
        predictor.set_image(rgb)
    except Exception as exc:
        logger.warning("SAM2 set_image failed: %s",exc)
        return []
    out=[]; expand=float(getattr(bubble_cfg,"sam2_prompt_expand_ratio",0.85)); min_score=float(getattr(bubble_cfg,"sam2_min_score",0.62))
    for g in groups[:int(getattr(mask_cfg,"source_direct_text_seed_max_candidates",48))]:
        tx0,ty0,tx1,ty1=[int(v) for v in g["bbox"]]; cx,cy=g["centroid"]
        iy=int(np.clip(round(cy),0,h-1)); ix=int(np.clip(round(cx),0,w-1))
        if any(m[iy,ix]>0 for m in existing_masks):
            continue
        bw=max(1,tx1-tx0); bh=max(1,ty1-ty0)
        pad=max(12,int(round(max(bw,bh)*expand)))
        box=np.array([max(0,tx0-pad),max(0,ty0-pad),min(w-1,tx1+pad),min(h-1,ty1+pad)],dtype=np.float32)
        pts=np.array([[float(cx),float(cy)]],dtype=np.float32); labels=np.array([1],dtype=np.int32)
        try:
            with torch.inference_mode():
                masks,scores,_=predictor.predict(point_coords=pts,point_labels=labels,box=box,multimask_output=True)
        except Exception as exc:
            logger.debug("SAM2 prompt failed: %s",exc)
            continue
        best=None
        for raw,score in zip(masks,scores):
            score=float(score)
            if score < min_score: continue
            m=(np.asarray(raw)>0).astype(np.uint8)*255
            area=int(cv2.countNonZero(m)); ar=area/page
            if area<300 or ar>float(getattr(mask_cfg,"source_direct_text_seed_barrier_max_area_ratio",0.11)): continue
            bb=_bbox(m)
            if bb is None: continue
            bx0,by0,bx1,by1=bb
            if not (bx0<=tx0 and by0<=ty0 and bx1>=tx1 and by1>=ty1): continue
            vals=gray[m>0]; white=float(np.mean(vals>225)); dark=float(np.mean(vals<180))
            if white<float(getattr(mask_cfg,"source_direct_text_seed_min_white_ratio",0.72)): continue
            if not (float(getattr(mask_cfg,"source_direct_text_seed_min_dark_ratio",0.02)) <= dark <= float(getattr(mask_cfg,"source_direct_text_seed_max_dark_ratio",0.24))): continue
            text_area=max(1,bw*bh); ratio=area/text_area
            if ratio>float(getattr(mask_cfg,"source_direct_text_seed_max_area_to_text_ratio",45.0)): continue
            rank=score + 0.15*white - 0.12*min(1.0,ratio/45.0)
            if best is None or rank>best[0]: best=(rank,m,score,bb,white,dark,ratio)
        if best is None: continue
        _,m,score,bb,white,dark,ratio=best
        if any(_iou(m,e)>0.72 for e in existing_masks): continue
        if any(_iou(m,b.mask)>0.72 for b in out if b.mask is not None): continue
        poly=mask_to_largest_polygon(m)
        if len(poly)<3: continue
        out.append(BubbleInstance(
            id=f"sam2-source-{len(out):04d}",polygon=poly,confidence=score,kind="speech",mask=m,safe_mask=m.copy(),block_ids=[],
            meta={"backend":"sam2","source_only":True,"prompt":"text_seed_point_plus_box","runtime_cached":True,"text_component_count":int(g["count"]),"container_bbox":[int(v) for v in bb],"white_ratio":white,"dark_ratio":dark,"area_to_text_ratio":ratio},
        ))
    return out


def run_source_detector_chain(
    source: np.ndarray,
    mask_cfg: MaskReplaceConfig,
    bubble_cfg: BubbleConfig,
    *,
    existing: list[BubbleInstance] | None = None,
    source_path: str | None = None,
    allow_expensive: bool = False,
    only_expensive: bool = False,
) -> tuple[list[BubbleInstance], list[dict]]:
    """Run ordered providers and return de-duplicated SOURCE hints + audit log."""
    chain=list(getattr(mask_cfg,"source_direct_detector_chain",[]) or [])
    existing=list(existing or [])
    hints=[]; audit=[]
    for name in chain:
        name=str(name).lower().strip()
        expensive = name in {"rtdetr_v2","sam2","mangalens"}
        if only_expensive and not expensive:
            continue
        if expensive and not allow_expensive:
            audit.append({"provider":name,"status":"skipped_cost_gate","count":0})
            continue
        provider=REGISTRY.get("source_detector",name)
        if provider is None:
            audit.append({"provider":name,"status":"unregistered","count":0})
            continue
        try:
            rows=provider(source,mask_cfg,bubble_cfg,existing=existing+hints,source_path=source_path) or []
        except TypeError:
            rows=provider(source,mask_cfg,bubble_cfg,existing=existing+hints) or []
        except Exception as exc:
            logger.warning("Source detector provider %s failed: %s",name,exc)
            audit.append({"provider":name,"status":"error","error":str(exc),"count":0})
            continue
        kept=[]
        for b in rows:
            if b.mask is None: continue
            if any(e.mask is not None and e.mask.shape==b.mask.shape and _iou(e.mask,b.mask)>0.72 for e in existing+hints):
                continue
            kept.append(b)
        hints.extend(kept)
        audit.append({"provider":name,"status":"ok","count":len(kept)})
    return hints,audit
