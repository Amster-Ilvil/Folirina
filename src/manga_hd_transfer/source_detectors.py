from __future__ import annotations

"""SOURCE-only container detector providers for precise raster transfer.

No provider in this module is allowed to inspect TARGET text or create a
source/target bubble matching problem.  Providers only add container hypotheses
in canonical SOURCE-original coordinates.
"""

from pathlib import Path
import inspect
import logging
import os
import cv2
import numpy as np

from .config import BubbleConfig, MaskReplaceConfig
from .geometry import mask_to_largest_polygon
from .models import BubbleInstance
from .cache import PageStageCache, image_stage_signature
from .layout_evidence import collect_koharu_layout_evidence_cached, filter_candidates_by_layout_authority
from .detector_policy import (
    primary_detector, source_auxiliary_providers, expensive_provider,
    detector_strategy, STRATEGY_PRIMARY_ONLY, koharu_is_primary,
)
from .model_downloads import discovered_model_path
from .plugins import REGISTRY, register_provider

logger = logging.getLogger(__name__)

_RTDETR_RUNTIME: dict[tuple, tuple[object, object]] = {}
_SAM2_RUNTIME: dict[tuple, object] = {}



def _local_model_revision(path: str | Path) -> tuple:
    """Cheap revision token for optional local detector weights.

    Model objects intentionally stay resident across pages, but replacing a local
    checkpoint at the same path must not keep serving the stale in-memory model.
    HF-style directories are keyed by the metadata of their direct config/weight
    files; ordinary checkpoints use size + mtime.
    """
    p = Path(path).expanduser()
    try:
        p = p.resolve()
        if p.is_file():
            st = p.stat()
            return (str(p), int(st.st_size), int(st.st_mtime_ns))
        if p.is_dir():
            rows = []
            interesting = {'.json', '.bin', '.pt', '.pth', '.ckpt', '.safetensors'}
            for child in sorted(p.iterdir(), key=lambda x: x.name):
                if not child.is_file() or child.suffix.casefold() not in interesting:
                    continue
                st = child.stat()
                rows.append((child.name, int(st.st_size), int(st.st_mtime_ns)))
            st = p.stat()
            return (str(p), int(st.st_mtime_ns), tuple(rows))
    except OSError:
        pass
    return (str(p), 'missing')


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


def _semanticize_pseudo_text_barrier_mask(
    flood_mask: np.ndarray,
    text_bbox: tuple[int, int, int, int] | list[int],
    container_bbox: tuple[int, int, int, int] | list[int],
    *,
    barrier_dilate_px: int,
) -> tuple[np.ndarray, list[int]]:
    """Convert a holey white-flood candidate into Direct semantic authority.

    ``pseudo_text_barrier`` deliberately uses dark printed glyphs as barriers.
    Glyphs that touch a component edge therefore create open concavities in the
    flood mask. Direct must not use those concavities as a write restriction or
    it will clip exactly the first/last translated text column. Repair only the
    already-proven text seed, clip the repair to the accepted container bbox, and
    fill the resulting outer topology. No pixel can escape the vetted bbox.
    """
    m=(flood_mask>0).astype(np.uint8)*255
    if cv2.countNonZero(m)==0:
        return np.zeros_like(m), [0,0,0,0]
    h,w=m.shape[:2]
    bx0,by0,bx1,by1=[int(v) for v in container_bbox]
    bx0=max(0,min(w,bx0)); bx1=max(bx0,min(w,bx1))
    by0=max(0,min(h,by0)); by1=max(by0,min(h,by1))
    tx0,ty0,tx1,ty1=[int(v) for v in text_bbox]
    seed_pad=max(2,int(barrier_dilate_px)+2)
    sx0=max(bx0,tx0-seed_pad); sy0=max(by0,ty0-seed_pad)
    sx1=min(bx1,tx1+seed_pad); sy1=min(by1,ty1+seed_pad)
    repaired=m.copy()
    if sx1>sx0 and sy1>sy0:
        repaired[sy0:sy1,sx0:sx1]=255
    poly=mask_to_largest_polygon(repaired)
    if len(poly)<3:
        return np.zeros_like(m), [sx0,sy0,sx1,sy1]
    semantic=np.zeros_like(m)
    pts=np.asarray(poly,dtype=np.int32).reshape(-1,1,2)
    cv2.fillPoly(semantic,[pts],255)
    guard=np.zeros_like(m)
    guard[by0:by1,bx0:bx1]=255
    semantic=cv2.bitwise_and(semantic,guard)
    return semantic,[sx0,sy0,sx1,sy1]


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
        # The white flood intentionally treats dark printed glyphs as barriers.
        # When a glyph touches the flood boundary this creates a *concavity*, not
        # merely an enclosed hole, so filling the largest contour alone is not
        # enough: the first/last vertical text column can remain cut out.  Stitch
        # only the proven text seed back into the accepted container, clipped to
        # its already-vetted bbox, then recover/fill the outer topology.  This is
        # still SOURCE-only evidence and cannot grow beyond the candidate bbox.
        semantic_mask,semantic_seed_bbox=_semanticize_pseudo_text_barrier_mask(
            m,[tx0,ty0,tx1,ty1],bb,barrier_dilate_px=int(dp),
        )
        poly=mask_to_largest_polygon(semantic_mask)
        if len(poly)<3 or cv2.countNonZero(semantic_mask)<300:
            continue
        out.append(BubbleInstance(
            id=f"pseudo-text-barrier-{len(out):04d}", polygon=poly,
            confidence=float(np.clip(0.78 + min(0.15,g["count"]*0.004),0,0.94)),
            kind="speech", mask=semantic_mask, safe_mask=semantic_mask.copy(), block_ids=[],
            meta={
                "backend":"pseudo_text_barrier","source_only":True,"text_component_count":int(g["count"]),
                "text_bbox":[int(tx0),int(ty0),int(tx1),int(ty1)],"barrier_threshold":int(thr),
                "barrier_dilate_px":int(dp),"white_ratio":float(white),"dark_ratio":float(dark),
                "area_to_text_ratio":float(ratio),"container_bbox":[int(v) for v in bb],
                "semantic_mask_fills_text_holes":True,
                "semantic_mask_stitches_boundary_text":True,
                "semantic_seed_bbox":[int(v) for v in semantic_seed_bbox],
                "white_flood_pixels":int(cv2.countNonZero(m)),
                "semantic_pixels":int(cv2.countNonZero(semantic_mask)),
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


@register_provider("source_detector","koharu_layout")
def detect_source_koharu_layout(source: np.ndarray, mask_cfg: MaskReplaceConfig, bubble_cfg: BubbleConfig, existing=None, **kwargs) -> list[BubbleInstance]:
    """Optional Koharu Layout SOURCE-only bubble hints.

    This remains a pure layout detector: it contributes bubble geometry only and
    never performs OCR or TARGET-side matching. Open/SFX text can be consumed by
    other mode-specific routes through the shared layout_evidence service.
    """
    evidence = collect_koharu_layout_evidence_cached(
        source, bubble_cfg, role="source_direct", image_path=kwargs.get("source_path"),
        cache=kwargs.get("cache"), cache_enabled=bool(kwargs.get("cache_enabled", True)) and bool(getattr(bubble_cfg, "koharu_layout_cache_enabled", True)),
        stats=kwargs.get("stats"), allow_missing=True,
    )
    if not evidence.available:
        return []
    out = evidence.bubble_instances(backend_name="koharu_layout", source_only=True)
    existing_masks = [b.mask for b in (existing or []) if b.mask is not None]
    kept: list[BubbleInstance] = []
    for row in out:
        if row.mask is None:
            continue
        if any(mask is not None and mask.shape == row.mask.shape and _iou(mask, row.mask) > 0.72 for mask in existing_masks):
            continue
        kept.append(row)
    return kept


@register_provider("source_detector","mangalens")
def detect_source_mangalens(source: np.ndarray, mask_cfg: MaskReplaceConfig, bubble_cfg: BubbleConfig, existing=None, **kwargs) -> list[BubbleInstance]:
    resolved=discovered_model_path("mangalens",getattr(bubble_cfg,"mangalens_model_path",None))
    if resolved is None:
        return []
    bubble_cfg.mangalens_model_path=str(resolved)
    from .bubbles import detect_mangalens_bubbles
    return detect_mangalens_bubbles(source, [], bubble_cfg)


@register_provider("source_detector","ysg_obb")
def detect_source_ysg_obb(source: np.ndarray, mask_cfg: MaskReplaceConfig, bubble_cfg: BubbleConfig, existing=None, **kwargs) -> list[BubbleInstance]:
    """Optional YSG oriented-box auxiliary for open/rotated manga text.

    It contributes conservative candidate geometry only.  Koharu Authority can
    veto these candidates later; the adapter never expands an existing mask.
    """
    resolved = discovered_model_path("ysg_obb", getattr(bubble_cfg, "ysg_obb_model_path", None))
    if resolved is None:
        return []
    bubble_cfg.ysg_obb_model_path = str(resolved)
    try:
        from .vision_runtime import run_ysg_obb
        payload = run_ysg_obb(
            source, model_path=str(resolved),
            confidence=float(getattr(bubble_cfg, "ysg_obb_confidence", .25)),
            iou=float(getattr(bubble_cfg, "ysg_obb_iou", .50)),
            imgsz=max(640, int(getattr(bubble_cfg, "ysg_obb_imgsz", 1600))),
            device=str(getattr(bubble_cfg, "device", "auto")),
        )
    except Exception as exc:
        logger.warning("YSG YOLO OBB isolated source detector unavailable: %s", exc)
        return []
    h,w = source.shape[:2]; out=[]
    include_other = bool(getattr(bubble_cfg, "ysg_obb_include_other", False))
    for row in list(payload.get("items") or []):
        label = str(row.get("label") or "").lower().strip()
        if label == "other" and not include_other:
            continue
        poly_raw = list(row.get("polygon") or [])
        if len(poly_raw) < 3:
            continue
        pts = np.asarray(poly_raw, dtype=np.float32).reshape(-1,2)
        pts[:,0] = np.clip(pts[:,0], 0, max(0,w-1)); pts[:,1] = np.clip(pts[:,1], 0, max(0,h-1))
        m = np.zeros((h,w), np.uint8); cv2.fillPoly(m, [np.rint(pts).astype(np.int32)], 255)
        if cv2.countNonZero(m) <= 0:
            continue
        if any(e.mask is not None and e.mask.shape == m.shape and _iou(e.mask,m) > .78 for e in (existing or [])):
            continue
        polygon = [[float(x),float(y)] for x,y in pts.tolist()]
        region_kind = "free_text" if label in {"kuangwai","other"} else "bubble"
        out.append(BubbleInstance(
            id=f"ysg-obb-source-{len(out):04d}", polygon=polygon,
            confidence=float(row.get("confidence", .7)), kind="speech",
            mask=m, safe_mask=m.copy(), block_ids=[],
            meta={
                "backend":"ysg_obb", "source_only":True, "obb":True,
                "ysg_label":label, "ysg_class_id":int(row.get("class_id",-1)),
                "region_kind":region_kind, "open_text":bool(region_kind == "free_text"),
                "isolated_runtime":True, "device":str(payload.get("device") or "isolated"),
            },
        ))
    return out


@register_provider("source_detector","rtdetr_v2")
def detect_source_rtdetr_v2(source: np.ndarray, mask_cfg: MaskReplaceConfig, bubble_cfg: BubbleConfig, existing=None, **kwargs) -> list[BubbleInstance]:
    """RT-DETR-v2 SOURCE detector executed in the isolated Torch worker."""
    resolved=discovered_model_path("rtdetr_v2",getattr(bubble_cfg,"rtdetr_model_path",None))
    model_path=str(resolved) if resolved is not None else getattr(bubble_cfg,"rtdetr_model_path",None)
    if resolved is not None: bubble_cfg.rtdetr_model_path=str(resolved)
    allow=bool(getattr(bubble_cfg,"rtdetr_allow_model_downloads",False))
    repo=str(getattr(bubble_cfg,"rtdetr_repo_name","ogkalu/comic-text-and-bubble-detector"))
    if not model_path and not allow: return []
    source_ref=str(Path(model_path).expanduser().resolve()) if model_path else repo
    try:
        from .vision_runtime import run_rtdetr
        payload=run_rtdetr(
            source, model_ref=source_ref, local_only=not allow,
            confidence=float(getattr(bubble_cfg,"rtdetr_confidence",0.30)),
            imgsz=max(256,int(getattr(bubble_cfg,"rtdetr_imgsz",640))),
            device=str(getattr(bubble_cfg,"device","auto")),
        )
    except Exception as exc:
        logger.warning("RT-DETR-v2 isolated source detector unavailable: %s",exc); return []
    h,w=source.shape[:2]; out=[]; device=str(payload.get("device") or "isolated")
    for row in list(payload.get("items") or []):
        box=list(row.get("box") or [])
        if len(box)!=4: continue
        x1,y1,x2,y2=[int(round(float(v))) for v in box]
        x1=max(0,min(w-1,x1)); y1=max(0,min(h-1,y1)); x2=max(x1+1,min(w,x2)); y2=max(y1+1,min(h,y2))
        m=np.zeros((h,w),np.uint8); m[y1:y2,x1:x2]=255
        poly=mask_to_largest_polygon(m)
        if len(poly)<3: continue
        out.append(BubbleInstance(
            id=f"rtdetr-source-{len(out):04d}",polygon=poly,confidence=float(row.get("confidence",0.8)),kind="speech",
            mask=m,safe_mask=m.copy(),block_ids=[],meta={"backend":"rtdetr_v2","source_only":True,"bbox_only_hint":True,"runtime_cached":True,"isolated_runtime":True,"device":device},
        ))
    return out


@register_provider("source_detector","sam2")
def detect_source_sam2(source: np.ndarray, mask_cfg: MaskReplaceConfig, bubble_cfg: BubbleConfig, existing=None, **kwargs) -> list[BubbleInstance]:
    """SAM 2 / 2.1 SOURCE-only refiner executed in the isolated Torch worker."""
    resolved=discovered_model_path("sam2",getattr(bubble_cfg,"sam2_checkpoint",None))
    checkpoint=str(resolved) if resolved is not None else getattr(bubble_cfg,"sam2_checkpoint",None)
    if resolved is not None: bubble_cfg.sam2_checkpoint=str(resolved)
    allow=bool(getattr(bubble_cfg,"sam2_allow_model_downloads",False))
    if not checkpoint and not allow: return []
    model_id=str(getattr(bubble_cfg,"sam2_model_id","facebook/sam2.1-hiera-tiny"))
    config_file=str(getattr(bubble_cfg,"sam2_config","configs/sam2.1/sam2.1_hiera_t.yaml"))
    gray=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY); h,w=gray.shape; page=max(1,h*w)
    groups=_cluster_text_components(gray,_compact_character_components(gray),int(getattr(mask_cfg,"source_direct_text_seed_min_components",6)))
    existing_masks=[b.mask for b in (existing or []) if b.mask is not None and b.mask.shape==gray.shape]
    expand=float(getattr(bubble_cfg,"sam2_prompt_expand_ratio",0.85)); max_candidates=int(getattr(mask_cfg,"source_direct_text_seed_max_candidates",48))
    usable=[]; prompts=[]
    for g in groups[:max_candidates]:
        tx0,ty0,tx1,ty1=[int(v) for v in g["bbox"]]; cx,cy=g["centroid"]
        iy=int(np.clip(round(cy),0,h-1)); ix=int(np.clip(round(cx),0,w-1))
        if any(m[iy,ix]>0 for m in existing_masks): continue
        bw=max(1,tx1-tx0); bh=max(1,ty1-ty0); pad=max(12,int(round(max(bw,bh)*expand)))
        box=[max(0,tx0-pad),max(0,ty0-pad),min(w-1,tx1+pad),min(h-1,ty1+pad)]
        usable.append(g); prompts.append({"cx":float(cx),"cy":float(cy),"box":box})
    if not prompts: return []
    try:
        from .vision_runtime import run_sam2
        payload=run_sam2(
            source, checkpoint=checkpoint, model_id=model_id, config_file=config_file, allow_download=allow,
            prompts=prompts, device=str(getattr(bubble_cfg,"device","auto")),
        )
    except Exception as exc:
        logger.warning("SAM2 isolated source detector unavailable: %s",exc); return []
    min_score=float(getattr(bubble_cfg,"sam2_min_score",0.62)); out=[]; results=list(payload.get("results") or [])
    for g,candidates in zip(usable,results):
        tx0,ty0,tx1,ty1=[int(v) for v in g["bbox"]]; bw=max(1,tx1-tx0); bh=max(1,ty1-ty0); best=None
        for row in list(candidates or []):
            score=float(row.get("score",0.0))
            if score<min_score: continue
            pts=np.asarray(row.get("polygon") or [],dtype=np.int32)
            if len(pts)<3: continue
            m=np.zeros((h,w),np.uint8); cv2.fillPoly(m,[pts.reshape(-1,1,2)],255)
            area=int(cv2.countNonZero(m)); ar=area/page
            if area<300 or ar>float(getattr(mask_cfg,"source_direct_text_seed_barrier_max_area_ratio",0.11)): continue
            bb=_bbox(m)
            if bb is None: continue
            bx0,by0,bx1,by1=bb
            if not (bx0<=tx0 and by0<=ty0 and bx1>=tx1 and by1>=ty1): continue
            vals=gray[m>0]; white=float(np.mean(vals>225)); dark=float(np.mean(vals<180))
            if white<float(getattr(mask_cfg,"source_direct_text_seed_min_white_ratio",0.72)): continue
            if not (float(getattr(mask_cfg,"source_direct_text_seed_min_dark_ratio",0.02)) <= dark <= float(getattr(mask_cfg,"source_direct_text_seed_max_dark_ratio",0.24))): continue
            ratio=area/max(1,bw*bh)
            if ratio>float(getattr(mask_cfg,"source_direct_text_seed_max_area_to_text_ratio",45.0)): continue
            rank=score+0.15*white-0.12*min(1.0,ratio/45.0)
            if best is None or rank>best[0]: best=(rank,m,score,bb,white,dark,ratio)
        if best is None: continue
        _,m,score,bb,white,dark,ratio=best
        if any(_iou(m,e)>0.72 for e in existing_masks): continue
        if any(_iou(m,b.mask)>0.72 for b in out if b.mask is not None): continue
        poly=mask_to_largest_polygon(m)
        if len(poly)<3: continue
        out.append(BubbleInstance(
            id=f"sam2-source-{len(out):04d}",polygon=poly,confidence=score,kind="speech",mask=m,safe_mask=m.copy(),block_ids=[],
            meta={"backend":"sam2","source_only":True,"prompt":"text_seed_point_plus_box","runtime_cached":True,"isolated_runtime":True,"device":str(payload.get("device") or "isolated"),"text_component_count":int(g["count"]),"container_bbox":[int(v) for v in bb],"white_ratio":white,"dark_ratio":dark,"area_to_text_ratio":ratio},
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
    primary_only: bool = False,
    fallback_only: bool = False,
    cache: PageStageCache | None = None,
    cache_enabled: bool = True,
    stats: dict | None = None,
) -> tuple[list[BubbleInstance], list[dict]]:
    """Run the configured SOURCE primary detector, then eligible auxiliaries.

    v2.0.91 makes detector scheduling explicit instead of encoding it in a
    legacy ordered chain.  The primary detector is single-select and always
    runs first.  Selected auxiliaries are multi-select and are requested by the
    caller only when policy/plan state allows them.  When Koharu is the primary
    detector its ALLOW/PROTECT/UNKNOWN map remains the hard semantic authority;
    choosing another primary intentionally gives that detector first priority.
    """
    existing = list(existing or [])
    hints: list[BubbleInstance] = []
    audit: list[dict] = []
    primary = primary_detector(bubble_cfg)
    auxiliaries = source_auxiliary_providers(bubble_cfg, include_refiner=True)
    strategy = detector_strategy(bubble_cfg)

    authority = None
    if koharu_is_primary(bubble_cfg):
        authority = collect_koharu_layout_evidence_cached(
            source, bubble_cfg, role="source_direct_authority", image_path=source_path,
            cache=cache,
            cache_enabled=bool(cache_enabled) and bool(getattr(bubble_cfg, "koharu_layout_cache_enabled", True)),
            stats=stats, allow_missing=True,
        )

    def _run_provider(name: str, phase: str, *, authority_filter: bool) -> list[BubbleInstance]:
        # The page-flow has already executed/cached the selected primary. Reuse
        # that canonical result so MangaLens/RT-DETR are not inferred twice on
        # Direct pages. Unit/plugin calls without a page cache keep the registry
        # path below.
        if phase == "primary" and cache is not None and source_path and bool(cache_enabled):
            # Page-flow has already executed/cached the selected primary.  Read
            # the canonical stage directly instead of importing the page-level
            # bubble service back into this detector-domain module.  This keeps
            # detector orchestration one-way while preserving the exact cache
            # key used by ``primary_bubbles_cached``.
            try:
                primary_sig = image_stage_signature(
                    source_path, bubble_cfg,
                    {"role": "source", "detector_policy_primary": name, "primary_only_stage": True},
                )
                rows = cache.load_bubbles("primary_source", primary_sig)
                if rows is not None:
                    # v2.3.22: cached primary detections are real detector output,
                    # not audit-only metadata.  The v2.3.20 cycle-breaking refactor
                    # returned them from this helper before extending ``hints``; the
                    # outer chain therefore reported a 10-row cache hit while
                    # returning zero rows to Direct. Direct then fell through to a
                    # partial pseudo-text fallback and left Japanese pixels behind.
                    kept: list[BubbleInstance] = []
                    for bubble in rows:
                        if bubble.mask is None:
                            continue
                        if any(
                            old.mask is not None and old.mask.shape == bubble.mask.shape and _iou(old.mask, bubble.mask) > 0.72
                            for old in existing + hints
                        ):
                            continue
                        kept.append(bubble)
                    hints.extend(kept)
                    if stats is not None:
                        stats["primary_detector_source"] = f"hit:{name}:{len(kept)}"
                    audit.append({
                        "provider": name, "status": "ok_cached_primary",
                        "count": len(kept), "cached_count": len(rows), "phase": phase,
                    })
                    return kept
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("Canonical primary detector cache unavailable for %s: %s", name, exc)
        provider = REGISTRY.get("source_detector", name)
        if provider is None:
            audit.append({"provider": name, "status": "unregistered", "count": 0, "phase": phase})
            return []
        try:
            optional = {
                "existing": existing + hints,
                "source_path": source_path,
                "cache": cache,
                "cache_enabled": cache_enabled,
                "stats": stats,
            }
            try:
                signature = inspect.signature(provider)
                accepts_varkw = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in signature.parameters.values()
                )
                call_kwargs = optional if accepts_varkw else {
                    key: value for key, value in optional.items() if key in signature.parameters
                }
            except (TypeError, ValueError):
                call_kwargs = {"existing": existing + hints}
            rows = provider(source, mask_cfg, bubble_cfg, **call_kwargs) or []
        except Exception as exc:
            logger.warning("Source detector provider %s failed: %s", name, exc)
            audit.append({"provider": name, "status": "error", "error": str(exc), "count": 0, "phase": phase})
            return []

        authority_audit: list[dict] = []
        if authority_filter and authority is not None:
            filtered_rows: list[BubbleInstance] = []
            for candidate in rows:
                region_kind = str((getattr(candidate, "meta", {}) or {}).get("region_kind") or "bubble")
                accepted_rows, candidate_audit = filter_candidates_by_layout_authority(
                    [candidate], authority, source.shape[:2], region_kind=region_kind, cfg=mask_cfg,
                    allow_unknown=True, meta_key="koharu_layout_authority",
                )
                authority_audit.extend(candidate_audit)
                filtered_rows.extend(accepted_rows)
            rows = filtered_rows
        kept: list[BubbleInstance] = []
        for bubble in rows:
            if bubble.mask is None:
                continue
            if any(
                old.mask is not None and old.mask.shape == bubble.mask.shape and _iou(old.mask, bubble.mask) > 0.72
                for old in existing + hints
            ):
                continue
            kept.append(bubble)
        hints.extend(kept)
        audit.append({
            "provider": name,
            "status": "ok",
            "count": len(kept),
            "phase": phase,
            "primary": bool(phase == "primary"),
            "authority": "Koharu ALLOW/PROTECT/UNKNOWN" if authority_filter and authority is not None and authority.available else "primary-priority",
            "authority_rejected": sum(1 for row in authority_audit if not row.get("accepted")),
            "authority_unknown": sum(1 for row in authority_audit if row.get("state") == "UNKNOWN"),
        })
        return kept

    # Primary detector may itself be model-expensive; explicit user selection is
    # permission to run it and is therefore not blocked by the auxiliary cost gate.
    if not fallback_only and not only_expensive:
        _run_provider(primary, "primary", authority_filter=False)
        if primary_only:
            for name in auxiliaries:
                audit.append({
                    "provider": name,
                    "status": "disabled_primary_only" if strategy == STRATEGY_PRIMARY_ONLY else "deferred_until_primary_insufficient",
                    "count": 0,
                    "phase": "fallback",
                })
            return hints, audit

    if strategy == STRATEGY_PRIMARY_ONLY:
        return hints, audit

    for name in auxiliaries:
        if name == primary:
            continue
        expensive = expensive_provider(name)
        if only_expensive and not expensive:
            continue
        if fallback_only and not only_expensive and expensive:
            audit.append({"provider": name, "status": "deferred_expensive_fallback", "count": 0, "phase": "fallback"})
            continue
        if expensive and not allow_expensive:
            audit.append({"provider": name, "status": "skipped_cost_gate", "count": 0, "phase": "fallback"})
            continue
        # Only Koharu-as-primary has a semantic PROTECT map that may hard-veto
        # auxiliaries.  If another detector is selected as primary, it retains
        # first priority and Koharu (if selected as auxiliary) is additive only.
        _run_provider(name, "fallback", authority_filter=koharu_is_primary(bubble_cfg))

    return hints, audit

