from __future__ import annotations

"""Persistent JSONL worker for isolated MangaLens / RT-DETR-v2 / SAM2 inference."""

import json
import os
import inspect
from pathlib import Path
import sys
import traceback
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch

_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}


def _select_device(preferred: str) -> str:
    p = str(preferred or "auto").lower().strip()
    if p == "cpu": return "cpu"
    if p == "cuda": return "cuda" if torch.cuda.is_available() else "cpu"
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if p == "mps": return "mps" if mps else "cpu"
    if torch.cuda.is_available(): return "cuda"
    if mps: return "mps"
    return "cpu"


def _largest_polygon(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    c = max(contours, key=cv2.contourArea)
    if len(c) < 3: return []
    eps = max(0.5, cv2.arcLength(c, True) * 0.002)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in approx]


def _mangalens(image: np.ndarray, req: dict[str, Any]) -> dict[str, Any]:
    from ultralytics import YOLO
    model_path = str(Path(req["model_path"]).expanduser().resolve())
    st = Path(model_path).stat(); device = _select_device(str(req.get("device", "auto")))
    key = ("mangalens", model_path, int(st.st_size), int(st.st_mtime_ns))
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = YOLO(model_path); _MODEL_CACHE[key] = model
    kwargs = dict(source=image, conf=float(req.get("confidence", .35)), imgsz=int(req.get("imgsz", 1024)), verbose=False)
    try:
        results = model.predict(device=device, **kwargs)
    except Exception as first:
        if device == "cpu": raise
        results = model.predict(device="cpu", **kwargs); device = "cpu-fallback"
    if not results: return {"items": [], "device": device}
    result = results[0]; masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "xy", None) is None:
        return {"items": [], "device": device}
    confs = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "conf", None) is not None:
        confs = boxes.conf.detach().cpu().numpy().tolist()
    items=[]
    for i, xy in enumerate(masks.xy):
        pts=np.asarray(xy,dtype=np.float32)
        if len(pts)<3: continue
        items.append({"polygon":pts.tolist(),"confidence":float(confs[i]) if i<len(confs) else .8})
    return {"items":items,"device":device,"isolated_runtime":True}


def _ysg_obb(image: np.ndarray, req: dict[str, Any]) -> dict[str, Any]:
    """Run the optional YSG oriented-box detector in Folirina's isolated runtime.

    This adapter intentionally implements only generic Ultralytics inference and
    output normalization.  It does not copy manga-translator-ui post-processing.
    """
    from ultralytics import YOLO
    model_path = Path(str(req["model_path"])).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"YSG YOLO OBB model missing: {model_path}")
    st = model_path.stat()
    key = ("ysg-obb", str(model_path), int(st.st_size), int(st.st_mtime_ns))
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = YOLO(str(model_path), task="obb")
        _MODEL_CACHE[key] = model
    device = _select_device(str(req.get("device", "auto")))
    kwargs = {
        "source": image,
        "imgsz": max(640, int(req.get("imgsz", 1600))),
        "conf": max(0.01, min(0.99, float(req.get("confidence", .25)))),
        "iou": max(0.01, min(0.95, float(req.get("iou", .50)))),
        "device": device,
        "verbose": False,
    }
    try:
        results = model.predict(**kwargs)
    except Exception:
        if device != "cpu":
            device = "cpu"; kwargs["device"] = "cpu"
            results = model.predict(**kwargs)
        else:
            raise
    raw = results[0] if isinstance(results, list) and results else next(iter(results), None)
    obb = getattr(raw, "obb", None) if raw is not None else None
    if obb is None:
        return {"items": [], "device": device, "isolated_runtime": True}
    def _np(value):
        if value is None: return np.asarray([])
        if hasattr(value, "detach"): value = value.detach()
        if hasattr(value, "cpu"): value = value.cpu()
        if hasattr(value, "numpy"): value = value.numpy()
        return np.asarray(value)
    boxes = _np(getattr(obb, "xyxyxyxy", None)).astype(np.float32)
    scores = _np(getattr(obb, "conf", None)).astype(np.float32).reshape(-1)
    classes = _np(getattr(obb, "cls", None)).astype(np.int32).reshape(-1)
    if boxes.ndim == 2 and boxes.shape[1] == 8:
        boxes = boxes.reshape(-1, 4, 2)
    labels = {0:"balloon",1:"qipao",2:"fangkuai",3:"changfangtiao",4:"kuangwai",5:"other"}
    items=[]
    if boxes.ndim == 3 and boxes.shape[1:] == (4,2):
        for i, poly in enumerate(boxes):
            cls = int(classes[i]) if i < len(classes) else -1
            score = float(scores[i]) if i < len(scores) else 1.0
            xs = poly[:,0]; ys = poly[:,1]
            items.append({
                "polygon": [[float(x), float(y)] for x,y in poly.tolist()],
                "box": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                "confidence": score, "class_id": cls, "label": labels.get(cls, f"class-{cls}"),
            })
    return {"items": items, "device": device, "isolated_runtime": True}


def _rtdetr(image: np.ndarray, req: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
    model_ref=str(req["model_ref"]); local_only=bool(req.get("local_only",True)); imgsz=max(256,int(req.get("imgsz",640)))
    device=_select_device(str(req.get("device","auto")))
    key=("rtdetr",model_ref,local_only,imgsz,device)
    pair=_MODEL_CACHE.get(key)
    if pair is None:
        proc=RTDetrImageProcessor.from_pretrained(model_ref,local_files_only=local_only,size={"width":imgsz,"height":imgsz})
        model=RTDetrV2ForObjectDetection.from_pretrained(model_ref,local_files_only=local_only).to(device).eval()
        pair=(proc,model); _MODEL_CACHE[key]=pair
    proc,model=pair
    pil=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    inputs=proc(images=pil,return_tensors="pt"); inputs={k:v.to(device) if hasattr(v,"to") else v for k,v in inputs.items()}
    try:
        with torch.inference_mode(): outputs=model(**inputs)
    except Exception as first:
        if device=="cpu": raise
        # Rebuild CPU copy rather than moving a possibly half-failed MPS graph.
        ckey=("rtdetr",model_ref,local_only,imgsz,"cpu")
        pair=_MODEL_CACHE.get(ckey)
        if pair is None:
            cproc=RTDetrImageProcessor.from_pretrained(model_ref,local_files_only=local_only,size={"width":imgsz,"height":imgsz})
            cmodel=RTDetrV2ForObjectDetection.from_pretrained(model_ref,local_files_only=local_only).to("cpu").eval(); pair=(cproc,cmodel); _MODEL_CACHE[ckey]=pair
        proc,model=pair; device="cpu-fallback"
        inputs=proc(images=pil,return_tensors="pt")
        with torch.inference_mode(): outputs=model(**inputs)
    sizes=torch.tensor([pil.size[::-1]],device="cpu" if device=="cpu-fallback" else device)
    result=proc.post_process_object_detection(outputs,target_sizes=sizes,threshold=float(req.get("confidence",.30)))[0]
    items=[]
    for box,score,label in zip(result["boxes"],result["scores"],result["labels"]):
        if int(label.item())!=0: continue
        items.append({"box":[float(v) for v in box.detach().cpu().tolist()],"confidence":float(score.detach().cpu().item())})
    return {"items":items,"device":device,"isolated_runtime":True}


def _sam2_predictor(req: dict[str, Any], device: str):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    checkpoint=str(req.get("checkpoint") or "").strip(); model_id=str(req.get("model_id") or "facebook/sam2.1-hiera-tiny")
    config_file=str(req.get("config_file") or "configs/sam2.1/sam2.1_hiera_t.yaml")
    if checkpoint:
        p=Path(checkpoint).expanduser().resolve(); st=p.stat(); key=("sam2",str(p),int(st.st_size),int(st.st_mtime_ns),config_file,device)
        pred=_MODEL_CACHE.get(key)
        if pred is None:
            from sam2.build_sam import build_sam2
            model=build_sam2(config_file,ckpt_path=str(p),device=device,mode="eval")
            pred=SAM2ImagePredictor(model); _MODEL_CACHE[key]=pred
        return pred
    if not bool(req.get("allow_download",False)):
        raise RuntimeError("SAM2 checkpoint missing and remote download disabled")
    key=("sam2-hf",model_id,device); pred=_MODEL_CACHE.get(key)
    if pred is None:
        try: pred=SAM2ImagePredictor.from_pretrained(model_id,device=device)
        except TypeError:
            pred=SAM2ImagePredictor.from_pretrained(model_id)
            if hasattr(pred,"model"): pred.model=pred.model.to(device)
        _MODEL_CACHE[key]=pred
    return pred


def _sam2(image: np.ndarray, req: dict[str, Any]) -> dict[str, Any]:
    device=_select_device(str(req.get("device","auto")))
    try:
        predictor=_sam2_predictor(req,device); predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    except Exception:
        if device=="cpu": raise
        device="cpu"; predictor=_sam2_predictor(req,device); predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    results=[]
    for prompt in list(req.get("prompts") or []):
        point=np.asarray([[float(prompt["cx"]),float(prompt["cy"])]],dtype=np.float32)
        labels=np.asarray([1],dtype=np.int32); box=np.asarray(prompt["box"],dtype=np.float32)
        try:
            with torch.inference_mode(): masks,scores,_=predictor.predict(point_coords=point,point_labels=labels,box=box,multimask_output=True)
        except Exception:
            results.append([]); continue
        candidates=[]
        for raw, score in zip(masks,scores):
            m=(np.asarray(raw)>0).astype(np.uint8)*255; poly=_largest_polygon(m)
            if len(poly)>=3: candidates.append({"polygon":poly,"score":float(score)})
        results.append(candidates)
    return {"results":results,"device":device,"isolated_runtime":True}


def _import_file_module(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _koharu_mask_geometry(mask: np.ndarray, image_shape: tuple[int, int]) -> tuple[list[list[float]], int]:
    """Convert a native Koharu instance mask to original-image geometry.

    Only polygon coordinates are scaled; the full-resolution binary mask is
    never materialized.  This keeps memory proportional to the RF-DETR mask
    size even for multi-megapixel photographed SOURCE pages.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if m.ndim != 2 or m.size == 0:
        return [], 0
    mh, mw = m.shape[:2]
    native_polygon = _largest_polygon(m)
    if not native_polygon:
        return [], 0
    ih, iw = int(image_shape[0]), int(image_shape[1])
    if (mh, mw) == (ih, iw):
        return [[float(x), float(y)] for x, y in native_polygon], int(cv2.countNonZero(m))
    sx = float(iw) / max(1.0, float(mw))
    sy = float(ih) / max(1.0, float(mh))
    polygon = [[float(x) * sx, float(y) * sy] for x, y in native_polygon]
    area = int(round(cv2.countNonZero(m) * sx * sy))
    return polygon, area


def _koharu_layout(image: np.ndarray, req: dict[str, Any]) -> dict[str, Any]:
    """Run Koharu Layout without letting RF-DETR upscale every query mask to a huge scan.

    ``rfdetr.predict`` post-processes instance masks to the *input PIL image size*.
    Passing a 4096x5824 photographed SOURCE therefore asks PyTorch to materialize
    all query masks at 24 MP, which can require >10 GiB before Folirina ever sees
    the masks.  Folirina now caps only the image supplied to RF-DETR, lets the
    model post-process in that bounded coordinate system, and scales boxes/
    polygons back to canonical original-image coordinates afterwards.
    """
    from PIL import Image
    root = Path(req["model_dir"]).expanduser().resolve()
    loader_path = root / "load_model.py"
    weights = root / "model.safetensors"
    if not loader_path.is_file() or not weights.is_file():
        raise FileNotFoundError("Koharu Layout model directory is incomplete")
    st = weights.stat(); key = ("koharu-layout", str(root), int(st.st_size), int(st.st_mtime_ns))
    model = _MODEL_CACHE.get(key)
    if model is None:
        loader = _import_file_module(loader_path, "folirina_koharu_layout_loader")
        load_fn = getattr(loader, "load_model", None)
        if not callable(load_fn):
            raise RuntimeError("Koharu Layout load_model.py has no load_model()")
        model = load_fn(str(weights))
        _MODEL_CACHE[key] = model

    orig_h, orig_w = image.shape[:2]
    model_shape = max(512, int(req.get("shape", 1152)))
    # RF-DETR already performs its network inference at ``model_shape``.  There
    # is no extra semantic detail in post-processing masks to a larger PIL
    # canvas, so cap the PIL input at the same 1152-pixel model scale.  This
    # avoids a ~14 GiB query-mask upsample on 4096x5824 SOURCE scans.
    postprocess_max_side = max(model_shape, int(req.get("postprocess_max_side", 1152)))
    downscale_trigger_side = max(postprocess_max_side, int(req.get("postprocess_downscale_trigger_side", 2048)))
    original_max_side = max(orig_h, orig_w)
    scale_down = (min(1.0, float(postprocess_max_side) / max(1.0, float(original_max_side)))
                  if original_max_side > downscale_trigger_side else 1.0)
    if scale_down < 0.999:
        infer_w = max(1, int(round(orig_w * scale_down)))
        infer_h = max(1, int(round(orig_h * scale_down)))
        infer_image = cv2.resize(image, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
    else:
        infer_image = image
        infer_h, infer_w = orig_h, orig_w

    rgb = cv2.cvtColor(infer_image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    threshold = min(
        float(req.get("text_threshold", .25)), float(req.get("sfx_threshold", .20)),
        float(req.get("bubble_threshold", .50)), float(req.get("panel_threshold", .50)),
    )
    kwargs = {
        "threshold": threshold,
        "shape": (model_shape, model_shape),
        "include_source_image": False,
    }
    try:
        det = model.predict(pil, **kwargs)
    except TypeError:
        kwargs.pop("include_source_image", None)
        det = model.predict(pil, **kwargs)

    xyxy = np.asarray(getattr(det, "xyxy", []), dtype=np.float32)
    conf = np.asarray(getattr(det, "confidence", np.ones(len(xyxy))), dtype=np.float32)
    class_id = np.asarray(getattr(det, "class_id", np.zeros(len(xyxy))), dtype=np.int32)
    masks = getattr(det, "mask", None)
    labels = {0:"text", 1:"onomatopoeia", 2:"bubble", 3:"panel"}
    thresholds = {
        "text": float(req.get("text_threshold",.25)),
        "onomatopoeia": float(req.get("sfx_threshold",.20)),
        "bubble": float(req.get("bubble_threshold",.50)),
        "panel": float(req.get("panel_threshold",.50)),
    }
    sx = float(orig_w) / max(1.0, float(infer_w))
    sy = float(orig_h) / max(1.0, float(infer_h))
    area_scale = sx * sy
    items=[]
    for i, box in enumerate(xyxy):
        label=labels.get(int(class_id[i]), f"class-{int(class_id[i])}")
        score=float(conf[i]) if i<len(conf) else 1.0
        if score < thresholds.get(label, threshold):
            continue
        polygon=[]; area=0
        if masks is not None and i < len(masks):
            polygon, area = _koharu_mask_geometry(np.asarray(masks[i]), (infer_h, infer_w))
            if polygon and (infer_h, infer_w) != (orig_h, orig_w):
                polygon = [[float(x) * sx, float(y) * sy] for x, y in polygon]
                area = int(round(float(area) * area_scale))
        raw_box = [float(v) for v in box.tolist()]
        scaled_box = [raw_box[0] * sx, raw_box[1] * sy, raw_box[2] * sx, raw_box[3] * sy]
        if not polygon:
            x0,y0,x1,y1=scaled_box
            polygon=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
            area=int(max(0,x1-x0)*max(0,y1-y0))
        items.append({"label":label,"confidence":score,"box":scaled_box,"polygon":polygon,"area":area})
    return {
        "items":items,
        "device":str(req.get("device","auto")),
        "isolated_runtime":True,
        "original_shape":[int(orig_h), int(orig_w)],
        "inference_shape":[int(infer_h), int(infer_w)],
        "postprocess_max_side":int(postprocess_max_side),
        "postprocess_downscale_trigger_side":int(downscale_trigger_side),
        "downscaled_for_postprocess":bool(scale_down < 0.999),
    }


def _manga_ocr(image: np.ndarray, req: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel
    root=str(Path(req["model_dir"]).expanduser().resolve()); device=_select_device(str(req.get("device","auto")))
    key=("manga-ocr",root,device)
    triple=_MODEL_CACHE.get(key)
    if triple is None:
        proc=ViTImageProcessor.from_pretrained(root,local_files_only=True)
        tok=AutoTokenizer.from_pretrained(root,local_files_only=True)
        model=VisionEncoderDecoderModel.from_pretrained(root,local_files_only=True).to(device).eval()
        triple=(proc,tok,model); _MODEL_CACHE[key]=triple
    proc,tok,model=triple
    pil=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    pixels=proc(images=pil,return_tensors="pt").pixel_values.to(device)
    with torch.inference_mode():
        out=model.generate(pixels,max_new_tokens=int(req.get("max_new_tokens",128)))
    text=tok.batch_decode(out,skip_special_tokens=True)[0]
    return {"text":str(text).strip(),"confidence":1.0,"device":device,"isolated_runtime":True}


def _baberu_ocr(image: np.ndarray, req: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    root=Path(req["model_dir"]).expanduser().resolve()
    script=root/"onnx_infer.py"; vision=root/"onnx"/"vision_int4.onnx"
    if not script.is_file() or not vision.is_file():
        raise FileNotFoundError("Baberu OCR ONNX model directory is incomplete")
    st=vision.stat(); key=("baberu-onnx",str(root),int(st.st_size),int(st.st_mtime_ns))
    ocr=_MODEL_CACHE.get(key)
    if ocr is None:
        mod=_import_file_module(script,"folirina_baberu_onnx")
        cls=getattr(mod,"BaberuOnnxOCR",None)
        if cls is None:
            raise RuntimeError("Baberu onnx_infer.py has no BaberuOnnxOCR")
        ocr=cls(root/"onnx",root/"tokenizer","vision_int4.onnx"); _MODEL_CACHE[key]=ocr
    pil=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    text=ocr(pil,max_new_tokens=int(req.get("max_new_tokens",128)))
    return {"text":str(text).strip(),"confidence":1.0,"device":"onnx-cpu","isolated_runtime":True}


def _load_48px_state_dict(model_path: Path) -> dict[str, Any]:
    kwargs = {"map_location": "cpu"}
    try:
        supports = "weights_only" in inspect.signature(torch.load).parameters
    except Exception:
        supports = True
    state = None; errors=[]
    attempts = (True, False) if supports else (None,)
    for weights_only in attempts:
        try:
            if weights_only is None:
                state = torch.load(model_path, **kwargs)
            else:
                state = torch.load(model_path, weights_only=weights_only, **kwargs)
            break
        except Exception as exc:
            errors.append(str(exc))
    if state is None:
        raise RuntimeError("48px checkpoint 无法反序列化：" + " | ".join(errors[-2:]))
    if isinstance(state, dict):
        for name in ("state_dict", "model_state_dict", "model"):
            nested = state.get(name)
            if isinstance(nested, dict) and nested:
                state = nested; break
    if not isinstance(state, dict) or not state:
        raise RuntimeError("48px checkpoint 格式异常")
    normalized={}
    for raw_key,value in state.items():
        key=str(raw_key); changed=True
        while changed:
            changed=False
            for prefix in ("model.","module.","_orig_mod.","net."):
                if key.startswith(prefix):
                    key=key[len(prefix):]; changed=True
        normalized[key]=value
    return normalized


def _ocr48px(image: np.ndarray, req: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    try:
        from ocr48px_runtime import load_ocr_class
    except Exception as exc:
        raise RuntimeError(f"48px runtime import failed: {exc}") from exc
    root=Path(req["model_dir"]).expanduser().resolve(); device=_select_device(str(req.get("device","auto")))
    model_file=root/"ocr_ar_48px.ckpt"; st=model_file.stat()
    key=("ocr48px",str(root),int(st.st_size),int(st.st_mtime_ns),device)
    bundle=_MODEL_CACHE.get(key)
    if bundle is None:
        OCR, model_path, dict_path = load_ocr_class(root)
        with dict_path.open("r",encoding="utf-8-sig") as fh:
            dictionary=[line.rstrip("\r\n") for line in fh]
        if len(dictionary)<1000 or dictionary[1:3] != ["<S>","</S>"]:
            raise RuntimeError("48px 字符表与官方模型不匹配")
        model=OCR(dictionary,768)
        model.load_state_dict(_load_48px_state_dict(model_path),strict=True)
        model.eval()
        try:
            model.to(device)
        except Exception:
            device="cpu"; model.to(device)
        bundle=(model,dictionary,device); _MODEL_CACHE[key]=bundle
    model,dictionary,device=bundle

    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    pil=Image.fromarray(rgb)
    orientation="horizontal"
    if pil.height > pil.width * 1.20:
        pil=pil.transpose(Image.Transpose.ROTATE_90); orientation="vertical-rotated-ccw"
    ratio=pil.width/max(1.0,float(pil.height)); width=max(4,int(round(ratio*48)))
    resized=pil.resize((width,48),Image.Resampling.LANCZOS)
    array=np.asarray(resized,dtype=np.uint8)
    max_width=4*(width+7)//4
    region=np.zeros((1,48,max_width,3),dtype=np.uint8); region[0,:,:width,:]=array
    tensor=(torch.from_numpy(region).float()-127.5)/127.5
    tensor=tensor.permute(0,3,1,2).to(device)
    kwargs={"beams_k":max(1,min(8,int(req.get("beams_k",5)))),"max_seq_length":max(8,min(384,int(req.get("max_seq_length",255))))}
    try:
        with torch.inference_mode(): recognized=model.infer_beam_batch_tensor(tensor,[width],**kwargs)
    except RuntimeError:
        if device=="cpu": raise
        device="cpu"; model.to("cpu"); tensor=tensor.to("cpu")
        with torch.inference_mode(): recognized=model.infer_beam_batch_tensor(tensor,[width],**kwargs)
    if len(recognized)!=1:
        raise RuntimeError("48px OCR 返回数量异常")
    indices, probability, *_ = recognized[0]
    chars=[]
    for raw_index in indices:
        idx=int(raw_index)
        if idx<0 or idx>=len(dictionary):
            raise RuntimeError(f"48px OCR 返回越界字符索引：{idx}")
        token=dictionary[idx]
        if token=="<S>": continue
        if token=="</S>": break
        chars.append(" " if token=="<SP>" else token)
    return {
        "text":"".join(chars).strip(), "confidence":float(probability or 0.0),
        "device":device, "input_width":width, "orientation":orientation,
        "isolated_runtime":True,
    }


def main() -> int:
    for raw in sys.stdin:
        raw=raw.strip()
        if not raw: continue
        rid=-1
        try:
            req=json.loads(raw); rid=int(req.get("request_id",-1))
            if req.get("op")=="shutdown": return 0
            image=np.load(str(req["image"]),allow_pickle=False)
            op=str(req.get("op"))
            if op=="mangalens": payload=_mangalens(image,req)
            elif op=="ysg_obb": payload=_ysg_obb(image,req)
            elif op=="rtdetr_v2": payload=_rtdetr(image,req)
            elif op=="sam2": payload=_sam2(image,req)
            elif op=="koharu_layout": payload=_koharu_layout(image,req)
            elif op=="manga_ocr": payload=_manga_ocr(image,req)
            elif op=="baberu_ocr": payload=_baberu_ocr(image,req)
            elif op=="ocr48px": payload=_ocr48px(image,req)
            else: raise ValueError(f"unsupported op: {op}")
            row={"ok":True,"request_id":rid,**payload}
        except Exception as exc:
            row={"ok":False,"request_id":rid,"error":f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-7000:]}"}
        print(json.dumps(row,ensure_ascii=False),flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())
