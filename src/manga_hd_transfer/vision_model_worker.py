from __future__ import annotations

"""Persistent JSONL worker for isolated MangaLens / RT-DETR-v2 / SAM2 inference."""

import json
import os
import inspect
import gc
from pathlib import Path
import sys
import traceback
import time
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch

try:
    from .accelerator_policy import apply_torch_worker_policy, autocast_context, detect_apple_hardware, env_precision, resolve_precision
    from .onnx_coreml import patch_inference_session_for_coreml, probe_coreml
except ImportError:  # executed directly by the isolated worker process
    from accelerator_policy import apply_torch_worker_policy, autocast_context, detect_apple_hardware, env_precision, resolve_precision
    from onnx_coreml import patch_inference_session_for_coreml, probe_coreml

apply_torch_worker_policy(torch)

_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}


def _trim_runtime_cache(*, unload_models: bool = False) -> dict[str, Any]:
    """Trim allocator state, optionally unloading resident model objects."""
    before = len(_MODEL_CACHE)
    if unload_models:
        _MODEL_CACHE.clear()
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()
    except Exception:
        pass
    return {
        "trimmed": True,
        "models_unloaded": bool(unload_models),
        "model_cache_entries_before": int(before),
        "model_cache_entries_released": int(before if unload_models else 0),
    }


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
    precision = resolve_precision(device, req.get("precision", env_precision()))
    key = ("mangalens", model_path, int(st.st_size), int(st.st_mtime_ns))
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = YOLO(model_path); _MODEL_CACHE[key] = model
    kwargs = dict(source=image, conf=float(req.get("confidence", .35)), imgsz=int(req.get("imgsz", 1024)), verbose=False)
    if device == "mps":
        kwargs["half"] = precision == "fp16"
    try:
        results = model.predict(device=device, **kwargs)
    except Exception as first:
        if device == "cpu": raise
        if device == "mps" and precision != "fp32":
            try:
                kwargs["half"] = False
                results = model.predict(device=device, **kwargs)
                precision = "fp32"
            except Exception:
                results = model.predict(device="cpu", **kwargs); device = "cpu-fallback"; precision = "fp32"
        else:
            results = model.predict(device="cpu", **kwargs); device = "cpu-fallback"; precision = "fp32"
    if not results: return {"items": [], "device": device, "precision": precision}
    result = results[0]; masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "xy", None) is None:
        return {"items": [], "device": device, "precision": precision}
    confs = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "conf", None) is not None:
        confs = boxes.conf.detach().cpu().numpy().tolist()
    items=[]
    for i, xy in enumerate(masks.xy):
        pts=np.asarray(xy,dtype=np.float32)
        if len(pts)<3: continue
        items.append({"polygon":pts.tolist(),"confidence":float(confs[i]) if i<len(confs) else .8})
    return {"items":items,"device":device,"precision":precision,"isolated_runtime":True}


def _ysg_obb(image: np.ndarray, req: dict[str, Any]) -> dict[str, Any]:
    """Run YSG checkpoints without confusing HBB and OBB geometry.

    The historical YSG 2.0 family shipped both an HBB checkpoint and an
    explicit OBB checkpoint. Older Folirina forced ``task=obb`` and then read
    only ``result.obb``; an HBB checkpoint therefore looked healthy but yielded
    no candidates. We now let Ultralytics infer the checkpoint task and
    normalize either output to a four-point polygon.
    """
    from ultralytics import YOLO
    model_path = Path(str(req["model_path"])).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"YSG YOLO model missing: {model_path}")
    st = model_path.stat()
    key = ("ysg-yolo", str(model_path), int(st.st_size), int(st.st_mtime_ns))
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = YOLO(str(model_path))
        task = str(getattr(model, "task", "") or "").lower()
        if task and task not in {"detect", "obb"}:
            raise RuntimeError(f"YSG checkpoint task is {task!r}; expected detect/HBB or OBB")
        _MODEL_CACHE[key] = model
    device = _select_device(str(req.get("device", "auto")))
    precision = resolve_precision(device, req.get("precision", env_precision()))
    kwargs = {
        "source": image,
        "imgsz": max(640, int(req.get("imgsz", 1600))),
        "conf": max(0.01, min(0.99, float(req.get("confidence", .25)))),
        "iou": max(0.01, min(0.95, float(req.get("iou", .50)))),
        "device": device,
        "verbose": False,
    }
    if device == "mps":
        kwargs["half"] = precision == "fp16"
    try:
        results = model.predict(**kwargs)
    except Exception as first:
        if device == "mps" and precision != "fp32":
            try:
                kwargs["half"] = False; results = model.predict(**kwargs); precision = "fp32"
            except Exception:
                device = "cpu"; precision = "fp32"; kwargs["device"] = "cpu"; kwargs.pop("half", None); results = model.predict(**kwargs)
        elif device != "cpu":
            device = "cpu"; precision = "fp32"; kwargs["device"] = "cpu"; kwargs.pop("half", None); results = model.predict(**kwargs)
        else:
            raise first
    raw = results[0] if isinstance(results, list) and results else next(iter(results), None)

    def _np(value):
        if value is None: return np.asarray([])
        if hasattr(value, "detach"): value = value.detach()
        if hasattr(value, "cpu"): value = value.cpu()
        if hasattr(value, "numpy"): value = value.numpy()
        return np.asarray(value)

    labels = {0:"balloon",1:"qipao",2:"fangkuai",3:"changfangtiao",4:"kuangwai",5:"other"}
    boxes = np.asarray([], dtype=np.float32); scores = np.asarray([], dtype=np.float32); classes = np.asarray([], dtype=np.int32)
    geometry_kind = "none"
    if raw is not None and getattr(raw, "obb", None) is not None:
        obb = raw.obb
        boxes = _np(getattr(obb, "xyxyxyxy", None)).astype(np.float32)
        scores = _np(getattr(obb, "conf", None)).astype(np.float32).reshape(-1)
        classes = _np(getattr(obb, "cls", None)).astype(np.int32).reshape(-1)
        if boxes.ndim == 2 and boxes.shape[1] == 8:
            boxes = boxes.reshape(-1, 4, 2)
        geometry_kind = "obb"
    elif raw is not None and getattr(raw, "boxes", None) is not None:
        hbb = raw.boxes
        xyxy = _np(getattr(hbb, "xyxy", None)).astype(np.float32)
        scores = _np(getattr(hbb, "conf", None)).astype(np.float32).reshape(-1)
        classes = _np(getattr(hbb, "cls", None)).astype(np.int32).reshape(-1)
        if xyxy.ndim == 1 and xyxy.size == 4:
            xyxy = xyxy.reshape(1, 4)
        if xyxy.ndim == 2 and xyxy.shape[1] == 4:
            x0,y0,x1,y1 = (xyxy[:,i] for i in range(4))
            boxes = np.stack([np.stack([x0,y0],1), np.stack([x1,y0],1), np.stack([x1,y1],1), np.stack([x0,y1],1)],1)
        geometry_kind = "hbb"

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
                "geometry_kind": geometry_kind,
            })
    model_task = str(getattr(model, "task", "") or geometry_kind or "unknown").lower()
    return {
        "items": items, "device": device, "precision": precision, "isolated_runtime": True,
        "model_task": model_task, "geometry_kind": geometry_kind,
        "legacy_hbb_compat": geometry_kind == "hbb",
    }


def _rtdetr(image: np.ndarray, req: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
    model_ref=str(req["model_ref"]); local_only=bool(req.get("local_only",True)); imgsz=max(256,int(req.get("imgsz",640)))
    device=_select_device(str(req.get("device","auto")))
    precision=resolve_precision(device, req.get("precision", env_precision()))
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
        with torch.inference_mode():
            with autocast_context(torch, device, precision):
                outputs=model(**inputs)
    except Exception as first:
        # MPS FP16 is the new M6 fast path, but it must never change the
        # historical behavior of a model with an unsupported low-precision op.
        # Retry the exact same MPS graph in FP32 before touching CPU.
        same_device_ok = False
        if device == "mps" and precision != "fp32":
            try:
                with torch.inference_mode():
                    outputs=model(**inputs)
                precision = "fp32"
                same_device_ok = True
            except Exception as second:
                first = second

        if not same_device_ok:
            if device == "cpu":
                raise first
            ckey=("rtdetr",model_ref,local_only,imgsz,"cpu")
            pair=_MODEL_CACHE.get(ckey)
            if pair is None:
                cproc=RTDetrImageProcessor.from_pretrained(
                    model_ref, local_files_only=local_only, size={"width":imgsz,"height":imgsz}
                )
                cmodel=RTDetrV2ForObjectDetection.from_pretrained(
                    model_ref, local_files_only=local_only
                ).to("cpu").eval()
                pair=(cproc,cmodel); _MODEL_CACHE[ckey]=pair
            proc,model=pair; device="cpu-fallback"; precision="fp32"
            inputs=proc(images=pil,return_tensors="pt")
            with torch.inference_mode():
                outputs=model(**inputs)
    sizes=torch.tensor([pil.size[::-1]],device="cpu" if device=="cpu-fallback" else device)
    result=proc.post_process_object_detection(outputs,target_sizes=sizes,threshold=float(req.get("confidence",.30)))[0]
    items=[]
    for box,score,label in zip(result["boxes"],result["scores"],result["labels"]):
        if int(label.item())!=0: continue
        items.append({"box":[float(v) for v in box.detach().cpu().tolist()],"confidence":float(score.detach().cpu().item())})
    return {"items":items,"device":device,"precision":precision,"isolated_runtime":True}


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
    precision=resolve_precision(device, req.get("precision", env_precision()))
    try:
        predictor=_sam2_predictor(req,device)
        with torch.inference_mode():
            with autocast_context(torch, device, precision):
                predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    except Exception as first:
        if device == "mps" and precision != "fp32":
            try:
                with torch.inference_mode():
                    predictor=_sam2_predictor(req,device)
                    predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                precision="fp32"
            except Exception:
                device="cpu"; precision="fp32"; predictor=_sam2_predictor(req,device); predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        elif device=="cpu":
            raise first
        else:
            device="cpu"; precision="fp32"; predictor=_sam2_predictor(req,device); predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    results=[]
    for prompt in list(req.get("prompts") or []):
        point=np.asarray([[float(prompt["cx"]),float(prompt["cy"])]],dtype=np.float32)
        labels=np.asarray([1],dtype=np.int32); box=np.asarray(prompt["box"],dtype=np.float32)
        try:
            with torch.inference_mode():
                with autocast_context(torch, device, precision):
                    masks,scores,_=predictor.predict(point_coords=point,point_labels=labels,box=box,multimask_output=True)
        except Exception:
            results.append([]); continue
        candidates=[]
        for raw, score in zip(masks,scores):
            m=(np.asarray(raw)>0).astype(np.uint8)*255; poly=_largest_polygon(m)
            if len(poly)>=3: candidates.append({"polygon":poly,"score":float(score)})
        results.append(candidates)
    return {"results":results,"device":device,"precision":precision,"isolated_runtime":True}


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
    effective_device = _select_device(str(req.get("device", "auto")))
    precision = resolve_precision(effective_device, req.get("precision", env_precision()))
    optimize_weights_fp16 = bool(req.get("optimize_weights_fp16", False)) and effective_device == "mps" and precision == "fp16"
    st = weights.stat(); key = ("koharu-layout", str(root), int(st.st_size), int(st.st_mtime_ns), effective_device, precision, optimize_weights_fp16)
    model = _MODEL_CACHE.get(key)
    model_cache_hit = model is not None
    model_load_seconds = 0.0
    optimize_seconds = 0.0
    optimized_weights_fp16 = False
    if model is None:
        load_started = time.perf_counter()
        loader = _import_file_module(loader_path, "folirina_koharu_layout_loader")
        load_fn = getattr(loader, "load_model", None)
        if not callable(load_fn):
            raise RuntimeError("Koharu Layout load_model.py has no load_model()")
        model = load_fn(str(weights))
        model_load_seconds = time.perf_counter() - load_started
        if hasattr(getattr(model, "model", None), "device"):
            model.model.device = torch.device(effective_device)
        # RF-DETR exposes an official inference-only optimization path that can
        # resident-optimize weights to FP16. Treat it as a best-effort M6 memory
        # optimization: never let its absence/failure affect the stable route.
        if optimize_weights_fp16 and callable(getattr(model, "optimize_for_inference", None)):
            optimize_started = time.perf_counter()
            try:
                model.optimize_for_inference(compile=False, inplace=True, dtype="float16")
                optimized_weights_fp16 = True
            except Exception:
                optimized_weights_fp16 = False
            optimize_seconds = time.perf_counter() - optimize_started
        _MODEL_CACHE[key] = model

    # RF-DETR captures its own default accelerator during construction. That
    # default can disagree with Folirina's requested device (notably CPU after
    # an MPS probe/fallback), so keep the model context authoritative too.
    model_context = getattr(model, "model", None)
    if model_context is not None and hasattr(model_context, "device"):
        model_context.device = torch.device(effective_device)

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
    observed_devices = {}
    fallback_reason = ""

    def predict_with_device_receipt():
        # Observe the actual forward pass, including RF-DETR's optimized path.
        network = (getattr(model_context, "inference_model", None)
                   if getattr(model, "_is_optimized_for_inference", False)
                   else getattr(model_context, "model", None))
        def capture_device(module, args):
            parameter = next(module.parameters(), None)
            if parameter is not None:
                observed_devices["model_device"] = str(parameter.device)
            if args and isinstance(args[0], torch.Tensor):
                observed_devices["input_device"] = str(args[0].device)
        hook = network.register_forward_pre_hook(capture_device) if network is not None else None
        try:
            return model.predict(pil, **kwargs)
        finally:
            if hook is not None:
                hook.remove()

    predict_started = time.perf_counter()
    try:
        with torch.inference_mode():
            with autocast_context(torch, effective_device, precision):
                det = predict_with_device_receipt()
    except TypeError:
        kwargs.pop("include_source_image", None)
        with torch.inference_mode():
            with autocast_context(torch, effective_device, precision):
                det = predict_with_device_receipt()
    except Exception as first:
        if effective_device != "mps":
            raise first
        fallback_reason = str(first)
        # Retry MPS in full precision first. If the backend still cannot place
        # the model (e.g. a runtime/OS mismatch), move RF-DETR's context to CPU
        # and make one explicit CPU attempt instead of leaking a stale MPS
        # device from the model's constructor.
        mps_error = first
        if precision != "fp32":
            try:
                with torch.inference_mode():
                    with autocast_context(torch, "mps", "fp32"):
                        det = predict_with_device_receipt()
                precision = "fp32"
            except Exception as exc:
                mps_error = exc
                det = None
        else:
            det = None
        if det is None:
            if model_context is not None and hasattr(model_context, "device"):
                model_context.device = torch.device("cpu")
            try:
                with torch.inference_mode():
                    with autocast_context(torch, "cpu", "fp32"):
                        det = predict_with_device_receipt()
                effective_device = "cpu-fallback"
                precision = "fp32"
            except Exception as cpu_error:
                raise RuntimeError(
                    f"RF-DETR failed on MPS and CPU fallback: MPS={mps_error}; CPU={cpu_error}"
                ) from cpu_error
    predict_seconds = time.perf_counter() - predict_started

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
        item = {"label":label,"confidence":score,"box":scaled_box,"polygon":polygon,"area":area}
        if label in {"text", "onomatopoeia"} and masks is not None and i < len(masks):
            try:
                from .koharu_mask_codec import encode_mask
            except ImportError:
                from koharu_mask_codec import encode_mask
            item["foreground_mask"] = encode_mask(np.asarray(masks[i]))
        items.append(item)
    return {
        "items":items,
        "device": effective_device,
        "requested_device": str(req.get("device", "auto")),
        "model_device": observed_devices.get("model_device", "unknown"),
        "input_device": observed_devices.get("input_device", "unknown"),
        "fallback_reason": fallback_reason,
        "precision": precision,
        "isolated_runtime":True,
        "original_shape":[int(orig_h), int(orig_w)],
        "inference_shape":[int(infer_h), int(infer_w)],
        "postprocess_max_side":int(postprocess_max_side),
        "postprocess_downscale_trigger_side":int(downscale_trigger_side),
        "downscaled_for_postprocess":bool(scale_down < 0.999),
        "model_cache_hit":bool(model_cache_hit),
        "model_load_seconds":round(float(model_load_seconds), 4),
        "optimize_seconds":round(float(optimize_seconds), 4),
        "optimized_weights_fp16":bool(optimized_weights_fp16),
        "optimize_weights_fp16_requested":bool(optimize_weights_fp16),
        "predict_seconds":round(float(predict_seconds), 4),
    }


def _manga_ocr(image: np.ndarray, req: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel
    root=str(Path(req["model_dir"]).expanduser().resolve()); device=_select_device(str(req.get("device","auto")))
    precision=resolve_precision(device, req.get("precision", env_precision()))
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
    try:
        with torch.inference_mode():
            with autocast_context(torch, device, precision):
                out=model.generate(pixels,max_new_tokens=int(req.get("max_new_tokens",128)))
    except Exception as first:
        if device == "mps" and precision != "fp32":
            with torch.inference_mode():
                out=model.generate(pixels,max_new_tokens=int(req.get("max_new_tokens",128)))
            precision="fp32"
        elif device != "cpu":
            device="cpu"; precision="fp32"; model=model.to("cpu"); pixels=pixels.to("cpu")
            with torch.inference_mode(): out=model.generate(pixels,max_new_tokens=int(req.get("max_new_tokens",128)))
        else:
            raise first
    text=tok.batch_decode(out,skip_special_tokens=True)[0]
    return {"text":str(text).strip(),"confidence":1.0,"device":device,"precision":precision,"isolated_runtime":True}


def _baberu_ocr(image: np.ndarray, req: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    import onnxruntime as ort
    root=Path(req["model_dir"]).expanduser().resolve()
    script=root/"onnx_infer.py"; vision=root/"onnx"/"vision_int4.onnx"
    if not script.is_file() or not vision.is_file():
        raise FileNotFoundError("Baberu OCR ONNX model directory is incomplete")
    st=vision.stat(); hw=detect_apple_hardware()
    requested=str(req.get("backend", "auto") or "auto").lower().strip()
    available=set(ort.get_available_providers())
    use_coreml=requested in {"auto","coreml","ane"} and hw.is_m6 and "CoreMLExecutionProvider" in available
    provider_label="onnx-cpu"
    key=("baberu-onnx",str(root),int(st.st_size),int(st.st_mtime_ns),use_coreml,requested)
    ocr=_MODEL_CACHE.get(key)
    if ocr is None:
        mod=None; cls=None
        if use_coreml:
            with patch_inference_session_for_coreml(
                ort,
                compute_units="ane" if requested == "ane" else "auto",
                cache_dir=req.get("coreml_cache_dir") or None,
                profile=bool(req.get("coreml_profile", False)),
            ):
                try:
                    mod=_import_file_module(script,"folirina_baberu_onnx_coreml")
                    cls=getattr(mod,"BaberuOnnxOCR",None)
                    if cls is None: raise RuntimeError("Baberu onnx_infer.py has no BaberuOnnxOCR")
                    ocr=cls(root/"onnx",root/"tokenizer","vision_int4.onnx")
                    provider_label="onnx-coreml+cpu"
                except Exception:
                    mod=_import_file_module(script,"folirina_baberu_onnx_cpu")
                    cls=getattr(mod,"BaberuOnnxOCR",None)
                    if cls is None: raise RuntimeError("Baberu onnx_infer.py has no BaberuOnnxOCR")
                    ocr=cls(root/"onnx",root/"tokenizer","vision_int4.onnx")
                    provider_label="onnx-cpu-fallback"
        else:
            mod=_import_file_module(script,"folirina_baberu_onnx")
            cls=getattr(mod,"BaberuOnnxOCR",None)
            if cls is None: raise RuntimeError("Baberu onnx_infer.py has no BaberuOnnxOCR")
            ocr=cls(root/"onnx",root/"tokenizer","vision_int4.onnx")
        _MODEL_CACHE[key]=(ocr,provider_label)
    ocr,provider_label=_MODEL_CACHE[key]
    pil=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    try:
        text=ocr(pil,max_new_tokens=int(req.get("max_new_tokens",128)))
    except Exception:
        if use_coreml:
            cpu_key=("baberu-onnx",str(root),int(st.st_size),int(st.st_mtime_ns),False,"cpu")
            cpu_ocr=_MODEL_CACHE.get(cpu_key)
            if cpu_ocr is None:
                cpu_mod=_import_file_module(script,"folirina_baberu_onnx_exec_cpu")
                cpu_cls=getattr(cpu_mod,"BaberuOnnxOCR",None)
                if cpu_cls is None: raise RuntimeError("Baberu onnx_infer.py has no BaberuOnnxOCR")
                cpu_ocr=cpu_cls(root/"onnx",root/"tokenizer","vision_int4.onnx")
                _MODEL_CACHE[cpu_key]=(cpu_ocr,"onnx-cpu-fallback")
            ocr,provider_label=_MODEL_CACHE[cpu_key]
            text=ocr(pil,max_new_tokens=int(req.get("max_new_tokens",128)))
        else:
            raise
    return {"text":str(text).strip(),"confidence":1.0,"device":provider_label,"isolated_runtime":True,
            "coreml": bool(use_coreml), "coreml_providers": sorted(available),
            "coreml_compute_units": requested}


def _hayai_env(req: dict[str, Any]) -> dict[str, str]:
    root = str(req.get("model_cache_dir") or "").strip()
    updates = {}
    if not bool(req.get("allow_download", False)):
        updates.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if root:
        Path(root).expanduser().mkdir(parents=True, exist_ok=True)
        # Hayai uses the standard HF cache stack. Keep all three variables in
        # sync so transformers/huggingface_hub revisions share the same cache.
        value = str(Path(root).expanduser().resolve())
        updates.update({"HF_HOME": value, "HUGGINGFACE_HUB_CACHE": value, "TRANSFORMERS_CACHE": value})
    return updates


def _hayai_build(req: dict[str, Any]):
    from hayai_ocr import HayaiOcr
    env_updates = _hayai_env(req)
    old = {k: os.environ.get(k) for k in env_updates}
    try:
        os.environ.update(env_updates)
        backend = str(req.get("backend", "auto") or "auto").lower().strip()
        kwargs: dict[str, Any] = {}
        quant = str(req.get("quantize", "none") or "none").lower().strip()
        if quant in {"int4", "int8"}:
            kwargs["quantize"] = quant
        if bool(req.get("use_v2", False)):
            kwargs["use_v2"] = True
        patches = int(req.get("max_num_patches", 384) or 384)
        if backend == "litert":
            patches = 256
        kwargs["max_num_patches"] = patches
        if backend == "litert":
            kwargs["backend"] = "litert"
            kwargs.setdefault("litert_quant", "dynamic_wi4" if quant == "int4" else "float")
        key = ("hayai", json.dumps(kwargs, sort_keys=True), tuple(sorted(env_updates.items())))
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached, key, backend
        try:
            obj = HayaiOcr(**kwargs)
        except TypeError:
            # Preserve compatibility with minor upstream API changes. The
            # documented default constructor remains the hard fallback.
            if backend == "litert":
                obj = HayaiOcr()
                backend = "torch-fallback"
            else:
                obj = HayaiOcr()
        except Exception:
            if backend == "litert":
                obj = HayaiOcr()
                backend = "torch-fallback"
            else:
                raise
        _MODEL_CACHE[key] = obj
        return obj, key, backend
    finally:
        for k, value in old.items():
            if value is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = value


def _hayai_text_meta(text: str, req: dict[str, Any], *, backend: str, batched: bool) -> dict[str, Any]:
    assumed = float(req.get("assumed_confidence", 0.74) or 0.74)
    # Hayai's public API exposes text, not a calibrated confidence score. Do
    # not pretend that this is a model probability; explicitly label it and
    # keep it configurable for the project's QA threshold.
    return {
        "confidence_source": "engine_no_score",
        "assumed_confidence": assumed,
        "qa_hint": "review_if_text_is_unexpected",
        "backend": "hayai_ocr", "engine_backend": backend,
        "max_num_patches": int(req.get("max_num_patches", 384) or 384),
        "batched": bool(batched), "isolated_runtime": True,
    }


def _hayai_ocr(image: np.ndarray, req: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    ocr, _key, backend = _hayai_build(req)
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    text = ocr(pil)
    text = str(text or "").strip()
    assumed = float(req.get("assumed_confidence", 0.74) or 0.74)
    return {"text": text, "confidence": assumed, "device": backend, "precision": "auto", **_hayai_text_meta(text, req, backend=backend, batched=False)}


def _hayai_ocr_batch(images: list[np.ndarray], req: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    if not images:
        return {"results": []}
    ocr, _key, backend = _hayai_build(req)
    pils = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
    try:
        texts = ocr(pils)
    except Exception:
        # A batch call must not turn an otherwise healthy recognizer into a
        # fatal path. Retry each image using the documented single-image API.
        texts = [ocr(img) for img in pils]
        batched = False
    else:
        batched = True
    assumed = float(req.get("assumed_confidence", 0.74) or 0.74)
    return {"results": [
        {"text": str(text or "").strip(), "confidence": assumed, "device": backend, "precision": "auto",
         **_hayai_text_meta(str(text or "").strip(), req, backend=backend, batched=batched)}
        for text in list(texts)
    ]}


def _manga_ocr_batch(images: list[np.ndarray], req: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel
    if not images:
        return {"results": []}
    root=str(Path(req["model_dir"]).expanduser().resolve()); device=_select_device(str(req.get("device","auto")))
    precision=resolve_precision(device, req.get("precision", env_precision()))
    key=("manga-ocr",root,device); triple=_MODEL_CACHE.get(key)
    if triple is None:
        proc=ViTImageProcessor.from_pretrained(root,local_files_only=True)
        tok=AutoTokenizer.from_pretrained(root,local_files_only=True)
        model=VisionEncoderDecoderModel.from_pretrained(root,local_files_only=True).to(device).eval()
        triple=(proc,tok,model); _MODEL_CACHE[key]=triple
    proc,tok,model=triple
    pils=[Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) for img in images]
    pixels=proc(images=pils,return_tensors="pt").pixel_values.to(device)
    try:
        with torch.inference_mode():
            with autocast_context(torch,device,precision):
                out=model.generate(pixels,max_new_tokens=int(req.get("max_new_tokens",128)))
    except Exception as first:
        if device=="mps" and precision!="fp32":
            with torch.inference_mode(): out=model.generate(pixels,max_new_tokens=int(req.get("max_new_tokens",128)))
            precision="fp32"
        elif device!="cpu":
            device="cpu"; precision="fp32"; model=model.to("cpu"); pixels=pixels.to("cpu")
            with torch.inference_mode(): out=model.generate(pixels,max_new_tokens=int(req.get("max_new_tokens",128)))
        else:
            raise first
    texts=tok.batch_decode(out,skip_special_tokens=True)
    return {"results":[{"text":str(t).strip(),"confidence":1.0,"device":device,"precision":precision,"isolated_runtime":True} for t in texts]}


def _ocr48px_batch(images: list[np.ndarray], req: dict[str, Any]) -> dict[str, Any]:
    if not images: return {"results": []}
    from PIL import Image
    from ocr48px_runtime import load_ocr_class
    root=Path(req["model_dir"]).expanduser().resolve(); device=_select_device(str(req.get("device","auto"))); precision=resolve_precision(device, req.get("precision", env_precision()))
    model_file=root/"ocr_ar_48px.ckpt"; st=model_file.stat(); key=("ocr48px",str(root),int(st.st_size),int(st.st_mtime_ns),device); bundle=_MODEL_CACHE.get(key)
    if bundle is None:
        OCR,model_path,dict_path=load_ocr_class(root)
        with dict_path.open("r",encoding="utf-8-sig") as fh: dictionary=[line.rstrip("\r\n") for line in fh]
        if len(dictionary)<1000 or dictionary[1:3] != ["<S>","</S>"]: raise RuntimeError("48px 字符表与官方模型不匹配")
        model=OCR(dictionary,768); model.load_state_dict(_load_48px_state_dict(model_path),strict=True); model.eval()
        try: model.to(device)
        except Exception: device="cpu"; model.to(device)
        bundle=(model,dictionary,device); _MODEL_CACHE[key]=bundle
    model,dictionary,device=bundle
    prepared=[]; widths=[]; orientations=[]
    for image in images:
        pil=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)); orientation="horizontal"
        if pil.height>pil.width*1.20: pil=pil.transpose(Image.Transpose.ROTATE_90); orientation="vertical-rotated-ccw"
        width=max(4,int(round(pil.width/max(1.0,float(pil.height))*48))); resized=pil.resize((width,48),Image.Resampling.LANCZOS)
        prepared.append(np.asarray(resized,dtype=np.uint8)); widths.append(width); orientations.append(orientation)
    max_width=max(4*(w+7)//4 for w in widths); region=np.zeros((len(prepared),48,max_width,3),dtype=np.uint8)
    for i,a in enumerate(prepared): region[i,:,:widths[i],:]=a
    tensor=(torch.from_numpy(region).float()-127.5)/127.5; tensor=tensor.permute(0,3,1,2).to(device)
    kwargs={"beams_k":max(1,min(8,int(req.get("beams_k",5)))),"max_seq_length":max(8,min(384,int(req.get("max_seq_length",255))))}
    try:
        with torch.inference_mode():
            with autocast_context(torch,device,precision): recognized=model.infer_beam_batch_tensor(tensor,widths,**kwargs)
    except RuntimeError as first:
        if device=="mps" and precision!="fp32":
            with torch.inference_mode(): recognized=model.infer_beam_batch_tensor(tensor,widths,**kwargs)
            precision="fp32"
        elif device=="cpu": raise first
        else:
            device="cpu"; precision="fp32"; model.to("cpu"); tensor=tensor.to("cpu")
            with torch.inference_mode(): recognized=model.infer_beam_batch_tensor(tensor,widths,**kwargs)
    if len(recognized)!=len(images): raise RuntimeError("48px OCR batch 返回数量异常")
    results=[]
    for rec,orientation,width in zip(recognized,orientations,widths):
        indices,probability,*_=rec; chars=[]
        for raw_index in indices:
            idx=int(raw_index)
            if idx<0 or idx>=len(dictionary): raise RuntimeError(f"48px OCR 返回越界字符索引：{idx}")
            token=dictionary[idx]
            if token=="<S>": continue
            if token=="</S>": break
            chars.append(" " if token=="<SP>" else token)
        results.append({"text":"".join(chars).strip(),"confidence":float(probability or 0.0),"device":device,"precision":precision,"input_width":width,"orientation":orientation,"isolated_runtime":True})
    return {"results":results}


def _baberu_ocr_batch(images: list[np.ndarray], req: dict[str, Any]) -> dict[str, Any]:
    return {"results":[_baberu_ocr(img,req) for img in images]}


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
    precision=resolve_precision(device, req.get("precision", env_precision()))
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
        with torch.inference_mode():
            with autocast_context(torch, device, precision):
                recognized=model.infer_beam_batch_tensor(tensor,[width],**kwargs)
    except RuntimeError as first:
        if device == "mps" and precision != "fp32":
            with torch.inference_mode():
                recognized=model.infer_beam_batch_tensor(tensor,[width],**kwargs)
            precision="fp32"
        elif device=="cpu":
            raise first
        else:
            device="cpu"; precision="fp32"; model.to("cpu"); tensor=tensor.to("cpu")
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
        "device":device, "precision":precision, "input_width":width, "orientation":orientation,
        "isolated_runtime":True,
    }


def main() -> int:
    for raw in sys.stdin:
        raw=raw.strip()
        if not raw: continue
        rid=-1
        try:
            req=json.loads(raw); rid=int(req.get("request_id",-1)); op=str(req.get("op"))
            if op=="shutdown": return 0
            if op=="trim_cache":
                payload=_trim_runtime_cache(unload_models=bool(req.get("unload_models", False)))
                row={"ok":True,"request_id":rid,**payload}
                print(json.dumps(row,ensure_ascii=False),flush=True)
                continue
            if bool(req.get("batch")):
                images=[np.load(str(path),allow_pickle=False) for path in list(req.get("images") or [])]
                if op=="manga_ocr": payload=_manga_ocr_batch(images,req)
                elif op=="baberu_ocr": payload=_baberu_ocr_batch(images,req)
                elif op=="hayai_ocr": payload=_hayai_ocr_batch(images,req)
                elif op=="ocr48px": payload=_ocr48px_batch(images,req)
                else: raise ValueError(f"unsupported batch op: {op}")
            else:
                image=np.load(str(req["image"]),allow_pickle=False)
                if op=="mangalens": payload=_mangalens(image,req)
                elif op=="ysg_obb": payload=_ysg_obb(image,req)
                elif op=="rtdetr_v2": payload=_rtdetr(image,req)
                elif op=="sam2": payload=_sam2(image,req)
                elif op=="koharu_layout": payload=_koharu_layout(image,req)
                elif op=="manga_ocr": payload=_manga_ocr(image,req)
                elif op=="baberu_ocr": payload=_baberu_ocr(image,req)
                elif op=="hayai_ocr": payload=_hayai_ocr(image,req)
                elif op=="ocr48px": payload=_ocr48px(image,req)
                else: raise ValueError(f"unsupported op: {op}")
            row={"ok":True,"request_id":rid,**payload}
        except Exception as exc:
            row={"ok":False,"request_id":rid,"error":f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-7000:]}"}
        print(json.dumps(row,ensure_ascii=False),flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())
