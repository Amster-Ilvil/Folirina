from __future__ import annotations

"""Side-effect-free optional runtime/model readiness catalog.

Normal GUI refreshes intentionally avoid importing torch/ultralytics/kornia or
model modules. This follows the same operational principle as Novel-formatter's
runtime catalog and BallonsTranslator's lazy registry: inspecting readiness must
not allocate a model, start a worker, create a venv, or download a weight.
"""

import importlib.util
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .model_downloads import model_local_paths, paddle_profile_marker_status
from .dependency_install import missing_dependency_modules


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    key: str
    label: str
    ready: bool
    installed: bool
    detail: str
    kind: str = "optional"

    def to_dict(self):
        return {"key": self.key, "label": self.label, "ready": self.ready, "installed": self.installed, "detail": self.detail, "kind": self.kind}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def probe_components(config: PipelineConfig | None = None, *, deep: bool = False) -> dict[str, ComponentStatus]:
    cfg = config or PipelineConfig()
    torch_ok = _has_module("torch")
    is_mac = platform.system() == "Darwin"
    arm = platform.machine().lower() in {"arm64", "aarch64"}
    if deep and torch_ok:
        from .runtime import device_info
        dev = device_info(cfg.runtime.device)
        mps_ready = bool(dev.mps_available)
        mps_detail = f"{dev.name} · {dev.note}"
    else:
        # Shallow GUI probes avoid importing ~hundreds of MB of PyTorch just to
        # render a settings page. Actual MPS support is verified when a job starts.
        mps_ready = bool(torch_ok and is_mac and arm)
        mps_detail = "Apple Silicon + PyTorch 已检测；任务启动时再验证 MPS。" if mps_ready else ("PyTorch 未安装。" if not torch_ok else "当前平台不推定为 Apple MPS。")

    rows: dict[str, ComponentStatus] = {}
    rows["mps"] = ComponentStatus("mps", "Apple MPS", mps_ready, torch_ok, mps_detail, "accelerator")
    if deep and torch_ok:
        from .runtime import device_info
        cuda_dev = device_info("cuda")
        cuda_ready = bool(cuda_dev.selected == "cuda" and cuda_dev.available)
        cuda_detail = f"{cuda_dev.name} · {cuda_dev.note}"
    else:
        # A shallow GUI refresh must never claim CUDA is ready merely because
        # the torch package exists.  Real CUDA availability is verified by the
        # background deep probe / job startup without blocking the GUI.
        cuda_ready = False
        if not torch_ok:
            cuda_detail = "PyTorch 未安装。"
        elif platform.system() == "Darwin":
            cuda_detail = "macOS Studio 默认使用 Apple MPS；Windows/Linux NVIDIA 显卡在任务启动时再验证。"
        else:
            cuda_detail = "PyTorch 已检测；任务启动时再验证 NVIDIA CUDA，失败自动回退 CPU。"
    rows["cuda"] = ComponentStatus("cuda", "NVIDIA CUDA", cuda_ready, torch_ok, cuda_detail, "accelerator")
    local_models = model_local_paths()
    lg_weights = local_models["lightglue"].is_file()
    loftr_weights = local_models["loftr"].is_file()
    lg_missing = missing_dependency_modules("lightglue")
    loftr_missing = missing_dependency_modules("loftr")
    lg_pkg = not lg_missing
    loftr_pkg = not loftr_missing
    try:
        from .deep_registration_runtime import runtime_status as _deep_registration_status
        deep_reg_status = _deep_registration_status()
        deep_reg_detail = deep_reg_status.detail
    except Exception as exc:
        deep_reg_detail = f"独立深度配准运行环境探测失败：{exc}"
    rows["lightglue"] = ComponentStatus(
        "lightglue", "LightGlue", lg_pkg and lg_weights, lg_pkg,
        ("SIFT 权重已缓存，可离线使用。" if lg_weights else "缺少 SIFT 权重；可在模型中心主动下载。")
        + ("" if lg_pkg else " 缺独立配准运行时") + (" " + deep_reg_detail if deep_reg_detail else ""),
    )
    rows["loftr"] = ComponentStatus(
        "loftr", "LoFTR", loftr_pkg and loftr_weights, loftr_pkg,
        ("outdoor 权重已缓存，可离线使用。" if loftr_weights else "缺少 outdoor 权重；可在模型中心主动下载。")
        + ("" if loftr_pkg else " 缺独立配准运行时") + (" " + deep_reg_detail if deep_reg_detail else ""),
    )
    mangalens_path = Path(cfg.bubbles.mangalens_model_path).expanduser() if cfg.bubbles.mangalens_model_path else local_models["mangalens"]
    mangalens_weight = mangalens_path.is_file()
    try:
        from .vision_runtime import runtime_status as _vision_runtime_status
        mangalens_runtime = _vision_runtime_status("mangalens")
    except Exception as exc:
        mangalens_runtime = None
        mangalens_detail = f"独立视觉运行环境探测失败：{exc}"
    else:
        mangalens_detail = mangalens_runtime.detail
    mangalens_pkg = bool(mangalens_runtime and mangalens_runtime.ready)
    rows["mangalens"] = ComponentStatus(
        "mangalens", "MangaLens", mangalens_pkg and mangalens_weight,
        mangalens_pkg,
        (f"权重：{mangalens_path}" if mangalens_weight else "缺少 best.pt；可在模型中心主动下载。") + " " + mangalens_detail,
    )
    ysg_path = Path(cfg.bubbles.ysg_obb_model_path).expanduser() if cfg.bubbles.ysg_obb_model_path else local_models["ysg_obb"]
    ysg_weight = ysg_path.is_file()
    try:
        ysg_runtime = _vision_runtime_status("ysg_obb")
        ysg_detail = ysg_runtime.detail
    except Exception as exc:
        ysg_runtime = None; ysg_detail = f"独立视觉运行环境探测失败：{exc}"
    ysg_pkg = bool(ysg_runtime and ysg_runtime.ready)
    rows["ysg_obb"] = ComponentStatus(
        "ysg_obb", "YSG YOLO OBB", ysg_pkg and ysg_weight, ysg_pkg,
        (f"权重：{ysg_path}" if ysg_weight else "缺少 ysgyolo_yolo26_2.0.pt；可在模型中心主动下载。") + " " + ysg_detail,
    )
    rtdetr_path = Path(cfg.bubbles.rtdetr_model_path).expanduser() if cfg.bubbles.rtdetr_model_path else local_models["rtdetr_v2"]
    rtdetr_files = all((rtdetr_path / n).is_file() for n in ("config.json", "preprocessor_config.json", "model.safetensors")) if rtdetr_path.is_dir() else False
    try:
        rtdetr_runtime = _vision_runtime_status("rtdetr_v2")
        rtdetr_detail = rtdetr_runtime.detail
    except Exception as exc:
        rtdetr_runtime = None; rtdetr_detail = f"独立视觉运行环境探测失败：{exc}"
    rtdetr_pkg = bool(rtdetr_runtime and rtdetr_runtime.ready)
    rows["rtdetr_v2"] = ComponentStatus(
        "rtdetr_v2", "RT-DETR-v2", rtdetr_pkg and rtdetr_files,
        rtdetr_pkg,
        (f"本地模型：{rtdetr_path}" if rtdetr_files else "缺少本地模型；可在模型中心主动下载。") + " " + rtdetr_detail,
    )
    sam2_path = Path(cfg.bubbles.sam2_checkpoint).expanduser() if cfg.bubbles.sam2_checkpoint else local_models["sam2"]
    sam2_weight = sam2_path.is_file()
    try:
        sam2_runtime = _vision_runtime_status("sam2")
        sam2_detail = sam2_runtime.detail
    except Exception as exc:
        sam2_runtime = None; sam2_detail = f"独立视觉运行环境探测失败：{exc}"
    sam2_pkg = bool(sam2_runtime and sam2_runtime.ready)
    rows["sam2"] = ComponentStatus(
        "sam2", "SAM 2.1", sam2_pkg and sam2_weight,
        sam2_pkg,
        (f"checkpoint：{sam2_path}" if sam2_weight else "缺少 Hiera Tiny checkpoint；可在模型中心主动下载。") + " " + sam2_detail,
    )
    for key, label, configured, required in (
        ("koharu_layout", "Koharu Layout", cfg.bubbles.koharu_layout_model_path, ("inference_config.json","load_model.py","model.safetensors")),
        ("manga_ocr", "Manga OCR", cfg.ocr.manga_ocr_model_path, ("config.json","preprocessor_config.json","model.safetensors")),
        ("baberu_ocr", "Baberu OCR", cfg.ocr.baberu_ocr_model_path, ("onnx_infer.py","onnx/vision_int4.onnx","onnx/decoder_prefill_int8.onnx","onnx/decoder_step_int8.onnx","tokenizer/vocab.json")),
    ):
        path=Path(configured).expanduser() if configured else local_models[key]
        files=bool(path.is_dir() and all((path/n).is_file() for n in required))
        try:
            status=_vision_runtime_status(key); runtime_ready=bool(status.ready); detail=status.detail
        except Exception as exc:
            runtime_ready=False; detail=f"独立运行环境探测失败：{exc}"
        rows[key]=ComponentStatus(
            key,label,runtime_ready and files,runtime_ready,
            (f"本地模型：{path}" if files else "模型尚未下载/导入。")+" "+detail,
        )
    mit48_path = Path(cfg.ocr.ocr48px_model_path).expanduser() if cfg.ocr.ocr48px_model_path else local_models["ocr48px"]
    mit48_files = bool(mit48_path.is_dir() and all((mit48_path / n).is_file() for n in ("ocr_ar_48px.ckpt", "alphabet-all-v7.txt", "upstream-source/manga_48px_core.py", "upstream-source/manga_48px_xpos.py")))
    mit48_runner = bool(str(cfg.ocr.ocr48px_command or "").strip())
    try:
        status=_vision_runtime_status("ocr48px"); mit48_native_runtime=bool(status.ready); mit48_detail=status.detail
    except Exception as exc:
        mit48_native_runtime=False; mit48_detail=f"独立运行环境探测失败：{exc}"
    rows["ocr48px"] = ComponentStatus(
        "ocr48px", "48px AR OCR", bool((mit48_files and mit48_native_runtime) or mit48_runner), bool(mit48_native_runtime or mit48_runner),
        (f"本地模型：{mit48_path}" if mit48_files else "模型尚未下载/导入。") +
        (f" 原生独立运行时已就绪。 {mit48_detail}" if mit48_native_runtime else "") +
        (" 外部 runner 已配置。" if mit48_runner else ""),
    )
    for key,label,path_value,command in (
        ("lama_manga","LaMa Manga",cfg.inpainting.lama_model_path,cfg.inpainting.lama_manga_command or cfg.inpainting.lama_command),
        ("aot_inpainting","AOT Inpainting",cfg.inpainting.aot_model_path,cfg.inpainting.aot_command),
        ("flux2_klein","FLUX.2 Klein",cfg.inpainting.flux2_klein_model_path,cfg.inpainting.flux2_klein_command),
        ("rorem_mixed","RORem Mixed",cfg.inpainting.rorem_mixed_model_path,cfg.inpainting.rorem_mixed_command),
    ):
        path=Path(path_value).expanduser() if path_value else local_models[key]
        exists=path.exists()
        rows[key]=ComponentStatus(
            key,label,bool(exists and command),bool(command),
            (f"模型：{path}" if exists else "模型尚未导入。") + (" 本地 runner 已配置。" if command else " 尚未配置本地 runner 命令。"),
        )
    torch_sr_missing = missing_dependency_modules("torch_sr")
    torch_sr_pkg = not torch_sr_missing
    rows["torch_sr"] = ComponentStatus(
        "torch_sr", "Torch 局部超分", torch_sr_pkg and bool(cfg.mask_replace.sr_model_path and Path(cfg.mask_replace.sr_model_path).expanduser().is_file()),
        torch_sr_pkg, "需要显式选择本地超分模型；MPS 可用时走 MPS。" + ("" if torch_sr_pkg else " 缺依赖：" + ", ".join(torch_sr_missing)),
    )
    active_profile = str(getattr(cfg.ocr, "paddle_model_profile", "legacy_v5_auto") or "legacy_v5_auto")
    try:
        from .paddle_profiles import get_paddle_model_profile
        active_pipeline = get_paddle_model_profile(active_profile).pipeline
    except Exception:
        active_pipeline = "ocr"
    paddle_dependency_key = "paddle_doc" if active_pipeline in {"vl", "structure"} else "paddle"
    paddle_missing = missing_dependency_modules(paddle_dependency_key)
    paddle_runtime = not paddle_missing
    paddle_marker = local_models["paddle"].is_file()
    selected_profile_ready, warmed_profiles = paddle_profile_marker_status(active_profile)
    # Explicit local det+rec directories are authoritative and need no marker.
    if getattr(cfg.ocr, "paddle_text_detection_model_dir", None) and getattr(cfg.ocr, "paddle_text_recognition_model_dir", None):
        selected_profile_ready = True
    try:
        if active_pipeline in {"vl", "structure"}:
            from .paddle_doc_runtime import runtime_status as _paddle_runtime_status
        else:
            from .paddle_runtime import runtime_status as _paddle_runtime_status
        paddle_status = _paddle_runtime_status()
        paddle_detail = paddle_status.detail
    except Exception as exc:
        paddle_detail = f"独立 Paddle 运行环境探测失败：{exc}"
    try:
        from .paddle_profiles import profile_label as _paddle_profile_label
        active_profile_label = _paddle_profile_label(active_profile)
    except Exception:
        active_profile_label = active_profile
    warmed_text = "、".join(warmed_profiles) if warmed_profiles else "无"
    rows["paddle"] = ComponentStatus(
        "paddle", "PaddleOCR", paddle_runtime and selected_profile_ready, paddle_runtime,
        ((f"当前模型已预热：{active_profile_label}。" if selected_profile_ready else f"当前模型尚未预热：{active_profile_label}。已缓存档位：{warmed_text}。"))
        + (" " + paddle_detail if paddle_detail else ""),
    )
    helper_source = Path(__file__).resolve().parents[2] / "tools" / "apple_live_text_helper" / "AppleLiveTextOCRHelper.swift"
    helper_binary = Path(__file__).resolve().parents[2] / "tools" / "apple_live_text_helper" / "bin" / "apple_live_text_helper"
    xcrun_ok = bool(shutil.which("xcrun")) if is_mac else False
    shortcuts_ok = bool(shutil.which("shortcuts")) if is_mac else False
    live_ready = bool(is_mac and helper_source.exists() and (helper_binary.exists() or xcrun_ok))
    rows["apple_live_text"] = ComponentStatus(
        "apple_live_text", "Apple Live Text", live_ready, helper_binary.exists(),
        ("Swift VisionKit Helper 已就绪。" if helper_binary.exists() else
         "首次 OCR 时可用 xcrun 编译 Swift VisionKit Helper。" if live_ready else
         "需要 macOS + Xcode Command Line Tools；仍可使用 ExtractText 快捷指令回退。"),
        "system",
    )
    rows["apple_shortcut"] = ComponentStatus(
        "apple_shortcut", "ExtractText 快捷指令", bool(is_mac and shortcuts_ok), bool(is_mac and shortcuts_ok),
        "调用 macOS shortcuts；需在快捷指令 App 中存在 ExtractText（从图像中提取文字）。" if shortcuts_ok else "未检测到 shortcuts 命令。",
        "system",
    )
    return rows
