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
from .model_downloads import model_local_paths


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
    local_models = model_local_paths()
    lg_weights = local_models["lightglue"].is_file()
    loftr_weights = local_models["loftr"].is_file()
    lg_pkg = _has_module("lightglue") and torch_ok
    loftr_pkg = _has_module("kornia") and torch_ok
    rows["lightglue"] = ComponentStatus(
        "lightglue", "LightGlue", lg_pkg and lg_weights, _has_module("lightglue"),
        ("SIFT 权重已缓存，可离线使用。" if lg_weights else "缺少 SIFT 权重；可在模型中心主动下载。") + ("" if _has_module("lightglue") else " 运行依赖未安装。"),
    )
    rows["loftr"] = ComponentStatus(
        "loftr", "LoFTR", loftr_pkg and loftr_weights, _has_module("kornia"),
        ("outdoor 权重已缓存，可离线使用。" if loftr_weights else "缺少 outdoor 权重；可在模型中心主动下载。") + ("" if _has_module("kornia") else " 运行依赖未安装。"),
    )
    mangalens_path = Path(cfg.bubbles.mangalens_model_path).expanduser() if cfg.bubbles.mangalens_model_path else local_models["mangalens"]
    mangalens_weight = mangalens_path.is_file()
    rows["mangalens"] = ComponentStatus(
        "mangalens", "MangaLens", _has_module("ultralytics") and mangalens_weight,
        _has_module("ultralytics"),
        (f"权重：{mangalens_path}" if mangalens_weight else "缺少 best.pt；可在模型中心主动下载。") + ("" if _has_module("ultralytics") else " 运行依赖未安装。"),
    )
    rtdetr_path = Path(cfg.bubbles.rtdetr_model_path).expanduser() if cfg.bubbles.rtdetr_model_path else local_models["rtdetr_v2"]
    rtdetr_files = all((rtdetr_path / n).is_file() for n in ("config.json", "preprocessor_config.json", "model.safetensors")) if rtdetr_path.is_dir() else False
    rtdetr_pkg = torch_ok and _has_module("transformers")
    rows["rtdetr_v2"] = ComponentStatus(
        "rtdetr_v2", "RT-DETR-v2", rtdetr_pkg and rtdetr_files,
        _has_module("transformers"),
        (f"本地模型：{rtdetr_path}" if rtdetr_files else "缺少本地模型；可在模型中心主动下载。") + ("" if rtdetr_pkg else " 运行依赖未安装。"),
    )
    sam2_path = Path(cfg.bubbles.sam2_checkpoint).expanduser() if cfg.bubbles.sam2_checkpoint else local_models["sam2"]
    sam2_weight = sam2_path.is_file()
    sam2_pkg = torch_ok and _has_module("sam2")
    rows["sam2"] = ComponentStatus(
        "sam2", "SAM 2.1", sam2_pkg and sam2_weight,
        _has_module("sam2"),
        (f"checkpoint：{sam2_path}" if sam2_weight else "缺少 Hiera Tiny checkpoint；可在模型中心主动下载。") + ("" if sam2_pkg else " 运行依赖未安装。"),
    )
    rows["torch_sr"] = ComponentStatus(
        "torch_sr", "Torch 局部超分", torch_ok and _has_module("spandrel") and bool(cfg.mask_replace.sr_model_path and Path(cfg.mask_replace.sr_model_path).expanduser().is_file()),
        torch_ok and _has_module("spandrel"), "需要显式选择本地超分模型；MPS 可用时走 MPS。",
    )
    paddle_runtime = _has_module("paddle") and _has_module("paddleocr")
    paddle_marker = local_models["paddle"].is_file()
    rows["paddle"] = ComponentStatus(
        "paddle", "PP-OCR", paddle_runtime and paddle_marker, paddle_runtime,
        ("PP-OCRv5 中/日文模型已主动预热/导入。" if paddle_marker else "可在模型中心主动下载/预热中日文模型。") + ("" if paddle_runtime else " PaddlePaddle / PaddleOCR 运行依赖未安装完整。"),
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
