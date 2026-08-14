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
    rows["lightglue"] = ComponentStatus("lightglue", "LightGlue", _has_module("lightglue") and torch_ok, _has_module("lightglue"), "困难页局部特征匹配；权重已缓存时 Auto 才自动升级。")
    rows["loftr"] = ComponentStatus("loftr", "LoFTR", _has_module("kornia") and torch_ok, _has_module("kornia"), "仅在低置信度困难页升级；Auto 默认禁止隐藏下载权重。")
    rows["mangalens"] = ComponentStatus(
        "mangalens", "MangaLens", _has_module("ultralytics") and bool(cfg.bubbles.mangalens_model_path and Path(cfg.bubbles.mangalens_model_path).expanduser().is_file()),
        _has_module("ultralytics"), "需要 Ultralytics + 显式本地权重；模型整册常驻复用。",
    )
    rows["torch_sr"] = ComponentStatus(
        "torch_sr", "Torch 局部超分", torch_ok and _has_module("spandrel") and bool(cfg.mask_replace.sr_model_path and Path(cfg.mask_replace.sr_model_path).expanduser().is_file()),
        torch_ok and _has_module("spandrel"), "需要显式选择本地超分模型；MPS 可用时走 MPS。",
    )
    rows["paddle"] = ComponentStatus("paddle", "PP-OCR", _has_module("paddleocr"), _has_module("paddleocr"), "只在 OCR/高清重排需要时加载。")
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
