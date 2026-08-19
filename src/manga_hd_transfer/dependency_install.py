from __future__ import annotations

"""Explicit optional-dependency installer used by the GUI.

Nothing is installed at import/startup time.  Every installation path is only
entered after the user explicitly presses an Install button in the model center.
The installer uses the model-center proxy/TLS settings. Pure GUI helpers may target
the GUI interpreter, while Paddle and all Torch-based model backends are installed
into compatible isolated runtimes and verified in fresh subprocesses before success.
"""

from dataclasses import dataclass
import importlib.metadata
import importlib.util
import os
import platform
import re
import subprocess
import sys
from typing import Callable, Iterable

from .model_downloads import model_home
from .tls_support import apply_runtime_tls_environment

ProgressFn = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class DependencyInstallResult:
    key: str
    message: str
    restart_recommended: bool = False
    verified_modules: tuple[str, ...] = ()


DEPENDENCY_LABELS: dict[str, str] = {
    "paddle": "PaddleOCR",
    "paddle_doc": "Paddle 文档解析（VL / Structure）",
    "lightglue": "LightGlue",
    "loftr": "LoFTR",
    "mangalens": "MangaLens",
    "ysg_obb": "YSG YOLO OBB",
    "rtdetr_v2": "RT-DETR-v2",
    "sam2": "SAM 2.1",
    "koharu_layout": "Koharu Layout RF-DETR Seg 2XL",
    "manga_ocr": "Manga OCR",
    "baberu_ocr": "Baberu OCR",
    "ocr48px": "48px AR OCR",
    "torch_sr": "MPS 局部超分",
}

DEPENDENCY_MODULES: dict[str, tuple[str, ...]] = {
    "paddle": ("paddle", "paddleocr"),
    "paddle_doc": ("paddle", "paddleocr", "paddlex"),
    "lightglue": ("torch", "torchvision", "kornia", "lightglue"),
    "loftr": ("torch", "kornia"),
    "mangalens": ("torch", "torchvision", "ultralytics"),
    "ysg_obb": ("torch", "torchvision", "ultralytics"),
    "rtdetr_v2": ("torch", "torchvision", "transformers", "safetensors"),
    "sam2": ("torch", "torchvision", "sam2"),
    "koharu_layout": ("torch", "torchvision", "rfdetr", "safetensors"),
    "manga_ocr": ("torch", "torchvision", "transformers", "safetensors"),
    "baberu_ocr": ("onnxruntime", "numpy", "PIL"),
    "ocr48px": ("torch", "einops", "numpy", "PIL"),
    "torch_sr": ("torch", "spandrel"),
}


def _emit(cb: ProgressFn | None, message: str) -> None:
    if cb is not None:
        cb(str(message))


def _install_env() -> dict[str, str]:
    env = os.environ.copy()
    proxy = str(env.get("MHD_MODEL_PROXY", "") or "").strip()
    if proxy:
        env["HTTP_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
        env["http_proxy"] = proxy
        env["https_proxy"] = proxy
    # SAM 2's CUDA extension is not useful on macOS/MPS.  Its own installation
    # guide explicitly allows disabling this extension without disabling image
    # inference.  Keeping this process-local avoids changing the user's shell.
    if platform.system() == "Darwin":
        env["SAM2_BUILD_CUDA"] = "0"
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return apply_runtime_tls_environment(env, model_home().parent / "runtime" / "gui-dependencies")


def _run(cmd: list[str], *, progress: ProgressFn | None = None, env_extra: dict[str, str] | None = None) -> tuple[int, str]:
    _emit(progress, "执行：" + " ".join(cmd))
    env = _install_env()
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items()})
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        tail.append(line)
        if len(tail) > 50:
            tail = tail[-50:]
        low = line.lower()
        if any(token in low for token in (
            "error", "failed", "installing", "successfully", "collecting",
            "looking in indexes", "requirement already satisfied", "warning",
        )):
            _emit(progress, line)
    code = int(proc.wait())
    return code, "\n".join(tail)


def _ensure_pip(*, progress: ProgressFn | None = None) -> None:
    code, _ = _run([sys.executable, "-m", "pip", "--version"], progress=None)
    if code == 0:
        return
    _emit(progress, "当前 Python 缺少 pip，正在尝试 ensurepip…")
    code, tail = _run([sys.executable, "-m", "ensurepip", "--upgrade"], progress=progress)
    if code != 0:
        raise RuntimeError("当前 GUI Python 无可用 pip，且 ensurepip 初始化失败。\n" + tail[-3000:])


def _pip_install(args: list[str], *, progress: ProgressFn | None = None, env_extra: dict[str, str] | None = None) -> tuple[int, str]:
    _ensure_pip(progress=progress)
    return _run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *args],
        progress=progress,
        env_extra=env_extra,
    )


def _pip_install_with_mirror(
    args: list[str], *, progress: ProgressFn | None = None,
    mirror: str = "https://pypi.tuna.tsinghua.edu.cn/simple",
    env_extra: dict[str, str] | None = None,
) -> None:
    code, tail = _pip_install(args, progress=progress, env_extra=env_extra)
    if code == 0:
        return
    _emit(progress, "默认 Python 包源安装失败，尝试备用 PyPI 镜像…")
    code, tail2 = _pip_install([*args, "-i", mirror], progress=progress, env_extra=env_extra)
    if code != 0:
        detail = (tail + "\n" + tail2).strip()
        raise RuntimeError("依赖安装失败。请检查网络/代理，或使用离线 wheel。\n" + detail[-5000:])


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", str(value).split("+", 1)[0])
    return tuple(int(x) for x in numbers[:4]) if numbers else (0,)


def _package_version_at_least(package: str, minimum: str) -> bool:
    try:
        return _version_tuple(importlib.metadata.version(package)) >= _version_tuple(minimum)
    except Exception:
        return False


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def missing_dependency_modules(key: str) -> tuple[str, ...]:
    """Return missing runtime pieces without importing heavy model packages.

    Paddle is intentionally special: it lives in an isolated Python 3.9-3.13
    venv, so absence from the GUI interpreter is not a missing dependency.
    """
    key = str(key)
    if key == "paddle":
        try:
            from .paddle_runtime import runtime_status
            return () if runtime_status().ready else ("paddle-isolated-runtime",)
        except Exception:
            return ("paddle-isolated-runtime",)
    if key == "paddle_doc":
        try:
            from .paddle_doc_runtime import runtime_status
            return () if runtime_status().ready else ("paddle-doc-isolated-runtime",)
        except Exception:
            return ("paddle-doc-isolated-runtime",)
    if key in {"lightglue", "loftr"}:
        try:
            from .deep_registration_runtime import runtime_status
            return () if runtime_status().ready else ("deep-registration-isolated-runtime",)
        except Exception:
            return ("deep-registration-isolated-runtime",)
    if key in {"mangalens", "ysg_obb", "rtdetr_v2", "sam2", "koharu_layout", "manga_ocr", "baberu_ocr", "ocr48px"}:
        try:
            from .vision_runtime import runtime_status
            return () if runtime_status(key).ready else ("vision-isolated-runtime",)
        except Exception:
            return ("vision-isolated-runtime",)
    modules = DEPENDENCY_MODULES.get(key, ())
    return tuple(name for name in modules if not _has_module(name))


def dependency_summary(key: str) -> str:
    missing = missing_dependency_modules(key)
    if not missing:
        return "依赖已安装"
    return "缺依赖：" + ", ".join(missing)


def _verify_imports(modules: Iterable[str], *, progress: ProgressFn | None = None) -> tuple[str, ...]:
    mods = tuple(dict.fromkeys(str(x) for x in modules if str(x)))
    if not mods:
        return ()
    _emit(progress, "正在用当前 GUI Python 验证运行依赖…")
    script = (
        "import importlib,sys\n"
        "mods=" + repr(list(mods)) + "\n"
        "bad=[]\n"
        "for m in mods:\n"
        "    try: importlib.import_module(m)\n"
        "    except Exception as e: bad.append((m,type(e).__name__,str(e)))\n"
        "print(repr(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    code, tail = _run([sys.executable, "-c", script], progress=None)
    if code != 0:
        raise RuntimeError(
            "依赖包已经执行安装，但新 Python 进程仍无法导入。通常是 wheel/架构或二进制依赖不匹配。\n" + tail[-5000:]
        )
    _emit(progress, "依赖导入验证通过：" + ", ".join(mods))
    return mods


def _ensure_torch(*, progress: ProgressFn | None = None, minimum: str | None = None, need_torchvision: bool = False) -> None:
    torch_ok = _has_module("torch") and (minimum is None or _package_version_at_least("torch", minimum))
    tv_ok = (not need_torchvision) or _has_module("torchvision")
    if torch_ok and tv_ok:
        _emit(progress, "PyTorch 运行时已满足要求，跳过。")
        return
    args = ["--upgrade"]
    args.append(f"torch>={minimum}" if minimum else "torch")
    if need_torchvision:
        args.append("torchvision")
    _emit(progress, "安装 PyTorch 运行时（macOS 将使用官方 pip wheel / MPS 路径）…")
    _pip_install_with_mirror(args, progress=progress)


def install_paddle_ocr_dependencies(progress: ProgressFn | None = None) -> DependencyInstallResult:
    """Create/repair an isolated PaddleOCR runtime.

    This mirrors Novel Formatter's working architecture: PaddlePaddle is not
    forced into the GUI's Python.  A compatible Python 3.9-3.13 interpreter is
    located, a dedicated venv is created, and Paddle/PaddleOCR are installed
    there.  On modern macOS this also guarantees an arm64 interpreter instead
    of a Rosetta x86_64 Python for which Paddle publishes no wheel.
    """
    from .paddle_runtime import ensure_runtime
    _emit(progress, f"GUI Python：{sys.executable}")
    _emit(progress, "PP-OCR 将使用独立 Python 运行环境，不修改 GUI Python。")
    status = ensure_runtime(progress)
    return DependencyInstallResult(
        "paddle",
        "PP-OCR 独立运行环境已安装并验证。" + (" " + status.detail if status.detail else ""),
        False,
        ("paddle-isolated-runtime",),
    )



def install_paddle_doc_dependencies(progress: ProgressFn | None = None) -> DependencyInstallResult:
    from .paddle_doc_runtime import ensure_runtime
    _emit(progress, f"GUI Python：{sys.executable}")
    _emit(progress, "PaddleOCR-VL / PP-StructureV3 使用独立文档解析 venv，不修改普通 PP-OCR 运行环境。")
    status = ensure_runtime(progress)
    return DependencyInstallResult(
        "paddle_doc",
        "Paddle 文档解析独立运行环境已安装并验证。" + (" " + status.detail if status.detail else ""),
        False,
        ("paddle-doc-isolated-runtime",),
    )

def _install_deep_registration_runtime(key: str, progress: ProgressFn | None) -> DependencyInstallResult:
    from .deep_registration_runtime import ensure_runtime
    _emit(progress, f"GUI Python：{sys.executable}")
    _emit(progress, "LightGlue / LoFTR 将使用独立 Python 3.10～3.13 运行环境；不再导入 GUI Python 的 Torch。")
    status = ensure_runtime(progress)
    return DependencyInstallResult(
        key,
        "LightGlue / LoFTR 独立配准运行环境已安装并真实导入验证。" + (" " + status.detail if status.detail else ""),
        False,
        ("deep-registration-isolated-runtime",),
    )


def _install_lightglue(progress: ProgressFn | None) -> DependencyInstallResult:
    return _install_deep_registration_runtime("lightglue", progress)


def _install_loftr(progress: ProgressFn | None) -> DependencyInstallResult:
    return _install_deep_registration_runtime("loftr", progress)


def _install_vision_runtime(key: str, progress: ProgressFn | None) -> DependencyInstallResult:
    from .vision_runtime import ensure_runtime
    _emit(progress, f"GUI Python：{sys.executable}")
    _emit(progress, "MangaLens / RT-DETR-v2 / SAM 2.1 将运行在独立 Python 3.10～3.13 Torch 环境；不再导入 GUI Python 3.14 的 Torch。")
    status = ensure_runtime(key, progress)
    return DependencyInstallResult(
        key,
        f"{DEPENDENCY_LABELS.get(key,key)} 独立视觉运行环境已安装并真实导入验证。 " + status.detail,
        False,
        ("vision-isolated-runtime",),
    )


def _install_mangalens(progress: ProgressFn | None) -> DependencyInstallResult:
    return _install_vision_runtime("mangalens", progress)


def _install_ysg_obb(progress: ProgressFn | None) -> DependencyInstallResult:
    return _install_vision_runtime("ysg_obb", progress)


def _install_rtdetr(progress: ProgressFn | None) -> DependencyInstallResult:
    return _install_vision_runtime("rtdetr_v2", progress)


def _install_sam2(progress: ProgressFn | None) -> DependencyInstallResult:
    return _install_vision_runtime("sam2", progress)


def _install_koharu_layout(progress: ProgressFn | None) -> DependencyInstallResult:
    return _install_vision_runtime("koharu_layout", progress)


def _install_manga_ocr(progress: ProgressFn | None) -> DependencyInstallResult:
    return _install_vision_runtime("manga_ocr", progress)


def _install_baberu_ocr(progress: ProgressFn | None) -> DependencyInstallResult:
    return _install_vision_runtime("baberu_ocr", progress)


def _install_ocr48px(progress: ProgressFn | None) -> DependencyInstallResult:
    return _install_vision_runtime("ocr48px", progress)


def _install_torch_sr(progress: ProgressFn | None) -> DependencyInstallResult:
    _ensure_torch(progress=progress)
    if not _has_module("spandrel"):
        _pip_install_with_mirror(["--upgrade", "spandrel>=0.4"], progress=progress)
    verified = _verify_imports(DEPENDENCY_MODULES["torch_sr"], progress=progress)
    return DependencyInstallResult("torch_sr", "MPS 局部超分运行依赖已安装并验证。", False, verified)


def install_model_dependencies(key: str, progress: ProgressFn | None = None) -> DependencyInstallResult:
    """Install and verify runtime dependencies for one built-in model."""
    key = str(key)
    _emit(progress, f"目标：{DEPENDENCY_LABELS.get(key, key)} · Python：{sys.executable}")
    dispatch = {
        "paddle": install_paddle_ocr_dependencies,
        "paddle_doc": install_paddle_doc_dependencies,
        "lightglue": _install_lightglue,
        "loftr": _install_loftr,
        "mangalens": _install_mangalens,
        "ysg_obb": _install_ysg_obb,
        "rtdetr_v2": _install_rtdetr,
        "sam2": _install_sam2,
        "koharu_layout": _install_koharu_layout,
        "manga_ocr": _install_manga_ocr,
        "baberu_ocr": _install_baberu_ocr,
        "ocr48px": _install_ocr48px,
        "torch_sr": _install_torch_sr,
    }
    fn = dispatch.get(key)
    if fn is None:
        raise KeyError(key)
    return fn(progress)


def install_all_model_dependencies(progress: ProgressFn | None = None) -> DependencyInstallResult:
    """Install missing dependencies for all downloadable built-in models."""
    keys = ("paddle", "lightglue", "loftr", "mangalens", "ysg_obb", "rtdetr_v2", "sam2", "koharu_layout", "manga_ocr", "baberu_ocr", "ocr48px")
    completed: list[str] = []
    for i, key in enumerate(keys, 1):
        label = DEPENDENCY_LABELS[key]
        if not missing_dependency_modules(key):
            _emit(progress, f"[{i}/{len(keys)}] {label}：依赖已齐全，跳过。")
            completed.append(label)
            continue
        _emit(progress, f"[{i}/{len(keys)}] 正在安装 {label} 依赖…")
        install_model_dependencies(key, progress)
        completed.append(label)
    # Every Torch-based model is verified inside an isolated Python 3.10-3.13
    # environment. Never import Torch/Ultralytics/Transformers/SAM2 into the GUI
    # Python 3.14 merely to validate installation.
    from .paddle_runtime import runtime_status as paddle_runtime_status
    from .deep_registration_runtime import runtime_status as deep_registration_status
    from .vision_runtime import runtime_status as vision_runtime_status
    paddle = paddle_runtime_status(); deep = deep_registration_status()
    if not paddle.ready:
        raise RuntimeError("PP-OCR 独立运行环境验证失败：" + paddle.detail)
    if not deep.ready:
        raise RuntimeError("LightGlue / LoFTR 独立运行环境验证失败：" + deep.detail)
    for key in ("mangalens", "ysg_obb", "rtdetr_v2", "sam2", "koharu_layout", "manga_ocr", "baberu_ocr", "ocr48px"):
        status = vision_runtime_status(key)
        if not status.ready:
            raise RuntimeError(f"{DEPENDENCY_LABELS[key]} 独立视觉运行环境验证失败：" + status.detail)
    return DependencyInstallResult(
        "all",
        "全部内置模型运行依赖已安装并验证；PP-OCR 与全部 Torch 模型均使用独立兼容 Python venv，GUI Python 3.14 不再加载这些 AI 运行库。",
        False,
        ("paddle-isolated-runtime", "deep-registration-isolated-runtime", "vision-isolated-runtime"),
    )
