from __future__ import annotations

"""Dedicated Paddle document-parser runtime (VL / PP-StructureV3).

Keep document parsing isolated from classic PP-OCR.  PaddleOCR's ``doc-parser``
extras are substantially heavier and evolve on a different dependency surface;
installing them into the same venv as the regular OCR worker made a repair for
VL/Structure capable of destabilising PP-OCRv6.  This module follows Paddle's
official installation path and owns a separate venv.
"""

from dataclasses import dataclass
import functools
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import time
from typing import Callable

from .model_downloads import model_home
from .paddle_runtime import (
    PaddleRuntimeStatus,
    _emit,
    _probe_python,
    find_compatible_python,
    _run,
    _pip_install_with_sources,
    _pip_install,
    _install_env,
    _dist_version,
)
from .tls_support import apply_runtime_tls_environment

ProgressFn = Callable[[str], None]

# Official Apple-Silicon guide currently pins PaddlePaddle 3.2.1 and requires
# 3.2.1+.  Keep native doc parsing deterministic there; other platforms can use
# compatible 3.x CPU wheels.
APPLE_PADDLE_SPEC = "paddlepaddle==3.2.1"
GENERIC_PADDLE_SPEC = "paddlepaddle>=3.2.1,<4"
DOC_PARSER_SPEC = "paddleocr[doc-parser]>=3.7,<4"


def runtime_root() -> Path:
    return model_home().parent / "runtime" / "paddle-doc-parser"


def venv_dir() -> Path:
    return runtime_root() / ".venv"


def venv_python() -> Path:
    return venv_dir() / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _doc_marker(py: Path) -> tuple[bool, str, dict]:
    script = r'''import json,platform
from importlib.metadata import version
import paddle, paddleocr, paddlex
from paddleocr import PaddleOCRVL, PPStructureV3
print(json.dumps({
  "ok": True,
  "machine": platform.machine().lower(),
  "paddle": version("paddlepaddle"),
  "paddleocr": version("paddleocr"),
  "paddlex": version("paddlex"),
}))'''
    try:
        proc = subprocess.run([str(py), "-c", script], capture_output=True, text=True, timeout=120, env=_install_env())
    except Exception as exc:
        return False, str(exc), {}
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "document parser import failed").strip()[-5000:], {}
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return False, proc.stdout.strip()[-3000:], {}
    return bool(payload.get("ok")), proc.stdout.strip().splitlines()[-1], payload


def _pip_check(py: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run([str(py), "-m", "pip", "check"], capture_output=True, text=True, timeout=180, env=_install_env())
    except Exception as exc:
        return False, str(exc)
    text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    return proc.returncode == 0, text or "No broken requirements found."


def _state_path() -> Path:
    return runtime_root() / "install-state.json"


def _write_state(py: Path, *, sources: list[str], repaired: bool) -> None:
    ok, detail, payload = _doc_marker(py)
    check_ok, check_detail = _pip_check(py)
    data = {
        "schema": "mhd.paddle_doc_runtime.v1",
        "python": str(py),
        "ready": bool(ok and check_ok),
        "versions": payload,
        "pip_check": check_detail[-4000:],
        "dependency_sources": sources,
        "repaired": bool(repaired),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = _state_path(); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


@functools.lru_cache(maxsize=1)
def runtime_status() -> PaddleRuntimeStatus:
    py = venv_python()
    if not py.exists():
        return PaddleRuntimeStatus(False, None, "Paddle 文档解析独立运行环境尚未安装。")
    ok_py, detail_py, info = _probe_python(py)
    if not ok_py:
        return PaddleRuntimeStatus(False, str(py), "文档解析 Python 不兼容：" + detail_py)
    ok, detail, payload = _doc_marker(py)
    if not ok:
        return PaddleRuntimeStatus(False, str(py), "文档解析依赖未就绪：" + detail[-4000:], architecture=str(info.get("machine") or ""))
    check_ok, check_detail = _pip_check(py)
    if not check_ok:
        return PaddleRuntimeStatus(False, str(py), "文档解析依赖存在冲突（pip check）：\n" + check_detail[-4000:], version=str(payload.get("paddleocr") or ""), architecture=str(info.get("machine") or ""))
    return PaddleRuntimeStatus(
        True, str(py),
        f"独立文档解析环境已就绪 · Paddle {payload.get('paddle')} · PaddleOCR {payload.get('paddleocr')} · PaddleX {payload.get('paddlex')}",
        version=str(payload.get("paddleocr") or ""), architecture=str(info.get("machine") or ""),
    )




def _install_paddle_engine(py: Path, spec: str, progress: ProgressFn | None) -> str:
    """Install PaddlePaddle with the official CPU index first on macOS."""
    if platform.system() == "Darwin":
        label = "Paddle 官方 CPU wheel 源"
        _emit(progress, "PaddlePaddle 优先使用官方 CPU wheel 源（Apple Silicon 官方安装路径）…")
        code, tail = _pip_install(
            py, [spec], progress,
            extra_args=["--index-url", "https://www.paddlepaddle.org.cn/packages/stable/cpu/"],
        )
        if code == 0:
            return label
        _emit(progress, "Paddle 官方 CPU wheel 源失败，才回退到通用依赖镜像列表。")
    return _pip_install_with_sources(py, [spec], progress, paddle_package=True)

def _base_python(progress: ProgressFn | None) -> Path:
    try:
        return find_compatible_python()
    except RuntimeError as original:
        bootstrap_enabled = str(os.environ.get("MHD_PADDLE_BOOTSTRAP_PYTHON", "1") or "1").strip().lower() not in {"0", "false", "no"}
        if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"} and bootstrap_enabled:
            from .standalone_python import ensure_standalone_python
            _emit(progress, "未发现兼容 Python，正在准备独立文档解析 Python…")
            return ensure_standalone_python(progress)
        raise original


def ensure_runtime(progress: ProgressFn | None = None, *, force_repair: bool = False) -> PaddleRuntimeStatus:
    root = runtime_root(); root.mkdir(parents=True, exist_ok=True)
    py = venv_python()
    sources: list[str] = []

    if py.exists():
        ok_py, detail, _ = _probe_python(py)
        if not ok_py:
            _emit(progress, "现有文档解析 venv 不兼容，将重建：" + detail)
            shutil.rmtree(venv_dir(), ignore_errors=True)

    if not py.exists():
        base = _base_python(progress)
        ok, detail, _ = _probe_python(base)
        if not ok:
            raise RuntimeError(detail)
        _emit(progress, f"创建独立 Paddle 文档解析环境：{base}（{detail}）")
        code, tail = _run([str(base), "-m", "venv", str(venv_dir())], progress, timeout=300)
        if code != 0:
            raise RuntimeError("创建 Paddle 文档解析 venv 失败。\n" + tail[-4000:])
        py = venv_python()
        code, tail = _run([str(py), "-m", "ensurepip", "--upgrade"], progress, timeout=300)
        if code != 0:
            raise RuntimeError("初始化 Paddle 文档解析 venv 的 pip 失败。\n" + tail[-4000:])
        force_repair = True

    status = runtime_status()
    if status.ready and not force_repair:
        return status

    _emit(progress, "初始化/修复文档解析 venv 的 pip 构建工具…")
    sources.append(_pip_install_with_sources(py, ["pip", "setuptools", "wheel"], progress, paddle_package=False))

    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        _emit(progress, "按 PaddleOCR Apple Silicon 官方路径安装 PaddlePaddle 3.2.1…")
        sources.append(_install_paddle_engine(py, APPLE_PADDLE_SPEC, progress))
    else:
        _emit(progress, "安装 PaddlePaddle 3.2.1+ 文档解析运行时…")
        sources.append(_install_paddle_engine(py, GENERIC_PADDLE_SPEC, progress))

    # Follow the official dependency group directly.  Do not separately force
    # paddlex[genai-client]: native Apple-Silicon inference does not need a GenAI
    # server client, and over-constraining PaddleX can create avoidable conflicts.
    _emit(progress, "安装 PaddleOCR 官方 doc-parser 依赖组…")
    sources.append(_pip_install_with_sources(py, [DOC_PARSER_SPEC], progress, paddle_package=False))

    ok, detail, _ = _doc_marker(py)
    if not ok:
        raise RuntimeError("Paddle 文档解析依赖安装后仍无法导入。\n" + detail[-6000:])
    check_ok, check_detail = _pip_check(py)
    if not check_ok:
        raise RuntimeError("Paddle 文档解析依赖安装完成，但 pip check 发现版本冲突：\n" + check_detail[-6000:])

    _write_state(py, sources=sources, repaired=force_repair)
    runtime_status.cache_clear()
    status = runtime_status()
    if not status.ready:
        raise RuntimeError(status.detail)
    _emit(progress, status.detail)
    return status


def repair_runtime(progress: ProgressFn | None = None) -> PaddleRuntimeStatus:
    return ensure_runtime(progress, force_repair=True)


def require_runtime_python() -> Path:
    status = runtime_status()
    if not status.ready or not status.python:
        raise RuntimeError("Paddle 文档解析运行环境尚未就绪。\n" + status.detail)
    return Path(status.python)


def worker_script() -> Path:
    return Path(__file__).with_name("paddle_worker.py")


def runtime_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    return apply_runtime_tls_environment(dict(base or os.environ.copy()), runtime_root())


__all__ = [
    "runtime_root", "venv_dir", "venv_python", "runtime_status", "ensure_runtime",
    "repair_runtime", "require_runtime_python", "worker_script", "runtime_environment",
]
