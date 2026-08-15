from __future__ import annotations

"""Explicit optional-dependency installer used by the GUI.

No dependency is installed at import/startup time.  This module is only invoked
from a worker after the user presses an Install button.
"""

from dataclasses import dataclass
import importlib.util
import os
import platform
import subprocess
import sys
from typing import Callable

ProgressFn = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class DependencyInstallResult:
    key: str
    message: str
    restart_recommended: bool = True


def _emit(cb: ProgressFn | None, message: str) -> None:
    if cb is not None:
        cb(str(message))


def _run(cmd: list[str], *, progress: ProgressFn | None = None) -> tuple[int, str]:
    _emit(progress, "执行：" + " ".join(cmd))
    env = os.environ.copy()
    proxy = str(env.get("MHD_MODEL_PROXY", "") or "").strip()
    if proxy:
        env["HTTP_PROXY"] = proxy; env["HTTPS_PROXY"] = proxy
        env["http_proxy"] = proxy; env["https_proxy"] = proxy
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
        if len(tail) > 30:
            tail = tail[-30:]
        # Keep the GUI useful without flooding it with every pip progress line.
        if any(token in line.lower() for token in ("error", "failed", "installing", "successfully", "collecting", "looking in indexes")):
            _emit(progress, line)
    code = int(proc.wait())
    return code, "\n".join(tail)


def _pip_install(args: list[str], *, progress: ProgressFn | None = None) -> tuple[int, str]:
    return _run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *args], progress=progress)


def install_paddle_ocr_dependencies(progress: ProgressFn | None = None) -> DependencyInstallResult:
    """Install PaddlePaddle + PaddleOCR into the exact Python running the GUI.

    macOS uses PaddlePaddle's official CPU wheel index because that is the
    publisher-supported Apple Silicon route.  PaddleOCR itself is then installed
    from the normal Python package index; if that fails, a Tsinghua PyPI mirror is
    attempted as a connectivity fallback.
    """
    _emit(progress, f"当前 Python：{sys.executable}")
    system = platform.system()
    machine = platform.machine().lower()

    if importlib.util.find_spec("paddle") is None:
        if system == "Darwin":
            # Paddle's current macOS documentation uses the official stable CPU
            # index; Apple Silicon is supported natively.
            primary = ["--upgrade", "paddlepaddle==3.2.0", "-i", "https://www.paddlepaddle.org.cn/packages/stable/cpu/"]
            fallback = ["--upgrade", "paddlepaddle>=3.0,<4"]
        else:
            primary = ["--upgrade", "paddlepaddle>=3.0,<4"]
            fallback = ["--upgrade", "paddlepaddle>=3.0,<4", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]
        code, tail = _pip_install(primary, progress=progress)
        if code != 0:
            _emit(progress, "PaddlePaddle 官方源安装失败，尝试备用 Python 包源…")
            code, tail2 = _pip_install(fallback, progress=progress)
            tail = (tail + "\n" + tail2).strip()
        if code != 0:
            raise RuntimeError(
                "PaddlePaddle 安装失败。请检查 DNS/代理后重试；也可以在可联网机器下载对应 wheel 后离线安装。\n" + tail[-4000:]
            )
    else:
        _emit(progress, "PaddlePaddle 已安装，跳过。")

    if importlib.util.find_spec("paddleocr") is None:
        code, tail = _pip_install(["--upgrade", "paddleocr>=3.0,<4"], progress=progress)
        if code != 0:
            _emit(progress, "PyPI 直连安装 PaddleOCR 失败，尝试清华 PyPI 镜像…")
            code, tail2 = _pip_install(
                ["--upgrade", "paddleocr>=3.0,<4", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
                progress=progress,
            )
            tail = (tail + "\n" + tail2).strip()
        if code != 0:
            raise RuntimeError(
                "PaddleOCR 安装失败。请检查 DNS/代理后重试；也可先离线安装依赖 wheel。\n" + tail[-4000:]
            )
    else:
        _emit(progress, "PaddleOCR 已安装，跳过。")

    arch = f"{system} {machine}".strip()
    return DependencyInstallResult(
        "paddle",
        f"PP-OCR 运行依赖已安装到当前 Python（{arch}）。建议重启程序后再下载/预热 PP-OCRv5 模型。",
        True,
    )
