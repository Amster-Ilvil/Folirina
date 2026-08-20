from __future__ import annotations

"""User-triggered compatible Python bootstrap for the isolated Paddle runtime.

The desktop GUI can legitimately run on a CPython release for which Paddle has
no wheel.  When the user presses "install Paddle dependencies" on Apple Silicon
and no compatible interpreter is installed, download a pinned python-build-
standalone runtime, verify it against the release SHA256SUMS, and use it only as
the parent interpreter of the Paddle venv.

Nothing here runs during application import/startup.
"""

import hashlib
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from typing import Callable, Iterable

from .storage_paths import model_home

ProgressFn = Callable[[str], None]

PYTHON_RELEASE = "20251010"
PYTHON_VERSION = "3.12.12"
_ARCHIVE_ARCH = {"arm64": "aarch64"}


def _emit(cb: ProgressFn | None, message: str) -> None:
    if cb is not None:
        cb(str(message))


def runtime_dir() -> Path:
    return model_home().parent / "runtime" / f"python-{PYTHON_VERSION}-standalone"


def runtime_python() -> Path:
    return runtime_dir() / "python" / "bin" / "python3"


def _download_dir() -> Path:
    return model_home().parent / "runtime" / "downloads" / "python-build-standalone"


def supported() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in _ARCHIVE_ARCH


def archive_name() -> str:
    machine = platform.machine().lower()
    arch = _ARCHIVE_ARCH.get(machine)
    if not arch:
        raise RuntimeError(f"当前架构不支持自动准备 Paddle Python：{machine or 'unknown'}")
    return f"cpython-{PYTHON_VERSION}+{PYTHON_RELEASE}-{arch}-apple-darwin-install_only_stripped.tar.gz"


def _roots() -> tuple[str, str]:
    release = PYTHON_RELEASE
    mirror = f"https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/{release}"
    official = f"https://github.com/astral-sh/python-build-standalone/releases/download/{release}"
    return mirror, official


def _curl_download(url: str, destination: Path, *, timeout: int = 1800) -> tuple[bool, str]:
    curl = shutil.which("curl") or ("/usr/bin/curl" if Path("/usr/bin/curl").exists() else None)
    if not curl:
        return False, "curl 不可用"
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    part.unlink(missing_ok=True)
    cmd = [
        str(curl), "--fail", "--location", "--retry", "3", "--retry-all-errors",
        "--connect-timeout", "20", "--speed-time", "30", "--speed-limit", "1024",
        "--output", str(part), url,
    ]
    proxy = str(os.environ.get("MHD_MODEL_PROXY", "") or "").strip()
    if proxy:
        cmd[1:1] = ["--proxy", proxy]
    ca = str(os.environ.get("MHD_CA_BUNDLE", "") or "").strip()
    if ca and Path(ca).expanduser().is_file():
        cmd[1:1] = ["--cacert", str(Path(ca).expanduser())]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        part.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0 or not part.is_file() or part.stat().st_size <= 0:
        part.unlink(missing_ok=True)
        return False, (proc.stderr or proc.stdout or f"curl exit {proc.returncode}")[-1500:]
    os.replace(part, destination)
    return True, "ok"


def _download_first(urls: Iterable[str], destination: Path, progress: ProgressFn | None, label: str, *, timeout: int = 1800) -> str:
    failures: list[str] = []
    for index, url in enumerate(urls, start=1):
        _emit(progress, f"{label}：下载源 {index} · {url.split('/')[2] if '//' in url else url}")
        ok, detail = _curl_download(url, destination, timeout=timeout)
        if ok:
            return url
        failures.append(f"{url}: {detail}")
    raise RuntimeError(f"{label} 下载失败。\n" + "\n".join(failures[-4:]))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_checksum(sums_path: Path, filename: str) -> str:
    for raw in sums_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.strip().split()
        if len(parts) < 2:
            continue
        name = parts[-1].lstrip("*")
        if name == filename and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            return parts[0].lower()
    return ""


def ensure_standalone_python(progress: ProgressFn | None = None) -> Path:
    if not supported():
        raise RuntimeError("自动准备兼容 Python 当前仅用于 Apple Silicon macOS。")
    py = runtime_python()
    if py.is_file() and os.access(py, os.X_OK):
        return py

    root = runtime_dir()
    downloads = _download_dir(); downloads.mkdir(parents=True, exist_ok=True)
    archive = archive_name()
    archive_path = downloads / archive
    sums_path = downloads / f"SHA256SUMS-{PYTHON_RELEASE}"
    mirror, official = _roots()

    _emit(progress, f"未发现兼容 Paddle Python；准备独立 Python {PYTHON_VERSION}（仅供 OCR venv 使用）…")
    if not archive_path.is_file() or archive_path.stat().st_size < 1_000_000:
        _download_first(
            (f"{mirror}/{archive}", f"{official}/{archive}"), archive_path, progress,
            "独立 Python", timeout=1800,
        )
    # Prefer the official checksum endpoint, then the mirror. The archive itself
    # may come from either source but is accepted only after checksum validation.
    if not sums_path.is_file() or sums_path.stat().st_size < 100:
        _download_first(
            (f"{official}/SHA256SUMS", f"{mirror}/SHA256SUMS"), sums_path, progress,
            "独立 Python SHA-256 清单", timeout=300,
        )
    expected = _expected_checksum(sums_path, archive)
    actual = _sha256(archive_path)
    if not expected or actual.lower() != expected:
        archive_path.unlink(missing_ok=True)
        sums_path.unlink(missing_ok=True)
        raise RuntimeError("独立 Python SHA-256 校验失败，下载文件已删除。")

    staging = root.with_name(root.name + ".new")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    _emit(progress, "独立 Python 校验通过，正在解压…")
    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            # Python 3.12 supports the data filter. Fall back only for older GUI
            # Pythons; this is a pinned, checksum-verified upstream archive.
            try:
                tf.extractall(staging, filter="data")
            except TypeError:
                tf.extractall(staging)
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(f"独立 Python 压缩包无法解压：{exc}") from exc

    staged_py = staging / "python" / "bin" / "python3"
    if not staged_py.is_file():
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("独立 Python 解压完成但缺少 python/bin/python3。")
    try:
        staged_py.chmod(staged_py.stat().st_mode | 0o111)
    except Exception:
        pass
    shutil.rmtree(root, ignore_errors=True)
    os.replace(staging, root)
    py = runtime_python()
    _emit(progress, f"独立 Python {PYTHON_VERSION} 已准备：{py}")
    return py
