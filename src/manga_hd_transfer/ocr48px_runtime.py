from __future__ import annotations

"""Pinned Manga Image Translator 48px AR OCR runtime preparation.

Folirina never downloads this optional OCR model during normal processing.
Model Center explicitly calls :func:`prepare_runtime_files`; inference only
accepts a fully verified local cache.  The model/network source remains outside
the application ZIP and is fetched from fixed upstream revisions on demand.
"""

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from typing import Callable

HF_REVISION = "3e29cd63a0ce7d1b4013b0a6e56da4cddaf4fe5b"
HF_ROOT = f"https://huggingface.co/zyddnys/manga-image-translator/resolve/{HF_REVISION}"
GITHUB_RELEASE_ROOT = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3"
MODEL_URLS = (
    f"{HF_ROOT}/ocr_ar_48px.ckpt?download=true",
    f"{GITHUB_RELEASE_ROOT}/ocr_ar_48px.ckpt",
)
DICT_URLS = (
    f"{HF_ROOT}/alphabet-all-v7.txt?download=true",
    f"{GITHUB_RELEASE_ROOT}/alphabet-all-v7.txt",
)
MODEL_SHA256 = "29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d"
DICT_SHA256 = "f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a"
MODEL_SIZE = 204_290_192
UPSTREAM_MODEL_SOURCE_SHA = "8a410854407f258a1bf5a5027beda09785cdcdd5"
UPSTREAM_XPOS_SOURCE_SHA = "cf2d9a7cb219e6590afb23b6fce6261cca134b10"
SOURCE_API_URL = "https://api.github.com/repos/zyddnys/manga-image-translator/git/blobs/" + UPSTREAM_MODEL_SOURCE_SHA
XPOS_API_URL = "https://api.github.com/repos/zyddnys/manga-image-translator/git/blobs/" + UPSTREAM_XPOS_SOURCE_SHA

ProgressFn = Callable[[int, int | None, str], None]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha_bytes(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def runtime_paths(root: Path) -> dict[str, Path]:
    root = Path(root).expanduser()
    src = root / "upstream-source"
    return {
        "model": root / "ocr_ar_48px.ckpt",
        "dictionary": root / "alphabet-all-v7.txt",
        "raw_model": src / "model_48px.py",
        "raw_xpos": src / "xpos_relative_position.py",
        "core": src / "manga_48px_core.py",
        "xpos": src / "manga_48px_xpos.py",
    }


def runtime_files_ready(root: Path) -> bool:
    p = runtime_paths(root)
    try:
        return (
            p["model"].is_file() and p["model"].stat().st_size == MODEL_SIZE
            and _sha256(p["model"]) == MODEL_SHA256
            and p["dictionary"].is_file() and _sha256(p["dictionary"]) == DICT_SHA256
            and p["core"].is_file() and p["xpos"].is_file()
        )
    except OSError:
        return False


def _emit(progress: ProgressFn | None, done: int, total: int | None, msg: str) -> None:
    if progress:
        progress(int(done), int(total) if total is not None else None, str(msg))


def _download_urllib(url: str, dst: Path, *, progress: ProgressFn | None, label: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Folirina-48px-AR/2.0.82"})
    with urllib.request.urlopen(req, timeout=60) as res:
        total = int(res.headers.get("Content-Length") or 0) or None
        done = 0
        with dst.open("wb") as out:
            while True:
                chunk = res.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk); done += len(chunk)
                _emit(progress, done, total, label)


def _download_curl(url: str, dst: Path, *, progress: ProgressFn | None, label: str) -> None:
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl unavailable")
    proc = subprocess.Popen([
        curl, "--location", "--fail", "--silent", "--show-error", "--retry", "3",
        "--connect-timeout", "20", "--max-time", "1800", "--output", str(dst), url,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    while proc.poll() is None:
        done = dst.stat().st_size if dst.exists() else 0
        _emit(progress, done, None, label + " · curl")
        try: proc.wait(timeout=.25)
        except subprocess.TimeoutExpired: pass
    stderr = proc.stderr.read() if proc.stderr else ""
    if proc.returncode != 0:
        raise RuntimeError(stderr[-1500:] or f"curl exit={proc.returncode}")


def _download_verified(urls: tuple[str, ...], dst: Path, *, sha256: str, size: int = 0,
                       progress: ProgressFn | None = None, label: str) -> Path:
    if dst.is_file():
        if (not size or dst.stat().st_size == size) and _sha256(dst) == sha256:
            _emit(progress, dst.stat().st_size, dst.stat().st_size, label + " · 已校验")
            return dst
        dst.unlink(missing_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in urls:
        for method in ("urllib", "curl"):
            tmp = dst.with_name(dst.name + ".part")
            tmp.unlink(missing_ok=True)
            try:
                if method == "urllib": _download_urllib(url, tmp, progress=progress, label=label)
                else: _download_curl(url, tmp, progress=progress, label=label)
                if size and tmp.stat().st_size != size:
                    raise RuntimeError(f"size mismatch {tmp.stat().st_size}/{size}")
                if _sha256(tmp) != sha256:
                    raise RuntimeError("SHA-256 mismatch")
                os.replace(tmp, dst)
                _emit(progress, dst.stat().st_size, dst.stat().st_size, label + " · 完成")
                return dst
            except Exception as exc:
                errors.append(f"{method}:{exc}")
                tmp.unlink(missing_ok=True)
    raise RuntimeError(label + " 下载失败：" + " | ".join(errors[-6:]))


def _fetch_verified_blob(api_url: str, dst: Path, expected_sha: str, *, progress: ProgressFn | None, label: str) -> Path:
    if dst.is_file():
        try:
            if _git_blob_sha_bytes(dst.read_bytes()) == expected_sha:
                return dst
        except OSError:
            pass
        dst.unlink(missing_ok=True)
    _emit(progress, 0, None, label)
    errors: list[str] = []
    for method in ("urllib", "curl"):
        try:
            if method == "urllib":
                req = urllib.request.Request(api_url, headers={"User-Agent":"Folirina-48px-AR/2.0.82","Accept":"application/vnd.github+json"})
                raw = urllib.request.urlopen(req, timeout=60).read()
            else:
                curl = shutil.which("curl")
                if not curl: raise RuntimeError("curl unavailable")
                raw = subprocess.check_output([curl,"--location","--fail","--silent","--show-error",api_url], timeout=120)
            payload = json.loads(raw.decode("utf-8"))
            if str(payload.get("sha") or "") != expected_sha:
                raise RuntimeError("GitHub blob SHA mismatch")
            data = base64.b64decode(str(payload.get("content") or "").replace("\n", ""))
            if _git_blob_sha_bytes(data) != expected_sha:
                raise RuntimeError("decoded blob SHA mismatch")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            return dst
        except Exception as exc:
            errors.append(f"{method}:{exc}")
    raise RuntimeError(label + " 获取失败：" + " | ".join(errors[-4:]))


def extract_model_core(source: str) -> str:
    start = source.find("class ConvNeXtBlock")
    end = source.find("\ndef convert_pl_model", start)
    if start < 0 or end <= start:
        raise RuntimeError("上游 48px OCR 源码结构与固定 checkpoint 不匹配")
    header = f'''# Generated at user-requested model preparation time.\n# Upstream blob: {UPSTREAM_MODEL_SOURCE_SHA}\nimport math\nfrom typing import Callable, List, Optional, Tuple, Union\nfrom collections import defaultdict\nimport einops\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom manga_48px_xpos import XPOS\n\n'''
    return header + source[start:end].rstrip() + "\n"


def prepare_runtime_files(root: Path, *, progress: ProgressFn | None = None) -> Path:
    """Explicitly download and verify official 48px weights + pinned source."""
    root = Path(root).expanduser(); root.mkdir(parents=True, exist_ok=True)
    p = runtime_paths(root)
    _download_verified(MODEL_URLS, p["model"], sha256=MODEL_SHA256, size=MODEL_SIZE, progress=progress, label="48px AR 官方权重")
    _download_verified(DICT_URLS, p["dictionary"], sha256=DICT_SHA256, progress=progress, label="48px AR 字符表")
    raw_model = _fetch_verified_blob(SOURCE_API_URL, p["raw_model"], UPSTREAM_MODEL_SOURCE_SHA, progress=progress, label="48px 固定模型源码")
    raw_xpos = _fetch_verified_blob(XPOS_API_URL, p["raw_xpos"], UPSTREAM_XPOS_SOURCE_SHA, progress=progress, label="48px 固定 XPOS 源码")
    p["core"].write_text(extract_model_core(raw_model.read_text(encoding="utf-8")), encoding="utf-8")
    p["xpos"].write_bytes(raw_xpos.read_bytes())
    if not runtime_files_ready(root):
        raise RuntimeError("48px AR 文件准备后校验失败")
    _emit(progress, 1, 1, "48px AR 原生运行文件已就绪")
    return root


def load_ocr_class(root: Path):
    """Load only from an already-prepared cache; never download here."""
    root = Path(root).expanduser()
    if not runtime_files_ready(root):
        raise RuntimeError("48px AR 模型未完整准备；请先在模型中心点击下载/校验。")
    p = runtime_paths(root); source_dir = p["core"].parent
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    name = "folirina_manga_48px_core"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, p["core"])
        if spec is None or spec.loader is None:
            raise RuntimeError("无法载入 48px AR 固定模型核心")
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    OCR = getattr(module, "OCR", None)
    if OCR is None:
        raise RuntimeError("48px AR 模型核心缺少 OCR 类")
    return OCR, p["model"], p["dictionary"]


__all__ = [
    "MODEL_SHA256", "DICT_SHA256", "MODEL_SIZE", "runtime_paths", "runtime_files_ready",
    "prepare_runtime_files", "load_ocr_class", "extract_model_core",
]
