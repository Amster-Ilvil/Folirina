from __future__ import annotations

"""Explicit, user-triggered download support for built-in optional models.

Nothing in this module downloads at import/probe time.  The GUI must call
``download_builtin_model`` from a worker only after the user presses Download.
Weights are written atomically through ``.part`` files and known upstream hashes
are verified where the publisher exposes them.
"""

from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen, build_opener, ProxyHandler
import socket
import shutil

from .config import PipelineConfig

ProgressFn = Callable[[int, int | None, str], None]


@dataclass(frozen=True, slots=True)
class ModelDownloadResult:
    key: str
    label: str
    path: str | None
    message: str
    config_updates: dict[str, Any] = field(default_factory=dict)


MODEL_LABELS = {
    "paddle": "PP-OCRv5（中/日文）",
    "lightglue": "LightGlue · SIFT",
    "loftr": "LoFTR · outdoor",
    "mangalens": "MangaLens · YOLO11n Bubble",
    "rtdetr_v2": "Comic Translate RT-DETR-v2",
    "sam2": "SAM 2.1 · Hiera Tiny",
}

_MODEL_PROXY_OVERRIDE: str = ""


def _open_url(req: Request, *, timeout: float):
    proxy = str(_MODEL_PROXY_OVERRIDE or "").strip()
    if proxy:
        opener = build_opener(ProxyHandler({"http": proxy, "https": proxy}))
        return opener.open(req, timeout=timeout)
    # Default urllib opener respects the user's existing system/environment proxy.
    return urlopen(req, timeout=timeout)


def model_home() -> Path:
    override = os.environ.get("MHD_MODEL_HOME", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    home = Path.home()
    system = platform.system()
    if system == "Darwin":
        return home / "Library" / "Application Support" / "Manga HD Transfer Studio" / "models"
    if system == "Windows":
        root = Path(os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local")))
        return root / "MangaHDTransfer" / "models"
    return home / ".local" / "share" / "manga-hd-transfer" / "models"


def torch_checkpoint_dir() -> Path:
    root = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
    return root / "hub" / "checkpoints"


def model_local_paths() -> dict[str, Path]:
    root = model_home()
    return {
        "lightglue": torch_checkpoint_dir() / "sift_lightglue_v0-1_arxiv.pth",
        "loftr": torch_checkpoint_dir() / "loftr_outdoor.ckpt",
        "mangalens": root / "mangalens" / "best.pt",
        "rtdetr_v2": root / "rtdetr_v2" / "comic-text-and-bubble-detector",
        "sam2": root / "sam2" / "sam2.1_hiera_tiny.pt",
        "paddle": root / "paddle" / "PP-OCRv5.ready.json",
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _configured_hf_endpoints() -> list[str]:
    """Ordered HF endpoints for explicit downloads.

    ``MHD_HF_SOURCE`` may be ``official``, ``mirror`` or ``auto``.  ``HF_ENDPOINT``
    is respected as a custom endpoint.  Auto tries the official endpoint first,
    then the commonly used mainland mirror, so a DNS failure does not end the
    download immediately.
    """
    custom = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    mode = os.environ.get("MHD_HF_SOURCE", "auto").strip().lower()
    official = "https://huggingface.co"
    mirror = "https://hf-mirror.com"
    if custom and custom not in {official, mirror}:
        return [custom, official, mirror]
    if mode == "mirror":
        return [mirror, official]
    if mode == "official":
        return [official]
    return [official, mirror]


def _hf_urls(repo: str, filename: str) -> list[str]:
    return [f"{base}/{repo}/resolve/main/{filename}?download=true" for base in _configured_hf_endpoints()]


def _download_file(
    url: str | list[str],
    destination: Path,
    *,
    sha256: str | None = None,
    progress: ProgressFn | None = None,
    label: str = "model",
    timeout: float = 60.0,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256 is None or _sha256(destination).lower() == sha256.lower():
            if progress:
                progress(destination.stat().st_size, destination.stat().st_size, f"{label} 已存在")
            return destination
        bad = destination.with_suffix(destination.suffix + ".invalid")
        try:
            if bad.exists():
                bad.unlink()
            destination.replace(bad)
        except OSError:
            destination.unlink(missing_ok=True)

    urls = [url] if isinstance(url, str) else list(url)
    if not urls:
        raise RuntimeError(f"{label} 没有可用下载源")
    errors: list[str] = []
    for source_index, current_url in enumerate(urls):
        part = destination.with_suffix(destination.suffix + ".part")
        existing = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "Manga-HD-Translation-Transfer/model-downloader"}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
        req = Request(current_url, headers=headers)
        try:
            if progress:
                progress(existing, None, f"{label} · 源 {source_index + 1}/{len(urls)}")
            try:
                resp = _open_url(req, timeout=timeout)
            except HTTPError as exc:
                if existing > 0 and exc.code in {400, 403, 416}:
                    part.unlink(missing_ok=True)
                    existing = 0
                    req = Request(current_url, headers={"User-Agent": headers["User-Agent"]})
                    resp = _open_url(req, timeout=timeout)
                else:
                    raise

            status = getattr(resp, "status", None) or getattr(resp, "code", 200)
            if existing > 0 and int(status) != 206:
                existing = 0
                mode = "wb"
            else:
                mode = "ab" if existing > 0 else "wb"
            length_raw = resp.headers.get("Content-Length")
            try:
                remaining = int(length_raw) if length_raw else None
            except (TypeError, ValueError):
                remaining = None
            total = (existing + remaining) if remaining is not None else None
            done = existing
            with part.open(mode) as f:
                while True:
                    chunk = resp.read(1024 * 512)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total, label)
                f.flush()
                os.fsync(f.fileno())
            resp.close()
            if total is not None and part.stat().st_size != total:
                raise RuntimeError(f"{label} 下载不完整：{part.stat().st_size} / {total} bytes")
            if sha256:
                got = _sha256(part)
                if got.lower() != sha256.lower():
                    raise RuntimeError(f"{label} SHA-256 校验失败：{got}")
            os.replace(part, destination)
            return destination
        except Exception as exc:  # try the next source
            errors.append(f"{current_url}: {exc}")
            if progress:
                progress(existing, None, f"{label} 当前下载源失败，尝试备用源…")
            continue

    detail = "\n".join(errors[-4:])
    raise RuntimeError(
        f"{label} 所有下载源均失败。若提示 DNS / nodename 错误，请在模型中心填写本机代理，或使用‘离线导入’。\n{detail}"
    )


def _hf(repo: str, filename: str) -> str:
    # Kept for compatibility with callers/tests; explicit downloads use
    # ``_hf_urls`` so official + mirror fallback is available.
    return f"https://huggingface.co/{repo}/resolve/main/{filename}?download=true"


def diagnose_download_network(timeout: float = 4.0) -> dict[str, dict[str, Any]]:
    """Small DNS/TCP diagnostic for the GUI; never downloads a model."""
    hosts = [
        "www.paddlepaddle.org.cn",
        "pypi.org",
        "huggingface.co",
        "hf-mirror.com",
        "github.com",
    ]
    out: dict[str, dict[str, Any]] = {}
    for host in hosts:
        try:
            rows = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            addresses = sorted({str(row[4][0]) for row in rows})[:3]
            out[host] = {"dns": True, "addresses": addresses}
        except Exception as exc:
            out[host] = {"dns": False, "error": str(exc)}
    return out


def apply_download_network_settings(*, proxy: str = "", hf_source: str = "auto") -> None:
    """Apply process-local download settings without clobbering system proxies."""
    global _MODEL_PROXY_OVERRIDE
    proxy = str(proxy or "").strip()
    hf_source = str(hf_source or "auto").strip().lower()
    os.environ["MHD_HF_SOURCE"] = hf_source if hf_source in {"auto", "official", "mirror"} else "auto"
    _MODEL_PROXY_OVERRIDE = proxy
    if proxy:
        os.environ["MHD_MODEL_PROXY"] = proxy
    else:
        os.environ.pop("MHD_MODEL_PROXY", None)


def _copytree_replace(source: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination


def import_builtin_model(key: str, source: str | Path, config: PipelineConfig | None = None) -> ModelDownloadResult:
    """Import a model obtained on another machine into the app's local cache."""
    key = str(key)
    src = Path(source).expanduser().resolve()
    paths = model_local_paths()
    if key not in MODEL_LABELS:
        raise KeyError(f"unknown built-in model: {key}")
    if key == "rtdetr_v2":
        if not src.is_dir() or not all((src / n).is_file() for n in ("config.json", "preprocessor_config.json", "model.safetensors")):
            raise ValueError("RT-DETR-v2 离线目录必须包含 config.json、preprocessor_config.json、model.safetensors。")
        dst = _copytree_replace(src, paths[key])
        return ModelDownloadResult(key, MODEL_LABELS[key], str(dst), "RT-DETR-v2 已离线导入并自动配置。", {"bubbles.rtdetr_model_path": str(dst), "bubbles.rtdetr_allow_model_downloads": False})
    if key == "paddle":
        if not src.is_dir():
            raise ValueError("PP-OCR 离线导入请选择包含官方模型目录的文件夹。")
        det_candidates = sorted([x for x in src.rglob("*") if x.is_dir() and "pp-ocrv5" in x.name.lower() and "det" in x.name.lower()])
        rec_candidates = sorted([x for x in src.rglob("*") if x.is_dir() and "pp-ocrv5" in x.name.lower() and "rec" in x.name.lower()])
        if not det_candidates or not rec_candidates:
            raise ValueError("未找到 PP-OCRv5 detection/recognition 模型目录。请选中 PaddleX official_models 或包含这两个模型的父目录。")
        root = model_home() / "paddle" / "offline"
        det_dst = _copytree_replace(det_candidates[0], root / det_candidates[0].name)
        rec_dst = _copytree_replace(rec_candidates[0], root / rec_candidates[0].name)
        marker = paths["paddle"]
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"offline": True, "det": str(det_dst), "rec": str(rec_dst)}, ensure_ascii=False, indent=2), encoding="utf-8")
        return ModelDownloadResult(key, MODEL_LABELS[key], str(root), "PP-OCRv5 模型已离线导入；将直接使用本地 detection/recognition 模型。", {"ocr.paddle_text_detection_model_dir": str(det_dst), "ocr.paddle_text_recognition_model_dir": str(rec_dst)})

    if not src.is_file():
        raise ValueError(f"{MODEL_LABELS[key]} 离线导入需要选择模型文件。")
    dst = paths[key]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    updates: dict[str, Any] = {}
    if key == "mangalens":
        updates["bubbles.mangalens_model_path"] = str(dst)
    elif key == "sam2":
        updates.update({"bubbles.sam2_checkpoint": str(dst), "bubbles.sam2_allow_model_downloads": False})
    return ModelDownloadResult(key, MODEL_LABELS[key], str(dst), f"{MODEL_LABELS[key]} 已离线导入。", updates)

def _download_paddle(cfg: PipelineConfig, progress: ProgressFn | None) -> ModelDownloadResult:
    if importlib.util.find_spec("paddle") is None or importlib.util.find_spec("paddleocr") is None:
        raise RuntimeError("PP-OCR 运行依赖未安装。请先点“安装 PP-OCR 依赖”，完成后重启程序再下载/预热模型。")
    from paddleocr import PaddleOCR

    # PaddleOCR/PaddleX supports BOS as an official model source. Prefer it for
    # the explicit preheat action so Hugging Face DNS failures do not block OCR.
    old_source = os.environ.get("PADDLE_PDX_MODEL_SOURCE")
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"
    langs = [("ch", "中文 PP-OCRv5"), ("japan", "日文 PP-OCRv5")]
    try:
        for i, (lang, label) in enumerate(langs):
            if progress:
                progress(i, len(langs), f"初始化 {label}（Paddle 官方 BOS 模型源）")
            kwargs: dict[str, Any] = {
                "lang": lang,
                "ocr_version": cfg.ocr.ocr_version,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            }
            if getattr(cfg.ocr, "paddle_text_detection_model_dir", None):
                kwargs["text_detection_model_dir"] = str(cfg.ocr.paddle_text_detection_model_dir)
            if getattr(cfg.ocr, "paddle_text_recognition_model_dir", None):
                kwargs["text_recognition_model_dir"] = str(cfg.ocr.paddle_text_recognition_model_dir)
            engine = PaddleOCR(**kwargs)
            del engine
    finally:
        if old_source is None:
            os.environ.pop("PADDLE_PDX_MODEL_SOURCE", None)
        else:
            os.environ["PADDLE_PDX_MODEL_SOURCE"] = old_source
    marker = model_local_paths()["paddle"]
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"ocr_version": cfg.ocr.ocr_version, "langs": ["ch", "japan"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress:
        progress(len(langs), len(langs), "PP-OCRv5 中/日文模型已预热")
    return ModelDownloadResult("paddle", MODEL_LABELS["paddle"], str(marker), "PP-OCRv5 中/日文模型已主动下载/预热。")


def download_builtin_model(key: str, config: PipelineConfig, progress: ProgressFn | None = None) -> ModelDownloadResult:
    key = str(key)
    if key not in MODEL_LABELS:
        raise KeyError(f"unknown built-in model: {key}")
    label = MODEL_LABELS[key]
    paths = model_local_paths()

    if key == "paddle":
        return _download_paddle(config, progress)

    if key == "lightglue":
        path = _download_file(
            "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/sift_lightglue.pth",
            paths[key], progress=progress, label=label,
        )
        return ModelDownloadResult(key, label, str(path), "LightGlue SIFT 权重已下载到 Torch 原生缓存；Auto 配准可离线升级。")

    if key == "loftr":
        path = _download_file(
            _hf_urls("kornia/loftr", "loftr_outdoor.ckpt") + ["http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt"],
            paths[key], progress=progress, label=label,
        )
        return ModelDownloadResult(key, label, str(path), "LoFTR outdoor 权重已下载到 Torch 原生缓存；后续可离线使用。")

    if key == "mangalens":
        path = _download_file(
            _hf_urls("huyvux3005/manga109-segmentation-bubble", "best.pt"),
            paths[key],
            sha256="4028152940f7c910f40192f46ede3b3f6c7129e5c76849c324d3564f8ac50198",
            progress=progress, label=label,
        )
        return ModelDownloadResult(
            key, label, str(path), "MangaLens best.pt 已下载并自动配置。",
            {"bubbles.mangalens_model_path": str(path)},
        )

    if key == "rtdetr_v2":
        root = paths[key]
        root.mkdir(parents=True, exist_ok=True)
        files = [
            ("config.json", None),
            ("preprocessor_config.json", None),
            ("model.safetensors", "037930a861e67870eb345be01b28cc70d7e2b7956528e48ee0ebdb0c093df80d"),
        ]
        for i, (name, digest) in enumerate(files):
            def cb(done: int, total: int | None, msg: str, *, _i=i, _name=name):
                if progress:
                    prefix = f"{_i + 1}/{len(files)} {_name}"
                    progress(done, total, prefix)
            _download_file(_hf_urls("ogkalu/comic-text-and-bubble-detector", name), root / name, sha256=digest, progress=cb, label=name)
        return ModelDownloadResult(
            key, label, str(root), "RT-DETR-v2 本地模型目录已下载并自动配置；运行时不再需要联网。",
            {"bubbles.rtdetr_model_path": str(root), "bubbles.rtdetr_allow_model_downloads": False},
        )

    if key == "sam2":
        path = _download_file(
            _hf_urls("facebook/sam2.1-hiera-tiny", "sam2.1_hiera_tiny.pt"),
            paths[key],
            sha256="7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
            progress=progress, label=label,
        )
        return ModelDownloadResult(
            key, label, str(path), "SAM 2.1 Hiera Tiny checkpoint 已下载并自动配置。",
            {"bubbles.sam2_checkpoint": str(path), "bubbles.sam2_allow_model_downloads": False},
        )

    raise AssertionError(key)


def apply_config_updates(config: PipelineConfig, updates: dict[str, Any]) -> None:
    for dotted, value in (updates or {}).items():
        parts = str(dotted).split(".")
        obj: Any = config
        for name in parts[:-1]:
            obj = getattr(obj, name)
        setattr(obj, parts[-1], value)


def discovered_paddle_model_dirs() -> tuple[Path | None, Path | None]:
    """Return explicitly imported PP-OCRv5 det/rec directories across restarts."""
    marker = model_local_paths()["paddle"]
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            det = Path(str(payload.get("det", ""))).expanduser() if payload.get("det") else None
            rec = Path(str(payload.get("rec", ""))).expanduser() if payload.get("rec") else None
            if det is not None and not det.is_dir():
                det = None
            if rec is not None and not rec.is_dir():
                rec = None
            if det is not None or rec is not None:
                return det, rec
        except Exception:
            pass
    root = model_home() / "paddle" / "offline"
    if root.is_dir():
        dets = sorted([x for x in root.iterdir() if x.is_dir() and "det" in x.name.lower()])
        recs = sorted([x for x in root.iterdir() if x.is_dir() and "rec" in x.name.lower()])
        return (dets[0] if dets else None, recs[0] if recs else None)
    return None, None


def discovered_model_path(key: str, configured: str | Path | None = None) -> Path | None:
    """Resolve a configured model first, then the explicit-download cache.

    This makes downloaded built-in models persist across app restarts even when
    the user did not export a custom config file.
    """
    if configured:
        p = Path(configured).expanduser()
        if key == "rtdetr_v2":
            if p.is_dir() and all((p / n).is_file() for n in ("config.json", "preprocessor_config.json", "model.safetensors")):
                return p
        elif p.is_file():
            return p
    local = model_local_paths().get(str(key))
    if local is None:
        return None
    if key == "rtdetr_v2":
        if local.is_dir() and all((local / n).is_file() for n in ("config.json", "preprocessor_config.json", "model.safetensors")):
            return local
        return None
    return local if local.is_file() else None
