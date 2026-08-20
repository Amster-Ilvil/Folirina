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
import logging
import os
from pathlib import Path
import platform
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen, build_opener, ProxyHandler, HTTPSHandler
import ssl
import socket
import shutil
import subprocess

from .config import PipelineConfig
from .storage_paths import model_home

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int | None, str], None]


@dataclass(frozen=True, slots=True)
class ModelDownloadResult:
    key: str
    label: str
    path: str | None
    message: str
    config_updates: dict[str, Any] = field(default_factory=dict)


MODEL_LABELS = {
    "paddle": "PaddleOCR（PP-OCRv6 / v5）",
    "lightglue": "LightGlue · SIFT",
    "loftr": "LoFTR · outdoor",
    "mangalens": "MangaLens · YOLO11n Bubble",
    "ysg_obb": "YSG YOLO OBB · Open/Rotated Text",
    "rtdetr_v2": "Comic Translate RT-DETR-v2",
    "sam2": "SAM 2.1 · Hiera Tiny",
    "koharu_layout": "Koharu Layout RF-DETR Seg 2XL",
    "manga_ocr": "Manga OCR",
    "baberu_ocr": "Baberu OCR · ONNX",
    "ocr48px": "48px AR OCR",
    "lama_manga": "LaMa Manga",
    "aot_inpainting": "AOT Inpainting",
    "flux2_klein": "FLUX.2 Klein",
    "rorem_mixed": "RORem Mixed",
}

_MODEL_PROXY_OVERRIDE: str = ""


def _open_url(req: Request, *, timeout: float):
    proxy = str(_MODEL_PROXY_OVERRIDE or "").strip()
    ca_bundle = str(os.environ.get("MHD_CA_BUNDLE", "") or "").strip()
    context = None
    if ca_bundle and Path(ca_bundle).expanduser().is_file():
        # Keep verification enabled, but allow a user/admin supplied CA (for
        # example a campus/company HTTPS proxy certificate).
        context = ssl.create_default_context(cafile=str(Path(ca_bundle).expanduser()))
    handlers = []
    if proxy:
        handlers.append(ProxyHandler({"http": proxy, "https": proxy}))
    if context is not None:
        handlers.append(HTTPSHandler(context=context))
    if handlers:
        return build_opener(*handlers).open(req, timeout=timeout)
    # Default urllib opener respects the user's existing system/environment proxy.
    return urlopen(req, timeout=timeout)


def torch_checkpoint_dir() -> Path:
    root = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
    return root / "hub" / "checkpoints"


def model_local_paths() -> dict[str, Path]:
    root = model_home()
    return {
        "lightglue": torch_checkpoint_dir() / "sift_lightglue_v0-1_arxiv.pth",
        "loftr": torch_checkpoint_dir() / "loftr_outdoor.ckpt",
        "mangalens": root / "mangalens" / "best.pt",
        "ysg_obb": root / "ysg_obb" / "ysgyolo_yolo26_2.0.pt",
        "rtdetr_v2": root / "rtdetr_v2" / "comic-text-and-bubble-detector",
        "sam2": root / "sam2" / "sam2.1_hiera_tiny.pt",
        "paddle": root / "paddle" / "PP-OCRv5.ready.json",
        "koharu_layout": root / "koharu_layout" / "koharu-layout-rfdetr-seg-2xl-1152",
        "manga_ocr": root / "manga_ocr" / "mayocream-manga-ocr",
        "baberu_ocr": root / "baberu_ocr",
        "ocr48px": root / "ocr48px" / "manga-image-translator-48px-ar",
        "lama_manga": root / "inpainting" / "lama-manga.safetensors",
        "aot_inpainting": root / "inpainting" / "aot-inpainting",
        "flux2_klein": root / "inpainting" / "FLUX.2-klein-base-4B",
        "rorem_mixed": root / "inpainting" / "RORem-mixed-GGUF",
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


def _hf_urls(repo: str, filename: str, revision: str = "main") -> list[str]:
    rev = str(revision or "main")
    return [f"{base}/{repo}/resolve/{rev}/{filename}?download=true" for base in _configured_hf_endpoints()]


def _hf_revision_fallback_urls(repo: str, filename: str, revision: str = "main") -> list[str]:
    """Try the verified/pinned revision first, then current main.

    Koharu's model repository can receive packaging fixes while Folirina keeps a
    tested revision pinned.  A stale/temporarily unavailable revision should not
    make the GUI downloader unusable.  SHA-256 is still enforced whenever the
    model spec provides one, so a main fallback cannot silently replace verified
    weights with different bytes.
    """
    rev = str(revision or "main")
    urls = _hf_urls(repo, filename, rev)
    if rev != "main":
        urls.extend(_hf_urls(repo, filename, "main"))
    return list(dict.fromkeys(urls))


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
            logger.info(
                "model download ready label=%s source=%s destination=%s bytes=%s sha256_verified=%s",
                label, current_url, destination, destination.stat().st_size, bool(sha256),
            )
            return destination
        except Exception as exc:  # try the next source
            errors.append(f"{current_url}: {exc}")
            if "SHA-256 校验失败" in str(exc):
                part.unlink(missing_ok=True)
                existing = 0
            logger.exception(
                "model download source failed label=%s source=%s destination=%s partial_bytes=%s",
                label, current_url, destination, existing,
            )
            if progress:
                progress(existing, None, f"{label} 当前下载源失败，尝试备用源…")
            continue

    detail = "\n".join(errors[-6:])
    logger.error(
        "model download exhausted all sources label=%s destination=%s failures=%s",
        label, destination, detail,
    )
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
        "pypi.tuna.tsinghua.edu.cn",
        "mirrors.aliyun.com",
        "repo.huaweicloud.com",
        "huggingface.co",
        "modelscope.cn",
        "aistudio.baidu.com",
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


def apply_download_network_settings(*, proxy: str = "", hf_source: str = "auto", ca_bundle: str = "", paddle_source: str = "auto") -> None:
    """Apply process-local model/dependency network settings.

    Paddle source selection is independent from the generic Hugging Face mirror
    selector because PaddleX supports ModelScope/BOS/AIStudio/HuggingFace hubs.
    """
    global _MODEL_PROXY_OVERRIDE
    from .paddle_sources import normalize_paddle_model_source
    proxy = str(proxy or "").strip()
    hf_source = str(hf_source or "auto").strip().lower()
    ca_bundle = str(ca_bundle or "").strip()
    os.environ["MHD_HF_SOURCE"] = hf_source if hf_source in {"auto", "official", "mirror"} else "auto"
    paddle_source = normalize_paddle_model_source(paddle_source)
    os.environ["MHD_PADDLE_MODEL_SOURCE"] = paddle_source
    _MODEL_PROXY_OVERRIDE = proxy
    if proxy:
        os.environ["MHD_MODEL_PROXY"] = proxy
    else:
        os.environ.pop("MHD_MODEL_PROXY", None)
    if ca_bundle:
        path = Path(ca_bundle).expanduser()
        if not path.is_file():
            raise ValueError(f"CA 证书文件不存在：{path}")
        os.environ["MHD_CA_BUNDLE"] = str(path.resolve())
    else:
        os.environ.pop("MHD_CA_BUNDLE", None)


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
            raise ValueError("未找到 PaddleOCR detection/recognition 模型目录。请选中 PaddleX official_models 或包含这两个模型的父目录。")
        root = model_home() / "paddle" / "offline"
        det_dst = _copytree_replace(det_candidates[0], root / det_candidates[0].name)
        rec_dst = _copytree_replace(rec_candidates[0], root / rec_candidates[0].name)
        marker = paths["paddle"]
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"offline": True, "det": str(det_dst), "rec": str(rec_dst)}, ensure_ascii=False, indent=2), encoding="utf-8")
        return ModelDownloadResult(key, MODEL_LABELS[key], str(root), "PaddleOCR 模型已离线导入；将直接使用本地 detection/recognition 模型。", {"ocr.paddle_text_detection_model_dir": str(det_dst), "ocr.paddle_text_recognition_model_dir": str(rec_dst)})

    directory_models = {
        "koharu_layout": ("inference_config.json", "load_model.py", "model.safetensors"),
        "manga_ocr": ("config.json", "preprocessor_config.json", "model.safetensors"),
        "baberu_ocr": ("onnx_infer.py", "onnx/vision_int4.onnx", "onnx/decoder_prefill_int8.onnx", "onnx/decoder_step_int8.onnx", "tokenizer/vocab.json"),
        "ocr48px": ("ocr_ar_48px.ckpt", "alphabet-all-v7.txt", "upstream-source/manga_48px_core.py", "upstream-source/manga_48px_xpos.py"),
        "aot_inpainting": ("config.json", "model.safetensors"),
        "rorem_mixed": ("rorem-mixed-unet-q4_K.gguf", "sdxl-version-marker.safetensors"),
    }
    if key in directory_models:
        if not src.is_dir() or not all((src / n).is_file() for n in directory_models[key]):
            required = "、".join(directory_models[key])
            raise ValueError(f"{MODEL_LABELS[key]} 离线目录缺少必要文件：{required}")
        dst = _copytree_replace(src, paths[key])
        updates: dict[str, Any] = {}
        if key == "koharu_layout":
            updates["bubbles.koharu_layout_model_path"] = str(dst)
        elif key in {"manga_ocr", "baberu_ocr", "ocr48px"}:
            updates[f"ocr.{key}_model_path"] = str(dst)
        elif key == "aot_inpainting":
            updates["inpainting.aot_model_path"] = str(dst)
        return ModelDownloadResult(key, MODEL_LABELS[key], str(dst), f"{MODEL_LABELS[key]} 已离线导入。", updates)

    if key == "flux2_klein":
        if not src.is_dir():
            raise ValueError(f"{MODEL_LABELS[key]} 体积较大，离线导入请选择完整模型目录。")
        dst = _copytree_replace(src, paths[key])
        field = "flux2_klein_model_path" if key == "flux2_klein" else "rorem_mixed_model_path"
        return ModelDownloadResult(
            key, MODEL_LABELS[key], str(dst), f"{MODEL_LABELS[key]} 完整目录已离线导入。",
            {f"inpainting.{field}": str(dst)},
        )

    if not src.is_file():
        raise ValueError(f"{MODEL_LABELS[key]} 离线导入需要选择模型文件。")
    dst = paths[key]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    updates: dict[str, Any] = {}
    if key == "mangalens":
        updates["bubbles.mangalens_model_path"] = str(dst)
    elif key == "ysg_obb":
        updates["bubbles.ysg_obb_model_path"] = str(dst)
    elif key == "sam2":
        updates.update({"bubbles.sam2_checkpoint": str(dst), "bubbles.sam2_allow_model_downloads": False})
    elif key == "lama_manga":
        updates["inpainting.lama_model_path"] = str(dst)
    return ModelDownloadResult(key, MODEL_LABELS[key], str(dst), f"{MODEL_LABELS[key]} 已离线导入。", updates)

def _download_paddle(cfg: PipelineConfig, progress: ProgressFn | None) -> ModelDownloadResult:
    from .paddle_runtime import ensure_runtime, worker_script as ocr_worker_script, runtime_root as ocr_runtime_root
    from .paddle_doc_runtime import ensure_runtime as ensure_doc_runtime, repair_runtime as repair_doc_runtime, worker_script as doc_worker_script, runtime_root as doc_runtime_root
    from .paddle_sources import paddle_model_source_attempts, paddle_source_environment, PADDLE_MODEL_SOURCE_LABELS
    from .paddle_profiles import resolve_paddle_model_selection
    from .tls_support import apply_runtime_tls_environment

    def runtime_progress(message: str) -> None:
        if progress:
            progress(0, None, str(message))

    selection = resolve_paddle_model_selection(cfg.ocr)
    pipeline = str(selection.get("pipeline") or "ocr")
    pipeline_version = str(selection.get("pipeline_version") or "")
    # Optional doc-parser dependencies are installed only for the selected
    # document parser. Normal PP-OCRv6 never pulls these heavier extras.
    status = ensure_doc_runtime(runtime_progress) if pipeline in {"vl", "structure"} else ensure_runtime(runtime_progress)
    if not status.python:
        raise RuntimeError("Paddle 独立运行环境未返回可用 Python。")
    py = Path(status.python)
    model_label = str(selection["label"] or selection["profile"])
    det_name = str(selection["det_name"] or "")
    rec_name = str(selection["rec_name"] or "")

    # Local det/rec directories apply only to the classic OCR pipeline. VL and
    # Structure own their complete parser pipeline and must not inherit those
    # directories accidentally.
    local_det = getattr(cfg.ocr, "paddle_text_detection_model_dir", None) if pipeline == "ocr" else None
    local_rec = getattr(cfg.ocr, "paddle_text_recognition_model_dir", None) if pipeline == "ocr" else None
    preferred = str(getattr(cfg.ocr, "paddle_model_source", "auto") or "auto")
    sources = ("local",) if pipeline == "ocr" and local_det and local_rec else paddle_model_source_attempts(preferred)

    # Explicit v6 pairs and document parsers need one initialization probe. The
    # legacy convenience profile retains the historical Chinese+Japanese warmup.
    if pipeline in {"vl", "structure"} or det_name or rec_name:
        langs = [("japan", model_label)]
    else:
        langs = [("ch", "中文 " + model_label), ("japan", "日文 " + model_label)]

    failures: list[str] = []
    dependency_repair_attempted = False
    chosen_source = "local" if sources == ("local",) else ""
    final_payload: dict[str, Any] = {}
    for source_index, source in enumerate(sources, start=1):
        label = "本地离线模型" if source == "local" else PADDLE_MODEL_SOURCE_LABELS.get(source, source)
        source_ok = True
        source_details: list[str] = []
        for lang_index, (lang, lang_label) in enumerate(langs, start=1):
            if progress:
                progress(
                    (source_index - 1) * len(langs) + lang_index - 1,
                    max(1, len(sources) * len(langs)),
                    f"{model_label} · 模型源 {source_index}/{len(sources)}：{label} · 初始化 {lang_label}",
                )
            cmd = [
                str(py), "-u", str(doc_worker_script() if pipeline in {"vl", "structure"} else ocr_worker_script()), "--probe", "--lang", lang,
                "--pipeline", pipeline,
                "--ocr-version", str(selection["ocr_version"] or cfg.ocr.ocr_version),
                "--model-profile", str(selection["profile"]),
            ]
            if pipeline_version:
                cmd += ["--pipeline-version", pipeline_version]
            if pipeline == "ocr":
                if det_name: cmd += ["--det-name", det_name]
                if rec_name: cmd += ["--rec-name", rec_name]
                if local_det: cmd += ["--det-dir", str(local_det)]
                if local_rec: cmd += ["--rec-dir", str(local_rec)]
            base = os.environ.copy()
            env = base if source == "local" else paddle_source_environment(source, base)
            proxy = str(os.environ.get("MHD_MODEL_PROXY", "") or "").strip()
            if proxy:
                for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                    env[key] = proxy
            env = apply_runtime_tls_environment(env, doc_runtime_root() if pipeline in {"vl", "structure"} else ocr_runtime_root())
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=5400)
            except Exception as exc:
                source_ok = False
                source_details.append(f"{lang_label}: {type(exc).__name__}: {exc}")
                break
            payload = None
            for line in (proc.stdout or "").splitlines():
                try: row = json.loads(line)
                except Exception: continue
                if isinstance(row, dict) and row.get("type") in {"ready", "probe", "init"}:
                    payload = row
            if proc.returncode != 0 or not payload or not payload.get("ok"):
                detail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
                err = str((payload or {}).get("error", "") or detail[-4500:])
                low_err = err.lower()
                dependency_error = pipeline in {"vl", "structure"} and any(token in low_err for token in (
                    "dependency error", "requires additional dependencies", "paddlex[ocr]",
                    "genai-client", "not available. please install",
                ))
                if dependency_error and not dependency_repair_attempted:
                    dependency_repair_attempted = True
                    runtime_progress(f"{model_label} 检测到文档解析依赖不完整，自动修复后重试当前模型源…")
                    repair_doc_runtime(runtime_progress)
                    # Re-run the same worker probe once. A dependency failure is
                    # independent of ModelScope/BOS/AIStudio/HF, so do not waste
                    # time cycling four hubs before repairing the environment.
                    try:
                        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=5400)
                    except Exception as exc:
                        source_ok = False
                        source_details.append(f"{lang_label}: {type(exc).__name__}: {exc}")
                        break
                    payload = None
                    for line in (proc.stdout or "").splitlines():
                        try: row = json.loads(line)
                        except Exception: continue
                        if isinstance(row, dict) and row.get("type") in {"ready", "probe", "init"}:
                            payload = row
                    if proc.returncode == 0 and payload and payload.get("ok"):
                        final_payload = dict(payload)
                        continue
                    detail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
                    err = str((payload or {}).get("error", "") or detail[-4500:])
                source_ok = False
                source_details.append(f"{lang_label}: {err}")
                break
            final_payload = dict(payload)
        if source_ok:
            chosen_source = source
            break
        failures.append(f"[{label}] " + " | ".join(source_details)[-5000:])

    if not chosen_source:
        raise RuntimeError(
            f"{model_label} 下载/初始化失败。程序已在输出任何 OCR 结果前依次尝试可用模型源。\n"
            + "\n\n".join(failures[-4:])[-12000:]
        )

    marker = model_local_paths()["paddle"]
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
        if not isinstance(existing, dict): existing = {}
    except Exception:
        existing = {}
    ready_profiles = dict(existing.get("ready_profiles") or {})
    ready_profiles[str(selection["profile"])] = {
        "label": model_label,
        "pipeline": pipeline,
        "pipeline_version": pipeline_version or None,
        "det_name": det_name or None,
        "rec_name": rec_name or None,
        "ocr_version": str(selection["ocr_version"] or cfg.ocr.ocr_version),
        "model_source": chosen_source,
        "model_source_label": "本地离线模型" if chosen_source == "local" else PADDLE_MODEL_SOURCE_LABELS.get(chosen_source, chosen_source),
        "worker_profile": final_payload.get("model_profile"),
        "runtime_kind": "document-parser" if pipeline in {"vl", "structure"} else "classic-ocr",
        "runtime_python": str(status.python or ""),
        "runtime_version": str(status.version or ""),
    }
    marker.write_text(json.dumps({
        "schema": "mhd.paddle_engines.v4",
        "active_profile": str(selection["profile"]),
        "active_label": model_label,
        "active_pipeline": pipeline,
        "active_pipeline_version": pipeline_version or None,
        "ready_profiles": ready_profiles,
        "runtime": status.python,
        "runtime_version": status.version,
        "isolated": True,
        "model_source": chosen_source,
        "model_source_label": "本地离线模型" if chosen_source == "local" else PADDLE_MODEL_SOURCE_LABELS.get(chosen_source, chosen_source),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress:
        progress(max(1, len(sources) * len(langs)), max(1, len(sources) * len(langs)), f"{model_label} 已就绪 · {chosen_source}")
    return ModelDownloadResult(
        "paddle", MODEL_LABELS["paddle"], str(marker),
        f"{model_label} 已下载/预热；引擎：{pipeline}{(' '+pipeline_version) if pipeline_version else ''}；模型源：{chosen_source}。",
        {"ocr.paddle_model_profile": str(selection["profile"])},
    )

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
            _hf_urls("huyvux3005/manga109-segmentation-bubble", "best.pt") + [
                "https://www.modelscope.cn/models/hgmzhn/manga-translator-ui/resolve/master/mangalens.pt"
            ],
            paths[key],
            sha256="4028152940f7c910f40192f46ede3b3f6c7129e5c76849c324d3564f8ac50198",
            progress=progress, label=label,
        )
        return ModelDownloadResult(
            key, label, str(path), "MangaLens best.pt 已下载并自动配置。",
            {"bubbles.mangalens_model_path": str(path)},
        )

    if key == "ysg_obb":
        path = _download_file(
            ["https://www.modelscope.cn/models/hgmzhn/manga-translator-ui/resolve/master/ysgyolo_yolo26_2.0.pt"],
            paths[key],
            sha256="889347d65c8636dd188a8ed4f312b29658543faaa69016b5958ddf0559980e22",
            progress=progress, label=label, timeout=180.0,
        )
        return ModelDownloadResult(
            key, label, str(path), "YSG YOLO OBB 已下载并校验；仅作为开放/框外/倾斜文字辅助检测器。",
            {"bubbles.ysg_obb_model_path": str(path)},
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

    if key == "ocr48px":
        from .ocr48px_runtime import prepare_runtime_files
        root = paths[key]
        prepare_runtime_files(root, progress=progress)
        return ModelDownloadResult(
            key, label, str(root),
            "48px AR 官方 checkpoint、字符表与固定版本网络源码已下载并通过哈希校验；可直接使用原生独立运行时。",
            {"ocr.ocr48px_model_path": str(root)},
        )

    if key in {"koharu_layout", "manga_ocr", "baberu_ocr", "aot_inpainting", "rorem_mixed"}:
        from .koharu_model_stack import model_spec
        spec = model_spec(key)
        root = paths[key]
        root.mkdir(parents=True, exist_ok=True)
        files = list(spec.files)
        for i, entry in enumerate(files):
            def cb(done: int, total: int | None, msg: str, *, _i=i, _name=entry.path):
                if progress:
                    progress(done, total, f"{_i + 1}/{len(files)} {_name}")
            urls = (
                _hf_revision_fallback_urls(str(spec.repo), entry.path, str(spec.revision or "main"))
                if key == "koharu_layout"
                else _hf_urls(str(spec.repo), entry.path, str(spec.revision or "main"))
            )
            _download_file(
                urls,
                root / entry.path,
                sha256=entry.sha256,
                progress=cb,
                label=entry.path,
                timeout=120.0,
            )
        updates: dict[str, Any] = {}
        if key == "koharu_layout":
            updates["bubbles.koharu_layout_model_path"] = str(root)
        elif key in {"manga_ocr", "baberu_ocr", "ocr48px"}:
            updates[f"ocr.{key}_model_path"] = str(root)
        elif key == "aot_inpainting":
            updates["inpainting.aot_model_path"] = str(root)
        elif key == "rorem_mixed":
            updates["inpainting.rorem_mixed_model_path"] = str(root)
        return ModelDownloadResult(key, label, str(root), f"{label} 文件已下载并校验。", updates)

    if key == "lama_manga":
        from .koharu_model_stack import model_spec
        spec = model_spec(key); entry = spec.files[0]
        path = _download_file(
            _hf_urls(str(spec.repo), entry.path, str(spec.revision or "main")),
            paths[key], sha256=entry.sha256, progress=progress, label=label, timeout=120.0,
        )
        return ModelDownloadResult(
            key, label, str(path), "LaMa Manga checkpoint 已下载；运行时通过本地/隔离 runner 调用。",
            {"inpainting.lama_model_path": str(path)},
        )

    if key == "flux2_klein":
        raise RuntimeError(
            f"{label} 属于超大/复合生成模型，Folirina 不会在普通‘下载/校验’中静默拉取数 GB～数十 GB。"
            "请使用模型中心‘离线导入’选择完整模型目录；导入后仍按需加载，不影响 Direct / Mask / Reletter。"
        )

    raise AssertionError(key)


def apply_config_updates(config: PipelineConfig, updates: dict[str, Any]) -> None:
    for dotted, value in (updates or {}).items():
        parts = str(dotted).split(".")
        obj: Any = config
        for name in parts[:-1]:
            obj = getattr(obj, name)
        setattr(obj, parts[-1], value)


def paddle_profile_marker_status(profile: str | None) -> tuple[bool, tuple[str, ...]]:
    """Return whether the selected Paddle profile has been preheated.

    The PaddleX cache may contain several model pairs at once. The legacy marker
    only proved that *some* PP-OCRv5 model was warmed, so it maps to the legacy
    compatibility profile instead of claiming every new profile is ready.
    """
    from .paddle_profiles import normalize_paddle_model_profile
    wanted = normalize_paddle_model_profile(profile)
    marker = model_local_paths()["paddle"]
    if not marker.is_file():
        return False, ()
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False, ()
    if not isinstance(payload, dict):
        return False, ()
    ready = dict(payload.get("ready_profiles") or {})
    if ready:
        keys = tuple(sorted(str(x) for x in ready))
        return wanted in ready, keys
    # v2.0.42 and older markers represented only the old PP-OCRv5 auto route.
    legacy_ready = bool(payload.get("ocr_version") == "PP-OCRv5" or payload.get("langs"))
    return bool(legacy_ready and wanted == "legacy_v5_auto"), (("legacy_v5_auto",) if legacy_ready else ())


def discovered_paddle_model_dirs() -> tuple[Path | None, Path | None]:
    """Return explicitly imported PaddleOCR det/rec directories across restarts."""
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
    directory_requirements = {
        "rtdetr_v2": ("config.json", "preprocessor_config.json", "model.safetensors"),
        "koharu_layout": ("inference_config.json", "load_model.py", "model.safetensors"),
        "manga_ocr": ("config.json", "preprocessor_config.json", "model.safetensors"),
        "baberu_ocr": ("onnx_infer.py", "onnx/vision_int4.onnx", "onnx/decoder_prefill_int8.onnx", "onnx/decoder_step_int8.onnx", "tokenizer/vocab.json"),
        "ocr48px": ("ocr_ar_48px.ckpt", "alphabet-all-v7.txt", "upstream-source/manga_48px_core.py", "upstream-source/manga_48px_xpos.py"),
        "aot_inpainting": ("config.json", "model.safetensors"),
        "rorem_mixed": ("rorem-mixed-unet-q4_K.gguf", "sdxl-version-marker.safetensors"),
    }
    if configured:
        p = Path(configured).expanduser()
        if key in directory_requirements:
            if p.is_dir() and all((p / n).is_file() for n in directory_requirements[key]):
                return p
        elif p.is_file():
            return p
    local = model_local_paths().get(str(key))
    if local is None:
        return None
    if key in directory_requirements:
        if local.is_dir() and all((local / n).is_file() for n in directory_requirements[key]):
            return local
        return None
    return local if local.is_file() else None
