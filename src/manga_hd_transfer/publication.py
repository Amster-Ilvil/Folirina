from __future__ import annotations

"""Final delivery: optional Upscayl then real Caesium CLI compression.

The editable/native page stays lossless and at TARGET coordinates. Only the
book-level delivery changes format/size. Neither backend downloads during a run.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import cv2
import numpy as np
from PIL import Image

from .config import DeliveryExportConfig
from .io_utils import load_json, save_json
from .publication_runtime import find_caesium, find_upscayl
from .cancellation import PipelineCancelled

_EXPORT_LOCK = threading.RLock()


def delivery_settings(value=None) -> DeliveryExportConfig:
    if isinstance(value, DeliveryExportConfig):
        return value
    return DeliveryExportConfig.model_validate(value or {})


def delivery_enabled(value=None) -> bool:
    cfg = delivery_settings(value)
    return cfg.compression_enabled or cfg.upscale_enabled


def delivery_path(requested: str | Path, value=None) -> Path:
    cfg = delivery_settings(value)
    extension = {"webp": ".webp", "jpeg": ".jpg", "png": ".png"}[cfg.format] if cfg.compression_enabled else ".png"
    return Path(requested).with_suffix(extension)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict:
    info = path.stat()
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def validate_delivery_runtime(value=None) -> dict:
    cfg = delivery_settings(value)
    runtime = {}
    if cfg.compression_enabled:
        binary = find_caesium(cfg.caesium_path)
        if binary is None:
            raise RuntimeError("未找到 Caesium CLT。请运行 scripts/install_output_tools.command，或在输出设置指定 caesiumclt。")
        runtime["caesium"] = _identity(binary)
    if cfg.upscale_enabled:
        binary, models = find_upscayl(cfg.upscayl_path, cfg.upscayl_models)
        if binary is None or models is None:
            raise RuntimeError("未找到 Upscayl 及其模型目录；请在输出设置指定程序和模型目录。")
        runtime["upscayl"] = _identity(binary)
        runtime["models"] = str(models)
        for extension in (".param", ".bin"):
            path = models / (cfg.upscale_model + extension)
            if not path.is_file():
                raise FileNotFoundError(f"Upscayl 模型不存在：{path}")
            runtime["model" + extension] = _identity(path)
    return runtime


def _check_cancel(cancel_cb) -> None:
    if cancel_cb is not None and cancel_cb():
        raise PipelineCancelled("publication_cancelled")


def _run(argv: list[str], cfg: DeliveryExportConfig, log_path: Path, cancel_cb=None) -> str:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(cfg.threads)
    deadline = time.monotonic() + cfg.timeout_seconds
    with log_path.open("w+b") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=log, env=env, shell=False)
        try:
            while process.poll() is None:
                _check_cancel(cancel_cb)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"导出工具运行超时（{cfg.timeout_seconds} 秒）：{Path(argv[0]).name}")
                try:
                    process.wait(timeout=0.15)
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        log.seek(max(0, log.seek(0, os.SEEK_END) - 12000))
        detail = log.read().decode("utf-8", errors="replace")
    _check_cancel(cancel_cb)
    if process.returncode:
        raise RuntimeError(f"{Path(argv[0]).name} 导出失败（{process.returncode}）：\n{detail[-3000:]}")
    return detail


def _compress(source: Path, output_dir: Path, cfg, runtime: dict, cancel_cb=None) -> tuple[Path, str]:
    fmt = cfg.format
    same_format = source.suffix.lower() in {"jpeg": {".jpg", ".jpeg"}, "png": {".png"}, "webp": {".webp"}}[fmt]
    argv = [runtime["caesium"]["path"], "--threads", str(cfg.threads), "-O", "all",
            "--format", "original" if same_format else fmt, "--json", "-o", str(output_dir)]
    if fmt == "png":
        argv += ["--lossless", "--png-opt-level", str(cfg.png_opt_level)]
    elif fmt == "webp" and cfg.lossless:
        argv += ["--lossless"]
    else:
        argv += ["-q", str(cfg.quality)]
        if fmt == "jpeg":
            argv += ["--jpeg-chroma-subsampling", "4:4:4"]
    argv += [str(source)]
    output_dir.mkdir()
    log = _run(argv, cfg, output_dir.parent / "caesium.log", cancel_cb)
    expected_suffix = {"webp": ".webp", "jpeg": ".jpg", "png": ".png"}[fmt]
    matches = [p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in ({".jpg", ".jpeg"} if fmt == "jpeg" else {expected_suffix})]
    if len(matches) != 1:
        raise RuntimeError("Caesium 未生成唯一有效结果。\n" + log[-1500:])
    return matches[0], log


def _upscale(source: Path, output: Path, cfg, runtime: dict, cancel_cb=None) -> str:
    argv = [runtime["upscayl"]["path"], "-i", str(source), "-o", str(output),
            "-m", runtime["models"], "-n", cfg.upscale_model, "-z", "4",
            "-s", str(cfg.upscale_scale), "-t", str(cfg.tile_size),
            "-j", "1:1:1", "-g", "0", "-f", "png"]
    from .runtime import accelerator_lock
    with accelerator_lock():
        return _run(argv, cfg, output.parent / "upscayl.log", cancel_cb)


def _obsolete_paths(requested: Path, dest: Path, previous: dict | None, source: Path) -> list[Path]:
    candidates = [requested]
    if previous and previous.get("output"):
        candidates.append(Path(previous["output"]))
    return list(dict.fromkeys(
        p for p in candidates
        if p.parent.resolve() == requested.parent.resolve() and p.stem == requested.stem
        and p.suffix.lower() in {".png", ".webp", ".jpg", ".jpeg"}
        and p.resolve() not in {source.resolve(), dest.resolve()}
    ))


def publish_delivery(source: str | Path, requested: str | Path, value=None, *, previous=None, cancel_cb=None) -> dict:
    """Validate a staged result, then atomically publish it; old files survive failures."""
    cfg = delivery_settings(value)
    source, requested = Path(source), Path(requested)
    if source.resolve() == requested.resolve():
        raise ValueError("压缩/超分导出目录必须与无损页面工作区分开。")
    dest = delivery_path(requested, cfg)
    if source.resolve() == dest.resolve():
        raise ValueError("不能将最终导出覆盖到编辑底稿。")
    with _EXPORT_LOCK:
        _check_cancel(cancel_cb)
        started = time.perf_counter()
        runtime = validate_delivery_runtime(cfg)
        settings = cfg.model_dump(mode="json")
        source_sha = _sha(source)
        with Image.open(source) as image:
            native_size = image.size
        scale = cfg.upscale_scale if cfg.upscale_enabled else 1
        expected_size = (native_size[0] * scale, native_size[1] * scale)
        if expected_size[0] * expected_size[1] > cfg.max_output_megapixels * 1_000_000:
            raise ValueError(f"导出超过 {cfg.max_output_megapixels} MP 限制，请降低超分倍率。")
        fingerprint = hashlib.sha256(json.dumps({"source": source_sha, "settings": settings, "runtime": runtime}, sort_keys=True).encode()).hexdigest()
        if previous and previous.get("fingerprint") == fingerprint and previous.get("output") == str(dest):
            if dest.is_file() and _sha(dest) == previous.get("output_sha256"):
                return dict(previous)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{dest.stem}-delivery-", dir=dest.parent) as td:
            staging = Path(td)
            image_source = source
            upscale_log = ""
            if cfg.upscale_enabled:
                image_source = staging / "upscaled.png"
                upscale_log = _upscale(source, image_source, cfg, runtime, cancel_cb)
            if cfg.compression_enabled:
                encoded, compression_log = _compress(image_source, staging / "compressed", cfg, runtime, cancel_cb)
            else:
                encoded = staging / "final.png"
                shutil.copyfile(image_source, encoded)
                compression_log = ""
            with Image.open(encoded) as image:
                image.load()
                actual_size = image.size
                actual_format = image.format
            expected_format = {"webp": "WEBP", "jpeg": "JPEG", "png": "PNG"}[cfg.format] if cfg.compression_enabled else "PNG"
            if actual_size != expected_size or actual_format != expected_format:
                raise RuntimeError(f"导出格式或尺寸错误：{actual_format} {actual_size}；预期 {expected_format} {expected_size}")
            lossless = cfg.compression_enabled and (cfg.format == "png" or (cfg.format == "webp" and cfg.lossless))
            if lossless:
                # Opaque final pages may be losslessly re-encoded as grayscale
                # or indexed PNG. Compare normalized BGR pixels, not channels.
                baseline = cv2.imread(str(image_source), cv2.IMREAD_COLOR)
                result = cv2.imread(str(encoded), cv2.IMREAD_COLOR)
                if baseline is None or result is None or not np.array_equal(baseline, result):
                    raise RuntimeError("无损压缩像素一致性校验失败，已保留旧结果。")
                if cfg.format == "png" and image_source.stat().st_size < encoded.stat().st_size:
                    shutil.copyfile(image_source, encoded)
            _check_cancel(cancel_cb)
            receipt = {
                "schema": "folirina.delivery.v1", "settings": settings, "runtime": runtime,
                "fingerprint": fingerprint, "source": str(source), "source_sha256": source_sha,
                "source_bytes": source.stat().st_size, "native_size": list(native_size),
                "requested": str(requested),
                "output": str(dest), "output_sha256": _sha(encoded), "output_bytes": encoded.stat().st_size,
                "output_size": list(actual_size), "format": actual_format,
                "compression_engine": "caesiumclt" if cfg.compression_enabled else "none",
                "upscale_engine": "upscayl-ncnn" if cfg.upscale_enabled else "none",
                "lossless_pixel_verified": bool(lossless),
                "upscayl_log": upscale_log[-6000:], "caesium_log": compression_log[-2000:],
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            }
            with encoded.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(encoded, dest)
        # Only retire known book mirrors of this page after successful publication.
        for old in _obsolete_paths(requested, dest, previous, source):
            old.unlink(missing_ok=True)
        return receipt


def publish_project_delivery(value, page_dir, requested, artifacts: dict, meta: dict, *, previous=None, cancel_cb=None) -> Path:
    cfg = delivery_settings(value)
    source = Path(page_dir) / "final.png"
    previous = previous or meta.get("delivery_export")
    receipt = publish_delivery(source, requested, cfg, previous=previous, cancel_cb=cancel_cb)
    artifacts["book_final"] = receipt["output"]
    if delivery_enabled(cfg):
        artifacts["delivery_final"] = receipt["output"]
        meta["delivery_export"] = receipt
    else:
        artifacts.pop("delivery_final", None)
        meta.pop("delivery_export", None)
    return Path(receipt["output"])


def snapshot_delivery_outputs(snapshot: Path, requested: Path, value, previous=None) -> None:
    """Extend the page transaction to its known external publication files."""
    dest = delivery_path(requested, value)
    paths = [requested, dest, *_obsolete_paths(requested, dest, previous, snapshot.parent / "unused")]
    rows = []
    backup = snapshot / "delivery_outputs"
    backup.mkdir()
    for index, path in enumerate(dict.fromkeys(paths)):
        exists = path.is_file()
        name = str(index)
        if exists:
            try:
                os.link(path, backup / name)
            except OSError:
                shutil.copy2(path, backup / name)
        rows.append({"path": str(path), "existed": exists, "backup": name})
    save_json(backup / "manifest.json", rows)


def _atomic_restore_file(source: Path, target: Path) -> None:
    """Restore a delivery file without depending on review-state orchestration."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp-restore", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def restore_delivery_outputs(snapshot: Path | None) -> None:
    if snapshot is None:
        return
    manifest = snapshot / "delivery_outputs" / "manifest.json"
    if not manifest.exists():
        return
    for row in load_json(manifest):
        path = Path(row["path"])
        if row["existed"]:
            _atomic_restore_file(manifest.parent / row["backup"], path)
        else:
            path.unlink(missing_ok=True)
