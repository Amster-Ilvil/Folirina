from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from .config import PipelineConfig
from .io_utils import save_json
from .lettering import find_default_font
from .pairing import pair_directories
from .pipeline import TransferPipeline
from .runtime import runtime_summary
from .review import serve_review
from .mode_contracts import SUPPORTED_MODES, SUPPORTED_MODE_ORDER
from .review_apply import apply_review_page
from .app_logging import configure_application_logging, install_exception_hooks

app = typer.Typer(add_completion=False, help="旧版低清中文汉化 → 高清日文原图：配准、清字、重新嵌字与出版 QA。")


def _load_config(path: Optional[Path]) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PipelineConfig.model_validate(payload)


def _setup_logging(verbose: bool) -> None:
    configure_application_logging(
        component="cli",
        level=logging.DEBUG if verbose else logging.INFO,
        console=True,
    )
    install_exception_hooks()


@app.command("init-config")
def init_config(output: Path = typer.Argument(Path("config.json"))):
    """Write a complete editable configuration file."""
    save_json(output, PipelineConfig().model_dump())
    typer.echo(f"Wrote {output}")


@app.command("pair")
def pair_cmd(
    source_dir: Path,
    target_dir: Path,
    config: Optional[Path] = typer.Option(None, "--config"),
):
    """Only pair old Chinese pages with HD Japanese pages; no OCR or editing."""
    cfg = _load_config(config)
    pairs, us, ut = pair_directories(source_dir, target_dir, cfg.pairing)
    typer.echo(json.dumps({"pairs": [p.__dict__ if hasattr(p, "__dict__") else {
        "source_path": p.source_path, "target_path": p.target_path, "confidence": p.confidence, "score": p.score,
        "source_index": p.source_index, "target_index": p.target_index, "reasons": p.reasons,
    } for p in pairs], "unmatched_source": us, "unmatched_target": ut}, ensure_ascii=False, indent=2))


@app.command("run")
def run_cmd(
    source_dir: Path = typer.Argument(..., help="旧版中文汉化图片目录"),
    target_dir: Path = typer.Argument(..., help="高清日文原图目录"),
    output_dir: Path = typer.Argument(..., help="输出目录"),
    config: Optional[Path] = typer.Option(None, "--config"),
    ocr_backend: Optional[str] = typer.Option(None, "--ocr-backend", help="paddle|sidecar|none"),
    registration_backend: Optional[str] = typer.Option(None, "--registration-backend", help="auto|opencv|lightglue|loftr"),
    mode: Optional[str] = typer.Option(None, "--mode", help="direct_patch|mask_replace|aligned_overlay_reveal|hybrid|reletter (legacy: auto|transparent_bubble_reveal)"),
    font: Optional[Path] = typer.Option(None, "--font", help="中文字体文件路径"),
    device: str = typer.Option("auto", "--device", help="auto|mps|cuda|cpu"),
    no_resume: bool = typer.Option(False, "--no-resume", help="禁用断点续跑"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the complete transfer engine on a book/folder pair."""
    _setup_logging(verbose)
    cfg = _load_config(config)
    if ocr_backend:
        cfg.ocr.backend = ocr_backend
    if registration_backend:
        cfg.registration.backend = registration_backend
    if mode:
        if mode not in SUPPORTED_MODES:
            raise typer.BadParameter("--mode must be " + "|".join(SUPPORTED_MODE_ORDER))
        cfg.transfer.mode = mode
    if font:
        cfg.lettering.font_path = str(font)
    if device not in {"auto","mps","cuda","cpu"}:
        raise typer.BadParameter("--device must be auto|mps|cuda|cpu")
    cfg.runtime.device=device; cfg.registration.device=device; cfg.bubbles.device=device; cfg.mask_replace.sr_device=device; cfg.direct_patch.sr_device=device
    if no_resume: cfg.batch.resume=False; cfg.batch.skip_completed=False
    pipe = TransferPipeline(cfg)
    try:
        def progress(done,total,pair,status,cache_hit,message):
            name=Path(pair.target_path).name if pair else ""
            typer.echo(f"[{done}/{total}] {name} · {'CACHE' if cache_hit else status} · {message}")
        book = pipe.run_book(source_dir, target_dir, output_dir, progress_cb=progress, resume=not no_resume)
    except RuntimeError as e:
        if "Paddle" in str(e):
            typer.echo("Paddle 运行环境未就绪。PP-OCRv6 与 VL/PP-StructureV3 使用彼此独立的 venv；可在 GUI 模型中心自动修复，或用 doctor 查看两套运行时状态；也可改用外部 OCR JSON / MD。", err=True)
        raise
    typer.echo(json.dumps(book.meta, ensure_ascii=False, indent=2))
    if book.meta.get("qa_errors", 0):
        typer.echo(f"完成，但有 {book.meta['qa_errors']} 个出版阻断级 QA 项；请运行 review。", err=True)
        raise typer.Exit(code=2)


@app.command("cleanup-workspace")
def cleanup_workspace_cmd(
    output_dir: Path = typer.Argument(..., help="已有输出目录（包含 pages/）"),
):
    """Remove reproducible page diagnostics while keeping GUI/manual restore files."""
    from .workspace_cleanup import cleanup_output_workspace
    stats = cleanup_output_workspace(output_dir)
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))



@app.command("review")
def review_cmd(
    output_dir: Path,
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    no_browser: bool = typer.Option(False, "--no-browser"),
    allow_remote: bool = typer.Option(False, "--allow-remote", help="显式允许绑定非 localhost；会启用一次性 URL token"),
    config: Optional[Path] = typer.Option(None, "--config"),
):
    """Open the review editor. Localhost is the safe default; remote bind is opt-in."""
    serve_review(
        output_dir, host=host, port=port, open_browser=not no_browser,
        config=_load_config(config), allow_remote=allow_remote,
    )


@app.command("apply-review")
def apply_review_cmd(
    page_dir: Path,
    config: Optional[Path] = typer.Option(None, "--config"),
):
    """Apply a page's review_overrides.json/manual_clear_mask.png and regenerate layers."""
    out = apply_review_page(page_dir, _load_config(config))
    typer.echo(str(out))


@app.command("doctor")
def doctor_cmd():
    """Report optional backend readiness without downloading models."""
    import cv2
    try:
        from .paddle_runtime import runtime_status as _paddle_runtime_status
        _paddle = _paddle_runtime_status()
    except Exception as _paddle_exc:
        _paddle = None
    try:
        from .paddle_doc_runtime import runtime_status as _paddle_doc_runtime_status
        _paddle_doc = _paddle_doc_runtime_status()
    except Exception as _paddle_doc_exc:
        _paddle_doc = None
    status = {
        "opencv": cv2.__version__,
        "sift": hasattr(cv2, "SIFT_create"),
        "paddleocr_installed": bool(_paddle and _paddle.ready),
        "paddleocr_runtime": (_paddle.detail if _paddle else f"ERROR: {_paddle_exc}"),
        "paddle_doc_parser_installed": bool(_paddle_doc and _paddle_doc.ready),
        "paddle_doc_parser_runtime": (_paddle_doc.detail if _paddle_doc else f"ERROR: {_paddle_doc_exc}"),
        "lightglue_installed": importlib.util.find_spec("lightglue") is not None,
        "kornia_installed": importlib.util.find_spec("kornia") is not None,
        "imagemagick": __import__("shutil").which("magick") is not None,
        "spandrel_installed": importlib.util.find_spec("spandrel") is not None,
        "ultralytics_installed": importlib.util.find_spec("ultralytics") is not None,
        "runtime": runtime_summary("auto"),
    }
    try:
        status["font"] = find_default_font()
    except Exception as e:
        status["font"] = f"ERROR: {e}"
    typer.echo(json.dumps(status, ensure_ascii=False, indent=2))


@app.command("architecture-audit")
def architecture_audit_cmd():
    """Check transfer-mode capability, artifact ownership and review isolation contracts."""
    from .architecture_audit import run_architecture_audit
    report = run_architecture_audit()
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise typer.Exit(code=1)


@app.command("selftest")
def selftest_cmd():
    """Run the built-in offline synthetic publication-pipeline acceptance check."""
    from .selftest import run_selftest
    report = run_selftest()
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
