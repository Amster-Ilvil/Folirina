from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Optional

import typer

from .config import PipelineConfig
from .io_utils import save_json
from .lettering import find_default_font
from .pairing import pair_directories
from .pipeline import TransferPipeline
from .review import serve_review
from .review_apply import apply_review_page

app = typer.Typer(add_completion=False, help="旧版低清中文汉化 → 高清日文原图：配准、清字、重新嵌字与出版 QA。")


def _load_config(path: Optional[Path]) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PipelineConfig.model_validate(payload)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s")


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
    font: Optional[Path] = typer.Option(None, "--font", help="中文字体文件路径"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the complete transfer engine on a book/folder pair."""
    _setup_logging(verbose)
    cfg = _load_config(config)
    if ocr_backend:
        cfg.ocr.backend = ocr_backend
    if registration_backend:
        cfg.registration.backend = registration_backend
    if font:
        cfg.lettering.font_path = str(font)
    pipe = TransferPipeline(cfg)
    try:
        book = pipe.run_book(source_dir, target_dir, output_dir)
    except RuntimeError as e:
        if "PaddleOCR" in str(e):
            typer.echo("PaddleOCR 未安装。可安装 `pip install -e '.[ocr]'`，或使用 --ocr-backend sidecar。", err=True)
        raise
    typer.echo(json.dumps(book.meta, ensure_ascii=False, indent=2))
    if book.meta.get("qa_errors", 0):
        typer.echo(f"完成，但有 {book.meta['qa_errors']} 个出版阻断级 QA 项；请运行 review。", err=True)
        raise typer.Exit(code=2)


@app.command("review")
def review_cmd(
    output_dir: Path,
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    no_browser: bool = typer.Option(False, "--no-browser"),
    config: Optional[Path] = typer.Option(None, "--config"),
):
    """Open the local three-pane review editor with text/match/mask editing."""
    serve_review(output_dir, host=host, port=port, open_browser=not no_browser, config=_load_config(config))


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
    status = {
        "opencv": cv2.__version__,
        "sift": hasattr(cv2, "SIFT_create"),
        "paddleocr_installed": importlib.util.find_spec("paddleocr") is not None,
        "lightglue_installed": importlib.util.find_spec("lightglue") is not None,
        "kornia_installed": importlib.util.find_spec("kornia") is not None,
        "imagemagick": __import__("shutil").which("magick") is not None,
    }
    try:
        status["font"] = find_default_font()
    except Exception as e:
        status["font"] = f"ERROR: {e}"
    typer.echo(json.dumps(status, ensure_ascii=False, indent=2))


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
