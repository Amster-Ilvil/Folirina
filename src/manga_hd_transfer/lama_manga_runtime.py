from __future__ import annotations

"""Lazy local runner wiring for the optional LaMa Manga checkpoint."""

from dataclasses import dataclass
from pathlib import Path
import shlex
import sys
import tempfile

import numpy as np

from .external_command import run_external_command
from .io_utils import read_image, write_image
from .storage_paths import model_home


@dataclass(frozen=True, slots=True)
class LamaMangaRunnerStatus:
    ready: bool
    model_path: Path
    python_path: Path
    runner_path: Path
    detail: str


def runner_path() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        root = Path(frozen_root)
        candidates = (
            root / "runners" / "lama_manga_runner.py",
            root.parent / "Resources" / "runners" / "lama_manga_runner.py",
            root / ".." / "Resources" / "runners" / "lama_manga_runner.py",
        )
        for bundled in candidates:
            if bundled.is_file():
                return bundled.resolve()
    return Path(__file__).resolve().parents[2] / "scripts" / "lama_manga_runner.py"


def runtime_python_path() -> Path:
    # Use the shared isolated Torch environment; never import Torch in the GUI
    # interpreter (which may be Python 3.14 or a deliberately light venv).
    from .deep_registration_runtime import venv_python
    return venv_python()


def status(model_path: str | Path | None = None) -> LamaMangaRunnerStatus:
    weights = Path(model_path).expanduser() if model_path else model_home() / "inpainting" / "lama-manga.safetensors"
    python = runtime_python_path()
    runner = runner_path()
    missing = []
    if not weights.is_file():
        missing.append(f"weight missing: {weights}")
    if not python.is_file():
        missing.append(f"isolated Torch Python missing: {python}")
    if not runner.is_file():
        missing.append(f"runner missing: {runner}")
    return LamaMangaRunnerStatus(
        ready=not missing,
        model_path=weights,
        python_path=python,
        runner_path=runner,
        detail="; ".join(missing) if missing else "LaMa Manga weight and isolated runner are ready.",
    )


def command_template(model_path: str | Path | None = None) -> str | None:
    current = status(model_path)
    if not current.ready:
        return None
    return " ".join((
        shlex.quote(str(current.python_path)),
        shlex.quote(str(current.runner_path)),
        "--input {input} --mask {mask} --output {output} --weights {model} --device auto",
    ))


def inpaint_koharu_text(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    model_path: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Run the configured LaMa Manga sidecar for one protected text mask."""
    current = status(model_path)
    if not current.ready:
        raise RuntimeError(current.detail)
    command = command_template() if model_path is None else " ".join((
        shlex.quote(str(current.python_path)),
        shlex.quote(str(current.runner_path)),
        "--input {input} --mask {mask} --output {output} --weights {model} --device auto",
    ))
    if not command:
        raise RuntimeError("LaMa Manga runner is unavailable")
    with tempfile.TemporaryDirectory(prefix="folirina-lama-manga-") as temp_dir:
        root = Path(temp_dir)
        input_path = root / "input.png"
        mask_path = root / "mask.png"
        output_path = root / "output.png"
        write_image(input_path, image)
        write_image(mask_path, (mask > 0).astype(np.uint8) * 255)
        proc = run_external_command(
            command,
            {"input": input_path, "mask": mask_path, "output": output_path, "model": current.model_path},
            timeout=600,
            allow_shell=False,
        )
        if proc.returncode != 0 or not output_path.is_file():
            raise RuntimeError(f"LaMa Manga runner failed ({proc.returncode}): {proc.stderr[-1600:]}")
        result = read_image(output_path)
        if result.shape != image.shape:
            raise RuntimeError(f"LaMa Manga output shape mismatch: {result.shape} vs {image.shape}")
        return result, {
            "stdout": proc.stdout[-1000:],
            "command_exit": proc.returncode,
            "shell": proc.shell,
            "model_path": str(current.model_path),
        }


__all__ = ["LamaMangaRunnerStatus", "command_template", "inpaint_koharu_text", "runner_path", "runtime_python_path", "status"]
