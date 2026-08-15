#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Privacy-safe application bootstrap for release packages.

Only the main GUI/runtime dependencies are installed here. OCR/ML dependencies
and model weights stay deferred to the existing in-app installation flow and
are never downloaded automatically during application startup.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    print("[bootstrap] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def install_main_dependencies() -> None:
    env = os.environ.copy()
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PIP_NO_INPUT", "1")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip>=24", "setuptools>=75", "wheel"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".[gui]"],
        cwd=ROOT,
        env=env,
        check=True,
    )


def launch_gui() -> int:
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    process = subprocess.Popen([sys.executable, str(ROOT / "run_gui.py")], cwd=ROOT, env=env)
    try:
        return int(process.wait())
    except KeyboardInterrupt:
        process.terminate()
        try:
            return int(process.wait(timeout=10))
        except subprocess.TimeoutExpired:
            process.kill()
            return int(process.wait(timeout=10))


def main() -> int:
    parser = argparse.ArgumentParser(description="Manga HD Transfer Studio release bootstrap")
    parser.add_argument("--install-main-deps", action="store_true", help="install only the core GUI/runtime dependencies")
    parser.add_argument("--launch", action="store_true", help="launch the GUI after preparation")
    args = parser.parse_args()

    if args.install_main_deps:
        install_main_dependencies()
    if args.launch:
        return launch_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
