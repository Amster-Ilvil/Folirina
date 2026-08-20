from __future__ import annotations

"""Stable user-data roots shared by model and isolated-runtime managers.

This module deliberately has no dependency on model downloaders or runtime
installers so storage-location discovery cannot participate in import cycles.
"""

import os
import platform
from pathlib import Path


def model_home() -> Path:
    override = str(os.environ.get("FOLIRINA_MODEL_HOME", "") or os.environ.get("MHD_MODEL_HOME", "") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    home = Path.home()
    system = platform.system()
    if system == "Darwin":
        preferred = home / "Library" / "Application Support" / "Folirina" / "models"
        legacy = home / "Library" / "Application Support" / "Manga HD Transfer Studio" / "models"
    elif system == "Windows":
        local = Path(os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local")))
        preferred = local / "Folirina" / "models"
        legacy = local / "MangaHDTransfer" / "models"
    else:
        preferred = home / ".local" / "share" / "folirina" / "models"
        legacy = home / ".local" / "share" / "manga-hd-transfer" / "models"
    if not preferred.exists() and legacy.exists():
        return legacy
    return preferred


def runtime_home() -> Path:
    return model_home().parent / "runtime"


__all__ = ["model_home", "runtime_home"]
