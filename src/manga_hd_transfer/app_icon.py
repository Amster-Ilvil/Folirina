from __future__ import annotations

"""Bundled application icon helpers.

Keeps the Studio window/app icon consistent across source runs and packaged
builds. Qt can load PNG on macOS/Linux and ICO on Windows; Windows also gets an
explicit AppUserModelID so the custom icon is used in the taskbar.
"""

from importlib import resources
from pathlib import Path
import sys

APP_USER_MODEL_ID = "AmsterIlvil.Folirina"


def icon_path() -> Path:
    package = resources.files("manga_hd_transfer.assets")
    filename = "folirina_app_icon.ico" if sys.platform.startswith("win") else "folirina_app_icon.png"
    return Path(str(package.joinpath(filename)))


def apply_application_icon(app=None, window=None) -> str | None:
    try:
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication
    except Exception:
        return None

    path = icon_path()
    icon = QIcon(str(path))
    qapp = app or QApplication.instance()
    if qapp is not None:
        try:
            qapp.setWindowIcon(icon)
        except Exception:
            pass
    if window is not None:
        try:
            window.setWindowIcon(icon)
        except Exception:
            pass
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
        except Exception:
            pass
    return str(path)
