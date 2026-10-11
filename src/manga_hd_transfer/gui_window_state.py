from __future__ import annotations

"""Remember window size/position between launches (all failures are ignored)."""

from PySide6.QtGui import QGuiApplication

from .gui_interaction_logic import geometry_fits


def restore_window_geometry(window, settings) -> bool:
    try:
        data = settings.value("window/geometry")
        if not data or not window.restoreGeometry(data):
            return False
        screen = window.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            if not geometry_fits(window.width(), window.height(), avail.width(), avail.height()):
                window.resize(min(1480, avail.width()), min(960, avail.height()))
        return True
    except Exception:
        return False


def save_window_geometry(window, settings) -> None:
    try:
        if not window.isMaximized() and not window.isFullScreen():
            settings.setValue("window/geometry", window.saveGeometry())
    except Exception:
        pass


__all__ = ["restore_window_geometry", "save_window_geometry"]
