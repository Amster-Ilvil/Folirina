from __future__ import annotations

"""Application-wide dialog polish: fit to screen + safe default button, without touching each dialog."""

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QTextEdit

from .gui_dialog_logic import clamp_dialog_min, pick_default_button


def polish_dialog(dialog: QDialog) -> None:
    if dialog.property("nfPolished") or isinstance(dialog, QMessageBox):
        return
    dialog.setProperty("nfPolished", True)
    try:
        screen = dialog.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            new_w, new_h = clamp_dialog_min(dialog.minimumWidth(), dialog.minimumHeight(), avail.width(), avail.height())
            if (new_w, new_h) != (dialog.minimumWidth(), dialog.minimumHeight()):
                dialog.setMinimumSize(new_w, new_h)
            if dialog.width() > avail.width() or dialog.height() > avail.height():
                dialog.resize(min(dialog.width(), int(avail.width() * 0.96)), min(dialog.height(), int(avail.height() * 0.96)))
        buttons = [b for b in dialog.findChildren(QPushButton) if b.isVisibleTo(dialog) and b.isEnabled()]
        if any(b.isDefault() for b in buttons):
            return
        has_input = bool(dialog.findChildren(QLineEdit) or dialog.findChildren(QPlainTextEdit) or dialog.findChildren(QTextEdit))
        index = pick_default_button([b.text() for b in buttons], has_input)
        if index is not None:
            buttons[index].setDefault(True)
    except Exception:
        pass            # 细节美化永远不能影响对话框本身


class _DialogPolishFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Show and isinstance(obj, QDialog):
            polish_dialog(obj)
        return False


def install_dialog_polish(app) -> None:
    flt = _DialogPolishFilter(app)
    app.installEventFilter(flt)
    app._nf_dialog_polish = flt


__all__ = ["install_dialog_polish", "polish_dialog"]
