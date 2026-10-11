from __future__ import annotations

"""Non-blocking toast for hints and success messages (guard prompts no longer need a modal)."""

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QMessageBox

from .gui_interaction_logic import toast_duration_ms, toast_origin
from .gui_theme import semantic_palette


def _host_window(widget):
    """Studio pages store the StudioWindow in ``self.window`` (shadowing QWidget.window())."""
    if widget is None:
        return None
    attr = getattr(widget, "window", None)
    if attr is not None and not callable(attr):
        return attr
    return widget.window()


class _Toast(QFrame):
    def __init__(self, host):
        super().__init__(host)
        self.setObjectName("nfToast")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        row = QHBoxLayout(self); row.setContentsMargins(14, 10, 18, 10); row.setSpacing(10)
        self._icon = QLabel(); self._icon.setFixedSize(22, 22); self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._text = QLabel(); self._text.setObjectName("nfToastText"); self._text.setWordWrap(True)
        self._text.setMaximumWidth(620)
        row.addWidget(self._icon); row.addWidget(self._text, 1)
        shadow = QGraphicsDropShadowEffect(self); shadow.setBlurRadius(28); shadow.setOffset(0, 8)
        shadow.setColor(QColor(15, 23, 42, 70)); self.setGraphicsEffect(shadow)
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self.hide)
        host.installEventFilter(self)
        self.hide()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Resize and self.isVisible():
            self._place()
        return False

    def _place(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            self.adjustSize()
            self.move(*toast_origin(parent.width(), parent.height(), self.width(), self.height()))

    def show_message(self, text: str, kind: str, ms: int) -> None:
        theme = getattr(self.parentWidget(), "theme_name", "light")
        pal = semantic_palette(theme)
        glyph, color = {"success": ("✓", pal["green"]), "warning": ("!", pal["orange"])}.get(kind, ("i", pal["accent"]))
        self._icon.setText(glyph)
        self._icon.setStyleSheet(f"background:{color};color:#FFFFFF;border-radius:11px;font-size:12px;font-weight:800;")
        self._text.setText(text)
        self._place(); self.show(); self.raise_()
        self._timer.start(ms)

    def mousePressEvent(self, event) -> None:
        self.hide()


def notify(widget, text: str, kind: str = "success", ms: int | None = None) -> None:
    """Show an auto-hiding toast; fall back to a message box when no visible host exists."""
    host = _host_window(widget)
    if host is None or not host.isVisible():
        QMessageBox.information(widget, "提示", str(text))
        return
    toast = getattr(host, "_nf_toast", None)
    if toast is None:
        toast = _Toast(host)
        host._nf_toast = toast
    toast.show_message(str(text), kind, int(ms or toast_duration_ms(str(text), kind)))


__all__ = ["notify"]
