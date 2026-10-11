from __future__ import annotations

"""Low-latency zoom support shared by Folirina graphics editors.

The editor used to apply a fixed ~15% transform for *every* wheel event while
``SmoothPixmapTransform`` stayed enabled.  High-resolution trackpads can emit
many small wheel events per frame, so that policy caused repeated full-viewport
smooth resampling and visible input lag.

This controller keeps the scene/image data untouched.  It only manages the
QGraphicsView transform:

* continuous exponential wheel scaling (mouse wheel + precision trackpad),
* cursor-anchored zoom,
* at most one additional transform per ~display frame during a wheel burst,
* fast pixmap transformation while the gesture is active,
* automatic high-quality repaint shortly after the gesture settles,
* native macOS pinch-zoom support through ``QNativeGestureEvent``.
"""

import math
from typing import Callable

from PySide6.QtCore import QEvent, QObject, QPoint, QTimer, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QGraphicsView


class GraphicsZoomController(QObject):
    """Frame-coalesced, cursor-anchored zoom controller for ``QGraphicsView``."""

    FRAME_MS = 16
    QUALITY_SETTLE_MS = 90
    MOUSE_NOTCH_FACTOR = 1.15
    PIXEL_EXPONENT = 0.0015

    def __init__(
        self,
        view: QGraphicsView,
        *,
        minimum_scale: float = 0.05,
        maximum_scale: float = 12.0,
        on_manual_zoom: Callable[[], None] | None = None,
        smooth_when_idle: bool = True,
    ) -> None:
        super().__init__(view)
        self.view = view
        self.minimum_scale = float(minimum_scale)
        self.maximum_scale = float(maximum_scale)
        self.on_manual_zoom = on_manual_zoom
        self.smooth_when_idle = bool(smooth_when_idle)

        self._pending = False
        self._requested_scale = self.current_scale()
        self._anchor = QPoint()
        self._fast_quality = False

        self._frame_timer = QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._frame_timer.timeout.connect(self._frame_tick)

        self._quality_timer = QTimer(self)
        self._quality_timer.setSingleShot(True)
        self._quality_timer.timeout.connect(self._restore_quality)

        # Avoid enlarging invalidated regions for antialiasing during every
        # transform.  Selection/brush overlays already include their own margin.
        self.view.setOptimizationFlag(
            QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing,
            True,
        )
        if self.smooth_when_idle:
            self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    def current_scale(self) -> float:
        return max(1.0e-9, float(self.view.transform().m11()))

    @classmethod
    def wheel_factor(cls, event) -> float:
        """Return a continuous zoom factor for mouse/trackpad wheel input."""
        pixel = event.pixelDelta().y() if hasattr(event, "pixelDelta") else 0
        if pixel:
            exponent = float(pixel) * cls.PIXEL_EXPONENT
        else:
            angle = event.angleDelta().y() if hasattr(event, "angleDelta") else 0
            if not angle:
                return 1.0
            exponent = (float(angle) / 120.0) * math.log(cls.MOUSE_NOTCH_FACTOR)
        # One malformed device event must never jump across the entire scale range.
        return max(0.5, min(2.0, math.exp(exponent)))

    @staticmethod
    def _event_anchor(event, view: QGraphicsView) -> QPoint:
        try:
            return event.position().toPoint()
        except Exception:
            return view.viewport().rect().center()

    def _begin_fast_quality(self) -> None:
        if not self._fast_quality:
            # BallonsTranslator similarly avoids globally forcing smooth pixmap
            # transforms on Qt6.  During an active gesture responsiveness matters
            # more than a transient sub-pixel interpolation pass.
            self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            self._fast_quality = True
        self._quality_timer.start(self.QUALITY_SETTLE_MS)

    def _restore_quality(self) -> None:
        if not self._fast_quality:
            return
        if self.smooth_when_idle:
            self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self._fast_quality = False
        self.view.viewport().update()

    def cancel_pending(self, *, restore_quality: bool = True) -> None:
        """Cancel delayed transforms before fit/reset/image replacement."""
        self._frame_timer.stop()
        self._quality_timer.stop()
        self._pending = False
        self._requested_scale = self.current_scale()
        if restore_quality:
            self._restore_quality()

    def _apply_target(self, target: float, anchor: QPoint) -> None:
        current = self.current_scale()
        target = max(self.minimum_scale, min(self.maximum_scale, float(target)))
        if math.isclose(current, target, rel_tol=1.0e-6, abs_tol=1.0e-8):
            return

        # Explicit anchor preservation is more stable than relying on
        # AnchorUnderMouse when the transform is applied later by a frame timer.
        scene_before = self.view.mapToScene(anchor)
        factor = target / current
        previous_anchor = self.view.transformationAnchor()
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        try:
            self.view.scale(factor, factor)
            scene_after = self.view.mapToScene(anchor)
            delta = scene_after - scene_before
            self.view.translate(delta.x(), delta.y())
        finally:
            self.view.setTransformationAnchor(previous_anchor)

    def _frame_tick(self) -> None:
        if not self._pending:
            return
        target = self._requested_scale
        anchor = QPoint(self._anchor)
        self._pending = False
        self._apply_target(target, anchor)

    def queue_factor(self, factor: float, anchor: QPoint | None = None) -> bool:
        current = self.current_scale()
        base = self._requested_scale if self._pending else current
        target = max(self.minimum_scale, min(self.maximum_scale, base * float(factor)))
        if math.isclose(base, target, rel_tol=1.0e-6, abs_tol=1.0e-8):
            return False

        if self.on_manual_zoom is not None:
            self.on_manual_zoom()
        self._begin_fast_quality()
        self._requested_scale = target
        self._anchor = QPoint(anchor if anchor is not None else self.view.viewport().rect().center())
        self._pending = True

        if not self._frame_timer.isActive():
            # First response is immediate; subsequent high-frequency input in the
            # same display frame is collapsed into one transform at the timer tick.
            self._frame_tick()
            self._frame_timer.start(self.FRAME_MS)
        return True

    def zoom_now(self, factor: float, anchor: QPoint | None = None) -> bool:
        """Immediate zoom for toolbar/button commands while retaining fast quality."""
        self.cancel_pending(restore_quality=False)
        current = self.current_scale()
        target = max(self.minimum_scale, min(self.maximum_scale, current * float(factor)))
        if math.isclose(current, target, rel_tol=1.0e-6, abs_tol=1.0e-8):
            return False
        if self.on_manual_zoom is not None:
            self.on_manual_zoom()
        self._begin_fast_quality()
        self._requested_scale = target
        self._apply_target(target, anchor or self.view.viewport().rect().center())
        return True

    def handle_wheel(self, event) -> bool:
        factor = self.wheel_factor(event)
        if math.isclose(factor, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-12):
            return False
        changed = self.queue_factor(factor, self._event_anchor(event, self.view))
        if changed:
            event.accept()
        return changed

    def handle_native_event(self, event) -> bool:
        """Handle macOS/Qt native pinch gestures without affecting other events."""
        if event.type() != QEvent.Type.NativeGesture:
            return False
        try:
            if event.gestureType() != Qt.NativeGestureType.ZoomNativeGesture:
                return False
            value = float(event.value())
        except Exception:
            return False
        if not math.isfinite(value) or abs(value) < 1.0e-8:
            return False
        factor = max(0.5, min(1.5, 1.0 + value))
        changed = self.queue_factor(factor, self._event_anchor(event, self.view))
        if changed:
            event.accept()
        return changed


__all__ = ["GraphicsZoomController"]
