from __future__ import annotations

import sys
import time
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QSize, QThread, QTimer, Signal, QRectF
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont, QImageReader, QImage, QIcon, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton, QToolButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QStackedWidget, QScrollArea,
    QFileDialog, QMessageBox, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy,
    QGraphicsView, QGraphicsScene, QProgressBar, QSplitter, QButtonGroup, QRadioButton,
    QPlainTextEdit, QDialog, QListWidget, QListWidgetItem, QMenu, QSlider,
)

from .config import PipelineConfig
from .version import __version__
from .models import PagePair
from .pairing import pair_directories, pairing_method
from .pipeline import TransferPipeline, PipelineCancelled
from .runtime_catalog import probe_components
from .io_utils import load_json, save_json, write_image
from .review_apply import apply_review_page, generate_remove_text_preview
from .manual_effect import map_target_bbox_to_source, build_manual_effect_masks, build_reveal_seed_mask, estimate_source_background, composite_source_text_delta
from .workspace import page_id_for_pair, resolve_page_workspace
from .schema_compat import as_dict, as_dict_rows, as_list, normalize_project, normalize_overrides, normalize_review_applied
from .result_state import commit_reviewed_result
from .manual_review_service import commit_manual_effect
from .page_management import (
    PAGE_TYPE_INFO, MANUAL_PAGE_TYPES, PageMark,
    default_mark, manual_mark, marks_from_json, marks_to_json, page_mark_key,
    page_type_color, page_type_label, resolve_mark,
)

APP_NAME = "Manga HD Transfer Studio"
VERSION = __version__

# KCC-Kindle-CHS inspired macOS palette, softened to the requested pale blue.
ACCENT = "#7398D2"
ACCENT_HOVER = "#6289C5"
BG = "#F7F9FC"
CARD = "#FFFFFF"
CARD_BLUE = "#FAFCFF"
TEXT = "#202A36"
MUTED = "#6F7C8E"
MUTED_2 = "#9AA6B5"
BORDER = "#E2E8F0"
BORDER_STRONG = "#D4DDE8"
BLUE_SOFT = "#EEF4FC"
GREEN = "#2E8B6D"
GREEN_SOFT = "#ECF8F3"
ORANGE = "#B97828"
ORANGE_SOFT = "#FFF6E8"
RED = "#C94D5D"
RED_SOFT = "#FFF0F2"


def _configure_responsive_dialog(dialog: QDialog, preferred: tuple[int, int], minimum: tuple[int, int]) -> None:
    """Keep editor dialogs inside the usable desktop on macOS/HiDPI screens.

    Requested sizes such as 1260x860 can exceed the *available* desktop after the
    macOS menu bar/Dock or display scaling is applied.  Qt may then create a
    window whose lower controls are outside the visible area.  Size against
    availableGeometry instead and leave a small safety margin.
    """
    pw, ph = map(int, preferred)
    mw, mh = map(int, minimum)
    screen = dialog.screen() or QApplication.primaryScreen()
    if screen is not None:
        geo = screen.availableGeometry()
        cap_w = max(520, int(round(geo.width() * 0.94)))
        cap_h = max(420, int(round(geo.height() * 0.90)))
        pw = min(pw, cap_w)
        ph = min(ph, cap_h)
        mw = min(mw, cap_w)
        mh = min(mh, cap_h)
    dialog.setMinimumSize(max(520, mw), max(420, mh))
    dialog.resize(max(dialog.minimumWidth(), pw), max(dialog.minimumHeight(), ph))
    dialog.setSizeGripEnabled(True)


def _fit_scene_rect(scene: QGraphicsScene) -> QRectF:
    """Slightly pad scene fitting so the outermost page pixels stay visible."""
    rect = scene.sceneRect()
    if rect.isNull() or rect.isEmpty():
        return rect
    return rect.adjusted(-3.0, -3.0, 3.0, 3.0)

STYLE = f"""
QMainWindow {{ background:{BG}; }}
QWidget {{ color:{TEXT}; font-size:13px; }}
QWidget#root {{ background:{BG}; }}
QFrame#topbar, QFrame#navBar, QFrame#statusShell {{ background:{CARD}; }}
QFrame#topbar {{ border-bottom:1px solid {BORDER}; }}
QFrame#navBar {{ border-bottom:1px solid {BORDER}; }}
QFrame#card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; }}
QFrame#cardBlue {{ background:{CARD_BLUE}; border:1px solid #DCE6F3; border-radius:12px; }}
QLabel#appTitle {{ font-size:22px; font-weight:700; letter-spacing:.2px; }}
QLabel#appSubtitle {{ color:{MUTED}; font-size:12px; }}
QLabel#pageTitle {{ font-size:16px; font-weight:700; }}
QLabel#sectionTitle {{ font-size:15px; font-weight:700; }}
QLabel#hint {{ color:{MUTED}; font-size:12px; }}
QLabel#quiet {{ color:{MUTED_2}; font-size:11px; }}
QLabel#badge {{ color:{ACCENT}; background:{BLUE_SOFT}; border-radius:10px; padding:4px 9px; font-weight:700; font-size:11px; }}
QPushButton {{
    min-height:29px; border-radius:7px; border:1px solid {BORDER_STRONG};
    background:#FFFFFF; padding:2px 11px;
}}
QPushButton:hover {{ background:#F2F6FB; border-color:#C1CFDF; }}
QPushButton:pressed {{ background:#EAF0F7; }}
QPushButton#primary {{ color:white; background:{ACCENT}; border:1px solid {ACCENT}; font-weight:650; min-height:33px; }}
QPushButton#primary:hover {{ background:{ACCENT_HOVER}; }}
QPushButton#softPrimary {{ color:{ACCENT_HOVER}; background:{BLUE_SOFT}; border:1px solid #CBDBF6; font-weight:600; }}
QPushButton#danger {{ color:{RED}; border-color:#E7AEB7; background:{RED_SOFT}; font-weight:650; }}
QPushButton#danger:hover {{ background:#FFE4E8; border-color:#DA8C99; }}
QPushButton#stopTask {{ color:white; background:{RED}; border:1px solid {RED}; min-width:86px; min-height:31px; font-weight:700; }}
QPushButton#stopTask:hover {{ background:#B94352; border-color:#B94352; }}
QPushButton#stopTask:disabled {{ color:#A7AFBA; background:#EEF1F5; border-color:#E0E5EB; }}
QPushButton#navButton {{ border:0; border-bottom:2px solid transparent; border-radius:0; background:transparent; color:{MUTED}; min-height:32px; padding:0 12px; font-weight:600; }}
QPushButton#navButton:hover {{ background:transparent; color:{TEXT}; }}
QPushButton#navButton:checked {{ background:transparent; color:{ACCENT_HOVER}; border-bottom:2px solid {ACCENT}; }}

QPushButton#segmented {{ border:0; border-bottom:2px solid transparent; border-radius:0; background:transparent; color:{MUTED}; min-height:32px; padding:0 12px; font-weight:600; }}
QPushButton#segmented:hover {{ background:transparent; color:{TEXT}; }}
QPushButton#segmented:checked {{ background:transparent; color:{ACCENT_HOVER}; border-bottom:2px solid {ACCENT}; }}
QPushButton#pageNav {{ min-width:72px; min-height:28px; padding:1px 10px; }}
QLabel#pageCounter {{ color:{TEXT}; background:{BLUE_SOFT}; border:1px solid #D7E3F4; border-radius:8px; padding:4px 10px; font-weight:650; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    min-height:29px; border:1px solid {BORDER}; border-radius:7px; background:white; padding:0 8px;
}}
QComboBox::drop-down {{ border:0; width:24px; }}
QTableWidget {{ background:white; border:1px solid {BORDER}; border-radius:9px; gridline-color:#EEF2F6; selection-background-color:{BLUE_SOFT}; selection-color:{TEXT}; }}
QHeaderView::section {{ background:#F7F9FC; color:{MUTED}; border:0; border-bottom:1px solid {BORDER}; padding:7px 8px; font-weight:600; }}
QScrollArea {{ border:0; background:transparent; }}
QCheckBox, QRadioButton {{ spacing:7px; min-height:25px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width:16px; height:16px; }}
QProgressBar {{ border:1px solid {BORDER}; border-radius:6px; background:#EDF2F7; text-align:center; min-height:20px; color:{MUTED}; }}
QProgressBar::chunk {{ background:{ACCENT}; border-radius:5px; }}
QGraphicsView {{ background:#F2F5F9; border:0; border-radius:8px; }}
QStatusBar {{ background:{CARD}; border-top:1px solid {BORDER}; color:{MUTED}; }}
QToolTip {{ background:#FFFFFF; color:{TEXT}; border:1px solid {BORDER}; padding:5px; }}
QScrollBar:vertical {{ background:transparent; width:8px; margin:2px; }}
QScrollBar::handle:vertical {{ background:#CBD5E1; min-height:28px; border-radius:4px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QScrollBar:horizontal {{ background:transparent; height:8px; margin:2px; }}
QScrollBar::handle:horizontal {{ background:#CBD5E1; min-width:28px; border-radius:4px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width:0; }}
"""


def _studio_default_config() -> PipelineConfig:
    cfg = PipelineConfig()
    # macOS: use the same Apple OCR family as Novel Formatter. ``apple`` means
    # Swift VisionKit Live Text first with ExtractText Shortcut fallback.
    if sys.platform == "darwin":
        cfg.ocr.backend = "apple"
        cfg.ocr.source_backend = "apple"
        cfg.ocr.target_backend = "apple"
        # Apple-silicon desktop defaults.  Keep the device on ``auto`` so an
        # available MPS runtime is selected, but use the M4 CPU efficiently
        # when PyTorch/MPS is unavailable (as it is in some headless or
        # sandboxed Python environments).  One prefetch worker avoids copying
        # several full-resolution pages into a 16 GB Mac at once.
        cfg.runtime.device = "auto"
        cfg.runtime.cpu_thread_ratio = 0.80
        cfg.runtime.max_cpu_threads = 10
        cfg.runtime.release_cache_every = 4
        cfg.batch.prefetch_workers = 1
    return cfg


@dataclass
class StudioState:
    source_dir: str = ""
    target_dir: str = ""
    output_dir: str = ""
    pairs: list[PagePair] = field(default_factory=list)
    selected_index: int = 0
    config: PipelineConfig = field(default_factory=_studio_default_config)
    last_project: Any = None
    last_result_path: str = ""
    projects_by_page: dict[str, Any] = field(default_factory=dict)
    batch_status: dict[str, tuple[str, str]] = field(default_factory=dict)
    page_marks: dict[str, dict[str, Any]] = field(default_factory=dict)
    unmatched_source: list[str] = field(default_factory=list)
    unmatched_target: list[str] = field(default_factory=list)


class PipelineWorker(QThread):
    done = Signal(object, str)
    failed = Signal(str)
    progress = Signal(int, int, str, str, bool)
    cancelled = Signal()

    def __init__(
        self, *, config: PipelineConfig, pair: PagePair | None = None,
        source_dir: str = "", target_dir: str = "", output_dir: str = "",
        page_mark: dict[str, Any] | PageMark | None = None,
        pairs_override: list[PagePair] | None = None,
        page_marks: dict[str, dict[str, Any]] | None = None,
    ):
        super().__init__()
        self.config = config
        self.pair = pair
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.output_dir = output_dir
        self.page_mark = page_mark
        self.pairs_override = list(pairs_override) if pairs_override is not None else None
        self.page_marks = dict(page_marks) if page_marks is not None else None
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def _cancelled(self):
        return self._cancel_requested

    def _progress(self, done, total, pair, status, cache_hit=False, message=""):
        label = Path(pair.target_path).name if pair is not None else ""
        text = message or status
        self.progress.emit(int(done), int(total), label, text, bool(cache_hit))

    def run(self):
        try:
            pipeline = TransferPipeline(self.config)
            out = Path(self.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            if self.pair is not None:
                page_root = out / "pages" / page_id_for_pair(self.pair)
                final_path = out / "final" / (Path(self.pair.target_path).stem + ".png")
                final_path.parent.mkdir(parents=True, exist_ok=True)
                project = pipeline.process_page(
                    self.pair, page_root, final_path,
                    page_mark=self.page_mark, cancel_cb=self._cancelled,
                )
                if self._cancelled():
                    self.cancelled.emit()
                    return
                self.done.emit(project, str(final_path))
            else:
                project = pipeline.run_book(
                    self.source_dir, self.target_dir, self.output_dir,
                    progress_cb=self._progress, cancel_cb=self._cancelled,
                    resume=self.config.batch.resume,
                    pairs_override=self.pairs_override, page_marks=self.page_marks,
                )
                if as_dict(getattr(project, "meta", {})).get("cancelled"):
                    self.cancelled.emit()
                self.done.emit(project, self.output_dir)
        except PipelineCancelled:
            self.cancelled.emit()
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")



class ComponentProbeWorker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config

    def run(self):
        try:
            self.done.emit(probe_components(self.config))
        except Exception as exc:
            self.failed.emit(str(exc))


class ImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = None
        self._current_key = None
        # Switching workbench tabs used to synchronously decode the same multi-MB
        # manga PNG/JPEG every time. Keep a tiny LRU of decoded QPixmaps; QPixmap is
        # implicitly shared by Qt, so reuse is cheap while the cache remains bounded.
        self._pixmap_cache: OrderedDict[tuple[str, int, int], QPixmap] = OrderedDict()
        self._pixmap_cache_limit = 5
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMinimumHeight(260)

    def set_image(self, path: str | Path | None):
        if not path:
            if self._item is not None:
                self._scene.clear(); self._item = None
            self._current_key = None
            return
        p = Path(path)
        try:
            st = p.stat()
        except OSError:
            if self._item is not None:
                self._scene.clear(); self._item = None
            self._current_key = None
            return
        key = (str(p.resolve()), int(st.st_mtime_ns), int(st.st_size))
        if key == self._current_key and self._item is not None:
            return
        pix = self._pixmap_cache.pop(key, None)
        if pix is None:
            pix = QPixmap(str(p))
            if pix.isNull():
                return
        self._pixmap_cache[key] = pix
        while len(self._pixmap_cache) > self._pixmap_cache_limit:
            self._pixmap_cache.popitem(last=False)
        self._scene.clear(); self._item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(self._item.boundingRect())
        self._current_key = key
        self.fit_to_window()

    def clear_cache(self):
        """Force the next refresh to decode the file again.

        Manual review can rewrite ``final_reviewed.png`` in-place.  Even though
        mtime-based cache keys normally catch that, explicitly invalidating the
        preview after a committed edit removes filesystem timestamp granularity
        as a source of stale UI pixels.
        """
        self._pixmap_cache.clear()
        self._current_key = None

    def fit_to_window(self):
        if self._item is not None:
            self.resetTransform()
            self.fitInView(_fit_scene_rect(self._scene), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_to_window()


class Card(QFrame):
    def __init__(self, title: str = "", subtitle: str = "", *, blue=False, parent=None):
        super().__init__(parent)
        self.setObjectName("cardBlue" if blue else "card")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 14)
        self.layout.setSpacing(10)
        if title:
            t = QLabel(title); t.setObjectName("sectionTitle"); self.layout.addWidget(t)
        if subtitle:
            s = QLabel(subtitle); s.setObjectName("hint"); s.setWordWrap(True); self.layout.addWidget(s)


class PathRow(QWidget):
    changed = Signal(str)

    def __init__(self, label: str, button_text: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(8)
        self.caption = QLabel(label); self.caption.setFixedWidth(88)
        self.path = QLabel("未选择"); self.path.setObjectName("hint"); self.path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.button = QPushButton(button_text)
        lay.addWidget(self.caption); lay.addWidget(self.path, 1); lay.addWidget(self.button)

    def set_path(self, path: str):
        self.path.setText(path or "未选择")
        self.path.setToolTip(path or "")


class ZoomPreviewView(QGraphicsView):
    """Full-page preview with mouse-wheel zoom and hand-drag panning."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = None
        self._zoom = 1.0
        self._auto_fit = True; self._fit_pending = False
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMinimumSize(360, 420)

    def set_image(self, path: str | Path | None):
        self._scene.clear(); self._item = None; self._zoom = 1.0; self.resetTransform()
        if not path:
            self._show_message("无图片")
            return
        p = Path(path)
        if not p.exists():
            self._show_message(f"图片不存在\n{p.name}")
            return
        reader = QImageReader(str(p)); reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            self._show_message(f"无法读取图片\n{p.name}")
            return
        pix = QPixmap.fromImage(image)
        self._item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(self._item.boundingRect())
        self.fit_to_window()

    def _show_message(self, text: str):
        item = self._scene.addText(text)
        item.setDefaultTextColor(QColor(MUTED))
        self._scene.setSceneRect(item.boundingRect().adjusted(-30, -30, 30, 30))

    def _apply_fit(self):
        self._fit_pending = False
        if self._item is None or not self._auto_fit or self.viewport().width() < 8 or self.viewport().height() < 8:
            return
        self.resetTransform()
        self.fitInView(_fit_scene_rect(self._scene), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = max(0.01, float(self.transform().m11()))

    def fit_to_window(self):
        if self._item is not None:
            self._auto_fit = True; self._apply_fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit and not self._fit_pending:
            self._fit_pending = True; QTimer.singleShot(0, self._apply_fit)

    def actual_size(self):
        if self._item is not None:
            self._auto_fit = False; self.resetTransform(); self._zoom = 1.0

    def zoom_by(self, factor: float):
        if self._item is None:
            return
        current = float(self.transform().m11())
        target = current * float(factor)
        if target < 0.05 or target > 12.0:
            return
        self._auto_fit = False
        self.scale(float(factor), float(factor)); self._zoom = target

    def wheelEvent(self, event):
        if self._item is None:
            super().wheelEvent(event); return
        self.zoom_by(1.16 if event.angleDelta().y() > 0 else (1.0 / 1.16))
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.fit_to_window(); event.accept()


class MaskPaintView(QGraphicsView):
    """Lightweight clear-mask overlay editor for page-local review.

    Left drag adds clear-mask pixels, right drag erases them. The editor never
    touches the source/target originals; it only writes manual_clear_mask.png.
    """
    mask_changed = Signal()

    def __init__(self, image_path: str | Path, mask: Any, parent=None):
        super().__init__(parent)
        import cv2 as _cv2
        import numpy as _np
        self._cv2 = _cv2; self._np = _np
        self._scene = QGraphicsScene(self); self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._image_path = str(image_path)
        reader = QImageReader(self._image_path); reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            raise ValueError(f"无法读取图片：{Path(self._image_path).name}")
        self._base_qimage = image.convertToFormat(QImage.Format.Format_RGBA8888)
        self._base_item = self._scene.addPixmap(QPixmap.fromImage(self._base_qimage))
        h, w = self._base_qimage.height(), self._base_qimage.width()
        arr = _np.asarray(mask, dtype=_np.uint8)
        if arr.shape != (h, w):
            raise ValueError("清除蒙版尺寸与目标图片不一致")
        self.mask = (arr > 0).astype(_np.uint8) * 255
        self.brush_size = 24
        self._painting = False; self._erase = False; self._last = None
        self._panning = False; self._pan_last = None
        self._auto_fit = True; self._fit_pending = False
        self._overlay_item = self._scene.addPixmap(QPixmap())
        self._scene.setSceneRect(0, 0, w, h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._refresh_overlay(); self.fit_to_window()

    def _refresh_overlay(self):
        h, w = self.mask.shape
        rgba = self._np.zeros((h, w, 4), dtype=self._np.uint8)
        sel = self.mask > 0
        rgba[sel, 0] = 230; rgba[sel, 1] = 70; rgba[sel, 2] = 85; rgba[sel, 3] = 110
        q = QImage(rgba.data, w, h, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        self._overlay_item.setPixmap(QPixmap.fromImage(q))
        self._overlay_item.setZValue(2)

    def _apply_fit(self):
        self._fit_pending = False
        if not self._auto_fit or self.viewport().width() < 8 or self.viewport().height() < 8:
            return
        self.resetTransform()
        self.fitInView(_fit_scene_rect(self._scene), Qt.AspectRatioMode.KeepAspectRatio)

    def fit_to_window(self):
        self._auto_fit = True
        self._apply_fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit and not self._fit_pending:
            self._fit_pending = True
            QTimer.singleShot(0, self._apply_fit)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
        current = float(self.transform().m11()); target = current * factor
        if 0.05 <= target <= 12.0:
            self._auto_fit = False
            self.scale(factor, factor)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.fit_to_window(); event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_last = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept(); return
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            super().mousePressEvent(event); return
        self._painting = True; self._erase = event.button() == Qt.MouseButton.RightButton
        pos = self.mapToScene(event.position().toPoint()); self._last = (int(pos.x()), int(pos.y()))
        self._paint_to(self._last); event.accept()

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_last is not None:
            now_view = event.position().toPoint()
            delta = now_view - self._pan_last
            self._pan_last = now_view
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept(); return
        if not self._painting or self._last is None:
            super().mouseMoveEvent(event); return
        pos = self.mapToScene(event.position().toPoint()); now = (int(pos.x()), int(pos.y()))
        val = 0 if self._erase else 255
        self._cv2.line(self.mask, self._last, now, int(val), max(1, int(self.brush_size)), lineType=self._cv2.LINE_AA)
        self._last = now; self._refresh_overlay(); self.mask_changed.emit(); event.accept()

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False; self._pan_last = None
            self.viewport().unsetCursor(); event.accept(); return
        if self._painting:
            self._painting = False; self._last = None; event.accept(); return
        super().mouseReleaseEvent(event)

    def _paint_to(self, pt):
        x, y = pt
        if 0 <= x < self.mask.shape[1] and 0 <= y < self.mask.shape[0]:
            self._cv2.circle(self.mask, (x, y), max(1, int(self.brush_size // 2)), 0 if self._erase else 255, -1, lineType=self._cv2.LINE_AA)
            self._refresh_overlay(); self.mask_changed.emit()


class MaskEditorDialog(QDialog):
    def __init__(self, image_path: str | Path, initial_mask: Any, parent=None):
        super().__init__(parent); self.setWindowTitle("清除蒙版编辑器")
        _configure_responsive_dialog(self, (980, 820), (720, 520))
        root = QVBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(8)
        hint = QLabel("红色区域 = 将被清除的日文。左键拖动增加，右键拖动擦除；滚轮缩放。只修改蒙版，不会直接破坏原图。")
        hint.setObjectName("hint"); hint.setWordWrap(True); root.addWidget(hint)
        self.view = MaskPaintView(image_path, initial_mask, self); root.addWidget(self.view, 1)
        brush_row = QHBoxLayout(); brush_row.addWidget(QLabel("画笔大小"))
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(4, 120); self.slider.setValue(24)
        self.size_label = QLabel("24 px"); brush_row.addWidget(self.slider, 1); brush_row.addWidget(self.size_label); root.addLayout(brush_row)
        action_row = QHBoxLayout()
        self.fit_button = QPushButton("适合窗口"); self.clear_button = QPushButton("清空蒙版"); self.save_button = QPushButton("保存蒙版"); self.save_button.setObjectName("primary"); self.cancel_button = QPushButton("取消")
        action_row.addWidget(self.fit_button); action_row.addWidget(self.clear_button); action_row.addStretch(1); action_row.addWidget(self.save_button); action_row.addWidget(self.cancel_button); root.addLayout(action_row)
        self.slider.valueChanged.connect(self._brush); self.fit_button.clicked.connect(self.view.fit_to_window)
        self.clear_button.clicked.connect(self._clear); self.save_button.clicked.connect(self.accept); self.cancel_button.clicked.connect(self.reject)
        QTimer.singleShot(0, self.view.fit_to_window)

    def _brush(self, value: int):
        self.view.brush_size = int(value); self.size_label.setText(f"{int(value)} px")

    def _clear(self):
        self.view.mask[:] = 0; self.view._refresh_overlay()

    def result_mask(self):
        return self.view.mask.copy()



class RevealMaskDialog(QDialog):
    """Brush editor that previews target cleanup + aligned Chinese glyph reveal live."""

    def __init__(self, target_path: str | Path, aligned_source: Any, source_mask: Any, target_clear_mask: Any, initial_mask: Any, parent=None):
        super().__init__(parent); self.setWindowTitle("擦除显字编辑器")
        _configure_responsive_dialog(self, (1040, 860), (740, 540))
        import cv2
        import numpy as np
        self._cv2 = cv2; self._np = np
        self._target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if self._target is None:
            raise ValueError(f"无法读取目标图片：{Path(target_path).name}")
        self._source = np.asarray(aligned_source, dtype=np.uint8)
        self._source_mask = (np.asarray(source_mask) > 0).astype(np.uint8) * 255
        self._clear_mask = (np.asarray(target_clear_mask) > 0).astype(np.uint8) * 255
        if self._source.shape[:2] != self._target.shape[:2] or self._source_mask.shape != self._target.shape[:2] or self._clear_mask.shape != self._target.shape[:2]:
            raise ValueError("擦除显字图层尺寸不一致")
        self._cleaned = cv2.inpaint(self._target, self._clear_mask, 3.0, cv2.INPAINT_TELEA) if cv2.countNonZero(self._clear_mask) else self._target.copy()
        self._source_background = estimate_source_background(self._source, self._source_mask)
        # Pre-render the fully revealed state once. Brush movement then only
        # selects pixels between TARGET and this cache instead of rebuilding a
        # full-page inpaint + delta composite on every mouse event.
        self._full_reveal = self._target.copy()
        all_clear = self._clear_mask > 0
        if np.any(all_clear):
            self._full_reveal[all_clear] = self._cleaned[all_clear]
        if cv2.countNonZero(self._source_mask):
            self._full_reveal, _ = composite_source_text_delta(
                self._full_reveal, self._source, self._source_mask,
                source_background=self._source_background,
            )

        root=QVBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(8)
        hint=QLabel("左键擦除/显字，右键恢复日文；滚轮缩放。预览中只在你的 Reveal Mask 范围内清除日文笔画并显示下层中文透明文字，目标彩图背景仍来自高清日文图。")
        hint.setObjectName("hint"); hint.setWordWrap(True); root.addWidget(hint)
        self.view=MaskPaintView(target_path, initial_mask, self); root.addWidget(self.view,1)
        brush_row=QHBoxLayout(); brush_row.addWidget(QLabel("画笔大小"))
        self.slider=QSlider(Qt.Orientation.Horizontal); self.slider.setRange(4,160); self.slider.setValue(32)
        self.size_label=QLabel("32 px"); brush_row.addWidget(self.slider,1); brush_row.addWidget(self.size_label); root.addLayout(brush_row)
        action_row=QHBoxLayout()
        self.fit_button=QPushButton("适合窗口"); self.auto_button=QPushButton("恢复自动建议"); self.clear_button=QPushButton("全部恢复日文"); self.save_button=QPushButton("保存擦除显字"); self.save_button.setObjectName("primary"); self.cancel_button=QPushButton("取消")
        action_row.addWidget(self.fit_button); action_row.addWidget(self.auto_button); action_row.addWidget(self.clear_button); action_row.addStretch(1); action_row.addWidget(self.save_button); action_row.addWidget(self.cancel_button); root.addLayout(action_row)
        self._auto_seed=(np.asarray(initial_mask)>0).astype(np.uint8)*255
        self.slider.valueChanged.connect(self._brush); self.fit_button.clicked.connect(self.view.fit_to_window)
        self.auto_button.clicked.connect(self._restore_auto); self.clear_button.clicked.connect(self._clear); self.save_button.clicked.connect(self.accept); self.cancel_button.clicked.connect(self.reject)
        self.view.mask_changed.connect(self._refresh_preview)
        self._brush(32); self._refresh_preview(); QTimer.singleShot(0, self.view.fit_to_window)

    def _brush(self, value:int):
        self.view.brush_size=int(value); self.size_label.setText(f"{int(value)} px")

    def _restore_auto(self):
        self.view.mask[:]=self._auto_seed; self.view._refresh_overlay(); self._refresh_preview()

    def _clear(self):
        self.view.mask[:]=0; self.view._refresh_overlay(); self._refresh_preview()

    def _refresh_preview(self):
        gate=self.view.mask>0
        out=self._target.copy()
        if self._np.any(gate):
            out[gate]=self._full_reveal[gate]
        rgb=self._cv2.cvtColor(out,self._cv2.COLOR_BGR2RGB)
        h,w=rgb.shape[:2]
        q=QImage(rgb.data,w,h,int(rgb.strides[0]),QImage.Format.Format_RGB888).copy()
        self.view._base_item.setPixmap(QPixmap.fromImage(q))
        # The live image is the truth; keep the red overlay extremely light so
        # the user can still see where the editable reveal window exists.
        rgba=self._np.zeros((h,w,4),dtype=self._np.uint8); sel=gate
        rgba[sel,0]=70; rgba[sel,1]=150; rgba[sel,2]=235; rgba[sel,3]=26
        qo=QImage(rgba.data,w,h,int(rgba.strides[0]),QImage.Format.Format_RGBA8888).copy()
        self.view._overlay_item.setPixmap(QPixmap.fromImage(qo)); self.view._overlay_item.setZValue(2)

    def result_mask(self):
        return self.view.mask.copy()

    def result_patch_bgra(self):
        """Return the exact visible reveal as a sparse BGRA commit patch.

        Alpha is 255 only where the accepted brush gate actually changes TARGET.
        Re-applying this patch therefore reproduces the editor preview exactly
        without copying untouched background pixels.
        """
        gate = self.view.mask > 0
        changed = gate & self._np.any(self._full_reveal != self._target, axis=2)
        patch = self._np.zeros((self._target.shape[0], self._target.shape[1], 4), dtype=self._np.uint8)
        patch[:, :, :3] = self._full_reveal
        patch[:, :, 3] = changed.astype(self._np.uint8) * 255
        return patch


class RegionSelectView(QGraphicsView):
    """Zoomable image view with one editable rectangle in image coordinates."""
    selection_changed = Signal(object)

    def __init__(self, image_path: str | Path, *, editable: bool, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self); self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._editable = bool(editable)
        reader = QImageReader(str(image_path)); reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            raise ValueError(f"无法读取图片：{Path(image_path).name}")
        self._pix = QPixmap.fromImage(image)
        self._scene.addPixmap(self._pix)
        self._rect_item = self._scene.addRect(QRectF(), QPen(QColor(ACCENT), 2.0))
        self._rect_item.setZValue(4)
        self._scene.setSceneRect(0, 0, self._pix.width(), self._pix.height())
        self._start = None
        self._box: list[int] = []
        self._panning = False; self._pan_last = None
        self._auto_fit = True; self._fit_pending = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.fit_to_window()

    def _apply_fit(self):
        self._fit_pending = False
        if not self._auto_fit or self.viewport().width() < 8 or self.viewport().height() < 8:
            return
        self.resetTransform(); self.fitInView(_fit_scene_rect(self._scene), Qt.AspectRatioMode.KeepAspectRatio)

    def fit_to_window(self):
        self._auto_fit = True; self._apply_fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit and not self._fit_pending:
            self._fit_pending = True; QTimer.singleShot(0, self._apply_fit)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
        current = float(self.transform().m11()); target = current * factor
        if 0.05 <= target <= 12.0:
            self._auto_fit = False; self.scale(factor, factor)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.fit_to_window(); event.accept()

    def _clamp(self, x: float, y: float) -> tuple[int, int]:
        return (
            max(0, min(self._pix.width(), int(round(x)))),
            max(0, min(self._pix.height(), int(round(y)))),
        )

    def set_box(self, bbox: list[int] | tuple[int, int, int, int] | None, *, emit: bool = False):
        if not bbox or len(bbox) != 4:
            self._box = []; self._rect_item.setRect(QRectF())
            if emit: self.selection_changed.emit([])
            return
        x0, y0 = self._clamp(float(bbox[0]), float(bbox[1])); x1, y1 = self._clamp(float(bbox[2]), float(bbox[3]))
        x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            self._box = []; self._rect_item.setRect(QRectF())
        else:
            self._box = [x0, y0, x1, y1]
            self._rect_item.setRect(QRectF(float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
        if emit: self.selection_changed.emit(list(self._box))

    def box(self) -> list[int]:
        return list(self._box)

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = True; self._pan_last = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor); event.accept(); return
        if not self._editable or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event); return
        p = self.mapToScene(event.position().toPoint()); self._start = self._clamp(p.x(), p.y())
        self.set_box([self._start[0], self._start[1], self._start[0] + 2, self._start[1] + 2])
        event.accept()

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_last is not None:
            now_view = event.position().toPoint(); delta = now_view - self._pan_last; self._pan_last = now_view
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept(); return
        if not self._editable or self._start is None:
            super().mouseMoveEvent(event); return
        p = self.mapToScene(event.position().toPoint()); end = self._clamp(p.x(), p.y())
        self.set_box([self._start[0], self._start[1], end[0], end[1]])
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = False; self._pan_last = None; self.viewport().unsetCursor(); event.accept(); return
        if self._editable and self._start is not None and event.button() == Qt.MouseButton.LeftButton:
            p = self.mapToScene(event.position().toPoint()); end = self._clamp(p.x(), p.y())
            start = self._start; self._start = None
            self.set_box([start[0], start[1], end[0], end[1]], emit=True)
            event.accept(); return
        super().mouseReleaseEvent(event)


def _bbox_iou(a: list[int] | tuple[int, int, int, int], b: list[int] | tuple[int, int, int, int]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = [int(v) for v in a]; bx0, by0, bx1, by1 = [int(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0); ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    aa = max(0, ax1 - ax0) * max(0, ay1 - ay0); ba = max(0, bx1 - bx0) * max(0, by1 - by0)
    den = aa + ba - inter
    return float(inter / den) if den > 0 else 0.0


def _json_dict_rows(value) -> list[dict[str, Any]]:
    """Accept old/stale JSON fields without ever calling .get on booleans."""
    return as_dict_rows(value)


def _manual_effect_unhandled_candidates(candidates: list[dict[str, Any]], existing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_rows = as_dict_rows(existing_rows)
    candidates = as_dict_rows(candidates)
    handled_boxes = [list(r.get("target_bbox", []) or []) for r in existing_rows if len(list(r.get("target_bbox", []) or [])) == 4]
    out: list[dict[str, Any]] = []
    for row in candidates:
        box = as_list(row.get("target_bbox"))
        if len(box) != 4:
            continue
        if any(_bbox_iou(box, hb) >= 0.45 for hb in handled_boxes):
            continue
        out.append(dict(row))
    return out


class ManualEffectDialog(QDialog):
    """Human recovery editor for detector-missed bubbles/open/SFX text."""

    def __init__(self, source_path: str | Path, target_path: str | Path, project: dict[str, Any], parent=None, *, initial_bbox: list[int] | None = None, initial_mode: str | None = None, commit_handler=None, trace_handler=None):
        super().__init__(parent); self.setWindowTitle("人工补漏 / 开放式效果字")
        _configure_responsive_dialog(self, (1260, 860), (780, 560))
        self.project = dict(project or {}); self.source_path=Path(source_path); self.target_path=Path(target_path); self._reveal_mask=None; self._reveal_patch=None
        self._commit_handler=commit_handler; self._trace_handler=trace_handler; self._committed_directly=False
        import cv2 as _cv2_dialog
        self._target_cv = _cv2_dialog.imread(str(self.target_path), _cv2_dialog.IMREAD_COLOR)
        self._initial_bbox=list(initial_bbox or []); self._initial_mode=str(initial_mode or "")
        root = QVBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(9)
        hint = QLabel("在右侧高清日文图上拖框。这个框不依赖 OCR、气泡检测器或自动候选；程序会利用已保存的页面配准，从旧中文版提取对应区域。开放式效果字模式只迁移 SOURCE 支持的中文笔画，并自动估计/清除 TARGET 的日文笔画，尽量保留紫色、网点和人物背景。")
        hint.setObjectName("hint"); hint.setWordWrap(True); root.addWidget(hint)
        split = QSplitter(Qt.Orientation.Horizontal)
        left = QFrame(); left.setObjectName("card"); ll=QVBoxLayout(left); ll.setContentsMargins(8,8,8,8); ll.addWidget(QLabel("旧版中文 · 自动映射参考")); self.source_view=RegionSelectView(source_path, editable=False, parent=self); ll.addWidget(self.source_view,1)
        right = QFrame(); right.setObjectName("card"); rl=QVBoxLayout(right); rl.setContentsMargins(8,8,8,8); rl.addWidget(QLabel("高清日文 · 在这里框选遗漏区域")); self.target_view=RegionSelectView(target_path, editable=True, parent=self); rl.addWidget(self.target_view,1)
        split.addWidget(left); split.addWidget(right); split.setSizes([620,620]); split.setChildrenCollapsible(False); root.addWidget(split,1)

        form=QFormLayout(); form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.mode=QComboBox(); self.mode.addItem("彩色开放式文字 · 擦除显字（只改文字）","reveal_text"); self.mode.addItem("彩色开放式文字 · 自动迁移（只改文字）","effect_text"); self.mode.addItem("白色气泡 · 文字迁移 + X/Y 微调（不贴背景）","white_bubble_text")
        self.diff=QSpinBox(); self.diff.setRange(8,96); self.diff.setValue(24); self.diff.setSuffix(" / 255")
        self.expand=QSpinBox(); self.expand.setRange(1,5); self.expand.setValue(2); self.expand.setSuffix(" px")
        self.feather=QSpinBox(); self.feather.setRange(0,4); self.feather.setValue(0); self.feather.setSuffix(" px")
        self.auto_clear=QCheckBox("自动估计并清除框内 TARGET 日文笔画"); self.auto_clear.setChecked(True)
        nudge=QWidget(); nl=QHBoxLayout(nudge); nl.setContentsMargins(0,0,0,0); nl.setSpacing(6)
        self.offset_x=QSpinBox(); self.offset_x.setRange(-100,100); self.offset_x.setSuffix(" px X")
        self.offset_y=QSpinBox(); self.offset_y.setRange(-100,100); self.offset_y.setSuffix(" px Y")
        nl.addWidget(self.offset_x); nl.addWidget(self.offset_y); nl.addStretch(1)
        form.addRow("人工模式",self.mode); form.addRow("差异灵敏度",self.diff); form.addRow("笔画扩张",self.expand); form.addRow("边缘羽化",self.feather); form.addRow("日文处理",self.auto_clear); form.addRow("SOURCE 微调",nudge)
        root.addLayout(form)
        self.info=QLabel("尚未框选区域"); self.info.setObjectName("quiet"); root.addWidget(self.info)
        self.mode_guide=QLabel("彩色/紫色/人物背景上的文字请选择“擦除显字”；普通白色对白气泡请选择“白色气泡 · 文字迁移”。选框只是搜索范围，不会作为整块写入范围。")
        self.mode_guide.setObjectName("hint"); self.mode_guide.setWordWrap(True); root.addWidget(self.mode_guide)
        row=QHBoxLayout(); fit=QPushButton("适合窗口"); reset=QPushButton("清除框选"); save=QPushButton("应用此人工区域"); save.setObjectName("primary"); cancel=QPushButton("取消")
        row.addWidget(fit); row.addWidget(reset); row.addStretch(1); row.addWidget(save); row.addWidget(cancel); root.addLayout(row)
        self.target_view.selection_changed.connect(self._selection_changed)
        fit.clicked.connect(self._fit); reset.clicked.connect(lambda: self.target_view.set_box([], emit=True)); save.clicked.connect(self._accept_checked); cancel.clicked.connect(self.reject)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        if self._initial_mode:
            mi = self.mode.findData(self._initial_mode)
            if mi >= 0:
                self.mode.setCurrentIndex(mi)
        self._mode_changed()
        QTimer.singleShot(0, self._apply_initial_state)

    def _trace(self, stage: str, **payload):
        if self._trace_handler is None:
            return
        try:
            self._trace_handler(str(stage), dict(payload))
        except Exception:
            pass

    def was_committed_directly(self) -> bool:
        return bool(self._committed_directly)

    def _fit(self):
        self.source_view.fit_to_window(); self.target_view.fit_to_window()

    def _apply_initial_state(self):
        self._fit()
        if len(self._initial_bbox) == 4:
            self.target_view.set_box(self._initial_bbox, emit=True)

    def _mode_changed(self):
        effect = (self.mode.currentData() in ("effect_text", "reveal_text"))
        for w in (self.diff,self.expand,self.feather,self.auto_clear): w.setEnabled(effect)

    def _selection_changed(self, bbox):
        box=list(bbox or [])
        if len(box)!=4:
            self.source_view.set_box([]); self.info.setText("尚未框选区域"); return
        src_box=map_target_bbox_to_source(self.project,box); self.source_view.set_box(src_box)
        recommendation="彩色开放式文字建议使用“擦除显字”"
        if self._target_cv is not None:
            import cv2, numpy as np
            h,w=self._target_cv.shape[:2]; x0,y0,x1,y1=[int(v) for v in box]
            x0=max(0,min(w,x0));x1=max(0,min(w,x1));y0=max(0,min(h,y0));y1=max(0,min(h,y1))
            roi=self._target_cv[y0:y1,x0:x1]
            if roi.size:
                hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV); gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                white=float(np.mean((gray>=220)&(hsv[...,1]<=42)))
                recommendation=("检测为白底区域：建议“白色气泡 · 文字迁移 + X/Y 微调”" if white>=0.68 else "检测为彩色/复杂区域：建议“彩色开放式文字 · 擦除显字”")
        self.info.setText(f"TARGET {box[0]},{box[1]}–{box[2]},{box[3]} · SOURCE 自动映射约 {src_box[0]},{src_box[1]}–{src_box[2]},{src_box[3]} · {recommendation}")

    def _row_payload(self) -> dict[str, Any]:
        import uuid
        if not hasattr(self, "_row_id"):
            self._row_id=f"manual-effect-{uuid.uuid4().hex[:10]}"
        return {
            "id": self._row_id,
            "enabled": True,
            "mode": str(self.mode.currentData() or "reveal_text"),
            "target_bbox": self.target_view.box(),
            "diff_threshold": int(self.diff.value()),
            "edge_threshold": 52.0,
            "expand_px": int(self.expand.value()),
            "feather_px": int(self.feather.value()),
            "auto_clear_target": bool(self.auto_clear.isChecked()),
            "source_offset_x": int(self.offset_x.value()),
            "source_offset_y": int(self.offset_y.value()),
            "origin": "manual_open_text_editor",
        }

    def _accept_checked(self):
        box=self.target_view.box()
        if len(box)!=4:
            QMessageBox.information(self,"没有框选","请先在右侧高清日文图上拖出要人工补漏的区域。")
            return
        self._trace("manual_apply_clicked", mode=str(self.mode.currentData() or ""), target_bbox=list(box))
        if self.mode.currentData()=="reveal_text":
            import cv2
            import numpy as np
            source=cv2.imread(str(self.source_path),cv2.IMREAD_COLOR); target=cv2.imread(str(self.target_path),cv2.IMREAD_COLOR)
            if source is None or target is None:
                QMessageBox.warning(self,"无法建立图层","源图或目标图读取失败。")
                self._trace("reveal_prepare_failed", reason="source_or_target_unreadable")
                return
            try:
                masks=build_manual_effect_masks(source,target,self.project,self._row_payload())
                seed=build_reveal_seed_mask(masks.source_mask,masks.target_clear_mask,padding_px=max(4,int(self.expand.value())+2))
                dlg=RevealMaskDialog(self.target_path,masks.aligned_source,masks.source_mask,masks.target_clear_mask,seed,self)
            except Exception as exc:
                self._trace("reveal_prepare_failed", reason=str(exc))
                QMessageBox.critical(self,"无法打开擦除显字编辑器",str(exc)); return
            self._trace("reveal_editor_opened", source_pixels=int(cv2.countNonZero(masks.source_mask)), target_clear_pixels=int(cv2.countNonZero(masks.target_clear_mask)))
            if dlg.exec()!=QDialog.DialogCode.Accepted:
                self._trace("reveal_editor_cancelled")
                return
            reveal_mask=dlg.result_mask()
            reveal_patch=dlg.result_patch_bgra()
            source_visible=cv2.bitwise_and(masks.source_mask, (reveal_mask>0).astype(np.uint8)*255)
            patch_pixels=int(cv2.countNonZero(reveal_patch[:, :, 3])) if reveal_patch is not None else 0
            self._trace("reveal_editor_saved", reveal_pixels=int(cv2.countNonZero(reveal_mask)), source_visible_pixels=int(cv2.countNonZero(source_visible)), patch_pixels=patch_pixels)
            if cv2.countNonZero(source_visible)==0:
                QMessageBox.warning(self,"没有可提交的中文文字","当前框选/画笔范围内没有可靠的 SOURCE 中文笔画。程序不会只清除日文。请缩小选框、调整 SOURCE X/Y 微调，或改用白色气泡模式。")
                self._trace("reveal_commit_blocked", reason="no_source_chinese")
                return
            if patch_pixels<=0:
                QMessageBox.warning(self,"Reveal 补丁为空","预览没有产生任何实际像素变化，因此不会保存空补丁。请调整选框、画笔或 X/Y 微调后重试。")
                self._trace("reveal_commit_blocked", reason="empty_patch")
                return
            self._reveal_mask=reveal_mask
            self._reveal_patch=reveal_patch
        # v1.2.0: commit from this button path itself.  The nested Reveal dialog
        # no longer returns to a parent dialog that must then be accepted a second
        # time before the workbench sees the patch.  A successful Save therefore
        # means the files were written, apply_review_page ran and the final image
        # was verified before both dialogs close.
        if self._commit_handler is not None:
            try:
                self._trace("direct_commit_started", row_id=str(self._row_payload().get("id", "")))
                self._commit_handler(self.result_row(), self.result_reveal_mask(), self.result_reveal_patch())
                self._committed_directly=True
                self._trace("direct_commit_succeeded", row_id=str(self._row_payload().get("id", "")))
                self.done(QDialog.DialogCode.Accepted)
            except Exception as exc:
                self._trace("direct_commit_failed", reason=str(exc))
                QMessageBox.critical(self,"人工补漏提交失败",str(exc))
            return
        self.accept()

    def result_row(self) -> dict[str, Any]:
        return self._row_payload()

    def result_reveal_mask(self):
        return None if self._reveal_mask is None else self._reveal_mask.copy()

    def result_reveal_patch(self):
        return None if self._reveal_patch is None else self._reveal_patch.copy()


class PagePreviewDialog(QDialog):
    """Large side-by-side source/target inspection used by the Page Manager."""

    def __init__(self, project_page: "ProjectPage"):
        super().__init__(project_page)
        self.project_page = project_page
        self._index = 0
        self.setWindowTitle("页面大图检查")
        self.setModal(False)
        _configure_responsive_dialog(self, (1320, 900), (820, 560))

        root = QVBoxLayout(self); root.setContentsMargins(14, 14, 14, 14); root.setSpacing(10)
        top = QHBoxLayout(); top.setSpacing(8)
        self.prev_btn = QPushButton("← 上一页"); self.prev_btn.setObjectName("pageNav")
        self.next_btn = QPushButton("下一页 →"); self.next_btn.setObjectName("pageNav")
        self.counter = QLabel("0 / 0"); self.counter.setObjectName("pageCounter")
        self.role_badge = QLabel("未分类"); self.role_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.role_badge.setMinimumWidth(150)
        top.addWidget(self.prev_btn); top.addWidget(self.counter); top.addWidget(self.next_btn); top.addSpacing(8)
        top.addWidget(self.role_badge); top.addStretch(1)
        for text, cb in [("适合窗口", self._fit), ("100%", self._actual), ("−", lambda: self._zoom(1/1.18)), ("+", lambda: self._zoom(1.18))]:
            b = QPushButton(text); b.clicked.connect(cb); top.addWidget(b)
        root.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QFrame(); left.setObjectName("card")
        ll = QVBoxLayout(left); ll.setContentsMargins(10,10,10,10); ll.setSpacing(7)
        self.source_title = QLabel("旧版中文"); self.source_title.setObjectName("sectionTitle")
        self.source_name = QLabel(); self.source_name.setObjectName("quiet"); self.source_name.setWordWrap(True)
        self.source_view = ZoomPreviewView()
        ll.addWidget(self.source_title); ll.addWidget(self.source_name); ll.addWidget(self.source_view, 1)
        right = QFrame(); right.setObjectName("card")
        rl = QVBoxLayout(right); rl.setContentsMargins(10,10,10,10); rl.setSpacing(7)
        self.target_title = QLabel("高清日文"); self.target_title.setObjectName("sectionTitle")
        self.target_name = QLabel(); self.target_name.setObjectName("quiet"); self.target_name.setWordWrap(True)
        self.target_view = ZoomPreviewView()
        rl.addWidget(self.target_title); rl.addWidget(self.target_name); rl.addWidget(self.target_view, 1)
        splitter.addWidget(left); splitter.addWidget(right); splitter.setSizes([650,650])
        root.addWidget(splitter, 1)

        bottom = QHBoxLayout(); bottom.setSpacing(8)
        bottom.addWidget(QLabel("页面类型"))
        self.role_combo = QComboBox(); self.role_combo.setMinimumWidth(210)
        for key in MANUAL_PAGE_TYPES:
            self.role_combo.addItem(page_type_label(key), key)
        self.apply_role = QPushButton("标记当前页"); self.apply_role.setObjectName("softPrimary")
        self.open_workbench = QPushButton("进入替换工作台")
        self.close_btn = QPushButton("关闭")
        bottom.addWidget(self.role_combo); bottom.addWidget(self.apply_role); bottom.addStretch(1)
        bottom.addWidget(self.open_workbench); bottom.addWidget(self.close_btn)
        root.addLayout(bottom)

        self.prev_btn.clicked.connect(lambda: self.set_index(self._index - 1))
        self.next_btn.clicked.connect(lambda: self.set_index(self._index + 1))
        self.apply_role.clicked.connect(self._apply_current_role)
        self.open_workbench.clicked.connect(self._go_workbench)
        self.close_btn.clicked.connect(self.close)

    def show_index(self, index: int):
        # Force the first load, then keep subsequent status/role refreshes cheap.
        self.set_index(index, force_reload=True)
        self.show(); self.raise_(); self.activateWindow()

    def _sync_role_ui(self):
        pairs = self.project_page.window.state.pairs
        if not (0 <= self._index < len(pairs)):
            return
        mark = self.project_page.window.page_mark_for_pair(pairs[self._index])
        color = page_type_color(mark.page_type)
        origin = "手动" if mark.origin == "manual" else "默认"
        self.role_badge.setText(f"{mark.label} · {origin}")
        self.role_badge.setToolTip(mark.reason or PAGE_TYPE_INFO.get(mark.page_type, {}).get("description", ""))
        self.role_badge.setStyleSheet(f"background:{color};color:white;border-radius:10px;padding:5px 10px;font-weight:700;")
        combo_index = self.role_combo.findData(mark.page_type)
        if combo_index < 0 and mark.page_type == "auto_no_text":
            combo_index = self.role_combo.findData("illustration")
        if combo_index >= 0 and self.role_combo.currentIndex() != combo_index:
            self.role_combo.blockSignals(True)
            try: self.role_combo.setCurrentIndex(combo_index)
            finally: self.role_combo.blockSignals(False)

    def set_index(self, index: int, *, force_reload: bool = False):
        pairs = self.project_page.window.state.pairs
        if not pairs:
            return
        new_index = max(0, min(int(index), len(pairs)-1))
        changed = new_index != self._index
        self._index = new_index
        pair = pairs[self._index]
        self.counter.setText(f"{self._index + 1} / {len(pairs)}")
        self.prev_btn.setEnabled(self._index > 0); self.next_btn.setEnabled(self._index + 1 < len(pairs))
        self._sync_role_ui()
        if changed or force_reload:
            self.source_name.setText(Path(pair.source_path).name)
            self.target_name.setText(Path(pair.target_path).name)
            self.source_name.setToolTip(pair.source_path); self.target_name.setToolTip(pair.target_path)
            self.source_view.set_image(pair.source_path); self.target_view.set_image(pair.target_path)
        if self.project_page.window.state.selected_index != self._index:
            self.project_page.window.set_selected_page(self._index)

    def refresh_current(self):
        # Progress updates should not decode two full-resolution pages again.
        self._sync_role_ui()

    def _apply_current_role(self):
        if self.project_page.window._busy_running():
            QMessageBox.information(self, "任务进行中", "请先停止或等待当前任务完成，再修改页面类型。")
            return
        self.project_page.window.mark_page_rows([self._index], str(self.role_combo.currentData() or "content"))
        self._sync_role_ui()

    def _go_workbench(self):
        self.project_page.window.set_selected_page(self._index)
        self.project_page.window.show_page(2)

    def _fit(self):
        self.source_view.fit_to_window(); self.target_view.fit_to_window()

    def _actual(self):
        self.source_view.actual_size(); self.target_view.actual_size()

    def _zoom(self, factor: float):
        self.source_view.zoom_by(factor); self.target_view.zoom_by(factor)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_PageUp):
            self.set_index(self._index - 1); return
        if key in (Qt.Key.Key_Right, Qt.Key.Key_PageDown):
            self.set_index(self._index + 1); return
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._zoom(1.18); return
        if key == Qt.Key.Key_Minus:
            self._zoom(1/1.18); return
        if key == Qt.Key.Key_0:
            self._actual(); return
        super().keyPressEvent(event)


class ProjectPage(QWidget):
    """Visual Page Manager + pairing workspace.

    The default view is deliberately thumbnail-first: classification happens from
    the page image, not from filenames.  A list/table view remains available for
    exact pairing diagnostics.  Pairing answers *which pages correspond* while
    Page Manager decides *whether the pair enters transfer processing*.
    """

    THUMB_SIZE = QSize(156, 214)
    THUMB_CANVAS = QSize(166, 226)

    def __init__(self, window: "StudioWindow"):
        super().__init__(); self.window = window
        self._table_signature = None
        self._thumb_signature = None
        self._thumb_generation = 0
        self._thumb_queue: list[int] = []
        self._thumb_items: dict[int, QListWidgetItem] = {}
        self._preview_dialog: PagePreviewDialog | None = None
        self._detail_side = "target"
        self._thumb_side = "target"
        # Scrolling must never compete with eager image decoding. Keep only a
        # bounded cache of already-scaled thumbnail pixels and load visible cards
        # after scrolling settles.
        self._thumb_image_cache: OrderedDict[tuple[str, int, int, int, int], QPixmap] = OrderedDict()
        self._thumb_image_cache_limit = 128
        self._thumb_loaded: set[int] = set()
        self._thumb_load_timer = QTimer(self); self._thumb_load_timer.setSingleShot(True)
        self._thumb_load_timer.timeout.connect(self._pump_thumbnails)

        root = QHBoxLayout(self); root.setContentsMargins(18,18,18,18); root.setSpacing(14)

        left = Card("页面管理", "默认所有配对页面都是正文。先看缩略图再按需手动分类；双击可并排放大旧中文与高清日文。Ctrl / Command / Shift 可多选批量标记。")
        self.source = PathRow("旧中文版", "选择目录")
        self.target = PathRow("高清日文版", "选择目录")
        self.output = PathRow("输出目录", "选择目录")
        left.layout.addWidget(self.source); left.layout.addWidget(self.target); left.layout.addWidget(self.output)

        pair_opts = QGridLayout(); pair_opts.setContentsMargins(0, 2, 0, 0)
        pair_opts.setHorizontalSpacing(34); pair_opts.setVerticalSpacing(6)
        self.prefer_name_pair = QCheckBox("优先名称 / 页码配对"); self.prefer_name_pair.setChecked(False)
        self.prefer_name_pair.setToolTip("可选加速：先锁定同名、仅扩展名不同、以及唯一页码相同的页面。默认关闭。")
        self.prefer_order_pair = QCheckBox("优先文件夹自然顺序"); self.prefer_order_pair.setChecked(False)
        self.prefer_order_pair.setToolTip("可选加速：对等长区间按自然排序一一对应。默认关闭，避免插页/缺页造成连锁错位。")
        pair_opts.addWidget(self.prefer_name_pair, 0, 0); pair_opts.addWidget(self.prefer_order_pair, 0, 1); pair_opts.setColumnStretch(2, 1)
        left.layout.addLayout(pair_opts)

        pair_row = QHBoxLayout(); pair_row.setSpacing(8)
        self.pair_btn = QPushButton("智能配对"); self.pair_btn.setObjectName("primary")
        self.refresh_btn = QPushButton("刷新")
        pair_row.addWidget(self.pair_btn); pair_row.addWidget(self.refresh_btn); pair_row.addStretch(1)
        left.layout.addLayout(pair_row)

        manager = QFrame(); manager.setObjectName("cardBlue")
        ml = QHBoxLayout(manager); ml.setContentsMargins(10,8,10,8); ml.setSpacing(8)
        cap = QLabel("页面类型"); cap.setStyleSheet("font-weight:600;")
        self.page_type = QComboBox(); self.page_type.setMinimumWidth(190)
        for key in MANUAL_PAGE_TYPES:
            self.page_type.addItem(page_type_label(key), key)
        self.apply_type = QPushButton("标记所选"); self.apply_type.setObjectName("softPrimary")
        self.reset_type = QPushButton("恢复正文")
        self.reset_type.setToolTip("删除手动页面类型标记，恢复默认“正文 / 需替换”。实际处理时若中文版没有气泡/文本框，会自动原样保留高清日文页。")
        ml.addWidget(cap); ml.addWidget(self.page_type); ml.addWidget(self.apply_type); ml.addWidget(self.reset_type); ml.addStretch(1)
        left.layout.addWidget(manager)

        viewbar = QHBoxLayout(); viewbar.setSpacing(6)
        self.thumb_view_btn = QPushButton("▦ 缩略图"); self.thumb_view_btn.setObjectName("segmented"); self.thumb_view_btn.setCheckable(True); self.thumb_view_btn.setChecked(True)
        self.list_view_btn = QPushButton("☷ 列表"); self.list_view_btn.setObjectName("segmented"); self.list_view_btn.setCheckable(True)
        self.view_group = QButtonGroup(self); self.view_group.setExclusive(True); self.view_group.addButton(self.thumb_view_btn); self.view_group.addButton(self.list_view_btn)
        self.filter_combo = QComboBox(); self.filter_combo.setMinimumWidth(120)
        for text, data in [("全部页面","all"),("需处理","process"),("已跳过","skip"),("手动标记","manual"),("处理失败","failed")]:
            self.filter_combo.addItem(text, data)
        self.visible_count = QLabel("0 页"); self.visible_count.setObjectName("quiet")
        self.thumb_target_btn = QPushButton("日文缩略图"); self.thumb_target_btn.setObjectName("segmented"); self.thumb_target_btn.setCheckable(True); self.thumb_target_btn.setChecked(True)
        self.thumb_source_btn = QPushButton("中文缩略图"); self.thumb_source_btn.setObjectName("segmented"); self.thumb_source_btn.setCheckable(True)
        self.thumb_side_group = QButtonGroup(self); self.thumb_side_group.setExclusive(True); self.thumb_side_group.addButton(self.thumb_target_btn); self.thumb_side_group.addButton(self.thumb_source_btn)
        viewbar.addWidget(self.thumb_view_btn); viewbar.addWidget(self.list_view_btn); viewbar.addSpacing(10)
        viewbar.addWidget(QLabel("缩略图")); viewbar.addWidget(self.thumb_target_btn); viewbar.addWidget(self.thumb_source_btn); viewbar.addSpacing(8)
        viewbar.addWidget(QLabel("筛选")); viewbar.addWidget(self.filter_combo); viewbar.addWidget(self.visible_count); viewbar.addStretch(1)
        left.layout.addLayout(viewbar)
        hint = QLabel("双击：并排放大 · 右键：快速标记 · 多选：批量分类")
        hint.setObjectName("quiet"); hint.setWordWrap(True); left.layout.addWidget(hint)

        self.view_stack = QStackedWidget()
        gallery_page = QWidget(); gallery_l = QHBoxLayout(gallery_page); gallery_l.setContentsMargins(0,0,0,0); gallery_l.setSpacing(12)
        self.thumb_list = QListWidget(); self.thumb_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumb_list.setIconSize(self.THUMB_CANVAS); self.thumb_list.setGridSize(QSize(202, 292))
        self.thumb_list.setResizeMode(QListWidget.ResizeMode.Adjust); self.thumb_list.setMovement(QListWidget.Movement.Static)
        self.thumb_list.setWrapping(True); self.thumb_list.setSpacing(8)
        self.thumb_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.thumb_list.setUniformItemSizes(True)
        self.thumb_list.setMinimumWidth(280)
        self.thumb_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.thumb_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.thumb_list.verticalScrollBar().valueChanged.connect(lambda _v: self._schedule_visible_thumbnails(80))
        self.thumb_list.horizontalScrollBar().valueChanged.connect(lambda _v: self._schedule_visible_thumbnails(80))
        gallery_l.addWidget(self.thumb_list, 7)

        detail = QFrame(); detail.setObjectName("cardBlue"); detail.setMinimumWidth(250); detail.setMaximumWidth(390)
        dl = QVBoxLayout(detail); dl.setContentsMargins(12,12,12,12); dl.setSpacing(8)
        dhead = QHBoxLayout(); self.detail_page = QLabel("未选择页面"); self.detail_page.setObjectName("sectionTitle")
        self.detail_badge = QLabel("—"); self.detail_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dhead.addWidget(self.detail_page); dhead.addStretch(1); dhead.addWidget(self.detail_badge); dl.addLayout(dhead)
        side = QHBoxLayout(); self.detail_target_btn = QPushButton("高清日文"); self.detail_target_btn.setObjectName("segmented"); self.detail_target_btn.setCheckable(True); self.detail_target_btn.setChecked(True)
        self.detail_source_btn = QPushButton("旧版中文"); self.detail_source_btn.setObjectName("segmented"); self.detail_source_btn.setCheckable(True)
        self.detail_side_group = QButtonGroup(self); self.detail_side_group.setExclusive(True); self.detail_side_group.addButton(self.detail_target_btn); self.detail_side_group.addButton(self.detail_source_btn)
        side.addWidget(self.detail_target_btn); side.addWidget(self.detail_source_btn); side.addStretch(1); dl.addLayout(side)
        self.detail_view = ImageView(); self.detail_view.setMinimumHeight(220); dl.addWidget(self.detail_view, 1)
        self.detail_names = QLabel(); self.detail_names.setObjectName("quiet"); self.detail_names.setWordWrap(True); dl.addWidget(self.detail_names)
        self.detail_reason = QLabel(); self.detail_reason.setObjectName("hint"); self.detail_reason.setWordWrap(True); dl.addWidget(self.detail_reason)
        self.detail_stats = QLabel(); self.detail_stats.setObjectName("quiet"); self.detail_stats.setWordWrap(True); dl.addWidget(self.detail_stats)
        detail_actions = QHBoxLayout(); self.open_preview_btn = QPushButton("双击放大 / 打开大图"); self.open_preview_btn.setObjectName("softPrimary")
        self.go_workbench_btn = QPushButton("替换工作台")
        detail_actions.addWidget(self.open_preview_btn); detail_actions.addWidget(self.go_workbench_btn); dl.addLayout(detail_actions)
        gallery_l.addWidget(detail, 3)
        self.view_stack.addWidget(gallery_page)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["页", "处理", "页面类型", "旧中文", "高清日文", "配对", "状态", "路线"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents); hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents); hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch); hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents); hh.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents); hh.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows); self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().setVisible(False); self.table.setAlternatingRowColors(False)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view_stack.addWidget(self.table)
        left.layout.addWidget(self.view_stack, 1)
        root.addWidget(left, 7)

        right_col = QVBoxLayout(); right_col.setSpacing(14)
        mode = Card("迁移方式", "页面默认都是正文；手动跳过页仍原样输出。正文页若中文版实际没有可迁移气泡/文本框，也只保留高清日文原页，不强行替换。")
        self.mode = QComboBox(); self.mode.addItem("自动 · Direct 优先", "auto"); self.mode.addItem("直接贴图 · 只迁移文字（保留彩图背景）", "direct_patch"); self.mode.addItem("精准蒙版迁移", "mask_replace"); self.mode.addItem("智能混合", "hybrid"); self.mode.addItem("高清重排", "reletter")
        mode.layout.addWidget(self.mode)
        self.show_experimental = QCheckBox("显示并启用实验模式")
        self.experimental_warning = QLabel("实验模式已启用：整页对齐后挖除日文以显示中文。勾选后会自动切换到该迁移方式；若之后手动改回 Auto，则不会运行实验路线。对齐误差和彩色露底风险较高，结果默认进入复核。")
        self.experimental_warning.setObjectName("hint"); self.experimental_warning.setWordWrap(True); self.experimental_warning.setVisible(False)
        mode.layout.addWidget(self.show_experimental); mode.layout.addWidget(self.experimental_warning)
        self.diff_check = QCheckBox("优先使用成对差异提取中文气泡/文本框"); self.diff_check.setChecked(True)
        self.exact_check = QCheckBox("同源页面启用像素级精确覆盖"); self.exact_check.setChecked(True)
        mode.layout.addWidget(self.diff_check); mode.layout.addWidget(self.exact_check); right_col.addWidget(mode)

        summary = Card("当前项目")
        self.sum_pairs = QLabel("0 页"); self.sum_pairs.setStyleSheet(f"font-size:18px;font-weight:700;color:{ACCENT_HOVER};")
        self.sum_hint = QLabel("等待页面配对"); self.sum_hint.setObjectName("hint"); self.sum_hint.setWordWrap(True)
        summary.layout.addWidget(self.sum_pairs); summary.layout.addWidget(self.sum_hint); right_col.addWidget(summary)

        actions = Card("开始处理", "“停止处理”采用安全停止：当前不可中断的底层图像调用返回后立即终止，不破坏已完成页面。")
        self.resume_check = QCheckBox("断点续跑 / 跳过已完成页面"); self.resume_check.setChecked(True)
        self.cache_check = QCheckBox("复用配准、OCR 与气泡缓存"); self.cache_check.setChecked(True)
        self.run_page = QPushButton("处理当前页"); self.run_page.setObjectName("softPrimary")
        self.run_book = QPushButton("处理整册"); self.run_book.setObjectName("primary")
        self.cancel = QPushButton("停止处理"); self.cancel.setObjectName("danger"); self.cancel.setEnabled(False)
        actions.layout.addWidget(self.resume_check); actions.layout.addWidget(self.cache_check); actions.layout.addWidget(self.run_page); actions.layout.addWidget(self.run_book); actions.layout.addWidget(self.cancel)
        right_col.addWidget(actions); right_col.addStretch(1); root.addLayout(right_col, 3)

        self.source.button.clicked.connect(lambda: self.window.choose_directory("source")); self.target.button.clicked.connect(lambda: self.window.choose_directory("target")); self.output.button.clicked.connect(lambda: self.window.choose_directory("output"))
        self.pair_btn.clicked.connect(self.window.auto_pair); self.refresh_btn.clicked.connect(self._force_thumbnail_refresh)
        self.table.itemSelectionChanged.connect(self._table_selection_changed); self.table.cellDoubleClicked.connect(lambda row, _col: self.open_preview(row))
        self.thumb_list.itemSelectionChanged.connect(self._thumb_selection_changed); self.thumb_list.currentItemChanged.connect(lambda *_: self._sync_detail_from_current())
        self.thumb_list.itemDoubleClicked.connect(lambda item: self.open_preview(int(item.data(Qt.ItemDataRole.UserRole))))
        self.thumb_list.customContextMenuRequested.connect(self._thumb_context_menu)
        self.table.customContextMenuRequested.connect(self._table_context_menu)
        self.thumb_view_btn.clicked.connect(lambda: self._set_view_mode(0)); self.list_view_btn.clicked.connect(lambda: self._set_view_mode(1))
        self.thumb_target_btn.clicked.connect(lambda: self._set_thumb_side("target")); self.thumb_source_btn.clicked.connect(lambda: self._set_thumb_side("source"))
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)
        self.detail_target_btn.clicked.connect(lambda: self._set_detail_side("target")); self.detail_source_btn.clicked.connect(lambda: self._set_detail_side("source"))
        self.open_preview_btn.clicked.connect(lambda: self.open_preview(self.window.state.selected_index)); self.go_workbench_btn.clicked.connect(self._go_workbench)
        self.mode.currentIndexChanged.connect(self._sync_config); self.show_experimental.toggled.connect(self._set_experimental_visible); self.diff_check.toggled.connect(self._sync_config); self.exact_check.toggled.connect(self._sync_config)
        self.prefer_name_pair.toggled.connect(self._sync_config); self.prefer_order_pair.toggled.connect(self._sync_config)
        self.run_page.clicked.connect(self.window.run_current_page); self.run_book.clicked.connect(self.window.run_book); self.cancel.clicked.connect(self.window.cancel_worker)
        self.resume_check.toggled.connect(self._sync_config); self.cache_check.toggled.connect(self._sync_config)
        self.apply_type.clicked.connect(self._apply_selected_type); self.reset_type.clicked.connect(self._reset_selected_type)

    def _set_experimental_visible(self, visible: bool):
        idx = self.mode.findData("aligned_overlay_reveal")
        if visible:
            if idx < 0:
                self.mode.addItem("实验 · 整页对齐挖洞显中文", "aligned_overlay_reveal")
                idx = self.mode.findData("aligned_overlay_reveal")
            # Avoid the v1.2.2 UX trap where users ticked "显示实验模式"
            # but remained on Auto (Auto intentionally cannot select this route).
            # The explicit checkbox is now treated as explicit opt-in and switches
            # the selector to the experimental route immediately.
            if idx >= 0 and self.mode.currentData() != "aligned_overlay_reveal":
                self.mode.setCurrentIndex(idx)
        elif idx >= 0:
            if self.mode.currentData() == "aligned_overlay_reveal":
                auto_idx = self.mode.findData("auto")
                if auto_idx >= 0:
                    self.mode.setCurrentIndex(auto_idx)
            idx = self.mode.findData("aligned_overlay_reveal")
            if idx >= 0:
                self.mode.removeItem(idx)
        self.experimental_warning.setVisible(bool(visible))
        self._sync_config()

    def _sync_config(self):
        cfg = self.window.state.config
        cfg.transfer.mode = self.mode.currentData(); cfg.mask_replace.paired_diff_enabled = self.diff_check.isChecked(); cfg.mask_replace.exact_identity_copy = self.exact_check.isChecked(); cfg.direct_patch.exact_identity_copy = self.exact_check.isChecked()
        # Merely exposing the experimental selector never opts Auto into this route.
        # Auto still needs allow_in_auto=true and require_explicit_mode=false.
        cfg.aligned_overlay_reveal.enabled = self.show_experimental.isChecked()
        cfg.pairing.prefer_name_pairing = self.prefer_name_pair.isChecked(); cfg.pairing.prefer_order_pairing = self.prefer_order_pair.isChecked()
        cfg.batch.resume = self.resume_check.isChecked(); cfg.batch.skip_completed = self.resume_check.isChecked(); cfg.cache.enabled = self.cache_check.isChecked()

    def _set_view_mode(self, index: int):
        self.view_stack.setCurrentIndex(index)
        self.thumb_view_btn.setChecked(index == 0); self.list_view_btn.setChecked(index == 1)
        self.set_current_page(self.window.state.selected_index)

    def _set_detail_side(self, side: str):
        self._detail_side = "source" if side == "source" else "target"
        self.detail_source_btn.setChecked(self._detail_side == "source"); self.detail_target_btn.setChecked(self._detail_side == "target")
        self._sync_detail_from_current()

    def _set_thumb_side(self, side: str):
        side = "source" if side == "source" else "target"
        if side == self._thumb_side:
            return
        self._thumb_side = side
        self.thumb_source_btn.setChecked(side == "source"); self.thumb_target_btn.setChecked(side == "target")
        self._thumb_signature = None
        self.refresh()

    def _show_page_context_menu(self, global_pos, rows: list[int]):
        if not rows:
            return
        menu = QMenu(self)
        title = menu.addAction(f"已选 {len(rows)} 页")
        title.setEnabled(False)
        menu.addSeparator()
        for key in MANUAL_PAGE_TYPES:
            action = menu.addAction(page_type_label(key))
            action.triggered.connect(lambda _checked=False, k=key, r=tuple(rows): self.window.mark_page_rows(list(r), k))
        menu.addSeparator()
        reset_action = menu.addAction("恢复默认正文")
        reset_action.triggered.connect(lambda _checked=False, r=tuple(rows): self.window.reset_page_rows(list(r)))
        if len(rows) == 1:
            menu.addSeparator()
            preview_action = menu.addAction("并排打开旧中文 / 高清日文")
            preview_action.triggered.connect(lambda _checked=False, row=rows[0]: self.open_preview(row))
            workbench_action = menu.addAction("进入替换工作台")
            workbench_action.triggered.connect(lambda _checked=False, row=rows[0]: (self.window.set_selected_page(row), self.window.show_page(2)))
        menu.exec(global_pos)

    def _thumb_context_menu(self, pos):
        item = self.thumb_list.itemAt(pos)
        if item is not None and not item.isSelected():
            self.thumb_list.clearSelection(); item.setSelected(True); self.thumb_list.setCurrentItem(item)
        rows = self._selected_thumb_rows()
        self._show_page_context_menu(self.thumb_list.viewport().mapToGlobal(pos), rows)

    def _table_context_menu(self, pos):
        item = self.table.itemAt(pos)
        if item is not None and not self.table.item(item.row(), 0).isSelected():
            self.table.clearSelection(); self.table.selectRow(item.row()); self.table.setCurrentCell(item.row(), 0)
        rows = self._selected_table_rows()
        self._show_page_context_menu(self.table.viewport().mapToGlobal(pos), rows)

    def _selected_table_rows(self) -> list[int]:
        model = self.table.selectionModel()
        return sorted({idx.row() for idx in model.selectedRows()}) if model is not None else []

    def _selected_thumb_rows(self) -> list[int]:
        result = []
        for item in self.thumb_list.selectedItems():
            try: result.append(int(item.data(Qt.ItemDataRole.UserRole)))
            except Exception: pass
        return sorted(set(result))

    def _selected_rows(self) -> list[int]:
        rows = self._selected_thumb_rows() if self.view_stack.currentIndex() == 0 else self._selected_table_rows()
        if not rows and self.window.state.pairs:
            rows = [max(0, min(self.window.state.selected_index, len(self.window.state.pairs)-1))]
        return rows

    def _apply_selected_type(self):
        rows = self._selected_rows()
        if rows: self.window.mark_page_rows(rows, str(self.page_type.currentData() or "content"))

    def _reset_selected_type(self):
        rows = self._selected_rows()
        if rows: self.window.reset_page_rows(rows)

    def _table_selection_changed(self):
        rows = self._selected_table_rows()
        if rows and self.view_stack.currentIndex() == 1:
            self.window.set_selected_page(self.table.currentRow() if self.table.currentRow() >= 0 else rows[0], sync_table=False)
        self._apply_filter()

    def _thumb_selection_changed(self):
        if self.view_stack.currentIndex() != 0:
            return
        item = self.thumb_list.currentItem()
        if item is None:
            selected = self.thumb_list.selectedItems(); item = selected[0] if selected else None
        if item is not None:
            self.window.set_selected_page(int(item.data(Qt.ItemDataRole.UserRole)), sync_table=True)
        self._apply_filter()

    def set_current_page(self, index: int, *, sync_table: bool = True):
        if not self.window.state.pairs:
            self._sync_detail_from_current(); return
        idx = max(0, min(int(index), len(self.window.state.pairs)-1))
        if sync_table and self.table.rowCount() > idx and self.table.currentRow() != idx:
            self.table.blockSignals(True)
            try: self.table.selectRow(idx)
            finally: self.table.blockSignals(False)
        item = self._thumb_items.get(idx)
        if item is not None and self.thumb_list.currentItem() is not item:
            self.thumb_list.blockSignals(True)
            try:
                self.thumb_list.setCurrentItem(item)
                item.setSelected(True)
                self.thumb_list.scrollToItem(item, QAbstractItemView.ScrollHint.EnsureVisible)
            finally: self.thumb_list.blockSignals(False)
        self._sync_detail(idx)

    def _sync_detail_from_current(self):
        item = self.thumb_list.currentItem()
        if item is not None:
            try: self._sync_detail(int(item.data(Qt.ItemDataRole.UserRole))); return
            except Exception: pass
        self._sync_detail(self.window.state.selected_index if self.window.state.pairs else -1)

    def _sync_detail(self, index: int):
        pairs = self.window.state.pairs
        if not (0 <= index < len(pairs)):
            self.detail_page.setText("未选择页面"); self.detail_badge.setText("—"); self.detail_badge.setStyleSheet("")
            self.detail_view.set_image(None); self.detail_names.setText(""); self.detail_reason.setText(""); self.detail_stats.setText(""); return
        pair = pairs[index]; mark = self.window.page_mark_for_pair(pair)
        self.detail_page.setText(f"第 {index+1} 页")
        self.detail_badge.setText(mark.label)
        color = page_type_color(mark.page_type)
        self.detail_badge.setStyleSheet(f"background:{color};color:white;border-radius:9px;padding:4px 8px;font-weight:700;")
        path = pair.source_path if self._detail_side == "source" else pair.target_path
        self.detail_view.set_image(path)
        self.detail_names.setText(f"旧中文：{Path(pair.source_path).name}\n高清日文：{Path(pair.target_path).name}")
        origin = "手动" if mark.origin == "manual" else "默认"
        description = PAGE_TYPE_INFO.get(mark.page_type, PAGE_TYPE_INFO["content"]).get("description", "")
        self.detail_reason.setText(f"{origin} · {description}\n{mark.reason or '尚未进行自动页面检查'}")
        method = {"name":"名称", "order":"顺序", "smart":"智能"}.get(pairing_method(pair), pairing_method(pair))
        self.detail_stats.setText(f"配对：{method} · {pair.confidence:.3f}　页面：{'手动分类' if mark.origin == 'manual' else '默认正文'}")
        combo_index = self.page_type.findData(mark.page_type)
        if combo_index >= 0: self.page_type.setCurrentIndex(combo_index)

    def _go_workbench(self):
        if self.window.state.pairs:
            self.window.set_selected_page(self.window.state.selected_index); self.window.show_page(2)

    def open_preview(self, index: int):
        if not self.window.state.pairs:
            return
        if self._preview_dialog is None:
            self._preview_dialog = PagePreviewDialog(self)
        self._preview_dialog.show_index(index)

    def _thumbnail_base_pixmap(self, path: Path) -> QPixmap | None:
        try:
            st = path.stat()
        except OSError:
            return None
        key = (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size), self.THUMB_SIZE.width(), self.THUMB_SIZE.height())
        cached = self._thumb_image_cache.pop(key, None)
        if cached is not None:
            self._thumb_image_cache[key] = cached
            return cached
        reader = QImageReader(str(path)); reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid():
            reader.setScaledSize(size.scaled(self.THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio))
        image = reader.read()
        if image.isNull():
            return None
        pix = QPixmap.fromImage(image)
        self._thumb_image_cache[key] = pix
        while len(self._thumb_image_cache) > self._thumb_image_cache_limit:
            self._thumb_image_cache.popitem(last=False)
        return pix

    def _thumbnail_icon(self, index: int) -> QIcon:
        pair = self.window.state.pairs[index]; mark = self.window.page_mark_for_pair(pair)
        path = Path(pair.source_path if self._thumb_side == "source" else pair.target_path)
        canvas = QPixmap(self.THUMB_CANVAS); canvas.fill(QColor("#F1F4F8"))
        painter = QPainter(canvas)
        try:
            pix = self._thumbnail_base_pixmap(path)
            if pix is not None and not pix.isNull():
                x = (canvas.width() - pix.width()) // 2
                y = 8 + max(0, (self.THUMB_SIZE.height() - pix.height()) // 2)
                painter.drawPixmap(x, y, pix)
            color = QColor(page_type_color(mark.page_type))
            painter.fillRect(0, 0, canvas.width(), 7, color)
            painter.fillRect(0, canvas.height()-28, canvas.width(), 28, color)
            painter.setPen(QColor("white")); f = painter.font(); f.setPointSize(9); f.setBold(True); painter.setFont(f)
            painter.drawText(6, canvas.height()-23, canvas.width()-12, 20, Qt.AlignmentFlag.AlignCenter, mark.label)
            painter.setPen(QColor(BORDER_STRONG)); painter.drawRect(0,0,canvas.width()-1,canvas.height()-1)
        finally:
            painter.end()
        return QIcon(canvas)

    def _placeholder_icon(self, index: int) -> QIcon:
        canvas = QPixmap(self.THUMB_CANVAS); canvas.fill(QColor("#F1F4F8")); painter = QPainter(canvas)
        try:
            painter.setPen(QColor(MUTED_2)); f=painter.font(); f.setPointSize(11); painter.setFont(f)
            painter.drawText(canvas.rect(), Qt.AlignmentFlag.AlignCenter, f"第 {index+1} 页\n加载中…")
        finally: painter.end()
        return QIcon(canvas)

    def _rebuild_thumbnails(self):
        selected = set(self._selected_thumb_rows()); current = self.window.state.selected_index
        self._thumb_generation += 1
        self._thumb_load_timer.stop(); self._thumb_queue = []; self._thumb_loaded.clear(); self._thumb_items.clear()
        self.thumb_list.setUpdatesEnabled(False)
        try:
            self.thumb_list.clear()
            for i, pair in enumerate(self.window.state.pairs):
                mark = self.window.page_mark_for_pair(pair)
                origin = "手动" if mark.origin == "manual" else "默认"
                item = QListWidgetItem(self._placeholder_icon(i), f"第 {i+1} 页\n{mark.label} · {origin}")
                item.setData(Qt.ItemDataRole.UserRole, i)
                action_text = "处理" if mark.should_process else "跳过"
                item.setToolTip(f"高清日文：{Path(pair.target_path).name}\n旧版中文：{Path(pair.source_path).name}\n{mark.label} · {origin} · {action_text}\n双击并排放大；右键快速标记")
                item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
                self.thumb_list.addItem(item); self._thumb_items[i] = item
        finally:
            self.thumb_list.setUpdatesEnabled(True)
        for idx in selected:
            if idx in self._thumb_items: self._thumb_items[idx].setSelected(True)
        self.set_current_page(current)
        self._apply_filter()
        self._schedule_visible_thumbnails(0)

    def _visible_thumbnail_indices(self) -> list[int]:
        if self.view_stack.currentIndex() != 0 or not self._thumb_items:
            return []
        viewport_rect = self.thumb_list.viewport().rect()
        # One extra row above/below avoids blank flashes on short wheel movements
        # without resuming the old full-book eager decode.
        margin = max(80, self.thumb_list.gridSize().height())
        wanted = viewport_rect.adjusted(0, -margin, 0, margin)
        out: list[int] = []
        for idx, item in self._thumb_items.items():
            if item.isHidden() or idx in self._thumb_loaded:
                continue
            rect = self.thumb_list.visualItemRect(item)
            if rect.isValid() and rect.intersects(wanted):
                out.append(idx)
        current = self.window.state.selected_index
        if current in self._thumb_items and current not in self._thumb_loaded and not self._thumb_items[current].isHidden():
            if current not in out:
                out.insert(0, current)
        return out

    def _schedule_visible_thumbnails(self, delay_ms: int = 60):
        if self.view_stack.currentIndex() != 0:
            return
        self._thumb_load_timer.start(max(0, int(delay_ms)))

    def _pump_thumbnails(self):
        indices = self._visible_thumbnail_indices()
        if not indices:
            return
        # Decoding scaled images is still synchronous in Qt. A tiny batch keeps
        # each event-loop slice short; wheel/trackpad scrolling restarts the timer
        # and therefore takes priority over thumbnail work.
        for idx in indices[:3]:
            item = self._thumb_items.get(idx)
            if item is None or idx >= len(self.window.state.pairs):
                continue
            try:
                item.setIcon(self._thumbnail_icon(idx))
            except Exception:
                item.setIcon(self._placeholder_icon(idx))
            self._thumb_loaded.add(idx)
        if len(indices) > 3:
            self._thumb_load_timer.start(18)

    def _force_thumbnail_refresh(self):
        self._thumb_image_cache.clear(); self._thumb_loaded.clear()
        self._thumb_signature = None; self._table_signature = None; self.refresh()

    def _filter_accepts(self, index: int) -> bool:
        if not (0 <= index < len(self.window.state.pairs)): return False
        kind = str(self.filter_combo.currentData() or "all"); pair = self.window.state.pairs[index]; mark = self.window.page_mark_for_pair(pair)
        if kind == "all": return True
        if kind == "process": return mark.should_process
        if kind == "skip": return not mark.should_process
        if kind == "manual": return mark.origin == "manual"
        if kind == "failed": return self.window.state.batch_status.get(Path(pair.target_path).name, ("", ""))[0] == "失败"
        return True

    def _apply_filter(self):
        visible = 0
        for idx, item in self._thumb_items.items():
            show = self._filter_accepts(idx); item.setHidden(not show); visible += int(show)
        # Keep list view consistent with the thumbnail filter.
        for row in range(self.table.rowCount()):
            show = self._filter_accepts(row); self.table.setRowHidden(row, not show)
        selected = len(self._selected_thumb_rows()) if self.view_stack.currentIndex() == 0 else len(self._selected_table_rows())
        suffix = f" · 已选 {selected}" if selected else ""
        self.visible_count.setText(f"可见 {visible} / {len(self.window.state.pairs)}{suffix}")
        self._schedule_visible_thumbnails(20)

    def refresh(self):
        s = self.window.state
        self.source.set_path(s.source_dir); self.target.set_path(s.target_dir); self.output.set_path(s.output_dir)
        mark_sig = tuple(sorted((str(k), str(v.get("page_type","")), str(v.get("origin","")), str(v.get("reason","")), int(v.get("bubble_regions",0) or 0), int(v.get("free_text_regions",0) or 0)) for k,v in s.page_marks.items()))
        pair_sig = tuple((str(p.source_path), str(p.target_path), round(float(p.confidence), 6), str(pairing_method(p))) for p in s.pairs)
        table_signature = (pair_sig, mark_sig, tuple(sorted((str(k), str(v[0]), str(v[1])) for k, v in s.batch_status.items())))
        if table_signature != self._table_signature:
            selected = set(self._selected_table_rows())
            self.table.setUpdatesEnabled(False)
            try:
                self.table.setRowCount(len(s.pairs))
                for i, pair in enumerate(s.pairs):
                    key = Path(pair.target_path).name; state, route = s.batch_status.get(key, ("等待", "—")); method = {"name":"名称", "order":"顺序", "smart":"智能"}.get(pairing_method(pair), pairing_method(pair))
                    mark = self.window.page_mark_for_pair(pair); origin = "手动" if mark.origin == "manual" else "默认"; action = "✓ 处理" if mark.should_process else "— 跳过"
                    vals = [str(i+1), action, f"{mark.label} · {origin}", Path(pair.source_path).name, key, f"{method} · {pair.confidence:.3f}", state, route]
                    for c, v in enumerate(vals):
                        it = QTableWidgetItem(v); it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        if c == 2:
                            color = QColor(page_type_color(mark.page_type)); it.setBackground(color.lighter(185)); it.setToolTip(mark.reason or PAGE_TYPE_INFO.get(mark.page_type, {}).get("description", ""))
                        if c == 1 and not mark.should_process: it.setForeground(QColor(MUTED))
                        self.table.setItem(i, c, it)
                for row in selected:
                    if 0 <= row < self.table.rowCount(): self.table.selectRow(row)
            finally: self.table.setUpdatesEnabled(True)
            self._table_signature = table_signature
        thumb_signature = (pair_sig, mark_sig, self._thumb_side)
        if thumb_signature != self._thumb_signature:
            self._thumb_signature = thumb_signature; self._rebuild_thumbnails()
        self.set_current_page(s.selected_index)
        self._apply_filter()
        process_count = sum(1 for pair in s.pairs if self.window.page_mark_for_pair(pair).should_process); skip_count = len(s.pairs)-process_count
        manual_count = sum(1 for pair in s.pairs if self.window.page_mark_for_pair(pair).origin == "manual"); unmatched_count = len(s.unmatched_source)+len(s.unmatched_target)
        self.sum_pairs.setText(f"{len(s.pairs)} 页")
        self.sum_hint.setText((f"处理 {process_count} · 跳过 {skip_count} · 手动标记 {manual_count} · 未匹配 {unmatched_count}" if s.pairs else "等待页面配对"))
        exp_visible = bool(s.config.aligned_overlay_reveal.enabled or s.config.transfer.mode == "aligned_overlay_reveal")
        if self.show_experimental.isChecked() != exp_visible:
            self.show_experimental.blockSignals(True); self.show_experimental.setChecked(exp_visible); self.show_experimental.blockSignals(False)
            self._set_experimental_visible(exp_visible)
        idx = self.mode.findData(s.config.transfer.mode)
        if idx >= 0: self.mode.setCurrentIndex(idx)
        self.diff_check.setChecked(s.config.mask_replace.paired_diff_enabled); self.exact_check.setChecked(bool(s.config.mask_replace.exact_identity_copy or s.config.direct_patch.exact_identity_copy))
        self.prefer_name_pair.setChecked(s.config.pairing.prefer_name_pairing); self.prefer_order_pair.setChecked(s.config.pairing.prefer_order_pairing)
        self.resume_check.setChecked(s.config.batch.resume); self.cache_check.setChecked(s.config.cache.enabled)
        if self._preview_dialog is not None and self._preview_dialog.isVisible():
            self._preview_dialog.refresh_current()

class ModelPage(QWidget):
    def __init__(self, window: "StudioWindow"):
        super().__init__(); self.window = window
        self._probe_cache = None
        self._probe_cache_at = 0.0
        self._probe_worker = None
        root = QGridLayout(self); root.setContentsMargins(18,18,18,18); root.setSpacing(14)
        self.ocr = self._choice_card("OCR", "摄影/跨版本页必须保留 OCR 证据；Apple 默认沿用 Novel Formatter 的系统 Live Text 路线。", [
            ("Apple Live Text（推荐）", "apple", "Swift VisionKit → ExtractText 快捷指令自动回退"),
            ("Apple 快捷指令", "apple_shortcut", "直接调用 ExtractText / 从图像中提取文字"),
            ("PP-OCRv5", "paddle", "低清中文主 OCR，跨平台"),
            ("Sidecar", "sidecar", "comic-text-detector 等外部结果"),
            ("关闭", "none", "纯蒙版替换可用"),
        ], self._set_ocr)
        self.reg = self._choice_card("页面配准", "先对齐整页，再做局部气泡/文本框精修。", [
            ("Auto / SIFT", "auto", "离线默认，优先稳健"),
            ("LightGlue", "lightglue", "SIFT / ALIKED / DISK"),
            ("LoFTR", "loftr", "困难页 dense fallback"),
        ], self._set_reg)
        self.bubble = self._choice_card("气泡结构", "蒙版替换优先使用成对差异；复杂版本再使用实例分割。", [
            ("几何白底气泡", "seeded_white", "无模型"),
            ("MangaLens", "mangalens", "YOLO11 instance segmentation"),
            ("Comic Translate RT-DETR-v2", "rtdetr_v2", "SOURCE-only，可选本地模型 / 明确允许后才下载"),
            ("SAM 2 / 2.1", "sam2", "SOURCE-only 点/框提示分割；默认不下载权重"),
            ("Sidecar", "sidecar", "外部专业气泡 mask"),
        ], self._set_bubble)
        root.addWidget(self.ocr,0,0); root.addWidget(self.reg,0,1); root.addWidget(self.bubble,1,0)
        status = Card("接入状态", "增强模型不会在后台静默下载。")
        self.status_labels = {}
        for key,name in [("paddle","PP-OCRv5"),("lightglue","LightGlue"),("loftr","LoFTR"),("mangalens","MangaLens"),("torch_sr","MPS 局部超分"),("apple_live_text","Apple Live Text"),("apple_shortcut","ExtractText 快捷指令")]:
            line=QHBoxLayout(); line.addWidget(QLabel(name)); line.addStretch(1); q=QLabel("检测中"); q.setObjectName("hint"); self.status_labels[key]=q; line.addWidget(q); status.layout.addLayout(line)
        root.addWidget(status,1,1)
        hardware = Card("Mac 加速与批量策略", "便宜步骤先跑；只有困难页才升级到 MPS 深度模型。")
        hform = QFormLayout()
        self.profile = QComboBox(); self.profile.addItem("智能平衡（推荐）", "balanced"); self.profile.addItem("省资源", "eco"); self.profile.addItem("性能优先", "fast")
        self.device = QComboBox(); self.device.addItem("自动选择", "auto"); self.device.addItem("Apple MPS", "mps"); self.device.addItem("CPU", "cpu")
        self.thread_ratio = QDoubleSpinBox(); self.thread_ratio.setRange(.20,1.0); self.thread_ratio.setSingleStep(.05); self.thread_ratio.setValue(.50); self.thread_ratio.setDecimals(2)
        self.mps_fraction = QDoubleSpinBox(); self.mps_fraction.setRange(.50,.95); self.mps_fraction.setSingleStep(.05); self.mps_fraction.setValue(.82); self.mps_fraction.setDecimals(2)
        hform.addRow("运行策略", self.profile); hform.addRow("推理设备", self.device); hform.addRow("CPU 线程比例", self.thread_ratio); hform.addRow("MPS 内存上限", self.mps_fraction)
        hardware.layout.addLayout(hform)
        self.device_status = QLabel(); self.device_status.setObjectName("hint"); self.device_status.setWordWrap(True); hardware.layout.addWidget(self.device_status)
        route = QLabel("Auto：同源快速判定 → SIFT/ORB → LightGlue/MPS → LoFTR/MPS。深度模型整册只加载一次。")
        route.setObjectName("quiet"); route.setWordWrap(True); hardware.layout.addWidget(route)
        root.addWidget(hardware,2,0,1,2)
        self.profile.currentIndexChanged.connect(self._set_profile); self.device.currentIndexChanged.connect(self._set_device); self.thread_ratio.valueChanged.connect(self._set_threads); self.mps_fraction.valueChanged.connect(self._set_mps_fraction)
        root.setRowStretch(3,1)

    def _choice_card(self, title, subtitle, rows, handler):
        card=Card(title,subtitle); group=QButtonGroup(card); group.setExclusive(True); card.group=group; card.radios={}
        for label,key,hint in rows:
            row=QHBoxLayout(); radio=QRadioButton(label); card.radios[key]=radio; group.addButton(radio); row.addWidget(radio); row.addStretch(1); h=QLabel(hint); h.setObjectName("quiet"); row.addWidget(h); card.layout.addLayout(row); radio.toggled.connect(lambda checked,k=key: handler(k) if checked else None)
        return card

    def _set_ocr(self,key):
        cfg = self.window.state.config.ocr
        cfg.backend = key
        cfg.source_backend = key
        cfg.target_backend = key
        self.window.statusBar().showMessage(f"OCR：{key}",2500)
    def _set_reg(self,key): self.window.state.config.registration.backend=key; self.window.statusBar().showMessage(f"配准：{key}",2500)
    def _set_bubble(self,key): self.window.state.config.bubbles.backend=key; self.window.statusBar().showMessage(f"气泡：{key}",2500)
    def _set_profile(self):
        key=self.profile.currentData() or "balanced"; cfg=self.window.state.config
        presets={
            "eco": (.35, .68, 1, 4),
            "balanced": (.50, .82, 2, 8),
            "fast": (.75, .90, 4, 12),
        }
        threads,mps_mem,prefetch,release_every=presets[key]
        cfg.runtime.cpu_thread_ratio=threads; cfg.runtime.mps_memory_fraction=mps_mem
        cfg.batch.prefetch_workers=prefetch; cfg.runtime.release_cache_every=release_every
        self.thread_ratio.blockSignals(True); self.thread_ratio.setValue(threads); self.thread_ratio.blockSignals(False)
        self.mps_fraction.blockSignals(True); self.mps_fraction.setValue(mps_mem); self.mps_fraction.blockSignals(False)
        self.device_status.setText(f"运行策略：{self.profile.currentText()} · 批量预检 {prefetch} 线程 · MPS 深度推理受控串行")
    def _set_device(self):
        key=self.device.currentData() or "auto"; cfg=self.window.state.config
        cfg.runtime.device=key; cfg.registration.device=key; cfg.bubbles.device=key; cfg.mask_replace.sr_device=key; cfg.direct_patch.sr_device=key
        self.refresh()
    def _set_threads(self): self.window.state.config.runtime.cpu_thread_ratio=float(self.thread_ratio.value())
    def _set_mps_fraction(self): self.window.state.config.runtime.mps_memory_fraction=float(self.mps_fraction.value())
    def refresh(self, force_probe: bool = False):
        cfg=self.window.state.config
        for card,key in ((self.ocr,cfg.ocr.backend),(self.reg,cfg.registration.backend),(self.bubble,cfg.bubbles.backend)):
            if key in card.radios: card.radios[key].setChecked(True)
        idx=self.device.findData(cfg.runtime.device)
        if idx>=0 and self.device.currentIndex()!=idx: self.device.blockSignals(True); self.device.setCurrentIndex(idx); self.device.blockSignals(False)
        self.thread_ratio.blockSignals(True); self.thread_ratio.setValue(cfg.runtime.cpu_thread_ratio); self.thread_ratio.blockSignals(False)
        self.mps_fraction.blockSignals(True); self.mps_fraction.setValue(cfg.runtime.mps_memory_fraction); self.mps_fraction.blockSignals(False)
        # Infer the closest UI preset from the concrete settings; config remains explicit.
        profile_key = "eco" if cfg.runtime.cpu_thread_ratio <= .40 else ("fast" if cfg.runtime.cpu_thread_ratio >= .65 else "balanced")
        pidx=self.profile.findData(profile_key)
        if pidx>=0 and self.profile.currentIndex()!=pidx: self.profile.blockSignals(True); self.profile.setCurrentIndex(pidx); self.profile.blockSignals(False)
        now = time.monotonic()
        stale = self._probe_cache is None or (now - self._probe_cache_at) > 30.0
        if force_probe or stale:
            self._start_probe()
        if self._probe_cache is None:
            self.device_status.setText("组件状态正在后台检测；切换功能区不会等待探测完成。")
            for q in self.status_labels.values():
                q.setText("检测中…")
            return
        self._apply_probe_statuses(self._probe_cache)

    def _start_probe(self):
        if self._probe_worker is not None and self._probe_worker.isRunning():
            return
        worker = ComponentProbeWorker(self.window.state.config.model_copy(deep=True))
        self._probe_worker = worker
        worker.done.connect(self._probe_done)
        worker.failed.connect(self._probe_failed)
        worker.finished.connect(self._probe_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _probe_done(self, statuses):
        self._probe_cache = statuses
        self._probe_cache_at = time.monotonic()
        self._apply_probe_statuses(statuses)

    def _probe_failed(self, message: str):
        self.device_status.setText(f"组件后台检测失败：{message}")

    def _probe_finished(self):
        self._probe_worker = None

    def _apply_probe_statuses(self, statuses):
        cfg=self.window.state.config
        mps=statuses["mps"]
        selected=cfg.runtime.device.upper() if cfg.runtime.device != "auto" else ("MPS" if mps.ready else "CPU / 自动")
        mac_hint = "；当前环境未启用 MPS，将使用已调优的 CPU" if sys.platform == "darwin" and not mps.ready else ""
        self.device_status.setText(f"计划设备：{selected} · {mps.detail}{mac_hint}")
        for key,st in statuses.items():
            if key in self.status_labels:
                self.status_labels[key].setText("已就绪" if st.ready else ("已安装 / 待模型" if st.installed else "未安装"))
                self.status_labels[key].setToolTip(st.detail)


class WorkbenchPage(QWidget):
    def __init__(self, window: "StudioWindow"):
        super().__init__(); self.window=window
        root=QHBoxLayout(self); root.setContentsMargins(18,18,18,18); root.setSpacing(14)
        split=QSplitter(Qt.Orientation.Horizontal); root.addWidget(split)

        preview=Card("替换工作台", "日文、旧中文版、最终结果、复核标注和蒙版始终绑定同一页；可直接在这里连续翻页。")
        toolbar=QGridLayout(); toolbar.setHorizontalSpacing(6); toolbar.setVerticalSpacing(5); self.view_buttons=[]
        for i,(label,key) in enumerate([("日文原图","target"),("旧中文版","source"),("最终结果","result"),("复核标注","review"),("迁移蒙版","mask"),("清除蒙版","clear_mask"),("中文迁移层","chinese_layer"),("只清日文","removed")]):
            b=QPushButton(label); b.setCheckable(True); b.setObjectName("segmented"); b.clicked.connect(lambda _=False,k=key:self.set_view(k)); toolbar.addWidget(b, i//4, i%4); self.view_buttons.append((b,key))
            toolbar.setColumnStretch(i%4, 1)
        preview.layout.addLayout(toolbar)
        page_nav=QHBoxLayout(); page_nav.setSpacing(6); page_nav.addStretch(1)
        self.prev_page=QPushButton("← 上一页"); self.prev_page.setObjectName("pageNav")
        self.page_counter=QLabel("0 / 0"); self.page_counter.setObjectName("pageCounter"); self.page_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.next_page=QPushButton("下一页 →"); self.next_page.setObjectName("pageNav")
        page_nav.addWidget(self.prev_page); page_nav.addWidget(self.page_counter); page_nav.addWidget(self.next_page); page_nav.addStretch(1)
        preview.layout.addLayout(page_nav)
        self.image=ImageView(); preview.layout.addWidget(self.image,1)
        footer=QHBoxLayout()
        self.page_caption=QLabel("未选择页面"); self.page_caption.setObjectName("hint"); self.page_caption.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.view_status=QLabel(""); self.view_status.setObjectName("quiet")
        footer.addWidget(self.page_caption,1); footer.addWidget(self.view_status,0,Qt.AlignmentFlag.AlignRight)
        preview.layout.addLayout(footer)
        split.addWidget(preview)

        side=QScrollArea(); self.side=side; side.setWidgetResizable(True)
        side.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host=QWidget(); self.side_host=host; host.setMinimumWidth(0); host.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        sl=QVBoxLayout(host); sl.setContentsMargins(0,0,4,0); sl.setSpacing(12); side.setWidget(host)
        mode=Card("替换策略 / 蒙版参数", "出版安全门禁已移除：Direct/Mask 优先完成替换；风险指标仅保留诊断，不再作为写入阻断。")
        self.publication_safety=QCheckBox("出版安全门禁已移除（兼容旧配置）")
        self.publication_safety.setChecked(False)
        self.publication_safety.setEnabled(False)
        self.publication_safety.setToolTip("v1.2.0 起该门禁不再参与 Direct/Mask 写入。仅保留基础几何有效性检查。")
        self.paired=QCheckBox("成对差异自动提取"); self.paired.setChecked(True)
        self.skip_ocr=QCheckBox("仅同源低噪声页允许跳过 OCR"); self.skip_ocr.setChecked(True)
        self.pixel_exact=QCheckBox("同源页面像素级精确覆盖"); self.pixel_exact.setChecked(True)
        self.full_bubble_patch=QCheckBox("蒙版内完整中文字形迁移（保留 TARGET 背景）")
        self.full_bubble_patch.setChecked(bool(getattr(self.window.state.config.mask_replace, "rigid_container_full_patch_enabled", True)))
        self.full_bubble_patch.setToolTip("只迁移 SOURCE 中文字形/透明度，TARGET 背景、肤色、衣服、网点和气泡底色都不允许被 SOURCE RGB 覆盖。")
        self.preserve_border=QCheckBox("保留高清日文气泡边线"); self.preserve_border.setChecked(True)
        self.blur_guard=QCheckBox("低清文字保护：模糊时禁止直接贴像素"); self.blur_guard.setChecked(True)
        self.blur_guard.setToolTip("摄影模糊、反光或低分辨率旧版先做光照归一化/墨迹重建；精准蒙版模式不会把文字交给 OCR 改写，不安全区域进入复核。")
        self.preserve_source_layout=QCheckBox("清晰旧中文版保留原字号/分列（推荐）"); self.preserve_source_layout.setChecked(True)
        self.preserve_source_layout.setToolTip("精准蒙版始终保留旧中文版真实字号/分列/符号；Apple Live Text 只提供识别证据。需要重新排字时请改用智能混合/高清重排，或手动编辑。")
        mode.layout.addWidget(self.publication_safety); mode.layout.addWidget(self.paired); mode.layout.addWidget(self.skip_ocr); mode.layout.addWidget(self.pixel_exact); mode.layout.addWidget(self.full_bubble_patch); mode.layout.addWidget(self.preserve_border); mode.layout.addWidget(self.blur_guard); mode.layout.addWidget(self.preserve_source_layout)
        photo_note=QLabel("当前策略：所有自动路径都以 TARGET 为唯一背景；白底和彩底都只清除日文文字并迁移 SOURCE 中文字形。SOURCE 的白纸、灰阶、肤色或旧背景 RGB 永远不会写进彩图。")
        photo_note.setObjectName("quiet"); photo_note.setWordWrap(True); mode.layout.addWidget(photo_note)
        sl.addWidget(mode)

        align=Card("局部对齐与清晰度")
        form=QFormLayout(); form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.local=QComboBox(); self.local.addItems(["ecc","bbox","global"]); self.sr=QComboBox(); self.sr.addItems(["auto","torch","lanczos","external","off"])
        self.fidelity=QComboBox(); self.fidelity.addItem("自动：光照归一化 → 墨迹 → OCR 重排","auto"); self.fidelity.addItem("只保留原像素","pixels"); self.fidelity.addItem("强制墨迹重建","ink"); self.fidelity.addItem("低清直接拒绝","reject")
        self.iou=QDoubleSpinBox(); self.iou.setRange(.2,1); self.iou.setSingleStep(.01); self.iou.setDecimals(3); self.iou.setValue(.80)
        self.coverage=QDoubleSpinBox(); self.coverage.setRange(.5,1); self.coverage.setSingleStep(.001); self.coverage.setDecimals(3); self.coverage.setValue(.985)
        self.sr_model=QLineEdit(); self.sr_model.setPlaceholderText("可选：本地 .pth/.safetensors 超分模型")
        self.sr_pick=QPushButton("选择…"); srrow=QWidget(); srl=QHBoxLayout(srrow); srl.setContentsMargins(0,0,0,0); srl.setSpacing(6); srl.addWidget(self.sr_model,1); srl.addWidget(self.sr_pick)
        form.addRow("局部几何",self.local); form.addRow("文字清晰策略",self.fidelity); form.addRow("源 patch 超分",self.sr); form.addRow("MPS 超分模型",srrow); form.addRow("Mask IoU 门槛",self.iou); form.addRow("目标覆盖率",self.coverage); align.layout.addLayout(form); sl.addWidget(align)

        stages=Card("检测 / 清除 / 写入分离", "参考成熟漫画翻译工具的分阶段设计：气泡检测、清除蒙版、去字预览和中文写入可独立检查，不必每改一点就重跑整页。")
        sf=QFormLayout(); sf.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); sf.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.bubble_backend=QComboBox(); self.bubble_backend.addItem("白色容器 / 结构检测（默认）","seeded_white"); self.bubble_backend.addItem("MangaLens / YOLO","mangalens"); self.bubble_backend.addItem("Comic Translate RT-DETR-v2（SOURCE-only）","rtdetr_v2"); self.bubble_backend.addItem("SAM 2 / 2.1（SOURCE-only）","sam2"); self.bubble_backend.addItem("Sidecar 外部蒙版","sidecar"); self.bubble_backend.addItem("关闭气泡检测","none")
        self.detector_size=QSpinBox(); self.detector_size.setRange(640,4096); self.detector_size.setSingleStep(128); self.detector_size.setValue(int(self.window.state.config.bubbles.mangalens_imgsz))
        self.clear_dilate=QSpinBox(); self.clear_dilate.setRange(0,20); self.clear_dilate.setValue(int(self.window.state.config.masking.max_dilation_px)); self.clear_dilate.setSuffix(" px")
        self.inpaint_backend=QComboBox(); self.inpaint_backend.addItem("自动","auto"); self.inpaint_backend.addItem("纯色纸面","solid"); self.inpaint_backend.addItem("OpenCV 修复","opencv"); self.inpaint_backend.addItem("LaMa（需配置）","lama")
        sf.addRow("气泡检测器",self.bubble_backend); sf.addRow("模型检测分辨率",self.detector_size); sf.addRow("清除 mask 最大扩张",self.clear_dilate); sf.addRow("去字修复",self.inpaint_backend); stages.layout.addLayout(sf)
        sr1=QHBoxLayout(); self.edit_clear_mask=QPushButton("编辑清除蒙版…"); self.remove_text_only=QPushButton("仅执行去字"); sr1.addWidget(self.edit_clear_mask,1); sr1.addWidget(self.remove_text_only,1); stages.layout.addLayout(sr1)
        sr2=QHBoxLayout(); self.apply_mask_review=QPushButton("应用蒙版到最终结果"); self.apply_mask_review.setObjectName("softPrimary"); self.reset_clear_mask=QPushButton("恢复自动蒙版"); sr2.addWidget(self.apply_mask_review,1); sr2.addWidget(self.reset_clear_mask,1); stages.layout.addLayout(sr2)
        stage_note=QLabel("手工蒙版保存为当前页 manual_clear_mask.png。只清日文不会重新配准、OCR 或搬字；确认后再应用到最终结果。")
        stage_note.setObjectName("quiet"); stage_note.setWordWrap(True); stages.layout.addWidget(stage_note); sl.addWidget(stages)

        recovery=Card("人工补漏 / 开放式效果字", "专门处理自动检测漏掉的开放式效果字、彩底文字、人物画面上的文字。无需 OCR，也不要求存在气泡边界；人工框选后直接进入最终复核链。")
        self.manual_effect_status=QLabel("当前页暂无人工补漏区域"); self.manual_effect_status.setObjectName("hint"); self.manual_effect_status.setWordWrap(True); recovery.layout.addWidget(self.manual_effect_status)
        self.manual_effect_candidate_status=QLabel("安全策略暂无待处理彩色/复杂文字候选"); self.manual_effect_candidate_status.setObjectName("quiet"); self.manual_effect_candidate_status.setWordWrap(True); recovery.layout.addWidget(self.manual_effect_candidate_status)
        self.manual_effect_candidate_target=QComboBox(); self.manual_effect_candidate_target.setMinimumWidth(0); self.manual_effect_candidate_target.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon); self.manual_effect_candidate_target.setMinimumContentsLength(8); recovery.layout.addWidget(self.manual_effect_candidate_target)
        erow=QHBoxLayout(); self.add_manual_effect=QPushButton("手动框选遗漏区域…"); self.add_manual_effect.setObjectName("softPrimary"); self.add_manual_effect_candidate=QPushButton("使用候选区域…"); self.undo_manual_effect=QPushButton("撤销最近区域"); erow.addWidget(self.add_manual_effect,1); erow.addWidget(self.add_manual_effect_candidate,1); erow.addWidget(self.undo_manual_effect,1); recovery.layout.addLayout(erow)
        enote=QLabel("推荐流程：自动跑整页 → QA 看漏项 → 框选开放式效果字 → 检查最终结果；若日文残留，再用上方“编辑清除蒙版”对刷几笔。")
        enote.setObjectName("quiet"); enote.setWordWrap(True); recovery.layout.addWidget(enote); sl.addWidget(recovery)

        qa=Card("当前页 QA")
        self.qa_label=QLabel("尚未处理"); self.qa_label.setObjectName("hint"); qa.layout.addWidget(self.qa_label)
        self.run=QPushButton("处理当前页"); self.run.setObjectName("primary"); qa.layout.addWidget(self.run); sl.addWidget(qa)

        manual=Card("低置信中文 / 待补文字", "默认先显示可恢复的中文候选；模糊、裁切或可能不准确的区域可接受、重新编辑或一键还原日文。")
        self.manual_status=QLabel("当前页没有待复核气泡"); self.manual_status.setObjectName("hint"); self.manual_status.setWordWrap(True); manual.layout.addWidget(self.manual_status)
        self.manual_target=QComboBox(); manual.layout.addWidget(self.manual_target)
        self.manual_text=QPlainTextEdit(); self.manual_text.setPlaceholderText("需要修改时，在这里输入完整中文译文…"); self.manual_text.setMaximumHeight(92); manual.layout.addWidget(self.manual_text)
        mrow=QHBoxLayout(); self.manual_orientation=QComboBox(); self.manual_orientation.addItem("自动排版","auto"); self.manual_orientation.addItem("竖排","vertical"); self.manual_orientation.addItem("横排","horizontal")
        self.manual_apply=QPushButton("重新编辑并高清排字"); self.manual_apply.setObjectName("softPrimary"); mrow.addWidget(self.manual_orientation); mrow.addWidget(self.manual_apply,1); manual.layout.addLayout(mrow)
        drow=QHBoxLayout(); self.candidate_accept=QPushButton("接受当前中文候选"); self.candidate_restore=QPushButton("还原日文"); drow.addWidget(self.candidate_accept,1); drow.addWidget(self.candidate_restore,1); manual.layout.addLayout(drow)

        # Long Chinese labels and combo-box contents must not force the review
        # sidebar wider than its viewport.  Qt's default minimumSizeHint for a
        # QComboBox is based on its longest item, which previously created the
        # horizontal clipping visible on narrow macOS windows.
        for widget in [self.local, self.sr, self.fidelity, self.sr_model, self.bubble_backend,
                       self.detector_size, self.clear_dilate, self.inpaint_backend, self.manual_target,
                       self.manual_effect_candidate_target, self.manual_orientation, self.manual_text]:
            widget.setMinimumWidth(0)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, widget.sizePolicy().verticalPolicy())
        for combo in [self.local, self.sr, self.fidelity, self.bubble_backend, self.inpaint_backend, self.manual_target, self.manual_effect_candidate_target, self.manual_orientation]:
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        for button in [self.sr_pick, self.edit_clear_mask, self.remove_text_only, self.apply_mask_review,
                       self.reset_clear_mask, self.add_manual_effect, self.add_manual_effect_candidate, self.undo_manual_effect, self.manual_apply,
                       self.candidate_accept, self.candidate_restore]:
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        sl.addWidget(manual); sl.addStretch(1)
        side.setMinimumWidth(0); side.setMaximumWidth(420); split.addWidget(side); self.split=split; self.preview_card=preview
        split.setSizes([850,340]); split.setStretchFactor(0,1); split.setStretchFactor(1,0); split.setChildrenCollapsible(False)
        self._compact_layout=None
        QTimer.singleShot(0, self._update_responsive_workbench)
        self.prev_page.clicked.connect(lambda: self._move_page(-1)); self.next_page.clicked.connect(lambda: self._move_page(1))
        self.run.clicked.connect(self._run)
        self.sr_pick.clicked.connect(self._choose_sr_model)
        self.manual_apply.clicked.connect(self._apply_manual_reletter)
        self.candidate_accept.clicked.connect(lambda: self._set_candidate_decision("accept"))
        self.candidate_restore.clicked.connect(lambda: self._set_candidate_decision("restore"))
        self.edit_clear_mask.clicked.connect(self._edit_clear_mask)
        self.remove_text_only.clicked.connect(self._remove_text_only)
        self.apply_mask_review.clicked.connect(self._apply_mask_review)
        self.reset_clear_mask.clicked.connect(self._reset_clear_mask)
        self.add_manual_effect.clicked.connect(self._add_manual_effect_region)
        self.add_manual_effect_candidate.clicked.connect(self._add_next_manual_effect_candidate)
        self.undo_manual_effect.clicked.connect(self._undo_manual_effect_region)
        for w in [self.paired,self.skip_ocr,self.pixel_exact,self.full_bubble_patch,self.preserve_border,self.blur_guard,self.preserve_source_layout]: w.toggled.connect(self._sync)
        self.local.currentTextChanged.connect(self._sync); self.fidelity.currentIndexChanged.connect(self._sync); self.sr.currentTextChanged.connect(self._sync); self.iou.valueChanged.connect(self._sync); self.coverage.valueChanged.connect(self._sync)
        self.bubble_backend.currentIndexChanged.connect(self._sync); self.detector_size.valueChanged.connect(self._sync); self.clear_dilate.valueChanged.connect(self._sync); self.inpaint_backend.currentIndexChanged.connect(self._sync)
        bi=self.bubble_backend.findData(self.window.state.config.bubbles.backend); self.bubble_backend.setCurrentIndex(max(0,bi))
        ii=self.inpaint_backend.findData(self.window.state.config.inpainting.backend); self.inpaint_backend.setCurrentIndex(max(0,ii))
        self.current_view="target"; self.set_view("target")

    def _update_responsive_workbench(self):
        """Switch the workbench to a vertical stack when horizontal space is tight.

        The previous fixed horizontal splitter required roughly preview(360) +
        sidebar(285) + margins.  On a narrow/split-screen macOS window Qt kept
        those minimums and simply clipped the right side of the settings panel.
        Compact mode gives both areas the full available width instead.
        """
        if not hasattr(self, "split") or not hasattr(self, "side"):
            return
        compact = self.width() < 920
        if self._compact_layout is compact:
            return
        self._compact_layout = compact
        if compact:
            self.split.setOrientation(Qt.Orientation.Vertical)
            self.side.setMinimumWidth(0); self.side.setMaximumWidth(16777215)
            self.side.setMinimumHeight(250); self.side.setMaximumHeight(16777215)
            h=max(520,self.height())
            self.split.setSizes([max(280,int(h*0.52)), max(250,int(h*0.48))])
            self.split.setStretchFactor(0,1); self.split.setStretchFactor(1,1)
        else:
            self.split.setOrientation(Qt.Orientation.Horizontal)
            self.side.setMinimumHeight(0); self.side.setMaximumHeight(16777215)
            self.side.setMinimumWidth(300); self.side.setMaximumWidth(420)
            w=max(900,self.width())
            side_w=min(390,max(320,int(w*0.30)))
            self.split.setSizes([max(520,w-side_w),side_w])
            self.split.setStretchFactor(0,1); self.split.setStretchFactor(1,0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._update_responsive_workbench)

    def _sync(self):
        c=self.window.state.config.mask_replace
        # v1.2.0: publication blocking is removed.  Keep the legacy config field
        # false so old saved configs cannot silently restore the retired gate.
        safety=False
        c.publication_safety_enabled=False; c.paired_diff_enabled=self.paired.isChecked(); c.paired_diff_skip_ocr=self.skip_ocr.isChecked(); c.exact_identity_copy=self.pixel_exact.isChecked(); c.rigid_container_full_patch_enabled=self.full_bubble_patch.isChecked(); c.rigid_container_full_patch_preserve_target_border=self.preserve_border.isChecked(); c.preserve_target_border=self.preserve_border.isChecked(); c.reject_blurry_source=self.blur_guard.isChecked(); c.fallback_reletter_on_blur=False; c.photo_pair_fallback_reletter_missing=False; c.photo_pair_prefer_reletter_with_ocr=False; c.strict_mask_replace_no_ocr_reletter=True; c.photo_pair_preserve_sharp_source_layout=self.preserve_source_layout.isChecked(); c.text_fidelity_mode=self.fidelity.currentData() or "auto"; c.local_fit=self.local.currentText(); c.sr_backend=self.sr.currentText(); c.sr_model_path=self.sr_model.text().strip() or None; c.min_mask_iou=float(self.iou.value()); c.min_target_coverage=float(self.coverage.value())
        d=self.window.state.config.direct_patch
        d.publication_safety_enabled=False
        d.source_direct_fail_on_artwork_rejections=bool(safety)
        d.allow_target_aware_colored_composite=True
        d.source_direct_colored_preserve_target_fill=True
        self.window.state.config.bubbles.backend=self.bubble_backend.currentData() or "seeded_white"; self.window.state.config.bubbles.mangalens_imgsz=int(self.detector_size.value()); self.window.state.config.bubbles.rtdetr_imgsz=int(self.detector_size.value()); self.window.state.config.masking.max_dilation_px=int(self.clear_dilate.value()); self.window.state.config.inpainting.backend=self.inpaint_backend.currentData() or "auto"

    def _current_page_dir(self) -> Path | None:
        ws=self._workspace()
        return ws.page_root if ws is not None else None

    def _edit_clear_mask(self):
        page_dir=self._current_page_dir(); pair=self.window.current_pair()
        if page_dir is None or pair is None or not (page_dir/"target_original.png").exists():
            QMessageBox.information(self,"尚未处理","请先处理当前页，生成目标图和自动清除蒙版。")
            return
        import cv2
        import numpy as np
        mask_path=page_dir/"manual_clear_mask.png"
        if not mask_path.exists(): mask_path=page_dir/"target_clear_mask.png"
        if not mask_path.exists(): mask_path=page_dir/"clear_mask.png"
        mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        target=cv2.imread(str(page_dir/"target_original.png"),cv2.IMREAD_COLOR)
        if target is None:
            QMessageBox.warning(self,"无法读取","当前页 target_original.png 无法读取。"); return
        if mask is None: mask=np.zeros(target.shape[:2],dtype=np.uint8)
        dlg=MaskEditorDialog(page_dir/"target_original.png",mask,self)
        if dlg.exec()!=QDialog.DialogCode.Accepted: return
        write_image(page_dir/"manual_clear_mask.png",dlg.result_mask())
        try: generate_remove_text_preview(page_dir,self.window.state.config.model_copy(deep=True))
        except Exception as exc: QMessageBox.warning(self,"去字预览失败",str(exc))
        self.current_view="clear_mask"
        for b,k in self.view_buttons: b.setChecked(k==self.current_view)
        self.window.statusBar().showMessage("已保存手工清除蒙版；可先检查『只清日文』再应用到最终结果。",5000); self.refresh()

    def _remove_text_only(self):
        page_dir=self._current_page_dir()
        if page_dir is None or not (page_dir/"target_original.png").exists():
            QMessageBox.information(self,"尚未处理","请先处理当前页。")
            return
        self._sync(); QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            out=generate_remove_text_preview(page_dir,self.window.state.config.model_copy(deep=True))
            self.current_view="removed"
            for b,k in self.view_buttons: b.setChecked(k=="removed")
            self.window.statusBar().showMessage(f"只清日文完成：{Path(out).name}",5000); self.refresh()
        except Exception as exc: QMessageBox.critical(self,"只清日文失败",str(exc))
        finally: QApplication.restoreOverrideCursor()

    def _apply_mask_review(self):
        page_dir=self._current_page_dir()
        if page_dir is None or not (page_dir/"project.json").exists():
            QMessageBox.information(self,"尚未处理","请先处理当前页。")
            return
        self._sync(); QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            final=apply_review_page(page_dir,self.window.state.config.model_copy(deep=True)); self.window.state.last_result_path=str(final)
            pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""; proj=self.window.state.projects_by_page.get(page_id)
            if proj is not None: proj.artifacts["final"]=str(final)
            self._sync_reviewed_book_final(final); self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage("已把当前清除蒙版应用到最终结果。",5000); self.refresh()
        except Exception as exc: QMessageBox.critical(self,"应用蒙版失败",str(exc))
        finally: QApplication.restoreOverrideCursor()

    def _reset_clear_mask(self):
        page_dir=self._current_page_dir()
        if page_dir is None: return
        for name in ("manual_clear_mask.png","removed_text_preview.png","remove_text_stage.json"):
            try: (page_dir/name).unlink(missing_ok=True)
            except OSError: pass
        self.current_view="clear_mask"
        for b,k in self.view_buttons: b.setChecked(k=="clear_mask")
        self.window.statusBar().showMessage("已恢复当前页自动清除蒙版。",4000); self.refresh()


    def _manual_effect_rows(self) -> list[dict[str, Any]]:
        page_dir=self._current_page_dir()
        if page_dir is None: return []
        path=page_dir/"review_overrides.json"
        if not path.exists(): return []
        data=normalize_overrides(load_json(path))
        return _json_dict_rows(data.get("manual_effect_regions"))

    def _manual_effect_candidates(self) -> list[dict[str, Any]]:
        ws=self._workspace()
        return as_dict_rows(getattr(ws, "manual_effect_candidates", [])) if ws is not None else []

    def _unhandled_manual_effect_candidates(self) -> list[dict[str, Any]]:
        # Only candidates that passed the stricter text-seeded actionability gate
        # are allowed to prefill a box automatically. Other Direct rejections are
        # still reported in diagnostics/status, but the user must frame them
        # manually so artwork is never mistaken for text.
        actionable=[x for x in self._manual_effect_candidates() if bool(x.get("auto_actionable", False))]
        return _manual_effect_unhandled_candidates(actionable, self._manual_effect_rows())

    def _apply_manual_effect_overrides(self, page_dir: Path, overrides: dict[str, Any]):
        save_json(page_dir/"review_overrides.json",overrides)
        rows=_json_dict_rows(overrides.get("manual_effect_regions"))
        frozen=page_dir/"manual_effect_base.png"
        if not rows:
            # Removing the final manual region must restore the exact page that
            # existed before omission repair, including every successful Chinese
            # replacement and previous QA decision.
            stable=page_dir/"final_auto.png"
            src=stable if stable.exists() else (frozen if frozen.exists() else page_dir/"final.png")
            dst=page_dir/"final_reviewed.png"
            if src.exists():
                shutil.copy2(src,dst); final=dst
            else:
                final=page_dir/"target_original.png"
            for name in ("manual_effect_transfer_layer.png","manual_effect_transfer_mask.png","manual_effect_clear_mask.png"):
                try: (page_dir/name).unlink(missing_ok=True)
                except OSError: pass
            try: frozen.unlink(missing_ok=True)
            except OSError: pass
            try: (page_dir/"final_auto.png").unlink(missing_ok=True)
            except OSError: pass
        else:
            final=apply_review_page(page_dir,self.window.state.config.model_copy(deep=True))
            # Never silently claim success when a saved Reveal preview patch was
            # not committed. This was the user's most confusing failure mode: the
            # editor showed Chinese, Apply returned normally, but final stayed old.
            expected = {
                str(x.get("id", "")): int(x.get("reveal_patch_pixels", 0) or 0)
                for x in rows if str(x.get("reveal_patch_file", "") or "")
            }
            if any(v > 0 for v in expected.values()):
                audit_path=page_dir/"review_applied.json"
                audit=normalize_review_applied(load_json(audit_path) if audit_path.exists() else {})
                got={str(x.get("id", "")): dict(x) for x in _json_dict_rows(audit.get("manual_effect_applied"))}
                failed=[rid for rid,pix in expected.items() if pix>0 and (rid not in got or not bool(got[rid].get("success")) or not bool(got[rid].get("preview_patch_exact")))]
                if failed:
                    raise RuntimeError("人工补漏预览补丁未写入最终结果："+", ".join(failed))
        self.window.state.last_result_path=str(final)
        pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""; proj=self.window.state.projects_by_page.get(page_id)
        if proj is not None:
            proj.artifacts["final"]=str(final)
            proj.artifacts["final_reviewed"]=str(final)
        self._sync_reviewed_book_final(final)
        # Manual commits rewrite a stable path in-place. Force the workbench to
        # decode it again instead of relying solely on filesystem timestamp keys.
        try:
            self.image.clear_cache()
        except Exception:
            pass
        self.current_view="result"
        for b,k in self.view_buttons: b.setChecked(k=="result")
        try:
            changed_rows=len(rows)
            self.window.statusBar().showMessage(
                f"人工补漏已应用：已同步 final_reviewed.png → final.png（{changed_rows} 个区域）。",
                6000,
            )
        except Exception:
            pass

    def _start_manual_gui_flow(self, page_dir: Path, preset_candidate: dict[str, Any] | None = None) -> None:
        save_json(page_dir/"manual_gui_flow.json", {
            "schema": "manga_hd_translation_transfer.manual_gui_flow.v1",
            "version": VERSION,
            "started_at": time.time(),
            "preset_candidate": dict(preset_candidate or {}),
            "steps": [],
        })

    def _trace_manual_gui_flow(self, page_dir: Path, stage: str, payload: dict[str, Any] | None = None) -> None:
        path=page_dir/"manual_gui_flow.json"
        data=as_dict(load_json(path)) if path.exists() else {
            "schema": "manga_hd_translation_transfer.manual_gui_flow.v1", "version": VERSION, "steps": []
        }
        steps=as_dict_rows(data.get("steps"))
        steps.append({"stage":str(stage), "time":time.time(), **as_dict(payload)})
        data["steps"]=steps; data["last_stage"]=str(stage); save_json(path,data)

    def _commit_manual_effect_dialog_result(self, page_dir: Path, row: dict[str, Any], reveal, reveal_patch, preset_candidate: dict[str, Any] | None = None) -> Path:
        """Qt adapter for the core manual-review transaction service."""
        def trace(stage: str, payload: dict[str, Any]):
            self._trace_manual_gui_flow(page_dir, stage, payload)

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = commit_manual_effect(
                page_dir, row, reveal, reveal_patch,
                self.window.state.config.model_copy(deep=True),
                preset_candidate=preset_candidate,
                trace=trace,
            )
        finally:
            QApplication.restoreOverrideCursor()

        self.window.state.last_result_path = str(result.final_reviewed)
        pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""; proj=self.window.state.projects_by_page.get(page_id)
        if proj is not None:
            proj.artifacts["final"] = str(result.final)
            proj.artifacts["final_reviewed"] = str(result.final_reviewed)
        self._sync_reviewed_book_final(result.final_reviewed)
        try:
            self.image.clear_cache()
        except Exception:
            pass
        self.current_view = "result"
        for b,k in self.view_buttons:
            b.setChecked(k == "result")
        self.window.statusBar().showMessage(
            f"人工补漏已直接提交 · 本页共 {result.region_count} 个区域 · final 已同步", 6000
        )
        self.refresh()
        return result.final_reviewed

    def _add_manual_effect_region(self, preset_candidate: dict[str, Any] | None = None):
        page_dir=self._current_page_dir(); ws=self._workspace()
        if page_dir is None or ws is None or not ws.project_path or not ws.project_path.exists():
            QMessageBox.information(self,"尚未处理","请先处理当前页。人工补漏复用本页已经保存的配准，不会重新跑 OCR。")
            return
        source_path=page_dir/"source_original.png"; target_path=page_dir/"target_original.png"
        if not source_path.exists() or not target_path.exists():
            QMessageBox.warning(self,"缺少页面缓存","当前页缺少 source_original.png 或 target_original.png，请重新处理当前页一次。")
            return
        project=normalize_project(load_json(ws.project_path))
        preset_candidate = as_dict(preset_candidate)
        initial_bbox = as_list(preset_candidate.get("target_bbox"))
        initial_mode = str(preset_candidate.get("suggested_manual_mode", "reveal_text") or "reveal_text") if preset_candidate else None
        self._start_manual_gui_flow(page_dir,preset_candidate)
        self._trace_manual_gui_flow(page_dir,"manual_dialog_opened",{"initial_bbox":initial_bbox,"initial_mode":initial_mode or ""})
        def _commit(row,reveal,reveal_patch):
            return self._commit_manual_effect_dialog_result(page_dir,row,reveal,reveal_patch,preset_candidate)
        def _trace(stage,payload):
            self._trace_manual_gui_flow(page_dir,stage,payload)
        try:
            dlg=ManualEffectDialog(source_path,target_path,project,self,initial_bbox=initial_bbox,initial_mode=initial_mode,commit_handler=_commit,trace_handler=_trace)
        except Exception as exc:
            self._trace_manual_gui_flow(page_dir,"manual_dialog_failed",{"reason":str(exc)})
            QMessageBox.critical(self,"无法打开人工补漏",str(exc)); return
        result=dlg.exec()
        if result!=QDialog.DialogCode.Accepted:
            self._trace_manual_gui_flow(page_dir,"manual_dialog_cancelled")
            return
        if dlg.was_committed_directly():
            self._trace_manual_gui_flow(page_dir,"dialogs_closed_after_commit")
            try: self.image.clear_cache()
            except Exception: pass
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.refresh()
            return
        # Compatibility fallback for externally constructed dialogs without a
        # commit handler. The normal workbench path above always commits directly.
        self._trace_manual_gui_flow(page_dir,"legacy_dialog_handoff")
        try:
            self._commit_manual_effect_dialog_result(page_dir,dlg.result_row(),dlg.result_reveal_mask(),dlg.result_reveal_patch(),preset_candidate)
            self._trace_manual_gui_flow(page_dir,"dialogs_closed_after_commit")
        except Exception as exc:
            self._trace_manual_gui_flow(page_dir,"legacy_commit_failed",{"reason":str(exc)})
            QMessageBox.critical(self,"人工补漏失败",str(exc))

    def _add_next_manual_effect_candidate(self):
        candidates = self._unhandled_manual_effect_candidates()
        idx=self.manual_effect_candidate_target.currentIndex()
        if not candidates:
            QMessageBox.information(self, "没有候选区域", "当前页没有新的自动候选开放式文字区域。")
            return
        if idx < 0 or idx >= len(candidates):
            idx=0
        self._add_manual_effect_region(candidates[idx])

    def _undo_manual_effect_region(self):
        page_dir=self._current_page_dir()
        if page_dir is None: return
        override_path=page_dir/"review_overrides.json"; overrides=normalize_overrides(load_json(override_path) if override_path.exists() else {})
        rows=_json_dict_rows(overrides.get("manual_effect_regions"))
        if not rows:
            QMessageBox.information(self,"没有人工区域","当前页没有可撤销的人工补漏区域。")
            return
        removed=rows.pop()
        for key in ("reveal_mask_file","reveal_patch_file"):
            name=str(removed.get(key,"") or "").strip()
            if name:
                try: (page_dir/name).unlink(missing_ok=True)
                except OSError: pass
        overrides["manual_effect_regions"]=rows; overrides["status"]="reviewed_with_manual_effect" if rows else "reviewed"
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._apply_manual_effect_overrides(page_dir,overrides)
            self.window.statusBar().showMessage(f"已撤销最近人工区域 · 剩余 {len(rows)} 个",4000); self.refresh()
        except Exception as exc: QMessageBox.critical(self,"撤销失败",str(exc))
        finally: QApplication.restoreOverrideCursor()

    def _choose_sr_model(self):
        path,_=QFileDialog.getOpenFileName(self,"选择本地超分模型",str(Path.home()),"Torch models (*.pth *.pt *.ckpt *.safetensors);;All files (*)")
        if path: self.sr_model.setText(path); self._sync()
    def _run(self): self._sync(); self.window.run_current_page()
    def set_view(self,key):
        self.current_view=key
        for b,k in self.view_buttons: b.setChecked(k==key)
        self.refresh()

    def _move_page(self, delta: int):
        self.window.set_selected_page(self.window.state.selected_index + int(delta))

    def _workspace(self):
        pair=self.window.current_pair()
        if pair is None: return None
        page_id=page_id_for_pair(pair)
        project=self.window.state.projects_by_page.get(page_id)
        return resolve_page_workspace(self.window.state.output_dir, pair, project)

    def _manual_queue(self):
        ws=self._workspace()
        return list(ws.review_regions) if ws is not None else []

    def _sync_reviewed_book_final(self, final: str | Path):
        """Synchronize reviewed output through one core result-state contract."""
        final_path = Path(final)
        if final_path.exists() and final_path.name == "final_reviewed.png":
            commit_reviewed_result(final_path.parent, final_path)
        pair=self.window.current_pair()
        if pair is None or not self.window.state.output_dir:
            return
        dst=Path(self.window.state.output_dir)/"final"/(Path(pair.target_path).stem+".png")
        try:
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(str(final_path),str(dst))
        except OSError:
            pass

    def _apply_manual_reletter(self):
        ws=self._workspace()
        queue=self._manual_queue()
        idx=self.manual_target.currentIndex()
        text=self.manual_text.toPlainText().strip()
        if ws is None or ws.page_root is None or not ws.project_path or not ws.project_path.exists() or not queue or idx < 0 or idx >= len(queue):
            QMessageBox.information(self,"没有待补文字","当前页没有需要人工补全的气泡。")
            return
        if not text:
            QMessageBox.information(self,"请输入完整中文","请先输入这个气泡的完整中文译文。")
            return
        page_dir=ws.page_root
        row=dict(queue[idx])
        override_path=page_dir/"review_overrides.json"
        overrides=normalize_overrides(load_json(override_path) if override_path.exists() else {})
        entries=_json_dict_rows(overrides.get("manual_reletter"))
        target_id=str(row.get("target_bubble_id", ""))
        item={
            "target_bubble_id": target_id,
            "target_bbox": as_list(row.get("target_bbox")),
            "text": text,
            "orientation": self.manual_orientation.currentData() or "auto",
            "reason": row.get("reason", ""),
        }
        entries=[x for x in entries if str(x.get("target_bubble_id", "")) != target_id]
        entries.append(item)
        overrides["manual_reletter"]=entries
        overrides["restore_target_bubbles"]=[x for x in list(overrides.get("restore_target_bubbles",[]) or []) if str(x)!=target_id]
        overrides["accept_candidate_targets"]=[x for x in list(overrides.get("accept_candidate_targets",[]) or []) if str(x)!=target_id]
        overrides["status"]="reviewed_with_manual_reletter"
        save_json(override_path,overrides)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            final=apply_review_page(page_dir,self.window.state.config.model_copy(deep=True))
            self.window.state.last_result_path=str(final)
            pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""
            p=self.window.state.projects_by_page.get(page_id)
            if p is not None: p.artifacts["final"]=str(final)
            self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage(f"已补字：{target_id}",5000)
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self,"补字失败",str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _set_candidate_decision(self, action: str):
        ws=self._workspace(); queue=self._manual_queue(); idx=self.manual_target.currentIndex()
        if ws is None or ws.page_root is None or not ws.project_path or not ws.project_path.exists() or not queue or idx < 0 or idx >= len(queue):
            QMessageBox.information(self,"没有待复核中文","当前页没有低置信中文候选。")
            return
        page_dir=ws.page_root; row=dict(queue[idx]); tid=str(row.get("target_bubble_id", ""))
        override_path=page_dir/"review_overrides.json"; overrides=normalize_overrides(load_json(override_path) if override_path.exists() else {})
        accepted=set(map(str,overrides.get("accept_candidate_targets",[]) or [])); restored=set(map(str,overrides.get("restore_target_bubbles",[]) or []))
        entries=[x for x in _json_dict_rows(overrides.get("manual_reletter")) if str(x.get("target_bubble_id", "")) != tid]
        if action=="accept": accepted.add(tid); restored.discard(tid); overrides["status"]="reviewed_candidate_accepted"
        else: restored.add(tid); accepted.discard(tid); overrides["status"]="reviewed_candidate_restored"
        overrides["accept_candidate_targets"]=sorted(accepted); overrides["restore_target_bubbles"]=sorted(restored); overrides["manual_reletter"]=entries
        save_json(override_path,overrides)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            final=apply_review_page(page_dir,self.window.state.config.model_copy(deep=True)); self.window.state.last_result_path=str(final)
            pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""; p=self.window.state.projects_by_page.get(page_id)
            if p is not None: p.artifacts["final"]=str(final)
            self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage(("已接受中文候选：" if action=="accept" else "已还原日文：")+tid,5000); self.refresh()
        except Exception as exc: QMessageBox.critical(self,"复核操作失败",str(exc))
        finally: QApplication.restoreOverrideCursor()

    def refresh(self):
        s=self.window.state
        total=len(s.pairs)
        if not total:
            self.image.set_image(None); self.page_caption.setText("未选择页面"); self.page_counter.setText("0 / 0")
            self.prev_page.setEnabled(False); self.next_page.setEnabled(False); self.view_status.setText("")
            self.manual_target.clear(); self.manual_effect_candidate_target.clear(); self.manual_status.setText("当前页没有待复核气泡"); self.manual_effect_status.setText("当前页暂无人工补漏区域"); self.manual_effect_candidate_status.setText("暂无待处理彩色/复杂文字候选")
            self.add_manual_effect.setEnabled(False); self.add_manual_effect_candidate.setEnabled(False); self.undo_manual_effect.setEnabled(False)
            return
        idx=max(0,min(s.selected_index,total-1)); pair=s.pairs[idx]
        ws=self._workspace()
        self.page_counter.setText(f"{idx+1} / {total}")
        self.prev_page.setEnabled(idx>0); self.next_page.setEnabled(idx<total-1)
        page_root=ws.page_root if ws else None
        manual_clear=(page_root/"manual_clear_mask.png") if page_root else None
        auto_clear=(page_root/"target_clear_mask.png") if page_root else None
        paths={
            "source": pair.source_path,
            "target": pair.target_path,
            "result": ws.result_path if ws else "",
            "review": ws.review_path if ws else "",
            "mask": ws.mask_path if ws else "",
            "clear_mask": str(manual_clear if manual_clear is not None and manual_clear.exists() else auto_clear) if page_root and ((manual_clear and manual_clear.exists()) or (auto_clear and auto_clear.exists())) else "",
            "chinese_layer": str(page_root/"chinese_transfer_layer.png") if page_root and (page_root/"chinese_transfer_layer.png").exists() else "",
            "removed": str(page_root/"removed_text_preview.png") if page_root and (page_root/"removed_text_preview.png").exists() else "",
        }
        q=ws.qa_summary if ws else {}
        if q:
            self.qa_label.setText(f"错误 {q.get('errors',0)} · 警告 {q.get('warnings',0)} · {'通过' if q.get('pass') else '需复核'}")
        else:
            self.qa_label.setText("尚未处理")
        queue=list(ws.review_regions) if ws else []
        effect_rows=self._manual_effect_rows()
        effect_candidates=self._unhandled_manual_effect_candidates()
        direct_diag={}
        if ws is not None:
            dm=(ws.meta or {}).get("direct_patch",{}) if isinstance((ws.meta or {}).get("direct_patch",{}),dict) else {}
            direct_diag=dm.get("diagnostics",{}) if isinstance(dm.get("diagnostics",{}),dict) else {}
        rejected_art=int(direct_diag.get("rejected_artwork_like",0) or 0)
        skipped_review=int(direct_diag.get("review_candidates_skipped",0) or 0)
        all_candidate_count=len(self._manual_effect_candidates())
        safety_on=False  # v1.1.0: publication blocking removed
        if effect_candidates:
            reason_counts={}
            for row in effect_candidates:
                key=str(row.get("reason", "manual_reveal")); reason_counts[key]=reason_counts.get(key,0)+1
            summary="、".join(f"{k}×{v}" for k,v in list(reason_counts.items())[:3])
            self.manual_effect_status.setText((f"已添加 {len(effect_rows)} 个人工补漏区域；" if effect_rows else "")+f"还有 {len(effect_candidates)} 个自动候选区域待检查。")
            prefix="出版安全门禁未自动写入" if safety_on else "自动检测仍无法确认/定位"
            self.manual_effect_candidate_status.setText(f"{prefix}这些复杂区域：{summary}。可进入“擦除显字”继续处理，背景仍使用高清 TARGET。")
        else:
            if rejected_art or skipped_review or all_candidate_count:
                prefix="Direct 出版安全拒绝" if safety_on else "Direct 非文字/几何候选跳过"
                self.manual_effect_candidate_status.setText(
                    f"{prefix} {rejected_art} 个、待复核 {skipped_review} 个；"
                    "当前没有可自动预填的文字候选。仍可使用“手动框选遗漏区域…”。"
                )
            else:
                self.manual_effect_candidate_status.setText("暂无待处理彩色/复杂文字候选")
        old_cid=self.manual_effect_candidate_target.currentData() if self.manual_effect_candidate_target.count() else None
        self.manual_effect_candidate_target.blockSignals(True); self.manual_effect_candidate_target.clear()
        reason_labels={
            "colored_target_requires_reveal":"彩色目标 · Direct 安全拒绝",
            "colored_complex_region_requires_reveal":"彩色复杂区域 · 建议擦除显字",
            "colored_text_manual_reveal_after_safe_reject":"彩色文字清除风险 · 建议擦除显字",
            "spiky_text_like_region_needs_manual_reveal":"开放式/爆发文字 · 建议擦除显字",
            "uncertain_container_alignment":"局部边界不确定 · 人工确认",
        }
        for i,row in enumerate(effect_candidates,1):
            box=as_list(row.get("target_bbox")); reason=str(row.get("reason","") or "")
            pos=f"{box[0]},{box[1]}–{box[2]},{box[3]}" if len(box)==4 else "未知区域"
            cid=f"{reason}:{pos}"
            self.manual_effect_candidate_target.addItem(f"{i}. {reason_labels.get(reason,reason or '安全策略候选')} · {pos}",cid)
        if old_cid:
            ci=self.manual_effect_candidate_target.findData(old_cid)
            if ci>=0: self.manual_effect_candidate_target.setCurrentIndex(ci)
        self.manual_effect_candidate_target.blockSignals(False)
        can_manual=bool(page_root and (page_root/"project.json").exists())
        self.undo_manual_effect.setEnabled(bool(effect_rows)); self.add_manual_effect.setEnabled(can_manual); self.add_manual_effect_candidate.setEnabled(bool(effect_candidates) and can_manual); self.manual_effect_candidate_target.setEnabled(bool(effect_candidates))
        current_id=self.manual_target.currentData() if self.manual_target.count() else None
        self.manual_target.blockSignals(True); self.manual_target.clear()
        for row in queue:
            tid=str(row.get("target_bubble_id", "待复核气泡")); sides=str(row.get("source_edge_sides", "")); candidate=bool(row.get("candidate_applied",False)); reason=str(row.get("reason", ""))
            if reason=="photographed_text_without_ocr_reletter": state="摄影中文字形 · 可能模糊/扭曲"
            elif candidate: state="已先替换中文候选 · 可能不完整/不准确"
            else: state="待补中文"
            self.manual_target.addItem(f"{tid} · {state} · {sides or '可编辑/还原'}", tid)
        if current_id:
            mi=self.manual_target.findData(current_id)
            if mi>=0: self.manual_target.setCurrentIndex(mi)
        self.manual_target.blockSignals(False)
        self.manual_status.setText(f"发现 {len(queue)} 个待复核区域；默认先给中文候选。可接受、重新编辑或还原日文。" if queue else "当前页没有待复核气泡")
        enabled=bool(queue)
        for widget in [self.manual_target,self.manual_text,self.manual_orientation,self.manual_apply,self.candidate_accept,self.candidate_restore]: widget.setEnabled(enabled)
        current_path=paths.get(self.current_view,"")
        self.image.set_image(current_path)
        source_name=Path(pair.source_path).name; target_name=Path(pair.target_path).name
        page_mark=self.window.page_mark_for_pair(pair)
        mark_suffix=f" · {page_mark.label}{' · 跳过' if not page_mark.should_process else ''}"
        self.page_caption.setText(f"第 {idx+1} 页 · {source_name} → {target_name}{mark_suffix}")
        passthrough_reason = str((ws.meta if ws else {}).get("passthrough_reason") or "")
        if not page_mark.should_process:
            self.qa_label.setText(f"页面管理：{page_mark.label} · 不进入精准蒙版")
        elif passthrough_reason == "source_no_transferable_text":
            self.qa_label.setText("无需替换 · 中文页未发现可迁移气泡/文本框")
        if current_path:
            if not page_mark.should_process:
                self.view_status.setText("跳过替换 · 高清日文原页已保留在最终输出")
            elif passthrough_reason == "source_no_transferable_text":
                self.view_status.setText("无需替换 · 高清日文原页已作为最终页输出")
            else:
                self.view_status.setText("已同步到当前页")
        elif self.current_view in {"result","review","mask","clear_mask","chinese_layer","removed"}:
            self.view_status.setText("本页尚无该输出")
        else:
            self.view_status.setText("")



class ExportPage(QWidget):
    def __init__(self, window: "StudioWindow"):
        super().__init__(); self.window=window
        root=QHBoxLayout(self); root.setContentsMargins(18,18,18,18); root.setSpacing(14)
        left=Card("出版输出", "高清日文母版始终保留，导出最终图、QA、工程 JSON 与可编辑图层。")
        self.out=PathRow("输出目录","选择目录"); left.layout.addWidget(self.out)
        self.debug=QCheckBox("保存配准 / 结构 / 蒙版 Debug 图"); self.debug.setChecked(True)
        self.layers=QCheckBox("输出 OpenRaster / PSD 可编辑图层"); self.layers.setChecked(True)
        left.layout.addWidget(self.debug); left.layout.addWidget(self.layers)
        self.run=QPushButton("开始整册处理"); self.run.setObjectName("primary"); left.layout.addWidget(self.run); left.layout.addStretch(1)
        root.addWidget(left,4)
        right=Card("输出结构")
        for title,desc in [
            ("final/","最终高清页面"),("pages/*/project.json","配准、气泡、匹配与 QA 工程数据"),("mask_transfer_layer.png","旧中文版气泡/文本框迁移图层"),("mask_transfer_mask.png","实际覆盖高清母版的精确蒙版"),("editable.ora / .psd","可编辑分层输出")]:
            row=QHBoxLayout(); t=QLabel(title); t.setStyleSheet("font-weight:600;"); d=QLabel(desc); d.setObjectName("hint"); row.addWidget(t); row.addStretch(1); row.addWidget(d); right.layout.addLayout(row)
        right.layout.addStretch(1); root.addWidget(right,6)
        self.out.button.clicked.connect(lambda:self.window.choose_directory("output")); self.run.clicked.connect(self.window.run_book); self.debug.toggled.connect(self._sync); self.layers.toggled.connect(self._sync)
    def _sync(self): self.window.state.config.export.save_debug=self.debug.isChecked(); self.window.state.config.export.layer_bundle=self.layers.isChecked()
    def refresh(self): self.out.set_path(self.window.state.output_dir); self.debug.setChecked(self.window.state.config.export.save_debug); self.layers.setChecked(self.window.state.config.export.layer_bundle)


class StudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = StudioState()
        self.state.config.transfer.mode = "auto"
        self.worker: PipelineWorker | None = None
        self._worker_is_single_page = False
        self._worker_page_id = ""
        self.setWindowTitle(APP_NAME)
        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            min_w = min(1020, max(860, int(geo.width() * 0.72)))
            min_h = min(680, max(580, int(geo.height() * 0.72)))
            self.setMinimumSize(QSize(min_w, min_h))
            self.resize(min(1240, int(geo.width() * 0.94)), min(820, int(geo.height() * 0.90)))
        else:
            self.setMinimumSize(QSize(900, 600)); self.resize(1240, 820)
        self.setStyleSheet(STYLE)

        root = QWidget(); root.setObjectName("root"); self.setCentralWidget(root)
        outer = QVBoxLayout(root); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)

        top = QFrame(); top.setObjectName("topbar")
        tl = QHBoxLayout(top); tl.setContentsMargins(20,12,20,10); tl.setSpacing(10)
        tb = QVBoxLayout()
        title = QLabel(APP_NAME); title.setObjectName("appTitle")
        sub = QLabel("旧版中文 → 高清日文 · Direct 贴图 / 精准蒙版分离 · 页面级 QA · 本地处理"); sub.setObjectName("appSubtitle")
        tb.addWidget(title); tb.addWidget(sub); tl.addLayout(tb); tl.addStretch(1)
        self.stop_button = QPushButton("■ 停止")
        self.stop_button.setObjectName("stopTask")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("安全停止当前迁移任务。已完成页面不会被删除。")
        self.stop_button.clicked.connect(self.cancel_worker)
        tl.addWidget(self.stop_button)
        badge = QLabel(f"v{VERSION} · MAC · MPS"); badge.setObjectName("badge"); tl.addWidget(badge)
        outer.addWidget(top)

        nav = QFrame(); nav.setObjectName("navBar")
        nl = QHBoxLayout(nav); nl.setContentsMargins(16,4,16,3); nl.setSpacing(4)
        self.nav_group = QButtonGroup(self); self.nav_group.setExclusive(True)
        self.stack = QStackedWidget(); self.pages = []
        for label in ["页面管理", "识别与配准", "替换工作台", "出版输出"]:
            b = QPushButton(label); b.setObjectName("navButton"); b.setCheckable(True)
            self.nav_group.addButton(b); nl.addWidget(b)
            b.clicked.connect(lambda _=False, i=len(self.pages): self.show_page(i))
            self.pages.append(b)
        nl.addStretch(1); outer.addWidget(nav)

        self.project = ProjectPage(self); self.models = ModelPage(self); self.workbench = WorkbenchPage(self); self.export = ExportPage(self)
        for w in [self.project, self.models, self.workbench, self.export]: self.stack.addWidget(w)
        outer.addWidget(self.stack, 1); self.pages[0].setChecked(True)

        self.progress = QProgressBar(); self.progress.setRange(0,100); self.progress.setValue(0); self.progress.setMaximumWidth(190)
        self.statusBar().addPermanentWidget(self.progress); self.statusBar().showMessage("就绪")
        self.refresh_all()

    def _busy_running(self) -> bool:
        return bool(self.worker is not None and self.worker.isRunning())

    def _set_busy(self, active: bool | None = None):
        busy = self._busy_running() if active is None else bool(active)
        self.stop_button.setEnabled(busy)
        if hasattr(self, "project"):
            self.project.cancel.setEnabled(busy)
            self.project.run_page.setEnabled(not busy)
            self.project.run_book.setEnabled(not busy)
            self.project.pair_btn.setEnabled(not busy)
            self.project.apply_type.setEnabled(not busy)
            self.project.reset_type.setEnabled(not busy)
            self.project.page_type.setEnabled(not busy)
        if hasattr(self, "export"):
            self.export.run.setEnabled(not busy)

    def show_page(self, index: int):
        self.stack.setCurrentIndex(index)
        for i,b in enumerate(self.pages): b.setChecked(i == index)
        if index == 0: self.project.refresh()
        elif index == 1: self.models.refresh()
        elif index == 2: self.workbench.refresh()
        elif index == 3: self.export.refresh()

    # ---------- Page Manager persistence / marking ----------
    def _page_management_path(self) -> Path | None:
        if not self.state.output_dir:
            return None
        return Path(self.state.output_dir) / "page_management.json"

    def save_page_marks(self):
        path = self._page_management_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_json(path, marks_to_json(self.state.page_marks))
        except Exception as exc:
            self.statusBar().showMessage(f"页面标记保存失败：{exc}", 5000)

    def load_page_marks(self):
        path = self._page_management_path()
        if path is None or not path.exists():
            return
        try:
            loaded = marks_from_json(load_json(path))
        except Exception as exc:
            self.statusBar().showMessage(f"页面标记读取失败：{exc}", 5000)
            return
        if not self.state.pairs:
            self.state.page_marks.update(loaded)
            return
        valid_keys = {page_mark_key(p) for p in self.state.pairs}
        valid_names = {Path(p.target_path).name for p in self.state.pairs}
        for key, value in loaded.items():
            mark = PageMark.from_dict(value)
            if key in valid_keys or mark.target_name in valid_names:
                self.state.page_marks[key] = mark.to_dict()

    def page_mark_for_pair(self, pair: PagePair) -> PageMark:
        return resolve_mark(self.state.page_marks, pair)

    def mark_page_rows(self, rows: list[int], page_type: str):
        if self._busy_running():
            QMessageBox.information(self, "任务进行中", "请先停止或等待当前任务完成，再修改页面类型。")
            return
        changed = 0
        for row in sorted(set(rows)):
            if 0 <= row < len(self.state.pairs):
                pair = self.state.pairs[row]
                self.state.page_marks[page_mark_key(pair)] = manual_mark(pair, page_type).to_dict()
                changed += 1
        if changed:
            self.save_page_marks(); self.project._table_signature = None; self.project._thumb_signature = None; self.project.refresh()
            self.statusBar().showMessage(f"已手动标记 {changed} 页为“{page_type_label(page_type)}”", 3500)

    def reset_page_rows(self, rows: list[int]):
        if self._busy_running():
            QMessageBox.information(self, "任务进行中", "请先停止或等待当前任务完成，再修改页面类型。")
            return
        selected = []
        for row in sorted(set(rows)):
            if 0 <= row < len(self.state.pairs):
                pair = self.state.pairs[row]
                self.state.page_marks.pop(page_mark_key(pair), None)
                selected.append(row)
        if not selected:
            return
        self.save_page_marks(); self.project._table_signature = None; self.project._thumb_signature = None; self.project.refresh()
        self.statusBar().showMessage(f"已恢复 {len(selected)} 页为默认正文", 3000)


    # ---------- Project selection / pairing ----------
    def choose_directory(self, kind: str):
        title = {"source":"选择旧中文版目录", "target":"选择高清日文版目录", "output":"选择输出目录"}[kind]
        start = getattr(self.state, f"{kind}_dir", "") or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, title, start)
        if not path:
            return
        # Preserve in-memory marks if the user classified pages before choosing an
        # output folder.  Load an existing project state first, then let current
        # in-session marks win and persist the merged result.
        in_session_marks = dict(self.state.page_marks) if kind == "output" else {}
        setattr(self.state, f"{kind}_dir", path)
        if kind in {"source", "target"}:
            self.state.pairs = []; self.state.selected_index = 0; self.state.batch_status.clear()
            self.state.page_marks.clear(); self.state.unmatched_source.clear(); self.state.unmatched_target.clear()
        if kind in {"source", "target", "output"}:
            self.state.last_project = None; self.state.last_result_path = ""; self.state.projects_by_page.clear()
        if kind == "output":
            self.load_page_marks()
            self.state.page_marks.update(in_session_marks)
            if self.state.page_marks:
                self.save_page_marks()
        self.project._table_signature = None; self.project._thumb_signature = None
        self.refresh_all()

    def _pair_now(self) -> tuple[int, int]:
        s = self.state
        pairs, us, ut = pair_directories(s.source_dir, s.target_dir, s.config.pairing)
        s.pairs = pairs; s.unmatched_source = list(us); s.unmatched_target = list(ut); s.selected_index = 0
        # Pairing changes do not erase persisted manual labels for matching target
        # pages. Every new row resolves to the default content type.
        self.load_page_marks()
        self.project._table_signature = None; self.project._thumb_signature = None
        return len(us), len(ut)

    def auto_pair(self):
        s = self.state
        if not s.source_dir or not s.target_dir:
            QMessageBox.warning(self, "缺少输入", "请先选择旧中文版和高清日文版目录。")
            return
        if self._busy_running():
            QMessageBox.information(self, "任务进行中", "请先停止或等待当前任务完成。")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            us_count, ut_count = self._pair_now()
            counts = {"name":0, "order":0, "smart":0}
            for pair in s.pairs:
                method = pairing_method(pair); counts[method] = counts.get(method, 0) + 1
            self.statusBar().showMessage(
                f"页面配对完成 · {len(s.pairs)} 对（名称 {counts.get('name',0)} / 顺序 {counts.get('order',0)} / 智能 {counts.get('smart',0)}）· 未匹配 {us_count+ut_count}", 5000
            )
        except Exception as exc:
            QMessageBox.critical(self, "配对失败", str(exc)); return
        finally:
            QApplication.restoreOverrideCursor(); self.project.refresh()

    def set_selected_page(self, index: int, *, sync_table: bool = True):
        if not self.state.pairs:
            self.state.selected_index = 0; return
        idx = max(0, min(int(index), len(self.state.pairs)-1))
        self.state.selected_index = idx
        if hasattr(self, "project"):
            self.project.set_current_page(idx, sync_table=sync_table)
        self.refresh_preview()

    def current_pair(self):
        if not self.state.pairs: return None
        return self.state.pairs[max(0, min(self.state.selected_index, len(self.state.pairs)-1))]

    def _default_output(self):
        created = False
        if not self.state.output_dir:
            if self.state.target_dir: self.state.output_dir = str(Path(self.state.target_dir).parent / "MHD_Transfer_Output")
            else: self.state.output_dir = str(Path.home() / "MHD_Transfer_Output")
            created = True
        if created and self.state.page_marks:
            self.save_page_marks()
        return self.state.output_dir

    # ---------- Processing ----------
    def run_current_page(self):
        pair = self.current_pair()
        if pair is None:
            QMessageBox.information(self, "没有页面", "请先完成页面配对。"); return
        mark = self.page_mark_for_pair(pair)
        self._start_worker(PipelineWorker(
            config=self.state.config.model_copy(deep=True), pair=pair,
            page_mark=mark.to_dict(), output_dir=self._default_output(),
        ))

    def run_book(self):
        if not self.state.source_dir or not self.state.target_dir:
            QMessageBox.information(self, "没有项目", "请先选择两套页面目录。"); return
        if self._busy_running():
            QMessageBox.information(self, "处理中", "已有任务正在运行。"); return
        if not self.state.pairs:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self._pair_now()
            except Exception as exc:
                QMessageBox.critical(self, "配对失败", str(exc)); return
            finally:
                QApplication.restoreOverrideCursor(); self.project.refresh()
        if not self.state.pairs:
            QMessageBox.information(self, "没有可处理页面", "没有找到可配对的页面。"); return
        self._start_worker(PipelineWorker(
            config=self.state.config.model_copy(deep=True), source_dir=self.state.source_dir,
            target_dir=self.state.target_dir, output_dir=self._default_output(),
            pairs_override=list(self.state.pairs), page_marks=dict(self.state.page_marks),
        ))

    def _start_worker(self, worker: PipelineWorker):
        if self._busy_running():
            QMessageBox.information(self, "处理中", "已有任务正在运行。"); return
        self.worker = worker
        self._worker_is_single_page = worker.pair is not None
        self._worker_page_id = page_id_for_pair(worker.pair) if worker.pair is not None else ""
        self.progress.setRange(0,100); self.progress.setValue(0); self.statusBar().showMessage("正在处理…")
        self._set_busy(True)
        worker.progress.connect(self._worker_progress)
        worker.done.connect(self._worker_done)
        worker.failed.connect(self._worker_failed)
        worker.cancelled.connect(lambda: self.statusBar().showMessage("已停止，已完成页面已保留", 5000))
        worker.finished.connect(self._worker_finished)
        worker.start()

    def cancel_worker(self):
        requested = False
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_cancel(); requested = True
        if requested:
            self.stop_button.setEnabled(False); self.project.cancel.setEnabled(False)
            self.statusBar().showMessage("正在安全停止…")

    def _find_pair_row_by_target_name(self, name: str) -> int:
        for idx, pair in enumerate(self.state.pairs):
            if Path(pair.target_path).name == name:
                return idx
        return -1

    def _worker_progress(self, done, total, name, message, cache_hit):
        self.progress.setRange(0, max(1,total)); self.progress.setValue(done)
        text = str(message or "")
        if cache_hit: state = "缓存"
        elif "跳过" in text: state = "跳过"
        elif "正在" in text: state = "处理中"
        elif "Error" in text or "失败" in text: state = "失败"
        elif "取消" in text or "停止" in text: state = "已停止"
        else: state = "完成"
        row = self._find_pair_row_by_target_name(name)
        if 0 <= row < self.project.table.rowCount():
            self.project.table.setItem(row, 6, QTableWidgetItem(state))
            self.project.table.setItem(row, 7, QTableWidgetItem(text or "—"))
        self.state.batch_status[name] = (state, text or "—")
        self.statusBar().showMessage(f"{done}/{total} · {name} · {text}")

    def _merge_project_page_mark(self, page):
        meta = getattr(page, "meta", {}) or {}
        pm = meta.get("page_management")
        if not pm:
            return
        key = str(getattr(page, "page_id", "") or "")
        if key:
            # A user manual mark remains authoritative; pipeline auto classification
            # must not replace it after processing.
            existing = PageMark.from_dict(self.state.page_marks.get(key)) if key in self.state.page_marks else None
            incoming = PageMark.from_dict(pm)
            if existing is None or existing.origin != "manual" or incoming.origin == "manual":
                self.state.page_marks[key] = incoming.to_dict()

    def _worker_finished(self):
        self.progress.setRange(0,100)
        self._set_busy(None)

    def _worker_done(self, project, path):
        self.progress.setRange(0,100); self.progress.setValue(100)
        if hasattr(project, "page_id"):
            self.state.last_project = project
            self.state.projects_by_page[str(project.page_id)] = project
            self._merge_project_page_mark(project)
        elif hasattr(project, "pages"):
            self.state.last_project = None
            for page in list(getattr(project, "pages", []) or []):
                if getattr(page, "page_id", None): self.state.projects_by_page[str(page.page_id)] = page
                self._merge_project_page_mark(page)
        else:
            self.state.last_project = None
        self.save_page_marks(); self.project._table_signature = None; self.project._thumb_signature = None
        self.state.last_result_path = path if str(path).lower().endswith(".png") else ""
        cancelled = bool(getattr(project, "meta", {}).get("cancelled")) if hasattr(project, "meta") else False
        skipped = int((getattr(project, "meta", {}) or {}).get("skipped_count", 0)) if hasattr(project, "meta") else 0
        if cancelled:
            self.statusBar().showMessage("已停止，已完成页面已保留", 5000)
        elif skipped:
            self.statusBar().showMessage(f"处理完成 · 自动/手动跳过 {skipped} 页", 5000)
        else:
            self.statusBar().showMessage("处理完成", 5000)
        if getattr(self, "_worker_is_single_page", False) and not cancelled:
            wanted = getattr(self, "_worker_page_id", "")
            if wanted:
                for i,pair in enumerate(self.state.pairs):
                    if page_id_for_pair(pair) == wanted:
                        self.set_selected_page(i); break
            self.workbench.current_view = "result"
            for b,k in self.workbench.view_buttons: b.setChecked(k == "result")
            self.show_page(2)
        else:
            self.refresh_current_page()

    def _worker_failed(self, message):
        self.progress.setRange(0,100); self.progress.setValue(0)
        self.statusBar().showMessage("处理失败", 5000)
        QMessageBox.critical(self, "处理失败", message)

    def refresh_preview(self):
        if self.stack.currentIndex() == 2: self.workbench.refresh()

    def refresh_current_page(self):
        idx = self.stack.currentIndex()
        if idx == 0: self.project.refresh()
        elif idx == 1: self.models.refresh()
        elif idx == 2: self.workbench.refresh()
        elif idx == 3: self.export.refresh()

    def refresh_all(self):
        self.refresh_current_page()


def main():
    app=QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME); app.setOrganizationName("MangaHDTransfer")
    if sys.platform == "darwin":
        font=QFont("SF Pro Text"); font.setPointSize(12); app.setFont(font)
    win=StudioWindow(); win.show(); return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
