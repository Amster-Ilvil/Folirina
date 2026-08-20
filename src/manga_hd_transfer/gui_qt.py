from __future__ import annotations

import sys
import logging
import time
import shutil
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from PySide6.QtCore import Qt, QSize, QThread, QTimer, Signal, QRectF, QSettings, QUrl
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont, QImageReader, QImage, QIcon, QPen, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton, QToolButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QStackedWidget, QScrollArea,
    QFileDialog, QMessageBox, QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy,
    QGraphicsView, QGraphicsScene, QProgressBar, QSplitter, QButtonGroup, QRadioButton,
    QPlainTextEdit, QDialog, QListWidget, QListWidgetItem, QMenu, QSlider,
)

from .config import PipelineConfig
from .pipeline_worker import PipelineWorker
from .gui_workers import (
    PageActionWorker, ComponentProbeWorker, ModelDownloadWorker, DependencyInstallWorker,
    AutoPrepareModelsWorker, ModelNetworkProbeWorker,
)
from .version import __version__
from .gui_theme import (
    ACCENT, ACCENT_HOVER, BG, CARD, CARD_BLUE, TEXT, MUTED, MUTED_2,
    BORDER, BORDER_STRONG, BLUE_SOFT, GREEN, GREEN_SOFT, ORANGE, ORANGE_SOFT,
    RED, RED_SOFT, normalize_theme, style_for_theme,
)
from .gui_responsive import ResponsiveCanvasView
from .app_logging import configure_application_logging, install_exception_hooks, runtime_log_dir
from .platform_support import desktop_platform_badge
from .gui_components import (
    _configure_responsive_dialog, _fit_scene_rect, StableThumbnailList, ImageView, Card, PageHero, OptionRow,
    PathRow, ZoomPreviewView, StableComboBox, WorkbenchPageRail,
)
from .studio_state import StudioState
from .studio_project_page import ProjectPage, PagePreviewDialog
from .studio_model_page import ModelPage
from .studio_export_page import ExportPage
from .studio_settings_page import SettingsPage
from .models import PagePair
from .pairing import pair_directories, pairing_method
from .model_downloads import apply_config_updates, model_home, model_local_paths, import_builtin_model, apply_download_network_settings, paddle_profile_marker_status
from .dependency_install import missing_dependency_modules, dependency_summary
from .paddle_profiles import PADDLE_MODEL_PROFILES, profile_label, backend_profile_key
from .runtime_preflight import plan_runtime_requirements, pending_model_requirements
from .gui_processing_policy import (
    compute_busy_state, classify_progress_state, worker_config_snapshot, completion_message,
)
from .io_utils import load_json, save_json, write_image
from .review_history import record_review_state, undo_review_state, redo_review_state, review_history_counts
from .font_catalog import discover_fonts
from .review_apply import apply_review_page, generate_remove_text_preview, apply_target_layer_erase_review, reset_target_layer_erase_review, apply_target_layer_restore_review, reset_target_layer_restore_review, apply_manual_force_transfer_review, reset_manual_force_transfer_review, manual_force_auto_evidence_masks
from .manual_effect import map_target_bbox_to_source, build_manual_effect_masks, build_reveal_seed_mask, estimate_source_background, composite_source_text_delta, clean_manual_target_text
from .modes.reletter import ocr_edit_blocks as _reletter_ocr_edit_blocks
from .modes.reletter import ocr_edit_render as _reletter_ocr_edit_render
from .modes.hybrid import ocr_edit_blocks as _hybrid_ocr_edit_blocks
from .modes.hybrid import ocr_edit_render as _hybrid_ocr_edit_render


def _ocr_mode_modules(mode: str):
    key = str(mode or "").strip().lower()
    if key == "hybrid":
        return _hybrid_ocr_edit_blocks, _hybrid_ocr_edit_render
    if key == "reletter":
        return _reletter_ocr_edit_blocks, _reletter_ocr_edit_render
    return None, None


def is_ocr_edit_mode(mode: str) -> bool:
    blocks, _ = _ocr_mode_modules(mode)
    return blocks is not None and bool(blocks.is_ocr_edit_mode(mode))


def ocr_edit_scope(mode: str):
    blocks, _ = _ocr_mode_modules(mode)
    return blocks.ocr_edit_scope(mode) if blocks is not None else ""


def load_ocr_blocks(page_dir, mode: str):
    blocks, _ = _ocr_mode_modules(mode)
    return blocks.load_ocr_blocks(page_dir, mode) if blocks is not None else []


def upsert_ocr_block(page_dir, mode: str, row):
    blocks, _ = _ocr_mode_modules(mode)
    if blocks is None:
        raise RuntimeError(f"OCR block editor unavailable for mode: {mode}")
    return blocks.upsert_ocr_block(page_dir, mode, row)


def delete_ocr_block(page_dir, mode: str, block_id):
    blocks, _ = _ocr_mode_modules(mode)
    if blocks is None:
        raise RuntimeError(f"OCR block editor unavailable for mode: {mode}")
    return blocks.delete_ocr_block(page_dir, mode, block_id)


def save_ocr_blocks(page_dir, mode: str, rows):
    blocks, _ = _ocr_mode_modules(mode)
    if blocks is None:
        raise RuntimeError(f"OCR block editor unavailable for mode: {mode}")
    return blocks.save_ocr_blocks(page_dir, mode, rows)


def recognize_manual_ocr_block(project, source_path, target_path, bbox, config, *, existing=None):
    mode = str((project.get("meta") or {}).get("transfer_mode") or getattr(config.transfer, "mode", "") or "").strip().lower()
    blocks, _ = _ocr_mode_modules(mode)
    if blocks is None:
        raise RuntimeError(f"OCR block editor unavailable for mode: {mode}")
    return blocks.recognize_manual_ocr_block(project, source_path, target_path, bbox, config, existing=existing)


def apply_ocr_edit_blocks(page_dir, project, cfg):
    mode = str((project.get("meta") or {}).get("transfer_mode") or getattr(cfg.transfer, "mode", "") or "").strip().lower()
    _, renderer = _ocr_mode_modules(mode)
    if renderer is None:
        raise RuntimeError(f"OCR edit render unavailable for mode: {mode}")
    return renderer.apply_ocr_edit_blocks(page_dir, project, cfg)


def reset_ocr_edit_blocks(page_dir, project, cfg):
    mode = str((project.get("meta") or {}).get("transfer_mode") or getattr(cfg.transfer, "mode", "") or "").strip().lower()
    _, renderer = _ocr_mode_modules(mode)
    if renderer is None:
        raise RuntimeError(f"OCR edit reset unavailable for mode: {mode}")
    return renderer.reset_ocr_edit_blocks(page_dir, project, cfg)

from .workspace import page_id_for_pair, resolve_page_workspace
from .workspace_guard import PageRunGuard
from .workspace_cleanup import cleanup_output_workspace
from .mode_contracts import clear_stale_mode_outputs
from .session_restore import scan_existing_results
from .schema_compat import as_dict, as_dict_rows, as_list, normalize_project, normalize_overrides, normalize_review_applied
from .result_state import commit_reviewed_result
from .manual_review_service import commit_manual_effect
from .page_management import (
    PAGE_TYPE_INFO, MANUAL_PAGE_TYPES, PageMark,
    default_mark, manual_mark, marks_from_json, marks_to_json, page_mark_key,
    page_type_color, page_type_label, resolve_mark,
)

APP_NAME = "Folirina"


def _application_icon() -> QIcon:
    """Load the bundled application icon without relying on the working directory."""
    path = Path(__file__).with_name("folirina_icon.png")
    return QIcon(str(path)) if path.is_file() else QIcon()
VERSION = __version__
QComboBox = StableComboBox

logger = logging.getLogger(__name__)
_THEME_SETTING_KEY = "ui/theme"
_SETTINGS_ORG = "Folirina"
_SETTINGS_APP = "Folirina"
_LEGACY_SETTINGS_ORG = "MangaHDTransfer"
_LEGACY_SETTINGS_APP = "MangaHDTransferStudio"


def _app_settings() -> QSettings:
    """Folirina settings with one-way migration from the pre-rename application."""
    current = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    if not current.contains(_THEME_SETTING_KEY):
        legacy = QSettings(_LEGACY_SETTINGS_ORG, _LEGACY_SETTINGS_APP)
        if legacy.contains(_THEME_SETTING_KEY):
            current.setValue(_THEME_SETTING_KEY, legacy.value(_THEME_SETTING_KEY))
    return current


def _saved_theme_name() -> str:
    return normalize_theme(_app_settings().value(_THEME_SETTING_KEY, "light"))


# KCC-Kindle-CHS inspired palette, now available in light and dark variants.








@dataclass













class RunLogDialog(QDialog):
    def __init__(self, page_root: str | Path, parent=None):
        super().__init__(parent)
        self.page_root = Path(page_root)
        self.setWindowTitle("当前页运行日志")
        _configure_responsive_dialog(self, (980, 720), (720, 520))
        root = QVBoxLayout(self); root.setContentsMargins(14,14,14,14); root.setSpacing(10)
        head = QHBoxLayout()
        title = QLabel(f"页面日志 · {self.page_root.name}"); title.setObjectName("sectionTitle")
        self.state_label = QLabel(); self.state_label.setObjectName("hint")
        head.addWidget(title); head.addStretch(1); head.addWidget(self.state_label)
        root.addLayout(head)
        self.text = QPlainTextEdit(); self.text.setReadOnly(True); self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        root.addWidget(self.text, 1)
        row = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新"); self.open_btn = QPushButton("打开日志目录"); self.export_btn = QPushButton("导出诊断包…"); self.close_btn = QPushButton("关闭")
        row.addWidget(self.refresh_btn); row.addWidget(self.open_btn); row.addWidget(self.export_btn); row.addStretch(1); row.addWidget(self.close_btn)
        root.addLayout(row)
        self.refresh_btn.clicked.connect(self.refresh)
        self.open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.page_root))))
        self.export_btn.clicked.connect(self.export_diagnostics)
        self.close_btn.clicked.connect(self.accept)
        self.refresh()

    def export_diagnostics(self):
        default_name = f"Folirina-diagnostics-{self.page_root.name}.zip"
        path, _ = QFileDialog.getSaveFileName(self, "导出当前页诊断包", str(self.page_root.parent / default_name), "ZIP (*.zip)")
        if not path:
            return
        if not path.lower().endswith(".zip"):
            path += ".zip"
        wanted = (
            "run.log", "run_trace.jsonl", "last_run_state.json", "project.json", "qa.json",
            "transfer_audit.json", "clear_mask.png", "target_clear_mask.png", "review_preview.png",
            "mask_transfer.json", "hybrid_transfer.json", "reletter.json", "direct_patch.json", "transparent_bubble_reveal.json",
        )
        try:
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name in wanted:
                    src = self.page_root / name
                    if src.exists() and src.is_file():
                        zf.write(src, arcname=name)
            QMessageBox.information(self, "诊断包已导出", f"已生成：\n{path}\n\n可直接把这个 ZIP 发给我分析。")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def refresh(self):
        state = {}
        try:
            p = self.page_root / "last_run_state.json"
            if p.exists():
                value = load_json(p); state = value if isinstance(value, dict) else {}
        except Exception:
            state = {}
        status = str(state.get("status") or "未知")
        mode = str(state.get("mode") or "—")
        strategy = str(state.get("selected_strategy") or "—")
        self.state_label.setText(f"{status} · 模式 {mode} · 策略 {strategy}")
        parts = []
        log_path = self.page_root / "run.log"
        if log_path.exists():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                parts.append("\n".join(lines[-1200:]))
            except Exception as exc:
                parts.append(f"无法读取 run.log：{exc}")
        else:
            parts.append("当前页还没有 run.log。重新处理本页后会自动生成。")
        if state:
            parts.append("\n\n--- last_run_state.json ---\n" + __import__('json').dumps(state, ensure_ascii=False, indent=2))
        self.text.setPlainText("".join(parts))
        cursor = self.text.textCursor(); cursor.movePosition(cursor.MoveOperation.End); self.text.setTextCursor(cursor)






class MaskPaintView(QGraphicsView):
    """Lightweight clear-mask overlay editor for page-local review.

    Left drag uses the selected add/erase mode; right drag is always an erase
    shortcut. The editor never touches source/target originals directly; callers
    decide which review-mask file is persisted.
    """
    mask_changed = Signal()

    def __init__(self, image_path: str | Path, mask: Any, parent=None, *, reference_mask: Any | None = None, reference_original_mask: Any | None = None):
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
        if reference_mask is None:
            self.reference_mask = _np.zeros((h, w), dtype=_np.uint8)
        else:
            ref = _np.asarray(reference_mask, dtype=_np.uint8)
            if ref.shape != (h, w):
                raise ValueError("OCR/自动检测参考蒙版尺寸与目标图片不一致")
            self.reference_mask = (ref > 0).astype(_np.uint8) * 255
        if reference_original_mask is None:
            self.reference_original_mask = self.reference_mask.copy()
        else:
            ref0 = _np.asarray(reference_original_mask, dtype=_np.uint8)
            if ref0.shape != (h, w):
                raise ValueError("原始 OCR/自动检测参考蒙版尺寸与目标图片不一致")
            self.reference_original_mask = (ref0 > 0).astype(_np.uint8) * 255
        self.brush_size = 24
        self.paint_mode = "add"
        self.edit_layer = "manual"
        self._painting = False; self._erase = False; self._last = None
        self._panning = False; self._pan_last = None
        self._auto_fit = True; self._fit_pending = False
        self._overlay_item = self._scene.addPixmap(QPixmap())
        self._scene.setSceneRect(0, 0, w, h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._refresh_overlay(); self.fit_to_window()

    def set_paint_mode(self, mode: str):
        self.paint_mode = "erase" if str(mode) == "erase" else "add"
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def set_edit_layer(self, layer: str):
        self.edit_layer = "reference" if str(layer) == "reference" else "manual"
        self._refresh_overlay()

    def active_mask(self):
        return self.reference_mask if self.edit_layer == "reference" else self.mask

    def fit_to_active_mask(self):
        active = self.active_mask()
        ys, xs = self._np.where(active > 0)
        if xs.size == 0:
            self.fit_to_window(); return
        self.fit_to_rect([int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)])

    def set_preview_image(self, image):
        """Replace only the displayed base while preserving the original cache."""
        if image is None:
            self._base_item.setPixmap(QPixmap.fromImage(self._base_qimage)); return
        arr=self._np.asarray(image,dtype=self._np.uint8)
        if arr.shape[:2] != self.mask.shape or arr.ndim != 3 or arr.shape[2] != 3:
            return
        rgb=self._cv2.cvtColor(arr,self._cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]
        q=QImage(rgb.data,w,h,int(rgb.strides[0]),QImage.Format.Format_RGB888).copy()
        self._base_item.setPixmap(QPixmap.fromImage(q))

    def fit_to_rect(self, bbox, *, margin_ratio: float = 0.18):
        if not bbox or len(bbox)!=4:
            self.fit_to_window(); return
        x0,y0,x1,y1=[float(v) for v in bbox]
        if x1<=x0 or y1<=y0:
            self.fit_to_window(); return
        pad=max(18.0,max(x1-x0,y1-y0)*float(margin_ratio))
        rect=QRectF(x0-pad,y0-pad,(x1-x0)+2*pad,(y1-y0)+2*pad).intersected(self._scene.sceneRect())
        self._auto_fit=False; self.resetTransform(); self.fitInView(rect,Qt.AspectRatioMode.KeepAspectRatio)

    def fit_to_mask(self):
        ys,xs=self._np.where(self.mask>0)
        if xs.size==0:
            self.fit_to_window(); return
        self.fit_to_rect([int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)])

    def _refresh_overlay(self):
        h, w = self.mask.shape
        rgba = self._np.zeros((h, w, 4), dtype=self._np.uint8)
        manual = self.mask > 0
        reference = self.reference_mask > 0
        ref_only = reference & (~manual)
        overlap = reference & manual
        # Cyan/blue = OCR or automatic detector reference; red = reviewer paint;
        # amber = overlap. Both layers are editable; the active layer is rendered
        # slightly stronger so it is obvious which pixels the brush will change.
        ref_alpha = 132 if self.edit_layer == "reference" else 82
        manual_alpha = 142 if self.edit_layer == "manual" else 98
        rgba[ref_only, 0] = 70; rgba[ref_only, 1] = 165; rgba[ref_only, 2] = 235; rgba[ref_only, 3] = ref_alpha
        rgba[manual, 0] = 230; rgba[manual, 1] = 70; rgba[manual, 2] = 85; rgba[manual, 3] = manual_alpha
        rgba[overlap, 0] = 245; rgba[overlap, 1] = 165; rgba[overlap, 2] = 45; rgba[overlap, 3] = 150
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
        self._painting = True
        self._erase = (event.button() == Qt.MouseButton.RightButton) or (event.button() == Qt.MouseButton.LeftButton and self.paint_mode == "erase")
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
        active = self.active_mask()
        self._cv2.line(active, self._last, now, int(val), max(1, int(self.brush_size)), lineType=self._cv2.LINE_AA)
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
            active = self.active_mask()
            self._cv2.circle(active, (x, y), max(1, int(self.brush_size // 2)), 0 if self._erase else 255, -1, lineType=self._cv2.LINE_AA)
            self._refresh_overlay(); self.mask_changed.emit()


class MaskEditorDialog(QDialog):
    def __init__(self, image_path: str | Path, initial_mask: Any, parent=None, *, title: str = "清除蒙版编辑器", hint_text: str | None = None, save_label: str = "保存蒙版", preview_fn=None, reference_mask: Any | None = None, reference_original_mask: Any | None = None, reference_label: str = "OCR / 自动检测", auto_assist_default: bool = True):
        super().__init__(parent); self.setWindowTitle(title)
        _configure_responsive_dialog(self, (1280, 900), (850, 600))
        root = QVBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(8)
        hint = QLabel(hint_text or "红色区域 = 将被清除的日文。左键拖动增加，右键拖动擦除；滚轮缩放。只修改蒙版，不会直接破坏原图。")
        hint.setObjectName("hint"); hint.setWordWrap(True); root.addWidget(hint)
        self.view = MaskPaintView(image_path, initial_mask, self, reference_mask=reference_mask, reference_original_mask=reference_original_mask); root.addWidget(self.view, 1)
        self._reference_label = str(reference_label or "OCR / 自动检测")
        self._preview_fn = preview_fn
        has_reference = bool(self.view.reference_mask.any() or self.view.reference_original_mask.any())
        if has_reference:
            layer_row = QHBoxLayout(); layer_row.addWidget(QLabel("编辑层"))
            self.layer_manual = QPushButton("人工蒙版")
            self.layer_reference = QPushButton(f"{self._reference_label} 蒙版")
            self.layer_manual.setCheckable(True); self.layer_reference.setCheckable(True); self.layer_manual.setChecked(True)
            self.layer_manual.setObjectName("softPrimary"); self.layer_reference.setObjectName("segmented")
            self.layer_group = QButtonGroup(self); self.layer_group.setExclusive(True); self.layer_group.addButton(self.layer_manual); self.layer_group.addButton(self.layer_reference)
            self.layer_hint = QLabel("当前画笔修改红色人工层")
            self.layer_hint.setObjectName("quiet")
            layer_row.addWidget(self.layer_manual); layer_row.addWidget(self.layer_reference); layer_row.addWidget(self.layer_hint, 1); root.addLayout(layer_row)
            self.layer_manual.clicked.connect(lambda: self._set_layer("manual")); self.layer_reference.clicked.connect(lambda: self._set_layer("reference"))
        else:
            self.layer_manual = None; self.layer_reference = None; self.layer_hint = None
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("画笔模式"))
        self.paint_add = QPushButton("涂抹蒙版")
        self.paint_erase = QPushButton("消除蒙版")
        self.paint_add.setCheckable(True); self.paint_erase.setCheckable(True)
        self.paint_add.setChecked(True)
        self.paint_add.setObjectName("softPrimary"); self.paint_erase.setObjectName("segmented")
        self.paint_modes = QButtonGroup(self); self.paint_modes.setExclusive(True)
        self.paint_modes.addButton(self.paint_add); self.paint_modes.addButton(self.paint_erase)
        self.mode_hint = QLabel("左键：涂抹 · 右键：临时擦除")
        self.mode_hint.setObjectName("quiet")
        mode_row.addWidget(self.paint_add); mode_row.addWidget(self.paint_erase); mode_row.addWidget(self.mode_hint, 1)
        self.auto_assist = QCheckBox(f"处理时使用 {self._reference_label} 辅助")
        self.auto_assist.setChecked(bool(auto_assist_default))
        self.auto_assist.setEnabled(bool(has_reference))
        self.auto_assist.setVisible(bool(has_reference))
        self.auto_assist.setToolTip("仅在人工涂抹附近使用 OCR/自动检测到的完整文字组；不会把整页自动结果无条件重新处理。")
        mode_row.addWidget(self.auto_assist)
        root.addLayout(mode_row)
        brush_row = QHBoxLayout(); brush_row.addWidget(QLabel("画笔大小"))
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(4, 120); self.slider.setValue(24)
        self.size_label = QLabel("24 px"); brush_row.addWidget(self.slider, 1); brush_row.addWidget(self.size_label); root.addLayout(brush_row)
        action_row = QHBoxLayout()
        self.focus_button=QPushButton("聚焦当前层"); self.fit_button = QPushButton("查看整页"); self.clear_button = QPushButton("清空当前层"); self.import_reference_button = QPushButton(f"复制 {self._reference_label} → 人工层"); self.import_reference_button.setEnabled(has_reference); self.import_reference_button.setVisible(has_reference); self.reset_reference_button = QPushButton("恢复自动原始") if has_reference else None; self.save_button = QPushButton(save_label); self.save_button.setObjectName("primary"); self.cancel_button = QPushButton("取消")
        self.import_reference_button.setToolTip("将当前蓝色 OCR/自动蒙版复制到红色人工层；蓝色层本身现在也可以直接涂抹或擦除。")
        action_row.addWidget(self.focus_button); action_row.addWidget(self.fit_button); action_row.addWidget(self.clear_button); action_row.addWidget(self.import_reference_button);
        if self.reset_reference_button is not None: action_row.addWidget(self.reset_reference_button)
        action_row.addStretch(1); action_row.addWidget(self.save_button); action_row.addWidget(self.cancel_button); root.addLayout(action_row)
        self.slider.valueChanged.connect(self._brush); self.fit_button.clicked.connect(self.view.fit_to_window); self.focus_button.clicked.connect(self.view.fit_to_active_mask)
        self.paint_add.clicked.connect(lambda: self._set_mode("add")); self.paint_erase.clicked.connect(lambda: self._set_mode("erase"))
        self.clear_button.clicked.connect(self._clear); self.import_reference_button.clicked.connect(self._import_reference); self.save_button.clicked.connect(self.accept); self.cancel_button.clicked.connect(self.reject)
        if self.reset_reference_button is not None: self.reset_reference_button.clicked.connect(self._reset_reference)
        if self._preview_fn is not None:
            self.view.mask_changed.connect(self._refresh_live_preview); self._refresh_live_preview()
        QTimer.singleShot(0, self.view.fit_to_active_mask if bool(self.view.mask.any()) else self.view.fit_to_window)

    def _set_layer(self, layer: str):
        ref = str(layer) == "reference"
        self.view.set_edit_layer("reference" if ref else "manual")
        if self.layer_manual is not None: self.layer_manual.setChecked(not ref)
        if self.layer_reference is not None: self.layer_reference.setChecked(ref)
        if self.layer_hint is not None:
            self.layer_hint.setText("当前画笔直接修改蓝色 OCR/自动蒙版" if ref else "当前画笔修改红色人工层")
        self.clear_button.setText("清空 OCR/自动蒙版" if ref else "清空人工蒙版")

    def _set_mode(self, mode: str):
        erase = str(mode) == "erase"
        self.paint_add.setChecked(not erase); self.paint_erase.setChecked(erase)
        self.view.set_paint_mode("erase" if erase else "add")
        self.mode_hint.setText("左键：消除蒙版 · 右键：消除" if erase else "左键：涂抹蒙版 · 右键：临时擦除")

    def _brush(self, value: int):
        self.view.brush_size = int(value); self.size_label.setText(f"{int(value)} px")

    def _clear(self):
        self.view.active_mask()[:] = 0; self.view._refresh_overlay(); self._refresh_live_preview()

    def _reset_reference(self):
        self.view.reference_mask[:] = self.view.reference_original_mask
        self.view._refresh_overlay(); self._refresh_live_preview(); self.view.fit_to_active_mask()

    def _import_reference(self):
        if not bool(self.view.reference_mask.any()):
            return
        self.view.mask[:] = self.view._np.maximum(self.view.mask, self.view.reference_mask)
        self.view._refresh_overlay(); self._refresh_live_preview(); self.view.fit_to_mask()

    def _refresh_live_preview(self):
        if self._preview_fn is None: return
        try: self.view.set_preview_image(self._preview_fn(self.view.mask.copy()))
        except Exception: self.view.set_preview_image(None)

    def result_mask(self):
        return self.view.mask.copy()

    def result_reference_mask(self):
        return self.view.reference_mask.copy()

    def reference_changed(self) -> bool:
        return not bool(self.view._np.array_equal(self.view.reference_mask, self.view.reference_original_mask))

    def use_auto_evidence(self) -> bool:
        return bool(self.auto_assist.isChecked() and self.auto_assist.isEnabled())



class ManualTextMaskPreviewDialog(QDialog):
    """Read-only preview of the *effective* manual white-bubble masks."""

    def __init__(self, target_path: str | Path, source_mask: Any, target_clear_mask: Any, diagnostics: dict[str, Any] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("白气泡 · 实际生效文字 Mask")
        _configure_responsive_dialog(self, (1020, 820), (720, 520))
        import cv2
        import numpy as np
        target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if target is None:
            raise ValueError(f"无法读取目标图片：{Path(target_path).name}")
        src = (np.asarray(source_mask) > 0)
        clr = (np.asarray(target_clear_mask) > 0)
        if src.shape != target.shape[:2] or clr.shape != target.shape[:2]:
            raise ValueError("白气泡文字 Mask 与目标页尺寸不一致")
        overlay = target.astype(np.float32)
        # BGR display overlay: green = SOURCE Chinese write mask; red = TARGET
        # Japanese clear mask; overlap = amber. Only visualization, never commit.
        green = np.array([80, 220, 80], np.float32)
        red = np.array([70, 70, 235], np.float32)
        amber = np.array([40, 190, 245], np.float32)
        only_src = src & ~clr
        only_clr = clr & ~src
        both = src & clr
        overlay[only_src] = overlay[only_src] * 0.36 + green * 0.64
        overlay[only_clr] = overlay[only_clr] * 0.36 + red * 0.64
        overlay[both] = overlay[both] * 0.32 + amber * 0.68
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        q = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(q)

        root = QVBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(8)
        hint = QLabel("绿色 = 最终会写入的 SOURCE 中文文字；红色 = 最终会清除的 TARGET 日文；橙色 = 两者重叠。这里显示的是经过 safe inset、边框环剥离、文字连通域过滤和 X/Y 微调后的真实生效 Mask，不显示选框本身。")
        hint.setObjectName("hint"); hint.setWordWrap(True); root.addWidget(hint)
        self.view = QGraphicsView(); self.scene = QGraphicsScene(self.view); self.view.setScene(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True); self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.item = self.scene.addPixmap(pix); self.scene.setSceneRect(self.item.boundingRect()); root.addWidget(self.view, 1)
        diag = dict(diagnostics or {})
        src_px = int(np.count_nonzero(src)); clr_px = int(np.count_nonzero(clr))
        inset = int(diag.get("container_border_inset_px", 0) or 0)
        stats = QLabel(f"SOURCE 写入：{src_px:,} px · TARGET 清除：{clr_px:,} px · 安全内缩：{inset}px")
        stats.setObjectName("quiet"); root.addWidget(stats)
        actions = QHBoxLayout(); fit = QPushButton("适合窗口"); close = QPushButton("关闭"); close.setObjectName("primary")
        actions.addWidget(fit); actions.addStretch(1); actions.addWidget(close); root.addLayout(actions)
        fit.clicked.connect(self.fit_to_window); close.clicked.connect(self.accept)
        QTimer.singleShot(0, self.fit_to_window)

    def fit_to_window(self):
        self.view.resetTransform(); self.view.fitInView(_fit_scene_rect(self.scene), Qt.AspectRatioMode.KeepAspectRatio)


class RevealMaskDialog(QDialog):
    """Brush editor that previews target cleanup + aligned Chinese glyph reveal live."""

    def __init__(self, target_path: str | Path, aligned_source: Any, source_mask: Any, target_clear_mask: Any, initial_mask: Any, parent=None, *, focus_bbox=None):
        super().__init__(parent); self.setWindowTitle("擦除显字编辑器")
        _configure_responsive_dialog(self, (1420, 940), (900, 620))
        import cv2
        import numpy as np
        self._cv2 = cv2; self._np = np
        self._target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if self._target is None:
            raise ValueError(f"无法读取目标图片：{Path(target_path).name}")
        self._source = np.asarray(aligned_source, dtype=np.uint8)
        self._source_mask = (np.asarray(source_mask) > 0).astype(np.uint8) * 255
        self._clear_mask = (np.asarray(target_clear_mask) > 0).astype(np.uint8) * 255
        self._focus_bbox = list(focus_bbox or [])
        if self._source.shape[:2] != self._target.shape[:2] or self._source_mask.shape != self._target.shape[:2] or self._clear_mask.shape != self._target.shape[:2]:
            raise ValueError("擦除显字图层尺寸不一致")
        self._cleaned, self._cleanup_diag = clean_manual_target_text(self._target, self._clear_mask, bbox=self._focus_bbox)
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
        self.focus_button=QPushButton("聚焦选区"); self.fit_button=QPushButton("查看整页"); self.auto_button=QPushButton("恢复自动建议"); self.clear_button=QPushButton("全部恢复日文"); self.save_button=QPushButton("保存擦除显字"); self.save_button.setObjectName("primary"); self.cancel_button=QPushButton("取消")
        action_row.addWidget(self.focus_button); action_row.addWidget(self.fit_button); action_row.addWidget(self.auto_button); action_row.addWidget(self.clear_button); action_row.addStretch(1); action_row.addWidget(self.save_button); action_row.addWidget(self.cancel_button); root.addLayout(action_row)
        self._auto_seed=(np.asarray(initial_mask)>0).astype(np.uint8)*255
        self.slider.valueChanged.connect(self._brush); self.fit_button.clicked.connect(self.view.fit_to_window); self.focus_button.clicked.connect(self._focus_selection)
        self.auto_button.clicked.connect(self._restore_auto); self.clear_button.clicked.connect(self._clear); self.save_button.clicked.connect(self.accept); self.cancel_button.clicked.connect(self.reject)
        self.view.mask_changed.connect(self._refresh_preview)
        self._brush(32); self._refresh_preview(); QTimer.singleShot(0, self._focus_selection if len(self._focus_bbox)==4 else self.view.fit_to_window)

    def _focus_selection(self):
        if len(self._focus_bbox)==4:
            self.view.fit_to_rect(self._focus_bbox, margin_ratio=0.30)
        else:
            self.view.fit_to_mask()

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



class OCRBlockEditorDialog(QDialog):
    """Manual ROI OCR + per-block typography editor.

    This dialog is hard-gated to the two OCR product flows. It stores its state
    under ``ocr_edit/mask_ocr`` or ``ocr_edit/ocr_reletter`` and never writes
    Direct / pure Mask / Reveal artifacts.
    """

    def __init__(self, page_dir: str | Path, source_path: str | Path, target_path: str | Path,
                 project: dict[str, Any], config: PipelineConfig, mode: str, parent=None):
        super().__init__(parent)
        if not is_ocr_edit_mode(mode):
            raise ValueError("人工 OCR 文本块只在 精准蒙版+OCR / OCR重排 中可用。")
        self.page_dir=Path(page_dir); self.source_path=Path(source_path); self.target_path=Path(target_path)
        self.project=dict(project or {}); self.config=config.model_copy(deep=True); self.mode=str(mode)
        self._blocks=load_ocr_blocks(self.page_dir,self.mode); self._current_id=""
        scope=ocr_edit_scope(self.mode)
        self.setWindowTitle("人工 OCR 文本块 · " + ("精准蒙版+OCR" if scope=="mask_ocr" else "OCR重排"))
        _configure_responsive_dialog(self,(1560,980),(1040,700))
        root=QVBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(8)
        hint=QLabel("在右侧 TARGET 上拖框；“重新 OCR”只识别这个 ROI。中文内容从配准后的 SOURCE 区域读取，TARGET OCR 只用于定位需要清除的日文。字体、字号、方向、断句和排版属于当前文本块，不会修改其他模式。")
        hint.setWordWrap(True); hint.setObjectName("hint"); root.addWidget(hint)
        top=QHBoxLayout(); self.block_combo=QComboBox(); self.block_combo.addItem("新建 OCR 文本块","")
        self.new_btn=QPushButton("新建"); self.delete_btn=QPushButton("删除"); self.delete_btn.setObjectName("dangerCompact")
        top.addWidget(QLabel("文本块")); top.addWidget(self.block_combo,1); top.addWidget(self.new_btn); top.addWidget(self.delete_btn); root.addLayout(top)
        split=QSplitter(Qt.Orientation.Horizontal)
        left=QFrame(); left.setObjectName("card"); ll=QVBoxLayout(left); ll.setContentsMargins(8,8,8,8); ll.addWidget(QLabel("SOURCE 中文 · 自动映射 ROI")); self.source_view=RegionSelectView(self.source_path,editable=False,parent=self); ll.addWidget(self.source_view,1)
        right=QFrame(); right.setObjectName("card"); rl=QVBoxLayout(right); rl.setContentsMargins(8,8,8,8); rl.addWidget(QLabel("TARGET 高清日文 · 拖框选择 / 调整 ROI")); self.target_view=RegionSelectView(self.target_path,editable=True,parent=self); rl.addWidget(self.target_view,1)
        split.addWidget(left); split.addWidget(right); split.setStretchFactor(0,1); split.setStretchFactor(1,1); root.addWidget(split,1)
        panel=QFrame(); panel.setObjectName("selectionPanel"); pl=QVBoxLayout(panel); pl.setContentsMargins(10,9,10,9); pl.setSpacing(7)
        ocrrow=QHBoxLayout(); self.ocr_btn=QPushButton("重新 OCR 当前选框"); self.ocr_btn.setObjectName("softPrimary"); self.ocr_status=QLabel("先框选区域"); self.ocr_status.setObjectName("hint"); ocrrow.addWidget(self.ocr_btn); ocrrow.addWidget(self.ocr_status,1); pl.addLayout(ocrrow)
        self.text=QPlainTextEdit(); self.text.setPlaceholderText("OCR 结果可直接编辑；重新 OCR 时保留当前字体和排版设置。")
        self.text.setMaximumHeight(105); pl.addWidget(self.text)
        r1=QHBoxLayout(); self.orientation=QComboBox(); self.orientation.addItem("自动","auto"); self.orientation.addItem("竖排","vertical"); self.orientation.addItem("横排","horizontal")
        self.break_mode=QComboBox(); self.break_mode.addItem("智能断句","smart"); self.break_mode.addItem("均衡断句","balanced"); self.break_mode.addItem("保留源换行","source")
        self.layout_mode=QComboBox(); self.layout_mode.addItem("智能缩放","smart_scaling"); self.layout_mode.addItem("严格字号","strict"); self.layout_mode.addItem("填充文本框","balloon_fill")
        r1.addWidget(QLabel("方向")); r1.addWidget(self.orientation); r1.addWidget(QLabel("断句")); r1.addWidget(self.break_mode); r1.addWidget(QLabel("排版")); r1.addWidget(self.layout_mode); r1.addStretch(1); pl.addLayout(r1)
        r2=QHBoxLayout(); self.font=QLineEdit(); self.font.setPlaceholderText("留空使用全局字体；也支持字体链")
        self.font_pick=QPushButton("选择字体…"); r2.addWidget(QLabel("字体")); r2.addWidget(self.font,1); r2.addWidget(self.font_pick); pl.addLayout(r2)
        r3=QHBoxLayout(); self.font_size=QSpinBox(); self.font_size.setRange(0,160); self.font_size.setSpecialValueText("自动"); self.font_size.setSuffix(" px")
        self.columns=QSpinBox(); self.columns.setRange(0,12); self.columns.setSpecialValueText("自动")
        self.line_spacing=QDoubleSpinBox(); self.line_spacing.setRange(-1.0,0.60); self.line_spacing.setSingleStep(0.02); self.line_spacing.setDecimals(2); self.line_spacing.setSpecialValueText("自动")
        r3.addWidget(QLabel("字号")); r3.addWidget(self.font_size); r3.addWidget(QLabel("列数")); r3.addWidget(self.columns); r3.addWidget(QLabel("行距")); r3.addWidget(self.line_spacing); r3.addStretch(1); pl.addLayout(r3)
        root.addWidget(panel)
        actions=QHBoxLayout(); actions.addStretch(1); cancel=QPushButton("取消"); self.save_btn=QPushButton("保存并应用"); self.save_btn.setObjectName("primary")
        actions.addWidget(cancel); actions.addWidget(self.save_btn); root.addLayout(actions)
        cancel.clicked.connect(self.reject); self.save_btn.clicked.connect(self._save); self.ocr_btn.clicked.connect(self._rerun_ocr); self.new_btn.clicked.connect(self._new)
        self.delete_btn.clicked.connect(self._delete); self.font_pick.clicked.connect(self._pick_font); self.block_combo.currentIndexChanged.connect(self._load_selected)
        self.target_view.selection_changed.connect(self._target_selection_changed)
        self._reload_combo()

    def _reload_combo(self, select_id: str=""):
        self._blocks=load_ocr_blocks(self.page_dir,self.mode)
        self.block_combo.blockSignals(True); self.block_combo.clear(); self.block_combo.addItem("新建 OCR 文本块","")
        for i,row in enumerate(self._blocks,1):
            text=str(row.get("render_text") or row.get("ocr_text") or "").replace("\n"," ").strip()
            self.block_combo.addItem(f"{i}. {text[:28] or '未识别'}",str(row.get("id") or ""))
        idx=self.block_combo.findData(select_id) if select_id else 0; self.block_combo.setCurrentIndex(max(0,idx)); self.block_combo.blockSignals(False)
        self._load_selected()

    def _selected_row(self):
        bid=str(self.block_combo.currentData() or "")
        return next((dict(x) for x in self._blocks if str(x.get("id") or "")==bid),None)

    def _set_combo_value(self, combo, value):
        i=combo.findData(str(value or "")); combo.setCurrentIndex(max(0,i))

    def _load_selected(self,*_):
        row=self._selected_row(); self._current_id=str(row.get("id") or "") if row else ""
        if not row:
            self.target_view.set_box([]); self.source_view.set_box([]); self.text.clear(); self.font.clear(); self.font_size.setValue(0); self.columns.setValue(0); self.line_spacing.setValue(-1.0)
            self._set_combo_value(self.orientation,"auto"); self._set_combo_value(self.break_mode,"smart"); self._set_combo_value(self.layout_mode,"smart_scaling"); self.ocr_status.setText("新建：请在 TARGET 上拖框"); return
        tb=list(row.get("target_bbox") or []); sb=list(row.get("source_bbox") or [])
        self.target_view.set_box(tb); self.source_view.set_box(sb); self.text.setPlainText(str(row.get("render_text") or row.get("ocr_text") or "")); self.font.setText(str(row.get("font_path") or "")); self.font_size.setValue(int(row.get("font_size") or 0)); self.columns.setValue(int(row.get("columns") or 0))
        spacing=row.get("line_spacing_ratio"); self.line_spacing.setValue(-1.0 if spacing is None else float(spacing)); self._set_combo_value(self.orientation,row.get("orientation") or "auto"); self._set_combo_value(self.break_mode,row.get("line_break_mode") or "smart"); self._set_combo_value(self.layout_mode,row.get("layout_mode") or "smart_scaling")
        conf=float(row.get("confidence") or 0.0); self.ocr_status.setText(f"OCR 置信度 {conf:.2f}" if conf else "可重新 OCR")

    def _target_selection_changed(self,bbox):
        if bbox and len(bbox)==4:
            try: self.source_view.set_box(map_target_bbox_to_source(self.project,list(bbox)))
            except Exception: self.source_view.set_box([])
            self.ocr_status.setText("选框已更新 · 可重新 OCR")

    def _current_style(self):
        return {
            "id":self._current_id,"render_text":self.text.toPlainText(),"orientation":self.orientation.currentData() or "auto",
            "line_break_mode":self.break_mode.currentData() or "smart","layout_mode":self.layout_mode.currentData() or "smart_scaling",
            "font_path":self.font.text().strip(),"font_size":int(self.font_size.value()),"columns":int(self.columns.value()),
            "line_spacing_ratio":None if float(self.line_spacing.value())<0 else float(self.line_spacing.value()),
        }

    def _rerun_ocr(self):
        bbox=self.target_view.box()
        if len(bbox)!=4:
            QMessageBox.information(self,"请先框选","请在右侧 TARGET 图上拖出要重新 OCR 的区域。"); return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            seed=self._selected_row() or self._current_style(); seed.update(self._current_style())
            row=recognize_manual_ocr_block(self.project,self.source_path,self.target_path,bbox,self.config,existing=seed)
            self._current_id=str(row.get("id") or ""); self.text.setPlainText(str(row.get("render_text") or row.get("ocr_text") or "")); self.source_view.set_box(list(row.get("source_bbox") or []))
            # Keep the freshly detected TARGET polygons until save.
            self._pending_ocr=row
            serr=str(row.get("source_ocr_error") or ""); terr=str(row.get("target_ocr_error") or "")
            if serr:
                self.ocr_status.setText("SOURCE OCR 失败，可手动输入文字："+serr[:80])
            else:
                suffix="；TARGET 定位失败，将使用保守局部清字" if terr else ""
                self.ocr_status.setText(f"重新 OCR 完成 · 置信度 {float(row.get('confidence') or 0):.2f}{suffix}")
        except Exception as exc:
            QMessageBox.critical(self,"人工 OCR 失败",str(exc))
        finally: QApplication.restoreOverrideCursor()

    def _save(self):
        bbox=self.target_view.box(); text=self.text.toPlainText().strip()
        if len(bbox)!=4:
            QMessageBox.information(self,"缺少选框","请先在 TARGET 上框选文本区域。"); return
        if not text:
            QMessageBox.information(self,"缺少文字","OCR 结果为空时可以手动输入中文，再保存文本块。"); return
        base=dict(getattr(self,"_pending_ocr",None) or self._selected_row() or {})
        base.update(self._current_style()); base["target_bbox"]=list(bbox); base["source_bbox"]=map_target_bbox_to_source(self.project,list(bbox)); base["render_text"]=text; base.setdefault("ocr_text",text); base["review_kind"]="manual_ocr"; base["box_locked"]=True; base["manual_override"]=True
        saved=upsert_ocr_block(self.page_dir,self.mode,base); self._current_id=str(saved.get("id") or ""); self.accept()

    def _new(self):
        self._current_id=""; self.block_combo.setCurrentIndex(0); self.target_view.set_box([]); self.source_view.set_box([]); self.text.clear(); self.ocr_status.setText("新建：请在 TARGET 上拖框")
        if hasattr(self,"_pending_ocr"): delattr(self,"_pending_ocr")

    def _delete(self):
        bid=str(self.block_combo.currentData() or "")
        if not bid: return
        if QMessageBox.question(self,"删除 OCR 文本块","删除当前人工 OCR 文本块？") != QMessageBox.StandardButton.Yes: return
        delete_ocr_block(self.page_dir,self.mode,bid); self._reload_combo(); self.ocr_status.setText("已删除，正在返回并重新合成"); self.accept()

    def _pick_font(self):
        start=str(Path(self.font.text()).expanduser().parent) if self.font.text().strip() else str(Path.home())
        path,_=QFileDialog.getOpenFileName(self,"选择当前 OCR 文本块字体",start,"Fonts (*.ttf *.otf *.ttc);;All Files (*)")
        if path: self.font.setText(path)

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

    def __init__(self, source_path: str | Path, target_path: str | Path, project: dict[str, Any], parent=None, *, initial_bbox: list[int] | None = None, initial_mode: str | None = None, commit_handler=None, trace_handler=None, config=None):
        super().__init__(parent); self.setWindowTitle("人工补漏 / 开放式效果字")
        _configure_responsive_dialog(self, (1500, 960), (980, 650))
        self.project = dict(project or {}); self.source_path=Path(source_path); self.target_path=Path(target_path); self._reveal_mask=None; self._reveal_patch=None
        self._config = config
        self._commit_handler=commit_handler; self._trace_handler=trace_handler; self._committed_directly=False
        import cv2 as _cv2_dialog
        self._target_cv = _cv2_dialog.imread(str(self.target_path), _cv2_dialog.IMREAD_COLOR)
        self._initial_bbox=list(initial_bbox or []); self._initial_mode=str(initial_mode or "")
        self._mode_locked_by_user = bool(self._initial_mode)
        self._applying_mode_programmatically = False
        root = QVBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(9)
        hint = QLabel("在右侧高清日文图上拖框。这个框不依赖 OCR、气泡检测器或自动候选；程序会利用已保存的页面配准，从旧中文版提取对应区域。开放式效果字模式只迁移 SOURCE 支持的中文笔画，并自动估计/清除 TARGET 的日文笔画，尽量保留紫色、网点和人物背景。")
        hint.setObjectName("hint"); hint.setWordWrap(True); root.addWidget(hint)
        split = QSplitter(Qt.Orientation.Horizontal)
        left = QFrame(); left.setObjectName("card"); ll=QVBoxLayout(left); ll.setContentsMargins(8,8,8,8); ll.addWidget(QLabel("旧版中文 · 自动映射参考")); self.source_view=RegionSelectView(source_path, editable=False, parent=self); ll.addWidget(self.source_view,1)
        right = QFrame(); right.setObjectName("card"); rl=QVBoxLayout(right); rl.setContentsMargins(8,8,8,8); rl.addWidget(QLabel("高清日文 · 在这里框选遗漏区域")); self.target_view=RegionSelectView(target_path, editable=True, parent=self); rl.addWidget(self.target_view,1)
        split.addWidget(left); split.addWidget(right); split.setSizes([390,980]); split.setStretchFactor(0,0); split.setStretchFactor(1,1); split.setChildrenCollapsible(False); root.addWidget(split,1)

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
        row=QHBoxLayout(); fit=QPushButton("适合窗口"); reset=QPushButton("清除框选"); self.preview_mask_btn=QPushButton("预览实际文字 Mask"); self.preview_mask_btn.setObjectName("softPrimary"); save=QPushButton("应用此人工区域"); save.setObjectName("primary"); cancel=QPushButton("取消")
        row.addWidget(fit); row.addWidget(reset); row.addWidget(self.preview_mask_btn); row.addStretch(1); row.addWidget(save); row.addWidget(cancel); root.addLayout(row)
        self.target_view.selection_changed.connect(self._selection_changed)
        fit.clicked.connect(self._fit); reset.clicked.connect(lambda: self.target_view.set_box([], emit=True)); self.preview_mask_btn.clicked.connect(self._preview_effective_masks); save.clicked.connect(self._accept_checked); cancel.clicked.connect(self.reject)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.mode.activated.connect(self._manual_mode_activated)
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
            logger.debug("manual editor trace callback failed", exc_info=True)

    def was_committed_directly(self) -> bool:
        return bool(self._committed_directly)

    def _manual_mode_activated(self, *_args):
        if not self._applying_mode_programmatically:
            self._mode_locked_by_user = True

    def _fit(self):
        self.source_view.fit_to_window(); self.target_view.fit_to_window()

    def _apply_initial_state(self):
        self._fit()
        if len(self._initial_bbox) == 4:
            self.target_view.set_box(self._initial_bbox, emit=True)

    def _mode_changed(self):
        effect = (self.mode.currentData() in ("effect_text", "reveal_text"))
        for w in (self.diff,self.expand,self.feather,self.auto_clear): w.setEnabled(effect)
        if hasattr(self, "preview_mask_btn"):
            self.preview_mask_btn.setEnabled(self.mode.currentData() == "white_bubble_text")
            self.preview_mask_btn.setToolTip("白气泡模式下显示经过边框剥离后的真实 SOURCE 写入 / TARGET 清除 Mask" if self.preview_mask_btn.isEnabled() else "请先选择“白色气泡 · 文字迁移”模式")

    def _selection_changed(self, bbox):
        box = list(bbox or [])
        if len(box) != 4:
            self.source_view.set_box([]); self.info.setText("尚未框选区域"); return
        src_box = map_target_bbox_to_source(self.project, box); self.source_view.set_box(src_box)
        recommendation = "彩色开放式文字建议使用“擦除显字”"
        suggested_mode = "reveal_text"
        if self._target_cv is not None:
            import cv2
            h, w = self._target_cv.shape[:2]; x0, y0, x1, y1 = [int(v) for v in box]
            x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1)); y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
            roi = self._target_cv[y0:y1, x0:x1]
            if roi.size:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV); gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                white = float(np.mean((gray >= 220) & (hsv[..., 1] <= 42)))
                if white >= 0.68:
                    suggested_mode = "white_bubble_text"
                    recommendation = "检测为白底区域：建议“白色气泡 · 文字迁移 + X/Y 微调”"
                else:
                    suggested_mode = "reveal_text"
                    recommendation = "检测为彩色/复杂区域：建议“彩色开放式文字 · 擦除显字”"
        if not self._mode_locked_by_user:
            mi = self.mode.findData(suggested_mode)
            if mi >= 0 and self.mode.currentIndex() != mi:
                self._applying_mode_programmatically = True
                try:
                    self.mode.setCurrentIndex(mi)
                finally:
                    self._applying_mode_programmatically = False
        self.info.setText(f"TARGET {box[0]},{box[1]}–{box[2]},{box[3]} · SOURCE 自动映射约 {src_box[0]},{src_box[1]}–{src_box[2]},{src_box[3]} · {recommendation}")

    def _preview_effective_masks(self):
        if self.mode.currentData() != "white_bubble_text":
            QMessageBox.information(self, "仅用于白气泡", "请先选择“白色气泡 · 文字迁移 + X/Y 微调（不贴背景）”。")
            return
        box = self.target_view.box()
        if len(box) != 4:
            QMessageBox.information(self, "没有框选", "请先在右侧高清日文图上框选白色气泡内部区域。")
            return
        import cv2
        source = cv2.imread(str(self.source_path), cv2.IMREAD_COLOR)
        target = cv2.imread(str(self.target_path), cv2.IMREAD_COLOR)
        if source is None or target is None:
            QMessageBox.warning(self, "无法预览", "源图或目标图读取失败。")
            return
        try:
            masks = build_manual_effect_masks(source, target, self.project, self._row_payload(), getattr(self, "_config", None))
            dlg = ManualTextMaskPreviewDialog(self.target_path, masks.source_mask, masks.target_clear_mask, masks.diagnostics, self)
            dlg.exec()
        except Exception as exc:
            QMessageBox.critical(self, "无法预览实际 Mask", str(exc))

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
                masks=build_manual_effect_masks(source,target,self.project,self._row_payload(),getattr(self, "_config", None))
                seed=build_reveal_seed_mask(masks.source_mask,masks.target_clear_mask,padding_px=max(4,int(self.expand.value())+2))
                try:
                    dlg=RevealMaskDialog(self.target_path,masks.aligned_source,masks.source_mask,masks.target_clear_mask,seed,self,focus_bbox=box)
                except TypeError as exc:
                    # Backward-compatible with third-party/test dialogs that
                    # implement the pre-v1.3.13 constructor.
                    if "focus_bbox" not in str(exc): raise
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







class WorkbenchPage(QWidget):
    def __init__(self, window: "StudioWindow"):
        super().__init__(); self.window=window
        root=QHBoxLayout(self); root.setContentsMargins(10,10,10,10); root.setSpacing(10)
        split=QSplitter(Qt.Orientation.Horizontal); root.addWidget(split)

        # Koharu-inspired workspace shell: page rail -> canvas -> inspector.
        # Unlike Koharu's web editor, this remains native Qt and keeps Folirina's
        # existing project/page state as the only source of truth.
        self.page_rail = WorkbenchPageRail(self)
        split.addWidget(self.page_rail)

        # Colortina-style editor composition: a compact control strip above a
        # large central canvas.  Avoid spending editor height on explanatory
        # copy; the actual manga page is the primary object in this workspace.
        preview=Card()
        preview.layout.setContentsMargins(12,10,12,10); preview.layout.setSpacing(7)
        page_nav=QHBoxLayout(); page_nav.setSpacing(6)
        compact_title=QLabel("替换工作台"); compact_title.setObjectName("sectionTitle")
        page_nav.addWidget(compact_title)
        self.activity_badge=QLabel("就绪"); self.activity_badge.setObjectName("activityBadge")
        page_nav.addWidget(self.activity_badge); page_nav.addStretch(1)
        self.prev_page=QPushButton("← 上一页"); self.prev_page.setObjectName("pageNav")
        self.page_counter=QLabel("0 / 0"); self.page_counter.setObjectName("pageCounter"); self.page_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.next_page=QPushButton("下一页 →"); self.next_page.setObjectName("pageNav")
        self.page_rail_toggle=QPushButton("收起页面栏"); self.page_rail_toggle.setCheckable(True); self.page_rail_toggle.setObjectName("pageNav")
        self.inspector_toggle=QPushButton("隐藏参数栏"); self.inspector_toggle.setCheckable(True); self.inspector_toggle.setObjectName("pageNav")
        page_nav.addWidget(self.prev_page); page_nav.addWidget(self.page_counter); page_nav.addWidget(self.next_page); page_nav.addSpacing(4); page_nav.addWidget(self.page_rail_toggle); page_nav.addWidget(self.inspector_toggle)
        preview.layout.addLayout(page_nav)

        toolbar=QGridLayout(); toolbar.setContentsMargins(0,0,0,0); toolbar.setHorizontalSpacing(4); toolbar.setVerticalSpacing(1); self.view_buttons=[]
        for i,(label,key) in enumerate([("日文原图","target"),("旧中文版","source"),("最终结果","result"),("复核标注","review"),("迁移蒙版","mask"),("清除蒙版","clear_mask"),("中文迁移层","chinese_layer"),("只清日文","removed"),("TARGET擦除预览","target_erase"),("TARGET恢复预览","target_restore")]):
            b=QPushButton(label); b.setCheckable(True); b.setObjectName("segmented"); b.clicked.connect(lambda _=False,k=key:self.set_view(k)); toolbar.addWidget(b, i//5, i%5); self.view_buttons.append((b,key))
            toolbar.setColumnStretch(i%5, 1)
        preview.layout.addLayout(toolbar)
        self.image=ImageView(); preview.layout.addWidget(self.image,1)
        footer=QHBoxLayout()
        self.page_caption=QLabel("未选择页面"); self.page_caption.setObjectName("hint"); self.page_caption.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.view_status=QLabel(""); self.view_status.setObjectName("quiet")
        footer.addWidget(self.page_caption,1); footer.addWidget(self.view_status,0,Qt.AlignmentFlag.AlignRight)
        preview.layout.addLayout(footer)
        split.addWidget(preview)

        side=QFrame(); self.side=side; side.setObjectName("inspectorShell")
        side_outer=QVBoxLayout(side); side_outer.setContentsMargins(0,0,0,0); side_outer.setSpacing(0)
        inspector_head=QFrame(); inspector_head.setObjectName("inspectorTabBar")
        inspector_head_layout=QHBoxLayout(inspector_head); inspector_head_layout.setContentsMargins(8,7,8,7); inspector_head_layout.setSpacing(5)
        self.inspector_settings_tab=QPushButton("参数"); self.inspector_review_tab=QPushButton("复核 / QA")
        self.inspector_settings_tab.setObjectName("inspectorTab"); self.inspector_review_tab.setObjectName("inspectorTab")
        for tab in (self.inspector_settings_tab,self.inspector_review_tab): tab.setCheckable(True); inspector_head_layout.addWidget(tab,1)
        self.inspector_settings_tab.setChecked(True)
        side_outer.addWidget(inspector_head)
        self.inspector_stack=QStackedWidget(); side_outer.addWidget(self.inspector_stack,1)

        settings_scroll=QScrollArea(); settings_scroll.setWidgetResizable(True)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host=QWidget(); self.side_host=host; host.setMinimumWidth(0); host.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        sl=QVBoxLayout(host); sl.setContentsMargins(0,0,4,0); sl.setSpacing(10); settings_scroll.setWidget(host)
        self.inspector_stack.addWidget(settings_scroll)

        review_scroll=QScrollArea(); review_scroll.setWidgetResizable(True)
        review_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        review_host=QWidget(); review_host.setMinimumWidth(0); review_host.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        rl=QVBoxLayout(review_host); rl.setContentsMargins(0,0,4,0); rl.setSpacing(10); review_scroll.setWidget(review_host)
        self.inspector_stack.addWidget(review_scroll)
        mode=Card("替换策略 / 蒙版参数", "出版安全门禁已移除：Direct/Mask 优先完成替换；风险指标仅保留诊断，不再作为写入阻断。")
        self.publication_safety=QCheckBox("出版安全门禁已移除（兼容旧配置）")
        self.publication_safety.setChecked(False)
        self.publication_safety.setEnabled(False)
        self.publication_safety.setToolTip("v1.2.0 起该门禁不再参与 Direct/Mask 写入。仅保留基础几何有效性检查。")
        self.paired=QCheckBox("成对差异自动提取"); self.paired.setChecked(True)
        self.skip_ocr=QCheckBox("Auto / Hybrid：成对差异安全时跳过 OCR"); self.skip_ocr.setChecked(True)
        self.skip_ocr.setToolTip("只影响 Auto / Hybrid 的 OCR 优化。明确选择“精准蒙版迁移”时始终 0 OCR，不创建、不加载也不调用任何 OCR backend。")
        self.pixel_exact=QCheckBox("同源页面像素级精确覆盖"); self.pixel_exact.setChecked(True)
        self.full_bubble_patch=QCheckBox("蒙版内完整中文字形迁移（保留 TARGET 背景）")
        self.full_bubble_patch.setChecked(bool(getattr(self.window.state.config.mask_replace, "rigid_container_full_patch_enabled", True)))
        self.full_bubble_patch.setToolTip("只迁移 SOURCE 中文字形/透明度，TARGET 背景、肤色、衣服、网点和气泡底色都不允许被 SOURCE RGB 覆盖。")
        self.preserve_border=QCheckBox("保留高清日文气泡边线"); self.preserve_border.setChecked(True)
        self.blur_guard=QCheckBox("低清文字保护：模糊时禁止直接贴像素"); self.blur_guard.setChecked(True)
        self.blur_guard.setToolTip("摄影模糊、反光或低分辨率旧版先做光照归一化/墨迹重建；精准蒙版模式完全不调用 OCR，不安全区域进入复核。")
        self.preserve_source_layout=QCheckBox("清晰旧中文版保留原字号/分列（推荐）"); self.preserve_source_layout.setChecked(True)
        self.preserve_source_layout.setToolTip("精准蒙版始终保留旧中文版真实字号/分列/符号，并且完全不调用 OCR。需要 OCR 识别或重新排字时请改用精准蒙版+OCR / OCR重排，或手动编辑。")
        mode.layout.addWidget(self.publication_safety); mode.layout.addWidget(self.paired); mode.layout.addWidget(self.skip_ocr); mode.layout.addWidget(self.pixel_exact); mode.layout.addWidget(self.full_bubble_patch); mode.layout.addWidget(self.preserve_border); mode.layout.addWidget(self.blur_guard); mode.layout.addWidget(self.preserve_source_layout)
        photo_note=QLabel("当前策略：所有自动路径都以 TARGET 为唯一背景；白底和彩底都只清除日文文字并迁移 SOURCE 中文字形。SOURCE 的白纸、灰阶、肤色或旧背景 RGB 永远不会写进彩图。")
        photo_note.setObjectName("quiet"); photo_note.setWordWrap(True); mode.layout.addWidget(photo_note)
        sl.addWidget(mode)

        align=Card("局部对齐与清晰度")
        form=QFormLayout(); form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.local=QComboBox(); self.local.addItems(["ecc","bbox","global"]); self.sr=QComboBox(); self.sr.addItems(["auto","torch","lanczos","external","off"])
        self.fidelity=QComboBox(); self.fidelity.addItem("自动：光照归一化 → 墨迹（精准蒙版不 OCR）","auto"); self.fidelity.addItem("只保留原像素","pixels"); self.fidelity.addItem("强制墨迹重建","ink"); self.fidelity.addItem("低清直接拒绝","reject")
        self.iou=QDoubleSpinBox(); self.iou.setRange(.2,1); self.iou.setSingleStep(.01); self.iou.setDecimals(3); self.iou.setValue(.80)
        self.coverage=QDoubleSpinBox(); self.coverage.setRange(.5,1); self.coverage.setSingleStep(.001); self.coverage.setDecimals(3); self.coverage.setValue(.985)
        self.sr_model=QLineEdit(); self.sr_model.setPlaceholderText("可选：本地 .pth/.safetensors 超分模型")
        self.sr_pick=QPushButton("选择…"); srrow=QWidget(); srl=QHBoxLayout(srrow); srl.setContentsMargins(0,0,0,0); srl.setSpacing(6); srl.addWidget(self.sr_model,1); srl.addWidget(self.sr_pick)
        form.addRow("局部几何",self.local); form.addRow("文字清晰策略",self.fidelity); form.addRow("源 patch 超分",self.sr); form.addRow("局部超分模型",srrow); form.addRow("Mask IoU 门槛",self.iou); form.addRow("目标覆盖率",self.coverage); align.layout.addLayout(form); sl.addWidget(align)

        stages=Card("检测 / 清除 / 写入分离", "参考成熟漫画翻译工具的分阶段设计：气泡检测、清除蒙版、去字预览和中文写入可独立检查，不必每改一点就重跑整页。")
        sf=QFormLayout(); sf.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); sf.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.detector_strategy=QComboBox(); self.detector_strategy.addItem("仅主检测器","primary_only"); self.detector_strategy.addItem("主检测器 + 按需辅助（推荐）","primary_conditional_aux"); self.detector_strategy.addItem("主检测器 + 始终辅助","primary_plus_aux")
        self.primary_detector=QComboBox(); self.primary_detector.addItem("Koharu Layout RF-DETR Seg 2XL（推荐）","koharu_layout"); self.primary_detector.addItem("MangaLens / YOLO","mangalens"); self.primary_detector.addItem("Comic Translate RT-DETR-v2","rtdetr_v2")
        aux_widget=QWidget(); aux_grid=QGridLayout(aux_widget); aux_grid.setContentsMargins(0,0,0,0); aux_grid.setHorizontalSpacing(8); aux_grid.setVerticalSpacing(4)
        self.aux_geometry_white=QCheckBox("白色结构"); self.aux_mangalens=QCheckBox("MangaLens"); self.aux_rtdetr=QCheckBox("RT-DETR-v2"); self.aux_ysg=QCheckBox("YSG OBB 开放字"); self.aux_koharu=QCheckBox("Koharu"); self.aux_ctd=QCheckBox("CTD Sidecar"); self.aux_sidecar=QCheckBox("Sidecar")
        for i,w in enumerate([self.aux_geometry_white,self.aux_mangalens,self.aux_rtdetr,self.aux_ysg,self.aux_koharu,self.aux_ctd,self.aux_sidecar]): aux_grid.addWidget(w,i//3,i%3)
        self.sam2_refine=QCheckBox("必要时使用 SAM 2 / 2.1 精修")
        cfg_b=self.window.state.config.bubbles
        si=self.detector_strategy.findData(getattr(cfg_b,"detector_strategy","primary_conditional_aux")); self.detector_strategy.setCurrentIndex(max(0,si))
        pi=self.primary_detector.findData(getattr(cfg_b,"primary_detector","koharu_layout")); self.primary_detector.setCurrentIndex(max(0,pi))
        selected_aux=set(getattr(cfg_b,"auxiliary_detectors",[]) or [])
        self.aux_geometry_white.setChecked("geometry_white" in selected_aux); self.aux_mangalens.setChecked("mangalens" in selected_aux); self.aux_rtdetr.setChecked("rtdetr_v2" in selected_aux); self.aux_ysg.setChecked("ysg_obb" in selected_aux); self.aux_koharu.setChecked("koharu_layout" in selected_aux); self.aux_ctd.setChecked("ctd_sidecar" in selected_aux); self.aux_sidecar.setChecked("sidecar" in selected_aux); self.sam2_refine.setChecked(bool(getattr(cfg_b,"sam2_refine_enabled",False)))
        self.detector_size=QSpinBox(); self.detector_size.setRange(640,4096); self.detector_size.setSingleStep(128); self.detector_size.setValue(int(self.window.state.config.bubbles.mangalens_imgsz))
        self.clear_dilate=QSpinBox(); self.clear_dilate.setRange(0,20); self.clear_dilate.setValue(int(self.window.state.config.masking.max_dilation_px)); self.clear_dilate.setSuffix(" px")
        self.inpaint_backend=QComboBox(); self.inpaint_backend.addItem("自动","auto"); self.inpaint_backend.addItem("纯色纸面","solid"); self.inpaint_backend.addItem("OpenCV 修复","opencv"); self.inpaint_backend.addItem("LaMa（兼容旧 runner）","lama"); self.inpaint_backend.addItem("LaMa Manga","lama_manga"); self.inpaint_backend.addItem("AOT Inpainting","aot_inpainting"); self.inpaint_backend.addItem("FLUX.2 Klein","flux2_klein"); self.inpaint_backend.addItem("RORem Mixed","rorem_mixed")
        sf.addRow("检测策略",self.detector_strategy); sf.addRow("主检测器",self.primary_detector); sf.addRow("辅助检测器",aux_widget); sf.addRow("边界精修",self.sam2_refine); sf.addRow("模型检测分辨率",self.detector_size); sf.addRow("清除 mask 最大扩张",self.clear_dilate); sf.addRow("去字修复",self.inpaint_backend); stages.layout.addLayout(sf)
        # Long review actions are arranged as a stable two-column grid.  Keeping
        # three unrelated controls on one narrow row caused macOS to elide labels
        # such as “仅擦 TARGET 日文层…” even though the whole-page scale itself was
        # correct.  Rows now expand to the inspector width before outer scaling.
        action_grid=QGridLayout(); action_grid.setContentsMargins(0,0,0,0); action_grid.setHorizontalSpacing(7); action_grid.setVerticalSpacing(7)
        action_grid.setColumnStretch(0,1); action_grid.setColumnStretch(1,1)
        self.edit_clear_mask=QPushButton("编辑自动清除蒙版…"); self.remove_text_only=QPushButton("预览自动去字")
        action_grid.addWidget(self.edit_clear_mask,0,0); action_grid.addWidget(self.remove_text_only,0,1)
        self.apply_mask_review=QPushButton("应用蒙版到最终结果"); self.apply_mask_review.setObjectName("softPrimary"); self.reset_clear_mask=QPushButton("恢复自动蒙版")
        action_grid.addWidget(self.apply_mask_review,1,0); action_grid.addWidget(self.reset_clear_mask,1,1)
        self.force_transfer_mask=QPushButton("人工强制迁移蒙版…"); self.force_transfer_mask.setObjectName("softPrimary"); self.reset_force_transfer_mask=QPushButton("清空强制迁移蒙版")
        action_grid.addWidget(self.force_transfer_mask,2,0); action_grid.addWidget(self.reset_force_transfer_mask,2,1)
        stages.layout.addLayout(action_grid)
        self.force_transfer_mask.setToolTip("自动漏识别或自动区域本身有误时使用：红色人工层和蓝色 OCR/自动蒙版都能直接涂抹/消除。蓝色修订会作为本页自动蒙版覆盖保存；重新处理后继续生效。自动也完全漏掉时仍可纯人工重跑。")
        self.reset_force_transfer_mask.setToolTip("删除人工强制迁移蒙版，并恢复第一次使用该工具前冻结的稳定结果。")
        self.target_erase_mode=QComboBox(); self.target_erase_mode.addItem("智能恢复 TARGET", "auto"); self.target_erase_mode.addItem("纯白涂抹", "pure_white"); self.target_erase_mode.setToolTip("智能恢复：白底用 TARGET 纸色、复杂背景用 inpaint；纯白涂抹：仍只对 TARGET 日文层生效，中文保护区保持不变。")
        self.target_erase_mode.setMinimumContentsLength(14); self.target_erase_mode.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        stages.layout.addWidget(self.target_erase_mode)
        target_grid=QGridLayout(); target_grid.setContentsMargins(0,0,0,0); target_grid.setHorizontalSpacing(7); target_grid.setVerticalSpacing(7); target_grid.setColumnStretch(0,1); target_grid.setColumnStretch(1,1)
        self.target_layer_erase=QPushButton("仅擦 TARGET 日文层…"); self.target_layer_erase.setObjectName("softPrimary"); self.reset_target_layer_erase=QPushButton("清空 TARGET 擦除")
        self.target_layer_restore=QPushButton("恢复 TARGET 日文层…"); self.target_layer_restore.setObjectName("softPrimary"); self.reset_target_layer_restore=QPushButton("清空 TARGET 恢复")
        target_grid.addWidget(self.target_layer_erase,0,0); target_grid.addWidget(self.reset_target_layer_erase,0,1)
        target_grid.addWidget(self.target_layer_restore,1,0); target_grid.addWidget(self.reset_target_layer_restore,1,1)
        stages.layout.addLayout(target_grid)
        self.target_layer_erase.setToolTip("最终收尾画笔：可刷日文残字、黑点、短线和符号。只重建 TARGET 母版层，中文迁移/重排/人工补漏图层自动保护。")
        self.target_layer_restore.setToolTip("反向画笔：对不应显示中文的区域，直接恢复 TARGET 原始日文图层与背景。")
        stage_note=QLabel("自动没有识别到、识别不完整，或者自动蒙版本身过大/错误时，使用“人工强制迁移蒙版”：红色人工层与蓝色 OCR/自动层都可直接编辑。自动区域会先收紧为真实文字像素，避免整块气泡/画面被当成清除区；只剩零星残字时再用“仅擦 TARGET 日文层”。")
        stage_note.setObjectName("quiet"); stage_note.setWordWrap(True); stages.layout.addWidget(stage_note); sl.addWidget(stages)

        recovery=Card("人工补漏 / 开放式效果字", "专门处理自动检测漏掉的开放式效果字、彩底文字、人物画面上的文字。无需 OCR，也不要求存在气泡边界；人工框选后直接进入最终复核链。")
        self.expand_direct_range=QCheckBox("扩大 Direct 候选范围（难页）"); self.expand_direct_range.setChecked(bool(getattr(self.window.state.config.direct_patch,"source_direct_expand_candidate_range",False))); self.expand_direct_range.setToolTip("可选恢复模式：允许更小/更细长/弱文字种子的 Direct 候选进入检查。默认关闭；同页配准与 TARGET 背景保护仍是硬条件。")
        recovery.layout.addWidget(self.expand_direct_range)
        self.manual_effect_status=QLabel("当前页暂无人工补漏区域"); self.manual_effect_status.setObjectName("hint"); self.manual_effect_status.setWordWrap(True); recovery.layout.addWidget(self.manual_effect_status)
        self.manual_effect_candidate_status=QLabel("安全策略暂无待处理彩色/复杂文字候选"); self.manual_effect_candidate_status.setObjectName("quiet"); self.manual_effect_candidate_status.setWordWrap(True); recovery.layout.addWidget(self.manual_effect_candidate_status)
        self.manual_effect_candidate_target=QComboBox(); self.manual_effect_candidate_target.setMinimumWidth(0); self.manual_effect_candidate_target.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon); self.manual_effect_candidate_target.setMinimumContentsLength(8); recovery.layout.addWidget(self.manual_effect_candidate_target)
        effect_actions=QGridLayout(); effect_actions.setContentsMargins(0,0,0,0); effect_actions.setHorizontalSpacing(7); effect_actions.setVerticalSpacing(7); effect_actions.setColumnStretch(0,1); effect_actions.setColumnStretch(1,1)
        self.add_manual_effect=QPushButton("手动框选遗漏区域…"); self.add_manual_effect.setObjectName("softPrimary"); self.add_manual_effect_candidate=QPushButton("使用候选区域…"); self.undo_manual_effect=QPushButton("撤销最近区域")
        effect_actions.addWidget(self.add_manual_effect,0,0); effect_actions.addWidget(self.add_manual_effect_candidate,0,1); effect_actions.addWidget(self.undo_manual_effect,1,0,1,2)
        recovery.layout.addLayout(effect_actions)
        enote=QLabel("推荐流程：自动跑整页 → QA 看漏项 → 人工补漏 → 检查最终结果；若仍有日文/黑点，优先用“仅擦 TARGET 日文层”收尾。只有要改变自动清字管线本身时才编辑自动清除蒙版。")
        enote.setObjectName("quiet"); enote.setWordWrap(True); recovery.layout.addWidget(enote); rl.addWidget(recovery)

        qa=Card("当前页 QA")
        self.qa_label=QLabel("尚未处理"); self.qa_label.setObjectName("hint"); qa.layout.addWidget(self.qa_label)
        qrow=QHBoxLayout(); self.run=QPushButton("处理当前页"); self.run.setObjectName("primary"); self.reprocess_current=QPushButton("重新处理当前页"); self.reprocess_current.setObjectName("softPrimary"); qrow.addWidget(self.run,1); qrow.addWidget(self.reprocess_current,1); qa.layout.addLayout(qrow)
        self.reprocess_current.setToolTip("处理后的编辑区域也支持人工蒙版。点击后会重新跑当前页自动流程，并自动重新应用本页已有的人工强制迁移蒙版、人工补漏、清除蒙版、TARGET 擦除/恢复等编辑结果。")
        rl.addWidget(qa)

        manual=Card("OCR 文本编辑 / 排版", "仅用于“精准蒙版+OCR”和“OCR重排”：可人工框选 ROI 重新 OCR，也可编辑已有自动 Region 的文字、字体、字号、方向、断句与排版。Direct / 纯精准蒙版 / Reveal 不读取这里的文本块。")
        self.ocr_block_status=QLabel("人工 OCR 文本块：当前模式不可用"); self.ocr_block_status.setObjectName("quiet"); self.ocr_block_status.setWordWrap(True); manual.layout.addWidget(self.ocr_block_status)
        ocr_actions=QHBoxLayout(); self.open_ocr_block_editor=QPushButton("人工 OCR / 编辑文本块…"); self.open_ocr_block_editor.setObjectName("softPrimary"); self.reset_ocr_blocks=QPushButton("清空人工 OCR")
        ocr_actions.addWidget(self.open_ocr_block_editor,1); ocr_actions.addWidget(self.reset_ocr_blocks); manual.layout.addLayout(ocr_actions)
        self.manual_status=QLabel("当前页没有待复核文字区域"); self.manual_status.setObjectName("hint"); self.manual_status.setWordWrap(True); manual.layout.addWidget(self.manual_status)
        self.manual_target=QComboBox(); manual.layout.addWidget(self.manual_target)
        self.manual_text=QPlainTextEdit(); self.manual_text.setPlaceholderText("可直接修改已自动重排成功的中文；保留换行时可手动插入换行符…"); self.manual_text.setMaximumHeight(106); manual.layout.addWidget(self.manual_text)
        mrow=QHBoxLayout(); self.manual_orientation=QComboBox(); self.manual_orientation.addItem("自动排版","auto"); self.manual_orientation.addItem("竖排","vertical"); self.manual_orientation.addItem("横排","horizontal")
        self.manual_break_mode=QComboBox(); self.manual_break_mode.addItem("智能断句","smart"); self.manual_break_mode.addItem("均衡分行","balanced"); self.manual_break_mode.addItem("保留手动换行","source")
        self.manual_layout_mode=QComboBox(); self.manual_layout_mode.addItem("智能缩放","smart_scaling"); self.manual_layout_mode.addItem("严格文本框","strict"); self.manual_layout_mode.addItem("智能气泡","balloon_fill")
        mrow.addWidget(self.manual_orientation); mrow.addWidget(self.manual_break_mode,1); mrow.addWidget(self.manual_layout_mode,1); manual.layout.addLayout(mrow)
        fpreset=QHBoxLayout(); self.manual_font_preset=QComboBox()
        for label,value in [("当前/自动","custom"),("黑体 / Sans","sans"),("宋体 / Serif","serif"),("圆体 / Rounded","rounded"),("漫画体 / Comic","comic")]: self.manual_font_preset.addItem(label,value)
        fpreset.addWidget(QLabel("字体预设")); fpreset.addWidget(self.manual_font_preset,1); manual.layout.addLayout(fpreset)
        frow=QHBoxLayout(); self.manual_font=QLineEdit(); self.manual_font.setPlaceholderText("字体：留空=当前/全局；支持预设、字体路径或 A;B;C 候选链")
        self.manual_font_pick=QPushButton("字体…"); frow.addWidget(self.manual_font,1); frow.addWidget(self.manual_font_pick); manual.layout.addLayout(frow)
        prow=QHBoxLayout(); self.manual_font_size=QSpinBox(); self.manual_font_size.setRange(0,160); self.manual_font_size.setSpecialValueText("自动"); self.manual_font_size.setSuffix(" px")
        self.manual_columns=QSpinBox(); self.manual_columns.setRange(0,12); self.manual_columns.setSpecialValueText("自动")
        self.manual_line_spacing=QDoubleSpinBox(); self.manual_line_spacing.setRange(-1.0,0.6); self.manual_line_spacing.setSingleStep(0.02); self.manual_line_spacing.setDecimals(2); self.manual_line_spacing.setSpecialValueText("自动"); self.manual_line_spacing.setValue(-1.0)
        prow.addWidget(QLabel("字号")); prow.addWidget(self.manual_font_size); prow.addSpacing(6); prow.addWidget(QLabel("列数")); prow.addWidget(self.manual_columns); prow.addSpacing(6); prow.addWidget(QLabel("行距")); prow.addWidget(self.manual_line_spacing); prow.addStretch(1); manual.layout.addLayout(prow)
        arow=QHBoxLayout(); self.manual_apply=QPushButton("应用到当前 Region"); self.manual_apply.setObjectName("softPrimary"); self.manual_reset=QPushButton("恢复自动重排"); arow.addWidget(self.manual_apply,1); arow.addWidget(self.manual_reset,1); manual.layout.addLayout(arow)
        hrow=QHBoxLayout(); self.manual_undo=QPushButton("撤销编辑"); self.manual_redo=QPushButton("重做编辑"); self.manual_history_status=QLabel(""); self.manual_history_status.setObjectName("quiet"); hrow.addWidget(self.manual_undo); hrow.addWidget(self.manual_redo); hrow.addWidget(self.manual_history_status,1); manual.layout.addLayout(hrow)
        drow=QHBoxLayout(); self.candidate_accept=QPushButton("接受当前中文候选"); self.candidate_restore=QPushButton("还原日文"); drow.addWidget(self.candidate_accept,1); drow.addWidget(self.candidate_restore,1); manual.layout.addLayout(drow)

        # Long Chinese labels and combo-box contents must not force the review
        # sidebar wider than its viewport.  Qt's default minimumSizeHint for a
        # QComboBox is based on its longest item, which previously created the
        # horizontal clipping visible on narrow macOS windows.
        for widget in [self.local, self.sr, self.fidelity, self.sr_model, self.detector_strategy, self.primary_detector,
                       self.detector_size, self.clear_dilate, self.inpaint_backend, self.target_erase_mode, self.manual_target,
                       self.manual_effect_candidate_target, self.manual_orientation, self.manual_break_mode, self.manual_layout_mode, self.manual_font_preset, self.manual_font, self.manual_font_size, self.manual_columns, self.manual_line_spacing, self.manual_text, self.reprocess_current]:
            widget.setMinimumWidth(0)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, widget.sizePolicy().verticalPolicy())
        for combo in [self.local, self.sr, self.fidelity, self.detector_strategy, self.primary_detector, self.inpaint_backend, self.target_erase_mode, self.manual_target, self.manual_effect_candidate_target, self.manual_orientation, self.manual_break_mode, self.manual_layout_mode, self.manual_font_preset]:
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        for button in [self.sr_pick, self.edit_clear_mask, self.remove_text_only, self.apply_mask_review,
                       self.reset_clear_mask, self.force_transfer_mask, self.reset_force_transfer_mask,
                       self.target_layer_erase, self.reset_target_layer_erase, self.target_layer_restore, self.reset_target_layer_restore,
                       self.add_manual_effect, self.add_manual_effect_candidate, self.undo_manual_effect, self.manual_apply,
                       self.candidate_accept, self.candidate_restore, self.manual_reset, self.manual_undo, self.manual_redo, self.manual_font_pick, self.open_ocr_block_editor, self.reset_ocr_blocks, self.reprocess_current]:
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        rl.addWidget(manual); rl.addStretch(1); sl.addStretch(1)
        # The inspector is an editor, not navigation.  Give it enough width for
        # full labels, combo-boxes and numeric fields; users can still resize it
        # with the splitter or hide it entirely when they want maximum canvas.
        for card in side.findChildren(Card):
            card.layout.setContentsMargins(12,10,12,10)
            card.layout.setSpacing(8)
        side.setMinimumWidth(430); side.setMaximumWidth(540); split.addWidget(side); self.split=split; self.preview_card=preview
        split.setSizes([160,690,440]); split.setStretchFactor(0,0); split.setStretchFactor(1,1); split.setStretchFactor(2,0); split.setChildrenCollapsible(False)
        self._compact_layout=None
        QTimer.singleShot(0, self._update_responsive_workbench)
        self._page_rail_user_hidden=False
        self.prev_page.clicked.connect(lambda: self._move_page(-1)); self.next_page.clicked.connect(lambda: self._move_page(1)); self.inspector_toggle.toggled.connect(self._toggle_inspector); self.page_rail_toggle.toggled.connect(self._toggle_page_rail)
        self.page_rail.collapse_requested.connect(lambda: self.page_rail_toggle.setChecked(True))
        self.page_rail.page_selected.connect(self.window.set_selected_page)
        self.inspector_settings_tab.clicked.connect(lambda: self._set_inspector_panel(0))
        self.inspector_review_tab.clicked.connect(lambda: self._set_inspector_panel(1))
        self.run.clicked.connect(self._run); self.reprocess_current.clicked.connect(self._reprocess_current_page)
        self.sr_pick.clicked.connect(self._choose_sr_model)
        self.manual_apply.clicked.connect(self._apply_manual_reletter)
        self.manual_reset.clicked.connect(self._reset_manual_reletter)
        self.open_ocr_block_editor.clicked.connect(self._open_ocr_block_editor)
        self.reset_ocr_blocks.clicked.connect(self._reset_ocr_blocks)
        self.manual_undo.clicked.connect(lambda: self._review_history_step("undo"))
        self.manual_redo.clicked.connect(lambda: self._review_history_step("redo"))
        self.manual_font_pick.clicked.connect(self._choose_manual_reletter_font); self.manual_font_preset.currentIndexChanged.connect(self._apply_manual_font_preset)
        self.manual_target.currentIndexChanged.connect(self._manual_selection_changed)
        self.candidate_accept.clicked.connect(lambda: self._set_candidate_decision("accept"))
        self.candidate_restore.clicked.connect(lambda: self._set_candidate_decision("restore"))
        self.edit_clear_mask.clicked.connect(self._edit_clear_mask)
        self.remove_text_only.clicked.connect(self._remove_text_only)
        self.apply_mask_review.clicked.connect(self._apply_mask_review)
        self.reset_clear_mask.clicked.connect(self._reset_clear_mask)
        self.force_transfer_mask.clicked.connect(self._edit_force_transfer_mask)
        self.reset_force_transfer_mask.clicked.connect(self._reset_force_transfer_mask)
        self.target_layer_erase.clicked.connect(self._edit_target_layer_erase)
        self.reset_target_layer_erase.clicked.connect(self._reset_target_layer_erase)
        self.target_layer_restore.clicked.connect(self._edit_target_layer_restore)
        self.reset_target_layer_restore.clicked.connect(self._reset_target_layer_restore)
        self.add_manual_effect.clicked.connect(self._add_manual_effect_region)
        self.add_manual_effect_candidate.clicked.connect(self._add_next_manual_effect_candidate)
        self.undo_manual_effect.clicked.connect(self._undo_manual_effect_region)
        for w in [self.paired,self.skip_ocr,self.pixel_exact,self.full_bubble_patch,self.preserve_border,self.blur_guard,self.preserve_source_layout]: w.toggled.connect(self._sync)
        self.local.currentTextChanged.connect(self._sync); self.fidelity.currentIndexChanged.connect(self._sync); self.sr.currentTextChanged.connect(self._sync); self.iou.valueChanged.connect(self._sync); self.coverage.valueChanged.connect(self._sync)
        self.detector_strategy.currentIndexChanged.connect(self._sync); self.primary_detector.currentIndexChanged.connect(self._sync); self.detector_size.valueChanged.connect(self._sync); self.clear_dilate.valueChanged.connect(self._sync); self.inpaint_backend.currentIndexChanged.connect(self._sync); self.expand_direct_range.toggled.connect(self._sync)
        for cb in [self.aux_geometry_white,self.aux_mangalens,self.aux_rtdetr,self.aux_koharu,self.aux_sidecar,self.sam2_refine]: cb.toggled.connect(self._sync)
        ii=self.inpaint_backend.findData(self.window.state.config.inpainting.backend); self.inpaint_backend.setCurrentIndex(max(0,ii))
        self.current_view="target"; self.set_view("target")

    def set_processing_busy(self, busy: bool):
        """Disable mutating review controls while any page writer is active."""
        if hasattr(self, "activity_badge"):
            self.activity_badge.setText("处理中…" if busy else "就绪")
            self.activity_badge.setProperty("busy", bool(busy))
            self.activity_badge.style().unpolish(self.activity_badge); self.activity_badge.style().polish(self.activity_badge)
        names = (
            "run", "reprocess_current", "edit_clear_mask", "remove_text_only", "apply_mask_review",
            "reset_clear_mask", "force_transfer_mask", "reset_force_transfer_mask",
            "target_layer_erase", "reset_target_layer_erase", "target_layer_restore",
            "reset_target_layer_restore", "add_manual_effect", "add_manual_effect_candidate",
            "undo_manual_effect", "manual_apply", "manual_reset", "manual_undo", "manual_redo",
            "candidate_accept", "candidate_restore",
        )
        if busy:
            self._busy_enabled_snapshot = {}
            for name in names:
                widget = getattr(self, name, None)
                if widget is None:
                    continue
                self._busy_enabled_snapshot[name] = bool(widget.isEnabled())
                widget.setEnabled(False)
        else:
            snapshot = getattr(self, "_busy_enabled_snapshot", {})
            for name, enabled in dict(snapshot).items():
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.setEnabled(bool(enabled))
            self._busy_enabled_snapshot = {}

    def _set_inspector_panel(self, index: int):
        index = 1 if int(index) == 1 else 0
        self.inspector_stack.setCurrentIndex(index)
        self.inspector_settings_tab.setChecked(index == 0)
        self.inspector_review_tab.setChecked(index == 1)

    def _toggle_page_rail(self, hidden: bool):
        self._page_rail_user_hidden=bool(hidden)
        self.page_rail.setVisible(not bool(hidden))
        self.page_rail_toggle.setText("展开页面栏" if hidden else "收起页面栏")
        QTimer.singleShot(0,self._update_responsive_workbench)

    def _toggle_inspector(self, hidden: bool):
        self.side.setVisible(not bool(hidden))
        self.inspector_toggle.setText("显示参数栏" if hidden else "隐藏参数栏")
        if not hidden:
            QTimer.singleShot(0,self._update_responsive_workbench)

    def _update_responsive_workbench(self):
        """Switch the workbench to a vertical stack when horizontal space is tight.

        The previous fixed horizontal splitter required roughly preview(360) +
        sidebar(285) + margins.  On a narrow/split-screen macOS window Qt kept
        those minimums and simply clipped the right side of the settings panel.
        Compact mode gives both areas the full available width instead.
        """
        if not hasattr(self, "split") or not hasattr(self, "side"):
            return
        if not self.side.isVisible():
            return
        compact = self.width() < 900
        if self._compact_layout is compact:
            return
        self._compact_layout = compact
        if compact:
            self.page_rail.setVisible(False)
            self.split.setOrientation(Qt.Orientation.Vertical)
            self.side.setMinimumWidth(0); self.side.setMaximumWidth(16777215)
            self.side.setMinimumHeight(250); self.side.setMaximumHeight(16777215)
            h=max(520,self.height())
            self.split.setSizes([0, max(280,int(h*0.56)), max(250,int(h*0.44))])
            self.split.setStretchFactor(0,0); self.split.setStretchFactor(1,1); self.split.setStretchFactor(2,1)
        else:
            self.page_rail.setVisible(not bool(getattr(self, "_page_rail_user_hidden", False)))
            self.split.setOrientation(Qt.Orientation.Horizontal)
            self.side.setMinimumHeight(0); self.side.setMaximumHeight(16777215)
            self.side.setMinimumWidth(430); self.side.setMaximumWidth(540)
            w=max(1100,self.width())
            rail_w=min(190,max(150,int(w*0.13)))
            side_w=min(500,max(430,int(w*0.31)))
            canvas_w=max(560,w-rail_w-side_w)
            self.split.setSizes([rail_w,canvas_w,side_w])
            self.split.setStretchFactor(0,0); self.split.setStretchFactor(1,1); self.split.setStretchFactor(2,0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._update_responsive_workbench)

    def _active_pixel_mode_config(self):
        cfg = self.window.state.config
        mode = str(cfg.transfer.mode or "direct_patch")
        if mode == "direct_patch": return cfg.direct_patch
        if mode == "mask_replace": return cfg.mask_replace
        if mode == "hybrid": return cfg.hybrid.mask
        if mode == "reletter": return cfg.reletter.candidates
        return None

    def refresh_mode_controls(self):
        """Load the selected mode's private defaults/state into shared UI shells."""
        c = self._active_pixel_mode_config()
        if c is None: return
        widgets = [self.paired,self.skip_ocr,self.pixel_exact,self.full_bubble_patch,self.preserve_border,self.blur_guard,self.preserve_source_layout,self.fidelity,self.local,self.sr,self.sr_model,self.iou,self.coverage,self.expand_direct_range]
        for w in widgets: w.blockSignals(True)
        try:
            self.paired.setChecked(bool(getattr(c,"paired_diff_enabled",True)))
            self.skip_ocr.setChecked(bool(getattr(c,"paired_diff_skip_ocr",True)))
            self.pixel_exact.setChecked(bool(getattr(c,"exact_identity_copy",True)))
            self.full_bubble_patch.setChecked(bool(getattr(c,"rigid_container_full_patch_enabled",True)))
            self.preserve_border.setChecked(bool(getattr(c,"preserve_target_border",True)))
            self.blur_guard.setChecked(bool(getattr(c,"reject_blurry_source",True)))
            self.preserve_source_layout.setChecked(bool(getattr(c,"photo_pair_preserve_sharp_source_layout",True)))
            idx=self.fidelity.findData(str(getattr(c,"text_fidelity_mode","auto"))); self.fidelity.setCurrentIndex(idx if idx>=0 else 0)
            idx=self.local.findText(str(getattr(c,"local_fit","ecc"))); self.local.setCurrentIndex(idx if idx>=0 else 0)
            idx=self.sr.findText(str(getattr(c,"sr_backend","auto"))); self.sr.setCurrentIndex(idx if idx>=0 else 0)
            self.sr_model.setText(str(getattr(c,"sr_model_path","") or ""))
            self.iou.setValue(float(getattr(c,"min_mask_iou",.80)))
            self.coverage.setValue(float(getattr(c,"min_target_coverage",.985)))
            self.expand_direct_range.setChecked(bool(getattr(c,"source_direct_expand_candidate_range",False)))
        finally:
            for w in widgets: w.blockSignals(False)

    def _sync(self):
        c=self._active_pixel_mode_config()
        # v1.2.0: publication blocking is removed.  Keep the legacy config field
        # false so old saved configs cannot silently restore the retired gate.
        safety=False
        if c is not None:
            c.publication_safety_enabled=False; c.paired_diff_enabled=self.paired.isChecked(); c.paired_diff_skip_ocr=self.skip_ocr.isChecked(); c.exact_identity_copy=self.pixel_exact.isChecked(); c.rigid_container_full_patch_enabled=self.full_bubble_patch.isChecked(); c.rigid_container_full_patch_preserve_target_border=self.preserve_border.isChecked(); c.preserve_target_border=self.preserve_border.isChecked(); c.reject_blurry_source=self.blur_guard.isChecked(); c.fallback_reletter_on_blur=False; c.photo_pair_fallback_reletter_missing=False; c.photo_pair_prefer_reletter_with_ocr=False; c.strict_mask_replace_no_ocr_reletter=True; c.photo_pair_preserve_sharp_source_layout=self.preserve_source_layout.isChecked(); c.text_fidelity_mode=self.fidelity.currentData() or "auto"; c.local_fit=self.local.currentText(); c.sr_backend=self.sr.currentText(); c.sr_model_path=self.sr_model.text().strip() or None; c.min_mask_iou=float(self.iou.value()); c.min_target_coverage=float(self.coverage.value())
        if str(self.window.state.config.transfer.mode or "") == "direct_patch":
            d=self.window.state.config.direct_patch
            d.publication_safety_enabled=False
            d.source_direct_fail_on_artwork_rejections=bool(safety)
            d.allow_target_aware_colored_composite=True
            d.source_direct_colored_preserve_target_fill=True
            d.source_direct_expand_candidate_range=bool(self.expand_direct_range.isChecked())
        bcfg=self.window.state.config.bubbles
        bcfg.detector_strategy=self.detector_strategy.currentData() or "primary_conditional_aux"
        bcfg.primary_detector=self.primary_detector.currentData() or "koharu_layout"
        aux=[]
        for key,cb in [("geometry_white",self.aux_geometry_white),("mangalens",self.aux_mangalens),("rtdetr_v2",self.aux_rtdetr),("ysg_obb",self.aux_ysg),("koharu_layout",self.aux_koharu),("ctd_sidecar",self.aux_ctd),("sidecar",self.aux_sidecar)]:
            if cb.isChecked() and key != bcfg.primary_detector: aux.append(key)
        bcfg.auxiliary_detectors=aux; bcfg.sam2_refine_enabled=bool(self.sam2_refine.isChecked())
        legacy=next((x for x in aux if x in {"mangalens","ysg_obb","rtdetr_v2","sidecar","ctd_sidecar"}),None)
        if legacy is None and "geometry_white" in aux: legacy="seeded_white"
        bcfg.backend=legacy or bcfg.primary_detector
        aux_enabled=bcfg.detector_strategy != "primary_only"
        for key,cb in [("geometry_white",self.aux_geometry_white),("mangalens",self.aux_mangalens),("rtdetr_v2",self.aux_rtdetr),("ysg_obb",self.aux_ysg),("koharu_layout",self.aux_koharu),("ctd_sidecar",self.aux_ctd),("sidecar",self.aux_sidecar)]: cb.setEnabled(aux_enabled and key != bcfg.primary_detector)
        self.sam2_refine.setEnabled(aux_enabled)
        bcfg.mangalens_imgsz=int(self.detector_size.value()); bcfg.rtdetr_imgsz=int(self.detector_size.value()); bcfg.ysg_obb_imgsz=max(640,int(self.detector_size.value())); self.window.state.config.masking.max_dilation_px=int(self.clear_dilate.value()); self.window.state.config.inpainting.backend=self.inpaint_backend.currentData() or "auto"

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
        try:
            with PageRunGuard(page_dir, "gui:manual-clear-mask"):
                write_image(page_dir/"manual_clear_mask.png",dlg.result_mask())
        except Exception as exc:
            QMessageBox.warning(self,"蒙版保存失败",str(exc)); return
        self.current_view="clear_mask"
        for b,k in self.view_buttons: b.setChecked(k==self.current_view)
        self.window.statusBar().showMessage("已保存自动清除蒙版修订；正在后台生成去字预览…",5000)
        cfg=self.window.state.config.model_copy(deep=True)
        self.window.run_page_action(
            "生成去字预览", lambda: generate_remove_text_preview(page_dir,cfg),
            lambda _out: self.refresh(), failure_title="去字预览失败",
        )

    def _remove_text_only(self):
        page_dir=self._current_page_dir()
        if page_dir is None or not (page_dir/"target_original.png").exists():
            QMessageBox.information(self,"尚未处理","请先处理当前页。")
            return
        self._sync(); cfg=self.window.state.config.model_copy(deep=True)

        def done(out):
            self.current_view="removed"
            for b,k in self.view_buttons: b.setChecked(k=="removed")
            self.window.statusBar().showMessage(f"只清日文完成：{Path(str(out)).name}",5000)
            self.refresh()

        self.window.run_page_action(
            "只清日文", lambda: generate_remove_text_preview(page_dir,cfg), done,
            failure_title="只清日文失败",
        )

    def _apply_mask_review(self):
        page_dir=self._current_page_dir()
        if page_dir is None or not (page_dir/"project.json").exists():
            QMessageBox.information(self,"尚未处理","请先处理当前页。")
            return
        self._sync(); cfg=self.window.state.config.model_copy(deep=True)

        def done(final):
            final=Path(str(final)); self.window.state.last_result_path=str(final)
            pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""; proj=self.window.state.projects_by_page.get(page_id)
            if proj is not None: proj.artifacts["final"]=str(final)
            self._sync_reviewed_book_final(final); self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage("已把当前清除蒙版应用到最终结果。",5000)
            self.refresh()

        self.window.run_page_action(
            "应用复核蒙版", lambda: apply_review_page(page_dir,cfg), done,
            failure_title="应用蒙版失败",
        )

    def _reset_clear_mask(self):
        page_dir=self._current_page_dir()
        if page_dir is None: return
        for name in ("manual_clear_mask.png","removed_text_preview.png","remove_text_stage.json"):
            try: (page_dir/name).unlink(missing_ok=True)
            except OSError: pass
        self.current_view="clear_mask"
        for b,k in self.view_buttons: b.setChecked(k=="clear_mask")
        self.window.statusBar().showMessage("已恢复当前页自动清除蒙版。",4000); self.refresh()


    def _edit_force_transfer_mask(self):
        page_dir = self._current_page_dir()
        if page_dir is None or not (page_dir / "target_original.png").exists() or not (page_dir / "project.json").exists():
            QMessageBox.information(self, "尚未处理", "请先处理当前页，再使用人工强制迁移蒙版。")
            return
        import cv2
        import numpy as np
        display_path = page_dir / "final_reviewed.png"
        if not display_path.exists():
            display_path = page_dir / "final.png"
        if not display_path.exists():
            display_path = page_dir / "target_original.png"
        target = cv2.imread(str(page_dir / "target_original.png"), cv2.IMREAD_COLOR)
        if target is None:
            QMessageBox.warning(self, "无法读取", "当前页 target_original.png 无法读取。")
            return
        mask_path = page_dir / "manual_force_transfer_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if mask is None or mask.shape != target.shape[:2]:
            mask = np.zeros(target.shape[:2], dtype=np.uint8)
        project = normalize_project(load_json(page_dir / "project.json"))
        try:
            auto_original, _auto_source, auto_diag = manual_force_auto_evidence_masks(
                page_dir, project=project, target=target, include_override=False
            )
            override_path = page_dir / "manual_force_auto_target_override.png"
            auto_target = cv2.imread(str(override_path), cv2.IMREAD_GRAYSCALE) if override_path.exists() else None
            if auto_target is None or auto_target.shape != target.shape[:2]:
                auto_target = auto_original.copy()
            else:
                auto_target = (auto_target > 0).astype(np.uint8) * 255
        except Exception:
            auto_original = np.zeros(target.shape[:2], dtype=np.uint8)
            auto_target = auto_original.copy()
            auto_diag = {}
        settings_path = page_dir / "manual_force_settings.json"
        old_settings = load_json(settings_path) if settings_path.exists() else {}
        auto_default = bool(old_settings.get("use_auto_evidence", True))
        dlg = MaskEditorDialog(
            display_path, mask, self,
            title="人工强制迁移蒙版 · 人工 + OCR/自动检测",
            hint_text=(
                "红色 = 人工蒙版；蓝色 = OCR/自动检测蒙版；橙色 = 两层重合。"
                "现在两层都可以直接编辑：先选“人工蒙版”或“OCR/自动检测蒙版”，再用“涂抹/消除”画笔修正。"
                "蓝色自动区域改错后会保存为本页自动蒙版修订，不需要先复制到人工层；“恢复自动原始”可撤回对蓝色层的修改。"
                "重新处理时只使用修订后的紧凑文字蒙版，避免把整块气泡/画面当成文字区域。"
            ),
            save_label="保存并按复合蒙版重跑",
            reference_mask=auto_target,
            reference_original_mask=auto_original,
            reference_label="OCR / 自动检测",
            auto_assist_default=auto_default,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        write_image(mask_path, dlg.result_mask())
        edited_auto = dlg.result_reference_mask()
        override_path = page_dir / "manual_force_auto_target_override.png"
        if np.array_equal(edited_auto, auto_original):
            try: override_path.unlink(missing_ok=True)
            except OSError: pass
            auto_override = False
        else:
            write_image(override_path, edited_auto); auto_override = True
        save_json(page_dir / "manual_force_settings.json", {
            "schema": "manga_hd_translation_transfer.manual_force_settings.v3",
            "owner_transfer_mode": str(as_dict(project.get("meta")).get("transfer_mode", "") or self.window.state.config.transfer.mode),
            "use_auto_evidence": bool(dlg.use_auto_evidence()),
            "auto_reference_pixels": int(cv2.countNonZero(edited_auto)),
            "auto_reference_original_pixels": int(cv2.countNonZero(auto_original)),
            "auto_target_override": bool(auto_override),
            "auto_reference": auto_diag,
        })
        # Editing the blue OCR/automatic layer can invalidate already-generated
        # Chinese transfer layers or a flattened final result.  Rebuild the page
        # from the original pair and then re-apply this page's saved review
        # state; otherwise stale symbols from old automatic layers can survive
        # even after the automatic mask itself was erased.
        self.current_view = "result"
        for b, k in self.view_buttons:
            b.setChecked(k == "result")
        self.window.statusBar().showMessage(
            "已保存人工/OCR/自动蒙版修订，正在清理旧图层并重新处理当前页…", 7000
        )
        self.refresh()
        self.window.run_current_page(reapply_review_after_process=True)

    def _reset_force_transfer_mask(self):
        page_dir = self._current_page_dir()
        if page_dir is None:
            return
        cfg=self.window.state.config.model_copy(deep=True)

        def done(final):
            if final is not None:
                final=Path(str(final)); self.window.state.last_result_path = str(final)
                self._sync_reviewed_book_final(final)
            self.current_view = "result"
            for b, k in self.view_buttons:
                b.setChecked(k == "result")
            self.window.statusBar().showMessage("已清空人工强制迁移蒙版，并恢复使用该工具前的稳定结果。", 6000)
            self.refresh()

        self.window.run_page_action(
            "清空强制迁移蒙版", lambda: reset_manual_force_transfer_review(page_dir,cfg), done,
            failure_title="清空强制迁移蒙版失败",
        )


    def _edit_target_layer_erase(self):
        page_dir = self._current_page_dir()
        if page_dir is None or not (page_dir / "target_original.png").exists():
            QMessageBox.information(self, "尚未处理", "请先处理当前页，再使用 TARGET 日文层擦除画笔。")
            return
        import cv2
        import numpy as np
        display_path = page_dir / "final_reviewed.png"
        if not display_path.exists():
            display_path = page_dir / "final.png"
        if not display_path.exists():
            QMessageBox.information(self, "没有最终结果", "当前页还没有最终结果可供收尾。")
            return
        target = cv2.imread(str(page_dir / "target_original.png"), cv2.IMREAD_COLOR)
        if target is None:
            QMessageBox.warning(self, "无法读取", "当前页 target_original.png 无法读取。")
            return
        mask_path = page_dir / "manual_target_layer_erase_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if mask is None or mask.shape != target.shape[:2]:
            mask = np.zeros(target.shape[:2], dtype=np.uint8)
        selected_fill_mode = str(self.target_erase_mode.currentData() or "auto")
        display = cv2.imread(str(display_path), cv2.IMREAD_COLOR)
        def _target_erase_live_preview(mask_value):
            from .review_apply import _protected_chinese_mask
            raw=(np.asarray(mask_value,dtype=np.uint8)>0).astype(np.uint8)*255
            if cv2.countNonZero(raw)==0: return display.copy()
            expanded=cv2.dilate(raw,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
            protect,_=_protected_chinese_mask(page_dir, target.shape[:2], margin_px=1)
            effective=expanded.copy(); effective[protect>0]=0
            if selected_fill_mode == "pure_white":
                cleaned=target.copy(); cleaned[effective>0]=255
            else:
                cleaned=cv2.inpaint(target,effective,3.0,cv2.INPAINT_TELEA)
            out=display.copy(); out[effective>0]=cleaned[effective>0]; return out
        dlg = MaskEditorDialog(
            display_path, mask, self,
            title="只擦 TARGET 日文层 · 中文硬保护",
            hint_text=("红色 = 笔刷范围。当前模式：" + ("纯白涂抹；" if selected_fill_mode == "pure_white" else "智能恢复 TARGET；") +
                       "可刷残留日文、黑点、短线、标点和符号。画布会实时预览背景修复；正式保存时仍按中文保护层计算有效区域。"),
            save_label="保存并应用 TARGET 擦除", preview_fn=_target_erase_live_preview,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        write_image(mask_path, dlg.result_mask())
        save_json(page_dir / "target_layer_erase_settings.json", {
            "schema": "manga_hd_translation_transfer.target_layer_erase_settings.v1",
            "dilate_px": 1,
            "protect_chinese_margin_px": 1,
            "fill_mode": selected_fill_mode,
        })
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            final = apply_target_layer_erase_review(page_dir, self.window.state.config.model_copy(deep=True))
            self.window.state.last_result_path = str(final)
            pair = self.window.current_pair(); page_id = page_id_for_pair(pair) if pair is not None else ""
            proj = self.window.state.projects_by_page.get(page_id)
            if proj is not None:
                proj.artifacts["final"] = str(final)
            self._sync_reviewed_book_final(final)
            self.current_view = "target_erase"
            for b, k in self.view_buttons:
                b.setChecked(k == self.current_view)
            self.window.statusBar().showMessage("TARGET 层擦除已应用：只改日文母版层，中文图层已硬保护。", 6000)
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, "TARGET 层擦除失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _edit_target_layer_restore(self):
        page_dir = self._current_page_dir()
        if page_dir is None or not (page_dir / "target_original.png").exists():
            QMessageBox.information(self, "尚未处理", "请先处理当前页，再使用 TARGET 日文层恢复画笔。")
            return
        import cv2
        import numpy as np
        display_path = page_dir / "final_reviewed.png"
        if not display_path.exists():
            display_path = page_dir / "final.png"
        if not display_path.exists():
            QMessageBox.information(self, "没有最终结果", "当前页还没有最终结果可供恢复。")
            return
        target = cv2.imread(str(page_dir / "target_original.png"), cv2.IMREAD_COLOR)
        if target is None:
            QMessageBox.warning(self, "无法读取", "当前页 target_original.png 无法读取。")
            return
        mask_path = page_dir / "manual_target_layer_restore_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if mask is None or mask.shape != target.shape[:2]:
            mask = np.zeros(target.shape[:2], dtype=np.uint8)
        display = cv2.imread(str(display_path), cv2.IMREAD_COLOR)
        def _target_restore_live_preview(mask_value):
            raw=(np.asarray(mask_value,dtype=np.uint8)>0).astype(np.uint8)*255
            if cv2.countNonZero(raw)==0: return display.copy()
            out=display.copy(); out[raw>0]=target[raw>0]; return out
        dlg = MaskEditorDialog(
            display_path, mask, self,
            title="恢复 TARGET 日文层",
            hint_text="红色 = 恢复范围。该工具会把笔刷区域直接恢复成 TARGET 原始日文图层与背景，可用于去掉误显示的中文。",
            save_label="保存并应用 TARGET 恢复", preview_fn=_target_restore_live_preview,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        write_image(mask_path, dlg.result_mask())
        save_json(page_dir / "target_layer_restore_settings.json", {
            "schema": "manga_hd_translation_transfer.target_layer_restore_settings.v1",
            "dilate_px": 0,
        })
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            final = apply_target_layer_restore_review(page_dir)
            self.window.state.last_result_path = str(final)
            pair = self.window.current_pair(); page_id = page_id_for_pair(pair) if pair is not None else ""
            proj = self.window.state.projects_by_page.get(page_id)
            if proj is not None:
                proj.artifacts["final"] = str(final)
            self._sync_reviewed_book_final(final)
            self.current_view = "target_restore"
            for b, k in self.view_buttons:
                b.setChecked(k == self.current_view)
            self.window.statusBar().showMessage("TARGET 层恢复已应用：笔刷区域已恢复原始日文图层。", 6000)
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, "TARGET 层恢复失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()


    def _reset_target_layer_erase(self):
        page_dir = self._current_page_dir()
        if page_dir is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            final = reset_target_layer_erase_review(page_dir)
            if final is not None:
                self.window.state.last_result_path = str(final)
                self._sync_reviewed_book_final(final)
            self.current_view = "result"
            for b, k in self.view_buttons:
                b.setChecked(k == "result")
            self.window.statusBar().showMessage("已清空 TARGET 层擦除，并恢复擦除前结果。", 5000)
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, "恢复 TARGET 层失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()


    def _reset_target_layer_restore(self):
        page_dir = self._current_page_dir()
        if page_dir is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            final = reset_target_layer_restore_review(page_dir)
            if final is not None:
                self.window.state.last_result_path = str(final)
                self._sync_reviewed_book_final(final)
            self.current_view = "result"
            for b, k in self.view_buttons:
                b.setChecked(k == "result")
            self.window.statusBar().showMessage("已清空 TARGET 层恢复，并恢复恢复前结果。", 5000)
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, "恢复 TARGET 层失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()


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
            logger.debug("preview cache invalidation failed", exc_info=True)
        self.current_view="result"
        for b,k in self.view_buttons: b.setChecked(k=="result")
        try:
            changed_rows=len(rows)
            self.window.statusBar().showMessage(
                f"人工补漏已应用：已同步 final_reviewed.png → final.png（{changed_rows} 个区域）。",
                6000,
            )

        except Exception:
            logger.debug("manual effect status update failed", exc_info=True)

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
        remembered_mode = str(row.get("mode", "") or "")
        self.window.state.last_manual_effect_mode = remembered_mode
        if remembered_mode:
            _app_settings().setValue("review/last_manual_effect_mode", remembered_mode)
        pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""; proj=self.window.state.projects_by_page.get(page_id)
        if proj is not None:
            proj.artifacts["final"] = str(result.final)
            proj.artifacts["final_reviewed"] = str(result.final_reviewed)
        self._sync_reviewed_book_final(result.final_reviewed)

        try:
            self.image.clear_cache()
        except Exception:
            logger.debug("manual effect preview cache invalidation failed", exc_info=True)
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
        if preset_candidate:
            initial_mode = str(preset_candidate.get("suggested_manual_mode", "reveal_text") or "reveal_text")
        else:
            remembered = str(getattr(self.window.state, "last_manual_effect_mode", "") or "")
            if not remembered:
                remembered = str(_app_settings().value("review/last_manual_effect_mode", "") or "")
            initial_mode = remembered if remembered in {"reveal_text", "effect_text", "white_bubble_text"} else None
        self._start_manual_gui_flow(page_dir,preset_candidate)
        self._trace_manual_gui_flow(page_dir,"manual_dialog_opened",{"initial_bbox":initial_bbox,"initial_mode":initial_mode or ""})
        def _commit(row,reveal,reveal_patch):
            return self._commit_manual_effect_dialog_result(page_dir,row,reveal,reveal_patch,preset_candidate)
        def _trace(stage,payload):
            self._trace_manual_gui_flow(page_dir,stage,payload)
        try:
            dlg=ManualEffectDialog(source_path,target_path,project,self,initial_bbox=initial_bbox,initial_mode=initial_mode,commit_handler=_commit,trace_handler=_trace,config=self.window.state.config)
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

    def _reprocess_current_page(self):
        page_dir = self._current_page_dir()
        if page_dir is None or not (page_dir/"project.json").exists():
            QMessageBox.information(self, "尚未处理", "请先处理当前页，再使用“重新处理当前页”。")
            return
        self._sync()
        self.window.run_current_page(reapply_review_after_process=True)

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
        return resolve_page_workspace(self.window.state.output_dir, pair, project, self.window.state.restored_page_roots.get(page_id))

    def _manual_queue(self):
        ws=self._workspace()
        return list(ws.review_regions) if ws is not None else []

    @staticmethod
    def _manual_region_key(row: dict) -> str:
        return str(row.get("target_region_id") or row.get("target_unit_id") or row.get("target_bubble_id") or "")


    def _ocr_editor_context(self):
        ws=self._workspace()
        if ws is None or ws.page_root is None or not ws.project_path or not ws.project_path.exists():
            return None
        try:
            project=normalize_project(load_json(ws.project_path))
        except Exception:
            return None
        mode=str(as_dict(project.get("meta")).get("transfer_mode") or self.window.state.config.transfer.mode or "").strip().lower()
        if not is_ocr_edit_mode(mode):
            return None
        pair=self.window.current_pair()
        if pair is None:
            return None
        source_path=ws.page_root/"source_original.png"
        target_path=ws.page_root/"target_original.png"
        if not source_path.exists(): source_path=Path(pair.source_path)
        if not target_path.exists(): target_path=Path(pair.target_path)
        return ws,project,mode,source_path,target_path

    def _open_ocr_block_editor(self):
        ctx=self._ocr_editor_context()
        if ctx is None:
            QMessageBox.information(self,"当前模式不可用","人工 OCR 文本块只属于“精准蒙版+OCR”和“OCR重排”。请先处理当前页并切换到其中一个 OCR 模式。")
            return
        ws,project,mode,source_path,target_path=ctx
        dialog=OCRBlockEditorDialog(ws.page_root,source_path,target_path,project,self.window.state.config,mode,parent=self)
        if dialog.exec()!=QDialog.DialogCode.Accepted:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            final=apply_ocr_edit_blocks(ws.page_root,project,self.window.state.config.model_copy(deep=True))
            self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage("人工 OCR 文本块已保存并局部重绘。",4500)
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self,"应用人工 OCR 失败",str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _reset_ocr_blocks(self):
        ctx=self._ocr_editor_context()
        if ctx is None:
            return
        ws,project,mode,_source,_target=ctx
        rows=load_ocr_blocks(ws.page_root,mode)
        if not rows:
            self.window.statusBar().showMessage("当前页没有人工 OCR 文本块。",2500); return
        if QMessageBox.question(self,"清空人工 OCR","清空当前页所有人工 OCR 文本块并恢复进入 OCR 编辑前的结果？") != QMessageBox.StandardButton.Yes:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            save_ocr_blocks(ws.page_root,mode,[])
            final=reset_ocr_edit_blocks(ws.page_root,project,self.window.state.config.model_copy(deep=True))
            self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage("已清空人工 OCR 文本块。",3500); self.refresh()
        except Exception as exc:
            QMessageBox.critical(self,"清空人工 OCR 失败",str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _manual_selection_changed(self, *_args):
        queue=self._manual_queue(); idx=self.manual_target.currentIndex()
        if idx < 0 or idx >= len(queue):
            return
        row=dict(queue[idx]); key=self._manual_region_key(row)
        page_dir=self._current_page_dir()
        override={}
        if page_dir is not None:
            op=page_dir/"review_overrides.json"
            overrides=normalize_overrides(load_json(op) if op.exists() else {})
            for item in _json_dict_rows(overrides.get("manual_reletter")):
                if self._manual_region_key(item)==key:
                    override=dict(item); break
        current={**row, **override}
        self.manual_text.blockSignals(True)
        self.manual_text.setPlainText(str(current.get("text") or current.get("auto_text") or ""))
        self.manual_text.blockSignals(False)
        ori=str(current.get("orientation") or current.get("auto_orientation") or "auto")
        oi=self.manual_orientation.findData(ori); self.manual_orientation.setCurrentIndex(max(0,oi))
        br=str(current.get("line_break_mode") or "smart")
        bi=self.manual_break_mode.findData(br); self.manual_break_mode.setCurrentIndex(max(0,bi))
        lm=str(current.get("layout_mode") or getattr(self.window.state.config.lettering,"layout_mode","smart_scaling"))
        li=self.manual_layout_mode.findData(lm); self.manual_layout_mode.setCurrentIndex(max(0,li))
        font_value=str(current.get("font_path") or current.get("auto_font_path") or "")
        self.manual_font.setText(font_value)
        if hasattr(self,"manual_font_preset"):
            preset=font_value if font_value in {"sans","serif","rounded","comic"} else "custom"
            pi=self.manual_font_preset.findData(preset); self.manual_font_preset.setCurrentIndex(max(0,pi))
        self.manual_font_size.setValue(int(current.get("font_size") or current.get("auto_font_size") or 0))
        self.manual_columns.setValue(int(current.get("columns") or 0))
        if hasattr(self,"manual_line_spacing"):
            spacing=current.get("line_spacing_ratio")
            self.manual_line_spacing.setValue(float(spacing) if spacing is not None else -1.0)
        is_auto_reletter=str(row.get("review_kind") or "")=="reletter_auto"
        self.manual_reset.setEnabled(bool(is_auto_reletter and override))
        self.candidate_accept.setEnabled(not is_auto_reletter)
        self.candidate_restore.setEnabled(not is_auto_reletter)
        self._refresh_review_history_status()

    def _apply_manual_font_preset(self):
        value=str(self.manual_font_preset.currentData() or "custom") if hasattr(self,"manual_font_preset") else "custom"
        if value != "custom": self.manual_font.setText(value)

    def _choose_manual_reletter_font(self):
        start=str(Path.home())
        current=self.manual_font.text().strip()
        if current and Path(current).exists(): start=str(Path(current).parent)
        path,_=QFileDialog.getOpenFileName(self,"选择当前 Region 字体",start,"Fonts (*.ttf *.ttc *.otf *.otc);;All files (*)")
        if path: self.manual_font.setText(path)

    def _refresh_review_history_status(self):
        page_dir=self._current_page_dir()
        if page_dir is None or not hasattr(self,"manual_history_status"):
            return
        undo_n,redo_n=review_history_counts(page_dir)
        self.manual_history_status.setText(f"历史 {undo_n} / 重做 {redo_n}")
        self.manual_undo.setEnabled(undo_n>0); self.manual_redo.setEnabled(redo_n>0)

    def _review_history_step(self, direction: str):
        page_dir=self._current_page_dir()
        if page_dir is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            state=undo_review_state(page_dir) if direction=="undo" else redo_review_state(page_dir)
            if state is None:
                self.window.statusBar().showMessage("没有可用的编辑历史。",2500); return
            final=apply_review_page(page_dir,self.window.state.config.model_copy(deep=True))
            self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage("已撤销人工编辑" if direction=="undo" else "已重做人工编辑",3500)
            self.refresh(); self._refresh_review_history_status()
        except Exception as exc:
            QMessageBox.critical(self,"编辑历史失败",str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _reset_manual_reletter(self):
        ws=self._workspace(); queue=self._manual_queue(); idx=self.manual_target.currentIndex()
        if ws is None or ws.page_root is None or not queue or idx < 0 or idx >= len(queue):
            return
        row=dict(queue[idx]); key=self._manual_region_key(row)
        if not key: return
        page_dir=ws.page_root; path=page_dir/"review_overrides.json"
        overrides=normalize_overrides(load_json(path) if path.exists() else {})
        entries=[x for x in _json_dict_rows(overrides.get("manual_reletter")) if self._manual_region_key(x)!=key]
        overrides["manual_reletter"]=entries
        overrides["status"]="reviewed" if not entries else "reviewed_with_manual_reletter"
        try:
            project_now=normalize_project(load_json(ws.project_path))
            overrides["owner_transfer_mode"]=str(as_dict(project_now.get("meta")).get("transfer_mode", "") or self.window.state.config.transfer.mode)
        except Exception:
            overrides["owner_transfer_mode"]=str(self.window.state.config.transfer.mode)
        record_review_state(page_dir, normalize_overrides(load_json(path) if path.exists() else {}), "reset_manual_reletter")
        save_json(path,overrides)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            final=apply_review_page(page_dir,self.window.state.config.model_copy(deep=True))
            self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage(f"已恢复自动重排：{key}",4500)
            self.refresh()
        except Exception as exc: QMessageBox.critical(self,"恢复自动重排失败",str(exc))
        finally: QApplication.restoreOverrideCursor()

    def _sync_reviewed_book_final(self, final: str | Path):
        """Synchronize reviewed output through one core result-state contract."""
        final_path = Path(final)
        if final_path.exists() and final_path.name == "final_reviewed.png":
            commit_reviewed_result(final_path.parent, final_path)
        pair=self.window.current_pair()
        if pair is None or not self.window.state.output_dir:
            return
        restored = bool(self.window.state.restored_page_origin.get(page_id_for_pair(pair)))
        final_name = (page_id_for_pair(pair) if restored else Path(pair.target_path).stem) + ".png"
        dst=Path(self.window.state.output_dir)/"final"/final_name
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
        region_key=self._manual_region_key(row)
        target_id=str(row.get("target_bubble_id", ""))
        item={
            "review_kind": str(row.get("review_kind") or ""),
            "target_region_id": str(row.get("target_region_id") or ""),
            "target_unit_id": str(row.get("target_unit_id") or ""),
            "target_bubble_id": target_id,
            "target_bbox": as_list(row.get("target_bbox")),
            "text": text,
            "orientation": self.manual_orientation.currentData() or "auto",
            "line_break_mode": self.manual_break_mode.currentData() or "smart",
            "layout_mode": self.manual_layout_mode.currentData() or "smart_scaling",
            "font_path": self.manual_font.text().strip(),
            "font_size": int(self.manual_font_size.value()),
            "columns": int(self.manual_columns.value()),
            "line_spacing_ratio": None if float(self.manual_line_spacing.value()) < 0 else float(self.manual_line_spacing.value()),
            "reason": row.get("reason", ""),
        }
        entries=[x for x in entries if self._manual_region_key(x) != region_key]
        entries.append(item)
        overrides["manual_reletter"]=entries
        # Bubble-candidate decisions belong only to Direct/Mask review. Successful
        # reletter regions are edited by immutable region id and never co-opt those
        # route-specific restore/accept lists.
        if str(row.get("review_kind") or "") != "reletter_auto":
            overrides["restore_target_bubbles"]=[x for x in list(overrides.get("restore_target_bubbles",[]) or []) if str(x)!=target_id]
            overrides["accept_candidate_targets"]=[x for x in list(overrides.get("accept_candidate_targets",[]) or []) if str(x)!=target_id]
        overrides["status"]="reviewed_with_manual_reletter"
        try:
            project_now=normalize_project(load_json(ws.project_path))
            overrides["owner_transfer_mode"]=str(as_dict(project_now.get("meta")).get("transfer_mode", "") or self.window.state.config.transfer.mode)
        except Exception:
            overrides["owner_transfer_mode"]=str(self.window.state.config.transfer.mode)
        record_review_state(page_dir, normalize_overrides(load_json(override_path) if override_path.exists() else {}), "apply_manual_reletter")
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
            self.window.statusBar().showMessage(f"已更新重排文字：{region_key or target_id}",5000)
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
        try:
            project_now=normalize_project(load_json(ws.project_path))
            overrides["owner_transfer_mode"]=str(as_dict(project_now.get("meta")).get("transfer_mode", "") or self.window.state.config.transfer.mode)
        except Exception:
            overrides["owner_transfer_mode"]=str(self.window.state.config.transfer.mode)
        record_review_state(page_dir, normalize_overrides(load_json(override_path) if override_path.exists() else {}), f"candidate_{action}")
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
        self.refresh_mode_controls()
        total=len(s.pairs)
        if not total:
            self.page_rail.set_pages([], 0)
            self.image.set_image(None); self.page_caption.setText("未选择页面"); self.page_counter.setText("0 / 0")
            self.prev_page.setEnabled(False); self.next_page.setEnabled(False); self.view_status.setText("")
            self.manual_target.clear(); self.manual_effect_candidate_target.clear(); self.manual_status.setText("当前页没有待复核气泡"); self.manual_effect_status.setText("当前页暂无人工补漏区域"); self.manual_effect_candidate_status.setText("暂无待处理彩色/复杂文字候选")
            self.add_manual_effect.setEnabled(False); self.add_manual_effect_candidate.setEnabled(False); self.undo_manual_effect.setEnabled(False)
            return
        idx=max(0,min(s.selected_index,total-1)); pair=s.pairs[idx]
        self.page_rail.set_pages(s.pairs, idx)
        ws=self._workspace()
        self.page_counter.setText(f"{idx+1} / {total}")
        self.prev_page.setEnabled(idx>0); self.next_page.setEnabled(idx<total-1)
        page_root=ws.page_root if ws else None
        current_mode=str(((ws.meta or {}).get("transfer_mode") if ws is not None else "") or self.window.state.config.transfer.mode or "").strip().lower()
        ocr_editor_enabled=bool(page_root and is_ocr_edit_mode(current_mode))
        if hasattr(self,"open_ocr_block_editor"):
            self.open_ocr_block_editor.setEnabled(ocr_editor_enabled)
            rows=load_ocr_blocks(page_root,current_mode) if ocr_editor_enabled else []
            self.reset_ocr_blocks.setEnabled(bool(rows))
            if ocr_editor_enabled:
                scope_label="精准蒙版+OCR" if ocr_edit_scope(current_mode)=="mask_ocr" else "OCR重排"
                if ocr_edit_scope(current_mode)=="mask_ocr":
                    self.ocr_block_status.setText(
                        f"{scope_label} · 人工 OCR 文本块 {len(rows)} 个 · 人工框选=强制 OCR；自动 OCR 仅处理完全无精准蒙版覆盖的区域"
                    )
                else:
                    self.ocr_block_status.setText(f"{scope_label} · 人工 OCR 文本块 {len(rows)} 个 · 仅影响当前 OCR 流程")
            else:
                self.ocr_block_status.setText("人工 OCR 文本块仅在“精准蒙版+OCR / OCR重排”可用；其他模式完全隔离。")
        if page_root is not None and hasattr(self, "target_erase_mode"):
            settings_path = page_root / "target_layer_erase_settings.json"
            if settings_path.exists():
                try:
                    erase_settings = as_dict(load_json(settings_path))
                    mode_value = str(erase_settings.get("fill_mode", "auto") or "auto")
                    mode_index = self.target_erase_mode.findData(mode_value)
                    if mode_index >= 0 and self.target_erase_mode.currentIndex() != mode_index:
                        self.target_erase_mode.blockSignals(True); self.target_erase_mode.setCurrentIndex(mode_index); self.target_erase_mode.blockSignals(False)
                except Exception:
                    logger.debug("failed to restore target-layer erase settings", exc_info=True)
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
            "target_erase": str(page_root/"target_layer_erase_preview.png") if page_root and (page_root/"target_layer_erase_preview.png").exists() else "",
            "target_restore": str(page_root/"target_layer_restore_preview.png") if page_root and (page_root/"target_layer_restore_preview.png").exists() else "",
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
        self.undo_manual_effect.setEnabled(bool(effect_rows)); self.add_manual_effect.setEnabled(can_manual); self.add_manual_effect_candidate.setEnabled(bool(effect_candidates) and can_manual); self.manual_effect_candidate_target.setEnabled(bool(effect_candidates)); self.reprocess_current.setEnabled(can_manual)
        current_id=self.manual_target.currentData() if self.manual_target.count() else None
        self.manual_target.blockSignals(True); self.manual_target.clear()
        has_auto_reletter=False
        for row in queue:
            key=self._manual_region_key(row) or "待复核区域"
            tid=str(row.get("target_bubble_id", "")); sides=str(row.get("source_edge_sides", "")); candidate=bool(row.get("candidate_applied",False)); reason=str(row.get("reason", "")); kind=str(row.get("review_kind") or "")
            if kind=="reletter_auto":
                state="已自动 OCR重排 · 可直接编辑"
                has_auto_reletter=True
                label=f"{key} · {state}"
            elif reason=="photographed_text_without_ocr_reletter":
                state="摄影中文字形 · 可能模糊/扭曲"; label=f"{tid or key} · {state} · {sides or '可编辑/还原'}"
            elif candidate:
                state="已先替换中文候选 · 可能不完整/不准确"; label=f"{tid or key} · {state} · {sides or '可编辑/还原'}"
            else:
                state="待补中文"; label=f"{tid or key} · {state} · {sides or '可编辑/还原'}"
            self.manual_target.addItem(label,key)
        if current_id:
            mi=self.manual_target.findData(current_id)
            if mi>=0: self.manual_target.setCurrentIndex(mi)
        self.manual_target.blockSignals(False)
        if queue:
            self.manual_status.setText((f"当前页 {len(queue)} 个 OCR重排 Region 均可人工修改；修改只重绘当前 Region。" if has_auto_reletter else f"发现 {len(queue)} 个待复核区域；默认先给中文候选。可接受、重新编辑或还原日文。"))
        else:
            self.manual_status.setText("当前页没有可编辑/待复核文字区域")
        enabled=bool(queue)
        for widget in [self.manual_target,self.manual_text,self.manual_orientation,self.manual_break_mode,self.manual_font_preset,self.manual_font,self.manual_font_pick,self.manual_font_size,self.manual_columns,self.manual_line_spacing,self.manual_apply,self.candidate_accept,self.candidate_restore,self.manual_reset]: widget.setEnabled(enabled)
        if enabled:
            self._manual_selection_changed()
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
                origin = s.restored_page_origin.get(page_id_for_pair(pair), "")
                run_state = {}
                if page_root is not None:
                    try:
                        state_path = page_root / "last_run_state.json"
                        if state_path.exists():
                            value = load_json(state_path); run_state = value if isinstance(value, dict) else {}
                    except Exception:
                        run_state = {}
                status = str(run_state.get("status") or "")
                mode_name = str(run_state.get("mode") or ((ws.meta or {}).get("transfer_mode") if ws else "") or self.window.state.config.transfer.mode)
                strategy_name = str(run_state.get("selected_strategy") or ((ws.meta or {}).get("selected_strategy") if ws else "") or "")
                if status == "failed":
                    self.view_status.setText(f"上次处理失败 · 当前显示上次成功结果 · 模式 {mode_name} · 点击顶部“日志”查看原因")
                else:
                    suffix = f" · 模式 {mode_name}" + (f" / {strategy_name}" if strategy_name else "")
                    self.view_status.setText((f"已恢复已有结果 · {origin} · 可继续人工补漏{suffix}" if origin else f"已同步到当前页{suffix}"))
        elif self.current_view in {"result","review","mask","clear_mask","chinese_layer","removed","target_erase"}:
            self.view_status.setText("本页尚无该输出")
        else:
            self.view_status.setText("")





class StudioWindow(QMainWindow):
    NAV_SECTIONS = [
        ("页面管理", "PAGE MANAGER"),
        ("识别与配准", "DETECT & ALIGN"),
        ("替换工作台", "TRANSFER DESK"),
        ("出版输出", "EXPORT"),
        ("设置", "SETTINGS"),
    ]

    @staticmethod
    def _vertical_nav_label(label: str) -> str:
        """Stable text-only navigation labels without decorative numbering."""
        return str(label)

    def __init__(self):
        super().__init__()
        self.state = StudioState()
        self.state.config.transfer.mode = "direct_patch"
        self.worker: PipelineWorker | None = None
        self._pending_pipeline_worker: PipelineWorker | None = None
        self._prepare_worker: AutoPrepareModelsWorker | None = None
        self._page_action_worker: PageActionWorker | None = None
        self._worker_is_single_page = False
        self._worker_page_id = ""
        self.setWindowTitle(APP_NAME)
        screen = QApplication.primaryScreen()
        # The outer native window may be small.  The business UI itself remains
        # at its known-good design geometry and is uniformly transformed by a
        # ResponsiveCanvasView after construction, so layouts never squeeze or
        # overlap individual controls.
        self.setMinimumSize(QSize(420, 280))
        if screen is not None:
            geo = screen.availableGeometry()
            # Start near the safe desktop design size when the monitor allows it,
            # but never request a window larger than the available work area.
            self.resize(
                max(420, min(1480, int(geo.width() * 0.94))),
                max(280, min(960, int(geo.height() * 0.92))),
            )
        else:
            self.resize(1480, 960)
        self._theme_name = _saved_theme_name()
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(style_for_theme(self._theme_name))

        root = QWidget(); root.setObjectName("root"); self.setCentralWidget(root)
        outer = QHBoxLayout(root); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)

        # Novel Formatter inspired workflow rail.  The business pages themselves
        # stay untouched: only their navigation shell moves from the top to the
        # left so page-management / recognition / transfer / export remain isolated.
        self.workflow_rail = QFrame(); self.workflow_rail.setObjectName("workflowRail")
        # Keep only the workflow rail narrow.  Do not steal width from the
        # right-hand editor/inspector where forms need room to remain legible.
        self.workflow_rail.setFixedWidth(188)
        rail = QVBoxLayout(self.workflow_rail); rail.setContentsMargins(10,12,10,10); rail.setSpacing(6)

        brand_title = QLabel("Folirina")
        brand_title.setObjectName("railProductName")
        brand_title.setToolTip(APP_NAME)
        rail.addWidget(brand_title)
        rail.addSpacing(14)
        rail_section = QLabel("工作流"); rail_section.setObjectName("railSection"); rail.addWidget(rail_section)

        self.nav_group = QButtonGroup(self); self.nav_group.setExclusive(True)
        self.stack = QStackedWidget(); self.pages = []
        for index, (label, _english) in enumerate(self.NAV_SECTIONS):
            button = QPushButton(self._vertical_nav_label(label))
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setAccessibleName(label)
            button.setToolTip(label)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.nav_group.addButton(button)
            button.setFixedHeight(42)
            rail.addWidget(button)
            button.clicked.connect(lambda _=False, i=index: self.show_page(i))
            self.pages.append(button)

        rail.addStretch(1)
        rail_tools_label = QLabel("工具"); rail_tools_label.setObjectName("railSection"); rail.addWidget(rail_tools_label)
        rail_tools = QGridLayout(); rail_tools.setContentsMargins(0,0,0,0); rail_tools.setHorizontalSpacing(6); rail_tools.setVerticalSpacing(6)
        self.theme_button = QPushButton(); self.theme_button.setObjectName("railTool")
        self.theme_button.clicked.connect(self.toggle_theme); self._refresh_theme_button()
        self.log_button = QPushButton("页面日志"); self.log_button.setObjectName("railTool")
        self.log_button.setToolTip("查看当前页最新处理日志、实际模式、OCR/配准/气泡绑定与失败原因。")
        self.log_button.clicked.connect(self.open_current_run_log)
        self.runtime_log_button = QPushButton("运行日志"); self.runtime_log_button.setObjectName("railTool")
        self.runtime_log_button.setToolTip(f"打开程序级运行日志目录：{runtime_log_dir()}")
        self.runtime_log_button.clicked.connect(self.open_runtime_log_dir)
        # Keep a hidden compatibility handle for worker-state code, but remove
        # the duplicate stop control from the lower-left rail.  ProjectPage owns
        # the single visible Stop button beside the processing actions.
        self.stop_button = QPushButton("停止任务")
        self.stop_button.setObjectName("stopTask"); self.stop_button.setEnabled(False); self.stop_button.setVisible(False)
        self.stop_button.setToolTip("安全停止当前迁移任务。已完成页面不会被删除。")
        self.stop_button.clicked.connect(self.cancel_worker)
        rail_tools.addWidget(self.theme_button,0,0); rail_tools.addWidget(self.log_button,0,1)
        rail_tools.addWidget(self.runtime_log_button,1,0,1,2)
        rail.addLayout(rail_tools)
        platform_badge = QLabel(desktop_platform_badge()); platform_badge.setObjectName("railPlatform"); platform_badge.setAlignment(Qt.AlignmentFlag.AlignCenter); rail.addWidget(platform_badge)
        version = QLabel(f"v{VERSION}"); version.setObjectName("railVersion")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter); version.setToolTip("Folirina"); rail.addWidget(version)
        outer.addWidget(self.workflow_rail, 0)

        content = QWidget(); content.setObjectName("contentShell")
        content_layout = QVBoxLayout(content); content_layout.setContentsMargins(0,0,0,0); content_layout.setSpacing(0)
        outer.addWidget(content, 1)

        self.project = ProjectPage(self)
        self.models = ModelPage(self)
        self.workbench = WorkbenchPage(self)
        self.export = ExportPage(self)
        self.settings = SettingsPage(self)
        for page in [self.project, self.models, self.workbench, self.export, self.settings]:
            self.stack.addWidget(page)
        content_layout.addWidget(self.stack, 1)
        self.pages[0].setChecked(True)

        self.progress = QProgressBar(); self.progress.setRange(0,100); self.progress.setValue(0); self.progress.setMaximumWidth(190)
        self.statusBar().addPermanentWidget(self.progress); self.statusBar().showMessage("就绪")
        # Coalesce rapid workflow-tab changes. The stacked page becomes visible
        # immediately; any expensive refresh is deferred to the next event-loop
        # slice and only the last requested tab is refreshed.
        self._pending_page_refresh = 0
        self._page_refresh_timer = QTimer(self); self._page_refresh_timer.setSingleShot(True)
        self._page_refresh_timer.timeout.connect(self._flush_page_refresh)

        # Whole-page scaling: detach the fully built business UI and embed that
        # *real* widget tree in one graphics proxy.  The inner layout is never
        # allowed below the safe 1480x960 desktop geometry.  For different
        # aspect ratios the logical canvas can grow, then one uniform transform
        # fits the entire interactive page into the actual OS window.
        design_root = self.takeCentralWidget()
        if design_root is None:
            raise RuntimeError("failed to detach design canvas")
        self._design_root = design_root
        self._responsive_scaler = ResponsiveCanvasView(
            design_root, design_size=QSize(1480, 960), maximum_scale=None,
            on_scale=self._responsive_scale_changed, parent=self,
        )
        self.setCentralWidget(self._responsive_scaler)
        QTimer.singleShot(0, self._responsive_scaler.capture)
        self.refresh_all()

    @property
    def theme_name(self) -> str:
        return self._theme_name

    def _refresh_theme_button(self) -> None:
        if not hasattr(self, "theme_button"):
            return
        if self._theme_name == "dark":
            self.theme_button.setText("浅色")
            self.theme_button.setToolTip("切换到浅色主题；选择会自动记住。")
        else:
            self.theme_button.setText("深色")
            self.theme_button.setToolTip("切换到深色主题；选择会自动记住。")

    def _apply_theme_style(self, scale: float | None = None) -> None:
        # The design widget is always styled at 100%.  ResponsiveCanvasView then
        # scales the already-laid-out page as one unit.  Scaling CSS here as well
        # would double-shrink fonts and controls and reintroduce layout collisions.
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(style_for_theme(self._theme_name, 1.0))
        canvas = getattr(self, "_responsive_scaler", None)
        if canvas is not None and hasattr(canvas, "sync_background"):
            QTimer.singleShot(0, canvas.sync_background)

    def set_theme(self, theme: str) -> None:
        name = normalize_theme(theme)
        changed = name != self._theme_name
        self._theme_name = name
        self._apply_theme_style()
        _app_settings().setValue(_THEME_SETTING_KEY, self._theme_name)
        self._refresh_theme_button()
        if hasattr(self, "project") and hasattr(self.project, "apply_theme"):
            self.project.apply_theme()
        if hasattr(self, "settings"):
            self.settings.sync_theme()
        if changed:
            logger.info("ui theme changed theme=%s", self._theme_name)
            self.statusBar().showMessage("已切换为深色主题" if self._theme_name == "dark" else "已切换为浅色主题", 3000)

    def toggle_theme(self):
        self.set_theme("light" if self._theme_name == "dark" else "dark")

    def _responsive_scale_changed(self, factor: float) -> None:
        # Do not restyle or resize any child here; only report the view transform.
        if hasattr(self, "settings"):
            self.settings.set_responsive_scale(factor)

    def resizeEvent(self, event):
        # The central ResponsiveCanvasView receives the native resize itself and
        # owns coalescing/settling. Scheduling it again from QMainWindow doubled
        # timer traffic on macOS while dragging the window.
        super().resizeEvent(event)

    def open_runtime_log_dir(self):
        path = runtime_log_dir()
        path.mkdir(parents=True, exist_ok=True)
        logger.info("open runtime log directory path=%s", path)
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            QMessageBox.information(self, "运行日志目录", str(path))

    def open_current_run_log(self):
        pair = self.current_pair()
        if pair is None:
            QMessageBox.information(self, "没有页面", "请先选择一个页面。")
            return
        out = str(self.state.output_dir or "").strip()
        if not out:
            QMessageBox.information(self, "没有输出目录", "当前项目还没有输出工作区。")
            return
        try:
            ws = resolve_page_workspace(out, pair, self.state.projects_by_page.get(page_id_for_pair(pair)), self.state.restored_page_roots.get(page_id_for_pair(pair)))
            page_root = Path(ws.page_root)
        except Exception as exc:
            QMessageBox.warning(self, "无法定位日志", str(exc)); return
        dlg = RunLogDialog(page_root, self); dlg.exec()

    def closeEvent(self, event):
        # Prevent QThread::~QThread aborts during PySide shutdown. The attached
        # macOS crash report showed ComponentProbeWorker still alive while QtCore
        # was finalizing. Read-only probes are drained here; processing workers
        # receive their existing cooperative cancellation signal.
        try:
            if hasattr(self, "models"):
                self.models.shutdown_background_probes()
        except Exception:
            logger.debug("background probe shutdown failed", exc_info=True)

        still_running = False
        try:
            if hasattr(self, "settings") and not self.settings.shutdown_background_workers():
                still_running = True
        except Exception:
            logger.debug("settings worker shutdown failed", exc_info=True)
            still_running = True
        for worker in (self.worker, self._prepare_worker, self._page_action_worker):
            try:
                if worker is None or not worker.isRunning():
                    continue
                if hasattr(worker, "request_cancel"):
                    worker.request_cancel()
                else:
                    worker.requestInterruption()
                worker.wait(1800)
                still_running = still_running or worker.isRunning()
            except Exception:
                logger.debug("worker shutdown failed", exc_info=True)
                still_running = True
        if still_running:
            # Never destroy a live QThread wrapper. Keep the window alive and let
            # the existing cooperative stop finish; the user can close again.
            self.statusBar().showMessage("正在安全结束后台任务，完成后即可退出。", 5000)
            event.ignore()
            return
        super().closeEvent(event)

    def _processing_busy_state(self):
        return compute_busy_state(
            pipeline_running=self.worker is not None and self.worker.isRunning(),
            prepare_running=self._prepare_worker is not None and self._prepare_worker.isRunning(),
            page_action_running=self._page_action_worker is not None and self._page_action_worker.isRunning(),
            settings_updating=hasattr(self, "settings") and self.settings.is_updating,
        )

    def _busy_running(self) -> bool:
        return self._processing_busy_state().busy

    def _set_busy(self, active: bool | None = None):
        state = self._processing_busy_state()
        busy = state.busy if active is None else bool(active)
        cancellable = state.cancellable
        self.stop_button.setEnabled(cancellable)
        if hasattr(self, "project"):
            self.project.cancel.setEnabled(busy)
            self.project.run_page.setEnabled(not busy)
            self.project.run_book.setEnabled(not busy)
            if hasattr(self.project, "continue_book"): self.project.continue_book.setEnabled(not busy)
            self.project.pair_btn.setEnabled(not busy)
            self.project.apply_type.setEnabled(not busy)
            self.project.reset_type.setEnabled(not busy)
            self.project.page_type.setEnabled(not busy)
        if hasattr(self, "workbench"):
            self.workbench.set_processing_busy(busy)
        if hasattr(self, "export"):
            self.export.run.setEnabled(not busy)

    def show_page(self, index: int):
        if not 0 <= int(index) < self.stack.count():
            return
        index = int(index)
        self.stack.setCurrentIndex(index)
        for i,b in enumerate(self.pages):
            b.setChecked(i == index)
        self._pending_page_refresh = index
        self._page_refresh_timer.start(18)

    def _flush_page_refresh(self):
        index = int(getattr(self, "_pending_page_refresh", self.stack.currentIndex()))
        if index != self.stack.currentIndex():
            return
        if index == 0: self.project.refresh()
        elif index == 1: self.models.refresh()
        elif index == 2: self.workbench.refresh()
        elif index == 3: self.export.refresh()
        elif index == 4: self.settings.refresh()

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

    def restore_existing_results(self):
        if self._busy_running():
            QMessageBox.information(self, "任务进行中", "请先停止或等待当前任务完成。")
            return
        start = self.state.output_dir or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "读取已有运行结果（请选择输出目录 / pages / 单页目录）", start)
        if not path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            session = scan_existing_results(path)
            self.state.output_dir = str(session.output_root)
            self.state.source_dir = session.source_dir
            self.state.target_dir = session.target_dir
            self.state.pairs = [row.pair for row in session.pages]
            self.state.selected_index = 0
            self.state.projects_by_page.clear(); self.state.batch_status.clear()
            self.state.restored_page_roots = {row.page_id: str(row.page_root) for row in session.pages}
            self.state.restored_page_origin = {row.page_id: "命令行/Codex 已有结果" for row in session.pages}
            self.state.unmatched_source.clear(); self.state.unmatched_target.clear()
            self.project._table_signature = None; self.project._thumb_signature = None
            self.load_page_marks()
            warning = f" · 跳过/警告 {len(session.warnings)}" if session.warnings else ""
            self.statusBar().showMessage(f"已恢复 {len(session.pages)} 页已有结果{warning}。可直接继续页面检查或进入替换工作台人工补漏。", 8000)
            # Restoring existing results is not the same as a fresh pairing pass.
            # Keep 项目文件 visible here; the normal auto_pair() completion path
            # owns collapsing it. This avoids hiding inputs while restored pairs /
            # thumbnails are still being inspected or corrected.
            self.refresh_all()
        except Exception as exc:
            QMessageBox.critical(self, "读取已有结果失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

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
            self.project._set_project_files_expanded(False)
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
            if self.state.target_dir: self.state.output_dir = str(Path(self.state.target_dir).parent / "Folirina_Output")
            else: self.state.output_dir = str(Path.home() / "Folirina_Output")
            created = True
        if created and self.state.page_marks:
            self.save_page_marks()
        return self.state.output_dir

    # ---------- Processing ----------
    def run_current_page(self, *, reapply_review_after_process: bool = False):
        pair = self.current_pair()
        if pair is None:
            QMessageBox.information(self, "没有页面", "请先完成页面配对。"); return
        mark = self.page_mark_for_pair(pair)
        # The worker must receive exactly the mode currently visible in the UI.
        # Restored sessions can otherwise retain an older config snapshot until
        # another control emits a signal.
        self.project._sync_config()
        self._start_worker(PipelineWorker(
            config=worker_config_snapshot(self.state.config), pair=pair,
            page_mark=mark.to_dict(), output_dir=self._default_output(),
            reapply_review_after_process=bool(reapply_review_after_process),
        ))

    def _run_book_explicit(self, *, resume: bool):
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

        # Flush the current mode/settings into the book worker snapshot too.
        self.project._sync_config()
        worker_cfg = worker_config_snapshot(self.state.config, resume=resume)
        mode_label = "继续处理整本" if resume else "从头处理整本"
        self.statusBar().showMessage(f"{mode_label} · 正在准备…")
        self._start_worker(PipelineWorker(
            config=worker_cfg, source_dir=self.state.source_dir,
            target_dir=self.state.target_dir, output_dir=self._default_output(),
            pairs_override=list(self.state.pairs), page_marks=dict(self.state.page_marks),
        ))

    def run_book(self):
        """From-scratch whole-book action. Existing page results are not skipped."""
        self._run_book_explicit(resume=False)

    def continue_book(self):
        """Resume whole-book action. Completed page workspaces survive app crashes."""
        self._run_book_explicit(resume=True)

    def _maybe_prepare_models_before_run(self, worker: PipelineWorker) -> bool:
        plan = plan_runtime_requirements(worker.config)
        if plan.errors:
            QMessageBox.warning(self, "运行前检查未通过", "\n".join(plan.errors))
            self.statusBar().showMessage("运行前检查未通过", 5000)
            return True
        if not plan.requirements:
            return False
        # Runtime probes for Paddle/Torch live in isolated Python processes and
        # must never run synchronously on the GUI thread.  Even when model files
        # are already cached, verify their runtime in AutoPrepareModelsWorker.
        labels = " / ".join(req.label for req in plan.requirements)
        self._pending_pipeline_worker = worker
        self.progress.setRange(0,0); self.progress.setValue(0)
        self.statusBar().showMessage("正在验证/自动准备：" + labels)
        self._set_busy(True)
        prep = AutoPrepareModelsWorker(worker.config.model_copy(deep=True))
        self._prepare_worker = prep
        prep.progress.connect(self._prepare_worker_progress)
        prep.done.connect(self._prepare_worker_done)
        prep.failed.connect(self._prepare_worker_failed)
        prep.cancelled.connect(self._prepare_worker_cancelled)
        prep.finished.connect(self._prepare_worker_finished)
        prep.finished.connect(prep.deleteLater)
        prep.start()
        return True

    def _prepare_worker_progress(self, message: str):
        self.statusBar().showMessage(str(message))
        if hasattr(self, "models") and self.stack.currentWidget() is self.models:
            self.models.model_download_status.setText(str(message))

    def _prepare_worker_done(self, payload: object):
        message = str((payload or {}).get("message", "自动准备完成。")) if isinstance(payload, dict) else "自动准备完成。"
        self.statusBar().showMessage(message, 5000)
        if hasattr(self, "models"):
            self.models.model_download_status.setText(message)
            self.models.refresh(force_probe=True)
        worker = self._pending_pipeline_worker
        self._pending_pipeline_worker = None
        if worker is None:
            self._set_busy(False)
            return
        self._launch_pipeline_worker(worker)

    def _prepare_worker_cancelled(self):
        self._pending_pipeline_worker = None
        self.progress.setRange(0,100); self.progress.setValue(0)
        self.statusBar().showMessage("已安全停止模型准备；当前安装/下载步骤已完整结束。", 6000)

    def _prepare_worker_failed(self, message: str):
        self._pending_pipeline_worker = None
        short = str(message).split("\n", 1)[0]
        self.progress.setRange(0,100); self.progress.setValue(0)
        self._set_busy(False)
        if hasattr(self, "models"):
            self.models.model_download_status.setText("自动准备失败：" + short)
            self.models.refresh(force_probe=True)
        self.statusBar().showMessage("自动准备失败", 5000)
        QMessageBox.warning(self, "自动准备模型失败", str(message)[-12000:])

    def _prepare_worker_finished(self):
        self._prepare_worker = None
        if self.worker is None or not self.worker.isRunning():
            self._set_busy(None)

    def _pipeline_cancelled(self):
        self.statusBar().showMessage("已停止，已完成页面已保留", 5000)

    def _launch_pipeline_worker(self, worker: PipelineWorker):
        """Single signal-wiring path after optional runtime/model preparation."""
        self.worker = worker
        self._worker_is_single_page = worker.pair is not None
        self._worker_page_id = page_id_for_pair(worker.pair) if worker.pair is not None else ""
        self._worker_reapply_review = bool(getattr(worker, "reapply_review_after_process", False))
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.statusBar().showMessage("正在处理…")
        self._set_busy(True)
        worker.progress.connect(self._worker_progress)
        worker.done.connect(self._worker_done)
        worker.failed.connect(self._worker_failed)
        worker.cancelled.connect(self._pipeline_cancelled)
        worker.finished.connect(self._worker_finished)
        worker.start()

    def _start_worker(self, worker: PipelineWorker):
        if self._busy_running():
            QMessageBox.information(self, "处理中", "已有任务正在运行。"); return
        if self._maybe_prepare_models_before_run(worker):
            return
        self._launch_pipeline_worker(worker)

    def run_page_action(
        self, label: str, action: Callable[[], object],
        on_done: Callable[[object], None], *, failure_title: str = "页面操作失败",
    ) -> bool:
        """Run a review/editor mutation in a worker thread with global busy gating."""
        if self._busy_running():
            QMessageBox.information(self, "任务进行中", "请等待当前处理/复核任务结束后再执行此操作。")
            return False
        worker = PageActionWorker(label, action)
        self._page_action_worker = worker
        self._set_busy(True)
        self.statusBar().showMessage(f"{label} · 正在处理…")

        def done(payload):
            try:
                on_done(payload)
            except Exception as exc:
                QMessageBox.critical(self, failure_title, str(exc))

        def failed(message: str):
            QMessageBox.critical(self, failure_title, str(message))
            self.statusBar().showMessage(f"{label}失败", 5000)

        def finished():
            self._page_action_worker = None
            self._set_busy(None)

        worker.done.connect(done)
        worker.failed.connect(failed)
        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        return True

    def cancel_worker(self):
        requested = False
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_cancel(); requested = True
        if self._prepare_worker is not None and self._prepare_worker.isRunning():
            self._prepare_worker.request_cancel(); requested = True
            self.statusBar().showMessage("已请求安全停止；当前安装/下载步骤结束后自动退出，不会强杀 venv 写入。", 7000)
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
        state = classify_progress_state(text, cache_hit=bool(cache_hit))
        row = self._find_pair_row_by_target_name(name)
        if 0 <= row < self.project.table.rowCount():
            self.project.table.setItem(row, 6, QTableWidgetItem(state))
            self.project.table.setItem(row, 7, QTableWidgetItem(text or "—"))
        self.state.batch_status[name] = (state, text or "—")
        if getattr(self, "_worker_is_single_page", False) and int(total) == 100:
            self.statusBar().showMessage(f"{int(done)}% · {name} · {text}")
        else:
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
        meta = dict(getattr(project, "meta", {}) or {}) if hasattr(project, "meta") else {}
        cancelled = bool(meta.get("cancelled"))
        self.statusBar().showMessage(completion_message(project), 5000)
        if getattr(self, "_worker_is_single_page", False) and not cancelled:
            wanted = getattr(self, "_worker_page_id", "")
            if wanted:
                for i,pair in enumerate(self.state.pairs):
                    if page_id_for_pair(pair) == wanted:
                        self.set_selected_page(i); break
            if bool(getattr(self, "_worker_reapply_review", False)):
                self.statusBar().showMessage("重新处理完成 · 已自动重新应用本页人工蒙版/复核结果", 6000)
            self.workbench.current_view = "result"
            for b,k in self.workbench.view_buttons: b.setChecked(k == "result")
            self.show_page(2)
        else:
            self.refresh_current_page()

    def _worker_failed(self, message):
        self.progress.setRange(0,100); self.progress.setValue(0)
        self.statusBar().showMessage("处理失败 · 已恢复上次成功结果，可查看日志", 7000)
        pair = self.current_pair()
        log_hint = ""
        if pair is not None and self.state.output_dir:
            try:
                ws = resolve_page_workspace(self.state.output_dir, pair, self.state.projects_by_page.get(page_id_for_pair(pair)), self.state.restored_page_roots.get(page_id_for_pair(pair)))
                log_hint = f"\n\n详细日志：{Path(ws.page_root) / 'run.log'}"

            except Exception:
                logger.debug("failed to resolve page log path after worker failure", exc_info=True)
        QMessageBox.critical(self, "处理失败", str(message) + log_hint)
        self.refresh_current_page()

    def refresh_preview(self):
        if self.stack.currentIndex() == 2: self.workbench.refresh()

    def refresh_current_page(self):
        idx = self.stack.currentIndex()
        if idx == 0: self.project.refresh()
        elif idx == 1: self.models.refresh()
        elif idx == 2: self.workbench.refresh()
        elif idx == 3: self.export.refresh()
        elif idx == 4: self.settings.refresh()

    def refresh_all(self):
        self.refresh_current_page()


def main():
    paths = configure_application_logging(component="gui", level=logging.DEBUG)
    install_exception_hooks()
    logger.info("GUI starting log_dir=%s", paths.directory)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME); app.setOrganizationName("Folirina")
    icon = _application_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    app.setStyleSheet(style_for_theme(_saved_theme_name()))
    if sys.platform == "darwin":
        font=QFont("SF Pro Text"); font.setPointSize(12); app.setFont(font)
    win = StudioWindow()
    if not icon.isNull():
        win.setWindowIcon(icon)
    win.show()
    result = int(app.exec())
    logger.info("GUI stopped exit_code=%s", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
