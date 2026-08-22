from __future__ import annotations

import sys
import logging
import time
import shutil
import zipfile
import copy
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
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
    PathRow, ZoomPreviewView, StableComboBox, StableSpinBox, StableDoubleSpinBox, StableSlider, WorkbenchPageRail,
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
from .app_icon import apply_application_icon
from .gui_processing_policy import (
    compute_busy_state, classify_progress_state, worker_config_snapshot, completion_message, page_completion_message,
)
from .io_utils import load_json, save_json, write_image
from .review_history import review_history_counts
from .region_workspace_state import RegionWorkspaceLinkState, selection_signature
from .selection_overlay import selection_edge_mask, selection_edge_thickness_for_scale
from .review_action_state import review_action_availability
from .font_catalog import discover_fonts
from .review_apply import apply_review_page, clear_ocr_review_blocks, generate_remove_text_preview, apply_target_layer_erase_review, reset_target_layer_erase_review, apply_target_layer_restore_review, reset_target_layer_restore_review, apply_manual_force_transfer_review, reset_manual_force_transfer_review, manual_force_auto_evidence_masks
from .manual_effect import map_target_bbox_to_source, registration_homography, build_manual_effect_masks, build_reveal_seed_mask, estimate_source_background, composite_source_text_delta, clean_manual_target_text
from .modes.reletter import ocr_edit_blocks as _reletter_ocr_edit_blocks
from .modes.reletter import ocr_edit_render as _reletter_ocr_edit_render
from .modes.hybrid import ocr_edit_blocks as _hybrid_ocr_edit_blocks
from .modes.hybrid import ocr_edit_render as _hybrid_ocr_edit_render
from . import ocr_edit_blocks as _shared_ocr_edit_blocks
from . import ocr_edit_render as _shared_ocr_edit_render


def _ocr_mode_modules(mode: str):
    key = str(mode or "").strip().lower()
    if key == "hybrid":
        return _hybrid_ocr_edit_blocks, _hybrid_ocr_edit_render
    if key == "reletter":
        return _reletter_ocr_edit_blocks, _reletter_ocr_edit_render
    # All other reviewer-supported modes are owned by the shared review layer.
    # This now includes both Reveal families under the dedicated review_ocr
    # scope; no Reveal automatic renderer is imported or called here.
    if _shared_ocr_edit_blocks.is_ocr_edit_mode(key):
        return _shared_ocr_edit_blocks, _shared_ocr_edit_render
    return None, None


def _manual_effect_ops_for_mode(mode: str):
    """Return the mode-owned manual open-text engine when available."""
    key = str(mode or "").strip().lower()
    if key == "mask_replace":
        from .modes.mask_replace import open_text_manual as ops
        return ops
    if key == "hybrid":
        from .modes.hybrid import open_text_manual as ops
        return ops
    from . import manual_effect as ops
    return ops


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
from .session_restore import scan_existing_results, expand_restored_session_pairs
from .schema_compat import as_dict, as_dict_rows, as_list, normalize_project, normalize_overrides, normalize_review_applied
from .result_state import commit_reviewed_result, atomic_copy_file
from .manual_review_service import (
    commit_manual_effect, apply_review_history_step, apply_review_overrides_transaction,
    run_manual_review_transaction,
)
from .region_selection import (
    selection_mask_from_spec, selection_bbox_from_spec, bbox_from_mask,
    spec_from_mask, project_selection_spec, snap_selection_mask_to_lineart,
    recognize_closed_region_from_selection,
)
from .region_brush_reveal import (
    stroke_bbox as brush_stroke_bbox, paint_reveal_stroke_inplace,
    compose_reveal_patch, mask_bbox as brush_mask_bbox, mask_counts as brush_mask_counts,
)
from .gui_dialogs import confirm_action
from .page_management import (
    PAGE_TYPE_INFO, MANUAL_PAGE_TYPES, PageMark,
    default_mark, manual_mark, marks_from_json, marks_to_json, page_mark_key,
    page_type_color, page_type_label, resolve_mark,
)

APP_NAME = "Folirina"
VERSION = __version__
QComboBox = StableComboBox
QSpinBox = StableSpinBox
QDoubleSpinBox = StableDoubleSpinBox
QSlider = StableSlider

logger = logging.getLogger(__name__)


def _confirm_destructive_action(parent, title: str, message: str, *, confirm_text: str = "确认", cancel_text: str = "取消") -> bool:
    return confirm_action(
        parent, title, message, confirm_text=confirm_text, cancel_text=cancel_text, destructive=True,
    )


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
    stroke_finished = Signal()

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
        # v2.3.64: page-sized RGBA/QPixmap rebuilds on every mouse move were the
        # main source of brush stutter. Keep a sparse 512px tile overlay instead;
        # a stroke only rebuilds the few tiles it actually touches.
        self._overlay_tile_size = 512
        self._overlay_tiles = {}
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

    def _overlay_tile_keys(self, bbox=None):
        ts=int(self._overlay_tile_size); h,w=self.mask.shape
        if bbox and len(bbox)==4:
            x0,y0,x1,y1=[int(v) for v in bbox]
            x0=max(0,min(w,x0)); x1=max(0,min(w,x1)); y0=max(0,min(h,y0)); y1=max(0,min(h,y1))
            if x1<=x0 or y1<=y0: return []
            tx0=x0//ts; tx1=(x1-1)//ts; ty0=y0//ts; ty1=(y1-1)//ts
            return [(tx,ty) for ty in range(ty0,ty1+1) for tx in range(tx0,tx1+1)]
        # Full refreshes are rare (open editor / change layer / reset). Scan only
        # tile-sized binary crops instead of allocating one page-sized RGBA image.
        keys=set(self._overlay_tiles.keys())
        for y0 in range(0,h,ts):
            for x0 in range(0,w,ts):
                y1=min(h,y0+ts); x1=min(w,x0+ts)
                if self._cv2.countNonZero(self.mask[y0:y1,x0:x1]) or self._cv2.countNonZero(self.reference_mask[y0:y1,x0:x1]):
                    keys.add((x0//ts,y0//ts))
        return sorted(keys)

    def _refresh_overlay(self, bbox=None):
        ts=int(self._overlay_tile_size); h,w=self.mask.shape
        ref_alpha = 132 if self.edit_layer == "reference" else 82
        manual_alpha = 142 if self.edit_layer == "manual" else 98
        for tx,ty in self._overlay_tile_keys(bbox):
            x0=tx*ts; y0=ty*ts; x1=min(w,x0+ts); y1=min(h,y0+ts)
            manual=self.mask[y0:y1,x0:x1]>0
            reference=self.reference_mask[y0:y1,x0:x1]>0
            key=(tx,ty); item=self._overlay_tiles.get(key)
            if not self._np.any(manual) and not self._np.any(reference):
                if item is not None:
                    self._scene.removeItem(item); self._overlay_tiles.pop(key,None)
                continue
            th,tw=manual.shape
            rgba=self._np.zeros((th,tw,4),dtype=self._np.uint8)
            ref_only=reference & (~manual); overlap=reference & manual
            rgba[ref_only,0]=70; rgba[ref_only,1]=165; rgba[ref_only,2]=235; rgba[ref_only,3]=ref_alpha
            rgba[manual,0]=230; rgba[manual,1]=70; rgba[manual,2]=85; rgba[manual,3]=manual_alpha
            rgba[overlap,0]=245; rgba[overlap,1]=165; rgba[overlap,2]=45; rgba[overlap,3]=150
            q=QImage(rgba.data,tw,th,int(rgba.strides[0]),QImage.Format.Format_RGBA8888).copy()
            pix=QPixmap.fromImage(q)
            if item is None:
                item=self._scene.addPixmap(pix); item.setZValue(2); self._overlay_tiles[key]=item
            else:
                item.setPixmap(pix)
            item.setPos(x0,y0)

    def _stroke_dirty_bbox(self, a, b=None):
        if b is None: b=a
        r=max(3,int(self.brush_size)+3)
        x0=min(int(a[0]),int(b[0]))-r; y0=min(int(a[1]),int(b[1]))-r
        x1=max(int(a[0]),int(b[0]))+r+1; y1=max(int(a[1]),int(b[1]))+r+1
        return [x0,y0,x1,y1]

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
        active = self.active_mask(); previous=self._last
        self._cv2.line(active, previous, now, int(val), max(1, int(self.brush_size)), lineType=self._cv2.LINE_AA)
        self._last = now; self._refresh_overlay(self._stroke_dirty_bbox(previous,now)); self.mask_changed.emit(); event.accept()

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False; self._pan_last = None
            self.viewport().unsetCursor(); event.accept(); return
        if self._painting:
            self._painting = False; self._last = None; self.stroke_finished.emit(); event.accept(); return
        super().mouseReleaseEvent(event)

    def _paint_to(self, pt):
        x, y = pt
        if 0 <= x < self.mask.shape[1] and 0 <= y < self.mask.shape[0]:
            active = self.active_mask()
            self._cv2.circle(active, (x, y), max(1, int(self.brush_size // 2)), 0 if self._erase else 255, -1, lineType=self._cv2.LINE_AA)
            self._refresh_overlay(self._stroke_dirty_bbox((x,y))); self.mask_changed.emit()


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
        self._preview_timer = QTimer(self); self._preview_timer.setSingleShot(True); self._preview_timer.setInterval(110)
        self._preview_timer.timeout.connect(self._refresh_live_preview)
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
        action_bar=QFrame(); action_bar.setObjectName("editorActionBar"); action_grid=QGridLayout(action_bar); action_grid.setContentsMargins(0,0,0,0); action_grid.setHorizontalSpacing(6); action_grid.setVerticalSpacing(5)
        self.focus_button=QPushButton("聚焦当前层"); self.fit_button = QPushButton("查看整页"); self.clear_button = QPushButton("清空当前层"); self.import_reference_button = QPushButton(f"复制 {self._reference_label} → 人工层"); self.import_reference_button.setEnabled(has_reference); self.import_reference_button.setVisible(has_reference); self.reset_reference_button = QPushButton("恢复自动原始") if has_reference else None; self.save_button = QPushButton(save_label); self.save_button.setObjectName("primary"); self.cancel_button = QPushButton("取消")
        self.import_reference_button.setToolTip("将当前蓝色 OCR/自动蒙版复制到红色人工层；蓝色层本身现在也可以直接涂抹或擦除。")
        for col,button in enumerate((self.focus_button,self.fit_button,self.clear_button,self.import_reference_button)):
            button.setMinimumWidth(0); action_grid.addWidget(button,0,col)
        next_col=4
        if self.reset_reference_button is not None:
            self.reset_reference_button.setMinimumWidth(0); action_grid.addWidget(self.reset_reference_button,0,next_col); next_col+=1
        action_grid.setColumnStretch(next_col,1); action_grid.addWidget(self.save_button,1,max(0,next_col-1)); action_grid.addWidget(self.cancel_button,1,next_col)
        root.addWidget(action_bar,0)
        self.slider.valueChanged.connect(self._brush); self.fit_button.clicked.connect(self.view.fit_to_window); self.focus_button.clicked.connect(self.view.fit_to_active_mask)
        self.paint_add.clicked.connect(lambda: self._set_mode("add")); self.paint_erase.clicked.connect(lambda: self._set_mode("erase"))
        self.clear_button.clicked.connect(self._clear); self.import_reference_button.clicked.connect(self._import_reference); self.save_button.clicked.connect(self.accept); self.cancel_button.clicked.connect(self.reject)
        if self.reset_reference_button is not None: self.reset_reference_button.clicked.connect(self._reset_reference)
        if self._preview_fn is not None:
            # Brush overlay updates immediately, but expensive full-image preview
            # work is debounced and forced once at stroke end.
            self.view.mask_changed.connect(self._schedule_live_preview)
            self.view.stroke_finished.connect(self._refresh_live_preview)
            QTimer.singleShot(0,self._refresh_live_preview)
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

    def _schedule_live_preview(self):
        if self._preview_fn is not None:
            self._preview_timer.start()

    def _refresh_live_preview(self):
        if self._preview_fn is None: return
        if self._preview_timer.isActive(): self._preview_timer.stop()
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
        action_bar=QFrame(); action_bar.setObjectName("editorActionBar"); action_grid=QGridLayout(action_bar); action_grid.setContentsMargins(0,0,0,0); action_grid.setHorizontalSpacing(6); action_grid.setVerticalSpacing(5)
        self.focus_button=QPushButton("聚焦选区"); self.fit_button=QPushButton("查看整页"); self.auto_button=QPushButton("恢复自动建议"); self.clear_button=QPushButton("全部恢复日文"); self.save_button=QPushButton("保存擦除显字"); self.save_button.setObjectName("primary"); self.cancel_button=QPushButton("取消")
        for col,button in enumerate((self.focus_button,self.fit_button,self.auto_button,self.clear_button)):
            button.setMinimumWidth(0); action_grid.addWidget(button,0,col)
        action_grid.setColumnStretch(4,1); action_grid.addWidget(self.save_button,1,3); action_grid.addWidget(self.cancel_button,1,4); root.addWidget(action_bar,0)
        self._auto_seed=(np.asarray(initial_mask)>0).astype(np.uint8)*255
        self.slider.valueChanged.connect(self._brush); self.fit_button.clicked.connect(self.view.fit_to_window); self.focus_button.clicked.connect(self._focus_selection)
        self.auto_button.clicked.connect(self._restore_auto); self.clear_button.clicked.connect(self._clear); self.save_button.clicked.connect(self.accept); self.cancel_button.clicked.connect(self.reject)
        self._preview_timer=QTimer(self); self._preview_timer.setSingleShot(True); self._preview_timer.setInterval(90); self._preview_timer.timeout.connect(self._refresh_preview)
        self.view.mask_changed.connect(lambda: self._preview_timer.start())
        self.view.stroke_finished.connect(self._refresh_preview)
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
        # Keep the expensive full-page preview out of mouse-move frequency. The
        # tile mask overlay remains responsive while this image updates at idle /
        # stroke end.
        self.view.set_preview_image(out)

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
    """Zoomable TARGET-space selection view used by every regional review tool.

    Rectangle remains the compatibility default.  Ellipse, freehand and smart
    closed-region selection share the same serializable ``selection_spec`` so
    renderers never depend on Qt scene geometry.
    """
    selection_changed = Signal(object)          # compatibility: emits bbox
    selection_spec_changed = Signal(object)
    brush_stroke_started = Signal(object)
    brush_stroke_segment = Signal(object)
    brush_stroke_finished = Signal()

    def __init__(self, image_path: str | Path, *, editable: bool, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self); self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._editable = bool(editable); self._image_path=str(image_path)
        reader = QImageReader(str(image_path)); reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            raise ValueError(f"无法读取图片：{Path(image_path).name}")
        self._pix = QPixmap.fromImage(image); self._scene.addPixmap(self._pix)
        self._overlay_item=self._scene.addPixmap(QPixmap()); self._overlay_item.setZValue(4)
        self._scene.setSceneRect(0, 0, self._pix.width(), self._pix.height())
        self._start=None; self._points=[]; self._selection_spec={}; self._selection_mask=np.zeros((self._pix.height(),self._pix.width()),np.uint8); self._selection_bbox=[]
        self._selection_mode="rect"; self._snap_enabled=False; self._snap_distance=10; self._snap_diag={}; self._snap_cv_cache=None
        self._panning=False; self._pan_last=None; self._auto_fit=True; self._fit_pending=False
        self._interaction_mode="selection"; self._brush_size=40; self._brush_last=None; self._brush_right=False
        self.setDragMode(QGraphicsView.DragMode.NoDrag); self.setInteractive(True); self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); self.fit_to_window()

    def _apply_fit(self):
        self._fit_pending=False
        if not self._auto_fit or self.viewport().width()<8 or self.viewport().height()<8: return
        self.resetTransform(); self.fitInView(_fit_scene_rect(self._scene),Qt.AspectRatioMode.KeepAspectRatio)

    def fit_to_window(self): self._auto_fit=True; self._apply_fit()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self._auto_fit and not self._fit_pending:
            self._fit_pending=True; QTimer.singleShot(0,self._apply_fit)

    def wheelEvent(self,event):
        factor=1.15 if event.angleDelta().y()>0 else 1.0/1.15; target=float(self.transform().m11())*factor
        if 0.05<=target<=12.0: self._auto_fit=False; self.scale(factor,factor)
        event.accept()

    def mouseDoubleClickEvent(self,event): self.fit_to_window(); event.accept()

    def _clamp(self,x:float,y:float)->tuple[int,int]:
        return max(0,min(self._pix.width()-1,int(round(x)))),max(0,min(self._pix.height()-1,int(round(y))))

    def set_interaction_mode(self,mode:str):
        key=str(mode or "selection").strip().lower()
        self._interaction_mode="brush" if key=="brush" else "selection"
        self._brush_last=None; self._brush_right=False
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if self._editable else Qt.CursorShape.ArrowCursor)

    def interaction_mode(self)->str: return str(self._interaction_mode)
    def set_brush_size(self,value:int): self._brush_size=max(2,min(320,int(value)))

    def set_selection_mode(self,mode:str):
        mode=str(mode or "rect").strip().lower(); self._selection_mode=mode if mode in {"rect","ellipse","freehand","smart"} else "rect"
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if self._editable else Qt.CursorShape.ArrowCursor)

    def set_snap_enabled(self,enabled:bool): self._snap_enabled=bool(enabled)
    def set_snap_distance(self,value:int): self._snap_distance=max(0,min(32,int(value)))
    def set_snap_image_path(self,path:str|Path):
        path=str(path)
        if path != self._image_path: self._snap_cv_cache=None
        self._image_path=path

    def _snap_image(self):
        if self._snap_cv_cache is None:
            image=cv2.imread(self._image_path,cv2.IMREAD_COLOR)
            if image is not None and image.shape[:2]==self._selection_mask.shape:
                self._snap_cv_cache=image
        return self._snap_cv_cache

    def snap_current_selection(self):
        """Nudge an already-good selection boundary toward nearby line art."""
        if not self._editable or cv2.countNonZero(self._selection_mask)<=0: return False
        image=self._snap_image()
        if image is None: return False
        snapped,diag=snap_selection_mask_to_lineart(image,self._selection_mask,max_distance=self._snap_distance)
        self._snap_diag=dict(diag or {})
        if not bool(diag.get("used_fallback")):
            spec=spec_from_mask(snapped,kind="smart",snapped=True,diagnostics=diag)
        else:
            spec=dict(self._selection_spec or {}); spec["diagnostics"]=dict(diag or {}); spec["snapped"]=False
        self._selection_mask[:]=np.where(snapped>0,255,0).astype(np.uint8); self._selection_bbox=bbox_from_mask(self._selection_mask); spec["bbox"]=list(self._selection_bbox); self._selection_spec=spec
        self._refresh_overlay(); self._emit_selection(); return True

    def recognize_closed_region(self):
        """Turn a rough box/lasso into a topologically enclosed manga region."""
        if not self._editable or cv2.countNonZero(self._selection_mask)<=0: return False
        image=self._snap_image()
        if image is None: return False
        closed,diag=recognize_closed_region_from_selection(
            image,self._selection_mask,
            gap_close=max(3,min(8,int(round(self._snap_distance*0.5)))),
            max_expand_px=max(0,self._snap_distance),
        )
        self._snap_diag=dict(diag or {})
        success=not bool(diag.get("used_fallback"))
        if success:
            self._selection_mask[:]=np.where(closed>0,255,0).astype(np.uint8)
            spec=spec_from_mask(self._selection_mask,kind="smart",snapped=True,diagnostics=diag)
        else:
            spec=dict(self._selection_spec or {}); spec["diagnostics"]=dict(diag or {}); spec["snapped"]=False
        self._selection_bbox=bbox_from_mask(self._selection_mask); spec["bbox"]=list(self._selection_bbox); self._selection_spec=spec
        self._refresh_overlay(); self._emit_selection(); return success

    def replace_display_image(self,path:str|Path):
        reader=QImageReader(str(path)); reader.setAutoTransform(True); image=reader.read()
        if image.isNull(): return False
        pix=QPixmap.fromImage(image)
        if pix.width()!=self._pix.width() or pix.height()!=self._pix.height(): return False
        self._pix=pix
        # The first scene item is the base image; keep selection overlay intact.
        items=[item for item in self._scene.items() if item is not self._overlay_item]
        if items:
            base=min(items,key=lambda item:item.zValue()); base.setPixmap(pix)
        return True

    def _refresh_overlay(self):
        # Never allocate a page-sized RGBA overlay while the mouse is moving.
        # A 4096x5824 page used to allocate ~95 MB per drag event here, which
        # made the selection system visibly stutter and could pressure macOS
        # memory.  Render only the tight selection ROI and position that pixmap
        # back in scene coordinates.
        box=list(self._selection_bbox or [])
        if len(box)!=4:
            self._overlay_item.setPixmap(QPixmap()); self._overlay_item.setPos(0,0); return
        x0,y0,x1,y1=[int(v) for v in box]
        crop=self._selection_mask[y0:y1,x0:x1]
        if crop.size==0 or cv2.countNonZero(crop)<=0:
            self._overlay_item.setPixmap(QPixmap()); self._overlay_item.setPos(0,0); return
        inside=crop>0; ch,cw=crop.shape; rgba=np.zeros((ch,cw,4),np.uint8)
        # v2.3.70: tight-ROI rendering must still expose the selection border.
        # A rectangular crop can be all-255, so MORPH_GRADIENT on the cropped
        # array returns no edge. selection_edge_mask uses an explicit zero
        # exterior and therefore remains visible for rect/ellipse/freehand/smart.
        edge_thickness=selection_edge_thickness_for_scale(float(self.transform().m11()))
        edge=selection_edge_mask(crop,thickness=edge_thickness)>0
        accent=QColor(ACCENT); r,g,b=accent.red(),accent.green(),accent.blue()
        rgba[inside,0]=r; rgba[inside,1]=g; rgba[inside,2]=b; rgba[inside,3]=34
        rgba[edge,0]=r; rgba[edge,1]=g; rgba[edge,2]=b; rgba[edge,3]=255
        q=QImage(rgba.data,cw,ch,int(rgba.strides[0]),QImage.Format.Format_RGBA8888).copy()
        self._overlay_item.setPixmap(QPixmap.fromImage(q)); self._overlay_item.setPos(x0,y0)

    def _emit_selection(self):
        # Drag paths maintain the bbox incrementally; do not rescan a 20-30 MP
        # mask merely to emit a Qt signal on mouse release.
        box=list(self._selection_bbox); self.selection_changed.emit(box); self.selection_spec_changed.emit(dict(self._selection_spec))

    def clear_selection(self,*,emit:bool=False):
        if len(self._selection_bbox)==4:
            x0,y0,x1,y1=self._selection_bbox; self._selection_mask[y0:y1,x0:x1]=0
        else:
            self._selection_mask[:]=0
        self._selection_bbox=[]; self._selection_spec={}; self._snap_diag={}; self._refresh_overlay()
        if emit: self._emit_selection()

    def set_selection_spec(self,spec:dict|None,*,emit:bool=False):
        data=dict(spec or {}); old_box=list(self._selection_bbox)
        mask=selection_mask_from_spec(data,self._selection_mask.shape,out=self._selection_mask,clear_bbox=old_box)
        box=selection_bbox_from_spec(data,self._selection_mask.shape)
        if not box or cv2.countNonZero(mask[box[1]:box[3],box[0]:box[2]])<=0:
            self.clear_selection(emit=emit); return
        data.setdefault("schema","folirina.region_selection.v1"); data.setdefault("kind","rect"); data["bbox"]=box
        self._selection_bbox=box; self._selection_spec=data; self._refresh_overlay()
        if emit: self._emit_selection()

    def set_box(self,bbox:list[int]|tuple[int,int,int,int]|None,*,emit:bool=False):
        if not bbox or len(bbox)!=4: self.clear_selection(emit=emit); return
        self.set_selection_spec({"schema":"folirina.region_selection.v1","kind":"rect","bbox":[int(v) for v in bbox],"points":[],"snapped":False},emit=emit)

    def box(self)->list[int]: return list(self._selection_bbox)
    def image_shape(self)->tuple[int,int]: return tuple(self._selection_mask.shape)
    def selection_spec(self)->dict: return dict(self._selection_spec)
    def selection_mask(self): return self._selection_mask.copy()
    def snap_diagnostics(self)->dict: return dict(self._snap_diag)

    def _apply_raw_spec(self,spec:dict,*,final:bool=False):
        old_box=list(self._selection_bbox)
        mask=selection_mask_from_spec(spec,self._selection_mask.shape,out=self._selection_mask,clear_bbox=old_box)
        box=selection_bbox_from_spec(spec,self._selection_mask.shape); kind=str(spec.get("kind") or "rect")
        nonempty=bool(len(box)==4 and cv2.countNonZero(mask[box[1]:box[3],box[0]:box[2]])>0)
        if final and nonempty and (self._snap_enabled or kind=="smart"):
            image=self._snap_image()
            if image is not None and image.shape[:2]==mask.shape:
                if kind=="smart":
                    snapped,diag=recognize_closed_region_from_selection(
                        image,mask,
                        gap_close=max(3,min(8,int(round(self._snap_distance*0.5)))),
                        max_expand_px=max(0,self._snap_distance),
                    )
                else:
                    snapped,diag=snap_selection_mask_to_lineart(image,mask,max_distance=self._snap_distance)
                self._snap_diag=dict(diag)
                self._selection_mask[:]=np.where(snapped>0,255,0).astype(np.uint8); mask=self._selection_mask
                if not bool(diag.get("used_fallback")):
                    spec=spec_from_mask(mask,kind="smart",snapped=True,diagnostics=diag)
                else:
                    spec=dict(spec); spec["diagnostics"]=dict(diag); spec["snapped"]=False
                box=bbox_from_mask(mask)
        self._selection_bbox=list(box or []); spec=dict(spec); spec["bbox"]=list(self._selection_bbox); self._selection_spec=spec; self._refresh_overlay()
        if final: self._emit_selection()

    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.MiddleButton or (self._interaction_mode=="selection" and event.button()==Qt.MouseButton.RightButton):
            self._panning=True; self._pan_last=event.position().toPoint(); self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor); event.accept(); return
        if not self._editable:
            super().mousePressEvent(event); return
        if self._interaction_mode=="brush":
            if event.button() not in (Qt.MouseButton.LeftButton,Qt.MouseButton.RightButton):
                super().mousePressEvent(event); return
            p=self.mapToScene(event.position().toPoint()); now=self._clamp(p.x(),p.y())
            self._brush_last=now; self._brush_right=event.button()==Qt.MouseButton.RightButton
            payload={"point":[now[0],now[1]],"right":bool(self._brush_right),"brush_size":int(self._brush_size)}
            self.brush_stroke_started.emit(dict(payload))
            self.brush_stroke_segment.emit({"from":[now[0],now[1]],"to":[now[0],now[1]],"right":bool(self._brush_right),"brush_size":int(self._brush_size)})
            event.accept(); return
        if event.button()!=Qt.MouseButton.LeftButton:
            super().mousePressEvent(event); return
        p=self.mapToScene(event.position().toPoint()); self._start=self._clamp(p.x(),p.y()); self._points=[list(self._start)]
        if self._selection_mode=="freehand":
            self._apply_raw_spec({"schema":"folirina.region_selection.v1","kind":"freehand","points":self._points,"bbox":[]})
        else:
            kind="smart" if self._selection_mode=="smart" else self._selection_mode
            self._apply_raw_spec({"schema":"folirina.region_selection.v1","kind":kind,"bbox":[self._start[0],self._start[1],self._start[0]+2,self._start[1]+2],"points":[]})
        event.accept()

    def mouseMoveEvent(self,event):
        if self._panning and self._pan_last is not None:
            now=event.position().toPoint(); delta=now-self._pan_last; self._pan_last=now; self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-delta.x()); self.verticalScrollBar().setValue(self.verticalScrollBar().value()-delta.y()); event.accept(); return
        if not self._editable:
            super().mouseMoveEvent(event); return
        if self._interaction_mode=="brush":
            if self._brush_last is None:
                super().mouseMoveEvent(event); return
            p=self.mapToScene(event.position().toPoint()); now=self._clamp(p.x(),p.y()); last=self._brush_last; self._brush_last=now
            self.brush_stroke_segment.emit({"from":[last[0],last[1]],"to":[now[0],now[1]],"right":bool(self._brush_right),"brush_size":int(self._brush_size)})
            event.accept(); return
        if self._start is None:
            super().mouseMoveEvent(event); return
        p=self.mapToScene(event.position().toPoint()); end=self._clamp(p.x(),p.y())
        if self._selection_mode=="freehand":
            if not self._points or abs(end[0]-self._points[-1][0])+abs(end[1]-self._points[-1][1])>=2: self._points.append([end[0],end[1]])
            self._apply_raw_spec({"schema":"folirina.region_selection.v1","kind":"freehand","points":self._points,"bbox":[]})
        else:
            kind="smart" if self._selection_mode=="smart" else self._selection_mode
            self._apply_raw_spec({"schema":"folirina.region_selection.v1","kind":kind,"bbox":[self._start[0],self._start[1],end[0],end[1]],"points":[]})
        event.accept()

    def mouseReleaseEvent(self,event):
        if self._panning and event.button() in (Qt.MouseButton.MiddleButton,Qt.MouseButton.RightButton):
            self._panning=False; self._pan_last=None; self.viewport().setCursor(Qt.CursorShape.CrossCursor if self._editable else Qt.CursorShape.ArrowCursor); event.accept(); return
        if self._editable and self._interaction_mode=="brush" and self._brush_last is not None and event.button() in (Qt.MouseButton.LeftButton,Qt.MouseButton.RightButton):
            self._brush_last=None; self._brush_right=False; self.brush_stroke_finished.emit(); event.accept(); return
        if self._editable and self._start is not None and event.button()==Qt.MouseButton.LeftButton:
            p=self.mapToScene(event.position().toPoint()); end=self._clamp(p.x(),p.y()); start=self._start; self._start=None
            if self._selection_mode=="freehand":
                if len(self._points)>=3: self._points.append(self._points[0])
                spec={"schema":"folirina.region_selection.v1","kind":"freehand","points":self._points,"bbox":[]}
            else:
                kind="smart" if self._selection_mode=="smart" else self._selection_mode; spec={"schema":"folirina.region_selection.v1","kind":kind,"bbox":[start[0],start[1],end[0],end[1]],"points":[]}
            self._apply_raw_spec(spec,final=True); event.accept(); return
        super().mouseReleaseEvent(event)



class OCRBlockEditorDialog(QDialog):
    """Manual ROI OCR + per-block typography editor.

    This dialog is available to the precise-transfer review families and OCR
    reletter.  It stores state under ``ocr_edit/mask_ocr`` or
    ``ocr_edit/ocr_reletter`` and remains a page-local review overlay.
    """

    def __init__(self, page_dir: str | Path, source_path: str | Path, target_path: str | Path,
                 project: dict[str, Any], config: PipelineConfig, mode: str, parent=None):
        super().__init__(parent)
        if not is_ocr_edit_mode(mode):
            raise ValueError("人工 OCR 文本块在当前整页模式不可用。")
        self.page_dir=Path(page_dir); self.source_path=Path(source_path); self.target_path=Path(target_path)
        self.project=dict(project or {}); self.config=config.model_copy(deep=True); self.mode=str(mode)
        self._blocks=load_ocr_blocks(self.page_dir,self.mode); self._current_id=""; self._ocr_worker=None
        scope=ocr_edit_scope(self.mode)
        mode_label={
            "direct_patch":"Direct · 人工 OCR",
            "mask_replace":"精准蒙版 · 人工 OCR",
            "hybrid":"精准蒙版+OCR",
            "reletter":"OCR重排",
        }.get(self.mode,"人工 OCR")
        self.setWindowTitle("人工 OCR 文本块 · " + mode_label)
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
        actions=QHBoxLayout(); actions.addStretch(1); self.cancel_btn=QPushButton("取消"); self.save_btn=QPushButton("保存并应用"); self.save_btn.setObjectName("primary")
        actions.addWidget(self.cancel_btn); actions.addWidget(self.save_btn); root.addLayout(actions)
        self.cancel_btn.clicked.connect(self.reject); self.save_btn.clicked.connect(self._save); self.ocr_btn.clicked.connect(self._rerun_ocr); self.new_btn.clicked.connect(self._new)
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
            self.ocr_status.setText("请先在右侧 TARGET 图上拖出 OCR 区域。")
            return
        worker=getattr(self,"_ocr_worker",None)
        if worker is not None and worker.isRunning():
            self.ocr_status.setText("OCR 正在处理中，请稍候…")
            return
        seed=self._selected_row() or self._current_style(); seed.update(self._current_style())
        bbox=list(bbox); project=dict(self.project); source_path=Path(self.source_path); target_path=Path(self.target_path); cfg=self.config.model_copy(deep=True)
        self.ocr_btn.setEnabled(False); self.save_btn.setEnabled(False); self.cancel_btn.setEnabled(False); self.new_btn.setEnabled(False); self.delete_btn.setEnabled(False)
        self.ocr_btn.setText("OCR 处理中…"); self.ocr_status.setText("正在识别 SOURCE 中文，并定位 TARGET 日文…")
        worker=PageActionWorker(
            "人工 ROI OCR",
            lambda: recognize_manual_ocr_block(project,source_path,target_path,bbox,cfg,existing=seed),
        )
        self._ocr_worker=worker

        def done(payload):
            row=dict(payload or {})
            self._current_id=str(row.get("id") or "")
            self.text.setPlainText(str(row.get("render_text") or row.get("ocr_text") or ""))
            self.source_view.set_box(list(row.get("source_bbox") or []))
            self._pending_ocr=row
            serr=str(row.get("source_ocr_error") or "").strip(); terr=str(row.get("target_ocr_error") or "").strip()
            text=str(row.get("render_text") or row.get("ocr_text") or "").strip()
            if serr:
                backend=str(row.get("source_backend") or "OCR")
                self.ocr_status.setText(f"{backend} SOURCE OCR 失败：{serr[:180]} · 可改 OCR 后端后重试，或手动输入中文。")
            elif not text:
                self.ocr_status.setText("OCR 已执行，但 SOURCE ROI 没识别到文字。请调整选框后重试。")
            else:
                suffix="；TARGET 定位失败，将使用保守局部清字" if terr else ""
                self.ocr_status.setText(f"重新 OCR 完成 · 置信度 {float(row.get('confidence') or 0):.2f}{suffix}")

        def failed(message):
            detail=str(message or "人工 OCR 未返回错误信息").strip()
            self.ocr_status.setText("人工 OCR 执行失败："+detail.splitlines()[0][:220])

        def finished():
            self._ocr_worker=None
            self.ocr_btn.setEnabled(True); self.save_btn.setEnabled(True); self.cancel_btn.setEnabled(True); self.new_btn.setEnabled(True); self.delete_btn.setEnabled(True); self.ocr_btn.setText("重新 OCR 当前选框")

        worker.done.connect(done); worker.failed.connect(failed); worker.finished.connect(finished); worker.finished.connect(worker.deleteLater)
        worker.start()

    def closeEvent(self,event):  # noqa: N802 - Qt API
        worker=getattr(self,"_ocr_worker",None)
        if worker is not None and worker.isRunning():
            self.ocr_status.setText("OCR 正在处理中，完成后即可关闭窗口。")
            event.ignore(); return
        super().closeEvent(event)

    def _save(self):
        bbox=self.target_view.box(); text=self.text.toPlainText().strip()
        if len(bbox)!=4:
            self.ocr_status.setText("缺少选框：请先在 TARGET 上框选文本区域。")
            return
        if not text:
            self.ocr_status.setText("OCR 结果为空：请重新 OCR，或手动输入中文后再保存。")
            return
        base=dict(getattr(self,"_pending_ocr",None) or self._selected_row() or {})
        base.update(self._current_style()); base["target_bbox"]=list(bbox); base["source_bbox"]=map_target_bbox_to_source(self.project,list(bbox)); base["render_text"]=text; base.setdefault("ocr_text",text); base["review_kind"]="manual_ocr"; base["box_locked"]=True; base["manual_override"]=True
        saved=upsert_ocr_block(self.page_dir,self.mode,base); self._current_id=str(saved.get("id") or ""); self.accept()

    def _new(self):
        self._current_id=""; self.block_combo.setCurrentIndex(0); self.target_view.set_box([]); self.source_view.set_box([]); self.text.clear(); self.ocr_status.setText("新建：请在 TARGET 上拖框")
        if hasattr(self,"_pending_ocr"): delattr(self,"_pending_ocr")

    def _delete(self):
        bid=str(self.block_combo.currentData() or "")
        if not bid: return
        if not _confirm_destructive_action(self, "删除 OCR 文本块", "删除当前人工 OCR 文本块？", confirm_text="删除"): return
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


class RegionCompositeDialog(QDialog):
    """Selection-first regional review workbench.

    Three visual authorities are always visible: original Japanese TARGET,
    registered old Chinese SOURCE and the current reviewed result.  The current
    result is the only editable canvas; selections are mirrored to the two
    references so the user can verify every local operation before committing.
    """

    _TOOLS = [
        ("直接贴图", "region_direct_patch", "把已配准 SOURCE 像素直接写入选区；适合两版画面完全一致的局部。"),
        ("精准蒙版", "region_precise_mask", "只清 TARGET 日文字形并迁移 SOURCE 中文原字形；现在只计算选区周边 ROI，不再为一个小框重跑整页。"),
        ("挖孔揭示", "region_hole_reveal", "把选区内部揭示为已配准 SOURCE；可向内缩保护气泡/文本框边线。"),
        ("透明文字", "region_transparent", "只迁移 SOURCE 中文笔画差量，尽量保留 TARGET 彩图、肤色和背景纹理。"),
        ("OCR / 排字", "region_ocr", "只 OCR 当前选区；可修改中文、方向、字号、断句和排版后再应用。"),
    ]

    def __init__(self, page_dir: str|Path, source_path: str|Path, target_path: str|Path,
                 display_path: str|Path, project: dict[str,Any], config: PipelineConfig,
                 parent=None, *, commit_handler=None, commit_finalize_handler=None, trace_handler=None):
        super().__init__(parent)
        self.page_dir=Path(page_dir); self.source_path=Path(source_path); self.target_path=Path(target_path)
        self.display_path=Path(display_path); self.project=dict(project or {}); self.config=config.model_copy(deep=True)
        self._commit_handler=commit_handler; self._commit_finalize_handler=commit_finalize_handler; self._trace_handler=trace_handler; self._applied=0; self._ocr_payload={}
        # One explicit linkage state coordinates the two visual tool groups.
        # Selection is orthogonal and persists across action-mode switches; OCR
        # authority is tied to the exact selection that produced it.
        self._link_state=RegionWorkspaceLinkState()
        self._ocr_programmatic_change=False
        self._ocr_worker=None
        self._ocr_busy=False
        self._ocr_request_signature=""
        self._commit_worker=None
        self.setWindowTitle("区域复合工具 · 选区系统 / Direct / 蒙版 / 挖孔 / 透明 / OCR")
        # v2.3.43: the old bottom-toolbar layout could advertise a safe dialog
        # size but still grow beyond the real macOS content area because child
        # size hints were wider/taller than the requested window.  Keep the
        # visual workbench in the centre and move every control into a bounded
        # side inspector with its own vertical scroll fallback.  The commit
        # buttons live outside that scroll area, so they can never disappear.
        _configure_responsive_dialog(self,(1480,920),(760,520))
        root=QVBoxLayout(self); root.setContentsMargins(10,10,10,10); root.setSpacing(7)

        intro=QLabel("在中间“当前结果”上建立选区；左侧固定显示高清日文 TARGET 与旧版中文 SOURCE。右侧选区和处理工具始终可见，同一区域可以连续叠加多个工具。")
        intro.setObjectName("hint"); intro.setWordWrap(True)
        intro.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        root.addWidget(intro)

        # A QSplitter can preserve stale pane sizes across the macOS native
        # maximise transition.  v2.3.46 therefore managed to give the centre
        # canvas the whole screen while leaving the inspector just beyond the
        # visible right edge.  Use a grid with two bounded side columns instead:
        # refs + elastic canvas + inspector.  The canvas is the only column that
        # may absorb/give up horizontal space.
        body=QWidget(); body.setObjectName("regionWorkspaceBody")
        body_grid=QGridLayout(body); body_grid.setContentsMargins(0,0,0,0); body_grid.setHorizontalSpacing(6); body_grid.setVerticalSpacing(0)

        # Reference column: true Japanese authority above, registered Chinese
        # source below.  It is deliberately narrow so the editable result keeps
        # most of the screen on 1480px-class Mac displays.
        refs=QSplitter(Qt.Orientation.Vertical); refs.setChildrenCollapsible(False); refs.setHandleWidth(4)
        refs.setMinimumWidth(180); refs.setMaximumWidth(300); refs.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Expanding)
        jp_card=QFrame(); jp_card.setObjectName("card"); jpl=QVBoxLayout(jp_card); jpl.setContentsMargins(6,6,6,6); jpl.setSpacing(4)
        jp_title=QLabel("高清日文原图 · TARGET 参考（始终不变）"); jp_title.setObjectName("sectionTitle"); jp_title.setWordWrap(True); jpl.addWidget(jp_title)
        self.jp_view=RegionSelectView(self.target_path,editable=False,parent=self); self.jp_view.setMinimumHeight(120); jpl.addWidget(self.jp_view,1)
        cn_card=QFrame(); cn_card.setObjectName("card"); cnl=QVBoxLayout(cn_card); cnl.setContentsMargins(6,6,6,6); cnl.setSpacing(4)
        cn_title=QLabel("旧版中文 · SOURCE 配准参考"); cn_title.setObjectName("sectionTitle"); cn_title.setWordWrap(True); cnl.addWidget(cn_title)
        self.source_view=RegionSelectView(self.source_path,editable=False,parent=self); self.source_view.setMinimumHeight(120); cnl.addWidget(self.source_view,1)
        refs.addWidget(jp_card); refs.addWidget(cn_card); refs.setSizes([320,320])

        current_card=QFrame(); current_card.setObjectName("card"); current_card.setMinimumWidth(320); current_card.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        cr=QVBoxLayout(current_card); cr.setContentsMargins(6,6,6,6); cr.setSpacing(4)
        current_title=QLabel("当前结果 · 在这里建立选区并连续处理"); current_title.setObjectName("sectionTitle"); current_title.setWordWrap(True); cr.addWidget(current_title)
        self.target_view=RegionSelectView(self.display_path if self.display_path.exists() else self.target_path,editable=True,parent=self)
        self.target_view.setMinimumHeight(300); self.target_view.set_snap_image_path(self.target_path); cr.addWidget(self.target_view,1)
        # Non-destructive brush reveal state.  The masks are page-sized uint8,
        # while live preview pixels are kept as sparse 256px tiles so a 4K page
        # never needs a full QPixmap rebuild for every mouse move.
        self._brush_base=cv2.imread(str(self.display_path if self.display_path.exists() else self.target_path),cv2.IMREAD_COLOR)
        self._brush_source=cv2.imread(str(self.source_path),cv2.IMREAD_COLOR)
        if self._brush_base is None or self._brush_base.shape[:2] != self.target_view.image_shape():
            raise ValueError("区域涂抹无法读取当前结果")
        if self._brush_source is None:
            raise ValueError("区域涂抹无法读取旧版中文 SOURCE")
        bh,bw=self._brush_base.shape[:2]
        self._brush_transparent=np.zeros((bh,bw),np.uint8); self._brush_hole=np.zeros((bh,bw),np.uint8)
        self._brush_history=[]; self._brush_redo_stack=[]; self._brush_stroke_before={}; self._brush_stroke_tiles=set()
        self._brush_preview_items={}; self._brush_source_tile_cache={}; self._brush_tile_size=256
        self._brush_interaction=False

        # Right inspector.  This replaces both the oversized top selection bar
        # and the oversized bottom processing panel from v2.3.42.  At normal
        # height every control is visible at once; on a smaller display only
        # this inspector scrolls while the canvases and footer stay fixed.
        inspector_scroll=QScrollArea(); inspector_scroll.setObjectName("regionInspectorScroll")
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inspector_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        inspector_scroll.setMinimumWidth(280); inspector_scroll.setMaximumWidth(340)
        inspector_scroll.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Expanding)
        inspector_host=QWidget(); inspector_host.setMinimumWidth(0)
        ip=QVBoxLayout(inspector_host); ip.setContentsMargins(5,2,5,2); ip.setSpacing(7)

        selection_panel=QFrame(); selection_panel.setObjectName("selectionPanel")
        sp=QVBoxLayout(selection_panel); sp.setContentsMargins(8,7,8,7); sp.setSpacing(5)
        title=QLabel("① 选区系统"); title.setObjectName("sectionTitle"); sp.addWidget(title)
        help_label=QLabel("左键拖选 / 手绘 · 右键或中键平移 · 滚轮缩放 · 双击适合窗口")
        help_label.setObjectName("quiet"); help_label.setWordWrap(True); help_label.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred); sp.addWidget(help_label)
        shapes_box=QWidget(); sg=QGridLayout(shapes_box); sg.setContentsMargins(0,0,0,0); sg.setHorizontalSpacing(5); sg.setVerticalSpacing(5)
        self.shape_group=QButtonGroup(self); self.shape_group.setExclusive(True); self.shape_buttons=[]
        shape_defs=[("矩形框","rect"),("椭圆框","ellipse"),("爆炸框 / 智能闭合","smart"),("手绘闭合","freehand")]
        for i,(label,key) in enumerate(shape_defs):
            b=QPushButton(label); b.setCheckable(True); b.setObjectName("segmented"); b.setProperty("selectionMode",key)
            b.setMinimumHeight(29); b.setMinimumWidth(96); b.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
            self.shape_group.addButton(b); self.shape_buttons.append(b); sg.addWidget(b,i//2,i%2)
        sg.setColumnStretch(0,1); sg.setColumnStretch(1,1); self.shape_buttons[0].setChecked(True); sp.addWidget(shapes_box)
        self.snap=QCheckBox("普通选区完成后自动吸附线稿"); self.snap.setChecked(False); sp.addWidget(self.snap)
        snap_row=QHBoxLayout(); snap_row.setContentsMargins(0,0,0,0); snap_row.addWidget(QLabel("吸附距离"))
        self.snap_distance=QSpinBox(); self.snap_distance.setRange(2,24); self.snap_distance.setValue(10); self.snap_distance.setSuffix(" px"); snap_row.addWidget(self.snap_distance); snap_row.addStretch(1); sp.addLayout(snap_row)
        refine_row=QGridLayout(); refine_row.setContentsMargins(0,0,0,0); refine_row.setHorizontalSpacing(5)
        self.smart_refine_btn=QPushButton("识别当前选框的闭合区域"); self.smart_refine_btn.setObjectName("softPrimary"); self.smart_refine_btn.setMinimumHeight(30)
        self.clear_selection_btn=QPushButton("清除选区"); self.clear_selection_btn.setMinimumHeight(30)
        refine_row.addWidget(self.smart_refine_btn,0,0); refine_row.addWidget(self.clear_selection_btn,0,1); refine_row.setColumnStretch(0,2); refine_row.setColumnStretch(1,1); sp.addLayout(refine_row)
        ip.addWidget(selection_panel)

        tool_panel=QFrame(); tool_panel.setObjectName("selectionPanel")
        tp=QVBoxLayout(tool_panel); tp.setContentsMargins(8,7,8,7); tp.setSpacing(5)
        tools_head=QLabel("② 区域处理工具"); tools_head.setObjectName("sectionTitle"); tp.addWidget(tools_head)
        tools_widget=QWidget(); tg=QGridLayout(tools_widget); tg.setContentsMargins(0,0,0,0); tg.setHorizontalSpacing(5); tg.setVerticalSpacing(5)
        self.tool_group=QButtonGroup(self); self.tool_group.setExclusive(True); self.tool_buttons=[]
        for i,(label,key,_hint) in enumerate(self._TOOLS):
            b=QPushButton(label); b.setCheckable(True); b.setObjectName("segmented"); b.setProperty("regionTool",key)
            b.setMinimumHeight(29); b.setMinimumWidth(96); b.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
            self.tool_group.addButton(b); self.tool_buttons.append(b); tg.addWidget(b,i//2,i%2)
        tg.setColumnStretch(0,1); tg.setColumnStretch(1,1); self.tool_buttons[1].setChecked(True); tp.addWidget(tools_widget)
        self.tool_hint=QLabel(); self.tool_hint.setObjectName("quiet"); self.tool_hint.setWordWrap(True); self.tool_hint.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred); tp.addWidget(self.tool_hint)
        self.performance_hint=QLabel("精准蒙版快速 ROI：只计算选区和安全外扩区，选区外仍强制 0 像素写入。")
        self.performance_hint.setObjectName("hint"); self.performance_hint.setWordWrap(True); self.performance_hint.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred); tp.addWidget(self.performance_hint)

        params_widget=QWidget(); pg=QGridLayout(params_widget); pg.setContentsMargins(0,0,0,0); pg.setHorizontalSpacing(5); pg.setVerticalSpacing(4)
        self.feather=QSpinBox(); self.feather.setRange(0,8); self.feather.setValue(0); self.feather.setSuffix(" px")
        self.inset=QSpinBox(); self.inset.setRange(0,12); self.inset.setValue(1); self.inset.setSuffix(" px")
        self.offset_x=QSpinBox(); self.offset_x.setRange(-100,100); self.offset_x.setSuffix(" px")
        self.offset_y=QSpinBox(); self.offset_y.setRange(-100,100); self.offset_y.setSuffix(" px")
        pg.addWidget(QLabel("边缘羽化"),0,0); pg.addWidget(self.feather,0,1)
        pg.addWidget(QLabel("挖孔内缩"),1,0); pg.addWidget(self.inset,1,1)
        pg.addWidget(QLabel("SOURCE X 微调"),2,0); pg.addWidget(self.offset_x,2,1)
        pg.addWidget(QLabel("SOURCE Y 微调"),3,0); pg.addWidget(self.offset_y,3,1)
        pg.setColumnStretch(1,1); tp.addWidget(params_widget)

        self.ocr_box=QFrame(); self.ocr_box.setObjectName("card")
        op=QVBoxLayout(self.ocr_box); op.setContentsMargins(7,6,7,6); op.setSpacing(5)
        ocrrow=QHBoxLayout(); self.ocr_btn=QPushButton("识别当前选区"); self.ocr_btn.setObjectName("softPrimary")
        self.ocr_status=QLabel("可先识别，再直接修改中文"); self.ocr_status.setObjectName("quiet"); self.ocr_status.setWordWrap(True); self.ocr_status.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        ocrrow.addWidget(self.ocr_btn); ocrrow.addWidget(self.ocr_status,1); op.addLayout(ocrrow)
        self.ocr_text=QPlainTextEdit(); self.ocr_text.setMaximumHeight(74); self.ocr_text.setPlaceholderText("区域 OCR 中文结果；可手工修改后再应用。"); op.addWidget(self.ocr_text)
        og=QGridLayout(); og.setHorizontalSpacing(5); og.setVerticalSpacing(4)
        self.ocr_orientation=QComboBox(); self.ocr_orientation.addItem("竖排","vertical"); self.ocr_orientation.addItem("横排","horizontal"); self.ocr_orientation.addItem("自动","auto"); self.ocr_orientation.setCurrentIndex(0)
        self.ocr_font_size=QSpinBox(); self.ocr_font_size.setRange(0,160); self.ocr_font_size.setSpecialValueText("自动"); self.ocr_font_size.setSuffix(" px")
        self.ocr_columns=QSpinBox(); self.ocr_columns.setRange(0,12); self.ocr_columns.setSpecialValueText("自动")
        self.ocr_break=QComboBox(); self.ocr_break.addItem("智能断句","smart"); self.ocr_break.addItem("均衡断句","balanced"); self.ocr_break.addItem("保留源换行","source")
        self.ocr_layout=QComboBox(); self.ocr_layout.addItem("智能缩放","smart_scaling"); self.ocr_layout.addItem("严格字号","strict"); self.ocr_layout.addItem("填充文本框","balloon_fill")
        og.addWidget(QLabel("方向"),0,0); og.addWidget(self.ocr_orientation,0,1); og.addWidget(QLabel("字号"),1,0); og.addWidget(self.ocr_font_size,1,1)
        og.addWidget(QLabel("列数"),2,0); og.addWidget(self.ocr_columns,2,1); og.addWidget(QLabel("断句"),3,0); og.addWidget(self.ocr_break,3,1)
        og.addWidget(QLabel("排版"),4,0); og.addWidget(self.ocr_layout,4,1); og.setColumnStretch(1,1); op.addLayout(og)
        tp.addWidget(self.ocr_box); ip.addWidget(tool_panel)

        brush_panel=QFrame(); brush_panel.setObjectName("selectionPanel")
        bp=QVBoxLayout(brush_panel); bp.setContentsMargins(8,7,8,7); bp.setSpacing(5)
        brush_head=QLabel("③ 涂抹揭示 / 挖孔"); brush_head.setObjectName("sectionTitle"); bp.addWidget(brush_head)
        brush_help=QLabel("左键实时揭示；右键临时恢复日文；中键平移。透明揭示使用软边透明，挖孔是硬剪裁。每一笔都可撤销/重做。")
        brush_help.setObjectName("quiet"); brush_help.setWordWrap(True); brush_help.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred); bp.addWidget(brush_help)
        brush_modes=QWidget(); bg=QGridLayout(brush_modes); bg.setContentsMargins(0,0,0,0); bg.setHorizontalSpacing(5); bg.setVerticalSpacing(5)
        self.brush_group=QButtonGroup(self); self.brush_group.setExclusive(True); self.brush_buttons=[]
        for i,(label,key) in enumerate((("透明揭示","transparent"),("挖孔揭示","hole"),("恢复日文","restore"))):
            b=QPushButton(label); b.setCheckable(True); b.setObjectName("segmented"); b.setProperty("brushRevealMode",key); b.setMinimumHeight(29); b.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
            self.brush_group.addButton(b); self.brush_buttons.append(b); bg.addWidget(b,0,i)
        bg.setColumnStretch(0,1); bg.setColumnStretch(1,1); bg.setColumnStretch(2,1); bp.addWidget(brush_modes)
        br=QGridLayout(); br.setHorizontalSpacing(5); br.setVerticalSpacing(4)
        self.brush_size=QSlider(Qt.Orientation.Horizontal); self.brush_size.setRange(4,240); self.brush_size.setValue(42)
        self.brush_size_label=QLabel("42 px")
        self.brush_feather=QSpinBox(); self.brush_feather.setRange(0,24); self.brush_feather.setValue(3); self.brush_feather.setSuffix(" px")
        self.brush_opacity=QSpinBox(); self.brush_opacity.setRange(10,100); self.brush_opacity.setValue(100); self.brush_opacity.setSuffix(" %")
        br.addWidget(QLabel("画笔"),0,0); br.addWidget(self.brush_size,0,1); br.addWidget(self.brush_size_label,0,2)
        br.addWidget(QLabel("透明羽化"),1,0); br.addWidget(self.brush_feather,1,1,1,2)
        br.addWidget(QLabel("透明强度"),2,0); br.addWidget(self.brush_opacity,2,1,1,2); br.setColumnStretch(1,1); bp.addLayout(br)
        self.brush_limit_selection=QCheckBox("仅在当前选区内涂抹"); self.brush_limit_selection.setChecked(False); bp.addWidget(self.brush_limit_selection)
        brush_actions=QGridLayout(); brush_actions.setHorizontalSpacing(5); brush_actions.setVerticalSpacing(5)
        self.brush_undo_btn=QPushButton("撤销一笔"); self.brush_redo_btn=QPushButton("重做一笔"); self.brush_clear_btn=QPushButton("全部恢复日文")
        brush_actions.addWidget(self.brush_undo_btn,0,0); brush_actions.addWidget(self.brush_redo_btn,0,1); brush_actions.addWidget(self.brush_clear_btn,1,0,1,2); brush_actions.setColumnStretch(0,1); brush_actions.setColumnStretch(1,1); bp.addLayout(brush_actions)
        self.brush_status=QLabel("尚未涂抹 · 选择“透明揭示”或“挖孔揭示”后直接在中间画布操作")
        self.brush_status.setObjectName("hint"); self.brush_status.setWordWrap(True); self.brush_status.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred); bp.addWidget(self.brush_status)
        ip.addWidget(brush_panel); ip.addStretch(1)
        inspector_scroll.setWidget(inspector_host)

        body_grid.addWidget(refs,0,0); body_grid.addWidget(current_card,0,1); body_grid.addWidget(inspector_scroll,0,2)
        body_grid.setColumnStretch(0,0); body_grid.setColumnStretch(1,1); body_grid.setColumnStretch(2,0); body_grid.setRowStretch(0,1)
        root.addWidget(body,1)
        self._region_body=body; self._region_body_grid=body_grid; self._region_refs=refs; self._region_current=current_card; self._region_inspector=inspector_scroll

        # Sticky footer: v2.3.42 placed this after a tall tools block and the
        # Apply button could end up outside the window.  Here it is never inside
        # a scroll area and long status text is explicitly allowed to shrink.
        footer_panel=QFrame(); footer_panel.setObjectName("selectionPanel")
        footer=QHBoxLayout(footer_panel); footer.setContentsMargins(8,6,8,6); footer.setSpacing(7)
        status_box=QWidget(); sv=QVBoxLayout(status_box); sv.setContentsMargins(0,0,0,0); sv.setSpacing(1)
        self.selection_status=QLabel("尚未选择区域 · 请在中间当前结果上左键拖选")
        self.selection_status.setObjectName("hint"); self.selection_status.setWordWrap(True); self.selection_status.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        self.undo_hint=QLabel("涂抹可在本窗口逐笔撤销/重做；提交后仍可在工作台撤销最近区域动作"); self.undo_hint.setObjectName("quiet"); self.undo_hint.setWordWrap(True); self.undo_hint.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        sv.addWidget(self.selection_status); sv.addWidget(self.undo_hint); footer.addWidget(status_box,1)
        self.apply_btn=QPushButton("应用当前工具到选区"); self.apply_btn.setObjectName("primary"); self.apply_btn.setMinimumHeight(36); self.apply_btn.setMinimumWidth(178); self.apply_btn.setMaximumWidth(240)
        close=QPushButton("完成"); close.setMinimumHeight(36); close.setMinimumWidth(82); close.setMaximumWidth(120)
        footer.addWidget(self.apply_btn,0); footer.addWidget(close,0); root.addWidget(footer_panel,0)

        for b in self.shape_buttons: b.clicked.connect(self._shape_changed)
        for b in self.tool_buttons: b.clicked.connect(self._tool_changed)
        for b in self.brush_buttons: b.clicked.connect(self._brush_mode_changed)
        self.snap.toggled.connect(lambda v:self.target_view.set_snap_enabled(bool(v)))
        self.snap_distance.valueChanged.connect(self.target_view.set_snap_distance)
        self.smart_refine_btn.clicked.connect(self._refine_current_selection)
        self.clear_selection_btn.clicked.connect(self._clear_selection)
        self.target_view.selection_changed.connect(self._selection_changed)
        self.target_view.brush_stroke_started.connect(self._brush_stroke_started)
        self.target_view.brush_stroke_segment.connect(self._brush_stroke_segment)
        self.target_view.brush_stroke_finished.connect(self._brush_stroke_finished)
        self.brush_size.valueChanged.connect(self._brush_size_changed)
        self.brush_feather.valueChanged.connect(lambda _v:self._refresh_all_brush_tiles())
        self.brush_opacity.valueChanged.connect(lambda _v:self._refresh_all_brush_tiles())
        self.offset_x.valueChanged.connect(lambda _v:self._brush_alignment_changed())
        self.offset_y.valueChanged.connect(lambda _v:self._brush_alignment_changed())
        self.brush_undo_btn.clicked.connect(self._brush_undo); self.brush_redo_btn.clicked.connect(self._brush_redo); self.brush_clear_btn.clicked.connect(self._brush_clear)
        self.ocr_text.textChanged.connect(self._ocr_text_changed)
        self.ocr_btn.clicked.connect(self._recognize_ocr); self.apply_btn.clicked.connect(self._apply_tool); close.clicked.connect(self.accept)
        self.target_view.set_selection_mode("rect"); self.target_view.set_interaction_mode("selection"); self.target_view.set_snap_enabled(False); self.target_view.set_snap_distance(10); self.target_view.set_brush_size(42)
        self._update_brush_controls(); self._tool_changed(); self._update_region_action_state(); QTimer.singleShot(0,self._finish_region_layout)

    def _trace_region(self,stage:str,payload:dict[str,Any]|None=None):
        if self._trace_handler is None: return
        try: self._trace_handler(str(stage),dict(payload or {}))
        except Exception: logger.debug("region workspace trace failed",exc_info=True)

    def _apply_region_column_widths(self):
        """Keep both side workspaces visible at every supported window width."""
        if not hasattr(self,"_region_refs") or not hasattr(self,"_region_inspector"):
            return
        available=max(760,int(self.width())-36)
        if available < 1080:
            refs_w, inspector_w = 180, 280
        elif available < 1320:
            refs_w, inspector_w = 210, 292
        elif available < 1540:
            refs_w, inspector_w = 250, 312
        else:
            refs_w, inspector_w = 280, 328
        self._region_refs.setFixedWidth(int(refs_w))
        self._region_inspector.setFixedWidth(int(inspector_w))

    def _finish_region_layout(self):
        """Finalize bounded columns and image fitting in the normal dialog."""
        # v2.3.48 intentionally restores the pre-v2.3.45 windowed editor size.
        # Keep the responsive three-column protections from v2.3.47, but never
        # promote/maximise the dialog here: the user explicitly wants the
        # original independent editor size back.
        self._apply_region_column_widths(); self._fit()
        # Cocoa can still settle the native frame one or two event-loop turns
        # later. Re-apply only the bounded columns/fit, never window geometry.
        for delay in (50,180):
            QTimer.singleShot(delay,self._settle_region_layout)

    def _settle_region_layout(self):
        self._apply_region_column_widths(); self._fit()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,"_region_refs"):
            self._apply_region_column_widths()

    def _fit(self):
        self.jp_view.fit_to_window(); self.source_view.fit_to_window(); self.target_view.fit_to_window()

    def _checked_property(self,group:QButtonGroup,name:str,default:str)->str:
        b=group.checkedButton(); return str(b.property(name) if b is not None else default)

    @staticmethod
    def _set_group_value(group:QButtonGroup, buttons, property_name:str, value:str|None):
        """Select one button (or none) without letting two action families lie.

        Region tools and brush tools live in separate Qt button groups for the
        layout, so Qt cannot make them mutually exclusive by itself.  Temporarily
        disabling exclusivity is the only reliable way to clear a checked button.
        """
        wanted=str(value or "")
        group.setExclusive(False)
        try:
            for button in buttons:
                button.setChecked(bool(wanted and str(button.property(property_name) or "") == wanted))
        finally:
            group.setExclusive(True)

    def _selection_snapshot(self)->tuple[dict[str,Any],list[int],str]:
        spec=copy.deepcopy(self.target_view.selection_spec())
        box=[int(v) for v in self.target_view.box()]
        sig=selection_signature(spec)
        return spec,box,sig

    def _pending_brush_pixels(self)->int:
        if not hasattr(self,"_brush_transparent"):
            return 0
        return int(brush_mask_counts(self._brush_transparent,self._brush_hole).get("union_pixels",0) or 0)

    def _invalidate_ocr_binding(self, *, keep_user_text:bool=False):
        self._ocr_payload={}; self._link_state.clear_ocr()
        self._ocr_programmatic_change=True
        try:
            if not keep_user_text: self.ocr_text.clear()
        finally:
            self._ocr_programmatic_change=False
        if hasattr(self,"ocr_status"):
            self.ocr_status.setText("选区已变化 · OCR 结果已解绑，请重新识别或输入当前区域中文")

    def _ocr_text_changed(self):
        if self._ocr_programmatic_change: return
        # Manual text is selection-owned too.  It may replace recognition text,
        # but detector polygons from an older selection can never follow it into
        # a new box.  Bind typed text to the current selection and discard stale
        # OCR geometry; a later selection change will invalidate the text again.
        if self.ocr_text.toPlainText().strip() and len(self.target_view.box())==4:
            self._ocr_payload={}
            self._link_state.bind_ocr(self.target_view.selection_spec())
        elif not self.ocr_text.toPlainText().strip():
            self._ocr_payload={}; self._link_state.clear_ocr()

    def _update_region_action_state(self):
        if not hasattr(self,"apply_btn"): return
        valid_selection=len(self.target_view.box())==4 and bool(selection_signature(self.target_view.selection_spec()))
        pending=self._pending_brush_pixels()
        enabled=self._link_state.can_apply(selection_valid=valid_selection,pending_brush_pixels=pending)
        if getattr(self,"_ocr_busy",False) and self._tool_key()=="region_ocr":
            enabled=False
        self.apply_btn.setEnabled(enabled)
        if self._link_state.active_family=="brush":
            self.apply_btn.setText("提交当前涂抹揭示" if pending else "先在画布涂抹")
        else:
            self.apply_btn.setText("应用当前工具到选区" if valid_selection else "先建立选区")

    def _shape_changed(self,*_):
        self._activate_selection_interaction(restore_region_tool=True)
        mode=self._checked_property(self.shape_group,"selectionMode","rect")
        self.target_view.set_selection_mode(mode)
        names={"rect":"矩形框","ellipse":"椭圆框","smart":"爆炸框 / 智能闭合","freehand":"手绘闭合"}
        # Rect/ellipse are deterministic shape changes: when an existing bbox is
        # present, update it immediately instead of merely changing the *next*
        # drag mode while leaving a contradictory old selection active.
        box=self.target_view.box()
        if len(box)==4 and mode in {"rect","ellipse"}:
            self.target_view.set_selection_spec({"schema":"folirina.region_selection.v1","kind":mode,"bbox":box,"points":[],"snapped":False},emit=True)
        else:
            suffix=" · 当前选区保持不变；重新拖选后使用此形状" if len(box)==4 else " · 请在中间当前结果上左键操作"
            self.selection_status.setText(f"已选择 {names.get(mode,mode)}{suffix}")
        self._update_region_action_state()

    def _tool_key(self)->str: return self._checked_property(self.tool_group,"regionTool","region_precise_mask")

    def _tool_changed(self,*_):
        key=self._tool_key(); self._link_state.activate_region(key)
        self._activate_selection_interaction(restore_region_tool=False)
        hints={k:h for _l,k,h in self._TOOLS}; self.tool_hint.setText(hints.get(key,""))
        is_ocr=key=="region_ocr"; self.ocr_box.setVisible(is_ocr); self.performance_hint.setVisible(key=="region_precise_mask")
        self.inset.setEnabled(key=="region_hole_reveal")
        self.feather.setEnabled(key in {"region_direct_patch","region_hole_reveal","region_transparent"})
        offsets=key!="region_ocr"; self.offset_x.setEnabled(offsets); self.offset_y.setEnabled(offsets)
        if key=="region_hole_reveal" and self.feather.value()==0: self.feather.setValue(1)
        self._trace_region("region_tool_activated",{"tool":key,"selection_bbox":self.target_view.box()})
        self._update_region_action_state()

    def _uncheck_brush_buttons(self):
        if not hasattr(self,"brush_group"): return
        self.brush_group.setExclusive(False)
        for button in self.brush_buttons: button.setChecked(False)
        self.brush_group.setExclusive(True)

    def _activate_selection_interaction(self, *, restore_region_tool:bool=False):
        self._brush_interaction=False
        self._link_state.active_family="region"
        if hasattr(self,"target_view"):
            self.target_view.set_interaction_mode("selection")
        self._uncheck_brush_buttons()
        if restore_region_tool and hasattr(self,"tool_group") and self.tool_group.checkedButton() is None:
            key=self._link_state.activate_region(self._link_state.last_region_tool)
            self._set_group_value(self.tool_group,self.tool_buttons,"regionTool",key)
        self._update_region_action_state()

    def _brush_mode_key(self)->str:
        return self._checked_property(self.brush_group,"brushRevealMode","transparent")

    def _brush_mode_changed(self,*_):
        key=self._brush_mode_key(); self._link_state.activate_brush(key)
        self._brush_interaction=True
        # A single action state must be visible.  Keep the last region tool in
        # linkage state so selecting a shape later can restore it automatically.
        checked=self.tool_group.checkedButton()
        if checked is not None:
            self._link_state.last_region_tool=str(checked.property("regionTool") or self._link_state.last_region_tool)
        self._set_group_value(self.tool_group,self.tool_buttons,"regionTool",None)
        self.target_view.set_interaction_mode("brush")
        self.target_view.set_brush_size(int(self.brush_size.value()))
        names={"transparent":"透明揭示","hole":"挖孔揭示","restore":"恢复日文"}
        self.selection_status.setText(f"涂抹模式 · {names.get(key,key)} · 左键涂抹，右键恢复，中键平移")
        self._trace_region("region_brush_activated",{"brush_mode":key,"selection_bbox":self.target_view.box()})
        self._update_brush_controls(); self._update_region_action_state()

    def _brush_size_changed(self,value:int):
        self.target_view.set_brush_size(int(value)); self.brush_size_label.setText(f"{int(value)} px")

    def _tile_bounds(self,key):
        tx,ty=[int(v) for v in key]; size=int(self._brush_tile_size); h,w=self._brush_transparent.shape
        x0=tx*size; y0=ty*size; return x0,y0,min(w,x0+size),min(h,y0+size)

    def _tiles_for_bbox(self,bbox):
        if not bbox or len(bbox)!=4: return set()
        x0,y0,x1,y1=[int(v) for v in bbox]; size=int(self._brush_tile_size)
        if x1<=x0 or y1<=y0: return set()
        return {(tx,ty) for ty in range(max(0,y0)//size,(max(0,y1-1)//size)+1) for tx in range(max(0,x0)//size,(max(0,x1-1)//size)+1)}

    def _remember_before_tiles(self,keys):
        for key in keys:
            if key in self._brush_stroke_before: continue
            x0,y0,x1,y1=self._tile_bounds(key)
            self._brush_stroke_before[key]=(self._brush_transparent[y0:y1,x0:x1].copy(),self._brush_hole[y0:y1,x0:x1].copy())
            self._brush_stroke_tiles.add(key)

    def _source_tile(self,key):
        dx=int(self.offset_x.value()); dy=int(self.offset_y.value()); cache_key=(int(key[0]),int(key[1]),dx,dy)
        cached=self._brush_source_tile_cache.get(cache_key)
        if cached is not None: return cached
        from .region_composite import _aligned_source_roi
        roi=self._tile_bounds(key)
        crop,_identity=_aligned_source_roi(self._brush_source,self._brush_base.shape[:2],self.project,roi,source_offset_x=dx,source_offset_y=dy)
        self._brush_source_tile_cache[cache_key]=crop
        return crop

    def _refresh_brush_tile(self,key):
        x0,y0,x1,y1=self._tile_bounds(key)
        tr=self._brush_transparent[y0:y1,x0:x1]; ho=self._brush_hole[y0:y1,x0:x1]
        alpha_present=bool(np.any(tr>0) or np.any(ho>0))
        item=self._brush_preview_items.get(key)
        if not alpha_present:
            if item is not None:
                try: self.target_view._scene.removeItem(item)
                except RuntimeError: pass
                self._brush_preview_items.pop(key,None)
            return
        base=self._brush_base[y0:y1,x0:x1]
        under=self._source_tile(key)
        preview,_patch=compose_reveal_patch(base,under,tr,ho,transparent_feather_px=int(self.brush_feather.value()),transparent_opacity=float(self.brush_opacity.value())/100.0)
        rgb=cv2.cvtColor(preview,cv2.COLOR_BGR2RGB); hh,ww=rgb.shape[:2]
        q=QImage(rgb.data,ww,hh,int(rgb.strides[0]),QImage.Format.Format_RGB888).copy(); pix=QPixmap.fromImage(q)
        if item is None:
            item=self.target_view._scene.addPixmap(pix); item.setZValue(2.5); self._brush_preview_items[key]=item
        else: item.setPixmap(pix)
        item.setPos(x0,y0)

    def _refresh_brush_tiles(self,keys):
        for key in set(keys or []): self._refresh_brush_tile(key)
        self._update_brush_controls()

    def _refresh_all_brush_tiles(self):
        if not hasattr(self,"_brush_transparent"): return
        box=brush_mask_bbox(self._brush_transparent,self._brush_hole)
        keys=set(self._brush_preview_items.keys())|self._tiles_for_bbox(box)
        self._refresh_brush_tiles(keys)

    def _brush_alignment_changed(self):
        if not hasattr(self,"_brush_source_tile_cache"): return
        self._brush_source_tile_cache.clear(); self._refresh_all_brush_tiles()

    def _brush_stroke_started(self,payload):
        self._brush_stroke_before={}; self._brush_stroke_tiles=set()

    def _brush_stroke_segment(self,payload):
        if not self._brush_interaction: return
        start=list((payload or {}).get("from") or []); end=list((payload or {}).get("to") or [])
        if len(start)!=2 or len(end)!=2: return
        diameter=int((payload or {}).get("brush_size") or self.brush_size.value())
        candidate=brush_stroke_bbox(self._brush_transparent.shape,[start,end],diameter)
        keys=self._tiles_for_bbox(candidate); self._remember_before_tiles(keys)
        limit=None
        if self.brush_limit_selection.isChecked():
            if cv2.countNonZero(self.target_view._selection_mask)<=0:
                self.brush_status.setText("已启用“仅在当前选区内涂抹”，但当前没有选区。先建立选区或关闭限制。")
                return
            limit=self.target_view._selection_mask
        mode="restore" if bool((payload or {}).get("right")) else self._brush_mode_key()
        diag=paint_reveal_stroke_inplace(self._brush_transparent,self._brush_hole,[start,end],diameter,mode,limit_mask=limit)
        if diag.changed_pixels>0: self._refresh_brush_tiles(self._tiles_for_bbox(diag.bbox))

    def _brush_stroke_finished(self):
        entries=[]; changed_keys=set()
        for key in sorted(self._brush_stroke_tiles):
            before=self._brush_stroke_before.get(key)
            if before is None: continue
            x0,y0,x1,y1=self._tile_bounds(key); after=(self._brush_transparent[y0:y1,x0:x1].copy(),self._brush_hole[y0:y1,x0:x1].copy())
            if np.array_equal(before[0],after[0]) and np.array_equal(before[1],after[1]): continue
            entries.append((key,before[0],before[1],after[0],after[1])); changed_keys.add(key)
        self._brush_stroke_before={}; self._brush_stroke_tiles=set()
        if entries:
            self._brush_history.append(entries)
            if len(self._brush_history)>100: self._brush_history=self._brush_history[-100:]
            self._brush_redo_stack.clear()
        self._refresh_brush_tiles(changed_keys)

    def _restore_brush_record(self,record,after:bool):
        keys=set()
        for key,btr,bho,atr,aho in record:
            x0,y0,x1,y1=self._tile_bounds(key)
            self._brush_transparent[y0:y1,x0:x1]=atr if after else btr
            self._brush_hole[y0:y1,x0:x1]=aho if after else bho
            keys.add(key)
        self._refresh_brush_tiles(keys)

    def _brush_undo(self):
        if not self._brush_history: return
        record=self._brush_history.pop(); self._restore_brush_record(record,False); self._brush_redo_stack.append(record); self._update_brush_controls()

    def _brush_redo(self):
        if not self._brush_redo_stack: return
        record=self._brush_redo_stack.pop(); self._restore_brush_record(record,True); self._brush_history.append(record); self._update_brush_controls()

    def _brush_clear(self):
        box=brush_mask_bbox(self._brush_transparent,self._brush_hole)
        keys=self._tiles_for_bbox(box)
        if not keys: return
        record=[]
        for key in sorted(keys):
            x0,y0,x1,y1=self._tile_bounds(key); btr=self._brush_transparent[y0:y1,x0:x1].copy(); bho=self._brush_hole[y0:y1,x0:x1].copy()
            if not np.any(btr) and not np.any(bho): continue
            atr=np.zeros_like(btr); aho=np.zeros_like(bho); record.append((key,btr,bho,atr,aho))
            self._brush_transparent[y0:y1,x0:x1]=0; self._brush_hole[y0:y1,x0:x1]=0
        if record:
            self._brush_history.append(record); self._brush_redo_stack.clear(); self._refresh_brush_tiles(keys)

    def _update_brush_controls(self):
        if not hasattr(self,"brush_status"): return
        counts=brush_mask_counts(self._brush_transparent,self._brush_hole)
        self.brush_undo_btn.setEnabled(bool(self._brush_history)); self.brush_redo_btn.setEnabled(bool(self._brush_redo_stack)); self.brush_clear_btn.setEnabled(counts["union_pixels"]>0)
        self.brush_status.setText(f"透明 {counts['transparent_pixels']:,} px · 挖孔 {counts['hole_pixels']:,} px · 共 {counts['union_pixels']:,} px · 撤销 {len(self._brush_history)} / 重做 {len(self._brush_redo_stack)}")
        self._update_region_action_state()

    def _clear_brush_preview_items(self):
        for item in list(self._brush_preview_items.values()):
            try: self.target_view._scene.removeItem(item)
            except RuntimeError: pass
        self._brush_preview_items.clear()

    def _reset_brush_session(self,final_path:Path|None=None):
        if final_path is not None and final_path.exists():
            image=cv2.imread(str(final_path),cv2.IMREAD_COLOR)
            if image is not None and image.shape==self._brush_base.shape:
                self._brush_base=image; self.target_view.replace_display_image(final_path)
        self._brush_transparent[:]=0; self._brush_hole[:]=0; self._brush_history.clear(); self._brush_redo_stack.clear(); self._brush_stroke_before={}; self._brush_stroke_tiles=set(); self._brush_source_tile_cache.clear(); self._clear_brush_preview_items(); self._update_brush_controls()

    def _build_brush_commit(self):
        import uuid
        box=brush_mask_bbox(self._brush_transparent,self._brush_hole)
        if len(box)!=4: raise ValueError("当前没有可提交的涂抹揭示")
        x0,y0,x1,y1=box
        tr=self._brush_transparent[y0:y1,x0:x1].copy(); ho=self._brush_hole[y0:y1,x0:x1].copy()
        from .region_composite import _aligned_source_roi
        under,_identity=_aligned_source_roi(self._brush_source,self._brush_base.shape[:2],self.project,(x0,y0,x1,y1),source_offset_x=int(self.offset_x.value()),source_offset_y=int(self.offset_y.value()))
        base=self._brush_base[y0:y1,x0:x1]
        _preview,patch=compose_reveal_patch(base,under,tr,ho,transparent_feather_px=int(self.brush_feather.value()),transparent_opacity=float(self.brush_opacity.value())/100.0)
        if cv2.countNonZero(patch[:,:,3])<=0: raise ValueError("涂抹揭示的有效透明区域为空")
        counts=brush_mask_counts(tr,ho); kinds=[]
        if counts["transparent_pixels"]: kinds.append("transparent")
        if counts["hole_pixels"]: kinds.append("hole")
        row={
            "id":f"region-brush-{uuid.uuid4().hex[:10]}","enabled":True,"mode":"region_brush_reveal","target_bbox":box,
            "reveal_patch_bbox":box,"reveal_mask_bbox":box,"origin":"region_composite_editor","tool_kind":"region_brush_reveal","owner_transfer_mode":"",
            "source_offset_x":int(self.offset_x.value()),"source_offset_y":int(self.offset_y.value()),"brush_size_px":int(self.brush_size.value()),
            "transparent_feather_px":int(self.brush_feather.value()),"transparent_opacity":float(self.brush_opacity.value())/100.0,
            "brush_reveal_kind":"+".join(kinds) if kinds else "none",**counts,
        }
        union=np.maximum(tr,ho).astype(np.uint8)
        return row,union,patch

    def _commit_brush_session(self, *, interactive:bool=True)->bool:
        if self._commit_handler is None:
            if interactive: QMessageBox.warning(self,"无法提交","当前区域编辑器没有连接复核提交器。")
            return False
        try: row,reveal,patch=self._build_brush_commit()
        except Exception as exc:
            if interactive: QMessageBox.information(self,"没有涂抹",str(exc))
            return False
        old_text=self.apply_btn.text(); self.apply_btn.setText("正在提交涂抹…"); self.apply_btn.setEnabled(False)
        self._trace_region("region_brush_commit_requested",{"target_bbox":list(row.get("target_bbox") or []),"pixels":int(row.get("union_pixels",0) or 0)})
        own_cursor=not self._link_state.applying
        if own_cursor: QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            QApplication.processEvents(); final=self._commit_handler(row,reveal,patch); self._applied+=1
            final_path=Path(str(final)) if final else self.page_dir/"final_reviewed.png"
            self._reset_brush_session(final_path)
            self.selection_status.setText(f"涂抹揭示已提交 · 已叠加 {self._applied} 步 · 可继续涂抹或切回选区工具")
            self._trace_region("region_brush_commit_succeeded",{"target_bbox":list(row.get("target_bbox") or []),"applied_count":self._applied})
            return True
        except Exception as exc:
            self._trace_region("region_brush_commit_failed",{"reason":str(exc)})
            if interactive: QMessageBox.critical(self,"涂抹揭示提交失败",str(exc))
            return False
        finally:
            if own_cursor: QApplication.restoreOverrideCursor()
            self.apply_btn.setText("提交当前涂抹揭示" if self._brush_interaction else old_text)
            self._update_region_action_state()

    def _apply_brush_session(self):
        self._commit_brush_session(interactive=True)

    def _clear_selection(self):
        self.target_view.clear_selection(emit=True); self.jp_view.clear_selection(); self.source_view.clear_selection()

    def _refine_current_selection(self):
        if len(self.target_view.box())!=4:
            QMessageBox.information(self,"没有选区","请先画矩形、椭圆或手绘闭合区域，再识别附近闭合边界。"); return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            QApplication.processEvents(); ok=self.target_view.recognize_closed_region()
            if not ok:
                diag=self.target_view.snap_diagnostics(); reason=str(diag.get("reason") or "")
                messages={
                    "no_closed_region":"当前选框内没有检测到可靠的封闭线稿区域。请让粗框完整包住气泡边界，并留少量外侧背景。",
                    "no_paintable_region":"选框内几乎全是结构线，无法形成可填充闭合区域。请稍微扩大选框。",
                    "empty_selection":"当前选区无效，请重新框选。",
                    "empty_image":"无法读取高清日文原图。",
                }
                QMessageBox.warning(self,"闭合区域识别失败",messages.get(reason,"未找到可靠闭合区域，已保留原选区；可扩大粗框后重试。"))
        finally: QApplication.restoreOverrideCursor()

    def _selection_changed(self,bbox):
        box=list(bbox or [])
        if len(box)!=4:
            stale=self._link_state.bind_selection({})
            if stale: self._invalidate_ocr_binding()
            self.jp_view.clear_selection(); self.source_view.clear_selection(); self.selection_status.setText("尚未选择区域 · 请在中间当前结果上左键拖选"); self._update_region_action_state(); return
        spec=self.target_view.selection_spec(); self.jp_view.set_selection_spec(spec)
        if self._link_state.bind_selection(spec): self._invalidate_ocr_binding()
        try:
            src_spec=project_selection_spec(
                spec, registration_homography(self.project), self.source_view.image_shape(), target_to_source=True
            )
            if src_spec: self.source_view.set_selection_spec(src_spec)
            else: self.source_view.set_box(map_target_bbox_to_source(self.project,box))
        except (ValueError, TypeError, np.linalg.LinAlgError):
            self.source_view.set_box(map_target_bbox_to_source(self.project,box))
        diag=spec.get("diagnostics") or {}; snap_text=""
        if spec.get("snapped"): snap_text=" · 已识别闭合区域"
        elif diag.get("used_fallback"): snap_text=" · 未找到可靠闭合区域，保留原选区"
        roi=diag.get("roi_fraction")
        roi_text=f" · 闭合分析仅计算 {float(roi)*100:.1f}% 页面" if isinstance(roi,(int,float)) and float(roi)<0.99 else ""
        names={"rect":"矩形","ellipse":"椭圆","smart":"智能闭合","freehand":"手绘闭合"}
        kind=str(spec.get("kind") or "rect")
        self.selection_status.setText(f"{names.get(kind,kind)} · TARGET {box[0]},{box[1]}–{box[2]},{box[3]}{snap_text}{roi_text}")
        self._update_region_action_state()

    def _ocr_mode_key(self)->str:
        return str(((self.project.get("meta") or {}).get("transfer_mode") or "")).strip().lower()

    def closeEvent(self,event):
        for worker,label in ((getattr(self,"_ocr_worker",None),"OCR"),(getattr(self,"_commit_worker",None),"区域提交")):
            if worker is not None and worker.isRunning():
                if hasattr(self,"ocr_status"):
                    self.ocr_status.setText(f"{label}仍在运行，请稍候完成后再关闭。")
                event.ignore(); return
        super().closeEvent(event)
    def _recognize_ocr(self):
        box=self.target_view.box()
        if len(box)!=4:
            QMessageBox.information(self,"没有选区","请先选择 OCR 区域。"); return
        worker=getattr(self,"_ocr_worker",None)
        if worker is not None and worker.isRunning():
            self.ocr_status.setText("区域 OCR 正在处理中，请稍候…")
            return
        modules=_ocr_mode_modules(self._ocr_mode_key())
        if modules[0] is None:
            self.ocr_status.setText("当前模式没有可用的区域 OCR 后端。")
            return
        recognizer=modules[0].recognize_manual_ocr_block
        request_sig=selection_signature(self.target_view.selection_spec())
        self._ocr_request_signature=request_sig
        project=copy.deepcopy(self.project); source_path=Path(self.source_path); target_path=Path(self.target_path)
        bbox=[int(v) for v in box]; cfg=self.config.model_copy(deep=True)
        self._ocr_busy=True; self.ocr_btn.setEnabled(False); self.ocr_btn.setText("识别中…")
        self.ocr_status.setText("正在识别当前选区… 可继续查看，不会卡住界面")
        self._update_region_action_state()
        worker=PageActionWorker(
            "区域复合 OCR",
            lambda: recognizer(project,source_path,target_path,bbox,cfg,existing=None),
            parent=self,
        )
        self._ocr_worker=worker

        def done(payload):
            row=dict(payload or {})
            current_sig=selection_signature(self.target_view.selection_spec())
            same_selection=bool(current_sig) and current_sig==request_sig
            self._ocr_payload=row
            if same_selection:
                self._link_state.bind_ocr(self.target_view.selection_spec())
            else:
                self._link_state.clear_ocr()
            self._ocr_programmatic_change=True
            try:
                self.ocr_text.setPlainText(str(row.get("render_text") or row.get("ocr_text") or ""))
            finally:
                self._ocr_programmatic_change=False
            backend=str(row.get("source_backend") or "OCR")
            serr=str(row.get("source_ocr_error") or "").strip()
            text_value=str(row.get("render_text") or row.get("ocr_text") or "").strip()
            if not same_selection:
                self.ocr_status.setText("OCR 已完成，但选区已变化 · 文本已填入，请确认后重新识别")
            elif serr:
                self.ocr_status.setText(f"{backend} · SOURCE OCR 失败：{serr[:180]}")
            elif not text_value:
                self.ocr_status.setText("OCR 已执行，但 SOURCE ROI 没识别到文字；可调整选区或手动输入。")
            else:
                self.ocr_status.setText(f"{backend} · 识别完成 · 置信度 {float(row.get('confidence') or 0.0):.2f}")

        def failed(message):
            detail=str(message or "区域 OCR 未返回错误信息").strip()
            self.ocr_status.setText("区域 OCR 失败："+detail.splitlines()[0][:220])

        def finished():
            self._ocr_busy=False
            self._ocr_worker=None
            self.ocr_btn.setEnabled(True); self.ocr_btn.setText("识别当前选区")
            self._update_region_action_state()

        worker.done.connect(done); worker.failed.connect(failed); worker.finished.connect(finished); worker.finished.connect(worker.deleteLater)
        worker.start()

    def _row(self, *, selection_spec:dict[str,Any]|None=None, selection_box:list[int]|None=None, tool_key:str|None=None)->dict[str,Any]:
        import uuid
        spec=copy.deepcopy(selection_spec if selection_spec is not None else self.target_view.selection_spec())
        box=[int(v) for v in (selection_box if selection_box is not None else self.target_view.box())]
        key=str(tool_key or self._tool_key())
        row={
            "id":f"region-action-{uuid.uuid4().hex[:10]}","enabled":True,"mode":key,"target_bbox":box,"selection_spec":spec,
            "source_offset_x":int(self.offset_x.value()),"source_offset_y":int(self.offset_y.value()),"feather_px":int(self.feather.value()),"inset_px":int(self.inset.value()),
            "origin":"region_composite_editor","tool_kind":"region_composite","owner_transfer_mode":"","ocr_allowed":key=="region_ocr",
        }
        if key=="region_ocr":
            payload=dict(self._ocr_payload or {}) if self._link_state.ocr_matches_current_selection() else {}
            payload.pop("id",None); row.update(payload)
            row.update({
                "render_text":self.ocr_text.toPlainText().strip(),"ocr_text":str(payload.get("ocr_text") or self.ocr_text.toPlainText().strip()),
                "selection_spec":spec,"target_bbox":box,"orientation":str(self.ocr_orientation.currentData() or "auto"),
                "font_size":int(self.ocr_font_size.value()),"columns":int(self.ocr_columns.value()),
                "line_break_mode":str(self.ocr_break.currentData() or "smart"),"layout_mode":str(self.ocr_layout.currentData() or "smart_scaling"),
            })
        return row

    def _apply_tool(self):
        if self._brush_interaction:
            self._apply_brush_session(); return
        if getattr(self,"_commit_worker",None) is not None and self._commit_worker.isRunning():
            self.selection_status.setText("区域处理仍在后台执行，请稍候…")
            return
        spec,box,sig=self._selection_snapshot()
        if len(box)!=4 or not sig:
            self._trace_region("region_apply_blocked",{"reason":"no_selection"})
            QMessageBox.information(self,"没有选区","请先在中间“当前结果”画出处理区域。"); return
        key=self._link_state.activate_region(self._tool_key())
        self._trace_region("region_apply_requested",{"tool":key,"target_bbox":box,"selection_signature":sig,"pending_brush_pixels":self._pending_brush_pixels()})
        if key=="region_ocr" and not self.ocr_text.toPlainText().strip():
            self._recognize_ocr()
            self.ocr_status.setText("正在 OCR；识别完成后请确认文字，再点击应用。")
            return
        if self._commit_handler is None:
            QMessageBox.warning(self,"无法提交","当前区域编辑器没有连接复核提交器。"); return
        spec,box,sig=self._selection_snapshot(); row=self._row(selection_spec=spec,selection_box=box,tool_key=key)
        if self._pending_brush_pixels()>0:
            # Brush patch persistence is already sparse and quick. Commit it first
            # so review ordering remains deterministic before launching the heavy
            # region/OCR render in the worker thread.
            if not self._commit_brush_session(interactive=False):
                QMessageBox.warning(self,"提交失败","未提交涂抹无法保存；原笔触仍保留。")
                return
        old_text=self.apply_btn.text(); self.apply_btn.setText("后台处理中…"); self._link_state.applying=True; self.apply_btn.setEnabled(False)
        self.selection_status.setText(self.selection_status.text()+" · 后台处理中")
        worker=PageActionWorker("区域复合提交",lambda:self._commit_handler(row,None,None),parent=self)
        self._commit_worker=worker

        def done(result):
            try:
                finalized=self._commit_finalize_handler(row,result) if self._commit_finalize_handler is not None else result
                final_path=Path(str(getattr(finalized,"final_reviewed",finalized) or self.page_dir/"final_reviewed.png"))
                self._applied+=1
                if final_path.exists():
                    self.target_view.replace_display_image(final_path)
                    refreshed=cv2.imread(str(final_path),cv2.IMREAD_COLOR)
                    if refreshed is not None and refreshed.shape==self._brush_base.shape: self._brush_base=refreshed
                    self._brush_source_tile_cache.clear()
                self.target_view.set_selection_spec(spec,emit=False)
                self._selection_changed(box)
                self.selection_status.setText(self.selection_status.text()+f" · 已叠加 {self._applied} 步")
                if key=="region_ocr": self.ocr_status.setText("当前 OCR 动作已提交；选区保持，可继续处理")
                self._trace_region("region_apply_succeeded",{"tool":key,"target_bbox":box,"applied_count":self._applied})
            except Exception as exc:
                self._trace_region("region_apply_failed",{"tool":key,"target_bbox":box,"reason":str(exc)})
                QMessageBox.critical(self,"区域工具应用失败",str(exc))

        def failed(message):
            detail=str(message or "区域处理失败").strip()
            self._trace_region("region_apply_failed",{"tool":key,"target_bbox":box,"reason":detail})
            QMessageBox.critical(self,"区域工具应用失败",detail)

        def finished():
            self._commit_worker=None; self._link_state.applying=False; self.apply_btn.setText(old_text); self._update_region_action_state()

        worker.done.connect(done); worker.failed.connect(failed); worker.finished.connect(finished); worker.finished.connect(worker.deleteLater)
        worker.start()

    def applied_count(self)->int: return int(self._applied)


class ManualEffectDialog(QDialog):
    """Human recovery editor for detector-missed bubbles/open/SFX text."""

    def __init__(self, source_path: str | Path, target_path: str | Path, project: dict[str, Any], parent=None, *, initial_bbox: list[int] | None = None, initial_mode: str | None = None, commit_handler=None, trace_handler=None, config=None, tool_kind: str = "manual_effect", owner_transfer_mode: str = "", ops_module=None):
        super().__init__(parent)
        self._tool_kind = str(tool_kind or "manual_effect")
        self._owner_transfer_mode = str(owner_transfer_mode or "").strip().lower()
        self._ops = ops_module or _manual_effect_ops_for_mode(self._owner_transfer_mode)
        self.setWindowTitle("开放文字框选" if self._tool_kind == "open_text_box" else "人工补漏 / 开放式效果字")
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
        if self._tool_kind == "open_text_box":
            hint = QLabel("在右侧高清日文图上只框住开放式文字。框选只是搜索范围，不会整块贴图：程序使用当前模式自己的精准蒙版引擎，只清 TARGET 日文字形，再迁移 SOURCE 中文原始笔画；人物、网点、背景和框线保持 TARGET。此工具不调用 OCR。")
        else:
            hint = QLabel("在右侧高清日文图上拖框。这个框不依赖 OCR、气泡检测器或自动候选；程序会利用已保存的页面配准，从旧中文版提取对应区域。开放式效果字模式只迁移 SOURCE 支持的中文笔画，并自动估计/清除 TARGET 的日文笔画，尽量保留紫色、网点和人物背景。")
        hint.setObjectName("hint"); hint.setWordWrap(True); root.addWidget(hint)
        split = QSplitter(Qt.Orientation.Horizontal)
        left = QFrame(); left.setObjectName("card"); ll=QVBoxLayout(left); ll.setContentsMargins(8,8,8,8); ll.addWidget(QLabel("旧版中文 · 自动映射参考")); self.source_view=RegionSelectView(source_path, editable=False, parent=self); ll.addWidget(self.source_view,1)
        right = QFrame(); right.setObjectName("card"); rl=QVBoxLayout(right); rl.setContentsMargins(8,8,8,8); rl.addWidget(QLabel("高清日文 · 在这里框选遗漏区域")); self.target_view=RegionSelectView(target_path, editable=True, parent=self); rl.addWidget(self.target_view,1)
        split.addWidget(left); split.addWidget(right); split.setSizes([390,980]); split.setStretchFactor(0,0); split.setStretchFactor(1,1); split.setChildrenCollapsible(False); root.addWidget(split,1)

        form=QFormLayout(); form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.mode=QComboBox()
        if self._tool_kind == "open_text_box":
            mode_label = "开放文字框选 · 精准蒙版原字迁移（不 OCR）" if self._owner_transfer_mode == "mask_replace" else "开放文字框选 · Hybrid 蒙版原字迁移（不 OCR）"
            self.mode.addItem(mode_label, "open_text_box")
            self.mode.setEnabled(False)
        else:
            self.mode.addItem("彩色开放式文字 · 擦除显字（只改文字）","reveal_text"); self.mode.addItem("彩色开放式文字 · 自动迁移（只改文字）","effect_text"); self.mode.addItem("白色气泡 · 文字迁移 + X/Y 微调（不贴背景）","white_bubble_text")
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
        self.mode_guide=QLabel("开放文字框选：只框文字本身，尽量不要碰相邻气泡边框、人物轮廓或分格线。保存后属于当前精准蒙版模式，不会创建 OCR 文本块。" if self._tool_kind == "open_text_box" else "彩色/紫色/人物背景上的文字请选择“擦除显字”；普通白色对白气泡请选择“白色气泡 · 文字迁移”。选框只是搜索范围，不会作为整块写入范围。")
        self.mode_guide.setObjectName("hint"); self.mode_guide.setWordWrap(True); root.addWidget(self.mode_guide)
        action_bar=QFrame(); action_bar.setObjectName("editorActionBar"); action_grid=QGridLayout(action_bar); action_grid.setContentsMargins(0,0,0,0); action_grid.setHorizontalSpacing(6); action_grid.setVerticalSpacing(5)
        fit=QPushButton("适合窗口"); reset=QPushButton("清除框选"); self.preview_mask_btn=QPushButton("预览实际文字 Mask"); self.preview_mask_btn.setObjectName("softPrimary"); save=QPushButton("应用此人工区域"); save.setObjectName("primary"); cancel=QPushButton("取消")
        for col,button in enumerate((fit,reset,self.preview_mask_btn)):
            button.setMinimumWidth(0); action_grid.addWidget(button,0,col)
        action_grid.setColumnStretch(3,1); action_grid.addWidget(save,1,2); action_grid.addWidget(cancel,1,3); root.addWidget(action_bar,0)
        self.target_view.selection_changed.connect(self._selection_changed)
        fit.clicked.connect(self._fit); reset.clicked.connect(lambda: self.target_view.set_box([], emit=True)); self.preview_mask_btn.clicked.connect(self._preview_effective_masks); save.clicked.connect(self._accept_checked); cancel.clicked.connect(self.reject)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.mode.activated.connect(self._manual_mode_activated)
        if self._tool_kind == "open_text_box":
            self._initial_mode = "open_text_box"
            self._mode_locked_by_user = True
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
        effect = (self.mode.currentData() in ("effect_text", "reveal_text", "open_text_box"))
        for w in (self.diff,self.expand,self.feather,self.auto_clear): w.setEnabled(effect)
        if hasattr(self, "preview_mask_btn"):
            previewable = self.mode.currentData() in {"white_bubble_text", "open_text_box"}
            self.preview_mask_btn.setEnabled(previewable)
            if self.mode.currentData() == "open_text_box":
                self.preview_mask_btn.setToolTip("显示开放文字框内真正会写入的 SOURCE 中文 Mask 与会清除的 TARGET 日文 Mask")
            else:
                self.preview_mask_btn.setToolTip("白气泡模式下显示经过边框剥离后的真实 SOURCE 写入 / TARGET 清除 Mask" if previewable else "请先选择可预览的文字迁移模式")

    def _selection_changed(self, bbox):
        box = list(bbox or [])
        if len(box) != 4:
            self.source_view.set_box([]); self.info.setText("尚未框选区域"); return
        src_box = self._ops.map_target_bbox_to_source(self.project, box); self.source_view.set_box(src_box)
        if self._tool_kind == "open_text_box":
            recommendation = "开放文字框选：将只迁移 SOURCE 中文笔画，不 OCR"
            suggested_mode = "open_text_box"
        else:
            recommendation = "彩色开放式文字建议使用“擦除显字”"
            suggested_mode = "reveal_text"
        if self._target_cv is not None and self._tool_kind != "open_text_box":
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
        if self.mode.currentData() not in {"white_bubble_text", "open_text_box"}:
            QMessageBox.information(self, "当前模式不可预览", "请使用“开放文字框选”或“白色气泡 · 文字迁移”模式。")
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
            if self.mode.currentData() == "open_text_box":
                preview = self._ops.render_open_text_box(source, target, self.project, self._row_payload(), getattr(self, "_config", None))
                source_mask = np.asarray(preview.get("source_mask"), np.uint8)
                clear_mask = np.asarray(preview.get("target_clear_mask"), np.uint8)
                diagnostics = dict(preview.get("diagnostics") or {})
            else:
                masks = self._ops.build_manual_effect_masks(source, target, self.project, self._row_payload(), getattr(self, "_config", None))
                source_mask, clear_mask, diagnostics = masks.source_mask, masks.target_clear_mask, masks.diagnostics
            dlg = ManualTextMaskPreviewDialog(self.target_path, source_mask, clear_mask, diagnostics, self)
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
            "tool_kind": self._tool_kind,
            "owner_transfer_mode": self._owner_transfer_mode,
            "ocr_allowed": False if self._tool_kind == "open_text_box" else None,
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
                masks=self._ops.build_manual_effect_masks(source,target,self.project,self._row_payload(),getattr(self, "_config", None))
                seed=self._ops.build_reveal_seed_mask(masks.source_mask,masks.target_clear_mask,padding_px=max(4,int(self.expand.value())+2))
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
        self.blur_guard.setToolTip("普通清晰旧中文版锁定 SOURCE 原始字形与抗锯齿；只有摄影/反光来源才允许进入照片专用清晰化。精准蒙版完全不调用 OCR。")
        self.preserve_source_layout=QCheckBox("清晰旧中文版保留原字号/分列（推荐）"); self.preserve_source_layout.setChecked(True)
        self.preserve_source_layout.setToolTip("精准蒙版始终保留旧中文版真实字号/分列/符号，并且完全不调用 OCR。需要 OCR 识别或重新排字时请改用精准蒙版+OCR / OCR重排，或手动编辑。")
        mode.layout.addWidget(self.publication_safety); mode.layout.addWidget(self.paired); mode.layout.addWidget(self.skip_ocr); mode.layout.addWidget(self.pixel_exact); mode.layout.addWidget(self.full_bubble_patch); mode.layout.addWidget(self.preserve_border); mode.layout.addWidget(self.blur_guard); mode.layout.addWidget(self.preserve_source_layout)
        photo_note=QLabel("当前策略：所有自动路径都以 TARGET 为唯一背景；白底和彩底都只清除日文文字并迁移 SOURCE 中文字形。SOURCE 的白纸、灰阶、肤色或旧背景 RGB 永远不会写进彩图。")
        photo_note.setObjectName("quiet"); photo_note.setWordWrap(True); mode.layout.addWidget(photo_note)
        sl.addWidget(mode)

        align=Card("局部对齐与清晰度")
        form=QFormLayout(); form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.local=QComboBox(); self.local.addItems(["ecc","bbox","global"]); self.sr=QComboBox(); self.sr.addItems(["auto","torch","lanczos","external","off"])
        self.fidelity=QComboBox(); self.fidelity.addItem("自动：清晰扫描保留原字；摄影来源再清晰化","auto"); self.fidelity.addItem("只保留原像素","pixels"); self.fidelity.addItem("强制墨迹重建（会改变原字，仅特殊情况）","ink"); self.fidelity.addItem("低清直接拒绝","reject")
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
        self.target_layer_restore=QPushButton("恢复 TARGET 日文层 / 擦蒙版…"); self.target_layer_restore.setObjectName("softPrimary"); self.reset_target_layer_restore=QPushButton("清空 TARGET 恢复")
        target_grid.addWidget(self.target_layer_erase,0,0); target_grid.addWidget(self.reset_target_layer_erase,0,1)
        target_grid.addWidget(self.target_layer_restore,1,0); target_grid.addWidget(self.reset_target_layer_restore,1,1)
        stages.layout.addLayout(target_grid)
        self.target_layer_erase.setToolTip("最终收尾画笔：可刷日文残字、黑点、短线和符号。只重建 TARGET 母版层，中文迁移/重排/人工补漏图层自动保护。")
        self.target_layer_restore.setToolTip("反向画笔：对不应显示中文的区域，直接恢复 TARGET 原始日文图层与背景。")
        stage_note=QLabel("自动没有识别到、识别不完整，或者自动蒙版本身过大/错误时，使用“人工强制迁移蒙版”：红色人工层与蓝色 OCR/自动层都可直接编辑。自动区域会先收紧为真实文字像素，避免整块气泡/画面被当成清除区；只剩零星残字时再用“仅擦 TARGET 日文层”。")
        stage_note.setObjectName("quiet"); stage_note.setWordWrap(True); stages.layout.addWidget(stage_note); sl.addWidget(stages)

        recovery=Card("区域复合处理 / 人工补漏", "整页模式只负责第一遍自动处理；这里可以在同一选区连续叠加 Direct、精准蒙版、挖孔揭示、透明文字与 OCR，而不切换或重跑整页模式。")
        self.region_composite_hint=QLabel("推荐：先运行一个最合适的整页模式，再打开“区域复合工具”。选区支持矩形、椭圆、爆炸框智能闭合和自由闭合；每个区域动作独立保存，选区外禁止写入。")
        self.region_composite_hint.setObjectName("hint"); self.region_composite_hint.setWordWrap(True); recovery.layout.addWidget(self.region_composite_hint)
        self.region_composite=QPushButton("区域复合工具…"); self.region_composite.setObjectName("primary"); self.region_composite.setToolTip("一个选区内连续调用 Direct / 精准蒙版 / 挖孔 / 透明文字 / OCR。处理后保留选区，便于继续叠加下一种工具。")
        recovery.layout.addWidget(self.region_composite)
        self.expand_direct_range=QCheckBox("扩大 Direct 候选范围（难页）"); self.expand_direct_range.setChecked(bool(getattr(self.window.state.config.direct_patch,"source_direct_expand_candidate_range",False))); self.expand_direct_range.setToolTip("可选恢复模式：允许更小/更细长/弱文字种子的 Direct 候选进入检查。默认关闭；同页配准与 TARGET 背景保护仍是硬条件。")
        recovery.layout.addWidget(self.expand_direct_range)
        self.open_text_box_hint=QLabel("精准蒙版专用：自动检测漏掉开放式文字时，直接框住文字。只迁移 SOURCE 中文原字形，不 OCR、不整块贴背景。")
        self.open_text_box_hint.setObjectName("hint"); self.open_text_box_hint.setWordWrap(True); recovery.layout.addWidget(self.open_text_box_hint)
        self.open_text_box=QPushButton("开放文字框选…"); self.open_text_box.setObjectName("primary"); self.open_text_box.setToolTip("仅用于精准蒙版 / 精准蒙版+OCR。手动画框后由当前模式自己的蒙版引擎清除 TARGET 日文并迁移 SOURCE 中文；Hybrid 不会因此触发 OCR。")
        recovery.layout.addWidget(self.open_text_box)
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

        manual=Card("OCR 文本编辑 / 排版", "人工 OCR 是独立的页面复核叠加层：Direct、精准蒙版、两种 Reveal、精准蒙版+OCR 和 OCR重排都可框选 ROI 重新 OCR，并编辑文字、字体、字号、方向、断句与排版；不会改变当前整页自动模式。")
        self.ocr_block_status=QLabel("人工 OCR 文本块：等待可编辑页面"); self.ocr_block_status.setObjectName("quiet"); self.ocr_block_status.setWordWrap(True); manual.layout.addWidget(self.ocr_block_status)
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
        self.region_composite.clicked.connect(self._open_region_composite_editor)
        self.open_text_box.clicked.connect(self._add_open_text_box_region)
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
        """Disable mutating review controls while any page writer is active.

        Repeated ``busy=True`` notifications are intentionally idempotent.  Model
        preparation can hand off directly to a pipeline worker; older builds took
        a second snapshot after controls were already disabled and later restored
        that all-False snapshot, leaving review buttons permanently unclickable.
        """
        busy=bool(busy)
        previous=bool(getattr(self,"_processing_busy",False))
        if busy == previous:
            if hasattr(self,"activity_badge"):
                self.activity_badge.setText("处理中…" if busy else "就绪")
            return
        self._processing_busy=busy
        if hasattr(self, "activity_badge"):
            self.activity_badge.setText("处理中…" if busy else "就绪")
            self.activity_badge.setProperty("busy", bool(busy))
            self.activity_badge.style().unpolish(self.activity_badge); self.activity_badge.style().polish(self.activity_badge)
        names = (
            "run", "reprocess_current", "edit_clear_mask", "remove_text_only", "apply_mask_review",
            "reset_clear_mask", "force_transfer_mask", "reset_force_transfer_mask",
            "target_layer_erase", "reset_target_layer_erase", "target_layer_restore",
            "reset_target_layer_restore", "region_composite", "open_text_box", "add_manual_effect", "add_manual_effect_candidate",
            "undo_manual_effect", "manual_apply", "manual_reset", "manual_undo", "manual_redo",
            "candidate_accept", "candidate_restore", "open_ocr_block_editor", "reset_ocr_blocks",
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
            # Snapshot restore is only a transient bridge. Recompute every review
            # control from persisted page artifacts after the worker has fully
            # released, so a stale False state can never survive page/mode/OCR
            # transitions. The single-shot runs after MainWindow clears its worker.
            QTimer.singleShot(0, self.refresh)

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
        # Horizontal mode needs page rail + usable canvas + the 430px review
        # inspector.  The old 900px threshold left a 900-1140px dead zone where
        # Qt honoured those minimums by pushing the inspector off-screen.
        compact = self.width() < 1180
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
            w=max(1180,self.width())
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
        target=cv2.imread(str(page_dir/"target_original.png"),cv2.IMREAD_COLOR)
        if target is None:
            QMessageBox.warning(self,"无法读取","当前页 target_original.png 无法读取。"); return
        # Use the same review-safe projection as “应用蒙版”.  Automatic renderer
        # artifacts may be whole-container masks and can overlap a face/artwork;
        # manual masks remain exact reviewer authority and are never projected.
        from .review_apply import _load_effective_clear_mask
        mask,_mask_source=_load_effective_clear_mask(page_dir,target.shape[:2])
        if mask is None or mask.shape != target.shape[:2]: mask=np.zeros(target.shape[:2],dtype=np.uint8)
        dlg=MaskEditorDialog(page_dir/"target_original.png",mask,self.window)
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
        mask_before=mask.copy()
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
            display_path, mask, self.window,
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
        edited_manual=dlg.result_mask(); edited_auto=dlg.result_reference_mask()
        manual_changed=not np.array_equal(edited_manual,mask_before)
        reference_changed=not np.array_equal(edited_auto,auto_original)
        write_image(mask_path,edited_manual)
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
        self.current_view="result"
        for b,k in self.view_buttons: b.setChecked(k=="result")
        if reference_changed:
            # Editing the blue automatic authority can invalidate the automatic
            # transfer itself, so keep the conservative full page rerun here.
            self.window.statusBar().showMessage("自动/OCR蒙版已修改，正在重新处理当前页并恢复人工复核…",7000)
            self.refresh(); self.window.run_current_page(reapply_review_after_process=True)
            return
        # Red manual-force paint is a review-layer operation. Replaying review
        # artifacts is enough and is dramatically faster than rerunning detection,
        # registration and the page renderer.
        cfg=self.window.state.config.model_copy(deep=True)
        def done(final):
            final=Path(str(final)); self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.window.statusBar().showMessage("人工强制迁移蒙版已应用（快速复核路径）。",5000); self.refresh()
        if manual_changed:
            self.window.run_page_action("应用人工强制迁移蒙版",lambda: apply_review_page(page_dir,cfg),done,failure_title="应用人工强制迁移蒙版失败")
        else:
            self.window.statusBar().showMessage("蒙版没有变化。",2500); self.refresh()

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
        # Chinese protection is page-stable during one editor session; building it
        # on every brush event repeatedly decoded several alpha/mask files.
        from .review_apply import _protected_chinese_mask
        protect,_protect_diag=_protected_chinese_mask(page_dir,target.shape[:2],margin_px=1)
        preview_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        def _target_erase_live_preview(mask_value):
            raw=(np.asarray(mask_value,dtype=np.uint8)>0).astype(np.uint8)*255
            if cv2.countNonZero(raw)==0: return display.copy()
            expanded=cv2.dilate(raw,preview_kernel,iterations=1)
            effective=expanded.copy(); effective[protect>0]=0
            out=display.copy()
            if cv2.countNonZero(effective)==0: return out
            if selected_fill_mode == "pure_white":
                out[effective>0]=255; return out
            # Live preview only needs the local brush neighbourhood. Inpaint a
            # padded ROI instead of the whole manga page; final save still uses
            # the authoritative full review service in a worker thread.
            pts=cv2.findNonZero(effective)
            if pts is None: return out
            x,y,w,h=cv2.boundingRect(pts); pad=12
            x0=max(0,x-pad); y0=max(0,y-pad); x1=min(target.shape[1],x+w+pad); y1=min(target.shape[0],y+h+pad)
            crop_target=target[y0:y1,x0:x1]
            crop_mask=effective[y0:y1,x0:x1]
            cleaned=cv2.inpaint(crop_target,crop_mask,3.0,cv2.INPAINT_TELEA)
            local=crop_mask>0
            out_crop=out[y0:y1,x0:x1]; out_crop[local]=cleaned[local]
            return out
        dlg = MaskEditorDialog(
            display_path, mask, self.window,
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
        cfg=self.window.state.config.model_copy(deep=True)
        def done(final):
            final=Path(str(final)); self.window.state.last_result_path=str(final)
            pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""
            proj=self.window.state.projects_by_page.get(page_id)
            if proj is not None: proj.artifacts["final"]=str(final)
            self._sync_reviewed_book_final(final); self.current_view="target_erase"
            for b,k in self.view_buttons: b.setChecked(k==self.current_view)
            self.window.statusBar().showMessage("TARGET 层擦除已应用：只改日文母版层，中文图层已硬保护。",6000)
            self.refresh()
        self.window.run_page_action(
            "TARGET 层擦除", lambda: apply_target_layer_erase_review(page_dir,cfg), done,
            failure_title="TARGET 层擦除失败",
        )

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
            display_path, mask, self.window,
            title="恢复 TARGET 日文层",
            hint_text="红色 = 恢复范围。该工具会把笔刷区域直接恢复成 TARGET 原始日文图层与背景；同时会把恢复区域反向写回相关清除/强制蒙版，用于真正擦去蒙版、恢复日文层。左键涂抹，右键或“消除蒙版”可反擦。",
            save_label="保存并应用 TARGET 恢复", preview_fn=_target_restore_live_preview,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        write_image(mask_path, dlg.result_mask())
        save_json(page_dir / "target_layer_restore_settings.json", {
            "schema": "manga_hd_translation_transfer.target_layer_restore_settings.v1",
            "dilate_px": 0,
        })
        def done(final):
            final=Path(str(final)); self.window.state.last_result_path=str(final)
            pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""
            proj=self.window.state.projects_by_page.get(page_id)
            if proj is not None: proj.artifacts["final"]=str(final)
            self._sync_reviewed_book_final(final); self.current_view="target_restore"
            for b,k in self.view_buttons: b.setChecked(k==self.current_view)
            self.window.statusBar().showMessage("TARGET 层恢复已应用：笔刷区域已恢复原始日文图层，并同步从相关蒙版中扣除。",6000)
            self.refresh()
        self.window.run_page_action(
            "TARGET 层恢复", lambda: apply_target_layer_restore_review(page_dir), done,
            failure_title="TARGET 层恢复失败",
        )


    def _reset_target_layer_erase(self):
        page_dir = self._current_page_dir()
        if page_dir is None:
            return
        def done(final):
            if final is not None:
                final=Path(str(final)); self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage("已清空 TARGET 层擦除，并恢复擦除前结果。",5000); self.refresh()
        self.window.run_page_action(
            "清空 TARGET 层擦除", lambda: reset_target_layer_erase_review(page_dir), done,
            failure_title="恢复 TARGET 层失败",
        )


    def _reset_target_layer_restore(self):
        page_dir = self._current_page_dir()
        if page_dir is None:
            return
        def done(final):
            if final is not None:
                final=Path(str(final)); self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage("已清空 TARGET 层恢复，并恢复恢复前结果。",5000); self.refresh()
        self.window.run_page_action(
            "清空 TARGET 层恢复", lambda: reset_target_layer_restore_review(page_dir), done,
            failure_title="恢复 TARGET 层失败",
        )


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
                atomic_copy_file(src,dst); final=dst
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

    def _commit_manual_effect_core(self, page_dir: Path, row: dict[str, Any], reveal, reveal_patch, preset_candidate: dict[str, Any] | None = None):
        """Qt-free portion of a manual/region commit; safe for PageActionWorker."""
        def trace(stage: str, payload: dict[str, Any]):
            self._trace_manual_gui_flow(page_dir, stage, payload)
        return commit_manual_effect(
            page_dir, row, reveal, reveal_patch,
            self.window.state.config.model_copy(deep=True),
            preset_candidate=preset_candidate,
            trace=trace,
        )

    def _finalize_manual_effect_commit(self, page_dir: Path, row: dict[str, Any], result, *, refresh_ui: bool = True) -> Path:
        """GUI-thread bookkeeping after the core transaction has completed."""
        self.window.state.last_result_path = str(result.final_reviewed)
        remembered_mode = str(row.get("mode", "") or "")
        if remembered_mode and not remembered_mode.startswith("region_"):
            self.window.state.last_manual_effect_mode = remembered_mode
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
        if refresh_ui:
            self.current_view = "result"
            for b,k in self.view_buttons:
                b.setChecked(k == "result")
        self.window.statusBar().showMessage(
            f"人工补漏已直接提交 · 本页共 {result.region_count} 个区域 · final 已同步", 6000
        )
        # RegionCompositeDialog updates its own image immediately. Rebuilding the
        # whole ProjectPage after every local region would decode thumbnails,
        # recalculate controls and visibly hitch the modal editor. Defer that
        # outer-page refresh until the region dialog closes.
        if refresh_ui:
            self.refresh()
        return result.final_reviewed

    def _commit_manual_effect_dialog_result(self, page_dir: Path, row: dict[str, Any], reveal, reveal_patch, preset_candidate: dict[str, Any] | None = None) -> Path:
        """Synchronous adapter retained for legacy/manual dialogs."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result=self._commit_manual_effect_core(page_dir,row,reveal,reveal_patch,preset_candidate)
        finally:
            QApplication.restoreOverrideCursor()
        return self._finalize_manual_effect_commit(page_dir,row,result)

    def _open_region_composite_editor(self):
        page_dir=self._current_page_dir(); ws=self._workspace()
        if page_dir is None or ws is None or not ws.project_path or not ws.project_path.exists():
            QMessageBox.information(self,"尚未处理","请先用任意整页模式处理当前页，再进入区域复合工具。")
            return
        source_path=page_dir/"source_original.png"; target_path=page_dir/"target_original.png"
        if not source_path.exists() or not target_path.exists():
            QMessageBox.warning(self,"缺少页面缓存","当前页缺少 SOURCE / TARGET 原图缓存，请重新处理当前页一次。")
            return
        display_path=page_dir/"final_reviewed.png"
        if not display_path.exists(): display_path=page_dir/"final.png"
        if not display_path.exists(): display_path=target_path
        try:
            project=normalize_project(load_json(ws.project_path))
        except Exception as exc:
            QMessageBox.critical(self,"无法读取页面项目",str(exc)); return
        self._start_manual_gui_flow(page_dir,{"origin":"region_composite"})
        self._trace_manual_gui_flow(page_dir,"region_composite_opened")
        def _commit(row,reveal,reveal_patch):
            self._trace_manual_gui_flow(page_dir,"region_action_commit",{"mode":str(row.get("mode") or ""),"target_bbox":list(row.get("target_bbox") or [])})
            return self._commit_manual_effect_core(page_dir,row,reveal,reveal_patch,{})
        def _finalize(row,result):
            return self._finalize_manual_effect_commit(page_dir,row,result,refresh_ui=False)
        def _trace(stage,payload):
            self._trace_manual_gui_flow(page_dir,stage,payload)
        try:
            dlg=RegionCompositeDialog(page_dir,source_path,target_path,display_path,project,self.window.state.config,self.window,commit_handler=_commit,commit_finalize_handler=_finalize,trace_handler=_trace)
        except Exception as exc:
            self._trace_manual_gui_flow(page_dir,"region_composite_failed",{"reason":str(exc)})
            QMessageBox.critical(self,"无法打开区域复合工具",str(exc)); return
        dlg.exec()
        self._trace_manual_gui_flow(page_dir,"region_composite_closed",{"applied_count":dlg.applied_count()})
        try: self.image.clear_cache()
        except Exception: pass
        self.current_view="result"
        for b,k in self.view_buttons: b.setChecked(k=="result")
        self.refresh()

    def _add_open_text_box_region(self):
        page_dir=self._current_page_dir(); ws=self._workspace()
        current_mode=str(((ws.meta or {}).get("transfer_mode") if ws is not None else "") or self.window.state.config.transfer.mode or "").strip().lower()
        if current_mode not in {"mask_replace", "hybrid"}:
            QMessageBox.information(self, "当前模式不可用", "“开放文字框选”只属于精准蒙版 / 精准蒙版+OCR。Direct 和整页挖孔保持各自的安全合同。")
            return
        self._add_manual_effect_region(
            None,
            forced_tool_kind="open_text_box",
            forced_mode="open_text_box",
            owner_transfer_mode=current_mode,
        )

    def _add_manual_effect_region(self, preset_candidate: dict[str, Any] | None = None, *, forced_tool_kind: str = "manual_effect", forced_mode: str | None = None, owner_transfer_mode: str | None = None):
        page_dir=self._current_page_dir(); ws=self._workspace()
        if page_dir is None or ws is None or not ws.project_path or not ws.project_path.exists():
            QMessageBox.information(self,"尚未处理","请先处理当前页。人工补漏复用本页已经保存的配准，不会重新跑 OCR。")
            return
        source_path=page_dir/"source_original.png"; target_path=page_dir/"target_original.png"
        if not source_path.exists() or not target_path.exists():
            QMessageBox.warning(self,"缺少页面缓存","当前页缺少 source_original.png 或 target_original.png，请重新处理当前页一次。")
            return
        project=normalize_project(load_json(ws.project_path))
        project_mode=str(as_dict(project.get("meta")).get("transfer_mode", "") or owner_transfer_mode or self.window.state.config.transfer.mode or "").strip().lower()
        manual_ops=_manual_effect_ops_for_mode(project_mode)
        preset_candidate = as_dict(preset_candidate)
        initial_bbox = as_list(preset_candidate.get("target_bbox"))
        if forced_mode:
            initial_mode = str(forced_mode)
        elif preset_candidate:
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
            dlg=ManualEffectDialog(
                source_path,target_path,project,self.window,
                initial_bbox=initial_bbox,initial_mode=initial_mode,commit_handler=_commit,trace_handler=_trace,
                config=self.window.state.config, tool_kind=str(forced_tool_kind or "manual_effect"),
                owner_transfer_mode=str(owner_transfer_mode or project_mode), ops_module=manual_ops,
            )
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
        reveal_files=[]
        for key in ("reveal_mask_file","reveal_patch_file"):
            name=str(removed.get(key,"") or "").strip()
            if name: reveal_files.append(name)
        overrides["manual_effect_regions"]=rows; overrides["status"]="reviewed_with_manual_effect" if rows else "reviewed"
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            run_manual_review_transaction(page_dir,lambda: self._apply_manual_effect_overrides(page_dir,overrides))
            # Delete no-longer-referenced sparse patch files only after the page
            # state has been successfully recomposed and published.
            for name in reveal_files:
                try: (page_dir/name).unlink(missing_ok=True)
                except OSError: logger.warning("unable to remove obsolete manual patch %s",page_dir/name)
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
            QMessageBox.information(self.window,"当前页面不可编辑","人工 OCR 需要当前页已有 project.json、SOURCE 和 TARGET。处理完成后的 Direct、精准蒙版、两种 Reveal、精准蒙版+OCR 与 OCR重排页面均可使用。")
            return
        ws,project,mode,source_path,target_path=ctx
        dialog=OCRBlockEditorDialog(ws.page_root,source_path,target_path,project,self.window.state.config,mode,parent=self.window)
        if dialog.exec()!=QDialog.DialogCode.Accepted:
            return
        cfg=self.window.state.config.model_copy(deep=True)
        def done(final):
            final=Path(str(final)); self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage("人工 OCR 文本块已保存并局部重绘。",4500); self.refresh()
        # Rebuild/replay may read several page artifacts and render text. Keep it
        # off the GUI thread so closing the OCR editor never freezes the workbench.
        self.window.run_page_action(
            "应用人工 OCR", lambda: apply_review_page(ws.page_root,cfg), done,
            failure_title="应用人工 OCR 失败",
        )

    def _reset_ocr_blocks(self):
        ctx=self._ocr_editor_context()
        if ctx is None:
            return
        ws,project,mode,_source,_target=ctx
        rows=load_ocr_blocks(ws.page_root,mode)
        if not rows:
            self.window.statusBar().showMessage("当前页没有人工 OCR 文本块。",2500); return
        if not _confirm_destructive_action(
            self.window,
            "清空人工 OCR",
            f"将清空当前页 {len(rows)} 个人工 OCR 文本块，并恢复不含人工 OCR 的复核结果。\n\n这不会修改当前整页自动模式，也不会删除 Direct / 精准蒙版 / Reveal 等自动产物。",
            confirm_text="清空",
        ):
            return
        cfg=self.window.state.config.model_copy(deep=True)
        logger.info("manual OCR clear requested mode=%s page=%s blocks=%d", mode, ws.page_root, len(rows))
        def action():
            return clear_ocr_review_blocks(ws.page_root,cfg)
        def done(final):
            final=Path(str(final)); self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            logger.info("manual OCR clear completed mode=%s page=%s", mode, ws.page_root); self.window.statusBar().showMessage("已清空人工 OCR 文本块，并恢复无 OCR 复核结果。",4500); self.refresh()
        self.window.run_page_action(
            "清空人工 OCR", action, done, failure_title="清空人工 OCR 失败",
        )

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
            step=apply_review_history_step(page_dir,direction,self.window.state.config.model_copy(deep=True))
            if step is None:
                self.window.statusBar().showMessage("没有可用的编辑历史。",2500); return
            final=step.final_reviewed
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
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            tx=apply_review_overrides_transaction(page_dir,overrides,self.window.state.config.model_copy(deep=True),history_reason="reset_manual_reletter")
            final=tx.final_reviewed
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
            atomic_copy_file(final_path,dst)
        except OSError as exc:
            logger.warning("reviewed book-final sync failed: %s",exc)

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
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            tx=apply_review_overrides_transaction(page_dir,overrides,self.window.state.config.model_copy(deep=True),history_reason="apply_manual_reletter")
            final=tx.final_reviewed
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
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            tx=apply_review_overrides_transaction(page_dir,overrides,self.window.state.config.model_copy(deep=True),history_reason=f"candidate_{action}")
            final=tx.final_reviewed; self.window.state.last_result_path=str(final)
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
            self.region_composite.setEnabled(False)
            self.open_text_box.setEnabled(False); self.open_text_box.setVisible(False); self.open_text_box_hint.setVisible(False)
            self.add_manual_effect.setEnabled(False); self.add_manual_effect_candidate.setEnabled(False); self.undo_manual_effect.setEnabled(False)
            if hasattr(self, "open_ocr_block_editor"):
                self.open_ocr_block_editor.setEnabled(False); self.reset_ocr_blocks.setEnabled(False)
                self.ocr_block_status.setText("人工 OCR 文本块：等待可编辑页面")
            return
        idx=max(0,min(s.selected_index,total-1)); pair=s.pairs[idx]
        self.page_rail.set_pages(s.pairs, idx)
        ws=self._workspace()
        self.page_counter.setText(f"{idx+1} / {total}")
        self.prev_page.setEnabled(idx>0); self.next_page.setEnabled(idx<total-1)
        page_root=ws.page_root if ws else None
        current_mode=str(((ws.meta or {}).get("transfer_mode") if ws is not None else "") or self.window.state.config.transfer.mode or "").strip().lower()
        open_text_box_available = current_mode in {"mask_replace", "hybrid"}
        self.open_text_box.setVisible(open_text_box_available)
        self.open_text_box_hint.setVisible(open_text_box_available)
        if open_text_box_available:
            mode_label = "精准蒙版" if current_mode == "mask_replace" else "精准蒙版+OCR"
            self.open_text_box_hint.setText(f"{mode_label}：自动检测漏掉开放式文字时，直接框住文字。只迁移 SOURCE 中文原字形，不 OCR、不整块贴背景。")
        action_state=review_action_availability(page_root)
        globally_busy=bool(getattr(self.window,"_busy_running",lambda:False)())
        if not globally_busy:
            # Recompute from persistent page artifacts instead of trusting a
            # possibly stale enabled-state snapshot from a previous worker.
            for name in (
                "edit_clear_mask","remove_text_only","apply_mask_review","force_transfer_mask",
                "target_layer_erase","target_layer_restore","reset_clear_mask","reset_force_transfer_mask",
                "reset_target_layer_erase","reset_target_layer_restore",
            ):
                widget=getattr(self,name,None)
                if widget is not None:
                    widget.setEnabled(bool(action_state.get(name,False)))

        # v2.3.74: the button and click handler must derive capability from the
        # exact same persisted project context.  Earlier refresh() used ws.meta
        # while _open_ocr_block_editor() re-read project.json; after mode changes
        # those two answers could diverge and the editor repeatedly became stale
        # disabled/enabled.  The context is now the single authority.
        ocr_ctx=self._ocr_editor_context()
        ocr_editor_enabled=bool(ocr_ctx is not None and not globally_busy)
        ocr_runtime_mode=str(ocr_ctx[2]) if ocr_ctx is not None else current_mode
        if hasattr(self,"open_ocr_block_editor"):
            self.open_ocr_block_editor.setEnabled(ocr_editor_enabled)
            rows=load_ocr_blocks(page_root,ocr_runtime_mode) if ocr_ctx is not None else []
            self.reset_ocr_blocks.setEnabled(bool(rows) and not globally_busy)
            if ocr_ctx is not None:
                scope = ocr_edit_scope(ocr_runtime_mode)
                scope_label = {
                    "direct_patch": "Direct · 人工 OCR",
                    "mask_replace": "精准蒙版 · 人工 OCR",
                    "aligned_overlay_reveal": "整页对齐挖孔 · 人工 OCR",
                    "transparent_bubble_reveal": "整页对齐透明 · 人工 OCR",
                    "hybrid": "精准蒙版+OCR",
                    "reletter": "OCR重排",
                }.get(ocr_runtime_mode, "人工 OCR")
                if scope=="mask_ocr":
                    if ocr_runtime_mode == "hybrid":
                        suffix = "人工框选=强制 OCR；自动 OCR 仅处理完全无精准蒙版覆盖的区域"
                    else:
                        suffix = "人工框选=局部强制 OCR；不会把整页模式改成 OCR 模式"
                elif scope=="review_ocr":
                    suffix = "仅作为人工复核叠加层；Reveal 自动 renderer 与原有图层完全不改"
                else:
                    suffix = "仅影响当前 OCR 重排复核层"
                self.ocr_block_status.setText(f"{scope_label} · 人工 OCR 文本块 {len(rows)} 个 · {suffix}")
            else:
                self.ocr_block_status.setText("当前页尚无可用的人工 OCR 编辑上下文；请先完成该页自动处理。")
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
        self.region_composite.setEnabled(bool(can_manual))
        self.open_text_box.setEnabled(bool(can_manual and open_text_box_available))
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
            if ocr_editor_enabled:
                self.manual_status.setText("当前页没有自动待复核文字区域；仍可使用上方“人工 OCR / 编辑文本块…”自行框选并编辑。")
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
        elif self.current_view in {"result","review","mask","clear_mask","chinese_layer","removed","target_erase","target_restore"}:
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
        rail_tools.addWidget(self.stop_button,2,0,1,2)
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
            if hasattr(self, "models") and not self.models.shutdown_write_workers():
                still_running = True
        except Exception:
            logger.debug("model write worker shutdown check failed", exc_info=True)
            still_running = True
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
            # Controller-owned worker references are the lifecycle authority.
            # Treat the tiny created-before-start / finished-before-slot windows
            # as busy too; otherwise conflicting actions can slip in between
            # QThread.start() and isRunning() becoming observable.
            pipeline_running=self.worker is not None,
            prepare_running=self._prepare_worker is not None,
            page_action_running=self._page_action_worker is not None,
            settings_updating=hasattr(self, "settings") and self.settings.is_updating,
            model_write_running=hasattr(self, "models") and self.models.has_write_task_running(),
        )

    def _busy_running(self) -> bool:
        return self._processing_busy_state().busy

    def _refresh_global_stop_visibility(self, state=None) -> None:
        if not hasattr(self, "stop_button") or not hasattr(self, "stack"):
            return
        state = state or self._processing_busy_state()
        # ProjectPage already owns the visible stop button beside its processing
        # actions. If the user changes workflow pages during a cancellable task,
        # surface the rail stop control so cancellation never becomes unreachable.
        visible = bool(state.cancellable and self.stack.currentIndex() != 0)
        self.stop_button.setVisible(visible)
        self.stop_button.setEnabled(bool(state.cancellable))

    def _set_busy(self, active: bool | None = None):
        state = self._processing_busy_state()
        busy = state.busy if active is None else bool(active)
        cancellable = state.cancellable
        self._refresh_global_stop_visibility(state)
        if hasattr(self, "project"):
            self.project.cancel.setEnabled(cancellable)
            self.project.run_page.setEnabled(not busy)
            self.project.run_book.setEnabled(not busy)
            if hasattr(self.project, "continue_book"): self.project.continue_book.setEnabled(not busy)
            self.project.pair_btn.setEnabled(not busy)
            self.project.apply_type.setEnabled(not busy)
            self.project.reset_type.setEnabled(not busy)
            self.project.page_type.setEnabled(not busy)
            if hasattr(self.project, "set_processing_busy"):
                self.project.set_processing_busy(busy)
        if hasattr(self, "models") and hasattr(self.models, "set_processing_busy"):
            self.models.set_processing_busy(busy)
        if hasattr(self, "workbench"):
            self.workbench.set_processing_busy(busy)
        if hasattr(self, "export"):
            self.export.set_processing_busy(busy)
        if hasattr(self, "settings") and hasattr(self.settings, "set_processing_busy"):
            self.settings.set_processing_busy(busy)

    def show_page(self, index: int):
        if not 0 <= int(index) < self.stack.count():
            return
        index = int(index)
        self.stack.setCurrentIndex(index)
        self._refresh_global_stop_visibility()
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
            full_pairs, unmatched_source, unmatched_target, expanded = expand_restored_session_pairs(session, self.state.config.pairing)
            self.state.pairs = full_pairs
            self.state.selected_index = 0
            self.state.projects_by_page.clear(); self.state.batch_status.clear()
            self.state.restored_page_roots = {row.page_id: str(row.page_root) for row in session.pages}
            self.state.restored_page_origin = {row.page_id: "命令行/Codex 已有结果" for row in session.pages}
            self.state.unmatched_source = list(unmatched_source); self.state.unmatched_target = list(unmatched_target)
            self.project._table_signature = None; self.project._thumb_signature = None
            self.load_page_marks()
            warning = f" · 跳过/警告 {len(session.warnings)}" if session.warnings else ""
            if expanded:
                pending = max(0, len(self.state.pairs) - len(session.pages))
                msg = f"已恢复 {len(session.pages)} 页已有结果，并重建整本 {len(self.state.pairs)} 页配对 · 待继续 {pending} 页{warning}"
            else:
                msg = f"已恢复 {len(session.pages)} 页已有结果{warning}。可直接继续页面检查或进入替换工作台人工补漏。"
            self.statusBar().showMessage(msg, 8000)
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
        prep = AutoPrepareModelsWorker(worker.config.model_copy(deep=True))
        self._prepare_worker = prep
        self._set_busy(None)
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
        # Whole-book progress emits once or more per page. Build this once instead
        # of linearly scanning every pair for every progress signal (O(n^2)).
        self._worker_pair_row_by_name = {}
        for row, pair in enumerate(self.state.pairs):
            self._worker_pair_row_by_name.setdefault(Path(pair.target_path).name, row)
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
                logger.exception("page action completion callback failed label=%s", label)
                detail = str(exc).strip() or f"{type(exc).__name__}：完成回调失败，但异常没有文本信息。请查看运行日志。"
                QMessageBox.critical(self, failure_title, detail)

        def failed(message: str):
            detail = str(message).strip() or "后台页面操作失败，但没有返回错误详情。请查看运行日志。"
            logger.error("page action failed label=%s detail=%s", label, detail)
            QMessageBox.critical(self, failure_title, detail)
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
        lookup = getattr(self, "_worker_pair_row_by_name", None)
        if isinstance(lookup, dict) and name in lookup:
            return int(lookup[name])
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
        self._worker_pair_row_by_name = {}
        worker = self.worker
        self.worker = None
        self._set_busy(None)
        if worker is not None:
            worker.deleteLater()

    def _worker_done(self, project, path):
        self.progress.setRange(0,100); self.progress.setValue(100)
        if hasattr(project, "page_id"):
            self.state.last_project = project
            self.state.projects_by_page[str(project.page_id)] = project
            self._merge_project_page_mark(project)
        elif hasattr(project, "pages"):
            self.state.last_project = None
            # Page project.json files are the authoritative persistent store and
            # resolve_page_workspace already reloads them. Do not duplicate an
            # entire processed book in GUI RAM after a long batch finishes.
            self.state.projects_by_page.clear()
            compact_updates = list((getattr(project, "meta", {}) or {}).get("page_management_updates") or [])
            if compact_updates:
                # Long-book streaming results expose compact mark updates so the
                # GUI does not defeat disk-backed pages by deserializing every
                # project.json again immediately after completion.
                for row in compact_updates:
                    key = str((row or {}).get("page_id") or "")
                    pm = (row or {}).get("page_management")
                    if not key or not pm:
                        continue
                    existing = PageMark.from_dict(self.state.page_marks.get(key)) if key in self.state.page_marks else None
                    incoming = PageMark.from_dict(pm)
                    if existing is None or existing.origin != "manual" or incoming.origin == "manual":
                        self.state.page_marks[key] = incoming.to_dict()
            else:
                for page in getattr(project, "pages", []) or []:
                    self._merge_project_page_mark(page)
        else:
            self.state.last_project = None
        self.save_page_marks(); self.project._table_signature = None; self.project._thumb_signature = None
        self.state.last_result_path = path if str(path).lower().endswith(".png") else ""
        meta = dict(getattr(project, "meta", {}) or {}) if hasattr(project, "meta") else {}
        cancelled = bool(meta.get("cancelled"))
        if getattr(self, "_worker_is_single_page", False):
            self.statusBar().showMessage(
                page_completion_message(project, reprocessed=bool(getattr(self, "_worker_reapply_review", False))),
                7000,
            )
        else:
            self.statusBar().showMessage(completion_message(project), 5000)
        if getattr(self, "_worker_is_single_page", False) and not cancelled:
            # The automatic renderer rewrites the same result pathname in-place
            # across mode switches. Force the workbench to decode the newly
            # published pixels even on filesystems where timestamp/size cache
            # keys can collide for two rapid writes.
            try:
                self.workbench.image.clear_cache()
            except Exception:
                logger.debug("failed to invalidate workbench image cache after page run", exc_info=True)
            wanted = getattr(self, "_worker_page_id", "")
            if wanted:
                for i,pair in enumerate(self.state.pairs):
                    if page_id_for_pair(pair) == wanted:
                        self.set_selected_page(i); break
            # page_completion_message already includes whether this was a
            # reprocess; keep its mode/pixel-effect diagnostics visible instead
            # of overwriting them with a generic review-reapply sentence.
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
    apply_application_icon(app=app)
    app.setStyleSheet(style_for_theme(_saved_theme_name()))
    if sys.platform == "darwin":
        font=QFont("SF Pro Text"); font.setPointSize(12); app.setFont(font)
    win = StudioWindow()
    apply_application_icon(app=app, window=win)
    win.show()
    result = int(app.exec())
    logger.info("GUI stopped exit_code=%s", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
