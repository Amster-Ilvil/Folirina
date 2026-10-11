from __future__ import annotations

"""Shared Qt views/dialogs for Studio manual review editors."""

import zipfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal, QRectF, QUrl
from PySide6.QtGui import QPixmap, QPainter, QColor, QImageReader, QImage, QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton,
    QVBoxLayout, QGraphicsScene, QGraphicsView, QSizePolicy, QButtonGroup, QCheckBox,
    QFrame, QGridLayout,
)
from .gui_components import _configure_responsive_dialog, _fit_scene_rect, StableSlider
from .gui_theme import ACCENT
from .io_utils import load_json
from .selection_overlay import selection_edge_mask, selection_edge_thickness_for_scale
from .region_selection import (
    selection_mask_from_spec, selection_bbox_from_spec, bbox_from_mask, spec_from_mask,
    snap_selection_mask_to_lineart, recognize_closed_region_from_selection,
)
from .manual_effect import clean_manual_target_text, composite_source_text_delta, estimate_source_background
from .graphics_zoom import GraphicsZoomController
from .gui_workers import PageActionWorker

QSlider = StableSlider

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
        self._zoom_controller = GraphicsZoomController(
            self,
            minimum_scale=0.05,
            maximum_scale=12.0,
            on_manual_zoom=lambda: setattr(self, "_auto_fit", False),
        )
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
        self._zoom_controller.cancel_pending(); self._auto_fit=False; self.resetTransform(); self.fitInView(rect,Qt.AspectRatioMode.KeepAspectRatio)

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
        self._zoom_controller.cancel_pending()
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
        if not self._zoom_controller.handle_wheel(event):
            event.accept()

    def event(self, event):
        controller = getattr(self, "_zoom_controller", None)
        if controller is not None and controller.handle_native_event(event):
            return True
        return super().event(event)

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
        self._preview_worker = None
        self._preview_version = 0
        self._preview_pending = False
        self._preview_closing = False
        self._pending_dialog_result = None
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
        if self._preview_closing: return
        if self._preview_fn is not None:
            self._preview_version += 1
            self._preview_timer.start()

    def _refresh_live_preview(self):
        if self._preview_closing or self._preview_fn is None: return
        if self._preview_timer.isActive(): self._preview_timer.stop()
        self._preview_version += 1
        if self._preview_worker is not None:
            self._preview_pending = True
            return
        version = self._preview_version
        snapshot = self.view.mask.copy()
        worker = PageActionWorker('复核蒙版实时预览', lambda: self._preview_fn(snapshot), parent=self)
        self._preview_worker = worker
        def done(result):
            if not self._preview_closing and version == self._preview_version:
                self.view.set_preview_image(result)
        def failed(message):
            # Never revert a later brush stroke to the original TARGET image.
            # Leave the previous preview visible and preserve the editable mask.
            logger = __import__('logging').getLogger(__name__)
            logger.warning('manual mask live preview failed: %s', message)
        def finished():
            self._preview_worker = None
            if self._preview_closing:
                QTimer.singleShot(0, self._finish_deferred_close)
                return
            if self._preview_pending or version != self._preview_version:
                self._preview_pending = False
                QTimer.singleShot(0, self._refresh_live_preview)
        worker.done.connect(done); worker.failed.connect(failed)
        worker.finished.connect(finished); worker.finished.connect(worker.deleteLater)
        worker.start()

    def _finish_deferred_close(self):
        if not self._preview_closing or self._preview_worker is not None:
            return
        result = self._pending_dialog_result
        self._pending_dialog_result = None
        # QDialog owns the QThread; finish the modal session only after it ends.
        super().done(int(result) if result is not None else int(QDialog.DialogCode.Rejected))

    def done(self, result):
        if self._preview_worker is not None:
            # A 2-5s image preview must not freeze the entire main Qt event loop
            # just because the user clicked Save/Cancel while it was running.
            self._preview_closing = True
            self._pending_dialog_result = int(result)
            self._preview_timer.stop()
            self._preview_pending = False
            self._preview_version += 1
            self.setEnabled(False)
            return
        self._preview_timer.stop()
        super().done(result)

    def closeEvent(self, event):
        if self._preview_worker is not None:
            self.done(QDialog.DialogCode.Rejected)
            event.ignore()
            return
        super().closeEvent(event)

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
        self._last_selection_preview_at=0.0
        self._selection_mode="rect"; self._snap_enabled=False; self._snap_distance=10; self._snap_diag={}; self._snap_cv_cache=None
        self._panning=False; self._pan_last=None; self._auto_fit=True; self._fit_pending=False
        self._zoom_controller = GraphicsZoomController(
            self,
            minimum_scale=0.05,
            maximum_scale=12.0,
            on_manual_zoom=lambda: setattr(self, "_auto_fit", False),
        )
        self._interaction_mode="selection"; self._brush_size=40; self._brush_last=None; self._brush_right=False
        self.setDragMode(QGraphicsView.DragMode.NoDrag); self.setInteractive(True); self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); self.fit_to_window()

    def _apply_fit(self):
        self._fit_pending=False
        if not self._auto_fit or self.viewport().width()<8 or self.viewport().height()<8: return
        self._zoom_controller.cancel_pending(); self.resetTransform(); self.fitInView(_fit_scene_rect(self._scene),Qt.AspectRatioMode.KeepAspectRatio)

    def fit_to_window(self): self._auto_fit=True; self._apply_fit()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self._auto_fit and not self._fit_pending:
            self._fit_pending=True; QTimer.singleShot(0,self._apply_fit)

    def wheelEvent(self,event):
        if not self._zoom_controller.handle_wheel(event): event.accept()

    def event(self,event):
        controller=getattr(self,"_zoom_controller",None)
        if controller is not None and controller.handle_native_event(event): return True
        return super().event(event)

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

    def replace_display_array(self,image_bgr:np.ndarray):
        """Replace the base pixmap from an in-memory BGR/RGB image.

        Used by the live OCR editor so transient previews never touch final.png
        or create editor-preview files on disk.
        """
        arr=np.asarray(image_bgr)
        if arr.ndim!=3 or arr.shape[2] not in (3,4): return False
        if arr.shape[1]!=self._pix.width() or arr.shape[0]!=self._pix.height(): return False
        if arr.shape[2]==3:
            rgb=cv2.cvtColor(arr,cv2.COLOR_BGR2RGB); fmt=QImage.Format.Format_RGB888
        else:
            rgb=cv2.cvtColor(arr,cv2.COLOR_BGRA2RGBA); fmt=QImage.Format.Format_RGBA8888
        q=QImage(rgb.data,rgb.shape[1],rgb.shape[0],int(rgb.strides[0]),fmt).copy()
        pix=QPixmap.fromImage(q); self._pix=pix
        items=[item for item in self._scene.items() if item is not self._overlay_item]
        if items:
            base=min(items,key=lambda item:item.zValue()); base.setPixmap(pix)
        return True

    def set_selection_overlay_visible(self,visible:bool):
        self._overlay_item.setVisible(bool(visible))

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
        # Continuous drags can generate hundreds of events per second on Mac
        # trackpads. Rebuilding the selection image at that rate saturates Qt.
        # Mouse release below always renders and commits the exact final shape.
        now=time.monotonic()
        if now-self._last_selection_preview_at < 0.025:
            event.accept(); return
        self._last_selection_preview_at=now
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


class _InlineOCRTextEdit(QPlainTextEdit):
    """IME-safe editor embedded directly inside the active OCR layout frame."""
    commit_requested = Signal()
    cancel_requested = Signal()
    focus_committed = Signal()

    def keyPressEvent(self,event):  # noqa: N802
        key=event.key(); mods=event.modifiers()
        if key in (Qt.Key.Key_Return,Qt.Key.Key_Enter) and bool(mods & (Qt.KeyboardModifier.ControlModifier|Qt.KeyboardModifier.MetaModifier)):
            self.commit_requested.emit(); event.accept(); return
        if key==Qt.Key.Key_Escape:
            self.cancel_requested.emit(); event.accept(); return
        super().keyPressEvent(event)

    def focusOutEvent(self,event):  # noqa: N802
        super().focusOutEvent(event)
        self.focus_committed.emit()


class OCRLayoutPreviewView(RegionSelectView):
    """Live-result view whose rectangle is a movable/resizable text frame.

    The frame is editor metadata only. It never becomes a TARGET erase mask and
    never appears in exported pixels. Drag inside to move, near an edge/corner
    to resize, or drag outside to create a new manual layout frame.
    """
    inline_text_changed = Signal(str)
    inline_edit_started = Signal()
    inline_edit_finished = Signal()

    def __init__(self,image_path,parent=None):
        super().__init__(image_path,editable=True,parent=parent)
        self.set_selection_mode("rect")
        self._frame_drag=None; self._frame_origin=[]; self._frame_press=None
        self._inline_enabled=False; self._inline_sync=False; self._inline_start_text=""
        self._inline_editor=_InlineOCRTextEdit()
        self._inline_editor.setPlaceholderText("在当前文字框内直接编辑中文…")
        self._inline_editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._inline_editor.setStyleSheet(
            "QPlainTextEdit{background:rgba(255,255,255,224);color:#151515;"
            "border:2px solid #1677ff;border-radius:4px;padding:4px;"
            "selection-background-color:#91caff;}"
        )
        self._inline_proxy=self._scene.addWidget(self._inline_editor)
        self._inline_proxy.setZValue(30); self._inline_proxy.hide()
        self._inline_editor.textChanged.connect(self._on_inline_text_changed)
        self._inline_editor.commit_requested.connect(self.finish_inline_edit)
        self._inline_editor.cancel_requested.connect(self.cancel_inline_edit)
        self._inline_editor.focus_committed.connect(self.finish_inline_edit)

    def set_inline_edit_enabled(self,enabled:bool):
        self._inline_enabled=bool(enabled)
        if not self._inline_enabled: self.finish_inline_edit()

    def set_inline_text(self,text:str):
        value=str(text or "")
        if self._inline_editor.toPlainText()==value: return
        self._inline_sync=True
        try: self._inline_editor.setPlainText(value)
        finally: self._inline_sync=False

    def _on_inline_text_changed(self):
        if self._inline_sync: return
        self.inline_text_changed.emit(self._inline_editor.toPlainText())

    def _sync_inline_editor_geometry(self):
        box=self.box()
        if len(box)!=4:
            self._inline_proxy.hide(); return
        x0,y0,x1,y1=[float(v) for v in box]; w=max(36.0,x1-x0); h=max(32.0,y1-y0)
        self._inline_proxy.setGeometry(QRectF(x0,y0,w,h))

    def begin_inline_edit(self,text:str|None=None):
        if not self._inline_enabled or len(self.box())!=4: return False
        if text is not None: self.set_inline_text(text)
        self._inline_start_text=self._inline_editor.toPlainText()
        self._sync_inline_editor_geometry(); self._inline_proxy.show(); self._inline_editor.show()
        self._inline_editor.setFocus(Qt.FocusReason.MouseFocusReason)
        cursor=self._inline_editor.textCursor(); cursor.movePosition(cursor.MoveOperation.End); self._inline_editor.setTextCursor(cursor)
        self.inline_edit_started.emit(); return True

    def finish_inline_edit(self):
        if not self._inline_proxy.isVisible(): return
        self._inline_proxy.hide(); self.setFocus(Qt.FocusReason.OtherFocusReason); self.inline_edit_finished.emit()

    def cancel_inline_edit(self):
        if not self._inline_proxy.isVisible(): return
        old=self._inline_start_text
        if self._inline_editor.toPlainText()!=old:
            self.set_inline_text(old); self.inline_text_changed.emit(old)
        self._inline_proxy.hide(); self.setFocus(Qt.FocusReason.OtherFocusReason); self.inline_edit_finished.emit()

    def set_box(self,bbox,*,emit:bool=False):
        super().set_box(bbox,emit=emit); self._sync_inline_editor_geometry()

    def clear_selection(self,*,emit:bool=False):
        self.finish_inline_edit(); super().clear_selection(emit=emit)

    def mouseDoubleClickEvent(self,event):
        if self._inline_enabled:
            p=self.mapToScene(event.position().toPoint())
            if self._frame_hit(p.x(),p.y()) is not None and self.begin_inline_edit():
                event.accept(); return
        super().mouseDoubleClickEvent(event)

    def _frame_hit(self,x:float,y:float):
        box=self.box()
        if len(box)!=4: return None
        x0,y0,x1,y1=map(float,box); tol=max(4.0,10.0/max(0.1,float(self.transform().m11())))
        if not (x0-tol<=x<=x1+tol and y0-tol<=y<=y1+tol): return None
        left=abs(x-x0)<=tol; right=abs(x-x1)<=tol; top=abs(y-y0)<=tol; bottom=abs(y-y1)<=tol
        if left and top: return "tl"
        if right and top: return "tr"
        if left and bottom: return "bl"
        if right and bottom: return "br"
        if left: return "l"
        if right: return "r"
        if top: return "t"
        if bottom: return "b"
        if x0<=x<=x1 and y0<=y<=y1: return "move"
        return None

    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton and self._interaction_mode=="selection":
            p=self.mapToScene(event.position().toPoint()); hit=self._frame_hit(p.x(),p.y())
            if hit is not None:
                self._frame_drag=hit; self._frame_origin=self.box(); self._frame_press=self._clamp(p.x(),p.y()); event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if self._frame_drag and self._frame_press and len(self._frame_origin)==4:
            p=self.mapToScene(event.position().toPoint()); now=self._clamp(p.x(),p.y()); px,py=self._frame_press
            dx,dy=now[0]-px,now[1]-py; x0,y0,x1,y1=self._frame_origin; kind=self._frame_drag
            if kind=="move":
                ww=x1-x0; hh=y1-y0; nx0=max(0,min(self._pix.width()-ww,x0+dx)); ny0=max(0,min(self._pix.height()-hh,y0+dy)); box=[nx0,ny0,nx0+ww,ny0+hh]
            else:
                if "l" in kind: x0+=dx
                if "r" in kind: x1+=dx
                if "t" in kind: y0+=dy
                if "b" in kind: y1+=dy
                x0=max(0,min(self._pix.width()-2,x0)); x1=max(x0+2,min(self._pix.width(),x1)); y0=max(0,min(self._pix.height()-2,y0)); y1=max(y0+2,min(self._pix.height(),y1)); box=[x0,y0,x1,y1]
            self.set_box([int(v) for v in box],emit=False); event.accept(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if self._frame_drag and event.button()==Qt.MouseButton.LeftButton:
            self._frame_drag=None; self._frame_press=None; self._frame_origin=[]; self._emit_selection(); event.accept(); return
        super().mouseReleaseEvent(event)


