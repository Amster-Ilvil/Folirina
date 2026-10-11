from __future__ import annotations

"""Mode-dispatched OCR block editor used by the Studio workbench."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout,
)
from .config import PipelineConfig
from .gui_workers import PageActionWorker
from .gui_components import _configure_responsive_dialog, StableComboBox, StableSpinBox, StableDoubleSpinBox
from .font_catalog import discover_fonts, import_font_to_library, persist_font_expression, font_library_dir
from .manual_effect import map_target_bbox_to_source
from .gui_dialogs import confirm_action
from .studio_review_views import RegionSelectView, OCRLayoutPreviewView
from .modes.reletter import ocr_edit_blocks as _reletter_ocr_edit_blocks
from .modes.reletter import ocr_edit_render as _reletter_ocr_edit_render
from .modes.hybrid import ocr_edit_blocks as _hybrid_ocr_edit_blocks
from .modes.hybrid import ocr_edit_render as _hybrid_ocr_edit_render
from . import ocr_edit_blocks as _shared_ocr_edit_blocks
from . import ocr_edit_render as _shared_ocr_edit_render

QComboBox = StableComboBox
QSpinBox = StableSpinBox
QDoubleSpinBox = StableDoubleSpinBox
logger = logging.getLogger(__name__)


def _confirm_destructive_action(parent, title: str, message: str, *, confirm_text: str = "确认", cancel_text: str = "取消") -> bool:
    return confirm_action(parent, title, message, confirm_text=confirm_text, cancel_text=cancel_text, destructive=True)


def _ocr_mode_modules(mode: str):
    key = str(mode or "").strip().lower()
    if key == "hybrid":
        return _hybrid_ocr_edit_blocks, _hybrid_ocr_edit_render
    if key == "reletter":
        return _reletter_ocr_edit_blocks, _reletter_ocr_edit_render
    if _shared_ocr_edit_blocks.is_ocr_edit_mode(key):
        return _shared_ocr_edit_blocks, _shared_ocr_edit_render
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


def recognize_manual_ocr_block(project, source_path, target_path, bbox, config, *, existing=None, mode: str | None = None):
    resolved_mode = str(mode or (project.get("meta") or {}).get("transfer_mode") or getattr(config.transfer, "mode", "") or "").strip().lower()
    blocks, _ = _ocr_mode_modules(resolved_mode)
    if blocks is None:
        raise RuntimeError(f"OCR block editor unavailable for mode: {resolved_mode}")
    return blocks.recognize_manual_ocr_block(project, source_path, target_path, bbox, config, existing=existing)


def apply_ocr_edit_blocks(page_dir, project, cfg):
    mode = str((project.get("meta") or {}).get("transfer_mode") or getattr(cfg.transfer, "mode", "") or "").strip().lower()
    _, renderer = _ocr_mode_modules(mode)
    if renderer is None:
        raise RuntimeError(f"OCR edit render unavailable for mode: {mode}")
    return renderer.apply_ocr_edit_blocks(page_dir, project, cfg)


def preview_ocr_edit_blocks(page_dir, project, cfg, blocks, *, mode: str | None = None):
    resolved_mode = str(mode or (project.get("meta") or {}).get("transfer_mode") or getattr(cfg.transfer, "mode", "") or "").strip().lower()
    _, renderer = _ocr_mode_modules(resolved_mode)
    fn = getattr(renderer, "preview_ocr_edit_blocks", None) if renderer is not None else None
    if not callable(fn):
        raise RuntimeError(f"OCR live preview unavailable for mode: {resolved_mode}")
    return fn(page_dir, project, cfg, blocks)


def reset_ocr_edit_blocks(page_dir, project, cfg):
    mode = str((project.get("meta") or {}).get("transfer_mode") or getattr(cfg.transfer, "mode", "") or "").strip().lower()
    _, renderer = _ocr_mode_modules(mode)
    if renderer is None:
        raise RuntimeError(f"OCR edit reset unavailable for mode: {mode}")
    return renderer.reset_ocr_edit_blocks(page_dir, project, cfg)

class OCRBlockEditorDialog(QDialog):
    """Live manual OCR/typesetting editor.

    Persistent OCR locator geometry is separated from the transient layout frame:
    TARGET bbox decides what old text may be cleared; layout_bbox only controls
    where the new lettering may be placed. Live preview always uses the exact
    mode-owned renderer in read-only mode and never writes page artifacts.
    """

    def __init__(self, page_dir: str | Path, source_path: str | Path, target_path: str | Path,
                 project: dict[str, Any], config: PipelineConfig, mode: str, parent=None):
        super().__init__(parent)
        if not is_ocr_edit_mode(mode):
            raise ValueError("人工 OCR 文本块在当前整页模式不可用。")
        self.page_dir=Path(page_dir); self.source_path=Path(source_path); self.target_path=Path(target_path)
        self.project=dict(project or {}); self.config=config.model_copy(deep=True); self.mode=str(mode)
        self._blocks=load_ocr_blocks(self.page_dir,self.mode); self._current_id=""; self._ocr_worker=None; self._preview_worker=None
        self._preview_dirty=False; self._loading_controls=False; self._save_after_preview=False; self._close_after_preview=False
        self._inline_sync=False
        # The large preview is intentionally dual-purpose.  Before a text block
        # has a locator, its blue rectangle is a convenient OCR locator proxy;
        # once text/layout exists the same rectangle becomes the layout frame.
        # Keeping the role explicit prevents the v2.3.86 ambiguity where users
        # drew the obvious large blue box but the OCR button still saw no TARGET
        # locator and therefore could not run.
        self._preview_box_role="locator"
        self._preview_timer=QTimer(self); self._preview_timer.setSingleShot(True); self._preview_timer.setInterval(150); self._preview_timer.timeout.connect(self._start_preview)
        mode_label={"direct_patch":"Direct · 人工 OCR","mask_replace":"精准蒙版 · 人工 OCR","hybrid":"精准蒙版+OCR","reletter":"OCR重排"}.get(self.mode,"人工 OCR")
        self.setWindowTitle("人工 OCR 实时排版编辑器 · " + mode_label)
        _configure_responsive_dialog(self,(1740,1050),(1120,740))
        root=QVBoxLayout(self); root.setContentsMargins(10,10,10,10); root.setSpacing(7)
        hint=QLabel("新建文本块时，可在 TARGET 或右侧“实时成品”上直接框选 OCR 区域。OCR 完成后可直接双击蓝色文字框，在框内编辑中文；输入会实时调用当前模式的排版器重排。Ctrl/⌘+Enter 完成框内编辑，Esc 撤销本次框内输入。定位框只负责清除原日文，排版框只负责中文位置，两者不会互相覆盖；只有“保存并应用”才写入正式结果。")
        hint.setWordWrap(True); hint.setObjectName("hint"); root.addWidget(hint)

        top=QHBoxLayout(); self.block_combo=QComboBox(); self.block_combo.addItem("新建 OCR 文本块","")
        self.new_btn=QPushButton("新建"); self.delete_btn=QPushButton("删除"); self.delete_btn.setObjectName("dangerCompact")
        self.auto_fit_btn=QPushButton("自动排版"); self.auto_fit_btn.setObjectName("softPrimary")
        top.addWidget(QLabel("文本块")); top.addWidget(self.block_combo,1); top.addWidget(self.auto_fit_btn); top.addWidget(self.new_btn); top.addWidget(self.delete_btn); root.addLayout(top)

        split=QSplitter(Qt.Orientation.Horizontal)
        source_card=QFrame(); source_card.setObjectName("card"); sl=QVBoxLayout(source_card); sl.setContentsMargins(7,7,7,7); sl.addWidget(QLabel("SOURCE 中文 · 自动映射")); self.source_view=RegionSelectView(self.source_path,editable=False,parent=self); sl.addWidget(self.source_view,1)
        target_card=QFrame(); target_card.setObjectName("card"); tl=QVBoxLayout(target_card); tl.setContentsMargins(7,7,7,7); tl.addWidget(QLabel("TARGET 日文定位框 · 只决定清除区域")); self.target_view=RegionSelectView(self.target_path,editable=True,parent=self); tl.addWidget(self.target_view,1)
        visible=self.page_dir/"final_reviewed.png"
        if not visible.exists(): visible=self.page_dir/"final.png"
        if not visible.exists(): visible=self.target_path
        preview_card=QFrame(); preview_card.setObjectName("card"); vl=QVBoxLayout(preview_card); vl.setContentsMargins(7,7,7,7)
        preview_head=QHBoxLayout(); preview_head.addWidget(QLabel("实时成品预览 · OCR 后双击蓝框直接编辑；拖动/缩放框实时重排")); self.inline_edit_btn=QPushButton("框内编辑"); self.inline_edit_btn.setObjectName("compactAction"); self.guide_visible=QCheckBox("显示蓝框"); self.guide_visible.setChecked(True); preview_head.addStretch(1); preview_head.addWidget(self.inline_edit_btn); preview_head.addWidget(self.guide_visible); vl.addLayout(preview_head)
        self.preview_view=OCRLayoutPreviewView(visible,parent=self); vl.addWidget(self.preview_view,1)
        self.preview_status=QLabel("实时预览等待参数"); self.preview_status.setObjectName("hint"); self.preview_status.setWordWrap(True); vl.addWidget(self.preview_status)
        split.addWidget(source_card); split.addWidget(target_card); split.addWidget(preview_card); split.setStretchFactor(0,1); split.setStretchFactor(1,1); split.setStretchFactor(2,2); root.addWidget(split,1)

        panel=QFrame(); panel.setObjectName("selectionPanel"); pl=QVBoxLayout(panel); pl.setContentsMargins(10,8,10,8); pl.setSpacing(6)
        ocrrow=QHBoxLayout(); self.ocr_btn=QPushButton("重新 OCR 当前定位框"); self.ocr_btn.setObjectName("softPrimary"); self.ocr_status=QLabel("先框选区域"); self.ocr_status.setObjectName("hint"); ocrrow.addWidget(self.ocr_btn); ocrrow.addWidget(self.ocr_status,1); pl.addLayout(ocrrow)
        self.text=QPlainTextEdit(); self.text.setPlaceholderText("兼容文本编辑区（与框内编辑双向同步）。回车可手工指定换行。")
        self.text.setMaximumHeight(92); pl.addWidget(self.text)

        r1=QHBoxLayout(); self.orientation=QComboBox(); self.orientation.addItem("自动","auto"); self.orientation.addItem("竖排","vertical"); self.orientation.addItem("横排","horizontal")
        self.break_mode=QComboBox(); self.break_mode.addItem("智能断句","smart"); self.break_mode.addItem("均衡断句","balanced"); self.break_mode.addItem("保留手工换行","source")
        self.layout_mode=QComboBox(); self.layout_mode.addItem("智能缩放","smart_scaling"); self.layout_mode.addItem("严格框内","strict"); self.layout_mode.addItem("气泡填充","balloon_fill")
        self.alignment=QComboBox(); self.alignment.addItem("自动","auto"); self.alignment.addItem("居中","center"); self.alignment.addItem("靠前/左","left"); self.alignment.addItem("靠后/右","right")
        self.layout_box_mode=QComboBox(); self.layout_box_mode.addItem("自动气泡/文本框","auto"); self.layout_box_mode.addItem("手动排版框","manual")
        self.reset_layout_btn=QPushButton("恢复自动框"); self.reset_layout_btn.setObjectName("compactAction")
        for label,widget in [("方向",self.orientation),("断句",self.break_mode),("排版",self.layout_mode),("对齐",self.alignment),("文字框",self.layout_box_mode)]: r1.addWidget(QLabel(label)); r1.addWidget(widget)
        r1.addWidget(self.reset_layout_btn); r1.addStretch(1); pl.addLayout(r1)

        r2=QHBoxLayout(); self.font=QLineEdit(); self.font.setPlaceholderText("留空使用全局字体；外部字体保存时自动导入 Folirina 字体库")
        self.font_pick=QPushButton("导入字体…"); self.font_catalog=QComboBox(); self.font_catalog.addItem("选择字体库字体",""); self.font_refresh=QPushButton("刷新"); self.font_refresh.setObjectName("compactAction"); self.font_refresh.setMaximumWidth(58)
        r2.addWidget(QLabel("字体")); r2.addWidget(self.font,2); r2.addWidget(self.font_pick); r2.addWidget(QLabel("字体库")); r2.addWidget(self.font_catalog,1); r2.addWidget(self.font_refresh); pl.addLayout(r2)

        r3=QHBoxLayout(); self.font_size=QSpinBox(); self.font_size.setRange(0,220); self.font_size.setSpecialValueText("自动"); self.font_size.setSuffix(" px")
        self.columns=QSpinBox(); self.columns.setRange(0,16); self.columns.setSpecialValueText("自动")
        self.line_spacing=QDoubleSpinBox(); self.line_spacing.setRange(-1.0,0.90); self.line_spacing.setSingleStep(0.02); self.line_spacing.setDecimals(2); self.line_spacing.setSpecialValueText("自动")
        self.letter_spacing=QDoubleSpinBox(); self.letter_spacing.setRange(-1.0,0.80); self.letter_spacing.setSingleStep(0.02); self.letter_spacing.setDecimals(2); self.letter_spacing.setSpecialValueText("自动")
        self.column_spacing=QDoubleSpinBox(); self.column_spacing.setRange(-1.0,0.80); self.column_spacing.setSingleStep(0.02); self.column_spacing.setDecimals(2); self.column_spacing.setSpecialValueText("自动")
        self.rotation=QDoubleSpinBox(); self.rotation.setRange(-180.0,180.0); self.rotation.setSingleStep(1.0); self.rotation.setDecimals(1); self.rotation.setSuffix("°")
        self.fill_color=QLineEdit("#000000"); self.fill_color.setMaximumWidth(88); self.fill_color.setPlaceholderText("#000000")
        for label,widget in [("字号",self.font_size),("列数",self.columns),("行距",self.line_spacing),("字距",self.letter_spacing),("列距",self.column_spacing),("旋转",self.rotation),("文字色",self.fill_color)]: r3.addWidget(QLabel(label)); r3.addWidget(widget)
        r3.addStretch(1); pl.addLayout(r3); root.addWidget(panel)

        actions=QHBoxLayout(); actions.addStretch(1); self.cancel_btn=QPushButton("取消"); self.save_btn=QPushButton("保存并应用"); self.save_btn.setObjectName("primary"); actions.addWidget(self.cancel_btn); actions.addWidget(self.save_btn); root.addLayout(actions)

        self.cancel_btn.clicked.connect(self.reject); self.save_btn.clicked.connect(self._save); self.ocr_btn.clicked.connect(lambda _checked=False: self._rerun_ocr()); self.new_btn.clicked.connect(self._new); self.delete_btn.clicked.connect(self._delete); self.auto_fit_btn.clicked.connect(self._reset_auto_typesetting)
        self.font_pick.clicked.connect(self._pick_font); self.block_combo.currentIndexChanged.connect(self._load_selected); self.font_refresh.clicked.connect(lambda: self._refresh_font_catalog(force=True)); self.font_catalog.currentIndexChanged.connect(self._apply_catalog_font); self.font.editingFinished.connect(self._persist_font_field)
        self.target_view.selection_changed.connect(self._target_selection_changed); self.preview_view.selection_changed.connect(self._layout_selection_changed); self.guide_visible.toggled.connect(self.preview_view.set_selection_overlay_visible); self.reset_layout_btn.clicked.connect(self._reset_layout_box)
        self.inline_edit_btn.clicked.connect(self._begin_inline_edit); self.preview_view.inline_text_changed.connect(self._inline_text_changed); self.preview_view.inline_edit_started.connect(lambda: self.preview_status.setText("框内编辑中 · Ctrl/⌘+Enter 完成，Esc 撤销；输入会实时重排")); self.preview_view.inline_edit_finished.connect(lambda: self._schedule_preview())
        self.text.textChanged.connect(self._panel_text_changed)
        for signal in [self.text.textChanged,self.orientation.currentIndexChanged,self.break_mode.currentIndexChanged,self.layout_mode.currentIndexChanged,self.alignment.currentIndexChanged,self.layout_box_mode.currentIndexChanged,self.font.textChanged,self.font_size.valueChanged,self.columns.valueChanged,self.line_spacing.valueChanged,self.letter_spacing.valueChanged,self.column_spacing.valueChanged,self.rotation.valueChanged,self.fill_color.textChanged]:
            signal.connect(self._schedule_preview)
        self._refresh_font_catalog(); self._reload_combo()

    def _panel_text_changed(self):
        if self._inline_sync: return
        value=self.text.toPlainText(); self.preview_view.set_inline_text(value); self.preview_view.set_inline_edit_enabled(bool(value.strip()))
        if value.strip(): self._preview_box_role="layout"

    def _inline_text_changed(self,value:str):
        if self._inline_sync: return
        self._inline_sync=True
        try:
            if self.text.toPlainText()!=str(value or ""):
                self.text.setPlainText(str(value or ""))
        finally: self._inline_sync=False
        self._preview_box_role="layout" if str(value or "").strip() else "locator"
        self.preview_view.set_inline_edit_enabled(bool(str(value or "").strip()))
        self._schedule_preview()

    def _begin_inline_edit(self):
        value=self.text.toPlainText()
        if not value.strip():
            self.preview_status.setText("先执行 OCR 或输入中文，随后即可在蓝框内直接编辑。")
            return False
        if len(self.preview_view.box())!=4:
            bbox=self.target_view.box()
            if len(bbox)==4:
                self._loading_controls=True
                try: self.preview_view.set_box(bbox,emit=False)
                finally: self._loading_controls=False
        self.preview_view.set_inline_text(value); self.preview_view.set_inline_edit_enabled(True)
        ok=self.preview_view.begin_inline_edit()
        if not ok: self.preview_status.setText("当前没有可编辑的文字框；请先确定 OCR 定位框。")
        return ok

    def _reload_combo(self, select_id: str=""):
        self._blocks=load_ocr_blocks(self.page_dir,self.mode); self.block_combo.blockSignals(True); self.block_combo.clear(); self.block_combo.addItem("新建 OCR 文本块","")
        for i,row in enumerate(self._blocks,1):
            text=str(row.get("render_text") or row.get("ocr_text") or "").replace("\n"," ").strip(); self.block_combo.addItem(f"{i}. {text[:28] or '未识别'}",str(row.get("id") or ""))
        idx=self.block_combo.findData(select_id) if select_id else 0; self.block_combo.setCurrentIndex(max(0,idx)); self.block_combo.blockSignals(False); self._load_selected()

    def _selected_row(self):
        bid=str(self.block_combo.currentData() or ""); return next((dict(x) for x in self._blocks if str(x.get("id") or "")==bid),None)

    def _set_combo_value(self,combo,value):
        i=combo.findData(str(value or "")); combo.setCurrentIndex(max(0,i))

    def _load_selected(self,*_):
        if hasattr(self,"_pending_ocr"):
            delattr(self,"_pending_ocr")
        self._loading_controls=True
        try:
            row=self._selected_row(); self._current_id=str(row.get("id") or "") if row else ""
            if not row:
                self._preview_box_role="locator"
                self.target_view.set_box([]); self.source_view.set_box([]); self.preview_view.clear_selection(); self.text.clear(); self.font.clear(); self.font_size.setValue(0); self.columns.setValue(0); self.line_spacing.setValue(-1.0); self.letter_spacing.setValue(-1.0); self.column_spacing.setValue(-1.0); self.rotation.setValue(0.0); self.fill_color.setText("#000000")
                for combo,val in [(self.orientation,"auto"),(self.break_mode,"smart"),(self.layout_mode,"smart_scaling"),(self.alignment,"center"),(self.layout_box_mode,"auto")]: self._set_combo_value(combo,val)
                self.ocr_status.setText("新建：可在 TARGET 或右侧实时成品上拖出 OCR 定位框")
            else:
                self._preview_box_role="layout"
                tb=list(row.get("target_bbox") or []); sb=list(row.get("source_bbox") or []); self.target_view.set_box(tb); self.source_view.set_box(sb); self.text.setPlainText(str(row.get("render_text") or row.get("ocr_text") or "")); self.font.setText(str(row.get("font_path") or "")); self.font_size.setValue(int(row.get("font_size") or 0)); self.columns.setValue(int(row.get("columns") or 0))
                spacing=row.get("line_spacing_ratio"); self.line_spacing.setValue(-1.0 if spacing is None else float(spacing)); ls=row.get("letter_spacing_ratio"); self.letter_spacing.setValue(-1.0 if ls is None else float(ls)); cs=row.get("column_spacing_ratio"); self.column_spacing.setValue(-1.0 if cs is None else float(cs)); self.rotation.setValue(float(row.get("rotation_degrees") or 0.0))
                fill=row.get("fill_color"); self.fill_color.setText(str(fill) if isinstance(fill,str) and fill else ("#%02x%02x%02x"%tuple(fill[:3]) if isinstance(fill,(list,tuple)) and len(fill)>=3 else "#000000"))
                for combo,val in [(self.orientation,row.get("orientation") or "auto"),(self.break_mode,row.get("line_break_mode") or "smart"),(self.layout_mode,row.get("layout_mode") or "smart_scaling"),(self.alignment,row.get("text_alignment") or "center"),(self.layout_box_mode,row.get("layout_box_mode") or "auto")]: self._set_combo_value(combo,val)
                if str(row.get("layout_box_mode") or "auto").lower()=="manual" and len(list(row.get("layout_bbox") or []))==4: self.preview_view.set_box(list(row.get("layout_bbox")))
                else: self.preview_view.clear_selection()
                if hasattr(self,"font_catalog"):
                    idx=self.font_catalog.findData(self.font.text().strip()); self.font_catalog.blockSignals(True); self.font_catalog.setCurrentIndex(max(0,idx)); self.font_catalog.blockSignals(False)
                conf=float(row.get("confidence") or 0.0); self.ocr_status.setText(f"OCR 置信度 {conf:.2f}" if conf else "可重新 OCR")
        finally:
            self._loading_controls=False
        self._panel_text_changed(); self._schedule_preview()

    def _sync_ocr_locator(self,bbox,*,mirror_preview:bool=False,origin:str="target") -> list[int]:
        box=[int(v) for v in list(bbox or [])] if bbox and len(bbox)==4 else []
        if len(box)!=4:
            return []
        self._loading_controls=True
        try:
            self.target_view.set_box(box,emit=False)
            try: self.source_view.set_box(map_target_bbox_to_source(self.project,box),emit=False)
            except Exception: self.source_view.set_box([],emit=False)
            if mirror_preview:
                # While this is a locator proxy layout_box_mode stays auto, so
                # the rectangle is never persisted as a lettering frame.
                self.preview_view.set_box(box,emit=False)
                self._set_combo_value(self.layout_box_mode,"auto")
            if mirror_preview or origin.startswith("preview") or not bool(self.text.toPlainText().strip()):
                self._preview_box_role="locator"
        finally:
            self._loading_controls=False
        self.ocr_status.setText("OCR 定位框已更新 · 可点击“重新 OCR 当前定位框”")
        self.preview_status.setText("蓝框当前是 OCR 定位框；OCR/输入中文后会自动切换为排版框。")
        logger.info("manual OCR locator updated mode=%s page=%s origin=%s bbox=%s",self.mode,self.page_dir,origin,box)
        self._schedule_preview()
        return box

    def _effective_ocr_locator(self) -> list[int]:
        bbox=self.target_view.box()
        if len(bbox)==4:
            return list(bbox)
        # v2.3.87 compatibility/usability bridge: the v2.3.86 UI made the large
        # preview look like the natural selection surface.  If a user has drawn
        # there first, adopt that exact TARGET-space box instead of rejecting the
        # OCR action with “please select TARGET”.
        pbox=self.preview_view.box()
        if len(pbox)==4:
            return self._sync_ocr_locator(pbox,mirror_preview=True,origin="preview_fallback")
        return []

    def _target_selection_changed(self,bbox):
        if self._loading_controls: return
        if bbox and len(bbox)==4:
            # For a new/empty block, mirror the TARGET locator into the larger
            # preview so the user can see and resize it in either pane. Existing
            # text blocks keep their independent manual/auto layout frame.
            empty_text=not bool(self.text.toPlainText().strip()) and not bool(getattr(self,"_pending_ocr",None))
            self._sync_ocr_locator(bbox,mirror_preview=empty_text,origin="target")
            return
        self._schedule_preview()

    def _layout_selection_changed(self,bbox):
        if self._loading_controls: return
        if bbox and len(bbox)==4 and (self._preview_box_role=="locator" or len(self.target_view.box())!=4):
            self._sync_ocr_locator(bbox,mirror_preview=False,origin="preview")
            return
        if bbox and len(bbox)==4:
            self._preview_box_role="layout"
            self._set_combo_value(self.layout_box_mode,"manual")
        self._schedule_preview()

    def _current_style(self):
        layout_box=self.preview_view.box() if str(self.layout_box_mode.currentData() or "auto")=="manual" else []
        return {"id":self._current_id,"render_text":self.text.toPlainText(),"orientation":self.orientation.currentData() or "auto","line_break_mode":self.break_mode.currentData() or "smart","layout_mode":self.layout_mode.currentData() or "smart_scaling","font_path":self.font.text().strip(),"font_size":int(self.font_size.value()),"columns":int(self.columns.value()),"line_spacing_ratio":None if float(self.line_spacing.value())<0 else float(self.line_spacing.value()),"letter_spacing_ratio":None if float(self.letter_spacing.value())<=-0.99 else float(self.letter_spacing.value()),"column_spacing_ratio":None if float(self.column_spacing.value())<=-0.99 else float(self.column_spacing.value()),"text_alignment":self.alignment.currentData() or "center","layout_bbox":list(layout_box),"layout_box_mode":self.layout_box_mode.currentData() or "auto","rotation_degrees":float(self.rotation.value()),"fill_color":self.fill_color.text().strip() or "#000000"}

    def _draft_row(self):
        base=dict(getattr(self,"_pending_ocr",None) or self._selected_row() or {}); base.update(self._current_style()); bbox=self.target_view.box(); base["id"]=str(base.get("id") or self._current_id or "__live_ocr_block__"); base["target_bbox"]=list(bbox)
        if len(bbox)==4:
            try: base["source_bbox"]=map_target_bbox_to_source(self.project,list(bbox))
            except Exception: pass
        base["render_text"]=self.text.toPlainText().strip(); base.setdefault("ocr_text",base["render_text"]); base["review_kind"]="manual_ocr"; base["box_locked"]=True; base["manual_override"]=True
        return base

    def _draft_blocks(self):
        row=self._draft_row(); rid=str(row.get("id") or ""); rows=[dict(x) for x in self._blocks if str(x.get("id") or "") not in {rid,str(self._current_id or "")}]
        if len(list(row.get("target_bbox") or []))==4 and str(row.get("render_text") or "").strip(): rows.append(row)
        return rows,row

    def _schedule_preview(self,*_):
        if self._loading_controls: return
        self._preview_dirty=True; self._preview_timer.start()

    def _start_preview(self):
        worker=getattr(self,"_preview_worker",None)
        if worker is not None and worker.isRunning(): self._preview_dirty=True; return
        rows,current=self._draft_blocks(); bbox=list(current.get("target_bbox") or [])
        if len(bbox)!=4:
            self.preview_status.setText("先在 TARGET 或右侧实时成品中确定 OCR 定位区域。")
            return
        if not str(current.get("render_text") or "").strip():
            # Do not render unrelated saved OCR blocks as if they were the new
            # block's live layout.  v2.3.86 could steal the last existing block's
            # layout box here and make the freshly drawn locator appear to jump.
            self._preview_dirty=False
            self.preview_status.setText("OCR 定位框已就绪 · 点击重新 OCR，或直接输入中文。")
            return
        self._preview_dirty=False; self.preview_status.setText("实时排版中…")
        project=dict(self.project); cfg=self.config.model_copy(deep=True); page_dir=Path(self.page_dir)
        worker=PageActionWorker("人工 OCR 实时排版",lambda:preview_ocr_edit_blocks(page_dir,project,cfg,rows,mode=self.mode)); self._preview_worker=worker
        def done(payload):
            data=dict(payload or {}); image=data.get("image"); state=dict(data.get("state") or {})
            if isinstance(image,np.ndarray): self.preview_view.replace_display_array(image)
            applied=[x for x in list(state.get("applied") or []) if isinstance(x,dict)]; cid=str(current.get("id") or ""); item=next((x for x in applied if str(x.get("id") or "")==cid),{})
            if item and bool(item.get("success")):
                layout=dict(item.get("layout") or {}); box=list(layout.get("container_bbox") or [])
                if str(self.layout_box_mode.currentData() or "auto")=="auto" and len(box)==4:
                    self._loading_controls=True
                    try: self.preview_view.set_box(box)
                    finally: self._loading_controls=False
                self._preview_box_role="layout"
                kind=str(layout.get("layout_kind") or "open"); fs=int(item.get("font_size") or 0); lines=list(item.get("lines") or []); cov=float(item.get("coverage_inside_safe") or 0.0); self.preview_status.setText(f"实时最终效果 · {kind} · {fs}px · {len(lines)} 行/列 · 安全覆盖 {cov*100:.1f}% · 框仅为编辑辅助，不写入成品")
            elif item:
                self.preview_status.setText("当前排版无法安全放入区域："+str(item.get("reason") or "layout_failed")+"。可缩小字号、调整排版框或字距。")
            else: self.preview_status.setText("当前参数没有可渲染文本。")
        def failed(message): self.preview_status.setText("实时预览失败："+str(message or "未知错误").splitlines()[0][:180])
        def finished():
            self._preview_worker=None
            if self._save_after_preview:
                self._save_after_preview=False; QTimer.singleShot(0,lambda:self._save(_from_preview=True)); return
            if self._close_after_preview:
                self._close_after_preview=False; QTimer.singleShot(0,self.reject); return
            if self._preview_dirty: QTimer.singleShot(0,self._start_preview)
        worker.done.connect(done); worker.failed.connect(failed); worker.finished.connect(finished); worker.finished.connect(worker.deleteLater); worker.start()

    def _reset_layout_box(self):
        self._preview_box_role="layout"
        self._loading_controls=True
        try: self._set_combo_value(self.layout_box_mode,"auto"); self.preview_view.clear_selection()
        finally: self._loading_controls=False
        self._schedule_preview()

    def _reset_auto_typesetting(self):
        self._preview_box_role="layout" if bool(self.text.toPlainText().strip()) else "locator"
        self._loading_controls=True
        try:
            self.font_size.setValue(0); self.columns.setValue(0); self.line_spacing.setValue(-1.0); self.letter_spacing.setValue(-1.0); self.column_spacing.setValue(-1.0); self.rotation.setValue(0.0); self._set_combo_value(self.alignment,"center"); self._set_combo_value(self.layout_box_mode,"auto"); self.preview_view.clear_selection()
        finally: self._loading_controls=False
        self._schedule_preview()

    def _rerun_ocr(self):
        bbox=self._effective_ocr_locator()
        if len(bbox)!=4:
            self.ocr_status.setText("请先在 TARGET 或右侧实时成品上拖出 OCR 定位区域。")
            logger.warning("manual OCR rerun ignored: missing locator mode=%s page=%s",self.mode,self.page_dir)
            return
        worker=getattr(self,"_ocr_worker",None)
        if worker is not None and worker.isRunning(): self.ocr_status.setText("OCR 正在处理中，请稍候…"); return
        seed=self._selected_row() or self._current_style(); seed.update(self._current_style()); bbox=list(bbox); project=dict(self.project); source_path=Path(self.source_path); target_path=Path(self.target_path); cfg=self.config.model_copy(deep=True)
        self.ocr_btn.setEnabled(False); self.save_btn.setEnabled(False); self.ocr_btn.setText("OCR 处理中…"); self.ocr_status.setText("正在识别 SOURCE 中文，并定位 TARGET 日文…")
        logger.info("manual OCR rerun requested mode=%s page=%s bbox=%s source=%s target=%s",self.mode,self.page_dir,bbox,source_path.name,target_path.name)
        worker=PageActionWorker(
            "人工 ROI OCR",
            lambda: recognize_manual_ocr_block(project,source_path,target_path,bbox,cfg,existing=seed,mode=self.mode),
        )
        self._ocr_worker=worker
        def done(payload):
            row=dict(payload or {}); self._current_id=str(row.get("id") or self._current_id); self._loading_controls=True
            try: self.text.setPlainText(str(row.get("render_text") or row.get("ocr_text") or "")); self.source_view.set_box(list(row.get("source_bbox") or []))
            finally: self._loading_controls=False
            self._pending_ocr=row; serr=str(row.get("source_ocr_error") or "").strip(); terr=str(row.get("target_ocr_error") or "").strip(); text=str(row.get("render_text") or row.get("ocr_text") or "").strip(); self._preview_box_role="layout" if text else "locator"
            self._panel_text_changed()
            logger.info("manual OCR rerun completed mode=%s page=%s bbox=%s source_backend=%s target_backend=%s text_chars=%d source_error=%s target_error=%s",self.mode,self.page_dir,row.get("target_bbox"),row.get("source_backend"),row.get("target_backend"),len(text),bool(serr),bool(terr))
            if serr: self.ocr_status.setText(f"{row.get('source_backend') or 'OCR'} SOURCE OCR 失败：{serr[:180]}")
            elif not text: self.ocr_status.setText("OCR 已执行，但 SOURCE ROI 没识别到文字。")
            else: self.ocr_status.setText(f"重新 OCR 完成 · 置信度 {float(row.get('confidence') or 0):.2f}"+("；TARGET 定位失败，将使用保守局部清字" if terr else ""))
            self._schedule_preview()
            if text: QTimer.singleShot(0,self._begin_inline_edit)
        def failed(message):
            logger.error("manual OCR rerun failed mode=%s page=%s bbox=%s error=%s",self.mode,self.page_dir,bbox,str(message or "未知错误").splitlines()[0][:500])
            self.ocr_status.setText("人工 OCR 执行失败："+str(message or "未知错误").splitlines()[0][:220])
        def finished(): self._ocr_worker=None; self.ocr_btn.setEnabled(True); self.save_btn.setEnabled(True); self.ocr_btn.setText("重新 OCR 当前定位框")
        worker.done.connect(done); worker.failed.connect(failed); worker.finished.connect(finished); worker.finished.connect(worker.deleteLater); worker.start()

    def closeEvent(self,event):  # noqa: N802
        if self._ocr_worker is not None and self._ocr_worker.isRunning(): self.ocr_status.setText("OCR 正在处理中，完成后即可关闭窗口。"); event.ignore(); return
        if self._preview_worker is not None and self._preview_worker.isRunning(): self._close_after_preview=True; self.preview_status.setText("正在结束实时预览…"); event.ignore(); return
        super().closeEvent(event)

    def _save(self,*_,_from_preview=False):
        if not _from_preview and self._preview_worker is not None and self._preview_worker.isRunning(): self._save_after_preview=True; self.preview_status.setText("等待当前实时预览完成后保存…"); return
        bbox=self.target_view.box(); text=self.text.toPlainText().strip()
        if len(bbox)!=4: self.ocr_status.setText("缺少定位框：请先在 TARGET 上框选原文字区域。"); return
        if not text: self.ocr_status.setText("OCR 结果为空：请重新 OCR，或手动输入中文后再保存。"); return
        color=self.fill_color.text().strip()
        if not (len(color) in {4,7} and color.startswith("#")):
            self.ocr_status.setText("文字颜色请使用 #RGB 或 #RRGGBB。"); return
        try: int(color[1:],16)
        except ValueError: self.ocr_status.setText("文字颜色格式无效。"); return
        try: self.font.setText(persist_font_expression(self.font.text().strip(), strict=True))
        except Exception as exc: self.ocr_status.setText(f"字体导入失败：{exc}"); return
        try:
            source_bbox=map_target_bbox_to_source(self.project,list(bbox))
        except Exception as exc:
            self.ocr_status.setText(f"SOURCE 映射失败：{str(exc)[:180]}")
            logger.error("manual OCR save mapping failed mode=%s page=%s bbox=%s error=%s",self.mode,self.page_dir,bbox,exc)
            return
        base=dict(getattr(self,"_pending_ocr",None) or self._selected_row() or {}); base.update(self._current_style()); base["target_bbox"]=list(bbox); base["source_bbox"]=source_bbox; base["render_text"]=text; base.setdefault("ocr_text",text); base["review_kind"]="manual_ocr"; base["box_locked"]=True; base["manual_override"]=True
        saved=upsert_ocr_block(self.page_dir,self.mode,base); self._current_id=str(saved.get("id") or ""); logger.info("manual OCR block saved mode=%s page=%s id=%s bbox=%s text_chars=%d",self.mode,self.page_dir,self._current_id,bbox,len(text)); self.accept()

    def _new(self):
        self._current_id=""; self._preview_box_role="locator"; self.block_combo.setCurrentIndex(0); self.target_view.set_box([]); self.source_view.set_box([]); self.preview_view.clear_selection(); self.preview_view.set_inline_edit_enabled(False); self.text.clear(); self.ocr_status.setText("新建：可在 TARGET 或右侧实时成品上拖出 OCR 定位框")
        if hasattr(self,"_pending_ocr"): delattr(self,"_pending_ocr")
        self._schedule_preview()

    def _delete(self):
        bid=str(self.block_combo.currentData() or "")
        if not bid: return
        if not _confirm_destructive_action(self,"删除 OCR 文本块","删除当前人工 OCR 文本块？",confirm_text="删除"): return
        delete_ocr_block(self.page_dir,self.mode,bid); self._reload_combo(); self.ocr_status.setText("已删除，正在返回并重新合成"); self.accept()

    def _refresh_font_catalog(self,*,force:bool=False):
        current=self.font.text().strip(); self.font_catalog.blockSignals(True); self.font_catalog.clear(); self.font_catalog.addItem("选择字体库字体","")
        for row in discover_fonts(limit=180,force=force): self.font_catalog.addItem(str(row.get("name") or Path(str(row.get("path") or "")).stem),str(row.get("path") or ""))
        idx=self.font_catalog.findData(current) if current else -1; self.font_catalog.setCurrentIndex(max(0,idx)); self.font_catalog.blockSignals(False); self.font_catalog.setToolTip(f"Folirina 持久字体库：{font_library_dir()}")

    def _apply_catalog_font(self):
        path=str(self.font_catalog.currentData() or "").strip()
        if path: self.font.setText(path)

    def _persist_font_field(self):
        raw=self.font.text().strip()
        if not raw: return
        try: stored=persist_font_expression(raw,strict=True)
        except Exception as exc: self.ocr_status.setText(f"字体不可用：{exc}"); return
        if stored!=raw: self.font.setText(stored); self._refresh_font_catalog(); self.ocr_status.setText("字体已复制到 Folirina 字体库，可长期复用。")
        self._schedule_preview()

    def _pick_font(self):
        current=self.font.text().strip(); start=str(Path(current).expanduser().parent) if current and Path(current).expanduser().exists() else str(Path.home()); path,_=QFileDialog.getOpenFileName(self,"导入当前 OCR 文本块字体",start,"Fonts (*.ttf *.otf *.ttc *.otc);;All Files (*)")
        if not path: return
        try: row=import_font_to_library(path)
        except Exception as exc: self.ocr_status.setText(f"字体导入失败：{exc}"); return
        self.font.setText(str(row.get("path") or "")); self._refresh_font_catalog(); self.ocr_status.setText(f"已导入字体库：{row.get('name') or Path(path).stem}"); self._schedule_preview()

