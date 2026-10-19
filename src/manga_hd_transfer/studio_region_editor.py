from __future__ import annotations

"""Region-composite and manual-effect editors for Studio review."""

import copy
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)
from .gui_toast import notify
from .config import PipelineConfig
from .gui_workers import PageActionWorker
from .gui_components import _configure_responsive_dialog, StableComboBox, StableSpinBox, StableSlider
from .region_workspace_state import RegionWorkspaceLinkState, selection_signature
from .region_brush_reveal import (
    stroke_bbox as brush_stroke_bbox, paint_reveal_stroke_inplace, compose_reveal_patch,
    mask_bbox as brush_mask_bbox, mask_counts as brush_mask_counts,
    label_verified_white_bubbles, touched_verified_white_bubble_labels,
    expand_to_verified_white_bubbles_inplace,
)
from .region_selection import project_selection_spec
from .manual_effect import map_target_bbox_to_source, registration_homography
from .region_alignment import aligned_source_roi as _aligned_source_roi
from .studio_review_views import RegionSelectView, RevealMaskDialog, ManualTextMaskPreviewDialog
from .studio_ocr_editor import _ocr_mode_modules
from .schema_compat import as_dict_rows, as_list

QComboBox = StableComboBox
QSpinBox = StableSpinBox
QSlider = StableSlider
logger = logging.getLogger(__name__)


def _manual_effect_ops_for_mode(mode: str):
    key = str(mode or "").strip().lower()
    if key == "mask_replace":
        from .modes.mask_replace import open_text_manual as ops
        return ops
    if key == "hybrid":
        from .modes.hybrid import open_text_manual as ops
        return ops
    from . import manual_effect as ops
    return ops

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
        ("白底气泡", "region_white_bubble", "框选白底气泡：先把完整气泡内区清成纯白，再显示配准后的中文笔画；保留气泡边框。"),
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
        self._ocr_dirty=False
        self._ocr_worker=None
        self._ocr_busy=False
        self._ocr_request_signature=""
        self._commit_worker=None
        self._embedded=False
        self._closing=False
        self._discard_confirmed=False
        self._preview_worker=None
        self._preview_epoch=0
        self._preview_dirty=set()
        self._preview_versions={}
        self._tile_counts={}
        self._brush_counts={"transparent_pixels":0,"hole_pixels":0,"restore_pixels":0,"union_pixels":0}
        self._preview_timer=QTimer(self); self._preview_timer.setSingleShot(True)
        # UI was creating stale QThreads at 60Hz even while a drag invalidated
        # their output; coalesce into a few 30fps previews and flush at release.
        self._preview_timer.setInterval(36); self._preview_timer.timeout.connect(self._flush_brush_preview)
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
        self._brush_target=cv2.imread(str(self.target_path),cv2.IMREAD_COLOR)
        from .white_bubble_composite import verified_white_bubble_mask
        self._brush_white = (verified_white_bubble_mask(self._brush_target, self.project.get("target_bubbles", []))
                             if self._brush_target is not None else None)
        if self._brush_white is not None and np.any(self._brush_white):
            self._brush_white_labels, self._brush_white_stats = label_verified_white_bubbles(self._brush_white)
        else:
            self._brush_white_labels = self._brush_white_stats = None
        if self._brush_base is None or self._brush_base.shape[:2] != self.target_view.image_shape():
            raise ValueError("区域涂抹无法读取当前结果")
        if self._brush_source is None:
            raise ValueError("区域涂抹无法读取旧版中文 SOURCE")
        bh,bw=self._brush_base.shape[:2]
        self._brush_transparent=np.zeros((bh,bw),np.uint8); self._brush_hole=np.zeros((bh,bw),np.uint8)
        self._brush_restore=np.zeros((bh,bw),np.uint8)
        self._brush_history=[]; self._brush_redo_stack=[]; self._brush_stroke_before={}; self._brush_stroke_tiles=set()
        self._brush_preview_items={}; self._brush_source_tile_cache={}; self._brush_tile_size=256
        # Inpainting and inward feathering need pixels *outside* a tile. Without
        # this halo a character crossing x=256 gets a new artificial feather
        # boundary, and the submitted (cropped) patch disagrees with preview.
        self._brush_halo=20
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
        brush_help=QLabel("左键使用当前画笔；右键恢复原始日文；中键平移。恢复会覆盖已提交的中文，可保存、撤销和重做。")
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
        self.brush_undo_btn=QPushButton("撤销一笔"); self.brush_redo_btn=QPushButton("重做一笔"); self.brush_clear_btn=QPushButton("清空本次涂抹")
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
        # Embedded editors must never start another dialog event loop to leave.
        self._leave_notice=QFrame(); notice=QHBoxLayout(self._leave_notice)
        notice.addWidget(QLabel("有未提交的笔触 / OCR 文本：提交后再返回，或明确放弃。"),1)
        self._discard_btn=QPushButton("放弃编辑并返回")
        self._keep_editing_btn=QPushButton("继续编辑")
        notice.addWidget(self._discard_btn); notice.addWidget(self._keep_editing_btn)
        root.addWidget(self._leave_notice); self._leave_notice.hide()
        self._keep_editing_btn.clicked.connect(self._leave_notice.hide)
        self._discard_btn.clicked.connect(self._discard_and_leave)

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
        return int(self._brush_counts["union_pixels"])

    def _invalidate_ocr_binding(self, *, keep_user_text:bool=False):
        self._ocr_payload={}; self._link_state.clear_ocr()
        if not keep_user_text: self._ocr_dirty=False
        self._ocr_programmatic_change=True
        try:
            if not keep_user_text: self.ocr_text.clear()
        finally:
            self._ocr_programmatic_change=False
        if hasattr(self,"ocr_status"):
            self.ocr_status.setText("选区已变化 · OCR 结果已解绑，请重新识别或输入当前区域中文")

    def _ocr_text_changed(self):
        if self._ocr_programmatic_change: return
        self._ocr_dirty=True
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
        self.inset.setEnabled(key in {"region_hole_reveal", "region_white_bubble"})
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

    def _halo_bounds(self,box,pad=None):
        x0,y0,x1,y1=map(int,box); h,w=self._brush_transparent.shape
        halo=max(12,int(self._brush_halo),int(self.brush_feather.value())+8) if pad is None else max(0,int(pad))
        return max(0,x0-halo),max(0,y0-halo),min(w,x1+halo),min(h,y1+halo)

    def _tiles_for_bbox(self,bbox):
        if not bbox or len(bbox)!=4: return set()
        x0,y0,x1,y1=[int(v) for v in bbox]; size=int(self._brush_tile_size)
        if x1<=x0 or y1<=y0: return set()
        return {(tx,ty) for ty in range(max(0,y0)//size,(max(0,y1-1)//size)+1) for tx in range(max(0,x0)//size,(max(0,x1-1)//size)+1)}

    def _remember_before_tiles(self,keys):
        for key in keys:
            if key in self._brush_stroke_before: continue
            x0,y0,x1,y1=self._tile_bounds(key)
            self._brush_stroke_before[key]=(self._brush_transparent[y0:y1,x0:x1].copy(),self._brush_hole[y0:y1,x0:x1].copy(),self._brush_restore[y0:y1,x0:x1].copy())
            self._brush_stroke_tiles.add(key)

    def _source_tile(self,key):
        dx=int(self.offset_x.value()); dy=int(self.offset_y.value()); cache_key=(int(key[0]),int(key[1]),dx,dy)
        cached=self._brush_source_tile_cache.get(cache_key)
        if cached is not None: return cached
        roi=self._tile_bounds(key)
        crop,_identity=_aligned_source_roi(self._brush_source,self._brush_base.shape[:2],self.project,roi,source_offset_x=dx,source_offset_y=dy)
        self._brush_source_tile_cache[cache_key]=crop
        return crop

    def _refresh_brush_tile(self,key):
        x0,y0,x1,y1=self._tile_bounds(key)
        tr=self._brush_transparent[y0:y1,x0:x1]; ho=self._brush_hole[y0:y1,x0:x1]
        restore=self._brush_restore[y0:y1,x0:x1]
        alpha_present=bool(np.any(tr>0) or np.any(ho>0) or np.any(restore>0))
        item=self._brush_preview_items.get(key)
        if not alpha_present:
            if item is not None:
                try: self.target_view._scene.removeItem(item)
                except RuntimeError: pass
                self._brush_preview_items.pop(key,None)
            return
        base=self._brush_base[y0:y1,x0:x1]
        under=self._source_tile(key)
        preview,_patch=compose_reveal_patch(base,under,tr,ho,transparent_feather_px=int(self.brush_feather.value()),transparent_opacity=float(self.brush_opacity.value())/100.0,target_bgr=self._brush_target[y0:y1,x0:x1] if self._brush_target is not None else None,verified_white_mask=self._brush_white[y0:y1,x0:x1] if getattr(self,"_brush_white",None) is not None else None,restore_mask=restore)
        rgb=cv2.cvtColor(preview,cv2.COLOR_BGR2RGB); hh,ww=rgb.shape[:2]
        q=QImage(rgb.data,ww,hh,int(rgb.strides[0]),QImage.Format.Format_RGB888).copy(); pix=QPixmap.fromImage(q)
        if item is None:
            item=self.target_view._scene.addPixmap(pix); item.setZValue(2.5); self._brush_preview_items[key]=item
        else: item.setPixmap(pix)
        item.setPos(x0,y0)

    def _refresh_brush_tiles(self,keys):
        for key in set(keys or []):
            x0,y0,x1,y1=self._tile_bounds(key)
            counts=brush_mask_counts(self._brush_transparent[y0:y1,x0:x1],self._brush_hole[y0:y1,x0:x1],self._brush_restore[y0:y1,x0:x1])
            previous=self._tile_counts.get(key,{})
            for name,value in counts.items():
                self._brush_counts[name]+=value-previous.get(name,0)
            if counts["union_pixels"]: self._tile_counts[key]=counts
            else: self._tile_counts.pop(key,None)
            self._preview_versions[key]=self._preview_versions.get(key,0)+1
            self._preview_dirty.add(key)
        if self._preview_dirty and not self._preview_timer.isActive(): self._preview_timer.start()
        self._update_brush_controls()

    def _flush_brush_preview(self):
        if self._preview_worker is not None or not self._preview_dirty: return
        epoch=self._preview_epoch
        keys=sorted(self._preview_dirty)[:4]
        self._preview_dirty.difference_update(keys)
        snapshots=[]
        for key in keys:
            x0,y0,x1,y1=self._tile_bounds(key)
            extended=self._halo_bounds((x0,y0,x1,y1))
            xa,ya,xb,yb=extended
            snapshots.append((key,self._preview_versions[key],(x0,y0,x1,y1),extended,
                self._brush_base[ya:yb,xa:xb].copy(),
                self._brush_transparent[ya:yb,xa:xb].copy(),self._brush_hole[ya:yb,xa:xb].copy(),
                self._brush_target[ya:yb,xa:xb].copy() if self._brush_target is not None else None,
                self._brush_white[ya:yb,xa:xb].copy() if getattr(self,"_brush_white",None) is not None else None,
                self._brush_restore[ya:yb,xa:xb].copy()))
        source=self._brush_source; project=self.project; shape=self._brush_base.shape[:2]
        dx=int(self.offset_x.value()); dy=int(self.offset_y.value())
        feather=int(self.brush_feather.value()); opacity=float(self.brush_opacity.value())/100.0
        cached={key:self._brush_source_tile_cache.get((key[0],key[1],dx,dy,tuple(ext))) for key,_,_,ext,*_ in snapshots}
        def render():
            rows=[]
            for key,version,roi,extended,base,tr,ho,target,white,restore in snapshots:
                if not np.any(tr) and not np.any(ho) and not np.any(restore):
                    rows.append((key,version,roi,None,None)); continue
                under=cached[key]
                if under is None:
                    under,_identity=_aligned_source_roi(source,shape,project,extended,source_offset_x=dx,source_offset_y=dy)
                # The expanded tile can be 300x300, while one pen segment is
                # often only 10-40px wide. Compose its active brush footprint
                # plus halo instead of the entire expanded square.
                bx0,by0,bx1,by1=brush_mask_bbox(tr,ho,restore)
                pad=max(12,feather+8)
                bx0=max(0,bx0-pad); by0=max(0,by0-pad)
                bx1=min(base.shape[1],bx1+pad); by1=min(base.shape[0],by1+pad)
                preview=base.copy()
                local,_patch=compose_reveal_patch(
                    base[by0:by1,bx0:bx1],under[by0:by1,bx0:bx1],
                    tr[by0:by1,bx0:bx1],ho[by0:by1,bx0:bx1],
                    transparent_feather_px=feather,transparent_opacity=opacity,
                    target_bgr=target[by0:by1,bx0:bx1] if target is not None else None,
                    verified_white_mask=white[by0:by1,bx0:bx1] if white is not None else None,
                    restore_mask=restore[by0:by1,bx0:bx1])
                preview[by0:by1,bx0:bx1]=local
                x0,y0,x1,y1=roi; xa,ya,_,_=extended
                preview=preview[y0-ya:y1-ya,x0-xa:x1-xa]
                rows.append((key,version,roi,cv2.cvtColor(preview,cv2.COLOR_BGR2RGB),under,extended))
            return rows
        worker=PageActionWorker("区域画笔预览",render,parent=self); self._preview_worker=worker
        def done(rows):
            if epoch!=self._preview_epoch: return
            for result in rows:
                key,version,roi,rgb,under,*extra=result
                # Alignment is independent of the mask version. Keep it even
                # when another pen segment arrived during rendering.
                if extra: self._brush_source_tile_cache[(key[0],key[1],dx,dy,tuple(extra[0]))]=under
                # Show intermediate frames during a live stroke; otherwise a
                # continuously changing tile never publishes until mouse-up.
                if self._preview_versions.get(key)!=version and not self._brush_stroke_tiles: continue
                item=self._brush_preview_items.get(key)
                if rgb is None:
                    if item is not None:
                        self.target_view._scene.removeItem(item); self._brush_preview_items.pop(key,None)
                    continue
                h,w=rgb.shape[:2]
                pix=QPixmap.fromImage(QImage(rgb.data,w,h,int(rgb.strides[0]),QImage.Format.Format_RGB888).copy())
                if item is None:
                    item=self.target_view._scene.addPixmap(pix); item.setZValue(2.5); self._brush_preview_items[key]=item
                else: item.setPixmap(pix)
                item.setPos(roi[0],roi[1])
        def failed(message):
            self._trace_region("region_brush_preview_failed",{"reason":str(message)})
            self.brush_status.setText("预览计算失败，笔触已保留；可撤销或重新提交。")
        def finished():
            self._preview_worker=None
            if self._preview_dirty: self._preview_timer.start()
        worker.done.connect(done); worker.failed.connect(failed)
        worker.finished.connect(finished); worker.finished.connect(worker.deleteLater); worker.start()

    def _refresh_all_brush_tiles(self):
        if not hasattr(self,"_brush_transparent"): return
        keys=set(self._brush_preview_items)|set(self._tile_counts)
        self._refresh_brush_tiles(keys)

    def _brush_alignment_changed(self):
        if not hasattr(self,"_brush_source_tile_cache"): return
        self._brush_source_tile_cache.clear(); self._refresh_all_brush_tiles()

    def _brush_stroke_started(self,payload):
        self._brush_stroke_before={}; self._brush_stroke_tiles=set()

    def _brush_stroke_segment(self,payload):
        if self._link_state.applying: return
        if not self._brush_interaction: return
        start=list((payload or {}).get("from") or []); end=list((payload or {}).get("to") or [])
        if len(start)!=2 or len(end)!=2: return
        diameter=int((payload or {}).get("brush_size") or self.brush_size.value())
        candidate=brush_stroke_bbox(self._brush_transparent.shape,[start,end],diameter)
        keys=self._tiles_for_bbox(candidate); self._remember_before_tiles(keys)
        limit=None
        if self.brush_limit_selection.isChecked():
            if len(self.target_view.box()) != 4:
                self.brush_status.setText("已启用“仅在当前选区内涂抹”，但当前没有选区。先建立选区或关闭限制。")
                return
            limit=self.target_view._selection_mask
        mode="restore" if bool((payload or {}).get("right")) else self._brush_mode_key()
        diag=paint_reveal_stroke_inplace(self._brush_transparent,self._brush_hole,[start,end],diameter,mode,limit_mask=limit,restore_mask=self._brush_restore)
        if diag.changed_pixels>0:
            expanded_boxes=[]
            if mode != "restore" and self._brush_white_labels is not None:
                labels=touched_verified_white_bubble_labels(
                    self._brush_transparent,self._brush_hole,self._brush_white_labels,diag.bbox
                )
                if labels:
                    boxes=[]
                    for label in labels:
                        if label < len(self._brush_white_stats):
                            x,y,bw,bh=[int(v) for v in self._brush_white_stats[label,:4]]
                            boxes.append((x,y,x+bw,y+bh))
                    extra_tiles=set().union(*(self._tiles_for_bbox(box) for box in boxes))
                    self._remember_before_tiles(extra_tiles)
                    expanded_boxes=expand_to_verified_white_bubbles_inplace(
                        self._brush_transparent,self._brush_hole,self._brush_white_labels,
                        self._brush_white_stats,labels
                    )
                    for label in labels:
                        if label < len(self._brush_white_stats):
                            x,y,bw,bh=[int(v) for v in self._brush_white_stats[label,:4]]
                            roi=self._brush_restore[y:y+bh,x:x+bw]
                            roi[self._brush_white_labels[y:y+bh,x:x+bw]==label]=0
            # Feather/Telea touch neighboring tiles; invalidate those borders
            # too so a tile seam cannot resurrect Japanese strokes.
            dirty=set(self._tiles_for_bbox(self._halo_bounds(diag.bbox)))
            for box in expanded_boxes:
                dirty.update(self._tiles_for_bbox(box))
            self._refresh_brush_tiles(dirty)

    def _brush_stroke_finished(self):
        entries=[]; changed_keys=set()
        for key in sorted(self._brush_stroke_tiles):
            before=self._brush_stroke_before.get(key)
            if before is None: continue
            x0,y0,x1,y1=self._tile_bounds(key); after=(self._brush_transparent[y0:y1,x0:x1].copy(),self._brush_hole[y0:y1,x0:x1].copy(),self._brush_restore[y0:y1,x0:x1].copy())
            if all(np.array_equal(a,b) for a,b in zip(before,after)): continue
            entries.append((key,before,after)); changed_keys.add(key)
        self._brush_stroke_before={}; self._brush_stroke_tiles=set()
        if entries:
            self._brush_history.append(entries)
            if len(self._brush_history)>100: self._brush_history=self._brush_history[-100:]
            self._brush_redo_stack.clear()
        self._refresh_brush_tiles(changed_keys)
        if self._preview_worker is None and self._preview_dirty:
            # Final stroke frame must not wait for an idle timer; this is the
            # first-frame feedback people expect from the older editor.
            self._preview_timer.start(0)

    def _restore_brush_record(self,record,after:bool):
        keys=set()
        for key,before,changed in record:
            x0,y0,x1,y1=self._tile_bounds(key)
            for mask,value in zip((self._brush_transparent,self._brush_hole,self._brush_restore),changed if after else before):
                mask[y0:y1,x0:x1]=value
            keys.update(self._tiles_for_bbox(self._halo_bounds((x0,y0,x1,y1))))
        self._refresh_brush_tiles(keys)

    def _brush_undo(self):
        if not self._brush_history: return
        record=self._brush_history.pop(); self._restore_brush_record(record,False); self._brush_redo_stack.append(record); self._update_brush_controls()

    def _brush_redo(self):
        if not self._brush_redo_stack: return
        record=self._brush_redo_stack.pop(); self._restore_brush_record(record,True); self._brush_history.append(record); self._update_brush_controls()

    def _brush_clear(self):
        box=brush_mask_bbox(self._brush_transparent,self._brush_hole,self._brush_restore)
        keys=self._tiles_for_bbox(box)
        if not keys: return
        record=[]
        for key in sorted(keys):
            x0,y0,x1,y1=self._tile_bounds(key)
            before=tuple(mask[y0:y1,x0:x1].copy() for mask in (self._brush_transparent,self._brush_hole,self._brush_restore))
            if not any(np.any(mask) for mask in before): continue
            record.append((key,before,tuple(np.zeros_like(mask) for mask in before)))
            self._brush_transparent[y0:y1,x0:x1]=0; self._brush_hole[y0:y1,x0:x1]=0; self._brush_restore[y0:y1,x0:x1]=0
        if record:
            self._brush_history.append(record); self._brush_redo_stack.clear(); self._refresh_brush_tiles(keys)

    def _update_brush_controls(self):
        if not hasattr(self,"brush_status"): return
        counts=self._brush_counts
        self.brush_undo_btn.setEnabled(bool(self._brush_history)); self.brush_redo_btn.setEnabled(bool(self._brush_redo_stack)); self.brush_clear_btn.setEnabled(counts["union_pixels"]>0)
        self.brush_status.setText(f"透明 {counts['transparent_pixels']:,} px · 挖孔 {counts['hole_pixels']:,} px · 恢复 {counts['restore_pixels']:,} px · 撤销 {len(self._brush_history)} / 重做 {len(self._brush_redo_stack)}")
        self._update_region_action_state()

    def _clear_brush_preview_items(self):
        for item in list(self._brush_preview_items.values()):
            try: self.target_view._scene.removeItem(item)
            except RuntimeError: pass
        self._brush_preview_items.clear()

    def _reset_brush_session(self,final_path:Path|None=None):
        self._preview_epoch+=1; self._preview_dirty.clear(); self._preview_timer.stop()
        self._tile_counts.clear()
        self._brush_counts={"transparent_pixels":0,"hole_pixels":0,"restore_pixels":0,"union_pixels":0}
        if final_path is not None and final_path.exists():
            image=cv2.imread(str(final_path),cv2.IMREAD_COLOR)
            if image is not None and image.shape==self._brush_base.shape:
                self._brush_base=image; self.target_view.replace_display_image(final_path)
        self._brush_transparent[:]=0; self._brush_hole[:]=0; self._brush_restore[:]=0; self._brush_history.clear(); self._brush_redo_stack.clear(); self._brush_stroke_before={}; self._brush_stroke_tiles=set(); self._brush_source_tile_cache.clear(); self._clear_brush_preview_items(); self._update_brush_controls()

    def _brush_commit_snapshot(self):
        box=brush_mask_bbox(self._brush_transparent,self._brush_hole,self._brush_restore)
        if len(box)!=4: raise ValueError("当前没有可提交的涂抹揭示")
        # Commit exactly the same padded context used by live preview.
        box=list(self._halo_bounds(box)); x0,y0,x1,y1=box
        return {"box":box,"tr":self._brush_transparent[y0:y1,x0:x1].copy(),
            "ho":self._brush_hole[y0:y1,x0:x1].copy(),"restore":self._brush_restore[y0:y1,x0:x1].copy(),"base":self._brush_base[y0:y1,x0:x1].copy(),
            "target":self._brush_target[y0:y1,x0:x1].copy() if self._brush_target is not None else None,
            "white":self._brush_white[y0:y1,x0:x1].copy() if getattr(self,"_brush_white",None) is not None else None,
            "source":self._brush_source,"shape":self._brush_base.shape[:2],"project":copy.deepcopy(self.project),
            "dx":int(self.offset_x.value()),"dy":int(self.offset_y.value()),
            "feather":int(self.brush_feather.value()),"opacity":float(self.brush_opacity.value())/100.0,
            "diameter":int(self.brush_size.value())}

    def _build_brush_commit(self, snapshot=None):
        import uuid
        snapshot=self._brush_commit_snapshot() if snapshot is None else snapshot
        box=snapshot["box"]
        x0,y0,x1,y1=box
        tr=snapshot["tr"]; ho=snapshot["ho"]
        under,_identity=_aligned_source_roi(snapshot["source"],snapshot["shape"],snapshot["project"],(x0,y0,x1,y1),source_offset_x=snapshot["dx"],source_offset_y=snapshot["dy"])
        restore=snapshot.get("restore")
        _preview,patch=compose_reveal_patch(snapshot["base"],under,tr,ho,transparent_feather_px=snapshot["feather"],transparent_opacity=snapshot["opacity"],target_bgr=snapshot["target"],verified_white_mask=snapshot.get("white"),restore_mask=restore)
        if cv2.countNonZero(patch[:,:,3])<=0: raise ValueError("涂抹揭示的有效透明区域为空")
        counts=brush_mask_counts(tr,ho,restore); kinds=[]
        if counts["transparent_pixels"]: kinds.append("transparent")
        if counts["hole_pixels"]: kinds.append("hole")
        if counts.get("restore_pixels"): kinds.append("restore")
        row={
            "id":f"region-brush-{uuid.uuid4().hex[:10]}","enabled":True,"mode":"region_brush_reveal","target_bbox":box,
            "reveal_patch_bbox":box,"reveal_mask_bbox":box,"origin":"region_composite_editor","tool_kind":"region_brush_reveal","owner_transfer_mode":"",
            "source_offset_x":snapshot["dx"],"source_offset_y":snapshot["dy"],"brush_size_px":snapshot["diameter"],
            "transparent_feather_px":snapshot["feather"],"transparent_opacity":snapshot["opacity"],
            "brush_reveal_kind":"+".join(kinds) if kinds else "none",**counts,
        }
        union=np.maximum(tr,ho).astype(np.uint8)
        if restore is not None: union=np.maximum(union,restore)
        return row,union,patch

    def _commit_brush_session(self, *, interactive:bool=True, after_commit=None)->bool:
        if self._link_state.applying: return False
        if self._commit_handler is None:
            if interactive: QMessageBox.warning(self,"无法提交","当前区域编辑器没有连接复核提交器。")
            return False
        try: snapshot=self._brush_commit_snapshot()
        except Exception as exc:
            if interactive: QMessageBox.information(self,"没有涂抹",str(exc))
            return False
        self._link_state.applying=True
        self.target_view.setEnabled(False); self._region_inspector.setEnabled(False)
        self.apply_btn.setText("后台提交涂抹…"); self.apply_btn.setEnabled(False)
        self._trace_region("region_brush_commit_requested",{"target_bbox":snapshot["box"],"pixels":self._pending_brush_pixels()})
        succeeded=[]
        def action():
            row,reveal,patch=self._build_brush_commit(snapshot)
            return row,self._commit_handler(row,reveal,patch)
        worker=PageActionWorker("区域涂抹提交",action,parent=self); self._commit_worker=worker
        def failed(message):
            self._trace_region("region_brush_commit_failed",{"reason":str(message)})
            if interactive: QMessageBox.critical(self,"涂抹揭示提交失败",str(message))
            else: self.selection_status.setText("涂抹提交失败，笔触已保留，请重试。")
        def done(payload):
            row,result=payload
            try:
                final=self._commit_finalize_handler(row,result) if self._commit_finalize_handler is not None else result
                final_path=Path(str(getattr(final,"final_reviewed",final) or self.page_dir/"final_reviewed.png"))
                self._reset_brush_session(final_path); self._applied+=1
                self.selection_status.setText(f"涂抹揭示已提交 · 已叠加 {self._applied} 步")
                self._trace_region("region_brush_commit_succeeded",{"target_bbox":snapshot["box"],"applied_count":self._applied})
                succeeded.append(True)
            except Exception as exc: failed(str(exc))
        def finished():
            self._commit_worker=None; self._link_state.applying=False
            self.target_view.setEnabled(True); self._region_inspector.setEnabled(True)
            self._update_brush_controls()
            if succeeded and after_commit is not None: QTimer.singleShot(0,after_commit)
        worker.done.connect(done); worker.failed.connect(failed); worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater); worker.start()
        return True

    def _apply_brush_session(self):
        self._commit_brush_session(interactive=True)

    def _clear_selection(self):
        self.target_view.clear_selection(emit=True); self.jp_view.clear_selection(); self.source_view.clear_selection()

    def _refine_current_selection(self):
        if len(self.target_view.box())!=4:
            notify(self, "请先画矩形、椭圆或手绘闭合区域，再识别附近闭合边界。", "warning"); return
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

    def can_leave(self):
        if self._closing: return True
        for worker,label in ((getattr(self,"_ocr_worker",None),"OCR"),(getattr(self,"_commit_worker",None),"区域提交"),(getattr(self,"_preview_worker",None),"画笔预览")):
            if worker is not None:
                if hasattr(self,"ocr_status"):
                    self.ocr_status.setText(f"{label}仍在运行，请稍候完成后再关闭。")
                self.selection_status.setText(f"{label}仍在运行，完成后即可返回。")
                return False
        if (self._pending_brush_pixels()>0 or self._ocr_dirty) and not self._discard_confirmed:
            self._leave_notice.show()
            self.selection_status.setText("编辑未提交：请提交，或在下方选择放弃。")
            return False
        self._preview_timer.stop()
        return True

    def embed_in(self, stack):
        self._embedded=True
        self.setParent(stack,Qt.WindowType.Widget)
        self.setMinimumSize(0,0)
        stack.addWidget(self); stack.setCurrentWidget(self)
        self.show()

    def _discard_and_leave(self):
        self._discard_confirmed=True
        self.accept()
        if not self._closing: self._discard_confirmed=False

    def done(self,result):
        if self._closing or not self.can_leave(): return
        # QDialog.done/close/reject may re-enter each other on Cocoa, especially
        # inside QGraphicsProxyWidget. Finish an embedded session explicitly and
        # emit once; do not invoke native dialog teardown for a child QWidget.
        self._closing=True
        self._preview_epoch+=1; self._preview_timer.stop()
        if self._embedded:
            self.hide()
            self.finished.emit(int(result))
        else:
            super().done(result)

    def closeEvent(self,event):
        if not self._closing: self.done(QDialog.DialogCode.Rejected)
        if self._closing: event.accept()
        else: event.ignore()
    def _recognize_ocr(self):
        box=self.target_view.box()
        if len(box)!=4:
            notify(self, "请先选择 OCR 区域。", "warning"); return
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
            self._ocr_dirty=bool(text_value)
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
            notify(self, "请先在中间“当前结果”画出处理区域。", "warning"); return
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
            # Commit the immutable brush snapshot first, then launch the region
            # action. This keeps ordering deterministic without blocking Qt.
            if not self._commit_brush_session(interactive=False,after_commit=self._apply_tool):
                QMessageBox.warning(self,"提交失败","未提交涂抹无法保存；原笔触仍保留。")
            return
        old_text=self.apply_btn.text(); self.apply_btn.setText("后台处理中…"); self._link_state.applying=True; self.apply_btn.setEnabled(False)
        self.target_view.setEnabled(False); self._region_inspector.setEnabled(False)
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
                if key=="region_ocr":
                    self._ocr_dirty=False
                    self.ocr_status.setText("当前 OCR 动作已提交；选区保持，可继续处理")
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
            self.target_view.setEnabled(True); self._region_inspector.setEnabled(True)

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
            notify(self, "请先在右侧高清日文图上框选白色气泡内部区域。", "warning")
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
            notify(self, "请先在右侧高清日文图上拖出要人工补漏的区域。", "warning")
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


