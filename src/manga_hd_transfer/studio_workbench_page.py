from __future__ import annotations

"""Studio workbench page composition and review workflow coordination."""

import time
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtWidgets import (
    QApplication, QWidget, QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFormLayout, QStackedWidget, QScrollArea, QFileDialog, QMessageBox,
    QCheckBox, QLineEdit, QSizePolicy, QSplitter, QPlainTextEdit, QDialog,
)
from .version import __version__
from .gui_components import (
    ImageView, Card, WorkbenchPageRail, StableComboBox, StableSpinBox, StableDoubleSpinBox,
)
from .io_utils import load_json, save_json, write_image
from .review_history import review_history_counts
from .review_action_state import review_action_availability
from .font_catalog import discover_fonts, import_font_to_library, persist_font_expression, font_library_dir
from .review_apply import (
    apply_review_page, apply_manual_ocr_review, clear_ocr_review_blocks, generate_remove_text_preview,
    apply_target_layer_erase_review, reset_target_layer_erase_review, apply_target_layer_restore_review,
    reset_target_layer_restore_review, reset_manual_force_transfer_review, manual_force_auto_evidence_masks,
)
from .workspace import page_id_for_pair, resolve_page_workspace
from .workspace_guard import PageRunGuard
from .schema_compat import as_dict, as_dict_rows, as_list, normalize_project, normalize_overrides, normalize_review_applied
from .result_state import commit_reviewed_result, atomic_copy_file
from .manual_review_service import (
    commit_manual_effect, apply_review_history_step, apply_review_overrides_transaction, run_manual_review_transaction,
)
from .gui_dialogs import confirm_action
from .studio_review_views import RunLogDialog, MaskEditorDialog
from .studio_ocr_editor import (
    OCRBlockEditorDialog, is_ocr_edit_mode, ocr_edit_scope, load_ocr_blocks, upsert_ocr_block,
    delete_ocr_block, save_ocr_blocks, recognize_manual_ocr_block, apply_ocr_edit_blocks,
    preview_ocr_edit_blocks, reset_ocr_edit_blocks,
)
from .studio_region_editor import (
    RegionCompositeDialog, ManualEffectDialog, _json_dict_rows, _manual_effect_unhandled_candidates,
    _manual_effect_ops_for_mode,
)

VERSION = __version__
QComboBox = StableComboBox
QSpinBox = StableSpinBox
QDoubleSpinBox = StableDoubleSpinBox
logger = logging.getLogger(__name__)

_SETTINGS_ORG = "Folirina"
_SETTINGS_APP = "Folirina"
_LEGACY_SETTINGS_ORG = "MangaHDTransfer"
_LEGACY_SETTINGS_APP = "MangaHDTransferStudio"
_THEME_SETTING_KEY = "ui/theme"

def _app_settings() -> QSettings:
    current = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    if not current.contains(_THEME_SETTING_KEY):
        legacy = QSettings(_LEGACY_SETTINGS_ORG, _LEGACY_SETTINGS_APP)
        if legacy.contains(_THEME_SETTING_KEY):
            current.setValue(_THEME_SETTING_KEY, legacy.value(_THEME_SETTING_KEY))
    return current

def _confirm_destructive_action(parent, title: str, message: str, *, confirm_text: str = "确认", cancel_text: str = "取消") -> bool:
    return confirm_action(parent, title, message, confirm_text=confirm_text, cancel_text=cancel_text, destructive=True)


from .studio_review_views import (
    MaskPaintView, ManualTextMaskPreviewDialog, RevealMaskDialog, RegionSelectView, OCRLayoutPreviewView,
)

from .studio_workbench_actions import WorkbenchActionsMixin

class WorkbenchPage(WorkbenchActionsMixin, QWidget):
    def __init__(self, window: "StudioWindow"):
        super().__init__(); self.window=window
        root=QHBoxLayout(self); root.setContentsMargins(10,10,10,10); root.setSpacing(10)
        self._editor_stack=QStackedWidget(); root.addWidget(self._editor_stack)
        split=QSplitter(Qt.Orientation.Horizontal); self._editor_stack.addWidget(split)
        self._region_editor=None

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
            b=QPushButton(label); b.setCheckable(True); b.setObjectName("segmented")
            b.setProperty("workbenchView", True)
            b.clicked.connect(lambda _=False,k=key:self.set_view(k)); toolbar.addWidget(b, i//5, i%5); self.view_buttons.append((b,key))
            toolbar.setColumnStretch(i%5, 1)
        # Layout parity with the supplied transfer-desk design: the artwork
        # remains above the ten view choices, not pushed below two toolbar rows.
        # Keep the same buttons, signal wiring and view keys; only their position
        # in the native Qt layout changes.
        self.page_subtitle=QLabel("等待配对页面")
        self.page_subtitle.setObjectName("workbenchPageSubtitle")
        preview.layout.addWidget(self.page_subtitle)
        # The ten always-fit overview views do not need a 3200px decode. A
        # 2048px preview keeps text legible at desktop fit scale while reducing
        # per-view decode/paint cost and allowing a larger bounded LRU.
        self.image=ImageView(max_decode_side=2048); preview.layout.addWidget(self.image,1)
        preview.layout.addLayout(toolbar)
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
        self.detector_size=QSpinBox(); self.detector_size.setRange(512,4096); self.detector_size.setSingleStep(16); self.detector_size.setValue(int(self.window.state.config.bubbles.koharu_layout_shape))
        self.clear_dilate=QSpinBox(); self.clear_dilate.setRange(0,20); self.clear_dilate.setValue(int(self.window.state.config.masking.max_dilation_px)); self.clear_dilate.setSuffix(" px")
        self.inpaint_backend=QComboBox(); self.inpaint_backend.addItem("自动","auto"); self.inpaint_backend.addItem("纯色纸面","solid"); self.inpaint_backend.addItem("OpenCV 修复","opencv"); self.inpaint_backend.addItem("LaMa（兼容旧 runner）","lama"); self.inpaint_backend.addItem("LaMa Manga","lama_manga"); self.inpaint_backend.addItem("AOT Inpainting","aot_inpainting"); self.inpaint_backend.addItem("FLUX.2 Klein","flux2_klein"); self.inpaint_backend.addItem("RORem Mixed","rorem_mixed")
        sf.addRow("检测策略",self.detector_strategy); sf.addRow("主检测器",self.primary_detector); sf.addRow("辅助检测器",aux_widget); sf.addRow("边界精修",self.sam2_refine); sf.addRow("主检测器输入尺寸",self.detector_size); sf.addRow("清除 mask 最大扩张",self.clear_dilate); sf.addRow("去字修复",self.inpaint_backend); stages.layout.addLayout(sf)
        # Long review actions are arranged as a stable two-column grid.  Keeping
        # three unrelated controls on one narrow row caused macOS to elide labels
        # such as “仅擦 TARGET 日文层…” even though the whole-page scale itself was
        # correct.  Rows now expand to the inspector width before outer scaling.
        # These are review actions, not configuration.  Keep their QWidget
        # instances/signals, but show them under the Review/QA inspector tab.
        review_mask_tools=Card("蒙版与 TARGET 复核工具", "修改本页蒙版、预览和恢复 TARGET；参数页只保留检测、清除和写入的配置。")
        action_grid=QGridLayout(); action_grid.setContentsMargins(0,0,0,0); action_grid.setHorizontalSpacing(7); action_grid.setVerticalSpacing(7)
        action_grid.setColumnStretch(0,1); action_grid.setColumnStretch(1,1)
        self.edit_clear_mask=QPushButton("编辑自动清除蒙版…"); self.remove_text_only=QPushButton("预览自动去字")
        action_grid.addWidget(self.edit_clear_mask,0,0); action_grid.addWidget(self.remove_text_only,0,1)
        self.apply_mask_review=QPushButton("应用蒙版到最终结果"); self.apply_mask_review.setObjectName("softPrimary"); self.reset_clear_mask=QPushButton("恢复自动蒙版")
        action_grid.addWidget(self.apply_mask_review,1,0); action_grid.addWidget(self.reset_clear_mask,1,1)
        self.force_transfer_mask=QPushButton("人工强制迁移蒙版…"); self.force_transfer_mask.setObjectName("softPrimary"); self.reset_force_transfer_mask=QPushButton("清空强制迁移蒙版")
        action_grid.addWidget(self.force_transfer_mask,2,0); action_grid.addWidget(self.reset_force_transfer_mask,2,1)
        review_mask_tools.layout.addLayout(action_grid)
        self.force_transfer_mask.setToolTip("自动漏识别或自动区域本身有误时使用：红色人工层和蓝色 OCR/自动蒙版都能直接涂抹/消除。蓝色修订会作为本页自动蒙版覆盖保存；重新处理后继续生效。自动也完全漏掉时仍可纯人工重跑。")
        self.reset_force_transfer_mask.setToolTip("删除人工强制迁移蒙版，并恢复第一次使用该工具前冻结的稳定结果。")
        self.target_erase_mode=QComboBox(); self.target_erase_mode.addItem("智能恢复 TARGET", "auto"); self.target_erase_mode.addItem("纯白涂抹", "pure_white"); self.target_erase_mode.setToolTip("智能恢复：白底用 TARGET 纸色、复杂背景用 inpaint；纯白涂抹：仍只对 TARGET 日文层生效，中文保护区保持不变。")
        self.target_erase_mode.setMinimumContentsLength(14); self.target_erase_mode.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        review_mask_tools.layout.addWidget(self.target_erase_mode)
        target_grid=QGridLayout(); target_grid.setContentsMargins(0,0,0,0); target_grid.setHorizontalSpacing(7); target_grid.setVerticalSpacing(7); target_grid.setColumnStretch(0,1); target_grid.setColumnStretch(1,1)
        self.target_layer_erase=QPushButton("仅擦 TARGET 日文层…"); self.target_layer_erase.setObjectName("softPrimary"); self.reset_target_layer_erase=QPushButton("清空 TARGET 擦除")
        self.target_layer_restore=QPushButton("恢复 TARGET 日文层 / 擦蒙版…"); self.target_layer_restore.setObjectName("softPrimary"); self.reset_target_layer_restore=QPushButton("清空 TARGET 恢复")
        target_grid.addWidget(self.target_layer_erase,0,0); target_grid.addWidget(self.reset_target_layer_erase,0,1)
        target_grid.addWidget(self.target_layer_restore,1,0); target_grid.addWidget(self.reset_target_layer_restore,1,1)
        review_mask_tools.layout.addLayout(target_grid)
        self.target_layer_erase.setToolTip("最终收尾画笔：可刷日文残字、黑点、短线和符号。只重建 TARGET 母版层，中文迁移/重排/人工补漏图层自动保护。")
        self.target_layer_restore.setToolTip("反向画笔：对不应显示中文的区域，直接恢复 TARGET 原始日文图层与背景。")
        stage_note=QLabel("自动没有识别到、识别不完整，或者自动蒙版本身过大/错误时，使用“人工强制迁移蒙版”：红色人工层与蓝色 OCR/自动层都可直接编辑。自动区域会先收紧为真实文字像素，避免整块气泡/画面被当成清除区；只剩零星残字时再用“仅擦 TARGET 日文层”。")
        stage_note.setObjectName("quiet"); stage_note.setWordWrap(True)
        review_mask_tools.layout.addWidget(stage_note)
        sl.addWidget(stages)
        rl.addWidget(review_mask_tools)

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
        frow=QHBoxLayout(); self.manual_font=QLineEdit(); self.manual_font.setPlaceholderText("字体：留空=当前/全局；外部字体可导入 Folirina 字体库；支持 A;B;C 候选链")
        self.manual_font.setToolTip(f"导入字体会复制到 Folirina 持久字体库：{font_library_dir()}")
        self.manual_font_pick=QPushButton("导入"); frow.addWidget(self.manual_font,1); frow.addWidget(self.manual_font_pick); manual.layout.addLayout(frow)
        flib=QHBoxLayout(); self.manual_font_catalog=QComboBox(); self.manual_font_catalog.addItem("选择字体库字体", "")
        for frow_item in discover_fonts(limit=180): self.manual_font_catalog.addItem(str(frow_item.get("name") or Path(str(frow_item.get("path") or "")).stem), str(frow_item.get("path") or ""))
        self.manual_font_refresh=QPushButton("刷新"); self.manual_font_refresh.setObjectName("compactAction"); self.manual_font_refresh.setMaximumWidth(64)
        flib.addWidget(QLabel("字体库")); flib.addWidget(self.manual_font_catalog,1); flib.addWidget(self.manual_font_refresh); manual.layout.addLayout(flib)
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
                       self.manual_effect_candidate_target, self.manual_orientation, self.manual_break_mode, self.manual_layout_mode, self.manual_font_preset, self.manual_font_catalog, self.manual_font, self.manual_font_size, self.manual_columns, self.manual_line_spacing, self.manual_text, self.reprocess_current]:
            widget.setMinimumWidth(0)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, widget.sizePolicy().verticalPolicy())
        for combo in [self.local, self.sr, self.fidelity, self.detector_strategy, self.primary_detector, self.inpaint_backend, self.target_erase_mode, self.manual_target, self.manual_effect_candidate_target, self.manual_orientation, self.manual_break_mode, self.manual_layout_mode, self.manual_font_preset, self.manual_font_catalog]:
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        for button in [self.sr_pick, self.edit_clear_mask, self.remove_text_only, self.apply_mask_review,
                       self.reset_clear_mask, self.force_transfer_mask, self.reset_force_transfer_mask,
                       self.target_layer_erase, self.reset_target_layer_erase, self.target_layer_restore, self.reset_target_layer_restore,
                       self.add_manual_effect, self.add_manual_effect_candidate, self.undo_manual_effect, self.manual_apply,
                       self.candidate_accept, self.candidate_restore, self.manual_reset, self.manual_undo, self.manual_redo, self.manual_font_pick, self.manual_font_refresh, self.open_ocr_block_editor, self.reset_ocr_blocks, self.reprocess_current]:
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
        # The supplied design starts with the page-thumbnail rail folded. It
        # remains fully available through the same visible toggle button.
        self._page_rail_user_hidden=True
        self.prev_page.clicked.connect(lambda: self._move_page(-1)); self.next_page.clicked.connect(lambda: self._move_page(1)); self.inspector_toggle.toggled.connect(self._toggle_inspector); self.page_rail_toggle.toggled.connect(self._toggle_page_rail)
        self.page_rail.collapse_requested.connect(lambda: self.page_rail_toggle.setChecked(True))
        self.page_rail_toggle.setChecked(True)
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
        self.manual_font_refresh.clicked.connect(lambda: self._refresh_manual_font_catalog(force=True)); self.manual_font_catalog.currentIndexChanged.connect(self._apply_manual_catalog_font); self.manual_font.editingFinished.connect(self._persist_manual_font_expression)
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
        # Existing projects may carry distinct detector/transfer settings.  Read
        # the authoritative state before a user signal can write any defaults.
        self.refresh_mode_controls()
        self.refresh_detector_controls()
        for w in [self.paired,self.skip_ocr,self.pixel_exact,self.full_bubble_patch,self.preserve_border,self.blur_guard,self.preserve_source_layout]: w.toggled.connect(self._sync)
        self.local.currentTextChanged.connect(self._sync); self.fidelity.currentIndexChanged.connect(self._sync); self.sr.currentTextChanged.connect(self._sync); self.iou.valueChanged.connect(self._sync); self.coverage.valueChanged.connect(self._sync)
        self.detector_strategy.currentIndexChanged.connect(self._sync)
        self.primary_detector.currentIndexChanged.connect(self._on_primary_detector_changed)
        self.detector_size.valueChanged.connect(self._sync)
        self.clear_dilate.valueChanged.connect(self._sync)
        self.inpaint_backend.currentIndexChanged.connect(self._sync)
        self.expand_direct_range.toggled.connect(self._sync)
        for cb in [self.aux_geometry_white,self.aux_mangalens,self.aux_rtdetr,self.aux_ysg,
                   self.aux_koharu,self.aux_ctd,self.aux_sidecar,self.sam2_refine]:
            cb.toggled.connect(self._sync)
        self.sr_model.editingFinished.connect(self._sync)
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
        QTimer.singleShot(0, lambda: self._update_responsive_workbench(force=True))

    def _toggle_inspector(self, hidden: bool):
        self.side.setVisible(not bool(hidden))
        self.inspector_toggle.setText("显示参数栏" if hidden else "隐藏参数栏")
        if not hidden:
            QTimer.singleShot(0,self._update_responsive_workbench)

    def _update_responsive_workbench(self, *, force: bool = False):
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
        if self._compact_layout is compact and not force:
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
            rail_w=0 if self._page_rail_user_hidden else min(190,max(150,int(w*0.13)))
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
        mode = str(self.window.state.config.transfer.mode or "direct_patch")
        widgets = [self.paired,self.skip_ocr,self.pixel_exact,self.full_bubble_patch,
                   self.preserve_border,self.blur_guard,self.preserve_source_layout,
                   self.fidelity,self.local,self.sr,self.sr_model,self.sr_pick,
                   self.iou,self.coverage]
        # These controls address candidate/pixel modes only.  Unsupported
        # pipelines must not show clickable but completely ineffective knobs.
        for w in widgets:
            w.setEnabled(c is not None)
        self.expand_direct_range.setEnabled(mode == "direct_patch")
        if c is None:
            return
        widgets = [w for w in widgets if w is not self.sr_pick] + [self.expand_direct_range]
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
            if mode == "direct_patch":
                self.expand_direct_range.setChecked(bool(getattr(self.window.state.config.direct_patch,"source_direct_expand_candidate_range",False)))
        finally:
            for w in widgets: w.blockSignals(False)

    @staticmethod
    def _primary_size_field(primary: str) -> str:
        return {
            "koharu_layout": "koharu_layout_shape",
            "mangalens": "mangalens_imgsz",
            "rtdetr_v2": "rtdetr_imgsz",
        }.get(str(primary), "koharu_layout_shape")

    def refresh_detector_controls(self):
        """Restore global detector/inpaint options after loading/switching project.

        Mode-specific controls are restored separately; neither refresh may
        call _sync or overwrite the project's saved config.
        """
        cfg=self.window.state.config
        b=cfg.bubbles
        widgets=[self.detector_strategy,self.primary_detector,self.detector_size,
                 self.aux_geometry_white,self.aux_mangalens,self.aux_rtdetr,
                 self.aux_ysg,self.aux_koharu,self.aux_ctd,self.aux_sidecar,
                 self.sam2_refine,self.clear_dilate,self.inpaint_backend]
        state=[w.blockSignals(True) for w in widgets]
        try:
            for widget,value in ((self.detector_strategy,str(b.detector_strategy)),
                                 (self.primary_detector,str(b.primary_detector)),
                                 (self.inpaint_backend,str(cfg.inpainting.backend))):
                idx=widget.findData(value)
                if idx>=0: widget.setCurrentIndex(idx)
            selected=set(b.auxiliary_detectors or [])
            for key,widget in (("geometry_white",self.aux_geometry_white),
                               ("mangalens",self.aux_mangalens),
                               ("rtdetr_v2",self.aux_rtdetr),
                               ("ysg_obb",self.aux_ysg),
                               ("koharu_layout",self.aux_koharu),
                               ("ctd_sidecar",self.aux_ctd),
                               ("sidecar",self.aux_sidecar)):
                widget.setChecked(key in selected)
            self.sam2_refine.setChecked(bool(b.sam2_refine_enabled))
            self.clear_dilate.setValue(int(cfg.masking.max_dilation_px))
            self.detector_size.setMinimum(512 if b.primary_detector == "koharu_layout" else 640)
            self.detector_size.setValue(int(getattr(b,self._primary_size_field(b.primary_detector))))
        finally:
            for widget,previous in zip(widgets,state):widget.blockSignals(previous)
        self._update_detector_control_availability()

    def _update_detector_control_availability(self):
        primary=self.primary_detector.currentData() or "koharu_layout"
        aux_enabled=self.detector_strategy.currentData()!="primary_only"
        for key,cb in (("geometry_white",self.aux_geometry_white),
                       ("mangalens",self.aux_mangalens),
                       ("rtdetr_v2",self.aux_rtdetr),
                       ("ysg_obb",self.aux_ysg),
                       ("koharu_layout",self.aux_koharu),
                       ("ctd_sidecar",self.aux_ctd),
                       ("sidecar",self.aux_sidecar)):
            cb.setEnabled(aux_enabled and key!=primary)
        # SAM 2 is an optional geometry refiner, not an auxiliary detector.
        # Keep it accessible even when detector_strategy == primary_only;
        # otherwise the UI said "disabled" while its saved True still ran.
        self.sam2_refine.setEnabled(True)

    def _on_primary_detector_changed(self, _index: int = -1):
        # Save the size of the *new* primary only.  Without this reload,
        # switching Koharu -> RT-DETR silently copied 1152 into RT-DETR's
        # previously independent 640px configuration, and vice versa.
        primary=self.primary_detector.currentData() or "koharu_layout"
        b=self.window.state.config.bubbles
        previous=self.detector_size.blockSignals(True)
        try:
            self.detector_size.setMinimum(512 if primary == "koharu_layout" else 640)
            self.detector_size.setValue(int(getattr(b,self._primary_size_field(primary))))
        finally:self.detector_size.blockSignals(previous)
        self._sync()

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
        self._update_detector_control_availability()
        setattr(bcfg,self._primary_size_field(bcfg.primary_detector),int(self.detector_size.value()))
        self.window.state.config.masking.max_dilation_px=int(self.clear_dilate.value())
        self.window.state.config.inpainting.backend=self.inpaint_backend.currentData() or "auto"

    def refresh(self):
        s=self.window.state
        self.refresh_mode_controls()
        self.refresh_detector_controls()
        total=len(s.pairs)
        if not total:
            self.page_rail.set_pages([], 0)
            self._view_paths = {}
            self.image.set_image(None); self.page_caption.setText("未选择页面"); self.page_counter.setText("0 / 0")
            self.page_subtitle.setText("等待配对页面")
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
        self.page_subtitle.setText(f"{Path(pair.target_path).stem} · 高清日文 ↔ 旧中文版 · 已配对")
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
        # Keep a snapshot for cheap 10-view switching.  It is invalidated by
        # the next full refresh (page change, review commit or processing result).
        self._view_paths = paths
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
                elif status == "integrity_failed":
                    self.view_status.setText(f"结果完整性检查未通过 · 模式 {mode_name} · 建议查看日志后重新处理")
                elif status == "partial":
                    self.view_status.setText(f"部分区域已完成 · 仍有区域失败/待复核 · 模式 {mode_name}")
                elif status == "degraded":
                    self.view_status.setText(f"已降级完成 · 当前结果来自安全回退路径 · 模式 {mode_name}")
                elif status == "noop":
                    self.view_status.setText(f"模式已执行但没有安全可发布区域 · 当前稳定结果已保留 · 模式 {mode_name}")
                elif status == "skipped":
                    self.view_status.setText(f"本页按页面规则跳过 · 已保留稳定输出 · 模式 {mode_name}")
                else:
                    suffix = f" · 模式 {mode_name}" + (f" / {strategy_name}" if strategy_name else "")
                    self.view_status.setText((f"已恢复已有结果 · {origin} · 可继续人工补漏{suffix}" if origin else f"已同步到当前页{suffix}"))
        elif self.current_view in {"result","review","mask","clear_mask","chinese_layer","removed","target_erase","target_restore"}:
            self.view_status.setText("本页尚无该输出")
        else:
            self.view_status.setText("")




