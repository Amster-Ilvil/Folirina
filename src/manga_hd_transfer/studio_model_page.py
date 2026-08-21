from __future__ import annotations

"""Recognition / registration / model-management page.

This page owns presentation and user selections only. Runtime installation and
download work remains in ``gui_workers``.
"""

import logging
import platform
import time
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QTimer, QSettings, QUrl
from PySide6.QtGui import QPixmap, QImageReader, QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFormLayout, QScrollArea, QFileDialog, QMessageBox, QSpinBox,
    QDoubleSpinBox, QLineEdit, QProgressBar, QButtonGroup, QCheckBox,
)

from .gui_components import Card, OptionRow, StableComboBox, StableSpinBox, StableDoubleSpinBox
from .gui_workers import ComponentProbeWorker, ModelDownloadWorker, DependencyInstallWorker, ModelNetworkProbeWorker
from .model_downloads import (
    apply_config_updates, model_home, model_local_paths, import_builtin_model,
    apply_download_network_settings, paddle_profile_marker_status,
)
from .dependency_install import missing_dependency_modules, dependency_summary
from .paddle_profiles import PADDLE_MODEL_PROFILES, profile_label, backend_profile_key
from .runtime_preflight import plan_runtime_requirements, pending_model_requirements
from .workspace import resolve_page_workspace
from .platform_support import desktop_platform_summary, platform_family
from .mode_contracts import get_mode_contract

QComboBox = StableComboBox
QSpinBox = StableSpinBox
QDoubleSpinBox = StableDoubleSpinBox

logger = logging.getLogger(__name__)


def _model_settings() -> QSettings:
    current = QSettings("Folirina", "Folirina")
    legacy = QSettings("MangaHDTransfer", "MangaHDTransferStudio")
    keys = (
        "models/hf_source", "models/proxy", "models/ca_bundle",
        "models/paddle_source", "models/paddle_profile",
        "models/paddle_det_name", "models/paddle_rec_name",
    )
    for key in keys:
        if not current.contains(key) and legacy.contains(key):
            current.setValue(key, legacy.value(key))
    return current


class ModelPage(QWidget):
    def __init__(self, window: "StudioWindow"):
        super().__init__(); self.window = window
        self._probe_cache = None
        self._probe_cache_at = 0.0
        self._probe_worker = None
        self._ocr_lock_enabled_state: bool | None = None
        self._model_download_worker = None
        self._model_download_key = ""
        self._dependency_worker = None
        self._dependency_key = ""
        self._network_probe_worker = None
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        page_scroll = QScrollArea(self); page_scroll.setWidgetResizable(True); page_scroll.setFrameShape(QFrame.Shape.NoFrame)
        page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page_scroll.setStyleSheet("QScrollArea { border:0; background:transparent; }")
        page_body = QWidget(); page_body.setObjectName("modelPageBody")
        root = QVBoxLayout(page_body); root.setContentsMargins(16,16,16,16); root.setSpacing(12)
        page_scroll.setWidget(page_body); outer.addWidget(page_scroll, 1)

        hero = Card(blue=True)
        hero.setObjectName("recognitionHero")
        hero_row = QHBoxLayout(); hero_row.setSpacing(14)
        hero_left = QVBoxLayout(); hero_left.setSpacing(8)
        hero_title_row = QHBoxLayout(); hero_title_row.setSpacing(12)
        hero_icon = QLabel("A"); hero_icon.setObjectName("heroIcon"); hero_icon.setFixedSize(52,52); hero_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hero_copy = QVBoxLayout(); hero_copy.setSpacing(3)
        hero_title = QLabel("智能识别 · 精准配准 · 高质量 OCR 回填"); hero_title.setObjectName("heroTitle")
        hero_desc = QLabel("自动识别台词气泡与文字区域，保持 SOURCE / TARGET 气泡身份一致，避免跨气泡错绑。")
        hero_desc.setObjectName("hint"); hero_desc.setWordWrap(True)
        hero_copy.addWidget(hero_title); hero_copy.addWidget(hero_desc)
        hero_title_row.addWidget(hero_icon); hero_title_row.addLayout(hero_copy,1)
        hero_left.addLayout(hero_title_row)
        stats = QHBoxLayout(); stats.setSpacing(0)
        self.hero_page_count = QLabel("页数\n—"); self.hero_page_count.setObjectName("heroStat")
        self.hero_ocr_state = QLabel("OCR\n待检测"); self.hero_ocr_state.setObjectName("heroStat")
        self.hero_reg_state = QLabel("配准\n待检测"); self.hero_reg_state.setObjectName("heroStat")
        self.hero_map_state = QLabel("气泡映射\n安全绑定"); self.hero_map_state.setObjectName("heroStat")
        for widget in (self.hero_page_count,self.hero_ocr_state,self.hero_reg_state,self.hero_map_state):
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter); stats.addWidget(widget,1)
        hero_left.addLayout(stats)
        self.hero_status = QLabel("✓ 每个目标气泡只接受其对应 SOURCE 气泡/Region 的文字；不确定映射会跳过，不会借用邻居文字。")
        self.hero_status.setObjectName("successHint"); self.hero_status.setWordWrap(True); hero_left.addWidget(self.hero_status)
        hero_row.addLayout(hero_left,5)

        preview_box = QFrame(); preview_box.setObjectName("heroPreviewFrame")
        preview_lay = QVBoxLayout(preview_box); preview_lay.setContentsMargins(8,8,8,8); preview_lay.setSpacing(5)
        preview_head = QHBoxLayout(); preview_head.addWidget(QLabel("当前页 · 气泡映射预览")); preview_head.addStretch(1)
        self.hero_mapping_badge = QLabel("映射保护开启"); self.hero_mapping_badge.setObjectName("successBadge"); preview_head.addWidget(self.hero_mapping_badge)
        preview_lay.addLayout(preview_head)
        self.hero_preview = QLabel("选择项目后显示当前 TARGET / 结果页")
        self.hero_preview.setObjectName("heroPreview"); self.hero_preview.setAlignment(Qt.AlignmentFlag.AlignCenter); self.hero_preview.setMinimumSize(360,165); self.hero_preview.setMaximumHeight(220)
        self.hero_preview.setScaledContents(False); preview_lay.addWidget(self.hero_preview,1)
        hero_row.addWidget(preview_box,6)
        hero.layout.addLayout(hero_row)
        root.addWidget(hero)

        pipeline = Card("当前视觉管线", "明确区分职责：配准负责 SOURCE→TARGET 坐标；主检测器负责第一语义判断；OCR 只在模式允许时识字；Inpainting 只处理既有清字 Mask。")
        pipeline_row = QHBoxLayout(); pipeline_row.setSpacing(8)
        self.pipeline_registration = QLabel("配准 · —"); self.pipeline_registration.setObjectName("stageChip")
        self.pipeline_detector = QLabel("主检测 · —"); self.pipeline_detector.setObjectName("stageChip")
        self.pipeline_ocr = QLabel("OCR · —"); self.pipeline_ocr.setObjectName("stageChip")
        self.pipeline_inpaint = QLabel("修补 · —"); self.pipeline_inpaint.setObjectName("stageChip")
        for chip in (self.pipeline_registration,self.pipeline_detector,self.pipeline_ocr,self.pipeline_inpaint):
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter); pipeline_row.addWidget(chip,1)
        pipeline.layout.addLayout(pipeline_row)
        self.pipeline_note = QLabel("执行顺序：页面配准 → 主检测器 → 模式自己的迁移 / OCR → 清字与写入。辅助检测器只按 Detector Policy 触发。")
        self.pipeline_note.setObjectName("quiet"); self.pipeline_note.setWordWrap(True); pipeline.layout.addWidget(self.pipeline_note)
        root.addWidget(pipeline)

        self.ocr = self._choice_card("OCR", "只负责识别文字内容，不负责页面配准，也不取代 Koharu 布局。仅 Auto / 精准蒙版+OCR / OCR重排可能调用；安全的视觉路线会自动跳过 OCR。SOURCE 与 TARGET 可单独覆盖。", [
            ("Apple Live Text（推荐）", "apple", "系统 Vision；无需下载本地模型"),
            ("Apple 快捷指令", "apple_shortcut", "直接调用 ExtractText / 从图像中提取文字"),
            ("PaddleOCR v6 Medium", "paddle_v6_medium", "传统 detection + recognition · 质量优先"),
            ("PaddleOCR v6 Small", "paddle_v6_small", "传统 detection + recognition · 更快更省资源"),
            ("PaddleOCR-VL 1.6", "paddle_vl_16", "独立 VLM 文档解析接口；不是 OCR v6 的档位"),
            ("Manga OCR", "manga_ocr", "日文漫画专用 · Koharu Layout 分区后逐区识别"),
            ("Baberu OCR", "baberu_ocr", "ONNX 轻量漫画 OCR · 日/中/英 · Layout 分区"),
            ("48px AR OCR", "ocr48px", "经典 Manga Image Translator 48px 自回归 OCR；原生独立 Torch/MPS 运行，外部 runner 仅作兼容回退"),
            ("PP-StructureV3", "paddle_structure_v3", "独立版面结构解析接口"),
            ("外部 OCR JSON / MD", "external", "导入云端/别台机器 OCR 结果，不在本机跑模型"),
            ("Sidecar", "sidecar", "每页 .ocr.json 外部结果"),
            ("关闭", "none", "纯蒙版替换可用"),
        ], self._set_ocr)
        self.reg = self._choice_card("页面配准", "所有自动迁移模式都会先做 SOURCE→TARGET 几何对齐，然后才运行 Koharu Global Layout。Auto / SIFT 会按成本逐级升级；显式 LightGlue / LoFTR 会直接使用所选深度后端（缓存命中除外）。", [
            ("Auto / SIFT", "auto", "离线默认，优先稳健"),
            ("LightGlue", "lightglue", "SIFT / ALIKED / DISK"),
            ("LoFTR", "loftr", "困难页 dense fallback"),
        ], self._set_reg)
        self.bubble = self._choice_card("主检测器", "所有模式共用同一个主检测器；它永远先运行、结果优先。Koharu Layout 能同时给出 text / SFX / bubble / panel 实例分割，因此仍是默认推荐。", [
            ("Koharu Layout RF-DETR Seg 2XL（推荐）", "koharu_layout", "完整漫画语义实例分割；作为主检测器时启用 ALLOW / PROTECT / UNKNOWN Authority Map"),
            ("MangaLens", "mangalens", "YOLO11 气泡实例分割；作为主检测器时不具备 Koharu 的 panel 保护语义"),
            ("Comic Translate RT-DETR-v2", "rtdetr_v2", "文字/气泡候选检测；作为主检测器时优先于所有辅助结果"),
        ], self._set_primary_detector)

        detector_policy = Card("检测策略 / 辅助检测器", "主检测器单选；辅助检测器可多选。按需辅助只在主结果不足时触发；始终辅助用于模型组合对比，但最终仍保持主检测器优先。")
        policy_form = QFormLayout(); policy_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.detector_strategy = QComboBox();
        self.detector_strategy.addItem("仅主检测器", "primary_only")
        self.detector_strategy.addItem("主检测器 + 按需辅助（推荐）", "primary_conditional_aux")
        self.detector_strategy.addItem("主检测器 + 始终辅助", "primary_plus_aux")
        policy_form.addRow("检测策略", self.detector_strategy)
        aux_box = QWidget(); aux_lay = QVBoxLayout(aux_box); aux_lay.setContentsMargins(0,0,0,0); aux_lay.setSpacing(4)
        self.detector_aux_checks = {}
        for key, label in [
            ("geometry_white", "白色容器 / 结构几何"),
            ("mangalens", "MangaLens"),
            ("rtdetr_v2", "RT-DETR-v2"),
            ("ysg_obb", "YSG YOLO OBB（框外 / 开放 / 倾斜文字）"),
            ("koharu_layout", "Koharu Layout（主检测器不是 Koharu 时可作为辅助）"),
            ("ctd_sidecar", "CTD 外部像素 Mask / Sidecar"),
            ("sidecar", "Sidecar 外部蒙版"),
        ]:
            cb = QCheckBox(label); self.detector_aux_checks[key] = cb; aux_lay.addWidget(cb)
        policy_form.addRow("辅助检测器", aux_box)
        self.sam2_refine = QCheckBox("必要时使用 SAM 2 / 2.1 做边界/缺口精修")
        policy_form.addRow("边界精修", self.sam2_refine)
        detector_policy.layout.addLayout(policy_form)
        self.detector_policy_card = detector_policy
        self.detector_strategy.currentIndexChanged.connect(self._detector_policy_changed)
        for cb in self.detector_aux_checks.values(): cb.toggled.connect(self._detector_policy_changed)
        self.sam2_refine.toggled.connect(self._detector_policy_changed)
        self.inpaint = self._choice_card("消字 / Inpainting", "AI 消字只处理既有清字 Mask；默认 Auto 仍保持原来的轻量安全路径。", [
            ("Auto（现有安全路径）", "auto", "白底优先阈值清除；复杂背景 OpenCV/已配置后端"),
            ("LaMa Manga", "lama_manga", "漫画/动画微调 LaMa；本地 runner"),
            ("AOT Inpainting", "aot_inpainting", "AOT-GAN 直接修补；本地 runner"),
            ("FLUX.2 Klein", "flux2_klein", "生成式修补 · Prompt · 大模型按需加载"),
            ("RORem Mixed", "rorem_mixed", "漫画向 SDXL inpainting · 正/负 Prompt"),
            ("OpenCV Telea", "opencv", "轻量离线回退"),
        ], self._set_inpaint)

        top_cards = QGridLayout(); top_cards.setHorizontalSpacing(12); top_cards.setVerticalSpacing(12)
        top_cards.addWidget(self.ocr, 0, 0, 1, 2)
        top_cards.addWidget(self.reg, 1, 0)
        top_cards.addWidget(self.bubble, 1, 1)
        top_cards.addWidget(self.detector_policy_card, 2, 0, 1, 2)
        top_cards.addWidget(self.inpaint, 3, 0, 1, 2)
        top_cards.setColumnStretch(0, 1); top_cards.setColumnStretch(1, 1)
        root.addLayout(top_cards)

        self._ocr_card_subtitle_enabled = "OCR 只识字，不决定页面布局；Koharu 先给出布局/保护语义。仅 Auto / 精准蒙版+OCR / OCR重排按需要调用 OCR，SOURCE 与 TARGET 可单独覆盖。"
        self._ocr_card_subtitle_disabled = "当前迁移方式是 0 OCR 路线：OCR 选项已锁定，历史 OCR 配置不会参与处理，也不会触发 OCR 模型预准备/下载。切回 Auto / 精准蒙版+OCR / OCR重排后恢复。"
        self._ocr_settings_subtitle_enabled = "不知道选哪一个时：普通漫画优先 Apple Live Text 或 PaddleOCR v6 Medium；本机跑不了模型时直接切到“外部 OCR JSON / MD”。"
        self._ocr_settings_subtitle_disabled = "当前迁移方式不使用 OCR：OCR 回填、SOURCE/TARGET 覆盖与 48px 回退均锁定；原有选择只保留配置，不会加载/下载 OCR，切回 OCR 模式后自动恢复。"
        settings = Card("OCR 回填来源与下载设置", self._ocr_settings_subtitle_enabled)
        self.ocr_settings_card = settings
        net = QGridLayout(); net.setHorizontalSpacing(6); net.setVerticalSpacing(6)
        net.addWidget(QLabel("下载源"),0,0)
        self.model_source=QComboBox(); self.model_source.addItem("自动：官方 → 国内镜像","auto"); self.model_source.addItem("仅官方直连","official"); self.model_source.addItem("国内 HF 镜像优先","mirror"); net.addWidget(self.model_source,0,1,1,2)
        net.addWidget(QLabel("HTTP/HTTPS 代理"),1,0)
        self.model_proxy=QLineEdit(); self.model_proxy.setPlaceholderText("可选，例如 http://127.0.0.1:7890"); net.addWidget(self.model_proxy,1,1)
        self.apply_model_network=QPushButton("应用"); self.test_model_network=QPushButton("网络诊断"); net.addWidget(self.apply_model_network,1,2); net.addWidget(self.test_model_network,1,3)
        net.addWidget(QLabel("自定义 CA"),2,0)
        ca_box=QHBoxLayout(); ca_box.setSpacing(4)
        self.model_ca_bundle=QLineEdit(); self.model_ca_bundle.setPlaceholderText("可选：公司/校园 HTTPS 代理的 PEM 证书")
        self.model_ca_pick=QPushButton("选择…"); self.model_ca_pick.setMaximumWidth(72)
        ca_box.addWidget(self.model_ca_bundle,1); ca_box.addWidget(self.model_ca_pick)
        net.addLayout(ca_box,2,1,1,3)
        net.addWidget(QLabel("Paddle 模型源"),3,0)
        self.paddle_model_source=QComboBox()
        self.paddle_model_source.addItem("自动重试：ModelScope → BOS → AIStudio → HF","auto")
        self.paddle_model_source.addItem("ModelScope","modelscope")
        self.paddle_model_source.addItem("百度 BOS","bos")
        self.paddle_model_source.addItem("AIStudio","aistudio")
        self.paddle_model_source.addItem("Hugging Face","huggingface")
        net.addWidget(self.paddle_model_source,3,1,1,3)
        net.addWidget(QLabel("Paddle 引擎 / 模型"),4,0)
        self.paddle_model_profile=QComboBox()
        for row in PADDLE_MODEL_PROFILES:
            if row.key == "legacy_v5_auto":
                continue
            self.paddle_model_profile.addItem(row.label, row.key)
        self.paddle_model_profile.setToolTip("PaddleOCR v6、PaddleOCR-VL 1.6、PP-StructureV3 是互相独立的运行接口；下载/预热只准备当前选择。")
        net.addWidget(self.paddle_model_profile,4,1,1,3)
        self.paddle_det_label=QLabel("自定义 Det 名")
        self.paddle_det_name=QLineEdit(); self.paddle_det_name.setPlaceholderText("例如 PP-OCRv6_medium_det")
        self.paddle_rec_label=QLabel("自定义 Rec 名")
        self.paddle_rec_name=QLineEdit(); self.paddle_rec_name.setPlaceholderText("例如 PP-OCRv6_medium_rec")
        net.addWidget(self.paddle_det_label,5,0); net.addWidget(self.paddle_det_name,5,1,1,3)
        net.addWidget(self.paddle_rec_label,6,0); net.addWidget(self.paddle_rec_name,6,1,1,3)
        self.paddle_profile_status=QLabel("已缓存引擎：检测中…"); self.paddle_profile_status.setObjectName("quiet"); self.paddle_profile_status.setWordWrap(True)
        net.addWidget(self.paddle_profile_status,7,0,1,4)

        role_choices = [
            ("跟随上方主选择", "inherit"),
            ("Apple Live Text", "apple"),
            ("PaddleOCR v6 Medium", "paddle_v6_medium"),
            ("PaddleOCR v6 Small", "paddle_v6_small"),
            ("PaddleOCR-VL 1.6", "paddle_vl_16"),
            ("Manga OCR", "manga_ocr"),
            ("Baberu OCR", "baberu_ocr"),
            ("48px AR OCR", "ocr48px"),
            ("PP-StructureV3", "paddle_structure_v3"),
            ("外部 OCR JSON / MD", "external"),
            ("Sidecar", "sidecar"),
            ("关闭", "none"),
        ]
        self.source_ocr_backend = QComboBox(); self.target_ocr_backend = QComboBox()
        for label, key in role_choices:
            self.source_ocr_backend.addItem(label, key); self.target_ocr_backend.addItem(label, key)
        net.addWidget(QLabel("SOURCE OCR"),8,0); net.addWidget(self.source_ocr_backend,8,1,1,3)
        net.addWidget(QLabel("TARGET OCR"),9,0); net.addWidget(self.target_ocr_backend,9,1,1,3)
        self.source_ocr_backend.setToolTip("可单独覆盖旧中文版/SOURCE 的 OCR。选择外部 OCR 后完全不需要本地运行该模型。")
        self.target_ocr_backend.setToolTip("可单独覆盖高清 TARGET 的 OCR；通常保持跟随主选择或 Apple。")

        def external_row(title):
            edit=QLineEdit(); edit.setPlaceholderText("PaddleOCR-VL / PP-Structure 导出的 .json 或 .md")
            pick=QPushButton("选择…"); pick.setMaximumWidth(70)
            box=QHBoxLayout(); box.setSpacing(4); box.addWidget(edit,1); box.addWidget(pick)
            return edit,pick,box
        self.external_source_path,self.external_source_pick,src_ext_box=external_row("SOURCE")
        self.external_target_path,self.external_target_pick,tgt_ext_box=external_row("TARGET")
        self.external_source_start=QSpinBox(); self.external_source_start.setRange(1,999999); self.external_source_start.setValue(1)
        self.external_target_start=QSpinBox(); self.external_target_start.setRange(1,999999); self.external_target_start.setValue(1)
        src_line=QHBoxLayout(); src_line.setSpacing(6); src_line.addLayout(src_ext_box,1); src_line.addWidget(QLabel("起始页")); src_line.addWidget(self.external_source_start)
        tgt_line=QHBoxLayout(); tgt_line.setSpacing(6); tgt_line.addLayout(tgt_ext_box,1); tgt_line.addWidget(QLabel("起始页")); tgt_line.addWidget(self.external_target_start)
        net.addWidget(QLabel("SOURCE 外部结果"),10,0); net.addLayout(src_line,10,1,1,3)
        net.addWidget(QLabel("TARGET 外部结果"),11,0); net.addLayout(tgt_line,11,1,1,3)
        external_hint=QLabel("外部 JSON 优先：可保留 block_bbox / polygon / reading order；MD 若旁边有同名 JSON 会自动使用 JSON。起始页表示外部结果第 1 项对应本地图片序列的第几页。")
        external_hint.setObjectName("quiet"); external_hint.setWordWrap(True); net.addWidget(external_hint,12,0,1,4)
        self._ocr_settings_widgets = [
            self.paddle_model_source, self.paddle_model_profile,
            self.paddle_det_name, self.paddle_rec_name,
            self.source_ocr_backend, self.target_ocr_backend,
            self.external_source_path, self.external_source_pick, self.external_source_start,
            self.external_target_path, self.external_target_pick, self.external_target_start,
        ]
        settings.layout.addLayout(net)

        status = Card("模型下载与接入状态", "如果你选择了本地 OCR / 配准 / 气泡模型，但本机还没有它们，处理开始前会自动补齐依赖并下载所需模型；这里仍可手动检查/修复。")
        self.status_labels = {}
        self.model_download_buttons = {}
        self.dependency_buttons = {}
        downloadable = {"paddle", "lightglue", "loftr", "mangalens", "ysg_obb", "rtdetr_v2", "sam2", "koharu_layout", "manga_ocr", "baberu_ocr", "ocr48px", "lama_manga", "aot_inpainting", "flux2_klein", "rorem_mixed"}
        status_rows = [
            ("paddle","PaddleOCR"), ("lightglue","LightGlue"), ("loftr","LoFTR"),
            ("mangalens","MangaLens"), ("ysg_obb","YSG YOLO OBB"), ("rtdetr_v2","RT-DETR-v2"), ("sam2","SAM 2.1"),
            ("koharu_layout","Koharu Layout"), ("manga_ocr","Manga OCR"), ("baberu_ocr","Baberu OCR"),
            ("ocr48px","48px AR OCR"), ("lama_manga","LaMa Manga"), ("aot_inpainting","AOT"), ("flux2_klein","FLUX.2 Klein"), ("rorem_mixed","RORem Mixed"),
            ("torch_sr","Torch 局部超分"), ("apple_live_text","Apple Live Text"),
            ("apple_shortcut","ExtractText 快捷指令"),
        ]
        status_scroll_host = QWidget()
        status_rows_layout = QVBoxLayout(status_scroll_host); status_rows_layout.setContentsMargins(0,0,0,0); status_rows_layout.setSpacing(8)
        for key,name in status_rows:
            line=QHBoxLayout(); line.setSpacing(6)
            name_label=QLabel(name); name_label.setMinimumWidth(92); line.addWidget(name_label)
            line.addStretch(1)
            q=QLabel("检测中"); q.setObjectName("hint"); q.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter); self.status_labels[key]=q; line.addWidget(q)
            if key in downloadable:
                dep=QPushButton("安装依赖"); dep.setObjectName("modelDownload"); dep.setMinimumWidth(68); dep.setMaximumWidth(88)
                if key == "paddle":
                    dep.setToolTip("创建/修复独立 PaddleOCR Python 3.9–3.13 venv；不会把 Paddle 强装进 GUI Python。")
                elif key in {"lightglue", "loftr"}:
                    dep.setToolTip("创建/修复共享的 LightGlue / LoFTR 独立 Python 3.10–3.13 venv；GUI Python 3.14 不再 import Torch。")
                elif key in {"mangalens", "ysg_obb", "rtdetr_v2", "sam2", "koharu_layout", "manga_ocr", "baberu_ocr", "ocr48px"}:
                    dep.setToolTip(f"把 {name} 安装到共享的独立 Python 3.10–3.13 Torch venv；GUI Python 3.14 不加载 Torch/Ultralytics/Transformers/SAM2。")
                elif key in {"lama_manga", "aot_inpainting", "flux2_klein", "rorem_mixed"}:
                    dep.setEnabled(False); dep.setToolTip("生成/修补模型使用本地 runner；只需导入模型并配置 runner 命令。")
                else:
                    dep.setToolTip(f"安装 {name} 的运行依赖，并在安装后真实导入验证。")
                dep.clicked.connect(lambda _=False,k=key:self._install_model_dependencies(k)); self.dependency_buttons[key]=dep; line.addWidget(dep)
                imp=QPushButton("离线导入"); imp.setObjectName("modelDownload"); imp.setMinimumWidth(68); imp.setMaximumWidth(82)
                imp.clicked.connect(lambda _=False,k=key:self._import_model(k)); line.addWidget(imp)
                b=QPushButton("下载/校验"); b.setObjectName("modelDownload"); b.setMinimumWidth(82); b.setMaximumWidth(96)
                if key == "paddle": b.setText("下载/预热")
                b.clicked.connect(lambda _=False,k=key:self._download_model(k)); self.model_download_buttons[key]=b; line.addWidget(b)
            status_rows_layout.addLayout(line)
        status_rows_layout.addStretch(1)
        status_scroll = QScrollArea(); status_scroll.setWidgetResizable(True); status_scroll.setFrameShape(QFrame.Shape.NoFrame); status_scroll.setWidget(status_scroll_host); status_scroll.setMinimumHeight(230); status_scroll.setMaximumHeight(310)
        status.layout.addWidget(status_scroll)

        self.model_progress=QProgressBar(); self.model_progress.setRange(0,100); self.model_progress.setValue(0); self.model_progress.setVisible(False); status.layout.addWidget(self.model_progress)
        self.model_download_status=QLabel("支持主动下载、代理/镜像回退和离线导入。PP-OCR 使用独立 OCR venv；LightGlue / LoFTR / MangaLens / RT-DETR-v2 / SAM 2.1 全部在独立 Python 3.10–3.13 Torch venv 中运行，GUI Python 3.14 不加载这些 AI 运行库。")
        self.model_download_status.setObjectName("quiet"); self.model_download_status.setWordWrap(True); status.layout.addWidget(self.model_download_status)
        model_actions=QHBoxLayout(); self.open_model_dir=QPushButton("打开模型目录"); self.open_model_dir.clicked.connect(self._open_model_dir); model_actions.addWidget(self.open_model_dir)
        self.install_all_dependencies=QPushButton("安装全部缺失依赖"); self.install_all_dependencies.setObjectName("softPrimary"); self.install_all_dependencies.clicked.connect(lambda:self._install_model_dependencies("all")); model_actions.addWidget(self.install_all_dependencies)
        model_actions.addStretch(1); status.layout.addLayout(model_actions)
        status.layout.addStretch(1)

        lower = QGridLayout(); lower.setHorizontalSpacing(14); lower.setVerticalSpacing(14)
        lower.addWidget(settings, 0, 0)
        lower.addWidget(status, 0, 1)
        lower.setColumnStretch(0, 6); lower.setColumnStretch(1, 5)
        root.addLayout(lower)

        # Local runners and acceleration settings belong to the same runtime
        # concern.  Keep them in one compact card instead of two distant panels.
        runtime = Card("本地模型与硬件加速", desktop_platform_summary())
        runtime_grid = QGridLayout(); runtime_grid.setContentsMargins(0,0,0,0); runtime_grid.setHorizontalSpacing(10); runtime_grid.setVerticalSpacing(8)
        runner_panel = QFrame(); runner_panel.setObjectName("cardBlue")
        runner_layout = QVBoxLayout(runner_panel); runner_layout.setContentsMargins(11,10,11,10); runner_layout.setSpacing(7)
        runner_title = QLabel("本地模型 Runner"); runner_title.setObjectName("runtimeSectionTitle"); runner_layout.addWidget(runner_title)
        rform = QFormLayout()
        rform.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.ocr48px_command = QLineEdit(); self.ocr48px_command.setPlaceholderText('可选兼容回退：python ocr48.py --input {input} --output {output}')
        self._ocr_settings_widgets.append(self.ocr48px_command)
        self.lama_manga_command = QLineEdit(); self.lama_manga_command.setPlaceholderText('支持 {input} {mask} {output} {model}')
        self.aot_command = QLineEdit(); self.aot_command.setPlaceholderText('支持 {input} {mask} {output} {model}')
        self.flux_command = QLineEdit(); self.flux_command.setPlaceholderText('支持 {input} {mask} {output} {model} {prompt}')
        self.rorem_command = QLineEdit(); self.rorem_command.setPlaceholderText('支持 {input} {mask} {output} {model} {prompt} {negative_prompt}')
        self.flux_prompt = QLineEdit()
        self.rorem_prompt = QLineEdit(); self.rorem_negative = QLineEdit()
        rform.addRow("48px 外部命令（可选）", self.ocr48px_command)
        rform.addRow("LaMa Manga 命令", self.lama_manga_command)
        rform.addRow("AOT 命令", self.aot_command)
        rform.addRow("FLUX.2 Klein 命令", self.flux_command)
        rform.addRow("FLUX Prompt", self.flux_prompt)
        rform.addRow("RORem Mixed 命令", self.rorem_command)
        rform.addRow("RORem Prompt", self.rorem_prompt)
        rform.addRow("RORem Negative", self.rorem_negative)
        runner_layout.addLayout(rform)
        runner_note = QLabel("仅填写你实际使用的外部 runner；留空时继续使用 Folirina 内置/已配置路径。")
        runner_note.setObjectName("quiet"); runner_note.setWordWrap(True); runner_layout.addWidget(runner_note)

        settings_cfg = _model_settings()
        saved_source=str(settings_cfg.value("models/hf_source","auto") or "auto")
        idx=self.model_source.findData(saved_source); self.model_source.setCurrentIndex(max(0,idx))
        self.model_proxy.setText(str(settings_cfg.value("models/proxy","") or ""))
        self.model_ca_bundle.setText(str(settings_cfg.value("models/ca_bundle","") or ""))
        config_paddle_source=str(getattr(self.window.state.config.ocr,"paddle_model_source","auto") or "auto")
        saved_paddle_source=str(settings_cfg.value("models/paddle_source",config_paddle_source) or config_paddle_source)
        pidx=self.paddle_model_source.findData(saved_paddle_source); self.paddle_model_source.setCurrentIndex(max(0,pidx))
        config_profile=str(getattr(self.window.state.config.ocr,"paddle_model_profile","ppocr_v6_medium") or "ppocr_v6_medium")
        saved_profile=str(settings_cfg.value("models/paddle_profile",config_profile) or config_profile)
        midx=self.paddle_model_profile.findData(saved_profile); self.paddle_model_profile.setCurrentIndex(midx if midx >= 0 else max(0,self.paddle_model_profile.findData("ppocr_v6_medium")))
        self.paddle_det_name.setText(str(settings_cfg.value("models/paddle_det_name",getattr(self.window.state.config.ocr,"paddle_text_detection_model_name","") or "") or ""))
        self.paddle_rec_name.setText(str(settings_cfg.value("models/paddle_rec_name",getattr(self.window.state.config.ocr,"paddle_text_recognition_model_name","") or "") or ""))
        ocr_cfg=self.window.state.config.ocr
        for combo, value in ((self.source_ocr_backend, getattr(ocr_cfg,"source_backend",None)), (self.target_ocr_backend, getattr(ocr_cfg,"target_backend",None))):
            key=str(value or "inherit"); idx=combo.findData(key); combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.external_source_path.setText(str(getattr(ocr_cfg,"external_source_ocr_path","") or ""))
        self.external_target_path.setText(str(getattr(ocr_cfg,"external_target_ocr_path","") or ""))
        self.external_source_start.setValue(int(getattr(ocr_cfg,"external_source_start_page",1) or 1))
        self.external_target_start.setValue(int(getattr(ocr_cfg,"external_target_start_page",1) or 1))
        inpaint_cfg=self.window.state.config.inpainting
        self.ocr48px_command.setText(str(getattr(ocr_cfg,"ocr48px_command","") or ""))
        self.lama_manga_command.setText(str(getattr(inpaint_cfg,"lama_manga_command","") or ""))
        self.aot_command.setText(str(getattr(inpaint_cfg,"aot_command","") or ""))
        self.flux_command.setText(str(getattr(inpaint_cfg,"flux2_klein_command","") or ""))
        self.flux_prompt.setText(str(getattr(inpaint_cfg,"flux2_klein_prompt","") or ""))
        self.rorem_command.setText(str(getattr(inpaint_cfg,"rorem_mixed_command","") or ""))
        self.rorem_prompt.setText(str(getattr(inpaint_cfg,"rorem_mixed_prompt","") or ""))
        self.rorem_negative.setText(str(getattr(inpaint_cfg,"rorem_mixed_negative_prompt","") or ""))
        self.paddle_model_profile.currentIndexChanged.connect(self._paddle_profile_changed)
        self.source_ocr_backend.currentIndexChanged.connect(self._role_ocr_changed)
        self.target_ocr_backend.currentIndexChanged.connect(self._role_ocr_changed)
        self.external_source_pick.clicked.connect(lambda:self._choose_external_ocr("source"))
        self.external_target_pick.clicked.connect(lambda:self._choose_external_ocr("target"))
        self.external_source_path.editingFinished.connect(self._role_ocr_changed)
        self.external_target_path.editingFinished.connect(self._role_ocr_changed)
        self.external_source_start.valueChanged.connect(self._role_ocr_changed)
        self.external_target_start.valueChanged.connect(self._role_ocr_changed)
        for edit in (self.ocr48px_command,self.lama_manga_command,self.aot_command,self.flux_command,self.flux_prompt,self.rorem_command,self.rorem_prompt,self.rorem_negative):
            edit.editingFinished.connect(self._sync_model_runners)
        self.paddle_det_name.editingFinished.connect(lambda:self._apply_model_network_settings(show_message=False))
        self.paddle_rec_name.editingFinished.connect(lambda:self._apply_model_network_settings(show_message=False))
        self.apply_model_network.clicked.connect(self._apply_model_network_settings)
        self.test_model_network.clicked.connect(self._test_model_network)
        self.model_ca_pick.clicked.connect(self._choose_model_ca_bundle)
        self._apply_model_network_settings(show_message=False)

        self._paddle_profile_changed()

        self._platform_family = platform_family()
        hardware_panel = QFrame(); hardware_panel.setObjectName("cardBlue")
        hardware_layout = QVBoxLayout(hardware_panel); hardware_layout.setContentsMargins(11,10,11,10); hardware_layout.setSpacing(7)
        hardware_title = QLabel("硬件加速"); hardware_title.setObjectName("runtimeSectionTitle"); hardware_layout.addWidget(hardware_title)
        hform = QFormLayout()
        hform.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.profile = QComboBox(); self.profile.addItem("智能平衡（推荐）", "balanced"); self.profile.addItem("省资源", "eco"); self.profile.addItem("性能优先", "fast")
        self.device = QComboBox(); self.device.addItem("自动选择", "auto"); self.device.addItem("Apple MPS（macOS）", "mps"); self.device.addItem("NVIDIA CUDA（Windows/Linux）", "cuda"); self.device.addItem("CPU", "cpu")
        self.thread_ratio = QDoubleSpinBox(); self.thread_ratio.setRange(.20,1.0); self.thread_ratio.setSingleStep(.05); self.thread_ratio.setValue(.50); self.thread_ratio.setDecimals(2)
        self.mps_fraction = QDoubleSpinBox(); self.mps_fraction.setRange(.50,.95); self.mps_fraction.setSingleStep(.05); self.mps_fraction.setValue(.82); self.mps_fraction.setDecimals(2)
        self.mps_memory_label = QLabel("MPS 内存上限")
        hform.addRow("运行策略", self.profile); hform.addRow("推理设备", self.device); hform.addRow("CPU 线程比例", self.thread_ratio); hform.addRow(self.mps_memory_label, self.mps_fraction)
        self.mps_memory_label.setVisible(self._platform_family == "macos"); self.mps_fraction.setVisible(self._platform_family == "macos")
        hardware_layout.addLayout(hform)
        self.device_status = QLabel(); self.device_status.setObjectName("hint"); self.device_status.setWordWrap(True); hardware_layout.addWidget(self.device_status)
        route = QLabel("自动按平台使用 Mac Apple MPS / Windows·Linux NVIDIA CUDA；不可用时自动回退 CPU。整册 Worker 常驻，模型只加载一次。")
        route.setObjectName("quiet"); route.setWordWrap(True); hardware_layout.addWidget(route)
        hardware_layout.addStretch(1)
        self.start_processing = QPushButton("处理当前页"); self.start_processing.setObjectName("softPrimary"); self.start_processing.setMinimumHeight(36)
        self.start_processing.clicked.connect(self.window.run_current_page); hardware_layout.addWidget(self.start_processing)

        runtime_grid.addWidget(runner_panel, 0, 0)
        runtime_grid.addWidget(hardware_panel, 0, 1)
        runtime_grid.setColumnStretch(0, 3); runtime_grid.setColumnStretch(1, 2)
        runtime.layout.addLayout(runtime_grid)
        root.addWidget(runtime)
        self.profile.currentIndexChanged.connect(self._set_profile); self.device.currentIndexChanged.connect(self._set_device); self.thread_ratio.valueChanged.connect(self._set_threads); self.mps_fraction.valueChanged.connect(self._set_mps_fraction)
        self.apply_transfer_mode_ocr_lock(getattr(self.window.state.config.transfer, "mode", "auto"))
        root.addStretch(1)

    def _refresh_hero_preview(self):
        pair = self.window.current_pair() if hasattr(self.window, "current_pair") else None
        count = len(getattr(self.window.state, "pairs", []) or [])
        self.hero_page_count.setText(f"页数\n{count if count else '—'}")
        cfg = self.window.state.config
        ocr_name = str(getattr(cfg.ocr, "source_backend", None) or getattr(cfg.ocr, "backend", "none") or "none")
        if not self._mode_requires_ocr_controls(getattr(cfg.transfer, "mode", "auto")):
            self.hero_ocr_state.setText("OCR\n0 OCR")
        else:
            self.hero_ocr_state.setText("OCR\n" + ("已选择" if ocr_name != "none" else "关闭"))
        self.hero_reg_state.setText("配准\n" + str(getattr(cfg.registration, "backend", "auto")).upper())
        if pair is None:
            self.hero_preview.setPixmap(QPixmap()); self.hero_preview.setText("选择项目后显示当前 TARGET / 结果页")
            return
        candidate = None
        try:
            out = str(getattr(self.window.state, "output_dir", "") or "")
            if out:
                ws = resolve_page_workspace(out, pair)
                for name in ("final_reviewed.png","final.png"):
                    p = Path(ws.page_root) / name
                    if p.exists(): candidate = p; break
        except Exception:
            candidate = None
        if candidate is None:
            candidate = Path(pair.target_path)
        reader = QImageReader(str(candidate)); reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid():
            reader.setScaledSize(size.scaled(QSize(760,240), Qt.AspectRatioMode.KeepAspectRatio))
        image = reader.read()
        if image.isNull():
            self.hero_preview.setPixmap(QPixmap()); self.hero_preview.setText(candidate.name)
        else:
            pix = QPixmap.fromImage(image)
            self.hero_preview.setPixmap(pix)
            self.hero_preview.setText("")

    def _choice_card(self, title, subtitle, rows, handler):
        card=Card(title,subtitle, blue=True); group=QButtonGroup(card); group.setExclusive(True); card.group=group; card.radios={}; card.option_rows={}
        card.layout.setSpacing(8)
        for label,key,hint in rows:
            opt = OptionRow(label, hint)
            radio = opt.radio
            card.radios[key]=radio; card.option_rows[key]=opt; group.addButton(radio)
            card.layout.addWidget(opt)
            def _on_toggled(checked, k=key, option=opt):
                option.set_selected(bool(checked))
                if checked:
                    handler(k)
            radio.toggled.connect(_on_toggled)
        return card

    @staticmethod
    def _mode_requires_ocr_controls(mode: str | None) -> bool:
        # The GUI must not maintain its own second list of OCR-capable modes.
        # Keep the selection state coupled to the same capability contract used
        # by the pipeline runtime so future mode changes cannot drift apart.
        try:
            return bool(get_mode_contract(str(mode or "auto")).may_use_ocr)
        except ValueError:
            # Invalid/unknown UI state should fail open visually rather than
            # trapping the user in a permanently disabled selector; the runtime
            # contract still rejects unsupported modes during processing.
            return True

    def _set_choice_card_enabled(self, card: Card, enabled: bool) -> None:
        card.setEnabled(bool(enabled))
        for option in getattr(card, "option_rows", {}).values():
            option.setEnabled(bool(enabled))
        for radio in getattr(card, "radios", {}).values():
            radio.setEnabled(bool(enabled))

    def apply_transfer_mode_ocr_lock(self, mode: str | None = None) -> None:
        mode_value = str(mode or self.window.state.config.transfer.mode or "auto")
        enabled = self._mode_requires_ocr_controls(mode_value)
        previous = self._ocr_lock_enabled_state
        self._ocr_lock_enabled_state = bool(enabled)
        self._set_choice_card_enabled(self.ocr, enabled)
        if getattr(self.ocr, "subtitle_label", None) is not None:
            self.ocr.subtitle_label.setText(self._ocr_card_subtitle_enabled if enabled else self._ocr_card_subtitle_disabled)
        if hasattr(self, "ocr_settings_card") and getattr(self.ocr_settings_card, "subtitle_label", None) is not None:
            self.ocr_settings_card.subtitle_label.setText(self._ocr_settings_subtitle_enabled if enabled else self._ocr_settings_subtitle_disabled)
        for widget in getattr(self, "_ocr_settings_widgets", []):
            widget.setEnabled(bool(enabled))
        if hasattr(self, "paddle_det_label"):
            self.paddle_det_label.setEnabled(bool(enabled) and str(self.paddle_model_profile.currentData() or "") == "custom")
        if hasattr(self, "paddle_rec_label"):
            self.paddle_rec_label.setEnabled(bool(enabled) and str(self.paddle_model_profile.currentData() or "") == "custom")
        if previous is not None and previous != bool(enabled) and hasattr(self, "window") and hasattr(self.window, "statusBar"):
            self.window.statusBar().showMessage(
                (f"当前迁移方式 {mode_value} 不使用 OCR，OCR 选项已锁定" if not enabled else f"当前迁移方式 {mode_value} 允许 OCR，OCR 选项已恢复"),
                2200,
            )

    def _paddle_profile_changed(self):
        custom = str(self.paddle_model_profile.currentData() or "") == "custom"
        ocr_enabled = self._mode_requires_ocr_controls(getattr(self.window.state.config.transfer, "mode", "auto"))
        for widget in (self.paddle_det_label, self.paddle_det_name, self.paddle_rec_label, self.paddle_rec_name):
            widget.setEnabled(bool(custom and ocr_enabled))
        self._apply_model_network_settings(show_message=False)
        self._probe_cache = None; self._probe_cache_at = 0.0
        if hasattr(self, "device_status"):
            QTimer.singleShot(0, lambda:self.refresh(force_probe=True))
        if custom:
            self.model_download_status.setText("自定义 PaddleOCR：请填写 detection / recognition model name，或在配置中指定本地模型目录。")
        else:
            self.model_download_status.setText(f"已选择独立 Paddle 引擎：{self.paddle_model_profile.currentText()}；点击“下载所选”只准备这个引擎。")
        if hasattr(self, "paddle_profile_status"):
            self.paddle_profile_status.setText("已缓存档位：正在刷新…")

    def _choose_model_ca_bundle(self):
        path,_=QFileDialog.getOpenFileName(self,"选择 PEM CA 证书","","PEM/证书 (*.pem *.crt *.cer);;所有文件 (*)")
        if path:
            self.model_ca_bundle.setText(path)
            self._apply_model_network_settings(show_message=True)

    def _apply_model_network_settings(self, _checked=False, *, show_message: bool = True):
        source=str(self.model_source.currentData() or "auto")
        paddle_source=str(self.paddle_model_source.currentData() or "auto") if hasattr(self,"paddle_model_source") else "auto"
        paddle_profile=str(self.paddle_model_profile.currentData() or "ppocr_v6_medium") if hasattr(self,"paddle_model_profile") else "ppocr_v6_medium"
        paddle_det_name=self.paddle_det_name.text().strip() if hasattr(self,"paddle_det_name") else ""
        paddle_rec_name=self.paddle_rec_name.text().strip() if hasattr(self,"paddle_rec_name") else ""
        proxy=self.model_proxy.text().strip()
        ca_bundle=self.model_ca_bundle.text().strip()
        try:
            apply_download_network_settings(proxy=proxy,hf_source=source,ca_bundle=ca_bundle,paddle_source=paddle_source)
        except Exception as exc:
            if show_message:
                QMessageBox.warning(self,"网络设置无效",str(exc))
            return
        try:
            self.window.state.config.ocr.paddle_model_source=paddle_source
            self.window.state.config.ocr.paddle_model_profile=paddle_profile
            self.window.state.config.ocr.paddle_text_detection_model_name=paddle_det_name or None
            self.window.state.config.ocr.paddle_text_recognition_model_name=paddle_rec_name or None

        except Exception:
            logger.warning("failed to update in-memory Paddle model settings", exc_info=True)
        settings=_model_settings()
        settings.setValue("models/hf_source",source); settings.setValue("models/proxy",proxy); settings.setValue("models/ca_bundle",ca_bundle); settings.setValue("models/paddle_source",paddle_source)
        settings.setValue("models/paddle_profile",paddle_profile); settings.setValue("models/paddle_det_name",paddle_det_name); settings.setValue("models/paddle_rec_name",paddle_rec_name)
        if show_message:
            proxy_msg = proxy if proxy else "系统直连/系统代理"
            ca_msg = Path(ca_bundle).name if ca_bundle else "系统 CA 自动"
            self.model_download_status.setText(f"下载设置已应用：{self.model_source.currentText()} · Paddle源：{self.paddle_model_source.currentText()} · 模型：{self.paddle_model_profile.currentText()} · {proxy_msg} · CA：{ca_msg}")

    def _test_model_network(self):
        if self._network_probe_worker is not None and self._network_probe_worker.isRunning():
            return
        self._apply_model_network_settings(show_message=False)
        self.model_download_status.setText("正在检查 Paddle / PyPI / Hugging Face / 镜像 / GitHub DNS…")
        worker=ModelNetworkProbeWorker(); self._network_probe_worker=worker
        worker.done.connect(self._network_probe_done); worker.failed.connect(lambda msg:self.model_download_status.setText(f"网络诊断失败：{msg}"))
        worker.finished.connect(self._network_probe_finished); worker.finished.connect(worker.deleteLater); worker.start()

    def _network_probe_done(self, payload):
        rows=[]
        for host,info in dict(payload or {}).items():
            rows.append(f"{host}：{'✓ DNS' if info.get('dns') else '✗ DNS'}")
        failed=sum(1 for info in dict(payload or {}).values() if not info.get('dns'))
        suffix="；存在 DNS 失败，建议填写本机代理或使用离线导入。" if failed else "；DNS 基本正常。"
        self.model_download_status.setText(" · ".join(rows)+suffix)

    def _network_probe_finished(self):
        self._network_probe_worker=None

    def _set_dependency_busy(self, busy: bool, active_key: str = ""):
        for key,button in self.dependency_buttons.items():
            button.setEnabled(not busy)
            button.setText("安装中…" if busy and key == active_key else "安装依赖")
        if hasattr(self, "install_all_dependencies"):
            self.install_all_dependencies.setEnabled(not busy)
            self.install_all_dependencies.setText("安装中…" if busy and active_key == "all" else "安装全部缺失依赖")
        # Avoid modifying the Python environment while a model loader/download
        # may be importing the same packages.
        for button in self.model_download_buttons.values():
            button.setEnabled(not busy and self._model_download_worker is None)

    def _install_model_dependencies(self, key: str):
        if self._dependency_worker is not None and self._dependency_worker.isRunning():
            self.window.statusBar().showMessage("已有依赖安装任务正在运行。", 3500)
            return
        if self._model_download_worker is not None and self._model_download_worker.isRunning():
            QMessageBox.information(self, "模型正在下载", "请等待当前模型下载完成后再安装运行依赖。")
            return
        self._apply_model_network_settings(show_message=False)
        labels={"paddle":"PaddleOCR","lightglue":"LightGlue","loftr":"LoFTR","mangalens":"MangaLens","ysg_obb":"YSG YOLO OBB","rtdetr_v2":"RT-DETR-v2","sam2":"SAM 2.1","all":"全部内置模型"}
        label=labels.get(str(key),str(key))
        missing=[] if key == "all" else list(missing_dependency_modules(key))
        pretty_missing=[("PaddleOCR 独立运行环境" if x == "paddle-isolated-runtime" else x) for x in missing]
        missing_text=("\n当前缺少："+", ".join(pretty_missing)) if pretty_missing else ""
        extra=""
        if key in {"sam2","all"} and platform.system() == "Darwin":
            extra += "\nSAM 2.1 在 macOS 会使用关闭 CUDA 扩展的 MPS/CPU 兼容安装路径。"
        if key in {"paddle","all"}:
            extra += "\nPP-OCR 不强装进 GUI Python：会寻找兼容的 Python 3.9～3.13 并建立独立 Paddle venv。"
            if platform.system() == "Darwin":
                extra += " Apple Silicon 若没有兼容 Python，可按校验清单准备独立 Python 3.12。"
            elif platform.system() == "Windows":
                extra += " Windows 会同时检查 py launcher 与 PATH 中的兼容 Python。"
        target_text="当前 GUI Python" if key not in {"paddle","all"} else "各自兼容的隔离运行环境"
        reply=QMessageBox.question(
            self,"安装模型运行依赖",
            f"将为 {label} 安装运行依赖到{target_text}。{missing_text}{extra}\n\n安装完成后会用独立子进程真实验证。继续吗？"
        )
        if reply!=QMessageBox.StandardButton.Yes:
            return
        self._dependency_key=str(key)
        self.model_download_status.setText(f"正在安装 {label} 运行依赖…")
        self.model_progress.setVisible(True); self.model_progress.setRange(0,0)
        self._set_dependency_busy(True,str(key))
        worker=DependencyInstallWorker(str(key)); self._dependency_worker=worker
        worker.progress.connect(lambda msg:self.model_download_status.setText(str(msg)))
        worker.done.connect(self._dependency_install_done); worker.failed.connect(self._dependency_install_failed)
        worker.finished.connect(self._dependency_install_finished); worker.finished.connect(worker.deleteLater); worker.start()

    def _dependency_install_done(self, result):
        msg=str(getattr(result,"message","依赖安装完成"))
        self.model_download_status.setText(msg); self.window.statusBar().showMessage(msg,8000)
        QMessageBox.information(self,"依赖安装并验证完成",msg)
        self._probe_cache=None; self._probe_cache_at=0.0

    def _dependency_install_failed(self, message: str):
        full=str(message)
        short=full.split("\n",1)[0][:500]
        self.model_download_status.setText(f"依赖安装失败：{short}")
        # Keep the complete compatibility/source diagnostics visible.  Paddle
        # failures often need the checked Python paths/architectures, not merely
        # the first generic sentence.
        QMessageBox.warning(self,"模型运行依赖安装失败",full[-8000:])

    def _dependency_install_finished(self):
        self._dependency_worker=None; self._dependency_key=""
        self._set_dependency_busy(False)
        self.model_progress.setVisible(False); self.model_progress.setRange(0,100); self.model_progress.setValue(0)
        QTimer.singleShot(0,lambda:self.refresh(force_probe=True))

    def _import_model(self, key: str):
        if key == "paddle":
            profile=str(self.paddle_model_profile.currentData() or "ppocr_v6_medium") if hasattr(self,"paddle_model_profile") else "ppocr_v6_medium"
            if profile in {"paddle_vl_16","pp_structure_v3"}:
                QMessageBox.information(self,"文档解析引擎", "PaddleOCR-VL 1.6 / PP-StructureV3 使用完整 PaddleX 产线，不接受 Det/Rec 双目录离线导入。可使用“下载所选”准备本地引擎，或直接选择“外部 OCR JSON / MD”导入其它机器/云端的识别结果。")
                return
        if key in {"paddle","rtdetr_v2","koharu_layout","manga_ocr","baberu_ocr","ocr48px","aot_inpainting","flux2_klein","rorem_mixed"}:
            source=QFileDialog.getExistingDirectory(self,"选择离线模型目录")
        else:
            filters={"lightglue":"PyTorch 权重 (*.pth *.pt);;所有文件 (*)","loftr":"LoFTR checkpoint (*.ckpt *.pth);;所有文件 (*)","mangalens":"YOLO 权重 (*.pt);;所有文件 (*)","ysg_obb":"YSG YOLO OBB 权重 (*.pt);;所有文件 (*)","sam2":"SAM2 checkpoint (*.pt *.pth);;所有文件 (*)","lama_manga":"SafeTensors (*.safetensors);;所有文件 (*)"}
            source,_=QFileDialog.getOpenFileName(self,"选择离线模型文件","",filters.get(key,"所有文件 (*)"))
        if not source:
            return
        try:
            result=import_builtin_model(key,source,self.window.state.config)
            apply_config_updates(self.window.state.config,getattr(result,"config_updates",{}) or {})
            self.model_download_status.setText(str(result.message)); self.window.statusBar().showMessage(str(result.message),6000)
            self._probe_cache=None; self._probe_cache_at=0.0; QTimer.singleShot(0,lambda:self.refresh(force_probe=True))
        except Exception as exc:
            QMessageBox.warning(self,"离线导入失败",str(exc))

    def _open_model_dir(self):
        root=model_home(); root.mkdir(parents=True,exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))

    def _set_model_download_busy(self, busy: bool, active_key: str = ""):
        for key,button in self.model_download_buttons.items():
            button.setEnabled(not busy)
            if busy and key == active_key:
                button.setText("下载中…")
            elif key == "paddle":
                button.setText("下载所选")
            else:
                button.setText("下载/校验")
        self.model_progress.setVisible(bool(busy))
        if not busy:
            self.model_progress.setRange(0,100); self.model_progress.setValue(0)

    def _download_model(self, key: str):
        self._apply_model_network_settings(show_message=False)
        if self._model_download_worker is not None and self._model_download_worker.isRunning():
            self.window.statusBar().showMessage("已有模型正在下载，请等待当前任务完成。",3500); return
        sizes={"paddle":"模型大小随档位变化","lightglue":"约 45 MB","loftr":"约 45 MB","mangalens":"约 12 MB","ysg_obb":"YSG OBB 模型","rtdetr_v2":"约 172 MB + 配置","sam2":"约 156 MB","baberu_ocr":"约 121 MB ONNX","manga_ocr":"约 460 MB","ocr48px":"约 195 MB checkpoint + 字表 + 固定网络源码","lama_manga":"约 200 MB","koharu_layout":"大型实例分割模型","aot_inpainting":"模型 + 配置","flux2_klein":"超大型目录，建议离线导入","rorem_mixed":"约 1.9 GiB UNet + SDXL 运行组件"}
        label={"paddle":self.paddle_model_profile.currentText(),"lightglue":"LightGlue SIFT","loftr":"LoFTR outdoor","mangalens":"MangaLens","ysg_obb":"YSG YOLO OBB","rtdetr_v2":"RT-DETR-v2","sam2":"SAM 2.1 Hiera Tiny","koharu_layout":"Koharu Layout RF-DETR Seg 2XL","manga_ocr":"Manga OCR","baberu_ocr":"Baberu OCR","ocr48px":"48px AR OCR","lama_manga":"LaMa Manga","aot_inpainting":"AOT Inpainting","flux2_klein":"FLUX.2 Klein","rorem_mixed":"RORem Mixed"}.get(key,key)
        self.model_download_status.setText(f"正在主动下载 {label} · {sizes.get(key,'')}。下载使用临时 .part 文件，完成后原子替换。")
        self._model_download_key=str(key); self._set_model_download_busy(True,key)
        self.model_progress.setRange(0,0)
        worker=ModelDownloadWorker(key,self.window.state.config.model_copy(deep=True)); self._model_download_worker=worker
        worker.progress.connect(self._model_download_progress); worker.done.connect(self._model_download_done); worker.failed.connect(self._model_download_failed)
        worker.finished.connect(self._model_download_finished); worker.finished.connect(worker.deleteLater); worker.start()

    def _model_download_progress(self, percent: int, message: str):
        if percent < 0:
            self.model_progress.setRange(0,0)
        else:
            self.model_progress.setRange(0,100); self.model_progress.setValue(int(percent))
        self.model_download_status.setText(str(message))

    def _model_download_done(self, result):
        try:
            apply_config_updates(self.window.state.config,getattr(result,"config_updates",{}) or {})
        except Exception as exc:
            self.model_download_status.setText(f"模型已下载，但自动写入配置失败：{exc}")
            return
        self.model_download_status.setText(str(getattr(result,"message","模型下载完成")))
        self.window.statusBar().showMessage(str(getattr(result,"message","模型下载完成")),6000)
        self._probe_cache=None; self._probe_cache_at=0.0
        QTimer.singleShot(0,lambda:self.refresh(force_probe=True))

    def _model_download_failed(self, message: str):
        full=str(message)
        short=full.split("\n",1)[0][:600]
        self.model_download_status.setText(f"下载失败：{short}")
        # Paddle failover diagnostics include each attempted hub/dependency
        # mirror. Keep them visible instead of truncating to the first sentence.
        QMessageBox.warning(self,"模型下载失败",full[-10000:])

    def _model_download_finished(self):
        self._model_download_worker=None; self._model_download_key=""; self._set_model_download_busy(False)

    def _set_ocr(self,key):
        if not self._mode_requires_ocr_controls(getattr(self.window.state.config.transfer, "mode", "auto")):
            return
        cfg = self.window.state.config.ocr
        cfg.backend = key
        cfg.source_backend = key
        cfg.target_backend = key
        profile = backend_profile_key(key)
        if profile:
            cfg.paddle_model_profile = profile
            if hasattr(self, "paddle_model_profile"):
                idx=self.paddle_model_profile.findData(profile)
                if idx >= 0:
                    self.paddle_model_profile.blockSignals(True); self.paddle_model_profile.setCurrentIndex(idx); self.paddle_model_profile.blockSignals(False)
        if hasattr(self,"source_ocr_backend"):
            for combo in (self.source_ocr_backend,self.target_ocr_backend):
                idx=combo.findData(key)
                combo.blockSignals(True); combo.setCurrentIndex(idx if idx >= 0 else 0); combo.blockSignals(False)
        self.window.statusBar().showMessage(f"OCR：{key}",2500)

    def _role_ocr_changed(self, *_args):
        if not hasattr(self,"source_ocr_backend"):
            return
        if not self._mode_requires_ocr_controls(getattr(self.window.state.config.transfer, "mode", "auto")):
            return
        cfg=self.window.state.config.ocr
        src=str(self.source_ocr_backend.currentData() or "inherit")
        tgt=str(self.target_ocr_backend.currentData() or "inherit")
        cfg.source_backend = None if src == "inherit" else src
        cfg.target_backend = None if tgt == "inherit" else tgt
        cfg.external_source_ocr_path = self.external_source_path.text().strip() or None
        cfg.external_target_ocr_path = self.external_target_path.text().strip() or None
        cfg.external_source_start_page = int(self.external_source_start.value())
        cfg.external_target_start_page = int(self.external_target_start.value())
        self.window.statusBar().showMessage(f"SOURCE OCR：{src} · TARGET OCR：{tgt}",2500)

    def _choose_external_ocr(self, role: str):
        if not self._mode_requires_ocr_controls(getattr(self.window.state.config.transfer, "mode", "auto")):
            return
        path,_=QFileDialog.getOpenFileName(self,"选择外部 OCR 结果","","OCR 结果 (*.json *.md *.markdown);;JSON (*.json);;Markdown (*.md *.markdown);;所有文件 (*)")
        if not path:
            return
        if role == "target":
            self.external_target_path.setText(path)
            idx=self.target_ocr_backend.findData("external")
            if idx >= 0: self.target_ocr_backend.setCurrentIndex(idx)
        else:
            self.external_source_path.setText(path)
            idx=self.source_ocr_backend.findData("external")
            if idx >= 0: self.source_ocr_backend.setCurrentIndex(idx)
        self._role_ocr_changed()

    def _sync_model_runners(self):
        cfg=self.window.state.config
        cfg.ocr.ocr48px_command=self.ocr48px_command.text().strip() or None
        cfg.inpainting.lama_manga_command=self.lama_manga_command.text().strip() or None
        cfg.inpainting.aot_command=self.aot_command.text().strip() or None
        cfg.inpainting.flux2_klein_command=self.flux_command.text().strip() or None
        cfg.inpainting.flux2_klein_prompt=self.flux_prompt.text().strip() or "Remove the text and reconstruct the surrounding artwork."
        cfg.inpainting.rorem_mixed_command=self.rorem_command.text().strip() or None
        cfg.inpainting.rorem_mixed_prompt=self.rorem_prompt.text().strip() or "clean manga background, reconstruct artwork, no text"
        cfg.inpainting.rorem_mixed_negative_prompt=self.rorem_negative.text().strip() or "letters, text, watermark, artifacts"
        self.window.statusBar().showMessage("本地模型 Runner 配置已更新",2500)
    def _set_reg(self,key): self.window.state.config.registration.backend=key; self.window.statusBar().showMessage(f"配准：{key}",2500)
    def _set_primary_detector(self,key):
        cfg=self.window.state.config.bubbles
        cfg.primary_detector=key
        self._detector_policy_changed()
        self.window.statusBar().showMessage(f"主检测器：{key}",2500)

    def _detector_policy_changed(self):
        if not hasattr(self, "detector_strategy"):
            return
        cfg=self.window.state.config.bubbles
        cfg.detector_strategy=self.detector_strategy.currentData() or "primary_conditional_aux"
        primary=str(getattr(cfg,"primary_detector","koharu_layout") or "koharu_layout")
        cfg.auxiliary_detectors=[key for key,cb in self.detector_aux_checks.items() if cb.isChecked() and key != primary]
        cfg.sam2_refine_enabled=bool(self.sam2_refine.isChecked())
        # Keep the legacy backend synchronized for old project readers/plugins.
        legacy = next((x for x in cfg.auxiliary_detectors if x in {"mangalens","ysg_obb","rtdetr_v2","sidecar","ctd_sidecar"}), None)
        if legacy is None and "geometry_white" in cfg.auxiliary_detectors:
            legacy="seeded_white"
        cfg.backend=legacy or (primary if primary in {"koharu_layout","mangalens","rtdetr_v2"} else "seeded_white")
        self._refresh_detector_policy_controls()

    def _refresh_detector_policy_controls(self):
        if not hasattr(self, "detector_aux_checks"):
            return
        cfg=self.window.state.config.bubbles
        primary=str(getattr(cfg,"primary_detector","koharu_layout") or "koharu_layout")
        strategy=str(getattr(cfg,"detector_strategy","primary_conditional_aux") or "primary_conditional_aux")
        idx=self.detector_strategy.findData(strategy)
        if idx>=0 and self.detector_strategy.currentIndex()!=idx:
            self.detector_strategy.blockSignals(True); self.detector_strategy.setCurrentIndex(idx); self.detector_strategy.blockSignals(False)
        selected=set(getattr(cfg,"auxiliary_detectors",[]) or [])
        aux_enabled = strategy != "primary_only"
        for key,cb in self.detector_aux_checks.items():
            checked = key in selected and key != primary
            if cb.isChecked()!=checked:
                cb.blockSignals(True); cb.setChecked(checked); cb.blockSignals(False)
            cb.setEnabled(aux_enabled and key != primary)
            if key == primary:
                cb.setToolTip("当前已作为主检测器运行，不需要重复作为辅助。")
            else:
                cb.setToolTip("仅在当前检测策略允许时运行。")
        self.sam2_refine.setEnabled(aux_enabled)
        if self.sam2_refine.isChecked()!=bool(getattr(cfg,"sam2_refine_enabled",False)):
            self.sam2_refine.blockSignals(True); self.sam2_refine.setChecked(bool(getattr(cfg,"sam2_refine_enabled",False))); self.sam2_refine.blockSignals(False)
    def _set_inpaint(self,key): self.window.state.config.inpainting.backend=key; self.window.statusBar().showMessage(f"Inpainting：{key}",2500)
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
        self.device_status.setText(f"运行策略：{self.profile.currentText()} · 批量预检 {prefetch} 线程 · 深度推理按设备能力受控执行")
    def _set_device(self):
        key=self.device.currentData() or "auto"; cfg=self.window.state.config
        cfg.runtime.device=key; cfg.registration.device=key; cfg.bubbles.device=key; cfg.mask_replace.sr_device=key; cfg.direct_patch.sr_device=key
        self.refresh()
    def _set_threads(self): self.window.state.config.runtime.cpu_thread_ratio=float(self.thread_ratio.value())
    def _set_mps_fraction(self): self.window.state.config.runtime.mps_memory_fraction=float(self.mps_fraction.value())
    def refresh(self, force_probe: bool = False):
        cfg=self.window.state.config
        if hasattr(self, "pipeline_registration"):
            mode=str(getattr(cfg.transfer,"mode","auto") or "auto")
            try:
                may_ocr=bool(get_mode_contract(mode).may_use_ocr)
            except Exception:
                may_ocr=True
            detector_labels={"koharu_layout":"Koharu Layout","mangalens":"MangaLens","rtdetr_v2":"RT-DETR-v2"}
            reg_labels={"auto":"Auto / SIFT","lightglue":"LightGlue","loftr":"LoFTR"}
            ocr_labels={"apple":"Apple Live Text","apple_shortcut":"Apple Shortcut","paddle_v6_medium":"PaddleOCR v6 M","paddle_v6_small":"PaddleOCR v6 S","paddle_vl_16":"PaddleOCR-VL","manga_ocr":"Manga OCR","baberu_ocr":"Baberu OCR","ocr48px":"48px AR","external":"External OCR","sidecar":"Sidecar","none":"关闭"}
            inpaint_labels={"auto":"Auto","solid":"Solid","opencv":"OpenCV","lama":"LaMa","lama_manga":"LaMa Manga","aot_inpainting":"AOT","flux2_klein":"FLUX.2","rorem_mixed":"RORem"}
            reg_key=str(getattr(cfg.registration,"backend","auto") or "auto")
            det_key=str(getattr(cfg.bubbles,"primary_detector","koharu_layout") or "koharu_layout")
            ocr_key=str(getattr(cfg.ocr,"backend","none") or "none")
            inp_key=str(getattr(cfg.inpainting,"backend","auto") or "auto")
            self.pipeline_registration.setText("配准 · "+reg_labels.get(reg_key,reg_key))
            self.pipeline_detector.setText("主检测 · "+detector_labels.get(det_key,det_key))
            self.pipeline_ocr.setText("OCR · "+(ocr_labels.get(ocr_key,ocr_key) if may_ocr else "当前模式 0 OCR"))
            self.pipeline_inpaint.setText("修补 · "+inpaint_labels.get(inp_key,inp_key))
        if hasattr(self, "hero_preview"):
            self._refresh_hero_preview()
        ocr_key=str(cfg.ocr.backend or "none")
        if ocr_key == "paddle":
            profile=str(getattr(cfg.ocr,"paddle_model_profile","ppocr_v6_medium") or "ppocr_v6_medium")
            reverse={"ppocr_v6_medium":"paddle_v6_medium","ppocr_v6_small":"paddle_v6_small","paddle_vl_16":"paddle_vl_16","pp_structure_v3":"paddle_structure_v3"}
            ocr_key=reverse.get(profile, "paddle_v6_medium")
        for card,key in ((self.ocr,ocr_key),(self.reg,cfg.registration.backend),(self.bubble,getattr(cfg.bubbles,"primary_detector","koharu_layout")),(self.inpaint,cfg.inpainting.backend)):
            if key in card.radios and not card.radios[key].isChecked():
                card.radios[key].blockSignals(True); card.radios[key].setChecked(True); card.radios[key].blockSignals(False)
        self._refresh_detector_policy_controls()
        if hasattr(self,"source_ocr_backend"):
            for combo,value in ((self.source_ocr_backend,getattr(cfg.ocr,"source_backend",None)),(self.target_ocr_backend,getattr(cfg.ocr,"target_backend",None))):
                key=str(value or "inherit"); idx=combo.findData(key)
                if idx >= 0 and combo.currentIndex()!=idx:
                    combo.blockSignals(True); combo.setCurrentIndex(idx); combo.blockSignals(False)
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
        self.apply_transfer_mode_ocr_lock(getattr(cfg.transfer, "mode", "auto"))
        if force_probe or stale:
            self._start_probe()
        if self._probe_cache is None:
            self.device_status.setText("组件状态正在后台检测；切换功能区不会等待探测完成。")
            for q in self.status_labels.values():
                q.setText("检测中…")
            return
        self._apply_probe_statuses(self._probe_cache)


    def shutdown_background_probes(self, timeout_ms: int = 2200) -> None:
        """Stop read-only probe QThreads before QApplication teardown.

        PySide/Qt aborts when a live QThread wrapper is destroyed during Python
        finalization. Component/network probes are read-only, so a bounded wait
        followed by termination is safer than letting Qt tear them down alive.
        Model downloads/dependency installs are intentionally not force-killed.
        """
        for attr in ("_probe_worker", "_network_probe_worker"):
            worker = getattr(self, attr, None)
            if worker is None:
                continue
            try:
                if worker.isRunning():
                    worker.requestInterruption()
                    if not worker.wait(max(0, int(timeout_ms))):
                        worker.terminate()
                        worker.wait(1200)
            except RuntimeError:
                pass
            setattr(self, attr, None)

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
        cuda=statuses.get("cuda")
        if self._platform_family == "macos":
            selected=cfg.runtime.device.upper() if cfg.runtime.device != "auto" else ("MPS" if mps.ready else "CPU / 自动")
            detail=mps.detail
        else:
            auto_selected = "CUDA" if bool(cuda and cuda.ready) else "CPU / 自动"
            selected=cfg.runtime.device.upper() if cfg.runtime.device != "auto" else auto_selected
            detail=(cuda.detail if cuda is not None else "CUDA 可用性在任务启动时由 PyTorch 验证；不可用时自动回退 CPU。")
        self.device_status.setText(f"计划设备：{selected} · {detail}")
        try:
            selected_key=str(self.paddle_model_profile.currentData() or getattr(cfg.ocr,"paddle_model_profile","ppocr_v6_medium"))
            selected_ready,warmed=paddle_profile_marker_status(selected_key)
            warmed_labels=[profile_label(x) for x in warmed]
            state_text="当前已就绪" if selected_ready else "当前未预热"
            self.paddle_profile_status.setText(f"{state_text} · 已缓存档位：" + (" / ".join(warmed_labels) if warmed_labels else "无"))
        except Exception as exc:
            self.paddle_profile_status.setText(f"Paddle 模型缓存状态读取失败：{exc}")
        local=model_local_paths()
        def _artifact_present(key: str) -> bool:
            if key == "mangalens":
                path=Path(cfg.bubbles.mangalens_model_path).expanduser() if cfg.bubbles.mangalens_model_path else local.get(key)
                return bool(path and path.is_file())
            if key == "rtdetr_v2":
                path=Path(cfg.bubbles.rtdetr_model_path).expanduser() if cfg.bubbles.rtdetr_model_path else local.get(key)
                return bool(path and path.is_dir() and all((path/n).is_file() for n in ("config.json","preprocessor_config.json","model.safetensors")))
            if key == "sam2":
                path=Path(cfg.bubbles.sam2_checkpoint).expanduser() if cfg.bubbles.sam2_checkpoint else local.get(key)
                return bool(path and path.is_file())
            path=local.get(key)
            return bool(path and path.exists())
        for key,st in statuses.items():
            if key in self.status_labels:
                artifact=_artifact_present(key) if key in self.model_download_buttons else False
                # ComponentProbeWorker already resolved runtime readiness off the
                # GUI thread. Never call missing_dependency_modules() here: for
                # isolated runtimes that can spawn Python import probes and used
                # to freeze DETECT & ALIGN tab switching for seconds.
                missing=() if bool(st.installed) else (("paddle-isolated-runtime",) if key == "paddle" else ("isolated-runtime",)) if key in self.dependency_buttons else ()
                if st.ready:
                    label="已就绪"
                elif artifact and missing:
                    label="模型已下载 / 缺依赖"
                elif artifact:
                    label="模型已下载 / 待验证"
                elif not missing and key in self.dependency_buttons:
                    label="依赖已装 / 待模型"
                elif missing:
                    label="缺依赖 / 待模型"
                else:
                    label="未安装"
                self.status_labels[key].setText(label)
                detail=str(st.detail)
                if missing:
                    readable=[("PaddleOCR 独立运行环境" if x == "paddle-isolated-runtime" else x) for x in missing]
                    detail += "\n缺少运行模块：" + ", ".join(readable)
                self.status_labels[key].setToolTip(detail)
                if key in self.dependency_buttons and self._dependency_worker is None:
                    self.dependency_buttons[key].setText("安装依赖" if missing else "检查/修复")
                    readable=[("PaddleOCR 独立运行环境" if x == "paddle-isolated-runtime" else x) for x in missing]
                    self.dependency_buttons[key].setToolTip(("缺少：" + ", ".join(readable)) if missing else "依赖已可发现；点击可再次验证/修复运行环境。")
                if key in self.model_download_buttons and self._model_download_worker is None:
                    if key == "paddle":
                        self.model_download_buttons[key].setText("重新预热所选" if st.ready else "下载所选")
                    else:
                        self.model_download_buttons[key].setText("校验/重下" if st.ready else "下载/校验")

__all__ = ["ModelPage"]
