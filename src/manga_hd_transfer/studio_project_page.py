from __future__ import annotations

"""Page Manager UI.

The project/page pairing and classification widgets are isolated from the main
StudioWindow shell. Signal semantics and calls into ``window`` remain unchanged.
"""

from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QTimer, Signal
from PySide6.QtGui import QPixmap, QPainter, QColor, QIcon, QImageReader, QActionGroup
from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFormLayout, QStackedWidget, QFileDialog, QMessageBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QListWidget, QListWidgetItem, QMenu, QButtonGroup, QDialog,
    QSplitter, QSizePolicy,
)

from .gui_components import (
    _configure_responsive_dialog, Card, PathRow, ImageView, StableThumbnailList,
    ZoomPreviewView, StableComboBox, StableSpinBox, StableDoubleSpinBox,
)

QComboBox = StableComboBox
QSpinBox = StableSpinBox
QDoubleSpinBox = StableDoubleSpinBox
from .gui_theme import semantic_palette
from .font_catalog import discover_fonts
from .io_utils import load_json
from .direct_patch_status import summarize_direct_patch_payload  # compatibility/public import; runtime uses safe wrapper
from . import direct_patch_status as _direct_patch_status
from .pairing import pairing_method
from .schema_compat import as_dict, normalize_project
from .workspace import page_id_for_pair, resolve_page_workspace
from .modes.registry import active_mode_ui_items, compact_mode_ui_label, get_mode_spec, is_legacy_mode
from .page_management import (
    PAGE_TYPE_INFO, MANUAL_PAGE_TYPES, page_type_color, page_type_label,
)


def _safe_direct_patch_summary(page_dir: str | Path) -> dict:
    """Never let a stale/mixed GUI module crash page selection.

    Historical in-place upgrades could leave ``studio_project_page`` referring
    to a function name that was not imported, which made selecting a thumbnail
    raise ``NameError``.  Resolve the status helper from its module object and
    fail closed to a small diagnostic payload if the workspace/status file is
    malformed.  Status rendering must never make page navigation unusable.
    """
    try:
        fn = getattr(_direct_patch_status, "summarize_direct_patch_payload", None)
        if callable(fn):
            return dict(fn(page_dir) or {})
    except Exception as exc:
        return {
            "payload": {}, "used": False, "accepted": False,
            "applied_count": 0, "region_count": 0, "missing_files": [],
            "reason": f"status_summary_error:{type(exc).__name__}",
            "strategy": "direct_borderless_overlay",
        }
    return {
        "payload": {}, "used": False, "accepted": False,
        "applied_count": 0, "region_count": 0, "missing_files": [],
        "reason": "status_summary_unavailable",
        "strategy": "direct_borderless_overlay",
    }


class AlignedCheckOption(QWidget):
    """Checkbox row with indicator/text baseline alignment on macOS/Qt.

    Native ``QCheckBox`` text can sit a few pixels lower than the indicator on
    some fonts, which makes the project options card look visually broken. Keep
    the semantics/signals of a normal checkbox while rendering the indicator and
    label in one explicit horizontal row.
    """

    toggled = Signal(bool)

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._checkbox = QCheckBox()
        # Keep the indicator itself on an explicit 18x18 logical-pixel grid.
        # Relying on the native empty-QCheckBox size is not stable on macOS:
        # font/DPR changes can shift the painted square vertically even though
        # the surrounding widget has the same height.  A local indicator rule
        # makes every project-option square share the exact same center line.
        self._checkbox.setFixedSize(22, 22)
        self._checkbox.setStyleSheet(
            "QCheckBox { min-height:22px; max-height:22px; min-width:22px; "
            "max-width:22px; padding:0; margin:0; spacing:0; } "
            "QCheckBox::indicator { width:18px; height:18px; margin:2px; }"
        )
        self._label = QLabel(text)
        self._label.setWordWrap(False)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self._label.setFixedHeight(22)
        self._label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._label.mousePressEvent = self._label_mouse_press_event
        row = QHBoxLayout(self); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(6)
        row.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        row.addWidget(self._checkbox, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        row.addWidget(self._label, 1, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.setFixedHeight(24)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFocusProxy(self._checkbox)
        self._checkbox.toggled.connect(self.toggled.emit)

    def _label_mouse_press_event(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._checkbox.toggle()
            event.accept()
            return
        QLabel.mousePressEvent(self._label, event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._checkbox.toggle()
            event.accept()
            return
        super().mousePressEvent(event)

    def isChecked(self) -> bool:
        return self._checkbox.isChecked()

    def setChecked(self, checked: bool) -> None:
        self._checkbox.setChecked(checked)

    def setToolTip(self, text: str) -> None:
        super().setToolTip(text)
        self._checkbox.setToolTip(text)
        self._label.setToolTip(text)


class CompactSection(QFrame):
    """Small collapsible settings section used by the Page Manager.

    Keeps low-frequency renderer options available without forcing every mode to
    show its full configuration at once.  Collapsing changes UI geometry only;
    the underlying widgets/configuration remain alive and unchanged.
    """

    toggled = Signal(bool)

    def __init__(self, title: str, summary: str = "", *, expanded: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self._title = str(title)
        self.setObjectName("compactSection")
        root = QVBoxLayout(self); root.setContentsMargins(8,6,8,7); root.setSpacing(5)
        head = QHBoxLayout(); head.setContentsMargins(0,0,0,0); head.setSpacing(7)
        self.toggle = QPushButton(); self.toggle.setObjectName("collapseToggle")
        self.toggle.setCheckable(True); self.toggle.setChecked(bool(expanded))
        self.toggle.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.summary = QLabel(str(summary or "")); self.summary.setObjectName("quiet")
        self.summary.setWordWrap(False); self.summary.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.summary.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        head.addWidget(self.toggle, 0); head.addWidget(self.summary, 1)
        root.addLayout(head)
        self.body = QWidget(); self.content_layout = QVBoxLayout(self.body)
        self.content_layout.setContentsMargins(0,2,0,0); self.content_layout.setSpacing(6)
        root.addWidget(self.body)
        self.toggle.toggled.connect(self.setExpanded)
        self.setExpanded(bool(expanded))

    def setSummary(self, text: str) -> None:
        self.summary.setText(str(text or ""))
        self.summary.setVisible(bool(str(text or "").strip()))

    def setExpanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        self.toggle.blockSignals(True); self.toggle.setChecked(expanded); self.toggle.blockSignals(False)
        self.toggle.setText(("▾ " if expanded else "▸ ") + self._title)
        self.body.setVisible(expanded)
        self.toggled.emit(expanded)
        self.updateGeometry()


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
        self._last_refresh_key = None
        self._thumb_generation = 0
        self._thumb_queue: list[int] = []
        self._thumb_retry_count = 0
        self._thumb_items: dict[int, QListWidgetItem] = {}
        self._preview_dialog: PagePreviewDialog | None = None
        self._detail_side = "target"
        self._thumb_side = "target"
        # Scrolling must never compete with eager image decoding. Keep only a
        # bounded cache of already-scaled thumbnail pixels and load visible cards
        # after scrolling settles.
        self._thumb_image_cache: OrderedDict[tuple[str, int, int, int, int], QPixmap] = OrderedDict()
        self._thumb_image_cache_limit = 96
        self._thumb_placeholder_cache: QIcon | None = None
        self._thumb_loaded: set[int] = set()
        self._thumb_load_timer = QTimer(self); self._thumb_load_timer.setSingleShot(True)
        self._thumb_load_timer.timeout.connect(self._pump_thumbnails)

        root = QHBoxLayout(self); root.setContentsMargins(16,16,16,16); root.setSpacing(12)

        left_col = QVBoxLayout(); left_col.setSpacing(10)

        # Project paths are necessary but low-frequency controls. Keep them in a
        # compact collapsible card so Page Manager owns the full left workspace
        # after pairing/restoring an existing session.
        self.project_card = QFrame(); self.project_card.setObjectName("card")
        project_layout = QVBoxLayout(self.project_card); project_layout.setContentsMargins(14,10,14,12); project_layout.setSpacing(8)
        project_head = QHBoxLayout(); project_head.setContentsMargins(0,0,0,0); project_head.setSpacing(8)
        project_title = QLabel("项目文件"); project_title.setObjectName("cardTitle")
        self.project_summary = QLabel("未选择输入"); self.project_summary.setObjectName("quiet")
        self.project_toggle = QPushButton("收起"); self.project_toggle.setObjectName("collapseToggle"); self.project_toggle.setCheckable(True); self.project_toggle.setChecked(True)
        project_head.addWidget(project_title); project_head.addWidget(self.project_summary, 1); project_head.addWidget(self.project_toggle)
        project_layout.addLayout(project_head)
        self.project_body = QWidget(); project_body_layout = QVBoxLayout(self.project_body); project_body_layout.setContentsMargins(0,2,0,0); project_body_layout.setSpacing(8)
        self.source = PathRow("旧中文版", "选择目录")
        self.target = PathRow("高清日文版", "选择目录")
        self.output = PathRow("输出目录", "选择目录")
        project_body_layout.addWidget(self.source); project_body_layout.addWidget(self.target); project_body_layout.addWidget(self.output)

        pair_opts = QGridLayout(); pair_opts.setContentsMargins(0, 2, 0, 0)
        pair_opts.setHorizontalSpacing(24); pair_opts.setVerticalSpacing(6)
        self.prefer_name_pair = AlignedCheckOption("优先名称 / 页码配对"); self.prefer_name_pair.setChecked(False)
        self.prefer_name_pair.setToolTip("可选加速：先锁定同名、仅扩展名不同、以及唯一页码相同的页面。默认关闭。")
        self.prefer_order_pair = AlignedCheckOption("优先文件夹自然顺序"); self.prefer_order_pair.setChecked(False)
        self.prefer_order_pair.setToolTip("可选加速：对等长区间按自然排序一一对应。默认关闭，避免插页/缺页造成连锁错位。")
        self.remake_pair_verify = AlignedCheckOption("重制增强配对 · AKAZE 二次核验"); self.remake_pair_verify.setChecked(False)
        self.remake_pair_verify.setToolTip("对低置信智能配对做模型无关的局部特征 + RANSAC 二次核验。只在强证据成立时提高置信度；不会自动删除、重排页面。默认关闭以保持旧项目行为完全不变。")
        pair_align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        pair_opts.addWidget(self.prefer_name_pair, 0, 0, alignment=pair_align)
        pair_opts.addWidget(self.prefer_order_pair, 0, 1, alignment=pair_align)
        pair_opts.addWidget(self.remake_pair_verify, 1, 0, 1, 2, alignment=pair_align)
        pair_opts.setRowMinimumHeight(0, 24); pair_opts.setRowMinimumHeight(1, 24)
        pair_opts.setColumnStretch(0, 1); pair_opts.setColumnStretch(1, 1)
        project_body_layout.addLayout(pair_opts)

        pair_row = QHBoxLayout(); pair_row.setSpacing(8)
        self.pair_btn = QPushButton("智能配对"); self.pair_btn.setObjectName("primary")
        self.restore_btn = QPushButton("读取已有运行结果…"); self.restore_btn.setObjectName("softPrimary")
        self.restore_btn.setToolTip("读取命令行/Codex 已生成的输出目录，恢复 SOURCE/TARGET、final、中间产物和人工复核状态，然后继续在 GUI 补漏。不会重新处理、覆盖已有结果或跳转到其它功能页。")
        self.refresh_btn = QPushButton("刷新"); self.refresh_btn.setObjectName("compactAction")
        pair_row.addWidget(self.pair_btn); pair_row.addWidget(self.restore_btn); pair_row.addWidget(self.refresh_btn); pair_row.addStretch(1)
        project_body_layout.addLayout(pair_row)
        project_layout.addWidget(self.project_body)
        self.project_body.setVisible(True)
        self.project_toggle.clicked.connect(self._set_project_files_expanded)

        left_col.addWidget(self.project_card, 0)
        manager_card = Card("页面管理")
        manager_hint = QLabel("缩略图快速检查 · 列表批量核对 · 右键快速标记 · Ctrl / Command / Shift 多选")
        manager_hint.setObjectName("quiet"); manager_hint.setWordWrap(True); manager_card.layout.addWidget(manager_hint)
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
        manager_card.layout.addWidget(manager)

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
        manager_card.layout.addLayout(viewbar)
        hint = QLabel("双击：并排放大 · 右键：快速标记 · 多选：批量分类")
        hint.setObjectName("quiet"); hint.setWordWrap(True); manager_card.layout.addWidget(hint)

        self.view_stack = QStackedWidget()
        gallery_page = QWidget(); gallery_l = QVBoxLayout(gallery_page); gallery_l.setContentsMargins(0,0,0,0); gallery_l.setSpacing(0)
        self.thumb_list = StableThumbnailList(); self.thumb_list.setObjectName("pageGallery"); self.thumb_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumb_list.setIconSize(self.THUMB_CANVAS); self.thumb_list.setGridSize(QSize(202, 292))
        self.thumb_list.setResizeMode(QListWidget.ResizeMode.Adjust); self.thumb_list.setMovement(QListWidget.Movement.Static)
        self.thumb_list.setWrapping(True); self.thumb_list.setSpacing(8)
        self.thumb_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.thumb_list.setUniformItemSizes(True)
        self.thumb_list.setMinimumWidth(280)
        self.thumb_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.thumb_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.thumb_list.verticalScrollBar().setSingleStep(28)
        self.thumb_list.verticalScrollBar().setPageStep(196)
        self.thumb_list.horizontalScrollBar().setSingleStep(28)
        self.thumb_list.verticalScrollBar().valueChanged.connect(lambda _v: self._schedule_visible_thumbnails(80))
        self.thumb_list.horizontalScrollBar().valueChanged.connect(lambda _v: self._schedule_visible_thumbnails(80))
        gallery_l.addWidget(self.thumb_list, 1)
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
        manager_card.layout.addWidget(self.view_stack, 1)

        # The page inspector belongs to the right-hand context column.  Keeping it
        # out of Page Manager lets the gallery/table use the complete left half.
        left_col.addWidget(manager_card, 1); root.addLayout(left_col, 1)

        right_col = QVBoxLayout(); right_col.setSpacing(10)
        self._right_col = right_col
        mode = Card("迁移方式")
        self.mode_card = mode
        mode.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.mode = QComboBox(); self.mode.setObjectName("routeSelector"); [self.mode.addItem(label, key) for label, key in active_mode_ui_items()]; self._last_mode_key = str(self.window.state.config.transfer.mode or "direct_patch")
        mode.layout.addWidget(self.mode)
        self.mode_summary = QLabel(""); self.mode_summary.setObjectName("hint"); self.mode_summary.setWordWrap(True); self.mode_summary.setMaximumHeight(42)
        mode.layout.addWidget(self.mode_summary)
        # HD reletter typography is now a first-class setting instead of a hidden
        # config/CLI-only option.  Pixel/mask transfer modes ignore this field.
        reletter_box = QFrame(); reletter_box.setObjectName("typographyPanel")
        self.reletter_box = reletter_box
        rbl = QVBoxLayout(reletter_box); rbl.setContentsMargins(10,9,10,9); rbl.setSpacing(7)

        self.reletter_font_preset = QComboBox()
        for label, value in [("自定义", "custom"), ("黑体 Sans", "sans"), ("宋体 Serif", "serif"), ("圆体 Rounded", "rounded"), ("漫画体 Comic", "comic")]:
            self.reletter_font_preset.addItem(label, value)
        self.reletter_font_preset.setMinimumWidth(126)
        rhead = QHBoxLayout(); rhead.setSpacing(7)
        rtitle = QLabel("OCR重排 · 字体与排版"); rtitle.setObjectName("typographyTitle")
        rhead.addWidget(rtitle); rhead.addStretch(1)
        preset_label = QLabel("预设"); preset_label.setObjectName("quiet")
        rhead.addWidget(preset_label); rhead.addWidget(self.reletter_font_preset)
        rbl.addLayout(rhead)

        font_row = QHBoxLayout(); font_row.setSpacing(6)
        self.reletter_font = QLineEdit(); self.reletter_font.setPlaceholderText("字体文件、sans / serif / rounded / comic，或 A;B;C 字体链")
        self.reletter_font.setText(str(self.window.state.config.lettering.font_path or ""))
        self.reletter_font_pick = QPushButton("浏览"); self.reletter_font_pick.setObjectName("compactAction"); self.reletter_font_pick.setMaximumWidth(64)
        self.reletter_font_default = QPushButton("默认"); self.reletter_font_default.setObjectName("compactAction"); self.reletter_font_default.setMaximumWidth(64)
        font_row.addWidget(self.reletter_font,1); font_row.addWidget(self.reletter_font_pick); font_row.addWidget(self.reletter_font_default); rbl.addLayout(font_row)

        catalog_row=QHBoxLayout(); catalog_row.setSpacing(6)
        catalog_label = QLabel("字体库"); catalog_label.setObjectName("quiet"); catalog_label.setFixedWidth(44)
        self.reletter_font_catalog=QComboBox(); self.reletter_font_catalog.addItem("自动扫描字体库", "")
        for frow in discover_fonts(limit=160): self.reletter_font_catalog.addItem(str(frow.get("name") or Path(frow.get("path","")).stem), str(frow.get("path") or ""))
        self.reletter_font_refresh=QPushButton("刷新"); self.reletter_font_refresh.setObjectName("compactAction"); self.reletter_font_refresh.setMaximumWidth(64)
        catalog_row.addWidget(catalog_label); catalog_row.addWidget(self.reletter_font_catalog,1); catalog_row.addWidget(self.reletter_font_refresh); rbl.addLayout(catalog_row)

        self.reletter_break_mode = QComboBox()
        self.reletter_break_mode.addItem("智能断句", "smart")
        self.reletter_break_mode.addItem("均衡分行", "balanced")
        self.reletter_break_mode.addItem("优先保留原换行", "source")
        self.reletter_break_mode.setMinimumWidth(116)
        self.reletter_layout_mode = QComboBox()
        self.reletter_layout_mode.addItem("智能缩放", "smart_scaling")
        self.reletter_layout_mode.addItem("严格文本框", "strict")
        self.reletter_layout_mode.addItem("智能气泡", "balloon_fill")
        self.reletter_layout_mode.setMinimumWidth(112)
        self.reletter_min_font = QSpinBox(); self.reletter_min_font.setRange(6,96); self.reletter_min_font.setSuffix(" px"); self.reletter_min_font.setValue(int(self.window.state.config.lettering.min_font_size))
        self.reletter_max_font = QSpinBox(); self.reletter_max_font.setRange(8,160); self.reletter_max_font.setSuffix(" px"); self.reletter_max_font.setValue(int(self.window.state.config.lettering.max_font_size))
        self.reletter_line_spacing = QDoubleSpinBox(); self.reletter_line_spacing.setRange(0.0,0.6); self.reletter_line_spacing.setSingleStep(0.02); self.reletter_line_spacing.setDecimals(2); self.reletter_line_spacing.setValue(float(self.window.state.config.lettering.line_spacing_ratio))
        self.reletter_koharu_flow_cells = QCheckBox("联合气泡分区 · Koharu")
        self.reletter_koharu_flow_cells.setChecked(bool(getattr(self.window.state.config.lettering, "koharu_flow_cells_enabled", False)))
        self.reletter_koharu_flow_cells.setToolTip("仅 OCR重排：当一个连通气泡包含多段独立对白时，按物理颈部/文字锚点切成互不重叠的排字区。默认关闭，不影响现有结果。")

        layout_grid = QGridLayout(); layout_grid.setContentsMargins(0,0,0,0); layout_grid.setHorizontalSpacing(7); layout_grid.setVerticalSpacing(6)
        layout_grid.addWidget(QLabel("断句"), 0, 0); layout_grid.addWidget(self.reletter_break_mode, 0, 1)
        layout_grid.addWidget(QLabel("适配"), 1, 0); layout_grid.addWidget(self.reletter_layout_mode, 1, 1)
        size_box = QWidget(); size_lay = QHBoxLayout(size_box); size_lay.setContentsMargins(0,0,0,0); size_lay.setSpacing(5)
        self.reletter_min_font.setMinimumWidth(78); self.reletter_max_font.setMinimumWidth(78); self.reletter_line_spacing.setMinimumWidth(82)
        size_lay.addWidget(QLabel("最小")); size_lay.addWidget(self.reletter_min_font)
        size_lay.addWidget(QLabel("最大")); size_lay.addWidget(self.reletter_max_font)
        size_lay.addSpacing(6); size_lay.addWidget(QLabel("行距")); size_lay.addWidget(self.reletter_line_spacing)
        size_lay.addStretch(1)
        layout_grid.addWidget(QLabel("字号"), 2, 0); layout_grid.addWidget(size_box, 2, 1)
        layout_grid.addWidget(self.reletter_koharu_flow_cells, 3, 0, 1, 2)
        layout_grid.setColumnStretch(1, 1)
        rbl.addLayout(layout_grid)
        self.reletter_section = CompactSection("OCR重排 · 字体与排版", "字体、断句、字号与行距 · 低频设置默认收起", expanded=False)
        self.reletter_section.content_layout.addWidget(reletter_box)
        mode.layout.addWidget(self.reletter_section)
        self.transparent_backend = QComboBox()
        for label, value in [
            ("后备检测 · 自动", "auto"), ("仅 Koharu Layout", "koharu_layout"), ("后备 · MangaLens", "mangalens"),
            ("后备 · 白气泡/文字种子", "seeded_white"), ("后备 · 白气泡/无种子", "unseeded_white"),
            ("后备 · RT-DETR-v2", "rtdetr_v2"), ("后备 · SAM2", "sam2"),
        ]: self.transparent_backend.addItem(label, value)
        self.transparent_clear_mode = QComboBox()
        self.transparent_clear_mode.addItem("混合：白色封闭框整区透明 / 彩色效果仅文字（推荐）", "hybrid")
        self.transparent_clear_mode.addItem("封闭白气泡/文本框整区透明（开放文字仍只透明文字）", "full_bubble")
        self.transparent_clear_mode.addItem("全页仅文字墨迹透明（最保守）", "text_only")
        self.transparent_protect_border = QCheckBox("透明洞保护气泡边框")
        self.transparent_protect_border.setChecked(True)
        self.transparent_suppress_page_furniture = QCheckBox("过滤页眉 / 页脚文字（推荐）")
        self.transparent_suppress_page_furniture.setChecked(bool(getattr(self.window.state.config.transparent_bubble_reveal, "suppress_page_furniture", True)))
        self.transparent_suppress_page_furniture.setToolTip("独立过滤靠近页面边缘的运行页眉、页脚等版面文字。OCR 会把页眉判为真实文字，因此该规则必须独立于 OCR / SOURCE gate。")
        self.transparent_verify_text_presence = QCheckBox("候选区必须验证 TARGET 文字存在（推荐）")
        self.transparent_verify_text_presence.setChecked(bool(getattr(self.window.state.config.transparent_bubble_reveal, "verify_target_text_presence", True)))
        self.transparent_verify_text_presence.setToolTip("对整页对齐模式的所有候选区做第二层 TARGET 文字存在验证，尽量过滤头发高光、汗滴、阴影碎块等假阳性。")
        self.transparent_ocr_text_presence = QCheckBox("OCR 二次确认候选区确实有文字（可选）")
        self.transparent_ocr_text_presence.setChecked(bool(getattr(self.window.state.config.transparent_bubble_reveal, "target_text_presence_ocr_enabled", False)))
        self.transparent_ocr_text_presence.setToolTip("只把当前 TARGET OCR 当作“有没有文字”的验证器，不参与翻译或重排。开启后可进一步过滤头发、高光、汗滴、线稿碎块等视觉假阳性；OCR 不可用时默认回退到 TARGET 文字结构验证。")
        self.transparent_restore_source_evidence = QCheckBox("恢复 SOURCE 翻译证据 gate（更保守）")
        self.transparent_restore_source_evidence.setChecked(bool(getattr(self.window.state.config.transparent_bubble_reveal, "require_source_translation_evidence", False)))
        self.transparent_restore_source_evidence.setToolTip("开启后，候选区除了 TARGET 几何/文字存在验证外，还必须满足 SOURCE↔TARGET 的翻译差异证据才会被替换。适合压制章节标题、无翻译开放字等真文本假候选。")
        self.transparent_expand = QSpinBox(); self.transparent_expand.setRange(0, 12); self.transparent_expand.setSuffix(" px"); self.transparent_expand.setValue(2)
        self.transparent_feather = QSpinBox(); self.transparent_feather.setRange(0, 8); self.transparent_feather.setSuffix(" px"); self.transparent_feather.setValue(1)
        self.transparent_widgets = [self.transparent_backend, self.transparent_clear_mode, self.transparent_protect_border, self.transparent_suppress_page_furniture, self.transparent_verify_text_presence, self.transparent_ocr_text_presence, self.transparent_restore_source_evidence, self.transparent_expand, self.transparent_feather]
        # Renderer-specific controls must disappear completely outside their
        # renderer.  Earlier builds merely disabled these widgets while their
        # QFormLayout labels and long combo boxes still consumed geometry, which
        # was one source of the Page Manager overlap on narrower screens.
        self.transparent_box = QFrame(); self.transparent_box.setObjectName("cardBlue")
        tbl = QGridLayout(self.transparent_box); tbl.setContentsMargins(9,8,9,8); tbl.setHorizontalSpacing(7); tbl.setVerticalSpacing(6)
        tbl.addWidget(QLabel("Koharu 后备"), 0, 0); tbl.addWidget(self.transparent_backend, 0, 1, 1, 3)
        tbl.addWidget(QLabel("清空策略"), 1, 0); tbl.addWidget(self.transparent_clear_mode, 1, 1, 1, 3)
        tbl.addWidget(QLabel("向外扩张"), 2, 0); tbl.addWidget(self.transparent_expand, 2, 1)
        tbl.addWidget(QLabel("边缘羽化"), 2, 2); tbl.addWidget(self.transparent_feather, 2, 3)
        tbl.addWidget(self.transparent_protect_border, 3, 0, 1, 4)
        tbl.addWidget(self.transparent_suppress_page_furniture, 4, 0, 1, 4)
        tbl.addWidget(self.transparent_verify_text_presence, 5, 0, 1, 4)
        tbl.addWidget(self.transparent_ocr_text_presence, 6, 0, 1, 4)
        tbl.addWidget(self.transparent_restore_source_evidence, 7, 0, 1, 4)
        tbl.setColumnStretch(1, 1); tbl.setColumnStretch(3, 1)
        self.transparent_section = CompactSection("整页对齐透明显中文 · 设置", "独立透明 Reveal：TARGET RGBA 上层透明，下层为整页对齐 SOURCE；不调用挖孔模式", expanded=False)
        self.transparent_section.content_layout.addWidget(self.transparent_box)
        mode.layout.addWidget(self.transparent_section)

        # v2.2 semantic front-end: this never replaces Direct/Mask/Reveal renderers.
        self.semantic_box = QFrame(); self.semantic_box.setObjectName("cardBlue")
        sgl = QGridLayout(self.semantic_box); sgl.setContentsMargins(9,8,9,8); sgl.setHorizontalSpacing(7); sgl.setVerticalSpacing(6)
        self.semantic_enabled = QCheckBox("启用漫画语义版面分析")
        self.semantic_enabled.setChecked(bool(getattr(self.window.state.config.semantic, "enabled", False)))
        self.semantic_backend = QComboBox()
        self.semantic_backend.addItem("自动：PP-DocLayoutV3 → 内置回退", "auto")
        self.semantic_backend.addItem("PP-DocLayoutV3", "pp_doclayout_v3")
        self.semantic_backend.addItem("内置 Manga Layout 回退", "heuristic")
        self.semantic_strategy = QComboBox()
        self.semantic_strategy.addItem("自动：页码/页眉忽略，正文处理", "auto")
        self.semantic_strategy.addItem("严格：低置信全部复核", "strict")
        self.semantic_strategy.addItem("宽松：尽量保留候选", "loose")
        self.semantic_strategy.addItem("仅分析：不参与处理", "analysis_only")
        self.semantic_apply_reveal = QCheckBox("让整页 Reveal 使用语义 ROI / 忽略区")
        self.semantic_apply_reveal.setChecked(bool(getattr(self.window.state.config.semantic, "apply_to_reveal", True)))
        self.semantic_save_overlay = QCheckBox("保存语义叠加图（调试）")
        self.semantic_save_overlay.setChecked(bool(getattr(self.window.state.config.semantic, "save_overlay", False)))
        sgl.addWidget(self.semantic_enabled, 0, 0, 1, 4)
        sgl.addWidget(QLabel("语义后端"), 1, 0); sgl.addWidget(self.semantic_backend, 1, 1, 1, 3)
        sgl.addWidget(QLabel("语义策略"), 2, 0); sgl.addWidget(self.semantic_strategy, 2, 1, 1, 3)
        sgl.addWidget(self.semantic_apply_reveal, 3, 0, 1, 4)
        sgl.addWidget(self.semantic_save_overlay, 4, 0, 1, 4)
        self.semantic_section = CompactSection("高级 · 漫画语义版面", "页码/页眉忽略、正文 ROI 与调试叠加图", expanded=False)
        self.semantic_section.content_layout.addWidget(self.semantic_box)
        mode.layout.addWidget(self.semantic_section)

        self.show_experimental = QCheckBox("显示旧版实验兼容项")
        self.experimental_warning = QLabel("整页模式现已使用独立 aligned_overlay_reveal 路线；不会再映射到 Transparent Reveal。")
        self.experimental_warning.setObjectName("hint"); self.experimental_warning.setWordWrap(True); self.experimental_warning.setVisible(False)
        self.direct_status = QLabel("")
        self.direct_status.setWordWrap(True)
        self.direct_status.setVisible(False)
        self.direct_result_status = QLabel("")
        self.direct_result_status.setWordWrap(True)
        self.direct_result_status.setVisible(False)
        self.experimental_status = QLabel("")
        self.experimental_status.setWordWrap(True)
        self.experimental_status.setVisible(False)
        self.experimental_result_status = QLabel("")
        self.experimental_result_status.setWordWrap(True)
        self.experimental_result_status.setVisible(False)
        mode.layout.addWidget(self.direct_status); mode.layout.addWidget(self.direct_result_status); mode.layout.addWidget(self.show_experimental); mode.layout.addWidget(self.experimental_warning); mode.layout.addWidget(self.experimental_status); mode.layout.addWidget(self.experimental_result_status)
        self.show_experimental.setVisible(False)
        self.experimental_warning.setVisible(False)
        self.direct_status.setVisible(False)
        self.direct_result_status.setVisible(False)
        self.experimental_status.setVisible(False)
        self.experimental_result_status.setVisible(False)
        self.direct_contract_box = QFrame(); self.direct_contract_box.setObjectName("cardBlue")
        dcl = QVBoxLayout(self.direct_contract_box); dcl.setContentsMargins(9,8,9,8); dcl.setSpacing(4)
        dct = QLabel("Direct 固定流程 · 模式专属"); dct.setStyleSheet("font-weight:650;"); dcl.addWidget(dct)
        self.direct_contract_source_top = QCheckBox("SOURCE 中文层在上 / TARGET 日文层在下")
        self.direct_contract_borderless = QCheckBox("不复制 SOURCE 边框 · TARGET 边框为唯一权威")
        self.direct_contract_clear_target = QCheckBox("先处理 TARGET 日文残留，再叠加中文")
        self.direct_contract_axis_lock = QCheckBox("最终中文贴图方向锁定 · 不翻转 / 不额外旋转 / 不局部漂移")
        for _w in (self.direct_contract_source_top, self.direct_contract_borderless, self.direct_contract_clear_target, self.direct_contract_axis_lock):
            _w.setChecked(True); _w.setEnabled(False); dcl.addWidget(_w)
        dcn = QLabel("以上 4 项属于 Direct 默认合同；除非明确修改 Direct 模式，否则不得改变。")
        dcn.setObjectName("quiet"); dcn.setWordWrap(True); dcl.addWidget(dcn)
        self.direct_contract_section = CompactSection("Direct 流程说明", "SOURCE 中文上层 · TARGET 边框权威 · 方向锁定", expanded=False)
        self.direct_contract_section.content_layout.addWidget(self.direct_contract_box)
        mode.layout.addWidget(self.direct_contract_section)

        self.direct_clarity_box = QFrame(); self.direct_clarity_box.setObjectName("cardBlue")
        dql = QVBoxLayout(self.direct_clarity_box); dql.setContentsMargins(9,8,9,8); dql.setSpacing(5)
        dqt = QLabel("白气泡中文清晰增强"); dqt.setStyleSheet("font-weight:650;"); dql.addWidget(dqt)
        self.direct_white_clarity_enabled = QCheckBox("启用漂白清底 + 中文文字掩膜清晰化")
        self.direct_white_clarity_enabled.setToolTip("参考夸克类净页思路：对白气泡区域先漂白清底、去灰底、压噪点，再保留 SOURCE 中文黑字。对白气泡场景下的 Direct / 精准蒙版 / 精准蒙版+OCR 生效：先漂白清底、去灰底、压噪点，再保留 SOURCE 中文黑字；彩页与非白气泡仍走原有安全路径。")
        dql.addWidget(self.direct_white_clarity_enabled)
        dqh = QLabel("用于 Direct / 精准蒙版 / 精准蒙版+OCR 的白气泡。精准蒙版与精准蒙版+OCR 默认关闭；关闭时保留对齐后的 SOURCE 中文字形，不再进入锐化 / photo-crisp / ink 重建。")
        dqh.setObjectName("quiet"); dqh.setWordWrap(True); dql.addWidget(dqh)
        self.direct_clarity_summary = QLabel("")
        self.direct_clarity_summary.setObjectName("hint"); self.direct_clarity_summary.setWordWrap(True); dql.addWidget(self.direct_clarity_summary)
        dqf = QGridLayout(); dqf.setContentsMargins(0,0,0,0); dqf.setHorizontalSpacing(8); dqf.setVerticalSpacing(6)
        self.direct_white_clarity_alpha_gamma = QDoubleSpinBox(); self.direct_white_clarity_alpha_gamma.setDecimals(2); self.direct_white_clarity_alpha_gamma.setRange(0.40, 1.20); self.direct_white_clarity_alpha_gamma.setSingleStep(0.02)
        self.direct_white_clarity_alpha_gamma.setToolTip("文字边缘灰阶曲线。1.00 保留中文版原始圆润抗锯齿；小于 1.00 才会主动加硬。默认 1.00。")
        self.direct_white_clarity_black_boost = QSpinBox(); self.direct_white_clarity_black_boost.setRange(0, 48); self.direct_white_clarity_black_boost.setSingleStep(1)
        self.direct_white_clarity_black_boost.setToolTip("黑字强化。0 表示不改变中文版原字灰阶；仅在确有需要时手动增加。默认 0。")
        self.direct_white_clarity_pure_white_floor = QSpinBox(); self.direct_white_clarity_pure_white_floor.setRange(235, 255); self.direct_white_clarity_pure_white_floor.setSingleStep(1)
        self.direct_white_clarity_pure_white_floor.setToolTip("白底最低亮度。数值越高，背景越接近纯白。默认 248。")
        self.direct_white_clarity_min_text_pixels = QSpinBox(); self.direct_white_clarity_min_text_pixels.setRange(1, 400); self.direct_white_clarity_min_text_pixels.setSingleStep(1)
        self.direct_white_clarity_min_text_pixels.setToolTip("最小文字像素门槛。过小区域不做增强，避免误对白纸块做无意义处理。默认 18。")
        for _spin in (self.direct_white_clarity_alpha_gamma, self.direct_white_clarity_black_boost, self.direct_white_clarity_pure_white_floor, self.direct_white_clarity_min_text_pixels):
            _spin.setMinimumWidth(88); _spin.setMaximumWidth(124)
        _dcfg = self.window.state.config.direct_patch
        self.direct_white_clarity_enabled.setChecked(bool(getattr(_dcfg, "direct_white_clarity_enhance_enabled", True)))
        self.direct_white_clarity_alpha_gamma.setValue(float(getattr(_dcfg, "direct_white_clarity_alpha_gamma", 1.0)))
        self.direct_white_clarity_black_boost.setValue(int(getattr(_dcfg, "direct_white_clarity_black_boost", 0)))
        self.direct_white_clarity_pure_white_floor.setValue(int(getattr(_dcfg, "direct_white_clarity_pure_white_floor", 248)))
        self.direct_white_clarity_min_text_pixels.setValue(int(getattr(_dcfg, "direct_white_clarity_min_text_pixels", 18)))
        dqf.addWidget(QLabel("字边清晰"), 0, 0); dqf.addWidget(self.direct_white_clarity_alpha_gamma, 0, 1)
        dqf.addWidget(QLabel("黑字强化"), 0, 2); dqf.addWidget(self.direct_white_clarity_black_boost, 0, 3)
        dqf.addWidget(QLabel("白底亮度"), 1, 0); dqf.addWidget(self.direct_white_clarity_pure_white_floor, 1, 1)
        dqf.addWidget(QLabel("最小文字像素"), 1, 2); dqf.addWidget(self.direct_white_clarity_min_text_pixels, 1, 3)
        dqf.setColumnStretch(1, 1); dqf.setColumnStretch(3, 1)
        dql.addLayout(dqf)
        self.direct_clarity_section = CompactSection("白气泡中文清晰增强", "漂白清底 + 保留 SOURCE 原始中文字形", expanded=False)
        self.direct_clarity_section.content_layout.addWidget(self.direct_clarity_box)
        mode.layout.addWidget(self.direct_clarity_section)

        self.hybrid_ocr_contract_box = QFrame(); self.hybrid_ocr_contract_box.setObjectName("cardBlue")
        hol = QVBoxLayout(self.hybrid_ocr_contract_box); hol.setContentsMargins(9,8,9,8); hol.setSpacing(4)
        hot = QLabel("精准蒙版+OCR · OCR 固定合同"); hot.setStyleSheet("font-weight:650;"); hol.addWidget(hot)
        self.hybrid_ocr_mask_claim = QCheckBox("有任何精准蒙版候选 → 禁止自动 OCR")
        self.hybrid_ocr_uncovered_only = QCheckBox("完全无蒙版覆盖 → 才允许自动 OCR")
        self.hybrid_ocr_manual_force = QCheckBox("人工 OCR 框选 → 按用户要求强制 OCR")
        for _w in (self.hybrid_ocr_mask_claim, self.hybrid_ocr_uncovered_only, self.hybrid_ocr_manual_force):
            _w.setChecked(True); _w.setEnabled(False); hol.addWidget(_w)
        hon = QLabel("蒙版结果即使进入 REVIEW 或回滚 TARGET，也仍然拥有该区域；系统不会擅自用 OCR 覆盖。需要 OCR 时可在人工 OCR 编辑器中手动框选。")
        hon.setObjectName("quiet"); hon.setWordWrap(True); hol.addWidget(hon)
        self.hybrid_ocr_contract_section = CompactSection("精准蒙版+OCR · OCR 合同", "有蒙版不自动 OCR · 无蒙版才 OCR · 人工框选强制", expanded=False)
        self.hybrid_ocr_contract_section.content_layout.addWidget(self.hybrid_ocr_contract_box)
        mode.layout.addWidget(self.hybrid_ocr_contract_section)

        self.diff_check = QCheckBox("优先使用成对差异提取中文气泡/文本框"); self.diff_check.setChecked(True)
        self.exact_check = QCheckBox("同源页面启用像素级精确覆盖"); self.exact_check.setChecked(True)
        self.mode_core_box = QFrame(); self.mode_core_box.setObjectName("compactOptions")
        core_l = QVBoxLayout(self.mode_core_box); core_l.setContentsMargins(8,5,8,5); core_l.setSpacing(4)
        core_l.addWidget(self.diff_check); core_l.addWidget(self.exact_check)
        # Put high-frequency switches directly below the mode summary; all longer
        # renderer panels stay in collapsible sections below.
        mode.layout.insertWidget(2, self.mode_core_box)
        right_col.addWidget(mode, 0)

        # Selected-page detail is a collapsible, vertically elastic inspector.
        # It absorbs whatever space remains after renderer-specific controls: Auto
        # has more room, while Reletter/Transparent can naturally make it shorter.
        self.detail_card = QFrame(); self.detail_card.setObjectName("card")
        detail_card_layout = QVBoxLayout(self.detail_card); detail_card_layout.setContentsMargins(12,9,12,11); detail_card_layout.setSpacing(8)
        detail_header = QHBoxLayout(); detail_header.setContentsMargins(0,0,0,0); detail_header.setSpacing(8)
        detail_title = QLabel("当前页详情"); detail_title.setObjectName("cardTitle")
        self.detail_toggle = QPushButton("展开"); self.detail_toggle.setObjectName("collapseToggle"); self.detail_toggle.setCheckable(True); self.detail_toggle.setChecked(False)
        detail_header.addWidget(detail_title); detail_header.addStretch(1); detail_header.addWidget(self.detail_toggle)
        detail_card_layout.addLayout(detail_header)
        self.detail_content = QWidget()
        detail_content_layout = QVBoxLayout(self.detail_content); detail_content_layout.setContentsMargins(0,0,0,0); detail_content_layout.setSpacing(8)
        self.detail_body = QWidget(); detail_body_layout = QHBoxLayout(self.detail_body); detail_body_layout.setContentsMargins(0,0,0,0); detail_body_layout.setSpacing(12)
        # The preview is the primary content of the inspector.  Keep the metadata
        # column deliberately narrow and let the image consume the remaining width.
        # This avoids the v2.0.72 layout where a 230px preview sat beside a very wide
        # text column even when the inspector had plenty of horizontal space.
        self.detail_view = ImageView(max_decode_side=1100); self.detail_view.setMinimumSize(280,240)
        self.detail_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        detail_body_layout.addWidget(self.detail_view, 5)
        self.detail_info_panel = QWidget(); self.detail_info_panel.setObjectName("detailInfoPanel")
        self.detail_info_panel.setMinimumWidth(220); self.detail_info_panel.setMaximumWidth(280)
        self.detail_info_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        detail_info = QVBoxLayout(self.detail_info_panel); detail_info.setContentsMargins(0,0,0,0); detail_info.setSpacing(5)
        dhead = QHBoxLayout(); self.detail_page = QLabel("未选择页面"); self.detail_page.setObjectName("sectionTitle")
        self.detail_badge = QLabel("—"); self.detail_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dhead.addWidget(self.detail_page); dhead.addStretch(1); dhead.addWidget(self.detail_badge); detail_info.addLayout(dhead)
        side = QHBoxLayout(); side.setSpacing(4)
        self.detail_target_btn = QPushButton("高清日文"); self.detail_target_btn.setObjectName("segmented"); self.detail_target_btn.setCheckable(True); self.detail_target_btn.setChecked(True)
        self.detail_source_btn = QPushButton("旧版中文"); self.detail_source_btn.setObjectName("segmented"); self.detail_source_btn.setCheckable(True)
        self.detail_side_group = QButtonGroup(self); self.detail_side_group.setExclusive(True); self.detail_side_group.addButton(self.detail_target_btn); self.detail_side_group.addButton(self.detail_source_btn)
        side.addWidget(self.detail_target_btn); side.addWidget(self.detail_source_btn); detail_info.addLayout(side)
        self.detail_names = QLabel(); self.detail_names.setObjectName("quiet"); self.detail_names.setWordWrap(True); self.detail_names.setMaximumHeight(38); detail_info.addWidget(self.detail_names)
        self.detail_reason = QLabel(); self.detail_reason.setObjectName("hint"); self.detail_reason.setWordWrap(True); self.detail_reason.setMaximumHeight(46); detail_info.addWidget(self.detail_reason)
        self.detail_stats = QLabel(); self.detail_stats.setObjectName("quiet"); self.detail_stats.setWordWrap(False); detail_info.addWidget(self.detail_stats)
        detail_info.addStretch(1)
        detail_body_layout.addWidget(self.detail_info_panel, 2)
        detail_body_layout.setStretch(0, 5); detail_body_layout.setStretch(1, 2)
        detail_content_layout.addWidget(self.detail_body, 1)
        # Actions span the full card instead of stealing width from the metadata
        # column.  They remain easy to hit even after whole-page scaling.
        detail_actions = QHBoxLayout(); detail_actions.setSpacing(7)
        self.open_preview_btn = QPushButton("放大 / 打开大图"); self.open_preview_btn.setObjectName("softPrimary")
        self.go_workbench_btn = QPushButton("进入替换工作台")
        detail_actions.addWidget(self.open_preview_btn,1); detail_actions.addWidget(self.go_workbench_btn,1)
        detail_content_layout.addLayout(detail_actions)
        detail_card_layout.addWidget(self.detail_content, 1)
        self.detail_card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.detail_toggle.clicked.connect(self._set_detail_expanded)
        self._set_detail_expanded(False)

        # Current-project status is intentionally one compact row.
        self.summary_card = QFrame(); self.summary_card.setObjectName("card")
        self.summary_card.setMinimumHeight(48); self.summary_card.setMaximumHeight(56)
        summary_layout = QHBoxLayout(self.summary_card); summary_layout.setContentsMargins(14,7,14,7); summary_layout.setSpacing(10)
        summary_title = QLabel("当前项目"); summary_title.setObjectName("cardTitle")
        self.sum_pairs = QLabel("0 页"); self.sum_pairs.setStyleSheet(f"font-size:17px;font-weight:750;color:{self._theme_colors()['accent']};")
        self.sum_hint = QLabel("等待页面配对"); self.sum_hint.setObjectName("quiet"); self.sum_hint.setWordWrap(False)
        summary_layout.addWidget(summary_title); summary_layout.addWidget(self.sum_pairs); summary_layout.addWidget(self.sum_hint, 1)
        right_col.addWidget(self.summary_card, 0)

        actions = Card("开始处理")
        self.actions_card = actions
        actions.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        # Kept as an internal compatibility widget for older saved UI state, but
        # batch semantics are now explicit buttons instead of a confusing toggle.
        self.resume_check = QCheckBox("断点续跑 / 跳过已完成页面"); self.resume_check.setChecked(True); self.resume_check.setVisible(False)
        self.cache_check = QCheckBox("复用配准、OCR 与气泡缓存"); self.cache_check.setChecked(True)
        self.run_page = QPushButton("处理当前页")
        self.run_book = QPushButton("从头处理整本")
        self.continue_book = QPushButton("继续处理整本")
        self.cancel = QPushButton("停止"); self.cancel.setEnabled(False)
        # The four processing actions are one semantic group.  Keep identical
        # height, width policy and palette so current-page/batch semantics do
        # not accidentally turn into a visual priority hierarchy.
        for _btn in (self.run_page, self.cancel, self.run_book, self.continue_book):
            _btn.setObjectName("pageProcessAction")
            _btn.setMinimumHeight(38); _btn.setMaximumHeight(38)
            _btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Two-by-two grid guarantees equal button widths in both rows, including
        # the Stop button which used to carry a fixed 96..132px width.
        process_grid = QGridLayout(); process_grid.setContentsMargins(0,0,0,0)
        process_grid.setHorizontalSpacing(8); process_grid.setVerticalSpacing(8)
        process_grid.addWidget(self.run_page, 0, 0)
        process_grid.addWidget(self.cancel, 0, 1)
        process_grid.addWidget(self.run_book, 1, 0)
        process_grid.addWidget(self.continue_book, 1, 1)
        process_grid.setColumnStretch(0, 1); process_grid.setColumnStretch(1, 1)
        actions.layout.addLayout(process_grid)

        action_opts = QHBoxLayout(); action_opts.setContentsMargins(0,2,0,0); action_opts.setSpacing(8)
        action_opts.addWidget(self.cache_check); action_opts.addStretch(1)
        actions.layout.addLayout(action_opts)
        right_col.addWidget(actions, 0)
        # Only the current-page inspector may absorb spare right-column height.
        # This prevents migration/actions cards from being stretched into large
        # blank boxes when the inspector is collapsed or a renderer has few options.
        right_col.addWidget(self.detail_card, 1)
        root.addLayout(right_col, 1)

        self.source.button.clicked.connect(lambda: self.window.choose_directory("source")); self.target.button.clicked.connect(lambda: self.window.choose_directory("target")); self.output.button.clicked.connect(lambda: self.window.choose_directory("output"))
        self.pair_btn.clicked.connect(self.window.auto_pair); self.restore_btn.clicked.connect(self.window.restore_existing_results); self.refresh_btn.clicked.connect(self._force_thumbnail_refresh)
        self.table.itemSelectionChanged.connect(self._table_selection_changed); self.table.cellDoubleClicked.connect(lambda row, _col: self.open_preview(row))
        self.thumb_list.itemSelectionChanged.connect(self._thumb_selection_changed)
        self.thumb_list.itemDoubleClicked.connect(lambda item: self.open_preview(int(item.data(Qt.ItemDataRole.UserRole))))
        self.thumb_list.customContextMenuRequested.connect(self._thumb_context_menu)
        self.table.customContextMenuRequested.connect(self._table_context_menu)
        self.thumb_view_btn.clicked.connect(lambda: self._set_view_mode(0)); self.list_view_btn.clicked.connect(lambda: self._set_view_mode(1))
        self.thumb_target_btn.clicked.connect(lambda: self._set_thumb_side("target")); self.thumb_source_btn.clicked.connect(lambda: self._set_thumb_side("source"))
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)
        self.detail_target_btn.clicked.connect(lambda: self._set_detail_side("target")); self.detail_source_btn.clicked.connect(lambda: self._set_detail_side("source"))
        self.open_preview_btn.clicked.connect(lambda: self.open_preview(self.window.state.selected_index)); self.go_workbench_btn.clicked.connect(self._go_workbench)
        self.mode.currentIndexChanged.connect(self._on_mode_changed); self.show_experimental.toggled.connect(self._set_experimental_visible); self.diff_check.toggled.connect(self._sync_config); self.exact_check.toggled.connect(self._sync_config)
        self.direct_white_clarity_enabled.toggled.connect(self._sync_config); self.direct_white_clarity_enabled.toggled.connect(self._update_direct_clarity_controls)
        self.direct_white_clarity_alpha_gamma.valueChanged.connect(self._sync_config); self.direct_white_clarity_black_boost.valueChanged.connect(self._sync_config); self.direct_white_clarity_pure_white_floor.valueChanged.connect(self._sync_config); self.direct_white_clarity_min_text_pixels.valueChanged.connect(self._sync_config)
        self.reletter_font.textChanged.connect(self._sync_config); self.reletter_min_font.valueChanged.connect(self._sync_config); self.reletter_max_font.valueChanged.connect(self._sync_config); self.reletter_line_spacing.valueChanged.connect(self._sync_config); self.reletter_break_mode.currentIndexChanged.connect(self._sync_config); self.reletter_layout_mode.currentIndexChanged.connect(self._sync_config); self.reletter_koharu_flow_cells.toggled.connect(self._sync_config)
        self.reletter_font_pick.clicked.connect(self._pick_reletter_font); self.reletter_font_default.clicked.connect(self._clear_reletter_font)
        self.reletter_font_preset.currentIndexChanged.connect(self._apply_reletter_font_preset)
        self.reletter_font_refresh.clicked.connect(self._refresh_reletter_font_catalog)
        self.reletter_font_catalog.currentIndexChanged.connect(self._apply_reletter_catalog_font)
        self.transparent_backend.currentIndexChanged.connect(self._sync_config); self.transparent_clear_mode.currentIndexChanged.connect(self._sync_config); self.transparent_protect_border.toggled.connect(self._sync_config); self.transparent_suppress_page_furniture.toggled.connect(self._sync_config); self.transparent_verify_text_presence.toggled.connect(self._sync_config); self.transparent_ocr_text_presence.toggled.connect(self._sync_config); self.transparent_restore_source_evidence.toggled.connect(self._sync_config); self.transparent_expand.valueChanged.connect(self._sync_config); self.transparent_feather.valueChanged.connect(self._sync_config)
        self.semantic_enabled.toggled.connect(self._sync_config); self.semantic_backend.currentIndexChanged.connect(self._sync_config); self.semantic_strategy.currentIndexChanged.connect(self._sync_config); self.semantic_apply_reveal.toggled.connect(self._sync_config); self.semantic_save_overlay.toggled.connect(self._sync_config)
        self.prefer_name_pair.toggled.connect(self._sync_config); self.prefer_order_pair.toggled.connect(self._sync_config); self.remake_pair_verify.toggled.connect(self._sync_config)
        self.run_page.clicked.connect(self.window.run_current_page); self.run_book.clicked.connect(self.window.run_book); self.continue_book.clicked.connect(self.window.continue_book); self.cancel.clicked.connect(self.window.cancel_worker)
        self.cache_check.toggled.connect(self._sync_config)
        self._update_mode_specific_controls()
        self._update_direct_status()
        self._update_direct_result_status()
        self.apply_type.clicked.connect(self._apply_selected_type); self.reset_type.clicked.connect(self._reset_selected_type)
        self._update_experimental_status()

    def set_processing_busy(self, busy: bool) -> None:
        """Lock renderer settings while a worker is running.

        The action card stays interactive so the Stop button remains usable; only
        settings that could make the running job diverge from the visible UI are
        locked.
        """
        busy = bool(busy)
        if hasattr(self, "mode_card"):
            self.mode_card.setEnabled(not busy)
        if hasattr(self, "cache_check"):
            self.cache_check.setEnabled(not busy)
        if not busy and hasattr(self, "mode"):
            self._update_mode_specific_controls()

    def _set_detail_expanded(self, expanded: bool | None = None):
        if expanded is None:
            expanded = bool(self.detail_toggle.isChecked())
        expanded = bool(expanded)
        self.detail_toggle.blockSignals(True)
        self.detail_toggle.setChecked(expanded)
        self.detail_toggle.setText("收起" if expanded else "展开")
        self.detail_toggle.blockSignals(False)
        self.detail_content.setVisible(expanded)
        if expanded:
            self.detail_card.setMinimumHeight(0)
            self.detail_card.setMaximumHeight(16777215)
            self.detail_card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        else:
            # 58px safely contains the title + 32px toggle + card margins at
            # 100% design scale, so the collapsed row is never vertically clipped.
            self.detail_card.setMinimumHeight(58)
            self.detail_card.setMaximumHeight(58)
            self.detail_card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.detail_card.updateGeometry()

    def _set_project_files_expanded(self, expanded: bool | None = None):
        value = self.project_toggle.isChecked() if expanded is None else bool(expanded)
        self.project_toggle.blockSignals(True)
        try:
            self.project_toggle.setChecked(value)
        finally:
            self.project_toggle.blockSignals(False)
        self.project_body.setVisible(value)
        self.project_toggle.setText("收起" if value else "展开")

    def _theme_colors(self) -> dict[str, str]:
        return semantic_palette(getattr(self.window, "theme_name", "light"))

    def _tone_style(self, tone: str) -> str:
        colors = self._theme_colors()
        return (
            f"color:{colors[tone]}; background:{colors[tone + '_soft']}; "
            f"border:1px solid {colors[tone + '_border']}; border-radius:8px; padding:6px 8px;"
        )

    def apply_theme(self) -> None:
        """Refresh legacy inline-styled elements after a global theme switch."""
        colors = self._theme_colors()
        if hasattr(self, "sum_pairs"):
            self.sum_pairs.setStyleSheet(f"font-size:18px;font-weight:700;color:{colors['accent']};")
        if hasattr(self, "direct_status"):
            self._update_direct_status()
        if hasattr(self, "direct_result_status"):
            self._update_direct_result_status()
        if hasattr(self, "experimental_status"):
            self._update_experimental_status()
        if hasattr(self, "experimental_result_status"):
            self._update_experimental_result_status()
        # Thumbnail image pixels stay cached; only the small neutral canvas/border is repainted.
        for index, item in list(getattr(self, "_thumb_items", {}).items()):
            try:
                item.setIcon(self._thumbnail_icon(index) if index in self._thumb_loaded else self._placeholder_icon(index))
            except (IndexError, RuntimeError):
                continue

    def _set_experimental_visible(self, visible: bool):
        # v2.2.2: aligned_overlay_reveal is now the normal whole-page route.
        # The legacy hidden checkbox must never add/remove/switch the real mode item.
        self.experimental_warning.setVisible(False)
        self._update_direct_status()
        self._update_direct_result_status()
        self._update_experimental_status()
        self._update_experimental_result_status()

    @staticmethod
    def _direct_patch_reason_label(reason: str) -> str:
        reason = str(reason or "").strip()
        labels = {
            "ok": "已执行 Direct 主链路",
            "no_direct_patch_payload": "暂无 Direct 产物记录",
            "no_regions_applied": "没有可写入的 Direct 区域",
            "direct_patch_rejected": "Direct 安全门槛未通过，页面保持原样",
        }
        if reason.startswith("direct_patch_rejected:"):
            return "Direct 被主链路拒绝：" + reason.split(":", 1)[1]
        return labels.get(reason, reason or "未提供原因")

    def _update_direct_status(self):
        current = str(self.mode.currentData() or "")
        active = current == "direct_patch"
        self.direct_status.setVisible(active)
        if not active:
            return
        self.direct_status.setText("直接贴图主链路：SOURCE 中文在上、TARGET 日文在下；白气泡/白文本框采用无边框内层贴图，保留 TARGET 边框；彩色/开放区域保留 TARGET 背景，仅叠加 SOURCE 中文。")
        self.direct_status.setStyleSheet(self._tone_style("green"))
        self.direct_status.setToolTip("绿色 = GUI 当前选择的就是 Direct 主运行链路")
        self.mode.setToolTip("直接贴图主链路：direct_patch")

    def _update_direct_result_status(self, index: int | None = None):
        if not hasattr(self, "direct_result_status"):
            return
        if str(self.mode.currentData() or "") != "direct_patch":
            self.direct_result_status.setVisible(False)
            return
        pairs = self.window.state.pairs
        idx = self.window.state.selected_index if index is None else int(index)
        if not (0 <= idx < len(pairs)):
            self.direct_result_status.setVisible(True)
            self.direct_result_status.setText("当前页 Direct 结果：尚未选择页面。")
            self.direct_result_status.setStyleSheet(f"color:{self._theme_colors()['muted']};")
            return
        pair = pairs[idx]
        ws = resolve_page_workspace(self.window.state.output_dir, pair, self.window.state.projects_by_page.get(page_id_for_pair(pair)), self.window.state.restored_page_roots.get(page_id_for_pair(pair)))
        page_dir = ws.page_root if ws is not None else None
        if page_dir is None:
            self.direct_result_status.setVisible(True)
            self.direct_result_status.setText("当前页 Direct 结果：暂无工作区。请先处理当前页。")
            self.direct_result_status.setStyleSheet(self._tone_style("orange"))
            return
        summary = _safe_direct_patch_summary(page_dir)
        self.direct_result_status.setVisible(True)
        if not summary.get("payload"):
            self.direct_result_status.setText("当前页 Direct 结果：暂无 direct_patch 产物。请重新处理当前页；已有旧结果不会自动切换到新 Direct 合同。")
            self.direct_result_status.setStyleSheet(self._tone_style("orange"))
            return
        used = bool(summary.get("used", False))
        accepted = bool(summary.get("accepted", False))
        applied = int(summary.get("applied_count", 0) or 0)
        region_count = int(summary.get("region_count", 0) or 0)
        strategy = str(summary.get("strategy", "") or "direct_borderless_overlay")
        reason = str(summary.get("reason", "") or "")
        missing = list(summary.get("missing_files", []) or [])
        if used and accepted and not missing:
            self.direct_result_status.setText(f"当前页 Direct 结果：已真正执行主链路 · {strategy} · 应用 {applied} 个区域 / 记录 {region_count} 个区域 · 3 项核心产物完整。")
            self.direct_result_status.setStyleSheet(self._tone_style("green"))
        elif used and missing:
            self.direct_result_status.setText("当前页 Direct 结果：有执行记录，但产物不完整：" + "、".join(missing) + "。建议重新处理当前页。")
            self.direct_result_status.setStyleSheet(self._tone_style("red"))
        else:
            self.direct_result_status.setText("当前页 Direct 结果：未完成安全贴图。原因：" + self._direct_patch_reason_label(reason) + "。")
            self.direct_result_status.setStyleSheet(self._tone_style("orange"))

    def _update_experimental_status(self):
        current = str(self.mode.currentData() or "")
        active = current == "aligned_overlay_reveal"
        self.experimental_status.setVisible(active)
        if not active:
            self.mode.setToolTip("当前迁移方式")
            return
        self.experimental_status.setText("整页独立模式：下一次处理将执行 aligned_overlay_reveal，按 TARGET 气泡开孔显示整页配准后的中文下层；不会调用 Transparent Reveal。")
        self.experimental_status.setStyleSheet(self._tone_style("green"))
        self.experimental_status.setToolTip("绿色 = 真正运行独立整页透明/挖孔路线")
        self.mode.setToolTip("整页独立模式：aligned_overlay_reveal")

    @staticmethod
    def _aligned_overlay_reason_label(reason: str) -> str:
        reason = str(reason or "").strip()
        labels = {
            "ok": "已通过实验路线门槛",
            "no_accepted_regions": "没有找到可安全挖洞的区域",
            "page_erase_area_cap": "预计清除面积过大，触发整页安全上限",
            "rejected_page_pair_verification": "中日页面同页校验失败",
            "feature_flag_disabled": "实验功能开关未真正开启",
            "aligned_overlay_reveal_disabled": "实验功能开关未真正开启",
            "unsupported_erase_source": "实验清除来源配置不受支持",
        }
        if reason.startswith("rejected_registration:"):
            return "页面配准未通过实验路线严格门槛：" + reason.split(":", 1)[1]
        return labels.get(reason, reason or "未提供原因")

    def _update_experimental_result_status(self, index: int | None = None):
        if not hasattr(self, "experimental_result_status"):
            return
        if str(self.mode.currentData() or "") != "aligned_overlay_reveal":
            self.experimental_result_status.setVisible(False)
            return
        pairs = self.window.state.pairs
        idx = self.window.state.selected_index if index is None else int(index)
        if not (0 <= idx < len(pairs)):
            self.experimental_result_status.setVisible(True)
            self.experimental_result_status.setText("当前页实验结果：尚未选择页面。")
            self.experimental_result_status.setStyleSheet(f"color:{self._theme_colors()['muted']};")
            return
        pair = pairs[idx]
        ws = resolve_page_workspace(self.window.state.output_dir, pair, self.window.state.projects_by_page.get(page_id_for_pair(pair)), self.window.state.restored_page_roots.get(page_id_for_pair(pair)))
        page_dir = ws.page_root if ws is not None else None
        payload = {}
        if page_dir is not None:
            meta_path = page_dir / "aligned_overlay_reveal.json"
            if meta_path.exists():
                try:
                    payload = as_dict(load_json(meta_path))
                except Exception:
                    payload = {}
            if not payload and (page_dir / "project.json").exists():
                try:
                    project = normalize_project(load_json(page_dir / "project.json"))
                    payload = as_dict(as_dict(project.get("meta")).get("aligned_overlay_reveal"))
                except Exception:
                    payload = {}
        if not payload:
            self.experimental_result_status.setVisible(True)
            self.experimental_result_status.setText("当前页实验结果：暂无 aligned_overlay_reveal 产物。请确认上方为绿色状态后，重新处理当前页；已有旧结果不会自动重跑。")
            self.experimental_result_status.setStyleSheet(self._tone_style("orange"))
            return
        used = bool(payload.get("used", False))
        accepted = bool(payload.get("accepted", False))
        reason = str(payload.get("reason", "") or "")
        triage = str(payload.get("page_triage", "") or "")
        diag = as_dict(payload.get("diagnostics"))
        applied = int(diag.get("applied_region_count", 0) or 0)
        required_files = [
            "aligned_overlay_reveal_mask.png", "aligned_overlay_reveal_layer.png",
            "aligned_overlay_reveal_regions.png", "aligned_overlay_reveal.json",
        ]
        missing = [name for name in required_files if page_dir is None or not (page_dir / name).exists()]
        self.experimental_result_status.setVisible(True)
        if used and accepted and not missing:
            self.experimental_result_status.setText(f"当前页实验结果：已真正执行 · {triage or 'REVIEW'} · 应用 {applied} 个区域 · 4 项核心产物完整。")
            self.experimental_result_status.setStyleSheet(self._tone_style("green"))
        elif missing and (used or accepted):
            self.experimental_result_status.setText("当前页实验结果：路线有执行记录，但产物不完整：" + "、".join(missing) + "。建议重新处理当前页，避免使用旧缓存结果。")
            self.experimental_result_status.setStyleSheet(self._tone_style("red"))
        else:
            self.experimental_result_status.setText("当前页实验结果：未应用挖洞。原因：" + self._aligned_overlay_reason_label(reason) + "。TARGET 已保持原样；可按原因调整配准/页面配对或转人工补漏。")
            self.experimental_result_status.setStyleSheet(self._tone_style("orange"))

    def _mode_summary_text(self, mode: str) -> str:
        summaries = {
            "direct_patch": "直接贴图 · SOURCE 中文在上、TARGET 日文在下 · 只写受控内层，TARGET 边框保持权威",
            "mask_replace": "精准蒙版 · 只在气泡/文本框的安全蒙版内迁移 SOURCE 中文，背景和结构由 TARGET 保留",
            "hybrid": "精准蒙版+OCR · 有任何蒙版候选就不自动 OCR；仅无蒙版区域或人工框选才使用 OCR",
            "aligned_overlay_reveal": "整页对齐挖孔显中文 · 上层 TARGET 裁掉气泡/文本框，显示下层整页对齐 SOURCE",
            "reletter": "OCR重排 · OCR 识别后重新排字；字体、断句和字号集中在下方折叠设置",
            "transparent_bubble_reveal": "整页对齐透明显中文 · TARGET 为独立 RGBA 上层，透明区域显示整页对齐 SOURCE 下层；不调用挖孔 renderer",
        }
        return summaries.get(str(mode or ""), compact_mode_ui_label(str(mode or "direct_patch")))

    def _reorder_mode_sections(self, mode: str) -> None:
        if not hasattr(self, "mode_card"):
            return
        layout = self.mode_card.layout
        sections = [
            getattr(self, "direct_clarity_section", None),
            getattr(self, "direct_contract_section", None),
            getattr(self, "hybrid_ocr_contract_section", None),
            getattr(self, "reletter_section", None),
            getattr(self, "transparent_section", None),
            getattr(self, "semantic_section", None),
        ]
        for section in sections:
            if section is not None:
                layout.removeWidget(section)
        order = {
            "direct_patch": ["direct_clarity_section", "direct_contract_section", "semantic_section"],
            "mask_replace": ["direct_clarity_section", "semantic_section"],
            "hybrid": ["direct_clarity_section", "hybrid_ocr_contract_section", "reletter_section", "semantic_section"],
            "reletter": ["reletter_section", "semantic_section"],
            "aligned_overlay_reveal": ["semantic_section"],
            "transparent_bubble_reveal": ["transparent_section", "semantic_section"],
        }.get(mode, ["semantic_section"])
        insert_at = 3  # mode selector + summary + compact core switches
        for name in order:
            section = getattr(self, name, None)
            if section is not None:
                layout.insertWidget(insert_at, section)
                insert_at += 1

    def _update_mode_specific_controls(self):
        """Show a compact, mode-specific control surface without changing config semantics."""
        mode = str(self.mode.currentData() or "direct_patch") if hasattr(self, "mode") else "direct_patch"
        if hasattr(self, "mode_summary"):
            self.mode_summary.setText(self._mode_summary_text(mode))

        # High-frequency rules: hidden when they do not apply instead of leaving
        # disabled rows that consume vertical space.
        diff_visible = mode in {"mask_replace", "hybrid", "reletter"}
        exact_visible = mode in {"direct_patch", "mask_replace", "hybrid"}
        if hasattr(self, "diff_check"):
            self.diff_check.setVisible(diff_visible); self.diff_check.setEnabled(diff_visible)
        if hasattr(self, "exact_check"):
            self.exact_check.setVisible(exact_visible); self.exact_check.setEnabled(exact_visible)
        if hasattr(self, "mode_core_box"):
            self.mode_core_box.setVisible(diff_visible or exact_visible)

        reletter_active = mode in {"reletter", "hybrid"}
        if hasattr(self, "reletter_section"):
            self.reletter_section.setVisible(reletter_active)
        if hasattr(self, "direct_contract_section"):
            self.direct_contract_section.setVisible(mode == "direct_patch")
        clarity_active = mode in {"direct_patch", "mask_replace", "hybrid"}
        if hasattr(self, "direct_clarity_section"):
            self.direct_clarity_section.setVisible(clarity_active)
            self._update_direct_clarity_controls()
        if hasattr(self, "hybrid_ocr_contract_section"):
            self.hybrid_ocr_contract_section.setVisible(mode == "hybrid")
        if hasattr(self, "hybrid_ocr_contract_box"):
            self.hybrid_ocr_contract_box.setVisible(mode == "hybrid")
        transparent_active = mode == "transparent_bubble_reveal"
        if hasattr(self, "transparent_section"):
            self.transparent_section.setVisible(transparent_active)
        for widget in getattr(self, "transparent_widgets", []):
            widget.setEnabled(transparent_active)
        # Semantic analysis remains accessible as a collapsed advanced section;
        # it does not occupy the screen unless explicitly opened.
        if hasattr(self, "semantic_section"):
            self.semantic_section.setVisible(True)

        self._reorder_mode_sections(mode)
        if hasattr(self.window, "models") and hasattr(self.window.models, "apply_transfer_mode_ocr_lock"):
            self.window.models.apply_transfer_mode_ocr_lock(mode)
        if hasattr(self, "mode_card"):
            self.mode_card.updateGeometry()
        if hasattr(self, "actions_card"):
            self.actions_card.updateGeometry()
        if hasattr(self, "detail_card"):
            self.detail_card.updateGeometry()

    def _update_direct_clarity_controls(self):
        active = str(self.mode.currentData() or "direct_patch") in {"direct_patch", "mask_replace", "hybrid"}
        enabled = bool(self.direct_white_clarity_enabled.isChecked()) if hasattr(self, "direct_white_clarity_enabled") else False
        for widget in (
            getattr(self, "direct_white_clarity_alpha_gamma", None),
            getattr(self, "direct_white_clarity_black_boost", None),
            getattr(self, "direct_white_clarity_pure_white_floor", None),
            getattr(self, "direct_white_clarity_min_text_pixels", None),
        ):
            if widget is not None:
                widget.setEnabled(active and enabled)
        if hasattr(self, "direct_clarity_summary"):
            if not active:
                self.direct_clarity_summary.setText("当前仅在 Direct / 精准蒙版 / 精准蒙版+OCR 模式下可用。")
            elif enabled:
                detail = (
                    f"字边 {self.direct_white_clarity_alpha_gamma.value():.2f} · "
                    f"黑字 {int(self.direct_white_clarity_black_boost.value())} · "
                    f"白底 {int(self.direct_white_clarity_pure_white_floor.value())}"
                )
                self.direct_clarity_summary.setText("已启用：" + detail)
                if hasattr(self, "direct_clarity_section"):
                    self.direct_clarity_section.setSummary("已启用 · " + detail)
            else:
                self.direct_clarity_summary.setText("已关闭：白气泡将沿用当前模式原始像素贴图，不做漂白清底与字边增强。")
                if hasattr(self, "direct_clarity_section"):
                    self.direct_clarity_section.setSummary("已关闭 · 沿用当前模式原始 SOURCE 像素")

    def _refresh_reletter_font_catalog(self):
        if not hasattr(self,"reletter_font_catalog"):
            return
        current=self.reletter_font.text().strip()
        self.reletter_font_catalog.blockSignals(True)
        self.reletter_font_catalog.clear(); self.reletter_font_catalog.addItem("自动扫描字体库", "")
        for frow in discover_fonts(limit=160):
            self.reletter_font_catalog.addItem(str(frow.get("name") or Path(frow.get("path","")).stem), str(frow.get("path") or ""))
        self.reletter_font_catalog.blockSignals(False)
        if current:
            idx=self.reletter_font_catalog.findData(current)
            if idx>=0: self.reletter_font_catalog.setCurrentIndex(idx)

    def _apply_reletter_catalog_font(self):
        if not hasattr(self,"reletter_font_catalog"):
            return
        path=str(self.reletter_font_catalog.currentData() or "")
        if path:
            self.reletter_font.setText(path)
            if hasattr(self,"reletter_font_preset"):
                idx=self.reletter_font_preset.findData("custom")
                if idx>=0: self.reletter_font_preset.setCurrentIndex(idx)

    def _apply_reletter_font_preset(self):
        if not hasattr(self, "reletter_font_preset"):
            return
        value = str(self.reletter_font_preset.currentData() or "custom")
        if value != "custom":
            self.reletter_font.setText(value)

    def _mode_candidate_config(self, mode: str):
        cfg = self.window.state.config
        if mode == "direct_patch": return cfg.direct_patch
        if mode == "mask_replace": return cfg.mask_replace
        if mode == "hybrid": return cfg.hybrid.mask
        if mode == "reletter": return cfg.reletter.candidates
        return None

    def _mode_lettering_config(self, mode: str):
        cfg = self.window.state.config
        if mode == "hybrid": return cfg.hybrid.lettering
        if mode == "reletter": return cfg.reletter.lettering
        return None

    def _save_mode_specific_ui(self, mode: str):
        candidate = self._mode_candidate_config(mode)
        if candidate is not None:
            if hasattr(candidate, "paired_diff_enabled") and mode in {"direct_patch", "mask_replace", "hybrid", "reletter"}:
                candidate.paired_diff_enabled = bool(self.diff_check.isChecked())
            if hasattr(candidate, "exact_identity_copy") and mode in {"direct_patch", "mask_replace", "hybrid"}:
                candidate.exact_identity_copy = bool(self.exact_check.isChecked())
            if mode in {"direct_patch", "mask_replace", "hybrid"}:
                if hasattr(candidate, "direct_white_clarity_enhance_enabled"):
                    candidate.direct_white_clarity_enhance_enabled = bool(self.direct_white_clarity_enabled.isChecked())
                if hasattr(candidate, "direct_white_clarity_alpha_gamma"):
                    candidate.direct_white_clarity_alpha_gamma = float(self.direct_white_clarity_alpha_gamma.value())
                if hasattr(candidate, "direct_white_clarity_black_boost"):
                    candidate.direct_white_clarity_black_boost = int(self.direct_white_clarity_black_boost.value())
                if hasattr(candidate, "direct_white_clarity_pure_white_floor"):
                    candidate.direct_white_clarity_pure_white_floor = int(self.direct_white_clarity_pure_white_floor.value())
                if hasattr(candidate, "direct_white_clarity_min_text_pixels"):
                    candidate.direct_white_clarity_min_text_pixels = int(self.direct_white_clarity_min_text_pixels.value())
        lettering = self._mode_lettering_config(mode)
        if lettering is not None:
            lettering.font_path = (self.reletter_font.text().strip() or None)
            lettering.line_break_mode = str(self.reletter_break_mode.currentData() or "smart")
            lettering.layout_mode = str(self.reletter_layout_mode.currentData() or "smart_scaling")
            min_size = int(self.reletter_min_font.value()); max_size = max(min_size, int(self.reletter_max_font.value()))
            lettering.min_font_size = min_size; lettering.max_font_size = max_size
            lettering.line_spacing_ratio = float(self.reletter_line_spacing.value())
            lettering.koharu_flow_cells_enabled = bool(self.reletter_koharu_flow_cells.isChecked())

    def _load_mode_specific_ui(self, mode: str):
        candidate = self._mode_candidate_config(mode)
        lettering = self._mode_lettering_config(mode)
        widgets = [self.diff_check, self.exact_check, self.direct_white_clarity_enabled, self.direct_white_clarity_alpha_gamma, self.direct_white_clarity_black_boost, self.direct_white_clarity_pure_white_floor, self.direct_white_clarity_min_text_pixels, self.reletter_font, self.reletter_break_mode, self.reletter_layout_mode, self.reletter_min_font, self.reletter_max_font, self.reletter_line_spacing, self.reletter_koharu_flow_cells]
        for widget in widgets: widget.blockSignals(True)
        try:
            if candidate is not None:
                self.diff_check.setChecked(bool(getattr(candidate, "paired_diff_enabled", True)))
                self.exact_check.setChecked(bool(getattr(candidate, "exact_identity_copy", True)))
                if mode in {"direct_patch", "mask_replace", "hybrid"}:
                    _clarity_default = True if mode == "direct_patch" else False
                    self.direct_white_clarity_enabled.setChecked(bool(getattr(candidate, "direct_white_clarity_enhance_enabled", _clarity_default)))
                    self.direct_white_clarity_alpha_gamma.setValue(float(getattr(candidate, "direct_white_clarity_alpha_gamma", 1.0)))
                    self.direct_white_clarity_black_boost.setValue(int(getattr(candidate, "direct_white_clarity_black_boost", 0)))
                    self.direct_white_clarity_pure_white_floor.setValue(int(getattr(candidate, "direct_white_clarity_pure_white_floor", 248)))
                    self.direct_white_clarity_min_text_pixels.setValue(int(getattr(candidate, "direct_white_clarity_min_text_pixels", 18)))
            if lettering is not None:
                current_font = str(lettering.font_path or "")
                self.reletter_font.setText(current_font)
                bidx = self.reletter_break_mode.findData(str(getattr(lettering, "line_break_mode", "smart")))
                if bidx >= 0: self.reletter_break_mode.setCurrentIndex(bidx)
                lidx = self.reletter_layout_mode.findData(str(getattr(lettering, "layout_mode", "smart_scaling")))
                if lidx >= 0: self.reletter_layout_mode.setCurrentIndex(lidx)
                self.reletter_min_font.setValue(int(lettering.min_font_size)); self.reletter_max_font.setValue(int(lettering.max_font_size))
                self.reletter_line_spacing.setValue(float(lettering.line_spacing_ratio))
                self.reletter_koharu_flow_cells.setChecked(bool(getattr(lettering, "koharu_flow_cells_enabled", False)))
        finally:
            for widget in widgets: widget.blockSignals(False)
        self._update_direct_clarity_controls()

    def _ensure_legacy_mode_item(self, stored_mode: str):
        if not is_legacy_mode(stored_mode): return
        if self.mode.findData(stored_mode) >= 0: return
        spec = get_mode_spec(stored_mode)
        self.mode.addItem(compact_mode_ui_label(spec.label), stored_mode)

    def _on_mode_changed(self):
        new_mode = str(self.mode.currentData() or "direct_patch")
        previous = str(getattr(self, "_last_mode_key", "") or "")
        if previous and previous != new_mode:
            self._save_mode_specific_ui(previous)
        self.window.state.config.transfer.mode = new_mode
        self._load_mode_specific_ui(new_mode)
        self._last_mode_key = new_mode
        self._sync_config()
        self._update_mode_specific_controls()
        self._update_direct_status()
        self._update_direct_result_status()
        if hasattr(self.window, "workbench") and hasattr(self.window.workbench, "refresh_mode_controls"):
            self.window.workbench.refresh_mode_controls()

    def _sync_config(self):
        cfg = self.window.state.config
        mode = str(self.mode.currentData() or "direct_patch")
        cfg.transfer.mode = mode
        self._save_mode_specific_ui(mode)
        # Merely exposing the experimental selector never opts Auto into this route.
        # Auto still needs allow_in_auto=true and require_explicit_mode=false.
        cfg.aligned_overlay_reveal.enabled = (str(cfg.transfer.mode or "") == "aligned_overlay_reveal")
        tcfg = cfg.transparent_bubble_reveal
        # Explicitly selecting the v2 route is the opt-in; it never enters Auto.
        tcfg.enabled = (str(cfg.transfer.mode or "") == "transparent_bubble_reveal")
        tcfg.bubble_backend = str(self.transparent_backend.currentData() or "auto")
        tcfg.clear_mode = str(self.transparent_clear_mode.currentData() or "hybrid")
        tcfg.protect_border = self.transparent_protect_border.isChecked()
        tcfg.suppress_page_furniture = bool(self.transparent_suppress_page_furniture.isChecked())
        tcfg.verify_target_text_presence = bool(self.transparent_verify_text_presence.isChecked())
        tcfg.target_text_presence_ocr_enabled = bool(self.transparent_ocr_text_presence.isChecked())
        tcfg.require_source_translation_evidence = bool(self.transparent_restore_source_evidence.isChecked())
        tcfg.expand_px = int(self.transparent_expand.value())
        tcfg.feather_px = int(self.transparent_feather.value())
        scfg = cfg.semantic
        scfg.enabled = bool(self.semantic_enabled.isChecked())
        scfg.backend = str(self.semantic_backend.currentData() or "auto")
        scfg.strategy = str(self.semantic_strategy.currentData() or "auto")
        scfg.apply_to_reveal = bool(self.semantic_apply_reveal.isChecked())
        scfg.save_overlay = bool(self.semantic_save_overlay.isChecked())
        self._update_experimental_status()
        cfg.pairing.prefer_name_pairing = self.prefer_name_pair.isChecked(); cfg.pairing.prefer_order_pairing = self.prefer_order_pair.isChecked(); cfg.pairing.remake_pair_verifier_enabled = self.remake_pair_verify.isChecked()
        cfg.cache.enabled = self.cache_check.isChecked()

    def _pick_reletter_font(self):
        start = self.reletter_font.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 OCR重排字体", start,
            "字体文件 (*.ttf *.ttc *.otf *.otc);;所有文件 (*)",
        )
        if path:
            self.reletter_font.setText(path)

    def _clear_reletter_font(self):
        self.reletter_font.clear()
        self._sync_config()

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
        menu.setObjectName("pageTypeMenu")
        # Keep every page-type row on one explicit geometry contract. macOS native
        # QMenu can otherwise paint the check column and selected background with
        # slightly different vertical metrics, making the blue row look shifted
        # relative to its text.
        menu.setStyleSheet("""
            QMenu#pageTypeMenu::item {
                min-height: 22px;
                padding: 4px 26px 4px 30px;
                margin: 0px;
            }
            QMenu#pageTypeMenu::indicator {
                width: 14px; height: 14px;
            }
            QMenu#pageTypeMenu::separator {
                height: 1px; margin: 4px 8px;
            }
        """)
        current_types = []
        for row in rows:
            if 0 <= row < len(self.window.state.pairs):
                current_types.append(self.window.page_mark_for_pair(self.window.state.pairs[row]).page_type)
        unique_types = set(current_types)
        current_type = next(iter(unique_types)) if len(unique_types) == 1 else None
        current_label = page_type_label(current_type) if current_type else "多种类型"
        title = menu.addAction(f"已选 {len(rows)} 页 · 当前：{current_label}")
        title.setEnabled(False)
        menu.addSeparator()
        type_group = QActionGroup(menu)
        type_group.setExclusive(True)
        for key in MANUAL_PAGE_TYPES:
            action = menu.addAction(page_type_label(key))
            action.setCheckable(True)
            action.setChecked(key == current_type)
            type_group.addAction(action)
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
        self._update_direct_result_status(idx)
        self._update_experimental_result_status(idx)

    def _sync_detail_from_current(self):
        item = self.thumb_list.currentItem()
        if item is not None:
            try: self._sync_detail(int(item.data(Qt.ItemDataRole.UserRole))); return
            except Exception: pass
        self._sync_detail(self.window.state.selected_index if self.window.state.pairs else -1)

    def _sync_detail(self, index: int):
        pairs = self.window.state.pairs
        if not (0 <= index < len(pairs)):
            # Expansion state is user-owned. Page changes, empty selection and
            # preview loading must never open or close the inspector implicitly.
            self.detail_page.setText("未选择页面"); self.detail_badge.setText("—"); self.detail_badge.setStyleSheet("")
            self.detail_view.set_image(None); self.detail_names.setText(""); self.detail_reason.setText(""); self.detail_stats.setText(""); return
        pair = pairs[index]; mark = self.window.page_mark_for_pair(pair)
        self.detail_page.setText(f"第 {index+1} 页")
        self.detail_badge.setText(mark.label)
        color = page_type_color(mark.page_type)
        self.detail_badge.setStyleSheet(f"background:{color};color:white;border-radius:9px;padding:4px 8px;font-weight:700;")
        path = pair.source_path if self._detail_side == "source" else pair.target_path
        preview_path = Path(path)
        # Load/sync the preview even while the inspector is collapsed, but never
        # change its expansion state. Only the user's 展开/收起 button controls it.
        self.detail_view.set_image(preview_path)
        names_text = f"旧中文：{Path(pair.source_path).name}\n高清日文：{Path(pair.target_path).name}"
        self.detail_names.setText(names_text); self.detail_names.setToolTip(names_text)
        origin = "手动" if mark.origin == "manual" else "默认"
        description = PAGE_TYPE_INFO.get(mark.page_type, PAGE_TYPE_INFO["content"]).get("description", "")
        reason_text = f"{origin} · {description}\n{mark.reason or '尚未进行自动页面检查'}"
        self.detail_reason.setText(reason_text); self.detail_reason.setToolTip(reason_text)
        method = {"name":"名称", "order":"顺序", "smart":"智能"}.get(pairing_method(pair), pairing_method(pair))
        stats_text = f"配对：{method} · {pair.confidence:.3f}　页面：{'手动分类' if mark.origin == 'manual' else '默认正文'}"
        self.detail_stats.setText(stats_text); self.detail_stats.setToolTip(stats_text)
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
        key = (str(path), int(st.st_mtime_ns), int(st.st_size), self.THUMB_SIZE.width(), self.THUMB_SIZE.height())
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
        canvas = QPixmap(self.THUMB_CANVAS); colors = self._theme_colors(); canvas.fill(QColor(colors["thumb_bg"]))
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
            painter.setPen(QColor(colors["border"])); painter.drawRect(0,0,canvas.width()-1,canvas.height()-1)
        finally:
            painter.end()
        return QIcon(canvas)

    def _placeholder_icon(self, index: int) -> QIcon:
        # A single generic placeholder is enough: the QListWidget item text already
        # carries the page number. Reusing one icon avoids allocating/painting
        # hundreds of QPixmaps when an existing large project is restored.
        if self._thumb_placeholder_cache is not None:
            return self._thumb_placeholder_cache
        canvas = QPixmap(self.THUMB_CANVAS); colors = self._theme_colors(); canvas.fill(QColor(colors["thumb_bg"])); painter = QPainter(canvas)
        try:
            painter.setPen(QColor(colors["muted_2"])); f=painter.font(); f.setPointSize(11); painter.setFont(f)
            painter.drawText(canvas.rect(), Qt.AlignmentFlag.AlignCenter, "加载中…")
        finally:
            painter.end()
        self._thumb_placeholder_cache = QIcon(canvas)
        return self._thumb_placeholder_cache

    def _rebuild_thumbnails(self):
        selected = set(self._selected_thumb_rows()); current = self.window.state.selected_index
        self._thumb_generation += 1
        self._thumb_load_timer.stop(); self._thumb_queue = []; self._thumb_retry_count = 0; self._thumb_loaded.clear(); self._thumb_items.clear()
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
        # Any scroll/filter/selection change invalidates the old visible queue.
        # Rebuild it only after the event stream settles; do not rescan every
        # 12–18 ms while individual thumbnails are decoded.
        self._thumb_queue = []
        self._thumb_retry_count = 0
        self._thumb_load_timer.start(max(0, int(delay_ms)))

    def _pump_thumbnails(self):
        if self.view_stack.currentIndex() != 0 or not self._thumb_items:
            return
        if not self._thumb_queue:
            self._thumb_queue = self._visible_thumbnail_indices()
            if not self._thumb_queue:
                # Immediately after restoring a session QListView may not yet
                # have valid visualItemRect geometry. Previously the 0 ms timer
                # returned here and never ran again, leaving the left gallery on
                # “加载中…” until a later scroll. Give Qt a few layout frames and
                # retry automatically.
                remaining = any(
                    (not item.isHidden()) and idx not in self._thumb_loaded
                    for idx, item in self._thumb_items.items()
                )
                if remaining and self._thumb_retry_count < 8:
                    self._thumb_retry_count += 1
                    self._thumb_load_timer.start(16 if self._thumb_retry_count <= 3 else 48)
                return
            self._thumb_retry_count = 0

        # Decode one scaled image per event-loop slice. This is intentionally
        # conservative for 4K/8K manga pages: first previews appear promptly but
        # trackpad scrolling and tab changes remain responsive.
        idx = self._thumb_queue.pop(0)
        item = self._thumb_items.get(idx)
        if item is not None and idx < len(self.window.state.pairs):
            try:
                item.setIcon(self._thumbnail_icon(idx))
            except Exception:
                item.setIcon(self._placeholder_icon(idx))
            self._thumb_loaded.add(idx)

        if self._thumb_queue:
            self._thumb_load_timer.start(10)
        else:
            # The just-decoded icon can change QListView geometry on some Qt/macOS
            # versions. Re-evaluate once, cheaply, to pick up another newly visible
            # card without entering a permanent polling loop.
            self._thumb_load_timer.start(24)

    def _force_thumbnail_refresh(self):
        self._thumb_image_cache.clear(); self._thumb_placeholder_cache = None; self._thumb_loaded.clear(); self._thumb_queue = []; self._thumb_retry_count = 0
        self._thumb_signature = None; self._table_signature = None; self._last_refresh_key = None; self.refresh()

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
        configured = sum(bool(str(value or "").strip()) for value in (s.source_dir, s.target_dir, s.output_dir))
        if s.pairs:
            self.project_summary.setText(f"已配对 {len(s.pairs)} 页 · {configured}/3 路径")
        elif configured:
            self.project_summary.setText(f"已配置 {configured}/3 路径")
        else:
            self.project_summary.setText("未选择输入")
        mark_sig = tuple(sorted((str(k), str(v.get("page_type","")), str(v.get("origin","")), str(v.get("reason","")), int(v.get("bubble_regions",0) or 0), int(v.get("free_text_regions",0) or 0)) for k,v in s.page_marks.items()))
        pair_sig = tuple((str(p.source_path), str(p.target_path), round(float(p.confidence), 6), str(pairing_method(p))) for p in s.pairs)
        table_signature = (pair_sig, mark_sig, tuple(sorted((str(k), str(v[0]), str(v[1])) for k, v in s.batch_status.items())))
        control_signature = (
            str(s.config.transfer.mode),
            bool(getattr(self._mode_candidate_config(str(s.config.transfer.mode)), "paired_diff_enabled", False)),
            bool(getattr(self._mode_candidate_config(str(s.config.transfer.mode)), "exact_identity_copy", False)),
            str(getattr(self._mode_lettering_config(str(s.config.transfer.mode)), "font_path", "") or ""),
            str(getattr(self._mode_lettering_config(str(s.config.transfer.mode)), "line_break_mode", "")),
            str(getattr(self._mode_lettering_config(str(s.config.transfer.mode)), "layout_mode", "")),
            bool(s.config.pairing.prefer_name_pairing), bool(s.config.pairing.prefer_order_pairing),
            bool(s.config.pairing.remake_pair_verifier_enabled), bool(s.config.cache.enabled),
        )
        refresh_key = (table_signature, self._thumb_side, self._detail_side, int(s.selected_index), str(self.filter_combo.currentData() or "all"), int(self.view_stack.currentIndex()), control_signature, s.source_dir, s.target_dir, s.output_dir)
        expected_thumb_signature = (pair_sig, mark_sig, self._thumb_side)
        if (
            refresh_key == self._last_refresh_key
            and self._table_signature == table_signature
            and self._thumb_signature == expected_thumb_signature
        ):
            # A pure workflow-tab switch should be O(1): no table walk, no filter
            # pass and—most importantly—no image decode. Explicit invalidation of
            # table/thumb signatures still forces a rebuild.
            self._schedule_visible_thumbnails(0)
            return
        self._last_refresh_key = refresh_key
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
                        if c == 1 and not mark.should_process: it.setForeground(QColor(self._theme_colors()["muted"]))
                        self.table.setItem(i, c, it)
                for row in selected:
                    if 0 <= row < self.table.rowCount(): self.table.selectRow(row)
            finally: self.table.setUpdatesEnabled(True)
            self._table_signature = table_signature
        thumb_signature = expected_thumb_signature
        if thumb_signature != self._thumb_signature:
            self._thumb_signature = thumb_signature; self._rebuild_thumbnails()
        self.set_current_page(s.selected_index)
        self._apply_filter()
        process_count = sum(1 for pair in s.pairs if self.window.page_mark_for_pair(pair).should_process); skip_count = len(s.pairs)-process_count
        manual_count = sum(1 for pair in s.pairs if self.window.page_mark_for_pair(pair).origin == "manual"); unmatched_count = len(s.unmatched_source)+len(s.unmatched_target)
        self.sum_pairs.setText(f"{len(s.pairs)} 页")
        self.sum_hint.setText((f"处理 {process_count} · 跳过 {skip_count} · 手动标记 {manual_count} · 未匹配 {unmatched_count}" if s.pairs else "等待页面配对"))
        exp_visible = False
        if self.show_experimental.isChecked() != exp_visible:
            self.show_experimental.blockSignals(True); self.show_experimental.setChecked(exp_visible); self.show_experimental.blockSignals(False)
            self._set_experimental_visible(exp_visible)
        # Restore v2 transparent-reveal controls without firing _sync_config half
        # way through the restore (which would overwrite saved backend/mode values
        # with whichever widgets happened to be initialized first).
        tcfg = s.config.transparent_bubble_reveal
        restore_widgets = [self.transparent_backend, self.transparent_clear_mode, self.transparent_protect_border, self.transparent_suppress_page_furniture, self.transparent_verify_text_presence, self.transparent_ocr_text_presence, self.transparent_restore_source_evidence, self.transparent_expand, self.transparent_feather, self.semantic_enabled, self.semantic_backend, self.semantic_strategy, self.semantic_apply_reveal, self.semantic_save_overlay, self.mode, self.reletter_font, self.reletter_font_catalog, self.reletter_font_preset, self.reletter_break_mode, self.reletter_layout_mode, self.reletter_min_font, self.reletter_max_font, self.reletter_line_spacing, self.reletter_koharu_flow_cells]
        for widget in restore_widgets: widget.blockSignals(True)
        try:
            t_idx = self.transparent_backend.findData(str(tcfg.bubble_backend))
            if t_idx >= 0: self.transparent_backend.setCurrentIndex(t_idx)
            c_idx = self.transparent_clear_mode.findData(str(tcfg.clear_mode))
            if c_idx >= 0: self.transparent_clear_mode.setCurrentIndex(c_idx)
            self.transparent_protect_border.setChecked(bool(tcfg.protect_border))
            self.transparent_suppress_page_furniture.setChecked(bool(getattr(tcfg, "suppress_page_furniture", True)))
            self.transparent_verify_text_presence.setChecked(bool(getattr(tcfg, "verify_target_text_presence", True)))
            self.transparent_ocr_text_presence.setChecked(bool(getattr(tcfg, "target_text_presence_ocr_enabled", False)))
            self.transparent_restore_source_evidence.setChecked(bool(getattr(tcfg, "require_source_translation_evidence", False)))
            self.transparent_expand.setValue(int(tcfg.expand_px)); self.transparent_feather.setValue(int(tcfg.feather_px))
            scfg = s.config.semantic
            self.semantic_enabled.setChecked(bool(getattr(scfg, "enabled", False)))
            si = self.semantic_backend.findData(str(getattr(scfg, "backend", "auto"))); self.semantic_backend.setCurrentIndex(si if si >= 0 else 0)
            si = self.semantic_strategy.findData(str(getattr(scfg, "strategy", "auto"))); self.semantic_strategy.setCurrentIndex(si if si >= 0 else 0)
            self.semantic_apply_reveal.setChecked(bool(getattr(scfg, "apply_to_reveal", True)))
            self.semantic_save_overlay.setChecked(bool(getattr(scfg, "save_overlay", False)))
            stored_mode = str(s.config.transfer.mode or "direct_patch")
            self._ensure_legacy_mode_item(stored_mode)
            idx = self.mode.findData(stored_mode)
            if idx >= 0: self.mode.setCurrentIndex(idx)
            mode_lettering = self._mode_lettering_config(stored_mode)
            current_font = str(getattr(mode_lettering, "font_path", "") or "")
            self.reletter_font.setText(current_font)
            preset_value = current_font if current_font in {"sans", "serif", "rounded", "comic"} else "custom"
            pidx = self.reletter_font_preset.findData(preset_value)
            if pidx >= 0: self.reletter_font_preset.setCurrentIndex(pidx)
            bidx = self.reletter_break_mode.findData(str(getattr(mode_lettering, "line_break_mode", "smart")))
            if bidx >= 0: self.reletter_break_mode.setCurrentIndex(bidx)
            lidx = self.reletter_layout_mode.findData(str(getattr(mode_lettering, "layout_mode", "smart_scaling")))
            if lidx >= 0: self.reletter_layout_mode.setCurrentIndex(lidx)
            
            if mode_lettering is not None:
                self.reletter_min_font.setValue(int(mode_lettering.min_font_size)); self.reletter_max_font.setValue(int(mode_lettering.max_font_size)); self.reletter_line_spacing.setValue(float(mode_lettering.line_spacing_ratio)); self.reletter_koharu_flow_cells.setChecked(bool(getattr(mode_lettering, "koharu_flow_cells_enabled", False)))
        finally:
            for widget in restore_widgets: widget.blockSignals(False)
        self._last_mode_key = str(s.config.transfer.mode or "direct_patch")
        self._load_mode_specific_ui(self._last_mode_key)
        self._update_mode_specific_controls()
        self._update_direct_status()
        self._update_direct_result_status()
        self.prefer_name_pair.setChecked(s.config.pairing.prefer_name_pairing); self.prefer_order_pair.setChecked(s.config.pairing.prefer_order_pairing); self.remake_pair_verify.setChecked(bool(s.config.pairing.remake_pair_verifier_enabled))
        self.resume_check.setChecked(s.config.batch.resume); self.cache_check.setChecked(s.config.cache.enabled)
        if self._preview_dialog is not None and self._preview_dialog.isVisible():
            self._preview_dialog.refresh_current()

__all__ = ["PagePreviewDialog", "ProjectPage"]
