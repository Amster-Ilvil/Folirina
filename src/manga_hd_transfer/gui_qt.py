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

from .gui_toast import notify
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
from .gui_shortcuts import install_shortcuts
from .gui_dialog_polish import install_dialog_polish
from .gui_window_state import restore_window_geometry, save_window_geometry
from .gui_runtime_feedback import RuntimePreparationCache, ProcessingStageClock, format_processing_heartbeat
from .app_logging import configure_application_logging, install_exception_hooks, runtime_log_dir
from .platform_support import desktop_platform_badge
from .gui_components import (
    _configure_responsive_dialog, _fit_scene_rect, StableThumbnailList, ImageView, Card, PageHero, OptionRow,
    PathRow, ZoomPreviewView, StableComboBox, StableSpinBox, StableDoubleSpinBox, StableSlider, WorkbenchPageRail,
)
from .studio_state import StudioState
from .studio_project_controller import StudioProjectControllerMixin
from .studio_workspace_page import WorkspacePage
from .studio_navigation import ROUTES, route_index
from .studio_lazy_pages import LazyStudioPageRegistry
from .project_workspace import ProjectWorkspaceManager
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
from .font_catalog import discover_fonts, import_font_to_library, persist_font_expression, font_library_dir
from .review_apply import apply_review_page, apply_manual_ocr_review, clear_ocr_review_blocks, generate_remove_text_preview, apply_target_layer_erase_review, reset_target_layer_erase_review, apply_target_layer_restore_review, reset_target_layer_restore_review, apply_manual_force_transfer_review, reset_manual_force_transfer_review, manual_force_auto_evidence_masks
from .manual_effect import map_target_bbox_to_source, registration_homography, build_manual_effect_masks, build_reveal_seed_mask, estimate_source_background, composite_source_text_delta, clean_manual_target_text
from .modes.reletter import ocr_edit_blocks as _reletter_ocr_edit_blocks
from .modes.reletter import ocr_edit_render as _reletter_ocr_edit_render
from .modes.hybrid import ocr_edit_blocks as _hybrid_ocr_edit_blocks
from .modes.hybrid import ocr_edit_render as _hybrid_ocr_edit_render
from . import ocr_edit_blocks as _shared_ocr_edit_blocks
from . import ocr_edit_render as _shared_ocr_edit_render


from .workspace import page_id_for_pair, resolve_page_workspace
from .workspace_guard import PageRunGuard
from .workspace_cleanup import cleanup_output_workspace
from .mode_contracts import clear_stale_mode_outputs
from .session_restore import scan_existing_results
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








# Heavy editor/review UI is isolated from the Studio composition root.
# Re-export the historical names here so extensions/tests importing from
# ``gui_qt`` remain compatible while implementation ownership lives in the
# dedicated workbench module.
from .studio_workbench_page import (
    RunLogDialog, MaskPaintView, MaskEditorDialog, ManualTextMaskPreviewDialog,
    RevealMaskDialog, RegionSelectView, OCRLayoutPreviewView, OCRBlockEditorDialog,
    RegionCompositeDialog, ManualEffectDialog, WorkbenchPage,
    is_ocr_edit_mode, ocr_edit_scope, load_ocr_blocks, upsert_ocr_block,
    delete_ocr_block, save_ocr_blocks, recognize_manual_ocr_block,
    apply_ocr_edit_blocks, preview_ocr_edit_blocks, reset_ocr_edit_blocks,
)


class StudioWindow(StudioProjectControllerMixin, QMainWindow):
    NAV_SECTIONS = [(route.label, route.english) for route in ROUTES]

    @staticmethod
    def _vertical_nav_label(label: str) -> str:
        """Stable text-only navigation labels without decorative numbering."""
        return str(label)

    def __init__(self):
        super().__init__()
        self.state = StudioState()
        self.state.config.transfer.mode = "direct_patch"
        stored_workspace_root = str(self._workspace_settings().value("workspace/root", "") or "").strip()
        self.project_workspace_manager = ProjectWorkspaceManager(stored_workspace_root or None)
        self.worker: PipelineWorker | None = None
        self._pending_pipeline_worker: PipelineWorker | None = None
        self._prepare_worker: AutoPrepareModelsWorker | None = None
        self._page_action_worker: PageActionWorker | None = None
        self._worker_is_single_page = False
        self._worker_page_id = ""
        self._ready_model_preparation = RuntimePreparationCache()
        self._pending_prepare_signature = ""
        self._processing_stage_clock = ProcessingStageClock()
        self._processing_stage_label = ""
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
        restore_window_geometry(self, self._workspace_settings())   # 记住上次的窗口大小与位置
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

        # Keep the fast project surfaces eager; construct heavier configuration,
        # workbench and export/settings pages only on first navigation.  This
        # mirrors Novel Formatter Phase 18's lazy-workspace boundary while
        # preserving one authoritative StudioState in this window.
        self.workspace = WorkspacePage(self)
        self.project = ProjectPage(self)
        self.stack.addWidget(self.workspace)
        self.stack.addWidget(self.project)
        self._lazy_pages = LazyStudioPageRegistry(self.stack)
        self._lazy_pages.register("models", self.page_index("models"), lambda: self._construct_lazy_page("models", ModelPage))
        self._lazy_pages.register("workbench", self.page_index("workbench"), lambda: self._construct_lazy_page("workbench", WorkbenchPage))
        self._lazy_pages.register("export", self.page_index("export"), lambda: self._construct_lazy_page("export", ExportPage))
        self._lazy_pages.register("settings", self.page_index("settings"), lambda: self._construct_lazy_page("settings", SettingsPage))
        # The global flow breadcrumb/next-step strip was removed intentionally.
        # Route switching stays in the left sidebar and each page owns its content.
        content_layout.addWidget(self.stack, 1)
        self.pages[0].setChecked(True)

        self.progress = QProgressBar(); self.progress.setRange(0,100); self.progress.setValue(0); self.progress.setMaximumWidth(190)
        self.statusBar().addPermanentWidget(self.progress); self.statusBar().showMessage("就绪")
        # Heartbeat updates the elapsed time without claiming the numerical page
        # progress advanced. Useful for long Koharu/registration startup on CPU.
        self._processing_heartbeat_timer = QTimer(self)
        self._processing_heartbeat_timer.setInterval(1000)
        self._processing_heartbeat_timer.timeout.connect(self._processing_heartbeat)
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
        self.workspace_restore_startup()
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
            notify(self, "请先选择一个页面。", "warning")
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
        save_window_geometry(self, self._workspace_settings())
        workbench=self.loaded_page("workbench")
        editor=getattr(workbench,"_region_editor",None)
        if editor is not None:
            editor.accept()
            if getattr(workbench,"_region_editor",None) is not None:
                event.ignore(); return
        self._save_workspace_session()
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
        visible = bool(state.cancellable and self.stack.currentIndex() != self.page_index("project"))
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

    def page_index(self, key: str) -> int:
        return route_index(key)

    def _construct_lazy_page(self, key: str, page_type):
        page = page_type(self)
        setattr(self, str(key), page)
        busy = self._processing_busy_state().busy
        if hasattr(page, "set_processing_busy"):
            page.set_processing_busy(busy)
        if key == "settings":
            if hasattr(page, "sync_theme"):
                page.sync_theme()
            scaler = getattr(self, "_responsive_scaler", None)
            factor = float(getattr(scaler, "scale_factor", 1.0) or 1.0) if scaler is not None else 1.0
            if hasattr(page, "set_responsive_scale"):
                page.set_responsive_scale(factor)
        return page

    def _ensure_page_loaded(self, index: int):
        route = ROUTES[int(index)]
        if route.key in {"workspace", "project"}:
            return self.stack.widget(int(index))
        return self._lazy_pages.get(route.key)

    def loaded_page(self, key: str):
        if str(key) in {"workspace", "project"}:
            return getattr(self, str(key), None)
        registry = getattr(self, "_lazy_pages", None)
        return None if registry is None else registry.loaded(str(key))

    def show_page_key(self, key: str) -> None:
        self.show_page(self.page_index(key))

    def show_page(self, index: int):
        if not 0 <= int(index) < self.stack.count():
            return
        index = int(index)
        workbench=self.loaded_page("workbench")
        editor=getattr(workbench,"_region_editor",None)
        if editor is not None and index != self.page_index("workbench"):
            editor.accept()
            if getattr(workbench,"_region_editor",None) is not None:
                return
        previous = self.stack.currentWidget()
        if previous is not None and hasattr(previous, "set_workspace_active"):
            previous.set_workspace_active(False)
        current = self._ensure_page_loaded(index)
        self.stack.setCurrentWidget(current)
        route_key = ROUTES[index].key
        if route_key != "workspace" and self.project_workspace_manager.active_project is not None:
            try:
                self.project_workspace_manager.touch(active_stage=route_key)
            except Exception:
                logger.debug("failed to persist active workspace stage=%s", route_key, exc_info=True)
        if hasattr(current, "set_workspace_active"):
            current.set_workspace_active(True)
        self._refresh_global_stop_visibility()
        for i,b in enumerate(self.pages):
            b.setChecked(i == index)
        self._pending_page_refresh = index
        self._page_refresh_timer.start(18)

    def _flush_page_refresh(self):
        index = int(getattr(self, "_pending_page_refresh", self.stack.currentIndex()))
        if index != self.stack.currentIndex():
            return
        page = self.stack.widget(index)
        refresh = getattr(page, "refresh", None)
        if callable(refresh):
            refresh()

    # ---------- Processing ----------
    def run_current_page(self, *, reapply_review_after_process: bool = False):
        pair = self.current_pair()
        if pair is None:
            notify(self, "请先完成页面配对。", "warning"); return
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
            notify(self, "请先选择两套页面目录。", "warning"); return
        if self._busy_running():
            QMessageBox.information(self, "处理中", "已有任务正在运行。"); return
        # Pairing is an explicit user action. Opening/restoring a project must
        # never silently trigger the expensive full-book matcher, and the batch
        # buttons must not use a restored processed subset as if it were the
        # complete book.
        if not bool(getattr(self.state, "book_pairing_ready", False)):
            QMessageBox.information(
                self, "尚未智能配对",
                "请先点击“智能配对”完成整本页面配对，再开始从头/继续处理整本。\n"
                "“读取已有运行结果”只读取已经处理过的页面，不会自动配对未处理页面。",
            )
            return
        if not self.state.pairs:
            QMessageBox.information(self, "没有可处理页面", "智能配对后没有找到可处理页面。"); return

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

    def _remember_processing_stage(self, message: str) -> None:
        self._processing_stage_clock.change(str(message))
        self._processing_stage_label = str(message)
        if not self._processing_heartbeat_timer.isActive():
            self._processing_heartbeat_timer.start()

    def _stop_processing_heartbeat(self) -> None:
        self._processing_heartbeat_timer.stop()
        self._processing_stage_clock.reset()
        self._processing_stage_label = ""

    def _processing_heartbeat(self) -> None:
        # Do not synthesize progress, and do not override completion/failure.
        if not self._busy_running() or not self._processing_stage_label:
            self._stop_processing_heartbeat()
            return
        elapsed = self._processing_stage_clock.duration()
        if elapsed < 2:
            return
        self.statusBar().showMessage(format_processing_heartbeat(
            stage=self._processing_stage_label, elapsed=elapsed,
            current=self.progress.value(), total=self.progress.maximum(),
            preparing=(self._prepare_worker is not None and self._prepare_worker.isRunning()),
        ))

    def _maybe_prepare_models_before_run(self, worker: PipelineWorker) -> bool:
        plan = plan_runtime_requirements(worker.config)
        if plan.errors:
            QMessageBox.warning(self, "运行前检查未通过", "\n".join(plan.errors))
            self.statusBar().showMessage("运行前检查未通过", 5000)
            return True
        if not plan.requirements:
            return False
        # A successful preparation for the identical configuration remains
        # valid for a short window. Do not re-launch slow independent runtime
        # probes/download checks on every page of a book. Changes to any model
        # path/backend/profile are part of the fingerprint and force recheck.
        signature = self._ready_model_preparation.key(worker.config)
        if self._ready_model_preparation.is_ready(signature):
            logger.info("reusing successfully prepared runtime for unchanged GUI configuration")
            return False
        # Runtime probes for Paddle/Torch live in isolated Python processes and
        # must never run synchronously on the GUI thread.  Even when model files
        # are already cached, verify their runtime in AutoPrepareModelsWorker.
        labels = " / ".join(req.label for req in plan.requirements)
        self._pending_pipeline_worker = worker
        self._pending_prepare_signature = signature
        self.progress.setRange(0,0); self.progress.setValue(0)
        self.statusBar().showMessage("正在验证/自动准备：" + labels)
        self._remember_processing_stage("正在验证/自动准备：" + labels)
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
        self._remember_processing_stage(str(message))
        self.statusBar().showMessage(str(message))
        if hasattr(self, "models") and self.stack.currentWidget() is self.models:
            self.models.model_download_status.setText(str(message))

    def _prepare_worker_done(self, payload: object):
        message = str((payload or {}).get("message", "自动准备完成。")) if isinstance(payload, dict) else "自动准备完成。"
        self.statusBar().showMessage(message, 5000)
        if hasattr(self, "models"):
            self.models.model_download_status.setText(message)
            # Do not force an expensive all-components probe while the main
            # inference worker is starting: both compete for CPU/RAM. Navigating
            # back to the models tab refreshes it asynchronously when idle.
        worker = self._pending_pipeline_worker
        self._pending_pipeline_worker = None
        if worker is None:
            self._stop_processing_heartbeat()
            self._set_busy(False)
            return
        self._ready_model_preparation.record_success(self._pending_prepare_signature)
        self._pending_prepare_signature = ""
        self._launch_pipeline_worker(worker)

    def _prepare_worker_cancelled(self):
        self._ready_model_preparation.invalidate()
        self._pending_prepare_signature = ""
        self._pending_pipeline_worker = None
        self._stop_processing_heartbeat()
        self.progress.setRange(0,100); self.progress.setValue(0)
        self.statusBar().showMessage("已安全停止模型准备；当前安装/下载步骤已完整结束。", 6000)

    def _prepare_worker_failed(self, message: str):
        self._ready_model_preparation.invalidate()
        self._pending_prepare_signature = ""
        self._pending_pipeline_worker = None
        self._stop_processing_heartbeat()
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
        self._stop_processing_heartbeat()
        self.statusBar().showMessage("已停止，已完成页面已保留", 5000)

    def _launch_pipeline_worker(self, worker: PipelineWorker):
        """Single signal-wiring path after optional runtime/model preparation."""
        self.worker = worker
        self._worker_is_single_page = worker.pair is not None
        self._worker_page_id = page_id_for_pair(worker.pair) if worker.pair is not None else ""
        self._worker_reapply_review = bool(getattr(worker, "reapply_review_after_process", False))
        if worker.pair is None:
            # Single-page reprocessing can leave full-resolution ndarray masks
            # resident after manual work. The batch reads saved project.json;
            # it must not compete with an unbounded GUI copy of those pages.
            self.state.projects_by_page.clear()
            self.state.last_project = None
        # Whole-book progress emits once or more per page. Build this once instead
        # of linearly scanning every pair for every progress signal (O(n^2)).
        self._worker_pair_row_by_name = {}
        for row, pair in enumerate(self.state.pairs):
            self._worker_pair_row_by_name.setdefault(Path(pair.target_path).name, row)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.statusBar().showMessage("正在处理…")
        self._remember_processing_stage("正在处理…")
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
            notify(self, "请等待当前处理/复核任务结束后再执行此操作。", "warning")
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
        self._remember_processing_stage(f"{name} · {message or '正在处理'}")
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
        self._stop_processing_heartbeat()
        self.progress.setRange(0,100)
        self._worker_pair_row_by_name = {}
        worker = self.worker
        self.worker = None
        self._set_busy(None)
        if worker is not None:
            worker.deleteLater()

    def _worker_done(self, project, path):
        self._stop_processing_heartbeat()
        self.progress.setRange(0,100); self.progress.setValue(100)
        if hasattr(project, "page_id"):
            self.state.remember_completed_project(project)
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
        self.save_page_marks(); self._save_workspace_session(active_stage="workbench"); self.project._table_signature = None; self.project._thumb_signature = None
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
            # Preserve the existing explicit cache-invalidation contract when
            # Workbench is already loaded.  If it is still lazy, the result
            # navigation path constructs it and clears the same cache there.
            if hasattr(self, "workbench"):
                try:
                    self.workbench.image.clear_cache()
                except Exception:
                    logger.debug("failed to invalidate loaded workbench image cache after page run", exc_info=True)
            wanted = getattr(self, "_worker_page_id", "")
            self._show_completed_result(wanted)
        elif hasattr(project, "pages"):
            # Whole-book runs also land on the newest successfully completed
            # page. Previously this branch only refreshed the current tab, so
            # finishing a batch left the user on the project page with no
            # visible indication that the result view was ready.
            wanted = ""
            pages = getattr(project, "pages", None)
            try:
                if pages is not None and len(pages):
                    newest = pages[-1]
                    wanted = str(getattr(newest, "page_id", "") or "")
                    if not wanted:
                        pair = getattr(newest, "pair", None)
                        wanted = page_id_for_pair(pair) if pair is not None else ""
            except Exception:
                logger.exception("could not resolve newest completed book page for result navigation")
            self._show_completed_result(wanted)
        else:
            self.refresh_current_page()

    def _show_completed_result(self, page_id: str = ""):
        """Select and display the result workspace after a successful run."""
        if page_id:
            for index, pair in enumerate(self.state.pairs):
                if page_id_for_pair(pair) == page_id:
                    self.set_selected_page(index)
                    break
        try:
            self.show_page_key("workbench")
            workbench = self.workbench
            workbench.image.clear_cache()
            workbench.current_view = "result"
            for button, key in workbench.view_buttons:
                button.setChecked(key == "result")
            workbench.refresh()
            logger.info("auto-navigated to result view page_id=%s", page_id or "current")
        except Exception:
            logger.exception("auto-navigation to completed result failed page_id=%s", page_id)
            self.refresh_current_page()

    def _worker_failed(self, message):
        self._stop_processing_heartbeat()
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
        if self.stack.currentIndex() == self.page_index("workbench"): self.workbench.refresh()

    def refresh_current_page(self):
        page = self.stack.currentWidget()
        refresh = getattr(page, "refresh", None)
        if callable(refresh):
            refresh()

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
    install_dialog_polish(app)  # 所有窗口：最小尺寸不超过屏幕、简单窗口回车即确认
    install_shortcuts(win)      # 快捷键：Ctrl+1…6 切页、PgUp/PgDn 翻页、1…0 切视图、Ctrl+Enter 处理当前页、F1 速查
    apply_application_icon(app=app, window=win)
    win.show()
    result = int(app.exec())
    logger.info("GUI stopped exit_code=%s", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
