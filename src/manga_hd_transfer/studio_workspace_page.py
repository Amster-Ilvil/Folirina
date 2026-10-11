from __future__ import annotations

"""Independent project/workspace home for Folirina Studio."""

from pathlib import Path
import time

from PySide6.QtCore import Qt, QTimer, QObject, Signal, QRunnable, QThreadPool
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QListWidget, QListWidgetItem, QLineEdit, QFileDialog, QMessageBox,
)

from .gui_components import Card

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _human_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _count_images(root: Path | None) -> int:
    if root is None or not root.exists():
        return 0
    count = 0
    try:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
                count += 1
    except OSError:
        pass
    return count


class _StatsSignals(QObject):
    ready = Signal(object)


class _StatsTask(QRunnable):
    """Snapshot paths only; never access a QWidget or mutable manager in a worker."""

    def __init__(self, key, source, target, output, include_size):
        super().__init__()
        self.key, self.source, self.target = key, source, target
        self.output, self.include_size = output, include_size
        self.signals = _StatsSignals()

    def run(self):
        result = {"key": self.key, "source": _count_images(self.source),
                  "target": _count_images(self.target), "pages": 0}
        try:
            pages = self.output / "pages" if self.output else None
            if pages is not None and pages.exists():
                result["pages"] = sum(p.is_dir() for p in pages.iterdir())
            if self.include_size:
                result["size"] = _human_bytes(sum(
                    p.stat().st_size for p in Path(self.key).rglob("*") if p.is_file()))
        except OSError:
            result["size"] = "—"
        self.signals.ready.emit(result)


class WorkspacePage(QWidget):
    """Project-oriented entry surface inspired by Novel Formatter's workspace.

    It owns no processing state.  The window/controller remains the composition
    root, while the non-Qt ProjectWorkspaceManager owns durable project folders.
    """

    def __init__(self, window: "StudioWindow"):
        super().__init__()
        self.window = window
        self._size_cache: dict[str, str] = {}
        self._stats_cache = {}
        self._stats_tasks = {}
        self._stats_size_pending = set()
        self._refresh_include_size = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(120)
        self._refresh_timer.timeout.connect(self._refresh_now)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        hero = Card("工作区", "项目化保存 SOURCE / TARGET、页面状态、处理结果和导出文件。整个项目目录可移动、备份并重新导入。")
        root_row = QHBoxLayout(); root_row.setSpacing(8)
        self.root_edit = QLineEdit(); self.root_edit.setReadOnly(True)
        self.root_edit.setPlaceholderText("Folirina Workspace")
        self.root_pick = QPushButton("选择位置…"); self.root_pick.setObjectName("softPrimary")
        self.root_open = QPushButton("打开工作区")
        root_row.addWidget(QLabel("存放位置")); root_row.addWidget(self.root_edit, 1); root_row.addWidget(self.root_pick); root_row.addWidget(self.root_open)
        hero.layout.addLayout(root_row)
        self.root_pick.clicked.connect(self.window.workspace_choose_root)
        self.root_open.clicked.connect(self._open_workspace_root)
        root.addWidget(hero)

        upper = QHBoxLayout(); upper.setSpacing(12)

        projects = Card("项目", "创建新项目，或导入另一个 Folirina 项目目录。双击最近项目即可切换并恢复。")
        action = QHBoxLayout(); action.setSpacing(8)
        self.create_btn = QPushButton("新建项目"); self.create_btn.setObjectName("primary")
        self.import_btn = QPushButton("导入项目文件夹…")
        self.import_backup_btn = QPushButton("导入备份 ZIP…")
        self.refresh_btn = QPushButton("刷新")
        action.addWidget(self.create_btn); action.addWidget(self.import_btn); action.addWidget(self.import_backup_btn); action.addWidget(self.refresh_btn); action.addStretch(1)
        projects.layout.addLayout(action)
        self.project_list = QListWidget(); self.project_list.setMinimumHeight(210)
        projects.layout.addWidget(self.project_list, 1)
        self.create_btn.clicked.connect(self.window.workspace_create_project)
        self.import_btn.clicked.connect(self.window.workspace_import_project)
        self.import_backup_btn.clicked.connect(self.window.workspace_import_backup)
        self.refresh_btn.clicked.connect(lambda: self.refresh(include_size=True))
        self.project_list.itemDoubleClicked.connect(self._open_selected_project)
        upper.addWidget(projects, 3)

        current = Card("当前项目")
        self.project_name = QLabel("未选择项目"); self.project_name.setObjectName("sectionTitle")
        self.project_path = QLabel("创建或打开项目后，输入和输出会固定在项目目录中。")
        self.project_path.setObjectName("quiet"); self.project_path.setWordWrap(True); self.project_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.project_stats = QLabel(""); self.project_stats.setObjectName("quiet"); self.project_stats.setWordWrap(True)
        self.resume_btn = QPushButton("继续上次工作")
        self.resume_btn.setObjectName("primary")
        current.layout.addWidget(self.project_name); current.layout.addWidget(self.project_path); current.layout.addWidget(self.project_stats); current.layout.addWidget(self.resume_btn)
        tool_grid = QGridLayout(); tool_grid.setHorizontalSpacing(8); tool_grid.setVerticalSpacing(8)
        self.open_project_btn = QPushButton("项目目录")
        self.backup_btn = QPushButton("备份项目…")
        self.audit_btn = QPushButton("检查项目")
        self.clear_cache_btn = QPushButton("清理项目缓存")
        self.delete_btn = QPushButton("删除项目")
        self.delete_btn.setObjectName("danger")
        tool_grid.addWidget(self.open_project_btn, 0, 0); tool_grid.addWidget(self.backup_btn, 0, 1)
        tool_grid.addWidget(self.audit_btn, 1, 0); tool_grid.addWidget(self.clear_cache_btn, 1, 1)
        tool_grid.addWidget(self.delete_btn, 2, 0, 1, 2)
        current.layout.addLayout(tool_grid)
        self.resume_btn.clicked.connect(self._resume_active_stage)
        self.open_project_btn.clicked.connect(self._open_active_project)
        self.backup_btn.clicked.connect(self.window.workspace_export_backup)
        self.audit_btn.clicked.connect(self.window.workspace_audit_active)
        self.clear_cache_btn.clicked.connect(self.window.workspace_clear_cache)
        self.delete_btn.clicked.connect(self.window.workspace_delete_active)
        upper.addWidget(current, 2)
        root.addLayout(upper, 1)

        flow = Card("项目流程", "工作区只负责项目生命周期；页面配对、模型设置、替换算法和出版输出仍保持原有独立功能页。")
        flow_grid = QGridLayout(); flow_grid.setHorizontalSpacing(8); flow_grid.setVerticalSpacing(8)
        self.source_status = QLabel("旧中文版：未导入"); self.source_status.setObjectName("quiet")
        self.target_status = QLabel("高清日文版：未导入"); self.target_status.setObjectName("quiet")
        self.output_status = QLabel("处理结果：0 页"); self.output_status.setObjectName("quiet")
        self.import_source_btn = QPushButton("导入旧中文版…")
        self.import_target_btn = QPushButton("导入高清日文版…")
        self.pages_btn = QPushButton("进入页面管理 →"); self.pages_btn.setObjectName("softPrimary")
        self.workbench_btn = QPushButton("进入替换工作台 →")
        self.export_btn = QPushButton("进入出版输出 →")
        flow_grid.addWidget(self.source_status, 0, 0); flow_grid.addWidget(self.import_source_btn, 0, 1)
        flow_grid.addWidget(self.target_status, 1, 0); flow_grid.addWidget(self.import_target_btn, 1, 1)
        flow_grid.addWidget(self.output_status, 2, 0); flow_grid.addWidget(self.pages_btn, 2, 1)
        flow_grid.addWidget(self.workbench_btn, 3, 0); flow_grid.addWidget(self.export_btn, 3, 1)
        flow_grid.setColumnStretch(0, 1); flow_grid.setColumnStretch(1, 1)
        flow.layout.addLayout(flow_grid)
        self.import_source_btn.clicked.connect(lambda: self.window.workspace_import_input("source"))
        self.import_target_btn.clicked.connect(lambda: self.window.workspace_import_input("target"))
        self.pages_btn.clicked.connect(lambda: self.window.show_page_key("project"))
        self.workbench_btn.clicked.connect(lambda: self.window.show_page_key("workbench"))
        self.export_btn.clicked.connect(lambda: self.window.show_page_key("export"))
        root.addWidget(flow)
        root.addStretch(0)
        self.refresh()

    def set_workspace_active(self, active: bool) -> None:
        if active:
            self.request_refresh(immediate=True)

    def request_refresh(self, *, immediate: bool = False, include_size: bool = False) -> None:
        self._refresh_include_size = self._refresh_include_size or bool(include_size)
        if not self.isVisible() and not immediate:
            return
        if immediate:
            self._refresh_timer.stop()
            self._refresh_now()
            return
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def refresh(self, *, include_size: bool = False) -> None:
        self._refresh_include_size = self._refresh_include_size or bool(include_size)
        self._refresh_timer.stop()
        self._refresh_now()

    def _refresh_now(self) -> None:
        include_size = bool(self._refresh_include_size)
        self._refresh_include_size = False
        manager = self.window.project_workspace_manager
        self.root_edit.setText(str(manager.workspace_root))
        active = manager.active_project
        active_text = str(active.resolve()) if active is not None else ""

        self.project_list.clear()
        for info in manager.list_projects():
            marker = "● " if active_text and str(Path(info.path).resolve()) == active_text else ""
            stamp = str(info.updated_at or "").replace("T", " ")[:16]
            item = QListWidgetItem(f"{marker}{info.name}\n{stamp or info.path}")
            item.setData(Qt.ItemDataRole.UserRole, info.path)
            item.setToolTip(info.path)
            self.project_list.addItem(item)
        if self.project_list.count() == 0:
            item = QListWidgetItem("暂无项目")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.project_list.addItem(item)

        enabled = active is not None
        for button in (
            self.resume_btn, self.open_project_btn, self.backup_btn, self.audit_btn, self.clear_cache_btn, self.delete_btn,
            self.import_source_btn, self.import_target_btn, self.pages_btn,
            self.workbench_btn, self.export_btn,
        ):
            button.setEnabled(enabled)

        if not enabled:
            self.project_name.setText("未选择项目")
            self.project_path.setText("创建或打开项目后，输入和输出会固定在项目目录中。")
            self.project_stats.setText("")
            self.source_status.setText("旧中文版：未导入")
            self.target_status.setText("高清日文版：未导入")
            self.output_status.setText("处理结果：0 页")
            return

        context = manager.project_context()
        self.project_name.setText(context.get("name") or active.name)
        self.project_path.setText(str(active))
        cache_key = str(active.resolve())
        cached = self._stats_cache.get(cache_key, {})
        source_count = cached.get("source", "统计中…")
        target_count = cached.get("target", "统计中…")
        output_pages = cached.get("pages", "统计中…")
        size_text = self._size_cache.get(cache_key, "未统计")
        if include_size or time.monotonic() - cached.get("at", 0) > 5:
            if cache_key not in self._stats_tasks:
                task = _StatsTask(cache_key, manager.source_dir, manager.target_dir,
                                  manager.output_dir, include_size)
                task.signals.ready.connect(self._stats_ready)
                self._stats_tasks[cache_key] = task
                QThreadPool.globalInstance().start(task)
            elif include_size:
                self._stats_size_pending.add(cache_key)
        stage = manager.active_stage()
        stage_labels = {
            "project": "页面管理", "pages": "页面管理", "models": "识别与配准",
            "workbench": "替换工作台", "export": "出版输出", "settings": "设置",
            "workspace": "页面管理",
        }
        stage_label = stage_labels.get(stage, "页面管理")
        self.resume_btn.setText(f"继续上次工作 · {stage_label}")
        self.project_stats.setText(
            f"项目大小 {size_text} · 已配对 {len(self.window.state.pairs)} 页 · "
            f"当前第 {self.window.state.selected_index + 1 if self.window.state.pairs else 0} 页"
        )
        self.source_status.setText(f"旧中文版：{source_count} 张")
        self.target_status.setText(f"高清日文版：{target_count} 张")
        self.output_status.setText(f"处理结果：{output_pages} 个页面工作区")

    def _stats_ready(self, result) -> None:
        key = result["key"]
        self._stats_tasks.pop(key, None)
        self._stats_cache[key] = dict(result, at=time.monotonic())
        if "size" in result:
            self._size_cache[key] = result["size"]
        pending = key in self._stats_size_pending
        self._stats_size_pending.discard(key)
        active = self.window.project_workspace_manager.active_project
        if active is not None and str(active.resolve()) == key:
            self.request_refresh(immediate=True, include_size=pending)


    def _resume_active_stage(self) -> None:
        stage = self.window.project_workspace_manager.active_stage()
        route = {
            "": "project", "workspace": "project", "pages": "project", "project": "project",
            "models": "models", "workbench": "workbench", "export": "export", "settings": "settings",
        }.get(stage, "project")
        self.window.show_page_key(route)

    def _open_selected_project(self, item: QListWidgetItem) -> None:
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if path:
            self.window.workspace_activate_project(path)

    def _open_workspace_root(self) -> None:
        path = Path(self.window.project_workspace_manager.workspace_root)
        path.mkdir(parents=True, exist_ok=True)
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            QMessageBox.information(self, "工作区目录", str(path))

    def _open_active_project(self) -> None:
        project = self.window.project_workspace_manager.active_project
        if project is None:
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(project))):
            QMessageBox.information(self, "项目目录", str(project))


__all__ = ["WorkspacePage"]
