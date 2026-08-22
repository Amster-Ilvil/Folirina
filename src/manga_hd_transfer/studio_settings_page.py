from __future__ import annotations

"""Settings / diagnostics / Git source-update page for the desktop studio."""

import logging
from pathlib import Path
import sys

from PySide6.QtCore import QProcess, QThread, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout,
    QWidget,
)

from .app_logging import runtime_log_dir
from .gui_components import Card, StableComboBox
from .gui_dialogs import confirm_action
from .platform_support import desktop_platform_summary
from .source_update import (
    DEFAULT_BRANCH, DEFAULT_REPOSITORY, SourceUpdateInfo, SourceUpdateResult,
    check_source_update, discover_project_root, install_source_update,
    normalize_branch, normalize_repository,
)
from .version import __version__

QComboBox = StableComboBox

logger = logging.getLogger(__name__)

class UpdateCheckWorker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, repo: str, branch: str, project_root: Path, parent=None):
        super().__init__(parent)
        self.repo = repo
        self.branch = branch
        self.project_root = project_root

    def run(self) -> None:
        try:
            info = check_source_update(self.repo, self.branch, project_root=self.project_root)
            self.done.emit(info)
        except BaseException as exc:
            logger.exception("source update check failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class UpdateInstallWorker(QThread):
    progress = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, info: SourceUpdateInfo, parent=None):
        super().__init__(parent)
        self.info = info

    def run(self) -> None:
        try:
            result = install_source_update(self.info, progress=self.progress.emit, refresh_install=True)
            self.done.emit(result)
        except BaseException as exc:
            logger.exception("source update install failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class SettingsPage(QWidget):
    def __init__(self, window: "StudioWindow"):
        super().__init__()
        self.window = window
        self._check_worker: UpdateCheckWorker | None = None
        self._install_worker: UpdateInstallWorker | None = None
        self._update_info: SourceUpdateInfo | None = None
        self._ui_scale = 1.0
        try:
            self.project_root = discover_project_root()
        except Exception:
            self.project_root = Path.cwd().resolve()

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget(); root = QVBoxLayout(body); root.setContentsMargins(16, 16, 16, 16); root.setSpacing(12)
        scroll.setWidget(body); outer.addWidget(scroll, 1)

        appearance = Card("界面与窗口", "主题可以即时切换并记住。主窗口在小屏幕/高 DPI 下会把完整设计页面作为一个整体等比例缩小，内部控件不重新挤压排版。")
        grid = QGridLayout(); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(8)
        grid.addWidget(QLabel("主题"), 0, 0)
        self.theme_combo = QComboBox(); self.theme_combo.addItem("浅色主题", "light"); self.theme_combo.addItem("深色主题", "dark")
        grid.addWidget(self.theme_combo, 0, 1)
        grid.addWidget(QLabel("当前界面缩放"), 1, 0)
        self.scale_label = QLabel("100% · 安全设计尺寸"); self.scale_label.setObjectName("hint")
        grid.addWidget(self.scale_label, 1, 1)
        self.responsive_hint = QLabel("内部布局始终不小于 1480×960 的安全尺寸；窗口更窄时只缩最终显示，不再挤压控件。宽屏/高屏只会给内部布局增加空间，再用同一个比例完整贴合窗口；大窗口也会继续跟随放大。")
        self.responsive_hint.setObjectName("quiet"); self.responsive_hint.setWordWrap(True)
        grid.addWidget(self.responsive_hint, 2, 0, 1, 2)
        grid.setColumnStretch(1, 1)
        appearance.layout.addLayout(grid)
        root.addWidget(appearance)

        updater = Card("Git 仓库更新", "检查 Git 仓库最新分支并直接升级本地代码。Git 工作树只允许 clean + fast-forward；ZIP/portable 安装会先克隆/下载到临时目录，校验后事务式替换，失败自动回滚。")
        repo_grid = QGridLayout(); repo_grid.setHorizontalSpacing(10); repo_grid.setVerticalSpacing(8)
        repo_grid.addWidget(QLabel("GitHub 仓库"), 0, 0)
        self.repo_edit = QLineEdit(DEFAULT_REPOSITORY); self.repo_edit.setReadOnly(True); self.repo_edit.setObjectName("lockedField")
        self.repo_edit.setToolTip("更新仓库已锁定，不能在界面、配置或环境变量中修改。")
        repo_grid.addWidget(self.repo_edit, 0, 1)
        repo_grid.addWidget(QLabel("更新分支"), 1, 0)
        self.branch_edit = QLineEdit(DEFAULT_BRANCH); self.branch_edit.setReadOnly(True); self.branch_edit.setObjectName("lockedField")
        self.branch_edit.setToolTip("更新分支已锁定为 main。")
        repo_grid.addWidget(self.branch_edit, 1, 1)
        locked = QLabel("已锁定 Folirina 官方仓库与 main 分支 · 只读")
        locked.setObjectName("quiet"); repo_grid.addWidget(locked, 0, 2, 2, 1)
        repo_grid.addWidget(QLabel("本地源码"), 2, 0)
        self.root_label = QLabel(str(self.project_root)); self.root_label.setObjectName("quiet"); self.root_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.root_label.setWordWrap(True); repo_grid.addWidget(self.root_label, 2, 1)
        repo_grid.setColumnStretch(1, 1)
        updater.layout.addLayout(repo_grid)

        button_row = QHBoxLayout(); button_row.setSpacing(8)
        self.check_btn = QPushButton("检查 Git 更新"); self.check_btn.setObjectName("softPrimary")
        self.update_btn = QPushButton("从 Git 更新并本地升级"); self.update_btn.setObjectName("primary"); self.update_btn.setEnabled(False)
        self.repo_open_btn = QPushButton("打开仓库")
        button_row.addWidget(self.check_btn); button_row.addWidget(self.update_btn); button_row.addWidget(self.repo_open_btn); button_row.addStretch(1)
        updater.layout.addLayout(button_row)

        self.update_status = QLabel("尚未检查更新"); self.update_status.setObjectName("hint"); self.update_status.setWordWrap(True)
        updater.layout.addWidget(self.update_status)
        self.remote_detail = QLabel(""); self.remote_detail.setObjectName("quiet"); self.remote_detail.setWordWrap(True); self.remote_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        updater.layout.addWidget(self.remote_detail)
        self.update_log = QPlainTextEdit(); self.update_log.setReadOnly(True); self.update_log.setMaximumBlockCount(600); self.update_log.setMinimumHeight(110)
        self.update_log.setPlaceholderText("更新检查和安装过程会记录在这里，同时写入程序级运行日志。")
        updater.layout.addWidget(self.update_log)
        safety = QLabel("升级保护：不会自动降级；不会覆盖有未提交修改的 Git 工作树；portable 更新只替换 src / scripts / tools / pyproject 等程序文件；不会删除 .venv、用户漫画、输出目录、模型或日志。")
        safety.setObjectName("quiet"); safety.setWordWrap(True); updater.layout.addWidget(safety)
        root.addWidget(updater)

        diagnostics = Card("运行与诊断", "发生启动、依赖、更新或 GUI 异常时，优先把这个目录里的日志发来排查。")
        diag_grid = QGridLayout(); diag_grid.setHorizontalSpacing(10); diag_grid.setVerticalSpacing(8)
        diag_grid.addWidget(QLabel("程序版本"), 0, 0); diag_grid.addWidget(QLabel(f"v{__version__}"), 0, 1)
        diag_grid.addWidget(QLabel("运行日志目录"), 1, 0)
        self.log_dir_label = QLabel(str(runtime_log_dir())); self.log_dir_label.setObjectName("quiet"); self.log_dir_label.setWordWrap(True); self.log_dir_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        diag_grid.addWidget(self.log_dir_label, 1, 1)
        self.open_logs_btn = QPushButton("打开运行日志目录"); diag_grid.addWidget(self.open_logs_btn, 2, 1, Qt.AlignmentFlag.AlignLeft)
        diag_grid.setColumnStretch(1, 1); diagnostics.layout.addLayout(diag_grid)
        root.addWidget(diagnostics)
        root.addStretch(1)

        self.repo_edit.setText(DEFAULT_REPOSITORY)
        self.branch_edit.setText(DEFAULT_BRANCH)
        self.sync_theme()

        self.theme_combo.currentIndexChanged.connect(self._theme_changed)
        self.check_btn.clicked.connect(self.check_updates)
        self.update_btn.clicked.connect(self.install_update)
        self.repo_open_btn.clicked.connect(self.open_repository)
        self.open_logs_btn.clicked.connect(self.window.open_runtime_log_dir)

    @property
    def is_updating(self) -> bool:
        return self._install_worker is not None

    @property
    def is_checking_updates(self) -> bool:
        return bool(self._check_worker is not None and self._check_worker.isRunning())

    def shutdown_background_workers(self, timeout_ms: int = 1800) -> bool:
        """Request a safe stop before QApplication tears down QThread wrappers.

        Update installation is transactional and must never be force-terminated;
        if it cannot finish promptly the main window simply refuses to close and
        the user can retry after completion.  The update-check thread is read-only
        but follows the same non-destructive policy.
        """
        all_stopped = True
        for worker in (self._check_worker, self._install_worker):
            try:
                if worker is None or not worker.isRunning():
                    continue
                worker.requestInterruption()
                worker.wait(max(0, int(timeout_ms)))
                if worker.isRunning():
                    all_stopped = False
            except Exception:
                logger.debug("settings worker shutdown failed", exc_info=True)
                all_stopped = False
        return all_stopped

    def sync_theme(self) -> None:
        index = self.theme_combo.findData(self.window.theme_name) if hasattr(self, "theme_combo") else -1
        if index >= 0 and self.theme_combo.currentIndex() != index:
            self.theme_combo.blockSignals(True)
            try:
                self.theme_combo.setCurrentIndex(index)
            finally:
                self.theme_combo.blockSignals(False)

    def set_responsive_scale(self, factor: float) -> None:
        self._ui_scale = float(factor)
        pct = int(round(self._ui_scale * 100))
        suffix = " · 跟随窗口放大" if pct > 100 else (" · 安全设计尺寸" if pct == 100 else " · 整页等比例缩放")
        self.scale_label.setText(f"{pct}%{suffix}")

    def refresh(self) -> None:
        self.sync_theme()
        self.set_responsive_scale(getattr(getattr(self.window, "_responsive_scaler", None), "current_scale", self._ui_scale))
        self.root_label.setText(str(self.project_root))
        self.log_dir_label.setText(str(runtime_log_dir()))
        if self._update_info is not None and not self.is_updating:
            self.update_btn.setEnabled(bool(self._update_info.available) and not self.window._busy_running())

    def set_processing_busy(self, busy: bool) -> None:
        busy = bool(busy)
        # Remote check is read-only and may remain available. Installation is a
        # filesystem mutation and must visually match the same global write gate
        # enforced again inside install_update().
        if hasattr(self, "update_btn"):
            self.update_btn.setEnabled(bool(self._update_info and self._update_info.available) and not busy and not self.is_updating)

    def _theme_changed(self) -> None:
        theme = str(self.theme_combo.currentData() or "light")
        self.window.set_theme(theme)

    def _normalized_repo_branch(self) -> tuple[str, str]:
        # Self-update target is intentionally not configurable. Keep the visible
        # fields synchronized with the code constants in case a caller attempted
        # to alter their text programmatically.
        repo = normalize_repository(DEFAULT_REPOSITORY)
        branch = normalize_branch(DEFAULT_BRANCH)
        self.repo_edit.setText(repo); self.branch_edit.setText(branch)
        return repo, branch

    def _append_log(self, message: str) -> None:
        text = str(message).strip()
        if text:
            self.update_log.appendPlainText(text)

    def check_updates(self) -> None:
        if self._check_worker is not None and self._check_worker.isRunning():
            return
        try:
            repo, branch = self._normalized_repo_branch()
        except Exception as exc:
            QMessageBox.warning(self, "Git 仓库设置无效", str(exc)); return
        self._update_info = None
        self.update_btn.setEnabled(False); self.check_btn.setEnabled(False)
        self.update_status.setText("正在连接 GitHub 检查远端 commit 和版本…")
        self.remote_detail.setText("")
        self._append_log(f"检查更新：{repo} · {branch}")
        worker = UpdateCheckWorker(repo, branch, self.project_root, self)
        self._check_worker = worker
        worker.done.connect(self._check_done)
        worker.failed.connect(self._check_failed)
        worker.finished.connect(self._check_finished)
        worker.start()

    def _check_done(self, payload: object) -> None:
        info = payload if isinstance(payload, SourceUpdateInfo) else None
        if info is None:
            self._check_failed("更新检查返回了无效结果"); return
        self._update_info = info
        self.update_status.setText(info.reason)
        detail = (
            f"本地：v{info.local_version} · commit {info.local_short}\n"
            f"远端：v{info.remote_version} · commit {info.remote_short} · {info.branch}\n"
            f"最新提交：{info.remote_message or '—'}"
        )
        if info.remote_date:
            detail += f" · {info.remote_date}"
        self.remote_detail.setText(detail)
        self._append_log(info.reason)
        self._append_log(f"remote={info.remote_short} version={info.remote_version}")
        self.update_btn.setEnabled(bool(info.available) and not self.window._busy_running())

    def _check_failed(self, message: str) -> None:
        self._update_info = None
        self.update_btn.setEnabled(False)
        self.update_status.setText("检查更新失败；当前版本不会受到影响。")
        self.remote_detail.setText(str(message))
        self._append_log("检查失败：" + str(message))

    def _check_finished(self) -> None:
        worker = self._check_worker
        self._check_worker = None
        self.check_btn.setEnabled(True)
        if worker is not None:
            worker.deleteLater()

    def install_update(self) -> None:
        if self.is_updating:
            return
        if self.window._busy_running():
            QMessageBox.information(self, "任务进行中", "请先等待当前页面处理/模型任务结束，再更新程序代码。")
            return
        info = self._update_info
        if info is None:
            QMessageBox.information(self, "请先检查更新", "请先点击“检查 Git 更新”，确认远端版本后再执行升级。")
            return
        if not info.available:
            QMessageBox.information(self, "无需更新", info.reason)
            return
        text = (
            f"将从 Git 仓库更新本地程序：\n\n"
            f"{info.repository}\n分支：{info.branch}\n"
            f"本地：v{info.local_version}\n远端：v{info.remote_version} · {info.remote_short}\n\n"
            "更新器会先校验并建立回滚点。用户项目、输出、模型、日志和 .venv 不会被删除。\n"
            "更新过程中不要强制结束程序。继续吗？"
        )
        if not confirm_action(
            self, "确认 Git 本地升级", text, confirm_text="升级", destructive=True,
        ):
            return
        if self.window._busy_running():
            QMessageBox.information(self, "任务状态已变化", "确认期间已有其它写任务启动；本次升级已取消。")
            return
        self.check_btn.setEnabled(False); self.update_btn.setEnabled(False)
        self.update_status.setText("正在执行事务式 Git 本地升级…")
        self._append_log("--- 开始本地升级 ---")
        worker = UpdateInstallWorker(info, self)
        self._install_worker = worker
        self.window._set_busy(True)
        worker.progress.connect(self._install_progress)
        worker.done.connect(self._install_done)
        worker.failed.connect(self._install_failed)
        worker.finished.connect(self._install_finished)
        worker.start()

    def _install_progress(self, message: str) -> None:
        self.update_status.setText(str(message))
        self._append_log(str(message))

    def _install_done(self, payload: object) -> None:
        result = payload if isinstance(payload, SourceUpdateResult) else None
        if result is None:
            self._install_failed("更新器返回了无效结果"); return
        self._append_log(f"更新完成：v{result.old_version} → v{result.new_version} · {result.commit[:10]} · {result.method}")
        self.update_status.setText(f"更新完成：v{result.new_version} · 需要重新启动程序")
        self.remote_detail.setText(f"已安装 commit {result.commit[:10]} · {result.method}")
        if confirm_action(
            self, "更新完成",
            f"本地代码已升级到 v{result.new_version}。\n\n需要重新启动才能载入新代码。现在重启吗？",
            confirm_text="立即重启", destructive=False,
        ):
            started = QProcess.startDetached(sys.executable, ["-m", "manga_hd_transfer.launcher"], str(result.project_root))
            ok = bool(started[0]) if isinstance(started, tuple) else bool(started)
            if ok:
                QApplication.quit()
            else:
                QMessageBox.information(self, "请手动重启", "自动重启失败，但代码已经更新完成。请关闭程序后重新打开。")

    def _install_failed(self, message: str) -> None:
        self.update_status.setText("升级失败；已执行回滚保护。")
        self.remote_detail.setText(str(message))
        self._append_log("升级失败：" + str(message))
        QMessageBox.critical(self, "Git 本地升级失败", str(message) + "\n\n更新器已尝试恢复更新前代码。详细过程请查看运行日志。")

    def _install_finished(self) -> None:
        worker = self._install_worker
        self._install_worker = None
        self.window._set_busy(None)
        self.check_btn.setEnabled(True)
        self.update_btn.setEnabled(bool(self._update_info and self._update_info.available))
        if worker is not None:
            worker.deleteLater()

    def open_repository(self) -> None:
        try:
            repo, _branch = self._normalized_repo_branch()
        except Exception as exc:
            QMessageBox.warning(self, "Git 仓库设置无效", str(exc)); return
        QDesktopServices.openUrl(QUrl(f"https://github.com/{repo}"))


__all__ = ["SettingsPage", "UpdateCheckWorker", "UpdateInstallWorker"]
