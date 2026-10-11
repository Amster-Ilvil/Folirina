from __future__ import annotations

"""Project/page/workspace controller mixin for StudioWindow.

This is deliberately Qt-light and contains no widget construction.  It owns the
state transitions that were previously embedded in the 5k-line gui_qt.py while
preserving ProjectPage's public call surface.
"""

from pathlib import Path

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QInputDialog

from .gui_toast import notify
from .io_utils import load_json, save_json
from .gui_dialogs import confirm_action
from .page_management import (
    PageMark, manual_mark, marks_from_json, marks_to_json, page_mark_key,
    page_type_label, resolve_mark,
)
from .pairing import pair_directories, pairing_method
from .project_workspace import ProjectWorkspaceManager
from .studio_state import studio_default_config
from .session_restore import scan_existing_results


_WORKSPACE_ROOT_KEY = "workspace/root"
_WORKSPACE_ACTIVE_KEY = "workspace/active_project"


class StudioProjectControllerMixin:
    """Stateful project/workspace operations mixed into StudioWindow."""

    # ---------- Durable project workspace ----------
    @staticmethod
    def _workspace_settings() -> QSettings:
        return QSettings("Folirina", "Folirina")

    def _save_workspace_session(self, *, active_stage: str | None = None) -> None:
        manager: ProjectWorkspaceManager = self.project_workspace_manager
        if manager.active_project is None:
            return
        try:
            manager.save_session(
                source_dir=self.state.source_dir,
                target_dir=self.state.target_dir,
                output_dir=self.state.output_dir,
                pairs=self.state.pairs,
                selected_index=self.state.selected_index,
                page_marks=self.state.page_marks,
                book_pairing_ready=self.state.book_pairing_ready,
                restored_processed_only=self.state.restored_processed_only,
                config=self.state.config.model_dump(mode="json") if hasattr(self.state.config, "model_dump") else {},
                active_stage=active_stage,
            )
            self._workspace_settings().setValue(_WORKSPACE_ACTIVE_KEY, str(manager.active_project))
        except Exception as exc:
            self.statusBar().showMessage(f"工作区状态保存失败：{exc}", 5000)

    def _clear_project_runtime_state(self) -> None:
        self.state.pairs = []
        self.state.selected_index = 0
        self.state.last_project = None
        self.state.last_result_path = ""
        self.state.projects_by_page.clear()
        self.state.batch_status.clear()
        self.state.page_marks.clear()
        self.state.unmatched_source.clear()
        self.state.unmatched_target.clear()
        self.state.restored_page_roots.clear()
        self.state.restored_page_origin.clear()
        self.state.book_pairing_ready = False
        self.state.restored_processed_only = False

    def _apply_workspace_session(self, session: dict) -> None:
        self._clear_project_runtime_state()
        self.state.source_dir = str(session.get("source_dir") or "")
        self.state.target_dir = str(session.get("target_dir") or "")
        # A restored project must not silently reuse an external legacy output
        # path even when session.json was written by an older Folirina version.
        manager = self.project_workspace_manager
        self.state.output_dir = (
            str(manager.output_dir)
            if manager.active_project is not None and manager.output_dir is not None
            else str(session.get("output_dir") or "")
        )
        self.state.pairs = list(session.get("pairs") or [])
        self.state.selected_index = min(
            max(0, int(session.get("selected_index") or 0)),
            max(0, len(self.state.pairs) - 1),
        )
        self.state.page_marks = dict(session.get("page_marks") or {})
        # The last project's in-memory settings must never leak into a newly
        # created/legacy project whose session has no config.  In particular,
        # UI controls on an already-loaded workbench would otherwise display
        # values from one project and silently overwrite the next on editing.
        self.state.config = studio_default_config()
        config_payload = session.get("config")
        if isinstance(config_payload, dict) and config_payload:
            try:
                self.state.config = self.state.config.__class__.model_validate(config_payload)
            except Exception:
                # Malformed old config: prefer safe defaults over foreign state.
                pass
        self.state.book_pairing_ready = bool(session.get("book_pairing_ready"))
        self.state.restored_processed_only = bool(session.get("restored_processed_only"))
        # page_management.json remains the page-manager authority when present;
        # session metadata is a portable fallback for newly moved projects.
        self.load_page_marks()
        if hasattr(self, "project"):
            self.project._table_signature = None
            self.project._thumb_signature = None

    def workspace_restore_startup(self) -> None:
        manager: ProjectWorkspaceManager = self.project_workspace_manager
        settings = self._workspace_settings()
        stored = str(settings.value(_WORKSPACE_ACTIVE_KEY, "") or "").strip()
        if not stored:
            return
        try:
            manager.open_project(stored)
            self._apply_workspace_session(manager.load_session())
            self.statusBar().showMessage(f"已恢复项目 · {manager.project_context().get('name', Path(stored).name)}", 3500)
        except Exception:
            # A moved/deleted project must not block application startup.
            settings.remove(_WORKSPACE_ACTIVE_KEY)
            manager.active_project = None

    def workspace_choose_root(self) -> None:
        if self._busy_running():
            notify(self, "请先停止或等待当前任务完成，再切换工作区。", "warning")
            return
        manager: ProjectWorkspaceManager = self.project_workspace_manager
        path = QFileDialog.getExistingDirectory(self, "选择 Folirina 工作区", str(manager.workspace_root))
        if not path:
            return
        self._save_workspace_session()
        manager.set_workspace_root(path, create=True)
        settings = self._workspace_settings()
        settings.setValue(_WORKSPACE_ROOT_KEY, str(manager.workspace_root))
        settings.remove(_WORKSPACE_ACTIVE_KEY)
        self._clear_project_runtime_state()
        self.state.source_dir = ""; self.state.target_dir = ""; self.state.output_dir = ""
        self._refresh_after_project_switch()
        self.statusBar().showMessage(f"工作区已切换：{manager.workspace_root}", 4000)

    def workspace_create_project(self) -> None:
        if self._busy_running():
            notify(self, "请先停止或等待当前任务完成，再创建项目。", "warning")
            return
        name, ok = QInputDialog.getText(self, "新建 Folirina 项目", "项目名称")
        if not ok or not str(name).strip():
            return
        self._save_workspace_session()
        info = self.project_workspace_manager.create_project(str(name).strip())
        self._workspace_settings().setValue(_WORKSPACE_ACTIVE_KEY, info.path)
        self._apply_workspace_session(self.project_workspace_manager.load_session())
        # A new project always owns its output tree immediately.
        self.state.output_dir = str(self.project_workspace_manager.output_dir or "")
        self._save_workspace_session(active_stage="workspace")
        self._refresh_after_project_switch()
        self.statusBar().showMessage(f"已创建项目 · {info.name}", 3500)

    def workspace_import_project(self) -> None:
        if self._busy_running():
            notify(self, "请先停止或等待当前任务完成，再导入项目。", "warning")
            return
        folder = QFileDialog.getExistingDirectory(self, "导入 Folirina 项目文件夹", str(self.project_workspace_manager.workspace_root))
        if not folder:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._save_workspace_session()
            info = self.project_workspace_manager.import_project_folder(folder, copy_into_workspace=True)
            self._workspace_settings().setValue(_WORKSPACE_ACTIVE_KEY, info.path)
            self._apply_workspace_session(self.project_workspace_manager.load_session())
            self._refresh_after_project_switch()
            self.statusBar().showMessage(f"已导入项目 · {info.name}", 4500)
        except Exception as exc:
            QMessageBox.critical(self, "导入项目失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def workspace_activate_project(self, path: str) -> None:
        if self._busy_running():
            notify(self, "请先停止或等待当前任务完成，再切换项目。", "warning")
            return
        try:
            self._save_workspace_session()
            info = self.project_workspace_manager.open_project(path)
            self._workspace_settings().setValue(_WORKSPACE_ACTIVE_KEY, info.path)
            self._apply_workspace_session(self.project_workspace_manager.load_session())
            self._refresh_after_project_switch()
            self.statusBar().showMessage(f"已打开项目 · {info.name}", 3500)
        except Exception as exc:
            QMessageBox.critical(self, "打开项目失败", str(exc))

    def workspace_export_backup(self) -> None:
        manager: ProjectWorkspaceManager = self.project_workspace_manager
        if manager.active_project is None:
            notify(self, "请先创建或打开项目。", "warning")
            return
        if self._busy_running():
            notify(self, "请先停止或等待当前任务完成，再备份项目。", "warning")
            return
        self._save_workspace_session()
        name = manager.project_context().get("name") or manager.active_project.name
        default = str(manager.workspace_root / f"{name}.folirina.zip")
        path, _ = QFileDialog.getSaveFileName(self, "备份 Folirina 项目", default, "Folirina 项目备份 (*.zip)")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            target = manager.export_project_backup(path, include_cache=False)
            self.statusBar().showMessage(f"项目备份完成：{target}", 6000)
        except Exception as exc:
            QMessageBox.critical(self, "项目备份失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def workspace_import_backup(self) -> None:
        if self._busy_running():
            notify(self, "请先停止或等待当前任务完成，再导入项目备份。", "warning")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "导入 Folirina 项目备份", str(self.project_workspace_manager.workspace_root), "Folirina 项目备份 (*.zip)"
        )
        if not path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._save_workspace_session()
            info = self.project_workspace_manager.import_project_archive(path)
            self._workspace_settings().setValue(_WORKSPACE_ACTIVE_KEY, info.path)
            self._apply_workspace_session(self.project_workspace_manager.load_session())
            self._refresh_after_project_switch()
            self.statusBar().showMessage(f"已导入项目备份 · {info.name}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "导入项目备份失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def workspace_audit_active(self) -> None:
        manager: ProjectWorkspaceManager = self.project_workspace_manager
        if manager.active_project is None:
            notify(self, "请先创建或打开项目。", "warning")
            return
        try:
            report = manager.audit_project()
        except Exception as exc:
            QMessageBox.warning(self, "项目检查失败", str(exc))
            return
        if report.get("ok"):
            QMessageBox.information(
                self, "项目检查通过",
                f"项目结构正常。\n配对：{report.get('pair_count', 0)} 页\n处理结果：{report.get('output_pages', 0)} 页",
            )
        else:
            issues = "\n".join(f"• {item}" for item in report.get("issues", []))
            QMessageBox.warning(self, "项目检查发现问题", issues or "发现未知问题")

    def workspace_delete_active(self) -> None:
        manager: ProjectWorkspaceManager = self.project_workspace_manager
        project = manager.active_project
        if project is None:
            return
        if self._busy_running():
            notify(self, "请先停止或等待当前任务完成，再删除项目。", "warning")
            return
        name = manager.project_context().get("name") or project.name
        if not confirm_action(
            self, "删除项目",
            f"确定删除“{name}”及其工作区内的输入副本、处理结果和缓存吗？\n\n{project}",
            confirm_text="删除项目", cancel_text="取消", destructive=True,
        ):
            return
        try:
            manager.delete_project(project)
            self._workspace_settings().remove(_WORKSPACE_ACTIVE_KEY)
            self._clear_project_runtime_state()
            self.state.source_dir = ""; self.state.target_dir = ""; self.state.output_dir = ""
            self._refresh_after_project_switch()
            self.statusBar().showMessage(f"已删除项目 · {name}", 3500)
        except Exception as exc:
            QMessageBox.critical(self, "删除项目失败", str(exc))

    def workspace_import_input(self, kind: str) -> None:
        manager: ProjectWorkspaceManager = self.project_workspace_manager
        if manager.active_project is None:
            notify(self, "请先在工作区创建或打开项目。", "warning")
            return
        if self._busy_running():
            notify(self, "请先停止或等待当前任务完成。", "warning")
            return
        title = "选择旧中文版目录" if kind == "source" else "选择高清日文版目录"
        start = getattr(self.state, f"{kind}_dir", "") or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, title, start)
        if not folder:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            destination = manager.import_input_directory(kind, folder)
            setattr(self.state, f"{kind}_dir", str(destination))
            self.state.output_dir = str(manager.output_dir or "")
            # New input invalidates pairings and page-local runtime references;
            # it does not delete output automatically, so the old result remains
            # available on disk until the user reruns or removes the project.
            self.state.pairs = []
            self.state.selected_index = 0
            self.state.batch_status.clear()
            self.state.page_marks.clear()
            self.state.unmatched_source.clear(); self.state.unmatched_target.clear()
            self.state.projects_by_page.clear(); self.state.restored_page_roots.clear(); self.state.restored_page_origin.clear()
            self.state.book_pairing_ready = False; self.state.restored_processed_only = False
            self._save_workspace_session(active_stage="pages")
            self._refresh_after_project_switch()
            label = "旧中文版" if kind == "source" else "高清日文版"
            self.statusBar().showMessage(f"{label}已复制到项目：{destination}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "导入输入失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def workspace_clear_cache(self) -> None:
        manager: ProjectWorkspaceManager = self.project_workspace_manager
        if manager.active_project is None:
            return
        try:
            removed = manager.clear_project_cache()
            self.statusBar().showMessage(f"已清理项目缓存 · {removed / (1024 * 1024):.1f} MB", 4000)
            if hasattr(self, "workspace"):
                self.workspace.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "清理缓存失败", str(exc))

    def _refresh_after_project_switch(self) -> None:
        if hasattr(self, "project"):
            self.project._table_signature = None
            self.project._thumb_signature = None
        self.refresh_current_page()
        if hasattr(self, "workspace"):
            self.workspace.refresh()

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
            self._save_workspace_session(active_stage="pages")
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

    def page_mark_for_pair(self, pair):
        return resolve_mark(self.state.page_marks, pair)

    def mark_page_rows(self, rows: list[int], page_type: str):
        if self._busy_running():
            notify(self, "请先停止或等待当前任务完成，再修改页面类型。", "warning")
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
            notify(self, "请先停止或等待当前任务完成，再修改页面类型。", "warning")
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
        # In a durable workspace, Page Manager's SOURCE/TARGET pickers become
        # imports into the project instead of hidden external references.
        if self.project_workspace_manager.active_project is not None:
            if kind in {"source", "target"}:
                self.workspace_import_input(kind)
                return
            if kind == "output":
                self.state.output_dir = str(self.project_workspace_manager.output_dir or "")
                self._save_workspace_session(active_stage="pages")
                self.project.refresh()
                self.statusBar().showMessage("工作区项目的处理结果固定保存在项目 output 目录。", 4000)
                return

        if kind == "output" and self.project_workspace_manager.active_project is None and not self.state.restored_processed_only:
            suggested=(Path(self.state.target_dir or self.state.source_dir).name if (self.state.target_dir or self.state.source_dir) else "未命名项目")
            name,ok=QInputDialog.getText(self,"创建命名项目","项目名称（将作为输出目录的父文件夹）",text=suggested)
            if not ok or not str(name).strip():
                return
            info=self.project_workspace_manager.create_project(str(name).strip())
            self._workspace_settings().setValue(_WORKSPACE_ACTIVE_KEY,info.path)
            self._apply_workspace_session(self.project_workspace_manager.load_session())
            self.state.output_dir=str(self.project_workspace_manager.output_dir)
            self._save_workspace_session(active_stage="pages")
            self._refresh_after_project_switch()
            self.statusBar().showMessage(f"输出将统一保存至：{self.state.output_dir}",5000)
            return
        title = {"source":"选择旧中文版目录", "target":"选择高清日文版目录", "output":"选择输出目录"}[kind]
        start = getattr(self.state, f"{kind}_dir", "") or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, title, start)
        if not path:
            return
        if kind in {"source", "target"} and self.project_workspace_manager.active_project is None:
            suggested=Path(path).name or "未命名项目"
            name, ok=QInputDialog.getText(self,"创建命名项目", "项目名称（所有输出均保存在该项目中）", text=suggested)
            if not ok or not str(name).strip():
                return
            try:
                manager=self.project_workspace_manager
                info=manager.create_project(str(name).strip())
                self._workspace_settings().setValue(_WORKSPACE_ACTIVE_KEY,info.path)
                self._apply_workspace_session(manager.load_session())
                destination=manager.import_input_directory(kind,path)
                setattr(self.state,f"{kind}_dir",str(destination))
                self.state.output_dir=str(manager.output_dir)
                self._save_workspace_session(active_stage="pages")
                self._refresh_after_project_switch()
                self.statusBar().showMessage(f"已创建项目：{info.path} · 输入已导入",5000)
            except Exception as exc:
                QMessageBox.critical(self,"创建项目或导入失败",str(exc))
            return
        in_session_marks = dict(self.state.page_marks) if kind == "output" else {}
        setattr(self.state, f"{kind}_dir", path)
        if kind in {"source", "target"}:
            self.state.pairs = []; self.state.selected_index = 0; self.state.batch_status.clear()
            self.state.page_marks.clear(); self.state.unmatched_source.clear(); self.state.unmatched_target.clear()
            self.state.book_pairing_ready = False
            self.state.restored_processed_only = False
        if kind in {"source", "target", "output"}:
            self.state.last_project = None; self.state.last_result_path = ""; self.state.projects_by_page.clear()
            self.state.restored_page_roots.clear(); self.state.restored_page_origin.clear()
        if kind == "output":
            self.load_page_marks()
            self.state.page_marks.update(in_session_marks)
            if self.state.page_marks:
                self.save_page_marks()
        self.project._table_signature = None; self.project._thumb_signature = None
        self.refresh_all()

    def restore_existing_results(self):
        if self._busy_running():
            notify(self, "请先停止或等待当前任务完成。", "warning")
            return
        start = self.state.output_dir or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "读取已有运行结果（请选择输出目录 / pages / 单页目录）", start)
        if not path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            scan_path = path
            manager = self.project_workspace_manager
            if manager.active_project is not None:
                # Durable projects own their result tree. Importing an external
                # result therefore makes a real project-local copy first.
                scan_path = str(manager.bind_existing_output(path))
            session = scan_existing_results(scan_path, processed_only=True)
            self.state.output_dir = str(session.output_root)
            self.state.source_dir = session.source_dir
            self.state.target_dir = session.target_dir
            self.state.pairs = [row.pair for row in session.pages]
            self.state.selected_index = 0
            self.state.projects_by_page.clear(); self.state.batch_status.clear()
            self.state.restored_page_roots = {row.page_id: str(row.page_root) for row in session.pages}
            self.state.restored_page_origin = {row.page_id: "命令行/Codex 已有结果" for row in session.pages}
            self.state.unmatched_source = []; self.state.unmatched_target = []
            self.state.book_pairing_ready = False
            self.state.restored_processed_only = True
            self.project._table_signature = None; self.project._thumb_signature = None
            self.load_page_marks()
            self._save_workspace_session(active_stage="pages")
            warning = f" · 跳过/警告 {len(session.warnings)}" if session.warnings else ""
            msg = (
                f"已读取 {len(session.pages)} 页已处理结果{warning}。当前未执行智能配对；"
                "可直接检查/补漏，处理整本前请点击“智能配对”。"
            )
            self.statusBar().showMessage(msg, 8000)
            self.refresh_all()
        except Exception as exc:
            QMessageBox.critical(self, "读取已有结果失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _pair_now(self) -> tuple[int, int]:
        s = self.state
        pairs, us, ut = pair_directories(s.source_dir, s.target_dir, s.config.pairing)
        s.pairs = pairs; s.unmatched_source = list(us); s.unmatched_target = list(ut); s.selected_index = 0
        s.book_pairing_ready = True
        s.restored_processed_only = False
        self.load_page_marks()
        self.project._table_signature = None; self.project._thumb_signature = None
        self._save_workspace_session(active_stage="pages")
        return len(us), len(ut)

    def auto_pair(self):
        s = self.state
        if not s.source_dir or not s.target_dir:
            QMessageBox.warning(self, "缺少输入", "请先选择旧中文版和高清日文版目录。")
            return
        if self._busy_running():
            notify(self, "请先停止或等待当前任务完成。", "warning")
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
        if hasattr(self, "workspace"):
            self.workspace.refresh()

    def set_selected_page(self, index: int, *, sync_table: bool = True):
        if not self.state.pairs:
            self.state.selected_index = 0; return
        idx = max(0, min(int(index), len(self.state.pairs)-1))
        self.state.selected_index = idx
        if hasattr(self, "project"):
            self.project.set_current_page(idx, sync_table=sync_table)
        self.refresh_preview()

    def current_pair(self):
        if not self.state.pairs:
            return None
        return self.state.pairs[max(0, min(self.state.selected_index, len(self.state.pairs)-1))]

    def _default_output(self):
        manager = self.project_workspace_manager
        if manager.active_project is not None:
            self.state.output_dir = str(manager.output_dir or "")
            Path(self.state.output_dir).mkdir(parents=True, exist_ok=True)
            return self.state.output_dir
        if not self.state.output_dir:
            # Headless/shortcut execution can reach here before the user
            # visits Workspace. Never scatter generated files in a generic
            # Folirina_Output folder alongside source books.
            source = self.state.target_dir or self.state.source_dir
            name = Path(source).name if source else '未命名项目'
            info = manager.create_project(name)
            self._workspace_settings().setValue(_WORKSPACE_ACTIVE_KEY, info.path)
            # Retain the current book selection; do NOT reset runtime state
            # by calling _apply_workspace_session during an active workflow.
            self.state.output_dir = str(manager.output_dir)
            self._save_workspace_session(active_stage='pages')
        # An explicit old-style output path remains readable for migration.
        return self.state.output_dir


__all__ = ["StudioProjectControllerMixin"]
