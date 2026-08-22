from __future__ import annotations

"""Headless audit for GUI interaction/lifecycle contracts.

The release runner may not have PySide6.  This audit therefore inspects the
source files that own widget wiring and controller state rather than importing
Qt.  It complements runtime tests by catching dead controls and lifecycle
regressions before a macOS build is published.
"""

import ast
import json
from pathlib import Path
from typing import Any


_GUI_FILES = (
    "gui_qt.py",
    "studio_project_page.py",
    "studio_model_page.py",
    "studio_export_page.py",
    "studio_settings_page.py",
)


def _read(package_root: Path, name: str) -> str:
    return (package_root / name).read_text(encoding="utf-8")


def _self_button_wiring_issues(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    buttons: dict[str, int] = {}
    connected: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "QPushButton":
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        buttons[target.attr] = int(node.lineno)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "connect":
            signal = node.func.value
            if isinstance(signal, ast.Attribute) and signal.attr in {"clicked", "pressed", "released", "toggled"}:
                owner = signal.value
                if (
                    isinstance(owner, ast.Attribute)
                    and isinstance(owner.value, ast.Name)
                    and owner.value.id == "self"
                ):
                    connected.add(owner.attr)
    return [f"{path.name}:{line}: self.{name}" for name, line in sorted(buttons.items()) if name not in connected]


def run_gui_interaction_audit(package_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parent
    sources = {name: _read(root, name) for name in _GUI_FILES}
    all_source = "\n".join(sources.values())
    dead_buttons: list[str] = []
    for name in _GUI_FILES:
        dead_buttons.extend(_self_button_wiring_issues(root / name))

    gui = sources["gui_qt.py"]
    project = sources["studio_project_page.py"]
    models = sources["studio_model_page.py"]
    export = sources["studio_export_page.py"]
    page_prep = _read(root, "pipeline_page_prep.py")

    checks = {
        "self_owned_buttons_connected": not dead_buttons,
        "no_native_question_sheets": "QMessageBox.question" not in all_source,
        "reletter_font_refresh_connected": "self.reletter_font_refresh.clicked.connect(self._refresh_reletter_font_catalog)" in project,
        "reletter_font_catalog_connected": "self.reletter_font_catalog.currentIndexChanged.connect(self._apply_reletter_catalog_font)" in project,
        "global_stop_becomes_visible_off_project_page": "self.stop_button.setVisible(visible)" in gui and "self.stack.currentIndex() != 0" in gui,
        "global_stop_is_attached_to_rail_layout": "rail_tools.addWidget(self.stop_button,2,0,1,2)" in gui,
        "pipeline_worker_released_after_finish": "worker = self.worker" in gui and "self.worker = None" in gui and "worker.deleteLater()" in gui,
        "model_write_participates_in_global_busy": "model_write_running=hasattr(self, \"models\") and self.models.has_write_task_running()" in gui,
        "model_write_start_refreshes_global_busy": models.count("self.window._set_busy(None)") >= 4,
        "model_write_close_guard_present": "def shutdown_write_workers" in models and "self.models.shutdown_write_workers()" in gui,
        "model_imports_share_write_gate": "self.model_import_buttons" in models and "button.setEnabled(not blocked)" in models,
        "unsupported_dependency_buttons_remain_disabled": "self._dependency_installable" in models and "bool(self._dependency_installable.get(key, True)) and not blocked" in models,
        "project_stop_only_enabled_when_cancellable": "self.project.cancel.setEnabled(cancellable)" in gui,
        "export_cleanup_busy_guarded": "if self.window._busy_running():" in export and "self.cleanup.setEnabled(not bool(busy))" in export,
        "export_cleanup_matches_component_mask_copy": "cleanup_output_workspace(out, aggressive_component_masks=True)" in export,
        "fresh_page_run_uses_strict_mode_cleanup": "clear_stale_mode_outputs(root, strict=True)" in page_prep,
        "shared_application_confirmation_dialog": "from .gui_dialogs import confirm_action" in all_source,
    }
    return {
        "schema": "folirina.gui_interaction_audit.v1",
        "pass": all(checks.values()),
        "checks": checks,
        "dead_buttons": dead_buttons,
        "check_count": len(checks),
    }


def main() -> int:
    report = run_gui_interaction_audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
