from __future__ import annotations

import ast
from pathlib import Path


GUI = Path(__file__).resolve().parents[1] / "src" / "manga_hd_transfer" / "gui_qt.py"
SOURCE = GUI.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _class_source(name: str) -> str:
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return ast.get_source_segment(SOURCE, node) or ""
    raise AssertionError(f"missing class {name}")


def test_editor_views_refit_after_layout_resize_and_allow_pan() -> None:
    mask = _class_source("MaskPaintView")
    region = _class_source("RegionSelectView")
    for body in (mask, region):
        assert "def resizeEvent" in body
        assert "_auto_fit" in body
        assert "QTimer.singleShot" in body
        assert "MiddleButton" in body
        assert "def mouseDoubleClickEvent" in body


def test_editor_dialogs_are_screen_aware_and_controls_are_not_one_overwide_row() -> None:
    mask = _class_source("MaskEditorDialog")
    reveal = _class_source("RevealMaskDialog")
    manual = _class_source("ManualEffectDialog")
    assert "_configure_responsive_dialog" in mask
    assert "_configure_responsive_dialog" in reveal
    assert "_configure_responsive_dialog" in manual
    assert "brush_row" in mask and "action_row" in mask
    assert "brush_row" in reveal and "action_row" in reveal
    assert "WrapLongRows" in manual


def test_workbench_preview_toolbar_wraps_and_sidebar_switches_vertical_when_narrow() -> None:
    workbench = _class_source("WorkbenchPage")
    assert "toolbar=QGridLayout()" in workbench
    assert "i//4" in workbench and "i%4" in workbench
    assert "def _update_responsive_workbench" in workbench
    assert "self.width() < 920" in workbench
    assert "Qt.Orientation.Vertical" in workbench
    assert "ScrollBarAlwaysOff" in workbench
    assert "AdjustToMinimumContentsLengthWithIcon" in workbench
    assert "WrapLongRows" in workbench


def test_manual_omission_baseline_is_owned_by_core_service_not_gui() -> None:
    workbench = _class_source("WorkbenchPage")
    assert "def _capture_manual_effect_base" not in workbench
    service = (Path(__file__).resolve().parents[1] / "src" / "manga_hd_transfer" / "manual_review_service.py").read_text()
    state = (Path(__file__).resolve().parents[1] / "src" / "manga_hd_transfer" / "result_state.py").read_text()
    assert "ensure_manual_baseline(page_dir)" in service
    assert "final_auto.png" in state
    assert "manual_effect_base.png" in state
    assert "Removing the final manual region" in workbench


def test_reveal_preview_uses_cached_full_reveal_for_interactive_brush_speed() -> None:
    reveal = _class_source("RevealMaskDialog")
    assert "self._full_reveal" in reveal
    refresh_start = reveal.index("def _refresh_preview")
    refresh = reveal[refresh_start:]
    assert "out[gate]=self._full_reveal[gate]" in refresh


def test_workbench_exposes_auto_manual_effect_candidates_without_widening_sidebar() -> None:
    workbench = _class_source("WorkbenchPage")
    assert "使用候选区域" in workbench
    assert "def _add_next_manual_effect_candidate" in workbench
    assert "_unhandled_manual_effect_candidates" in SOURCE
    assert "manual_effect_candidates" in workbench


def test_workbench_marks_publication_safety_as_removed_and_forces_legacy_config_off() -> None:
    workbench = _class_source("WorkbenchPage")
    assert "出版安全门禁已移除" in workbench
    assert "publication_safety_enabled=False" in workbench
    assert "safety=False" in workbench


def test_manual_effect_ui_explains_mode_scope_and_white_bubble_nudge():
    from pathlib import Path
    src = (Path(__file__).parents[1] / "src" / "manga_hd_transfer" / "gui_qt.py").read_text()
    assert "彩色开放式文字 · 擦除显字（只改文字）" in src
    assert "白色气泡 · 文字迁移 + X/Y 微调（不贴背景）" in src
    assert "选框只是搜索范围，不会作为整块写入范围" in src
    assert "没有可提交的中文文字" in src
    assert "Reveal 补丁为空" in src


def test_manual_review_sync_uses_shared_core_result_state_contract() -> None:
    workbench = _class_source("WorkbenchPage")
    assert "commit_reviewed_result" in SOURCE
    sync_start = workbench.index("def _sync_reviewed_book_final")
    sync = workbench[sync_start: sync_start + 700]
    assert "commit_reviewed_result(final_path.parent, final_path)" in sync
    core = (Path(__file__).resolve().parents[1] / "src" / "manga_hd_transfer" / "result_state.py").read_text()
    assert "review_sync.json" in core
    assert "final_reviewed.png" in core and "final.png" in core
    assert "review_sync.v3" in core


def test_manual_effect_apply_announces_final_sync_to_user() -> None:
    workbench = _class_source("WorkbenchPage")
    assert "showMessage" in workbench
    assert "final_reviewed.png → final.png" in workbench


def test_reveal_save_commits_directly_before_parent_dialog_closes() -> None:
    manual = _class_source("ManualEffectDialog")
    assert "commit_handler=None" in manual
    assert "self._commit_handler(self.result_row(), self.result_reveal_mask(), self.result_reveal_patch())" in manual
    assert "self._committed_directly=True" in manual
    assert "self.done(QDialog.DialogCode.Accepted)" in manual
    commit_pos = manual.index("self._commit_handler(self.result_row()")
    done_pos = manual.index("self.done(QDialog.DialogCode.Accepted)")
    assert commit_pos < done_pos


def test_workbench_passes_commit_callback_and_core_service_owns_final_verification() -> None:
    workbench = _class_source("WorkbenchPage")
    assert "commit_handler=_commit" in workbench
    assert "trace_handler=_trace" in workbench
    assert "manual_gui_flow.json" in workbench
    assert "reveal_editor_saved" in _class_source("ManualEffectDialog")
    service = (Path(__file__).resolve().parents[1] / "src" / "manga_hd_transfer" / "manual_review_service.py").read_text()
    assert "final_verified" in service
    assert "preview_patch_exact" in service
    assert "commit_manual_effect" in workbench


def test_core_manual_commit_reopens_saved_patch_and_verifies_final_pixels() -> None:
    service = (Path(__file__).resolve().parents[1] / "src" / "manga_hd_transfer" / "manual_review_service.py").read_text()
    assert "preview_patch_exact" in service
    assert "cv2.imread" in service
    assert "np.array_equal(reviewed[sel], patch[:, :, :3][sel])" in service
    assert "final_reviewed" in service and "final.png" in service


def test_manual_gui_legacy_json_rows_are_type_safe() -> None:
    workbench = _class_source("WorkbenchPage")
    assert "_json_dict_rows" in SOURCE
    assert '_json_dict_rows(audit.get("manual_effect_applied"))' in workbench
    assert '_json_dict_rows(overrides.get("manual_effect_regions"))' in workbench


def test_experimental_checkbox_is_explicit_opt_in_and_switches_route() -> None:
    project = _class_source("ProjectPage")
    assert "显示并启用实验模式" in project
    method_start = project.index("def _set_experimental_visible")
    method = project[method_start:method_start + 1800]
    assert 'findData("aligned_overlay_reveal")' in method
    assert 'self.mode.setCurrentIndex(idx)' in method
    assert "Auto intentionally cannot select this route" in method
    assert "cfg.aligned_overlay_reveal.enabled = self.show_experimental.isChecked()" in project
