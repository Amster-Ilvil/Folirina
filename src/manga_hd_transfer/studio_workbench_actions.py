from __future__ import annotations

"""Workbench commands, review actions, and state transitions.

Kept separate from ``WorkbenchPage`` so the page module remains UI composition only.
"""

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
from .gui_toast import notify
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


class WorkbenchActionsMixin:
    def _current_page_dir(self) -> Path | None:
        ws=self._workspace()
        return ws.page_root if ws is not None else None

    def _edit_clear_mask(self):
        page_dir=self._current_page_dir(); pair=self.window.current_pair()
        if page_dir is None or pair is None or not (page_dir/"target_original.png").exists():
            notify(self, "请先处理当前页，生成目标图和自动清除蒙版。", "warning")
            return
        import cv2
        import numpy as np
        target=cv2.imread(str(page_dir/"target_original.png"),cv2.IMREAD_COLOR)
        if target is None:
            QMessageBox.warning(self,"无法读取","当前页 target_original.png 无法读取。"); return
        # Use the same review-safe projection as “应用蒙版”.  Automatic renderer
        # artifacts may be whole-container masks and can overlap a face/artwork;
        # manual masks remain exact reviewer authority and are never projected.
        from .review_apply import _load_effective_clear_mask
        mask,_mask_source=_load_effective_clear_mask(page_dir,target.shape[:2])
        if mask is None or mask.shape != target.shape[:2]: mask=np.zeros(target.shape[:2],dtype=np.uint8)
        dlg=MaskEditorDialog(page_dir/"target_original.png",mask,self.window)
        if dlg.exec()!=QDialog.DialogCode.Accepted: return
        try:
            with PageRunGuard(page_dir, "gui:manual-clear-mask"):
                write_image(page_dir/"manual_clear_mask.png",dlg.result_mask())
        except Exception as exc:
            QMessageBox.warning(self,"蒙版保存失败",str(exc)); return
        self.current_view="clear_mask"
        for b,k in self.view_buttons: b.setChecked(k==self.current_view)
        self.window.statusBar().showMessage("已保存自动清除蒙版修订；正在后台生成去字预览…",5000)
        cfg=self.window.state.config.model_copy(deep=True)
        self.window.run_page_action(
            "生成去字预览", lambda: generate_remove_text_preview(page_dir,cfg),
            lambda _out: self.refresh(), failure_title="去字预览失败",
        )

    def _remove_text_only(self):
        page_dir=self._current_page_dir()
        if page_dir is None or not (page_dir/"target_original.png").exists():
            notify(self, "请先处理当前页。", "warning")
            return
        self._sync(); cfg=self.window.state.config.model_copy(deep=True)

        def done(out):
            self.current_view="removed"
            for b,k in self.view_buttons: b.setChecked(k=="removed")
            self.window.statusBar().showMessage(f"只清日文完成：{Path(str(out)).name}",5000)
            self.refresh()

        self.window.run_page_action(
            "只清日文", lambda: generate_remove_text_preview(page_dir,cfg), done,
            failure_title="只清日文失败",
        )

    def _apply_mask_review(self):
        page_dir=self._current_page_dir()
        if page_dir is None or not (page_dir/"project.json").exists():
            notify(self, "请先处理当前页。", "warning")
            return
        self._sync(); cfg=self.window.state.config.model_copy(deep=True)

        def done(final):
            final=Path(str(final)); self.window.state.last_result_path=str(final)
            pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""; proj=self.window.state.projects_by_page.get(page_id)
            if proj is not None: proj.artifacts["final"]=str(final)
            self._sync_reviewed_book_final(final); self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage("已把当前清除蒙版应用到最终结果。",5000)
            self.refresh()

        self.window.run_page_action(
            "应用复核蒙版", lambda: apply_review_page(page_dir,cfg), done,
            failure_title="应用蒙版失败",
        )

    def _reset_clear_mask(self):
        page_dir=self._current_page_dir()
        if page_dir is None or not (page_dir / "manual_clear_mask.png").exists():
            return
        self._sync()
        cfg=self.window.state.config.model_copy(deep=True)
        from .manual_review_service import reset_manual_clear_mask_review
        def done(final):
            if final is not None:
                final=Path(str(final))
                self.window.state.last_result_path=str(final)
                self._sync_reviewed_book_final(final)
            self.current_view="result" if final is not None else "clear_mask"
            for b,k in self.view_buttons: b.setChecked(k==self.current_view)
            self.window.statusBar().showMessage("已恢复自动蒙版并同步重建本页最终结果。",5000)
            self.refresh()
        self.window.run_page_action(
            "恢复自动蒙版", lambda: reset_manual_clear_mask_review(page_dir,cfg), done,
            failure_title="恢复自动蒙版失败",
        )


    def _edit_force_transfer_mask(self):
        page_dir = self._current_page_dir()
        if page_dir is None or not (page_dir / "target_original.png").exists() or not (page_dir / "project.json").exists():
            notify(self, "请先处理当前页，再使用人工强制迁移蒙版。", "warning")
            return
        import cv2
        import numpy as np
        display_path = page_dir / "final_reviewed.png"
        if not display_path.exists():
            display_path = page_dir / "final.png"
        if not display_path.exists():
            display_path = page_dir / "target_original.png"
        target = cv2.imread(str(page_dir / "target_original.png"), cv2.IMREAD_COLOR)
        if target is None:
            QMessageBox.warning(self, "无法读取", "当前页 target_original.png 无法读取。")
            return
        mask_path = page_dir / "manual_force_transfer_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if mask is None or mask.shape != target.shape[:2]:
            mask = np.zeros(target.shape[:2], dtype=np.uint8)
        mask_before=mask.copy()
        project = normalize_project(load_json(page_dir / "project.json"))
        try:
            auto_original, _auto_source, auto_diag = manual_force_auto_evidence_masks(
                page_dir, project=project, target=target, include_override=False
            )
            override_path = page_dir / "manual_force_auto_target_override.png"
            auto_target = cv2.imread(str(override_path), cv2.IMREAD_GRAYSCALE) if override_path.exists() else None
            if auto_target is None or auto_target.shape != target.shape[:2]:
                auto_target = auto_original.copy()
            else:
                auto_target = (auto_target > 0).astype(np.uint8) * 255
        except Exception:
            auto_original = np.zeros(target.shape[:2], dtype=np.uint8)
            auto_target = auto_original.copy()
            auto_diag = {}
        settings_path = page_dir / "manual_force_settings.json"
        old_settings = load_json(settings_path) if settings_path.exists() else {}
        auto_default = bool(old_settings.get("use_auto_evidence", True))
        dlg = MaskEditorDialog(
            display_path, mask, self.window,
            title="人工强制迁移蒙版 · 人工 + OCR/自动检测",
            hint_text=(
                "红色 = 人工蒙版；蓝色 = OCR/自动检测蒙版；橙色 = 两层重合。"
                "现在两层都可以直接编辑：先选“人工蒙版”或“OCR/自动检测蒙版”，再用“涂抹/消除”画笔修正。"
                "蓝色自动区域改错后会保存为本页自动蒙版修订，不需要先复制到人工层；“恢复自动原始”可撤回对蓝色层的修改。"
                "重新处理时只使用修订后的紧凑文字蒙版，避免把整块气泡/画面当成文字区域。"
            ),
            save_label="保存并按复合蒙版重跑",
            reference_mask=auto_target,
            reference_original_mask=auto_original,
            reference_label="OCR / 自动检测",
            auto_assist_default=auto_default,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        edited_manual=dlg.result_mask(); edited_auto=dlg.result_reference_mask()
        manual_changed=not np.array_equal(edited_manual,mask_before)
        reference_changed=not np.array_equal(edited_auto,auto_original)
        write_image(mask_path,edited_manual)
        override_path = page_dir / "manual_force_auto_target_override.png"
        if np.array_equal(edited_auto, auto_original):
            try: override_path.unlink(missing_ok=True)
            except OSError: pass
            auto_override = False
        else:
            write_image(override_path, edited_auto); auto_override = True
        save_json(page_dir / "manual_force_settings.json", {
            "schema": "manga_hd_translation_transfer.manual_force_settings.v3",
            "owner_transfer_mode": str(as_dict(project.get("meta")).get("transfer_mode", "") or self.window.state.config.transfer.mode),
            "use_auto_evidence": bool(dlg.use_auto_evidence()),
            "auto_reference_pixels": int(cv2.countNonZero(edited_auto)),
            "auto_reference_original_pixels": int(cv2.countNonZero(auto_original)),
            "auto_target_override": bool(auto_override),
            "auto_reference": auto_diag,
        })
        self.current_view="result"
        for b,k in self.view_buttons: b.setChecked(k=="result")
        if reference_changed:
            # Editing the blue automatic authority can invalidate the automatic
            # transfer itself, so keep the conservative full page rerun here.
            self.window.statusBar().showMessage("自动/OCR蒙版已修改，正在重新处理当前页并恢复人工复核…",7000)
            self.refresh(); self.window.run_current_page(reapply_review_after_process=True)
            return
        # Red manual-force paint is a review-layer operation. Replaying review
        # artifacts is enough and is dramatically faster than rerunning detection,
        # registration and the page renderer.
        cfg=self.window.state.config.model_copy(deep=True)
        def done(final):
            final=Path(str(final)); self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.window.statusBar().showMessage("人工强制迁移蒙版已应用（快速复核路径）。",5000); self.refresh()
        if manual_changed:
            self.window.run_page_action("应用人工强制迁移蒙版",lambda: apply_review_page(page_dir,cfg),done,failure_title="应用人工强制迁移蒙版失败")
        else:
            self.window.statusBar().showMessage("蒙版没有变化。",2500); self.refresh()

    def _reset_force_transfer_mask(self):
        page_dir = self._current_page_dir()
        if page_dir is None:
            return
        cfg=self.window.state.config.model_copy(deep=True)

        def done(final):
            if final is not None:
                final=Path(str(final)); self.window.state.last_result_path = str(final)
                self._sync_reviewed_book_final(final)
            self.current_view = "result"
            for b, k in self.view_buttons:
                b.setChecked(k == "result")
            self.window.statusBar().showMessage("已清空人工强制迁移蒙版，并恢复使用该工具前的稳定结果。", 6000)
            self.refresh()

        self.window.run_page_action(
            "清空强制迁移蒙版", lambda: reset_manual_force_transfer_review(page_dir,cfg), done,
            failure_title="清空强制迁移蒙版失败",
        )


    def _edit_target_layer_erase(self):
        page_dir = self._current_page_dir()
        if page_dir is None or not (page_dir / "target_original.png").exists():
            notify(self, "请先处理当前页，再使用 TARGET 日文层擦除画笔。", "warning")
            return
        import cv2
        import numpy as np
        display_path = page_dir / "final_reviewed.png"
        if not display_path.exists():
            display_path = page_dir / "final.png"
        if not display_path.exists():
            QMessageBox.information(self, "没有最终结果", "当前页还没有最终结果可供收尾。")
            return
        target = cv2.imread(str(page_dir / "target_original.png"), cv2.IMREAD_COLOR)
        if target is None:
            QMessageBox.warning(self, "无法读取", "当前页 target_original.png 无法读取。")
            return
        mask_path = page_dir / "manual_target_layer_erase_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if mask is None or mask.shape != target.shape[:2]:
            mask = np.zeros(target.shape[:2], dtype=np.uint8)
        selected_fill_mode = str(self.target_erase_mode.currentData() or "auto")
        display = cv2.imread(str(display_path), cv2.IMREAD_COLOR)
        # Chinese protection is page-stable during one editor session; building it
        # on every brush event repeatedly decoded several alpha/mask files.
        from .review_apply import _protected_chinese_mask
        protect,_protect_diag=_protected_chinese_mask(page_dir,target.shape[:2],margin_px=1)
        preview_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        def _target_erase_live_preview(mask_value):
            raw=(np.asarray(mask_value,dtype=np.uint8)>0).astype(np.uint8)*255
            if cv2.countNonZero(raw)==0: return display.copy()
            expanded=cv2.dilate(raw,preview_kernel,iterations=1)
            effective=expanded.copy(); effective[protect>0]=0
            out=display.copy()
            if cv2.countNonZero(effective)==0: return out
            if selected_fill_mode == "pure_white":
                out[effective>0]=255; return out
            # Live preview only needs the local brush neighbourhood. Inpaint a
            # padded ROI instead of the whole manga page; final save still uses
            # the authoritative full review service in a worker thread.
            pts=cv2.findNonZero(effective)
            if pts is None: return out
            x,y,w,h=cv2.boundingRect(pts); pad=12
            x0=max(0,x-pad); y0=max(0,y-pad); x1=min(target.shape[1],x+w+pad); y1=min(target.shape[0],y+h+pad)
            crop_target=target[y0:y1,x0:x1]
            crop_mask=effective[y0:y1,x0:x1]
            cleaned=cv2.inpaint(crop_target,crop_mask,3.0,cv2.INPAINT_TELEA)
            local=crop_mask>0
            out_crop=out[y0:y1,x0:x1]; out_crop[local]=cleaned[local]
            return out
        dlg = MaskEditorDialog(
            display_path, mask, self.window,
            title="只擦 TARGET 日文层 · 中文硬保护",
            hint_text=("红色 = 笔刷范围。当前模式：" + ("纯白涂抹；" if selected_fill_mode == "pure_white" else "智能恢复 TARGET；") +
                       "可刷残留日文、黑点、短线、标点和符号。画布会实时预览背景修复；正式保存时仍按中文保护层计算有效区域。"),
            save_label="保存并应用 TARGET 擦除", preview_fn=_target_erase_live_preview,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        write_image(mask_path, dlg.result_mask())
        save_json(page_dir / "target_layer_erase_settings.json", {
            "schema": "manga_hd_translation_transfer.target_layer_erase_settings.v1",
            "dilate_px": 1,
            "protect_chinese_margin_px": 1,
            "fill_mode": selected_fill_mode,
        })
        cfg=self.window.state.config.model_copy(deep=True)
        def done(final):
            final=Path(str(final)); self.window.state.last_result_path=str(final)
            pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""
            proj=self.window.state.projects_by_page.get(page_id)
            if proj is not None: proj.artifacts["final"]=str(final)
            self._sync_reviewed_book_final(final); self.current_view="target_erase"
            for b,k in self.view_buttons: b.setChecked(k==self.current_view)
            self.window.statusBar().showMessage("TARGET 层擦除已应用：只改日文母版层，中文图层已硬保护。",6000)
            self.refresh()
        self.window.run_page_action(
            "TARGET 层擦除", lambda: apply_target_layer_erase_review(page_dir,cfg), done,
            failure_title="TARGET 层擦除失败",
        )

    def _edit_target_layer_restore(self):
        page_dir = self._current_page_dir()
        if page_dir is None or not (page_dir / "target_original.png").exists():
            notify(self, "请先处理当前页，再使用 TARGET 日文层恢复画笔。", "warning")
            return
        import cv2
        import numpy as np
        display_path = page_dir / "final_reviewed.png"
        if not display_path.exists():
            display_path = page_dir / "final.png"
        if not display_path.exists():
            QMessageBox.information(self, "没有最终结果", "当前页还没有最终结果可供恢复。")
            return
        target = cv2.imread(str(page_dir / "target_original.png"), cv2.IMREAD_COLOR)
        if target is None:
            QMessageBox.warning(self, "无法读取", "当前页 target_original.png 无法读取。")
            return
        mask_path = page_dir / "manual_target_layer_restore_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if mask is None or mask.shape != target.shape[:2]:
            mask = np.zeros(target.shape[:2], dtype=np.uint8)
        display = cv2.imread(str(display_path), cv2.IMREAD_COLOR)
        def _target_restore_live_preview(mask_value):
            raw=(np.asarray(mask_value,dtype=np.uint8)>0).astype(np.uint8)*255
            if cv2.countNonZero(raw)==0: return display.copy()
            out=display.copy(); out[raw>0]=target[raw>0]; return out
        dlg = MaskEditorDialog(
            display_path, mask, self.window,
            title="恢复 TARGET 日文层",
            hint_text="红色 = 恢复范围。该工具会把笔刷区域直接恢复成 TARGET 原始日文图层与背景；同时会把恢复区域反向写回相关清除/强制蒙版，用于真正擦去蒙版、恢复日文层。左键涂抹，右键或“消除蒙版”可反擦。",
            save_label="保存并应用 TARGET 恢复", preview_fn=_target_restore_live_preview,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        write_image(mask_path, dlg.result_mask())
        save_json(page_dir / "target_layer_restore_settings.json", {
            "schema": "manga_hd_translation_transfer.target_layer_restore_settings.v1",
            "dilate_px": 0,
        })
        def done(final):
            final=Path(str(final)); self.window.state.last_result_path=str(final)
            pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""
            proj=self.window.state.projects_by_page.get(page_id)
            if proj is not None: proj.artifacts["final"]=str(final)
            self._sync_reviewed_book_final(final); self.current_view="target_restore"
            for b,k in self.view_buttons: b.setChecked(k==self.current_view)
            self.window.statusBar().showMessage("TARGET 层恢复已应用：笔刷区域已恢复原始日文图层，并同步从相关蒙版中扣除。",6000)
            self.refresh()
        self.window.run_page_action(
            "TARGET 层恢复", lambda: apply_target_layer_restore_review(page_dir, self.window.state.config.model_copy(deep=True)), done,
            failure_title="TARGET 层恢复失败",
        )


    def _reset_target_layer_erase(self):
        page_dir = self._current_page_dir()
        if page_dir is None:
            return
        def done(final):
            if final is not None:
                final=Path(str(final)); self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage("已清空 TARGET 层擦除，并恢复擦除前结果。",5000); self.refresh()
        self.window.run_page_action(
            "清空 TARGET 层擦除", lambda: reset_target_layer_erase_review(page_dir), done,
            failure_title="恢复 TARGET 层失败",
        )


    def _reset_target_layer_restore(self):
        page_dir = self._current_page_dir()
        if page_dir is None:
            return
        def done(final):
            if final is not None:
                final=Path(str(final)); self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage("已清空 TARGET 层恢复，并恢复恢复前结果。",5000); self.refresh()
        self.window.run_page_action(
            "清空 TARGET 层恢复", lambda: reset_target_layer_restore_review(page_dir, self.window.state.config.model_copy(deep=True)), done,
            failure_title="恢复 TARGET 层失败",
        )


    def _manual_effect_rows(self) -> list[dict[str, Any]]:
        page_dir=self._current_page_dir()
        if page_dir is None: return []
        path=page_dir/"review_overrides.json"
        if not path.exists(): return []
        data=normalize_overrides(load_json(path))
        return _json_dict_rows(data.get("manual_effect_regions"))

    def _manual_effect_candidates(self) -> list[dict[str, Any]]:
        ws=self._workspace()
        return as_dict_rows(getattr(ws, "manual_effect_candidates", [])) if ws is not None else []

    def _unhandled_manual_effect_candidates(self) -> list[dict[str, Any]]:
        # Only candidates that passed the stricter text-seeded actionability gate
        # are allowed to prefill a box automatically. Other Direct rejections are
        # still reported in diagnostics/status, but the user must frame them
        # manually so artwork is never mistaken for text.
        actionable=[x for x in self._manual_effect_candidates() if bool(x.get("auto_actionable", False))]
        return _manual_effect_unhandled_candidates(actionable, self._manual_effect_rows())

    def _apply_manual_effect_overrides(self, page_dir: Path, overrides: dict[str, Any]):
        save_json(page_dir/"review_overrides.json",overrides)
        rows=_json_dict_rows(overrides.get("manual_effect_regions"))
        frozen=page_dir/"manual_effect_base.png"
        if not rows:
            # Removing the final manual region must restore the exact page that
            # existed before omission repair, including every successful Chinese
            # replacement and previous QA decision.
            stable=page_dir/"final_auto.png"
            src=stable if stable.exists() else (frozen if frozen.exists() else page_dir/"final.png")
            dst=page_dir/"final_reviewed.png"
            if src.exists():
                atomic_copy_file(src,dst); final=dst
            else:
                final=page_dir/"target_original.png"
            for name in ("manual_effect_transfer_layer.png","manual_effect_transfer_mask.png","manual_effect_clear_mask.png"):
                try: (page_dir/name).unlink(missing_ok=True)
                except OSError: pass
            try: frozen.unlink(missing_ok=True)
            except OSError: pass
            try: (page_dir/"final_auto.png").unlink(missing_ok=True)
            except OSError: pass
        else:
            final=apply_review_page(page_dir,self.window.state.config.model_copy(deep=True))
            # Never silently claim success when a saved Reveal preview patch was
            # not committed. This was the user's most confusing failure mode: the
            # editor showed Chinese, Apply returned normally, but final stayed old.
            expected = {
                str(x.get("id", "")): int(x.get("reveal_patch_pixels", 0) or 0)
                for x in rows if str(x.get("reveal_patch_file", "") or "")
            }
            if any(v > 0 for v in expected.values()):
                audit_path=page_dir/"review_applied.json"
                audit=normalize_review_applied(load_json(audit_path) if audit_path.exists() else {})
                got={str(x.get("id", "")): dict(x) for x in _json_dict_rows(audit.get("manual_effect_applied"))}
                failed=[rid for rid,pix in expected.items() if pix>0 and (rid not in got or not bool(got[rid].get("success")) or not bool(got[rid].get("preview_patch_exact")))]
                if failed:
                    raise RuntimeError("人工补漏预览补丁未写入最终结果："+", ".join(failed))
        self.window.state.last_result_path=str(final)
        pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""; proj=self.window.state.projects_by_page.get(page_id)
        if proj is not None:
            proj.artifacts["final"]=str(final)
            proj.artifacts["final_reviewed"]=str(final)
        self._sync_reviewed_book_final(final)
        # Manual commits rewrite a stable path in-place. Force the workbench to
        # decode it again instead of relying solely on filesystem timestamp keys.

        try:
            self.image.clear_cache()
        except Exception:
            logger.debug("preview cache invalidation failed", exc_info=True)
        self.current_view="result"
        for b,k in self.view_buttons: b.setChecked(k=="result")
        try:
            changed_rows=len(rows)
            self.window.statusBar().showMessage(
                f"人工补漏已应用：已同步 final_reviewed.png → final.png（{changed_rows} 个区域）。",
                6000,
            )

        except Exception:
            logger.debug("manual effect status update failed", exc_info=True)

    def _start_manual_gui_flow(self, page_dir: Path, preset_candidate: dict[str, Any] | None = None) -> None:
        save_json(page_dir/"manual_gui_flow.json", {
            "schema": "manga_hd_translation_transfer.manual_gui_flow.v1",
            "version": VERSION,
            "started_at": time.time(),
            "preset_candidate": dict(preset_candidate or {}),
            "steps": [],
        })

    def _trace_manual_gui_flow(self, page_dir: Path, stage: str, payload: dict[str, Any] | None = None) -> None:
        path=page_dir/"manual_gui_flow.json"
        data=as_dict(load_json(path)) if path.exists() else {
            "schema": "manga_hd_translation_transfer.manual_gui_flow.v1", "version": VERSION, "steps": []
        }
        steps=as_dict_rows(data.get("steps"))
        steps.append({"stage":str(stage), "time":time.time(), **as_dict(payload)})
        data["steps"]=steps; data["last_stage"]=str(stage); save_json(path,data)

    def _commit_manual_effect_core(self, page_dir: Path, row: dict[str, Any], reveal, reveal_patch, preset_candidate: dict[str, Any] | None = None):
        """Qt-free portion of a manual/region commit; safe for PageActionWorker."""
        def trace(stage: str, payload: dict[str, Any]):
            self._trace_manual_gui_flow(page_dir, stage, payload)
        return commit_manual_effect(
            page_dir, row, reveal, reveal_patch,
            self.window.state.config.model_copy(deep=True),
            preset_candidate=preset_candidate,
            trace=trace,
        )

    def _finalize_manual_effect_commit(self, page_dir: Path, row: dict[str, Any], result, *, refresh_ui: bool = True) -> Path:
        """GUI-thread bookkeeping after the core transaction has completed."""
        self.window.state.last_result_path = str(result.final_reviewed)
        remembered_mode = str(row.get("mode", "") or "")
        if remembered_mode and not remembered_mode.startswith("region_"):
            self.window.state.last_manual_effect_mode = remembered_mode
            _app_settings().setValue("review/last_manual_effect_mode", remembered_mode)
        pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""; proj=self.window.state.projects_by_page.get(page_id)
        if proj is not None:
            proj.artifacts["final"] = str(result.final)
            proj.artifacts["final_reviewed"] = str(result.final_reviewed)
        self._sync_reviewed_book_final(result.final_reviewed)
        try:
            self.image.clear_cache()
        except Exception:
            logger.debug("manual effect preview cache invalidation failed", exc_info=True)
        if refresh_ui:
            self.current_view = "result"
            for b,k in self.view_buttons:
                b.setChecked(k == "result")
        self.window.statusBar().showMessage(
            f"人工补漏已直接提交 · 本页共 {result.region_count} 个区域 · final 已同步", 6000
        )
        # RegionCompositeDialog updates its own image immediately. Rebuilding the
        # whole ProjectPage after every local region would decode thumbnails,
        # recalculate controls and visibly hitch the modal editor. Defer that
        # outer-page refresh until the region dialog closes.
        if refresh_ui:
            self.refresh()
        return result.final_reviewed

    def _commit_manual_effect_dialog_result(self, page_dir: Path, row: dict[str, Any], reveal, reveal_patch, preset_candidate: dict[str, Any] | None = None) -> Path:
        """Synchronous adapter retained for legacy/manual dialogs."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result=self._commit_manual_effect_core(page_dir,row,reveal,reveal_patch,preset_candidate)
        finally:
            QApplication.restoreOverrideCursor()
        return self._finalize_manual_effect_commit(page_dir,row,result)

    def _open_region_composite_editor(self):
        if getattr(self,"_region_editor",None) is not None:
            self._editor_stack.setCurrentWidget(self._region_editor)
            return
        page_dir=self._current_page_dir(); ws=self._workspace()
        if page_dir is None or ws is None or not ws.project_path or not ws.project_path.exists():
            notify(self, "请先用任意整页模式处理当前页，再进入区域复合工具。", "warning")
            return
        source_path=page_dir/"source_original.png"; target_path=page_dir/"target_original.png"
        if not source_path.exists() or not target_path.exists():
            QMessageBox.warning(self,"缺少页面缓存","当前页缺少 SOURCE / TARGET 原图缓存，请重新处理当前页一次。")
            return
        display_path=page_dir/"final_reviewed.png"
        if not display_path.exists(): display_path=page_dir/"final.png"
        if not display_path.exists(): display_path=target_path
        try:
            project=normalize_project(load_json(ws.project_path))
        except Exception as exc:
            QMessageBox.critical(self,"无法读取页面项目",str(exc)); return
        self._start_manual_gui_flow(page_dir,{"origin":"region_composite"})
        self._trace_manual_gui_flow(page_dir,"region_composite_opened")
        def _commit(row,reveal,reveal_patch):
            self._trace_manual_gui_flow(page_dir,"region_action_commit",{"mode":str(row.get("mode") or ""),"target_bbox":list(row.get("target_bbox") or [])})
            return self._commit_manual_effect_core(page_dir,row,reveal,reveal_patch,{})
        def _finalize(row,result):
            return self._finalize_manual_effect_commit(page_dir,row,result,refresh_ui=False)
        def _trace(stage,payload):
            self._trace_manual_gui_flow(page_dir,stage,payload)
        try:
            dlg=RegionCompositeDialog(page_dir,source_path,target_path,display_path,project,self.window.state.config,self.window,commit_handler=_commit,commit_finalize_handler=_finalize,trace_handler=_trace)
        except Exception as exc:
            self._trace_manual_gui_flow(page_dir,"region_composite_failed",{"reason":str(exc)})
            QMessageBox.critical(self,"无法打开区域复合工具",str(exc)); return
        self._region_editor=dlg
        def closed(_result):
            self._trace_manual_gui_flow(page_dir,"region_composite_closed",{"applied_count":dlg.applied_count()})
            self._editor_stack.setCurrentIndex(0)
            self._editor_stack.removeWidget(dlg)
            self._region_editor=None
            dlg.deleteLater()
            self.image.clear_cache()
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.refresh()
        dlg.finished.connect(closed)
        dlg.embed_in(self._editor_stack)

    def _add_open_text_box_region(self):
        page_dir=self._current_page_dir(); ws=self._workspace()
        current_mode=str(((ws.meta or {}).get("transfer_mode") if ws is not None else "") or self.window.state.config.transfer.mode or "").strip().lower()
        if current_mode not in {"mask_replace", "hybrid"}:
            QMessageBox.information(self, "当前模式不可用", "“开放文字框选”只属于精准蒙版 / 精准蒙版+OCR。Direct 和整页挖孔保持各自的安全合同。")
            return
        self._add_manual_effect_region(
            None,
            forced_tool_kind="open_text_box",
            forced_mode="open_text_box",
            owner_transfer_mode=current_mode,
        )

    def _add_manual_effect_region(self, preset_candidate: dict[str, Any] | None = None, *, forced_tool_kind: str = "manual_effect", forced_mode: str | None = None, owner_transfer_mode: str | None = None):
        page_dir=self._current_page_dir(); ws=self._workspace()
        if page_dir is None or ws is None or not ws.project_path or not ws.project_path.exists():
            notify(self, "请先处理当前页。人工补漏复用本页已经保存的配准，不会重新跑 OCR。", "warning")
            return
        source_path=page_dir/"source_original.png"; target_path=page_dir/"target_original.png"
        if not source_path.exists() or not target_path.exists():
            QMessageBox.warning(self,"缺少页面缓存","当前页缺少 source_original.png 或 target_original.png，请重新处理当前页一次。")
            return
        project=normalize_project(load_json(ws.project_path))
        project_mode=str(as_dict(project.get("meta")).get("transfer_mode", "") or owner_transfer_mode or self.window.state.config.transfer.mode or "").strip().lower()
        manual_ops=_manual_effect_ops_for_mode(project_mode)
        preset_candidate = as_dict(preset_candidate)
        initial_bbox = as_list(preset_candidate.get("target_bbox"))
        if forced_mode:
            initial_mode = str(forced_mode)
        elif preset_candidate:
            initial_mode = str(preset_candidate.get("suggested_manual_mode", "reveal_text") or "reveal_text")
        else:
            remembered = str(getattr(self.window.state, "last_manual_effect_mode", "") or "")
            if not remembered:
                remembered = str(_app_settings().value("review/last_manual_effect_mode", "") or "")
            initial_mode = remembered if remembered in {"reveal_text", "effect_text", "white_bubble_text"} else None
        self._start_manual_gui_flow(page_dir,preset_candidate)
        self._trace_manual_gui_flow(page_dir,"manual_dialog_opened",{"initial_bbox":initial_bbox,"initial_mode":initial_mode or ""})
        def _commit(row,reveal,reveal_patch):
            return self._commit_manual_effect_dialog_result(page_dir,row,reveal,reveal_patch,preset_candidate)
        def _trace(stage,payload):
            self._trace_manual_gui_flow(page_dir,stage,payload)
        try:
            dlg=ManualEffectDialog(
                source_path,target_path,project,self.window,
                initial_bbox=initial_bbox,initial_mode=initial_mode,commit_handler=_commit,trace_handler=_trace,
                config=self.window.state.config, tool_kind=str(forced_tool_kind or "manual_effect"),
                owner_transfer_mode=str(owner_transfer_mode or project_mode), ops_module=manual_ops,
            )
        except Exception as exc:
            self._trace_manual_gui_flow(page_dir,"manual_dialog_failed",{"reason":str(exc)})
            QMessageBox.critical(self,"无法打开人工补漏",str(exc)); return
        result=dlg.exec()
        if result!=QDialog.DialogCode.Accepted:
            self._trace_manual_gui_flow(page_dir,"manual_dialog_cancelled")
            return
        if dlg.was_committed_directly():
            self._trace_manual_gui_flow(page_dir,"dialogs_closed_after_commit")
            try: self.image.clear_cache()
            except Exception: pass
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.refresh()
            return
        # Compatibility fallback for externally constructed dialogs without a
        # commit handler. The normal workbench path above always commits directly.
        self._trace_manual_gui_flow(page_dir,"legacy_dialog_handoff")
        try:
            self._commit_manual_effect_dialog_result(page_dir,dlg.result_row(),dlg.result_reveal_mask(),dlg.result_reveal_patch(),preset_candidate)
            self._trace_manual_gui_flow(page_dir,"dialogs_closed_after_commit")
        except Exception as exc:
            self._trace_manual_gui_flow(page_dir,"legacy_commit_failed",{"reason":str(exc)})
            QMessageBox.critical(self,"人工补漏失败",str(exc))

    def _add_next_manual_effect_candidate(self):
        candidates = self._unhandled_manual_effect_candidates()
        idx=self.manual_effect_candidate_target.currentIndex()
        if not candidates:
            QMessageBox.information(self, "没有候选区域", "当前页没有新的自动候选开放式文字区域。")
            return
        if idx < 0 or idx >= len(candidates):
            idx=0
        self._add_manual_effect_region(candidates[idx])

    def _undo_manual_effect_region(self):
        page_dir=self._current_page_dir()
        if page_dir is None: return
        override_path=page_dir/"review_overrides.json"; overrides=normalize_overrides(load_json(override_path) if override_path.exists() else {})
        rows=_json_dict_rows(overrides.get("manual_effect_regions"))
        if not rows:
            QMessageBox.information(self,"没有人工区域","当前页没有可撤销的人工补漏区域。")
            return
        removed=rows.pop()
        reveal_files=[]
        for key in ("reveal_mask_file","reveal_patch_file"):
            name=str(removed.get(key,"") or "").strip()
            if name: reveal_files.append(name)
        overrides["manual_effect_regions"]=rows; overrides["status"]="reviewed_with_manual_effect" if rows else "reviewed"
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            run_manual_review_transaction(page_dir,lambda: self._apply_manual_effect_overrides(page_dir,overrides))
            # Delete no-longer-referenced sparse patch files only after the page
            # state has been successfully recomposed and published.
            for name in reveal_files:
                try: (page_dir/name).unlink(missing_ok=True)
                except OSError: logger.warning("unable to remove obsolete manual patch %s",page_dir/name)
            self.window.statusBar().showMessage(f"已撤销最近人工区域 · 剩余 {len(rows)} 个",4000); self.refresh()
        except Exception as exc: QMessageBox.critical(self,"撤销失败",str(exc))
        finally: QApplication.restoreOverrideCursor()

    def _choose_sr_model(self):
        path,_=QFileDialog.getOpenFileName(self,"选择本地超分模型",str(Path.home()),"Torch models (*.pth *.pt *.ckpt *.safetensors);;All files (*)")
        if path: self.sr_model.setText(path); self._sync()
    def _run(self): self._sync(); self.window.run_current_page()

    def _reprocess_current_page(self):
        page_dir = self._current_page_dir()
        if page_dir is None or not (page_dir/"project.json").exists():
            notify(self, "请先处理当前页，再使用“重新处理当前页”。", "warning")
            return
        self._sync()
        self.window.run_current_page(reapply_review_after_process=True)

    def set_view(self,key):
        # View-only changes must not rerun JSON loading, candidate merging,
        # font catalogue updates and the full inspector layout.  Those stay in
        # refresh(), called on page/processing/review state changes.
        self.current_view=key
        for b,k in self.view_buttons: b.setChecked(k==key)
        paths=getattr(self,"_view_paths",None)
        if paths is None or not self.window.state.pairs:
            self.refresh()
            return
        current_path=paths.get(key,"")
        self.image.set_image(current_path)
        if not current_path:
            self.view_status.setText("本页尚无该视图输出")
        else:
            self.view_status.setText("当前显示："+next((b.text() for b,k in self.view_buttons if k==key),key))

    def _move_page(self, delta: int):
        self.window.set_selected_page(self.window.state.selected_index + int(delta))

    def _workspace(self):
        pair=self.window.current_pair()
        if pair is None: return None
        page_id=page_id_for_pair(pair)
        project=self.window.state.projects_by_page.get(page_id)
        return resolve_page_workspace(self.window.state.output_dir, pair, project, self.window.state.restored_page_roots.get(page_id))

    def _manual_queue(self):
        ws=self._workspace()
        return list(ws.review_regions) if ws is not None else []

    @staticmethod
    def _manual_region_key(row: dict) -> str:
        return str(row.get("target_region_id") or row.get("target_unit_id") or row.get("target_bubble_id") or "")


    def _ocr_editor_context(self):
        ws=self._workspace()
        if ws is None or ws.page_root is None or not ws.project_path or not ws.project_path.exists():
            return None
        try:
            project=normalize_project(load_json(ws.project_path))
        except Exception:
            return None
        mode=str(as_dict(project.get("meta")).get("transfer_mode") or self.window.state.config.transfer.mode or "").strip().lower()
        if not is_ocr_edit_mode(mode):
            return None
        pair=self.window.current_pair()
        if pair is None:
            return None
        source_path=ws.page_root/"source_original.png"
        target_path=ws.page_root/"target_original.png"
        if not source_path.exists(): source_path=Path(pair.source_path)
        if not target_path.exists(): target_path=Path(pair.target_path)
        return ws,project,mode,source_path,target_path

    def _open_ocr_block_editor(self):
        ctx=self._ocr_editor_context()
        if ctx is None:
            QMessageBox.information(self.window,"当前页面不可编辑","人工 OCR 需要当前页已有 project.json、SOURCE 和 TARGET。处理完成后的 Direct、精准蒙版、两种 Reveal、精准蒙版+OCR 与 OCR重排页面均可使用。")
            return
        ws,project,mode,source_path,target_path=ctx
        dialog=OCRBlockEditorDialog(ws.page_root,source_path,target_path,project,self.window.state.config,mode,parent=self.window)
        if dialog.exec()!=QDialog.DialogCode.Accepted:
            return
        cfg=self.window.state.config.model_copy(deep=True)
        def done(final):
            final=Path(str(final)); self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage("人工 OCR 文本块已保存并局部重绘。",4500); self.refresh()
        # Rebuild/replay may read several page artifacts and render text. Keep it
        # off the GUI thread so closing the OCR editor never freezes the workbench.
        self.window.run_page_action(
            "应用人工 OCR", lambda: apply_manual_ocr_review(ws.page_root,cfg), done,
            failure_title="应用人工 OCR 失败",
        )

    def _reset_ocr_blocks(self):
        ctx=self._ocr_editor_context()
        if ctx is None:
            return
        ws,project,mode,_source,_target=ctx
        rows=load_ocr_blocks(ws.page_root,mode)
        if not rows:
            self.window.statusBar().showMessage("当前页没有人工 OCR 文本块。",2500); return
        if not _confirm_destructive_action(
            self.window,
            "清空人工 OCR",
            f"将清空当前页 {len(rows)} 个人工 OCR 文本块，并恢复不含人工 OCR 的复核结果。\n\n这不会修改当前整页自动模式，也不会删除 Direct / 精准蒙版 / Reveal 等自动产物。",
            confirm_text="清空",
        ):
            return
        cfg=self.window.state.config.model_copy(deep=True)
        logger.info("manual OCR clear requested mode=%s page=%s blocks=%d", mode, ws.page_root, len(rows))
        def action():
            return clear_ocr_review_blocks(ws.page_root,cfg)
        def done(final):
            final=Path(str(final)); self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            logger.info("manual OCR clear completed mode=%s page=%s", mode, ws.page_root); self.window.statusBar().showMessage("已清空人工 OCR 文本块，并恢复无 OCR 复核结果。",4500); self.refresh()
        self.window.run_page_action(
            "清空人工 OCR", action, done, failure_title="清空人工 OCR 失败",
        )

    def _manual_selection_changed(self, *_args):
        queue=self._manual_queue(); idx=self.manual_target.currentIndex()
        if idx < 0 or idx >= len(queue):
            return
        row=dict(queue[idx]); key=self._manual_region_key(row)
        page_dir=self._current_page_dir()
        override={}
        if page_dir is not None:
            op=page_dir/"review_overrides.json"
            overrides=normalize_overrides(load_json(op) if op.exists() else {})
            for item in _json_dict_rows(overrides.get("manual_reletter")):
                if self._manual_region_key(item)==key:
                    override=dict(item); break
        current={**row, **override}
        self.manual_text.blockSignals(True)
        self.manual_text.setPlainText(str(current.get("text") or current.get("auto_text") or ""))
        self.manual_text.blockSignals(False)
        ori=str(current.get("orientation") or current.get("auto_orientation") or "auto")
        oi=self.manual_orientation.findData(ori); self.manual_orientation.setCurrentIndex(max(0,oi))
        br=str(current.get("line_break_mode") or "smart")
        bi=self.manual_break_mode.findData(br); self.manual_break_mode.setCurrentIndex(max(0,bi))
        lm=str(current.get("layout_mode") or getattr(self.window.state.config.lettering,"layout_mode","smart_scaling"))
        li=self.manual_layout_mode.findData(lm); self.manual_layout_mode.setCurrentIndex(max(0,li))
        font_value=str(current.get("font_path") or current.get("auto_font_path") or "")
        self.manual_font.setText(font_value)
        if hasattr(self,"manual_font_preset"):
            preset=font_value if font_value in {"sans","serif","rounded","comic"} else "custom"
            pi=self.manual_font_preset.findData(preset); self.manual_font_preset.setCurrentIndex(max(0,pi))
        self.manual_font_size.setValue(int(current.get("font_size") or current.get("auto_font_size") or 0))
        self.manual_columns.setValue(int(current.get("columns") or 0))
        if hasattr(self,"manual_line_spacing"):
            spacing=current.get("line_spacing_ratio")
            self.manual_line_spacing.setValue(float(spacing) if spacing is not None else -1.0)
        is_auto_reletter=str(row.get("review_kind") or "")=="reletter_auto"
        self.manual_reset.setEnabled(bool(is_auto_reletter and override))
        self.candidate_accept.setEnabled(not is_auto_reletter)
        self.candidate_restore.setEnabled(not is_auto_reletter)
        self._refresh_review_history_status()

    def _apply_manual_font_preset(self):
        value=str(self.manual_font_preset.currentData() or "custom") if hasattr(self,"manual_font_preset") else "custom"
        if value != "custom": self.manual_font.setText(value)

    def _refresh_manual_font_catalog(self, *, force: bool = False):
        if not hasattr(self, "manual_font_catalog"):
            return
        current=self.manual_font.text().strip(); self.manual_font_catalog.blockSignals(True); self.manual_font_catalog.clear(); self.manual_font_catalog.addItem("选择字体库字体", "")
        for row in discover_fonts(limit=180, force=force):
            self.manual_font_catalog.addItem(str(row.get("name") or Path(str(row.get("path") or "")).stem), str(row.get("path") or ""))
        idx=self.manual_font_catalog.findData(current) if current else -1; self.manual_font_catalog.setCurrentIndex(max(0,idx)); self.manual_font_catalog.blockSignals(False)
        self.manual_font_catalog.setToolTip(f"Folirina 持久字体库：{font_library_dir()}")

    def _apply_manual_catalog_font(self):
        if not hasattr(self, "manual_font_catalog"):
            return
        path=str(self.manual_font_catalog.currentData() or "").strip()
        if path:
            self.manual_font.setText(path)
            if hasattr(self, "manual_font_preset"):
                idx=self.manual_font_preset.findData("custom")
                if idx>=0: self.manual_font_preset.setCurrentIndex(idx)

    def _persist_manual_font_expression(self):
        raw=self.manual_font.text().strip()
        if not raw:
            return
        try:
            stored=persist_font_expression(raw, strict=True)
        except Exception as exc:
            self.window.statusBar().showMessage(f"字体不可用：{exc}", 4500); return
        if stored != raw:
            self.manual_font.setText(stored); self._refresh_manual_font_catalog()

    def _choose_manual_reletter_font(self):
        start=str(Path.home())
        current=self.manual_font.text().strip()
        if current and Path(current).exists(): start=str(Path(current).parent)
        path,_=QFileDialog.getOpenFileName(self,"导入当前 Region 字体到 Folirina 字体库",start,"Fonts (*.ttf *.ttc *.otf *.otc);;All files (*)")
        if not path:
            return
        try:
            row=import_font_to_library(path)
        except Exception as exc:
            QMessageBox.warning(self,"字体导入失败",str(exc)); return
        self.manual_font.setText(str(row.get("path") or "")); self._refresh_manual_font_catalog()

    def _refresh_review_history_status(self):
        page_dir=self._current_page_dir()
        if page_dir is None or not hasattr(self,"manual_history_status"):
            return
        undo_n,redo_n=review_history_counts(page_dir)
        self.manual_history_status.setText(f"历史 {undo_n} / 重做 {redo_n}")
        self.manual_undo.setEnabled(undo_n>0); self.manual_redo.setEnabled(redo_n>0)

    def _review_history_step(self, direction: str):
        page_dir=self._current_page_dir()
        if page_dir is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            step=apply_review_history_step(page_dir,direction,self.window.state.config.model_copy(deep=True))
            if step is None:
                self.window.statusBar().showMessage("没有可用的编辑历史。",2500); return
            final=step.final_reviewed
            self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage("已撤销人工编辑" if direction=="undo" else "已重做人工编辑",3500)
            self.refresh(); self._refresh_review_history_status()
        except Exception as exc:
            QMessageBox.critical(self,"编辑历史失败",str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _reset_manual_reletter(self):
        ws=self._workspace(); queue=self._manual_queue(); idx=self.manual_target.currentIndex()
        if ws is None or ws.page_root is None or not queue or idx < 0 or idx >= len(queue):
            return
        row=dict(queue[idx]); key=self._manual_region_key(row)
        if not key: return
        page_dir=ws.page_root; path=page_dir/"review_overrides.json"
        overrides=normalize_overrides(load_json(path) if path.exists() else {})
        entries=[x for x in _json_dict_rows(overrides.get("manual_reletter")) if self._manual_region_key(x)!=key]
        overrides["manual_reletter"]=entries
        overrides["status"]="reviewed" if not entries else "reviewed_with_manual_reletter"
        try:
            project_now=normalize_project(load_json(ws.project_path))
            overrides["owner_transfer_mode"]=str(as_dict(project_now.get("meta")).get("transfer_mode", "") or self.window.state.config.transfer.mode)
        except Exception:
            overrides["owner_transfer_mode"]=str(self.window.state.config.transfer.mode)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            tx=apply_review_overrides_transaction(page_dir,overrides,self.window.state.config.model_copy(deep=True),history_reason="reset_manual_reletter")
            final=tx.final_reviewed
            self.window.state.last_result_path=str(final); self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage(f"已恢复自动重排：{key}",4500)
            self.refresh()
        except Exception as exc: QMessageBox.critical(self,"恢复自动重排失败",str(exc))
        finally: QApplication.restoreOverrideCursor()

    def _sync_reviewed_book_final(self, final: str | Path):
        """Synchronize reviewed output through one core result-state contract."""
        final_path = Path(final)
        if final_path.exists() and final_path.name == "final_reviewed.png":
            commit_reviewed_result(final_path.parent, final_path)
        pair=self.window.current_pair()
        if pair is None or not self.window.state.output_dir:
            return
        restored = bool(self.window.state.restored_page_origin.get(page_id_for_pair(pair)))
        final_name = (page_id_for_pair(pair) if restored else Path(pair.target_path).stem) + ".png"
        dst=Path(self.window.state.output_dir)/"final"/final_name
        try:
            dst.parent.mkdir(parents=True,exist_ok=True)
            atomic_copy_file(final_path,dst)
        except OSError as exc:
            logger.warning("reviewed book-final sync failed: %s",exc)

    def _apply_manual_reletter(self):
        ws=self._workspace()
        queue=self._manual_queue()
        idx=self.manual_target.currentIndex()
        text=self.manual_text.toPlainText().strip()
        if ws is None or ws.page_root is None or not ws.project_path or not ws.project_path.exists() or not queue or idx < 0 or idx >= len(queue):
            QMessageBox.information(self,"没有待补文字","当前页没有需要人工补全的气泡。")
            return
        if not text:
            notify(self, "请先输入这个气泡的完整中文译文。", "warning")
            return
        page_dir=ws.page_root
        try:
            managed_font=persist_font_expression(self.manual_font.text().strip(), strict=True)
        except Exception as exc:
            QMessageBox.warning(self,"字体不可用",str(exc)); return
        if managed_font != self.manual_font.text().strip():
            self.manual_font.setText(managed_font); self._refresh_manual_font_catalog()
        row=dict(queue[idx])
        override_path=page_dir/"review_overrides.json"
        overrides=normalize_overrides(load_json(override_path) if override_path.exists() else {})
        entries=_json_dict_rows(overrides.get("manual_reletter"))
        region_key=self._manual_region_key(row)
        target_id=str(row.get("target_bubble_id", ""))
        item={
            "review_kind": str(row.get("review_kind") or ""),
            "target_region_id": str(row.get("target_region_id") or ""),
            "target_unit_id": str(row.get("target_unit_id") or ""),
            "target_bubble_id": target_id,
            "target_bbox": as_list(row.get("target_bbox")),
            "text": text,
            "orientation": self.manual_orientation.currentData() or "auto",
            "line_break_mode": self.manual_break_mode.currentData() or "smart",
            "layout_mode": self.manual_layout_mode.currentData() or "smart_scaling",
            "font_path": managed_font,
            "font_size": int(self.manual_font_size.value()),
            "columns": int(self.manual_columns.value()),
            "line_spacing_ratio": None if float(self.manual_line_spacing.value()) < 0 else float(self.manual_line_spacing.value()),
            "reason": row.get("reason", ""),
        }
        entries=[x for x in entries if self._manual_region_key(x) != region_key]
        entries.append(item)
        overrides["manual_reletter"]=entries
        # Bubble-candidate decisions belong only to Direct/Mask review. Successful
        # reletter regions are edited by immutable region id and never co-opt those
        # route-specific restore/accept lists.
        if str(row.get("review_kind") or "") != "reletter_auto":
            overrides["restore_target_bubbles"]=[x for x in list(overrides.get("restore_target_bubbles",[]) or []) if str(x)!=target_id]
            overrides["accept_candidate_targets"]=[x for x in list(overrides.get("accept_candidate_targets",[]) or []) if str(x)!=target_id]
        overrides["status"]="reviewed_with_manual_reletter"
        try:
            project_now=normalize_project(load_json(ws.project_path))
            overrides["owner_transfer_mode"]=str(as_dict(project_now.get("meta")).get("transfer_mode", "") or self.window.state.config.transfer.mode)
        except Exception:
            overrides["owner_transfer_mode"]=str(self.window.state.config.transfer.mode)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            tx=apply_review_overrides_transaction(page_dir,overrides,self.window.state.config.model_copy(deep=True),history_reason="apply_manual_reletter")
            final=tx.final_reviewed
            self.window.state.last_result_path=str(final)
            pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""
            p=self.window.state.projects_by_page.get(page_id)
            if p is not None: p.artifacts["final"]=str(final)
            self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage(f"已更新重排文字：{region_key or target_id}",5000)
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self,"补字失败",str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _set_candidate_decision(self, action: str):
        ws=self._workspace(); queue=self._manual_queue(); idx=self.manual_target.currentIndex()
        if ws is None or ws.page_root is None or not ws.project_path or not ws.project_path.exists() or not queue or idx < 0 or idx >= len(queue):
            QMessageBox.information(self,"没有待复核中文","当前页没有低置信中文候选。")
            return
        page_dir=ws.page_root; row=dict(queue[idx]); tid=str(row.get("target_bubble_id", ""))
        override_path=page_dir/"review_overrides.json"; overrides=normalize_overrides(load_json(override_path) if override_path.exists() else {})
        accepted=set(map(str,overrides.get("accept_candidate_targets",[]) or [])); restored=set(map(str,overrides.get("restore_target_bubbles",[]) or []))
        entries=[x for x in _json_dict_rows(overrides.get("manual_reletter")) if str(x.get("target_bubble_id", "")) != tid]
        if action=="accept": accepted.add(tid); restored.discard(tid); overrides["status"]="reviewed_candidate_accepted"
        else: restored.add(tid); accepted.discard(tid); overrides["status"]="reviewed_candidate_restored"
        overrides["accept_candidate_targets"]=sorted(accepted); overrides["restore_target_bubbles"]=sorted(restored); overrides["manual_reletter"]=entries
        try:
            project_now=normalize_project(load_json(ws.project_path))
            overrides["owner_transfer_mode"]=str(as_dict(project_now.get("meta")).get("transfer_mode", "") or self.window.state.config.transfer.mode)
        except Exception:
            overrides["owner_transfer_mode"]=str(self.window.state.config.transfer.mode)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            tx=apply_review_overrides_transaction(page_dir,overrides,self.window.state.config.model_copy(deep=True),history_reason=f"candidate_{action}")
            final=tx.final_reviewed; self.window.state.last_result_path=str(final)
            pair=self.window.current_pair(); page_id=page_id_for_pair(pair) if pair is not None else ""; p=self.window.state.projects_by_page.get(page_id)
            if p is not None: p.artifacts["final"]=str(final)
            self._sync_reviewed_book_final(final)
            self.current_view="result"
            for b,k in self.view_buttons: b.setChecked(k=="result")
            self.window.statusBar().showMessage(("已接受中文候选：" if action=="accept" else "已还原日文：")+tid,5000); self.refresh()
        except Exception as exc: QMessageBox.critical(self,"复核操作失败",str(exc))
        finally: QApplication.restoreOverrideCursor()



__all__ = ["WorkbenchActionsMixin"]
