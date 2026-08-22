from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import os
import shutil
import tempfile

import cv2
import numpy as np

from .config import PipelineConfig
from .io_utils import load_json, save_json, write_image
from .result_state import ensure_manual_baseline, commit_reviewed_result
from .review_apply import apply_review_page
from .review_history import undo_review_state, redo_review_state, record_review_state
from .schema_compat import as_dict, as_dict_rows, as_list, normalize_overrides, normalize_review_applied
from .review_artifacts import safe_page_artifact_path

TraceFn = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class ManualCommitResult:
    row_id: str
    final_reviewed: Path
    final: Path
    region_count: int
    reveal_patch_pixels: int
    preview_patch_exact: bool


def _trace(cb: TraceFn | None, stage: str, **payload: Any) -> None:
    if cb is not None:
        cb(stage, dict(payload))


_REGION_SEMANTIC_KEYS = (
    "mode", "target_bbox", "selection_spec", "source_offset_x", "source_offset_y",
    "feather_px", "inset_px", "diff_threshold", "edge_threshold", "expand_px",
    "auto_clear_target", "render_text", "ocr_text", "target_ocr_polygons",
    "orientation", "font_path", "font_size", "columns", "line_break_mode",
    "layout_mode", "owner_transfer_mode",
)


def _same_region_action_semantics(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Detect an exact duplicate region action while ignoring its generated id.

    Brush reveal rows are deliberately excluded because two strokes can share a
    bbox while carrying different saved patch pixels.
    """
    ma = str(a.get("mode", "") or "").strip().lower()
    mb = str(b.get("mode", "") or "").strip().lower()
    if ma != mb or not ma.startswith("region_") or ma == "region_brush_reveal":
        return False
    def stable(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(repr(row.get(key, None)) for key in _REGION_SEMANTIC_KEYS)
    return stable(a) == stable(b)


# Files that one manual-review commit is allowed to mutate.  Keep this list
# review-layer-only: automatic renderer products and original inputs are never
# rolled back here.  A failed local edit should be indistinguishable from no
# edit at all, including project/review synchronization metadata.
_MANUAL_COMMIT_MUTABLE_NAMES = (
    "review_overrides.json", "project.json", "final.png", "final_reviewed.png",
    "final_auto.png", "manual_effect_base.png", "review_sync.json",
    "review_applied.json", "review_base.png", "review_preview.png",
    "text_layer_reviewed.png", "inpainted_reviewed.png",
    "direct_patch_layer_reviewed.png", "mask_transfer_layer_reviewed.png",
    "hybrid_transfer_layer_reviewed.png", "hybrid_text_layer_reviewed.png",
    "reletter_text_layer_reviewed.png", "manual_effect_transfer_layer.png",
    "manual_effect_transfer_mask.png", "manual_effect_clear_mask.png",
    "editable_reviewed.ora", "editable_reviewed.psd",
    "target_layer_erase_base.png", "target_layer_erase_effective_mask.png",
    "target_layer_erase_chinese_protect_mask.png", "target_layer_erase_preview.png",
    "target_layer_erase.json", "target_layer_restore_base.png",
    "target_layer_restore_effective_mask.png", "target_layer_restore_preview.png",
    "target_layer_restore.json",
)


class _ManualCommitSnapshot:
    """Filesystem transaction guard for one explicit manual edit.

    Review rendering touches several page-local artifacts.  Previously an
    exception after ``review_overrides.json`` was saved could leave the new row
    persisted while ``final.png`` still represented an earlier state.  Snapshot
    only the known mutable review artifacts and restore them atomically on any
    failure.  This deliberately avoids copying immutable SOURCE/TARGET and mode
    renderer products.
    """

    def __init__(self, page_dir: Path, extra_names: tuple[str, ...] = ()) -> None:
        self.page_dir = page_dir
        self.names = tuple(dict.fromkeys((*_MANUAL_COMMIT_MUTABLE_NAMES, *extra_names)))
        self._tmp = tempfile.TemporaryDirectory(prefix="folirina-manual-commit-")
        self.backup = Path(self._tmp.name)
        self.existed: set[str] = set()
        for name in self.names:
            src = page_dir / name
            if src.is_file():
                self.existed.add(name)
                dst = self.backup / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    @staticmethod
    def _restore_file(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".rollback", dir=str(dst.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        finally:
            tmp.unlink(missing_ok=True)

    def rollback(self) -> None:
        for name in self.names:
            dst = self.page_dir / name
            if name in self.existed:
                self._restore_file(self.backup / name, dst)
            else:
                try:
                    dst.unlink(missing_ok=True)
                except OSError:
                    pass

    def close(self) -> None:
        self._tmp.cleanup()



@dataclass(frozen=True, slots=True)
class ReviewHistoryStepResult:
    direction: str
    state: dict[str, Any]
    final_reviewed: Path
    final: Path


def apply_review_history_step(
    page_dir: str | Path,
    direction: str,
    config: PipelineConfig,
) -> ReviewHistoryStepResult | None:
    """Apply one undo/redo step as a page-local filesystem transaction.

    ``undo_review_state`` changes both review JSON files before the compositor
    runs. If rendering then fails, leaving those JSONs advanced would make the
    visible pixels and history disagree. Snapshot all mutable review artifacts,
    perform the history step, render, and roll everything back on any failure.
    """
    page_dir = Path(page_dir)
    key = str(direction or "").strip().lower()
    if key not in {"undo", "redo"}:
        raise ValueError(f"unsupported review history direction: {direction}")
    snapshot = _ManualCommitSnapshot(page_dir, ("review_history.json",))
    try:
        state = undo_review_state(page_dir) if key == "undo" else redo_review_state(page_dir)
        if state is None:
            return None
        final_reviewed = Path(apply_review_page(page_dir, config))
        final = page_dir / "final.png"
        if not final_reviewed.exists() or not final.exists():
            raise RuntimeError("编辑历史已更新，但复核结果没有完整发布。")
        return ReviewHistoryStepResult(
            direction=key, state=dict(state), final_reviewed=final_reviewed, final=final,
        )
    except Exception:
        snapshot.rollback()
        raise
    finally:
        snapshot.close()


@dataclass(frozen=True, slots=True)
class ReviewOverrideApplyResult:
    final_reviewed: Path
    final: Path
    history_reason: str


def apply_review_overrides_transaction(
    page_dir: str | Path,
    overrides: dict[str, Any],
    config: PipelineConfig,
    *,
    history_reason: str = "",
) -> ReviewOverrideApplyResult:
    """Persist one explicit review-state edit and render it transactionally."""
    page_dir = Path(page_dir)
    snapshot = _ManualCommitSnapshot(page_dir, ("review_history.json",))
    try:
        override_path = page_dir / "review_overrides.json"
        if history_reason:
            current = normalize_overrides(load_json(override_path) if override_path.exists() else {})
            record_review_state(page_dir, current, history_reason)
        save_json(override_path, normalize_overrides(overrides))
        final_reviewed = Path(apply_review_page(page_dir, config))
        final = page_dir / "final.png"
        if not final_reviewed.exists() or not final.exists():
            raise RuntimeError("复核状态已保存，但最终结果没有完整发布。")
        return ReviewOverrideApplyResult(
            final_reviewed=final_reviewed, final=final, history_reason=str(history_reason or ""),
        )
    except Exception:
        snapshot.rollback()
        raise
    finally:
        snapshot.close()


def run_manual_review_transaction(
    page_dir: str | Path,
    operation: Callable[[], Any],
    *,
    extra_names: tuple[str, ...] = (),
) -> Any:
    """Run an arbitrary page-local review mutation with filesystem rollback."""
    snapshot = _ManualCommitSnapshot(Path(page_dir), extra_names)
    try:
        return operation()
    except Exception:
        snapshot.rollback()
        raise
    finally:
        snapshot.close()

def commit_manual_effect(
    page_dir: str | Path,
    row: dict[str, Any],
    reveal: np.ndarray | None,
    reveal_patch: np.ndarray | None,
    config: PipelineConfig,
    *,
    preset_candidate: dict[str, Any] | None = None,
    trace: TraceFn | None = None,
) -> ManualCommitResult:
    """Persist, apply and verify one manual effect transaction.

    This module is intentionally Qt-free.  GUI code only gathers user input and
    invokes this transaction; all file/schema/review semantics live here.
    """
    page_dir = Path(page_dir)
    row = as_dict(row)
    if not row.get("id"):
        raise ValueError("manual effect row is missing id")
    rid = str(row["id"])
    if safe_page_artifact_path(page_dir, f"{rid}_reveal_mask.png") is None:
        raise ValueError("manual effect row id contains an unsafe page-artifact path")
    mode = str(row.get("mode", ""))
    reveal_names: list[str] = []
    if reveal is not None:
        reveal_names.append(f"{rid}_reveal_mask.png")
    if reveal_patch is not None:
        reveal_names.append(f"{rid}_reveal_patch.png")
    snapshot = _ManualCommitSnapshot(page_dir, tuple(reveal_names))
    try:
        override_path = page_dir / "review_overrides.json"
        overrides = normalize_overrides(load_json(override_path) if override_path.exists() else {})
        preset = as_dict(preset_candidate)
        if preset:
            row["candidate_reason"] = str(preset.get("reason", ""))
            row["source_candidate_bbox"] = as_list(preset.get("source_bbox"))
            row["suggested_manual_mode"] = str(preset.get("suggested_manual_mode", "reveal_text") or "reveal_text")

        ensure_manual_baseline(page_dir)
        _trace(trace, "commit_files_started", row_id=rid, mode=mode)

        if reveal is not None:
            mask_name = f"{rid}_reveal_mask.png"
            write_image(page_dir / mask_name, reveal)
            row["reveal_mask_file"] = mask_name
        if reveal_patch is not None:
            if reveal_patch.ndim != 3 or reveal_patch.shape[2] != 4:
                raise ValueError("Reveal patch must be BGRA")
            patch_name = f"{rid}_reveal_patch.png"
            write_image(page_dir / patch_name, reveal_patch)
            row["reveal_patch_file"] = patch_name
            row["reveal_patch_pixels"] = int(cv2.countNonZero(reveal_patch[:, :, 3]))
            if mode == "reveal_text" and int(row["reveal_patch_pixels"]) <= 0:
                raise ValueError("Reveal 补丁为空，拒绝提交。")

        rows = []
        duplicate_region_ids: list[str] = []
        for value in as_dict_rows(overrides.get("manual_effect_regions")):
            existing = dict(value)
            if str(existing.get("id", "")) == rid:
                continue
            if _same_region_action_semantics(existing, row):
                duplicate_region_ids.append(str(existing.get("id", "")))
                continue
            rows.append(existing)
        rows.append(row)
        overrides["manual_effect_regions"] = rows
        owner_mode = str(row.get("owner_transfer_mode", "") or "").strip().lower()
        if owner_mode:
            overrides["owner_transfer_mode"] = owner_mode
        overrides["status"] = "reviewed_with_manual_effect"
        save_json(override_path, overrides)
        _trace(
            trace, "overrides_saved", row_id=rid, region_count=len(rows),
            reveal_patch_pixels=int(row.get("reveal_patch_pixels", 0) or 0),
            deduplicated_region_ids=duplicate_region_ids,
        )

        final_reviewed = Path(apply_review_page(page_dir, config))
        _trace(trace, "review_applied", row_id=rid)

        final_local = page_dir / "final.png"
        reviewed = cv2.imread(str(final_reviewed), cv2.IMREAD_COLOR)
        local = cv2.imread(str(final_local), cv2.IMREAD_COLOR) if final_local.exists() else None
        # All current review routes commit through result_state.  Keep one defensive
        # compatibility fallback for a third-party/old route that returns reviewed
        # pixels but has not mirrored them yet; do not rewrite the file twice during
        # the normal path.
        if reviewed is not None and (local is None or local.shape != reviewed.shape or not np.array_equal(reviewed, local)):
            commit_reviewed_result(page_dir, final_reviewed)
            local = cv2.imread(str(final_local), cv2.IMREAD_COLOR)
        if reviewed is None or local is None or reviewed.shape != local.shape or not np.array_equal(reviewed, local):
            raise RuntimeError("人工补漏已生成 reviewed 结果，但 final.png 未同步为同一像素。")

        audit_path = page_dir / "review_applied.json"
        audit = normalize_review_applied(load_json(audit_path) if audit_path.exists() else {})
        got = {str(x.get("id", "")): dict(x) for x in as_dict_rows(audit.get("manual_effect_applied"))}
        rec = as_dict(got.get(rid))
        if not rec:
            raise RuntimeError("人工补漏区域没有进入 review_applied.json；提交结果与复核审计脱节。")
        if not bool(rec.get("success")):
            reason = str(rec.get("reason") or "region_action_no_effective_authority")
            # This transaction is about to roll back review_overrides.json,
            # review_applied.json and published rasters.  Do not tell the user the
            # failed row "has entered" the audit as if it will remain persisted.
            raise RuntimeError(f"人工补漏执行记录为失败，提交已自动回滚：{reason}")

        preview_exact = bool(rec.get("preview_patch_exact", False))
        if mode == "reveal_text":
            if not preview_exact:
                raise RuntimeError("Reveal 预览补丁与最终结果不一致。")
            patch_file = page_dir / str(row.get("reveal_patch_file", ""))
            patch = cv2.imread(str(patch_file), cv2.IMREAD_UNCHANGED)
            if patch is None or patch.ndim != 3 or patch.shape[2] != 4:
                raise RuntimeError("已保存的 Reveal patch 无法重新读取。")
            sel = patch[:, :, 3] > 0
            if not np.any(sel) or not np.array_equal(reviewed[sel], patch[:, :, :3][sel]):
                raise RuntimeError("最终结果没有逐像素包含 Reveal 编辑器中保存的中文补丁。")

        _trace(trace, "final_verified", row_id=rid, final_reviewed=str(final_reviewed), final=str(final_local), preview_patch_exact=preview_exact)
        result = ManualCommitResult(
            row_id=rid,
            final_reviewed=final_reviewed,
            final=final_local,
            region_count=len(rows),
            reveal_patch_pixels=int(row.get("reveal_patch_pixels", 0) or 0),
            preview_patch_exact=preview_exact,
        )
    except Exception:
        try:
            snapshot.rollback()
            _trace(trace, "commit_rolled_back", row_id=rid, mode=mode)
        except Exception as rollback_exc:
            _trace(trace, "commit_rollback_failed", row_id=rid, mode=mode, error=str(rollback_exc))
            raise RuntimeError(f"人工补漏提交失败，且回滚未能完整恢复：{rollback_exc}") from rollback_exc
        raise
    finally:
        snapshot.close()
    return result

