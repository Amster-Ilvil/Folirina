from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .config import PipelineConfig
from .io_utils import load_json, save_json, write_image
from .result_state import ensure_manual_baseline, commit_reviewed_result
from .review_apply import apply_review_page
from .schema_compat import as_dict, as_dict_rows, as_list, normalize_overrides, normalize_review_applied

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
    override_path = page_dir / "review_overrides.json"
    overrides = normalize_overrides(load_json(override_path) if override_path.exists() else {})
    row = as_dict(row)
    if not row.get("id"):
        raise ValueError("manual effect row is missing id")
    rid = str(row["id"])
    mode = str(row.get("mode", ""))
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
        ok, encoded = cv2.imencode(".png", reveal_patch)
        if not ok:
            raise ValueError("无法保存人工补漏预览补丁")
        encoded.tofile(page_dir / patch_name)
        row["reveal_patch_file"] = patch_name
        row["reveal_patch_pixels"] = int(cv2.countNonZero(reveal_patch[:, :, 3]))
        if mode == "reveal_text" and int(row["reveal_patch_pixels"]) <= 0:
            raise ValueError("Reveal 补丁为空，拒绝提交。")

    rows = [dict(x) for x in as_dict_rows(overrides.get("manual_effect_regions")) if str(x.get("id", "")) != rid]
    rows.append(row)
    overrides["manual_effect_regions"] = rows
    overrides["status"] = "reviewed_with_manual_effect"
    save_json(override_path, overrides)
    _trace(trace, "overrides_saved", row_id=rid, region_count=len(rows), reveal_patch_pixels=int(row.get("reveal_patch_pixels", 0) or 0))

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
    if not bool(rec.get("success")):
        raise RuntimeError("人工补漏区域没有进入 review_applied.json 的成功记录。")

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
    return ManualCommitResult(
        row_id=rid,
        final_reviewed=final_reviewed,
        final=final_local,
        region_count=len(rows),
        reveal_patch_pixels=int(row.get("reveal_patch_pixels", 0) or 0),
        preview_patch_exact=preview_exact,
    )
