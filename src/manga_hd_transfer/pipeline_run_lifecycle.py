from __future__ import annotations

from pathlib import Path
from typing import Callable
import inspect

import cv2
import numpy as np

from .config import PipelineConfig
from .cache import RESUME_SCHEMA
from .io_utils import load_json, save_json
from .mode_contracts import clear_stale_mode_outputs
from .models import PagePair, PageProject, QAItem
from .qa import qa_summary
from .result_state import (
    create_run_snapshot, discard_run_snapshot, invalidate_manual_review_state,
    restore_run_snapshot, run_snapshot_has_existing,
)
from .run_trace import PageRunTrace
from .run_receipt import write_run_receipt
from .workspace_guard import PageRunGuard, cleanup_orphan_temp_files
from .workspace_integrity import validate_page_workspace


def _selected_strategy(project: PageProject) -> str:
    meta = project.meta or {}
    selected = str(meta.get("selected_strategy") or meta.get("strategy") or "")
    if not selected:
        selected = str((meta.get("planner") or {}).get("strategy") or "")
    if not selected:
        selected = str((meta.get("transfer_planner") or {}).get("strategy") or "")
    if not selected:
        selected = str((meta.get("mode_isolation") or {}).get("selected_strategy") or "")
    return selected


def _mode_effect_summary(project: PageProject, mode: str, selected_strategy: str) -> dict[str, object]:
    """Return mode-neutral publication counts for single-page diagnostics.

    Older lifecycle logging accidentally read only ``meta.reletter`` and thus
    reported ``successful_regions=0`` for Direct/Mask/Hybrid even when those
    modes had published pixels.  Keep diagnostics mode-aware without coupling
    any renderer implementation into the lifecycle.
    """
    meta = project.meta or {}
    keys = [str(mode or "").strip().lower(), str(selected_strategy or "").strip().lower()]
    owner: dict = {}
    for key in keys:
        probe = meta.get(key)
        if isinstance(probe, dict):
            owner = probe
            break
    applied = None
    for key in ("applied_count", "successful_regions", "accepted_count", "region_count"):
        if key in owner:
            try:
                applied = int(owner.get(key) or 0)
                break
            except (TypeError, ValueError):
                pass
    records = owner.get("records") if isinstance(owner, dict) else None
    if applied is None and isinstance(records, list):
        applied = sum(1 for row in records if isinstance(row, dict) and bool(row.get("applied", True)))
    if applied is None:
        applied = 0
    failed = 0
    for key in ("failed_regions", "failed_count"):
        if key in owner:
            try:
                failed = int(owner.get(key) or 0)
            except (TypeError, ValueError):
                failed = 0
            break
    return {
        "mode": str(mode or ""),
        "selected_strategy": str(selected_strategy or ""),
        "applied_regions": int(applied),
        "failed_regions": int(failed),
    }


def _published_result_effect(
    root: Path,
    snapshot: Path | None,
    project: PageProject,
    *,
    mode: str,
    selected_strategy: str,
) -> dict[str, object]:
    """Describe whether a real rerun changed the visible automatic page.

    Shared registration/layout caches are intentional evidence caches. Renderer
    outputs are cleared before each fresh page run.  On a same-page rerun or a
    mode switch, compare the transaction snapshot with the newly published
    ``final.png`` so the GUI can distinguish "new mode ran but produced the same
    pixels" from "old result was reused".
    """
    effect = _mode_effect_summary(project, mode, selected_strategy)
    effect.update({
        "previous_result_available": False,
        "previous_mode": "",
        "current_mode": str(mode or ""),
        "mode_changed": False,
        "pixel_comparison_available": False,
        "pixel_identical": False,
        "changed_pixels": None,
    })
    if snapshot is None:
        return effect
    old_project_path = snapshot / "project.json"
    old_final_path = snapshot / "final.png"
    new_final_path = root / "final.png"
    old_mode = ""
    if old_project_path.exists():
        try:
            old_obj = load_json(old_project_path)
            if isinstance(old_obj, dict):
                old_mode = str(((old_obj.get("meta") or {}).get("transfer_mode") or "")).strip().lower()
        except Exception:
            old_mode = ""
    effect["previous_mode"] = old_mode
    effect["mode_changed"] = bool(old_mode and old_mode != str(mode or "").strip().lower())
    effect["previous_result_available"] = bool(old_final_path.exists())
    if not old_final_path.exists() or not new_final_path.exists():
        return effect
    old = cv2.imread(str(old_final_path), cv2.IMREAD_COLOR)
    new = cv2.imread(str(new_final_path), cv2.IMREAD_COLOR)
    if old is None or new is None:
        return effect
    effect["pixel_comparison_available"] = True
    if old.shape != new.shape:
        effect["pixel_identical"] = False
        effect["changed_pixels"] = int(max(old.shape[0] * old.shape[1], new.shape[0] * new.shape[1]))
        effect["shape_changed"] = True
        effect["previous_shape"] = list(old.shape[:2])
        effect["current_shape"] = list(new.shape[:2])
        return effect
    changed = int(np.count_nonzero(np.any(old != new, axis=2)))
    effect["changed_pixels"] = changed
    effect["pixel_identical"] = bool(changed == 0)
    effect["shape_changed"] = False
    return effect


def run_page_lifecycle(
    *,
    config: PipelineConfig,
    pair: PagePair,
    page_root: str | Path,
    final_path: str | Path | None,
    page_mark,
    cancel_cb,
    progress_cb=None,
    prefetched_images=None,
    process_impl: Callable[..., PageProject],
    get_trace: Callable[[], object | None],
    set_trace: Callable[[object | None], None],
    version: str,
) -> PageProject:
    """Single-writer, transaction-safe wrapper around the pure page orchestration."""
    root = Path(page_root)
    mode = config.transfer.mode.lower().strip()
    root.mkdir(parents=True, exist_ok=True)
    trace = PageRunTrace(root, mode=mode)
    previous_trace = get_trace()
    set_trace(trace)
    snapshot = None
    try:
        # The lock must precede the snapshot.  Otherwise a second request can
        # copy a page while the active writer is midway through publication and
        # can also leak a pointless full-page backup before PageRunBusyError.
        with PageRunGuard(root, mode):
            orphan_cleanup = cleanup_orphan_temp_files(root)
            trace.event("workspace_guard_acquired", orphan_temp_cleanup=orphan_cleanup)
            snapshot = create_run_snapshot(root, trace.run_id)
            trace.event(
                "run_start",
                selected_mode=mode, version=version,
                source=str(pair.source_path), target=str(pair.target_path),
                ocr_source=str(getattr(config.ocr, "source_backend", None) or getattr(config.ocr, "backend", "")),
                ocr_target=str(getattr(config.ocr, "target_backend", None) or getattr(config.ocr, "backend", "")),
                registration_backend=str(getattr(config.registration, "backend", "")),
                previous_result_snapshot=run_snapshot_has_existing(snapshot),
            )
            try:
                impl_kwargs = {"page_mark": page_mark, "cancel_cb": cancel_cb}
                # Preserve compatibility with plugins/subclasses written before
                # v2.0.94, whose overridden _process_page_impl may not yet
                # accept the optional progress callback.  Inspect the bound
                # callable instead of catching TypeError, so genuine TypeErrors
                # raised *inside* the implementation still propagate normally.
                try:
                    params = inspect.signature(process_impl).parameters.values()
                    accepts_progress = any(p.name == "progress_cb" for p in params) or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in params
                    )
                except (TypeError, ValueError):
                    accepts_progress = True
                if accepts_progress:
                    impl_kwargs["progress_cb"] = progress_cb
                try:
                    params2 = inspect.signature(process_impl).parameters.values()
                    accepts_prefetched = any(p.name == "prefetched_images" for p in params2) or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in params2
                    )
                except (TypeError, ValueError):
                    accepts_prefetched = True
                if accepts_prefetched and prefetched_images is not None:
                    impl_kwargs["prefetched_images"] = prefetched_images
                project = process_impl(pair, root, final_path, **impl_kwargs)
            except Exception as exc:
                # Restore the previous published page instead of leaving the
                # stale/half-cleared automatic result visible after a failed
                # Paddle/OCR/model run.
                try:
                    clear_stale_mode_outputs(root)
                    invalidate_manual_review_state(root)
                except Exception:
                    pass
                restored = restore_run_snapshot(root, snapshot)
                if restored.get("success"):
                    discard_run_snapshot(snapshot)
                    snapshot = None
                trace.exception("run_failed", exc, previous_result_restore=restored)
                try:
                    save_json(root / "last_run_state.json", {
                        "schema": "manga_hd_translation_transfer.run_state.v2",
                        "status": "failed",
                        "mode": mode,
                        "run_id": trace.run_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "previous_result_restored": restored,
                        "recovery_backup_retained": bool(snapshot),
                        "run_log": str(trace.text_path),
                        "run_trace": str(trace.jsonl_path),
                        "orphan_temp_cleanup": orphan_cleanup,
                    })
                except Exception:
                    # Diagnostics must never mask the processing exception that
                    # explains why the page failed. The append-only run trace
                    # has already received the failure when writable.
                    pass
                raise

            try:
                # The transaction does not end when the pixel renderer returns.
                # Receipt/integrity/project publication can also fail (disk full,
                # permission loss, malformed plugin metadata). Keep the previous
                # coherent page snapshot until every common commit step succeeds.
                selected_strategy = _selected_strategy(project)
                publish_effect = _published_result_effect(
                    root, snapshot, project, mode=mode, selected_strategy=selected_strategy,
                )
                project.meta["publish_effect"] = publish_effect
                if bool(publish_effect.get("mode_changed")):
                    project.meta["mode_transition"] = {
                        "from": str(publish_effect.get("previous_mode") or ""),
                        "to": str(mode),
                        "pixel_comparison_available": bool(publish_effect.get("pixel_comparison_available")),
                        "pixel_identical": bool(publish_effect.get("pixel_identical")),
                        "changed_pixels": publish_effect.get("changed_pixels"),
                        "applied_regions": int(publish_effect.get("applied_regions") or 0),
                    }
                project.meta["resume_contract"] = RESUME_SCHEMA
                project.meta["run_receipt"] = write_run_receipt(
                    root, project, requested_mode=mode, selected_strategy=selected_strategy, run_id=trace.run_id,
                )
                integrity = validate_page_workspace(root, project, mode, selected_strategy=selected_strategy)
                project.meta["workspace_integrity"] = integrity
                if not integrity.get("pass"):
                    existing = {q.code for q in project.qa}
                    if "workspace_integrity_failed" not in existing:
                        project.qa.append(QAItem(
                            "workspace_integrity_failed", "error",
                            "Page workspace failed post-run integrity validation.",
                            meta={"issues": list(integrity.get("issues") or []), "mode": mode},
                        ))
                project.meta["qa_summary"] = qa_summary(project.qa)
                save_json(root / "qa.json", {"summary": qa_summary(project.qa), "issues": [x.to_dict() for x in project.qa]})
                save_json(root / "project.json", project.to_dict())
                trace.event(
                    "run_success", selected_strategy=selected_strategy,
                    applied_regions=int(publish_effect.get("applied_regions") or 0),
                    failed_regions=int(publish_effect.get("failed_regions") or 0),
                    previous_mode=str(publish_effect.get("previous_mode") or ""),
                    mode_changed=bool(publish_effect.get("mode_changed")),
                    pixel_identical=bool(publish_effect.get("pixel_identical")) if publish_effect.get("pixel_comparison_available") else None,
                    changed_pixels=publish_effect.get("changed_pixels"),
                    qa_pass=bool((project.meta.get("qa_summary") or {}).get("pass", False)),
                    workspace_integrity=bool(integrity.get("pass")),
                )
                save_json(root / "last_run_state.json", {
                    "schema": "manga_hd_translation_transfer.run_state.v2",
                    "status": "success" if integrity.get("pass") else "integrity_failed",
                    "mode": mode,
                    "run_id": trace.run_id,
                    "selected_strategy": selected_strategy,
                    "publish_effect": publish_effect,
                    "workspace_integrity": integrity,
                    "run_log": str(trace.text_path),
                    "run_trace": str(trace.jsonl_path),
                    "orphan_temp_cleanup": orphan_cleanup,
                })
            except Exception as exc:
                try:
                    clear_stale_mode_outputs(root)
                    invalidate_manual_review_state(root)
                except Exception:
                    pass
                restored = restore_run_snapshot(root, snapshot)
                if restored.get("success"):
                    discard_run_snapshot(snapshot)
                    snapshot = None
                trace.exception("run_commit_failed", exc, previous_result_restore=restored)
                try:
                    save_json(root / "last_run_state.json", {
                        "schema": "manga_hd_translation_transfer.run_state.v2",
                        "status": "failed", "stage": "commit", "mode": mode,
                        "run_id": trace.run_id, "error_type": type(exc).__name__, "error": str(exc),
                        "previous_result_restored": restored,
                        "recovery_backup_retained": bool(snapshot),
                        "run_log": str(trace.text_path), "run_trace": str(trace.jsonl_path),
                        "orphan_temp_cleanup": orphan_cleanup,
                    })
                except Exception:
                    pass
                raise
            discard_run_snapshot(snapshot)
            snapshot = None
            return project
    finally:
        set_trace(previous_trace)
