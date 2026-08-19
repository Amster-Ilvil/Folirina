from __future__ import annotations

from pathlib import Path
from typing import Callable
import inspect

from .config import PipelineConfig
from .io_utils import save_json
from .mode_contracts import clear_stale_mode_outputs
from .models import PagePair, PageProject, QAItem
from .qa import qa_summary
from .result_state import (
    create_run_snapshot, discard_run_snapshot, invalidate_manual_review_state,
    restore_run_snapshot,
)
from .run_trace import PageRunTrace
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


def run_page_lifecycle(
    *,
    config: PipelineConfig,
    pair: PagePair,
    page_root: str | Path,
    final_path: str | Path | None,
    page_mark,
    cancel_cb,
    progress_cb=None,
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
    snapshot = create_run_snapshot(root, trace.run_id)
    trace.event(
        "run_start",
        selected_mode=mode, version=version,
        source=str(pair.source_path), target=str(pair.target_path),
        ocr_source=str(getattr(config.ocr, "source_backend", None) or getattr(config.ocr, "backend", "")),
        ocr_target=str(getattr(config.ocr, "target_backend", None) or getattr(config.ocr, "backend", "")),
        registration_backend=str(getattr(config.registration, "backend", "")),
        previous_result_snapshot=bool(snapshot),
    )
    try:
        with PageRunGuard(root, mode):
            orphan_cleanup = cleanup_orphan_temp_files(root)
            trace.event("workspace_guard_acquired", orphan_temp_cleanup=orphan_cleanup)
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
                trace.exception("run_failed", exc, previous_result_restore=restored)
                save_json(root / "last_run_state.json", {
                    "schema": "manga_hd_translation_transfer.run_state.v2",
                    "status": "failed",
                    "mode": mode,
                    "run_id": trace.run_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "previous_result_restored": restored,
                    "run_log": str(trace.text_path),
                    "run_trace": str(trace.jsonl_path),
                    "orphan_temp_cleanup": orphan_cleanup,
                })
                raise

            selected_strategy = _selected_strategy(project)
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
            reletter_meta = (project.meta or {}).get("reletter") or {}
            trace.event(
                "run_success", selected_strategy=selected_strategy,
                successful_regions=int(reletter_meta.get("successful_regions") or 0),
                failed_regions=int(reletter_meta.get("failed_regions") or 0),
                qa_pass=bool((project.meta.get("qa_summary") or {}).get("pass", False)),
                workspace_integrity=bool(integrity.get("pass")),
            )
            save_json(root / "last_run_state.json", {
                "schema": "manga_hd_translation_transfer.run_state.v2",
                "status": "success" if integrity.get("pass") else "integrity_failed",
                "mode": mode,
                "run_id": trace.run_id,
                "selected_strategy": selected_strategy,
                "workspace_integrity": integrity,
                "run_log": str(trace.text_path),
                "run_trace": str(trace.jsonl_path),
                "orphan_temp_cleanup": orphan_cleanup,
            })
            discard_run_snapshot(snapshot)
            return project
    finally:
        set_trace(previous_trace)
