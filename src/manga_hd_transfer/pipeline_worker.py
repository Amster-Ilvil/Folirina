from __future__ import annotations

"""Long-running book/page processing worker.

Separated from widgets so GUI pages do not own pipeline execution semantics.
"""

import shutil
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from .config import PipelineConfig
from .models import PagePair
from .page_management import PageMark
from .pipeline import TransferPipeline, PipelineCancelled
from .workspace import page_id_for_pair
from .page_review_state import has_reapplicable_review_state, clear_reprocess_generated_artifacts
from .review_apply import apply_review_page
from .schema_compat import as_dict
from .io_utils import load_json

class PipelineWorker(QThread):
    done = Signal(object, str)
    failed = Signal(str)
    progress = Signal(int, int, str, str, bool)
    cancelled = Signal()

    def __init__(
        self, *, config: PipelineConfig, pair: PagePair | None = None,
        source_dir: str = "", target_dir: str = "", output_dir: str = "",
        page_mark: dict[str, Any] | PageMark | None = None,
        pairs_override: list[PagePair] | None = None,
        page_marks: dict[str, dict[str, Any]] | None = None,
        reapply_review_after_process: bool = False,
    ):
        super().__init__()
        self.config = config
        self.pair = pair
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.output_dir = output_dir
        self.page_mark = page_mark
        self.pairs_override = list(pairs_override) if pairs_override is not None else None
        self.page_marks = dict(page_marks) if page_marks is not None else None
        self.reapply_review_after_process = bool(reapply_review_after_process)
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def _cancelled(self):
        return self._cancel_requested

    def _progress(self, done, total, pair, status, cache_hit=False, message=""):
        label = Path(pair.target_path).name if pair is not None else ""
        text = message or status
        self.progress.emit(int(done), int(total), label, text, bool(cache_hit))


    def _page_progress(self, percent: int, stage: str, message: str):
        label = Path(self.pair.target_path).name if self.pair is not None else ""
        text = str(message or stage or "正在处理")
        self.progress.emit(int(percent), 100, label, text, False)

    def run(self):
        try:
            pipeline = TransferPipeline(self.config)
            out = Path(self.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            if self.pair is not None:
                page_root = out / "pages" / page_id_for_pair(self.pair)
                restored = any(isinstance(r, str) and r.startswith("restored_page_id:") for r in (self.pair.reasons or []))
                final_name = (page_id_for_pair(self.pair) if restored else Path(self.pair.target_path).stem) + ".png"
                final_path = out / "final" / final_name
                final_path.parent.mkdir(parents=True, exist_ok=True)
                should_reapply_review = bool(self.reapply_review_after_process and has_reapplicable_review_state(page_root, str(self.config.transfer.mode or "")))
                if should_reapply_review:
                    clear_reprocess_generated_artifacts(page_root)
                project = pipeline.process_page(
                    self.pair, page_root, final_path,
                    page_mark=self.page_mark, cancel_cb=self._cancelled,
                    progress_cb=self._page_progress,
                )
                if self._cancelled():
                    self.cancelled.emit()
                    return
                emitted_path = str(final_path)
                if should_reapply_review and (page_root / "project.json").exists():
                    reviewed = apply_review_page(page_root, self.config.model_copy(deep=True))
                    emitted_path = str(reviewed)
                    if final_path.exists() or Path(reviewed).exists():
                        try:
                            final_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(str(reviewed), str(final_path))
                            emitted_path = str(final_path)
                        except OSError:
                            emitted_path = str(reviewed)
                    project.artifacts["final"] = str(page_root / "final.png")
                    if Path(reviewed).exists():
                        project.artifacts["final_reviewed"] = str(reviewed)
                    stable = page_root / "final_auto.png"
                    if stable.exists():
                        project.artifacts["final_auto"] = str(stable)
                    sync_path = page_root / "review_sync.json"
                    if sync_path.exists():
                        project.meta["review_sync"] = as_dict(load_json(sync_path))
                    project.meta["reapplied_review_after_process"] = True
                self.done.emit(project, emitted_path)
            else:
                project = pipeline.run_book(
                    self.source_dir, self.target_dir, self.output_dir,
                    progress_cb=self._progress, cancel_cb=self._cancelled,
                    resume=self.config.batch.resume,
                    pairs_override=self.pairs_override, page_marks=self.page_marks,
                )
                if as_dict(getattr(project, "meta", {})).get("cancelled"):
                    self.cancelled.emit()
                self.done.emit(project, self.output_dir)
        except PipelineCancelled:
            self.cancelled.emit()
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

__all__ = ["PipelineWorker"]
