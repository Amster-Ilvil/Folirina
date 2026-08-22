from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .cache import PageStageCache
from .models import BookProject, BubbleInstance, PagePair, PageProject, QAItem, RegistrationResult, TextBlock, TextUnit, UnitMatch
from .ocr import OCRBackend
from .page_management import PageMark
from .modes.reletter.executor import ReletterExecutor as ReletterModeExecutor
from .modes.reletter.regions import detect_target_text_regions as detect_reletter_target_text_regions
from .modes.hybrid.executor import ReletterExecutor as HybridReletterExecutor
from .modes.hybrid.regions import detect_target_text_regions as detect_hybrid_target_text_regions
from .runtime import configure_runtime
from .pipeline_passthrough import emit_passthrough_page
from .pipeline_run_lifecycle import run_page_lifecycle
from .pipeline_page_flow import run_page_flow
from .pipeline_page_prep import prefetch_page_images
from .pipeline_ocr_service import (
    build_ocr_backend_soft, recognize_cached, recognize_source_rectified_cached,
)
from .pipeline_bubble_service import detect_bubbles, bubbles_cached
from .pipeline_match_service import accepted_matches
from .book_orchestration import run_book_orchestration
from .version import __version__
from .modes.runtime_binding import bind_mode_runtime_config


class PipelineCancelled(RuntimeError):
    """Cooperative cancellation raised only at safe stage boundaries."""


def _check_cancel(cancel_cb, stage: str = "") -> None:
    if cancel_cb is not None and cancel_cb():
        raise PipelineCancelled(stage or "cancelled")



class TransferPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        source_ocr: OCRBackend | None = None,
        target_ocr: OCRBackend | None = None,
    ) -> None:
        source_config = config or PipelineConfig()
        self.config = bind_mode_runtime_config(source_config)
        self._source_ocr = source_ocr
        self._target_ocr = target_ocr
        self._ocr_soft_failures: list[str] = []
        configure_runtime(self.config.runtime)
        # Keep all optional PyTorch backends on the same explicit device policy.
        if self.config.registration.device == "auto":
            self.config.registration.device = self.config.runtime.device
        if self.config.bubbles.device == "auto":
            self.config.bubbles.device = self.config.runtime.device
        if self.config.mask_replace.sr_device == "auto":
            self.config.mask_replace.sr_device = self.config.runtime.device

    def _build_ocr_backend_soft(self, lang: str, backend_name: str, *, role: str) -> OCRBackend:
        return build_ocr_backend_soft(
            self.config.ocr, lang, backend_name, role=role, soft_failures=self._ocr_soft_failures
        )

    @property
    def source_ocr(self) -> OCRBackend:
        if self._source_ocr is None:
            backend_name = self.config.ocr.source_backend or self.config.ocr.backend
            self._source_ocr = self._build_ocr_backend_soft(
                self.config.ocr.source_lang, backend_name, role="source"
            )
        return self._source_ocr

    @property
    def target_ocr(self) -> OCRBackend:
        if self._target_ocr is None:
            backend_name = self.config.ocr.target_backend or self.config.ocr.backend
            self._target_ocr = self._build_ocr_backend_soft(self.config.ocr.target_lang, backend_name, role="target")
        return self._target_ocr

    def _passthrough_page(
        self,
        pair: PagePair,
        page_root: str | Path,
        final_path: str | Path | None,
        mark: PageMark,
        *,
        source: np.ndarray | None = None,
        target: np.ndarray | None = None,
        registration: RegistrationResult | None = None,
        passthrough_reason: str = "page_manager_exclusion",
        extra_meta: dict | None = None,
        qa: list[QAItem] | None = None,
    ) -> PageProject:
        """Compatibility delegate to passthrough persistence service."""
        return emit_passthrough_page(
            self.config, pair, page_root, final_path, mark, source=source, target=target,
            registration=registration, passthrough_reason=passthrough_reason,
            extra_meta=extra_meta, qa=qa,
        )


    def _emit_aligned_overlay_page(
        self, pair, page_root, final_path, mark, *, source, target, registration, pair_check,
        result, requested_mode, planner_decision, cache_stats,
    ):
        from .pipeline_reveal_persistence import emit_aligned_overlay_page
        return emit_aligned_overlay_page(
            self.config, pair, page_root, final_path, mark, source=source, target=target,
            registration=registration, pair_check=pair_check, result=result,
            requested_mode=requested_mode, planner_decision=planner_decision, cache_stats=cache_stats,
        )

    def _emit_transparent_bubble_page(
        self, pair, page_root, final_path, mark, *, source, target, registration, pair_check,
        result, planner_decision, cache_stats,
    ):
        from .pipeline_reveal_persistence import emit_transparent_bubble_page
        return emit_transparent_bubble_page(
            self.config, pair, page_root, final_path, mark, source=source, target=target,
            registration=registration, pair_check=pair_check, result=result,
            planner_decision=planner_decision, cache_stats=cache_stats,
        )

    def _bubbles(self, image: np.ndarray, blocks, image_path: str | Path) -> list[BubbleInstance]:
        """Compatibility delegate to the isolated bubble service."""
        return detect_bubbles(image, blocks, image_path, self.config.bubbles)

    def _recognize_cached(self, role: str, backend: OCRBackend, image: np.ndarray, image_path: str | Path, cache: PageStageCache, stats: dict) -> list:
        """Compatibility delegate to the isolated OCR cache service."""
        return recognize_cached(
            role, backend, image, image_path, ocr_config=self.config.ocr, cache=cache,
            cache_enabled=bool(self.config.cache.ocr), stats=stats,
        )

    def _recognize_source_rectified_cached(
        self, backend: OCRBackend, source: np.ndarray, source_path: str | Path,
        target_shape: tuple[int, int], registration, cache: PageStageCache, stats: dict,
    ) -> list:
        """Compatibility delegate to the isolated OCR rectification service."""
        return recognize_source_rectified_cached(
            backend, source, source_path, target_shape, registration,
            ocr_config=self.config.ocr, cache=cache,
            cache_enabled=bool(self.config.cache.ocr), stats=stats,
        )

    def _reletter_executor(self, cancel_cb=None):
        mode = str(getattr(self.config.transfer, "mode", "reletter") or "reletter").strip().lower()
        if mode == "hybrid":
            executor_cls = HybridReletterExecutor
            detect_regions_fn = detect_hybrid_target_text_regions
            flow_cells = bool(getattr(self.config.hybrid.lettering, "koharu_flow_cells_enabled", False))
        else:
            executor_cls = ReletterModeExecutor
            detect_regions_fn = detect_reletter_target_text_regions
            flow_cells = bool(getattr(self.config.reletter.lettering, "koharu_flow_cells_enabled", False))
        detector = (
            (lambda image, bubble: detect_regions_fn(image, bubble, koharu_flow_cells=True))
            if flow_cells else detect_regions_fn
        )
        return executor_cls(
            self.config,
            trace=getattr(self, "_run_trace", None),
            cancel_check=lambda stage: _check_cancel(cancel_cb, stage),
            detect_regions=detector,
        )

    def _recognize_target_driven_reletter_regions(
        self, backend: OCRBackend, source: np.ndarray, target: np.ndarray,
        source_path: str | Path,
        source_bubbles: list[BubbleInstance], target_bubbles: list[BubbleInstance],
        stats: dict, cancel_cb=None,
    ) -> tuple[list[TextBlock], list[TextBlock], list[BubbleInstance], list[BubbleInstance], dict]:
        """Compatibility delegate to the isolated Reletter OCR executor."""
        return self._reletter_executor(cancel_cb).recognize_target_driven_regions(
            backend, source, target, source_path, source_bubbles, target_bubbles, stats
        )

    def _recognize_paired_regions_text_only(
        self, backend: OCRBackend, source: np.ndarray, source_path: str | Path,
        source_bubbles: list[BubbleInstance], target_bubbles: list[BubbleInstance],
        cache: PageStageCache, stats: dict, cancel_cb=None,
    ) -> tuple[list[TextBlock], list[TextBlock], list[BubbleInstance], list[BubbleInstance]]:
        """Compatibility delegate for transcript-only paired-region OCR."""
        return self._reletter_executor(cancel_cb).recognize_paired_regions_text_only(
            backend, source, source_path, source_bubbles, target_bubbles, cache, stats
        )

    def _bubbles_cached(self, role: str, image: np.ndarray, blocks, image_path: str | Path, cache: PageStageCache, stats: dict) -> list[BubbleInstance]:
        """Compatibility delegate to the isolated bubble cache service."""
        return bubbles_cached(
            role, image, blocks, image_path, bubble_config=self.config.bubbles, cache=cache,
            cache_enabled=bool(self.config.cache.bubbles), stats=stats,
        )

    def _accepted_matches(
        self, pair: PagePair, registration_confidence: float, source_units: list[TextUnit],
        target_units: list[TextUnit], matches: list[UnitMatch],
    ) -> list[UnitMatch]:
        """Compatibility delegate to the isolated text-match policy."""
        return accepted_matches(
            pair, registration_confidence, source_units, target_units, matches, config=self.config
        )

    def process_page(
        self,
        pair: PagePair,
        page_root: str | Path,
        final_path: str | Path | None = None,
        *,
        page_mark: PageMark | dict | None = None,
        cancel_cb=None,
        progress_cb=None,
        prefetched_images=None,
    ) -> PageProject:
        """Single-page entry point for all transfer modes.

        Direct is a first-class Folirina mode and uses the same transaction-safe
        page lifecycle as Mask/OCR/Reletter/Reveal.  Its renderer and strict
        semantic support guard remain Direct-owned, but there is no external
        vendor tree, subprocess, alternate PYTHONPATH, or legacy runtime bridge.
        """
        return run_page_lifecycle(
            config=self.config, pair=pair, page_root=page_root, final_path=final_path,
            page_mark=page_mark, cancel_cb=cancel_cb, progress_cb=progress_cb, prefetched_images=prefetched_images,
            process_impl=self._process_page_impl,
            get_trace=lambda: getattr(self, "_run_trace", None),
            set_trace=lambda value: setattr(self, "_run_trace", value),
            version=__version__,
        )

    def _process_page_impl(
        self,
        pair: PagePair,
        page_root: str | Path,
        final_path: str | Path | None = None,
        *,
        page_mark: PageMark | dict | None = None,
        cancel_cb=None,
        progress_cb=None,
        prefetched_images=None,
    ) -> PageProject:
        """Compatibility entry for the isolated single-page flow orchestrator."""
        return run_page_flow(
            config=self.config, pair=pair, page_root=page_root, final_path=final_path,
            page_mark=page_mark, cancel_cb=cancel_cb, progress_cb=progress_cb, prefetched_images=prefetched_images,
            trace=getattr(self, "_run_trace", None),
            check_cancel=_check_cancel, passthrough_page=self._passthrough_page,
            get_source_backend=lambda: self.source_ocr,
            get_target_backend=lambda: self.target_ocr,
            get_reletter_executor=lambda: self._reletter_executor(cancel_cb),
        )

    def run_book(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
        output_dir: str | Path,
        *,
        progress_cb=None,
        cancel_cb=None,
        resume: bool | None = None,
        pairs_override: list[PagePair] | None = None,
        page_marks: dict | None = None,
    ) -> BookProject:
        return run_book_orchestration(
            config=self.config, process_page=self.process_page,
            cancelled_exception=PipelineCancelled, source_dir=source_dir,
            target_dir=target_dir, output_dir=output_dir, progress_cb=progress_cb,
            cancel_cb=cancel_cb, resume=resume, pairs_override=pairs_override,
            page_marks=page_marks, prefetch_page_images=prefetch_page_images,
        )
