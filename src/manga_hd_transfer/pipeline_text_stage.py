from __future__ import annotations

"""OCR + bubble orchestration for one page.

This is the application-stage boundary between page geometry and downstream
matching/rendering.  It intentionally does not import ``pipeline.py`` or any
transfer renderer.  The caller supplies lazy backend/executor factories so Direct
and OCR-skipping routes remain side-effect free.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .bubbles import assign_blocks_to_bubbles
from .cache import PageStageCache
from .detector_policy import (
    koharu_is_primary, primary_detector, detector_strategy,
    STRATEGY_PRIMARY_ONLY,
)
from .layout_evidence import collect_koharu_layout_evidence_cached
from .models import BubbleInstance, TextBlock
from .ocr import OCRBackend
from .pipeline_bubble_service import bubbles_cached, primary_bubbles_cached
from .pipeline_ocr_service import (
    recognize_cached,
    recognize_source_rectified_cached,
    should_rectify_source,
)


def _bubble_mask_overlap(a: BubbleInstance, b: BubbleInstance) -> float:
    am = getattr(a, "mask", None); bm = getattr(b, "mask", None)
    if not isinstance(am, np.ndarray) or not isinstance(bm, np.ndarray) or am.shape != bm.shape:
        return 0.0
    aa = am > 0; bb = bm > 0
    denom = max(1, int(np.count_nonzero(aa)))
    return float(np.count_nonzero(aa & bb) / denom)


def _layout_has_uncovered_semantic_bubbles(
    layout_bubbles: list[BubbleInstance],
    paired_bubbles: list[BubbleInstance],
    *,
    min_confidence: float = 0.70,
    min_overlap: float = 0.45,
) -> int:
    """Count strong primary-layout containers not represented by Paired Diff."""
    missing = 0
    for row in layout_bubbles:
        if float(getattr(row, "confidence", 0.0)) < float(min_confidence):
            continue
        if not any(_bubble_mask_overlap(row, old) >= float(min_overlap) for old in paired_bubbles):
            missing += 1
    return missing


def _detector_only_blocks_and_bubbles(
    role: str,
    image: np.ndarray,
    image_path: str | Path,
    *,
    config: Any,
    cache: PageStageCache,
    stats: dict[str, str],
) -> tuple[list[TextBlock], list[BubbleInstance]]:
    """Return geometry from the selected Detector Policy without hidden Koharu.

    Koharu primary can provide text/SFX and bubble instances. MangaLens/RT-DETR
    primaries provide bubble geometry only; OCR-capable modes still obtain text
    content from the selected OCR backend later. Auxiliaries are only consulted
    through ``bubbles_cached`` according to the configured strategy.
    """
    if koharu_is_primary(config.bubbles):
        evidence = collect_koharu_layout_evidence_cached(
            image, config.bubbles, role=f'text_stage_{role}', image_path=image_path,
            cache=cache, cache_enabled=bool(config.cache.bubbles) and bool(getattr(config.bubbles, 'koharu_layout_cache_enabled', True)),
            stats=stats, allow_missing=True,
        )
        if not evidence.available:
            if detector_strategy(config.bubbles) != STRATEGY_PRIMARY_ONLY:
                # Primary failure is UNKNOWN, not a semantic veto.  Conditional
                # auxiliaries are allowed to recover geometry on this side only;
                # this is especially important for large photographed SOURCE
                # pages where the primary runtime can fail while TARGET succeeds.
                stats[f"{role}_primary_detector_unavailable"] = str(
                    (evidence.diagnostics or {}).get("reason") or "unavailable"
                )
                bubbles = bubbles_cached(
                    role, image, [], image_path, bubble_config=config.bubbles,
                    cache=cache, cache_enabled=bool(config.cache.bubbles), stats=stats,
                )
                return [], bubbles
            return [], []
        blocks = evidence.text_blocks(
            include_sfx=True, backend_name='koharu_layout',
            source_only=(role == 'source'), target_only=(role == 'target'),
        )
        bubbles = evidence.bubble_instances(
            backend_name='koharu_layout', source_only=(role == 'source'), target_only=(role == 'target'),
        )
        assign_blocks_to_bubbles(blocks, bubbles)
        return blocks, bubbles

    # The selected non-Koharu primary was globally prefetched before mode logic.
    # Reuse it and let the policy adapter invoke auxiliaries only if this route
    # actually needs them. With zero OCR blocks, a non-empty primary is already
    # sufficient; a completely empty primary can trigger conditional fallback.
    primary_bubbles_cached(
        role, image, image_path, bubble_config=config.bubbles, cache=cache,
        cache_enabled=bool(config.cache.bubbles), stats=stats,
    )
    bubbles = bubbles_cached(
        role, image, [], image_path, bubble_config=config.bubbles, cache=cache,
        cache_enabled=bool(config.cache.bubbles), stats=stats,
    )
    return [], bubbles


def _layout_only_blocks_and_bubbles(*args, **kwargs):
    """Compatibility alias; now dispatches through the selected Detector Policy."""
    return _detector_only_blocks_and_bubbles(*args, **kwargs)


@dataclass
class TextStageResult:
    source_backend: OCRBackend | None
    source_blocks: list[TextBlock]
    target_blocks: list[TextBlock]
    source_bubbles: list[BubbleInstance]
    target_bubbles: list[BubbleInstance]
    target_driven_reletter_regions: bool = False
    target_driven_reletter_diagnostics: dict = field(default_factory=dict)


def _full_page_ocr_and_bubbles(
    *,
    config: Any,
    source_backend: OCRBackend,
    target_backend: OCRBackend,
    source: np.ndarray,
    target: np.ndarray,
    source_path: str | Path,
    target_path: str | Path,
    registration: Any,
    cache: PageStageCache,
    stats: dict[str, str],
):
    if should_rectify_source(config.ocr, source_backend, registration):
        source_blocks = recognize_source_rectified_cached(
            source_backend,
            source,
            source_path,
            target.shape[:2],
            registration,
            ocr_config=config.ocr,
            cache=cache,
            cache_enabled=bool(config.cache.ocr),
            stats=stats,
        )
    else:
        source_blocks = recognize_cached(
            "source",
            source_backend,
            source,
            source_path,
            ocr_config=config.ocr,
            cache=cache,
            cache_enabled=bool(config.cache.ocr),
            stats=stats,
        )
    target_blocks = recognize_cached(
        "target",
        target_backend,
        target,
        target_path,
        ocr_config=config.ocr,
        cache=cache,
        cache_enabled=bool(config.cache.ocr),
        stats=stats,
    )
    source_bubbles = bubbles_cached(
        "source",
        source,
        source_blocks,
        source_path,
        bubble_config=config.bubbles,
        cache=cache,
        cache_enabled=bool(config.cache.bubbles),
        stats=stats,
    )
    target_bubbles = bubbles_cached(
        "target",
        target,
        target_blocks,
        target_path,
        bubble_config=config.bubbles,
        cache=cache,
        cache_enabled=bool(config.cache.bubbles),
        stats=stats,
    )
    return source_blocks, target_blocks, source_bubbles, target_bubbles


def run_text_stage(
    *,
    config: Any,
    mode: str,
    direct_container_fast: bool,
    direct_container_plan: Any | None,
    use_paired_diff: bool,
    paired_diff: Any | None,
    source: np.ndarray,
    target: np.ndarray,
    source_path: str | Path,
    target_path: str | Path,
    registration: Any,
    cache: PageStageCache,
    stats: dict[str, str],
    get_source_backend: Callable[[], OCRBackend],
    get_target_backend: Callable[[], OCRBackend],
    get_reletter_executor: Callable[[], Any],
) -> TextStageResult:
    source_backend: OCRBackend | None = None
    target_driven = False
    diagnostics: dict = {}

    if direct_container_fast and direct_container_plan is not None:
        stats["ocr_source"] = "skipped_source_direct_container"
        stats["ocr_target"] = "skipped_source_direct_container"
        stats["bubbles_source"] = "source_direct_container"
        stats["bubbles_target"] = "source_direct_container"
        return TextStageResult(
            None,
            [],
            [],
            direct_container_plan.source_bubbles,
            direct_container_plan.target_bubbles,
        )

    # v2.0.83 mode hard-isolation: explicit Precise Mask is a visual/pixel
    # transfer route.  It must never instantiate an OCR backend, even when
    # paired-diff is uncertain.  Use whatever OCR-free geometry is available;
    # the transfer stage can still run unseeded white-container completion and
    # other visual fallbacks.  Missing regions remain reviewable rather than
    # silently crossing into OCR.
    if mode == "mask_replace":
        stats["ocr_source"] = "skipped_explicit_mask_contract"
        stats["ocr_target"] = "skipped_explicit_mask_contract"
        source_layout_blocks, source_layout_bubbles = _layout_only_blocks_and_bubbles(
            "source", source, source_path, config=config, cache=cache, stats=stats
        )
        target_layout_blocks, target_layout_bubbles = _layout_only_blocks_and_bubbles(
            "target", target, target_path, config=config, cache=cache, stats=stats
        )
        layout_present = bool(source_layout_blocks or source_layout_bubbles or target_layout_blocks or target_layout_bubbles)
        if use_paired_diff and paired_diff is not None:
            stats["bubbles_source"] = f"paired_diff_with_{primary_detector(config.bubbles)}" if layout_present else "paired_diff_visual_only"
            stats["bubbles_target"] = f"paired_diff_with_{primary_detector(config.bubbles)}" if layout_present else "paired_diff_visual_only"
            return TextStageResult(
                None,
                source_layout_blocks,
                target_layout_blocks,
                paired_diff.source_bubbles,
                paired_diff.target_bubbles,
            )
        stats["bubbles_source"] = f"{primary_detector(config.bubbles)}_visual_only" if layout_present else "visual_completion_only"
        stats["bubbles_target"] = f"{primary_detector(config.bubbles)}_visual_only" if layout_present else "visual_completion_only"
        return TextStageResult(
            None, source_layout_blocks, target_layout_blocks,
            source_layout_bubbles, target_layout_bubbles,
        )

    if (
        mode != "reletter"
        and use_paired_diff
        and paired_diff is not None
        and paired_diff.safe_to_skip_ocr
        and config.mask_replace.paired_diff_skip_ocr
    ):
        source_layout_blocks, source_layout_bubbles = _layout_only_blocks_and_bubbles(
            "source", source, source_path, config=config, cache=cache, stats=stats
        )
        target_layout_blocks, target_layout_bubbles = _layout_only_blocks_and_bubbles(
            "target", target, target_path, config=config, cache=cache, stats=stats
        )
        semantic_gaps = _layout_has_uncovered_semantic_bubbles(
            target_layout_bubbles,
            list(getattr(paired_diff, "target_bubbles", []) or []),
            min_confidence=float(getattr(config.mask_replace, "koharu_semantic_bubble_min_confidence", 0.70)),
        )
        if semantic_gaps:
            # Paired Diff can be internally complete for its own rigid regions
            # while still omitting a strong Koharu open/coloured container.
            # OCR-capable modes must not skip their content fallback in that
            # situation.  Explicit Mask is handled by the zero-OCR semantic
            # registered-ink completion in the transfer stage above.
            stats["ocr_skip_cancelled_semantic_gaps"] = str(semantic_gaps)
        else:
            stats["ocr_source"] = "skipped_paired_diff"
            stats["ocr_target"] = "skipped_paired_diff"
            layout_present = bool(source_layout_blocks or target_layout_blocks)
            stats["bubbles_source"] = f"paired_diff_with_{primary_detector(config.bubbles)}" if layout_present else "paired_diff"
            stats["bubbles_target"] = f"paired_diff_with_{primary_detector(config.bubbles)}" if layout_present else "paired_diff"
            return TextStageResult(
                None,
                source_layout_blocks,
                target_layout_blocks,
                paired_diff.source_bubbles,
                paired_diff.target_bubbles,
            )

    # The selected primary detector is the preferred geometry provider for OCR-capable
    # modes as well. OCR remains responsible only for text content.
    source_layout_blocks, source_layout_bubbles = _layout_only_blocks_and_bubbles(
        "source", source, source_path, config=config, cache=cache, stats=stats
    )
    target_layout_blocks, target_layout_bubbles = _layout_only_blocks_and_bubbles(
        "target", target, target_path, config=config, cache=cache, stats=stats
    )

    source_backend = get_source_backend()
    can_crop_reletter_ocr = bool(
        getattr(source_backend, "supports_crop_recognition", True)
        or getattr(source_backend, "supports_region_query", False)
    )
    paired_geometry_available = bool(
        use_paired_diff
        and paired_diff is not None
        and paired_diff.source_bubbles
        and paired_diff.target_bubbles
    )
    layout_geometry_available = bool(source_layout_bubbles and target_layout_bubbles)

    if mode == "reletter" and (layout_geometry_available or paired_geometry_available) and can_crop_reletter_ocr:
        executor = get_reletter_executor()
        preferred_source_bubbles = source_layout_bubbles if layout_geometry_available else paired_diff.source_bubbles
        preferred_target_bubbles = target_layout_bubbles if layout_geometry_available else paired_diff.target_bubbles
        geometry_route = primary_detector(config.bubbles) if layout_geometry_available else "paired_diff"
        stats["reletter_geometry"] = geometry_route
        source_blocks, target_blocks, source_bubbles, target_bubbles, diagnostics = (
            executor.recognize_target_driven_regions(
                source_backend,
                source,
                target,
                source_path,
                preferred_source_bubbles,
                preferred_target_bubbles,
                stats,
            )
        )
        diagnostics = {**dict(diagnostics or {}), "preferred_geometry": geometry_route}
        target_driven = bool(source_blocks and target_blocks)
        if target_driven:
            return TextStageResult(
                source_backend,
                source_blocks,
                target_blocks,
                source_bubbles,
                target_bubbles,
                True,
                diagnostics,
            )
        if geometry_route != "paired_diff" and paired_geometry_available:
            # Koharu is preferred, not mandatory proof. A rare layout miss must
            # not regress a page that paired-diff already bound safely.
            source_blocks, target_blocks, source_bubbles, target_bubbles, retry_diag = (
                executor.recognize_target_driven_regions(
                    source_backend, source, target, source_path,
                    paired_diff.source_bubbles, paired_diff.target_bubbles, stats,
                )
            )
            diagnostics = {
                **dict(diagnostics or {}),
                "primary_detector_fallback": "paired_diff",
                "paired_diff_retry": dict(retry_diag or {}),
            }
            target_driven = bool(source_blocks and target_blocks)
            if target_driven:
                stats["reletter_geometry"] = f"{geometry_route}->paired_diff"
                return TextStageResult(
                    source_backend, source_blocks, target_blocks, source_bubbles,
                    target_bubbles, True, diagnostics,
                )
        # Conservative compatibility fallback for unusually faint/synthetic pages.
        if bool(getattr(source_backend, "region_text_only", False)) and paired_geometry_available:
            source_blocks, target_blocks, source_bubbles, target_bubbles = (
                executor.recognize_paired_regions_text_only(
                    source_backend,
                    source,
                    source_path,
                    paired_diff.source_bubbles,
                    paired_diff.target_bubbles,
                    cache,
                    stats,
                )
            )
            return TextStageResult(
                source_backend,
                source_blocks,
                target_blocks,
                source_bubbles,
                target_bubbles,
                False,
                diagnostics,
            )

        source_blocks, target_blocks, source_bubbles, target_bubbles = _full_page_ocr_and_bubbles(
            config=config,
            source_backend=source_backend,
            target_backend=get_target_backend(),
            source=source,
            target=target,
            source_path=source_path,
            target_path=target_path,
            registration=registration,
            cache=cache,
            stats=stats,
        )
        return TextStageResult(
            source_backend,
            source_blocks,
            target_blocks,
            source_bubbles,
            target_bubbles,
            False,
            diagnostics,
        )

    if paired_geometry_available and bool(getattr(source_backend, "region_text_only", False)):
        executor = get_reletter_executor()
        source_blocks, target_blocks, source_bubbles, target_bubbles = (
            executor.recognize_paired_regions_text_only(
                source_backend,
                source,
                source_path,
                paired_diff.source_bubbles,
                paired_diff.target_bubbles,
                cache,
                stats,
            )
        )
        return TextStageResult(
            source_backend,
            source_blocks,
            target_blocks,
            source_bubbles,
            target_bubbles,
            False,
            diagnostics,
        )

    source_blocks, target_blocks, source_bubbles, target_bubbles = _full_page_ocr_and_bubbles(
        config=config,
        source_backend=source_backend,
        target_backend=get_target_backend(),
        source=source,
        target=target,
        source_path=source_path,
        target_path=target_path,
        registration=registration,
        cache=cache,
        stats=stats,
    )
    return TextStageResult(
        source_backend,
        source_blocks,
        target_blocks,
        source_bubbles,
        target_bubbles,
        False,
        diagnostics,
    )


__all__ = ["TextStageResult", "run_text_stage"]
