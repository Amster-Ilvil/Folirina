from __future__ import annotations

"""Workspace and input preparation for one page run.

The module owns page-local cleanup, PageMark normalization, image decoding and
SOURCE-candidate discovery.  It deliberately stops before registration and any
renderer decision, keeping the orchestration entry point small while preserving
all previous side-effect ordering.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cache import PageStageCache
from .io_utils import read_image
from .mode_contracts import archive_review_state_if_mode_changed, clear_stale_mode_outputs
from .models import PagePair
from .page_management import PageMark
from .result_state import invalidate_manual_review_state
from .source_candidate_service import _load_additional_source_specs, _resolve_secondary_source_spec


@dataclass
class WorkspacePreparation:
    page_root: Path
    mark: PageMark
    review_mode_archive: Any
    stale_mode_cleanup: Any


@dataclass
class PrefetchedPageImages:
    source_path: str
    target_path: str
    source: np.ndarray
    target: np.ndarray


@dataclass
class PageInputContext:
    authority_source_path: str | Path
    source_path_local: str | Path
    target_path_local: str | Path
    authority_source: np.ndarray
    source: np.ndarray
    target: np.ndarray
    replace_source_specs: list[dict]
    secondary_source_spec: dict | None
    secondary_source_available: bool
    stage_cache: PageStageCache
    cache_stats: dict[str, str]


def prepare_workspace(
    pair: PagePair,
    page_root: str | Path,
    page_mark: PageMark | dict | None,
    *,
    mode: str,
) -> WorkspacePreparation:
    root = Path(page_root)
    root.mkdir(parents=True, exist_ok=True)
    review_mode_archive = archive_review_state_if_mode_changed(root, mode)
    stale_mode_cleanup = clear_stale_mode_outputs(root, strict=True)
    invalidate_manual_review_state(root)
    mark = (
        page_mark
        if isinstance(page_mark, PageMark)
        else PageMark.from_dict(page_mark)
        if page_mark
        else PageMark(
            page_type="content",
            origin="default",
            source_name=Path(pair.source_path).name,
            target_name=Path(pair.target_path).name,
        )
    )
    return WorkspacePreparation(root, mark, review_mode_archive, stale_mode_cleanup)


def prefetch_page_images(pair: PagePair) -> PrefetchedPageImages:
    """Decode one SOURCE/TARGET pair for bounded book-level look-ahead.

    No workspace or cache state is touched here; the normal page lifecycle still
    owns all transactional side effects. The returned arrays are consumed by at
    most one page run and are never shared across renderers.
    """
    source = read_image(pair.source_path)
    target = read_image(pair.target_path)
    return PrefetchedPageImages(
        source_path=str(pair.source_path), target_path=str(pair.target_path),
        source=source, target=target,
    )


def load_page_inputs(
    pair: PagePair,
    page_root: str | Path,
    config: Any,
    *,
    prefetched_images: PrefetchedPageImages | None = None,
) -> PageInputContext:
    authority_source_path = pair.source_path
    source_path_local = authority_source_path
    target_path_local = pair.target_path
    if (
        prefetched_images is not None
        and str(prefetched_images.source_path) == str(authority_source_path)
        and str(prefetched_images.target_path) == str(target_path_local)
    ):
        authority_source = prefetched_images.source
        source = authority_source
        target = prefetched_images.target
    else:
        authority_source = read_image(authority_source_path)
        source = authority_source
        target = read_image(target_path_local)

    replace_source_specs = _load_additional_source_specs(
        authority_source_path, config.replace_translation
    )
    secondary_source_spec = _resolve_secondary_source_spec(
        authority_source_path, target_path_local, config.dual_source
    )
    if secondary_source_spec is not None and all(
        str(row.get("path")) != str(secondary_source_spec.get("path"))
        for row in replace_source_specs
    ):
        replace_source_specs.append(secondary_source_spec)

    stage_cache = PageStageCache(page_root, enabled=config.cache.enabled)
    return PageInputContext(
        authority_source_path=authority_source_path,
        source_path_local=source_path_local,
        target_path_local=target_path_local,
        authority_source=authority_source,
        source=source,
        target=target,
        replace_source_specs=replace_source_specs,
        secondary_source_spec=secondary_source_spec,
        secondary_source_available=secondary_source_spec is not None,
        stage_cache=stage_cache,
        cache_stats={},
    )


__all__ = [
    "WorkspacePreparation",
    "PrefetchedPageImages",
    "PageInputContext",
    "prepare_workspace",
    "prefetch_page_images",
    "load_page_inputs",
]
