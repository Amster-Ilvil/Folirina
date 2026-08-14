from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

from .io_utils import load_json, stem_id
from .models import PagePair
from .schema_compat import as_dict, as_dict_rows, normalize_project, normalize_route_meta
from .result_state import resolve_result_state


@dataclass(slots=True)
class PageWorkspace:
    """Resolved on-disk/in-memory state for one paired page.

    The GUI must never reuse a result from a previously selected page.  This
    small, Qt-free resolver gives every workbench view one shared page identity
    and can therefore be regression-tested without starting a GUI.
    """

    page_id: str
    page_root: Path | None = None
    project_path: Path | None = None
    project_data: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    qa_summary: dict[str, Any] = field(default_factory=dict)
    review_regions: list[dict[str, Any]] = field(default_factory=list)
    manual_effect_candidates: list[dict[str, Any]] = field(default_factory=list)
    result_path: str = ""
    review_path: str = ""
    mask_path: str = ""

    @property
    def processed(self) -> bool:
        return bool(self.project_path and self.project_path.exists()) or bool(self.result_path)


def page_id_for_pair(pair: PagePair) -> str:
    return stem_id(pair.target_path)


def _v0816_ascii_page_id(path: str | Path) -> str:
    """Return the cache id produced by v0.8.16 and earlier.

    Older releases stripped all non-ASCII characters.  Keep this only as a
    read-only compatibility probe so existing projects can still be previewed
    after upgrading; new output always uses Unicode-safe ``stem_id``.
    """
    stem = Path(path).stem
    return re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("_") or "page"


def _existing_path(*values: str | Path | None) -> str:
    for value in values:
        if not value:
            continue
        try:
            path = Path(value)
            if path.exists() and path.is_file():
                return str(path)
        except (OSError, TypeError, ValueError):
            continue
    return ""




def _safe_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists() and path.is_file():
            data = load_json(path)
            return data if isinstance(data, dict) else {}
    except Exception:
        # A partially written/corrupt optional UI cache must not crash page
        # navigation.  Processing code still reports its own hard failures.
        return {}
    return {}


def _project_dict(project: Any, page_id: str) -> dict[str, Any]:
    if project is None or str(getattr(project, "page_id", "")) != page_id:
        return {}
    try:
        data = project.to_dict() if hasattr(project, "to_dict") else {}
        return normalize_project(data)
    except Exception:
        return {}


def resolve_page_workspace(
    output_dir: str | Path | None,
    pair: PagePair,
    in_memory_project: Any = None,
) -> PageWorkspace:
    """Resolve artifacts for exactly ``pair`` and never a global "last result".

    Resolution deliberately prefers reviewed page-local output.  Batch output
    is a fallback only for the same page id, so changing the selected row can no
    longer display Japanese/source page N with the final image from page N-1.
    """

    page_id = page_id_for_pair(pair)
    root: Path | None = None
    if output_dir:
        try:
            pages_root = Path(output_dir) / "pages"
            canonical = pages_root / page_id
            legacy_single = pages_root / Path(pair.target_path).stem
            legacy_batch = pages_root / _v0816_ascii_page_id(pair.target_path)
            # v0.8.16 single-page processing used the raw target stem while its
            # batch path used the old ASCII-only stem_id(). Prefer new Unicode-
            # safe output, then either legacy layout. This is intentionally read-
            # only compatibility; new writes never recreate collision-prone ids.
            if canonical.exists():
                root = canonical
            elif legacy_single.exists():
                root = legacy_single
            elif legacy_batch.exists():
                root = legacy_batch
            else:
                root = canonical
        except (TypeError, ValueError):
            root = None

    disk_project: dict[str, Any] = {}
    project_path: Path | None = None
    if root is not None:
        project_path = root / "project.json"
        disk_project = normalize_project(_safe_json(project_path))

    memory_project = _project_dict(in_memory_project, page_id)
    # Disk first, then the current in-memory object.  The latter may contain
    # artifact paths from a just-finished worker before the filesystem metadata
    # has been re-read by the UI.
    project_data = dict(disk_project)
    project_data.update(memory_project)

    artifacts = as_dict(disk_project.get("artifacts"))
    artifacts.update(as_dict(memory_project.get("artifacts")))
    meta = as_dict(disk_project.get("meta"))
    meta.update(as_dict(memory_project.get("meta")))

    result_path = ""
    review_path = ""
    mask_path = ""
    qa_summary: dict[str, Any] = as_dict(meta.get("qa_summary"))

    if root is not None:
        state = resolve_result_state(
            root,
            artifacts,
            extra_candidates=[Path(output_dir) / "final" / f"{Path(pair.target_path).stem}.png" if output_dir else None],
        )
        result_path = str(state.current) if state.current is not None else ""
        review_path = _existing_path(
            root / "review_preview.png",
            artifacts.get("review_preview"),
            result_path,
        )
        mask_path = _existing_path(
            root / "manual_transfer_mask.png",
            root / "mask_transfer_mask.png",
            artifacts.get("mask_transfer_mask"),
            root / "clear_mask.png",
            artifacts.get("clear_mask"),
        )
        qa_data = _safe_json(root / "qa.json")
        if isinstance(qa_data.get("summary"), dict):
            qa_summary = dict(qa_data["summary"])
    else:
        result_path = _existing_path(artifacts.get("final"), artifacts.get("book_final"))
        review_path = _existing_path(artifacts.get("review_preview"), result_path)
        mask_path = _existing_path(artifacts.get("mask_transfer_mask"), artifacts.get("clear_mask"))

    direct_meta = normalize_route_meta(meta.get("direct_patch"))
    aligned_meta = normalize_route_meta(meta.get("aligned_overlay_reveal"))
    direct_used = bool(direct_meta.get("used"))
    aligned_route = bool(
        aligned_meta.get("used")
        or str(meta.get("transfer_mode", "")) == "aligned_overlay_reveal"
        or str(meta.get("transfer_planner", {}).get("strategy", "")) == "aligned_overlay_reveal"
    )
    if aligned_route:
        mm = aligned_meta
    elif direct_used:
        mm = direct_meta
    else:
        mm = normalize_route_meta(meta.get("mask_replace"))
    review_regions = as_dict_rows(mm.get("review_regions") or mm.get("manual_reletter_required"))
    route_diag = as_dict(mm.get("diagnostics"))
    direct_diag = as_dict(direct_meta.get("diagnostics"))
    if aligned_route:
        candidate_source = mm.get("manual_effect_candidates") or route_diag.get("manual_effect_candidates")
    else:
        # Backward compatibility: v0.8-v1.2 pages could route through Mask while
        # storing omission/Reveal candidates under Direct diagnostics.  Preserve
        # those candidates before looking at the active route.
        candidate_source = (
            direct_meta.get("manual_effect_candidates")
            or direct_diag.get("manual_effect_candidates")
            or mm.get("manual_effect_candidates")
            or route_diag.get("manual_effect_candidates")
        )
    manual_effect_candidates = as_dict_rows(candidate_source)

    if aligned_route:
        candidates = [artifacts.get("aligned_overlay_reveal_mask")]
        if root is not None:
            candidates[:0] = [root / "manual_aligned_overlay_reveal_mask.png", root / "aligned_overlay_reveal_mask.png"]
        mask_path = _existing_path(*candidates)
    elif direct_used:
        candidates = [artifacts.get("direct_patch_regions")]
        if root is not None:
            candidates[:0] = [root / "manual_direct_patch_regions.png", root / "direct_patch_regions.png"]
        mask_path = _existing_path(*candidates)

    return PageWorkspace(
        page_id=page_id,
        page_root=root,
        project_path=project_path,
        project_data=project_data,
        artifacts=artifacts,
        meta=meta,
        qa_summary=qa_summary,
        review_regions=review_regions,
        manual_effect_candidates=manual_effect_candidates,
        result_path=result_path,
        review_path=review_path,
        mask_path=mask_path,
    )
