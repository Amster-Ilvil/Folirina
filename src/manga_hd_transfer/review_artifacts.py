from __future__ import annotations

"""Page-local review artifact ownership and safe path resolution.

Manual review rows may reference sparse Reveal masks/patches by filename.  Those
references come from JSON and therefore must never be allowed to escape the page
workspace.  The same inventory is shared by mode-switch archiving and automatic
run rollback so a failed mode change restores *all* user-authored review input,
not only the fixed mask filenames.
"""

from pathlib import Path
from typing import Any

from .io_utils import load_json
from .schema_compat import as_dict_rows, normalize_overrides


STATIC_REVIEW_INPUTS: tuple[str, ...] = (
    "review_overrides.json",
    "review_history.json",
    "manual_force_transfer_mask.png",
    "manual_force_auto_target_override.png",
    "manual_force_settings.json",
    "manual_clear_mask.png",
    "manual_japanese_clear_mask.png",
    "manual_transfer_mask.png",
    "manual_direct_patch_regions.png",
    "manual_target_layer_erase_mask.png",
    "manual_target_layer_restore_mask.png",
)

_DYNAMIC_REVIEW_FILE_KEYS: tuple[str, ...] = (
    "reveal_mask_file",
    "reveal_patch_file",
)


def safe_page_artifact_path(page_root: str | Path, value: object) -> Path | None:
    """Resolve one JSON-referenced *leaf* artifact inside ``page_root``.

    Generated review artifacts are intentionally flat page-local files. Reject
    separators, dot-paths, NULs, and an existing symlink that resolves outside
    the workspace.  This both protects malformed/imported projects and gives all
    review consumers one consistent path contract.
    """
    name = str(value or "").strip()
    if not name or name in {".", ".."} or "\x00" in name or "/" in name or "\\" in name:
        return None
    if Path(name).name != name:
        return None
    root = Path(page_root)
    candidate = root / name
    try:
        resolved_root = root.resolve()
        resolved_candidate = candidate.resolve(strict=False)
        resolved_candidate.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate


def _load_overrides(page_root: Path) -> dict[str, Any]:
    path = page_root / "review_overrides.json"
    if not path.is_file():
        return {}
    try:
        payload = load_json(path)
        return normalize_overrides(payload if isinstance(payload, dict) else {})
    except Exception:
        return {}


def dynamic_review_artifact_names(
    page_root: str | Path,
    overrides: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return safe sparse artifact names referenced by manual review rows."""
    root = Path(page_root)
    data = normalize_overrides(overrides) if isinstance(overrides, dict) else _load_overrides(root)
    names: list[str] = []
    seen: set[str] = set()
    for row in as_dict_rows(data.get("manual_effect_regions")):
        for key in _DYNAMIC_REVIEW_FILE_KEYS:
            path = safe_page_artifact_path(root, row.get(key))
            if path is None:
                continue
            name = path.name
            if name not in seen:
                seen.add(name)
                names.append(name)
    return tuple(names)


def review_input_names_for_snapshot(page_root: str | Path) -> tuple[str, ...]:
    """Presence contract for every user-authored input of the current page."""
    return tuple(dict.fromkeys((*STATIC_REVIEW_INPUTS, *dynamic_review_artifact_names(page_root))))


def existing_review_input_paths(page_root: str | Path) -> list[Path]:
    root = Path(page_root)
    paths: list[Path] = []
    for name in review_input_names_for_snapshot(root):
        path = safe_page_artifact_path(root, name)
        if path is not None and path.is_file():
            paths.append(path)
    return paths


__all__ = [
    "STATIC_REVIEW_INPUTS",
    "safe_page_artifact_path",
    "dynamic_review_artifact_names",
    "review_input_names_for_snapshot",
    "existing_review_input_paths",
]
