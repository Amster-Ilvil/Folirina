from __future__ import annotations

import logging
from pathlib import Path

from .io_utils import load_json
from .mode_contracts import clear_stale_mode_outputs
from .schema_compat import as_dict, normalize_overrides, normalize_project

logger = logging.getLogger(__name__)


def has_reapplicable_review_state(page_root: str | Path, requested_mode: str | None = None) -> bool:
    """Return True only for manual state owned by the same automatic mode.

    This module intentionally contains no Qt imports and no pipeline imports, so
    review-state ownership can be reused by GUI workers, CLI tools and tests.
    """
    page_root = Path(page_root)
    requested = str(requested_mode or "").strip().lower()
    old_mode = ""
    project_path = page_root / "project.json"
    if project_path.exists():
        try:
            old_project = normalize_project(load_json(project_path))
            old_mode = str(as_dict(old_project.get("meta")).get("transfer_mode", "") or "").strip().lower()
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.debug("Unable to read prior project review ownership: %s", exc)
            old_mode = ""
    if requested and old_mode and requested != old_mode:
        return False

    file_markers = (
        "manual_force_transfer_mask.png",
        "manual_force_auto_target_override.png",
        "manual_force_settings.json",
        "manual_clear_mask.png",
        "manual_japanese_clear_mask.png",
        "manual_target_layer_erase_mask.png",
        "manual_target_layer_restore_mask.png",
    )
    if any((page_root / name).exists() for name in file_markers):
        return True

    override_path = page_root / "review_overrides.json"
    if not override_path.exists():
        return False
    try:
        overrides = normalize_overrides(load_json(override_path))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.debug("Unable to parse review overrides: %s", exc)
        return False
    if any(overrides.get(key) for key in (
        "manual_effect_regions", "manual_reletter", "restore_target_bubbles",
        "accept_candidate_targets", "accepted_source_units",
    )):
        return True
    if any(as_dict(overrides.get(key)) for key in ("text_overrides", "match_overrides", "unit_actions")):
        return True
    return bool(str(overrides.get("page_force_action", "") or "").strip())


def clear_reprocess_generated_artifacts(page_root: str | Path) -> list[str]:
    """Clear derived outputs while preserving manual review input state.

    Returns removed relative file names for diagnostics. The function is a pure
    page-workspace service and does not depend on Qt.
    """
    page_root = Path(page_root)
    stale_files = (
        "review_applied.json", "review_base.png",
        "manual_effect_transfer_layer.png", "manual_effect_transfer_mask.png",
        "manual_effect_clear_mask.png",
        "manual_force_transfer_layer.png", "manual_force_source_mask.png",
        "manual_force_apply.json", "manual_force_transfer.json",
        "manual_force_auto_target_evidence.png", "manual_force_auto_source_evidence.png",
        "manual_force_auto_removed_mask.png",
        "target_layer_erase_preview.png", "target_layer_restore_preview.png",
        "removed_text_preview.png", "remove_text_stage.json",
        "editable_reviewed.ora", "editable_reviewed.psd",
    )
    preexisting = {name for name in stale_files if (page_root / name).exists()}
    clear_stale_mode_outputs(page_root)
    removed: list[str] = [name for name in stale_files if name in preexisting and not (page_root / name).exists()]
    for name in stale_files:
        path = page_root / name
        try:
            existed = path.exists()
            path.unlink(missing_ok=True)
            if existed and name not in removed:
                removed.append(name)
        except OSError as exc:
            logger.warning("Unable to clear stale page artifact %s: %s", path, exc)
    return removed


__all__ = ["has_reapplicable_review_state", "clear_reprocess_generated_artifacts"]
