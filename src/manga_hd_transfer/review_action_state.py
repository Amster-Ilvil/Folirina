from __future__ import annotations

"""Pure page-artifact policy for review/workbench button availability.

Kept outside Qt so restored-session operability can be tested without PySide6.
"""

from pathlib import Path


def review_action_availability(page_root: str | Path | None) -> dict[str, bool]:
    if page_root is None:
        return {
            "can_review": False,
            "edit_clear_mask": False,
            "remove_text_only": False,
            "apply_mask_review": False,
            "force_transfer_mask": False,
            "target_layer_erase": False,
            "target_layer_restore": False,
            "reset_clear_mask": False,
            "reset_force_transfer_mask": False,
            "reset_target_layer_erase": False,
            "reset_target_layer_restore": False,
        }
    root=Path(page_root)
    project=(root/"project.json").exists()
    target=(root/"target_original.png").exists()
    final=(root/"final_reviewed.png").exists() or (root/"final.png").exists()
    clear=any((root/name).exists() for name in ("manual_clear_mask.png","target_clear_mask.png","clear_mask.png"))
    return {
        "can_review": bool(project),
        "edit_clear_mask": bool(target and clear),
        "remove_text_only": bool(target and clear),
        "apply_mask_review": bool(project),
        "force_transfer_mask": bool(project and target),
        "target_layer_erase": bool(target and final),
        "target_layer_restore": bool(target and final),
        "reset_clear_mask": bool((root/"manual_clear_mask.png").exists()),
        "reset_force_transfer_mask": bool(any((root/name).exists() for name in ("manual_force_transfer_mask.png","manual_force_auto_target_override.png"))),
        "reset_target_layer_erase": bool((root/"manual_target_layer_erase_mask.png").exists()),
        "reset_target_layer_restore": bool((root/"manual_target_layer_restore_mask.png").exists()),
    }


__all__=["review_action_availability"]
