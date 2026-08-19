from __future__ import annotations

"""Post-run page workspace integrity validation.

Atomic files prevent byte-level truncation; this validator checks the next layer:
that key page artifacts are decodable, have the TARGET dimensions, belong to the
selected mode, and that state JSON is parseable.  It is intentionally lightweight
and runs after every page route, including passthrough/reveal early returns.
"""

import json
from pathlib import Path
from typing import Any

import cv2

from .mode_contracts import mode_artifact_violations


_PAGE_IMAGE_NAMES = (
    "final.png", "final_reviewed.png", "review_preview.png",
    "target_original.png",
    "text_layer.png", "chinese_transfer_layer.png",
    "mask_transfer_layer.png", "mask_transfer_mask.png",
    "direct_patch_layer.png", "direct_patch_regions.png",
    "aligned_overlay_reveal_layer.png", "aligned_overlay_reveal_mask.png",
)

_JSON_NAMES = (
    "project.json", "qa.json", "direct_patch.json", "mask_transfer.json",
    "transparent_bubble_reveal.json", "aligned_overlay_reveal.json",
)


def _decode_shape(path: Path) -> tuple[int, int] | None:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None
    if img is None or img.ndim < 2:
        return None
    return int(img.shape[0]), int(img.shape[1])


def _valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def validate_page_workspace(page_root: str | Path, project: Any, mode: str, *, selected_strategy: str = "") -> dict:
    root = Path(page_root)
    issues: list[dict] = []
    target = root / "target_original.png"
    target_shape = _decode_shape(target) if target.exists() else None

    final = root / "final.png"
    if not final.exists():
        issues.append({"code": "missing_final", "path": "final.png"})
    elif _decode_shape(final) is None:
        issues.append({"code": "invalid_image", "path": "final.png"})

    for name in _PAGE_IMAGE_NAMES:
        p = root / name
        if not p.exists():
            continue
        shape = _decode_shape(p)
        if shape is None:
            issues.append({"code": "invalid_image", "path": name})
            continue
        if target_shape is not None and shape != target_shape:
            issues.append({"code": "shape_mismatch", "path": name, "shape": list(shape), "target_shape": list(target_shape)})

    for name in _JSON_NAMES:
        p = root / name
        if p.exists() and not _valid_json(p):
            issues.append({"code": "invalid_json", "path": name})

    for violation in mode_artifact_violations(mode, root, selected_strategy=selected_strategy):
        issues.append({"code": "mode_artifact_leak", "detail": violation})

    # No finished run should leave atomic-write debris behind. The active page
    # lock is deliberately excluded; validation happens while the guard is held.
    temp_files = sorted(p.name for p in root.glob(".*.tmp") if p.is_file())
    for name in temp_files:
        issues.append({"code": "orphan_temp_file", "path": name})

    # Validate route-specific expected output only when the subsystem says it was
    # actually used; blank/no-text pages remain valid passthrough results.
    meta = getattr(project, "meta", {}) or {}
    direct_used = bool((meta.get("direct_patch") or {}).get("used"))
    mask_used = bool((meta.get("mask_replace") or {}).get("used"))
    reletter_meta = meta.get("reletter") or {}
    reletter_count = int(reletter_meta.get("successful_regions") or reletter_meta.get("applied_count") or 0)
    required: list[str] = []
    if direct_used:
        required.append("direct_patch_layer.png")
    if mask_used:
        required.extend(["mask_transfer_layer.png", "mask_transfer_mask.png"])
    if reletter_count > 0:
        required.append("text_layer.png")
    for name in required:
        if not (root / name).exists():
            issues.append({"code": "missing_route_artifact", "path": name})

    return {
        "schema": "manga_hd_translation_transfer.workspace_integrity.v1",
        "pass": not issues,
        "mode": str(mode),
        "selected_strategy": str(selected_strategy or ""),
        "target_shape": list(target_shape) if target_shape else None,
        "issues": issues,
        "checked_images": [n for n in _PAGE_IMAGE_NAMES if (root / n).exists()],
        "checked_json": [n for n in _JSON_NAMES if (root / n).exists()],
    }
