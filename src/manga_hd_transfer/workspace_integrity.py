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
from .run_receipt import validate_run_receipt


_PAGE_IMAGE_NAMES = (
    "final.png", "final_reviewed.png", "review_preview.png",
    "target_original.png",
    "text_layer.png", "chinese_transfer_layer.png",
    "mask_transfer_layer.png", "mask_transfer_mask.png",
    "hybrid_transfer_layer.png", "hybrid_transfer_mask.png", "hybrid_text_layer.png",
    "reletter_text_layer.png",
    "direct_patch_layer.png", "direct_patch_regions.png",
    "aligned_overlay_reveal_layer.png", "aligned_overlay_reveal_mask.png",
)

_JSON_NAMES = (
    "project.json", "qa.json", "run_receipt.json", "direct_patch.json", "mask_transfer.json",
    "hybrid_transfer.json", "reletter.json",
    "transparent_bubble_reveal.json", "aligned_overlay_reveal.json",
)

# A passthrough/direct-reject page owns no transfer pixels.  These generic files
# are valid for rendering modes but are stale evidence if they survive a route
# that explicitly kept TARGET unchanged.
_PASSTHROUGH_FORBIDDEN_DERIVED = (
    "transfer_audit.json", "clear_mask.png", "target_clear_mask.png",
    "text_layer.png", "chinese_transfer_layer.png", "review_preview.png",
    "hybrid_transfer_layer.png", "hybrid_transfer_mask.png", "hybrid_text_layer.png",
    "reletter_text_layer.png",
    "replace_translation",
)


def _decode_shape(path: Path) -> tuple[int, int] | None:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except (cv2.error, OSError):
        img = None
    if img is None or img.ndim < 2:
        return None
    return int(img.shape[0]), int(img.shape[1])


def _valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
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

    strategy_key = str(selected_strategy or "").strip().lower()
    if strategy_key in {"passthrough", "direct_reject"}:
        for name in _PASSTHROUGH_FORBIDDEN_DERIVED:
            if (root / name).exists():
                issues.append({"code": "passthrough_stale_artifact", "path": name})

    # No finished run should leave atomic-write debris behind. The active page
    # lock is deliberately excluded; validation happens while the guard is held.
    temp_files = sorted(p.name for p in root.glob(".*.tmp") if p.is_file())
    for name in temp_files:
        issues.append({"code": "orphan_temp_file", "path": name})

    # Validate the route metadata itself. Renderer-owned runtime checks are
    # produced inside the route; the common lifecycle only enforces their result.
    meta = getattr(project, "meta", {}) or {}
    project_mode = str(meta.get("transfer_mode") or "").strip().lower()
    requested_mode = str(mode or "").strip().lower()
    if project_mode and project_mode != requested_mode:
        issues.append({"code": "project_mode_mismatch", "expected": requested_mode, "actual": project_mode})
    execution = meta.get("mode_execution")
    if isinstance(execution, dict) and execution.get("pass") is False:
        issues.append({"code": "mode_execution_failed", "violations": list(execution.get("violations") or [])})
    isolation = meta.get("mode_isolation")
    if isinstance(isolation, dict) and isolation.get("pass") is False:
        issues.append({"code": "mode_isolation_failed", "violations": list(isolation.get("violations") or [])})
    issues.extend(validate_run_receipt(root, requested_mode=requested_mode, selected_strategy=selected_strategy))

    # Validate route-specific expected output only when the subsystem says it was
    # actually used; blank/no-text pages remain valid passthrough results.
    direct_used = bool((meta.get("direct_patch") or {}).get("used"))
    mask_used = bool((meta.get("mask_replace") or {}).get("used"))
    hybrid_meta = meta.get("hybrid") or {}
    hybrid_used = bool(hybrid_meta.get("used"))
    reletter_meta = meta.get("reletter") or {}
    reletter_count = int(reletter_meta.get("successful_regions") or reletter_meta.get("applied_count") or 0)
    if requested_mode == "reletter" and bool(reletter_meta.get("target_driven_regions_used")):
        region_diag = reletter_meta.get("target_driven_region_diagnostics") or {}
        recognized = int(region_diag.get("recognized_regions") or region_diag.get("region_count") or 0)
        if recognized > 0 and reletter_count <= 0:
            issues.append({
                "code": "reletter_no_published_regions",
                "recognized_regions": recognized,
                "successful_regions": reletter_count,
            })
    hybrid_reletter_count = int(hybrid_meta.get("reletter_fallback_success_count") or 0)
    required: list[str] = []
    if direct_used:
        required.append("direct_patch_layer.png")
    if mask_used:
        required.extend(["mask_transfer_layer.png", "mask_transfer_mask.png"])
    if hybrid_used:
        required.extend(["hybrid_transfer_layer.png", "hybrid_transfer_mask.png"])
    if hybrid_reletter_count > 0:
        required.append("hybrid_text_layer.png")
    if reletter_count > 0:
        required.append("reletter_text_layer.png")
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
