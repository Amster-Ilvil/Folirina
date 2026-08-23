from __future__ import annotations

"""Explicit transfer-mode capability and workspace-isolation contracts.

The project historically accumulated mode checks directly inside the main
pipeline.  That made it easy for a fix in one route to accidentally activate a
renderer, editor or stale artifact from another route.  This module is the single
source of truth for *which subsystem a mode may use* and for cleaning/auditing
mode-owned artifacts.

Low-level pure stages (decode, registration, geometry, OCR cache helpers) may be
shared.  Renderers and review state are not shared unless the contract says so.
"""

from pathlib import Path
from typing import Iterable

from .modes.base import ModeContract
from .modes.registry import (
    ACTIVE_MODE_ORDER, LEGACY_MODE_ORDER, SUPPORTED_MODE_ORDER, SUPPORTED_MODES, get_mode_spec,
)
from .review_artifacts import STATIC_REVIEW_INPUTS, existing_review_input_paths

def get_mode_contract(mode: str) -> ModeContract:
    return get_mode_spec(mode).contract


MODE_DERIVED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    key: tuple(get_mode_spec(key).owned_artifacts)
    for key in SUPPORTED_MODE_ORDER
    if get_mode_spec(key).owned_artifacts
}


# Derived artifacts are safe to delete before a fresh automatic pass. Manual
# *input* files are intentionally absent from this list. Ownership comes from mode specs.
COMMON_DERIVED_ARTIFACTS = (
    "run_receipt.json", "transfer_audit.json", "qa.json",
    "clear_mask.png", "target_clear_mask.png",
    "final_reviewed.png", "review_applied.json", "review_base.png",
    "review_preview.png", "editable_reviewed.ora", "editable_reviewed.psd",
    "editable.ora", "editable.psd", "inpainted.png",
    "text_layer.png", "text_layer_reviewed.png", "chinese_transfer_layer.png",
    "review_overrides.template.json", "removed_text_preview.png", "remove_text_stage.json",
)

# Generated evidence bundles that belong to the last automatic pass.  Manual
# OCR editor state lives under ``ocr_edit`` and is intentionally not included.
COMMON_DERIVED_DIRECTORIES = ("replace_translation",)


def clear_stale_mode_outputs(page_root: str | Path, *, strict: bool = False) -> dict[str, int]:
    """Remove old renderer outputs before a fresh automatic process.

    This is deliberately mode-agnostic: fresh processing should start from a
    clean derived workspace and then recreate only the selected route's outputs.
    User-authored masks/overrides are preserved and are separately owner-gated.
    """
    root = Path(page_root)
    removed = 0
    failures: list[str] = []
    for name in COMMON_DERIVED_ARTIFACTS:
        p = root / name
        try:
            if p.exists():
                p.unlink()
                removed += 1
        except OSError as exc:
            failures.append(f"{p}: {exc}")
    for names in MODE_DERIVED_ARTIFACTS.values():
        for name in names:
            p = root / name
            try:
                if p.exists():
                    p.unlink()
                    removed += 1
            except OSError as exc:
                failures.append(f"{p}: {exc}")
    removed_dirs = 0
    for name in COMMON_DERIVED_DIRECTORIES:
        path = root / name
        if not path.exists():
            continue
        try:
            import shutil
            if path.is_dir():
                shutil.rmtree(path)
                removed_dirs += 1
            else:
                path.unlink()
                removed += 1
        except OSError as exc:
            failures.append(f"{path}: {exc}")
    if failures and strict:
        joined = "\n".join(failures[:20])
        raise RuntimeError(
            "无法完整清除上一模式的派生产物；为避免不同模式结果交叉污染，本次处理已在写入新结果前停止。\n" + joined
        )
    return {"removed": removed, "removed_dirs": removed_dirs, "failures": len(failures)}


def artifact_ownership_snapshot(page_root: str | Path) -> dict[str, list[str]]:
    root = Path(page_root)
    out: dict[str, list[str]] = {}
    for owner, names in MODE_DERIVED_ARTIFACTS.items():
        present = [name for name in names if (root / name).exists()]
        if present:
            out[owner] = present
    return out


def mode_artifact_violations(mode: str, page_root: str | Path, *, selected_strategy: str | None = None) -> list[str]:
    """Return renderer artifacts that are impossible for the requested route.

    Auto is allowed to leave either Direct or Mask artifacts depending on the
    selected strategy. Hybrid may leave Mask + Reletter artifacts. Explicit modes
    are strict.
    """
    key = str(mode or "").strip().lower()
    snap = artifact_ownership_snapshot(page_root)
    allowed: set[str]
    if key == "auto":
        strategy = str(selected_strategy or "").strip().lower()
        if strategy == "direct_patch":
            allowed = {"direct_patch"}
        elif strategy == "aligned_overlay_reveal":
            allowed = {"aligned_overlay_reveal"}
        else:
            allowed = {"mask_replace"}
    elif key == "hybrid":
        # Hybrid owns its own mask/text artifacts. It may *execute* a private
        # mask-first + OCR fallback pipeline, but must never publish Mask or
        # Reletter-owned files into the same workspace.
        allowed = {"hybrid"}
    elif key == "aligned_overlay_reveal":
        allowed = {"aligned_overlay_reveal"}
    else:
        allowed = {key}
    violations: list[str] = []
    for owner, names in snap.items():
        if owner not in allowed:
            violations.extend(f"{owner}:{name}" for name in names)
    return sorted(violations)


def review_owner_compatible(owner_mode: str | None, requested_mode: str) -> bool:
    owner = str(owner_mode or "").strip().lower()
    requested = str(requested_mode or "").strip().lower()
    if not owner:
        return True
    return owner == requested

# User-authored review inputs are mode-owned. They are archived, not deleted, when
# the user switches automatic transfer mode for an already-processed page.
MODE_OWNED_REVIEW_INPUTS = STATIC_REVIEW_INPUTS


def archive_review_state_if_mode_changed(page_root: str | Path, requested_mode: str) -> dict:
    """Transactionally archive incompatible manual review inputs on mode switch.

    A partial move is worse than no archive at all: the new renderer could see
    half of the previous mode's masks/history while the other half has already
    disappeared. Stage complete copies first, then remove the live inputs, then
    publish the archive. Any failure restores the live review inputs and aborts
    the mode switch.
    """
    import os
    import shutil
    import tempfile
    import uuid

    from .io_utils import load_json, save_json

    root = Path(page_root)
    project_path = root / "project.json"
    if not project_path.exists():
        return {"changed": False, "old_mode": "", "new_mode": str(requested_mode), "archived": []}
    try:
        obj = load_json(project_path)
        old_mode = str(((obj.get("meta") or {}).get("transfer_mode") or "")).strip().lower() if isinstance(obj, dict) else ""
    except Exception:
        old_mode = ""
    new_mode = str(requested_mode or "").strip().lower()
    if not old_mode or old_mode == new_mode:
        return {"changed": False, "old_mode": old_mode, "new_mode": new_mode, "archived": []}

    sources = existing_review_input_paths(root)
    review_root = root / "review_archive"
    if review_root.is_symlink():
        raise RuntimeError("review archive path is an unsafe symlink")
    if review_root.exists() and not review_root.is_dir():
        raise RuntimeError("review archive path exists but is not a directory")
    review_root.mkdir(parents=True, exist_ok=True)
    try:
        review_root.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"review archive path escapes page workspace: {exc}") from exc
    archive = review_root / f"{old_mode}_to_{new_mode}"
    stage = Path(tempfile.mkdtemp(prefix=f".txn-{old_mode}-to-{new_mode}-", dir=str(review_root)))
    removed_sources: list[Path] = []
    published: list[str] = []
    try:
        try:
            for src in sources:
                shutil.copy2(src, stage / src.name)
        except OSError as exc:
            raise RuntimeError(f"review archive staging failed: {exc}") from exc

        try:
            for src in sources:
                src.unlink()
                removed_sources.append(src)
        except OSError as exc:
            for src in removed_sources:
                try:
                    shutil.copy2(stage / src.name, src)
                except OSError:
                    pass
            raise RuntimeError(f"review archive source removal failed: {exc}") from exc

        try:
            archive.mkdir(parents=True, exist_ok=True)
            for src in sources:
                staged = stage / src.name
                dst = archive / src.name
                tmp = archive / f".{src.name}.{uuid.uuid4().hex}.tmp"
                shutil.copy2(staged, tmp)
                os.replace(tmp, dst)
                published.append(src.name)
            save_json(archive / "archive_manifest.json", {
                "schema": "manga_hd_translation_transfer.review_mode_archive.v2",
                "old_mode": old_mode,
                "new_mode": new_mode,
                "files": published,
            })
        except Exception as exc:
            # The archive is recovery-only. The live page state is authoritative:
            # restore every source before propagating the failure so processing
            # cannot continue with a half-switched review contract.
            for src in sources:
                if not src.exists():
                    try:
                        shutil.copy2(stage / src.name, src)
                    except OSError:
                        pass
            raise RuntimeError(f"review archive publish failed: {exc}") from exc

        return {
            "changed": True, "old_mode": old_mode, "new_mode": new_mode,
            "archived": published, "archive_dir": str(archive),
        }
    finally:
        shutil.rmtree(stage, ignore_errors=True)

def mode_scoped_config_payload(config: object) -> dict:
    """Return only configuration namespaces that can affect the selected mode.

    Completed-page resume fingerprints used to hash almost the entire global
    configuration. Changing a Reletter font therefore invalidated completed
    Direct pages, and tuning Transparent Reveal invalidated Mask pages. That is a
    maintenance form of cross-mode coupling even when pixels were correct.

    Stage caches remain independently keyed by their own relevant config. This
    payload only scopes the *page completion/resume* fingerprint.
    """
    if hasattr(config, "model_dump"):
        cfg = config.model_dump(mode="json")
    else:
        cfg = dict(config)  # type: ignore[arg-type]
    mode = str(((cfg.get("transfer") or {}).get("mode") or "direct_patch")).strip().lower()
    get_mode_contract(mode)  # validate

    common = {k: cfg.get(k) for k in ("page_management", "pairing", "registration", "qa") if k in cfg}
    # Book-level alignment resource limits affect *how* a directory pairing is
    # solved, not the rendered result once a concrete SOURCE/TARGET pair has
    # been selected. Keep them out of per-page resume identity so tuning long-
    # book memory/performance does not invalidate completed pages or mode locks.
    if isinstance(common.get("pairing"), dict):
        common["pairing"] = dict(common["pairing"])
        common["pairing"].pop("smart_alignment_full_matrix_max_cells", None)
        common["pairing"].pop("smart_alignment_band", None)
    common["transfer"] = {"mode": mode}

    keys_by_mode = {
        "direct_patch": ("direct_patch", "bubbles", "dual_source", "replace_translation"),
        "mask_replace": ("mask_replace", "bubbles", "matching", "masking"),
        "hybrid": ("hybrid", "ocr", "bubbles", "matching", "masking", "inpainting"),
        "reletter": ("reletter", "ocr", "bubbles", "matching", "masking", "inpainting"),
        "transparent_bubble_reveal": ("transparent_bubble_reveal", "bubbles"),
        "aligned_overlay_reveal": ("aligned_overlay_reveal", "bubbles"),
        "auto": (
            "direct_patch", "mask_replace", "aligned_overlay_reveal", "ocr",
            "bubbles", "matching", "masking", "dual_source", "replace_translation",
        ),
    }
    payload = dict(common)
    for key in keys_by_mode[mode]:
        if key in cfg:
            payload[key] = cfg[key]
    if mode == "auto" and isinstance(payload.get("mask_replace"), dict):
        # Legacy Auto is frozen. Explicit Mask-only renderer contracts must not
        # change Auto fingerprints or execution defaults.
        for key in (
            "paired_diff_dense_flow_geometry_only",
            "paired_diff_render_use_global_raster",
            "paired_diff_forbid_dense_glyph_warp",
            "paired_diff_proxy_warn_iou",
            "paired_diff_proxy_warn_area_ratio",
        ):
            payload["mask_replace"].pop(key, None)

    if mode == "transparent_bubble_reveal":
        reveal_cfg = cfg.get("transparent_bubble_reveal") or {}
        if bool(reveal_cfg.get("target_text_presence_ocr_enabled", False)) and "ocr" in cfg:
            payload["ocr"] = cfg["ocr"]

    # Reletter currently shares only paired-diff *geometry* extraction with the
    # mask subsystem. Hash that geometry subset instead of every Mask renderer
    # tuning knob, so pixel-transfer adjustments do not invalidate Reletter jobs.
    if mode == "reletter":
        mr = ((cfg.get("reletter") or {}).get("candidates") or {})
        geometry = {
            k: v for k, v in mr.items()
            if str(k).startswith("paired_diff_")
            or str(k).startswith("photo_pair_")
        }
        payload["reletter_geometry"] = geometry
    return payload


def mode_execution_violations(
    mode: str,
    *,
    direct_used: bool = False,
    mask_used: bool = False,
    reletter_used: bool = False,
    transparent_used: bool = False,
    aligned_used: bool = False,
) -> list[str]:
    """Validate *executed subsystems*, not only files left in the workspace.

    Artifact isolation catches most cross-mode leaks after rendering, but a future
    refactor could still call an unauthorized subsystem and later delete/overwrite
    its artifacts. This runtime contract closes that gap and gives regression tests
    a stable place to assert mode boundaries.
    """
    c = get_mode_contract(mode)
    violations: list[str] = []
    if direct_used and not c.direct:
        violations.append("direct")
    if mask_used and not c.mask_replace and not (c.orchestrator and c.may_fallback_to_mask):
        violations.append("mask_replace")
    if reletter_used and not c.reletter:
        violations.append("reletter")
    if transparent_used and not c.transparent_reveal:
        violations.append("transparent_bubble_reveal")
    if aligned_used and not c.aligned_reveal:
        violations.append("aligned_overlay_reveal")
    return sorted(set(violations))


def ocr_route_executed(route: str | None) -> bool:
    """Return True only when a recorded OCR route actually touched OCR work.

    Empty routes and explicit ``skipped_*`` markers are non-execution.  Cache
    hits still count as OCR usage because the mode consumed OCR-derived content.
    """
    value = str(route or "").strip().lower()
    if not value or value.startswith("skipped"):
        return False
    return value not in {"none", "null", "disabled", "geometry_only"}


def mode_ocr_execution_violations(mode: str, stats: dict | None) -> list[str]:
    """Audit OCR usage against the selected mode capability contract."""
    contract = get_mode_contract(mode)
    if contract.may_use_ocr:
        return []
    stats = stats or {}
    violations: list[str] = []
    if ocr_route_executed(stats.get("ocr_source")):
        violations.append("ocr_source")
    if ocr_route_executed(stats.get("ocr_target")):
        violations.append("ocr_target")
    return violations
