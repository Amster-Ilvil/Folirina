from __future__ import annotations

from pathlib import Path

"""Fast architecture checks for transfer-mode isolation.

This is intentionally lightweight enough to run before packaging a release. It
checks the contracts that are easy to regress when adding a new detector/editor.
The full pixel regressions remain in ``selftest.py``.
"""

from .workspace_guard import PageRunGuard, cleanup_orphan_temp_files
from .workspace_integrity import validate_page_workspace
from .cache import RESUME_SCHEMA
from .run_receipt import write_run_receipt, validate_run_receipt
from .module_contracts import audit_module_contracts
from .config import PipelineConfig



ACTIVE_PIXEL_MODE_PACKAGES = ("direct_patch", "mask_replace", "hybrid")
ACTIVE_MODE_PACKAGES = ("direct_patch", "mask_replace", "aligned_overlay_reveal", "hybrid", "reletter")
PRIVATE_RENDERER_HELPERS = (
    "geometry_ops.py", "selection_policy.py", "raster_primitives.py", "quality_ops.py",
    "warp_ops.py", "photo_text_ops.py", "content_audit.py", "transfer_models.py", "text_transfer.py",
)
FORBIDDEN_SHARED_RENDERER_MODULES = (
    "mask_transfer", "mask_geometry", "mask_selection", "mask_raster_primitives", "mask_quality",
    "mask_warp", "mask_photo_text", "mask_content_audit", "mask_transfer_models", "text_only_transfer",
    "reletter_executor", "reletter_binding", "reletter_regions", "ocr_edit_blocks", "ocr_edit_render",
    "lettering", "manual_effect", "masking", "inpainting", "pipeline_ocr_cleanup",
    "aligned_overlay_reveal", "aligned_overlay_reveal_core", "aligned_overlay_reveal_mode",
)


def _mode_package_cross_import_violations(package_root: Path) -> list[str]:
    violations: list[str] = []
    modes_root = package_root / "modes"
    for owner in ACTIVE_MODE_PACKAGES:
        owner_root = modes_root / owner
        for path in owner_root.rglob("*.py"):
            src = path.read_text(encoding="utf-8")
            for other in ACTIVE_MODE_PACKAGES:
                if other == owner:
                    continue
                markers = (
                    f"modes.{other}",
                    f"from ..{other}",
                    f"import ..{other}",
                    f"from ...modes.{other}",
                    f"import manga_hd_transfer.modes.{other}",
                )
                if any(marker in src for marker in markers):
                    violations.append(f"{owner}:{path.relative_to(package_root)}->{other}")
                    break
    return violations


def _shared_renderer_import_violations(package_root: Path) -> list[str]:
    violations: list[str] = []
    modes_root = package_root / "modes"
    for owner in ACTIVE_MODE_PACKAGES:
        for path in (modes_root / owner).rglob("*.py"):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not (stripped.startswith("from ") or stripped.startswith("import ")):
                    continue
                # Same-directory imports (``from .foo``) are the desired private
                # capsule wiring. Only parent/root renderer imports are forbidden.
                if stripped.startswith("from .") and not stripped.startswith("from .."):
                    continue
                for module in FORBIDDEN_SHARED_RENDERER_MODULES:
                    if f".{module} " in stripped or f".{module} import" in stripped or f"manga_hd_transfer.{module}" in stripped:
                        violations.append(f"{owner}:{path.relative_to(package_root)}:{module}")
    return sorted(set(violations))


def _active_mode_private_file_checks(package_root: Path) -> dict[str, bool]:
    modes_root = package_root / "modes"
    return {
        "direct_private_container_renderer": (modes_root / "direct_patch" / "container_renderer.py").exists(),
        "direct_private_pixel_helpers": all((modes_root / "direct_patch" / f).exists() for f in PRIVATE_RENDERER_HELPERS),
        "mask_private_pixel_helpers": all((modes_root / "mask_replace" / f).exists() for f in PRIVATE_RENDERER_HELPERS),
        "hybrid_private_pixel_helpers": all((modes_root / "hybrid" / f).exists() for f in PRIVATE_RENDERER_HELPERS),
        "hybrid_private_reletter_stack": all((modes_root / "hybrid" / f).exists() for f in ("executor.py","binding.py","regions.py","flow_cells.py","layout_policy.py","ocr_edit_blocks.py","ocr_edit_render.py","lettering_ops.py","manual_effect_ops.py","masking_ops.py","inpainting_ops.py","ocr_cleanup.py")),
        "reletter_private_stack": all((modes_root / "reletter" / f).exists() for f in ("executor.py","binding.py","regions.py","flow_cells.py","layout_policy.py","text_transfer.py","ocr_edit_blocks.py","ocr_edit_render.py","lettering_ops.py","manual_effect_ops.py","masking_ops.py","inpainting_ops.py","ocr_cleanup.py")),
        "direct_private_runtime_stack": all((modes_root / "direct_patch" / f).exists() for f in ("masking_ops.py","inpainting_ops.py","pixel_stage.py","execution_stage.py")),
        "mask_private_runtime_stack": all((modes_root / "mask_replace" / f).exists() for f in ("masking_ops.py","inpainting_ops.py","pixel_stage.py","execution_stage.py")),
        "mask_private_open_text_tool": (modes_root / "mask_replace" / "open_text_manual.py").exists(),
        "hybrid_private_open_text_tool": (modes_root / "hybrid" / "open_text_manual.py").exists(),
        "hybrid_private_stage_stack": all((modes_root / "hybrid" / f).exists() for f in ("pixel_stage.py","execution_stage.py")),
        "reletter_private_stage_stack": all((modes_root / "reletter" / f).exists() for f in ("pixel_stage.py","execution_stage.py")),
        "aligned_private_stack": all((modes_root / "aligned_overlay_reveal" / f).exists() for f in ("renderer.py","core.py","hole_renderer.py","bridge.py","validator.py","contract.py","runner.py")),
    }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

from .mode_contracts import (
    MODE_DERIVED_ARTIFACTS,
    MODE_OWNED_REVIEW_INPUTS,
    SUPPORTED_MODES, ACTIVE_MODE_ORDER, LEGACY_MODE_ORDER,
    get_mode_contract,
    mode_execution_violations,
    mode_ocr_execution_violations,
)


def run_architecture_audit() -> dict:
    contracts = {name: get_mode_contract(name) for name in sorted(SUPPORTED_MODES)}
    artifact_owner: dict[str, str] = {}
    duplicate_artifacts: list[str] = []
    for owner, names in MODE_DERIVED_ARTIFACTS.items():
        for name in names:
            prev = artifact_owner.get(name)
            if prev is not None and prev != owner:
                duplicate_artifacts.append(f"{name}:{prev},{owner}")
            artifact_owner[name] = owner

    from .paddle_runtime import runtime_root as paddle_ocr_runtime_root
    from .paddle_doc_runtime import runtime_root as paddle_doc_runtime_root

    module_audit = audit_module_contracts()
    package_root = Path(__file__).resolve().parent
    pipeline_src = _read_text(package_root / "pipeline_pixel_transfer_stage.py")
    direct_overlay_src = _read_text(package_root / "modes" / "direct_patch" / "overlay.py")
    mask_transfer_src = _read_text(package_root / "mask_transfer.py")
    direct_arbitration_src = _read_text(package_root / "pipeline_direct_arbitration.py")
    pipeline_main_src = _read_text(package_root / "pipeline.py")
    pipeline_route_src = _read_text(package_root / "pipeline_route_stage.py")
    transfer_execution_src = _read_text(package_root / "pipeline_transfer_execution.py")
    project_assembly_src = _read_text(package_root / "pipeline_project_assembly.py")
    reveal_persistence_src = _read_text(package_root / "pipeline_reveal_persistence.py")
    gui_src = _read_text(package_root / "gui_qt.py")
    mode_cross_import_violations = _mode_package_cross_import_violations(package_root)
    shared_renderer_import_violations = _shared_renderer_import_violations(package_root)
    private_file_checks = _active_mode_private_file_checks(package_root)
    cfg_probe = PipelineConfig()

    checks = {
        "five_active_modes": len(ACTIVE_MODE_ORDER) == 5,
        "auto_is_legacy_only": "auto" in LEGACY_MODE_ORDER and contracts["auto"].orchestrator and sum(int(x.orchestrator) for x in contracts.values()) == 1,
        "direct_is_pixel_only": contracts["direct_patch"].direct and not contracts["direct_patch"].reletter and not contracts["direct_patch"].may_render_text,
        "mask_is_pixel_only": contracts["mask_replace"].mask_replace and not contracts["mask_replace"].reletter and not contracts["mask_replace"].may_render_text,
        "mask_forbids_ocr": not contracts["mask_replace"].may_use_ocr and mode_ocr_execution_violations("mask_replace", {"ocr_source": "hit"}) == ["ocr_source"],
        "direct_forbids_ocr": not contracts["direct_patch"].may_use_ocr and mode_ocr_execution_violations("direct_patch", {"ocr_source": "miss"}) == ["ocr_source"],
        "reveal_optional_ocr_capability": (
            not contracts["transparent_bubble_reveal"].may_use_ocr
            and not contracts["aligned_overlay_reveal"].may_use_ocr
            and contracts["transparent_bubble_reveal"].may_use_presence_ocr
            and contracts["aligned_overlay_reveal"].may_use_presence_ocr
        ),
        "reletter_is_text_only": contracts["reletter"].reletter and not contracts["reletter"].direct and not contracts["reletter"].mask_replace,
        "hybrid_is_explicit_combination": contracts["hybrid"].reletter and contracts["hybrid"].mask_replace and contracts["hybrid"].may_fallback_to_reletter,
        "reveal_routes_are_isolated": contracts["transparent_bubble_reveal"].explicit_isolated_route and contracts["aligned_overlay_reveal"].explicit_isolated_route,
        "mode_artifact_owners_unique": not duplicate_artifacts,
        "review_inputs_unique": len(set(MODE_OWNED_REVIEW_INPUTS)) == len(MODE_OWNED_REVIEW_INPUTS),
        "direct_runtime_rejects_reletter": mode_execution_violations("direct_patch", reletter_used=True) == ["reletter"],
        "mask_runtime_rejects_direct": mode_execution_violations("mask_replace", direct_used=True) == ["direct"],
        "reletter_runtime_rejects_mask": mode_execution_violations("reletter", mask_used=True) == ["mask_replace"],
        "hybrid_runtime_allows_mask_plus_reletter": not mode_execution_violations("hybrid", mask_used=True, reletter_used=True),
        "auto_runtime_allows_direct_or_mask": not mode_execution_violations("auto", direct_used=True) and not mode_execution_violations("auto", mask_used=True),
        "page_single_writer_guard_available": callable(getattr(PageRunGuard, "acquire", None)) and callable(getattr(PageRunGuard, "release", None)),
        "orphan_temp_cleanup_available": callable(cleanup_orphan_temp_files),
        "workspace_integrity_validator_available": callable(validate_page_workspace),
        "resume_contract_is_v3": str(RESUME_SCHEMA) == "folirina-page-resume-v3",
        "run_receipt_boundary_available": callable(write_run_receipt) and callable(validate_run_receipt),
        "native_direct_has_no_vendor_bridge": not (Path(__file__).resolve().parent / "direct_v234_bridge.py").exists(),
        "paddle_doc_runtime_isolated_from_classic_ocr": paddle_ocr_runtime_root() != paddle_doc_runtime_root(),
        "progressive_module_boundaries_pass": bool(module_audit.get("pass")),
        # v2.3.32: mode isolation must cover renderer policy dependencies, not only
        # entrypoints/artifacts. These static guards make a future accidental
        # config/helper re-share fail the release audit immediately.
        "hybrid_mask_config_is_private": (
            cfg_probe.hybrid.mask is not cfg_probe.mask_replace
            and str(getattr(cfg_probe.hybrid.mask, "renderer_owner", "")) == "hybrid"
            and str(getattr(cfg_probe.mask_replace, "renderer_owner", "")) == "mask_replace"
        ),
        "pipeline_resolves_hybrid_private_mask_config": (
            "def _resolve_private_pixel_stage" in pipeline_src
            and "from .modes.hybrid import pixel_stage" in pipeline_src
            and "return config.hybrid.mask" in _read_text(package_root / "modes" / "hybrid" / "pixel_stage.py")
        ),
        "direct_clarity_is_mode_owned": (
            "from .source_clarity import enhance_white_source_patch" in direct_overlay_src
            and (package_root / "modes" / "direct_patch" / "source_clarity.py").exists()
        ),
        "pixel_modes_have_private_transfer_ops": all(
            (package_root / "modes" / name / "transfer_ops.py").exists() for name in ACTIVE_PIXEL_MODE_PACKAGES
        ),
        "pixel_modes_have_private_raster_policy": all(
            (package_root / "modes" / name / "raster_policy.py").exists() for name in ACTIVE_PIXEL_MODE_PACKAGES
        ),
        "pixel_modes_have_private_source_clarity": all(
            (package_root / "modes" / name / "source_clarity.py").exists() for name in ACTIVE_PIXEL_MODE_PACKAGES
        ),
        "active_pixel_stage_uses_dispatcher_only": (
            "def _resolve_private_pixel_stage" in pipeline_src
            and "from .mask_transfer import" not in pipeline_src
            and "from .modes.direct_patch import pixel_stage" in pipeline_src
            and "from .modes.mask_replace import pixel_stage" in pipeline_src
            and "from .modes.hybrid import pixel_stage" in pipeline_src
            and "from .modes.reletter import pixel_stage" in pipeline_src
            and "transfer_rigid_container_rasters" not in pipeline_src
            and "transfer_paired_diff_regions" not in pipeline_src
        ),
        "active_mode_packages_have_no_cross_imports": not mode_cross_import_violations,
        "legacy_shared_mask_transfer_not_used_by_active_stage": "from .mask_transfer import" not in pipeline_src,
        "all_active_mode_private_files_present": all(private_file_checks.values()),
        "active_modes_do_not_import_shared_renderer_modules": not shared_renderer_import_violations,
        "direct_arbitration_uses_private_container_renderer": (
            "from .modes.direct_patch.container_renderer import build_source_direct_container_plan" in direct_arbitration_src
            and "from .direct_containers import build_source_direct_container_plan" not in direct_arbitration_src
        ),
        "pipeline_uses_mode_private_reletter_executors": (
            "from .modes.reletter.executor import ReletterExecutor as ReletterModeExecutor" in pipeline_main_src
            and "from .modes.hybrid.executor import ReletterExecutor as HybridReletterExecutor" in pipeline_main_src
            and "from .reletter_executor import" not in pipeline_main_src
        ),
        "aligned_route_uses_private_renderer": (
            "from .modes.aligned_overlay_reveal.renderer import" in pipeline_route_src
            and "from .modes.aligned_overlay_reveal.hole_renderer import" in pipeline_route_src
            and "from .aligned_overlay_reveal import" not in pipeline_route_src
            and "from .aligned_overlay_reveal_mode import" not in pipeline_route_src
        ),
        "gui_ocr_editor_dispatches_private_mode_modules": (
            "from .modes.reletter import ocr_edit_blocks as _reletter_ocr_edit_blocks" in gui_src
            and "from .modes.hybrid import ocr_edit_blocks as _hybrid_ocr_edit_blocks" in gui_src
            and "from .ocr_edit_blocks import" not in gui_src
            and "from .ocr_edit_render import" not in gui_src
        ),
        "pixel_stage_has_no_shared_reletter_renderer_helpers": (
            "from .reletter_binding import" not in pipeline_src
            and "from .reletter_layout import" not in pipeline_src
            and "from .mask_transfer import" not in pipeline_src
        ),
        "transfer_execution_dispatches_private_layout_policy": (
            "def _resolve_private_execution_stage" in transfer_execution_src
            and "from .modes.hybrid import execution_stage" in transfer_execution_src
            and "from .modes.reletter import execution_stage" in transfer_execution_src
            and "from .reletter_layout import" not in transfer_execution_src
            and "from .layout_policy import" in _read_text(package_root / "modes" / "hybrid" / "execution_stage.py")
            and "from .layout_policy import" in _read_text(package_root / "modes" / "reletter" / "execution_stage.py")
        ),
        "aligned_persistence_uses_private_result_model": (
            "from .modes.aligned_overlay_reveal.core import AlignedOverlayResult" in reveal_persistence_src
            and "from .aligned_overlay_reveal import AlignedOverlayResult" not in reveal_persistence_src
        ),
        "transfer_execution_dispatches_private_runtime_pixels": (
            "def _resolve_private_execution_stage" in transfer_execution_src
            and "from .inpainting import" not in transfer_execution_src
            and "from .masking import" not in transfer_execution_src
            and "from .lettering import" not in transfer_execution_src
            and "from .pipeline_ocr_cleanup import" not in transfer_execution_src
            and "from .modes.direct_patch import execution_stage" in transfer_execution_src
            and "from .modes.mask_replace import execution_stage" in transfer_execution_src
            and "from .modes.hybrid import execution_stage" in transfer_execution_src
            and "from .modes.reletter import execution_stage" in transfer_execution_src
            and "from . import masking_ops, inpainting_ops" in _read_text(package_root / "modes" / "direct_patch" / "execution_stage.py")
            and "from . import masking_ops, inpainting_ops" in _read_text(package_root / "modes" / "mask_replace" / "execution_stage.py")
            and "from . import masking_ops, inpainting_ops, lettering_ops, ocr_cleanup" in _read_text(package_root / "modes" / "hybrid" / "execution_stage.py")
            and "from . import masking_ops, inpainting_ops, lettering_ops, ocr_cleanup" in _read_text(package_root / "modes" / "reletter" / "execution_stage.py")
        ),
        "hybrid_has_unique_owned_artifacts": bool(get_mode_contract("hybrid")) and set((package_root / "modes" / "hybrid" / "definition.py").read_text(encoding="utf-8").split()) and all(
            name in (package_root / "modes" / "hybrid" / "definition.py").read_text(encoding="utf-8")
            for name in ("hybrid_transfer_layer.png", "hybrid_transfer_mask.png", "hybrid_transfer.json", "hybrid_text_layer.png")
        ),
        "reletter_has_unique_owned_artifacts": all(
            name in (package_root / "modes" / "reletter" / "definition.py").read_text(encoding="utf-8")
            for name in ("reletter_text_layer.png", "reletter.json")
        ),
        "hybrid_metadata_does_not_claim_mask_replace": (
            'mode in {"auto", "mask_replace"}' in project_assembly_src
            and '"mask_records": transfer_records_to_dict(active_records) if mode == "hybrid" else []' in project_assembly_src
        ),
        "reletter_metadata_is_mode_scoped": (
            '"used": bool(mode == "reletter" and lettering)' in project_assembly_src
            and '"editable_regions": list(composition.reletter_editable_regions) if mode == "reletter" else []' in project_assembly_src
        ),
        "active_renderers_do_not_import_shared_white_clarity": (
            "from ...white_source_clarity" not in direct_overlay_src
            and "from .white_source_clarity" not in mask_transfer_src
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "contracts": {k: v.to_dict() for k, v in contracts.items()},
        "duplicate_artifacts": duplicate_artifacts,
        "artifact_owners": artifact_owner,
        "mode_owned_review_inputs": list(MODE_OWNED_REVIEW_INPUTS),
        "module_boundaries": module_audit,
        "mode_cross_import_violations": mode_cross_import_violations,
        "shared_renderer_import_violations": shared_renderer_import_violations,
        "private_file_checks": private_file_checks,
    }


def main() -> int:
    import json
    report = run_architecture_audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
