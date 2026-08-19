from __future__ import annotations

"""Static dependency contracts for progressively extracted modules.

The project is intentionally being decomposed in small, regression-tested steps.
This file prevents newly extracted domain/workspace modules from drifting back
into imports of the monolithic GUI or pipeline entry points.
"""

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModuleContract:
    module: str
    layer: str
    forbidden_imports: tuple[str, ...]
    purpose: str


MODULE_CONTRACTS: tuple[ModuleContract, ...] = (
    ModuleContract(
        "pipeline_page_flow", "application-page-orchestration",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Single-page stage sequencing between preparation, route, OCR, matching, render composition and persistence",
    ),
    ModuleContract(
        "pipeline_run_lifecycle", "application-lifecycle",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Single-writer page transaction, snapshot restore, run trace and post-run workspace integrity",
    ),
    ModuleContract(
        "pipeline_passthrough", "application-persistence",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Unchanged TARGET persistence for page-manager/content/route rejection passthrough results",
    ),
    ModuleContract(
        "pipeline_route_stage", "application-route-orchestration",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Post-registration explicit Reveal, Direct SOURCE arbitration and Auto strategy selection",
    ),
    ModuleContract(
        "pipeline_content_gate", "application-policy",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "OCR-backed strong negative evidence gate for preserving pages without transferable SOURCE text",
    ),
    ModuleContract(
        "app_logging", "infrastructure",
        ("gui_qt", "pipeline", "review_apply", "mask_transfer"),
        "Standard-library application log routing and uncaught exception capture before GUI import",
    ),
    ModuleContract(
        "platform_support", "infrastructure",
        ("gui_qt", "pipeline", "review_apply", "mask_transfer"),
        "Side-effect-free desktop platform labels and capability hints",
    ),
    ModuleContract(
        "pipeline_transfer_execution", "application-render-orchestration",
        ("pipeline", "gui_qt", "review_apply"),
        "Stable Direct/Mask/Reletter renderer sequencing, completion/fallback coordination, inpaint and mode QA",
    ),
    ModuleContract(
        "pipeline_transfer_composition", "application-composition",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Post-render transfer audit, review/edit metadata projection and completion display composition",
    ),
    ModuleContract(
        "pipeline_artifact_export", "application-persistence",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Lossless page artifact/layer/debug/editable-bundle export after renderer completion",
    ),
    ModuleContract(
        "pipeline_project_assembly", "application-assembly",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "PageProject metadata schema assembly plus final mode-contract QA persistence",
    ),
    ModuleContract(
        "pipeline_direct_arbitration", "application-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Primary/secondary Direct SOURCE plan construction, evidence scoring and safe arbitration",
    ),
    ModuleContract(
        "pipeline_reveal_persistence", "application-persistence",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Aligned/transparent Reveal artifact persistence, QA and mode-isolation metadata",
    ),
    ModuleContract(
        "book_orchestration", "application-orchestration",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Book planning, resume admission, checkpoints, progress and final manifests",
    ),
    ModuleContract(
        "pipeline_page_prep", "application-stage",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Page workspace cleanup, PageMark normalization, image loading and SOURCE candidate discovery",
    ),
    ModuleContract(
        "pipeline_registration_service", "application-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Cache-aware page registration and OCR-free same-page verification",
    ),
    ModuleContract(
        "remake_pairing", "domain-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer", "registration"),
        "Optional model-free AKAZE/RANSAC second-opinion evidence for smart page pairing",
    ),
    ModuleContract(
        "layout_evidence", "application-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer", "direct_containers", "reletter_executor"),
        "Shared Koharu Layout evidence normalization plus canonical SOURCE/TARGET memory/disk cache for all modes",
    ),
    ModuleContract(
        "koharu_flow_cells", "domain-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer", "lettering", "reletter_executor"),
        "Optional Reletter-only non-overlapping joined-balloon flow-cell partition with topology-gated neck recovery",
    ),
    ModuleContract(
        "pipeline_ocr_service", "application-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "OCR backend construction, caching and source rectification without transfer ownership",
    ),
    ModuleContract(
        "pipeline_bubble_service", "application-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Bubble detector dispatch and cache restoration independent from page orchestration",
    ),
    ModuleContract(
        "pipeline_paired_diff_service", "application-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Cache-aware paired-difference geometry gate before OCR/rendering",
    ),
    ModuleContract(
        "pipeline_text_stage", "application-stage",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Lazy OCR and bubble orchestration with Reletter region routing",
    ),
    ModuleContract(
        "pipeline_match_service", "application-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Text-unit construction, paired identity matching and auto-acceptance policy",
    ),
    ModuleContract(
        "studio_project_page", "ui-page",
        ("gui_qt", "pipeline", "review_apply", "mask_transfer"),
        "Page Manager and side-by-side page preview isolated from StudioWindow",
    ),
    ModuleContract(
        "studio_model_page", "ui-page",
        ("gui_qt", "pipeline", "review_apply", "mask_transfer"),
        "Recognition/registration/model-management page with workers delegated",
    ),
    ModuleContract(
        "studio_export_page", "ui-page",
        ("gui_qt", "pipeline", "review_apply", "mask_transfer"),
        "Publication/export page isolated from StudioWindow",
    ),
    ModuleContract(
        "gui_theme", "ui-theme",
        ("gui_qt", "pipeline", "review_apply", "mask_transfer"),
        "Shared palette and Qt stylesheet without workflow dependencies",
    ),
    ModuleContract(
        "gui_components", "ui-component",
        ("gui_qt", "pipeline", "review_apply", "mask_transfer"),
        "Reusable lightweight Qt views/cards independent from Studio workflow state",
    ),
    ModuleContract(
        "pipeline_worker", "ui-worker",
        ("gui_qt",),
        "Long-running page/book worker isolated from Studio widgets while preserving pipeline semantics",
    ),
    ModuleContract(
        "review_common", "review-domain",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer", "review_manual_force", "review_manual_effect"),
        "Shared review schema/image helpers with no workflow ownership",
    ),
    ModuleContract(
        "review_manual_force", "application-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Page-guarded reviewer-activated force-transfer service",
    ),
    ModuleContract(
        "review_manual_effect", "application-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Manual effect/reveal compositing service independent from the review dispatcher",
    ),
    ModuleContract(
        "source_candidate_service", "application-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Sidecar/secondary SOURCE discovery and monochrome-vs-colour rendition classification",
    ),
    ModuleContract(
        "transfer_policy", "application-policy",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Transfer matching summaries, replace-translation evidence export and review-only overlay policy",
    ),
    ModuleContract(
        "mask_quality", "domain-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer", "direct_containers"),
        "Renderer-independent sharpness, pixel enhancement and super-resolution policy",
    ),
    ModuleContract(
        "mask_warp", "domain-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer", "direct_containers"),
        "Renderer-independent source warp, local subpixel refinement and photo-pair salvage geometry",
    ),
    ModuleContract(
        "mask_photo_text", "domain-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer", "direct_containers"),
        "Renderer-independent photographed text normalization and ink reconstruction",
    ),
    ModuleContract(
        "review_target_layer", "application-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Page-guarded TARGET erase/restore review operations isolated from the review dispatcher",
    ),
    ModuleContract(
        "mask_transfer_models", "domain-model",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer", "direct_containers"),
        "Renderer-independent transfer record/result dataclasses shared by Mask and Direct routes",
    ),
    ModuleContract(
        "mask_transfer_audit", "domain-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer", "direct_containers"),
        "Renderer-independent transfer record serialization, summary counters and review-row projection",
    ),
    ModuleContract(
        "mask_content_audit", "domain-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer", "direct_containers"),
        "Renderer-independent content completeness audit, bounded repair and transfer triage decisions",
    ),
    ModuleContract(
        "mask_raster_primitives", "domain",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer", "direct_containers"),
        "Pure mask raster primitives: boundary cleanup, glyph-footprint rescue and alpha/write-envelope transforms",
    ),
    ModuleContract(
        "mask_geometry", "domain",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer", "direct_containers"),
        "Renderer-independent mask geometry, coverage scoring and bubble correspondence",
    ),
    ModuleContract(
        "mask_selection", "domain",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer", "direct_containers"),
        "Renderer-independent mask route/candidate eligibility decisions",
    ),
    ModuleContract(
        "reletter_executor", "application-service",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Reletter-only OCR execution over already-paired SOURCE/TARGET regions",
    ),
    ModuleContract(
        "gui_workers", "ui-worker",
        ("gui_qt", "pipeline", "review_apply", "mask_transfer"),
        "Qt background workers for page actions, probes, downloads and runtime preparation",
    ),
    ModuleContract(
        "reletter_binding", "domain",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "SOURCE/TARGET bubble identity, subregion binding and Region OCR normalization",
    ),
    ModuleContract(
        "reletter_layout", "domain",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Reletter layout hints and source-layout projection",
    ),
    ModuleContract(
        "transfer_completion", "domain",
        ("pipeline", "gui_qt", "review_apply"),
        "Mask-transfer completion candidate filtering",
    ),
    ModuleContract(
        "page_review_state", "workspace",
        ("pipeline", "gui_qt", "review_apply"),
        "Page-local review ownership and stale derived-artifact cleanup",
    ),
    ModuleContract(
        "studio_state", "application-state",
        ("pipeline", "gui_qt", "review_apply", "mask_transfer"),
        "Qt-independent Studio state/default configuration",
    ),
)


def _relative_import_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                names.append(str(node.module).split(".")[0])
            elif node.module:
                names.append(str(node.module))
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return names


def audit_module_contracts(package_root: str | Path | None = None) -> dict:
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parent
    rows: list[dict] = []
    for contract in MODULE_CONTRACTS:
        path = root / f"{contract.module}.py"
        imports = _relative_import_names(path) if path.exists() else []
        violations = sorted({
            forbidden
            for forbidden in contract.forbidden_imports
            if any(name == forbidden or name.endswith(f".{forbidden}") for name in imports)
        })
        rows.append({
            "module": contract.module,
            "layer": contract.layer,
            "purpose": contract.purpose,
            "exists": path.exists(),
            "imports": imports,
            "forbidden_imports": list(contract.forbidden_imports),
            "violations": violations,
            "pass": bool(path.exists() and not violations),
        })
    return {"pass": all(row["pass"] for row in rows), "modules": rows}


__all__ = ["ModuleContract", "MODULE_CONTRACTS", "audit_module_contracts"]
