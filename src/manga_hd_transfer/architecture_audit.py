from __future__ import annotations

"""Fast architecture checks for transfer-mode isolation.

This is intentionally lightweight enough to run before packaging a release. It
checks the contracts that are easy to regress when adding a new detector/editor.
The full pixel regressions remain in ``selftest.py``.
"""

from .workspace_guard import PageRunGuard, cleanup_orphan_temp_files
from .workspace_integrity import validate_page_workspace
from .module_contracts import audit_module_contracts

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
        "paddle_doc_runtime_isolated_from_classic_ocr": paddle_ocr_runtime_root() != paddle_doc_runtime_root(),
        "progressive_module_boundaries_pass": bool(module_audit.get("pass")),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "contracts": {k: v.to_dict() for k, v in contracts.items()},
        "duplicate_artifacts": duplicate_artifacts,
        "artifact_owners": artifact_owner,
        "mode_owned_review_inputs": list(MODE_OWNED_REVIEW_INPUTS),
        "module_boundaries": module_audit,
    }


def main() -> int:
    import json
    report = run_architecture_audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
