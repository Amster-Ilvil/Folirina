from __future__ import annotations

"""Unified primary/auxiliary detector policy.

The policy is intentionally small and runtime-agnostic.  It decides *which*
providers are eligible and *when* auxiliaries may run; individual detector
adapters keep ownership of inference and post-processing.
"""

from typing import Any, Iterable

STRATEGY_PRIMARY_ONLY = "primary_only"
STRATEGY_CONDITIONAL = "primary_conditional_aux"
STRATEGY_ALWAYS = "primary_plus_aux"

VALID_STRATEGIES = {
    STRATEGY_PRIMARY_ONLY,
    STRATEGY_CONDITIONAL,
    STRATEGY_ALWAYS,
}
VALID_PRIMARY = {"koharu_layout", "mangalens", "rtdetr_v2"}
VALID_AUXILIARY = {"geometry_white", "mangalens", "rtdetr_v2", "ysg_obb", "koharu_layout", "sidecar", "ctd_sidecar", "sam2"}
EXPENSIVE = {"mangalens", "rtdetr_v2", "ysg_obb", "sam2"}


def detector_strategy(cfg: Any) -> str:
    value = str(getattr(cfg, "detector_strategy", STRATEGY_CONDITIONAL) or STRATEGY_CONDITIONAL).strip().lower()
    return value if value in VALID_STRATEGIES else STRATEGY_CONDITIONAL


def primary_detector(cfg: Any) -> str:
    value = str(getattr(cfg, "primary_detector", "koharu_layout") or "koharu_layout").strip().lower()
    return value if value in VALID_PRIMARY else "koharu_layout"


def auxiliary_detectors(cfg: Any) -> list[str]:
    primary = primary_detector(cfg)
    out: list[str] = []
    for raw in list(getattr(cfg, "auxiliary_detectors", []) or []):
        value = str(raw or "").strip().lower()
        if value not in VALID_AUXILIARY or value == primary or value in out:
            continue
        out.append(value)
    # Backward compatibility: v2.0.90 and older projects only stored
    # ``bubbles.backend``.  Preserve an explicitly selected legacy fallback
    # until the new GUI writes the detector-policy fields.
    legacy = str(getattr(cfg, "backend", "") or "").strip().lower()
    legacy_map = {"seeded_white": "geometry_white", "mangalens": "mangalens", "rtdetr_v2": "rtdetr_v2", "ysg_obb": "ysg_obb", "sidecar": "sidecar", "ctd_sidecar": "ctd_sidecar", "koharu_layout": "koharu_layout", "sam2": "sam2"}
    legacy = legacy_map.get(legacy, "")
    if legacy and legacy != primary and legacy not in out:
        out.append(legacy)
    return out


def sam2_refine_enabled(cfg: Any) -> bool:
    return bool(getattr(cfg, "sam2_refine_enabled", False))


def koharu_is_primary(cfg: Any) -> bool:
    return primary_detector(cfg) == "koharu_layout"


def policy_uses_koharu(cfg: Any) -> bool:
    if koharu_is_primary(cfg):
        return True
    return detector_strategy(cfg) != STRATEGY_PRIMARY_ONLY and "koharu_layout" in auxiliary_detectors(cfg)


def should_run_auxiliaries(cfg: Any, *, primary_sufficient: bool) -> bool:
    strategy = detector_strategy(cfg)
    if strategy == STRATEGY_PRIMARY_ONLY:
        return False
    if strategy == STRATEGY_ALWAYS:
        return True
    return not bool(primary_sufficient)


def source_provider_name(name: str) -> str:
    """Map UI/logical detector names to SOURCE detector registry providers."""
    value = str(name or "").strip().lower()
    if value == "geometry_white":
        return "pseudo_text_barrier"
    return value


def source_auxiliary_providers(cfg: Any, *, include_refiner: bool = True) -> list[str]:
    out: list[str] = []
    for name in auxiliary_detectors(cfg):
        provider = source_provider_name(name)
        if provider and provider not in out:
            out.append(provider)
    if include_refiner and sam2_refine_enabled(cfg) and "sam2" not in out and primary_detector(cfg) != "sam2":
        out.append("sam2")
    return out


def bubble_auxiliary_backends(cfg: Any) -> list[str]:
    """Map logical auxiliaries to generic page bubble detector backends."""
    out: list[str] = []
    for name in auxiliary_detectors(cfg):
        # YSG OBB is a semantic/open-text auxiliary, not a generic bubble pass;
        # CTD sidecar is SOURCE-only.  Keeping them out here prevents duplicate
        # full-page inference and accidental treatment of open text as a rigid bubble.
        if name in {"ysg_obb", "ctd_sidecar"}:
            continue
        backend = "seeded_white" if name == "geometry_white" else name
        if backend not in out:
            out.append(backend)
    if sam2_refine_enabled(cfg) and "sam2" not in out:
        out.append("sam2")
    return out


def transparent_auxiliary_backends(cfg: Any) -> list[str]:
    """Transparent Reveal uses text-contour geometry before broad white fill."""
    out: list[str] = []
    for name in auxiliary_detectors(cfg):
        if name == "geometry_white":
            for backend in ("target_text_contour", "seeded_white"):
                if backend not in out:
                    out.append(backend)
            continue
        if name == "sidecar":
            # Transparent Reveal has no sidecar target-mask contract.  Keep it
            # unavailable here rather than silently interpreting a SOURCE file.
            continue
        if name not in out:
            out.append(name)
    if sam2_refine_enabled(cfg) and "sam2" not in out:
        out.append("sam2")
    return out


def expensive_provider(name: str) -> bool:
    return str(name or "").strip().lower() in EXPENSIVE


def configured_runtime_detectors(cfg: Any) -> list[str]:
    """Return only model-backed detectors that can actually be scheduled."""
    names: list[str] = [primary_detector(cfg)]
    if detector_strategy(cfg) != STRATEGY_PRIMARY_ONLY:
        names.extend(auxiliary_detectors(cfg))
        if sam2_refine_enabled(cfg):
            names.append("sam2")
    out: list[str] = []
    for name in names:
        if name in {"koharu_layout", "mangalens", "rtdetr_v2", "ysg_obb", "sam2"} and name not in out:
            out.append(name)
    return out


def uncovered_blocks(blocks: Iterable[Any], bubbles: Iterable[Any]) -> int:
    """Cheap primary-sufficiency signal for OCR-capable bubble routes."""
    assigned: set[str] = set()
    for bubble in bubbles:
        assigned.update(str(x) for x in list(getattr(bubble, "block_ids", []) or []))
    total = 0
    for block in blocks:
        bid = str(getattr(block, "id", ""))
        if bid and bid not in assigned:
            total += 1
    return total


__all__ = [
    "STRATEGY_PRIMARY_ONLY", "STRATEGY_CONDITIONAL", "STRATEGY_ALWAYS",
    "detector_strategy", "primary_detector", "auxiliary_detectors",
    "sam2_refine_enabled", "koharu_is_primary", "policy_uses_koharu",
    "should_run_auxiliaries", "source_auxiliary_providers",
    "bubble_auxiliary_backends", "transparent_auxiliary_backends",
    "expensive_provider", "configured_runtime_detectors", "uncovered_blocks",
]
