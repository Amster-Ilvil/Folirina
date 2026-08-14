from __future__ import annotations

"""Small, dependency-light provider registry for precise transfer extensions.

The core pipeline owns the contracts and third-party inspired providers register
behind them.  This keeps source detectors, registration refiners, mask refiners,
and QA checks independent instead of growing backend-specific branches in the
main pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

Provider = Callable[..., Any]


@dataclass
class ProviderRegistry:
    _providers: dict[str, dict[str, Provider]] = field(default_factory=lambda: {
        "source_detector": {},
        "registration_refiner": {},
        "mask_refiner": {},
        "qa_check": {},
    })

    def register(self, category: str, name: str, provider: Provider, *, replace: bool = False) -> Provider:
        category = str(category).strip().lower()
        name = str(name).strip().lower()
        if category not in self._providers:
            self._providers[category] = {}
        if not replace and name in self._providers[category]:
            raise KeyError(f"Provider already registered: {category}:{name}")
        self._providers[category][name] = provider
        return provider

    def get(self, category: str, name: str) -> Provider | None:
        return self._providers.get(str(category).strip().lower(), {}).get(str(name).strip().lower())

    def names(self, category: str) -> list[str]:
        return sorted(self._providers.get(str(category).strip().lower(), {}))

    def snapshot(self) -> dict[str, list[str]]:
        return {k: sorted(v) for k, v in self._providers.items()}


REGISTRY = ProviderRegistry()


def register_provider(category: str, name: str, *, replace: bool = False):
    def deco(fn: Provider) -> Provider:
        REGISTRY.register(category, name, fn, replace=replace)
        return fn
    return deco

@register_provider("qa_check", "source_direct_invariants")
def source_direct_invariants(plan) -> list[str]:
    """Non-negotiable precise-transfer invariants used before OCR is skipped."""
    if plan is None:
        return ["missing_plan"]
    d=getattr(plan,"diagnostics",{}) or {}
    issues=[]
    if int(d.get("border_pixels_written",-1)) != 0:
        issues.append("alignment_border_was_written")
    if bool(d.get("ocr_used",True)):
        issues.append("ocr_used_on_source_direct_route")
    if bool(d.get("target_bubble_matching_used",True)):
        issues.append("target_bubble_matching_used")
    result=getattr(plan,"result",None)
    for r in (getattr(result,"records",[]) if result is not None else []):
        if not bool(getattr(r,"content_complete",False)):
            issues.append(f"content_incomplete:{getattr(r,'source_bubble_id','?')}")
    return issues
