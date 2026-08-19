from __future__ import annotations

"""Registration and same-page verification service for the page pipeline.

This module owns the cache-aware registration step and the OCR-free same-page
safety gate.  It deliberately has no dependency on ``pipeline.py`` or any
transfer renderer so it can be tested and reused independently.
"""

from typing import Any

import numpy as np

from .cache import PageStageCache, registration_stage_signature
from .models import PagePair, RegistrationResult
from .page_pairing import PagePairingCheck, verify_registered_page_pair
from .registration import register_images


def register_page(
    source: np.ndarray, target: np.ndarray, registration_config: Any
) -> RegistrationResult:
    """Run one uncached registration for alternate SOURCE arbitration."""
    return register_images(source, target, registration_config)


def register_page_cached(
    pair: PagePair,
    source: np.ndarray,
    target: np.ndarray,
    registration_config: Any,
    *,
    cache: PageStageCache,
    cache_enabled: bool,
    stats: dict[str, str],
) -> RegistrationResult:
    """Return cached page registration or compute/store it exactly once."""
    sig = registration_stage_signature(pair, registration_config)
    registration = cache.load_registration(sig) if cache_enabled else None
    if registration is None:
        registration = register_images(source, target, registration_config)
        if cache_enabled:
            cache.save_registration(sig, registration)
        stats["registration"] = "miss"
    else:
        stats["registration"] = "hit"
    return registration


def verify_same_page(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    pairing_config: Any,
) -> PagePairingCheck:
    """Run the destructive-safety same-page check, or the configured identity fallback."""
    if bool(getattr(pairing_config, "same_page_precheck_enabled", True)):
        return verify_registered_page_pair(
            source,
            target,
            registration,
            max_side=int(getattr(pairing_config, "same_page_max_side", 720)),
            min_confidence=float(getattr(pairing_config, "same_page_min_confidence", 0.72)),
            min_valid_ratio=float(getattr(pairing_config, "same_page_min_valid_ratio", 0.45)),
        )
    return PagePairingCheck(
        True,
        float(registration.confidence),
        float(registration.confidence),
        float(registration.confidence),
        float(registration.confidence),
        {"disabled": True, "ocr_used": False},
    )


def verify_same_page_strict(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    pairing_config: Any,
) -> PagePairingCheck:
    """Always run the OCR-free same-page check (used for alternate SOURCE evidence)."""
    return verify_registered_page_pair(
        source,
        target,
        registration,
        max_side=int(getattr(pairing_config, "same_page_max_side", 720)),
        min_confidence=float(getattr(pairing_config, "same_page_min_confidence", 0.72)),
        min_valid_ratio=float(getattr(pairing_config, "same_page_min_valid_ratio", 0.45)),
    )


__all__ = ["register_page", "register_page_cached", "verify_same_page", "verify_same_page_strict"]
