from __future__ import annotations

import numpy as np

from .schema import SemanticLayoutResult
from .paddle_layout import analyze_with_paddlex
from .heuristic_layout import analyze_heuristic


def analyze_semantic_layout(image: np.ndarray, cfg, *, role: str = "target") -> SemanticLayoutResult:
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return SemanticLayoutResult(False, "disabled", diagnostics={"status": "disabled", "role": role})
    backend = str(getattr(cfg, "backend", "auto") or "auto").strip().lower()
    strategy = str(getattr(cfg, "strategy", "auto") or "auto").strip().lower()
    if backend in {"auto", "pp_doclayout_v3", "paddle"}:
        result = analyze_with_paddlex(image, cfg, strategy=strategy)
        if result.available:
            result.diagnostics["role"] = role
            return result
        if backend not in {"auto"} or not bool(getattr(cfg, "fallback_heuristic", True)):
            result.diagnostics["role"] = role
            return result
        fallback = analyze_heuristic(image, cfg, strategy=strategy)
        fallback.diagnostics["paddle_fallback_reason"] = result.diagnostics
        fallback.diagnostics["role"] = role
        return fallback
    result = analyze_heuristic(image, cfg, strategy=strategy)
    result.diagnostics["role"] = role
    return result
