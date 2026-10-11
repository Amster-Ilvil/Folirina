from __future__ import annotations


def bind_mode_runtime_config(config):
    """Return a runtime copy whose legacy aliases point at mode-owned state.

    The established pipeline may continue consuming ``mask_replace`` and
    ``lettering`` as common engine interfaces. Hybrid/Reletter receive private
    snapshots before execution, so tuning the user-facing Mask/Reletter mode does
    not silently alter another mode. The source config object is never mutated.
    """
    runtime = config.model_copy(deep=True) if hasattr(config, "model_copy") else config.copy(deep=True)
    mode = str(runtime.transfer.mode or "").strip().lower()
    if mode == "hybrid":
        runtime.mask_replace = runtime.hybrid.mask.model_copy(deep=True)
        runtime.lettering = runtime.hybrid.lettering.model_copy(deep=True)
    elif mode == "reletter":
        runtime.mask_replace = runtime.reletter.candidates.model_copy(deep=True)
        runtime.lettering = runtime.reletter.lettering.model_copy(deep=True)
        # Reletter is OCR-authoritative: silently substituting a different OCR
        # engine changes the text that gets re-typeset while still presenting
        # the result as the user's selected backend.  The mode specification
        # explicitly declares fallback_policy="none", so enforce that on the
        # runtime copy only.  Direct/Mask/Hybrid retain the global optional
        # backend soft-fallback policy.
        runtime.ocr.soft_fail_fallback_backend = "none"
    return runtime
