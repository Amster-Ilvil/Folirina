from __future__ import annotations

"""Deterministic PaddleOCR download-source selection.

This module is intentionally small and side-effect free.  It exists so model
source policy is shared by explicit preheat/download and the persistent OCR
worker startup, rather than each call site hard-coding its own mirror.
"""

import os
from typing import Mapping

PADDLE_MODEL_SOURCES: tuple[str, ...] = ("modelscope", "bos", "aistudio", "huggingface")
PADDLE_MODEL_SOURCE_LABELS: dict[str, str] = {
    "auto": "自动重试",
    "modelscope": "ModelScope",
    "bos": "百度 BOS",
    "aistudio": "AIStudio",
    "huggingface": "Hugging Face",
}


def normalize_paddle_model_source(value: str | None) -> str:
    source = str(value or "auto").strip().lower()
    aliases = {
        "": "auto",
        "default": "auto",
        "automatic": "auto",
        "hf": "huggingface",
        "hugging_face": "huggingface",
        "model_scope": "modelscope",
        "baidu": "bos",
        "baidu_bos": "bos",
        "ai_studio": "aistudio",
    }
    source = aliases.get(source, source)
    return source if source in PADDLE_MODEL_SOURCES else "auto"


def paddle_model_source_attempts(preferred: str | None = "auto") -> tuple[str, ...]:
    """Return deterministic, duplicate-free PaddleX model source attempts.

    Explicit source selection is strict.  Auto first honors the app-specific
    override, then an existing PaddleX override, then tries regional sources
    followed by Hugging Face.  All fallback happens before OCR page output.
    """
    normalized = normalize_paddle_model_source(preferred)
    if normalized != "auto":
        return (normalized,)

    ordered: list[str] = []
    for name in ("MHD_PADDLE_MODEL_SOURCE", "PADDLE_PDX_MODEL_SOURCE"):
        source = normalize_paddle_model_source(os.environ.get(name))
        if source != "auto":
            ordered.append(source)
    ordered.extend(PADDLE_MODEL_SOURCES)
    return tuple(dict.fromkeys(ordered))


def paddle_source_environment(source: str | None, base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    normalized = normalize_paddle_model_source(source)
    if normalized == "auto":
        env.pop("PADDLE_PDX_MODEL_SOURCE", None)
    else:
        env["PADDLE_PDX_MODEL_SOURCE"] = normalized
    # Keep the JSONL worker protocol clean and avoid pointless hub telemetry.
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return env
