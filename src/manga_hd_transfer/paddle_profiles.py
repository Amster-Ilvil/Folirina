from __future__ import annotations

"""Paddle OCR/document-parsing engine catalog.

v2.0.44 makes the *engine* explicit instead of presenting every Paddle family as
one ambiguous "PaddleOCR" choice.  Local PP-OCRv6, PaddleOCR-VL 1.6 and
PP-StructureV3 are separate runtime routes with separate cache markers and GUI
choices.  Old v5 server/mobile keys are intentionally migrated to the legacy v5
auto compatibility route so existing project files remain loadable without
surfacing those removed choices again.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PaddleModelProfile:
    key: str
    label: str
    pipeline: str  # ocr|vl|structure
    detection_model: str | None = None
    recognition_model: str | None = None
    ocr_version: str | None = None
    pipeline_version: str | None = None
    supports_japanese: bool = True
    note: str = ""


PADDLE_MODEL_PROFILES: tuple[PaddleModelProfile, ...] = (
    PaddleModelProfile(
        "ppocr_v6_medium", "PaddleOCR v6 Medium（推荐·高质量）", "ocr",
        "PP-OCRv6_medium_det", "PP-OCRv6_medium_rec", "PP-OCRv6", None, True,
        "传统检测+识别 OCR；中/日文统一模型，质量优先。",
    ),
    PaddleModelProfile(
        "ppocr_v6_small", "PaddleOCR v6 Small（推荐·快速）", "ocr",
        "PP-OCRv6_small_det", "PP-OCRv6_small_rec", "PP-OCRv6", None, True,
        "传统检测+识别 OCR；更小更快。",
    ),
    PaddleModelProfile(
        "paddle_vl_16", "PaddleOCR-VL 1.6（文档解析/VLM）", "vl",
        None, None, None, "v1.6", True,
        "独立文档解析产线；不是 PP-OCRv6 的模型档位。",
    ),
    PaddleModelProfile(
        "pp_structure_v3", "PP-StructureV3（版面结构解析）", "structure",
        None, None, "PP-OCRv5", None, True,
        "独立版面结构产线；默认关闭表格/公式/图表等漫画不需要模块。",
    ),
    PaddleModelProfile(
        "legacy_v5_auto", "PP-OCRv5 自动（仅兼容旧项目）", "ocr",
        None, None, "PP-OCRv5", None, True,
        "兼容已有项目；不再在主界面提供 v5 Server/Mobile 档位。",
    ),
    PaddleModelProfile(
        "custom", "自定义 PP-OCR 模型名 / 本地目录", "ocr",
        None, None, None, None, True,
        "只用于传统 PaddleOCR detection/recognition。",
    ),
)

_PROFILE_MAP = {row.key: row for row in PADDLE_MODEL_PROFILES}


def normalize_paddle_model_profile(value: str | None) -> str:
    key = str(value or "legacy_v5_auto").strip().lower()
    aliases = {
        "ppocrv6_medium": "ppocr_v6_medium",
        "v6_medium": "ppocr_v6_medium",
        "paddle_v6_medium": "ppocr_v6_medium",
        "ppocrv6_small": "ppocr_v6_small",
        "v6_small": "ppocr_v6_small",
        "paddle_v6_small": "ppocr_v6_small",
        "paddleocr-vl-1.6": "paddle_vl_16",
        "paddleocr_vl_1_6": "paddle_vl_16",
        "vl1.6": "paddle_vl_16",
        "vl16": "paddle_vl_16",
        "paddle_vl16": "paddle_vl_16",
        "pp-structurev3": "pp_structure_v3",
        "ppstructurev3": "pp_structure_v3",
        "structure_v3": "pp_structure_v3",
        "paddle_structure_v3": "pp_structure_v3",
        # Removed UI profiles: keep old configs loadable but migrate them to the
        # explicit compatibility route instead of silently selecting a removed
        # model pair.
        "ppocr_v5_server": "legacy_v5_auto",
        "ppocrv5_server": "legacy_v5_auto",
        "v5_server": "legacy_v5_auto",
        "ppocr_v5_mobile": "legacy_v5_auto",
        "ppocrv5_mobile": "legacy_v5_auto",
        "v5_mobile": "legacy_v5_auto",
        "pp-ocrv5": "legacy_v5_auto",
        "ppocrv5": "legacy_v5_auto",
        "paddle_legacy_v5": "legacy_v5_auto",
        "auto": "legacy_v5_auto",
        "": "legacy_v5_auto",
    }
    key = aliases.get(key, key)
    return key if key in _PROFILE_MAP else "legacy_v5_auto"


def get_paddle_model_profile(value: str | None) -> PaddleModelProfile:
    return _PROFILE_MAP[normalize_paddle_model_profile(value)]


def resolve_paddle_model_selection(ocr_config, profile_override: str | None = None) -> dict[str, str | None]:
    """Resolve a concrete worker engine selection from OCRConfig-like settings."""
    profile = get_paddle_model_profile(profile_override or getattr(ocr_config, "paddle_model_profile", None))
    det_name = profile.detection_model
    rec_name = profile.recognition_model
    ocr_version = profile.ocr_version or str(getattr(ocr_config, "ocr_version", "PP-OCRv5") or "PP-OCRv5")
    if profile.key == "custom":
        det_name = str(getattr(ocr_config, "paddle_text_detection_model_name", "") or "").strip() or None
        rec_name = str(getattr(ocr_config, "paddle_text_recognition_model_name", "") or "").strip() or None
    return {
        "profile": profile.key,
        "label": profile.label,
        "pipeline": profile.pipeline,
        "pipeline_version": profile.pipeline_version,
        "det_name": det_name,
        "rec_name": rec_name,
        "ocr_version": ocr_version,
    }


def profile_label(value: str | None) -> str:
    return get_paddle_model_profile(value).label


def backend_profile_key(backend: str | None) -> str | None:
    """Map independent OCR-backend choices to Paddle engine profiles."""
    key = str(backend or "").strip().lower()
    mapping = {
        "paddle_v6_medium": "ppocr_v6_medium",
        "paddle_v6_small": "ppocr_v6_small",
        "paddle_vl_16": "paddle_vl_16",
        "paddle_vl16": "paddle_vl_16",
        "paddle_structure_v3": "pp_structure_v3",
        "pp_structure_v3": "pp_structure_v3",
        "paddle_legacy_v5": "legacy_v5_auto",
    }
    return mapping.get(key)


__all__ = [
    "PaddleModelProfile", "PADDLE_MODEL_PROFILES", "normalize_paddle_model_profile",
    "get_paddle_model_profile", "resolve_paddle_model_selection", "profile_label",
    "backend_profile_key",
]
