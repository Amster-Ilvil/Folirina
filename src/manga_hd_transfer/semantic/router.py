from __future__ import annotations

from .schema import SemanticBlock

_IGNORE = {"number", "page_number", "header", "header_image", "footer", "footer_image", "footnote", "copyright"}
_PROCESS = {"text", "vertical_text", "dialogue", "narration", "dialogue_candidate", "narration_candidate", "effect_text", "sfx", "bubble"}
_REVIEW = {"aside_text", "unknown", "doc_title", "paragraph_title", "chapter_title", "other_text", "artwork_text", "furigana"}


def semantic_type_for_label(raw_label: str) -> str:
    label = str(raw_label or "unknown").strip().lower()
    if label in {"number", "page_number"}:
        return "page_number"
    if label in {"header", "header_image"}:
        return "header"
    if label in {"footer", "footer_image", "footnote"}:
        return "footer"
    if label in {"text", "vertical_text"}:
        return "dialogue_candidate"
    if label in {"sfx", "effect_text"}:
        return "effect_text"
    if label == "bubble":
        return "bubble"
    if label in {"doc_title", "paragraph_title"}:
        return "chapter_title"
    if label == "aside_text":
        return "aside_text"
    return label or "unknown"


def route_label(raw_label: str, confidence: float, strategy: str = "auto") -> tuple[str, bool, str]:
    label = str(raw_label or "unknown").strip().lower()
    strategy = str(strategy or "auto").strip().lower()
    semantic_type = semantic_type_for_label(label)
    if label in _IGNORE or semantic_type in _IGNORE:
        return "IGNORE", False, semantic_type
    if strategy == "analysis_only":
        return "REVIEW", False, semantic_type
    if strategy == "strict":
        if label in _PROCESS and float(confidence) >= 0.62:
            return "PROCESS", True, semantic_type
        return "REVIEW", False, semantic_type
    if strategy == "loose":
        if label in _PROCESS or label in _REVIEW:
            return "PROCESS", True, semantic_type
        return "REVIEW", False, semantic_type
    if label in _PROCESS:
        return "PROCESS", True, semantic_type
    return "REVIEW", False, semantic_type


def route_blocks(blocks: list[SemanticBlock], strategy: str = "auto") -> list[SemanticBlock]:
    for block in blocks:
        action, processable, semantic_type = route_label(block.raw_label, block.confidence, strategy)
        block.action = action
        block.processable = processable
        block.semantic_type = semantic_type
    return blocks
