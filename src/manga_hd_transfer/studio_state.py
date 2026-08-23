from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from .config import PipelineConfig
from .models import PagePair


def studio_default_config() -> PipelineConfig:
    """Return GUI defaults without importing Qt.

    Keeping state/default policy in this module lets CLI/tests inspect the studio
    configuration without loading the large GUI module or PySide6.
    """
    cfg = PipelineConfig()
    if sys.platform == "darwin":
        cfg.ocr.backend = "apple"
        cfg.ocr.source_backend = "apple"
        cfg.ocr.target_backend = "apple"
    return cfg


@dataclass
class StudioState:
    source_dir: str = ""
    target_dir: str = ""
    output_dir: str = ""
    pairs: list[PagePair] = field(default_factory=list)
    selected_index: int = 0
    config: PipelineConfig = field(default_factory=studio_default_config)
    last_project: Any = None
    last_result_path: str = ""
    projects_by_page: dict[str, Any] = field(default_factory=dict)
    batch_status: dict[str, tuple[str, str]] = field(default_factory=dict)
    page_marks: dict[str, dict[str, Any]] = field(default_factory=dict)
    unmatched_source: list[str] = field(default_factory=list)
    unmatched_target: list[str] = field(default_factory=list)
    last_manual_effect_mode: str = ""
    restored_page_roots: dict[str, str] = field(default_factory=dict)
    restored_page_origin: dict[str, str] = field(default_factory=dict)
    # True only after the user explicitly presses 智能配对. Restoring existing
    # results deliberately loads the processed subset only and must never imply
    # that a full-book pairing has been performed.
    book_pairing_ready: bool = False
    restored_processed_only: bool = False


__all__ = ["StudioState", "studio_default_config"]
