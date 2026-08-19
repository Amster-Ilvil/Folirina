from __future__ import annotations

"""Page-level admission and marking for manga transfer jobs.

The transfer pipeline is deliberately page-aware: covers, title pages, tables of
contents, chapter splash pages and other image-only assets stay in reading order.
As of v0.8.20 every newly paired page defaults to content and only explicit manual
classifications bypass transfer. A content page may still become an unchanged
runtime passthrough when real source OCR/bubble evidence proves that there is no
Chinese speech/narration container to migrate.

The UI keeps these decisions separate from pairing. Pairing answers *which two
pages correspond*; Page Manager stores explicit user page classifications.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2

from .config import PipelineConfig
from .io_utils import read_image, stem_id
from .models import PagePair
from .paired_diff import PairedDiffResult, extract_paired_diff_bubbles
from .registration import register_images


PAGE_TYPE_INFO: dict[str, dict[str, Any]] = {
    "content": {
        "label": "正文 / 需替换",
        "process": True,
        "color": "#6C63D8",
        "description": "正常进入配准、精准蒙版与复核流程。",
    },
    "auto_no_text": {
        "label": "无气泡/文本框（自动）",
        "process": False,
        "color": "#A8B1BF",
        "description": "自动检测为无气泡/文本框页面，直接保留高清日文页。",
    },
    "cover": {
        "label": "封面",
        "process": False,
        "color": "#F26B4A",
        "description": "封面作为整页图片保留，不做文字迁移。",
    },
    "title_page": {
        "label": "扉页 / 书名页",
        "process": False,
        "color": "#C97A57",
        "description": "扉页作为整页图片保留。",
    },
    "toc_page": {
        "label": "目录",
        "process": False,
        "color": "#4F8FEF",
        "description": "目录页作为整页图片保留。",
    },
    "chapter_title": {
        "label": "单话首页 / 章节页",
        "process": False,
        "color": "#8F6CCF",
        "description": "单话首页或章节 splash 页面作为整页图片保留。",
    },
    "illustration": {
        "label": "插图 / 纯图片",
        "process": False,
        "color": "#32A47C",
        "description": "无须替换的纯图片页直接保留高清日文页。",
    },
    "frontispiece": {
        "label": "卷首插画",
        "process": False,
        "color": "#EC6F9E",
        "description": "卷首插画直接保留高清日文页。",
    },
    "blank": {
        "label": "空白页",
        "process": False,
        "color": "#D2D2D7",
        "description": "空白页直接保留。",
    },
    "back_matter": {
        "label": "后记 / 版权 / 广告",
        "process": False,
        "color": "#C68A2D",
        "description": "后记、版权、广告等整页资产不做漫画气泡迁移。",
    },
    "skip": {
        "label": "手动跳过",
        "process": False,
        "color": "#7D7D83",
        "description": "用户明确要求跳过处理，最终输出高清日文原页。",
    },
}

MANUAL_PAGE_TYPES = (
    "content",
    "cover",
    "title_page",
    "toc_page",
    "chapter_title",
    "illustration",
    "frontispiece",
    "blank",
    "back_matter",
    "skip",
)


@dataclass(slots=True)
class PageMark:
    page_type: str = "content"
    origin: str = "default"  # default|auto|manual
    confidence: float = 0.0
    reason: str = ""
    bubble_regions: int = 0
    free_text_regions: int = 0
    registration_confidence: float = 0.0
    source_name: str = ""
    target_name: str = ""

    @property
    def should_process(self) -> bool:
        return bool(PAGE_TYPE_INFO.get(self.page_type, PAGE_TYPE_INFO["content"])["process"])

    @property
    def label(self) -> str:
        return str(PAGE_TYPE_INFO.get(self.page_type, PAGE_TYPE_INFO["content"])["label"])

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["should_process"] = self.should_process
        data["label"] = self.label
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PageMark":
        raw = dict(data) if isinstance(data, Mapping) else {}
        page_type = str(raw.get("page_type") or "content")
        origin = str(raw.get("origin") or "default")
        reason = str(raw.get("reason") or "")
        if page_type not in PAGE_TYPE_INFO:
            page_type = "content"
        # v0.8.19 could persist geometry-only auto_no_text decisions.  v0.8.20
        # deliberately removes that admission policy: old automatic marks migrate
        # back to the new default (content), while explicit manual page types stay
        # authoritative.
        if page_type == "auto_no_text" and origin != "manual":
            page_type = "content"
            origin = "default"
            reason = "legacy_auto_no_text_reset_to_content"
        return cls(
            page_type=page_type,
            origin=origin,
            confidence=float(raw.get("confidence") or 0.0),
            reason=reason,
            bubble_regions=int(raw.get("bubble_regions") or 0),
            free_text_regions=int(raw.get("free_text_regions") or 0),
            registration_confidence=float(raw.get("registration_confidence") or 0.0),
            source_name=str(raw.get("source_name") or ""),
            target_name=str(raw.get("target_name") or ""),
        )


def page_mark_key(pair: PagePair) -> str:
    """Stable per-project key shared with the workspace cache layout."""
    return stem_id(pair.target_path)


def default_mark(pair: PagePair) -> PageMark:
    return PageMark(
        page_type="content",
        origin="default",
        source_name=Path(pair.source_path).name,
        target_name=Path(pair.target_path).name,
    )


def manual_mark(pair: PagePair, page_type: str) -> PageMark:
    page_type = page_type if page_type in PAGE_TYPE_INFO else "content"
    return PageMark(
        page_type=page_type,
        origin="manual",
        confidence=1.0,
        reason="manual_page_type",
        source_name=Path(pair.source_path).name,
        target_name=Path(pair.target_path).name,
    )


def resolve_mark(marks: Mapping[str, Mapping[str, Any] | PageMark] | None, pair: PagePair) -> PageMark:
    if not marks:
        return default_mark(pair)
    raw = marks.get(page_mark_key(pair))
    if raw is None:
        # A renamed target can still be resolved from the saved target basename.
        target_name = Path(pair.target_path).name
        for value in marks.values():
            mark = value if isinstance(value, PageMark) else PageMark.from_dict(value)
            if mark.target_name and mark.target_name == target_name:
                return mark
        return default_mark(pair)
    return raw if isinstance(raw, PageMark) else PageMark.from_dict(raw)


def page_type_label(page_type: str) -> str:
    return str(PAGE_TYPE_INFO.get(page_type, PAGE_TYPE_INFO["content"])["label"])


def page_type_color(page_type: str) -> str:
    return str(PAGE_TYPE_INFO.get(page_type, PAGE_TYPE_INFO["content"])["color"])


def should_process_type(page_type: str) -> bool:
    return bool(PAGE_TYPE_INFO.get(page_type, PAGE_TYPE_INFO["content"])["process"])


def _resize_for_scan(image, max_side: int):
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = float(max_side) / float(max(h, w))
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def _region_counts(result: PairedDiffResult | None) -> tuple[int, int, int]:
    if result is None:
        return 0, 0, 0
    bubble = 0
    free = 0
    total = 0
    for rec in list(result.records or []):
        total += 1
        if str(getattr(rec, "region_kind", "bubble")) == "free_text":
            free += 1
        else:
            bubble += 1
    supplement = getattr(result, "supplemental", None)
    if supplement is not None:
        for rec in list(supplement.records or []):
            total += 1
            if str(getattr(rec, "region_kind", "bubble")) == "free_text":
                free += 1
            else:
                bubble += 1
    return bubble, free, total


def classify_from_paired_diff(
    pair: PagePair,
    *,
    registration_confidence: float,
    result: PairedDiffResult | None,
    config: PipelineConfig,
) -> PageMark:
    """Legacy geometry probe retained for config/script compatibility.

    The default v0.8.20 config disables automatic admission, so this normally
    returns ``content`` while still recording region diagnostics. Older callers
    can explicitly enable ``auto_skip_no_text_boxes`` if they need the historical
    behavior, but the Studio UI no longer invokes this path.
    """
    bubble_count, free_count, total = _region_counts(result)
    pm_cfg = config.page_management
    enough_registration = registration_confidence >= float(pm_cfg.auto_skip_min_registration_confidence)
    if result is None:
        return PageMark(
            page_type="content", origin="auto",
            confidence=min(1.0, max(0.0, registration_confidence)),
            reason="no_geometry_evidence;keep_for_processing",
            bubble_regions=0, free_text_regions=0,
            registration_confidence=registration_confidence,
            source_name=Path(pair.source_path).name,
            target_name=Path(pair.target_path).name,
        )
    if bool(pm_cfg.auto_skip_no_text_boxes) and enough_registration and bubble_count == 0:
        reason = "no_balloon_or_textbox_regions"
        if free_count:
            reason += f";free_text_only={free_count}"
        elif total == 0:
            reason += ";no_transfer_regions"
        return PageMark(
            page_type="auto_no_text",
            origin="auto",
            confidence=min(1.0, max(0.0, registration_confidence)),
            reason=reason,
            bubble_regions=bubble_count,
            free_text_regions=free_count,
            registration_confidence=registration_confidence,
            source_name=Path(pair.source_path).name,
            target_name=Path(pair.target_path).name,
        )
    return PageMark(
        page_type="content",
        origin="auto",
        confidence=min(1.0, max(0.0, registration_confidence)),
        reason=(f"transfer_regions={total};bubble_regions={bubble_count};free_text_regions={free_count}"
                if enough_registration else "registration_too_low_for_auto_skip"),
        bubble_regions=bubble_count,
        free_text_regions=free_count,
        registration_confidence=registration_confidence,
        source_name=Path(pair.source_path).name,
        target_name=Path(pair.target_path).name,
    )


def analyze_pair_for_page_mark(
    pair: PagePair,
    config: PipelineConfig,
    *,
    cancel_cb: Callable[[], bool] | None = None,
) -> PageMark:
    """Fast, OCR-free page admission probe used by the Page Manager.

    The analysis is intentionally independent from text recognition.  It uses a
    downscaled OpenCV registration plus the same paired-difference geometry used
    by the transfer pipeline, so page labels and actual processing follow one
    definition of a transferable balloon/text box.
    """
    if cancel_cb is not None and cancel_cb():
        raise InterruptedError("page analysis cancelled")
    source = _resize_for_scan(read_image(pair.source_path), int(config.page_management.scan_max_side))
    target = _resize_for_scan(read_image(pair.target_path), int(config.page_management.scan_max_side))
    reg_cfg = config.registration.model_copy(deep=True)
    # Page Manager must stay deterministic and must never trigger optional model
    # downloads merely to decide whether a page is an asset page.
    reg_cfg.backend = "opencv"
    reg_cfg.allow_model_downloads = False
    reg_cfg.max_features = min(int(reg_cfg.max_features), 3200)
    registration = register_images(source, target, reg_cfg)
    if cancel_cb is not None and cancel_cb():
        raise InterruptedError("page analysis cancelled")

    paired = None
    gate = min(
        float(config.mask_replace.paired_diff_min_registration_confidence),
        float(config.mask_replace.photo_pair_min_registration_confidence),
        float(config.page_management.auto_skip_min_registration_confidence),
    )
    if registration.confidence >= gate:
        try:
            paired = extract_paired_diff_bubbles(source, target, registration, config.mask_replace)
        except Exception:
            paired = None
    if cancel_cb is not None and cancel_cb():
        raise InterruptedError("page analysis cancelled")
    return classify_from_paired_diff(
        pair,
        registration_confidence=float(registration.confidence),
        result=paired,
        config=config,
    )


def marks_to_json(marks: Mapping[str, Mapping[str, Any] | PageMark]) -> dict[str, Any]:
    pages: dict[str, Any] = {}
    for key, value in marks.items():
        mark = value if isinstance(value, PageMark) else PageMark.from_dict(value)
        pages[str(key)] = mark.to_dict()
    return {
        "schema": "manga_hd_translation_transfer.page_management.v1",
        "pages": pages,
    }


def marks_from_json(data: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    root = dict(data) if isinstance(data, Mapping) else {}
    pages_value = root.get("pages", {})
    raw_pages = dict(pages_value) if isinstance(pages_value, Mapping) else {}
    return {str(key): PageMark.from_dict(value).to_dict() for key, value in raw_pages.items()}


__all__ = [
    "PAGE_TYPE_INFO",
    "MANUAL_PAGE_TYPES",
    "PageMark",
    "page_mark_key",
    "default_mark",
    "manual_mark",
    "resolve_mark",
    "page_type_label",
    "page_type_color",
    "should_process_type",
    "classify_from_paired_diff",
    "analyze_pair_for_page_mark",
    "marks_to_json",
    "marks_from_json",
]
