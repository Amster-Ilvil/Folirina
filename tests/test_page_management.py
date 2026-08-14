from pathlib import Path

import cv2
import numpy as np
import pytest

from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.models import PagePair
from manga_hd_transfer.page_management import (
    PageMark,
    analyze_pair_for_page_mark,
    classify_from_paired_diff,
    manual_mark,
    marks_from_json,
    marks_to_json,
    page_mark_key,
    resolve_mark,
)
from manga_hd_transfer.paired_diff import DiffBubbleRecord, PairedDiffResult
from manga_hd_transfer.pipeline import PipelineCancelled, TransferPipeline


def _pair(source: str = "source.png", target: str = "target.png") -> PagePair:
    return PagePair(source, target, 0, 0, 0.99, 0.0, ["test"])


def _result(records):
    return PairedDiffResult(
        source_bubbles=[], target_bubbles=[],
        change_mask=np.zeros((32, 32), np.uint8), records=list(records),
        threshold=0.0, noise_floor=0.0,
    )


def test_manual_page_type_is_authoritative_and_serializable():
    pair = _pair()
    mark = manual_mark(pair, "cover")
    marks = {page_mark_key(pair): mark.to_dict()}
    restored = marks_from_json(marks_to_json(marks))
    resolved = resolve_mark(restored, pair)
    assert resolved.origin == "manual"
    assert resolved.page_type == "cover"
    assert resolved.should_process is False


def test_missing_geometry_never_auto_discards_page():
    cfg = PipelineConfig()
    mark = classify_from_paired_diff(
        _pair(), registration_confidence=0.99, result=None, config=cfg,
    )
    assert mark.should_process is True
    assert "no_geometry_evidence" in mark.reason


def test_no_balloon_or_textbox_stays_default_content_without_auto_scan():
    cfg = PipelineConfig()
    mark = classify_from_paired_diff(
        _pair(), registration_confidence=0.99, result=_result([]), config=cfg,
    )
    assert cfg.page_management.auto_skip_no_text_boxes is False
    assert mark.page_type == "content"
    assert mark.should_process is True


def test_free_text_only_page_is_not_auto_classified_anymore():
    cfg = PipelineConfig()
    rec = DiffBubbleRecord(
        source_id="s", target_id="t", change_density=0.2, mask_iou=0.9,
        confidence=0.95, bbox_target=(1, 1, 10, 10), region_kind="free_text",
    )
    mark = classify_from_paired_diff(
        _pair(), registration_confidence=0.99, result=_result([rec]), config=cfg,
    )
    assert mark.page_type == "content"
    assert mark.should_process is True
    assert mark.free_text_regions == 1


def test_bubble_page_remains_processable():
    cfg = PipelineConfig()
    rec = DiffBubbleRecord(
        source_id="s", target_id="t", change_density=0.2, mask_iou=0.9,
        confidence=0.95, bbox_target=(1, 1, 10, 10), region_kind="bubble",
    )
    mark = classify_from_paired_diff(
        _pair(), registration_confidence=0.99, result=_result([rec]), config=cfg,
    )
    assert mark.page_type == "content"
    assert mark.should_process is True
    assert mark.bubble_regions == 1


def test_manual_skip_passthrough_does_not_read_source_or_run_ocr(tmp_path: Path):
    target = np.full((80, 64, 3), 247, np.uint8)
    target[10:30, 12:28] = (20, 30, 40)
    target_path = tmp_path / "target.png"
    cv2.imwrite(str(target_path), target)
    pair = _pair(str(tmp_path / "missing-source.png"), str(target_path))
    cfg = PipelineConfig()
    cfg.export.save_project_json = True
    pipeline = TransferPipeline(cfg)
    mark = manual_mark(pair, "illustration")
    final = tmp_path / "final" / "target.png"

    project = pipeline.process_page(pair, tmp_path / "pages" / "target", final, page_mark=mark)

    assert project.meta["passthrough"] is True
    assert project.meta["page_management"]["page_type"] == "illustration"
    assert Path(project.artifacts["final"]).exists()
    out = cv2.imread(str(final), cv2.IMREAD_COLOR)
    assert np.array_equal(out, target)
    assert (tmp_path / "pages" / "target" / "page_management.json").exists()


def test_single_page_cancel_is_checked_before_expensive_work(tmp_path: Path):
    pair = _pair(str(tmp_path / "missing-source.png"), str(tmp_path / "missing-target.png"))
    with pytest.raises(PipelineCancelled):
        TransferPipeline(PipelineConfig()).process_page(
            pair, tmp_path / "page", page_mark=PageMark(page_type="content"),
            cancel_cb=lambda: True,
        )


def test_legacy_page_probe_no_longer_changes_default_page_type(tmp_path: Path):
    # The legacy probe may still be called by old scripts, but v0.8.20 disables
    # automatic admission: newly paired pages remain content by default.
    image = np.full((720, 520, 3), 255, np.uint8)
    cv2.rectangle(image, (35, 35), (485, 685), (0, 0, 0), 4)
    cv2.circle(image, (260, 300), 145, (0, 0, 0), 5)
    for y in range(140, 620, 55):
        cv2.line(image, (95, y), (430, y + 23), (0, 0, 0), 3)
    source = tmp_path / "art-cn.png"; target = tmp_path / "art-jp.png"
    cv2.imwrite(str(source), image); cv2.imwrite(str(target), image)
    pair = _pair(str(source), str(target))

    mark = analyze_pair_for_page_mark(pair, PipelineConfig())

    assert mark.registration_confidence >= 0.72
    assert mark.page_type == "content"
    assert mark.should_process is True

def test_page_manager_mark_change_overrides_resume_cache(tmp_path: Path):
    image = np.full((90, 70, 3), 235, np.uint8)
    image[15:40, 20:45] = (12, 20, 28)
    source = tmp_path / "source.png"; target = tmp_path / "target.png"
    cv2.imwrite(str(source), image); cv2.imwrite(str(target), image)
    pair = _pair(str(source), str(target))
    key = page_mark_key(pair)
    cfg = PipelineConfig(); cfg.transfer.mode = "mask_replace"
    pipeline = TransferPipeline(cfg)
    out = tmp_path / "book"

    first = pipeline.run_book(
        tmp_path, tmp_path, out, pairs_override=[pair],
        page_marks={key: manual_mark(pair, "cover").to_dict()},
    )
    assert first.pages[0].meta["page_management"]["page_type"] == "cover"

    second = pipeline.run_book(
        tmp_path, tmp_path, out, pairs_override=[pair],
        page_marks={key: manual_mark(pair, "title_page").to_dict()},
    )
    assert second.meta["resumed_count"] == 0
    assert second.pages[0].meta["page_management"]["page_type"] == "title_page"

def test_content_page_with_conclusive_no_source_text_is_passthrough(tmp_path: Path):
    from manga_hd_transfer.ocr import InjectedOCRBackend

    source_image = np.full((220, 180, 3), 255, np.uint8)
    target_image = source_image.copy()
    cv2.rectangle(target_image, (40, 60), (140, 150), (0, 0, 0), 3)
    source = tmp_path / "source.png"; target = tmp_path / "target.png"
    cv2.imwrite(str(source), source_image); cv2.imwrite(str(target), target_image)
    pair = _pair(str(source), str(target))
    cfg = PipelineConfig(); cfg.transfer.mode = "mask_replace"; cfg.mask_replace.paired_diff_enabled = False
    cfg.cache.enabled = False; cfg.export.layer_bundle = False; cfg.export.save_debug = False

    book = TransferPipeline(cfg, InjectedOCRBackend([]), InjectedOCRBackend([])).run_book(
        tmp_path, tmp_path, tmp_path / "out", pairs_override=[pair], page_marks={},
    )

    assert book.meta["skipped_count"] == 1
    assert book.pages[0].meta["passthrough"] is True
    assert book.pages[0].meta["passthrough_reason"] == "source_no_transferable_text"
    assert book.pages[0].meta["page_management"]["page_type"] == "content"
    out = cv2.imread(book.pages[0].artifacts["final"], cv2.IMREAD_COLOR)
    assert np.array_equal(out, target_image)


def test_legacy_auto_no_text_mark_migrates_back_to_default_content():
    pair = _pair()
    key = page_mark_key(pair)
    restored = marks_from_json({
        "pages": {
            key: {
                "page_type": "auto_no_text", "origin": "auto",
                "source_name": "source.png", "target_name": "target.png",
            }
        }
    })
    mark = resolve_mark(restored, pair)
    assert mark.page_type == "content"
    assert mark.origin == "default"
    assert mark.should_process is True


def test_manual_skips_still_emit_one_final_page_per_pair(tmp_path: Path):
    images = []
    pairs = []
    marks = {}
    for idx in range(3):
        target = np.full((90, 70, 3), 235 - idx * 10, np.uint8)
        target_path = tmp_path / f"target-{idx:03d}.png"
        cv2.imwrite(str(target_path), target)
        images.append(target)
        pair = _pair(str(tmp_path / f"missing-{idx:03d}.png"), str(target_path))
        pairs.append(pair)
        marks[page_mark_key(pair)] = manual_mark(pair, "illustration").to_dict()

    book = TransferPipeline(PipelineConfig()).run_book(
        tmp_path, tmp_path, tmp_path / "out-skips", pairs_override=pairs, page_marks=marks,
    )

    finals = sorted((tmp_path / "out-skips" / "final").glob("*.png"))
    assert len(book.pages) == len(pairs) == 3
    assert len(finals) == 3
    for page, expected in zip(book.pages, images):
        assert page.meta["passthrough"] is True
        rendered = cv2.imread(page.artifacts["final"], cv2.IMREAD_COLOR)
        assert np.array_equal(rendered, expected)


def test_single_page_cancel_also_stops_after_decode_boundary(tmp_path: Path):
    image = np.full((64, 48, 3), 255, np.uint8)
    source = tmp_path / "source.png"; target = tmp_path / "target.png"
    cv2.imwrite(str(source), image); cv2.imwrite(str(target), image)
    pair = _pair(str(source), str(target))
    calls = {"n": 0}

    def cancel_after_first_boundary():
        calls["n"] += 1
        return calls["n"] >= 2

    final = tmp_path / "final.png"
    with pytest.raises(PipelineCancelled):
        TransferPipeline(PipelineConfig()).process_page(
            pair, tmp_path / "page", page_mark=PageMark(page_type="content"),
            cancel_cb=cancel_after_first_boundary,
        )
    assert calls["n"] >= 2
    assert not final.exists()
