from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from manga_hd_transfer.config import MaskReplaceConfig, PipelineConfig
from manga_hd_transfer.mask_transfer import (
    _expand_target_clear_mask_with_text_components,
    transfer_ocr_guided_text_units,
    transfer_paired_diff_regions,
)
from manga_hd_transfer.models import PagePair, RegistrationResult, TextUnit, UnitMatch
from manga_hd_transfer.paired_diff import extract_paired_diff_bubbles as extract_main
from manga_hd_transfer.paired_diff_v08 import extract_paired_diff_bubbles as extract_structural
from manga_hd_transfer.pipeline import TransferPipeline


def _reg(w: int, h: int, confidence: float = 0.99) -> RegistrationResult:
    return RegistrationResult(
        np.eye(3, dtype=np.float64), "identity", confidence, 1.0, 0.0, 1.0, 100,
        (w, h), (w, h), {},
    )


def _colored_text_pair() -> tuple[np.ndarray, np.ndarray]:
    h, w = 400, 600
    target = np.full((h, w, 3), (80, 160, 210), np.uint8)
    source = target.copy()
    # Stable artwork anchor around, but not through, the text candidate.
    cv2.rectangle(target, (30, 30), (570, 370), (20, 90, 140), 3)
    cv2.rectangle(source, (30, 30), (570, 370), (20, 90, 140), 3)
    # Japanese-like target glyph groups.
    for x in (210, 232, 254):
        for y in (150, 180, 210):
            cv2.rectangle(target, (x, y), (x + 8, y + 16), (20, 20, 20), -1)
            cv2.line(target, (x - 3, y + 8), (x + 12, y + 8), (20, 20, 20), 2)
    # Chinese-like source glyph groups with different structure.
    for y in (155, 185, 215):
        for x in (190, 222, 254):
            cv2.rectangle(source, (x, y), (x + 15, y + 7), (15, 15, 15), -1)
            cv2.line(source, (x + 7, y - 3), (x + 7, y + 12), (15, 15, 15), 2)
    return source, target


def test_colored_complex_text_is_detected_and_transferred_ink_only():
    source, target = _colored_text_pair()
    h, w = target.shape[:2]
    cfg = MaskReplaceConfig()
    cfg.paired_diff_local_flow_enabled = False
    pd = extract_structural(source, target, _reg(w, h), cfg)
    assert any(r.region_kind == "complex_text" for r in pd.records)

    result = transfer_paired_diff_regions(
        pd.aligned_source, target, pd.source_bubbles, pd.target_bubbles, cfg,
    )
    rec = next(r for r in result.records if r.geometry_mode == "complex_text")
    assert rec.applied
    assert rec.clarity_mode == "complex-text-ink-transfer"
    assert result.clear_mask is not None and cv2.countNonZero(result.clear_mask) > 0
    assert cv2.countNonZero(result.composite_mask) > cv2.countNonZero(result.clear_mask)
    # Stable colored artwork outside the write footprint must be byte-identical.
    outside = result.composite_mask == 0
    assert np.array_equal(result.image[outside], target[outside])


def test_low_confidence_complex_text_is_kept_as_review_candidate():
    source, target = _colored_text_pair()
    h, w = target.shape[:2]
    cfg = MaskReplaceConfig()
    cfg.paired_diff_local_flow_enabled = False
    cfg.paired_diff_low_confidence_candidate_threshold = 0.995
    pd = extract_structural(source, target, _reg(w, h), cfg)
    result = transfer_paired_diff_regions(
        pd.aligned_source, target, pd.source_bubbles, pd.target_bubbles, cfg,
    )
    rec = next(r for r in result.records if r.geometry_mode == "complex_text")
    assert rec.applied
    assert rec.candidate and rec.review_required and rec.restorable and rec.editable
    # v0.8.22+ content-completeness review takes precedence when both gates fire.
    assert rec.reason in {"applied_low_confidence_text_candidate", "applied_incomplete_review_candidate"}


def test_main_paired_diff_recovers_structural_text_when_raw_has_no_container():
    source, target = _colored_text_pair()
    h, w = target.shape[:2]
    cfg = MaskReplaceConfig()
    cfg.paired_diff_local_flow_enabled = False
    pd = extract_main(source, target, _reg(w, h, 0.70), cfg)
    assert pd.method == "structural_v08"
    assert pd.diagnostics.get("raw_empty_structural_recovery") is True
    assert any(r.region_kind == "complex_text" for r in pd.records)
    assert pd.safe_to_skip_ocr is False


def test_stable_colored_art_does_not_create_complex_text_candidate():
    source, target = _colored_text_pair()
    target = source.copy()
    h, w = target.shape[:2]
    cfg = MaskReplaceConfig()
    cfg.paired_diff_local_flow_enabled = False
    pd = extract_structural(source, target, _reg(w, h), cfg)
    assert not pd.records


def test_target_clear_expands_to_tall_vertical_japanese_component():
    target = np.full((220, 220, 3), 255, np.uint8)
    mask = np.zeros((220, 220), np.uint8)
    mask[45:175, 65:155] = 255
    # Tall narrow target glyph group just outside the imperfect interior mask.
    target[72:146, 157:163] = 0
    expanded = _expand_target_clear_mask_with_text_components(target, mask)
    assert expanded[100, 160] == 255
    # Unrelated distant artwork remains excluded.
    target[20:35, 20:180] = 0
    expanded2 = _expand_target_clear_mask_with_text_components(target, mask)
    assert expanded2[25, 80] == 0


def test_ocr_guided_geometry_uses_raster_ink_not_ocr_relettering():
    h, w = 400, 400
    target = np.full((h, w, 3), 240, np.uint8)
    source = target.copy()
    for x in (175, 185, 195):
        cv2.rectangle(target, (x, 170), (x + 4, 220), (20, 20, 20), -1)
    for y in (177, 192, 207):
        cv2.rectangle(source, (160, y), (215, y + 5), (15, 15, 15), -1)
    su = TextUnit("s", [(150, 155), (225, 155), (225, 230), (150, 230)], ["sb"], "中文字", 0.95, "speech")
    tu = TextUnit("t", [(155, 160), (220, 160), (220, 225), (155, 225)], ["tb"], "日本語", 0.95, "speech")
    match = UnitMatch("s", "t", 0.90, 0.10)
    cfg = MaskReplaceConfig()
    result = transfer_ocr_guided_text_units(
        source, target, [su], [tu], [match], _reg(w, h), cfg,
    )
    assert result.applied_count == 1
    rec = result.records[0]
    assert rec.geometry_mode == "ocr_guided_components"
    assert rec.clarity_mode == "ocr-guided-ink-transfer"
    assert rec.sr_backend == "ocr-guided-components"
    assert result.clear_mask is not None and cv2.countNonZero(result.clear_mask) > 0
    outside = result.composite_mask == 0
    assert np.array_equal(result.image[outside], target[outside])


def test_pipeline_writes_explicit_transfer_audit_artifacts(tmp_path: Path):
    # A clean same-layout bubble pair keeps this test independent of external OCR.
    h, w = 480, 360
    target = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(target, (10, 10), (350, 470), (30, 30, 30), 3)
    cv2.ellipse(target, (180, 240), (90, 120), 0, 0, 360, (0, 0, 0), 4)
    cv2.ellipse(target, (180, 240), (85, 115), 0, 0, 360, (255, 255, 255), -1)
    for x in (160, 180, 200):
        cv2.rectangle(target, (x, 205), (x + 5, 275), (0, 0, 0), -1)
    source = target.copy()
    cv2.ellipse(source, (180, 240), (85, 115), 0, 0, 360, (255, 255, 255), -1)
    cv2.rectangle(source, (135, 220), (225, 230), (0, 0, 0), -1)
    cv2.rectangle(source, (145, 252), (215, 262), (0, 0, 0), -1)
    sp = tmp_path / "cn.png"; tp = tmp_path / "jp.png"
    cv2.imwrite(str(sp), source); cv2.imwrite(str(tp), target)
    pair = PagePair(str(sp), str(tp), 0, 0, 0.99, 0.0, ["test"])
    cfg = PipelineConfig()
    cfg.transfer.mode = "mask_replace"
    cfg.ocr.backend = "none"; cfg.ocr.source_backend = "none"; cfg.ocr.target_backend = "none"
    cfg.export.layer_bundle = False; cfg.export.save_debug = False
    project = TransferPipeline(cfg).process_page(pair, tmp_path / "page")

    for name in (
        "source_original.png", "target_clear_mask.png", "chinese_transfer_layer.png",
        "final.png", "review_preview.png", "transfer_audit.json",
    ):
        assert (tmp_path / "page" / name).exists(), name
    audit = project.meta["transfer_audit"]
    assert audit["registration"]["confidence"] >= 0.0
    assert "candidate_detection" in audit and "ocr_evidence" in audit and "transfer" in audit
    assert project.artifacts["target_clear_mask"].endswith("target_clear_mask.png")
