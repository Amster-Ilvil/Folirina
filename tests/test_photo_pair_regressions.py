from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from manga_hd_transfer.cache import PageStageCache
from manga_hd_transfer.config import MaskReplaceConfig, PipelineConfig, QAConfig
from manga_hd_transfer.mask_transfer import transfer_bubble_patches
from manga_hd_transfer.models import BubbleInstance, PagePair, RegistrationResult, TextBlock
from manga_hd_transfer.ocr import OCRBackend, InjectedOCRBackend
from manga_hd_transfer.paired_diff import PairedDiffResult, DiffBubbleRecord
from manga_hd_transfer.pipeline import TransferPipeline
from manga_hd_transfer.qa import run_mask_replace_qa


def _rect_bubble(bid: str, box, shape, *, photo=False, paired_id=None, block_ids=None):
    x0, y0, x1, y1 = map(int, box)
    mask = np.zeros(shape, np.uint8)
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    meta = {"mask_is_interior": True}
    if photo:
        meta["paired_diff_method"] = "photo_pair"
        if paired_id:
            meta["paired_target_id" if bid.startswith("s") else "paired_source_id"] = paired_id
    return BubbleInstance(
        bid,
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        .99,
        "speech",
        list(block_ids or []),
        mask,
        mask.copy(),
        meta,
    )


def test_photo_pair_uses_registered_size_not_raw_photo_pixels():
    """A 3x phone photo must not be rejected merely because its raw bbox is larger."""
    target = np.full((300, 400, 3), 255, np.uint8)
    source = np.full((900, 1200, 3), 245, np.uint8)
    tbox = (100, 80, 300, 220)
    sbox = tuple(v * 3 for v in tbox)
    # Japanese target / Chinese source glyph patterns.
    for x in (175, 200, 225):
        cv2.rectangle(target, (x, 115), (x + 5, 190), 0, -1)
    cv2.putText(source, "CN", (sbox[0] + 70, sbox[1] + 220), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (20, 20, 20), 9, cv2.LINE_AA)
    # Smooth phone-lighting drift.
    grad = np.linspace(-18, 18, source.shape[1], dtype=np.float32)[None, :, None]
    source = np.clip(source.astype(np.float32) + grad, 0, 255).astype(np.uint8)

    sb = _rect_bubble("s-photo", sbox, source.shape[:2], photo=True, paired_id="t-photo", block_ids=["s"])
    tb = _rect_bubble("t-photo", tbox, target.shape[:2], photo=True, paired_id="s-photo")
    H = np.array([[1 / 3, 0, 0], [0, 1 / 3, 0], [0, 0, 1]], np.float64)
    reg = RegistrationResult(H, "known-photo-scale", .99, 1.0, 0.0, .9, 40, (1200, 900), (400, 300), {})
    cfg = MaskReplaceConfig(
        min_match_confidence=.1,
        photo_pair_min_transfer_iou=.70,
        photo_pair_min_transfer_coverage=.85,
        photo_pair_max_spill_ratio=.30,
        local_fit="global",
        preserve_target_border=False,
        feather_px=0,
        photo_pair_min_direct_side_px=40,
        sr_backend="off",
    )
    out = transfer_bubble_patches(source, target, [sb], [tb], reg, cfg)
    assert out.applied_count == 1, [r.to_dict() for r in out.records]
    assert out.records[0].reason != "target_bubble_much_smaller_after_registration"
    assert out.records[0].clarity_mode in {"photo-crisp-ink", "photo-normalized-pixels", "ink-reconstruction", "pixels"}


def test_empty_mask_replace_is_publication_error():
    pair = PagePair("source.png", "target.png", 0, 0, .99, 0.01, [])
    reg = RegistrationResult(np.eye(3), "identity", .99, 1.0, 0.0, .9, 20, (100, 100), (100, 100), {})
    issues = run_mask_replace_qa(pair, reg, [], [], [], QAConfig(fail_empty_mask_replace=True), MaskReplaceConfig())
    assert any(x.code == "mask_replace_no_candidates" and x.severity == "error" for x in issues)


class RecordingOCRBackend(OCRBackend):
    def __init__(self):
        self.shapes = []

    def recognize(self, image: np.ndarray, *, image_path=None):
        self.shapes.append(image.shape[:2])
        return []


def test_rectified_photo_ocr_preserves_source_sampling_density(tmp_path):
    cfg = PipelineConfig()
    cfg.cache.enabled = False
    cfg.ocr.rectify_preserve_source_resolution = True
    cfg.ocr.rectify_max_scale = 3.0
    cfg.ocr.rectify_max_long_side = 3600
    backend = RecordingOCRBackend()
    pipe = TransferPipeline(cfg, source_ocr=backend, target_ocr=backend)
    source = np.full((3650, 2400, 3), 255, np.uint8)
    # Source photo -> clean target approximately matches the real benchmark ratio.
    H = np.array([[850 / 2400, 0, 0], [0, 1200 / 3650, 0], [0, 0, 1]], np.float64)
    reg = RegistrationResult(H, "known", .90, .9, .5, .7, 100, (2400, 3650), (850, 1200), {})
    dummy = tmp_path / "source.jpg"
    dummy.write_bytes(b"x")
    stats = {}
    pipe._recognize_source_rectified_cached(
        backend, source, dummy, (1200, 850), reg, PageStageCache(tmp_path / "page", enabled=False), stats
    )
    assert backend.shapes
    h, w = backend.shapes[-1]
    assert w > 1700 and h > 2400  # not collapsed to 850x1200 before OCR
    assert max(h, w) <= 3600
    assert stats["ocr_source"] == "miss_rectified"


def test_photo_mask_replace_never_ocr_reletters_even_with_legacy_fallback_flags(tmp_path, monkeypatch):
    """Precise Mask preserves source glyphs even if an old config enables OCR fallbacks."""
    import manga_hd_transfer.pipeline as pipeline_mod
    from manga_hd_transfer.io_utils import write_image

    target = np.full((520, 760, 3), 238, np.uint8)
    # Two clean white bubbles with black outlines.
    cv2.ellipse(target, (210, 180), (105, 75), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(target, (210, 180), (105, 75), 0, 0, 360, 0, 3)
    cv2.ellipse(target, (550, 330), (120, 90), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(target, (550, 330), (120, 90), 0, 0, 360, 0, 3)
    for x in (195, 215, 235):
        cv2.rectangle(target, (x, 145), (x + 5, 215), 0, -1)
    for x in (530, 550, 570):
        cv2.rectangle(target, (x, 290), (x + 5, 370), 0, -1)
    source = target.copy()
    cv2.ellipse(source, (210, 180), (100, 70), 0, 0, 360, (255, 255, 255), -1)
    cv2.putText(source, "CN", (160, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 3, cv2.LINE_AA)
    cv2.ellipse(source, (550, 330), (115, 85), 0, 0, 360, (255, 255, 255), -1)
    cv2.putText(source, "TEXT", (475, 345), cv2.FONT_HERSHEY_SIMPLEX, .7, 0, 2, cv2.LINE_AA)
    sp, tp = tmp_path / "source.png", tmp_path / "target.png"
    write_image(sp, source); write_image(tp, target)

    sblocks = [
        TextBlock("s1", [(160, 135), (260, 135), (260, 220), (160, 220)], "第一处中文", .99, reading_order=0),
        TextBlock("s2", [(470, 280), (635, 280), (635, 380), (470, 380)], "第二处中文", .99, reading_order=1),
    ]
    tblocks = [
        TextBlock("t1", [(160, 135), (260, 135), (260, 220), (160, 220)], "日本語一", .99, reading_order=0),
        TextBlock("t2", [(470, 280), (635, 280), (635, 380), (470, 380)], "日本語二", .99, reading_order=1),
    ]
    sm = np.zeros(target.shape[:2], np.uint8); cv2.ellipse(sm, (210, 180), (100, 70), 0, 0, 360, 255, -1)
    tm = np.zeros(target.shape[:2], np.uint8); cv2.ellipse(tm, (210, 180), (100, 70), 0, 0, 360, 255, -1)
    sb = BubbleInstance("photo-src-000", [(110,110),(310,110),(310,250),(110,250)], .99, "speech", ["paired"], sm, sm.copy(), {"paired_diff_method":"photo_pair","mask_is_interior":True,"paired_target_id":"photo-dst-000"})
    tb = BubbleInstance("photo-dst-000", [(110,110),(310,110),(310,250),(110,250)], .99, "speech", [], tm, tm.copy(), {"paired_diff_method":"photo_pair","mask_is_interior":True,"paired_source_id":"photo-src-000"})
    pd = PairedDiffResult(
        [sb], [tb], tm.copy(),
        [DiffBubbleRecord(sb.id, tb.id, .8, .99, .99, (110,110,310,250), "photo_pair", .08, .08)],
        100.0, 120.0, method="photo_pair", safe_to_skip_ocr=False,
        diagnostics={"incomplete_requires_ocr": True},
    )
    reg = RegistrationResult(np.eye(3), "identity", .99, 1.0, 0.0, .9, 50, (760,520), (760,520), {})
    monkeypatch.setattr(pipeline_mod, "extract_paired_diff_bubbles", lambda *args, **kwargs: pd)
    monkeypatch.setattr(pipeline_mod, "register_images", lambda *args, **kwargs: reg)

    cfg = PipelineConfig()
    cfg.transfer.mode = "mask_replace"
    cfg.export.layer_bundle = False
    cfg.export.save_debug = False
    cfg.cache.enabled = False
    cfg.bubbles.backend = "seeded_white"
    cfg.matching.review_confidence = .25
    cfg.qa.registration_min_confidence = .25
    cfg.qa.match_min_confidence = .25
    cfg.mask_replace.min_match_confidence = .1
    cfg.mask_replace.source_direct_container_enabled = False  # isolate mocked photo_pair route
    cfg.mask_replace.photo_pair_min_direct_side_px = 40
    cfg.mask_replace.local_fit = "global"
    cfg.mask_replace.feather_px = 0
    cfg.mask_replace.photo_pair_require_ocr_evidence = True
    # Simulate an old saved config from v0.8.14 with every OCR fallback enabled.
    # Strict Precise Mask semantics must still win.
    cfg.mask_replace.photo_pair_prefer_reletter_with_ocr = True
    cfg.mask_replace.photo_pair_fallback_reletter_missing = True
    cfg.mask_replace.fallback_reletter_on_blur = True
    pipe = TransferPipeline(cfg, InjectedOCRBackend(sblocks), InjectedOCRBackend(tblocks))
    project = pipe.process_page(PagePair(str(sp), str(tp), 0, 0, .99, .01, []), tmp_path / "out")

    # Newer completion routes may recover the second obvious bubble as well;
    # the contract under test is that Precise Mask never OCR-reletters either.
    assert project.meta["mask_replace"]["applied_count"] >= 1
    # v0.8.15: Precise Mask mode is glyph-faithful. Even with high-confidence
    # OCR and legacy fallback flags, OCR must not rewrite wording/punctuation.
    assert project.meta["mask_replace"]["strict_no_ocr_reletter"] is True
    assert project.meta["reletter_applied_count"] == 0
    assert project.lettering == []
    assert not any(q.code == "photo_pair_ocr_evidence_missing" for q in project.qa)


def test_small_photo_pair_uses_crisp_ink_without_ocr():
    """v0.8.2: small photographed bubbles are no longer rejected before transfer."""
    target = np.full((180, 240, 3), 255, np.uint8)
    source = np.full((360, 480, 3), 238, np.uint8)
    tbox = (70, 55, 170, 115)  # min side 60 < legacy 88px rejection gate
    sbox = tuple(v * 2 for v in tbox)
    # Clean target Japanese-like vertical bars.
    for x in (105, 120, 135):
        cv2.rectangle(target, (x, 70), (x + 3, 102), 0, -1)
    # Phone source: translated glyph strokes + an artificial photographed border.
    cv2.rectangle(source, (sbox[0], sbox[1]), (sbox[2], sbox[3]), (15, 15, 15), 4)
    cv2.putText(source, "CN", (sbox[0] + 32, sbox[1] + 82), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (25, 25, 25), 5, cv2.LINE_AA)
    source = cv2.GaussianBlur(source, (5, 5), 1.0)

    sb = _rect_bubble("s-small", sbox, source.shape[:2], photo=True, paired_id="t-small", block_ids=["paired"])
    tb = _rect_bubble("t-small", tbox, target.shape[:2], photo=True, paired_id="s-small")
    H = np.array([[.5, 0, 0], [0, .5, 0], [0, 0, 1]], np.float64)
    reg = RegistrationResult(H, "known", .98, .98, .1, .9, 50, (480, 360), (240, 180), {})
    cfg = MaskReplaceConfig(
        min_match_confidence=.1,
        local_fit="global",
        preserve_target_border=False,
        feather_px=0,
        sr_backend="off",
    )
    result = transfer_bubble_patches(source, target, [sb], [tb], reg, cfg)
    assert result.applied_count == 1, [r.to_dict() for r in result.records]
    rec = result.records[0]
    assert rec.clarity_mode == "photo-crisp-ink"
    # The source's photographed rectangle edge must not become a second inner border.
    assert np.mean(result.image[58:62, 80:160]) > 235
    # But translated ink must remain visibly dark in the interior.
    assert np.mean(cv2.cvtColor(result.image[68:108, 85:155], cv2.COLOR_BGR2GRAY) < 120) > .015


def test_mask_replace_qa_accepts_strong_photo_registration_over_weak_pair_heuristic():
    """Phone-vs-scan perceptual pairing can be weak; verified geometry is stronger evidence."""
    from manga_hd_transfer.mask_transfer import MaskTransferRecord

    pair = PagePair("phone.jpg", "master.jpg", 0, 0, .31, .50, ["review_recommended"])
    reg = RegistrationResult(np.eye(3), "opencv-sift", .93, .90, .6, .65, 500, (100, 100), (100, 100), {})
    rec = MaskTransferRecord("s", "t", .95, True, "applied")
    rec.geometry_mode = "photo_pair"
    rec.mask_iou = .95
    rec.target_coverage = .98
    rec.spill_ratio = .01
    rec.sharpness = 200.0
    issues = run_mask_replace_qa(pair, reg, [], [], [rec], QAConfig(), MaskReplaceConfig())
    item = next(x for x in issues if x.code == "page_pair_low_confidence")
    assert item.severity == "warning"
    assert not any(x.code == "page_pair_low_confidence" and x.severity == "error" for x in issues)


def test_edge_clipped_photo_source_is_not_published_as_complete_translation():
    """v0.8.3: a camera-cropped source bubble must not become partial Chinese output."""
    target = np.full((180, 240, 3), 255, np.uint8)
    # Japanese-like target text inside a complete target bubble.
    tbox = (70, 45, 200, 135)
    for x in (120, 138, 156, 174):
        cv2.rectangle(target, (x, 67), (x + 4, 112), 0, -1)

    # The expected 2x photographed bubble would extend to x=400, but the camera
    # frame ends at x=370. This intentionally reproduces the real 009 failure:
    # source mask touches the image edge and covers ~85-90% of the target.
    source = np.full((360, 370, 3), 240, np.uint8)
    sbox = (140, 90, 370, 270)
    cv2.putText(source, "CN", (205, 190), cv2.FONT_HERSHEY_SIMPLEX, 2.1, (20, 20, 20), 5, cv2.LINE_AA)
    sb = _rect_bubble("s-edge", sbox, source.shape[:2], photo=True, paired_id="t-edge", block_ids=["paired"])
    tb = _rect_bubble("t-edge", tbox, target.shape[:2], photo=True, paired_id="s-edge")
    H = np.array([[.5, 0, 0], [0, .5, 0], [0, 0, 1]], np.float64)
    reg = RegistrationResult(H, "known-edge-crop", .99, .99, .1, .9, 50, (370, 360), (240, 180), {})
    cfg = MaskReplaceConfig(
        min_match_confidence=.1,
        local_fit="global",
        preserve_target_border=False,
        feather_px=0,
        sr_backend="off",
        photo_pair_min_transfer_iou=.70,
        photo_pair_min_transfer_coverage=.84,
        photo_pair_max_spill_ratio=.30,
        photo_pair_edge_clip_guard_enabled=True,
        photo_pair_edge_clip_min_target_coverage=.94,
    )
    out = transfer_bubble_patches(source, target, [sb], [tb], reg, cfg)
    # v0.8.6 review-first behaviour: show recoverable Chinese instead of silently
    # leaving Japanese, but flag it as a non-publishable candidate.
    assert out.applied_count == 1, [r.to_dict() for r in out.records]
    rec = out.records[0]
    assert rec.source_edge_clipped is True
    assert "right" in rec.source_edge_sides
    assert rec.target_coverage >= cfg.photo_pair_min_transfer_coverage
    assert rec.target_coverage < cfg.photo_pair_edge_clip_min_target_coverage
    assert rec.reason == "applied_low_confidence_candidate"
    assert rec.candidate and rec.review_required and rec.restorable and rec.editable
    assert rec.review_reason == "source_text_region_clipped_at_page_edge"
    assert not np.array_equal(out.image, target)


def test_edge_touching_photo_bubble_with_near_complete_coverage_still_transfers():
    """Touching the camera edge alone is not fatal when the captured text is complete."""
    target = np.full((160, 220, 3), 255, np.uint8)
    tbox = (150, 55, 210, 115)
    for x in (172, 184):
        cv2.rectangle(target, (x, 72), (x + 3, 100), 0, -1)
    source = np.full((320, 420, 3), 242, np.uint8)
    # Full mapped width is 120px in source; source frame ends exactly on the mask.
    sbox = (300, 110, 420, 230)
    cv2.putText(source, "OK", (315, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (15, 15, 15), 4, cv2.LINE_AA)
    sb = _rect_bubble("s-edge-ok", sbox, source.shape[:2], photo=True, paired_id="t-edge-ok", block_ids=["paired"])
    tb = _rect_bubble("t-edge-ok", tbox, target.shape[:2], photo=True, paired_id="s-edge-ok")
    H = np.array([[.5, 0, 0], [0, .5, 0], [0, 0, 1]], np.float64)
    reg = RegistrationResult(H, "known-edge-complete", .99, .99, .1, .9, 50, (420, 320), (220, 160), {})
    cfg = MaskReplaceConfig(
        min_match_confidence=.1,
        local_fit="global",
        preserve_target_border=False,
        feather_px=0,
        sr_backend="off",
        photo_pair_min_transfer_iou=.70,
        photo_pair_min_transfer_coverage=.84,
        photo_pair_max_spill_ratio=.30,
        photo_pair_edge_clip_guard_enabled=True,
        photo_pair_edge_clip_min_target_coverage=.94,
    )
    out = transfer_bubble_patches(source, target, [sb], [tb], reg, cfg)
    assert out.applied_count == 1, [r.to_dict() for r in out.records]
    rec = out.records[0]
    assert rec.source_edge_clipped is True
    assert rec.target_coverage >= .94
    assert rec.reason.startswith("applied")


def test_edge_clipped_rejection_has_specific_blocking_qa():
    from manga_hd_transfer.mask_transfer import MaskTransferRecord

    pair = PagePair("phone.jpg", "master.jpg", 0, 0, .99, .01, [])
    reg = RegistrationResult(np.eye(3), "known", .95, 1.0, 0.0, .9, 30, (100, 100), (100, 100), {})
    rec = MaskTransferRecord("s", "t", .9, False, "source_text_region_clipped_at_page_edge")
    rec.geometry_mode = "photo_pair"
    rec.target_coverage = .86
    rec.mask_iou = .85
    rec.source_edge_clipped = True
    rec.source_edge_sides = "right"
    issues = run_mask_replace_qa(pair, reg, [], [], [rec], QAConfig(), MaskReplaceConfig())
    item = next(x for x in issues if x.code == "mask_replace_source_translation_clipped")
    assert item.severity == "error"
    assert item.meta["source_edge_sides"] == "right"


def test_edge_clip_integrity_error_does_not_mislabel_strong_registration_as_bad_pair():
    from manga_hd_transfer.mask_transfer import MaskTransferRecord

    pair = PagePair("phone.jpg", "master.jpg", 0, 0, .30, .60, ["review_recommended"])
    reg = RegistrationResult(np.eye(3), "opencv-sift", .90, .90, .5, .6, 100, (100, 100), (100, 100), {})
    ok = MaskTransferRecord("s-ok", "t-ok", .95, True, "applied")
    ok.geometry_mode = "photo_pair"
    clipped = MaskTransferRecord("s-clip", "t-clip", .85, False, "source_text_region_clipped_at_page_edge")
    clipped.geometry_mode = "photo_pair"
    clipped.source_edge_clipped = True
    clipped.source_edge_sides = "right"
    clipped.target_coverage = .86
    issues = run_mask_replace_qa(pair, reg, [], [], [ok, clipped], QAConfig(), MaskReplaceConfig())
    page_item = next(x for x in issues if x.code == "page_pair_low_confidence")
    assert page_item.severity == "warning"
    assert any(x.code == "mask_replace_source_translation_clipped" and x.severity == "error" for x in issues)


def test_edge_clipped_photo_region_never_falls_back_to_partial_ocr_reletter(tmp_path, monkeypatch):
    """v0.8.3: source-edge integrity blocks OCR fallback as well as pixel transfer."""
    import manga_hd_transfer.pipeline as pipeline_mod
    from manga_hd_transfer.io_utils import write_image

    h, w = 320, 760
    target = np.full((h, w, 3), 245, np.uint8)
    # Complete clean target bubble stops before the target image edge.
    cv2.ellipse(target, (690, 160), (65, 70), 0, 0, 360, 255, -1)
    cv2.ellipse(target, (690, 160), (65, 70), 0, 0, 360, 0, 3)
    for x in (675, 690, 705):
        cv2.rectangle(target, (x, 125), (x + 4, 195), 0, -1)

    source = target.copy()
    # Simulate a phone frame cutting away the right side of the translated
    # bubble. The paired source mask reaches x == image width; only partial OCR
    # text can physically exist in this input.
    source[:, 755:] = 238
    cv2.rectangle(source, (660, 130), (748, 195), 255, -1)
    cv2.putText(source, "PART", (662, 175), cv2.FONT_HERSHEY_SIMPLEX, .75, 0, 2, cv2.LINE_AA)

    sp, tp = tmp_path / "source.png", tmp_path / "target.png"
    write_image(sp, source); write_image(tp, target)

    # OCR deliberately returns a plausible but incomplete Chinese string. This
    # must NOT be used to reletter the clean target bubble.
    sblocks = [TextBlock("s1", [(655,120),(754,120),(754,205),(655,205)], "残缺中文", .99, reading_order=0)]
    tblocks = [TextBlock("t1", [(640,120),(735,120),(735,205),(640,205)], "日本語全文", .99, reading_order=0)]

    sm = np.zeros((h, w), np.uint8)
    cv2.rectangle(sm, (635, 95), (759, 225), 255, -1)  # touches right frame
    tm = np.zeros((h, w), np.uint8)
    cv2.rectangle(tm, (625, 95), (755, 225), 255, -1)  # complete HD region
    sb = BubbleInstance(
        "photo-src-edge", [(635,95),(760,95),(760,225),(635,225)], .99, "speech", ["paired"],
        sm, sm.copy(), {"paired_diff_method":"photo_pair","mask_is_interior":True,"paired_target_id":"photo-dst-edge"}
    )
    tb = BubbleInstance(
        "photo-dst-edge", [(625,95),(755,95),(755,225),(625,225)], .99, "speech", [],
        tm, tm.copy(), {"paired_diff_method":"photo_pair","mask_is_interior":True,"paired_source_id":"photo-src-edge"}
    )
    pd = PairedDiffResult(
        [sb], [tb], tm.copy(),
        [DiffBubbleRecord(sb.id, tb.id, .8, .90, .95, (625,95,755,225), "photo_pair", .08, .08)],
        100.0, 120.0, method="photo_pair", safe_to_skip_ocr=False,
        diagnostics={"incomplete_requires_ocr": True},
    )
    reg = RegistrationResult(np.eye(3), "identity", .99, 1.0, 0.0, .9, 50, (w,h), (w,h), {})
    monkeypatch.setattr(pipeline_mod, "extract_paired_diff_bubbles", lambda *args, **kwargs: pd)
    monkeypatch.setattr(pipeline_mod, "register_images", lambda *args, **kwargs: reg)

    cfg = PipelineConfig()
    cfg.transfer.mode = "mask_replace"
    cfg.export.layer_bundle = False
    cfg.export.save_debug = False
    cfg.cache.enabled = False
    cfg.bubbles.backend = "seeded_white"
    cfg.matching.review_confidence = .20
    cfg.qa.registration_min_confidence = .20
    cfg.qa.match_min_confidence = .20
    cfg.mask_replace.min_match_confidence = .1
    cfg.mask_replace.local_fit = "global"
    cfg.mask_replace.feather_px = 0
    cfg.mask_replace.source_mask_expand_px = 0
    cfg.mask_replace.photo_pair_salvage_max_expand_px = 0
    cfg.mask_replace.photo_pair_min_transfer_iou = .70
    cfg.mask_replace.photo_pair_min_transfer_coverage = .84
    cfg.mask_replace.photo_pair_edge_clip_min_target_coverage = .94
    cfg.mask_replace.photo_pair_require_ocr_evidence = True

    pipe = TransferPipeline(cfg, InjectedOCRBackend(sblocks), InjectedOCRBackend(tblocks))
    project = pipe.process_page(PagePair(str(sp), str(tp), 0, 0, .99, .01, []), tmp_path / "out")

    assert project.meta["mask_replace"]["applied_count"] == 1
    rec = project.meta["mask_replace"]["records"][0]
    assert rec["reason"] == "applied_low_confidence_candidate"
    assert rec["candidate"] is True and rec["review_required"] is True
    # Partial OCR must still NOT be promoted into a clean-looking trusted reletter.
    assert project.meta["reletter_applied_count"] == 0
    assert not any(x.success and x.text == "残缺中文" for x in project.lettering)
    queue = project.meta["mask_replace"]["manual_reletter_required"]
    assert len(queue) == 1 and queue[0]["target_bubble_id"] == "photo-dst-edge"
    assert queue[0]["candidate_applied"] is True
    assert any(q.code == "mask_replace_low_confidence_candidate" and q.severity == "error" for q in project.qa)
    assert (tmp_path / "out" / "review_preview.png").exists()


def test_photo_structural_supplement_keeps_distant_bright_text():
    from manga_hd_transfer.paired_diff import _structural_supplement_for_photo
    shape = (400, 400)
    target = np.full((shape[0], shape[1], 3), 245, np.uint8)
    pm = np.zeros(shape, np.uint8); cv2.rectangle(pm, (40, 40), (140, 130), 255, -1)
    sm = np.zeros(shape, np.uint8); cv2.rectangle(sm, (230, 220), (280, 320), 255, -1)
    # Put target text-like dark strokes inside the structural region while keeping
    # the average bright enough for a manga speech area.
    for x in (242, 255, 268): cv2.rectangle(target, (x, 235), (x+1, 300), 0, -1)
    pb = BubbleInstance('photo-dst-0', [(40,40),(140,40),(140,130),(40,130)], .9, 'speech', [], pm, pm.copy(), {'paired_diff_method':'photo_pair'})
    ps = BubbleInstance('photo-src-0', [(40,40),(140,40),(140,130),(40,130)], .9, 'speech', ['x'], pm.copy(), pm.copy(), {'paired_diff_method':'photo_pair'})
    photo = PairedDiffResult([ps],[pb],pm.copy(),[],1,1,method='photo_pair',safe_to_skip_ocr=False)
    sb = BubbleInstance('diff-src-0', [(230,220),(280,220),(280,320),(230,320)], .8, 'speech', ['x'], sm.copy(), sm.copy(), {})
    tb = BubbleInstance('diff-dst-0', [(230,220),(280,220),(280,320),(230,320)], .8, 'speech', [], sm.copy(), sm.copy(), {})
    rec = DiffBubbleRecord('diff-src-0','diff-dst-0',.05,1,.8,(230,220,281,321),method='structural_v08',region_kind='free_text',changed_pixels=150)
    structural = PairedDiffResult([sb],[tb],sm.copy(),[rec],1,1,method='structural_v08',safe_to_skip_ocr=False,aligned_source=target.copy())
    out = _structural_supplement_for_photo(photo, structural, target)
    assert out is not None and len(out.records) == 1
    assert out.target_bubbles[0].meta['photo_source'] is True


def test_photo_structural_supplement_rejects_near_photo_region():
    from manga_hd_transfer.paired_diff import _structural_supplement_for_photo
    shape = (400, 400)
    target = np.full((shape[0], shape[1], 3), 245, np.uint8)
    pm = np.zeros(shape, np.uint8); cv2.rectangle(pm, (40, 40), (140, 130), 255, -1)
    sm = np.zeros(shape, np.uint8); cv2.rectangle(sm, (100, 135), (160, 200), 255, -1)
    for x in (112, 125, 138): cv2.rectangle(target, (x, 145), (x+3, 185), 0, -1)
    pb = BubbleInstance('photo-dst-0', [(40,40),(140,40),(140,130),(40,130)], .9, 'speech', [], pm, pm.copy(), {'paired_diff_method':'photo_pair'})
    ps = BubbleInstance('photo-src-0', [(40,40),(140,40),(140,130),(40,130)], .9, 'speech', ['x'], pm.copy(), pm.copy(), {'paired_diff_method':'photo_pair'})
    photo = PairedDiffResult([ps],[pb],pm.copy(),[],1,1,method='photo_pair',safe_to_skip_ocr=False)
    sb = BubbleInstance('diff-src-0', [(100,135),(160,135),(160,200),(100,200)], .8, 'speech', ['x'], sm.copy(), sm.copy(), {})
    tb = BubbleInstance('diff-dst-0', [(100,135),(160,135),(160,200),(100,200)], .8, 'speech', [], sm.copy(), sm.copy(), {})
    rec = DiffBubbleRecord('diff-src-0','diff-dst-0',.08,1,.8,(100,135,161,201),method='structural_v08',region_kind='free_text',changed_pixels=160)
    structural = PairedDiffResult([sb],[tb],sm.copy(),[rec],1,1,method='structural_v08',safe_to_skip_ocr=False,aligned_source=target.copy())
    assert _structural_supplement_for_photo(photo, structural, target) is None


def test_photo_color_sfx_rebuilds_changed_red_group_only():
    from manga_hd_transfer.mask_transfer import transfer_photo_color_sfx
    target = np.full((600, 800, 3), 245, np.uint8)
    source = target.copy()
    # Stable red artwork: identical in both images and must not trigger.
    cv2.circle(target, (700, 500), 22, (30, 30, 190), -1)
    cv2.circle(source, (700, 500), 22, (30, 30, 190), -1)
    # Target/source SFX use different red shapes in the same broad area.
    for x in (60, 95, 130):
        cv2.rectangle(target, (x, 55), (x + 18, 130), (25, 25, 210), -1)
    for x in (55, 90, 125, 160):
        cv2.circle(source, (x + 10, 92), 18, (25, 25, 210), -1)
    out = transfer_photo_color_sfx(source, target, MaskReplaceConfig())
    assert out.applied_count == 1
    assert out.records[0].clarity_mode == 'color-sfx-rebuild'
    # Stable art stays intact.
    assert np.array_equal(out.image[500, 700], target[500, 700])


def test_cross_rendition_clear_mask_absorbs_nearby_old_glyph_fringe():
    from manga_hd_transfer.mask_transfer import _expand_target_clear_mask_with_text_components

    target = np.full((180, 220, 3), 255, np.uint8)
    # Balloon outline; detector mask intentionally starts below a compact old glyph.
    cv2.ellipse(target, (110, 90), (70, 55), 0, 0, 360, 0, 2)
    cv2.putText(target, "A", (92, 52), cv2.FONT_HERSHEY_SIMPLEX, .65, 0, 2, cv2.LINE_AA)
    mask = np.zeros(target.shape[:2], np.uint8)
    cv2.ellipse(mask, (110, 98), (62, 42), 0, 0, 360, 255, -1)
    before = mask.copy()
    expanded = _expand_target_clear_mask_with_text_components(target, mask)
    # Compact glyph pixels immediately above the imperfect detector mask are added.
    glyph = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) < 190
    assert np.count_nonzero((expanded > 0) & glyph) > np.count_nonzero((before > 0) & glyph)
    # Expansion remains local instead of flooding the page.
    assert cv2.countNonZero(expanded) < cv2.countNonZero(mask) * 1.30


def test_cross_rendition_route_detects_monochrome_source_to_colour_target():
    from manga_hd_transfer.pipeline import _cross_rendition_monochrome_source

    mono = np.full((80, 100, 3), 230, np.uint8)
    cv2.putText(mono, "CN", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
    colour = mono.copy()
    colour[:, :50] = (40, 150, 230)
    assert _cross_rendition_monochrome_source(mono, colour)
    assert not _cross_rendition_monochrome_source(colour, colour)


def test_monochrome_to_color_hint_for_skipping_empty_structural_supplement():
    from manga_hd_transfer.paired_diff import _is_monochrome_to_color_pair
    mono = np.full((120, 90, 3), 245, np.uint8)
    cv2.putText(mono, "A", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20,20,20), 2)
    color = mono.copy()
    color[:, :30] = (30, 180, 240)
    assert _is_monochrome_to_color_pair(mono, color)
    assert not _is_monochrome_to_color_pair(color, color)


def test_glyph_footprint_rescue_keeps_complete_source_raster_without_ocr():
    """v0.8.16: a translated glyph straddling the target mask is moved, not clipped/retyped."""
    from manga_hd_transfer.mask_transfer import transfer_paired_diff_regions

    shape = (600, 800)
    target = np.full((shape[0], shape[1], 3), 255, np.uint8)
    tm = np.zeros(shape, np.uint8)
    cv2.ellipse(tm, (200, 180), (75, 70), 0, 0, 360, 255, -1)
    # Old Japanese lettering that must be cleared by the mask path.
    for x in (180, 195, 210):
        cv2.rectangle(target, (x, 145), (x + 3, 215), 0, -1)

    aligned_source = np.full_like(target, 245)
    # Two compact vertical "columns".  The upper/right source glyph footprint is
    # deliberately outside the curved target mask, reproducing the real 007 page.
    for x in (200, 230):
        for y in (120, 147, 174, 201):
            cv2.rectangle(aligned_source, (x, y), (x + 18, y + 19), (20, 20, 20), 2)
            cv2.line(aligned_source, (x + 3, y + 10), (x + 15, y + 10), (20, 20, 20), 2)

    poly = [(125, 110), (275, 110), (275, 250), (125, 250)]
    sb = BubbleInstance(
        "photo-src-000", poly, .99, "speech", ["paired"], tm.copy(), tm.copy(),
        {"paired_diff_method": "photo_pair", "paired_target_id": "photo-dst-000", "mask_is_interior": True},
    )
    tb = BubbleInstance(
        "photo-dst-000", poly, .99, "speech", [], tm.copy(), tm.copy(),
        {"paired_diff_method": "photo_pair", "paired_source_id": "photo-src-000", "mask_is_interior": True},
    )
    cfg = MaskReplaceConfig(
        photo_pair_glyph_rescue_max_area_ratio=.08,
        preserve_target_border=False,
        feather_px=0,
    )
    out = transfer_paired_diff_regions(aligned_source, target, [sb], [tb], cfg)
    assert out.applied_count == 1
    rec = out.records[0]
    assert rec.clarity_mode == "photo-glyph-footprint-rescue"
    assert rec.sr_scale == 1.0  # translation alone is enough; glyph shapes stay 1:1
    assert abs(rec.local_dx) + abs(rec.local_dy) > 0

    # Nearly all source raster ink survives, but after rescue it is fully inside
    # the target bubble.  No OCR transcript/font renderer is involved in this path.
    src_dark = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2GRAY) < 180
    out_dark = cv2.cvtColor(out.image, cv2.COLOR_BGR2GRAY) < 180
    assert np.count_nonzero(out_dark & (tm > 0)) >= np.count_nonzero(src_dark) * 0.90
    assert np.count_nonzero(out_dark & (tm == 0)) == 0
