from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.config import MaskReplaceConfig
from manga_hd_transfer.mask_transfer import transfer_rigid_container_rasters
from manga_hd_transfer.models import BubbleInstance


def _bubble(bid: str, tid: str, mask: np.ndarray) -> BubbleInstance:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    poly = [(float(x), float(y)) for x, y in contour]
    return BubbleInstance(bid, poly, 0.98, "speech", ["text"], mask, mask.copy(), {"paired_target_id": tid})


def test_rigid_container_clears_whole_target_and_uses_uniform_scale() -> None:
    cfg = MaskReplaceConfig()
    src = np.full((220, 220, 3), 255, np.uint8)
    tgt = np.full((170, 180, 3), 255, np.uint8)
    sm = np.zeros(src.shape[:2], np.uint8)
    tm = np.zeros(tgt.shape[:2], np.uint8)
    cv2.ellipse(sm, (110, 110), (70, 76), 0, 0, 360, 255, -1)
    cv2.ellipse(tm, (90, 85), (52, 58), 0, 0, 360, 255, -1)
    # Chinese source: one tall glyph-like raster and punctuation, deliberately at
    # different positions from the Japanese target ink.
    cv2.rectangle(src, (92, 70), (105, 132), (0, 0, 0), -1)
    cv2.rectangle(src, (115, 78), (127, 128), (25, 25, 25), -1)
    cv2.circle(src, (110, 145), 4, (0, 0, 0), -1)
    # Japanese target strokes elsewhere in the same container.
    cv2.rectangle(tgt, (65, 60), (74, 116), (0, 0, 0), -1)
    cv2.rectangle(tgt, (118, 58), (127, 120), (0, 0, 0), -1)

    sb = _bubble("src-0", "dst-0", sm)
    tb = _bubble("dst-source-unused", "unused", tm)
    tb.id = "dst-0"
    result = transfer_rigid_container_rasters(src, tgt, tgt, [sb], [tb], cfg)
    assert result.applied_count == 1
    rec = result.records[0]
    assert rec.geometry_mode == "rigid_uniform_container"
    assert rec.clarity_mode == "locked-source-container-patch"
    assert rec.content_complete
    assert rec.source_ink_coverage >= 0.995
    assert rec.target_residual_ratio == 0.0
    # The source raster is resized by a single scalar. The tall source glyph must
    # stay tall rather than inheriting the deliberately different X/Y ellipse scale.
    gray = cv2.cvtColor(result.image, cv2.COLOR_BGR2GRAY)
    ink = ((gray < 120) & (tm > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    comps = [tuple(map(int, stats[i])) for i in range(1, n) if int(stats[i, cv2.CC_STAT_AREA]) >= 12]
    assert comps
    ratios = [h / max(1, w) for x, y, w, h, area in comps]
    assert max(ratios) > 3.0


def test_rigid_container_rejects_coloured_burst_background() -> None:
    cfg = MaskReplaceConfig()
    src = np.full((180, 180, 3), 255, np.uint8)
    tgt = np.full((180, 180, 3), 255, np.uint8)
    sm = np.zeros((180, 180), np.uint8); tm = np.zeros((180, 180), np.uint8)
    cv2.rectangle(sm, (30, 30), (150, 150), 255, -1)
    cv2.rectangle(tm, (30, 30), (150, 150), 255, -1)
    src[60:120, 80:96] = 0
    tgt[30:151, 30:151] = (0, 235, 255)  # saturated yellow burst
    tgt[60:120, 105:120] = 0
    sb = _bubble("src-0", "dst-0", sm); tb = _bubble("tmp", "unused", tm); tb.id = "dst-0"
    result = transfer_rigid_container_rasters(src, tgt, tgt, [sb], [tb], cfg)
    assert result.applied_count == 0
    assert np.array_equal(result.image, tgt)


def test_rigid_container_does_not_require_ocr_or_text_components() -> None:
    cfg = MaskReplaceConfig()
    src = np.full((140, 140, 3), 255, np.uint8); tgt = src.copy()
    sm = np.zeros((140, 140), np.uint8); tm = np.zeros((140, 140), np.uint8)
    cv2.ellipse(sm, (70, 70), (45, 48), 0, 0, 360, 255, -1)
    cv2.ellipse(tm, (70, 70), (44, 47), 0, 0, 360, 255, -1)
    # disconnected CJK-like strokes + punctuation: the route must move the whole
    # raster field without needing component grouping.
    for x in (52, 62, 76, 86):
        cv2.rectangle(src, (x, 48), (x + 4, 90), 0, -1)
    cv2.circle(src, (70, 104), 3, 0, -1)
    cv2.rectangle(tgt, (58, 55), (64, 100), 0, -1)
    sb = _bubble("src-0", "dst-0", sm); tb = _bubble("tmp", "unused", tm); tb.id = "dst-0"
    result = transfer_rigid_container_rasters(src, tgt, tgt, [sb], [tb], cfg)
    assert result.applied_count == 1
    assert result.records[0].sr_backend == "rigid-container-full-patch"

def test_incomplete_complex_candidate_is_not_published(monkeypatch) -> None:
    # Structural supplements are intentionally optional.  If their independent
    # content audit fails, the page must stay byte-for-byte at the pre-candidate
    # target instead of publishing a clipped/half-erased text patch.
    import manga_hd_transfer.mask_transfer as mt

    cfg = MaskReplaceConfig()
    src = np.full((120, 120, 3), 255, np.uint8)
    tgt = np.full((120, 120, 3), 255, np.uint8)
    sm = np.zeros((120, 120), np.uint8); tm = np.zeros((120, 120), np.uint8)
    cv2.rectangle(sm, (30, 30), (90, 90), 255, -1)
    cv2.rectangle(tm, (30, 30), (90, 90), 255, -1)
    sb = _bubble("src-c", "dst-c", sm)
    tb = _bubble("tmp", "unused", tm); tb.id = "dst-c"
    tb.meta["paired_region_kind"] = "complex_text"

    source_ink = np.zeros((120, 120), np.uint8)
    cv2.rectangle(source_ink, (45, 40), (54, 80), 255, -1)
    target_ink = np.zeros((120, 120), np.uint8)
    cv2.rectangle(target_ink, (65, 40), (74, 80), 255, -1)
    write = tm.copy()
    # Candidate intentionally fails: it contains none of the expected source ink
    # and leaves target-only dark ink in place.
    bad = tgt.copy()
    bad[target_ink > 0] = 0

    def fake_transfer(aligned_source, rendered, mask, cfg):
        return bad.copy(), write.copy(), source_ink.copy(), {
            "target_ink_mask": target_ink.copy(),
            "clear_mask": write.copy(),
            "boundary_touch": False,
        }

    monkeypatch.setattr(mt, "_transfer_open_complex_text_region", fake_transfer)
    result = mt.transfer_paired_diff_regions(src, tgt, [sb], [tb], cfg)
    assert len(result.records) == 1
    rec = result.records[0]
    # Newer review-first policy publishes a reversible candidate instead of
    # restoring Japanese when recoverable Chinese pixels exist.
    assert rec.applied
    assert rec.reason == "applied_incomplete_review_candidate"
    assert rec.candidate and rec.review_required and rec.restorable and rec.editable
    assert not rec.content_complete
    assert cv2.countNonZero(result.composite_mask) > 0


def test_rigid_container_accepts_slight_inner_size_difference() -> None:
    cfg = MaskReplaceConfig()
    src = np.full((220, 220, 3), 255, np.uint8)
    tgt = np.full((208, 205, 3), 255, np.uint8)
    sm = np.zeros(src.shape[:2], np.uint8)
    tm = np.zeros(tgt.shape[:2], np.uint8)
    cv2.rectangle(sm, (24, 24), (194, 192), 255, -1)
    cv2.rectangle(tm, (26, 22), (179, 187), 255, -1)
    # Dense vertical Chinese dialogue plus punctuation close to the target edge.
    for x in (76, 92, 110, 128, 146):
        cv2.rectangle(src, (x, 52), (x + 9, 154), (0, 0, 0), -1)
    cv2.circle(src, (104, 171), 5, (0, 0, 0), -1)
    # Japanese target ink occupies a slightly different inner footprint.
    for x in (62, 78, 138):
        cv2.rectangle(tgt, (x, 48), (x + 8, 148), (0, 0, 0), -1)
    sb = _bubble("src-fit", "dst-fit", sm)
    tb = _bubble("tmp", "unused", tm); tb.id = "dst-fit"
    result = transfer_rigid_container_rasters(src, tgt, tgt, [sb], [tb], cfg)
    assert result.applied_count == 1
    rec = result.records[0]
    assert rec.content_complete
    assert rec.source_ink_coverage >= 0.985
    assert rec.target_residual_ratio <= 0.02
    assert rec.geometry_mode == "rigid_uniform_container"


def test_rigid_container_solidifies_text_notches_before_clear_and_clip() -> None:
    """Regression for v0.8.26: text-touching notches are not real container holes."""
    cfg = MaskReplaceConfig()
    h = w = 180
    src = np.full((h, w, 3), 255, np.uint8)
    tgt = np.full((h, w, 3), 255, np.uint8)
    sm = np.zeros((h, w), np.uint8)
    tm = np.zeros((h, w), np.uint8)
    cv2.rectangle(sm, (30, 28), (150, 150), 255, -1)
    cv2.rectangle(tm, (30, 28), (150, 150), 255, -1)

    # Simulate bright-region detector masks being cut open by dark glyph columns
    # that touch the top edge of the detected white component.
    sm[28:72, 57:64] = 0
    tm[28:74, 108:115] = 0
    # Chinese source and Japanese target glyphs occupy those notches.
    src[34:70, 58:63] = 0
    src[78:122, 72:79] = 0
    tgt[34:72, 109:114] = 0
    tgt[82:124, 118:125] = 0

    sb = _bubble("src-notch", "dst-notch", sm)
    tb = _bubble("tmp", "unused", tm); tb.id = "dst-notch"
    result = transfer_rigid_container_rasters(src, tgt, tgt, [sb], [tb], cfg)
    assert result.applied_count == 1
    rec = result.records[0]
    assert rec.content_complete
    assert rec.target_residual_ratio <= cfg.rigid_container_acceptance_max_target_residual

    gray = cv2.cvtColor(result.image, cv2.COLOR_BGR2GRAY)
    # Source Chinese notch stroke survives because source clipping uses the solid
    # interior, while the Japanese-only notch is cleared.
    assert float(np.mean(gray[38:68, 58:63])) < 120.0
    assert float(np.mean(gray[38:68, 109:114])) > 235.0
    # The exported clear mask itself must cover the Japanese notch.
    assert np.all(result.clear_mask[38:68, 109:114] > 0)


def test_rigid_container_full_patch_copies_white_background_with_text() -> None:
    cfg = MaskReplaceConfig()
    src = np.full((160, 160, 3), 235, np.uint8)
    tgt = np.full((160, 160, 3), 180, np.uint8)
    sm = np.zeros((160, 160), np.uint8); tm = np.zeros((160, 160), np.uint8)
    cv2.rectangle(sm, (35, 30), (125, 130), 255, -1)
    cv2.rectangle(tm, (35, 30), (125, 130), 255, -1)
    # Source white textbox with Chinese-like strokes.
    src[sm > 0] = 255
    cv2.rectangle(src, (65, 48), (72, 104), 0, -1)
    cv2.rectangle(src, (85, 56), (92, 112), 0, -1)
    # Target has a grey-tinted box area and Japanese strokes at different x.
    tgt[tm > 0] = 250
    cv2.rectangle(tgt, (48, 48), (55, 104), 0, -1)
    cv2.rectangle(tgt, (103, 56), (110, 112), 0, -1)
    sb = _bubble("src-1", "dst-1", sm); tb = _bubble("tmp", "unused", tm); tb.id = "dst-1"
    result = transfer_rigid_container_rasters(src, tgt, tgt, [sb], [tb], cfg)
    assert result.applied_count == 1
    rec = result.records[0]
    assert rec.sr_backend == "rigid-container-full-patch"
    assert rec.content_complete
    # The target Japanese-only x positions are cleared by the opaque source patch.
    gray = cv2.cvtColor(result.image, cv2.COLOR_BGR2GRAY)
    assert float(np.mean(gray[58:100, 49:54])) > 230.0
    assert float(np.mean(gray[64:108, 104:109])) > 230.0
    # Source Chinese x positions are present.
    assert float(np.mean(gray[50:102, 66:71])) < 80.0
    assert float(np.mean(gray[58:110, 86:91])) < 80.0


def test_rigid_container_full_patch_accepts_white_spiky_burst() -> None:
    cfg = MaskReplaceConfig()
    h = w = 240
    src = np.full((h, w, 3), 230, np.uint8)
    tgt = np.full((h, w, 3), (180, 210, 230), np.uint8)
    sm = np.zeros((h, w), np.uint8); tm = np.zeros((h, w), np.uint8)

    def star(cx: int, cy: int, ro: int, ri: int, n: int = 10) -> np.ndarray:
        pts = []
        for i in range(n * 2):
            a = -np.pi / 2 + i * np.pi / n
            r = ro if i % 2 == 0 else ri
            pts.append((int(round(cx + r * np.cos(a))), int(round(cy + r * np.sin(a)))))
        return np.asarray(pts, np.int32)

    cv2.fillPoly(sm, [star(120, 120, 86, 38)], 255)
    cv2.fillPoly(tm, [star(120, 120, 84, 37)], 255)
    src[sm > 0] = 255; tgt[tm > 0] = 255
    # Chinese source strokes vs Japanese target strokes in different columns.
    for x in (102, 114, 126):
        cv2.rectangle(src, (x, 88), (x + 7, 148), 0, -1)
    for x in (92, 142):
        cv2.rectangle(tgt, (x, 92), (x + 7, 146), 0, -1)

    sb = _bubble("src-star", "dst-star", sm)
    tb = _bubble("tmp", "unused", tm); tb.id = "dst-star"
    result = transfer_rigid_container_rasters(src, tgt, tgt, [sb], [tb], cfg)
    assert result.applied_count == 1
    rec = result.records[0]
    assert rec.sr_backend == "rigid-container-full-patch"
    assert rec.clarity_mode == "locked-source-container-patch"
    assert rec.content_complete
    assert rec.source_ink_coverage >= 0.985
    assert rec.target_residual_ratio <= 0.02
