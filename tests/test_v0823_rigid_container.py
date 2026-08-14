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
    assert rec.clarity_mode in {"locked-source-raster", "locked-source-container-patch"}
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
    assert result.records[0].sr_backend in {"rigid-container-raster", "rigid-container-full-patch"}

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
