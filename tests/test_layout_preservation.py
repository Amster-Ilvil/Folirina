from manga_hd_transfer.config import MaskReplaceConfig
from manga_hd_transfer.mask_transfer import MaskTransferRecord
from manga_hd_transfer.pipeline import _should_preserve_transferred_layout, _preserved_layout_looks_complete


def test_sharp_transferred_scan_stays_source_typeset_even_when_ocr_exists():
    cfg = MaskReplaceConfig()
    rec = MaskTransferRecord('s','t',.99,True,'applied')
    rec.clarity_mode = 'photo-crisp-ink'
    rec.relative_sharpness = 4.2
    assert _should_preserve_transferred_layout(rec, cfg)


def test_blurry_or_review_candidate_still_promotes_to_ocr_reletter():
    cfg = MaskReplaceConfig()
    rec = MaskTransferRecord('s','t',.99,True,'applied')
    rec.clarity_mode = 'photo-crisp-ink'
    rec.relative_sharpness = .55
    assert not _should_preserve_transferred_layout(rec, cfg)
    rec.relative_sharpness = 3.0
    rec.review_required = True
    assert not _should_preserve_transferred_layout(rec, cfg)


def test_preserved_layout_rejected_when_vertical_columns_collapse():
    source = {
        "orientation": "vertical",
        "columns": 2,
        "rows": 6,
        "glyph_pitch_px": 18.0,
        "ink_bbox_size": [40, 112],
        "container_size": [56, 128],
    }
    # Simulate the real regression: sharp transfer but the leading column got clipped away.
    target = {
        "orientation": "vertical",
        "columns": 1,
        "rows": 6,
        "glyph_pitch_px": 17.0,
        "ink_bbox_size": [17, 110],
        "container_size": [56, 128],
    }
    assert not _preserved_layout_looks_complete(source, target)


def test_preserved_layout_kept_when_grid_stays_complete():
    source = {
        "orientation": "vertical",
        "columns": 2,
        "rows": 6,
        "glyph_pitch_px": 18.0,
        "ink_bbox_size": [40, 112],
        "container_size": [56, 128],
    }
    target = {
        "orientation": "vertical",
        "columns": 2,
        "rows": 6,
        "glyph_pitch_px": 17.5,
        "ink_bbox_size": [38, 110],
        "container_size": [56, 128],
    }
    assert _preserved_layout_looks_complete(source, target)


def test_masked_layout_profile_has_local_bbox_dependency():
    import numpy as np
    from manga_hd_transfer.pipeline import _masked_layout_profile
    image = np.full((80, 60, 3), 255, np.uint8)
    mask = np.zeros((80, 60), np.uint8)
    mask[10:70, 8:52] = 255
    # Two rough vertical ink columns.
    image[18:60, 18:23] = 0
    image[15:58, 36:41] = 0
    profile = _masked_layout_profile(image, mask, "测试文字测试文字", "vertical")
    assert profile
    assert profile["orientation"] == "vertical"
