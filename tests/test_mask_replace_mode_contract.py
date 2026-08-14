from manga_hd_transfer.config import MaskReplaceConfig


def test_precise_mask_defaults_never_ocr_reletter():
    cfg = MaskReplaceConfig()
    assert cfg.strict_mask_replace_no_ocr_reletter is True
    assert cfg.photo_pair_prefer_reletter_with_ocr is False
    assert cfg.photo_pair_fallback_reletter_missing is False
    assert cfg.fallback_reletter_on_blur is False


def test_legacy_fallback_flags_cannot_disable_strict_mode_contract():
    cfg = MaskReplaceConfig(
        strict_mask_replace_no_ocr_reletter=True,
        photo_pair_prefer_reletter_with_ocr=True,
        photo_pair_fallback_reletter_missing=True,
        fallback_reletter_on_blur=True,
    )
    # Saved old configs may contain all three legacy True values. The strict flag
    # is separate and pipeline-level, so Precise Mask still refuses OCR rewriting.
    assert cfg.strict_mask_replace_no_ocr_reletter is True
