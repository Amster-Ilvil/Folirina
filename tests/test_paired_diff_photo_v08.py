import cv2
import numpy as np

from manga_hd_transfer.config import MaskReplaceConfig
from manga_hd_transfer.mask_transfer import transfer_paired_diff_regions
from manga_hd_transfer.models import RegistrationResult
from manga_hd_transfer.paired_diff import extract_paired_diff_bubbles


def test_structural_paired_diff_survives_smooth_photo_illumination():
    target = np.full((520, 420, 3), 245, np.uint8)
    cv2.rectangle(target, (20, 20), (400, 500), (35, 35, 35), 3)
    cv2.ellipse(target, (210, 250), (115, 145), 0, 0, 360, (10, 10, 10), 4)
    cv2.ellipse(target, (210, 250), (110, 140), 0, 0, 360, (255, 255, 255), -1)
    for x in (185, 210, 235):
        cv2.rectangle(target, (x, 205), (x + 7, 295), (0, 0, 0), -1)

    source = target.copy()
    cv2.ellipse(source, (210, 250), (110, 140), 0, 0, 360, (255, 255, 255), -1)
    cv2.rectangle(source, (150, 220), (270, 235), (0, 0, 0), -1)
    cv2.rectangle(source, (165, 265), (255, 280), (0, 0, 0), -1)
    # Camera-like low-frequency exposure gradient. Raw RGB differences cover most
    # of the page; structural ink mismatch should still isolate the balloon.
    yy, xx = np.mgrid[0:520, 0:420]
    gain = (0.76 + 0.22 * (xx / 419.0) + 0.05 * (yy / 519.0))[..., None]
    source = np.clip(source.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    reg = RegistrationResult(
        np.eye(3, dtype=np.float64), "identity-photo", 0.995, 1.0, 0.0, 0.8, 100,
        (420, 520), (420, 520), {},
    )
    cfg = MaskReplaceConfig(paired_diff_free_text_enabled=False, paired_diff_max_region_ratio=0.30)
    pd = extract_paired_diff_bubbles(source, target, reg, cfg)
    bubbles = [r for r in pd.records if r.region_kind == "bubble"]
    assert len(bubbles) == 1
    assert bubbles[0].changed_pixels >= cfg.paired_diff_min_changed_pixels

    out = transfer_paired_diff_regions(pd.aligned_source, target, pd.source_bubbles, pd.target_bubbles, cfg)
    assert out.applied_count == 1
    assert out.records[0].reason == "applied_paired_target_driven"
    assert out.records[0].target_coverage == 1.0
    assert out.records[0].spill_ratio == 0.0


def test_v08_structural_compat_module_remains_operational():
    """The absorbed v0.8 detector remains callable independently of photo_pair."""
    from manga_hd_transfer.paired_diff_v08 import extract_paired_diff_bubbles as extract_v08

    target = np.full((520, 420, 3), 245, np.uint8)
    cv2.rectangle(target, (20, 20), (400, 500), (35, 35, 35), 3)
    cv2.ellipse(target, (210, 250), (115, 145), 0, 0, 360, (10, 10, 10), 4)
    cv2.ellipse(target, (210, 250), (110, 140), 0, 0, 360, (255, 255, 255), -1)
    for x in (185, 210, 235):
        cv2.rectangle(target, (x, 205), (x + 7, 295), (0, 0, 0), -1)

    source = target.copy()
    cv2.ellipse(source, (210, 250), (110, 140), 0, 0, 360, (255, 255, 255), -1)
    cv2.rectangle(source, (150, 220), (270, 235), (0, 0, 0), -1)
    cv2.rectangle(source, (165, 265), (255, 280), (0, 0, 0), -1)
    yy, xx = np.mgrid[0:520, 0:420]
    gain = (0.76 + 0.22 * (xx / 419.0) + 0.05 * (yy / 519.0))[..., None]
    source = np.clip(source.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    reg = RegistrationResult(
        np.eye(3, dtype=np.float64), "identity-photo", 0.995, 1.0, 0.0, 0.8, 100,
        (420, 520), (420, 520), {},
    )
    cfg = MaskReplaceConfig(paired_diff_free_text_enabled=False, paired_diff_max_region_ratio=0.30)
    pd = extract_v08(source, target, reg, cfg)
    assert pd.aligned_source is not None
    assert any(r.region_kind == "bubble" for r in pd.records)
    out = transfer_paired_diff_regions(pd.aligned_source, target, pd.source_bubbles, pd.target_bubbles, cfg)
    assert out.applied_count >= 1
