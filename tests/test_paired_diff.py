import cv2
import numpy as np

from manga_hd_transfer.config import MaskReplaceConfig
from manga_hd_transfer.mask_transfer import transfer_bubble_patches
from manga_hd_transfer.models import RegistrationResult
from manga_hd_transfer.paired_diff import extract_paired_diff_bubbles


def _page():
    img = np.full((900, 700, 3), 225, np.uint8)
    # panel-like structure to keep the background non-trivial
    cv2.rectangle(img, (35, 35), (665, 865), (40, 40, 40), 3)
    cv2.line(img, (350, 35), (350, 865), (70, 70, 70), 2)
    for center in [(190, 280), (500, 300)]:
        cv2.ellipse(img, center, (105, 135), 0, 0, 360, (10, 10, 10), 4)
        cv2.ellipse(img, center, (100, 130), 0, 0, 360, (255, 255, 255), -1)
    # target/Japanese-like glyphs
    for x in (165, 190, 215):
        cv2.rectangle(img, (x, 240), (x + 6, 320), (0, 0, 0), -1)
    for x in (475, 500, 525):
        cv2.rectangle(img, (x, 250), (x + 6, 325), (0, 0, 0), -1)
    return img


def test_paired_diff_selects_only_changed_bubble_and_pixel_exact_copy():
    target = _page()
    source = target.copy()
    # Translate only the left balloon; leave the right one untouched.
    cv2.ellipse(source, (190, 280), (100, 130), 0, 0, 360, (255, 255, 255), -1)
    cv2.rectangle(source, (130, 252), (250, 264), (0, 0, 0), -1)
    cv2.rectangle(source, (145, 286), (235, 298), (0, 0, 0), -1)

    reg = RegistrationResult(
        np.eye(3, dtype=np.float64), "identity", 0.999, 1.0, 0.0, 0.8, 100,
        (700, 900), (700, 900), {},
    )
    cfg = MaskReplaceConfig()
    pd = extract_paired_diff_bubbles(source, target, reg, cfg)
    assert len(pd.source_bubbles) == 1
    assert len(pd.target_bubbles) == 1

    out = transfer_bubble_patches(source, target, pd.source_bubbles, pd.target_bubbles, reg, cfg)
    assert out.applied_count == 1
    assert out.records[0].reason == "applied_exact_identity"
    use = out.composite_mask > 0
    assert np.array_equal(out.image[use], source[use])
    assert np.array_equal(out.image[~use], target[~use])
