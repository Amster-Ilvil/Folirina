from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.text_only_transfer import (
    cleanup_white_container_line_artifacts,
    white_container_paper_mask,
)


def test_white_container_paper_mask_stays_inside_neutral_bubble_not_colored_art():
    h, w = 90, 120
    target = np.zeros((h, w, 3), np.uint8)
    target[:] = (180, 110, 60)  # coloured art background
    region = np.zeros((h, w), np.uint8)
    region[10:80, 10:110] = 255
    # White bubble interior inside the region.
    target[20:70, 25:95] = (245, 245, 245)
    source_mask = np.zeros((h, w), np.uint8)
    source_mask[32:58, 50:58] = 255

    paper = white_container_paper_mask(target, region, source_mask)
    # Core bubble interior should be admitted.
    assert int(paper[45, 60]) > 0
    # Coloured region outside the bubble should stay excluded.
    assert int(paper[15, 15]) == 0


def test_cleanup_white_container_line_artifacts_removes_faint_line_but_keeps_supported_text():
    h, w = 80, 120
    target = np.full((h, w, 3), 246, np.uint8)
    image = target.copy()
    paper = np.zeros((h, w), np.uint8)
    paper[10:70, 12:108] = 255
    # Unsupported faint guide line near the top of the bubble.
    cv2.line(image, (28, 20), (78, 20), (210, 210, 210), 1, lineType=cv2.LINE_AA)
    # Legitimate supported Chinese punctuation / stroke must survive.
    cv2.circle(image, (62, 46), 2, (15, 15, 15), -1)
    source_mask = np.zeros((h, w), np.uint8)
    cv2.circle(source_mask, (62, 46), 3, 255, -1)

    out, removed, diag = cleanup_white_container_line_artifacts(
        image, target, paper, source_mask,
    )
    assert diag["white_line_artifacts_removed"] > 0
    assert cv2.countNonZero(removed) > 0
    # The faint line is cleaned back toward paper.
    assert float(np.mean(out[18:23, 34:72])) > 235.0
    # Supported dark punctuation remains dark.
    assert int(out[46, 62, 0]) < 60


def test_target_white_container_text_mask_keeps_edge_glyph_but_rejects_long_border():
    from manga_hd_transfer.text_only_transfer import target_white_container_text_mask

    h, w = 90, 130
    target = np.full((h, w, 3), 248, np.uint8)
    paper = np.zeros((h, w), np.uint8)
    paper[8:82, 8:122] = 255
    # High-resolution box border: long line touching the paper boundary.
    cv2.line(target, (8, 8), (121, 8), (20, 20, 20), 1)
    cv2.line(target, (8, 8), (8, 81), (20, 20, 20), 1)
    # Compact target glyph very close to the left edge (analogous to Japanese と).
    cv2.line(target, (12, 20), (12, 37), (20, 20, 20), 3)
    cv2.line(target, (12, 28), (22, 28), (20, 20, 20), 3)
    # A small punctuation mark near the right edge.
    cv2.circle(target, (116, 20), 2, (20, 20, 20), -1)

    mask = target_white_container_text_mask(target, paper)
    assert int(mask[28, 12]) > 0
    assert int(mask[20, 116]) > 0
    # Border must not be selected for deletion.
    assert int(mask[8, 60]) == 0
    assert int(mask[50, 8]) == 0


def test_white_residual_cleanup_removes_large_edge_japanese_component_without_erasing_source_text():
    from manga_hd_transfer.text_only_transfer import cleanup_target_residual_text_in_white_container

    h, w = 100, 150
    target = np.full((h, w, 3), 248, np.uint8)
    paper = np.zeros((h, w), np.uint8); paper[8:92, 8:142] = 255
    # Simulated TARGET Japanese glyph near the left edge.
    cv2.line(target, (15, 20), (15, 55), (15, 15, 15), 5)
    cv2.line(target, (15, 35), (35, 35), (15, 15, 15), 5)
    # Simulated HD box border must survive.
    cv2.line(target, (8, 8), (141, 8), (20, 20, 20), 1)
    image = target.copy()
    # Valid SOURCE Chinese glyph elsewhere.
    cv2.rectangle(image, (75, 25), (88, 60), (10, 10, 10), 3)
    source_mask = np.zeros((h, w), np.uint8); cv2.rectangle(source_mask, (75, 25), (88, 60), 255, 3)

    out, removed, diag = cleanup_target_residual_text_in_white_container(image, target, paper, source_mask)
    assert diag["white_residual_text_removed"] > 0
    assert int(out[35, 15, 0]) > 220
    assert int(out[25, 75, 0]) < 80
    assert int(out[8, 70, 0]) < 80


def test_target_container_border_mask_preserves_outline_not_edge_glyph():
    from manga_hd_transfer.text_only_transfer import target_container_border_mask
    h,w=100,140
    target=np.full((h,w,3),248,np.uint8)
    region=np.zeros((h,w),np.uint8); region[10:90,10:130]=255
    # box outline
    cv2.rectangle(target,(10,10),(129,89),(20,20,20),1)
    # compact glyph near left edge
    cv2.line(target,(16,28),(16,44),(20,20,20),3)
    cv2.line(target,(16,36),(27,36),(20,20,20),3)
    mask=target_container_border_mask(target,region,band_px=4)
    assert int(mask[10,70])>0
    assert int(mask[50,10])>0
    assert int(mask[36,16])==0


def test_source_container_border_is_not_treated_as_chinese_text():
    from manga_hd_transfer.text_only_transfer import remove_container_boundary_line_components
    h,w=100,120
    region=np.zeros((h,w),np.uint8); region[10:90,15:105]=255
    mask=np.zeros((h,w),np.uint8)
    # Source box rule touching the container boundary.
    cv2.line(mask,(15,12),(104,12),255,2)
    # Compact Chinese-like strokes safely inside.
    cv2.rectangle(mask,(55,35),(61,61),255,-1)
    out,removed=remove_container_boundary_line_components(mask,region)
    assert removed>0
    assert cv2.countNonZero(out[10:18,:])==0
    assert cv2.countNonZero(out[35:62,55:62])>0


def test_white_clear_overshoot_must_not_protect_nonpaper_background():
    from manga_hd_transfer.text_only_transfer import transfer_text_only
    h,w=100,140
    target=np.zeros((h,w,3),np.uint8)
    target[:]=(15,15,15)
    target[25:85,30:115]=(248,248,248)
    source=target.copy()
    # Chinese-like source stroke on the white paper.
    cv2.rectangle(source,(65,42),(72,70),(15,15,15),-1)
    # Japanese-like target stroke shifted away.
    cv2.rectangle(target,(80,42),(87,70),(15,15,15),-1)
    region=np.zeros((h,w),np.uint8)
    # Deliberately overshoot 8px above the true white box into black artwork.
    region[17:86,28:117]=255
    out,write,smask,diag=transfer_text_only(target,source,region,white_container=True,clear_dilate_px=2)
    # Black artwork above the actual white paper must remain byte-stable.
    assert np.array_equal(out[17:24,35:108],target[17:24,35:108])


def test_white_container_write_envelope_stays_inside_box_border_and_rules():
    from manga_hd_transfer.text_only_transfer import white_container_write_envelope

    h, w = 110, 150
    target = np.full((h, w, 3), 248, np.uint8)
    region = np.zeros((h, w), np.uint8)
    region[10:100, 12:138] = 255
    # Dark narration-box outline.
    cv2.rectangle(target, (12, 10), (137, 99), (16, 16, 16), 1)
    # Interior paper and a short Japanese-like stroke.
    paper = np.zeros((h, w), np.uint8)
    paper[11:99, 13:137] = 255
    cv2.line(target, (35, 28), (35, 60), (18, 18, 18), 3)

    env, diag = white_container_write_envelope(target, region, paper, inset_px=2, border_guard_px=2)
    assert diag["border_pixels"] > 0
    assert diag["envelope_pixels"] > 0
    # Border/rule itself must not be writable.
    assert int(env[10, 70]) == 0
    assert int(env[55, 12]) == 0
    # Interior paper remains writable.
    assert int(env[40, 70]) > 0


def test_transfer_text_only_white_container_removes_top_rule_black_line_regression():
    from manga_hd_transfer.text_only_transfer import transfer_text_only

    h, w = 120, 160
    target = np.zeros((h, w, 3), np.uint8)
    target[:] = (18, 18, 18)
    # White narration box with black outline on dark panel.
    target[18:102, 28:132] = (248, 248, 248)
    cv2.rectangle(target, (28, 18), (131, 101), (15, 15, 15), 1)
    source = target.copy()
    # Chinese source text.
    cv2.rectangle(source, (58, 38), (67, 80), (15, 15, 15), -1)
    # Japanese target text shifted plus a faint residual line near the top edge.
    cv2.rectangle(target, (86, 38), (95, 80), (15, 15, 15), -1)
    cv2.line(target, (38, 26), (120, 26), (30, 30, 30), 1)
    region = np.zeros((h, w), np.uint8)
    # Overshoot the mapped region by a few px around the box.
    region[14:104, 25:135] = 255

    out, write, smask, diag = transfer_text_only(target, source, region, white_container=True, clear_dilate_px=2)
    assert diag["white_container_write_envelope"]["removed_pixels"] > 0
    # Published box outline must remain dark / untouched.
    assert int(out[18, 80, 0]) < 60
    assert int(out[60, 28, 0]) < 60
    # Interior writing happens.
    assert cv2.countNonZero(write) > 0
    assert cv2.countNonZero(smask) > 0
    # The top residual line should not survive as a dark strip inside the box.
    assert float(np.mean(out[25:28, 48:112, 0])) > 225.0
