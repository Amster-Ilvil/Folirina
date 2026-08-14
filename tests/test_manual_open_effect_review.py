from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.io_utils import save_json, write_image
from manga_hd_transfer.manual_effect import (
    align_source_to_target,
    estimate_open_text_masks,
    map_target_bbox_to_source,
    estimate_source_background,
    composite_source_text_delta,
)
from manga_hd_transfer.review_apply import apply_review_page, _manual_effect_overlay_base_path


def _purple_page(h=180, w=240):
    page = np.empty((h, w, 3), np.uint8)
    page[:] = (145, 72, 115)  # BGR purple-ish background
    # mild artwork line kept identical on both editions
    cv2.line(page, (12, 150), (220, 115), (35, 35, 35), 3, cv2.LINE_AA)
    return page


def _draw_source_chinese_like(img):
    # Dense white/black stroke group at x=42..84.
    cv2.rectangle(img, (42, 50), (49, 100), (245, 245, 245), -1)
    cv2.rectangle(img, (42, 70), (75, 77), (245, 245, 245), -1)
    cv2.rectangle(img, (68, 50), (75, 104), (245, 245, 245), -1)
    cv2.line(img, (45, 55), (72, 96), (20, 20, 20), 2, cv2.LINE_AA)


def _draw_target_japanese_like(img):
    cv2.rectangle(img, (118, 54), (125, 105), (245, 245, 245), -1)
    cv2.rectangle(img, (118, 76), (151, 83), (245, 245, 245), -1)
    cv2.rectangle(img, (144, 52), (151, 102), (245, 245, 245), -1)
    cv2.line(img, (122, 56), (148, 98), (20, 20, 20), 2, cv2.LINE_AA)


def _identity_project(source_path="", target_path=""):
    return {
        "pair": {"source_path": source_path, "target_path": target_path, "source_index": 0, "target_index": 0, "confidence": 1.0, "score": 1.0, "reasons": []},
        "registration": {"matrix": np.eye(3).tolist(), "method": "identity", "confidence": 1.0},
        "meta": {"transfer_mode": "auto", "passthrough": True},
        "target_bubbles": [], "source_units": [], "target_units": [], "target_blocks": [], "matches": [],
    }


def test_identity_manual_alignment_preserves_exact_source_pixels():
    src = _purple_page(80, 100)
    src[20:30, 30:40] = (1, 2, 3)
    aligned, locked = align_source_to_target(src, src.shape[:2], _identity_project())
    assert locked is True
    assert np.array_equal(src, aligned)


def test_open_effect_masks_separate_source_and_target_strokes():
    src = _purple_page(); tgt = _purple_page()
    _draw_source_chinese_like(src); _draw_target_japanese_like(tgt)
    sm, cm, diag = estimate_open_text_masks(src, tgt, [30, 35, 165, 118], diff_threshold=20, edge_threshold=35, expand_px=2)
    assert diag["source_pixels"] > 120
    assert diag["target_clear_pixels"] > 120
    assert np.count_nonzero(sm[45:110, 38:82]) > 80
    assert np.count_nonzero(sm[45:110, 112:158]) < np.count_nonzero(sm[45:110, 38:82]) * 0.45
    assert np.count_nonzero(cm[45:110, 112:158]) > 80




def test_delta_composite_reveals_text_without_copying_source_background():
    src = _purple_page(); tgt = _purple_page()
    _draw_source_chinese_like(src); _draw_target_japanese_like(tgt)
    sm, cm, _diag = estimate_open_text_masks(src, tgt, [30, 35, 165, 118], diff_threshold=20, edge_threshold=35, expand_px=2)
    cleaned = tgt.copy()
    bg = estimate_source_background(src, sm)
    out, diag = composite_source_text_delta(cleaned, src, sm, source_background=bg)
    # Background inside the broad region but away from the source glyphs should
    # remain the original purple target, not the old source patch colour block.
    assert np.array_equal(out[60, 95], tgt[60, 95])
    # Source text contribution is still transferred.
    assert np.mean(np.abs(out[45:110, 38:82].astype(np.int16) - tgt[45:110, 38:82].astype(np.int16))) > 5.0
    assert diag["delta_pixels"] > 0

def test_target_bbox_maps_back_through_registration():
    project = _identity_project()
    project["registration"]["matrix"] = [[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, 0.0, 1.0]]
    box = map_target_bbox_to_source(project, [30, 40, 70, 80])
    assert box == [10, 10, 30, 30]


def test_passthrough_page_can_be_completed_by_manual_open_effect(tmp_path: Path):
    src = _purple_page(); tgt = _purple_page()
    _draw_source_chinese_like(src); _draw_target_japanese_like(tgt)
    source_path = tmp_path / "source.png"; target_path = tmp_path / "target.png"
    write_image(source_path, src); write_image(target_path, tgt)
    write_image(tmp_path / "source_original.png", src)
    write_image(tmp_path / "target_original.png", tgt)
    write_image(tmp_path / "final.png", tgt)
    project = _identity_project(str(source_path), str(target_path))
    save_json(tmp_path / "project.json", project)
    save_json(tmp_path / "review_overrides.json", {
        "status": "reviewed_with_manual_effect",
        "manual_effect_regions": [{
            "id": "manual-effect-test",
            "mode": "effect_text",
            "target_bbox": [30, 35, 165, 118],
            "diff_threshold": 20,
            "edge_threshold": 35,
            "expand_px": 2,
            "feather_px": 0,
            "auto_clear_target": True,
        }],
    })
    cfg = PipelineConfig(); cfg.inpainting.backend = "opencv"
    out = apply_review_page(tmp_path, cfg)
    result = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert result is not None
    # Chinese-like source strokes have been introduced.
    assert np.mean(np.abs(result[45:110, 38:82].astype(np.int16) - tgt[45:110, 38:82].astype(np.int16))) > 5.0
    # Japanese-like target group is materially reduced versus the original target.
    before = np.mean(np.abs(tgt[45:110, 112:158].astype(np.int16) - _purple_page()[45:110, 112:158].astype(np.int16)))
    after = np.mean(np.abs(result[45:110, 112:158].astype(np.int16) - _purple_page()[45:110, 112:158].astype(np.int16)))
    assert after < before * 0.85
    assert (tmp_path / "manual_effect_transfer_layer.png").exists()
    assert (tmp_path / "manual_effect_clear_mask.png").exists()


def test_reveal_seed_and_window_gate_both_layers():
    from manga_hd_transfer.manual_effect import build_reveal_seed_mask, apply_reveal_window
    sm = np.zeros((80, 100), np.uint8); cm = np.zeros_like(sm)
    sm[20:26, 20:30] = 255; cm[20:26, 60:70] = 255
    seed = build_reveal_seed_mask(sm, cm, padding_px=3)
    assert np.count_nonzero(seed[17:29, 17:33]) > np.count_nonzero(sm[17:29, 17:33])
    assert np.count_nonzero(seed[17:29, 57:73]) > np.count_nonzero(cm[17:29, 57:73])
    reveal = np.zeros_like(sm); reveal[15:35, 10:40] = 255
    gsm, gcm = apply_reveal_window(sm, cm, reveal)
    assert np.count_nonzero(gsm) == np.count_nonzero(sm)
    assert np.count_nonzero(gcm) == 0


def test_reveal_text_manual_mask_persists_and_controls_transfer(tmp_path: Path):
    src = _purple_page(); tgt = _purple_page()
    _draw_source_chinese_like(src); _draw_target_japanese_like(tgt)
    source_path = tmp_path / "source.png"; target_path = tmp_path / "target.png"
    write_image(source_path, src); write_image(target_path, tgt)
    write_image(tmp_path / "source_original.png", src)
    write_image(tmp_path / "target_original.png", tgt)
    write_image(tmp_path / "final.png", tgt)
    project = _identity_project(str(source_path), str(target_path))
    save_json(tmp_path / "project.json", project)
    # Broad reviewer window covers both old Chinese and Japanese text groups.
    reveal = np.zeros(tgt.shape[:2], np.uint8)
    reveal[38:116, 32:160] = 255
    write_image(tmp_path / "reveal-test.png", reveal)
    save_json(tmp_path / "review_overrides.json", {
        "status": "reviewed_with_manual_effect",
        "manual_effect_regions": [{
            "id": "manual-reveal-test",
            "mode": "reveal_text",
            "target_bbox": [30, 35, 165, 118],
            "diff_threshold": 20,
            "edge_threshold": 35,
            "expand_px": 2,
            "feather_px": 0,
            "auto_clear_target": True,
            "reveal_mask_file": "reveal-test.png",
        }],
    })
    cfg = PipelineConfig(); cfg.inpainting.backend = "opencv"
    out = apply_review_page(tmp_path, cfg)
    result = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert result is not None
    assert np.mean(np.abs(result[45:110, 38:82].astype(np.int16) - tgt[45:110, 38:82].astype(np.int16))) > 5.0
    before = np.mean(np.abs(tgt[45:110, 112:158].astype(np.int16) - _purple_page()[45:110, 112:158].astype(np.int16)))
    after = np.mean(np.abs(result[45:110, 112:158].astype(np.int16) - _purple_page()[45:110, 112:158].astype(np.int16)))
    assert after < before * 0.85
    applied = __import__('json').loads((tmp_path / "review_applied.json").read_text())
    rows = applied.get("manual_effect_applied", [])
    assert rows and rows[0]["mode"] == "reveal_text"


def test_delta_composite_dark_text_never_creates_light_halo():
    h, w = 96, 120
    base = np.empty((h, w, 3), np.uint8); base[:] = (150, 80, 120)
    source = np.full((h, w, 3), 248, np.uint8)
    cv2.putText(source, "CN", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (12, 12, 12), 3, cv2.LINE_AA)
    mask = np.zeros((h, w), np.uint8)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    mask[gray < 235] = 255
    # Deliberately expand into white source background to reproduce the old halo case.
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)
    bg = estimate_source_background(source, mask)
    out, diag = composite_source_text_delta(base, source, mask, source_background=bg)
    sel = mask > 0
    assert np.max(out[sel].astype(np.int16) - base[sel].astype(np.int16)) <= 0
    assert diag["dark_components"] >= 1


def test_reveal_final_uses_full_text_background_estimate_for_partial_gate(tmp_path: Path):
    """Preview/final must not change text colour when the brush reveals only part of a glyph."""
    from manga_hd_transfer.manual_effect import build_manual_effect_masks, estimate_source_background, composite_source_text_delta, apply_reveal_window

    src = _purple_page(); tgt = _purple_page()
    _draw_source_chinese_like(src); _draw_target_japanese_like(tgt)
    source_path = tmp_path / "source.png"; target_path = tmp_path / "target.png"
    write_image(source_path, src); write_image(target_path, tgt)
    write_image(tmp_path / "source_original.png", src)
    write_image(tmp_path / "target_original.png", tgt)
    write_image(tmp_path / "final.png", tgt)
    project = _identity_project(str(source_path), str(target_path))
    save_json(tmp_path / "project.json", project)
    row = {
        "id": "partial-reveal", "mode": "reveal_text",
        "target_bbox": [30, 35, 165, 118], "diff_threshold": 20,
        "edge_threshold": 35, "expand_px": 2, "feather_px": 0,
        "auto_clear_target": False,
    }
    masks = build_manual_effect_masks(src, tgt, project, row)
    reveal = np.zeros(tgt.shape[:2], np.uint8)
    reveal[48:84, 38:62] = 255
    write_image(tmp_path / "partial-reveal.png", reveal)
    row["reveal_mask_file"] = "partial-reveal.png"
    save_json(tmp_path / "review_overrides.json", {
        "status": "reviewed_with_manual_effect",
        "manual_effect_regions": [row],
    })
    cfg = PipelineConfig(); cfg.inpainting.backend = "opencv"
    result = cv2.imread(str(apply_review_page(tmp_path, cfg)), cv2.IMREAD_COLOR)

    gated_source, _ = apply_reveal_window(masks.source_mask, masks.target_clear_mask, reveal)
    full_bg = estimate_source_background(masks.aligned_source, masks.source_mask)
    expected, _ = composite_source_text_delta(tgt.copy(), masks.aligned_source, gated_source, source_background=full_bg)
    assert np.array_equal(result, expected)


def test_manual_omission_overlay_preserves_unrelated_automatic_replacements_even_with_transfer_layer(tmp_path: Path):
    """Regression: adding one missed SFX must not rebuild the rest of the page.

    v1.0.1 routed any page with a Direct/Mask layer through the layer-review
    compositor.  If ``final.png`` also contained successful output from another
    automatic route, those unrelated pixels disappeared back to TARGET/Japanese.
    """
    src = _purple_page(); tgt = _purple_page()
    _draw_source_chinese_like(src); _draw_target_japanese_like(tgt)
    final = tgt.copy()
    # Simulate an already-correct automatic Chinese replacement outside the new
    # manual omission rectangle.  It intentionally does not exist in the Direct
    # layer below, reproducing the destructive v1.0.1 composition path.
    final[132:158, 182:222] = (12, 210, 35)
    source_path = tmp_path / "source.png"; target_path = tmp_path / "target.png"
    write_image(source_path, src); write_image(target_path, tgt)
    write_image(tmp_path / "source_original.png", src)
    write_image(tmp_path / "target_original.png", tgt)
    write_image(tmp_path / "final.png", final)
    project = _identity_project(str(source_path), str(target_path))
    project["meta"] = {"transfer_mode": "direct_patch", "passthrough": False, "direct_patch": {"used": True, "review_regions": []}}
    save_json(tmp_path / "project.json", project)
    blank_layer = np.zeros((tgt.shape[0], tgt.shape[1], 4), np.uint8)
    ok, data = cv2.imencode(".png", blank_layer); assert ok; data.tofile(tmp_path / "direct_patch_layer.png")
    save_json(tmp_path / "review_overrides.json", {
        "status": "reviewed_with_manual_effect",
        "manual_effect_regions": [{
            "id": "manual-overlay-preserve",
            "mode": "effect_text",
            "target_bbox": [30, 35, 165, 118],
            "diff_threshold": 20,
            "edge_threshold": 35,
            "expand_px": 2,
            "feather_px": 0,
            "auto_clear_target": True,
        }],
    })
    cfg = PipelineConfig(); cfg.inpainting.backend = "opencv"
    out = apply_review_page(tmp_path, cfg)
    result = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert result is not None
    # The unrelated successful automatic replacement is bit-exact preserved.
    assert np.array_equal(result[132:158, 182:222], final[132:158, 182:222])
    # The requested manual region still changes.
    assert np.mean(np.abs(result[45:110, 38:82].astype(np.int16) - final[45:110, 38:82].astype(np.int16))) > 3.0
    applied = __import__('json').loads((tmp_path / "review_applied.json").read_text())
    assert applied["mode"] == "manual_effect_only"


def test_manual_effect_frozen_base_is_authoritative(tmp_path: Path):
    src = _purple_page(); tgt = _purple_page()
    _draw_source_chinese_like(src); _draw_target_japanese_like(tgt)
    final = tgt.copy(); final[132:158, 182:222] = (30, 30, 30)
    frozen = final.copy(); frozen[132:158, 182:222] = (20, 180, 220)
    write_image(tmp_path / "source_original.png", src); write_image(tmp_path / "target_original.png", tgt)
    write_image(tmp_path / "final.png", final); write_image(tmp_path / "manual_effect_base.png", frozen)
    project = _identity_project(str(tmp_path / "source_original.png"), str(tmp_path / "target_original.png"))
    project["meta"] = {"transfer_mode": "direct_patch", "passthrough": False, "direct_patch": {"used": True}}
    save_json(tmp_path / "project.json", project)
    save_json(tmp_path / "review_overrides.json", {"manual_effect_regions": [{
        "id": "frozen-base", "mode": "effect_text", "target_bbox": [30,35,165,118],
        "diff_threshold": 20, "edge_threshold": 35, "expand_px": 2, "auto_clear_target": True,
    }]})
    cfg=PipelineConfig(); cfg.inpainting.backend="opencv"
    result=cv2.imread(str(apply_review_page(tmp_path,cfg)),cv2.IMREAD_COLOR)
    assert result is not None
    assert np.array_equal(result[132:158,182:222], frozen[132:158,182:222])


def test_stale_manual_effect_base_cannot_override_newer_automatic_final(tmp_path: Path):
    old=np.full((40,50,3),20,np.uint8)
    new=np.full((40,50,3),220,np.uint8)
    write_image(tmp_path/"manual_effect_base.png",old)
    write_image(tmp_path/"final.png",new)
    import os, time
    now=time.time()
    os.utime(tmp_path/"manual_effect_base.png",(now-10,now-10))
    os.utime(tmp_path/"final.png",(now,now))
    assert _manual_effect_overlay_base_path(tmp_path) == tmp_path/"final.png"


def test_reveal_preview_commit_patch_is_applied_bit_exact(tmp_path: Path):
    """The pixels accepted in Reveal preview must be the pixels in final_reviewed."""
    src = _purple_page(); tgt = _purple_page()
    source_path = tmp_path / "source.png"; target_path = tmp_path / "target.png"
    write_image(source_path, src); write_image(target_path, tgt)
    write_image(tmp_path / "source_original.png", src)
    write_image(tmp_path / "target_original.png", tgt)
    base = tgt.copy(); base[140:155, 180:215] = (25, 180, 35)
    write_image(tmp_path / "final.png", base)
    project = _identity_project(str(source_path), str(target_path))
    save_json(tmp_path / "project.json", project)

    reveal = np.zeros(tgt.shape[:2], np.uint8)
    reveal[40:120, 30:170] = 255
    write_image(tmp_path / "commit-reveal.png", reveal)
    patch = np.zeros((tgt.shape[0], tgt.shape[1], 4), np.uint8)
    patch[64:76, 82:96, :3] = (4, 4, 4)
    patch[64:76, 82:96, 3] = 255
    ok, data = cv2.imencode(".png", patch); assert ok
    data.tofile(tmp_path / "commit-patch.png")
    save_json(tmp_path / "review_overrides.json", {
        "status": "reviewed_with_manual_effect",
        "manual_effect_regions": [{
            "id": "commit-exact", "mode": "reveal_text",
            "target_bbox": [30, 35, 170, 120],
            "diff_threshold": 20, "edge_threshold": 35, "expand_px": 2,
            "feather_px": 0, "auto_clear_target": True,
            "reveal_mask_file": "commit-reveal.png",
            "reveal_patch_file": "commit-patch.png",
        }],
    })
    out = apply_review_page(tmp_path, PipelineConfig())
    result = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert result is not None
    assert np.all(result[64:76, 82:96] == 4)
    # Unrelated automatic replacement is preserved exactly.
    assert np.array_equal(result[140:155, 180:215], base[140:155, 180:215])
    applied = __import__('json').loads((tmp_path / "review_applied.json").read_text())
    row = applied["manual_effect_applied"][0]
    assert row["preview_patch_applied"] is True
    assert row["preview_patch_exact"] is True
    assert applied["manual_effect_preview_patch_verified"] is True


def test_reveal_preview_patch_survives_direct_review_route(tmp_path: Path):
    """Regression: manual_clear/direct-layer presence must not swallow Reveal commit."""
    src = _purple_page(); tgt = _purple_page()
    source_path = tmp_path / "source.png"; target_path = tmp_path / "target.png"
    write_image(source_path, src); write_image(target_path, tgt)
    write_image(tmp_path / "source_original.png", src)
    write_image(tmp_path / "target_original.png", tgt)
    automatic = tgt.copy(); automatic[145:160, 182:216] = (40, 170, 30)
    write_image(tmp_path / "final.png", automatic)
    project = _identity_project(str(source_path), str(target_path))
    project["meta"] = {
        "transfer_mode": "auto", "passthrough": False,
        "direct_patch": {"used": True, "review_regions": []},
        "mask_replace": {"used": False, "records": []},
    }
    save_json(tmp_path / "project.json", project)
    blank = np.zeros((tgt.shape[0], tgt.shape[1], 4), np.uint8)
    ok, data = cv2.imencode(".png", blank); assert ok
    data.tofile(tmp_path / "direct_patch_layer.png")
    # Presence of this file forces the general review compositor instead of the
    # manual-effect-only fast path, matching the user's real workbench state.
    write_image(tmp_path / "manual_clear_mask.png", np.zeros(tgt.shape[:2], np.uint8))

    reveal = np.zeros(tgt.shape[:2], np.uint8); reveal[35:125, 30:175] = 255
    write_image(tmp_path / "route-reveal.png", reveal)
    patch = np.zeros((tgt.shape[0], tgt.shape[1], 4), np.uint8)
    patch[70:84, 90:108, :3] = (8, 8, 8); patch[70:84, 90:108, 3] = 255
    ok, data = cv2.imencode(".png", patch); assert ok
    data.tofile(tmp_path / "route-patch.png")
    save_json(tmp_path / "review_overrides.json", {
        "status": "reviewed_with_manual_effect",
        "manual_effect_regions": [{
            "id": "route-exact", "mode": "reveal_text",
            "target_bbox": [30, 35, 175, 125],
            "diff_threshold": 20, "edge_threshold": 35, "expand_px": 2,
            "feather_px": 0, "auto_clear_target": True,
            "reveal_mask_file": "route-reveal.png",
            "reveal_patch_file": "route-patch.png",
        }],
    })
    result = cv2.imread(str(apply_review_page(tmp_path, PipelineConfig())), cv2.IMREAD_COLOR)
    assert result is not None
    assert np.all(result[70:84, 90:108] == 8)
    assert np.array_equal(result[145:160, 182:216], automatic[145:160, 182:216])
    applied = __import__('json').loads((tmp_path / "review_applied.json").read_text())
    assert applied["manual_effect_preview_patch_verified"] is True
    assert applied["manual_effect_applied"][0]["preview_patch_exact"] is True


def test_large_reveal_roi_never_becomes_a_background_write_mask(tmp_path: Path):
    """A generous manual box is a search area; distant art remains bit-exact."""
    src = _purple_page(220, 300); tgt = _purple_page(220, 300)
    # Same person/artwork in both editions.
    cv2.circle(src, (235, 105), 42, (70, 155, 220), -1)
    cv2.circle(tgt, (235, 105), 42, (70, 155, 220), -1)
    cv2.line(src, (210, 65), (270, 145), (25, 25, 25), 4, cv2.LINE_AA)
    cv2.line(tgt, (210, 65), (270, 145), (25, 25, 25), 4, cv2.LINE_AA)
    _draw_source_chinese_like(src); _draw_target_japanese_like(tgt)
    source_path = tmp_path / "source.png"; target_path = tmp_path / "target.png"
    write_image(source_path, src); write_image(target_path, tgt)
    write_image(tmp_path / "source_original.png", src); write_image(tmp_path / "target_original.png", tgt); write_image(tmp_path / "final.png", tgt)
    save_json(tmp_path / "project.json", _identity_project(str(source_path), str(target_path)))
    save_json(tmp_path / "review_overrides.json", {"manual_effect_regions": [{
        "id": "large-roi", "mode": "effect_text", "target_bbox": [18, 22, 290, 190],
        "diff_threshold": 20, "edge_threshold": 35, "expand_px": 2, "auto_clear_target": True,
    }]})
    result = cv2.imread(str(apply_review_page(tmp_path, PipelineConfig())), cv2.IMREAD_COLOR)
    assert result is not None
    # The person/skin/artwork area is outside the text corridor and must remain exact.
    assert np.array_equal(result[65:150, 205:285], tgt[65:150, 205:285])
    # Chinese transfer still occurs.
    assert np.mean(np.abs(result[45:110, 38:82].astype(np.int16) - tgt[45:110, 38:82].astype(np.int16))) > 4.0


def test_reveal_never_clears_target_when_source_chinese_is_missing():
    src = _purple_page(); tgt = _purple_page(); _draw_target_japanese_like(tgt)
    sm, cm, diag = estimate_open_text_masks(src, tgt, [30, 35, 165, 118], diff_threshold=20, edge_threshold=35, expand_px=2)
    assert cv2.countNonZero(sm) == 0
    assert cv2.countNonZero(cm) == 0
    assert diag["manual_text_corridor"]["source_text_required"] is True
    assert diag["manual_text_corridor"]["clear_suppressed_without_source"] is True


def test_legacy_full_patch_is_text_only_and_preserves_target_background(tmp_path: Path):
    """The old full_patch name is retained for compatibility but may not copy SOURCE RGB."""
    h, w = 160, 220
    source = np.full((h, w, 3), 250, np.uint8)
    target = np.full((h, w, 3), (215, 235, 250), np.uint8)
    cv2.rectangle(source, (40, 30), (180, 135), (255,255,255), -1)
    cv2.rectangle(target, (40, 30), (180, 135), (255,255,255), -1)
    cv2.putText(source, "CN", (70, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(target, "JP", (90, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 3, cv2.LINE_AA)
    sp=tmp_path/"source.png"; tp=tmp_path/"target.png"
    write_image(sp,source); write_image(tp,target); write_image(tmp_path/"source_original.png",source); write_image(tmp_path/"target_original.png",target); write_image(tmp_path/"final.png",target)
    save_json(tmp_path/"project.json",_identity_project(str(sp),str(tp)))
    save_json(tmp_path/"review_overrides.json",{"manual_effect_regions":[{
        "id":"legacy-full","mode":"full_patch","target_bbox":[40,30,181,136],"auto_clear_target":True,
    }]})
    result=cv2.imread(str(apply_review_page(tmp_path,PipelineConfig())),cv2.IMREAD_COLOR)
    assert result is not None
    # Outside the confirmed white container, target colour is bit-exact.
    assert np.array_equal(result[0:25],target[0:25])
    # Old JP pixels change and SOURCE paper cannot tint the surrounding colour.
    assert np.mean(np.abs(result[70:105,70:150].astype(np.int16)-target[70:105,70:150].astype(np.int16)))>2.0
    assert np.array_equal(result[145:159],target[145:159])


def test_white_bubble_mode_honors_manual_source_xy_nudge():
    source=np.full((140,200,3),255,np.uint8); target=np.full_like(source,255)
    cv2.putText(source,"CN",(35,80),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(target,"JP",(95,80),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),3,cv2.LINE_AA)
    from manga_hd_transfer.manual_effect import build_manual_effect_masks
    project=_identity_project()
    base=build_manual_effect_masks(source,target,project,{"mode":"white_bubble_text","target_bbox":[20,20,180,115],"source_offset_x":0,"source_offset_y":0})
    shifted=build_manual_effect_masks(source,target,project,{"mode":"white_bubble_text","target_bbox":[20,20,180,115],"source_offset_x":18,"source_offset_y":7})
    y0,x0=np.where(base.source_mask>0); y1,x1=np.where(shifted.source_mask>0)
    assert x0.size and x1.size
    assert abs((float(np.mean(x1))-float(np.mean(x0)))-18.0) < 2.5
    assert abs((float(np.mean(y1))-float(np.mean(y0)))-7.0) < 2.5


def test_empty_reveal_patch_is_not_reported_as_success(tmp_path: Path):
    src=_purple_page(); tgt=_purple_page(); _draw_source_chinese_like(src); _draw_target_japanese_like(tgt)
    write_image(tmp_path/'source_original.png',src); write_image(tmp_path/'target_original.png',tgt); write_image(tmp_path/'final.png',tgt)
    save_json(tmp_path/'project.json',_identity_project(str(tmp_path/'source_original.png'),str(tmp_path/'target_original.png')))
    reveal=np.zeros(tgt.shape[:2],np.uint8); reveal[35:120,30:170]=255; write_image(tmp_path/'empty-r.png',reveal)
    patch=np.zeros((*tgt.shape[:2],4),np.uint8); ok,data=cv2.imencode('.png',patch); assert ok; data.tofile(tmp_path/'empty-p.png')
    save_json(tmp_path/'review_overrides.json',{'manual_effect_regions':[{
        'id':'empty-reveal','mode':'reveal_text','target_bbox':[30,35,170,120],
        'reveal_mask_file':'empty-r.png','reveal_patch_file':'empty-p.png','reveal_patch_pixels':0,
        'diff_threshold':20,'edge_threshold':35,'expand_px':2,'auto_clear_target':True,
    }]})
    out=Path(apply_review_page(tmp_path,PipelineConfig())); result=cv2.imread(str(out))
    audit=__import__('json').loads((tmp_path/'review_applied.json').read_text())
    row=audit['manual_effect_applied'][0]
    assert row['success'] is False
    assert row['preview_patch_pixels']==0
    assert np.array_equal(result,tgt)


def test_white_bubble_manual_mode_replaces_existing_auto_text_instead_of_ghosting(tmp_path: Path):
    h,w=180,220
    source=np.full((h,w,3),255,np.uint8); target=np.full_like(source,255)
    cv2.rectangle(source,(35,25),(185,150),(0,0,0),2); cv2.rectangle(target,(35,25),(185,150),(0,0,0),2)
    cv2.putText(source,'CN',(70,95),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(target,'JP',(95,95),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),3,cv2.LINE_AA)
    automatic=target.copy(); cv2.putText(automatic,'OLD',(55,125),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2,cv2.LINE_AA)
    write_image(tmp_path/'source_original.png',source); write_image(tmp_path/'target_original.png',target); write_image(tmp_path/'final.png',automatic)
    save_json(tmp_path/'project.json',_identity_project(str(tmp_path/'source_original.png'),str(tmp_path/'target_original.png')))
    save_json(tmp_path/'review_overrides.json',{'manual_effect_regions':[{
        'id':'white-nudge','mode':'white_bubble_text','target_bbox':[38,28,183,148],
        'source_offset_x':4,'source_offset_y':-3,'auto_clear_target':True,
    }]})
    result=cv2.imread(str(apply_review_page(tmp_path,PipelineConfig())))
    # Existing OLD text is removed instead of surviving under the new source layer.
    old_roi=result[105:135,50:125]
    auto_old=automatic[105:135,50:125]
    assert np.mean(np.abs(old_roi.astype(np.int16)-auto_old.astype(np.int16)))>4.0
    # Container exterior remains exact.
    assert np.array_equal(result[:20],automatic[:20])



def test_core_review_commit_mirrors_reviewed_to_final_and_freezes_baseline(tmp_path: Path):
    src = _purple_page(); tgt = _purple_page()
    _draw_source_chinese_like(src); _draw_target_japanese_like(tgt)
    write_image(tmp_path / "source_original.png", src)
    write_image(tmp_path / "target_original.png", tgt)
    write_image(tmp_path / "final.png", tgt)
    save_json(tmp_path / "project.json", _identity_project(str(tmp_path / "source_original.png"), str(tmp_path / "target_original.png")))
    save_json(tmp_path / "review_overrides.json", {"manual_effect_regions": [{
        "id": "sync-core", "mode": "effect_text", "target_bbox": [30, 35, 165, 118],
        "diff_threshold": 20, "edge_threshold": 35, "expand_px": 2, "auto_clear_target": True,
    }]})
    reviewed = Path(apply_review_page(tmp_path, PipelineConfig()))
    assert reviewed.name == "final_reviewed.png"
    final_img = cv2.imread(str(tmp_path / "final.png"), cv2.IMREAD_COLOR)
    reviewed_img = cv2.imread(str(reviewed), cv2.IMREAD_COLOR)
    auto_img = cv2.imread(str(tmp_path / "final_auto.png"), cv2.IMREAD_COLOR)
    assert final_img is not None and reviewed_img is not None and auto_img is not None
    assert np.array_equal(final_img, reviewed_img)
    assert np.array_equal(auto_img, tgt)
    sync = __import__('json').loads((tmp_path / "review_sync.json").read_text())
    assert sync["synced"] is True
    assert sync["schema"].endswith("review_sync.v3")


def test_repeated_manual_effects_rebuild_from_immutable_final_auto(tmp_path: Path):
    h, w = 300, 260
    src = _purple_page(h, w); tgt = _purple_page(h, w)
    # First text group.
    _draw_source_chinese_like(src); _draw_target_japanese_like(tgt)
    # Second independent text group lower on the page.
    cv2.rectangle(src, (45, 190), (52, 244), (245,245,245), -1)
    cv2.rectangle(src, (45, 212), (80, 219), (245,245,245), -1)
    cv2.rectangle(src, (73, 188), (80, 246), (245,245,245), -1)
    cv2.line(src, (48, 194), (77, 238), (20,20,20), 2, cv2.LINE_AA)
    cv2.rectangle(tgt, (122, 188), (129, 244), (245,245,245), -1)
    cv2.rectangle(tgt, (122, 213), (157, 220), (245,245,245), -1)
    cv2.rectangle(tgt, (150, 188), (157, 244), (245,245,245), -1)
    cv2.line(tgt, (125, 192), (154, 238), (20,20,20), 2, cv2.LINE_AA)

    write_image(tmp_path / "source_original.png", src)
    write_image(tmp_path / "target_original.png", tgt)
    write_image(tmp_path / "final.png", tgt)
    save_json(tmp_path / "project.json", _identity_project(str(tmp_path / "source_original.png"), str(tmp_path / "target_original.png")))
    row1 = {
        "id": "repeat-one", "mode": "effect_text", "target_bbox": [30, 35, 165, 118],
        "diff_threshold": 20, "edge_threshold": 35, "expand_px": 2, "auto_clear_target": True,
    }
    save_json(tmp_path / "review_overrides.json", {"manual_effect_regions": [row1]})
    first_path = Path(apply_review_page(tmp_path, PipelineConfig()))
    first = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    frozen1 = cv2.imread(str(tmp_path / "final_auto.png"), cv2.IMREAD_COLOR)
    assert first is not None and frozen1 is not None

    row2 = {
        "id": "repeat-two", "mode": "effect_text", "target_bbox": [30, 172, 170, 255],
        "diff_threshold": 20, "edge_threshold": 35, "expand_px": 2, "auto_clear_target": True,
    }
    save_json(tmp_path / "review_overrides.json", {"manual_effect_regions": [row1, row2]})
    second_path = Path(apply_review_page(tmp_path, PipelineConfig()))
    second = cv2.imread(str(second_path), cv2.IMREAD_COLOR)
    frozen2 = cv2.imread(str(tmp_path / "final_auto.png"), cv2.IMREAD_COLOR)
    assert second is not None and frozen2 is not None

    # Adding a second manual region must not re-darken/re-composite the first one.
    assert np.array_equal(second[35:120, 30:166], first[35:120, 30:166])
    # The immutable baseline is unchanged even though final.png now mirrors review.
    assert np.array_equal(frozen2, frozen1)
    # The second region actually contributed pixels.
    assert np.mean(np.abs(second[180:255, 35:170].astype(np.int16) - first[180:255, 35:170].astype(np.int16))) > 1.0
    # Page-local final always mirrors the newest reviewed output.
    mirrored = cv2.imread(str(tmp_path / "final.png"), cv2.IMREAD_COLOR)
    assert np.array_equal(mirrored, second)


def test_manual_reveal_accepts_legacy_boolean_direct_patch_meta(tmp_path: Path):
    """Old projects may store meta.direct_patch as True/False rather than a dict."""
    src = _purple_page(); tgt = _purple_page()
    _draw_source_chinese_like(src); _draw_target_japanese_like(tgt)
    write_image(tmp_path / "source_original.png", src)
    write_image(tmp_path / "target_original.png", tgt)
    write_image(tmp_path / "final.png", tgt)
    project = _identity_project(str(tmp_path / "source_original.png"), str(tmp_path / "target_original.png"))
    project["meta"] = {"transfer_mode": "direct_patch", "direct_patch": True, "mask_replace": False}
    save_json(tmp_path / "project.json", project)

    blank = np.zeros((tgt.shape[0], tgt.shape[1], 4), np.uint8)
    ok, encoded = cv2.imencode(".png", blank); assert ok
    encoded.tofile(tmp_path / "direct_patch_layer.png")
    # Force the general Direct review compositor where the legacy bool used to
    # hit `meta.get('direct_patch', {}).get('used')` and crash.
    write_image(tmp_path / "manual_clear_mask.png", np.zeros(tgt.shape[:2], np.uint8))

    reveal = np.zeros(tgt.shape[:2], np.uint8); reveal[35:125, 30:175] = 255
    write_image(tmp_path / "legacy-bool-reveal.png", reveal)
    patch = np.zeros((tgt.shape[0], tgt.shape[1], 4), np.uint8)
    patch[70:84, 90:108, :3] = (9, 9, 9); patch[70:84, 90:108, 3] = 255
    ok, encoded = cv2.imencode(".png", patch); assert ok
    encoded.tofile(tmp_path / "legacy-bool-patch.png")
    save_json(tmp_path / "review_overrides.json", {
        "status": "reviewed_with_manual_effect",
        "manual_effect_regions": [{
            "id": "legacy-bool-route", "mode": "reveal_text",
            "target_bbox": [30, 35, 175, 125],
            "diff_threshold": 20, "edge_threshold": 35, "expand_px": 2,
            "feather_px": 0, "auto_clear_target": True,
            "reveal_mask_file": "legacy-bool-reveal.png",
            "reveal_patch_file": "legacy-bool-patch.png",
        }],
    })
    out = apply_review_page(tmp_path, PipelineConfig())
    result = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert result is not None
    assert np.all(result[70:84, 90:108] == 9)
    audit = __import__('json').loads((tmp_path / "review_applied.json").read_text())
    assert audit["manual_effect_preview_patch_verified"] is True
