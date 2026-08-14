from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.direct_containers import _expand_white_gap_mask, _warp_similarity_patch, _protect_registered_geometry_boundary, _constrain_direct_write_to_target_border, build_source_direct_container_plan, _source_direct_registration_gate
from manga_hd_transfer.mask_transfer import _rigid_target_write_envelope, _expand_safe_write_mask, _repair_content_region, _evaluate_content_completeness, finalize_transfer_records, MaskTransferRecord, _subpixel_translation_refine, _pixel_enhance_text_raster, transfer_rigid_container_rasters
from manga_hd_transfer.io_utils import read_image, write_image, save_json
from manga_hd_transfer.models import PagePair, RegistrationResult, TextUnit, BubbleInstance
from manga_hd_transfer.pipeline import TransferPipeline, _load_additional_source_specs, _filter_uncovered_white_completion_pairs, _blocking_direct_invariant_issues
from manga_hd_transfer.matching import match_units
from manga_hd_transfer.inpainting import threshold_clear
from manga_hd_transfer.text_only_transfer import transfer_text_only


def _same_layout_page(chinese: bool) -> np.ndarray:
    img = np.full((900, 650, 3), 255, np.uint8)
    cv2.rectangle(img, (25, 25), (625, 875), (0, 0, 0), 4)
    cv2.line(img, (25, 420), (625, 420), (0, 0, 0), 4)
    cv2.circle(img, (120, 150), 65, (0, 0, 0), 4)
    cv2.rectangle(img, (430, 520), (590, 760), (0, 0, 0), 4)
    cv2.ellipse(img, (330, 255), (130, 90), 0, 0, 360, (0, 0, 0), 4)
    pts = (
        [(300, 220), (320, 220), (340, 220), (360, 220),
         (300, 245), (320, 245), (340, 245), (360, 245),
         (300, 270), (320, 270), (340, 270), (360, 270)]
        if chinese
        else
        [(290, 215), (310, 215), (330, 215), (350, 215), (370, 215),
         (290, 240), (310, 240), (330, 240), (350, 240), (370, 240),
         (290, 265), (310, 265), (330, 265), (350, 265), (370, 265),
         (290, 290), (330, 290), (370, 290)]
    )
    for x, y in pts:
        cv2.rectangle(img, (x, y), (x + 8, y + 12), (0, 0, 0), -1)
    return img




def _same_layout_color_target() -> np.ndarray:
    img = np.full((900, 650, 3), (220, 240, 255), np.uint8)
    cv2.rectangle(img, (25, 25), (625, 875), (40, 40, 40), 4)
    cv2.line(img, (25, 420), (625, 420), (40, 40, 40), 4)
    cv2.circle(img, (120, 150), 65, (210, 160, 80), -1)
    cv2.circle(img, (120, 150), 65, (0, 0, 0), 4)
    cv2.rectangle(img, (430, 520), (590, 760), (215, 235, 205), -1)
    cv2.rectangle(img, (430, 520), (590, 760), (0, 0, 0), 4)
    cv2.ellipse(img, (330, 255), (130, 90), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (330, 255), (130, 90), 0, 0, 360, (0, 0, 0), 4)
    pts = [
        (290, 215), (310, 215), (330, 215), (350, 215), (370, 215),
        (290, 240), (310, 240), (330, 240), (350, 240), (370, 240),
        (290, 265), (310, 265), (330, 265), (350, 265), (370, 265),
        (290, 290), (330, 290), (370, 290),
    ]
    for x, y in pts:
        cv2.rectangle(img, (x, y), (x + 8, y + 12), (0, 0, 0), -1)
    return img

def _pair(tmp_path: Path) -> tuple[PagePair, Path, Path]:
    source_path = tmp_path / "cn.png"
    target_path = tmp_path / "jp.png"
    write_image(source_path, _same_layout_page(True))
    write_image(target_path, _same_layout_page(False))
    return PagePair(str(source_path), str(target_path), 0, 0, .99, .01, []), source_path, target_path


def _base_config(mode: str) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.transfer.mode = mode
    cfg.registration.backend = "opencv"
    cfg.registration.min_matches = 6
    cfg.export.layer_bundle = False
    cfg.qa.fail_empty_mask_replace = False
    return cfg


def test_direct_integer_translation_keeps_raster_bit_exact() -> None:
    crop = np.full((17, 19, 3), 255, np.uint8)
    crop[4:13, 7:10] = (0, 0, 0)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((17, 19), np.uint8); mask[2:15, 2:17] = 255
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    bx, by, wcrop, wgray, wmask, wboundary = _warp_similarity_patch(
        crop, gray, mask, boundary, (30, 40), (39.0, 48.0), (42.0, 46.0), 1.0, 0.0
    )
    assert (bx, by) == (33, 38)
    assert np.array_equal(wcrop, crop)
    assert np.array_equal(wgray, gray)
    assert np.array_equal(wmask, mask)
    assert np.array_equal(wboundary, boundary)


def test_direct_same_canvas_fractional_registration_keeps_target_background() -> None:
    source = _same_layout_page(True)
    target = _same_layout_page(False)
    # Give TARGET a colour wash outside the lettering so copying SOURCE paper is
    # immediately detectable.
    color_target = target.copy()
    paper = np.all(color_target > 245, axis=2)
    color_target[paper] = (210, 235, 250)
    reg = RegistrationResult(
        matrix=np.array([[1.0, 0.0, 0.2], [0.0, 1.0, -0.2], [0.0, 0.0, 1.0]], dtype=float),
        method="opencv_affine", confidence=0.99, inlier_ratio=0.99, reprojection_error=0.1,
        spatial_coverage=0.9, num_matches=100, source_size=(650, 900), target_size=(650, 900), diagnostics={},
    )
    cfg = PipelineConfig().direct_patch
    plan = build_source_direct_container_plan(source, color_target, reg, cfg)
    assert plan is not None
    assert plan.diagnostics["identity_pixel_locked_regions"] >= 1
    use = plan.result.composite_mask > 0
    assert np.count_nonzero(use) > 0
    # v1.0.7 contract: Direct may alter lettering pixels, but SOURCE white/gray
    # paper is never allowed to replace coloured TARGET background.
    unchanged_bg = (~use) & paper
    assert np.array_equal(plan.result.image[unchanged_bg], color_target[unchanged_bg])
    assert not np.any(np.all(plan.result.image[use] == source[use], axis=1) & np.all(source[use] > 245, axis=1))
    assert all(bool(b.meta.get("identity_pixel_lock")) for b in plan.target_bubbles)


def test_direct_affine_geometry_snap_fills_white_sliver_without_stretching_text() -> None:
    source = _same_layout_page(True)
    jp = _same_layout_page(False)
    H = np.array([[0.94, 0.0, 18.0], [0.0, 1.03, -8.0], [0.0, 0.0, 1.0]], dtype=float)
    target = cv2.warpPerspective(jp, H, (650, 900), flags=cv2.INTER_NEAREST, borderValue=(255, 255, 255))
    reg = RegistrationResult(
        matrix=H, method="affine", confidence=0.99, inlier_ratio=0.99, reprojection_error=0.1,
        spatial_coverage=0.9, num_matches=100, source_size=(650, 900), target_size=(650, 900), diagnostics={},
    )
    cfg = PipelineConfig().direct_patch
    cfg.source_direct_max_axis_scale_delta = 0.20
    plan = build_source_direct_container_plan(source, target, reg, cfg)
    assert plan is not None
    assert plan.diagnostics["geometry_snapped_regions"] >= 1
    assert plan.diagnostics["geometry_snap_gap_pixels"] > 0
    assert any(bool(b.meta.get("geometry_snap", {}).get("accepted")) for b in plan.target_bubbles)
    # The raster transform remains a single similarity scale even though the
    # envelope followed an anisotropic affine page mapping.
    assert plan.diagnostics["final_raster_transform"] == "local_similarity_only"


def test_direct_and_mask_have_independent_config_namespaces() -> None:
    cfg = PipelineConfig()
    original_mask_value = cfg.mask_replace.source_direct_min_registration_confidence
    cfg.direct_patch.source_direct_min_registration_confidence = 0.987
    assert cfg.mask_replace.source_direct_min_registration_confidence == original_mask_value
    assert cfg.direct_patch.source_direct_min_registration_confidence == 0.987




def test_direct_patch_accepts_monochrome_source_to_colored_target(tmp_path: Path) -> None:
    source_path = tmp_path / "cn_bw.png"
    target_path = tmp_path / "jp_color.png"
    write_image(source_path, _same_layout_page(True))
    write_image(target_path, _same_layout_color_target())
    pair = PagePair(str(source_path), str(target_path), 0, 0, .99, .01, [])
    project = TransferPipeline(_base_config("direct_patch")).process_page(pair, tmp_path / "color_out")
    assert project.meta.get("transfer_mode") == "direct_patch"
    assert project.meta.get("direct_patch", {}).get("used") is True
    diag = project.meta.get("transfer_audit", {}).get("candidate_detection", {}).get("source_direct_container_diagnostics", {})
    assert diag.get("accepted", 0) >= 1
    assert diag.get("source_saturation_p90", 999.0) < 35.0



def test_completion_filter_keeps_only_uncovered_plain_white_bubbles() -> None:
    from manga_hd_transfer.models import BubbleInstance
    def b(i, box, **meta):
        x0,y0,x1,y1=box
        return BubbleInstance(id=i, polygon=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)], confidence=1.0, kind="speech", meta=meta)
    src=[b("s0",(0,0,50,50)), b("s1",(60,0,110,50)), b("s2",(120,0,170,50))]
    dst=[
        b("t0",(0,0,50,50), backend="unseeded_white", white_ratio=.95, dark_ratio=.05),
        b("t1",(60,0,110,50), backend="unseeded_white", white_ratio=.95, dark_ratio=.05),
        b("t2",(120,0,170,50), backend="target_colored_container", target_colored_recovery=True, white_ratio=.95, dark_ratio=.05),
    ]
    ks,kt=_filter_uncovered_white_completion_pairs(src,dst,[(0,0,52,52)])
    assert [x.id for x in kt] == ["t1"]
    assert [x.id for x in ks] == ["s1"]


def test_direct_patch_uses_whole_raster_contract_and_not_mask_route(tmp_path: Path) -> None:
    pair, _, _ = _pair(tmp_path)
    project = TransferPipeline(_base_config("direct_patch")).process_page(pair, tmp_path / "out")

    direct = project.meta.get("direct_patch", {})
    audit = project.meta.get("transfer_audit", {})
    detection = audit.get("candidate_detection", {})

    assert project.meta.get("transfer_mode") == "direct_patch"
    assert direct.get("used") is True
    assert direct.get("contract") == "text_only_target_background"
    assert detection.get("direct_patch_used") is True
    assert detection.get("mask_route_used") is False
    assert project.meta.get("mask_replace", {}).get("used") is False
    assert project.meta.get("mask_replace", {}).get("records") == []
    assert project.meta.get("direct_patch", {}).get("records")
    assert str(project.meta.get("cache", {}).get("ocr_source", "")).startswith("skipped_source_direct")
    transfer_meta = project.meta.get("transfer_audit", {}).get("transfer", {})
    assert transfer_meta.get("triage_reject", 0) == 0
    assert transfer_meta.get("triage_safe", 0) >= 1
    assert (tmp_path / "out" / "direct_patch_layer.png").exists()
    assert (tmp_path / "out" / "direct_patch_regions.png").exists()
    assert (tmp_path / "out" / "direct_patch.json").exists()
    assert not (tmp_path / "out" / "mask_transfer_layer.png").exists()
    assert not (tmp_path / "out" / "mask_transfer_mask.png").exists()
    assert not (tmp_path / "out" / "mask_transfer.json").exists()
    assert not any(item.code.startswith("mask_replace_") for item in project.qa)


def test_direct_reject_is_passthrough_and_never_falls_back_to_mask_or_ocr(tmp_path: Path) -> None:
    pair, _, target_path = _pair(tmp_path)
    cfg = _base_config("direct_patch")
    # Basic route validity remains: disabling Direct itself must passthrough and
    # must never silently become Mask/OCR.
    cfg.direct_patch.source_direct_container_enabled = False

    out = tmp_path / "reject_out"
    project = TransferPipeline(cfg).process_page(pair, out)

    assert project.meta.get("passthrough") is True
    assert project.meta.get("passthrough_reason") == "direct_patch_rejected"
    assert project.meta.get("transfer_planner", {}).get("strategy") == "direct_reject"
    assert project.meta.get("direct_patch", {}).get("used") is False
    assert not (out / "mask_transfer_layer.png").exists()
    assert not (out / "mask_transfer_mask.png").exists()
    assert not (out / "direct_patch_layer.png").exists()

    final = read_image(out / "final.png")
    target = read_image(target_path)
    assert np.array_equal(final, target)


def test_white_gap_fill_expands_only_safe_source_paper() -> None:
    cfg = PipelineConfig().direct_patch
    mask = np.zeros((11, 11), np.uint8)
    mask[3:8, 3:8] = 1
    source_gray = np.full((11, 11), 255, np.uint8)
    target_gray = np.full((11, 11), 235, np.uint8)
    # Simulate residual Japanese ink just outside the mapped source interior.
    target_gray[2, 5] = 40
    # Simulate the true bubble outline nearby; the grow helper must not cross it.
    target_edges = np.full((11, 11), 3.0, np.float32)
    target_edges[1, :] = 0.2
    expanded, diag = _expand_white_gap_mask(mask, source_gray, target_gray, target_edges, cfg)
    assert diag["added_pixels"] > 0
    assert expanded[2, 5] == 1
    # The strong-outline row remains protected.
    assert int(np.count_nonzero(expanded[1, :])) == 0


def test_mask_gap_fill_stays_inside_safe_envelope() -> None:
    cfg = PipelineConfig().mask_replace
    base = np.zeros((11, 11), np.uint8)
    base[4:7, 4:7] = 255
    envelope = np.zeros((11, 11), np.uint8)
    envelope[2:9, 2:9] = 255
    source = np.full((11, 11, 3), 255, np.uint8)
    target = np.full((11, 11, 3), 235, np.uint8)
    # residual target dark pixel in the safe ring should be swallowed
    target[3, 5] = (20, 20, 20)
    # but the outer strong edge row must remain blocked
    target[1, :] = (0, 0, 0)
    expanded, diag = _expand_safe_write_mask(base, envelope, source, target, cfg)
    assert diag["added_pixels"] > 0
    assert int(np.count_nonzero(expanded)) > int(np.count_nonzero(base))
    assert int(np.count_nonzero(expanded[1, :])) == 0
    assert np.all(expanded[envelope == 0] == 0)


def test_content_auto_repair_recovers_missing_source_and_target_residual() -> None:
    cfg = PipelineConfig().mask_replace
    h = w = 31
    safe = np.zeros((h, w), np.uint8); safe[4:27, 4:27] = 255
    current = np.zeros((h, w), np.uint8); current[10:21, 9:22] = 255
    source = np.full((h, w, 3), 255, np.uint8)
    # Chinese stroke sits just outside the initial write mask but inside safe envelope.
    source[7:12, 13:18] = 0
    source_ink = np.zeros((h, w), np.uint8); source_ink[7:12, 13:18] = 255
    target = np.full((h, w, 3), 255, np.uint8)
    # Residual Japanese component on the other side.
    target[21:26, 13:18] = 0
    target_ink = np.zeros((h, w), np.uint8); target_ink[21:26, 13:18] = 255
    rendered = target.copy(); use = current > 0; rendered[use] = source[use]
    rec = MaskTransferRecord('s','t',0.95,True,'test')
    _evaluate_content_completeness(rec, source_ink, target_ink, rendered, cfg, tolerance_px=1, min_source_coverage=0.95, max_target_residual=0.05)
    assert not rec.content_complete
    repaired, repair_mask, diag = _repair_content_region(
        rec, rendered, source, target, current, safe, source_ink, target_ink, cfg,
        tolerance_px=1, min_source_coverage=0.95, max_target_residual=0.05,
    )
    assert rec.repair_attempted
    assert diag['improved']
    assert rec.content_complete
    assert rec.source_ink_coverage >= 0.95
    assert rec.target_residual_ratio <= 0.05
    assert int(np.count_nonzero(repair_mask)) >= int(np.count_nonzero(current))
    assert np.mean(repaired[7:12, 13:18]) < 100
    assert np.mean(repaired[21:26, 13:18]) > 180


def test_finalize_transfer_records_keeps_applied_regions_safe_without_publication_blocking() -> None:
    cfg = PipelineConfig().mask_replace
    cfg.publication_safety_enabled = True  # ignored legacy value
    safe = MaskTransferRecord('s1','t1',0.95,True,'ok',content_check='checked',source_ink_coverage=1.0,target_residual_ratio=0.0,content_complete=True)
    review = MaskTransferRecord('s2','t2',0.95,True,'ok',content_check='checked',source_ink_coverage=0.8,target_residual_ratio=0.2,content_complete=False)
    reject = MaskTransferRecord('s3','t3',0.95,False,'nope')
    finalize_transfer_records([safe, review, reject], cfg)
    assert safe.triage_state == 'SAFE'
    assert review.triage_state == 'SAFE'
    assert reject.triage_state == 'REJECT'


def test_subpixel_refine_improves_fractional_mask_alignment() -> None:
    cfg = PipelineConfig().mask_replace
    cfg.local_subpixel_min_iou_gain = 0.0001
    src = np.zeros((64, 64), np.uint8)
    cv2.ellipse(src, (32, 32), (15, 10), 0, 0, 360, 255, -1)
    M = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], np.float32)
    tgt = cv2.warpAffine(src, M, (64, 64), flags=cv2.INTER_LINEAR, borderValue=0)
    dx, dy, score, diag = _subpixel_translation_refine(src, tgt, 0.0, 0.0, cfg)
    assert diag["after"] >= diag["before"]
    assert score >= diag["before"]
    assert abs(dx) <= cfg.local_subpixel_radius_px + 1e-6
    assert abs(dy) <= cfg.local_subpixel_radius_px + 1e-6


def test_pixel_enhance_preserves_background_and_increases_text_sharpness() -> None:
    cfg = PipelineConfig().mask_replace
    img = np.full((80, 80, 3), 255, np.uint8)
    cv2.putText(img, "A", (22, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (55, 55, 55), 4, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    mask = np.zeros((80, 80), np.uint8); mask[12:68, 12:68] = 255
    before_bg = img[0:10, 0:10].copy()
    enhanced, diag = _pixel_enhance_text_raster(img, mask, cfg)
    assert enhanced is not None
    assert diag["after_sharpness"] > diag["before_sharpness"]
    assert np.array_equal(enhanced[0:10, 0:10], before_bg)


def test_replace_translation_bundle_is_exported(tmp_path: Path) -> None:
    pair, _, _ = _pair(tmp_path)
    project = TransferPipeline(_base_config("direct_patch")).process_page(pair, tmp_path / "rt_out")
    rt = project.meta.get("replace_translation", {})
    assert (tmp_path / "rt_out" / "replace_translation" / "source_ocr.json").exists()
    assert (tmp_path / "rt_out" / "replace_translation" / "target_ocr.json").exists()
    assert (tmp_path / "rt_out" / "replace_translation" / "unit_matches.json").exists()
    assert (tmp_path / "rt_out" / "replace_translation" / "summary.json").exists()
    assert rt.get("selected_source_kind") in {"primary", "alternate", "high_translated"}
    assert isinstance(rt.get("artifacts"), dict)


def test_load_additional_source_specs_reads_replace_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "page.png"
    write_image(source_path, _same_layout_page(True))
    alt = tmp_path / "page_alt.png"
    write_image(alt, _same_layout_page(True))
    save_json(tmp_path / "page.replace_sources.json", {"sources": [{"path": alt.name, "kind": "high_translated"}]})
    specs = _load_additional_source_specs(source_path, PipelineConfig().replace_translation)
    assert len(specs) == 1
    assert specs[0]["path"] == str(alt)
    assert specs[0]["kind"] == "high_translated"


def test_matching_reasons_include_replace_translation_priors() -> None:
    cfg = PipelineConfig().matching
    src = [TextUnit(id="s1", polygon=[(0,0),(20,0),(20,20),(0,20)], block_ids=["b1"], text="你好世界", kind="speech")]
    tgt = [TextUnit(id="t1", polygon=[(1,1),(21,1),(21,21),(1,21)], block_ids=["c1"], text="abcd", kind="speech")]
    reg = RegistrationResult(
        matrix=np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float), method="identity", confidence=1.0,
        inlier_ratio=1.0, reprojection_error=0.0, spatial_coverage=1.0, num_matches=20,
        source_size=(32,32), target_size=(32,32), diagnostics={}
    )
    result = match_units(src, tgt, reg, cfg)
    assert len(result.matches) == 1
    reasons = result.matches[0].reasons
    assert any(r.startswith("projected_iou=") for r in reasons)
    assert any(r.startswith("text_length=") for r in reasons)


def test_threshold_clear_only_clears_dark_pixels() -> None:
    cfg = PipelineConfig().inpainting
    img = np.full((20, 20, 3), 245, np.uint8)
    img[9:11, 9:11] = (0, 0, 0)
    img[6:8, 6:8] = (220, 220, 220)
    mask = np.zeros((20, 20), np.uint8)
    mask[4:16, 4:16] = 255
    result = threshold_clear(img, mask, cfg)
    assert result.method == "threshold_clear"
    assert np.mean(result.image[9:11, 9:11]) > 220
    assert np.mean(result.image[6:8, 6:8]) >= 215


def test_dual_source_directory_actually_runs_secondary_direct(tmp_path: Path) -> None:
    pair, _, _ = _pair(tmp_path)
    secondary = tmp_path / "secondary"
    secondary.mkdir()
    # Same translated content in a higher-evidence channel. Prefer-secondary must
    # make the planner execute and select this Direct plan, not just suggest it.
    write_image(secondary / "cn.png", _same_layout_page(True))
    cfg = _base_config("direct_patch")
    cfg.dual_source.enabled = True
    cfg.dual_source.secondary_source_dir = str(secondary)
    cfg.dual_source.prefer_secondary_for_direct = True
    project = TransferPipeline(cfg).process_page(pair, tmp_path / "dual_out")
    rt = project.meta.get("replace_translation", {})
    planner = project.meta.get("transfer_planner", {})
    assert project.meta.get("direct_patch", {}).get("used") is True
    assert rt.get("secondary_source_available") is True
    # v0.9 arbitration no longer selects secondary merely because the prefer flag
    # is on. Identical evidence ties in favor of the primary authority source.
    assert rt.get("secondary_source_selected") is False
    assert rt.get("selected_source_kind") == "primary"
    assert Path(rt.get("selected_source_path", "")).name == "cn.png"
    assert planner.get("evidence", {}).get("secondary_source_selected") is False
    assert len(rt.get("dual_source_arbitration", rt.get("arbitration", [])) or []) >= 2
    assert (tmp_path / "dual_out" / "source_authority_original.png").exists()


def test_planner_exposes_evidence_and_secondary_retry_action() -> None:
    from manga_hd_transfer.transfer_planner import choose_transfer_strategy
    d = choose_transfer_strategy(
        "direct_patch",
        same_page=True,
        same_page_confidence=0.91,
        direct_plan_available=False,
        direct_plan_safe=False,
        secondary_source_available=True,
        secondary_source_selected=False,
    )
    payload = d.to_dict()
    assert payload["strategy"] == "direct_reject"
    assert payload["evidence"]["secondary_source_available"] is True
    assert "retry_direct_with_secondary_source" in payload["force_actions"]
    assert "force_mask_replace" in payload["force_actions"]


def test_matching_diagnostics_include_top_candidates_and_force_actions() -> None:
    cfg = PipelineConfig().matching
    cfg.max_cost = 0.01
    src = [TextUnit(id="s1", polygon=[(0,0),(12,0),(12,12),(0,12)], block_ids=["b1"], text="较长的中文译文", kind="speech")]
    tgt = [
        TextUnit(id="t1", polygon=[(40,40),(52,40),(52,52),(40,52)], block_ids=["c1"], text="JP", kind="speech"),
        TextUnit(id="t2", polygon=[(70,70),(82,70),(82,82),(70,82)], block_ids=["c2"], text="JP2", kind="speech"),
    ]
    reg = RegistrationResult(
        matrix=np.eye(3), method="identity", confidence=0.55, inlier_ratio=1.0,
        reprojection_error=0.0, spatial_coverage=1.0, num_matches=20,
        source_size=(100,100), target_size=(100,100), diagnostics={}
    )
    result = match_units(src, tgt, reg, cfg)
    assert "s1" in result.unmatched_source
    diag = result.diagnostics
    assert len(diag["top_candidates"]["s1"]) == 2
    assert diag["rejected_over_max_cost"]
    assert "force_match" in diag["force_actions"]
    reasons = diag["top_candidates"]["s1"][0]["reasons"]
    assert any(x.startswith("registration_penalty=") for x in reasons)


def test_fast_dark_pixel_clear_white_container_only() -> None:
    from manga_hd_transfer.mask_transfer import _fast_dark_pixel_clear
    cfg = PipelineConfig().mask_replace
    img = np.full((40, 40, 3), 248, np.uint8)
    img[15:22, 16:24] = 20
    env = np.zeros((40,40), np.uint8); env[8:32, 8:32] = 255
    cleared, actual, diag = _fast_dark_pixel_clear(img, env, cfg)
    assert cleared is not None
    assert diag["reason"] == "applied"
    assert cv2.countNonZero(actual) > 0
    assert np.mean(cleared[15:22, 16:24]) > 180
    # Strongly textured/dark container is rejected from this fast white path.
    textured = np.full((40, 40, 3), 120, np.uint8)
    rejected, _, diag2 = _fast_dark_pixel_clear(textured, env, cfg)
    assert rejected is None
    assert diag2["reason"] == "not_white_container"


def test_review_ui_exposes_force_actions() -> None:
    from manga_hd_transfer.review import _HTML
    assert "force_direct_patch" in _HTML
    assert "force_mask_replace" in _HTML
    assert "force_match" in _HTML
    assert "skip_unit" in _HTML


def test_replace_translation_regions_use_overlap_threshold() -> None:
    from manga_hd_transfer.pipeline import _replace_translation_regions
    s = TextUnit(id="s", polygon=[(0,0),(20,0),(20,20),(0,20)], block_ids=[], text="中文译文", kind="speech")
    t = TextUnit(id="t", polygon=[(2,2),(22,2),(22,22),(2,22)], block_ids=[], text="日本語", kind="speech")
    from manga_hd_transfer.models import UnitMatch
    m = UnitMatch("s", "t", 0.9, 0.1, "one_to_one", ["overlap=0.810", "projected_iou=0.681"])
    rows = _replace_translation_regions([s], [t], [m], overlap_threshold=0.30)
    assert rows[0]["translated_text"] == "中文译文"
    assert rows[0]["matched"] is True
    assert abs(rows[0]["overlap"] - 0.81) < 1e-6
    rows2 = _replace_translation_regions([s], [t], [m], overlap_threshold=0.90)
    assert rows2[0]["matched"] is False


def test_review_force_direct_really_reprocesses_page(tmp_path: Path) -> None:
    from manga_hd_transfer.review_apply import rerun_page_with_force
    pair, _, _ = _pair(tmp_path)
    out = tmp_path / "force_out"
    TransferPipeline(_base_config("direct_patch")).process_page(pair, out)
    forced = rerun_page_with_force(out, "direct_patch", _base_config("direct_patch"))
    assert forced.exists()
    import json
    project = json.loads((out / "project.json").read_text(encoding="utf-8"))
    action = json.loads((out / "force_action_result.json").read_text(encoding="utf-8"))
    assert project["meta"]["transfer_mode"] == "direct_patch"
    assert action["action"] == "force_direct_patch"


def _fake_direct_plan(shape: tuple[int, int], *, safe: bool = True, applied: int = 1, candidate_count: int = 1, boundary: float = 0.5):
    from types import SimpleNamespace
    mask = np.zeros(shape, np.uint8); mask[4:-4, 4:-4] = 255
    bubble = SimpleNamespace(mask=mask)
    recs = [SimpleNamespace(review_required=False, content_check="checked", content_complete=True) for _ in range(applied)]
    result = SimpleNamespace(applied_count=applied, records=recs)
    return SimpleNamespace(
        safe_to_skip_other_paths=safe,
        result=result,
        source_bubbles=[bubble],
        diagnostics={"candidate_count": candidate_count, "median_boundary_distance": boundary, "review_candidates_skipped": 0, "rejected_alignment": 0},
    )


def test_dual_source_arbitration_primary_sharper_wins() -> None:
    from types import SimpleNamespace
    from manga_hd_transfer.dual_source import build_direct_source_evidence, select_direct_source_candidate
    cfg = PipelineConfig().dual_source
    sharp = np.full((80,80,3), 255, np.uint8); cv2.putText(sharp, "A", (20,58), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3, cv2.LINE_8)
    soft = cv2.GaussianBlur(sharp, (0,0), 2.0)
    reg = SimpleNamespace(confidence=0.98, reprojection_error=0.5)
    pair = SimpleNamespace(confidence=0.99)
    plan = _fake_direct_plan((80,80))
    primary = build_direct_source_evidence(path="p", kind="primary", is_secondary=False, source=sharp, registration=reg, pair_check=pair, plan=plan, config=cfg)
    secondary = build_direct_source_evidence(path="s", kind="secondary_dir", is_secondary=True, source=soft, registration=reg, pair_check=pair, plan=plan, config=cfg)
    selected = select_direct_source_candidate([(primary, "P"), (secondary, "S")])
    assert selected is not None
    assert selected[1] == "P"
    assert primary.sharpness > secondary.sharpness


def test_dual_source_arbitration_secondary_sharper_wins() -> None:
    from types import SimpleNamespace
    from manga_hd_transfer.dual_source import build_direct_source_evidence, select_direct_source_candidate
    cfg = PipelineConfig().dual_source
    sharp = np.full((80,80,3), 255, np.uint8); cv2.putText(sharp, "A", (20,58), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3, cv2.LINE_8)
    soft = cv2.GaussianBlur(sharp, (0,0), 2.0)
    reg = SimpleNamespace(confidence=0.98, reprojection_error=0.5)
    pair = SimpleNamespace(confidence=0.99)
    plan = _fake_direct_plan((80,80))
    primary = build_direct_source_evidence(path="p", kind="primary", is_secondary=False, source=soft, registration=reg, pair_check=pair, plan=plan, config=cfg)
    secondary = build_direct_source_evidence(path="s", kind="secondary_dir", is_secondary=True, source=sharp, registration=reg, pair_check=pair, plan=plan, config=cfg)
    selected = select_direct_source_candidate([(primary, "P"), (secondary, "S")])
    assert selected is not None
    assert selected[1] == "S"


def test_dual_source_arbitration_rejects_bad_secondary_registration() -> None:
    from types import SimpleNamespace
    from manga_hd_transfer.dual_source import build_direct_source_evidence, select_direct_source_candidate
    cfg = PipelineConfig().dual_source
    img = np.full((80,80,3), 255, np.uint8); cv2.putText(img, "A", (20,58), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3, cv2.LINE_8)
    pair = SimpleNamespace(confidence=0.99)
    plan = _fake_direct_plan((80,80))
    primary = build_direct_source_evidence(path="p", kind="primary", is_secondary=False, source=img, registration=SimpleNamespace(confidence=0.97, reprojection_error=0.6), pair_check=pair, plan=plan, config=cfg)
    secondary = build_direct_source_evidence(path="s", kind="secondary_dir", is_secondary=True, source=img, registration=SimpleNamespace(confidence=0.40, reprojection_error=12.0), pair_check=pair, plan=plan, config=cfg)
    selected = select_direct_source_candidate([(primary, "P"), (secondary, "S")])
    assert selected is not None and selected[1] == "P"
    assert "registration_confidence_below_gate" in secondary.reject_reasons
    assert "reprojection_error_above_gate" in secondary.reject_reasons


def test_publication_gate_report_and_benchmark_discovery(tmp_path: Path) -> None:
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("publication_gate", Path(__file__).parents[1] / "scripts" / "publication_gate.py")
    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    work = tmp_path / "bench" / "work1"; work.mkdir(parents=True)
    save_json(work / "labels.json", {"work_id":"work1", "pages":[]})
    assert len(mod.discover_works(tmp_path / "bench")) == 1
    summary = mod.GateSummary(
        schema=mod.SCHEMA, generated_at="now", benchmark_root=str(tmp_path), version="test", page_count=0,
        measured_pair_pages=0, page_pair_accuracy=None, measured_match_pages=0, identity_match_accuracy=None,
        auto_pass_pages=0, auto_pass_residual_failures=0, auto_pass_border_damage_failures=0,
        safe_area_overflow_failures=0, direct_silent_mask_fallbacks=0, review_pages=0, review_rate=0.0,
        median_seconds_per_page=None, pass_gate=True, thresholds=mod.asdict(mod.GateThresholds()), pages=[]
    )
    out = tmp_path / "report"; out.mkdir()
    jp, mp = mod.write_report(summary, out)
    assert jp.exists() and mp.exists()
    assert "Publication Gate Report" in mp.read_text(encoding="utf-8")


def test_rigid_target_write_envelope_keeps_two_pixel_target_border() -> None:
    cfg = PipelineConfig().mask_replace
    mask = np.zeros((20, 20), np.uint8)
    mask[3:17, 3:17] = 255
    env = _rigid_target_write_envelope(mask, cfg)
    assert env[3, 10] == 0
    assert env[4, 10] == 0
    assert env[5, 10] == 255
    assert cv2.countNonZero(env) < cv2.countNonZero(mask)


def test_gap_fill_cannot_grow_back_into_rigid_target_border() -> None:
    cfg = PipelineConfig().mask_replace
    target_mask = np.zeros((25, 25), np.uint8)
    target_mask[2:23, 2:23] = 255
    safe = _rigid_target_write_envelope(target_mask, cfg)
    base = np.zeros((25, 25), np.uint8)
    base[8:17, 8:17] = 255
    source = np.full((25, 25, 3), 255, np.uint8)
    target = np.full((25, 25, 3), 245, np.uint8)
    expanded, diag = _expand_safe_write_mask(base, safe, source, target, cfg)
    protected_ring = cv2.bitwise_and(target_mask, cv2.bitwise_not(safe))
    assert diag["added_pixels"] > 0
    assert np.count_nonzero((expanded > 0) & (protected_ring > 0)) == 0


def test_direct_registration_gate_now_uses_basic_geometry_confidence_only() -> None:
    source = _same_layout_page(True)
    target = _same_layout_color_target()
    cfg = PipelineConfig().direct_patch
    cfg.publication_safety_enabled = True  # ignored legacy value
    reg = RegistrationResult(
        matrix=np.eye(3), method="real_pair_like", confidence=0.80,
        inlier_ratio=0.75, reprojection_error=0.84, spatial_coverage=0.66,
        num_matches=80, source_size=(650,900), target_size=(650,900), diagnostics={},
    )
    ok, relaxed = _source_direct_registration_gate(source, target, reg, cfg)
    assert ok is True and relaxed is False
    bad = RegistrationResult(
        matrix=np.eye(3), method="bad", confidence=0.20,
        inlier_ratio=0.40, reprojection_error=3.0, spatial_coverage=0.25,
        num_matches=20, source_size=(650,900), target_size=(650,900), diagnostics={},
    )
    ok2, relaxed2 = _source_direct_registration_gate(source, target, bad, cfg)
    assert ok2 is False and relaxed2 is False



def test_publication_safety_is_disabled_by_default() -> None:
    cfg = PipelineConfig()
    assert cfg.mask_replace.publication_safety_enabled is False
    assert cfg.direct_patch.publication_safety_enabled is False
    assert cfg.direct_patch.source_direct_fail_on_artwork_rejections is False
    assert cfg.direct_patch.allow_target_aware_colored_composite is True
    assert cfg.direct_patch.source_direct_colored_preserve_target_fill is True


def test_aggressive_triage_does_not_downgrade_applied_region() -> None:
    cfg = PipelineConfig().mask_replace
    rec = MaskTransferRecord('s','t',0.20,True,'applied',content_check='checked',content_complete=False,review_required=True)
    finalize_transfer_records([rec], cfg)
    assert rec.triage_state == 'SAFE'


def test_aggressive_direct_registration_ignores_publication_quality_thresholds() -> None:
    source = _same_layout_page(True)
    target = _same_layout_color_target()
    reg = RegistrationResult(
        matrix=np.eye(3), method="usable_but_low_quality", confidence=0.40,
        inlier_ratio=0.25, reprojection_error=4.0, spatial_coverage=0.20,
        num_matches=12, source_size=(650,900), target_size=(650,900), diagnostics={},
    )
    cfg = PipelineConfig().direct_patch
    ok, relaxed = _source_direct_registration_gate(source, target, reg, cfg)
    assert ok is True and relaxed is False


def test_geometry_snap_protects_container_border_but_not_internal_japanese_ink() -> None:
    cfg = PipelineConfig().direct_patch
    geom = np.zeros((40, 40), np.uint8); geom[3:37, 3:37] = 1
    dist = np.full((40, 40), 5.0, np.float32)
    # Strong target outline at the proposed container boundary.
    dist[3:5, 3:37] = 0.0; dist[35:37, 3:37] = 0.0; dist[3:37, 3:5] = 0.0; dist[3:37, 35:37] = 0.0
    # Japanese glyph stroke well inside the balloon. It must remain writable.
    dist[18:22, 18:22] = 0.0
    protected, diag = _protect_registered_geometry_boundary(geom, dist, cfg)
    assert np.count_nonzero(protected[3:5, 3:37]) == 0
    assert np.all(protected[18:22, 18:22])
    assert diag["protected_boundary_pixels"] > 0
    assert diag["internal_dark_writable_pixels"] >= 16



def test_direct_target_border_guard_shrinks_all_sides_and_keeps_internal_text_writable() -> None:
    cfg = PipelineConfig().direct_patch
    use = np.zeros((44, 52), np.uint8); use[4:40, 5:47] = 1
    geom = use.copy()
    dist = np.full((44, 52), 6.0, np.float32)
    # Structural target lines on the container boundary.
    dist[4:7, 5:47] = 0.0; dist[37:40, 5:47] = 0.0
    dist[4:40, 5:8] = 0.0; dist[4:40, 44:47] = 0.0
    # Japanese glyph stroke well inside the box: line detection must not protect it.
    dist[21:24, 25:28] = 0.0
    safe, protected, diag = _constrain_direct_write_to_target_border(use, geom, dist, cfg)
    assert diag["enabled"] is True
    assert diag["target_inset_px"] == 2
    assert diag["protected_pixels"] > 0
    assert not np.any(safe[4:7, 5:47])
    assert not np.any(safe[37:40, 5:47])
    assert not np.any(safe[4:40, 5:8])
    assert not np.any(safe[4:40, 44:47])
    assert np.all(safe[21:24, 25:28])
    assert np.any(protected[4:7, 5:47])


def test_direct_v121_defaults_use_stronger_source_inset_and_target_border_restore() -> None:
    cfg = PipelineConfig().direct_patch
    assert cfg.source_direct_border_inset_px == 5
    assert cfg.source_direct_alignment_search_px == 16
    assert cfg.source_direct_progressive_inset_steps == 5
    assert cfg.source_direct_progressive_max_outer_dark_ratio == 0.18
    assert cfg.source_direct_target_border_guard_enabled is True
    assert cfg.source_direct_target_border_inset_px == 2
    assert cfg.source_direct_exact_target_border_restore is True


def test_direct_plan_preserves_target_container_border_byte_exact() -> None:
    source = _same_layout_page(True)
    target = _same_layout_page(False)
    reg = RegistrationResult(
        matrix=np.eye(3), method="identity", confidence=0.99, inlier_ratio=0.99, reprojection_error=0.0,
        spatial_coverage=0.95, num_matches=100, source_size=(650, 900), target_size=(650, 900), diagnostics={},
    )
    cfg = PipelineConfig().direct_patch
    plan = build_source_direct_container_plan(source, target, reg, cfg)
    assert plan is not None
    assert plan.diagnostics["target_border_guard_enabled"] is True
    assert plan.diagnostics["target_border_protected_pixels"] > 0
    assert plan.diagnostics["target_border_changed_after_restore"] == 0
    # The ellipse outline itself is TARGET structure and must remain identical.
    border = np.zeros(target.shape[:2], np.uint8)
    cv2.ellipse(border, (330, 255), (130, 90), 0, 0, 360, 255, 4)
    use_border = border > 0
    assert np.array_equal(plan.result.image[use_border], target[use_border])

def test_cross_rendition_direct_has_large_region_safety_cap() -> None:
    cfg = PipelineConfig().direct_patch
    assert 0.01 < cfg.source_direct_cross_rendition_max_auto_area_ratio <= 0.05


def test_missing_optional_ocr_backend_soft_falls_back_to_null(monkeypatch) -> None:
    import manga_hd_transfer.pipeline as pipeline_module
    from manga_hd_transfer.ocr import NullOCRBackend
    cfg = PipelineConfig(); cfg.ocr.backend = "paddle"; cfg.ocr.soft_fail_missing_backend = True
    pipe = TransferPipeline(cfg)
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("PaddleOCR is not installed")
    monkeypatch.setattr(pipeline_module, "build_backend", unavailable)
    assert isinstance(pipe.source_ocr, NullOCRBackend)
    assert pipe._ocr_soft_failures


def test_review_local_clear_mask_preserves_unrelated_automatic_final_pixels(tmp_path: Path) -> None:
    """A local QA edit must not reconstruct unrelated automatic output to JP."""
    from manga_hd_transfer.review_apply import apply_review_page
    target = np.full((120, 160, 3), 245, np.uint8)
    source = target.copy()
    final = target.copy()
    # Simulate a successful automatic replacement from a supplemental route not
    # represented by the editable Direct layer.
    final[80:105, 105:145] = (20, 40, 210)
    write_image(tmp_path / "target_original.png", target)
    write_image(tmp_path / "source_original.png", source)
    write_image(tmp_path / "final.png", final)
    blank = np.zeros((120, 160, 4), np.uint8)
    ok, data = cv2.imencode(".png", blank); assert ok; data.tofile(tmp_path / "direct_patch_layer.png")
    auto_clear = np.zeros((120,160), np.uint8); write_image(tmp_path / "target_clear_mask.png", auto_clear)
    manual_clear = auto_clear.copy(); manual_clear[20:35, 20:35] = 255; write_image(tmp_path / "manual_clear_mask.png", manual_clear)
    project = {
        "pair": {"source_path": str(tmp_path / "source_original.png"), "target_path": str(tmp_path / "target_original.png"), "source_index": 0, "target_index": 0, "confidence": 1.0, "score": 1.0, "reasons": []},
        "registration": {"matrix": np.eye(3).tolist(), "method": "identity", "confidence": 1.0},
        "meta": {"transfer_mode": "direct_patch", "passthrough": False, "direct_patch": {"used": True, "review_regions": []}},
        "target_bubbles": [], "source_units": [], "target_units": [], "target_blocks": [], "matches": [],
    }
    save_json(tmp_path / "project.json", project)
    save_json(tmp_path / "review_overrides.json", {"status": "reviewed"})
    out = apply_review_page(tmp_path, PipelineConfig())
    result = read_image(out)
    assert np.array_equal(result[80:105,105:145], final[80:105,105:145])
    report = __import__('json').loads((tmp_path / "review_applied.json").read_text())
    assert report["automatic_final_preserved_outside_review"] is True
    assert report["review_change_pixels"] > 0


def _completion_bubble(bid: str, bbox: tuple[int,int,int,int], *, white: float, dark: float, sat_median: float, sat_p75: float) -> BubbleInstance:
    x0,y0,x1,y1=bbox
    mask=np.zeros((200,200),np.uint8); mask[y0:y1,x0:x1]=255
    return BubbleInstance(
        id=bid, polygon=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)], confidence=.8,
        kind="speech", block_ids=[], mask=mask, safe_mask=mask.copy(),
        meta={"backend":"unseeded_white","white_ratio":white,"dark_ratio":dark,
              "saturation_median":sat_median,"saturation_p75":sat_p75},
    )


def test_completion_keeps_dense_small_neutral_white_balloon_below_old_084_threshold() -> None:
    cfg=PipelineConfig().mask_replace
    src=_completion_bubble("s",(20,20,80,80),white=.76,dark=.21,sat_median=0,sat_p75=0)
    dst=_completion_bubble("t",(20,20,80,80),white=.810,dark=.139,sat_median=0,sat_p75=0)
    ss,tt=_filter_uncovered_white_completion_pairs([src],[dst],[],cfg)
    assert ss == [src] and tt == [dst]


def test_completion_does_not_white_patch_light_colored_burst_component() -> None:
    cfg=PipelineConfig().mask_replace
    src=_completion_bubble("s",(20,20,100,100),white=.88,dark=.10,sat_median=0,sat_p75=0)
    dst=_completion_bubble("t",(20,20,100,100),white=.58,dark=.07,sat_median=32,sat_p75=48)
    ss,tt=_filter_uncovered_white_completion_pairs([src],[dst],[],cfg)
    assert ss == [] and tt == []


def test_content_incomplete_direct_invariant_is_diagnostic_not_planner_blocker() -> None:
    issues=["content_incomplete:direct-src-0000","alignment_border_was_written"]
    assert _blocking_direct_invariant_issues(issues) == ["alignment_border_was_written"]
    assert _blocking_direct_invariant_issues(["content_incomplete:direct-src-0000"]) == []


def test_text_only_transfer_never_copies_source_paper_into_colored_target() -> None:
    h, w = 120, 160
    source = np.full((h, w, 3), 255, np.uint8)
    target = np.full((h, w, 3), (145, 80, 205), np.uint8)
    # shared artwork line should remain target-coloured; text differs by edition.
    cv2.line(source, (10, 100), (150, 100), (20, 20, 20), 2)
    cv2.line(target, (10, 100), (150, 100), (20, 20, 20), 2)
    cv2.putText(source, "CN", (30, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(target, "JP", (83, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
    region = np.zeros((h, w), np.uint8); region[15:85, 15:145] = 255
    out, write, _sm, diag = transfer_text_only(target, source, region, tolerance_px=3)
    assert diag["background_policy"] == "target_only"
    assert cv2.countNonZero(write) > 0
    # A background sample inside the editable rectangle remains exactly TARGET.
    assert np.array_equal(out[25, 25], target[25, 25])
    # SOURCE white paper cannot appear anywhere that was untouched by text write.
    no_write = (write == 0) & (region > 0)
    assert np.array_equal(out[no_write], target[no_write])


def test_rigid_transfer_keeps_reverse_target_driven_pair_in_mixed_batch() -> None:
    h, w = 180, 240
    source = np.full((h, w, 3), 255, np.uint8)
    target = np.full((h, w, 3), 255, np.uint8)
    # Two separated white containers with different source/target lettering.
    boxes = [(20, 25, 100, 85), (130, 95, 220, 160)]
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(source, (x0,y0), (x1,y1), (0,0,0), 2)
        cv2.rectangle(target, (x0,y0), (x1,y1), (0,0,0), 2)
    cv2.putText(source, "C", (45,65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)
    cv2.putText(target, "J", (55,65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)
    cv2.putText(source, "N", (155,140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)
    cv2.putText(target, "P", (170,140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)
    def bubble(bid, box, meta):
        x0,y0,x1,y1=box
        m=np.zeros((h,w),np.uint8); cv2.rectangle(m,(x0,y0),(x1,y1),255,-1)
        b=BubbleInstance(bid, [(x0,y0),(x1,y0),(x1,y1),(x0,y1)], confidence=.99, meta=meta)
        b.mask=m; b.safe_mask=m.copy(); return b
    s1=bubble('s1',boxes[0],{'paired_target_id':'t1'})
    t1=bubble('t1',boxes[0],{'paired_source_id':'s1'})
    # Relation for the second pair exists on TARGET only: this used to disappear
    # whenever the same batch also contained an ordinary paired_target_id pair.
    s2=bubble('s2',boxes[1],{'target_driven_recovery':True,'backend':'unseeded_white'})
    t2=bubble('t2',boxes[1],{'paired_source_id':'s2','target_driven_recovery':True,'backend':'unseeded_white'})
    cfg=PipelineConfig().mask_replace
    result=transfer_rigid_container_rasters(source,target,target.copy(),[s1,s2],[t1,t2],cfg)
    ids={(r.source_bubble_id,r.target_bubble_id) for r in result.records}
    assert ('s1','t1') in ids
    assert ('s2','t2') in ids
