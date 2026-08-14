from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from manga_hd_transfer.aligned_overlay_reveal import build_aligned_overlay_plan, execute_aligned_overlay
from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.io_utils import save_json, write_image
from manga_hd_transfer.models import PagePair, RegistrationResult
from manga_hd_transfer.result_state import commit_automatic_result
from manga_hd_transfer.schema_compat import normalize_project
from manga_hd_transfer.transfer_planner import choose_transfer_strategy
from manga_hd_transfer.workspace import resolve_page_workspace


def _reg(w: int, h: int, confidence: float = 0.99) -> RegistrationResult:
    return RegistrationResult(
        matrix=np.eye(3, dtype=np.float64),
        method="identity-test",
        confidence=confidence,
        inlier_ratio=0.98,
        reprojection_error=0.15,
        spatial_coverage=0.92,
        num_matches=80,
        source_size=(w, h),
        target_size=(w, h),
        diagnostics={"route": "identity-test"},
    )


def _white_pair(h: int = 180, w: int = 240) -> tuple[np.ndarray, np.ndarray]:
    source = np.full((h, w, 3), 255, np.uint8)
    target = source.copy()
    # Shared bubble border: this must never be selected as text or damaged.
    cv2.ellipse(source, (120, 90), (80, 58), 0, 0, 360, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.ellipse(target, (120, 90), (80, 58), 0, 0, 360, (20, 20, 20), 2, cv2.LINE_AA)
    # SOURCE and TARGET use deliberately different text geometry inside the same
    # white container.  The strokes are close enough to form one text corridor.
    cv2.line(source, (99, 68), (99, 103), (10, 10, 10), 4, cv2.LINE_AA)
    cv2.line(source, (91, 82), (108, 82), (10, 10, 10), 4, cv2.LINE_AA)
    cv2.line(source, (122, 70), (122, 104), (10, 10, 10), 4, cv2.LINE_AA)
    cv2.line(source, (114, 96), (132, 96), (10, 10, 10), 4, cv2.LINE_AA)

    cv2.line(target, (103, 68), (103, 103), (10, 10, 10), 4, cv2.LINE_AA)
    cv2.line(target, (95, 74), (112, 74), (10, 10, 10), 4, cv2.LINE_AA)
    cv2.line(target, (126, 70), (126, 104), (10, 10, 10), 4, cv2.LINE_AA)
    cv2.line(target, (117, 86), (135, 86), (10, 10, 10), 4, cv2.LINE_AA)
    return source, target


def test_feature_is_default_off_and_auto_cannot_silently_select_it():
    cfg = PipelineConfig()
    assert cfg.aligned_overlay_reveal.enabled is False
    assert cfg.aligned_overlay_reveal.allow_in_auto is False
    assert cfg.aligned_overlay_reveal.require_explicit_mode is True
    decision = choose_transfer_strategy(
        "auto", same_page=True, same_page_confidence=0.99,
        direct_plan_available=False, direct_plan_safe=False,
        aligned_plan_available=True, aligned_plan_safe=True,
        aligned_auto_allowed=False,
    )
    assert decision.strategy == "mask_replace"


def test_white_bubble_ink_only_reveal_changes_text_but_preserves_shared_border():
    source, target = _white_pair()
    h, w = target.shape[:2]
    cfg = PipelineConfig().aligned_overlay_reveal.model_copy(deep=True)
    cfg.enabled = True
    cfg.region_group_radius_px = 12
    cfg.max_component_span_ratio = 0.30
    cfg.max_component_area_ratio = 0.02
    cfg.safe_white_ratio = 0.70
    cfg.safe_registration_confidence = 0.95
    plan = build_aligned_overlay_plan(source, target, _reg(w, h), cfg)
    assert plan.accepted is True
    assert cv2.countNonZero(plan.erase_mask) > 0
    assert cv2.countNonZero(plan.source_ink_mask) > 0
    result = execute_aligned_overlay(plan, source, target, cfg)
    assert result.accepted is True
    assert result.applied_count > 0
    assert not np.array_equal(result.image, target)

    # Shared outer border is byte-for-byte untouched by the text corridor.
    border_probe = np.zeros((h, w), np.uint8)
    cv2.ellipse(border_probe, (120, 90), (80, 58), 0, 0, 360, 255, 7, cv2.LINE_8)
    sel = border_probe > 0
    assert np.array_equal(result.image[sel], target[sel])
    # Ink-only contract: page-level full-raster mask remains empty by default.
    assert cv2.countNonZero(plan.full_raster_mask) == 0
    assert result.diagnostics["source_background_authority"] is False


def test_color_target_never_exposes_bw_source_background():
    source, target = _white_pair()
    h, w = target.shape[:2]
    # Give TARGET a saturated coloured background while SOURCE stays B/W.
    color_target = np.empty_like(target)
    color_target[:] = (145, 70, 125)
    # Keep the same Japanese strokes on the colour master.
    dark = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) < 100
    color_target[dark] = target[dark]

    cfg = PipelineConfig().aligned_overlay_reveal.model_copy(deep=True)
    cfg.enabled = True
    cfg.prefer_source_ink_only = False  # deliberately request the risky fallback
    cfg.region_group_radius_px = 12
    cfg.max_component_span_ratio = 0.30
    cfg.max_component_area_ratio = 0.02
    plan = build_aligned_overlay_plan(source, color_target, _reg(w, h), cfg)
    result = execute_aligned_overlay(plan, source, color_target, cfg)

    # Strict colour gating may reject the whole page; either way a white SOURCE
    # background may never replace saturated TARGET pixels.
    assert cv2.countNonZero(plan.full_raster_mask) == 0
    assert not np.any(np.all(result.image == (255, 255, 255), axis=2) & np.all(color_target == (145, 70, 125), axis=2))
    if not plan.accepted:
        assert np.array_equal(result.image, color_target)


def test_poor_registration_keeps_target_exactly_untouched():
    source, target = _white_pair()
    h, w = target.shape[:2]
    cfg = PipelineConfig().aligned_overlay_reveal.model_copy(deep=True)
    cfg.enabled = True
    plan = build_aligned_overlay_plan(source, target, _reg(w, h, confidence=0.50), cfg)
    assert plan.accepted is False
    assert plan.reason.startswith("rejected_registration")
    result = execute_aligned_overlay(plan, source, target, cfg)
    assert np.array_equal(result.image, target)
    assert cv2.countNonZero(result.erase_mask) == 0


def test_explicit_planner_never_falls_back_to_mask_when_candidate_rejected():
    rejected = choose_transfer_strategy(
        "aligned_overlay_reveal", same_page=True, same_page_confidence=0.99,
        direct_plan_available=False, direct_plan_safe=False,
        aligned_plan_available=False, aligned_plan_safe=False,
    )
    assert rejected.strategy.startswith("aligned_overlay_reveal")
    assert rejected.strategy != "mask_replace"
    assert rejected.fallback_allowed is False
    assert "aligned_overlay" in rejected.reason


def test_schema_and_workspace_route_aligned_candidates(tmp_path: Path):
    source = tmp_path / "src.png"; target = tmp_path / "jp.png"
    image = np.full((30, 40, 3), 255, np.uint8)
    write_image(source, image); write_image(target, image)
    pair = PagePair(str(source), str(target), 0, 0, 1.0, 1.0, [])
    out = tmp_path / "out"
    root = out / "pages" / "jp"
    root.mkdir(parents=True)
    write_image(root / "final.png", image)
    write_image(root / "aligned_overlay_reveal_mask.png", np.zeros((30, 40), np.uint8))
    candidate = {"workflow": "manual_effect", "target_bbox": [2, 3, 20, 25], "reason": "aligned_overlay:test"}
    raw = {
        "meta": {
            "transfer_mode": "aligned_overlay_reveal",
            "transfer_planner": {"strategy": "aligned_overlay_reveal"},
            "aligned_overlay_reveal": {"used": True, "manual_effect_candidates": [candidate], "review_regions": [candidate]},
        },
        "artifacts": {"aligned_overlay_reveal_mask": str(root / "aligned_overlay_reveal_mask.png")},
    }
    normalized = normalize_project(raw)
    assert normalized["meta"]["aligned_overlay_reveal"]["used"] is True
    save_json(root / "project.json", normalized)
    ws = resolve_page_workspace(out, pair)
    assert ws.manual_effect_candidates == [candidate]
    assert ws.review_regions == [candidate]
    assert Path(ws.mask_path).name == "aligned_overlay_reveal_mask.png"


def test_automatic_result_commit_is_owned_by_result_state(tmp_path: Path):
    image = np.full((16, 24, 3), 123, np.uint8)
    book = tmp_path / "book" / "p.png"
    local, external = commit_automatic_result(tmp_path / "page", image, book)
    assert local.exists() and external == book and book.exists()
    assert np.array_equal(cv2.imread(str(local)), image)
    assert np.array_equal(cv2.imread(str(book)), image)


def test_pipeline_explicit_aligned_route_writes_owned_artifacts(tmp_path: Path, monkeypatch):
    from manga_hd_transfer.pipeline import TransferPipeline

    source_img, target_img = _white_pair()
    h, w = target_img.shape[:2]
    source = tmp_path / "source.png"; target = tmp_path / "target.png"
    write_image(source, source_img); write_image(target, target_img)
    pair = PagePair(str(source), str(target), 0, 0, 1.0, 0.0, [])

    cfg = PipelineConfig()
    cfg.cache.enabled = False
    cfg.transfer.mode = "aligned_overlay_reveal"
    cfg.pairing.same_page_precheck_enabled = False
    cfg.aligned_overlay_reveal.enabled = True
    cfg.aligned_overlay_reveal.region_group_radius_px = 12
    cfg.aligned_overlay_reveal.max_component_span_ratio = 0.30
    cfg.aligned_overlay_reveal.max_component_area_ratio = 0.02

    monkeypatch.setattr("manga_hd_transfer.pipeline.register_images", lambda *_a, **_k: _reg(w, h))
    root = tmp_path / "page"
    project = TransferPipeline(cfg).process_page(pair, root)

    assert project.meta["transfer_mode"] == "aligned_overlay_reveal"
    assert project.meta["aligned_overlay_reveal"]["accepted"] is True
    assert project.meta["direct_patch"]["used"] is False
    assert project.meta["mask_replace"]["used"] is False
    for name in (
        "final.png", "aligned_overlay_reveal_layer.png", "aligned_overlay_reveal_mask.png",
        "aligned_overlay_reveal_source_ink.png", "aligned_overlay_reveal_regions.png",
        "aligned_overlay_reveal.json", "project.json", "qa.json",
    ):
        assert (root / name).exists(), name
    final = cv2.imread(str(root / "final.png"), cv2.IMREAD_COLOR)
    assert final is not None and not np.array_equal(final, target_img)


def test_white_bubble_erase_source_uses_closed_target_container_envelope():
    source, target = _white_pair()
    h, w = target.shape[:2]
    cfg = PipelineConfig().aligned_overlay_reveal.model_copy(deep=True)
    cfg.enabled = True
    cfg.erase_source = "white_bubble_interior"
    cfg.region_group_radius_px = 12
    cfg.max_component_span_ratio = 0.30
    cfg.max_component_area_ratio = 0.02
    plan = build_aligned_overlay_plan(source, target, _reg(w, h), cfg)
    assert plan.accepted is True
    assert plan.diagnostics["erase_source"] == "white_bubble_interior"
    assert any(r.diagnostics["white_container_pixels"] > 0 for r in plan.regions)
    result = execute_aligned_overlay(plan, source, target, cfg)
    assert result.applied_count > 0


def test_v123_real_page_defaults_are_relaxed_but_feature_stays_opt_in():
    cfg = PipelineConfig().aligned_overlay_reveal
    assert cfg.enabled is False
    assert cfg.allow_in_auto is False
    assert cfg.require_explicit_mode is True
    assert cfg.min_registration_confidence == 0.80
    assert cfg.max_reprojection_error == 1.8
    assert cfg.min_inlier_ratio == 0.65
    assert cfg.min_spatial_coverage == 0.50
    assert cfg.erase_source == "hybrid"
    assert cfg.max_component_area_ratio == 0.03
    assert cfg.max_component_span_ratio == 0.40
    assert cfg.max_single_region_area_ratio == 0.10
    assert cfg.max_erase_area_ratio_per_page == 0.25


def test_medium_registration_081_now_builds_visible_candidate_with_defaults():
    source, target = _white_pair()
    h, w = target.shape[:2]
    cfg = PipelineConfig().aligned_overlay_reveal.model_copy(deep=True)
    cfg.enabled = True
    cfg.region_group_radius_px = 12
    plan = build_aligned_overlay_plan(source, target, _reg(w, h, confidence=0.81), cfg)
    assert plan.accepted is True
    assert cv2.countNonZero(plan.erase_mask) > 0
    result = execute_aligned_overlay(plan, source, target, cfg)
    assert result.applied_count > 0
    assert result.diagnostics["changed_pixels"] > 0
    assert not np.array_equal(result.image, target)


def test_registration_rejection_reports_metrics_and_thresholds():
    source, target = _white_pair()
    h, w = target.shape[:2]
    cfg = PipelineConfig().aligned_overlay_reveal.model_copy(deep=True)
    cfg.enabled = True
    plan = build_aligned_overlay_plan(source, target, _reg(w, h, confidence=0.79), cfg)
    assert plan.accepted is False
    assert plan.reason == "rejected_registration:registration_confidence"
    assert plan.diagnostics["registration_gate"]["confidence"] == 0.79
    assert plan.diagnostics["thresholds"]["min_registration_confidence"] == 0.80
    assert plan.diagnostics["thresholds"]["max_reprojection_error"] == 1.8
