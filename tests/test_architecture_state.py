from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np

from manga_hd_transfer import __version__
from manga_hd_transfer.cache import load_completed_page
from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.io_utils import load_json, save_json, write_image
from manga_hd_transfer.manual_review_service import commit_manual_effect
from manga_hd_transfer.models import PagePair
from manga_hd_transfer.result_state import (
    commit_reviewed_result,
    ensure_manual_baseline,
    manual_baseline_path,
    resolve_result_state,
)
from manga_hd_transfer.schema_compat import merge_review_overrides


def test_version_has_one_runtime_source_and_matches_package_metadata():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M)
    assert match and match.group(1) == __version__
    gui = (root / "src" / "manga_hd_transfer" / "gui_qt.py").read_text(encoding="utf-8")
    assert "VERSION = __version__" in gui


def test_review_result_state_commit_is_atomic_consistent_and_keeps_stable_base(tmp_path: Path):
    auto = np.full((30, 40, 3), 20, np.uint8)
    reviewed = np.full((30, 40, 3), 190, np.uint8)
    write_image(tmp_path / "final.png", auto)
    stable = ensure_manual_baseline(tmp_path)
    stable_before = cv2.imread(str(stable), cv2.IMREAD_COLOR)
    assert stable_before is not None and np.array_equal(stable_before, auto)

    write_image(tmp_path / "final_reviewed.png", reviewed)
    commit_reviewed_result(tmp_path, tmp_path / "final_reviewed.png", update_project=False)

    final = cv2.imread(str(tmp_path / "final.png"), cv2.IMREAD_COLOR)
    stable_after = cv2.imread(str(stable), cv2.IMREAD_COLOR)
    assert final is not None and np.array_equal(final, reviewed)
    assert stable_after is not None and np.array_equal(stable_after, auto)
    sync = load_json(tmp_path / "review_sync.json")
    assert sync["schema"].endswith("review_sync.v3")
    assert sync["synced"] is True
    assert len(sync["sha256"]) == 64
    state = resolve_result_state(tmp_path)
    assert state.current is not None
    assert state.stable_manual_base == stable
    assert manual_baseline_path(tmp_path) == stable


def test_partial_web_review_merge_never_erases_qt_manual_effect_state():
    existing = {
        "manual_effect_regions": [{"id": "reveal-1", "mode": "reveal_text"}],
        "manual_reletter": [{"target_bubble_id": "b1", "text": "中文"}],
        "notes": "old",
    }
    merged = merge_review_overrides(existing, {
        "notes": "new",
        "text_overrides": {"u1": "修正"},
        "accepted_source_units": ["u1"],
    })
    assert merged["notes"] == "new"
    assert merged["manual_effect_regions"] == existing["manual_effect_regions"]
    assert merged["manual_reletter"] == existing["manual_reletter"]
    assert merged["text_overrides"] == {"u1": "修正"}


def test_core_manual_review_service_commits_reveal_without_qt(tmp_path: Path):
    h, w = 120, 160
    target = np.full((h, w, 3), (140, 70, 120), np.uint8)
    source = target.copy()
    cv2.putText(source, "CN", (18, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(target, "JP", (82, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
    write_image(tmp_path / "source_original.png", source)
    write_image(tmp_path / "target_original.png", target)
    write_image(tmp_path / "final.png", target)
    save_json(tmp_path / "project.json", {
        "pair": {"source_path": str(tmp_path / "source_original.png"), "target_path": str(tmp_path / "target_original.png"), "source_index": 0, "target_index": 0, "confidence": 1.0, "score": 1.0, "reasons": []},
        "registration": {"matrix": np.eye(3).tolist(), "method": "identity", "confidence": 1.0},
        "meta": {"transfer_mode": "auto", "passthrough": True},
    })
    reveal = np.zeros((h, w), np.uint8); reveal[15:100, 8:150] = 255
    patch = np.zeros((h, w, 4), np.uint8)
    patch[40:58, 35:55, :3] = 0; patch[40:58, 35:55, 3] = 255
    row = {
        "id": "service-reveal", "mode": "reveal_text", "target_bbox": [8, 15, 150, 100],
        "diff_threshold": 20, "edge_threshold": 35, "expand_px": 2, "auto_clear_target": True,
    }
    stages = []
    result = commit_manual_effect(
        tmp_path, row, reveal, patch, PipelineConfig(),
        trace=lambda stage, payload: stages.append(stage),
    )
    assert result.final_reviewed.exists() and result.final.exists()
    assert result.preview_patch_exact is True
    assert "overrides_saved" in stages and "final_verified" in stages
    a = cv2.imread(str(result.final_reviewed), cv2.IMREAD_COLOR)
    b = cv2.imread(str(result.final), cv2.IMREAD_COLOR)
    assert a is not None and b is not None and np.array_equal(a, b)


def test_cache_resume_treats_legacy_boolean_meta_as_cache_miss_not_exception(tmp_path: Path):
    source = tmp_path / "s.png"; target = tmp_path / "t.png"; final = tmp_path / "final.png"
    image = np.full((16, 16, 3), 255, np.uint8)
    write_image(source, image); write_image(target, image); write_image(final, image)
    pair = PagePair(str(source), str(target), 0, 0, 1.0, 1.0, [])
    save_json(tmp_path / "project.json", {"meta": True, "registration": True})
    # Invalid legacy cache state is simply ignored; it must never crash resume.
    assert load_completed_page(tmp_path, pair, PipelineConfig(), final) is None


def test_first_manual_baseline_uses_newer_reviewed_result_when_no_legacy_base(tmp_path: Path):
    import os, time
    automatic = np.full((20, 24, 3), 40, np.uint8)
    reviewed = np.full((20, 24, 3), 180, np.uint8)
    write_image(tmp_path / "final.png", automatic)
    write_image(tmp_path / "final_reviewed.png", reviewed)
    now = time.time()
    os.utime(tmp_path / "final.png", (now - 10, now - 10))
    os.utime(tmp_path / "final_reviewed.png", (now, now))
    stable = ensure_manual_baseline(tmp_path)
    got = cv2.imread(str(stable), cv2.IMREAD_COLOR)
    assert got is not None and np.array_equal(got, reviewed)


def test_page_management_legacy_boolean_state_is_ignored_safely():
    from manga_hd_transfer.page_management import PageMark, marks_from_json
    assert PageMark.from_dict(True).page_type == "content"
    assert marks_from_json(True) == {}
    assert marks_from_json({"pages": True}) == {}
