from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.io_utils import save_json, write_image
from manga_hd_transfer.models import PagePair
from manga_hd_transfer.review_apply import apply_review_page
from manga_hd_transfer.schema_compat import normalize_overrides, normalize_project, normalize_review_applied
from manga_hd_transfer.workspace import resolve_page_workspace, page_id_for_pair


def _page(h=160, w=220):
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (145, 72, 115)
    return img


def _source_target():
    src = _page(); tgt = _page()
    cv2.putText(src, "CN", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245,245,245), 3, cv2.LINE_AA)
    cv2.putText(tgt, "JP", (105, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245,245,245), 3, cv2.LINE_AA)
    return src, tgt


def test_normalize_project_converts_all_legacy_bool_shapes():
    raw = {
        "pair": {"source_path":"s.png", "target_path":"t.png", "source_index":0, "target_index":0, "confidence":1.0, "score":1.0, "reasons":[]},
        "registration": True,
        "artifacts": True,
        "meta": {
            "direct_patch": True,
            "mask_replace": False,
            "qa_summary": True,
            "review_sync": False,
            "auto_applied_match_ids": True,
        },
        "source_units": True,
        "target_units": [False, {"id":"ok"}],
        "target_blocks": False,
        "target_bubbles": True,
        "matches": [True, {"relation":"one_to_one"}],
    }
    p = normalize_project(raw)
    assert isinstance(p["registration"], dict)
    assert isinstance(p["artifacts"], dict)
    assert p["meta"]["direct_patch"]["used"] is True
    assert p["meta"]["mask_replace"]["used"] is False
    assert isinstance(p["meta"]["qa_summary"], dict)
    assert p["source_units"] == []
    assert p["target_units"] == [{"id":"ok"}]
    assert p["matches"] == [{"relation":"one_to_one"}]


def test_normalize_review_json_drops_bool_rows_and_maps():
    overrides = normalize_overrides({
        "manual_effect_regions": [True, {"id":"r1"}],
        "manual_reletter": False,
        "text_overrides": True,
        "match_overrides": False,
        "unit_actions": True,
        "restore_target_bubbles": True,
        "accept_candidate_targets": [False, "a"],
    })
    assert overrides["manual_effect_regions"] == [{"id":"r1"}]
    assert overrides["manual_reletter"] == []
    assert overrides["text_overrides"] == {}
    assert overrides["restore_target_bubbles"] == []
    assert overrides["accept_candidate_targets"] == [False, "a"]

    audit = normalize_review_applied({"manual_effect_applied":[True,{"id":"r1","success":True}]})
    assert audit["manual_effect_applied"] == [{"id":"r1","success":True}]


def test_apply_manual_effect_survives_legacy_bool_fields_end_to_end(tmp_path: Path):
    src, tgt = _source_target()
    write_image(tmp_path / "source_original.png", src)
    write_image(tmp_path / "target_original.png", tgt)
    write_image(tmp_path / "final.png", tgt)
    pair = {
        "source_path": str(tmp_path / "source_original.png"),
        "target_path": str(tmp_path / "target_original.png"),
        "source_index":0, "target_index":0, "confidence":1.0, "score":1.0, "reasons":[],
    }
    save_json(tmp_path / "project.json", {
        "pair": pair,
        "registration": {"matrix": np.eye(3).tolist(), "method":"identity", "confidence":1.0},
        "artifacts": True,
        "meta": {
            "transfer_mode":"auto",
            "passthrough": True,
            "direct_patch": True,
            "mask_replace": False,
            "qa_summary": True,
        },
        "source_units": True,
        "target_units": False,
        "target_blocks": True,
        "target_bubbles": False,
        "matches": [True],
    })
    save_json(tmp_path / "review_overrides.json", {
        "status":"reviewed_with_manual_effect",
        "manual_effect_regions":[
            True,
            {
                "id":"legacy-bool-region",
                "mode":"effect_text",
                "target_bbox":[20,35,190,115],
                "diff_threshold":20,
                "edge_threshold":35,
                "expand_px":2,
                "auto_clear_target":True,
            },
        ],
        "manual_reletter": True,
        "text_overrides": False,
        "match_overrides": True,
        "unit_actions": False,
        "restore_target_bubbles": True,
        "accept_candidate_targets": False,
    })
    out = apply_review_page(tmp_path, PipelineConfig())
    result = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert result is not None
    assert result.shape == tgt.shape
    assert (tmp_path / "final.png").exists()
    assert (tmp_path / "review_applied.json").exists()


def test_workspace_resolver_never_exposes_bool_candidates_or_review_rows(tmp_path: Path):
    source = tmp_path / "source.png"; target = tmp_path / "target.png"
    write_image(source, np.full((20,20,3), 255, np.uint8))
    write_image(target, np.full((20,20,3), 255, np.uint8))
    pair = PagePair(str(source), str(target), 0, 0, 1.0, 1.0, [])
    page_id = page_id_for_pair(pair)
    root = tmp_path / "out" / "pages" / page_id
    root.mkdir(parents=True)
    write_image(root / "final.png", np.full((20,20,3),255,np.uint8))
    save_json(root / "project.json", {
        "pair": {"source_path":pair.source_path,"target_path":pair.target_path,"source_index":pair.source_index,"target_index":pair.target_index,"confidence":pair.confidence,"score":pair.score,"reasons":pair.reasons},
        "artifacts": True,
        "meta": {
            "direct_patch": {
                "used": True,
                "review_regions": [True, {"target_bubble_id":"r"}],
                "manual_effect_candidates": [False, {"reason":"x","target_bbox":[1,1,5,5]}],
                "diagnostics": True,
            },
            "qa_summary": True,
        },
    })
    ws = resolve_page_workspace(tmp_path / "out", pair)
    assert ws.review_regions == [{"target_bubble_id":"r"}]
    assert ws.manual_effect_candidates == [{"reason":"x","target_bbox":[1,1,5,5]}]
    assert isinstance(ws.qa_summary, dict)
