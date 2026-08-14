from __future__ import annotations

from pathlib import Path

from manga_hd_transfer.direct_containers import _append_manual_effect_candidate
from manga_hd_transfer.io_utils import save_json
from manga_hd_transfer.models import PagePair
from manga_hd_transfer.workspace import resolve_page_workspace


def test_manual_effect_candidate_dedup_prefers_tighter_text_seed() -> None:
    rows: list[dict] = []
    _append_manual_effect_candidate(
        rows,
        source_bbox=(100, 100, 250, 220),
        target_bbox=[80, 90, 210, 190],
        reason="colored_target_requires_reveal",
        source_hint=False,
        hint_backend="contour",
        compact_components=8,
        compact_ratio=.05,
        spiky=True,
    )
    _append_manual_effect_candidate(
        rows,
        source_bbox=(110, 110, 235, 210),
        target_bbox=[88, 96, 198, 184],
        reason="colored_target_requires_reveal",
        source_hint=True,
        hint_backend="pseudo_text_barrier",
        compact_components=4,
        compact_ratio=.03,
        spiky=True,
    )
    assert len(rows) == 1
    assert rows[0]["target_bbox"] == [88, 96, 198, 184]
    assert rows[0]["source_hint"] is True
    assert rows[0]["suggested_manual_mode"] == "reveal_text"


def test_workspace_reads_candidate_from_diagnostics_for_older_page_cache(tmp_path: Path) -> None:
    out = tmp_path / "out"
    page = out / "pages" / "jp"
    page.mkdir(parents=True)
    candidate = {
        "target_bbox": [10, 20, 70, 90],
        "source_bbox": [12, 18, 72, 88],
        "reason": "colored_target_requires_reveal",
        "workflow": "manual_effect",
        "suggested_manual_mode": "reveal_text",
    }
    save_json(page / "project.json", {
        "page_id": "jp",
        "meta": {
            "direct_patch": {
                "used": False,
                "diagnostics": {"manual_effect_candidates": [candidate]},
            },
            "mask_replace": {"used": True},
        },
        "artifacts": {},
    })
    pair = PagePair("cn.png", "jp.png", 0, 0, 1.0, 0.0, [])
    ws = resolve_page_workspace(out, pair)
    assert len(ws.manual_effect_candidates) == 1
    assert ws.manual_effect_candidates[0]["target_bbox"] == [10, 20, 70, 90]
