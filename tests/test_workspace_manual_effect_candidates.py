from __future__ import annotations

from pathlib import Path

import numpy as np

from manga_hd_transfer.aligned_overlay_reveal import (
    AlignedOverlayPlan,
    AlignedOverlayRegion,
    _sync_manual_candidate_actionability,
)
from manga_hd_transfer.io_utils import save_json
from manga_hd_transfer.models import PagePair
from manga_hd_transfer.workspace import resolve_page_workspace


def test_workspace_exposes_direct_manual_effect_candidates(tmp_path: Path) -> None:
    out = tmp_path / "out"
    page_root = out / "pages" / "jp_page"
    page_root.mkdir(parents=True)
    pair = PagePair("src.png", str(tmp_path / "jp_page.png"), 0, 0, 0.99, 0.01, [])
    candidate = {
        "target_bbox": [10, 20, 70, 90],
        "source_bbox": [8, 18, 68, 88],
        "reason": "colored_text_needs_manual_reveal",
        "workflow": "manual_effect",
        "suggested_manual_mode": "reveal_text",
    }
    save_json(page_root / "project.json", {
        "meta": {
            "direct_patch": {
                "used": True,
                "manual_effect_candidates": [candidate],
                "review_regions": [],
            }
        }
    })
    ws = resolve_page_workspace(out, pair)
    assert len(ws.manual_effect_candidates) == 1
    assert ws.manual_effect_candidates[0]["target_bbox"] == [10, 20, 70, 90]
    assert ws.manual_effect_candidates[0]["workflow"] == "manual_effect"


def test_aligned_artwork_risk_candidates_are_not_auto_prefilled() -> None:
    """Diagnostics retain risky rows, while GUI auto-prefill sees only safe ones."""
    shape = (40, 50)
    z = np.zeros(shape, np.uint8)

    def region(rid: str, triage: str) -> AlignedOverlayRegion:
        return AlignedOverlayRegion(
            id=rid,
            target_bbox=(2, 2, 20, 20),
            source_bbox=(2, 2, 20, 20),
            erase_mask=z.copy(),
            source_ink_mask=z.copy(),
            full_raster_mask=z.copy(),
            composite_mode="ink_only",
            triage=triage,
            reason="synthetic",
            white_ratio=0.2,
            color_ratio=0.2,
            erase_area_ratio=0.0,
            source_ink_pixels=30,
            target_ink_pixels=30,
            border_guard_px=2,
        )

    rows = [
        {"id": "rejected_art", "auto_actionable": True},
        {"id": "withheld_face", "auto_actionable": True},
        {"id": "real_text", "auto_actionable": True},
    ]
    plan = AlignedOverlayPlan(
        accepted=True,
        reason="ok",
        aligned_source=np.zeros((40, 50, 3), np.uint8),
        valid_mask=np.full(shape, 255, np.uint8),
        erase_mask=z.copy(),
        source_ink_mask=z.copy(),
        full_raster_mask=z.copy(),
        regions=[region("rejected_art", "REJECT"), region("withheld_face", "REVIEW"), region("real_text", "REVIEW")],
        diagnostics={"manual_effect_candidates": rows},
    )

    _sync_manual_candidate_actionability(plan, ["real_text"], ["withheld_face"])
    by_id = {row["id"]: row for row in rows}
    assert by_id["rejected_art"]["auto_actionable"] is False
    assert by_id["rejected_art"]["manual_prefill_safety_gate"] == "route_rejected"
    assert by_id["withheld_face"]["auto_actionable"] is False
    assert by_id["withheld_face"]["manual_prefill_safety_gate"] == "colour_artwork_risk"
    assert by_id["real_text"]["auto_actionable"] is True
    assert by_id["real_text"]["manual_prefill_safety_gate"] == "evidence_gated_text"
    assert plan.diagnostics["manual_effect_auto_actionable_ids"] == ["real_text"]
    assert plan.diagnostics["manual_effect_safety_withheld_ids"] == ["rejected_art", "withheld_face"]


def test_workspace_uses_newer_automatic_final_instead_of_stale_review(tmp_path: Path) -> None:
    import os, time
    import numpy as np, cv2
    out=tmp_path/"out"
    page_root=out/"pages"/"jp_page"
    page_root.mkdir(parents=True)
    pair=PagePair("src.png",str(tmp_path/"jp_page.png"),0,0,.99,.01,[])
    save_json(page_root/"project.json",{"meta":{}})
    cv2.imwrite(str(page_root/"final_reviewed.png"),np.full((8,8,3),10,np.uint8))
    cv2.imwrite(str(page_root/"final.png"),np.full((8,8,3),240,np.uint8))
    now=time.time()
    os.utime(page_root/"final_reviewed.png",(now-20,now-20))
    os.utime(page_root/"final.png",(now,now))
    ws=resolve_page_workspace(out,pair)
    assert Path(ws.result_path).name == "final.png"


def test_workspace_uses_newer_review_when_it_is_current(tmp_path: Path) -> None:
    import os, time
    import numpy as np, cv2
    out=tmp_path/"out"
    page_root=out/"pages"/"jp_page"
    page_root.mkdir(parents=True)
    pair=PagePair("src.png",str(tmp_path/"jp_page.png"),0,0,.99,.01,[])
    save_json(page_root/"project.json",{"meta":{}})
    cv2.imwrite(str(page_root/"final.png"),np.full((8,8,3),240,np.uint8))
    cv2.imwrite(str(page_root/"final_reviewed.png"),np.full((8,8,3),10,np.uint8))
    now=time.time()
    os.utime(page_root/"final.png",(now-20,now-20))
    os.utime(page_root/"final_reviewed.png",(now,now))
    ws=resolve_page_workspace(out,pair)
    assert Path(ws.result_path).name == "final_reviewed.png"
