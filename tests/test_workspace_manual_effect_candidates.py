from __future__ import annotations

from pathlib import Path

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
