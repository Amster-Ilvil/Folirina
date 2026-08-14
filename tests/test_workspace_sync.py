from pathlib import Path

from manga_hd_transfer.io_utils import save_json
from manga_hd_transfer.models import PagePair
from manga_hd_transfer.workspace import page_id_for_pair, resolve_page_workspace


def _pair(tmp_path: Path, number: int) -> PagePair:
    src = tmp_path / "cn" / f"cn-{number:03d}.jpg"
    tgt = tmp_path / "jp" / f"jp-{number:03d}.png"
    src.parent.mkdir(exist_ok=True)
    tgt.parent.mkdir(exist_ok=True)
    src.write_bytes(b"source")
    tgt.write_bytes(b"target")
    return PagePair(str(src), str(tgt), number - 1, number - 1, 1.0, 0.0, [])


def _build_page(out: Path, pair: PagePair, marker: bytes):
    page_id = page_id_for_pair(pair)
    root = out / "pages" / page_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "final.png").write_bytes(marker + b"-final")
    (root / "review_preview.png").write_bytes(marker + b"-review")
    (root / "mask_transfer_mask.png").write_bytes(marker + b"-mask")
    save_json(
        root / "project.json",
        {
            "page_id": page_id,
            "artifacts": {
                "final": str(root / "final.png"),
                "review_preview": str(root / "review_preview.png"),
                "mask_transfer_mask": str(root / "mask_transfer_mask.png"),
            },
            "meta": {
                "qa_summary": {"errors": 0, "warnings": 1, "pass": True},
                "mask_replace": {"review_regions": [{"target_bubble_id": f"bubble-{page_id}"}]},
            },
        },
    )
    return root


def test_workspace_views_are_page_scoped_and_never_reuse_previous_result(tmp_path):
    out = tmp_path / "out"
    p1 = _pair(tmp_path, 1)
    p2 = _pair(tmp_path, 2)
    r1 = _build_page(out, p1, b"one")
    r2 = _build_page(out, p2, b"two")

    w1 = resolve_page_workspace(out, p1)
    w2 = resolve_page_workspace(out, p2)

    assert w1.page_id != w2.page_id
    assert Path(w1.result_path) == r1 / "final.png"
    assert Path(w2.result_path) == r2 / "final.png"
    assert Path(w1.review_path) == r1 / "review_preview.png"
    assert Path(w2.mask_path) == r2 / "mask_transfer_mask.png"
    assert w1.review_regions[0]["target_bubble_id"].endswith(w1.page_id)
    assert w2.review_regions[0]["target_bubble_id"].endswith(w2.page_id)


def test_workspace_does_not_fall_back_to_other_page_when_current_result_missing(tmp_path):
    out = tmp_path / "out"
    p1 = _pair(tmp_path, 1)
    p2 = _pair(tmp_path, 2)
    _build_page(out, p1, b"one")

    w2 = resolve_page_workspace(out, p2)
    assert w2.result_path == ""
    assert w2.review_path == ""
    assert w2.mask_path == ""


def test_reviewed_result_has_priority_for_same_page(tmp_path):
    out = tmp_path / "out"
    pair = _pair(tmp_path, 3)
    root = _build_page(out, pair, b"three")
    reviewed = root / "final_reviewed.png"
    reviewed.write_bytes(b"reviewed")

    ws = resolve_page_workspace(out, pair)
    assert Path(ws.result_path) == reviewed


def test_qa_json_can_refresh_summary_without_mutating_project(tmp_path):
    out = tmp_path / "out"
    pair = _pair(tmp_path, 4)
    root = _build_page(out, pair, b"four")
    save_json(root / "qa.json", {"summary": {"errors": 2, "warnings": 3, "pass": False}, "issues": []})

    ws = resolve_page_workspace(out, pair)
    assert ws.qa_summary == {"errors": 2, "warnings": 3, "pass": False}


def test_page_id_preserves_cjk_names_without_collapsing_to_page(tmp_path):
    src = tmp_path / "cn" / "中文页一.jpg"
    tgt1 = tmp_path / "jp" / "第一話.jpg"
    tgt2 = tmp_path / "jp" / "第二話.jpg"
    p1 = PagePair(str(src), str(tgt1), 0, 0, 1.0, 0.0, [])
    p2 = PagePair(str(src), str(tgt2), 0, 1, 1.0, 0.0, [])
    assert page_id_for_pair(p1) == "第一話"
    assert page_id_for_pair(p2) == "第二話"
    assert page_id_for_pair(p1) != page_id_for_pair(p2)


def test_workspace_can_read_v0816_legacy_single_page_folder(tmp_path):
    out = tmp_path / "out"
    src = tmp_path / "cn" / "src.jpg"
    tgt = tmp_path / "jp" / "第 一 話.jpg"
    pair = PagePair(str(src), str(tgt), 0, 0, 1.0, 0.0, [])
    legacy = out / "pages" / Path(pair.target_path).stem
    legacy.mkdir(parents=True)
    (legacy / "final.png").write_bytes(b"legacy")
    save_json(legacy / "project.json", {"page_id": page_id_for_pair(pair), "artifacts": {}, "meta": {}})
    ws = resolve_page_workspace(out, pair)
    assert ws.page_root == legacy
    assert Path(ws.result_path) == legacy / "final.png"


def test_workspace_can_read_v0816_ascii_batch_folder(tmp_path):
    out = tmp_path / "out"
    src = tmp_path / "cn" / "src.jpg"
    tgt = tmp_path / "jp" / "第一話.jpg"
    pair = PagePair(str(src), str(tgt), 0, 0, 1.0, 0.0, [])
    legacy = out / "pages" / "page"
    legacy.mkdir(parents=True)
    (legacy / "final.png").write_bytes(b"legacy-batch")
    save_json(legacy / "project.json", {"page_id": "page", "artifacts": {}, "meta": {}})
    ws = resolve_page_workspace(out, pair)
    assert ws.page_id == "第一話"
    assert ws.page_root == legacy
    assert Path(ws.result_path) == legacy / "final.png"
