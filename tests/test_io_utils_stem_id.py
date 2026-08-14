from manga_hd_transfer.io_utils import stem_id


def test_stem_id_preserves_cjk_and_sanitizes_unsafe_spacing():
    assert stem_id("/tmp/第一話 01?.png") == "第一話_01"
    assert stem_id("/tmp/p-007(5).jpeg") == "p-007_5"
