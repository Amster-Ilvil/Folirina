from manga_hd_transfer import __version__


def test_version_v0831_or_newer_keeps_v0831_contract():
    parts = tuple(int(x) for x in __version__.split('.')[:3])
    assert parts >= (0, 8, 31)
