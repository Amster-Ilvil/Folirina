from manga_hd_transfer import __version__


def test_version_v0829():
    current = tuple(int(x) for x in __version__.split("."))
    assert current >= (0, 8, 29)
