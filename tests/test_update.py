from manga_hd_transfer.update import UpdateInfo, _version_tuple


def test_update_version_comparison():
    assert _version_tuple("v0.8.7") > _version_tuple("0.8.6")
    assert _version_tuple("0.8.6") == _version_tuple("0.8.6")


def test_update_info_availability():
    # Keep this test independent of the package's current release number.
    assert UpdateInfo("99.0.0", "v99.0.0", "https://example.test/a.zip").available
    assert not UpdateInfo("0.0.0", "v0.0.0", "https://example.test/a.zip").available
