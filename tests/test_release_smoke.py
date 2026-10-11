from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from manga_hd_transfer import __version__
from manga_hd_transfer.modes.registry import (
    ACTIVE_MODE_ORDER,
    LEGACY_MODE_ORDER,
    active_mode_ui_items,
    get_mode_spec,
)

ROOT = Path(__file__).resolve().parents[1]


def test_package_and_project_version_agree():
    with (ROOT / "pyproject.toml").open("rb") as f:
        metadata = tomllib.load(f)
    assert __version__ == metadata["project"]["version"]
    assert __version__.startswith("3.")


def test_six_active_modes_are_unique_and_available():
    assert len(ACTIVE_MODE_ORDER) == len(set(ACTIVE_MODE_ORDER)) == 6
    assert "auto" not in ACTIVE_MODE_ORDER
    assert "auto" in LEGACY_MODE_ORDER
    assert [key for _, key in active_mode_ui_items()] == list(ACTIVE_MODE_ORDER)
    for name in ACTIVE_MODE_ORDER:
        assert get_mode_spec(name).key == name
    with pytest.raises(ValueError):
        get_mode_spec("mode-that-does-not-exist")


def test_desktop_entrypoints_and_platform_icons_present():
    assert (ROOT / "run_gui.py").is_file()
    assert (ROOT / "scripts" / "build_desktop.py").is_file()
    for filename in ("icon.ico", "icon.icns", "icon.png"):
        assert (ROOT / "assets" / filename).stat().st_size > 0


def test_release_documentation_includes_platforms_and_star_history():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    notes = (ROOT / "RELEASE_NOTES.md").read_text(encoding="utf-8")
    assert "api.star-history.com/svg?repos=Amster-Ilvil/Folirina" in readme
    assert "GitHub Releases" in readme
    for token in ("Windows", "Linux", "Apple Silicon", "Intel"):
        assert token in readme
        assert token in notes
