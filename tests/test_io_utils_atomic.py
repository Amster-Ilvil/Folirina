from __future__ import annotations

import json

import pytest

from manga_hd_transfer import io_utils


def test_save_json_atomic_round_trip_unicode(tmp_path):
    path = tmp_path / "page_management.json"
    payload = {"页面": "扉页", "nested": {"类型": "目录"}, "count": 3}
    io_utils.save_json(path, payload)
    assert io_utils.load_json(path) == payload
    assert not list(tmp_path.glob(".page_management.json.*.tmp"))


def test_save_json_replace_failure_keeps_previous_state_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "page_management.json"
    old = {"page": "cover", "origin": "manual"}
    path.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")

    def fail_replace(_src, _dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(io_utils.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        io_utils.save_json(path, {"page": "content"})

    assert io_utils.load_json(path) == old
    assert not list(tmp_path.glob(".page_management.json.*.tmp"))
