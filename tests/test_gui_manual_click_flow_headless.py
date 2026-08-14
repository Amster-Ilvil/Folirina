from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import cv2
import numpy as np

from manga_hd_transfer.manual_effect import build_manual_effect_masks, build_reveal_seed_mask
from manga_hd_transfer.io_utils import write_image

GUI = Path(__file__).resolve().parents[1] / "src" / "manga_hd_transfer" / "gui_qt.py"
SOURCE = GUI.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _method_source(class_name: str, method_name: str) -> str:
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    return textwrap.dedent(ast.get_source_segment(SOURCE, child) or "")
    raise AssertionError(f"missing {class_name}.{method_name}")


class _DialogCode:
    Accepted = 1


class _QDialog:
    DialogCode = _DialogCode


class _MessageBox:
    @staticmethod
    def information(*args, **kwargs):
        raise AssertionError("unexpected information dialog")

    @staticmethod
    def warning(*args, **kwargs):
        raise AssertionError("unexpected warning dialog")

    @staticmethod
    def critical(*args, **kwargs):
        raise AssertionError("unexpected critical dialog")


class _Value:
    def __init__(self, value): self._value = value
    def value(self): return self._value
    def currentData(self): return self._value


class _TargetView:
    def __init__(self, box): self._box = list(box)
    def box(self): return list(self._box)


class _RevealDialog:
    def __init__(self, target_path, aligned_source, source_mask, target_clear_mask, seed, parent):
        self._mask = seed.copy()
        self._patch = np.zeros((*seed.shape, 4), np.uint8)
        # Simulate the exact visible editor commit: black Chinese on any SOURCE
        # text pixels selected by the seed.  Alpha remains sparse.
        sel = (source_mask > 0) & (seed > 0)
        self._patch[sel, :3] = 0
        self._patch[sel, 3] = 255
    def exec(self): return _DialogCode.Accepted
    def result_mask(self): return self._mask.copy()
    def result_patch_bgra(self): return self._patch.copy()


def _identity_project(src: Path, tgt: Path):
    return {
        "pair": {"source_path": str(src), "target_path": str(tgt), "source_index": 0, "target_index": 0, "confidence": 1.0, "score": 1.0, "reasons": []},
        "registration": {"matrix": np.eye(3).tolist(), "method": "identity", "confidence": 1.0},
    }


def test_exact_manual_dialog_click_handler_commits_on_reveal_save(tmp_path: Path):
    source = np.full((160, 220, 3), (170, 80, 150), np.uint8)
    target = source.copy()
    cv2.putText(source, "CN", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (5, 5, 5), 3, cv2.LINE_AA)
    cv2.putText(target, "JP", (105, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (5, 5, 5), 3, cv2.LINE_AA)
    sp = tmp_path / "source.png"; tp = tmp_path / "target.png"
    write_image(sp, source); write_image(tp, target)

    traces = []
    committed = []

    class Self:
        source_path = sp
        target_path = tp
        project = _identity_project(sp, tp)
        target_view = _TargetView([15, 30, 190, 120])
        mode = _Value("reveal_text")
        expand = _Value(2)
        _reveal_mask = None
        _reveal_patch = None
        _committed_directly = False

        def _row_payload(self):
            return {
                "id": "gui-flow-real-handler",
                "mode": "reveal_text",
                "target_bbox": self.target_view.box(),
                "diff_threshold": 20,
                "edge_threshold": 35.0,
                "expand_px": 2,
                "feather_px": 0,
                "auto_clear_target": True,
                "source_offset_x": 0,
                "source_offset_y": 0,
            }
        def _trace(self, stage, **payload): traces.append(stage)
        def result_row(self): return self._row_payload()
        def result_reveal_mask(self): return self._reveal_mask
        def result_reveal_patch(self): return self._reveal_patch
        def done(self, code): self.done_code = code
        def accept(self): self.done_code = _DialogCode.Accepted
        def _commit_handler(self, row, reveal, patch):
            committed.append((row, reveal.copy(), patch.copy()))

    namespace = {
        "QMessageBox": _MessageBox,
        "QDialog": _QDialog,
        "RevealMaskDialog": _RevealDialog,
        "build_manual_effect_masks": build_manual_effect_masks,
        "build_reveal_seed_mask": build_reveal_seed_mask,
    }
    exec(_method_source("ManualEffectDialog", "_accept_checked"), namespace)
    handler = namespace["_accept_checked"]
    obj = Self()
    handler(obj)

    assert obj.done_code == _DialogCode.Accepted
    assert obj._committed_directly is True
    assert len(committed) == 1
    assert cv2.countNonZero(committed[0][1]) > 0
    assert cv2.countNonZero(committed[0][2][:, :, 3]) > 0
    assert traces.index("reveal_editor_saved") < traces.index("direct_commit_started") < traces.index("direct_commit_succeeded")
