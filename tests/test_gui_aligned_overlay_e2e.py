from __future__ import annotations

import base64
import hashlib
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox

from manga_hd_transfer.gui_qt import StudioWindow


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "real_page_006"
REAL_SOURCE_SHA256 = "1783c6533b8f4a22413ccf1fc70fcd8c01ac934246d7c793bcc0683938f535b1"
REAL_TARGET_SHA256 = "8c6f403b0efde369a4e4db31ee548576ba5b6bb5a3250545689f2af7db3eeb3f"


def _make_gui_pair(root: Path) -> tuple[Path, Path, Path]:
    """Create a deterministic cross-rendition manga-like pair for GUI smoke coverage."""
    source_dir = root / "source_cn"
    target_dir = root / "target_jp"
    output_dir = root / "output"
    source_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    h, w = 760, 560
    source = np.full((h, w, 3), 245, np.uint8)
    target = np.full((h, w, 3), (235, 240, 245), np.uint8)

    for img in (source, target):
        cv2.rectangle(img, (15, 15), (w - 15, h - 15), (25, 25, 25), 4)
        for y in (250, 500):
            cv2.line(img, (20, y), (w - 20, y), (35, 35, 35), 3)
        cv2.line(img, (300, 250), (300, h - 20), (35, 35, 35), 3)
        for x in range(50, 500, 75):
            cv2.rectangle(img, (x, 55), (x + 42, 125), (50, 50, 50), 2)
            cv2.line(img, (x, 125), (x + 42, 55), (80, 80, 80), 1)
        for y in range(150, 230, 16):
            cv2.line(img, (40, y), (520, y), (90, 90, 90), 1)

        rng = np.random.default_rng(123)
        for i in range(70):
            x = int(rng.integers(30, w - 30))
            y = int(rng.integers(30, h - 30))
            if ((x - 150) ** 2 / 110**2 + (y - 365) ** 2 / 85**2 < 1.4):
                continue
            if ((x - 430) ** 2 / 95**2 + (y - 620) ** 2 / 80**2 < 1.4):
                continue
            if 330 < x < 510 and 75 < y < 230:
                continue
            r = 2 + (i % 5)
            angle = math.radians((i * 37) % 180)
            dx = int(round(r * 3 * math.cos(angle)))
            dy = int(round(r * 3 * math.sin(angle)))
            shade = 60 + (i % 3) * 15
            cv2.circle(img, (x, y), r, (shade, shade, shade), 1)
            cv2.line(img, (x - dx, y - dy), (x + dx, y + dy), (55, 55, 55), 1)
            cv2.putText(
                img, str(i % 10), (x + 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.22, (70, 70, 70), 1, cv2.LINE_AA,
            )

        cv2.ellipse(img, (150, 365), (96, 71), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (430, 620), (81, 66), 0, 0, 360, (255, 255, 255), -1)
        cv2.rectangle(img, (343, 88), (497, 217), (255, 255, 255), -1)
        cv2.ellipse(img, (150, 365), (100, 75), 0, 0, 360, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.ellipse(img, (430, 620), (85, 70), 0, 0, 360, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.rectangle(img, (340, 85), (500, 220), (25, 25, 25), 2)

    cv2.rectangle(target, (305, 505), (545, 745), (160, 215, 245), -1)
    cv2.line(target, (300, 500), (300, h - 20), (35, 35, 35), 3)
    cv2.rectangle(target, (15, 15), (w - 15, h - 15), (25, 25, 25), 4)

    for x in (120, 150, 180):
        cv2.line(source, (x, 330), (x, 395), (10, 10, 10), 4, cv2.LINE_AA)
        cv2.line(source, (x - 12, 350), (x + 12, 350), (10, 10, 10), 4, cv2.LINE_AA)
        cv2.line(source, (x - 10, 378), (x + 10, 378), (10, 10, 10), 3, cv2.LINE_AA)
    for x in (405, 430, 455):
        cv2.line(source, (x, 590), (x, 650), (10, 10, 10), 4, cv2.LINE_AA)
        cv2.line(source, (x - 10, 615), (x + 10, 615), (10, 10, 10), 4, cv2.LINE_AA)
    for x in (380, 415, 450):
        cv2.line(source, (x, 115), (x, 190), (10, 10, 10), 5, cv2.LINE_AA)
        cv2.line(source, (x - 12, 145), (x + 12, 145), (10, 10, 10), 4, cv2.LINE_AA)

    for x in (125, 155, 185):
        cv2.line(target, (x, 325), (x, 398), (10, 10, 10), 4, cv2.LINE_AA)
        cv2.line(target, (x - 12, 340), (x + 12, 340), (10, 10, 10), 4, cv2.LINE_AA)
        cv2.line(target, (x - 10, 370), (x + 10, 370), (10, 10, 10), 3, cv2.LINE_AA)
    for x in (410, 435, 460):
        cv2.line(target, (x, 585), (x, 654), (10, 10, 10), 4, cv2.LINE_AA)
        cv2.line(target, (x - 10, 602), (x + 10, 602), (10, 10, 10), 4, cv2.LINE_AA)
    for x in (385, 420, 455):
        cv2.line(target, (x, 110), (x, 195), (10, 10, 10), 5, cv2.LINE_AA)
        cv2.line(target, (x - 12, 130), (x + 12, 130), (10, 10, 10), 4, cv2.LINE_AA)

    source_path = source_dir / "006.png"
    target_path = target_dir / "006.png"
    assert cv2.imwrite(str(source_path), source)
    assert cv2.imwrite(str(target_path), target)
    return source_dir, target_dir, output_dir


def _restore_fixture(prefix: str, output: Path, expected_sha256: str) -> None:
    parts = sorted(FIXTURE_ROOT.glob(f"{prefix}.b64.part*"))
    assert parts, f"missing fixture chunks for {prefix}"
    payload = "".join(p.read_text(encoding="ascii").strip() for p in parts)
    raw = base64.b64decode(payload, validate=True)
    assert hashlib.sha256(raw).hexdigest() == expected_sha256
    output.write_bytes(raw)


def _make_real_page_pair(root: Path) -> tuple[Path, Path, Path]:
    """Restore a downsampled copy of the user's actual CN/JP page pair.

    The originals are 1440x2048 CN B/W and 1117x1600 colour JP.  The committed
    regression copies preserve the actual page content/layout at ~240 px width so
    GUI CI remains fast while registration/mask behaviour is still exercised.
    """
    source_dir = root / "real_source_cn"
    target_dir = root / "real_target_jp"
    output_dir = root / "real_output"
    source_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    _restore_fixture("source_cn", source_dir / "006.jpg", REAL_SOURCE_SHA256)
    _restore_fixture("target_jp", target_dir / "006.jpg", REAL_TARGET_SHA256)
    return source_dir, target_dir, output_dir


def _save_evidence(window: StudioWindow, project, final_path: Path, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(artifact_dir / "window-after-processing.png"))
    if final_path.exists():
        shutil.copy2(final_path, artifact_dir / "final.png")
    if project is None:
        return
    page_root = Path(project.artifacts.get("project_json", final_path.parent / "project.json")).parent
    for name in (
        "project.json", "aligned_overlay_reveal.json", "aligned_overlay_reveal_mask.png",
        "aligned_overlay_reveal_regions.png", "aligned_overlay_reveal_layer.png",
    ):
        candidate = page_root / name
        if candidate.exists():
            shutil.copy2(candidate, artifact_dir / name)


def _run_pair_through_real_gui(qtbot, monkeypatch, source_dir: Path, target_dir: Path, output_dir: Path, artifact_dir: Path):
    """Use only StudioWindow controls after substituting the native Finder result."""
    selections = iter((str(source_dir), str(target_dir), str(output_dir)))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: next(selections))
    modal_errors: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "critical",
        lambda _p, _t, text, *_a, **_k: modal_errors.append(str(text)) or 0,
    )

    window = StudioWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    qtbot.mouseClick(window.project.source.button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(window.project.target.button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(window.project.output.button, Qt.MouseButton.LeftButton)
    assert Path(window.state.source_dir) == source_dir
    assert Path(window.state.target_dir) == target_dir
    assert Path(window.state.output_dir) == output_dir

    qtbot.mouseClick(window.project.pair_btn, Qt.MouseButton.LeftButton)
    assert len(window.state.pairs) == 1
    assert window.project.mode.currentData() == "auto"

    qtbot.mouseClick(window.project.show_experimental, Qt.MouseButton.LeftButton)
    assert window.project.show_experimental.isChecked()
    assert window.project.mode.currentData() == "aligned_overlay_reveal"
    assert window.state.config.transfer.mode == "aligned_overlay_reveal"
    assert window.state.config.aligned_overlay_reveal.enabled is True
    assert window.project.experimental_warning.isVisible()

    # Critical requirement: invoke the application only through its real GUI button.
    qtbot.mouseClick(window.project.run_page, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: bool(window.state.last_result_path)
        and window.worker is not None
        and not window.worker.isRunning(),
        timeout=120_000,
    )

    final_path = Path(window.state.last_result_path)
    project = window.state.last_project
    _save_evidence(window, project, final_path, artifact_dir)
    return window, project, final_path, modal_errors


def _assert_aligned_gui_result(window: StudioWindow, project, final_path: Path, modal_errors: list[str], *, min_changed: int) -> dict:
    assert not modal_errors, "GUI worker failed: " + "\n".join(modal_errors)
    assert final_path.exists()
    assert window.stack.currentIndex() == 2, "single-page completion should jump to 替换工作台"
    assert window.workbench.current_view == "result"
    assert project is not None

    aligned = project.meta.get("aligned_overlay_reveal", {})
    assert project.meta.get("transfer_mode") == "aligned_overlay_reveal"
    assert aligned.get("used") is True, aligned
    assert aligned.get("accepted") is True, aligned
    assert int(aligned.get("diagnostics", {}).get("changed_pixels", 0)) > min_changed, aligned
    assert project.meta.get("direct_patch", {}).get("used") is False
    assert project.meta.get("mask_replace", {}).get("used") is False
    return aligned


def test_aligned_overlay_is_run_by_clicking_the_real_gui(qtbot, monkeypatch, tmp_path: Path):
    source_dir, target_dir, output_dir = _make_gui_pair(tmp_path)
    artifact_root = Path(os.environ.get("GUI_E2E_ARTIFACT_DIR", str(tmp_path / "gui-artifacts")))
    window, project, final_path, errors = _run_pair_through_real_gui(
        qtbot, monkeypatch, source_dir, target_dir, output_dir, artifact_root / "synthetic",
    )
    _assert_aligned_gui_result(window, project, final_path, errors, min_changed=500)

    target = cv2.imread(str(target_dir / "006.png"), cv2.IMREAD_COLOR)
    final = cv2.imread(str(final_path), cv2.IMREAD_COLOR)
    assert target is not None and final is not None and target.shape == final.shape
    assert int(np.count_nonzero(np.any(final != target, axis=2))) > 500
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    coloured = hsv[..., 1] >= 20
    assert np.array_equal(final[coloured], target[coloured])


def test_real_uploaded_page_is_processed_by_clicking_the_gui(qtbot, monkeypatch, tmp_path: Path):
    """Regression for the user's page that previously stayed entirely Japanese."""
    source_dir, target_dir, output_dir = _make_real_page_pair(tmp_path)
    artifact_root = Path(os.environ.get("GUI_E2E_ARTIFACT_DIR", str(tmp_path / "gui-artifacts")))
    window, project, final_path, errors = _run_pair_through_real_gui(
        qtbot, monkeypatch, source_dir, target_dir, output_dir, artifact_root / "real-page-006",
    )
    aligned = _assert_aligned_gui_result(window, project, final_path, errors, min_changed=200)

    diagnostics = aligned.get("diagnostics", {})
    assert diagnostics.get("registration_gate_passed") is True, diagnostics
    assert int(diagnostics.get("applied_count", 0)) > 0, diagnostics
    assert diagnostics.get("nearly_unchanged") is False, diagnostics

    target = cv2.imread(str(target_dir / "006.jpg"), cv2.IMREAD_COLOR)
    final = cv2.imread(str(final_path), cv2.IMREAD_COLOR)
    assert target is not None and final is not None and target.shape == final.shape
    changed = np.any(final != target, axis=2)
    assert int(np.count_nonzero(changed)) > 200, "real-page GUI result remained effectively all Japanese TARGET"

    # The source is B/W while this target is colour.  Any saturated TARGET pixel is
    # background authority and must remain byte-for-byte unchanged after reveal.
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    coloured = hsv[..., 1] >= 20
    assert int(np.count_nonzero(coloured)) > 100
    assert np.array_equal(final[coloured], target[coloured])
