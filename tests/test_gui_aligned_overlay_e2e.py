from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox

from manga_hd_transfer.gui_qt import StudioWindow


pytestmark = pytest.mark.gui_e2e


def _make_gui_pair(root: Path) -> tuple[Path, Path, Path]:
    """Build a registration-rich manga-like CN SOURCE / colour JP TARGET pair.

    This fixture intentionally exercises the same cross-rendition path as the real
    page: SOURCE is grayscale, TARGET has colour, panel geometry is shared, and the
    white dialogue containers contain different glyph geometry.
    """
    source_dir = root / "source_cn"
    target_dir = root / "target_jp"
    output_dir = root / "output"
    source_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    h, w = 760, 560
    source = np.full((h, w, 3), 245, np.uint8)
    target = np.full((h, w, 3), (220, 235, 250), np.uint8)

    # Shared page/panel/building structure gives the real registration stage enough
    # geometry; no registration function is mocked in this test.
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

    # White containers are re-drawn after the coloured TARGET background, so the
    # experimental route can use TARGET-owned near-white envelopes safely.
    for img in (source, target):
        cv2.ellipse(img, (150, 365), (96, 71), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (430, 620), (81, 66), 0, 0, 360, (255, 255, 255), -1)
        cv2.rectangle(img, (343, 88), (497, 217), (255, 255, 255), -1)
        cv2.ellipse(img, (150, 365), (100, 75), 0, 0, 360, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.ellipse(img, (430, 620), (85, 70), 0, 0, 360, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.rectangle(img, (340, 85), (500, 220), (25, 25, 25), 2)

    # SOURCE pseudo-Chinese strokes.
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

    # TARGET pseudo-Japanese strokes deliberately differ inside the same containers.
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

    # Shared fine detail improves SIFT/ORB repeatability on CI runners.
    rng = np.random.default_rng(7)
    for x, y in rng.integers([25, 25], [w - 25, h - 25], size=(120, 2)):
        if ((x - 150) ** 2 / 100**2 + (y - 365) ** 2 / 75**2 < 1.3):
            continue
        if ((x - 430) ** 2 / 85**2 + (y - 620) ** 2 / 70**2 < 1.3):
            continue
        if 335 < x < 505 and 80 < y < 225:
            continue
        cv2.circle(source, (int(x), int(y)), 1, (100, 100, 100), -1)
        cv2.circle(target, (int(x), int(y)), 1, (100, 100, 100), -1)

    source_path = source_dir / "006.png"
    target_path = target_dir / "006.png"
    assert cv2.imwrite(str(source_path), source)
    assert cv2.imwrite(str(target_path), target)
    return source_dir, target_dir, output_dir


def test_aligned_overlay_is_run_by_clicking_the_real_gui(qtbot, monkeypatch, tmp_path: Path):
    source_dir, target_dir, output_dir = _make_gui_pair(tmp_path)

    # Native Finder dialogs cannot be interacted with on Actions. Only the OS dialog
    # result is substituted; every application-side signal/slot remains real GUI code.
    selections = iter((str(source_dir), str(target_dir), str(output_dir)))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: next(selections))
    modal_errors: list[str] = []
    monkeypatch.setattr(QMessageBox, "critical", lambda _p, _t, text, *_a, **_k: modal_errors.append(str(text)) or 0)

    window = StudioWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    # 1) Use the actual directory chooser buttons.
    qtbot.mouseClick(window.project.source.button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(window.project.target.button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(window.project.output.button, Qt.MouseButton.LeftButton)
    assert Path(window.state.source_dir) == source_dir
    assert Path(window.state.target_dir) == target_dir
    assert Path(window.state.output_dir) == output_dir

    # 2) Use the actual pairing button. The generated pair has matching filenames.
    qtbot.mouseClick(window.project.pair_btn, Qt.MouseButton.LeftButton)
    assert len(window.state.pairs) == 1
    assert window.project.mode.currentData() == "auto"

    # 3) This is the v1.2.2 regression: clicking the checkbox must actually select
    #    the experimental route, not merely reveal a hidden combo-box item.
    qtbot.mouseClick(window.project.show_experimental, Qt.MouseButton.LeftButton)
    assert window.project.show_experimental.isChecked()
    assert window.project.mode.currentData() == "aligned_overlay_reveal"
    assert window.state.config.transfer.mode == "aligned_overlay_reveal"
    assert window.state.config.aligned_overlay_reveal.enabled is True
    assert window.project.experimental_warning.isVisible()

    # 4) Click the real "处理当前页" button. This creates and starts PipelineWorker
    #    through StudioWindow.run_current_page; this test never calls Pipeline directly.
    qtbot.mouseClick(window.project.run_page, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: bool(window.state.last_result_path)
        and window.worker is not None
        and not window.worker.isRunning(),
        timeout=90_000,
    )

    assert not modal_errors, "GUI worker failed: " + "\n".join(modal_errors)
    final_path = Path(window.state.last_result_path)
    assert final_path.exists()
    assert window.stack.currentIndex() == 2, "single-page completion should jump to 替换工作台"
    assert window.workbench.current_view == "result"

    project = window.state.last_project
    assert project is not None
    aligned = project.meta.get("aligned_overlay_reveal", {})
    assert project.meta.get("transfer_mode") == "aligned_overlay_reveal"
    assert aligned.get("used") is True
    assert aligned.get("accepted") is True
    assert int(aligned.get("diagnostics", {}).get("changed_pixels", 0)) > 500
    assert project.meta.get("direct_patch", {}).get("used") is False
    assert project.meta.get("mask_replace", {}).get("used") is False

    target = cv2.imread(str(target_dir / "006.png"), cv2.IMREAD_COLOR)
    final = cv2.imread(str(final_path), cv2.IMREAD_COLOR)
    assert target is not None and final is not None and target.shape == final.shape
    changed = np.any(final != target, axis=2)
    assert int(np.count_nonzero(changed)) > 500, "GUI result remained effectively all TARGET/Japanese"

    # Coloured TARGET background is still authoritative. Outside near-white regions,
    # the experiment must not paste grayscale SOURCE background into the page.
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    coloured = hsv[..., 1] >= 20
    assert np.array_equal(final[coloured], target[coloured])

    # Preserve CI evidence even when the test later gains stricter assertions.
    artifact_dir = Path(os.environ.get("GUI_E2E_ARTIFACT_DIR", str(tmp_path / "gui-artifacts")))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(artifact_dir / "window-after-processing.png"))
    shutil.copy2(final_path, artifact_dir / "final.png")
    page_root = Path(project.artifacts.get("project_json", final_path.parent / "project.json")).parent
    for name in ("project.json", "aligned_overlay_reveal.json", "aligned_overlay_reveal_mask.png", "aligned_overlay_reveal_regions.png"):
        candidate = page_root / name
        if candidate.exists():
            shutil.copy2(candidate, artifact_dir / name)
