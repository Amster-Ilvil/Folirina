from __future__ import annotations

import os
import cv2
from pathlib import Path

import numpy as np

from .cache import page_job_fingerprint
from .config import PipelineConfig
from .storage_clone import publish_independent_png
from .io_utils import read_image, save_json, stem_id, write_image
from .mode_contracts import get_mode_contract, mode_artifact_violations
from .models import PagePair, PageProject, QAItem, RegistrationResult
from .page_management import PageMark
from .qa import qa_summary


def _replace_with_hardlink(alias: Path, target: Path) -> bool:
    """Make a lossless same-filesystem alias without duplicating page bytes."""
    try:
        alias.unlink(missing_ok=True)
        os.link(target, alias)
        return True
    except OSError:
        return False


def emit_passthrough_page(
    config: PipelineConfig,
    pair: PagePair,
    page_root: str | Path,
    final_path: str | Path | None,
    mark: PageMark,
    *,
    source: np.ndarray | None = None,
    target: np.ndarray | None = None,
    registration: RegistrationResult | None = None,
    passthrough_reason: str = "page_manager_exclusion",
    extra_meta: dict | None = None,
    qa: list[QAItem] | None = None,
) -> PageProject:
    """Persist an unchanged HD target page as a first-class PageProject."""
    page_root = Path(page_root)
    page_root.mkdir(parents=True, exist_ok=True)
    if target is None:
        target = read_image(pair.target_path)
    if source is None:
        # Manual exclusions do not need source pixels. Use target dimensions as
        # a harmless identity-registration placeholder and avoid extra I/O.
        source_size = (int(target.shape[1]), int(target.shape[0]))
    else:
        source_size = (int(source.shape[1]), int(source.shape[0]))
    target_size = (int(target.shape[1]), int(target.shape[0]))
    if registration is None:
        registration = RegistrationResult(
            matrix=np.eye(3, dtype=np.float64),
            method="page_manager_passthrough",
            confidence=1.0, inlier_ratio=1.0, reprojection_error=0.0,
            spatial_coverage=1.0, num_matches=0,
            source_size=source_size, target_size=target_size,
            diagnostics={"page_management": mark.to_dict()},
        )
    final = Path(final_path) if final_path is not None else page_root / "final.png"
    final.parent.mkdir(parents=True, exist_ok=True)
    write_image(final, target)
    # A passthrough content page can still need manual omission repair later.
    # Preserve the pair inside the page workspace when SOURCE pixels are
    # available so Codex/CLI results can be restored in the GUI without
    # rerunning registration/OCR. Keep the authority alias lossless and
    # non-duplicating on the common primary-source path.
    passthrough_artifacts = {"final": str(final), "book_final": str(final)}
    if source is not None:
        source_original = page_root / "source_original.png"
        source_authority_original = page_root / "source_authority_original.png"
        target_original = page_root / "target_original.png"
        persistent_level = int(max(0, min(9, getattr(config.export, "persistent_png_compression", 4))))
        persistent_png = [cv2.IMWRITE_PNG_COMPRESSION, persistent_level]
        source_method = publish_independent_png(pair.source_path, source_original) if bool(getattr(config.export, "prefer_input_reflink", True)) else None
        target_method = publish_independent_png(pair.target_path, target_original) if bool(getattr(config.export, "prefer_input_reflink", True)) else None
        if source_method is None:
            write_image(source_original, source, params=persistent_png)
        if target_method is None:
            write_image(target_original, target, params=persistent_png)
        if not _replace_with_hardlink(source_authority_original, source_original):
            source_authority_original = source_original
        passthrough_artifacts.update({
            "source_original": str(source_original),
            "source_authority_original": str(source_authority_original),
            "target_original": str(target_original),
        })
    try:
        job_fingerprint = page_job_fingerprint(pair, config)
    except Exception:
        # A manually excluded target page can still be preserved even if its
        # optional source counterpart was moved after pairing. Such a page is
        # simply not eligible for resume fingerprint reuse.
        job_fingerprint = ""
    requested_mode = str((extra_meta or {}).get("transfer_mode") or config.transfer.mode or "").strip().lower()
    mode_contract = get_mode_contract(requested_mode)
    artifact_violations = mode_artifact_violations(requested_mode, page_root)
    passthrough_qa = list(qa or [])
    if artifact_violations:
        passthrough_qa.append(QAItem(
            "mode_artifact_leak", "error",
            "Passthrough page contains renderer artifacts owned by another transfer mode.",
            meta={"requested_mode": requested_mode, "violations": artifact_violations},
        ))
    project = PageProject(
        page_id=stem_id(pair.target_path), pair=pair, registration=registration,
        source_blocks=[], target_blocks=[], source_bubbles=[], target_bubbles=[],
        source_units=[], target_units=[], matches=[], lettering=[], qa=passthrough_qa,
        artifacts=passthrough_artifacts,
        meta={
            "page_management": mark.to_dict(),
            "passthrough": True,
            "passthrough_reason": str(passthrough_reason or "page_manager_exclusion"),
            "transfer_mode": requested_mode,
            "mode_contract": mode_contract.to_dict(),
            "mode_execution": {
                "pass": True, "violations": [],
                "direct_used": False, "mask_used": False, "reletter_used": False,
                "transparent_used": False, "aligned_used": False,
            },
            "mode_isolation": {
                "pass": not artifact_violations,
                "violations": artifact_violations,
                "requested_mode": requested_mode,
                "selected_strategy": "passthrough",
            },
            "job_fingerprint": job_fingerprint,
            "registration_route": registration.diagnostics.get("route", registration.method),
            "qa_summary": qa_summary(passthrough_qa),
            **dict(extra_meta or {}),
        },
    )
    save_json(page_root / "page_management.json", mark.to_dict())
    if config.export.save_project_json:
        save_json(page_root / "project.json", project.to_dict())
    return project
