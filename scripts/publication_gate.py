#!/usr/bin/env python3
from __future__ import annotations

"""Run publication-quality gates on paired real-book benchmark sets.

The benchmark format is intentionally data-only so private books can live outside
this repository.  See docs/PUBLICATION_GATE.md and benchmarks/README.md.
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.io_utils import read_image
from manga_hd_transfer.models import PagePair
from manga_hd_transfer.pairing import pair_directories
from manga_hd_transfer.pipeline import TransferPipeline

SCHEMA = "manga-hd-transfer/publication-gate/v1"


@dataclass(slots=True)
class GateThresholds:
    page_pair_accuracy_min: float = 0.995
    identity_match_accuracy_min: float = 0.99
    auto_pass_visible_japanese_residual_max: float = 0.0
    auto_pass_border_damage_max: float = 0.0
    glyph_safe_area_overflow_max: float = 0.0
    explicit_direct_silent_mask_fallback_max: int = 0
    review_rate_max: float = 0.15


@dataclass(slots=True)
class PageGateResult:
    work_id: str
    page_id: str
    source: str
    target: str
    tags: list[str]
    pair_correct: bool | None = None
    identity_match_accuracy: float | None = None
    target_residual_ratio: float | None = None
    border_damage_ratio: float | None = None
    safe_area_overflow: float | None = None
    needs_review: bool = False
    auto_pass: bool = False
    elapsed_seconds: float = 0.0
    qa_errors: int = 0
    qa_warnings: int = 0
    failure: str = ""
    archive_dir: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GateSummary:
    schema: str
    generated_at: str
    benchmark_root: str
    version: str
    page_count: int
    measured_pair_pages: int
    page_pair_accuracy: float | None
    measured_match_pages: int
    identity_match_accuracy: float | None
    auto_pass_pages: int
    auto_pass_residual_failures: int
    auto_pass_border_damage_failures: int
    safe_area_overflow_failures: int
    direct_silent_mask_fallbacks: int
    review_pages: int
    review_rate: float
    median_seconds_per_page: float | None
    pass_gate: bool
    thresholds: dict[str, Any]
    pages: list[dict[str, Any]]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: Path | None) -> PipelineConfig:
    if path is None:
        cfg = PipelineConfig()
        cfg.transfer.mode = "auto"
        return cfg
    return PipelineConfig.model_validate(_load_json(path))


def _rel(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else root / p


def discover_works(root: Path) -> list[tuple[Path, dict]]:
    rows: list[tuple[Path, dict]] = []
    for labels_path in sorted(root.glob("*/labels.json")):
        payload = _load_json(labels_path)
        if isinstance(payload, dict) and isinstance(payload.get("pages"), list):
            rows.append((labels_path.parent, payload))
    return rows


def _expected_pair_map(work: Path, labels: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in labels.get("pages", []):
        src = _rel(work, row.get("primary"))
        tgt = _rel(work, row.get("target"))
        if src and tgt:
            out[str(src.resolve())] = str(tgt.resolve())
    return out


def _actual_pair_map(work: Path, labels: dict, cfg: PipelineConfig) -> dict[str, str]:
    primary_dir = _rel(work, labels.get("primary_dir", "primary_cn"))
    target_dir = _rel(work, labels.get("target_dir", "target_jp"))
    if primary_dir is None or target_dir is None or not primary_dir.exists() or not target_dir.exists():
        return {}
    pairs, _, _ = pair_directories(primary_dir, target_dir, cfg.pairing)
    return {str(Path(p.source_path).resolve()): str(Path(p.target_path).resolve()) for p in pairs}


def _record_metrics(project) -> tuple[float, float, bool, int, int]:
    records = []
    for key in ("direct_patch", "mask_replace"):
        meta = (project.meta or {}).get(key, {}) or {}
        records.extend(meta.get("records", []) or [])
    residual = max([float(r.get("target_residual_ratio", 0.0) or 0.0) for r in records] or [0.0])
    overflow = max([float(r.get("spill_ratio", 0.0) or 0.0) for r in records] or [0.0])
    needs_review = bool(any(
        bool(r.get("review_required")) or str(r.get("triage_state", "")).upper() in {"REVIEW", "REJECT"}
        for r in records
    ))
    qa_errors = sum(1 for q in project.qa if str(q.severity).lower() == "error")
    qa_warnings = sum(1 for q in project.qa if str(q.severity).lower() == "warning")
    return residual, overflow, needs_review, qa_errors, qa_warnings


def _border_damage_ratio(page_dir: Path) -> float:
    target_path = page_dir / "target_original.png"
    final_path = page_dir / "final.png"
    if not target_path.exists() or not final_path.exists():
        return 0.0
    target = read_image(target_path)
    final = read_image(final_path)
    if target.shape != final.shape:
        return 1.0
    mask_candidates = [page_dir / "direct_patch_regions.png", page_dir / "mask_transfer_mask.png", page_dir / "target_clear_mask.png"]
    mask = next((cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in mask_candidates if p.exists()), None)
    if mask is None:
        return float(np.mean(np.any(target != final, axis=2)))
    use = (mask > 0).astype(np.uint8) * 255
    ring_outer = cv2.dilate(use, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    ring_inner = cv2.dilate(use, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    guard = (ring_outer > 0) & (ring_inner == 0)
    n = int(np.count_nonzero(guard))
    if n == 0:
        return 0.0
    delta = np.max(np.abs(final.astype(np.int16) - target.astype(np.int16)), axis=2)
    return float(np.count_nonzero(guard & (delta > 8)) / n)


def _match_accuracy(project, expected: list) -> float | None:
    if not expected:
        return None
    actual = {(m.source_unit_id, m.target_unit_id) for m in project.matches if m.relation == "one_to_one"}
    wanted = set()
    for item in expected:
        if isinstance(item, dict):
            wanted.add((str(item.get("source_unit_id", "")), str(item.get("target_unit_id", ""))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            wanted.add((str(item[0]), str(item[1])))
    wanted.discard(("", ""))
    if not wanted:
        return None
    return float(len(actual & wanted) / len(wanted))


def _archive_failure(page_dir: Path, archive_root: Path, result: PageGateResult) -> None:
    dest = archive_root / result.work_id / result.page_id
    dest.mkdir(parents=True, exist_ok=True)
    wanted = [
        "source_original.png", "source_authority_original.png", "target_original.png", "final.png",
        "debug_registration.png", "debug_matching.png", "debug_direct_patch.png", "debug_mask_replace.png",
        "target_clear_mask.png", "direct_patch_regions.png", "mask_transfer_mask.png", "transfer_audit.json",
        "project.json", "qa.json",
    ]
    import shutil
    for name in wanted:
        src = page_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
    project_payload = {}
    qa_payload = {}
    try:
        if (page_dir / "project.json").exists():
            project_payload = _load_json(page_dir / "project.json")
        if (page_dir / "qa.json").exists():
            qa_payload = _load_json(page_dir / "qa.json")
    except Exception:
        pass
    meta = project_payload.get("meta", {}) if isinstance(project_payload, dict) else {}
    rt = meta.get("replace_translation", {}) if isinstance(meta, dict) else {}
    evidence = {
        "matching_diagnostics": rt.get("matching_diagnostics", {}) if isinstance(rt, dict) else {},
        "force_actions": rt.get("force_actions", []) if isinstance(rt, dict) else [],
        "qa_codes": [row.get("code") for row in qa_payload.get("issues", []) if isinstance(row, dict)] if isinstance(qa_payload, dict) else [],
        "selected_source_path": rt.get("selected_source_path") if isinstance(rt, dict) else None,
        "selected_arbitration_evidence": rt.get("selected_arbitration_evidence", {}) if isinstance(rt, dict) else {},
    }
    (dest / "failure_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    rt_summary = page_dir / "replace_translation" / "summary.json"
    if rt_summary.exists():
        shutil.copy2(rt_summary, dest / "replace_translation_summary.json")
    (dest / "gate_result.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    result.archive_dir = str(dest)


def run_gate(root: Path, cfg: PipelineConfig, output: Path, thresholds: GateThresholds) -> GateSummary:
    works = discover_works(root)
    if not works:
        raise SystemExit(f"No benchmark works found under {root}; expected */labels.json")
    pages: list[PageGateResult] = []
    pair_checks: list[bool] = []
    output.mkdir(parents=True, exist_ok=True)
    archive_root = output / "failures"
    for work, labels in works:
        work_id = str(labels.get("work_id") or work.name)
        expected_pairs = _expected_pair_map(work, labels)
        actual_pairs = _actual_pair_map(work, labels, cfg)
        for row in labels.get("pages", []):
            page_id = str(row.get("id") or Path(str(row.get("target", "page"))).stem)
            source_path = _rel(work, row.get("primary"))
            target_path = _rel(work, row.get("target"))
            tags = [str(x) for x in row.get("tags", [])]
            result = PageGateResult(work_id, page_id, str(source_path or ""), str(target_path or ""), tags)
            if source_path is None or target_path is None or not source_path.exists() or not target_path.exists():
                result.failure = "benchmark_file_missing"
                pages.append(result)
                continue
            expected_target = expected_pairs.get(str(source_path.resolve()))
            actual_target = actual_pairs.get(str(source_path.resolve()))
            if expected_target is not None:
                result.pair_correct = bool(actual_target == expected_target)
                pair_checks.append(result.pair_correct)
            page_dir = output / "pages" / work_id / page_id
            pair = PagePair(str(source_path), str(target_path), int(row.get("source_index", 0)), int(row.get("target_index", 0)), 1.0, 0.0, ["publication_gate"])
            started = time.perf_counter()
            try:
                project = TransferPipeline(cfg).process_page(pair, page_dir)
                result.elapsed_seconds = time.perf_counter() - started
                result.identity_match_accuracy = _match_accuracy(project, row.get("expected_unit_matches", []))
                residual, overflow, review, errors, warnings = _record_metrics(project)
                result.target_residual_ratio = residual
                result.safe_area_overflow = overflow
                result.needs_review = review
                result.qa_errors = errors
                result.qa_warnings = warnings
                result.border_damage_ratio = _border_damage_ratio(page_dir)
                result.auto_pass = bool(not review and errors == 0 and not bool((project.meta or {}).get("passthrough")))
                result.meta = {
                    "transfer_mode": (project.meta or {}).get("transfer_mode"),
                    "registration_method": project.registration.method,
                    "registration_confidence": project.registration.confidence,
                    "selected_source": ((project.meta or {}).get("replace_translation") or {}).get("selected_source_path"),
                    "selected_source_kind": ((project.meta or {}).get("replace_translation") or {}).get("selected_source_kind"),
                }
            except Exception as exc:
                result.elapsed_seconds = time.perf_counter() - started
                result.failure = f"{type(exc).__name__}: {exc}"
            hard_fail = bool(
                result.failure
                or (result.auto_pass and (result.target_residual_ratio or 0.0) > thresholds.auto_pass_visible_japanese_residual_max)
                or (result.auto_pass and (result.border_damage_ratio or 0.0) > thresholds.auto_pass_border_damage_max)
                or (result.safe_area_overflow or 0.0) > thresholds.glyph_safe_area_overflow_max
            )
            if hard_fail:
                _archive_failure(page_dir, archive_root, result)
            pages.append(result)

    pair_accuracy = float(np.mean(pair_checks)) if pair_checks else None
    match_values = [p.identity_match_accuracy for p in pages if p.identity_match_accuracy is not None]
    match_accuracy = float(np.mean(match_values)) if match_values else None
    auto = [p for p in pages if p.auto_pass]
    residual_failures = sum(1 for p in auto if (p.target_residual_ratio or 0.0) > thresholds.auto_pass_visible_japanese_residual_max)
    border_failures = sum(1 for p in auto if (p.border_damage_ratio or 0.0) > thresholds.auto_pass_border_damage_max)
    overflow_failures = sum(1 for p in pages if (p.safe_area_overflow or 0.0) > thresholds.glyph_safe_area_overflow_max)
    # Contract is additionally enforced by unit tests/release audit. The gate
    # reports any artifact evidence that an explicit direct page used mask output.
    silent_fallbacks = 0
    for p in pages:
        if p.meta.get("transfer_mode") == "direct_patch":
            page_dir = output / "pages" / p.work_id / p.page_id
            if (page_dir / "mask_transfer_mask.png").exists() or (page_dir / "mask_transfer.json").exists():
                silent_fallbacks += 1
    review_pages = sum(1 for p in pages if p.needs_review or p.failure)
    review_rate = review_pages / max(1, len(pages))
    times = [p.elapsed_seconds for p in pages if p.elapsed_seconds > 0]
    median_time = float(np.median(times)) if times else None
    conditions = [
        pair_accuracy is None or pair_accuracy >= thresholds.page_pair_accuracy_min,
        match_accuracy is None or match_accuracy >= thresholds.identity_match_accuracy_min,
        residual_failures == 0,
        border_failures == 0,
        overflow_failures == 0,
        silent_fallbacks <= thresholds.explicit_direct_silent_mask_fallback_max,
        review_rate <= thresholds.review_rate_max,
        not any(p.failure for p in pages),
    ]
    from manga_hd_transfer import __version__
    return GateSummary(
        schema=SCHEMA, generated_at=datetime.now().astimezone().isoformat(), benchmark_root=str(root), version=__version__,
        page_count=len(pages), measured_pair_pages=len(pair_checks), page_pair_accuracy=pair_accuracy,
        measured_match_pages=len(match_values), identity_match_accuracy=match_accuracy,
        auto_pass_pages=len(auto), auto_pass_residual_failures=residual_failures,
        auto_pass_border_damage_failures=border_failures, safe_area_overflow_failures=overflow_failures,
        direct_silent_mask_fallbacks=silent_fallbacks, review_pages=review_pages, review_rate=review_rate,
        median_seconds_per_page=median_time, pass_gate=all(conditions), thresholds=asdict(thresholds),
        pages=[asdict(p) for p in pages],
    )


def write_report(summary: GateSummary, output: Path) -> tuple[Path, Path]:
    json_path = output / "publication_gate.json"
    md_path = output / f"GATE_REPORT_{summary.version}.md"
    json_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    def pct(v):
        return "N/A" if v is None else f"{100*v:.3f}%"
    rows = [
        "# Publication Gate Report",
        "",
        f"- Schema: `{summary.schema}`",
        f"- Version: `{summary.version}`",
        f"- Pages: **{summary.page_count}**",
        f"- Overall: **{'PASS' if summary.pass_gate else 'FAIL'}**",
        "",
        "| Metric | Result | Gate |",
        "|---|---:|---:|",
        f"| Page pair accuracy | {pct(summary.page_pair_accuracy)} | ≥ {100*summary.thresholds['page_pair_accuracy_min']:.2f}% |",
        f"| Identity match accuracy | {pct(summary.identity_match_accuracy)} | ≥ {100*summary.thresholds['identity_match_accuracy_min']:.2f}% |",
        f"| Auto-pass residual failures | {summary.auto_pass_residual_failures} | 0 |",
        f"| Auto-pass border damage failures | {summary.auto_pass_border_damage_failures} | 0 |",
        f"| Safe-area overflow failures | {summary.safe_area_overflow_failures} | 0 |",
        f"| Direct silent Mask fallbacks | {summary.direct_silent_mask_fallbacks} | 0 |",
        f"| Review rate | {pct(summary.review_rate)} | ≤ {100*summary.thresholds['review_rate_max']:.1f}% (initial target) |",
        f"| Median seconds/page | {summary.median_seconds_per_page if summary.median_seconds_per_page is not None else 'N/A'} | observed baseline |",
        "",
        "## Failed / Review Pages",
        "",
    ]
    flagged = [p for p in summary.pages if p.get("failure") or p.get("needs_review") or not p.get("auto_pass")]
    if not flagged:
        rows.append("None.")
    else:
        for p in flagged:
            rows.append(f"- `{p['work_id']}/{p['page_id']}` — failure={p.get('failure') or '-'}, review={p.get('needs_review')}, archive=`{p.get('archive_dir') or '-'}`")
    md_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Manga HD Transfer publication benchmark gates")
    ap.add_argument("benchmark_root", type=Path, help="Root containing work_id/labels.json benchmark sets")
    ap.add_argument("--output", type=Path, default=ROOT / "gate_output")
    ap.add_argument("--config", type=Path, default=None, help="Optional PipelineConfig JSON")
    ap.add_argument("--review-rate-max", type=float, default=0.15)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_config(args.config)
    thresholds = GateThresholds(review_rate_max=float(args.review_rate_max))
    summary = run_gate(args.benchmark_root.resolve(), cfg, args.output.resolve(), thresholds)
    json_path, md_path = write_report(summary, args.output.resolve())
    print(f"Publication gate: {'PASS' if summary.pass_gate else 'FAIL'}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0 if summary.pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
