#!/usr/bin/env python3
from __future__ import annotations

"""Run one real SOURCE/TARGET pair and emit a publication-oriented acceptance report.

The script never bundles input images. It is intended for local/private regression
pages where expected translated regions have been manually confirmed.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from manga_hd_transfer.config import PipelineConfig  # noqa: E402
from manga_hd_transfer.models import PagePair  # noqa: E402
from manga_hd_transfer.pipeline import TransferPipeline  # noqa: E402
from manga_hd_transfer.io_utils import save_json  # noqa: E402


def _load_expected(path: str | None) -> dict:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected-regions JSON must be an object")
    return payload


def _active_mask(page_dir: Path, mode: str, shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    candidates = []
    if mode == "direct_patch":
        candidates.append((page_dir / "direct_patch_regions.png", "direct_patch_regions"))
    else:
        candidates.append((page_dir / "mask_transfer_mask.png", "mask_transfer_mask"))
    candidates += [
        (page_dir / "target_clear_mask.png", "target_clear_mask"),
        (page_dir / "clear_mask.png", "clear_mask"),
    ]
    for path, label in candidates:
        if path.exists():
            m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape != shape:
                    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                return m, label
    return np.zeros(shape, np.uint8), "none"


def _expected_union(expected: dict, shape: tuple[int, int]) -> tuple[np.ndarray, list[dict]]:
    h, w = shape
    union = np.zeros((h, w), np.uint8)
    rows = []
    for i, row in enumerate(expected.get("regions", []) or []):
        if not isinstance(row, dict) or "bbox" not in row:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in row["bbox"]]
        x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
        y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
        if x1 <= x0 or y1 <= y0:
            continue
        union[y0:y1, x0:x1] = 255
        rows.append({"name": str(row.get("name") or f"region_{i:02d}"), "bbox": [x0, y0, x1, y1]})
    return union, rows


def _markdown(report: dict) -> str:
    m = report["metrics"]
    lines = [
        f"# Real Pair Acceptance — {report['version']}",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Result: **{'PASS' if report['pass'] else 'FAIL'}**",
        f"- Registration: `{m['registration_method']}` / {m['registration_confidence']:.4f}",
        f"- Same-page confidence: {m['same_page_confidence']:.4f}",
        f"- Applied records: {m['applied_records']}",
        f"- SAFE / REVIEW / REJECT: {m['triage_safe']} / {m['triage_review']} / {m['triage_reject']}",
        f"- Outside-transfer changed pixels: {m['outside_transfer_changed_pixels']}",
        f"- Outside-expected write pixels: {m['outside_expected_write_pixels']}",
        f"- Protected target-border changed pixels: {m['protected_border_changed_pixels']}",
        f"- QA errors / warnings: {m['qa_errors']} / {m['qa_warnings']}",
        "",
        "## Expected regions",
    ]
    for row in report.get("expected_regions", []):
        lines.append(f"- `{row['name']}` bbox={row['bbox']} write_coverage={row['write_coverage']:.4f}")
    if report.get("fail_reasons"):
        lines += ["", "## Fail reasons"] + [f"- {x}" for x in report["fail_reasons"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto", "mask_replace", "direct_patch", "hybrid"])
    ap.add_argument("--ocr-backend", default="none")
    ap.add_argument("--expected-regions-json")
    args = ap.parse_args()

    source = str(Path(args.source).resolve())
    target = str(Path(args.target).resolve())
    out = Path(args.output).resolve()
    page_dir = out / "page"
    page_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig()
    cfg.transfer.mode = args.mode
    cfg.registration.backend = "opencv"
    cfg.ocr.backend = args.ocr_backend
    cfg.ocr.source_backend = args.ocr_backend
    cfg.ocr.target_backend = args.ocr_backend
    cfg.export.layer_bundle = False
    cfg.export.save_debug = True
    cfg.qa.fail_empty_mask_replace = False

    pair = PagePair(source, target, 0, 0, 0.99, 0.01, [])
    project = TransferPipeline(cfg).process_page(pair, page_dir)

    target_img = cv2.imread(target, cv2.IMREAD_COLOR)
    final_img = cv2.imread(str(page_dir / "final.png"), cv2.IMREAD_COLOR)
    if target_img is None or final_img is None:
        raise RuntimeError("target/final image missing")
    active_mask, mask_name = _active_mask(page_dir, args.mode, target_img.shape[:2])
    delta = cv2.absdiff(target_img, final_img).max(axis=2)
    outside_transfer_changed = int(np.count_nonzero((delta > 5) & (active_mask == 0)))

    expected_cfg = _load_expected(args.expected_regions_json)
    expected_union, expected_rows = _expected_union(expected_cfg, target_img.shape[:2])
    outside_expected_write = int(np.count_nonzero((active_mask > 0) & (expected_union == 0))) if expected_rows else 0
    for row in expected_rows:
        x0, y0, x1, y1 = row["bbox"]
        roi = active_mask[y0:y1, x0:x1]
        row["write_coverage"] = float(np.count_nonzero(roi) / max(1, roi.size))

    records = list((project.meta.get("mask_replace", {}) or {}).get("records", []) or [])
    if args.mode == "direct_patch":
        records = list((project.meta.get("direct_patch", {}) or {}).get("records", []) or [])
    triage = [str(r.get("triage_state", "")) for r in records]
    protected_border_changed = int(sum(
        int((((r.get("meta") or {}).get("target_border_preservation") or {}).get("changed_after_restore", 0)) or 0)
        for r in records
    ))
    qa_errors = sum(1 for q in project.qa if q.severity == "error")
    qa_warnings = sum(1 for q in project.qa if q.severity == "warning")
    same_page_conf = float((project.meta.get("page_pairing_check", {}) or {}).get("confidence", 0.0))

    direct_files = [page_dir / "direct_patch_layer.png", page_dir / "direct_patch_regions.png", page_dir / "direct_patch.json"]
    mask_files = [page_dir / "mask_transfer_layer.png", page_dir / "mask_transfer_mask.png", page_dir / "mask_transfer.json"]
    namespace_ok = True
    if args.mode == "direct_patch":
        namespace_ok = not any(p.exists() for p in mask_files)
    elif args.mode == "mask_replace":
        namespace_ok = not any(p.exists() for p in direct_files)

    outside_expected_max = int(expected_cfg.get("outside_expected_write_max", 100)) if expected_rows else 10**12
    fail_reasons = []
    if outside_transfer_changed != 0:
        fail_reasons.append(f"outside_transfer_changed_pixels={outside_transfer_changed}")
    if expected_rows and outside_expected_write > outside_expected_max:
        fail_reasons.append(f"outside_expected_write_pixels={outside_expected_write}>{outside_expected_max}")
    if protected_border_changed != 0:
        fail_reasons.append(f"protected_border_changed_pixels={protected_border_changed}")
    if qa_errors:
        fail_reasons.append(f"qa_errors={qa_errors}")
    if not namespace_ok:
        fail_reasons.append("direct_mask_namespace_violation")

    report = {
        "schema": "manga-hd-transfer/real-pair-acceptance/v1",
        "version": __import__("manga_hd_transfer").__version__,
        "mode": args.mode,
        "source": source,
        "target": target,
        "mask_artifact": mask_name,
        "pass": not fail_reasons,
        "fail_reasons": fail_reasons,
        "metrics": {
            "registration_method": project.registration.method,
            "registration_confidence": float(project.registration.confidence),
            "same_page_confidence": same_page_conf,
            "applied_records": int(sum(1 for r in records if bool(r.get("applied", True)))),
            "triage_safe": triage.count("SAFE"),
            "triage_review": triage.count("REVIEW"),
            "triage_reject": triage.count("REJECT"),
            "outside_transfer_changed_pixels": outside_transfer_changed,
            "outside_expected_write_pixels": outside_expected_write,
            "protected_border_changed_pixels": protected_border_changed,
            "qa_errors": qa_errors,
            "qa_warnings": qa_warnings,
            "namespace_ok": namespace_ok,
        },
        "expected_regions": expected_rows,
        "qa_codes": [q.code for q in project.qa],
        "artifacts": project.artifacts,
    }
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "real_pair_acceptance.json", report)
    (out / "REAL_PAIR_ACCEPTANCE.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
