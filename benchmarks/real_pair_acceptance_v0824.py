"""Real-pair acceptance runner for v0.8.24 locked-raster transfer.

The repository intentionally does not bundle copyrighted manga pages. Supply a
translated Chinese scan and the corresponding HD Japanese page on the command
line. The runner disables OCR so the raster/geometry fallback itself is tested.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.models import PagePair
from manga_hd_transfer.pipeline import TransferPipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("output")
    ap.add_argument("--expect-records", type=int, default=None)
    ns = ap.parse_args()
    out = Path(ns.output)
    cfg = PipelineConfig()
    cfg.transfer.mode = "mask_replace"
    cfg.ocr.backend = "none"
    cfg.ocr.source_backend = "none"
    cfg.ocr.target_backend = "none"
    cfg.cache.enabled = False
    cfg.export.layer_bundle = False
    cfg.export.save_debug = False
    pair = PagePair(ns.source, ns.target, 0, 0, 1.0, 0.0, ["real-pair-acceptance"])
    project = TransferPipeline(cfg).process_page(pair, out, out / "final_export.png")
    records = list(project.meta.get("mask_replace", {}).get("records", []) or [])
    summary = project.meta.get("qa_summary", {})
    report = {
        "source": ns.source,
        "target": ns.target,
        "records": len(records),
        "geometry_applied": sum(bool(x.get("applied")) for x in records),
        "content_complete": sum(bool(x.get("content_complete")) for x in records),
        "content_unverified": sum(bool(x.get("applied")) and not str(x.get("content_check", "")).startswith("checked") for x in records),
        "qa": summary,
        "record_routes": [
            {
                "target": x.get("target_bubble_id"),
                "geometry": x.get("geometry_mode"),
                "clarity": x.get("clarity_mode"),
                "content_complete": x.get("content_complete"),
                "source_ink_coverage": x.get("source_ink_coverage"),
                "target_residual_ratio": x.get("target_residual_ratio"),
                "sr_backend": x.get("sr_backend"),
                "uniform_scale": x.get("sr_scale"),
                "local_dx": x.get("local_dx"),
                "local_dy": x.get("local_dy"),
                "target_coverage": x.get("target_coverage"),
                "spill_ratio": x.get("spill_ratio"),
            }
            for x in records
        ],
    }
    (out / "real_pair_acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ok = bool(summary.get("pass")) and report["geometry_applied"] == len(records) and report["content_complete"] == len(records)
    if ns.expect_records is not None:
        ok = ok and len(records) == ns.expect_records
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
