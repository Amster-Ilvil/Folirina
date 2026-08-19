from __future__ import annotations

"""Folirina v2.3.4-based Direct subprocess bridge with v2.3.13 strict mask guard.

Direct mode is deliberately executed in a clean Python subprocess whose
PYTHONPATH points at the isolated v2.3.4 ``manga_hd_transfer`` source tree copied
from the user-provided, known-good archive, plus two Direct-only v2.3.13 safety patches. This prevents newer shared
page-flow, mode-contract, QA, detector orchestration, or import state from
silently changing Direct behavior while allowing every other renderer to keep
using the current codebase.
"""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from .models import (
    BubbleInstance,
    LetteringResult,
    PagePair,
    PageProject,
    QAItem,
    RegistrationResult,
    TextBlock,
    TextUnit,
    UnitMatch,
)

V234_ARCHIVE_SHA256 = "63d1df8d9bff426f22362a777ce7fe33e25da4aa9f68ad6f31053847a5607bc5"
V234_LANE = "Folirina v2.3.4 Direct Contract Guard + v2.3.13 Strict Mask Guard"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _vendor_root() -> Path:
    return _project_root() / "vendor" / "v2.3.4-direct-contract-guard"


_VENDOR_VERIFIED = False

def _verify_vendor_integrity() -> None:
    """Fail closed unless the embedded v2.3.4-based Direct tree matches its patched manifest."""
    global _VENDOR_VERIFIED
    if _VENDOR_VERIFIED:
        return
    vendor = _vendor_root()
    declared_line = (vendor / "SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
    declared = declared_line.split()[0] if declared_line else ""
    if declared != V234_ARCHIVE_SHA256:
        raise RuntimeError(f"内置 Direct v2.3.4 源归档指纹不匹配：{declared}")
    manifest = json.loads((vendor / "SOURCE_FILE_SHA256.json").read_text(encoding="utf-8"))
    files = dict(manifest.get("files") or {}) if isinstance(manifest, dict) else {}
    if not files:
        raise RuntimeError("内置 Direct v2.3.4 源文件指纹清单为空")
    source_root = vendor / "src" / "manga_hd_transfer"
    bad: list[str] = []
    for rel, expected in sorted(files.items()):
        path = source_root / rel
        if not path.exists():
            bad.append(rel + ":missing")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != str(expected):
            bad.append(rel + ":sha256")
    if bad:
        raise RuntimeError("内置 Direct v2.3.4 源文件完整性失败：" + "、".join(bad[:12]))
    _VENDOR_VERIFIED = True


def _polygon(value: Any):
    return [tuple(map(float, p)) for p in (value or [])]


def _page_pair(row: dict[str, Any]) -> PagePair:
    return PagePair(
        source_path=str(row.get("source_path") or ""),
        target_path=str(row.get("target_path") or ""),
        source_index=int(row.get("source_index") or 0),
        target_index=int(row.get("target_index") or 0),
        confidence=float(row.get("confidence") or 0.0),
        score=float(row.get("score") or 0.0),
        reasons=list(row.get("reasons") or []),
    )


def _registration(row: dict[str, Any]) -> RegistrationResult:
    return RegistrationResult(
        matrix=np.asarray(row.get("matrix") or np.eye(3), dtype=np.float64),
        method=str(row.get("method") or ""),
        confidence=float(row.get("confidence") or 0.0),
        inlier_ratio=float(row.get("inlier_ratio") or 0.0),
        reprojection_error=float(row.get("reprojection_error") or 0.0),
        spatial_coverage=float(row.get("spatial_coverage") or 0.0),
        num_matches=int(row.get("num_matches") or 0),
        source_size=tuple(int(x) for x in (row.get("source_size") or [0, 0])),
        target_size=tuple(int(x) for x in (row.get("target_size") or [0, 0])),
        diagnostics=dict(row.get("diagnostics") or {}),
    )


def _text_block(row: dict[str, Any]) -> TextBlock:
    return TextBlock(
        id=str(row.get("id") or ""), polygon=_polygon(row.get("polygon")),
        text=str(row.get("text") or ""), confidence=float(row.get("confidence") or 0.0),
        kind=str(row.get("kind") or "unknown"), reading_order=int(row.get("reading_order") or 0),
        bubble_id=row.get("bubble_id"), meta=dict(row.get("meta") or {}),
    )


def _bubble(row: dict[str, Any]) -> BubbleInstance:
    return BubbleInstance(
        id=str(row.get("id") or ""), polygon=_polygon(row.get("polygon")),
        confidence=float(row.get("confidence") or 0.0), kind=str(row.get("kind") or "speech"),
        block_ids=list(row.get("block_ids") or []), meta=dict(row.get("meta") or {}),
    )


def _text_unit(row: dict[str, Any]) -> TextUnit:
    return TextUnit(
        id=str(row.get("id") or ""), polygon=_polygon(row.get("polygon")),
        block_ids=list(row.get("block_ids") or []), text=str(row.get("text") or ""),
        confidence=float(row.get("confidence") or 0.0), kind=str(row.get("kind") or "unknown"),
        reading_order=int(row.get("reading_order") or 0), bubble_id=row.get("bubble_id"),
        meta=dict(row.get("meta") or {}),
    )


def _unit_match(row: dict[str, Any]) -> UnitMatch:
    return UnitMatch(
        source_unit_id=str(row.get("source_unit_id") or ""),
        target_unit_id=str(row.get("target_unit_id") or ""),
        confidence=float(row.get("confidence") or 0.0), cost=float(row.get("cost") or 0.0),
        relation=str(row.get("relation") or "one_to_one"), reasons=list(row.get("reasons") or []),
    )


def _lettering(row: dict[str, Any]) -> LetteringResult:
    return LetteringResult(
        unit_id=str(row.get("unit_id") or ""), text=str(row.get("text") or ""),
        font_path=str(row.get("font_path") or ""), font_size=int(row.get("font_size") or 0),
        orientation=str(row.get("orientation") or "vertical"), lines=list(row.get("lines") or []),
        bbox=tuple(int(x) for x in (row.get("bbox") or [0, 0, 0, 0])),
        coverage_inside_safe=float(row.get("coverage_inside_safe") or 0.0),
        success=bool(row.get("success")), reason=str(row.get("reason") or ""),
    )


def _qa(row: dict[str, Any]) -> QAItem:
    return QAItem(
        code=str(row.get("code") or ""), severity=str(row.get("severity") or "info"),
        message=str(row.get("message") or ""), unit_id=row.get("unit_id"), value=row.get("value"),
        threshold=row.get("threshold"), meta=dict(row.get("meta") or {}),
    )


def project_from_v234_dict(row: dict[str, Any]) -> PageProject:
    project = PageProject(
        page_id=str(row.get("page_id") or ""), pair=_page_pair(dict(row.get("pair") or {})),
        registration=_registration(dict(row.get("registration") or {})),
        source_blocks=[_text_block(dict(x)) for x in (row.get("source_blocks") or [])],
        target_blocks=[_text_block(dict(x)) for x in (row.get("target_blocks") or [])],
        source_bubbles=[_bubble(dict(x)) for x in (row.get("source_bubbles") or [])],
        target_bubbles=[_bubble(dict(x)) for x in (row.get("target_bubbles") or [])],
        source_units=[_text_unit(dict(x)) for x in (row.get("source_units") or [])],
        target_units=[_text_unit(dict(x)) for x in (row.get("target_units") or [])],
        matches=[_unit_match(dict(x)) for x in (row.get("matches") or [])],
        lettering=[_lettering(dict(x)) for x in (row.get("lettering") or [])],
        qa=[_qa(dict(x)) for x in (row.get("qa") or [])],
        artifacts={str(k): str(v) for k, v in dict(row.get("artifacts") or {}).items()},
        meta=dict(row.get("meta") or {}),
    )
    # In-memory observability only; the page's on-disk project.json remains the
    # exact file written by the v2.3.4 engine.
    project.meta["direct_execution_lane"] = "embedded-v2.3.4-contract-guard+v2.3.13-strict-mask"
    project.meta["direct_execution_archive_sha256"] = V234_ARCHIVE_SHA256
    return project


def run_direct_v234_page(
    *, config, pair: PagePair, page_root: str | Path, final_path: str | Path | None,
    page_mark=None, cancel_cb=None, progress_cb=None,
) -> PageProject:
    _verify_vendor_integrity()
    vendor = _vendor_root()
    vendor_src = vendor / "src"
    runner = vendor / "run_page.py"
    if not (vendor_src / "manga_hd_transfer" / "pipeline.py").exists() or not runner.exists():
        raise RuntimeError(f"缺少内置 {V234_LANE}：{vendor}")

    if progress_cb is not None:
        progress_cb(1, "direct_v234", "直接贴图 · 启动 Folirina v2.3.4 Contract Guard 独立引擎")

    page_root = Path(page_root)
    page_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="folirina-direct-v234-") as tmp:
        tmp = Path(tmp)
        request_path = tmp / "request.json"
        response_path = tmp / "response.json"
        cancel_path = tmp / "cancel.requested"
        mark_payload = page_mark.to_dict() if hasattr(page_mark, "to_dict") else (dict(page_mark) if isinstance(page_mark, dict) else None)
        request = {
            "config": config.model_dump(mode="json"),
            "pair": asdict(pair),
            "page_root": str(page_root),
            "final_path": str(final_path) if final_path is not None else None,
            "page_mark": mark_payload,
            "cancel_path": str(cancel_path),
        }
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        env = os.environ.copy()
        # Deliberately DO NOT append the current src tree. The subprocess sees
        # only the isolated v2.3.4-based Direct package as ``manga_hd_transfer``.
        env["PYTHONPATH"] = str(vendor_src)
        env["FOLIRINA_DIRECT_EXECUTION_LANE"] = "v2.3.4-contract-guard+v2.3.13-strict-mask"
        proc = subprocess.Popen(
            [sys.executable, str(runner), str(request_path), str(response_path)],
            cwd=str(vendor), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True,
        )
        cancel_sent = False
        cancel_started = 0.0
        while proc.poll() is None:
            if cancel_cb is not None and cancel_cb():
                if not cancel_sent:
                    cancel_path.touch()
                    cancel_sent = True
                    cancel_started = time.monotonic()
                elif time.monotonic() - cancel_started > 8.0:
                    proc.terminate()
            time.sleep(0.08)
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        payload: dict[str, Any] = {}
        if response_path.exists():
            try:
                payload = json.loads(response_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        status = str(payload.get("status") or "")
        if status == "cancelled" or cancel_sent:
            # Import locally to avoid a module cycle at import time.
            from .pipeline import PipelineCancelled
            raise PipelineCancelled(str(payload.get("message") or "Direct v2.3.4 cancelled"))
        if proc.returncode != 0 or status != "ok":
            message = str(payload.get("message") or "Direct v2.3.4 独立引擎执行失败")
            trace = str(payload.get("traceback") or stderr or "")
            raise RuntimeError(message + ("\n\n" + trace[-12000:] if trace else ""))
        project = project_from_v234_dict(dict(payload.get("project") or {}))
        planner = dict(project.meta.get("transfer_planner") or {})
        execution = {
            "schema": "folirina.direct_v234_execution.v1",
            "engine": V234_LANE,
            "base_archive_sha256": V234_ARCHIVE_SHA256,
            "direct_patch_level": "v2.3.13-strict-semantic-mask",
            "mask_authority": "source_bubble_or_textbox_hints_only",
            "support_guard": "fail_closed_exact_semantic_mask",
            "selected_strategy": str(planner.get("strategy") or ""),
            "reason": str(planner.get("reason") or ""),
            "direct_plan_available": bool((planner.get("evidence") or {}).get("direct_plan_available", False)),
            "direct_plan_safe": bool(planner.get("direct_plan_safe", False)),
            "applied_count": int((project.meta.get("direct_patch") or {}).get("applied_count", 0) or 0),
        }
        try:
            (page_root / "direct_v234_execution.json").write_text(
                json.dumps(execution, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        project.meta["direct_v234_execution"] = execution
        if progress_cb is not None:
            progress_cb(100, "direct_v234", "直接贴图 · v2.3.4 Contract Guard + 严格蒙版保护完成")
        return project
