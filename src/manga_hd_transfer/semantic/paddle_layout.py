from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from .schema import SemanticBlock, SemanticLayoutResult
from .router import route_blocks

_RUNTIME: dict[tuple[str, str, float], object] = {}


def _unwrap_result(res: Any) -> dict[str, Any]:
    if isinstance(res, dict):
        if isinstance(res.get("res"), dict):
            return res["res"]
        return res
    for attr in ("json", "res"):
        try:
            value = getattr(res, attr)
            if callable(value):
                value = value()
            if isinstance(value, dict):
                return value.get("res", value)
        except Exception:
            pass
    try:
        data = json.loads(str(res))
        return data.get("res", data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _polygon_from_box(box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    return [(float(x0), float(y0)), (float(x1), float(y0)), (float(x1), float(y1)), (float(x0), float(y1))]


def analyze_with_paddlex(image: np.ndarray, cfg, *, strategy: str = "auto") -> SemanticLayoutResult:
    model_dir = str(getattr(cfg, "paddle_model_dir", "") or "").strip()
    allow_downloads = bool(getattr(cfg, "paddle_allow_model_downloads", False))
    if not model_dir and not allow_downloads:
        return SemanticLayoutResult(False, "pp_doclayout_v3", diagnostics={
            "status": "unavailable",
            "reason": "model_dir_missing_and_downloads_disabled",
            "model_name": str(getattr(cfg, "paddle_model_name", "PP-DocLayoutV3")),
        })
    try:
        from paddlex import create_model  # type: ignore
    except Exception as exc:
        return SemanticLayoutResult(False, "pp_doclayout_v3", diagnostics={"status": "unavailable", "reason": "paddlex_missing", "error": str(exc)})

    model_name = str(getattr(cfg, "paddle_model_name", "PP-DocLayoutV3"))
    device = str(getattr(cfg, "paddle_device", "cpu") or "cpu")
    threshold = float(getattr(cfg, "paddle_threshold", 0.30))
    key = (model_dir or model_name, device, threshold)
    try:
        model = _RUNTIME.get(key)
        if model is None:
            kwargs: dict[str, Any] = {"model_name": model_name, "device": device, "threshold": threshold}
            if model_dir:
                kwargs["model_dir"] = str(Path(model_dir).expanduser())
            model = create_model(**kwargs)
            _RUNTIME[key] = model
        output = model.predict(image, batch_size=1, layout_nms=True)
        first = next(iter(output))
        payload = _unwrap_result(first)
        boxes = payload.get("boxes", []) if isinstance(payload, dict) else []
    except Exception as exc:
        return SemanticLayoutResult(False, "pp_doclayout_v3", diagnostics={"status": "error", "reason": "predict_failed", "error": str(exc)})

    blocks: list[SemanticBlock] = []
    for idx, row in enumerate(boxes if isinstance(boxes, list) else []):
        if not isinstance(row, dict):
            continue
        coord = row.get("coordinate") or row.get("bbox") or row.get("box")
        if not isinstance(coord, (list, tuple)) or len(coord) != 4:
            continue
        x0, y0, x1, y1 = [int(round(float(v))) for v in coord]
        if x1 <= x0 or y1 <= y0:
            continue
        label = str(row.get("label", "unknown"))
        score = float(row.get("score", 0.0) or 0.0)
        poly_raw = row.get("polygon_points") or row.get("polygon")
        polygon = _polygon_from_box((x0, y0, x1, y1))
        if isinstance(poly_raw, (list, tuple)) and len(poly_raw) >= 3:
            try:
                polygon = [(float(p[0]), float(p[1])) for p in poly_raw]
            except Exception:
                pass
        blocks.append(SemanticBlock(
            id=f"pp-layout-{idx:04d}", source="pp_doclayout_v3", raw_label=label,
            semantic_type="unknown", confidence=score, bbox=(x0, y0, x1, y1), polygon=polygon,
            reading_order=int(row.get("order")) if row.get("order") is not None else None,
            meta={"cls_id": row.get("cls_id")},
        ))
    route_blocks(blocks, strategy)
    return SemanticLayoutResult(True, "pp_doclayout_v3", blocks, diagnostics={
        "status": "ok", "model_name": model_name, "threshold": threshold, "block_count": len(blocks),
    })
