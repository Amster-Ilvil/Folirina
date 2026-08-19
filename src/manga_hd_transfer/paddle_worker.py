from __future__ import annotations

"""JSONL Paddle worker for three independent local engines.

- ``ocr``: PaddleOCR (PP-OCRv6 / legacy/custom det+rec)
- ``vl``: PaddleOCR-VL 1.6 document parser
- ``structure``: PP-StructureV3 layout parser

The worker intentionally normalizes every route to the same light-weight block
protocol consumed by the desktop process. Heavy Paddle/PaddleX modules stay in
the isolated OCR venv.
"""

import argparse
import json
import os
from typing import Any


_SKIP_LABELS = {
    "number", "footnote", "header", "header_image", "footer", "footer_image",
    "aside_text", "image", "figure", "chart", "formula", "table", "seal",
}


def _first_present(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-None value without truth-testing it.

    PaddleOCR/PaddleX 3.x result dictionaries intentionally contain numpy
    arrays (for example ``rec_scores``, ``rec_polys`` and ``rec_boxes``).
    Using ``a or b`` on those arrays raises the classic ambiguous truth-value
    ValueError, so all result normalization must use explicit ``is not None``
    checks instead.
    """
    for key in keys:
        if key in mapping:
            value = mapping.get(key)
            if value is not None:
                return value
    return default


def _to_builtin(value: Any) -> Any:
    """Convert numpy-like containers/scalars to plain JSON-safe Python."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _to_builtin(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _to_builtin(value.item())
        except Exception:
            pass
    return value


def _as_list(value: Any) -> list[Any]:
    value = _to_builtin(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _safe_float(value: Any, default: float = 0.0) -> float:
    value = _to_builtin(value)
    if isinstance(value, list):
        value = value[0] if value else default
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_mapping(obj: Any) -> Any:
    if isinstance(obj, (dict, list, tuple, str, int, float, bool)) or obj is None:
        return obj
    for attr in ("json", "res", "result"):
        if hasattr(obj, attr):
            try:
                value = getattr(obj, attr)
                value = value() if callable(value) else value
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except Exception:
                        pass
                if value is not None:
                    return value
            except Exception:
                pass
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(vars(obj))
        except Exception:
            pass
    return obj


def _find_ocr_dict(obj: Any) -> dict[str, Any] | None:
    obj = _as_mapping(obj)
    if isinstance(obj, dict):
        if "rec_texts" in obj and ("rec_polys" in obj or "dt_polys" in obj or "rec_boxes" in obj):
            return obj
        for value in obj.values():
            found = _find_ocr_dict(value)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_ocr_dict(value)
            if found is not None:
                return found
    return None


def _find_parsing_rows(obj: Any) -> list[Any] | None:
    obj = _as_mapping(obj)
    if isinstance(obj, dict):
        rows = obj.get("parsing_res_list")
        if isinstance(rows, list):
            return rows
        for value in obj.values():
            found = _find_parsing_rows(value)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_parsing_rows(value)
            if found is not None:
                return found
    return None


def _poly_from_row(row: Any) -> list[list[float]]:
    row = _as_mapping(row)
    if not isinstance(row, dict):
        return []
    poly = _first_present(row, "block_polygon_points", "polygon_points", "polygon")
    poly = _as_list(poly)
    if len(poly) >= 3:
        try:
            return [[float(p[0]), float(p[1])] for p in poly]
        except Exception:
            pass
    bbox = _first_present(row, "block_bbox", "bbox", "coordinate")
    bbox = _as_list(bbox)
    if len(bbox) >= 4:
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
            return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
        except Exception:
            pass
    return []


def _rows_from_ocr_result(res: Any) -> list[dict[str, Any]]:
    data = _find_ocr_dict(res)
    if data is None:
        return []
    texts = _as_list(_first_present(data, "rec_texts", default=[]))
    raw_scores = _as_list(_first_present(data, "rec_scores", default=[]))
    scores = [_safe_float(v, 1.0) for v in raw_scores]
    if len(scores) < len(texts):
        scores.extend([1.0] * (len(texts) - len(scores)))
    elif len(scores) > len(texts):
        scores = scores[:len(texts)]

    polys_raw = _first_present(data, "rec_polys")
    polys = _as_list(polys_raw) if polys_raw is not None else []
    if len(polys) != len(texts):
        dt_raw = _first_present(data, "dt_polys")
        dt_polys = _as_list(dt_raw) if dt_raw is not None else []
        polys = dt_polys if len(dt_polys) == len(texts) else []

    if not polys:
        boxes = _as_list(_first_present(data, "rec_boxes", default=[]))
        if len(boxes) != len(texts):
            return []
        converted: list[list[list[float]]] = []
        for b in boxes:
            b = _as_list(b)
            if len(b) < 4:
                return []
            converted.append([
                [float(b[0]), float(b[1])], [float(b[2]), float(b[1])],
                [float(b[2]), float(b[3])], [float(b[0]), float(b[3])],
            ])
        polys = converted

    rows: list[dict[str, Any]] = []
    for i, (text, score, poly) in enumerate(zip(texts, scores, polys)):
        poly = _as_list(poly)
        try:
            pts = [[float(_as_list(p)[0]), float(_as_list(p)[1])] for p in poly]
        except Exception:
            continue
        text_s = str(_to_builtin(text) if text is not None else "").strip()
        if len(pts) < 3 or not text_s:
            continue
        rows.append({
            "id": f"ocr-{i:04d}", "polygon": pts, "text": text_s,
            "confidence": _safe_float(score, 1.0), "kind": "text",
        })
    return rows


def _rows_from_parsing_result(res: Any, pipeline: str) -> list[dict[str, Any]]:
    rows = _find_parsing_rows(res)
    if rows is None:
        # Some Structure outputs expose an ``overall_ocr_res`` instead of
        # parsing blocks. Reuse the regular OCR normalization when available.
        return _rows_from_ocr_result(res)
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(rows):
        row = _as_mapping(raw)
        if not isinstance(row, dict):
            continue
        label_value = _first_present(row, "block_label", "label", "type", default="text")
        label = str(_to_builtin(label_value) if label_value is not None else "text").strip().lower()
        if label in _SKIP_LABELS:
            continue
        text_value = _first_present(row, "block_content", "content", "text", default="")
        text = str(_to_builtin(text_value) if text_value is not None else "").strip()
        if not text:
            continue
        poly = _poly_from_row(row)
        if len(poly) < 3:
            continue
        score = _first_present(row, "score", "confidence", default=0.95)
        out.append({
            "id": f"{pipeline}-{i:04d}",
            "polygon": poly,
            "text": text,
            "confidence": _safe_float(score, 0.95),
            "kind": label,
            "label": label,
            "meta": {
                "block_label": label,
                "block_id": _to_builtin(row.get("block_id")),
                "group_id": _to_builtin(row.get("group_id")),
                "block_order": _to_builtin(row.get("block_order")),
                "orientation_hint": "vertical" if "vertical" in label else ("horizontal" if "text" in label else "auto"),
            },
        })
    return out


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def _exception_detail(exc: BaseException) -> str:
    rows: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(rows) < 6:
        seen.add(id(current))
        rows.append(f"{type(current).__name__}: {current}")
        nxt = current.__cause__ if current.__cause__ is not None else current.__context__
        current = nxt if isinstance(nxt, BaseException) else None
    return " <- ".join(rows)


def _engine(args):
    pipeline = str(args.pipeline or "ocr").lower()
    if pipeline == "vl":
        from paddleocr import PaddleOCRVL
        return PaddleOCRVL(
            pipeline_version=args.pipeline_version or "v1.6",
            vl_rec_backend="native",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=True,
            use_chart_recognition=False,
            use_seal_recognition=False,
            use_ocr_for_image_block=False,
            format_block_content=True,
            merge_layout_blocks=True,
        )
    if pipeline == "structure":
        from paddleocr import PPStructureV3
        # Manga pages do not need table/formula/chart/seal models. Keeping these
        # modules disabled avoids downloading unrelated weights and lowers memory.
        kwargs: dict[str, Any] = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "use_region_detection": False,
            "use_table_recognition": False,
            "use_formula_recognition": False,
            "use_seal_recognition": False,
            "use_chart_recognition": False,
        }
        if args.lang:
            kwargs["lang"] = args.lang
        if args.device:
            kwargs["device"] = args.device
        return PPStructureV3(**kwargs)

    from paddleocr import PaddleOCR
    kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    # Explicit model names are authoritative. Do not also pass lang/ocr_version:
    # PaddleOCR 3.x may remap the model pair when convenience selectors are used.
    if args.det_name:
        kwargs["text_detection_model_name"] = args.det_name
    if args.rec_name:
        kwargs["text_recognition_model_name"] = args.rec_name
    if not args.det_name and not args.rec_name:
        kwargs["lang"] = args.lang
        kwargs["ocr_version"] = args.ocr_version
    if args.det_dir:
        kwargs["text_detection_model_dir"] = args.det_dir
    if args.rec_dir:
        kwargs["text_recognition_model_dir"] = args.rec_dir
    if args.device:
        kwargs["device"] = args.device
    return PaddleOCR(**kwargs)


def _predict(engine, path: str, pipeline: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    try:
        iterator = engine.predict(path)
        for page_result in iterator:
            try:
                batch = (
                    _rows_from_ocr_result(page_result)
                    if pipeline == "ocr"
                    else _rows_from_parsing_result(page_result, pipeline)
                )
            except Exception as exc:
                raise RuntimeError(
                    f"result_normalization_failed[{pipeline}]: {type(exc).__name__}: {exc}"
                ) from exc
            for row in batch:
                row = dict(row)
                row["id"] = f"{pipeline}-{offset:04d}"
                rows.append(row)
                offset += 1
    except RuntimeError as exc:
        if str(exc).startswith("result_normalization_failed["):
            raise
        raise RuntimeError(
            f"engine_predict_failed[{pipeline}]: {type(exc).__name__}: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"engine_predict_failed[{pipeline}]: {type(exc).__name__}: {exc}"
        ) from exc
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="japan")
    p.add_argument("--pipeline", default="ocr", choices=["ocr", "vl", "structure"])
    p.add_argument("--pipeline-version", default="")
    p.add_argument("--ocr-version", default="PP-OCRv5")
    p.add_argument("--model-profile", default="legacy_v5_auto")
    p.add_argument("--det-name", default="")
    p.add_argument("--rec-name", default="")
    p.add_argument("--det-dir", default="")
    p.add_argument("--rec-dir", default="")
    p.add_argument("--device", default="")
    p.add_argument("--preheat", action="store_true")
    p.add_argument("--probe", action="store_true", help="initialize/download models and exit")
    args = p.parse_args()
    try:
        engine = _engine(args)
    except Exception as exc:
        _emit({"ok": False, "type": "init", "error": _exception_detail(exc)})
        return 2
    _emit({
        "ok": True, "type": "probe" if args.probe else "ready", "pid": os.getpid(),
        "model_source": os.environ.get("PADDLE_PDX_MODEL_SOURCE", "local" if args.det_dir and args.rec_dir else "default"),
        "pipeline": args.pipeline,
        "pipeline_version": args.pipeline_version or None,
        "ocr_version": args.ocr_version, "lang": args.lang,
        "model_profile": "local_dirs" if args.det_dir and args.rec_dir else args.model_profile,
        "text_detection_model_name": args.det_name or None,
        "text_recognition_model_name": args.rec_name or None,
    })
    if args.preheat or args.probe:
        return 0
    for line in __import__("sys").stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "close":
                _emit({"ok": True, "type": "closed"}); return 0
            if req.get("cmd") != "predict":
                raise ValueError("unknown command")
            path = str(req.get("path") or "")
            rows = _predict(engine, path, args.pipeline)
            _emit({"ok": True, "type": "result", "blocks": rows})
        except Exception as exc:
            _emit({
                "ok": False, "type": "result",
                "error": _exception_detail(exc),
                "stage": f"normalize_{args.pipeline}_result",
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
