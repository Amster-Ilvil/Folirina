from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import LetteringConfig
from .geometry import rasterize_polygon
from .models import LetteringResult, TextUnit

_CJK = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uf900-\ufaff]")
_FORBID_LINE_START = set("，。！？：；、）》】」』〉〕］｝…—～!?;:,.%）)")
_FORBID_LINE_END = set("（《【「『〈〔［｛(［【《")


def find_default_font(explicit: str | None = None) -> str:
    candidates = [
        explicit,
        os.environ.get("MHD_FONT"),
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    fc = shutil.which("fc-match")
    if fc:
        try:
            out = subprocess.check_output([fc, "-f", "%{file}\n", "Noto Sans CJK SC"], text=True, timeout=3).splitlines()
            if out and Path(out[0]).exists():
                return out[0]
        except Exception:
            pass
    raise FileNotFoundError("No usable font found. Set lettering.font_path or MHD_FONT.")


def _font_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, stroke_width: int = 0) -> float:
    if not text:
        return 0.0
    box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    return float(box[2] - box[0])


def _is_cjk(text: str) -> bool:
    return len(_CJK.findall(text)) / max(1, len(text)) >= 0.25


def _normalize_for_layout(text: str) -> str:
    # Preserve semantic text; only strip OCR-introduced outer whitespace.
    return text.strip().replace("\r\n", "\n").replace("\r", "\n")


def _balanced_cjk_lines(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int,
    stroke_width: int,
) -> list[str] | None:
    text = text.replace("\n", "")
    n = len(text)
    if n == 0:
        return []
    # Precompute prefix widths approximately by direct substring measurement. Manga bubbles
    # are short, so O(n^3) worst-case remains small and avoids kerning inaccuracies.
    widths = [[0.0] * (n + 1) for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n + 1):
            widths[i][j] = _font_width(draw, text[i:j], font, stroke_width)

    min_lines = max(1, int(ceil(_font_width(draw, text, font, stroke_width) / max(1, max_width))))
    for line_count in range(min_lines, min(max_lines, n) + 1):
        target = _font_width(draw, text, font, stroke_width) / line_count
        inf = 1e30
        dp = np.full((line_count + 1, n + 1), inf, dtype=np.float64)
        prev = np.full((line_count + 1, n + 1), -1, dtype=np.int32)
        dp[0, 0] = 0.0
        for k in range(1, line_count + 1):
            for j in range(1, n + 1):
                # At least one char per remaining line.
                i_min = k - 1
                for i in range(i_min, j):
                    if not np.isfinite(dp[k - 1, i]):
                        continue
                    width = widths[i][j]
                    if width > max_width:
                        continue
                    piece = text[i:j]
                    penalty = ((width - target) / max(1.0, target)) ** 2
                    if piece and piece[0] in _FORBID_LINE_START:
                        penalty += 6.0
                    if piece and piece[-1] in _FORBID_LINE_END:
                        penalty += 6.0
                    if len(piece) == 1 and line_count > 1:
                        penalty += 0.8
                    score = dp[k - 1, i] + penalty
                    if score < dp[k, j]:
                        dp[k, j] = score
                        prev[k, j] = i
        if not np.isfinite(dp[line_count, n]):
            continue
        lines: list[str] = []
        j = n
        for k in range(line_count, 0, -1):
            i = int(prev[k, j])
            if i < 0:
                lines = []
                break
            lines.append(text[i:j])
            j = i
        if lines:
            return list(reversed(lines))
    return None


def _balanced_word_lines(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int,
    stroke_width: int,
) -> list[str] | None:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) > 1:
        lines: list[str] = []
        for p in paragraphs:
            part = _balanced_word_lines(p, draw, font, max_width, max_lines - len(lines), stroke_width)
            if not part:
                return None
            lines.extend(part)
        return lines if len(lines) <= max_lines else None
    words = text.split()
    if not words:
        return []
    lines = []
    cur = words[0]
    for word in words[1:]:
        trial = f"{cur} {word}"
        if _font_width(draw, trial, font, stroke_width) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    if any(_font_width(draw, line, font, stroke_width) > max_width for line in lines) or len(lines) > max_lines:
        return None
    return lines


def _safe_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _safe_center(mask: np.ndarray) -> tuple[float, float]:
    m = cv2.moments((mask > 0).astype(np.uint8))
    if abs(m["m00"]) < 1e-8:
        box = _safe_bbox(mask)
        if box is None:
            return 0.0, 0.0
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def _coverage(text_mask: np.ndarray, safe_mask: np.ndarray) -> float:
    total = cv2.countNonZero(text_mask)
    if total == 0:
        return 0.0
    inside = cv2.countNonZero(cv2.bitwise_and(text_mask, safe_mask))
    return inside / total


def _render_horizontal_candidate(
    shape: tuple[int, int],
    safe_mask: np.ndarray,
    text: str,
    font: ImageFont.FreeTypeFont,
    font_size: int,
    cfg: LetteringConfig,
) -> tuple[np.ndarray, list[str], tuple[int, int, int, int], float] | None:
    h, w = shape
    box = _safe_bbox(safe_mask)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    avail_w, avail_h = x1 - x0, y1 - y0
    if avail_w <= 2 or avail_h <= 2:
        return None
    dummy = Image.new("L", (8, 8), 0)
    draw = ImageDraw.Draw(dummy)
    max_width = max(1, int(avail_w * (1.0 - 2 * cfg.side_padding_ratio)))
    if _is_cjk(text):
        lines = _balanced_cjk_lines(text, draw, font, max_width, cfg.max_lines, cfg.stroke_width)
    else:
        lines = _balanced_word_lines(text, draw, font, max_width, cfg.max_lines, cfg.stroke_width)
    if not lines:
        return None
    sample_box = draw.textbbox((0, 0), "国Ag", font=font, stroke_width=cfg.stroke_width)
    line_h = max(1, sample_box[3] - sample_box[1])
    spacing = max(0, round(font_size * cfg.line_spacing_ratio))
    total_h = line_h * len(lines) + spacing * (len(lines) - 1)
    if total_h > avail_h:
        return None

    cx, cy = _safe_center(safe_mask)
    offsets = [(0, 0), (-4, 0), (4, 0), (0, -4), (0, 4), (-8, 0), (8, 0), (0, -8), (0, 8)]
    best = None
    for ox, oy in offsets:
        canvas = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(canvas)
        top = cy - total_h / 2 + oy
        minx, miny, maxx, maxy = w, h, 0, 0
        for li, line in enumerate(lines):
            tb = d.textbbox((0, 0), line, font=font, stroke_width=cfg.stroke_width)
            lw = tb[2] - tb[0]
            x = round(cx - lw / 2 + ox)
            y = round(top + li * (line_h + spacing) - tb[1])
            d.text((x, y), line, font=font, fill=255, stroke_width=cfg.stroke_width, stroke_fill=255)
            minx, miny = min(minx, x), min(miny, y + tb[1])
            maxx, maxy = max(maxx, x + lw), max(maxy, y + tb[3])
        arr = np.asarray(canvas, dtype=np.uint8)
        cov = _coverage(arr, safe_mask)
        score = cov - 0.0005 * (abs(ox) + abs(oy))
        if best is None or score > best[0]:
            best = (score, arr, (max(0, minx), max(0, miny), min(w, maxx), min(h, maxy)), cov)
    assert best is not None
    return best[1], lines, best[2], best[3]


def _render_vertical_candidate(
    shape: tuple[int, int],
    safe_mask: np.ndarray,
    text: str,
    font: ImageFont.FreeTypeFont,
    font_size: int,
    cfg: LetteringConfig,
) -> tuple[np.ndarray, list[str], tuple[int, int, int, int], float] | None:
    h, w = shape
    box = _safe_bbox(safe_mask)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    avail_w, avail_h = x1 - x0, y1 - y0
    dummy = Image.new("L", (8, 8), 0)
    d0 = ImageDraw.Draw(dummy)
    char_box = d0.textbbox((0, 0), "国", font=font, stroke_width=cfg.stroke_width)
    cell_h = max(1, int((char_box[3] - char_box[1]) * 1.05))
    cell_w = max(1, int((char_box[2] - char_box[0]) * 1.08))
    rows = max(1, avail_h // cell_h)
    text = text.replace("\n", "")
    preferred_cols = int(getattr(cfg, "preferred_columns", 0) or 0)
    if preferred_cols > 0:
        cols = min(cfg.max_lines, max(1, preferred_cols))
        rows = ceil(len(text) / cols) if text else 0
    else:
        cols = ceil(len(text) / rows) if text else 0
        rows = ceil(len(text) / cols) if cols else 0
    if cols == 0 or rows == 0 or cols * cell_w > avail_w or rows * cell_h > avail_h or cols > cfg.max_lines:
        return None
    # Balance chars across columns, filling right-to-left.
    columns = [text[i * rows : (i + 1) * rows] for i in range(cols)]
    cx, cy = _safe_center(safe_mask)
    total_w, total_h = cols * cell_w, rows * cell_h
    canvas = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(canvas)
    left = round(cx - total_w / 2)
    top = round(cy - total_h / 2)
    for ci, column in enumerate(columns):
        x = left + (cols - 1 - ci) * cell_w
        for ri, ch in enumerate(column):
            tb = d.textbbox((0, 0), ch, font=font, stroke_width=cfg.stroke_width)
            cw = tb[2] - tb[0]
            y = top + ri * cell_h - tb[1]
            d.text((x + (cell_w - cw) / 2, y), ch, font=font, fill=255, stroke_width=cfg.stroke_width, stroke_fill=255)
    arr = np.asarray(canvas, dtype=np.uint8)
    cov = _coverage(arr, safe_mask)
    return arr, columns, (max(0, left), max(0, top), min(w, left + total_w), min(h, top + total_h)), cov


def _fit_text_uncropped(
    image_shape: tuple[int, int],
    safe_mask: np.ndarray,
    unit: TextUnit,
    text: str,
    config: LetteringConfig | None = None,
) -> LetteringResult:
    cfg = config or LetteringConfig()

    # v0.8.6: render lettering above target resolution and downsample exactly once.
    # This improves small CJK edge quality without ever resizing the manga page.
    factor = max(1, int(getattr(cfg, "supersample_factor", 1)))
    if factor > 1:
        h, w = image_shape
        hi_mask = cv2.resize(
            safe_mask, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST
        )
        hi_cfg = cfg.model_copy(deep=True) if hasattr(cfg, "model_copy") else cfg.copy(deep=True)
        hi_cfg.supersample_factor = 1
        hi_cfg.min_font_size = max(1, int(cfg.min_font_size) * factor)
        hi_cfg.max_font_size = max(hi_cfg.min_font_size, int(cfg.max_font_size) * factor)
        if getattr(cfg, "preferred_font_size", None):
            hi_cfg.preferred_font_size = max(1, int(round(cfg.preferred_font_size * factor)))
        hi_cfg.stroke_width = max(0, int(cfg.stroke_width) * factor)
        hi = _fit_text_uncropped((h * factor, w * factor), hi_mask, unit, text, hi_cfg)
        if hi.success and hi.text_mask is not None:
            low_mask = cv2.resize(hi.text_mask, (w, h), interpolation=cv2.INTER_LANCZOS4)
            # Keep true grayscale alpha from supersampling; do not threshold it back
            # to a jagged binary mask.
            x0, y0, x1, y1 = hi.bbox
            return LetteringResult(
                unit_id=hi.unit_id,
                text=hi.text,
                font_path=hi.font_path,
                font_size=max(1, int(round(hi.font_size / factor))),
                orientation=hi.orientation,
                lines=hi.lines,
                bbox=(
                    max(0, int(round(x0 / factor))),
                    max(0, int(round(y0 / factor))),
                    min(w, int(round(x1 / factor))),
                    min(h, int(round(y1 / factor))),
                ),
                coverage_inside_safe=hi.coverage_inside_safe,
                success=True,
                reason=hi.reason,
                text_mask=low_mask,
            )
        return LetteringResult(
            unit_id=hi.unit_id, text=hi.text, font_path=hi.font_path,
            font_size=max(1, int(round(hi.font_size / factor))) if hi.font_size else 0,
            orientation=hi.orientation, lines=hi.lines,
            bbox=tuple(int(round(v / factor)) for v in hi.bbox),
            coverage_inside_safe=hi.coverage_inside_safe, success=False,
            reason=hi.reason, text_mask=None,
        )

    font_path = find_default_font(cfg.font_path)
    text = _normalize_for_layout(text)
    if not text:
        return LetteringResult(unit.id, text, font_path, 0, cfg.orientation, [], (0, 0, 0, 0), 0.0, False, "empty_text")
    box = _safe_bbox(safe_mask)
    if box is None:
        return LetteringResult(unit.id, text, font_path, 0, cfg.orientation, [], (0, 0, 0, 0), 0.0, False, "empty_safe_mask")
    aspect = (box[3] - box[1]) / max(1, box[2] - box[0])
    orientation = cfg.orientation
    if orientation == "auto":
        orientation = "vertical" if aspect >= cfg.vertical_aspect_threshold else "horizontal"

    preferred = int(getattr(cfg, "preferred_font_size", 0) or 0)
    if preferred > 0:
        tol = max(0.05, float(getattr(cfg, "preferred_font_tolerance_ratio", 0.22)))
        lo = max(cfg.min_font_size, int(round(preferred * (1.0 - tol))))
        hi = min(cfg.max_font_size, int(round(preferred * (1.0 + tol))))
        preferred = int(np.clip(preferred, lo, hi))
        # Search close to the translated source typography first. Only shrink when
        # needed; never jump directly to the global maximum for a short phrase.
        sizes = [preferred]
        for delta in range(1, max(preferred-lo, hi-preferred)+1):
            down = preferred - delta; up = preferred + delta
            if down >= lo: sizes.append(down)
            if up <= hi: sizes.append(up)
    else:
        sizes = list(range(cfg.max_font_size, cfg.min_font_size - 1, -1))

    for font_size in sizes:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError as e:
            return LetteringResult(unit.id, text, font_path, 0, orientation, [], (0, 0, 0, 0), 0.0, False, f"font_error:{e}")
        if orientation == "vertical":
            candidate = _render_vertical_candidate(image_shape, safe_mask, text, font, font_size, cfg)
        else:
            candidate = _render_horizontal_candidate(image_shape, safe_mask, text, font, font_size, cfg)
        if candidate is None:
            continue
        text_mask, lines, bbox, cov = candidate
        if cov >= cfg.min_safe_coverage:
            return LetteringResult(
                unit_id=unit.id,
                text=text,
                font_path=font_path,
                font_size=font_size,
                orientation=orientation,
                lines=lines,
                bbox=bbox,
                coverage_inside_safe=cov,
                success=True,
                text_mask=text_mask,
            )
    return LetteringResult(
        unit_id=unit.id,
        text=text,
        font_path=font_path,
        font_size=cfg.min_font_size,
        orientation=orientation,
        lines=[],
        bbox=box,
        coverage_inside_safe=0.0,
        success=False,
        reason="no_layout_fits_safe_area",
    )


def fit_text(
    image_shape: tuple[int, int],
    safe_mask: np.ndarray,
    unit: TextUnit,
    text: str,
    config: LetteringConfig | None = None,
) -> LetteringResult:
    """Fit text in a local bubble canvas, then paste the mask back to page space.

    v0.8.6 supersampling rendered a 4x mask for the *entire manga page* for every
    OCR-relettered bubble. On a 1440x2048 page that meant repeatedly allocating
    ~5760x8192 PIL canvases for every font-size candidate, which becomes a severe
    batch bottleneck as soon as Apple Live Text starts returning real text.

    Lettering only needs the safe bubble region. Crop to that region (with a small
    guard), perform all 4x layout work locally, then restore the result to the
    original page coordinates. Visual output is equivalent but memory/time scale
    with the bubble instead of the whole page.
    """
    cfg = config or LetteringConfig()
    h, w = image_shape
    if safe_mask.shape[:2] != (h, w):
        return _fit_text_uncropped(image_shape, safe_mask, unit, text, cfg)
    box = _safe_bbox(safe_mask)
    if box is None:
        return _fit_text_uncropped(image_shape, safe_mask, unit, text, cfg)
    x0, y0, x1, y1 = box
    pad = max(3, int(getattr(cfg, "stroke_width", 0)) * 2 + 2)
    cx0 = max(0, x0 - pad); cy0 = max(0, y0 - pad)
    cx1 = min(w, x1 + pad); cy1 = min(h, y1 + pad)
    cw, ch = cx1 - cx0, cy1 - cy0
    # Tiny/synthetic test canvases and near-full-page free text do not benefit
    # from an extra crop/restore step.
    if cw <= 0 or ch <= 0 or (cw * ch) >= int(h * w * 0.78):
        return _fit_text_uncropped(image_shape, safe_mask, unit, text, cfg)

    local_mask = safe_mask[cy0:cy1, cx0:cx1].copy()
    local = _fit_text_uncropped((ch, cw), local_mask, unit, text, cfg)
    bx0, by0, bx1, by1 = local.bbox
    shifted_bbox = (
        max(0, min(w, bx0 + cx0)),
        max(0, min(h, by0 + cy0)),
        max(0, min(w, bx1 + cx0)),
        max(0, min(h, by1 + cy0)),
    )
    full_mask = None
    if local.success and local.text_mask is not None:
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[cy0:cy1, cx0:cx1] = local.text_mask
    return LetteringResult(
        unit_id=local.unit_id, text=local.text, font_path=local.font_path,
        font_size=local.font_size, orientation=local.orientation, lines=local.lines,
        bbox=shifted_bbox, coverage_inside_safe=local.coverage_inside_safe,
        success=local.success, reason=local.reason, text_mask=full_mask,
    )


def composite_text(image: np.ndarray, result: LetteringResult, config: LetteringConfig | None = None) -> np.ndarray:
    cfg = config or LetteringConfig()
    if not result.success or result.text_mask is None:
        return image.copy()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    layer = Image.new("RGB", pil.size, cfg.fill)
    mask = Image.fromarray(result.text_mask)
    out = Image.composite(layer, pil, mask)
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)


def polygon_safe_mask(unit: TextUnit, shape: tuple[int, int], margin: int = 4) -> np.ndarray:
    mask = rasterize_polygon(unit.polygon, shape)
    if margin > 0 and cv2.countNonZero(mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        eroded = cv2.erode(mask, k)
        if cv2.countNonZero(eroded) > 0:
            mask = eroded
    return mask
