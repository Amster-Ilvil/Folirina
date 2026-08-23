from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from math import ceil
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from fontTools.ttLib import TTFont, TTCollection
except Exception:  # optional; PIL rendering still works without fontTools
    TTFont = None
    TTCollection = None

from .config import LetteringConfig
from .geometry import rasterize_polygon
from .models import LetteringResult, TextUnit

_CJK = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uf900-\ufaff]")
_FORBID_LINE_START = set("，。！？：；、）》】」』〉〕］｝…—～!?;:,.%）)")
_FORBID_LINE_END = set("（《【「『〈〔［｛(［【《")


def _split_font_candidates(explicit: str | None = None) -> list[str]:
    raw: list[str] = []
    if explicit:
        raw.extend(str(explicit).replace("\n", ";").split(";"))
    env = os.environ.get("MHD_FONT")
    if env:
        raw.extend(str(env).replace("\n", ";").split(";"))
    defaults = [
        "sans",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    raw.extend(defaults)
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not item:
            continue
        for token in re.split(r"[|,]", str(item)):
            token = token.strip()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
    return out


def _resolve_font_candidate(candidate: str) -> str | None:
    aliases = {
        "sans": ["Noto Sans CJK SC", "Noto Sans CJK JP", "PingFang SC", "Microsoft YaHei UI", "WenQuanYi Zen Hei"],
        "serif": ["Noto Serif CJK SC", "Noto Serif CJK JP", "Songti SC", "Source Han Serif SC"],
        "rounded": ["Hiragino Maru Gothic ProN", "Arial Rounded MT Bold", "Noto Sans CJK SC"],
        "comic": ["Klee One", "LXGW WenKai", "DFPPOPCorn-W12", "Noto Sans CJK SC"],
    }
    if Path(candidate).exists():
        return str(Path(candidate))
    family_candidates = aliases.get(candidate.strip().lower(), [candidate])
    fc = shutil.which("fc-match")
    for fam in family_candidates:
        if fc:
            try:
                out = subprocess.check_output([fc, "-f", "%{file}\n", fam], text=True, timeout=3).splitlines()
                if out and Path(out[0]).exists():
                    return out[0]
            except Exception:
                pass
    return None


@lru_cache(maxsize=128)
def _font_codepoints(path: str) -> frozenset[int] | None:
    if TTFont is None:
        return None
    try:
        cps: set[int] = set(); coll = None
        # Pillow/ImageFont.truetype(path, size) renders face index 0 unless an
        # explicit index is supplied. Coverage must therefore be checked against
        # that same face; unioning every face in a TTC could falsely claim that
        # the actually rendered face contains a glyph that only exists elsewhere.
        if str(path).lower().endswith((".ttc", ".otc")) and TTCollection is not None:
            coll = TTCollection(path, lazy=True)
            fonts = list(coll.fonts[:1])
        else:
            fonts = [TTFont(path, lazy=True)]
        for font in fonts:
            try:
                cmap = font.getBestCmap() or {}
                cps.update(int(k) for k in cmap.keys())
            except Exception:
                pass
            try:
                font.close()
            except Exception:
                pass
        if coll is not None:
            try: coll.close()
            except Exception: pass
        return frozenset(cps) if cps else None
    except Exception:
        return None


def _font_covers_text(path: str, text: str) -> bool:
    cps = _font_codepoints(path)
    if cps is None:
        return True
    needed = {ord(ch) for ch in str(text or "") if not ch.isspace() and ch not in "\n\r\t"}
    # ASCII punctuation is harmless even when cmap lookup is unusual in collections.
    needed = {cp for cp in needed if cp >= 0x80 or chr(cp).isalnum()}
    return not needed or needed.issubset(cps)


def find_font_for_text(explicit: str | None, text: str) -> str:
    resolved_seen: set[str] = set()
    fallback = None
    for cand in _split_font_candidates(explicit):
        resolved = _resolve_font_candidate(cand)
        if not resolved or not Path(resolved).exists() or resolved in resolved_seen:
            continue
        resolved_seen.add(resolved)
        fallback = fallback or resolved
        if _font_covers_text(resolved, text):
            return resolved
    if fallback:
        return fallback
    raise FileNotFoundError("No usable font found. Set lettering.font_path or MHD_FONT.")


def _layout_anchor(mask: np.ndarray, cfg: LetteringConfig) -> tuple[float, float]:
    box = _safe_bbox(mask)
    if box is None:
        return _safe_center(mask)
    x0, y0, x1, y1 = box
    rx = getattr(cfg, "anchor_x_ratio", None)
    ry = getattr(cfg, "anchor_y_ratio", None)
    if rx is None or ry is None:
        return _safe_center(mask)
    rx = float(np.clip(float(rx), 0.0, 1.0)); ry = float(np.clip(float(ry), 0.0, 1.0))
    return x0 + rx * max(1, x1-x0), y0 + ry * max(1, y1-y0)


def _bbox_shape_penalty(bbox: tuple[int,int,int,int], safe_mask: np.ndarray, cfg: LetteringConfig) -> float:
    safe = _safe_bbox(safe_mask)
    if safe is None:
        return 0.0
    sw = max(1.0, safe[2]-safe[0]); sh = max(1.0, safe[3]-safe[1])
    bw = max(1.0, bbox[2]-bbox[0]); bh = max(1.0, bbox[3]-bbox[1])
    penalty = 0.0
    prw = getattr(cfg, "preferred_bbox_width_ratio", None)
    prh = getattr(cfg, "preferred_bbox_height_ratio", None)
    if prw is not None:
        penalty += abs((bw/sw) - float(prw)) * 0.12
    if prh is not None:
        penalty += abs((bh/sh) - float(prh)) * 0.12
    return float(penalty)


def find_default_font(explicit: str | None = None) -> str:
    for cand in _split_font_candidates(explicit):
        resolved = _resolve_font_candidate(cand)
        if resolved and Path(resolved).exists():
            return resolved
    raise FileNotFoundError("No usable font found. Set lettering.font_path or MHD_FONT.")


def _tracked_text_metrics(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
    stroke_width: int = 0, letter_spacing_px: float = 0.0,
) -> tuple[float, tuple[float, float, float, float], list[float]]:
    """Measure one horizontal line with optional manual tracking.

    Pillow's normal ``textbbox`` includes font kerning but has no tracking
    control.  The live OCR editor exposes letter spacing, so measurement and
    painting must use the exact same per-character advances; otherwise a preview
    can fit while the committed raster overflows.
    """
    if not text:
        return 0.0, (0.0, 0.0, 0.0, 0.0), []
    tracking = float(letter_spacing_px)
    if abs(tracking) < 1e-6:
        box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        return float(box[2] - box[0]), tuple(float(v) for v in box), [0.0]
    offsets: list[float] = []
    cursor = 0.0
    minx = miny = float('inf'); maxx = maxy = float('-inf')
    for index, ch in enumerate(text):
        offsets.append(cursor)
        box = draw.textbbox((cursor, 0), ch, font=font, stroke_width=stroke_width)
        minx=min(minx,float(box[0])); miny=min(miny,float(box[1])); maxx=max(maxx,float(box[2])); maxy=max(maxy,float(box[3]))
        try:
            advance=float(draw.textlength(ch,font=font))
        except Exception:
            advance=float(box[2]-box[0])
        cursor += advance
        if index < len(text)-1:
            cursor += tracking
    if not np.isfinite(minx):
        return 0.0, (0.0,0.0,0.0,0.0), offsets
    return float(maxx-minx), (minx,miny,maxx,maxy), offsets


def _font_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
    stroke_width: int = 0, letter_spacing_px: float = 0.0,
) -> float:
    return _tracked_text_metrics(draw,text,font,stroke_width,letter_spacing_px)[0]


def _draw_horizontal_tracked(
    draw: ImageDraw.ImageDraw, text: str, origin_x: float, origin_y: float,
    font: ImageFont.FreeTypeFont, *, stroke_width: int, letter_spacing_px: float,
) -> tuple[float,float,float,float]:
    width, box, offsets = _tracked_text_metrics(draw,text,font,stroke_width,letter_spacing_px)
    if abs(float(letter_spacing_px)) < 1e-6:
        draw.text((origin_x,origin_y),text,font=font,fill=255,stroke_width=stroke_width,stroke_fill=255)
        return (origin_x+box[0],origin_y+box[1],origin_x+box[2],origin_y+box[3])
    for ch,dx in zip(text,offsets):
        draw.text((origin_x+dx,origin_y),ch,font=font,fill=255,stroke_width=stroke_width,stroke_fill=255)
    return (origin_x+box[0],origin_y+box[1],origin_x+box[2],origin_y+box[3])


def _font_visual_metrics(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    stroke_width: int = 0,
) -> dict[str, int]:
    """Measure the *actual selected face* instead of assuming one CJK sample.

    External fonts often have different side bearings, ascender/descender values
    or glyph boxes than PingFang/Noto.  Deriving horizontal line pitch and
    vertical cells from only ``国`` / ``国Ag`` can therefore make the fit search
    accept a size whose real glyphs overlap after drawing.  Measure every unique
    character used by this block plus a small baseline sample so measurement and
    rendering share the same FreeType face and pixel size.
    """
    chars = list(dict.fromkeys((str(text or "").replace("\n", "") + "国Ag，。！？（）【】")))
    boxes: list[tuple[int, int, int, int]] = []
    for ch in chars:
        if not ch or ch.isspace():
            continue
        try:
            b = draw.textbbox((0, 0), ch, font=font, stroke_width=stroke_width)
        except Exception:
            continue
        if b[2] > b[0] and b[3] > b[1]:
            boxes.append(tuple(int(v) for v in b))
    if not boxes:
        b = draw.textbbox((0, 0), "国Ag", font=font, stroke_width=stroke_width)
        boxes = [tuple(int(v) for v in b)]
    top = min(b[1] for b in boxes); bottom = max(b[3] for b in boxes)
    max_w = max(max(1, b[2] - b[0]) for b in boxes)
    max_h = max(max(1, b[3] - b[1]) for b in boxes)
    union_h = max(1, bottom - top)
    # Keep a small floor tied to requested px size. Some decorative fonts report
    # tiny ink boxes for punctuation-only strings even though their line pitch is
    # normal. This floor is deliberately conservative and does not change normal
    # CJK fonts whose actual glyph boxes are larger.
    requested = max(1, int(getattr(font, "size", 0) or 1))
    return {
        "line_h": max(union_h, int(round(requested * 0.72))),
        "glyph_w": max(max_w, int(round(requested * 0.48))),
        "glyph_h": max(max_h, int(round(requested * 0.72))),
    }


def _is_cjk(text: str) -> bool:
    return len(_CJK.findall(text)) / max(1, len(text)) >= 0.25


def _normalize_for_layout(text: str) -> str:
    # Preserve semantic text; only strip OCR-introduced outer whitespace.
    return text.strip().replace("\r\n", "\n").replace("\r", "\n")


def _split_cjk_phrases(text: str) -> list[str]:
    text = text.replace("\n", "")
    if not text:
        return []
    hard_punct = set("，。！？；：、…,.!?;:")
    close_punct = set("）】》」』〉〕］｝")
    light_attach = set("的吗呢吧啊呀啦哦嘛呗哇")
    units: list[str] = []
    cur = ""
    for idx, ch in enumerate(text):
        cur += ch
        nxt = text[idx + 1] if idx + 1 < len(text) else ""
        boundary = False
        if ch in hard_punct or ch in close_punct:
            boundary = True
        elif len(cur) >= 2 and (ch in "的了着过给和与及并而但却就再又还才都也把被向对从让比跟" or nxt in hard_punct):
            boundary = True
        elif len(cur) >= 4 and nxt:
            boundary = True
        if boundary:
            units.append(cur)
            cur = ""
    if cur:
        if units and len(cur) == 1 and cur in light_attach:
            units[-1] += cur
        else:
            units.append(cur)
    merged: list[str] = []
    for u in units:
        if merged and len(u) == 1 and (u in light_attach or u in close_punct):
            merged[-1] += u
        else:
            merged.append(u)
    return merged


def _line_penalty(piece: str, width: float, target: float, *, line_index: int, line_count: int) -> float:
    penalty = ((width - target) / max(1.0, target)) ** 2
    if piece and piece[0] in _FORBID_LINE_START:
        penalty += 7.5
    if piece and piece[-1] in _FORBID_LINE_END:
        penalty += 7.5
    if len(piece) == 1 and line_count > 1:
        penalty += 1.2
    if line_index == line_count - 1 and width < target * 0.55 and line_count > 1:
        penalty += 0.7
    return penalty


def _phrase_balanced_lines(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int,
    stroke_width: int,
) -> list[str] | None:
    text = text.replace("\n", "")
    units = _split_cjk_phrases(text)
    if not units:
        return []
    if len(units) == 1 and len(units[0]) >= 7:
        return None
    widths = [_font_width(draw, u, font, stroke_width) for u in units]
    if any(w > max_width * 1.15 for w in widths):
        return None
    prefix = [0.0]
    for w in widths:
        prefix.append(prefix[-1] + w)
    space = _font_width(draw, " ", font, stroke_width) * 0.0
    total = prefix[-1] + max(0, len(units)-1) * space
    min_lines = max(1, int(ceil(total / max(1, max_width))))
    m = len(units)
    for line_count in range(min_lines, min(max_lines, m) + 1):
        target = total / line_count
        inf = 1e30
        dp = np.full((line_count + 1, m + 1), inf, dtype=np.float64)
        prev = np.full((line_count + 1, m + 1), -1, dtype=np.int32)
        dp[0, 0] = 0.0
        for k in range(1, line_count + 1):
            for j in range(1, m + 1):
                i_min = k - 1
                for i in range(i_min, j):
                    if not np.isfinite(dp[k - 1, i]):
                        continue
                    piece = "".join(units[i:j])
                    width = _font_width(draw, piece, font, stroke_width)
                    if width > max_width:
                        continue
                    score = dp[k - 1, i] + _line_penalty(piece, width, target, line_index=k-1, line_count=line_count)
                    # Prefer keeping original phrase units intact and balanced.
                    score += 0.02 * max(0, (j - i) - 2)
                    if score < dp[k, j]:
                        dp[k, j] = score
                        prev[k, j] = i
        if not np.isfinite(dp[line_count, m]):
            continue
        lines: list[str] = []
        j = m
        for k in range(line_count, 0, -1):
            i = int(prev[k, j])
            if i < 0:
                lines = []
                break
            lines.append("".join(units[i:j]))
            j = i
        if lines:
            return list(reversed(lines))
    return None


def _balanced_cjk_lines(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int,
    stroke_width: int,
    *,
    phrase_first: bool = True,
) -> list[str] | None:
    text = text.replace("\n", "")
    n = len(text)
    if n == 0:
        return []
    if phrase_first:
        phrase_lines = _phrase_balanced_lines(text, draw, font, max_width, max_lines, stroke_width)
        if phrase_lines:
            return phrase_lines
    # Fallback: character-level DP, similar to the old implementation, used when
    # phrase grouping alone cannot fit a long clause.
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
                i_min = k - 1
                for i in range(i_min, j):
                    if not np.isfinite(dp[k - 1, i]):
                        continue
                    width = widths[i][j]
                    if width > max_width:
                        continue
                    piece = text[i:j]
                    score = dp[k - 1, i] + _line_penalty(piece, width, target, line_index=k-1, line_count=line_count)
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


def _source_lines_if_fit(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int,
    stroke_width: int,
) -> list[str] | None:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) <= 1 or len(lines) > max_lines:
        return None
    if any(_font_width(draw, line, font, stroke_width) > max_width for line in lines):
        return None
    if any(line and line[0] in _FORBID_LINE_START for line in lines):
        return None
    return lines


def _draw_vertical_glyph(
    draw: ImageDraw.ImageDraw,
    ch: str,
    x: float,
    y: float,
    cell_w: int,
    cell_h: int,
    font: ImageFont.FreeTypeFont,
    cfg: LetteringConfig,
) -> None:
    tb = draw.textbbox((0, 0), ch, font=font, stroke_width=cfg.stroke_width)
    cw = tb[2] - tb[0]
    chh = tb[3] - tb[1]
    # Center by visual ink bounds. A non-zero left bearing is common in
    # imported/display fonts and must not shift every glyph within its column.
    px = x + (cell_w - cw) / 2 - tb[0]
    py = y - tb[1]
    if bool(getattr(cfg, "vertical_punctuation", True)):
        if ch in "、，。．":
            px += cell_w * 0.22
            py -= cell_h * 0.20
        elif ch in "：；":
            px += cell_w * 0.10
            py -= cell_h * 0.08
        elif ch in "）】》」』〉〕］｝":
            px += cell_w * 0.10
        elif ch in "（【《「『〈〔［｛":
            px -= cell_w * 0.08
    draw.text((px, py), ch, font=font, fill=255, stroke_width=cfg.stroke_width, stroke_fill=255)



def _runs_1d(valid: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(valid, dtype=bool).reshape(-1)
    if arr.size == 0:
        return []
    padded = np.pad(arr.astype(np.int8), (1, 1), constant_values=0)
    d = np.diff(padded)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b > a]


def _run_near_anchor(runs: list[tuple[int, int]], anchor: float) -> tuple[int, int] | None:
    if not runs:
        return None
    containing = [r for r in runs if r[0] <= anchor < r[1]]
    if containing:
        return max(containing, key=lambda r: r[1] - r[0])
    return max(runs, key=lambda r: (r[1] - r[0]) - 0.35 * abs(((r[0] + r[1]) * 0.5) - anchor))


def _horizontal_safe_span(mask: np.ndarray, y0: int, y1: int, anchor_x: float) -> tuple[int, int] | None:
    h, w = mask.shape[:2]
    y0 = max(0, min(h, int(y0))); y1 = max(0, min(h, int(y1)))
    if y1 <= y0:
        return None
    # A column is usable only when the whole glyph-height band is inside the
    # safe region. This is the mask-aware equivalent of BallonsTranslator's
    # per-line edge tests and prevents curved balloon edges clipping glyphs.
    valid = np.all(mask[y0:y1, :] > 0, axis=0)
    return _run_near_anchor(_runs_1d(valid), anchor_x)


def _vertical_safe_span(mask: np.ndarray, x0: int, x1: int, anchor_y: float) -> tuple[int, int] | None:
    h, w = mask.shape[:2]
    x0 = max(0, min(w, int(x0))); x1 = max(0, min(w, int(x1)))
    if x1 <= x0:
        return None
    valid = np.all(mask[:, x0:x1] > 0, axis=1)
    return _run_near_anchor(_runs_1d(valid), anchor_y)


def _hard_shape_metrics(text_mask: np.ndarray, safe_mask: np.ndarray) -> tuple[float, int, int]:
    core = text_mask > 0
    total = int(np.count_nonzero(core))
    if total <= 0:
        return 0.0, 0, 0
    outside = int(np.count_nonzero(core & (safe_mask <= 0)))
    return float((total - outside) / total), outside, total


def _variable_cjk_lines(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    widths: list[int],
    stroke_width: int,
    measure_cache: dict[tuple[int, int], float] | None = None,
    letter_spacing_px: float = 0.0,
) -> list[str] | None:
    text = text.replace("\n", "")
    n = len(text); line_count = len(widths)
    if n == 0 or line_count <= 0 or line_count > n:
        return None
    inf = 1e30
    dp = np.full((line_count + 1, n + 1), inf, dtype=np.float64)
    prev = np.full((line_count + 1, n + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    # Width measurement is independent of row geometry.  A curved balloon tries
    # several row counts and vertical offsets at the same font size, so share one
    # exact substring cache across those candidates instead of calling PIL for the
    # same CJK slice hundreds of times.  This follows the pre-measured glyph/word
    # length idea used by established manga lettering engines while preserving our
    # final hard-mask validation.
    cache = measure_cache if measure_cache is not None else {}
    for k in range(1, line_count + 1):
        max_w = max(1.0, float(widths[k - 1]))
        target = max_w * 0.80
        min_j = k
        max_j = n - (line_count - k)
        for j in range(min_j, max_j + 1):
            # Keep at least one char in every earlier line.
            for i in range(k - 1, j):
                if not np.isfinite(dp[k - 1, i]):
                    continue
                key = (i, j)
                width = cache.get(key)
                if width is None:
                    width = _font_width(draw, text[i:j], font, stroke_width, letter_spacing_px)
                    cache[key] = width
                if width > max_w:
                    continue
                piece = text[i:j]
                score = dp[k - 1, i] + _line_penalty(piece, width, target, line_index=k - 1, line_count=line_count)
                # Prefer using the available balloon width without forcing one
                # character into a lonely final line.
                fill = width / max_w
                score += (0.78 - min(0.78, fill)) ** 2 * 0.45
                if len(piece) == 1 and line_count > 1:
                    score += 0.8
                if score < dp[k, j]:
                    dp[k, j] = score; prev[k, j] = i
    if not np.isfinite(dp[line_count, n]):
        return None
    out: list[str] = []
    j = n
    for k in range(line_count, 0, -1):
        i = int(prev[k, j])
        if i < 0:
            return None
        out.append(text[i:j]); j = i
    return list(reversed(out))


def _variable_word_lines(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    widths: list[int],
    stroke_width: int,
    measure_cache: dict[tuple[int, int], float] | None = None,
    letter_spacing_px: float = 0.0,
) -> list[str] | None:
    words = [w for w in text.replace("\n", " ").split() if w]
    if not words or len(widths) <= 0 or len(widths) > len(words):
        return None
    L, n = len(widths), len(words)
    inf = 1e30
    dp = np.full((L + 1, n + 1), inf, dtype=np.float64)
    prev = np.full((L + 1, n + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    for k in range(1, L + 1):
        max_w = max(1.0, float(widths[k - 1])); target = max_w * 0.82
        for j in range(k, n - (L - k) + 1):
            for i in range(k - 1, j):
                if not np.isfinite(dp[k - 1, i]):
                    continue
                piece = " ".join(words[i:j])
                key = (i, j)
                width = measure_cache.get(key) if measure_cache is not None else None
                if width is None:
                    width = _font_width(draw, piece, font, stroke_width, letter_spacing_px)
                    if measure_cache is not None:
                        measure_cache[key] = width
                if width > max_w:
                    continue
                score = dp[k - 1, i] + ((width - target) / max(1.0, target)) ** 2
                if score < dp[k, j]:
                    dp[k, j] = score; prev[k, j] = i
    if not np.isfinite(dp[L, n]):
        return None
    out: list[str] = []; j = n
    for k in range(L, 0, -1):
        i = int(prev[k, j])
        if i < 0:
            return None
        out.append(" ".join(words[i:j])); j = i
    return list(reversed(out))


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
    dummy = Image.new("L", (8, 8), 0); measure = ImageDraw.Draw(dummy)
    font_metrics = _font_visual_metrics(measure, text, font, cfg.stroke_width)
    line_h = int(font_metrics["line_h"])
    tracking = float(font_size) * float(np.clip(getattr(cfg,"letter_spacing_ratio",0.0),-0.25,0.80))
    spacing = max(0, round(font_size * cfg.line_spacing_ratio))
    step_h = max(1, line_h + spacing)
    compact = text.replace("\n", "")
    token_count = max(1, len(compact) if _is_cjk(text) else len(text.split()))
    max_lines = min(int(cfg.max_lines), token_count, max(1, (avail_h + spacing) // step_h))
    cx, cy = _layout_anchor(safe_mask, cfg)
    side_pad = max(1, int(round(font_size * max(0.0, float(cfg.side_padding_ratio)))))
    source_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    mode = str(getattr(cfg, "line_break_mode", "smart") or "smart").lower()
    best = None
    # Shared for every line-count/offset candidate at this font size.
    cjk_measure_cache: dict[tuple[int, int], float] = {}
    word_measure_cache: dict[tuple[int, int], float] = {}

    for line_count in range(1, max_lines + 1):
        if mode == "source" and len(source_lines) > 1 and line_count != len(source_lines):
            continue
        total_h = line_h * line_count + spacing * (line_count - 1)
        if total_h > avail_h:
            continue
        base_top = cy - total_h / 2.0
        y_offsets = [0, -step_h * 0.25, step_h * 0.25, -step_h * 0.5, step_h * 0.5]
        for yoff in y_offsets:
            top = int(round(base_top + yoff))
            spans: list[tuple[int, int]] = []
            widths: list[int] = []
            ok = True
            for li in range(line_count):
                ly0 = top + li * step_h
                span = _horizontal_safe_span(safe_mask, ly0, ly0 + line_h, cx)
                if span is None:
                    ok = False; break
                a, b = span
                a += side_pad; b -= side_pad
                if b - a < max(2, int(font_size * 0.55)):
                    ok = False; break
                spans.append((a, b)); widths.append(b - a)
            if not ok:
                continue
            if mode == "source" and len(source_lines) > 1:
                lines = source_lines if all(_font_width(measure, ln, font, cfg.stroke_width, tracking) <= widths[i] for i, ln in enumerate(source_lines)) else None
            elif _is_cjk(text):
                lines = _variable_cjk_lines(
                    text, measure, font, widths, cfg.stroke_width, cjk_measure_cache, tracking
                )
            else:
                lines = _variable_word_lines(
                    text, measure, font, widths, cfg.stroke_width, word_measure_cache, tracking
                )
            if not lines or len(lines) != line_count:
                continue
            canvas = Image.new("L", (w, h), 0); d = ImageDraw.Draw(canvas)
            minx, miny, maxx, maxy = w, h, 0, 0
            utilizations = []
            for li, line in enumerate(lines):
                lw, tb, _ = _tracked_text_metrics(d,line,font,cfg.stroke_width,tracking)
                a, b = spans[li]
                align=str(getattr(cfg,"text_alignment","center") or "center").lower()
                if align == "left":
                    ink_left=float(a)
                elif align == "right":
                    ink_left=float(b)-float(lw)
                else:
                    ink_left=(float(a)+float(b)-float(lw))/2.0
                # Align the real ink box, including negative bearings.
                x=float(ink_left)-float(tb[0])
                y=float(top + li * step_h)-float(tb[1])
                rb=_draw_horizontal_tracked(d,line,x,y,font,stroke_width=cfg.stroke_width,letter_spacing_px=tracking)
                minx=min(minx,int(np.floor(rb[0]))); miny=min(miny,int(np.floor(rb[1])))
                maxx=max(maxx,int(np.ceil(rb[2]))); maxy=max(maxy,int(np.ceil(rb[3])))
                utilizations.append(lw / max(1.0, float(b - a)))
            arr = np.asarray(canvas, dtype=np.uint8)
            cov, outside, total = _hard_shape_metrics(arr, safe_mask)
            if total <= 0 or outside != 0 or cov < max(float(cfg.min_safe_coverage), 0.9999):
                continue
            rbbox = (max(0, minx), max(0, miny), min(w, maxx), min(h, maxy))
            score = float(np.mean(utilizations)) - 0.018 * float(np.std(utilizations))
            score -= 0.001 * abs(float(yoff))
            score -= _bbox_shape_penalty(rbbox, safe_mask, cfg)
            if best is None or score > best[0]:
                best = (score, arr, lines, rbbox, cov)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def _allocate_vertical_counts(total: int, capacities: list[int]) -> list[int] | None:
    cols = len(capacities)
    if cols <= 0 or total < cols or sum(capacities) < total or any(c <= 0 for c in capacities):
        return None
    counts: list[int] = []
    remain = int(total)
    for i, cap in enumerate(capacities):
        left_cols = cols - i
        later_cap = sum(capacities[i + 1:])
        min_here = max(1, remain - later_cap)
        max_here = min(int(cap), remain - (left_cols - 1))
        if max_here < min_here:
            return None
        ideal = int(round(remain / max(1, left_cols)))
        take = int(np.clip(ideal, min_here, max_here))
        counts.append(take); remain -= take
    return counts if remain == 0 else None


def _rebalance_vertical_counts(text: str, counts: list[int], max_rows: int) -> list[int]:
    counts = [int(c) for c in counts if int(c) > 0]
    if not counts or sum(counts) != len(text):
        return counts
    # Avoid an isolated punctuation/one-character final column when a small
    # boundary shift can keep the punctuation with its preceding phrase.
    if len(counts) > 1 and counts[-1] == 1 and counts[-2] >= 3:
        counts[-2] -= 1
        counts[-1] += 1
    # Do not begin a column with closing punctuation. Shift the punctuation back
    # into the previous column whenever capacity allows.
    starts = []
    pos = 0
    for c in counts:
        starts.append(pos)
        pos += c
    for i in range(1, len(counts)):
        start = starts[i]
        if start < len(text) and text[start] in _FORBID_LINE_START and counts[i-1] < max_rows and counts[i] > 1:
            counts[i-1] += 1
            counts[i] -= 1
            for j in range(i, len(starts)):
                starts[j] += 1
    # General single-character column smoothing.
    for i in range(len(counts)-1, 0, -1):
        if counts[i] == 1 and counts[i-1] >= 3 and counts[i] + 1 <= max_rows:
            counts[i-1] -= 1
            counts[i] += 1
    return counts


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
    if avail_w <= 2 or avail_h <= 2:
        return None
    dummy = Image.new("L", (8, 8), 0); d0 = ImageDraw.Draw(dummy)
    font_metrics = _font_visual_metrics(d0, text, font, cfg.stroke_width)
    glyph_h = int(font_metrics["glyph_h"]); glyph_w = int(font_metrics["glyph_w"])
    letter_ratio=float(np.clip(getattr(cfg,"letter_spacing_ratio",0.0),-0.25,0.80))
    cell_h = max(1, int(round(glyph_h * max(0.70, 1.04 + letter_ratio))))
    cell_w = max(1, int(round(glyph_w * max(0.72, 1.08 + float(np.clip(getattr(cfg, "column_spacing_ratio", 0.06),-0.25,0.80))))))
    text = text.replace("\n", "")
    if not text:
        return None
    max_cols = max(1, min(int(cfg.max_lines), avail_w // max(1, cell_w), len(text)))
    preferred_cols = int(getattr(cfg, "preferred_columns", 0) or 0)
    # Source/target hints are preferences. Explore neighbouring counts first, then
    # the full range so curved balloons can choose the largest readable layout.
    est = int(np.clip(int(round(np.sqrt(max(1, len(text)) * max(0.35, avail_w / max(1.0, avail_h))))), 1, max_cols))
    order: list[int] = []
    for c in [preferred_cols, est, est - 1, est + 1]:
        if 1 <= c <= max_cols and c not in order:
            order.append(c)
    for c in range(1, max_cols + 1):
        if c not in order:
            order.append(c)
    cx, cy = _layout_anchor(safe_mask, cfg)
    side_pad = max(1, int(round(font_size * max(0.0, float(cfg.side_padding_ratio)))))
    best = None

    for cols in order:
        total_w = cols * cell_w
        if total_w > avail_w:
            continue
        base_left = cx - total_w / 2.0
        x_offsets = [0, -cell_w * 0.25, cell_w * 0.25, -cell_w * 0.5, cell_w * 0.5]
        for xoff in x_offsets:
            left = int(round(base_left + xoff))
            spans: list[tuple[int, int]] = []
            capacities: list[int] = []
            ok = True
            # Logical vertical CJK order is right-to-left.
            for ci in range(cols):
                xx = left + (cols - 1 - ci) * cell_w
                span = _vertical_safe_span(safe_mask, xx, xx + cell_w, cy)
                if span is None:
                    ok = False; break
                a, b = span
                a += side_pad; b -= side_pad
                cap = max(0, (b - a) // max(1, cell_h))
                if cap <= 0:
                    ok = False; break
                spans.append((a, b)); capacities.append(cap)
            if not ok:
                continue
            counts = _allocate_vertical_counts(len(text), capacities)
            if counts is None:
                continue
            smoothed = _rebalance_vertical_counts(text, counts, max(capacities))
            if len(smoothed) == len(counts) and sum(smoothed) == len(text) and all(0 < smoothed[i] <= capacities[i] for i in range(cols)):
                counts = smoothed
            canvas = Image.new("L", (w, h), 0); d = ImageDraw.Draw(canvas)
            pos = 0; rendered_cols: list[str] = []; tops: list[int] = []
            global_top = int(round(cy - max(counts) * cell_h / 2.0))
            minx, miny, maxx, maxy = w, h, 0, 0
            for ci, cnt in enumerate(counts):
                column = text[pos:pos + cnt]; pos += cnt; rendered_cols.append(column)
                xx = left + (cols - 1 - ci) * cell_w
                a, b = spans[ci]
                align=str(getattr(cfg,"text_alignment","center") or "center").lower()
                if align == "left":
                    desired_top=a
                elif align == "right":
                    desired_top=max(a,b-cnt*cell_h)
                else:
                    desired_top=global_top
                top = int(np.clip(desired_top, a, max(a, b - cnt * cell_h)))
                tops.append(top)
                for ri, ch in enumerate(column):
                    yy = top + ri * cell_h
                    _draw_vertical_glyph(d, ch, xx, yy, cell_w, cell_h, font, cfg)
                minx = min(minx, xx); miny = min(miny, top)
                maxx = max(maxx, xx + cell_w); maxy = max(maxy, top + cnt * cell_h)
            arr = np.asarray(canvas, dtype=np.uint8)
            cov, outside, total = _hard_shape_metrics(arr, safe_mask)
            if total <= 0 or outside != 0 or cov < max(float(cfg.min_safe_coverage), 0.9999):
                continue
            bbox = (max(0, minx), max(0, miny), min(w, maxx), min(h, maxy))
            score = cov
            if preferred_cols > 0:
                score -= 0.030 * abs(cols - preferred_cols)
            score -= 0.010 * float(np.std(np.asarray(counts, dtype=np.float32)))
            score -= 0.002 * float(np.std(np.asarray(tops, dtype=np.float32)) / max(1.0, cell_h))
            score -= 0.001 * abs(float(xoff))
            score -= _bbox_shape_penalty(bbox, safe_mask, cfg)
            if best is None or score > best[0]:
                best = (score, arr, rendered_cols, bbox, cov)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


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
            # Lanczos may create a one-pixel alpha fringe outside the low-res safe
            # mask even though the supersampled glyph itself fitted perfectly.
            # Remove only that resampling fringe; the real glyph core already has
            # an eroded container margin, so this never serves as a substitute for
            # fitting oversized text.
            low_mask[safe_mask <= 0] = 0
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

    font_path = find_font_for_text(cfg.font_path, text)
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
        # v2.0.30: the recovered source pitch can be slightly too large after
        # photo perspective, OCR crop padding, or a tighter HD target balloon.
        # Older code stopped at the tolerance band and returned
        # ``no_layout_fits_safe_area`` even though a modestly smaller setting
        # would have produced a clean, safe result.  Preserve the source-derived
        # size first, then shrink only as a last-resort fit policy.
        if bool(getattr(cfg, "preferred_font_allow_shrink_fallback", True)):
            seen = set(sizes)
            sizes.extend(s for s in range(lo - 1, cfg.min_font_size - 1, -1) if s not in seen)
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


def textbox_safe_mask(
    bubble_safe_mask: np.ndarray | None,
    text_region_mask: np.ndarray | None,
    *,
    orientation: str = "auto",
) -> np.ndarray | None:
    """Constrain a bubble-safe region to the actual text box area.

    HD relettering used the entire bubble safe-mask as the layout canvas. On
    irregular or multi-lobed balloons this makes the rendered Chinese drift away
    from the original Japanese text position, even when OCR/matching is correct.
    Derive a tighter rectangular text box from the cleared Japanese ink region,
    expand it conservatively, then clip it back to the bubble-safe interior.

    If the inferred text region is unavailable or too small, return the original
    bubble-safe mask so existing behaviour is preserved as a fallback.
    """
    base = None if bubble_safe_mask is None else (bubble_safe_mask > 0).astype(np.uint8) * 255
    region = None if text_region_mask is None else (text_region_mask > 0).astype(np.uint8) * 255
    if region is None or cv2.countNonZero(region) == 0:
        return base
    if base is not None and cv2.countNonZero(base) == 0:
        base = None

    box = _safe_bbox(region)
    if box is None:
        return base if base is not None else region
    x0, y0, x1, y1 = box
    tw = max(1, x1 - x0)
    th = max(1, y1 - y0)
    aspect = th / max(1, tw)
    orient = orientation if orientation in {"horizontal", "vertical"} else ("vertical" if aspect >= 1.45 else "horizontal")

    # Vertical manga bubbles need more width than the raw ink strokes suggest;
    # horizontal lines need more height. Keep the other direction tighter so the
    # typeset text stays near the original Japanese centre.
    if orient == "vertical":
        pad_x = max(6, int(round(tw * 0.95)))
        pad_y = max(4, int(round(th * 0.18)))
        min_w = max(22, int(round(tw * 2.2)))
        min_h = max(18, int(round(th * 1.15)))
    else:
        pad_x = max(4, int(round(tw * 0.18)))
        pad_y = max(6, int(round(th * 0.90)))
        min_w = max(18, int(round(tw * 1.15)))
        min_h = max(22, int(round(th * 2.2)))

    cx = int(round((x0 + x1) / 2))
    cy = int(round((y0 + y1) / 2))
    half_w = max(int(np.ceil(min_w / 2)), int(np.ceil(tw / 2)) + pad_x)
    half_h = max(int(np.ceil(min_h / 2)), int(np.ceil(th / 2)) + pad_y)

    out = np.zeros_like(region)
    rx0 = max(0, cx - half_w); ry0 = max(0, cy - half_h)
    rx1 = min(out.shape[1], cx + half_w); ry1 = min(out.shape[0], cy + half_h)
    out[ry0:ry1, rx0:rx1] = 255

    if base is not None:
        out = cv2.bitwise_and(out, base)
    # Always preserve the original text region itself.
    out = cv2.bitwise_or(out, region)
    if base is not None:
        out = cv2.bitwise_and(out, base)

    if cv2.countNonZero(out) == 0:
        return base if base is not None else region
    # Tiny gaps caused by clipping through a curved bubble should not fragment the
    # layout region. Close lightly, then re-clip to the base bubble.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if base is not None:
        out = cv2.bitwise_and(out, base)
    return out if cv2.countNonZero(out) > 0 else (base if base is not None else region)
