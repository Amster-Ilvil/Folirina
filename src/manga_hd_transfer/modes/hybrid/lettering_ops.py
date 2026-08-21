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

from ...config import LetteringConfig
from ...geometry import rasterize_polygon
from ...models import LetteringResult, TextUnit

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
        cps: set[int] = set()
        if str(path).lower().endswith((".ttc", ".otc")) and TTCollection is not None:
            coll = TTCollection(path, lazy=True)
            fonts = list(coll.fonts)
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
    px = x + (cell_w - cw) / 2
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
    mode = str(getattr(cfg, "line_break_mode", "smart") or "smart").lower()
    lines = None
    if mode == "source":
        has_explicit_breaks = len([ln for ln in text.split("\n") if ln.strip()]) > 1
        lines = _source_lines_if_fit(text, draw, font, max_width, cfg.max_lines, cfg.stroke_width)
        # If explicit/manual breaks exist, prefer shrinking the font until they
        # fit rather than silently reflowing them at a larger size.
        if has_explicit_breaks and lines is None:
            return None
    if lines is None and _is_cjk(text):
        if mode == "balanced":
            compact = text.replace("\n", "")
            lines = _balanced_cjk_lines(compact, draw, font, max_width, cfg.max_lines, cfg.stroke_width, phrase_first=False)
        else:
            lines = _balanced_cjk_lines(text, draw, font, max_width, cfg.max_lines, cfg.stroke_width, phrase_first=True)
    elif lines is None:
        lines = _balanced_word_lines(text, draw, font, max_width, cfg.max_lines, cfg.stroke_width)
    if not lines:
        return None
    sample_box = draw.textbbox((0, 0), "国Ag", font=font, stroke_width=cfg.stroke_width)
    line_h = max(1, sample_box[3] - sample_box[1])
    spacing = max(0, round(font_size * cfg.line_spacing_ratio))
    total_h = line_h * len(lines) + spacing * (len(lines) - 1)
    if total_h > avail_h:
        return None

    cx, cy = _layout_anchor(safe_mask, cfg)
    step = max(3, int(round(font_size * 0.18)))
    offsets = [(0, 0), (-step, 0), (step, 0), (0, -step), (0, step), (-2*step, 0), (2*step, 0), (0, -2*step), (0, 2*step)]
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
        rbbox = (max(0, minx), max(0, miny), min(w, maxx), min(h, maxy))
        score = cov - 0.0005 * (abs(ox) + abs(oy)) - _bbox_shape_penalty(rbbox, safe_mask, cfg)
        if best is None or score > best[0]:
            best = (score, arr, rbbox, cov)
    assert best is not None
    return best[1], lines, best[2], best[3]


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
    dummy = Image.new("L", (8, 8), 0)
    d0 = ImageDraw.Draw(dummy)
    char_box = d0.textbbox((0, 0), "国", font=font, stroke_width=cfg.stroke_width)
    cell_h = max(1, int((char_box[3] - char_box[1]) * 1.04))
    cell_w = max(1, int((char_box[2] - char_box[0]) * (1.08 + max(0.0, float(getattr(cfg, "column_spacing_ratio", 0.06))))))
    text = text.replace("\n", "")
    if not text:
        return None
    max_rows = max(1, avail_h // cell_h)
    max_cols = max(1, min(cfg.max_lines, avail_w // cell_w))
    if max_rows <= 0 or max_cols <= 0:
        return None
    preferred_cols = int(getattr(cfg, "preferred_columns", 0) or 0)
    est_cols = int(np.clip(int(ceil(len(text) / max(1, max_rows))), 1, max_cols))
    candidate_cols: list[int] = []
    for c in [preferred_cols, est_cols, est_cols-1, est_cols+1, max(1, int(round(np.sqrt(max(1, len(text)) * (avail_w / max(1.0, avail_h))))))]:
        if 1 <= c <= max_cols and c not in candidate_cols:
            candidate_cols.append(c)
    for c in range(1, max_cols + 1):
        if c not in candidate_cols:
            candidate_cols.append(c)
    cx, cy = _layout_anchor(safe_mask, cfg)
    best = None
    phrase_units = _split_cjk_phrases(text) if _is_cjk(text) else list(text)
    for cols in candidate_cols:
        rows = int(ceil(len(text) / cols))
        if rows > max_rows or cols > max_cols:
            continue
        # Column balancing: distribute char counts almost evenly but prefer the
        # original/source columns when available.
        base = len(text) // cols
        rem = len(text) % cols
        counts = [base + (1 if i < rem else 0) for i in range(cols)]
        # Re-balance around phrase boundaries when possible.
        if _is_cjk(text) and len(phrase_units) > 1:
            remaining = [len(u) for u in phrase_units]
            counts = []
            total_left = len(text)
            units_left = remaining[:]
            for col in range(cols):
                target_n = int(round(total_left / max(1, cols - col)))
                acc = 0
                while units_left and (acc < target_n or acc == 0):
                    nxt = units_left[0]
                    if acc > 0 and acc + nxt > max_rows and acc >= max(1, target_n - 1):
                        break
                    acc += units_left.pop(0)
                    if acc >= target_n and acc >= max(1, base):
                        break
                acc = int(np.clip(acc, 1, max_rows))
                counts.append(acc)
                total_left -= acc
            if sum(counts) != len(text):
                counts = [base + (1 if i < rem else 0) for i in range(cols)]
        counts = _rebalance_vertical_counts(text, counts, max_rows)
        if any(cn > max_rows or cn <= 0 for cn in counts):
            continue
        cols = len(counts)
        total_w, total_h = cols * cell_w, max(counts) * cell_h
        if total_w > avail_w or total_h > avail_h:
            continue
        canvas = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(canvas)
        left = round(cx - total_w / 2)
        top = round(cy - total_h / 2)
        pos = 0
        rendered_cols: list[str] = []
        for ci, cnt in enumerate(counts):
            column = text[pos:pos + cnt]
            pos += cnt
            rendered_cols.append(column)
            x = left + (cols - 1 - ci) * cell_w
            for ri, ch in enumerate(column):
                y = top + ri * cell_h
                _draw_vertical_glyph(d, ch, x, y, cell_w, cell_h, font, cfg)
        arr = np.asarray(canvas, dtype=np.uint8)
        cov = _coverage(arr, safe_mask)
        if cov < cfg.min_safe_coverage:
            continue
        score = cov
        if preferred_cols > 0:
            score -= 0.035 * abs(cols - preferred_cols)
        score -= 0.012 * np.std(np.asarray(counts, dtype=np.float32))
        # Prefer larger column counts only when they preserve readable balance.
        score -= 0.0015 * cols
        bbox = (max(0, left), max(0, top), min(w, left + total_w), min(h, top + total_h))
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
