from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import PairingConfig
from .io_utils import list_images, read_image
from .models import PageFingerprint, PagePair


def _dhash(gray: np.ndarray, hash_size: int = 8) -> int:
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def _hamming64(a: int, b: int) -> float:
    return (a ^ b).bit_count() / 64.0


def _edge_descriptor(gray: np.ndarray) -> list[float]:
    thumb = cv2.resize(gray, (96, 128), interpolation=cv2.INTER_AREA)
    thumb = cv2.GaussianBlur(thumb, (5, 5), 0)
    gx = cv2.Sobel(thumb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(thumb, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    out: list[float] = []
    for yy in range(4):
        for xx in range(4):
            cell = mag[yy * 32 : (yy + 1) * 32, xx * 24 : (xx + 1) * 24]
            out.append(float(np.mean(cell)))
    arr = np.asarray(out, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        arr /= norm
    return arr.tolist()


def fingerprint_image(path: str | Path, index: int) -> PageFingerprint:
    img = read_image(path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blurring/downsampling reduces the influence of different language glyphs.
    smooth = cv2.GaussianBlur(gray, (7, 7), 1.5)
    return PageFingerprint(
        path=str(path),
        index=index,
        width=w,
        height=h,
        dhash=_dhash(smooth),
        edge_hist=_edge_descriptor(smooth),
    )


@dataclass(slots=True)
class PairCost:
    total: float
    hash_cost: float
    aspect_cost: float
    edge_cost: float
    order_cost: float


def fingerprint_cost(
    a: PageFingerprint,
    b: PageFingerprint,
    source_count: int,
    target_count: int,
    config: PairingConfig,
) -> PairCost:
    hash_cost = _hamming64(a.dhash, b.dhash)
    aspect_cost = min(1.0, abs(np.log(max(a.aspect, 1e-6) / max(b.aspect, 1e-6))))
    ea = np.asarray(a.edge_hist, dtype=np.float32)
    eb = np.asarray(b.edge_hist, dtype=np.float32)
    edge_cost = float(min(1.0, np.linalg.norm(ea - eb) / np.sqrt(2.0)))
    pa = a.index / max(1, source_count - 1)
    pb = b.index / max(1, target_count - 1)
    order_cost = min(1.0, abs(pa - pb))
    total = (
        config.hash_weight * hash_cost
        + config.aspect_weight * aspect_cost
        + config.edge_weight * edge_cost
        + config.order_weight * order_cost
    )
    return PairCost(total, hash_cost, aspect_cost, edge_cost, order_cost)


def pair_fingerprints(
    source: list[PageFingerprint], target: list[PageFingerprint], config: PairingConfig | None = None
) -> tuple[list[PagePair], list[int], list[int]]:
    cfg = config or PairingConfig()
    n, m = len(source), len(target)
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))

    costs = [[fingerprint_cost(source[i], target[j], n, m, cfg) for j in range(m)] for i in range(n)]

    # Sequence alignment preserves reading order while allowing missing/extra pages.
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    prev: list[list[tuple[int, int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + cfg.gap_penalty
        prev[i][0] = (i - 1, 0, "skip_source")
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + cfg.gap_penalty
        prev[0][j] = (0, j - 1, "skip_target")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [
                (dp[i - 1, j - 1] + costs[i - 1][j - 1].total, (i - 1, j - 1, "pair")),
                (dp[i - 1, j] + cfg.gap_penalty, (i - 1, j, "skip_source")),
                (dp[i, j - 1] + cfg.gap_penalty, (i, j - 1, "skip_target")),
            ]
            val, step = min(choices, key=lambda x: x[0])
            dp[i, j] = val
            prev[i][j] = step

    pairs_rev: list[tuple[int, int]] = []
    unmatched_source: list[int] = []
    unmatched_target: list[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = prev[i][j]
        if step is None:
            break
        pi, pj, op = step
        if op == "pair":
            pairs_rev.append((i - 1, j - 1))
        elif op == "skip_source":
            unmatched_source.append(i - 1)
        else:
            unmatched_target.append(j - 1)
        i, j = pi, pj

    page_pairs: list[PagePair] = []
    for si, tj in reversed(pairs_rev):
        c = costs[si][tj]
        if c.total > cfg.max_pair_cost:
            unmatched_source.append(si)
            unmatched_target.append(tj)
            continue
        confidence = float(np.clip(1.0 - c.total / max(cfg.max_pair_cost, 1e-6), 0.0, 1.0))
        reasons = [
            f"dhash={c.hash_cost:.3f}",
            f"edge={c.edge_cost:.3f}",
            f"aspect={c.aspect_cost:.3f}",
            f"order={c.order_cost:.3f}",
        ]
        if confidence < cfg.confidence_floor:
            reasons.append("review_recommended")
        page_pairs.append(
            PagePair(
                source_path=source[si].path,
                target_path=target[tj].path,
                source_index=si,
                target_index=tj,
                confidence=confidence,
                score=c.total,
                reasons=reasons,
            )
        )

    return page_pairs, sorted(set(unmatched_source)), sorted(set(unmatched_target))


def pair_directories(
    source_dir: str | Path, target_dir: str | Path, config: PairingConfig | None = None
) -> tuple[list[PagePair], list[str], list[str]]:
    source_paths = list_images(source_dir)
    target_paths = list_images(target_dir)
    source_fps = [fingerprint_image(p, i) for i, p in enumerate(source_paths)]
    target_fps = [fingerprint_image(p, i) for i, p in enumerate(target_paths)]
    pairs, us, ut = pair_fingerprints(source_fps, target_fps, config)
    return pairs, [str(source_paths[i]) for i in us], [str(target_paths[i]) for i in ut]
