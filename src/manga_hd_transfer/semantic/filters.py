from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .schema import SemanticLayoutResult, SemanticBlock


@dataclass(slots=True)
class SemanticCandidateDecision:
    action: str  # KEEP|DROP|REVIEW
    reason: str
    process_overlap: float = 0.0
    ignore_overlap: float = 0.0
    review_overlap: float = 0.0
    matched_block_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "process_overlap": float(self.process_overlap),
            "ignore_overlap": float(self.ignore_overlap),
            "review_overlap": float(self.review_overlap),
            "matched_block_ids": list(self.matched_block_ids),
        }


def _bbox_mask(shape: tuple[int, int], block: SemanticBlock, pad: int = 0) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = block.bbox
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad); x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
    out = np.zeros((h, w), np.uint8)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = 255
    return out


def semantic_action_masks(layout: SemanticLayoutResult, shape: tuple[int, int], *, pad: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    process = np.zeros(shape, np.uint8); ignore = np.zeros(shape, np.uint8); review = np.zeros(shape, np.uint8)
    if not layout.available:
        return process, ignore, review
    for block in layout.blocks:
        bm = _bbox_mask(shape, block, pad=pad)
        if block.action == "PROCESS": process = np.maximum(process, bm)
        elif block.action == "IGNORE": ignore = np.maximum(ignore, bm)
        else: review = np.maximum(review, bm)
    return process, ignore, review


def decide_candidate(mask: np.ndarray, layout: SemanticLayoutResult, *, strategy: str = "auto") -> SemanticCandidateDecision:
    if not layout.available or cv2.countNonZero(mask) <= 0:
        return SemanticCandidateDecision("KEEP", "semantic_unavailable")
    process, ignore, review = semantic_action_masks(layout, mask.shape[:2], pad=5)
    area = max(1, cv2.countNonZero(mask))
    po = cv2.countNonZero(cv2.bitwise_and(mask, process)) / area
    io = cv2.countNonZero(cv2.bitwise_and(mask, ignore)) / area
    ro = cv2.countNonZero(cv2.bitwise_and(mask, review)) / area
    ids = []
    x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8))
    cx, cy = x + w * 0.5, y + h * 0.5
    for b in layout.blocks:
        x0, y0, x1, y1 = b.bbox
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            ids.append(b.id)
    if io >= 0.20 and po < 0.12:
        return SemanticCandidateDecision("DROP", "semantic_ignore_overlap", po, io, ro, tuple(ids))
    if po >= 0.06:
        return SemanticCandidateDecision("KEEP", "semantic_process_support", po, io, ro, tuple(ids))
    if str(strategy).lower() == "strict":
        return SemanticCandidateDecision("DROP", "strict_without_process_support", po, io, ro, tuple(ids))
    if ro >= 0.08:
        return SemanticCandidateDecision("REVIEW", "semantic_review_support", po, io, ro, tuple(ids))
    return SemanticCandidateDecision("KEEP", "semantic_neutral", po, io, ro, tuple(ids))


def constrain_text_only_mask(mask: np.ndarray, layout: SemanticLayoutResult, *, pad: int = 10) -> np.ndarray:
    if not layout.available or cv2.countNonZero(mask) <= 0:
        return mask
    process, _ignore, review = semantic_action_masks(layout, mask.shape[:2], pad=pad)
    allow = cv2.bitwise_or(process, review)
    if cv2.countNonZero(cv2.bitwise_and(mask, allow)) <= 0:
        return mask
    out = cv2.bitwise_and(mask, allow)
    return out if cv2.countNonZero(out) > 0 else mask
