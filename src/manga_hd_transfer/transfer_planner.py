from __future__ import annotations

"""Auditable transfer-route selection for Direct / Mask / fallback workflows."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class TransferDecision:
    requested_mode: str
    strategy: str
    fallback_allowed: bool
    reason: str
    same_page_confidence: float = 0.0
    direct_plan_safe: bool = False
    evidence: dict = field(default_factory=dict)
    force_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "requested_mode": self.requested_mode,
            "strategy": self.strategy,
            "fallback_allowed": bool(self.fallback_allowed),
            "reason": self.reason,
            "same_page_confidence": float(self.same_page_confidence),
            "direct_plan_safe": bool(self.direct_plan_safe),
            "evidence": dict(self.evidence),
            "force_actions": list(self.force_actions),
        }


def choose_transfer_strategy(
    requested_mode: str,
    *,
    same_page: bool,
    same_page_confidence: float,
    direct_plan_available: bool,
    direct_plan_safe: bool,
    secondary_source_available: bool = False,
    secondary_source_selected: bool = False,
    aligned_plan_available: bool = False,
    aligned_plan_safe: bool = False,
    aligned_auto_allowed: bool = False,
) -> TransferDecision:
    mode = str(requested_mode or "auto").strip().lower()
    base_evidence = {
        "same_page": bool(same_page),
        "same_page_confidence": float(same_page_confidence),
        "direct_plan_available": bool(direct_plan_available),
        "direct_plan_safe": bool(direct_plan_safe),
        "secondary_source_available": bool(secondary_source_available),
        "secondary_source_selected": bool(secondary_source_selected),
        "aligned_plan_available": bool(aligned_plan_available),
        "aligned_plan_safe": bool(aligned_plan_safe),
        "aligned_auto_allowed": bool(aligned_auto_allowed),
    }
    common_actions = ["force_direct_patch", "force_mask_replace", "skip_unit"]
    if secondary_source_available and not secondary_source_selected:
        common_actions = ["retry_direct_with_secondary_source"] + common_actions

    if mode == "aligned_overlay_reveal":
        if not same_page:
            return TransferDecision(
                mode, "aligned_overlay_reveal_reject", False, "page_pair_verification_failed",
                same_page_confidence, False, base_evidence, common_actions,
            )
        if not aligned_plan_available:
            return TransferDecision(
                mode, "aligned_overlay_reveal_reject", False, "no_accepted_aligned_overlay_plan",
                same_page_confidence, False, base_evidence, common_actions,
            )
        return TransferDecision(
            mode, "aligned_overlay_reveal", False, "explicit_aligned_overlay_reveal",
            same_page_confidence, bool(aligned_plan_safe), base_evidence,
            ["force_direct_patch", "force_mask_replace", "skip_unit"],
        )

    if mode == "direct_patch":
        if not same_page:
            return TransferDecision(
                mode, "direct_reject", False, "page_pair_verification_failed",
                same_page_confidence, False, base_evidence, common_actions,
            )
        if not direct_plan_available:
            return TransferDecision(
                mode, "direct_reject", False, "no_safe_direct_container_plan",
                same_page_confidence, False, base_evidence, common_actions,
            )
        return TransferDecision(
            mode, "direct_patch", False,
            "explicit_direct_patch_secondary" if secondary_source_selected else "explicit_direct_patch",
            same_page_confidence, direct_plan_safe, base_evidence,
            ["force_mask_replace", "skip_unit"],
        )

    if mode == "auto":
        if same_page and direct_plan_available and direct_plan_safe:
            return TransferDecision(
                mode, "direct_patch", True,
                "safe_secondary_direct_plan" if secondary_source_selected else "safe_source_direct_plan",
                same_page_confidence, True, base_evidence,
                ["force_mask_replace", "skip_unit"],
            )
        if same_page and aligned_auto_allowed and aligned_plan_available:
            return TransferDecision(
                mode, "aligned_overlay_reveal", True,
                "explicitly_allowed_aligned_overlay_fallback",
                same_page_confidence, bool(aligned_plan_safe), base_evidence,
                ["force_direct_patch", "force_mask_replace", "skip_unit"],
            )
        return TransferDecision(
            mode, "mask_replace", True, "direct_not_safe_fallback_to_mask",
            same_page_confidence, direct_plan_safe, base_evidence, common_actions,
        )

    return TransferDecision(
        mode, mode, mode in {"hybrid", "reletter"}, "explicit_mode",
        same_page_confidence, direct_plan_safe, base_evidence, common_actions,
    )
