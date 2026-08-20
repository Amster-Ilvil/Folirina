from __future__ import annotations

from typing import Any


def as_dict(value: Any, *, bool_key: str | None = None) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, bool) and bool_key:
        return {bool_key: bool(value), "legacy_boolean_flag": True}
    return {}


def as_dict_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [dict(value)]
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(x) for x in value if isinstance(x, dict)]


def as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def as_scalar_list(value: Any) -> list[Any]:
    return [x for x in as_list(value) if not isinstance(x, (dict, list, tuple))]


def normalize_project(value: Any) -> dict[str, Any]:
    root = as_dict(value)
    root["pair"] = as_dict(root.get("pair"))
    root["registration"] = as_dict(root.get("registration"))
    root["artifacts"] = as_dict(root.get("artifacts"))

    meta = as_dict(root.get("meta"))
    meta["direct_patch"] = as_dict(meta.get("direct_patch"), bool_key="used")
    meta["mask_replace"] = as_dict(meta.get("mask_replace"), bool_key="used")
    meta["hybrid"] = as_dict(meta.get("hybrid"), bool_key="used")
    meta["reletter"] = as_dict(meta.get("reletter"), bool_key="used")
    meta["aligned_overlay_reveal"] = as_dict(meta.get("aligned_overlay_reveal"), bool_key="used")
    meta["transparent_bubble_reveal"] = as_dict(meta.get("transparent_bubble_reveal"), bool_key="used")
    meta["qa_summary"] = as_dict(meta.get("qa_summary"))
    meta["review_sync"] = as_dict(meta.get("review_sync"))
    for key in ("auto_applied_match_ids",):
        meta[key] = as_scalar_list(meta.get(key))
    root["meta"] = meta

    for key in ("source_units", "target_units", "target_blocks", "target_bubbles", "matches"):
        root[key] = as_dict_rows(root.get(key))
    return root


def normalize_overrides(value: Any) -> dict[str, Any]:
    root = as_dict(value)
    for key in ("manual_effect_regions", "manual_reletter"):
        root[key] = as_dict_rows(root.get(key))
    for key in ("text_overrides", "match_overrides", "unit_actions"):
        root[key] = as_dict(root.get(key))
    for key in ("restore_target_bubbles", "accept_candidate_targets", "accepted_source_units"):
        root[key] = as_scalar_list(root.get(key))
    return root


def normalize_review_applied(value: Any) -> dict[str, Any]:
    root = as_dict(value)
    for key in ("manual_effect_applied", "manual_reletter_applied", "unresolved_candidates", "unreviewed_recommended"):
        root[key] = as_dict_rows(root.get(key))
    return root


def normalize_route_meta(value: Any) -> dict[str, Any]:
    route = as_dict(value, bool_key="used")
    route["diagnostics"] = as_dict(route.get("diagnostics"))
    for key in ("review_regions", "manual_reletter_required", "manual_effect_candidates"):
        route[key] = as_dict_rows(route.get(key))
    return route

_REVIEW_MUTABLE_FIELDS = (
    "text_overrides", "match_overrides", "accepted_source_units", "unit_actions",
    "page_force_action", "status", "notes",
)


def merge_review_overrides(existing: Any, updates: Any) -> dict[str, Any]:
    """Merge Web/GUI review edits without destroying unrelated manual state.

    The web review UI edits only a subset of review fields. Older code replaced
    the entire JSON document and could silently erase ``manual_effect_regions``
    created by the Qt Reveal workflow. This function is the single non-destructive
    merge contract for those partial updates.
    """
    root = normalize_overrides(existing)
    incoming = as_dict(updates)
    for key in _REVIEW_MUTABLE_FIELDS:
        if key in incoming:
            root[key] = incoming[key]
    return normalize_overrides(root)
