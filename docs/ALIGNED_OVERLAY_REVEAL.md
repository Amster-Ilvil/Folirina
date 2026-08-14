# Aligned Overlay Reveal (Experimental) — v1.2.3

## Contract

`aligned_overlay_reveal` is an opt-in experimental route for near-identical SOURCE Chinese / TARGET HD Japanese pages. It never becomes the default route. TARGET remains the background, texture and colour authority. SOURCE contributes aligned text ink by default.

## Route order

1. Reuse the existing page registration.
2. Apply stricter registration gates (`confidence`, `inlier_ratio`, `reprojection_error`, `spatial_coverage`).
3. Build SOURCE-only and TARGET-only dark-ink seeds after registration.
4. Recover complete small text-like connected components from those conservative seeds.
5. Protect shared long/large structures (panel lines, bubble outlines, artwork contours).
6. Build a local text corridor and progressively increase boundary protection when outer dark density is high.
7. Inpaint only TARGET text erase pixels.
8. Composite only SOURCE text ink by default.
9. Triage every region and the page as SAFE / REVIEW / REJECT.
10. Persist artifacts in Pipeline and commit `final.png` through `result_state`.

## Safety invariants

- `enabled=false` changes no Direct / Mask / Reveal / Auto behaviour.
- Explicit Direct and Mask never silently fall into this route.
- Auto can consider this route only when all three are true: `enabled=true`, `allow_in_auto=true`, `require_explicit_mode=false`. Direct remains higher priority.
- SOURCE background RGB is never authoritative on coloured TARGET pixels.
- The optional full-raster path is restricted to proven near-white / low-saturation local corridors and is inset away from protected structures.
- A failed registration gate returns the untouched TARGET.
- Erase masks are clamped by structural guards and hard per-region / per-page area caps.
- REVIEW / REJECT rows are handed to the existing `manual_effect` / Reveal workflow.

## Main configuration

```json
{
  "aligned_overlay_reveal": {
    "enabled": false,
    "allow_in_auto": false,
    "require_explicit_mode": true,
    "min_registration_confidence": 0.80,
    "max_reprojection_error": 1.8,
    "min_inlier_ratio": 0.65,
    "min_spatial_coverage": 0.50,
    "erase_source": "hybrid",
    "border_protect_px": 2,
    "progressive_inset_steps": 3,
    "max_outer_dark_ratio": 0.22,
    "prefer_source_ink_only": true,
    "allow_full_source_raster_on_white": true,
    "min_target_white_ratio_for_full_raster": 0.82,
    "forbid_full_raster_on_color_target": true,
    "text_corridor_enabled": true,
    "max_erase_area_ratio_per_page": 0.25,
    "max_single_region_area_ratio": 0.10,
    "default_triage": "review",
    "force_review_if_any_color_exposure": true,
    "force_review_if_source_bg_visible": true
  }
}
```

Additional deterministic thresholds are exposed in `config.example.json`; they are isolated to this experimental namespace.

## Artifacts

- `aligned_overlay_reveal_layer.png`: changed pixels as RGBA.
- `aligned_overlay_reveal_mask.png`: final TARGET erase mask.
- `aligned_overlay_reveal_source_ink.png`: final SOURCE ink write mask.
- `aligned_overlay_reveal_regions.png`: SAFE / REVIEW / REJECT diagnostic overlay.
- `aligned_overlay_reveal.json`: registration metrics, per-region policy, triage and manual candidates.
- `target_clear_mask.png`: compatibility alias for review tooling.

## Manual review

Any REVIEW / REJECT region is exported as `manual_effect_candidates` with SOURCE and TARGET bboxes. The existing Reveal editor can override or repair those areas. Manual review continues to use the stable `final_auto.png` baseline and `final.png == final_reviewed.png` synchronization invariant after commit.

## External design references

The implementation is independent. It follows the same broad separation used by established comic-translation tools: detection/mask construction, removal/inpainting, text transfer and manual correction are separate stages. No GPL source code from those projects is copied into this repository.
