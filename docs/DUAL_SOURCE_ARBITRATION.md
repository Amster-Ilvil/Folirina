# Dual SOURCE Arbitration — v0.9.0-alpha.1

Primary SOURCE remains the translation authority. Secondary SOURCE is only a candidate raster/geometry evidence channel for Direct Patch.

## Candidate gates

A candidate cannot win publication arbitration if any of these fail:

- same-page confidence below `arbitration_min_same_page_confidence`
- registration confidence below `arbitration_min_registration_confidence`
- reprojection error above `arbitration_max_reprojection_error_px`
- Direct plan is not `safe_to_skip_other_paths`

## Score

The score combines:

- same-page confidence
- registration confidence
- reprojection-error score
- accepted Direct containers / candidate coverage
- median boundary alignment distance
- masked SOURCE Laplacian sharpness
- review / content-incomplete / rejected-alignment risk

Identity and geometry carry more weight than sharpness, preventing a sharp wrong or badly registered secondary scan from winning. Exact/near ties prefer primary.

The full evidence list and selected evidence are written to `project.meta.replace_translation.arbitration` and `selected_arbitration_evidence`.
