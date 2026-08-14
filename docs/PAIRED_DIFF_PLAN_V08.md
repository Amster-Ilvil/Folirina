# Paired-Diff v0.8: photographed source → clean scan

## Failure in v0.7

The old page can be a phone photograph while the target is a flat scan. Global registration may still be excellent, but raw intensity residuals are dominated by illumination, page curl, halftone phase and resampling. A whole-page percentile therefore raises the diff threshold, while source/target balloon masks differ by a few edge pixels and fail strict target-coverage gates.

## v0.8 pipeline

1. Global source→target registration remains SIFT/RANSAC affine/homography.
2. Warp the source into target coordinates.
3. Estimate conservative DIS dense optical flow from heavily blurred target/source pages; clamp flow magnitude.
4. Build structural ink maps with CLAHE + adaptive thresholding and compare them with a small morphological tolerance.
5. Restrict candidate mismatch to bright local neighbourhoods to suppress artwork differences.
6. Snap mismatch seeds to target-side enclosed components by treating dark line art as a topological barrier.
7. Score enclosed components by area, rectangular fill, target white ratio, changed-pixel count and change density.
8. Remove accepted bubble masks from the mismatch and run a local-density pass for free text / SFX.
9. Composite from the dense-aligned source using the target mask as the sole write geometry. Preserve target balloon outline with a small inset.
10. Keep source-mask IoU as diagnostics, not a hard rejection condition, because source photography changes edge geometry.

## Real-pair acceptance (2026-08-11 sample)

- registration: OpenCV SIFT + affine
- confidence: 0.9044287622
- inlier ratio: 0.9206060606
- median reprojection error: 0.584 px
- paired regions: 10 bubble/text boxes + 2 free-text/SFX
- target-driven transfer: 12 / 12 applied
- QA errors: 0
- QA warnings: 0

The provided user sample is used only as an external regression input and is not bundled in the source release.
