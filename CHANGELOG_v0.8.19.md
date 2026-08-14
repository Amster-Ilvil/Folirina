# v0.8.19

## Visual Page Manager

- Replaced filename-first page inspection with a thumbnail-first gallery.
- Added target/source thumbnail switching, type-color overlays, filters, visible/selected counts, and synchronized detail preview.
- Added multi-selection and right-click batch classification directly on thumbnails and list rows.
- Added a non-modal side-by-side full-page preview dialog with previous/next navigation, fit, 100%, wheel zoom, panning, and direct classification.
- Double-click on a thumbnail or diagnostic row opens the large paired preview.

## Stability

- Full-resolution preview images are no longer decoded again for ordinary progress/role refreshes.
- Page-management JSON writes are atomic (temporary sibling + fsync + replace), preserving the previous state if replacement fails.
- Existing page admission, auto-skip, stop/cancel, workbench sync, precise-mask and glyph-footprint rescue paths are unchanged and regression-tested.
