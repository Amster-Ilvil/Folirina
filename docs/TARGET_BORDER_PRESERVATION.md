# TARGET Border Preservation

## Why this exists

Rigid white-container transfer intentionally keeps the HD TARGET speech-bubble / narration-box outline. A real B/W Chinese -> colour Japanese regression page exposed a subtle ordering bug: the patch was initially inset from the target outline, but later gap-fill used the full target mask and could grow back into the protected anti-aliased border. This made an existing target border appear darker/thicker.

## v0.9.0a3 contract

1. `_rigid_target_write_envelope()` is the single writable envelope for rigid clear, full source raster patch, and gap-fill.
2. Default protected inset is 2 px for the full-patch path.
3. Gap-fill may never write outside that envelope.
4. When `rigid_container_exact_target_border_restore=true`, the protected target ring is restored byte-for-byte from TARGET after compositing and removed from the write mask.
5. Each rigid record publishes `meta.target_border_preservation` with `protected_pixels`, `changed_before_restore`, and `changed_after_restore`.
6. `changed_after_restore > 0` forces Review.

## Real-pair interpretation

The two highlighted dark lines in the supplied regression screenshot are structural borders already present in the Japanese TARGET. The left line is the upper outline of the speech balloon (thin/halftone because of the colour scan). The right line is the upper border of the rectangular narration box. They must not be erased as Japanese-text residuals. The right border did expose a compositing bug because source pixels had darkened part of the target anti-aliasing; v0.9.0a3 prevents this.
