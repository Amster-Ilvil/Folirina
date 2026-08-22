from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from typing import Literal




class TransferModeConfig(BaseModel):
    # v2.3: Auto is no longer an active user mode. It remains accepted only for
    # old project compatibility; new projects start from explicit Direct Patch.
    mode: Literal[
        "auto", "direct_patch", "mask_replace", "aligned_overlay_reveal",
        "transparent_bubble_reveal", "hybrid", "reletter",
    ] = "direct_patch"


class RuntimeConfig(BaseModel):
    device: str = "auto"  # auto|mps|cuda|cpu
    cpu_thread_ratio: float = 0.50
    min_cpu_threads: int = 1
    max_cpu_threads: int = 8
    mps_fallback: bool = True
    mps_memory_fraction: float = 0.82
    release_cache_every: int = 8
    # Long-running books should not retain full-page NumPy masks for every finished page.
    detach_completed_page_arrays: bool = True


class BatchConfig(BaseModel):
    resume: bool = True
    stop_on_error: bool = False
    skip_completed: bool = True
    # ``strict`` keeps the historical completed-page fingerprint contract.
    # ``continue`` is the GUI's explicit “继续处理整本” contract: preserve an
    # already-successful page when SOURCE/TARGET paths and transfer mode still
    # match, even if later UI tuning changed the current job fingerprint. This
    # prevents a crash/restart from turning continuation into an accidental
    # from-scratch rerun. Batch config is intentionally outside mode-scoped
    # render identity, so this policy never changes pixels of newly processed pages.
    resume_policy: Literal["strict", "continue"] = "strict"
    prefetch_workers: int = 2
    # Small books may eagerly prefetch completed PageProjects. Larger books use a
    # bounded sliding window so resume I/O/JSON decode overlaps current-page work
    # without materializing the whole book.
    resume_prefetch_page_limit: int = 48
    resume_prefetch_window: int = 16
    # Threaded JSON prefetch only pays off once completed page projects are large
    # enough; tiny metadata files stay on the lower-overhead sequential path.
    resume_prefetch_min_project_bytes: int = 16384
    # Above this size BookProject.pages becomes a disk-backed lazy Sequence. The
    # authoritative pages/<id>/project.json files are streamed into book/QA output.
    stream_book_page_threshold: int = 96
    # One-page CPU/I/O look-ahead. Accelerator/model execution remains strictly
    # single-lane; only SOURCE/TARGET decode for the next page is overlapped.
    # v2.3.58 stability rollback: speculative 4K decode introduced in v2.3.57
    # can block a whole-book worker on a pathological image/codec.  Keep the
    # implementation available for explicit experiments, but production defaults
    # to the proven synchronous page decode path.
    decode_prefetch_enabled: bool = False
    decode_prefetch_pages: int = 1
    decode_prefetch_experimental_opt_in: bool = False
    # Per-page project transactions are already the authoritative resume boundary.
    # Coalesce book-level control files so a 300-1000 page run does not fsync two
    # redundant JSON files after every page. Terminal/cancel/failure states still
    # force an immediate checkpoint.
    checkpoint_every: int = 4
    checkpoint_max_interval_seconds: float = 15.0
    save_manifest_every: int = 8


class CacheConfig(BaseModel):
    enabled: bool = True
    registration: bool = True
    ocr: bool = True
    bubbles: bool = True
    # Stage caches are recomputable. Keep only recent pages so batch disk usage
    # does not grow without bound while retaining fast backtracking near the
    # current page. 0 disables automatic pruning.
    retain_recent_page_caches: int = 8


class PageManagementConfig(BaseModel):
    # v0.8.20: every newly paired page is content by default.  The old geometry-
    # only post-pair scan was too easy to misclassify splash/art pages and also
    # duplicated work already performed by the real transfer pipeline.  Keep the
    # legacy fields for config-file compatibility, but they are disabled by
    # default and are no longer exposed by the Studio UI.
    auto_skip_no_text_boxes: bool = False
    auto_scan_after_pair: bool = False
    auto_skip_min_registration_confidence: float = 0.72
    scan_max_side: int = 1100
    # A page marked as content is only *eligible* for transfer.  When real source
    # OCR/bubble analysis conclusively finds no Chinese speech/narration box, emit
    # the unchanged HD target instead of forcing inpaint/replacement.
    skip_transfer_when_source_has_no_text_boxes: bool = True


class PixelTransferConfigBase(BaseModel):
    # Deprecated in v1.0.6: kept only so older config files still deserialize.
    # Runtime Direct/Mask code ignores this field and never publication-blocks.
    # Former v1.0.5 policy switch documentation:
    # contract.  The user's workflow prioritizes replacement completeness over
    # conservative publication blocking, so the default is OFF.  Geometry still
    # has to be computable, but artwork/colour/IoU/coverage/spill/content QA gates
    # become diagnostics rather than write blockers.
    publication_safety_enabled: bool = False
    # v1.0.7: SOURCE paper/background is never copied into TARGET. Both Mask and
    # Direct transfer lettering only; TARGET artwork/colour remains authoritative.
    text_only_background_preservation: bool = True
    require_source_text: bool = True
    enabled_kinds: list[str] = Field(default_factory=lambda: ["speech", "narration"])
    # Paired-difference extraction: after page registration, detect only regions
    # that were actually changed by the old translation. This can bypass OCR in
    # pure mask-replace mode and is especially precise for same-source editions.
    paired_diff_enabled: bool = True
    paired_diff_skip_ocr: bool = True
    # Do not let structural-diff noise rewrite page furniture such as a small
    # chapter title, running header, copyright mark, or page number near an edge.
    paired_diff_protect_page_furniture: bool = True
    paired_diff_page_furniture_top_ratio: float = 0.10
    paired_diff_page_furniture_bottom_ratio: float = 0.965
    paired_diff_page_furniture_max_width_ratio: float = 0.18
    paired_diff_page_furniture_max_height_ratio: float = 0.12
    paired_diff_clear_thin_edge_text: bool = True
    paired_diff_clear_thin_edge_text_max_width_ratio: float = 0.10
    paired_diff_clear_thin_edge_text_max_height_ratio: float = 0.82
    paired_diff_min_registration_confidence: float = 0.90
    # Photographed old editions need a separate, lower geometry gate. Raw pixel
    # differences are unreliable under glare/white-balance drift, so the fallback
    # trusts target bubble geometry and compares local ink structure instead.
    photo_pair_fallback_enabled: bool = True
    photo_pair_min_registration_confidence: float = 0.78
    photo_pair_noise_floor_trigger: float = 72.0
    photo_pair_target_dark_threshold: int = 185
    photo_pair_border_dilate_px: int = 1
    photo_pair_min_region_ratio: float = 0.001
    photo_pair_max_region_ratio: float = 0.085
    photo_pair_min_side_px: int = 18
    photo_pair_min_fill_ratio: float = 0.30
    photo_pair_min_boundary_dark: float = 0.28
    photo_pair_min_target_dark_density: float = 0.008
    photo_pair_max_aspect_ratio: float = 3.6
    photo_pair_min_source_target_iou: float = 0.42
    # Transfer-time geometry can be slightly looser for photographed pages because
    # the clean target mask is used to erase Japanese text and constrain output.
    photo_pair_min_transfer_iou: float = 0.74
    photo_pair_min_transfer_coverage: float = 0.84
    photo_pair_max_spill_ratio: float = 0.27
    # Photo pages first get deterministic local illumination flattening so we
    # preserve real antialiased Chinese glyph pixels without carrying camera glare.
    # Only regions that remain too soft after normalization are binarized; very
    # small/unsafe glyphs are relettered from OCR instead.
    photo_pair_force_ink_reconstruction: bool = False
    # Publication-oriented photographed-text recovery. Rather than pasting the
    # phone-photo pixels, rebuild only the registered dark ink over the clean HD
    # target paper using a soft antialiased alpha. This removes camera blur/glare
    # and prevents a second photographed bubble outline from being copied.
    photo_pair_crisp_text_enabled: bool = True
    photo_pair_crisp_border_guard_px: int = 3
    photo_pair_crisp_detail_floor: float = 7.0
    photo_pair_crisp_alpha_gamma: float = 1.08
    photo_pair_crisp_unsharp_amount: float = 0.78
    photo_pair_crisp_min_component_area: int = 2
    photo_pair_crisp_max_ink_ratio: float = 0.32
    photo_pair_normalize_text_pixels: bool = True
    photo_pair_normalize_contrast_gain: float = 1.70
    photo_pair_normalize_unsharp_amount: float = 0.32
    photo_pair_normalize_detail_floor: float = 6.0
    photo_pair_normalize_min_relative_sharpness: float = 0.52
    photo_pair_reletter_below_relative_sharpness: float = 0.18
    photo_pair_fallback_reletter_missing: bool = False
    photo_pair_require_ocr_evidence: bool = True
    photo_pair_max_local_scale_change: float = 0.42
    photo_pair_min_ink_change: float = 0.18
    photo_pair_min_ink_density: float = 0.012
    photo_pair_relaxed_min_ink_density: float = 0.006
    photo_pair_relaxed_min_ink_change: float = 0.65
    photo_pair_relaxed_min_boundary_dark: float = 0.52
    photo_pair_relaxed_max_region_ratio: float = 0.012
    photo_pair_source_thresholds: list[int] = Field(default_factory=lambda: [130, 150, 170, 190, 210, 225, 235])
    photo_pair_source_search_radius: int = 70
    photo_pair_max_candidates: int = 48
    photo_pair_large_region_ratio: float = 0.035
    photo_pair_large_min_ink_density: float = 0.030
    # v0.8.22 cross-rendition white-container guard.  Monochrome translated
    # scans paired with a colour master can make pale train walls, signs, eyes or
    # book labels look like speech balloons.  Refine leaky candidates to a true
    # high-white enclosed core, then require speech-like fill/boundary/text ink.
    # This guard is only applied to monochrome->colour photo_pair routing, so it
    # does not weaken ordinary grayscale/photo editions.
    photo_pair_cross_rendition_white_guard_enabled: bool = True
    photo_pair_cross_rendition_white_threshold: int = 235
    photo_pair_cross_rendition_min_white_ratio: float = 0.82
    photo_pair_cross_rendition_min_fill_ratio: float = 0.56
    photo_pair_cross_rendition_min_inner_dark_density: float = 0.012
    photo_pair_cross_rendition_min_ring_dark: float = 0.45
    photo_pair_cross_rendition_refine_min_area_fraction: float = 0.12
    photo_pair_cross_rendition_refine_min_keep_fraction: float = 0.15
    photo_pair_min_direct_side_px: int = 88
    # Small photographed bubbles and slightly under-segmented masks should not
    # be hard-rejected up front. Prefer deterministic ink recovery and allow a
    # tiny source-mask expansion salvage pass before falling back to OCR.
    photo_pair_prefer_ink_below_relative_sharpness: float = 0.92
    photo_pair_prefer_ink_min_gain: float = 1.08
    # When OCR is available on photographed editions, prefer crisp HD re-lettering
    # for matched dialogue bubbles instead of publishing transferred camera pixels.
    # Mask transfer remains the geometry/identity layer and continues to cover OCR-
    # missing regions; accepted OCR matches simply inpaint over those bubbles and
    # redraw the Chinese text sharply on the HD target page.
    # v0.8.15 mode contract: precise mask replacement must preserve the source
    # glyphs verbatim. OCR may still provide detection/review evidence, but it may
    # not replace punctuation, symbols or wording unless the user explicitly uses
    # Hybrid/Reletter or manual relettering. This gate overrides legacy configs
    # that still contain older OCR-fallback flags.
    strict_mask_replace_no_ocr_reletter: bool = True
    photo_pair_prefer_reletter_with_ocr: bool = False
    photo_pair_prefer_reletter_min_confidence: float = 0.72
    # v0.8.11: OCR is for reading text, not automatically replacing a clean
    # translated scan's original typesetting.  If mask transfer already rebuilt
    # sharp source glyphs at or above this relative-sharpness level, preserve
    # that exact layout instead of re-typesetting from transcript-only OCR.
    photo_pair_preserve_sharp_source_layout: bool = True
    photo_pair_preserve_layout_min_relative_sharpness: float = 1.15
    photo_pair_preserve_layout_clarity_modes: list[str] = Field(default_factory=lambda: [
        "photo-crisp-ink", "photo-recentered-ink", "ink-reconstruction", "pixels",
    ])
    # Experimental vivid-red SFX shape transfer. Disabled by default because
    # photographed stylized glyph shadows can be harder to reconstruct cleanly
    # than ordinary black dialogue; the detector remains available for opt-in tests.
    photo_pair_color_sfx_enabled: bool = False
    # v2.0.93 Koharu semantic completion.  Layout text/bubble detections are
    # first-class transfer candidates even when Paired Diff refuses to treat a
    # coloured/open region as a rigid white container.  The renderer writes
    # registered SOURCE glyph ink only and preserves the TARGET background.
    koharu_semantic_recovery_enabled: bool = True
    koharu_semantic_bubble_min_confidence: float = 0.70
    koharu_semantic_text_min_confidence: float = 0.75
    koharu_semantic_max_existing_overlap: float = 0.28
    koharu_semantic_standalone_min_change_ratio: float = 0.18
    koharu_semantic_include_sfx: bool = False
    photo_pair_salvage_max_expand_px: int = 3
    photo_pair_salvage_coverage_margin: float = 0.08
    photo_pair_salvage_spill_extra: float = 0.04
    # A photographed translation can be physically cropped by the camera/page
    # boundary. A loose global coverage gate must never turn that truncated
    # source into a "successful" full-bubble replacement. If the source
    # container touches an image edge, require substantially higher target
    # coverage; edge-touching bubbles whose text is still complete (for example
    # a tiny bubble with only its outline clipped) continue to pass.
    photo_pair_edge_clip_guard_enabled: bool = True
    photo_pair_edge_clip_margin_px: int = 2
    photo_pair_edge_clip_min_target_coverage: float = 0.94
    # Review-first behaviour for damaged/uncertain translated sources. Instead of
    # leaving Japanese in the automatic preview, publish the recoverable Chinese
    # pixels as a clearly flagged candidate, while preserving one-click restore
    # and manual HD reletter editing in the review workflow.
    photo_pair_low_confidence_candidate_enabled: bool = True
    photo_pair_candidate_min_coverage: float = 0.55
    photo_pair_candidate_clear_target: bool = True
    # Phone photos may have local aspect drift. Never use anisotropic bbox fitting
    # for dialogue glyphs: a uniform local scale + translation protects CJK shapes
    # from being squeezed/stretched while ECC can still refine translation.
    photo_pair_uniform_local_fit: bool = True
    photo_pair_max_axis_scale_delta: float = 0.10
    # v0.8.7 cross-rendition route: black/white translated scans and coloured
    # Japanese masters often have slightly different balloon geometry. For these
    # pairs, trust the clean target container and recover registered Chinese ink
    # directly in target space instead of rejecting the bubble because its source
    # outline is larger/smaller. Additional containers are admitted only when
    # both editions contain text-like ink and the registered ink differs enough.
    # v0.8.29 source-driven fast whole-container route.  For same-layout B/W
    # Chinese -> colour/HD Japanese pages, detect the translated source container
    # once and copy its interior directly.  The target outline is alignment-only:
    # it is never drawn or copied into the final image.  Same-size pages are true
    # same-coordinate copies; different scan resolutions use one global isotropic
    # scale + tiny translation search, never X/Y stretch or OCR relettering.
    source_direct_container_enabled: bool = True
    # v0.8.33: rejected artwork-like candidates are evidence that the cheap
    # source-only plan saw ambiguous page content.  Do not publish a partial
    # direct plan in that case; let target-aware paired recovery handle it.
    source_direct_fail_on_artwork_rejections: bool = False
    # Optional Comic-Translate-style detector fallback: only a local, user-supplied
    # MangaLens/Ultralytics model is used, only on the translated SOURCE page, and
    # only after the cheap contour plan cannot safely finish the page.
    source_direct_detector_fallback_enabled: bool = False
    source_direct_min_registration_confidence: float = 0.82
    # v1.0 real-pair gate: B/W translated screenshots/scans can score slightly
    # below the aggregate confidence threshold even when geometry is excellent.
    # Permit a narrow cross-rendition exception only when independent registration
    # quality evidence (inliers/reprojection/coverage) is also strong.
    source_direct_cross_rendition_relaxed_gate_enabled: bool = True
    source_direct_cross_rendition_min_registration_confidence: float = 0.78
    source_direct_cross_rendition_min_inlier_ratio: float = 0.68
    source_direct_cross_rendition_max_reprojection_error: float = 1.40
    source_direct_cross_rendition_min_spatial_coverage: float = 0.55
    # Never auto-Direct a huge white region on B/W -> colour pairs. Large masks
    # are often panel/skin/censor artwork rather than speech containers.
    source_direct_cross_rendition_max_auto_area_ratio: float = 0.040
    source_direct_min_uniform_scale: float = 0.25
    source_direct_max_uniform_scale: float = 2.50
    # v0.8.30: page registration is now location geometry only. Moderate scan
    # anisotropy/perspective may be used to locate the corresponding container,
    # while final source pixels are always projected back to a local similarity
    # (uniform scale + rotation + translation) so CJK glyphs are never squeezed.
    source_direct_max_axis_scale_delta: float = 0.18
    source_direct_max_rotation_deg: float = 5.0
    source_direct_max_perspective: float = 0.00020
    source_direct_max_local_mapping_anisotropy: float = 0.18
    source_direct_local_similarity_refine: bool = True
    source_direct_local_scale_refine_ratio: float = 0.025
    source_direct_local_angle_refine_deg: float = 0.35
    source_direct_alignment_coarse_step: int = 2
    source_direct_max_source_saturation_p90: float = 35.0
    source_direct_outline_dark_threshold: int = 160
    source_direct_target_edge_threshold: int = 175
    source_direct_min_area_ratio: float = 0.00045
    source_direct_max_area_ratio: float = 0.085
    source_direct_min_side_px: int = 30
    source_direct_max_aspect: float = 4.0
    source_direct_min_contour_fill: float = 0.20
    source_direct_min_source_white_ratio: float = 0.78
    source_direct_min_source_dark_ratio: float = 0.008
    source_direct_max_source_dark_ratio: float = 0.20
    source_direct_spiky_solidity: float = 0.84
    source_direct_spiky_min_vertices: int = 10
    source_direct_spiky_min_dark_ratio: float = 0.025
    # Conservative artwork guard for closed white regions such as shirts/doors.
    # True oval/rectangular text containers are usually convex; lower-solidity
    # shapes must show much denser compact text evidence or they fall back.
    source_direct_white_min_solidity: float = 0.90
    source_direct_low_solidity_min_compact_ratio: float = 0.030
    source_direct_border_inset_px: int = 5
    source_direct_alignment_search_px: int = 16
    source_direct_max_boundary_distance: float = 5.2
    # v0.9.0a7: same-canvas Direct must actually be pixel-exact. Registration
    # estimated from different JP/CN glyphs can contain a small false sub-pixel
    # translation; if the source container outline already sits on the target
    # outline at its original coordinates, lock the raster to those coordinates
    # and do not resample or locally slide it.
    source_direct_identity_lock_enabled: bool = True
    source_direct_identity_lock_boundary_distance: float = 0.85
    # For different-resolution / mildly affine scans, geometry and lettering are
    # intentionally decoupled: the container envelope may follow the registered
    # page geometry while the Chinese raster itself remains similarity-only. This
    # fills tiny white slivers up to the target container without stretching CJK.
    source_direct_geometry_snap_enabled: bool = True
    source_direct_geometry_snap_min_overlap: float = 0.78
    source_direct_geometry_snap_min_area_ratio: float = 0.72
    source_direct_geometry_snap_max_area_ratio: float = 1.38
    source_direct_geometry_snap_edge_distance_px: float = 0.75
    source_direct_geometry_snap_boundary_guard_px: int = 3
    source_direct_geometry_snap_source_ink_threshold: int = 220
    # v1.2.1: final Direct write envelope is independently clamped inside the
    # registered TARGET container. This is intentionally separate from SOURCE
    # outline fitting: source inset prevents importing the old outline, while
    # target inset/line guard prevents clear/draw operations from touching the
    # HD target bubble/text-box border. Protected pixels are restored byte-exact.
    source_direct_target_border_guard_enabled: bool = True
    source_direct_target_border_inset_px: int = 2
    source_direct_target_border_guard_px: int = 3
    source_direct_target_border_edge_distance_px: float = 1.0
    source_direct_exact_target_border_restore: bool = True
    source_direct_target_min_white_ratio: float = 0.58
    source_direct_target_min_dark_ratio: float = 0.004
    source_direct_target_max_dark_ratio: float = 0.24
    source_direct_white_max_high_sat_ratio: float = 0.35
    # v0.8.34.1: when the registered source bubble is slightly smaller than the
    # target balloon, a tiny white ring can remain outside the writable interior
    # and leave Japanese fragments untouched. Expand only into source-paper pixels
    # that stay away from strong target edges, so this fixes residual text without
    # painting across bubble borders or artwork.
    source_direct_white_gap_fill_enabled: bool = True
    source_direct_white_gap_fill_max_px: int = 3
    source_direct_white_gap_fill_source_white_threshold: int = 238
    source_direct_white_gap_fill_target_edge_distance_px: float = 1.35
    source_direct_white_gap_fill_target_value_min: int = 120
    source_direct_white_gap_fill_target_black_threshold: int = 170
    # v1.3.9: confirmed near-solid white speech/narration containers use a
    # full protected-interior TARGET-paper reset before Chinese ink is drawn.
    # This mirrors mature manga editors' "blank balloon then typeset" workflow
    # and removes JP remnants by construction while preserving the HD border.
    white_container_full_clear_enabled: bool = True
    white_container_full_clear_min_paper_ratio: float = 0.68
    white_container_full_clear_max_robust_spread: float = 14.0
    # v1.3.10: write/clear/manual envelopes are explicit configuration rather
    # than hidden hard-coded geometry. TARGET clear may reach the HD border;
    # SOURCE Chinese remains slightly inset.
    white_container_write_inset_px: int = 1
    white_container_write_border_guard_px: int = 1
    white_container_clear_inset_px: int = 0
    white_container_clear_border_guard_px: int = 0
    # v2.3.27: Direct white-bubble clarity enhancement. Rebuild the SOURCE patch
    # as clean paper + SOURCE-derived Chinese ink so low-resolution CN scans do
    # not carry gray paper/noise into high-resolution JP masters.
    direct_white_clarity_enhance_enabled: bool = True
    direct_white_clarity_alpha_gamma: float = 1.0
    direct_white_clarity_black_boost: int = 0
    direct_white_clarity_pure_white_floor: int = 248
    direct_white_clarity_min_text_pixels: int = 18
    white_container_manual_inset_min_px: int = 1
    white_container_manual_inset_max_px: int = 4
    white_container_manual_inset_ratio: float = 0.02
    # Border alignment is geometry-only. Never drag the Chinese raster several
    # pixels just to improve an outline score; page registration remains the
    # text-position authority.
    source_direct_text_raster_shift_limit_enabled: bool = True
    source_direct_text_raster_max_local_shift_px: int = 1
    # Unhinted regions that only look white because they are skin/clothes must
    # never be destructively edited unless the TARGET interior passes the same
    # uniform-paper test used by full-clear.
    source_direct_unhinted_white_requires_full_clear: bool = True
    source_direct_colored_sat_pixel_threshold: int = 80
    source_direct_colored_min_high_sat_ratio: float = 0.65
    source_direct_colored_min_saturation_median: float = 65.0
    source_direct_white_min_ink_change: float = 0.07
    source_direct_colored_min_ink_change: float = 0.18
    # v0.8.31: PanelCleaner-inspired progressive interior fitting.  Geometry and
    # visible border are separate masks; choose the smallest safe inward offset
    # that excludes the source outline instead of copying a fixed-width border.
    source_direct_dynamic_border_enabled: bool = True
    source_direct_dynamic_border_max_px: int = 10
    source_direct_dynamic_border_dark_ratio: float = 0.22
    source_direct_progressive_inset_steps: int = 5
    source_direct_progressive_max_outer_dark_ratio: float = 0.18
    # For saturated jagged containers, preserve the HD target fill/colour. Clear
    # target lettering without OCR, then transfer only the source Chinese ink.
    # White containers still use full-interior hard overwrite for zero residuals.
    source_direct_colored_preserve_target_fill: bool = True
    source_direct_colored_clear_dark_threshold: int = 185
    source_direct_colored_clear_dilate_px: int = 3
    # v1.3.3 regression restore: retain the v1.3.0 component-local AA fringe
    # cleanup and post-composite residual sweep on coloured containers.
    source_direct_colored_antialias_expand_px: int = 2
    source_direct_colored_antialias_value_margin: int = 10
    source_direct_colored_antialias_max_saturation: int = 92
    source_direct_colored_residual_cleanup_enabled: bool = True
    source_direct_colored_residual_expand_px: int = 2
    # Cotrans-inspired connected-component cleanup for coloured SFX/dialogue:
    # keep compact dark glyph components, reject very large decorative blobs,
    # then dilate only the retained text-like clear mask.
    source_direct_colored_component_refine_enabled: bool = True
    source_direct_colored_component_min_area_px: int = 2
    source_direct_colored_component_max_area_ratio: float = 0.12
    source_direct_colored_inpaint_radius: float = 3.0
    source_direct_source_ink_threshold: int = 215
    source_direct_source_ink_dilate_px: int = 1
    source_direct_source_ink_alpha_gamma: float = 0.92
    # Cost-aware local fitting: try the page-derived similarity first. Expensive
    # scale/angle probes are only opened when its border score is uncertain.
    source_direct_adaptive_variant_search: bool = True
    source_direct_variant_probe_boundary_distance: float = 0.25
    # Three-way automatic review state. Ambiguous containers are skipped rather
    # than painted; only accepted regions are eligible for the fast no-OCR path.
    source_direct_review_boundary_distance: float = 3.2
    source_direct_reject_boundary_distance: float = 5.2
    # v0.8.32: source text-seed/barrier completion. Compact glyph-like source
    # clusters seed a local barrier flood when pure contour discovery misses a
    # large/locally broken bubble (e.g. starburst connected to sky screentone).
    source_direct_text_seed_completion_enabled: bool = True
    source_direct_text_seed_min_components: int = 6
    source_direct_text_seed_max_candidates: int = 48
    source_direct_text_seed_barrier_max_area_ratio: float = 0.11
    source_direct_text_seed_min_white_ratio: float = 0.72
    source_direct_text_seed_min_dark_ratio: float = 0.02
    source_direct_text_seed_max_dark_ratio: float = 0.24
    source_direct_text_seed_max_area_to_text_ratio: float = 45.0
    source_direct_small_unhinted_min_compact_components: int = 6
    source_direct_small_unhinted_area_ratio: float = 0.0020
    source_direct_artwork_low_solidity_min_ink_change: float = 0.18
    source_direct_colored_max_target_residual_ratio: float = 0.05
    source_direct_colored_min_source_ink_pixels: int = 24
    # Ordered source-only detector provider chain. Built-ins are zero-model;
    # optional Koharu/RT-DETR/MangaLens/SAM2 remain lazy and never run on TARGET
    # in precise mode. Koharu contributes layout evidence only (no OCR).
    source_direct_detector_chain: list[str] = Field(default_factory=lambda: ["koharu_layout", "pseudo_text_barrier", "sidecar", "ysg_obb", "rtdetr_v2", "sam2", "mangalens"])

    # v0.8.23 locked-raster container transfer.  For same-layout translated
    # scans (especially monochrome Chinese -> colour Japanese), geometry is used
    # only to identify the corresponding container.  The source lettering itself
    # is then moved as one raster with a single uniform scale + translation.
    # No affine shear, per-axis resize, OCR reflow, or connected-glyph rebuilding
    # is allowed.  Clean white target containers are cleared as a whole before
    # the source raster is composited, so Japanese remnants cannot survive merely
    # because a character touched the detector's inner mask edge.
    rigid_container_transfer_enabled: bool = True
    rigid_container_min_source_white_ratio: float = 0.78
    rigid_container_min_target_white_ratio: float = 0.75
    rigid_container_min_fill_ratio: float = 0.55
    rigid_container_min_source_dark_ratio: float = 0.020
    rigid_container_min_target_dark_ratio: float = 0.015
    rigid_container_max_dark_ratio: float = 0.30
    rigid_container_max_source_saturation_p90: float = 28.0
    rigid_container_max_target_saturation_median: float = 36.0
    # v0.8.28: white burst / jagged narration containers can occupy much less
    # of their bounding box than oval/rectangular balloons. Keep the same white,
    # ink and source/target pairing guards, but allow a lower geometric fill.
    rigid_container_spiky_white_enabled: bool = True
    rigid_container_spiky_min_fill_ratio: float = 0.30
    rigid_container_spiky_min_white_ratio: float = 0.78
    rigid_container_spiky_max_aspect: float = 3.5
    rigid_container_spiky_solidity_threshold: float = 0.76
    rigid_container_max_aspect_log_delta: float = 0.16
    rigid_container_min_uniform_scale: float = 0.35
    rigid_container_max_uniform_scale: float = 1.85
    # v2.0.80: a white statistical match is insufficient on mixed-content
    # pages.  Before a rigid container is allowed to copy SOURCE ink, prove that
    # the aligned SOURCE crop contains compact edition-exclusive text support.
    # This rejects face/shirt/highlight false positives that otherwise look like
    # white containers.
    rigid_container_source_text_support_enabled: bool = True
    rigid_container_source_text_tolerance_px: int = 2
    rigid_container_source_text_unique_ratio: float = 0.055
    rigid_container_source_text_max_component_fraction: float = 0.10
    # v2.0.85: changed-text evidence may still contain edition-exclusive artwork
    # edges (helmet rings, glasses, buttons). Reject support masks dominated by
    # a large hollow art component so precise mask / hybrid never upgrade those
    # regions into Chinese patch containers. When the evidence is ambiguous, the
    # page falls back to safer non-rigid routes or simply leaves the Japanese
    # untouched instead of bleaching illustration content.
    rigid_container_support_large_hollow_reject_enabled: bool = True
    rigid_container_support_large_hollow_min_pixels: int = 240
    rigid_container_support_large_hollow_min_fraction: float = 0.45
    rigid_container_support_large_hollow_max_ratio: float = 0.90
    rigid_container_support_large_hollow_min_extent: float = 0.30
    # v2.0.86: enforce the same art-vs-text proof before unseeded white
    # completion pairs are admitted at all, not only immediately before render.
    rigid_container_unseeded_textlike_pair_gate_enabled: bool = True
    # Re-check the *actual placed* source ink after scaling/translation. This is
    # a final renderer-level invariant against future pairing/fallback changes.
    rigid_container_placed_ink_shape_gate_enabled: bool = True
    rigid_container_min_source_text_pixels: int = 96
    rigid_container_min_source_text_ratio: float = 0.01
    rigid_container_source_inset_px: int = 1
    rigid_container_target_inset_px: int = 1
    rigid_container_alpha_floor: float = 0.055
    # v0.8.26: the detector mask is not automatically the true container
    # interior. Japanese/Chinese glyphs that touch the bright-region boundary can
    # cut black notches into it. Reconstruct a solid white-container interior
    # before both clearing and source-raster clipping.
    rigid_container_solidify_enabled: bool = True
    rigid_container_solidify_radius_ratio: float = 0.065
    rigid_container_solidify_min_radius_px: int = 3
    rigid_container_solidify_max_radius_px: int = 12
    rigid_container_solidify_boundary_guard_px: int = 2
    rigid_container_solidify_max_added_ratio: float = 0.24
    # v0.8.24: container-fit acceptance keeps whole white bubbles/textboxes on
    # the rigid raster path even when the inner printable area differs slightly
    # between B/W Chinese and colour Japanese pages.
    rigid_container_min_ink_coverage: float = 0.985
    rigid_container_min_mask_containment: float = 0.955
    rigid_container_acceptance_min_source_coverage: float = 0.985
    rigid_container_acceptance_max_target_residual: float = 0.02
    # v0.8.27: for same-layout B/W Chinese -> colour Japanese pages, overlay the
    # whole source bubble/textbox interior (white background + Chinese text) onto
    # the target container. This avoids residual Japanese and preserves Chinese
    # glyph shapes better than alpha-only lettering compositing.
    rigid_container_full_patch_enabled: bool = True
    rigid_container_full_patch_preserve_target_border: bool = True
    rigid_container_full_patch_target_inset_px: int = 2
    # v0.9.0a3: gap-fill and final raster writes must never grow back across the
    # protected target container outline. After compositing, the protected ring
    # is restored byte-for-byte from TARGET and removed from the write mask.
    rigid_container_exact_target_border_restore: bool = True
    rigid_container_full_patch_mask_blur_px: int = 1
    rigid_container_min_scale_factor: float = 0.88
    rigid_container_scale_step: float = 0.01
    rigid_container_max_shift_px: int = 7
    rigid_container_offset_patience: int = 24
    rigid_container_paper_percentile: float = 90.0
    # v0.8.25: OCR-free completion pass for same-layout B/W -> colour pages.
    # It detects enclosed white containers on both editions, pairs them through
    # page registration, and sends only the matched pair to the rigid raster gate.
    # This recovers dialogue boxes that paired-diff/text detection missed entirely.
    rigid_container_unseeded_completion_enabled: bool = True
    rigid_container_unseeded_min_registration_confidence: float = 0.72
    rigid_container_unseeded_white_threshold: int = 210
    rigid_container_unseeded_max_saturation: int = 70
    rigid_container_unseeded_min_area_ratio: float = 0.0005
    rigid_container_unseeded_max_area_ratio: float = 0.12
    rigid_container_unseeded_min_white_ratio: float = 0.55
    rigid_container_unseeded_min_fill_ratio: float = 0.30
    rigid_container_unseeded_min_dark_ratio: float = 0.012
    rigid_container_unseeded_max_dark_ratio: float = 0.28
    rigid_container_unseeded_max_aspect: float = 5.0
    # v1.3.12: completion must not re-introduce a stricter hard-coded aspect
    # gate after the detector already accepted a tall/narrow narration box.
    rigid_container_unseeded_completion_max_aspect: float = 5.0
    # Short narration boxes may contain only one compact glyph/dash component.
    # Permit that case only when the container itself is a strong, neutral,
    # near-rectangular white box; the registered SOURCE/TARGET ink-change gate
    # still has to pass before any automatic write is allowed.
    rigid_container_unseeded_short_text_rectangle_enabled: bool = True
    rigid_container_unseeded_short_text_min_rect_fill: float = 0.88
    rigid_container_unseeded_short_text_min_white_ratio: float = 0.72
    rigid_container_unseeded_short_text_min_dark_ratio: float = 0.002
    rigid_container_unseeded_min_pair_coverage: float = 0.82
    rigid_container_unseeded_min_pair_iou: float = 0.30
    rigid_container_unseeded_existing_overlap: float = 0.62
    rigid_container_unseeded_max_candidates: int = 96
    # v0.9.0a2 publication gate: an unseeded white region is not automatically
    # text merely because it is white and contains compact dark components. White
    # clothes, signs, windows and paving can satisfy that geometry. Require the
    # registered SOURCE/TARGET ink to differ like translated text before any
    # automatic rigid raster write is allowed.
    rigid_container_unseeded_ink_change_gate_enabled: bool = True
    rigid_container_unseeded_ink_threshold: int = 190
    rigid_container_unseeded_ink_match_tolerance_px: int = 2
    rigid_container_unseeded_min_source_ink_density: float = 0.025
    rigid_container_unseeded_min_target_ink_density: float = 0.012
    rigid_container_unseeded_min_ink_change_score: float = 0.08
    rigid_container_unseeded_max_ink_density_ratio: float = 3.5
    photo_pair_target_driven_enabled: bool = True
    photo_pair_relaxed_target_candidates: bool = True
    photo_pair_relaxed_target_min_source_ink: float = 0.018
    photo_pair_relaxed_target_min_target_ink: float = 0.008
    photo_pair_relaxed_target_min_change: float = 0.35
    photo_pair_relaxed_target_max_dark_density: float = 0.22
    photo_pair_relaxed_target_min_boundary_dark: float = 0.45
    photo_pair_relaxed_target_max_area_ratio: float = 0.035
    photo_pair_relaxed_target_confidence_cap: float = 0.84
    # Small speech balloons are especially sensitive to cross-edition layout
    # drift: the Chinese glyph block can sit several pixels outside the Japanese
    # target's corresponding text positions. Recover the complete source ink from
    # a slightly wider neighbourhood, then uniformly fit the whole glyph block
    # into the clean target balloon. This preserves character shapes and avoids
    # mixed Japanese/Chinese remnants without OCR.
    photo_pair_recenter_small_text_enabled: bool = True
    photo_pair_recenter_max_area_ratio: float = 0.006
    photo_pair_recenter_pad_ratio: float = 0.30
    photo_pair_recenter_fit_ratio: float = 0.88
    photo_pair_recenter_dark_threshold: int = 175
    photo_pair_recenter_max_source_saturation_p90: float = 24.0
    # v0.8.16: source-glyph footprint rescue. Cross-edition bubble geometry can
    # clip only a few strokes (for example the outer half of a small CJK glyph)
    # even when the translated glyph block is complete in the aligned source.
    # Detect that specific mask-boundary loss from source pixels, harvest the
    # complete local text cluster, then move/shrink the *whole raster glyph block*
    # just enough to fit inside the clean target bubble. This never calls OCR and
    # never reflows/retypesets text.
    photo_pair_glyph_rescue_enabled: bool = True
    photo_pair_glyph_rescue_max_area_ratio: float = 0.028
    photo_pair_glyph_rescue_pad_ratio: float = 0.30
    photo_pair_glyph_rescue_cluster_gap_ratio: float = 0.07
    photo_pair_glyph_rescue_min_outside_ink_ratio: float = 0.003
    photo_pair_glyph_rescue_min_coverage: float = 0.995
    photo_pair_glyph_rescue_max_shift_ratio: float = 0.14
    photo_pair_glyph_rescue_min_scale: float = 0.86
    photo_pair_glyph_rescue_scale_step: float = 0.02
    photo_pair_glyph_rescue_safe_inset_px: int = 1
    photo_pair_glyph_rescue_dark_threshold: int = 175
    paired_diff_pixel_threshold: int = 28
    paired_diff_noise_margin: int = 10
    paired_diff_white_threshold: int = 225
    paired_diff_min_component_area: int = 220
    paired_diff_min_region_area: int = 1600
    paired_diff_max_region_ratio: float = 0.12
    paired_diff_dilate_px: int = 8
    paired_diff_close_px: int = 5
    paired_diff_min_mask_iou: float = 0.55
    paired_diff_min_change_density: float = 0.012
    paired_diff_search_radius: int = 70
    # v0.8 compatibility/fallback engine. Kept alongside the v0.8.3 photo_pair
    # route so photographed pages that defeat closed-container detection can still
    # use low-frequency local flow + structural ink/barrier analysis.
    paired_diff_structural_fallback_enabled: bool = True
    paired_diff_structural: bool = True
    paired_diff_ink_tolerance_px: int = 2
    paired_diff_local_flow_enabled: bool = True
    paired_diff_flow_max_side: int = 700
    paired_diff_flow_blur_sigma: float = 4.0
    paired_diff_flow_max_shift_px: float = 18.0
    paired_diff_local_mean_threshold: int = 205
    paired_diff_seed_dilate_px: int = 3
    paired_diff_seed_min_area: int = 12
    paired_diff_barrier_dark_threshold: int = 190
    paired_diff_barrier_dilate_px: int = 1
    paired_diff_min_enclosed_area: int = 800
    paired_diff_min_white_ratio: float = 0.82
    paired_diff_min_rect_fill: float = 0.45
    paired_diff_min_changed_pixels: int = 35
    paired_diff_min_enclosed_change_density: float = 0.008
    paired_diff_free_text_enabled: bool = True
    paired_diff_free_density_window: int = 31
    paired_diff_free_density_threshold: float = 0.020
    paired_diff_free_min_component_area: int = 500
    paired_diff_free_min_changed_pixels: int = 100
    paired_diff_free_min_local_mean: int = 220
    paired_diff_free_exclude_bubble_px: int = 8
    paired_diff_free_mask_dilate_px: int = 2
    # v0.8.21 complex/open text route.  The original structural detector gated
    # changed strokes by near-white local luminance, which misses coloured burst
    # balloons, SFX captions and text printed directly on artwork.  These knobs
    # admit only compact, two-sided text-like ink clusters and keep the write path
    # ink-only so panel art is never replaced by a rectangular crop.
    paired_diff_complex_text_enabled: bool = True
    paired_diff_complex_local_window: int = 31
    paired_diff_complex_min_ink_density: float = 0.014
    paired_diff_complex_max_ink_density: float = 0.42
    paired_diff_complex_min_change_density: float = 0.014
    paired_diff_complex_min_changed_pixels: int = 70
    paired_diff_complex_min_compact_components: int = 3
    paired_diff_complex_max_region_ratio: float = 0.16
    paired_diff_complex_region_pad_ratio: float = 0.16
    paired_diff_complex_clear_dilate_px: int = 2
    paired_diff_complex_group_gap_px: int = 5
    paired_diff_complex_min_source_ink_pixels: int = 16
    paired_diff_complex_min_target_ink_pixels: int = 12
    # v0.9.0a2: structural supplements must prove that SOURCE/TARGET ink identity
    # actually changed. This rejects same-artwork texture islands (roads, foliage,
    # clothing) that differ photometrically after colourization but are not text.
    paired_diff_supplement_ink_identity_gate_enabled: bool = True
    paired_diff_supplement_free_min_ink_change_score: float = 0.45
    paired_diff_supplement_free_min_source_ink_density: float = 0.025
    paired_diff_supplement_complex_min_ink_change_score: float = 0.55
    paired_diff_supplement_bubble_min_ink_change_score: float = 0.35
    paired_diff_supplement_max_ink_density_ratio: float = 3.5
    # v2.0.89: Koharu Layout is renderer authority, not advisory metadata.  When
    # both page sides have layout evidence, paired-diff regions must have
    # region-kind-appropriate semantic support on both sides before rendering.
    # If either layout inference is unavailable the gate fails open so an
    # optional model/runtime failure cannot disable the legacy visual route.
    paired_diff_koharu_layout_safety_gate_enabled: bool = True
    paired_diff_koharu_layout_bubble_min_overlap: float = 0.25
    paired_diff_koharu_layout_text_min_overlap: float = 0.35
    paired_diff_koharu_layout_sfx_min_overlap: float = 0.35
    paired_diff_koharu_layout_panel_only_min_overlap: float = 0.20
    # v0.8.22 saturated burst-balloon route.  A yellow/red coloured burst is
    # treated as a flat-colour text container: clear only target glyph strokes
    # and rebuild only registered source glyph ink.  Never paste the monochrome
    # source background or star outline.
    paired_diff_saturated_container_enabled: bool = True
    paired_diff_saturated_min_saturation: int = 72
    paired_diff_saturated_min_value: int = 160
    paired_diff_saturated_hue_tolerance: int = 10
    paired_diff_saturated_min_pixels: int = 180
    paired_diff_saturated_min_dominant_fraction: float = 0.34
    # The colour component must actually explain the structural text candidate.
    # This prevents nearby blue signs/artwork from hijacking an ordinary white
    # speech balloon on mixed-colour pages such as the real p-066 sample.
    paired_diff_saturated_min_region_overlap_ratio: float = 0.15
    paired_diff_saturated_min_source_bright_ratio: float = 0.75
    paired_diff_saturated_component_close_px: int = 4
    paired_diff_saturated_core_erode_px: int = 5
    paired_diff_saturated_text_dark_threshold: int = 182
    paired_diff_saturated_clear_dilate_px: int = 2
    # v1.3.3 regression restore: recover antialiased TARGET fringe and choose
    # Telea automatically on gradients/halftone instead of flattening them.
    paired_diff_saturated_antialias_expand_px: int = 2
    paired_diff_saturated_antialias_contrast: int = 8
    paired_diff_saturated_antialias_max_saturation: int = 96
    paired_diff_saturated_inpaint_radius: float = 3.0
    paired_diff_saturated_flat_std_threshold: float = 10.0
    paired_diff_saturated_duplicate_overlap: float = 0.82
    paired_diff_structural_min_registration_confidence: float = 0.62
    paired_diff_low_confidence_candidate_threshold: float = 0.64
    # OCR is evidence/geometry only in strict Precise Mask mode. It may recover
    # a region that paired-diff missed, but it never re-typesets final glyphs.
    ocr_guided_component_transfer_enabled: bool = True
    ocr_guided_min_registration_confidence: float = 0.62
    ocr_guided_candidate_min_match_confidence: float = 0.42
    ocr_guided_auto_apply_min_match_confidence: float = 0.64
    ocr_guided_min_ocr_confidence: float = 0.45
    ocr_guided_max_region_ratio: float = 0.22
    ocr_guided_region_pad_ratio: float = 0.14
    ocr_guided_max_existing_overlap: float = 0.18
    paired_diff_target_driven_transfer: bool = True
    paired_diff_target_border_inset_px: int = 2
    exact_identity_copy: bool = True
    exact_identity_translation_px: float = 0.35
    exact_identity_scale_error: float = 0.0025
    exact_identity_mask_iou: float = 0.985
    exact_identity_changed_fringe_px: int = 10
    preserve_target_border: bool = True
    border_inset_px: int = 3
    feather_px: int = 1
    min_match_confidence: float = 0.68
    min_mask_iou: float = 0.80
    min_target_coverage: float = 0.985
    max_spill_ratio: float = 0.04
    local_fit: str = "ecc"  # global|bbox|ecc
    ecc_iterations: int = 80
    ecc_epsilon: float = 1e-5
    max_local_scale_change: float = 0.22
    max_local_translation_ratio: float = 0.16
    # v0.8.34.4: refine the ECC translation with a bounded sub-pixel search on
    # container geometry. This keeps final CJK raster shape unchanged while
    # reducing the 0.5-1px halo/misalignment visible on HD targets.
    local_subpixel_refine_enabled: bool = True
    local_subpixel_step: float = 0.5
    local_subpixel_radius_px: float = 1.0
    local_subpixel_min_iou_gain: float = 0.0015
    # Pixel Enhance sits between raw pixel transfer and binary ink rebuild. It
    # preserves the original Chinese glyph raster/layout, but strengthens soft
    # antialiased edges when SOURCE is lower-resolution than TARGET.
    pixel_enhance_enabled: bool = True
    pixel_enhance_sharpness_trigger: float = 58.0
    pixel_enhance_relative_trigger: float = 0.58
    pixel_enhance_upscale: float = 2.0
    pixel_enhance_unsharp_amount: float = 0.55
    pixel_enhance_unsharp_sigma: float = 0.85
    pixel_enhance_max_darkening: int = 28
    sr_backend: str = "auto"  # auto|torch|lanczos|external|off
    sr_command: str | None = None
    # Shell execution is opt-in. Plain external commands are parsed to argv and
    # launched with shell=False so an untrusted config cannot inject shell syntax.
    sr_allow_shell: bool = False
    sr_model_path: str | None = None
    sr_device: str = "auto"  # auto|mps|cuda|cpu
    sr_precision: str = "fp32"  # fp32|fp16
    sr_tile_size: int = 512
    sr_tile_overlap: int = 24
    sr_fallback_cpu: bool = True
    sr_timeout_seconds: int = 120
    sr_max_scale: float = 4.0
    sr_min_trigger: float = 1.18
    sharpen_amount: float = 0.28
    warn_sharpness_below: float = 18.0
    # v0.8.35 white-bubble fast clear. Clear only target dark glyph pixels,
    # fill with local paper tone, then run tiny Telea only on those glyph pixels.
    # Textured/coloured regions keep the established component-aware path.
    fast_dark_pixel_clear_enabled: bool = True
    fast_dark_pixel_clear_threshold: int = 185
    fast_dark_pixel_clear_min_white_ratio: float = 0.72
    fast_dark_pixel_clear_inpaint_radius: float = 1.5

    # Text fidelity guard. Direct pixel transfer is only safe when the source
    # lettering is already sharp enough. Photographed/blurred editions should
    # never be silently pasted as publication output.
    text_fidelity_mode: str = "auto"  # auto|pixels|ink|reject
    min_pixel_text_sharpness: float = 42.0
    min_relative_text_sharpness: float = 0.34
    ink_reconstruction_enabled: bool = True
    ink_target_white_ratio: float = 0.58
    ink_target_white_threshold: int = 205
    ink_adaptive_block_size: int = 31
    ink_adaptive_c: int = 11
    ink_min_component_area: int = 2
    ink_max_component_area_ratio: float = 0.08
    ink_min_ratio: float = 0.004
    ink_max_ratio: float = 0.34
    reject_blurry_source: bool = True
    fallback_reletter_on_blur: bool = False
    normalize_background: bool = True
    source_mask_expand_px: int = 1
    target_mask_expand_px: int = 0
    # v0.8.34.2: after target-driven mask fitting, slightly grow the final write
    # mask inside a proven safe target envelope to recover source strokes clipped
    # by 1-3px geometry differences and to let source white cover residual target
    # text fragments. Strong target edges remain excluded.
    mask_write_gap_fill_enabled: bool = True
    mask_write_gap_fill_max_px: int = 3
    mask_write_gap_fill_source_white_threshold: int = 238
    mask_write_gap_fill_source_dark_threshold: int = 205
    mask_write_gap_fill_target_edge_distance_px: float = 1.35
    mask_write_gap_fill_target_dark_threshold: int = 185
    reject_if_target_smaller_ratio: float = 0.52
    # Applied means raster write succeeded; these fields make content
    # completeness a separate publication contract.  The post-transfer audit
    # compares registered source ink against the final image and checks whether
    # target-exclusive Japanese ink remains.
    content_completeness_enabled: bool = True
    content_completeness_min_ink_pixels: int = 18
    content_completeness_min_source_coverage: float = 0.90
    content_completeness_max_target_residual: float = 0.10
    content_completeness_tolerance_px: int = 2
    content_completeness_gate_erode_px: int = 4
    # v0.8.34.3: turn completeness QA into a bounded self-repair loop. One extra
    # pass may grow the write mask inside the already trusted target envelope,
    # clear compact target-only glyph remnants, and re-audit before Review.
    content_auto_repair_enabled: bool = True
    content_auto_repair_max_growth_px: int = 5
    content_auto_repair_residual_dilate_px: int = 1
    content_auto_repair_inpaint_radius: float = 2.5
    content_auto_repair_min_gain: float = 0.01
    # Unified per-region publication triage. SAFE means independently verified;
    # REVIEW means useful/reversible but not fully verified; REJECT means no safe
    # automatic publication candidate should be trusted.
    triage_safe_confidence: float = 0.82
    triage_reject_confidence: float = 0.55
    # In Auto mode, a verified pixel/mask result must not be overwritten by OCR
    # reletter merely because an OCR match also exists. REVIEW stays reversible;
    # only REJECT or uncovered regions are eligible for heavy fallback.
    auto_preserve_safe_and_review_pixel_results: bool = True
    save_patch_artifacts: bool = True


class MaskReplaceConfig(PixelTransferConfigBase):
    """Precise-mask mode configuration.

    Mask geometry may use structural/dense alignment as detection evidence, but
    final SOURCE glyph pixels must remain shape-preserving.  The mask decides
    *where* pixels are allowed to write; it must never bend the glyph raster.
    """
    # v2.3.32 renderer ownership is explicit. Shared low-level geometry may be
    # reused, but policy/clarity/failure behaviour must resolve through this
    # namespace only.
    renderer_owner: str = "mask_replace"
    # A failed content audit must never remain visibly published in Precise Mask.
    # Restore the pre-region TARGET and leave an explicit review record instead.
    incomplete_pixel_policy: str = "restore_target"  # restore_target|keep_review

    # v2.3.5 glyph-integrity contract.  Dense DIS flow remains available for
    # candidate/mask discovery, but final Chinese pixels are sampled from the
    # global page registration only.
    paired_diff_dense_flow_geometry_only: bool = True
    paired_diff_render_use_global_raster: bool = True
    paired_diff_forbid_dense_glyph_warp: bool = True
    paired_diff_proxy_warn_iou: float = 0.08
    paired_diff_proxy_warn_area_ratio: float = 0.02

    # v2.3.6 photographed-page integrity contract. A SOURCE crop touching the
    # camera/page edge is incomplete evidence. Precise Mask is a high-fidelity
    # mode, so it must not publish partial bubbles or page-edge free-text noise.
    photo_pair_low_confidence_candidate_enabled: bool = False
    photo_pair_reject_edge_clipped_open_text: bool = True
    photo_pair_reject_edge_clipped_complex_text: bool = True
    photo_pair_edge_clipped_review_required: bool = True
    # v2.3.67: white-bubble Chinese raster enhancement is an explicit opt-in.
    # When disabled, a verified white TARGET bubble keeps the aligned SOURCE
    # Chinese raster and skips optional sharpening / crisp-ink / ink-rebuild
    # ladders. TARGET Japanese clearing and structure protection still run.
    direct_white_clarity_enhance_enabled: bool = False
    direct_white_clarity_alpha_gamma: float = 1.0
    direct_white_clarity_black_boost: int = 0
    direct_white_clarity_pure_white_floor: int = 248
    direct_white_clarity_min_text_pixels: int = 18
    # v2.3.36 Precise-Mask structure protection. SOURCE lettering is allowed to
    # contribute only interior glyph raster; source balloon outlines/tails/burst
    # rays and nearby artwork are filtered independently. TARGET outline/tail is
    # then restored byte-exactly from the HD page.
    mask_source_structure_guard_enabled: bool = True
    mask_source_structure_guard_ratio: float = 0.035
    mask_source_structure_guard_min_px: int = 5
    mask_source_structure_guard_max_px: int = 14
    mask_source_structure_min_component_area: int = 10
    mask_source_structure_min_aspect: float = 2.6
    mask_source_structure_min_span_ratio: float = 0.16
    mask_source_spiky_boundary_band_px: int = 14
    # v2.3.59: SOURCE text authority is derived independently from geometry.
    # A boundary/ray guard may never delete compact SOURCE lettering merely
    # because the glyph became connected to the balloon outline in a scan.
    mask_source_lettering_relief_enabled: bool = True
    mask_source_lettering_relief_dilate_px: int = 1
    mask_target_structure_guard_enabled: bool = True
    mask_target_structure_text_relief_enabled: bool = True
    mask_target_structure_text_relief_dilate_px: int = 1
    mask_target_border_probe_dilate_px: int = 4
    mask_target_border_restore_fringe_px: int = 2
    mask_target_immutable_band_enabled: bool = True
    mask_target_ordinary_inner_band_px: int = 6
    mask_target_spiky_inner_band_px: int = 18
    mask_target_outer_band_px: int = 3
    mask_target_band_text_margin_px: int = 3
    mask_target_spiky_glyph_relief_enabled: bool = True
    mask_target_spiky_glyph_relief_dilate_px: int = 1
    # v2.3.59: true burst/free-text geometry may describe only the paired-diff
    # text lane and contain deep notches.  A validated inner ellipse completes
    # the white burst paper for TARGET-language clearing only; it never expands
    # SOURCE write authority.
    mask_target_spiky_safe_core_ellipse_enabled: bool = True
    mask_target_spiky_safe_core_inset_ratio: float = 0.045
    mask_target_spiky_safe_core_min_inset_px: int = 6
    # v2.3.59: publication guard against SOURCE lettering loss caused by either
    # SOURCE structure filtering or TARGET-side write protection. This measures
    # the pre-guard lettering authority instead of auditing an already-clipped mask.
    mask_raw_source_completeness_enabled: bool = True
    mask_raw_source_min_coverage: float = 0.965


class DirectPatchConfig(PixelTransferConfigBase):
    """Independent Source-Direct configuration namespace.

    The implementation can reuse mature source-container geometry helpers, but
    this config is intentionally *not* the mask_replace namespace. Saved projects
    can therefore tune direct-paste safety without silently changing mask mode.
    """
    strict_no_fallback: bool = True
    require_same_page_precheck: bool = True
    same_page_min_confidence: float = 0.72
    same_page_max_side: int = 720
    same_page_min_valid_ratio: float = 0.45
    # Direct uses SOURCE only as lettering evidence/alpha. TARGET is always the
    # background authority, including ordinary white containers. SOURCE paper,
    # scan gray, colour or texture must never be copied into a colour master.
    allow_target_aware_colored_composite: bool = True
    source_direct_colored_preserve_target_fill: bool = True
    # v1.0.7: Direct no longer copies SOURCE paper/background RGB into a colour
    # master.  It always clears TARGET-only lettering and draws SOURCE-only ink.
    source_direct_text_only: bool = True
    source_direct_text_only_tolerance_px: int = 4
    source_direct_text_only_complete_white_max_region_ratio: float = 0.040
    # A direct route may use homography/affine to locate the matching container,
    # but final CJK raster is still similarity-only.
    source_direct_container_enabled: bool = True
    # Optional GUI recovery mode for unusually small/spiky/weakly seeded
    # containers. Off by default; page-pair and TARGET-background gates remain.
    source_direct_expand_candidate_range: bool = False

    # v2.3.1 direct-mode lock: SOURCE sits on top of TARGET and uses a
    # borderless interior overlay for white bubbles/textboxes. TARGET remains
    # the outline authority, so border lines are never copied from SOURCE.
    direct_borderless_overlay_enabled: bool = True
    direct_source_on_top: bool = True
    direct_remove_source_border_lines: bool = True
    direct_clear_target_text_before_overlay: bool = True
    direct_white_overlay_border_guard_px: int = 2
    # Direct-only rigid completion can receive an interior TARGET mask whose
    # outline itself lies just outside the mask. Probe a narrow dilated ring when
    # detecting HD balloon borders so full-white clearing cannot nick the target
    # outline. Mask/Hybrid configs do not own this option and remain unchanged.
    direct_rigid_target_border_probe_dilate_px: int = 3
    direct_rigid_target_border_restore_fringe_px: int = 1
    # v2.3.31: after a full white-container clear, restore only TARGET dark
    # structural line components that were actually lightened near the outer
    # registered container boundary. This catches partial oval/rule segments the
    # conservative pre-write border detector may miss, without resurrecting
    # central Japanese lettering.
    direct_post_structural_restore_enabled: bool = True
    direct_post_structural_restore_target_dark_max: int = 205
    direct_post_structural_restore_min_lighten: int = 18
    direct_post_structural_restore_edge_ratio_x: float = 0.16
    direct_post_structural_restore_edge_ratio_y: float = 0.12
    direct_post_structural_restore_boundary_band_px: int = 6
    direct_post_structural_restore_min_area_px: int = 10
    direct_post_structural_restore_fringe_px: int = 1
    direct_post_structural_restore_fringe_gray_max: int = 238

    # Direct mode should preserve typesetting orientation. Location can still be
    # refined by registration, but final SOURCE overlay is axis-locked.
    source_direct_local_similarity_refine: bool = True
    source_direct_local_angle_refine_deg: float = 0.0
    source_direct_text_raster_max_local_shift_px: int = 0
    source_direct_axis_lock_rotation: bool = True


class AlignedOverlayRevealConfig(BaseModel):
    """Experimental whole-page registered erase-to-reveal route.

    Default OFF.  The route is deliberately stricter than Direct and keeps
    TARGET as the background/colour authority.  Full SOURCE raster is only a
    tightly gated white-corridor fallback; ink-only transfer is the default.
    """
    enabled: bool = False
    allow_in_auto: bool = False
    require_explicit_mode: bool = True

    min_registration_confidence: float = 0.78
    max_reprojection_error: float = 1.8
    min_inlier_ratio: float = 0.65
    min_spatial_coverage: float = 0.50

    erase_source: str = "hybrid"  # target_text_ink|white_bubble_interior|hybrid
    border_protect_px: int = 2
    progressive_inset_steps: int = 3
    max_outer_dark_ratio: float = 0.22

    prefer_source_ink_only: bool = True
    allow_full_source_raster_on_white: bool = True
    min_target_white_ratio_for_full_raster: float = 0.82
    forbid_full_raster_on_color_target: bool = True

    text_corridor_enabled: bool = True
    max_erase_area_ratio_per_page: float = 0.25
    max_single_region_area_ratio: float = 0.10

    default_triage: str = "review"
    force_review_if_any_color_exposure: bool = True
    force_review_if_source_bg_visible: bool = True

    # Deterministic mask/refinement knobs.  These are intentionally conservative
    # and operate only inside the experimental route.
    source_ink_threshold: int = 220
    target_ink_threshold: int = 220
    source_antialias_threshold: int = 238
    source_ink_antialias_px: int = 1
    ink_difference_delta: int = 14
    registration_tolerance_px: int = 3
    erase_dilate_px: int = 1
    text_corridor_radius_px: int = 4
    region_group_radius_px: int = 8
    region_close_radius_px: int = 2
    region_bbox_pad_px: int = 3
    min_component_area_px: int = 2
    min_region_ink_pixels: int = 10
    max_component_area_ratio: float = 0.03
    max_component_span_ratio: float = 0.40
    white_container_max_area_ratio: float = 0.45
    white_container_search_pad_px: int = 96
    structural_component_min_area_px: int = 900
    structural_component_min_span_ratio: float = 0.08
    structural_component_min_bbox_ratio: float = 0.012
    white_threshold: int = 238
    color_saturation_threshold: int = 55
    review_color_ratio: float = 0.08
    reject_color_ratio: float = 0.35
    max_color_ratio_for_full_raster: float = 0.03
    full_raster_inset_px: int = 2
    safe_white_ratio: float = 0.85
    safe_color_ratio: float = 0.04
    safe_registration_confidence: float = 0.82
    safe_max_region_erase_ratio: float = 0.08
    inpaint_radius: float = 2.5

    # v2.3.38 aligned-hole private TARGET container completion.  Geometry-only
    # supplemental candidates must look like actual translated text after whole-page
    # registration; this prevents white clothes/panels/windows from becoming SOURCE holes.
    target_container_min_confidence: float = 0.55
    target_container_dedupe_iou: float = 0.60
    target_container_dedupe_cover: float = 0.82
    supplemental_max_area_ratio: float = 0.04
    supplemental_ink_threshold: int = 190
    supplemental_ink_match_tolerance_px: int = 2
    supplemental_min_source_ink_density: float = 0.012
    supplemental_min_target_ink_density: float = 0.008
    supplemental_min_ink_change_score: float = 0.12
    supplemental_min_target_text_pixels: int = 25
    supplemental_min_source_text_pixels: int = 25

    # OCR-free private text-barrier completion for closed narration boxes and
    # starburst/irregular white bubbles missed by the selected primary detector.
    target_text_barrier_enabled: bool = True
    target_text_barrier_min_components: int = 4
    target_text_barrier_max_candidates: int = 48
    target_text_barrier_max_area_ratio: float = 0.055
    target_text_barrier_max_area_to_text_ratio: float = 45.0
    target_text_barrier_min_white_ratio: float = 0.72
    target_text_barrier_min_dark_ratio: float = 0.02
    target_text_barrier_max_dark_ratio: float = 0.24


class TransparentBubbleRevealConfig(BaseModel):
    """Whole-page TARGET-alpha reveal over a registered Chinese lower layer.

    This route is deliberately explicit-only. It never calls Direct Patch, Mask
    Transfer, inpaint, or lettering; SOURCE is used only as a whole-page warped
    lower layer and all bubble detection runs on TARGET.
    """
    enabled: bool = False
    require_explicit_mode: bool = True

    min_registration_confidence: float = 0.75
    max_reprojection_error: float = 2.5
    min_inlier_ratio: float = 0.55

    bubble_backend: str = "auto"  # Koharu-first fallback selector: auto|target_text_contour|koharu_layout|seeded_white|unseeded_white|mangalens|rtdetr_v2|sam2
    target_text_seed_backend: str = "heuristic"  # fallback: heuristic|koharu_layout|auto; legacy paddle is ignored (0-OCR mode contract)
    target_text_seed_min_components: int = 2
    target_text_seed_max_candidates: int = 96
    suppress_page_furniture: bool = True  # reject running headers/footers/page furniture independently of text/OCR/source gates
    page_furniture_top_ratio: float = 0.14
    page_furniture_bottom_ratio: float = 0.94
    verify_target_text_presence: bool = True  # second-stage target-side textness/seed verification for all candidates
    target_text_presence_min_components: int = 2
    target_text_presence_small_region_single_component: bool = True
    target_text_presence_ocr_enabled: bool = False  # optional TARGET OCR verifier; never used for translation/relettering
    target_text_presence_ocr_min_confidence: float = 0.35
    target_text_presence_ocr_fail_open: bool = True
    require_source_translation_evidence: bool = False  # optional legacy/strict gate to suppress untranslated or unsupported candidates
    seed_fallback_protect_detected_white_bubbles: bool = True  # prevent open/seed fallback candidates from swallowing already-detected white speech bubbles
    seeded_container_quality_min_fill_ratio: float = 0.56  # reject overgrown recovered white containers with too much empty bbox area
    seeded_container_quality_min_compactness: float = 0.31  # reject sprawling recovered white containers that look unlike compact speech interiors
    target_text_contour_min_white_ratio: float = 0.55
    target_text_contour_max_dark_ratio: float = 0.22
    target_text_contour_max_area_to_text_ratio: float = 18.0
    target_text_contour_min_area_to_text_ratio: float = 1.20
    target_text_contour_padding_ratio: float = 1.20
    target_text_contour_thresholds: list[int] = Field(default_factory=lambda: [205, 215, 225, 235])
    min_bubble_confidence: float = 0.30
    expand_px: int = 2
    inset_px: int = 0
    protect_border: bool = True
    border_protect_px: int = 2

    clear_mode: str = "hybrid"  # hybrid|full_bubble|text_only; safest default for mixed/color pages
    text_ink_threshold: int = 190
    paper_white_threshold: int = 215
    composite_ink_threshold: int = 110  # strict dark ink for CN paste; protects skin
    feather_px: int = 1
    output_format: str = "rgba_png"  # rgba_png|composite_rgb
    keep_jp_outside_bubbles: bool = True

    max_clear_area_ratio: float = 0.35
    force_review_if_over_ratio: bool = True

class PairingConfig(BaseModel):
    # v0.8.13: deterministic cheap anchors first; visual matching only fills gaps.
    # When both are enabled: unique normalized name/page number → equal-length
    # natural-sort intervals → legacy smart visual+order matcher.
    prefer_name_pairing: bool = False
    prefer_order_pairing: bool = False
    gap_penalty: float = 0.52
    max_pair_cost: float = 0.62
    order_weight: float = 0.16
    hash_weight: float = 0.58
    aspect_weight: float = 0.10
    edge_weight: float = 0.16
    confidence_floor: float = 0.45
    # Full O(N*M) sequence alignment is retained for ordinary books. Large smart
    # segments use a diagonal band; the band automatically widens for count
    # differences and falls back to full alignment only if no valid path exists.
    smart_alignment_full_matrix_max_cells: int = 250000
    smart_alignment_band: int = 64
    # Destructive transfer gets a second OCR-free same-page verification after
    # registration. This catches 80->81 style page shifts even when filenames or
    # order pairing were wrong.
    same_page_precheck_enabled: bool = True
    same_page_min_confidence: float = 0.72
    same_page_max_side: int = 720
    same_page_min_valid_ratio: float = 0.45
    # Optional clean-room remake-style second opinion for *smart* book pairing.
    # Disabled by default so existing projects retain byte-for-byte pairing
    # semantics unless the user explicitly opts in. Strong positive AKAZE/RANSAC
    # evidence may raise a weak smart-pair confidence; inconclusive evidence never
    # deletes or reorders pairs.
    remake_pair_verifier_enabled: bool = False
    remake_pair_verify_confidence_ceiling: float = 0.78
    remake_pair_verify_max_side: int = 1000
    remake_pair_verify_ratio_test: float = 0.76
    remake_pair_verify_min_good_matches: int = 18
    remake_pair_verify_min_inlier_ratio: float = 0.45
    remake_pair_verify_min_spatial_coverage: float = 0.08
    remake_pair_verify_max_median_error: float = 4.5
    remake_pair_verify_max_boost_confidence: float = 0.92


class RegistrationConfig(BaseModel):
    backend: str = "auto"  # auto|opencv|lightglue|loftr
    device: str = "auto"  # deep backends: auto|mps|cuda|cpu
    allow_model_downloads: bool = False  # auto route never downloads hidden weights by default
    fast_identity: bool = True
    fast_identity_max_side: int = 768
    fast_identity_blur_sigma: float = 5.0
    fast_identity_min_phase_response: float = 0.72
    fast_identity_min_correlation: float = 0.965
    fast_identity_max_shift_px: float = 3.0
    fast_identity_large_shift_px: float = 40.0
    fast_identity_large_shift_min_correlation: float = 0.985
    fast_identity_large_shift_min_phase_response: float = 0.80
    # v0.8.30: cheap SIFT/ORB registration pass before the full-resolution
    # feature stage. Same-layout pages with different scan sizes/crops usually
    # settle here, avoiding a needlessly expensive 1800px feature extraction.
    quick_opencv: bool = True
    quick_opencv_max_side: int = 1000
    quick_opencv_max_features: int = 2800
    quick_opencv_accept_confidence: float = 0.72
    quick_opencv_max_median_error: float = 3.5
    quick_opencv_min_spatial_coverage: float = 0.18
    # v0.8.31: after feature/RANSAC registration, optionally run a tiny residual
    # alignment on colour-insensitive structure maps.  ECC is strictly gated by
    # the original feature inliers; it may reduce sub-pixel scan drift but may not
    # materially worsen feature reprojection.
    structure_refine_enabled: bool = True
    structure_refine_min_confidence: float = 0.72
    structure_refine_max_side: int = 900
    structure_refine_ecc_iterations: int = 60
    structure_refine_ecc_epsilon: float = 1e-5
    structure_refine_gauss_size: int = 5
    structure_refine_min_ecc: float = 0.70
    structure_refine_min_correlation_gain: float = 0.001
    structure_refine_max_shift_px: float = 3.0
    structure_refine_max_rotation_deg: float = 0.35
    structure_refine_max_feature_error_worsen_px: float = 0.20
    deep_max_side: int = 1800
    feature: str = "sift"  # sift|orb|aliked|disk
    max_features: int = 6000
    ratio_test: float = 0.76
    min_matches: int = 10
    ransac_threshold: float = 4.0
    model_preference: list[str] = Field(default_factory=lambda: ["similarity", "affine", "homography"])
    min_inlier_ratio: float = 0.28
    max_median_error: float = 7.0
    min_spatial_coverage: float = 0.08
    review_confidence: float = 0.55
    allow_reflection: bool = False


class OCRConfig(BaseModel):
    backend: str = "paddle"  # paddle_*|paddle|manga_ocr|baberu_ocr|ocr48px|external|apple|sidecar|none
    source_backend: str | None = None
    target_backend: str | None = None
    source_lang: str = "ch"
    target_lang: str = "japan"
    ocr_version: str = "PP-OCRv5"
    # PaddleX model-hub source. ``auto`` retries ModelScope/BOS/AIStudio/HF
    # before OCR starts; an explicit source disables fallback for reproducibility.
    paddle_model_source: str = "auto"
    # Model profile is independent from download source.  Keep legacy_v5_auto as
    # the default so existing projects do not silently change OCR output.
    paddle_model_profile: str = "ppocr_v6_medium"
    # Advanced explicit PaddleX model names. Used when profile == custom.
    paddle_text_detection_model_name: str | None = None
    paddle_text_recognition_model_name: str | None = None
    # Optional explicit local model directories. When both are set, PaddleOCR
    # stays fully offline regardless of the selected hub source/profile.
    paddle_text_detection_model_dir: str | None = None
    paddle_text_recognition_model_dir: str | None = None
    min_confidence: float = 0.66
    retry_confidence: float = 0.80
    sidecar_suffix: str = ".ocr.json"
    # Whole-book external OCR exports. Structured PaddleOCR-VL / PP-Structure
    # JSON is preferred because it preserves per-block polygons. Markdown can be
    # selected too; a matching JSON companion is discovered automatically when
    # present. start_page is the 1-based local image ordinal corresponding to
    # external result item 0 (e.g. pages 5-52 => start_page=5).
    external_source_ocr_path: str | None = None
    external_target_ocr_path: str | None = None
    external_source_start_page: int = 1
    external_target_start_page: int = 1
    external_ocr_ignore_labels: list[str] = Field(default_factory=lambda: [
        "number", "footnote", "header", "header_image", "footer", "footer_image",
        "aside_text", "image", "figure", "chart", "formula", "table", "seal",
    ])
    # Koharu-compatible crop OCR engines. Whole-page recognition uses the
    # optional Koharu Layout detector; recognize_region can call the crop model
    # directly. Models are local/explicit and are never pulled at import time.
    manga_ocr_model_path: str | None = None
    baberu_ocr_model_path: str | None = None
    ocr48px_model_path: str | None = None
    koharu_layout_model_path: str | None = None
    koharu_layout_text_threshold: float = 0.25
    koharu_layout_sfx_threshold: float = 0.20
    koharu_layout_shape: int = 1152
    koharu_crop_padding_ratio: float = 0.08
    koharu_ocr_max_new_tokens: int = 128
    # 48px AR OCR uses a verified official checkpoint + pinned upstream network
    # source in the isolated Torch runtime. Nothing is downloaded at startup.
    # An external command remains an optional compatibility fallback.
    # Placeholders: {input}, {output}. The runner writes UTF-8 text or JSON.
    ocr48px_command: str | None = None
    ocr48px_allow_shell: bool = False
    ocr48px_timeout_seconds: int = 120
    preserve_line_breaks: bool = True
    retry_low_confidence: bool = True
    retry_scale: float = 2.0
    # Missing optional OCR packages must not crash Direct/Mask/Auto processing.
    # The pipeline falls back to a Null OCR evidence backend and leaves uncertain
    # regions for QA/manual review instead of downloading models silently.
    soft_fail_missing_backend: bool = True
    # Extra deterministic variants for photographed/reflective old editions.
    photo_ocr_preprocess: bool = True
    photo_ocr_clahe_clip: float = 2.2
    photo_ocr_deskew_local: bool = True
    rectify_source_with_registration: bool = True
    rectify_min_registration_confidence: float = 0.65
    # Preserve the old photo's sampling density during registration-assisted OCR.
    # A 2400px phone photo must not be collapsed to an 850px master before OCR.
    rectify_preserve_source_resolution: bool = True
    rectify_max_scale: float = 3.0
    rectify_max_long_side: int = 3600
    # v0.8.9: default macOS Apple OCR follows Novel-formatter: a Swift
    # VisionKit ImageAnalyzer/Live Text helper first, then the user's macOS
    # ExtractText shortcut.  This path is text-only; manga bubble geometry is
    # supplied by the paired-difference detector instead of Vision bounding boxes.
    apple_shortcut_name: str = "ExtractText"
    apple_live_text_timeout: float = 45.0
    apple_live_text_assumed_confidence: float = 0.88
    apple_live_text_region_padding_ratio: float = 0.08
    apple_live_text_region_min_side_px: int = 28
    apple_live_text_region_whiten_outside_mask: bool = True
    # If both VisionKit and ExtractText are unavailable, mask_replace must still
    # finish with its reversible low-confidence Chinese glyph candidate instead
    # of failing every page in a long batch. QA will mark missing OCR evidence.
    apple_live_text_soft_fail: bool = True


class BubbleConfig(BaseModel):
    # v2.0.91 unified detector policy.  The main detector is single-select and
    # always gets first decision.  Auxiliary detectors are multi-select and run
    # either conditionally or on every page according to detector_strategy.
    # Legacy ``backend`` remains for project compatibility and old plugins.
    detector_strategy: str = "primary_conditional_aux"  # primary_only|primary_conditional_aux|primary_plus_aux
    primary_detector: str = "koharu_layout"  # koharu_layout|mangalens|rtdetr_v2
    auxiliary_detectors: list[str] = Field(default_factory=lambda: ["geometry_white"])
    sam2_refine_enabled: bool = False
    backend: str = "seeded_white"  # legacy fallback selector; synchronized by the GUI
    # Global layout contract: all modes prefer Koharu Layout for geometry, while
    # OCR/text rendering remains controlled solely by the selected transfer mode.
    prefer_koharu_layout: bool = True  # deprecated compatibility field; v2.0.91 detector policy supersedes this
    koharu_layout_cache_enabled: bool = True
    sidecar_suffix: str = ".bubbles.json"
    mangalens_model_path: str | None = None
    device: str = "auto"  # auto|mps|cuda|cpu
    mangalens_confidence: float = 0.35
    mangalens_imgsz: int = 1600
    # Apache-2.0 Comic-Translate RT-DETR-v2 source detector adapter. The
    # pretrained repo is only accessed when model downloads are explicitly allowed.
    rtdetr_model_path: str | None = None
    rtdetr_repo_name: str = "ogkalu/comic-text-and-bubble-detector"
    rtdetr_confidence: float = 0.30
    rtdetr_imgsz: int = 640
    rtdetr_allow_model_downloads: bool = False
    # YSG YOLO OBB is an auxiliary open/rotated-text detector.  It is never a
    # semantic authority: Koharu PROTECT remains final when Koharu is primary.
    ysg_obb_model_path: str | None = None
    ysg_obb_confidence: float = 0.25
    ysg_obb_iou: float = 0.50
    ysg_obb_imgsz: int = 1600
    ysg_obb_include_other: bool = False
    # Optional SAM 2 / SAM 2.1 SOURCE-only segmentation fallback. It is never
    # downloaded silently: either give a local checkpoint+config or explicitly
    # allow a Hugging Face model id. It only refines SOURCE container geometry.
    sam2_checkpoint: str | None = None
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_model_id: str = "facebook/sam2.1-hiera-tiny"
    sam2_allow_model_downloads: bool = False
    sam2_min_score: float = 0.62
    sam2_prompt_expand_ratio: float = 0.85
    # High-resolution text/bubble/panel instance segmentation. It is optional
    # and isolated from the stable geometric/default route.
    koharu_layout_model_path: str | None = None
    koharu_layout_text_threshold: float = 0.25
    koharu_layout_sfx_threshold: float = 0.20
    koharu_layout_bubble_threshold: float = 0.50
    koharu_layout_panel_threshold: float = 0.50
    koharu_layout_shape: int = 1152
    # RF-DETR segmentation postprocess resizes every query mask to the PIL input
    # size. Cap that input on photographed/4K SOURCE pages to avoid multi-GiB
    # temporary tensors; geometry is scaled back to original coordinates.
    koharu_layout_postprocess_max_side: int = 1152
    # Preserve normal ~1600px TARGET input; only very large scans are pre-scaled.
    koharu_layout_postprocess_downscale_trigger_side: int = 2048
    white_threshold: int = 205
    min_area_ratio: float = 0.0015
    max_area_ratio: float = 0.38
    safe_margin_px: int = 8
    safe_margin_ratio: float = 0.035
    close_kernel: int = 3
    search_radius: int = 32


class MatchingConfig(BaseModel):
    auto_apply_kinds: list[str] = Field(default_factory=lambda: ["speech", "narration"])
    centroid_weight: float = 0.34
    overlap_weight: float = 0.28
    projected_iou_weight: float = 0.10
    text_length_weight: float = 0.08
    shape_weight: float = 0.10
    order_weight: float = 0.06
    kind_weight: float = 0.04
    replace_translation_overlap_gate: float = 0.30
    replace_translation_overlap_bonus: float = 0.05
    replace_translation_many_to_one_overlap: float = 0.58
    registration_confidence_penalty_weight: float = 0.12
    diagnostics_top_k: int = 3
    max_cost: float = 0.74
    review_confidence: float = 0.60
    unmatched_cost: float = 0.78


class MaskingConfig(BaseModel):
    dilation_ratio: float = 0.07
    min_dilation_px: int = 1
    max_dilation_px: int = 6
    bubble_border_protection_px: int = 2
    clip_to_bubble: bool = True
    pixel_mask_priority_no_extra_dilation: bool = True
    border_guard_enabled: bool = True
    border_guard_overlap_ratio_max: float = 0.12
    border_guard_dark_ratio_min: float = 0.22
    border_guard_max_erode_px: int = 2
    # Reletter-only TARGET authority completion. Target-driven Region polygons are
    # derived from the original Japanese glyph island, so a uniform white Region
    # can be cleared as a whole rather than trusting a fragmented component mask.
    # This prevents Japanese strokes/punctuation surviving under newly rendered
    # Chinese while remaining clipped to the paired bubble safe interior.
    reletter_region_full_clear_enabled: bool = True
    reletter_region_full_clear_min_white_ratio: float = 0.68
    reletter_region_full_clear_max_spread: float = 48.0
    reletter_region_completion_pad_ratio: float = 0.10
    reletter_region_completion_max_pad_px: int = 14


class InpaintingConfig(BaseModel):
    backend: str = "auto"  # auto|solid|threshold_clear|opencv|lama|lama_manga|aot_inpainting|flux2_klein|rorem_mixed
    solid_variance_threshold: float = 75.0
    opencv_radius: float = 3.0
    prefer_threshold_clear_for_white: bool = True
    threshold_clear_dark_offset: int = 52
    threshold_clear_min_dark_ratio: float = 0.008
    threshold_clear_max_variance: float = 140.0
    lama_command: str | None = None
    lama_allow_shell: bool = False
    lama_timeout_seconds: int = 120
    # Optional model-aware inpainting adapters. Large model runtimes stay out of
    # the GUI process and are invoked through local command wrappers. Commands
    # may use {input}, {mask}, {output}, {model}, {prompt}, {negative_prompt}.
    lama_model_path: str | None = None
    lama_manga_command: str | None = None
    aot_model_path: str | None = None
    aot_command: str | None = None
    flux2_klein_model_path: str | None = None
    flux2_klein_command: str | None = None
    flux2_klein_prompt: str = "Remove the text and reconstruct the surrounding artwork."
    rorem_mixed_model_path: str | None = None
    rorem_mixed_command: str | None = None
    rorem_mixed_prompt: str = "clean manga background, reconstruct artwork, no text"
    rorem_mixed_negative_prompt: str = "letters, text, watermark, artifacts"
    model_allow_shell: bool = False
    model_timeout_seconds: int = 600
    # auto stays conservative: only use heavy AI inpainting if explicitly
    # enabled; otherwise preserve the established white/Telea logic.
    auto_use_ai_models: bool = False


class LetteringConfig(BaseModel):
    font_path: str | None = None
    # Reletter-only typography policy. ``smart`` keeps source/target layout hints,
    # uses phrase-aware CJK breaking and falls back to character-level DP only
    # when necessary. ``balanced`` ignores phrase hints; ``source`` preserves
    # explicit OCR/manual line breaks whenever they fit.
    line_break_mode: str = "smart"  # smart|balanced|source
    # Reletter layout policy, inspired by mature manga translators' separation of
    # line-breaking from geometry constraints. ``strict`` never expands beyond the
    # detected TARGET text box; ``smart_scaling`` may fall back to the bubble-safe
    # region only after the text-box layout fails; ``balloon_fill`` may use the
    # bubble-safe region earlier to preserve a larger source-like font size.
    layout_mode: str = "smart_scaling"  # strict|smart_scaling|balloon_fill
    # Vertical CJK punctuation is opt-in here because raster transfer modes do not
    # use the lettering engine. Reletter/manual relettering may position comma/
    # full stop toward the upper-right of a glyph cell, closer to print manga.
    vertical_punctuation: bool = True
    min_font_size: int = 10
    max_font_size: int = 72
    min_safe_coverage: float = 0.997
    line_spacing_ratio: float = 0.16
    side_padding_ratio: float = 0.04
    # Reletter-only vertical column gap. 0 keeps glyph cells adjacent; a small
    # positive ratio improves manga readability without widening raster-transfer modes.
    column_spacing_ratio: float = 0.06
    # Ephemeral TARGET layout hints. Pipeline/manual editor populate these per Region;
    # they are ignored unless relettering actually calls the lettering engine.
    anchor_x_ratio: float | None = None
    anchor_y_ratio: float | None = None
    preferred_bbox_width_ratio: float | None = None
    preferred_bbox_height_ratio: float | None = None
    orientation: str = "auto"  # horizontal|vertical|auto
    vertical_aspect_threshold: float = 2.25
    stroke_width: int = 0
    fill: tuple[int, int, int] = (0, 0, 0)
    stroke_fill: tuple[int, int, int] = (255, 255, 255)
    max_lines: int = 8
    # Optional per-bubble layout hints populated from the translated source scan.
    # These are intentionally ephemeral: OCR identifies Unicode; source pixels
    # remain the authority for typography and column density.
    preferred_font_size: int | None = None
    preferred_columns: int | None = None
    preferred_font_tolerance_ratio: float = 0.22
    # Source-layout hints are preferences, not a reason to leave a balloon empty.
    # Try the recovered source size/column density first; if it cannot safely fit
    # the HD target container, progressively shrink down to min_font_size.
    preferred_font_allow_shrink_fallback: bool = True
    # Render glyph masks at higher resolution, then downsample once with Lanczos.
    # This avoids jagged/soft small CJK text without scaling the manga page itself.
    supersample_factor: int = 4
    # Optional Reletter-only joined-balloon flow partition. Inspired by Koharu's
    # bubble-aware layout concept, implemented independently in Python. Disabled
    # by default so existing Direct/Mask/Reletter output remains unchanged.
    koharu_flow_cells_enabled: bool = False


class HybridMaskConfig(PixelTransferConfigBase):
    """Hybrid-private high-fidelity transfer defaults.

    Shared geometry/mask primitives are reused, but these defaults are not read
    from the user-facing Precise Mask mode. Hybrid may therefore evolve without
    silently changing Mask, and vice versa.
    """
    # v2.3.32 Hybrid owns a physically separate renderer policy namespace.
    renderer_owner: str = "hybrid"
    # Incomplete mask-stage pixels are not published; they are restored so the
    # private Hybrid OCR/reletter fallback can own the completion cleanly.
    incomplete_pixel_policy: str = "restore_target"  # restore_target|keep_review; any Mask candidate owns the region, automatic OCR is uncovered-only

    # Hybrid uses the same glyph-integrity principle for its first stage: dense
    # alignment may discover masks, never bend the final Chinese raster.
    paired_diff_dense_flow_geometry_only: bool = True
    paired_diff_render_use_global_raster: bool = True
    paired_diff_forbid_dense_glyph_warp: bool = True
    paired_diff_proxy_warn_iou: float = 0.08
    paired_diff_proxy_warn_area_ratio: float = 0.02

    # Damaged page-edge source regions are not valid automatic translation
    # material. Hybrid may OCR/reletter complete missed regions, but must never
    # invent missing text from a physically cropped source photograph.
    photo_pair_low_confidence_candidate_enabled: bool = False
    photo_pair_reject_edge_clipped_open_text: bool = True
    photo_pair_reject_edge_clipped_complex_text: bool = True
    photo_pair_edge_clipped_review_required: bool = True
    # v2.3.67: Hybrid mirrors Precise Mask product semantics but keeps its own
    # private config/renderer. White-bubble Chinese enhancement is opt-in and is
    # OFF by default; OCR completion remains independent of this switch.
    direct_white_clarity_enhance_enabled: bool = False
    direct_white_clarity_alpha_gamma: float = 1.0
    direct_white_clarity_black_boost: int = 0
    direct_white_clarity_pure_white_floor: int = 248
    direct_white_clarity_min_text_pixels: int = 18
    # v2.3.36 Hybrid-private copy of the same product rule. These fields are
    # intentionally separate from MaskReplaceConfig so future tuning cannot
    # silently alter the other mode.
    hybrid_source_structure_guard_enabled: bool = True
    hybrid_source_structure_guard_ratio: float = 0.035
    hybrid_source_structure_guard_min_px: int = 5
    hybrid_source_structure_guard_max_px: int = 14
    hybrid_source_structure_min_component_area: int = 10
    hybrid_source_structure_min_aspect: float = 2.6
    hybrid_source_structure_min_span_ratio: float = 0.16
    hybrid_source_spiky_boundary_band_px: int = 14
    # v2.3.59 Hybrid-private mirror of Precise Mask lettering/structure authority.
    hybrid_source_lettering_relief_enabled: bool = True
    hybrid_source_lettering_relief_dilate_px: int = 1
    hybrid_target_structure_guard_enabled: bool = True
    hybrid_target_structure_text_relief_enabled: bool = True
    hybrid_target_structure_text_relief_dilate_px: int = 1
    hybrid_target_border_probe_dilate_px: int = 4
    hybrid_target_border_restore_fringe_px: int = 2
    hybrid_target_immutable_band_enabled: bool = True
    hybrid_target_ordinary_inner_band_px: int = 6
    hybrid_target_spiky_inner_band_px: int = 18
    hybrid_target_outer_band_px: int = 3
    hybrid_target_band_text_margin_px: int = 3
    hybrid_target_spiky_glyph_relief_enabled: bool = True
    hybrid_target_spiky_glyph_relief_dilate_px: int = 1
    hybrid_target_spiky_safe_core_ellipse_enabled: bool = True
    hybrid_target_spiky_safe_core_inset_ratio: float = 0.045
    hybrid_target_spiky_safe_core_min_inset_px: int = 6
    # v2.3.59 Hybrid-private mirror of Precise Mask SOURCE completeness.
    hybrid_raw_source_completeness_enabled: bool = True
    hybrid_raw_source_min_coverage: float = 0.965
    # OCR policy: automatic OCR is a fallback only for truly uncovered regions.
    # Any region that already has a Mask-stage candidate/ownership is excluded
    # from automatic OCR; the GUI manual OCR box may still explicitly override.
    hybrid_auto_ocr_uncovered_only: bool = True
    hybrid_auto_ocr_block_if_any_mask_candidate: bool = True
    hybrid_manual_ocr_force_allowed: bool = True


class HybridModeConfig(BaseModel):
    """Hybrid owns private Mask-like candidates and private lettering settings.

    Code primitives are shared, but mutable defaults/state are not borrowed from
    the user-facing Mask or Reletter modes.
    """
    mask: HybridMaskConfig = Field(default_factory=HybridMaskConfig)
    lettering: LetteringConfig = Field(default_factory=LetteringConfig)
    # Hybrid is mask-first. OCR/relettering is a completion stage, not a gate
    # that can cancel the whole page before paired visual evidence is consumed.
    continue_with_paired_visual_evidence_when_ocr_unavailable: bool = True
    mask_first: bool = True
    ocr_reletter_fallback_enabled: bool = True
    block_reletter_for_edge_clipped_source: bool = True


class ReletterModeConfig(BaseModel):
    """Reletter owns its candidate geometry snapshot and typography settings."""
    candidates: PixelTransferConfigBase = Field(default_factory=PixelTransferConfigBase)
    lettering: LetteringConfig = Field(default_factory=LetteringConfig)


class DualSourceConfig(BaseModel):
    enabled: bool = False
    secondary_source_dir: str | None = None
    # Compatibility flag: secondary is evaluated when available, but v0.9
    # arbitration no longer lets this flag override a materially better primary.
    prefer_secondary_for_direct: bool = True
    accept_secondary_direct: bool = True
    secondary_ink_fallback: bool = True
    recursive_lookup: bool = False
    arbitration_enabled: bool = True
    arbitration_min_same_page_confidence: float = 0.72
    arbitration_min_registration_confidence: float = 0.78
    arbitration_max_reprojection_error_px: float = 7.0
    arbitration_reprojection_good_px: float = 3.0
    arbitration_boundary_distance_good_px: float = 2.5
    arbitration_sharpness_reference: float = 180.0
    arbitration_same_page_weight: float = 0.24
    arbitration_registration_weight: float = 0.20
    arbitration_reprojection_weight: float = 0.12
    arbitration_coverage_weight: float = 0.18
    arbitration_boundary_weight: float = 0.10
    arbitration_sharpness_weight: float = 0.12
    arbitration_risk_weight: float = 0.04
    arbitration_rejected_score_multiplier: float = 0.25


class ReplaceTranslationConfig(BaseModel):
    export_enabled: bool = True
    export_dirname: str = "replace_translation"
    export_matches_json: bool = True
    export_summary_json: bool = True
    export_ocr_json: bool = True
    additional_source_manifest_suffix: str = ".replace_sources.json"
    additional_source_enabled: bool = True
    additional_source_max_candidates: int = 2
    additional_source_retry_direct: bool = True


class QAConfig(BaseModel):
    registration_min_confidence: float = 0.55
    ocr_min_confidence: float = 0.66
    match_min_confidence: float = 0.60
    residual_dark_ratio_max: float = 0.06
    lettering_safe_coverage_min: float = 0.997
    min_font_size: int = 10
    fail_on_error: bool = False
    # Publication safety: a mask-replace page that produced zero candidates must
    # not silently report success. Users can disable this for known blank pages.
    fail_empty_mask_replace: bool = True


class ExportConfig(BaseModel):
    # v1.3.12 compact-by-default workspace.  The files needed for GUI restore,
    # review and manual repair remain lossless; expensive diagnostic duplicates
    # are opt-in.  This cuts a typical 1600px page workspace from ~35-55 MB to
    # roughly 8-15 MB before cache, without changing final pixels.
    save_inpainted: bool = False
    save_debug: bool = False
    save_masks: bool = True
    save_component_masks: bool = False
    save_review_preview_always: bool = False
    save_project_json: bool = True
    image_format: str = "png"
    tiff: bool = False
    layer_bundle: bool = False
    # Preserve independent SOURCE/TARGET workspace copies while using CoW clones
    # for already-PNG inputs when the filesystem supports it (APFS/reflink).
    prefer_input_reflink: bool = True
    # Lossless compression levels: persistent colour pages stay balanced for
    # throughput; sparse masks/layers favour disk size.
    persistent_png_compression: int = 4
    sparse_png_compression: int = 9


class SemanticLayoutConfig(BaseModel):
    """Front-end semantic layout layer; renderers remain unchanged.

    PP-DocLayoutV3 is the preferred provider.  When its runtime/model is absent,
    the optional heuristic fallback still provides a deterministic analysis layer
    for QA and safe semantic filtering without downloading anything silently.
    """
    enabled: bool = False
    backend: str = "auto"  # auto|pp_doclayout_v3|heuristic
    strategy: str = "auto"  # auto|strict|loose|analysis_only
    apply_to_reveal: bool = True
    apply_to_direct: bool = False  # hook reserved; stable renderer untouched in v2.2.0
    apply_to_mask: bool = False  # hook reserved; stable renderer untouched in v2.2.0
    fallback_heuristic: bool = True
    paddle_model_name: str = "PP-DocLayoutV3"
    paddle_model_dir: str | None = None
    paddle_allow_model_downloads: bool = False
    paddle_device: str = "cpu"
    paddle_threshold: float = 0.30
    heuristic_min_components: int = 2
    max_blocks: int = 128
    header_top_ratio: float = 0.14
    footer_bottom_ratio: float = 0.94
    save_overlay: bool = False
    save_json: bool = True
    semantic_roi_pad_px: int = 10


class PipelineConfig(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _migrate_mode_owned_settings(cls, value):
        """Seed new per-mode state from legacy shared namespaces once.

        This is a compatibility migration only. After loading, Hybrid/Reletter
        own independent copies and subsequent edits never write back to Mask or
        to each other.
        """
        if not isinstance(value, dict):
            return value
        data = dict(value)
        legacy_lettering = data.get("lettering")
        legacy_mask = data.get("mask_replace")
        if "hybrid" not in data and (legacy_lettering is not None or legacy_mask is not None):
            data["hybrid"] = {
                "mask": dict(legacy_mask or {}),
                "lettering": dict(legacy_lettering or {}),
            }
        if "reletter" not in data and (legacy_lettering is not None or legacy_mask is not None):
            data["reletter"] = {
                "candidates": dict(legacy_mask or {}),
                "lettering": dict(legacy_lettering or {}),
            }
        return data

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    page_management: PageManagementConfig = Field(default_factory=PageManagementConfig)
    transfer: TransferModeConfig = Field(default_factory=TransferModeConfig)
    direct_patch: DirectPatchConfig = Field(default_factory=DirectPatchConfig)
    mask_replace: MaskReplaceConfig = Field(default_factory=MaskReplaceConfig)
    aligned_overlay_reveal: AlignedOverlayRevealConfig = Field(default_factory=AlignedOverlayRevealConfig)
    transparent_bubble_reveal: TransparentBubbleRevealConfig = Field(default_factory=TransparentBubbleRevealConfig)
    hybrid: HybridModeConfig = Field(default_factory=HybridModeConfig)
    reletter: ReletterModeConfig = Field(default_factory=ReletterModeConfig)
    semantic: SemanticLayoutConfig = Field(default_factory=SemanticLayoutConfig)
    pairing: PairingConfig = Field(default_factory=PairingConfig)
    registration: RegistrationConfig = Field(default_factory=RegistrationConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    bubbles: BubbleConfig = Field(default_factory=BubbleConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    inpainting: InpaintingConfig = Field(default_factory=InpaintingConfig)
    lettering: LetteringConfig = Field(default_factory=LetteringConfig)
    dual_source: DualSourceConfig = Field(default_factory=DualSourceConfig)
    replace_translation: ReplaceTranslationConfig = Field(default_factory=ReplaceTranslationConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
