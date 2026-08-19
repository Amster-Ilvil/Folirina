from ..base import ModeContract, ModeSpec
SPEC = ModeSpec(
    key="aligned_overlay_reveal", label="整页对齐显中文（透明 / 挖洞）", config_root="aligned_overlay_reveal",
    contract=ModeContract("aligned_overlay_reveal", aligned_reveal=True, may_use_presence_ocr=True, may_use_mask_pixels=True),
    owned_artifacts=("aligned_overlay_reveal.json","aligned_overlay_reveal_layer.png","aligned_overlay_reveal_mask.png","aligned_overlay_reveal_hole_mask.png","aligned_overlay_reveal_erase_mask.png","aligned_overlay_reveal_regions.png","aligned_overlay_reveal_source_ink.png"),
    owned_paths=("src/manga_hd_transfer/aligned_overlay_reveal.py","src/manga_hd_transfer/aligned_overlay_reveal_core.py","src/manga_hd_transfer/aligned_overlay_reveal_mode.py","src/manga_hd_transfer/modes/aligned_overlay_reveal/"),
    fallback_policy="none",
    workflow=('page_pair', 'registration', 'target_bubbles', 'hole_plan', 'hole_render', 'aligned_qa', 'aligned_persist'),
    ui_defaults={},
)
