from ..base import ModeContract, ModeSpec
SPEC = ModeSpec(
    key="transparent_bubble_reveal", label="旧 Transparent Reveal（冻结兼容）", config_root="transparent_bubble_reveal", visible=False, legacy=True,
    contract=ModeContract("transparent_bubble_reveal", transparent_reveal=True, may_use_presence_ocr=True, may_use_mask_pixels=True),
    owned_artifacts=("transparent_bubble_reveal.json","final_rgba.png","jp_layer_rgba.png","cn_layer_rgb.png"),
    owned_paths=("src/manga_hd_transfer/transparent_bubble_reveal.py","src/manga_hd_transfer/modes/transparent_bubble_reveal/"),
    fallback_policy="none",
    workflow=('page_pair', 'registration', 'transparent_candidates', 'transparent_render', 'transparent_qa', 'transparent_persist'),
    ui_defaults={},
)
