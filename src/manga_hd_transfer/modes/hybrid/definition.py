from ..base import ModeContract, ModeSpec
SPEC = ModeSpec(
    key="hybrid", label="精准蒙版+OCR", config_root="hybrid",
    contract=ModeContract("hybrid", mask_replace=True, reletter=True, may_use_ocr=True, may_render_text=True, may_use_mask_pixels=True, may_fallback_to_reletter=True, manual_reletter_editor=True),
    owned_artifacts=("hybrid_transfer_layer.png","hybrid_transfer_layer_reviewed.png","hybrid_transfer_mask.png","hybrid_transfer.json","hybrid_text_layer.png","hybrid_text_layer_reviewed.png"),
    owned_paths=("src/manga_hd_transfer/modes/hybrid/",),
    fallback_policy="hybrid_owned_reletter_only",
    workflow=('page_pair', 'registration', 'hybrid_private_candidates', 'hybrid_mask_stage', 'hybrid_reletter_stage', 'hybrid_qa', 'hybrid_persist'),
    ui_defaults={"paired_diff_enabled": True, "exact_identity_copy": True, "line_break_mode":"smart", "layout_mode":"smart_scaling"},
)
