from ..base import ModeContract, ModeSpec
SPEC = ModeSpec(
    key="mask_replace", label="精准蒙版 · 原字保真 / 拍照边缘保护（0 OCR）", config_root="mask_replace",
    contract=ModeContract("mask_replace", mask_replace=True, may_use_mask_pixels=True),
    owned_artifacts=("mask_transfer_layer.png","mask_transfer_layer_reviewed.png","mask_transfer_mask.png","mask_transfer.json"),
    owned_paths=("src/manga_hd_transfer/modes/mask_replace/",),
    fallback_policy="none",
    workflow=('page_pair', 'registration', 'visual_candidates', 'mask_render', 'mask_qa', 'mask_persist'),
    ui_defaults={"paired_diff_enabled": True, "exact_identity_copy": True},
)
