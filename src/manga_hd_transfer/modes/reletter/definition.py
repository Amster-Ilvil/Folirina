from ..base import ModeContract, ModeSpec
SPEC = ModeSpec(
    key="reletter", label="OCR重排", config_root="reletter",
    contract=ModeContract("reletter", reletter=True, may_use_ocr=True, may_render_text=True, manual_reletter_editor=True),
    owned_artifacts=("reletter_text_layer.png","reletter_text_layer_reviewed.png","reletter.json"),
    owned_paths=("src/manga_hd_transfer/modes/reletter/",),
    fallback_policy="none",
    workflow=('page_pair', 'registration', 'reletter_private_candidates', 'ocr', 'reletter_render', 'reletter_qa', 'reletter_persist'),
    ui_defaults={"paired_diff_enabled": True, "line_break_mode":"smart", "layout_mode":"smart_scaling"},
)
