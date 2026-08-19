from ..base import ModeContract, ModeSpec
SPEC = ModeSpec(
    key="auto", label="旧 Auto · 冻结兼容", config_root="legacy_auto", visible=False, legacy=True,
    contract=ModeContract("auto", orchestrator=True, direct=True, mask_replace=True, may_use_ocr=True, may_use_mask_pixels=True, may_fallback_to_mask=True),
    owned_paths=("src/manga_hd_transfer/modes/legacy_auto/",),
    fallback_policy="legacy_frozen_orchestrator",
    workflow=('page_pair', 'registration', 'legacy_direct_arbitration', 'legacy_mask_fallback', 'legacy_persist'),
    ui_defaults={},
)
