from ..base import ModeContract, ModeSpec
SPEC = ModeSpec(
    key="direct_patch", label="直接贴图 · 无边框内层贴图（SOURCE 在上 / TARGET 在下）", config_root="direct_patch",
    contract=ModeContract("direct_patch", direct=True, may_use_mask_pixels=True),
    owned_artifacts=("direct_patch_layer.png","direct_patch_layer_reviewed.png","direct_patch_regions.png","direct_patch.json"),
    owned_paths=(
        "src/manga_hd_transfer/direct_containers.py",
        "src/manga_hd_transfer/pipeline_direct_arbitration.py",
        "src/manga_hd_transfer/direct_patch_mode.py",
        "src/manga_hd_transfer/direct_patch_bridge.py",
        "src/manga_hd_transfer/direct_patch_runner.py",
        "src/manga_hd_transfer/modes/direct_patch/",
    ),
    fallback_policy="none",
    workflow=('page_pair', 'registration', 'direct_candidates', 'direct_render', 'direct_qa', 'direct_persist'),
    ui_defaults={"exact_identity_copy": True},
)
