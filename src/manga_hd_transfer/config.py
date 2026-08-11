from __future__ import annotations

from pydantic import BaseModel, Field


class PairingConfig(BaseModel):
    gap_penalty: float = 0.52
    max_pair_cost: float = 0.62
    order_weight: float = 0.16
    hash_weight: float = 0.58
    aspect_weight: float = 0.10
    edge_weight: float = 0.16
    confidence_floor: float = 0.45


class RegistrationConfig(BaseModel):
    backend: str = "auto"  # auto|opencv|lightglue|loftr
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
    backend: str = "paddle"  # common fallback: paddle|sidecar|none
    source_backend: str | None = None
    target_backend: str | None = None
    source_lang: str = "ch"
    target_lang: str = "japan"
    ocr_version: str = "PP-OCRv5"
    min_confidence: float = 0.66
    retry_confidence: float = 0.80
    sidecar_suffix: str = ".ocr.json"
    preserve_line_breaks: bool = True
    retry_low_confidence: bool = True
    retry_scale: float = 2.0


class BubbleConfig(BaseModel):
    backend: str = "seeded_white"  # seeded_white|sidecar|none
    sidecar_suffix: str = ".bubbles.json"
    white_threshold: int = 205
    min_area_ratio: float = 0.0015
    max_area_ratio: float = 0.38
    safe_margin_px: int = 8
    safe_margin_ratio: float = 0.035
    close_kernel: int = 3
    search_radius: int = 32


class MatchingConfig(BaseModel):
    auto_apply_kinds: list[str] = Field(default_factory=lambda: ["speech", "narration"])
    centroid_weight: float = 0.42
    overlap_weight: float = 0.34
    shape_weight: float = 0.10
    order_weight: float = 0.08
    kind_weight: float = 0.06
    max_cost: float = 0.74
    review_confidence: float = 0.60
    unmatched_cost: float = 0.78


class MaskingConfig(BaseModel):
    dilation_ratio: float = 0.07
    min_dilation_px: int = 1
    max_dilation_px: int = 6
    bubble_border_protection_px: int = 2
    clip_to_bubble: bool = True


class InpaintingConfig(BaseModel):
    backend: str = "auto"  # auto|solid|opencv|lama
    solid_variance_threshold: float = 75.0
    opencv_radius: float = 3.0
    lama_command: str | None = None
    lama_timeout_seconds: int = 120


class LetteringConfig(BaseModel):
    font_path: str | None = None
    min_font_size: int = 10
    max_font_size: int = 72
    min_safe_coverage: float = 0.997
    line_spacing_ratio: float = 0.16
    side_padding_ratio: float = 0.04
    orientation: str = "horizontal"  # horizontal|vertical|auto
    vertical_aspect_threshold: float = 2.25
    stroke_width: int = 0
    fill: tuple[int, int, int] = (0, 0, 0)
    stroke_fill: tuple[int, int, int] = (255, 255, 255)
    max_lines: int = 8


class QAConfig(BaseModel):
    registration_min_confidence: float = 0.55
    ocr_min_confidence: float = 0.66
    match_min_confidence: float = 0.60
    residual_dark_ratio_max: float = 0.06
    lettering_safe_coverage_min: float = 0.997
    min_font_size: int = 10
    fail_on_error: bool = False


class ExportConfig(BaseModel):
    save_inpainted: bool = True
    save_debug: bool = True
    save_masks: bool = True
    save_project_json: bool = True
    image_format: str = "png"
    tiff: bool = False
    layer_bundle: bool = True


class PipelineConfig(BaseModel):
    pairing: PairingConfig = Field(default_factory=PairingConfig)
    registration: RegistrationConfig = Field(default_factory=RegistrationConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    bubbles: BubbleConfig = Field(default_factory=BubbleConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    inpainting: InpaintingConfig = Field(default_factory=InpaintingConfig)
    lettering: LetteringConfig = Field(default_factory=LetteringConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
