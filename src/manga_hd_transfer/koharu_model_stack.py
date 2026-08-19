from __future__ import annotations

"""Optional Koharu-inspired manga model stack for Folirina.

The core application never imports or downloads these heavy model runtimes at
startup.  This module is deliberately metadata-only so the GUI and runtime
preflight can expose the processors without mutating a working Direct / Mask /
Reletter installation.
"""

from dataclasses import dataclass, field
from typing import Literal

Stage = Literal["detection", "ocr", "inpainting"]


@dataclass(frozen=True, slots=True)
class ModelFile:
    path: str
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class KoharuModelSpec:
    key: str
    label: str
    stage: Stage
    repo: str | None
    revision: str | None = None
    license: str = "unknown"
    runtime: str = "external"
    files: tuple[ModelFile, ...] = field(default_factory=tuple)
    recommended: dict[str, float | int | str] = field(default_factory=dict)
    notes: str = ""


KOHARU_MODEL_SPECS: dict[str, KoharuModelSpec] = {
    "koharu_layout": KoharuModelSpec(
        key="koharu_layout",
        label="Koharu Layout RF-DETR Seg 2XL · 1152",
        stage="detection",
        repo="mayocream/koharu-layout-rfdetr-seg-2xl-1152",
        revision="aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2",
        license="model-specific / Manga109 terms",
        runtime="rfdetr-1.7",
        files=(
            ModelFile("inference_config.json"),
            ModelFile("load_model.py"),
            ModelFile("model.safetensors", "9bf6d2cbd7793c956d8c857bb1672a396eb7f100eb0682f86830d05e31168efb"),
        ),
        recommended={"shape": 1152, "text_threshold": 0.25, "sfx_threshold": 0.20, "bubble_threshold": 0.50, "panel_threshold": 0.50},
        notes="Instance segmentation classes: text, onomatopoeia, bubble, panel.",
    ),
    "manga_ocr": KoharuModelSpec(
        key="manga_ocr",
        label="Manga OCR",
        stage="ocr",
        repo="mayocream/manga-ocr",
        license="Apache-2.0",
        runtime="transformers",
        files=(
            ModelFile("config.json"),
            ModelFile("preprocessor_config.json"), ModelFile("special_tokens_map.json"),
            ModelFile("tokenizer_config.json"), ModelFile("vocab.txt"),
            ModelFile("model.safetensors"),
        ),
        notes="Japanese manga crop recognizer; layout detection is supplied separately.",
    ),
    "baberu_ocr": KoharuModelSpec(
        key="baberu_ocr",
        label="Baberu OCR · ONNX 121 MB",
        stage="ocr",
        repo="genshiai-daichi/baberu-ocr",
        revision="d9cc13153e9a1cd8fdfa3b7b1cc329da2020aeae",
        license="Apache-2.0",
        runtime="onnxruntime",
        files=(
            ModelFile("onnx_infer.py"),
            ModelFile("onnx/vision_int4.onnx"),
            ModelFile("onnx/decoder_prefill_int8.onnx"),
            ModelFile("onnx/decoder_step_int8.onnx"),
            ModelFile("tokenizer/vocab.json"),
        ),
        notes="Crop OCR for Japanese/Chinese/English; Folirina feeds layout text crops.",
    ),
    "ocr48px": KoharuModelSpec(
        key="ocr48px",
        label="Manga Image Translator 48px AR OCR",
        stage="ocr",
        repo="zyddnys/manga-image-translator",
        revision="3e29cd63a0ce7d1b4013b0a6e56da4cddaf4fe5b",
        license="GPL-3.0 upstream runtime; optional external cache",
        runtime="isolated-torch-pinned-source",
        files=(
            ModelFile("ocr_ar_48px.ckpt", "29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d"),
            ModelFile("alphabet-all-v7.txt", "f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a"),
        ),
        recommended={"text_height": 48, "beam_size": 5, "max_sequence_length": 255},
        notes="Official classic 48px autoregressive manga OCR. The verified checkpoint/dictionary and exact pinned upstream source are fetched only on explicit user request; GPL source is never bundled in the Folirina ZIP. Native inference runs in the isolated Torch worker; external-command mode remains an optional fallback.",
    ),
    "lama_manga": KoharuModelSpec(
        key="lama_manga",
        label="LaMa Manga",
        stage="inpainting",
        repo="mayocream/lama-manga",
        license="MIT",
        runtime="external-command",
        files=(ModelFile("lama-manga.safetensors", "a790515e9da839b8d89af7d565ceb110d908b7d6fbdb991f2acb2ec7d9b08bdb"),),
        notes="Manga/anime fine-tuned LaMa checkpoint; adapter never replaces target artwork outside the mask.",
    ),
    "aot_inpainting": KoharuModelSpec(
        key="aot_inpainting",
        label="AOT Inpainting",
        stage="inpainting",
        repo="mayocream/aot-inpainting",
        license="MIT",
        runtime="external-command",
        files=(ModelFile("config.json"), ModelFile("model.safetensors")),
        notes="AOT-GAN generator checkpoint; invoked through an isolated/local runner.",
    ),
    "flux2_klein": KoharuModelSpec(
        key="flux2_klein",
        label="FLUX.2 Klein",
        stage="inpainting",
        repo="black-forest-labs/FLUX.2-klein-base-4B",
        license="Apache-2.0",
        runtime="diffusers/external-command",
        recommended={"prompt": "Remove the text and reconstruct the surrounding artwork."},
        notes="Large generative image-editing stack. Directory import is supported; the core ZIP never bundles it.",
    ),
    "rorem_mixed": KoharuModelSpec(
        key="rorem_mixed",
        label="RORem Mixed",
        stage="inpainting",
        repo="mayocream/RORem-mixed-GGUF",
        license="OpenRAIL++",
        runtime="stable-diffusion.cpp/external-command",
        files=(ModelFile("rorem-mixed-unet-q4_K.gguf"), ModelFile("sdxl-version-marker.safetensors")),
        recommended={"prompt": "clean manga background, reconstruct artwork, no text", "negative_prompt": "letters, text, watermark, artifacts"},
        notes="Manga-focused SDXL inpainting UNet; requires the matching SDXL VAE/text encoders in the chosen runner.",
    ),
}


def model_keys(stage: Stage | None = None) -> tuple[str, ...]:
    rows = KOHARU_MODEL_SPECS.values()
    if stage is not None:
        rows = (row for row in rows if row.stage == stage)
    return tuple(row.key for row in rows)


def model_spec(key: str) -> KoharuModelSpec:
    try:
        return KOHARU_MODEL_SPECS[str(key)]
    except KeyError as exc:
        raise KeyError(f"unknown Koharu model adapter: {key}") from exc


__all__ = ["ModelFile", "KoharuModelSpec", "KOHARU_MODEL_SPECS", "model_keys", "model_spec"]
