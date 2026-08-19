from __future__ import annotations

"""Side-effect-free planning for task-time optional model/runtime preparation.

The GUI and CLI should decide *what a selected configuration explicitly needs*
before launching the expensive pipeline.  This module only plans/validates; it
never imports heavy model packages, starts workers or downloads anything.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import PipelineConfig
from .mode_contracts import get_mode_contract
from .detector_policy import configured_runtime_detectors, primary_detector, detector_strategy
from .model_downloads import model_local_paths, paddle_profile_marker_status
from .paddle_profiles import backend_profile_key, normalize_paddle_model_profile, profile_label, get_paddle_model_profile


@dataclass(frozen=True, slots=True)
class RuntimeRequirement:
    key: str
    label: str
    profile: str | None = None
    reason: str = ""
    runtime_key: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimePreflightPlan:
    requirements: tuple[RuntimeRequirement, ...]
    errors: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.requirements


def _paddle_profile_for_backend(config: PipelineConfig, backend: str | None) -> str | None:
    key = str(backend or "").strip().lower()
    if key in {"", "inherit"}:
        return None
    if key == "paddle":
        return normalize_paddle_model_profile(getattr(config.ocr, "paddle_model_profile", None))
    return backend_profile_key(key)


def required_paddle_profiles(config: PipelineConfig) -> tuple[str, ...]:
    """Return distinct Paddle profiles actually used by main/SOURCE/TARGET OCR."""
    cfg = config.ocr
    main = str(cfg.backend or "none")
    src = main if cfg.source_backend in (None, "", "inherit") else str(cfg.source_backend)
    tgt = main if cfg.target_backend in (None, "", "inherit") else str(cfg.target_backend)
    order: list[str] = []
    for backend in (main, src, tgt):
        profile = _paddle_profile_for_backend(config, backend)
        if profile and profile not in order:
            order.append(profile)
    return tuple(order)


def _external_path_error(config: PipelineConfig, role: str, backend: str) -> str | None:
    if str(backend or "").lower() != "external":
        return None
    value = getattr(config.ocr, f"external_{role}_ocr_path", None)
    if not value:
        return f"{role.upper()} OCR 已选择“外部 OCR JSON / MD”，但尚未选择结果文件。"
    path = Path(str(value)).expanduser()
    if not path.is_file():
        return f"{role.upper()} 外部 OCR 文件不存在：{path}"
    return None


def plan_runtime_requirements(config: PipelineConfig) -> RuntimePreflightPlan:
    """Plan model/runtime preparation for the selected mode.

    v2.0.90 makes Koharu Global Layout the shared first semantic detector for
    every transfer mode, so its runtime is always prepared once up front. OCR
    requirements are mode-contract gated: a 0-OCR visual route must not download
    or initialize an OCR model merely because an old project still remembers an
    OCR selection. Registration remains independent because every automatic
    SOURCE→TARGET route needs geometric alignment before semantic layout.
    """
    rows: list[RuntimeRequirement] = []
    errors: list[str] = []

    def add(key: str, label: str, *, profile: str | None = None, reason: str = "", runtime_key: str | None = None):
        token = (key, profile or "")
        if any((r.key, r.profile or "") == token for r in rows):
            return
        rows.append(RuntimeRequirement(key, label, profile, reason, runtime_key))

    mode = str(config.transfer.mode or "auto").lower()
    contract = get_mode_contract(mode)

    # v2.0.91 detector policy: prepare only the selected primary and explicitly
    # enabled model-backed auxiliaries.  "Primary only" must not download/load
    # detectors the user did not select.
    detector_labels = {
        "koharu_layout": "Koharu Layout RF-DETR Seg 2XL",
        "mangalens": "MangaLens",
        "ysg_obb": "YSG YOLO OBB",
        "rtdetr_v2": "RT-DETR-v2",
        "sam2": "SAM 2.1",
    }
    primary_key = primary_detector(config.bubbles)
    scheduled_detectors = configured_runtime_detectors(config.bubbles)
    add(primary_key, detector_labels[primary_key], reason=f"主检测器 · {detector_strategy(config.bubbles)}", runtime_key=primary_key)

    # OCR model choices only matter when the selected mode contract permits OCR.
    # Reveal routes use OCR only when the dedicated TARGET-presence switch is on.
    reveal_mode = mode in {"transparent_bubble_reveal", "aligned_overlay_reveal"}
    reveal_ocr_enabled = bool(getattr(config.transparent_bubble_reveal, "target_text_presence_ocr_enabled", False))
    ocr_runtime_enabled = bool(
        contract.may_use_ocr
        or (reveal_mode and bool(getattr(contract, "may_use_presence_ocr", False)) and reveal_ocr_enabled)
    )
    if ocr_runtime_enabled:
        main = str(config.ocr.backend or "none")
        src = main if config.ocr.source_backend in (None, "", "inherit") else str(config.ocr.source_backend)
        tgt = main if config.ocr.target_backend in (None, "", "inherit") else str(config.ocr.target_backend)
        profiles = []
        if reveal_mode:
            profile = _paddle_profile_for_backend(config, tgt)
            if profile:
                profiles.append(profile)
        else:
            profiles.extend(required_paddle_profiles(config))
        for profile in profiles:
            family = get_paddle_model_profile(profile).pipeline
            runtime_key = "paddle_doc" if family in {"vl", "structure"} else "paddle"
            add("paddle", profile_label(profile), profile=profile, reason="TARGET 文字存在验证" if reveal_mode else "当前 OCR 选择", runtime_key=runtime_key)

        role_backends = (("target", tgt),) if reveal_mode else (("source", src), ("target", tgt))
        for role, backend in role_backends:
            err = _external_path_error(config, role, backend)
            if err:
                errors.append(err)
            normalized = str(backend or "").lower().replace("-", "_")
            if normalized == "manga_ocr":
                add("manga_ocr", "Manga OCR", reason=f"{role.upper()} OCR 明确选择 Manga OCR", runtime_key="manga_ocr")
            elif normalized == "baberu_ocr":
                add("baberu_ocr", "Baberu OCR", reason=f"{role.upper()} OCR 明确选择 Baberu OCR", runtime_key="baberu_ocr")
            elif normalized in {"ocr48px","48px"}:
                from .model_downloads import discovered_model_path
                native = discovered_model_path("ocr48px", getattr(config.ocr,"ocr48px_model_path",None))
                if native is not None:
                    add("ocr48px", "48px AR OCR", reason=f"{role.upper()} OCR 明确选择 48px AR", runtime_key="ocr48px")
                elif not getattr(config.ocr,"ocr48px_command",None):
                    errors.append("48px AR OCR 已选择，但原生模型尚未下载/导入，也没有配置外部 ocr48px_command。")

    # Registration is a geometric stage, not a semantic detector.  It must run
    # before Koharu so SOURCE/TARGET evidence can be compared in one coordinate
    # system.  Auto/SIFT stays cheap; explicit deep choices are prepared here.
    reg = str(config.registration.backend or "auto").lower()
    if reg == "lightglue":
        add("lightglue", "LightGlue", reason="页面配准明确选择 LightGlue")
    elif reg == "loftr":
        add("loftr", "LoFTR", reason="页面配准明确选择 LoFTR")

    for key in scheduled_detectors:
        if key == primary_key:
            continue
        role = "边界精修" if key == "sam2" else "辅助检测器"
        add(key, detector_labels[key], reason=f"{role} · {detector_strategy(config.bubbles)}", runtime_key=key)

    return RuntimePreflightPlan(tuple(rows), tuple(errors))


def model_artifact_ready(config: PipelineConfig, requirement: RuntimeRequirement) -> bool:
    """Check cached/local artifact presence without importing heavy runtimes."""
    key = requirement.key
    if key == "paddle":
        # Explicit local det+rec is authoritative for classic OCR profiles.
        if getattr(config.ocr, "paddle_text_detection_model_dir", None) and getattr(config.ocr, "paddle_text_recognition_model_dir", None):
            det = Path(str(config.ocr.paddle_text_detection_model_dir)).expanduser()
            rec = Path(str(config.ocr.paddle_text_recognition_model_dir)).expanduser()
            if det.is_dir() and rec.is_dir():
                return True
        ready, _ = paddle_profile_marker_status(requirement.profile)
        return bool(ready)

    local = model_local_paths()
    if key == "lightglue":
        return bool(local["lightglue"].is_file())
    if key == "loftr":
        return bool(local["loftr"].is_file())
    if key == "mangalens":
        configured = getattr(config.bubbles, "mangalens_model_path", None)
        path = Path(str(configured)).expanduser() if configured else local["mangalens"]
        return path.is_file()
    if key == "ysg_obb":
        configured = getattr(config.bubbles, "ysg_obb_model_path", None)
        path = Path(str(configured)).expanduser() if configured else local["ysg_obb"]
        return path.is_file()
    if key == "rtdetr_v2":
        configured = getattr(config.bubbles, "rtdetr_model_path", None)
        path = Path(str(configured)).expanduser() if configured else local["rtdetr_v2"]
        return path.is_dir() and all((path / name).is_file() for name in ("config.json", "preprocessor_config.json", "model.safetensors"))
    if key == "sam2":
        configured = getattr(config.bubbles, "sam2_checkpoint", None)
        path = Path(str(configured)).expanduser() if configured else local["sam2"]
        return path.is_file()
    if key == "koharu_layout":
        configured = getattr(config.bubbles,"koharu_layout_model_path",None) or getattr(config.ocr,"koharu_layout_model_path",None)
        path=Path(str(configured)).expanduser() if configured else local["koharu_layout"]
        return path.is_dir() and all((path/name).is_file() for name in ("inference_config.json","load_model.py","model.safetensors"))
    if key == "manga_ocr":
        configured=getattr(config.ocr,"manga_ocr_model_path",None); path=Path(str(configured)).expanduser() if configured else local["manga_ocr"]
        return path.is_dir() and all((path/name).is_file() for name in ("config.json","preprocessor_config.json","model.safetensors"))
    if key == "baberu_ocr":
        configured=getattr(config.ocr,"baberu_ocr_model_path",None); path=Path(str(configured)).expanduser() if configured else local["baberu_ocr"]
        return path.is_dir() and all((path/name).is_file() for name in ("onnx_infer.py","onnx/vision_int4.onnx","onnx/decoder_prefill_int8.onnx","onnx/decoder_step_int8.onnx","tokenizer/vocab.json"))
    return True


def pending_model_requirements(config: PipelineConfig, plan: RuntimePreflightPlan | None = None) -> tuple[RuntimeRequirement, ...]:
    p = plan or plan_runtime_requirements(config)
    return tuple(row for row in p.requirements if not model_artifact_ready(config, row))


__all__ = [
    "RuntimeRequirement", "RuntimePreflightPlan", "required_paddle_profiles",
    "plan_runtime_requirements", "model_artifact_ready", "pending_model_requirements",
]
