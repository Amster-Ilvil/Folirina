from __future__ import annotations

"""Hardware-adaptive scheduling, resolution and bounded-prefetch policy.

The historical ``M6RuntimePlan`` name is retained for compatibility, but the
policy is now cross-device: live Apple Silicon/CUDA capability is detected
first, memory pressure is read at recommendation time, and only known default
detector sizes are changed. Explicit user sizes remain authoritative.
"""

from dataclasses import dataclass
from typing import Any
import os

from .accelerator_policy import AppleHardwareInfo, detect_apple_hardware, recommended_cpu_threads, recommended_mps_memory_fraction, resolve_precision, memory_budget_gb, recommended_batch


@dataclass(frozen=True, slots=True)
class M6RuntimePlan:
    enabled: bool
    profile: str
    memory_gb: float | None
    cpu_threads: int
    mps_memory_fraction: float
    mps_precision: str
    decode_prefetch_enabled: bool
    decode_prefetch_pages: int
    max_inflight_images: int
    ocr_batch_size: int
    koharu_shape: int
    koharu_postprocess_max_side: int
    mangalens_imgsz: int
    ysg_obb_imgsz: int
    rtdetr_imgsz: int
    accelerator_backend: str = "cpu"
    accelerator_name: str = ""
    available_memory_gb: float | None = None
    hardware_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "profile": self.profile,
            "memory_gb": self.memory_gb,
            "cpu_threads": self.cpu_threads,
            "mps_memory_fraction": self.mps_memory_fraction,
            "mps_precision": self.mps_precision,
            "decode_prefetch_enabled": self.decode_prefetch_enabled,
            "decode_prefetch_pages": self.decode_prefetch_pages,
            "max_inflight_images": self.max_inflight_images,
            "ocr_batch_size": self.ocr_batch_size,
            "koharu_shape": self.koharu_shape,
            "koharu_postprocess_max_side": self.koharu_postprocess_max_side,
            "mangalens_imgsz": self.mangalens_imgsz,
            "ysg_obb_imgsz": self.ysg_obb_imgsz,
            "rtdetr_imgsz": self.rtdetr_imgsz,
            "accelerator_backend": self.accelerator_backend,
            "accelerator_name": self.accelerator_name,
            "available_memory_gb": self.available_memory_gb,
            "hardware_fingerprint": self.hardware_fingerprint,
        }


AdaptiveRuntimePlan = M6RuntimePlan


def _accelerator_auto_enabled(config: Any | None, hw: AppleHardwareInfo) -> bool:
    explicit = getattr(config, "m6_auto", True) if config is not None else True
    if not bool(explicit):
        return False
    # Hardware scheduling must not depend on whether the lightweight GUI process
    # imports Torch/MLX. Folirina intentionally hosts heavy Apple-Silicon models
    # in isolated workers, so packaged GUI builds can report ``apple_gpu`` while
    # the worker itself has a fully functional MPS runtime.  Use hardware
    # identity for sizing/resource policy; model preflight still validates the
    # concrete worker backend before inference.
    real_apple_accel = bool(
        hw.is_apple_silicon
        and (hw.mps_available or hw.mlx_available or hw.accelerator_backend == "mps")
    )
    # M6 packaged GUI builds intentionally keep Torch/MLX out of the main
    # interpreter, so hardware detection reports ``apple_gpu`` even though the
    # isolated worker provides MPS. Limit this packaging exception to M6; older
    # Apple hardware without any accelerator runtime keeps the legacy disabled
    # semantics.
    packaged_m6 = bool(hw.is_m6 and hw.accelerator_backend == "apple_gpu")
    synthetic_m6 = bool(hw.is_m6 and hw.available_memory_gb is None and hw.accelerator_backend == "cpu")
    return bool(real_apple_accel or packaged_m6 or synthetic_m6 or hw.accelerator_backend in {"cuda", "directml", "migraphx", "openvino"})


def _tier(memory_gb: float | None, *, family: str = "apple") -> str:
    mem = float(memory_gb or 0.0)
    prefix = family.lower() or "apple"
    if mem <= 17.0:
        return f"{prefix}_conservative"
    if mem <= 25.0:
        return f"{prefix}_balanced"
    return f"{prefix}_performance"


def effective_model_size(model: str, requested: int, *, image_shape: tuple[int, int] | None = None,
                         hw: AppleHardwareInfo | None = None, dynamic: bool = True) -> int:
    """Return a safe model input side while keeping explicit custom settings.

    For M6 ``requested`` values equal to project defaults are considered auto
    defaults and can move between memory tiers. Any non-default custom value is
    retained and only bounded by a conservative hard ceiling.
    """
    hw = hw or detect_apple_hardware()
    base = max(256, int(requested))
    if not hw.is_apple_silicon or not dynamic:
        return base
    model = str(model or "").lower().strip()
    defaults = {
        "koharu_layout": 1152,
        "mangalens": 1600,
        "ysg_obb": 1600,
        "rtdetr_v2": 640,
    }
    caps = {
        "conservative": {"koharu_layout": 1152, "mangalens": 1152, "ysg_obb": 1152, "rtdetr_v2": 640},
        "balanced": {"koharu_layout": 1344, "mangalens": 1344, "ysg_obb": 1344, "rtdetr_v2": 768},
        "performance": {"koharu_layout": 1536, "mangalens": 1536, "ysg_obb": 1536, "rtdetr_v2": 896},
    }
    mem = float(hw.memory_gb or 0.0)
    tier = "conservative" if mem <= 17.0 else ("balanced" if mem <= 25.0 else "performance")
    # Unified-memory pressure may only move an automatic default downward.  It
    # never changes an explicit custom size. This prevents a 32 GB M6 with only
    # ~2 GB free from still attempting the performance detector resolution.
    available = hw.available_memory_gb
    if hw.is_m6 and available is not None:
        live = float(available)
        if 0.0 < live < 5.0:
            tier = "conservative"
        elif 0.0 < live < 8.0 and tier == "performance":
            tier = "balanced"
    hard_cap = int(caps[tier].get(model, base)) if hw.is_m6 else base
    # Explicit user values are authoritative. Only M6 auto-defaults are adapted.
    # For defaults the adaptive value replaces the shipped default; using max()
    # here would prevent 1600px MangaLens/YSG defaults from ever down-clamping.
    if base == defaults.get(model):
        return hard_cap
    return base


def build_m6_runtime_plan(config: Any | None = None, *, hw: AppleHardwareInfo | None = None) -> M6RuntimePlan:
    hw = hw or detect_apple_hardware()
    enabled = _accelerator_auto_enabled(config, hw)
    try:
        from .runtime_calibration import hardware_fingerprint
        fingerprint = hardware_fingerprint(hw)
    except Exception:
        fingerprint = ""
    if not enabled:
        return M6RuntimePlan(
            enabled=False, profile=hw.profile, memory_gb=hw.memory_gb,
            cpu_threads=int(getattr(config, "max_cpu_threads", 8) or 8) if config is not None else 8,
            mps_memory_fraction=float(getattr(config, "mps_memory_fraction", .82) if config is not None else .82),
            mps_precision=resolve_precision("mps", str(getattr(config, "mps_precision", "auto") if config is not None else "auto"), hw),
            decode_prefetch_enabled=False, decode_prefetch_pages=1, max_inflight_images=1,
            ocr_batch_size=1, koharu_shape=1152, koharu_postprocess_max_side=1152,
            mangalens_imgsz=1600, ysg_obb_imgsz=1600, rtdetr_imgsz=640,
            accelerator_backend=hw.accelerator_backend, accelerator_name=hw.accelerator_name,
            available_memory_gb=hw.available_memory_gb, hardware_fingerprint=fingerprint,
        )

    profile = _tier(hw.memory_gb, family=(hw.generation.lower() or "apple"))
    if config is not None and not bool(getattr(config, "cpu_thread_auto", True)):
        threads = max(1, int(getattr(config, "max_cpu_threads", 8) or 8))
    else:
        threads = recommended_cpu_threads(hw, default=int(getattr(config, "max_cpu_threads", 8) or 8) if config is not None else 8)
    if config is not None and not bool(getattr(config, "mps_memory_auto", True)):
        fraction = float(getattr(config, "mps_memory_fraction", recommended_mps_memory_fraction(hw)))
    else:
        fraction = recommended_mps_memory_fraction(hw)
    precision = resolve_precision("mps", str(getattr(config, "mps_precision", "auto") if config is not None else "auto"), hw)
    base_sizes = (1152, 1152, 1152, 640)
    if hw.is_m6:
        size_map = {
            "conservative": (1152, 1152, 1152, 640),
            "balanced": (1344, 1344, 1344, 768),
            "performance": (1536, 1536, 1536, 896),
        }
        tier = "conservative" if float(hw.memory_gb or 0.0) <= 17.0 else ("balanced" if float(hw.memory_gb or 0.0) <= 25.0 else "performance")
        if hw.available_memory_gb is not None:
            live = float(hw.available_memory_gb)
            if 0.0 < live < 5.0:
                tier = "conservative"
            elif 0.0 < live < 8.0 and tier == "performance":
                tier = "balanced"
        base_sizes = size_map[tier]
    auto_prefetch = bool(getattr(config, "decode_prefetch_auto", True) if config is not None else True)
    # Decode look-ahead is CPU/I/O work and should follow the hardware plan,
    # not whether the lightweight GUI interpreter imports Torch. Apple Silicon
    # workers can expose MPS independently of the packaged GUI environment.
    real_accelerator = bool(
        (hw.is_apple_silicon and (hw.mps_available or hw.mlx_available or hw.accelerator_backend == "mps"))
        or (hw.is_m6 and hw.accelerator_backend == "apple_gpu")
        or (hw.is_m6 and hw.available_memory_gb is None and hw.accelerator_backend == "cpu")
        or hw.accelerator_backend in {"cuda", "directml", "migraphx", "openvino"}
    )
    synthetic_m6 = bool(hw.is_m6 and hw.available_memory_gb is None and hw.accelerator_backend == "cpu")
    prefetch = auto_prefetch and (real_accelerator or synthetic_m6)
    if prefetch and not synthetic_m6 and memory_budget_gb(hw) < 3.0:
        prefetch = False
    if hw.is_m6 and hw.available_memory_gb is None:
        # Backward-compatible deterministic profile used by existing callers
        # that provide only total RAM/chip facts. Real detection always carries
        # live available-memory information and uses the adaptive formula below.
        batch = int({"conservative": 4, "balanced": 6, "performance": 8}["conservative" if float(hw.memory_gb or 0.0) <= 17.0 else ("balanced" if float(hw.memory_gb or 0.0) <= 25.0 else "performance")])
    else:
        batch = recommended_batch(hw, memory_cost_per_item=0.95, cap=8, cpu_default=4)
    if hw.is_m6 and hw.available_memory_gb is None:
        inflight = 2 if float(hw.memory_gb or 0.0) > 25.0 else 1
        prefetch_pages = 2 if float(hw.memory_gb or 0.0) > 25.0 else 1
    else:
        live_budget = memory_budget_gb(hw)
        shared_accelerator = bool(hw.unified_memory or hw.accelerator_backend in {"mps", "mlx", "apple_gpu"})
        inflight = 2 if live_budget >= 8.0 and (hw.is_apple_silicon or hw.accelerator_backend in {"cuda", "directml", "migraphx"}) else 1
        # Keep heavy model execution single-lane. Prefetch only CPU/I/O.
        prefetch_pages = 2 if live_budget >= 10.0 and (hw.is_apple_silicon or hw.accelerator_backend in {"cuda", "directml", "migraphx"}) else 1
    return M6RuntimePlan(
        enabled=True, profile=profile, memory_gb=hw.memory_gb, cpu_threads=threads,
        mps_memory_fraction=fraction, mps_precision=precision, decode_prefetch_enabled=prefetch,
        decode_prefetch_pages=prefetch_pages, max_inflight_images=inflight, ocr_batch_size=batch,
        koharu_shape=base_sizes[0], koharu_postprocess_max_side=base_sizes[1],
        mangalens_imgsz=base_sizes[2], ysg_obb_imgsz=base_sizes[2], rtdetr_imgsz=base_sizes[3],
        accelerator_backend=hw.accelerator_backend, accelerator_name=hw.accelerator_name,
        available_memory_gb=hw.available_memory_gb, hardware_fingerprint=fingerprint,
    )


def choose_detector_size(model: str, requested: int, image_shape: tuple[int, int] | None = None,
                         config: Any | None = None) -> int:
    hw = detect_apple_hardware()
    enabled = _accelerator_auto_enabled(config, hw)
    dynamic = bool(getattr(config, "m6_dynamic_resolution_auto", True) if config is not None else True)
    return effective_model_size(model, requested, image_shape=image_shape, hw=hw, dynamic=(enabled and dynamic))


def choose_ocr_batch_size(backend: str, requested: int, config: Any | None = None, *, hw: AppleHardwareInfo | None = None) -> int:
    """Return a safe OCR batch size for the selected backend.

    Hayai and 48px expose real batched inference; Baberu currently stays
    conservative because its public wrapper is single-image oriented; NDL is a
    full-page subprocess backend and therefore uses one page per request.
    """
    plan = build_m6_runtime_plan(config, hw=hw)
    base = max(1, int(requested or 1))
    name = str(backend or "").lower().strip()
    if name in {"hayai_ocr", "ocr48px", "manga_ocr"}:
        cost = 0.72 if name == "hayai_ocr" else 0.22
        cap = 12 if name == "hayai_ocr" else 16
        if name == "manga_ocr":
            cap = 12
        calibrated = recommended_batch(hw or detect_apple_hardware(), memory_cost_per_item=cost, cap=cap, cpu_default=2)
        if plan.enabled:
            try:
                from .runtime_calibration import safe_batch_ceiling
                ceiling = safe_batch_ceiling(name)
                if ceiling:
                    calibrated = min(calibrated, int(ceiling))
            except Exception:
                pass
        return min(base, calibrated if plan.enabled else base)
    if name == "baberu_ocr":
        calibrated = recommended_batch(hw or detect_apple_hardware(), memory_cost_per_item=0.5, cap=2, cpu_default=1)
        if plan.enabled:
            try:
                from .runtime_calibration import safe_batch_ceiling
                ceiling = safe_batch_ceiling(name)
                if ceiling:
                    calibrated = min(calibrated, int(ceiling))
            except Exception:
                pass
        return min(base, calibrated if plan.enabled else 1)
    return 1


def m6_ocr_backend_order(*, prefer: str = "auto", japanese_vertical: bool = True, config: Any | None = None, hw: AppleHardwareInfo | None = None) -> tuple[str, ...]:
    """Stable preference order for the user's Japanese manga OCR stack.

    This function does not change the existing Paddle default. It is consumed
    only by the explicit M6 Japanese-auto backend and diagnostics.
    """
    key = str(prefer or "auto").lower().strip()
    if key in {"hayai_ocr", "ndlocr_lite", "ocr48px"}:
        return (key,)
    if japanese_vertical:
        return ("hayai_ocr", "ocr48px", "ndlocr_lite")
    return ("hayai_ocr", "ndlocr_lite", "ocr48px")


def m6_cache_token(config: Any | None = None, *, hw: AppleHardwareInfo | None = None) -> str:
    """Broad M6 policy discriminator retained for whole-pipeline compatibility."""
    hw = hw or detect_apple_hardware()
    plan = build_m6_runtime_plan(config, hw=hw)
    return f"m6:{int(plan.enabled)}:{plan.profile}:{plan.memory_gb}:{plan.koharu_shape}:{plan.mangalens_imgsz}:{plan.ysg_obb_imgsz}:{plan.rtdetr_imgsz}:{plan.ocr_batch_size}"


def m6_layout_cache_token(config: Any | None = None, *, hw: AppleHardwareInfo | None = None) -> str:
    """Stage-specific M6 token for Koharu Layout cache identity.

    Only values that can change Koharu's actual output belong here.  OCR batch
    size, unrelated detector sizes, CPU look-ahead and other pipeline knobs must
    not invalidate a layout cache that is otherwise identical.
    """
    hw = hw or detect_apple_hardware()
    plan = build_m6_runtime_plan(config, hw=hw)
    return (
        f"m6-layout-v1:{int(plan.enabled)}:{plan.profile}:{plan.memory_gb}:"
        f"{plan.koharu_shape}:{plan.koharu_postprocess_max_side}:{plan.mps_precision}"
    )


def should_auto_decode_prefetch(config: Any | None = None, *, hw: AppleHardwareInfo | None = None) -> bool:
    hw = hw or detect_apple_hardware()
    # Decode look-ahead is CPU/I/O work and should follow the hardware plan,
    # not whether the lightweight GUI interpreter imports Torch. Apple Silicon
    # workers can expose MPS independently of the packaged GUI environment.
    real_accelerator = bool(
        (hw.is_apple_silicon and (hw.mps_available or hw.mlx_available or hw.accelerator_backend == "mps"))
        or (hw.is_m6 and hw.accelerator_backend == "apple_gpu")
        or (hw.is_m6 and hw.available_memory_gb is None and hw.accelerator_backend == "cpu")
        or hw.accelerator_backend in {"cuda", "directml", "migraphx", "openvino"}
    )
    synthetic_m6 = bool(hw.is_m6 and hw.available_memory_gb is None and hw.accelerator_backend == "cpu")
    if not (real_accelerator or synthetic_m6):
        return False
    if not synthetic_m6 and memory_budget_gb(hw) < 3.0:
        return False
    if config is None:
        return True
    explicit = bool(getattr(config, "decode_prefetch_enabled", False))
    opt_in = bool(getattr(config, "decode_prefetch_experimental_opt_in", False))
    auto = bool(getattr(config, "decode_prefetch_auto", True))
    return bool(auto or (explicit and opt_in))


__all__ = [
    "M6RuntimePlan", "AdaptiveRuntimePlan", "build_m6_runtime_plan", "choose_detector_size",
    "effective_model_size", "choose_ocr_batch_size", "m6_ocr_backend_order",
    "m6_cache_token", "m6_layout_cache_token", "should_auto_decode_prefetch",
]
