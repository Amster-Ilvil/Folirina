import numpy as np

from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.coordinate_space import SourceCoordinateSpace
from manga_hd_transfer.models import RegistrationResult
from manga_hd_transfer.plugins import REGISTRY
# imports register providers
import manga_hd_transfer.source_detectors  # noqa: F401
import manga_hd_transfer.direct_containers  # noqa: F401
import manga_hd_transfer.structure_registration  # noqa: F401
from manga_hd_transfer.source_detectors import detect_source_rtdetr_v2, detect_source_sam2, run_source_detector_chain


def test_provider_registry_closes_reference_integration_gaps():
    assert {"pseudo_text_barrier", "sidecar", "ctd_sidecar", "debubble_white", "rtdetr_v2", "sam2", "mangalens"} <= set(REGISTRY.names("source_detector"))
    assert "structure_ecc" in REGISTRY.names("registration_refiner")
    assert {"progressive_border", "colored_text_components"} <= set(REGISTRY.names("mask_refiner"))
    assert "source_direct_invariants" in REGISTRY.names("qa_check")


def test_semantic_fallbacks_do_not_download_by_default():
    cfg = PipelineConfig()
    im = np.full((80, 100, 3), 255, np.uint8)
    assert detect_source_rtdetr_v2(im, cfg.mask_replace, cfg.bubbles) == []
    assert detect_source_sam2(im, cfg.mask_replace, cfg.bubbles) == []


def test_expensive_only_chain_does_not_repeat_cheap_providers():
    cfg = PipelineConfig()
    im = np.full((80, 100, 3), 255, np.uint8)
    _, audit = run_source_detector_chain(im, cfg.mask_replace, cfg.bubbles, allow_expensive=True, only_expensive=True)
    names = [row["provider"] for row in audit]
    assert "pseudo_text_barrier" not in names
    assert "sidecar" not in names
    assert {"rtdetr_v2", "sam2", "mangalens"} <= set(names)


def test_canonical_coordinate_space_projects_affine_to_local_similarity():
    reg = RegistrationResult(
        matrix=np.array([[0.75, 0.01, 12.0], [-0.005, 0.77, 20.0], [0.0, 0.0, 1.0]], dtype=float),
        method="test", confidence=1.0, inlier_ratio=1.0, reprojection_error=0.0,
        spatial_coverage=1.0, num_matches=10, source_size=(1440, 2048), target_size=(1117, 1600), diagnostics={},
    )
    cs = SourceCoordinateSpace.from_registration(reg)
    sim = cs.local_similarity(700, 900)
    assert sim.scale > 0
    assert sim.anisotropy > 0
    x, y = cs.map_point(700, 900)
    assert abs(x - sim.target_x) < 1e-9 and abs(y - sim.target_y) < 1e-9
