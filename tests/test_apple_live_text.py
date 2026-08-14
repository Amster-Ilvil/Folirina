from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from manga_hd_transfer.apple_live_text import AppleAutoLiveTextBackend, AppleLiveTextError
from manga_hd_transfer.config import OCRConfig, PipelineConfig
from manga_hd_transfer.models import BubbleInstance, TextBlock
from manga_hd_transfer.ocr import OCRBackend, build_backend
from manga_hd_transfer.pipeline import TransferPipeline
from manga_hd_transfer.cache import PageStageCache


def _rect_mask(shape, box):
    h, w = shape
    x0, y0, x1, y1 = box
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def _bubble(bid, box, shape, *, target=None, source=None):
    x0, y0, x1, y1 = box
    poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    meta = {}
    if target:
        meta["paired_target_id"] = target
    if source:
        meta["paired_source_id"] = source
    mask = _rect_mask(shape, box)
    return BubbleInstance(bid, poly, 0.95, "speech", [], mask, mask.copy(), meta)


def test_apple_alias_now_builds_live_text_auto_backend():
    backend = build_backend(OCRConfig(backend="apple"), "ch", "apple")
    assert isinstance(backend, AppleAutoLiveTextBackend)
    assert getattr(backend, "region_text_only", False) is True


def test_auto_live_text_falls_back_once_and_stays_on_shortcut():
    cfg = OCRConfig(backend="apple")
    backend = AppleAutoLiveTextBackend("ch", cfg)
    calls = {"helper": 0, "shortcut": 0}

    class FailingHelper:
        def recognize(self, image, image_path=None):
            calls["helper"] += 1
            raise AppleLiveTextError("VisionKit unavailable")
        def close(self):
            pass

    class Shortcut:
        def recognize(self, image, image_path=None):
            calls["shortcut"] += 1
            h, w = image.shape[:2]
            return [TextBlock("s", [(0,0),(w,0),(w,h),(0,h)], "中文", .9, meta={"backend":"apple_shortcut"})]

    backend._get_helper = lambda: FailingHelper()  # type: ignore[method-assign]
    backend._get_shortcut = lambda: Shortcut()  # type: ignore[method-assign]
    img = np.full((80, 100, 3), 255, np.uint8)
    first = backend.recognize(img)
    second = backend.recognize(img)
    assert first[0].text == "中文" and second[0].text == "中文"
    assert calls == {"helper": 1, "shortcut": 2}
    assert first[0].meta["apple_auto_route"] == "shortcut"
    assert "VisionKit unavailable" in first[0].meta["apple_live_text_fallback_reason"]


class RegionTextBackend(OCRBackend):
    region_text_only = True
    def recognize(self, image, *, image_path=None):
        h, w = image.shape[:2]
        return [TextBlock("tmp", [(0,0),(w,0),(w,h),(0,h)], "测试中文", .91, meta={"backend":"fake_live_text", "apple_auto_route":"visionkit_live_text"})]


def test_paired_region_text_only_uses_bubble_geometry_without_target_ocr(tmp_path: Path):
    cfg = PipelineConfig()
    cfg.cache.enabled = False
    cfg.cache.ocr = False
    source = np.full((220, 180, 3), 255, np.uint8)
    cv2.putText(source, "ABC", (45, 100), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2)
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)

    sb = _bubble("photo-src-000", (30, 40, 120, 150), source.shape[:2], target="photo-dst-000")
    tb = _bubble("photo-dst-000", (35, 45, 125, 155), source.shape[:2], source="photo-src-000")
    pipeline = TransferPipeline(cfg, source_ocr=RegionTextBackend(), target_ocr=RegionTextBackend())
    blocks, target_blocks, source_bubbles, target_bubbles = pipeline._recognize_paired_regions_text_only(
        pipeline.source_ocr, source, source_path, [sb], [tb], PageStageCache(tmp_path / "page", enabled=False), {}
    )
    assert len(blocks) == 1 and blocks[0].text == "测试中文"
    assert blocks[0].polygon == sb.polygon
    assert blocks[0].bubble_id == sb.id
    assert blocks[0].meta["paired_target_id"] == tb.id
    assert source_bubbles[0].block_ids == [blocks[0].id]
    assert len(target_blocks) == 1
    assert target_blocks[0].polygon == tb.polygon
    assert target_blocks[0].meta["synthetic_geometry_only"] is True
    assert target_bubbles[0].block_ids == [target_blocks[0].id]


def test_auto_live_text_soft_fail_disables_both_routes_after_first_total_failure():
    cfg = OCRConfig(backend="apple", apple_live_text_soft_fail=True)
    backend = AppleAutoLiveTextBackend("ch", cfg)
    calls = {"helper": 0, "shortcut": 0}

    class FailingHelper:
        def recognize(self, image, image_path=None):
            calls["helper"] += 1
            raise AppleLiveTextError("helper failed")
        def close(self):
            pass

    class FailingShortcut:
        def recognize(self, image, image_path=None):
            calls["shortcut"] += 1
            raise AppleLiveTextError("shortcut failed")

    backend._get_helper = lambda: FailingHelper()  # type: ignore[method-assign]
    backend._get_shortcut = lambda: FailingShortcut()  # type: ignore[method-assign]
    img = np.full((40, 60, 3), 255, np.uint8)
    assert backend.recognize(img) == []
    assert backend.recognize(img) == []
    assert calls == {"helper": 1, "shortcut": 1}


def test_paired_region_text_only_records_vertical_ink_orientation_hint(tmp_path: Path):
    cfg = PipelineConfig()
    cfg.cache.enabled = False
    cfg.cache.ocr = False
    source = np.full((240, 180, 3), 255, np.uint8)
    # Four stacked glyph-like blocks: transcript backend itself has no bbox,
    # so direction must be inferred from source ink in the known balloon crop.
    for y in (60, 88, 116, 144):
        cv2.rectangle(source, (78, y), (92, y + 16), (0, 0, 0), -1)
    source_path = tmp_path / "source_vertical.png"
    cv2.imwrite(str(source_path), source)
    sb = _bubble("photo-src-000", (45, 35, 125, 190), source.shape[:2], target="photo-dst-000")
    tb = _bubble("photo-dst-000", (50, 40, 130, 195), source.shape[:2], source="photo-src-000")
    pipeline = TransferPipeline(cfg, source_ocr=RegionTextBackend(), target_ocr=RegionTextBackend())
    blocks, *_ = pipeline._recognize_paired_regions_text_only(
        pipeline.source_ocr, source, source_path, [sb], [tb],
        PageStageCache(tmp_path / "page_vertical", enabled=False), {},
    )
    assert blocks[0].meta["orientation_hint"] == "vertical"
    assert blocks[0].meta["orientation_evidence"]["reason"] in {
        "ink_bbox_tall", "component_flow_vertical", "ambiguous_cjk_manga_default"
    }


def test_paired_region_text_only_records_horizontal_ink_orientation_hint(tmp_path: Path):
    cfg = PipelineConfig()
    cfg.cache.enabled = False
    cfg.cache.ocr = False
    source = np.full((180, 260, 3), 255, np.uint8)
    for x in (65, 95, 125, 155, 185):
        cv2.rectangle(source, (x, 84), (x + 18, 100), (0, 0, 0), -1)
    source_path = tmp_path / "source_horizontal.png"
    cv2.imwrite(str(source_path), source)
    sb = _bubble("photo-src-000", (35, 55, 225, 130), source.shape[:2], target="photo-dst-000")
    tb = _bubble("photo-dst-000", (40, 60, 230, 135), source.shape[:2], source="photo-src-000")
    pipeline = TransferPipeline(cfg, source_ocr=RegionTextBackend(), target_ocr=RegionTextBackend())
    blocks, *_ = pipeline._recognize_paired_regions_text_only(
        pipeline.source_ocr, source, source_path, [sb], [tb],
        PageStageCache(tmp_path / "page_horizontal", enabled=False), {},
    )
    assert blocks[0].meta["orientation_hint"] == "horizontal"
