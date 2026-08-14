from __future__ import annotations

import cv2

from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.io_utils import write_image
from manga_hd_transfer.models import PagePair, TextBlock
from manga_hd_transfer.ocr import InjectedOCRBackend
from manga_hd_transfer.pipeline import TransferPipeline
from .helpers import draw_fake_text, make_art_page


def test_end_to_end_pipeline_with_injected_ocr(tmp_path):
    target = make_art_page(1000,1400)
    # Text inside first bubble in target coordinates.
    target_box=(235,660,365,740)
    draw_fake_text(target,target_box,3)
    source = cv2.resize(target,(700,980),interpolation=cv2.INTER_AREA)
    # Replace target-language fake text with a slightly different source-language pattern.
    sx=lambda x: x*.7; sy=lambda y: y*.7
    source_box=(sx(target_box[0]),sy(target_box[1]),sx(target_box[2]),sy(target_box[3]))
    cv2.rectangle(source,(int(source_box[0]),int(source_box[1])),(int(source_box[2]),int(source_box[3])),(255,255,255),-1)
    draw_fake_text(source,source_box,2)

    sp=tmp_path/"cn.png"; tp=tmp_path/"jp.png"
    write_image(sp,source); write_image(tp,target)
    source_block=TextBlock("sblock",[(source_box[0],source_box[1]),(source_box[2],source_box[1]),(source_box[2],source_box[3]),(source_box[0],source_box[3])],"这是旧版中文译文。",.99,reading_order=0)
    target_block=TextBlock("tblock",[(target_box[0],target_box[1]),(target_box[2],target_box[1]),(target_box[2],target_box[3]),(target_box[0],target_box[3])],"日本語",.99,reading_order=0)
    cfg=PipelineConfig()
    cfg.registration.backend="opencv"
    cfg.registration.feature="sift"
    cfg.registration.min_matches=6
    cfg.registration.review_confidence=.35
    cfg.qa.registration_min_confidence=.35
    cfg.matching.review_confidence=.35
    cfg.qa.match_min_confidence=.35
    cfg.lettering.max_font_size=48
    cfg.lettering.min_font_size=12
    cfg.export.layer_bundle=False
    pipe=TransferPipeline(cfg,InjectedOCRBackend([source_block]),InjectedOCRBackend([target_block]))
    pair=PagePair(str(sp),str(tp),0,0,.99,.01,[])
    project=pipe.process_page(pair,tmp_path/"out")
    assert project.meta["auto_applied_count"]==1
    assert (tmp_path/"out"/"final.png").exists()
    assert (tmp_path/"out"/"project.json").exists()
    assert any(l.success for l in project.lettering)
    # No publication-blocking QA errors expected for the synthetic page.
    errors=[q for q in project.qa if q.severity=="error"]
    assert not errors, [q.to_dict() for q in errors]
