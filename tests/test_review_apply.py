from __future__ import annotations

import json
import cv2

from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.io_utils import write_image
from manga_hd_transfer.models import PagePair, TextBlock
from manga_hd_transfer.ocr import InjectedOCRBackend
from manga_hd_transfer.pipeline import TransferPipeline
from manga_hd_transfer.review_apply import apply_review_page
from .helpers import draw_fake_text, make_art_page


def test_review_override_can_regenerate_page(tmp_path):
    target=make_art_page(1000,1400); box=(235,660,365,740); draw_fake_text(target,box,3)
    source=cv2.resize(target,(700,980)); sb=tuple(v*.7 for v in box)
    cv2.rectangle(source,(int(sb[0]),int(sb[1])),(int(sb[2]),int(sb[3])),(255,255,255),-1); draw_fake_text(source,sb,2)
    sp,tp=tmp_path/'s.png',tmp_path/'t.png'; write_image(sp,source); write_image(tp,target)
    s=TextBlock('s',[(sb[0],sb[1]),(sb[2],sb[1]),(sb[2],sb[3]),(sb[0],sb[3])],'初始中文',.99,reading_order=0)
    t=TextBlock('t',[(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])],'日本語',.99,reading_order=0)
    cfg=PipelineConfig(); cfg.registration.backend='opencv'; cfg.qa.registration_min_confidence=.3; cfg.matching.review_confidence=.3; cfg.qa.match_min_confidence=.3; cfg.export.layer_bundle=False
    page_dir=tmp_path/'out'; project=TransferPipeline(cfg,InjectedOCRBackend([s]),InjectedOCRBackend([t])).process_page(PagePair(str(sp),str(tp),0,0,.99,.01),page_dir)
    sid=project.source_units[0].id; tid=project.target_units[0].id
    (page_dir/'review_overrides.json').write_text(json.dumps({'text_overrides':{sid:'复核后的中文文本'},'match_overrides':{sid:tid},'accepted_source_units':[sid],'status':'approved'}),encoding='utf-8')
    out=apply_review_page(page_dir,cfg)
    assert out.exists()
    assert (page_dir/'review_applied.json').exists()
    payload=json.loads((page_dir/'review_applied.json').read_text(encoding='utf-8'))
    assert payload['status']=='approved'
    assert payload['lettering'][0]['text']=='复核后的中文文本'
