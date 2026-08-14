from __future__ import annotations
import cv2
from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.io_utils import write_image
from manga_hd_transfer.models import PagePair
from manga_hd_transfer.pipeline import TransferPipeline
from .helpers import make_art_page


def test_registration_stage_cache_reused_when_page_is_regenerated(tmp_path):
    target=make_art_page(700,900); source=target.copy()
    cv2.rectangle(source,(170,260),(290,330),(255,255,255),-1); cv2.rectangle(source,(190,280),(270,292),(0,0,0),-1)
    sp,tp=tmp_path/'s.png',tmp_path/'t.png'; write_image(sp,source); write_image(tp,target)
    pair=PagePair(str(sp),str(tp),0,0,.99,.01,[])
    cfg=PipelineConfig(); cfg.ocr.backend='none'; cfg.transfer.mode='mask_replace'; cfg.registration.backend='auto'; cfg.export.layer_bundle=False
    pipe=TransferPipeline(cfg); page=tmp_path/'page'
    first=pipe.process_page(pair,page)
    second=pipe.process_page(pair,page)
    assert second.meta['cache']['registration']=='hit'
    assert second.registration.method.endswith('+cache')
