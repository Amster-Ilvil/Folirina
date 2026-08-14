from __future__ import annotations
from pathlib import Path
import cv2
from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.io_utils import write_image
from manga_hd_transfer.pipeline import TransferPipeline
from .helpers import make_art_page


def _make_pair(target_path: Path, source_path: Path, offset: int):
    target=make_art_page(700,900)
    source=target.copy()
    x=120+offset*40
    cv2.ellipse(target,(x+90,300),(75,95),0,0,360,(0,0,0),4)
    cv2.ellipse(target,(x+90,300),(70,90),0,0,360,(255,255,255),-1)
    for xx in (x+65,x+90,x+115): cv2.rectangle(target,(xx,270),(xx+5,330),(0,0,0),-1)
    source=target.copy()
    cv2.ellipse(source,(x+90,300),(70,90),0,0,360,(255,255,255),-1)
    cv2.rectangle(source,(x+45,280),(x+135,291),(0,0,0),-1)
    cv2.rectangle(source,(x+55,310),(x+125,321),(0,0,0),-1)
    write_image(target_path,target); write_image(source_path,source)


def test_batch_resume_skips_completed_pages(tmp_path):
    sd,td,out=tmp_path/'src',tmp_path/'dst',tmp_path/'out'
    sd.mkdir(); td.mkdir()
    for i in range(2): _make_pair(td/f'{i:03}.png',sd/f'{i:03}.png',i)
    cfg=PipelineConfig(); cfg.transfer.mode='mask_replace'; cfg.registration.backend='auto'; cfg.export.layer_bundle=False
    cfg.mask_replace.paired_diff_min_registration_confidence=.80
    first=TransferPipeline(cfg).run_book(sd,td,out)
    assert len(first.pages)==2
    second=TransferPipeline(cfg).run_book(sd,td,out)
    assert len(second.pages)==2
    assert second.meta['resumed_count']==2
    assert all(p.meta.get('batch_resume_hit') for p in second.pages)
