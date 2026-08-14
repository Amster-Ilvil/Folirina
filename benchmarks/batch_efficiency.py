from __future__ import annotations
import json, tempfile, time
from pathlib import Path
import cv2
import numpy as np

from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.io_utils import write_image
from manga_hd_transfer.pipeline import TransferPipeline


def page(i: int):
    h,w=900,700
    target=np.full((h,w,3),255,np.uint8)
    cv2.rectangle(target,(30,30),(670,870),(0,0,0),4)
    cv2.line(target,(50,500+i*3),(650,470+i*2),(20,20,20),3)
    cv2.circle(target,(520-i*7,650+i*5),95,(20,20,20),4)
    cx=160+(i%3)*150; cy=230+(i//3)*190
    cv2.ellipse(target,(cx,cy),(85,105),0,0,360,(0,0,0),4)
    cv2.ellipse(target,(cx,cy),(80,100),0,0,360,(255,255,255),-1)
    for x in (cx-25,cx,cx+25): cv2.rectangle(target,(x,cy-35),(x+6,cy+35),(0,0,0),-1)
    source=target.copy()
    cv2.ellipse(source,(cx,cy),(80,100),0,0,360,(255,255,255),-1)
    cv2.rectangle(source,(cx-48,cy-22),(cx+48,cy-10),(0,0,0),-1)
    cv2.rectangle(source,(cx-38,cy+12),(cx+38,cy+24),(0,0,0),-1)
    return source,target


def main():
    with tempfile.TemporaryDirectory(prefix='mhd-batch-') as td:
        root=Path(td); sd=root/'src'; tg=root/'target'; out=root/'out'; sd.mkdir(); tg.mkdir()
        for i in range(6):
            s,t=page(i); write_image(sd/f'{i:03d}.png',s); write_image(tg/f'{i:03d}.png',t)
        cfg=PipelineConfig(); cfg.transfer.mode='mask_replace'; cfg.registration.backend='auto'; cfg.export.layer_bundle=False; cfg.export.save_debug=False
        cfg.mask_replace.paired_diff_min_registration_confidence=.80
        st=time.perf_counter(); first=TransferPipeline(cfg).run_book(sd,tg,out); first_s=time.perf_counter()-st
        st=time.perf_counter(); second=TransferPipeline(cfg).run_book(sd,tg,out); second_s=time.perf_counter()-st
        report={
            'pages':6, 'first_run_seconds':round(first_s,4), 'resume_run_seconds':round(second_s,4),
            'speedup':round(first_s/max(second_s,1e-6),2), 'resume_hits':second.meta.get('resumed_count',0),
            'fast_route_pages':sum(1 for p in first.pages if p.registration.method.startswith('fast-phase-identity')),
            'qa_errors':first.meta.get('qa_errors',0), 'failed':first.meta.get('failed_count',0),
        }
        print(json.dumps(report,ensure_ascii=False,indent=2))
        Path('benchmarks/batch_efficiency_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
