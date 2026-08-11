from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .io_utils import write_image
from .models import PagePair, TextBlock
from .ocr import InjectedOCRBackend
from .pipeline import TransferPipeline
from .qa import qa_summary


def _page(width=1000, height=1400):
    img=np.full((height,width,3),255,np.uint8)
    cv2.rectangle(img,(35,35),(width-35,height-35),(0,0,0),5)
    cv2.line(img,(40,480),(width-40,520),(0,0,0),4)
    cv2.circle(img,(180,250),85,(20,20,20),4)
    cv2.circle(img,(780,880),120,(20,20,20),5)
    cv2.ellipse(img,(300,700),(170,110),0,0,360,(0,0,0),5)
    cv2.ellipse(img,(720,300),(140,90),-10,0,360,(0,0,0),5)
    cv2.circle(img,(650,1050),100,(0,0,0),4)
    cv2.circle(img,(620,1020),8,(0,0,0),-1); cv2.circle(img,(680,1020),8,(0,0,0),-1)
    return img


def _fake_text(img, box, lines):
    x0,y0,x1,y1=map(int,box); gap=max(4,(y1-y0)//(lines*2+1)); y=y0+gap
    for _ in range(lines):
        cv2.rectangle(img,(x0+5,y),(x1-5,min(y+gap,y1-2)),(0,0,0),-1); y+=gap*2


def run_selftest() -> dict:
    with tempfile.TemporaryDirectory(prefix="mhd-selftest-") as td:
        root=Path(td)
        target=_page(); box=(235,660,365,740); _fake_text(target,box,3)
        source=cv2.resize(target,(700,980),interpolation=cv2.INTER_AREA)
        sb=tuple(v*.7 for v in box)
        cv2.rectangle(source,(int(sb[0]),int(sb[1])),(int(sb[2]),int(sb[3])),(255,255,255),-1); _fake_text(source,sb,2)
        sp,tp=root/"cn.png",root/"jp.png"; write_image(sp,source); write_image(tp,target)
        sblock=TextBlock("s",[(sb[0],sb[1]),(sb[2],sb[1]),(sb[2],sb[3]),(sb[0],sb[3])],"这是离线自检中文译文。",.99,reading_order=0)
        tblock=TextBlock("t",[(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])],"日本語",.99,reading_order=0)
        cfg=PipelineConfig(); cfg.registration.backend="opencv"; cfg.registration.min_matches=6; cfg.registration.review_confidence=.35; cfg.qa.registration_min_confidence=.35; cfg.matching.review_confidence=.35; cfg.qa.match_min_confidence=.35; cfg.export.layer_bundle=False
        p=TransferPipeline(cfg,InjectedOCRBackend([sblock]),InjectedOCRBackend([tblock]))
        project=p.process_page(PagePair(str(sp),str(tp),0,0,.99,.01,[]),root/"out")
        summary=qa_summary(project.qa)
        checks={
            "registration_confidence": project.registration.confidence >= .35,
            "auto_match": project.meta.get("auto_applied_count")==1,
            "lettering": bool(project.lettering and project.lettering[0].success),
            "qa": summary["pass"],
            "final_exists": (root/"out"/"final.png").exists(),
            "project_exists": (root/"out"/"project.json").exists(),
        }
        return {"pass": all(checks.values()), "checks": checks, "registration": project.registration.to_dict(), "qa": summary}
