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
        cfg=PipelineConfig(); cfg.transfer.mode="reletter"; cfg.registration.backend="opencv"; cfg.registration.min_matches=6; cfg.registration.review_confidence=.35; cfg.qa.registration_min_confidence=.35; cfg.matching.review_confidence=.35; cfg.qa.match_min_confidence=.35; cfg.export.layer_bundle=False
        p=TransferPipeline(cfg,InjectedOCRBackend([sblock]),InjectedOCRBackend([tblock]))
        pair=PagePair(str(sp),str(tp),0,0,.99,.01,[])
        project=p.process_page(pair,root/"out")
        summary=qa_summary(project.qa)
        checks={
            "registration_confidence": project.registration.confidence >= .35,
            "auto_match": project.meta.get("auto_applied_count")==1,
            "lettering": bool(project.lettering and project.lettering[0].success),
            "qa": summary["pass"],
            "final_exists": (root/"out"/"final.png").exists(),
            "project_exists": (root/"out"/"project.json").exists(),
        }

        mcfg=cfg.model_copy(deep=True); mcfg.transfer.mode="mask_replace"
        mcfg.mask_replace.min_match_confidence=.30; mcfg.mask_replace.min_mask_iou=.58
        mcfg.mask_replace.min_target_coverage=.86; mcfg.mask_replace.max_spill_ratio=.18
        mcfg.mask_replace.local_fit="bbox"; mcfg.mask_replace.feather_px=0
        mp=TransferPipeline(mcfg,InjectedOCRBackend([sblock]),InjectedOCRBackend([tblock]))
        mproject=mp.process_page(pair,root/"mask_out")
        mask_meta = mproject.meta.get("mask_replace", {})
        mask_records = list(mask_meta.get("records", []) or [])
        mask_checks={
            "applied": int(mask_meta.get("applied_count") or 0) >= 1,
            "content_gate": all(
                (not bool(r.get("applied")))
                or bool(r.get("content_complete"))
                or str(r.get("content_check", "")) == "insufficient_source_ink_evidence"
                for r in mask_records
            ),
            "mask_qa": qa_summary(mproject.qa)["pass"],
            "layer_exists": (root/"mask_out"/"mask_transfer_layer.png").exists(),
            "mask_exists": (root/"mask_out"/"mask_transfer_mask.png").exists(),
            "records_exist": (root/"mask_out"/"mask_transfer.json").exists(),
        }
        # Independent Direct Patch contract: same-layout page, full SOURCE raster
        # interior, no OCR rewrite and direct-specific artifacts.
        def direct_page(cn: bool):
            img=np.full((900,650,3),255,np.uint8)
            cv2.rectangle(img,(25,25),(625,875),(0,0,0),4)
            cv2.line(img,(25,420),(625,420),(0,0,0),4)
            cv2.circle(img,(120,150),65,(0,0,0),4)
            cv2.rectangle(img,(430,520),(590,760),(0,0,0),4)
            cv2.ellipse(img,(330,255),(130,90),0,0,360,(0,0,0),4)
            pts=(
                [(300,220),(320,220),(340,220),(360,220),(300,245),(320,245),(340,245),(360,245),(300,270),(320,270),(340,270),(360,270)]
                if cn else
                [(290,215),(310,215),(330,215),(350,215),(370,215),(290,240),(310,240),(330,240),(350,240),(370,240),(290,265),(310,265),(330,265),(350,265),(370,265),(290,290),(330,290),(370,290)]
            )
            for x,y in pts: cv2.rectangle(img,(x,y),(x+8,y+12),(0,0,0),-1)
            return img

        dsp,dtp=root/"direct_cn.png",root/"direct_jp.png"
        write_image(dsp,direct_page(True)); write_image(dtp,direct_page(False))
        dcfg=PipelineConfig(); dcfg.transfer.mode="direct_patch"; dcfg.registration.backend="opencv"; dcfg.registration.min_matches=6; dcfg.export.layer_bundle=False; dcfg.qa.fail_empty_mask_replace=False
        dp=TransferPipeline(dcfg)
        dproject=dp.process_page(PagePair(str(dsp),str(dtp),0,0,.99,.01,[]),root/"direct_out")
        dmeta=dproject.meta.get("direct_patch",{})
        direct_checks={
            "mode_is_independent": dproject.meta.get("transfer_mode")=="direct_patch",
            "used": bool(dmeta.get("used")),
            "no_ocr": str(dproject.meta.get("cache",{}).get("ocr_source","")).startswith("skipped_source_direct"),
            "text_only_contract": dmeta.get("contract")=="text_only_target_background",
            "pair_precheck": bool(dproject.meta.get("page_pairing_check",{}).get("same_page")),
            "direct_layer_exists": (root/"direct_out"/"direct_patch_layer.png").exists(),
            "direct_json_exists": (root/"direct_out"/"direct_patch.json").exists(),
            "mask_artifacts_absent": not (root/"direct_out"/"mask_transfer_layer.png").exists() and not (root/"direct_out"/"mask_transfer_mask.png").exists(),
            "mask_meta_empty": not bool(dproject.meta.get("mask_replace",{}).get("used")) and not list(dproject.meta.get("mask_replace",{}).get("records",[]) or []),
        }

        return {
            "pass": all(checks.values()) and all(mask_checks.values()) and all(direct_checks.values()),
            "checks": checks,
            "mask_replace_checks": mask_checks,
            "direct_patch_checks": direct_checks,
            "registration": project.registration.to_dict(),
            "qa": summary,
        }
