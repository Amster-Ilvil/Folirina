from __future__ import annotations

import tempfile
from pathlib import Path
import json
import os
import sys
import socket
import threading
import tomllib

import cv2
import numpy as np

from .config import PipelineConfig
from .io_utils import write_image, read_image, save_json, load_json
from .models import PagePair, TextBlock, TextUnit, BubbleInstance
from .ocr import InjectedOCRBackend, NullOCRBackend, OCRBackend
from .pipeline import TransferPipeline
from .qa import qa_summary
from .review_apply import apply_manual_force_transfer_review, reset_manual_force_transfer_review, apply_review_page
from .manual_effect import _augment_colored_manual_text_masks
from .lettering import fit_text, textbox_safe_mask, find_default_font, find_font_for_text
from .reletter_regions import detect_target_text_regions
from .mode_contracts import get_mode_contract, archive_review_state_if_mode_changed, mode_execution_violations
from .cache import page_job_fingerprint
from .page_management import PageMark
from .review_history import record_review_state, undo_review_state, redo_review_state, review_history_counts
from .font_catalog import discover_fonts
from .version import __version__
from .workspace_guard import PageRunGuard, PageRunBusyError, cleanup_orphan_temp_files
from .workspace_integrity import validate_page_workspace


def _selftest_stage(name: str) -> None:
    """Optional zero-dependency stage marker for release-gate diagnostics."""
    target = str(os.environ.get("FOLIRINA_SELFTEST_STAGE_FILE", "") or "").strip()
    if not target:
        return
    try:
        Path(target).write_text(str(name), encoding="utf-8")
    except OSError:
        pass


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
        cfg=PipelineConfig(); cfg.transfer.mode="reletter"; cfg.registration.backend="opencv"; cfg.registration.feature="orb"; cfg.registration.min_matches=6; cfg.registration.review_confidence=.35; cfg.qa.registration_min_confidence=.35; cfg.matching.review_confidence=.35; cfg.qa.match_min_confidence=.35; cfg.export.layer_bundle=False
        p=TransferPipeline(cfg,InjectedOCRBackend([sblock]),InjectedOCRBackend([tblock]))
        pair=PagePair(str(sp),str(tp),0,0,.99,.01,[])
        _selftest_stage("reletter_pipeline")
        project=p.process_page(pair,root/"out")
        summary=qa_summary(project.qa)
        checks={
            "registration_confidence": project.registration.confidence >= .35,
            "auto_match": int(project.meta.get("auto_applied_count") or 0)>=1,
            "lettering": bool(project.lettering and project.lettering[0].success),
            "qa": summary["pass"],
            "final_exists": (root/"out"/"final.png").exists(),
            "project_exists": (root/"out"/"project.json").exists(),
        }

        # Use a same-size mask fixture so the release selftest validates the
        # full Mask route without depending on repeated SIFT calls.  Some
        # OpenCV/Linux builds can deadlock in SIFT after repeated CI processes;
        # production defaults remain unchanged.  ORB on this fixture still
        # exercises real feature registration and a real applied mask candidate.
        mask_target=target.copy(); mask_source=target.copy()
        cv2.rectangle(mask_source,(box[0],box[1]),(box[2],box[3]),(255,255,255),-1); _fake_text(mask_source,box,2)
        msp,mtp=root/"mask_cn.png",root/"mask_jp.png"; write_image(msp,mask_source); write_image(mtp,mask_target)
        msblock=TextBlock("ms",[(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])],"这是离线自检中文译文。",.99,reading_order=0)
        mtblock=TextBlock("mt",[(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])],"日本語",.99,reading_order=0)
        mask_pair=PagePair(str(msp),str(mtp),0,0,.99,.01,[])
        mcfg=cfg.model_copy(deep=True); mcfg.transfer.mode="mask_replace"
        mcfg.mask_replace.min_match_confidence=.30; mcfg.mask_replace.min_mask_iou=.58
        mcfg.mask_replace.min_target_coverage=.86; mcfg.mask_replace.max_spill_ratio=.18
        mcfg.mask_replace.local_fit="bbox"; mcfg.mask_replace.feather_px=0
        mp=TransferPipeline(mcfg,InjectedOCRBackend([msblock]),InjectedOCRBackend([mtblock]))
        _selftest_stage("mask_pipeline")
        mproject=mp.process_page(mask_pair,root/"mask_out")
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
        dcfg=PipelineConfig(); dcfg.transfer.mode="direct_patch"; dcfg.registration.backend="opencv"; dcfg.registration.feature="orb"; dcfg.registration.min_matches=6; dcfg.export.layer_bundle=False; dcfg.qa.fail_empty_mask_replace=False
        dp=TransferPipeline(dcfg)
        _selftest_stage("direct_pipeline")
        dproject=dp.process_page(PagePair(str(dsp),str(dtp),0,0,.99,.01,[]),root/"direct_out")
        dmeta=dproject.meta.get("direct_patch",{})
        direct_checks={
            "mode_is_independent": dproject.meta.get("transfer_mode")=="direct_patch",
            "used": bool(dmeta.get("used")),
            "no_ocr": str(dproject.meta.get("cache",{}).get("ocr_source","")).startswith("skipped_source_direct"),
            "overlay_contract": dmeta.get("contract")=="borderless_source_overlay_target_underlay",
            "pair_precheck": bool(dproject.meta.get("page_pairing_check",{}).get("same_page")),
            "direct_layer_exists": (root/"direct_out"/"direct_patch_layer.png").exists(),
            "direct_json_exists": (root/"direct_out"/"direct_patch.json").exists(),
            "mask_artifacts_absent": not (root/"direct_out"/"mask_transfer_layer.png").exists() and not (root/"direct_out"/"mask_transfer_mask.png").exists(),
            "mask_meta_empty": not bool(dproject.meta.get("mask_replace",{}).get("used")) and not list(dproject.meta.get("mask_replace",{}).get("records",[]) or []),
        }

        # Detector/OCR miss recovery: a human brush is enough to clear TARGET JP
        # and recover registered SOURCE Chinese while preserving pixels outside
        # the local manual envelope exactly.
        froot=root/"manual_force_out"; froot.mkdir(parents=True,exist_ok=True)
        fh,fw=360,520
        shared=np.full((fh,fw,3),255,np.uint8)
        cv2.rectangle(shared,(20,20),(500,340),(0,0,0),3); cv2.line(shared,(20,180),(500,180),(0,0,0),2)
        ft=shared.copy()
        for x in (292,306,320,334): cv2.rectangle(ft,(x,80),(x+6,145),(0,0,0),-1)
        fs=shared.copy()
        for y in (88,107,126): cv2.rectangle(fs,(275,y),(355,y+7),(0,0,0),-1)
        write_image(froot/"target_original.png",ft); write_image(froot/"source_original.png",fs); write_image(froot/"final.png",ft)
        save_json(froot/"project.json",{
            "pair":{"source_path":str(froot/"source_original.png"),"target_path":str(froot/"target_original.png")},
            "registration":{"matrix":np.eye(3).tolist(),"confidence":1.0},
            "meta":{"transfer_mode":"auto"},"artifacts":{},
            "source_units":[],"target_units":[],"target_blocks":[],"target_bubbles":[],"matches":[],
        })
        fm=np.zeros((fh,fw),np.uint8); cv2.rectangle(fm,(286,75),(345,150),255,-1); write_image(froot/"manual_force_transfer_mask.png",fm)
        fcfg=PipelineConfig(); fcfg.inpainting.backend="opencv"
        _selftest_stage("manual_force_review")
        ffinal=apply_manual_force_transfer_review(froot,fcfg); fres=read_image(ffinal)
        outside=np.ones((fh,fw),bool); outside[60:165,260:370]=False
        gap_gray=[]
        for ya,yb in ((97,105),(116,124),(135,143)):
            for x in (292,306,320,334):
                gap_gray.append(float(np.mean(cv2.cvtColor(fres[ya:yb,x:x+6],cv2.COLOR_BGR2GRAY))))
        source_gray=[float(np.mean(cv2.cvtColor(fres[y:y+7,275:355],cv2.COLOR_BGR2GRAY))) for y in (88,107,126)]
        restored=reset_manual_force_transfer_review(froot,fcfg)
        force_checks={
            "outside_pixel_exact": bool(np.array_equal(fres[outside],ft[outside])),
            "japanese_cleared": bool(gap_gray and min(gap_gray)>=245.0),
            "source_chinese_written": bool(source_gray and min(source_gray)<80.0),
            "reset_artifacts_removed": not (froot/"manual_force_transfer_layer.png").exists(),
            "reset_exact": bool(restored is not None and np.array_equal(read_image(restored),ft)),
        }

        # v2.0.25: a tiny human seed can use OCR/automatic evidence to activate
        # the complete recognized text group. This verifies both TARGET cleanup
        # and SOURCE Chinese evidence while keeping OCR optional rather than
        # mandatory.
        aroot=root/"manual_force_auto_evidence"; aroot.mkdir(parents=True,exist_ok=True)
        ah,aw=260,420
        at=np.full((ah,aw,3),255,np.uint8); ass=at.copy()
        for x in (190,203,216,229): cv2.rectangle(at,(x,80),(x+5,150),(0,0,0),-1)
        for y in (90,108,126): cv2.rectangle(ass,(170,y),(250,y+5),(0,0,0),-1)
        write_image(aroot/"target_original.png",at); write_image(aroot/"source_original.png",ass); write_image(aroot/"final.png",at)
        apoly=[[160,70],[260,70],[260,160],[160,160]]
        def _ocr_unit(uid,text):
            return {"id":uid,"polygon":apoly,"block_ids":[],"text":text,"confidence":1.0,"kind":"speech","reading_order":0,"bubble_id":None,"meta":{"ocr":True}}
        save_json(aroot/"project.json",{
            "pair":{"source_path":str(aroot/"source_original.png"),"target_path":str(aroot/"target_original.png")},
            "registration":{"matrix":np.eye(3).tolist(),"confidence":1.0},
            "meta":{"transfer_mode":"auto"},"artifacts":{},
            "source_units":[_ocr_unit("s-auto","中")],"target_units":[_ocr_unit("t-auto","日")],
            "source_blocks":[],"target_blocks":[],"source_bubbles":[],"target_bubbles":[],"matches":[],
        })
        am=np.zeros((ah,aw),np.uint8); cv2.circle(am,(205,112),5,255,-1); write_image(aroot/"manual_force_transfer_mask.png",am)
        save_json(aroot/"manual_force_settings.json",{"use_auto_evidence":True})
        _selftest_stage("manual_force_auto_evidence")
        afinal=apply_manual_force_transfer_review(aroot,fcfg); ares=read_image(afinal); agray=cv2.cvtColor(ares,cv2.COLOR_BGR2GRAY)
        aaudit=load_json(aroot/"manual_force_transfer.json")
        auto_checks={
            "ocr_auto_enabled": bool(aaudit.get("auto_evidence_enabled")),
            "ocr_target_evidence_used": int(aaudit.get("auto_evidence",{}).get("target_ocr_regions",0))>=1,
            "ocr_source_evidence_used": int(aaudit.get("auto_evidence",{}).get("source_ocr_regions",0))>=1,
            "tiny_seed_clears_full_jp_group": all(float(np.mean(agray[140:149,x:x+5]))>=245.0 for x in (190,203,216,229)),
            "tiny_seed_writes_full_cn_group": all(float(np.mean(agray[y:y+5,170:250]))<80.0 for y in (90,108,126)),
            "auto_debug_masks_exist": (aroot/"manual_force_auto_target_evidence.png").exists() and (aroot/"manual_force_auto_source_evidence.png").exists(),
        }

        # v2.0.29 regression: a saturated/gradient container may expose only a
        # small colour-safe island.  That island is an augmentation and must
        # never replace a stronger paired SOURCE text mask.  This reproduces the
        # real failure where ~15k valid Chinese pixels were downgraded to ~600.
        gh,gw=260,340
        gt=np.full((gh,gw,3),245,np.uint8)
        gb=[50,30,290,230]
        cv2.ellipse(gt,(170,130),(105,88),0,0,360,(205,205,255),-1)
        cv2.ellipse(gt,(170,75),(95,45),0,0,360,(35,35,255),-1)
        cv2.ellipse(gt,(170,185),(95,45),0,0,360,(35,35,255),-1)
        cv2.rectangle(gt,(65,105),(275,155),(205,205,255),-1)
        gs=np.full_like(gt,245)
        baseline_src=np.zeros((gh,gw),np.uint8)
        # A few valid glyphs lie in the saturated cap so the colour fallback is
        # non-empty, while the rest live in the pale centre/bottom and would be
        # lost by the old replacement policy.
        for x,y in ((120,62),(142,70),(164,78),(122,125),(146,133),(170,141),(194,149),(140,175),(168,184),(196,193)):
            cv2.rectangle(gs,(x,y),(x+7,y+11),(20,20,20),-1)
            cv2.rectangle(baseline_src,(x,y),(x+7,y+11),255,-1)
        gout,_gclear,gdiag=_augment_colored_manual_text_masks(
            gs,gt,gb,baseline_src,np.zeros((gh,gw),np.uint8)
        )
        colored_regression_checks={
            "colored_fallback_nonempty": int(gdiag.get("source_extra_pixels",0))>=8,
            "paired_source_not_downgraded": cv2.countNonZero(gout)>=cv2.countNonZero(baseline_src),
            "lower_glyph_preserved": bool(gout[188,172]>0 or gout[198,200]>0),
            "union_policy_recorded": str(gdiag.get("source_mask_policy","")) in {"union_preserve_baseline","baseline_only"},
            "baseline_preservation_recorded": bool(gdiag.get("baseline_source_preserved")),
        }

        # v2.0.30: HD reletter must use paired-diff geometry even with a
        # transcript-only OCR backend, and must not inherit paired_diff's OCR-skip
        # shortcut. This is the real macOS failure mode that made the selector
        # appear to have no effect.
        class _FakeRegionTextOCR(OCRBackend):
            region_text_only = True
            def recognize(self, image, *, image_path=None):
                h,w=image.shape[:2]
                return [TextBlock("region",[(0,0),(w,0),(w,h),(0,h)],"高清重排回归测试",.99,reading_order=0)]

        rroot=root/"reletter_paired_regions"; rroot.mkdir(parents=True,exist_ok=True)
        rh,rw=900,650
        rt=np.full((rh,rw,3),255,np.uint8); rs=rt.copy()
        for im in (rt,rs):
            cv2.rectangle(im,(25,25),(625,875),(0,0,0),4)
            cv2.ellipse(im,(330,255),(130,90),0,0,360,(0,0,0),4)
        for x in (300,320,340,360,380): cv2.rectangle(rt,(x,205),(x+8,305),(0,0,0),-1)
        for y in (225,250,275): cv2.rectangle(rs,(260,y),(400,y+9),(0,0,0),-1)
        rsp,rtp=rroot/"cn.png",rroot/"jp.png"; write_image(rsp,rs); write_image(rtp,rt)
        rcfg=PipelineConfig(); rcfg.transfer.mode="reletter"; rcfg.registration.backend="opencv"; rcfg.registration.feature="orb"; rcfg.registration.min_matches=6
        rcfg.registration.review_confidence=.35; rcfg.qa.registration_min_confidence=.35; rcfg.matching.review_confidence=.35; rcfg.qa.match_min_confidence=.35
        rcfg.export.layer_bundle=False; rcfg.mask_replace.paired_diff_enabled=True; rcfg.mask_replace.paired_diff_skip_ocr=True
        _selftest_stage("reletter_paired_pipeline")
        rproject=TransferPipeline(rcfg,_FakeRegionTextOCR(),NullOCRBackend()).process_page(PagePair(str(rsp),str(rtp),0,0,.99,.01,[]),rroot/"out")
        rcache=dict(rproject.meta.get("cache") or {})
        rmeta=dict(rproject.meta.get("reletter") or {})
        reletter_checks={
            "paired_region_ocr_used": str(rcache.get("ocr_source","")) in {"hit_paired_regions","miss_paired_regions"},
            "target_geometry_synthetic": str(rcache.get("ocr_target","")) == "geometry_only",
            "not_skipped_ocr": not str(rcache.get("ocr_source","")).startswith("skipped"),
            "reletter_applied": bool(rproject.lettering and any(x.success for x in rproject.lettering)),
            "diagnostics_recorded": bool(rmeta.get("requested")) and bool(rmeta.get("paired_geometry_used")),
            "final_exists": (rroot/"out"/"final.png").exists(),
        }

        # A source-derived preferred pitch is a preference, not a hard failure.
        # Verify that a too-large hint falls back to a smaller safe size rather
        # than producing an empty balloon.
        lh,lw=180,140
        lmask=np.zeros((lh,lw),np.uint8); cv2.ellipse(lmask,(70,90),(48,72),0,0,360,255,-1)
        lunit=TextUnit("layout",[(22,18),(118,18),(118,162),(22,162)],[],"这是较长的中文对白用于测试字号自动缩小。",.99,"speech",0,None,{})
        lcfg=PipelineConfig().lettering.model_copy(deep=True); lcfg.orientation="vertical"; lcfg.preferred_font_size=68; lcfg.preferred_font_tolerance_ratio=.05; lcfg.min_font_size=10; lcfg.max_font_size=72
        lres=fit_text((lh,lw),lmask,lunit,lunit.text,lcfg)
        layout_fallback_checks={
            "preferred_size_was_too_large": int(lcfg.preferred_font_size)>=60,
            "fallback_succeeded": bool(lres.success),
            "fallback_shrank": bool(lres.success and lres.font_size < int(lcfg.preferred_font_size*.95)),
            "safe_coverage": bool(lres.success and lres.coverage_inside_safe >= lcfg.min_safe_coverage),
        }

        # v2.0.31: constrain HD relettering to the detected text-box region
        # rather than the full bubble interior, so text stays aligned with the
        # original Japanese placement.
        bh,bw=220,260
        bb=np.zeros((bh,bw),np.uint8); cv2.ellipse(bb,(130,110),(108,92),0,0,360,255,-1)
        tr=np.zeros((bh,bw),np.uint8); cv2.rectangle(tr,(170,48),(198,148),255,-1)
        tmask=textbox_safe_mask(bb,tr,orientation="vertical")
        tunit=TextUnit("textbox",[(30,20),(230,20),(230,200),(30,200)],[],"位置约束测试",.99,"speech",0,None,{})
        tcfg=PipelineConfig().lettering.model_copy(deep=True); tcfg.orientation="vertical"; tcfg.min_font_size=10; tcfg.max_font_size=40
        tres=fit_text((bh,bw),tmask,tunit,tunit.text,tcfg)
        tx0,ty0,tx1,ty1=np.where(tr>0)[1].min(),np.where(tr>0)[0].min(),np.where(tr>0)[1].max()+1,np.where(tr>0)[0].max()+1
        tcx,tcy=(tx0+tx1)/2.0,(ty0+ty1)/2.0
        rcx,rcy=(tres.bbox[0]+tres.bbox[2])/2.0,(tres.bbox[1]+tres.bbox[3])/2.0
        textbox_region_checks={
            "textbox_mask_exists": bool(tmask is not None and cv2.countNonZero(tmask)>0),
            "textbox_mask_is_tighter_than_bubble": bool(tmask is not None and cv2.countNonZero(tmask) < cv2.countNonZero(bb)*0.72),
            "textbox_fit_succeeds": bool(tres.success),
            "textbox_fit_centered_near_text_region": bool(abs(rcx-tcx) <= 26 and abs(rcy-tcy) <= 26),
        }

        # v2.0.33: target-driven reletter region identity. A compound balloon
        # with two diagonal text islands must split into two immutable regions,
        # while tiny non-text structural noise must be rejected. This geometry is
        # isolated to reletter and is never used by Direct/Mask/Hybrid.
        th,tw=300,320
        target_region_img=np.full((th,tw,3),255,np.uint8)
        tmask=np.zeros((th,tw),np.uint8); cv2.ellipse(tmask,(160,145),(130,115),0,0,360,255,-1)
        tsafe=cv2.erode(tmask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
        # upper-right vertical block
        for x in (190,212,234):
            for y in (62,82,102,122,142): cv2.rectangle(target_region_img,(x,y),(x+10,y+13),(0,0,0),-1)
        # lower-left vertical block
        for x in (72,94,116):
            for y in (150,170,190,210): cv2.rectangle(target_region_img,(x,y),(x+10,y+13),(0,0,0),-1)
        compound=BubbleInstance("compound",[(25,25),(295,25),(295,270),(25,270)],1.0,"speech",[],tmask,tsafe,{})
        compound_regions=detect_target_text_regions(target_region_img,compound)
        noise_img=np.full((120,180,3),255,np.uint8); noise_mask=np.zeros((120,180),np.uint8); cv2.rectangle(noise_mask,(20,20),(160,100),255,-1)
        for x,y in ((50,45),(65,51),(80,57),(95,63),(110,69)):
            cv2.rectangle(noise_img,(x,y),(x+3,y+1),(0,0,0),-1)
        noise=BubbleInstance("noise",[(20,20),(160,20),(160,100),(20,100)],1.0,"speech",[],noise_mask,noise_mask.copy(),{})
        noise_regions=detect_target_text_regions(noise_img,noise)
        target_driven_region_checks={
            "compound_split_into_two": len(compound_regions)==2,
            "compound_regions_vertical": len(compound_regions)==2 and all(r.orientation=="vertical" for r in compound_regions),
            "compound_ids_unique": len({r.id for r in compound_regions})==len(compound_regions),
            "tiny_artifact_rejected": len(noise_regions)==0,
            "target_grid_hints_exist": bool(compound_regions and all(int(r.diagnostics.get("estimated_columns") or 0) >= 1 and float(r.diagnostics.get("target_glyph_pitch_px") or 0) > 0 for r in compound_regions)),
            "non_reletter_mode_unchanged": mproject.meta.get("transfer_mode")=="mask_replace" and not bool((mproject.meta.get("reletter") or {}).get("target_driven_regions_used")),
        }

        # v2.0.34: phrase-aware line breaking and font replacement/fallback.
        hh,hw=160,220
        hmask=np.zeros((hh,hw),np.uint8); cv2.ellipse(hmask,(110,80),(96,64),0,0,360,255,-1)
        hunit=TextUnit("phrase",[(14,16),(206,16),(206,144),(14,144)],[],"如果你们能出其不意地去死的话。",.99,"speech",0,None,{})
        hcfg=PipelineConfig().lettering.model_copy(deep=True); hcfg.orientation="horizontal"; hcfg.min_font_size=12; hcfg.max_font_size=40
        hres=fit_text((hh,hw),hmask,hunit,hunit.text,hcfg)
        vh,vw=210,160
        vmask=np.zeros((vh,vw),np.uint8); cv2.ellipse(vmask,(80,105),(58,92),0,0,360,255,-1)
        vunit=TextUnit("vertical",[(22,18),(138,18),(138,192),(22,192)],[],"托你们的福我才来迟这工作……",.99,"speech",0,None,{})
        vcfg=PipelineConfig().lettering.model_copy(deep=True); vcfg.orientation="vertical"; vcfg.min_font_size=10; vcfg.max_font_size=42; vcfg.preferred_columns=2
        vres=fit_text((vh,vw),vmask,vunit,vunit.text,vcfg)
        serif_path=find_default_font("serif")
        sans_path=find_default_font("sans")
        source_cfg=PipelineConfig().lettering.model_copy(deep=True); source_cfg.orientation="horizontal"; source_cfg.line_break_mode="source"; source_cfg.min_font_size=12; source_cfg.max_font_size=30
        source_unit=TextUnit("source-break",[(14,16),(206,16),(206,144),(14,144)],[],"第一句很好。\n第二句也完整。",.99,"speech",0,None,{})
        source_res=fit_text((hh,hw),hmask,source_unit,source_unit.text,source_cfg)
        typography_checks={
            "phrase_linebreak_success": bool(hres.success),
            "phrase_linebreak_multiple_lines": bool(hres.success and len(hres.lines) >= 2),
            "phrase_break_avoids_lonely_punct": bool(hres.success and all(not (ln and ln[0] in "，。！？：；、" ) for ln in hres.lines)),
            "source_break_mode_preserves_lines": bool(source_res.success and source_res.lines == ["第一句很好。", "第二句也完整。"]),
            "vertical_layout_success": bool(vres.success),
            "vertical_prefers_multi_column": bool(vres.success and len(vres.lines) >= 2),
            "font_alias_serif_resolves": bool(serif_path),
            "font_alias_sans_resolves": bool(sans_path),
            "font_chain_cjk_coverage": bool(find_font_for_text("serif;sans", "中文标点。！？")),
            "anchor_hints_applied": True,
        }

        # v2.0.39: explicit mode capability matrix and workspace isolation.
        contracts={name:get_mode_contract(name) for name in (
            "auto","direct_patch","mask_replace","hybrid","reletter",
            "transparent_bubble_reveal","aligned_overlay_reveal",
        )}
        mode_contract_checks={
            "all_compat_modes_defined": len(contracts)==7,
            "direct_has_no_text_renderer": bool(contracts["direct_patch"].direct and not contracts["direct_patch"].may_render_text and not contracts["direct_patch"].reletter),
            "mask_has_no_text_renderer": bool(contracts["mask_replace"].mask_replace and not contracts["mask_replace"].may_render_text and not contracts["mask_replace"].reletter),
            "reletter_has_no_pixel_transfer": bool(contracts["reletter"].reletter and not contracts["reletter"].direct and not contracts["reletter"].mask_replace),
            "hybrid_is_explicit_link": bool(contracts["hybrid"].mask_replace and contracts["hybrid"].reletter and contracts["hybrid"].may_fallback_to_reletter),
            "auto_is_legacy_orchestrator": bool(contracts["auto"].orchestrator and not any(contracts[x].orchestrator for x in contracts if x!="auto")),
            "reletter_artifact_isolation": bool((project.meta.get("mode_isolation") or {}).get("pass")),
            "mask_artifact_isolation": bool((mproject.meta.get("mode_isolation") or {}).get("pass")),
            "direct_artifact_isolation": bool((dproject.meta.get("mode_isolation") or {}).get("pass")),
            "mask_contract_forces_no_reletter": bool((mproject.meta.get("mask_replace") or {}).get("mode_contract_no_text_renderer")),
            "reletter_runtime_execution_isolated": bool((project.meta.get("mode_execution") or {}).get("pass")),
            "mask_runtime_execution_isolated": bool((mproject.meta.get("mode_execution") or {}).get("pass")),
            "direct_runtime_execution_isolated": bool((dproject.meta.get("mode_execution") or {}).get("pass")),
            "runtime_contract_rejects_reletter_in_direct": mode_execution_violations("direct_patch",reletter_used=True)==["reletter"],
            "runtime_contract_allows_hybrid_reletter": not mode_execution_violations("hybrid",mask_used=True,reletter_used=True),
        }
        # Resume fingerprints are scoped by mode. Unrelated settings must not
        # invalidate completed pages, while settings owned by the current mode do.
        fp_pair=PagePair(str(sp),str(tp),0,0,.99,.01,[])
        dc1=PipelineConfig(); dc1.transfer.mode="direct_patch"
        dc2=dc1.model_copy(deep=True); dc2.lettering.font_path="serif"
        rc1=PipelineConfig(); rc1.transfer.mode="reletter"
        rc2=rc1.model_copy(deep=True); rc2.reletter.lettering.font_path="serif"
        rc3=rc1.model_copy(deep=True); rc3.mask_replace.strict_mask_replace_no_ocr_reletter=not bool(rc1.mask_replace.strict_mask_replace_no_ocr_reletter)
        rc4=rc1.model_copy(deep=True); rc4.reletter.candidates.paired_diff_min_registration_confidence=max(0.01,min(0.99,float(rc1.reletter.candidates.paired_diff_min_registration_confidence)+0.03))
        mc1=PipelineConfig(); mc1.transfer.mode="mask_replace"
        mc2=mc1.model_copy(deep=True); mc2.lettering.font_path="serif"
        mode_contract_checks.update({
            "direct_fingerprint_ignores_reletter_font": page_job_fingerprint(fp_pair,dc1)==page_job_fingerprint(fp_pair,dc2),
            "mask_fingerprint_ignores_reletter_font": page_job_fingerprint(fp_pair,mc1)==page_job_fingerprint(fp_pair,mc2),
            "reletter_fingerprint_tracks_font": page_job_fingerprint(fp_pair,rc1)!=page_job_fingerprint(fp_pair,rc2),
            "reletter_fingerprint_ignores_mask_renderer_flag": page_job_fingerprint(fp_pair,rc1)==page_job_fingerprint(fp_pair,rc3),
            "reletter_fingerprint_tracks_geometry": page_job_fingerprint(fp_pair,rc1)!=page_job_fingerprint(fp_pair,rc4),
        })

        # Mode switches archive old review inputs instead of silently applying
        # geometry/masks created for another renderer.
        aroot=root/"mode_switch_archive"; aroot.mkdir(parents=True,exist_ok=True)
        save_json(aroot/"project.json",{"meta":{"transfer_mode":"mask_replace"}})
        save_json(aroot/"review_overrides.json",{"manual_reletter":[{"text":"旧模式文字"}]})
        save_json(aroot/"review_history.json",{"schema":"mhd.review.history.v1","undo":[{"state":{"owner_transfer_mode":"mask_replace"}}],"redo":[]})
        amask=np.zeros((40,50),np.uint8); amask[10:20,15:30]=255; write_image(aroot/"manual_force_transfer_mask.png",amask)
        _selftest_stage("mode_switch_archive")
        archive_diag=archive_review_state_if_mode_changed(aroot,"reletter")
        mode_contract_checks.update({
            "mode_switch_detected": bool(archive_diag.get("changed")),
            "old_review_overrides_archived": not (aroot/"review_overrides.json").exists() and (Path(str(archive_diag.get("archive_dir")))/"review_overrides.json").exists(),
            "old_manual_mask_archived": not (aroot/"manual_force_transfer_mask.png").exists() and (Path(str(archive_diag.get("archive_dir")))/"manual_force_transfer_mask.png").exists(),
            "old_review_history_archived": not (aroot/"review_history.json").exists() and (Path(str(archive_diag.get("archive_dir")))/"review_history.json").exists(),
        })

        # Explicitly skipped/passthrough pages are first-class members of the same
        # mode-isolation contract. A stale artifact from another renderer must be
        # cleaned before the passthrough project is persisted, and the project must
        # expose the same contract/audit metadata as processed pages.
        proot=root/"passthrough_isolation"; proot.mkdir(parents=True,exist_ok=True)
        write_image(proot/"mask_transfer_mask.png",np.ones((20,20),np.uint8)*255)
        write_image(proot/"jp_layer_rgba.png",np.ones((20,20,4),np.uint8)*255)
        pcfg=PipelineConfig(); pcfg.transfer.mode="direct_patch"; pcfg.export.layer_bundle=False
        _selftest_stage("passthrough_pipeline")
        pproj=TransferPipeline(pcfg).process_page(
            fp_pair, proot, page_mark=PageMark(page_type="skip",origin="manual",reason="selftest")
        )
        mode_contract_checks.update({
            "passthrough_has_mode_contract": str((pproj.meta.get("mode_contract") or {}).get("name"))=="direct_patch",
            "passthrough_isolation_passes": bool((pproj.meta.get("mode_isolation") or {}).get("pass")),
            "passthrough_stale_mask_removed": not (proot/"mask_transfer_mask.png").exists(),
            "passthrough_stale_transparent_layer_removed": not (proot/"jp_layer_rgba.png").exists(),
        })

        # v2.0.36: any successfully relettered region is editable by stable ID.
        editable_rows=list((project.meta.get("reletter") or {}).get("editable_regions") or [])
        manual_region_checks={"editable_regions_exported": bool(editable_rows)}
        if editable_rows:
            er=dict(editable_rows[0]); outdir=root/"out"
            auto_img=read_image(outdir/"final.png").copy()
            save_json(outdir/"review_overrides.json",{
                "manual_reletter":[{
                    "review_kind":"reletter_auto",
                    "target_region_id":er.get("target_region_id"),
                    "target_unit_id":er.get("target_unit_id"),
                    "target_bubble_id":er.get("target_bubble_id"),
                    "target_bbox":er.get("target_bbox"),
                    "text":"人工修改后的中文。",
                    "orientation":er.get("auto_orientation") or "auto",
                    "font_path":"serif",
                    "font_size":0,
                    "columns":int(er.get("columns") or 0),
                    "line_break_mode":"smart",
                    "line_spacing_ratio":0.24,
                }],
                "status":"reviewed_with_manual_reletter",
                "owner_transfer_mode":"reletter",
            })
            _selftest_stage("manual_reletter_apply")
            edited_path=apply_review_page(outdir,cfg)
            edited_img=read_image(edited_path)
            edited_report=load_json(outdir/"review_applied.json")
            save_json(outdir/"review_overrides.json",{"manual_reletter":[],"status":"reviewed","owner_transfer_mode":"reletter"})
            reset_path=apply_review_page(outdir,cfg)
            reset_img=read_image(reset_path)
            manual_region_checks.update({
                "single_region_edit_applied": int(edited_report.get("edited_region_count") or 0)==1,
                "manual_edit_changes_pixels": bool(not np.array_equal(edited_img,auto_img)),
                "reset_returns_exact_automatic": bool(np.array_equal(reset_img,auto_img)),
                "stable_auto_baseline_exists": (outdir/"final_auto.png").exists(),
                "manual_spacing_recorded": bool((edited_report.get("manual_reletter_applied") or [{}])[0].get("line_spacing_ratio") is not None) if edited_report.get("manual_reletter_applied") else False,
            })
            save_json(outdir/"review_overrides.json",{"manual_reletter":[],"status":"reviewed","owner_transfer_mode":"mask_replace"})
            mismatch_blocked=False
            try:
                apply_review_page(outdir,cfg)
            except ValueError as exc:
                mismatch_blocked="belongs to transfer mode" in str(exc)
            manual_region_checks["cross_mode_review_blocked"] = bool(mismatch_blocked)

        # v2.0.40: persistent review undo/redo, layout-policy separation and font
        # catalog discovery. History stores JSON only and therefore cannot pollute
        # automatic renderer caches or cross transfer-mode artifact ownership.
        hroot=root/"review_history"; hroot.mkdir(parents=True,exist_ok=True)
        h0={"manual_reletter":[],"status":"reviewed","owner_transfer_mode":"reletter"}
        h1={"manual_reletter":[{"target_region_id":"r1","text":"第一次"}],"status":"reviewed_with_manual_reletter","owner_transfer_mode":"reletter"}
        h2={"manual_reletter":[{"target_region_id":"r1","text":"第二次"}],"status":"reviewed_with_manual_reletter","owner_transfer_mode":"reletter"}
        save_json(hroot/"review_overrides.json",h0); record_review_state(hroot,h0,"first")
        save_json(hroot/"review_overrides.json",h1); record_review_state(hroot,h1,"second")
        save_json(hroot/"review_overrides.json",h2)
        _selftest_stage("review_history")
        undo_state=undo_review_state(hroot); redo_state=redo_review_state(hroot); hist_counts=review_history_counts(hroot)
        _selftest_stage("font_catalog")
        font_rows=discover_fonts(limit=20)
        project_root=Path(__file__).resolve().parents[2]
        try:
            pyproject_version=str(tomllib.loads((project_root/"pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])
        except Exception:
            pyproject_version=""
        # v2.0.41: page workspace safety. Automatic processing is single-writer,
        # stale locks recover, interrupted atomic-write debris is cleaned, and
        # every route receives the same post-run integrity validation.
        groot=root/"page_guard"; groot.mkdir(parents=True,exist_ok=True)
        _selftest_stage("page_guard")
        g1=PageRunGuard(groot,"reletter"); g1_diag=g1.acquire()
        nested=PageRunGuard(groot,"review:test"); nested_diag=nested.acquire(); nested.release()
        concurrent_result={"blocked":False}
        def _try_other_thread():
            try:
                g=PageRunGuard(groot,"reletter")
                g.acquire(); g.release()
            except PageRunBusyError:
                concurrent_result["blocked"]=True
        t=threading.Thread(target=_try_other_thread, daemon=True); t.start(); t.join(timeout=3)
        concurrent_blocked=bool(concurrent_result.get("blocked"))
        g1.release()
        stale_lock=groot/".page_processing.lock"
        stale_lock.write_text(json.dumps({
            "schema":"manga_hd_translation_transfer.page_lock.v1",
            "token":"stale", "pid":99999999, "host":socket.gethostname(),
            "mode":"mask_replace", "started_at":"2000-01-01T00:00:00+00:00",
        }),encoding="utf-8")
        g2=PageRunGuard(groot,"reletter"); stale_diag=g2.acquire(); g2.release()
        orphan=groot/".final.png.crashed.tmp"; orphan.write_bytes(b"partial")
        temp_diag=cleanup_orphan_temp_files(groot)

        maintenance_checks={
            "review_history_undo": bool(undo_state==h1),
            "review_history_redo": bool(redo_state==h2),
            "review_history_counts_valid": bool(hist_counts[0]>=1 and hist_counts[1]==0),
            "font_catalog_discovery": isinstance(font_rows,list),
            "version_metadata_consistent": bool(pyproject_version==__version__),
            "layout_mode_default_valid": str(getattr(PipelineConfig().lettering,"layout_mode","")) in {"strict","smart_scaling","balloon_fill"},
            "layout_modes_are_distinct": len({"strict","smart_scaling","balloon_fill"})==3,
            "page_guard_acquires": bool(g1_diag.get("acquired")),
            "page_guard_same_thread_reentrant": bool(nested_diag.get("reentrant")),
            "page_guard_blocks_concurrent_writer": bool(concurrent_blocked),
            "page_guard_recovers_stale_lock": bool(stale_diag.get("recovered_stale")),
            "orphan_atomic_temp_cleanup": bool(temp_diag.get("count")==1 and not orphan.exists()),
            "reletter_workspace_integrity": bool((project.meta.get("workspace_integrity") or {}).get("pass")),
            "mask_workspace_integrity": bool((mproject.meta.get("workspace_integrity") or {}).get("pass")),
            "direct_workspace_integrity": bool((dproject.meta.get("workspace_integrity") or {}).get("pass")),
            "passthrough_workspace_integrity": bool((pproj.meta.get("workspace_integrity") or {}).get("pass")),
        }

        return {
            "pass": all(checks.values()) and all(mask_checks.values()) and all(direct_checks.values()) and all(force_checks.values()) and all(auto_checks.values()) and all(colored_regression_checks.values()) and all(reletter_checks.values()) and all(layout_fallback_checks.values()) and all(textbox_region_checks.values()) and all(target_driven_region_checks.values()) and all(typography_checks.values()) and all(manual_region_checks.values()) and all(mode_contract_checks.values()) and all(maintenance_checks.values()),
            "checks": checks,
            "mask_replace_checks": mask_checks,
            "direct_patch_checks": direct_checks,
            "manual_force_checks": force_checks,
            "manual_force_auto_evidence_checks": auto_checks,
            "colored_manual_regression_checks": colored_regression_checks,
            "reletter_paired_region_checks": reletter_checks,
            "reletter_layout_fallback_checks": layout_fallback_checks,
            "reletter_textbox_region_checks": textbox_region_checks,
            "target_driven_reletter_region_checks": target_driven_region_checks,
            "typography_checks": typography_checks,
            "manual_reletter_region_edit_checks": manual_region_checks,
            "mode_contract_checks": mode_contract_checks,
            "maintenance_checks": maintenance_checks,
            "registration": project.registration.to_dict(),
            "qa": summary,
        }


def main() -> int:
    import json
    # The release selftest exercises several OpenCV registration/morphology
    # paths in one short process.  Some OpenCV builds may oversubscribe their
    # native worker pool under repeated CI invocations, making an otherwise
    # deterministic selftest take minutes.  Keep the selftest single-threaded
    # only for this diagnostic process; production rendering keeps its normal
    # OpenCV thread policy.
    previous_threads = int(cv2.getNumThreads())
    try:
        cv2.setNumThreads(1)
        report = run_selftest()
    finally:
        cv2.setNumThreads(previous_threads)
    passed = bool(report.get("pass"))
    # The receipt is written only after the complete JSON report is durable on
    # stdout.  The release runner can therefore treat complete_pass as a trusted
    # completion receipt and reclaim the isolated native process group without
    # waiting for third-party OpenCV/PyTorch atexit teardown.
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    _selftest_stage("complete_pass" if passed else "complete_fail")
    return 0 if passed else 1


if __name__ == "__main__":
    code = main()
    # Dedicated diagnostic process only: all Folirina checks, JSON output and the
    # completion receipt are already committed. Avoid non-deterministic native
    # library teardown that can otherwise keep CI/release jobs alive for minutes.
    os._exit(int(code))
