from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.bubbles import detect_seeded_white_bubbles
from manga_hd_transfer.config import BubbleConfig, MaskReplaceConfig, PipelineConfig
from manga_hd_transfer.mask_transfer import transfer_bubble_patches
from manga_hd_transfer.models import RegistrationResult, TextBlock


def _block(box, text):
    x0,y0,x1,y1=box
    return TextBlock("b",[(x0,y0),(x1,y0),(x1,y1),(x0,y1)],text,.99,reading_order=0)


def test_mask_replace_preserves_hd_border_and_transfers_cn_patch():
    # HD target: clean bubble + Japanese-like vertical dark bars.
    target=np.full((600,800,3),255,np.uint8)
    cv2.ellipse(target,(400,300),(180,110),0,0,360,(0,0,0),5)
    for x in (370,400,430):
        cv2.rectangle(target,(x,250),(x+9,350),(0,0,0),-1)
    tbox=(350,240,450,360)

    # Old Chinese edition: lower resolution and a visibly different two-line pattern.
    source=cv2.resize(target,(400,300),interpolation=cv2.INTER_AREA)
    sx,sy=.5,.5
    sbox=(int(tbox[0]*sx),int(tbox[1]*sy),int(tbox[2]*sx),int(tbox[3]*sy))
    cv2.rectangle(source,(sbox[0]-4,sbox[1]-4),(sbox[2]+4,sbox[3]+4),(255,255,255),-1)
    cv2.rectangle(source,(sbox[0]+10,sbox[1]+18),(sbox[2]-10,sbox[1]+28),(0,0,0),-1)
    cv2.rectangle(source,(sbox[0]+20,sbox[1]+42),(sbox[2]-20,sbox[1]+52),(0,0,0),-1)

    sb=_block(sbox,"旧版中文")
    tb=_block(tbox,"日本語")
    bcfg=BubbleConfig(white_threshold=200,safe_margin_px=8)
    source_bubbles=detect_seeded_white_bubbles(source,[sb],bcfg)
    target_bubbles=detect_seeded_white_bubbles(target,[tb],bcfg)
    assert len(source_bubbles)==1 and len(target_bubbles)==1

    reg=RegistrationResult(
        matrix=np.array([[2.,0.,0.],[0.,2.,0.],[0.,0.,1.]],np.float64),
        method="known-scale",confidence=.99,inlier_ratio=1.,reprojection_error=0.,spatial_coverage=.9,num_matches=30,
        source_size=(400,300),target_size=(800,600),diagnostics={},
    )
    cfg=MaskReplaceConfig(
        min_match_confidence=.35,min_mask_iou=.65,min_target_coverage=.90,max_spill_ratio=.12,
        local_fit="bbox",sr_backend="lanczos",sr_min_trigger=1.05,preserve_target_border=True,border_inset_px=3,feather_px=0,
    )
    result=transfer_bubble_patches(source,target,source_bubbles,target_bubbles,reg,cfg)
    assert result.applied_count==1, [r.to_dict() for r in result.records]
    rec=result.records[0]
    assert rec.sr_scale>1.5
    assert rec.target_coverage>=cfg.min_target_coverage
    # Border is HD target border: replacement is inset, so this point must remain unchanged.
    assert np.array_equal(result.image[300,220],target[300,220])
    # Japanese vertical bars are replaced by the source two-line pattern.
    before=np.mean(cv2.cvtColor(target[250:350,350:450],cv2.COLOR_BGR2GRAY)<80)
    after=np.mean(cv2.cvtColor(result.image[250:350,350:450],cv2.COLOR_BGR2GRAY)<80)
    assert abs(after-before)>0.01
    assert result.layer_rgba[...,3].max()==255


def test_pipeline_mask_replace_mode(tmp_path):
    from manga_hd_transfer.io_utils import write_image
    from manga_hd_transfer.models import PagePair
    from manga_hd_transfer.ocr import InjectedOCRBackend
    from manga_hd_transfer.pipeline import TransferPipeline
    from .helpers import make_art_page

    target=make_art_page(1000,1400)
    tbox=(235,660,365,740)
    # Japanese-like vertical pattern.
    for x in (270,300,330):
        cv2.rectangle(target,(x,670),(x+8,730),(0,0,0),-1)
    source=cv2.resize(target,(700,980),interpolation=cv2.INTER_AREA)
    sx=.7; sy=.7
    sbox=(tbox[0]*sx,tbox[1]*sy,tbox[2]*sx,tbox[3]*sy)
    cv2.rectangle(source,(int(sbox[0]-6),int(sbox[1]-5)),(int(sbox[2]+6),int(sbox[3]+5)),(255,255,255),-1)
    cv2.rectangle(source,(int(sbox[0]+10),int(sbox[1]+16)),(int(sbox[2]-10),int(sbox[1]+24)),(0,0,0),-1)
    cv2.rectangle(source,(int(sbox[0]+18),int(sbox[1]+38)),(int(sbox[2]-18),int(sbox[1]+46)),(0,0,0),-1)
    sp=tmp_path/'cn_mask.png'; tp=tmp_path/'jp_mask.png'; write_image(sp,source); write_image(tp,target)
    sblk=TextBlock('s',[(sbox[0],sbox[1]),(sbox[2],sbox[1]),(sbox[2],sbox[3]),(sbox[0],sbox[3])],'中文气泡',.99,reading_order=0)
    tblk=TextBlock('t',[(tbox[0],tbox[1]),(tbox[2],tbox[1]),(tbox[2],tbox[3]),(tbox[0],tbox[3])],'日本語',.99,reading_order=0)
    cfg=PipelineConfig(); cfg.transfer.mode='mask_replace'; cfg.registration.backend='opencv'; cfg.registration.feature='sift'; cfg.registration.min_matches=6; cfg.qa.registration_min_confidence=.30
    cfg.mask_replace.min_match_confidence=.30; cfg.mask_replace.min_mask_iou=.60; cfg.mask_replace.min_target_coverage=.88; cfg.mask_replace.max_spill_ratio=.15; cfg.mask_replace.local_fit='bbox'; cfg.mask_replace.feather_px=0
    cfg.mask_replace.source_direct_container_enabled=False  # isolate legacy matched-mask route
    cfg.mask_replace.paired_diff_enabled=False  # isolate OCR-seeded legacy transfer
    cfg.export.layer_bundle=False
    pipe=TransferPipeline(cfg,InjectedOCRBackend([sblk]),InjectedOCRBackend([tblk]))
    pair=PagePair(str(sp),str(tp),0,0,.99,.01,[])
    project=pipe.process_page(pair,tmp_path/'mask_out')
    assert project.meta['transfer_mode']=='mask_replace'
    assert project.meta['mask_replace']['applied_count']==1
    assert (tmp_path/'mask_out'/'mask_transfer_layer.png').exists()
    assert (tmp_path/'mask_out'/'mask_transfer_mask.png').exists()
    assert (tmp_path/'mask_out'/'final.png').exists()


def test_mask_replace_review_can_shrink_transfer_mask(tmp_path):
    from manga_hd_transfer.export import write_rgba
    from manga_hd_transfer.io_utils import save_json, write_image
    from manga_hd_transfer.review_apply import apply_review_page

    target=np.full((120,160,3),255,np.uint8)
    layer=np.zeros((120,160,4),np.uint8)
    layer[30:90,40:120,:3]=0
    layer[30:90,40:120,3]=255
    write_image(tmp_path/'target_original.png',target)
    write_rgba(tmp_path/'mask_transfer_layer.png',layer)
    save_json(tmp_path/'project.json',{'meta':{'transfer_mode':'mask_replace'}})
    manual=np.zeros((120,160),np.uint8); manual[30:90,40:80]=255
    write_image(tmp_path/'manual_transfer_mask.png',manual)
    out=apply_review_page(tmp_path,PipelineConfig())
    img=cv2.imread(str(out))
    assert np.all(img[60,60]==0)      # retained left half
    assert np.all(img[60,100]==255)   # erased right half restored to HD target


def test_blurry_source_uses_deterministic_ink_reconstruction():
    target=np.full((360,480,3),255,np.uint8)
    cv2.rectangle(target,(120,80),(360,280),(0,0,0),3)
    for x in (210,235,260):
        cv2.rectangle(target,(x,125),(x+5,235),(0,0,0),-1)
    tbox=(145,105,335,255)

    # Simulate a photographed Chinese patch: low contrast + optical blur.
    source=cv2.resize(target,(240,180),interpolation=cv2.INTER_AREA)
    sbox=tuple(int(v*.5) for v in tbox)
    cv2.rectangle(source,(sbox[0],sbox[1]),(sbox[2],sbox[3]),(245,245,245),-1)
    cv2.putText(source,'CN',(sbox[0]+12,sbox[1]+48),cv2.FONT_HERSHEY_SIMPLEX,.7,(35,35,35),2,cv2.LINE_AA)
    source=cv2.GaussianBlur(source,(9,9),2.2)
    # Smooth illumination gradient similar to phone glare/exposure.
    grad=np.linspace(0,18,source.shape[1],dtype=np.float32)[None,:,None]
    source=np.clip(source.astype(np.float32)+grad,0,255).astype(np.uint8)

    sb=_block(sbox,'中文')
    tb=_block(tbox,'日本語')
    bcfg=BubbleConfig(white_threshold=190,safe_margin_px=6)
    source_bubbles=detect_seeded_white_bubbles(source,[sb],bcfg)
    target_bubbles=detect_seeded_white_bubbles(target,[tb],bcfg)
    assert source_bubbles and target_bubbles
    reg=RegistrationResult(
        matrix=np.array([[2.,0.,0.],[0.,2.,0.],[0.,0.,1.]],np.float64),
        method='scale',confidence=.99,inlier_ratio=1.,reprojection_error=0.,spatial_coverage=.9,num_matches=20,
        source_size=(240,180),target_size=(480,360),diagnostics={},
    )
    cfg=MaskReplaceConfig(
        min_match_confidence=.25,min_mask_iou=.50,min_target_coverage=.80,max_spill_ratio=.20,
        local_fit='bbox',preserve_target_border=True,border_inset_px=3,feather_px=0,
        sr_backend='lanczos',sr_min_trigger=1.0,text_fidelity_mode='auto',
        min_pixel_text_sharpness=500.0,min_relative_text_sharpness=.95,
        ink_reconstruction_enabled=True,reject_blurry_source=True,
    )
    result=transfer_bubble_patches(source,target,source_bubbles,target_bubbles,reg,cfg)
    assert result.applied_count==1, [r.to_dict() for r in result.records]
    rec=result.records[0]
    assert rec.clarity_mode=='ink-reconstruction'
    assert rec.ink_ratio>0
    # Reconstructed glyphs are exact black/white, not a soft grey pasted photo.
    roi=result.image[110:250,150:330]
    assert np.any(np.all(roi==0,axis=2))


def test_blurry_source_is_rejected_when_ink_reconstruction_is_unsafe():
    target=np.full((280,420,3),(120,170,220),np.uint8)  # coloured/non-white target
    cv2.rectangle(target,(100,70),(320,220),(0,0,0),3)
    source=cv2.resize(target,(210,140),interpolation=cv2.INTER_AREA)
    source=cv2.GaussianBlur(source,(11,11),3.0)
    sbox=(60,45,150,110); tbox=(120,90,300,220)
    sb=_block(sbox,'中文'); tb=_block(tbox,'日本語')
    # Use explicit synthetic bubble instances to isolate the fidelity guard.
    from manga_hd_transfer.models import BubbleInstance
    sm=np.zeros(source.shape[:2],np.uint8); cv2.rectangle(sm,(60,45),(150,110),255,-1)
    tm=np.zeros(target.shape[:2],np.uint8); cv2.rectangle(tm,(120,90),(300,220),255,-1)
    sbub=BubbleInstance('s',[(60,45),(150,45),(150,110),(60,110)],.99,'narration',['b'],sm,sm.copy(),{})
    tbub=BubbleInstance('t',[(120,90),(300,90),(300,220),(120,220)],.99,'narration',['b'],tm,tm.copy(),{})
    reg=RegistrationResult(np.array([[2.,0.,0.],[0.,2.,0.],[0.,0.,1.]],np.float64),'scale',.99,1.,0.,.9,20,(210,140),(420,280),{})
    cfg=MaskReplaceConfig(
        min_match_confidence=.1,min_mask_iou=.4,min_target_coverage=.7,max_spill_ratio=.3,
        local_fit='bbox',preserve_target_border=False,feather_px=0,
        min_pixel_text_sharpness=1e6,min_relative_text_sharpness=1.0,
        text_fidelity_mode='auto',ink_reconstruction_enabled=True,reject_blurry_source=True,
        ink_target_white_ratio=.9,
    )
    result=transfer_bubble_patches(source,target,[sbub],[tbub],reg,cfg)
    assert result.applied_count==0
    assert result.records and result.records[0].reason=='source_text_too_blurry_for_pixel_transfer'


def test_photo_pair_small_bubble_uses_ink_fallback_instead_of_hard_reject():
    from manga_hd_transfer.models import BubbleInstance

    target=np.full((320,420,3),255,np.uint8)
    tbox=(260,70,335,145)  # small target bubble (< photo_pair_min_direct_side_px)
    tm=np.zeros(target.shape[:2],np.uint8)
    cv2.rectangle(tm,(tbox[0],tbox[1]),(tbox[2],tbox[3]),255,-1)
    cv2.rectangle(target,(tbox[0],tbox[1]),(tbox[2],tbox[3]),(0,0,0),2)
    for x in (286,303):
        cv2.rectangle(target,(x,88),(x+5,128),(0,0,0),-1)

    source=cv2.resize(target,(210,160),interpolation=cv2.INTER_AREA)
    sbox=(130,35,167,72)
    sm=np.zeros(source.shape[:2],np.uint8)
    cv2.rectangle(sm,(sbox[0],sbox[1]),(sbox[2],sbox[3]),255,-1)
    cv2.rectangle(source,(sbox[0],sbox[1]),(sbox[2],sbox[3]),(245,245,245),-1)
    cv2.putText(source,'CN',(sbox[0]+2,sbox[1]+23),cv2.FONT_HERSHEY_SIMPLEX,.38,(50,50,50),1,cv2.LINE_AA)
    source=cv2.GaussianBlur(source,(7,7),1.6)

    sbub=BubbleInstance('photo-src-000',[(sbox[0],sbox[1]),(sbox[2],sbox[1]),(sbox[2],sbox[3]),(sbox[0],sbox[3])],.99,'speech',['b'],sm,sm.copy(),{'paired_target_id':'photo-dst-000'})
    tbub=BubbleInstance('photo-dst-000',[(tbox[0],tbox[1]),(tbox[2],tbox[1]),(tbox[2],tbox[3]),(tbox[0],tbox[3])],.99,'speech',['b'],tm,tm.copy(),{'paired_diff_method':'photo_pair'})
    reg=RegistrationResult(np.array([[2.,0.,0.],[0.,2.,0.],[0.,0.,1.]],np.float64),'scale',.99,1.,0.,.9,20,(210,160),(420,320),{})
    cfg=MaskReplaceConfig(
        min_match_confidence=.1,min_mask_iou=.4,min_target_coverage=.7,max_spill_ratio=.3,
        photo_pair_min_transfer_iou=.4,photo_pair_min_transfer_coverage=.7,photo_pair_max_spill_ratio=.3,
        preserve_target_border=False,feather_px=0,local_fit='bbox',
        min_pixel_text_sharpness=1.0,min_relative_text_sharpness=0.05,
        text_fidelity_mode='auto',ink_reconstruction_enabled=True,reject_blurry_source=True,
    )
    result=transfer_bubble_patches(source,target,[sbub],[tbub],reg,cfg)
    assert result.applied_count==1, [r.to_dict() for r in result.records]
    assert result.records[0].clarity_mode in {'photo-crisp-ink','ink-reconstruction'}


def test_photo_pair_prefers_ink_reconstruction_when_normalized_pixels_remain_soft():
    from manga_hd_transfer.models import BubbleInstance

    target=np.full((360,480,3),255,np.uint8)
    tbox=(150,95,330,250)
    tm=np.zeros(target.shape[:2],np.uint8)
    cv2.rectangle(tm,(tbox[0],tbox[1]),(tbox[2],tbox[3]),255,-1)
    cv2.rectangle(target,(tbox[0],tbox[1]),(tbox[2],tbox[3]),(0,0,0),3)
    for x in (210,235,260):
        cv2.rectangle(target,(x,130),(x+5,225),(0,0,0),-1)

    source=cv2.resize(target,(240,180),interpolation=cv2.INTER_AREA)
    sbox=(75,48,165,126)
    sm=np.zeros(source.shape[:2],np.uint8)
    cv2.rectangle(sm,(sbox[0],sbox[1]),(sbox[2],sbox[3]),255,-1)
    cv2.rectangle(source,(sbox[0],sbox[1]),(sbox[2],sbox[3]),(245,245,245),-1)
    cv2.putText(source,'CN',(sbox[0]+8,sbox[1]+30),cv2.FONT_HERSHEY_SIMPLEX,.60,(45,45,45),2,cv2.LINE_AA)
    source=cv2.GaussianBlur(source,(9,9),2.0)
    grad=np.linspace(0,20,source.shape[1],dtype=np.float32)[None,:,None]
    source=np.clip(source.astype(np.float32)+grad,0,255).astype(np.uint8)

    sbub=BubbleInstance('photo-src-001',[(sbox[0],sbox[1]),(sbox[2],sbox[1]),(sbox[2],sbox[3]),(sbox[0],sbox[3])],.99,'speech',['b'],sm,sm.copy(),{'paired_target_id':'photo-dst-001'})
    tbub=BubbleInstance('photo-dst-001',[(tbox[0],tbox[1]),(tbox[2],tbox[1]),(tbox[2],tbox[3]),(tbox[0],tbox[3])],.99,'speech',['b'],tm,tm.copy(),{'paired_diff_method':'photo_pair'})
    reg=RegistrationResult(np.array([[2.,0.,0.],[0.,2.,0.],[0.,0.,1.]],np.float64),'scale',.99,1.,0.,.9,20,(240,180),(480,360),{})
    cfg=MaskReplaceConfig(
        min_match_confidence=.1,min_mask_iou=.4,min_target_coverage=.7,max_spill_ratio=.3,
        photo_pair_min_transfer_iou=.4,photo_pair_min_transfer_coverage=.7,photo_pair_max_spill_ratio=.3,
        preserve_target_border=False,feather_px=0,local_fit='bbox',
        text_fidelity_mode='auto',ink_reconstruction_enabled=True,reject_blurry_source=True,
        min_pixel_text_sharpness=1.0,min_relative_text_sharpness=0.05,
        photo_pair_prefer_ink_below_relative_sharpness=1.20,
    )
    result=transfer_bubble_patches(source,target,[sbub],[tbub],reg,cfg)
    assert result.applied_count==1, [r.to_dict() for r in result.records]
    assert result.records[0].clarity_mode in {'photo-crisp-ink','ink-reconstruction'}
