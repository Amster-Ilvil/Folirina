from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from manga_hd_transfer.bubbles import detect_seeded_white_bubbles
from manga_hd_transfer.config import BubbleConfig, MaskReplaceConfig
from manga_hd_transfer.mask_transfer import transfer_bubble_patches
from manga_hd_transfer.models import RegistrationResult, TextBlock


def block(i, box, text):
    x0,y0,x1,y1=box
    return TextBlock(str(i),[(x0,y0),(x1,y0),(x1,y1),(x0,y1)],text,.99,reading_order=0)


def run(seed=731, trials=24):
    rng=np.random.default_rng(seed); rows=[]
    for i in range(trials):
        W,H=900,1200; target=np.full((H,W,3),255,np.uint8)
        cx=int(rng.integers(330,570)); cy=int(rng.integers(350,850)); rx=int(rng.integers(125,205)); ry=int(rng.integers(80,135))
        cv2.rectangle(target,(25,25),(W-25,H-25),(0,0,0),4); cv2.line(target,(30,210),(W-30,260),(0,0,0),3)
        cv2.ellipse(target,(cx,cy),(rx,ry),0,0,360,(0,0,0),5)
        tbox=(cx-rx//3,cy-ry//2,cx+rx//3,cy+ry//2)
        for x in np.linspace(tbox[0]+12,tbox[2]-12,3).astype(int):cv2.rectangle(target,(x,tbox[1]+8),(x+7,tbox[3]-8),(0,0,0),-1)

        scale=float(rng.uniform(.42,.78)); dx=int(rng.integers(-14,15)); dy=int(rng.integers(-14,15))
        small=cv2.resize(target,(int(W*scale),int(H*scale)),interpolation=cv2.INTER_AREA)
        source=cv2.warpAffine(small,np.array([[1,0,dx],[0,1,dy]],np.float32),(small.shape[1],small.shape[0]),borderValue=(255,255,255))
        sbox=tuple([tbox[0]*scale+dx,tbox[1]*scale+dy,tbox[2]*scale+dx,tbox[3]*scale+dy])
        cv2.rectangle(source,(int(sbox[0]-4),int(sbox[1]-4)),(int(sbox[2]+4),int(sbox[3]+4)),(255,255,255),-1)
        cv2.rectangle(source,(int(sbox[0]+7),int(sbox[1]+10)),(int(sbox[2]-7),int(sbox[1]+18)),(0,0,0),-1)
        cv2.rectangle(source,(int(sbox[0]+14),int(sbox[1]+28)),(int(sbox[2]-14),int(sbox[1]+36)),(0,0,0),-1)

        bcfg=BubbleConfig(white_threshold=200,safe_margin_px=8)
        sb=block('s',sbox,'中文'); tb=block('t',tbox,'日本語')
        sbs=detect_seeded_white_bubbles(source,[sb],bcfg); tbs=detect_seeded_white_bubbles(target,[tb],bcfg)
        Hm=np.array([[1/scale,0,-dx/scale],[0,1/scale,-dy/scale],[0,0,1]],np.float64)
        reg=RegistrationResult(Hm,'known',.99,1,0,.9,40,(source.shape[1],source.shape[0]),(W,H),{})
        cfg=MaskReplaceConfig(min_match_confidence=.30,min_mask_iou=.62,min_target_coverage=.90,max_spill_ratio=.13,local_fit='ecc',sr_backend='lanczos',sr_min_trigger=1.05,preserve_target_border=True,border_inset_px=3,feather_px=0,max_local_scale_change=.30)
        result=transfer_bubble_patches(source,target,sbs,tbs,reg,cfg)
        rec=result.records[0] if result.records else None
        border_pt=(cx-rx,cy)
        border_ok=bool(np.array_equal(result.image[border_pt[1],border_pt[0]],target[border_pt[1],border_pt[0]]))
        passed=bool(rec and rec.applied and rec.target_coverage>=cfg.min_target_coverage and rec.spill_ratio<=cfg.max_spill_ratio and border_ok)
        rows.append({'trial':i,'scale':scale,'dx':dx,'dy':dy,'pass':passed,'border_ok':border_ok,'record':rec.to_dict() if rec else None})
    passed=sum(1 for r in rows if r['pass'])
    report={'seed':seed,'trials':trials,'passed':passed,'pass_rate':passed/trials,'rows':rows}
    Path('benchmarks/mask_replace_latest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'trials':trials,'passed':passed,'pass_rate':report['pass_rate']},ensure_ascii=False,indent=2))
    if passed!=trials: raise SystemExit(1)


if __name__=='__main__':run()
