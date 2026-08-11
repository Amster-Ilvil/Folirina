from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manga_hd_transfer.config import LetteringConfig, RegistrationConfig
from manga_hd_transfer.lettering import fit_text
from manga_hd_transfer.models import TextUnit
from manga_hd_transfer.registration import register_images


def art(seed: int, w=800, h=1100):
    rng=np.random.default_rng(seed)
    img=np.full((h,w,3),255,np.uint8)
    cv2.rectangle(img,(25,25),(w-25,h-25),(0,0,0),4)
    for _ in range(28):
        x=int(rng.integers(50,w-50)); y=int(rng.integers(50,h-50)); r=int(rng.integers(8,55))
        if rng.random()<.5: cv2.circle(img,(x,y),r,(int(rng.integers(0,90)),)*3,int(rng.integers(2,5)))
        else:
            x2=int(np.clip(x+rng.integers(-100,100),10,w-10)); y2=int(np.clip(y+rng.integers(-100,100),10,h-10)); cv2.line(img,(x,y),(x2,y2),(0,0,0),int(rng.integers(1,4)))
    cv2.ellipse(img,(w//2,h//2),(150,100),0,0,360,(0,0,0),4)
    return img


def registration_benchmark(trials=20):
    rng=np.random.default_rng(20260811)
    errors=[]; confidences=[]; failures=[]
    for i in range(trials):
        src=art(i)
        angle=float(rng.uniform(-3.0,3.0)); scale=float(rng.uniform(.86,1.16)); dx=float(rng.uniform(-35,35)); dy=float(rng.uniform(-35,35))
        a=cv2.getRotationMatrix2D((400,550),angle,scale); a[:,2]+=(dx,dy)
        dst=cv2.warpAffine(src,a,(850,1160),borderValue=(255,255,255))
        # Language glyph differences are simulated with different bars in the bubble.
        cv2.rectangle(src,(350,520),(450,535),(0,0,0),-1)
        cv2.rectangle(dst,(360,525),(445,537),(0,0,0),-1); cv2.rectangle(dst,(375,550),(430,561),(0,0,0),-1)
        reg=register_images(src,dst,RegistrationConfig(backend='opencv',feature='sift',min_matches=8,review_confidence=.40))
        pts=np.array([[[100,100],[650,140],[120,900],[680,900],[400,550]]],np.float32)
        expected=cv2.transform(pts,a)[0]; actual=cv2.perspectiveTransform(pts,reg.matrix)[0]
        err=float(np.median(np.linalg.norm(expected-actual,axis=1)))
        errors.append(err); confidences.append(reg.confidence)
        if err>5.0 or reg.confidence<.40: failures.append({'trial':i,'error':err,'confidence':reg.confidence,'method':reg.method})
    return {'trials':trials,'median_error_px':float(np.median(errors)),'p95_error_px':float(np.quantile(errors,.95)),'min_confidence':float(np.min(confidences)),'failures':failures,'pass':not failures}


def lettering_benchmark():
    shape=(600,600); safe=np.zeros(shape,np.uint8); cv2.ellipse(safe,(300,300),(190,125),0,0,360,255,-1); safe=cv2.erode(safe,np.ones((17,17),np.uint8))
    texts=['你好。','这是中文漫画排字测试。','为了验证不同长度的中文对白能够安全地放进气泡，我们需要进行自动断行与字号搜索。','「标点」不能随便跑到行首，排版也不能越过气泡安全区！']
    results=[]
    for i,text in enumerate(texts):
        unit=TextUnit(f'u{i}',[(100,175),(500,175),(500,425),(100,425)],[f'b{i}'],text,.99,'speech',i)
        r=fit_text(shape,safe,unit,text,LetteringConfig(max_font_size=54,min_font_size=10))
        results.append({'text_len':len(text),'success':r.success,'font_size':r.font_size,'coverage':r.coverage_inside_safe})
    return {'cases':results,'pass':all(x['success'] and x['coverage']>=.997 for x in results)}


def main():
    report={'registration':registration_benchmark(),'lettering':lettering_benchmark()}
    report['pass']=all(v['pass'] for v in report.values())
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(0 if report['pass'] else 1)

if __name__=='__main__': main()
