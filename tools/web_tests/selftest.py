"""Native Web bridge smoke/regression tests. Run: python tools/web_tests/selftest.py
Optional real sample:
  FOLIRINA_WEB_TEST_CN=/path/cn.png FOLIRINA_WEB_TEST_JP=/path/jp.png python tools/web_tests/selftest.py
"""
from __future__ import annotations
import base64
from pathlib import Path
import os
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from web_bridge import _edit_selection, _selection_spec, process_render, process_register, process_warp
from manga_hd_transfer.region_selection import selection_mask_from_spec


def synthetic_pair():
    h,w=430,330
    t=np.full((h,w,3),(216,186,154),np.uint8)
    s=np.full((h,w,3),230,np.uint8)
    for im in (s,t):
        cv2.rectangle(im,(24,30),(296,395),(55,55,55),2)
        cv2.line(im,(24,180),(296,180),(30,30,30),3)
        cv2.rectangle(im,(65,205),(250,326),(241,241,241),-1)
    cv2.putText(t,'JAPAN',(84,250),cv2.FONT_HERSHEY_SIMPLEX,.72,(15,15,15),2)
    cv2.putText(s,'CHINA',(84,250),cv2.FONT_HERSHEY_SIMPLEX,.72,(15,15,15),2)
    return s,t


def run():
    s,t=synthetic_pair();h,w=t.shape[:2]
    stroke={'shape':'brush','brush':30,'points':[{'x':102,'y':235},{'x':156,'y':235},{'x':208,'y':235}]}
    for ed in (stroke,{'shape':'ellipse','start':{'x':80,'y':205},'end':{'x':240,'y':295}}):
        mask=_edit_selection(ed,w,h)
        spec=_selection_spec(mask)
        rebuilt=selection_mask_from_spec(spec,(h,w))
        assert not np.any(rebuilt[mask==0]),'Selection leaked outside authority'
        assert np.array_equal(rebuilt,mask),'Mask changed while serializing'
    for mode in ('texture','hole'):
        edit={**stroke,'kind':'transfer','mode':mode}
        result=process_render(s,t,np.eye(3).tolist(),[edit])
        raw=base64.b64decode(result['png'].split(',',1)[1]);img=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        mask=_edit_selection(stroke,w,h)
        assert np.count_nonzero(np.any(img!=t,axis=2)&(mask==0))==0
        assert result['receipts'][0]['changed']>0
    data=process_render(s,t,np.eye(3).tolist(),[{'kind':'restore',**stroke}])
    img=cv2.imdecode(np.frombuffer(base64.b64decode(data['png'].split(',')[1]),np.uint8),1)
    assert np.array_equal(img,t)
    print('PASS: native manual modes, mask authority, restoration, PNG encoding')

    c=os.getenv('FOLIRINA_WEB_TEST_CN');j=os.getenv('FOLIRINA_WEB_TEST_JP')
    if c and j:
        source=cv2.imread(c);target=cv2.imread(j)
        assert source is not None and target is not None
        registration=process_register(source,target)
        print('Real page registration:',registration['method'],registration['confidence'])
        assert registration['confidence']>=.55
        ww=process_warp(source,target,registration['matrix']);assert ww['png'].startswith('data:image/png;base64,')
        print('PASS: real-page original OpenCV registration & warp')

if __name__=='__main__':run()
