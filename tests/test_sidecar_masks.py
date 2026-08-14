from __future__ import annotations

import json
import cv2
import numpy as np

from manga_hd_transfer.bubbles import load_bubble_sidecar
from manga_hd_transfer.config import BubbleConfig, MaskingConfig
from manga_hd_transfer.io_utils import write_image
from manga_hd_transfer.masking import build_clear_mask
from manga_hd_transfer.models import TextUnit, UnitMatch
from manga_hd_transfer.ocr import SidecarOCRBackend


def test_sidecar_pixel_masks_are_used_for_clear_mask(tmp_path):
    img=np.full((200,200,3),255,np.uint8)
    cv2.circle(img,(100,100),75,(0,0,0),3)
    page=tmp_path/'p.png'; write_image(page,img)
    textmask=np.zeros((200,200),np.uint8); cv2.rectangle(textmask,(90,90),(110,110),255,-1)
    write_image(tmp_path/'textmask.png',textmask)
    (tmp_path/'p.ocr.json').write_text(json.dumps({"blocks":[{"id":"tb","bbox":[60,70,140,130],"text":"日文","confidence":.99,"mask_path":"textmask.png"}]}),encoding='utf-8')
    blocks=SidecarOCRBackend().recognize(img,image_path=page)
    bubblemask=np.zeros((200,200),np.uint8); cv2.circle(bubblemask,(100,100),72,255,-1); write_image(tmp_path/'bubble.png',bubblemask)
    (tmp_path/'p.bubbles.json').write_text(json.dumps({"bubbles":[{"id":"bu","polygon":[[28,100],[100,28],[172,100],[100,172]],"mask_path":"bubble.png"}]}),encoding='utf-8')
    bubbles=load_bubble_sidecar(img,page,blocks,BubbleConfig(backend='sidecar'))
    unit=TextUnit('tu',bubbles[0].polygon,['tb'],'日文',.99,'speech',0,'bu')
    res=build_clear_mask(img.shape[:2],blocks,[unit],bubbles,[UnitMatch('su','tu',.99,.01)],MaskingConfig(dilation_ratio=0,min_dilation_px=0,max_dilation_px=0))
    # Pixel segmentation is much smaller than the deliberately oversized OCR bbox.
    assert 350 <= cv2.countNonZero(res.mask) <= 500
