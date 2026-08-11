from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.masking import build_clear_mask
from manga_hd_transfer.models import BubbleInstance, TextBlock, TextUnit, UnitMatch


def test_text_mask_is_clipped_inside_protected_bubble():
    shape=(300,300)
    bubble=np.zeros(shape,np.uint8)
    cv2.circle(bubble,(150,150),100,255,-1)
    safe=cv2.erode(bubble,np.ones((9,9),np.uint8))
    b=BubbleInstance("bu",[(50,50),(250,50),(250,250),(50,250)],mask=bubble,safe_mask=safe,block_ids=["tb"])
    tb=TextBlock("tb",[(70,130),(230,130),(230,170),(70,170)],"日文",.99,bubble_id="bu")
    unit=TextUnit("tu",b.polygon,["tb"],"日文",.99,"speech",0,"bu")
    match=UnitMatch("su","tu",.99,.01)
    res=build_clear_mask(shape,[tb],[unit],[b],[match])
    assert cv2.countNonZero(res.mask)>0
    outside=cv2.bitwise_and(res.mask,cv2.bitwise_not(bubble))
    assert cv2.countNonZero(outside)==0
