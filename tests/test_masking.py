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


def test_transcript_only_geometry_clears_target_ink_not_whole_balloon():
    shape = (260, 260)
    bubble = np.zeros(shape, np.uint8)
    cv2.ellipse(bubble, (130,130), (82,96), 0, 0, 360, 255, -1)
    safe = cv2.erode(bubble, np.ones((13,13),np.uint8))
    image = np.full((shape[0], shape[1], 3), 255, np.uint8)
    cv2.ellipse(image, (130,130), (82,96), 0, 0, 360, (0,0,0), 2)
    # Japanese-like vertical glyph strokes in the center.
    for y in (78, 102, 126, 150):
        cv2.rectangle(image, (121,y), (133,y+14), (20,20,20), -1)
    b = BubbleInstance('bu-synth', [(48,34),(212,34),(212,226),(48,226)], mask=bubble, safe_mask=safe, block_ids=['tb-synth'])
    tb = TextBlock('tb-synth', b.polygon, '□', 1.0, bubble_id='bu-synth', meta={'synthetic_geometry_only': True})
    unit = TextUnit('tu-synth', b.polygon, ['tb-synth'], '□', 1.0, 'speech', 0, 'bu-synth')
    res = build_clear_mask(shape, [tb], [unit], [b], [UnitMatch('su','tu-synth',1.0,0.0)], target_image=image)
    mask_area = cv2.countNonZero(res.mask)
    bubble_area = cv2.countNonZero(safe)
    assert mask_area > 100
    assert mask_area < bubble_area * 0.25
    assert res.mask[126, 126] > 0
    # Balloon outline must remain protected.
    assert res.mask[130, 48] == 0
