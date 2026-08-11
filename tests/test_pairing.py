from __future__ import annotations

import cv2

from manga_hd_transfer.config import PairingConfig
from manga_hd_transfer.pairing import pair_directories
from manga_hd_transfer.io_utils import write_image
from .helpers import make_art_page


def test_directory_pairing_handles_extra_page(tmp_path):
    s=tmp_path/"s"; t=tmp_path/"t"; s.mkdir(); t.mkdir()
    for i in range(3):
        img=make_art_page(500,700)
        cv2.circle(img,(80+i*100,600),20+i*5,(0,0,0),-1)
        write_image(s/f"{i+1:03d}.png",img)
        resized=cv2.resize(img,(550,770))
        write_image(t/f"{i+1:03d}.png",resized)
    extra=make_art_page(500,700)
    cv2.putText(extra,"STAFF",(120,350),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),4)
    write_image(s/"004_staff.png",extra)
    pairs, us, ut=pair_directories(s,t,PairingConfig(max_pair_cost=.7,gap_penalty=.35))
    assert len(pairs)==3
    assert len(us)==1
    assert not ut
