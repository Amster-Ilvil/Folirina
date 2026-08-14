from __future__ import annotations

import numpy as np

from manga_hd_transfer.config import MatchingConfig
from manga_hd_transfer.models import RegistrationResult, TextUnit
from manga_hd_transfer.matching import match_units


def unit(uid, box, order):
    x0, y0, x1, y1 = box
    return TextUnit(uid, [(x0,y0),(x1,y0),(x1,y1),(x0,y1)], [uid+"-b"], "文本", .99, "speech", order)


def test_hungarian_unit_matching_after_registration():
    src = [unit("s0", (10,20,80,70),0), unit("s1",(200,250,290,320),1)]
    dst = [unit("t0", (30,35,100,85),0), unit("t1",(220,265,310,335),1)]
    reg = RegistrationResult(np.array([[1,0,20],[0,1,15],[0,0,1]],float), "test", .99,.9,1, .5,20,(400,400),(400,400))
    result = match_units(src,dst,reg,MatchingConfig(max_cost=.8,review_confidence=.4))
    pairs = {(m.source_unit_id,m.target_unit_id) for m in result.matches if m.relation=="one_to_one"}
    assert pairs == {("s0","t0"),("s1","t1")}
    assert not result.unmatched_source
