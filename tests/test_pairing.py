from __future__ import annotations

from pathlib import Path
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


def test_name_pairing_matches_same_page_number_across_edition_names(tmp_path, monkeypatch):
    s=tmp_path/"s"; t=tmp_path/"t"; s.mkdir(); t.mkdir()
    img=make_art_page(320,480)
    write_image(s/"006(2).jpg", img)
    write_image(t/"p-006(1).jpeg", cv2.resize(img,(360,540)))

    # Name/page-number priority should avoid the expensive visual fingerprint path.
    import manga_hd_transfer.pairing as pairing_mod
    def fail_fingerprint(*args, **kwargs):
        raise AssertionError("filename pairing should not fingerprint a locked page")
    monkeypatch.setattr(pairing_mod, "fingerprint_image", fail_fingerprint)

    cfg=PairingConfig(prefer_name_pairing=True, prefer_order_pairing=False)
    pairs,us,ut=pair_directories(s,t,cfg)
    assert len(pairs)==1 and not us and not ut
    assert pairs[0].source_index==0 and pairs[0].target_index==0
    assert "pairing=name" in pairs[0].reasons
    assert "name_match=page_number" in pairs[0].reasons


def test_order_pairing_matches_natural_folder_order_when_names_differ(tmp_path, monkeypatch):
    s=tmp_path/"s"; t=tmp_path/"t"; s.mkdir(); t.mkdir()
    for i,name in enumerate(["cn_a.png","cn_b.png","cn_c.png"]):
        img=make_art_page(300,420)
        cv2.putText(img,str(i),(40,200),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),4)
        write_image(s/name,img)
    for i,name in enumerate(["raw_x.png","raw_y.png","raw_z.png"]):
        # Deliberately make visual content unrelated: order mode must still win.
        img=make_art_page(300,420)
        cv2.rectangle(img,(30+i*20,40),(250,360-i*30),(0,0,0),8)
        write_image(t/name,img)

    import manga_hd_transfer.pairing as pairing_mod
    def fail_fingerprint(*args, **kwargs):
        raise AssertionError("equal-length order pairing should not fingerprint locked pages")
    monkeypatch.setattr(pairing_mod, "fingerprint_image", fail_fingerprint)

    cfg=PairingConfig(prefer_name_pairing=False, prefer_order_pairing=True)
    pairs,us,ut=pair_directories(s,t,cfg)
    assert not us and not ut and len(pairs)==3
    assert [(p.source_index,p.target_index) for p in pairs]==[(0,0),(1,1),(2,2)]
    assert all("pairing=order" in p.reasons for p in pairs)


def test_name_then_order_then_smart_respects_extra_page_segment(tmp_path):
    s=tmp_path/"s"; t=tmp_path/"t"; s.mkdir(); t.mkdir()
    pages=[]
    for i in range(4):
        img=make_art_page(420,600)
        cv2.circle(img,(80+i*70,500-i*50),25,(0,0,0),-1)
        pages.append(img)
    # Exact/page-number anchors at the ends.
    write_image(s/"001_cn.png",pages[0]); write_image(t/"001_jp.png",cv2.resize(pages[0],(462,660)))
    write_image(s/"002_cn.png",pages[1])
    extra=make_art_page(420,600); cv2.putText(extra,"EXTRA",(60,300),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,0),4)
    write_image(s/"0025_extra.png",extra)
    write_image(s/"003_cn.png",pages[2]); write_image(t/"zzz_003_jp.png",cv2.resize(pages[2],(462,660)))
    write_image(t/"middle_unknown.png",cv2.resize(pages[1],(462,660)))

    cfg=PairingConfig(prefer_name_pairing=True, prefer_order_pairing=True,max_pair_cost=.72,gap_penalty=.35)
    pairs,us,ut=pair_directories(s,t,cfg)
    by_target={Path(p.target_path).name: p for p in pairs}
    assert "001_jp.png" in by_target and "pairing=name" in by_target["001_jp.png"].reasons
    assert "zzz_003_jp.png" in by_target and "pairing=name" in by_target["zzz_003_jp.png"].reasons
    assert "middle_unknown.png" in by_target
    assert Path(by_target["middle_unknown.png"].source_path).name=="002_cn.png"
    assert "pairing=order" in by_target["middle_unknown.png"].reasons
    assert any(Path(p).name=="0025_extra.png" for p in us)
    assert not ut


def test_pairing_can_disable_both_priorities_and_use_legacy_smart(tmp_path):
    s=tmp_path/"s"; t=tmp_path/"t"; s.mkdir(); t.mkdir()
    img=make_art_page(360,520)
    cv2.circle(img,(90,400),35,(0,0,0),-1)
    write_image(s/"same.png",img)
    write_image(t/"same.png",cv2.resize(img,(396,572)))
    cfg=PairingConfig(prefer_name_pairing=False,prefer_order_pairing=False,max_pair_cost=.75)
    pairs,us,ut=pair_directories(s,t,cfg)
    assert len(pairs)==1 and not us and not ut
    assert "pairing=smart" in pairs[0].reasons


def test_pairing_priority_shortcuts_are_off_by_default():
    cfg = PairingConfig()
    assert cfg.prefer_name_pairing is False
    assert cfg.prefer_order_pairing is False
