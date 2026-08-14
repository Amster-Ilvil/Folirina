from __future__ import annotations
import cv2
from manga_hd_transfer.config import RegistrationConfig
from manga_hd_transfer.registration import register_images
from .helpers import make_art_page


def test_auto_registration_uses_cheap_fast_identity_for_same_source_page():
    target = make_art_page(900, 1200)
    source = target.copy()
    # Simulate translated glyphs inside one balloon while keeping page artwork fixed.
    cv2.rectangle(source, (210, 650), (385, 755), (255,255,255), -1)
    cv2.rectangle(source, (235, 675), (355, 688), (0,0,0), -1)
    cv2.rectangle(source, (245, 710), (345, 723), (0,0,0), -1)
    cfg=RegistrationConfig(backend='auto', fast_identity_min_correlation=.93)
    result=register_images(source,target,cfg)
    assert result.method == 'fast-phase-identity'
    assert result.confidence > .90
    assert result.diagnostics['route'] == 'fast_identity'


def test_auto_registration_fast_resize_for_lowres_same_edition():
    target=make_art_page(900,1200)
    source=cv2.resize(target,(600,800),interpolation=cv2.INTER_AREA)
    # Change a small translated text patch at source resolution.
    cv2.rectangle(source,(145,435),(255,500),(255,255,255),-1)
    cv2.rectangle(source,(165,455),(235,466),(0,0,0),-1)
    cfg=RegistrationConfig(backend='auto', fast_identity_min_correlation=.90)
    result=register_images(source,target,cfg)
    assert result.method == 'fast-resize-phase'
    assert abs(result.matrix[0,0]-1.5) < .01
    assert abs(result.matrix[1,1]-1.5) < .01
