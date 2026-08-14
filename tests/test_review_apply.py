from __future__ import annotations

import json
import cv2
import numpy as np

from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.io_utils import write_image
from manga_hd_transfer.models import PagePair, TextBlock
from manga_hd_transfer.ocr import InjectedOCRBackend
from manga_hd_transfer.pipeline import TransferPipeline
from manga_hd_transfer.review_apply import apply_review_page
from .helpers import draw_fake_text, make_art_page


def test_review_override_can_regenerate_page(tmp_path):
    target=make_art_page(1000,1400); box=(235,660,365,740); draw_fake_text(target,box,3)
    source=cv2.resize(target,(700,980)); sb=tuple(v*.7 for v in box)
    cv2.rectangle(source,(int(sb[0]),int(sb[1])),(int(sb[2]),int(sb[3])),(255,255,255),-1); draw_fake_text(source,sb,2)
    sp,tp=tmp_path/'s.png',tmp_path/'t.png'; write_image(sp,source); write_image(tp,target)
    s=TextBlock('s',[(sb[0],sb[1]),(sb[2],sb[1]),(sb[2],sb[3]),(sb[0],sb[3])],'初始中文',.99,reading_order=0)
    t=TextBlock('t',[(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])],'日本語',.99,reading_order=0)
    cfg=PipelineConfig(); cfg.registration.backend='opencv'; cfg.qa.registration_min_confidence=.3; cfg.matching.review_confidence=.3; cfg.qa.match_min_confidence=.3; cfg.export.layer_bundle=False
    page_dir=tmp_path/'out'; project=TransferPipeline(cfg,InjectedOCRBackend([s]),InjectedOCRBackend([t])).process_page(PagePair(str(sp),str(tp),0,0,.99,.01),page_dir)
    sid=project.source_units[0].id; tid=project.target_units[0].id
    (page_dir/'review_overrides.json').write_text(json.dumps({'text_overrides':{sid:'复核后的中文文本'},'match_overrides':{sid:tid},'accepted_source_units':[sid],'status':'approved'}),encoding='utf-8')
    out=apply_review_page(page_dir,cfg)
    assert out.exists()
    assert (page_dir/'review_applied.json').exists()
    payload=json.loads((page_dir/'review_applied.json').read_text(encoding='utf-8'))
    assert payload['status']=='approved'
    assert payload['lettering'][0]['text']=='复核后的中文文本'


def test_mask_replace_review_can_apply_manual_reletter(tmp_path):
    target = make_art_page(800, 1100)
    bubble = (240, 220, 520, 380)
    cv2.rectangle(target, (bubble[0], bubble[1]), (bubble[2], bubble[3]), (255, 255, 255), -1)
    draw_fake_text(target, bubble, 3)
    write_image(tmp_path / 'target_original.png', target)

    bgra = cv2.cvtColor(target, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = 0
    ok, data = cv2.imencode('.png', bgra)
    assert ok
    data.tofile(tmp_path / 'mask_transfer_layer.png')

    (tmp_path / 'bubbles').mkdir()
    mask = np.zeros(target.shape[:2], np.uint8)
    cv2.rectangle(mask, (bubble[0], bubble[1]), (bubble[2], bubble[3]), 255, -1)
    write_image(tmp_path / 'bubbles' / 'b1.png', mask)
    write_image(tmp_path / 'bubbles' / 'b1_safe.png', mask)

    (tmp_path / 'project.json').write_text(json.dumps({
        'meta': {'transfer_mode': 'mask_replace'},
        'target_bubbles': [{'id': 'b1', 'polygon': [[bubble[0], bubble[1]], [bubble[2], bubble[1]], [bubble[2], bubble[3]], [bubble[0], bubble[3]]], 'confidence': 1.0, 'kind': 'speech', 'block_ids': [], 'meta': {}}]
    }), encoding='utf-8')
    (tmp_path / 'review_overrides.json').write_text(json.dumps({
        'status': 'approved',
        'manual_reletter': [{'target_bubble_id': 'b1', 'text': '手动补全中文', 'orientation': 'horizontal'}]
    }, ensure_ascii=False), encoding='utf-8')

    out = apply_review_page(tmp_path, PipelineConfig())
    assert out.exists()
    payload = json.loads((tmp_path / 'review_applied.json').read_text(encoding='utf-8'))
    assert payload['status'] == 'approved'
    assert len(payload['manual_reletter_applied']) == 1
    assert payload['manual_reletter_applied'][0]['text'] == '手动补全中文'
    text_rgba = cv2.imread(str(tmp_path / 'text_layer_reviewed.png'), cv2.IMREAD_UNCHANGED)
    assert text_rgba is not None and text_rgba.shape[2] == 4
    assert int((text_rgba[:, :, 3] > 0).sum()) > 0


def test_mask_replace_candidate_can_restore_original_target(tmp_path):
    target = np.full((260, 360, 3), 255, np.uint8)
    box = (100, 70, 260, 190)
    cv2.rectangle(target, (150, 95), (155, 165), 0, -1)  # Japanese-like original text
    write_image(tmp_path / 'target_original.png', target)

    candidate = target.copy()
    candidate[box[1]:box[3], box[0]:box[2]] = 255
    cv2.putText(candidate, 'CN?', (125, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    bgra = cv2.cvtColor(candidate, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = 0
    bgra[box[1]:box[3], box[0]:box[2], 3] = 255
    ok, data = cv2.imencode('.png', bgra); assert ok; data.tofile(tmp_path / 'mask_transfer_layer.png')

    project = {
        'meta': {
            'transfer_mode': 'mask_replace',
            'mask_replace': {'manual_reletter_required': [{
                'target_bubble_id': 'candidate-1', 'target_bbox': list(box),
                'candidate_applied': True, 'restorable': True, 'editable': True,
                'reason': 'source_text_region_clipped_at_page_edge'
            }]}
        },
        'target_bubbles': []
    }
    (tmp_path / 'project.json').write_text(json.dumps(project), encoding='utf-8')
    (tmp_path / 'review_overrides.json').write_text(json.dumps({
        'status': 'restored', 'restore_target_bubbles': ['candidate-1']
    }), encoding='utf-8')

    out = apply_review_page(tmp_path, PipelineConfig())
    restored = cv2.imread(str(out))
    assert np.array_equal(restored, target)
    payload = json.loads((tmp_path / 'review_applied.json').read_text(encoding='utf-8'))
    assert payload['restored_targets'] == ['candidate-1']
    assert payload['unresolved_candidates'] == []
