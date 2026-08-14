"""Run paired-difference mask replacement on one translated/source page pair.

Usage:
  python benchmarks/paired_page_acceptance.py translated.png japanese.jpg out_dir
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.models import PagePair
from manga_hd_transfer.pipeline import TransferPipeline


def main():
    if len(sys.argv)!=4:
        raise SystemExit('usage: paired_page_acceptance.py TRANSLATED JAPANESE OUT_DIR')
    source,target,out=map(Path,sys.argv[1:])
    cfg=PipelineConfig(); cfg.transfer.mode='mask_replace'; cfg.ocr.backend='none'; cfg.ocr.source_backend='none'; cfg.ocr.target_backend='none'; cfg.export.layer_bundle=False; cfg.export.save_debug=True
    pair=PagePair(str(source),str(target),0,0,1.0,0.0,['paired-page-acceptance'])
    project=TransferPipeline(cfg).process_page(pair,out)
    result=cv2.imread(project.artifacts['final']); src=cv2.imread(str(source)); tgt=cv2.imread(str(target)); mask=cv2.imread(project.artifacts['mask_transfer_mask'],0)
    # Source page acts as the reference only for pixels that actually differ from target.
    d=np.mean(np.abs(src.astype(np.int16)-tgt.astype(np.int16)),axis=2); changed=d>=cfg.mask_replace.paired_diff_pixel_threshold
    e=np.mean(np.abs(result.astype(np.int16)-src.astype(np.int16)),axis=2)
    use=mask>0
    metrics={
        'detected_regions':len(project.meta.get('paired_diff',{}).get('records',[])),
        'applied_regions':project.meta.get('mask_replace',{}).get('applied_count',0),
        'qa':project.meta.get('qa_summary',{}),
        'inside_mask_exact':float(np.all(result[use]==src[use],axis=1).mean()) if np.any(use) else 0.0,
        'outside_mask_target_exact':float(np.all(result[~use]==tgt[~use],axis=1).mean()) if np.any(~use) else 0.0,
        'changed_pixel_reconstruction':float(((e==0)&changed).sum()/max(1,changed.sum())),
    }
    print(json.dumps(metrics,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
