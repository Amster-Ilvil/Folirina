from __future__ import annotations

"""Layout-guided local manga OCR adapters.

Manga OCR and Baberu OCR are crop recognizers.  Whole-page OCR therefore uses
Koharu Layout text/SFX instances only as geometry, then recognizes each crop.
No OCR result may expand or alter the detector mask; this keeps the integration
usable as evidence without changing Direct/Mask renderer contracts.
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .config import OCRConfig
from .external_command import run_external_command
from .geometry import bbox_polygon
from .io_utils import write_image
from .model_downloads import discovered_model_path
from .models import TextBlock
from .ocr import OCRBackend


class KoharuCropOCRBackend(OCRBackend):
    retry_crops = False

    def __init__(self, backend: str, lang: str, config: OCRConfig) -> None:
        self.backend = str(backend)
        self.lang = str(lang)
        self.config = config
        if self.backend == "manga_ocr":
            self.model_path = discovered_model_path("manga_ocr", config.manga_ocr_model_path)
        elif self.backend == "baberu_ocr":
            self.model_path = discovered_model_path("baberu_ocr", config.baberu_ocr_model_path)
        elif self.backend == "ocr48px":
            self.model_path = discovered_model_path("ocr48px", config.ocr48px_model_path)
        else:
            raise ValueError(self.backend)
        if self.backend != "ocr48px" and self.model_path is None:
            raise RuntimeError(f"{self.backend} 模型未就绪；请在模型中心下载/离线导入。")
        if self.backend == "ocr48px" and self.model_path is None and not config.ocr48px_command:
            raise RuntimeError("48px AR OCR 已选择，但原生模型尚未下载/导入，也没有配置外部 ocr48px_command。")

    def _recognize_crop_text(self, crop: np.ndarray) -> tuple[str, float, dict]:
        if crop.size == 0:
            return "", 0.0, {"reason":"empty_crop"}
        if self.backend == "manga_ocr":
            from .vision_runtime import run_manga_ocr
            row = run_manga_ocr(
                crop, model_dir=str(self.model_path), device="auto",
                max_new_tokens=int(self.config.koharu_ocr_max_new_tokens),
            )
            return str(row.get("text") or "").strip(), float(row.get("confidence",1.0)), {
                "backend":"manga_ocr", "device":row.get("device"), "isolated_runtime":True,
            }
        if self.backend == "baberu_ocr":
            from .vision_runtime import run_baberu_ocr
            row = run_baberu_ocr(
                crop, model_dir=str(self.model_path),
                max_new_tokens=int(self.config.koharu_ocr_max_new_tokens),
            )
            return str(row.get("text") or "").strip(), float(row.get("confidence",1.0)), {
                "backend":"baberu_ocr", "device":row.get("device"), "isolated_runtime":True,
            }
        return self._run_48px(crop)

    def _run_48px(self, crop: np.ndarray) -> tuple[str, float, dict]:
        if self.model_path is not None:
            from .vision_runtime import run_ocr48px
            row = run_ocr48px(
                crop, model_dir=str(self.model_path), device="auto",
                beams_k=5, max_seq_length=255,
            )
            return str(row.get("text") or "").strip(), float(row.get("confidence",0.0)), {
                "backend":"ocr48px", "device":row.get("device"), "isolated_runtime":True,
                "input_width":row.get("input_width"), "orientation":row.get("orientation"),
                "native":True,
            }
        command = str(self.config.ocr48px_command or "")
        with tempfile.TemporaryDirectory(prefix="folirina-ocr48px-") as td:
            root=Path(td); inp=root/"input.png"; out=root/"output.txt"
            write_image(inp,crop)
            proc=run_external_command(
                command,{"input":inp,"output":out},
                timeout=int(self.config.ocr48px_timeout_seconds),
                allow_shell=bool(self.config.ocr48px_allow_shell),
            )
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(f"48px OCR command failed ({proc.returncode}): {proc.stderr[-2000:]}")
            raw=out.read_text(encoding="utf-8",errors="replace").strip(); text=raw; confidence=1.0
            if raw.startswith("{"):
                try:
                    payload=json.loads(raw); text=str(payload.get("text") or ""); confidence=float(payload.get("confidence",1.0))
                except Exception:
                    pass
            return text.strip(), confidence, {"backend":"ocr48px","command_exit":proc.returncode,"shell":proc.shell}

    def recognize_region(self, page_image: np.ndarray, bbox: tuple[int,int,int,int], *, image_path=None) -> list[TextBlock]:
        x0,y0,x1,y1=[int(v) for v in bbox]; h,w=page_image.shape[:2]
        x0=max(0,min(w,x0)); x1=max(x0,min(w,x1)); y0=max(0,min(h,y0)); y1=max(y0,min(h,y1))
        crop=page_image[y0:y1,x0:x1]
        text,conf,meta=self._recognize_crop_text(crop)
        if not text:
            return []
        return [TextBlock(
            id=f"{self.backend}-region-0000", polygon=bbox_polygon((0,0,max(1,x1-x0),max(1,y1-y0))),
            text=text, confidence=conf, kind="unknown", reading_order=0, meta=meta,
        )]

    def recognize(self, image: np.ndarray, *, image_path=None) -> list[TextBlock]:
        layout = discovered_model_path("koharu_layout", self.config.koharu_layout_model_path)
        if layout is None:
            raise RuntimeError(
                f"{self.backend} 是区域 OCR。整页识别需要 Koharu Layout 模型；"
                "请下载/导入 Layout 模型，或只在已知 Region 上使用该 OCR。"
            )
        from .vision_runtime import run_koharu_layout
        payload=run_koharu_layout(
            image, model_dir=str(layout), device="auto",
            text_threshold=float(self.config.koharu_layout_text_threshold),
            sfx_threshold=float(self.config.koharu_layout_sfx_threshold),
            bubble_threshold=1.0, panel_threshold=1.0,
            shape=int(self.config.koharu_layout_shape),
        )
        h,w=image.shape[:2]; pad_ratio=float(self.config.koharu_crop_padding_ratio)
        regions=[]
        for row in list(payload.get("items") or []):
            if str(row.get("label")) not in {"text","onomatopoeia"}:
                continue
            box=row.get("box") or []
            if len(box)!=4:
                pts=np.asarray(row.get("polygon") or [],dtype=np.float32)
                if len(pts)<3: continue
                x0,y0=pts.min(axis=0); x1,y1=pts.max(axis=0); box=[x0,y0,x1,y1]
            x0,y0,x1,y1=[float(v) for v in box]; pw=max(2.0,(x1-x0)*pad_ratio); ph=max(2.0,(y1-y0)*pad_ratio)
            ix0=max(0,int(np.floor(x0-pw))); iy0=max(0,int(np.floor(y0-ph))); ix1=min(w,int(np.ceil(x1+pw))); iy1=min(h,int(np.ceil(y1+ph)))
            regions.append((ix0,iy0,ix1,iy1,row))
        # Manga pages are commonly vertical: right-most candidates first, then top.
        regions.sort(key=lambda r:(-r[0],r[1]))
        blocks=[]
        for i,(x0,y0,x1,y1,row) in enumerate(regions):
            text,conf,meta=self._recognize_crop_text(image[y0:y1,x0:x1])
            if not text: continue
            meta.update({"layout_backend":"koharu_layout","layout_label":row.get("label"),"layout_confidence":float(row.get("confidence",0.0))})
            blocks.append(TextBlock(
                id=f"{self.backend}-{i:04d}", polygon=bbox_polygon((x0,y0,x1,y1)), text=text,
                confidence=min(float(conf),float(row.get("confidence",1.0))), kind="sfx" if row.get("label")=="onomatopoeia" else "unknown",
                reading_order=len(blocks), meta=meta,
            ))
        return blocks


__all__=["KoharuCropOCRBackend"]
