"""Local background reconstruction for bounded, outlined text on artwork."""
from __future__ import annotations

import cv2
import numpy as np


def outlined_text_clear(image, clear, scope):
    """Include the white halo next to erased dark glyphs, within text authority.

    White glyph outlines can connect to a white hand or clothing. Grow only a
    bounded distance from the dark strokes instead of flood-filling that white
    component into the artwork.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), .8)
    envelope = cv2.dilate(scope, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    dark = ((gray < 160) & (smooth < 185) & (clear > 0) & (envelope > 0)).astype(np.uint8)
    near = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
    halo = ((smooth > 225) & near & (envelope > 0)).astype(np.uint8) * 255
    halo = cv2.dilate(halo, np.ones((3, 3), np.uint8))
    halo[envelope == 0] = 0
    return np.maximum(clear, halo)


def composite_outlined_source(base, source, text_gate):
    """Transfer dark and white glyph opacity from a paired SOURCE text box.

    Side bands estimate the source paper independently for each row. This keeps
    the outline, while preventing source paper and screentone from becoming ink.
    """
    ys, xs = np.where(text_gate > 0)
    if not len(xs):
        return base.copy(), np.zeros(text_gate.shape, np.uint8), np.zeros(text_gate.shape, np.uint8)
    x0, x1 = int(xs.min()), int(xs.max())+1
    y0, y1 = int(ys.min()), int(ys.max())+1
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bands = np.concatenate((gray[y0:y1,max(0,x0-10):x0],gray[y0:y1,x1:min(gray.shape[1],x1+10)]),axis=1)
    if bands.shape[1] == 0:
        return base.copy(), np.zeros(text_gate.shape,np.uint8), np.zeros(text_gate.shape,np.uint8)
    paper = np.median(bands,axis=1)[:,None]
    crop = gray[y0:y1,x0:x1]
    dark = np.clip((paper-crop)/np.maximum(40,paper-20),0,1)
    dark[(paper-crop)<25]=0
    near = cv2.dilate((dark>.15).astype(np.uint8),np.ones((9,9),np.uint8))>0
    white = np.clip((crop-paper)/np.maximum(35,255-paper),0,1)
    white[((crop-paper)<25)|~near]=0
    gate = text_gate[y0:y1,x0:x1]>0
    dark[~gate]=0;white[~gate]=0
    out=base.copy();roi=out[y0:y1,x0:x1].astype(np.float32)
    roi=roi*(1-dark[:,:,None])
    roi=roi*(1-white[:,:,None])+255*white[:,:,None]
    out[y0:y1,x0:x1]=np.rint(roi).clip(0,255).astype(np.uint8)
    ink=np.zeros(text_gate.shape,np.uint8);write=ink.copy()
    ink[y0:y1,x0:x1]=(dark>.08).astype(np.uint8)*255
    write[y0:y1,x0:x1]=((dark>0)|(white>0)).astype(np.uint8)*255
    return out,write,ink


def clean_text_background(image: np.ndarray, clear: np.ndarray, scope: np.ndarray):
    """Use an installed LaMa checkpoint for textured text; never fetch weights.

    The semantic text footprint includes white glyph outlines that a dark-ink
    mask misses. Only that bounded footprint is admitted to reconstruction.
    """
    from .lama_manga_runtime import status, inpaint_koharu_text

    ring = cv2.dilate(scope, np.ones((17, 17), np.uint8)) > 0
    ring &= scope == 0
    values = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[ring]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inside = gray[scope > 0]
    white_paper = len(inside) > 0 and np.mean(inside >= 248) > .78
    textured = not white_paper and len(values) >= 32 and float(np.std(values)) > 12
    bounded = np.count_nonzero(scope) <= image.shape[0] * image.shape[1] * .025
    if textured and bounded and status().ready:
        effective = outlined_text_clear(image, np.maximum(clear, scope), scope)
        cleaned, details = inpaint_koharu_text(image, effective)
        return cleaned, effective, {**details, "backend": "lama-manga-textured", "authority_allow_mask": effective}
    return cv2.inpaint(image, clear, 2.5, cv2.INPAINT_TELEA), clear, {"backend": "opencv-telea"}


def outlined_source_support(source, ink, gate):
    """Recover the white outline and enclosed dark core of raster lettering."""
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    bright = ((gray >= 238) & (gate > 0)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    support = ink.copy()
    area = max(1, np.count_nonzero(gate))
    near = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    for contour in contours:
        size = cv2.contourArea(contour)
        if size < 12 or size > area * .45:
            continue
        filled = np.zeros_like(ink)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
        if np.count_nonzero((filled > 0) & (near > 0)) < max(8, size * .08):
            continue
        support = np.maximum(support, filled)
    return cv2.bitwise_and(support, gate)
