from __future__ import annotations

import cv2
import numpy as np


def selection_edge_mask(mask: np.ndarray, *, thickness: int = 2) -> np.ndarray:
    """Return a visible inner border for a binary selection crop.

    The selection UI renders only a tight ROI for performance. A rectangular
    ROI can therefore be entirely 255 and morphology-gradient sees no outside
    background, producing an invisible border. Erode with an explicit zero
    border so image-boundary/ROI-boundary selections retain a deterministic
    inner outline.
    """
    src=(np.asarray(mask,dtype=np.uint8)>0).astype(np.uint8)*255
    if src.size==0 or cv2.countNonZero(src)==0:
        return np.zeros_like(src)
    k=max(1,min(8,int(thickness)))
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(k*2+1,k*2+1))
    eroded=cv2.erode(src,kernel,iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    return cv2.subtract(src,eroded)


def selection_edge_thickness_for_scale(scale: float, *, display_px: float = 2.6, max_image_px: int = 14) -> int:
    """Choose image-space edge thickness that stays visible after view scaling.

    QGraphicsView scales the overlay pixmap together with the manga page.  A fixed
    two-image-pixel outline can shrink below one physical pixel when a 2K/4K page
    is fitted to the editor.  Convert the desired screen-space thickness back to
    image pixels and clamp it to a conservative range.
    """
    s=max(0.05, float(scale or 1.0))
    px=int(np.ceil(float(display_px)/s))
    return max(2, min(int(max_image_px), px))
