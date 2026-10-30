"""Local document thresholding for verified white speech paper only.

Sauvola-style neighbourhood statistics separate slow illumination changes from
ink; the resulting support retains the original greyscale antialias ramp.
This is not a page-wide binarizer and must not be used on artwork/colour paper.
"""
import cv2
import numpy as np


def shadow_free_ink(gray, support):
    use = np.asarray(support, bool)
    g = np.asarray(gray, np.uint8)
    empty = np.zeros_like(g, bool)
    mid = use & (g >= 110) & (g < 210)
    if np.count_nonzero(mid) < max(100, np.count_nonzero(use) * .08):
        return None
    ys, xs = np.where(use)
    x0, x1 = max(0,int(xs.min())-32), min(g.shape[1],int(xs.max())+33)
    y0, y1 = max(0,int(ys.min())-32), min(g.shape[0],int(ys.max())+33)
    f = g[y0:y1,x0:x1].astype(np.float32)
    mean = cv2.boxFilter(f,-1,(31,31),borderType=cv2.BORDER_REPLICATE)
    variance = np.maximum(0,cv2.boxFilter(f*f,-1,(31,31),borderType=cv2.BORDER_REPLICATE)-mean*mean)
    threshold = mean * (1 + .18 * (np.sqrt(variance)/128 - 1))
    core = (f < threshold).astype(np.uint8)
    # Include one antialias pixel, not a broad shadow halo.
    near = cv2.dilate(core,np.ones((3,3),np.uint8)) > 0
    near &= f < (mean - 4)
    ink = empty.copy(); ink[y0:y1,x0:x1] = near
    ink &= use
    return ink
