"""Bounded TARGET-colour repair of neutral paper leaked by raster transfers."""
from __future__ import annotations

import cv2
import numpy as np


def repair_coloured_paper_leaks(current: np.ndarray, target: np.ndarray,
                               scope: np.ndarray, *, protected_ink: np.ndarray | None = None):
    """Repair unexpected white/gray paper, never restore TARGET Japanese ink.

    The caller supplies write authority. Bright original highlights and SOURCE
    ink are protected. A locally verified flat colour or smooth linear gradient
    must explain the surrounding TARGET paper; complex artwork is left alone.
    """
    if current.shape != target.shape or current.shape[:2] != scope.shape:
        raise ValueError('coloured paper repair shape mismatch')
    if protected_ink is not None and protected_ink.shape != scope.shape:
        raise ValueError('protected source ink shape mismatch')
    out = current.copy(); repaired = np.zeros(scope.shape, np.uint8)
    neutral = (current.min(2) >= 230) & (np.ptp(current, axis=2) <= 20)
    original_white = (target.min(2) >= 230) & (np.ptp(target, axis=2) <= 20)
    candidate = neutral & ~original_white & (scope > 0)
    if protected_ink is not None:
        candidate &= protected_ink == 0
    if not np.any(candidate):
        return out, repaired, {'repaired_pixels': 0}
    x, y, w, h = cv2.boundingRect(candidate.astype(np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate[y:y+h,x:x+w].astype(np.uint8), connectivity=8)
    for label in range(1, count):
        xx, yy, ww, hh, _area = map(int, stats[label]); xx += x; yy += y
        pad = 18
        xa, ya = max(0,xx-pad), max(0,yy-pad)
        xb, yb = min(scope.shape[1],xx+ww+pad), min(scope.shape[0],yy+hh+pad)
        ref = target[ya:yb,xa:xb]
        hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
        paper = (hsv[:,:,1] >= 50) & (hsv[:,:,2] >= 180)
        if np.count_nonzero(paper) < 40:
            continue
        bins = hsv[:,:,0] // 8
        hist = np.bincount(bins[paper], minlength=23)
        dominant = int(hist.argmax())
        if hist[dominant] / np.count_nonzero(paper) < .70:
            continue
        paper &= bins == dominant
        # Dark glyph antialiasing retains the paper's hue. Hue alone therefore
        # cannot make those pixels valid background samples.
        paper &= hsv[:,:,2] >= np.percentile(hsv[:,:,2][paper],35)
        sy, sx = np.nonzero(paper)
        if len(sx) > 1024:
            take = np.linspace(0,len(sx)-1,1024,dtype=int); sy,sx=sy[take],sx[take]
        colours = ref[sy,sx].astype(np.float32)
        coords = np.column_stack((sx/max(1,xb-xa),sy/max(1,yb-ya),np.ones(len(sx))))
        fit = np.linalg.lstsq(coords,colours,rcond=None)[0]
        residual = np.max(np.abs(coords@fit-colours),axis=1)
        inliers = residual <= max(3.,float(np.median(residual))*3.)
        if np.count_nonzero(inliers) < 40:
            continue
        fit = np.linalg.lstsq(coords[inliers],colours[inliers],rcond=None)[0]
        residual = np.max(np.abs(coords[inliers]@fit-colours[inliers]),axis=1)
        # No flat-fill fallback on unverified textured/multi-colour artwork.
        if np.percentile(residual,90) > 9:
            continue
        component = (labels[yy-y:yy-y+hh,xx-x:xx-x+ww] == label).astype(np.uint8)
        cy,cx = np.nonzero(component)
        gy,gx = cy+yy,cx+xx
        at = np.column_stack(((gx-xa)/max(1,xb-xa),(gy-ya)/max(1,yb-ya),np.ones(len(gx))))
        colour = np.clip(np.rint(at@fit),0,255).astype(np.uint8)
        flat = np.max(np.percentile(colours[inliers],90,axis=0)-np.percentile(colours[inliers],10,axis=0)) <= 2
        if flat:
            colour[:] = np.rint(np.median(colours[inliers],axis=0)).astype(np.uint8)
        original = target[gy,gx]
        # Reuse unchanged TARGET colour/texture only when it agrees with paper;
        # original black/coloured Japanese glyphs can never pass this test.
        same_paper = np.max(np.abs(original.astype(np.int16)-colour.astype(np.int16)),axis=1) <= (0 if flat else 2)
        colour[same_paper] = original[same_paper]
        valid = (colour.max(1) >= 180) & (np.ptp(colour,axis=1) >= 30)
        out[gy[valid],gx[valid]] = colour[valid]
        repaired[gy[valid],gx[valid]] = 255

        # SOURCE paper transferred onto a colour bubble is often antialiased
        # against the page.  Its white core is easy to detect, but the 1–5px
        # pale-colour fringe is not neutral/bright enough for `candidate`.
        # Inspect only a tight halo around each *verified* white-paper seed and
        # accept pixels whose colour is mathematically consistent with TARGET
        # paper mixed toward white. This avoids broad yellowing/despeckle passes.
        halo_radius = 12
        padded_component = np.pad(component,halo_radius)
        kernel_size = halo_radius * 2 + 1
        halo = cv2.dilate(padded_component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))) > 0
        hy,hx = np.nonzero(halo)
        hg_y,hg_x = hy+yy-halo_radius,hx+xx-halo_radius
        inside = (hg_y>=ya)&(hg_y<yb)&(hg_x>=xa)&(hg_x<xb)
        hg_y,hg_x=hg_y[inside],hg_x[inside]
        in_scope = scope[hg_y,hg_x] > 0
        if protected_ink is not None:
            in_scope &= protected_ink[hg_y,hg_x] == 0
        in_scope &= ~original_white[hg_y,hg_x]
        if not np.any(in_scope):
            continue
        hg_y,hg_x = hg_y[in_scope],hg_x[in_scope]
        hat = np.column_stack(((hg_x-xa)/max(1,xb-xa),(hg_y-ya)/max(1,yb-ya),np.ones(len(hg_x))))
        paper_colour = np.clip(np.rint(hat@fit),0,255).astype(np.uint8)
        if flat:
            paper_colour[:] = np.rint(np.median(colours[inliers],axis=0)).astype(np.uint8)
        target_pixels = target[hg_y,hg_x]
        same_paper = np.max(np.abs(target_pixels.astype(np.int16)-paper_colour.astype(np.int16)),axis=1) <= (0 if flat else 2)
        paper_colour[same_paper] = target_pixels[same_paper]
        white_direction = 255.0-paper_colour.astype(np.float32)
        current_delta = current[hg_y,hg_x].astype(np.float32)-paper_colour.astype(np.float32)
        denom = np.sum(white_direction*white_direction,axis=1)
        alpha = np.divide(np.sum(current_delta*white_direction,axis=1),denom,
                          out=np.zeros(len(hg_x),np.float32),where=denom>1)
        mismatch = np.max(np.abs(current_delta-alpha[:,None]*white_direction),axis=1)
        chromatic_paper = (paper_colour.max(1)>=180) & (np.ptp(paper_colour,axis=1)>=30)
        fringe = (chromatic_paper & (alpha>=0.025) & (alpha<=1.10) & (mismatch<=9)
                  & (repaired[hg_y,hg_x]==0))
        fy,fx=hg_y[fringe],hg_x[fringe]
        out[fy,fx]=paper_colour[fringe]
        repaired[fy,fx]=255
    return out, repaired, {'repaired_pixels': int(np.count_nonzero(repaired))}
