"""Local registered ink comparison for preserving identical SOURCE lettering."""
import cv2
import numpy as np


def equivalent_registered_ink(source, target, mask):
    if source is None or source.shape != target.shape or mask.shape != target.shape[:2]:
        return False
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return False
    x, y, w, h = cv2.boundingRect(points)
    if min(w, h) < 6:
        return False
    gate = mask[y:y+h, x:x+w] > 0
    # Dense container masks include burst rays/box borders. Compare lettering
    # in the central interior so differences in those background structures do
    # not dominate an otherwise identical number/string comparison.
    if np.mean(gate) >= 0.45:
        radius = max(1, int(min(w, h)*0.20))
        core = cv2.erode(gate.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        if np.count_nonzero(core) >= np.count_nonzero(gate)*0.25:
            gate = core
    signals = []
    for image in (source, target):
        gray = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        signals.append(gray - cv2.GaussianBlur(gray, (0, 0), 4))
    a, b = signals
    sa, sb = float(a[gate].std()), float(b[gate].std())
    if min(sa, sb) < 1:
        return False
    corr = float(np.mean((a[gate]-a[gate].mean())*(b[gate]-b[gate].mean()))/(sa*sb))
    ma, mb = (a < -18) & gate, (b < -18) & gate
    if min(np.count_nonzero(ma), np.count_nonzero(mb)) < 32:
        return False
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    near_a = cv2.dilate(ma.astype(np.uint8), kernel) > 0
    near_b = cv2.dilate(mb.astype(np.uint8), kernel) > 0
    compact_match = True
    for ink, near in ((ma, near_b), (mb, near_a)):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
        for index in range(1, count):
            _, _, cw, ch, area = stats[index]
            if area < 64 or max(cw, ch) > max(100, int(max(w, h)*0.65)) or max(cw,ch)/max(1,min(cw,ch)) > 5:
                continue
            if np.mean(near[labels == index]) < 0.94:
                compact_match = False
                break
    if compact_match and corr >= 0.70 and np.mean(near_b[ma]) >= 0.90 and np.mean(near_a[mb]) >= 0.90:
        return True
    # A colour edition may fill the same outlined effect glyph red while the
    # SOURCE leaves its centre white. Compare its two outlines, not RGB fill.
    src = source[y:y+h, x:x+w]
    dst = target[y:y+h, x:x+w]
    if np.median(cv2.cvtColor(src, cv2.COLOR_BGR2HSV)[..., 1][gate]) > 30:
        return False
    if np.mean(cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)[..., 1][gate] > 45) < 0.05:
        return False
    ea = (cv2.Canny(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), 40, 100) > 0) & gate
    eb = (cv2.Canny(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), 40, 100) > 0) & gate
    if min(np.count_nonzero(ea), np.count_nonzero(eb)) < 32:
        return False
    nea = cv2.dilate(ea.astype(np.uint8), kernel) > 0
    neb = cv2.dilate(eb.astype(np.uint8), kernel) > 0
    return bool(corr >= 0.50 and np.mean(nea[eb]) >= 0.98 and np.mean(neb[ea]) >= 0.80)
