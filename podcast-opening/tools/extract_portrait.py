"""Cut Brian's circular portrait out of the Canva end-screen (frame at 3.2 s) with clean edges.

The portrait is a person cut-out clipped by a circle at the shoulders, on a flat light background
with thin outline text behind it. Thin strokes are opened away, the shoulder circle is fitted from
the outer blazer edges, the white shirt is filled, and the light fringe is un-premultiplied.
usage: python3 extract_portrait.py frame.png out.png
"""
import json
import sys

import cv2
import numpy as np

src, out = sys.argv[1], sys.argv[2]
f = cv2.imread(src).astype(np.float32)
bg = np.median(f[5:60, 5:60].reshape(-1, 3), 0)
m = (np.abs(f - bg).sum(2) > 30).astype(np.uint8) * 255
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((17, 17), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(m)
m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
x, y, bw, bh = cv2.boundingRect(m)

pts = []
for yy in range(int(y + 0.80 * bh), y + bh - 6, 4):
    xs = np.nonzero(m[yy])[0]
    if len(xs):
        pts += [(xs.min(), yy), (xs.max(), yy)]
p = np.array(pts, np.float64)
cx, cy, c = np.linalg.lstsq(np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))], (p ** 2).sum(1), rcond=None)[0]
r = np.sqrt(c + cx * cx + cy * cy)

H, W = m.shape
yy, xx = np.mgrid[0:H, 0:W]
disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r - 2) ** 2
shoulders = int(y + 0.72 * bh)
m = (((m > 0) | (disc & (yy > shoulders))) & disc).astype(np.uint8) * 255
ff = m.copy()
cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 255)
m = m | cv2.bitwise_not(ff)                     # the white shirt inside the collar

a = cv2.GaussianBlur(cv2.erode(m, np.ones((5, 5), np.uint8)).astype(np.float32) / 255, (0, 0), 1.6)
a3 = np.clip(a, 1e-3, 1)[..., None]
col = np.clip((f - (1 - a3) * bg) / a3, 0, 255)
x, y, bw, bh = cv2.boundingRect(m)
pad = 40
x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + bw + pad, W), min(y + bh + pad, H)
cv2.imwrite(out, np.dstack([col[y0:y1, x0:x1], a[y0:y1, x0:x1] * 255]).astype(np.uint8))
json.dump({"cx": cx - x0, "cy": cy - y0, "r": r}, open(out.rsplit(".", 1)[0] + ".json", "w"))
print("portrait", x1 - x0, "x", y1 - y0, "circle r", round(r))
