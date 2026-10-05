"""Cut Brian's circular portrait out of the Canva end-screen (frame at 3.2 s) with clean edges.

The portrait is a person cut-out clipped by a circle at the shoulders, on a flat light background
with thin outline text behind it. Thin strokes are opened away, the shoulder circle is fitted from
the outer blazer edges, the white shirt is filled, pink text specks are dropped, the edge is
trimmed, and any light fringe is recoloured from just inside the silhouette (no halo on navy).
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

# --- clean edge: no light halo, no pink specks from the old outline text -----------------------
fb, fg, fr = f[..., 0], f[..., 1], f[..., 2]
band = (cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0) & (cv2.erode(m, np.ones((31, 31), np.uint8)) == 0)
pink = band & (fr - fg > 18) & (fb > fg - 8) & (fr > 170)
m[pink] = 0
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
m = cv2.erode(m, np.ones((7, 7), np.uint8))                    # trim the outer fringe
a = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.3)
# colours from safely inside the silhouette, spread outward to the edge
inner = (cv2.erode(m, np.ones((17, 17), np.uint8)) > 0).astype(np.float32)
fill = cv2.GaussianBlur(f * inner[..., None], (0, 0), 6) / (cv2.GaussianBlur(inner, (0, 0), 6)[..., None] + 1e-4)
edge = ((m > 0) & (inner == 0)).astype(np.float32)
edge = cv2.GaussianBlur(edge, (0, 0), 2)[..., None]
lum = f.mean(2, keepdims=True)
lighter = np.clip((lum - fill.mean(2, keepdims=True)) / 40.0, 0, 1)   # only pixels lighter than inside
col = f * (1 - edge * lighter) + fill * (edge * lighter)
# white shirt reaches the circle edge legitimately: keep it there
shirt = (yy > shoulders) & ((xx - cx) ** 2 + (yy - cy) ** 2 > (r - 40) ** 2)
col = np.where(shirt[..., None], f, col)
x, y, bw, bh = cv2.boundingRect(m)
pad = 40
x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + bw + pad, W), min(y + bh + pad, H)
cv2.imwrite(out, np.dstack([col[y0:y1, x0:x1], a[y0:y1, x0:x1] * 255]).astype(np.uint8))
json.dump({"cx": cx - x0, "cy": cy - y0, "r": r}, open(out.rsplit(".", 1)[0] + ".json", "w"))
print("portrait", x1 - x0, "x", y1 - y0, "circle r", round(r))
