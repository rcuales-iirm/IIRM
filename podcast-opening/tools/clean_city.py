"""Remove the burned-in 'Build and Cover Your ASSets' title from the city shot (src frames 0-57).

Frame 7 is the last frame before the old title animates in. It is aligned to every later
frame with a partial-affine fit on the right-hand skyline (no text there), smoothed over time, and pixels that
differ strongly from it inside the old-title area are replaced with the aligned clean plate.
"""
import sys
from common import *
from common import smooth_affines

src, out = sys.argv[1], sys.argv[2]
frames = read_frames(src, 0, 57)
ref = frames[7]
feat = np.zeros((H, W), np.uint8)
feat[:, int(W * 0.62):] = 255          # skyline only, never the old title
zone = np.zeros((H, W), np.float32)
zone[int(H * 0.04):int(H * 0.95), :int(W * 0.63)] = 1
zone = cv2.GaussianBlur(zone, (0, 0), 40)

# 1) per-frame camera estimates, 2) smooth them over time so the patch can't wobble
raw, keep, tu, dups = [], [], [], []
u = 0
for i, f in enumerate(frames):
    raw.append(np.float32([[1, 0, 0], [0, 1, 0]]) if i <= 7 else align_affine(ref, f, feat))
    dup = i > 0 and np.abs(cv2.resize(f, (480, 270)).astype(np.int16) - cv2.resize(frames[i - 1], (480, 270))).mean() < 0.3
    u += 0 if dup else 1
    tu.append(u)                    # camera time only advances on new (non-duplicated) frames
    dups.append(dup)
    keep.append(i >= 7 and not dup)
smooth = smooth_affines(raw, keep, t=tu)

wr = Writer(out)
prev = None
for i, f in enumerate(frames):
    if i <= 7:
        wr.write(f)
        continue
    if dups[i]:                     # a repeated source frame repeats its cleaned frame exactly
        wr.write(prev)
        continue
    A = smooth[i]
    plate = warp(ref, A)
    # match exposure drift of the plate to the current frame on the clean skyline
    g = (f[:, int(W*0.65):].reshape(-1, 3).mean(0) + 1) / (plate[:, int(W*0.65):].reshape(-1, 3).mean(0) + 1)
    plate = np.clip(plate * g, 0, 255).astype(np.uint8)
    d = np.abs(f.astype(np.int16) - plate.astype(np.int16)).max(2).astype(np.uint8)
    m = (cv2.GaussianBlur(d, (0, 0), 3) > 14).astype(np.uint8) * 255
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    m = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 12) * zone
    m = m[..., None]
    prev = (f * (1 - m) + plate * m).astype(np.uint8)
    wr.write(prev)
    if i in (20, 40, 57):
        cv2.imwrite(out + f'.{i}.jpg', cv2.resize(np.hstack([f, (f*(1-m)+plate*m).astype(np.uint8)]), (1920, 540)))
wr.close()
