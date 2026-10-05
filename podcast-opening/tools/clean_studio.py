"""Remove the burned-in gold 'BRIAN WOODS' from the settled studio shot (src frames 345-392),
keeping the existing light trails.

Frame 392 (name fully formed, no trails) is the reference. Its name is detected by colour,
the wall behind is rebuilt with a vertically-weighted Laplace fill (keeps the panel mouldings
straight), and that plate is tracked onto every frame along a time-smoothed path. Anything brighter than the tracked
reference inside the mask — the light trails and sparks — is added back on top.
"""
import sys
from common import *
from common import laplace_fill, smooth_affines

src, out = sys.argv[1], sys.argv[2]
frames = read_frames(src, 345, 392)
R = frames[-1]
hsv = cv2.cvtColor(R, cv2.COLOR_BGR2HSV)
gold = ((hsv[..., 0] >= 8) & (hsv[..., 0] <= 32) & (hsv[..., 1] > 50) & (hsv[..., 2] > 70)) | (hsv[..., 2] > 200)
zone = np.zeros((H, W), np.uint8)
zone[int(H * 0.30):int(H * 0.78), int(W * 0.553):int(W * 0.94)] = 1
text = (gold & (zone > 0)).astype(np.uint8) * 255
text = cv2.morphologyEx(text, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
hole = cv2.dilate(text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) & (zone * 255)
# the plate is rebuilt over the whole name block so it stays clean wherever the drifting name lands
block = np.zeros((H, W), np.uint8)
ys, xs = np.nonzero(hole)
block[max(ys.min() - 90, 0):ys.max() + 90, max(xs.min() - 40, int(W * 0.553)):min(xs.max() + 120, W - 8)] = 255

half = cv2.resize(R, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
hh = cv2.resize(block, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) // 255
fill = cv2.resize(laplace_fill(half, hh, wx=0.03, wy=1.0).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
rng = np.random.default_rng(2)
plate = np.clip(fill + rng.normal(0, 1.4, (H, W, 1)), 0, 255).astype(np.float32)
soft = cv2.GaussianBlur(hole.astype(np.float32) / 255, (0, 0), 8)
bsoft = cv2.GaussianBlur(cv2.erode(block, np.ones((31, 31), np.uint8)).astype(np.float32) / 255, (0, 0), 10)
# wide, soft composite mask over the rebuilt block so any residual mismatch is a gentle gradient
bwide = cv2.GaussianBlur(cv2.erode(block, np.ones((121, 121), np.uint8)).astype(np.float32) / 255, (0, 0), 30)
plate = R * (1 - bsoft[..., None]) + plate * bsoft[..., None]
cv2.imwrite(out + '.plate.jpg', cv2.resize(plate.astype(np.uint8), (1280, 720)))

# track using the background only (exclude Brian, the desk and the name/trail area)
feat = np.full((H, W), 255, np.uint8)
feat[int(H * 0.15):, int(W * 0.08):int(W * 0.66)] = 0
feat[int(H * 0.25):int(H * 0.95), int(W * 0.5):int(W * 0.97)] = 0
feat[int(H * 0.9):, :] = 0

core = cv2.dilate(text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))).astype(np.float32) / 255
core = cv2.GaussianBlur(core, (0, 0), 3)
q = 4


def ecc(f, region, mask, init=None, gain=1.0):
    """ECC affine fit R -> f inside region (x0, x1, y0, y1), returned in full-frame coordinates."""
    x0, x1, y0, y1 = region
    a = cv2.resize(cv2.cvtColor(R[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), None, fx=1 / q, fy=1 / q).astype(np.float32) * gain
    b = cv2.resize(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), None, fx=1 / q, fy=1 / q).astype(np.float32) * gain
    tm = cv2.resize(mask[y0:y1, x0:x1], (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    M = np.eye(2, 3, dtype=np.float32)
    try:
        _, M = cv2.findTransformECC(a, b, M, cv2.MOTION_AFFINE,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), tm, 5)
    except cv2.error:
        pass
    # back to full-res coordinates of the whole frame
    S1 = np.array([[q, 0, x0], [0, q, y0], [0, 0, 1]], np.float32)
    M3 = np.vstack([M, [0, 0, 1]])
    return (S1 @ M3 @ np.linalg.inv(S1))[:2]


TEXT_REGION = (int(W * 0.5), W, int(H * 0.2), int(H * 0.85))
text_mask = cv2.dilate(text, np.ones((37, 37), np.uint8))
# wall + shelves around the name, minus Brian, the name and where the trails sweep
WALL_REGION = (int(W * 0.45), W, 0, int(H * 0.9))
wall_mask = np.full((H, W), 255, np.uint8)
wall_mask[:, :int(W * 0.6)] = 0
wall_mask[int(H * 0.2):int(H * 0.85), int(W * 0.5):int(W * 0.95)] = 0
wall_mask[int(H * 0.08):, int(W * 0.45):int(W * 0.6)] = 0


def text_motion(f):
    """The name is a screen-space overlay with its own drift/scale: track it separately."""
    return ecc(f, TEXT_REGION, text_mask)


def wall_motion(f):
    return ecc(f, WALL_REGION, wall_mask, gain=3.0)


# estimate wall (A) and name (B) motion for every frame, then smooth both paths over time
I = np.float32([[1, 0, 0], [0, 1, 0]])


def camera_path():
    """R -> frame i for the wall. Measured on the wall's own panel lines just above and below the
    old name (high-pass phase correlation between consecutive frames, so every shift is tiny and
    reliable), then chained back from R. The wall only drifts a few pixels here, so translation
    is enough; a fitted zoom term amplified noise into visible slides."""
    q2 = 2
    bands = [(int(H * 0.03), int(H * 0.19)), (int(H * 0.65), int(H * 0.80))]   # above / below the name
    x0, x1 = int(W * 0.60), int(W * 0.96)

    def hp(f, y0, y1):
        g = cv2.cvtColor(cv2.resize(f[y0:y1, x0:x1], None, fx=1 / q2, fy=1 / q2, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        return g - cv2.GaussianBlur(g, (0, 0), 8)

    P = [[hp(f, *bd) for bd in bands] for f in frames]
    win = [cv2.createHanningWindow((p.shape[1], p.shape[0]), cv2.CV_32F) for p in P[0]]
    d = [np.mean([cv2.phaseCorrelate(P[i][k], P[i + 1][k], window=win[k])[0] for k in range(2)], 0) * q2
         for i in range(len(frames) - 1)]
    out = []
    for i in range(len(frames)):
        t = -np.sum(d[i:], 0) if i < len(frames) - 1 else np.zeros(2)
        out.append(np.float32([[1, 0, t[0]], [0, 1, t[1]]]))
    return out


rawA = camera_path()
rawB, tu, dups = [], [], []
u = 0
for i, f in enumerate(frames):
    last = i == len(frames) - 1
    rawB.append(I if last else text_motion(f))
    dup = i > 0 and np.abs(cv2.resize(f, (480, 270)).astype(np.int16) - cv2.resize(frames[i - 1], (480, 270))).mean() < 0.3
    u += 0 if dup else 1
    tu.append(u)
    dups.append(dup)
keep = [not d for d in dups]
keep[0] = False  # first frames sit inside the original whip-pan; they are not used in the edit
SA, SB = smooth_affines(rawA, keep, t=tu, sigma=1.5), smooth_affines(rawB, keep, t=tu, sigma=2.0)

wr = Writer(out)
prev = None
for i, f in enumerate(frames):
    if dups[i] and prev is not None:
        wr.write(prev)
        continue
    A, B = SA[i], SB[i]
    Pt = warp(plate.astype(np.float32), A)
    Tt = warp(R.astype(np.float32), B)
    letters = warp(soft, B)
    m = np.maximum(letters, warp(bwide, A))[..., None]
    c = warp(core, B)[..., None]
    ff = f.astype(np.float32)
    # low-frequency exposure/shift match: plate follows the live wall wherever the name isn't
    valid = (1 - np.clip(cv2.dilate(letters, np.ones((41, 41), np.uint8)) * 1.5, 0, 1))
    vs = cv2.resize(valid, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    ds = cv2.resize(ff - Pt, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    num = cv2.GaussianBlur(ds * vs[..., None], (0, 0), 12)
    den = cv2.GaussianBlur(vs, (0, 0), 12)[..., None] + 1e-4
    Pc = Pt + cv2.resize(num / den, (W, H), interpolation=cv2.INTER_LINEAR)
    # light trails/sparks = whatever is brighter than the tracked name; never inside the letters
    raw = np.clip((ff - np.maximum(Tt, Pc) - 14) * 1.15, 0, 255)  # drop the faint halo of the old name
    # inside the old letters the difference is unreliable (glints), so continue the (mostly horizontal) trails across them
    keep = (1 - c)
    num = cv2.GaussianBlur(raw * keep, (0, 0), sigmaX=22, sigmaY=3)
    den = cv2.GaussianBlur(keep, (0, 0), sigmaX=22, sigmaY=3)[..., None] + 1e-4
    trails = raw * keep + np.clip(num / den, 0, 255) * c
    o = ff * (1 - m) + (Pc + trails) * m
    prev = np.clip(o, 0, 255).astype(np.uint8)
    wr.write(prev)
    if i in (0, 12, 24, 46):
        cv2.imwrite(out + f'.{i}.jpg', cv2.resize(np.hstack([f, np.clip(o, 0, 255).astype(np.uint8)]), (1920, 540)))
wr.close()
