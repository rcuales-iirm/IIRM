"""Remove the backdrop sign ('BUILD WEALTH. PROTECT ASSETS...' + PODCAST badge) from the stage shot.

The stage shot (src frames 190-232) is a locked-off camera, so one clean plate is built from the
temporal median, the sign area is rebuilt with a harmonic (Laplace) fill from the
surrounding backdrop, and that plate is composited into every frame with a feathered mask.
"""
import sys
from common import *
from common import laplace_fill

S = 3.0  # polygons below are in 1280x720 preview coordinates
# generous search region around the sign; podium microphones (x>862, y>288) are excluded
REGION = [(440, 120), (1000, 50), (1010, 250), (870, 256), (870, 302), (852, 340), (822, 348), (560, 390), (450, 320)]
# PODCAST pill + microphone badge, always removed in full
BADGE = [(584, 296), (776, 258), (868, 254), (868, 302), (850, 338), (820, 346), (592, 370)]


def poly_mask(polys):
    m = np.zeros((H, W), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.int32(np.array(p) * S)], 255)
    return m


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    frames = read_frames(src, 190, 232)
    med = np.median(np.stack(frames[::3]), axis=0).astype(np.uint8)
    region = poly_mask([REGION])
    g = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), 30))
    hole = ((hp > 9) & (region > 0)).astype(np.uint8) * 255
    hole = cv2.dilate(hole, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    hole = cv2.morphologyEx(hole, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91)))
    hole = (hole & region) | poly_mask([BADGE])
    cv2.imwrite(out + '.mask.jpg', cv2.resize(np.maximum(med, hole[..., None] // 2), (1280, 720)))
    small = cv2.resize(med, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    hs = cv2.resize(hole, (W // 4, H // 4), interpolation=cv2.INTER_NEAREST) // 255
    filled = cv2.resize(laplace_fill(small, hs).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    rng = np.random.default_rng(1)
    grain = rng.normal(0, 1.6, (H, W, 1)).astype(np.float32)
    feather = cv2.GaussianBlur(hole.astype(np.float32) / 255, (0, 0), 10)[..., None]
    plate = np.clip(filled + grain, 0, 255)
    cv2.imwrite(out + '.plate.jpg', cv2.resize(plate.astype(np.uint8), (1280, 720)))
    wr = Writer(out)
    for f in frames:
        # carry each frame's low-frequency light flicker into the patched area
        delta = cv2.GaussianBlur(f.astype(np.float32) - med, (0, 0), 60)
        wr.write(np.clip(f * (1 - feather) + (plate + delta) * feather, 0, 255).astype(np.uint8))
    wr.close()
