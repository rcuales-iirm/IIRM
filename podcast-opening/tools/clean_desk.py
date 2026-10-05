"""Remove the open notebook from the desk in the settled desk shot, keeping Brian's hands.

The book area is filled with the desk's own wood grain (a clear strip of desk to the right,
tiled horizontally) and colour-matched to the surrounding desk with a smooth correction field.
usage: python3 clean_desk.py clean_studio.mkv out.mkv first_index
"""
import sys
from common import *

# book outline in 4K frame coordinates (generous); fingers are kept by the skin test below
HAND = np.int32([(1990, 1990), (2450, 1990), (2450, 2060), (2390, 2085), (2300, 2095), (2200, 2100),
                 (2100, 2090), (2030, 2075), (1990, 2050)])
HAND_L = np.int32([(1300, 1990), (1900, 1990), (1900, 2110), (1300, 2110)])  # Brian's other hand
# follows the book's own slanted top edge, so Brian's left hand (above it) is left alone
BOOK = np.int32([(1595, 2096), (1800, 2070), (1960, 2044), (2150, 2028), (2450, 2016), (2620, 2004), (2990, 2004),
                 (2990, 2160), (1595, 2160)])


def book_mask(f):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [BOOK], 255)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    # skin: warm, saturated and much brighter than the brown desk; only searched around the right
    # hand's fingertips (the only part of Brian that overlaps the book)
    hand = np.zeros((H, W), np.uint8)
    cv2.fillPoly(hand, [HAND, HAND_L], 255)
    bgr = f.astype(np.float32) + 1
    br = bgr[..., 0] / bgr[..., 2]            # blue/red: paper ~0.9, skin ~0.7, wood ~0.4
    skin = ((hand > 0) & (hsv[..., 0] < 22) & (hsv[..., 1] > 62) & (hsv[..., 1] < 125) & (hsv[..., 2] > 150)
            & (br > 0.55) & (br < 0.82))
    m[skin] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.dilate(m, np.ones((15, 15), np.uint8))
    sk = cv2.morphologyEx((skin * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # fill only holes fully enclosed by skin (nail highlights); gaps between fingers stay open
    cnts, _ = cv2.findContours(sk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 400]
    sk = np.zeros_like(sk)
    cv2.drawContours(sk, cnts, -1, 255, -1)
    sk = cv2.erode(sk, np.ones((3, 3), np.uint8))  # trim the paper fringe around fingertips
    m[sk > 0] = 0
    return m


def remove_book(f):
    m = book_mask(f)
    # tile a clear strip of desk (right of the book) across the book area, mirrored so grain joins
    tex = f.copy()
    a, b = 3000, 3780
    span = b - a
    for x in range(1450, 3000):
        k = (x - 1450) % (2 * span)
        tex[:, x] = f[:, a + (k if k < span else 2 * span - 1 - k)]
    # match light and colour to the desk around the book only (never to the bright fingers)
    desk = np.zeros((H, W), np.float32)
    desk[2000:, 1300:3400] = 1
    hand = np.zeros((H, W), np.uint8)
    cv2.fillPoly(hand, [HAND, HAND_L], 255)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    wood = (hsv[..., 0] >= 5) & (hsv[..., 0] <= 22) & (hsv[..., 1] > 105) & (hsv[..., 2] > 55) & (hsv[..., 2] < 190)
    desk[(m > 0) | (cv2.dilate(hand, np.ones((41, 41), np.uint8)) > 0) | ~wood] = 0
    diff = f.astype(np.float32) - tex.astype(np.float32)
    num = cv2.GaussianBlur(diff * desk[..., None], (0, 0), 45)
    den = cv2.GaussianBlur(desk, (0, 0), 45)[..., None] + 1e-4
    fill = tex.astype(np.float32) + num / den
    # sharp against the hands, soft toward the untouched desk on the left and right
    outer = np.zeros((H, W), np.uint8)
    cv2.rectangle(outer, (1540, 1950), (3020, H), 255, -1)
    outer = cv2.GaussianBlur(cv2.erode(outer, np.ones((41, 41), np.uint8)).astype(np.float32) / 255, (0, 0), 22)
    mf = (cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.5) * outer)[..., None]
    return np.clip(f * (1 - mf) + fill * mf, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    src, out, first = sys.argv[1], sys.argv[2], int(sys.argv[3])
    cap = cv2.VideoCapture(src)
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i >= first:
            frames.append(f)
        i += 1
    wr = Writer(out)
    for k, f in enumerate(frames):
        o = remove_book(f)
        wr.write(o)
        if k in (0, len(frames) - 1):
            cv2.imwrite(out + f".{k}.jpg", cv2.resize(o[1500:, 1200:3400], (1100, 330)))
    wr.close()
