"""Assemble the cleaned shots into the retimed 4K base plate (no graphics).

Edit decisions (output seconds):
  0.00 - 4.15  City aerial   (cleaned src 0-57, slowed ~0.48x)       -> show title
  3.85 - 8.15  Stage         (cleaned src 190-232, slowed, slow push) -> tagline
  7.85 - END   Studio desk   (cleaned src 352-392 at ~0.45x, then a steady hold with a slow push)
                                                                     -> host lower-third, episode card
Overlaps are 0.3 s cross-dissolves. 24p content inside the 30p file is de-duplicated first and
in-between frames are rebuilt with optical flow, so slow sections stay smooth.
"""
import sys
import json
from common import *

work, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
END = 15.4


def unique(frames):
    keep = [frames[0]]
    for f in frames[1:]:
        if np.abs(cv2.resize(f, (480, 270)).astype(np.int16) - cv2.resize(keep[-1], (480, 270))).mean() > 0.35:
            keep.append(f)
    return keep


def load(path, a=None, b=None):
    cap = cv2.VideoCapture(path)
    fr, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if (a is None or i >= a) and (b is None or i <= b):
            fr.append(f)
        i += 1
    return fr


city = unique(load(f"{work}/clean_city.mkv"))
stage = unique(load(f"{work}/clean_stage.mkv"))
front = unique(load(f"{work}/clean_studio.mkv", 7, 47))
print("unique frames", len(city), len(stage), len(front), flush=True)


def ease_front(u, play=0.42):
    """Play the desk shot at ~0.45x (light trails included), decelerate smoothly, then hold the
    last frame. A true hold can't wobble the way ultra-slow optical-flow frames can."""
    x = min(u / play, 1.0)
    return 1 - (1 - x) ** 2


SHOTS = [
    # frames, start, end, position(u) in [0,1], push-in (scale_from, scale_to, cx, cy)
    (city, 0.00, 4.15, lambda u: u, None),
    (stage, 3.85, 8.15, lambda u: u, (1.0, 1.04, 0.55, 0.40)),
    # stage dissolves straight into Brian at his podcast desk (office/notebook shot not used)
    (front, 7.85, END, ease_front, (1.0, 1.035, 0.45, 0.40)),
]


def shot_frame(shot, t):
    frames, a, b, pos, push = shot
    u = min(max((t - a) / (b - a), 0.0), 1.0)
    p = pos(u) * (len(frames) - 1)
    i = int(np.floor(p))
    j = min(i + 1, len(frames) - 1)
    f = interp(frames[i], frames[j], p - i)
    if push:
        s = push[0] + (push[1] - push[0]) * (u * u * (3 - 2 * u))
        cx, cy = push[2] * W, push[3] * H
        M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
        f = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return f


wr = Writer(out)
n = int(round(END * FPS))
for k in range(n):
    t = k / FPS
    act = [s for s in SHOTS if s[1] - 1e-6 <= t <= s[2] + 1e-6]
    if len(act) == 1:
        f = shot_frame(act[0], t)
    else:
        s0, s1 = act[0], act[1]
        w = (t - s1[1]) / (s0[2] - s1[1])
        w = w * w * (3 - 2 * w)
        f = cv2.addWeighted(shot_frame(s0, t), 1 - w, shot_frame(s1, t), w, 0)
    wr.write(f)
    if k % 30 == 0:
        print("base", k, "/", n, flush=True)
wr.close()
json.dump({"duration": END, "fps": FPS, "frames": n}, open(out + ".json", "w"))
