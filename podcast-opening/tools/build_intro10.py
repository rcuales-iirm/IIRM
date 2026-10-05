"""Base plate for the 10-second recurring intro (no graphics).

  0.00 - 3.15  City aerial (cleaned), ~0.6x                    -> show title
  2.85 - 5.00  Stage (cleaned backdrop), slow push              -> tagline + gold line
  5.00 - 10.0  Hard cut to Brian seated at his podcast desk     -> "Build and Cover Your ASSets / With Brian Woods"

The desk shot uses only the settled, trail-free frames (src 383-392) with the notebook removed
(clean_desk.py). Those ~0.3 s of real motion are played forward and back on a slow, eased
breathing-length cycle, so he stays relaxed and natural while the camera pushes in gently.

usage: python3 build_intro10.py WORK_DIR OUT.mkv
"""
import json
import sys

from common import *

work, out = sys.argv[1], sys.argv[2]
END = 10.0
CUT = 5.0


def load(path, a=0):
    cap = cv2.VideoCapture(path)
    fr, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i >= a:
            fr.append(f)
        i += 1
    return fr


def unique(frames):
    keep = [frames[0]]
    for f in frames[1:]:
        if np.abs(cv2.resize(f, (480, 270)).astype(np.int16) - cv2.resize(keep[-1], (480, 270))).mean() > 0.35:
            keep.append(f)
    return keep


city = unique(load(f"{work}/clean_city.mkv"))
stage = unique(load(f"{work}/clean_stage.mkv"))
desk = unique(load(f"{work}/clean_desk.mkv", 2))   # clean_desk starts at studio frame 36; skip the last spark
print("unique frames", len(city), len(stage), len(desk), flush=True)


def at(frames, p):
    p = min(max(p, 0.0), len(frames) - 1.0)
    i = int(np.floor(p))
    j = min(i + 1, len(frames) - 1)
    return interp(frames[i], frames[j], p - i)


def push(f, s, cx, cy):
    M = np.float32([[s, 0, cx * W - s * cx * W], [0, s, cy * H - s * cy * H]])
    return cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def city_frame(t):
    return at(city, t / 3.15 * (len(city) - 1))


def stage_frame(t):
    u = (t - 2.85) / (CUT - 2.85)
    return push(at(stage, u * (len(stage) - 1)), 1.0 + 0.03 * smooth(u), 0.55, 0.40)


BREATH = 3.4  # seconds per forward-and-back cycle


def desk_frame(t):
    tt = t - CUT
    p = (len(desk) - 1) * (0.5 - 0.5 * np.cos(2 * np.pi * tt / BREATH))
    u = tt / (END - CUT)
    # gentle push toward Brian, keeping the wall to his right free for the title
    return push(at(desk, p), 1.0 + 0.05 * smooth(u), 0.40, 0.42)


wr = Writer(out)
n = int(round(END * FPS))
for k in range(n):
    t = k / FPS
    if t < 2.85:
        f = city_frame(t)
    elif t < 3.15:
        w = smooth((t - 2.85) / 0.3)
        f = cv2.addWeighted(city_frame(t), 1 - w, stage_frame(t), w, 0)
    elif t < CUT:
        f = stage_frame(t)
    else:
        f = desk_frame(t)
    wr.write(f)
    if k % 30 == 0:
        print("intro10", k, "/", n, flush=True)
wr.close()
json.dump({"duration": END, "fps": FPS, "frames": n}, open(out + ".json", "w"))
