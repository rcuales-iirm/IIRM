"""Branded "Thank you / Subscribe" end screen (replaces the red Canva template).

  python3 endscreen.py WORK_DIR portrait.png OUT_PREFIX [Intro.mp4]

Navy-graded city footage behind Brian's circular portrait (gold ring draws on), "Thank you /
for listening to Build and Cover Your ASSets", then a large subscribe card: a cursor clicks
SUBSCRIBE (-> SUBSCRIBED) and the bell (one gentle swing). Same fonts, colours and motion as the
intro. Writes OUT_PREFIX_4k.mp4 and OUT_PREFIX_1080p.mp4 with music and soft UI sounds.
"""
import json
import subprocess
import sys
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common import W, H, FPS, Writer, interp
from graphics import GFX, prog, out_cubic, in_out_cubic, out_quint, clamp

work, portrait_path, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
music_src = sys.argv[4] if len(sys.argv) > 4 else None
END = 9.4
g = GFX(W, H)
C = g.C
P = g.p

# ---------------------------------------------------------------- background: navy-graded city
cap = cv2.VideoCapture(f"{work}/clean_city.mkv")
city = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    if not city or np.abs(cv2.resize(f, (480, 270)).astype(np.int16) - cv2.resize(city[-1], (480, 270))).mean() > 0.35:
        city.append(cv2.resize(f, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
navy = np.array(C["navy"][::-1], np.float32)
yy, xx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
vign = np.clip(1 - 0.55 * (((xx - W / 4) / (W / 4)) ** 2 + ((yy - H / 4) / (H / 4)) ** 2), 0.25, 1)[..., None]


def background(t):
    p = t / END * (len(city) - 1) * 0.8
    i = int(p)
    f = interp(city[i], city[min(i + 1, len(city) - 1)], p - i).astype(np.float32)
    f = cv2.GaussianBlur(f, (0, 0), 3)
    lum = f.mean(2, keepdims=True)
    f = navy * 0.78 + (0.35 * f + 0.65 * lum) * 0.30   # navy grade, footage as a quiet texture
    f = f * vign
    s = 1.0 + 0.03 * t / END
    M = np.float32([[s, 0, W / 4 * (1 - s)], [0, s, H / 4 * (1 - s)]])
    f = cv2.warpAffine(f, M, (W // 2, H // 2), borderMode=cv2.BORDER_REFLECT)
    return cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------- portrait in a gold-ringed circle
por = Image.open(portrait_path).convert("RGBA")
geo = json.load(open(portrait_path.rsplit(".", 1)[0] + ".json"))
R = 560                                   # on-screen radius (design px)
sc = R / geo["r"]
por = por.resize((int(por.width * sc), int(por.height * sc)), Image.LANCZOS)
pcx, pcy = geo["cx"] * sc, geo["cy"] * sc
D = 2 * R
disc = Image.new("RGBA", (D, D), (0, 0, 0, 0))
backing = Image.new("RGBA", (D, D), C["navy"] + (255,))
grad = Image.new("L", (D, D), 0)
ImageDraw.Draw(grad).ellipse((0, 0, D, D), fill=255)
light = Image.new("RGBA", (D, D), (30, 52, 92, 255))           # slightly lifted navy behind him
backing = Image.composite(light, backing, grad.filter(ImageFilter.GaussianBlur(D // 5)))
disc.alpha_composite(backing)
disc.alpha_composite(por, (int(R - pcx), int(R - pcy)))
mask = Image.new("L", (D * 4, D * 4), 0)
ImageDraw.Draw(mask).ellipse((0, 0, D * 4 - 1, D * 4 - 1), fill=255)
disc.putalpha(Image.fromarray(np.minimum(np.asarray(disc.getchannel("A")),
                                         np.asarray(mask.resize((D, D), Image.LANCZOS)))))
AVX, AVY = 1240, 1080


def ring(layer, t):
    p = in_out_cubic(prog(t, 0.55, 0.9))
    if p <= 0:
        return
    rr = P(R + 26)
    big = Image.new("RGBA", (2 * rr + 40, 2 * rr + 40), (0, 0, 0, 0))
    ImageDraw.Draw(big).arc((20, 20, 20 + 2 * rr, 20 + 2 * rr), -90, -90 + 360 * p, fill=C["gold"] + (255,),
                            width=P(9))
    g.put(layer, big, P(AVX) - rr - 20, P(AVY) - rr - 20)


# ---------------------------------------------------------------- subscribe card
CX0, CY0, CW, CH = 2020, 1300, 1520, 280
BTN = (CX0 + 860, CY0 + 85, 430, 110)       # x, y, w, h
BELL = (CX0 + 1385, CY0 + 140)             # centre


def rounded(w, h, r, fill, outline=None, width=0):
    im = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w * 2 - 1, h * 2 - 1), radius=r * 2, fill=fill, outline=outline,
                                         width=width * 2)
    return im.resize((w, h), Image.LANCZOS)


def bell_icon(size, filled, angle):
    S = size * 4
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    gold = C["gold"] + (255,)
    lw = S // 14
    body = [(S * .5, S * .14), (S * .72, S * .3), (S * .76, S * .62), (S * .86, S * .74), (S * .14, S * .74),
            (S * .24, S * .62), (S * .28, S * .3)]
    if filled:
        d.polygon(body, fill=gold)
    else:
        d.line(body + [body[0]], fill=gold, width=lw, joint="curve")
    d.ellipse((S * .42, S * .76, S * .58, S * .9), fill=gold)
    d.ellipse((S * .45, S * .07, S * .55, S * .17), fill=gold)
    im = im.rotate(angle, resample=Image.BICUBIC, center=(S * .5, S * .12))
    return im.resize((size, size), Image.LANCZOS)


def cursor_icon(size):
    S = size * 4
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    pts = [(0, 0), (0, S * .78), (S * .2, S * .6), (S * .34, S * .92), (S * .46, S * .86), (S * .32, S * .55),
           (S * .58, S * .55)]
    sh = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(sh).polygon([(x + S * .04, y + S * .05) for x, y in pts], fill=(0, 0, 0, 110))
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(S // 40)))
    ImageDraw.Draw(im).polygon(pts, fill=(255, 255, 255, 255), outline=C["navy_deep"] + (255,), width=S // 28)
    return im.resize((size, size), Image.LANCZOS)


CURSOR = cursor_icon(P(70))
T_CLICK1, T_CLICK2 = 4.45, 5.75


def ease_path(t, pts):
    """pts: [(time, x, y)], eased between keys."""
    if t <= pts[0][0]:
        return pts[0][1:]
    for (t0, x0, y0), (t1, x1, y1) in zip(pts, pts[1:]):
        if t <= t1:
            u = in_out_cubic((t - t0) / (t1 - t0))
            return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    return pts[-1][1:]


def card(layer, t):
    pin = out_cubic(prog(t, 2.35, 0.7))
    pout = in_out_cubic(prog(t, 8.45, 0.45))
    op = pin * (1 - pout)
    if op <= 0:
        return
    dy = P(60) * (1 - pin)
    panel = rounded(P(CW), P(CH), P(36), (17, 37, 70, 235), C["gold"] + (255,), max(1, P(3)))
    g.put(layer, panel, P(CX0), P(CY0) + dy, op)
    g.put_text(layer, [("BRIAN WOODS", "Montserrat", 700, "white")], 60, P(CX0 + 70), P(CY0 + 125) + dy, 0.06,
               op=op * clamp(prog(t, 2.6, 0.4)))
    g.put_text(layer, [("Don\u2019t miss an episode", "Montserrat", 500, "warm_white")], 44, P(CX0 + 70),
               P(CY0 + 200) + dy, 0.02, op=op * clamp(prog(t, 2.75, 0.4)))
    # subscribe button: gold -> "SUBSCRIBED" after the click
    bx, by, bw, bh = BTN
    done = t >= T_CLICK1 + 0.08
    press = 1 - 0.05 * np.sin(np.pi * clamp((t - T_CLICK1) / 0.18))
    if done:
        btn = rounded(P(bw), P(bh), P(bh // 2), C["navy"] + (255,), C["gold"] + (255,), max(1, P(4)))
        label, col = "SUBSCRIBED", "gold"
    else:
        btn = rounded(P(bw), P(bh), P(bh // 2), C["gold"] + (255,))
        label, col = "SUBSCRIBE", "navy"
    if abs(press - 1) > 1e-3:
        btn = btn.resize((int(btn.width * press), int(btn.height * press)), Image.LANCZOS)
    bop = op * clamp(prog(t, 2.85, 0.4))
    bxc, byc = P(bx + bw / 2), P(by + bh / 2) + dy
    g.put(layer, btn, bxc - btn.width / 2, byc - btn.height / 2, bop)
    size = 44 if done else 46
    g.put_text(layer, [(label, "Montserrat", 700, col)], size, bxc, byc + P(size * 0.36), 0.08, op=bop,
               align="center", shadow=0)
    # bell: one damped swing after it's clicked
    filled = t >= T_CLICK2 + 0.06
    sw = 0.0
    if t > T_CLICK2:
        k = t - T_CLICK2
        sw = 14 * np.exp(-3.2 * k) * np.sin(2 * np.pi * 2.2 * k)
    b = bell_icon(P(96), filled, sw)
    g.put(layer, b, P(BELL[0]) - b.width / 2, P(BELL[1]) - b.height / 2 + dy, op * clamp(prog(t, 3.0, 0.4)))
    # soft gold ripple on each click (single ring, no flashing)
    for tc, (cx, cy) in ((T_CLICK1, (bx + bw / 2, by + bh / 2)), (T_CLICK2, BELL)):
        pp = prog(t, tc, 0.6)
        if 0 < pp < 1:
            rr = P(40 + 120 * out_cubic(pp))
            im = Image.new("RGBA", (2 * rr + 8, 2 * rr + 8), (0, 0, 0, 0))
            ImageDraw.Draw(im).ellipse((4, 4, 2 * rr + 4, 2 * rr + 4), outline=C["gold_light"] + (255,),
                                       width=max(2, P(5)))
            g.put(layer, im, P(cx) - rr - 4, P(cy) - rr - 4 + dy, 0.5 * (1 - pp) * op)
    # cursor: glides in, clicks subscribe, then the bell, then leaves
    path = [(3.55, 3700, 2150), (4.35, BTN[0] + BTN[2] * 0.55, BTN[1] + BTN[3] * 0.6),
            (5.05, BTN[0] + BTN[2] * 0.55, BTN[1] + BTN[3] * 0.6), (5.65, BELL[0] + 8, BELL[1] + 18),
            (6.3, BELL[0] + 8, BELL[1] + 18), (7.0, 3720, 2200)]
    if 3.55 < t < 7.0:
        cxp, cyp = ease_path(t, path)
        cop = clamp(prog(t, 3.55, 0.3)) * (1 - clamp(prog(t, 6.6, 0.4)))
        cs = 1.0
        for tc in (T_CLICK1, T_CLICK2):
            cs *= 1 - 0.12 * np.sin(np.pi * clamp((t - tc + 0.05) / 0.2))
        cur = CURSOR if abs(cs - 1) < 1e-3 else CURSOR.resize((int(CURSOR.width * cs), int(CURSOR.height * cs)))
        g.put(layer, cur, P(cxp), P(cyp) + dy, cop)


def overlay(t):
    L = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pout = in_out_cubic(prog(t, 8.45, 0.45))
    # portrait
    pin = out_quint(prog(t, 0.3, 0.9))
    if pin > 0:
        s = 0.94 + 0.06 * pin
        im = disc if s >= 0.999 else disc.resize((int(disc.width * s), int(disc.height * s)), Image.LANCZOS)
        g.put(L, im, P(AVX) - im.width / 2, P(AVY) - im.height / 2, pin * (1 - pout))
    if pout < 1:
        sub = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ring(sub, t)
        g.put(L, sub, 0, 0, 1 - pout)
    # text
    X = P(2020)
    t_out = 8.4
    g.masked_line(L, [("Thank you", "Cinzel", 600, "white")], 210, X, P(800), t, 0.9, 0.85, t_out, 0.45,
                  tracking=0.01)
    g.line_draw(L, X, P(880), 300, 8, t, 1.35, 0.6, t_out + 0.1, 0.35)
    g.masked_line(L, [("for listening to", "Montserrat", 500, "warm_white")], 66, X, P(1010), t, 1.5, 0.7,
                  t_out + 0.05, 0.4, tracking=0.03)
    size = g.fit_runs([("Build and Cover Your ASSets", "Montserrat", 800, "gold")], 96, 1520)
    g.masked_line(L, [("Build and Cover Your ASSets", "Montserrat", 800, "gold")], size, X, P(1150), t, 1.65,
                  0.75, t_out + 0.1, 0.4)
    card(L, t)
    return L


def render():
    v4 = prefix + "_4k.mp4"
    wav = prefix + "_audio.wav"
    make_audio(wav)
    wr = Writer(v4, codec=("-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow",
                           "-crf", "15", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k", "-shortest",
                           "-movflags", "+faststart"))
    n = int(round(END * FPS))
    for k in range(n):
        t = k / FPS
        bg = background(t)
        bg_op = out_cubic(prog(t, 0.0, 0.5)) * (1 - in_out_cubic(prog(t, 8.75, 0.55)))
        f = navy * (1 - bg_op) + bg * bg_op if bg_op < 1 else bg
        L = np.asarray(overlay(t)).astype(np.float32) / 255
        a = L[..., 3:4]
        f = f * (1 - a) + L[..., 2::-1] * 255 * a
        wr.write(np.clip(f + 0.5, 0, 255).astype(np.uint8))
        if k % 30 == 0:
            print("end screen", k, "/", n, flush=True)
    wr.close()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v4, "-vf", "scale=1920:1080:flags=lanczos", "-c:v",
                    "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "copy",
                    "-movflags", "+faststart", prefix + "_1080p.mp4"], check=True)


def make_audio(path):
    SR = 48000
    n = int(END * SR)
    mix = np.zeros((n, 2))
    if music_src:   # the intro's own instrumental bed, its closing bars, faded in and out
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "3.6", "-i", music_src, "-vn", "-ac", "2", "-ar",
                              str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
        mus = np.frombuffer(raw, np.float32).reshape(-1, 2)[:n].astype(np.float64)
        tt = np.arange(len(mus)) / SR
        mus *= (np.clip(tt / 0.6, 0, 1) * np.clip((END - 0.1 - tt) / 1.4, 0, 1) ** 1.5)[:, None]
        mix[:len(mus)] += mus * 0.9
    t = np.arange(n) / SR

    def add(t0, sig, db):
        i = int(t0 * SR)
        j = min(n, i + len(sig))
        mix[i:j] += (sig[:j - i] * 10 ** (db / 20))[:, None]

    def tone(freqs, dur, decay):
        tt = np.arange(int(dur * SR)) / SR
        s = sum(a * np.sin(2 * np.pi * fr * tt) for fr, a in freqs) * np.exp(-tt / decay)
        return s * np.clip(tt / 0.004, 0, 1) / max(1e-9, np.abs(s).max())

    rng = np.random.default_rng(3)

    def click():
        tt = np.arange(int(0.06 * SR)) / SR
        s = rng.standard_normal(len(tt)) * np.exp(-tt / 0.006) * 0.6 + np.sin(2 * np.pi * 180 * tt) * np.exp(
            -tt / 0.02)
        return s / np.abs(s).max()

    add(0.9, tone([(698.46, 1), (1046.5, .35)], 2.0, 0.7), -27)          # "Thank you" (F, as in the intro)
    add(T_CLICK1, click(), -30)
    add(T_CLICK1 + 0.02, tone([(1046.5, 1), (1396.9, .4)], 0.8, 0.25), -32)
    add(T_CLICK2, click(), -30)
    add(T_CLICK2 + 0.03, tone([(1396.9, 1), (2093, .3)], 0.9, 0.3), -33)  # bell: high, short, soft
    peak = np.abs(mix).max()
    if peak > 0.89:
        mix *= 0.89 / peak
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    if len(sys.argv) > 5 and sys.argv[5] == "--preview":
        for tt in (1.2, 3.2, 5.0, 6.4):
            bg = background(tt)
            L = np.asarray(overlay(tt)).astype(np.float32) / 255
            a = L[..., 3:4]
            f = bg * (1 - a) + L[..., 2::-1] * 255 * a
            cv2.imwrite(f"{prefix}_preview_{tt}.jpg", cv2.resize(np.clip(f, 0, 255).astype(np.uint8), (1280, 720)))
    else:
        render()
