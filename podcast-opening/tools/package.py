"""Render the reusable episode package as transparent QuickTime Animation (alpha) clips (plus an opaque outro).

  python3 package.py OUT_DIR [--size 1920x1080] [--side left|right]

Text comes from config/brand.json (host) and config/episode.json (guest, topics, episode).
Clips:  lower_third_host.mov, lower_third_guest.mov, topic_heading_<n>.mov,
        outro_overlay.mov (alpha, sits over footage), outro_card.mp4 (navy card with sound).
"""
import argparse
import os
import subprocess

import numpy as np

from graphics import GFX, load_json

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--size", default="1920x1080")
ap.add_argument("--side", default="left", choices=["left", "right"])
args = ap.parse_args()
W, H = map(int, args.size.split("x"))
os.makedirs(args.out, exist_ok=True)
brand, ep = load_json("brand.json"), load_json("episode.json")
g = GFX(W, H, brand, ep)
FPS = 30
# design-space anchors (3840x2160): lower-thirds sit inside the 8% title-safe margin
LT_X = 307 if args.side == "left" else 2300
LT_Y = 1560


def write(name, seconds, draw, opaque=False):
    path = os.path.join(args.out, name)
    if opaque:
        cmd = ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    else:
        # QuickTime Animation: lossless 8-bit alpha, opens in Premiere, Resolve, Final Cut and After Effects
        cmd = ["-c:v", "qtrle", "-pix_fmt", "argb"]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", *cmd, path], stdin=subprocess.PIPE)
    for k in range(int(seconds * FPS)):
        L = draw(k / FPS)
        if opaque:
            bg = np.zeros((H, W, 4), np.uint8)
            bg[..., :3] = g.C["navy"]
            a = np.asarray(L)[..., 3:4] / 255.0
            bg[..., :3] = (bg[..., :3] * (1 - a) + np.asarray(L)[..., :3] * a).astype(np.uint8)
            bg[..., 3] = 255
            p.stdin.write(bg.tobytes())
        else:
            p.stdin.write(np.asarray(L).tobytes())
    p.stdin.close()
    p.wait()
    print("wrote", path, flush=True)
    return path


def blank():
    from PIL import Image
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def lt(name, label, role):
    def draw(t):
        L = blank()
        g.scrim(L, "ellipse", min(1, t / 0.5) * (1 - max(0, min(1, (t - 5.1) / 0.4))),
                cx=(LT_X + 650) / 3840, cy=(LT_Y + 140) / 2160, rx=0.26, ry=0.15, strength=0.35)
        g.lower_third(L, t, 0.3, 5.0, name, label, role, LT_X, LT_Y)
        return L
    return draw


h, gu = brand["host"], ep["guest"]
write("lower_third_host.mov", 6, lt(h["name"], h["role_label"], h["role"]))
write("lower_third_guest.mov", 6, lt(gu["name"].upper(), gu["role_label"], gu["role"]))
for i, topic in enumerate(ep["topics"], 1):
    def draw(t, topic=topic):
        L = blank()
        g.topic_heading(L, t, 0.3, 4.0, topic)
        return L
    write(f"topic_heading_{i}.mov", 5, draw)
write("outro_overlay.mov", 6, lambda t: g.outro(t, background=False))
card = write("outro_card_silent.mp4", 6, lambda t: g.outro(t, background=False), opaque=True)

# outro card with a soft swell + the same single chime as the opening's microphone pulse
import wave  # noqa: E402
SR = 48000
n = 6 * SR
t = np.arange(n) / SR
swell = (np.sin(2 * np.pi * 174.61 * t) + 0.5 * np.sin(2 * np.pi * 261.63 * t) + 0.35 * np.sin(2 * np.pi * 349.23 * t))
swell *= np.clip(t / 1.2, 0, 1) * np.clip((6 - t) / 1.6, 0, 1) * 0.06
c0 = int(1.6 * SR)
tc = np.arange(n - c0) / SR
chime = np.zeros(n)
chime[c0:] = (np.sin(2 * np.pi * 698.46 * tc) * np.exp(-tc / 0.7) + 0.35 * np.sin(2 * np.pi * 1046.5 * tc)
              * np.exp(-tc / 0.45)) * 0.045 * np.clip(tc / 0.006, 0, 1)
mix = np.stack([swell + chime, swell + np.roll(chime, int(0.012 * SR))], 1)
wav = os.path.join(args.out, "_outro.wav")
with wave.open(wav, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", card, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                "-shortest", os.path.join(args.out, "outro_card.mp4")], check=True)
os.remove(card)
os.remove(wav)
