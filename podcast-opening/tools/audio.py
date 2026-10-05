"""Audio for the opening: the original instrumental bed (no speech in Intro.mp4), gently
time-stretched to the new edit, plus a few subtle synthesised transition sounds.

usage: python3 audio.py Intro.mp4 out.wav [duration] [opening|intro10]
"""
import subprocess
import sys
import wave

import numpy as np

SR = 48000
src, out = sys.argv[1], sys.argv[2]
DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 15.4
PRESET = sys.argv[4] if len(sys.argv) > 4 else "opening"

# Where the music has fully faded, so the episode's first spoken words are clean.
MUSIC_FADE = (14.35, 15.35)
STRETCH = True
# Transition sounds: (time, kind, gain dB)
CUES = [
    (3.62, "whoosh", -24),   # title out -> stage
    (6.45, "chime", -27),    # single microphone pulse
    (7.62, "whoosh", -25),   # stage -> Brian at the desk
    (11.80, "whoosh", -28),  # lower-third -> episode card
    (14.60, "swell", -26),   # hand-off into the episode
]
if PRESET == "intro10":
    # the bed at its natural tempo, faded well before the episode's first words
    MUSIC_FADE = (8.7, 9.8)
    STRETCH = False
    CUES = [
        (2.62, "whoosh", -25),   # title -> tagline
        (4.70, "whoosh", -26),   # cut to Brian
        (5.95, "chime", -28),    # "With Brian Woods"
        (9.15, "swell", -27),    # hand-off into the episode
    ]


def load_music():
    # 13.1 s bed stretched (pitch preserved) so its natural ending lands under the episode card
    tempo = 13.1 / (MUSIC_FADE[1] - 0.15) if STRETCH else 1.0
    af = f"rubberband=tempo={tempo:.4f}" if STRETCH else "anull"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-vn", "-af", af,
                          "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def env(n, attack, release):
    t = np.arange(n) / SR
    a = np.clip(t / attack, 0, 1)
    return a * np.exp(-np.maximum(t - attack, 0) / release)


def onepole_lp(x, fc):
    """Time-varying one-pole low-pass (fc array in Hz)."""
    y = np.zeros_like(x)
    a = np.exp(-2 * np.pi * fc / SR)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a[i]) * x[i] + a[i] * acc
        y[i] = acc
    return y


def whoosh(rng, length=0.9):
    n = int(length * SR)
    t = np.linspace(0, 1, n)
    shape = np.sin(np.pi * t ** 0.8) ** 2               # swell in, ease out
    fc = 250 + 2000 * np.sin(np.pi * t) ** 1.5           # filter opens then closes
    out = []
    for ch in range(2):
        noise = rng.standard_normal(n)
        lp = onepole_lp(onepole_lp(onepole_lp(noise, fc), fc), fc)  # 18 dB/oct: airy, not hissy
        bp = lp - onepole_lp(lp, np.full(n, 180.0))      # remove rumble
        out.append(bp * shape)
    s = np.stack(out, 1)
    pan = np.linspace(-0.35, 0.35, n)[:, None]           # slight left -> right drift
    s *= np.hstack([1 - pan[:, :1], 1 + pan[:, :1]]) * 0.5 + 0.5
    return s / (np.abs(s).max() + 1e-9)


def chime():
    # soft bell in F (the bed sits around F / B-flat): F5 with a quiet C6 and an inharmonic shimmer
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    tone = (np.sin(2 * np.pi * 698.46 * t) * env(n, 0.006, 0.7)
            + 0.35 * np.sin(2 * np.pi * 1046.5 * t) * env(n, 0.006, 0.45)
            + 0.12 * np.sin(2 * np.pi * 698.46 * 2.76 * t) * env(n, 0.004, 0.18))
    s = np.stack([tone, np.roll(tone, int(0.012 * SR))], 1)  # tiny stereo widening
    return s / np.abs(s).max()


def swell():
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    e = np.sin(np.pi * np.clip(t / 1.0, 0, 1)) ** 2
    tone = (np.sin(2 * np.pi * 174.61 * t) + 0.5 * np.sin(2 * np.pi * 349.23 * t)
            + 0.25 * np.sin(2 * np.pi * 523.25 * t)) * e
    s = np.stack([tone, tone], 1)
    return s / np.abs(s).max()


def main():
    n = int(DUR * SR)
    mix = np.zeros((n, 2), np.float64)
    music = load_music()[:n]
    m = np.ones(len(music))
    tt = np.arange(len(music)) / SR
    a, b = MUSIC_FADE
    m *= np.clip((b - tt) / (b - a), 0, 1) ** 1.5
    mix[:len(music)] += music * m[:, None]

    rng = np.random.default_rng(7)
    for t0, kind, db in CUES:
        s = {"whoosh": lambda: whoosh(rng), "chime": chime, "swell": swell}[kind]()
        s = s * 10 ** (db / 20)
        i = int(t0 * SR)
        j = min(n, i + len(s))
        mix[i:j] += s[:j - i]

    peak = np.abs(mix).max()
    if peak > 0.89:  # keep true-peak headroom
        mix *= 0.89 / peak
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


main()
