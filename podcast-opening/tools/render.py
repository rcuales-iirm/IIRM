"""Composite the graphics onto the cleaned base plate and encode the finished opening.

  python3 render.py --base work/base_4k.mkv --audio work/opening_audio.wav --out out/opening \
      [--number 12 --title "Protecting What You Build" | --general] [--episode episode.mp4] [--overlay]

Writes <out>_4k.mp4 and <out>_1080p.mp4 (H.264 + AAC). --general ends on the show lockup
instead of the episode card, so one file serves every episode. --episode dissolves into the episode
footage (its own audio is kept; the music has already faded out). --overlay also writes the
graphics alone as a 1080p QuickTime Animation file with alpha for use in an NLE.
"""
import argparse
import json
import subprocess

import cv2
import numpy as np

from graphics import GFX, load_json

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--audio", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--number")
ap.add_argument("--title")
ap.add_argument("--episode")
ap.add_argument("--overlay", action="store_true")
ap.add_argument("--general", action="store_true", help="general intro: show lockup instead of episode card")
args = ap.parse_args()

meta = json.load(open(args.base + ".json"))
fps, n = meta["fps"], meta["frames"]
ep = load_json("episode.json")
if args.number:
    ep["episode_number"] = args.number
if args.title:
    ep["episode_title"] = args.title

ending = "general" if args.general else "episode"
cap = cv2.VideoCapture(args.base)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
g = GFX(W, H, episode=ep)

open_mp4 = args.out + "_4k.mp4"
enc = subprocess.Popen(
    ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps),
     "-i", "-", "-i", args.audio,
     "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv420p",
     "-profile:v", "high", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
     "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", open_mp4],
    stdin=subprocess.PIPE)
ovl = None
if args.overlay:
    ovl = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "1920x1080", "-r", str(fps),
         "-i", "-", "-c:v", "qtrle", "-pix_fmt", "argb", args.out + "_graphics_alpha_1080p.mov"], stdin=subprocess.PIPE)
    g2 = GFX(1920, 1080, episode=ep)

for k in range(n):
    ok, f = cap.read()
    if not ok:
        break
    t = k / fps
    L = np.asarray(g.opening(t, ending))
    a = L[..., 3:4].astype(np.float32) / 255
    if a.max() > 0:
        rgb = L[..., 2::-1].astype(np.float32)  # RGBA -> BGR
        f = (f.astype(np.float32) * (1 - a) + rgb * a + 0.5).astype(np.uint8)
    enc.stdin.write(f.tobytes())
    if ovl:
        ovl.stdin.write(np.asarray(g2.opening(t, ending)).tobytes())
    if k % 60 == 0:
        print("render", k, "/", n, flush=True)
enc.stdin.close()
enc.wait()
if ovl:
    ovl.stdin.close()
    ovl.wait()

final4k = open_mp4
if args.episode:
    # 0.5 s dissolve into the episode, timed to land after the music has faded.
    dur = n / fps
    final4k = args.out + "_with_episode_4k.mp4"
    has_audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                "-of", "csv=p=0", args.episode], capture_output=True, text=True).stdout.strip()
    ea = "[1:a]aresample=48000," if has_audio else "anullsrc=r=48000:cl=stereo,"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", open_mp4, "-i", args.episode, "-filter_complex",
         f"[1:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
         f"fps={fps},format=yuv420p,setsar=1,settb=1/{fps}[e];[0:v]fps={fps},format=yuv420p,setsar=1,settb=1/{fps}[o];"
         f"[o][e]xfade=transition=fade:duration=0.5:offset={dur - 0.5:.3f}[v];"
         f"{ea}aformat=channel_layouts=stereo[ea];[0:a][ea]acrossfade=d=0.5:c1=tri:c2=tri[a]",
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", final4k], check=True)

subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", final4k, "-vf", "scale=1920:1080:flags=lanczos",
                "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "copy",
                "-movflags", "+faststart", final4k.replace("_4k.mp4", "_1080p.mp4")], check=True)
print("done", final4k)
