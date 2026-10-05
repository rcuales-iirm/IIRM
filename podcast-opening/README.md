# Build and Cover Your Assets: podcast opening and episode package

This folder holds a finished opening for Brian Woods's podcast, made from `Intro.mp4`. It also holds the scripts that rebuilt it, so each new episode only needs a new episode number and title.

**Brand:** navy, warm amber, gold and white. The fonts are Montserrat (sans) and Cinzel (serif), both under the SIL Open Font License (see `fonts/OFL-*.txt`). No other font families are used.

## Deliverables

| File | What it is |
|---|---|
| `renders/bcya_intro_general_4k.mp4` | **General intro template:** one file for every episode. Same opening, but it ends on a show sign-off (THE PODCAST / Build and Cover Your ASSets / tagline) instead of an episode card |
| `renders/bcya_intro_general_1080p.mp4` | The general intro at 1920×1080 |
| `renders/bcya_intro_general_graphics_alpha_1080p.mov` | The general intro's graphics on their own, with alpha |
| `renders/bcya_opening_4k.mp4` | **Per-episode version.** 3840×2160, 30 fps, H.264 + AAC, 15.4 s, using the template card `EPISODE [NUMBER]` / `[EPISODE TITLE]` |
| `renders/bcya_opening_1080p.mp4` | The same opening at 1920×1080 |
| `renders/bcya_opening_sample_episode_1080p.mp4` | A filled-in example ("Episode 1 / Protecting What You Build") showing how a real card reads |
| `renders/bcya_clean_plate_4k.mp4` | The cleaned, retimed footage with the soundtrack and **no** graphics, for building the opening in an NLE |
| `renders/bcya_opening_graphics_alpha_1080p.mov` | The opening's graphics on their own, with alpha |
| `renders/package/lower_third_host.mov` | Host lower-third, 6 s, alpha |
| `renders/package/lower_third_guest.mov` | Guest lower-third with placeholder text, alpha |
| `renders/package/topic_heading_1.mov` | Brief topic heading, 5 s, alpha |
| `renders/package/outro_overlay.mov` | "Thanks for listening / Follow Build and Cover Your Assets", for placing over footage, alpha |
| `renders/package/outro_card.mp4` | The same outro on a navy card, with a soft swell and chime |

All alpha files are 1080p QuickTime Animation (`.mov`, lossless 8-bit alpha). Premiere Pro, DaVinci Resolve, Final Cut Pro and After Effects all read them natively. For 4K timelines, re-render the package with `--size 3840x2160` instead of scaling it up.

## Timeline of the opening

| Time (s) | Footage | Graphics |
|---|---|---|
| 0.0 – 4.0 | City aerial, slowed to about 0.5× and cleaned of its old title | A navy scrim fades up on the left. **Build and / Cover Your / ASSets** reveals line by line behind a mask (0.15 s stagger), with "ASSets" in gold. A thin gold underline draws across at 1.15 s. Everything exits upward by 3.9 s |
| 4.0 – 8.0 | Stage, with the backdrop sign removed and a slow 4 % push-in | A microphone ring appears at 4.2 s. **BUILD WEALTH.**, **PROTECT ASSETS.** and **ACHIEVE SUCCESS.** rise in at 4.45, 5.05 and 5.65 s. The microphone pulses once at 6.45 s (one ring, no flashing). Everything clears by 7.95 s |
| 8.0 – 9.7 | Brian in the office, where the original light trails begin | — |
| 9.6 – 12.0 | Brian at the studio desk (a two-frame cut from the office, since a long dissolve between two angles of Brian would double-expose him), with the burned-in name removed and the light trails kept | Lower-third on the empty wall to Brian's right, clear of his face, hands and microphone. A gold line draws, then **BRIAN WOODS**, then **Host \| Real Estate, Insurance & Risk Management**. The name stays readable from about 10.2 s to 11.9 s |
| 12.0 – 15.4 | The same studio shot, easing into a near-hold | **General intro:** "THE PODCAST", a gold line, the title echoed from the opening ("ASSets" in gold) and the tagline in small caps. **Per-episode version:** episode card: **EPISODE [NUMBER]**, a gold line, then the title in Cinzel (up to 3 lines, wrapped and shrunk to fit automatically). It is fully readable from 12.7 s to 14.75 s and then exits. This is where the opening hands off to the episode footage |

The episode section was extended from the requested 2 s to about 3.4 s so the title can be read. The other section boundaries are 0.3 s cross-dissolves.

### Audio

- `Intro.mp4` has **no speech**. Its soundtrack is an instrumental bed in F / B-flat. That bed is kept as the theme and stretched 15 % (pitch preserved) to fit the new edit.
- Four soft filtered "whoosh" transitions, one chime in F for the microphone pulse, and a low swell at the hand-off are all synthesised in `tools/audio.py`.
- The music fades out between **14.35 s and 15.35 s**, so it is gone before Brian's first word in the episode. The opening measures −19.6 LUFS integrated with a −4.7 dBFS peak. Normalise to your platform target (for example −16 LUFS for podcasts and YouTube) in the final episode mix.

## Changes made to the original footage

`Intro.mp4` already had titles burned into almost every shot. The rule was to "replace or cleanly transition, never cover", so:

- **City shot:** the old "Build and Cover Your ASSets" was removed. The pre-title frame was tracked onto every later frame (`tools/clean_city.py`). The tracking path is smoothed over time, and repeated 24p frames are kept identical. Without this the patched area visibly shook against the skyline: per-frame estimates wobbled by up to ~12 px.
- **Stage shot:** the "BUILD WEALTH. PROTECT ASSETS. ACHIEVE SUCCESS" sign and the PODCAST badge are part of the backdrop and have no clean frame. The backdrop was rebuilt with a harmonic fill (`tools/clean_stage.py`), and the new tagline replaces the old one in the same place.
- **Studio shot:** the gold "BRIAN WOODS" title was removed. A rebuilt wall plate was tracked separately from the drifting name, and the light trails were added back (`tools/clean_studio.py`). Where a trail crosses the old letters it has been reconstructed, so it can look slightly smeared in a 4K freeze-frame. At speed it reads as normal motion blur.
- **Dropped shots:** the "Turning Insights Into Opportunities" aerial and the "WELCOME" blueprint shot are not used. The brief assigns 0–4 s to the city and 4–8 s to the stage, and both dropped shots had burned-in titles that would have competed with the new ones.
- Brian's appearance, the studio and the microphone placement are unchanged: no reframing, no beauty work and no added effects on his face. The only motion added is a 2.5 % digital push on the final studio shot.

## Making an episode

Requirements: Python 3.10+, `ffmpeg` with `libx264` and `rubberband`, and `pip install numpy opencv-python-headless pillow scipy`.

```bash
cd podcast-opening/tools          # put Intro.mp4 in podcast-opening/ (not committed)
mkdir -p work out
# one-time: clean shots + retimed base plate + audio (about 20 min at 4K)
python3 clean_city.py   ../Intro.mp4 work/clean_city.mkv
python3 clean_stage.py  ../Intro.mp4 work/clean_stage.mkv
python3 clean_studio.py ../Intro.mp4 work/clean_studio.mkv
python3 build_base.py work ../Intro.mp4 work/base_4k.mkv
python3 audio.py ../Intro.mp4 work/opening_audio.wav 15.4

# general intro (any episode): about 5 min
python3 render.py --base work/base_4k.mkv --audio work/opening_audio.wav --out out/bcya_intro_general --general

# per episode: about 5 min. --episode appends the episode with a 0.5 s dissolve
python3 render.py --base work/base_4k.mkv --audio work/opening_audio.wav --out out/ep12 \
    --number 12 --title "Protecting What You Build" [--episode ep12_footage.mp4] [--overlay]

# reusable package (edit config/episode.json for guest name, role and topics first)
python3 package.py out/package            # 1080p alpha clips, lower-thirds on the left
python3 package.py out/package_4k --size 3840x2160 --side right
```

All on-screen wording lives in `config/brand.json` (show, tagline, host) and `config/episode.json` (episode number and title, guest, topics). Colours are defined once in `brand.json`.

## Compositing by hand in an NLE

To build the opening in Premiere, Resolve or Final Cut instead of with `render.py`:

1. Put `renders/bcya_clean_plate_4k.mp4` on V1. Do not use `Intro.mp4` directly, because its burned-in titles would show under the new ones.
2. Put `bcya_opening_graphics_alpha_1080p.mov` on V2, scaled to 200 % for a 4K timeline, with **Straight** alpha interpretation and blend mode Normal. Both clips start at 00:00:00:00.
3. The clean plate already carries the finished soundtrack (music, transition sounds and the fade-out).
4. Cut to the episode at 15.4 s, or overlap the episode by 0.5 s with a cross dissolve starting at 14.9 s.
5. Change the episode card by re-running `render.py --overlay` with `--number` and `--title`. Do not retype the text in the NLE, so the typography stays consistent.

For lower-thirds and topic headings, drop the package `.mov` files on a track above the episode footage. Their timing is built in: in at 0.3 s, out about 0.7 s before the clip ends. Trim the tail to shorten the hold, never the head.

## Still needs a video editor

- Placing the package clips (lower-thirds, topic headings, outro) at the right moments in each episode.
- Adding each episode's guest name, title and topics to `config/episode.json` and rendering the package for that episode.
- Final loudness normalisation of the full episode mix.
- Optional: replacing the stretched original bed with a licensed or custom theme if a longer or different cue is wanted. `tools/audio.py` takes any music file in place of `Intro.mp4`'s audio with a one-line change.
