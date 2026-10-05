"""Motion-graphics engine for the Build and Cover Your Assets podcast.

Everything is laid out in 3840x2160 design units and scaled, so the same code renders the 4K
opening and the 1080p reusable package. Two font families only: Montserrat (sans) and Cinzel
(serif). Animations are masked slides, fades, short line draws and a single microphone pulse —
no flashing, particles or spinning text.
"""
import json
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "fonts")
DW, DH = 3840, 2160  # design space


def load_json(name):
    with open(os.path.join(ROOT, "config", name)) as f:
        return json.load(f)


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def prog(t, t0, dur):
    return clamp((t - t0) / dur) if dur > 0 else float(t >= t0)


def out_cubic(x):
    return 1 - (1 - x) ** 3


def in_cubic(x):
    return x ** 3


def in_out_cubic(x):
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def out_quint(x):
    return 1 - (1 - x) ** 5


class GFX:
    def __init__(self, width=3840, height=2160, brand=None, episode=None):
        self.w, self.h = width, height
        self.k = width / DW
        self.brand = brand or load_json("brand.json")
        self.ep = episode or load_json("episode.json")
        c = self.brand["colors"]
        self.C = {n: hex_rgb(v) for n, v in c.items()}
        self._grad_cache = {}

    # ------------------------------------------------------------ primitives
    def p(self, v):
        return int(round(v * self.k))

    @lru_cache(maxsize=64)
    def font(self, family, weight, size):
        return ImageFont.truetype(os.path.join(FONT_DIR, f"{family}-{weight}.ttf"), max(4, self.p(size)))

    @lru_cache(maxsize=256)
    def text(self, runs, size, tracking=0.0, shadow=0.55):
        """Render text runs [(string, family, weight, colour_name), ...] on one baseline.

        Returns (RGBA image, baseline_y_in_image, ink_width, left_pad). tracking is in em.
        """
        fonts = [self.font(f, wgt, size) for _, f, wgt, _ in runs]
        trk = tracking * self.p(size)
        asc = max(f.getmetrics()[0] for f in fonts)
        desc = max(f.getmetrics()[1] for f in fonts)
        # measure
        width = 0.0
        for (s, *_), f in zip(runs, fonts):
            if tracking:
                width += sum(f.getlength(ch) + trk for ch in s)
            else:
                width += f.getlength(s)
        if tracking:
            width -= trk
        pad = self.p(60)
        img = Image.new("RGBA", (int(width) + 2 * pad, asc + desc + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        x = pad
        base = pad + asc
        for (s, _, _, col), f in zip(runs, fonts):
            rgb = self.C[col]
            if tracking:
                for ch in s:
                    d.text((x, base), ch, font=f, fill=rgb + (255,), anchor="ls")
                    x += f.getlength(ch) + trk
            else:
                d.text((x, base), s, font=f, fill=rgb + (255,), anchor="ls")
                x += f.getlength(s)
        if shadow:
            a = img.getchannel("A").filter(ImageFilter.GaussianBlur(self.p(14)))
            a = a.point(lambda v: int(v * shadow))
            sh = Image.new("RGBA", img.size, self.C["navy_deep"] + (0,))
            sh.putalpha(a)
            sh = sh.transform(sh.size, Image.AFFINE, (1, 0, 0, 0, 1, -self.p(6)))
            img = Image.alpha_composite(sh, img)
        return img, base, int(width), pad

    def text_width(self, runs, size, tracking=0.0):
        return self.text(runs, size, tracking, 0)[2]

    def gold_bar(self, length, thick):
        """Horizontal gold gradient bar (amber -> gold -> light gold)."""
        key = (length, thick)
        if key not in self._grad_cache:
            L, T = max(1, self.p(length)), max(1, self.p(thick))
            x = np.linspace(0, 1, L)[None, :, None]
            a, g, l = (np.array(self.C[n], np.float32) for n in ("amber", "gold", "gold_light"))
            col = np.where(x < 0.5, a + (g - a) * (x / 0.5), g + (l - g) * ((x - 0.5) / 0.5))
            arr = np.repeat(col, T, axis=0)
            alpha = np.full((T, L, 1), 255, np.float32)
            self._grad_cache[key] = Image.fromarray(np.concatenate([arr, alpha], 2).astype(np.uint8), "RGBA")
        return self._grad_cache[key]

    @staticmethod
    def fade(img, op):
        if op >= 0.999:
            return img
        img = img.copy()
        img.putalpha(img.getchannel("A").point(lambda v: int(v * op)))
        return img

    def put(self, layer, img, x, y, op=1.0, clip=None):
        """Alpha-composite img at integer (x, y) on layer, optionally clipped to (y0, y1) in px."""
        if op <= 0.002:
            return
        x, y = int(round(x)), int(round(y))
        if clip is not None:
            y0, y1 = clip
            top, bot = max(0, y0 - y), min(img.height, y1 - y)
            if bot <= top:
                return
            img = img.crop((0, top, img.width, bot))
            y += top
        img = self.fade(img, op)
        # crop to canvas
        if x >= layer.width or y >= layer.height or x + img.width <= 0 or y + img.height <= 0:
            return
        cx0, cy0 = max(0, -x), max(0, -y)
        img = img.crop((cx0, cy0, min(img.width, layer.width - x), min(img.height, layer.height - y)))
        layer.alpha_composite(img, (x + cx0, y + cy0))

    def put_text(self, layer, runs, size, x, baseline, tracking=0.0, op=1.0, dy=0.0, clip=None,
                 align="left", shadow=0.55):
        img, base, w, pad = self.text(tuple(runs), size, tracking, shadow)
        if align == "center":
            x = x - w / 2
        elif align == "right":
            x = x - w
        self.put(layer, img, x - pad, baseline - base + dy, op, clip)
        return w

    def masked_line(self, layer, runs, size, x, baseline, t, t0, dur=0.7, t_out=None, out_dur=0.45,
                    tracking=0.0, align="left", rise=None, shadow=0.55):
        """A line that slides up into view from behind an invisible mask, and out again upward."""
        f = self.font(runs[0][1], runs[0][2], size)
        asc, desc = f.getmetrics()
        rise = self.p(size * 1.05) if rise is None else rise
        clip = (baseline - asc - self.p(30), baseline + desc + self.p(30))
        pin = out_quint(prog(t, t0, dur))
        dy = (1 - pin) * rise
        op = clamp(pin * 1.6)
        if t_out is not None and t >= t_out:
            po = in_out_cubic(prog(t, t_out, out_dur))
            dy = -po * rise
            op = 1 - clamp(po * 1.2)
        if pin <= 0:
            return
        self.put_text(layer, runs, size, x, baseline, tracking, op, dy, clip, align, shadow)

    def line_draw(self, layer, x, y, length, thick, t, t0, dur=0.6, t_out=None, out_dur=0.4,
                  origin="left", op=1.0):
        """Thin gold line drawing on (and retracting off) from its origin."""
        p = in_out_cubic(prog(t, t0, dur))
        start = 0.0
        if t_out is not None and t >= t_out:
            start = in_out_cubic(prog(t, t_out, out_dur))
        if p <= 0 or start >= 1:
            return
        L = self.p(length)
        bar = self.gold_bar(length, thick)
        a, b = int(L * start), int(L * p)
        if origin == "center":
            a, b = int(L / 2 * (1 - p)), int(L / 2 * (1 + p))
            if start > 0:
                a, b = int(L / 2 * start), int(L - L / 2 * start)
        if b - a < 1:
            return
        self.put(layer, bar.crop((a, 0, b, bar.height)), x + a, y - bar.height // 2, op)

    def scrim(self, layer, kind, op, **kw):
        """Soft navy gradients that keep type readable without boxing it in."""
        if op <= 0.002:
            return
        key = (kind, tuple(sorted(kw.items())))
        if key not in self._grad_cache:
            H, W = self.h, self.w
            if kind == "left":
                x = np.linspace(0, 1, W)[None, :] / kw.get("reach", 0.6)
                a = kw.get("strength", 0.55) * np.clip(1 - x, 0, 1) ** 1.6
                a = np.repeat(a, H, 0)
            else:  # ellipse
                cx, cy, rx, ry = kw["cx"] * W, kw["cy"] * H, kw["rx"] * W, kw["ry"] * H
                yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
                r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
                a = kw.get("strength", 0.4) * np.clip(1 - r, 0, 1) ** 1.3
            rgb = np.zeros((H, W, 3), np.uint8) + np.array(self.C["navy_deep"], np.uint8)
            self._grad_cache[key] = Image.fromarray(np.dstack([rgb, (a * 255).astype(np.uint8)]), "RGBA")
        self.put(layer, self._grad_cache[key], 0, 0, op)

    @lru_cache(maxsize=4)
    def mic_icon(self, size):
        """Gold ring with a studio microphone glyph (drawn 4x and downsampled for clean edges)."""
        S = self.p(size) * 4
        img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        gold = self.C["gold"] + (255,)
        lw = max(4, S // 26)
        d.ellipse((lw, lw, S - lw, S - lw), outline=gold, width=lw)
        cx = S / 2
        cw, ch = S * 0.17, S * 0.34
        top = S * 0.2
        d.rounded_rectangle((cx - cw / 2, top, cx + cw / 2, top + ch), radius=cw / 2, fill=gold)
        aw = S * 0.27
        d.arc((cx - aw / 2, top + ch * 0.35, cx + aw / 2, top + ch * 0.35 + aw * 1.1), 0, 180, fill=gold,
              width=lw)
        stem_top = top + ch * 0.35 + aw * 1.1 - lw
        d.line((cx, stem_top, cx, S * 0.76), fill=gold, width=lw)
        d.line((cx - S * 0.1, S * 0.76, cx + S * 0.1, S * 0.76), fill=gold, width=lw)
        return img.resize((S // 4, S // 4), Image.LANCZOS)

    def mic(self, layer, cx, cy, size, t, t0, pulse_t=None, t_out=None):
        pin = out_cubic(prog(t, t0, 0.5))
        if pin <= 0:
            return
        op = pin
        s = 0.9 + 0.1 * pin
        if pulse_t is not None:
            pp = prog(t, pulse_t, 0.9)
            if 0 < pp < 1:
                s *= 1 + 0.07 * np.sin(np.pi * pp)
                # one soft expanding ring
                R = self.p(size) * (0.5 + 0.55 * out_cubic(pp))
                ring = Image.new("RGBA", (int(2 * R + 8), int(2 * R + 8)), (0, 0, 0, 0))
                ImageDraw.Draw(ring).ellipse((4, 4, 2 * R + 4, 2 * R + 4), outline=self.C["gold_light"] + (255,),
                                             width=max(2, self.p(5)))
                self.put(layer, ring, cx - R - 4, cy - R - 4, 0.55 * (1 - pp) * op)
        if t_out is not None:
            op *= 1 - in_out_cubic(prog(t, t_out, 0.4))
        icon = self.mic_icon(size)
        if abs(s - 1) > 1e-3:
            icon = icon.resize((max(1, int(icon.width * s)), max(1, int(icon.height * s))), Image.LANCZOS)
        self.put(layer, icon, cx - icon.width / 2, cy - icon.height / 2, op)

    # ------------------------------------------------------------ reusable components
    def fit_runs(self, runs, size, max_w, tracking=0.0, min_size=None):
        """Shrink a single line until it fits max_w (design px)."""
        min_size = min_size or size * 0.75
        while size > min_size and self.text_width(tuple(runs), size, tracking) > self.p(max_w):
            size -= 2
        return size

    def wrap(self, text, family, weight, size, max_w):
        words, lines, cur = text.split(), [], ""
        f = self.font(family, weight, size)
        for wd in words:
            trial = (cur + " " + wd).strip()
            if f.getlength(trial) <= self.p(max_w) or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
        return lines

    def lower_third(self, layer, t, t0, t_out, name, label, role, x, y_line, max_w=1300):
        """Gold line, then the name, then 'Label | Role' — every element revealed by a mask."""
        X = self.p(x)
        Y = self.p(y_line)
        self.line_draw(layer, X, Y, 220, 7, t, t0, 0.45, t_out + 0.2, 0.35)
        name_size = self.fit_runs([(name, "Cinzel", 700, "white")], 128, max_w)
        self.masked_line(layer, [(name, "Cinzel", 700, "white")], name_size, X, Y + self.p(150), t, t0 + 0.12,
                         0.7, t_out + 0.05, tracking=0.02)
        if not role:  # label only, e.g. "HOST"
            self.masked_line(layer, [(label.upper(), "Montserrat", 600, "gold")], 50, X, Y + self.p(240), t,
                             t0 + 0.28, 0.7, t_out, tracking=0.3)
            return
        runs = [(label, "Montserrat", 600, "gold"), ("  |  ", "Montserrat", 500, "gold"),
                (role, "Montserrat", 500, "warm_white")]
        rs = self.fit_runs(runs, 50, max_w, 0.02, 40)
        if self.text_width(tuple(runs), rs, 0.02) <= self.p(max_w):
            self.masked_line(layer, runs, rs, X, Y + self.p(245), t, t0 + 0.28, 0.7, t_out, tracking=0.02)
        else:  # very long roles: label on its own line, role wrapped below
            self.masked_line(layer, [(label.upper(), "Montserrat", 600, "gold")], 44, X, Y + self.p(240), t,
                             t0 + 0.28, 0.7, t_out, tracking=0.25)
            for i, ln in enumerate(self.wrap(role, "Montserrat", 500, 50, max_w)[:2]):
                self.masked_line(layer, [(ln, "Montserrat", 500, "warm_white")], 50, X,
                                 Y + self.p(320 + 72 * i), t, t0 + 0.36 + 0.08 * i, 0.7, t_out, tracking=0.02)

    def episode_card(self, layer, t, t0, t_out, number, title, x, y_top, max_w=1300):
        X, Y = self.p(x), self.p(y_top)
        self.masked_line(layer, [(f"EPISODE {number}", "Montserrat", 600, "gold")], 62, X, Y, t, t0, 0.6,
                         t_out + 0.15, tracking=0.3)
        self.line_draw(layer, X, Y + self.p(58), 260, 7, t, t0 + 0.15, 0.55, t_out + 0.25, 0.35)
        size = 116
        lines = self.wrap(title, "Cinzel", 600, size, max_w)
        while len(lines) > 3 and size > 70:
            size -= 6
            lines = self.wrap(title, "Cinzel", 600, size, max_w)
        for i, ln in enumerate(lines[:3]):
            self.masked_line(layer, [(ln, "Cinzel", 600, "white")], size, X,
                             Y + self.p(200) + self.p(size * 1.22) * i, t, t0 + 0.3 + 0.12 * i, 0.75,
                             t_out + 0.06 * i, tracking=0.01)

    def topic_heading(self, layer, t, t0, t_out, heading, x=307, y=300, label="TOPIC"):
        X, Y = self.p(x), self.p(y)
        self.masked_line(layer, [(label, "Montserrat", 600, "gold")], 44, X, Y, t, t0, 0.6, t_out + 0.1,
                         tracking=0.3)
        self.line_draw(layer, X, Y + self.p(44), 180, 6, t, t0 + 0.1, 0.5, t_out + 0.2, 0.35)
        size = self.fit_runs([(heading, "Montserrat", 700, "white")], 100, 2800)
        self.masked_line(layer, [(heading, "Montserrat", 700, "white")], size, X, Y + self.p(170), t, t0 + 0.22,
                         0.7, t_out)

    # ------------------------------------------------------------ the opening
    def show_lockup(self, layer, t, t0, t_out, x, y_top):
        """General-intro ending: echoes the opening title as a sign-off, no episode details."""
        X, Y = self.p(x), self.p(y_top)
        show = self.brand["show"]
        self.masked_line(layer, [("THE PODCAST", "Montserrat", 600, "gold")], 54, X, Y, t, t0, 0.6, t_out + 0.15,
                         tracking=0.3)
        self.line_draw(layer, X, Y + self.p(52), 260, 7, t, t0 + 0.15, 0.55, t_out + 0.25, 0.35)
        colours = ["white", "white", "gold"]
        for i, ln in enumerate(show["title_lines"]):
            self.masked_line(layer, [(ln, "Montserrat", 800, colours[i])], 122, X, Y + self.p(210 + 140 * i), t,
                             t0 + 0.3 + 0.12 * i, 0.75, t_out + 0.06 * i)
        runs = [(" ".join(a + " " + b for a, b in show["tagline"]), "Montserrat", 600, "warm_white")]
        size = self.fit_runs(runs, 40, 1220, 0.12, 30)
        self.masked_line(layer, runs, size, X, Y + self.p(210 + 140 * 2 + 120), t, t0 + 0.75, 0.7, t_out,
                         tracking=0.12)

    def opening(self, t, ending="episode"):
        """ending: 'episode' (EPISODE n / title card) or 'general' (show lockup, any episode)."""
        L = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        show, host = self.brand["show"], self.brand["host"]

        # 0-4 s: show title over the city
        if t < 4.3:
            self.scrim(L, "left", out_cubic(prog(t, 0.05, 0.7)) * (1 - in_out_cubic(prog(t, 3.7, 0.5))),
                       reach=0.62, strength=0.62)
            x, size, pitch, b0 = 307, 228, 262, 820
            colours = ["white", "white", "gold"]
            for i, ln in enumerate(show["title_lines"]):
                self.masked_line(L, [(ln, "Montserrat", 800, colours[i])], size, self.p(x), self.p(b0 + pitch * i),
                                 t, 0.35 + 0.15 * i, 0.8, 3.45 + 0.07 * i, 0.45)
            widest = max(self.text_width(((ln, "Montserrat", 800, "white"),), size) for ln in show["title_lines"])
            self.line_draw(L, self.p(x), self.p(b0 + pitch * 2 + 92), widest / self.k, 9, t, 1.15, 0.85, 3.4, 0.45)

        # 4-8 s: tagline on the stage backdrop (old sign removed), mic with one pulse
        if 3.95 < t < 8.2:
            cx, size = 1790, 116
            self.scrim(L, "ellipse", out_cubic(prog(t, 4.0, 0.6)) * (1 - in_out_cubic(prog(t, 7.6, 0.5))),
                       cx=cx / DW, cy=0.36, rx=0.30, ry=0.30, strength=0.42)
            self.mic(L, self.p(cx), self.p(455), 120, t, 4.2, pulse_t=6.45, t_out=7.55)
            for i, (a, b) in enumerate(show["tagline"]):
                runs = [(a + " ", "Montserrat", 700, "white"), (b, "Montserrat", 700, "gold")]
                self.masked_line(L, runs, size, self.p(cx), self.p(720 + 175 * i), t, 4.45 + 0.6 * i, 0.75,
                                 7.5 + 0.06 * i, 0.45, tracking=0.08, align="center", rise=self.p(60))

        # host lower-third on the studio wall, right of Brian (clear of face, hands and mic)
        if 8.4 < t < 12.3:
            self.scrim(L, "ellipse", out_cubic(prog(t, 8.45, 0.5)) * (1 - in_out_cubic(prog(t, 11.8, 0.4))),
                       cx=0.78, cy=0.62, rx=0.25, ry=0.16, strength=0.35)
            self.lower_third(L, t, 8.6, 11.75, host["name"], host["role_label"], host["role"], 2300, 1235)

        # episode card, then hand-off to the episode
        if t > 11.95:
            self.scrim(L, "ellipse", out_cubic(prog(t, 12.0, 0.5)) * (1 - in_out_cubic(prog(t, 14.75, 0.5))),
                       cx=0.78, cy=0.55, rx=0.26, ry=0.24, strength=0.35)
            if ending == "general":
                self.show_lockup(L, t, 12.05, 14.75, 2300, 760)
            else:
                self.episode_card(L, t, 12.05, 14.75, self.ep["episode_number"], self.ep["episode_title"], 2300,
                                  900)
        return L

    # ------------------------------------------------------------ outro
    def outro(self, t, background=True):
        L = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        bg_op = out_cubic(prog(t, 0.0, 0.6)) * (1 - in_out_cubic(prog(t, 5.4, 0.6)))
        if background:
            bg = Image.new("RGBA", (self.w, self.h), self.C["navy"] + (255,))
            self.put(L, bg, 0, 0, bg_op)
        else:
            self.scrim(L, "ellipse", bg_op, cx=0.5, cy=0.5, rx=0.55, ry=0.5, strength=0.7)
        cx = self.p(1920)
        self.mic(L, cx, self.p(700), 120, t, 0.5, pulse_t=1.6, t_out=5.3)
        self.masked_line(L, [("Thanks for listening", "Cinzel", 600, "white")], 150, cx, self.p(1060), t, 0.7, 0.8,
                         5.2, align="center", tracking=0.01)
        self.line_draw(L, cx - self.p(300), self.p(1150), 600, 7, t, 1.0, 0.7, 5.25, 0.4, origin="center")
        runs = [("Follow ", "Montserrat", 600, "warm_white"),
                (self.brand["show"]["title_plain"], "Montserrat", 700, "gold")]
        self.masked_line(L, runs, 64, cx, self.p(1290), t, 1.25, 0.75, 5.15, align="center", tracking=0.04)
        return L
