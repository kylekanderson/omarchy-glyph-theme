# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10", "numpy>=1.26", "scipy>=1.11"]
# ///
"""Render the Corvid wallpapers, unlock image and previews.

Every image is built from monospace glyphs laid on a character grid, then
lit with a bloom/halo pass and a chromatic-aberration pass on the frame.

    uv run src/generate.py                 # everything, 5120x2880
    uv run src/generate.py --only perch    # one wallpaper
    uv run src/generate.py --size 3840x2160 --fast
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import shapes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Palette (kept in sync with colors.toml)
# --------------------------------------------------------------------------

WHITE = (244, 242, 255)
LAVENDER = (168, 160, 232)
ORANGE = (255, 122, 58)
EMBER = (255, 86, 38)
DEEP_EMBER = (170, 44, 20)
HALO = (0.80, 0.78, 1.00)  # tint of the wide glow
FRINGE_SHIFT = 4.6  # chromatic aberration, in units (px at 5K)

# Glyph tiers, from the silhouette edge inwards.
EDGE = ".:+-/ϟ1%'"
MID = "01%&○/+#$ϟ"
CORE = "@#0&%✱○■8$1"

FONT_CANDIDATES = [
    os.environ.get("CORVID_FONT", ""),
    "~/Library/Fonts/JetBrainsMonoNerdFont-Bold.ttf",
    "~/Library/Fonts/JetBrainsMono-Bold.ttf",
    "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Bold.ttf",
    "/usr/share/fonts/TTF/JetBrainsMono-Bold.ttf",
    "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf",
]


def find_font() -> str:
    for cand in FONT_CANDIDATES:
        if cand and Path(cand).expanduser().exists():
            return str(Path(cand).expanduser())
    sys.exit("❌ JetBrains Mono not found. Set CORVID_FONT=/path/to/font.ttf")


FONT_PATH = find_font()


@lru_cache(maxsize=None)
def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, max(4, int(size)))


def scale(color, k):
    return tuple(max(0, min(255, int(c * k))) for c in color)


def lerp(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------


class Scene:
    """A black canvas with three layers: glyphs, frame and a soft haze."""

    def __init__(self, w: int, h: int, seed: int):
        self.w, self.h = w, h
        # One unit is 1px on a 5120x2880 canvas. Using the tighter axis keeps
        # glyphs and frames the same physical size on 21:9, 32:9 or 3:2 canvases.
        self.u = min(w / 5120, h / 2880)
        self.rng = np.random.default_rng(seed)
        self.glyph = Image.new("RGB", (w, h))
        self.frame = Image.new("RGB", (w, h))
        self.haze = Image.new("L", (w, h))
        self.dg = ImageDraw.Draw(self.glyph)
        self.df = ImageDraw.Draw(self.frame)
        self.dh = ImageDraw.Draw(self.haze)
        self._stars = []

    # -- glyph primitives --------------------------------------------------

    def put(self, x, y, ch, size, color, layer="glyph"):
        d = self.dg if layer == "glyph" else self.df
        s = size
        if ch == "■":
            r = s * 0.31
            d.rectangle([x - r, y - r, x + r, y + r], fill=color)
        elif ch == "✱":
            arm, wdt = s * 0.30, max(1, int(s * 0.12))
            for a in (0, 45, 90, 135):
                dx, dy = math.cos(math.radians(a)) * arm, math.sin(math.radians(a)) * arm
                d.line([x - dx, y - dy, x + dx, y + dy], fill=color, width=wdt)
        elif ch == "ϟ":
            pts = [(0.16, -0.40), (-0.12, 0.03), (0.12, -0.03), (-0.16, 0.42)]
            d.line([(x + px * s, y + py * s) for px, py in pts], fill=color,
                   width=max(1, int(s * 0.09)), joint="curve")
        else:
            d.text((x, y), ch, font=font(s), fill=color, anchor="mm")

    # -- silhouettes -------------------------------------------------------

    def grid_from_shape(self, shape, cols, rows, flip=False, ss=6):
        """Rasterise normalised polygons to a boolean cols x rows grid."""
        im = Image.new("L", (cols * ss, rows * ss))
        d = ImageDraw.Draw(im)
        for poly in shapes.mirror(shape) if flip else shape:
            d.polygon([(x / 100 * cols * ss, y / 100 * rows * ss) for x, y in poly], fill=255)
        a = np.asarray(im, dtype=np.float32).reshape(rows, ss, cols, ss).mean(axis=(1, 3)) / 255
        return a > 0.42

    def fill(self, occ, x0, y0, pitch, *, ry=1.08, size=None, colorize=None,
             orange=0.03, drop=0.06, haze=70, bright=1.0, tiers=(EDGE, MID, CORE),
             blocks=0.035, skip=None):
        """Lay glyphs on every occupied cell. Returns the cell centres used."""
        rows, cols = occ.shape
        size = size or pitch * 1.05
        dist = ndimage.distance_transform_edt(np.pad(occ, 1))[1:-1, 1:-1]
        rng = self.rng
        py = pitch * ry
        placed = []
        for r in range(rows):
            for c in range(cols):
                if not occ[r, c]:
                    continue
                x, y = x0 + (c + 0.5) * pitch, y0 + (r + 0.5) * py
                if skip is not None and skip(x, y):
                    continue
                if haze:
                    self.dh.rectangle([x - pitch / 2, y - py / 2, x + pitch / 2, y + py / 2], fill=haze)
                d = dist[r, c]
                edge = d <= 1.0
                if rng.random() < (drop * 2.6 if edge else drop):
                    continue
                tier = tiers[0] if edge else tiers[1] if d <= 2.0 else tiers[2]
                ch = tier[rng.integers(len(tier))]
                if not edge and rng.random() < blocks:
                    ch = "■"
                k = rng.uniform(0.45, 0.8) if edge else rng.uniform(0.75, 1.0)
                if ch == "■":
                    k = 1.0
                if colorize:
                    col = colorize(x, y, d, rng)
                elif rng.random() < orange and ch != "■":
                    col = ORANGE
                    k = max(k, 0.9)
                elif edge and rng.random() < 0.25:
                    col = LAVENDER
                else:
                    col = WHITE
                self.put(x, y, ch, size, scale(col, k * bright))
                placed.append((x, y))
        return placed

    def shape(self, shape, cx, cy, width, height, pitch, flip=False, **kw):
        """Place a normalised shape inside a box centred at (cx, cy)."""
        ry = kw.get("ry", 1.08)
        cols, rows = int(width / pitch), int(height / (pitch * ry))
        occ = self.grid_from_shape(shape, cols, rows, flip=flip)
        x0 = cx - cols * pitch / 2
        y0 = cy - rows * pitch * ry / 2
        return self.fill(occ, x0, y0, pitch, **kw)

    # -- ambient -----------------------------------------------------------

    def stars(self, n, size, **kw):
        """Queue a starfield; it's drawn at render time, around the silhouettes."""
        self._stars.append((n, size, kw))

    def _draw_stars(self, n, size, *, region=None, bright=(0.12, 0.45), orange=0.04,
                    chars=".·+.·*"):
        x0, y0, x1, y1 = region or (0, 0, self.w, self.h)
        rng = self.rng
        haze = self.haze.load()
        for _ in range(n):
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            if haze[min(int(x), self.w - 1), min(int(y), self.h - 1)]:
                continue
            col = ORANGE if rng.random() < orange else lerp(WHITE, LAVENDER, rng.random())
            self.put(x, y, chars[rng.integers(len(chars))], size * rng.uniform(0.7, 1.15),
                     scale(col, rng.uniform(*bright)))

    def frame_dashed(self, margin, step, size, *, top="-", side="|", corner="*",
                     bright=0.55, sides=True):
        m, w, h = margin, self.w, self.h
        col = scale(WHITE, bright)
        n = int((w - 2 * m) / step)
        sx = (w - 2 * m) / n
        for i in range(1, n):
            for yy in (m, h - m):
                self.put(m + i * sx, yy, top, size, col, layer="frame")
        if sides:
            k = int((h - 2 * m) / (step * 0.62))
            sy = (h - 2 * m) / k
            for j in range(1, k):
                for xx in (m, w - m):
                    ch = side if isinstance(side, str) else side[0 if xx == m else 1]
                    self.put(xx, m + j * sy, ch, size, col, layer="frame")
        for xx in (m, w - m):
            for yy in (m, h - m):
                self.put(xx, yy, corner, size, scale(WHITE, bright * 1.2), layer="frame")

    def text(self, x, y, s, size, color, pitch=None, layer="glyph"):
        pitch = pitch or size * 0.62
        for i, ch in enumerate(s):
            if ch != " ":
                self.put(x + i * pitch, y, ch, size, color, layer=layer)

    # -- lighting ----------------------------------------------------------

    def render(self, *, glow=1.0, bloom=1.0, haze=1.0, halo=HALO, fast=False) -> Image.Image:
        u = self.u
        for n, size, kw in self._stars:
            self._draw_stars(n, size, **kw)
        g = np.asarray(self.glyph, dtype=np.float32) / 255
        fr = np.asarray(self.frame, dtype=np.float32) / 255

        def blur(img, radius, down):
            small = img.resize((max(1, self.w // down), max(1, self.h // down)), Image.BILINEAR)
            small = small.filter(ImageFilter.GaussianBlur(radius / down))
            return np.asarray(small.resize((self.w, self.h), Image.BICUBIC), dtype=np.float32) / 255

        down = 8 if fast else 4
        wide = blur(self.glyph, 70 * u, down)
        mid = blur(self.glyph, 22 * u, max(1, down // 2))
        tight = np.asarray(self.glyph.filter(ImageFilter.GaussianBlur(4 * u)), dtype=np.float32) / 255
        fog = blur(self.haze, 90 * u, down)[..., None]

        lum = wide.mean(axis=2, keepdims=True)
        halo_rgb = np.array(halo, dtype=np.float32)
        out = (
            fog * 0.16 * haze * halo_rgb
            + (0.55 * wide + 0.9 * lum * halo_rgb) * 1.25 * glow
            + mid * 0.55 * bloom
            + tight * 0.6 * bloom
            + g
        )

        # Frame: offset red and blue channels for a chromatic fringe.
        shift = max(1, int(round(u * FRINGE_SHIFT)))
        fglow = np.asarray(self.frame.filter(ImageFilter.GaussianBlur(5 * u)), dtype=np.float32) / 255
        f = fr + fglow * 0.9
        fringe = np.zeros_like(f)
        fringe[..., 0] = np.roll(f[..., 0] * 1.1 + f[..., 2] * 0.25, shift, axis=1)
        fringe[..., 1] = f[..., 1] * 0.82
        fringe[..., 2] = np.roll(f[..., 2] * 1.2 + f[..., 0] * 0.2, -shift, axis=1)
        out += fringe

        # Soft shoulder so overlapping glow never clips harshly, then dither.
        out = np.where(out > 0.8, 0.8 + (1 - np.exp(-(out - 0.8) * 5)) * 0.2, out)
        out += self.rng.uniform(-0.5, 0.5, out.shape[:2])[..., None] / 255
        return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))


# --------------------------------------------------------------------------
# Wallpapers
# --------------------------------------------------------------------------


def wp_perch(w, h, fast=False):
    """The raven, framed. The theme's signature image."""
    s = Scene(w, h, seed=7)
    u = s.u
    pitch = 40 * u
    box = h * 0.62
    cx, cy = w * 0.5, h * 0.52
    s.stars(260, 26 * u, bright=(0.06, 0.28))
    s.shape(shapes.PERCH, cx, cy, box * 1.05, box, pitch)
    # The branch: long, thin and dimmer, in its own glyph set.
    s.shape(shapes.BRANCH, cx, cy, box * 1.05, box, pitch,
            tiers=("=-~_", "=#-~", "=#"), bright=0.62, orange=0.05, blocks=0, haze=40)
    s.frame_dashed(64 * u, 54 * u, 30 * u)
    return s.render(fast=fast)


def wp_glide(w, h, fast=False):
    """A huge raven gliding overhead, wings spanning the screen."""
    s = Scene(w, h, seed=11)
    u = s.u
    s.stars(900, 24 * u, bright=(0.05, 0.32))
    box = min(h * 1.28, w * 0.86)  # the shape spans ~67% of its box vertically
    s.shape(shapes.GLIDE, w * 0.5, h * 0.47, box, box, 34 * u, drop=0.14, bright=0.92)
    return s.render(glow=0.6, bloom=0.8, fast=fast)


def glide_small(s, cx, cy, span, pitch=None, **kw):
    """A distant gliding raven, `span` wide. Fine pitch keeps its feathers legible."""
    pitch = pitch or span / 70
    kw.setdefault("drop", 0.02)
    kw.setdefault("haze", 30)
    kw.setdefault("size", pitch * 1.25)
    return s.shape(shapes.GLIDE, cx, cy, span, span, pitch, **kw)


def wp_wire(w, h, fast=False):
    """Three ravens on a sagging wire across the lower third."""
    s = Scene(w, h, seed=23)
    u = s.u
    pitch = 26 * u
    s.stars(700, 22 * u, region=(0, 0, w, h * 0.7), bright=(0.05, 0.3))

    def wire_y(x):
        t = x / w
        return h * 0.60 + h * 0.08 * (1 - (2 * t - 1) ** 2)  # sags in the middle

    # (x position, box height, facing left?)
    birds = [(0.18, 0.30, False), (0.40, 0.24, True), (0.70, 0.40, True)]
    for fx, fh, flip in birds:
        box = h * fh
        bx = w * fx
        feet = wire_y(bx)
        # place the shape so its toe line sits on the wire
        cy = feet - box * (shapes.PERCH_FEET_Y / 100 - 0.5) - pitch * 0.4
        s.shape(shapes.PERCH, bx, cy, box * 1.05, box, box / 40, flip=flip)

    # The wire itself: a catenary of dashes that passes behind the birds.
    step = pitch * 0.55
    x = -step
    while x < w + step:
        y = wire_y(x)
        if 0 <= x < w and s.haze.getpixel((int(x), int(y))) == 0:
            ch = "=" if s.rng.random() < 0.06 else "-"
            s.put(x, y, ch, 30 * u, scale(lerp(WHITE, LAVENDER, 0.35), s.rng.uniform(0.5, 0.75)))
        x += step

    # a distant pair heading out
    glide_small(s, w * 0.84, h * 0.17, w * 0.09)
    glide_small(s, w * 0.90, h * 0.25, w * 0.055)
    s.frame_dashed(64 * u, 54 * u, 30 * u, top="=", side=("|", "|"), corner="+")
    return s.render(fast=fast)


def wp_murmuration(w, h, fast=False):
    """A flock of thousands, swirling across the screen as a ribbon of glyphs."""
    s = Scene(w, h, seed=31)
    u = s.u
    rng = s.rng
    s.stars(350, 20 * u, bright=(0.04, 0.22))

    def spine(t):
        # a long rising sweep with one lazy loop in it
        x = w * (0.06 + 0.80 * t) + w * 0.07 * math.sin(t * math.tau * 1.1)
        y = h * (0.74 - 0.42 * t + 0.10 * math.sin(t * math.tau * 1.6))
        return x, y

    def width(t):
        # the ribbon twists: wide sheets pinched to near-nothing at the folds
        return h * (0.012 + 0.10 * abs(math.sin(t * math.pi * 2.6 + 0.4)) * math.sin(math.pi * t) ** 0.6)

    n = 14000 if not fast else 5000
    for _ in range(n):
        t = rng.beta(1.6, 1.5)
        x, y = spine(t)
        x2, y2 = spine(min(1, t + 1e-3))
        tx, ty = x2 - x, y2 - y
        norm = math.hypot(tx, ty) or 1
        nx, ny = -ty / norm, tx / norm
        wd = width(t)
        off = rng.normal(0, 0.5) * wd
        along = rng.normal(0, h * 0.008)
        px, py = x + nx * off + tx / norm * along, y + ny * off + ty / norm * along
        dens = math.exp(-(off / wd) ** 2 * 1.2) * (0.6 + 0.4 * min(1, h * 0.05 / wd))
        ch = "v" if rng.random() < 0.6 else rng.choice(list("^~-·.˅"))
        size = rng.uniform(10, 20) * u * (0.85 + 0.4 * dens)
        col = ORANGE if rng.random() < 0.012 else lerp(LAVENDER, WHITE, min(1, dens))
        s.put(px, py, ch, size, scale(col, rng.uniform(0.35, 1.0) * (0.35 + 0.65 * min(1, dens))))
        if rng.random() < 0.05:
            s.dh.ellipse([px - 26 * u, py - 26 * u, px + 26 * u, py + 26 * u], fill=50)

    # a few nearer ravens break from the flock
    glide_small(s, w * 0.80, h * 0.20, w * 0.16)
    glide_small(s, w * 0.67, h * 0.12, w * 0.08)
    glide_small(s, w * 0.91, h * 0.36, w * 0.06)
    s.frame_dashed(64 * u, 54 * u, 30 * u, sides=False, corner="+")
    return s.render(glow=0.9, fast=fast)


def wp_ember(w, h, fast=False):
    """The raven again, facing left, burning orange, with rising sparks."""
    s = Scene(w, h, seed=43)
    u = s.u
    rng = s.rng
    box = h * 0.66
    cx, cy = w * 0.56, h * 0.53
    left = cx - box * 1.05 / 2

    def colorize(x, y, d, r):
        # white-hot at the head, cooling to deep ember at the tail
        t = min(1, max(0, (x - left) / (box * 1.05)))
        t = 1 - t  # facing left: head is on the left
        base = lerp(lerp(WHITE, ORANGE, 0.55), DEEP_EMBER, t ** 1.3) if t > 0.15 else lerp(WHITE, ORANGE, t / 0.15 * 0.55)
        return base if d > 1 or r.random() > 0.3 else lerp(base, EMBER, 0.6)

    s.shape(shapes.PERCH, cx, cy, box * 1.05, box, 40 * u, flip=True, colorize=colorize, haze=50)
    s.shape(shapes.BRANCH, cx, cy, box * 1.05, box, 40 * u, flip=True,
            tiers=("=-~_", "=#-~", "=#"), colorize=lambda *_: DEEP_EMBER, blocks=0, haze=30)

    # sparks drifting up and to the right
    for _ in range(1100):
        x = rng.uniform(left - box * 0.2, w)
        y = rng.uniform(0, h)
        near = math.exp(-(((x - cx) / (w * 0.35)) ** 2 + ((y - cy) / (h * 0.45)) ** 2))
        if rng.random() > 0.15 + 0.85 * near:
            continue
        col = lerp(EMBER, ORANGE, rng.random()) if rng.random() < 0.85 else WHITE
        s.put(x, y, rng.choice(list("··.:'+*ϟ")), rng.uniform(14, 30) * u,
              scale(col, rng.uniform(0.2, 0.9) * (0.3 + 0.7 * near)))
    return s.render(glow=1.05, halo=(1.0, 0.55, 0.35), fast=fast)


def wp_nocturne(w, h, fast=False):
    """A crescent moon of glyphs, shaded by density, with a raven crossing it."""
    s = Scene(w, h, seed=53)
    u = s.u
    rng = s.rng
    s.stars(1100, 22 * u, bright=(0.05, 0.4))
    pitch = 26 * u
    ry = 1.08
    R = h * 0.38
    mx, my = w * 0.62, h * 0.48
    ramp = " .:-=+*%#@■"
    rbox, rcx, rcy = R * 1.32, mx + R * 0.02, my - R * 0.04  # raven box, inside the moon
    # The raven is cut out of the moon's grid, so sample its shape on the
    # same cell lattice the moon uses.
    rcols, rrows = int(rbox / pitch), int(rbox / (pitch * ry))
    rocc = s.grid_from_shape(shapes.GLIDE, rcols, rrows)

    def in_raven(x, y):
        c = int((x - (rcx - rbox / 2)) / pitch)
        r = int((y - (rcy - rbox / 2)) / (pitch * ry))
        return 0 <= r < rrows and 0 <= c < rcols and rocc[r, c]

    craters = [(-0.55, -0.45, 0.16), (-0.30, 0.55, 0.20), (-0.78, 0.15, 0.11),
               (0.55, -0.55, 0.12), (0.62, 0.42, 0.14)]
    cols, rows = int(2 * R / pitch) + 2, int(2 * R / (pitch * ry)) + 2
    x0, y0 = mx - cols * pitch / 2, my - rows * pitch * ry / 2
    sun = np.array([-0.45, -0.22, 0.86])  # from the front-left: a waning gibbous
    sun /= np.linalg.norm(sun)
    for r in range(rows):
        for c in range(cols):
            x, y = x0 + (c + 0.5) * pitch, y0 + (r + 0.5) * pitch * ry
            nx, ny = (x - mx) / R, (y - my) / R
            rr = nx * nx + ny * ny
            if rr > 1:
                continue
            nz = math.sqrt(1 - rr)
            lit = max(0.0, nx * sun[0] + ny * sun[1] + nz * sun[2])
            for cxr, cyr, crr in craters:
                if (nx - cxr) ** 2 + (ny - cyr) ** 2 < crr * crr:
                    lit *= 0.62
            lit = lit ** 0.9 + rng.normal(0, 0.05)
            if in_raven(x, y):
                continue  # the raven is a hole in the moon
            s.dh.rectangle([x - pitch / 2, y - pitch / 2, x + pitch / 2, y + pitch / 2],
                           fill=int(25 + 50 * max(0, lit)))
            idx = int(np.clip(lit, 0, 0.999) * len(ramp))
            ch = ramp[idx]
            if ch == " ":
                if rng.random() < 0.1:
                    s.put(x, y, ".", pitch, scale(LAVENDER, 0.25))
                continue
            col = ORANGE if rng.random() < 0.012 else lerp(LAVENDER, WHITE, min(1, lit * 1.3))
            s.put(x, y, ch, pitch * 1.05, scale(col, 0.35 + 0.65 * min(1, lit)))

    return s.render(glow=0.7, fast=fast)


def wp_void(w, h, fast=False):
    """Mostly nothing: a frame, a few stars and a small raven keeping watch."""
    s = Scene(w, h, seed=61)
    u = s.u
    s.stars(180, 24 * u, bright=(0.05, 0.3))
    box = h * 0.2
    s.shape(shapes.PERCH, w * 0.83, h * 0.76, box * 1.05, box, 15 * u, haze=40)
    s.shape(shapes.BRANCH, w * 0.83, h * 0.76, box * 1.05, box, 15 * u,
            tiers=("=-~_", "=#-~", "=#"), bright=0.6, blocks=0, haze=20)
    s.text(w * 0.06, h * 0.86, "~ $ _", 34 * u, scale(WHITE, 0.55))
    s.frame_dashed(64 * u, 54 * u, 30 * u, side=(">", "<"))
    return s.render(fast=fast)


WALLPAPERS = {
    "perch": ("1-perch", wp_perch),
    "glide": ("2-glide", wp_glide),
    "wire": ("3-wire", wp_wire),
    "murmuration": ("4-murmuration", wp_murmuration),
    "ember": ("5-ember", wp_ember),
    "nocturne": ("6-nocturne", wp_nocturne),
    "void": ("7-void", wp_void),
}


# --------------------------------------------------------------------------
# Unlock + previews
# --------------------------------------------------------------------------


def to_transparent(img: Image.Image) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) / 255
    alpha = a.max(axis=2, keepdims=True)
    rgb = np.where(alpha > 1e-4, a / np.maximum(alpha, 1e-4), 0)
    out = np.concatenate([rgb, alpha], axis=2)
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")


def make_unlock(fast=False):
    w, h = 800, 520
    s = Scene(w * 4, h * 4, seed=71)  # render big, downsample for crisp glyphs
    box = s.h * 0.86
    s.shape(shapes.PERCH, s.w / 2, s.h * 0.5, box * 1.05, box, box / 42, haze=0, drop=0.03)
    img = s.render(glow=0.7, haze=0, fast=fast).resize((w, h), Image.LANCZOS)
    return to_transparent(img)


def make_preview_unlock(unlock: Image.Image):
    w, h = 1920, 1080
    img = Image.new("RGB", (w, h), (0, 0, 0))
    u = unlock.resize((int(unlock.width * 0.9), int(unlock.height * 0.9)), Image.LANCZOS)
    img.paste(u, ((w - u.width) // 2, int(h * 0.42 - u.height / 2)), u)
    d = ImageDraw.Draw(img)
    bw, bh = 520, 64
    x0, y0 = (w - bw) // 2, int(h * 0.72)
    d.rectangle([x0, y0, x0 + bw, y0 + bh], outline=(244, 242, 255), width=2)
    f = ImageFont.truetype(FONT_PATH, 26)
    d.text((x0 + 24, y0 + bh // 2), "Enter Password", font=f, fill=(110, 106, 140), anchor="lm")
    return img


def make_preview(wall: Image.Image):
    """A mock Omarchy desktop: Waybar, a terminal and an editor, over 1-perch."""
    W, H = 2560, 1440
    img = wall.resize((W, H), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    f = ImageFont.truetype(FONT_PATH, 22)
    fs = ImageFont.truetype(FONT_PATH, 20)
    fg, dim, acc, bg = (216, 214, 228), (110, 106, 140), (240, 118, 58), (8, 8, 10, 218)
    pal = {
        "red": (255, 107, 94), "green": (143, 214, 176), "yellow": (244, 184, 96),
        "blue": (139, 156, 255), "magenta": (196, 155, 255), "cyan": (138, 212, 230),
    }
    # Waybar
    d.rectangle([0, 0, W, 34], fill=(0, 0, 0, 255))
    for i, n in enumerate("12345"):
        d.text((24 + i * 34, 17), n, font=fs, fill=acc if i == 0 else dim, anchor="lm")
    d.text((W // 2, 17), "Thursday 14:34", font=fs, fill=fg, anchor="mm")
    d.text((W - 24, 17), "󰕾  󰤨  󰁹 98%", font=fs, fill=fg, anchor="rm")

    gap, top = 14, 34 + 14

    def window(x0, y0, x1, y1, active):
        d.rectangle([x0, y0, x1, y1], fill=bg)
        col = (242, 240, 255) if active else (40, 40, 48)
        d.rectangle([x0, y0, x1, y1], outline=col, width=2)

    # Terminal (left)
    lx1 = int(W * 0.40)
    window(gap, top, lx1, H // 2 - gap // 2, True)
    lines = [
        [("~/dev/corvid ", pal["blue"]), ("main ", pal["magenta"]), ("❯ ", acc), ("fastfetch", fg)],
        [],
        [("   ", acc), ("OS     ", pal["blue"]), ("Arch Linux · Omarchy", fg)],
        [("   ", acc), ("WM     ", pal["blue"]), ("Hyprland", fg)],
        [("   ", acc), ("Theme  ", pal["blue"]), ("Corvid", fg)],
        [("   ", acc), ("Term   ", pal["blue"]), ("Ghostty", fg)],
        [("   ", acc), ("Font   ", pal["blue"]), ("JetBrainsMono Nerd Font", fg)],
        [],
        [("  ", None)] + [("███", c) for c in [(30, 30, 36), *pal.values(), fg]],
        [],
        [("~/dev/corvid ", pal["blue"]), ("main ", pal["magenta"]), ("❯ ", acc), ("_", fg)],
    ]
    for i, segs in enumerate(lines):
        x = gap + 26
        for text, col in segs:
            if col:
                d.text((x, top + 34 + i * 36), text, font=f, fill=col, anchor="lm")
            x += int(f.getlength(text))

    # btop-ish (bottom left)
    window(gap, H // 2 + gap // 2, lx1, H - gap, False)
    y = H // 2 + 60
    d.text((gap + 26, y), "cpu", font=f, fill=acc, anchor="lm")
    for i in range(14):
        v = 0.5 + 0.45 * math.sin(i * 1.7)
        d.text((gap + 26, y + 40 + i * 30), f"C{i:<2}", font=fs, fill=dim, anchor="lm")
        bx = gap + 90
        bw = int((lx1 - bx - 120) * v)
        col = pal["green"] if v < 0.6 else pal["yellow"] if v < 0.85 else pal["red"]
        d.rectangle([bx, y + 32 + i * 30, bx + bw, y + 46 + i * 30], fill=col + (200,))
        d.text((lx1 - 30, y + 40 + i * 30), f"{int(v * 100):>3}%", font=fs, fill=fg, anchor="rm")

    # Editor (right)
    rx0 = lx1 + gap
    window(rx0, top, W - gap, H - gap, False)
    code = [
        [("-- ", dim), ("The raven keeps watch.", dim)],
        [("local ", pal["magenta"]), ("corvid ", fg), ("= ", acc), ("require", pal["blue"]), ("(", fg),
         ('"corvid"', pal["green"]), (")", fg)],
        [],
        [("function ", pal["magenta"]), ("corvid", pal["blue"]), (".", fg), ("perch", pal["blue"]),
         ("(branch, glyphs)", fg)],
        [("  ", None), ("if ", pal["magenta"]), ("not ", pal["magenta"]), ("branch ", fg), ("then ", pal["magenta"])],
        [("    ", None), ("return ", pal["magenta"]), ("nil", pal["yellow"]), (", ", fg),
         ('"nowhere to land"', pal["green"])],
        [("  ", None), ("end", pal["magenta"])],
        [("  ", None), ("for ", pal["magenta"]), ("i ", fg), ("= ", acc), ("1", pal["yellow"]),
         (", ", fg), ("#glyphs ", fg), ("do", pal["magenta"])],
        [("    ", None), ("branch", fg), (":", fg), ("draw", pal["cyan"]), ("(glyphs[i], ", fg),
         ("0.42", pal["yellow"]), (")", fg)],
        [("  ", None), ("end", pal["magenta"])],
        [("  ", None), ("return ", pal["magenta"]), ("branch", fg)],
        [("end", pal["magenta"])],
        [],
        [("corvid", fg), (".", fg), ("perch", pal["blue"]), ("(", fg), ("wire", fg), (", ", fg),
         ('"@#%&01"', pal["green"]), (")", fg), ("  ", None), ("-- TODO: ember mode", pal["red"])],
    ]
    for i, segs in enumerate(code):
        y = top + 40 + i * 36
        d.text((rx0 + 24, y), f"{i + 1:>3}", font=fs, fill=(70, 68, 90), anchor="lm")
        x = rx0 + 90
        if i == 8:
            d.rectangle([rx0 + 2, y - 17, W - gap - 2, y + 17], fill=(38, 36, 58, 255))
        for text, col in segs:
            if col:
                d.text((x, y), text, font=f, fill=col, anchor="lm")
            x += int(f.getlength(text))
    sy = H - gap - 40
    d.rectangle([rx0 + 2, sy - 18, W - gap - 2, sy + 18], fill=(20, 20, 26, 255))
    d.rectangle([rx0 + 2, sy - 18, rx0 + 120, sy + 18], fill=acc + (255,))
    d.text((rx0 + 61, sy), "NORMAL", font=fs, fill=(0, 0, 0), anchor="mm")
    d.text((rx0 + 140, sy), "corvid.lua", font=fs, fill=fg, anchor="lm")
    return img


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="5120x2880", help="wallpaper size, WxH")
    ap.add_argument("--only", choices=list(WALLPAPERS) + ["extras"], action="append")
    ap.add_argument("--fast", action="store_true", help="cheaper blurs, for iterating")
    ap.add_argument("--out", default=str(ROOT), help="theme root to write into")
    ap.add_argument("--format", default="webp", choices=["webp", "png"])
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    out = Path(args.out)
    (out / "backgrounds").mkdir(parents=True, exist_ok=True)
    todo = args.only or list(WALLPAPERS) + ["extras"]

    perch = None
    for key in todo:
        if key == "extras":
            continue
        name, fn = WALLPAPERS[key]
        print(f"🎨 {name} ({w}x{h})", flush=True)
        img = fn(w, h, fast=args.fast)
        path = out / "backgrounds" / f"{name}.{args.format}"
        if args.format == "webp":
            img.save(path, quality=94, method=6)
        else:
            img.save(path, optimize=True)
        if key == "perch":
            perch = img

    if "extras" in todo:
        print("🎨 unlock + previews", flush=True)
        unlock = make_unlock(fast=args.fast)
        unlock.save(out / "unlock.png", optimize=True)
        make_preview_unlock(unlock).save(out / "preview-unlock.png", optimize=True)
        if perch is None:
            perch = wp_perch(2560, 1440, fast=args.fast)
        make_preview(perch).save(out / "preview.png", optimize=True)
    print("✅ done")


if __name__ == "__main__":
    main()
