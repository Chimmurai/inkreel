"""Promotional art for the Microsoft Store page, made by the app itself and strictly two-colour (ink and paper, no grey, no anti-aliasing).

The picture is a moonlit sea (scene 07 of tools/make_listing_scenes.py) run through InkReel at the Engraving level, with a loose film
garland hanging over it in three depths: far strips that are small and hazy, mid ones, and big near ones cut by the frame. Every frame
on the film is another scene, also run through the app. No text anywhere, as the Store asks for art.

    python tools/make_listing_art.py     ->  store/listing_assets/SuperHeroArt_1920x1080.png, SuperHeroArt_3840x2160.png,
                                              StoreBoxArt_1080x1080.png, StorePosterArt_1440x2160.png

The box art is a plain square of the same sea, with no film on it. The 4K hero and the poster are the same drawing doubled with nearest-neighbour, so every dot stays a crisp square, like a 2x zoom in the app.
Needs the scenes first (tools/make_listing_scenes.py) and the project environment: it calls `python -m benvid`, and the IRNM1 or DirectML
parts need onnxruntime in that environment (this script itself only uses Engraving)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = os.path.join(ROOT, "store", "listing_assets", "scenes")
OUT = os.path.join(ROOT, "store", "listing_assets")
HERO_SCENE = "07_crescent_sea.png"
FILM_SCENES = ["01_lake_dawn", "02_night_city", "03_still_life", "04_lighthouse", "06_dunes_aerial", "05_sea_moon", "07_crescent_sea"]
BLUE = np.load(os.path.join(ROOT, "benvid", "bluenoise128.npy"))
LETTERBOX = 138                                                   # black bars of the 2.39:1 window in the 16:9 hero

_cache: dict = {}


def render(scene: str, crop: tuple[float, float] | None, height: int, method: int = 5) -> np.ndarray:
    """a scene through the app at exactly `height` px tall; returns a bool array (True = paper/white).
    `crop` = (aspect, x-centre 0..1) takes a full-height window of the 1920x1080 source first; (x, y, w, h) takes that pixel box"""
    key = (scene, crop, height, method)
    if key in _cache:
        return _cache[key]
    src = Image.open(os.path.join(SCENES, scene)).convert("RGB")
    if crop and len(crop) == 4:
        x, y, w, h = crop
        src = src.crop((x, y, x + w, y + h))
    elif crop:
        aspect, cx = crop
        w = int(round(src.height * aspect))
        x0 = int(np.clip(cx * src.width - w / 2, 0, src.width - w))
        src = src.crop((x0, 0, x0 + w, src.height))
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.png"), os.path.join(d, "b.png")
        src.save(a)
        subprocess.run([sys.executable, "-m", "benvid", a, b, "--method", str(method), "--size", f"x{height}"],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = np.asarray(Image.open(b).convert("L")) > 127
    _cache[key] = out
    return out


def save(canvas: np.ndarray, name: str, scale: int = 1) -> None:
    if scale > 1:
        canvas = np.repeat(np.repeat(canvas, scale, 0), scale, 1)
    Image.fromarray((canvas * 255).astype(np.uint8)).convert("RGB").save(os.path.join(OUT, name))
    print(name, canvas.shape[::-1])


# ------------------------------------------------------------------------------------------------------------------------ the film
class Strip:
    """a length of film: `half` = half the width, `fh` = frame height (frames are 4:3), seeded so every strip shows other frames"""

    def __init__(self, half: int, fh: int, seed: int):
        self.half, self.fh, self.seed = half, fh, seed
        self.fw = int(round(fh * 4 / 3))
        self.gap = max(8, int(fh * 0.36))
        self.pitch = self.fw + self.gap
        self.hole_w, self.hole_h = max(4, int(fh * 0.17)), max(5, int(fh * 0.22))
        self.hole_p = self.pitch // 4
        self.edge = max(1, int(round(half / 28)))

    def frame(self, name: str) -> np.ndarray:
        method = 4 if "dunes" in name else 5
        t = render(name + ".png", (self.fw / self.fh, 0.5), self.fh, method)
        z = np.zeros((self.fh, self.fw), bool)
        z[:min(self.fh, t.shape[0]), :min(self.fw, t.shape[1])] = t[:self.fh, :self.fw]
        return z

    def flat(self, length: int) -> np.ndarray:
        half = self.half
        fl = np.zeros((2 * half, length), bool)
        fl[:self.edge, :] = True
        fl[-self.edge:, :] = True
        margin = max(3, int(half * 0.1))
        for x in range(4, length - self.hole_w - 2, self.hole_p):                      # sprocket holes (white squares)
            for ry in (margin, 2 * half - margin - self.hole_h):
                fl[ry:ry + self.hole_h, x:x + self.hole_w] = True
        rs = np.random.RandomState(self.seed)
        top = (2 * half - self.fh) // 2
        bd = max(1, self.edge)
        x = int(rs.randint(0, self.pitch))
        k = int(rs.randint(0, len(FILM_SCENES)))
        while x + self.fw < length - 4:
            if rs.rand() > 0.12:                                                       # now and then a blank frame
                fl[top:top + self.fh, x:x + self.fw] = self.frame(FILM_SCENES[k % len(FILM_SCENES)])
            fl[top - bd:top, x - bd:x + self.fw + bd] = True
            fl[top + self.fh:top + self.fh + bd, x - bd:x + self.fw + bd] = True
            fl[top - bd:top + self.fh + bd, x - bd:x] = True
            fl[top - bd:top + self.fh + bd, x + self.fw:x + self.fw + bd] = True
            x += self.pitch
            k += int(rs.randint(1, 4))
        return fl


def path(xs, ys):
    return xs, ys, np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])


def swag(x0, x1, ytop, sag, n=8000):
    """a strip hung between two points `ytop` high, sagging `sag` px in the middle"""
    xs = np.linspace(x0, x1, n)
    xc, hw = (x0 + x1) / 2, (x1 - x0) / 2
    return path(xs, ytop + sag * (1 - ((xs - xc) / hw) ** 2))


def lay(p, strip: Strip, shape, off=0):
    """lays a flat strip along a path (forward mapping: every film pixel lands on one canvas pixel) -> (values, covered)"""
    H, W = shape
    xs, ys, s = p
    flat = strip.flat(int(s[-1]) + off + 80)
    half = strip.half
    sv = np.arange(0, s[-1], 0.35)
    px, py = np.interp(sv, s, xs), np.interp(sv, s, ys)
    dx, dy = np.gradient(px), np.gradient(py)
    nrm = np.hypot(dx, dy)
    nx, ny = -dy / nrm, dx / nrm
    tv = np.arange(-half, half, 0.4)
    X = px[:, None] + nx[:, None] * tv[None, :]
    Y = py[:, None] + ny[:, None] * tv[None, :]
    si = np.clip((sv + off).astype(int), 0, flat.shape[1] - 1)
    ti = np.clip((tv + half).astype(int), 0, 2 * half - 1)
    val = flat[ti[None, :], si[:, None]]
    xi, yi = np.round(X).astype(int), np.round(Y).astype(int)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    v = np.zeros((H, W), bool)
    c = np.zeros((H, W), np.uint8)
    v[yi[ok], xi[ok]] = val[ok]
    c[yi[ok], xi[ok]] = 1
    c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))               # close the pinholes the forward mapping leaves
    return v, c.astype(bool)


def put(out, v, c, halo=0, haze=0.0):
    """draws a laid strip onto `out`: a black halo round it lifts it off the picture; `haze` thins it with blue noise (far strips)"""
    H, W = out.shape
    if halo:
        d = cv2.dilate(c.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * halo + 1, 2 * halo + 1))).astype(bool)
        out[d] = False
    if haze:
        n = np.tile(BLUE, (H // 128 + 1, W // 128 + 1))[:H, :W]
        v = v & (n < haze)
    out[c] = v[c]


def dust(out, seed, rows, count=170):
    """specks and a few scratches, like old film"""
    H, W = out.shape
    rs = np.random.RandomState(seed)
    for _ in range(count):
        x, y = int(rs.randint(0, W)), int(rs.randint(*rows))
        out[y:y + int(rs.randint(1, 3)), x:x + int(rs.randint(1, 3))] = True
    for _ in range(3):
        x, y0 = int(rs.randint(W // 10, W - W // 10)), int(rs.randint(rows[0], rows[0] + (rows[1] - rows[0]) // 3))
        for y in range(y0, y0 + int(rs.randint(120, 300)), 3):
            out[y:y + 2, x] = True


# ------------------------------------------------------------------------------------------------------------------------ the sheets
def hero() -> None:
    H, W = 1080, 1920
    base = render(HERO_SCENE, None, H, 5)
    base[:LETTERBOX] = False
    base[H - LETTERBOX:] = False
    out = base.copy()
    # far: small and hazy, seen through the night air
    put(out, *lay(swag(-120, 1010, 4, 215), Strip(26, 24, 1), (H, W), 5), haze=0.62)
    put(out, *lay(swag(1010, 2100, 4, 150), Strip(26, 24, 2), (H, W), 17), haze=0.62)
    # mid
    put(out, *lay(swag(-160, 900, 24, 330), Strip(58, 56, 3), (H, W), 0), halo=5)
    put(out, *lay(swag(900, 2080, 24, 120), Strip(58, 56, 4), (H, W), 37), halo=5)
    # near: big, in front of everything, cut by the frame
    put(out, *lay(swag(-430, 540, -70, 580), Strip(92, 88, 5), (H, W), 11), halo=8)
    t = np.linspace(0, 1, 6000)
    put(out, *lay(path(1745 + 95 * np.sin(t * 2.6) + 40 * t, -150 + t * 740), Strip(70, 66, 6), (H, W), 23), halo=7)
    dust(out, 21, (150, 930))
    save(out, "SuperHeroArt_1920x1080.png")
    save(out, "SuperHeroArt_3840x2160.png", 2)


def box() -> None:
    """a plain square: the moon, the sloop and the sea, nothing laid over it"""
    out = render(HERO_SCENE, (840, 110, 820, 820), 1080, 5)
    save(out, "StoreBoxArt_1080x1080.png")


def poster() -> None:
    H, W = 1080, 720                                                                  # drawn at half size, then doubled
    out = render(HERO_SCENE, (W / H, 0.705), H, 5).copy()
    put(out, *lay(swag(-60, 800, 2, 120), Strip(22, 20, 21), (H, W), 3), haze=0.62)
    put(out, *lay(swag(-120, 860, -14, 210), Strip(42, 40, 22), (H, W), 7), halo=4)
    t = np.linspace(0, 1, 5000)
    put(out, *lay(path(70 + 50 * np.sin(t * 3.2 + 0.4), -120 + t * 700), Strip(62, 58, 23), (H, W), 19), halo=6)
    dust(out, 24, (30, H - 30), 70)
    save(out, "StorePosterArt_1440x2160.png", 2)


def main() -> int:
    hero()
    box()
    poster()
    return 0


if __name__ == "__main__":
    sys.exit(main())
