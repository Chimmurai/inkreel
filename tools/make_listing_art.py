"""Promotional art for the Microsoft Store page, made by the app itself: frames of the source scenes (tools/make_listing_scenes.py)
are rendered through InkReel (Engraving, Classic palette, native size, so every pixel stays a crisp square) and laid on a film strip
over a dithered halo. No text, as the Store asks for art.

    python tools/make_listing_art.py     ->  store/listing_assets/SuperHeroArt_1920x1080.png, StoreBoxArt_1080x1080.png,
                                              StoreBoxArt_1080x1080_centered.png, StorePosterArt_1440x2160.png

Needs the scenes first and the project environment (it calls `python -m benvid`)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = os.path.join(ROOT, "store", "listing_assets", "scenes")
OUT = os.path.join(ROOT, "store", "listing_assets")
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0


def bayer(shape):
    h, w = shape
    return np.tile(BAYER, (h // 4 + 1, w // 4 + 1))[:h, :w]


def render(scene: str, crop: tuple[float, float] | None, height: int, method: int = 5) -> np.ndarray:
    """a scene through the app at exactly `height` px tall; returns a bool array (True = paper/white)"""
    src = Image.open(os.path.join(SCENES, scene)).convert("RGB")
    if crop:                                                  # (aspect, x-centre 0..1) crop of the 1920x1080 source
        aspect, cx = crop
        w = int(round(src.height * aspect))
        x0 = int(np.clip(cx * src.width - w / 2, 0, src.width - w))
        src = src.crop((x0, 0, x0 + w, src.height))
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.png"), os.path.join(d, "b.png")
        src.save(a)
        subprocess.run([sys.executable, "-m", "benvid", a, b, "--method", str(method), "--size", f"x{height}"],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return np.asarray(Image.open(b).convert("L")) > 127


def halo(h, w, cx, cy, rx, ry, power=1.6, floor=0.0, gain=0.9):
    """bool field of white dots, densest around (cx, cy): the light that spills out of the film"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    level = np.clip(1 - d, 0, 1) ** power * gain + floor
    return level > bayer((h, w))


def corners(canvas, x0, y0, x1, y1, arm=70, th=6):
    """viewfinder brackets, white, like the corners round the plate on the site"""
    for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        xs = sorted((x, x + sx * arm))
        ys = sorted((y - th // 2 if sy == 1 else y - th // 2, y + th - th // 2))
        canvas[ys[0]:ys[1], xs[0]:xs[1]] = True
        ys2 = sorted((y, y + sy * arm))
        xs2 = sorted((x - th // 2, x + th - th // 2))
        canvas[ys2[0]:ys2[1], xs2[0]:xs2[1]] = True


def sprockets_h(canvas, y, x0, x1, hole=(22, 30), pitch=58):
    for x in range(x0 + 14, x1 - hole[0], pitch):
        canvas[y:y + hole[1], x:x + hole[0]] = False


def sprockets_v(canvas, x, y0, y1, hole=(30, 22), pitch=58):
    for y in range(y0 + 14, y1 - hole[1], pitch):
        canvas[y:y + hole[1], x:x + hole[0]] = False


def frame_into(canvas, img, x, y, border=5):
    h, w = img.shape
    canvas[y - border:y + h + border, x - border:x + w + border] = False
    canvas[y:y + h, x:x + w] = img


def save(canvas, name):
    Image.fromarray((canvas * 255).astype(np.uint8)).convert("RGB").save(os.path.join(OUT, name))
    print(name, canvas.shape[::-1])


def hero():
    W, H = 1920, 1080
    c = halo(H, W, W / 2, H / 2, 1250, 640, power=1.3, gain=0.95)
    sh, fw = 330, 586                                           # frame height / width (16:9)
    strip_h = sh + 2 * 74
    y0 = (H - strip_h) // 2
    c[y0:y0 + strip_h, :] = True                                # white strip
    sprockets_h(c, y0 + 20, 0, W)
    sprockets_h(c, y0 + strip_h - 20 - 30, 0, W)
    scenes = [("05_sea_moon.png", 5), ("01_lake_dawn.png", 5), ("02_night_city.png", 5), ("04_lighthouse.png", 5), ("03_still_life.png", 5)]
    gap = 40
    x = -fw // 2 - 20
    for i, (sc, m) in enumerate(scenes):
        if x + fw > 0 and x < W:
            img = render(sc, None, sh, m)
            x0 = max(x, 0)
            visible = img[:, x0 - x:W - x]
            frame_into(c, visible if x < 0 or x + fw > W else img, x0, y0 + 74)
        x += fw + gap
    corners(c, 110, 90, W - 110, H - 90)
    save(c, "SuperHeroArt_1920x1080.png")


def box():
    W = H = 1080
    c = halo(H, W, W / 2, H / 2, 780, 780, power=1.2, gain=0.95)
    s = 640
    img = render("05_sea_moon.png", (1.0, 0.5), s, 5)
    x = (W - s) // 2
    c[x - 52:x + s + 52, x - 52:x + s + 52] = True              # white mount
    frame_into(c, img, x, x, border=6)
    corners(c, 100, 100, W - 100, H - 100, arm=84, th=7)
    save(c, "StoreBoxArt_1080x1080.png")
    save(c, "StoreBoxArt_1080x1080_centered.png")


def poster():
    W, H = 1440, 2160
    c = halo(H, W, W / 2, H / 2, 900, 1250, power=1.2, gain=0.95)
    fh = 480
    fw = int(fh * 4 / 3)
    sw = fw + 2 * 92
    sx = (W - sw) // 2
    c[:, sx:sx + sw] = True
    sprockets_v(c, sx + 24, 0, H)
    sprockets_v(c, sx + sw - 24 - 30, 0, H)
    scenes = [("01_lake_dawn.png", (4 / 3, 0.5)), ("03_still_life.png", (4 / 3, 0.55)), ("04_lighthouse.png", (4 / 3, 0.4)), ("06_dunes_aerial.png", (4 / 3, 0.5))]
    gap = 34
    total = len(scenes) * fh + (len(scenes) - 1) * gap
    y = (H - total) // 2
    for sc, cr in scenes:
        frame_into(c, render(sc, cr, fh, 5 if "dunes" not in sc else 4), sx + 92, y)
        y += fh + gap
    corners(c, 80, 80, W - 80, H - 80, arm=80, th=7)
    save(c, "StorePosterArt_1440x2160.png")


def main() -> int:
    hero()
    box()
    poster()
    return 0


if __name__ == "__main__":
    sys.exit(main())
