"""Source pictures for the Microsoft Store screenshots and art: seven procedurally drawn scenes (our own, no third-party photos).

    python tools/make_listing_scenes.py          ->  store/listing_assets/scenes/*.png  (1920x1080, grayscale)

Each scene is built to show something InkReel is good at: soft skies, haze, glitter, hard shadows, foliage, rippled water, fine texture,
and a moonlit sea with a Milky Way (07, the frame of the Store hero).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "store", "listing_assets", "scenes")
W, H = 1920, 1080
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def fbm(h: int, w: int, seed: int, base: int = 4, octaves: int = 6, stretch: float = 1.0) -> np.ndarray:
    """smooth noise 0..1; `stretch` > 1 makes it wider than tall (cloud streaks)"""
    r = np.random.RandomState(seed)
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gw = max(2, int(base * 2 ** o))
        gh = max(2, int(base * 2 ** o * stretch))
        g = r.rand(gh, gw).astype(np.float32)
        acc += amp * np.asarray(Image.fromarray(g, "F").resize((w, h), Image.BICUBIC))
        tot += amp
        amp *= 0.5
    acc /= tot
    return (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def blur(a: np.ndarray, r: float) -> np.ndarray:
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))
    return np.asarray(im).astype(np.float32) / 255


def ridge(seed: int, base_y: float, amp: float, rough: float = 5) -> np.ndarray:
    n = fbm(1, W, seed, base=3, octaves=7)[0] if False else np.asarray(Image.fromarray(
        np.random.RandomState(seed).rand(1, 4).astype(np.float32), "F").resize((W, 1), Image.BICUBIC))[0]
    f = np.zeros(W, np.float32)
    r = np.random.RandomState(seed)
    a = 1.0
    for o in range(7):
        g = r.rand(3 + 2 ** (o + 1)).astype(np.float32)
        f += a * np.interp(np.linspace(0, len(g) - 1, W), np.arange(len(g)), g)
        a *= 1 / rough * 2.2 if False else 0.55
    f = (f - f.min()) / (f.max() - f.min())
    return base_y - amp * f


def finish(img: np.ndarray, name: str, grain: float = 0.012, soft: float = 0.7) -> None:
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(soft))
    a = np.asarray(im).astype(np.float32) / 255
    a += (np.random.RandomState(7).randn(H, W).astype(np.float32)) * grain
    out = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    os.makedirs(OUT, exist_ok=True)
    out.save(os.path.join(OUT, name + ".png"))
    print(name)


# ---------------------------------------------------------------------------------------------------------------- 1. mountain lake
def lake() -> None:
    hor = 700
    sky = 0.22 + 0.68 * smooth(0, hor, YY) ** 1.3
    cl = fbm(H, W, 11, base=3, octaves=6, stretch=3.2)
    sky = sky * (0.78 + 0.34 * smooth(0.45, 0.8, cl)) - 0.18 * smooth(0.62, 0.9, cl) * (1 - smooth(0, hor, YY))
    sx, sy = 1260, 560
    d = np.hypot(XX - sx, (YY - sy) * 1.05)
    sky += 0.55 * np.exp(-(d / 260) ** 2) + 0.6 * smooth(46, 40, d)
    img = sky.copy()
    haze = 0.86
    layers = [(2, 520, 150, 0.72), (5, 560, 190, 0.55), (9, 620, 170, 0.38), (14, 690, 120, 0.2)]
    for i, (seed, by, amp, tone) in enumerate(layers):
        top = ridge(seed, by, amp)
        mask = (YY >= top[None, :]).astype(np.float32)
        mask = blur(mask, 1.0)
        mist = smooth(top.min() - 20, hor, YY)
        body = tone + (haze - tone) * (0.25 + 0.7 * mist) * (0.6 if i > 1 else 1.0)
        tex = fbm(H, W, 30 + i, base=14, octaves=5)
        body = body * (0.88 + 0.24 * tex)
        # light on the slopes facing the sun
        slope = np.gradient(top)
        lit = np.clip(-slope * 0.06, -0.5, 0.5)[None, :] * smooth(top.min(), top.min() + 160, YY) * (1 - mist)
        body = body + lit * 0.5
        img = img * (1 - mask) + body * mask
    # pines on the nearest ridge
    pim = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(pim)
    top = ridge(14, 690, 120)
    rs = np.random.RandomState(4)
    for _ in range(170):
        x = int(rs.randint(0, W))
        yb = int(top[x]) + int(rs.randint(10, 60))
        h = int(rs.randint(40, 130))
        wd = h * 0.22
        for k in range(5):
            yk = yb - h * (k / 5.0)
            wk = wd * (1.0 - k / 5.5)
            dr.polygon([(x - wk, yk), (x + wk, yk), (x, yk - h * 0.3)], fill=255)
    pm = blur(np.asarray(pim).astype(np.float32) / 255, 0.6)
    img = img * (1 - pm) + 0.07 * pm
    # lake: reflect the sky and the ridges with ripples
    up = img[:hor].copy()
    refl = up[::-1][: H - hor]
    if refl.shape[0] < H - hor:
        refl = np.vstack([refl, np.repeat(refl[-1:], H - hor - refl.shape[0], axis=0)])
    ripple = fbm(H - hor, W, 21, base=24, octaves=4, stretch=14)
    rows = np.arange(H - hor)[:, None]
    shift = ((ripple - 0.5) * (6 + rows * 0.09)).astype(np.float32)
    xi = np.clip((np.arange(W)[None, :] + shift).astype(np.int32), 0, W - 1)
    refl = np.take_along_axis(refl, xi, axis=1)
    streak = smooth(0.45, 0.7, fbm(H - hor, W, 22, base=6, octaves=5, stretch=26))
    refl = refl * (0.78 + 0.22 * streak) - 0.06
    # sun path on the water
    wat = 1 - 0.45 * smooth(0, H - hor, rows)
    glow = np.exp(-(((np.arange(W)[None, :] - sx) / (60 + rows * 0.55)) ** 2)) * (0.35 + 0.65 * streak)
    refl = refl * wat + 0.5 * glow
    img[hor:] = refl
    img[hor - 2:hor + 2] = blur(img, 1.0)[hor - 2:hor + 2]
    # dark reeds in the foreground
    rim = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(rim)
    for _ in range(1100):
        x = int(rs.randint(-20, W + 20))
        bias = 1 - abs(x - 330) / 700 if x < 1000 else 1 - abs(x - 1700) / 600
        hgt = int(rs.rand() * 260 * max(bias, 0.1)) + 20
        if rs.rand() > max(bias, 0.15):
            continue
        yb = H + 6
        lean = rs.randn() * 26
        dr.line([(x, yb), (x + lean * 0.4, yb - hgt * 0.55), (x + lean, yb - hgt)], fill=255, width=int(rs.randint(2, 5)))
    rm = blur(np.asarray(rim).astype(np.float32) / 255, 0.7)
    img = img * (1 - rm) + 0.04 * rm
    finish(img, "01_lake_dawn")


# ---------------------------------------------------------------------------------------------------------------- 2. night city
def city() -> None:
    hor = 700
    img = 0.05 + 0.5 * smooth(0, hor, YY) ** 2.2
    rs = np.random.RandomState(5)
    st = (rs.rand(H, W) > 0.9985).astype(np.float32) * (1 - smooth(0, hor * 0.8, YY)) * rs.rand(H, W)
    img = img + blur(st * 1.6, 0.6)
    # moon with maria
    mx, my, mr = 1420, 250, 120
    d = np.hypot(XX - mx, YY - my)
    disc = smooth(mr + 1, mr - 1, d)
    nz = fbm(H, W, 3, base=5, octaves=5)
    moon = 0.9 - 0.34 * smooth(0.5, 0.75, nz)
    z = np.sqrt(np.clip(1 - (d / mr) ** 2, 0, 1))
    moon = moon * (0.7 + 0.3 * z)
    img = img * (1 - disc) + moon * disc + 0.3 * np.exp(-((d - mr) / 140) ** 2) * (d > mr)
    # clouds crossing the moon
    cl = smooth(0.62, 0.8, fbm(H, W, 8, base=3, octaves=5, stretch=4)) * smooth(480, 120, YY) * 0.5
    img = img * (1 - cl) + 0.28 * cl
    # skyline layers
    for li, (seed, base, hmin, hmax, tone, lit) in enumerate([(1, 700, 110, 330, 0.17, 0.0), (2, 700, 160, 470, 0.08, 0.22), (3, 700, 120, 560, 0.025, 0.55)]):
        r = np.random.RandomState(seed)
        bim = np.zeros((H, W), np.float32)
        win = np.zeros((H, W), np.float32)
        x = -20
        while x < W + 20:
            bw = int(r.randint(70, 190)) if li < 2 else int(r.randint(100, 260))
            bh = int(r.randint(hmin, hmax))
            top = base - bh
            bim[top:base, max(x, 0):x + bw] = 1
            if r.rand() > 0.55:   # antenna / stepped roof
                aw = bw // 5
                bim[top - int(r.randint(25, 80)):top, x + bw // 2 - aw // 2:x + bw // 2 + aw // 2] = 1
            if lit > 0:
                cell_w, cell_h = (14, 22) if li == 2 else (9, 14)
                ys = np.arange(top + 14, base - 6, cell_h)
                xs = np.arange(x + 10, x + bw - 10, cell_w)
                for yy_ in ys:
                    for xx_ in xs:
                        if 0 <= xx_ < W and r.rand() < lit:
                            win[yy_:yy_ + cell_h - 9, xx_:xx_ + cell_w - 6] = 0.55 + 0.4 * r.rand()
            x += bw + int(r.randint(0, 24))
        bim = blur(bim, 0.8)
        g = np.clip((YY - (base - hmax)) / hmax, 0, 1)
        body = tone + 0.1 * (1 - g) * (1 if li == 0 else 0.3)
        img = img * (1 - bim) + body * bim
        img = np.where(win > 0, img * (1 - bim) + win * bim, img)
    # river with reflections and light streaks
    base = 700
    up = img[:base]
    refl = up[::-1][: H - base]
    rows = np.arange(H - base)[:, None]
    rip = fbm(H - base, W, 41, base=40, octaves=3, stretch=30)
    shift = ((rip - 0.5) * (8 + rows * 0.12)).astype(np.float32)
    xi = np.clip((np.arange(W)[None, :] + shift).astype(np.int32), 0, W - 1)
    refl = np.take_along_axis(refl[:H - base], xi, axis=1)
    refl = blur(refl, 1.2) * 0.62
    streak = smooth(0.55, 0.75, fbm(H - base, W, 42, base=5, octaves=5, stretch=40))
    refl = refl * (0.7 + 0.5 * streak)
    refl = refl * (1 - 0.5 * smooth(0, H - base, rows))
    img[base:] = refl
    # a bridge rail and lamp posts in the foreground
    fim = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(fim)
    dr.rectangle([0, 1000, W, H], fill=255)
    dr.line([(0, 1000), (W, 1000)], fill=255, width=10)
    for i in range(-1, 8):
        x = 140 + i * 300
        dr.line([(x, 1000), (x, 880)], fill=255, width=7)
        dr.line([(x, 880), (x + 40, 872)], fill=255, width=5)
    fm = blur(np.asarray(fim).astype(np.float32) / 255, 0.7)
    img = img * (1 - fm) + 0.02 * fm
    for i in range(-1, 8):
        x = 140 + i * 300 + 40
        l = np.exp(-(((XX - x) ** 2 + (YY - 878) ** 2) / 900))
        img = img + 0.8 * l + 0.18 * np.exp(-(((XX - x) ** 2 + (YY - 878) ** 2) / 14000))
    finish(img, "02_night_city", grain=0.014)


# ---------------------------------------------------------------------------------------------------------------- 3. still life
def still() -> None:
    wall_y = 640
    L = np.array([-0.62, 0.78], np.float32)   # light from upper left (x, z)
    img = np.full((H, W), 0.5, np.float32)
    wall = 0.38 + 0.12 * smooth(0, 700, 700 - YY) + 0.0 * XX
    table = 0.62 - 0.22 * smooth(wall_y, H, YY) + 0.0
    img = np.where(YY < wall_y, wall, table)
    # window light on the wall: a skewed patch split by mullions
    qs = (XX - 520) + (YY - 120) * 0.38
    inw = smooth(0, 6, qs) * smooth(0, 6, 420 - qs) * smooth(60, 70, YY) * smooth(640, 560, YY)
    bars = 1 - 0.9 * ((np.abs(((qs - 210) % 420) - 0) < 11) | (np.abs(YY - 330) < 10)).astype(np.float32)
    inw = inw * np.clip(bars, 0, 1)
    inw = blur(inw, 5)
    img = img + 0.5 * inw * (0.85 + 0.15 * fbm(H, W, 51, base=6, octaves=4))
    # the same light falling across the table
    qt = (XX - 700) + (YY - wall_y) * -1.4
    pt = smooth(0, 30, qt) * smooth(0, 30, 900 - qt) * smooth(wall_y, wall_y + 40, YY)
    img = img + 0.14 * blur(pt, 8)
    # table grain
    wood = fbm(H, W, 52, base=3, octaves=5, stretch=14)
    img = img + 0.09 * (wood - 0.5) * smooth(wall_y, wall_y + 30, YY)

    shadows = np.zeros((H, W), np.float32)

    def revolve(cx, y_top, y_bot, prof, tone, spec=0.0):
        nonlocal img, shadows
        ys = np.arange(y_top, y_bot)
        t = (ys - y_top) / max(1, (y_bot - y_top))
        r = prof(t)
        m = np.zeros((H, W), np.float32)
        lam = np.zeros((H, W), np.float32)
        for yi, ri in zip(ys, r):
            if ri < 1:
                continue
            xs = np.arange(int(cx - ri) - 1, int(cx + ri) + 2)
            nx = np.clip((xs - cx) / ri, -1, 1)
            nzv = np.sqrt(np.clip(1 - nx * nx, 0, 1))
            lm = np.clip(nx * L[0] + nzv * L[1], 0, 1)
            lm = 0.1 + 0.9 * lm + 0.12 * (nzv < 0.35) * (1 - lm)   # rim light
            inside = np.abs(xs - cx) <= ri
            m[yi, xs[inside]] = 1
            lam[yi, xs[inside]] = lm[inside] + spec * np.exp(-(((nx[inside] + 0.5) / 0.09) ** 2)) * 0.7
        m = blur(m, 0.8)
        lam = blur(lam, 0.8) / np.maximum(blur(m, 0.8), 1e-3) * m
        body = tone * (0.35 + 0.8 * lam)
        img = img * (1 - m) + np.clip(body, 0, 1) * m
        # cast shadow along the table
        sh = np.zeros((H, W), np.float32)
        for yi, ri in zip(ys, r):
            h = y_bot - yi
            xo = int(h * 0.82)
            yy_ = int(y_bot + h * 0.22)
            if yy_ >= H:
                continue
            x0, x1 = int(cx - ri + xo), int(cx + ri + xo)
            sh[max(yy_ - 2, 0):yy_ + 2, max(x0, 0):max(min(x1, W), 0)] = 1
        shadows = np.maximum(shadows, sh)
        # contact shadow under the object
        c = np.exp(-(((XX - cx) / (r.max() * 1.1)) ** 2 + ((YY - y_bot - 4) / 14) ** 2))
        shadows = np.maximum(shadows, c)

    # shadows must lie under objects: draw objects, then multiply shadow in the table region before objects paint
    objs = []
    def circ(t, c, rr):
        return np.sqrt(np.clip(1 - ((t - c) / rr) ** 2, 0, 1))

    def vase(t):
        t = np.clip(t, 0, 1)
        return 22 + 8 * smooth(0.06, 0.0, t) + 66 * smooth(0.12, 0.5, t) * (1 - 0.45 * smooth(0.72, 1.0, t))

    def pear(t):
        t = np.clip(t, 0, 1)
        return np.maximum(74 * circ(t, 0.66, 0.34), 50 * circ(t, 0.31, 0.31) * (0.6 + 0.4 * smooth(0.05, 0.4, t)))

    def apple(t):
        return 84 * circ(np.clip(t, 0, 1), 0.5, 0.5)

    objs = [(1180, 280, 800, vase, 0.78, 0.35), (810, 590, 820, pear, 0.72, 0.1), (1560, 620, 800, apple, 0.55, 0.05)]    # first pass: shadows only
    img0 = img.copy()
    for cx, yt, yb, prof, tone, spec in objs:
        revolve(cx, yt, yb, prof, tone, spec)
    img_objs = img
    img = img0 * (1 - 0.62 * blur(shadows, 9) * smooth(wall_y - 4, wall_y + 10, YY))
    for cx, yt, yb, prof, tone, spec in objs:
        # repaint objects over the shadowed table
        ys = np.arange(yt, yb)
        t = (ys - yt) / max(1, (yb - yt))
        r = prof(t)
        m = np.zeros((H, W), np.float32)
        lam = np.zeros((H, W), np.float32)
        for yi, ri in zip(ys, r):
            if ri < 1:
                continue
            xs = np.arange(int(cx - ri) - 1, int(cx + ri) + 2)
            nx = np.clip((xs - cx) / ri, -1, 1)
            nzv = np.sqrt(np.clip(1 - nx * nx, 0, 1))
            lm = np.clip(nx * L[0] + nzv * L[1], 0, 1)
            lm = 0.1 + 0.9 * lm + 0.12 * (nzv < 0.35) * (1 - lm)
            inside = np.abs(xs - cx) <= ri
            m[yi, xs[inside]] = 1
            lam[yi, xs[inside]] = lm[inside] + spec * np.exp(-(((nx[inside] + 0.5) / 0.09) ** 2)) * 0.7
        m = blur(m, 0.8)
        lam = blur(lam, 0.8) / np.maximum(blur(m, 0.8), 1e-3) * m
        body = np.clip(tone * (0.35 + 0.8 * lam), 0, 1)
        # texture: stripes on the vase, dimples on the fruit
        if cx == 1180:
            body = body * (1 - 0.7 * ((np.abs(YY - 470) < 5) | (np.abs(YY - 640) < 5)).astype(np.float32))
        else:
            body = body * (0.92 + 0.12 * fbm(H, W, 60 + cx % 7, base=30, octaves=3))
        img = img * (1 - m) + body * m
    # a sprig leaning on the vase
    sp = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(sp)
    r = np.random.RandomState(9)
    pts = [(1210, 290)]
    for k in range(1, 9):
        pts.append((pts[-1][0] + 18 + r.randn() * 4, pts[-1][1] - 38 - r.rand() * 10))
    dr.line(pts, fill=255, width=4)
    for k, (px, py) in enumerate(pts[1:], 1):
        for s in (-1, 1):
            dr.ellipse([px + s * 26 - 22, py - 9, px + s * 26 + 22, py + 9], fill=255)
    spm = blur(np.asarray(sp).astype(np.float32) / 255, 0.7)
    img = img * (1 - spm) + 0.14 * spm
    finish(img, "03_still_life", grain=0.012)


# ---------------------------------------------------------------------------------------------------------------- 4. oak on a hill
def oak() -> None:
    """a lighthouse on a cliff in a storm: a beam cutting through cloud, hard silhouettes, breaking sea"""
    hor = 640
    cl = fbm(H, W, 71, base=3, octaves=7, stretch=2.2)
    img = 0.16 + 0.5 * smooth(0, hor, YY) ** 1.4
    img = img * (0.45 + 0.9 * cl)
    img += 0.22 * smooth(0.55, 0.85, fbm(H, W, 72, base=5, octaves=5, stretch=3)) * smooth(hor, 200, YY)
    lx, ly = 560, 330           # the lamp
    # the beam: a soft wedge to the right, brighter where it meets clouds
    ang = np.arctan2(YY - ly, XX - lx)
    dist = np.hypot(XX - lx, YY - ly)
    beam = smooth(0.11, 0.0, np.abs(ang - 0.08)) * (XX > lx) * np.exp(-dist / 1500)
    img = img + 0.85 * beam * (0.55 + 0.9 * cl)
    img += 0.9 * np.exp(-(dist / 38) ** 2) + 0.25 * np.exp(-(dist / 190) ** 2)
    # sea
    rows = (YY - hor).clip(1, None)
    persp = 1.0 / (rows / 220 + 0.2)
    wv = fbm(H, W, 73, base=70, octaves=4, stretch=0.1)
    crest = 0.5 + 0.5 * np.sin(persp * 26 + (wv - 0.5) * 16 + np.sin(XX / (140 / persp.clip(0.3, None))))
    foam = smooth(0.8, 0.96, crest) * smooth(0.45, 0.7, fbm(H, W, 74, base=40, octaves=4, stretch=0.2))
    sea_img = 0.06 + 0.12 * crest * (1 - smooth(hor, H, YY)) + 0.5 * foam * (0.35 + 0.65 * smooth(hor, H, YY))
    sea_img = 0.05 + 0.07 * (0.5 + 0.5 * np.sin(persp * 30 + (wv - 0.5) * 3.0)) + 0.5 * foam * 0.12
    sea_img += 0.4 * np.exp(-(((XX - 1300) / (230 + rows * 0.6)) ** 2)) * smooth(0.4, 0.9, 0.5 + 0.5 * np.sin(persp * 24 + (wv - 0.5) * 2.5)) * smooth(hor + 260, hor, YY)
    img = np.where(YY >= hor, sea_img, img)
    img[hor - 1:hor + 2] = blur(img, 1.0)[hor - 1:hor + 2]
    # cliff
    xs = np.arange(W, dtype=np.float32)
    top = 900 - 330 * np.exp(-((xs - 420) / 330) ** 2) - 60 * np.exp(-((xs - 780) / 200) ** 2) + 26 * np.sin(xs / 61) * np.exp(-((xs - 420) / 500) ** 2)
    top += 30 * (np.interp(xs, np.linspace(0, W, 40), np.random.RandomState(8).rand(40)) - 0.5)
    cmask = blur((YY >= top[None, :]).astype(np.float32), 1.0)
    rock = fbm(H, W, 75, base=18, octaves=5)
    lit = np.clip(1 - (XX - 300) / 900, 0, 1)
    cbody = 0.05 + 0.14 * smooth(0.35, 0.8, rock) * (0.35 + 0.65 * lit)
    img = img * (1 - cmask) + cbody * cmask
    # the tower: a cylinder, lit from the right (the beam side), with bands
    tw0, tw1, tb = 520, 600, int(top[560])
    tt = 405
    tower = np.zeros((H, W), np.float32)
    for y in range(tt, tb + 20):
        f = (y - tt) / (tb + 20 - tt)
        half = 26 + 22 * f
        x0, x1 = int(560 - half), int(560 + half)
        tower[y, x0:x1] = 1
    tower = blur(tower, 0.8)
    u = np.clip((XX - 560) / (26 + 22 * np.clip((YY - tt) / (tb + 20 - tt), 0, 1)), -1, 1)
    cyl = 0.18 + 0.7 * smooth(-1, 1, u * 0.9 + 0.1) ** 1.3
    band = ((((YY - tt) // 62).astype(np.int32)) % 2 == 0)
    tone = cyl * np.where(band, 1.0, 0.18)
    img = img * (1 - tower) + tone * tower
    # lantern room, gallery and roof
    rm = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(rm)
    dr.rectangle([524, 350, 596, 404], fill=255)
    dr.rectangle([508, 398, 612, 410], fill=255)
    dr.polygon([(516, 350), (604, 350), (560, 304)], fill=255)
    dr.line([(560, 304), (560, 280)], fill=255, width=3)
    rmm = blur(np.asarray(rm).astype(np.float32) / 255, 0.7)
    img = img * (1 - rmm) + 0.04 * rmm
    lamp = Image.new("L", (W, H), 0)
    ImageDraw.Draw(lamp).rectangle([532, 358, 588, 396], fill=255)
    lm = blur(np.asarray(lamp).astype(np.float32) / 255, 0.8)
    img = img * (1 - lm) + 0.98 * lm
    img += 0.35 * np.exp(-(((XX - 560) ** 2 + (YY - 377) ** 2) / 4200))
    # surf at the foot of the cliff
    finish(img, "04_lighthouse", grain=0.013)


# ---------------------------------------------------------------------------------------------------------------- 5. sea and moon
def sea() -> None:
    hor = 520
    img = 0.03 + 0.42 * smooth(0, hor, YY) ** 2.0
    cl = fbm(H, W, 101, base=3, octaves=6, stretch=4)
    img = img * (0.8 + 0.4 * smooth(0.45, 0.8, cl)) * (1 - 0.4 * smooth(0.6, 0.85, cl) * smooth(hor, 80, YY))
    rs = np.random.RandomState(3)
    st = (rs.rand(H, W) > 0.9988).astype(np.float32) * (1 - smooth(0, hor * 0.9, YY)) * rs.rand(H, W)
    img += blur(st * 1.7, 0.6)
    mx, my, mr = 980, 270, 82
    d = np.hypot(XX - mx, YY - my)
    disc = smooth(mr + 1, mr - 1, d)
    nz = fbm(H, W, 102, base=6, octaves=4)
    img = img * (1 - disc) + (0.95 - 0.18 * smooth(0.5, 0.7, nz)) * disc
    img += 0.35 * np.exp(-(d / 210) ** 2) + 0.12 * np.exp(-(d / 520) ** 2)
    # sea: perspective waves; the glitter column is the specular reflection of the moon
    rows = (YY - hor).clip(1, None)
    persp = 1.0 / (rows / 200 + 0.18)
    sea_mask = YY >= hor
    wave = fbm(H, W, 103, base=60, octaves=4, stretch=0.1)
    ph = rows * 0.0
    crest = 0.5 + 0.5 * np.sin(persp * 28 + (wave - 0.5) * 18 + 0.8 * np.sin(XX / (160 / persp.clip(0.3, None)) + persp * 2))
    spec_w = 40 + rows * 0.55
    glit = np.exp(-(((XX - mx) + (wave - 0.5) * 120 * persp.clip(0, 2)) / spec_w) ** 2)
    bright = glit * smooth(0.55, 0.92, crest)
    sea_base = 0.045 + 0.1 * smooth(hor, hor + 120, YY) * (1 - smooth(hor + 120, H, YY)) * 0 + 0.05 * (1 - smooth(hor, H, YY))
    sea_img = sea_base * (0.7 + 0.5 * crest) + 0.95 * bright + 0.2 * np.exp(-(((XX - mx) / (200 + rows * 0.8)) ** 2)) * crest
    img = np.where(sea_mask, sea_img, img)
    img[hor - 1:hor + 2] = blur(img, 1.0)[hor - 1:hor + 2]
    # rocks
    rm = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(rm)
    r = np.random.RandomState(6)

    def rock(x0, x1, ytop, seed):
        rr = np.random.RandomState(seed)
        n = 26
        xs = np.linspace(x0, x1, n)
        ys = [H + 20] + [ytop + rr.rand() * 90 + 140 * (abs(i - n / 2) / (n / 2)) ** 1.4 for i in range(1, n - 1)] + [H + 20]
        pts = [(x0, H + 20)] + [(x, y) for x, y in zip(xs[1:-1], ys[1:-1])] + [(x1, H + 20)]
        dr.polygon(pts, fill=255)

    rock(-40, 520, 800, 1)
    rock(1450, W + 40, 880, 2)
    rm_ = blur(np.asarray(rm).astype(np.float32) / 255, 1.2)
    ed = np.clip(blur(rm_, 4) - rm_, 0, 1) * 1.0
    img = img * (1 - rm_) + 0.012 * rm_ + 0.0 * ed
    finish(img, "05_sea_moon", grain=0.014)


# ---------------------------------------------------------------------------------------------------------------- 6. dunes
def dunes() -> None:
    """dunes from above: asymmetric crests with the sun low on the left, so every crest throws a hard edge of shadow"""
    warp = fbm(H, W, 111, base=3, octaves=5)
    warp2 = fbm(H, W, 112, base=7, octaves=4)
    ph = XX * 0.0062 + YY * 0.0016 + 2.6 * warp + 0.45 * warp2
    u = ph % 1.0
    h = np.where(u < 0.78, (u / 0.78) ** 1.2, ((1 - u) / 0.22) ** 1.5)       # long windward slope, steep lee
    h = h * (0.55 + 0.65 * fbm(H, W, 113, base=2, octaves=3))
    rip = 0.5 + 0.5 * np.sin(ph * 60 + 3 * fbm(H, W, 114, base=20, octaves=3))
    h = h + 0.004 * rip
    gx = np.gradient(blur(h, 2.0), axis=1) * 13
    gy = np.gradient(blur(h, 2.0), axis=0) * 13
    lam = np.clip(0.55 - 1.5 * gx + 0.2 * gy, 0, 1)
    img = 0.07 + 0.88 * lam ** 0.9
    img = img * (0.94 + 0.12 * rip) * (0.85 + 0.25 * fbm(H, W, 115, base=3, octaves=3))
    # a line of footprints crossing the slope
    fm = np.zeros((H, W), np.float32)
    for k in range(34):
        t = k / 33
        x = 140 + t * 1650
        y = 900 - 520 * t + 40 * np.sin(t * 6) + (14 if k % 2 else -14)
        fm += np.exp(-(((XX - x) / 9) ** 2 + ((YY - y) / 14) ** 2))
    img = img * (1 - 0.55 * np.clip(fm, 0, 1))
    finish(img, "06_dunes_aerial", grain=0.011)


# ---------------------------------------------------------------------------------------------------------------- 7. crescent over the sea (the cinematic frame)
def _star_field(hor: int, band: np.ndarray) -> np.ndarray:
    """dense small stars, thicker inside the galactic band, a few bright ones with a glow and a glint"""
    rs = np.random.RandomState(9)
    sky = 1 - smooth(hor * 0.5, hor * 0.97, YY)
    dens = (0.0022 + 0.0085 * band ** 1.5) * sky
    pts = rs.rand(H, W) < dens
    mag = rs.rand(H, W) ** 3.4
    st = np.where(pts, 0.32 + 0.68 * mag, 0).astype(np.float32)
    img = blur(st * 2.2, 0.5)
    ys, xs = np.nonzero(pts & (mag > 0.86))
    for y, x in zip(ys, xs):
        if y < 12 or x < 12 or y > hor - 40 or x > W - 13:
            continue
        r = 1.4 + 1.8 * float(mag[y, x])
        g = np.exp(-(((XX[y - 11:y + 12, x - 11:x + 12] - x) ** 2 + (YY[y - 11:y + 12, x - 11:x + 12] - y) ** 2) / (r * r)))
        img[y - 11:y + 12, x - 11:x + 12] += 0.45 * g
        if mag[y, x] > 0.975:                                      # the brightest get a small cross
            ln = int(5 + 9 * mag[y, x])
            img[y, max(0, x - ln):x + ln + 1] += 0.3
            img[max(0, y - ln):y + ln + 1, x] += 0.3
    return img


def _draw_boat(img: np.ndarray, sx: int, sy: int, s: float = 1.0) -> np.ndarray:
    k = 3
    pad = 200
    w = h = int(2 * pad * k)
    val = Image.new("L", (w, h), 0)
    msk = Image.new("L", (w, h), 0)
    dv, dm = ImageDraw.Draw(val), ImageDraw.Draw(msk)
    c = pad * k

    def P(x, y):
        return (c + x * s * k, c + y * s * k)
    # hull: a curved dark shape with a slight sheer
    hull = [P(-52, -4), P(-40, 9), P(-8, 14), P(34, 11), P(56, -6), P(30, -2), P(-12, -2)]
    dm.polygon(hull, fill=255)
    dv.polygon(hull, fill=8)
    # main sail (lit), jib (a little darker)
    main = [P(3, -6), P(3, -118), P(46, -10)]
    jib = [P(-4, -6), P(-4, -92), P(-46, -9)]
    dm.polygon(main, fill=255)
    dm.polygon(jib, fill=255)
    dv.polygon(main, fill=150)
    dv.polygon(jib, fill=96)
    dm.line([P(0, -6), P(0, -130)], fill=255, width=int(2.4 * k * s))          # mast
    dv.line([P(0, -6), P(0, -130)], fill=6, width=int(2.4 * k * s))
    dm.polygon([P(0, -130), P(0, -122), P(16, -127)], fill=255)                  # pennant
    dv.polygon([P(0, -130), P(0, -122), P(16, -127)], fill=10)
    dm.line([P(-52, -4), P(3, -6)], fill=255, width=int(2 * k * s))               # boom / deck line
    dv.line([P(-52, -4), P(3, -6)], fill=10, width=int(2 * k * s))
    val = val.resize((w // k, h // k), Image.LANCZOS)
    msk = msk.resize((w // k, h // k), Image.LANCZOS)
    v = np.asarray(val).astype(np.float32) / 255
    m = np.asarray(msk).astype(np.float32) / 255
    # shade the sails: brighter toward the moon (right) and toward the top, a seam or two
    hh, ww = v.shape
    gy, gx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    lit = 0.78 + 0.42 * np.clip((gx - ww / 2) / (46 * s) , -1, 1) * 0.5 - 0.22 * np.clip((gy - hh / 2) / (120 * s), -1, 1)
    sail = (v > 0.2).astype(np.float32)
    v = np.where(sail > 0, v * lit, v)
    seam = 0.85 + 0.15 * np.sin((gy - hh / 2) / (7 * s) + 1.7)
    v = np.where(sail > 0, v * seam, v)
    # paste into the frame
    x0, y0 = sx - ww // 2, sy - hh // 2
    sub = (slice(max(0, y0), min(H, y0 + hh)), slice(max(0, x0), min(W, x0 + ww)))
    sv = (slice(sub[0].start - y0, sub[0].stop - y0), slice(sub[1].start - x0, sub[1].stop - x0))
    out = img.copy()
    out[sub] = out[sub] * (1 - m[sv]) + v[sv] * m[sv]
    # the boat in the water: a stretched, broken, dimmer mirror
    mir_v, mir_m = v[::-1], m[::-1]
    yy = np.arange(hh)
    for i in range(0, 90, 1):
        yd = sy + 6 + i
        if yd >= H:
            break
        src = int(hh / 2 + 6 - 1 - i * 1.1)
        if src < 0 or src >= hh:
            continue
        row_m = m[::-1][hh - 1 - src] if False else m[src]
        row_v = v[src]
        fade = (1 - i / 95.0) ** 1.4
        wob = int(3 * np.sin(i * 0.9 + 1.0 * (i % 7)))
        xs0 = x0 + wob
        a, b = max(0, xs0), min(W, xs0 + ww)
        if b <= a:
            continue
        mm = row_m[a - xs0:b - xs0] * fade * (0.55 + 0.45 * np.sin(i * 1.9) ** 2)
        out[yd, a:b] = out[yd, a:b] * (1 - mm) + row_v[a - xs0:b - xs0] * mm * 0.9
    return out


def crescent() -> None:
    """a wide, low-key frame: a crescent moon with its faint dark limb, a Milky Way across dense stars, a fine swell with a broken
    glitter path and one small sloop on it. Composed for a 2.39:1 window (rows ~138..942); the 16:9 sheet around it is dark."""
    hor = 650
    # galactic band: a diagonal from upper left down to the horizon right of centre, with dust lanes in it
    p0x, p0y, p1x, p1y = 40.0, 230.0, 1300.0, 650.0
    ln = float(np.hypot(p1x - p0x, p1y - p0y))
    dist = ((XX - p0x) * (p1y - p0y) - (YY - p0y) * (p1x - p0x)) / ln
    along = ((XX - p0x) * (p1x - p0x) + (YY - p0y) * (p1y - p0y)) / ln / ln
    width = 150 + 90 * smooth(0, 1, along)
    band = np.exp(-(dist / width) ** 2)
    mw_n = fbm(H, W, 210, base=7, octaves=7, stretch=1.6)
    lanes = smooth(0.52, 0.72, fbm(H, W, 211, base=4, octaves=5, stretch=2.2))
    core = np.exp(-(dist / (width * 0.42)) ** 2)
    milky = band * (0.35 + 0.9 * mw_n) * (1 - 0.8 * lanes * band) + 0.5 * core * mw_n
    sky = 0.014 + 0.022 * smooth(0, hor, YY) + 0.16 * smooth(hor * 0.45, hor, YY) ** 2.2
    img = sky + 0.24 * milky * (1 - 0.4 * smooth(hor - 160, hor, YY))
    img += _star_field(hor, band + 0.5 * core)
    # the moon: a lit crescent with surface, the rest of the disc faintly there (earthshine)
    mx, my, mr = 1345, 318, 132
    d = np.hypot(XX - mx, YY - my)
    disc = smooth(mr + 1.2, mr - 1.2, d)
    d2 = np.hypot(XX - (mx + 84), YY - (my - 36))
    cut = smooth(mr - 3.0, mr + 0.6, d2)
    cres = disc * cut
    tex = fbm(H, W, 300, base=14, octaves=5)
    maria = smooth(0.42, 0.62, fbm(H, W, 301, base=4, octaves=4))
    limb = 0.82 + 0.3 * smooth(mr - 46, mr, d)
    surf = (0.66 + 0.42 * tex - 0.30 * maria) * limb
    craters = np.zeros((H, W), np.float32)
    rc = np.random.RandomState(302)
    for _ in range(70):
        a = rc.rand() * 6.28
        rr = (0.12 + 0.88 * rc.rand()) * mr * 0.95
        cx, cy = mx + rr * np.cos(a), my + rr * np.sin(a)
        cr = 3 + 9 * rc.rand() ** 1.8
        dd = np.hypot(XX[int(max(0, cy - 30)):int(cy + 30), int(max(0, cx - 30)):int(cx + 30)] - cx,
                      YY[int(max(0, cy - 30)):int(cy + 30), int(max(0, cx - 30)):int(cx + 30)] - cy)
        sl = (slice(int(max(0, cy - 30)), int(cy + 30)), slice(int(max(0, cx - 30)), int(cx + 30)))
        craters[sl] += -0.26 * np.exp(-(dd / (cr * 0.8)) ** 2) + 0.10 * np.exp(-((dd - cr * 1.15) / 2.4) ** 2)
    surf = np.clip(surf + craters, 0.2, 1.05)
    ear = disc * (1 - cres) * (0.036 + 0.024 * tex)
    rim = np.exp(-((d - mr) / 1.8) ** 2) * 0.10 * (1 - cres)
    img = img * (1 - disc) + ear + rim + cres * surf * 0.97 + img * disc * 0.2 * (1 - cres)
    dg = np.hypot(XX - (mx - 52), YY - (my + 24))
    img += 0.22 * np.exp(-(dg / 150) ** 2) + 0.09 * np.exp(-(dg / 520) ** 2) + 0.035 * np.exp(-(dg / 1100) ** 2)
    # sea: two swells of growing spacing, ripples, the sky's glow in it, the moon's glitter
    rows = (YY - hor).clip(0, None)
    wave = fbm(H, W, 205, base=8, octaves=4, stretch=7)
    wave2 = fbm(H, W, 207, base=14, octaves=4, stretch=9)
    ph1 = 104.7 * np.log1p(0.02 * rows)
    ph2 = 17.95 * np.log1p(0.0175 * rows)
    rip = 0.5 + 0.5 * np.sin(ph1 + 9 * (wave2 - 0.5) + 0.9 * np.sin(XX / (50 + rows * 0.7) + ph1 * 0.25))
    swl = 0.5 + 0.5 * np.sin(ph2 + 3.2 * (wave - 0.5) + 1.1 * np.sin(XX / (230 + rows * 1.2) + ph2 * 0.3))
    crest = 0.35 * rip + 0.65 * swl
    near = smooth(0, 290, rows)
    refl = 0.115 + 0.12 * np.exp(-rows / 90.0) + 0.07 * np.exp(-(((XX - 900) / 700) ** 2)) * np.exp(-rows / 220.0)
    sea_img = refl * (0.25 + 1.5 * crest) * (1 - 0.3 * near)
    spec_w = 24 + rows * 0.55
    dash = smooth(0.34, 0.68, fbm(H, W, 206, base=5, octaves=3, stretch=16))
    gx0 = mx - 60 + (wave - 0.5) * 150 * (1 - near * 0.4)
    glit = np.exp(-(((XX - gx0) / spec_w) ** 2)) * (0.25 + 0.75 * dash)
    bright = glit * smooth(0.40, 0.82, 0.5 * rip + 0.5 * swl) * (0.5 + 0.5 * swl)
    halo = 0.14 * np.exp(-(((XX - (mx - 60)) / (200 + rows * 1.2)) ** 2)) * crest * np.exp(-rows / 380.0)
    sea_img = sea_img + 0.95 * bright + halo + 0.05 * band * crest * np.exp(-rows / 120.0)
    img = np.where(YY >= hor, sea_img, img)
    img[hor - 1:hor + 2] = blur(img, 1.0)[hor - 1:hor + 2]
    img += 0.12 * np.exp(-(((YY - hor) / 9) ** 2)) * (0.4 + 0.6 * np.exp(-(((XX - (mx - 60)) / 700) ** 2)))
    # the sloop, in the path of light
    img = _draw_boat(img, 1195, 666, 1.45)
    # vignette
    v = np.hypot((XX - W / 2) / (W / 2), (YY - H / 2) / (H / 2))
    img *= 1 - 0.5 * smooth(0.45, 1.3, v)
    finish(img, "07_crescent_sea", grain=0.012)


def main() -> int:
    for fn in (lake, city, still, oak, sea, dunes, crescent):
        fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
