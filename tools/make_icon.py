"""Іконка й знак InkReel: котушка плівки, права половина якої «розсипається» у ч/б растр (сама суть програми).
Запуск: python tools/make_icon.py  ->  benvid/assets/{inkreel.ico, ikr.ico, app_icon.png, mark_w.png, mascot*.png}"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benvid", "assets")
M = 1024                                    # робоче полотно (суперсемплінг)

BAYER = np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26], [12, 44, 4, 36, 14, 46, 6, 38],
                  [60, 28, 52, 20, 62, 30, 54, 22], [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
                  [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]], np.float32) / 64.0


def reel(tile=True, elem=4, steep=False) -> Image.Image:
    """-> L-зображення 1024x1024: білий знак на чорному (tile=True - з плиткою й рамкою, False - лише знак, фон чорний)"""
    yy, xx = np.mgrid[0:M, 0:M].astype(np.float32)
    cx = cy = M / 2
    R = 440 if not tile else 350
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= R * R
    if steep:                                                         # малі розміри: більша суцільна частина, короткий растр
        g = np.clip((xx - (cx + R * 0.25)) / (R * 0.8), 0, 1)
    else:
        g = np.clip((xx - (cx - R * 0.55)) / (R * 1.55), 0, 1)       # 0 ліворуч -> 1 праворуч, довгий перехід
    g = np.clip(g + (yy - cy) / (R * 8), 0, 1)                        # трохи по діагоналі
    th = BAYER[(yy.astype(int) // elem) % 8, (xx.astype(int) // elem) % 8]
    white = disc & ((g < th) | (g < 0.04))
    img = Image.fromarray((white * 255).astype(np.uint8), "L")
    d = ImageDraw.Draw(img)
    for k in range(6):                                                # 6 отворів котушки
        a = math.radians(30 + 60 * k)
        ox, oy = cx + math.cos(a) * R * 0.58, cy + math.sin(a) * R * 0.58
        r = R * 0.2
        d.ellipse((ox - r, oy - r, ox + r, oy + r), fill=0)
    r = R * 0.14
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=0)               # втулка
    d.ellipse((cx - r * 0.45, cy - r * 0.45, cx + r * 0.45, cy + r * 0.45), fill=255)
    if tile:                                                          # рамка-плитка у стилі кнопок інтерфейсу
        d.rounded_rectangle((36, 36, M - 37, M - 37), radius=120, outline=255, width=30)
    return img


def render(size: int, tile=True, hard=True) -> Image.Image:
    elem = max(2, int(round(M / size * (2 if size <= 48 else 1))))   # один елемент растру ~1-2 пікс. результату
    im = reel(tile, elem, size <= 64).resize((size, size), Image.LANCZOS)
    if hard and size <= 96:                                          # малі розміри - суворо чорно-білі
        im = im.point(lambda v: 255 if v >= 128 else 0)
    return im


def tile_rgba(size: int, hard=True) -> Image.Image:
    """іконка-плитка з округлими кутами: за межами плитки - прозоро (на панелі завдань нема чорного квадрата)"""
    im = render(size, True, hard).convert("RGBA")
    k = 4                                                              # суперсемплінг маски для гладких країв
    m = Image.new("L", (size * k, size * k), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size * k - 1, size * k - 1), radius=int(size * k * 0.2), fill=255)
    im.putalpha(m.resize((size, size), Image.LANCZOS))
    return im


def ikr_doc(size: int) -> Image.Image:
    """значок ФАЙЛУ .ikr: біла сторінка з підігнутим кутом, знак InkReel і чорна смуга «IKR» знизу (прозорий фон, без плитки програми).
    Так файл у Провіднику відрізняється від ярлика програми й від безіменного «Файл IKR»"""
    from PIL import ImageFont
    k = 4
    S = size * k
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pw, ph = int(S * 0.74), int(S * 0.94)
    x0, y0 = (S - pw) // 2, (S - ph) // 2
    x1, y1 = x0 + pw, y0 + ph
    fold = int(S * 0.24)
    lw = max(k, int(S * 0.05))
    poly = [(x0, y0), (x1 - fold, y0), (x1, y0 + fold), (x1, y1), (x0, y1)]
    d.polygon(poly, fill=(250, 250, 250, 255))
    band = int(ph * 0.22)
    d.rectangle((x0, y1 - band, x1, y1), fill=(0, 0, 0, 255))                         # смуга з підписом
    d.line(poly + [poly[0]], fill=(0, 0, 0, 255), width=lw, joint="curve")             # контур сторінки
    d.polygon([(x1 - fold, y0), (x1 - fold, y0 + fold), (x1, y0 + fold)], fill=(0, 0, 0, 255))     # підігнутий кут
    if size >= 40:
        try:
            f = ImageFont.truetype(os.path.join(OUT, "DejaVuSansMono.ttf"), int(band * 0.72))
            d.text(((x0 + x1) / 2, y1 - band / 2 - lw * 0.15), "IKR", font=f, fill=(255, 255, 255, 255), anchor="mm")
        except OSError:
            pass
    im = im.resize((size, size), Image.LANCZOS)
    md = int(size * 0.46)                                                              # знак InkReel (котушка, що розсипається на растр)
    if md >= 6:
        m = render(md, False)
        cy = (y0 + (ph - band) * 0.52 + (fold * 0.12)) / k
        im.paste((0, 0, 0, 255), (int(round(size / 2 - md / 2)), int(round(cy - md / 2))), m)
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    big = tile_rgba(256, False)
    big.save(os.path.join(OUT, "app_icon.png"))
    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = [tile_rgba(s) for s in sizes]
    frames[-1].save(os.path.join(OUT, "inkreel.ico"), sizes=[(s, s) for s in sizes], append_images=frames[:-1])
    docs = [ikr_doc(s) for s in sizes]                                                 # значок файлів .ikr
    docs[-1].save(os.path.join(OUT, "ikr.ico"), sizes=[(s, s) for s in sizes], append_images=docs[:-1])
    # знак для шапки (білий на прозорому/чорному): без плитки
    for name, s in (("mascot_w", 56), ("mascot", 56)):
        m = render(s, False).convert("L")
        a = m if name == "mascot_w" else m.point(lambda v: 255 - v)
        rgba = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        col = (255, 255, 255, 255) if name == "mascot_w" else (0, 0, 0, 255)
        mask = m if name == "mascot_w" else m
        px = Image.new("RGBA", (s, s), col)
        rgba.paste(px, (0, 0), mask)
        rgba.save(os.path.join(OUT, name + ".png"))
    print("ok", os.listdir(OUT)[:3])


if __name__ == "__main__":
    main()
