"""Малює картинки майстра інсталятора (installer\\art\\*.bmp): чорно-біла гама з дізерингом у дусі самої програми.
    python tools/make_installer_art.py
Inno Setup бере BMP: велика картинка на першій і останній сторінці (164x314 і, для екранів з масштабом, 2x / 3x), мала - у шапці інших (55x55 ...)."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "benvid", "assets")
OUT = os.path.join(ROOT, "installer", "art")
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0


def dither(h: int, w: int, field: np.ndarray, cell: int) -> np.ndarray:
    """field 0..1 (h, w) -> чорно-біла картинка з упорядкованим дізерингом, клітинка cell px"""
    yy, xx = np.mgrid[0:h, 0:w]
    thr = BAYER[(yy // cell) % 4, (xx // cell) % 4]
    return np.where(field > thr, 255, 0).astype(np.uint8)


def big(scale: int) -> Image.Image:
    w, h = 164 * scale, 314 * scale
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = yy / h
    wave = 0.5 + 0.5 * np.sin(xx / (w / 5.0) + t * 7.0)               # м'які смуги: «відео, що розкладається на чорне й біле»
    field = np.clip(0.04 + 0.95 * (t ** 1.6) * (0.55 + 0.45 * wave) - 0.0 * xx, 0, 1)
    img = Image.fromarray(dither(h, w, field, max(1, scale)), "L").convert("RGB")
    logo = Image.open(os.path.join(ASSETS, "mascot_w.png")).convert("RGBA")
    k = max(1, (w * 6 // 10) // logo.width)
    logo = logo.resize((logo.width * k, logo.height * k), Image.NEAREST)
    pad = Image.new("RGBA", (logo.width + 8 * scale, logo.height + 8 * scale), (0, 0, 0, 255))
    pad.paste(logo, (4 * scale, 4 * scale), logo)
    img.paste(pad.convert("RGB"), ((w - pad.width) // 2, 28 * scale))
    return img


def small(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), (0, 0, 0))
    m = Image.open(os.path.join(ASSETS, "mascot_w.png")).convert("RGBA")
    k = max(1, (size * 9 // 10) // m.width)
    m = m.resize((m.width * k, m.height * k), Image.NEAREST)
    img.paste(m, ((size - m.width) // 2, (size - m.height) // 2), m)
    return img


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for s in (1, 2, 3):
        big(s).save(os.path.join(OUT, f"wizard_{s}x.bmp"), "BMP")
    for px in (55, 82, 110):
        small(px).save(os.path.join(OUT, f"small_{px}.bmp"), "BMP")
    print("готово:", OUT)


if __name__ == "__main__":
    main()
