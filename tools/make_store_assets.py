"""Логотипи для Microsoft Store: усе з тієї самої котушки (tools/make_icon.py), кожен розмір малюється заново, а не стискається з одного.

    python tools/make_store_assets.py            ->  store/Assets/*.png (пакет MSIX) і store/listing_assets/*.png (сторінка в Partner Center)
    python tools/make_store_assets.py <тека>     ->  лише ресурси пакета в цю теку (так викликає tools/make_msix.py)

Картинки сторінки: StoreAppTile 300x300 (значок), StoreBoxArt 1080x1080 (з написом), SuperHeroArt 1920x1080 - за правилами Store
без назви й будь-якого тексту. Знімки екрана Store приймає лише справжні: їх робить автор у Windows (див. store/STORE.md).

Імена файлів - за правилами MSIX (масштаби .scale-100/125/150/200/400, розміри .targetsize-N, «без плити» _altform-unplated):
Windows сам вибирає потрібний; resources.pri збирає tools/make_msix.py (makepri з Windows SDK)."""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import make_icon  # noqa: E402

ASSETS = os.path.join(ROOT, "benvid", "assets")
SCALES = (100, 125, 150, 200, 400)
TARGETS = (16, 24, 32, 48, 256)
BLACK = (0, 0, 0, 255)

# (ім'я, базовий розмір на масштабі 100 у пікселях, які масштаби робимо)
SQUARE = (("Square44x44Logo", 44), ("Square150x150Logo", 150), ("SmallTile", 71), ("StoreLogo", 50))
BASE_NAMES = ("Square44x44Logo", "Square150x150Logo", "SmallTile", "StoreLogo", "Wide310x150Logo", "LargeTile", "IkrFile")   # усе, що названо в AppxManifest.xml


def px(base: int, scale: int) -> int:
    return round(base * scale / 100)


def plated(size: int) -> Image.Image:
    """повна плитка (чорне тло з білою рамкою й котушкою) - для плиток Пуску й списку програм"""
    return make_icon.render(size, True, size <= 96).convert("RGBA")


def unplated(size: int) -> Image.Image:
    """та сама плитка з прозорими округлими кутами - для панелі завдань і Alt+Tab (Windows не підкладає під неї свою плиту)"""
    return make_icon.tile_rgba(size, size <= 96)


def emblem(size: int) -> Image.Image:
    """плитка для великих картинок: чорне поле навколо рамки рівної ширини й з тими самими заокругленнями (а не квадрат із зрізаними кутами),
    тож сяйво, що її обступає, не впирається в пряму лінію"""
    im = plated(size)
    k = 4
    m = Image.new("L", (size * k, size * k), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size * k - 1, size * k - 1), radius=round(size * k * 156 / make_icon.M), fill=255)
    im.putalpha(m.resize((size, size), Image.LANCZOS))
    return im


def _logo(k: int) -> Image.Image:
    """пікселізований напис InkReel, збільшений у k (ціле) разів: пікселі лишаються рівними квадратами"""
    lg = Image.open(os.path.join(ASSETS, "logo_w.png")).convert("RGBA")
    return lg.resize((lg.width * k, lg.height * k), Image.NEAREST)


def wide(w: int, h: int, scale: int = 100) -> Image.Image:
    """широка плитка: котушка ліворуч, напис праворуч, чорне тло. Напис - у ціле число разів більший за базовий (не більше, ніж масштаб),
    але так, щоб котушка лишалась не нижчою за 60 % висоти"""
    im = Image.new("RGBA", (w, h), BLACK)
    pad = round(h * 0.09)
    lw = _logo(1).width
    k, s = 1, min(h - 2 * pad, w - lw - 3 * pad)
    for kk in range(max(1, scale // 100), 0, -1):
        ss = min(h - 2 * pad, w - lw * kk - 3 * pad)
        if ss >= 0.6 * h or kk == 1:
            k, s = kk, ss
            break
    s = max(s, 16)
    im.alpha_composite(plated(s), (pad, (h - s) // 2))
    lg = _logo(k)
    im.alpha_composite(lg, (2 * pad + s, (h - lg.height) // 2))
    return im


def large(s: int, scale: int = 100) -> Image.Image:
    """велика плитка: котушка зверху, напис під нею"""
    im = Image.new("RGBA", (s, s), BLACK)
    icon = round(s * 0.62)
    lg = _logo(max(1, scale // 100))
    gap = round(s * 0.05)
    top = (s - icon - gap - lg.height) // 2
    im.alpha_composite(plated(icon), ((s - icon) // 2, top))
    im.alpha_composite(lg, ((s - lg.width) // 2, top + icon + gap))
    return im


def package_assets() -> dict[str, Image.Image]:
    """ресурси, на які посилається AppxManifest.xml (Assets\\<ім'я>.png розкриваються в .scale-N / .targetsize-N)"""
    out: dict[str, Image.Image] = {}
    for name, base in SQUARE:
        for sc in SCALES:
            if name == "Square44x44Logo":
                out[f"{name}.scale-{sc}.png"] = plated(px(base, sc))
            elif name == "StoreLogo":
                out[f"{name}.scale-{sc}.png"] = unplated(px(base, sc))
            else:
                out[f"{name}.scale-{sc}.png"] = plated(px(base, sc))
    for t in TARGETS:
        out[f"Square44x44Logo.targetsize-{t}.png"] = plated(t)
        out[f"Square44x44Logo.targetsize-{t}_altform-unplated.png"] = unplated(t)
        out[f"Square44x44Logo.targetsize-{t}_altform-lightunplated.png"] = unplated(t)
    for sc in SCALES:
        out[f"Wide310x150Logo.scale-{sc}.png"] = wide(px(310, sc), px(150, sc), sc)
    for sc in (100, 125, 150, 200):                    # 400 % (1240 px) не робимо: Windows збільшить 200 %, а зайві мегабайти - у кожне оновлення
        out[f"LargeTile.scale-{sc}.png"] = large(px(310, sc), sc)
    for sc in SCALES:                                  # значок файлів .ikr (білий аркуш із підписом IKR; без плитки програми)
        out[f"IkrFile.scale-{sc}.png"] = make_icon.ikr_doc(px(44, sc))
    for name in BASE_NAMES:                            # файл без масштабу - запасний: так само названий у маніфесті, тож є й без resources.pri
        out[f"{name}.png"] = out[f"{name}.scale-200.png"]
    return out


def _noise() -> np.ndarray:
    """карта порогів «синього шуму» тієї ж програми (benvid/bluenoise128.npy): так InkReel і малює півтони"""
    return np.load(os.path.join(ROOT, "benvid", "bluenoise128.npy"))


def halo(w: int, h: int, cell: int, cx: float, cy: float, rx: float, ry: float, power: float = 2.0, peak: float = 0.6) -> Image.Image:
    """чорно-біле «сяйво»: гладкий спад яскравості від центру (cx, cy, rx, ry - частки розміру) з піком peak, розкладений на пікселі
    шумом InkReel. Піксель сяйва - квадрат cell x cell, як у піксельному інтерфейсі програми"""
    gw, gh = -(-w // cell), -(-h // cell)
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    d = np.hypot((xx / gw - cx) / rx, (yy / gh - cy) / ry)
    light = peak * np.clip(1.0 - d, 0.0, 1.0) ** power
    bn = _noise()
    thr = np.tile(bn, (gh // bn.shape[0] + 1, gw // bn.shape[1] + 1))[:gh, :gw]
    on = (light > thr).astype(np.uint8) * 255
    small = Image.fromarray(on, "L").resize((gw * cell, gh * cell), Image.NEAREST).crop((0, 0, w, h))
    return small.convert("RGBA")


def hero(w: int = 1920, h: int = 1080) -> Image.Image:
    """16:9 «супергерой» для сторінки магазину. Правило Store: без назви й будь-якого тексту - лише картинка: сяйво з пікселів-півтонів і котушка"""
    im = halo(w, h, max(2, round(h / 270)), 0.5, 0.5, 0.5, 0.85)
    s = round(h * 0.52)
    im.alpha_composite(emblem(s), ((w - s) // 2, (h - s) // 2))
    return im


def box_art(side: int = 1080) -> Image.Image:
    """1:1 (1080 або 2160): котушка й напис на чистому чорному, усе в верхніх двох третинах - низ Store часом затуляє власними написами"""
    im = halo(side, side, max(2, round(side / 270)), 0.5, 0.28, 0.5, 0.27)       # сяйво гасне вище напису: той лишається на чистому чорному
    icon = round(side * 0.42)
    top = round(side * 0.07)
    im.alpha_composite(emblem(icon), ((side - icon) // 2, top))
    lg = _logo(max(1, round(side * 0.5) // _logo(1).width))
    im.alpha_composite(lg, ((side - lg.width) // 2, top + icon + round(side * 0.035)))
    return im


def listing_assets() -> dict[str, Image.Image]:
    """картинки для завантаження в Partner Center (Store listings -> Store logos / Super hero art)"""
    return {"StoreBoxArt_1080x1080.png": box_art(1080),
            "StoreAppTile_300x300.png": plated(300),
            "SuperHeroArt_1920x1080.png": hero(1920, 1080)}


def write(out_dir: str, files: dict[str, Image.Image]) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for name, im in files.items():
        im.save(os.path.join(out_dir, name), optimize=True)
        names.append(name)
    return names


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        n = write(argv[0], package_assets())
        print(f"{len(n)} файлів -> {argv[0]}")
        return
    n1 = write(os.path.join(ROOT, "store", "Assets"), package_assets())
    n2 = write(os.path.join(ROOT, "store", "listing_assets"), listing_assets())
    print(f"store/Assets: {len(n1)} файлів; store/listing_assets: {', '.join(n2)}")


if __name__ == "__main__":
    main()
