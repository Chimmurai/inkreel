"""Перегляд результату: масштаб колесом і зсув. Чиста логіка без tkinter - тому її можна перевіряти тестами.

Головне правило: на екрані завжди чіткі пікселі. Збільшення - лише ЦІЛЕ (1×, 2×, 3× …) і найближчим сусідом: піксель результату
стає квадратом n×n одного кольору, без згладжування й проміжних відтінків. Зменшення (картинка більша за вікно) - усереднення
(cv2.INTER_AREA): інакше растр і дрібний текст перетворюються на кашу. «Вписати» - найбільший цілий масштаб, що влазить у вікно
(більша за вікно картинка - плавне зменшення). Рахується лише видимий шматок, тож 64× на кадрі 4K коштує стільки ж, скільки 1×."""
from __future__ import annotations

import cv2
import numpy as np

# рівні масштабу для колеса: нижче 1 - зменшення, від 1 - цілі кратні
LEVELS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64)
CENTER = (0.5, 0.5)


def fit_scale(cw: int, ch: int, w: int, h: int) -> float:
    """масштаб «вписати»: більше за вікно - плавне зменшення, менше - ЦІЛЕ збільшення"""
    sc = min(cw / w, ch / h)
    return sc if sc < 1.0 else float(max(1, int(sc)))


def scaled_size(w: int, h: int, s: float) -> tuple[int, int]:
    return (int(w * s), int(h * s)) if s >= 1 else (max(2, int(w * s)), max(2, int(h * s)))


def geometry(cw: int, ch: int, w: int, h: int, zoom: float | None, vc=CENTER):
    """-> (масштаб, W, H, ox, oy): розмір масштабованого зображення і зсув вікна перегляду в його пікселях. zoom None - «вписати»
    (по центру). Зсув від'ємний, коли зображення менше за вікно: воно по центру, навколо - поле. vc - центр вікна в частках
    розміру зображення"""
    s = zoom if zoom is not None else fit_scale(cw, ch, w, h)
    W, H = scaled_size(w, h, s)
    fx, fy = vc if zoom is not None else CENTER

    def off(full, win, f):
        if full <= win:
            return -((win - full) // 2)
        return min(max(int(round(f * full - win / 2)), 0), full - win)
    return s, W, H, off(W, cw, fx), off(H, ch, fy)


def compose(mask: np.ndarray, lut, cw: int, ch: int, zoom: float | None, vc=CENTER):
    """-> (шматок, k, x, y, обрізка) або None.
    шматок - видима частина результату в пікселях ДЖЕРЕЛА (сірий (h,w) або, якщо є lut 256×1×3, RGB), k - цілий множник для
    збільшення найближчим сусідом (PIL.Image.NEAREST), x, y - куди класти готове зображення на полотні, обрізка
    (rx, ry, vw, vh) або None - яку частину збільшеного шматка лишити (зсув на частку пікселя)"""
    h, w = mask.shape
    s, W, H, ox, oy = geometry(cw, ch, w, h, zoom, vc)
    x0, x1, y0, y1 = max(0, ox), min(W, ox + cw), max(0, oy), min(H, oy + ch)
    if x1 <= x0 or y1 <= y0:
        return None
    g8 = mask * 255 if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8)
    if s >= 1:
        k = int(s)
        sx0, sx1, sy0, sy1 = x0 // k, -(-x1 // k), y0 // k, -(-y1 // k)            # які пікселі джерела видно
        img = g8[sy0:sy1, sx0:sx1]
        if lut is not None:
            img = cv2.applyColorMap(np.ascontiguousarray(img), lut)               # байти йдуть як у таблиці: RGB для PPM
        crop = (x0 - sx0 * k, y0 - sy0 * k, x1 - x0, y1 - y0)
        if crop == (0, 0, img.shape[1] * k, img.shape[0] * k):
            crop = None
        return img, k, max(0, -ox), max(0, -oy), crop
    img = cv2.resize(g8, (W, H), interpolation=cv2.INTER_AREA)[y0:y1, x0:x1]
    if lut is not None:
        img = cv2.applyColorMap(np.ascontiguousarray(img), lut)
    return img, 1, max(0, -ox), max(0, -oy), None


def finish(img: np.ndarray, k: int, crop) -> np.ndarray:
    """готове зображення з результату compose без PIL (запасний шлях і тести): збільшення найближчим сусідом + обрізка"""
    if k > 1:
        img = np.repeat(np.repeat(img, k, 0), k, 1)
    if crop:
        img = img[crop[1]:crop[1] + crop[3], crop[0]:crop[0] + crop[2]]
    return np.ascontiguousarray(img)


def step(s0: float, fit: float, d: int, zoomed: bool):
    """один крок колеса від показаного масштабу s0. -> (змінилось, новий масштаб або None = «вписати»).
    Угору - наступний рівень (цілий, чіткий), униз - попередній, але не далі за «вписати»"""
    if d > 0:
        new = next((v for v in LEVELS if v > s0 * 1.0001), None)
        return (False, None) if new is None else (True, new)
    if not zoomed:
        return False, None
    new = next((v for v in reversed(LEVELS) if v < s0 * 0.9999), None)
    if new is None or new <= fit * 1.0001:
        return True, None
    return True, new


def anchored(cw: int, ch: int, w: int, h: int, s0: float, ox0: int, oy0: int, new: float, mx: float, my: float):
    """-> vc для масштабу new так, щоб точка зображення під курсором (mx, my) лишилась під ним"""
    px, py = min(max((ox0 + mx) / s0, 0), w), min(max((oy0 + my) / s0, 0), h)
    W1, H1 = scaled_size(w, h, new)
    return (px * new - mx + cw / 2) / W1, (py * new - my + ch / 2) / H1


def pan(cw: int, ch: int, w: int, h: int, zoom: float, vc, dx: int, dy: int):
    """зсув зображення на (dx, dy) пікселів екрана (перетягування) -> новий vc"""
    s, W, H, ox, oy = geometry(cw, ch, w, h, zoom, vc)
    if W > cw:
        ox = min(max(ox - dx, 0), W - cw)
    if H > ch:
        oy = min(max(oy - dy, 0), H - ch)
    return ((ox + cw / 2) / W if W > cw else 0.5, (oy + ch / 2) / H if H > ch else 0.5)


def label(zoom: float) -> str:
    pct = zoom * 100
    return f"{pct:g}%"
