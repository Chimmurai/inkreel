"""Піксель-арт з `benvid/assets`, перефарбований під тему (чорне -> тло, біле -> колір тексту), як `_img` у gui.py; повертає QImage.
Кеш за (ім'я, тло, текст): перемикання палітри не перекодовує малюнки, що вже бачили."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image
from PySide6.QtGui import QImage

from .theme import THEME

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_CACHE: dict = {}
_INK: dict = {}


def ink(name: str):
    """межі видимих пікселів (x0, y0, x1, y1; кінець не включно): кнопки центрують сам малюнок, а не прозорі поля"""
    if name not in _INK:
        try:
            im = Image.open(os.path.join(ASSETS, name + ".png")).convert("RGBA")
            _INK[name] = im.getchannel("A").getbbox() or (0, 0) + im.size
        except Exception:  # noqa: BLE001
            _INK[name] = None
    return _INK[name]


def _hex(c) -> tuple[int, int, int]:
    return c.red(), c.green(), c.blue()


def img(name: str, rot: float = 0.0) -> QImage | None:
    bg, fg = _hex(THEME.bg), _hex(THEME.fg)
    key = (name, bg, fg)
    if not rot and key in _CACHE:
        return _CACHE[key]
    try:
        im = Image.open(os.path.join(ASSETS, name + ".png")).convert("RGBA")
        if rot:
            im = im.rotate(rot, resample=Image.NEAREST)
        a = np.array(im, np.float32)
        if (bg, fg) != ((0, 0, 0), (255, 255, 255)):
            lo, hi = np.array(bg, np.float32), np.array(fg, np.float32)
            t = a[..., :3].mean(-1, keepdims=True) / 255.0
            a[..., :3] = lo + (hi - lo) * t
        arr = np.ascontiguousarray(a.round().astype(np.uint8))
        q = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGBA8888)
        q = q.copy()                                   # свій буфер: масив далі не потрібен
        q.ink = None if rot else ink(name)
        if not rot:
            _CACHE[key] = q
        return q
    except Exception:  # noqa: BLE001
        return None
