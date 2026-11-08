"""Кольорові палітри результату. Бінаризація лишається суворо двобітною: палітра - це лише два кольори, в які
перефарбовуються «туш» (темні пікселі джерела) і «папір» (світлі). «Класика» - чисто чорно-біле, як було.

Кожна пара кольорів - ОДНА палітра. «Навпаки» (туш і папір міняються місцями) - окремий перемикач, а не друга палітра:
у ключі це суфікс INV («Синька~»). Палітра застосовується в самому кінці (перегляд, відео, GIF, фото), тож обробка й кеш
кадрів від палітри та інверсії не залежать - перемикання миттєве."""
from __future__ import annotations

import numpy as np

CLASSIC = "Класика"
INV = "~"                      # суфікс ключа: туш і папір поміняні місцями

# назва -> (туш, папір), RGB. Правило: туш ТЕМНІША за папір у кожній палітрі - тож світле на фото лишається світлим, куди б
# не перемикали стрілками; «навпаки» це міняє (світле стає темним). Порядок = порядок перемикання: нейтральні, теплі, холодні.
PALETTES: dict[str, tuple[str, str]] = {
    CLASSIC: ("#000000", "#FFFFFF"),
    "Туш і папір": ("#1B1A17", "#F1EADB"),
    "Сепія": ("#3A2616", "#E9D6AF"),
    "Газета": ("#26262A", "#D8D5CA"),
    "Кармін": ("#A3171E", "#F7F0E4"),
    "Плакат": ("#101010", "#F0483A"),
    "Апельсин": ("#0F2350", "#FF8A1E"),
    "Бурштин": ("#150D00", "#FFB000"),
    "Захід": ("#45106A", "#FFA977"),
    "Фламінго": ("#B3125A", "#FFE1EB"),
    "Неон": ("#0E0B24", "#FF3FA4"),
    "Фіалка": ("#2B1A70", "#FFE45C"),
    "Лаванда": ("#33235F", "#E9DFFF"),
    "Кіберпанк": ("#2A0A4A", "#19E3FF"),
    "Синька": ("#16379A", "#F3F0E6"),
    "Лід": ("#0B2A4A", "#D4EEFF"),
    "Бірюза": ("#0A4F5C", "#F4EAD2"),
    "М'ята": ("#0F3D33", "#D8F5E4"),
    "Термінал": ("#06130A", "#7CFF9B"),
    "Ретро LCD": ("#0F380F", "#9BBC0F"),
}
NAMES = list(PALETTES)

# назви зі старих налаштувань -> (нова назва, чи «навпаки»)
LEGACY: dict[str, tuple[str, bool]] = {
    "Ризограф": ("Синька", False),          # синьо-біла й біло-синя були двома палітрами - тепер одна + інверсія
    "Game Boy": ("Ретро LCD", False),       # без чужої торгової марки в назві
}
# у старих версіях ці палітри мали світлу тушшу на темному папері (негатив); щоб збережений вибір виглядав так само
_WAS_REVERSED = {"Синька", "Бурштин", "Термінал", "Неон"}


def split(key: str | None) -> tuple[str, bool]:
    """ключ -> (назва палітри, чи «навпаки»); невідома назва = Класика"""
    k = key or CLASSIC
    inv = k.endswith(INV)
    if inv:
        k = k[:-len(INV)]
    return (k if k in PALETTES else CLASSIC), inv


def key(name: str | None, inverted: bool = False) -> str:
    """(назва, «навпаки») -> ключ, який приймають усі функції нижче й експорт"""
    return split(name)[0] + (INV if inverted else "")


def migrate(name: str | None, inverted: bool | None = None) -> tuple[str, bool]:
    """збережені налаштування (будь-якої версії) -> (назва, «навпаки»). Нові налаштування мають явний прапорець;
    у старих його немає - тоді палітри з _WAS_REVERSED вмикаємо «навпаки», щоб вигляд не змінився"""
    if name in LEGACY:
        return LEGACY[name]
    if inverted is None and name in _WAS_REVERSED:
        return name, True
    return split(name)[0], bool(inverted)


def _rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def colors(k: str | None) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """-> (туш RGB, папір RGB) з урахуванням «навпаки»"""
    name, inv = split(k)
    ink, paper = _rgb(PALETTES[name][0]), _rgb(PALETTES[name][1])
    return (paper, ink) if inv else (ink, paper)


def is_classic(k: str | None) -> bool:
    """чорне на білому «як є» - єдиний випадок, коли можна писати справжні 1-бітні/сірі файли без перефарбування"""
    return split(k) == (CLASSIC, False)


def is_negative(k: str | None) -> bool:
    """біле на чорному: теж чисто чорно-біле (сірий канал, швидко), лише з протилежною полярністю"""
    return split(k) == (CLASSIC, True)


def lut(k: str | None, bgr: bool = False) -> np.ndarray:
    """таблиця 2×3 uint8: рядок 0 - туш, рядок 1 - папір"""
    t = np.array(colors(k), np.uint8)
    return t[:, ::-1].copy() if bgr else t


def paint(mask01: np.ndarray, k: str | None, bgr: bool = False) -> np.ndarray:
    """маска (h,w) 0/1 -> кольорове зображення (h,w,3) uint8"""
    return lut(k, bgr)[(mask01 > 0).astype(np.uint8)]


def pil_palette(k: str | None) -> list[int]:
    """плоский список для PIL putpalette: [туш RGB, папір RGB]"""
    a, b = colors(k)
    return [*a, *b]


def _lum(c) -> float:
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c)


def mix(a, b, t: float):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def contrast(a, b) -> float:
    """коефіцієнт контрасту двох кольорів (WCAG): 1 - однакові, 21 - чорне на білому"""
    def rel(c):
        v = [x / 255.0 for x in c]
        v = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
        return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
    la, lb = rel(a), rel(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def theme(k: str | None) -> dict:
    """кольори інтерфейсу з двох кольорів палітри. Звичайно темніший - тло, світліший - текст і рамки. «Навпаки» (суфікс «~»)
    інвертує й вікно: тло і текст міняються місцями (у «Класика~» - біле вікно з чорним текстом), тож кнопка «навпаки»
    діє на весь інтерфейс, а не лише на картинку"""
    name, inv = split(k)
    if name == CLASSIC:
        if inv:
            return dict(BG="#ffffff", FG="#000000", DIM="#656565", PANEL="#ebebeb")
        return dict(BG="#000000", FG="#ffffff", DIM="#9a9a9a", PANEL="#141414")
    a, b = colors(name)
    bg, fg = (a, b) if _lum(a) <= _lum(b) else (b, a)
    if inv:
        bg, fg = fg, bg
    t = 0.58
    dim = mix(bg, fg, t)
    while contrast(dim, bg) < 3.6 and t < 0.95:          # другорядний текст має лишатися читабельним і на яскравих палітрах
        t += 0.04
        dim = mix(bg, fg, t)
    return dict(BG=_hex(bg), FG=_hex(fg), DIM=_hex(dim), PANEL=_hex(mix(bg, fg, 0.08)))
