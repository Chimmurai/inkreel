"""Пасхалки InkReel. Тут лише чиста логіка без tkinter (лічильник кліків, цитати, розфокусування, оберти, крихти) -
тому вона перевіряється тестами (tests/test_eggs.py); вікно (gui.App) тільки викликає її й малює.

Що де сховано:
  · котушка в шапці (клік) - крутиться й каже фразу; так само крутиться, коли збереження завершилось;
  · лого «InkReel» ×5 або клік по цитаті в футері - картинка плавно розфокусовується під слова філософії
    («між чорним і білим відкриваються незліченні відтінки») й повертає різкість;
  · квадратик палітри ×7 - «вечірка»: вікно перебирає всі палітри й повертається до вашої;
  · печиво в футері (поки кнопка донату вимкнена) ×7 - відкушується, з'їдається й лишає крихти."""
from __future__ import annotations

import math
import random
import time

import cv2
import numpy as np

from . import i18n

# ---------------------------------------------------------------- лічильник кліків
class Clicks:
    """n кліків поспіль, між якими не більше gap секунд: n-й повертає True (і лічильник скидається)"""

    def __init__(self, n: int, gap: float = 1.2, clock=time.monotonic):
        self.n, self.gap, self.clock = n, gap, clock
        self.count, self.last = 0, -1e9

    def hit(self) -> bool:
        now = self.clock()
        if now - self.last > self.gap:
            self.count = 0
        self.last = now
        self.count += 1
        if self.count >= self.n:
            self.count = 0
            return True
        return False

    def reset(self):
        self.count = 0


# ---------------------------------------------------------------- фрази котушки (ключі перекладу - українські рядки)
QUOTES = (
    "Кожен піксель — маленький вибір: туш чи папір.",
    "Розфокусуй погляд — і побачиш відтінки.",
    "Двадцять чотири кадри на секунду — і світ рухається.",
    "Ніч — найкращий час для чорно-білого.",
    "Не знаєш, що обрати — спробуй «Гравюру».",
    "Я лише котушка, але вірю в тебе.",
    "Менше кольору — більше змісту.",
    "Тут усе пахне друкарською фарбою.",
    "Крутиться плівка, крутиться…",
    "Сіре — це просто багато чорних крапок.",
)


def pick_quote(last: int = -1, rng=random) -> int:
    """номер наступної фрази: щоразу нова, не та сама, що щойно була"""
    n = len(QUOTES)
    if last < 0 or last >= n:
        return rng.randrange(n)
    i = rng.randrange(n - 1)
    return i + 1 if i >= last else i


# ---------------------------------------------------------------- розфокусування під слова філософії
DEFOCUS_TOTAL = 5.2          # секунд усього: розфокусуємо - тримаємо - повертаємо різкість


def _ramp(x: float, a: float, b: float) -> float:
    return 0.0 if x <= a else 1.0 if x >= b else (x - a) / (b - a)


def _smooth(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)


def defocus_curve(t: float, total: float = DEFOCUS_TOTAL) -> float:
    """рівень розфокусування 0..1 у момент t: плавно вгору (30 %), тримаємо (44 %), плавно назад (26 %)"""
    f = t / total
    if f <= 0.0 or f >= 1.0:
        return 0.0
    return _smooth(_ramp(f, 0.0, 0.30)) * (1.0 - _smooth(_ramp(f, 0.74, 1.0)))


def defocus_sigma(amount: float, h: int, w: int) -> float:
    """σ розмиття в пікселях результату: достатньо велика, щоб растр і штрихи злилися в сірі відтінки, але не стерла зміст"""
    return amount * min(4.0, max(1.8, min(h, w) / 160.0))


def defocus(mask: np.ndarray, amount: float) -> np.ndarray:
    """0/1-маска -> плавні відтінки 0..1 (float32); amount 0 - маска без змін (той самий масив). Великі кадри (4K) -
    розмиваємо вдвічі меншими, щоб кадр не коштував десятків мілісекунд"""
    if amount < 0.004:
        return mask
    h, w = mask.shape[:2]
    sig = defocus_sigma(amount, h, w)
    m = mask.astype(np.float32)
    if h * w > 2_500_000 and h >= 4 and w >= 4:
        small = cv2.resize(m, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sig / 2, borderType=cv2.BORDER_REPLICATE)
        return np.clip(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
    return cv2.GaussianBlur(m, (0, 0), sig, borderType=cv2.BORDER_REPLICATE)


def philosophy_parts(lang: str | None = None) -> tuple[str, str]:
    """філософія двома рядками: «Весь наш світ ніби поділений на чорне й біле.» + решта"""
    from . import guide
    t = guide.PHILOSOPHY[lang or i18n.lang()]
    a, sep, b = t.partition(". ")
    return (a + ".", b) if sep else (t, "")


def caption_alphas(t: float, total: float = DEFOCUS_TOTAL) -> tuple[float, float]:
    """прозорість першого й другого рядка підпису 0..1: перший з'являється із початком розфокусування, другий - трохи пізніше,
    обидва гаснуть, коли різкість повертається"""
    f = t / total
    fade = 1.0 - _ramp(f, 0.78, 0.97)
    return _ramp(f, 0.05, 0.22) * fade, _ramp(f, 0.34, 0.52) * fade


def blend_hex(bg: str, fg: str, a: float) -> str:
    """колір між bg і fg (#rrggbb): a=0 - bg, a=1 - fg"""
    a = min(1.0, max(0.0, a))
    ch = lambda i: round(int(bg[i:i + 2], 16) * (1 - a) + int(fg[i:i + 2], 16) * a)
    return "#%02x%02x%02x" % (ch(1), ch(3), ch(5))


# ---------------------------------------------------------------- котушка
SPIN_MS = 700


def spin_angles(n: int = 14) -> list[float]:
    """кути кадрів одного повного оберту: швидкий старт і гальмування; від 0° до 360° (останній збігається з першим)"""
    n = max(2, n)
    return [360.0 * (1.0 - (1.0 - i / (n - 1)) ** 2) for i in range(n)]


# ---------------------------------------------------------------- вечірка палітр
def party_order(names, current: str) -> list[str]:
    """усі палітри по колу, починаючи з наступної за поточною; остання - поточна (тож вечірка закінчується там, де почалась)"""
    names = list(names)
    i = names.index(current) if current in names else -1
    return names[i + 1:] + names[:i + 1]


# ---------------------------------------------------------------- печиво
# відкуси відносно центру печива (dx, dy, радіус): першу, праву верхню, малює сама іконка; ці - в порядку «з'їдання»
# (по краю навколо, останній - шматок із середини; сьомий клік з'їдає решту)
BITES = ((-8.5, -3.1, 3.0), (-7.4, 5.2, 3.0), (0.8, 9.0, 3.0), (7.8, 4.5, 3.0), (-1.6, -8.9, 3.0), (1.0, 1.0, 4.5))
COOKIE_CLICKS = len(BITES) + 1       # останній клік з'їдає печиво
EATEN = 99                           # «кількість відкусів» з'їденого печива


def crumbs_burst(rng=random, n: int = 18, speed: float = 60.0) -> list[list[float]]:
    """крихти з центру печива: [x, y, vx, vy, розмір] (пікселі й пікселі/с); підстрибують убік і вгору, далі падають"""
    out = []
    for _ in range(n):
        ang = rng.uniform(-math.pi * 0.92, -math.pi * 0.08)
        sp = rng.uniform(0.3, 1.0) * speed
        out.append([0.0, 0.0, math.cos(ang) * sp, math.sin(ang) * sp * 0.6, float(rng.choice((1, 2, 2, 3)))])
    return out


def crumbs_step(ps, dt: float, g: float = 140.0, floor: float | None = None) -> list[list[float]]:
    """крок фізики: гравітація й рух; крихти, що впали нижче floor (px від центру), зникають"""
    out = []
    for p in ps:
        p[3] += g * dt
        p[0] += p[2] * dt
        p[1] += p[3] * dt
        if floor is None or p[1] < floor:
            out.append(p)
    return out
