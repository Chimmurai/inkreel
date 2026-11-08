"""Режим «Символи»: те саме решето з двох кольорів, але шматочки - справжні знаки шрифту, складені сіткою, як текст.

Чорнило й символи - два боки однієї філософії: кадр складається лише з чорного й білого, а тон передається густиною.
Чорнило малює штрихи й крапки де завгодно, символи - лише з готового набору знаків у клітинках сітки. Усе, що стоїть ДО малювання
(знебарвлення, тонова карта, автоконтраст, пам'ять кадрів), у двох режимів спільне; результат - та сама маска {0,1}, тож програвач,
експорт і палітри не змінюються.

ШРИФТ І УМОВИ РЕНДЕРУ - як у AsciiForge: Consolas із теки Windows (читається під час роботи, у програму не вкладається), клітинка
вдвічі вища за ширину (8x16), кегль = висота клітинки, якір 'la' у (0, 0), знаки білі на чорному, як у терміналі. Немає Consolas (інша
система) - вбудований DejaVu Sans Mono (вільна ліцензія). Алфавіт - майже весь шрифт: латиниця з діакритикою, грецька, кирилиця, рамки,
блоки, геометрія, стрілки, математика (без комбінівних знаків: вони ламають сітку тексту).

ЯК ВИБИРАЄТЬСЯ ЗНАК (рівні 3-5):
  1. Тон. Той самий тон, що й у чорнила (згладжена основа + деталі + локальне нормування), вирівняний за процентилями кадру. Далі він
     ріжеться НЕ на кілька сходинок, а на багато сірого: 8 (рівень 3), 16 (4) або 32 (5) рівні, з синім шумом між рівнями - тому
     плавні переходи лишаються плавними, а текстура поля живе (різні знаки однакової густини).
  2. Піксель до пікселя. Клітинка кадру (16x8 значень) порівнюється з бітами кожного знака: вартість = сума квадратів різниць
     + СИЛЬНИЙ штраф за різницю густини (як у AsciiForge: 5 за відсоток) - знак має ту саму «масу», що й клітинка, а форма вирішує, який
     саме з однаково густих (лінія, куточок, літера, крапка) стане на цьому місці. Нічиї розв'язує постійний «жереб» клітинки (без мерехтіння).
  2b. Крупна структура. Розкид середніх по блоках 4x4 (0 - рівна текстура, велике - край, штрих, літера) у клітинки й у знака мають збігатись:
     пласка клітинка (заставка, небо, шкіра) бере рівну текстуру - ░▒▓, крапки, а не довільну літеру тієї ж густини (вона давала «салат»), а на краях і
     літерах діє лише збіг пікселів, як і раніше.
  2c. Світлі тони. Білих знаків густіше за ▒ (0.5) майже нема (▓ 0.83, █ 1), тож густина рівної клітинки між ними - не найближча з двох, а
     змішується між клітинками за синім шумом (_gap_dither): середній тон точний, знаки лишаються білими на чорному (без інверсії).
  2d. Різноманіття в рівних клітинках. У Consolas між ▒ (0.5) і █ майже немає густих знаків, а на рівному тлі однакових за густиною знаків мало, тож
     поля ставали смугами з одного знака. У пласкій клітинці «жереб» сильніший (FLAT_JIT) і штраф за крупну структуру слабший (FLAT_LAM): знаки майже
     однакової густини, але різної форми, змагаються, і поле стає текстурою. Жереб сталий для клітинки, тож відео не мерехтить; краї й літери не змінюються.
  3. Класи знаків. Знаки з замкненою діркою у густому тілі (◘ ◙ ▣ ...) не вживаються - вони давали у світлих полях ряди «квадратів із крапкою»;
     неповні блоки (▇ ▉ ...) теж - смугастість. Суцільний █ - лише там, де клітинка справді світла (густина вирішує сама); напівтіні ░▒▓
     трохи дорожчі; решта алфавіту безкоштовна - рідкісні знаки з'являються там, де їхня форма точніша.
  4. Час (рівень 5). Тон кадрів усереднюється, порогові таблиці згладжуються, знак клітинки тримається, поки зміна не стала помітною.
Алфавіт росте з рівнем: 3 - ASCII, Latin-1, рамки, блоки, геометрія; 4 - ще розширена латиниця, грецька, кирилиця, символи; 5 - весь шрифт.
Рівні 1-2 - сходинки густини (' ' . : - = + * # % @ █) за середнім тоном клітинки: 1 - жорсткі, 2 - сходинки чергуються за синім шумом.
Клітинка канонічна: 8x16, 16x32 або 32x64 px (три позиції «Щільності знаків»), на будь-якому розмірі кадру; тільки на зовсім малих кадрах
вона зменшується, щоб у кадрі лишилося хоч 20 знаків по ширині й 5 рядків. Усе важке (тон, збіг) рахується на сітці 16x8 значень на клітинку.
Текстовий вихід (.txt): ті самі знаки рядками (густий знак = світла клітинка, як у терміналі на чорному)."""
from __future__ import annotations

import contextlib
import os
import threading
import unicodedata
from functools import lru_cache

import cv2
import numpy as np

from . import core as C

CW_MIN = 8                      # клітинка не вужча за 8 px: знак лишається знаком
# три позиції «Щільності знаків» - канонічні клітинки (ширина; висота вдвічі більша): 1 - 32x64 (крупно), 2 - 16x32, 3 - 8x16 (дрібно, типово).
# Розмір не залежить від кадру: на 4096 px це 128 / 256 / 512 знаків по ширині, на 1920 - 60 / 120 / 240
DENS_CW = (32, 16, 8)
DENS_N = len(DENS_CW)
DENS_DEFAULT = 3
FALLBACK_FONT = "DejaVuSansMono.ttf"      # у assets: на системах без Consolas
LEVELS = {3: 8, 4: 16, 5: 32}   # на скільки рівнів сірого ріжеться кадр (AsciiForge: 6)
EQ = 0.5                        # наскільки тон вирівнюється за процентилями (0 - ні, 1 - повністю)
WD = 3.9                        # штраф за різницю густини, на одиницю густини й піксель клітинки (AsciiForge: 500 / 128)
JITTER = 1.5                    # «жереб» для нічиїх: амплітуда у тих самих одиницях, що й штрафи (1 = помилка в одному пікселі)
FLAT_SPAN = 0.12                # крупна структура клітинки (std блоків 4x4), до якої вона вважається «рівною»: від 0 (пласка) до цього числа вплив лінійно слабне
FLAT_JIT = 12.0                 # у пласкій клітинці «жереб» сильніший у стільки разів: знаки майже однакової густини (різної форми) змагаються, поле не стає смугами ▒▓
FLAT_LAM = 0.5                  # ... а штраф за крупну структуру знака слабшає до цієї частки: помірно «зернистий» знак теж годиться пласкій клітинці
HYST = 0.03                     # утримання знака між кадрами, частка від кількості пікселів клітинки (рівень 5)
HYST_STILL = 6.0                # у клітинці, де тон майже не змінився, утримання більше в стільки+1 разів (нерухоме не дріботить), у рухомій - HYST
STILL_D = 0.06                  # середня зміна тону в клітинці між кадрами (0..1), з якої клітинка вважається рухомою
GAP_MIN = 0.04                  # прогалина між сусідніми густинами знаків, від якої густину клітинки змішують між двома сусідніми (див. _gap_dither)
GAP_FLAT = 0.06                 # ... лише в рівних клітинках (крупна структура до цього числа): на краях і літерах густина вирішує, як і раніше
GAP_COARSE = 0.1                # знак годиться рівній клітинці, якщо його крупна структура не більша (решта - штрихи, куточки, половинки блоків)
GAP_KEEP = 0.12                 # утримання вибору «нижча/вища густина» між кадрами: частка проміжку, всередині якої клітинка лишає минулий вибір
WARM = 3                        # скільки кадрів перед шматком відео розігріває пам'ять рівня 5 повністю (з вибором знаків: так утримання знаків стає таким, як у послідовному розрахунку)
WARM_CHEAP = 8                  # ... а перед ними стільки кадрів - лише тон (повільне усереднення тону й вирівнювання за ці кадри встигає зійтись)

G_BLANK, G_ZOO, G_BLOCK, G_SHADE, G_SOLID, G_PART, G_HOLE = range(7)
# штраф за клас знака (у частках від кількості пікселів клітинки; None - клас недоступний).
# Суцільний █ - лише для справді білого (на яскравих клітинках густина вирішує сама, він найточніший); неповні блоки ▇ ▉ - ні, смугастість;
# «квадрат із діркою» (◘ ◙ ▣ ...) - ні: у світлому полі виходили ряди крапок у квадратах
PEN = {
    3: {G_BLANK: 0.0, G_ZOO: 0.0, G_BLOCK: 0.0, G_SHADE: 0.05, G_SOLID: 0.02, G_PART: None, G_HOLE: None},
    4: {G_BLANK: 0.0, G_ZOO: 0.0, G_BLOCK: 0.0, G_SHADE: 0.03, G_SOLID: 0.02, G_PART: None, G_HOLE: None},
    5: {G_BLANK: 0.0, G_ZOO: 0.0, G_BLOCK: 0.0, G_SHADE: 0.03, G_SOLID: 0.02, G_PART: None, G_HOLE: None},
}
# діапазони Unicode: (від, до, з якого рівня доступні). Порядок важливий: при однакових бітах лишається перший (ASCII)
RANGES = (
    (0x20, 0x7E, 3), (0xA1, 0xFF, 3), (0x2500, 0x25FF, 3), (0x2022, 0x2022, 3),
    (0x100, 0x24F, 4), (0x370, 0x3FF, 4), (0x400, 0x52F, 4), (0x2000, 0x20CF, 4), (0x2100, 0x27FF, 4), (0x2900, 0x2BFF, 4),
    (0x250, 0x2FF, 5), (0x530, 0x58F, 5), (0x1D00, 0x1DBF, 5), (0x1E00, 0x1FFF, 5), (0x2C60, 0x2C7F, 5), (0xA720, 0xA7FF, 5),
    (0xFB00, 0xFB06, 5),
    (0xE3F, 0xE3F, 5), (0x2E17, 0x2E17, 5), (0xA717, 0xA71F, 5), (0xFB13, 0xFB17, 5), (0xFFFC, 0xFFFC, 5),      # решта знаків Consolas із набору користувача (1676), що не потрапили в діапазони вище
)
_BAD = {"Mn", "Mc", "Me", "Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp", "Zs"}      # комбінівні, службові, пробіли (звичайний ' ' беремо окремо)
_LOCK = threading.Lock()
_NB_LOCK = threading.RLock()
_NB_SAFE = None
_RAMP = " .:-=+*#%@"            # сходинки густини для рівнів 1-2 (зверху додається суцільний █)
_CHUNK = 2048


def _guard():
    """Шар потоків numba «workqueue» не витримує паралельних викликів з кількох потоків; omp й tbb - витримують"""
    global _NB_SAFE
    if _NB_SAFE is None and C._nb is not None:
        try:
            _NB_SAFE = C._nb.threading_layer() in ("omp", "tbb")
        except Exception:  # noqa: BLE001
            pass
    return contextlib.nullcontext() if _NB_SAFE else _NB_LOCK


def dens_of(d) -> int:
    """позиція «Щільності» 1..DENS_N (1 - крупно, DENS_N - дрібно)"""
    try:
        return min(DENS_N, max(1, int(d)))
    except (TypeError, ValueError):
        return DENS_DEFAULT


def cell_for(w: int, h: int, dens: int = DENS_DEFAULT) -> tuple[int, int]:
    """розмір клітинки (cw, ch) у px: канонічна для позиції (32x64, 16x32, 8x16) на будь-якому кадрі, що вміщає хоч один знак. Інтерфейс не дає зробити
    результат меншим за одну клітинку (video.glyph_cell); щоб кадр, переданий напряму (CLI, тести), не ламався, клітинка лише тоді меншає вдвічі,
    доки не вміститься"""
    cw = DENS_CW[dens_of(dens) - 1]
    while cw > CW_MIN and (w < cw or h < 2 * cw):
        cw //= 2
    return cw, 2 * cw


def grid_for(w: int, h: int, dens: int = DENS_DEFAULT) -> tuple[int, int]:
    """(рядків, стовпців) знаків для кадру w x h"""
    cw, ch = cell_for(w, h, dens)
    return -(-h // ch), -(-w // cw)


def _starts(n: int, k: int) -> np.ndarray:
    """n px -> k цілих частин, що різняться не більше ніж на 1 px: початки частин (k + 1 значень, останнє = n)"""
    return (np.arange(k + 1, dtype=np.int64) * n) // k


def edge_sizes(w: int, h: int, cw: int, ch: int) -> tuple[np.ndarray, np.ndarray]:
    """видимі розміри клітинок кадру w x h: усі повні cw x ch, крім останнього рядка й стовпчика (їх видно лише частково). Сітка РІВНА:
    раніше зайві пікселі кадру, не кратного клітинці, розкидались по рядках (кожен ~3-й стискався на 1 px), і візерунок ░▒▓ на межі таких
    рядків зсувався на піксель - вздовж кадру ішли шви «накладання». Тепер крок знаків однаковий усюди, а неповний знак лишається на краю
    кадру, як у терміналі. -> (висоти рядків, ширини стовпчиків)"""
    rows, cols = -(-h // ch), -(-w // cw)
    vh, vw = np.full(rows, ch, np.int64), np.full(cols, cw, np.int64)
    vh[-1], vw[-1] = h - (rows - 1) * ch, w - (cols - 1) * cw
    return vh, vw


def canvas_img(img: np.ndarray, w: int, h: int, cw: int, ch: int) -> np.ndarray:
    """кадр -> кадр, добитий до цілого числа клітинок (повторенням краю): тон рахується на всій сітці однаковим кроком, а не
    розтягує кадр по клітинках. Кадр, кратний клітинці, повертається як є"""
    rows, cols = -(-h // ch), -(-w // cw)
    ex, ey = cols * cw - w, rows * ch - h
    if not ex and not ey:
        return img
    ih, iw = img.shape[:2]
    px, py = int(round(iw * ex / w)), int(round(ih * ey / h))
    if not px and not py:
        return img
    return cv2.copyMakeBorder(img, 0, py, 0, px, cv2.BORDER_REPLICATE)


def fdims(cw: int) -> tuple[int, int]:
    """сітка значень однієї клітинки (висота, ширина): 16x8 для клітинок від 8 px, 8x4 для дрібніших (кратні 4 - потрібно для _coarse)"""
    fw = 8 if cw >= 8 else 4
    return 2 * fw, fw


# ------------------------------------------------------------------ шрифт і банк знаків
def font_path() -> tuple[str, bool]:
    """-> (шлях до шрифту, це Consolas?). Спершу свій шрифт (INKREEL_GLYPH_FONT), потім Consolas Windows, потім вбудований"""
    own = os.environ.get("INKREEL_GLYPH_FONT")
    if own and os.path.isfile(own):
        return own, "consola" in os.path.basename(own).lower()
    win = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or r"C:\Windows"
    cands = [os.path.join(win, "Fonts", "consola.ttf")]
    la = os.environ.get("LOCALAPPDATA")
    if la:
        cands.append(os.path.join(la, "Microsoft", "Windows", "Fonts", "consola.ttf"))
    for p in cands:
        if os.path.isfile(p):
            return p, True
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", FALLBACK_FONT), False


def _bits(a: np.ndarray) -> np.ndarray:
    """тінь знака (0..1) -> біти: поріг за середнім (обмежений 0.12..0.5), тож тонкі штрихи не рвуться"""
    thr = float(np.clip(a.mean(), 0.12, 0.5))
    return (a > thr).astype(np.uint8)


def _tier(c: str) -> int:
    """з якого рівня якості знак доступний (3 - з початку, 5 - лише в «Ультрі»)"""
    o = ord(c)
    for lo, hi, t in RANGES:
        if lo <= o <= hi:
            return t
    return 5


def _solid(k: np.ndarray) -> bool:
    """густий знак без «дірок» усередині своєї рамки (█ ▇ ▉ ...); решта густих (◘ - з діркою) сюди не потрапляє"""
    d = float(k.mean())
    if d < 0.6:
        return False
    ys, xs = np.nonzero(k)
    box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return (box - int(k.sum())) < 0.03 * k.size


def _hole(k: np.ndarray) -> bool:
    """«ящик із чимось усередині» (◘ ◙ ▣ ▦ ...): густий знак на всю ширину клітинки, у якому є замкнена діра, а разом із дірою він -
    майже повний прямокутник. Букви (B 8 @ D) вужчі за клітинку й не прямокутні; одиночні темні піксели (▓ ▒) дірками не вважаються"""
    if float(k.mean()) < 0.45:
        return False
    dark = (k == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    h, w = k.shape
    inner = np.zeros(k.shape, bool)
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a >= 3 and x > 0 and y > 0 and x + bw < w and y + bh < h:
            inner |= lab == i
    if not inner.any():
        return False
    full = (k > 0) | inner
    ys, xs = np.nonzero(full)
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    return bool(bw >= w - 1 and bh >= h // 3 and full.sum() >= 0.9 * bw * bh)


_BOXES = "\u25d8\u25d9"      # ◘ ◙ («інверсна куля»): ящик із круглою діркою; у деяких шрифтах дірка торкається краю, тож за формою їх не впіймати


def _class(c: str, k: np.ndarray) -> int:
    o = ord(c)
    if c == " ":
        return G_BLANK
    if 0x2591 <= o <= 0x2593:
        return G_SHADE
    if c in _BOXES or _hole(k):
        return G_HOLE
    if _solid(k):
        return G_SOLID if float(k.mean()) >= 0.97 else G_PART
    if 0x2580 <= o <= 0x259F:
        return G_BLOCK
    return G_ZOO


class Bank:
    """набір знаків для клітинки cw x ch: біти (G, ch, cw) (1 = світлий піксель знака), значення на сітці 16x8, густина, класи, рівні"""

    def __init__(self, cw: int, ch: int, chars: list[str] | None = None, ink: np.ndarray | None = None):
        self.cw, self.ch = cw, ch
        if chars is None:
            chars, ink = self._render(cw, ch)
        self.chars = list(chars)
        self.ink = np.ascontiguousarray(ink, dtype=np.uint8)               # (G, ch, cw)
        n = len(self.chars)
        self.dens = self.ink.reshape(n, -1).mean(axis=1).astype(np.float32)
        self.group = np.array([_class(c, k) for c, k in zip(self.chars, self.ink)], np.int8)
        self.tier = np.array([_tier(c) for c in self.chars], np.int8)
        fh, fw = fdims(cw)
        if (fh, fw) == (ch, cw):
            f = self.ink.reshape(n, -1).astype(np.float32)
        else:
            f = np.stack([cv2.resize(k.astype(np.float32), (fw, fh), interpolation=cv2.INTER_AREA).ravel() for k in self.ink])
        self.feat = np.ascontiguousarray(f, dtype=np.float32)               # (G, D): значення знака на сітці клітинки
        self.D = self.feat.shape[1]
        self.fn = (self.feat * self.feat).sum(axis=1)
        self.fdens = self.feat.mean(axis=1)
        self.coarse = _coarse(self.feat, fh, fw)                           # (G,): «крупна структура» знака: розкид середніх по блоках 4x4
        self.index = {c: i for i, c in enumerate(self.chars)}
        self.ramp = self._ramp()
        self._lv: dict = {}
        self._ad: dict = {}

    @staticmethod
    def _render(cw: int, ch: int) -> tuple[list[str], np.ndarray]:
        from PIL import Image, ImageDraw, ImageFont
        path, real = font_path()
        if real:                                                         # Consolas - як у AsciiForge: кегль = висота клітинки, якір 'la' у (0, 0)
            size = float(ch)
        else:                                                            # інший шрифт: кегль, за якого його ascent+descent = висота клітинки
            a, d = ImageFont.truetype(path, size=100).getmetrics()
            size = ch * 100.0 / max(1, a + d)
        font = ImageFont.truetype(path, size=max(4, int(round(size))))

        def draw(c: str) -> np.ndarray:
            im = Image.new("L", (cw, ch), 0)
            if c != " ":
                ImageDraw.Draw(im).text((0, 0), c, font=font, fill=255)
            return np.asarray(im).astype(np.float32) / 255.0

        missing = _bits(draw("\U0010fffe")).tobytes()                    # знак, якого шрифт не має, малює «порожню рамку»
        seen: set[bytes] = set()
        chars: list[str] = []
        inks: list[np.ndarray] = []
        for lo, hi, _t in RANGES:
            for o in range(lo, hi + 1):
                c = chr(o)
                if o != 0x20 and unicodedata.category(c) in _BAD:
                    continue
                k = np.ones((ch, cw), np.uint8) if c == "█" else _bits(draw(c))
                if c != " " and (k.sum() == 0 or k.tobytes() == missing):
                    continue
                key = k.tobytes()
                if key in seen:                                        # однакові на цьому розмірі знаки - один (як у symbol_screen.py)
                    continue
                seen.add(key)
                chars.append(c)
                inks.append(k)
        return chars, np.stack(inks)

    def _ramp(self) -> np.ndarray:
        """знаки для сходинок густини: порожній, ASCII-сходинки, а зверху - суцільний █ (білий лишається білим), якщо його немає - найгустіший
        знак без дірок"""
        out = [self.index[c] for c in _RAMP if c in self.index]
        full = [i for i in range(len(self.chars)) if self.group[i] == G_SOLID]
        if full and out:
            top = max(full, key=lambda i: self.dens[i])
        else:
            ok = [i for i in range(len(self.chars)) if self.group[i] == G_ZOO and self.chars[i].isprintable()
                  and not 0x2580 <= ord(self.chars[i]) <= 0x259F]
            top = max(ok, key=lambda i: self.dens[i]) if ok and out else None
        if top is not None and self.dens[top] > self.dens[out].max() + 0.05:
            out.append(top)
        if not out:
            out = [int(np.argmin(self.dens)), int(np.argmax(self.dens))]
        out.sort(key=lambda i: self.dens[i])
        clean = [out[0]]
        for i in out[1:]:
            if self.dens[i] - self.dens[clean[-1]] >= 0.02:
                clean.append(i)
        return np.array(clean, np.int32)

    def level(self, level: int):
        """(індекси знаків рівня, 2*значення^T, зсув = -(|значення|^2 + штраф), густина, позиція знака в наборі або -1); кешується"""
        got = self._lv.get(level)
        if got is not None:
            return got
        lv = min(5, max(3, level))
        pens = PEN[lv]
        pen = np.array([1e3 if pens.get(int(g)) is None else pens[int(g)] for g in self.group], np.float32)
        use = (self.tier <= lv) & (pen < 1e2)
        sel = np.nonzero(use)[0]
        pos = np.full(len(self.chars), -1, np.int32)
        pos[sel] = np.arange(len(sel), dtype=np.int32)
        got = (sel, np.ascontiguousarray(2.0 * self.feat[sel].T), (-(self.fn[sel] + pen[sel] * self.D)).astype(np.float32),
               self.fdens[sel].copy(), pos, self.coarse[sel].copy())
        self._lv[level] = got
        return got


def _ladder_of(bank: Bank, level: int) -> np.ndarray:
    """відсортовані різні густини знаків рівня, придатних для рівної клітинки (кешується в банку): між ▒ й ▓ інших рівних текстур немає -
    половинки блоків ▄ ▐ й куточки ▙ ▜ густини 0.5-0.75 мають велику крупну структуру, у рівному полі вони дають смуги й не вживаються"""
    got = bank._ad.get(level)
    if got is None:
        _, _, _, dens, _, sg = bank.level(level)
        got = bank._ad[level] = np.unique(dens[sg <= GAP_COARSE])
    return got


@lru_cache(maxsize=12)
def _bank(cw: int, ch: int) -> Bank:
    with _LOCK:
        return Bank(cw, ch)


def bank_for(w: int, h: int, dens: int = DENS_DEFAULT) -> Bank:
    return _bank(*cell_for(w, h, dens))


# ------------------------------------------------------------------ тон
if C._nb is not None:
    @C._nb.njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _smooth_nb(t, old, out):
        h, w = t.shape
        c5 = np.float32(1.0) / np.float32(0.2)
        for y in C._nb.prange(h):
            for x in range(w):
                d = t[y, x] - old[y, x]
                a = min(max(abs(d) * c5, np.float32(0.3)), np.float32(1.0))
                v = old[y, x] + a * d
                old[y, x] = v
                out[y, x] = v

    @C._nb.njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _eq_nb(t, qq, cdf, w, m, out):
        """((1 - m)·t + m·y), y = (1 - w)·interp(t; qq -> 0..1) + w·interp(t; 0..1 -> cdf): одним проходом (раніше два np.interp у float64 і ще п'ять проходів)"""
        h, wd = t.shape
        nq = qq.shape[0]
        nc = cdf.shape[0] - 1
        one = np.float32(1.0)
        for y in C._nb.prange(h):
            for x in range(wd):
                v = t[y, x]
                if v <= qq[0]:
                    a = np.float32(0.0)
                elif v >= qq[nq - 1]:
                    a = one
                else:
                    lo = 0
                    hi = nq - 1
                    while hi - lo > 1:
                        mid = (lo + hi) >> 1
                        if qq[mid] <= v:
                            lo = mid
                        else:
                            hi = mid
                    a = (lo + (v - qq[lo]) / (qq[lo + 1] - qq[lo])) / np.float32(nq - 1)
                if w > 0.0:
                    if v <= 0.0:
                        b = cdf[0]
                    elif v >= 1.0:
                        b = cdf[nc]
                    else:
                        pos = v * np.float32(nc)
                        j = min(int(pos), nc - 1)
                        b = cdf[j] + (cdf[j + 1] - cdf[j]) * (pos - j)
                    a = (one - w) * a + w * b
                out[y, x] = (one - m) * v + m * a


def _smooth(t: np.ndarray, mem: dict) -> np.ndarray:
    """часове усереднення тону (рівень 5): де кадр майже не змінився - сильно, де рухається - наздоганяємо"""
    old = mem.get("T")
    if old is None or old.shape != t.shape:
        mem["T"] = t.copy()
        return t
    if C._nb is not None and t.dtype == np.float32 and old.dtype == np.float32 and t.ndim == 2 and t.flags.c_contiguous and old.flags.c_contiguous:
        out = np.empty_like(t)
        _smooth_nb(t, old, out)
        return out
    a = np.clip(np.abs(t - old) / 0.2, 0.3, 1.0)
    old += a * (t - old)
    return old.copy()


EQ_BINS = 64
EQ_CLIP = 2.5            # у «плоскому» вирівнюванні жодна сходинка тону не отримує більше за стільки середніх часток кадру
EQ_BLEND = 0.4            # часова інерція вирівнювання: частка нового кадру в пам'яті процентилів
EQ_PLATEAU = (0.35, 0.6)  # частка кадру в одній з 64 сходинок тону: від першого до другого числа вирівнювання плавно стає «плоским»


def _equalize(t: np.ndarray, mem: dict | None) -> np.ndarray:
    """вирівнювання тону за процентилями кадру (кожна сходинка тону отримує порівну пікселів): півтони розходяться ширше, насичені білі й чорні
    плями трохи стискаються - тож світле у фото лишається текстурою, а не пласкою плямою. Майже рівний кадр не чіпаємо - шум став би візерунком.
    Коли ж більшість кадру - одна пласка ділянка (заставка, слайд, титри), процентилі їй нічого путнього не скажуть: усі «сходинки» цієї ділянки
    збігаються, і вона отримувала довільний тон посередині (біле поле ставало шумом ▀■). У такому кадрі переходимо (плавно) на вирівнювання з
    обмеженням, як CLAHE: пласка ділянка лишається собою"""
    sub = t[::3, ::3] if t.shape[0] > 96 else t
    q = np.percentile(sub, np.linspace(0, 100, 33)).astype(np.float32)
    m = EQ * float(np.clip((float(q[31] - q[1]) - 0.05) / 0.25, 0.0, 1.0))
    h = np.bincount(np.clip((sub * EQ_BINS).astype(np.int32), 0, EQ_BINS - 1).ravel(), minlength=EQ_BINS).astype(np.float64)
    h /= max(h.sum(), 1.0)
    cap = EQ_CLIP / EQ_BINS
    h = np.minimum(h, cap) + np.maximum(h - cap, 0.0).sum() / EQ_BINS   # зрізане повертаємо порівну - відображення лишається строго зростаючим
    cdf = np.concatenate([[0.0], np.cumsum(h)]).astype(np.float32)
    cdf /= cdf[-1]
    lo, hi = EQ_PLATEAU
    w = float(np.clip((float(np.max(np.bincount(np.clip((sub * EQ_BINS).astype(np.int32), 0, EQ_BINS - 1).ravel(), minlength=EQ_BINS))) / sub.size - lo) / (hi - lo), 0.0, 1.0))
    if mem is not None:
        old = mem.get("q")
        if old is not None and float(np.abs(old - q).max()) < 0.25:       # різкий стрибок - інша сцена
            q = old + EQ_BLEND * (q - old)
            cdf = mem["cdf"] + EQ_BLEND * (cdf - mem["cdf"])
            w = mem["w"] + EQ_BLEND * (w - mem["w"])
        mem["q"], mem["cdf"], mem["w"] = q.copy(), cdf.copy(), w
    if m <= 0.0:
        return t
    qq = np.maximum.accumulate(q + np.arange(33, dtype=np.float32) * 1e-5)
    if C._nb is not None and t.dtype == np.float32 and t.ndim == 2 and t.flags.c_contiguous:
        out = np.empty_like(t)
        _eq_nb(t, qq.astype(np.float32), np.ascontiguousarray(cdf, dtype=np.float32), np.float32(w), np.float32(m), out)
        return out
    y = np.interp(t, qq, np.linspace(0.0, 1.0, 33, dtype=np.float32)).astype(np.float32)
    if w > 0.0:
        y2 = np.interp(t, np.linspace(0.0, 1.0, EQ_BINS + 1, dtype=np.float32), cdf).astype(np.float32)
        y = ((1.0 - w) * y + w * y2).astype(np.float32)
    return ((1.0 - m) * t + m * y).astype(np.float32)


LAM = 10.0                      # вага «крупної структури» (на піксель клітинки): пласка клітинка тягне рівномірну текстуру (░▒▓, крапки), а не випадкову букву


def _coarse(F: np.ndarray, fh: int, fw: int) -> np.ndarray:
    """(n, fh*fw) значень клітинок -> (n,): розкид середніх по блоках 4x4 (0 - рівна текстура, велике - край, штрих, літера)"""
    n = F.shape[0]
    b = F.reshape(n, fh // 4, 4, fw // 4, 4).mean(axis=(2, 4)).reshape(n, -1)
    return b.std(axis=1).astype(np.float32)


if C._nb is not None:
    @C._nb.njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _cell_stats_nb(F, fh, fw):
        """одним проходом: середнє клітинки й розкид середніх по блоках 4x4 (те саме, що F.mean(axis=1) і _coarse)"""
        n = F.shape[0]
        bh, bw = fh // 4, fw // 4
        nb = bh * bw
        fm = np.empty(n, np.float32)
        si = np.empty(n, np.float32)
        for i in C._nb.prange(n):
            tot = 0.0
            s1 = 0.0
            s2 = 0.0
            for by in range(bh):
                for bx in range(bw):
                    acc = 0.0
                    for dy in range(4):
                        o = (by * 4 + dy) * fw + bx * 4
                        for dx in range(4):
                            acc += F[i, o + dx]
                    tot += acc
                    b = acc * 0.0625
                    s1 += b
                    s2 += b * b
            fm[i] = tot / (fh * fw)
            mu = s1 / nb
            si[i] = np.sqrt(max(s2 / nb - mu * mu, 0.0))
        return fm, si


def _cells(a: np.ndarray, rows: int, cols: int, fh: int, fw: int) -> np.ndarray:
    """карта (rows*fh, cols*fw) -> клітинки (rows*cols, fh*fw)"""
    return a.reshape(rows, fh, cols, fw).transpose(0, 2, 1, 3).reshape(rows * cols, fh * fw)


# ------------------------------------------------------------------ вибір
_JT = (np.random.default_rng(1234).random(4096).astype(np.float32) * np.float32(JITTER))      # таблиця «жеребу»: у кожної клітинки свій зсув у ній

if C._nb is not None:
    @C._nb.njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _pick_nb(S, bias, dens, fm, wdd, prev, hyst, jt, n0, out, sg, si, lam, jm, lm):         # bias: зсув знака (-|f|^2 - штраф класу), додається тут, а не окремим проходом по всьому S; hyst: утримання для кожної клітинки (n,); jm, lm: сила «жеребу» й штрафу крупної структури для кожної клітинки
        """S (n, G): 2F.f; bias (G,): -|f|^2 - штраф класу; віднімаємо штраф за різницю густини й крупної структури, додаємо утримання й «жереб», беремо найкращий"""
        n, G = S.shape
        for i in C._nb.prange(n):
            m = fm[i]
            q = si[i]
            lq = lam * lm[i]
            jq = jm[i]
            best = -3.0e38
            bi = 0
            p = prev[i]
            o = (((n0 + i) * 2654435761) >> 8) & 4095
            row = S[i]
            for g in range(G):
                d = sg[g] - q
                v = row[g] + bias[g] - wdd * abs(dens[g] - m) - lq * d * d + jt[(g + o) & 4095] * jq
                if v > best:
                    best = v
                    bi = g
            if p >= 0:                                      # утримання минулого знака: перевіряємо, чи він не поступається найкращому менш ніж на hyst
                d = sg[p] - q
                v = row[p] + bias[p] - wdd * abs(dens[p] - m) - lq * d * d + jt[(p + o) & 4095] * jq + hyst[i]
                if v > best:
                    bi = p
            out[i] = bi


def _pick(S: np.ndarray, dens: np.ndarray, fm: np.ndarray, wdd: float, prev: np.ndarray, hyst, n0: int,
          sg: np.ndarray | None = None, si: np.ndarray | None = None, lam: float = 0.0,
          jm: np.ndarray | None = None, lm: np.ndarray | None = None, bias: np.ndarray | None = None) -> np.ndarray:
    n = S.shape[0]
    out = np.empty(n, np.int32)
    jm = np.ones(n, np.float32) if jm is None else np.ascontiguousarray(jm, dtype=np.float32)
    lm = np.ones(n, np.float32) if lm is None else np.ascontiguousarray(lm, dtype=np.float32)
    hyst = np.ascontiguousarray(np.broadcast_to(np.asarray(hyst, np.float32), (n,)))       # число - однакове для всіх клітинок, масив - своє
    if sg is None:
        sg, si, lam = np.zeros(S.shape[1], np.float32), np.zeros(n, np.float32), 0.0
    if C._nb is not None:
        with _guard():
            _pick_nb(S, np.zeros(S.shape[1], np.float32) if bias is None else np.ascontiguousarray(bias, dtype=np.float32),
                     dens, fm, wdd, prev, hyst, _JT, n0, out, sg, si, float(lam), jm, lm)
        return out
    if bias is not None:
        S += bias
    S -= wdd * np.abs(dens[None, :] - fm[:, None])
    S -= (lam * lm)[:, None] * (sg[None, :] - si[:, None]) ** 2
    S += _JT[(np.arange(S.shape[1])[None, :] + (((n0 + np.arange(n)) * 2654435761 >> 8) & 4095)[:, None]) & 4095] * jm[:, None]
    ok = np.nonzero(prev >= 0)[0]
    S[ok, prev[ok]] += hyst[ok]
    return S.argmax(axis=1).astype(np.int32)


def _gap_dither(fm: np.ndarray, si: np.ndarray, ad: np.ndarray, rows: int, cols: int, prev_d: np.ndarray | None = None) -> np.ndarray:
    """Світлі тони без сходинок. У наборі знаків (білих на чорному) між густинами є великі прогалини - найбільша між ▒ (0.5) і ▓ (0.83): тон між ними
    округлявся до найближчого й лягав плато. Тут густина, до якої тягнеться клітинка, - не точний тон, а одна з двох сусідніх доступних густин;
    яка саме - вирішує синій шум клітинки (як у рівні 2): середня густина по сусідніх клітинках точна, а знаки лишаються справжніми.
    Лише для рівних клітинок (на краях і літерах вирішує форма). prev_d - густина минулого знака: у межах GAP_KEEP від порога вибір не змінюється"""
    k = np.clip(np.searchsorted(ad, fm, side="right"), 1, len(ad) - 1)
    lo, hi = ad[k - 1], ad[k]
    gap = hi - lo
    use = (gap > GAP_MIN) & (si < GAP_FLAT) & (fm > lo) & (fm < hi)
    if not use.any():
        return fm
    bn = C._tile(C._blue(), cols, rows, 1).reshape(-1)[:fm.size].astype(np.float32)
    d = (fm - lo) / np.maximum(gap, 1e-6) - bn                           # > 0 - вища густина
    up = d > 0
    if prev_d is not None:
        same = (np.abs(prev_d - lo) < 1e-4) | (np.abs(prev_d - hi) < 1e-4)          # минулий знак стояв на одному з цих двох щаблів
        hold = same & (np.abs(d) < GAP_KEEP)
        up = np.where(hold, np.abs(prev_d - hi) < 1e-4, up)
    return np.where(use, np.where(up, hi, lo), fm).astype(np.float32)


def _match(tgt: np.ndarray, bank: Bank, rows: int, cols: int, level: int, mem: dict | None, tdiff: np.ndarray | None = None) -> np.ndarray:
    """tgt - рівні, нормовані до 0..1, на сітці клітинок (rows*fh, cols*fw) -> індекси знаків у банку (rows*cols)"""
    sel, fT2, bias, dens, pos, sg = bank.level(level)
    fh, fw = fdims(bank.cw)
    F = np.ascontiguousarray(_cells(tgt, rows, cols, fh, fw), dtype=np.float32)
    if C._nb is not None and fh % 4 == 0 and fw % 4 == 0:
        fm, si = _cell_stats_nb(F, fh, fw)
    else:
        fm = F.mean(axis=1).astype(np.float32)
        si = _coarse(F, fh, fw)
    lam = float(LAM * bank.D)
    n = rows * cols
    prev = np.full(n, -1, np.int32)
    hyst = np.zeros(n, np.float32)
    if mem is not None and level >= 5:
        pi = mem.get("idx")
        if pi is not None and pi.size == n and mem.get("level") == level and mem.get("bank") == (bank.cw, bank.ch):
            prev = pos[pi].astype(np.int32)
            hyst = np.full(n, HYST * bank.D, np.float32)
            if tdiff is not None:                                       # нерухома клітинка тримає знак міцніше: інакше шум відео перемикає її між майже однаковими знаками
                hyst *= (1.0 + HYST_STILL * np.clip(1.0 - tdiff / STILL_D, 0.0, 1.0)).astype(np.float32)
    wdd = float(WD * bank.D)
    ad = _ladder_of(bank, level)
    fl = np.clip(1.0 - si / FLAT_SPAN, 0.0, 1.0).astype(np.float32)             # 1 - пласка клітинка, 0 - край/літера
    jm = (1.0 + (FLAT_JIT - 1.0) * fl).astype(np.float32)
    lm = (1.0 - (1.0 - FLAT_LAM) * fl).astype(np.float32)
    fmd = _gap_dither(fm, si, ad, rows, cols, np.where(prev >= 0, dens[np.maximum(prev, 0)], np.nan) if mem is not None and level >= 5 else None)
    out = np.empty(n, np.int32)
    for a in range(0, n, _CHUNK):
        b = min(n, a + _CHUNK)
        S = F[a:b] @ fT2                                                # 2F.f; зсув знака (- |f|^2 - штраф класу) додає вибір
        out[a:b] = _pick(S, dens, fmd[a:b], wdd, prev[a:b], hyst[a:b], a, sg, si[a:b], lam, jm[a:b], lm[a:b], bias)
    idx = sel[out].astype(np.int32)
    if mem is not None:
        mem["idx"], mem["level"], mem["bank"] = idx.copy(), level, (bank.cw, bank.ch)
    return idx


def _ladder(t: np.ndarray, bank: Bank, rows: int, cols: int, dither: bool) -> np.ndarray:
    """рівні 1-2: сходинки густини за середнім вирівняним тоном клітинки; dither - сусідні сходинки чергуються за синім шумом"""
    m = cv2.resize(t, (cols, rows), interpolation=cv2.INTER_AREA)
    rd = bank.dens[bank.ramp]
    d = m * float(rd[-1])
    k = np.clip(np.searchsorted(rd, d, side="right") - 1, 0, len(rd) - 2)
    lo, hi = rd[k], rd[k + 1]
    frac = np.clip((d - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    up = frac > C._tile(C._blue(), cols, rows, 1) if dither else frac >= 0.5
    return bank.ramp[np.where(up, k + 1, k)].astype(np.int32)


def tone_size(w: int, h: int, dens: int = DENS_DEFAULT) -> tuple[int, int]:
    """розмір тонової карти «Символів» для кадру w x h: значення клітинок на сітці (див. plan)"""
    bank = bank_for(w, h, dens)
    rows, cols = grid_for(w, h, dens)
    fh, fw = fdims(bank.cw)
    return cols * fw, rows * fh


def prime(st, img: np.ndarray, w: int, h: int, auto: bool = True, dens: int = DENS_DEFAULT) -> None:
    """найдешевший прогрів пам'яті нормування тону кадром, що лише декодований (див. ink5.prime), - на сітці «Символів»"""
    if st is not None:
        gw, gh = tone_size(w, h, dens)
        bank = bank_for(w, h, dens)
        C.tone_prime(canvas_img(img, w, h, bank.cw, bank.ch), gw, gh, auto, C.tmem_of(st))


def plan(img: np.ndarray, w: int, h: int, p, st=None, warm_only: bool = False) -> tuple[np.ndarray, Bank]:
    """кадр -> (індекси знаків (rows, cols), банк). st - InkState (пам'ять кадрів) на рівні 5. warm_only - лише оновити пам'ять тону
    (без вибору знаків, це найдорожча частина): дешевий розігрів перед шматком відео"""
    level = min(5, int(p.method))                                        # шостий рівень (нейро) у символах не діє
    dens = dens_of(getattr(p, "gdens", DENS_DEFAULT))
    bank = bank_for(w, h, dens)
    rows, cols = grid_for(w, h, dens)
    fh, fw = fdims(bank.cw)
    gw, gh = cols * fw, rows * fh                                       # тон рахується одразу на сітці значень клітинок
    img = canvas_img(img, w, h, bank.cw, bank.ch)
    mem = tmem = None
    if level >= 5 and st is not None:
        if getattr(st, "glyph", None) is None:
            st.glyph = {}
        mem, tmem = st.glyph, C.tmem_of(st)
    mi, info, lf = None, {"cut": True}, None
    if level >= 5 and st is not None and C._nb is not None:            # відео: шум датчика й стиснення тонмап підсилює в рази (у темних місцях - в 10 і більше),
        from . import ink5                                              # тож його гасимо ДО підсилення - тим самим часовим усередненням світлоти, що й у «Чорнила»
        mi = ink5._mem(st)
        if mi.shape == (gw, gh) and mi.gf is not None and mi.Ls is not None:
            info["cut"] = False
            lf = lambda L: ink5._lfilter(L, mi, ink5.P, info)             # noqa: E731
    with (ink5._guard() if lf is not None else contextlib.nullcontext()):
        t, Lf = C.tone_map(img, gw, gh, p.auto, lf, deband=level >= 4, tmem=tmem)      # тонова карта - та сама, що в чорнила (з пам'яттю нормування у відео)
    if mi is not None:
        mi.shape, mi.gf, mi.Ls = (gw, gh), Lf, ink5._small_l(Lf)
        mi.tdf = mi.F = mi.m = mi.pg = None
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    if mem is not None:
        t = _smooth(t, mem)
    fl, pk = tmem.fade if tmem is not None else (0.0, 1.0)
    if pk < 1.0 or fl > 0.0:                                            # згасання/поява: вирівнюємо кадр у його власному діапазоні тонів,
        span = max(pk - fl, 0.02)                                        # інакше процентилі знову розтягли б тьмяний кадр до повного
        t = (fl + span * _equalize(np.clip((t - fl) / span, 0.0, 1.0).astype(np.float32), mem)).astype(np.float32)
    else:
        t = _equalize(t, mem)
    if warm_only:
        if mem is not None:
            mem["tp"] = t.copy()
        return None, bank
    if level <= 2:
        return _ladder(t, bank, rows, cols, dither=level == 2).reshape(rows, cols), bank
    tdiff = None
    if mem is not None:                                                 # наскільки змінився тон у кожній клітинці (по пікселях): мірило «тут нічого не рухається»
        tp = mem.get("tp")
        if tp is not None and tp.shape == t.shape:
            tdiff = _cells(np.abs(t - tp), rows, cols, fh, fw).mean(axis=1).astype(np.float32)
        mem["tp"] = t.copy()
    nl = LEVELS[min(5, level)]
    bn = C._tile(C._blue(), gw, gh, 1)                                  # сірі рівні перемежовуються синім шумом: середній тон точний
    lev = np.clip(np.floor(t * (nl - 1) + bn), 0, nl - 1)
    tgt = (lev * (1.0 / (nl - 1))).astype(np.float32)
    return _match(tgt, bank, rows, cols, level, mem, tdiff).reshape(rows, cols), bank


def prewarm():
    """прогнати «Символи» (рівні 3 і 5) на двох малих кадрах: збирає банк знаків і компілює (або читає з кешу) ядра тону й вибору знака.
    Викликається в одному процесі пулу перед усіма іншими - решта лише підвантажує готове"""
    if C._nb is None:
        return
    yy, xx = np.mgrid[0:300, 0:534]
    base = (127 + 90 * np.sin(xx / 23.0) * np.cos(yy / 17.0) + 30 * ((xx // 40 + yy // 40) % 2)).clip(0, 255).astype(np.uint8)
    cv2.circle(base, (260, 150), 70, 255, -1)
    img = np.dstack([base, np.roll(base, 7, 1), 255 - base])
    for level in (3, 5):
        p = C.Params(method=level, glyph=True)
        st = C.InkState() if level == 5 else None
        for k in range(2):
            plan(np.roll(img, k * 3, 1), 534, 300, p, st)
    yy, xx = np.mgrid[0:96, 0:160]
    C._deband((20.0 + (xx // 16) * 1.5 + (yy // 16) * 0.7).astype(np.float32), 720)       # «східці» кодека гасяться від 540p: ядра окремо


def render_idx(idx: np.ndarray, bank: Bank, w: int, h: int) -> np.ndarray:
    """індекси знаків -> маска (h, w) uint8, 1 = світло (знак білий на чорному, як у терміналі); сітка рівна, неповний останній рядок/стовпчик обрізається краєм кадру (див. edge_sizes)"""
    rows, cols = idx.shape
    img = bank.ink[idx].transpose(0, 2, 1, 3).reshape(rows * bank.ch, cols * bank.cw)
    return np.ascontiguousarray(img[:h, :w])                           # рівна сітка: неповний останній рядок/стовпчик просто обрізається краєм кадру


def binarize(img: np.ndarray, w: int, h: int, p, st=None) -> np.ndarray:
    idx, bank = plan(img, w, h, p, st)
    return render_idx(idx, bank, w, h)


def to_text(idx: np.ndarray, bank: Bank) -> str:
    """індекси знаків -> текст для .txt: густий знак = світла клітинка (як у терміналі на чорному тлі)"""
    ch = bank.chars
    return "\n".join("".join(ch[i] for i in row).rstrip() for row in idx) + "\n"


def text(img: np.ndarray, w: int, h: int, p, st=None) -> str:
    idx, bank = plan(img, w, h, p, st)
    return to_text(idx, bank)


def text_from_mask(mask: np.ndarray, w: int, h: int, dens: int = DENS_DEFAULT) -> str:
    """готова маска знаків (напр. поточний кадр у вікні) -> текст: клітинки впізнаються за точним збігом із бітами знаків
    (на кадрі, не кратному клітинці, останній рядок і стовпчик - за видимою частиною знака, див. edge_sizes)"""
    bank = bank_for(w, h, dens)
    rows, cols = grid_for(w, h, dens)
    cw, ch = bank.cw, bank.ch
    vh, vw = edge_sizes(w, h, cw, ch)
    m = np.zeros((rows * ch, cols * cw), np.uint8)
    m[:h, :w] = (mask[:h, :w] > 0)
    n_g = len(bank.chars)
    cells = m.reshape(rows, ch, cols, cw).transpose(0, 2, 1, 3)
    tables: dict = {}
    flats: dict = {}
    out = np.zeros((rows, cols), np.int32)
    for r in range(rows):
        for c in range(cols):
            key = (int(vh[r]), int(vw[c]))
            hh, ww = key
            tab = tables.get(key)
            if tab is None:
                tab = tables[key] = {bank.ink[i, :hh, :ww].tobytes(): i for i in range(n_g)[::-1]}      # першим у списку знаком (ASCII) виграє
            cell = cells[r, c, :hh, :ww]
            got = tab.get(cell.tobytes())
            if got is None:                                              # клітинка, що не збіглась (маска не з цього банку): найближчий знак
                fl = flats.get(key)
                if fl is None:
                    fl = flats[key] = bank.ink[:, :hh, :ww].reshape(n_g, -1).astype(np.int16)
                got = int(np.abs(fl - cell.reshape(-1).astype(np.int16)[None, :]).sum(axis=1).argmin())
            out[r, c] = got
    return to_text(out, bank)
