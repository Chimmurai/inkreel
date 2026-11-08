"""Ядро BenVid: єдиний автоматичний бінаризатор кадрів.  Результат: uint8 {0,1}, 1 = світлий піксель.
Жодного сірого: кожен піксель або білий, або чорний. Налаштувань для користувача немає.

Кадр малює «людина з олівцем», а не фільтр, що проганяє всі пікселі однаково:
  1. ТОН. Lab-світлота (кольори дають правильну яскравість) -> автоконтраст -> різкість -> нормування до
     локального середовища -> карта тону 0..1. Для малих розмірів контраст і різкість вищі, щоб тримались деталі.
  2. ЗОНИ. Тон згладжується й ріжеться на куски (зв'язні області одного рівня тону). Для кожного куска
     рахуються: розмір, середній тон, витягнутість, узгодженість напрямку градієнта.
  3. СТИЛЬ куска (м'які ваги, не жорсткий вибір):
       штрихи - зв'язні куски з чітким напрямом форми й витягнуті куски: лінії вздовж ізофот/осі куска,
                у глибоких тінях ще й перехресні;
       крапки - дрібні куски й світлі півтони: нерегулярні плямки, розмір від тону;
       зерно  - великі рівні куски без напряму: дрібне рівне зерно (синій шум).
     У кожному стилі густина білого дорівнює тону, тож тон передається точно.
  4. ПЕРЕХОДИ. Ваги стилів згладжені, а стиль у пікселі вибирає плавне випадкове поле - стилі перетікають
     один в одного плямами, як у малюнку, без швів.
Малі кадри (<200 px по висоті) без штрихів: лише крапки й зерно. Усі масштаби прив'язані до висоти кадру.
Рівень 5 («Гравюра», раніше «Ультра») винесено в ink5.py: той самий розподіл на стилі, але штрихи йдуть за формою предметів.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from functools import lru_cache

import math
import threading

import cv2
import numpy as np

from . import gpu as _gpu

try:                                   # numba прискорює дрібні кадри; без неї вони рахуються простіше
    import numba as _nb
except Exception:  # noqa: BLE001
    _nb = None
else:
    from . import nbcache as _nbcache
    _nbcache.install()                 # у зібраному exe: кеш ядер у теці програми й спільний для вікна та обробників (див. nbcache.py)

REF_H = 720.0
# внутрішні коефіцієнти (підібрані на тестовому наборі; для малих кадрів окремі)
TUNE = dict(gain=2.6, gain_s=2.6, curve=2.4, curve_s=2.0, rel=0.45, abs_w=0.5, lc=0.25, lc_s=0.2,
            sigma=1.0, edge_w=3.0, passes=4, ink=14.0)
# Масштаб візерунка понад 720p. Колишнє лінійне масштабування (крок штрихів, довжина штриха, зерно, крапки, контур - усе в k = h/720 разів)
# тримало розмір візерунка ВІДНОСНО кадру сталим, тож додаткові пікселі висоти йшли в той самий грубий малюнок (а зерно на ≥1008p
# узагалі ставало 2x2 - грубішим, ніж на 720p). Тепер розмір візерунка росте повільніше: k**pow (pow < 1), зерно - один піксель.
# Заміри (Ультра, 14 скріншотів 1080p, 11 - 1440p): помилка дрібного тону (σ 1.5 px) -36%/-45%, збіг країв 0.9964 -> 0.9981 / 0.9844 -> 0.9947,
# кореляція деталей у смугах 1.2-4.8 px вища; OCR тексту не гірший. На k <= 1 (до 720p включно) - без змін. Бітрейт H.264 «Високою» +19%.
PAT = dict(pow=0.5, ink=0.5, grain=1)


def pat_k(k: float, e: float | None = None) -> float:
    """масштаб розмірів візерунка для висоти кадру з масштабом k = h/720: до 720p - k, вище - k**pow"""
    e = PAT["pow"] if e is None else e
    return k ** e if k > 1.0 else k


def grain_cell(h: int) -> int:
    """комірка зерна (px): PAT['grain'] - завжди (1 - окремі пікселі), 0 - колишнє правило (2 від 1008p)"""
    g = int(PAT["grain"])
    return g if g > 0 else (1 if h < 1008 else 2)


ND = 12                         # кількість напрямів штриха
_S = np.sin(np.arange(ND) * np.pi / ND).astype(np.float32)
_C = np.cos(np.arange(ND) * np.pi / ND).astype(np.float32)
_S2 = np.sin(np.arange(ND) * np.pi / ND + np.deg2rad(60)).astype(np.float32)
_C2 = np.cos(np.arange(ND) * np.pi / ND + np.deg2rad(60)).astype(np.float32)


@dataclass
class Params:
    """Внутрішні константи (в інтерфейсі не показуються)."""
    period: float = 4.0        # крок штрихів, px на висоті 720
    wobble: float = 1.0        # «тремтіння руки» штрихів
    auto: bool = True          # автоконтраст + авто-CLAHE
    invert: bool = False
    method: int = 4            # 1 ескіз (поріг), 2 контур (адаптивний), 3 півтон (растр), 4 малюнок (основний), 5 гравюра (раніше «Ультра», для відео - з пам'яттю кадрів);
                               # 6 нейро - ОКРЕМА ГІЛКА, не наступник 5: навчена мережа, що наближає вигляд «Гравюри» на відеокарті (benvid/student)
    glyph: bool = False        # «Символи»: той самий тон і рівні 1-5, але кадр складається знаками шрифту (див. glyph.py)
    gdens: int = 3             # «Символи»: щільність знаків: 1 (клітинка 32×64 px) .. 3 (8×16 px, типово)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> "Params":
        p = Params()
        for f in fields(Params):
            if d and f.name in d:
                try:
                    setattr(p, f.name, type(getattr(p, f.name))(d[f.name]))
                except (TypeError, ValueError):
                    pass
        if p.glyph and p.method > MAIN_LEVELS:          # IRNM1 - гілка лише для чорнила; у символах діють п'ять рівнів
            p.method = MAIN_LEVELS
        return p


# ------------------------------------------------------------------ ресурси
@lru_cache(maxsize=1)
def _blue() -> np.ndarray:
    import os
    return np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bluenoise128.npy")).astype(np.float32)


@lru_cache(maxsize=1)
def _blobs() -> np.ndarray:
    """поріг для нерегулярних плямок: ранг відстані до найближчого «насіння» (насіння = найнижчі ранги синього шуму)"""
    bn = _blue()
    n = bn.shape[0]
    seeds = (bn < 0.045).astype(np.uint8)
    dist = cv2.distanceTransform(np.tile(1 - seeds, (3, 3)), cv2.DIST_L2, 5)[n:2 * n, n:2 * n]
    dist = dist + np.random.default_rng(5).random((n, n)).astype(np.float32) * 0.3
    return (np.argsort(np.argsort(dist.ravel())).reshape(n, n).astype(np.float32) / (n * n - 1))


def _tile_hash(ty, tx):
    """хеш номера плитки 128×128 -> випадкові зсув/поворот цієї плитки (ціле, без знаку), працює і з numpy-масивами"""
    h = (ty * 374761393 + tx * 668265263 + 12345) & 0x7FFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0x7FFFFFFF
    return h ^ (h >> 16)


_TILE_CACHE: dict = {}
_TILE_LOCK = threading.Lock()


def _tile(t: np.ndarray, w: int, h: int, cell: int) -> np.ndarray:
    """кеш обгортка: плитка залежить лише від (шаблон, розмір, комірка), тож на відео рахується один раз, а не на кожен кадр
    (було ≈20 мс на кадр 720p і десятки МБ тимчасових масивів - вони ще й гальмують паралельні процеси)"""
    key = (id(t), t.shape, w, h, cell)
    with _TILE_LOCK:
        hit = _TILE_CACHE.get(key)
    if hit is not None and hit[0] is t:
        return hit[1]
    r = _tile_build(t, w, h, cell)
    r.flags.writeable = False
    with _TILE_LOCK:                                    # потоки господаря рахують кадри разом
        while len(_TILE_CACHE) >= 4:
            _TILE_CACHE.pop(next(iter(_TILE_CACHE)))
        _TILE_CACHE[key] = (t, r)
    return r


def _tile_build(t: np.ndarray, w: int, h: int, cell: int) -> np.ndarray:
    """поріг-плитка 128×128 (безшовна) на весь кадр. Кожна плитка в кадрі отримує власний випадковий зсув і
    дзеркало/транспонування, тож візерунок ніколи не повторюється точно (інакше на великих кадрах видно квадратну сітку)"""
    yy = (np.arange(h) // cell)[:, None]
    xx = (np.arange(w) // cell)[None, :]
    hs = _tile_hash(yy >> 7, xx >> 7).astype(np.int64)
    iy = (yy + (hs & 127)) & 127
    ix = (xx + ((hs >> 7) & 127)) & 127
    sw = ((hs >> 14) & 1) == 1
    iy, ix = np.where(sw, ix, iy), np.where(sw, iy, ix)
    ix = np.where(((hs >> 15) & 1) == 1, 127 - ix, ix)
    return np.ascontiguousarray(t[iy, ix])


@lru_cache(maxsize=6)
def _grid(w: int, h: int):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx, yy


@lru_cache(maxsize=6)
def _wobble(w: int, h: int, k: float) -> np.ndarray:
    rng = np.random.default_rng(2)
    sw, sh = max(2, w // 4), max(2, h // 4)
    a = cv2.GaussianBlur(rng.standard_normal((sh, sw)).astype(np.float32), (0, 0), max(0.8, 1.5 * k))
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    return a * (1.0 * k / (a.std() + 1e-6))


@lru_cache(maxsize=4)
def _field(w: int, h: int, sig: float, seed: int) -> np.ndarray:
    """плавне випадкове поле 0..1 (рівномірно розподілене) розміром кадру"""
    sw, sh = max(4, w // 8), max(4, h // 8)
    a = cv2.GaussianBlur(np.random.default_rng(seed).standard_normal((sh, sw)).astype(np.float32),
                         (0, 0), max(0.8, sig / 8))
    r = np.argsort(np.argsort(a.ravel())).reshape(sh, sw).astype(np.float32) / (sh * sw - 1)
    return cv2.resize(r, (w, h), interpolation=cv2.INTER_CUBIC)


@lru_cache(maxsize=1)
def _clahe():
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _local_mean(gf: np.ndarray, w: int, h: int, sigmas) -> np.ndarray:
    """середнє по околах кількох масштабів; рахуємо на зменшеній копії - швидко й гладко"""
    s = 8 if min(w, h) >= 160 else 4 if min(w, h) >= 80 else 1
    small = cv2.resize(gf, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
    acc = 0
    for sg in sigmas:
        acc = acc + cv2.GaussianBlur(small, (0, 0), max(0.5, sg / s), borderType=cv2.BORDER_REFLECT)
    return cv2.resize(acc / len(sigmas), (w, h), interpolation=cv2.INTER_LINEAR)


def _gray(img: np.ndarray, w: int, h: int, auto: bool) -> np.ndarray:
    g = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2LAB)[:, :, 0] if img.ndim == 3 else img
    shrink = g.shape[1] >= w and g.shape[0] >= h
    g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    if auto:
        lo, hi = np.percentile(g[::4, ::4], [1, 99])
        if hi - lo >= 1:
            lut = np.clip((np.arange(256) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
            g = cv2.LUT(g, lut)
        if hi - lo < 110:          # слабкий контраст -> м'яко вирівнюємо локально
            g = _clahe().apply(g)
    return g


def _sm_py(x, a, b):
    z = np.clip((x - a) / (b - a), 0.0, 1.0)
    return z * z * (3 - 2 * z)


def _sm(x, a, b):
    """гладкий перехід smoothstep: 0 при x ≤ a, 1 при x ≥ b. Для float32-масивів - одним проходом ядра (раніше шість проміжних масивів)"""
    if _nb is not None and isinstance(x, np.ndarray) and x.dtype == np.float32 and x.ndim == 2 and x.flags.c_contiguous:
        out = np.empty_like(x)
        _sm_nb(x, float(a), float(b), out)
        return out
    return _sm_py(x, a, b)


# ------------------------------------------------------------------ текст
def _gb(x: np.ndarray, sigma: float) -> np.ndarray:
    """гаусове розмиття; великі sigma - на зменшеній копії (у десятки разів швидше)"""
    f = int(sigma // 3)
    if f < 2:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    hh, ww = x.shape
    sm = cv2.resize(x, (max(2, ww // f), max(2, hh // f)), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), sigma / f)
    return cv2.resize(sm, (ww, hh), interpolation=cv2.INTER_LINEAR)


def _level_w(ts: np.ndarray, lab: np.ndarray, size: np.ndarray, k: float, s: int, stats=None) -> np.ndarray:
    """Вага «рівності» 1..0 на кожну велику рівну ділянку (компоненту lab): на великому околі (~45 px на 720p) тон усередині
    ділянки майже не змінюється (панель, стіна, титри) -> 1; у плавному градієнті (віньєтка на шкірі, небо) він повзе -> 0.
    Заливати суцільним чорним/білим градієнт не можна: зникають півтони й виходять різкі межі «зон». Міряємо в межах
    ОДНІЄЇ ділянки (тонка лінія поміж двома рівними заливками - не градієнт); літери й краї не псують вимір,
    шум і східці згладжені. Малі ділянки (<1% кадру) - завжди 1."""
    lv = np.ones(ts.shape, np.float32)
    area = ts.shape[0] * ts.shape[1]
    big = [c for c in np.argsort(-size)[:9] if c > 0 and size[c] >= 0.01 * area][:8]
    if not big:
        return lv
    sg = max(1.5, 6.0 * k / s)
    R = max(3, int(round(20 * k / s)))
    kb = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * R + 1, 2 * R + 1))
    sm_r = max(1.0, R / 3.0)
    hh, ww = ts.shape
    mg = int(math.ceil(3.0 * sg + R + 3.0 * sm_r)) + 3          # найдальший вплив на піксель ділянки: гаус sg, потім dilate/erode R, потім гаус sm_r
    for c in big:
        if stats is not None:                                    # рахуємо лише в охоплюючому прямокутнику ділянки з запасом: решта кадру на неї не впливає
            bx, by, bw, bh = (int(v) for v in stats[c][:4])
            ya, yb, xa, xb = max(0, by - mg), min(hh, by + bh + mg), max(0, bx - mg), min(ww, bx + bw + mg)
        else:
            ya, yb, xa, xb = 0, hh, 0, ww
        m = (lab[ya:yb, xa:xb] == c)
        tsc = ts[ya:yb, xa:xb]
        mf = m.astype(np.float32)
        den = cv2.GaussianBlur(mf, (0, 0), sg)
        tl = cv2.GaussianBlur(tsc * mf, (0, 0), sg) / np.maximum(den, 1e-3)
        valid = m & (den > 0.5)
        hi = np.where(valid, tl, -9.0).astype(np.float32)
        lo = np.where(valid, tl, 9.0).astype(np.float32)
        rb = np.maximum(cv2.dilate(hi, kb) - cv2.erode(lo, kb), 0.0)
        w = 1.0 - _sm(cv2.GaussianBlur(rb, (0, 0), sm_r), DEBAND["lv0"], DEBAND["lv1"])
        lv[ya:yb, xa:xb][m] = w[m]
    return lv


BIG_H = 1600             # від цієї висоти (4K) вмикаються економні за пам'яттю шляхи; нижче - усе як було, до біта


def _text_snap(t: np.ndarray, w: int, h: int, k: float, level: bool = False) -> np.ndarray:
    """Текст, титри, інтерфейс: де в околі лише два рівні (фон і штрих) з великим контрастом - тон стає чорним/білим
    (чисті літери замість розсипу точок), і це розтікається на рівний фон навколо, щоб не було плям.
    Фотографії (плавні півтони) майже не зачіпає.
    level: (кадри від DEBAND['h0'], рівні 4-5) цілу рівну ділянку заливаємо лише там, де вона справді рівна, а не плавний
    градієнт: після прибирання східців кодека градієнти стали б «рівними» за порогом розмаху й зливалися б в одну заливку."""
    s = 1 if min(w, h) < 200 else 2 if min(w, h) < 500 else 3
    ts = cv2.resize(t, (max(8, w // s), max(8, h // s)), interpolation=cv2.INTER_AREA) if s > 1 else t
    hh, ww = ts.shape
    r = max(2, int(round(5 * k / s)))
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    mn, mx = cv2.erode(ts, ker), cv2.dilate(ts, ker)
    rng = mx - mn
    near = ((ts < mn + 0.22 * rng) | (ts > mx - 0.22 * rng)).astype(np.float32)
    f = cv2.blur(near, (2 * r + 1, 2 * r + 1))
    rt = max(2, int(round(4 * k / s)))                      # штрих тонший за ~2rt: верхній/нижній «циліндр» (top-hat)
    kt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rt + 1, 2 * rt + 1))
    thin = np.maximum(ts - cv2.morphologyEx(ts, cv2.MORPH_OPEN, kt), cv2.morphologyEx(ts, cv2.MORPH_CLOSE, kt) - ts)
    wgt = _sm(rng, 0.45, 0.7) * _sm(f, 0.85, 0.97) * _sm(cv2.dilate(thin, ker), 0.25, 0.45)   # великі форми - не текст
    if wgt.max() < 0.05:                                    # тексту немає - нічого не чіпаємо (звичайний шлях)
        return t
    sg = max(2.0, 4.0 * r)
    num_m, num_h = 0.5 * (mn + mx) * wgt, 0.5 * rng * wgt
    den = _gb(wgt, sg) + 1e-4
    mid = _gb(num_m, sg) / den
    hr = _gb(num_h, sg) / den
    sg2 = sg * 6                                            # далеко від тексту беремо найближчі значення з широкого вікна
    near_ok = den > 0.02
    if level:
        # вікно гауса обрізане на ~3 sg2: далі нього (~1/3 кадру від тексту) чисельник і знаменник нульові, «середина» тексту
        # виходила 0, і будь-яка рівна сіра ділянка там білішала. Тягнемо до середніх по всьому тексту.
        tw = max(float(wgt.sum()), 1e-6)
        mg, hg = float(num_m.sum()) / tw, float(num_h.sum()) / tw
        den2 = _gb(wgt, sg2) + 1e-4
        mid = np.where(near_ok, mid, (_gb(num_m, sg2) + 1e-4 * mg) / den2)
        hr = np.where(near_ok, hr, (_gb(num_h, sg2) + 1e-4 * hg) / den2)
    else:
        den2 = _gb(wgt, sg2) + 1e-6
        mid = np.where(near_ok, mid, _gb(num_m, sg2) / den2)
        hr = np.where(near_ok, hr, _gb(num_h, sg2) / den2)
    rngd = cv2.dilate(rng, ker)
    flat = 1.0 - _sm(rngd, 0.06, 0.22)                      # розтікаємо лише на рівний фон, не на деталі сцени
    spread = np.clip(_gb(cv2.dilate(wgt, ker), sg * 1.2) * 4.0, 0, 1) * flat
    # цілі рівні ділянки, що торкаються тексту, обробляємо повністю - без плям навколо літер
    fm = (rngd < 0.2).astype(np.uint8)
    nlab, lab, cstats, _ = cv2.connectedComponentsWithStats(fm, connectivity=4)
    if nlab > 1:
        hit = np.bincount(lab.ravel(), weights=((spread > 0.5) & (fm > 0)).ravel().astype(np.float64), minlength=nlab)
        size = np.bincount(lab.ravel(), minlength=nlab)
        ok = (hit > 0) & (size > 0)
        ok[0] = False
        okf = ok[lab].astype(np.float32)
        if level:
            okf = okf * _level_w(ts, lab, size, k, s, cstats)
        region = cv2.dilate(okf, ker)                       # разом із краями навколо літер
        clear = _sm(np.abs(ts - mid), 0.1, 0.2)             # ділянка явно темна чи світла відносно тексту
        spread = np.maximum(spread, region * clear)
    # заливка фону навколо літер - лише там, де фон і так майже чорний чи білий. Середньо-сірий фон (синя заставка, сіра панель) заливати
    # не можна: від того в рівному полі виростали чорні «зони» й квадрати довкола титрів (тон поля ~0.36 стрибав у 0), а півтон губився
    bgw = np.clip(1.0 - cv2.dilate(wgt, ker), 0.0, 1.0)
    sgb = sg * 2.0
    nbg = _gb(bgw, sgb)
    tb = np.where(nbg > 0.05, _gb(ts * bgw, sgb) / np.maximum(nbg, 1e-4), 0.0)       # тон фону поза літерами (де фону немає - як було)
    spread = np.maximum(spread * _sm(np.abs(tb - 0.5), 0.22, 0.36), wgt)
    if s > 1 and h >= BIG_H and _nb is not None:
        # 4K: розтягування трьох полів на весь кадр і ~10 проміжних масивів по 33 МБ - пам'ять, а не обчислення (×100 від 720p);
        # одним проходом: вибірка з малих полів + змішування (та сама білінійна інтерполяція, що в cv2.resize)
        return _snap_blend_nb(np.ascontiguousarray(t, dtype=np.float32), np.ascontiguousarray(mid, dtype=np.float32),
                              np.ascontiguousarray(hr, dtype=np.float32), np.ascontiguousarray(spread, dtype=np.float32))
    if s > 1:
        mid, hr, spread = (cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in (mid, hr, spread))
    if _nb is not None and t.dtype == np.float32 and t.shape == mid.shape == hr.shape == spread.shape:
        # той самий ланцюжок рівно в float32, але одним проходом: раніше 12 проміжних масивів на весь кадр (на 1080p-2K - майже половина часу
        # _text_snap, і це пам'ять, а не обчислення); результат побітово той самий (див. test_snap_tail_is_bit_exact)
        return _snap_tail_nb(np.ascontiguousarray(t), np.ascontiguousarray(mid, dtype=np.float32), np.ascontiguousarray(hr, dtype=np.float32),
                             np.ascontiguousarray(spread, dtype=np.float32))
    d = np.maximum(0.45 * hr, 0.03)
    z = np.clip((t - (mid - d)) / (2 * d), 0.0, 1.0)
    snapped = z * z * (3 - 2 * z)
    return (t + spread * (snapped - t)).astype(np.float32)


# ------------------------------------------------------------------ тон
PCA_CAP = 1.3          # проєкція кольорів не розтягується більше ніж у стільки разів (див. _gray_pca)


_PCA_PROBES = cv2.cvtColor(np.array([[[40, 40, 220], [60, 200, 60], [220, 60, 10], [30, 220, 230], [140, 170, 220],
                                       [128, 128, 128], [40, 30, 30], [250, 250, 250]]], np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)


def _pca_smooth(tm, cut: bool, pa: tuple) -> tuple:
    """параметри проєкції знебарвлення кадру (vL, va, vb, off, scale, Lo, mix) -> згладжені в часі. Проєкція p = (c·v - off)·scale + Lo -
    афінна за кольором c, тож усереднюємо її КОЕФІЦІЄНТИ: тоді сірий КОЖНОГО кольору плавно йде від старого значення до нового (без
    перегонів осі, зсуву й розтягу порізно, які давали сплески), а швидкість обмежена - не більше PCA_STEP рівнів сірого за кадр
    для типових кольорів. Склейка (або перший кадр) - беремо кадр як є"""
    vL, va, vb, off, scale, Lo, mix = pa
    new = np.array([vL * scale, va * scale, vb * scale, Lo - off * scale])
    S = tm.pca
    if S is None or cut:
        tm.pca = new
        return pa
    d = _PCA_PROBES @ (new[:3] - S[:3]) + (new[3] - S[3])
    k = TNORM["a"]
    big = float(np.abs(d).max()) * mix                                       # найбільший зсув сірого за кадр, якби взяли нове одразу
    if big * k > TNORM["pstep"]:
        k = TNORM["pstep"] / big
    S = S + k * (new - S)
    tm.pca = S
    return (float(S[0]), float(S[1]), float(S[2]), 0.0, 1.0, float(S[3]), float(mix))


def _gray_pca(img: np.ndarray, mix: float = 0.5, cap: float | None = None, tm=None, cut: bool = True) -> np.ndarray:
    """розумне «знебарвлення»: суміш Lab-світлоти й проєкції на напрям найбільшого розкиду кольорів кадру
    (щоб блакитний текст на червоному не зливався з фоном).
    Проєкцію нормують так, щоб її розкид (1-99 перцентилі) збігався з розкидом світлоти. У майже рівному кадрі з рідким
    текстом (титри, заставка) обидва розкиди майже нульові, їх частка - випадкова, і шум кодека на тлі роздувається в
    рази. Напрям найбільшого розкиду не може бути «бідніший» за напрям світлоти, тож законне розтягнення більше за cap
    (трохи більше 1) неможливе - cap його й обрізає.
    tm (ToneMem відео) й cut: у відео проєкція згладжується між кадрами (_pca_smooth; склейка скидає) - інакше щоразу, коли в кадр
    заходить титр чи інший колір, напрям, зсув і розтяг проєкції стрибають, і кольорове тло «дихає» по сірому.
    Без tm (фото, окремі кадри) - як було."""
    if img.ndim == 2:
        return img
    lab = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    st = max(1, min(6, min(lab.shape[:2]) // 8))
    raw = lab[::st, ::st].reshape(-1, 3).astype(np.float32)
    mu = raw.mean(0)
    sub = raw - mu
    sub[:, 1:] *= 0.8
    cov = np.cov(sub.T) + np.eye(3) * 1e-6
    if _nb is not None:
        _, evec = np.linalg.eigh(cov)
        v = evec[:, -1].copy()
        if v[0] < 0:
            v = -v
        v[1:] *= 0.8
        lo, hi = np.percentile(raw @ v - float(mu @ v), [1, 99])          # перцентилі за вибіркою, не за повним кадром
        Lo, Hi = np.percentile(L[::8, ::8], [1, 99])
        scale = max(Hi - Lo, 1.0) / max(hi - lo, 1e-3)
        if cap is not None:
            scale = min(scale, cap)
        pa = (float(v[0]), float(v[1]), float(v[2]), float(float(mu @ v) + lo), float(scale), float(Lo), float(mix))
        if tm is not None:
            pa = _pca_smooth(tm, cut, pa)
        r = _gpu.run("pca", lambda g: g.pca(lab, *pa))
        return r if r is not None else _pca_nb(lab, *pa).astype(np.float32)
    _, evec = np.linalg.eigh(cov)
    v = evec[:, -1].copy()
    if v[0] < 0:
        v = -v
    v[1:] *= 0.8
    p = cv2.transform(lab.astype(np.float32), v.reshape(1, 3).astype(np.float32)).reshape(lab.shape[:2]) - float(mu @ v)
    lo, hi = np.percentile(p[::8, ::8], [1, 99])
    Lo, Hi = np.percentile(L[::8, ::8], [1, 99])
    scale = max(Hi - Lo, 1.0) / max(hi - lo, 1e-3)
    if cap is not None:
        scale = min(scale, cap)
    p = (p - lo) * scale + Lo
    return np.clip((1 - mix) * L + mix * p, 0, 255).astype(np.uint8)


def _tone_py(L, base, loc, sg, prm):
    lo, rngb, gam, gain, rel, abs_w, lc, curve = prm
    detail = L - base
    b = np.clip((base - lo) / rngb, 0, 1) ** gam
    t_abs = b + gain * detail / rngb * (0.6 + 1.6 * np.sqrt(b * (1 - b)))
    t = (1 - rel) * t_abs + rel * (0.5 + ((L - loc) + abs_w * (L - 128.0)) / 110.0)
    t += lc * np.clip(detail / np.maximum(sg * 2.0, 7.0), -1, 1)
    x = np.clip(t, 0, 1) * 2 - 1
    t = (0.5 + 0.5 * np.tanh(curve * x) / np.tanh(curve)).astype(np.float32)
    t[t >= 0.995] = 1.01          # майже біле/чорне - суцільне, без поодиноких крапок
    t[t <= 0.005] = -0.01
    return t


if _nb is not None:
    @_nb.njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _tone_nb(L, base, loc, sg, prm):
        h, w = L.shape
        out = np.empty((h, w), np.float32)
        lo, rngb, gam, gain, rel, abs_w, lc, curve = prm[0], prm[1], prm[2], prm[3], prm[4], prm[5], prm[6], prm[7]
        tc = np.tanh(curve)
        NB = 2048
        bl = np.empty(NB + 1, np.float32)          # таблиці замість pow/sqrt/tanh на кожен піксель
        wl = np.empty(NB + 1, np.float32)
        for i in range(NB + 1):
            b0 = i / NB
            bl[i] = b0 ** gam
            wl[i] = 0.6 + 1.6 * np.sqrt(bl[i] * (1.0 - bl[i]))
        NT = 4096
        th = np.empty(NT + 1, np.float32)
        for i in range(NT + 1):
            th[i] = 0.5 + 0.5 * np.tanh(curve * (i / NT * 2.0 - 1.0)) / tc
        for y in _nb.prange(h):
            for x in range(w):
                l = L[y, x]
                bs = base[y, x]
                d = l - bs
                ib = int(min(max((bs - lo) / rngb, 0.0), 1.0) * NB + 0.5)
                b = bl[ib]
                t_abs = b + gain * d / rngb * wl[ib]
                t = (1.0 - rel) * t_abs + rel * (0.5 + ((l - loc[y, x]) + abs_w * (l - 128.0)) / 110.0)
                t += lc * min(max(d / max(sg[y, x] * 2.0, 7.0), -1.0), 1.0)
                v = min(max(t, 0.0), 1.0)
                r = th[int(v * NT + 0.5)]
                if r >= 0.995:
                    r = 1.01
                elif r <= 0.005:
                    r = -0.01
                out[y, x] = r
        return out

    @_nb.njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _pca_nb(lab, vL, va, vb, off, scale, Lo, mix):
        h, w = lab.shape[:2]
        out = np.empty((h, w), np.float32)
        for y in _nb.prange(h):
            for x in range(w):
                l = np.float32(lab[y, x, 0])
                p = (l * vL + lab[y, x, 1] * va + lab[y, x, 2] * vb - off) * scale + Lo
                r = (1.0 - mix) * l + mix * p
                out[y, x] = min(max(r, 0.0), 255.0)
        return out
    _tone_kernel = _tone_nb
else:
    _tone_kernel = _tone_py


# ------------------------------------------------------------------ блоки кодека й смуги
# Кодек (H.264 та ін.) на темних плавних ділянках лишає «східці»: плями-плато по ~16 px, що різняться на 1-6 рівнів; те саме
# дають смуги 8-бітних градієнтів. Тонмап підсилює різницю, і градієнт стає мозаїкою прямокутників різної густини крапок.
# Гладимо ЛИШЕ околи, де переважає «плато» (кодек стер текстуру: енергія дрібних деталей ~0): справжня текстура, шум, текст і краї
# (там плато немає) не чіпаються. Середнє T беремо лише з плато (тож біля країв немає ореолів), східці вище t1 вважаємо краєм
# предмета й лишаємо. Зміна яскравості обмежена ~t1/2 (кілька рівнів із 255). Кадри нижче h0 (144p-480p) не чіпаємо взагалі.
DEBAND = dict(h0=540,            # висота кадру результату, з якої діє
              rad=0.035,         # радіус гладження - частка висоти (~25 px на 720p, трохи більше за макроблок)
              e0=0.25, e1=0.7,   # енергія дрібних деталей: нижче e0 - плато, вище e1 - справжня текстура
              f0=0.3, f1=0.6,    # яка частка плато має бути в околі, щоб там гладити
              t0=1.5, t1=4.5,    # східець до t0 згладжується повністю, понад t1 - край, не чіпаємо
              lv0=0.08, lv1=0.18)  # _level_w: розмах тону рівного тла на ~45 px: до lv0 - рівне, понад lv1 - плавний градієнт


def _pl_py(L, hp2, e0, e1, G, LG):
    G[...] = 1.0 - _sm(np.sqrt(np.maximum(hp2, 0.0)), e0, e1)
    LG[...] = L * G


def _fin_py(L, T, F, t0, t1, out):
    d = T - L
    out[...] = L + F * (1.0 - _sm(np.abs(d), t0, t1)) * d


if _nb is not None:
    @_nb.njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _pl_nb(L, hp2, e0, e1, G, LG):
        """вага «плато» 1..0 за енергією дрібних деталей (hp2 - її квадрат) і L, помножена на неї"""
        h, w = L.shape
        inv = 1.0 / (e1 - e0)
        for y in _nb.prange(h):
            for x in range(w):
                e = np.sqrt(max(hp2[y, x], 0.0))          # розмиття дає -1e-15 замість 0: корінь від'ємного був би NaN
                z = min(max((e - e0) * inv, 0.0), 1.0)
                g = 1.0 - z * z * (3.0 - 2.0 * z)
                G[y, x] = g
                LG[y, x] = L[y, x] * g

    @_nb.njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _fin_nb(L, T, F, t0, t1, out):
        """L += F * (T - L) там, де |T - L| малий (м'яко до t1): східці гладяться, краї лишаються"""
        h, w = L.shape
        inv = 1.0 / (t1 - t0)
        for y in _nb.prange(h):
            for x in range(w):
                l = L[y, x]
                d = T[y, x] - l
                z = min(max((abs(d) - t0) * inv, 0.0), 1.0)
                out[y, x] = l + F[y, x] * (1.0 - z * z * (3.0 - 2.0 * z)) * d
    _pl, _fin = _pl_nb, _fin_nb
else:
    _pl, _fin = _pl_py, _fin_py


def deband_active(h: int) -> bool:
    return h >= DEBAND["h0"]


def _deband(L: np.ndarray, h: int) -> np.ndarray:
    """L float32 (H, W), 0..255 -> L без «східців» кодека й смуг у плавних ділянках (див. коментар вище)"""
    d = DEBAND
    H, W = L.shape
    hp = L - cv2.GaussianBlur(L, (0, 0), 1.2)
    hp2 = cv2.blur(hp * hp, (7, 7))
    G = np.empty_like(L)
    LG = np.empty_like(L)
    r = _gpu.run("pl", lambda g: g.pl(L, hp2, d["e0"], d["e1"]))
    if r is not None:
        G, LG = r
    else:
        _pl(L, hp2, d["e0"], d["e1"], G, LG)
    sig = float(np.clip(d["rad"] * h, 12.0, 48.0)) / 2.0
    f = int(np.clip(round(sig / 1.6), 1, 10))
    sz = (max(2, W // f), max(2, H // f))
    Gd = cv2.resize(G, sz, interpolation=cv2.INTER_AREA)
    if float(Gd.max()) <= d["f0"]:                        # плато ніде немає (текстура, шум) - нічого гладити
        return L
    LGd = cv2.resize(LG, sz, interpolation=cv2.INTER_AREA)
    Ld = cv2.resize(L, sz, interpolation=cv2.INTER_AREA)
    sd = max(0.8, sig / f)
    Gb = cv2.GaussianBlur(Gd, (0, 0), sd, borderType=cv2.BORDER_REFLECT)
    LGb = cv2.GaussianBlur(LGd, (0, 0), sd, borderType=cv2.BORDER_REFLECT)
    Tg = cv2.GaussianBlur(Ld, (0, 0), sd, borderType=cv2.BORDER_REFLECT)
    Tl = np.where(Gb > 0.03, LGb / np.maximum(Gb, 1e-3), Tg).astype(np.float32)
    Fl = _sm(Gb, d["f0"], d["f1"]).astype(np.float32)
    T = cv2.resize(Tl, (W, H), interpolation=cv2.INTER_LINEAR)
    F = cv2.resize(Fl, (W, H), interpolation=cv2.INTER_LINEAR)
    out = np.empty_like(L)
    r = _gpu.run("fin", lambda g: g.fin(L, T, F, d["t0"], d["t1"]))
    if r is not None:
        return r
    _fin(L, T, F, d["t0"], d["t1"], out)
    return out


def gray_stage(img: np.ndarray, w: int, h: int, auto: bool, deband: bool = False, tm=None, cut: bool = True) -> np.ndarray:
    """перший (дешевий) етап тонмапа: зміна розміру й знебарвлення -> світлота float32 (h, w), 0..255.
    deband: прибрати «східці» кодека й смуги (лише кадри від DEBAND['h0']). Від тієї ж висоти знебарвлення не розтягує
    проєкцію кольорів понад PCA_CAP (шум у рівних кадрах із титрами); нижче - усе, як було.
    tm, cut: часова пам'ять відео (ToneMem) і ознака склейки - параметри знебарвлення згладжуються між кадрами (див. _gray_pca)."""
    shrink = img.shape[1] >= w and img.shape[0] >= h
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)   # спершу розмір, потім усе інше
    big = deband_active(h)
    L = _gray_pca(img, 0.5 if auto else 0.0, PCA_CAP if big else None, tm if auto else None, cut).astype(np.float32)
    return _deband(L, h) if (deband and big) else L


# ------------------------------------------------------------------ часова пам'ять нормування (відео, рівень 5)
# Автоконтраст кадру (діапазон lo..hi і гама) міряється на кожному кадрі окремо, тож у відео він «пульсує»: у кадр заходить світлий
# предмет чи титр - hi стрибає, і тло одразу темнішає; спалах чи згасання - нормування їх стирає (титр лишається повністю білим, поки
# не зникне різко). Пам'ять згладжує дрібні коливання й обмежує швидкість руху меж діапазону за кадр: повільні зміни (дрейф)
# нормування наздоганяє, а швидкі (згасання, спалах, поява титру) показуються як є - плавно, без стрибка. Монтажна склейка
# скидає пам'ять: нова сцена нормується з нуля. Фото й будь-які кадри без пам'яті не змінюються.
TNORM = dict(a=0.3,                 # частка шляху до нового значення за кадр (згладжує коливання менші за швидкісну межу)
             q=0.04,                # найбільший крок кожного краю діапазону за кадр - частка поточного діапазону (≈ ×2 за 17 кадрів)
             qg=0.03,               # те саме для гами (в логарифмі)
             dz=0.04,               # «мертва зона» відставання (частка діапазону), менше якої тьмяність тексту не враховується
             win=((4, 0.12, 0.60), (12, 0.20, 0.45), (24, 0.28, 0.35)),   # вікна пошуку плавної зміни краю: (кадрів, найменша зміна - частка
                                                                          # діапазону, найбільша частка одного кроку в ній: різкий стрибок не рахується)
             eps=0.012,             # крок проти напрямку менший за цю частку діапазону - шум, плавності не псує
             rev=0.06, quiet=0.02, ncalm=2, slack=0.02,   # відпускаємо: зміна пішла назад на rev діапазону за 4 кадри / край стоїть (< quiet) ncalm кадрів поспіль
             tw=160,                # ширина мініатюри для ознак склейки
             pstep=10.0,            # найбільший зсув сірого типових кольорів за кадр через знебарвлення (PCA), рівнів: без «стрибків» кольорового тла
             brk=0.30,              # «точка зриву»: зміст кадру змінився так, що середина тонів (25-й, 50-й, 75-й процентилі) зсунулась за ОДИН кадр більше за цю частку діапазону - не титр
                                    # і не згасання, а інша картинка: пам'ять нормування скидається (як при склейці), контраст «перескакує»
             brkmin=48.0,           # ... діапазон для цієї частки не вужчий за стільки рівнів світлоти (у темному кадрі шум не дає зриву)
             far=0.30, nfar=4, still=0.08,   # другий запобіжник: ціль далі від стану за far діапазону, її справжні межі останні nfar кадрів стоять (розкид менший за still)
             bref=0.5, ema=0.25,            # ... і «тіло» кадру відійшло від згладженого тіла (за ним стоїть нормування) на частку bref від порога brk: титр тіло не зсуває - не зрив
             cutm=18.0, cutq=60.0, cutf=0.30, cutc=0.30, cutc2=0.50)   # ознаки склейки: «сильна різниця» (рівні, частка різких) і кореляція: нижче cutc - склейка завжди, нижче cutc2 - за сильної різниці
PRIME = 24                          # скільки кадрів ПЕРЕД розігрівом шматка відео лише декодуються й дають мініатюрну статистику


def _gray_thumb(img: np.ndarray, w: int) -> np.ndarray:
    h = max(8, int(round(img.shape[0] * w / max(1, img.shape[1]))))
    t = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if t.ndim == 3:
        t = cv2.cvtColor(t[:, :, :3], cv2.COLOR_BGR2GRAY)
    return t.astype(np.float32)


def _norm_raw(bs_: np.ndarray, true: bool = False, tm=None):
    """сирі параметри нормування кадру за вибіркою згладженої бази: (lo, hi, гама). true: ще й справжні 0.5-й і 99.5-й
    процентилі (у майже рівному кадрі lo, hi - це 0..255, а справжні показують, яким тьмяним той кадр є насправді).
    tm: пам'ять нормування - їй віддається ще й «тіло» кадру (25-й, 50-й, 75-й процентилі) для точки зриву"""
    lo, hi, q1, q2, q3 = np.percentile(bs_, [0.5, 99.5, 25, 50, 75])
    if tm is not None:
        tm.qnow = (float(q1), float(q2), float(q3))
    lo0, hi0 = lo, hi
    if hi - lo < 24:                      # майже рівний кадр: тримаємось абсолютної яскравості
        lo, hi = 0.0, 255.0
    rngb = float(hi - lo)
    bs = np.clip((bs_ - lo) / rngb, 0, 1)
    med = min(max(float(np.median(bs)), 0.08), 0.92)
    gam = float(np.clip(np.log(0.5) / np.log(med), 0.5, 2.0)) ** 0.6
    return (lo, hi, gam, lo0, hi0) if true else (lo, hi, gam)


class ToneMem:
    """Часова пам'ять нормування тону одного відео (одна на послідовність кадрів, лежить в InkState)."""
    __slots__ = ("lo", "hi", "gam", "thumb", "raw", "hist", "hold", "dire", "calm", "fresh", "fade", "pca", "qnow", "qprev", "qref")

    def __init__(self):
        self.lo = self.hi = self.gam = None
        self.thumb = None
        self.pca = None                 # згладжена статистика знебарвлення (_gray_pca)
        self.fade = (0.0, 1.0)          # (floor, peak) останнього кадру: у які тони потрапили його найтемніше й найсвітліше (0, 1 - звичайний кадр)
        self.raw = []                   # справжні межі (lo_t, hi_t) останніх кадрів
        self.hist = []                  # стан (lo, hi) після кроку на останніх кадрах (до попереднього включно)
        self.hold = [None, None]        # затримане значення краю (lo, hi) під час плавної зміни (згасання, поява), інакше None
        self.dire = [0, 0]              # напрям тієї зміни
        self.calm = [0, 0]              # скільки кадрів поспіль край уже стоїть
        self.fresh = [0, 0]             # скільки кадрів минуло з початку обліку зміни краю (склейка, кінець попередньої зміни)
        self.qnow = self.qprev = None   # «тіло» поточного й минулого кадру: (25-й, 50-й, 75-й процентилі бази)
        self.qref = None                # згладжене «тіло» (за яким стоїть нинішнє нормування): титр його не зсуває, інша картинка - зсуває

    def _reset(self, lo, hi, gam, lo_t, hi_t):
        self.qprev = self.qref = self.qnow
        self.lo, self.hi, self.gam = float(lo), float(hi), float(gam)
        self.raw = [(float(lo_t), float(hi_t))]
        self.hist = [(self.lo, self.hi)]
        self.hold, self.dire, self.calm, self.fresh = [None, None], [0, 0], [0, 0], [0, 0]

    def _trend(self, j: int, rng: float):
        """плавна одностайна зміна краю j за останні кадри (до поточного включно): (напрям ±1, ширина вікна) або (0, 0).
        Повторювані кадри (джерело з меншою частотою кадрів) не заважають: шукається чиста зміна за вікно, а не за кадр"""
        d, r = TNORM, self.raw
        n = min(len(r), self.fresh[j] + 1)                                   # лише кадри, що належать нинішній зміні
        for W, th, share in d["win"]:
            if n <= W:
                break
            seg = [x[j] for x in r[len(r) - 1 - W:]]
            D = seg[-1] - seg[0]
            if abs(D) < th * rng:
                continue
            sg = 1.0 if D > 0 else -1.0
            st = [(seg[k + 1] - seg[k]) * sg for k in range(W)]
            if min(st) < -d["eps"] * rng or max(st) > share * abs(D):
                continue
            return int(sg), W
        return 0, 0

    def observe(self, img: np.ndarray) -> bool:
        """кадр -> змінилась сцена? За мініатюрою 160 px: склейка - це коли ЗАГАЛЬНИЙ вигляд кадру не схожий на попередній (кореляція
        низька) й вони ще й сильно різняться. Новий титр, зміна тексту, глітч, спалах при тій самій сцені - не склейка (пам'ять
        нормування їх пережовує плавно, а не скидається - інакше тло смикається щоразу). Перший кадр й інший розмір - склейка."""
        d = TNORM
        g = _gray_thumb(img, d["tw"])
        P = self.thumb
        self.thumb = g
        if P is None or P.shape != g.shape:
            return True
        ad = np.abs(g - P)
        big = bool(ad.mean() > d["cutm"] or (ad > d["cutq"]).mean() > d["cutf"])
        a0, b0 = g - g.mean(), P - P.mean()
        sa, sb = float(a0.std()), float(b0.std())
        if sa <= 3.0 or sb <= 3.0:                                         # рівний кадр (чорний, заставка): кореляції немає - судимо за різницею
            return big
        corr = float((a0 * b0).mean() / (sa * sb))
        return bool(corr < d["cutc"] or (big and corr < d["cutc2"]))

    def break_point(self, lo, hi, rng) -> bool:
        """Точка зриву. Пам'ять нормування обмежує швидкість меж діапазону (так титр, глітч, спалах не смикають тло), але коли змінилась уся
        картинка (не склейка за мініатюрою: схожа розкладка, інша яскравість), нормування мало б перестрибнути, а воно повзло десятки кадрів
        і нова картинка стояла пересвіченою чи тьмяною. Зрив - у двох випадках: (1) «тіло» кадру (25-й, 50-й, 75-й процентилі, їх титр не рухає)
        за один кадр зсунулось більше за brk діапазону; (2) ціль далеко від стану (> far діапазону), а її справжні межі останні nfar кадрів стоять -
        це вже не флуктуація. Плавні згасання (край іде одним боком, hold) й повільний дрейф сюди не потрапляють"""
        d = TNORM
        if self.qnow is not None and self.qprev is not None:
            ref = max(rng, d["brkmin"])
            if max(abs(a - b) for a, b in zip(self.qnow, self.qprev)) > d["brk"] * ref:
                return True
        n = d["nfar"]
        if self.hold[0] is None and self.hold[1] is None and len(self.raw) >= n and self.qnow is not None and self.qref is not None:
            lag = max(abs(float(lo) - self.lo), abs(float(hi) - self.hi))
            shift = max(abs(a - b) for a, b in zip(self.qnow, self.qref))
            if lag > d["far"] * rng and shift > d["bref"] * d["brk"] * max(rng, d["brkmin"]):
                w_ = self.raw[-n:]                                                 # справжні межі за останні n кадрів: стоять (нова картинка) чи йдуть (поява, згасання)
                span = max(max(x[j] for x in w_) - min(x[j] for x in w_) for j in (0, 1)) / max(w_[-1][1] - w_[-1][0], 24.0)
                if span < d["still"]:
                    return True
        return False

    def step(self, lo, hi, gam, cut: bool, lo_t=None, hi_t=None):
        """сирі параметри кадру -> параметри, якими його слід нормувати (lo_t, hi_t - справжні межі кадру, без «рівного кадру»).
        Дрібні відхилення згладжуються, швидкий рух меж обмежений (див. TNORM); плавна зміна яскравості (згасання, поява: край
        ідеться одним боком багато кадрів) ТРИМАЄ нормування там, де воно було до неї, - тож зміну видно, а не стерто
        автоконтрастом. Тримається, поки кадр лишається всередині затриманого діапазону (інакше світле просто «засвітилось» би).
        Склейка скидає пам'ять. Якщо сирі значення збіглися зі збереженими - повертаються самі (біт-у-біт)"""
        lo_t = lo if lo_t is None else lo_t
        hi_t = hi if hi_t is None else hi_t
        if cut or self.lo is None:
            self._reset(lo, hi, gam, lo_t, hi_t)
            return lo, hi, gam
        d = TNORM
        rng = max(self.hi - self.lo, 24.0)
        self.raw.append((float(lo_t), float(hi_t)))
        del self.raw[:-(d["win"][-1][0] + 1)]
        if self.break_point(lo, hi, rng):                                         # інша картинка, а не титр чи згасання: контраст перестрибує
            self._reset(lo, hi, gam, lo_t, hi_t)
            return lo, hi, gam
        self.qprev = self.qnow
        if self.qnow is not None:
            self.qref = self.qnow if self.qref is None else tuple(r + d["ema"] * (q - r) for r, q in zip(self.qref, self.qnow))
        for j in (0, 1):
            self.fresh[j] += 1
            h_ = self.hold[j]
            if h_ is None:
                sg, W = self._trend(j, rng)
                if sg:
                    cand = self.hist[max(0, len(self.hist) - 1 - W)][j]       # стан до початку зміни...
                    r0 = self.raw[len(self.raw) - 1 - W][j]                   # ...але не вужчий за сам кадр на її початку (стан міг іще наздоганяти)
                    cand = max(cand, r0) if j else min(cand, r0)
                    inside = cand >= hi_t - d["slack"] * rng if j else cand <= lo_t + d["slack"] * rng
                    if inside:
                        self.hold[j], self.dire[j], self.calm[j] = cand, sg, 0
            else:
                d4 = self.raw[-1][j] - self.raw[-1 - min(4, self.fresh[j], len(self.raw) - 1)][j]
                out = h_ < hi_t - d["slack"] * rng if j else h_ > lo_t + d["slack"] * rng
                if out or d4 * self.dire[j] < -d["rev"] * rng:
                    self.hold[j] = None                                       # кадр вийшов за затримане / зміна пішла назад
                else:
                    self.calm[j] = self.calm[j] + 1 if abs(d4) < d["quiet"] * rng else 0
                    if self.calm[j] >= d["ncalm"]:
                        self.hold[j] = None
                if self.hold[j] is None:
                    self.fresh[j] = 0                                         # стару зміну вдруге не шукаємо
        a, lim = d["a"], d["q"] * rng
        cur, tgt = (self.lo, self.hi), (float(lo), float(hi))
        new = [self.hold[j] if self.hold[j] is not None else cur[j] + min(max(a * (tgt[j] - cur[j]), -lim), lim) for j in (0, 1)]
        lg = math.log(self.gam)
        ng = math.exp(lg + min(max(a * (math.log(float(gam)) - lg), -d["qg"]), d["qg"]))
        if new[1] - new[0] < 8.0:                                         # (не буває: краї йдуть до діапазону не вужчого за 24) про всяк випадок
            self._reset(lo, hi, gam, lo_t, hi_t)
            return lo, hi, gam
        self.lo, self.hi, self.gam = new[0], new[1], ng
        self.hist.append((new[0], new[1]))
        del self.hist[:-(d["win"][-1][0] + 1)]
        if new[0] == tgt[0] and new[1] == tgt[1] and ng == float(gam):
            return lo, hi, gam
        return new[0], new[1], ng


def tone_prime(img: np.ndarray, w: int, h: int, auto: bool, tm: ToneMem) -> None:
    """прогрів пам'яті нормування кадром, що лише декодований (малювати його не треба): ті самі сирі параметри, що рахує tone_map на
    кадрі w x h, - без деталей, ядра тону й усього наступного. Дешевше за повний кадр на порядок"""
    cut = tm.observe(img)
    if auto:
        L = gray_stage(img, w, h, True, True, tm, cut)
        lo, hi, gam, lo_t, hi_t = _norm_raw(_tone_base(L, w, h)[::3, ::3], True, tm)
        tm.step(lo, hi, gam, cut, lo_t, hi_t)


def tone_warm(L: np.ndarray, w: int, h: int, auto: bool, tm: ToneMem, cut: bool) -> None:
    """прогрів пам'яті нормування кадром, для якого вже є світлота L (ink5.warm; cut - з tm.observe): ті самі сирі параметри, що в tone_map"""
    if auto:
        lo, hi, gam, lo_t, hi_t = _norm_raw(_tone_base(L, w, h)[::3, ::3], True, tm)
        tm.step(lo, hi, gam, cut, lo_t, hi_t)


def _tone_base(L: np.ndarray, w: int, h: int) -> np.ndarray:
    """база тону: світлота, двічі згладжена з збереженням країв (на зменшеній копії), на повний розмір"""
    k = h / REF_H
    s = 1 if min(w, h) < 240 else 2 if min(w, h) < 480 else 4
    Ls = cv2.resize(L, (max(2, w // s), max(2, h // s)), interpolation=cv2.INTER_AREA) if s > 1 else L
    sp = max(1.5, 5 * k / s)
    base = cv2.bilateralFilter(Ls, -1, 22, sp)
    if min(w, h) >= 60:
        base = cv2.bilateralFilter(base, -1, 22, sp)
    if s > 1:
        base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
    return base


def tone_map(img: np.ndarray, w: int, h: int, auto: bool, lfilter=None, deband: bool = False, tmem: ToneMem | None = None):
    """-> (карта тону 0..1, світлота). База (згладжена з збереженням країв) стискається й вирівнюється за
    яскравістю кадру, деталі додаються поверх із підсиленням - так видно і світле, і темне.
    Потім локальне нормування й S-подібна крива для виразності.
    lfilter (необов'язково): функція світлота -> світлота, що викликається одразу після знебарвлення (часове
    усереднення відео: підсилення деталей нижче сильно роздуває шум, тож гасити його треба до нього).
    deband: див. gray_stage (рівні 4 і 5).
    tmem (необов'язково): часова пам'ять нормування відео (ToneMem) - див. TNORM; без неї кадр нормується лише за собою."""
    k = h / REF_H
    cut = tmem.observe(img) if (tmem is not None and auto) else True
    L = gray_stage(img, w, h, auto, deband, tmem, cut)
    if lfilter is not None:
        L = lfilter(L)
    small = h < 300
    gain = TUNE['gain_s'] if small else TUNE['gain']
    curve = TUNE['curve_s'] if small else TUNE['curve']
    base = _tone_base(L, w, h)
    peak, floor = 1.0, 0.0
    if auto:
        lo, hi, gam, lo_t, hi_t = _norm_raw(base[::3, ::3], True, tmem)
        if tmem is not None:
            lo_r, hi_r = lo, hi
            lo, hi, gam = tmem.step(lo, hi, gam, cut, lo_t, hi_t)
            if lo_r is not lo or hi_r is not hi:
                # згасання/поява: пам'ять тримає діапазон ширшим за кадр - найсвітліше й найтемніше в кадрі мають тьмяніти разом із ним
                # (їх яскравість відносно нормування): інакше локальний контраст (суцільні ділянки біля насичення) знову розтягнув би їх
                # до чорного/білого. Нижче тон усього кадру стискується в [floor, peak]
                r_, dz = max(float(hi - lo), 1.0), TNORM["dz"]
                peak = float(min(1.0, max(0.0, (float(hi_t) - float(lo)) / r_) / (1.0 - dz)) ** float(gam))
                floor = float((max(0.0, min(1.0, (float(lo_t) - float(lo)) / r_) - dz) / (1.0 - dz)) ** float(gam))
            tmem.fade = (floor, peak)
        rngb = float(hi - lo)
    else:
        lo, rngb, gam = 0.0, 255.0, 1.0
    loc = _local_mean(L, w, h, (30 * k, 107 * k))
    detail = L - base
    sg = np.sqrt(_blur_ds(detail * detail, max(1.0, 8 * k), 4 if h >= 360 else 2))
    prm = np.array([lo, rngb, gam, gain, TUNE['rel'], TUNE['abs_w'], TUNE['lc_s'] if small else TUNE['lc'], curve],
                   np.float64)
    t = _gpu.run("tone", lambda g: g.tone(L, base, loc, sg, prm))
    if t is None:
        t = _tone_kernel(L, base, loc, sg, prm)
    t = _text_snap(t, w, h, k, bool(deband) and deband_active(h))
    if peak < 1.0 or floor > 0.0:
        t = (np.float32(floor) + np.float32(max(peak - floor, 0.0)) * t).astype(np.float32)       # згасання/поява: зображення те саме, лише тьмяніше
    return t, L


# ------------------------------------------------------------------ зони
_AREA_FLOOR = 1280.0 * 720.0 / (1920.0 * 1080.0)     # поправка на розмір кадру зростає лише до 1080p; на 4K зони «дрібніші» не стають
                                                      # (інакше вага крапок падала в 5 разів, а зерна ставало в рази більше)


def _zone_params(cnt, jxx, jyy, jxy, tsum, sxs, sys_, sxx, syy, sxy, sw, sh, w, h, allow_strokes):
    """статистики зон -> ваги стилів [крапки, штрихи, зерно] і напрям (cos/sin подвійного кута) на зону"""
    cnt = np.maximum(cnt, 1)
    tone = tsum / cnt
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    ang_g = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    mx, my = sxs / cnt, sys_ / cnt
    cxx, cyy, cxy = sxx / cnt - mx * mx, syy / cnt - my * my, sxy / cnt - mx * my
    tr, det = cxx + cyy, np.sqrt(np.maximum((cxx - cyy) ** 2 + 4 * cxy ** 2, 0))
    l1, l2 = (tr + det) / 2, np.maximum((tr - det) / 2, 1e-3)
    elong = np.sqrt(l1 / l2)
    ang_ax = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    area = cnt / (sw * sh)
    # --- м'які ваги стилів
    small = 1 - _sm(area * 1e3 / max(1280.0 * 720.0 / (w * h), _AREA_FLOOR), 4, 40)
    tmid = 1 - np.abs(tone - 0.5) * 2
    w_str = (_sm(coh, 0.3, 0.7) + 0.6 * _sm(elong, 1.8, 4.0)) * _sm(tmid, 0.1, 0.5) * (1 - 0.6 * small)
    w_str = np.clip(w_str, 0, 1)
    w_dot = np.clip(small * 0.9 + _sm(tone, 0.6, 0.9) * 0.5 * (1 - w_str), 0, 1)
    w_grn = np.clip(1 - w_str - w_dot, 0, 1) + 0.05
    ws = np.stack([w_dot, w_str, w_grn])
    if not allow_strokes:
        ws[1] = 0
    ws /= ws.sum(0, keepdims=True)
    use_ax = _sm(elong, 1.8, 3.5)
    cx = use_ax * np.cos(2 * ang_ax) + (1 - use_ax) * np.cos(2 * (ang_g + np.pi / 2))
    sx = use_ax * np.sin(2 * ang_ax) + (1 - use_ax) * np.sin(2 * (ang_g + np.pi / 2))
    amp = np.maximum(coh, use_ax)
    return ws, cx * amp, sx * amp


def _zones(t, gf, w, h):
    """зони за рівнем тону (зв'язні компоненти) на зменшеній копії -> (zone, ts, gx, gy, n, sw, sh, s, k)"""
    k = h / REF_H
    s = 4 if min(w, h) >= 200 else 2 if min(w, h) >= 64 else 1
    sw, sh = max(4, w // s), max(4, h // s)
    ts = cv2.GaussianBlur(cv2.resize(t, (sw, sh), interpolation=cv2.INTER_AREA), (0, 0), max(0.8, 6.0 * k / s))
    L = 6
    lab = np.minimum((ts * L).astype(np.int32), L - 1)
    zone = np.zeros((sh, sw), np.int32)
    n = 0
    for l in range(L):
        m = (lab == l).astype(np.uint8)
        if not m.any():
            continue
        c, cl = cv2.connectedComponents(m, connectivity=4)
        zone[cl > 0] = cl[cl > 0] + n
        n += c - 1
    n += 1
    g = cv2.GaussianBlur(cv2.resize(gf, (sw, sh), interpolation=cv2.INTER_AREA), (0, 0), max(0.8, 2 * k / s + 0.3))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return zone, ts, gx, gy, n, sw, sh, s, k


if _nb is not None:
    @_nb.njit(cache=True, nogil=True)
    def _zone_stats_nb(zone, ts, gx, gy, n):
        st = np.zeros((10, n))
        sh, sw = zone.shape
        for y in range(sh):
            for x in range(sw):
                z = zone[y, x]
                a, b = gx[y, x], gy[y, x]
                st[0, z] += 1.0
                st[1, z] += a * a
                st[2, z] += b * b
                st[3, z] += a * b
                st[4, z] += ts[y, x]
                st[5, z] += x
                st[6, z] += y
                st[7, z] += x * x
                st[8, z] += y * y
                st[9, z] += x * y
        return st


def analyze(t: np.ndarray, gf: np.ndarray, w: int, h: int, allow_strokes: bool):
    """-> (ваги стилів [крапки, штрихи, зерно], індекс напряму штриха) на повному розмірі (запасний варіант без numba)"""
    zone, ts, gx, gy, n, sw, sh, s, k = _zones(t, gf, w, h)
    flat = zone.ravel()

    def zsum(a):
        return np.bincount(flat, np.asarray(a).ravel().astype(np.float64), minlength=n)

    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float64)
    ws, cx, sx = _zone_params(zsum(np.ones((sh, sw))), zsum(gx * gx), zsum(gy * gy), zsum(gx * gy), zsum(ts), zsum(xx),
                              zsum(yy), zsum(xx * xx), zsum(yy * yy), zsum(xx * yy), sw, sh, w, h, allow_strokes)
    sl = max(0.8, max(1.0, 10 * k) / s)                          # згладжування робимо на малій копії

    def up(v, sig):
        v = cv2.GaussianBlur(v[zone].astype(np.float32), (0, 0), sig)
        return cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR) if v.shape != (h, w) else v

    W = [up(ws[i], sl) for i in range(3)]
    qi = None
    if allow_strokes:
        C2, S2 = up(cx, sl * 0.7), up(sx, sl * 0.7)
        fb = np.clip(1 - np.hypot(C2, S2) * 3, 0, 1)          # де напряму майже немає - діагональ 45°
        C2 = C2 * (1 - fb)
        S2 = S2 * (1 - fb) + fb
        qi = np.round(0.5 * np.arctan2(S2, C2) * (ND / np.pi)).astype(np.int32) % ND
    return W, qi


def analyze_small(t, gf, w, h):
    """те саме, що analyze, але поля лишаються на малому розмірі: Pk (sh, sw, 4) = [вага крапок, вага штрихів, cos, sin
    подвійного кута напряму] (ваги вже нормовано; вага зерна = 1 - перші дві); розгортання робить ядро"""
    zone, ts, gx, gy, n, sw, sh, s, k = _zones(t, gf, w, h)
    st = _zone_stats_nb(zone, ts, gx, gy, n)
    ws, cx, sx = _zone_params(*st, sw, sh, w, h, True)
    sl = max(0.8, max(1.0, 10 * k) / s)
    W = [cv2.GaussianBlur(ws[i][zone].astype(np.float32), (0, 0), sl) for i in range(3)]
    tot = W[0] + W[1] + W[2] + 1e-6
    C = cv2.GaussianBlur(cx[zone].astype(np.float32), (0, 0), sl * 0.7)
    S = cv2.GaussianBlur(sx[zone].astype(np.float32), (0, 0), sl * 0.7)
    fb = np.clip(1 - np.hypot(C, S) * 3, 0, 1)             # де напряму майже немає - діагональ 45°
    return np.ascontiguousarray(np.stack([W[0] / tot, W[1] / tot, C * (1 - fb), S * (1 - fb) + fb], axis=-1),
                                dtype=np.float32)


# ------------------------------------------------------------------ дрібні кадри: пряме двійкове пошукування
def _dbs_py(b, WE, S2, hk, passes, Wp):          # запасний варіант без numba: не робимо нічого (лишається впорядкований поріг)
    return b


if _nb is not None:
    @_nb.njit(cache=True, nogil=True)
    def _dbs_nb(b, WE, S2, hk, passes, Wp):
        """WE = Wt*E з полями r (без перевірок меж у циклі), S2 = згортка Wt з hk^2 (готова); при зміні пікселя
        оновлюється лише WE"""
        h, w = b.shape
        k = hk.shape[0]
        for _ in range(passes):
            changed = 0
            for y in range(h):
                for xi in range(w):
                    x = xi if y % 2 == 0 else w - 1 - xi
                    d = 1.0 - 2.0 * b[y, x]
                    s = 0.0
                    for i in range(k):
                        for j in range(k):
                            s += hk[i, j] * WE[y + i, x + j]
                    if 2.0 * d * s + S2[y, x] < -1e-7:
                        b[y, x] = 1.0 - b[y, x]
                        changed += 1
                        for i in range(k):
                            for j in range(k):
                                WE[y + i, x + j] += d * hk[i, j] * Wp[y + i, x + j]
            if changed == 0:
                break
        return b
    _dbs_kernel = _dbs_nb
else:
    _dbs_kernel = _dbs_py


@lru_cache(maxsize=8)
def _gauss7(sigma: float = 0.85):
    ax = np.arange(-3, 4)
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2)).astype(np.float32)
    return g / g.sum()


def _small(t: np.ndarray, w: int, h: int, init: np.ndarray | None = None, passes: int | None = None) -> np.ndarray:
    """дрібні кадри: починаємо з порогу по синьому шуму й покращуємо, перемикаючи окремі пікселі, поки згладжена
    маска не збіжиться з тоном (краї важать більше, тому контури й текст чіткі).
    init (h, w) {0,1} - початкова маска замість порогу (відео: попередній кадр, тож незмінне лишається незмінним)"""
    b0 = (t > _tile(_blue(), w, h, 1)).astype(np.float32) if init is None else init.astype(np.float32)
    if _nb is None:
        return b0.astype(np.uint8)
    tt = np.clip(t, 0, 1).astype(np.float32)
    g = _gauss7(round(float(TUNE['sigma']), 3))
    E = (cv2.filter2D(b0, -1, g, borderType=cv2.BORDER_REFLECT) -
         cv2.filter2D(tt, -1, g, borderType=cv2.BORDER_REFLECT)).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(tt, cv2.CV_32F, 1, 0), cv2.Sobel(tt, cv2.CV_32F, 0, 1))
    Wt = (1.0 + TUNE['edge_w'] * np.clip(mag * 0.8, 0, 1)).astype(np.float32)
    r = g.shape[0] // 2
    Wp = np.zeros((h + 2 * r, w + 2 * r), np.float64)
    Wp[r:r + h, r:r + w] = Wt
    WE = Wp.copy()
    WE[r:r + h, r:r + w] *= E
    S2 = cv2.filter2D(Wt, -1, g * g, borderType=cv2.BORDER_CONSTANT).astype(np.float64)
    return _dbs_kernel(b0, WE, S2, g.astype(np.float64), int(TUNE['passes'] if passes is None else passes), Wp).astype(np.uint8)


# ------------------------------------------------------------------ стилі
@lru_cache(maxsize=1)
def _ppf_lut():
    x = np.linspace(0.0, 1.0, 1025)
    p = np.clip(x, 1e-4, 1 - 1e-4)
    q = np.where(p < 0.5, p, 1 - p)
    tt = np.sqrt(-2 * np.log(q))
    v = tt - ((0.010328 * tt + 0.802853) * tt + 2.515517) / (((0.001308 * tt + 0.189269) * tt + 1.432788) * tt + 1)
    return np.where(p < 0.5, -v, v).astype(np.float32)


def _ppf(p):
    """обернена функція нормального розподілу (таблиця)"""
    return _ppf_lut().take((np.asarray(p, np.float32) * 1024.0 + 0.5).astype(np.int32))


def _smooth_noise(w, h, sig, seed, amp):
    sw, sh = max(2, w // 4), max(2, h // 4)
    a = cv2.GaussianBlur(np.random.default_rng(seed).standard_normal((sh, sw)).astype(np.float32),
                         (0, 0), max(0.8, sig / 4))
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    return a * (amp / (a.std() + 1e-6))


@lru_cache(maxsize=4)
def _hand(w: int, h: int, k: float, p: float, wobble: float):
    """«рука»: лінії пливуть, товщина й розриви штриха міняються, кут блукає"""
    return (_smooth_noise(w, h, 12 * k, 2, 0.55 * p * wobble), _smooth_noise(w, h, 6 * k, 3, 0.12),
            np.round(_smooth_noise(w, h, 14 * k, 5, 1.2)).astype(np.int32))


def _strokes_thr(qi, w, h, period, wobble):
    k = h / REF_H
    p = max(2.0, period * k)
    xx, yy = _grid(w, h)
    wob, jit, ang = _hand(w, h, k, p, wobble)
    q = (qi + ang) % ND
    c, sn = _C[q], _S[q]
    u0 = yy * c - xx * sn
    v0 = yy * sn + xx * c
    u = u0 + wob
    u *= 1.0 / p
    u -= np.floor(u)
    thr1 = np.abs(2 * u - 1) + jit
    u2 = 0.5 * u0 - 0.8660254 * v0                # друга сітка під 60°
    u2 += wob * 0.7 + 2
    u2 *= 1.0 / p
    u2 -= np.floor(u2)
    thr2 = np.abs(2 * u2 - 1) + jit * 0.7
    return thr1, thr2


def _mix_py(t, Wd, Ws, Wg, qi, wob, jit, ang, cg, cb, pp):
    h, w = t.shape
    thr_g = _tile(_blue(), w, h, cg)
    thr_d = _tile(_blobs(), w, h, cb)
    xx, yy = _grid(w, h)
    q = (qi + ang) % ND
    c, sn = _C[q], _S[q]
    u0 = yy * c - xx * sn
    v0 = yy * sn + xx * c
    u = u0 + wob
    u *= 1.0 / pp
    u -= np.floor(u)
    thr1 = np.abs(2 * u - 1) + jit
    u2 = 0.5 * u0 - 0.8660254 * v0
    u2 += wob * 0.7 + 2
    u2 *= 1.0 / pp
    u2 -= np.floor(u2)
    thr2 = np.abs(2 * u2 - 1) + jit * 0.7
    wd = np.clip((0.4 - t) * 5.0, 0, 1)                                # у глибоких тінях - перехресні штрихи
    ta = t + (np.sqrt(np.maximum(t, 0.0)) - t) * wd
    thr = Wd * thr_d + Ws * np.maximum(thr1, thr2 * wd) + Wg * thr_g   # стилі змішуються плавно, не за швами
    te = t + Ws * (ta - t)
    pur = Wd * Wd + Ws * Ws + Wg * Wg                                  # змішаний поріг стиснутий -> вирівнюємо тон
    z = np.clip((pur - 0.6) / 0.4, 0, 1)
    z = z * z * (3 - 2 * z)
    clean = np.clip(te, 0, 1)
    tc = z * te + (1 - z) * (0.5 + 0.2887 * np.sqrt(pur) * _ppf(clean))
    tc[t >= 1.0] = 2.0
    tc[t <= 0.0] = -1.0
    return tc.astype(np.float32), thr.astype(np.float32), clean.astype(np.float32)


if _nb is not None:
    @_nb.njit(cache=True, nogil=True, inline="always")
    def _tl(tab, y, x, c):
        """вибірка з плитки 128×128 з випадковим зсувом/відбиттям для кожної плитки (див. _tile)"""
        yy = y // c
        xx = x // c
        h = (np.int64(yy >> 7) * 374761393 + np.int64(xx >> 7) * 668265263 + 12345) & 0x7FFFFFFF
        h = ((h ^ (h >> 13)) * 1274126177) & 0x7FFFFFFF
        h = h ^ (h >> 16)
        iy = (yy + (h & 127)) & 127
        ix = (xx + ((h >> 7) & 127)) & 127
        if (h >> 14) & 1:
            iy, ix = ix, iy
        if (h >> 15) & 1:
            ix = 127 - ix
        return tab[iy, ix]


if _nb is not None:
    @_nb.njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _mix_nb(t, Wd, Ws, Wg, qi, wob, jit, ang, cg, cb, pp, blue, blobs, Cn, Sn, ppf):
        h, w = t.shape
        tc = np.empty((h, w), np.float32)
        thr = np.empty((h, w), np.float32)
        clean = np.empty((h, w), np.float32)
        inv = 1.0 / pp
        for y in _nb.prange(h):
            for x in range(w):
                tt = t[y, x]
                a_d, a_s, a_g = Wd[y, x], Ws[y, x], Wg[y, x]
                q = (qi[y, x] + ang[y, x]) % 12
                c = Cn[q]
                sn = Sn[q]
                u0 = y * c - x * sn
                v0 = y * sn + x * c
                u = (u0 + wob[y, x]) * inv
                u -= np.floor(u)
                thr1 = abs(2.0 * u - 1.0) + jit[y, x]
                u2 = (0.5 * u0 - 0.8660254 * v0 + wob[y, x] * 0.7 + 2.0) * inv
                u2 -= np.floor(u2)
                thr2 = abs(2.0 * u2 - 1.0) + jit[y, x] * 0.7
                wd = min(max((0.4 - tt) * 5.0, 0.0), 1.0)
                ta = tt + (np.sqrt(max(tt, 0.0)) - tt) * wd
                tg = _tl(blue, y, x, cg)
                td = _tl(blobs, y, x, cb)
                th = a_d * td + a_s * max(thr1, thr2 * wd) + a_g * tg
                te = tt + a_s * (ta - tt)
                pur = a_d * a_d + a_s * a_s + a_g * a_g
                z = min(max((pur - 0.6) / 0.4, 0.0), 1.0)
                z = z * z * (3.0 - 2.0 * z)
                cl = min(max(te, 0.0), 1.0)
                r = z * te + (1.0 - z) * (0.5 + 0.2887 * np.sqrt(pur) * ppf[int(cl * 1024.0 + 0.5)])
                if tt >= 1.0:
                    r = 2.0
                elif tt <= 0.0:
                    r = -1.0
                tc[y, x] = r
                thr[y, x] = th
                clean[y, x] = cl
        return tc, thr, clean

    def _mix(t, Wd, Ws, Wg, qi, wob, jit, ang, cg, cb, pp):
        return _mix_nb(t, Wd, Ws, Wg, qi, wob, jit, ang, cg, cb, pp, _blue(), _blobs(), _C, _S, _ppf_lut())
else:
    _mix = _mix_py


if _nb is not None:
    @_nb.njit(cache=True, nogil=True, parallel=True)
    def _sm_nb(x, a, b, out):
        """суворо в float32, без fastmath і FMA: ті самі біти, що ланцюжок numpy у _sm_py"""
        h, w = x.shape
        a32 = np.float32(a)
        d32 = np.float32(b - a)
        zero = np.float32(0.0)
        one = np.float32(1.0)
        for y in _nb.prange(h):
            for i in range(w):
                z = min(max((x[y, i] - a32) / d32, zero), one)
                out[y, i] = z * z * (np.float32(3.0) - np.float32(2.0) * z)


if _nb is not None:
    @_nb.njit(cache=True, nogil=True, inline="always")
    def _bil(a, fy, fx):
        """білінійна вибірка з малого поля (півпіксельні координати, як у cv2.resize INTER_LINEAR)"""
        sh, sw = a.shape
        fy = min(max(fy, 0.0), sh - 1.0)
        fx = min(max(fx, 0.0), sw - 1.0)
        y0, x0 = int(fy), int(fx)
        y1, x1 = min(y0 + 1, sh - 1), min(x0 + 1, sw - 1)
        dy, dx = fy - y0, fx - x0
        return (a[y0, x0] * (1 - dx) + a[y0, x1] * dx) * (1 - dy) + (a[y1, x0] * (1 - dx) + a[y1, x1] * dx) * dy

    @_nb.njit(cache=True, nogil=True, parallel=True)
    def _snap_tail_nb(t, mid, hr, sp):
        """кінець _text_snap, поелементно й суворо в float32 (без fastmath, без FMA): d = max(0.45·hr, 0.03); z = clip((t - (mid - d)) / (2d), 0, 1);
        out = t + sp·(z²(3 - 2z) - t). Порядок і типи дій - як у numpy-варіанті, тож біти ті самі"""
        h, w = t.shape
        out = np.empty((h, w), np.float32)
        c045, c003 = np.float32(0.45), np.float32(0.03)
        zero, one, two, three = np.float32(0.0), np.float32(1.0), np.float32(2.0), np.float32(3.0)
        for y in _nb.prange(h):
            for x in range(w):
                tv = t[y, x]
                d = c045 * hr[y, x]
                if not (d >= c003):                    # np.maximum: NaN проходить, як і в numpy
                    d = c003 if d == d else d
                z = (tv - (mid[y, x] - d)) / (two * d)
                if z < zero:
                    z = zero
                elif z > one:
                    z = one
                out[y, x] = tv + sp[y, x] * (z * z * (three - two * z) - tv)
        return out

    @_nb.njit(cache=True, nogil=True, parallel=True)
    def _snap_blend_nb(t, mid, hr, sp):
        """кінець _text_snap для 4K: t + spread·(snapped − t), де mid/hr/spread - малі поля, що білінійно розтягуються на кадр"""
        h, w = t.shape
        sh, sw = mid.shape
        ry, rx = sh / h, sw / w
        out = np.empty((h, w), np.float32)
        for y in _nb.prange(h):
            fy = (y + 0.5) * ry - 0.5
            for x in range(w):
                fx = (x + 0.5) * rx - 0.5
                m = _bil(mid, fy, fx)
                d = max(0.45 * _bil(hr, fy, fx), 0.03)
                tv = t[y, x]
                z = min(max((tv - (m - d)) / (2.0 * d), 0.0), 1.0)
                out[y, x] = tv + _bil(sp, fy, fx) * (z * z * (3.0 - 2.0 * z) - tv)
        return out

    @_nb.njit(cache=True, nogil=True, inline="always")
    def _fatan2(y, x):
        ax, ay = abs(x), abs(y)
        mx = max(ax, ay)
        if mx < 1e-12:
            return 0.0
        a = min(ax, ay) / mx
        s = a * a
        r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a
        if ay > ax:
            r = 1.5707963 - r
        if x < 0:
            r = 3.1415927 - r
        return -r if y < 0 else r

    @_nb.njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _mix2_nb(t, Pk, wob, jit, ang, cg, cb, pp, blue, blobs, Cn, Sn, ppf):
        """змішування стилів. Pk (sh,sw,4) - малі поля [вага крапок, вага штрихів, cos, sin подвійного кута напряму],
        розгортаються білінійно прямо тут. -> tc, thr, dm, де dm = (маска першої проходки) - (чистий тон)"""
        h, w = t.shape
        sh, sw = Pk.shape[0], Pk.shape[1]
        tc = np.empty((h, w), np.float32)
        thr = np.empty((h, w), np.float32)
        dm = np.empty((h, w), np.float32)
        inv = 1.0 / pp
        ry, rx = sh / h, sw / w
        x0 = np.empty(w, np.int32)
        x1 = np.empty(w, np.int32)
        dxs = np.empty(w, np.float32)
        for x in range(w):
            fx = min(max((x + 0.5) * rx - 0.5, 0.0), sw - 1.0)
            i0 = int(fx)
            x0[x] = i0
            x1[x] = min(i0 + 1, sw - 1)
            dxs[x] = fx - i0
        for y in _nb.prange(h):
            fy = min(max((y + 0.5) * ry - 0.5, 0.0), sh - 1.0)
            y0 = int(fy)
            y1 = min(y0 + 1, sh - 1)
            dy = fy - y0
            for x in range(w):
                dx = dxs[x]
                i0, i1 = x0[x], x1[x]
                w00, w01, w10, w11 = (1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy
                a_d = Pk[y0, i0, 0] * w00 + Pk[y0, i1, 0] * w01 + Pk[y1, i0, 0] * w10 + Pk[y1, i1, 0] * w11
                a_s = Pk[y0, i0, 1] * w00 + Pk[y0, i1, 1] * w01 + Pk[y1, i0, 1] * w10 + Pk[y1, i1, 1] * w11
                c2 = Pk[y0, i0, 2] * w00 + Pk[y0, i1, 2] * w01 + Pk[y1, i0, 2] * w10 + Pk[y1, i1, 2] * w11
                s2 = Pk[y0, i0, 3] * w00 + Pk[y0, i1, 3] * w01 + Pk[y1, i0, 3] * w10 + Pk[y1, i1, 3] * w11
                a_g = 1.0 - a_d - a_s
                tt = t[y, x]
                qd = int(_fatan2(s2, c2) * 1.9098593 + 12.5)            # 6/pi; +12 - щоб значення були додатні
                q = qd + ang[y, x]
                q = q % 12
                c = Cn[q]
                sn = Sn[q]
                u0 = y * c - x * sn
                v0 = y * sn + x * c
                wb = wob[y, x]
                jt = jit[y, x]
                u = (u0 + wb) * inv
                u -= int(u + 1000.0) - 1000
                thr1 = abs(2.0 * u - 1.0) + jt
                u2 = (0.5 * u0 - 0.8660254 * v0 + wb * 0.7 + 2.0) * inv
                u2 -= int(u2 + 1000.0) - 1000
                thr2 = abs(2.0 * u2 - 1.0) + jt * 0.7
                wd = min(max((0.4 - tt) * 5.0, 0.0), 1.0)
                ta = tt + (np.sqrt(max(tt, 0.0)) - tt) * wd
                tg = _tl(blue, y, x, cg)
                td = _tl(blobs, y, x, cb)
                th = a_d * td + a_s * max(thr1, thr2 * wd) + a_g * tg
                te = tt + a_s * (ta - tt)
                pur = a_d * a_d + a_s * a_s + a_g * a_g
                z = min(max((pur - 0.6) / 0.4, 0.0), 1.0)
                z = z * z * (3.0 - 2.0 * z)
                cl = min(max(te, 0.0), 1.0)
                r = z * te + (1.0 - z) * (0.5 + 0.2887 * np.sqrt(pur) * ppf[int(cl * 1024.0 + 0.5)])
                if tt >= 1.0:
                    r = 2.0
                elif tt <= 0.0:
                    r = -1.0
                tc[y, x] = r
                thr[y, x] = th
                dm[y, x] = (1.0 if r > th else 0.0) - cl
        return tc, thr, dm

    @_nb.njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _pass2_nb(tc, thr, t, es, amt):
        """друга проходка: поправка густини (мала карта похибки розгортається тут) -> маска"""
        h, w = t.shape
        sh, sw = es.shape
        out = np.empty((h, w), np.uint8)
        ry, rx = sh / h, sw / w
        for y in _nb.prange(h):
            fy = (y + 0.5) * ry - 0.5
            for x in range(w):
                tt = t[y, x]
                if tt >= 1.0:
                    out[y, x] = 1
                elif tt <= 0.0:
                    out[y, x] = 0
                else:
                    fx = (x + 0.5) * rx - 0.5
                    out[y, x] = 1 if (tc[y, x] - amt * _bil(es, fy, fx)) > thr[y, x] else 0
        return out

    @_nb.njit(cache=True, nogil=True, parallel=True)
    def _ink_nb(m, dog, thr):
        h, w = m.shape
        out = np.empty((h, w), np.uint8)
        for y in _nb.prange(h):
            for x in range(w):
                d = dog[y, x]
                out[y, x] = (1 if d > 0 else 0) if abs(d) > thr else m[y, x]
        return out


def _blur_ds(a, sigma, f):
    """гауссове розмиття великого сигма на зменшеній в f разів копії (результат гладкий, тож майже без втрат)"""
    if f <= 1 or sigma < 2.0 * f:
        return cv2.GaussianBlur(a, (0, 0), sigma)
    h, w = a.shape[:2]
    sm = cv2.resize(a, (max(2, w // f), max(2, h // f)), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), sigma / f)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)


def _ink(m, L, k, thr):
    """підкреслення деталей: на різких тонких контурах (текст, лінії, краї) піксель відповідає знаку перепаду
    світлоти, тож вони лишаються суцільними всередині будь-якої текстури"""
    kk = max(k, 0.5)
    dog = cv2.GaussianBlur(L, (0, 0), 0.7 * kk) - _blur_ds(L, 2.0 * kk, 2 if L.shape[0] >= 400 else 1)
    if _nb is not None:
        m8 = np.ascontiguousarray(m, dtype=np.uint8)
        r = _gpu.run("ink", lambda g: g.ink(m8, dog, float(thr)))
        return r if r is not None else _ink_nb(m8, dog, float(thr))
    strong = np.abs(dog) > thr
    return np.where(strong, dog > 0, m.astype(bool)).astype(np.uint8)


METHODS = {1: "Ескіз", 2: "Контур", 3: "Півтон", 4: "Малюнок", 5: "Гравюра", 6: "IRNM1"}   # назви - лише для людей; усередині (.ikr, налаштування) рівні - числа 1-6
MAIN_LEVELS = 5                    # п'ять рівнів однієї шкали «швидкість ↔ якість»; шостий (BRANCH) - окрема гілка, в шкалу не входить
BRANCH = 6
METHOD_INFO = {
    1: "найшвидше: один поріг на весь кадр",
    2: "швидко: поріг за околом, добре для ліній і тексту",
    3: "збалансовано: напівтони шумом, як у газеті",
    4: "якісно: штрихи, крапки й зерно за змістом",
    5: "найякісніше: штрихи йдуть за формою, текст чіткий, без мерехтіння",
    6: "InkreelNeuralModel v1 — окрема гілка, не вищий рівень: нейромережа, у рази швидше",
}


GLYPH_INFO = {
    1: "найшвидше: знак за яскравістю клітинки",
    2: "швидко: сходинки густини чергуються, тон точніший",
    3: "збалансовано: 8 відтінків, знак за формою - лінії, краї, рамки",
    4: "якісно: 16 відтінків, ще літери й кирилиця, найточніший тон",
    5: "найякісніше: 32 відтінки, весь шрифт, без мерехтіння у відео",
}


def _gray_resized(img, w, h):
    shrink = img.shape[1] >= w and img.shape[0] >= h
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _m_threshold(img, w, h):
    """1. Поріг Оцу на весь кадр"""
    g = _gray_resized(img, w, h)
    _, m = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return m.astype(np.uint8)


def _m_adaptive(img, w, h):
    """2. Адаптивний поріг по околу (гаусове вікно ~h/30) + прибирання поодиноких піксельних плям"""
    g = _gray_resized(img, w, h)
    g = cv2.GaussianBlur(g, (0, 0), 0.6) if h >= 200 else g
    bs = max(11, (int(h / 30) | 1))
    m = cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, 7)
    lo, hi = float(np.percentile(g, 2)), float(np.percentile(g, 98))
    if hi - lo < 18:                                   # майже рівний кадр - без шуму з нічого
        m = (g > (lo + hi) / 2).astype(np.uint8)
    return m.astype(np.uint8)


def _m_halftone(img, w, h):
    """3. Тонова карта + статичний синій шум: класичний растр, швидко й без артефактів повторів"""
    t, _ = tone_map(img, w, h, True)
    return (t > _tile(_blue(), w, h, grain_cell(h))).astype(np.uint8)


def binarize(img: np.ndarray, w: int, h: int, p: Params | None = None, aux: dict | None = None) -> np.ndarray:
    """img: BGR/GRAY ndarray будь-якого розміру -> маска (h, w) uint8 {0,1}, 1 = світло.
    aux (необов'язково): {"state": InkState} - пам'ять попереднього кадру для рівня 5"""
    p = p or Params()
    if p.glyph:
        from . import glyph
        m = glyph.binarize(img, w, h, p, aux.get("state") if aux else None)
        return m ^ 1 if p.invert else m
    if p.method in (1, 2, 3):
        m = {1: _m_threshold, 2: _m_adaptive, 3: _m_halftone}[p.method](img, w, h)
        return m ^ 1 if p.invert else m
    if p.method == BRANCH:
        m = _neuro(img, w, h, p, aux)
    elif p.method == 5:
        m = _ultra(img, w, h, p, aux.get("state") if aux else None)
    else:
        m = _manga(img, w, h, p)[0]
    return m ^ 1 if p.invert else m


def _manga(img, w, h, p):
    """основний (4) рендер: суміш крапок, штрихів, зерна й суцільних заливок за змістом; -> (маска, тон, світлота)"""
    k = h / REF_H
    t, gf = tone_map(img, w, h, p.auto, deband=True)
    if h < 290:
        m = _ink(_small(t, w, h), gf, k, TUNE["ink"]) if TUNE["ink"] > 0 else _small(t, w, h)
        return m, t, gf
    kp = pat_k(k)                                    # понад 720p візерунок росте повільніше за кадр (PAT)
    pp = max(2.0, p.period * kp)
    wob, jit, ang = _hand(w, h, k, pp, p.wobble)
    cg, cb = grain_cell(h), max(1, int(kp + 0.5))
    sg = max(1.0, 3.0 * kp)
    if _nb is not None:
        Pk = analyze_small(t, gf, w, h)
        ma = (t, Pk, wob, jit, ang, cg, cb, pp, _blue(), _blobs(), _C, _S, _ppf_lut())
        r = _gpu.run("mix2", lambda g: g.mix2(*ma))
        tc, thr, dm = r if r is not None else _mix2_nb(*ma)
        f = 2 if h < 1008 else 4                      # похибку густини міряємо на зменшеній копії
        es = cv2.GaussianBlur(cv2.resize(dm, (max(2, w // f), max(2, h // f)), interpolation=cv2.INTER_AREA),
                              (0, 0), max(0.8, sg / f))
        m = _gpu.run("pass2", lambda g: g.pass2(tc, thr, t, es, 0.8))
        if m is None:
            m = _pass2_nb(tc, thr, t, es, 0.8)
    else:
        W, qi = analyze(t, gf, w, h, True)
        tot = W[0] + W[1] + W[2] + 1e-6
        Wd, Ws, Wg = W[0] / tot, W[1] / tot, W[2] / tot
        tc, thr, clean = _mix(t, Wd, Ws, Wg, qi, wob, jit, ang, cg, cb, pp)
        solid1, solid0 = t >= 1.0, t <= 0.0
        m = tc > thr
        err = _blur_ds(m.astype(np.float32) - clean, sg, 2 if h < 1008 else 4)
        err *= 0.8
        tc2 = tc - err
        tc2[solid1] = 2.0
        tc2[solid0] = -1.0
        m = (tc2 > thr).astype(np.uint8)
    if h > REF_H and _nb is not None:                 # понад 720p візерунок не вміщає дрібну деталь: де чисте півтонове зерно точніше - беремо його (див. ink5._halftone_map)
        from . import ink5
        ht = (t > _tile(_blue(), w, h, cg)).astype(np.uint8)
        ph = ink5._halftone_map(t, m, ht, k, ink5.P, gf)
        if float(ph.max()) > 0.0:
            m = ink5._select(m, ht, ph, ink5._static_fields(w, h, cg, cb)[2])
    if TUNE["ink"] > 0:
        m = _ink(m, gf, pat_k(k, PAT["ink"]), TUNE["ink"] * 1.3)
    return m, t, gf


class InkState:
    """Пам'ять попереднього кадру для рівня 5 (поля штрихів, маска, карта вибору): де картинка не змінилась, візерунок
    тримається - штрихи не «кипять». Один екземпляр на послідовність кадрів."""

    def __init__(self):
        self.ink5 = None
        self.glyph = None                       # пам'ять режиму «Символи» (ознаки клітинок і знаки минулого кадру)
        self.tm = ToneMem()                     # часова пам'ять нормування тону (обидва режими), див. TNORM


def tmem_of(st) -> "ToneMem | None":
    """часова пам'ять нормування тону зі стану відео (створюється, якщо її там ще нема); None - стану нема (фото)"""
    if st is None:
        return None
    t = getattr(st, "tm", None)
    if t is None:
        t = st.tm = ToneMem()
    return t


def _ultra(img: np.ndarray, w: int, h: int, p: Params, st: "InkState | None" = None) -> np.ndarray:
    """рівень 5 «Гравюра»: гравюра за формою - штрихи йдуть вздовж форм предметів (див. ink5.py)"""
    from . import ink5
    return ink5.ultra(img, w, h, p, st)


def _neuro(img: np.ndarray, w: int, h: int, p: Params, aux: dict | None = None) -> np.ndarray:
    """рівень 6 IRNM1: навчена мережа (benvid/student) на відеокарті, сама вибирає, на чому рахувати. aux["film"] - FilmState відео
    (згладжує загальний опис кадру в часі). Немає onnxruntime чи моделі, кадр замалий або відеокарта відмовила - рахує «Гравюру»
    (той самий вигляд, лише повільніше): рівень ніколи не лишає без картинки"""
    from . import student
    if img.shape[0] != h or img.shape[1] != w:
        img = _resized_bgr(img, w, h)
    m = student.binarize(img, aux.get("film") if aux else None)
    if m is not None:
        return m
    return _ultra(img, w, h, p, aux.get("state") if aux else None)


def neuro_warm(img: np.ndarray, w: int, h: int, film) -> None:
    """розігрів згладжування рівня 6 кадром, що йде перед шматком відео: лише опис кадру (мережа-«глобус», на процесорі), маска не рахується"""
    from . import student
    if img.shape[0] != h or img.shape[1] != w:
        img = _resized_bgr(img, w, h)
    student.warm(img, film)


def _resized_bgr(img, w, h):
    shrink = img.shape[1] >= w and img.shape[0] >= h
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)


def to_image(mask: np.ndarray) -> np.ndarray:
    """маска {0,1} -> 8-бітне зображення 0/255 (суто чорно-біле)"""
    return mask.astype(np.uint8) * 255
