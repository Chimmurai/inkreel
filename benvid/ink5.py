"""Рівень 5 «Ультра»: гравюра за формою.

Це прямий наступник рівня 4: ті самі тон, зони, крапки, зерно й контури, але штрихи тепер не в кількох
фіксованих напрямах, а ідуть за формою (волосся, складки, вода, тіні), як у ручній гравюрі пером.

  1. ПОЛЕ НАПРЯМІВ. Багатомасштабний структурний тензор тону (зважений узгодженістю): напрям штриха - вздовж
     ізофот, а там, де структури нема, - гладке продовження сусідніх напрямів. Поле сильно згладжене адаптивно:
     у шумних місцях напрям береться з великого околу, тож штрихи йдуть паралельно, а не звиваються.
  2. ШТРИХ. «Згортка вздовж лінії» (LIC) вузькосмугового шуму: рівномірні, трохи живі штрихи, крок ~3.6 px.
     Довжина штриха обмежена кривизною поля (штрих не зрізає повороти), у світлому - короткі «мазки», у темному -
     довгі лінії; у тінях друга сім'я під 60° (перехресне штрихування). Поріг штриха однаково розподілений, тож
     світлота по штриху = тон.
  3. ЗМІШУВАННЯ. Ваги стилів (штрихи/крапки/зерно) - з тих самих зон, що й у рівні 4; поріг змішується плавно.
  4. ТОЧНИЙ ТОН. Кілька ітерацій зворотного зв'язку по похибці густини (інтегральний регулятор) - світлота
     локально дорівнює тону, а не «приблизно».
  5. КОНТУР. Тонкі різкі краї (текст, лінії) - суцільні, як у рівні 4.
  6. ЧИСТА ЧОРНОТА Й БІЛОТА. У темному (світлому) оточенні чорна (біла) точка тону трохи зсувається, тож на рівних
     темних фонах інтерфейсів нема пилу з поодиноких крапок - так само чисто, як у рівні 4.
  7. ЧАС. Світлота усереднюється з минулими кадрами (шум датчика й стиснення нижче підсилюється в рази), напрям і
     довжина штриха згладжуються між кадрами, а піксель утримує значення, поки тон не змінився помітно (гістерезис),
     тож штрихи не «кипять». Малі кадри (<290p) - пошук із початком від минулої маски.
  8. РУХ. Шум зерна, плямки крапок і шум штрихів - нерухомі в координатах екрана, тож на рухомому вмісті око чіпляється за сам
     візерунок. Оптичний потік між кадрами (motion.py) дає зсув візерунка D(p) для кожної зони: візерунок їде разом із
     вмістом, а нове місце бере візерунок без зсуву. Нерухома сцена - зсувів нема, шлях побітово такий самий, як без руху;
     швидка панорама, що пояснюється потоком, більше не вважається монтажною склейкою.
Усі масштаби прив'язані до висоти кадру (як у рівні 4); понад 720p розмір візерунка росте повільніше за кадр, а зерно - окремі
пікселі (core.PAT), тож додаткові пікселі дають додаткову деталь.
"""
from __future__ import annotations

import contextlib
import math
import threading
from functools import lru_cache

import cv2
import numpy as np

from . import core as C
from . import gpu as _gpu
from . import motion as _mot

try:
    import numba as nb
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # noqa: BLE001
    nb = None
    HAVE_NUMBA = False
else:
    from . import nbcache as _nbcache
    _nbcache.install()

# внутрішні коефіцієнти (підібрані на тестовому наборі скріншотів, в інтерфейсі їх нема)
P = dict(
    ah=240,                                  # висота сітки аналізу (px) - поле напрямів і ваги рахуються на ній
    scales=(4.0, 8.0, 16.0, 32.0),           # масштаби структурного тензора (px на висоті 720)
    pw=3.0, E0=0.004, E1=0.015,              # вага масштабу = узгодженість^pw * енергія
    a0=0.05, a1=0.35,                        # поріг «є структура» -> вага штрихів
    s1=6.0, s2=20.0, c_lo=0.35, c_hi=0.8,    # згладжування напряму: малим і великим вікном
    Lmin=2.5, Lmax=20.0, dev=1.3,            # довжина штриха: мін/макс (px на 720p), допуск зрізання повороту
    lightd=0.6, u0=0.55, u1=0.95,            # у світлому штрихи коротшають і поступаються крапкам
    period=3.6, q=0.3,                       # крок штрихів (px на 720p) і «чистота» періоду
    iters=4, gain=0.7,                       # зворотний зв'язок по густині
    dz=0.08, dzh=0.05, dzs=5.0, dzb0=0.06, dzb1=0.14,   # «чисте» чорне/біле: підйом чорної точки в темних околах, спуск білої - у світлих
    ink=1.3,
    tden=0.85, tmax=8.0,                     # часове усереднення світлоти: макс. вага історії; верхня межа оцінки шуму (рівнів)
    tg0=1.0, tg1=3.0, tk0=1.8, tk1=4.0,      # ворота різниці: мін. і множники рівня шуму (рівні світлоти)
    tb0=0.5, tb1=1.6, cutm=18.0, cutq=60.0, cutc=0.3,   # ворота зсуву (фейд); ознаки склейки: різниця (рівні світлоти) і кореляція мініатюр
    hy=0.05, ema=0.5,                        # гістерезис маски між кадрами; згладжування полів
    spass=6, shold=0.12,                     # малі кадри: проходи пошуку; де тон змінився менше - старт із минулої маски
    guard=1, r0=1.25, r1=2.0,                # запобіжник: де штрихи гірші за зерно L4 за вірністю деталей - беремо зерно
    dzd=0.05,                                # понад 720p чорна точка піднімається слабше (до 1080p: dz - dzd): темна деталь лишається, а не пил
    hg=1, h0=0.6, h1=1.5, hb0=3.0, hb1=4.0, hd0=0.2, hd1=0.35, hq=3.0,            # запобіжник 2 (понад 720p): де чисте півтонове зерно (як рівень 3) передає деталі в 1.5-2.8 раза точніше за штрихи/крапки - беремо його;
                                             # hq: на скільки «важче» його взяти біля 720p (до 1080p плавно знімається)
    g0=0.06, g1=0.20,                        # ворота контрасту: де розмах тону малий - штрихів нема
    ta=1,                                    # шум і штрихи їдуть разом із вмістом (motion.py); 0 - нерухомі відносно екрана
)

_TILE = 1024


class _Tm:
    """прості таймери етапів для налагодження (dbg['tm'])"""

    def __init__(self, on):
        import time
        self.on = on
        self.t = time.perf_counter() if on else 0.0
        self.d = {}
        self._time = time

    def __call__(self, name):
        if self.on:
            n = self._time.perf_counter()
            self.d[name] = self.d.get(name, 0.0) + (n - self.t) * 1000
            self.t = n


# ------------------------------------------------------------------ ресурси
@lru_cache(maxsize=6)
def _ring_tile(T: int, period: float, q: float, seed: int) -> np.ndarray:
    """періодична плитка T×T: білий шум, пропущений крізь вузьке кільце в частотній області -> майже періодичні
    хвилі (крок period px) випадкового напряму; після «згортки вздовж лінії» лишаються штрихи поперек потоку"""
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((T, T)).astype(np.float32)
    F = np.fft.rfft2(n)
    fy = np.fft.fftfreq(T)[:, None]
    fx = np.fft.rfftfreq(T)[None, :]
    r = np.sqrt(fx * fx + fy * fy)
    f0 = 1.0 / period
    G = np.exp(-0.5 * ((r - f0) / (q * f0)) ** 2)
    o = np.fft.irfft2(F * G, s=(T, T)).astype(np.float32)
    return np.ascontiguousarray(o / o.std())


@lru_cache(maxsize=3)
def _static_fields(w: int, h: int, cg: int, cb: int):
    """порогові поля зерна (синій шум), крапок (плямки) і випадкове поле вибору - не залежать від кадру"""
    tg = np.ascontiguousarray(C._tile(C._blue(), w, h, cg), dtype=np.float32)
    td = np.ascontiguousarray(C._tile(C._blobs(), w, h, cb), dtype=np.float32)
    tr = np.ascontiguousarray(np.roll(tg, (61, 97), axis=(0, 1)))
    return tg, td, tr


@lru_cache(maxsize=1)
def _phi_tab() -> np.ndarray:
    from math import erf, sqrt
    x = np.linspace(-6, 6, 4097)
    return np.array([0.5 * (1 + erf(v / sqrt(2))) for v in x], np.float32)


NLT = 4                                       # відліків таблиці σ(L) на піксель довжини
LMAX_T = 64


if HAVE_NUMBA:
    @njit(cache=True, nogil=True, fastmath=True, inline="always")
    def _lic_px(nf, M, SH, ox, oy, x, y, vx, vy, L):
        """відрізок вздовж потоку з трикутною вагою (напіввисота L); далі 8 px - кожен другий відлік із подвійною вагою"""
        fx = x + ox + 2048.5
        fy = y + oy + 2048.5
        gx = fx
        gy = fy
        acc = nf[((int(fy) & M) << SH) | (int(fx) & M)]
        n = int(L)
        ws = 1.0 / (L + 1.0)
        n1 = min(n, 8)
        for i in range(1, n1 + 1):
            fx += vx
            fy += vy
            gx -= vx
            gy -= vy
            acc += (1.0 - i * ws) * (nf[((int(fy) & M) << SH) | (int(fx) & M)] + nf[((int(gy) & M) << SH) | (int(gx) & M)])
        if n > 8:
            fx += vx
            fy += vy
            gx -= vx
            gy -= vy
            for i in range(10, n + 1, 2):
                fx += vx
                fy += vy
                gx -= vx
                gy -= vy
                acc += 2.0 * (1.0 - i * ws) * (nf[((int(fy) & M) << SH) | (int(fx) & M)] + nf[((int(gy) & M) << SH) | (int(gx) & M)])
                fx += vx
                fy += vy
                gx -= vx
                gy -= vy
        return acc

    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _calib_nb(noise, M, nl):
        """σ виходу LIC для L = 0, 1/NLT, 2/NLT, ... (напрям змінюється з кроком, тож нема перекосу від осі)"""
        sd = np.empty(nl, np.float32)
        nf = noise.reshape(-1)
        SH = 0
        while (1 << SH) < M + 1:
            SH += 1
        S = 160
        for j in prange(nl):
            L = j / NLT
            s1 = 0.0
            s2 = 0.0
            cnt = 0
            for a in range(6):
                ang = 0.37 + a * 0.52
                vx = math.cos(ang)
                vy = math.sin(ang)
                for y in range(0, S, 2):
                    for x in range(0, S, 2):
                        z = _lic_px(nf, M, SH, 311, 517, x + 40, y + 40, vx, vy, L)
                        s1 += z
                        s2 += z * z
                        cnt += 1
            m = s1 / cnt
            sd[j] = math.sqrt(max(s2 / cnt - m * m, 1e-6))
        return sd


@lru_cache(maxsize=6)
def _sigma_tab(T: int, period: float, q: float, seed: int) -> np.ndarray:
    tile = _ring_tile(T, period, q, seed)
    return _calib_nb(tile, T - 1, LMAX_T * NLT + 1)


# ------------------------------------------------------------------ поле напрямів (на малій сітці)
def _gb(a: np.ndarray, s: float) -> np.ndarray:
    """гаусове розмиття; великі sigma - на зменшеній копії (поля гладкі, тож без втрат)"""
    if s < 3.5:
        return cv2.GaussianBlur(a, (0, 0), max(s, 0.5))
    f = 2 if s < 8 else 4
    h, w = a.shape
    sm = cv2.resize(a, (max(2, w // f), max(2, h // f)), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), s / f)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)


def _sm(x, a, b):
    return C._sm(x, a, b)


def _flow(ts: np.ndarray, kk: float, p: dict):
    """-> c2, s2 (одиничний вектор подвійного кута напряму ВЗДОВЖ ізофот), best (сила структури 0..1)"""
    gx = cv2.Sobel(ts, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    gy = cv2.Sobel(ts, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    J = (gx * gx, gy * gy, gx * gy)
    A = np.zeros_like(ts)
    B = np.zeros_like(ts)
    Cc = np.zeros_like(ts)
    Wt = np.zeros_like(ts)
    best = np.zeros_like(ts)
    fast = HAVE_NUMBA and ts.dtype == np.float32 and ts.flags.c_contiguous
    for s in p["scales"]:
        Js = [_gb(j, s * kk) for j in J]
        if fast:
            _flow_scale_nb(Js[0], Js[1], Js[2], float(p["pw"]), float(p["E0"]), float(p["E1"]), A, B, Cc, Wt, best)
            continue
        en = Js[0] + Js[1]
        coh = np.sqrt((Js[0] - Js[1]) ** 2 + 4 * Js[2] ** 2) / (en + 1e-12)
        wgt = (coh ** p["pw"]) * _sm(np.sqrt(en), p["E0"], p["E1"])
        inv = 1.0 / (en + 1e-12)
        A += wgt * Js[0] * inv
        B += wgt * Js[1] * inv
        Cc += wgt * Js[2] * inv
        Wt += wgt
        np.maximum(best, wgt, out=best)
    Jf = [_gb(j, p["scales"][-1] * 1.5 * kk) for j in J]
    if fast:
        c2, s2, w_ = np.empty_like(ts), np.empty_like(ts), np.empty_like(ts)
        _flow_pre_nb(A, B, Cc, Wt, Jf[0], Jf[1], Jf[2], best, 0.02, c2, s2, w_)
    else:
        enf = Jf[0] + Jf[1] + 1e-12
        eps = 0.02
        A += eps * Jf[0] / enf
        B += eps * Jf[1] / enf
        Cc += eps * Jf[2] / enf
        Wt += eps
        c2 = -(A - B) / Wt                               # напрям потоку = градієнт + 90° -> подвійний кут + 180°
        s2 = -(2 * Cc) / Wt
        w_ = best + 0.02

    def smo(sig):
        d = _gb(w_, sig * kk) + 1e-9
        return _gb(c2 * w_, sig * kk) / d, _gb(s2 * w_, sig * kk) / d

    ca, sa = smo(p["s1"])
    cb, sb = smo(p["s2"])
    if fast:
        oc, os_ = np.empty_like(ts), np.empty_like(ts)
        _flow_mix_nb(np.ascontiguousarray(ca, dtype=np.float32), np.ascontiguousarray(sa, dtype=np.float32), np.ascontiguousarray(cb, dtype=np.float32),
                     np.ascontiguousarray(sb, dtype=np.float32), float(p["c_lo"]), float(p["c_hi"]), oc, os_)
        return oc, os_, best.astype(np.float32)
    cons = np.sqrt(ca * ca + sa * sa)                # узгодженість напрямів у вікні
    wa = _sm(cons, p["c_lo"], p["c_hi"])
    c = ca * wa + cb * (1 - wa)
    s = sa * wa + sb * (1 - wa)
    nr = np.sqrt(c * c + s * s) + 1e-9
    return (c / nr).astype(np.float32), (s / nr).astype(np.float32), best.astype(np.float32)


# ------------------------------------------------------------------ ядра
if HAVE_NUMBA:
    @njit(cache=True, nogil=True, inline="always")
    def _tile_at(tile, cell, y, x):
        """поріг-плитка кадру в довільній точці (також поза кадром): те саме, що core._tile_build, але без масиву на весь кадр -
        потрібна, коли координату шуму зсунуто разом із вмістом"""
        yy = y // cell
        xx = x // cell
        hh = ((yy >> 7) * 374761393 + (xx >> 7) * 668265263 + 12345) & 0x7FFFFFFF
        hh = ((hh ^ (hh >> 13)) * 1274126177) & 0x7FFFFFFF
        hs = hh ^ (hh >> 16)
        iy = (yy + (hs & 127)) & 127
        ix = (xx + ((hs >> 7) & 127)) & 127
        if (hs >> 14) & 1:
            tmp = iy
            iy = ix
            ix = tmp
        if (hs >> 15) & 1:
            ix = 127 - ix
        return tile[iy, ix]

    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _flow_scale_nb(J0, J1, J2, pw, E0, E1, A, B, Cc, Wt, best):
        """один масштаб поля напрямів: когерентність, вага й накопичення A, B, Cc, Wt, best (де енергії нема, вага 0 і степінь не рахується)"""
        h, w = J0.shape
        for y in prange(h):
            for x in range(w):
                a = J0[y, x]
                b = J1[y, x]
                c = J2[y, x]
                en = a + b
                se = np.sqrt(en)
                z = min(max((se - E0) / (E1 - E0), 0.0), 1.0)
                if z <= 0.0:
                    continue
                zs = z * z * (3.0 - 2.0 * z)
                d = a - b
                coh = np.sqrt(d * d + 4.0 * c * c) / (en + 1e-12)
                if pw == 3.0:
                    cp = coh * coh * coh
                else:
                    cp = coh ** pw
                wgt = cp * zs
                inv = 1.0 / (en + 1e-12)
                A[y, x] += wgt * a * inv
                B[y, x] += wgt * b * inv
                Cc[y, x] += wgt * c * inv
                Wt[y, x] += wgt
                if wgt > best[y, x]:
                    best[y, x] = wgt

    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _flow_pre_nb(A, B, Cc, Wt, F0, F1, F2, best, eps, c2, s2, w_):
        """додає слабкий загальний напрям (eps) і дає c2, s2 (подвійний кут вздовж ізофот) та вагу w_"""
        h, w = A.shape
        for y in prange(h):
            for x in range(w):
                enf = F0[y, x] + F1[y, x] + 1e-12
                wt = Wt[y, x] + eps
                a = A[y, x] + eps * F0[y, x] / enf
                b = B[y, x] + eps * F1[y, x] / enf
                c = Cc[y, x] + eps * F2[y, x] / enf
                c2[y, x] = -(a - b) / wt
                s2[y, x] = -(2.0 * c) / wt
                w_[y, x] = best[y, x] + 0.02

    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _flow_mix_nb(ca, sa, cb, sb, c_lo, c_hi, oc, os_):
        h, w = ca.shape
        for y in prange(h):
            for x in range(w):
                a = ca[y, x]
                b = sa[y, x]
                cons = np.sqrt(a * a + b * b)
                z = min(max((cons - c_lo) / (c_hi - c_lo), 0.0), 1.0)
                wa = z * z * (3.0 - 2.0 * z)
                c = a * wa + cb[y, x] * (1.0 - wa)
                s_ = b * wa + sb[y, x] * (1.0 - wa)
                nr = np.sqrt(c * c + s_ * s_) + 1e-9
                oc[y, x] = c / nr
                os_[y, x] = s_ / nr

    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _render_nb(t, Fk, noise, M, sd, phi, Tg, Td, ppf, o1y, o1x, o2y, o2x, Dk, bn, bl, cg, cb):
        """змішування стилів із штрихами за формою. t (h,w) тон; Fk (sh,sw,5) = [cos2, sin2, L(px), вага штрихів,
        вага крапок]; -> tc (ціль у просторі порогу), thr (поріг), cl (чистий тон 0..1).
        Dk (sh,sw,2) - зсув візерунка (px кадру) разом із вмістом; (1,1,2) - нема зсуву: поля Tg/Td беруться з масивів, як завжди;
        інакше пороги беруться з плиток bn (зерно, комірка cg) і bl (крапки, комірка cb) у зсунутій точці"""
        h, w = t.shape
        nf = noise.reshape(-1)
        SH = 0
        while (1 << SH) < M + 1:
            SH += 1
        usd = Dk.shape[0] > 1
        sh, sw = Fk.shape[0], Fk.shape[1]
        nsd = sd.shape[0]
        tc = np.empty((h, w), np.float32)
        thr = np.empty((h, w), np.float32)
        cl_ = np.empty((h, w), np.float32)
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
        for y in prange(h):
            fy = min(max((y + 0.5) * ry - 0.5, 0.0), sh - 1.0)
            y0 = int(fy)
            y1 = min(y0 + 1, sh - 1)
            dy = fy - y0
            for x in range(w):
                tt = t[y, x]
                if tt >= 1.0:
                    tc[y, x] = 2.0
                    thr[y, x] = 0.5
                    cl_[y, x] = 1.0
                    continue
                if tt <= 0.0:
                    tc[y, x] = -1.0
                    thr[y, x] = 0.5
                    cl_[y, x] = 0.0
                    continue
                dx = dxs[x]
                i0, i1 = x0[x], x1[x]
                w00, w01, w10, w11 = (1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy
                a_s = Fk[y0, i0, 3] * w00 + Fk[y0, i1, 3] * w01 + Fk[y1, i0, 3] * w10 + Fk[y1, i1, 3] * w11
                a_d = Fk[y0, i0, 4] * w00 + Fk[y0, i1, 4] * w01 + Fk[y1, i0, 4] * w10 + Fk[y1, i1, 4] * w11
                a_g = 1.0 - a_d - a_s
                ox_ = 0
                oy_ = 0
                if usd:
                    ox_ = int(np.floor(Dk[y0, i0, 0] * w00 + Dk[y0, i1, 0] * w01 + Dk[y1, i0, 0] * w10 + Dk[y1, i1, 0] * w11 + 0.5))
                    oy_ = int(np.floor(Dk[y0, i0, 1] * w00 + Dk[y0, i1, 1] * w01 + Dk[y1, i0, 1] * w10 + Dk[y1, i1, 1] * w11 + 0.5))
                wd = min(max((0.4 - tt) * 5.0, 0.0), 1.0)
                ta = tt + (np.sqrt(max(tt, 0.0)) - tt) * wd
                ts_ = 0.5
                if a_s > 0.003:
                    c2 = Fk[y0, i0, 0] * w00 + Fk[y0, i1, 0] * w01 + Fk[y1, i0, 0] * w10 + Fk[y1, i1, 0] * w11
                    s2 = Fk[y0, i0, 1] * w00 + Fk[y0, i1, 1] * w01 + Fk[y1, i0, 1] * w10 + Fk[y1, i1, 1] * w11
                    L = Fk[y0, i0, 2] * w00 + Fk[y0, i1, 2] * w01 + Fk[y1, i0, 2] * w10 + Fk[y1, i1, 2] * w11
                    nr = 1.0 / (np.sqrt(c2 * c2 + s2 * s2) + 1e-9)
                    c2 *= nr
                    s2 *= nr
                    vx = np.sqrt(max((1.0 + c2) * 0.5, 0.0))
                    vy = np.sqrt(max((1.0 - c2) * 0.5, 0.0))
                    if s2 < 0:
                        vy = -vy
                    si = min(int(L * 4.0 + 0.5), nsd - 1)
                    z = _lic_px(nf, M, SH, (o1y + ox_) & M, (o1x + oy_) & M, x, y, vx, vy, L)
                    v = min(max(z / sd[si], -5.999), 5.999)
                    f = (v + 6.0) * (4096.0 / 12.0)
                    ii = int(f)
                    u1 = phi[ii] * (1.0 - (f - ii)) + phi[ii + 1] * (f - ii)
                    ts_ = u1
                    if wd > 0.0:
                        vx2 = vx * 0.5 - vy * 0.8660254
                        vy2 = vx * 0.8660254 + vy * 0.5
                        z = _lic_px(nf, M, SH, (o2y + ox_) & M, (o2x + oy_) & M, x, y, vx2, vy2, L)
                        v = min(max(z / sd[si], -5.999), 5.999)
                        f = (v + 6.0) * (4096.0 / 12.0)
                        ii = int(f)
                        u2 = phi[ii] * (1.0 - (f - ii)) + phi[ii + 1] * (f - ii)
                        ts_ = max(u1, u2 * wd)
                if usd:
                    tg = _tile_at(bn, cg, y + oy_, x + ox_)
                    td = _tile_at(bl, cb, y + oy_, x + ox_)
                else:
                    tg = Tg[y, x]
                    td = Td[y, x]
                th = a_d * td + a_s * ts_ + a_g * tg
                te = tt + a_s * (ta - tt)
                pur = a_d * a_d + a_s * a_s + a_g * a_g
                z = min(max((pur - 0.6) / 0.4, 0.0), 1.0)
                z = z * z * (3.0 - 2.0 * z)
                cl = min(max(te, 0.0), 1.0)
                tc[y, x] = z * te + (1.0 - z) * (0.5 + 0.2887 * np.sqrt(pur) * ppf[int(cl * 1024.0 + 0.5)])
                thr[y, x] = th
                cl_[y, x] = min(max(tt, 0.0), 1.0)
        return tc, thr, cl_

    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _fb_nb(tc, thr, cumf, prev, hy, out):
        """маска = (ціль - поправка - поріг) > гістерезис; суцільні пікселі закладені в tc (2 / -1), тож поправка
        (обмежена ±0.9) їх не зсуває. prev (h,w) uint8 і hy - гістерезис щодо попереднього кадру (hy = 0 - без)"""
        h, w = tc.shape
        for y in prange(h):
            for x in range(w):
                v = tc[y, x] - cumf[y, x] - thr[y, x]
                if prev.shape[0] == h:
                    m = v > (hy - 2.0 * hy * prev[y, x])
                else:
                    m = v > 0.0
                out[y, x] = 1 if m else 0

    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _fb_acc_nb(tc, thr, cumf, prev, hy, f, acc):
        """те саме, що _fb_nb + _blk_u8, але маска не записується: з неї одразу беруться суми блоків f×f (проміжний
        прохід зворотного зв'язку потребує лише їх). Рядок маски рахується суцільним циклом (векторизується), далі - суми по f"""
        h, w = tc.shape
        usep = prev.shape[0] == h
        nfull = w // f
        for by in prange(acc.shape[0]):
            y0 = by * f
            y1 = min(h, y0 + f)
            for xb in range(acc.shape[1]):
                acc[by, xb] = 0.0
            mrow = np.empty(w, np.uint8)
            for y in range(y0, y1):
                if usep:
                    for x in range(w):
                        v = tc[y, x] - cumf[y, x] - thr[y, x]
                        mrow[x] = v > (hy - 2.0 * hy * prev[y, x])
                else:
                    for x in range(w):
                        mrow[x] = (tc[y, x] - cumf[y, x] - thr[y, x]) > 0.0
                if f == 4:
                    for xb in range(nfull):
                        x0 = xb * 4
                        acc[by, xb] += np.float64(np.int32(mrow[x0]) + np.int32(mrow[x0 + 1]) + np.int32(mrow[x0 + 2]) + np.int32(mrow[x0 + 3]))
                elif f == 2:
                    for xb in range(nfull):
                        x0 = xb * 2
                        acc[by, xb] += np.float64(np.int32(mrow[x0]) + np.int32(mrow[x0 + 1]))
                else:
                    for xb in range(nfull):
                        sm_ = 0
                        for x in range(xb * f, xb * f + f):
                            sm_ += mrow[x]
                        acc[by, xb] += sm_
                for x in range(nfull * f, w):
                    acc[by, nfull] += mrow[x]

    @njit(cache=True, nogil=True, parallel=True)
    def _blk_u8(a, f, acc):
        h, w = a.shape
        for by in prange(acc.shape[0]):
            for xb in range(acc.shape[1]):
                s = 0
                for y in range(by * f, min(h, (by + 1) * f)):
                    for x in range(xb * f, min(w, (xb + 1) * f)):
                        s += a[y, x]
                acc[by, xb] = s

    @njit(cache=True, nogil=True, parallel=True)
    def _blk_nb(a, f, acc):
        h, w = a.shape
        for by in prange(acc.shape[0]):
            for xb in range(acc.shape[1]):
                acc[by, xb] = 0.0
            for y in range(by * f, min(h, (by + 1) * f)):
                for x in range(w):
                    acc[by, x // f] += a[y, x]

    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _tdl_nb(L, Lp, bg, rho, g0, g1, out):
        """рекурсивне усереднення світлоти в часі: де різниця з минулим кадром мала (шум датчика, стиснення) - беремо
        історію, де велика (рух, нова деталь) - новий кадр. bg (h,w) - ворота зсуву (фейд, зміна освітлення)"""
        h, w = L.shape
        inv = 1.0 / (g1 - g0)
        for y in prange(h):
            for x in range(w):
                l = L[y, x]
                d = abs(l - Lp[y, x])
                z = min(max((d - g0) * inv, 0.0), 1.0)
                z = z * z * (3.0 - 2.0 * z)
                wt = rho * bg[y, x] * (1.0 - z)
                out[y, x] = wt * Lp[y, x] + (1.0 - wt) * l


def _block_mean(a_sum, f, h, w):
    """суми блоків f×f -> середні (краї рахуються за справжньою площею блока)"""
    sh, sw = a_sum.shape
    cy = np.minimum(f, h - np.arange(sh) * f).astype(np.float32)[:, None]
    cx = np.minimum(f, w - np.arange(sw) * f).astype(np.float32)[None, :]
    return a_sum / (cy * cx)



_NOD = np.zeros((1, 1, 2), np.float32)


def _draw_cpu(t, F, noise, T, sd, Tg, Td, f, iters, gain, sg, prev, hy, cl_s, Dk=None, cg=1, cb=1):
    """штрихи + зворотний зв'язок по густині (numba); cl_s - список з одним елементом: блоки чистого тону лишаються між викликами"""
    h, w = t.shape
    tc, thr, cl = _render_nb(t, F, noise, T - 1, sd, _phi_tab(), Tg, Td, C._ppf_lut(), 311, 517, 700, 123,
                             _NOD if Dk is None else Dk, C._blue(), C._blobs(), cg, cb)
    sh_, sw_ = -(-h // f), -(-w // f)
    acc = np.zeros((sh_, sw_), np.float64)
    if cl_s[0] is None:
        _blk_nb(cl, f, acc)
        cl_s[0] = _block_mean(acc, f, h, w).astype(np.float32)
    cum = np.zeros((sh_, sw_), np.float32)
    out = np.empty((h, w), np.uint8)
    for _ in range(iters):
        _fb_acc_nb(tc, thr, cv2.resize(cum, (w, h), interpolation=cv2.INTER_LINEAR), prev, hy, f, acc)
        es = cv2.GaussianBlur((_block_mean(acc, f, h, w) - cl_s[0]).astype(np.float32), (0, 0), max(0.8, sg))
        cum += gain * es
        np.clip(cum, -0.9, 0.9, out=cum)
    _fb_nb(tc, thr, cv2.resize(cum, (w, h), interpolation=cv2.INTER_LINEAR), prev, hy, out)
    return out


def _draw_gpu(g, t, F, noise, T, sd, Tg, Td, f, iters, gain, sg, prev, hy, cl_s, Dk=None, cg=1, cb=1):
    """те саме на відеокарті: ціль, поріг і маска лишаються на пристрої, у процесор ходять лише малі сітки блоків"""
    h, w = t.shape
    tc, thr, cl = g.render(t, F, noise, T - 1, sd, _phi_tab(), Tg, Td, C._ppf_lut(), 311, 517, 700, 123, Dk, C._blue(), C._blobs(), cg, cb)
    sh_, sw_ = -(-h // f), -(-w // f)
    if cl_s[0] is None:
        cl_s[0] = _block_mean(g.blocks_f32(cl, f, h, w), f, h, w).astype(np.float32)
    cum = np.zeros((sh_, sw_), np.float32)
    for _ in range(iters):
        ob = g.fb(tc, thr, cum, prev, hy, w, h)
        es = cv2.GaussianBlur((_block_mean(g.blocks_u8(ob, f, h, w), f, h, w) - cl_s[0]).astype(np.float32), (0, 0), max(0.8, sg))
        cum += gain * es
        np.clip(cum, -0.9, 0.9, out=cum)
    return g.down(g.fb(tc, thr, cum, prev, hy, w, h), (h, w), np.uint8)


# ------------------------------------------------------------------ плутанина між кадрами
class _Mem:
    """пам'ять попереднього кадру: розмір, усереднені тон/світлота (повні й на сітці аналізу), поля штрихів, маска, карта вибору"""
    __slots__ = ("shape", "gf", "Ls", "tdf", "F", "m", "pg", "D", "bh", "ph")

    def __init__(self):
        self.shape = self.gf = self.Ls = self.tdf = self.F = self.m = self.pg = self.D = self.bh = self.ph = None


_SL = 160                                                   # ширина малої копії світлоти для воріт і виявлення склейки


def _small_l(L):
    h, w = L.shape
    sw = min(_SL, w)
    return cv2.resize(L, (sw, max(8, int(round(h * sw / w)))), interpolation=cv2.INTER_AREA)


def _lfilter(L, mem, pr, info):
    """часове усереднення світлоти. Шум датчика й стиснення нижче в тонмапі підсилюється в рази, тож гасимо його тут:
    рівень шуму міряємо за медіаною різниці з минулим кадром, і лише в межах цього рівня беремо історію"""
    Lp = mem.gf
    h, w = L.shape
    if Lp is None or Lp.shape != L.shape or mem.Ls is None:
        info["cut"] = True
        return L
    Ls = _small_l(L)
    dd = Ls - mem.Ls
    ad = np.abs(dd)
    a0, b0 = Ls - Ls.mean(), mem.Ls - mem.Ls.mean()
    sa, sb = float(a0.std()), float(b0.std())
    corr = float((a0 * b0).mean() / (sa * sb)) if (sa > 3.0 and sb > 3.0) else 1.0
    if ad.mean() > pr["cutm"] or (ad > pr["cutq"]).mean() > 0.30 or corr < pr["cutc"]:      # монтажна склейка
        info["cut"] = True
        return L
    # 60-й процентиль |різниці| (а не медіана): чорні смуги й плоскі ділянки з нульовою різницею не мають занижувати шум
    sig = float(np.percentile(np.abs(L[::4, ::4] - Lp[::4, ::4]), 60)) / 0.8416
    sig = min(sig, pr["tmax"])
    g0 = max(pr["tg0"], pr["tk0"] * sig)
    g1 = max(pr["tg1"], pr["tk1"] * sig)
    bias = cv2.GaussianBlur(dd, (0, 0), 2.0)
    bg = (1.0 - _sm(np.abs(bias), pr["tb0"], pr["tb1"])).astype(np.float32)
    bgf = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    out = np.empty_like(L)
    r = _gpu.run("tdl", lambda g: g.tdl(L, Lp, bgf, pr["tden"], g0, g1))
    if r is not None:
        out = r
    else:
        _tdl_nb(L, Lp, bgf, pr["tden"], g0, g1, out)
    info["sig"] = sig
    return out


WARM = 10                    # скільки кадрів перед шматком відео прогрівають пам'ять (останній - повним малюванням)


def warm(st, img, w, h, p, **over):
    """дешевий розігрів часової пам'яті: лише світлота (кадр не малюється). Для шматків відео й після перемотування:
    так усереднення в часі збігається з тим, що було б при послідовній обробці"""
    mem = _mem(st)
    if mem is None:
        return
    pr = {**P, **over}
    with _guard():
        tm = C.tmem_of(st)
        cut = tm.observe(img)
        L = C.gray_stage(img, w, h, p.auto, True, tm, cut)
        if pr["tden"] > 0 and mem.shape == (w, h) and mem.gf is not None and mem.Ls is not None:
            L = _lfilter(L, mem, pr, {"cut": False})
        C.tone_warm(L, w, h, p.auto, tm, cut)                       # пам'ять нормування тону рухається разом із пам'яттю світлоти
    mem.shape, mem.gf, mem.Ls = (w, h), L, _small_l(L)
    mem.tdf = mem.F = mem.m = mem.pg = mem.D = mem.bh = mem.ph = None


def prime(st, img, w, h, auto: bool = True) -> None:
    """найдешевший прогрів перед шматком відео: кадр лише декодований - з нього лише сирі параметри нормування тону для пам'яті
    (повільні зміни яскравості, як згасання, тягнуться довше, ніж кадрів у розігріві світлоти)"""
    if st is not None:
        C.tone_prime(img, w, h, auto, C.tmem_of(st))


def prewarm():
    """прогнати «Ультру» на двох малих кадрах (нижче й вище межі 290 px): компілює або підвантажує з кешу всі ядра - і для
    окремого кадру, і для часової пам'яті. Викликається в одному процесі пулу перед усіма іншими"""
    if not HAVE_NUMBA or C._nb is None:
        return
    yy, xx = np.mgrid[0:300, 0:534]
    base = (127 + 90 * np.sin(xx / 23.0) * np.cos(yy / 17.0) + 30 * ((xx // 40 + yy // 40) % 2)).clip(0, 255).astype(np.uint8)
    cv2.circle(base, (260, 150), 70, 255, -1)
    p = C.Params(method=5)
    for w, h in ((534, 300), (320, 180)):
        a = cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA)
        img = np.dstack([a, np.roll(a, 7, 1), 255 - a])
        st = C.InkState()
        warm(st, img, w, h, p)
        for k in range(2):
            C.binarize(np.roll(img, k * 3, 1), w, h, p, {"state": st})
    # «східці» кодека гасяться лише від 540p, а тут кадри менші: ядра прибирання компілюємо (або беремо з кешу) окремо
    yy, xx = np.mgrid[0:96, 0:160]
    C._deband((20.0 + (xx // 16) * 1.5 + (yy // 16) * 0.7).astype(np.float32), 720)
    z = np.full((8, 8), 0.5, np.float32)                  # кінець _text_snap виконується лише на кадрах із текстом: компілюємо (або беремо з кешу) заздалегідь
    C._snap_tail_nb(z, z, z, z)
    _err_grid(np.zeros((32, 32), np.uint8), np.zeros((32, 32), np.float32), 1.0, 8, 8)       # карти захисту >720p рахуються лише на великих кадрах: ядро помилки теж компілюємо заздалегідь
    _select_nb(np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.uint8), np.zeros((2, 2), np.float32), np.zeros((8, 8), np.float32))
    _fb_acc_nb(np.zeros((8, 8), np.float32), np.zeros((8, 8), np.float32), np.zeros((8, 8), np.float32), np.zeros((1, 1), np.uint8), 0.0, 4, np.zeros((2, 2), np.float64))
    z8 = np.zeros((8, 8), np.float32)                     # ядра, що на кадрах 300 px висоти не зустрічаються (поправки рахуються лише на плато / від 320 px): теж заздалегідь
    _clip_add_nb(z8, np.zeros((2, 2), np.float32))
    _dz_apply_nb(z8, np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32))
    _rec_acc_nb(z8.copy(), z8, z8, z8, True)
    _rec_acc_nb(z8.copy(), z8, z8, z8, False)


def _mem(st):
    if st is None:
        return None
    m = getattr(st, "ink5", None)
    if m is None:
        m = st.ink5 = _Mem()
    return m



def _bp(a: np.ndarray, k: float) -> np.ndarray:
    """смуговий фільтр деталей: G(0.7k) - G(2.2k); другий доданок гладкий, тож на зменшеній копії"""
    lo = cv2.GaussianBlur(a, (0, 0), max(0.6, 0.7 * k))
    h, w = a.shape
    sm = cv2.resize(a, (max(2, w // 2), max(2, h // 2)), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), max(0.6, 2.2 * k / 2))
    return lo - cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)


def _bt_get(t, k, half, bcache):
    """смуговий фільтр деталей самого тону. Однаковий для карти штрихів/зерна (_guard_map) і півтонової (_halftone_map): той самий кадр,
    k і масштаб, тож рахуємо раз на кадр (bcache - словник, що живе один кадр)"""
    key = (t.shape, float(k), bool(half))
    if bcache is not None and bcache.get("key") == key:
        return bcache["bt"]
    tt = np.clip(t, 0, 1).astype(np.float32)
    kk = k
    if half:
        h, w = t.shape
        tt = cv2.resize(tt, (max(2, w // 2), max(2, h // 2)), interpolation=cv2.INTER_AREA)
        kk = k / 2
    bt = _bp(tt, kk)
    if bcache is not None:
        bcache["key"], bcache["bt"] = key, bt
    return bt


if HAVE_NUMBA:
    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _err_grid_nb(lo8, sm, bt, gh, gw):
        """середнє по клітинках gh×gw від (lo - U - bt)², де lo = lo8/255, U - білінійне розтягування малої копії sm на весь кадр (півпіксельні
        центри, як у cv2.resize). Те саме, що (lo - resize(sm) - bt)² з подальшим resize INTER_AREA у (gw, gh), але без п'яти проміжних масивів
        на кадр. Окремий швидкий шлях, коли sm рівно вдвічі менша, а клітинка - 4×4 px (звичайний випадок): ваги U по x - сталі 0.25/0.75"""
        h, w = lo8.shape
        shh, sww = sm.shape
        c255 = np.float32(1.0 / 255.0)
        ry, rx = shh / h, sww / w
        fast = (w == 4 * gw) and (2 * sww == w) and (h % gh == 0)
        x0a = np.empty(w, np.int32)
        x1a = np.empty(w, np.int32)
        dxa = np.empty(w, np.float32)
        for x in range(w):
            fx = min(max((x + 0.5) * rx - 0.5, 0.0), sww - 1.0)
            i0 = int(fx)
            x0a[x] = i0
            x1a[x] = min(i0 + 1, sww - 1)
            dxa[x] = fx - i0
        xe = np.empty(gw + 1, np.int64)
        for g in range(gw + 1):
            xe[g] = min(w, (g * w + gw - 1) // gw)
        E = np.zeros((gh, gw), np.float32)
        q1 = np.float32(0.25)
        q3 = np.float32(0.75)
        for gy in prange(gh):
            ya = (gy * h + gh - 1) // gh
            yb = min(h, ((gy + 1) * h + gh - 1) // gh)
            acc = np.zeros(gw, np.float64)
            col = np.empty(sww, np.float32)
            colp = np.empty(sww + 2, np.float32)
            row = np.empty(w, np.float32)
            for y in range(ya, yb):
                fy = min(max((y + 0.5) * ry - 0.5, 0.0), shh - 1.0)
                y0 = int(fy)
                y1 = min(y0 + 1, shh - 1)
                dy = np.float32(fy - y0)
                for j in range(sww):
                    col[j] = sm[y0, j] * (np.float32(1.0) - dy) + sm[y1, j] * dy
                if fast:
                    colp[0] = col[0]
                    colp[sww + 1] = col[sww - 1]
                    for j in range(sww):
                        colp[j + 1] = col[j]
                    for g in range(gw):
                        j0 = 2 * g
                        x = 4 * g
                        c0 = colp[j0]
                        c1 = colp[j0 + 1]
                        c2 = colp[j0 + 2]
                        c3 = colp[j0 + 3]
                        d0 = np.float32(lo8[y, x]) * c255 - (q1 * c0 + q3 * c1) - bt[y, x]
                        d1 = np.float32(lo8[y, x + 1]) * c255 - (q3 * c1 + q1 * c2) - bt[y, x + 1]
                        d2 = np.float32(lo8[y, x + 2]) * c255 - (q1 * c1 + q3 * c2) - bt[y, x + 2]
                        d3 = np.float32(lo8[y, x + 3]) * c255 - (q3 * c2 + q1 * c3) - bt[y, x + 3]
                        acc[g] += np.float64(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3)
                else:
                    for x in range(w):
                        dx = dxa[x]
                        d = np.float32(lo8[y, x]) * c255 - (col[x0a[x]] * (np.float32(1.0) - dx) + col[x1a[x]] * dx) - bt[y, x]
                        row[x] = d * d
                    for g in range(gw):
                        s_ = np.float32(0.0)
                        for x in range(xe[g], xe[g + 1]):
                            s_ += row[x]
                        acc[g] += s_
            nr = max(1, yb - ya)
            for g in range(gw):
                nc = max(1, xe[g + 1] - xe[g])
                E[gy, g] = acc[g] / (nr * nc)
        return E


def _err_grid(m, bt, k, sh, sw):
    """середня за клітинками сітки (sh×sw) квадратна помилка смугового фільтра маски m відносно смугового фільтра тону bt (див. _bp).
    Маска - 0/255 у uint8: гаус у цілих у ~3 рази швидший, а похибка округлення (≤0.002) далеко нижча за енергію деталей"""
    h, w = m.shape
    m8 = (m != 0).view(np.uint8) * np.uint8(255)
    lo8 = cv2.GaussianBlur(m8, (0, 0), max(0.6, 0.7 * k))
    sm = cv2.resize(m8, (max(2, w // 2), max(2, h // 2)), interpolation=cv2.INTER_AREA).astype(np.float32) * np.float32(1.0 / 255.0)
    sm = cv2.GaussianBlur(sm, (0, 0), max(0.6, 2.2 * k / 2))
    return _err_grid_nb(lo8, sm, bt, sh, sw)


def _detail_err(m, bt, k, half, sh, sw):
    """-> середня за клітинками сітки (sh×sw) квадратна помилка смугового фільтра маски m відносно тону (bt з _bt_get). Нижче 4K -
    одним проходом ядра (_err_grid); від 4K (half) - на половинній копії, як і раніше"""
    h, w = m.shape
    if HAVE_NUMBA and not half:
        return _err_grid(m, bt, k, sh, sw)
    mf = m.astype(np.float32)
    kk = k
    if half:
        mf = cv2.resize(mf, (max(2, w // 2), max(2, h // 2)), interpolation=cv2.INTER_AREA)
        kk = k / 2
    e = (_bp(mf, kk) - bt) ** 2
    return cv2.resize(e, (sw, sh), interpolation=cv2.INTER_AREA)


def _guard_cal(k: float) -> float:
    """Чим більший кадр, тим точніше дрібне зерно (комірка 1 px) «влучає» в тон, тож без поправки порівняння штрихів із зерном
    на великих кадрах майже скрізь обирає зерно - замість штрихування виходить пил (1080p: зерну діставалось 20% площі проти 8% на
    720p). Поріг масштабуємо так, щоб частка зерна лишалась як на 720p (≈8%): до 720p рівно 1 (без змін), 1080p - 1.6, 1440p і вище - 2.5.
    Виміряно на 8 скріншотах (частка зерна: 720p 0.078; 1080p при 1.6 ≈0.08; 1440p при 2.5 ≈0.079)"""
    return float(np.interp(k, (1.0, 1.5, 2.0), (1.0, 1.6, 2.5)))


def _guard_map(t, mS, mG, a_s4, k, pr, bcache=None):
    """-> p_grain на малій сітці (h/4): ймовірність взяти зерно L4 замість штрихів. Де штрихи за помилкою деталей
    у >= r0..r1 разів гірші, а рівень 4 теж тут не мав штрихів"""
    h, w = t.shape
    sh, sw = max(2, h // 4), max(2, w // 4)
    half = h >= C.BIG_H                    # 4K: результат йде на сітку h/4, тож смуговий фільтр рахуємо на половинній копії (у 4 рази менше даних)
    bt = _bt_get(t, k, half, bcache)
    eS = _detail_err(mS, bt, k, half, sh, sw)
    eG = _detail_err(mG, bt, k, half, sh, sw)
    sg = max(1.0, 6.0 * k) / 4
    eS = cv2.GaussianBlur(eS, (0, 0), sg)
    eG = cv2.GaussianBlur(eG, (0, 0), sg)
    r = np.log2((eS + 2e-4) / (eG + 2e-4))
    c = _guard_cal(k)
    x = np.clip((r - math.log2(pr["r0"] * c)) / (math.log2(pr["r1"] * c) - math.log2(pr["r0"] * c)), 0, 1)
    x = x * x * (3 - 2 * x)
    x = cv2.GaussianBlur(x, (0, 0), max(0.8, 4.0 * k / 4))
    a4 = cv2.resize(a_s4, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return (x * np.clip(1 - 1.5 * a4, 0, 1)).astype(np.float32)


if HAVE_NUMBA:
    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _select_nb(mS, mG, pg, Tr):
        """mG там, де поріг вибору Tr нижчий за (білінійно розтягнуту) карту pg, інакше mS. Координати по x рахуються раз на кадр,
        по y - раз на рядок: pg спершу змішується вертикально (один рядок малої сітки), потім горизонтально"""
        h, w = mS.shape
        sh, sw = pg.shape
        out = np.empty((h, w), np.uint8)
        ry, rx = sh / h, sw / w
        x0a = np.empty(w, np.int32)
        x1a = np.empty(w, np.int32)
        dxa = np.empty(w, np.float32)
        for x in range(w):
            fx = min(max((x + 0.5) * rx - 0.5, 0.0), sw - 1.0)
            i0 = int(fx)
            x0a[x] = i0
            x1a[x] = min(i0 + 1, sw - 1)
            dxa[x] = fx - i0
        for y in prange(h):
            fy = min(max((y + 0.5) * ry - 0.5, 0.0), sh - 1.0)
            y0 = int(fy)
            y1 = min(y0 + 1, sh - 1)
            dy = np.float32(fy - y0)
            col = np.empty(sw, np.float32)
            nz = False
            for j in range(sw):
                c = pg[y0, j] * (np.float32(1.0) - dy) + pg[y1, j] * dy
                col[j] = c
                if c != 0.0:
                    nz = True
            if not nz:                                  # у цьому рядку вибирати нічого (p = 0 скрізь): беремо mS, якщо Tr не від'ємний
                for x in range(w):
                    out[y, x] = mG[y, x] if Tr[y, x] < 0.0 else mS[y, x]
                continue
            for x in range(w):
                dx = dxa[x]
                p = col[x0a[x]] * (np.float32(1.0) - dx) + col[x1a[x]] * dx
                out[y, x] = mG[y, x] if Tr[y, x] < p else mS[y, x]
        return out


HALF_H = 1600                          # від 4K (як BIG_H) помилку рахуємо на половинній копії; нижче - повна: на половині губиться саме дрібна деталь, заради якої запобіжник


def _halftone_map(t, mC, mH, k, pr, gf, bcache=None):
    """-> p_halftone на малій сітці (h/4): ймовірність взяти чисте півтонове зерно (рівень 3) замість штрихів/крапок/зерна L4. Те саме мірило, що в _guard_map
    (помилка смугового фільтра деталей відносно тону), але порівнюється вже зібраний кадр із чистим півтоном: де візерунок стилізації не вміщає
    дрібну деталь (темні й середні ділянки з текстурою), півтон точніший. Понад 720p; до 1080p поріг плавно знижується"""
    h, w = t.shape
    sh, sw = max(2, h // 4), max(2, w // 4)
    kk = k
    half = h >= HALF_H                                # від HALF_H помилку рахуємо на половинній копії (результат однаково йде на сітку h/4)
    bt = _bt_get(t, k, half, bcache)
    eC = _detail_err(mC, bt, k, half, sh, sw)
    eH = _detail_err(mH, bt, k, half, sh, sw)
    sg = max(1.0, 6.0 * kk) / 4
    eC = cv2.GaussianBlur(eC, (0, 0), sg)
    eH = cv2.GaussianBlur(eH, (0, 0), sg)
    r = np.log2((eC + 2e-4) / (eH + 2e-4))
    hs = float(np.clip((k - 1.0) / 0.5, 0.0, 1.0))
    gl = cv2.GaussianBlur(cv2.resize(gf.astype(np.float32) * np.float32(1.0 / 255.0), (sw, sh), interpolation=cv2.INTER_AREA), (0, 0), max(1.0, 3.0 * k / 4))
    dk = 1.0 - _sm(gl, pr["hd0"], pr["hd1"])                             # 1 - темна округа джерела (там вірність деталі важливіша), 0 - середні й світлі (стилізація міцніша)
    h0 = pr["hb0"] + (pr["h0"] - pr["hb0"]) * dk + pr["hq"] * (1.0 - hs)
    h1 = pr["hb1"] + (pr["h1"] - pr["hb1"]) * dk + pr["hq"] * (1.0 - hs)
    x = np.clip((r - h0) / (h1 - h0), 0, 1)
    x = x * x * (3 - 2 * x)
    return cv2.GaussianBlur(x, (0, 0), max(0.8, 4.0 * kk / 4)).astype(np.float32)


def _shift_static(T, Dk, w, h):
    """поле вибору Tr (масив на весь кадр) теж їде разом із вмістом: те саме округлене зсування, що й у ядрі малювання"""
    Df = cv2.resize(Dk, (w, h), interpolation=cv2.INTER_LINEAR)
    gx, gy = C._grid(w, h)
    return cv2.remap(T, gx + np.floor(Df[..., 0] + 0.5), gy + np.floor(Df[..., 1] + 0.5), cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)


def _select(mS, mG, pg, Tr):
    r = _gpu.run("select", lambda g: g.select(mS, mG, pg, Tr))
    return r if r is not None else _select_nb(mS, mG, pg, Tr)


def _ultra_small(img, w, h, p, st=None, pr=P):
    """малі кадри: пряме двійкове пошукування, як у рівні 4, але з більшою кількістю проходів, усередненою в часі
    світлотою й стартом від попереднього кадру - що не змінилось, те не мерехтить"""
    k = h / C.REF_H
    mem = _mem(st)
    have = mem is not None and mem.shape == (w, h) and mem.gf is not None
    info = {"cut": not have}
    lf = (lambda L: _lfilter(L, mem, pr, info)) if (have and pr["tden"] > 0) else None
    t, gf = C.tone_map(img, w, h, p.auto, lf, tmem=C.tmem_of(st))
    cut = bool(info["cut"])
    init = None
    if not cut and mem.m is not None and mem.m.shape == (h, w) and mem.tdf is not None and mem.tdf.shape == t.shape:
        d = np.abs(t - mem.tdf)
        cut = bool(d.mean() > 0.09 or (d > 0.25).mean() > 0.35)
        if not cut:
            hold = cv2.GaussianBlur(d, (0, 0), 1.0) < pr["shold"]
            init = np.where(hold, mem.m, t > C._tile(C._blue(), w, h, 1)).astype(np.uint8)
    m = C._small(t, w, h, init, max(C.TUNE["passes"], pr["spass"]))
    if C.TUNE["ink"] > 0:
        m = C._ink(m, gf, k, C.TUNE["ink"])
    if mem is not None:
        mem.shape, mem.gf, mem.Ls, mem.tdf, mem.m = (w, h), gf, _small_l(gf), t, m
        mem.F = mem.pg = mem.ph = None
    return m


_LOCK = threading.RLock()
_SAFE = None


def _guard():
    """Шар потоків numba «workqueue» не витримує паралельних викликів із кількох потоків (процес аварійно завершується);
    omp й tbb - витримують. Тож блокування потрібне лише для workqueue (або доки шар ще невідомий)."""
    global _SAFE
    if _SAFE is None and HAVE_NUMBA:
        try:
            _SAFE = nb.threading_layer() in ("omp", "tbb")
        except Exception:  # noqa: BLE001
            pass
    return contextlib.nullcontext() if _SAFE else _LOCK


def ultra(img: np.ndarray, w: int, h: int, p, st=None, dbg: dict | None = None, **over) -> np.ndarray:
    """img (BGR/GRAY) -> маска (h, w) uint8 {0,1}, 1 = світло. st - InkState (пам'ять кадрів) або None"""
    with _guard():
        return _ultra(img, w, h, p, st, dbg, over)


def _ultra(img, w, h, p, st, dbg, over):
    if not HAVE_NUMBA or C._nb is None:
        return C._manga(img, w, h, p)[0]
    if h < 290:
        return _ultra_small(img, w, h, p, st, {**P, **over})
    pr = {**P, **over}
    tm = _Tm(dbg is not None)
    k = h / C.REF_H
    mem = _mem(st)
    have = mem is not None and mem.shape == (w, h) and mem.gf is not None
    info = {"cut": not have}
    lf = (lambda L: _lfilter(L, mem, pr, info)) if (have and pr["tden"] > 0) else None
    t, gf = C.tone_map(img, w, h, p.auto, lf, deband=True, tmem=C.tmem_of(st))
    tm("tone")
    t = _recover(t, gf, k)
    if pr["dz"] > 0:
        hs_ = float(np.clip((k - 1.0) / 0.5, 0.0, 1.0))                  # понад 720p мертва зона слабшає (плавно до 1080p): 1.5-2 рази більше пікселів на деталь - пилом вона вже не стає
        t = _deadzone(t, k, {**pr, "dz": pr["dz"] - pr["dzd"] * hs_} if hs_ > 0 else pr)
    tm("recover")
    # --- сітка аналізу
    sc = min(1.0, pr["ah"] / h)
    lw, lh = max(16, int(round(w * sc))), max(16, int(round(h * sc)))
    kk = k * (lh / h)
    td = cv2.resize(t, (lw, lh), interpolation=cv2.INTER_AREA).astype(np.float32)
    cut = bool(info["cut"])
    if not cut and mem.tdf is not None and mem.tdf.shape == td.shape:
        d = np.abs(td - mem.tdf)
        cut = bool(d.mean() > 0.09 or (d > 0.25).mean() > 0.35)         # монтажна склейка: історію скидаємо
    ts = cv2.GaussianBlur(cv2.bilateralFilter(td, -1, 0.12, max(2.0, 3 * kk)), (0, 0), max(0.8, 1.2 * kk))
    tm("prep")
    Dk, bmv = None, None
    mp = {**_mot.TA, **{k_[3:]: v_ for k_, v_ in over.items() if k_.startswith("ta_")}}      # налагодження: ta_decay=… тощо
    if mem is not None and pr["ta"]:                                   # шум і штрихи їдуть разом із вмістом (motion.py)
        if mem.tdf is not None and mem.tdf.shape == td.shape:
            if float(np.abs(td - mem.tdf).max()) < 0.004:             # кадр повторився (60 -> 30 fps, стоп-кадр): руху нема, потік не рахуємо
                bmv, bh, expl = np.zeros((lh, lw, 2), np.float32), mem.bh, True
            else:
                bmv, bh, expl = _mot.analyze(td, mem.tdf, ts, kk, None if cut else mem.bh, mp)
            if cut and expl:
                cut = False                                            # кадри різні через рух (швидка панорама), а не склейку
            if cut:
                mem.D, mem.bh, bmv = None, None, None
            else:
                mem.bh = bh
                mem.D = _mot.advance(mem.D, bmv, w / lw, kk, mp)
                Dk = mem.D if _mot.active(mem.D, mp) else None
                bmv = bmv if np.any(bmv) else None
        else:
            mem.D = mem.bh = None
    tm("motion")
    c2, s2, best = _flow(ts, kk, pr)
    tm("flow")
    conf = _sm(best, pr["a0"], pr["a1"]).astype(np.float32)
    if pr["g1"] > 0:                                   # штрихи - для форми: у слабоконтрастних рівних місцях лишається зерно
        r = max(1, int(round(6 * kk)))
        ker = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
        rng_ = cv2.GaussianBlur(cv2.dilate(ts, ker) - cv2.erode(ts, ker), (0, 0), max(1.0, 2 * kk))
        conf = conf * _sm(rng_, pr["g0"], pr["g1"])
    conf = cv2.GaussianBlur(conf, (0, 0), max(1.0, 3 * kk))
    # довжина штриха: не довша за ту, що ще йде за поворотом поля; у світлому - коротка
    sx = lw / w
    dcx = cv2.Sobel(c2, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    dcy = cv2.Sobel(c2, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    dsx = cv2.Sobel(s2, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    dsy = cv2.Sobel(s2, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    kap = cv2.GaussianBlur(0.5 * np.sqrt(dcx ** 2 + dcy ** 2 + dsx ** 2 + dsy ** 2) * sx, (0, 0), max(1.0, 2 * kk))
    kp = C.pat_k(k)                                    # понад 720p візерунок росте повільніше за кадр (core.PAT)
    Lc = np.sqrt(2 * pr["dev"] * kp / np.maximum(kap, 1e-4))
    tl = cv2.GaussianBlur(np.clip(td, 0, 1), (0, 0), max(1.0, 4 * kk))
    u = _sm(tl, pr["u0"], pr["u1"])
    L = np.maximum(np.clip(Lc, pr["Lmin"] * kp, pr["Lmax"] * kp) * (1 - 0.75 * u), pr["Lmin"] * kp)
    Pk4 = C.analyze_small(t, gf, w, h)
    a_s4 = cv2.resize(Pk4[..., 1], (lw, lh), interpolation=cv2.INTER_LINEAR)
    a_d4 = cv2.resize(Pk4[..., 0], (lw, lh), interpolation=cv2.INTER_LINEAR)
    a_s = np.clip(np.maximum(conf * (1 - pr["lightd"] * u), a_s4), 0, 1)
    if pr.get("grain_only"):
        a_s = np.zeros_like(a_s)
    a_d = np.minimum(a_d4, 1 - a_s)
    tm("fields")
    Fk = np.stack([c2, s2, L, a_s, a_d], -1).astype(np.float32)
    if not cut and mem.F is not None and mem.F.shape == Fk.shape:        # часове згладжування полів
        a = pr["ema"]
        Fk = a * Fk + (1 - a) * (mem.F if bmv is None else _mot.warp_grid(mem.F, bmv))      # з минулого кадру беремо те місце, звідки приїхав вміст
        nr = np.sqrt(Fk[..., 0] ** 2 + Fk[..., 1] ** 2) + 1e-9
        Fk[..., 0] /= nr
        Fk[..., 1] /= nr
        Fk = np.ascontiguousarray(Fk)
    # --- штрихи, змішування, зворотний зв'язок
    per = round(max(2.6, pr["period"] * kp), 2)
    T = _TILE if max(w, h) <= 1600 else 2 * _TILE
    noise = _ring_tile(T, per, pr["q"], 5)
    sd = _sigma_tab(T, per, pr["q"], 5)
    cg, cb = C.grain_cell(h), max(1, int(kp + 0.5))
    Tg, Td, Tr = _static_fields(w, h, cg, cb)
    f = 4 if h >= 360 else 2
    hold = (not cut) and mem is not None and mem.m is not None and mem.m.shape == (h, w)
    prev = mem.m if hold else np.zeros((1, 1), np.uint8)
    hy = pr["hy"] if hold else 0.0
    if hold and bmv is not None and Dk is not None:                    # гістерезис порівнює з тим місцем минулого кадру, звідки приїхав піксель
        bf = cv2.resize(bmv * np.float32(w / lw), (w, h), interpolation=cv2.INTER_LINEAR)
        if float(np.abs(bf).max()) >= 0.5:
            gx_, gy_ = C._grid(w, h)
            prev = cv2.remap(prev, gx_ + np.round(bf[..., 0]), gy_ + np.round(bf[..., 1]), cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    sg = max(1.0, 3.0 * kp) / f
    cl_s = [None]
    bcache = {}

    def draw(F, iters):
        tm("pre-draw")
        out = None
        if _gpu.kernel_ok("draw"):
            out = _gpu.run("draw", lambda g: _draw_gpu(g, t, F, noise, T, sd, Tg, Td, f, iters, pr["gain"], sg, prev, hy, cl_s, Dk, cg, cb))
        if out is None:
            out = _draw_cpu(t, F, noise, T, sd, Tg, Td, f, iters, pr["gain"], sg, prev, hy, cl_s, Dk, cg, cb)
        tm("feedback")
        return out

    out = draw(Fk, int(pr["iters"]))
    pg = None
    if pr["guard"] and not pr.get("grain_only"):
        need = float(((Fk[..., 3] > 0.25) & (a_s4 < 0.6)).mean())
        if need > 0.02:
            FG = Fk.copy()
            FG[..., 3] = 0.0
            FG[..., 4] = a_d4
            outG = draw(FG, 3)
            tm("pre-guard")
            pg = _guard_map(t, out, outG, a_s4, k, pr, bcache)
            tm("guard-map")
            if mem is not None and not cut and mem.pg is not None and mem.pg.shape == pg.shape:
                pg = 0.5 * pg + 0.5 * mem.pg
            out = _select(out, outG, pg, Tr if Dk is None else _shift_static(Tr, Dk, w, h))
            tm("select")
    ph_keep = None
    if pr["hg"] and h > C.REF_H and not pr.get("grain_only"):         # лише понад 720p: до 720p кадр лишається таким, як був
        Tgs = Tg if Dk is None else _shift_static(Tg, Dk, w, h)
        ht = t > Tgs
        if hold:                                                          # гістерезис, як у зерні: майже на порозі лишаємо минуле
            ht = np.where(np.abs(t - Tgs) < hy, prev > 0, ht)
        ht = ht.astype(np.uint8)
        ph = _halftone_map(t, out, ht, k, pr, gf, bcache)
        if mem is not None and not cut and mem.ph is not None and mem.ph.shape == ph.shape:
            ph = 0.5 * ph + 0.5 * mem.ph                                    # карта вибору усереднюється в часі, як і карта зерна L4
        tm("ht-map")
        ph_keep = ph
        if float(ph.max()) > 0.0:
            out = _select(out, ht, ph, Tr if Dk is None else _shift_static(Tr, Dk, w, h))
        tm("ht-select")
    m = C._ink(out, gf, C.pat_k(k, C.PAT["ink"]), C.TUNE["ink"] * pr["ink"])
    tm("ink")
    if mem is not None:
        mem.shape, mem.gf, mem.Ls, mem.tdf, mem.F, mem.m, mem.pg = (w, h), gf, _small_l(gf), td, Fk, m, pg
        mem.ph = ph_keep
    if dbg is not None:
        dbg.update(t=t, gf=gf, conf=conf, a_s=a_s, a_d=a_d, L=L, c2=c2, s2=s2, cut=cut, tm=tm.d, b=bmv, D=None if mem is None else mem.D)
    return m


if HAVE_NUMBA:
    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _dz_apply_nb(t, lo_s, hi_s):
        """(t - lo)/(1 - lo - hi), де lo і hi - малі поля, білінійно розтягнуті на кадр (півпіксельні центри, як у cv2.resize); суцільні 0 і 1 лишаються.
        Одним проходом замість двох розтягувань на весь кадр і п'яти проміжних масивів"""
        h, w = t.shape
        sh, sw = lo_s.shape
        out = np.empty((h, w), np.float32)
        ry, rx = sh / h, sw / w
        x0a = np.empty(w, np.int32)
        x1a = np.empty(w, np.int32)
        dxa = np.empty(w, np.float32)
        for x in range(w):
            fx = min(max((x + 0.5) * rx - 0.5, 0.0), sw - 1.0)
            i0 = int(fx)
            x0a[x] = i0
            x1a[x] = min(i0 + 1, sw - 1)
            dxa[x] = fx - i0
        one = np.float32(1.0)
        for y in prange(h):
            fy = min(max((y + 0.5) * ry - 0.5, 0.0), sh - 1.0)
            y0 = int(fy)
            y1 = min(y0 + 1, sh - 1)
            dy = np.float32(fy - y0)
            lc = np.empty(sw, np.float32)
            hc = np.empty(sw, np.float32)
            for j in range(sw):
                lc[j] = lo_s[y0, j] * (one - dy) + lo_s[y1, j] * dy
                hc[j] = hi_s[y0, j] * (one - dy) + hi_s[y1, j] * dy
            for x in range(w):
                v = t[y, x]
                if v >= 1.0 or v <= 0.0:
                    out[y, x] = v
                else:
                    dx = dxa[x]
                    lo = lc[x0a[x]] * (one - dx) + lc[x1a[x]] * dx
                    hi = hc[x0a[x]] * (one - dx) + hc[x1a[x]] * dx
                    out[y, x] = (v - lo) / (one - lo - hi)
        return out

    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _clip_add_nb(t, ds):
        """clip(t + розтягнута на кадр мала поправка ds, 0, 1) одним проходом (білінійно, півпіксельні центри)"""
        h, w = t.shape
        sh, sw = ds.shape
        out = np.empty((h, w), np.float32)
        ry, rx = sh / h, sw / w
        x0a = np.empty(w, np.int32)
        x1a = np.empty(w, np.int32)
        dxa = np.empty(w, np.float32)
        for x in range(w):
            fx = min(max((x + 0.5) * rx - 0.5, 0.0), sw - 1.0)
            i0 = int(fx)
            x0a[x] = i0
            x1a[x] = min(i0 + 1, sw - 1)
            dxa[x] = fx - i0
        one = np.float32(1.0)
        for y in prange(h):
            fy = min(max((y + 0.5) * ry - 0.5, 0.0), sh - 1.0)
            y0 = int(fy)
            y1 = min(y0 + 1, sh - 1)
            dy = np.float32(fy - y0)
            col = np.empty(sw, np.float32)
            for j in range(sw):
                col[j] = ds[y0, j] * (one - dy) + ds[y1, j] * dy
            for x in range(w):
                dx = dxa[x]
                out[y, x] = min(max(t[y, x] + (col[x0a[x]] * (one - dx) + col[x1a[x]] * dx), np.float32(0.0)), one)
        return out

    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _rec_acc_nb(delta, pz, gt, r, hi_side):
        """delta -= pz·gt·0.42·clip(-r,0,1)^0.8 (світла сторона) або += pz·gt·0.42·clip(r,0,1)^0.8 (темна)"""
        h, w = delta.shape
        for y in prange(h):
            for x in range(w):
                if pz[y, x] <= 0.0:                         # поза плато поправки нема: степінь (повільна) лише там, де вона потрібна
                    continue
                v = -r[y, x] if hi_side else r[y, x]
                v = min(max(v, np.float32(0.0)), np.float32(1.0))
                if v > 0.0:
                    a = pz[y, x] * gt[y, x] * np.float32(0.42) * np.float32(v ** 0.8)
                    if hi_side:
                        delta[y, x] -= a
                    else:
                        delta[y, x] += a


def _deadzone(t, k, pr):
    """«Чисте» чорне й біле: де в околі майже все чорне (або біле), решта тону - не деталь, а пил із рідких крапок.
    Чорна точка піднімається (нижче неї - суцільне чорне), діапазон [lo, 1-hi] розтягується на [0, 1]. Околи, де є тон,
    і градієнти не чіпаємо - крапки там лишаються і плавно починаються від нуля."""
    h, w = t.shape
    f = 4 if h >= 360 else 2
    sm_ = cv2.resize(np.clip(t, 0.0, 1.0), (max(8, w // f), max(8, h // f)), interpolation=cv2.INTER_AREA)
    tb = cv2.GaussianBlur(sm_, (0, 0), max(1.0, pr["dzs"] * k / f))
    lo = pr["dz"] * (1.0 - _sm(tb, pr["dzb0"], pr["dzb1"]))
    hi = pr["dzh"] * _sm(tb, 1.0 - pr["dzb1"], 1.0 - pr["dzb0"])
    if HAVE_NUMBA and t.dtype == np.float32 and t.flags.c_contiguous:
        return _dz_apply_nb(t, np.ascontiguousarray(lo, dtype=np.float32), np.ascontiguousarray(hi, dtype=np.float32))
    lo = cv2.resize(lo.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    hi = cv2.resize(hi.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    out = (t - lo) / (1.0 - lo - hi)
    return np.where((t >= 1.0) | (t <= 0.0), t, out).astype(np.float32)


def _recover(t, gf, k):
    """повертає півтони в пересвічені/чорні плато, але лише де там справді є структура (а не шум чи градієнт інтерфейсу)"""
    h, w = t.shape
    sc = min(1.0, 320.0 / h)
    dn = (lambda a: cv2.resize(a, (max(8, int(w * sc)), max(8, int(h * sc))), interpolation=cv2.INTER_AREA)) if sc < 1 else (lambda a: a)
    kk = k * sc
    ts_, gs_ = dn(t), dn(gf)
    delta = np.zeros_like(ts_)
    prep = None                                         # локальна структура джерела однакова для обох боків: рахується раз і лише коли є плато
    for hi_side in (True, False):
        plate = (ts_ > 0.96) if hi_side else (ts_ < 0.04)
        pz = cv2.GaussianBlur(plate.astype(np.float32), (0, 0), max(2.0, 3 * kk)) > 0.75
        if not pz.any():
            continue
        pz = cv2.GaussianBlur(pz.astype(np.float32), (0, 0), max(1.5, 2 * kk))
        if prep is None:
            loc = cv2.GaussianBlur(gs_, (0, 0), max(2.0, 5 * kk))
            d = gs_ - loc
            sdr = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), max(3.0, 10 * kk)))
            prep = (_sm(sdr, 2.5, 7.0), np.clip(d / (2.2 * (sdr + 4.0)), -1, 1))
        gt, r = prep
        if HAVE_NUMBA and delta.dtype == np.float32 and delta.flags.c_contiguous:
            _rec_acc_nb(delta, np.ascontiguousarray(pz, dtype=np.float32), np.ascontiguousarray(gt, dtype=np.float32),
                        np.ascontiguousarray(r, dtype=np.float32), hi_side)
        elif hi_side:
            delta -= pz * gt * 0.42 * np.clip(-r, 0, 1) ** 0.8
        else:
            delta += pz * gt * 0.42 * np.clip(r, 0, 1) ** 0.8
    if not delta.any():
        return t
    if HAVE_NUMBA and t.dtype == np.float32 and t.flags.c_contiguous and sc < 1:
        return _clip_add_nb(t, delta)
    if sc < 1:
        delta = cv2.resize(delta, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(t + delta, 0, 1).astype(t.dtype)
