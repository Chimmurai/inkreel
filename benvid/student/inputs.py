"""Входи студента для одного кадру (чистий numpy, без torch): 4 канали шуму, дрібна копія, опис масштабу, доповнення до кратного 16.
Усе бітово збігається з алгоритмом Ультра (tests/test_student.py звіряє з ink5 і з tools/train_ink.py)."""
from __future__ import annotations

import math
import os
import threading

import cv2
import numpy as np

REF_H = 720.0
PERIOD, RING_Q, RING_SEED = 3.6, 0.3, 5
TILE, TILE_BIG = 1024, 2048
RING_OX, RING_OY = 311, 517
ROLL = (61, 97)
SMALL = 96
NOISE_CH = 4
MIN_H = 48                                          # нижче студент не навчався

_lock = threading.Lock()
_BLUE = None
_BLOBS = None


def blue() -> np.ndarray:
    global _BLUE
    if _BLUE is None:
        _BLUE = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bluenoise128.npy")).astype(np.float32)
    return _BLUE


def blobs() -> np.ndarray:
    global _BLOBS
    if _BLOBS is None:
        bn = blue()
        n = bn.shape[0]
        seeds = (bn < 0.045).astype(np.uint8)
        dist = cv2.distanceTransform(np.tile(1 - seeds, (3, 3)), cv2.DIST_L2, 5)[n:2 * n, n:2 * n]
        dist = dist + np.random.default_rng(5).random((n, n)).astype(np.float32) * 0.3
        _BLOBS = np.argsort(np.argsort(dist.ravel())).reshape(n, n).astype(np.float32) / (n * n - 1)
    return _BLOBS


def _tile_hash(ty, tx):
    h = (ty * 374761393 + tx * 668265263 + 12345) & 0x7FFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0x7FFFFFFF
    return h ^ (h >> 16)


def _tile_at(t: np.ndarray, ys: np.ndarray, xs: np.ndarray, cell: int) -> np.ndarray:
    yy = (ys.astype(np.int64) // cell)[:, None]
    xx = (xs.astype(np.int64) // cell)[None, :]
    hs = _tile_hash(yy >> 7, xx >> 7).astype(np.int64)
    iy = (yy + (hs & 127)) & 127
    ix = (xx + ((hs >> 7) & 127)) & 127
    sw = ((hs >> 14) & 1) == 1
    iy, ix = np.where(sw, ix, iy), np.where(sw, iy, ix)
    ix = np.where(((hs >> 15) & 1) == 1, 127 - ix, ix)
    return t[iy, ix]


def pat_k(k: float) -> float:
    return k ** 0.5 if k > 1.0 else k


def ring_period(h: int) -> float:
    return round(max(2.6, PERIOD * pat_k(h / REF_H)), 2)


def grain_cells(h: int) -> tuple[int, int]:
    return 1, max(1, int(pat_k(h / REF_H) + 0.5))


def ring_T(w: int, h: int) -> int:
    return TILE if max(w, h) <= 1600 else TILE_BIG


def make_ring_tile(T: int, period: float, q: float = RING_Q, seed: int = RING_SEED) -> np.ndarray:
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


_rings: dict = {}
RING_KEEP = 24                                      # плиток на диску (2-8 МБ кожна): кожна нова висота кадру дає нову, без межі тека росла б безмежно


def _prune_rings(d: str, keep: int = None) -> None:
    """лишити в теці d не більше keep найсвіжіше використаних плиток; видалена плитка просто перерахується, якщо знадобиться знову"""
    keep = RING_KEEP if keep is None else keep
    try:
        found = []
        for f in os.listdir(d):
            if f.startswith("ring_T") and f.endswith(".npy") and ".tmp" not in f:
                p = os.path.join(d, f)
                try:
                    found.append((os.path.getmtime(p), p))
                except OSError:
                    pass
        found.sort()
        for _, p in found[:max(0, len(found) - keep)]:
            try:
                os.remove(p)
            except OSError:
                pass
    except OSError:
        pass


def ring_tile(T: int, period: float) -> np.ndarray:
    """кільцева плитка (float16) - у пам'яті (останні 3) і на диску в теці програми (не більше RING_KEEP), щоб наступного запуску не рахувати FFT"""
    key = (T, int(round(period * 100)))
    with _lock:
        a = _rings.get(key)
        if a is not None:
            return a
    path = None
    d = None
    try:
        from ..log import app_dir
        d = os.path.join(app_dir(), "student")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"ring_T{T}_p{key[1]:04d}.npy")
        if os.path.isfile(path):
            a = np.load(path)
            try:
                os.utime(path, None)                # «використано щойно»: найдавніші за цим часом приберуться першими
            except OSError:
                pass
    except Exception:  # noqa: BLE001
        path = None
    if a is None:
        a = make_ring_tile(T, period).astype(np.float16)
        if path:
            try:
                tmp = f"{path}.{os.getpid()}.tmp.npy"
                np.save(tmp, a)
                os.replace(tmp, path)
                _prune_rings(d)
            except OSError:
                pass
    with _lock:
        if len(_rings) >= 3:
            _rings.pop(next(iter(_rings)))
        _rings[key] = a
    return a


def noise_window(w: int, h: int, y0: int, x0: int, ch: int, cw: int) -> np.ndarray:
    """4 канали шуму для вікна кадру w×h: Tg, Td, Tr (у [-1, 1]) і кільцевий (σ = 1) -> (4, ch, cw) float32"""
    ys = np.arange(y0, y0 + ch)
    xs = np.arange(x0, x0 + cw)
    cg, cb = grain_cells(h)
    out = np.empty((NOISE_CH, ch, cw), np.float32)
    out[0] = _tile_at(blue(), ys, xs, cg) * 2 - 1
    out[1] = _tile_at(blobs(), ys, xs, cb) * 2 - 1
    out[2] = _tile_at(blue(), (ys - ROLL[0]) % h, (xs - ROLL[1]) % w, cg) * 2 - 1
    T = ring_T(w, h)
    a = ring_tile(T, ring_period(h))
    out[3] = a[np.ix_((ys + RING_OY) % T, (xs + RING_OX) % T)]
    return out


_noise_cache: dict = {}


def noise_for(w: int, h: int, dtype=np.float16) -> np.ndarray:
    """шум для кадру w×h, доповненого до кратного 16: (1, 4, H, W). Залежить лише від розміру кадру - кешується (останні 2 розміри)"""
    key = (w, h, np.dtype(dtype).str)
    with _lock:
        a = _noise_cache.get(key)
    if a is not None:
        return a
    H, W = h + (-h) % 16, w + (-w) % 16
    a = np.ascontiguousarray(noise_window(w, h, 0, 0, H, W)[None].astype(dtype))
    with _lock:
        if len(_noise_cache) >= 2:
            _noise_cache.pop(next(iter(_noise_cache)))
        _noise_cache[key] = a
    return a


def meta_of(h: int) -> np.ndarray:
    k = h / REF_H
    kp = pat_k(k)
    return np.array([math.log2(k), math.log2(max(kp, 1e-3)), ring_period(h) / 4.0 - 1.0, grain_cells(h)[1] - 1, 1.0 if h < 290 else 0.0], np.float32)


def small_of(bgr: np.ndarray) -> np.ndarray:
    """дрібна копія кадру: (3, 96, 96) RGB 0..1 float32"""
    s = cv2.resize(bgr, (SMALL, SMALL), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(s[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0


def to_bgr(img: np.ndarray) -> np.ndarray:
    """будь-який кадр застосунку (GRAY, BGR, BGRA) -> BGR uint8"""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


_LUT = {}


def image_tensor(bgr: np.ndarray, dtype=np.float16) -> np.ndarray:
    """кадр -> (1, 3, H, W) RGB 0..1, сторони доповнено віддзеркаленням до кратних 16"""
    h, w = bgr.shape[:2]
    ph, pw = (-h) % 16, (-w) % 16
    pad = cv2.copyMakeBorder(bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT_101) if (ph or pw) else bgr
    key = np.dtype(dtype).str
    lut = _LUT.get(key)
    if lut is None:
        lut = _LUT[key] = (np.arange(256, dtype=np.float32) / 255.0).astype(dtype)      # те саме значення, що float32 /255 з наступним приведенням
    return np.ascontiguousarray(lut[pad[:, :, ::-1]].transpose(2, 0, 1))[None]
