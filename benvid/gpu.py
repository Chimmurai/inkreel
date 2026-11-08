"""Прискорення відеокартою (OpenCL) - для будь-якого виробника: NVIDIA, AMD, Intel (драйвер відеокарти вже містить OpenCL).

Принцип: ті самі за-піксельні ядра, що й у numba (core.py, ink5.py), переписані на OpenCL C. Кожне ядро - заміна один-в-один:
приймає й повертає numpy-масиви, тож решта коду (низькорозмірний аналіз, cv2) не змінюється. Безпека:
  • пристрій обирається лише з відеокарт (процесорні OpenCL-пристрої ігноруються, крім тестів: INKREEL_GPU=cpu);
  • кожне ядро перевіряється проти numba на синтетичних даних і міряється проти нього ж (probe): вмикаються лише ті, що
    правильні й швидші; результат кешується у %LOCALAPPDATA%\\InkReel\\gpu.json (ключ: пристрій, драйвер, хеш коду ядер);
  • probe іде в окремому процесі з тайм-аутом - збій або зависання драйвера не зачіпає програму;
  • будь-яка помилка під час роботи вимикає GPU в цьому процесі, а кадр дорахується на CPU (виклик повертає None).
Режим: INKREEL_GPU = auto (типово) | off | cpu (дозволити процесорний OpenCL - лише для тестів)."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

import importlib.util

cl = None
HAVE_CL = importlib.util.find_spec("pyopencl") is not None


def _import_cl():
    """pyopencl підвантажується лише коли справді потрібна відеокарта (імпорт ≈ секунда, воркерам без GPU він ні до чого)"""
    global cl, HAVE_CL
    if cl is None and HAVE_CL:
        try:
            import pyopencl
            cl = pyopencl
        except Exception:  # noqa: BLE001
            HAVE_CL = False
    return cl


KERNELS = ("draw", "tone", "pca", "pl", "fin", "tdl", "select", "ink", "mix2", "pass2")

_SRC = r"""
#pragma OPENCL FP_CONTRACT OFF
static inline float clampf(float v, float a, float b){ return fmin(fmax(v, a), b); }

static inline float lic_px(__global const float* noise, int M, int ox, int oy, int x, int y, float vx, float vy, float L){
    float fx = x + ox + 2048.5f, fy = y + oy + 2048.5f, gx = fx, gy = fy;
    int W = M + 1;
    float acc = noise[((int)fy & M) * W + ((int)fx & M)];
    int n = (int)L;
    float ws = 1.0f / (L + 1.0f);
    for (int i = 1; i <= n; i++){
        fx += vx; fy += vy; gx -= vx; gy -= vy;
        float wt;
        if (i > 8){ if (i & 1) continue; wt = 2.0f * (1.0f - i * ws); } else wt = 1.0f - i * ws;
        acc += wt * (noise[((int)fy & M) * W + ((int)fx & M)] + noise[((int)gy & M) * W + ((int)gx & M)]);
    }
    return acc;
}

static inline float phi_look(__global const float* phi, float z, float sdv){
    float v = clampf(z / sdv, -5.999f, 5.999f);
    float f = (v + 6.0f) * (4096.0f / 12.0f);
    int ii = (int)f;
    return phi[ii] * (1.0f - (f - ii)) + phi[ii + 1] * (f - ii);
}

/* поріг-плитка кадру в довільній точці (як core._tile_build): потрібна, коли координату шуму зсунуто разом із вмістом */
static inline float tile_at(__global const float* tile, int cell, int y, int x){
    int yy = y >= 0 ? y / cell : -((-y + cell - 1) / cell);
    int xx = x >= 0 ? x / cell : -((-x + cell - 1) / cell);
    long hh = (((long)(yy >> 7)) * 374761393L + ((long)(xx >> 7)) * 668265263L + 12345L) & 0x7FFFFFFFL;
    hh = ((hh ^ (hh >> 13)) * 1274126177L) & 0x7FFFFFFFL;
    long hs = hh ^ (hh >> 16);
    int iy = (yy + (int)(hs & 127L)) & 127;
    int ix = (xx + (int)((hs >> 7) & 127L)) & 127;
    if ((hs >> 14) & 1L){ int tmp = iy; iy = ix; ix = tmp; }
    if ((hs >> 15) & 1L) ix = 127 - ix;
    return tile[iy * 128 + ix];
}

__kernel void render(__global const float* t, __global const float* Fk, __global const float* noise, int M,
                     __global const float* sd, int nsd, __global const float* phi, __global const float* Tg,
                     __global const float* Td, __global const float* ppf, int o1y, int o1x, int o2y, int o2x,
                     int w, int h, int sw, int sh,
                     __global const float* Dk, int usd, __global const float* bn, __global const float* bl, int cg, int cb,
                     __global float* tc, __global float* thr, __global float* cl_){
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int p = y * w + x;
    float ry = (float)sh / h, rx = (float)sw / w;
    float tt = t[p];
    if (tt >= 1.0f){ tc[p] = 2.0f; thr[p] = 0.5f; cl_[p] = 1.0f; return; }
    if (tt <= 0.0f){ tc[p] = -1.0f; thr[p] = 0.5f; cl_[p] = 0.0f; return; }
    float fx = fmin(fmax((x + 0.5f) * rx - 0.5f, 0.0f), sw - 1.0f);
    int i0 = (int)fx; int i1 = min(i0 + 1, sw - 1); float dx = fx - i0;
    float fy = fmin(fmax((y + 0.5f) * ry - 0.5f, 0.0f), sh - 1.0f);
    int y0 = (int)fy; int y1 = min(y0 + 1, sh - 1); float dy = fy - y0;
    float w00 = (1 - dx) * (1 - dy), w01 = dx * (1 - dy), w10 = (1 - dx) * dy, w11 = dx * dy;
#define FK(c) (Fk[(y0*sw+i0)*5+c]*w00 + Fk[(y0*sw+i1)*5+c]*w01 + Fk[(y1*sw+i0)*5+c]*w10 + Fk[(y1*sw+i1)*5+c]*w11)
    float a_s = FK(3), a_d = FK(4);
    float a_g = 1.0f - a_d - a_s;
    int oxo = 0, oyo = 0;
    if (usd){
#define DK(c) (Dk[(y0*sw+i0)*2+c]*w00 + Dk[(y0*sw+i1)*2+c]*w01 + Dk[(y1*sw+i0)*2+c]*w10 + Dk[(y1*sw+i1)*2+c]*w11)
        oxo = (int)floor(DK(0) + 0.5f);
        oyo = (int)floor(DK(1) + 0.5f);
    }
    float wd = clampf((0.4f - tt) * 5.0f, 0.0f, 1.0f);
    float ta = tt + (sqrt(fmax(tt, 0.0f)) - tt) * wd;
    float ts_ = 0.5f;
    if (a_s > 0.003f){
        float c2 = FK(0), s2 = FK(1), L = FK(2);
        float nr = 1.0f / (sqrt(c2 * c2 + s2 * s2) + 1e-9f);
        c2 *= nr; s2 *= nr;
        float vx = sqrt(fmax((1.0f + c2) * 0.5f, 0.0f));
        float vy = sqrt(fmax((1.0f - c2) * 0.5f, 0.0f));
        if (s2 < 0) vy = -vy;
        int si = min((int)(L * 4.0f + 0.5f), nsd - 1);
        float z = lic_px(noise, M, (o1y + oxo) & M, (o1x + oyo) & M, x, y, vx, vy, L);
        float u1 = phi_look(phi, z, sd[si]);
        ts_ = u1;
        if (wd > 0.0f){
            float vx2 = vx * 0.5f - vy * 0.8660254f;
            float vy2 = vx * 0.8660254f + vy * 0.5f;
            z = lic_px(noise, M, (o2y + oxo) & M, (o2x + oyo) & M, x, y, vx2, vy2, L);
            float u2 = phi_look(phi, z, sd[si]);
            ts_ = fmax(u1, u2 * wd);
        }
    }
    float tgv = usd ? tile_at(bn, cg, y + oyo, x + oxo) : Tg[p];
    float tdv = usd ? tile_at(bl, cb, y + oyo, x + oxo) : Td[p];
    float th = a_d * tdv + a_s * ts_ + a_g * tgv;
    float te = tt + a_s * (ta - tt);
    float pur = a_d * a_d + a_s * a_s + a_g * a_g;
    float z = clampf((pur - 0.6f) / 0.4f, 0.0f, 1.0f);
    z = z * z * (3.0f - 2.0f * z);
    float cl = clampf(te, 0.0f, 1.0f);
    tc[p] = z * te + (1.0f - z) * (0.5f + 0.2887f * sqrt(pur) * ppf[(int)(cl * 1024.0f + 0.5f)]);
    thr[p] = th;
    cl_[p] = clampf(tt, 0.0f, 1.0f);
}

/* маска = (ціль - поправка - поріг) > гістерезис; поправка cum (мала сітка csh x csw) розгортається білінійно, як cv2.resize */
__kernel void fb(__global const float* tc, __global const float* thr, __global const float* cum, int csw, int csh,
                 __global const uchar* prev, int use_prev, float hy, int w, int h, __global uchar* out){
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int p = y * w + x;
    float fx = fmin(fmax((x + 0.5f) * ((float)csw / w) - 0.5f, 0.0f), csw - 1.0f);
    float fy = fmin(fmax((y + 0.5f) * ((float)csh / h) - 0.5f, 0.0f), csh - 1.0f);
    int x0 = (int)fx, y0 = (int)fy; int x1 = min(x0 + 1, csw - 1), y1 = min(y0 + 1, csh - 1);
    float dx = fx - x0, dy = fy - y0;
    float c = (cum[y0*csw+x0] * (1 - dx) + cum[y0*csw+x1] * dx) * (1 - dy) + (cum[y1*csw+x0] * (1 - dx) + cum[y1*csw+x1] * dx) * dy;
    float v = tc[p] - c - thr[p];
    int m;
    if (use_prev) m = v > (hy - 2.0f * hy * prev[p]); else m = v > 0.0f;
    out[p] = m ? 1 : 0;
}

/* суми блоків f x f: uchar -> int, float -> float (по одному блоку на потік) */
__kernel void blk_u8(__global const uchar* a, int f, int w, int h, int bw, int bh, __global int* acc){
    int bx = get_global_id(0), by = get_global_id(1);
    if (bx >= bw || by >= bh) return;
    int s = 0;
    for (int y = by * f; y < min(h, (by + 1) * f); y++)
        for (int x = bx * f; x < min(w, (bx + 1) * f); x++) s += a[y * w + x];
    acc[by * bw + bx] = s;
}
__kernel void blk_f32(__global const float* a, int f, int w, int h, int bw, int bh, __global float* acc){
    int bx = get_global_id(0), by = get_global_id(1);
    if (bx >= bw || by >= bh) return;
    float s = 0;
    for (int y = by * f; y < min(h, (by + 1) * f); y++)
        for (int x = bx * f; x < min(w, (bx + 1) * f); x++) s += a[y * w + x];
    acc[by * bw + bx] = s;
}

__kernel void tone(__global const float* L, __global const float* base, __global const float* loc, __global const float* sg,
                   __global const float* bl, __global const float* wl, __global const float* th,
                   float lo, float rngb, float gain, float rel, float abs_w, float lc, int NB, int NT, int n,
                   __global float* out){
    int p = get_global_id(0);
    if (p >= n) return;
    float l = L[p], bs = base[p], d = l - bs;
    int ib = (int)(clampf((bs - lo) / rngb, 0.0f, 1.0f) * NB + 0.5f);
    float b = bl[ib];
    float t_abs = b + gain * d / rngb * wl[ib];
    float t = (1.0f - rel) * t_abs + rel * (0.5f + ((l - loc[p]) + abs_w * (l - 128.0f)) / 110.0f);
    t += lc * clampf(d / fmax(sg[p] * 2.0f, 7.0f), -1.0f, 1.0f);
    float v = clampf(t, 0.0f, 1.0f);
    float r = th[(int)(v * NT + 0.5f)];
    if (r >= 0.995f) r = 1.01f; else if (r <= 0.005f) r = -0.01f;
    out[p] = r;
}

__kernel void pca(__global const uchar* lab, float vL, float va, float vb, float off, float scale, float Lo, float mix,
                  int n, __global float* out){
    int p = get_global_id(0);
    if (p >= n) return;
    float l = lab[p * 3];
    float q = (l * vL + lab[p * 3 + 1] * va + lab[p * 3 + 2] * vb - off) * scale + Lo;
    out[p] = clampf((1.0f - mix) * l + mix * q, 0.0f, 255.0f);
}

__kernel void pl(__global const float* L, __global const float* hp2, float e0, float e1, int n, __global float* G, __global float* LG){
    int p = get_global_id(0);
    if (p >= n) return;
    float e = sqrt(fmax(hp2[p], 0.0f));
    float z = clampf((e - e0) / (e1 - e0), 0.0f, 1.0f);
    float g = 1.0f - z * z * (3.0f - 2.0f * z);
    G[p] = g; LG[p] = L[p] * g;
}

__kernel void fin(__global const float* L, __global const float* T, __global const float* F, float t0, float t1, int n, __global float* out){
    int p = get_global_id(0);
    if (p >= n) return;
    float l = L[p], d = T[p] - l;
    float z = clampf((fabs(d) - t0) / (t1 - t0), 0.0f, 1.0f);
    out[p] = l + F[p] * (1.0f - z * z * (3.0f - 2.0f * z)) * d;
}

__kernel void tdl(__global const float* L, __global const float* Lp, __global const float* bg, float rho, float g0, float g1, int n, __global float* out){
    int p = get_global_id(0);
    if (p >= n) return;
    float l = L[p], d = fabs(l - Lp[p]);
    float z = clampf((d - g0) / (g1 - g0), 0.0f, 1.0f);
    z = z * z * (3.0f - 2.0f * z);
    float wt = rho * bg[p] * (1.0f - z);
    out[p] = wt * Lp[p] + (1.0f - wt) * l;
}

static inline float bil(__global const float* a, int sh, int sw, float fy, float fx){
    fy = fmin(fmax(fy, 0.0f), sh - 1.0f); fx = fmin(fmax(fx, 0.0f), sw - 1.0f);
    int y0 = (int)fy, x0 = (int)fx; int y1 = min(y0 + 1, sh - 1), x1 = min(x0 + 1, sw - 1);
    float dy = fy - y0, dx = fx - x0;
    return (a[y0*sw+x0] * (1 - dx) + a[y0*sw+x1] * dx) * (1 - dy) + (a[y1*sw+x0] * (1 - dx) + a[y1*sw+x1] * dx) * dy;
}

__kernel void select_k(__global const uchar* mS, __global const uchar* mG, __global const float* pg, int psh, int psw,
                       __global const float* Tr, int w, int h, __global uchar* out){
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int p = y * w + x;
    float pr = bil(pg, psh, psw, (y + 0.5f) * ((float)psh / h) - 0.5f, (x + 0.5f) * ((float)psw / w) - 0.5f);
    out[p] = Tr[p] < pr ? mG[p] : mS[p];
}

__kernel void ink_k(__global const uchar* m, __global const float* dog, float thr, int n, __global uchar* out){
    int p = get_global_id(0);
    if (p >= n) return;
    float d = dog[p];
    out[p] = fabs(d) > thr ? (d > 0 ? 1 : 0) : m[p];
}

static inline float tl(__global const float* tab, int y, int x, int c){
    int yy = y / c, xx = x / c;
    long h = (((long)(yy >> 7)) * 374761393L + ((long)(xx >> 7)) * 668265263L + 12345L) & 0x7FFFFFFFL;
    h = ((h ^ (h >> 13)) * 1274126177L) & 0x7FFFFFFFL;
    h = h ^ (h >> 16);
    int iy = (yy + (int)(h & 127)) & 127;
    int ix = (xx + (int)((h >> 7) & 127)) & 127;
    if ((h >> 14) & 1){ int s = iy; iy = ix; ix = s; }
    if ((h >> 15) & 1) ix = 127 - ix;
    return tab[iy * 128 + ix];
}

static inline float fatan2(float y, float x){
    float ax = fabs(x), ay = fabs(y);
    float mx = fmax(ax, ay);
    if (mx < 1e-12f) return 0.0f;
    float a = fmin(ax, ay) / mx;
    float s = a * a;
    float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
    if (ay > ax) r = 1.5707963f - r;
    if (x < 0) r = 3.1415927f - r;
    return y < 0 ? -r : r;
}

__kernel void mix2(__global const float* t, __global const float* Pk, int sh, int sw, __global const float* wob,
                   __global const float* jit, __global const int* ang, int cg, int cb, float pp,
                   __global const float* blue, __global const float* blobs, __global const float* Cn, __global const float* Sn,
                   __global const float* ppf, int w, int h, __global float* tc, __global float* thr, __global float* dm){
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int p = y * w + x;
    float inv = 1.0f / pp;
    float fx = fmin(fmax((x + 0.5f) * ((float)sw / w) - 0.5f, 0.0f), sw - 1.0f);
    int i0 = (int)fx; int i1 = min(i0 + 1, sw - 1); float dx = fx - i0;
    float fy = fmin(fmax((y + 0.5f) * ((float)sh / h) - 0.5f, 0.0f), sh - 1.0f);
    int y0 = (int)fy; int y1 = min(y0 + 1, sh - 1); float dy = fy - y0;
    float w00 = (1 - dx) * (1 - dy), w01 = dx * (1 - dy), w10 = (1 - dx) * dy, w11 = dx * dy;
#define PK(c) (Pk[(y0*sw+i0)*4+c]*w00 + Pk[(y0*sw+i1)*4+c]*w01 + Pk[(y1*sw+i0)*4+c]*w10 + Pk[(y1*sw+i1)*4+c]*w11)
    float a_d = PK(0), a_s = PK(1), c2 = PK(2), s2 = PK(3);
    float a_g = 1.0f - a_d - a_s;
    float tt = t[p];
    int qd = (int)(fatan2(s2, c2) * 1.9098593f + 12.5f);
    int q = ((qd + ang[p]) % 12 + 12) % 12;
    float c = Cn[q], sn = Sn[q];
    float u0 = y * c - x * sn, v0 = y * sn + x * c;
    float wb = wob[p], jt = jit[p];
    float u = (u0 + wb) * inv;
    u -= (int)(u + 1000.0f) - 1000;
    float thr1 = fabs(2.0f * u - 1.0f) + jt;
    float u2 = (0.5f * u0 - 0.8660254f * v0 + wb * 0.7f + 2.0f) * inv;
    u2 -= (int)(u2 + 1000.0f) - 1000;
    float thr2 = fabs(2.0f * u2 - 1.0f) + jt * 0.7f;
    float wd = clampf((0.4f - tt) * 5.0f, 0.0f, 1.0f);
    float ta = tt + (sqrt(fmax(tt, 0.0f)) - tt) * wd;
    float tg = tl(blue, y, x, cg), td = tl(blobs, y, x, cb);
    float th = a_d * td + a_s * fmax(thr1, thr2 * wd) + a_g * tg;
    float te = tt + a_s * (ta - tt);
    float pur = a_d * a_d + a_s * a_s + a_g * a_g;
    float z = clampf((pur - 0.6f) / 0.4f, 0.0f, 1.0f);
    z = z * z * (3.0f - 2.0f * z);
    float cl = clampf(te, 0.0f, 1.0f);
    float r = z * te + (1.0f - z) * (0.5f + 0.2887f * sqrt(pur) * ppf[(int)(cl * 1024.0f + 0.5f)]);
    if (tt >= 1.0f) r = 2.0f; else if (tt <= 0.0f) r = -1.0f;
    tc[p] = r; thr[p] = th; dm[p] = (r > th ? 1.0f : 0.0f) - cl;
}

__kernel void pass2(__global const float* tc, __global const float* thr, __global const float* t, __global const float* es,
                    int sh, int sw, float amt, int w, int h, __global uchar* out){
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int p = y * w + x;
    float tt = t[p];
    if (tt >= 1.0f) out[p] = 1;
    else if (tt <= 0.0f) out[p] = 0;
    else out[p] = (tc[p] - amt * bil(es, sh, sw, (y + 0.5f) * ((float)sh / h) - 0.5f, (x + 0.5f) * ((float)sw / w) - 0.5f)) > thr[p] ? 1 : 0;
}
"""

SRC_HASH = hashlib.sha1(_SRC.encode()).hexdigest()[:12]
_BUILD = ["-cl-std=CL1.2"]

_lock = threading.RLock()
_dev = None            # Gpu або False (немає/вимкнено)
_state = {"mode": None, "reason": "", "use": None}
_allowed = True        # False у процесах пулу без «місця» для відеокарти: вони рахують на CPU


def mode() -> str:
    m = (os.environ.get("INKREEL_GPU") or "auto").strip().lower()
    return m if m in ("auto", "off", "cpu", "on") else "auto"


def _cache_file() -> str:
    from .log import app_dir
    return os.path.join(app_dir(), "gpu.json")


def list_devices(allow_cpu: bool = False):
    """[(платформа, пристрій)] - відеокарти першими: дискретні (з власною пам'яттю) перед вбудованими"""
    out = []
    if not _import_cl():
        return out
    try:
        plats = cl.get_platforms()
    except Exception:  # noqa: BLE001
        return out
    for p in plats:
        try:
            devs = p.get_devices()
        except Exception:  # noqa: BLE001
            continue
        for d in devs:
            gpu = bool(d.type & cl.device_type.GPU)
            if gpu or (allow_cpu and bool(d.type & cl.device_type.CPU)):
                try:
                    unified = bool(d.host_unified_memory)
                except Exception:  # noqa: BLE001
                    unified = False
                if not d.available or not d.compiler_available:
                    continue
                out.append((0 if gpu and not unified else 1 if gpu else 2, -d.global_mem_size, p, d))
    out.sort(key=lambda r: r[:2])
    return [(p, d) for _, _, p, d in out]


def lanes_for(mem_bytes: int, m: str) -> int:
    """скільки викликів можуть іти на відеокарті одночасно: кожна смуга тримає свої буфери кадру (на 1440p це сотні МБ), тож за обсягом
    пам'яті: 8 ГБ і більше - 4, від 4 - 3, від 2 - 2, менше - 1. INKREEL_GPU_LANES=1..8 задає вручну (1 - як раніше, по черзі)"""
    try:
        v = int(os.environ.get("INKREEL_GPU_LANES", "0"))
        if 1 <= v <= 8:
            return v
    except ValueError:
        pass
    if m == "cpu":
        return 2
    gb = mem_bytes / (1 << 30)
    return 4 if gb >= 7.5 else 3 if gb >= 3.5 else 2 if gb >= 1.8 else 1


class Gpu:
    """Контекст, програма й таблиці одного пристрою + пул «смуг» (черга команд і буфери на кожну); викликається з потоків одного процесу"""

    def __init__(self, plat, dev):
        _import_cl()
        self.dev = dev
        self.name = dev.name.strip()
        self.vendor = dev.vendor.strip()
        self.driver = dev.driver_version.strip()
        self.mem = int(dev.global_mem_size)
        self.ctx = cl.Context([dev])
        self.q = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, _SRC).build(options=_BUILD)
        self.k = {n: getattr(self.prg, n) for n in (
            "render", "fb", "blk_u8", "blk_f32", "tone", "pca", "pl", "fin", "tdl", "select_k", "ink_k", "mix2", "pass2")}
        self._bufs = {}
        self._tabs = {}
        self.lock = threading.RLock()
        # «смуги»: кожна має власну чергу команд, власні ядра (cl.Kernel не потокобезпечне) й буфери; контекст, програма й таблиці -
        # спільні. Раніше всі потоки господаря ходили в одну чергу під одним замком: завантаження, ядро й вивантаження кожного
        # виклику йшли строго по одному, поки процесор простоював, - на графіку це синхронні провали навантаження ЦП й «пилка» GPU
        self.root = self
        self._free: "queue.LifoQueue[Gpu]" = queue.LifoQueue()
        self._made = 0
        self._lane_lock = threading.Lock()
        self.lanes_max = lanes_for(self.mem, mode())
        self.lane_wait = 0.0                      # скільки секунд потоки разом чекали вільної смуги (для діагностики)
        self.lane_calls = 0

    def _spawn_lane(self) -> "Gpu":
        ln = object.__new__(Gpu)
        ln.dev, ln.name, ln.vendor, ln.driver, ln.mem = self.dev, self.name, self.vendor, self.driver, self.mem
        ln.ctx, ln.prg = self.ctx, self.prg
        ln.q = cl.CommandQueue(self.ctx)
        ln.k = {n: cl.Kernel(self.prg, n) for n in self.k}               # власні об'єкти ядер: set_arg+enqueue не потокобезпечні
        ln._bufs = {}
        ln._tabs = self._tabs                                       # таблиці лише читаються після створення
        ln.lock = threading.RLock()
        ln.root = self
        return ln

    def lane(self) -> "Gpu":
        """вільна смуга (створюється, доки їх менше за ліміт; інакше чекаємо звільнення). Повернути: release()"""
        root = self.root
        try:
            ln = root._free.get_nowait()
        except queue.Empty:
            ln = None
            with root._lane_lock:
                if root._made < root.lanes_max:
                    ln = root._spawn_lane()
                    root._made += 1
            if ln is None:
                t0 = time.perf_counter()
                ln = root._free.get()
                root.lane_wait += time.perf_counter() - t0
        root.lane_calls += 1
        return ln

    def release(self, ln: "Gpu"):
        self.root._free.put(ln)

    # ---------------- буфери
    def buf(self, tag, nbytes):
        nbytes = max(4, int(nbytes))
        b = self._bufs.get(tag)
        if b is None or b.size < nbytes:
            b = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, nbytes)
            self._bufs[tag] = b
        return b

    def trim(self, force=False):
        """великі буфери кадру не тримаємо між викликами: відеокартою ділиться багато процесів, а кадр 4K - це ≈420 МБ на процес.
        Контекст лишається (≈260 МБ), буфери виділяються наново за мікросекунди"""
        if force or sum(b.size for b in self._bufs.values()) > LEAN_BYTES:
            self.q.finish()
            self._bufs.clear()

    def up(self, tag, a):
        a = np.ascontiguousarray(a)
        b = self.buf(tag, a.nbytes)
        cl.enqueue_copy(self.q, b, a, is_blocking=False)
        return b

    def _wait(self, ev):
        """чекати подію, не «крутячи» ядро ЦП: при блокуючому читанні драйвер NVIDIA тримає ядро на 100%, поки відеокарта віддана іншим
        процесам пулу (контексти різних процесів вона обслуговує по черзі) - 8 таких процесів з'їдали по ядру кожен, нічого не рахуючи.
        Спершу опитуємо (малі ядра закінчуються за долі мілісекунди), потім спимо короткими кроками"""
        self.q.flush()
        t0 = time.perf_counter()
        while True:
            st = ev.get_info(cl.event_info.COMMAND_EXECUTION_STATUS)
            if st == cl.command_execution_status.COMPLETE:
                return
            if st < 0:
                ev.wait()                                  # помилка виконання: виняток із пояснення драйвера
                return
            if time.perf_counter() - t0 > 0.0003:
                time.sleep(0.0004)

    def down(self, b, shape, dtype):
        out = np.empty(shape, dtype)
        self._wait(cl.enqueue_copy(self.q, out, b, is_blocking=False))
        return out

    def tab(self, key, make):
        """таблиці, що не змінюються між кадрами, живуть на пристрої"""
        b = self._tabs.get(key)
        if b is None:
            a = np.ascontiguousarray(make())
            b = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
            self._tabs[key] = b
        return b

    @staticmethod
    def _g(w, h):
        return (int(w), int(h))

    # ---------------- ядра: заміни numba-версій
    def render(self, t, Fk, noise, M, sd, phi, Tg, Td, ppf, o1y, o1x, o2y, o2x, Dk=None, bn=None, bl=None, cg=1, cb=1):
        h, w = t.shape
        sh, sw = Fk.shape[:2]
        tb, fb_ = self.up("t", t.astype(np.float32, copy=False)), self.up("Fk", Fk.astype(np.float32, copy=False))
        nz = self.tab(("noise", noise.__array_interface__["data"][0], noise.shape), lambda: noise.astype(np.float32))
        sdb = self.tab(("sd", sd.__array_interface__["data"][0], sd.shape), lambda: sd.astype(np.float32))
        phib = self.tab(("phi", phi.__array_interface__["data"][0]), lambda: phi.astype(np.float32))
        ppfb = self.tab(("ppf", ppf.__array_interface__["data"][0]), lambda: ppf.astype(np.float32))
        tgb = self.tab(("Tg", Tg.__array_interface__["data"][0], Tg.shape), lambda: Tg.astype(np.float32))
        tdb = self.tab(("Td", Td.__array_interface__["data"][0], Td.shape), lambda: Td.astype(np.float32))
        tc, thr, cl_ = (self.buf(n, h * w * 4) for n in ("tc", "thr", "cl"))
        usd = Dk is not None and Dk.shape[0] > 1
        dkb = self.up("Dk", np.ascontiguousarray(Dk, np.float32)) if usd else self.buf("Dk", 4)
        bnb = self.tab(("bn", bn.__array_interface__["data"][0]), lambda: bn.astype(np.float32)) if usd else self.buf("bn0", 4)
        blb = self.tab(("bl", bl.__array_interface__["data"][0]), lambda: bl.astype(np.float32)) if usd else self.buf("bl0", 4)
        self.k["render"](self.q, self._g(w, h), None, tb, fb_, nz, np.int32(M), sdb, np.int32(sd.shape[0]), phib, tgb, tdb, ppfb,
                         np.int32(o1y), np.int32(o1x), np.int32(o2y), np.int32(o2x), np.int32(w), np.int32(h),
                         np.int32(sw), np.int32(sh), dkb, np.int32(1 if usd else 0), bnb, blb, np.int32(cg), np.int32(cb), tc, thr, cl_)
        return tc, thr, cl_

    def fb(self, tc, thr, cum, prev, hy, w, h):
        """-> маска uint8 (h, w); tc/thr - буфери пристрою з render; cum - мала сітка поправки (numpy)"""
        cum = np.ascontiguousarray(cum, np.float32)
        cb = self.up("cum", cum)
        use = prev is not None and prev.shape == (h, w)
        pb = self.up("prev", prev) if use else self.buf("prev", 4)
        out = self.buf("fbout", h * w)
        self.k["fb"](self.q, self._g(w, h), None, tc, thr, cb, np.int32(cum.shape[1]), np.int32(cum.shape[0]), pb,
                     np.int32(1 if use else 0), np.float32(hy), np.int32(w), np.int32(h), out)
        return out

    def blocks_u8(self, buf, f, h, w):
        bh, bw = -(-h // f), -(-w // f)
        acc = self.buf("acc", bh * bw * 4)
        self.k["blk_u8"](self.q, self._g(bw, bh), None, buf, np.int32(f), np.int32(w), np.int32(h), np.int32(bw), np.int32(bh), acc)
        return self.down(acc, (bh, bw), np.int32).astype(np.float64)

    def blocks_f32(self, buf, f, h, w):
        bh, bw = -(-h // f), -(-w // f)
        acc = self.buf("accf", bh * bw * 4)
        self.k["blk_f32"](self.q, self._g(bw, bh), None, buf, np.int32(f), np.int32(w), np.int32(h), np.int32(bw), np.int32(bh), acc)
        return self.down(acc, (bh, bw), np.float32).astype(np.float64)

    def tone(self, L, base, loc, sg, prm):
        h, w = L.shape
        lo, rngb, gam, gain, rel, abs_w, lc, curve = (float(v) for v in prm[:8])
        return self._tone3(L, base, loc, sg, lo, rngb, gam, gain, rel, abs_w, lc, curve, h, w)

    def _tone3(self, L, base, loc, sg, lo, rngb, gam, gain, rel, abs_w, lc, curve, h, w):
        NB, NT = 2048, 4096
        b0 = np.arange(NB + 1) / NB
        bl_a = (b0 ** gam).astype(np.float32)
        wl_a = (0.6 + 1.6 * np.sqrt(bl_a * (1.0 - bl_a))).astype(np.float32)
        th_a = (0.5 + 0.5 * np.tanh(curve * (np.arange(NT + 1) / NT * 2.0 - 1.0)) / np.tanh(curve)).astype(np.float32)
        blb = self.tab(("bl", gam), lambda: bl_a)
        wlb = self.tab(("wl", gam), lambda: wl_a)
        thb = self.tab(("th", curve), lambda: th_a)
        n = h * w
        o = self.buf("o", n * 4)
        self.k["tone"](self.q, (n,), None, self.up("a0", L.astype(np.float32, copy=False)), self.up("a1", base.astype(np.float32, copy=False)),
                       self.up("a2", loc.astype(np.float32, copy=False)), self.up("a3", sg.astype(np.float32, copy=False)),
                       blb, wlb, thb, np.float32(lo), np.float32(rngb), np.float32(gain), np.float32(rel), np.float32(abs_w),
                       np.float32(lc), np.int32(NB), np.int32(NT), np.int32(n), o)
        return self.down(o, (h, w), np.float32)

    def pca(self, lab, vL, va, vb, off, scale, Lo, mix):
        h, w = lab.shape[:2]
        n = h * w
        o = self.buf("o", n * 4)
        self.k["pca"](self.q, (n,), None, self.up("a0", np.ascontiguousarray(lab, np.uint8)), np.float32(vL), np.float32(va),
                      np.float32(vb), np.float32(off), np.float32(scale), np.float32(Lo), np.float32(mix), np.int32(n), o)
        return self.down(o, (h, w), np.float32)

    def pl(self, L, hp2, e0, e1):
        h, w = L.shape
        n = h * w
        G, LG = self.buf("o", n * 4), self.buf("o2", n * 4)
        self.k["pl"](self.q, (n,), None, self.up("a0", L), self.up("a1", hp2), np.float32(e0), np.float32(e1), np.int32(n), G, LG)
        return self.down(G, (h, w), np.float32), self.down(LG, (h, w), np.float32)

    def fin(self, L, T, F, t0, t1):
        h, w = L.shape
        n = h * w
        o = self.buf("o", n * 4)
        self.k["fin"](self.q, (n,), None, self.up("a0", L), self.up("a1", T), self.up("a2", F), np.float32(t0), np.float32(t1), np.int32(n), o)
        return self.down(o, (h, w), np.float32)

    def tdl(self, L, Lp, bg, rho, g0, g1):
        h, w = L.shape
        n = h * w
        o = self.buf("o", n * 4)
        self.k["tdl"](self.q, (n,), None, self.up("a0", L), self.up("a1", Lp), self.up("a2", bg), np.float32(rho), np.float32(g0),
                      np.float32(g1), np.int32(n), o)
        return self.down(o, (h, w), np.float32)

    def select(self, mS, mG, pg, Tr):
        h, w = mS.shape
        o = self.buf("o", h * w)
        self.k["select_k"](self.q, self._g(w, h), None, self.up("a0", mS), self.up("a1", mG), self.up("a2", pg.astype(np.float32, copy=False)),
                           np.int32(pg.shape[0]), np.int32(pg.shape[1]), self.up("a3", Tr.astype(np.float32, copy=False)),
                           np.int32(w), np.int32(h), o)
        return self.down(o, (h, w), np.uint8)

    def ink(self, m, dog, thr):
        h, w = m.shape
        n = h * w
        o = self.buf("o", n)
        self.k["ink_k"](self.q, (n,), None, self.up("a0", m), self.up("a1", dog.astype(np.float32, copy=False)), np.float32(thr), np.int32(n), o)
        return self.down(o, (h, w), np.uint8)

    def mix2(self, t, Pk, wob, jit, ang, cg, cb, pp, blue, blobs, Cn, Sn, ppf):
        h, w = t.shape
        sh, sw = Pk.shape[:2]
        n = h * w
        tc, thr, dm = self.buf("o", n * 4), self.buf("o2", n * 4), self.buf("o3", n * 4)
        tb = lambda nm, a: self.tab((nm, a.__array_interface__["data"][0], a.shape), lambda: a.astype(np.float32))
        self.k["mix2"](self.q, self._g(w, h), None, self.up("a0", t.astype(np.float32, copy=False)), self.up("a1", Pk.astype(np.float32, copy=False)),
                       np.int32(sh), np.int32(sw), self.up("a2", wob.astype(np.float32, copy=False)), self.up("a3", jit.astype(np.float32, copy=False)),
                       self.up("a4", ang.astype(np.int32, copy=False)), np.int32(cg), np.int32(cb), np.float32(pp), tb("blue", blue),
                       tb("blobs", blobs), tb("Cn", Cn), tb("Sn", Sn), tb("ppf", ppf), np.int32(w), np.int32(h), tc, thr, dm)
        return (self.down(tc, (h, w), np.float32), self.down(thr, (h, w), np.float32), self.down(dm, (h, w), np.float32))

    def pass2(self, tc, thr, t, es, amt):
        h, w = t.shape
        o = self.buf("o", h * w)
        self.k["pass2"](self.q, self._g(w, h), None, self.up("a0", tc), self.up("a1", thr), self.up("a2", t), self.up("a3", es.astype(np.float32, copy=False)),
                        np.int32(es.shape[0]), np.int32(es.shape[1]), np.float32(amt), np.int32(w), np.int32(h), o)
        return self.down(o, (h, w), np.uint8)


# ---------------------------------------------------------------- доступ
def _verdict_file_ok() -> bool:
    """є файл вердикту з ok=true (без імпорту pyopencl: у звичайному запуску без відеокарти це нічого не коштує)"""
    try:
        with open(_cache_file(), encoding="utf-8") as f:
            return bool(json.load(f).get("ok"))
    except Exception:  # noqa: BLE001
        return False


def _make() -> "Gpu | None":
    m = mode()
    if m == "off" or not HAVE_CL:
        _state["reason"] = "вимкнено" if m == "off" else "pyopencl не встановлено"
        return None
    if m == "auto" and not _state.get("probing") and not _verdict_file_ok():
        _state["reason"] = "перевірка відеокарти ще не проходила"
        _state["retry"] = time.time() + 15
        return None
    devs = list_devices(allow_cpu=(m == "cpu"))
    if not devs:
        _state["reason"] = "відеокарти з OpenCL не знайдено"
        return None
    try:
        return Gpu(*devs[0])
    except Exception as e:  # noqa: BLE001
        _state["reason"] = f"не вдалося ініціалізувати: {e}"
        return None


def device() -> "Gpu | None":
    """пристрій цього процесу або None; створюється один раз"""
    global _dev
    if not _allowed:
        return None
    if _dev is False and _state.get("retry") and time.time() > _state["retry"]:
        _state["retry"] = 0
        _dev = None                                      # вердикту не було - дивимось знову (probe міг закінчитись)
    if _dev is None:
        with _lock:
            if _dev is None:
                _dev = _make() or False
    return _dev or None


def disable(reason: str = ""):
    global _dev
    with _lock:
        _dev = False
        _state["reason"] = reason or _state["reason"]


def kernel_ok(name: str) -> bool:
    """ядро name можна використати: GPU є, працює, а probe підтвердив коректність і виграш (режим cpu/on - без перевірки виграшу)"""
    g = device()
    if g is None:
        return False
    if mode() in ("cpu", "on") or _state.get("probing"):
        return True
    use = _state["use"]
    if use is None:
        use = _state["use"] = _load_verdict(g)
    return bool(use.get(name))


ACC = [0.0, 0]                          # [секунд потоків у викликах run (смуга + обмін + ядро), кількість викликів] - лише для журналу
_OOM_MARKS = ("MEM_OBJECT_ALLOCATION_FAILURE", "OUT_OF_HOST_MEMORY")
LEAN_BYTES = 64 << 20                  # буфери понад цей обсяг (від ~1080p) віддаються пристрою після кожного виклику


def run(name: str, fn):
    """fn(gpu) -> результат або None; будь-яка помилка вимикає GPU в процесі й повертає None (виклик дорахує CPU).
    Виняток - нестача відеопам'яті (багато процесів ділять одну відеокарту): цей виклик рахується на CPU, GPU лишається"""
    if not kernel_ok(name):
        return None
    g = device()
    t_run = time.perf_counter()
    try:
        ln = g.lane()
        ok = False
        try:
            r = fn(ln)
            ok = True
            return r
        finally:
            try:
                ln.trim(not ok)                    # після помилки: дочекатись черги й віддати буфери (вони можуть бути в польоті)
            except Exception:  # noqa: BLE001
                pass
            g.release(ln)
            ACC[0] += time.perf_counter() - t_run
            ACC[1] += 1
    except Exception as e:  # noqa: BLE001
        if any(m in str(e) for m in _OOM_MARKS):
            _state["oom"] = _state.get("oom", 0) + 1
            if _state["oom"] < 200:
                return None
        disable(f"помилка під час роботи: {e}")
        try:
            from .log import get_logger
            get_logger().warning("GPU вимкнено: %s", e)
        except Exception:  # noqa: BLE001
            pass
        return None


def info() -> dict:
    g = device()
    use = _state["use"] if _state["use"] is not None else (_load_verdict(g) if g else {})
    return dict(ok=g is not None, name=g.name if g else "", vendor=g.vendor if g else "", driver=g.driver if g else "",
                reason=_state["reason"], kernels=[k for k in KERNELS if use.get(k)], mode=mode())


# ---------------------------------------------------------------- probe (перевірка й заміри)
def _key(g) -> str:
    return f"{g.vendor}|{g.name}|{g.driver}|{SRC_HASH}"


def _key_dev(d) -> str:
    return f"{d.vendor.strip()}|{d.name.strip()}|{d.driver_version.strip()}|{SRC_HASH}"


def set_allowed(v: bool):
    global _allowed
    _allowed = bool(v)


def worker_slots(px: int = 1280 * 720) -> int:
    """скільки процесів пулу отримають відеокарту (у кожного свій контекст і буфери, тож їх небагато). 0 - пулові процеси
    рахують на CPU. Рішення - з готового вердикту probe, без створення контексту в головному процесі"""
    m = mode()
    if m == "off" or not HAVE_CL:
        return 0
    devs = list_devices(allow_cpu=(m == "cpu"))
    if not devs:
        return 0
    d = devs[0][1]
    cpu = os.cpu_count() or 2
    if m in ("cpu", "on"):
        return 2
    try:
        with open(_cache_file(), encoding="utf-8") as f:
            v = json.load(f)
        if v.get("key") != _key_dev(d) or not v.get("ok") or not v.get("use", {}).get("draw"):
            return 0
    except Exception:  # noqa: BLE001
        return 0
    vram = d.global_mem_size / (1 << 20) * 0.55          # решту беруть робочий стіл, браузер, відеокодек
    bufs = 55 * px / 1e6                                  # МБ буферів кадру на процес (4K ≈ 420 МБ)
    if bufs > LEAN_BYTES / (1 << 20):   # великі кадри: буфери віддаються після кожного виклику (див. Gpu.trim)
        by_mem = int((vram - 3 * bufs) // (260 + 10 * px / 1e6))     # у кожного процесу контекст ≈260 МБ й таблиці; буфери одночасно виділяють лише кілька
        cap = 12 if px >= 2_500_000 else 8
    else:
        by_mem = int(vram // (260 + bufs))
        cap = 8
    return max(0, min(cap, by_mem, max(1, cpu // 2 if px >= 2_500_000 else cpu // 3)))


_slots_cache = [0.0, 0]


def slots_cached() -> int:
    """worker_slots() із кешем на кілька секунд: для напису в інтерфейсі, що малюється часто"""
    now = time.time()
    if now - _slots_cache[0] > 5:
        try:
            _slots_cache[:] = [now, worker_slots()]
        except Exception:  # noqa: BLE001
            _slots_cache[:] = [now, 0]
    return _slots_cache[1]


def _load_verdict(g) -> dict:
    try:
        with open(_cache_file(), encoding="utf-8") as f:
            d = json.load(f)
        if d.get("key") == _key(g):
            return {k: bool(v) for k, v in d.get("use", {}).items()}
    except Exception:  # noqa: BLE001
        pass
    return {}


def probe_cached() -> dict | None:
    """готовий вердикт для пристрою за замовчуванням або None, якщо перевірки ще не було"""
    g = device()
    if g is None:
        return None
    try:
        with open(_cache_file(), encoding="utf-8") as f:
            d = json.load(f)
        return d if d.get("key") == _key(g) else None
    except Exception:  # noqa: BLE001
        return None


def start_probe_background():
    """якщо відеокарта є, а перевірки цього пристрою ще не було - запускає її в окремому процесі без очікування"""
    if not HAVE_CL or mode() != "auto":
        return
    try:
        devs = list_devices()
        if not devs:
            return
        try:
            with open(_cache_file(), encoding="utf-8") as f:
                v = json.load(f)
            good = v.get("ok") and v.get("use", {}).get("draw")
            # вердикт без «draw» міг бути випадково поганим замором (зайнята відеокарта): перевіряємо знову не частіше за раз на 6 годин
            if v.get("key") == _key_dev(devs[0][1]) and (good or time.time() - v.get("ts", 0) < 6 * 3600):
                return
        except Exception:  # noqa: BLE001
            pass
        if getattr(sys, "frozen", False):
            args = [sys.executable, "--gpu-probe"]
        else:
            args = [sys.executable, "-m", "benvid.gpu", "--probe"]
        flags = 0x08000000 if os.name == "nt" else 0
        subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    except Exception:  # noqa: BLE001
        pass


def _bench(fn, reps=5):
    """мс: найкращий із кількох прогонів після розгону (відеокарта в режимі енергозбереження спершу працює на низькій частоті,
    а випадкова затримка не повинна ні вмикати, ні вимикати ядро надовго)"""
    t_end = time.perf_counter() + 0.25
    fn()
    while time.perf_counter() < t_end:
        fn()
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1000


def run_probe(w: int = 1280, h: int = 720, save: bool = True) -> dict:
    """перевіряє кожне ядро проти numba на синтетичних даних і міряє обидва; пише вердикт у gpu.json"""
    res = dict(ok=False, key="", name="", use={}, ms={}, err={}, reason="")
    global _dev
    _state["probing"] = True                  # перевірка створює пристрій і без готового вердикту
    if _dev is False or _dev is None:
        _dev = None
    try:
        return _run_probe(res, w, h, save)
    finally:
        _state["probing"] = False
        _dev, _state["use"] = None, None      # далі - звичайний шлях, з новим вердиктом


def _run_probe(res, w, h, save):
    from . import core as C, ink5
    g = device()
    if g is None:
        res["reason"] = _state["reason"]
        return res
    res.update(key=_key(g), name=g.name)
    if C._nb is None or not ink5.HAVE_NUMBA:
        res["reason"] = "numba немає: порівнювати нема з чим"
        return res
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.clip(0.5 + 0.35 * np.sin(xx / 37.0) * np.cos(yy / 29.0) + rng.normal(0, 0.05, (h, w)), -0.02, 1.02).astype(np.float32)
    t[:40] = 1.01
    t[-30:] = -0.01
    use, ms, err = {}, {}, {}

    def check(name, cpu, gpu_, cmp):
        try:
            a = cpu()
            b = g_call(gpu_)
            ok, msg = cmp(a, b)
            if not ok:
                err[name] = msg
                use[name] = False
                return
            tc_ = _bench(cpu)
            tg_ = _bench(lambda: g_call(gpu_))
            ms[name] = (round(tc_, 2), round(tg_, 2))
            use[name] = tg_ < tc_ * 0.85
        except Exception as e:  # noqa: BLE001
            err[name] = str(e)[:200]
            use[name] = False

    def g_call(f):
        with g.lock:
            return f(g)

    def close(tol, frac=0.995):
        def c(a, b):
            a = np.asarray(a, np.float64)
            b = np.asarray(b, np.float64)
            if not np.isfinite(b).all():
                return False, "NaN у результаті"
            f = float((np.abs(a - b) <= tol).mean())
            return f >= frac, f"збіг {f:.4f}"
        return c

    def same(frac):
        def c(a, b):
            f = float((np.asarray(a) == np.asarray(b)).mean())
            return f >= frac, f"збіг {f:.4f}"
        return c

    # --- render + fb
    sh, sw = h // 3, w // 3
    Fk = np.stack([np.cos(xx[:sh, :sw]), np.sin(xx[:sh, :sw]), 6 + 4 * np.sin(yy[:sh, :sw] / 9), 0.7 * np.ones((sh, sw)), 0.2 * np.ones((sh, sw))], -1).astype(np.float32)
    Fk[..., 0] = np.cos(xx[:sh, :sw] / 20)
    Fk[..., 1] = np.sin(xx[:sh, :sw] / 20)
    T = ink5._TILE
    noise = ink5._ring_tile(T, 3.6, 0.3, 5)
    sd = ink5._sigma_tab(T, 3.6, 0.3, 5)
    Tg, Td, Tr = ink5._static_fields(w, h, 1, 1)
    f_ = 4
    prev = (rng.random((h, w)) > 0.5).astype(np.uint8)
    dargs = (t, Fk, noise, T, sd, Tg, Td, f_, 3, 0.7, 1.0, prev, 0.05)
    Dk_ = np.stack([30 * np.sin(yy[:sh, :sw] / 17) - 3.3, -20 * np.cos(xx[:sh, :sw] / 23) + 1.7], -1).astype(np.float32)     # зсув візерунка разом із вмістом
    check("draw", lambda: np.stack([ink5._draw_cpu(*dargs, [None]), ink5._draw_cpu(*dargs, [None], Dk_, 1, 1)]),
          lambda G: np.stack([ink5._draw_gpu(G, *dargs, [None]), ink5._draw_gpu(G, *dargs, [None], Dk_, 1, 1)]), same(0.985))
    # --- tone / pca / pl / fin / tdl
    L = (60 + 120 * np.clip(0.5 + 0.4 * np.sin(xx / 51.0) * np.cos(yy / 33.0), 0, 1) + rng.normal(0, 2, (h, w))).astype(np.float32)
    base = (L * 0.8 + 20).astype(np.float32)
    loc = (L * 0.9 + 10).astype(np.float32)
    sgm = np.full((h, w), 5.0, np.float32)
    prm = np.array([5.0, 230.0, 0.9, 1.1, 0.35, 0.2, 0.15, 3.0], np.float64)
    check("tone", lambda: C._tone_nb(L, base, loc, sgm, prm), lambda G: G.tone(L, base, loc, sgm, prm), close(2e-3, 0.99))
    lab = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    pa = (0.9, 0.1, -0.2, 40.0, 1.1, 20.0, 0.5)
    check("pca", lambda: C._pca_nb(lab, *pa), lambda G: G.pca(lab, *pa), close(0.05, 0.995))
    hp2 = np.abs(rng.normal(0, 0.5, (h, w))).astype(np.float32)
    Gc, LGc = np.empty_like(L), np.empty_like(L)

    def pl_cpu():
        C._pl_nb(L, hp2, 0.25, 0.7, Gc, LGc)
        return np.stack([Gc, LGc])
    check("pl", pl_cpu, lambda G: np.stack(G.pl(L, hp2, 0.25, 0.7)), close(1e-3, 0.999))
    Tt = (L + rng.normal(0, 2, (h, w))).astype(np.float32)
    Ff = np.ones((h, w), np.float32) * 0.8
    oc = np.empty_like(L)

    def fin_cpu():
        C._fin_nb(L, Tt, Ff, 1.5, 4.5, oc)
        return oc
    check("fin", fin_cpu, lambda G: G.fin(L, Tt, Ff, 1.5, 4.5), close(1e-3, 0.999))
    Lp = (L + rng.normal(0, 1.5, (h, w))).astype(np.float32)
    bg = np.ones((h, w), np.float32)
    ot = np.empty_like(L)

    def tdl_cpu():
        ink5._tdl_nb(L, Lp, bg, 0.85, 1.5, 4.5, ot)
        return ot
    check("tdl", tdl_cpu, lambda G: G.tdl(L, Lp, bg, 0.85, 1.5, 4.5), close(1e-3, 0.999))
    # --- select / ink
    mS = prev
    mG = (rng.random((h, w)) > 0.4).astype(np.uint8)
    pg = rng.random((h // 4, w // 4)).astype(np.float32)
    check("select", lambda: ink5._select_nb(mS, mG, pg, Tr), lambda G: G.select(mS, mG, pg, Tr), same(0.999))
    dog = rng.normal(0, 0.5, (h, w)).astype(np.float32)
    check("ink", lambda: C._ink_nb(mS, dog, 0.8), lambda G: G.ink(mS, dog, 0.8), same(0.9999))
    # --- mix2 + pass2
    Pk = np.stack([0.2 * np.ones((sh, sw)), 0.6 * np.ones((sh, sw)), np.cos(xx[:sh, :sw] / 20), np.sin(xx[:sh, :sw] / 20)], -1).astype(np.float32)
    wob, jit, ang = C._hand(w, h, 1.0, 3.6, 1.0)
    margs = (t, Pk, wob, jit, ang, 1, 1, 3.6, C._blue(), C._blobs(), C._C, C._S, C._ppf_lut())
    check("mix2", lambda: np.stack(C._mix2_nb(*margs)), lambda G: np.stack(G.mix2(*margs)), close(5e-3, 0.99))
    es = rng.normal(0, 0.05, (h // 4, w // 4)).astype(np.float32)
    tc2, thr2, _ = C._mix2_nb(*margs)
    check("pass2", lambda: C._pass2_nb(tc2, thr2, t, es, 0.8), lambda G: G.pass2(tc2, thr2, t, es, 0.8), same(0.999))

    res.update(ts=time.time(), ok=any(use.values()), use={k: bool(use.get(k)) for k in KERNELS}, ms=ms, err=err)
    if save:
        try:
            with open(_cache_file(), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
        except Exception:  # noqa: BLE001
            pass
    return res


if __name__ == "__main__":
    if "--probe" in sys.argv:
        r = run_probe()
        print(json.dumps(r, ensure_ascii=False, indent=1))
