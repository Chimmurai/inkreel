"""Навчання швидкої моделі («студента»), що повторює рівень «Ультра» (дистиляція). Дані - з tools/make_dataset.py.

    python tools/train_ink.py cuda                                 # PyTorch бачить відеокарту?
    python tools/train_ink.py info                                 # розмір моделей: параметри, MAC на піксель
    python tools/train_ink.py bench  --preset all --h 1080         # швидкість кожної моделі на відеокарті (мс на кадр, к/с)
    python tools/train_ink.py train  --hours 9                     # навчання (Ctrl+C безпечний; --resume продовжує; --preset auto сам обирає розмір моделі)
    python tools/train_ink.py eval   --run RUN                     # метрики на val (повні кадри)
    python tools/train_ink.py infer  --run RUN --image a.png       # застосувати до картинки (--ref: порівняти з справжнім Ultra)
    python tools/train_ink.py export --run RUN                     # ONNX для ReShade / рушія
    python tools/train_ink.py noise  --w 1920 --h 1080             # текстури шуму під розмір кадру (вхід моделі в ONNX)

Чому модель отримує не лише кадр. Алгоритм - не функція «колір -> маска» в одній точці: растр і штрихи малюються проти ФІКСОВАНИХ
шумових текстур (синій шум 128×128 з випадковим зсувом/дзеркалом плиток, плямки, кільцевий шум для штрихів), які не залежать від
кадру, а лише від координат і висоти. Згорткова мережа не може вгадати їхню фазу, тому ті самі текстури подаються їй на вхід (4 канали),
і вона вчиться «порівнювати тон із шумом», як алгоритм. Глобальне нормування контрасту алгоритм бере з усього кадру (відсотки яскравості,
PCA кольору), тож модель отримує ще й дрібну копію кадру 96×96 і опис масштабу (висота кадру задає крок штрихів).
Усе це на виході (ONNX) - окремі входи; `train_ink.py noise` пише текстури під потрібний розмір кадру."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEF_DATA = r"E:\Inkreel_Dataset"

# ---------------------------------------------------------------- константи алгоритму (збігаються з benvid.core / benvid.ink5; перевіряє tests/test_train_ink.py)
REF_H = 720.0
PERIOD, RING_Q, RING_SEED = 3.6, 0.3, 5          # ink5.P: крок штрихів (px на 720p), «чистота» кільця; seed кільцевого шуму
TILE, TILE_BIG = 1024, 2048                       # ink5._TILE: період кільцевого шуму (для кадрів з довгою стороною > 1600 - вдвічі більший)
RING_OX, RING_OY = 311, 517                       # зсув координат шуму в ядрі малювання
ROLL = (61, 97)                                   # Tr = np.roll(Tg, (61, 97)): поле вибору «штрихи / зерно»
SMALL = 96                                        # сторона дрібної копії кадру (глобальний опис)
NOISE_CH = 4                                      # Tg, Td, Tr, кільцевий шум
META_N = 5
NOISE_MODE = "all"                                # all | tiles (без кільцевого) | none: лише для абляцій; основний режим - all
BUCKETS = (256, 192, 128, 96, 64)                 # розмір вирізки (сторона) за меншою стороною картинки
CROPS_PER = {256: 4, 192: 3, 128: 2, 96: 1, 64: 1}            # вирізок з одного декодованого файлу
BATCH_CROPS = {256: 16, 192: 28, 128: 64, 96: 112, 64: 128}   # вирізок у батчі (приблизно однакова кількість пікселів)

PRESETS = {                                       # (канали на рівнях 1/2, 1/4, 1/8, 1/16), cf - ознак на повній роздільності, hh - голова
    "tiny": dict(c=(12, 24, 48, 64), cf=8, hh=24),
    "fast": dict(c=(16, 32, 64, 96), cf=8, hh=32),
    "base": dict(c=(24, 48, 96, 128), cf=8, hh=32),
    "large": dict(c=(32, 64, 128, 192), cf=12, hh=48),
    # «lean»: без pixel_unshuffle/shuffle (6-вимірні перестановки повільні в DirectML) - їх замінюють згортка 4×4/2 і транспонована 2×2/2; вужча повнороздільна голова
    "tinyl": dict(c=(12, 24, 48, 64), cf=9, hh=16, lean=True),
    "fastl": dict(c=(16, 32, 64, 96), cf=9, hh=16, lean=True),
    "basel": dict(c=(24, 48, 96, 128), cf=9, hh=24, lean=True),
    # «q» (quick): найдорожчі місця за заміром на DirectML - білінійне підвищення + згортка 3×3 на 1/2 роздільності, повні Res-блоки на 1/2 і FiLM голови на повному кадрі.
    # cu - підвищення транспонованою 2×2/2 (без Resize), lite - Res-блоки на 1/2 з однією згорткою, hf=False - без FiLM у повнороздільній голові
    "tinyq": dict(c=(12, 24, 48, 64), cf=9, hh=16, lean=True, cu=True, lite=True, hf=False),
    "fastq": dict(c=(16, 32, 64, 96), cf=9, hh=16, lean=True, cu=True, lite=True, hf=False),
    "baseq": dict(c=(24, 48, 96, 128), cf=9, hh=24, lean=True, cu=True, lite=True, hf=False),
    # «fq_*»: fastq з прибраними шарами - для IRNM1 важлива швидкість, а не повторення дрібної структури Ultra. Ваги з fastq (run ink1) переносяться (`--init-from`),
    # тож вони не з нуля, а дотреновуються. h2=False - без другої 1×1-згортки на повному кадрі, r1=False - без залишкових блоків на 1/2 роздільності,
    # nb=(n3, n4) - скільки залишкових блоків на 1/8 і 1/16 (було 2 і 2)
    "fq_noh2": dict(c=(16, 32, 64, 96), cf=9, hh=16, lean=True, cu=True, lite=True, hf=False, h2=False),
    "fq_nor1": dict(c=(16, 32, 64, 96), cf=9, hh=16, lean=True, cu=True, lite=True, hf=False, r1=False),
    "fq_nb1": dict(c=(16, 32, 64, 96), cf=9, hh=16, lean=True, cu=True, lite=True, hf=False, nb=(1, 1)),
    "fq_a": dict(c=(16, 32, 64, 96), cf=9, hh=16, lean=True, cu=True, lite=True, hf=False, h2=False, r1=False),
    "fq_b": dict(c=(16, 32, 64, 96), cf=9, hh=16, lean=True, cu=True, lite=True, hf=False, h2=False, r1=False, nb=(1, 1)),
    "fq_c": dict(c=(16, 32, 64, 96), cf=9, hh=8, lean=True, cu=True, lite=True, hf=False, h2=False, r1=False, nb=(1, 1)),
    # «ir1» = fq_b + фіксований орієнтований банк фільтрів по кільцевому шуму (ob = кількість напрямків): модель одержує готову «енергію штриха» за напрямками
    # й не мусить сама відкривати, що кільцевий канал - носій штрихів (ink1 його ігнорує). Єдина модель IRNM1; навчається зі штрафом за відсутність штрихів
    "ir1": dict(c=(16, 32, 64, 96), cf=9, hh=16, lean=True, cu=True, lite=True, hf=False, h2=False, r1=False, nb=(1, 1), ob=6),
    "fq_d": dict(c=(16, 32, 48, 64), cf=9, hh=8, lean=True, cu=True, lite=True, hf=False, h2=False, r1=False, nb=(1, 1)),
}


def log(*a, **k):
    print(*a, **k, flush=True)


# ================================================================ шуми й ознаки (numpy, без benvid)
_BLUE = None
_BLOBS = None


def blue() -> np.ndarray:
    global _BLUE
    if _BLUE is None:
        _BLUE = np.load(os.path.join(ROOT, "benvid", "bluenoise128.npy")).astype(np.float32)
    return _BLUE


def blobs() -> np.ndarray:
    """поріг плямок: ранг відстані до найближчого «насіння» (копія core._blobs)"""
    global _BLOBS
    if _BLOBS is None:
        bn = blue()
        n = bn.shape[0]
        seeds = (bn < 0.045).astype(np.uint8)
        dist = cv2.distanceTransform(np.tile(1 - seeds, (3, 3)), cv2.DIST_L2, 5)[n:2 * n, n:2 * n]
        dist = dist + np.random.default_rng(5).random((n, n)).astype(np.float32) * 0.3
        _BLOBS = np.argsort(np.argsort(dist.ravel())).reshape(n, n).astype(np.float32) / (n * n - 1)
    return _BLOBS


def tile_hash(ty, tx):
    h = (ty * 374761393 + tx * 668265263 + 12345) & 0x7FFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0x7FFFFFFF
    return h ^ (h >> 16)


def tile_at(t: np.ndarray, ys: np.ndarray, xs: np.ndarray, cell: int) -> np.ndarray:
    """поріг-плитка (core._tile_build) у довільних абсолютних координатах кадру: ys (n,), xs (m,) -> (n, m)"""
    yy = (ys.astype(np.int64) // cell)[:, None]
    xx = (xs.astype(np.int64) // cell)[None, :]
    hs = tile_hash(yy >> 7, xx >> 7).astype(np.int64)
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
    """(cg, cb): комірка синього шуму і плямок"""
    return 1, max(1, int(pat_k(h / REF_H) + 0.5))


def ring_T(w: int, h: int) -> int:
    return TILE if max(w, h) <= 1600 else TILE_BIG


def make_ring_tile(T: int, period: float, q: float = RING_Q, seed: int = RING_SEED) -> np.ndarray:
    """копія ink5._ring_tile: білий шум, пропущений крізь вузьке кільце частот -> майже періодичні хвилі"""
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


class RingCache:
    """кільцеві плитки на диску (float16, один файл на (T, крок)) + кеш відкритих файлів; ключ - (T, крок×100)"""

    def __init__(self, folder: str, keep: int = 48):
        self.dir, self.keep, self.mem = folder, keep, {}
        os.makedirs(folder, exist_ok=True)

    def path(self, T: int, per: float) -> str:
        return os.path.join(self.dir, f"ring_T{T}_p{int(round(per * 100)):04d}.npy")

    def ensure(self, T: int, per: float) -> str:
        p = self.path(T, per)
        if not os.path.isfile(p):
            tmp = f"{p}.{os.getpid()}.tmp"
            np.save(tmp, make_ring_tile(T, per).astype(np.float16))
            os.replace(tmp if os.path.isfile(tmp) else tmp + ".npy", p)
        return p

    def get(self, T: int, per: float) -> np.ndarray:
        key = (T, int(round(per * 100)))
        a = self.mem.get(key)
        if a is None:
            if len(self.mem) >= self.keep:
                self.mem.pop(next(iter(self.mem)))
            a = np.load(self.ensure(T, per), mmap_mode="r")
            self.mem[key] = a
        return a


def noise_window(w: int, h: int, y0: int, x0: int, ch: int, cw: int, ring: RingCache | None, mode: str | None = None) -> np.ndarray:
    """4 канали шуму для вікна кадру w×h: Tg, Td, Tr (синій шум, плямки, поле вибору; у [-1, 1]) і кільцевий шум (σ = 1) -> (4, ch, cw) float32.
    mode - лише для абляцій (all | tiles | none); за замовчуванням глобальний NOISE_MODE"""
    mode = mode or NOISE_MODE
    ys = np.arange(y0, y0 + ch)
    xs = np.arange(x0, x0 + cw)
    cg, cb = grain_cells(h)
    out = np.empty((NOISE_CH, ch, cw), np.float32)
    out[0] = tile_at(blue(), ys, xs, cg) * 2 - 1
    out[1] = tile_at(blobs(), ys, xs, cb) * 2 - 1
    out[2] = tile_at(blue(), (ys - ROLL[0]) % h, (xs - ROLL[1]) % w, cg) * 2 - 1
    if ring is None or mode != "all":
        out[3] = 0
    else:
        T = ring_T(w, h)
        a = ring.get(T, ring_period(h))
        out[3] = a[np.ix_((ys + RING_OY) % T, (xs + RING_OX) % T)]
    if mode == "none":
        out[:3] = 0
    return out


def meta_of(h: int) -> np.ndarray:
    """опис масштабу кадру: від висоти залежить усе - крок штрихів, плямки, режим малих кадрів (< 290 px)"""
    k = h / REF_H
    kp = pat_k(k)
    return np.array([math.log2(k), math.log2(max(kp, 1e-3)), ring_period(h) / 4.0 - 1.0, grain_cells(h)[1] - 1, 1.0 if h < 290 else 0.0], np.float32)


def small_of(bgr: np.ndarray) -> np.ndarray:
    """дрібна копія кадру для глобального опису: (3, 96, 96) RGB 0..1"""
    s = cv2.resize(bgr, (SMALL, SMALL), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(s[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0


# ================================================================ дані
def imread(path: str, gray: bool = False) -> np.ndarray | None:
    try:
        a = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:  # noqa: BLE001
        return None
    if a is None:
        return None
    if gray:
        return a if a.ndim == 2 else cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    return a[:, :, :3] if a.shape[2] >= 3 else cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2BGR)


def bucket_of(h: int, w: int) -> int:
    m = min(h, w)
    for s in BUCKETS:
        if m >= s:
            return s
    return BUCKETS[-1]


def load_items(root: str, split: str, limit: int = 0, seed: int = 0) -> list[tuple]:
    """[(вхід, мішень, w, h)] зі manifest.jsonl (шляхи від кореня датасету)"""
    out = []
    with open(os.path.join(root, "manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("split") == split:
                out.append((r["inp"], r["ink"], int(r["w"]), int(r["h"])))
    if limit and len(out) > limit:
        random.Random(seed).shuffle(out)
        out = sorted(out[:limit])
    return out


def _ring_job(arg):
    folder, T, per = arg
    RingCache(folder).ensure(T, per)
    return T


def prepare_rings(root: str, items, log_=log, procs: int = 0) -> RingCache:
    """генерує відсутні кільцеві плитки (по одній на пару (T, крок); для повного датасету ≈ 630 штук, ≈ 3,5 ГБ) у кількох процесах"""
    ring = RingCache(os.path.join(root, "_cache"))
    need = sorted({(ring_T(w, h), ring_period(h)) for _, _, w, h in items})
    miss = [(T, p) for T, p in need if not os.path.isfile(ring.path(T, p))]
    if miss:
        n = procs or max(1, min((os.cpu_count() or 2) - 2, 12))
        log_(f"кільцеві шуми: потрібно {len(need)}, генерую {len(miss)} (разово" + (f", {n} процесів" if len(miss) > 8 and n > 1 else "") + ")…")
        t0 = time.time()
        if n > 1 and len(miss) > 8:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(n) as pool:
                jobs = [(ring.dir, T, p) for T, p in sorted(miss, reverse=True)]          # великі (T = 2048) першими - рівномірніше завантаження
                for i, _ in enumerate(pool.imap_unordered(_ring_job, jobs, chunksize=1), 1):
                    if i % 40 == 0 or i == len(miss):
                        log_(f"  {i} з {len(miss)}  ({time.time() - t0:.0f} с)")
        else:
            for i, (T, p) in enumerate(miss, 1):
                ring.ensure(T, p)
                if i % 40 == 0 or i == len(miss):
                    log_(f"  {i} з {len(miss)}  ({time.time() - t0:.0f} с)")
    return ring


class InkData:
    """Dataset: елемент - (індекс, S) -> K вирізок S×S з одного файлу. Імпорт torch - лише тут (щоб numpy-частину можна було тестувати без нього)"""

    def __init__(self, root, items, ring_dir=None, fixed=False, seed=0, mode=None):
        self.root, self.items, self.fixed, self.seed = root, items, fixed, seed
        self.mode = mode or NOISE_MODE          # зберігаємо тут: у процесах Windows (spawn) глобальна змінна модуля не передається
        self.ring = None
        self.ring_dir = ring_dir or os.path.join(root, "_cache")

    def __len__(self):
        return len(self.items)

    def _rng(self, i, S):
        if self.fixed:
            return np.random.default_rng([self.seed, i, S])
        return np.random.default_rng()

    def __getitem__(self, key):
        import torch
        i, S = key
        if self.ring is None:
            self.ring = RingCache(self.ring_dir)
        rin, rink, w0, h0 = self.items[i]
        img = imread(os.path.join(self.root, rin))
        msk = imread(os.path.join(self.root, rink), gray=True)
        if img is None or msk is None or img.shape[:2] != msk.shape[:2]:
            raise RuntimeError(f"не прочиталась пара {rin}")
        h, w = img.shape[:2]
        K = CROPS_PER[S] if not self.fixed else 1
        rng = self._rng(i, S)
        small = small_of(img)
        meta = meta_of(h)
        xs, ns, ts, vs = [], [], [], []
        for _ in range(K):
            ch, cw = min(S, h), min(S, w)
            y0 = int(rng.integers(0, h - ch + 1))
            x0 = int(rng.integers(0, w - cw + 1))
            x = np.zeros((3, S, S), np.uint8)
            n = np.zeros((NOISE_CH, S, S), np.float16)
            t = np.zeros((1, S, S), np.uint8)
            v = np.zeros((1, S, S), np.uint8)
            x[:, :ch, :cw] = img[y0:y0 + ch, x0:x0 + cw, ::-1].transpose(2, 0, 1)
            n[:, :ch, :cw] = noise_window(w, h, y0, x0, ch, cw, self.ring, self.mode)
            t[0, :ch, :cw] = msk[y0:y0 + ch, x0:x0 + cw] > 127
            v[0, :ch, :cw] = 1
            xs.append(x), ns.append(n), ts.append(t), vs.append(v)
        f = torch.from_numpy
        return dict(x=f(np.stack(xs)), n=f(np.stack(ns)), t=f(np.stack(ts)), v=f(np.stack(vs)),
                    small=f(np.repeat(small[None], K, 0)).half(), meta=f(np.repeat(meta[None], K, 0)))


def collate(items):
    import torch
    return {k: torch.cat([it[k] for it in items], 0) for k in items[0]}


class BucketSampler:
    """батчі з однаковим S: файли групуються за розміром вирізки; за епоху кожен файл - раз.
    У навчанні неповний хвіст групи відкидається (однаковий розмір батча), але група, меншa за батч, дає один неповний - щоб на малих наборах не було порожніх епох"""

    def __init__(self, items, shuffle=True, seed=0, limit_s=256, scale=1.0):
        self.scale = scale
        self.groups: dict[int, list[int]] = {}
        for i, (_, _, w, h) in enumerate(items):
            self.groups.setdefault(min(bucket_of(h, w), limit_s), []).append(i)
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def _imgs_per_batch(self, S):
        return max(1, round(BATCH_CROPS[S] * self.scale / CROPS_PER[S]))

    def _n_batches(self, S, size):
        n = self._imgs_per_batch(S)
        if not self.shuffle or size < n:
            return -(-size // n)
        return size // n

    def __len__(self):
        return sum(self._n_batches(S, len(g)) for S, g in self.groups.items())

    def __iter__(self):
        rng = random.Random(self.seed * 100003 + self.epoch)
        self.epoch += 1
        batches = []
        for S, g in self.groups.items():
            g = list(g)
            if self.shuffle:
                rng.shuffle(g)
            n = self._imgs_per_batch(S)
            for j in range(self._n_batches(S, len(g))):
                batches.append([(i, S) for i in g[j * n:(j + 1) * n]])
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)


# ================================================================ модель
def gabor_kernels(K: int = 6, size: int = 8, period: float = 4.0, su: float = 1.6, sv: float = 2.6) -> np.ndarray:
    """K напрямків × 2 фази (cos, sin) -> (2K, 1, size, size) float32; хвиля йде впоперек штриха (u), обвідна витягнута вздовж штриха (v);
    кожне ядро без середнього й з нормою 1 (на білому шумі σ = 1 відповідь має σ ≈ 1)"""
    c = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) - c
    out = []
    for k in range(K):
        th = math.pi * k / K
        u = x * math.cos(th) + y * math.sin(th)
        v = -x * math.sin(th) + y * math.cos(th)
        env = np.exp(-0.5 * ((u / su) ** 2 + (v / sv) ** 2))
        for ph in (np.cos, np.sin):
            g = env * ph(2 * math.pi * u / period)
            g = g - g.mean() if ph is np.cos else g
            out.append(g / np.sqrt((g * g).sum()))
    return np.asarray(out, np.float32)[:, None]


def build_model(preset: str = "fast", **over):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cfg = {**PRESETS[preset], **over}
    c1, c2, c3, c4 = cfg["c"]
    cf, hh = cfg["cf"], cfg["hh"]
    ACT = 0.05

    def act(x):
        return F.leaky_relu(x, ACT)

    def conv(i, o, k=3, s=1):
        m = nn.Conv2d(i, o, k, s, k // 2)
        nn.init.kaiming_normal_(m.weight, a=ACT, nonlinearity="leaky_relu")
        nn.init.zeros_(m.bias)
        return m

    class Res(nn.Module):
        def __init__(s, c):
            super().__init__()
            s.a, s.b = conv(c, c), conv(c, c)
            nn.init.zeros_(s.b.weight)

        def forward(s, x, g=None):
            h = act(s.a(x))
            if g is not None:
                h = torch.addcmul(g[1], h, g[0])              # h·(1 + γ) + β одним проходом по пам'яті (g[0] уже містить 1 + γ)
            return x + s.b(h)

    class ResL(nn.Module):
        """легкий залишковий блок: x + conv(act(x)); одна згортка замість двох, без FiLM"""
        def __init__(s, c):
            super().__init__()
            s.a = conv(c, c)
            nn.init.zeros_(s.a.weight)

        def forward(s, x, g=None):
            return x + s.a(act(x))

    class Ident(nn.Module):
        """прибраний блок: пропускає вхід як є (імена решти шарів лишаються, тож ваги з повнішої моделі підходять)"""
        def forward(s, x, g=None):
            return x

    def up2(i, o):
        m = nn.ConvTranspose2d(i, o, 2, 2)
        nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / i))
        nn.init.zeros_(m.bias)
        return m

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.cfg = cfg
            s.cu, s.hf, s.lite = bool(cfg.get("cu")), bool(cfg.get("hf", True)), bool(cfg.get("lite"))
            R1 = ResL if cfg.get("lite") else Res
            s.lean = bool(cfg.get("lean"))
            s.stem = conv(3 + NOISE_CH, c1, 4, 2) if s.lean else conv(4 * (3 + NOISE_CH), c1)
            if s.lean:
                s.stem.padding = (1, 1)
            n3, n4 = cfg.get("nb", (2, 2))
            s.ob = int(cfg.get("ob", 0))
            if s.ob:                                                 # орієнтований банк: |фільтр(кільцевий шум)|² за K напрямками -> 1×1 у ознаки 1/2
                kb = torch.from_numpy(gabor_kernels(s.ob, 8, 4.0))
                s.oc, s.os = nn.Conv2d(1, s.ob, 8, 2, 3, bias=False), nn.Conv2d(1, s.ob, 8, 2, 3, bias=False)
                with torch.no_grad():
                    s.oc.weight.copy_(kb[0::2])
                    s.os.weight.copy_(kb[1::2])
                s.oc.weight.requires_grad_(False)
                s.os.weight.requires_grad_(False)
                s.op = nn.Conv2d(s.ob, c1, 1)
                nn.init.zeros_(s.op.weight)
                nn.init.zeros_(s.op.bias)
            s.r1 = R1(c1) if cfg.get("r1", True) else Ident()
            s.d1, s.r2 = conv(c1, c2, 3, 2), Res(c2)
            s.d2, s.r3a, s.r3b = conv(c2, c3, 3, 2), Res(c3), (Res(c3) if n3 > 1 else Ident())
            s.d3, s.r4a, s.r4b = conv(c3, c4, 3, 2), Res(c4), (Res(c4) if n4 > 1 else Ident())
            mu = up2 if s.cu else conv
            s.u3, s.q3 = mu(c4, c3), Res(c3)
            s.u2, s.q2 = mu(c3, c2), Res(c2)
            s.u1, s.q1 = mu(c2, c1), (R1(c1) if cfg.get("r1", True) else Ident())
            if s.lean:
                s.o = nn.ConvTranspose2d(c1, cf, 2, 2)
                nn.init.normal_(s.o.weight, 0.0, math.sqrt(2.0 / c1))
                nn.init.zeros_(s.o.bias)
            else:
                s.o = conv(c1, 4 * cf)
            s.h1, s.h3 = conv(cf + 3 + NOISE_CH, hh, 1), conv(hh, 1, 1)
            s.h2 = conv(hh, hh, 1) if cfg.get("h2", True) else None
            nn.init.normal_(s.h3.weight, 0.0, 0.01)                  # старт із логітів ≈ логіт середньої частки білого (0.35), а не з великих випадкових
            nn.init.constant_(s.h3.bias, -0.62)
            s.fl = [c4, c3, c2] + ([] if s.lite else [c1]) + ([hh] if s.hf else [])          # лише ті FiLM, що справді використовуються
            s.ctx = nn.Sequential(nn.Linear(32 + META_N, 96), nn.LeakyReLU(ACT), nn.Linear(96, 96), nn.LeakyReLU(ACT))
            s.film = nn.Linear(96, 2 * sum(s.fl))
            nn.init.zeros_(s.film.weight)
            nn.init.zeros_(s.film.bias)
            s.register_buffer("lev", torch.linspace(0.02, 0.98, 16).view(1, 16, 1, 1), persistent=False)
            s.register_buffer("glev", torch.tensor([0.005, 0.01, 0.02, 0.04, 0.08, 0.16]).view(1, 6, 1, 1), persistent=False)

        def desc(s, small):
            """глобальний опис кадру з дрібної копії: середнє/розкид/коваріація кольору, CDF яскравості, статистика градієнта -> (B, 32)"""
            m = small.mean((2, 3))
            d = small - m[:, :, None, None]
            sd = (d.pow(2).mean((2, 3)) + 1e-6).sqrt()                      # без нуля під коренем: на однорідних кадрах градієнт sqrt(0) = NaN
            cov = torch.stack([(d[:, 0] * d[:, 1]).mean((1, 2)), (d[:, 0] * d[:, 2]).mean((1, 2)), (d[:, 1] * d[:, 2]).mean((1, 2))], 1) * 4
            y = 0.299 * small[:, 0] + 0.587 * small[:, 1] + 0.114 * small[:, 2]
            cdf = (y.unsqueeze(1) > s.lev).to(small.dtype).mean((2, 3))
            g = (y[:, :-1, 1:] - y[:, :-1, :-1]).abs() + (y[:, 1:, :-1] - y[:, :-1, :-1]).abs()
            gc = (g.unsqueeze(1) > s.glev).to(small.dtype).mean((2, 3))
            return torch.cat([m, sd, cov, cdf, gc, g.mean((1, 2))[:, None]], 1)

        def glob(s, small, meta):
            """глобальна гілка (малі тензори, не залежить від розміру кадру): -> 5 пар (1 + γ, β) форми (B,C,1,1) для FiLM"""
            ctx = s.ctx(torch.cat([s.desc(small.float()), meta.float()], 1).to(s.ctx[0].weight.dtype))
            gb = s.film(ctx)
            gs, o = [], 0
            for c in s.fl:
                gs.append((1 + gb[:, o:o + c, None, None], gb[:, o + c:o + 2 * c, None, None]))
                o += 2 * c
            return gs

        def forward(s, x, noise, small, meta):
            """x (B,3,H,W) RGB 0..1; noise (B,4,H,W); small (B,3,96,96); meta (B,5); H, W кратні 16 -> логіти (B,1,H,W) (маска: > 0)"""
            return s.body(x, noise, s.glob(small, meta))

        def ofeat(s, noise):
            """внесок орієнтованого банка в ознаки 1/2 (B, c1, H/2, W/2): залежить лише від кільцевого шуму, тобто від розміру кадру, а не від зображення -
            тому застосунок рахує його один раз на розмір кадру й подає в тіло готовим (`of`), а не заново на кожному кадрі"""
            rn = noise[:, 3:4]
            return s.op(s.oc(rn) ** 2 + s.os(rn) ** 2)

        def body(s, x, noise, gs, of=None):
            """згорткова частина на повному кадрі; gs - результат glob; of - готовий s.ofeat(noise) (лише для ob > 0)"""
            z = torch.cat([x * 2 - 1, noise], 1)
            f1 = act(s.stem(z if s.lean else F.pixel_unshuffle(z, 2)))
            if s.ob:
                f1 = f1 + (s.ofeat(noise) if of is None else of)
            f1 = s.r1(f1)
            f2 = s.r2(act(s.d1(f1)))
            f3 = s.r3b(s.r3a(act(s.d2(f2))))
            f4 = s.r4b(s.r4a(act(s.d3(f3)), gs[0]), gs[0])
            ups = (lambda t: t) if s.cu else (lambda t: F.interpolate(t, scale_factor=2.0, mode="bilinear", align_corners=False))
            u = s.q3(act(s.u3(ups(f4))) + f3, gs[1])
            u = s.q2(act(s.u2(ups(u))) + f2, gs[2])
            u = s.q1(act(s.u1(ups(u))) + f1, None if s.lite else gs[3])
            full = s.o(u) if s.lean else F.pixel_shuffle(s.o(u), 2)
            h = act(s.h1(torch.cat([full, x * 2 - 1, noise], 1)))
            if s.h2 is not None:
                h = act(s.h2(h))
            if s.hf:
                h = torch.addcmul(gs[-1][1], h, gs[-1][0])
            return s.h3(h)

    return Net()


def count_macs(model, h=256, w=256):
    """MAC на піксель кадру (згортки й лінійні шари; решта дрібниці)"""
    import torch
    import torch.nn as nn
    tot = [0]

    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            tot[0] += o.numel() // o.shape[0] * m.in_channels * m.kernel_size[0] * m.kernel_size[1] // m.groups
        elif isinstance(m, nn.ConvTranspose2d):
            tot[0] += i[0].numel() // i[0].shape[0] * m.out_channels * m.kernel_size[0] * m.kernel_size[1] // m.groups
        elif isinstance(m, nn.Linear):
            tot[0] += m.in_features * m.out_features

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))]
    dummy = example_inputs(h, w, "cpu")
    with torch.no_grad():
        model(*dummy)
    for x in hs:
        x.remove()
    return tot[0] / (h * w)


def example_inputs(h, w, device, dtype=None):
    import torch
    dt = dtype or torch.float32
    return (torch.rand(1, 3, h, w, device=device, dtype=dt), torch.randn(1, NOISE_CH, h, w, device=device, dtype=dt),
            torch.rand(1, 3, SMALL, SMALL, device=device, dtype=dt), torch.from_numpy(meta_of(h))[None].to(device))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ================================================================ втрати й метрики
def losses(logit, t, v, lam=2.0):
    """BCE + помилка середнього тону на масштабах 4 і 16 (щоб щільність крапок/штрихів була вірною, навіть коли фазу не вгадано)"""
    import torch
    import torch.nn.functional as F
    logit = logit.float()
    t = t.float()
    v = v.float()
    bce = (F.binary_cross_entropy_with_logits(logit, t, reduction="none") * v).sum() / v.sum().clamp_min(1)
    p = torch.sigmoid(logit)
    tone = torch.zeros((), device=logit.device)
    for k in (4, 16):
        if p.shape[-1] >= k and p.shape[-2] >= k:
            vv = F.avg_pool2d(v, k)
            a = F.avg_pool2d(p * v, k) / vv.clamp_min(1e-3)
            b = F.avg_pool2d(t * v, k) / vv.clamp_min(1e-3)
            tone = tone + ((a - b).abs() * (vv > 0.5)).sum() / (vv > 0.5).sum().clamp_min(1)
    return bce + lam * tone, bce, tone


_SB: dict = {}


def stroke_bank(dev):
    import torch
    k = _SB.get(str(dev))
    if k is None:
        k = _SB[str(dev)] = torch.from_numpy(gabor_kernels(6, 9, 4.0)).to(dev)
    return k


def stroke_energy(q, v):
    """q у [-1, 1] (B,1,H,W) -> (локальна енергія за 6 напрямками (B,6,H,W), вага «далеко від краю вирізки» (B,1,H,W)):
    квадратурний банк Габора з кроком хвилі ≈ крок штриха, згладжений 5×5. Штрихи дають енергію в одному напрямку, розсип/дизер - рівномірно
    слабку, гладка смуга - майже нуль"""
    import torch.nn.functional as F
    g = F.conv2d(q, stroke_bank(q.device), padding=4)
    e = F.avg_pool2d(g[:, 0::2] ** 2 + g[:, 1::2] ** 2, 5, 1, 2)
    return e, 1 - F.max_pool2d(1 - v, 9, 1, 4)


def stroke_err(pred, t, v):
    """відносна помилка енергії штрихів: 0 - як у Ultra, 1 - ніби штрихів нема взагалі (pred - у [0, 1], м'який або жорсткий)"""
    ep, w = stroke_energy(pred * 2 - 1, v)
    et, _ = stroke_energy(t * 2 - 1, v)
    n = (w.sum() * ep.shape[1]).clamp_min(1)
    ref = ((et * w).sum() / n).clamp_min(1e-3)
    return ((ep - et).abs() * w).sum() / n / ref


def kd_loss(logit, teacher_logit, v):
    """дистиляція: м'які мітки вчителя (sigmoid його логіта) замість лише жорсткої маски Ultra. Фазу штриха Ultra жодна мережа не вгадає, а вже навчений
    вчитель дає гладку «усереднену» відповідь, яку менша мережа вчить легше й стабільніше"""
    import torch
    import torch.nn.functional as F
    p = torch.sigmoid(teacher_logit.float())
    v = v.float()
    return (F.binary_cross_entropy_with_logits(logit.float(), p, reduction="none") * v).sum() / v.sum().clamp_min(1)


def load_partial(model, sd):
    """переносить із state_dict лише ті ваги, чиї ім'я й форма збігаються (решта лишається ініціалізованою) -> (перенесено, пропущено)"""
    cur = model.state_dict()
    new = {k: sd[k] for k, v in cur.items() if k in sd and tuple(sd[k].shape) == tuple(v.shape)}
    model.load_state_dict(new, strict=False)
    return sorted(new), sorted(k for k in cur if k not in new)


def _ckpt_path(root, name):
    p = name if (os.path.isdir(name) or os.path.isfile(name)) else os.path.join(root, "runs", name)
    if os.path.isdir(p):
        p = os.path.join(p, "best.pt") if os.path.isfile(os.path.join(p, "best.pt")) else os.path.join(p, "last.pt")
    return p


def metrics(logit, t, v):
    """-> словник для усереднення: точність, F1 білого, помилка тону (8×8), BCE, stroke (відносна помилка енергії штрихів жорсткої маски: 0 - як Ultra, 1 - штрихів нема)"""
    import torch
    import torch.nn.functional as F
    logit = logit.float()
    t = t.float()
    v = v.float()
    pred = (logit > 0).float()
    n = v.sum().clamp_min(1)
    acc = ((pred == t).float() * v).sum() / n
    tp = (pred * t * v).sum()
    f1 = 2 * tp / ((pred * v).sum() + (t * v).sum()).clamp_min(1)
    vv = F.avg_pool2d(v, 8)
    tone = ((F.avg_pool2d(pred * v, 8) - F.avg_pool2d(t * v, 8)).abs() / vv.clamp_min(1e-3) * (vv > 0.5)).sum() / (vv > 0.5).sum().clamp_min(1)
    bce = (F.binary_cross_entropy_with_logits(logit, t, reduction="none") * v).sum() / n
    return dict(acc=acc.item(), f1=f1.item(), tone=tone.item(), bce=bce.item(), stroke=stroke_err(pred, t, v).item())


# ================================================================ повний кадр: підготовка входів, застосування
def frame_inputs(bgr: np.ndarray, ring: RingCache | None):
    """повний кадр -> (x, noise, small, meta) для моделі (numpy); сторони кадру доповнюються до кратних 16 (віддзеркаленням)"""
    h, w = bgr.shape[:2]
    ph, pw = (-h) % 16, (-w) % 16
    pad = cv2.copyMakeBorder(bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT_101) if (ph or pw) else bgr
    H, W = pad.shape[:2]
    x = np.ascontiguousarray(pad[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
    n = noise_window(w, h, 0, 0, H, W, ring)[None]
    return x, n, small_of(bgr)[None], meta_of(h)[None], (h, w)


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fwd(model, x, n, s, m, device, half=True):
    """прямий прохід без градієнтів: на відеокарті - fp16 через autocast (ваги модель лишає fp32), на CPU - fp32; входи - fp32"""
    import torch
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=half and device.type == "cuda"):
        return model(x, n, s, m)


def run_frame(model, bgr, ring, device, half=True):
    import torch
    x, n, s, m, (h, w) = frame_inputs(bgr, ring)
    t = lambda a: torch.from_numpy(a).to(device, torch.float32)          # noqa: E731
    o = fwd(model, t(x), t(n), t(s), t(m), device, half)
    return (o[0, 0, :h, :w] > 0).cpu().numpy().astype(np.uint8)


# ================================================================ чекпойнти
def save_ckpt(path, model, ema, opt, scaler, st, args):
    import torch
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):         # зіпсовані ваги ніколи не затирають справний чекпойнт
        log("УВАГА: ваги містять NaN/inf - чекпойнт не записано")
        return
    tmp = path + ".tmp"
    torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict() if scaler else None,
                    step=st["step"], elapsed=st["elapsed"], best=st.get("best"), ema_n=getattr(ema, "n", 0), args=vars(args)), tmp)
    os.replace(tmp, path)


def load_model(run_or_ckpt: str, which="ema", device=None):
    import torch
    p = run_or_ckpt
    if os.path.isdir(p):
        p = os.path.join(p, "best.pt") if os.path.isfile(os.path.join(p, "best.pt")) else os.path.join(p, "last.pt")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    global NOISE_MODE
    NOISE_MODE = ck["args"].get("noise", "all")
    m = build_model(ck["args"].get("preset", "fast"))
    m.load_state_dict(ck[which] if which in ck and ck[which] else ck["model"])
    m.eval()
    return (m.to(device) if device else m), ck


class EMA:
    def __init__(self, model, decay=0.9995):
        import copy
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    def update(self, model):
        import torch
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        with torch.no_grad():
            ep = list(self.m.parameters())
            torch._foreach_mul_(ep, d)                                                   # групові операції - менше запусків ядер на відеокарті
            torch._foreach_add_(ep, [p.detach() for p in model.parameters()], alpha=1 - d)

    def state_dict(self):
        return self.m.state_dict()


# ================================================================ оцінка
def pick_diverse(cand, k=8, pool=48):
    """прев'ю: з найкращих за штрихами кандидатів беремо k максимально несхожих (за мініатюрою кольорів),
    щоб не було восьми кадрів з одного відео; вибір детермінований і залежить лише від цілі та вхідних кадрів"""
    cs = sorted(cand, key=lambda c: (-c[0], c[1]))[:pool]
    if len(cs) <= k:
        return [c[2] for c in cs]

    def thumb(c):
        im = np.ascontiguousarray(c[2][0].transpose(1, 2, 0)).astype(np.float32)
        return cv2.resize(im, (6, 6), interpolation=cv2.INTER_AREA).ravel() / 255.0
    th = [thumb(c) for c in cs]
    chosen = [0]
    while len(chosen) < k:
        best, bd = None, -1.0
        for j in range(len(cs)):
            if j in chosen:
                continue
            d = min(float(np.abs(th[j] - th[q]).mean()) for q in chosen)
            if d > bd:
                best, bd = j, d
        chosen.append(best)
    return [cs[j][2] for j in chosen]


def evaluate(model, root, items, ring, device, n_crop=2, full=12, seed=1, half=True, smax=256):
    """-> (метрики за вирізками, метрики за повними кадрами, прев'ю-рядки)"""
    import torch
    bench_old = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False          # повні кадри мають різні розміри - автопідбір алгоритмів на кожному був би повільним
    f32 = torch.float32
    ds = InkData(root, items, fixed=True, seed=seed)
    ds.ring = ring
    agg, cnt, prev, cand = {}, 0, [], []
    for i in range(len(items)):
        S = min(bucket_of(items[i][3], items[i][2]), smax)
        for rep in range(n_crop):
            ds.seed = seed + rep
            b = collate([ds[(i, S)]])
            lg = fwd(model, b["x"].to(device, f32).div_(255.0), b["n"].to(device, f32), b["small"].to(device, f32), b["meta"].to(device), device, half)
            m = metrics(lg, b["t"].to(device), b["v"].to(device))
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
            cnt += 1
            if rep == 0:                                    # у прев'ю - вирізки, де є що дивитись (штрихи/півтони Ultra), а не рівні фони й чорні кути; вибір залежить лише від цілі, тож кадри однакові в усіх перевірках
                tt, vv = b["t"].to(device).float(), b["v"].to(device).float()
                en, wv = stroke_energy(tt * 2 - 1, vv)
                fill = float((tt * vv).sum() / vv.sum().clamp_min(1))
                sc = float((en * wv).sum() / (wv.sum() * en.shape[1]).clamp_min(1)) * (1.0 if 0.12 < fill < 0.88 else 0.1) * (1.0 if S >= 128 else 0.5)
                cand.append((sc, i, (b["x"][0].numpy(), b["t"][0, 0].numpy(), (lg[0, 0] > 0).cpu().numpy(), b["v"][0, 0].numpy())))
    crop = {k: v / max(cnt, 1) for k, v in agg.items()}
    prev = pick_diverse(cand, 8)
    fa, fc = {}, 0
    for i in range(0, len(items), max(1, len(items) // max(full, 1))):
        if fc >= full:
            break
        rin, rink, w, h = items[i]
        if w * h > 1.3e6:
            continue
        img, msk = imread(os.path.join(root, rin)), imread(os.path.join(root, rink), gray=True)
        pr = run_frame(model, img, ring, device, half)
        t = torch.from_numpy((msk > 127).astype(np.float32))[None, None]
        lg = torch.from_numpy(pr.astype(np.float32) * 20 - 10)[None, None]
        m = metrics(lg, t, torch.ones_like(t))
        for k, v in m.items():
            fa[k] = fa.get(k, 0.0) + v
        fc += 1
    torch.backends.cudnn.benchmark = bench_old
    return crop, {k: v / max(fc, 1) for k, v in fa.items()}, prev


def save_preview(prev, path):
    rows = []
    for x, t, p, v in prev:
        im = np.ascontiguousarray(x.transpose(1, 2, 0)[:, :, ::-1])
        tt = cv2.cvtColor(t.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        pp = cv2.cvtColor(p.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        df = np.where((t != p)[:, :, None], np.array([0, 0, 255], np.uint8), np.array([40, 40, 40], np.uint8)).astype(np.uint8)
        rows.append(np.hstack([im, tt, pp, df]))
    if rows:
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in rows]
        cv2.imwrite(path, np.vstack(rows))


def _endless(dl):
    """нескінченний обхід завантажувача (епоха за епохою); порожня епоха - помилка, а не вічний цикл"""
    while True:
        n = 0
        for b in dl:
            n += 1
            yield b
        if n == 0:
            raise RuntimeError("завантажувач даних не дав жодного батча")


def bench_ms(preset: str, h: int, w: int, iters: int = 40, dev=None) -> float:
    """мс на кадр h×w у fp16 на відеокарті (PyTorch без оптимізацій; channels_last)"""
    import torch
    dev = dev or get_device()
    m = build_model(preset).to(dev).eval().half().to(memory_format=torch.channels_last)
    ins = example_inputs(h, w, dev, torch.float16)
    ins = (ins[0].contiguous(memory_format=torch.channels_last), ins[1].contiguous(memory_format=torch.channels_last), ins[2], ins[3])
    prev = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    try:
        with torch.no_grad():
            for _ in range(8):
                m(*ins)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters):
                m(*ins)
            torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1000
    finally:
        torch.backends.cudnn.benchmark = prev
        del m
        torch.cuda.empty_cache()


def pick_preset(budget_ms: float, h: int = 1080, log_=log) -> str:
    """найбільша модель, що вкладається в бюджет: мс на кадр висотою h (16:9) у fp16 (бенчмарк у PyTorch; у TensorRT / ONNX Runtime буде не повільніше)"""
    hh = (h // 16) * 16
    ww = int(round(h * 16 / 9 / 16) * 16)
    best = "tiny"
    for p in ("tiny", "fast", "base", "large"):
        ms = bench_ms(p, hh, ww)
        log_(f"  {p:<6} {ms:6.2f} мс на кадр {ww}x{hh}")
        if ms > budget_ms:
            break
        best = p
    return best


# ================================================================ навчання
def cmd_train(a):
    import torch
    dev = get_device()
    root = a.data
    run = a.out or os.path.join(root, "runs", a.run)
    os.makedirs(run, exist_ok=True)
    if dev.type == "cpu" and not a.smoke:
        log("УВАГА: CUDA не знайдено - навчання на CPU буде дуже повільним. Запусти InkReel.bat, пункт 2 (ставить PyTorch з CUDA).")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    global NOISE_MODE
    NOISE_MODE = a.noise
    last = os.path.join(run, "last.pt")
    ck = torch.load(last, map_location="cpu", weights_only=False) if (a.resume and os.path.isfile(last)) else None
    if ck:                                                          # продовження - тією самою моделлю
        a.preset = ck["args"].get("preset", a.preset)
        NOISE_MODE = a.noise = ck["args"].get("noise", a.noise)
    elif a.preset == "auto":
        if dev.type == "cuda":
            log(f"підбираю розмір моделі: найбільша, що вкладається в {a.budget_ms:g} мс на кадр {a.h}p (fp16, PyTorch):")
            a.preset = pick_preset(a.budget_ms, a.h)
        else:
            a.preset = "fast"
        log(f"обрано модель: {a.preset}")
    items = load_items(root, "train", a.limit, a.seed)
    vitems = load_items(root, "val", a.val_n, a.seed)
    if not items:
        raise SystemExit(f"у {root} нема навчальних зразків (manifest.jsonl)")
    log(f"train {len(items)} файлів, val {len(vitems)}; пристрій: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    ring = prepare_rings(root, items + vitems)
    model = build_model(a.preset)
    teacher = None
    if a.init_from and not ck:                                      # старт не з нуля: перенести ваги вже навченої моделі (EMA), що збігаються за іменем і формою
        src = torch.load(_ckpt_path(root, a.init_from), map_location="cpu", weights_only=False)
        ok, skip = load_partial(model, src["ema"] if src.get("ema") else src["model"])
        log(f"ваги з {a.init_from}: перенесено {len(ok)} тензорів, нових/змінених {len(skip)}" + (f" ({', '.join(skip[:6])}{'…' if len(skip) > 6 else ''})" if skip else ""))
    log(f"модель {a.preset}: {n_params(model) / 1e6:.2f} млн параметрів, {count_macs(model) / 1e3:.1f} тис. MAC на піксель "
        f"(1080p ≈ {count_macs(model) * 1920 * 1080 / 1e9:.0f} ГМАС)")
    model = model.to(dev).to(memory_format=torch.channels_last)
    ema = EMA(model, a.ema)
    ema.m.to(dev)
    if a.kd:                                                        # вчитель: готова модель (EMA) у fp16, лише вперед
        _nm = NOISE_MODE
        teacher, _ = load_model(_ckpt_path(root, a.kd), "ema", dev)
        NOISE_MODE = _nm
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        teacher = teacher.to(memory_format=torch.channels_last)
        log(f"дистиляція з {a.kd} (вага {a.kd_w:g})")
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.wd), dict(params=no_decay, weight_decay=0.0)], lr=a.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 12, growth_interval=500) if dev.type == "cuda" else None
    st = dict(step=0, elapsed=0.0, best=None)
    if ck:
        model.load_state_dict(ck["model"])
        ema.m.load_state_dict(ck["ema"])
        ema.n = ck.get("ema_n", 0)
        opt.load_state_dict(ck["opt"])
        if scaler and ck.get("scaler"):
            sc = dict(ck["scaler"])
            if sc.get("scale", 1e9) < 256:                              # після частих переповнень масштаб міг упасти майже до нуля - градієнти fp16 тоді втрачаються
                sc["scale"], sc["_growth_tracker"] = 1024.0, 0
            scaler.load_state_dict(sc)
        st.update(step=ck["step"], elapsed=ck["elapsed"], best=ck.get("best"))
        log(f"продовжую: крок {st['step']}, пройшло {st['elapsed'] / 3600:.2f} год")
    del ck
    if a.add_min > 0:                                               # дотренування: додати N хвилин до вже пройденого часу
        if st["elapsed"] <= 0:
            raise SystemExit("--add-min: нема чого продовжувати (немає last.pt у цьому прогоні)")
        a.hours = st["elapsed"] / 3600.0 + a.add_min / 60.0
        log(f"дотренування: +{a.add_min:g} хв, нова загальна тривалість {a.hours:.2f} год")
    total = a.hours * 3600.0
    ds = InkData(root, items)
    sampler = BucketSampler(items, True, a.seed + st["step"], a.smax, a.batch_scale)
    if len(sampler) == 0:
        raise SystemExit("нема з чого складати батчі")
    nw = a.workers if a.workers is not None else max(2, min((os.cpu_count() or 4) - 4, 14))
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=nw, collate_fn=collate, persistent_workers=nw > 0,
                                     prefetch_factor=6 if nw > 0 else None, pin_memory=dev.type == "cuda")
    log(f"процесів завантаження: {nw}; епоха = {len(sampler)} батчів; навчання {a.hours:g} год, lr {a.lr:g}")
    open(os.path.join(run, "args.json"), "w", encoding="utf-8").write(json.dumps(vars(a), ensure_ascii=False, indent=1))
    dt = torch.float16 if dev.type == "cuda" else torch.float32
    t_start = t_last_log = t_last_eval = t_last_save = time.time()
    el0 = st["elapsed"]
    acc = torch.zeros(4, device=dev)              # суми втрат за інтервал: на відеокарті, щоб не синхронізуватись щокроку
    acc_n = acc_px = 0
    bad = 0
    log_f = open(os.path.join(run, "log.csv"), "a", encoding="utf-8")
    if log_f.tell() == 0:
        log_f.write("step,hours,lr,loss,bce,tone,val_acc,val_f1,val_tone,val_bce,full_acc,full_f1,full_tone,val_stroke,full_stroke\n")
    lr = a.lr
    import copy

    def take_snap():
        return dict(m=copy.deepcopy(model.state_dict()), e=copy.deepcopy(ema.m.state_dict()), n=ema.n, o=copy.deepcopy(opt.state_dict()),
                    s=scaler.state_dict() if scaler else None)

    def weights_ok():
        return all(bool(torch.isfinite(p).all()) for p in model.parameters())

    snap, lr_scale, good, rollbacks = take_snap(), 1.0, 0, 0       # знімок у пам'яті для відкату, якщо втрата стане NaN
    try:
        for b in _endless(dl):
            st["elapsed"] = el0 + time.time() - t_start
            fr = st["elapsed"] / total
            if fr >= 1.0 or (a.max_steps and st["step"] >= a.max_steps):
                break
            warm = min(1.0, (st["step"] + 1) / a.warmup)
            lr = a.lr * warm * lr_scale * (a.min_lr + (1 - a.min_lr) * 0.5 * (1 + math.cos(math.pi * fr)))
            for g in opt.param_groups:
                g["lr"] = lr
            x = b["x"].to(dev, non_blocking=True).to(dt).div_(255.0).contiguous(memory_format=torch.channels_last)
            nz = b["n"].to(dev, non_blocking=True).to(dt).contiguous(memory_format=torch.channels_last)
            sm = b["small"].to(dev, non_blocking=True)
            mt = b["meta"].to(dev, non_blocking=True)
            t_, v_ = b["t"].to(dev, non_blocking=True), b["v"].to(dev, non_blocking=True)
            try:
                with torch.autocast(device_type=dev.type, dtype=dt, enabled=dev.type == "cuda"):
                    lg = model(x, nz, sm, mt)
                    if teacher is not None:
                        with torch.no_grad():
                            lt = teacher(x, nz, sm, mt)
                loss, bce, tone = losses(lg, t_, v_, a.lam)
                strk = stroke_err(torch.sigmoid(lg.float()), t_.float(), v_.float()) if a.stroke_w > 0 else torch.zeros((), device=dev)
                if a.stroke_w > 0:
                    loss = loss + a.stroke_w * strk
                if teacher is not None:
                    loss = loss + a.kd_w * kd_loss(lg, lt, v_)
                if (scaler is None or st["step"] % 10 == 0) and not bool(torch.isfinite(loss)):       # на відеокарті перевірка - раз на 10 кроків (GradScaler сам пропускає поганий крок)
                    bad += 1
                    opt.zero_grad(set_to_none=True)
                    if bad >= 3:                                            # відкат до останнього справного знімка й менша швидкість навчання
                        rollbacks += 1
                        if rollbacks > 20:
                            raise RuntimeError("втрата стала нескінченною/NaN, 20 відкатів не допомогли - зменш --lr")
                        model.load_state_dict(snap["m"])
                        ema.m.load_state_dict(snap["e"])
                        ema.n = snap["n"]
                        opt.load_state_dict(copy.deepcopy(snap["o"]))
                        if scaler and snap["s"]:
                            scaler.load_state_dict(snap["s"])
                        lr_scale *= 0.5
                        bad = good = 0
                        log(f"втрата NaN - відкат до знімка (крок {st['step']}), швидкість навчання ×{lr_scale:g} (відкатів: {rollbacks})")
                    continue
                bad = 0
                opt.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                log("не вистачило пам'яті відеокарти на батчі - пропускаю (якщо повторюється: --batch-scale 0.5)")
                continue
            ema.update(model)
            st["step"] += 1
            good += 1
            if good % 1000 == 0 and lr_scale < 1.0:
                lr_scale = min(1.0, lr_scale * 1.25)
            if st["step"] % 300 == 0 and bad == 0 and weights_ok():
                snap = take_snap()
            acc += torch.stack([loss.detach(), bce.detach(), tone.detach(), strk.detach()]).nan_to_num(0.0, posinf=0.0, neginf=0.0)
            acc_n += 1
            acc_px += x.shape[0] * x.shape[2] * x.shape[3]
            now = time.time()
            if now - t_last_log > a.log_s:
                m_ = (acc / max(acc_n, 1)).tolist()
                log(f"крок {st['step']:>7}  {st['elapsed'] / 3600:5.2f}/{a.hours:g} год  lr {lr:.2e}  втрата {m_[0]:.4f} (bce {m_[1]:.4f}, тон {m_[2]:.4f}, штрихи {m_[3]:.3f})  "
                    f"{acc_px / (now - t_last_log) / 1e6:6.1f} Мп/с  {len(x):>3} вирізок")
                t_last_log = now
                acc.zero_()
                acc_n = acc_px = 0
            if now - t_last_save > a.save_min * 60:
                save_ckpt(last, model, ema, opt, scaler, st, a)
                t_last_save = now
            if now - t_last_eval > a.eval_min * 60:
                _eval_and_log(ema.m, root, vitems, ring, dev, run, st, lr, a, log_f, model, opt, scaler)
                save_ckpt(last, model, ema, opt, scaler, st, a)               # разом із оновленим «найкращим» результатом
                t_last_eval = t_last_save = time.time()
    except KeyboardInterrupt:
        log("\nзупинено вручну - зберігаю…")
    save_ckpt(last, model, ema, opt, scaler, st, a)
    try:
        _eval_and_log(ema.m, root, vitems, ring, dev, run, st, 0.0, a, log_f, model, opt, scaler)
        save_ckpt(last, model, ema, opt, scaler, st, a)
    except KeyboardInterrupt:
        log("фінальну перевірку пропущено")
    log(f"готово: крок {st['step']}, {st['elapsed'] / 3600:.2f} год. Найкраща модель: {os.path.join(run, 'best.pt')}")
    log_f.close()


def _eval_and_log(ema_model, root, vitems, ring, dev, run, st, lr, a, log_f, model, opt, scaler):
    if not vitems:
        return
    t0 = time.time()
    crop, full, prev = evaluate(ema_model, root, vitems, ring, dev, full=a.full_n, smax=a.smax)
    log(f"  перевірка (EMA): вирізки acc {crop['acc']:.4f} F1 {crop['f1']:.4f} тон {crop['tone']:.4f} bce {crop['bce']:.4f} штрихи {crop['stroke']:.3f}"
        + (f" | повні кадри acc {full.get('acc', 0):.4f} F1 {full.get('f1', 0):.4f} тон {full.get('tone', 0):.4f} штрихи {full.get('stroke', 0):.3f}" if full else "") + f"  ({time.time() - t0:.0f} с)")
    save_preview(prev, os.path.join(run, f"preview_{st['step']:07d}.png"))
    for f in sorted(f for f in os.listdir(run) if f.startswith("preview_"))[:-6]:
        try:
            os.remove(os.path.join(run, f))
        except OSError:
            pass
    log_f.write(f"{st['step']},{st['elapsed'] / 3600:.3f},{lr:.3e},,,,{crop['acc']:.5f},{crop['f1']:.5f},{crop['tone']:.5f},{crop['bce']:.5f},"
                f"{full.get('acc', 0):.5f},{full.get('f1', 0):.5f},{full.get('tone', 0):.5f},{crop['stroke']:.5f},{full.get('stroke', 0):.5f}\n")
    log_f.flush()
    score = crop["bce"] + a.stroke_sel * (0.5 * crop["stroke"] + 0.5 * full.get("stroke", crop["stroke"]))      # лише bce обирав би гладку модель без штрихів
    if st.get("best") is None or score < st["best"]:
        st["best"] = score
        save_ckpt(os.path.join(run, "best.pt"), model, _EmaView(ema_model), opt, scaler, st, a)
        log(f"  найкраща на цей момент: оцінка {score:.4f} (bce {crop['bce']:.4f} + {a.stroke_sel:g}·штрихи) -> best.pt")


class _EmaView:
    """обгортка, щоб save_ckpt записав у поле ema ваги переданої моделі"""

    def __init__(self, m):
        self.m = m

    def state_dict(self):
        return self.m.state_dict()


# ================================================================ інші команди
def cmd_info(a):
    for p in (list(PRESETS) if a.preset in ("auto", "all") else [a.preset]):
        m = build_model(p)
        mac = count_macs(m)
        log(f"{p:<6} {n_params(m) / 1e6:.3f} млн параметрів, {mac / 1e3:.2f} тис. MAC на піксель (1080p ≈ {mac * 1920 * 1080 / 1e9:.0f} ГМАС на кадр)")


def cmd_bench(a):
    import torch
    dev = get_device()
    names = list(PRESETS) if a.preset == "all" else [p for p in PRESETS if not PRESETS[p].get("cu")] if a.preset == "auto" else [a.preset]
    h = (a.h // 16) * 16
    w = a.w or int(round(a.h * 16 / 9 / 16) * 16)
    if dev.type != "cuda":
        for p in names:
            m = build_model(p)
            log(f"{p}: {n_params(m) / 1e6:.2f} млн параметрів, {count_macs(m) / 1e3:.1f} тис. MAC/піксель")
        log("CUDA нема - швидкість на відеокарті не заміряю (запусти InkReel.bat, пункт 2).")
        return 1
    log(f"{torch.cuda.get_device_name(0)}; кадр {w}x{h}; fp16, PyTorch без оптимізацій (у TensorRT / ONNX Runtime має бути не повільніше). Закрий ігри й важкі програми.")
    if a.profile:
        profile_ms("fast" if a.preset in ("all", "auto") else a.preset, h, w, dev)
        return 0
    pick = None
    for p in names:
        m = build_model(p)
        macs, npar = count_macs(m), n_params(m)
        del m
        ms = bench_ms(p, h, w, a.iters, dev)
        ok = ms <= a.budget_ms
        log(f"  {p:<6} {npar / 1e6:5.2f} млн парам., {macs / 1e3:5.1f} тис. MAC/піксель  ->  {ms:6.2f} мс = {1000 / ms:5.0f} к/с  {'✓' if ok else '(повільно)'}")
        if ok:
            pick = p
    log(f"Бюджет {a.budget_ms:g} мс на кадр: " + (f"найбільша модель, що вкладається, - {pick} (її й візьме `train` за замовчуванням)." if pick else
        "жодна не вкладається - бери tiny і/або зменшуй кадр."))


def profile_ms(preset: str, h: int, w: int, dev, top: int = 14):
    """де йде час: топ операцій за часом на відеокарті (torch.profiler), fp16, один кадр h×w"""
    import torch
    from torch.profiler import ProfilerActivity, profile
    m = build_model(preset).to(dev).eval().half().to(memory_format=torch.channels_last)
    ins = example_inputs(h, w, dev, torch.float16)
    ins = (ins[0].contiguous(memory_format=torch.channels_last), ins[1].contiguous(memory_format=torch.channels_last), ins[2], ins[3])
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    with torch.no_grad():
        for _ in range(5):
            m(*ins)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        with profile(activities=acts) as prof:
            for _ in range(5):
                m(*ins)
            if dev.type == "cuda":
                torch.cuda.synchronize()
    key = "cuda_time_total" if dev.type == "cuda" else "cpu_time_total"
    log(f"--- профіль {preset} {w}x{h} (5 кадрів сумарно) ---")
    log(prof.key_averages().table(sort_by=key, row_limit=top, max_name_column_width=60))


def cmd_cuda(a):
    """перевірка, що PyTorch справді бачить відеокарту й рахує на ній (код повернення 0 - усе гаразд)"""
    import torch
    log(f"PyTorch {torch.__version__}; збірка CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        log("CUDA недоступна: PyTorch без CUDA або драйвер NVIDIA застарий/відсутній.")
        return 1
    try:
        x = torch.rand(1, 8, 64, 64, device="cuda", dtype=torch.float16)
        w = torch.rand(8, 8, 3, 3, device="cuda", dtype=torch.float16)
        y = torch.nn.functional.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        assert bool(torch.isfinite(y).all())
    except Exception as e:  # noqa: BLE001
        log(f"CUDA є, але обчислення не спрацювали: {e}")
        return 1
    p = torch.cuda.get_device_properties(0)
    log(f"відеокарта: {p.name}, {p.total_memory / 2 ** 30:.1f} ГБ, compute {p.major}.{p.minor} - працює")
    return 0


def cmd_noise(a):
    """текстури шуму під розмір кадру (для шейдера/рушія): NPZ із каналами Tg, Td, Tr, кільцевим шумом і мета"""
    ring = RingCache(os.path.join(a.data, "_cache"))
    n = noise_window(a.w, a.h, 0, 0, a.h, a.w, ring)
    out = a.out or f"ink_noise_{a.w}x{a.h}.npz"
    np.savez_compressed(out, noise=n.astype(np.float16), meta=meta_of(a.h), w=a.w, h=a.h)
    log(f"записано {out}: noise {n.shape} (float16), meta {meta_of(a.h).tolist()}")


def cmd_eval(a):
    dev = get_device()
    model, ck = load_model(a.run, a.which, dev)
    root = ck["args"].get("data", a.data) if not a.data_override else a.data
    items = load_items(root, a.split, a.n, 1)
    ring = prepare_rings(root, items)
    crop, full, _ = evaluate(model, root, items, ring, dev, full=a.full)
    log(f"вирізки: {crop}\nповні кадри: {full}")


def cmd_infer(a):
    import torch
    dev = get_device()
    model, ck = load_model(a.run, "ema", dev)
    img = imread(a.image)
    if img is None:
        raise SystemExit(f"не прочиталось: {a.image}")
    if a.height:
        w = int(round(img.shape[1] * a.height / img.shape[0] / 2) * 2)
        h = a.height
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if h <= img.shape[0] else cv2.INTER_CUBIC)
    root = a.data
    ring = RingCache(os.path.join(root, "_cache"))
    t0 = time.time()
    pred = run_frame(model, img, ring, dev)
    log(f"{img.shape[1]}x{img.shape[0]}: {time.time() - t0:.2f} с (з прогрівом)")
    out = a.out or os.path.splitext(a.image)[0] + "_student.png"
    cv2.imwrite(out, pred * 255)
    log("записано", out)
    if a.ref:
        sys.path.insert(0, ROOT)
        os.environ["INKREEL_GPU"] = "off"
        from benvid.core import Params, binarize
        h, w = img.shape[:2]
        ref = binarize(img, w, h, Params(method=5))
        cv2.imwrite(os.path.splitext(out)[0] + "_ultra.png", ref * 255)
        agree = float((ref == pred).mean())
        t = torch.from_numpy(ref.astype(np.float32))[None, None]
        p = torch.from_numpy(pred.astype(np.float32) * 20 - 10)[None, None]
        m = metrics(p, t, torch.ones_like(t))
        log(f"збіг пікселів із справжнім Ultra: {agree * 100:.2f}%  F1 {m['f1']:.4f}  помилка тону (8×8) {m['tone']:.4f}")


def _to_fp16(path):
    import onnx
    from onnxconverter_common import float16
    m32 = onnx.load(path)
    own = {nd.name for nd in m32.graph.node if nd.op_type == "Cast"}
    m16 = float16.convert_float_to_float16(m32, keep_io_types=False)
    for nd in m16.graph.node:                        # власні Cast(to=float32) з .float()/.to() у моделі - тепер теж у fp16 (додані конвертером не чіпаємо)
        if nd.op_type == "Cast" and nd.name in own:
            for at in nd.attribute:
                if at.name == "to" and at.i == onnx.TensorProto.FLOAT:
                    at.i = onnx.TensorProto.FLOAT16
    onnx.save(m16, path)
    log("перетворено у fp16 (входи й вихід теж fp16):", path)


def _export_parts(a, model, ck, out):
    """два ONNX: <out>_global.onnx (small, meta -> 5 пар FiLM (1+γ, β); крихітний, лишається в fp32, годиться для CPU)
    і <out> (image, noise, FiLM -> logits; саме він іде на відеокарту, у fp16 з --fp16). Без дрібних операцій глобальної гілки на GPU."""
    import torch
    import torch.nn as nn
    fl = model.fl
    gnames = [n for i in range(len(fl)) for n in (f"g{i}", f"b{i}")]

    class G(nn.Module):
        def __init__(s):
            super().__init__()
            s.m = model

        def forward(s, small, meta):
            return tuple(t for pair in s.m.glob(small, meta) for t in pair)

    has_of = bool(getattr(model, "ob", 0))                                  # орієнтований банк: його внесок тіло одержує готовим (вхід ofeat), див. Net.ofeat

    class B(nn.Module):
        def __init__(s):
            super().__init__()
            s.m = model

        def forward(s, x, noise, *f):
            of, f = (f[0], f[1:]) if has_of else (None, f)
            y = s.m.body(x, noise, [(f[2 * i], f[2 * i + 1]) for i in range(len(fl))], of=of)
            return (y > 0).to(torch.uint8) if a.mask else y                 # маска (1 = світло) прямо з відеокарти: у 2-4 рази менше даних назад

    class O(nn.Module):
        def __init__(s):
            super().__init__()
            s.m = model

        def forward(s, noise):
            return s.m.ofeat(noise)

    h, w = 272, 480
    ins = example_inputs(h, w, "cpu")
    with torch.no_grad():
        film = G()(ins[2], ins[3])
    gpath = os.path.splitext(out)[0] + "_global.onnx"

    def exp(m, args, path, **kw):
        kw.setdefault("opset_version", 17)
        try:
            torch.onnx.export(m, args, path, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(m, args, path, **kw)
        log("записано", path)

    exp(G(), (ins[2], ins[3]), gpath, input_names=["small", "meta"], output_names=gnames,
        dynamic_axes={"small": {0: "b"}, "meta": {0: "b"}, **{n: {0: "b"} for n in gnames}})
    with torch.no_grad():
        ofv = O()(ins[1]) if has_of else None
    opath = os.path.splitext(out)[0] + "_of.onnx"
    exp(B(), (ins[0], ins[1], *([ofv] if has_of else []), *film), out, input_names=["image", "noise", *(["ofeat"] if has_of else []), *gnames],
        output_names=["logits"],
        dynamic_axes={"image": {0: "b", 2: "h", 3: "w"}, "noise": {0: "b", 2: "h", 3: "w"}, "logits": {0: "b", 2: "h", 3: "w"},
                      **({"ofeat": {0: "b", 2: "h2", 3: "w2"}} if has_of else {}), **{n: {0: "b"} for n in gnames}})
    if has_of and not a.fp16:                                               # лише fp32: застосунок рахує його на процесорі один раз на розмір кадру
        exp(O(), (ins[1],), opath, input_names=["noise"], output_names=["ofeat"], dynamic_axes={"noise": {0: "b", 2: "h", 3: "w"}, "ofeat": {0: "b", 2: "h2", 3: "w2"}})
    if a.fp16:
        _to_fp16(out)
    else:
        try:
            import onnxruntime as ort
            sg = ort.InferenceSession(gpath, providers=["CPUExecutionProvider"])
            sb = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
            x = example_inputs(128, 192, "cpu")
            fo = sg.run(None, dict(small=x[2].numpy(), meta=x[3].numpy()))
            extra = {}
            if has_of:
                extra["ofeat"] = ort.InferenceSession(opath, providers=["CPUExecutionProvider"]).run(None, {"noise": x[1].numpy()})[0]
            r = sb.run(None, {"image": x[0].numpy(), "noise": x[1].numpy(), **extra, **dict(zip(gnames, fo))})[0]
            with torch.no_grad():
                ref = model(*x).numpy()
            if a.mask:
                log(f"перевірка onnxruntime (дві частини): маска відрізняється у {float((r != (ref > 0)).mean()) * 100:.4f}% пікселів")
            else:
                log(f"перевірка onnxruntime (дві частини): макс. відхилення логітів {abs(r - ref).max():.2e}")
        except ImportError:
            log("onnxruntime не встановлено - перевірку ONNX пропущено")
    json.dump(dict(parts="global: small,meta -> g0,b0..g4,b4 (FiLM: 1+gamma, beta), fp32; body: image,noise,[ofeat],g0,b0..g4,b4 -> logits; of (лише ir1): noise -> ofeat, fp32",
                   preset=ck["args"].get("preset"), step=ck.get("step")),
              open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def cmd_export(a):
    import torch
    half = False
    if a.random:                                    # неотренована модель - лише для заміру швидкості/сумісності (DirectML тощо)
        name = "fast" if a.preset in ("auto", "all") else a.preset
        model, ck = build_model(name).eval(), dict(args=dict(preset=name), step=0)
        out = a.out or f"ink_random_{name}.onnx"
    else:
        model, ck = load_model(a.run, "ema", torch.device("cpu"))
        out = a.out or os.path.join(a.run if os.path.isdir(a.run) else os.path.dirname(a.run), "ink_student.onnx")
    model.float()
    h, w = 272, 480
    ins = example_inputs(h, w, "cpu")
    if a.parts:
        return _export_parts(a, model, ck, out)
    kw = dict(input_names=["image", "noise", "small", "meta"], output_names=["logits"],
              dynamic_axes={"image": {0: "b", 2: "h", 3: "w"}, "noise": {0: "b", 2: "h", 3: "w"}, "small": {0: "b"}, "meta": {0: "b"},
                            "logits": {0: "b", 2: "h", 3: "w"}}, opset_version=17)
    try:
        torch.onnx.export(model, ins, out, dynamo=False, **kw)           # перевірений класичний експортер
    except TypeError:                                                    # старий torch без параметра dynamo
        torch.onnx.export(model, ins, out, **kw)
    log("записано", out)
    if a.fp16:                                      # fp16-граф для відеокарти: перетворення готового fp32-графа
        _to_fp16(out)
    try:
        if a.fp16:
            raise ImportError
        import onnxruntime as ort
        s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        h2, w2 = 128, 192
        x = example_inputs(h2, w2, "cpu")
        r = s.run(None, dict(image=x[0].numpy(), noise=x[1].numpy(), small=x[2].numpy(), meta=x[3].numpy()))[0]
        with torch.no_grad():
            ref = model(*x).numpy()
        log(f"перевірка onnxruntime: макс. відхилення логітів {abs(r - ref).max():.2e}")
    except ImportError:
        log("onnxruntime не встановлено - перевірку ONNX пропущено (pip install onnxruntime)")
    meta = dict(inputs=dict(image="(1,3,H,W) RGB float 0..1; H, W кратні 16 (доповни віддзеркаленням)",
                            noise="(1,4,H,W): Tg, Td, Tr у [-1,1] і кільцевий шум; див. `train_ink.py noise --w W --h H`",
                            small="(1,3,96,96) RGB 0..1: кадр, зменшений усередненням (area)", meta="(1,5): див. meta_of(h) у train_ink.py"),
                output="logits (1,1,H,W); біле = logit > 0", preset=ck["args"].get("preset"), step=ck.get("step"))
    json.dump(meta, open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Навчання студента для рівня «Ультра»")
    ap.add_argument("cmd", choices=["train", "info", "bench", "cuda", "eval", "infer", "export", "noise", "prepare"])
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--run", default="ink1", help="назва прогону (тека runs/НАЗВА у датасеті) або шлях до теки/чекпойнта для eval/infer/export")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preset", default="auto", choices=["auto", "all"] + list(PRESETS),
                    help="auto - найбільша модель, що вкладається в --budget-ms на кадр --h (під час навчання); all - лише для bench")
    ap.add_argument("--budget-ms", dest="budget_ms", type=float, default=25.0, help="бюджет часу моделі на кадр висотою --h (мс, fp16, PyTorch без злиття операцій - у TensorRT/DirectML удвічі-втричі швидше) для auto / bench")
    ap.add_argument("--hours", type=float, default=9.0)
    ap.add_argument("--add-min", dest="add_min", type=float, default=0.0, help="дотренування: додати стільки хвилин до вже пройденого часу (разом з --resume)")
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--min-lr", dest="min_lr", type=float, default=0.01, help="частка lr наприкінці")
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=2.0, help="вага втрати середнього тону")
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--eval-min", dest="eval_min", type=float, default=20.0)
    ap.add_argument("--save-min", dest="save_min", type=float, default=10.0)
    ap.add_argument("--log-s", dest="log_s", type=float, default=60.0)
    ap.add_argument("--val-n", dest="val_n", type=int, default=200)
    ap.add_argument("--full-n", dest="full_n", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="брати лише стільки файлів train (для швидких перевірок)")
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smax", type=int, default=256, help="найбільша сторона вирізки (256; менше - швидше й легше для пам'яті)")
    ap.add_argument("--batch-scale", dest="batch_scale", type=float, default=1.0, help="множник розміру батча (0.5 - якщо не вистачає пам'яті відеокарти)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", dest="init_from", default=None, help="почати не з нуля: взяти ваги (EMA) з цього прогону, де ім'я й форма тензора збігаються (для легших пресетів fq_*)")
    ap.add_argument("--kd", default=None, help="дистиляція: прогін-вчитель (його м'які відповіді додаються до втрати)")
    ap.add_argument("--kd-w", dest="kd_w", type=float, default=1.0, help="вага втрати дистиляції")
    ap.add_argument("--stroke-w", dest="stroke_w", type=float, default=1.0, help="вага втрати енергії штрихів (0 - вимкнути)")
    ap.add_argument("--stroke-sel", dest="stroke_sel", type=float, default=0.5, help="вага метрики штрихів в оцінці «найкращої» моделі: bce + вага·штрихи")
    ap.add_argument("--smoke", action="store_true", help="пробний прогін ≈ 6 хв на частині даних")
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--w", type=int, default=0)
    ap.add_argument("--profile", action="store_true", help="bench: показати, на які операції йде час")
    ap.add_argument("--random", action="store_true", help="export: неотренована модель (для заміру швидкості)")
    ap.add_argument("--fp16", action="store_true", help="export: перетворити граф у fp16 (потрібен onnxconverter-common)")
    ap.add_argument("--mask", action="store_true", help="export --parts: частина на відеокарті віддає готову маску uint8 (logit > 0), а не логіти")
    ap.add_argument("--parts", action="store_true", help="export: два ONNX (глобальна гілка окремо + швидка згорткова частина)")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--image", default=None)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--noise", default="all", choices=["all", "tiles", "none"], help=argparse.SUPPRESS)
    ap.add_argument("--which", default="ema")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--full", type=int, default=12)
    ap.add_argument("--data-override", dest="data_override", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.hours, a.limit, a.val_n, a.eval_min, a.save_min, a.full_n, a.run = 0.1, 6000, 60, 3.0, 3.0, 4, "smoke"
        a.warmup = 100
    if a.cmd == "train":
        return cmd_train(a)
    if a.cmd == "info":
        return cmd_info(a)
    if a.cmd == "bench":
        return cmd_bench(a)
    if a.cmd == "cuda":
        return cmd_cuda(a)
    if a.cmd == "noise":
        return cmd_noise(a)
    if a.cmd == "prepare":
        items = load_items(a.data, "train") + load_items(a.data, "val")
        prepare_rings(a.data, items)
        return 0
    if a.cmd in ("eval", "infer", "export") and not (a.cmd == "export" and a.random) and not (os.path.isdir(a.run) or os.path.isfile(a.run)):
        a.run = os.path.join(a.data, "runs", a.run)
    return dict(eval=cmd_eval, infer=cmd_infer, export=cmd_export)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
