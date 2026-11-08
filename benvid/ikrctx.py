"""Стискання кадрів .ikr (кодек «ctx», формат файлу версії 2): контекстне арифметичне кодування чорно-білих кадрів без втрат.

Навіщо. Кадр - 1 біт на піксель, але зерно й штрихи Ink/Ultra не стискаються «байтовими» стискачами (zlib бачить випадкові байти): XOR із попереднім
кадром + zlib давали ~30% від «сирих» бітів на кадр. Тут кожен піксель кодується арифметично за КОНТЕКСТОМ - уже відомі сусіди в цьому кадрі
(ліворуч, вище) і в попередньому кадрі (той самий піксель і його сусіди): ймовірність кожного біта навчається за ходом кадрів, тож стабільний візерунок майже
нічого не коштує, а рухомий - стільки, скільки в ньому справжньої новизни. Файли виходять приблизно вдвічі меншими, ніж із zlib (чорнило 4-5: ~-48..-50%,
символи: ~-46..-59%).

Як влаштовано.
  - контекст - 12 біт: 6 пікселів цього кадра (x-1, x-2 у рядку; x-1, x, x+1 вище; два рядки вище), 5 пікселів попереднього (x,y), (x+1,y), (x,y+1), (x-1,y),
    (x,y-1) і ознака «кадр без попереднього» (перший у проміжку). Таблиця з 4096 записів лежить у кеші процесора - швидко;
  - ймовірність біта = середнє двох лічильників: швидкого (зсув 4) і повільного (зсув 6) - так вона і швидко підлаштовується, і не стрибає;
  - арифметичний кодер той самий за будовою, що в LZMA (32-бітний діапазон, перенос), на цілих числах: однаковий результат на будь-якому ПК;
  - лічильники й попередній кадр переходять від кадра до кадра В МЕЖАХ проміжку з KEY кадрів і скидаються на ключовому кадрі: довільний доступ,
    як і раніше, - розпаковка від найближчого ключового (до KEY-1 кадрів), а проміжки незалежні.
Запис кадра: байти арифметичного кодера + CRC32 цих байтів (пошкодження файлу не мовчить).
Потрібна numba (вона є в складі програми); без неї новий файл пишеться кодеком zlib (версія 1), а файл «ctx» не відкривається з поясненням.
"""
from __future__ import annotations

import zlib

import numpy as np

try:
    import numba as _nb
except Exception:  # noqa: BLE001
    _nb = None
else:
    from . import nbcache as _nbcache
    _nbcache.install()

NCTX = 1 << 12
SH_FAST, SH_SLOW = 4, 6
PMIN, PMAX = 16, 65520
CRC = 4


def available() -> bool:
    return _nb is not None


def _prior() -> np.ndarray:
    """початкова ймовірність «1» для кожного контексту: у кадрі з попереднім - «як у попередньому кадрі» (довіра тим більша, чим більше сусідів попереднього
    кадра такі самі), у ключовому - за сусідами зліва/зверху. Без цього на малих кадрах навчання 4096 контекстів з нуля коштувало б більше, ніж виграш"""
    c = np.arange(NCTX)
    bit = lambda k: (c >> k) & 1
    kf = bit(11)
    pv = bit(6)
    same = sum((bit(k) == pv).astype(np.int32) for k in (7, 8, 9, 10))        # скільки з 4 сусідів попереднього кадра збігаються з його пікселем
    conf = 0.80 + 0.045 * same                                                # 0.80 .. 0.98
    p_t = np.where(pv == 1, conf, 1.0 - conf)
    n1 = bit(0) + bit(2) + bit(3) + bit(4)                                    # зліва, зліва-вище, вище, справа-вище
    p_k = (n1 + 1.0) / 6.0
    p = np.where(kf == 1, p_k, p_t)
    return np.clip(np.rint(p * 65536), PMIN, PMAX).astype(np.int32)


_PRIOR = _prior()


def new_state() -> np.ndarray:
    """лічильники ймовірностей (швидкі й повільні): початкові значення - _PRIOR; ключовий кадр проміжку починає з них наново"""
    return np.stack([_PRIOR, _PRIOR]).copy()


def _pad_cur(h: int, w: int) -> np.ndarray:
    return np.zeros((h + 2, w + 4), np.uint8)


def _pad_prev(m: np.ndarray | None, h: int, w: int) -> np.ndarray:
    pp = np.zeros((h + 2, w + 2), np.uint8)
    if m is not None:
        pp[1:h + 1, 1:w + 1] = m
    return pp


if _nb is not None:
    @_nb.njit(cache=True, nogil=True)
    def _enc(cp, pp, kf, h, w, st, out):
        low = np.int64(0)
        rng = np.int64(0xFFFFFFFF)
        cache = np.int64(0)
        csize = np.int64(1)
        pos = 0
        for y in range(h):
            for x in range(w):
                a = cp[y + 2, x + 2]
                c = (cp[y + 2, x + 1] | (cp[y + 2, x] << 1) | (cp[y + 1, x + 1] << 2) | (cp[y + 1, x + 2] << 3) | (cp[y + 1, x + 3] << 4)
                     | (cp[y, x + 2] << 5) | (pp[y + 1, x + 1] << 6) | (pp[y + 1, x + 2] << 7) | (pp[y + 2, x + 1] << 8)
                     | (pp[y + 1, x] << 9) | (pp[y, x + 1] << 10) | (kf << 11))
                p1 = st[0, c]
                p2 = st[1, c]
                p = (p1 + p2) >> 1
                if p < 16:
                    p = 16
                elif p > 65520:
                    p = 65520
                bound = (rng * p) >> 16
                if a:
                    rng = bound
                    st[0, c] = p1 + ((65536 - p1) >> 4)
                    st[1, c] = p2 + ((65536 - p2) >> 6)
                else:
                    low += bound
                    rng -= bound
                    st[0, c] = p1 - (p1 >> 4)
                    st[1, c] = p2 - (p2 >> 6)
                while rng < 16777216:
                    rng = (rng << 8) & 0xFFFFFFFF
                    # shift_low
                    if low < 0xFF000000 or low >= 0x100000000:
                        carry = low >> 32
                        t = cache
                        while True:
                            out[pos] = (t + carry) & 0xFF
                            pos += 1
                            t = 0xFF
                            csize -= 1
                            if csize == 0:
                                break
                        cache = (low >> 24) & 0xFF
                    csize += 1
                    low = (low & 0x00FFFFFF) << 8
        for _ in range(5):
            if low < 0xFF000000 or low >= 0x100000000:
                carry = low >> 32
                t = cache
                while True:
                    out[pos] = (t + carry) & 0xFF
                    pos += 1
                    t = 0xFF
                    csize -= 1
                    if csize == 0:
                        break
                cache = (low >> 24) & 0xFF
            csize += 1
            low = (low & 0x00FFFFFF) << 8
        return pos

    @_nb.njit(cache=True, nogil=True)
    def _dec(buf, cp, pp, kf, h, w, st):
        pos = 1
        code = np.int64(0)
        for _ in range(4):
            code = (code << 8) | buf[pos]
            pos += 1
        rng = np.int64(0xFFFFFFFF)
        for y in range(h):
            for x in range(w):
                c = (cp[y + 2, x + 1] | (cp[y + 2, x] << 1) | (cp[y + 1, x + 1] << 2) | (cp[y + 1, x + 2] << 3) | (cp[y + 1, x + 3] << 4)
                     | (cp[y, x + 2] << 5) | (pp[y + 1, x + 1] << 6) | (pp[y + 1, x + 2] << 7) | (pp[y + 2, x + 1] << 8)
                     | (pp[y + 1, x] << 9) | (pp[y, x + 1] << 10) | (kf << 11))
                p1 = st[0, c]
                p2 = st[1, c]
                p = (p1 + p2) >> 1
                if p < 16:
                    p = 16
                elif p > 65520:
                    p = 65520
                bound = (rng * p) >> 16
                if code < bound:
                    rng = bound
                    cp[y + 2, x + 2] = 1
                    st[0, c] = p1 + ((65536 - p1) >> 4)
                    st[1, c] = p2 + ((65536 - p2) >> 6)
                else:
                    code -= bound
                    rng -= bound
                    st[0, c] = p1 - (p1 >> 4)
                    st[1, c] = p2 - (p2 >> 6)
                while rng < 16777216:
                    rng = (rng << 8) & 0xFFFFFFFF
                    code = ((code << 8) | buf[pos]) & 0xFFFFFFFF
                    pos += 1
        return pos


def encode(mask: np.ndarray, prev: np.ndarray | None, st: np.ndarray) -> bytes:
    """кадр (h, w) {0,1} -> байти (арифметичний код + CRC32). prev - попередній кадр цього проміжку (None - ключовий кадр); st - лічильники (змінюються)"""
    m = np.ascontiguousarray(mask, np.uint8)
    h, w = m.shape
    cp = _pad_cur(h, w)
    cp[2:h + 2, 2:w + 2] = m
    out = np.empty(h * w * 2 + 64, np.uint8)
    n = _enc(cp, _pad_prev(prev, h, w), 0 if prev is not None else 1, h, w, st, out)
    b = out[:n].tobytes()
    return b + zlib.crc32(b).to_bytes(CRC, "little")


def decode(rec: bytes, h: int, w: int, prev: np.ndarray | None, st: np.ndarray) -> np.ndarray:
    """записи кадра -> (h, w) uint8 {0,1}; ValueError - запис пошкоджено"""
    if len(rec) < CRC + 5:
        raise ValueError("запис кадра закороткий")
    b, crc = rec[:-CRC], int.from_bytes(rec[-CRC:], "little")
    if zlib.crc32(b) != crc:
        raise ValueError("контрольна сума кадра не збігається")
    buf = np.frombuffer(b + b"\x00" * 8, np.uint8)
    cp = _pad_cur(h, w)
    _dec(buf, cp, _pad_prev(prev, h, w), 0 if prev is not None else 1, h, w, st)
    return cp[2:h + 2, 2:w + 2].copy()
