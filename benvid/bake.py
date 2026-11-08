"""Власний «спечений» формат InkReel (.ikr): уже пораховані чорно-білі кадри без втрат, із довільним доступом і звуком.

Навіщо. Рівні 4-5 і «Символи» рахуються довго, а результат - це лише біти. Файл .ikr зберігає саме їх: його можна відкрити в InkReel
як відео «із замочком» (ніякого перерахунку й набиття буфера: програвання, перемотка, масштаб, експорт будь-якого моменту чи кадра в
будь-який формат працюють одразу), а з метаданих повернутись у звичайний режим для того ж відео (вихідний файл і налаштування).

Розкладка файлу:
    8 байт   b"INKREEL\\x00"
    u32      довжина заголовка (LE)
    заголовок  JSON UTF-8: format, version, width, height, fps [num, den], key, params, src, ... (усе, що відомо на початку)
    кадри    запис на кадр. Кодек "ctx" (версія 2, типовий): контекстне арифметичне кодування за сусідами в кадрі й у попередньому (benvid/ikrctx.py;
             приблизно вдвічі менше за zlib); кодек "zlib" (версія 1 і запасний): zlib(упаковані біти кадра) - ключовий кадр (кожен key-й) - або zlib(XOR
             із попереднім кадром). Кодек записано в заголовку ("codec"), файли без нього - zlib
    звук     (необов'язково) готовий AAC/Opus-потік як є
    індекс   N записів (u64 зсув, u32 довжина)
    хвіст    JSON UTF-8: frames, index [зсув, довжина], audio [зсув, довжина, розширення] | null
    трейлер  20 байт: b"IKR!" + u64 зсув хвоста + u64 довжина хвоста
Запис іде в «<файл>.part» і перейменовується лише на close(): обірваний експорт не лишає битого файлу. Довільний доступ: кадр
розпаковується від найближчого ключового (до key-1 XOR), а послідовне читання йде без повторів.
Кадр - 1 біт на піксель, рядки по (w+7)//8 байтів (як np.packbits(..., axis=-1)); 1 - біле.
"""
from __future__ import annotations

import json
import os
import struct
import threading
import time
import zlib

import numpy as np

from . import ikrctx

MAGIC = b"INKREEL\x00"
TRAILER = b"IKR!"
FORMAT = "inkreel-baked"
VERSION = 2
EXT = ".ikr"
CODECS = ("ctx", "zlib")
KEY = 30                       # ключовий кадр кожні KEY кадрів: далі XOR із попереднім (де візерунок тримався - майже самі нулі)
_TRL = struct.Struct("<4sQQ")
_REC = np.dtype([("off", "<u8"), ("len", "<u4")])


class BakeError(ValueError):
    """файл не є .ikr, обірваний або пошкоджений"""


def is_baked(path: str) -> bool:
    """за ВМІСТОМ: починається з підпису формату (розширення часто губиться)"""
    try:
        with open(path, "rb") as f:
            return f.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


class BakeWriter:
    """послідовний запис кадрів. add_packed(кадр (h, (w+7)//8) uint8) або add_mask((h, w) {0,1}); set_audio_file(шлях) - звук;
    close() -> dict(frames, bytes, path); abort() - прибрати недописаний файл"""

    def __init__(self, path: str, w: int, h: int, fps_num: int, fps_den: int, meta: dict | None = None, key: int = KEY, level: int = 4,
                 codec: str = "auto"):
        if codec not in ("auto",) + CODECS:
            raise ValueError(f"невідомий кодек {codec!r}")
        self.codec = ("ctx" if ikrctx.available() else "zlib") if codec == "auto" else codec
        if self.codec == "ctx" and not ikrctx.available():
            raise ValueError("кодек ctx потребує numba")
        self.path, self.w, self.h = path, int(w), int(h)
        self.wb = (self.w + 7) // 8
        self.fps = (int(fps_num), int(fps_den))
        self.key, self.level = max(1, int(key)), level
        self._tmp = path + ".part"
        self._f = open(self._tmp, "wb")
        self._idx: list[tuple[int, int]] = []
        self._prev: np.ndarray | None = None            # zlib: попередній кадр упакований; ctx: попередній кадр маскою (h, w)
        self._st: np.ndarray | None = None              # ctx: лічильники ймовірностей поточного проміжку
        self._audio: tuple[str, str] | None = None
        head = dict(format=FORMAT, version=VERSION if self.codec == "ctx" else 1, codec=self.codec, width=self.w, height=self.h, fps=list(self.fps),
                    key=self.key, created=time.strftime("%Y-%m-%dT%H:%M:%S"), **(meta or {}))
        hb = json.dumps(head, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._f.write(MAGIC + struct.pack("<I", len(hb)) + hb)
        self._closed = False

    @property
    def frames(self) -> int:
        return len(self._idx)

    def add_packed(self, pk: np.ndarray):
        pk = np.ascontiguousarray(pk, np.uint8)
        if pk.shape != (self.h, self.wb):
            raise ValueError(f"кадр {pk.shape}, а очікується {(self.h, self.wb)}")
        i = len(self._idx)
        if self.codec == "ctx":
            self._add_ctx(np.unpackbits(pk, axis=-1, count=self.w), i)
            return
        d = pk if (i % self.key == 0 or self._prev is None) else np.bitwise_xor(pk, self._prev)
        z = zlib.compress(d.tobytes(), self.level)
        self._idx.append((self._f.tell(), len(z)))
        self._f.write(z)
        self._prev = pk

    def _add_ctx(self, m: np.ndarray, i: int):
        if i % self.key == 0 or self._st is None:                 # ключовий кадр: лічильники й попередній кадр - з нуля, проміжок незалежний
            self._st, self._prev = ikrctx.new_state(), None
        z = ikrctx.encode(m, self._prev, self._st)
        self._idx.append((self._f.tell(), len(z)))
        self._f.write(z)
        self._prev = m

    def add_mask(self, m01: np.ndarray):
        m = np.asarray(m01)
        if m.shape != (self.h, self.w):
            raise ValueError(f"кадр {m.shape}, а очікується {(self.h, self.w)}")
        if self.codec == "ctx":
            self._add_ctx((m > 0).astype(np.uint8), len(self._idx))
        else:
            self.add_packed(np.packbits(m > 0, axis=-1))

    def set_audio_file(self, path: str, ext: str = ".m4a"):
        """звук, що вже закодовано (AAC/Opus...): кладеться в кінець файлу як є"""
        self._audio = (path, ext)

    def close(self) -> dict:
        if self._closed:
            return dict(frames=len(self._idx), bytes=os.path.getsize(self.path), path=self.path)
        f = self._f
        audio = None
        if self._audio and os.path.isfile(self._audio[0]):
            off = f.tell()
            with open(self._audio[0], "rb") as a:
                while True:
                    b = a.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            if f.tell() > off:
                audio = [off, f.tell() - off, self._audio[1]]
        io = f.tell()
        arr = np.zeros(len(self._idx), _REC)
        if self._idx:
            arr["off"] = [o for o, _ in self._idx]
            arr["len"] = [n for _, n in self._idx]
        f.write(arr.tobytes())
        tail = json.dumps(dict(frames=len(self._idx), index=[io, arr.nbytes], audio=audio), separators=(",", ":")).encode("utf-8")
        to = f.tell()
        f.write(tail)
        f.write(_TRL.pack(TRAILER, to, len(tail)))
        f.flush()
        f.close()
        self._closed = True
        os.replace(self._tmp, self.path)
        return dict(frames=len(self._idx), bytes=os.path.getsize(self.path), path=self.path)

    def abort(self):
        if not self._closed:
            try:
                self._f.close()
            except OSError:
                pass
            self._closed = True
        for p in (self._tmp,):
            try:
                os.remove(p)
            except OSError:
                pass


class BakeReader:
    """читання з довільним доступом; кожен екземпляр має свій дескриптор і свій кеш (для іншого потоку - clone())"""

    def __init__(self, path: str, _shared=None):
        self.path = path
        self._f = open(path, "rb")
        self._lock = threading.Lock()
        self._last: tuple[int, np.ndarray] | None = None
        self._cx: tuple[int, np.ndarray, np.ndarray] | None = None      # ctx: (номер кадра, його маска, лічильники після нього)
        if _shared is not None:
            self.head, self.tail, self.index = _shared
        else:
            try:
                self.head, self.tail, self.index = self._parse()
            except Exception:
                self._f.close()
                raise
        self.w, self.h = int(self.head["width"]), int(self.head["height"])
        self.wb = (self.w + 7) // 8
        self.key = int(self.head.get("key", KEY))
        self.codec = str(self.head.get("codec", "zlib"))
        self.frames = int(self.tail["frames"])
        num, den = self.head["fps"]
        self.fps = num / den
        self.fps_frac = (int(num), int(den))

    def _parse(self):
        f = self._f
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size < len(MAGIC) + 4 + _TRL.size:
            raise BakeError("файл замалий для .ikr")
        f.seek(0)
        if f.read(len(MAGIC)) != MAGIC:
            raise BakeError("це не файл InkReel (.ikr)")
        hl = struct.unpack("<I", f.read(4))[0]
        if hl > 1 << 24:
            raise BakeError("пошкоджений заголовок")
        try:
            head = json.loads(f.read(hl).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            raise BakeError("пошкоджений заголовок") from e
        if head.get("format") != FORMAT:
            raise BakeError("це не файл InkReel (.ikr)")
        if int(head.get("version", 0)) > VERSION or str(head.get("codec", "zlib")) not in CODECS:
            raise BakeError("файл створено новішою версією InkReel")
        if str(head.get("codec", "zlib")) == "ctx" and not ikrctx.available():
            raise BakeError("для цього файлу потрібна numba (pip install numba)")
        f.seek(size - _TRL.size)
        tag, to, tl = _TRL.unpack(f.read(_TRL.size))
        if tag != TRAILER or to + tl + _TRL.size != size:
            raise BakeError("файл обірваний або пошкоджений (нема кінця)")
        f.seek(to)
        try:
            tail = json.loads(f.read(tl).decode("utf-8"))
            io, il = tail["index"]
            f.seek(io)
            raw = f.read(il)
        except Exception as e:  # noqa: BLE001
            raise BakeError("пошкоджений індекс") from e
        n = int(tail["frames"])
        if il != n * _REC.itemsize or len(raw) != il or n < 1:
            raise BakeError("пошкоджений індекс")
        index = np.frombuffer(raw, _REC).copy()
        if int((index["off"].astype(np.uint64) + index["len"].astype(np.uint64)).max()) > size:
            raise BakeError("індекс вказує за межі файлу")
        return head, tail, index

    # ------------------------------------------------------------------
    @property
    def meta(self) -> dict:
        return self.head

    def info(self) -> dict:
        """як video_info: fps, width, height, frames, duration"""
        return dict(fps=self.fps, width=self.w, height=self.h, frames=self.frames, duration=self.frames / self.fps if self.fps else 0.0)

    def clone(self) -> "BakeReader":
        return BakeReader(self.path, _shared=(self.head, self.tail, self.index))

    def close(self):
        try:
            self._f.close()
        except OSError:
            pass

    def _raw(self, i: int) -> np.ndarray:
        off, ln = int(self.index["off"][i]), int(self.index["len"][i])
        self._f.seek(off)
        try:
            d = zlib.decompress(self._f.read(ln))
        except zlib.error as e:
            raise BakeError(f"кадр {i}: пошкоджені дані") from e
        if len(d) != self.h * self.wb:
            raise BakeError(f"кадр {i}: неправильний розмір")
        return np.frombuffer(d, np.uint8).reshape(self.h, self.wb)

    def packed(self, i: int) -> np.ndarray:
        """кадр i -> (h, (w+7)//8) uint8 (новий масив, його можна міняти)"""
        if not 0 <= i < self.frames:
            raise IndexError(i)
        if self.codec == "ctx":
            return np.packbits(self._ctx_mask(i), axis=-1)
        with self._lock:
            k = i - i % self.key
            cur = None
            if self._last is not None:
                li, la = self._last
                if k <= li <= i:                                  # той самий ключовий проміжок: продовжуємо ланцюжок з кешу
                    cur, k = la, li + 1
                    if li == i:
                        return la.copy()
            if cur is None:
                cur = self._raw(k).copy()
                k += 1
            for j in range(k, i + 1):
                cur = np.bitwise_xor(cur, self._raw(j))
            self._last = (i, cur)
            return cur.copy()

    def _ctx_rec(self, i: int) -> bytes:
        off, ln = int(self.index["off"][i]), int(self.index["len"][i])
        self._f.seek(off)
        return self._f.read(ln)

    def _ctx_mask(self, i: int) -> np.ndarray:
        """кадр i кодека ctx: розпаковка від найближчого ключового (або з кешу, якщо це продовження того ж проміжку); -> новий масив (h, w) {0,1}"""
        with self._lock:
            k = i - i % self.key
            prev = st = None
            if self._cx is not None:
                li, lm, ls = self._cx
                if k <= li <= i:
                    if li == i:
                        return lm.copy()
                    prev, st, k = lm, ls, li + 1
            if prev is None:
                prev, st = None, ikrctx.new_state()
            try:
                for j in range(k, i + 1):
                    if j % self.key == 0:
                        prev, st = None, ikrctx.new_state()
                    prev = ikrctx.decode(self._ctx_rec(j), self.h, self.w, prev, st)
            except ValueError as e:
                self._cx = None
                raise BakeError(f"кадр {j}: {e}") from e
            self._cx = (i, prev, st)
            return prev.copy()

    def mask(self, i: int) -> np.ndarray:
        """кадр i -> (h, w) uint8 {0,1}"""
        if self.codec == "ctx":
            if not 0 <= i < self.frames:
                raise IndexError(i)
            return self._ctx_mask(i)
        return np.unpackbits(self.packed(i), axis=-1, count=self.w)

    def audio(self):
        """-> (bytes, розширення) або None"""
        a = self.tail.get("audio")
        if not a:
            return None
        with self._lock:
            self._f.seek(int(a[0]))
            return self._f.read(int(a[1])), a[2]

    def audio_to_file(self, folder: str) -> str | None:
        """вбудований звук -> файл у теці (для ffmpeg/програвача); None, якщо звуку нема"""
        a = self.audio()
        if a is None:
            return None
        os.makedirs(folder, exist_ok=True)
        p = os.path.join(folder, "audio" + a[1])
        with open(p, "wb") as f:
            f.write(a[0])
        return p


def open_baked(path: str) -> BakeReader:
    return BakeReader(path)


def picked_frames(n: int, fps: float, limit: float | None, start: int = 0, end: int | None = None) -> list[int]:
    """номери кадрів файлу для експорту: діапазон [start, end) і необов'язковий ліміт fps за правилом video.fps_ratio
    (вихідний кадр k = кадр int(k * ratio + 0.5) - кадри за часом, тривалість не міняється)"""
    from .video import fps_ratio
    end = n if end is None else min(n, int(end))
    a = max(0, min(int(start), end))
    ratio = fps_ratio(fps, limit)
    if ratio == 1.0:
        return list(range(a, end))
    out_n = max(1 if end > 0 else 0, int(round(end / ratio)))
    out_a = min(out_n, int(round(a / ratio)))
    last = max(0, end - 1)
    return [min(last, int(k * ratio + 0.5)) for k in range(out_a, out_n)]


def packed_chunks(rd: BakeReader, idxs: list[int], chunk: int = 32, cancel=None, pause=None, progress=None):
    """генератор шматків (n, h, (w+7)//8) uint8 у порядку idxs - так само, як video.packed_video (експорт бере їх звідси без розрахунку)"""
    total = max(1, len(idxs))
    for s in range(0, len(idxs), chunk):
        if pause is not None:
            while pause.is_set():
                if cancel is not None and cancel.is_set():
                    break
                time.sleep(0.05)
        if cancel is not None and cancel.is_set():
            from .video import Cancelled
            raise Cancelled()
        part = [rd.packed(i) for i in idxs[s:s + chunk]]
        if progress:
            progress(min(0.999, (s + len(part)) / total))
        yield np.stack(part)


class BakedSource:
    """джерело для плеєра: кадри вже готові (kind='video', baked=True). read() віддає ЗАПАКОВАНІ біти, не BGR - плеєр кладе їх у буфер
    без обробки (див. Player._feeder); для кадру як маски - mask()"""
    baked = True

    def __init__(self, path: str, reader: BakeReader | None = None):
        self.path = path
        self.rd = reader or BakeReader(path)
        i = self.rd.info()
        self.info = dict(i, kind="video", starts=None, baked=True, locked=True, note="")

    def packed(self, idx: int) -> np.ndarray | None:
        try:
            return self.rd.packed(int(idx))
        except (IndexError, BakeError):
            return None

    def read(self, idx: int):
        pk = self.packed(idx)
        return None if pk is None else np.unpackbits(pk, axis=-1, count=self.rd.w)

    def reset(self):
        pass

    def clone(self) -> "BakedSource":
        return BakedSource(self.path, self.rd.clone())

    def close(self):
        self.rd.close()
