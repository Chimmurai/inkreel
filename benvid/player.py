"""Плеєр із попереднім буфером (як на YouTube), без tkinter.

Кадри обробляються пулом процесів наперед і складаються в пам'ять (1 біт на піксель, без обмеження за часом - ліміт
лише половина оперативної пам'яті); відтворення йде з буфера за годинником, тому fps точний, а інтерфейс нічого не
рахує. Якщо буфер закінчився - «буферизація»: годинник стоїть, доки не накопичиться запас.

Швидкість: кілька потоків-декодерів (кожен зі своїм VideoCapture, шматками по 8 кадрів), кадри до воркерів ідуть через
спільну пам'ять (кільце слотів), а не через pickle. Готові кадри лишаються: перемотка назад і експорт їх не рахують знову.
"""
from __future__ import annotations

import contextlib
import os
import threading
import time
from concurrent.futures import CancelledError
from multiprocessing import shared_memory

import cv2
import numpy as np

from .audio import AudioOut
from .media import open_source
from .core import Params, binarize
from .glyph import WARM as GLYPH_WARM, WARM_CHEAP as GLYPH_CHEAP
from .video import (_frame_job, _frame_job_shm, _ink_warm_job, _job, default_workers, fps_ratio, glyph_cell, kill_pool, make_pool, out_size,
                    ram_bytes, workers_for)

START_S, CHUNK, MAX_STRIDE = 1.0, 8, 8
PRE_K = 40               # скільки кадрів перед шматком (у порядку, в якому їх рахують) віддавати на розігрів часової пам'яті при ліміті к/с
CHUNK5 = 24              # рівень 5 без пулу (GIF, один процес): довші шматки - менше витрат на розігрів часової пам'яті
CHUNK6 = 32              # рівень 6 (нейро): кожен шматок починається з розігріву згладжування (кілька кадрів лише на декодування), тож довші шматки
RAMP0, RAMP_STEP = 3, 0.8   # новий пул: спершу 3 процеси, далі +1 кожні 0.8 с (кожен новий процес - секунди імпорту й завантаження ядер)


class _Ring:
    """кільце слотів у спільній пам'яті для кадрів BGR одного розміру"""

    def __init__(self, w, h, n):
        self.w, self.h, self.n = w, h, n
        self.shm = shared_memory.SharedMemory(create=True, size=max(1, w * h * 3 * n))
        self.name = self.shm.name
        self.arr = np.ndarray((n, h, w, 3), np.uint8, buffer=self.shm.buf)
        self.free = list(range(n))
        self.out = 0
        self.dead = False

    def destroy(self):
        self.arr = None
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:  # noqa: BLE001
            pass


class _Stream:
    """спільна пам'ять одного шматка «Ультри» в пулі: воркер кладе туди кожен готовий кадр і збільшує лічильник, плеєр забирає кадри
    одразу (а не чекає, поки дорахується весь шматок - на слабкому ПК це пів хвилини простою), і може скасувати шматок прапорцем"""
    HEAD = 32

    def __init__(self, gen, idxs, w, h):
        self.gen, self.idxs, self.w, self.h = gen, list(idxs), w, h
        self.fb = h * ((w + 7) // 8)
        self.shm = shared_memory.SharedMemory(create=True, size=self.HEAD + max(1, len(self.idxs)) * self.fb)
        self.name = self.shm.name
        self.hdr = np.ndarray((4,), np.int64, buffer=self.shm.buf)
        self.hdr[:] = 0
        self.sent = 0                                   # скільки кадрів уже віддано плеєру

    def ready(self) -> int:
        return int(self.hdr[0]) if self.hdr is not None else self.sent

    def frame(self, k):
        return np.array(np.ndarray((self.h, (self.w + 7) // 8), np.uint8, buffer=self.shm.buf, offset=self.HEAD + k * self.fb))

    def cancel(self):
        if self.hdr is not None:
            self.hdr[1] = 1

    def destroy(self):
        self.hdr = None
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:  # noqa: BLE001
            pass


class Player:
    def __init__(self, workers: int | None = None, feeders: int | None = None, ink: int | None = None,
                 nb_threads: int | None = None):
        """feeders - потоки, що декодують і (на рівні 5) самі рахують кадри; ink - скільки кадрів «Ультри» рахується одночасно;
        nb_threads - скільки потоків numba дозволено кожному такому потоку (None - усі). Типові значення - за кількістю ядер;
        параметри потрібні для замірів (benvid.bench)"""
        self.workers = max(1, workers or default_workers())
        cpu = os.cpu_count() or 2
        self.feeders = max(1, feeders or (1 if self.workers == 1 else 2 if cpu < 8 else 4))
        self._ink_n = max(1, min(ink or max(1, min(4, cpu - 1)), self.feeders))
        self._nb_threads = nb_threads
        self._hold = False
        self.params = Params()
        self.info = None
        self.path = None
        self.size_key = "240p"
        self.size = (2, 2)
        self.fps, self.n = 24.0, 0
        self._lock = threading.RLock()
        self._slot_cv = threading.Condition(self._lock)       # вільний слот кільця: потоки-подавачі чекають, а не опитують кожну мілісекунду
        self._ex = None
        self._ex_n = 0
        self._ex_t0 = 0.0
        self._urgent = True                                  # після відкриття/перемотки перший кадр у поточній позиції рахується терміново (див. _claim)
        self.ramp = True                                     # плавний розгін пулу; вимикається лише для замірів
        self._inproc = threading.Lock()
        self._ink_gate = threading.Semaphore(self._ink_n)    # рівень 5 без пулу (GIF): кілька кадрів одночасно (cv2/numba не тримають GIL)
        self._inflight = 0                                   # шматки «Ультри» відео, що зараз у пулі процесів
        self._futs: set = set()
        self._ink_ready = False                              # ядра «Ультри» прогріто в одному процесі пулу (далі решта лише підвантажує кеш)
        self._ink_warm = None
        self._ink_fail = 0                                   # підряд невдалих шматків «Ультри»: після кількох - рахуємо без пулу
        self.audio = AudioOut()
        self._aud_on = False
        self._src = None
        self._src_id = 0
        self.kind = "video"
        self.baked = False                                   # режим із замочком: відкрито файл .ikr - кадри готові, нічого не рахується (benvid.bake)
        self.baked_meta: dict = {}
        self._seed = None                                    # (id джерела, .ikr, кадр оригіналу, кінець): готові кадри замість розрахунку (open(seed=...))
        self._starts = None
        self._gen = 0
        self._frames: dict[int, np.ndarray] = {}
        self._have = np.zeros(1, bool)
        self._pend = np.zeros(1, bool)
        self._head = 0
        self._state = "paused"           # paused | playing | buffering
        self._want_play = False
        self._suspended = False
        self._base_frame, self._base_t = 0.0, 0.0
        self._alive = True
        self._done_times: list[float] = []
        self._shown = -1
        self._fails: dict[int, int] = {}
        self._stride = 1                  # слабкий ПК: рахуємо кожен N-й кадр, показуємо попередній
        self._gap = 1                     # найбільший проміжок між порахованими кадрами в буфері (≥ кроку; 1 - буфер без дірок)
        self._limit = None                # ліміт кадрів на секунду (None - як в оригіналі): у буфер рахуються лише кадри, що потраплять у відео
        self._ratio = 1.0                 # у скільки разів вихідний ряд рідший за кадри файлу (video.fps_ratio)
        self._behind = False              # черга дораховує початок відео позаду позиції (див. _claim)
        self._wprev = None
        self._want = None                 # bool по кадрах файлу: потрібні при ліміті (None - усі)
        self._gb = 1                      # найбільший проміжок між потрібними кадрами (при ліміті 24 з 60 - 3)
        self._cands: dict = {}            # крок -> номери кадрів, які рахуємо (кожен крок-й з потрібних)
        self._streams: list = []          # шматки «Ультри» в пулі, чиї готові кадри забираємо не чекаючи кінця (_Stream)
        self._stream_ev = threading.Event()
        self._stalls = 0
        self._start_s = START_S
        self.loop = False
        self._ring = None
        self.mem_cap = max(1 << 30, ram_bytes() // 2)
        self._threads = [threading.Thread(target=self._feeder, args=(i,), daemon=True) for i in range(self.feeders)]
        for t in self._threads:
            t.start()
        self._last_done = self._ui_t = time.perf_counter()    # для сторожа: коли востаннє готовий кадр і коли востаннє відповів інтерфейс
        self._stall_s = 8.0                                    # скільки секунд очікування кадрів сторож уважає застоєм
        threading.Thread(target=self._watch, daemon=True, name="benvid-watch").start()
        threading.Thread(target=self._pump_streams, daemon=True, name="benvid-streams").start()

    # ------------------------------------------------------------ керування
    def warm(self):
        """створити й прогріти пул процесів - у фоновому потоці: старт процесів на Windows займає до секунди, а викликають це з GUI"""
        threading.Thread(target=self._pool, daemon=True).start()

    def open(self, path: str, size_key: str | None = None, start: int = 0, seed: dict | None = None) -> dict:
        """start - кадр, з якого починається буфер (розморозка: там, де зупинились, а не з нуля).
        seed - dict(path=файл .ikr, base=кадр оригіналу, якому відповідає перший кадр файлу): уже пораховані кадри беруться з нього замість
        розрахунку. Діє лише коли розмір файлу збігається з розміром результату, а ліміту к/с нема (інакше буфер був би інший)"""
        src = open_source(path)
        info = src.info
        with self._lock:
            if self._src is not None:
                self._src.close()
                try:
                    from .video import _CAPS
                    _CAPS.release_path(self.path)
                except Exception:  # noqa: BLE001
                    pass
                if self.params.method == 6 and not getattr(src, "baked", False):        # друге відео не повинно рахуватись на сеансі, «обкладеному» першим (див. student.recycle)
                    threading.Thread(target=self._recycle_neuro, daemon=True, name="benvid-recycle").start()
            self._src = src
            self._src_id += 1
            self.baked = bool(getattr(src, "baked", False))
            self.baked_meta = dict(src.rd.meta) if self.baked else {}
            if self.baked:
                size_key = "Як у джерела"
            if size_key:
                self.size_key = size_key
            self.path, self.info, self.kind = path, info, info["kind"]
            self._starts = info.get("starts")
            self.fps = info["fps"] or 24.0
            self.n = max(1, info["frames"])
            self._fails = {}
            self._ink_fail = 0
            self._stride, self._stalls, self._start_s = 1, 0, START_S
            self._build_want()
            self._aud_on = False
            if self.baked:
                threading.Thread(target=self._load_baked_audio, args=(src.clone(),), daemon=True).start()
            elif info["kind"] == "video":
                from .video import find_ffmpeg
                threading.Thread(target=self.audio.load, args=(path, find_ffmpeg()), daemon=True).start()
            else:
                self.audio.stop()
                self.audio.pcm = None
            self._state, self._want_play, self._base_frame = "paused", False, 0.0
            self._apply_size()
            st = int(min(max(0, start), self.n - 1))
            self._seed = self._check_seed(seed) if (seed and not self.baked and self._want is None and info["kind"] == "video") else None
            self._restart(st)
            self._base_frame = float(st)
        return info

    @staticmethod
    def _recycle_neuro():
        try:
            from . import student
            student.recycle()
        except Exception:  # noqa: BLE001
            pass

    def _check_seed(self, seed: dict):
        """-> (id джерела, шлях .ikr, кадр оригіналу для першого кадру файлу, кінець діапазону) або None, коли файл не підходить"""
        from .bake import BakeError, BakeReader
        try:
            rd = BakeReader(str(seed["path"]))
        except (OSError, BakeError, KeyError):
            return None
        try:
            base = int(seed.get("base") or 0)
            if (rd.w, rd.h) != tuple(self.size) or base < 0 or base >= self.n:
                return None
            return (self._src_id, str(seed["path"]), base, min(self.n, base + rd.frames), self.params.to_dict(), tuple(self.size))
        finally:
            rd.close()

    def _seed_rest(self, sd, busy):
        """ще не взяті кадри з діапазону файлу .ikr (до 64 підряд) або None"""
        if sd is None or sd[0] != self._src_id:
            return None
        free = np.flatnonzero(~busy[sd[2]:sd[3]])
        if not len(free):
            return None
        out = [sd[2] + int(free[0])]
        for v in free[1:64]:
            if sd[2] + int(v) != out[-1] + 1:
                break
            out.append(sd[2] + int(v))
        return out

    def _feed_seed(self, gen, idxs, sd):
        """кадри шматка беруться з пораховано-спеченого файлу, а не рахуються (див. open(seed=...))"""
        from .bake import BakeError, BakeReader
        done, rd = 0, None
        try:
            rd = BakeReader(sd[1])
            for idx in idxs:
                if gen != self._gen or not self._alive:
                    break
                self._done(gen, idx, rd.packed(idx - sd[2]))
                done += 1
        except (OSError, BakeError, IndexError):
            with self._lock:                                  # файл зник чи пошкоджений: далі кадри рахуються звичайним шляхом
                if self._seed is sd:
                    self._seed = None
        finally:
            if rd is not None:
                rd.close()
        if done < len(idxs):
            self._release_pend(gen, idxs[done:])

    def _load_baked_audio(self, src):
        """звук із файлу .ikr: вийняти потік у тимчасову теку, декодувати, прибрати"""
        import shutil
        import tempfile
        from .video import find_ffmpeg
        d = tempfile.mkdtemp(prefix="ikr_snd_")
        try:
            p = src.rd.audio_to_file(d)
            if p:
                self.audio.load(p, find_ffmpeg())
            else:
                self.audio.stop()
                self.audio.pcm = None
                self.audio.status = "немає доріжки"
        except Exception:  # noqa: BLE001
            pass
        finally:
            src.close()
            shutil.rmtree(d, ignore_errors=True)

    def hold(self, v: bool):
        """призупинити/відновити взяття нових кадрів у роботу (на час замірів, щоб вони не ділили процесор з буфером)"""
        with self._lock:
            self._hold = bool(v)

    def auto_pick(self) -> str:
        """підібрати розмір під цей ПК і ПОТОЧНИЙ рівень якості (блокує 1-2 с: викликати з фонового потоку)"""
        from .autosize import pick
        with self._lock:
            src, info, workers, method, glyph, gdens = self._src, self.info, self.workers, int(self.params.method), bool(self.params.glyph), int(self.params.gdens)
        if src is None:
            return "240p"
        if self.baked:
            return "Як у джерела"
        self.hold(True)
        try:
            c = src.clone()
            try:
                fr = c.read(min(info["frames"] - 1, max(0, info["frames"] // 3)))
            finally:
                c.close()
            if fr is None:
                return "240p"
            if info["kind"] == "image":
                return "Як у джерела" if info["height"] <= 1080 else "1080p"
            in_proc = method == 5 and not self._ink_pool()          # «Ультра» GIF рахується в потоках цього процесу, решта - у пулі
            if method == 6:
                workers = 1                                         # IRNM1: мережа одна, кадри ідуть через неї по черзі
            return pick(fr, self.fps, workers, src_h=info["height"], method=method, threads=self._ink_n if in_proc else 1, glyph=glyph, gdens=gdens)
        finally:
            self.hold(False)

    def _apply_size(self):
        if self.info:
            self.size = (int(self.info["width"]), int(self.info["height"])) if self.baked else out_size(self.info["width"], self.info["height"], self.size_key, glyph_cell(self.params))

    def _restart(self, frame: int):
        """скинути буфер і почати обробку з кадру frame"""
        self._gen += 1
        self._cancel_inflight()
        if self._ex is not None and (self.params.method == 6 or self.baked):
            # «Нейро» рахується в цьому процесі, а готовий .ikr нічого не рахує: пул процесів Ink/Ultra тут простоює (0% процесора), але кожен
            # його процес тримає пам'ять і контекст відеокарти - разом із моделлю на DirectML це вичерпувало відеопам'ять (повільнішання, потім
            # помилки DmlCommandRecorder і відкат на процесор). Тож пул закриваємо одразу; наступний режим Ink створить новий
            ex, self._ex = self._ex, None
            self._ink_ready, self._ink_warm = False, None
            kill_pool(ex)
        self._frames.clear()
        self._have = np.zeros(max(1, self.n), bool)
        self._pend = np.zeros(max(1, self.n), bool)
        self._head = int(frame)
        self._behind = False
        self._shown = -1
        self._urgent = True
        self._gap = self._gap_of(self._stride)          # нові кадри йдуть із поточним кроком

    # ------------------------------------------------------------ ліміт кадрів на секунду
    def _build_want(self):
        """які кадри файлу потрібні при ліміті к/с: ті самі, що бере експорт (вихідний k = кадр int(k*ratio + 0.5), див. video.fps_ratio).
        Без ліміту (а також для GIF, фото й спеченого файлу) - усі"""
        r = 1.0
        if self._limit and self.kind == "video" and not self.baked and self._starts is None:
            r = fps_ratio(self.fps, self._limit)
        self._ratio = r
        self._cands = {}
        self._wprev = None
        if r == 1.0:
            self._want, self._gb = None, 1
            return
        n = max(1, self.n)
        k = np.arange(max(1, int(round(n / r))))
        want = np.zeros(n, bool)
        want[np.minimum(n - 1, (k * r + 0.5).astype(np.int64))] = True
        wi = np.flatnonzero(want)
        self._want = want
        self._wprev = np.maximum.accumulate(np.where(want, np.arange(n), 0))     # для кожного кадру - останній потрібний не пізніше за нього
        self._gb = int(np.diff(wi).max()) if len(wi) > 1 else 1

    def _gap_of(self, st: int) -> int:
        """найбільший проміжок між порахованими кадрами при кроці st (крок - у кадрах ВИХІДНОГО ряду, тобто потрібних)"""
        return max(1, int(st)) * self._gb

    def _cand(self, st: int):
        """номери кадрів, які рахуємо при кроці st: кожен st-й із потрібних (без ліміту - кожен st-й кадр файлу)"""
        st = max(1, int(st))
        c = self._cands.get(st)
        if c is None or (len(c) and c[-1] >= self.n):                       # (лічильник кадрів міг зменшитись - читання не вдалось)
            wi = np.arange(self.n) if self._want is None else np.flatnonzero(self._want[:self.n])
            if len(self._cands) > 4:
                self._cands = {}
            c = self._cands[st] = wi[::st]
        return c

    def _floor_cand(self, pos: int, st: int | None = None) -> int:
        """кадр, що має бути порахований, щоб показати позицію pos: останній з тих, що рахуємо, не пізніше за pos"""
        c = self._cand(self._stride if st is None else st)
        if not len(c):
            return int(pos)
        return int(c[max(0, int(np.searchsorted(c, int(pos), "right")) - 1)])

    def set_fps_limit(self, limit):
        """ліміт кадрів на секунду (None - як в оригіналі). Далі в буфер рахуються лише кадри, що потраплять у відео з таким лімітом
        (при 24 з 60 - 2,5 раза менше роботи), а на екрані тримається останній порахований. Уже готові кадри лишаються"""
        with self._lock:
            lim = float(limit) if limit else None
            if lim == self._limit:
                return
            old = (self._ratio, self._want is None)
            self._limit = lim
            self._build_want()
            if (self._ratio, self._want is None) == old:
                return
            self._gap = max(self._gap, self._gap_of(self._stride))     # дірки між кадрами не вважаємо розривом буфера
            self._shown = -1                                           # перемалювати поточний кадр за новим рядом
            self._behind = False
            self._head = self._floor_cand(int(self._pos_nolock()))     # і почати від кадру, що показується на поточній позиції

    def set_size(self, key: str):
        with self._lock:
            if self.baked:                                  # розмір спеченого файлу сталий
                return
            if key == self.size_key and self.info and out_size(self.info["width"], self.info["height"], key, glyph_cell(self.params)) == self.size \
                    and self._frames:
                return
            self.size_key = key
            self._apply_size()
            if self._seed is not None and tuple(self.size) != self._seed[5]:
                self._seed = None                           # інший розмір: кадри файлу .ikr більше не підходять, рахуємо наново
            self._stride, self._stalls, self._start_s = 1, 0, START_S
            self._restart(int(self._pos_nolock()))
            self._to_buffering()

    def set_params(self, params: Params):
        with self._lock:
            if self.baked:                                  # нічого не перераховується; налаштування лише запам'ятовуються (повернення до звичайного режиму)
                self.params = params
                return
            if bool(params.glyph) != bool(self.params.glyph):
                self._ink_ready, self._ink_warm = False, None      # чорнило й символи мають різні ядра: прогрів - наново (з кешу це секунда)
            self.params = params
            old = tuple(self.size)
            self._apply_size()                              # «Символи»: найменший розмір - одна клітинка знака при поточній крупності, тож крупність може змінити розмір
            if self._seed is not None and (params.to_dict() != self._seed[4] or tuple(self.size) != self._seed[5]):
                self._seed = None                           # інші налаштування: готові кадри файлу .ikr більше не підходять
            if tuple(self.size) != old:
                self._stride, self._stalls, self._start_s = 1, 0, START_S
            self._restart(int(self._pos_nolock()))
            self._to_buffering()

    def suspend(self, v: bool):
        """на час експорту не рахувати наперед (щоб не ділити процесор)"""
        with self._lock:
            self._suspended = v
            self._ink_fail = 0                         # обрив пулу при експорті - не «невдалі шматки»
            if v and self._ex is not None:             # експорт має власний пул: цей звільняє процеси й пам'ять (одразу, а не коли доробить)
                kill_pool(self._ex, wait=2.0)
                self._ex = None

    def snapshot(self):
        """(розмір, {кадр: упаковані біти}) - для експорту без повторної обробки"""
        with self._lock:
            return self.size, dict(self._frames)

    def _to_buffering(self):
        if self._want_play:
            self._base_frame = self._pos_nolock()
            self._state = "buffering"

    def _pos_nolock(self) -> float:
        if self._state == "playing":
            if self._starts is not None:             # GIF: кадри різної тривалості
                t = self._starts[int(self._base_frame)] + (time.perf_counter() - self._base_t)
                return float(min(max(0, int(np.searchsorted(self._starts, t, "right")) - 1), self.n - 1))
            return min(self._base_frame + (time.perf_counter() - self._base_t) * self.fps, self.n - 1)
        return self._base_frame

    def _elapsed_end(self) -> bool:
        if self._starts is not None:
            return self._starts[int(self._base_frame)] + (time.perf_counter() - self._base_t) >= self.duration
        return self._base_frame + (time.perf_counter() - self._base_t) * self.fps >= self.n - 1

    @property
    def duration(self) -> float:
        if self._starts is not None:
            return float(self.info["duration"])
        return (self.n - 1) / self.fps if self.fps else 0.0

    def time_of(self, frame: float) -> float:
        if self._starts is not None:
            return float(self._starts[min(max(0, int(frame)), self.n - 1)])
        return frame / self.fps

    def position(self) -> float:
        with self._lock:
            return self._pos_nolock()

    @property
    def state(self) -> str:
        return self._state

    @property
    def playing(self) -> bool:
        return self._want_play

    def play(self):
        with self._lock:
            if not self.info or self._want_play or self.n <= 1:
                return
            if self._base_frame >= self.n - 1:
                self.seek(0)
            self._want_play = True
            self._state = "buffering"
            self._base_frame = self._pos_nolock()

    def pause(self):
        with self._lock:
            self._base_frame = self._pos_nolock()
            self._want_play, self._state = False, "paused"

    def set_muted(self, v: bool):
        with self._lock:
            self.audio.muted = v
            if self._aud_on:                  # перезапустити звук із поточного місця (для winsound mute = тиша)
                self.audio.pause()
                self._aud_on = False

    def set_volume(self, v: float, restart: bool = False):
        """гучність 0..1 (положення повзунка). sounddevice змінює її на льоту; winsound - лише з перезапуском звуку
        з поточного місця (restart=True, коли повзунок відпущено)"""
        with self._lock:
            self.audio.set_volume(v)
            if restart and self._aud_on and self.audio.backend == "ws":
                self.audio.pause()
                self._aud_on = False

    def toggle(self):
        self.pause() if self._want_play else self.play()

    @property
    def stride(self) -> int:
        """скільки кадрів між порахованими на екрані зараз (для підпису «кожен N-й»): 1, якщо попереду кадри вже без дірок -
        навіть коли режим рідкого рахунку вмикався раніше, а потім буфер догнали до повного"""
        g, gb = self._gap, self._gb
        if g <= gb:
            return 1
        try:
            a = max(0, min(self.n - 1, int(self._pos_nolock())))
            b = min(self.n, a + int(2 * self.fps) + 1)
            done = self._have[a:b] if self._want is None else (self._have[a:b] | ~self._want[a:b])
            return 1 if bool(done.all()) else max(1, -(-int(g) // gb))
        except Exception:  # noqa: BLE001
            return int(g)

    def _key(self, i: int) -> int:
        """кадр буфера, що показується на позиції i: точний, а коли в буфері дірки (рідкий режим) - найближчий попередній порахований"""
        if self._wprev is not None and 0 <= i < len(self._wprev):
            i = int(self._wprev[i])                         # з лімітом к/с показуємо кадри ВИХІДНОГО ряду, навіть коли в буфері вже є інші (пораховані до зміни ліміту)
        if self._gap > 1 and not self._have[i]:
            for d in range(1, self._gap):
                if i - d >= 0 and self._have[i - d]:
                    return i - d
        return i

    def _keys(self, idx):
        """те саме для масиву номерів кадрів"""
        g = self._gap
        if self._wprev is not None:
            idx = self._wprev[np.minimum(idx, len(self._wprev) - 1)]
        if g <= 1:
            return idx
        k = idx.copy()
        for d in range(1, g):
            cand = np.maximum(idx - d, 0)
            k = np.where(~self._have[k] & self._have[cand], cand, k)
        return k

    def _avail(self, pos: int, cnt: int) -> bool:
        idx = self._keys(np.arange(pos, min(self.n, pos + cnt)))
        return bool(self._have[idx].all())

    def seek(self, frame: float):
        with self._lock:
            f = int(min(max(0, frame), max(0, self.n - 1)))
            self._base_frame, self._base_t = float(f), time.perf_counter()
            self._shown = -1
            if not self._have[self._key(f)]:
                self._gen += 1                # скасувати шматки, що летять не звідти
                self._cancel_inflight()
                self._pend[:] = False
                self._head = self._floor_cand(f)
                self._behind = False
                self._urgent = True
            self._to_buffering()

    def step(self, d: int):
        self.pause()
        with self._lock:
            if self._want is not None:                       # при ліміті к/с крок - до сусіднього кадру ВИХІДНОГО ряду, а не файлу
                c = self._cand(1)
                j = int(np.searchsorted(c, int(self._pos_nolock()), "right")) - 1
                self.seek(int(c[min(max(0, j) + d, len(c) - 1)]) if d >= 0 or j + d >= 0 else 0)
                return
        self.seek(int(self.position()) + d)

    # ------------------------------------------------------------ кадри й стан
    def _evict(self, pos):
        """пам'ять повна: звільняємо позаду від найстарішого; наперед нічого не викидаємо (рендер просто чекає)"""
        per = max(1, (self.size[0] * self.size[1] + 7) // 8)
        if len(self._frames) * per <= self.mem_cap:
            return
        keep = int(self.mem_cap * 0.9 / per)
        behind = sorted(k for k in self._frames if k < pos - int(2 * self.fps))
        for k in behind[:max(0, len(self._frames) - keep)]:
            del self._frames[k]
            self._have[k] = False

    def update(self):
        """викликати з GUI кожні кілька мс: перемикає playing/buffering і повертає (idx, mask01) нового кадру або None"""
        self._ui_t = time.perf_counter()
        with self._lock:
            pos = int(self._pos_nolock())
            if self._want_play:
                if self._state == "playing" and pos < self.n and not self._have[self._key(min(pos, len(self._have) - 1))]:
                    self._base_frame, self._state = float(pos), "buffering"     # буфер вичерпано
                    self._on_stall()
                if self._state == "buffering":
                    need = max(1, min(int(self._start_s * self.fps), self.n - pos))
                    if self._avail(pos, need):
                        self._base_t, self._state = time.perf_counter(), "playing"
                    elif not self._have[self._key(pos)] and not self._pend[self._floor_cand(pos)] \
                            and self._head > self._floor_cand(pos):
                        self._head = self._floor_cand(pos)                 # самовідновлення: потрібний кадр не в черзі
                if self._state == "playing" and self._elapsed_end():
                    if self.loop and self.n > 1:
                        self.seek(0)
                        pos = 0
                    else:
                        self._base_frame, self._want_play, self._state = float(self.n - 1), False, "paused"
                        pos = self.n - 1
            if self._state == "playing" and not self._aud_on:
                self._aud_on = self.audio.play_from(self.time_of(self._pos_nolock()))   # False: звук ще декодується
            elif self._state == "playing" and self._aud_on and (self._starts is None or self.kind == "video"):
                self.audio.sync((self._base_frame / self.fps if self._starts is None else float(self._starts[min(int(self._base_frame), self.n - 1)]))
                                + (time.perf_counter() - self._base_t))
            elif self._state != "playing" and self._aud_on:
                self.audio.pause()
                self._aud_on = False
            key = self._key(pos)
            if key != self._shown and key in self._frames:
                self._shown = key
                pk = self._frames[key]
                self._evict(pos)
            else:
                return None
        return pos, np.unpackbits(pk, axis=-1, count=self.size[0])

    def _on_stall(self):
        """буфер скінчився під час відтворення: вимагаємо більший запас, а якщо ПК не встигає - рахуємо рідше"""
        self._stalls += 1
        self._start_s = min(10.0, START_S * 2 ** self._stalls)
        r = self.render_fps()
        if self._stalls >= 2 and self.kind == "video" and r > 0 and self._stride < MAX_STRIDE:
            need = int(np.ceil(self.fps / self._ratio / max(r, 1.0) * 1.15))        # з лімітом к/с потрібна швидкість - за вихідним рядом
            if need > self._stride:
                self._stride = min(MAX_STRIDE, need)
                self._gap = max(self._gap, self._gap_of(self._stride))

    def buffer_progress(self):
        """-> (є, потрібно): скільки кадрів із запасу, потрібного перед стартом відтворення, уже порахованo"""
        with self._lock:
            if not self.info or self.n <= 0:
                return 0, 1
            pos = min(int(self._pos_nolock()), self.n - 1)
            need = max(1, min(int(self._start_s * self.fps), self.n - pos))
            idx = self._keys(np.arange(pos, min(self.n, pos + need)))
            return int(self._have[idx].sum()), need

    def waiting_frame(self) -> bool:
        """кадр поточної позиції ще не порахований (після перемотування, зміни розміру чи якості) - на екрані застаріле"""
        with self._lock:
            if not self.info or self.n <= 0:
                return False
            return not self._have[self._key(min(int(self._pos_nolock()), self.n - 1))]

    def refresh(self):
        """показати поточний кадр знову (після експорту, коли на екрані лишився кадр із нього)"""
        with self._lock:
            self._shown = -1

    def buffered_end(self) -> int:
        """останній кадр безперервного буфера від позиції (за кроку >1 порожнини між порахованими кадрами - не дірки)"""
        with self._lock:
            k = self._key(int(self._pos_nolock()))
            if not self._have[k]:
                return k
            tr = np.flatnonzero(self._have[k:self.n])
            br = np.flatnonzero(np.diff(tr) > self._gap)
            return k + int(tr[br[0]] if len(br) else tr[-1])

    def buffered_ranges(self):
        """усі готові відрізки буфера [[перший, останній], ...] - показуємо на смузі перемотування"""
        with self._lock:
            h = self._have[:self.n].astype(np.int8)
        d = np.diff(np.concatenate([[0], h, [0]]))
        out = []
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if out and int(a) - out[-1][1] <= self._gap:           # у рідкому режимі порожнини між кадрами - не дірки
                out[-1][1] = int(b) - 1
            else:
                out.append([int(a), int(b) - 1])
        return out

    @property
    def cached(self) -> int:
        return int(self._have.sum())

    def cached_in(self, a: int, b: int) -> int:
        """скільки потрібних кадрів з діапазону [a, b) уже пораховано (при ліміті к/с - лише ті, що потраплять у відео)"""
        a, b = max(0, a), max(0, b)
        if self._want is None:
            return int(self._have[a:b].sum())
        return int((self._have[a:b] & self._want[a:b]).sum())

    def total_in(self, a: int, b: int) -> int:
        """скільки кадрів із діапазону [a, b) потрібно порахувати (при ліміті к/с - лише ті, що потраплять у відео)"""
        a, b = max(0, a), max(0, b)
        return max(0, min(b, self.n) - a) if self._want is None else int(self._want[a:min(b, self.n)].sum())

    def render_fps(self) -> float:
        """СКІЛЬКИ КАДРІВ У СЕКУНДУ обробляє програма разом - усі потоки й процеси, що віддають готові кадри (один лічильник): за проміжок
        між першим і останнім готовим кадром останніх 3 с (а не «кадри / 3 с», що занижувало швидкість на старті й після перемотки);
        0, коли нове вже секунду не з'являється (буфер повний або стоїть)"""
        now = time.perf_counter()
        t = [x for x in self._done_times if x > now - 3.0]
        if not t or now - t[-1] > 1.0:
            return 0.0
        span = t[-1] - t[0]
        return (len(t) - 1) / span if len(t) >= 3 and span > 0.25 else len(t) / 3.0

    # ------------------------------------------------------------ фон
    def _pool(self):
        """пул процесів (створюється один; перестворюється, коли змінився розмір кадру). Під замком лише облік - сам запуск процесів
        (на Windows це десятки мілісекунд на процес) іде поза ним, інакше інтерфейс, якому теж потрібен замок, підвисав би"""
        with self._lock:
            old = None
            if self._ex is not None and self._eff() != self._ex_n:      # розмір кадру змінився - пул під нову пам'ять
                old, self._ex = self._ex, None
        if old is not None:
            kill_pool(old, wait=2.0)              # СПЕРШУ звільнити пам'ять: два пули по ~1 ГБ на процес на 4K = MemoryError при старті нових;
            # заодно кількість процесів нового пулу рахується з уже вільної пам'яті
        with self._lock:                                      # два потоки не повинні створити по пулу
            n = self._eff()
            fresh = False
            if n > 1 and self._ex is None:
                self._ex_n = n
                self._ex_t0 = time.perf_counter()
                self._ex = make_pool(n, self.size)
                fresh = True
            ex = self._ex
        if fresh:
            try:
                z = np.zeros((36, 64, 3), np.uint8)
                pd = Params().to_dict()
                for _ in range(min(n, RAMP0 if self.ramp else n)):   # прогрівається лише початкова група; решта додається в міру розгону
                    # (_conc): 27 процесів, що одночасно імпортують numpy/cv2/numba й читають кеш ядер, душили диск і процесор -
                    # система підвисала, а перший кадр чекав у черзі за прогрівом усіх
                    ex.submit(_frame_job, (z, 64, 36, pd))
                    ex.submit(_frame_job, (z, 534, 300, pd))
            except Exception:  # noqa: BLE001
                pass                              # пул закрили, поки його прогрівали (перемотка, експорт) - далі створимо новий
        return ex

    def _conc(self) -> int:
        """скільки процесів пулу вже можна задіяти: з нового пулу - RAMP0, далі +1 кожні RAMP_STEP с до повного розміру.
        Процес створюється, лише коли для нього є завдання, тож кількість завдань у польоті = кількість процесів"""
        n = self._ex_n
        if not self.ramp or n <= RAMP0 + 1 or getattr(self._ex, "is_host", False):       # потоки одного процесу стартують миттєво
            return n
        return min(n, RAMP0 + int((time.perf_counter() - self._ex_t0) / RAMP_STEP))

    def reset_pool(self):
        """пул перестворюється (напр. після ввімкнення/вимкнення відеокарти): воркери успадковують середовище при старті"""
        with self._lock:
            ex, self._ex = self._ex, None
            self._ink_ready, self._ink_warm = False, None
        kill_pool(ex)
        with self._lock:
            if self.info:
                self._restart(int(self._pos_nolock()))
                self._to_buffering()

    def _drop_pool(self, ex):
        """пул зламався (процес вбито, пам'яті не стало) або закритий: наступний _pool() створить новий"""
        with self._lock:
            if ex is not None and self._ex is ex:
                self._ex = None
        kill_pool(ex)

    def _eff(self) -> int:
        """процесів з огляду на поточний розмір кадру. Рівень 6 (нейро) - завжди 1: модель одна, на відеокарті, у цьому процесі (потоки-подавачі
        лише декодують паралельно, а саму мережу ганяють по черзі)"""
        if self.params.method == 6:
            return 1
        return workers_for(self.size[0], self.size[1], self.workers) if self.workers > 1 else 1

    def _ring_for(self, w, h):
        r = self._ring
        if r is not None and (r.w, r.h) == (w, h):
            return r
        if r is not None:
            r.dead = True
            if r.out == 0:
                r.destroy()
        n = max(2, min(self._eff() + 3, (600 << 20) // max(1, w * h * 3)))      # великі кадри (4K): менше слотів
        self._ring = _Ring(w, h, n)
        return self._ring

    def _ink_pool(self) -> bool:
        """«Ультра» відео рахується в процесах пулу (нижчий пріоритет - вікно й система не тормозять, і працюють усі ядра), а не в
        потоках цього процесу. Без пулу (1 процес) і для GIF/фото - старий шлях; після кількох невдалих шматків - теж"""
        return self.params.method == 5 and self.kind == "video" and self._eff() > 1 and self._ink_fail < 3

    def _chunk5(self, far: bool = False) -> int:
        """довжина шматка «Ультри» в пулі: чим більше процесів, тим коротші шматки - буфер наповнюється рівніше, а після перемотки
        менше застарілої роботи. Але кожен шматок починається з розігріву часової пам'яті (≈3 кадри роботи) і з нового малюнка штрихів:
        короткі шматки - це і повільніше, і частіші «шви» (штрихи починають малюватись заново). Тому не коротше 6 кадрів на будь-якому
        розмірі; далеко від програвача на 4K (кадр секунди) шматок 12 замість 24 - щоб усі процеси не віддавали результат одною хвилею"""
        if far:
            return CHUNK5 if self.size[0] * self.size[1] <= 2_300_000 else CHUNK5 // 2
        if self.params.glyph:                                # символи: кожен кадр дорожчий, розігрів - повні кадри; кадри віддаються по мірі готовності, тож довший шматок - менше розігріву, а не довше очікування
            return max(24, min(2 * CHUNK5, 192 // max(1, self._eff())))
        return max(6, min(CHUNK5, 96 // max(1, self._eff())))

    def _ink_room(self) -> bool:
        """у пулі є місце для ще одного шматка «Ультри»: ядра прогріто (один процес компілює або читає кеш, решта потім лише
        підвантажує - без паралельної компіляції в десятках процесів), у польоті не більше за кількість процесів + 1"""
        with self._lock:
            if self._ink_ready:
                c = self._conc()
                return self._inflight < max(2, 2 * c if c >= self._ex_n else c)   # після розгону черга глибша за кількість процесів: ядра не чекають новий шматок
            if self._ink_warm is not None:
                return False
            self._ink_warm = True                          # місце зайнято: другий подавач не замовить прогрів удруге
        fu = None
        try:                                               # замовлення - поза замком: запуск процесу не повинен гальмувати інтерфейс
            ex = self._pool()
            if ex is not None:
                fu = ex.submit(_ink_warm_job, bool(self.params.glyph))
        except Exception:  # noqa: BLE001
            ex, fu = True, None
        with self._lock:
            if fu is not None:
                self._ink_warm = fu
            elif ex is None:
                self._ink_warm = None                      # пулу немає (один процес): повернемось, коли з'явиться
            else:
                self._ink_ready = True                     # прогрів замовити не вдалося - гальмувати далі нема сенсу
        if fu is not None:
            fu.add_done_callback(self._ink_warmed)
        return False

    def _ink_warmed(self, fu):
        if fu.cancelled():
            self._ink_warm = None                    # пул закрито до кінця прогріву: почати знову, коли знадобиться
            return
        try:
            layer = fu.result()
        except Exception:  # noqa: BLE001
            layer = None
        if layer and getattr(self._ex, "is_host", False) and str(layer) not in ("omp", "tbb"):
            # шар потоків numba «workqueue» не витримує паралельних викликів із кількох потоків (ink5 тоді ставить усе під замок, і потоки
            # господаря рахували б по черзі): повертаємось до пулу процесів
            threading.Thread(target=self._to_processes, args=(str(layer),), daemon=True).start()
            return
        self._ink_ready = True                       # (й після помилки: гальмувати далі нема сенсу)

    def _to_processes(self, layer: str):
        try:
            from .log import get_logger
            get_logger().warning("numba threading layer %r is not thread-safe: switching to a process pool", layer)
        except Exception:  # noqa: BLE001
            pass
        os.environ["INKREEL_HOST"] = "0"
        self.reset_pool()

    @property
    def warming(self) -> bool:
        """«Ультру» ще прогріває перший процес пулу (на першому запуску - компіляція ядер, до пів хвилини)"""
        return self.params.method == 5 and self.kind == "video" and not self._ink_ready and self._ink_warm is not None

    def _cancel_inflight(self):
        """шматки пулу, що ще не почали рахуватись, скасовуються разом із поколінням (ті, що вже йдуть, дорахуються й відкинуться)"""
        for f in list(self._futs):
            f.cancel()
        for st in list(self._streams):                     # ті, що вже йдуть, зупиняються між кадрами (їх кадри все одно відкинуті: покоління інше)
            st.cancel()

    def _submit_ink(self, gen, path, idxs, w, h, pd, cold=False, pre=None):
        """шматок «Ультри» - у пул: воркер сам декодує, рахує з розігрівом часової пам'яті й віддає кадри через спільну пам'ять по мірі
        готовності. cold - терміновий одиночний кадр у поточній позиції: без розігріву (він коштує ≈3 кадри роботи)"""
        step = (idxs[1] - idxs[0]) if len(idxs) > 1 else max(1, self._stride)
        ex = self._pool()
        if ex is None:
            self._release_pend(gen, idxs)
            return
        stream = None
        try:
            stream = _Stream(gen, idxs, w, h)
        except Exception:  # noqa: BLE001
            stream = None                                   # спільної пам'яті не вистачило: шматок віддасться цілим, як раніше
        try:
            job = ((path, idxs[0], len(idxs), w, h, pd, step, cold, 1.0, None, None) if self._want is None else
                   (path, idxs[0], len(idxs), w, h, pd, 1, cold, 1.0, list(idxs), list(pre or ())))
            fu = ex.submit(_job, (*job, stream.name if stream is not None else None))
        except Exception:  # noqa: BLE001
            if stream is not None:
                stream.destroy()
            self._release_pend(gen, idxs)                    # пул зламано або закрито: шматок повернеться в чергу, пул буде новий
            self._drop_pool(ex)
            return
        with self._lock:
            self._inflight += 1
            self._futs.add(fu)
            if stream is not None:
                self._streams.append(stream)
        if stream is not None:
            self._stream_ev.set()
        fu.add_done_callback(lambda f, g=gen, ii=idxs, e=ex, st=stream: self._cb_chunk(g, ii, f, e, st))

    def _pump_streams(self):
        """забирає готові кадри шматків, що ще рахуються в пулі (кожен кадр - одразу в буфер)"""
        while self._alive:
            if not self._streams:
                self._stream_ev.wait(0.5)
                self._stream_ev.clear()
                continue
            try:
                for st in list(self._streams):
                    self._take_ready(st)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.03)

    def _take_ready(self, st):
        with self._lock:
            while st.hdr is not None and st.sent < st.ready():
                k = st.sent
                st.sent += 1
                self._done(st.gen, st.idxs[k], st.frame(k))

    def _cb_chunk(self, gen, idxs, fut, ex=None, st=None):
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self._futs.discard(fut)
        if st is not None:                                     # решту кадрів, що встигли лягти в спільну пам'ять, забираємо в будь-якому разі (навіть коли воркер упав)
            try:
                self._take_ready(st)
            finally:
                with self._lock:
                    if st in self._streams:
                        self._streams.remove(st)
                    idxs = idxs[st.sent:]
                st.destroy()
            if not idxs:
                return
        try:
            res = fut.result()
        except CancelledError:
            self._release_pend(gen, idxs)
            return
        except Exception:  # noqa: BLE001
            with self._lock:                                   # як у _cb: повторити шматок, а пул перестворити
                if ex is not None and self._ex is not ex:      # пул замінено навмисно (інший розмір, експорт): це не збій шматка
                    if gen == self._gen:
                        for i in idxs:
                            if i < len(self._pend):
                                self._pend[i] = False
                        self._head = min(self._head, idxs[0])
                    return
                self._ink_fail += 1
                if gen == self._gen:
                    for i in idxs:
                        if i < len(self._pend):
                            self._pend[i] = False
                    self._head = min(self._head, idxs[0])
                    self._fails[idxs[0]] = self._fails.get(idxs[0], 0) + 1
                    if self._fails[idxs[0]] > 5:
                        self._head = max(self._head, idxs[-1] + 1)
                if ex is None or self._ex is ex:
                    kill_pool(self._ex)
                    self._ex = None
            return
        self._ink_fail = 0
        got = 0 if st is not None else len(res)                # (з потоку всі готові кадри вже віддані)
        for k in range(min(got, len(idxs))):
            self._done(gen, idxs[k], res[k])
        if got < len(idxs):                                    # воркер не прочитав частину шматка: як при збої читання
            self._read_failed(gen, idxs[got], idxs[got + 1:], None)

    def _claim(self):
        """-> (gen, [кадри], src_id, size, params, ring, у пул «Ультри»?, шлях, кадри перед шматком або None, терміновий?) або None.
        Бере до CHUNK підряд ще не готових кадрів (з кроком чи за лімітом к/с - лише тих, що рахуємо); pre - кадри, що йдуть перед шматком"""
        with self._lock:
            if not self.info or self._src is None or self._suspended or self._hold:
                return None
            if len(self._frames) * max(1, (self.size[0] * self.size[1] + 7) // 8) > self.mem_cap:
                return None
            sd = self._seed
            pool5 = (not self.baked) and self._ink_pool()
            room = (not pool5) or self._ink_room()
            if not room and sd is None:
                return None
            n = self.n
            pos = min(int(self._pos_nolock()), n - 1)
            st = self._stride
            base = self._floor_cand(pos, st)                  # кадр, що показується на позиції (при кроці чи ліміті к/с - не завжди сама позиція)
            if self._head < base and not self._behind:
                self._head = base
            h = self._head
            busy = self._have | self._pend
            while True:
                ci = self._cand(st)
                j = -1
                j0 = int(np.searchsorted(ci, h))
                while j0 < len(ci):
                    nz = np.flatnonzero(~busy[ci[j0:j0 + 8192]])
                    if len(nz):
                        j = j0 + int(nz[0])
                        break
                    j0 += 8192
                if j >= 0 or st <= 1:
                    break
                # рідкий режим дорахований до кінця (буфер повний за цим кроком): процесор простоює, тож крок зменшуємо вдвічі й
                # докладаємо пропущені кадри від позиції; коли дірок нема - режим «кожен N-й» вимикається сам
                st = self._stride = max(1, st // 2)
                h = self._floor_cand(pos, st)
            if j < 0 and not self._behind and st <= 1 and self.kind == "video" and pos > 0:
                # попереду все готове, а позаду позиції (початок відео, куди ще не дійшли після зміни налаштувань чи перемотки) - порожньо:
                # якщо все відео вміщається в пам'ять, дорахувати й початок, інакше він чекав би перемотки
                per = max(1, (self.size[0] * self.size[1] + 7) // 8)
                tot = n if self._want is None else int(self._want[:n].sum())
                if tot * per <= self.mem_cap * 0.85:
                    hole = np.flatnonzero(~busy[ci[:int(np.searchsorted(ci, pos))]]) if len(ci) else []
                    if len(hole):
                        self._behind = True
                        self._head = h = int(ci[hole[0]])
                        j = int(np.searchsorted(ci, h))
            if j < 0:
                back = self._seed_rest(sd, busy)
                if back:                                      # попереду все є, а позаду позиції у файлі .ikr ще є готові кадри: підтягуємо й їх (перемотка назад - миттєва)
                    self._pend[back] = True
                    return self._gen, back, self._src_id, self.size, self.params.to_dict(), None, False, self.path, [], False
                self._head = n
                done = self._have[pos:n] if self._want is None else (self._have[pos:n] | ~self._want[pos:n])
                if st <= 1 and self._gap > self._gb and not self._pend[pos:n].any() and done.all():
                    self._gap, self._stalls, self._start_s = self._gb, 0, START_S        # дірок нема: і запас перед стартом знову типовий
                return None
            h = int(ci[j])
            seeded = sd is not None and sd[0] == self._src_id and sd[2] <= h < sd[3]      # кадр уже пораховано у файлі .ikr: беремо його звідти
            if not room and not seeded:
                return None                                   # цей кадр треба рахувати, а пул зайнятий
            if seeded:
                pool5 = False
            urgent = bool(pool5 and self._urgent and h == self._floor_cand(pos, st))
            self._urgent = False
            # після відкриття чи перемотки перший кадр - окремим завданням без розігріву: інакше він чекає цілий шматок (на 4K -
            # десятки секунд: шматок віддається лише повністю, а розігрів - ще ≈3 кадри), і вікно довго показує нічого
            lim = 1 if urgent else self._chunk5(h - pos > 3 * self.fps) if pool5 else 64 if (self.baked or seeded) else CHUNK5 if self.params.method == 5 else CHUNK6 if self.params.method == 6 else CHUNK
            idxs = [h]
            jj = j + 1
            while len(idxs) < lim and jj < len(ci) and not busy[ci[jj]] and (not seeded or ci[jj] < sd[3]):
                idxs.append(int(ci[jj]))
                jj += 1
            pre = [int(v) for v in ci[max(0, j - PRE_K):j]]       # кадри перед шматком у порядку ряду (при ліміті чи кроці проміжки нерівні)
            self._pend[idxs] = True
            self._head = idxs[-1] + 1
            ring = self._ring_for(*self.size) if self._eff() > 1 and not pool5 and not self.baked and not seeded else None
            return self._gen, idxs, self._src_id, self.size, self.params.to_dict(), ring, pool5, self.path, pre, urgent

    def _release_pend(self, gen, idxs):
        """кадри шматка не пораховано (шматок скасовано, пул відмовив): вони знову вільні, і черга вертається до них - інакше після
        збою лишалася б дірка в буфері, яку довелося б «долікувати» зупинкою відтворення"""
        with self._lock:
            if gen == self._gen:
                for i in idxs:
                    if i < len(self._pend):
                        self._pend[i] = False
                if len(idxs):
                    self._head = min(self._head, int(min(idxs)))

    def _feeder(self, k):
        try:
            from . import procs
            procs.lower_thread_priority()                    # інтерфейс (звичайний пріоритет) завжди випереджає подачу кадрів
        except Exception:  # noqa: BLE001
            pass
        src, sid = None, -1
        cont = None                                           # (ключ, пам'ять «Ультри», останній порахований кадр): наступний шматок поруч її продовжує
        if self._nb_threads:                                  # numba рахує потоки окремо для кожного потоку Python
            try:
                import numba
                numba.set_num_threads(max(1, min(int(self._nb_threads), numba.config.NUMBA_NUM_THREADS)))
            except Exception:  # noqa: BLE001
                pass
        while self._alive:
            try:
                job = self._claim()
            except Exception:  # noqa: BLE001
                job = None
                time.sleep(0.2)
            if job is None:
                if src is not None and sid != self._src_id:              # джерело змінилось (інше відео, режим із замочком): свій декодер більше не потрібен, а Windows не дає перейменувати відкритий файл
                    try:
                        src.close()
                    except Exception:  # noqa: BLE001
                        pass
                    src, sid, cont = None, -1, None
                time.sleep(0.01)
                continue
            gen, idxs, jsid, (w, h), pd, ring, pool5, path, pre_c, cold = job
            try:
                if pool5:
                    self._submit_ink(gen, path, idxs, w, h, pd, cold, pre_c)
                    continue
                if jsid != sid:                          # свій декодер для кожного потоку й кожного файлу
                    with self._lock:
                        base = self._src
                    if src is not None:
                        try:
                            src.close()
                        except Exception:  # noqa: BLE001
                            pass
                    src, sid = (base.clone() if base is not None else None), jsid
                if src is None:
                    self._release_pend(gen, idxs)
                    continue
                if getattr(src, "baked", False):                 # режим із замочком: кадри лише читаються з файлу, без обробки й пулу
                    for pos_i, idx in enumerate(idxs):
                        if gen != self._gen or not self._alive:
                            self._release_pend(gen, idxs[pos_i:])
                            break
                        pk = src.packed(idx)
                        if pk is None:
                            self._read_failed(gen, idx, idxs[pos_i + 1:], src)
                            break
                        self._done(gen, idx, pk)
                    continue
                sd = self._seed
                if sd is not None and sd[0] == jsid and idxs and sd[2] <= idxs[0] and idxs[-1] < sd[3] and sd[4] == pd and sd[5] == (w, h):
                    self._feed_seed(gen, idxs, sd)               # розморозка: кадри вже пораховані у файлі .ikr
                    continue
                tmp = None
                ckey = None
                neuro = pd.get("method") == 6
                try:
                    from . import procs
                    procs.set_thread_priority(0 if neuro else -1)
                except Exception:  # noqa: BLE001
                    pass
                if neuro:                                        # IRNM1: згладжування опису кадру; шматок одразу за попереднім його продовжує
                    from .core import neuro_warm
                    from .student import FilmState, WARM as NEURO_WARM
                    ckey = (jsid, w, h, tuple(sorted(pd.items())))
                    if cont is not None and cont[0] == ckey and pre_c and cont[2] == pre_c[-1]:
                        tmp = cont[1]
                    else:
                        tmp = FilmState()
                        for wi in list(pre_c)[-NEURO_WARM:]:
                            if gen != self._gen or not self._alive:
                                break
                            w0 = src.read(wi)
                            if w0 is not None:
                                neuro_warm(w0, w, h, tmp)
                    cont = None
                if pd.get("method") == 5:                        # «Ультра»: кадри шматка йдуть підряд з часовою пам'яттю
                    from . import ink5, glyph
                    from .core import PRIME, InkState
                    pp = Params.from_dict(pd)
                    gly = bool(pd.get("glyph"))                  # символи: пам'ять - ознаки клітинок, розігрів короткий і повний
                    nwarm = GLYPH_WARM + GLYPH_CHEAP if gly else ink5.WARM
                    ckey = (jsid, w, h, tuple(sorted(pd.items())))
                    if cont is not None and cont[0] == ckey and pre_c and cont[2] == pre_c[-1]:
                        tmp = cont[1]                            # шматок іде одразу за попереднім: пам'ять уже на місці, розігрів (на слабкому ПК - пів хвилини) не потрібен
                    else:
                        tmp = InkState()
                        pre = list(pre_c)[-(nwarm + PRIME // 2):]
                        nprime = max(0, len(pre) - nwarm)        # найдальші: лише мініатюра для пам'яті нормування тону (у плеєрі коротше, ніж в експорті)
                        for j, wi in enumerate(pre):             # розігрів: далекі кадри - лише світлота, останній - повністю
                            if gen != self._gen or not self._alive:
                                break
                            w0 = src.read(wi)
                            if w0 is None:
                                continue
                            if j < nprime:
                                (glyph.prime(tmp, w0, w, h, pp.auto, pp.gdens) if gly else ink5.prime(tmp, w0, w, h, pp.auto))
                                continue
                            if w0.shape[1::-1] != (w, h):
                                w0 = cv2.resize(w0, (w, h), interpolation=cv2.INTER_AREA)
                            if gly and j < len(pre) - GLYPH_WARM:
                                glyph.plan(w0, w, h, pp, tmp, warm_only=True)        # символи: далекі кадри розігріву - лише тон
                            elif j < len(pre) - 1 and not gly:
                                ink5.warm(tmp, w0, w, h, pp)
                            else:
                                with self._ink_gate:
                                    binarize(w0, w, h, pp, {"state": tmp})
                    cont = None                                  # поки шматок іде, пам'ять «в роботі»: збій посеред нього не лишає напівоновленого стану
                for pos_i, idx in enumerate(idxs):
                    if gen != self._gen or not self._alive:
                        self._release_pend(gen, idxs[pos_i:])
                        break
                    f = src.read(idx)
                    if f is None:
                        self._read_failed(gen, idx, idxs[pos_i + 1:], src)
                        break
                    sh, sw = f.shape[:2]
                    if (sw, sh) != (w, h):
                        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA if w * h < sw * sh else cv2.INTER_CUBIC)
                    if tmp is not None:
                        with (contextlib.nullcontext() if neuro else self._ink_gate):          # нейро: замок лише всередині моделі (на відеокарту), підготовка кадрів потоками паралельно
                            pk = np.packbits(binarize(f, w, h, Params.from_dict(pd), {"film": tmp} if neuro else {"state": tmp}), axis=-1)
                        cont = (ckey, tmp, idx)                  # пам'ять тепер відповідає кадру idx
                        self._done(gen, idx, pk)
                        continue
                    if ring is None:
                        with self._inproc:                       # без пулу кадр рахується тут: не більше одного великого за раз
                            pk = _frame_job((f, w, h, pd))
                        self._done(gen, idx, pk)
                        continue
                    slot = None
                    with self._slot_cv:
                        while self._alive and gen == self._gen:
                            if ring.free and ring.out < self._conc() + 2:
                                slot = ring.free.pop()
                                ring.out += 1
                                break
                            self._slot_cv.wait(0.05)
                    if slot is None:
                        self._release_pend(gen, idxs[pos_i:])
                        break
                    ex = None
                    try:
                        ring.arr[slot] = f
                        ex = self._pool()
                        fu = ex.submit(_frame_job_shm, (ring.name, slot, w, h, pd))
                    except Exception:  # noqa: BLE001
                        self._give_slot(ring, slot)          # слот не лишається «зайнятим назавжди», якщо кадр не вдалося подати
                        if ex is not None:
                            self._drop_pool(ex)              # зламаний або вже закритий пул - наступний кадр піде в новий
                        raise
                    fu.add_done_callback(lambda fu_, g=gen, i=idx, r=ring, s=slot, e=ex: self._cb(g, i, r, s, fu_, e))
            except Exception:  # noqa: BLE001
                self._release_pend(gen, idxs)
                time.sleep(0.05)

    def _read_failed(self, gen, idx, rest, src):
        with self._lock:
            if gen != self._gen:                              # запізніла відповідь про старий файл/позицію - лічильників не чіпати
                return
            self._release_pend(gen, [idx] + list(rest))
            self._fails[idx] = self._fails.get(idx, 0) + 1
            if idx >= self.n - 3 or self._fails[idx] > 3:     # лічильник кадрів перебільшений - кінець тут
                if idx > 0 and not self._have[idx]:
                    self.n = max(1, min(self.n, idx))
            elif gen == self._gen:
                getattr(src, "reset", lambda: None)()
                self._head = min(self._head, idx)             # спробувати ще раз

    def _give_slot(self, ring, slot):
        """слот кільця знову вільний (кадр порахований, скасований або не був поданий)"""
        with self._slot_cv:
            ring.free.append(slot)
            ring.out -= 1
            if ring.dead and ring.out == 0:
                ring.destroy()
            self._slot_cv.notify()

    def _cb(self, gen, idx, ring, slot, fut, ex=None):
        self._give_slot(ring, slot)
        try:
            self._done(gen, idx, fut.result())
        except Exception:  # noqa: BLE001
            with self._lock:                                   # не лишати дірок у буфері: повторити, пул перестворити
                if gen == self._gen and idx < len(self._pend):
                    self._pend[idx] = False
                    self._head = min(self._head, idx)
                    self._fails[idx] = self._fails.get(idx, 0) + 1
                    if self._fails[idx] > 5:
                        self._head = max(self._head, idx + 1)
                if ex is None or self._ex is ex:                # не зносити новий пул через запізнілі помилки старого
                    kill_pool(self._ex)
                    self._ex = None

    def _done(self, gen, idx, pk):
        with self._lock:
            if gen != self._gen:
                return
            self._pend[idx] = False
            self._frames[idx] = pk
            self._have[idx] = True
            self._last_done = time.perf_counter()
            self._done_times.append(self._last_done)
            if len(self._done_times) > 600:
                del self._done_times[:300]

    def _watch(self):
        """сторож (запис у журнал англійською): якщо відтворення стоїть (кадри не готуються понад 8 с, хоч мають, або інтерфейс не відповідає), один раз пише в журнал, де
        застрягли всі потоки. Без цього зависання (напр. в драйвері відеокарти) не лишає в журналі жодного сліду"""
        import sys
        import traceback
        warned = None
        buf_since = None                                                  # коли сторож уперше побачив «чекаю кадрів»: довга пауза до натискання «грати» застоєм не є
        t_stat = time.perf_counter()
        while self._alive:
            time.sleep(2.0)
            try:
                now = time.perf_counter()
                if now - t_stat >= 10.0:                                  # куди йде час IRNM1: у журнал раз на 10 с, поки він рахує
                    t_stat = now
                    from . import log, student
                    st = student.stats_take()
                    if st and st["n"] >= 5:
                        log.get().info("neuro: %.1f fps in the model | per frame ms: description %.1f, mask total %.1f (waiting for the GPU lock %.1f, GPU run %.1f) | "
                                       "%dx%d | player says %.0f fps | open #%d | %s | threads %d", st["fps"], st["film"], st["mask"], st["wait"], st["run"],
                                       st["size"][0], st["size"][1], self.render_fps(), self._src_id, student.current(), threading.active_count())
                got = self._lock.acquire(timeout=1.0)
                if got:
                    try:
                        busy = bool(self._want_play and self.info and self._src is not None and not self._suspended and not self._hold)
                        buffering = self._state == "buffering"
                        head, n, have, pend, gen = self._head, self.n, int(self._have.sum()), int(self._pend.sum()), self._gen
                        stride, method = self._stride, self.params.method
                    finally:
                        self._lock.release()
                else:
                    busy, buffering, head, n, have, pend, gen, stride, method = True, False, -1, self.n, -1, -1, self._gen, -1, -1
                if busy and buffering:
                    if buf_since is None:
                        buf_since = now
                else:
                    buf_since = None
                stuck_frames = busy and buffering and now - max(self._last_done, buf_since) > self._stall_s
                stuck_ui = busy and now - self._ui_t > 8.0
                if not (stuck_frames or stuck_ui or not got):
                    warned = None
                    continue
                tag = (self._last_done, self._ui_t, got)
                if warned == tag:
                    continue
                warned = tag
                from . import log
                from . import student
                why = "frames are not produced" if stuck_frames else "the UI does not respond" if stuck_ui else "the player lock cannot be taken"
                lines = [f"watchdog: {why} (last frame {now - self._last_done:.0f} s ago, waiting for frames {now - (buf_since or now):.0f} s, UI {now - self._ui_t:.0f} s ago); "
                         f"gen {gen}, head {head}/{n}, "
                         f"have {have}, pending {pend}, stride {stride}, level {method}, neuro backend {student.current()}"]
                names = {t.ident: t.name for t in threading.enumerate()}
                for tid, fr in sys._current_frames().items():
                    lines.append(f"--- thread {names.get(tid, tid)}")
                    lines.append("".join(traceback.format_stack(fr)[-6:]).rstrip())
                log.get().warning("\n".join(lines))
            except Exception:  # noqa: BLE001
                pass

    def close(self):
        self._alive = False
        self._stream_ev.set()
        self.audio.close()
        time.sleep(0.05)
        if self._ex:
            kill_pool(self._ex)
        with self._lock:                                     # спільна пам'ять шматків, що не дорахувались, не повинна лишатись у системі
            sts, self._streams = list(self._streams), []
        for st in sts:
            st.destroy()
        if self._src is not None:
            self._src.close()
        r = self._ring
        if r is not None:
            r.dead = True
            if r.out == 0:
                r.destroy()
