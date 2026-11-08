"""«Господар обробки»: один допоміжний процес із потоками замість пулу з десятків процесів.

УВАГА (28.10.2026): типово ВИМКНЕНО (INKREEL_HOST=1 вмикає). Заміри на i7-14700KF показали, що пул процесів швидший на 45-90%: у потоках
одного процесу десятки малих викликів cv2/numpy на кадр сперечаються за GIL (мікротест thr_vs_proc2.py, 15 потоків: cv2 - 28%, numpy - 49%
швидкості 15 процесів; великі numba-ядра масштабуються однаково). Нижче - початкові міркування, заради яких господаря й робили.

Навіщо. Пул із N процесів (по 27 на 28 ядрах) на Windows коштував трьох речей:
  * старт: усі процеси разом імпортують numpy/cv2/numba і читають кеш ядер - диск і процесор душаться, система підвисає, а перший кадр
    стоїть у черзі за прогрівом решти;
  * пам'ять: у кожного процесу свої копії таблиць, полів і контексту відеокарти (≈260 МБ OpenCL-контекст + кадрові буфери на 4K ≈ 420 МБ);
  * відеокарта: контексти різних процесів вона обслуговує по черзі (time-slicing; без MPS, якого на Windows нема, - по одному процесу
    в кожен момент), тож 8-12 процесів із GPU не прискорюють, а заважають одне одному, а очікування кожного «крутить» ядро ЦП.
Так само влаштовані схожі програми (waifu2x/Real-ESRGAN ncnn-vulkan, ffmpeg): один процес, один контекст пристрою, потоки на етапи.

Тут: ОДИН процес (нижчий пріоритет, тож вікно лишається жвавим) із ThreadPoolExecutor. numba-ядра й cv2 не тримають GIL, а ink5.ultra
сам береже від паралельних викликів шару потоків numba, який цього не витримує. Інтерфейс - як у ProcessPoolExecutor (submit/shutdown/
_processes), тож плеєр, експорт і kill_pool працюють з обома. Вимкнути й повернути пул процесів: змінна INKREEL_HOST=0."""
from __future__ import annotations

import itertools
import multiprocessing
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor


class HostBroken(RuntimeError):
    """процес-господар помер (або закритий): усі завдання в польоті провалюються, плеєр створює новий пул"""


def _pack_exc(e: BaseException) -> BaseException:
    """виняток має дійти до батька навіть коли сам він не піклюється"""
    try:
        import pickle
        pickle.loads(pickle.dumps(e))
        return e
    except Exception:  # noqa: BLE001
        return RuntimeError(f"{type(e).__name__}: {e}")


_PROF = {"lock": threading.Lock(), "n": 0, "thr": 0, "idle": 0, "convoy": 0, "labels": {}, "conv_labels": {}}


def _sample_threads():
    """один знімок: де зараз стоїть кожен потік обробки (innermost кадр коду benvid). Потік без кадру benvid - простоює в очікуванні
    завдання. Якщо ≥60% активних потоків стоять в одному місці - це «збіг» (convoy): усі чекають одного й того ж (замок, смуга GPU, канал)"""
    import sys as _s
    fr = _s._current_frames()
    labs, idle, thr = {}, 0, 0
    for t in threading.enumerate():
        if not t.name.startswith("ink_"):
            continue
        thr += 1
        f = fr.get(t.ident)
        lab = None
        while f is not None:
            fn = f.f_code.co_filename
            if os.sep + "benvid" + os.sep in fn or "/benvid/" in fn:
                lab = f"{os.path.basename(fn)}:{f.f_code.co_name}"
                break
            f = f.f_back
        if lab is None:
            idle += 1
        else:
            labs[lab] = labs.get(lab, 0) + 1
    with _PROF["lock"]:
        _PROF["n"] += 1
        _PROF["thr"] += thr
        _PROF["idle"] += idle
        for k, v in labs.items():
            _PROF["labels"][k] = _PROF["labels"].get(k, 0) + v
        act = thr - idle
        if act >= 4:
            k, v = max(labs.items(), key=lambda kv: kv[1])
            if v >= 0.6 * act:
                _PROF["convoy"] += 1
                _PROF["conv_labels"][k] = _PROF["conv_labels"].get(k, 0) + 1


def _take_prof() -> dict:
    with _PROF["lock"]:
        d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _PROF.items() if k != "lock"}
        _PROF.update(n=0, thr=0, idle=0, convoy=0, labels={}, conv_labels={})
    top = sorted(d["labels"].items(), key=lambda kv: -kv[1])[:8]
    d["labels"] = top
    d["conv_labels"] = sorted(d["conv_labels"].items(), key=lambda kv: -kv[1])[:3]
    return d


class _Gov:
    """Страж пам'яті господаря. Усі потоки живуть в одному процесі, тож «Пам'ять» його одна цифра, і якщо вона росте (проміжні масиви
    всіх потоків одночасно, буфери драйвера відеокарти, що не повертаються системі), ніщо інше цього не зупинить: батько рахує потоки
    за оцінкою, а не за фактом. Тут фактична пам'ять процесу (і вільна у системі) порівнюється з бюджетом: понад нього НОВІ завдання не
    стартують, доки не звільниться - потоки, що вже йдуть, доробляють. Хоч одне завдання йде завжди, тож зависань не буває, а на
    звичайних розмірах страж нічого не змінює. Коли ж усе стоїть і пам'ять усе ще над бюджетом - віддаємо системі, що можна (див. _trim)"""

    def __init__(self, soft: int, floor: int):
        self.soft, self.floor = int(soft), int(floor)
        self.over, self.running = False, 0
        self.cv = threading.Condition()
        self.rss = self.peak = 0
        self.waits = self.trims = 0
        self.over_s = 0.0
        self._t_trim = 0.0

    def enter(self):
        with self.cv:
            waited = False
            while self.over and self.running > 0:
                waited = True
                self.cv.wait(0.5)
            if waited:
                self.waits += 1
            self.running += 1

    def leave(self):
        with self.cv:
            self.running -= 1
            self.cv.notify_all()

    def update(self, rss: int, free: int, dt: float):
        self.rss = rss
        self.peak = max(self.peak, rss)
        low = bool(free) and free < self.floor
        with self.cv:
            was = self.over
            if self.over:
                self.over = not (rss < 0.85 * self.soft and not (bool(free) and free < 1.5 * self.floor))
                self.over_s += dt
            else:
                self.over = rss > self.soft or low
            idle = self.over and self.running == 0
            if was and not self.over:
                self.cv.notify_all()
        if idle and time.perf_counter() - self._t_trim > 2.0:
            self._t_trim = time.perf_counter()
            self.trims += 1
            _trim()


def _trim():
    """віддати системі те, що процес більше не використовує: буфери відеокарти й вільні сторінки купи (масиви numpy й так звільняються
    лічильником посилань; збирач сміття тут не допомагає, а в одному процесі з відправником він іноді наражається на їхні буфери)"""
    try:
        from . import gpu
        g = gpu._dev
        if g:
            for ln in [g, *list(g._free.queue)]:                # смуги, вільні в цей момент (завдань нема - вільні всі), і корінь
                ln.trim(force=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        import ctypes
        if os.name == "nt":
            ctypes.CDLL("ucrtbase")._heapmin()                   # купа CRT, з якою зібраний Python (не застарілий msvcrt.dll)
        else:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


_GOV: _Gov | None = None


def _gov_limits() -> tuple[int, int]:
    """(бюджет пам'яті господаря, поріг вільної пам'яті системи). Бюджет - те саме, що й у video.workers_for: 40% ОЗП, але не більше 80% вільного
    на старті (разом із тим, що процес уже займає). INKREEL_HOST_MEM_MB задає бюджет вручну"""
    from . import video
    total, free, mine = video.ram_bytes(), video.ram_free_bytes(), video.proc_rss_bytes()
    try:
        env = int(os.environ.get("INKREEL_HOST_MEM_MB", "0"))
    except ValueError:
        env = 0
    if env > 0:
        soft = env << 20
    else:
        soft = 0.4 * total
        if free:
            soft = min(soft, 0.8 * (free + mine))
        soft = max(soft, 1 << 30)
    return int(soft), int(max(1 << 30, 0.04 * total))


def _stats(sent: dict) -> dict:
    """лічильники господаря для журналу: скільки секунд відправник писав у канал і чи потоки чекали вільної смуги відеокарти"""
    d = {"send_busy": round(sent.get("busy", 0.0), 2), "sent": sent.get("n", 0), "prof": _take_prof()}
    try:
        from . import video, gpu
        d["acc"] = video.take_acc()                            # секунди потоків по стадіях (розігрів, читання, бінаризація, упакування)
        d["gpu_busy"] = round(gpu.ACC[0], 2)                   # секунди потоків у викликах відеокарти (у т.ч. очікування смуги) і кількість викликів
        d["gpu_calls"] = gpu.ACC[1]
    except Exception:  # noqa: BLE001
        pass
    g = _GOV
    if g is not None:                                          # фактична пам'ять господаря (МБ): поточна, пік, ліміт; скільки разів стража тримав завдання
        d.update(rss=g.rss >> 20, rss_peak=g.peak >> 20, mem_soft=g.soft >> 20, mem_waits=g.waits, mem_over_s=round(g.over_s, 1), mem_trims=g.trims)
    try:
        from . import gpu
        g = gpu._dev
        if g:
            d.update(lanes=g._made, lanes_max=g.lanes_max, lane_calls=g.lane_calls, lane_wait=round(g.lane_wait, 2))
    except Exception:  # noqa: BLE001
        pass
    return d


def _host_main(conn, nthreads: int, gpu_on: bool):
    """тіло процесу-господаря: приймає ("job", id, fn, args) / ("cancel", id) / ("stop",), віддає (id, "ok"|"err"|"cancel", значення)"""
    try:
        from . import video
        video._lower_priority()
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import gpu
        gpu.set_allowed(bool(gpu_on))
        gpu.LEAN_BYTES = 1 << 30                 # відеокарту тримає один процес: буфери кадру 4K можна не віддавати після кожного виклику
    except Exception:  # noqa: BLE001
        pass
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001
        pass
    global _GOV
    try:
        _GOV = _Gov(*_gov_limits())
    except Exception:  # noqa: BLE001
        _GOV = None
    gov = _GOV
    pool = ThreadPoolExecutor(max(1, int(nthreads)), thread_name_prefix="ink")
    futs: dict = {}
    # Усе, що йде до батька, віддає ОДИН потік-відправник: потоки обчислень лише кладуть повідомлення в чергу й не чекають на канал.
    # Раніше кожен потік писав у канал сам під спільним замком: пока відправлявся великий готовий шматок, усі інші на найближчому
    # «лічильнику кадрів» стояли на тому ж замку (синхронний провал навантаження всіх ядер), а потік, що закінчив шматок, сам
    # висів на відправці замість брати наступне завдання
    cv = threading.Condition()
    outbox: list = []
    ticks: dict = {}
    sent = {"busy": 0.0, "n": 0}

    def post(msg):
        with cv:
            outbox.append(msg)
            cv.notify()

    def sender():
        t_stat = 0.0
        while True:
            with cv:
                while not outbox and not ticks:
                    cv.wait(2.0)
                    if time.perf_counter() - t_stat > 2.0:
                        break
                msgs, outbox[:] = list(outbox), []
                tk, done_ids = dict(ticks), {m[0] for m in msgs}
                ticks.clear()
            t0 = time.perf_counter()

            def out(m):
                try:
                    conn.send(m)
                except (OSError, ValueError, EOFError):      # батька нема: працювати нема для кого
                    os._exit(0)
                except Exception as ex:  # noqa: BLE001           # результат не піклюється: помилкою, а не мовчанням
                    try:
                        conn.send((m[0], "err", RuntimeError(f"result: {type(ex).__name__}: {ex}")))
                    except Exception:  # noqa: BLE001
                        os._exit(0)
            for i, n in tk.items():
                if i not in done_ids:
                    out((i, "tick", n))
            for m in msgs:
                out(m)
            if time.perf_counter() - t_stat > 2.0:
                t_stat = time.perf_counter()
                out((-1, "stat", _stats(sent)))
            sent["busy"] += time.perf_counter() - t0
            sent["n"] += len(msgs)

    threading.Thread(target=sender, daemon=True, name="ink-host-sender").start()

    def sampler():
        """діагностика «де стоять потоки» (INKREEL_PROF=1). Типово ВИМКНЕНА: знімок стеків 10 разів на секунду (sys._current_frames) тримав
        проміжні масиви кадрів усіх потоків живими - господар із 15 потоками розростався на 4K до десятків ГБ (виміряно: 2 потоки, Символи
        4096x1728, 4 завдання - ≈ 4.5 ГБ за 16 с проти ≈ 0.5 ГБ без знімків)"""
        while True:
            time.sleep(0.1)
            if futs:                                  # лише поки є завдання
                try:
                    _sample_threads()
                except Exception:  # noqa: BLE001
                    pass

    def memwatch():
        while True:
            time.sleep(0.3)
            if gov is not None:                       # пам'ять процесу й системи - три рази на секунду
                try:
                    from . import video
                    gov.update(video.proc_rss_bytes(), video.ram_free_bytes(), 0.3)
                except Exception:  # noqa: BLE001
                    pass

    if os.environ.get("INKREEL_PROF") == "1":
        threading.Thread(target=sampler, daemon=True, name="ink-host-sampler").start()
    threading.Thread(target=memwatch, daemon=True, name="ink-host-memwatch").start()

    def run(i, fn, args):
        """завдання з «лічильником»: шматок відео віддається лише цілим, тож без нього смуга прогресу стоїть, доки не дорахується
        перший шматок (десятки секунд), а потім стрибає. video._job викликає tick(скільки кадрів готово) після кожного кадру"""
        last = [0.0]

        def tick(n):
            t = time.perf_counter()
            if t - last[0] >= 0.2:
                last[0] = t
                with cv:
                    ticks[i] = int(n)
                    cv.notify()

        try:
            from . import video
            video._TICKER.fn = tick
        except Exception:  # noqa: BLE001
            video = None
        if gov is not None:
            gov.enter()
        try:
            return fn(*args)
        finally:
            if gov is not None:
                gov.leave()
            if video is not None:
                video._TICKER.fn = None

    def done(i, f):
        futs.pop(i, None)
        if f.cancelled():
            post((i, "cancel", None))
            return
        e = f.exception()
        if e is not None:
            post((i, "err", _pack_exc(e)))
            return
        post((i, "ok", f.result()))

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        op = msg[0]
        if op == "job":
            _, i, fn, args = msg
            f = pool.submit(run, i, fn, args)
            futs[i] = f
            f.add_done_callback(lambda f_, i_=i: done(i_, f_))
        elif op == "cancel":
            f = futs.get(msg[1])
            if f is not None:
                f.cancel()                                # лише ті, що ще в черзі; ті, що йдуть, дорахуються й будуть відкинуті
        elif op == "stop":
            break
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    finally:
        os._exit(0)


class _HostFuture(Future):
    def __init__(self, host, i):
        super().__init__()
        self._host, self._id = host, i

    def cancel(self):
        ok = super().cancel()
        if ok:
            self._host._cancel(self._id)
        return ok


class RenderHost:
    """інтерфейс як у ProcessPoolExecutor: submit(fn, *args) -> Future, shutdown(), _processes (для kill_pool)"""
    is_host = True

    def __init__(self, nthreads: int, gpu_on: bool = False):
        ctx = multiprocessing.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_host_main, args=(child, int(nthreads), bool(gpu_on)), daemon=True)
        self._proc.start()
        child.close()
        self.nthreads = int(nthreads)
        self._processes = {self._proc.pid: self._proc}
        self._futs: dict = {}
        self._prof: dict = {}
        self.stats: dict = {}                       # лічильники господаря (раз на пару секунд): див. _stats
        self._part: dict = {}                       # номер завдання -> скільки кадрів уже готово всередині нього (поки воно йде)
        self._ids = itertools.count()
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._dead: BaseException | None = None
        self._closed = False
        threading.Thread(target=self._read, daemon=True, name="ink-host-reader").start()

    # ---------------------------------------------------------------- батьківська сторона
    def submit(self, fn, *args):
        with self._lock:
            if self._dead is not None:
                raise HostBroken("processing host is not available") from self._dead
            if self._closed:
                raise RuntimeError("cannot schedule new futures after shutdown")
            i = next(self._ids)
            f = _HostFuture(self, i)
            self._futs[i] = f
        try:
            with self._send_lock:
                self._conn.send(("job", i, fn, args))
        except Exception as e:  # noqa: BLE001
            self._futs.pop(i, None)
            self._die(e)
            raise HostBroken("processing host is not available") from e
        return f

    def _merge_prof(self, p):
        if not p:
            return
        a = self._prof
        for k in ("n", "thr", "idle", "convoy"):
            a[k] = a.get(k, 0) + int(p.get(k, 0))
        for key, lst in (("labels", p.get("labels", ())), ("conv_labels", p.get("conv_labels", ()))):
            d = a.setdefault(key, {})
            for lab, c in lst:
                d[lab] = d.get(lab, 0) + c

    def take_profile(self) -> dict:
        """накопичений знімок «де стоять потоки господаря» з минулого виклику (для журналу експорту)"""
        a, self._prof = self._prof, {}
        return a

    def partial_frames(self) -> int:
        """скільки кадрів уже порахували завдання, що ще йдуть (для плавної смуги прогресу експорту)"""
        try:
            return int(sum(list(self._part.values())))
        except Exception:  # noqa: BLE001
            return 0

    def _cancel(self, i):
        self._part.pop(i, None)
        self._futs.pop(i, None)
        try:
            with self._send_lock:
                self._conn.send(("cancel", i))
        except Exception:  # noqa: BLE001
            pass

    def _read(self):
        err: BaseException = EOFError("host exited")
        while True:
            try:
                i, kind, val = self._conn.recv()
            except (EOFError, OSError, ValueError) as e:
                err = e
                break
            except Exception as e:  # noqa: BLE001           # результат не розпіклювався
                err = e
                break
            if kind == "tick":
                if i in self._futs:
                    self._part[i] = val
                continue
            if kind == "stat":
                self._merge_prof(val.pop("prof", None))
                self.stats = val
                continue
            self._part.pop(i, None)
            f = self._futs.pop(i, None)
            if f is None:
                continue
            if kind == "cancel":
                Future.cancel(f)
            elif f.set_running_or_notify_cancel():
                if kind == "ok":
                    f.set_result(val)
                else:
                    f.set_exception(val)
        self._die(err)

    def _die(self, err: BaseException):
        with self._lock:
            if self._dead is None:
                self._dead = err
            pend, self._futs = list(self._futs.values()), {}
            self._part.clear()
        for f in pend:
            try:
                if f.set_running_or_notify_cancel():
                    f.set_exception(HostBroken("processing host stopped"))
            except Exception:  # noqa: BLE001
                pass

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        with self._lock:
            self._closed = True
            pend = list(self._futs.values()) if cancel_futures else []
        for f in pend:
            Future.cancel(f)
        if cancel_futures:
            self._futs.clear()
        try:
            with self._send_lock:
                self._conn.send(("stop",))
        except Exception:  # noqa: BLE001
            pass
        if wait:
            self._proc.join(5)
