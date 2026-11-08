"""Автоматичний вибір розміру: заміряємо, скільки коштує кадр НА ЦЬОМУ ПК тим рівнем якості, який вибрано зараз, і беремо
найбільший розмір, що встигає в реальному часі (з запасом). Зразок кадру - з самого файлу.

Рівні 1-4 й «Ультру» (5) відео рахує пул процесів: вартість міряємо в одному потоці й множимо на кількість обробників.
«Ультра» GIF у плеєрі йде в потоках цього процесу (кожен кадр шматка спирається на попередній), тому її міряємо кількома
потоками одночасно (threads > 1), і на воркерів уже не множимо: виміряна швидкість містить усе, що на цьому ПК справді виходить."""
from __future__ import annotations

import threading
import time

import cv2

from .core import InkState, Params, binarize

CANDIDATES = (144, 240, 360, 480, 720, 1080)
EFFICIENCY = 0.65          # частка потужності, що реально лягає на обробку (решта - декодування, інтерфейс, пересилання)
HEADROOM = 1.3             # запас швидкості над fps відео
SMALL_H = 240              # дрібний шлях обробки (< 290 px висоти) міряємо на цій висоті
BIG_H = {5: (480, 720)}           # великий шлях: у «Ультри» великі сталі витрати (аналіз на фіксованій сітці) - міряємо на двох висотах
BIG_DEFAULT = (480,)


def _dims(frame, h):
    return max(2, int(h * frame.shape[1] / frame.shape[0]) // 2 * 2), h


def _per_pixel(frame, h, method, threads=1, reps=2, glyph=False, gdens=3) -> float:
    """секунд стінного часу на піксель, якщо кадри висоти h рахують `threads` потоків одночасно (1 - як один воркер)"""
    w, h = _dims(frame, h)
    p = Params(method=method, glyph=glyph, gdens=gdens)

    def state():
        if method == 6:                                         # нейро: пам'ять - лише згладжування опису кадру
            from .student import FilmState
            return {"film": FilmState()}
        st = InkState() if method == 5 else None
        return {"state": st} if st is not None else None

    if threads <= 1:
        aux = state()
        binarize(frame, w, h, p, aux)                           # прогрів: кеш numba, перший кадр без часової пам'яті
        t = time.perf_counter()
        for _ in range(reps):
            binarize(frame, w, h, p, aux)
        return (time.perf_counter() - t) / reps / (w * h)
    bar = threading.Barrier(threads + 1)
    errs = []

    def run():
        aux = None
        try:
            aux = state()
            binarize(frame, w, h, p, aux)
        except Exception as e:  # noqa: BLE001
            errs.append(e)
            aux = None
        try:
            bar.wait(120)
        except threading.BrokenBarrierError:
            return
        if aux is None:
            return
        try:
            for _ in range(reps):
                binarize(frame, w, h, p, aux)
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=run, daemon=True) for _ in range(threads)]
    for t in ts:
        t.start()
    bar.wait(120)
    t0 = time.perf_counter()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    if errs:
        raise errs[0]
    return dt / (threads * reps) / (w * h)


def measure(frame, method: int = 4, threads: int = 1, glyph: bool = False, gdens: int = 3):
    """-> (с/піксель дрібного шляху, {висота: с/піксель великого шляху}). threads=1 - один потік, як воркер пулу."""
    pinned = threads <= 1
    old_cv = cv2.getNumThreads()
    old_nb = None
    if pinned:
        cv2.setNumThreads(1)
        try:
            import numba
            old_nb = numba.get_num_threads()
            numba.set_num_threads(1)
        except Exception:  # noqa: BLE001
            old_nb = None
    try:
        small = _per_pixel(frame, SMALL_H, method, threads, glyph=glyph, gdens=gdens)
        big = {h: _per_pixel(frame, h, method, threads, glyph=glyph, gdens=gdens) for h in BIG_H.get(method, BIG_DEFAULT)}
        return small, big
    finally:
        if pinned:
            cv2.setNumThreads(old_cv)
            if old_nb is not None:
                try:
                    import numba
                    numba.set_num_threads(old_nb)
                except Exception:  # noqa: BLE001
                    pass


def _frame_cost(small: float, big: dict, w: float, h: int, method: int) -> float:
    """секунд на кадр w×h. Малий шлях (< 290 px) - за виміром на 240p; великий - за найбільшою виміряною висотою, що не
    перевищує h (далі лінійно вгору - з запасом: сталі витрати на піксель із ростом кадру лише розмазуються).
    Нижче найменшої виміряної висоти «Ультра» не дешевшає пропорційно пікселям (її аналіз іде на фіксованій сітці), тож беремо
    вартість найменшого виміряного кадру цілком - це теж із запасом."""
    if h < 290:
        return small * w * h
    below = [k for k in big if k <= h]
    if below:
        k = max(below)
        return big[k] * w * h
    k = min(big)
    kw = k * w / h
    return big[k] * kw * k if method == 5 else big[k] * w * h


def pick(frame, src_fps: float, workers: int, src_h: int | None = None, costs=None, method: int = 4, threads: int = 1,
         glyph: bool = False, gdens: int = 3) -> str:
    """-> ключ розміру ('360p', ...). Не більше, ніж висота джерела. method - рівень якості, threads - скільки потоків рахують
    «Ультру» (на рівнях 1-4 рахує пул з `workers` процесів)."""
    small, big = costs or measure(frame, method, threads, glyph, gdens)
    par = 1 if threads > 1 else max(1, workers)
    aspect = frame.shape[1] / frame.shape[0]
    cap = src_h or frame.shape[0]
    best = CANDIDATES[0]
    for h in CANDIDATES:
        if h > cap * 1.05 and h != CANDIDATES[0]:
            break
        w = h * aspect
        cost = _frame_cost(small, big, w, h, method)
        fps = par * EFFICIENCY / max(cost, 1e-9)
        if fps >= src_fps * HEADROOM:
            best = h
    return f"{best}p"
