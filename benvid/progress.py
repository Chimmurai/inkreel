"""Прогрес: єдиний формат для консолі й GUI.  Колбек: progress(frac 0..1, label)."""
from __future__ import annotations

import sys
import time


def fmt_t(sec: float) -> str:
    m, s = divmod(int(sec + 0.5), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def sub(progress, lo: float, hi: float, label: str):
    """частина загальної смуги [lo,hi] для одного етапу: повертає колбек f -> progress(lo+(hi-lo)f, label)"""
    if progress is None:
        return None
    return lambda f: progress(lo + (hi - lo) * min(1.0, max(0.0, f)), label)


class Clock:
    """рахує час і ETA за пройденою часткою; час на паузі не рахується.
    Швидкість і ETA беруться за останні ~WIN секунд, а не за весь час: старт (запуск обробника, прогрів ядер, перший шматок) довго
    стоїть без кадрів, і середнє за весь час тримало б «0 к/с», а потім занижувало б швидкість ще хвилини"""
    WIN = 12.0                           # вікно швидкості, с
    MIN_SPAN = 3.0                       # раніше за стільки секунд історії вірити їй рано - беремо середнє

    def __init__(self):
        self.t0 = time.time()
        self._held = 0.0                 # скільки секунд уже простояли на паузі
        self._since = None               # коли поточну паузу почато
        self._hist: list = []            # (минуло с без пауз, частка) - не частіше разу на 0.25 с

    def hold(self, on: bool):
        """пауза (True) / продовження (False)"""
        if on and self._since is None:
            self._since = time.time()
        elif not on and self._since is not None:
            self._held += time.time() - self._since
            self._since = None

    def elapsed(self) -> float:
        now = self._since if self._since is not None else time.time()
        return max(0.0, now - self.t0 - self._held)

    def _mark(self, frac: float, el: float):
        h = self._hist
        if h and (frac < h[-1][1] or el - h[-1][0] < 0.25):
            return
        h.append((el, frac))
        while len(h) > 2 and h[1][0] < el - self.WIN:        # старіше вікна - геть, але лишаємо одну точку «до» вікна
            h.pop(0)

    def speed(self):
        """частка за секунду за останні ~WIN с; None, коли історії ще замало або кадри не йшли"""
        h = self._hist[:]                                    # копія: історію дописує потік експорту, а читає інтерфейс
        if len(h) < 2:
            return None
        (t0, f0), (t1, f1) = h[0], h[-1]
        if t1 - t0 < self.MIN_SPAN or f1 <= f0:
            return None
        return (f1 - f0) / (t1 - t0)

    def eta(self, frac: float):
        el = self.elapsed()
        self._mark(frac, el)
        if frac < 0.03:
            return None
        r = self.speed()
        return el * (1 - frac) / frac if r is None else (1 - frac) / r


class ConsoleBar:
    """Рядок у консолі: 'Кодування  [██████░░░░░░] 52%  мин 0:12  залишилось ~0:11'"""
    def __init__(self, label="", width=28, stream=None):
        self.label, self.width, self.clock, self.out = label, width, Clock(), stream or sys.stderr
        self._last = -1.0

    def __call__(self, frac: float, label: str | None = None):
        if label:
            self.label = label
        frac = min(1.0, max(0.0, frac))
        if frac < 1.0 and frac - self._last < 0.005 and self._last >= 0:
            return
        self._last = frac
        n = int(frac * self.width)
        eta = self.clock.eta(frac)
        tail = f"  залишилось ~{fmt_t(eta)}" if eta is not None and frac < 1 else ""
        self.out.write(f"\r  {self.label:<16} [{'█' * n}{'░' * (self.width - n)}] {frac*100:3.0f}%  "
                       f"{fmt_t(self.clock.elapsed())}{tail}   ")
        self.out.flush()

    def close(self, note="готово"):
        self(1.0)
        self.out.write(f"\r  ✓ {self.label:<14} {note} за {fmt_t(self.clock.elapsed())}{' ' * 40}\n")
        self.out.flush()
