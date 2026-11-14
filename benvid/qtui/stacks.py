"""Прокрутка правої панелі (ScrollStack) і сторінка з екранів (PageStack): код перенесено з `benvid/gui.py` дослівно; працює через Tk-подібні методи
віджетів (`place`, `winfo_*`, `after`), які дає `qtui/tkcompat.py`."""
from __future__ import annotations

import time


class ScrollStack:
    """вертикальна прокрутка стопки віджетів у «вікні» view: кожен віджет ставиться place-ом у свою висоту, зсув - спільний.
    Без вкладених вікон Canvas, тож нічого не мерехтить, а все, що за межами вікна, обрізає сам Tk.
    items - [(віджет, висота(ширина_вмісту, висота_вікна) -> px, підготовка(ширина_вмісту) або None)];
    bar_always - смужка видна завжди (сторінка з футером), інакше лише коли вміст не влазить;
    gap - проміжок між вмістом і смужкою (на сторінці 0: вміст має власне поле M, і воно однакове з обох боків)"""
    NOTCH = 64

    def __init__(self, view, bar, items, bar_always=False, gap=0):
        self.view, self.bar, self.items, self.bar_always, self.gap = view, bar, items, bar_always, gap
        self.off = self.max_off = self.total = 0
        self.enabled, self._busy, self._anim, self._target = True, False, None, 0
        bar.cmd = lambda f: self.scroll_to(f * self.max_off)
        bar.page_cmd = lambda s: self.scroll_by(s * max(40, self.view.winfo_height() - 48), animate=True)
        view.bind("<Configure>", lambda e: self.layout(), add="+")

    @property
    def overflow(self) -> bool:
        return self.enabled and self.max_off > 0

    def _pin(self, hs, vh):
        """зсув після зміни розмірів: тримаємо в межах (сторінка з екранами перевизначає: лишається на своєму екрані)"""
        return min(max(self.off, 0), self.max_off)

    def layout(self):
        if self._busy:
            return
        self._busy = True
        try:
            vw, vh = self.view.winfo_width(), self.view.winfo_height()
            if vw < 8 or vh < 8:
                return
            bw = self.bar.bw + self.gap
            cw = vw - bw if (self.bar_always and self.enabled) else vw

            def measure(cw):
                for _, _, prep in self.items:
                    if prep:
                        prep(cw)
                return [max(1, int(h(cw, vh))) for _, h, _ in self.items]
            hs = measure(cw)
            if not self.bar_always and self.enabled and sum(hs) > vh:
                cw = vw - bw
                hs = measure(cw)
            self.total = sum(hs)
            self.max_off = max(0, self.total - vh) if self.enabled else 0
            self.off = self._pin(hs, vh)
            y = -self.off
            for (fr, _, _), h in zip(self.items, hs):
                fr.place(x=0, y=y, width=cw, height=h)
                y += h
            if self.enabled and (self.bar_always or self.max_off > 0):
                self.bar.place(x=vw - self.bar.bw, y=0, width=self.bar.bw, height=vh)
                self.bar.set(self.off / self.total, (self.off + vh) / self.total)
            else:
                self.bar.place_forget()
        finally:
            self._busy = False

    def scroll_to(self, y, animate=False):
        y = min(max(int(y), 0), self.max_off)
        self._target = y
        if self._anim:
            self.view.after_cancel(self._anim)
            self._anim = None
        if not animate or abs(y - self.off) < 4:
            self.off = y
            self.layout()
            return
        start, t0, dur = self.off, time.perf_counter(), 0.2

        def step():
            k = min(1.0, (time.perf_counter() - t0) / dur)
            self.off = int(round(start + (y - start) * (1 - (1 - k) ** 3)))      # плавне гальмування
            self.layout()
            self._anim = self.view.after(15, step) if k < 1.0 else None
        step()

    def scroll_by(self, dy, animate=True):
        self.scroll_to((self._target if self._anim else self.off) + dy, animate)

    def notch(self, d):
        """одне «клацання» колеса: d > 0 - вгору (вміст їде вниз)"""
        self.scroll_by(-d * self.NOTCH, True)

    def set_enabled(self, v):
        self.enabled = v
        if not v:
            self.off = 0
        self.layout()


class PageStack(ScrollStack):
    """сторінка з екранів (програма, футер), кожен - на все вікно. Прокрутка не повзе по пікселях, а перемикає екран плавною
    анімацією й фіксується на ньому: колесо, PageUp/PageDown, клік по доріжці смужки, «?» і F1 - один жест = один перехід,
    в обидва боки. Екран вищий за вікно (низьке вікно) спершу прокручується всередині, а за його краєм - перехід далі.
    Положення - пара (екран cur, зсув rel у ньому), а не число пікселів: зміна розміру вікна його не збиває.
    tcur/trel - куди їдемо (збігається з cur/rel, коли стоїмо)."""
    FLIP = 0.46            # перехід на висоту вікна, с (коротша відстань - пропорційно коротше)
    COOL = 0.20            # після переходу колесо ще мить мовчить: інерційний тачпад шле хвіст подій - це один жест, а не кілька
    EDGE = 0.30            # так само після прокрутки до краю екрана (щоб той самий жест не перекинув далі)

    def __init__(self, view, bar, items):
        super().__init__(view, bar, items, bar_always=True)
        self.cur = self.rel = self.tcur = self.trel = 0
        self.tops, self.exts = [0] * len(items), [0] * len(items)
        self._free, self._until, self._done = False, 0.0, None     # _free: їдемо чи тягнуть смужку - положення тоді число off
        bar.cmd = self._drag
        bar.page_cmd = self.step
        bar.release_cmd = self._release

    # --- розкладка: де чиї верхи, і «стоянка» після зміни розмірів
    def _pin(self, hs, vh):
        y = 0
        for i, h in enumerate(hs):
            self.tops[i], self.exts[i] = y, max(0, h - vh)
            y += h
        if not self.enabled:
            self.cur = self.rel = self.tcur = self.trel = 0
            return 0
        if self._free:
            return min(max(self.off, 0), self.max_off)
        self.cur = min(self.cur, len(hs) - 1)
        self.rel = min(max(self.rel, 0), self.exts[self.cur])
        self.tcur, self.trel = self.cur, self.rel
        return min(self.tops[self.cur] + self.rel, self.max_off)

    def _locate(self, y):
        """положення y (px від верху сторінки) -> найближча «стоянка» (екран, зсув у ньому); між екранами - той, до якого ближче"""
        n = len(self.tops)
        i = max((k for k in range(n) if y >= self.tops[k]), default=0)
        end = self.tops[i] + self.exts[i]
        if y <= end or i == n - 1:
            return i, min(max(y - self.tops[i], 0), self.exts[i])
        return (i, self.exts[i]) if y - end < self.tops[i + 1] - y else (i + 1, 0)

    @property
    def idle(self) -> bool:
        return self._anim is None and not self._free

    def _stop_anim(self):
        if self._anim:
            self.view.after_cancel(self._anim)
            self._anim = None

    # --- рух
    def _glide(self, i, rel, done=None):
        """плавно до (екран i, зсув rel); done() - коли доїхали"""
        i = min(max(i, 0), len(self.tops) - 1)
        rel = min(max(rel, 0), self.exts[i])
        self._stop_anim()
        self.tcur, self.trel, self._done = i, rel, done
        y, start, vh = self.tops[i] + rel, self.off, max(1, self.view.winfo_height())
        dist = abs(y - start)
        flip = dist > 0.5 * vh                              # перехід між екранами (а не крок усередині одного)
        dur = min(self.FLIP, max(0.16, self.FLIP * dist / vh)) if dist >= 4 else 0.0
        t0 = time.perf_counter()
        if flip:
            self._until = t0 + dur + self.COOL
        self._free = True

        def finish():
            self._anim, self._free = None, False
            self.cur, self.rel = i, rel
            self.layout()                                   # стає на своє місце вже за актуальними розмірами
            cb, self._done = self._done, None
            if cb:
                cb()

        def step():
            k = min(1.0, (time.perf_counter() - t0) / dur) if dur > 0 else 1.0
            e = k * k * (3 - 2 * k) if flip else 1 - (1 - k) ** 3          # екрани - плавний розгін і гальмування; крок - лише гальмування
            self.off = int(round(start + (y - start) * e))
            self.layout()
            if k < 1.0:
                self._anim = self.view.after(15, step)
            else:
                finish()
        step()

    def go(self, i, rel=None, done=None):
        """до екрана i (з анімацією); rel - зсув усередині (за замовчуванням: уперед - угору екрана, назад - у низ)"""
        i = min(max(i, 0), len(self.tops) - 1)
        self._glide(i, (0 if i >= self.tcur else self.exts[i]) if rel is None else rel, done)

    def _ready(self) -> bool:
        return time.perf_counter() >= self._until

    def _move(self, d, amount):
        """d > 0 - вниз, d < 0 - вгору: прокрутити всередині екрана на amount px, а коли дійшли до краю - перейти до сусіднього"""
        if not self.enabled or not d:
            return
        i, rel = self.tcur, self.trel
        if d > 0:
            if rel < self.exts[i]:
                self._inner(i, min(self.exts[i], rel + amount))
            elif i + 1 < len(self.tops) and self._ready():
                self._glide(i + 1, 0)
        elif rel > 0:
            self._inner(i, max(0, rel - amount))
        elif i > 0 and self._ready():
            self._glide(i - 1, self.exts[i - 1])

    def _inner(self, i, rel):
        self._glide(i, rel)
        if rel in (0, self.exts[i]):
            self._until = max(self._until, time.perf_counter() + self.EDGE)

    def notch(self, d):
        """одне «клацання» колеса: d > 0 - вгору"""
        self._move(-1 if d > 0 else 1, self.NOTCH)

    def step(self, d):
        """PageUp/PageDown і клік по доріжці смужки: d = ±1"""
        self._move(d, max(40, self.view.winfo_height() - 48))

    def scroll_by(self, dy, animate=True):
        self._move(1 if dy > 0 else -1, abs(dy))

    def scroll_to(self, y, animate=False):
        """на положення y (px від верху сторінки): найближча «стоянка»"""
        i, rel = self._locate(min(max(int(y), 0), self.max_off))
        if animate:
            self._glide(i, rel)
            return
        self._stop_anim()
        self._free = False
        self.cur = self.tcur = i
        self.rel = self.trel = rel
        self.layout()

    # --- смужка сторінки: тягнуть повзунок - екрани їдуть за ним, відпустили - «прилипають» до найближчого
    def _drag(self, f):
        self._stop_anim()
        self._free = True
        self.off = int(round(f * self.max_off))
        self.layout()

    def _release(self):
        if self._free and self._anim is None:
            self._glide(*self._locate(self.off))

    def set_enabled(self, v):
        if not v:
            self._stop_anim()
            self._free = False
        super().set_enabled(v)
