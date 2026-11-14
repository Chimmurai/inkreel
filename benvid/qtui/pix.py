"""Піксельні віджети InkReel: кнопка, повзунок (з прапорцями фрагмента й буфером), панель якості, прогрес, смужка прокрутки.
Малювання перенесено з `benvid/gui.py` майже дослівно: ті самі виклики create_* на `TkCanvas` (див. tkcanvas.py), ті самі розміри й кольори,
тож вигляд збігається з Tk до пікселя (перевірка: `tools/ui_diff.py widgets`)."""
from __future__ import annotations

import math

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QSizePolicy

from .. import eggs
from ..i18n import tr
from . import assets
from .consts import FLAG_H, FONT, FONT_S, H_M, ICON_GAP, THUMB
from .theme import THEME, tk_font
from .tkcanvas import TkCanvas


def BG() -> str:
    return THEME.bg.name()


def FG() -> str:
    return THEME.fg.name()


def DIM() -> str:
    return THEME.dim.name()


def tw(text: str, font=FONT) -> int:
    """ширина рядка в пікселях у шрифті font"""
    return QFontMetrics(tk_font(font)).horizontalAdvance(text)


class _Base(TkCanvas):
    def __init__(self, parent=None):
        super().__init__(parent, bg=BG())
        THEME.changed.connect(self._theme)

    def _theme(self):
        self.bg = BG()
        self.draw()
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.draw()

    def config(self, cursor=None):
        if cursor is not None:
            self.setCursor({"hand2": Qt.PointingHandCursor, "arrow": Qt.ArrowCursor, "sb_h_double_arrow": Qt.SizeHorCursor}.get(cursor, Qt.ArrowCursor))

    def draw(self):
        raise NotImplementedError


class PixButton(_Base):
    """плоска пікселізована кнопка: чорна з білою рамкою, при натисканні інвертується. Значок і підпис центруються разом як одне ціле.
    join="l" - кнопка «склеєна» з лівим сусідом: власну ліву лінію рамки не малює"""
    ICON_W = 20
    FLASH = True

    def __init__(self, parent=None, text="", image=None, command=None, width=None, height=H_M, toggle=False, icon=None, join=None):
        super().__init__(parent)
        w = width or (len(text) * 9 + 22 if text else 38)
        self.text, self.cmd, self.on, self.enabled, self.toggle = text, command, False, True, toggle
        self.im, self._bw, self._bh, self.hover, self.icon, self.join = image, w, height, False, icon, join
        self.bites, self.crumbs = 0, []
        self.on_press = None
        self._pressed = False
        self._want = (w, height)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setFixedHeight(height)
        self.resize(w, height)

    def sizeHint(self):
        return QSize(*self._want)

    def minimumSizeHint(self):
        return QSize(min(self._want[0], 24), self._want[1])

    def resizeEvent(self, e):
        self._bw, self._bh = self.width(), self.height()
        self.draw()

    def set_icon(self, name):
        self.icon = name
        self.draw()
        self.update()

    def set_text(self, t):
        self.text = t
        self.draw()
        self.update()

    def set_on(self, v):
        self.on = v
        self.draw()
        self.update()

    def set_enabled(self, v):
        self.enabled = v
        self.config(cursor="hand2" if v else "arrow")
        self.draw()
        self.update()

    def enterEvent(self, e):
        self.hover = True
        self.draw()
        self.update()

    def leaveEvent(self, e):
        self.hover = False
        self.draw()
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.on_press:
            self.on_press()                              # як bind("<Button-1>", add="+") у Tk: спрацьовує й на вимкненій кнопці (пасхалка «печиво»)
        if e.button() == Qt.LeftButton and self.enabled:
            self._pressed = True
            self.draw()
            self.update()

    def mouseReleaseEvent(self, e):
        was = self._pressed
        self._pressed = False
        if self.toggle and was and self.enabled:
            self.on = not self.on
        self.draw()
        self.update()
        p = e.position()
        if was and self.enabled and 0 <= p.x() <= self._bw and 0 <= p.y() <= self._bh and self.cmd:
            self.cmd()

    def draw(self):
        self.delete("all")
        bgc, fgc, dim = BG(), FG(), DIM()
        inv = self.on or (self._pressed and self.FLASH)
        bg, fg = (fgc, bgc) if inv else (bgc, fgc)
        if not self.enabled:
            fg = dim
        w, h = self._bw, self._bh
        x0 = -1 if self.join == "l" else 0
        self.create_rectangle(x0, 0, w - 1, h - 1, fill=bg, outline=fgc if self.enabled else dim)
        if self.hover and self.enabled and not inv:
            self.create_rectangle(2, 2, w - 3, h - 3, outline=fgc, dash=(1, 2))
        im, cy = self.im, h // 2
        ink = (getattr(im, "ink", None) or (0, 0, im.width(), im.height())) if im is not None and not self.icon else None
        iw = self.ICON_W if self.icon else (ink[2] - ink[0] if ink else 0)
        t = tw(self.text) if self.text else 0
        x = (w - (iw + (ICON_GAP if iw and t else 0) + t)) // 2
        if self.icon:
            self._icon(self.icon, x + iw // 2, cy, fg, bg)
        elif ink:
            self.create_image(x - ink[0], cy - (ink[3] - ink[1]) // 2 - ink[1], image=im, anchor="nw")
        if self.text:
            self.create_text(x + iw + (ICON_GAP if iw else 0), cy, text=self.text, fill=fg, font=FONT, anchor="w")

    def _icon(self, n, cx, cy, c, bg=None):
        bg = bg or BG()
        r = lambda x0, y0, x1, y1: self.create_rectangle(cx + x0, cy + y0, cx + x1, cy + y1, fill=c, outline=c)
        p = lambda *pts: self.create_polygon(*[v + (cx if i % 2 == 0 else cy) for i, v in enumerate(pts)], fill=c, outline=c)
        if n == "play":
            p(-6, -9, 8, 0, -6, 9)
        elif n == "pause":
            r(-8, -9, -3, 9)
            r(3, -9, 8, 9)
        elif n == "stop":
            r(-8, -8, 8, 8)
        elif n == "prev":
            r(-9, -8, -6, 8)
            p(8, -8, -5, 0, 8, 8)
        elif n == "rew":
            p(0, -8, -9, 0, 0, 8)
            p(9, -8, 0, 0, 9, 8)
        elif n == "fwd":
            p(0, -8, 9, 0, 0, 8)
            p(-9, -8, 0, 0, -9, 8)
        elif n in ("snd", "mute"):
            r(-9, -3, -5, 3)
            p(-5, -3, 1, -8, 1, 8, -5, 3)
            if n == "snd":
                self.create_arc(cx + 1, cy - 6, cx + 11, cy + 6, start=-60, extent=120, style="arc", outline=c, width=2)
            else:
                self.create_line(cx + 3, cy - 5, cx + 11, cy + 5, fill=c, width=2)
                self.create_line(cx + 3, cy + 5, cx + 11, cy - 5, fill=c, width=2)
        elif n == "left":
            p(5, -8, -5, 0, 5, 8)
        elif n == "right":
            p(-5, -8, 5, 0, -5, 8)
        elif n == "next":
            r(6, -8, 9, 8)
            p(-8, -8, 5, 0, -8, 8)
        elif n == "up":
            p(-7, 5, 0, -5, 7, 5)
        elif n == "menu":
            r(-8, -7, 8, -5)
            r(-8, -1, 8, 1)
            r(-8, 5, 8, 7)
        elif n == "cookie":
            if self.bites < eggs.EATEN:
                self.create_oval(cx - 9, cy - 9, cx + 9, cy + 9, fill=c, outline=c)
                self.create_oval(cx + 3, cy - 13, cx + 13, cy - 3, fill=bg, outline=bg)
                for dx, dy in ((-5, -3), (1, 0), (-2, 5), (5, 5), (-6, 3), (-1, -6)):
                    self.create_rectangle(cx + dx, cy + dy, cx + dx + 2, cy + dy + 2, fill=bg, outline=bg)
                for dx, dy, rr in eggs.BITES[:self.bites]:
                    self.create_oval(cx + dx - rr, cy + dy - rr, cx + dx + rr, cy + dy + rr, fill=bg, outline=bg)
            for x, y, _, _, sz in self.crumbs:
                self.create_rectangle(cx + x, cy + y, cx + x + sz, cy + y + sz, fill=c, outline=c)
        elif n == "swap":
            R, step, cnt = 8.0, 16.0, 7
            for a0 in (200.0, 20.0):
                pts = []
                for k in range(cnt + 1):
                    a = math.radians(a0 + k * step)
                    pts += [cx + R * math.cos(a), cy - R * math.sin(a)]
                self.create_line(*pts, fill=c, width=2)
                a = math.radians(a0 + cnt * step)
                ex, ey = cx + R * math.cos(a), cy - R * math.sin(a)
                tx, ty = -math.sin(a), -math.cos(a)
                nx, ny = -ty, tx
                self.create_polygon(ex + 4.5 * tx, ey + 4.5 * ty, ex - 0.5 * tx + 3.5 * nx, ey - 0.5 * ty + 3.5 * ny,
                                    ex - 0.5 * tx - 3.5 * nx, ey - 0.5 * ty - 3.5 * ny, fill=c, outline=c)


class PixSlider(_Base):
    """повзунок/смуга перемотування: піксельна доріжка з квадратною ручкою; marks - початок і кінець фрагмента (прапорці-дужки «[ ]»)"""

    def __init__(self, parent=None, from_=0.0, to=1.0, command=None, height=22, on_release=None, steps=None):
        super().__init__(parent)
        self.lo, self.hi, self.cmd, self.on_release, self.steps = from_, to, command, on_release, steps
        self.v, self.drag, self.h, self.ranges = from_, False, height, []
        self.marks, self.mark_cb, self._dm, self._hot = None, None, None, None
        self.setFixedHeight(height)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, v, redraw=True):
        self.v = min(max(v, self.lo), self.hi)
        if redraw and not self.drag:
            self.draw()
            self.update()

    def get(self):
        return self.v

    def set_ranges(self, rs):
        if rs != self.ranges:
            self.ranges = rs
            self.draw()
            self.update()

    def _mark_x(self, v):
        W = max(20, self.width())
        return THUMB + (v - self.lo) / max(1e-9, self.hi - self.lo) * (W - 1 - 2 * THUMB)

    def _ty(self):
        return FLAG_H + (self.h - FLAG_H) // 2 if self.marks is not None else self.h // 2

    def _flag_at(self, x, y):
        if not self.marks or self.hi <= self.lo or y > FLAG_H + 2:
            return None
        xa, xb = self._mark_x(self.marks[0]), self._mark_x(self.marks[1])
        da, db = abs(x - xa), abs(x - xb)
        if min(da, db) > 12:
            return None
        if da == db:
            return 0 if x <= (xa + xb) / 2 else 1
        return 0 if da < db else 1

    def _set_hot(self, i):
        if i != self._hot:
            self._hot = i
            self.config(cursor="sb_h_double_arrow" if i is not None else "hand2")
            self.draw()
            self.update()

    def mouseMoveEvent(self, e):
        p = e.position()
        if e.buttons() & Qt.LeftButton:
            self._mv(p.x(), p.y())
        elif self._dm is None and not self.drag:
            self._set_hot(self._flag_at(p.x(), p.y()))

    def leaveEvent(self, e):
        self._set_hot(None)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            p = e.position()
            self._mv(p.x(), p.y())

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._up()

    def _mv(self, x, y):
        w = max(1, self.width() - 1 - 2 * THUMB)
        f = min(max((x - THUMB) / w, 0.0), 1.0)
        v = self.lo + f * (self.hi - self.lo)
        if self.marks and self.hi > self.lo:
            if self._dm is None and not self.drag:
                self._dm = self._flag_at(x, y)
                if self._dm is not None:
                    self._hot = self._dm
            if self._dm is not None:
                v = round(v)
                lo, hi = (self.lo, self.marks[1] - 1) if self._dm == 0 else (self.marks[0] + 1, self.hi)
                self.marks[self._dm] = min(max(v, lo), max(lo, hi))
                self.draw()
                self.update()
                if self.mark_cb:
                    self.mark_cb(self._dm, self.marks[self._dm])
                return
        if self.steps:
            v = round(v)
        self.drag, self.v = True, v
        self.draw()
        self.update()
        if self.cmd:
            self.cmd(self.v)

    def _up(self):
        if self._dm is not None:
            self._dm = None
            self._hot = None
            self.config(cursor="hand2")
            self.draw()
            self.update()
            return
        self.drag = False
        self.draw()
        self.update()
        if self.on_release:
            self.on_release(self.v)

    def _flag(self, x, side, c):
        top, bot = 1, FLAG_H - 1
        x0, x1 = (x - 1, x + 7) if side > 0 else (x - 7, x + 1)
        self.create_rectangle(x - 1, top, x + 1, bot, fill=c, outline=c)
        self.create_rectangle(x0, top, x1, top + 2, fill=c, outline=c)
        self.create_rectangle(x0, bot - 2, x1, bot, fill=c, outline=c)

    def draw(self):
        self.delete("all")
        BGc, FGc, DIMc = BG(), FG(), DIM()
        W = max(20, self.width())
        y = self._ty()
        f = 0 if self.hi == self.lo else (self.v - self.lo) / (self.hi - self.lo)
        L = W - 1 - 2 * THUMB
        x = THUMB + f * L
        self.create_rectangle(THUMB, y - 3, W - 1 - THUMB, y + 3, outline=FGc, fill=BGc)
        if self.steps and self.hi > self.lo:
            for i in range(int(self.lo), int(self.hi) + 1):
                tx = THUMB + (i - self.lo) / (self.hi - self.lo) * L
                self.create_line(tx, y + 5, tx, y + 10, fill=FGc)
        if self.hi > self.lo:
            for a, b in self.ranges:
                ax = THUMB + (a - self.lo) / (self.hi - self.lo) * L
                bx = THUMB + (b - self.lo) / (self.hi - self.lo) * L
                self.create_rectangle(ax, y - 2, max(bx, ax + 1), y + 2, outline=DIMc, fill=DIMc)
        if x > THUMB + 1:
            self.create_rectangle(THUMB + 1, y - 2, x, y + 2, outline=FGc, fill=FGc)
        if self.marks and self.hi > self.lo:
            ax, bx = int(round(self._mark_x(self.marks[0]))), int(round(self._mark_x(self.marks[1])))
            lim = self.marks[0] > self.lo or self.marks[1] < self.hi
            if lim:
                for u, v in ((THUMB, ax), (bx, W - 1 - THUMB)):
                    if v - u > 1:
                        for j, yy in enumerate((y - 3, y - 2, y - 1, y, y + 1, y + 2, y + 3)):
                            self.create_line(u + 1, yy, v, yy, fill=BGc, dash=(1, 1), dashoffset=j % 2)
                self.create_rectangle(ax, 1, bx, 3, fill=FGc, outline=FGc)
            for i, (fx, side) in enumerate(((ax, 1), (bx, -1))):
                c = FGc if (lim or self._hot == i or self._dm == i) else DIMc
                self._flag(fx, side, c)
                self.create_line(fx, FLAG_H - 1, fx, y - 4, fill=c)
        self.create_rectangle(x - THUMB, y - 8, x + THUMB, y + 8, outline=FGc, fill=FGc if self.drag else BGc)


class QualityBar(_Base):
    """баланс «швидкість ↔ якість»: п'ять гуртків на одній лінії, зафарбований лише обраний; branch - ромб на пунктирному відгалуженні"""
    PAD = 10
    BR = 64

    def __init__(self, parent=None, n=5, value=4, command=None, ends=("швидко", "якісно"), branch=None):
        super().__init__(parent)
        self.n, self.v, self.cmd, self.hov, self.ends = n, value, command, None, ends
        self.branch = branch
        self.branch_on = True
        self.enabled = True
        self.setFixedHeight(44)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _xs(self):
        W = max(60, self.width())
        span = W - 2 * self.PAD - (self.BR if self.branch else 0)
        return [self.PAD + i * span / (self.n - 1) for i in range(self.n)]

    def _bx(self):
        return max(60, self.width()) - self.PAD - 2

    def _at(self, x):
        xs = self._xs()
        if self.branch and self.branch_on:
            xs = xs + [self._bx()]
        return min(range(len(xs)), key=lambda i: abs(xs[i] - x)) + 1

    def _hover(self, i):
        if not self.enabled:
            i = None
        if i != self.hov:
            self.hov = i
            self.draw()
            self.update()

    def set_enabled(self, v):
        self.enabled = bool(v)
        self.config(cursor="hand2" if v else "arrow")
        self.hov = None
        self.draw()
        self.update()

    def set_branch(self, on):
        self.branch_on = bool(on)
        self.hov = None
        self.draw()
        self.update()

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._click(e.position().x())
        else:
            self._hover(self._at(e.position().x()))

    def leaveEvent(self, e):
        self._hover(None)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._click(e.position().x())

    def _click(self, x):
        if not self.enabled:
            return
        i = self._at(x)
        self.hov = i
        if i != self.v:
            self.v = i
            self.draw()
            self.update()
            if self.cmd:
                self.cmd(i)

    def set(self, v):
        self.v = min(max(int(v), 1), self.n + (1 if self.branch else 0))
        self.draw()
        self.update()

    def draw(self):
        self.delete("all")
        BGc, FGc, DIMc = BG(), FG(), DIM()
        W, y = max(60, self.width()), 14
        xs = self._xs()
        col = FGc if self.enabled else DIMc
        shown = self.v if (self.v <= self.n or self.branch_on) else self.n
        self.create_line(xs[0], y, xs[-1], y, fill=col, width=2)
        for i, x in enumerate(xs, 1):
            on = i == shown
            r = 9 if on else 7
            self.create_oval(x - r, y - r, x + r, y + r, outline=col, fill=col if on else BGc, width=2)
            if i == self.hov and not on:
                self.create_oval(x - 3, y - 3, x + 3, y + 3, outline=FGc, fill=FGc)
        right = W
        if self.branch:
            bx = self._bx()
            bc = col if self.branch_on else DIMc
            on = shown == self.n + 1
            self.create_line(xs[-1] + 11, y, bx - 12, y, fill=bc, width=2, dash=(3, 3))
            r = 10 if on else 8
            self.create_polygon(bx, y - r, bx + r, y, bx, y + r, bx - r, y, outline=bc, fill=bc if on else BGc, width=2)
            if self.hov == self.n + 1 and not on and self.branch_on:
                self.create_polygon(bx, y - 4, bx + 4, y, bx, y + 4, bx - 4, y, outline=FGc, fill=FGc)
            self.create_text(W, 43, text=tr(self.branch).upper(), anchor="se", fill=DIMc, font=FONT_S)
            right = xs[-1] + 9
        self.create_text(0, 43, text=tr(self.ends[0]).upper(), anchor="sw", fill=DIMc, font=FONT_S)
        self.create_text(right, 43, text=tr(self.ends[1]).upper(), anchor="se", fill=DIMc, font=FONT_S)


class PixProgress(_Base):
    """прогрес із квадратних блоків"""

    def __init__(self, parent=None, height=18):
        super().__init__(parent)
        self.f, self.h = 0.0, height
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, f):
        self.f = min(1.0, max(0.0, f))
        self.draw()
        self.update()

    def draw(self):
        self.delete("all")
        W = max(20, self.width())
        self.create_rectangle(0, 0, W - 1, self.h - 1, outline=FG())
        n = (W - 8) // 8
        k = int(self.f * n + 1e-6)
        pad = (W - (n * 8 - 1)) // 2
        for i in range(k):
            self.create_rectangle(pad + i * 8, 4, pad + i * 8 + 6, self.h - 5, fill=FG(), outline=FG())


class PixScroll(_Base):
    """смужка прокрутки: доріжка з рамкою й квадратний повзунок із трьома рисками. set(first, last) - видима частина (як tk.Scrollbar);
    cmd(f) - f у 0..1; page_cmd(±1) - клік по доріжці повз повзунок; release_cmd() - повзунок відпустили"""
    MIN = 24

    def __init__(self, parent=None, width=12, cmd=None):
        super().__init__(parent)
        self.bw, self.cmd, self.page_cmd, self.release_cmd = width, cmd, None, None
        self.first, self.last, self._hot, self._grab = 0.0, 1.0, False, None
        self.setFixedWidth(width)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

    def set(self, first, last):
        first, last = min(max(first, 0.0), 1.0), min(max(last, 0.0), 1.0)
        if (first, last) != (self.first, self.last):
            self.first, self.last = first, last
            self.draw()
            self.update()

    def _geo(self):
        L = max(8, self.height() - 4)
        return L, min(L, max(self.MIN, (self.last - self.first) * L))

    def _y0y1(self):
        L, n = self._geo()
        span = 1.0 - (self.last - self.first)
        f = 0.0 if span <= 1e-9 else self.first / span
        y0 = 2 + f * (L - n)
        return y0, y0 + n

    def _set_hot(self, v):
        if v != self._hot:
            self._hot = v
            self.draw()
            self.update()

    def mouseMoveEvent(self, e):
        y = e.position().y()
        if e.buttons() & Qt.LeftButton:
            if self._grab is not None and self.cmd:
                L, n = self._geo()
                self.cmd(min(max((y - self._grab - 2) / max(1.0, L - n), 0.0), 1.0))
        else:
            a, b = self._y0y1()
            self._set_hot(a <= y <= b)

    def leaveEvent(self, e):
        self._set_hot(False)

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        y = e.position().y()
        y0, y1 = self._y0y1()
        if y0 <= y <= y1:
            self._grab = y - y0
            self.draw()
            self.update()
        elif self.page_cmd:
            self.page_cmd(-1 if y < y0 else 1)

    def mouseReleaseEvent(self, e):
        grabbed, self._grab = self._grab is not None, None
        self.draw()
        self.update()
        if grabbed and self.release_cmd:
            self.release_cmd()

    def draw(self):
        self.delete("all")
        BGc, FGc = BG(), FG()
        W, H = self.bw, max(10, self.height())
        self.create_rectangle(0, 0, W - 1, H - 1, outline=FGc)
        y0, y1 = self._y0y1()
        lit = self._hot or self._grab is not None
        self.create_rectangle(2, y0, W - 3, y1, fill=FGc if lit else BGc, outline=FGc)
        if y1 - y0 >= 14:
            m = (y0 + y1) / 2
            for dy in (-3, 0, 3):
                self.create_line(4, m + dy, W - 5, m + dy, fill=BGc if lit else FGc)
