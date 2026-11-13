"""Графічна програма в стилі медіапрогравача, оформлена чорно-білим піксель-артом.   python -m benvid

Лише стандартний tkinter (+ numpy/opencv). Відтворення йде в реальному часі за годинником (benvid.player):
fps відео зберігається, кадри, які не встигли, пропускаються.
"""
from __future__ import annotations

import base64
import os
import queue
import contextlib
import re
import sys
import textwrap
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import cv2
import numpy as np

from .core import BRANCH, GLYPH_INFO, MAIN_LEVELS, METHOD_INFO, METHODS, Params
from . import eggs, guide, i18n, meta, palette, popup, view
from . import log as _log
from . import gpu
from .engine import load_settings, save_settings
from .i18n import tn, tr
from .media import GIF_EXT, GIF_FORMATS, IMAGE_EXT, export_gif_source, export_still, save_still, save_text, still_formats
from .player import Player
from .progress import Clock, fmt_t
from .video import (FORMATS, IKR_FORMAT, QUALITY, SIZES, VIDEO_EXT, Cancelled, default_workers, detect_hw_encoder, export_video, fps_options, fps_ratio, out_size, parse_fps_limit,
                    parse_size)

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
BG, PANEL, EDGE, FG, DIM = "#000000", "#141414", "#ffffff", "#ffffff", "#9a9a9a"
FONT = ("Consolas", 10, "bold")
FONT_S = ("Consolas", 9)
FONT_B = ("Consolas", 12, "bold")
AUTO = "Авто"
SIZE_KEYS = [AUTO] + list(SIZES)
SB_W = 12                  # ширина смужки прокрутки сторінки й правої панелі
MIN_MAIN = 520             # нижче цієї висоти вікна застосунок уже не стискається, а сторінка прокручується

# --- сітка інтерфейсу. Усі відступи й розміри кратні 4, щоб краї елементів у різних місцях лягали на одні й ті самі лінії:
M = 8                      # поле вікна, проміжок між блоками, відступ усередині рамок
G = 4                      # проміжок між сусідніми кнопками однієї групи (ґратка розмірів, пара списків)
GG = 16                    # проміжок між групами кнопок в одному ряду
H_S, H_M, H_L = 24, 30, 38     # висоти керування: дрібні (ґратка, поля, списки) / звичайні (шапка, палітра, «Зберегти») / транспорт
BAR_H = 20                 # висота інвертованої смужки із заголовком секції
ICON_GAP = 8               # проміжок між значком і підписом усередині кнопки
HEAD_W = 144               # ширина кнопок «Відкрити / Зберегти / Стоп» (по найдовшому підпису: «Зберегти фото»)
EDGE_W = 22                # вузька кнопка на правому краю в F11: висуває праву панель
W_SQ = 38                  # ширина малих кнопок з одним знаком: «?», UA, EN, стрілки палітри
TOG_W = 56                 # ширина перемикачів «Звук» і «Весь» в експорті
Q_W = 88                   # ширина списку якості кодування
FOOT_GAP = 16              # проміжок між блоками футера
SIDE_W = 310               # права панель: 2 рамки + 2 відступи по M + 4 кнопки по 70 і 3 проміжки по G = 2 + 16 + 292


def _ink(name):
    """межі видимих пікселів картинки (x0, y0, x1, y1; кінець не включно) - кнопки центрують сам малюнок, а не прозорі поля
    навколо нього (в одних значків вони зверху, в інших знизу)"""
    if name not in _INK:
        try:
            from PIL import Image
            im = Image.open(os.path.join(ASSETS, name + ".png")).convert("RGBA")
            _INK[name] = im.getchannel("A").getbbox() or (0, 0) + im.size
        except Exception:  # noqa: BLE001
            _INK[name] = None
    return _INK[name]


_INK: dict = {}
_FONTS: dict = {}


def _tw(text, font=FONT):
    """ширина рядка в пікселях у шрифті font (для центрування й підгонки; потребує готового вікна Tk)"""
    f = _FONTS.get(font)
    if f is None:
        f = _FONTS[font] = tkfont.Font(font=font)
    return f.measure(text)


_IMG_CACHE: dict = {}
_KEYS: dict = {}          # клас віджета -> набір його параметрів (див. App._apply_theme)


def _select_value(w):
    """у полі «Кадри: 25 к/с» виділити лише те, що після двокрапки"""
    try:
        t = w.get()
        i = t.find(":")
        a = i + 1 if i >= 0 else 0
        while a < len(t) and t[a] == " ":
            a += 1
        w.selection_range(a, "end")
        w.icursor("end")
    except tk.TclError:
        pass


def _clear_sel(w):
    try:
        w.selection_clear()
    except tk.TclError:
        pass


def _select_all(w):
    try:
        w.selection_range(0, "end")
    except tk.TclError:
        pass


def _img(name, rot=0.0):
    """піксель-арт з теки assets, перефарбований під поточну тему (чорне -> тло, біле -> колір тексту); rot - поворот у градусах
    (оберт котушки-пасхалки). Об'єкт отримує атрибут ink - межі видимого малюнка (див. _ink)"""
    key = (name, BG, FG)
    root = getattr(tk, "_default_root", None)
    if _IMG_CACHE.get("root") is not root:               # малюнки належать вікну: нове вікно (тести) - новий кеш
        _IMG_CACHE.clear()
        _IMG_CACHE["root"] = root
    if not rot and key in _IMG_CACHE:                    # перемикання теми не перекодовує малюнки заново (під навантаженням це давало «запізнілі» кнопки)
        return _IMG_CACHE[key]
    path = os.path.join(ASSETS, name + ".png")
    ph = None
    try:
        if rot or (BG, FG) != ("#000000", "#ffffff"):
            try:
                import io
                from PIL import Image
                im = Image.open(path).convert("RGBA")
                if rot:
                    im = im.rotate(rot, resample=Image.NEAREST)
                a = np.array(im, np.float32)
                lo = np.array([int(BG[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
                hi = np.array([int(FG[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
                t = a[..., :3].mean(-1, keepdims=True) / 255.0
                a[..., :3] = lo + (hi - lo) * t
                buf = io.BytesIO()
                Image.fromarray(a.round().astype(np.uint8), "RGBA").save(buf, "PNG")
                ph = tk.PhotoImage(data=base64.b64encode(buf.getvalue()))
            except Exception:  # noqa: BLE001
                ph = None
        if ph is None:
            ph = tk.PhotoImage(file=path)
        ph.ink = None if rot else _ink(name)
        if not rot:
            _IMG_CACHE[key] = ph
        return ph
    except Exception:  # noqa: BLE001
        return None


def _zoom2(im):
    """піксель-арт удвічі без згладжування: «герой» футера (котушка й логотип)"""
    try:
        return im.zoom(2) if im is not None else None
    except tk.TclError:
        return None


class PixButton(tk.Canvas):
    """плоска пікселізована кнопка: чорна з білою рамкою, при натисканні інвертується.
    width/height - бажаний розмір; якщо менеджер розміщення розтягне кнопку (fill / sticky), малюнок підлаштовується під
    фактичний розмір. Значок і підпис центруються разом, як одне ціле (проміжок між ними завжди однаковий).
    join="l" - кнопка «склеєна» з лівим сусідом: власну ліву лінію рамки не малює (між двома лишається одна лінія)"""
    ICON_W = 20                                      # ширина малованих значків (play, cookie, up…)
    FLASH = True                                     # при натисканні кнопка інвертується (висока вузька кнопка F11 цього не робить: на всю висоту це було б різке блимання)

    def __init__(self, parent, text="", image=None, command=None, width=None, height=H_M, toggle=False, icon=None, join=None,
                 **kw):
        w = width or (len(text) * 9 + 22 if text else 38)
        super().__init__(parent, width=w, height=height, bg=BG, highlightthickness=0, bd=0, cursor="hand2", **kw)
        self.text, self.cmd, self.on, self.enabled, self.toggle = text, command, False, True, toggle
        self.im, self._bw, self._bh, self.hover, self.icon, self.join = image, w, height, False, icon, join
        self.bites, self.crumbs = 0, []                  # пасхалка «печиво»: скільки відкусів і крихти, що падають
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Enter>", lambda e: self._hov(True))
        self.bind("<Leave>", lambda e: self._hov(False))
        self.bind("<Configure>", self._resize)
        self._pressed = False
        self.draw()

    def _resize(self, e):
        if (e.width, e.height) != (self._bw, self._bh):
            self._bw, self._bh = e.width, e.height
            self.draw()

    def _hov(self, v):
        self.hover = v
        self.draw()

    def set_icon(self, name):
        self.icon = name
        self.draw()

    def set_on(self, v):
        self.on = v
        self.draw()

    def set_enabled(self, v):
        self.enabled = v
        self.config(cursor="hand2" if v else "arrow")
        self.draw()

    def draw(self):
        self.delete("all")
        inv = self.on or (self._pressed and self.FLASH)
        bg, fg = (FG, BG) if inv else (BG, FG)
        if not self.enabled:
            fg = DIM
        w, h = self._bw, self._bh
        x0 = -1 if self.join == "l" else 0
        self.create_rectangle(x0, 0, w - 1, h - 1, fill=bg, outline=FG if self.enabled else DIM)
        if self.hover and self.enabled and not inv:
            self.create_rectangle(2, 2, w - 3, h - 3, outline=FG, dash=(1, 2))
        im, cy = self.im, h // 2
        ink = (getattr(im, "ink", None) or (0, 0, im.width(), im.height())) if im is not None and not self.icon else None
        iw = self.ICON_W if self.icon else (ink[2] - ink[0] if ink else 0)
        tw = _tw(self.text) if self.text else 0
        x = (w - (iw + (ICON_GAP if iw and tw else 0) + tw)) // 2            # значок + підпис - одним блоком посередині
        if self.icon:
            self._icon(self.icon, x + iw // 2, cy, fg, bg)
        elif ink:
            self.create_image(x - ink[0], cy - (ink[3] - ink[1]) // 2 - ink[1], image=im, anchor="nw")
        if self.text:
            self.create_text(x + iw + (ICON_GAP if iw else 0), cy, text=self.text, fill=fg, font=FONT, anchor="w")

    def _icon(self, n, cx, cy, c, bg=None):
        bg = bg or BG
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
        elif n == "menu":                                    # «бургер»: три смужки
            r(-8, -7, 8, -5)
            r(-8, -1, 8, 1)
            r(-8, 5, 8, 7)
        elif n == "cookie":                                  # печиво з відкушеним краєм і шоколадними крапками
            if self.bites < eggs.EATEN:                      # (пасхалка: кліки відкушують його далі - App._cookie_click)
                self.create_oval(cx - 9, cy - 9, cx + 9, cy + 9, fill=c, outline=c)
                self.create_oval(cx + 3, cy - 13, cx + 13, cy - 3, fill=bg, outline=bg)
                for dx, dy in ((-5, -3), (1, 0), (-2, 5), (5, 5), (-6, 3), (-1, -6)):
                    self.create_rectangle(cx + dx, cy + dy, cx + dx + 2, cy + dy + 2, fill=bg, outline=bg)
                for dx, dy, rr in eggs.BITES[:self.bites]:
                    self.create_oval(cx + dx - rr, cy + dy - rr, cx + dx + rr, cy + dy + rr, fill=bg, outline=bg)
            for x, y, _, _, sz in self.crumbs:
                self.create_rectangle(cx + x, cy + y, cx + x + sz, cy + y + sz, fill=c, outline=c)
        elif n == "swap":                                   # дві кругові стрілки: поміняти туш і папір місцями
            import math
            R, step, cnt = 8.0, 16.0, 7
            for a0 in (200.0, 20.0):                        # дві дуги по 112°, друга - перша, повернута на 180°
                pts = []
                for k in range(cnt + 1):
                    a = math.radians(a0 + k * step)
                    pts += [cx + R * math.cos(a), cy - R * math.sin(a)]
                self.create_line(*pts, fill=c, width=2)
                a = math.radians(a0 + cnt * step)           # наконечник у кінці дуги, за напрямом руху
                ex, ey = cx + R * math.cos(a), cy - R * math.sin(a)
                tx, ty = -math.sin(a), -math.cos(a)
                nx, ny = -ty, tx
                self.create_polygon(ex + 4.5 * tx, ey + 4.5 * ty, ex - 0.5 * tx + 3.5 * nx, ey - 0.5 * ty + 3.5 * ny,
                                    ex - 0.5 * tx - 3.5 * nx, ey - 0.5 * ty - 3.5 * ny, fill=c, outline=c)

    def _press(self, e):
        if self.enabled:
            self._pressed = True
            self.draw()

    def _release(self, e):
        was = self._pressed
        self._pressed = False
        if self.toggle and was and self.enabled:
            self.on = not self.on
        self.draw()
        if was and self.enabled and 0 <= e.x <= self._bw and 0 <= e.y <= self._bh and self.cmd:
            self.cmd()


FLAG_H = 13                          # висота смужки над доріжкою, де живуть прапорці фрагмента
THUMB = 5                            # половина ширини ручки = поле доріжки: ручка в крайніх положеннях лягає врівень з краєм віджета
# Контур прямокутника в Tk зайнятий і по x0, і по x1 (включно), тож ручка 2·THUMB+1 пікселів завширшки; щоб праве ребро
# не вилазило за полотно (там воно «з'їдалось» на максимумі), хід центра ручки - від THUMB до ширина-1-THUMB


class PixSlider(tk.Canvas):
    """повзунок/смуга перемотування: піксельна доріжка з квадратною ручкою.
    Якщо задано marks (початок і кінець фрагмента для експорту), над доріжкою є окрема смужка з двома прапорцями-дужками
    «[ ]»: вони не накладаються на ручку в жодному положенні (ні на початку, ні в кінці), коли обрано все відео - ледь
    помітні, а коли фрагмент обмежено - яскраві, з'єднані планкою, а ділянки доріжки поза фрагментом «перфоровані»."""
    def __init__(self, parent, from_=0.0, to=1.0, command=None, height=22, on_release=None, steps=None, **kw):
        super().__init__(parent, height=height, bg=BG, highlightthickness=0, bd=0, cursor="hand2", **kw)
        self.lo, self.hi, self.cmd, self.on_release, self.steps = from_, to, command, on_release, steps
        self.v, self.drag, self.h, self.ranges = from_, False, height, []
        self.marks, self.mark_cb, self._dm, self._hot = None, None, None, None      # прапорці початку/кінця фрагмента
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<ButtonPress-1>", self._mv)
        self.bind("<B1-Motion>", self._mv)
        self.bind("<ButtonRelease-1>", self._up)
        self.bind("<Motion>", self._hover)
        self.bind("<Leave>", lambda e: self._set_hot(None))

    def set(self, v, redraw=True):
        self.v = min(max(v, self.lo), self.hi)
        if redraw and not self.drag:
            self.draw()

    def get(self):
        return self.v

    def set_ranges(self, rs):
        if rs != self.ranges:
            self.ranges = rs
            self.draw()

    def _mark_x(self, v):
        W = max(20, self.winfo_width())
        return THUMB + (v - self.lo) / max(1e-9, self.hi - self.lo) * (W - 1 - 2 * THUMB)

    def _ty(self):
        """вертикальний центр доріжки: з прапорцями - під їхньою смужкою, без них - посередині"""
        return FLAG_H + (self.h - FLAG_H) // 2 if self.marks is not None else self.h // 2

    def _flag_at(self, e):
        """номер прапорця під курсором (лише в смужці прапорців, не ближче ніж на 12 px по горизонталі) або None"""
        if not self.marks or self.hi <= self.lo or e.y > FLAG_H + 2:
            return None
        xa, xb = self._mark_x(self.marks[0]), self._mark_x(self.marks[1])
        da, db = abs(e.x - xa), abs(e.x - xb)
        if min(da, db) > 12:
            return None
        if da == db:
            return 0 if e.x <= (xa + xb) / 2 else 1
        return 0 if da < db else 1

    def _set_hot(self, i):
        if i != self._hot:
            self._hot = i
            self.config(cursor="sb_h_double_arrow" if i is not None else "hand2")
            self.draw()

    def _hover(self, e):
        if self._dm is None and not self.drag:
            self._set_hot(self._flag_at(e))

    def _mv(self, e):
        w = max(1, self.winfo_width() - 1 - 2 * THUMB)
        f = min(max((e.x - THUMB) / w, 0.0), 1.0)
        v = self.lo + f * (self.hi - self.lo)
        if self.marks and self.hi > self.lo:
            if self._dm is None and not self.drag:                      # натиснули в смужці прапорців - тягнемо найближчий
                self._dm = self._flag_at(e)
                if self._dm is not None:
                    self._hot = self._dm
            if self._dm is not None:
                v = round(v)
                lo, hi = (self.lo, self.marks[1] - 1) if self._dm == 0 else (self.marks[0] + 1, self.hi)
                self.marks[self._dm] = min(max(v, lo), max(lo, hi))
                self.draw()
                if self.mark_cb:
                    self.mark_cb(self._dm, self.marks[self._dm])
                return
        if self.steps:
            v = round(v)
        self.drag, self.v = True, v
        self.draw()
        if self.cmd:
            self.cmd(self.v)

    def _up(self, e):
        if self._dm is not None:
            self._dm = None
            self._hot = None
            self.config(cursor="hand2")
            self.draw()
            return
        self.drag = False
        self.draw()
        if self.on_release:
            self.on_release(self.v)

    def _flag(self, x, side, c):
        """дужка «[» (side>0) або «]» (side<0): ніжка 3 px, два серіфи, тонка лінія-«виска» до доріжки"""
        top, bot = 1, FLAG_H - 1
        x0, x1 = (x - 1, x + 7) if side > 0 else (x - 7, x + 1)
        self.create_rectangle(x - 1, top, x + 1, bot, fill=c, outline=c)
        self.create_rectangle(x0, top, x1, top + 2, fill=c, outline=c)
        self.create_rectangle(x0, bot - 2, x1, bot, fill=c, outline=c)

    def draw(self):
        self.delete("all")
        W = max(20, self.winfo_width())
        y = self._ty()
        f = 0 if self.hi == self.lo else (self.v - self.lo) / (self.hi - self.lo)
        L = W - 1 - 2 * THUMB
        x = THUMB + f * L
        self.create_rectangle(THUMB, y - 3, W - 1 - THUMB, y + 3, outline=FG, fill=BG)
        if self.steps and self.hi > self.lo:                          # поділки щаблів
            for i in range(int(self.lo), int(self.hi) + 1):
                tx = THUMB + (i - self.lo) / (self.hi - self.lo) * L
                self.create_line(tx, y + 5, tx, y + 10, fill=FG)
        if self.hi > self.lo:
            for a, b in self.ranges:
                ax = THUMB + (a - self.lo) / (self.hi - self.lo) * L
                bx = THUMB + (b - self.lo) / (self.hi - self.lo) * L
                self.create_rectangle(ax, y - 2, max(bx, ax + 1), y + 2, outline=DIM, fill=DIM)
        if x > THUMB + 1:
            self.create_rectangle(THUMB + 1, y - 2, x, y + 2, outline=FG, fill=FG)
        if self.marks and self.hi > self.lo:                          # фрагмент: прапорці-дужки над доріжкою
            ax, bx = int(round(self._mark_x(self.marks[0]))), int(round(self._mark_x(self.marks[1])))
            lim = self.marks[0] > self.lo or self.marks[1] < self.hi
            if lim:
                for u, v in ((THUMB, ax), (bx, W - 1 - THUMB)):       # поза фрагментом доріжка «перфорована»
                    if v - u > 1:
                        for j, yy in enumerate((y - 3, y - 2, y - 1, y, y + 1, y + 2, y + 3)):
                            self.create_line(u + 1, yy, v, yy, fill=BG, dash=(1, 1), dashoffset=j % 2)
                self.create_rectangle(ax, 1, bx, 3, fill=FG, outline=FG)                    # планка між прапорцями
            for i, (fx, side) in enumerate(((ax, 1), (bx, -1))):
                c = FG if (lim or self._hot == i or self._dm == i) else DIM
                self._flag(fx, side, c)
                self.create_line(fx, FLAG_H - 1, fx, y - 4, fill=c)       # «виска» до доріжки: де саме межа
        self.create_rectangle(x - THUMB, y - 8, x + THUMB, y + 8, outline=FG, fill=FG if self.drag else BG)


class QualityBar(tk.Canvas):
    """баланс «швидкість ↔ якість»: п'ять гуртків на одній лінії, зафарбований лише обраний. branch - підпис додаткової позиції (n+1):
    окрема гілка, не продовження шкали, - ромб на пунктирному відгалуженні від останнього гуртка"""
    def __init__(self, parent, n=5, value=4, command=None, ends=("швидко", "якісно"), branch=None, **kw):
        super().__init__(parent, height=44, bg=BG, highlightthickness=0, bd=0, cursor="hand2", **kw)
        self.n, self.v, self.cmd, self.hov, self.ends = n, value, command, None, ends
        self.branch = branch
        self.branch_on = True            # гілка доступна (у «Символах» її нема: там лише п'ять рівнів)
        self.enabled = True
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Motion>", lambda e: self._hover(self._at(e.x)))
        self.bind("<Leave>", lambda e: self._hover(None))
        self.bind("<ButtonPress-1>", self._click)
        self.bind("<B1-Motion>", self._click)

    PAD = 10                         # поле до центра крайнього гуртка: радіус обраного (9) + його лінія (1) - гурток лягає врівень з краєм
    BR = 64                          # місце під відгалуження справа: пунктир і ромб

    def _xs(self):
        W = max(60, self.winfo_width())
        span = W - 2 * self.PAD - (self.BR if self.branch else 0)
        return [self.PAD + i * span / (self.n - 1) for i in range(self.n)]

    def _bx(self):
        return max(60, self.winfo_width()) - self.PAD - 2

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

    def set_enabled(self, v):
        self.enabled = bool(v)
        self.config(cursor="hand2" if v else "arrow")
        self.hov = None
        self.draw()

    def set_branch(self, on):
        """гілка доступна чи ні (недоступна - тьмяна й не клікається)"""
        self.branch_on = bool(on)
        self.hov = None
        self.draw()

    def _click(self, e):
        if not self.enabled:
            return
        i = self._at(e.x)
        self.hov = i
        if i != self.v:
            self.v = i
            self.draw()
            if self.cmd:
                self.cmd(i)

    def set(self, v):
        self.v = min(max(int(v), 1), self.n + (1 if self.branch else 0))
        self.draw()

    def draw(self):
        self.delete("all")
        W, y = max(60, self.winfo_width()), 14
        xs = self._xs()
        col = FG if self.enabled else DIM
        shown = self.v if (self.v <= self.n or self.branch_on) else self.n         # гілка недоступна: показуємо п'ятий (так її й рахує режим)
        self.create_line(xs[0], y, xs[-1], y, fill=col, width=2)
        for i, x in enumerate(xs, 1):
            on = i == shown
            r = 9 if on else 7
            self.create_oval(x - r, y - r, x + r, y + r, outline=col, fill=col if on else BG, width=2)
            if i == self.hov and not on:
                self.create_oval(x - 3, y - 3, x + 3, y + 3, outline=FG, fill=FG)
        right = W
        if self.branch:
            bx = self._bx()
            bc = col if self.branch_on else DIM
            on = shown == self.n + 1
            self.create_line(xs[-1] + 11, y, bx - 12, y, fill=bc, width=2, dash=(3, 3))          # відгалуження від шкали, а не її продовження
            r = 10 if on else 8
            self.create_polygon(bx, y - r, bx + r, y, bx, y + r, bx - r, y, outline=bc, fill=bc if on else BG, width=2)
            if self.hov == self.n + 1 and not on and self.branch_on:
                self.create_polygon(bx, y - 4, bx + 4, y, bx, y + 4, bx - 4, y, outline=FG, fill=FG)
            self.create_text(W, 43, text=tr(self.branch).upper(), anchor="se", fill=DIM, font=FONT_S)
            right = xs[-1] + 9
        self.create_text(0, 43, text=tr(self.ends[0]).upper(), anchor="sw", fill=DIM, font=FONT_S)
        self.create_text(right, 43, text=tr(self.ends[1]).upper(), anchor="se", fill=DIM, font=FONT_S)


class PixProgress(tk.Canvas):
    """прогрес із квадратних блоків"""
    def __init__(self, parent, height=18, **kw):
        super().__init__(parent, height=height, bg=BG, highlightthickness=0, bd=0, **kw)
        self.f, self.h = 0.0, height
        self.bind("<Configure>", lambda e: self.draw())

    def set(self, f):
        self.f = min(1.0, max(0.0, f))
        self.draw()

    def draw(self):
        self.delete("all")
        W = max(20, self.winfo_width())
        self.create_rectangle(0, 0, W - 1, self.h - 1, outline=FG)
        n = (W - 8) // 8
        k = int(self.f * n + 1e-6)
        pad = (W - (n * 8 - 1)) // 2                  # ряд блоків по центру: однакові поля зліва й справа
        for i in range(k):
            self.create_rectangle(pad + i * 8, 4, pad + i * 8 + 6, self.h - 5, fill=FG, outline=FG)


class PixScroll(tk.Canvas):
    """смужка прокрутки в піксельному стилі: доріжка з рамкою й квадратний повзунок із трьома рисками.
    set(first, last) - видима частина вмісту в частках (як у tk.Scrollbar); cmd(f) - f у 0..1: де має стати верх повзунка
    (0 - початок, 1 - кінець прокрутки); page_cmd(±1) - клік по доріжці повз повзунок (на сторінку вгору/вниз);
    release_cmd() - повзунок відпустили (сторінка з екранами після цього «прилипає» до найближчого)"""
    MIN = 24

    def __init__(self, parent, width=12, cmd=None, **kw):
        super().__init__(parent, width=width, bg=BG, highlightthickness=0, bd=0, cursor="arrow", **kw)
        self.bw, self.cmd, self.page_cmd, self.release_cmd = width, cmd, None, None
        self.first, self.last, self._hot, self._grab = 0.0, 1.0, False, None
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Motion>", lambda e: self._set_hot(self._y0y1()[0] <= e.y <= self._y0y1()[1]))
        self.bind("<Leave>", lambda e: self._set_hot(False))

    def set(self, first, last):
        first, last = min(max(first, 0.0), 1.0), min(max(last, 0.0), 1.0)
        if (first, last) != (self.first, self.last):
            self.first, self.last = first, last
            self.draw()

    def _geo(self):
        """-> (довжина доріжки, довжина повзунка)"""
        L = max(8, self.winfo_height() - 4)
        return L, min(L, max(self.MIN, (self.last - self.first) * L))

    def _y0y1(self):
        L, n = self._geo()
        span = 1.0 - (self.last - self.first)
        f = 0.0 if span <= 1e-9 else self.first / span                # положення в 0..1 по ходу прокрутки
        y0 = 2 + f * (L - n)
        return y0, y0 + n

    def _set_hot(self, v):
        if v != self._hot:
            self._hot = v
            self.draw()

    def _press(self, e):
        y0, y1 = self._y0y1()
        if y0 <= e.y <= y1:
            self._grab = e.y - y0
            self.draw()
        elif self.page_cmd:
            self.page_cmd(-1 if e.y < y0 else 1)

    def _move(self, e):
        if self._grab is None or not self.cmd:
            return
        L, n = self._geo()
        self.cmd(min(max((e.y - self._grab - 2) / max(1.0, L - n), 0.0), 1.0))

    def _release(self, e):
        grabbed, self._grab = self._grab is not None, None
        self.draw()
        if grabbed and self.release_cmd:
            self.release_cmd()

    def draw(self):
        self.delete("all")
        W, H = self.bw, max(10, self.winfo_height())
        self.create_rectangle(0, 0, W - 1, H - 1, outline=FG)
        y0, y1 = self._y0y1()
        lit = self._hot or self._grab is not None
        self.create_rectangle(2, y0, W - 3, y1, fill=FG if lit else BG, outline=FG)
        if y1 - y0 >= 14:
            m = (y0 + y1) / 2
            for dy in (-3, 0, 3):
                self.create_line(4, m + dy, W - 5, m + dy, fill=BG if lit else FG)


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


def _above_owner(win, widget):
    """мала спливаюча хмаринка (підказка, фраза котушки) має бути над вікном програми, але НЕ над чужими вікнами. Раніше для цього
    ставився -topmost: такий Toplevel лежить над усім на екрані, доки живий, і може зависнути над іншою програмою (Alt+Tab, поки
    підказка висить). «Власник» (transient) дає потрібне: вікно завжди над своїм власником і ховається разом із ним"""
    try:
        win.transient(widget.winfo_toplevel())
    except tk.TclError:
        pass


class Tip:
    """підказка при наведенні: з'являється за ~0,6 с, зникає, щойно курсор пішов чи натиснули кнопку миші.
    text - рядок або функція без аргументів (тоді текст береться в момент показу: завжди актуальною мовою й станом).
    Одночасно видно не більше однієї підказки; сама зникає за HIDE мс (на випадок, коли подію «курсор пішов» не доставили:
    вікно втратило фокус, віджет сховали). Вікно будується прихованим і з'являється вже на місці - без блимання в куті екрана;
    місце рахується за монітором під курсором (benvid.popup)"""
    DELAY = 600
    HIDE = 12000
    _cur: "Tip | None" = None

    def __init__(self, widget, text):
        self.w, self.text, self._job, self._win, self._gone = widget, text, None, None, None
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")
        widget.bind("<ButtonPress>", self._leave, add="+")
        widget.bind("<Destroy>", self._leave, add="+")

    @classmethod
    def hide_all(cls):
        """прибрати підказку, що зараз висить (зміна вікна, фокус, натискання клавіші)"""
        t = cls._cur
        if t is not None:
            t._leave()

    def _enter(self, e=None):
        self._leave()
        self._job = self.w.after(self.DELAY, self._show)

    def _leave(self, e=None):
        for name in ("_job", "_gone"):
            j = getattr(self, name)
            if j:
                try:
                    self.w.after_cancel(j)
                except tk.TclError:
                    pass
                setattr(self, name, None)
        if self._win is not None:
            try:
                self._win.destroy()
            except tk.TclError:
                pass
            self._win = None
        if Tip._cur is self:
            Tip._cur = None

    def _show(self):
        self._job = None
        t = self.text() if callable(self.text) else self.text
        if not t:
            return
        Tip.hide_all()                                   # стара підказка (інший віджет) не лишається поруч із новою
        try:
            x, y = self.w.winfo_pointerxy()
            win = tk.Toplevel(self.w)
            win.withdraw()                               # спершу будуємо невидимим: на екрані воно з'являється одразу на своєму місці
            win.wm_overrideredirect(True)
            _above_owner(win, self.w)
            tk.Label(win, text=t, bg=BG, fg=FG, font=FONT_S, justify="left", anchor="w", padx=M, pady=G, bd=0,
                     highlightthickness=1, highlightbackground=FG, wraplength=340).pack()
            win.update_idletasks()
            w, h = win.winfo_reqwidth(), win.winfo_reqheight()
            px, py = popup.near_pointer(x, y, w, h, popup.work_area(self.w, x, y))
            win.wm_geometry(f"+{px}+{py}")
            win.deiconify()
            self._win = win
            Tip._cur = self
            self._gone = self.w.after(self.HIDE, self._leave)
        except tk.TclError:
            self._win = None


class App(tk.Tk):
    TITLE = "InkReel"

    def __init__(self, open_path=None):
        if os.name == "nt":                          # власна група на панелі завдань (інакше - значок Python)
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("InkReel.App")
            except Exception:  # noqa: BLE001
                pass
        try:
            from . import procs
            _log.quiet_console()                       # консоль без «mmco: unref short failure»
            procs.crisp_text()                         # чіткий текст на моніторах із масштабом 125-200% (до створення вікна)
        except Exception:  # noqa: BLE001
            pass
        super().__init__()
        _FONTS.clear()                               # шрифти живуть у конкретному Tk: від попереднього вікна (тести) їх не беремо
        try:
            from . import procs
            procs.fine_timer()
            import gc
            gc.set_threshold(50000, 20, 20)             # рідші паузи збирача сміття під час відтворення
        except Exception:  # noqa: BLE001
            pass
        self.title(self.TITLE)
        self._set_icon()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()      # маленькі екрани/ноутбуки: вікно не більше за екран
        self.geometry(f"{min(1280, sw - 40)}x{min(860, sh - 90)}")
        self.minsize(min(1000, sw - 40), min(640, sh - 90))
        self._loc = []                     # локалізовані підписи віджетів: (віджет, параметр, ключ, ВЕЛИКІ, підстановки)
        self._strips = []
        self._last_out = None
        self._title_name = ""
        self._last_lay_t = 0.0
        self._lay_req = None
        self.q = queue.Queue()
        self.cancel = threading.Event()
        self.player = Player()
        probe = threading.Timer(3.0, gpu.start_probe_background)      # перевірка відеокарти - у фоні, окремим процесом
        probe.daemon = True                                          # не-фоновий потік, що завис у драйвері, тримав би процес живим після закриття вікна
        probe.start()
        self.params = Params()
        self.exporting = False
        self.pause_ev = threading.Event()  # пауза рендеру (експорту відео): кнопка відтворення й Пробіл
        self._pausable = False
        self._clock = None
        self._exp_f, self._exp_label, self._exp_total = 0.0, None, 0
        self._hud, self._hud_sig, self._hud_t = None, None, 0.0
        self.ready = False
        self._photo = None
        self._zoom = None                  # None - вписати у вікно; число - масштаб (цілий >= 1: чіткі пікселі; < 1: зменшення)
        self._vc = (0.5, 0.5)              # центр вікна перегляду в частках розміру зображення (має сенс лише при масштабуванні)
        self._wheel_acc = 0
        self._drag = None
        self._last_disp = 0.0
        self._last_info_t = 0.0
        self._last_rng_t = 0.0
        self._last_mask = None
        self._last_bgr = None
        self._egg_clicks = {"logo": eggs.Clicks(5, 1.4), "swatch": eggs.Clicks(7, 1.6),
                            "cookie": eggs.Clicks(eggs.COOKIE_CLICKS, 1.6)}
        self._defocus, self._egg_t0, self._egg_job = 0.0, None, None        # розфокусування під філософію
        self._tip_n, self._empty_shown = -1, False                          # «Порада» (guide.TIPS): номер і чи вже показана на порожньому екрані
        self._spin_job, self._spin_frames, self._spin_key = None, None, None
        self._bub, self._bub_job, self._last_quote = None, None, -1
        self._party_job, self._party_orig, self._cookie_job = None, "", None
        self._save_job = None
        self._pal_job, self._pal_t = None, 0.0                             # відкладене перемальовування кадра після швидкої серії кроків палітри
        self._photo_key, self._paint_sig = None, None                      # режим/розмір кадра на полотні й підпис його розкладки (швидкий шлях _paint)
        self._seek_job, self._seek_t, self._seek_val = None, 0.0, 0     # обмеження частоти seek при перетягуванні повзунка перемотування
        self._auto_size = None
        self._auto_tok = 0
        self._last_dir = None
        self._fs = False                                 # «кіно»: лише картинка (F, подвійний клік)
        self._wfs = False                                # F11: те саме вікно програми, розтягнуте на весь екран (з усім інтерфейсом)
        self._out_dir = None
        self._seek_hold = False
        self.configure(bg="#000000")
        self.ui = load_settings().get("ui", {})
        self._side_open = False                          # лише для F11: права панель висунута кнопкою на правому краю (у звичайному вікні панель завжди на місці)
        il = i18n.install_lang()                                        # нова установка з іншою мовою інсталятора - застосувати її один раз
        if il and self.ui.get("install_lang") != il:
            self.ui["lang"], self.ui["install_lang"] = il, il
        i18n.set_lang(self.ui.get("lang") or i18n.detect())            # перший запуск - мова системи, далі - вибір користувача
        pn, pinv = palette.migrate(self.ui.get("palette"), self.ui.get("pal_inv"))      # старі назви/без прапорця «навпаки»
        self.ui["palette"], self.ui["pal_inv"] = pn, pinv
        self._set_theme_globals(palette.key(pn, pinv))
        self.imgs = {n: _img(n + "_w") for n in ("play", "pause", "stop", "prev", "next", "open", "save", "film", "gear", "x",
                                          "loop", "sound")}
        
        self.logo, self.mascot = _img("logo_w"), _img("mascot_w")
        self.logo2, self.mascot2 = _zoom2(self.logo), _zoom2(self.mascot)
        self.dither, self.film = _img("dither_light"), _img("film_tile")
        self.configure(bg=BG)
        self._build()
        self._pm, self._pm_last = None, None                 # лічильник плавності показу (див. _meter)
        self._npaint, self._fr_idx = 0, 0
        self._pacc = dict(img=0.0, items=0.0, n=0, view=None)
        self._pgen, self._cx, self._cfut, self._cpend, self._cinfo, self._pc = 0, None, None, False, (0, 0, 0, 0.0), 0.0     # фонове зменшення кадра (див. _frame_redraw)
        self.after(10, self._tick)
        self.after(100, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<space>", lambda e: None if self._editing() or self._on_footer() else self._toggle())             # у полі вводу (розмір, кадри) пробіл і стрілки - для самого поля; на екрані футера плеєр не керується
        self.bind("<Left>", lambda e: None if self._editing() or self._on_footer() else self._step(-1))
        self.bind("<Right>", lambda e: None if self._editing() or self._on_footer() else self._step(1))
        self.bind("<Shift-Left>", lambda e: None if self._editing() or self._on_footer() else self._jump(-10))
        self.bind("<Shift-Right>", lambda e: None if self._editing() or self._on_footer() else self._jump(10))
        for cls in ("Entry", "TEntry", "TCombobox", "Text"):                        # Ctrl+C/V/X/A у кириличній розкладці: Tk чекає латинський keysym
            self.bind_class(cls, "<Control-Key>", self._ctrl_key, add="+")
        self.bind_all("<ButtonPress-1>", self._click_release_focus, add="+")       # клік по картинці/кнопках забирає фокус з поля вводу: інакше пробіл, стрілки й літери лишаються «в полі»
        self.bind("<Key>", self._on_letter)              # F, M, L, I - в будь-якій розкладці
        self.bind("<F11>", lambda e: self._window_fullscreen())
        self.bind("<Escape>", self._on_escape)
        self.bind("<Up>", lambda e: None if self._on_footer() else self._vol_step(5, key=True))
        self.bind("<Down>", lambda e: None if self._on_footer() else self._vol_step(-5, key=True))
        self.bind("<bracketleft>", lambda e: None if self._on_footer() else self._set_mark_here(0))
        self.bind("<bracketright>", lambda e: None if self._on_footer() else self._set_mark_here(1))
        self.bind("<Home>", lambda e: None if self._editing() or self._on_footer() else self._stop())
        self.bind("<F1>", lambda e: self._goto_guide())
        for ev in ("<Unmap>", "<Configure>"):                    # вікно згорнулось, зсунулось чи змінило розмір: підказка вже не при ділі
            self.bind(ev, lambda e: Tip.hide_all() if e.widget is self else None, add="+")      # (фокус не чіпаємо: самі підказки його не беруть, а завислу прибере таймер HIDE)
        self.bind("<Control-o>", lambda e: self._open() if not self.exporting else None)
        self.bind("<Control-s>", lambda e: self._export())
        self.bind("<Prior>", lambda e: None if self._typing() else self.pstack.step(-1))
        self.bind("<Next>", lambda e: None if self._typing() else self.pstack.step(1))
        for ks in ("<plus>", "<equal>", "<KP_Add>"):
            self.bind(ks, lambda e: None if self._typing() or self._on_footer() else self._zoom_step(1))
        for ks in ("<minus>", "<KP_Subtract>"):
            self.bind(ks, lambda e: None if self._typing() or self._on_footer() else self._zoom_step(-1))
        for ks in ("<Key-0>", "<Key-KP_0>"):                  # саме «Key-»: у Tk «<1>» без типу події - це натиснення лівої кнопки миші!
            self.bind(ks, lambda e: None if self._typing() or self._on_footer() else self._zoom_reset())
        for ks in ("<Key-1>", "<Key-KP_1>"):
            self.bind(ks, lambda e: None if self._typing() or self._on_footer() else self._zoom_100())
        self.screen.bind("<Double-Button-1>", lambda e: self._fullscreen())
        self._apply_lang(first=True)
        self.after(300, self.player.warm)
        self.report_callback_exception = lambda t, v, tb: _log.get().error("помилка інтерфейсу", exc_info=(t, v, tb))
        self._hook_drop()
        if open_path and os.path.isfile(open_path):
            self.after(200, lambda: self._open(open_path))

    # ------------------------------------------------------------ побудова
    def _tile_strip(self, parent, which, height):
        c = tk.Canvas(parent, height=height, bg=BG, highlightthickness=0, bd=0)
        def redraw(e=None):
            c.delete("all")
            tile = getattr(self, which)
            if tile is None:
                return
            tw, th = tile.width(), tile.height()
            cw = max(c.winfo_width(), 10)
            off = (cw % tw) // 2                  # неповні плитки - порівну з обох країв, а не лише з правого
            # одна картинка на всю смугу замість сотні плиток-елементів: Tk сам розмножує плитку в `copy -to` (на C), а полотно малює один елемент.
            # Раніше кожна зміна палітри чи розміру створювала ≈130 create_image на смугу, а їх шість
            try:
                strip = tk.PhotoImage(master=c, width=cw + off + tw, height=height)
                strip.tk.call(strip, "copy", tile, "-to", 0, 0, cw + off + tw, height)
                c._strip = strip                  # тримаємо посилання: інакше Tk зітре картинку
                c.create_image(-off, 0, image=strip, anchor="nw")
            except tk.TclError:                   # запасний шлях: плитка за плиткою
                for x in range(-off, cw + tw, tw):
                    for y in range(0, height, th):
                        c.create_image(x, y, image=tile, anchor="nw")
        c.bind("<Configure>", redraw)
        c.redraw = redraw
        self._strips.append(c)
        return c

    # --- локалізація й підказки: підписи віджетів перекладаються на місці
    def _L(self, w, key, attr="text", upper=False, **kw):
        """локалізований підпис віджета: запам'ятовуємо ключ, і перемикання мови оновлює підпис на місці"""
        self._loc.append((w, attr, key, upper, kw))
        self._loc_set(w, attr, key, upper, kw)
        return w

    @staticmethod
    def _loc_set(w, attr, key, upper, kw):
        s = tr(key, **kw)
        s = s.upper() if upper else s
        try:
            if isinstance(w, PixButton) and attr == "text":
                w.text = s
                w.draw()
            else:
                w.config(**{attr: s})
        except tk.TclError:
            pass

    def _tip(self, w, key):
        """підказка при наведенні: key - ключ перекладу або функція, що повертає готовий текст"""
        Tip(w, key if callable(key) else (lambda: tr(key)))
        return w

    def _build(self):
        self._strips = []
        # сторінка з двох екранів на все вікно: застосунок і під ним футер. До футера - прокруткою вниз (колесо, PageDown, «?», F1):
        # екрани не повзуть, а плавно перемикаються й фіксуються (PageStack); смужка прокрутки завжди видна - підказка, що внизу є ще екран
        self.page = tk.Frame(self, bg=BG)
        self.page.pack(fill="both", expand=True)
        self.main = tk.Frame(self.page, bg=BG)
        self.foot = tk.Frame(self.page, bg=BG)
        self.page_bar = PixScroll(self.page, SB_W)
        self.pstack = PageStack(self.page, self.page_bar,
                                [(self.main, lambda w, h: max(h, MIN_MAIN), None),
                                 (self.foot, lambda w, h: max(h, self.foot.winfo_reqheight()), self._foot_fit)])
        main = self.main
        # Вертикальний ритм головного екрана (усі проміжки - M): поле, шапка, смужка, центр, смужка, перемотування, транспорт, поле.
        # --- шапка: маскот, логотип, меню; вертикально все по центру ряду
        top = tk.Frame(main, bg=BG)
        top.pack(fill="x", padx=M, pady=(M, 0))
        if self.mascot:
            self._lbl_mascot = tk.Label(top, image=self.mascot, bg=BG, bd=0)
            self._lbl_mascot.pack(side="left", padx=(0, M))
            self._lbl_mascot.bind("<Button-1>", lambda e: self._reel_click())
        if self.logo:
            self._lbl_logo = tk.Label(top, image=self.logo, bg=BG, bd=0)
            self._lbl_logo.pack(side="left", padx=(0, GG))
            self._lbl_logo.bind("<Button-1>", lambda e: self._logo_click())
        else:
            tk.Label(top, text="InkReel", bg=BG, fg=FG, font=("Consolas", 22, "bold")).pack(side="left", padx=(0, GG))
        self.btn_open = self._L(PixButton(top, "", self.imgs["open"], self._open, width=HEAD_W), "Відкрити")
        self.btn_export = PixButton(top, "", self.imgs["save"], self._export, width=HEAD_W)
        self.btn_stop = self._L(PixButton(top, "", self.imgs["x"], self._stop_export, width=HEAD_W), "Стоп")
        for b in (self.btn_open, self.btn_export, self.btn_stop):          # три кнопки однакової ширини, проміжок M
            b.pack(side="left", padx=(0, M))
        self.btn_stop.set_enabled(False)
        # праворуч: [ім'я файлу]  [?]  [UA|EN]; UA і EN - пара «склеєних» кнопок, права кнопка врівень з правим краєм панелі
        self.btn_en = PixButton(top, "EN", command=lambda: self._set_lang(i18n.EN), width=W_SQ, height=H_M, join="l")
        self.btn_ua = PixButton(top, "UA", command=lambda: self._set_lang(i18n.UK), width=W_SQ, height=H_M)
        self.btn_help = PixButton(top, "?", command=self._goto_guide, width=W_SQ, height=H_M)
        self.btn_en.pack(side="right")
        self.btn_ua.pack(side="right")
        self.btn_help.pack(side="right", padx=(0, M))
        self.title_lbl = tk.Label(top, text="", bg=BG, fg=DIM, font=FONT, anchor="e")
        self.title_lbl.pack(side="right", fill="x", expand=True, padx=(M, M))
        self.title_lbl.bind("<Configure>", lambda e: self._fit_title())
        self._strip1 = self._tile_strip(main, "dither", 8)
        self._strip1.pack(fill="x", padx=M, pady=M)

        # --- центр: екран + права панель (верх і низ в обох однакові, проміжок між ними M)
        mid = tk.Frame(main, bg=BG)
        mid.pack(fill="both", expand=True, padx=M)
        frame = tk.Frame(mid, bg=FG, bd=0)
        frame.pack(side="left", fill="both", expand=True)
        self.screen = tk.Canvas(frame, bg=BG, highlightthickness=0, bd=0, width=320, height=200)
        self.screen.pack(fill="both", expand=True, padx=1, pady=1)     # рамка екрана - 1 px, як і всі інші лінії інтерфейсу
        self.screen.bind("<ButtonPress-1>", self._screen_press)        # клік - пауза/відтворення, перетягування - зсув при масштабі
        self.screen.bind("<B1-Motion>", self._screen_drag)
        self.screen.bind("<ButtonRelease-1>", self._screen_release)
        self.screen.bind("<Button-2>", lambda e: self._zoom_reset())   # середня кнопка - вписати
        self.screen.bind("<Configure>", lambda e: self._redraw())
        self.bind_all("<MouseWheel>", self._on_wheel)                  # колесо: одне на все вікно (див. _on_wheel)
        self.bind_all("<Button-4>", self._on_wheel)                    # X11 (колесо як кнопки 4/5)
        self.bind_all("<Button-5>", self._on_wheel)
        # права панель: вікно фіксованої ширини; вміст завжди тягнеться на всю висоту (низ панелі = низ екрана), а коли не влазить - прокручується
        self.edge = PixButton(mid, "", command=self._toggle_side, width=EDGE_W, height=H_M, icon="left")
        self.edge.FLASH = False   # лише в F11: висуває/ховає праву панель
        self._tip(self.edge, "Права панель: сховати / показати")
        self.side = tk.Frame(mid, bg=BG, width=SIDE_W)
        self.side.pack(side="left", fill="y", padx=(M, 0))
        self.side.pack_propagate(False)
        self.side_in = tk.Frame(self.side, bg=BG)
        self.side_bar = PixScroll(self.side, SB_W)
        self.sstack = ScrollStack(self.side, self.side_bar,
                                  [(self.side_in, lambda w, h: max(self.side_in.winfo_reqheight(), h), None)], gap=M)
        self._build_side()
        self._apply_side()

        # --- низ: перемотування й транспорт
        self._strip2 = self._tile_strip(main, "film", 12)
        self._strip2.pack(fill="x", padx=M, pady=(M, 0))
        bot = tk.Frame(main, bg=BG)
        bot.pack(fill="x", padx=M, pady=M)
        self.seek = PixSlider(bot, 0, 1, command=self._on_seek_drag, on_release=self._on_seek_end, height=34)      # ставиться в _layout_bottom
        row = tk.Frame(bot, bg=BG)
        row.pack(fill="x")
        self.tr = tk.Frame(row, bg=BG)
        self.bot_row = row
        self.btn_rew = PixButton(self.tr, icon="rew", command=lambda: self._jump(-10), width=48, height=H_L)
        self.btn_prev = PixButton(self.tr, icon="prev", command=lambda: self._step(-1), width=48, height=H_L)
        self.btn_play = PixButton(self.tr, icon="play", command=self._toggle, width=64, height=H_L)
        self.btn_stopp = PixButton(self.tr, icon="stop", command=self._stop, width=48, height=H_L)
        self.btn_next = PixButton(self.tr, icon="next", command=lambda: self._step(1), width=48, height=H_L)
        self.btn_fwd = PixButton(self.tr, icon="fwd", command=lambda: self._jump(10), width=48, height=H_L)
        for i, (b, tip) in enumerate(((self.btn_rew, "−10 с   [Shift + ←]"), (self.btn_prev, "Кадр назад   [←]"),
                                      (self.btn_play, lambda: tr("Пауза / продовження рендеру   [Пробіл]" if self.exporting else "Відтворення / пауза   [Пробіл]")),
                                      (self.btn_stopp, "Стоп: на початок   [Home]"),
                                      (self.btn_next, "Кадр уперед   [→]"), (self.btn_fwd, "+10 с   [Shift + →]"))):
            b.pack(side="left", padx=(M if i else 0, 0))
            self._tip(b, tip)
        self.btn_snd = PixButton(row, icon="snd", command=self._mute, width=48, height=H_L)
        try:
            vol0 = min(100, max(0, int(self.ui.get("volume", 100))))
        except (TypeError, ValueError):
            vol0 = 100
        self.vol = PixSlider(row, 0, 100, command=self._on_volume, on_release=self._on_volume_end, height=26, width=96)
        self.vol.set(vol0)
        self.player.set_volume(vol0 / 100)
        self.btn_snd.set_icon("mute" if vol0 == 0 else "snd")
        self.btn_loop = self._L(PixButton(row, "", command=self._toggle_loop, width=72, height=H_L, toggle=True), "Цикл")
        self.btn_photo = self._L(PixButton(row, "", command=self._save_frame, width=152, height=H_L), "Кадр → фото")
        self.time_lbl = tk.Label(row, text="0:00 / 0:00", bg=BG, fg=FG, font=FONT_B, anchor="w")
        self.stat_lbl = tk.Label(row, text="", bg=BG, fg=DIM, font=FONT_S, anchor="e")
        self.stat_lbl.bind("<Configure>", lambda e: setattr(self, "_last_rng_t", 0.0))      # змінилась ширина - рядок стану добирається знову
        self._stat_full = ""
        self._f_stat = tkfont.Font(font=FONT_S)
        self._layout_bottom("video")
        self._tip(self.btn_snd, "Звук: вимкнути / увімкнути   [M]")
        self._tip(self.vol, "Гучність   [↑ ↓]  або колесо миші")
        self._tip(self.btn_loop, "Повтор: після кінця починати спочатку   [L]")
        self._tip(self.btn_photo, "Зберегти поточний кадр окремим фото")
        self._tip(self.seek, "Перемотування. Прапорці [ ] над шкалою — фрагмент для збереження")
        self._tip(self.stat_lbl, lambda: self._stat_full)
        self._tip(self.btn_open, "Відкрити відео, GIF або фото   [Ctrl + O]")
        self._tip(self.btn_export, "Зберегти результат   [Ctrl + S]")
        self._tip(self.btn_stop, "Зупинити збереження")
        self._tip(self.btn_help, "Гайд, клавіші, про програму   [F1]")
        self._tip(self.btn_ua, "Українська")
        self._tip(self.btn_en, "English")
        self._build_footer()

    # --- будівельні цеглини однакових розмірів: смужка заголовка, рамка фіксованої висоти, перенос тексту по ширині
    def _titlebar(self, parent, key):
        """інвертована смужка із заголовком (скрізь однакової висоти BAR_H, текст починається там, де вміст під нею) -> Label"""
        bar = tk.Frame(parent, bg=FG, height=BAR_H)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        lbl = self._L(tk.Label(bar, bg=FG, fg=BG, font=FONT, anchor="w", padx=M), key, upper=True)
        lbl.pack(fill="both", expand=True)
        return lbl

    @staticmethod
    def _boxed(parent, h=H_S, w=1):
        """рамка фіксованої висоти: поле введення чи список усередині неї розтягується рівно на цю висоту, яким би не був шрифт"""
        f = tk.Frame(parent, bg=BG, height=h, width=w)
        f.pack_propagate(False)
        return f

    @staticmethod
    def _autowrap(lbl):
        """текст підпису переноситься по його фактичній ширині (панель може стати вужчою, коли з'являється смужка прокрутки)"""
        lbl.bind("<Configure>", lambda e: lbl.config(wraplength=max(60, e.width)))
        return lbl

    def _section(self, parent, key, last=False):
        """рамка з написом на інвертованій смузі; -> внутрішній Frame. Сам заголовок лишається в self._sec_title.
        Між секціями проміжок M; остання (last) тягнеться до низу, щоб низ панелі збігався з низом екрана"""
        f = tk.Frame(parent, bg=BG, highlightbackground=FG, highlightthickness=1)
        if last:
            f.pack(fill="both", expand=True)
        else:
            f.pack(fill="x", pady=(0, M))
        self._sec_title = self._titlebar(f, key)
        inner = tk.Frame(f, bg=BG)
        inner.pack(fill="both" if last else "x", expand=last, padx=M, pady=M)
        return inner

    def _build_side(self):
        side = self.side_in
        wrap0 = SIDE_W - 2 - 2 * M                          # ширина вмісту секції; далі перенос слідкує за фактичною (_autowrap)
        s = self._section(side, "Якість")
        self.q_title = self._sec_title                      # у смузі заголовка - кодова назва поточного рівня
        try:
            m0 = min(BRANCH, max(1, int(self.ui.get("method", 4))))
        except (TypeError, ValueError):
            m0 = 4
        self.params.method = m0
        self.params.glyph = bool(self.ui.get("glyph", False))
        self._branch_back = False                          # у «Символах» гілка IRNM1 вимкнена; ця позначка повертає її при поверненні до чорнила
        if self.params.glyph and m0 > MAIN_LEVELS:
            self.params.method = m0 = MAIN_LEVELS
            self._branch_back = True
        try:
            self.params.gdens = min(3, max(1, int(self.ui.get("gdens3", 3))))     # «gdens3»: колишня шкала 1..5 (ключ «gdens») тут не годиться
        except (TypeError, ValueError):
            self.params.gdens = 3
        self.player.params = self.params
        row = tk.Frame(s, bg=BG)                           # два способи дивитись на кадр: чорнило (штрихи й крапки) або символи (знаки шрифту)
        row.pack(fill="x", pady=(0, M))
        self.btn_ink = self._L(PixButton(row, "", command=lambda: self._set_glyph(False), height=H_S), "Чорнило")
        self.btn_sym = self._L(PixButton(row, "", command=lambda: self._set_glyph(True), height=H_S, join="l"), "Символи")
        self.btn_ink.pack(side="left", fill="x", expand=True)
        self.btn_sym.pack(side="left", fill="x", expand=True)
        self.btn_ink.set_on(not self.params.glyph)
        self.btn_sym.set_on(self.params.glyph)
        self._tip(self.btn_ink, "Чорнило: кадр малюється штрихами, крапками й зерном")
        self._tip(self.btn_sym, "Символи: кадр складається зі знаків шрифту (як текст у консолі); ті самі п'ять рівнів якості")
        self.m_slider = QualityBar(s, MAIN_LEVELS, m0, command=self._apply_method, branch="IRNM1")
        self.m_slider.set_branch(not self.params.glyph)
        self.m_slider.pack(fill="x")
        self.m_info = tk.Label(s, text="", bg=BG, fg=DIM, font=FONT_S, anchor="nw", justify="left", wraplength=wrap0, height=2)
        self.m_info.pack(fill="x", pady=(G, 0))
        self._autowrap(self.m_info)
        self.dens_box = tk.Frame(s, bg=BG)                  # лише в «Символах»: скільки знаків вміщається в кадр (три чітких позиції)
        self.dens_lbl = tk.Label(self.dens_box, text="", bg=BG, fg=FG, font=FONT_S, anchor="nw", justify="left", height=2)
        self.dens_lbl.pack(fill="x", pady=(G, 0))
        self._autowrap(self.dens_lbl)
        self.dens_bar = QualityBar(self.dens_box, 3, self.params.gdens, command=self._apply_gdens, ends=("крупно", "дрібно"))
        self.dens_bar.pack(fill="x")
        if self.params.glyph:
            self.dens_box.pack(fill="x", after=self.m_info)
        self._tip(self.m_slider, "Баланс «швидкість ↔ якість»: п'ять рівнів. Чим вище, тим більше роботи на кадр. Ромб справа - IRNM1 (InkreelNeuralModel v1): "
                                 "не шостий щабель, а окрема гілка (навчена нейромережа на відеокарті); у «Символах» її нема")
        self._tip(self.dens_bar, "Щільність знаків: розмір знака 32×64, 16×32 або 8×16 px. Чим дрібніше, тим більше знаків уміщається в той самий кадр і тим довше рахується")
        self._show_method(m0)

        s = self._section(side, "Розмір результату")
        sz = self.ui.get("size", AUTO)
        if sz not in SIZE_KEYS and not (isinstance(sz, str) and "x" in sz.lower() and sz.lower().rstrip("p") == sz.lower()):
            sz = AUTO                                   # застарілий пресет (напр. 2K) з файлу налаштувань
        self.size_var = tk.StringVar(value=sz)
        self.size_btns = {}
        # ґратка 4 × 3: чотири рівні колонки кнопок (розтягуються до країв вмісту) і вузькі колонки-проміжки по G; ряди через G
        grid = tk.Frame(s, bg=BG)
        grid.pack(fill="x")
        for c in range(7):
            if c % 2:
                grid.columnconfigure(c, minsize=G)
            else:
                grid.columnconfigure(c, weight=1, uniform="sz")
        rows = (len(SIZE_KEYS) + 3) // 4
        for i, k in enumerate(SIZE_KEYS):
            b = PixButton(grid, "", command=lambda k=k: self._set_size(k), width=48, height=H_S)
            if k == AUTO:
                self._L(b, "Авто")
                self._tip(b, "Найбільший розмір, який ПК тягне в реальному часі на обраному рівні якості")
            elif k == "Як у джерела":
                self._L(b, "Джерело")
                self._tip(b, "Розмір оригіналу")
            else:
                b.text = k.replace("x", "×")
                b.draw()
            r, c = divmod(i, 4)
            b.grid(row=r, column=2 * c, sticky="ew", pady=(0, G if r < rows - 1 else 0))
            self.size_btns[k] = b
        # «свій розмір»: підпис ліворуч, два поля рівної ширини, «OK» праворуч (його ширина = ширина кнопки ґратки)
        row = tk.Frame(s, bg=BG)
        row.pack(fill="x", pady=(M, 0))
        self._L(tk.Label(row, bg=BG, fg=FG, font=FONT_S), "свій:").pack(side="left", padx=(0, G))
        self.w_var, self.h_var = tk.StringVar(), tk.StringVar()
        for v in (self.w_var, None, self.h_var):
            if v is None:
                tk.Label(row, text="×", bg=BG, fg=FG, font=FONT_S).pack(side="left", padx=G)
                continue
            box = self._boxed(row)
            box.pack(side="left", fill="x", expand=True)
            e = tk.Entry(box, textvariable=v, width=4, bg=BG, fg=FG, insertbackground=FG, selectbackground=FG, selectforeground=BG, relief="flat",
                         disabledbackground=BG, disabledforeground=FG, readonlybackground=BG, highlightbackground=FG, highlightcolor=FG, highlightthickness=1, font=FONT, justify="center")
            e.pack(fill="both", expand=True)
            e.bind("<Return>", lambda ev: self._apply_custom())
            e.bind("<<Paste>>", self._paste_replace)            # вставка замінює вміст поля; «1920x1080» з буфера лягає одразу в обидва поля
            e.bind("<FocusOut>", lambda ev: _clear_sel(ev.widget))
            e.bind("<FocusIn>", lambda ev: ev.widget.after(1, lambda w=ev.widget: _select_all(w)))     # клік у поле виділяє його вміст: набір чи вставка замінюють
            if v is self.w_var:
                self.ent_w = e
            else:
                self.ent_h = e
        self.btn_ok = PixButton(row, "OK", command=self._apply_custom, width=70, height=H_S)
        self.btn_ok.pack(side="left", padx=(G, 0))
        self.size_lbl = tk.Label(s, text="", bg=BG, fg=DIM, font=FONT_S, anchor="nw", justify="left", wraplength=wrap0, height=2)
        self.size_lbl.pack(fill="x", pady=(G, 0))
        self._autowrap(self.size_lbl)

        s = self._section(side, "Палітра")
        self.pal_var = tk.StringVar(value=palette.split(self.ui.get("palette"))[0])
        self.pal_inv = bool(self.ui.get("pal_inv"))                  # «навпаки»: туш і папір міняються місцями
        prow = tk.Frame(s, bg=BG)
        prow.pack(fill="x")
        self.btn_pprev = PixButton(prow, "", command=lambda: self._step_palette(-1), width=W_SQ, height=H_M, icon="left")
        self.btn_pprev.pack(side="left")
        self.pal_sw = tk.Canvas(prow, width=30, height=18, bg=BG, highlightthickness=0, bd=0)
        self.pal_sw.pack(side="left", padx=(M, 0))
        self.pal_sw.bind("<Button-1>", lambda e: self._swatch_click())
        self.pal_name = tk.Label(prow, text="", bg=BG, fg=FG, font=FONT, anchor="w")
        self.pal_name.pack(side="left", fill="x", expand=True, padx=M)
        self.btn_pnext = PixButton(prow, "", command=lambda: self._step_palette(1), width=W_SQ, height=H_M, icon="right")
        self.btn_pnext.pack(side="right")
        self.btn_pinv = PixButton(prow, "", command=self._toggle_inv, width=W_SQ, height=H_M, icon="swap")
        self.btn_pinv.pack(side="right", padx=(0, G))
        self.btn_pinv.set_on(self.pal_inv)
        self._tip(self.btn_pprev, "Попередня палітра")
        self._tip(self.btn_pnext, "Наступна палітра")
        self._tip(self.btn_pinv, "Навпаки: туш і папір міняються місцями   [I]")
        self._draw_swatch()

        s = self._section(side, "Експорт", last=True)
        row = tk.Frame(s, bg=BG)
        row.pack(fill="x")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:  # noqa: BLE001
            pass
        self._style_combos()
        self.fmt_var = tk.StringVar(value=self.ui.get("fmt", "MP4 (H.264)"))          # ключі - українські назви з video/media
        self.q_var = tk.StringVar(value=self.ui.get("quality", "Висока"))
        self._fmt_keys = list(FORMATS)
        fbox = self._boxed(row)                                      # формат - на всю вільну ширину, якість - вузька колонка праворуч
        fbox.pack(side="left", fill="x", expand=True)
        self.fmt_cb = ttk.Combobox(fbox, values=[], state="readonly", width=1, style="Pix.TCombobox", font=FONT)
        self.fmt_cb.pack(fill="both", expand=True)
        self.q_box = self._boxed(row, w=Q_W)
        self.q_box.pack(side="left", padx=(G, 0))
        self.q_cb = ttk.Combobox(self.q_box, values=[], state="readonly", width=1, style="Pix.TCombobox", font=FONT)
        self.q_cb.pack(fill="both", expand=True)
        self.fmt_cb.bind("<<ComboboxSelected>>", self._on_fmt_pick)
        self.q_cb.bind("<<ComboboxSelected>>", self._on_q_pick)
        self._tip(self.q_cb, "Якість кодування. «Висока» - програмний x264 без поспіху; «Середня» й «Мала» - апаратне "
                             "кодування відеокарти, швидше й менші файли")
        row = tk.Frame(s, bg=BG)
        row.pack(fill="x", pady=(M, 0))
        self.row2 = row
        self.btn_aud = self._L(PixButton(row, "", command=lambda: None, width=TOG_W, height=H_S, toggle=True), "Звук")
        self.btn_aud.pack(side="left")
        self.btn_aud.set_on(self.ui.get("audio", True))
        self.btn_rng = self._L(PixButton(row, "", command=self._reset_range, width=TOG_W, height=H_S), "Весь")
        self.btn_rng.pack(side="right")
        self.rng_lbl = tk.Label(row, text="", bg=BG, fg=FG, font=FONT_S, anchor="w", justify="left")
        self.rng_lbl.pack(side="left", fill="x", expand=True, padx=M)
        self._tip(self.btn_aud, "Додати до файлу звук оригіналу")
        self._tip(self.btn_rng, "Скинути фрагмент: зберігати все відео")
        try:
            self.fps_sel = max(0.0, float(self.ui.get("fps", 0) or 0))   # ліміт кадрів на секунду, який хоче користувач (0 - як в оригіналі); діє, коли він нижчий за fps відео
        except (TypeError, ValueError):
            self.fps_sel = 0.0
        self._fps_vals = [0]
        self._fps_orig = self._fps_shown = ""
        self.row_fps = tk.Frame(s, bg=BG)
        self.row_fps.pack(fill="x", pady=(M, 0))
        fpsbox = self._boxed(self.row_fps)
        fpsbox.pack(fill="x")
        self.fps_cb = ttk.Combobox(fpsbox, values=[], state="normal", width=1, style="Pix.TCombobox", font=FONT)     # не лише список: число можна вписати
        self.fps_cb.pack(fill="both", expand=True)
        self.fps_cb.bind("<<ComboboxSelected>>", self._on_fps_pick)
        self.fps_cb.bind("<Return>", self._on_fps_typed)
        self.fps_cb.bind("<KP_Enter>", self._on_fps_typed)
        self.fps_cb.bind("<FocusOut>", self._on_fps_typed, add="+")
        self.fps_cb.bind("<FocusOut>", lambda e: _clear_sel(self.fps_cb), add="+")
        self.fps_cb.bind("<FocusIn>", lambda e: self.fps_cb.after(1, lambda: _select_value(self.fps_cb)))
        self.fps_cb.bind("<Button-1>", self._fps_click)
        for ev in ("<B1-Motion>", "<Double-Button-1>", "<Triple-Button-1>"):             # текст «Кадри:» не виділяється мишкою; змінюється лише значення після двокрапки
            self.fps_cb.bind(ev, lambda e: None if "arrow" in str(self.fps_cb.identify(e.x, e.y)) else "break")
        self.fps_cb.bind("<<Paste>>", self._paste_replace)
        self._tip(self.fps_cb, "Кадрів на секунду в результаті: виберіть зі списку або впишіть своє число (Enter). «Як в оригіналі» - без змін; "
                               "нижче значення - легший файл і швидший розрахунок: кадри беруться за часом, тривалість і звук лишаються")
        self.btn_export2 = PixButton(s, "", self.imgs["save"], self._export, width=100, height=H_M)
        self.btn_export2.pack(fill="x", pady=(M, 0))
        self.btn_lock = PixButton(s, "", command=self._lock_toggle, width=100, height=H_M)     # відео: «порахувати все» у файл .ikr (замок) / «звичайний режим» назад
        self.btn_lock.pack(fill="x", pady=(M, 0), after=self.btn_export2)
        self._tip(self.btn_lock, lambda: tr("Звичайний режим: повернутись до оригіналу цього відео з тими самими налаштуваннями. Уже пораховані кадри лишаються")
                  if self.locked else tr("Порахувати все: обробити все відео наперед і зберегти у файл .ikr (без втрат). Він відкриється в режимі замка - "
                                         "кадри готові, перемотка й експорт миттєві, нічого не перераховується; файл можна відкрити в InkReel будь-коли"))
        self.est_lbl = tk.Label(s, text="", bg=BG, fg=DIM, font=FONT_S, anchor="w")
        self.est_lbl.pack(fill="x", pady=(G, 0))
        self.prog = PixProgress(s, height=14)
        self.prog.pack(fill="x", pady=(G, 0))
        self.kind = "video"
        self._prog_msg, self._prog_done = None, False
        self.prog_lbl = tk.Label(s, text="", bg=BG, fg=DIM, font=FONT_S, anchor="nw", justify="left", wraplength=wrap0, height=3)
        self.prog_lbl.pack(fill="both", expand=True, pady=(G, 0))
        self._autowrap(self.prog_lbl)
        self.prog_lbl.bind("<Button-1>", lambda e: self._open_out())
        self._tip(self.prog_lbl, lambda: (self._last_out or "") if self._prog_done else "")
        self._sync_combos()
        self._refresh_btns()
        self._update_size_label()

    # --- футер: другий екран сторінки (на все вікно): котушка з логотипом, гайд, філософія, про програму, місце під донат
    def _build_footer(self):
        f = self.foot
        self._foot_key = None
        self.guide_sel = self.ui.get("guide") if self.ui.get("guide") in guide.SECTIONS else "start"
        # екран «в рамці» плівки: стрічка вгорі й унизу; вміст між ними - рівно по центру вільного місця (pack без fill)
        self._tile_strip(f, "film", 12).pack(fill="x", padx=M, pady=(M, 0))
        self._tile_strip(f, "film", 12).pack(side="bottom", fill="x", padx=M, pady=(0, M))
        body = tk.Frame(f, bg=BG)
        body.pack(fill="both", expand=True)
        col = tk.Frame(body, bg=BG)
        col.pack(expand=True)
        self.foot_col = col
        head = tk.Frame(col, bg=BG)                                         # котушка й логотип, подвоєні (піксель-арт 2x)
        head.pack(pady=(FOOT_GAP, 0))
        self._foot_mascot = self._foot_logo = None
        if self.mascot2:
            self._foot_mascot = tk.Label(head, image=self.mascot2, bg=BG, bd=0)
            self._foot_mascot.pack(side="left", padx=(0, GG))
        if self.logo2:
            self._foot_logo = tk.Label(head, image=self.logo2, bg=BG, bd=0)
            self._foot_logo.pack(side="left")
        self._foot_strut = tk.Frame(col, height=1, width=900, bg=BG)       # ширина колонки: її задає _foot_fit
        self._foot_strut.pack(pady=(FOOT_GAP - 1, 0))
        box = tk.Frame(col, bg=BG, highlightbackground=FG, highlightthickness=1)
        box.pack(fill="x")
        self._titlebar(box, "Гайд та інфо")
        tabs = tk.Frame(box, bg=BG)
        tabs.pack(fill="x", padx=M, pady=(M, 0))
        self.guide_tabs = {}
        for i, sec in enumerate(guide.SECTIONS):          # шість вкладок рівної ширини на всю ширину рамки, між ними M
            if i:
                tabs.columnconfigure(2 * i - 1, minsize=M)
            tabs.columnconfigure(2 * i, weight=1, uniform="tab")
            b = PixButton(tabs, "", command=lambda s=sec: self._guide_pick(s), width=60, height=H_M)
            b.grid(row=0, column=2 * i, sticky="ew")
            self.guide_tabs[sec] = b
        self.guide_box = tk.Frame(box, bg=BG)             # висота = найвищий розділ: вкладки не стрибають
        self.guide_box.pack(fill="x", padx=M, pady=M)
        self.guide_box.pack_propagate(False)
        self.guide_lbl = tk.Label(self.guide_box, text="", bg=BG, fg=FG, font=FONT_S, justify="left", anchor="nw")
        self.guide_lbl.pack(fill="both", expand=True)
        self._measure = tk.Label(col, font=FONT_S, justify="left")        # лише міряти висоту тексту; ніколи не показується
        q = tk.Frame(col, bg=BG)
        q.pack(fill="x", pady=(FOOT_GAP, 0))
        self._tile_strip(q, "dither", 8).pack(fill="x")
        self.phil_lbl = tk.Label(q, text="", bg=BG, fg=FG, font=FONT_B, justify="center", cursor="hand2")
        self.phil_lbl.pack(pady=GG)
        self.phil_lbl.bind("<Button-1>", lambda e: self._quote_click())
        self._tile_strip(q, "dither", 8).pack(fill="x")
        self.tip_lbl = tk.Label(q, text="", bg=BG, fg=DIM, font=FONT_S, justify="center")        # «Порада»: не показується під час відтворення й експорту
        self._tip(self.phil_lbl, "…розфокусуй погляд")
        bar = tk.Frame(col, bg=BG)
        bar.pack(fill="x", pady=(FOOT_GAP, 16))
        bar.columnconfigure(0, weight=1, uniform="foot")
        bar.columnconfigure(2, weight=1, uniform="foot")
        self.about_lbl = tk.Label(bar, text="", bg=BG, fg=DIM, font=FONT_S, justify="left", anchor="nw", padx=0, pady=0)
        self.about_lbl.grid(row=0, column=0, sticky="nw")
        if meta.links():                                  # клік по «InkReel 1.0.0»: меню з офіційними посиланнями (сайт, GitHub, Store)
            self.about_lbl.config(cursor="hand2")
            self.about_lbl.bind("<Button-1>", self._links_menu)
            self._tip(self.about_lbl, "Офіційні джерела: сайт, GitHub, Microsoft Store")
        self.btn_donate = None
        if meta.donate_enabled():
            self.btn_donate = PixButton(bar, "", command=self._donate, width=288, height=H_M, icon="cookie")
            self.btn_donate.grid(row=0, column=1)
            self.btn_donate.bind("<Button-1>", lambda e: self._cookie_click(), add="+")
            self._L(self.btn_donate, "Пригостити автора печивом")
            self.btn_donate.set_enabled(bool(meta.DONATE))
            self._tip(self.btn_donate, "Скоро тут з'являться посилання" if not meta.DONATE else "Подякувати автору печивом")
        self.btn_top = PixButton(bar, "", command=lambda: self.pstack.go(0), width=120, height=H_M, icon="up")
        self.btn_top.grid(row=0, column=2, sticky="ne")
        self._L(self.btn_top, "Нагору")
        self._fill_footer()

    def _fill_footer(self):
        """тексти футера поточною мовою"""
        for sec, b in self.guide_tabs.items():
            b.text = guide.title(sec)
            b.set_on(sec == self.guide_sel)
        self.guide_lbl.config(text=guide.body(self.guide_sel))
        self.phil_lbl.config(text="«" + guide.philosophy() + "»" if i18n.lang() == i18n.UK else "“" + guide.philosophy() + "”")
        self._fill_tip()
        a = f"{meta.APP} {meta.VERSION}" + (f"\n{meta.AUTHOR}" if meta.AUTHOR else "")
        self.about_lbl.config(text=a)
        self._foot_key = None

    def _tip_visible(self) -> bool:
        """«Порада» мовчить, поки грає відео чи йде експорт"""
        pl = getattr(self, "player", None)
        return not (self.exporting or (pl is not None and pl.playing and getattr(pl, "path", None)))

    def _fill_tip(self):
        """порада під філософією у футері (або нічого, коли йде відтворення чи експорт)"""
        if not hasattr(self, "tip_lbl"):
            return
        if self._tip_n < 0:
            self._tip_n = guide.tip(-1)
        if self._tip_visible():
            self.tip_lbl.config(text=guide.tip_text(self._tip_n))
            self.tip_lbl.pack(pady=(0, M))                  # відступ знизу малий: до рядка з версією ще FOOT_GAP, а футер має влізти у вікно 640 px
        else:
            self.tip_lbl.pack_forget()

    def _foot_fit(self, cw):
        """ширина колонки футера під ширину вікна; висота тексту гайда - за найвищим розділом, щоб вкладки не стрибали"""
        W = max(560, min(cw - 48, 980))
        key = (W, i18n.lang())
        if key == self._foot_key:
            return
        self._foot_key = key
        self._foot_strut.config(width=W)
        wrap = W - 2 - 2 * M - 4                          # ширина тексту = ширина рамки без ліній, відступів і запасу
        self.guide_lbl.config(wraplength=wrap)
        self.phil_lbl.config(wraplength=min(W - 60, 760))
        self.tip_lbl.config(wraplength=W - 60)           # найдовша порада - один рядок (інакше футер на мінімальному вікні вищий на рядок і залежить від випадкової поради)
        self.about_lbl.config(wraplength=max(120, W // 3 - 10))
        tallest = 0
        for sec in guide.SECTIONS:
            self._measure.config(text=guide.body(sec), wraplength=wrap)
            tallest = max(tallest, self._measure.winfo_reqheight())
        self.guide_box.config(height=tallest + 4)

    def _guide_pick(self, sec):
        self.guide_sel = sec
        self._fill_footer()
        self.pstack.layout()
        self._save_soon()

    def _show_app(self):
        """відкрили файл чи почали збереження, поки на екрані футер: назад до програми - усе відбувається там"""
        if self.pstack.tcur > 0:
            self.pstack.go(0)

    def _goto_guide(self):
        """«?» і F1: до футера з гайдом; ще раз - назад до програми"""
        if self.pstack.tcur == 0:
            self._tip_n = guide.tip(self._tip_n)            # у футер зайшли заново: нова порада
            self._fill_tip()
        self.pstack.go(0 if self.pstack.tcur > 0 else 1)

    def _links_menu(self, event=None):
        """офіційні посилання: сайт, код і релізи на GitHub, сторінка в Microsoft Store"""
        m = tk.Menu(self, tearoff=0, bg=BG, fg=FG, activebackground=FG, activeforeground=BG, bd=1, relief="flat", font=FONT_S)
        for name, url in meta.links():
            m.add_command(label=tr(name), command=lambda u=url: webbrowser.open(u))
        b = self.about_lbl
        try:
            m.tk_popup(b.winfo_rootx(), b.winfo_rooty() + b.winfo_height())
        finally:
            m.grab_release()

    def _donate(self):
        links = list(meta.DONATE)
        if not links:
            return
        if len(links) == 1:
            webbrowser.open(links[0][1])
            return
        m = tk.Menu(self, tearoff=0, bg=BG, fg=FG, activebackground=FG, activeforeground=BG, bd=1, relief="flat", font=FONT_S)
        for name, url in links:
            m.add_command(label=name, command=lambda u=url: webbrowser.open(u))
        b = self.btn_donate
        try:
            m.tk_popup(b.winfo_rootx(), b.winfo_rooty() + b.winfo_height())
        finally:
            m.grab_release()

    # ------------------------------------------------------------ пасхалки (логіка - у benvid/eggs.py, тут лише малювання)
    def _bubble(self, widget, text, ms=3600):
        """хмаринка з фразою під віджетом; сама зникає"""
        self._bubble_close()
        try:
            win = tk.Toplevel(self)
            win.withdraw()                               # невидимим до готовності: без блимання в куті екрана
            win.wm_overrideredirect(True)
            _above_owner(win, self)
            tk.Label(win, text=text, bg=BG, fg=FG, font=FONT_S, justify="left", padx=M, pady=G, bd=0, highlightthickness=1,
                     highlightbackground=FG, wraplength=300).pack()
            win.update_idletasks()
            w, h = win.winfo_reqwidth(), win.winfo_reqheight()
            cx, cy = widget.winfo_rootx() + widget.winfo_width() // 2, widget.winfo_rooty() + widget.winfo_height() // 2
            x, y = popup.under_widget(widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_width(), widget.winfo_height(),
                                      w, h, popup.work_area(self, cx, cy))
            win.wm_geometry(f"+{x}+{y}")
            win.deiconify()
            self._bub = win
            self._bub_job = self.after(ms, self._bubble_close)
        except tk.TclError:
            self._bub = None

    def _bubble_close(self):
        if self._bub_job:
            try:
                self.after_cancel(self._bub_job)
            except tk.TclError:
                pass
            self._bub_job = None
        if self._bub is not None:
            try:
                self._bub.destroy()
            except tk.TclError:
                pass
            self._bub = None

    def _reel_click(self):
        """котушка в шапці: крутиться й каже фразу"""
        self._spin_reel()
        self._last_quote = eggs.pick_quote(self._last_quote)
        lbl = getattr(self, "_lbl_mascot", None)
        if lbl is not None:
            self._bubble(lbl, tr(eggs.QUOTES[self._last_quote]))

    def _spin_reel(self):
        """один оберт котушки (кадри повернутої картинки; будуються при першому разі й при зміні теми)"""
        lbl = getattr(self, "_lbl_mascot", None)
        if lbl is None or self._spin_job is not None:
            return
        if self._spin_key != (BG, FG):
            self._spin_frames = [_img("mascot_w", rot=a) for a in eggs.spin_angles()[:-1]]
            self._spin_key = (BG, FG)
        frames = self._spin_frames
        if not frames or any(f is None for f in frames):
            return
        step = max(10, int(eggs.SPIN_MS / len(frames)))

        def go(i):
            try:
                if i >= len(frames):
                    lbl.configure(image=self.mascot)
                    self._spin_job = None
                    return
                lbl.configure(image=frames[i])
                self._spin_job = self.after(step, lambda: go(i + 1))
            except tk.TclError:
                self._spin_job = None
        go(0)

    def _logo_click(self):
        """лого ×5: картинка розфокусовується під слова філософії (без картинки - котушка каже фразу)"""
        if self._egg_clicks["logo"].hit():
            if self._last_mask is not None:
                self._egg_defocus()
            else:
                self._reel_click()

    def _quote_click(self):
        """клік по цитаті в футері: нагору до картинки - і вона розфокусовується під ці слова (щойно перехід доїде)"""
        st = self.pstack
        if st.tcur > 0:
            st.go(0, done=self._egg_defocus)
        else:
            self._egg_defocus()

    def _egg_defocus(self):
        if self._egg_t0 is not None or self._last_mask is None:
            return
        self._egg_t0 = time.perf_counter()
        self._egg_tick()

    def _egg_tick(self):
        t = time.perf_counter() - self._egg_t0
        if t >= eggs.DEFOCUS_TOTAL:
            self._defocus, self._egg_t0, self._egg_job = 0.0, None, None
            self._redraw()
            return
        self._defocus = eggs.defocus_curve(t)
        self._redraw()
        self._egg_job = self.after(33, self._egg_tick)

    def _draw_caption(self, cw, ch):
        """підпис, поки картинка розфокусована: два рядки філософії проявляються в рамці внизу екрана (рамка з'являється,
        коли перший рядок уже читається; Tk не вміє напівпрозорість, тож проявляємо колір тексту від тла до кольору)"""
        a1, a2 = eggs.caption_alphas(time.perf_counter() - self._egg_t0)
        if max(a1, a2) < 0.25:
            return
        l1, l2 = eggs.philosophy_parts()
        fnt = tkfont.Font(font=FONT_B)
        per = max(16, max(200, min(cw - 80, 660)) // max(1, fnt.measure("0")))      # символів у рядку (шрифт моноширинний)
        wrapped = lambda s: "\n".join(textwrap.wrap(s, per)) if s else ""
        shade = lambda a: eggs.blend_hex(BG, FG, (a - 0.25) / 0.75)
        c = self.screen
        t2 = c.create_text(cw // 2, ch - 34, text=wrapped(l2), anchor="s", justify="center", font=FONT_B, fill=shade(a2))
        b2 = c.bbox(t2)
        t1 = c.create_text(cw // 2, b2[1] - 10, text=wrapped(l1), anchor="s", justify="center", font=FONT_B, fill=shade(a1))
        b1 = c.bbox(t1)
        r = c.create_rectangle(min(b1[0], b2[0]) - 16, b1[1] - 12, max(b1[2], b2[2]) + 16, b2[3] + 12, fill=BG, outline=FG)
        c.tag_lower(r, t2)

    def _swatch_click(self):
        """квадратик палітри ×7: вечірка - вікно перебирає всі палітри й повертається до вашої"""
        if self._egg_clicks["swatch"].hit() and not self.exporting and self._party_job is None:
            cur = self._party_orig = self.pal_var.get()
            self._party(eggs.party_order(palette.NAMES, cur), cur, 0)

    def _party(self, order, orig, i):
        if i < len(order):
            name = order[i]
            self.pal_var.set(name)
            self._apply_theme(self._pal_key())
            self._draw_swatch()
            self._redraw()
            self._party_job = self.after(170, lambda: self._party(order, orig, i + 1))
            return
        self._party_job = None
        if self.pal_var.get() != orig:
            self.pal_var.set(orig)
            self._apply_theme(self._pal_key())
            self._draw_swatch()
            self._redraw()

    def _party_stop(self):
        """закрити вікно посеред вечірки - зберігається вибір користувача, а не випадкова палітра"""
        if self._party_job is None:
            return
        self.after_cancel(self._party_job)
        self._party_job = None
        self.pal_var.set(self._party_orig)               # (вікно зараз закривається: перефарбовувати вже нема кого)

    def _cookie_click(self):
        """печиво в футері (поки кнопка донату вимкнена): кліки відкушують його, сьомий - з'їдає й розсипає крихти"""
        b, ck = self.btn_donate, self._egg_clicks["cookie"]
        if b is None or b.enabled or b.bites >= eggs.EATEN:
            return
        if self._cookie_job:
            self.after_cancel(self._cookie_job)
            self._cookie_job = None
        if ck.hit():
            b.bites, b.crumbs = eggs.EATEN, eggs.crumbs_burst()
            b.draw()
            self._bubble(b, tr("Ням!"), 1500)
            self._crumbs_tick(time.perf_counter())
            self._cookie_job = self.after(2600, self._cookie_rebake)
        else:
            b.bites = ck.count
            b.draw()
            self._cookie_job = self.after(int(ck.gap * 1000) + 120, self._cookie_rebake)

    def _cookie_rebake(self):
        """кліки припинились (або печиво давно з'їдене): нове печиво"""
        self._cookie_job = None
        self._egg_clicks["cookie"].reset()
        b = self.btn_donate
        if b is not None:
            b.bites, b.crumbs = 0, []
            b.draw()

    def _crumbs_tick(self, last):
        b = self.btn_donate
        if b is None or not b.crumbs:
            return
        now = time.perf_counter()
        b.crumbs = eggs.crumbs_step(b.crumbs, min(0.05, now - last), floor=b._bh / 2)
        b.draw()
        if b.crumbs:
            self.after(30, lambda: self._crumbs_tick(now))

    # ------------------------------------------------------------ стан інтерфейсу
    def _refresh_btns(self):
        cur = self.size_var.get()
        for k, b in self.size_btns.items():
            b.set_on(k == cur)

    def _update_size_label(self):
        info = self.player.info
        if not info:
            self.size_lbl.config(text="")
            return
        w, h = self.player.size
        tag = tr("авто") + " " if self.size_var.get() == AUTO else ""
        src = f"{info['width']}×{info['height']}"
        if self.player.kind != "image" and info.get("fps"):
            src += tr(" · {f:.3f} к/с", f=info["fps"])
        if self.locked:                                       # замок: розмір сталий - такий, яким файл .ikr було порахований
            nm = (self.player.baked_meta.get("src") or {}).get("name", "")
            self.size_lbl.config(text=tr("→ {w}×{h} · замок: розмір сталий\nоригінал {n}", w=w, h=h, n=nm or src))
            return
        if self.params.glyph:                                 # символи: скільки знаків у рядку й скільки рядків
            from . import glyph
            if hasattr(self, "dens_lbl"):
                self._show_dens()
            rows, cols = glyph.grid_for(w, h, self.params.gdens)
            cw, ch = glyph.cell_for(w, h, self.params.gdens)
            self.size_lbl.config(text=tr("→ {tag}{w}×{h} · {c}×{r} символів ({cw}×{ch} px)\nджерело {src}", tag=tag, w=w, h=h, c=cols, r=rows, cw=cw, ch=ch, src=src))
            return
        self.size_lbl.config(text=tr("→ {tag}{w}×{h}\nджерело {src}", tag=tag, w=w, h=h, src=src))

    def _set_size(self, key):
        if self._lock_guard():
            return
        self.size_var.set(key)
        self._refresh_btns()
        if key == AUTO:
            self._auto_resolve()
        else:
            self._auto_size = None
            self.player.set_size(key)
        self._update_size_label()
        self._note_min_size()
        self._save_soon()

    def _note_min_size(self):
        """«Символи»: результат не буває меншим за один знак (при поточній крупності); якщо вибраний розмір менший - його піднято, і про це кажемо"""
        try:
            if not (self.params.glyph and self.ready and self.player.info and not self.locked and not self.exporting):
                return
            from .video import out_size
            info = self.player.info
            if out_size(info["width"], info["height"], self._size_key()) != tuple(self.player.size):
                w, h = self.player.size
                self._say("Розмір піднято до одного символа: {w}×{h}", w=w, h=h)
        except Exception:  # noqa: BLE001
            pass

    def _size_key(self):
        """фактичний ключ розміру (для 'Авто' - те, що підібрано)"""
        k = self.size_var.get()
        return (self._auto_size or self.player.size_key) if k == AUTO else k

    def _auto_resolve(self, keep=False):
        """автовибір розміру у фоні; поки рахується - тимчасово 240p (keep=True: лишається поточний розмір - так при зміні
        рівня якості картинка не блимає малою)"""
        if not keep:
            self._auto_size = None
        if not self.ready:
            return
        if not keep:
            self.player.set_size("240p")
        tok = self._auto_tok = getattr(self, "_auto_tok", 0) + 1
        pl = self.player
        fallback = pl.size_key if keep else "240p"

        def work():
            try:
                k = pl.auto_pick()
            except Exception:  # noqa: BLE001
                _log.get().exception("автовибір розміру")
                k = fallback
            finally:
                pl.hold(False)                                    # про всяк випадок: буфер не повинен лишитись на паузі
            self.q.put(lambda: self._auto_done(tok, k))          # у Tk можна лише з головного потоку
        threading.Thread(target=work, daemon=True).start()

    def _auto_done(self, tok, key):
        if tok != self._auto_tok or self.size_var.get() != AUTO:
            return
        self._auto_size = key
        self.player.set_size(key)
        self._update_size_label()

    def _valid_custom(self, w, h):
        if not (w or h):
            return False
        return all(v is None or 2 <= v <= 4096 for v in (w, h))

    def _apply_custom(self):
        if self._lock_guard():
            return

        def num(s):
            s = s.strip()
            return int(s) if s.isdigit() else None
        w, h = num(self.w_var.get()), num(self.h_var.get())
        if not self._valid_custom(w, h):
            self._say("свій розмір: ціле число 2…4096 (ширина, висота або обидві)")
            return
        key = f"{w or ''}x{h or ''}"
        self._set_size(key)

    def _show_method(self, n):
        """у смузі заголовка секції - лише кодова назва поточного рівня"""
        vp = self._vp()
        n = vp.method if self.locked else n
        if vp.glyph:
            n = min(MAIN_LEVELS, n)                          # у символах шостого рівня нема
        lock = "  ·  " + tr("ЗАМОК") if self.locked else ""
        pos = f"  {n}/{MAIN_LEVELS}" if n <= MAIN_LEVELS else ""                    # IRNM1 - гілка: місця на шкалі «n із 5» не має
        self.q_title.config(text=f"{tr('Якість').upper()}  ·  {tr(METHODS[n]).upper()}{pos}{lock}")
        self.m_info.config(text=tr((GLYPH_INFO if vp.glyph else METHOD_INFO)[n]))

    def _on_method_drag(self, v):
        self._show_method(int(round(v)))

    def _show_dens(self):
        """підпис над повзунком щільності: скільки знаків і яка клітинка при поточному розмірі результату"""
        from . import glyph
        d = glyph.dens_of(self._vp().gdens)
        info = self.player.info
        txt = tr("Щільність знаків") + f"  {d}/{glyph.DENS_N}"                # два короткі рядки: довгий в одному не вміщається в панель
        if info:
            w, h = self.player.size
            rows, cols = glyph.grid_for(w, h, d)
            cw, ch = glyph.cell_for(w, h, d)
            txt += "\n" + tr("{c}×{r} · клітинка {cw}×{ch} px", c=cols, r=rows, cw=cw, ch=ch)
        self.dens_lbl.config(text=txt)

    def _apply_gdens(self, n):
        if self._lock_guard():
            return
        n = min(3, max(1, int(n)))
        self.dens_bar.set(n)
        if n != self.params.gdens:
            self.params.gdens = n
            self._reparam()                                   # крупніший знак може підняти й розмір результату (мінімум - одна клітинка): плеєр уже знає новий
            self._show_dens()
            self._update_size_label()
            self._note_min_size()
            self._save_soon()

    def _apply_method(self, n):
        if self._lock_guard():
            return
        self._branch_back = False
        self._show_method(n)
        if n != self.params.method:
            self.params.method = n
            self.m_slider.set(n)
            if n == BRANCH:
                self._neuro_start()
            self._reparam()
            self._save_soon()

    def _neuro_start(self):
        """IRNM1 вибрано: у фоні вибираємо, на чому рахувати (кілька секунд на першому разі, далі з пам'яті); немає моделі чи onnxruntime - кажемо одразу"""
        try:
            from . import student
            from .student import runtime as sr
            if sr.mode() == "off" or not sr.files_ok() or sr._ort() is None:
                self._say("IRNM1 тут недоступна (немає нейромережі чи onnxruntime): рахується як «Гравюра»")
            else:
                student.start_background()
        except Exception:  # noqa: BLE001
            pass

    def _reparam(self):
        """параметри кадру змінились (рівень або чорнило/символи): плеєр рахує наново, розмір «Авто» підбирається під нову вартість"""
        if self.ready:
            auto = self.size_var.get() == AUTO and self.player.kind != "image"
            if auto:
                self.player.hold(True)
            self.player.set_params(self.params)
            if auto:
                self._auto_resolve(keep=True)

    def _set_glyph(self, v):
        if self._lock_guard():
            return
        v = bool(v)
        if v == self.params.glyph:
            return
        self.params.glyph = v
        self.btn_ink.set_on(not v)
        self.btn_sym.set_on(v)
        if v and self.params.method > MAIN_LEVELS:                  # у символах п'ять рівнів: IRNM1 відкладається й повертається з чорнилом
            self._branch_back = True
            self.params.method = MAIN_LEVELS
        elif not v and self._branch_back:
            self._branch_back = False
            self.params.method = BRANCH
        self.m_slider.set_branch(not v)
        self.m_slider.set(self.params.method)
        if v:
            self.dens_box.pack(fill="x", after=self.m_info)
        else:
            self.dens_box.pack_forget()
        self._show_method(self.params.method)
        if self.kind == "image":                              # у списку форматів фото є .txt лише для символів
            vals = list(still_formats(v))
            self._fmt_keys = vals
            if self.fmt_var.get() not in vals:
                self.fmt_var.set(vals[0])
            self._sync_combos()
        self._reparam()
        self._show_dens()
        self._update_size_label()
        self._note_min_size()
        self._save_soon()

    def _style_combos(self):
        """списки й поля: усе в кольорах теми - рамка поля й випадного списку (за замовчуванням тема «clam» малює сіру), стрілка, виділення тексту,
        смужка прокрутки довгого списку. Викликається й при зміні палітри (тоді вже створені списки перефарбовуються напряму)"""
        style = ttk.Style()
        style.configure("Pix.TCombobox", fieldbackground=BG, background=BG, foreground=FG, arrowcolor=FG,
                        bordercolor=FG, lightcolor=BG, darkcolor=BG, selectbackground=FG, selectforeground=BG,
                        insertcolor=FG, padding=3)
        style.map("Pix.TCombobox", fieldbackground=[("readonly", BG), ("disabled", BG)],
                  foreground=[("readonly", FG), ("disabled", DIM)], background=[("active", BG), ("readonly", BG), ("disabled", BG)],
                  selectbackground=[("readonly", BG), ("!focus", BG)], selectforeground=[("readonly", FG), ("!focus", FG)],
                  arrowcolor=[("disabled", DIM), ("active", FG), ("readonly", FG)],
                  bordercolor=[("disabled", DIM), ("focus", FG), ("active", FG)],
                  lightcolor=[("focus", BG), ("active", BG)], darkcolor=[("focus", BG), ("active", BG)])
        style.configure("ComboboxPopdownFrame", background=BG, bordercolor=FG, lightcolor=BG, darkcolor=BG, relief="solid")
        for sb in ("Vertical.TScrollbar", "TScrollbar"):                     # смужка в довгому списку (власна смужка програми - PixScroll, це лише для випадних списків)
            style.configure(sb, background=BG, troughcolor=BG, bordercolor=FG, lightcolor=BG, darkcolor=BG, arrowcolor=FG, gripcount=0)
            style.map(sb, background=[("active", FG), ("pressed", FG)], arrowcolor=[("active", FG), ("pressed", BG)])
        self.option_add("*TCombobox*Listbox.background", BG)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", FG)
        self.option_add("*TCombobox*Listbox.selectForeground", BG)
        self.option_add("*TCombobox*Listbox.font", FONT_S)
        for cb in (getattr(self, "fmt_cb", None), getattr(self, "q_cb", None), getattr(self, "fps_cb", None)):   # списки, що вже створені
            if cb is not None:
                try:
                    pop = str(self.tk.call("ttk::combobox::PopdownWindow", cb))
                    self.tk.call(pop + ".f.l", "configure", "-background", BG, "-foreground", FG,
                                 "-selectbackground", FG, "-selectforeground", BG, "-highlightthickness", 0, "-borderwidth", 0)
                    self.tk.call(pop, "configure", "-background", BG, "-borderwidth", 0)
                except Exception:  # noqa: BLE001
                    pass

    def _set_theme_globals(self, key):
        """кольори вікна з ключа палітри (з «~» - вікно теж інвертоване: тло і текст міняються місцями)"""
        global BG, PANEL, EDGE, FG, DIM
        t = palette.theme(key)
        BG, PANEL, EDGE, FG, DIM = t["BG"], t["PANEL"], t["FG"], t["FG"], t["DIM"]

    def _apply_theme(self, key):
        """миттєво перефарбувати всю програму: старі кольори віджетів -> нові"""
        o_bg, o_fg, o_dim = BG, FG, DIM
        self._set_theme_globals(key)
        if (o_bg, o_fg, o_dim) == (BG, FG, DIM):
            return
        _rgb = self.winfo_rgb
        _rc = {}

        def rgb(c):                                         # один winfo_rgb на колір, а не на кожен параметр кожного віджета (було ≈700 викликів Tcl на крок палітри)
            v = _rc.get(c)
            if v is None:
                v = _rc[c] = _rgb(c)
            return v
        mp = {rgb(o_bg): BG, rgb(o_fg): FG, rgb(o_dim): DIM}
        # лише канонічні назви параметрів: «bg» і «background» - один параметр, і друге читання бачило б уже НОВИЙ колір та
        # перефарбовувало б його вдруге (при «навпаки» тло й текст міняються місцями: чорне -> біле -> знову чорне)
        opts = ("background", "foreground", "highlightbackground", "highlightcolor", "insertbackground",
                "activebackground", "activeforeground", "selectbackground", "selectforeground", "disabledforeground",
                "troughcolor", "disabledbackground", "readonlybackground")

        def walk(w):
            if not isinstance(w, ttk.Widget):
                have = _KEYS.get(type(w))                   # набір параметрів однаковий для всього класу: один keys() на клас, а не на кожен із ≈140 віджетів
                if have is None:
                    try:
                        have = frozenset(w.keys())
                    except Exception:  # noqa: BLE001
                        have = frozenset()
                    if type(w) is not tk.Toplevel:
                        _KEYS[type(w)] = have
                new = {}
                for o in opts:                              # спершу читаємо ВСІ старі кольори, потім ставимо нові разом
                    if o in have:
                        try:
                            nv = mp.get(rgb(str(w.cget(o))))
                        except Exception:  # noqa: BLE001
                            continue
                        if nv:
                            new[o] = nv
                if new:
                    try:
                        w.configure(**new)
                    except tk.TclError:
                        pass
            for c in w.winfo_children():
                walk(c)

        walk(self)                                           # (кореневе вікно теж проходить через walk: окремо його не ставимо)
        old_imgs = self.imgs
        self.imgs = {n: _img(n + "_w") for n in old_imgs}
        self.logo, self.mascot = _img("logo_w"), _img("mascot_w")
        self.logo2, self.mascot2 = _zoom2(self.logo), _zoom2(self.mascot)
        self.dither, self.film = _img("dither_light"), _img("film_tile")
        by_id = {id(v): k for k, v in old_imgs.items()}

        def refresh(w):
            if isinstance(w, PixButton) and w.im is not None and id(w.im) in by_id:
                w.im = self.imgs[by_id[id(w.im)]]
            if isinstance(w, (PixButton, PixSlider, PixProgress, QualityBar, PixScroll)):
                w.draw()
            for c in w.winfo_children():
                refresh(c)

        refresh(self)
        for lbl, im in ((getattr(self, "_lbl_mascot", None), self.mascot), (getattr(self, "_lbl_logo", None), self.logo),
                        (getattr(self, "_foot_mascot", None), self.mascot2), (getattr(self, "_foot_logo", None), self.logo2)):
            if lbl is not None and im is not None:
                lbl.configure(image=im)
        for st in self._strips:
            st.redraw()
        self._style_combos()
        self._draw_swatch()
        self.update_idletasks()                          # під час рендеру черга подій зайнята: без цього частина кнопок домальовувалась пізніше за решту

    def _pal_key(self) -> str:
        """ключ поточної палітри з урахуванням «навпаки» - саме його приймають перегляд і всі експорти"""
        return palette.key(self.pal_var.get(), self.pal_inv)

    def _step_palette(self, d):
        i = palette.NAMES.index(self.pal_var.get()) if self.pal_var.get() in palette.NAMES else 0
        self.pal_var.set(palette.NAMES[(i + d) % len(palette.NAMES)])
        self._set_palette()

    def _toggle_inv(self):
        """туш і папір міняються місцями: лише перефарбування - нічого не перераховується"""
        self.pal_inv = not self.pal_inv
        self.btn_pinv.set_on(self.pal_inv)
        self._set_palette()

    def _draw_swatch(self):
        """назва + два квадратики кольорів (ліворуч туш, праворуч папір), у порядку «навпаки», якщо воно ввімкнене"""
        self.pal_name.config(text=tr(self.pal_var.get()))
        sw = getattr(self, "pal_sw", None)
        if sw is None:
            return
        sw.delete("all")
        sw.configure(bg=BG)
        ink, paper = palette.colors(self._pal_key())
        hx = lambda c: "#%02x%02x%02x" % c
        sw.create_rectangle(0, 0, 14, 17, fill=hx(ink), outline=FG)
        sw.create_rectangle(15, 0, 29, 17, fill=hx(paper), outline=FG)

    def _set_palette(self):
        self._apply_theme(self._pal_key())
        self._draw_swatch()
        now = time.perf_counter()
        if now - getattr(self, "_pal_t", 0.0) < 0.15:         # швидка серія кроків (стрілки з автоповтором, клацання): інтерфейс перефарбовується щоразу,
            if self._pal_job:                                 # а важкий кадр - раз після останнього кроку, інакше черга подій не встигає за рукою
                self.after_cancel(self._pal_job)
            self._pal_job = self.after(60, self._pal_redraw)
        else:
            self._redraw()
        self._pal_t = now
        self._save_soon()

    def _pal_redraw(self):
        self._pal_job = None
        self._redraw()

    def _save_soon(self):
        if self._save_job:
            self.after_cancel(self._save_job)
        self._save_job = self.after(500, self._save_now)

    def _save_now(self):
        self._save_job = None
        save_settings({"ui": dict(size=self.size_var.get(), fmt=self.fmt_var.get() if self.kind == "video" else self.ui.get("fmt", "MP4 (H.264)"),
                                  quality=self.q_var.get(), audio=self.btn_aud.on, palette=self.pal_var.get(), pal_inv=bool(self.pal_inv),
                                  method=self.params.method, glyph=bool(self.params.glyph), gdens3=int(self.params.gdens), volume=int(self.vol.get()), lang=i18n.lang(),
                                  guide=self.guide_sel, fps=self.fps_sel)})

    # ------------------------------------------------------------ мова, повідомлення, списки
    def _set_lang(self, code):
        if code == i18n.lang():
            return
        i18n.set_lang(code)
        self._apply_lang()
        self._save_soon()

    def _apply_lang(self, first=False):
        """мову змінено (або щойно задано): усі підписи, списки й тексти - поточною мовою"""
        for w, attr, key, upper, kw in self._loc:
            self._loc_set(w, attr, key, upper, kw)
        self.btn_ua.set_on(i18n.lang() == i18n.UK)
        self.btn_en.set_on(i18n.lang() == i18n.EN)
        self._show_method(self.params.method)
        self.m_slider.draw()
        self.dens_bar.draw()
        self._show_dens()
        self._sync_combos()
        self._draw_swatch()
        self._update_size_label()
        self._update_range()
        self._export_labels()
        self._refresh_title()
        self._fill_footer()
        self._say_again()
        for lbl in (self.stat_lbl, self.time_lbl, self.est_lbl):
            lbl._last_txt = None
        if not first:
            self._redraw()
            self._relayout()

    def _refresh_title(self):
        self._fit_title()

    def _fit_title(self):
        """ім'я відкритого файлу в шапці займає лише вільне місце між кнопками й «?»: що не влізло - скорочується в кінці
        імені, розширення лишається («дуже-довга-наз….mp4»), тож шапка не розпирає й нічого не налізає одне на одне"""
        lbl = self.title_lbl
        full, fg = (self._title_name, FG) if self.ready else (tr("— файл не відкрито —"), DIM)
        room = lbl.winfo_width() - 4
        txt = full
        if room > 24 and _tw(full) > room:
            base, ext = os.path.splitext(full)
            while base and _tw(base + "…" + ext) > room:
                base = base[:-1]
            txt = base + "…" + ext
        if lbl.cget("text") != txt:
            lbl.config(text=txt)
        lbl.config(fg=fg)

    def _say(self, key="", done=False, **kw):
        """повідомлення в нижньому блоці правої панелі (прогрес, результат, помилки). key - ключ перекладу, kw - підстановки:
        повідомлення пам'ятається так, щоб мову можна було змінити на ходу"""
        self._prog_msg, self._prog_done = (key, kw), done
        self.prog_lbl.config(text=tr(key, **kw) if key else "")

    def _say_again(self):
        key, kw = self._prog_msg if self._prog_msg else ("", {})
        self.prog_lbl.config(text=tr(key, **kw) if key else "")

    @staticmethod
    def _short_path(p, n=36):
        """«…\\тека\\файл.mp4» - щоб шлях влазив у вузьку панель, а ім'я файлу лишалось цілим"""
        p = p.rstrip("\\/")
        name, d = os.path.basename(p), os.path.basename(os.path.dirname(p))
        s = os.path.join(d, name) if d else name
        if os.path.dirname(os.path.dirname(p)) not in ("", os.path.dirname(p)):
            s = "…" + os.sep + s
        return s if len(s) <= n else "…" + s[-(n - 1):]

    def _export_labels(self):
        t = tr({"video": "Зберегти…", "gif": "Зберегти GIF", "image": "Зберегти фото"}[self.kind])
        for b in (self.btn_export, self.btn_export2):
            b.text = t
            b.draw()
        self.btn_lock.text = tr("Звичайний режим") if self.locked else tr("Порахувати все")
        self.btn_lock.draw()

    def _sync_combos(self):
        """у списках формату й якості - переклад; а в налаштуваннях і в експорті лишаються українські ключі"""
        keys = self._fmt_keys
        self.fmt_cb.config(values=[tr(k) for k in keys])
        if self.fmt_var.get() in keys:
            self.fmt_cb.current(keys.index(self.fmt_var.get()))
        qk = list(QUALITY)
        self.q_cb.config(values=[tr(k) for k in qk])
        if self.q_var.get() in qk:
            self.q_cb.current(qk.index(self.q_var.get()))
        self._sync_fps()

    def _src_fps(self) -> float:
        try:
            return float(self.player.fps or 0) if self.kind == "video" else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def _fps_limit(self):
        """ліміт кадрів на секунду для експорту: число або None (як в оригіналі). Діє, лише якщо він нижчий за fps цього відео"""
        src = self._src_fps()
        return self.fps_sel if src and 1.0 <= self.fps_sel < src - 0.5 else None

    @staticmethod
    def _fps_name(v) -> str:
        return tr("Кадри: {f} к/с", f=f"{v:g}")

    def _sync_fps(self):
        """список «кадрів на секунду»: як в оригіналі й стандартні значення нижче за частоту відео; вибір пам'ятається й для інших відео"""
        src = self._src_fps()
        self._fps_vals = [0] + fps_options(src)
        self._fps_orig = tr("Кадри: як в оригіналі ({f:g} к/с)", f=round(src, 3)) if src else tr("Кадри: як в оригіналі")
        self.fps_cb.config(values=[self._fps_orig] + [self._fps_name(v) for v in self._fps_vals[1:]])
        self._show_fps()
        self._apply_fps_limit()

    def _apply_fps_limit(self):
        """ліміт кадрів на секунду діє й на перегляд: плеєр рахує в буфер лише кадри, що потраплять у відео (не весь потік джерела)"""
        try:
            self.player.set_fps_limit(self._fps_limit())
        except Exception:  # noqa: BLE001
            _log.get().exception("fps limit")

    def _show_fps(self):
        """у полі - ліміт, що діє зараз: «як в оригіналі», пункт списку чи вписане своє число"""
        lim = self._fps_limit()
        if lim and lim in self._fps_vals[1:]:
            self.fps_cb.current(self._fps_vals.index(lim))
        elif lim:
            self.fps_cb.set(self._fps_name(lim))
        else:
            self.fps_cb.current(0)
        self._fps_shown = self.fps_cb.get()

    def _on_fps_pick(self, e=None):
        i = self.fps_cb.current()
        if 0 <= i < len(self._fps_vals):
            self.fps_sel = float(self._fps_vals[i])
            self._fps_shown = self.fps_cb.get()
            self._apply_fps_limit()
            self._save_soon()

    def _fps_click(self, e):
        """клік у текст поля: курсор у поле й виділено лише значення після двокрапки (його й замінюють набором чи вставкою); стрілка списку - як звичайно"""
        cb = self.fps_cb
        if "arrow" in str(cb.identify(e.x, e.y)):
            return None
        cb.focus_set()
        cb.after(1, lambda: _select_value(cb))
        return "break"

    def _on_fps_typed(self, e=None):
        """Enter чи вихід з поля: вписане число стає лімітом (не нижче 1; від fps відео й вище чи 0 - як в оригіналі; не число - лишається попереднє).
        Нічого не вписано (текст той самий, що ми показали) - вибір, який пам'ятається для інших відео, не чіпаємо"""
        src = self._src_fps()
        text = self.fps_cb.get()
        if src and text != self._fps_shown:
            v = parse_fps_limit(text, src)
            if v is not None and v != self.fps_sel:
                self.fps_sel = v
                self._apply_fps_limit()
                self._save_soon()
        if src:
            self._show_fps()
        if e is not None and getattr(e, "keysym", "") in ("Return", "KP_Enter"):
            self.fps_cb.selection_clear()
            self.focus_set()

    def _on_fmt_pick(self, e=None):
        i = self.fmt_cb.current()
        if 0 <= i < len(self._fmt_keys):
            self.fmt_var.set(self._fmt_keys[i])
            self._save_soon()

    def _on_q_pick(self, e=None):
        qk = list(QUALITY)
        i = self.q_cb.current()
        if 0 <= i < len(qk):
            self.q_var.set(qk[i])
            self._save_soon()

    def _relayout(self):
        """вміст міг змінити висоту (інший тип файлу, мова, довший текст): перерахувати обидві прокрутки"""
        self.sstack.layout()
        self.pstack.layout()

    # ------------------------------------------------------------ відео
    def _layout_bottom(self, kind):
        """нижня панель: транспорт, звук, цикл, кадр->фото, час - усе в природну ширину; уся решта місця справа - рядок стану"""
        for w in (self.tr, self.btn_snd, self.vol, self.btn_loop, self.btn_photo, self.time_lbl, self.stat_lbl):
            w.pack_forget()
        if kind != "image":
            self.tr.pack(side="left")
            if kind == "video":
                self.btn_snd.pack(side="left", padx=(GG, 0))
                self.vol.pack(side="left", padx=(G, 0))                # повзунок гучності «прикріплений» до кнопки звуку
            self.btn_loop.pack(side="left", padx=(GG, 0))
            self.btn_photo.pack(side="left", padx=(GG, 0))
            if not self.seek.winfo_ismapped():
                self.seek.pack(fill="x", pady=(0, M), before=self.bot_row)
        else:
            self.seek.pack_forget()
        self.time_lbl.pack(side="left", padx=(0 if kind == "image" else GG, 0))     # фото: ряд починається врівень з краєм поля
        self.stat_lbl.pack(side="left", fill="x", expand=True, padx=(M, 0))         # останнім: бере лише те місце, що лишилось

    def _fit_status(self, segs, sep=" · "):
        """segs - [(важливість, текст)] у порядку показу (менше число = важливіше). Поки рядок не влазить у ширину підпису,
        прибираємо найменш важливу частину: час, прапорці й кнопки ніколи не затискаються довгим текстом стану"""
        avail = self.stat_lbl.winfo_width() - 8
        items = list(segs)
        self._stat_full = sep.join(t for _, t in items)
        while items:
            txt = sep.join(t for _, t in items)
            if self._f_stat.measure(txt) <= avail:
                return txt
            items.pop(max(range(len(items)), key=lambda i: items[i][0]))
        return ""

    def _apply_kind(self, kind):
        """інтерфейс під тип файлу: відео - повний, GIF - без параметрів кодека, фото - без часової шкали"""
        self.kind = kind
        if kind == "video":
            vals = list(FORMATS)
        elif kind == "gif":
            vals = list(GIF_FORMATS)
        else:
            vals = list(still_formats(self.params.glyph))
        self._fmt_keys = vals
        self.fmt_var.set(self.ui.get("fmt", vals[0]) if kind == "video" and self.ui.get("fmt") in vals else vals[0])
        self._sync_combos()
        if kind == "video":
            self.q_box.pack(side="left", padx=(G, 0))
            self.row2.pack(fill="x", pady=(M, 0), before=self.btn_export2)
            self.row_fps.pack(fill="x", pady=(M, 0), before=self.btn_export2)
            self.btn_lock.pack(fill="x", pady=(M, 0), after=self.btn_export2)
        else:
            self.q_box.pack_forget()
            self.row2.pack_forget()
            self.row_fps.pack_forget()
            self.btn_lock.pack_forget()
        self.seek.marks = [0, max(1, self.player.n - 1)] if kind == "video" and self.player.n else None
        self.seek.mark_cb = self._on_mark
        self.seek.draw()
        self._update_range()
        self._layout_bottom(kind)
        self._export_labels()
        self.after_idle(self.pstack.layout)

    def bring_front(self, path=""):
        """інший запуск: показати це вікно й відкрити файл, якщо його передали"""
        try:
            if self.state() == "iconic":
                self.deiconify()
            self.lift()
            self.attributes("-topmost", True)               # Windows не дає програмі просто вийти на передній план: на мить -topmost
            self.focus_force()
        except Exception:  # noqa: BLE001
            pass
        self.after(300, self._untop)                        # і ЗАВЖДИ знімаємо, навіть якщо вище щось пішло не так
        if path and os.path.isfile(path) and not self.exporting:
            self._open(path)

    def _untop(self):
        """вікно програми не повинно лишатися «поверх усіх»: інакше інші вікна відкриваються під ним"""
        try:
            if not (self._fs or self._wfs) and str(self.attributes("-topmost")).lower() in ("1", "true", "yes"):
                self.attributes("-topmost", False)
        except tk.TclError:
            pass

    def _open(self, path=None, start=0, size=None, seed=None):
        """start - кадр, з якого стартує буфер; size - готовий ключ розміру (тоді «Авто» не підбирається знову); seed - див. Player.open"""
        exts = tuple(VIDEO_EXT) + IMAGE_EXT + GIF_EXT
        path = path or filedialog.askopenfilename(
            parent=self, title=tr("Відкрити відео, GIF або фото"), initialdir=self._last_dir or None,
            filetypes=[(tr("Усі медіа"), " ".join("*" + e for e in exts)), (tr("Відео"), " ".join("*" + e for e in VIDEO_EXT)),
                       ("GIF", "*.gif"), (tr("Фото"), " ".join("*" + e for e in IMAGE_EXT)), (tr("Усі файли"), "*.*")])
        if not path:
            return
        try:
            want = self.size_var.get()
            info = self.player.open(path, size or ("240p" if want == AUTO else want), start=start, seed=seed)
        except Exception as e:  # noqa: BLE001
            _log.get().exception("не вдалося відкрити %s", path)
            messagebox.showerror("InkReel", tr("Не вдалося відкрити файл:\n{e}\n\nДеталі: {p}", e=e, p=_log.shown_path()), parent=self)
            return
        self.ready = True
        self._show_app()
        self._last_mask = self._last_bgr = None
        self._zoom, self._vc = None, view.CENTER           # новий файл - знову «вписати»
        self._lock_p = None
        self._apply_kind(info["kind"])
        self._title_name = os.path.basename(path)
        self._refresh_title()
        self.title(f"{self.TITLE} — {self._title_name}")
        self.seek.lo, self.seek.hi = 0, max(1, info["frames"] - 1)
        self.seek.ranges = []
        self.seek.set(min(max(0, int(start)), max(0, info["frames"] - 1)))
        self._reset_range()
        if self.player.baked or size:
            self._auto_tok = getattr(self, "_auto_tok", 0) + 1       # запізнілий автовибір попереднього файлу не повинен чіпати цей
        elif want == AUTO:
            self._auto_resolve()
        self._sync_lock()
        self._update_size_label()
        self.btn_play.set_icon("play")
        self._say("⚠ {note}", note=tr(info["note"])) if info.get("note") else self._say()
        self._remember_dir(path)
        self.after_idle(self._relayout)

    # ------------------------------------------------------------ режим замка (файл .ikr: кадри готові, нічого не рахується)
    @property
    def locked(self) -> bool:
        return bool(getattr(self, "ready", False) and getattr(self.player, "baked", False))

    def _vp(self):
        """налаштування, які ПОКАЗУЄ інтерфейс: у замку - ті, з якими порахований файл, інакше поточні"""
        if not self.locked:
            return self.params
        if getattr(self, "_lock_p", None) is None:
            self._lock_p = Params.from_dict(dict(self.player.baked_meta.get("params") or {}))
        return self._lock_p

    def _lock_guard(self) -> bool:
        """True - зараз замок, зміна відхилена (з підказкою)"""
        if self.locked:
            self._say("Режим замка: кадри вже пораховані. Щоб змінити налаштування, натисни «Звичайний режим»")
            return True
        return False

    def _sync_lock(self):
        """вигляд регуляторів під режим: в замку якість, чорнило/символи й розмір вимкнені й показують те, з чим порахований файл"""
        lk = self.locked
        vp = self._vp()
        for w in (self.m_slider, self.dens_bar, self.btn_ink, self.btn_sym, self.btn_ok, *self.size_btns.values()):
            w.set_enabled(not lk)
        for e in (self.ent_w, self.ent_h):
            e.config(state="disabled" if lk else "normal")
        self.m_slider.set(vp.method)
        self.dens_bar.set(vp.gdens)
        self.btn_ink.set_on(not vp.glyph)
        self.btn_sym.set_on(bool(vp.glyph))
        if vp.glyph:
            self.dens_box.pack(fill="x", after=self.m_info)
        else:
            self.dens_box.pack_forget()
        for k, b in self.size_btns.items():
            b.set_on(False if lk else k == self.size_var.get())
        self._show_method(vp.method)
        self._show_dens()
        self._export_labels()
        self.after_idle(self._relayout)

    def _lock_toggle(self):
        if not self.ready or self.exporting or self.kind != "video":
            return
        if self.locked:
            self._unfreeze()
        else:
            self._export(freeze=True)

    def _enter_lock(self, path):
        """експорт .ikr закінчено: відкрити його в режимі замка"""
        try:
            self._open(path)
        except Exception:  # noqa: BLE001
            _log.get().exception("не вдалося відкрити готовий файл .ikr")
            return
        if self.locked:
            self._say("Готово: усе відео пораховане, нічого не перераховується.\n«Звичайний режим» — повернутись до оригіналу")

    def _frozen_size(self, sm, size):
        """ключ розміру, що дає ТОЙ САМИЙ розмір, яким порахований файл: тоді розморожене відео не треба перераховувати й «Авто» не підбирається знов"""
        size = (int(size[0]), int(size[1]))
        sw, sh = int(sm.get("width") or 0), int(sm.get("height") or 0)
        cur = self.size_var.get()
        cands = []
        if sw and sh:                                            # пресети рахуються від розміру оригіналу
            cands = [self._auto_size if cur == AUTO else cur] + [k for k in self.size_btns if k != AUTO]
        cands.append(f"{size[0]}x{size[1]}")                     # крайній випадок: цей самий розмір числами
        for k in cands:
            if not k:
                continue
            try:
                if out_size(sw or size[0], sh or size[1], k) == size:
                    return k
            except Exception:  # noqa: BLE001
                continue
        return None

    @staticmethod
    def _same_original(sm, path) -> bool:
        """оригінал той самий, з якого порахований файл (розмір і час зміни з заголовка .ikr): тільки тоді його кадри можна брати замість розрахунку"""
        try:
            st = os.stat(path)
        except OSError:
            return False
        return int(sm.get("size") or -1) == st.st_size and abs(float(sm.get("mtime") or 0) - st.st_mtime) < 2.0

    def _unfreeze(self):
        """назад до звичайного режиму: відкрити оригінал, повернути налаштування, з якими порахований файл, і ту саму мить.
        Пораховані кадри не викидаються: плеєр бере їх із самого файлу .ikr, тож буфер не рахується заново з нуля"""
        m = dict(self.player.baked_meta or {})
        sm = dict(m.get("src") or {})
        ikr = self.player.path
        path = sm.get("path") or ""
        if not (path and os.path.isfile(path)):
            if not messagebox.askokcancel(
                    "InkReel", tr("Оригіналу «{n}» поряд немає, а файл .ikr зберігає лише готовий результат, без вихідного відео.\n\n"
                                  "Щоб повернутись до звичайного режиму й міняти налаштування, потрібен оригінал: кадри доведеться порахувати знову.\n"
                                  "Вказати оригінал?", n=sm.get("name") or "?"), parent=self):
                return
            exts = tuple(VIDEO_EXT)
            path = filedialog.askopenfilename(
                parent=self, title=tr("Знайти оригінал: {n}", n=sm.get("name") or "?"), initialdir=os.path.dirname(path) if path else self._last_dir or None,
                filetypes=[(tr("Відео"), " ".join("*" + e for e in exts if e != ".ikr")), (tr("Усі файли"), "*.*")])
            if not path:
                return
        pl = self.player
        i = int(pl.position())
        k = (float(sm.get("fps") or 0) / float(pl.fps)) if sm.get("fps") and pl.fps else 1.0
        base = int((m.get("range") or [0, 0])[0] or 0)
        idx = base + int(round(i * k))
        pd = dict(m.get("params") or {})
        for key in ("method", "glyph", "gdens"):                # налаштування файлу стають поточними (інакше «розморожене» виглядало б інакше)
            if key in pd:
                setattr(self.params, key, type(getattr(self.params, key))(pd[key]))
        self.m_slider.set(self.params.method)
        self.dens_bar.set(self.params.gdens)
        self.btn_ink.set_on(not self.params.glyph)
        self.btn_sym.set_on(bool(self.params.glyph))
        key = self._frozen_size(sm, pl.size)                     # розмір лишається таким, яким порахований файл
        if key and not (self.size_var.get() == AUTO and self._auto_size == key):
            self.size_var.set(key)
            self._auto_size = None
            if key not in self.size_btns:                        # свій розмір: показати його в полях
                cw, ch = parse_size(key)
                self.w_var.set(str(cw or ""))
                self.h_var.set(str(ch or ""))
            self._refresh_btns()
        seed = None
        if key and abs(k - 1.0) < 1e-3 and self._same_original(sm, path) and Params.from_dict(pd).to_dict() == self.params.to_dict():
            seed = dict(path=ikr, base=base)                     # налаштування, розмір і оригінал ті самі - кадри з файлу точно такі, які порахував би плеєр
        self._open(path, start=idx, size=key, seed=seed)
        if self.locked:                                          # відкрилось знов .ikr (мало б не статись)
            return
        self._show_method(self.params.method)
        self._show_dens()
        self._update_size_label()
        self._save_soon()
        self._say("Звичайний режим: оригінал відкрито, налаштування файлу повернуто")

    def _hook_drop(self):
        """перетягування файлу у вікно: власний обробник WM_DROPFILES (Unicode), запасний - пакет windnd"""
        def got(ps):
            ps = [p for p in ps if os.path.isfile(p)]
            if not ps:
                self.q.put(lambda: self._say("Перетягни файл (відео, GIF або фото), а не теку"))
            elif self.exporting:
                self.q.put(lambda: self._say("Іде експорт — спершу дочекайся завершення або натисни Стоп"))
            else:
                self.q.put(lambda: self._open(ps[0]))
                if len(ps) > 1:
                    self.q.put(lambda n=len(ps): self._say("Відкрито перший із {n}", n=n))
        from . import dnd
        if dnd.hook(self, got):
            return
        try:
            import windnd

            def drop(files):
                out = []
                for f in files:
                    if isinstance(f, bytes):
                        for enc in ("utf-8", "mbcs"):
                            try:
                                out.append(f.decode(enc))
                                break
                            except Exception:  # noqa: BLE001
                                continue
                    else:
                        out.append(str(f))
                got(out)
            windnd.hook_dropfiles(self, func=drop)
        except Exception:  # noqa: BLE001
            _log.get().info("перетягування файлів недоступне")

    def _set_icon(self):
        """значок вікна: .ico на Windows (заголовок, панель завдань, Alt+Tab), PNG - деінде; замість пера Tk"""
        try:
            ico = os.path.join(ASSETS, "inkreel.ico")
            if os.name == "nt" and os.path.isfile(ico):
                self.iconbitmap(default=ico)
            self._icon_img = tk.PhotoImage(file=os.path.join(ASSETS, "app_icon.png"))
            self.iconphoto(True, self._icon_img)
        except Exception:  # noqa: BLE001
            pass

    def _src_dir(self):
        return self._out_dir or os.path.dirname(self.player.path or "") or None

    def _remember_dir(self, path):
        self._last_dir = os.path.dirname(os.path.abspath(path))
        self._out_dir = None

    def _toggle_loop(self):
        self.player.loop = self.btn_loop.on

    def _remember_window(self):
        """запам'ятати, яким було вікно, щоб після виходу з повного екрана повернути нормальний розмір"""
        try:
            self._pre_fs = (self.state(), self.winfo_width(), self.winfo_height(), self.winfo_x(), self.winfo_y())
        except tk.TclError:
            self._pre_fs = None

    def _apply_fullscreen(self, was):
        """виставити стан вікна за _fs/_wfs: рамка зникає, якщо діє будь-який із двох режимів; вийшли з обох - нормальний розмір"""
        full = self._fs or self._wfs
        self.attributes("-fullscreen", full)
        if not full:
            self._untop()                                    # вийшли з повного екрана - ніякого «поверх усіх» не лишається
            if was:
                self.after(60, self._leave_fullscreen_size)
        self._apply_side()
        self.pstack.set_enabled(not self._fs)                # «кіно» - лише картинка: ні смужки, ні футера
        self.after(50, self._redraw)

    def _fullscreen(self, v=None):
        """«кіно» (F, подвійний клік): на весь екран лише картинка"""
        new = (not self._fs) if v is None else v
        if new == self._fs:
            return
        was = self._fs or self._wfs
        if new and not was:
            self._remember_window()
        self._fs = new
        self._apply_fullscreen(was)

    def _window_fullscreen(self, v=None):
        """F11: програма розтягується на весь екран як є, з усім інтерфейсом; із «кіно» F11 виходить повністю"""
        if self._fs and v is None:
            was = True
            self._fs = self._wfs = False
            self._apply_fullscreen(was)
            return
        new = (not self._wfs) if v is None else v
        if new == self._wfs:
            return
        was = self._fs or self._wfs
        if new and not was:
            self._remember_window()
        self._wfs = new
        self._side_open = False                              # у F11 панель спершу схована; кнопка справа її висуває
        self._apply_fullscreen(was)

    def _on_escape(self, e=None):
        if self._fs:
            self._fullscreen(False)
        elif self._wfs:
            self._window_fullscreen(False)

    def _apply_side(self):
        """права панель: у звичайному вікні завжди на місці; в F11 схована, а вузька кнопка на правому краю висуває її (і ховає назад);
        у «кіно» немає ні панелі, ні кнопки"""
        wfs = self._wfs and not self._fs
        show = not self._fs and (not self._wfs or self._side_open)
        try:
            if show:
                if not self.side.winfo_ismapped():
                    self.side.pack(side="left", fill="y", padx=(M, 0))
            else:
                self.side.pack_forget()
            if wfs:
                self.edge.set_icon("right" if self._side_open else "left")
                if not self.edge.winfo_ismapped():
                    self.edge.pack(side="right", fill="y", padx=(M, 0))
            else:
                self.edge.pack_forget()
        except (tk.TclError, AttributeError):
            pass

    def _toggle_side(self):
        self._side_open = not self._side_open
        with self._no_paint():                              # панель зникає за один перемальовок, без проміжного кадру (на Windows там мигало світлим)
            self._apply_side()
            self.update_idletasks()
        self.after(30, self._redraw)

    @contextlib.contextmanager
    def _no_paint(self):
        """Windows: на час зміни розкладки вимкнути перемальовування вікна (WM_SETREDRAW) і після неї перемалювати все одразу. Деінде нічого не робить"""
        hwnd = 0
        if os.name == "nt":
            try:
                import ctypes
                hwnd = int(self.winfo_id())
                ctypes.windll.user32.SendMessageW(hwnd, 0x000B, 0, 0)          # WM_SETREDRAW: вимкнути
            except Exception:  # noqa: BLE001
                hwnd = 0
        try:
            yield
        finally:
            if hwnd:
                try:
                    import ctypes
                    ctypes.windll.user32.SendMessageW(hwnd, 0x000B, 1, 0)      # увімкнути
                    ctypes.windll.user32.RedrawWindow(hwnd, None, None, 0x0001 | 0x0004 | 0x0080 | 0x0100)   # INVALIDATE | ERASE | ALLCHILDREN | UPDATENOW
                except Exception:  # noqa: BLE001
                    pass

    def _leave_fullscreen_size(self):
        """після виходу з повного екрана вікно не лишається розміром на весь екран (лише з рамкою): повертаємо розмір, який було до входу, а якщо він і так займав
        майже весь екран (або вікно було розгорнуте) - беремо звичайний розмір за розмірами монітора і ставимо по центру"""
        if self._fs or self._wfs:
            return
        try:
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            pre = getattr(self, "_pre_fs", None)
            if self.state() == "zoomed":
                self.state("normal")
            st, w, h, x, y = pre if pre else ("zoomed", 0, 0, 0, 0)
            mw, mh = self.minsize()
            if st == "zoomed" or w < 200 or h < 200 or w >= sw * 0.95 or h >= sh * 0.90:
                w, h = int(sw * 0.72), int(sh * 0.78)
                x, y = None, None
            w, h = max(mw or 1, min(w, sw)), max(mh or 1, min(h, sh))
            if x is None:
                x, y = (sw - w) // 2, max(0, (sh - h) // 2 - 20)
            self.geometry(f"{w}x{h}+{max(0, x)}+{max(0, y)}")
            self.after(50, self._redraw)
        except tk.TclError:
            pass

    @staticmethod
    def _letter_of(e) -> str:
        """латинська літера натиснутої клавіші в будь-якій розкладці. Keysym - якщо це латинська літера; для кирилиці - за
        фізичною клавішею (на Windows keycode = віртуальна клавіша A-Z, однакова в англійській, українській, російській
        розкладках) і за схожими літерами, які користувач натисне, побачивши підказку: «м» -> M, «і» -> I"""
        ks = e.keysym or ""
        if len(ks) == 1 and ks.isascii() and ks.isalpha():
            return ks.lower()
        alias = {"cyrillic_em": "m", "ukrainian_i": "i"}.get(ks.lower())
        if alias:
            return alias
        if sys.platform == "win32" and 65 <= e.keycode <= 90:
            return chr(e.keycode).lower()
        return ""

    def _on_footer(self) -> bool:
        """зараз на екрані футера (гайд, про програму), а не на екрані з відео: клавіші плеєра (пауза, кадри, гучність, фрагмент) тут не діють"""
        try:
            return self.pstack.tcur != 0 or self.pstack.cur != 0
        except Exception:  # noqa: BLE001
            return False

    def _on_letter(self, e):
        if self._on_footer() and self._letter_of(e) in ("m", "l", "i", "f"):
            return
        if e.state & 0x4 or e.state & (0x20000 if sys.platform == "win32" else 0x8) or self._editing():
            return                                          # Ctrl/Alt-комбінації й набір тексту - не наші; список «лише вибір» (формат, якість) після вибору лишається у фокусі, але літер не приймає - F/M/L/I мають працювати
        {"f": self._fullscreen, "m": self._key_mute, "l": self._key_loop, "i": self._toggle_inv}.get(self._letter_of(e), lambda: None)()

    def _key_loop(self):
        self.btn_loop.set_on(not self.btn_loop.on)
        self._toggle_loop()

    def _key_mute(self):
        if self.ready and self.kind == "video":
            self._mute()

    def _mute(self):
        m = not self.player.audio.muted
        if not m and self.vol.get() < 1:                       # вмикаємо звук при нульовій гучності - піднімемо до половини
            self.vol.set(50)
            self.player.set_volume(0.5, restart=True)
        self.player.set_muted(m)
        self._sync_snd_icon()
        self._save_soon()

    def _sync_snd_icon(self):
        self.btn_snd.set_icon("mute" if (self.player.audio.muted or self.vol.get() < 1) else "snd")

    def _on_volume(self, v):
        """повзунок гучності тягнуть: звук змінюється на льоту (sounddevice); рух повзунка вимикає «без звуку»"""
        v = int(round(v))
        self.player.set_volume(v / 100)
        if v > 0 and self.player.audio.muted:
            self.player.set_muted(False)
        self._sync_snd_icon()

    def _on_volume_end(self, v):
        self.player.set_volume(int(round(v)) / 100, restart=True)       # winsound: застосувати з перезапуском
        self._save_soon()

    def _paste_replace(self, e):
        """вставка в поле «ширина», «висота» чи «кадри»: вміст поля ЗАМІНЮЄТЬСЯ тим, що в буфері. У поля розміру «1920x1080» (х, ×, *) лягає одразу в обидва"""
        w = e.widget
        try:
            txt = self.clipboard_get()
        except tk.TclError:
            return "break"
        txt = (txt.strip().splitlines() or [""])[0].strip()
        if w in (getattr(self, "ent_w", None), getattr(self, "ent_h", None)):
            m = re.match(r"^(\d{1,5})\s*[xX\u0445\u0425\u00d7*]\s*(\d{1,5})$", txt)        # x, кирилична х, ×, *
            if m:
                self.w_var.set(m.group(1))
                self.h_var.set(m.group(2))
                return "break"
            txt = re.sub(r"\D", "", txt)[:5]
        try:
            w.delete(0, "end")
            w.insert(0, txt)
            w.icursor("end")
        except tk.TclError:
            pass
        return "break"

    def _ctrl_key(self, e):
        """Ctrl+C/V/X/A за фізичною клавішею (на Windows keycode = віртуальна клавіша A-Z у будь-якій розкладці); латинську розкладку Tk обробляє сам"""
        if sys.platform != "win32" or (e.keysym or "").lower() in ("c", "v", "x", "a"):
            return None
        ev = {67: "<<Copy>>", 86: "<<Paste>>", 88: "<<Cut>>"}.get(e.keycode)
        try:
            if ev:
                e.widget.event_generate(ev)
                return "break"
            if e.keycode == 65:
                w = e.widget
                if isinstance(w, tk.Text):
                    w.tag_add("sel", "1.0", "end-1c")
                elif hasattr(w, "selection_range"):
                    w.selection_range(0, "end")
                    w.icursor("end")
                return "break"
        except tk.TclError:
            pass
        return None

    def _click_release_focus(self, e):
        try:
            w = e.widget
            if isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Listbox, tk.Text)) or str(w).endswith("popdown.f.l") or "popdown" in str(w):
                return
            if self._typing():
                self.after_idle(self.focus_set)
        except (KeyError, tk.TclError):
            pass

    def _editing(self) -> bool:
        """фокус у полі, де друкують (рядок вводу чи список, що дозволяє вписати): пробіл, стрілки, Home тоді не керують плеєром.
        Список лише для вибору (формат, якість) сюди не входить: після вибору в ньому пробіл і стрілки й далі працюють із відео"""
        try:
            w = self.focus_get()
            if isinstance(w, ttk.Combobox):
                return str(w.cget("state")) != "readonly"
            return isinstance(w, (tk.Entry, ttk.Entry))
        except (KeyError, tk.TclError):
            return False

    def _typing(self):
        """фокус у списку/полі вводу: стрілки ↑/↓ належать йому, а не гучності"""
        try:
            return isinstance(self.focus_get(), (ttk.Combobox, tk.Entry, ttk.Entry))
        except (KeyError, tk.TclError):
            return False

    def _vol_step(self, d, key=False):
        if not self.ready or self.kind != "video" or (key and self._typing()):
            return
        v = min(100, max(0, int(self.vol.get()) + d))
        self.vol.set(v)
        self._on_volume(v)
        self._on_volume_end(v)

    def _save_frame(self):
        """поточний кадр (відео/GIF) -> окреме 1-бітне фото"""
        if self._last_mask is None:
            return
        base = os.path.splitext(os.path.basename(self.player.path))[0]
        types = [("PNG", "*.png"), ("BMP", "*.bmp"), ("TIFF", "*.tif"), ("WebP", "*.webp")]
        vp = self._vp()                                       # в замку - ті налаштування, з якими файл порахований
        if vp.glyph:
            types.append((tr("Текст (.txt, символи)"), "*.txt"))
        out = filedialog.asksaveasfilename(parent=self, title=tr("Зберегти кадр як фото"), initialdir=self._src_dir(), defaultextension=".png",
                                           initialfile=f"{base}_{int(self.player.position()) + 1:05d}.png", filetypes=types)
        if not out:
            return
        try:
            if os.path.splitext(out)[1].lower() == ".txt":
                from . import glyph
                mh, mw = self._last_mask.shape[:2]
                save_text(glyph.text_from_mask(self._last_mask, mw, mh, vp.gdens), out)
            else:
                save_still(self._last_mask, out, self._pal_key())
            self._last_out = out
            self._say("Кадр збережено:\n{p}\nклік — відкрити теку", done=True, p=self._short_path(out))
        except Exception as e:  # noqa: BLE001
            self._say("Помилка: {e}", e=e)

    def _toggle(self):
        if not self.ready:
            return
        if self.exporting:                       # під час рендеру ця ж кнопка ставить на паузу сам рендер
            self._toggle_export_pause()
            return
        self.player.toggle()
        self.btn_play.set_icon("pause" if self.player.playing else "play")

    def _toggle_export_pause(self):
        """пауза/продовження експорту відео: нових шматків пул не бере, процесор вільний; GIF і фото рахуються за мить - для них паузи нема"""
        if not self._pausable:
            return
        on = not self.pause_ev.is_set()
        (self.pause_ev.set if on else self.pause_ev.clear)()
        if self._clock is not None:
            self._clock.hold(on)
        self.btn_play.set_icon("play" if on else "pause")
        self._export_status()

    def _play_icon(self) -> str:
        """значок кнопки: під час рендеру - «пауза» (ідe) або «грати» (стоїть), інакше - за станом плеєра"""
        if self.exporting:
            return "play" if self.pause_ev.is_set() else "pause"
        return "pause" if self.player.playing else "play"

    def _stop(self):
        if not self.ready or self.exporting:
            return
        self.player.pause()
        self.player.seek(0)
        self.btn_play.set_icon("play")

    def _jump(self, sec):
        if self.ready and not self.exporting:
            self.player.seek(self.player.position() + sec * self.player.fps)

    def _step(self, d):
        if not self.ready or self.exporting:
            return
        self.player.step(d)
        self.btn_play.set_icon("play")

    def _range(self):
        """-> (початок, кінець-ексклюзивно) або None, якщо вибрано все відео"""
        m = self.seek.marks
        if not m or not self.player.n or (m[0] <= 0 and m[1] >= self.player.n - 1):
            return None
        return int(m[0]), int(m[1]) + 1

    def _update_range(self):
        r = self._range()
        fps = self.player.fps or 24.0
        if r is None:
            self.rng_lbl.config(text=tr("Фрагмент: усе відео\n[ ] — межі фрагмента"))
            self.btn_rng.set_enabled(False)
        else:
            a, b = r
            self.rng_lbl.config(text=tr("Фрагмент: {a} – {b}\n({d}, {n})", a=fmt_t(a / fps), b=fmt_t(b / fps), d=fmt_t((b - a) / fps),
                                        n=tn(b - a, "{n} кадр|{n} кадри|{n} кадрів")))
            self.btn_rng.set_enabled(True)

    def _on_mark(self, i, v):
        self._update_range()
        if self.ready:
            self.player.seek(int(v))                 # показати кадр під прапорцем

    def _reset_range(self):
        if self.seek.marks:
            self.seek.marks[:] = [0, max(1, self.player.n - 1)]
            self.seek.draw()
            self._update_range()

    def _set_mark_here(self, i):
        m = self.seek.marks
        if not m or not self.ready:
            return
        pos = int(self.player.position())
        if i == 0:
            m[0] = min(pos, m[1] - 1)
        else:
            m[1] = max(pos, m[0] + 1)
        self.seek.draw()
        self._update_range()

    SEEK_GAP = 0.04                                   # повзунок перемотування: не частіше за 25 переходів на секунду

    def _on_seek_drag(self, v):
        """мишка шле до тисячі подій на секунду, а кожен seek скасовує рендер, що ще йде: без обмеження жоден кадр не встигав домалюватись,
        доки повзунок тримають. Тепер найновіше значення віддається плеєру раз на SEEK_GAP, а решта подій лише запам'ятовується"""
        self._seek_hold = True
        self._seek_val = v
        if not self.ready:
            return
        wait = self.SEEK_GAP - (time.perf_counter() - self._seek_t)
        if wait <= 0 and self._seek_job is None:
            self._seek_apply()
        elif self._seek_job is None:
            self._seek_job = self.after(max(1, int(wait * 1000)), self._seek_apply)

    def _seek_apply(self):
        self._seek_job = None
        self._seek_t = time.perf_counter()
        if self.ready and self._seek_hold:
            self.player.seek(int(self._seek_val))

    def _on_seek_end(self, v):
        self._seek_hold = False
        if self._seek_job is not None:
            self.after_cancel(self._seek_job)
            self._seek_job = None
        if self.ready:
            self.player.seek(int(v))

    def _meter(self, shown, redraw_s, tick_s, gap_s, skipped):
        """лічильник плавності показу: раз на 10 с відтворення пише в журнал, скільки кадрів показано/пропущено, скільки коштує малювання кадра й яка найдовша пауза між кадрами"""
        m = self._pm
        now = time.perf_counter()
        if m is None or not self.ready or self.player.state != "playing":
            self._pm = None if self.player.state != "playing" else dict(t0=now, n=0, skip=0, rd=0.0, rdmax=0.0, tkmax=0.0, gapmax=0.0)
            if self._pm is None:
                self._pm_last = None                       # після паузи/перемотки різниця індексів кадрів - не пропуск
            return
        m["tkmax"] = max(m["tkmax"], tick_s)
        if shown:
            m["n"] += 1
            m["skip"] += skipped
            m["rd"] += redraw_s
            m["rdmax"] = max(m["rdmax"], redraw_s)
            m["gapmax"] = max(m["gapmax"], gap_s)
            m["pc"] = max(m.get("pc", 0.0), self._pc)
        if now - m["t0"] >= 10.0 and m["n"]:
            pl = self.player
            pa = self._pacc
            _log.get().info("показ: %d кадрів за %.0f с, пропущено %d (%.0f%%); укладання кадра на полотно в середньому %.1f мс, макс %.0f мс; найдовший такт інтерфейсу %.0f мс, найдовша пауза між кадрами %.0f мс; зменшення кадра у фоні до готовності макс %.0f мс; з них у середньому: PIL+вставка в Tk %.1f мс, решта полотна %.1f мс; картинка %s, полотно %dx%d; файл %.3f к/с, %dx%d, запас %.1f с",
                            m["n"], now - m["t0"], m["skip"], 100.0 * m["skip"] / max(1, m["n"] + m["skip"]), 1000 * m["rd"] / m["n"], 1000 * m["rdmax"],
                            1000 * m["tkmax"], 1000 * m["gapmax"], 1000 * m.get("pc", 0.0), 1000 * pa["img"] / max(1, pa["n"]), 1000 * pa["items"] / max(1, pa["n"]),
                            "x".join(str(v) for v in (pa["view"] or ())[:3]), *((pa["view"] or (0, 0, 0, 0, 0))[3:5]), pl.fps, pl.size[0], pl.size[1], (pl.buffered_end() - int(pl.position())) / max(pl.fps, 1e-6))
            self._pm = dict(t0=now, n=0, skip=0, rd=0.0, rdmax=0.0, tkmax=0.0, gapmax=0.0)
            self._pacc.update(img=0.0, items=0.0, n=0)

    def _tick(self):
        t0 = time.perf_counter()
        try:
            r = self.player.update() if self.ready else None
            shown = skipped = 0
            rd = gap = 0.0
            t1 = time.perf_counter()
            n0 = self._npaint
            if r:
                self._last_mask = r[1]
                self._fr_idx = int(r[0])
                self._frame_redraw()
            self._poll_compose()
            if self._npaint != n0:                           # кадр справді лягнув на полотно (у цьому такті)
                rd = time.perf_counter() - t1
                shown = 1
                prev = self._pm_last
                idx = self._fr_idx
                skipped = max(0, idx - prev[0] - 1) if prev and self.player.state == "playing" and idx > prev[0] else 0
                gap = t1 - prev[1] if prev else 0.0
                self._pm_last = (idx, t1)
            now = time.perf_counter()
            if now - self._last_lay_t > 0.3:
                self._last_lay_t = now
                rq = (self.side_in.winfo_reqheight(), self.foot.winfo_reqheight(), self.main.winfo_reqheight())
                if rq != self._lay_req:
                    self._lay_req = rq
                    self._relayout()
            if now - self._hud_t > 0.1:
                self._hud_t = now
                self._hud_update()
            if self.ready and now - self._last_info_t > 0.15:
                self._last_info_t = now
                pl = self.player
                pos = pl.position()
                if not self._seek_hold:
                    self.seek.set(pos)
                if pl.kind == "image":
                    self._txt(self.time_lbl, tr("ФОТО {w}×{h}", w=pl.info["width"], h=pl.info["height"]))
                else:
                    self._txt(self.time_lbl, f"{fmt_t(pl.time_of(pos))} / {fmt_t(pl.duration)}")
                self.btn_play.set_icon(self._play_icon())
                if now - self._last_rng_t > 0.5:             # важчі й рідше змінні підписи - не частіше, щоб не збивати ритм кадрів
                    self._last_rng_t = now
                    w = max(1, self.seek.winfo_width() - 16)
                    k = max(1.0, (pl.n - 1) / w)             # злити відрізки в межах пікселя
                    rs = []
                    for a, b in pl.buffered_ranges():
                        if rs and a - rs[-1][1] <= k:
                            rs[-1][1] = b
                        else:
                            rs.append([a, b])
                    self.seek.set_ranges([tuple(r) for r in rs])
                    fps = pl.fps
                    ahead = (pl.buffered_end() - int(pos)) / fps
                    st = {"playing": "відтворення", "buffering": "буферизація…", "paused": "пауза"}[pl.state]
                    if pl.kind == "image":
                        segs = [(0, f"{pl.size[0]}×{pl.size[1]}"), (1, tr("обробка…" if self._last_mask is None else "готово"))]
                    else:
                        a = pl.audio.status
                        snd = "♪" if a == "ок" else "♪ " + tr("без звуку" if a == "немає доріжки" else "немає sounddevice"
                                                          if a.startswith("встанови") else a)
                        segs = [(0, tr(st)), (1, tr("запас {s:.1f} с", s=ahead))]
                        if pl.stride > 1:
                            segs.append((2, tr("кожен {n}-й", n=pl.stride)))
                        segs += [(3 if a != "ок" else 6, snd), (4, tr("обробка {r:.0f} к/с", r=pl.render_fps())),     # один лічильник: усі потоки разом
                                 (7, f"{pl.workers}×"), (5, tr("файл {f:.2f} к/с", f=fps))]
                        if self.params.method == BRANCH and not self.params.glyph:
                            try:
                                from .student import label
                                if label():
                                    segs.insert(2, (6, "IRNM1: " + label()))             # чим рахує модель: TensorRT / CUDA / DirectML · GPU або CPU
                            except Exception:  # noqa: BLE001
                                pass
                    self._txt(self.stat_lbl, self._fit_status(segs))
                    self._estimate(pl)
        except Exception:  # noqa: BLE001
            _log.get().exception('помилка')
        try:
            self._meter(shown, rd, time.perf_counter() - t0, gap, skipped)
        except Exception:  # noqa: BLE001
            self._pm = None
        self.after(max(1, 8 - int((time.perf_counter() - t0) * 1000)), self._tick)

    @staticmethod
    def _txt(lbl, text):
        """підпис міняємо лише коли текст справді інший (зайве перемальовування - це мікро-затримки відтворення)"""
        if getattr(lbl, "_last_txt", None) != text:
            lbl._last_txt = text
            lbl.config(text=text)

    def _estimate(self, pl):
        """скільки приблизно триватиме експорт: ще не пораховані кадри / поточна швидкість обробки"""
        txt = ""
        if not self.exporting and pl.kind != "image":
            r = pl.render_fps()
            a, b = self._range() or (0, pl.n)
            tot, got = pl.total_in(a, b), pl.cached_in(a, b)
            todo = max(0, tot - got)
            if todo == 0:
                txt = tr("Експорт: усе пораховано — миттєво")
            elif r > 1:
                txt = tr("Експорт ≈ {t}  (готово {p}%)", t=fmt_t(todo / r), p=100 * got // max(1, tot))
        self._txt(self.est_lbl, txt)

    def _lut(self):
        """таблиця рівнів сірого -> колір палітри (256×1×3, RGB); None для чорно-білої Класики"""
        pn = self._pal_key() if hasattr(self, "pal_var") else palette.CLASSIC
        if palette.is_classic(pn):
            return None
        if getattr(self, "_lut_for", None) != pn:
            ink, paper = palette.colors(pn)
            t = np.arange(256, dtype=np.float32)[:, None] / 255.0
            self._lut_arr = (np.array(ink, np.float32) * (1 - t) + np.array(paper, np.float32) * t).astype(np.uint8).reshape(256, 1, 3)
            self._lut_for = pn
        return self._lut_arr

    # ------------------------------------------------------------ перегляд: масштаб колесом, зсув перетягуванням
    # (вся логіка - у benvid/view.py: збільшення лише ціле й найближчим сусідом, зменшення - усереднення)
    def _compose(self, cw, ch):
        mask = self._last_mask
        if mask is None:
            return None
        if self._defocus > 0.004:                         # пасхалка: 0/1-маска стає плавними відтінками
            mask = eggs.defocus(mask, self._defocus)
        return view.compose(mask, self._lut(), cw, ch, self._zoom, self._vc)

    def _frame_redraw(self):
        """новий кадр відтворення: найважче в показі - зменшення кадра до вікна (view.compose, ≈10 мс на 1440p, OpenCV віддає GIL) - іде у фоновому потоці, поки інтерфейс
        малює попередній кадр; інтерфейс лише кладе готове на полотно (_poll_compose). Журнал показу (кадр 29.9 к/с, 1440p) показував ≈30-36 мс на кадр у потоці інтерфейсу
        при бюджеті 33 мс, тож 14-21% кадрів пропускалось. Складніші випадки (розфокус-пасхалка, підпис) - як раніше, синхронно"""
        cw, ch = self.screen.winfo_width(), self.screen.winfo_height()
        if cw < 8 or ch < 8 or self._last_mask is None or self._defocus > 0.004 or self._egg_t0 is not None:
            self._redraw()
            return
        if self._cfut is not None and not self._cfut.done():
            self._cpend = True                        # попередній ще рахується: коли скінчить, візьмемо найновіший кадр
            return
        mask, lut, zoom, vc = self._last_mask, self._lut(), self._zoom, self._vc
        self._cinfo = (self._pgen, cw, ch, time.perf_counter())
        if self._cx is None:
            from concurrent.futures import ThreadPoolExecutor
            self._cx = ThreadPoolExecutor(1, thread_name_prefix="benvid-compose")
        self._cfut = self._cx.submit(view.compose, mask, lut, cw, ch, zoom, vc)

    def _poll_compose(self):
        """забрати готовий зі фонового потоку кадр і намалювати; True, якщо щось намалювали"""
        f = self._cfut
        if f is None or not f.done():
            return False
        self._cfut = None
        gen, cw, ch, t0 = self._cinfo
        drawn = False
        try:
            comp = f.result()
            if gen == self._pgen and (self.screen.winfo_width(), self.screen.winfo_height()) == (cw, ch):
                self._pc = time.perf_counter() - t0          # скільки кадр ішов від початку зменшення до готовності (для журналу показу)
                self._paint(comp, cw, ch)
                drawn = True
        except Exception:  # noqa: BLE001
            _log.get().exception('помилка')
        if self._cpend:
            self._cpend = False
            self._frame_redraw()
        return drawn

    def _redraw(self):
        cw, ch = self.screen.winfo_width(), self.screen.winfo_height()
        if cw < 8 or ch < 8:
            return
        self._pgen += 1                               # будь-яке синхронне перемальовування скасовує зменшення кадра, що ще рахується у фоні (воно вже застаріле)
        self._paint(self._compose(cw, ch), cw, ch)

    def _paint(self, comp, cw, ch):
        """покласти вже складений кадр (результат view.compose) на полотно разом із написами"""
        self._npaint += 1
        _tp0 = time.perf_counter()
        if comp is None:
            self.screen.delete("all")
            self._paint_sig = None
            wrap = max(160, min(cw - 40, 620))
            self.screen.create_text(cw // 2, ch // 2 - 36, text=tr("Відкрий відео, GIF або фото\nабо просто перетягни файл у це вікно"),
                                    fill=FG, font=FONT_B, justify="center", width=wrap)
            self.screen.create_text(cw // 2, ch // 2 + 44, text=guide.philosophy(), fill=DIM, font=FONT_S, justify="center", width=min(wrap, 460))
            if not self._empty_shown:                 # порожній екран показано заново: нова порада
                self._empty_shown = True
                self._tip_n = guide.tip(self._tip_n)
            if self._tip_visible():
                self.screen.create_text(cw // 2, ch // 2 + 104, text=guide.tip_text(self._tip_n), fill=DIM, font=FONT_S, justify="center",
                                        width=min(wrap, 460))
            self._draw_hud()
            return
        self._empty_shown = False
        img, zoom, px, py, crop = comp
        photo = None
        try:                                          # найшвидше: PIL кладе піксельний блок прямо в PhotoImage (без розбору тексту)
            from PIL import Image, ImageTk
            im = Image.fromarray(img)
            if zoom > 1:
                im = im.resize((im.width * zoom, im.height * zoom), Image.NEAREST)
            if crop:
                im = im.crop((crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3]))
            ph = self._photo
            # ImageTk.PhotoImage не має публічного `.mode` (лише приватний), тож колишня перевірка `ph.mode` кидала AttributeError, і щодруга картинка
            # йшла запасним шляхом через PGM-текст (21-25 мс замість ≈10 на 2580×1080, у журналі показу це режим «?»): режим тримаємо самі
            if isinstance(ph, ImageTk.PhotoImage) and self._photo_key == (im.width, im.height, im.mode) and (ph.width(), ph.height()) == (im.width, im.height):
                ph.paste(im)                          # той самий розмір - перевикористати
            else:
                ph = ImageTk.PhotoImage(im)
                self._photo_key = (im.width, im.height, im.mode)
            photo = ph
        except Exception:  # noqa: BLE001
            photo = None
        if photo is None:
            arr = view.finish(img, zoom, crop)
            try:                                      # запасний шлях: PGM/PPM без стиснення
                magic = b"P6" if arr.ndim == 3 else b"P5"
                photo = tk.PhotoImage(data=b"%s %d %d 255\n" % (magic, arr.shape[1], arr.shape[0]) + arr.tobytes())
            except Exception:  # noqa: BLE001
                rgb = arr if arr.ndim == 2 else np.ascontiguousarray(arr[..., ::-1])      # PNG-кодер чекає BGR
                ok, buf = cv2.imencode(".png", rgb, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                photo = tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
        reused = photo is self._photo
        self._photo = photo
        _tp1 = time.perf_counter()
        # Швидкий шлях: картинку вже змінено на місці (paste), а розкладка полотна та сама (позиція, розмір, масштаб, мова, тема, без пасхалки):
        # елементи полотна не чіпаємо - ні delete("all"), ні нових create_*, лише рухомий HUD. Раніше кожен кадр перебудовував усе (≈15 мс)
        sig = (cw, ch, px, py, photo.width(), photo.height(), self._zoom, i18n.lang(), BG, FG)
        if reused and sig == self._paint_sig and self._egg_t0 is None:
            self._pan_cursor(cw, ch)
            self._draw_hud()
            pa = self._pacc
            pa["img"] += _tp1 - _tp0
            pa["items"] += time.perf_counter() - _tp1
            pa["n"] += 1
            return
        self.screen.delete("all")
        self._paint_sig = sig if self._egg_t0 is None else None
        self.screen.create_image(px, py, image=photo, anchor="nw")
        self._pan_cursor(cw, ch)
        if px > 0 or py > 0:                          # картинка менша за поле: рамка по її краю (з проміжком 2 px) - тло вікна збігається з кольором туші, тож без неї край зливається
            self.screen.create_rectangle(px - 3, py - 3, px + photo.width() + 2, py + photo.height() + 2, outline=FG, width=1)
        if self._zoom is not None:                    # позначка масштабу в кутку
            t = self.screen.create_text(cw - M - 6, M + 3, anchor="ne", fill=FG, font=FONT, text=view.label(self._zoom) + tr("   0 — вписати"))
            bb = self.screen.bbox(t)
            r = self.screen.create_rectangle(bb[0] - 6, bb[1] - 3, bb[2] + 6, bb[3] + 3, fill=BG, outline=FG)
            self.screen.tag_lower(r, t)
        if self._egg_t0 is not None:
            self._draw_caption(cw, ch)
        self._draw_hud()
        pa = self._pacc                                   # розбивка для журналу показу: PIL+вставка в Tk (img) і решта полотна (items)
        pa["img"] += _tp1 - _tp0
        pa["items"] += time.perf_counter() - _tp1
        pa["n"] += 1
        pa["view"] = (photo.width(), photo.height(), getattr(photo, "_PhotoImage__mode", "?"), cw, ch)

    # ------------------------------------------------------------ що робить програма зараз (щоб не здавалось, що вона зависла)
    def _hud_info(self):
        """-> (текст, чи крутиться лічильник) або None, коли все спокійно (іде відтворення чи стоїть готовий кадр). Будь-яке очікування -
        рендер, буферизація, прогрів «Гравюри», обробка кадра, пропуск кадрів - супроводжується написом і анімацією"""
        if self.exporting:
            f = self._exp_f
            gtag = " · GPU" if self.params.method == 5 and not self.params.glyph and gpu.slots_cached() else ""        # відеокарта бере участь у рендері
            if self.params.method == BRANCH and not self.params.glyph:
                try:
                    from .student import current
                    from .student import label
                    gtag = " · " + label() if current() in ("cuda", "dml", "trt") else ""
                except Exception:  # noqa: BLE001
                    gtag = ""
            if self.pause_ev.is_set():
                return tr("ПАУЗА · {pct:.0f}%", pct=f * 100), False
            if self._exp_total > 1 and f > 0:
                return tr("РЕНДЕР · кадр {cur}/{tot}", cur=min(self._exp_total, int(round(f * self._exp_total))), tot=self._exp_total) + gtag, True
            return tr("РЕНДЕР · {pct:.0f}%", pct=f * 100) + gtag, True
        if not self.ready:
            return None
        pl = self.player
        if pl.warming:
            return tr("ГОТУЮ СИМВОЛИ… перший запуск довший" if self.params.glyph else "ГОТУЮ ГРАВЮРУ… перший запуск довший"), True
        if self.params.method == BRANCH and not self.params.glyph:
            try:
                from .student import preparing
                if preparing():
                    from .student import stage_label
                    sl = stage_label()
                    return (tr("ГОТУЮ IRNM1… перевіряю {b}", b=sl) if sl else tr("ГОТУЮ IRNM1… перший запуск довший")), True
            except Exception:  # noqa: BLE001
                pass
        if pl.playing and pl.state == "buffering":
            have, need = pl.buffer_progress()
            t = tr("БУФЕРИЗАЦІЯ · {have}/{need} кадрів", have=have, need=need)
            if pl.stride > 1:
                t += " · " + tr("кожен {n}-й", n=pl.stride)
            return t, True
        if pl.waiting_frame():
            return tr("ОБРОБКА…"), True
        if pl.playing and pl.stride > 1:
            return tr("ПК не встигає · кожен {n}-й кадр", n=pl.stride), False
        return None

    def _hud_update(self):
        info = self._hud_info()
        phase = int(time.perf_counter() * 8) % 8 if info and info[1] else 0
        sig = (info, phase)
        if sig != self._hud_sig:
            self._hud_sig, self._hud = sig, info
            self._draw_hud()

    _RING = ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1))

    def _draw_hud(self):
        """напис зі «крутилкою» у лівому верхньому куті екрана (праворуч - позначка масштабу)"""
        c = self.screen
        c.delete("hud")
        info = self._hud
        if not info:
            return
        cw = c.winfo_width()
        if cw < 120:
            return
        text, spin = info
        phase = self._hud_sig[1] if self._hud_sig else 0
        pad, cell, gap = 6, 4, 1
        ring = 3 * cell + 2 * gap
        x0, y0 = M + 6, M + 3
        tx = x0 + pad + (ring + pad if spin else 0)
        t = c.create_text(tx, y0 + 3, anchor="nw", fill=FG, font=FONT, text=text, width=max(80, cw - tx - 2 * M - 120), tags="hud")
        bb = c.bbox(t)
        top, bot = bb[1] - 3, max(bb[3] + 3, y0 + ring + 2 * pad - 3)
        r = c.create_rectangle(x0, top, bb[2] + pad, bot, fill=BG, outline=FG, tags="hud")
        c.tag_lower(r, t)
        if spin:                                  # кільце 3×3 квадратиків: «голова» і два слідки біжать колом
            ry = (top + bot) // 2 - ring // 2
            for i, (gx, gy) in enumerate(self._RING):
                d = (phase - i) % 8
                col = FG if d == 0 else DIM if d in (1, 2) else None
                px, py = x0 + pad + gx * (cell + gap), ry + gy * (cell + gap)
                c.create_rectangle(px, py, px + cell - 1, py + cell - 1, fill=col or BG, outline=col or DIM, tags="hud")

    def _pan_cursor(self, cw, ch):
        """над збільшеною картинкою, що не влізла у вікно, - рука (її можна тягнути); під час перетягування - «хрест» (див. _screen_drag)"""
        mask = self._last_mask
        want = ""
        if mask is not None and self._zoom is not None:
            h, w = mask.shape
            s_, W, H = view.geometry(cw, ch, w, h, self._zoom, self._vc)[:3]
            if W > cw or H > ch:
                want = "hand2"
        d = self._drag
        if d is not None and d.get("moved"):
            want = "fleur"
        if want != getattr(self, "_scr_cursor", None):
            self._scr_cursor = want
            self.screen.config(cursor=want)

    def _zoom_to(self, new, mx=None, my=None):
        """поставити масштаб new (None - вписати), лишивши точку під курсором (mx, my; типово центр) на місці"""
        mask = self._last_mask
        if mask is None:
            return
        cw, ch = self.screen.winfo_width(), self.screen.winfo_height()
        h, w = mask.shape
        fit = view.fit_scale(cw, ch, w, h)
        if new is None or abs(new - fit) <= fit * 1e-4:               # те саме, що «вписати»
            self._zoom, self._vc = None, view.CENTER
        else:
            s0, W0, H0, ox0, oy0 = view.geometry(cw, ch, w, h, self._zoom, self._vc)
            self._vc = view.anchored(cw, ch, w, h, s0, ox0, oy0, new, cw / 2 if mx is None else mx, ch / 2 if my is None else my)
            self._zoom = new
        self._redraw()

    def _zoom_step(self, d, mx=None, my=None):
        """один крок рівнів масштабу: d > 0 - збільшити, d < 0 - зменшити (до «вписати» і не далі)"""
        mask = self._last_mask
        if mask is None:
            return
        cw, ch = self.screen.winfo_width(), self.screen.winfo_height()
        h, w = mask.shape
        s0 = view.geometry(cw, ch, w, h, self._zoom, self._vc)[0]
        changed, new = view.step(s0, view.fit_scale(cw, ch, w, h), d, self._zoom is not None)
        if changed:
            self._zoom_to(new, mx, my)

    def _zoom_reset(self):
        if self._zoom is not None:
            self._zoom, self._vc = None, view.CENTER
            self._redraw()

    def _zoom_100(self):
        """1:1 - кожен піксель результату = піксель екрана"""
        if self._last_mask is not None and self._zoom != 1:
            self._zoom_to(1)

    def _in(self, w, anc):
        """чи віджет w є anc або лежить усередині нього"""
        while w is not None:
            if w is anc:
                return True
            w = getattr(w, "master", None)
        return False

    def _on_wheel(self, e):
        """колесо - одне на все вікно, віджет визначаємо за позицією курсора (на Windows подія приходить у віджет з фокусом,
        а не під курсором): над картинкою - масштаб, над повзунком звуку - гучність, над правою панеллю, якщо вона не
        влазить, - її прокрутка, деінде - прокрутка сторінки до футера й назад"""
        try:
            w = self.winfo_containing(e.x_root, e.y_root)
        except (KeyError, tk.TclError):
            return
        if w is None or w.winfo_toplevel() is not self:
            return                                       # випадний список комбобокса, підказки й інші вікна - не наші
        d = getattr(e, "delta", 0)
        if getattr(e, "num", 0) == 4:
            d = 120
        elif getattr(e, "num", 0) == 5:
            d = -120
        self._wheel_acc += d
        steps = int(self._wheel_acc / 120)               # дрібні кроки тачпада накопичуємо до цілого «клацання»
        if not steps:
            return "break"
        self._wheel_acc -= steps * 120
        sg = 1 if steps > 0 else -1
        for _ in range(abs(steps)):
            if w is self.screen:
                self._zoom_step(sg, e.x_root - self.screen.winfo_rootx(), e.y_root - self.screen.winfo_rooty())
            elif self._in(w, self.vol):
                self._vol_step(5 * sg)
            elif self._in(w, self.side) and self.sstack.overflow:
                self.sstack.notch(sg)
            elif self.pstack.enabled:
                self.pstack.notch(sg)
        return "break"

    def _screen_press(self, e):
        self._drag = dict(x=e.x, y=e.y, sx=e.x, sy=e.y, moved=False)

    def _screen_drag(self, e):
        d = self._drag
        if d is None:
            return
        if not d["moved"] and abs(e.x - d["sx"]) + abs(e.y - d["sy"]) < 4:
            return                                       # ще клік, а не перетягування
        d["moved"] = True
        mask = self._last_mask
        if self._zoom is None or mask is None:
            return
        h, w = mask.shape
        self._vc = view.pan(self.screen.winfo_width(), self.screen.winfo_height(), w, h, self._zoom, self._vc,
                            e.x - d["x"], e.y - d["y"])
        d["x"], d["y"] = e.x, e.y
        self._redraw()

    def _screen_release(self, e):
        d, self._drag = self._drag, None
        self._pan_cursor(self.screen.winfo_width(), self.screen.winfo_height())
        if d is not None and not d["moved"]:
            self._toggle()

    # ------------------------------------------------------------ експорт
    def _export(self, freeze=False):
        if not self.ready or self.exporting:
            return
        kind, fmt = self.kind, self.fmt_var.get()
        if freeze:                                              # «Порахувати все»: завжди .ikr, потім файл відкривається в режимі замка
            if kind != "video":
                return
            fmt = IKR_FORMAT
        base = os.path.splitext(os.path.basename(self.player.path))[0] + "_bw"
        rng = self._range() if kind == "video" else None
        if rng:
            base += f"_{int(rng[0] / (self.player.fps or 24))}-{int(rng[1] / (self.player.fps or 24))}s"
        a0, b0 = rng if rng else (0, None)
        if kind == "video":
            ext, codec = FORMATS[fmt]
        elif kind == "gif":
            ext, codec = GIF_FORMATS[fmt], ("png" if fmt == "PNG-послідовність" else "gif")
        else:
            ext, codec = still_formats(True)[fmt], "still"
        if codec == "png":
            out = filedialog.askdirectory(parent=self, title=tr("Тека для PNG-кадрів"), initialdir=self._src_dir())
            if out:
                out = os.path.join(out, base)
        else:
            out = filedialog.asksaveasfilename(parent=self, title=tr("Порахувати все й зберегти як") if freeze else tr("Зберегти як"), initialdir=self._src_dir(),
                                               initialfile=base + ext, defaultextension=ext, filetypes=[(tr(fmt), "*" + ext)])
        if not out:
            return
        if os.path.normcase(os.path.abspath(out)) == os.path.normcase(os.path.abspath(self.player.path)):
            self._say("Не можна зберегти поверх оригіналу — вибери інше ім'я")
            return
        workers = default_workers()
        pal_name = self._pal_key()
        self._show_app()
        self.player.pause()
        self.btn_play.set_icon("play")
        cache = None
        if kind == "video" and not self.player.baked:   # уже пораховані плеєром кадри цього розміру не рахуємо вдруге (з .ikr кадри читаються з самого файлу)
            psize, snap = self.player.snapshot()
            if psize == out_size(self.player.info["width"], self.player.info["height"], self._size_key()):
                cache = snap
        self.player.suspend(True)
        self.exporting = True
        self.cancel.clear()
        self.pause_ev.clear()
        self._pausable = kind == "video"
        self._exp_f, self._exp_label = 0.0, None
        fps_lim = self._fps_limit() if kind == "video" else None
        self._exp_total = max(1, int(round(((b0 or self.player.n) - a0) / fps_ratio(self.player.fps, fps_lim)))) if kind == "video" else 0
        for b in (self.btn_export, self.btn_export2):
            b.set_enabled(False)
        self.btn_open.set_enabled(False)
        self.btn_stop.set_enabled(True)
        self.btn_play.set_enabled(self._pausable)
        self.btn_play.set_icon("pause")
        clock = self._clock = Clock()
        aud = self.btn_aud.on
        params, size_key, q, nv, path = Params(**self.params.to_dict()), self._size_key(), self.q_var.get(), True, self.player.path

        def prog(f, label=None):
            eta = clock.eta(f)
            self.q.put(lambda: self._show_prog(f, label, clock.elapsed(), eta))

        def live(n, mask):                        # свіжий готовий кадр - на екран: видно, що саме зараз рахується
            self.q.put(lambda: self._show_live(mask))

        def work():
            try:
                from . import procs
                procs.lower_thread_priority()          # потік експорту - нижче за інтерфейс: вікно відповідає під будь-яким навантаженням
            except Exception:  # noqa: BLE001
                pass
            try:
                if kind == "video":
                    # апаратний кодер (NVENC/AMF/Quick Sync) - для «Середньої»/«Малої»; «Висока» кодується програмним x264:
                    # на дізері він чистіший і менший за апаратні
                    use_nv = (detect_hw_encoder() or False) if (nv and q != "Висока") else False
                    try:
                        r = export_video(path, out, params, fmt, size_key, q, use_nv, workers, None, prog, self.cancel, aud, cache, a0, b0, pal_name,
                                         pause=self.pause_ev, preview=live, fps=fps_lim)
                    except Cancelled:
                        raise
                    except RuntimeError:
                        if not use_nv:
                            raise
                        r = export_video(path, out, params, fmt, size_key, q, False, workers, None, prog, self.cancel, aud, cache, a0, b0, pal_name,
                                         pause=self.pause_ev, preview=live, fps=fps_lim)
                elif kind == "gif":
                    r = export_gif_source(path, out, params, fmt, size_key, workers, prog, self.cancel, palette=pal_name)
                else:
                    r = export_still(path, out, params, size_key, prog, palette=pal_name)
                self._last_out = out
                kw = dict(frames=r["frames"], w=r["size"][0], h=r["size"][1], mb=r["bytes"] / 1e6, p=self._short_path(out))
                self.q.put(lambda: self._export_done("Готово: {frames} · {w}×{h} · {mb:.1f} МБ\n{p}\nклік — відкрити теку", 1.0, True, **kw))
                if freeze:
                    self.q.put(lambda: self._enter_lock(out))
            except Cancelled:
                self.q.put(lambda: self._export_done("Скасовано", 0.0))
            except Exception as e:  # noqa: BLE001
                _log.get().exception('помилка експорту')
                self.q.put(lambda e=e: self._export_done("Помилка: {e}\n(деталі: {p})", 0.0, False, e=e, p=_log.shown_path()))

        self._exp_thread = threading.Thread(target=work, daemon=True)
        self._exp_thread.start()

    def _show_prog(self, f, label, el, eta):
        if not self.exporting:                   # запізнілий звіт із черги після завершення чи скасування
            return
        self._exp_f, self._exp_label = f, label
        self.prog.set(f)
        self._export_status(el, eta)

    def _show_live(self, mask):
        """готовий кадр результату - на екран (під час експорту; ~3 рази на секунду)"""
        if self.exporting and not self.pause_ev.is_set():
            self._last_mask = mask
            self._redraw()

    def _export_status(self, el=None, eta=None):
        """підпис прогресу: відсоток, час, кадр і швидкість; на паузі - підказка, як продовжити"""
        f, clk = self._exp_f, self._clock
        if el is None:
            el = clk.elapsed() if clk is not None else 0.0
            eta = clk.eta(f) if clk is not None else None
        lab = tr(self._exp_label or "Обробка")
        if self.pause_ev.is_set():
            self._say("ПАУЗА  {pct:3.0f}%\nминуло {el}\n[Пробіл] — продовжити", pct=f * 100, el=fmt_t(el))
        elif eta is not None and f < 1 and self._exp_total > 1 and el >= 1.0:
            done = min(self._exp_total, int(round(f * self._exp_total)))
            sp = clk.speed() if clk is not None else None          # швидкість за останні секунди, а не середня за весь час
            fps = sp * self._exp_total if sp else done / max(el, 1e-6)
            self._say("{lab}  {pct:3.0f}%\nминуло {el}  залишилось ~{eta}\nкадр {cur}/{tot} · {fps:.1f} к/с", lab=lab, pct=f * 100,
                      el=fmt_t(el), eta=fmt_t(eta), cur=done, tot=self._exp_total, fps=fps)
        elif eta is not None and f < 1:
            self._say("{lab}  {pct:3.0f}%\nминуло {el}  залишилось ~{eta}", lab=lab, pct=f * 100, el=fmt_t(el), eta=fmt_t(eta))
        else:
            self._say("{lab}  {pct:3.0f}%\nминуло {el}", lab=lab, pct=f * 100, el=fmt_t(el))

    def _export_done(self, key, f, done=False, **kw):
        """кінець експорту (успіх, скасування, помилка). Кожен крок захищений окремо: збій одного (напр. годинника) не повинен лишити
        плеєр призупиненим, а кнопки - вимкненими: тоді вікно виглядає «завислим», хоч нічого не рахується"""
        self.exporting = False
        self.pause_ev.clear()
        self._pausable = False

        def step(fn, *a):
            try:
                fn(*a)
            except Exception:  # noqa: BLE001
                _log.get().exception("export: finishing step failed")

        try:
            if self._clock is not None:
                step(self._clock.hold, False)
            self.btn_play.set_enabled(True)
            self.btn_play.set_icon("play")
            step(self.player.suspend, False)
            step(self.player.refresh)                # на екрані мав лишитися кадр із позиції плеєра, а не останній кадр експорту
            self.prog.set(f)
            if "frames" in kw:                       # у підсумку - число з іменником у правильній формі
                kw["frames"] = tn(kw["frames"], "{n} кадр|{n} кадри|{n} кадрів")
            self._say(key, done=done, **kw)
        except Exception:  # noqa: BLE001
            _log.get().exception("export: finishing step failed")
        finally:
            for b in (self.btn_export, self.btn_export2):
                b.set_enabled(True)
            self.btn_open.set_enabled(True)
            self.btn_stop.set_enabled(False)
        if done:
            self._spin_reel()                        # готово: котушка робить оберт

    def _open_out(self):
        out = getattr(self, "_last_out", None)
        if not out or not self._prog_done:
            return
        d = out if os.path.isdir(out) else os.path.dirname(out)
        self._out_dir = d
        try:
            if os.name == "nt":
                os.startfile(d)  # noqa: S606
            else:
                import subprocess
                subprocess.Popen(["xdg-open", d])
        except Exception:  # noqa: BLE001
            pass

    def _stop_export(self):
        self.cancel.set()

    def _poll(self):
        try:
            while True:
                self.q.get_nowait()()
        except queue.Empty:
            pass
        except Exception:  # noqa: BLE001
            _log.get().exception('помилка')
        self.after(50, self._poll)

    def _close(self):
        """закриття вікна: процес мусить справді зникнути (інакше невидимий InkReel у диспетчері завдань тримає слухача одиночного запуску
        й нові запуски «передаються» мертвому вікну). Кожен крок окремо - збій одного не повинен лишити решту живими"""
        from . import procs

        def step(fn, *a):
            try:
                fn(*a)
            except Exception:  # noqa: BLE001
                _log.get().exception("закриття: крок не вдався")
        step(procs.arm_exit_watchdog, 10.0)                # страховка від зависання будь-якого з кроків нижче (драйвер звуку, диск); далі - 4 с
        step(self._party_stop)
        self.cancel.set()
        lst = getattr(self, "_single", None)
        if lst:
            step(lst.close)                                # порт вільний одразу: швидкий повторний запуск стане слухачем сам
        t = getattr(self, "_exp_thread", None)
        if self.exporting and t is not None:
            step(t.join, 3)
        step(self._save_now)
        step(self.player.close)
        step(procs.release_everything, 4.0)                # усі пули й дочірні процеси + аварійний вихід за 4 с
        try:
            self.destroy()
        except Exception:  # noqa: BLE001
            _log.get().exception("закриття: вікно не знищено")


def _deliver(app, path, wait=2.0) -> bool:
    """файл, переданий повторним запуском: показати у вікні й підтвердити лише тоді, коли вікно справді його обробило (див. single.py)"""
    ev = threading.Event()

    def run():
        try:
            app.bring_front(path)
        finally:
            ev.set()
    app.q.put(run)
    return ev.wait(wait)


def main(path=None):
    _log.install_hooks()
    _log.log_environment()
    from . import single
    lst = single.acquire(os.path.abspath(path) if path else "")
    if lst is False:                                  # InkReel уже запущено: файл передано йому
        return
    app = App(path)
    app._single = lst or None
    if lst:
        lst.start(lambda p: _deliver(app, p))
    try:
        app.mainloop()
    finally:
        from . import procs
        procs.release_everything(4.0, lst or None)         # вікно зникло будь-яким шляхом - процес не лишається у фоні


if __name__ == "__main__":
    main()
