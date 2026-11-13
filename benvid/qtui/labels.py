"""Підписи й поля в стилі Tk: `TkLabel` (розмір і розміщення тексту як у tk.Label: бордюр 2 px і поле 1 px навколо тексту, якір, перенос по
пікселях жадібно за пробілами), `TkImage` (малюнок без полів), `TkStrip` (смуга з плиток через один елемент)."""
from __future__ import annotations

import math

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QFontMetrics, QImage, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import assets
from .consts import FONT
from .theme import THEME, tk_font

_ANCHOR = {"nw": (0, 0), "n": (.5, 0), "ne": (1, 0), "w": (0, .5), "center": (.5, .5), "e": (1, .5), "sw": (0, 1), "s": (.5, 1), "se": (1, 1)}


def wrap_text(text: str, width: int, fm: QFontMetrics) -> list[str]:
    """жадібний перенос за пробілами, як у Tk: рядок не довший за width пікселів (довге слово рубається посимвольно)"""
    out: list[str] = []
    for para in str(text).split("\n"):
        if width <= 0:
            out.append(para)
            continue
        line = ""
        words = para.split(" ")
        for k, word in enumerate(words):
            cand = word if not line else line + " " + word
            # Tk міряє слово разом із пробілом, що за ним (крім останнього слова абзацу): рядок «влазить», лише якщо влазить і цей пробіл
            if fm.horizontalAdvance(cand + (" " if k < len(words) - 1 else "")) <= width:
                line = cand
                continue
            if line:
                out.append(line)
            while fm.horizontalAdvance(word) > width and len(word) > 1:
                k = len(word)
                while k > 1 and fm.horizontalAdvance(word[:k]) > width:
                    k -= 1
                out.append(word[:k])
                word = word[k:]
            line = word
        out.append(line)
    return out


class TkLabel(QWidget):
    """tk.Label: default bd=2, padx=pady=1; розмір = текст + 2·(bd+pad); height - у рядках (0: за текстом)"""

    def __init__(self, parent=None, text="", dim=False, font=FONT, anchor="center", justify="left", lines=0, wraplength=0, bd=2, padx=1, pady=1,
                 fg=None, bg=None, inverted=False, autowrap=False):
        super().__init__(parent)
        self.autowrap = autowrap                 # як App._autowrap: перенос по фактичній ширині підпису
        self.text_, self.font_, self.anchor, self.justify = text, font, anchor, justify
        self.lines, self.wrap = lines, wraplength
        self.bd, self.padx, self.pady = bd, padx, pady
        self.dim, self.inverted = dim, inverted
        self._fg, self._bg = fg, bg
        self.clicked = None
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        THEME.changed.connect(self.update)

    # --- API як у Tk
    def config(self, text=None, fg=None, wraplength=None, **_):
        changed = False
        if text is not None and text != self.text_:
            self.text_, changed = text, True
        if fg is not None:
            self._fg = fg
            changed = True
        if wraplength is not None and wraplength != self.wrap:
            self.wrap, changed = wraplength, True
        if changed:
            self.updateGeometry()
            self.update()

    configure = config

    def cget(self, key):
        return {"text": self.text_, "wraplength": self.wrap}.get(key)

    def winfo_width(self):
        return self.width()

    def winfo_reqheight(self):
        return self.sizeHint().height()

    # --- розмір
    def _fm(self):
        return QFontMetrics(tk_font(self.font_))

    def _inner_w(self):
        return self.width() - 2 * (self.bd + self.padx)

    def _lines(self, width):
        fm = self._fm()
        w = self.wrap if self.wrap else (width if self.wrap == 0 and False else 0)
        return wrap_text(self.text_, w, fm) if w else str(self.text_).split("\n")

    def sizeHint(self):
        fm = self._fm()
        lines = self._lines(0)
        n = self.lines or max(1, len(lines))
        w = max((fm.horizontalAdvance(s) for s in lines), default=0)
        if self.wrap:
            w = min(max(w, 1), self.wrap)
        return QSize(w + 2 * (self.bd + self.padx), n * fm.height() + 2 * (self.bd + self.pady))

    def minimumSizeHint(self):
        return QSize(10, self.sizeHint().height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.autowrap and self.wrap != max(60, self.width()):
            self.wrap = max(60, self.width())
            self.update()

    def mousePressEvent(self, e):
        if self.clicked and e.button() == Qt.LeftButton:
            self.clicked()

    def paintEvent(self, e):
        p = QPainter(self)
        bg = THEME.fg if self.inverted else THEME.bg
        fg = THEME.bg if self.inverted else (THEME.dim if self.dim else THEME.fg)
        if self._fg is not None:
            from PySide6.QtGui import QColor
            fg = QColor(self._fg)
        p.fillRect(self.rect(), bg)
        p.setFont(tk_font(self.font_))
        p.setPen(fg)
        fm = self._fm()
        lines = self._lines(self._inner_w())
        ix, iy = self.bd + self.padx, self.bd + self.pady
        iw, ih = self.width() - 2 * ix, self.height() - 2 * iy
        th = fm.height() * len(lines)
        fx, fy = _ANCHOR.get(self.anchor, (.5, .5))
        top = iy + math.floor(fy * (ih - th))              # текст вищий за поле (ih < th): Tk піднімає його, а не опускає (floor, не int)
        for k, s in enumerate(lines):
            lw = fm.horizontalAdvance(s)
            if self.justify == "center":
                x = ix + (iw - lw) // 2
            elif self.justify == "right":
                x = ix + iw - lw
            else:
                x = ix + int(fx * (iw - max(fm.horizontalAdvance(t) for t in lines))) if self.anchor in ("e", "ne", "se", "n", "s", "center") else ix
            p.drawText(QRect(x, top + k * fm.height(), lw + 2, fm.height()), Qt.AlignLeft | Qt.AlignTop | Qt.TextDontClip, s)


class TkImage(QWidget):
    """tk.Label з малюнком, bd=0"""

    def __init__(self, parent=None, name="", scale=1):
        super().__init__(parent)
        self.name, self.scale = name, scale
        self.clicked = None
        self._shown = None
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        THEME.changed.connect(self._retheme)
        self._retheme()

    def _retheme(self):
        im = assets.img(self.name)
        if im is not None and self.scale > 1:
            im = im.scaled(im.width() * self.scale, im.height() * self.scale, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self.im = im
        sz = (im.width(), im.height()) if im is not None else (0, 0)
        self.setFixedSize(*sz)
        self.update()

    def show_image(self, im):
        """показати іншу картинку того ж розміру (кадр анімації); None - назад до звичайної"""
        self._shown = im
        self.update()

    def mousePressEvent(self, e):
        if self.clicked and e.button() == Qt.LeftButton:
            self.clicked()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.bg)
        im = self._shown or self.im
        if im is not None:
            p.drawImage((self.width() - im.width()) // 2, (self.height() - im.height()) // 2, im)


class TkStrip(QWidget):
    """смуга з плиток (дитер, плівка): плитка тайлиться на всю ширину, неповні плитки порівну з обох країв (як App._tile_strip)"""

    def __init__(self, parent=None, which="dither_light", height=8):
        super().__init__(parent)
        self.which, self.h = which, height
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        THEME.changed.connect(self.update)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.bg)
        tile = assets.img(self.which)
        if tile is None:
            return
        tw, th = tile.width(), tile.height()
        cw = max(self.width(), 10)
        off = (cw % tw) // 2
        for x in range(-off, cw + tw, tw):
            for y in range(0, self.h, th):
                p.drawImage(x, y, tile)
