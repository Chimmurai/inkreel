"""Поверхня кадра. Логіка перегляду (масштаб лише цілий, зменшення усередненням, зсув) - та сама `benvid/view.py`, що й у Tk; тут лише показ.

`view.compose` (зменшення кадра до вікна, OpenCV віддає GIL) рахується у фоновому потоці, готове кладеться сигналом у потік інтерфейсу, де
це один `drawImage` без перебудови чого-небудь: у Tk кожен кадр створював заново всі елементи полотна й пересилав пікселі через Tk-фото."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .. import palette, view
from ..i18n import tr
from . import theme
from .theme import THEME, font


def lut_for(key: str):
    """таблиця сірий -> колір палітри (256×1×3 RGB); None для чорно-білої «Класики» (як у Tk)"""
    if palette.is_classic(key):
        return None
    ink, paper = palette.colors(key)
    t = np.arange(256, dtype=np.float32)[:, None] / 255.0
    return (np.array(ink, np.float32) * (1 - t) + np.array(paper, np.float32) * t).astype(np.uint8).reshape(256, 1, 3)


def to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        im = QImage(arr.data, w, h, arr.strides[0], QImage.Format_Grayscale8)
    else:
        im = QImage(arr.data, w, h, arr.strides[0], QImage.Format_RGB888)
    im._keep = arr                         # масив живе стільки, скільки QImage (Qt лише дивиться в ці байти)
    return im


class FrameSurface(QWidget):
    composed = Signal(object, object, int)      # (результат compose, номер покоління, мс у потоці)
    zoomChanged = Signal(object)                # масштаб або None («вписати»)

    def __init__(self, parent=None):
        super().__init__(parent)
        THEME.changed.connect(self._theme_changed)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setMinimumSize(320, 200)
        self.mask: np.ndarray | None = None
        self.zoom: float | None = None
        self.vc = view.CENTER
        self.lut = None
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="qt-compose")
        self._busy = False
        self._pending = False
        self._gen = 0
        self._img: QImage | None = None
        self._pos = (0, 0)
        self._shown = 0
        self._drag = None
        self._empty = None
        self._tip_n = -1
        self.stats = dict(frames=0, compose_ms=0.0, paint_ms=0.0)
        self.composed.connect(self._on_composed)
        self.setCursor(Qt.ArrowCursor)

    # --------------------------------------------------------------------------------- вхід
    def set_palette(self, key: str):
        self.lut = lut_for(key)
        self.refresh()

    def set_mask(self, mask: np.ndarray | None):
        self.mask = mask
        if mask is None:
            self._img = None
            self.rebuild_empty()
            self.update()
        else:
            self.refresh()

    def reset_view(self):
        self.zoom, self.vc = None, view.CENTER
        self.zoomChanged.emit(None)

    def refresh(self):
        """перерахувати показ (новий кадр, нова палітра чи розмір вікна); якщо потік ще зайнятий, візьме найновіше після нього"""
        if self.mask is None or self.width() < 8 or self.height() < 8:
            return
        if self._busy:
            self._pending = True
            return
        self._busy = True
        mask, lut, cw, ch, zoom, vc, gen = self.mask, self.lut, self.width(), self.height(), self.zoom, self.vc, self._gen

        def work():
            t0 = time.perf_counter()
            comp = view.compose(mask, lut, cw, ch, zoom, vc)
            if comp is not None:
                img, k, x, y, crop = comp
                comp = (view.finish(img, k, crop), x, y)
            self.composed.emit(comp, (gen, cw, ch), int((time.perf_counter() - t0) * 1000))
        self._pool.submit(work)

    def _on_composed(self, comp, tag, ms):
        self._busy = False
        gen, cw, ch = tag
        if comp is not None and gen == self._gen and (cw, ch) == (self.width(), self.height()):
            arr, x, y = comp
            self._img, self._pos = to_qimage(arr), (x, y)
            self.stats["compose_ms"] += ms
            if self._empty is not None:
                self._empty.hide()
            self.update()
        if self._pending:
            self._pending = False
            self.refresh()

    def _theme_changed(self):
        self.rebuild_empty()
        self.update()

    def rebuild_empty(self):
        """порожній екран (файл не відкрито) - три написи точно як у Tk-версії: прохання відкрити файл, філософія, випадкова порада"""
        from .. import guide
        from ..i18n import tr as _tr
        from .consts import FONT_B, FONT_S
        from .tkcanvas import TkCanvas
        if self._empty is None:
            self._empty = TkCanvas(self)
            self._empty.setGeometry(0, 0, self.width(), self.height())
            self._tip_n = guide.tip(-1)
        c = self._empty
        c.bg = THEME.bg.name()
        c.delete("all")
        cw, ch = self.width(), self.height()
        wrap = max(160, min(cw - 40, 620))
        c.create_text(cw // 2, ch // 2 - 36, text=_tr("Відкрий відео, GIF або фото\nабо просто перетягни файл у це вікно"), fill=THEME.fg.name(), font=FONT_B, justify="center", width=wrap)
        c.create_text(cw // 2, ch // 2 + 44, text=guide.philosophy(), fill=THEME.dim.name(), font=FONT_S, justify="center", width=min(wrap, 460))
        c.create_text(cw // 2, ch // 2 + 104, text=guide.tip_text(self._tip_n), fill=THEME.dim.name(), font=FONT_S, justify="center", width=min(wrap, 460))
        c.setGeometry(0, 0, cw, ch)
        c.setVisible(self._img is None)
        c.update()

    # --------------------------------------------------------------------------------- події
    def resizeEvent(self, e):
        self._gen += 1
        self.rebuild_empty()
        self.refresh()

    def wheelEvent(self, e):
        if self.mask is None:
            e.accept()                                   # як у Tk: над порожнім екраном колесо нічого не прокручує
            return
        d = 1 if e.angleDelta().y() > 0 else -1
        h, w = self.mask.shape
        cw, ch = self.width(), self.height()
        s0, W, H, ox, oy = view.geometry(cw, ch, w, h, self.zoom, self.vc)
        fit = view.fit_scale(cw, ch, w, h)
        changed, new = view.step(s0, fit, d, self.zoom is not None)
        if not changed:
            return
        if new is None:
            self.reset_view()
        else:
            m = e.position()
            self.vc = view.anchored(cw, ch, w, h, s0, ox, oy, new, m.x(), m.y())
            self.zoom = new
            self.zoomChanged.emit(new)
        self._gen += 1
        self.refresh()

    def mouseDoubleClickEvent(self, e):
        self.window().toggle_cinema() if hasattr(self.window(), "toggle_cinema") else None

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_0, Qt.Key_Insert) and self.mask is not None:
            self.reset_view()
            self._gen += 1
            self.refresh()
        else:
            super().keyPressEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.zoom is not None:
            self._drag = (e.position(), self.vc)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._drag is not None and self.mask is not None:
            p0, vc0 = self._drag
            d = e.position() - p0
            h, w = self.mask.shape
            self.vc = view.pan(self.width(), self.height(), w, h, self.zoom, vc0, int(d.x()), int(d.y()))
            self._gen += 1
            self.refresh()

    def mouseReleaseEvent(self, e):
        self._drag = None
        self.setCursor(Qt.ArrowCursor)

    # --------------------------------------------------------------------------------- малювання
    def paintEvent(self, e):
        t0 = time.perf_counter()
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.bg)
        if self._img is None or self.mask is None:
            return                                    # порожній екран малює дочірнє полотно _empty (див. rebuild_empty)
        x, y = self._pos
        p.drawImage(x, y, self._img)
        if x > 0 or y > 0:                         # картинка менша за поле: рамка з проміжком 2 px, як у Tk (інакше край зливається з тлом)
            p.setPen(QPen(THEME.fg, 1))
            p.drawRect(QRectF(x - 2.5, y - 2.5, self._img.width() + 4, self._img.height() + 4))
        if self.zoom is not None:
            txt = view.label(self.zoom) + tr("   0 — вписати")
            p.setFont(font(theme.NORMAL, True))
            fm = p.fontMetrics()
            w = fm.horizontalAdvance(txt) + 12
            r = QRectF(self.width() - w - 14, 10, w, fm.height() + 6)
            p.fillRect(r, THEME.bg)
            p.setPen(QPen(THEME.fg, 1))
            p.drawRect(r.adjusted(0.5, 0.5, -0.5, -0.5))
            p.drawText(r, Qt.AlignCenter, txt)
        self.stats["frames"] += 1
        self.stats["paint_ms"] += (time.perf_counter() - t0) * 1000
