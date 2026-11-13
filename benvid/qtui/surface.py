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

from .. import eggs, palette, view
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
    clicked = Signal()                          # клік по картинці без перетягування: пауза/відтворення

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
        self.defocus, self.egg_t0, self._cap = 0.0, None, None
        self.hud, self.hud_phase, self._cursor_now, self._dbl = None, 0, None, False
        self.stats = dict(frames=0, compose_ms=0.0, paint_ms=0.0)
        self.composed.connect(self._on_composed)
        self.setCursor(Qt.ArrowCursor)

    # --------------------------------------------------------------------------------- вхід
    def set_palette(self, key: str):
        self.lut = lut_for(key)
        self.refresh()

    def set_defocus(self, level: float, t0):
        """пасхалка «розфокус»: рівень розмиття маски й час початку (для підпису; None - пасхалка скінчилась)"""
        self.defocus, self.egg_t0 = level, t0
        self.refresh()
        self.update()

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
        lvl = self.defocus

        def work():
            t0 = time.perf_counter()
            m = eggs.defocus(mask, lvl) if lvl > 0.004 else mask             # пасхалка: 0/1-маска стає плавними відтінками
            comp = view.compose(m, lut, cw, ch, zoom, vc)
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
        self._hud_items(c, cw)                                    # «ГОТУЮ ГРАВЮРУ…» тощо видно й поки першого кадра ще нема
        c.setGeometry(0, 0, cw, ch)
        c.setVisible(self._img is None)
        c.update()

    # --------------------------------------------------------------------------------- події
    def resizeEvent(self, e):
        self._gen += 1
        self.rebuild_empty()
        self.refresh()

    # --------------------------------------------------------------------------------- масштаб (лише цілі кроки; логіка в view.py, як у Tk)
    def zoom_to(self, new, mx=None, my=None):
        """поставити масштаб new (None - вписати), лишивши точку під курсором (mx, my; типово центр) на місці"""
        if self.mask is None:
            return
        cw, ch = self.width(), self.height()
        h, w = self.mask.shape
        fit = view.fit_scale(cw, ch, w, h)
        if new is None or abs(new - fit) <= fit * 1e-4:               # те саме, що «вписати»
            self.reset_view()
        else:
            s0, W0, H0, ox0, oy0 = view.geometry(cw, ch, w, h, self.zoom, self.vc)
            self.vc = view.anchored(cw, ch, w, h, s0, ox0, oy0, new, cw / 2 if mx is None else mx, ch / 2 if my is None else my)
            self.zoom = new
            self.zoomChanged.emit(new)
        self._gen += 1
        self.refresh()
        self.update()

    def zoom_step(self, d, mx=None, my=None):
        """один крок рівнів масштабу: d > 0 - збільшити, d < 0 - зменшити (до «вписати» і не далі)"""
        if self.mask is None:
            return
        cw, ch = self.width(), self.height()
        h, w = self.mask.shape
        s0 = view.geometry(cw, ch, w, h, self.zoom, self.vc)[0]
        changed, new = view.step(s0, view.fit_scale(cw, ch, w, h), d, self.zoom is not None)
        if changed:
            self.zoom_to(new, mx, my)

    def zoom_reset(self):
        if self.zoom is not None and self.mask is not None:
            self.zoom_to(None)

    def zoom_100(self):
        """1:1 - кожен піксель результату = піксель екрана"""
        if self.mask is not None and self.zoom != 1:
            self.zoom_to(1)

    # --------------------------------------------------------------------------------- події
    def wheelEvent(self, e):
        if self.mask is None:
            e.accept()                                   # як у Tk: над порожнім екраном колесо нічого не прокручує
            return
        m = e.position()
        self.zoom_step(1 if e.angleDelta().y() > 0 else -1, m.x(), m.y())
        e.accept()

    def mouseDoubleClickEvent(self, e):
        p = e.position()                                 # як у Tk: перший клік уже перемкнув паузу, другий (його відпускання) - вдруге; а подвійний клік - «кіно»
        self._drag = dict(x=p.x(), y=p.y(), sx=p.x(), sy=p.y(), moved=False)
        self.window().toggle_cinema() if hasattr(self.window(), "toggle_cinema") else None

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            p = e.position()
            self._drag = dict(x=p.x(), y=p.y(), sx=p.x(), sy=p.y(), moved=False)

    def mouseMoveEvent(self, e):
        d = self._drag
        if d is None:
            return
        p = e.position()
        if not d["moved"] and abs(p.x() - d["sx"]) + abs(p.y() - d["sy"]) < 4:
            return                                       # ще клік, а не перетягування
        d["moved"] = True
        if self.zoom is None or self.mask is None:
            return
        h, w = self.mask.shape
        self.vc = view.pan(self.width(), self.height(), w, h, self.zoom, self.vc, int(p.x() - d["x"]), int(p.y() - d["y"]))
        d["x"], d["y"] = p.x(), p.y()
        self._gen += 1
        self.refresh()
        self._pan_cursor()

    def mouseReleaseEvent(self, e):
        d, self._drag = self._drag, None
        self._pan_cursor()
        if d is not None and not d["moved"] and e.button() == Qt.LeftButton:
            self.clicked.emit()                          # клік по картинці - пауза/відтворення

    def _pan_cursor(self):
        """над збільшеною картинкою, що не влізла у вікно, - рука (її можна тягнути); під час перетягування - «хрест»"""
        want = Qt.ArrowCursor
        if self.mask is not None and self.zoom is not None:
            h, w = self.mask.shape
            s_, W, H = view.geometry(self.width(), self.height(), w, h, self.zoom, self.vc)[:3]
            if W > self.width() or H > self.height():
                want = Qt.PointingHandCursor
        d = self._drag
        if d is not None and d.get("moved"):
            want = Qt.SizeAllCursor
        if want != self._cursor_now:
            self._cursor_now = want
            self.setCursor(want)

    # --------------------------------------------------------------------------------- що робить програма зараз
    _RING = ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1))

    def set_hud(self, info, phase=0):
        """напис зі «крутилкою» у лівому верхньому куті (info - (текст, чи крутиться) або None)"""
        if (info, phase) != (self.hud, self.hud_phase):
            self.hud, self.hud_phase = info, phase
            if self._img is None and self._empty is not None:
                self.rebuild_empty()
            self.update()

    # --------------------------------------------------------------------------------- малювання
    def paintEvent(self, e):
        t0 = time.perf_counter()
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.bg)
        if self._img is None or self.mask is None:
            return                                        # порожній екран (і HUD на ньому) малює дочірнє полотно _empty (див. rebuild_empty)
        x, y = self._pos
        p.drawImage(x, y, self._img)
        self._overlay(p, with_picture=True)
        self.stats["frames"] += 1
        self.stats["paint_ms"] += (time.perf_counter() - t0) * 1000

    def _builder(self):
        from .tkcanvas import TkCanvas
        if self._cap is None:
            self._cap = TkCanvas(self)
            self._cap.hide()
        self._cap.delete("all")
        return self._cap

    def _overlay(self, p, with_picture=True):
        """усе, що Tk малював елементами полотна поверх картинки: рамка (картинка менша за поле), позначка масштабу, підпис розфокуса, HUD.
        Елементи збираємо на прихованому TkCanvas і малюємо тим самим кодом (піксель у піксель як у Tk)"""
        from .consts import FONT, M
        c = self._builder()
        cw, ch = self.width(), self.height()
        fg, bg = THEME.fg.name(), THEME.bg.name()
        if with_picture:
            x, y = self._pos
            if x > 0 or y > 0:                           # картинка менша за поле: рамка по її краю (з проміжком 2 px) - тло збігається з кольором туші, тож без неї край зливається
                c.create_rectangle(x - 3, y - 3, x + self._img.width() + 2, y + self._img.height() + 2, outline=fg, width=1)
            if self.zoom is not None:                    # позначка масштабу в кутку
                t = c.create_text(cw - M - 6, M + 3, anchor="ne", fill=fg, font=FONT, text=view.label(self.zoom) + tr("   0 — вписати"))
                bb = c.bbox(t)
                r = c.create_rectangle(bb[0] - 6, bb[1] - 3, bb[2] + 6, bb[3] + 3, fill=bg, outline=fg)
                c.tag_lower(r, t)
            if self.egg_t0 is not None:
                self._caption_items(c, cw, ch)
        self._hud_items(c, cw)
        c.draw_items(p)

    def _caption_items(self, c, cw, ch):
        """підпис, поки картинка розфокусована: два рядки філософії проявляються в рамці внизу екрана (рамка з'являється, коли перший рядок
        уже читається; напівпрозорості нема, тож проявляємо колір тексту від тла до кольору) - як App._draw_caption"""
        import textwrap

        from PySide6.QtGui import QFontMetrics

        from .consts import FONT_B
        from .theme import tk_font
        a1, a2 = eggs.caption_alphas(time.perf_counter() - self.egg_t0)
        if max(a1, a2) < 0.25:
            return
        l1, l2 = eggs.philosophy_parts()
        per = max(16, max(200, min(cw - 80, 660)) // max(1, QFontMetrics(tk_font(FONT_B)).horizontalAdvance("0")))   # символів у рядку (шрифт моноширинний)
        wrapped = lambda s: "\n".join(textwrap.wrap(s, per)) if s else ""
        bg, fg = THEME.bg.name(), THEME.fg.name()
        shade = lambda a: eggs.blend_hex(bg, fg, (a - 0.25) / 0.75)
        t2 = c.create_text(cw // 2, ch - 34, text=wrapped(l2), anchor="s", justify="center", font=FONT_B, fill=shade(a2))
        b2 = c.bbox(t2)
        t1 = c.create_text(cw // 2, b2[1] - 10, text=wrapped(l1), anchor="s", justify="center", font=FONT_B, fill=shade(a1))
        b1 = c.bbox(t1)
        r = c.create_rectangle(min(b1[0], b2[0]) - 16, b1[1] - 12, max(b1[2], b2[2]) + 16, b2[3] + 12, fill=bg, outline=fg)
        c.tag_lower(r, t2)

    def _hud_items(self, c, cw):
        """напис зі «крутилкою» у лівому верхньому куті екрана (праворуч - позначка масштабу) - як App._draw_hud"""
        from .consts import FONT, M
        info = self.hud
        if not info or cw < 120:
            return
        text, spin = info
        fg, bg, dim = THEME.fg.name(), THEME.bg.name(), THEME.dim.name()
        pad, cell, gap = 6, 4, 1
        ring = 3 * cell + 2 * gap
        x0, y0 = M + 6, M + 3
        tx = x0 + pad + (ring + pad if spin else 0)
        t = c.create_text(tx, y0 + 3, anchor="nw", fill=fg, font=FONT, text=text, width=max(80, cw - tx - 2 * M - 120))
        bb = c.bbox(t)
        top, bot = bb[1] - 3, max(bb[3] + 3, y0 + ring + 2 * pad - 3)
        r = c.create_rectangle(x0, top, bb[2] + pad, bot, fill=bg, outline=fg)
        c.tag_lower(r, t)
        if spin:                                          # кільце 3×3 квадратиків: «голова» і два слідки біжать колом
            ry = (top + bot) // 2 - ring // 2
            for i, (gx, gy) in enumerate(self._RING):
                d = (self.hud_phase - i) % 8
                col = fg if d == 0 else dim if d in (1, 2) else None
                px, py = x0 + pad + gx * (cell + gap), ry + gy * (cell + gap)
                c.create_rectangle(px, py, px + cell - 1, py + cell - 1, fill=col or bg, outline=col or dim)
