"""Піксельні віджети InkReel на QPainter: кнопка, повзунок з буфером, панель якості (п'ять кіл і ромб IRNM1), рамка-секція.
Мишу Qt захоплює сам (подія відпускання приходить віджету навіть поза його межами), тож «повзунок не відпускають і накопичуються події»
з Tk тут не буває: подій стільки, скільки кадрів вікна, а перемальовування лише потрібного прямокутника."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from . import theme
from .theme import THEME, font


class _Pix(QWidget):
    """база: перемальовується, коли змінилась тема"""

    def __init__(self, parent=None):
        super().__init__(parent)
        THEME.changed.connect(self.update)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)


class PixButton(_Pix):
    clicked = Signal()

    def __init__(self, text: str = "", parent=None, checkable: bool = False, min_w: int = 0, h: int = 28):
        super().__init__(parent)
        self._text, self._on, self._checkable = text, False, checkable
        self._hover = self._down = False
        self._min_w, self._h = min_w, h
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)
        self.setFixedHeight(h)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def text(self) -> str:
        return self._text

    def setText(self, t: str):
        self._text = t
        self.updateGeometry()
        self.update()

    def isOn(self) -> bool:
        return self._on

    def setOn(self, v: bool):
        if v != self._on:
            self._on = v
            self.update()

    def sizeHint(self):
        w = font().pixelSize() * 0.62 * max(1, len(self._text)) + 22
        return QSize(int(max(self._min_w, w)), self._h)

    def minimumSizeHint(self):
        return QSize(24, self._h)                   # кнопки стискаються в вузькій панелі, а не розпирають її

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = self._down = False
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._down = True
            self.update()

    def mouseReleaseEvent(self, e):
        was = self._down
        self._down = False
        self.update()
        if was and self.rect().contains(e.position().toPoint()) and self.isEnabled():
            if self._checkable:
                self._on = not self._on
            self.clicked.emit()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self.clicked.emit()
        else:
            super().keyPressEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        lit = self._down or self._on
        bg = THEME.fg if lit else (THEME.panel if self._hover else THEME.bg)
        fg = THEME.bg if lit else (THEME.fg if self.isEnabled() else THEME.dim)
        p.fillRect(self.rect(), bg)
        p.setPen(QPen(THEME.fg if self.isEnabled() else THEME.dim, 1))
        p.drawRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1))
        p.setPen(fg)
        p.setFont(font(theme.NORMAL, True))
        p.drawText(self.rect(), Qt.AlignCenter, self._text)


class PixSlider(_Pix):
    """доріжка з квадратною ручкою; `ranges` - вже порахований буфер (світлі відрізки). valueChanged іде під час перетягування, released - у кінці"""
    valueChanged = Signal(float)
    released = Signal(float)

    THUMB = 5

    def __init__(self, lo: float = 0.0, hi: float = 1.0, parent=None, h: int = 26):
        super().__init__(parent)
        self.lo, self.hi, self.v = lo, hi, lo
        self.ranges: list[tuple[float, float]] = []
        self.dragging = False
        self.setFixedHeight(h)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(False)

    def setRange(self, lo: float, hi: float):
        self.lo, self.hi = lo, max(hi, lo + 1e-9)
        self.update()

    def setValue(self, v: float):
        v = min(max(v, self.lo), self.hi)
        if v != self.v:
            self.v = v
            if not self.dragging:
                self.update()

    def setRanges(self, rs):
        rs = [tuple(r) for r in rs]
        if rs != self.ranges:
            self.ranges = rs
            self.update()

    def _x(self, v: float) -> float:
        w = self.width() - 1 - 2 * self.THUMB
        return self.THUMB + (v - self.lo) / (self.hi - self.lo) * w

    def _v_at(self, x: float) -> float:
        w = max(1, self.width() - 1 - 2 * self.THUMB)
        return self.lo + min(max((x - self.THUMB) / w, 0.0), 1.0) * (self.hi - self.lo)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.dragging = True
            self._move(e)

    def mouseMoveEvent(self, e):
        if self.dragging:
            self._move(e)

    def _move(self, e):
        self.v = self._v_at(e.position().x())
        self.update()
        self.valueChanged.emit(self.v)

    def mouseReleaseEvent(self, e):
        if self.dragging:
            self.dragging = False
            self.update()
            self.released.emit(self.v)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.bg)
        cy = self.height() / 2
        p.setPen(QPen(THEME.dim, 1))
        p.drawLine(QPointF(self.THUMB, cy), QPointF(self.width() - self.THUMB, cy))
        p.setPen(QPen(THEME.fg, 3))
        for a, b in self.ranges:
            p.drawLine(QPointF(self._x(a), cy), QPointF(self._x(b), cy))
        x = self._x(self.v)
        r = QRectF(x - self.THUMB, cy - self.THUMB - 2, 2 * self.THUMB, 2 * self.THUMB + 4)
        p.fillRect(r, THEME.fg)
        p.setPen(QPen(THEME.bg, 1))
        p.drawLine(QPointF(x, r.top() + 3), QPointF(x, r.bottom() - 3))


class PixProgress(_Pix):
    def __init__(self, parent=None, h: int = 14):
        super().__init__(parent)
        self.f = 0.0
        self.setFixedHeight(h)

    def set(self, f: float):
        f = min(max(f, 0.0), 1.0)
        if abs(f - self.f) > 1e-4:
            self.f = f
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.bg)
        p.setPen(QPen(THEME.fg, 1))
        p.drawRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1))
        p.fillRect(QRectF(2, 2, (self.width() - 4) * self.f, self.height() - 4), THEME.fg)


class QualityBar(_Pix):
    """п'ять кіл на одній лінії, обране заповнене; праворуч на пунктирному відгалуженні ромб IRNM1 (окрема гілка, а не шостий щабель)"""
    chosen = Signal(int)

    def __init__(self, parent=None, branch: bool = True):
        super().__init__(parent)
        self.level = 5
        self.branch = branch
        self.setFixedHeight(40)
        self.setCursor(Qt.PointingHandCursor)

    def setLevel(self, n: int):
        if n != self.level:
            self.level = n
            self.update()

    def _centers(self):
        w = self.width()
        left, right = 14, w - 14 - (46 if self.branch else 0)
        xs = [left + i * (right - left) / 4 for i in range(5)]
        return xs, (w - 14 if self.branch else None), self.height() / 2

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        xs, dx, cy = self._centers()
        x = e.position().x()
        cands = [(abs(x - cx), i + 1) for i, cx in enumerate(xs)]
        if dx is not None:
            cands.append((abs(x - dx), 6))
        d, n = min(cands)
        if d < 22:
            self.setLevel(n)
            self.chosen.emit(n)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), THEME.bg)
        xs, dx, cy = self._centers()
        p.setPen(QPen(THEME.fg, 1))
        p.drawLine(QPointF(xs[0], cy), QPointF(xs[-1], cy))
        if dx is not None:
            pen = QPen(THEME.fg, 1, Qt.DotLine)
            p.setPen(pen)
            p.drawLine(QPointF(xs[-1], cy), QPointF(dx, cy))
        for i, cx in enumerate(xs):
            on = self.level == i + 1
            p.setPen(QPen(THEME.fg, 2))
            p.setBrush(THEME.fg if on else THEME.bg)
            p.drawEllipse(QPointF(cx, cy), 7, 7)
        if dx is not None:
            on = self.level == 6
            p.setPen(QPen(THEME.fg, 2))
            p.setBrush(THEME.fg if on else THEME.bg)
            p.drawPolygon([QPointF(dx, cy - 9), QPointF(dx + 9, cy), QPointF(dx, cy + 9), QPointF(dx - 9, cy)])


class Section(QWidget):
    """секція правої панелі: рамка з білою смужкою-заголовком"""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        THEME.changed.connect(self.update)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 24, 8, 8)
        lay.setSpacing(6)
        self.body = lay

    def setTitle(self, t: str):
        self._title = t
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.bg)
        p.setPen(QPen(THEME.fg, 1))
        p.drawRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1))
        p.fillRect(QRectF(0, 0, self.width(), 18), THEME.fg)
        p.setPen(THEME.bg)
        p.setFont(font(theme.SMALL, True))
        p.drawText(QRectF(8, 0, self.width() - 16, 18), Qt.AlignVCenter | Qt.AlignLeft, self._title.upper())
