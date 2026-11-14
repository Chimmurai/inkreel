"""Тонкий шар «Tk-подібних» методів для Qt-віджетів, щоб перенесену логіку прокрутки й розкладки (ScrollStack/PageStack з колишнього вікна на Tk) можна було
взяти дослівно: `place`, `place_forget`, `winfo_*`, `after`, `bind("<Configure>")`."""
from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QWidget

from .theme import THEME


class Frame(QWidget):
    """tk.Frame: заливка кольором теми (тло чи, для рамки, текст) і методи в стилі Tk"""

    def __init__(self, parent=None, role="bg"):
        super().__init__(parent)
        self.role = role                       # "bg" - тло теми, "fg" - колір тексту (рамка екрана, інвертована смужка)
        self._conf = []
        self._timers: dict[int, QTimer] = {}
        self._tid = 0
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        THEME.changed.connect(self.update)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.fg if self.role == "fg" else THEME.bg)

    # --- Tk-подібне
    def place(self, x=0, y=0, width=None, height=None):
        self.setGeometry(int(x), int(y), int(width if width is not None else self.width()), int(height if height is not None else self.height()))
        self.show()

    def place_forget(self):
        self.hide()

    def winfo_width(self):
        return self.width()

    def winfo_height(self):
        return self.height()

    def winfo_reqheight(self):
        lay = self.layout()
        return lay.sizeHint().height() if lay is not None else self.sizeHint().height()

    def winfo_ismapped(self):
        return self.isVisible()

    def bind(self, ev, fn, add=None):
        if ev == "<Configure>":
            self._conf.append(fn)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        for fn in list(self._conf):
            fn(None)

    def after(self, ms, fn=None):
        if fn is None:
            return None
        self._tid += 1
        tid = self._tid
        t = QTimer(self)
        t.setSingleShot(True)

        def run():
            self._timers.pop(tid, None)
            fn()
        t.timeout.connect(run)
        self._timers[tid] = t
        t.start(max(0, int(ms)))
        return tid

    def after_cancel(self, tid):
        t = self._timers.pop(tid, None)
        if t is not None:
            t.stop()

    def after_idle(self, fn):
        return self.after(0, fn)
