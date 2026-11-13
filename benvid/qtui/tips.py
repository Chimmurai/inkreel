"""Підказки при наведенні (`Tip`) і хмаринка пасхалок (`Bubble`): перенесено з `gui.Tip` / `App._bubble`.

Підказка з'являється за ~0,6 с, зникає, щойно курсор пішов чи натиснули кнопку миші або клавішу, не більше однієї одночасно, сама
зникає за 12 с. Вікно будується невидимим і з'являється вже на місці; місце рахується за монітором під курсором (`benvid.popup`).
Текст - рядок або функція без аргументів (тоді береться в момент показу: завжди актуальною мовою й станом)."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QTimer, Qt
from PySide6.QtGui import QCursor, QFontMetrics, QGuiApplication, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from .. import popup
from .consts import FONT_S, G, M
from .labels import wrap_text
from .theme import THEME, tk_font


def _area(pt: QPoint):
    """робоча область екрана під точкою як (лівий, верхній, правий, нижній) - у координатах Qt"""
    sc = QGuiApplication.screenAt(pt) or QGuiApplication.primaryScreen()
    g = sc.availableGeometry()
    return g.left(), g.top(), g.left() + g.width(), g.top() + g.height()


class Bubble(QWidget):
    """плаский напис у рамці 1 px (як tk.Label з highlightthickness=1, padx=M, pady=G), без фокусу й миші"""

    def __init__(self, text: str, wrap: int = 340):
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.fm = QFontMetrics(tk_font(FONT_S))
        self.lines = wrap_text(text, wrap, self.fm)
        tw = max((self.fm.horizontalAdvance(s) for s in self.lines), default=0)
        self.resize(tw + 2 * M + 2, len(self.lines) * self.fm.height() + 2 * G + 2)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.bg)
        p.setPen(THEME.fg)
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)
        p.setFont(tk_font(FONT_S))
        for k, s in enumerate(self.lines):
            p.drawText(1 + M, 1 + G + k * self.fm.height() + self.fm.ascent(), s)

    def place_near_pointer(self, pt: QPoint):
        x, y = popup.near_pointer(pt.x(), pt.y(), self.width(), self.height(), _area(pt))
        self.move(x, y)

    def place_under(self, widget: QWidget):
        tl = widget.mapToGlobal(QPoint(0, 0))
        c = QPoint(tl.x() + widget.width() // 2, tl.y() + widget.height() // 2)
        x, y = popup.under_widget(tl.x(), tl.y(), widget.width(), widget.height(), self.width(), self.height(), _area(c))
        self.move(x, y)


class _Global(QObject):
    """натискання клавіші чи втрата фокусу вікном прибирають підказку, що висить"""

    def eventFilter(self, o, e):
        if e.type() in (QEvent.KeyPress, QEvent.WindowDeactivate):
            Tip.hide_all()
        return False


class Tip(QObject):
    DELAY = 600
    HIDE = 12000
    _cur: "Tip | None" = None
    _glob: _Global | None = None

    def __init__(self, widget: QWidget, text):
        super().__init__(widget)
        self.w, self.text, self._win = widget, text, None
        self._job = QTimer(self)
        self._job.setSingleShot(True)
        self._job.timeout.connect(self._show)
        self._gone = QTimer(self)
        self._gone.setSingleShot(True)
        self._gone.timeout.connect(self._leave)
        widget.installEventFilter(self)
        if Tip._glob is None and QApplication.instance() is not None:
            Tip._glob = _Global(QApplication.instance())
            QApplication.instance().installEventFilter(Tip._glob)

    @classmethod
    def hide_all(cls):
        """прибрати підказку, що зараз висить (зміна вікна, фокус, натискання клавіші)"""
        t = cls._cur
        if t is not None:
            t._leave()

    def eventFilter(self, o, e):
        if o is self.w:
            t = e.type()
            if t == QEvent.Enter:
                self._leave()
                self._job.start(self.DELAY)
            elif t in (QEvent.Leave, QEvent.MouseButtonPress, QEvent.Hide, QEvent.Wheel):
                self._leave()
        return False

    def _leave(self):
        self._job.stop()
        self._gone.stop()
        if self._win is not None:
            self._win.close()
            self._win.deleteLater()
            self._win = None
        if Tip._cur is self:
            Tip._cur = None

    def _show(self):
        t = self.text() if callable(self.text) else self.text
        if not t or not self.w.isVisible():
            return
        Tip.hide_all()                                   # стара підказка (інший віджет) не лишається поруч із новою
        win = Bubble(t, 340)
        win.place_near_pointer(QCursor.pos())
        win.show()
        self._win = win
        Tip._cur = self
        self._gone.start(self.HIDE)
