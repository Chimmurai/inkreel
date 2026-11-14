"""Будівельні цеглини розкладки в стилі Tk: рамка-секція із смужкою заголовка, поле введення, випадний список (піксельні, 1:1 з виглядом Tk-версії)."""
from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QKeySequence, QPainter
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLineEdit, QSizePolicy, QVBoxLayout

from .consts import BAR_H, FONT, H_S, M
from .labels import TkLabel
from .theme import THEME, tk_font
from .tkcompat import Frame


class SectionBox(Frame):
    """tk.Frame з рамкою 1 px (highlightthickness=1, колір тексту): зверху смужка заголовка, нижче вміст із полем M"""

    def __init__(self, title_key="", parent=None, last=False):
        super().__init__(parent, role="bg")
        self.last = last
        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)
        self.bar = Frame(self, role="fg")
        self.bar.setFixedHeight(BAR_H)
        bl = QHBoxLayout(self.bar)
        bl.setContentsMargins(0, 0, 0, 0)
        self.title = TkLabel(self.bar, "", anchor="w", inverted=True, padx=M, font=FONT)
        self.title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        bl.addWidget(self.title)
        outer.addWidget(self.bar)
        self.inner = Frame(self)
        self.body = QVBoxLayout(self.inner)
        self.body.setContentsMargins(M, M, M, M)
        self.body.setSpacing(0)
        outer.addWidget(self.inner, 1 if last else 0)

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setPen(THEME.fg)
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)


class PixEntry(QLineEdit):
    """поле введення: плоске, рамка 1 px кольором тексту, шрифт Consolas bold, текст по центру (як колись tk.Entry)"""

    def __init__(self, parent=None, bare=False, select="all"):
        """bare - поле без власної рамки й заввишки (всередині списку); select: при фокусі виділяти «all» чи лише «value» (після двокрапки: «Кадри: 25 к/с»)"""
        super().__init__(parent)
        self.bare, self.select_mode = bare, select
        self.paste_hook = None                         # функція(поле): вставка ЗАМІНЮЄ вміст (див. SizeMixin._paste_replace)
        self.setFont(tk_font(FONT))
        if not bare:
            self.setAlignment(Qt.AlignCenter)
            self.setFixedHeight(H_S)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        THEME.changed.connect(self._style)
        self._style()

    def _style(self):
        bg, fg = THEME.bg.name(), THEME.fg.name()
        border = "none" if self.bare else f"1px solid {fg}"
        self.setStyleSheet(f"QLineEdit {{ background: {bg}; color: {fg}; border: {border}; padding: 0; selection-background-color: {fg}; selection-color: {bg}; }}")

    def select_value(self):
        """«Кадри: 25 к/с»: виділити лише те, що після двокрапки (його й замінюють набором чи вставкою) - як gui._select_value"""
        if self.select_mode != "value":
            return self.selectAll()
        t = self.text()
        i = t.find(":")
        a = i + 1 if i >= 0 else 0
        while a < len(t) and t[a] == " ":
            a += 1
        self.setSelection(a, len(t) - a)

    def focusInEvent(self, e):
        super().focusInEvent(e)
        QTimer.singleShot(1, self.select_value)          # клік у поле виділяє його вміст: набір чи вставка замінюють

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        self.deselect()

    def keyPressEvent(self, e):
        if self.paste_hook and (e.matches(QKeySequence.Paste) or (e.key() == Qt.Key_Insert and e.modifiers() & Qt.ShiftModifier)):
            self.paste_hook(self)
            e.accept()
            return
        super().keyPressEvent(e)

    def contextMenuEvent(self, e):
        pass                                             # як у Tk-версії контекстного меню нема


class PixCombo(QComboBox):
    """випадний список: рамка 1 px, текст Consolas bold, стрілка праворуч (стиль ttk «clam» у кольорах теми)"""

    def __init__(self, parent=None, editable=False):
        super().__init__(parent)
        self.setFont(tk_font(FONT))
        self.setFixedHeight(H_S)
        self.setEditable(editable)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if not editable:                                  # список «лише вибір»: після вибору фокус не лишається в ньому - F, M, L, I, пробіл і стрілки належать плеєру
            self.setFocusPolicy(Qt.ClickFocus)
            self.activated.connect(lambda *_: self.clearFocus())
        THEME.changed.connect(self._style)
        self._style()

    def _style(self):
        bg, fg = THEME.bg.name(), THEME.fg.name()
        self.setStyleSheet(
            f"QComboBox {{ background: {bg}; color: {fg}; border: 1px solid {fg}; padding: 0 4px; selection-background-color: {fg}; selection-color: {bg}; }}"
            f"QComboBox::drop-down {{ border: none; border-left: 1px solid {fg}; width: 18px; background: {bg}; }}"
            f"QComboBox::down-arrow {{ image: none; border-left: 5px solid {bg}; border-right: 5px solid {bg}; border-top: 6px solid {fg}; width: 0; height: 0; }}"
            f"QComboBox QAbstractItemView {{ background: {bg}; color: {fg}; border: 1px solid {fg}; selection-background-color: {fg}; selection-color: {bg}; outline: 0; }}")
