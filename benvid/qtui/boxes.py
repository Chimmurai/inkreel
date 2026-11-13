"""Будівельні цеглини розкладки в стилі Tk: рамка-секція із смужкою заголовка, поле введення, випадний список (піксельні, 1:1 з виглядом Tk-версії)."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QSizePolicy, QVBoxLayout, QWidget

from .consts import BAR_H, FONT, FONT_S, H_S, M
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
    """поле введення: плоске, рамка 1 px кольором тексту, шрифт Consolas bold, текст по центру (як tk.Entry у gui.py)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(tk_font(FONT))
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(H_S)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        THEME.changed.connect(self._style)
        self._style()

    def _style(self):
        bg, fg = THEME.bg.name(), THEME.fg.name()
        self.setStyleSheet(f"QLineEdit {{ background: {bg}; color: {fg}; border: 1px solid {fg}; padding: 0; selection-background-color: {fg}; selection-color: {bg}; }}")


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
