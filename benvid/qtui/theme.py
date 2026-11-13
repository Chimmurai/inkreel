"""Тема інтерфейсу: два кольори палітри -> фарби вікна. Усе вікно перефарбовується одним сигналом `changed`, а не обходом віджетів
(у Tk це було ≈140 віджетів і ≈3000 викликів Tcl на крок палітри)."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QFont

from .. import palette

FONT_FAMILY = "Consolas"
SMALL, NORMAL, BIG = 12, 13, 16            # розміри шрифту в пікселях (не в пунктах: однакові на будь-якому DPI Windows)


def font(px: int = NORMAL, bold: bool = False) -> QFont:
    f = QFont(FONT_FAMILY)
    f.setStyleHint(QFont.Monospace)
    f.setPixelSize(px)
    f.setBold(bold)
    f.setHintingPreference(QFont.PreferFullHinting)
    return f


class Theme(QObject):
    changed = Signal()

    def __init__(self, key: str | None = None):
        super().__init__()
        self.key = key or palette.CLASSIC
        self._load()

    def _load(self):
        t = palette.theme(self.key)
        self.bg, self.fg, self.dim, self.panel = (QColor(t[k]) for k in ("BG", "FG", "DIM", "PANEL"))

    def set_key(self, key: str):
        if key == self.key:
            return
        self.key = key
        self._load()
        self.changed.emit()

    def css(self) -> str:
        """таблиця стилів для того, що Qt малює сам (смужки прокрутки, підказки): одна на всю програму"""
        bg, fg, dim = self.bg.name(), self.fg.name(), self.dim.name()
        return f"""
        QWidget {{ background: {bg}; color: {fg}; }}
        QToolTip {{ background: {bg}; color: {fg}; border: 1px solid {fg}; padding: 3px; }}
        QScrollArea {{ border: none; }}
        QScrollBar:vertical {{ background: {bg}; width: 12px; border: 1px solid {fg}; margin: 0; }}
        QScrollBar::handle:vertical {{ background: {fg}; min-height: 24px; margin: 2px; }}
        QScrollBar::handle:vertical:hover {{ background: {fg}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: {bg}; }}
        """


THEME = Theme()
