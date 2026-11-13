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


_FONT_CACHE: dict = {}


def tk_font(spec) -> QFont:
    """шрифт у записі Tk: ("Consolas", 10, "bold") -> QFont із тим самим пікселем-розміром (пункти Tk при 96 DPI: px = pt·96/72)"""
    key = tuple(spec) if isinstance(spec, (tuple, list)) else ("Consolas", 10)
    f = _FONT_CACHE.get(key)
    if f is None:
        fam, pt, style = key[0], key[1], key[2:]
        f = QFont(fam)
        f.setStyleHint(QFont.Monospace)
        f.setPixelSize(max(1, round(float(pt) * 96 / 72)))
        f.setBold("bold" in style)
        f.setHintingPreference(QFont.PreferFullHinting)
        f.setStyleStrategy(QFont.PreferDefault)
        _FONT_CACHE[key] = f
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
        QComboBox {{ border: 1px solid {fg}; padding: 3px 6px; background: {bg}; color: {fg}; selection-background-color: {fg}; selection-color: {bg}; }}
        QComboBox::drop-down {{ border: none; width: 18px; }}
        QComboBox QAbstractItemView {{ border: 1px solid {fg}; background: {bg}; color: {fg}; selection-background-color: {fg}; selection-color: {bg}; outline: 0; }}
        QScrollBar:vertical {{ background: {bg}; width: 12px; border: 1px solid {fg}; margin: 0; }}
        QScrollBar::handle:vertical {{ background: {fg}; min-height: 24px; margin: 2px; }}
        QScrollBar::handle:vertical:hover {{ background: {fg}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: {bg}; }}
        """


THEME = Theme()
