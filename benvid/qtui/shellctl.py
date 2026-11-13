"""Вікно як вікно (Qt): «кіно» (F), F11 і права панель-«язичок» `edge`, Escape, масштаб клавішами, перетягування файлу, одиночний запуск (`bring_front`),
значок, розмір вікна. Перенесено з `gui.App` (`_fullscreen`, `_window_fullscreen`, `_on_escape`, `_apply_side`, `_toggle_side`, `_remember_window`, `_leave_fullscreen_size`,
`_zoom_step/_zoom_reset/_zoom_100`, `_hook_drop`, `bring_front`, `_deliver`, `_set_icon`)."""
from __future__ import annotations

import os
import threading

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QGuiApplication, QIcon

from .. import log as _log
from ..i18n import tr
from . import assets


class ShellMixin:
    # ------------------------------------------------------------------ значок і початковий розмір
    def _set_icon(self):
        """значок вікна: .ico на Windows (заголовок, панель завдань, Alt+Tab), PNG - деінде"""
        try:
            ico = os.path.join(assets.ASSETS, "inkreel.ico")
            png = os.path.join(assets.ASSETS, "app_icon.png")
            icon = QIcon(ico) if os.name == "nt" and os.path.isfile(ico) else QIcon(png)
            self.setWindowIcon(icon)
            app = QGuiApplication.instance()
            if app is not None:
                app.setWindowIcon(icon)
        except Exception:  # noqa: BLE001
            pass

    def _initial_size(self):
        """маленькі екрани/ноутбуки: вікно не більше за екран"""
        sc = QGuiApplication.primaryScreen()
        g = sc.availableGeometry() if sc is not None else None
        sw, sh = (g.width(), g.height()) if g is not None else (1920, 1080)
        self.setMinimumSize(min(1000, sw - 40), min(640, sh - 90))
        self.resize(min(1280, sw - 40), min(860, sh - 90))

    # ------------------------------------------------------------------ «кіно», F11, права панель
    def toggle_cinema(self):
        self._fullscreen()

    def _remember_window(self):
        """запам'ятати, яким було вікно, щоб після виходу з повного екрана повернути нормальний розмір"""
        self._pre_fs = (bool(self.windowState() & Qt.WindowMaximized), self.width(), self.height(), self.x(), self.y())

    def _apply_fullscreen(self, was):
        """виставити стан вікна за _fs/_wfs: рамка зникає, якщо діє будь-який із двох режимів; вийшли з обох - нормальний розмір"""
        full = self._fs or self._wfs
        if full:
            self.showFullScreen()
        else:
            self.showNormal()
            if was:
                QTimer.singleShot(60, self._leave_fullscreen_size)
        self._apply_side()
        self.pstack.set_enabled(not self._fs)                # «кіно» - лише картинка: ні смужки, ні футера
        self.surface.refresh()

    def _fullscreen(self, v=None):
        """«кіно» (F, подвійний клік): на весь екран лише картинка"""
        new = (not self._fs) if v is None else v
        if new == self._fs:
            return
        was = self._fs or self._wfs
        if new and not was:
            self._remember_window()
        self._fs = new
        self._apply_fullscreen(was)

    def _window_fullscreen(self, v=None):
        """F11: програма розтягується на весь екран як є, з усім інтерфейсом; із «кіно» F11 виходить повністю"""
        if self._fs and v is None:
            self._fs = self._wfs = False
            self._apply_fullscreen(True)
            return
        new = (not self._wfs) if v is None else v
        if new == self._wfs:
            return
        was = self._fs or self._wfs
        if new and not was:
            self._remember_window()
        self._wfs = new
        self._side_open = False                              # у F11 панель спершу схована; кнопка справа її висуває
        self._apply_fullscreen(was)

    def _on_escape(self):
        if self._fs:
            self._fullscreen(False)
        elif self._wfs:
            self._window_fullscreen(False)

    def _apply_side(self):
        """права панель: у звичайному вікні завжди на місці; в F11 схована, а вузька кнопка на правому краю висуває її (і ховає назад);
        у «кіно» немає ні панелі, ні кнопки"""
        wfs = self._wfs and not self._fs
        show = not self._fs and (not self._wfs or self._side_open)
        self.side.setVisible(show)
        self.gap_side.on(show)
        self.edge.setVisible(wfs)
        self.gap_edge.on(wfs)
        if wfs:
            self.edge.set_icon("right" if self._side_open else "left")
        self.mid_lay.invalidate()
        self.mid_lay.activate()

    def _toggle_side(self):
        self._side_open = not self._side_open
        self._apply_side()
        QTimer.singleShot(30, self.surface.refresh)

    def _leave_fullscreen_size(self):
        """після виходу з повного екрана вікно не лишається розміром на весь екран (лише з рамкою): повертаємо розмір, який було до входу, а якщо він і так займав
        майже весь екран (або вікно було розгорнуте) - беремо звичайний розмір за розмірами монітора і ставимо по центру"""
        if self._fs or self._wfs:
            return
        sc = self.screen() or QGuiApplication.primaryScreen()
        g = sc.availableGeometry()
        sw, sh = g.width(), g.height()
        pre = getattr(self, "_pre_fs", None)
        if self.windowState() & Qt.WindowMaximized:
            self.showNormal()
        zoomed, w, h, x, y = pre if pre else (True, 0, 0, 0, 0)
        mw, mh = self.minimumWidth(), self.minimumHeight()
        if zoomed or w < 200 or h < 200 or w >= sw * 0.95 or h >= sh * 0.90:
            w, h = int(sw * 0.72), int(sh * 0.78)
            x = None
        w, h = max(mw or 1, min(w, sw)), max(mh or 1, min(h, sh))
        if x is None:
            x, y = g.x() + (sw - w) // 2, g.y() + max(0, (sh - h) // 2 - 20)
        self.setGeometry(max(g.x(), x), max(g.y(), y), w, h)
        self.surface.refresh()

    # ------------------------------------------------------------------ масштаб картинки клавішами
    def _zoom_step(self, d, mx=None, my=None):
        self.surface.zoom_step(d, mx, my)

    def _zoom_reset(self):
        self.surface.zoom_reset()

    def _zoom_100(self):
        self.surface.zoom_100()

    # ------------------------------------------------------------------ перетягування файлу у вікно
    def _init_drop(self):
        self.setAcceptDrops(True)

    def _drop_paths(self, e):
        md = e.mimeData()
        return [u.toLocalFile() for u in md.urls() if u.isLocalFile()] if md.hasUrls() else []

    def dragEnterEvent(self, e):
        if self._drop_paths(e):
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if self._drop_paths(e):
            e.acceptProposedAction()

    def dropEvent(self, e):
        self._got_files(self._drop_paths(e))
        e.acceptProposedAction()

    def _got_files(self, ps):
        """файли, кинуті у вікно: відкриваємо перший, текою чи під час експорту - підказка"""
        ps = [p for p in ps if os.path.isfile(p)]
        if not ps:
            self._say("Перетягни файл (відео, GIF або фото), а не теку")
        elif self.export.running:
            self._say("Іде експорт — спершу дочекайся завершення або натисни Стоп")
        else:
            self._open(ps[0])
            if len(ps) > 1:
                self._say("Відкрито перший із {n}", n=len(ps))

    # ------------------------------------------------------------------ одиночний запуск
    def bring_front(self, path=""):
        """інший запуск: показати це вікно й відкрити файл, якщо його передали"""
        try:
            if self.windowState() & Qt.WindowMinimized:
                self.setWindowState(self.windowState() & ~Qt.WindowMinimized)
            self.show()
            self.raise_()
            self.activateWindow()
        except Exception:  # noqa: BLE001
            pass
        if path and os.path.isfile(path) and not self.export.running:
            self._open(path)

    def _on_bring_front(self, path, ev):
        try:
            self.bring_front(path)
        finally:
            ev.set()

    def deliver(self, path, wait=2.0) -> bool:
        """файл, переданий повторним запуском: показати у вікні й підтвердити лише тоді, коли вікно справді його обробило (див. single.py).
        Викликається з потоку слухача: справа йде в потік інтерфейсу сигналом"""
        ev = threading.Event()
        self.bring_front_req.emit(path, ev)
        return ev.wait(wait)
