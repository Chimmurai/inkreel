"""Головне вікно на Qt: шапка, поверхня кадра, права панель (якість, розмір, палітра), смуга відтворення.

Стан цього кроку (вертикальний зріз): відкриття файла, відтворення, перемотування, рівні 1–6, «Чорнило/Символи», розміри, палітри з «навпаки»,
масштаб колесом і зсув, мови UA/EN. Ще НЕ перенесено з Tk: експорт, фрагмент [ ], режим замка .ikr, «Авто» розмір, футер з гайдом, підказки,
пасхалки, перетягування файлу, «Кадр → фото». Список і стан - у `docs/QT_PORT.md`."""
from __future__ import annotations

import os
import sys
import time

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

from .. import i18n, palette
from ..core import BRANCH, MAIN_LEVELS, METHOD_INFO, METHODS, Params
from ..engine import load_settings, save_settings
from ..i18n import tr
from ..media import GIF_EXT, IMAGE_EXT
from ..player import Player
from ..video import SIZES, VIDEO_EXT
from . import theme
from .surface import FrameSurface
from .theme import THEME, font
from .widgets import PixButton, PixSlider, QualityBar, Section

AUTO = "Авто"
SIZE_KEYS = [AUTO] + list(SIZES)


def fmt_t(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


class Label(QLabel):
    def __init__(self, text: str = "", dim: bool = False, px: int = theme.SMALL, bold: bool = False, parent=None):
        super().__init__(text, parent)
        self._dim = dim
        self.setFont(font(px, bold))
        THEME.changed.connect(self._retint)
        self._retint()

    def _retint(self):
        c = THEME.dim if self._dim else THEME.fg
        self.setStyleSheet(f"color: {c.name()}; background: transparent;")


class MainWindow(QMainWindow):
    TITLE = "InkReel"
    SEEK_GAP = 0.04

    def __init__(self, path: str | None = None):
        super().__init__()
        ui = (load_settings() or {}).get("ui", {})
        i18n.set_lang(ui.get("lang") or i18n.lang())
        self.ui = ui
        self.params = Params()
        self.params.method = int(ui.get("method", MAIN_LEVELS)) if str(ui.get("method", "")).isdigit() or isinstance(ui.get("method"), int) else MAIN_LEVELS
        self.params.glyph = bool(ui.get("glyph", False))
        self.pal_name = ui.get("palette") if ui.get("palette") in palette.NAMES else palette.CLASSIC
        self.pal_inv = bool(ui.get("pal_inv", False))
        self.size_key = ui.get("size") if ui.get("size") in SIZE_KEYS else AUTO
        self.player = Player()
        self.ready = False
        self._seek_t, self._seek_pending = 0.0, None
        self._last_info = 0.0
        self.setWindowTitle(self.TITLE)
        self.resize(1500, 900)
        self._build()
        self._apply_theme()
        self._tr()
        self.tick = QTimer(self)
        self.tick.timeout.connect(self._tick)
        self.tick.start(8)
        QShortcut(QKeySequence("Space"), self, activated=self._toggle)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self._open)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self._step(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self._step(1))
        QShortcut(QKeySequence("I"), self, activated=self._toggle_inv)
        if path and os.path.isfile(path):
            QTimer.singleShot(50, lambda: self._open(path))

    # ------------------------------------------------------------------------------ побудова
    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        left = QVBoxLayout()
        left.setSpacing(8)
        outer.addLayout(left, 1)

        top = QHBoxLayout()
        top.setSpacing(8)
        self.logo = Label("InkReel", px=theme.BIG + 6, bold=True)
        self.btn_open = PixButton("", min_w=120)
        self.btn_open.clicked.connect(lambda: self._open())
        self.title_lbl = Label("", dim=True, px=theme.NORMAL)
        self.title_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.btn_ua, self.btn_en = PixButton("UA", checkable=False, min_w=44), PixButton("EN", checkable=False, min_w=44)
        self.btn_ua.clicked.connect(lambda: self._set_lang("uk"))
        self.btn_en.clicked.connect(lambda: self._set_lang("en"))
        for w in (self.logo, self.btn_open):
            top.addWidget(w)
        top.addWidget(self.title_lbl, 1)
        top.addWidget(self.btn_ua)
        top.addWidget(self.btn_en)
        left.addLayout(top)

        self.surface = FrameSurface()
        self.surface.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.surface, 1)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.btn_play = PixButton("▶", min_w=44)
        self.btn_play.clicked.connect(self._toggle)
        self.seek = PixSlider()
        self.seek.valueChanged.connect(self._on_seek_drag)
        self.seek.released.connect(self._on_seek_end)
        self.time_lbl = Label("", px=theme.NORMAL, bold=True)
        self.time_lbl.setMinimumWidth(150)
        bar.addWidget(self.btn_play)
        bar.addWidget(self.seek, 1)
        bar.addWidget(self.time_lbl)
        left.addLayout(bar)
        self.stat_lbl = Label("", dim=True)
        left.addWidget(self.stat_lbl)

        # права панель
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(344)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 8, 0)
        sl.setSpacing(8)
        scroll.setWidget(side)
        outer.addWidget(scroll)
        self.scroll = scroll

        self.sec_q = Section()
        row = QHBoxLayout()
        self.btn_ink, self.btn_sym = PixButton("", min_w=90), PixButton("", min_w=90)
        self.btn_ink.clicked.connect(lambda: self._set_glyph(False))
        self.btn_sym.clicked.connect(lambda: self._set_glyph(True))
        row.addWidget(self.btn_ink)
        row.addWidget(self.btn_sym)
        self.sec_q.body.addLayout(row)
        self.qbar = QualityBar()
        self.qbar.chosen.connect(self._set_level)
        self.sec_q.body.addWidget(self.qbar)
        self.q_info = Label("", dim=True)
        self.q_info.setWordWrap(True)
        self.sec_q.body.addWidget(self.q_info)
        sl.addWidget(self.sec_q)

        self.sec_size = Section()
        grid = QGridLayout()
        grid.setSpacing(4)
        self.size_btns = {}
        for i, k in enumerate(SIZE_KEYS):
            b = PixButton(k, min_w=64, h=26)
            b.clicked.connect(lambda _=None, k=k: self._set_size(k))
            self.size_btns[k] = b
            grid.addWidget(b, i // 4, i % 4)
        self.sec_size.body.addLayout(grid)
        sl.addWidget(self.sec_size)

        self.sec_pal = Section()
        prow = QHBoxLayout()
        self.btn_pprev, self.btn_pnext = PixButton("◀", min_w=36), PixButton("▶", min_w=36)
        self.pal_lbl = Label("", px=theme.NORMAL, bold=True)
        self.pal_lbl.setAlignment(Qt.AlignCenter)
        self.btn_pinv = PixButton("⇄", min_w=36)
        self.btn_pprev.clicked.connect(lambda: self._step_palette(-1))
        self.btn_pnext.clicked.connect(lambda: self._step_palette(1))
        self.btn_pinv.clicked.connect(self._toggle_inv)
        for w in (self.btn_pprev, self.pal_lbl, self.btn_pnext, self.btn_pinv):
            prow.addWidget(w, 1 if w is self.pal_lbl else 0)
        self.sec_pal.body.addLayout(prow)
        sl.addWidget(self.sec_pal)
        sl.addStretch(1)

        self.todo = Label("", dim=True)
        self.todo.setWordWrap(True)
        sl.addWidget(self.todo)

    # ------------------------------------------------------------------------------ тема й переклад
    def _pal_key(self) -> str:
        return palette.key(self.pal_name, self.pal_inv)

    def _apply_theme(self):
        THEME.set_key(self._pal_key())
        self.setStyleSheet(THEME.css())
        self.surface.set_palette(self._pal_key())

    def _tr(self):
        L = i18n.lang()
        self.btn_open.setText(tr("Відкрити"))
        self.btn_ink.setText(tr("Чорнило"))
        self.btn_sym.setText(tr("Символи"))
        self.btn_ink.setOn(not self.params.glyph)
        self.btn_sym.setOn(self.params.glyph)
        self.btn_ua.setOn(L == i18n.UK)
        self.btn_en.setOn(L == i18n.EN)
        self.sec_q.setTitle(tr("Якість") + f" · {tr(METHODS[self.params.method])}")
        self.sec_size.setTitle(tr("Розмір результату"))
        self.sec_pal.setTitle(tr("Палітра"))
        self.pal_lbl.setText(tr(self.pal_name))
        self.q_info.setText(tr(METHOD_INFO[self.params.method]))
        self.qbar.setLevel(self.params.method)
        for k, b in self.size_btns.items():
            b.setText(tr(k) if k == AUTO else ("Джерело" if (k == "Як у джерела" and L == i18n.UK) else "Source" if k == "Як у джерела" else k))
            b.setOn(k == self.size_key)
        self.todo.setText(tr("Qt-інтерфейс 1.1, робоча версія: експорт, фрагмент, замок .ikr, «Авто» розмір, футер і гайд ще на Tk-версії."))
        self.setWindowTitle(self.TITLE + (f" — {self._name}" if self.ready else ""))
        self.title_lbl.setText(self._name if self.ready else tr("— файл не відкрито —"))

    _name = ""

    def _set_lang(self, code: str):
        if code != i18n.lang():
            i18n.set_lang(code)
            self._tr()
            self._save()

    def _save(self):
        d = dict(self.ui)
        d.update(palette=self.pal_name, pal_inv=bool(self.pal_inv), method=self.params.method, glyph=bool(self.params.glyph), size=self.size_key, lang=i18n.lang())
        try:
            save_settings({"ui": d})
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------------------ керування
    def _set_level(self, n: int):
        self.params.method = n
        if n == BRANCH:
            self.params.glyph = False
        self._tr()
        self._reparam()
        self._save()

    def _set_glyph(self, v: bool):
        self.params.glyph = v
        if v and self.params.method == BRANCH:
            self.params.method = MAIN_LEVELS
        self._tr()
        self._reparam()
        self._save()

    def _reparam(self):
        if self.ready:
            self.player.set_params(self.params)

    def _set_size(self, k: str):
        self.size_key = k
        self._tr()
        if self.ready:
            self.player.set_size("240p" if k == AUTO else k)
        self._save()

    def _step_palette(self, d: int):
        i = palette.NAMES.index(self.pal_name) if self.pal_name in palette.NAMES else 0
        self.pal_name = palette.NAMES[(i + d) % len(palette.NAMES)]
        self._palette_changed()

    def _toggle_inv(self):
        self.pal_inv = not self.pal_inv
        self._palette_changed()

    def _palette_changed(self):
        t0 = time.perf_counter()
        self._apply_theme()
        self.pal_lbl.setText(tr(self.pal_name))
        self.last_palette_ms = (time.perf_counter() - t0) * 1000
        self._save_later()

    def _save_later(self):
        if not hasattr(self, "_save_timer"):
            self._save_timer = QTimer(self)
            self._save_timer.setSingleShot(True)
            self._save_timer.timeout.connect(self._save)
        self._save_timer.start(500)

    def _open(self, path: str | None = None):
        if not path:
            exts = tuple(VIDEO_EXT) + tuple(IMAGE_EXT) + tuple(GIF_EXT)
            path, _ = QFileDialog.getOpenFileName(self, tr("Відкрити відео, GIF або фото"), self.ui.get("dir", ""),
                                                  tr("Усі медіа") + " (" + " ".join("*" + e for e in exts) + ");;" + tr("Усі файли") + " (*.*)")
        if not path:
            return
        try:
            info = self.player.open(path, "240p" if self.size_key == AUTO else self.size_key)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "InkReel", tr("Не вдалося відкрити файл:\n{e}\n\nДеталі: {p}", e=e, p=""))
            return
        self.ready = True
        self._name = os.path.basename(path)
        self.player.set_params(self.params)
        self.surface.reset_view()
        self.surface.set_mask(None)
        self.seek.setRange(0, max(1, info["frames"] - 1))
        self.seek.setValue(0)
        self.btn_play.setText("▶")
        self._tr()

    def _toggle(self):
        if self.ready:
            self.player.toggle()

    def _step(self, d: int):
        if self.ready:
            self.player.step(d)

    def _on_seek_drag(self, v: float):
        """не частіше за SEEK_GAP: кожен seek скасовує рендер, що ще йде (див. CHANGELOG 1.0.1)"""
        self._seek_pending = v
        if self.ready and time.perf_counter() - self._seek_t >= self.SEEK_GAP:
            self._seek_t = time.perf_counter()
            self.player.seek(int(v))
            self._seek_pending = None

    def _on_seek_end(self, v: float):
        self._seek_pending = None
        if self.ready:
            self.player.seek(int(v))

    # ------------------------------------------------------------------------------ такт
    def _tick(self):
        if not self.ready:
            return
        r = self.player.update()
        if r:
            self.surface.set_mask(r[1])
        now = time.perf_counter()
        if self._seek_pending is not None and now - self._seek_t >= self.SEEK_GAP:
            self._seek_t = now
            self.player.seek(int(self._seek_pending))
            self._seek_pending = None
        if now - self._last_info > 0.15:
            self._last_info = now
            pl = self.player
            pos = pl.position()
            if not self.seek.dragging:
                self.seek.setValue(pos)
            if pl.kind == "image":
                self.time_lbl.setText(tr("ФОТО {w}×{h}", w=pl.info["width"], h=pl.info["height"]))
            else:
                self.time_lbl.setText(f"{fmt_t(pl.time_of(pos))} / {fmt_t(pl.duration)}")
            self.btn_play.setText("❚❚" if pl.state == "playing" else "▶")
            rs = []
            k = max(1.0, (pl.n - 1) / max(1, self.seek.width() - 16))
            for a, b in pl.buffered_ranges():
                if rs and a - rs[-1][1] <= k:
                    rs[-1][1] = b
                else:
                    rs.append([a, b])
            self.seek.setRanges(rs)
            st = {"playing": "відтворення", "buffering": "буферизація…", "paused": "пауза"}[pl.state]
            self.stat_lbl.setText(f"{tr(st)} · {tr('обробка {r:.0f} к/с', r=pl.render_fps())} · {pl.workers}×")

    def closeEvent(self, e):
        self._save()
        try:
            self.player.close()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(e)


def main(argv=None) -> int:
    import multiprocessing

    from PySide6.QtWidgets import QApplication
    multiprocessing.freeze_support()
    argv = sys.argv if argv is None else argv
    app = QApplication.instance() or QApplication(argv)
    w = MainWindow(argv[1] if len(argv) > 1 else None)
    w.show()
    return app.exec()
