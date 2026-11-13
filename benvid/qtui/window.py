"""Головне вікно на Qt: розкладка перенесена з `benvid/gui.py` (`_build`, `_build_side`) з тими самими константами, відступами й віджетами,
що й Tk-версія, тож вигляд 1:1 (звірка: `tools/ui_diff.py screen`).

Стан: екран застосунку (шапка, смуги, екран кадра, права панель з чотирма секціями, перемотування й транспорт) і футер (`footer.py`) перенесено, логіка: відкриття файла,
відтворення, перемотування, рівні, «Чорнило/Символи», розміри, палітри, мова, експорт. Решта чеклиста - у `docs/QT_PORT.md`."""
from __future__ import annotations

import os
import sys
import time

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QFileDialog, QGridLayout, QHBoxLayout, QMainWindow, QMessageBox, QSizePolicy, QVBoxLayout, QWidget

from .. import i18n, palette
from ..core import BRANCH, MAIN_LEVELS, METHOD_INFO, METHODS, Params
from ..engine import load_settings, save_settings
from ..i18n import tn, tr
from ..media import GIF_EXT, GIF_FORMATS, IMAGE_EXT, still_formats
from ..player import Player
from ..progress import fmt_t
from ..video import FORMATS, QUALITY, SIZES, VIDEO_EXT, out_size
from . import assets
from .boxes import PixCombo, PixEntry, SectionBox
from .consts import (AUTO, BAR_H, EDGE_W, FONT, FONT_B, FONT_S, G, GG, H_L, H_M, H_S, HEAD_W, M, MIN_MAIN, SB_W, SIDE_W, TOG_W, W_SQ)
from .export import ExportController
from .footer import FooterMixin
from .labels import TkImage, TkLabel, TkStrip
from .pix import PixButton, PixProgress, PixScroll, PixSlider, QualityBar
from .stacks import PageStack, ScrollStack
from .surface import FrameSurface
from .theme import THEME
from .tkcompat import Frame

SIZE_KEYS = [AUTO] + list(SIZES)
Q_W = 88


def _hbox(parent=None, margins=(0, 0, 0, 0), spacing=0):
    lay = QHBoxLayout(parent)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return lay


def _vbox(parent=None, margins=(0, 0, 0, 0), spacing=0):
    lay = QVBoxLayout(parent)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return lay


def _fixed(b: PixButton, w: int) -> PixButton:
    b.setFixedWidth(w)
    return b


class MainWindow(FooterMixin, QMainWindow):
    TITLE = "InkReel"
    SEEK_GAP = 0.04

    @property
    def _tip_n(self) -> int:
        """номер поради: спільний для порожнього екрана (поверхня) і футера"""
        return self.surface._tip_n

    @_tip_n.setter
    def _tip_n(self, v: int):
        self.surface._tip_n = v

    def __init__(self, path: str | None = None):
        super().__init__()
        ui = (load_settings() or {}).get("ui", {})
        i18n.set_lang(ui.get("lang") or i18n.lang())
        self.ui = ui
        self.params = Params()
        try:
            m0 = min(BRANCH, max(1, int(ui.get("method", 4))))
        except (TypeError, ValueError):
            m0 = 4
        self.params.method = m0
        self.params.glyph = bool(ui.get("glyph", False))
        if self.params.glyph and m0 > MAIN_LEVELS:
            self.params.method = MAIN_LEVELS
        try:
            self.params.gdens = min(3, max(1, int(ui.get("gdens3", 3))))
        except (TypeError, ValueError):
            self.params.gdens = 3
        self.pal_name = palette.split(ui.get("palette"))[0]
        if self.pal_name not in palette.NAMES:
            self.pal_name = palette.CLASSIC
        self.pal_inv = bool(ui.get("pal_inv"))
        sz = ui.get("size", AUTO)
        self.size_key = sz if sz in SIZE_KEYS else AUTO
        self.player = Player()
        self.player.params = self.params
        self.ready = False
        self.kind = "video"
        self.path = ""
        self._name = ""
        self._seek_t, self._seek_pending = 0.0, None
        self._last_info = 0.0
        self._exp_msg = ("", {})
        self.setWindowTitle(self.TITLE)
        self.resize(1280, 860)
        THEME.set_key(self._pal_key())
        self._build()
        self._apply_theme()
        self._tr()
        self.export = ExportController(self)
        self.export.progress.connect(self._on_progress)
        self.export.live.connect(self._on_live)
        self.export.finished.connect(self._on_export_done)
        self.tick = QTimer(self)
        self.tick.timeout.connect(self._tick)
        self.tick.start(8)
        QShortcut(QKeySequence("Space"), self, activated=self._toggle)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self._open)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self._step(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self._step(1))
        QShortcut(QKeySequence("I"), self, activated=self._toggle_inv)
        QShortcut(QKeySequence(Qt.Key_F1), self, activated=self._goto_guide)
        QShortcut(QKeySequence(Qt.Key_PageUp), self, activated=lambda: None if self._typing() else self.pstack.step(-1))
        QShortcut(QKeySequence(Qt.Key_PageDown), self, activated=lambda: None if self._typing() else self.pstack.step(1))
        self._wheel_acc = 0
        if path and os.path.isfile(path):
            QTimer.singleShot(50, lambda: self._open(path))

    # ------------------------------------------------------------------------------ розкладка (як App._build)
    def _build(self):
        self.page = Frame(self)
        self.setCentralWidget(self.page)
        self.main = Frame(self.page)
        self.foot = Frame(self.page)
        self.page_bar = PixScroll(self.page, SB_W)
        self.pstack = PageStack(self.page, self.page_bar, [(self.main, lambda w, h: max(h, MIN_MAIN), None),
                                                           (self.foot, lambda w, h: max(h, self.foot.winfo_reqheight()), self._foot_fit)])
        main = _vbox(self.main)

        # шапка: маскот, логотип, меню; праворуч [ім'я файлу] [?] [UA|EN]
        top = Frame(self.main)
        tl = _hbox(top, (M, M, M, 0))
        self.lbl_mascot = TkImage(top, "mascot_w")
        self.lbl_logo = TkImage(top, "logo_w")
        tl.addWidget(self.lbl_mascot, 0, Qt.AlignVCenter)
        tl.addSpacing(M)
        tl.addWidget(self.lbl_logo, 0, Qt.AlignVCenter)
        tl.addSpacing(GG)
        self.btn_open = _fixed(PixButton(top, "", assets.img("open_w"), self._open, width=HEAD_W), HEAD_W)
        self.btn_export = _fixed(PixButton(top, "", assets.img("save_w"), self._export, width=HEAD_W), HEAD_W)
        self.btn_stop = _fixed(PixButton(top, "", assets.img("x_w"), lambda: self.export.stop(), width=HEAD_W), HEAD_W)
        for b in (self.btn_open, self.btn_export, self.btn_stop):
            tl.addWidget(b, 0, Qt.AlignVCenter)
            tl.addSpacing(M)
        self.btn_stop.set_enabled(False)
        self.title_lbl = TkLabel(top, "", dim=True, font=FONT, anchor="e")
        self.title_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        tl.addSpacing(0)
        tl.addWidget(self.title_lbl, 1, Qt.AlignVCenter)
        tl.addSpacing(M)
        self.btn_help = _fixed(PixButton(top, "?", command=self._goto_guide, width=W_SQ, height=H_M), W_SQ)
        self.btn_ua = _fixed(PixButton(top, "UA", command=lambda: self._set_lang(i18n.UK), width=W_SQ, height=H_M), W_SQ)
        self.btn_en = _fixed(PixButton(top, "EN", command=lambda: self._set_lang(i18n.EN), width=W_SQ, height=H_M, join="l"), W_SQ)
        tl.addWidget(self.btn_help, 0, Qt.AlignVCenter)
        tl.addSpacing(M)
        tl.addWidget(self.btn_ua, 0, Qt.AlignVCenter)
        tl.addWidget(self.btn_en, 0, Qt.AlignVCenter)
        main.addWidget(top)

        strip1 = Frame(self.main)
        _vbox(strip1, (M, M, M, M)).addWidget(TkStrip(strip1, "dither_light", 8))
        main.addWidget(strip1)

        # центр: екран у рамці 1 px + права панель
        mid = Frame(self.main)
        ml = _hbox(mid, (M, 0, M, 0))
        frame = Frame(mid, role="fg")
        _vbox(frame, (1, 1, 1, 1))
        self.surface = FrameSurface(frame)
        frame.layout().addWidget(self.surface)
        ml.addWidget(frame, 1)
        ml.addSpacing(M)
        self.side = Frame(mid)
        self.side.setFixedWidth(SIDE_W)
        ml.addWidget(self.side)
        self.side_in = Frame(self.side)
        self.side_bar = PixScroll(self.side, SB_W)
        self.sstack = ScrollStack(self.side, self.side_bar, [(self.side_in, lambda w, h: max(self.side_in.winfo_reqheight(), h), None)], gap=M)
        self._build_side()
        main.addWidget(mid, 1)

        strip2 = Frame(self.main)
        _vbox(strip2, (M, M, M, 0)).addWidget(TkStrip(strip2, "film_tile", 12))
        main.addWidget(strip2)

        # низ: перемотування й транспорт
        bot = Frame(self.main)
        bl = _vbox(bot, (M, M, M, M))
        self.seek = PixSlider(bot, 0, 1, command=self._on_seek_drag, on_release=self._on_seek_end, height=34)
        bl.addWidget(self.seek)
        bl.addSpacing(M)
        row = Frame(bot)
        rl = _hbox(row)
        self.tr = Frame(row)
        tl2 = _hbox(self.tr)
        self.btn_rew = PixButton(self.tr, icon="rew", command=lambda: None, width=48, height=H_L)
        self.btn_prev = PixButton(self.tr, icon="prev", command=lambda: self._step(-1), width=48, height=H_L)
        self.btn_play = PixButton(self.tr, icon="play", command=self._toggle, width=64, height=H_L)
        self.btn_stopp = PixButton(self.tr, icon="stop", command=lambda: None, width=48, height=H_L)
        self.btn_next = PixButton(self.tr, icon="next", command=lambda: self._step(1), width=48, height=H_L)
        self.btn_fwd = PixButton(self.tr, icon="fwd", command=lambda: None, width=48, height=H_L)
        for i, (b, w) in enumerate(((self.btn_rew, 48), (self.btn_prev, 48), (self.btn_play, 64), (self.btn_stopp, 48), (self.btn_next, 48), (self.btn_fwd, 48))):
            _fixed(b, w)
            if i:
                tl2.addSpacing(M)
            tl2.addWidget(b)
        rl.addWidget(self.tr)
        rl.addSpacing(GG)
        self.btn_snd = _fixed(PixButton(row, icon="snd", command=lambda: None, width=48, height=H_L), 48)
        self.vol = PixSlider(row, 0, 100, command=lambda v: None, height=26)
        self.vol.setFixedWidth(96)
        self.vol.set(100)
        rl.addWidget(self.btn_snd)
        rl.addSpacing(G)
        rl.addWidget(self.vol, 0, Qt.AlignVCenter)
        rl.addSpacing(GG)
        self.btn_loop = _fixed(PixButton(row, "", command=lambda: None, width=72, height=H_L, toggle=True), 72)
        rl.addWidget(self.btn_loop)
        rl.addSpacing(GG)
        self.btn_photo = _fixed(PixButton(row, "", command=lambda: None, width=152, height=H_L), 152)
        rl.addWidget(self.btn_photo)
        rl.addSpacing(GG)
        self.time_lbl = TkLabel(row, "0:00 / 0:00", font=FONT_B, anchor="w")
        rl.addWidget(self.time_lbl, 0, Qt.AlignVCenter)
        rl.addSpacing(M)
        self.stat_lbl = TkLabel(row, "", dim=True, font=FONT_S, anchor="e")
        self.stat_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        rl.addWidget(self.stat_lbl, 1, Qt.AlignVCenter)
        bl.addWidget(row)
        main.addWidget(bot)
        self._build_footer()

    def _section(self, parent_lay, last=False):
        s = SectionBox("", None, last=last)
        parent_lay.addWidget(s, 1 if last else 0)
        if not last:
            parent_lay.addSpacing(M)
        return s

    def _build_side(self):
        lay = _vbox(self.side_in)
        wrap0 = SIDE_W - 2 - 2 * M
        # --- якість
        s = self._section(lay)
        self.sec_q = s
        b = s.body
        row = Frame(s.inner)
        rl = _hbox(row)
        self.btn_ink = PixButton(row, "", command=lambda: self._set_glyph(False), height=H_S)
        self.btn_sym = PixButton(row, "", command=lambda: self._set_glyph(True), height=H_S, join="l")
        for w in (self.btn_ink, self.btn_sym):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            rl.addWidget(w, 1)
        self.btn_ink.set_on(not self.params.glyph)
        self.btn_sym.set_on(self.params.glyph)
        b.addWidget(row)
        b.addSpacing(M)
        self.m_slider = QualityBar(s.inner, MAIN_LEVELS, self.params.method, command=self._set_level, branch="IRNM1")
        self.m_slider.set_branch(not self.params.glyph)
        b.addWidget(self.m_slider)
        b.addSpacing(G)
        self.m_info = TkLabel(s.inner, "", dim=True, font=FONT_S, anchor="nw", lines=2, wraplength=wrap0, autowrap=True)
        self.m_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        b.addWidget(self.m_info)
        self.dens_box = Frame(s.inner)
        dl = _vbox(self.dens_box)
        self.dens_lbl = TkLabel(self.dens_box, "", font=FONT_S, anchor="nw", lines=2, autowrap=True)
        dl.addSpacing(G)
        dl.addWidget(self.dens_lbl)
        self.dens_bar = QualityBar(self.dens_box, 3, self.params.gdens, command=self._apply_gdens, ends=("крупно", "дрібно"))
        dl.addWidget(self.dens_bar)
        b.addWidget(self.dens_box)
        self.dens_box.setVisible(self.params.glyph)

        # --- розмір
        s = self._section(lay)
        self.sec_size = s
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(G)
        grid.setVerticalSpacing(G)
        self.size_btns = {}
        for i, k in enumerate(SIZE_KEYS):
            bt = PixButton(s.inner, "", command=lambda k=k: self._set_size(k), width=48, height=H_S)
            bt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            r, c = divmod(i, 4)
            grid.addWidget(bt, r, c)
            self.size_btns[k] = bt
        for c in range(4):
            grid.setColumnStretch(c, 1)
        b = s.body
        b.addLayout(grid)
        b.addSpacing(M)
        row = Frame(s.inner)
        rl = _hbox(row)
        self.lbl_custom = TkLabel(row, "", font=FONT_S, anchor="w")
        rl.addWidget(self.lbl_custom, 0, Qt.AlignVCenter)
        rl.addSpacing(G)
        self.ent_w, self.ent_h = PixEntry(row), PixEntry(row)
        rl.addWidget(self.ent_w, 1)
        rl.addSpacing(G)
        rl.addWidget(TkLabel(row, "×", font=FONT_S), 0, Qt.AlignVCenter)
        rl.addSpacing(G)
        rl.addWidget(self.ent_h, 1)
        rl.addSpacing(G)
        self.btn_ok = PixButton(row, "OK", command=lambda: None, width=70, height=H_S)
        self.btn_ok.setFixedWidth(70)
        rl.addWidget(self.btn_ok)
        b.addWidget(row)
        b.addSpacing(G)
        self.size_lbl = TkLabel(s.inner, "", dim=True, font=FONT_S, anchor="nw", lines=2, wraplength=wrap0, autowrap=True)
        self.size_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        b.addWidget(self.size_lbl)

        # --- палітра
        s = self._section(lay)
        self.sec_pal = s
        prow = Frame(s.inner)
        pl = _hbox(prow)
        self.btn_pprev = _fixed(PixButton(prow, "", command=lambda: self._step_palette(-1), width=W_SQ, height=H_M, icon="left"), W_SQ)
        self.pal_sw = PaletteSwatch(prow)
        self.pal_lbl = TkLabel(prow, "", font=FONT, anchor="w")
        self.btn_pnext = _fixed(PixButton(prow, "", command=lambda: self._step_palette(1), width=W_SQ, height=H_M, icon="right"), W_SQ)
        self.btn_pinv = _fixed(PixButton(prow, "", command=self._toggle_inv, width=W_SQ, height=H_M, icon="swap"), W_SQ)
        self.btn_pinv.set_on(self.pal_inv)
        pl.addWidget(self.btn_pprev)
        pl.addSpacing(M)
        pl.addWidget(self.pal_sw, 0, Qt.AlignVCenter)
        pl.addSpacing(M)
        pl.addWidget(self.pal_lbl, 1, Qt.AlignVCenter)
        pl.addSpacing(M)
        pl.addWidget(self.btn_pinv)
        pl.addSpacing(G)
        pl.addWidget(self.btn_pnext)
        s.body.addWidget(prow)

        # --- експорт
        s = self._section(lay, last=True)
        self.sec_exp = s
        b = s.body
        row = Frame(s.inner)
        rl = _hbox(row)
        self.fmt_cb, self.q_cb = PixCombo(row), PixCombo(row)
        self.q_cb.setFixedWidth(Q_W)
        rl.addWidget(self.fmt_cb, 1)
        rl.addSpacing(G)
        rl.addWidget(self.q_cb)
        b.addWidget(row)
        b.addSpacing(M)
        self.row2 = Frame(s.inner)
        r2 = _hbox(self.row2)
        self.btn_aud = _fixed(PixButton(self.row2, "", command=lambda: None, width=TOG_W, height=H_S, toggle=True), TOG_W)
        self.btn_aud.set_on(bool(self.ui.get("audio", True)))
        self.rng_lbl = TkLabel(self.row2, "", font=FONT_S, anchor="w")
        self.btn_rng = _fixed(PixButton(self.row2, "", command=lambda: None, width=TOG_W, height=H_S), TOG_W)
        r2.addWidget(self.btn_aud)
        r2.addSpacing(M)
        r2.addWidget(self.rng_lbl, 1, Qt.AlignVCenter)
        r2.addSpacing(M)
        r2.addWidget(self.btn_rng)
        b.addWidget(self.row2)
        b.addSpacing(M)
        self.fps_cb = PixCombo(s.inner, editable=True)
        self.row_fps = self.fps_cb
        b.addWidget(self.fps_cb)
        b.addSpacing(M)
        self.btn_export2 = PixButton(s.inner, "", assets.img("save_w"), self._export, width=100, height=H_M)
        self.btn_export2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        b.addWidget(self.btn_export2)
        b.addSpacing(M)
        self.btn_lock = PixButton(s.inner, "", command=lambda: None, width=100, height=H_M)
        self.btn_lock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        b.addWidget(self.btn_lock)
        b.addSpacing(G)
        self.est_lbl = TkLabel(s.inner, "", dim=True, font=FONT_S, anchor="w")
        b.addWidget(self.est_lbl)
        b.addSpacing(G)
        self.prog = PixProgress(s.inner, height=14)
        b.addWidget(self.prog)
        b.addSpacing(G)
        self.prog_lbl = TkLabel(s.inner, "", dim=True, font=FONT_S, anchor="nw", lines=3, wraplength=wrap0, autowrap=True)
        self.prog_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        b.addWidget(self.prog_lbl, 1)

    # ------------------------------------------------------------------------------ тема й переклад
    def _pal_key(self) -> str:
        return palette.key(self.pal_name, self.pal_inv)

    def _apply_theme(self):
        THEME.set_key(self._pal_key())
        self.surface.set_palette(self._pal_key())
        self.pal_sw.update()

    def _tr(self):
        L = i18n.lang()
        self.btn_open.set_text(tr("Відкрити"))
        self.btn_export.set_text(tr({"video": "Зберегти…", "gif": "Зберегти GIF", "image": "Зберегти фото"}[self.kind]))
        self.btn_stop.set_text(tr("Стоп"))
        self.btn_export2.set_text(tr({"video": "Зберегти…", "gif": "Зберегти GIF", "image": "Зберегти фото"}[self.kind]))
        self.btn_ink.set_text(tr("Чорнило"))
        self.btn_sym.set_text(tr("Символи"))
        self.btn_ink.set_on(not self.params.glyph)
        self.btn_sym.set_on(self.params.glyph)
        self.btn_ua.set_on(L == i18n.UK)
        self.btn_en.set_on(L == i18n.EN)
        self.sec_q.title.config(text=(tr("Якість") + "  ·  " + tr(METHODS[self.params.method]) + (f"  {self.params.method}/5" if self.params.method <= MAIN_LEVELS else "")).upper())
        self.sec_size.title.config(text=tr("Розмір результату").upper())
        self.sec_pal.title.config(text=tr("Палітра").upper())
        self.sec_exp.title.config(text=tr("Експорт").upper())
        self.m_info.config(text=tr(METHOD_INFO[self.params.method]))
        self.m_slider.set(self.params.method)
        self.m_slider.draw()
        self.m_slider.update()
        self.lbl_custom.config(text=tr("свій:"))
        self.btn_lock.set_text(tr("Порахувати все"))
        self.btn_aud.set_text(tr("Звук"))
        self.btn_rng.set_text(tr("Весь"))
        self.btn_loop.set_text(tr("Цикл"))
        self.btn_photo.set_text(tr("Кадр → фото"))
        for k, b in self.size_btns.items():
            b.set_text(tr("Авто") if k == AUTO else (tr("Джерело") if k == "Як у джерела" else k.replace("x", "×")))
            b.set_on(k == self.size_key)
        self._fill_formats(keep=True)
        self._sync_fps()
        self._update_range()
        self.surface.rebuild_empty()
        self._fill_footer()
        if self._exp_msg[0]:
            self.prog_lbl.config(text=tr(self._exp_msg[0], **self._exp_msg[1]))
        self.pal_lbl.config(text=tr(self.pal_name))
        self.title_lbl.config(text=self._name if self.ready else tr("— файл не відкрито —"))
        self.setWindowTitle(self.TITLE + (f" — {self._name}" if self.ready else ""))

    def _src_fps(self) -> float:
        try:
            return float(self.player.fps or 0) if self.kind == "video" else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def _sync_fps(self):
        """список «кадрів на секунду»: як в оригіналі й стандартні значення нижче за частоту відео (як App._sync_fps)"""
        from ..video import fps_options
        src = self._src_fps()
        orig = tr("Кадри: як в оригіналі ({f:g} к/с)", f=round(src, 3)) if src else tr("Кадри: як в оригіналі")
        vals = [orig] + [tr("Кадри: {f} к/с", f=f"{v:g}") for v in fps_options(src)]
        self.fps_cb.blockSignals(True)
        self.fps_cb.clear()
        self.fps_cb.addItems(vals)
        self.fps_cb.setCurrentIndex(0)
        self.fps_cb.blockSignals(False)

    def _update_range(self):
        self.rng_lbl.config(text=tr("Фрагмент: усе відео\n[ ] — межі фрагмента"))
        self.btn_rng.set_enabled(False)

    def _formats(self) -> dict:
        if self.kind == "video":
            return {k: v[0] for k, v in FORMATS.items()}
        if self.kind == "gif":
            return dict(GIF_FORMATS)
        return dict(still_formats(self.params.glyph))

    def _fill_formats(self, keep: bool = False):
        fmts = list(self._formats())
        cur = self.fmt_cb.currentData() if keep else None
        self.fmt_cb.blockSignals(True)
        self.fmt_cb.clear()
        for k in fmts:
            self.fmt_cb.addItem(tr(k), k)
        want = cur if cur in fmts else (self.ui.get("fmt") if self.kind == "video" and self.ui.get("fmt") in fmts else fmts[0])
        self.fmt_cb.setCurrentIndex(fmts.index(want))
        self.fmt_cb.blockSignals(False)
        self.q_cb.blockSignals(True)
        cq = self.q_cb.currentData() if keep else None
        self.q_cb.clear()
        for k in QUALITY:
            self.q_cb.addItem(tr(k), k)
        qk = list(QUALITY)
        self.q_cb.setCurrentIndex(qk.index(cq if cq in qk else (self.ui.get("quality") if self.ui.get("quality") in qk else qk[0])))
        self.q_cb.blockSignals(False)
        self.q_cb.setVisible(self.kind == "video")

    def _set_lang(self, code: str):
        if code != i18n.lang():
            i18n.set_lang(code)
            self._tr()
            self.pstack.layout()
            self._save()

    def _save(self):
        d = dict(self.ui)
        d.update(palette=self.pal_name, pal_inv=bool(self.pal_inv), method=self.params.method, glyph=bool(self.params.glyph), size=self.size_key,
                 lang=i18n.lang(), gdens3=int(self.params.gdens), guide=self.guide_sel)
        try:
            save_settings({"ui": d})
        except Exception:  # noqa: BLE001
            pass

    def _save_later(self):
        if not hasattr(self, "_save_timer"):
            self._save_timer = QTimer(self)
            self._save_timer.setSingleShot(True)
            self._save_timer.timeout.connect(self._save)
        self._save_timer.start(500)

    # ------------------------------------------------------------------------------ керування
    def _set_level(self, n: int):
        self.params.method = n
        if n == BRANCH:
            self.params.glyph = False
        self._tr()
        self._reparam()
        self._save()

    def _apply_gdens(self, n: int):
        self.params.gdens = n
        self._reparam()
        self._save()

    def _set_glyph(self, v: bool):
        self.params.glyph = v
        if v and self.params.method == BRANCH:
            self.params.method = MAIN_LEVELS
        self.m_slider.set_branch(not v)
        self.dens_box.setVisible(v)
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
        self.btn_pinv.set_on(self.pal_inv)
        self._palette_changed()

    def _palette_changed(self):
        t0 = time.perf_counter()
        self._apply_theme()
        self.pal_lbl.config(text=tr(self.pal_name))
        self.last_palette_ms = (time.perf_counter() - t0) * 1000
        self._save_later()

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
        self._show_app()
        self.ready = True
        self.path = path
        self.kind = info["kind"]
        self._fill_formats()
        self._name = os.path.basename(path)
        self.player.set_params(self.params)
        self.surface.reset_view()
        self.surface.set_mask(None)
        self.seek.lo, self.seek.hi = 0, max(1, info["frames"] - 1)
        self.seek.set(0)
        self.btn_play.set_icon("play")
        self._tr()

    def _toggle(self):
        if self.export.running:
            if self.kind == "video":
                self.export.toggle_pause()
                self._on_progress(self.export.f, self.export.label, self.export.clock.elapsed() if self.export.clock else 0.0, None)
            return
        if self.ready:
            self.player.toggle()

    def _step(self, d: int):
        if self.ready:
            self.player.step(d)

    def _on_seek_drag(self, v: float):
        self._seek_pending = v
        if self.ready and time.perf_counter() - self._seek_t >= self.SEEK_GAP:
            self._seek_t = time.perf_counter()
            self.player.seek(int(v))
            self._seek_pending = None

    def _on_seek_end(self, v: float):
        self._seek_pending = None
        if self.ready:
            self.player.seek(int(v))

    # ------------------------------------------------------------------------------ експорт
    def _size_key(self) -> str:
        return "720p" if self.size_key == AUTO else self.size_key

    def _export(self):
        if not self.ready or self.export.running:
            return
        from ..video import FORMATS as VF
        kind, fmt = self.kind, self.fmt_cb.currentData()
        base = os.path.splitext(os.path.basename(self.path))[0] + "_bw"
        if kind == "video":
            ext, codec = VF[fmt]
        elif kind == "gif":
            ext, codec = GIF_FORMATS[fmt], ("png" if fmt == "PNG-послідовність" else "gif")
        else:
            ext, codec = still_formats(True)[fmt], "still"
        src_dir = os.path.dirname(self.path)
        if codec == "png":
            out = QFileDialog.getExistingDirectory(self, tr("Тека для PNG-кадрів"), src_dir)
            out = os.path.join(out, base) if out else ""
        else:
            out, _ = QFileDialog.getSaveFileName(self, tr("Зберегти як"), os.path.join(src_dir, base + ext), tr(fmt) + " (*" + ext + ")")
        if not out:
            return
        if os.path.normcase(os.path.abspath(out)) == os.path.normcase(os.path.abspath(self.path)):
            self._say("Не можна зберегти поверх оригіналу — вибери інше ім'я")
            return
        self._show_app()
        pl = self.player
        pl.pause()
        self.btn_play.set_icon("play")
        size_key = self._size_key()
        cache = None
        if kind == "video" and not pl.baked:
            psize, snap = pl.snapshot()
            if psize == out_size(pl.info["width"], pl.info["height"], size_key):
                cache = snap
        pl.suspend(True)
        for b in (self.btn_export, self.btn_export2, self.btn_open):
            b.set_enabled(False)
        self.btn_stop.set_enabled(True)
        self._say("Обробка…")
        self.export.start(kind=kind, path=self.path, out=out, params=self.params, fmt=fmt, size_key=size_key, quality=self.q_cb.currentData() or "Висока",
                          pal_name=self._pal_key(), audio=self.btn_aud.on, cache=cache, src_fps=pl.fps or 24.0, n_frames=pl.n)
        self._last_out = out

    def _say(self, key: str = "", **kw):
        self._exp_msg = (key, kw)
        self.prog_lbl.config(text=tr(key, **kw) if key else "")

    def _on_progress(self, f, label, el, eta):
        if not self.export.running:
            return
        self.export.f, self.export.label = f, label
        self.prog.set(f)
        self.prog_lbl.config(text=self.export.status(f, label, el, eta, tr))

    def _on_live(self, mask):
        if self.export.running and not self.export.pause_ev.is_set():
            self.surface.set_mask(mask)

    def _on_export_done(self, key, f, ok, kw):
        pl = self.player
        for fn in (lambda: pl.suspend(False), pl.refresh):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
        self.prog.set(f)
        if "frames" in kw:
            kw["frames"] = tn(kw["frames"], "{n} кадр|{n} кадри|{n} кадрів")
        self._say(key, **kw)
        for b in (self.btn_export, self.btn_export2, self.btn_open):
            b.set_enabled(True)
        self.btn_stop.set_enabled(False)

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
            if not self.seek.drag:
                self.seek.set(pos)
            if pl.kind == "image":
                self.time_lbl.config(text=tr("ФОТО {w}×{h}", w=pl.info["width"], h=pl.info["height"]))
            else:
                self.time_lbl.config(text=f"{fmt_t(pl.time_of(pos))} / {fmt_t(pl.duration)}")
            self.btn_play.set_icon("pause" if pl.state == "playing" else "play")
            rs = []
            k = max(1.0, (pl.n - 1) / max(1, self.seek.width() - 16))
            for a, b in pl.buffered_ranges():
                if rs and a - rs[-1][1] <= k:
                    rs[-1][1] = b
                else:
                    rs.append([a, b])
            self.seek.set_ranges([tuple(r) for r in rs])
            st = {"playing": "відтворення", "buffering": "буферизація…", "paused": "пауза"}[pl.state]
            self.stat_lbl.config(text=f"{tr(st)} · {tr('обробка {r:.0f} к/с', r=pl.render_fps())} · {pl.workers}×")

    def _typing(self) -> bool:
        """фокус у полі введення: клавіші-літери й PgUp/PgDn тоді належать полю"""
        from PySide6.QtWidgets import QLineEdit
        return isinstance(QApplication.focusWidget(), QLineEdit)

    @staticmethod
    def _inside(w, root) -> bool:
        while w is not None:
            if w is root:
                return True
            w = w.parentWidget()
        return False

    def wheelEvent(self, e):
        """колесо - одне на все вікно (як App._on_wheel): над картинкою масштаб (її wheelEvent приймає подію), над правою панеллю, якщо вона
        не влазить, - її прокрутка, деінде - прокрутка сторінки до футера й назад"""
        self._wheel_acc += e.angleDelta().y()
        steps = int(self._wheel_acc / 120)               # дрібні кроки тачпада накопичуємо до цілого «клацання»
        if not steps:
            e.accept()
            return
        self._wheel_acc -= steps * 120
        sg = 1 if steps > 0 else -1
        w = QApplication.widgetAt(e.globalPosition().toPoint())
        for _ in range(abs(steps)):
            if self._inside(w, self.side) and self.sstack.overflow:
                self.sstack.notch(sg)
            elif self.pstack.enabled:
                self.pstack.notch(sg)
        e.accept()

    def resizeEvent(self, e):
        super().resizeEvent(e)

    def closeEvent(self, e):
        self.export.stop()
        self.export.join(3.0)
        self._save()
        try:
            self.player.close()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(e)


class PaletteSwatch(QWidget):
    """два квадратики кольорів палітри (ліворуч туш, праворуч папір), як App._draw_swatch: 30×18"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(30, 18)
        THEME.changed.connect(self.update)

    def paintEvent(self, e):
        from PySide6.QtGui import QColor, QPainter
        win = self.window()
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.bg)
        try:
            ink, paper = palette.colors(win._pal_key())
        except Exception:  # noqa: BLE001
            return
        for x0, x1, c in ((0, 14, ink), (15, 29, paper)):
            p.fillRect(x0 + 1, 1, x1 - x0 - 1, 16, QColor(*c))
            p.setPen(THEME.fg)
            p.drawRect(x0, 0, x1 - x0, 17)


def main(argv=None) -> int:
    import multiprocessing

    from PySide6.QtWidgets import QApplication
    multiprocessing.freeze_support()
    argv = sys.argv if argv is None else argv
    app = QApplication.instance() or QApplication(argv)
    w = MainWindow(argv[1] if len(argv) > 1 else None)
    w.show()
    return app.exec()
