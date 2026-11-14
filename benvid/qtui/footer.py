"""Футер (другий екран сторінки) на Qt: перенесено з `App._build_footer` та супутніх методів `benvid/gui.py` з тими самими відступами,
шрифтами й розрахунком ширини (`_foot_fit`). Підмішується до `MainWindow`.

Не перенесено (у чеклисті `docs/QT_PORT.md`): підказки наведенням (Tip) і пасхалки на логотипі/цитаті/печиві."""
from __future__ import annotations

import webbrowser

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QMenu, QSizePolicy, QVBoxLayout

from .. import guide, i18n, meta
from ..i18n import tr
from .boxes import SectionBox
from .consts import FONT_B, FONT_S, FOOT_GAP, GG, H_M, M
from .labels import TkImage, TkLabel, TkStrip
from .pix import PixButton
from .theme import THEME, tk_font
from .tkcompat import Frame


def _vb(parent, margins=(0, 0, 0, 0), spacing=0):
    lay = QVBoxLayout(parent)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return lay


def _hb(parent, margins=(0, 0, 0, 0), spacing=0):
    lay = QHBoxLayout(parent)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return lay


class UniformRow(Frame):
    """ряд рівних колонок із проміжком gap, як tk grid з `uniform` і weight=1: зайві пікселі ділимо так само, як Tk
    (накопичувально: колонка i отримує floor(extra·(i+1)/n) − floor(extra·i/n) додаткових пікселів)"""

    def __init__(self, parent, gap, height):
        super().__init__(parent)
        self.gap = gap
        self.items: list = []
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def add(self, w):
        w.setParent(self)
        self.items.append(w)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        n = len(self.items)
        if not n:
            return
        avail = self.width() - (n - 1) * self.gap
        base = max(1, avail // n)
        extra = max(0, avail - base * n)
        x = 0
        for i, w in enumerate(self.items):
            wi = base + (extra * (i + 1)) // n - (extra * i) // n
            w.setGeometry(x, 0, wi, self.height())
            w.show()
            x += wi + self.gap


class FooterMixin:
    """методи футера: `_build_footer`, `_fill_footer`, `_foot_fit`, `_guide_pick`, `_goto_guide`, `_links_menu`, `_donate`"""

    # --- будова (як App._build_footer): плівка вгорі й унизу, між нею - колонка рівно по центру вільного місця
    def _build_footer(self):
        f = self.foot
        self._foot_key = None
        self.guide_sel = self.ui.get("guide") if self.ui.get("guide") in guide.SECTIONS else "start"
        fl = _vb(f, (M, M, M, M))
        fl.addWidget(TkStrip(f, "film_tile", 12))
        fl.addStretch(1)
        col = Frame(f)
        self.foot_col = col
        fl.addWidget(col, 0, Qt.AlignHCenter)
        fl.addStretch(1)
        fl.addWidget(TkStrip(f, "film_tile", 12))
        cl = _vb(col)

        head = Frame(col)                                        # котушка й логотип, подвоєні (піксель-арт 2x)
        hl = _hb(head)
        self._foot_mascot = TkImage(head, "mascot_w", scale=2)
        self._foot_logo = TkImage(head, "logo_w", scale=2)
        hl.addWidget(self._foot_mascot)
        hl.addSpacing(GG)
        hl.addWidget(self._foot_logo)
        cl.addSpacing(FOOT_GAP)
        cl.addWidget(head, 0, Qt.AlignHCenter)
        cl.addSpacing(FOOT_GAP)

        box = SectionBox("", col)
        self.foot_box = box
        cl.addWidget(box)
        tabs = UniformRow(box.inner, M, H_M)
        self.guide_tabs = {}
        for sec in guide.SECTIONS:                               # шість вкладок рівної ширини на всю ширину рамки, між ними M
            b = PixButton(tabs, "", command=lambda s=sec: self._guide_pick(s), width=60, height=H_M)
            tabs.add(b)
            self.guide_tabs[sec] = b
        box.body.addWidget(tabs)
        box.body.addSpacing(M)
        self.guide_box = Frame(box.inner)                         # висота = найвищий розділ: вкладки не стрибають
        gl = _vb(self.guide_box)
        self.guide_lbl = TkLabel(self.guide_box, "", font=FONT_S, anchor="nw", justify="left")
        self.guide_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gl.addWidget(self.guide_lbl)
        self.guide_box.setFixedHeight(40)
        box.body.addWidget(self.guide_box)
        self._measure = TkLabel(None, "", font=FONT_S, justify="left")   # лише міряти висоту тексту; ніколи не показується

        q = Frame(col)
        ql = _vb(q)
        cl.addSpacing(FOOT_GAP)
        cl.addWidget(q)
        ql.addWidget(TkStrip(q, "dither_light", 8))
        ql.addSpacing(GG)
        self.phil_lbl = TkLabel(q, "", font=FONT_B, anchor="center", justify="center")
        self.phil_lbl.setCursor(Qt.PointingHandCursor)
        ql.addWidget(self.phil_lbl, 0, Qt.AlignHCenter)
        ql.addSpacing(GG)
        ql.addWidget(TkStrip(q, "dither_light", 8))
        self.tip_wrap = Frame(q)                                  # «Порада»: не показується під час відтворення й експорту
        _vb(self.tip_wrap, (0, 0, 0, M))
        self.tip_lbl = TkLabel(self.tip_wrap, "", dim=True, font=FONT_S, anchor="center", justify="center")
        self.tip_wrap.layout().addWidget(self.tip_lbl, 0, Qt.AlignHCenter)
        ql.addWidget(self.tip_wrap)

        bar = Frame(col)
        cl.addSpacing(FOOT_GAP)
        cl.addWidget(bar)
        cl.addSpacing(16)
        bl = QGridLayout(bar)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        bl.setColumnStretch(0, 1)
        bl.setColumnStretch(2, 1)
        self.about_lbl = TkLabel(bar, "", dim=True, font=FONT_S, anchor="nw", justify="left", padx=0, pady=0)
        bl.addWidget(self.about_lbl, 0, 0, Qt.AlignLeft | Qt.AlignTop)
        if meta.links():                                          # клік по «InkReel 1.0.0»: меню з офіційними посиланнями
            self.about_lbl.setCursor(Qt.PointingHandCursor)
            self.about_lbl.clicked = self._links_menu
        self.btn_donate = None
        if meta.donate_enabled():
            self.btn_donate = PixButton(bar, "", command=self._donate, width=288, height=H_M, icon="cookie")
            self.btn_donate.setFixedWidth(288)
            bl.addWidget(self.btn_donate, 0, 1, Qt.AlignHCenter | Qt.AlignTop)
            self.btn_donate.set_enabled(bool(meta.DONATE))
        self.btn_top = PixButton(bar, "", command=lambda: self.pstack.go(0), width=120, height=H_M, icon="up")
        self.btn_top.setFixedWidth(120)
        bl.addWidget(self.btn_top, 0, 2, Qt.AlignRight | Qt.AlignTop)
        self._fill_footer()

    def _fill_footer(self):
        """тексти футера поточною мовою"""
        self.foot_box.title.config(text=tr("Гайд та інфо").upper())
        for sec, b in self.guide_tabs.items():
            b.set_text(guide.title(sec))
            b.set_on(sec == self.guide_sel)
        self.guide_lbl.config(text=guide.body(self.guide_sel))
        ph = guide.philosophy()
        self.phil_lbl.config(text="«" + ph + "»" if i18n.lang() == i18n.UK else "“" + ph + "”")
        self._fill_tip()
        a = f"{meta.APP} {meta.VERSION}" + (f"\n{meta.AUTHOR}" if meta.AUTHOR else "")
        self.about_lbl.config(text=a)
        if self.btn_donate is not None:
            self.btn_donate.set_text(tr("Пригостити автора печивом"))
        self.btn_top.set_text(tr("Нагору"))
        self._foot_key = None

    def _tip_visible(self) -> bool:
        """«Порада» мовчить, поки грає відео чи йде експорт"""
        pl = getattr(self, "player", None)
        exporting = getattr(getattr(self, "export", None), "running", False)
        return not (exporting or (pl is not None and pl.playing and getattr(pl, "path", None)))

    def _fill_tip(self):
        """порада під філософією у футері (або нічого, коли йде відтворення чи експорт)"""
        if not hasattr(self, "tip_lbl"):
            return
        if self._tip_n < 0:
            self._tip_n = guide.tip(-1)
        if self._tip_visible():
            self.tip_lbl.config(text=guide.tip_text(self._tip_n))
            self.tip_wrap.show()                          # відступ знизу малий: до рядка з версією ще FOOT_GAP, а футер має влізти у вікно 640 px
        else:
            self.tip_wrap.hide()

    def _foot_fit(self, cw):
        """ширина колонки футера під ширину вікна; висота тексту гайда - за найвищим розділом, щоб вкладки не стрибали"""
        W = max(560, min(cw - 48, 980))
        key = (W, i18n.lang())
        if key == self._foot_key:
            return
        self._foot_key = key
        self.foot_col.setFixedWidth(W)
        wrap = W - 2 - 2 * M - 4                          # ширина тексту = ширина рамки без ліній, відступів і запасу
        self.guide_lbl.config(wraplength=wrap)
        self.phil_lbl.config(wraplength=min(W - 60, 760))
        self.tip_lbl.config(wraplength=W - 60)            # найдовша порада - один рядок
        self.about_lbl.config(wraplength=max(120, W // 3 - 10))
        tallest = 0
        for sec in guide.SECTIONS:
            self._measure.config(text=guide.body(sec), wraplength=wrap)
            tallest = max(tallest, self._measure.winfo_reqheight())
        self.guide_box.setFixedHeight(tallest + 4)

    def _guide_pick(self, sec):
        self.guide_sel = sec
        self._fill_footer()
        self.pstack.layout()
        self._save_later()

    def _show_app(self):
        """відкрили файл чи почали збереження, поки на екрані футер: назад до програми - усе відбувається там"""
        if self.pstack.tcur > 0:
            self.pstack.go(0)

    def _goto_guide(self):
        """«?» і F1: до футера з гайдом; ще раз - назад до програми"""
        if self.pstack.tcur == 0:
            self._tip_n = guide.tip(self._tip_n)            # у футер зайшли заново: нова порада
            self._fill_tip()
        self.pstack.go(0 if self.pstack.tcur > 0 else 1)

    # --- меню посилань і донату
    def _popup(self, items, under):
        m = QMenu(self)
        m.setFont(tk_font(FONT_S))
        bg, fg = THEME.bg.name(), THEME.fg.name()
        m.setStyleSheet(f"QMenu {{ background: {bg}; color: {fg}; border: 1px solid {fg}; }}"
                        f"QMenu::item {{ padding: 3px 16px; }}"
                        f"QMenu::item:selected {{ background: {fg}; color: {bg}; }}")
        for name, url in items:
            m.addAction(name, lambda u=url: webbrowser.open(u))
        m.exec(under.mapToGlobal(under.rect().bottomLeft()))

    def _links_menu(self, event=None):
        """офіційні посилання: сайт, код і релізи на GitHub, сторінка в Microsoft Store"""
        self._popup([(tr(n), u) for n, u in meta.links()], self.about_lbl)

    def _donate(self):
        links = list(meta.DONATE)
        if not links:
            return
        if len(links) == 1:
            webbrowser.open(links[0][1])
            return
        self._popup(links, self.btn_donate)
