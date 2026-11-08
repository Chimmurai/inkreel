"""Вирівнювання інтерфейсу: усе лежить на одній сітці (відступи M / G / GG, висоти керування 24 / 30 / 38), краї однакових
елементів збігаються, нічого не вилазить за свої кнопки.

Тест відкриває справжнє вікно, тому там, де немає екрана (Linux без DISPLAY), пропускається:
    xvfb-run -a python -m pytest tests/test_layout.py
Налаштування користувача не чіпає: файл налаштувань підміняється тимчасовим."""
from __future__ import annotations

import os
import sys

import pytest
from conftest import settle

pytestmark = pytest.mark.skipif(sys.platform.startswith("linux") and not os.environ.get("DISPLAY"),
                                reason="немає екрана (запускай під xvfb-run)")

CHAR_W = 7          # Consolas 9-10 pt при 96 dpi: 7 px на знак - за цим міряємо підписи, де шрифт контейнера вужчий


def box(w):
    x, y = w.winfo_rootx(), w.winfo_rooty()
    return x, y, x + w.winfo_width(), y + w.winfo_height()


@pytest.fixture(params=["uk", "en"])
def lang_app(app, request):
    app._set_lang(request.param)
    app.pstack.scroll_to(0)
    settle(app)
    return app


def near(a, b, tol=1):
    return abs(a - b) <= tol


def test_page_margins_and_rhythm(lang_app):
    """поле вікна M з усіх боків, між блоками головного екрана скрізь M"""
    from benvid.gui import M
    a = lang_app
    main = box(a.main)
    top, s1, mid, s2, seek, row = (box(w) for w in (a.btn_open.master, a._strip1, a.side.master, a._strip2, a.seek, a.bot_row))
    assert top[1] - main[1] == M
    for name, b in (("шапка", top), ("смужка 1", s1), ("центр", mid), ("смужка 2", s2), ("перемотування", seek), ("транспорт", row)):
        assert b[0] - main[0] == M, f"{name}: ліве поле"
        assert main[2] - b[2] == M, f"{name}: праве поле"
    assert s1[1] - top[3] == M and mid[1] - s1[3] == M and s2[1] - mid[3] == M
    assert seek[1] - s2[3] == M and row[1] - seek[3] == M and main[3] - row[3] == M


def test_side_panel_matches_screen(lang_app):
    """панель праворуч: верх і низ збігаються з екраном, між ними M, секції однієї ширини й проміжок між ними M"""
    import tkinter as tk
    from benvid.gui import M
    a = lang_app
    scr, side = box(a.screen.master), box(a.side)
    assert scr[1] == side[1] and scr[3] == side[3]
    assert side[0] - scr[2] == M
    if a.sstack.overflow:
        pytest.skip("екран замалий: панель прокручується")
    secs = [box(w) for w in a.side_in.winfo_children() if isinstance(w, tk.Frame)]
    assert len(secs) == 5                          # «Якість», «Розмір результату», «Палітра», «Вид», «Експорт»
    for x, y in zip(secs, secs[1:]):
        assert y[1] - x[3] == M
    assert all(b[0] == side[0] and b[2] == side[2] for b in secs)
    assert secs[0][1] == side[1] and secs[-1][3] == side[3]


def test_header(lang_app):
    from benvid.gui import HEAD_W, H_M, M
    a = lang_app
    for b in (a.btn_open, a.btn_export, a.btn_stop):
        assert (b.winfo_width(), b.winfo_height()) == (HEAD_W, H_M)
    o, e, s = (box(b) for b in (a.btn_open, a.btn_export, a.btn_stop))
    assert e[0] - o[2] == M and s[0] - e[2] == M
    mascot = box(a._lbl_mascot)
    mid_y = (mascot[1] + mascot[3]) / 2
    for b in (a.btn_open, a.btn_export, a.btn_stop, a.btn_help, a.btn_ua, a.btn_en):
        bb = box(b)
        assert near((bb[1] + bb[3]) / 2, mid_y), "кнопки шапки не по центру ряду"
    ua, en, hp, side = box(a.btn_ua), box(a.btn_en), box(a.btn_help), box(a.side)
    assert en[2] == side[2], "EN - врівень з правим краєм панелі"
    assert ua[2] == en[0], "UA і EN - склеєна пара"
    assert ua[0] - hp[2] == M
    assert hp[3] - hp[1] == H_M and ua[3] - ua[1] == H_M


def test_size_grid(lang_app):
    """ґратка розмірів: однакові кнопки, проміжки G по обох осях, краї врівень з вмістом секції"""
    from benvid.gui import G, H_S, SIZE_KEYS
    a = lang_app
    if a.sstack.overflow:
        pytest.skip("екран замалий: панель прокручується")
    bb = [box(a.size_btns[k]) for k in SIZE_KEYS]
    ws = [b[2] - b[0] for b in bb]
    assert max(ws) - min(ws) <= 1
    assert all(b[3] - b[1] == H_S for b in bb)
    xs, ys = sorted({b[0] for b in bb}), sorted({b[1] for b in bb})
    assert len(xs) == 4 and len(ys) == (len(SIZE_KEYS) + 3) // 4
    assert all(near(xs[i + 1] - xs[i], ws[0] + G) for i in range(3))
    assert all(ys[i + 1] - ys[i] == H_S + G for i in range(len(ys) - 1))
    inner = box(a.size_btns[SIZE_KEYS[0]].master.master)
    assert xs[0] == inner[0] and max(b[2] for b in bb) == inner[2]
    ok = box(a.btn_ok)
    assert ok[2] == inner[2] and ok[0] == xs[-1], "«OK» стоїть під останньою колонкою ґратки"


def test_control_heights(lang_app):
    """три висоти керування: 24 (ґратка, поля, списки, перемикачі), 30 (шапка, палітра, збереження, футер), 38 (транспорт)"""
    from benvid.gui import H_L, H_M, H_S
    a = lang_app
    small = [a.ent_w, a.ent_h, a.btn_ok, a.fmt_cb, a.q_cb, a.btn_aud, a.btn_rng]
    mid = [a.btn_pprev, a.btn_pnext, a.btn_pinv, a.btn_export2, a.btn_open, a.btn_help, a.btn_ua, a.btn_en]
    large = [a.btn_rew, a.btn_prev, a.btn_play, a.btn_stopp, a.btn_next, a.btn_fwd, a.btn_snd, a.btn_loop, a.btn_photo]
    assert {w.winfo_height() for w in small} == {H_S}
    assert {w.winfo_height() for w in mid} == {H_M}
    assert {w.winfo_height() for w in large} == {H_L}


def test_export_rows_are_flush(lang_app):
    """у секції «Експорт» усе впритул до країв вмісту: списки, перемикачі, кнопка збереження"""
    from benvid.gui import G
    a = lang_app
    if a.sstack.overflow:
        pytest.skip("екран замалий: панель прокручується")
    inner = box(a.btn_export2.master)
    fmt, q, aud, rng, sv = (box(w) for w in (a.fmt_cb, a.q_cb, a.btn_aud, a.btn_rng, a.btn_export2))
    assert fmt[0] == inner[0] and q[2] == inner[2] and q[0] - fmt[2] == G
    assert aud[0] == inner[0] and rng[2] == inner[2]
    assert sv[0] == inner[0] and sv[2] == inner[2]
    assert (aud[2] - aud[0]) == (rng[2] - rng[0])
    prog = box(a.prog)
    assert prog[0] == inner[0] and prog[2] == inner[2]


def test_transport_spacing(lang_app):
    from benvid.gui import G, GG, M
    a = lang_app
    seq = [box(b) for b in (a.btn_rew, a.btn_prev, a.btn_play, a.btn_stopp, a.btn_next, a.btn_fwd)]
    assert all(y[0] - x[2] == M for x, y in zip(seq, seq[1:]))
    assert len({b[1] for b in seq}) == 1
    snd, vol, loop, photo, tm = (box(w) for w in (a.btn_snd, a.vol, a.btn_loop, a.btn_photo, a.time_lbl))
    assert snd[0] - seq[-1][2] == GG and vol[0] - snd[2] == G
    assert loop[0] - vol[2] == GG and photo[0] - loop[2] == GG and tm[0] - photo[2] == GG
    assert seq[0][0] == box(a.seek)[0], "транспорт починається врівень зі шкалою"
    assert box(a.stat_lbl)[2] == box(a.seek)[2], "рядок стану закінчується врівень зі шкалою"


def test_footer(lang_app):
    """футер: вкладки однакової ширини на всю рамку, кнопки низу однакової висоти, колонка по центру сторінки"""
    from benvid.gui import H_M, M
    a = lang_app
    a.pstack.scroll_to(10 ** 6)
    settle(a)
    tabs = [box(b) for b in a.guide_tabs.values()]
    ws = [t[2] - t[0] for t in tabs]
    assert max(ws) - min(ws) <= 1 and {t[3] - t[1] for t in tabs} == {H_M}
    assert all(near(y[0] - x[2], M) for x, y in zip(tabs, tabs[1:]))
    col, page = box(a.foot_col), box(a.foot)
    assert near(col[0] - page[0], page[2] - col[2]), "колонка футера не по центру сторінки"
    assert box(a.btn_top)[2] == col[2] and a.btn_top.winfo_height() == H_M
    if a.btn_donate is not None:
        d = box(a.btn_donate)
        assert d[3] - d[1] == H_M and near((d[0] + d[2]) / 2, (col[0] + col[2]) / 2, 1)
    a.pstack.scroll_to(0)
    settle(a)


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def test_labels_fit_their_buttons(lang_app):
    """підписи кнопок у обох мовах влазять з запасом, навіть якщо шрифт ширший за контейнерний: міряємо по 7 px на знак"""
    from benvid import gui
    a = lang_app
    bad = []
    for w in _walk(a):
        if isinstance(w, gui.PixButton) and w.text and w.winfo_ismapped() | (w in a.guide_tabs.values()):
            icon = (gui.PixButton.ICON_W if w.icon else (w.im.width() if w.im is not None else 0))
            ink = getattr(w.im, "ink", None)
            if ink:
                icon = ink[2] - ink[0]
            need = len(w.text) * CHAR_W + (icon + gui.ICON_GAP if icon else 0) + 2 * 6
            if need > w.winfo_width():
                bad.append((w.text, need, w.winfo_width()))
    assert not bad, bad


def test_status_labels_fit(lang_app):
    """довгі підписи в панелі не впираються в край: найдовші можливі рядки в обох мовах"""
    from benvid.i18n import tr
    a = lang_app
    est_w = a.est_lbl.winfo_width()
    for txt in (tr("Експорт: усе пораховано — миттєво"), tr("Експорт ≈ {t}  (готово {p}%)", t="12:34", p=100)):
        assert len(txt) * CHAR_W <= est_w, txt
    rng_w = a.rng_lbl.winfo_width()
    for txt in (tr("Фрагмент: усе відео\n[ ] — межі фрагмента"), tr("Фрагмент: {a} – {b}\n({d}, {n})", a="10:00", b="12:34", d="2:34", n="3 696 кадрів")):
        assert max(len(line) for line in txt.splitlines()) * CHAR_W <= rng_w + 20, txt     # запас 20: тривалість понад 10 хв - рідкісна


def test_photo_mode_keeps_alignment(app, tmp_path):
    """фото: шкали й транспорту немає, але ряд внизу починається врівень з полем, а панель так само сягає низу екрана"""
    from PIL import Image
    from benvid.gui import M
    import numpy as np

    app._set_lang("uk")
    p = tmp_path / "photo.png"
    arr = (np.linspace(0, 255, 160 * 120).reshape(120, 160)).astype(np.uint8)
    Image.fromarray(arr).convert("RGB").save(p)
    app._open(str(p))
    settle(app, rounds=12, pause=0.1)
    assert app.kind == "image"
    main, row = box(app.main), box(app.bot_row)
    assert box(app.time_lbl)[0] == row[0] == main[0] + M
    scr, side = box(app.screen.master), box(app.side)
    assert scr[3] == side[3] and scr[1] == side[1]
    assert main[3] - row[3] == M
    assert app.btn_export.text and app.btn_export.winfo_width() == app.btn_open.winfo_width()
