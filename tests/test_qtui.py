"""Інтерфейс на Qt (benvid/qtui): тема, піксельні віджети, поверхня кадра, головне вікно з справжнім плеєром.
Запускається на платформі `offscreen` (без екрана й без відеокарти), тож працює в CI; пропускається, якщо PySide6 не встановлено."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from benvid import engine, i18n, palette, view  # noqa: E402
from benvid.qtui.consts import AUTO  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def spin(app, cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    return bool(cond())


# ---------------------------------------------------------------- тема
def test_theme_follows_the_palette_and_emits_once(qapp):
    from benvid.qtui.theme import THEME
    seen = []
    THEME.changed.connect(lambda: seen.append(1))
    THEME.set_key(palette.CLASSIC)
    n = len(seen)
    THEME.set_key("Плакат")
    t = palette.theme("Плакат")
    assert (THEME.bg.name(), THEME.fg.name()) == (t["BG"].lower(), t["FG"].lower())
    assert len(seen) == n + 1, "зміна палітри - один сигнал на все вікно"
    THEME.set_key("Плакат")
    assert len(seen) == n + 1, "та сама палітра нічого не перефарбовує"
    THEME.set_key(palette.CLASSIC)


# ---------------------------------------------------------------- віджети
def test_button_emits_clicked_only_inside_and_slider_reports_drag_and_release(qapp):
    from benvid.qtui.widgets import PixButton, PixSlider
    b = PixButton("OK")
    b.resize(100, 28)
    b.show()
    got = []
    b.clicked.connect(lambda: got.append(1))
    QTest.mouseClick(b, Qt.LeftButton, pos=QPoint(50, 14))
    assert got == [1]
    QTest.mousePress(b, Qt.LeftButton, pos=QPoint(50, 14))
    QTest.mouseRelease(b, Qt.LeftButton, pos=QPoint(250, 14))                # відпустили поза кнопкою: клік не зараховується
    assert got == [1]
    s = PixSlider(0, 100)
    s.resize(300, 26)
    s.show()
    vals, rel = [], []
    s.valueChanged.connect(vals.append)
    s.released.connect(rel.append)
    QTest.mousePress(s, Qt.LeftButton, pos=QPoint(10, 13))
    for x in range(10, 290, 20):
        QTest.mouseMove(s, QPoint(x, 13))
    QTest.mouseRelease(s, Qt.LeftButton, pos=QPoint(290, 13))
    assert len(vals) > 5 and vals == sorted(vals), "значення зростають, поки тягнуть"
    assert len(rel) == 1 and rel[0] > 90, "відпускання дає кінцеве значення один раз"


def test_quality_bar_picks_the_nearest_circle_or_the_irnm1_diamond(qapp):
    from benvid.qtui.widgets import QualityBar
    q = QualityBar()
    q.resize(300, 40)
    q.show()
    got = []
    q.chosen.connect(got.append)
    xs, dx, cy = q._centers()
    QTest.mouseClick(q, Qt.LeftButton, pos=QPoint(int(xs[1]), int(cy)))
    QTest.mouseClick(q, Qt.LeftButton, pos=QPoint(int(dx), int(cy)))
    assert got == [2, 6] and q.level == 6


# ---------------------------------------------------------------- поверхня кадра
def test_surface_composes_off_the_ui_thread_and_zooms_by_whole_steps(qapp):
    from benvid.qtui.surface import FrameSurface
    s = FrameSurface()
    s.resize(640, 400)
    s.show()
    mask = (np.random.RandomState(1).rand(90, 160) > 0.5).astype(np.uint8)
    s.set_mask(mask)
    assert spin(qapp, lambda: s._img is not None)
    assert s._img.width() <= 640 and s._img.height() <= 400
    assert s.zoom is None
    before = s._img.width()
    ev = __import__("PySide6.QtGui", fromlist=["QWheelEvent"]).QWheelEvent(
        QPointF(320, 200), QPointF(320, 200), QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    s.wheelEvent(ev)
    assert s.zoom is not None and s.zoom in view.LEVELS
    assert spin(qapp, lambda: s._img is not None and s._img.width() != before or s._img.width() == 640)


def test_surface_recolours_with_the_palette_without_a_new_frame(qapp):
    from benvid.qtui.surface import FrameSurface
    s = FrameSurface()
    s.resize(320, 200)
    s.show()
    s.set_mask((np.random.RandomState(2).rand(50, 80) > 0.5).astype(np.uint8))
    assert spin(qapp, lambda: s._img is not None)
    gray = s._img.format()
    s.set_palette("Бурштин")
    assert spin(qapp, lambda: s._img.format() != gray)                       # з таблицею кольорів кадр стає RGB


# ---------------------------------------------------------------- головне вікно
@pytest.fixture()
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(engine, "_OLD", str(tmp_path / "old.json"))
    (tmp_path / "settings.json").write_text(json.dumps({"ui": {"lang": "uk", "palette": "Класика", "size": "240p", "method": 2}}), encoding="utf-8")
    from benvid.qtui.window import MainWindow
    w = MainWindow()
    w.resize(1300, 800)
    w.show()
    yield w
    w.close()
    w.ready = False                                      # закрите вікно лишається живим у Qt: його такт не повинен ходити до плеєра (у зламаного пулу замок може тримати чужий потік)
    w.tick.stop()


def test_window_palette_language_and_level_controls(qapp, win):
    from benvid.qtui.theme import THEME
    win._step_palette(1)
    assert win.pal_name == palette.NAMES[1] and THEME.key == win._pal_key()
    win._toggle_inv()
    assert palette.INV in THEME.key or THEME.key.endswith(palette.INV)
    win._toggle_inv()
    win._set_level(4)
    assert win.params.method == 4 and win.m_slider.v == 4
    win._set_glyph(True)
    assert win.params.glyph and win.btn_sym.on and win.dens_box.isVisibleTo(win) and not win.m_slider.branch_on
    win._set_glyph(False)
    win._set_lang("en")
    assert i18n.lang() == "en" and win.btn_open.text == "Open"
    win._set_lang("uk")
    assert win.btn_open.text == "Відкрити"


def test_window_opens_a_picture_and_shows_the_real_engine_output(qapp, win, tmp_path):
    from PIL import Image
    p = tmp_path / "pic.png"
    rng = np.random.RandomState(5)
    Image.fromarray((rng.rand(120, 200) * 255).astype(np.uint8)).save(p)
    win._open(str(p))
    assert win.ready and win._name == "pic.png"
    assert spin(qapp, lambda: win.surface._img is not None, timeout=60), "плеєр порахував кадр, поверхня його показала"
    assert win.surface.mask is not None and win.surface.mask.shape[1] > 20


def test_window_exports_a_photo_a_gif_and_a_video_with_progress(qapp, win, tmp_path, monkeypatch):
    """експорт на справжніх функціях двигуна: фото PNG, відео MP4 з прогресом, скасування"""
    from PIL import Image
    from benvid.qtui import window as W
    pic = tmp_path / "pic.png"
    Image.fromarray((np.random.RandomState(6).rand(90, 160) * 255).astype(np.uint8)).save(pic)
    win._open(str(pic))
    assert spin(qapp, lambda: win.surface._img is not None, timeout=60)
    assert win.kind == "image"
    out = tmp_path / "out.png"
    monkeypatch.setattr(W.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    win._export()
    assert spin(qapp, lambda: not win.export.running and out.exists() and "Готово" in win.prog_lbl.text_, timeout=60), win.prog_lbl.text_
    assert "Готово" in win.prog_lbl.text_ and win.btn_export.enabled and not win.btn_stop.enabled
    assert Image.open(out).size[0] > 10

    sys_path = os.path.dirname(__file__)
    import sys
    sys.path.insert(0, sys_path)
    from test_video import make_video
    vid = tmp_path / "clip.mp4"
    make_video(str(vid), n=40, w=160, h=90)
    win._open(str(vid))
    assert win.kind == "video" and win.fmt_cb.count() >= 1 and win.q_cb.isVisibleTo(win)
    out2 = tmp_path / "clip_bw.mp4"
    monkeypatch.setattr(W.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out2), "")))
    seen = []
    win.export.progress.connect(lambda f, *a: seen.append(f))
    win._export()
    assert win.export.running and not win.btn_export.enabled and win.btn_stop.enabled
    assert spin(qapp, lambda: not win.export.running and "Готово" in win.prog_lbl.text_, timeout=120), win.prog_lbl.text_
    assert out2.exists() and out2.stat().st_size > 1000 and "Готово" in win.prog_lbl.text_
    assert seen and max(seen) > 0.5 and win.prog.f == 1.0

    win._export()                                    # скасування: стоп під час роботи
    assert spin(qapp, lambda: win.export.running, timeout=5)
    win.export.stop()
    assert spin(qapp, lambda: not win.export.running, timeout=60)
    assert spin(qapp, lambda: win.btn_export.enabled, timeout=10), "після скасування кнопки знову активні (сигнал finished доходить трохи пізніше за running=False)"


def test_footer_page_opens_with_f1_picks_guide_tabs_and_returns(qapp, win):
    from benvid import guide
    win.resize(1280, 860)
    spin(qapp, lambda: win.pstack.max_off > 0, timeout=5)
    assert win.pstack.max_off > 0 and win.page_bar.isVisible(), "сторінка з двох екранів, смужка завжди видна"
    assert win.foot_col.width() == max(560, min(win.page.width() - win.page_bar.bw - 48, 980))
    assert list(win.guide_tabs) == list(guide.SECTIONS)
    win._goto_guide()
    assert spin(qapp, lambda: win.pstack.cur == 1 and win.pstack.idle, timeout=5), "F1 / «?»: до футера"
    sec = guide.SECTIONS[2]
    win.guide_tabs[sec].cmd()
    assert win.guide_sel == sec and win.guide_lbl.text_ == guide.body(sec) and win.guide_tabs[sec].on
    assert win.guide_box.height() >= win.guide_lbl.winfo_reqheight() - 6, "висота гайда - за найвищим розділом"
    win._goto_guide()
    assert spin(qapp, lambda: win.pstack.cur == 0 and win.pstack.idle, timeout=5), "ще раз - назад до програми"


def test_footer_tip_is_silent_while_exporting_and_texts_follow_the_language(qapp, win):
    assert win.tip_wrap.isVisibleTo(win.foot) and win.tip_lbl.text_
    win.export.running = True
    win._fill_tip()
    assert not win.tip_wrap.isVisibleTo(win.foot)
    win.export.running = False
    win._set_lang("en")
    assert win.btn_top.text and win.btn_top.text != "Нагору"
    assert win.phil_lbl.text_.startswith("“") and win.foot_box.title.text_ == win.foot_box.title.text_.upper()
    win._set_lang("uk")
    assert win.phil_lbl.text_.startswith("«") and win.btn_top.text == "Нагору"


def test_footer_tabs_share_extra_pixels_like_tk_grid(qapp):
    from benvid.qtui.footer import UniformRow
    from benvid.qtui.pix import PixButton
    row = UniformRow(None, 8, 30)
    bs = [PixButton(row, "", width=60, height=30) for _ in range(6)]
    for b in bs:
        row.add(b)
    row.resize(962, 30)
    row.show()
    assert [b.width() for b in bs] == [153, 154, 154, 153, 154, 154] and bs[1].x() == 161 and sum(b.width() for b in bs) + 5 * 8 == 962


def test_label_wrap_counts_the_space_after_a_word_like_tk(qapp):
    from PySide6.QtGui import QFontMetrics
    from benvid.qtui.labels import wrap_text
    from benvid.qtui.theme import tk_font
    fm = QFontMetrics(tk_font(("Consolas", 12, "bold")))
    w = fm.horizontalAdvance("aaaa bbbb ")
    assert wrap_text("aaaa bbbb cccc", w, fm)[0] == "aaaa bbbb", "слово + пробіл за ним влазять"
    assert wrap_text("aaaa bbbb cccc", w - 1, fm)[0] == "aaaa", "слово влазить, а разом із наступним пробілом - ні (Tk переносить)"
    w = fm.horizontalAdvance("aaaa bbbb")
    assert wrap_text("aaaa bbbb", w, fm) == ["aaaa bbbb"], "останнє слово абзацу пробілу не має"


def test_tips_show_after_a_delay_one_at_a_time_and_vanish_on_leave(qapp, win, monkeypatch):
    from PySide6.QtCore import QEvent
    from benvid.qtui.tips import Tip
    monkeypatch.setattr(Tip, "DELAY", 30)
    a, b = win.btn_open, win.btn_export
    state = {"n": 0}

    def text():
        state["n"] += 1
        return "підказка №%d" % state["n"]
    from PySide6.QtWidgets import QWidget
    plain = QWidget(win)                                 # без вбудованої підказки: інакше два таймери змагаються
    plain.setGeometry(5, 5, 30, 30)
    plain.show()
    Tip(plain, text)
    QApplication.sendEvent(plain, QEvent(QEvent.Enter))
    assert Tip._cur is None, "підказка не одразу"
    assert spin(qapp, lambda: Tip._cur is not None, timeout=3)
    t1 = Tip._cur
    assert t1._win.isVisible() and t1._win.lines[0].startswith("підказка №")      # (справжній Enter від показу віджета теж може запустити таймер: номер не важливий)
    QApplication.sendEvent(win.btn_open, QEvent(QEvent.Enter))
    assert spin(qapp, lambda: Tip._cur is not None and Tip._cur is not t1, timeout=3), "перейшли на іншу кнопку - підказка тепер її"
    assert t1._win is None, "стара прибрана"
    QApplication.sendEvent(win.btn_open, QEvent(QEvent.Leave))
    assert Tip._cur is None
    Tip.hide_all()


def test_every_tip_text_is_translated_and_every_widget_exists(qapp, win):
    from benvid.qtui import eggs_ui
    i18n.set_lang("en")
    try:
        for attr, key in eggs_ui.TIPS.items():
            assert getattr(win, attr, None) is not None, attr
            if attr in ("btn_ua", "btn_en"):             # назви мов однакові в обох мовах
                continue
            assert i18n.tr(key) != key, "нема англійського тексту підказки: " + key
        for key in (*eggs_ui.SIZE_TIPS.values(), eggs_ui.LOCK_TIP_ON, eggs_ui.LOCK_TIP_OFF, *eggs_ui.PLAY_TIPS, *eggs_ui.DONATE_TIPS):
            assert i18n.tr(key) != key, key
    finally:
        i18n.set_lang("uk")


def test_logo_x5_defocuses_the_picture_with_a_caption_then_restores_sharpness(qapp, win):
    import benvid.eggs as eggs
    mask = (np.random.RandomState(3).rand(90, 160) > 0.5).astype(np.uint8)
    win.surface.set_mask(mask)
    assert spin(qapp, lambda: win.surface._img is not None)
    sharp = win.surface.grab().toImage()
    for _ in range(5):
        win.lbl_logo.clicked()
    assert win._egg_t0 is not None
    assert spin(qapp, lambda: win.surface.defocus > 0.05, timeout=5), "розфокус наростає"
    win._egg_t0 -= 2.0                                   # мить, коли підпис уже проявився
    win.surface.egg_t0 = win._egg_t0
    win.surface.update()
    blurred = win.surface.grab().toImage()
    assert blurred != sharp
    win._egg_t0 = time.perf_counter() - eggs.DEFOCUS_TOTAL - 1
    win._egg_tick()
    assert win.surface.defocus == 0.0 and win.surface.egg_t0 is None and win._egg_t0 is None


def test_logo_x5_without_a_picture_spins_the_reel_with_a_phrase_and_the_bubble_goes_away(qapp, win):
    import benvid.eggs as eggs
    for _ in range(5):
        win.lbl_logo.clicked()
    assert win._bub is not None and win._bub.isVisible() and win._last_quote >= 0
    assert spin(qapp, lambda: win._spin_i is None and win.lbl_mascot._shown is None and not win._spin_job.isActive(), timeout=5), "оберт скінчився"
    win._bubble_close()
    assert win._bub is None
    assert len(eggs.spin_angles()) > 2


def test_swatch_x7_runs_the_palette_party_and_returns_to_the_users_palette(qapp, win):
    win._step_palette(1)
    mine = win.pal_name
    seen = set()
    win.pal_lbl.config  # (підпис палітри оновлюється разом з нею)
    for _ in range(7):
        win._swatch_click()
    assert win._party_seq is not None
    end = time.time() + 20
    while time.time() < end and win._party_seq is not None:
        qapp.processEvents()
        seen.add(win.pal_name)
        time.sleep(0.01)
    assert win._party_seq is None and len(seen) > 5, "перебрала кілька палітр"
    assert spin(qapp, lambda: win.pal_name == mine, timeout=3)


def test_closing_in_the_middle_of_the_party_keeps_the_users_palette(qapp, win):
    mine = win.pal_name
    for _ in range(7):
        win._swatch_click()
    assert spin(qapp, lambda: win.pal_name != mine, timeout=3)
    win._party_stop()
    assert win.pal_name == mine and win._party_seq is None


def test_cookie_is_bitten_then_eaten_into_crumbs_then_rebaked(qapp, win):
    import benvid.eggs as eggs
    b = win.btn_donate
    if b is None:
        pytest.skip("у цій збірці кнопки донату нема")
    b.set_enabled(False)
    for k in range(eggs.COOKIE_CLICKS - 1):
        win._cookie_click()
        assert b.bites == k + 1 and not b.crumbs
    win._cookie_click()
    assert b.bites == eggs.EATEN and b.crumbs and win._bub is not None
    assert spin(qapp, lambda: not b.crumbs or b.bites == 0, timeout=5)
    win._cookie_rebake()
    assert b.bites == 0 and b.crumbs == []
    b.set_enabled(bool(__import__("benvid").meta.DONATE))


@pytest.fixture()
def vwin(qapp, win, tmp_path):
    """вікно з відкритим коротким відео"""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_video import make_video
    vid = tmp_path / "clip.mp4"
    make_video(str(vid), n=60, w=160, h=90)
    win._open(str(vid))
    assert win.kind == "video" and win.ready
    return win


def test_bottom_row_and_side_panel_follow_the_file_kind(qapp, win, tmp_path):
    from PIL import Image
    vis = lambda w: w.isVisibleTo(win)
    assert vis(win.btn_snd) and vis(win.btn_loop) and vis(win.seek)
    p = tmp_path / "pic.png"
    Image.fromarray((np.random.RandomState(1).rand(60, 90) * 255).astype(np.uint8)).save(p)
    win._open(str(p))
    assert win.kind == "image"
    spin(qapp, lambda: False, timeout=0.3)
    assert not any(vis(w) for w in (win.tr, win.btn_snd, win.vol, win.btn_loop, win.btn_photo, win.seek, win.row2, win.btn_lock, win.fps_cb))
    assert vis(win.time_lbl) and win.seek.marks is None and win.time_lbl.x() < 40, "фото: ряд починається врівень з краєм поля"
    g = tmp_path / "a.gif"
    frames = [Image.fromarray((np.random.RandomState(i).rand(40, 60) * 255).astype(np.uint8)) for i in range(4)]
    frames[0].save(g, save_all=True, append_images=frames[1:], duration=80, loop=0)
    win._open(str(g))
    assert win.kind == "gif" and vis(win.tr) and vis(win.btn_loop) and vis(win.btn_photo) and vis(win.seek)
    assert not vis(win.btn_snd) and not vis(win.vol) and not vis(win.row2) and not vis(win.btn_lock)


def test_loop_stop_and_jump_buttons_and_hotkeys(qapp, vwin, monkeypatch):
    w = vwin
    w.btn_loop.cmd() if False else None
    QTest.mouseClick(w.btn_loop, Qt.LeftButton, pos=QPoint(10, 10))
    assert w.btn_loop.on and w.player.loop is True
    QTest.keyClick(w, Qt.Key_L)
    assert not w.btn_loop.on and w.player.loop is False, "L - те саме, що кнопка «Цикл»"
    seeks = []
    monkeypatch.setattr(w.player, "seek", lambda f: seeks.append(f))
    monkeypatch.setattr(w.player, "position", lambda: 100.0)
    w._jump(10)
    assert seeks[-1] == 100.0 + 10 * w.player.fps
    QTest.keyClick(w, Qt.Key_Left, Qt.ShiftModifier)
    assert seeks[-1] == 100.0 - 10 * w.player.fps
    QTest.keyClick(w, Qt.Key_Home)
    assert seeks[-1] == 0 and w.btn_play.icon == "play"
    n = len(seeks)
    w.export.running = True
    w._jump(10)
    w._stop()
    assert len(seeks) == n, "під час експорту перемотка й стоп мовчать"
    w.export.running = False


def test_space_arrows_and_letters_work_in_any_layout_but_not_in_fields_or_footer(qapp, vwin, monkeypatch):
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent
    w = vwin
    got = []
    monkeypatch.setattr(w, "_toggle", lambda: got.append("toggle"))
    monkeypatch.setattr(w, "_step", lambda d: got.append(("step", d)))
    QTest.keyClick(w, Qt.Key_Space)
    QTest.keyClick(w, Qt.Key_Right)
    assert got == ["toggle", ("step", 1)]
    ev = QKeyEvent(QEvent.KeyPress, 0x41C, Qt.NoModifier, 0x4D, 0x4D, 0, "м")        # «м» у кириличній розкладці: клавіша M
    assert w._letter_of(ev) == "m"
    ev = QKeyEvent(QEvent.KeyPress, 0x456, Qt.NoModifier, 0x49, 0x49, 0, "і")
    assert w._letter_of(ev) == "i"
    inv = w.pal_inv
    QApplication.sendEvent(w, ev)
    assert w.pal_inv != inv, "«і» перемикає «навпаки», як I"
    w._toggle_inv()
    got.clear()
    w.ent_w.setFocus()
    QApplication.sendEvent(w.ent_w, QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier, " "))
    assert got == [], "пробіл у полі вводу - для поля"
    w.ent_w.clearFocus()
    w.pstack.tcur = 1
    QTest.keyClick(w, Qt.Key_Space)
    QTest.keyClick(w, Qt.Key_Right)
    assert got == [], "на екрані футера плеєр не керується"
    w.pstack.tcur = 0


def test_mute_volume_keys_wheel_and_the_saved_volume(qapp, vwin, monkeypatch):
    w = vwin
    w.vol.set(60)
    w._on_volume(60)
    w._on_volume_end(60)
    assert w.btn_snd.icon == "snd" and abs(w.player.volume - 0.6) < 0.02 if hasattr(w.player, "volume") else True
    QTest.keyClick(w, Qt.Key_M)
    assert w.btn_snd.icon == "mute" and w._muted()
    w._on_volume(70)
    assert w.btn_snd.icon == "snd" and not w._muted(), "рух повзунка вмикає звук"
    w.vol.set(0)
    w._on_volume(0)
    assert w.btn_snd.icon == "mute"
    QTest.keyClick(w, Qt.Key_M)
    assert w._muted(), "M при нульовій гучності спершу вмикає «без звуку» (як у Tk)"
    QTest.keyClick(w, Qt.Key_M)
    assert w.vol.v == 50 and not w._muted(), "а вмикаючи звук при нульовій гучності, піднімає її до половини"
    QTest.keyClick(w, Qt.Key_Up)
    assert w.vol.v == 55
    for _ in range(30):
        QTest.keyClick(w, Qt.Key_Down)
    assert w.vol.v == 0
    w.vol.set(40)
    w._save()
    assert json.loads(open(engine.SETTINGS_FILE, encoding="utf-8").read())["ui"]["volume"] == 40


def test_fragment_marks_flags_reset_and_export_range(qapp, vwin, monkeypatch, tmp_path):
    from benvid.qtui import window as W
    w = vwin
    n = w.player.n
    assert w.seek.marks == [0, n - 1] and w._range() is None and not w.btn_rng.enabled
    monkeypatch.setattr(w.player, "position", lambda: 10.0)
    QTest.keyClick(w, Qt.Key_BracketLeft)
    monkeypatch.setattr(w.player, "position", lambda: 40.0)
    QTest.keyClick(w, Qt.Key_BracketRight)
    assert w.seek.marks == [10, 40] and w._range() == (10, 41) and w.btn_rng.enabled
    assert "Фрагмент:" in w.rng_lbl.text_ and "усе відео" not in w.rng_lbl.text_
    out = tmp_path / "frag.mp4"
    names = []
    monkeypatch.setattr(W.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (names.append(a[2]) or str(out), "")))
    kw = {}
    monkeypatch.setattr(w.export, "start", lambda **k: kw.update(k))
    w._export()
    w.player.suspend(False)                              # _export зупинив плеєр для експорту, якого тут не буде
    assert kw["a0"] == 10 and kw["b0"] == 41 and "_bw_" in names[0] and names[0].endswith("s.mp4"), (kw, names)
    w.btn_rng.cmd()
    assert w.seek.marks == [0, n - 1] and w._range() is None and not w.btn_rng.enabled
    w.seek.marks[:] = [5, 20]
    w._update_range()
    w._open(w.path)
    assert w.seek.marks == [0, w.player.n - 1], "новий файл - фрагмент скинуто"


def test_custom_size_is_validated_applied_and_pastes_replace_the_field(qapp, vwin, monkeypatch):
    w = vwin
    sizes = []
    monkeypatch.setattr(w.player, "set_size", lambda k: sizes.append(k))
    before = w.size_key
    for tw, th in (("1", ""), ("5000", ""), ("", ""), ("abc", "1"), ("100", "9999")):
        w.ent_w.setText(tw)
        w.ent_h.setText(th)
        w._apply_custom()
        assert w.size_key == before and "2…4096" in w.prog_lbl.text_, (tw, th)
    w.ent_w.setText("")
    w.ent_h.setText("720")
    QTest.keyClick(w.ent_h, Qt.Key_Return)                   # Enter у полі - те саме, що «OK»
    assert w.size_key == "x720" and sizes[-1] == "x720" and not any(b.on for b in w.size_btns.values())
    w.ent_w.setText("800")
    w.ent_h.setText("")
    w.btn_ok.cmd()
    assert w.size_key == "800x" and sizes[-1] == "800x"
    cb = QApplication.clipboard()
    cb.setText("1920x1080")
    w.ent_w.setText("1")
    w.ent_w.paste_hook(w.ent_w)
    assert (w.ent_w.text(), w.ent_h.text()) == ("1920", "1080"), "«1920x1080» лягає в обидва поля"
    cb.setText("1280х720")                                # кирилична «х»
    w._paste_replace(w.ent_h)
    assert (w.ent_w.text(), w.ent_h.text()) == ("1280", "720")
    cb.setText("abc 12345678\nдруга")
    w._paste_replace(w.ent_w)
    assert w.ent_w.text() == "12345", "у поле розміру лягають лише цифри (до п'яти), лише перший рядок"


def test_saved_custom_size_survives_a_restart_but_a_dead_preset_does_not(qapp, win, tmp_path):
    from benvid.qtui.window import MainWindow
    for saved, want in (("x720", "x720"), ("1280x", "1280x"), ("2K", "Авто"), ("720p", "720p"), (None, "Авто")):
        (tmp_path / "settings.json").write_text(json.dumps({"ui": {"lang": "uk", "size": saved}} if saved else {"ui": {"lang": "uk"}}), encoding="utf-8")
        w2 = MainWindow()
        try:
            assert w2.size_key == want, (saved, w2.size_key)
        finally:
            w2.close()


def test_auto_size_is_picked_in_the_background_and_a_stale_answer_is_ignored(qapp, vwin, monkeypatch):
    w = vwin
    monkeypatch.setattr(w.player, "auto_pick", lambda: "480p")
    sizes = []
    real = w.player.set_size
    monkeypatch.setattr(w.player, "set_size", lambda k: (sizes.append(k), real(k)))
    w._set_size("720p")
    assert w._auto_size is None and sizes[-1] == "720p"
    w._set_size(AUTO)
    assert sizes[-1] == "240p", "поки рахується - тимчасово 240p"
    assert spin(qapp, lambda: w._auto_size == "480p", timeout=10)
    assert sizes[-1] == "480p" and w._size_key() == "480p" and "авто" in w.size_lbl.text_
    tok = w._auto_tok
    w._auto_done(tok - 1, "1080p")
    assert w._auto_size == "480p", "запізнілий токен ігнорується"
    w._set_size("360p")
    w._auto_done(w._auto_tok, "1080p")
    assert w._auto_size is None and w._size_key() == "360p", "вибрали інший розмір - «Авто» вже не діє"


def test_size_label_and_min_size_note_for_ink_and_glyphs(qapp, vwin, monkeypatch):
    from benvid.qtui import sizectl
    w = vwin
    w._set_size("360p")
    t = w.size_lbl.text_
    assert t.startswith("→") and "джерело 160×90" in t and "символ" not in t
    monkeypatch.setattr(w, "_reparam", lambda: None)
    w._set_glyph(True)
    t = w.size_lbl.text_
    assert "символів" in t and "px)" in t and "Щільність знаків" in w.dens_lbl.text_ and "клітинка" in w.dens_lbl.text_
    monkeypatch.setattr(sizectl, "out_size", lambda *a: (1, 1))
    w._note_min_size()
    assert "Розмір піднято до одного символа" in w.prog_lbl.text_
    w._set_glyph(False)
    assert "символів" not in w.size_lbl.text_
    i18n.set_lang("en")
    try:
        w._tr()
        assert w.size_lbl.text_.startswith("→") and "source" in w.size_lbl.text_.lower()
    finally:
        i18n.set_lang("uk")
        w._tr()


def test_level_irnm1_branch_steps_aside_in_glyphs_and_comes_back(qapp, win, monkeypatch):
    started = []
    monkeypatch.setattr(win, "_neuro_start", lambda: started.append(1))
    win._set_level(BRANCH_LEVEL())
    assert win.params.method == BRANCH_LEVEL() and started == [1]
    win._set_glyph(True)
    assert win.params.method == 5 and win._branch_back and win.params.glyph
    win._set_glyph(False)
    assert win.params.method == BRANCH_LEVEL() and not win._branch_back
    win._set_glyph(True)
    win._set_level(3)
    win._set_glyph(False)
    assert win.params.method == 3, "вибір рівня в символах скасовує повернення IRNM1"


def BRANCH_LEVEL():
    from benvid.core import BRANCH
    return BRANCH


def test_fps_limit_list_typed_number_and_what_is_remembered(qapp, vwin, monkeypatch):
    w = vwin
    lims = []
    monkeypatch.setattr(w.player, "set_fps_limit", lambda v: lims.append(v))
    src = w._src_fps()
    assert abs(src - 23.976) < 0.01
    items = [w.fps_cb.itemText(i) for i in range(w.fps_cb.count())]
    assert "оригіналі" in items[0] and "23.976" in items[0] and items[1:] == ["Кадри: 20 к/с", "Кадри: 15 к/с", "Кадри: 12 к/с", "Кадри: 10 к/с"]
    assert w._fps_limit() is None and w.fps_cb.currentIndex() == 0
    w._on_fps_pick(2)
    assert w.fps_sel == 15.0 and w._fps_limit() == 15.0 and lims[-1] == 15.0
    le = w.fps_cb.lineEdit()
    le.setText("7,5")
    w._on_fps_typed()
    assert w.fps_sel == 7.5 and lims[-1] == 7.5 and "7.5" in w.fps_cb.currentText()
    le.setText("999")
    w._on_fps_typed()
    assert w.fps_sel == 0.0 and w._fps_limit() is None and lims[-1] is None and w.fps_cb.currentIndex() == 0, "від частоти відео й вище - як в оригіналі"
    le.setText("нічого")
    w._on_fps_pick(3)
    le.setText("нічого")
    w._on_fps_typed()
    assert w.fps_sel == 12.0, "не число - лишається попереднє"
    n = len(lims)
    w._on_fps_typed()
    assert len(lims) == n, "нічого не вписано - вибір не чіпаємо"
    le.setText("Кадри: 25 к/с")
    le.select_value()
    assert le.selectedText() == "25 к/с", "виділяється лише значення після двокрапки"


def test_fps_limit_does_not_apply_to_photos_and_gifs_and_goes_to_the_export(qapp, vwin, monkeypatch, tmp_path):
    from benvid.qtui import window as W
    w = vwin
    w._on_fps_pick(1)
    assert w.fps_sel == 20.0
    out = tmp_path / "x.mp4"
    monkeypatch.setattr(W.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    kw = {}
    monkeypatch.setattr(w.export, "start", lambda **k: kw.update(k))
    w._export()
    w.player.suspend(False)
    assert kw["fps"] == 20.0
    w.kind = "image"
    assert w._src_fps() == 0.0 and w._fps_limit() is None, "для не-відео ліміт не діє"
    w.kind = "video"


def test_format_quality_audio_fps_and_size_are_saved(qapp, vwin):
    w = vwin
    fmts = [w.fmt_cb.itemData(i) for i in range(w.fmt_cb.count())]
    assert len(fmts) >= 2
    w.fmt_cb.setCurrentIndex(1)
    w._on_fmt_pick(1)
    w.q_cb.setCurrentIndex(w.q_cb.count() - 1)
    w._on_q_pick(w.q_cb.count() - 1)
    w.btn_aud.set_on(False)
    w._on_fps_pick(3)
    w._set_size("360p")
    w._save()
    ui = json.loads(open(engine.SETTINGS_FILE, encoding="utf-8").read())["ui"]
    assert ui["fmt"] == fmts[1] and ui["quality"] == w.q_cb.currentData() and ui["audio"] is False and ui["fps"] == 12.0 and ui["size"] == "360p"
    w.kind = "gif"                                            # для GIF/фото формат відео в налаштуваннях не перезаписується
    w._fill_formats()
    w._save()
    assert json.loads(open(engine.SETTINGS_FILE, encoding="utf-8").read())["ui"]["fmt"] == fmts[1]
    w.kind = "video"


def test_bake_all_enters_the_lock_and_normal_mode_returns_to_the_original(qapp, vwin, tmp_path, monkeypatch):
    from benvid.qtui import window as W
    w = vwin
    orig = w.path
    out = tmp_path / "clip_bw.ikr"
    monkeypatch.setattr(W.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    w._set_level(3)
    w._set_size("360p")
    w._export(freeze=True)
    assert w.export.running and w._freeze_out == str(out)
    assert spin(qapp, lambda: not w.export.running and w.locked, timeout=180), w.prog_lbl.text_
    assert out.exists() and w.path == str(out) and w.btn_lock.text == "Звичайний режим"
    assert not w.m_slider.enabled and not w.btn_ink.enabled and not w.btn_ok.enabled and not w.ent_w.isEnabled()
    assert not any(b.on for b in w.size_btns.values()) and "ЗАМОК" in w.sec_q.title.text_ and w.m_slider.v == 3
    assert "пораховане" in w.prog_lbl.text_
    for act in (lambda: w._set_size("720p"), lambda: w._set_level(5), lambda: w._set_glyph(True), lambda: w._apply_custom(), lambda: w._apply_gdens(1)):
        act()
        assert "Режим замка" in w.prog_lbl.text_ and w.params.method == 3 and not w.params.glyph and w.size_key == "360p"
        w._say()
    assert w.btn_lock.text == "Звичайний режим" and w.fmt_cb.isVisibleTo(w)
    w._lock_toggle()                                     # «Звичайний режим»
    assert spin(qapp, lambda: not w.locked and w.path == orig, timeout=60), w.prog_lbl.text_
    assert w.m_slider.enabled and w.btn_ink.enabled and w.btn_lock.text == "Порахувати все" and w.params.method == 3 and w.size_key == "360p"
    assert "Звичайний режим: оригінал відкрито" in w.prog_lbl.text_ and "ЗАМОК" not in w.sec_q.title.text_


def test_normal_mode_asks_for_the_original_when_it_is_gone_and_cancel_keeps_the_lock(qapp, vwin, tmp_path, monkeypatch):
    from benvid.qtui import filectl, window as W
    w = vwin
    out = tmp_path / "gone.ikr"
    monkeypatch.setattr(W.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    w._export(freeze=True)
    assert spin(qapp, lambda: not w.export.running and w.locked, timeout=180)
    w.player.baked_meta.setdefault("src", {})["path"] = str(tmp_path / "нема.mp4")
    asked = []
    monkeypatch.setattr(filectl.QMessageBox, "question", staticmethod(lambda *a, **k: (asked.append(a[2]), filectl.QMessageBox.Cancel)[1]))
    w._lock_toggle()
    assert asked and "Оригіналу" in asked[0] and w.locked, "без оригіналу питаємо, скасування лишає замок"


def test_open_reports_a_file_note_and_remembers_the_folder_and_clicking_the_result_opens_it(qapp, win, tmp_path, monkeypatch):
    from benvid.qtui import filectl
    from PIL import Image
    p = tmp_path / "ф.png"
    Image.fromarray((np.random.RandomState(2).rand(40, 60) * 255).astype(np.uint8)).save(p)
    assert win._last_dir is None
    win._open(str(p))
    assert win._last_dir == str(tmp_path) and win._src_dir() == str(tmp_path)
    opened = []
    monkeypatch.setattr(filectl.os, "startfile", lambda d: opened.append(d), raising=False)
    win._last_out = str(tmp_path / "res.png")
    win._say("Готово", done=True)
    win.prog_lbl.clicked()
    assert opened == [str(tmp_path)] and win._out_dir == str(tmp_path)
    win._say("Помилка", done=False)
    opened.clear()
    win.prog_lbl.clicked()
    assert opened == [], "клік працює лише по підсумку успішного експорту"
    assert win._short_path(str(tmp_path / "a" / "b" / "дуже-довга-назва-файлу-для-перевірки-скорочення.mp4"), 20).startswith("…")


def test_failed_open_shows_the_error_and_leaves_the_window_unchanged(qapp, win, tmp_path, monkeypatch):
    from benvid.qtui import filectl
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"not a video")
    shown = []
    monkeypatch.setattr(filectl.QMessageBox, "critical", staticmethod(lambda *a, **k: shown.append(a[2])))
    win._open(str(bad))
    assert shown and not win.ready


def test_window_seek_slider_is_throttled(qapp, win, monkeypatch):
    seeks = []
    monkeypatch.setattr(win.player, "seek", lambda f: seeks.append(int(f)))
    win.ready = True
    win._seek_t = 0.0
    t0 = time.perf_counter()
    for i in range(300):
        win._on_seek_drag(float(i))
        time.sleep(0.0004)
    spent = time.perf_counter() - t0
    assert 1 <= len(seeks) <= int(spent / win.SEEK_GAP) + 3
    win._on_seek_end(999.0)
    assert seeks[-1] == 999
    win.ready = False
