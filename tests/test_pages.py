"""Сторінка з екранів (програма, футер): PageStack - прокрутка, що не повзе по пікселях, а плавно перемикає екран на все вікно й
фіксується на ньому; один жест = один перехід, назад - так само.

Є екран (xvfb) - тести відкривають справжні вікна, інакше пропускаються:  xvfb-run -a python -m pytest tests/test_pages.py"""
from __future__ import annotations

import gc
import os
import sys
import time
from types import SimpleNamespace

import pytest
from conftest import pump, settle

pytestmark = pytest.mark.skipif(sys.platform.startswith("linux") and not os.environ.get("DISPLAY"),
                                reason="немає екрана (запускай під xvfb-run)")

VH = 300


@pytest.fixture()
def rig():
    """два екрани в «вікні» 400 x 300: висоти екранів задаються списком rig.hs (за замовчуванням обидва на все вікно)"""
    import tkinter as tk
    from benvid import gui

    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"Tk недоступний: {e}")
    root.geometry(f"412x{VH}+0+0")
    view = tk.Frame(root, width=412, height=VH)
    view.pack(fill="both", expand=True)
    view.pack_propagate(False)
    a, b = tk.Frame(view, bg="#111111"), tk.Frame(view, bg="#eeeeee")
    bar = gui.PixScroll(view, gui.SB_W)
    r = SimpleNamespace(root=root, view=view, a=a, b=b, bar=bar, hs=[None, None])
    r.stack = gui.PageStack(view, bar, [(a, lambda w, h: r.hs[0] or h, None), (b, lambda w, h: r.hs[1] or h, None)])
    root.update()
    r.stack.layout()
    root.update()
    yield r
    r.stack._stop_anim()
    root.destroy()
    # осиротілий Tk-інтерпретатор має вмерти ЗАРАЗ і в цьому потоці: інакше збирач сміття знищить його в чужому (потік плеєра в
    # наступних тестах) - Tcl такого не терпить і валить процес
    r.__dict__.clear()
    del a, b, bar, view, root, r
    gc.collect()


def arrive(r, timeout=3.0):
    """дочекатися кінця анімації; -> час, що пройшов"""
    t0 = time.perf_counter()
    assert pump(r.root, lambda: r.stack.idle, timeout), "анімація не закінчилась"
    return time.perf_counter() - t0


def wait(r, s):
    end = time.perf_counter() + s
    while time.perf_counter() < end:
        r.root.update()
        time.sleep(0.01)


def y_of(r, w):
    return w.winfo_y()


def test_two_full_screens(rig):
    st = rig.stack
    assert st.tops == [0, VH] and st.exts == [0, 0] and st.max_off == VH
    assert (st.cur, st.off) == (0, 0)
    assert rig.a.winfo_height() == VH and rig.b.winfo_height() == VH
    assert y_of(rig, rig.a) == 0 and y_of(rig, rig.b) == VH
    assert rig.bar.winfo_ismapped(), "смужка сторінки завжди видна: підказка, що внизу є ще екран"
    assert rig.bar.last - rig.bar.first == pytest.approx(0.5), "повзунок - половина доріжки: екранів два"


def test_wheel_down_flips_whole_screen_with_animation_and_locks(rig):
    st = rig.stack
    seen = []
    orig = st.layout

    def spy():
        orig()
        seen.append(st.off)
    st.layout = spy
    st.notch(-1)                                         # колесо «вниз» (d < 0)
    took = arrive(rig)
    st.layout = orig
    assert (st.cur, st.off) == (1, VH) and y_of(rig, rig.b) == 0 and y_of(rig, rig.a) == -VH
    assert 0.25 < took < 1.2, took                      # справжня анімація, а не стрибок
    assert len(set(seen)) >= 6 and seen == sorted(seen), "проміжні кадри ростуть монотонно"
    st.notch(-1)                                         # далі нікуди: футер - останній екран
    assert arrive(rig) < 0.2 and st.cur == 1


def test_one_gesture_is_one_flip(rig):
    st = rig.stack
    st.notch(-1)
    st.notch(1)                                          # одразу «назад» тим самим жестом: ігнорується
    st.notch(1)
    arrive(rig)
    assert st.cur == 1
    wait(rig, st.COOL + 0.1)
    st.notch(1)                                          # новий жест - назад
    arrive(rig)
    assert (st.cur, st.off) == (0, 0)


def test_back_by_wheel_page_keys_and_programmatic_go(rig):
    st = rig.stack
    st.step(1)
    arrive(rig)
    assert st.cur == 1
    wait(rig, st.COOL + 0.1)
    st.step(-1)
    arrive(rig)
    assert st.cur == 0
    st.go(1)
    arrive(rig)
    st.go(0)                                             # програмний перехід не чекає «остигання»
    arrive(rig)
    assert (st.cur, st.off) == (0, 0)


def test_go_calls_done_once_when_arrived(rig):
    st = rig.stack
    calls = []
    st.go(1, done=lambda: calls.append(st.off))
    assert not calls
    arrive(rig)
    assert calls == [VH]
    st.go(1, done=lambda: calls.append("again"))         # вже там: миттєво
    assert calls == [VH, "again"]


def test_tall_screen_scrolls_inside_before_flipping(rig):
    st = rig.stack
    rig.hs[0] = VH + 200                                 # низьке вікно: програма не стискається, лишок - 200 px
    st.layout()
    assert st.exts[0] == 200 and st.tops[1] == VH + 200
    for _ in range(3):
        st.notch(-1)
    arrive(rig)
    assert (st.cur, st.rel) == (0, 192)                  # 3 клацання по NOTCH = 192
    st.notch(-1)
    arrive(rig)
    assert (st.cur, st.rel) == (0, 200), "до самого краю екрана"
    st.notch(-1)                                         # той самий жест не перекидає далі
    arrive(rig)
    assert st.cur == 0
    wait(rig, st.EDGE + 0.1)
    st.notch(-1)
    arrive(rig)
    assert (st.cur, st.rel) == (1, 0)
    wait(rig, st.COOL + 0.1)
    st.notch(1)                                          # назад - у низ головного екрана, де щойно були
    arrive(rig)
    assert (st.cur, st.rel) == (0, 200)


def test_resize_keeps_the_screen(rig):
    st = rig.stack
    st.go(1)
    arrive(rig)
    rig.view.configure(height=VH + 80)
    rig.root.geometry(f"412x{VH + 80}")
    rig.root.update()
    st.layout()
    assert st.cur == 1 and y_of(rig, rig.b) == 0 and st.off == st.tops[1], "футер лишається на екрані після зміни розміру"
    assert rig.b.winfo_height() == VH + 80


def test_bar_drag_follows_then_snaps_to_nearest(rig):
    st = rig.stack
    rig.bar.cmd(0.4)                                     # тягнуть повзунок: екрани їдуть за ним
    assert st.off == int(round(0.4 * st.max_off)) and not st.idle
    rig.bar.release_cmd()                                # відпустили: ближче до першого - стає на нього
    arrive(rig)
    assert (st.cur, st.off) == (0, 0)
    rig.bar.cmd(0.7)
    rig.bar.release_cmd()
    arrive(rig)
    assert (st.cur, st.off) == (1, VH)


def test_bar_track_click_flips(rig):
    st = rig.stack
    rig.bar.page_cmd(1)
    arrive(rig)
    assert st.cur == 1


def test_scroll_to_maps_to_stops(rig):
    st = rig.stack
    st.scroll_to(10 ** 6)
    assert (st.cur, st.off) == (1, VH)
    st.scroll_to(0)
    assert (st.cur, st.off) == (0, 0)
    st.scroll_to(100)                                    # між екранами - той, до якого ближче
    assert (st.cur, st.off) == (0, 0)
    st.scroll_to(200)
    assert (st.cur, st.off) == (1, VH)


def test_disabled_shows_only_the_first_screen(rig):
    st = rig.stack
    st.go(1)
    arrive(rig)
    st.set_enabled(False)                                # повноекранний режим
    assert (st.cur, st.off) == (0, 0) and not rig.bar.winfo_ismapped()
    st.notch(-1)
    st.step(1)
    assert st.idle and st.off == 0
    st.set_enabled(True)
    rig.root.update()
    assert st.cur == 0 and rig.bar.winfo_ismapped()


# ------------------------------------------------------------------ у справжньому вікні
def _wheel_over(a, w, delta):
    a.update()
    ev = SimpleNamespace(x_root=w.winfo_rootx() + min(8, max(1, w.winfo_width() // 2)),
                         y_root=w.winfo_rooty() + min(8, max(1, w.winfo_height() // 2)), delta=delta, num=0)
    return a._on_wheel(ev)


def _settled(a, timeout=3.0):
    assert pump(a, lambda: a.pstack.idle, timeout)
    settle(a, rounds=2)


def test_footer_is_a_screen_of_its_own(app):
    from benvid.gui import FOOT_GAP, M
    app._set_lang("uk")
    app.pstack.scroll_to(10 ** 6)
    settle(app)
    page, foot, col = (tuple(f(w) for f in (lambda w: w.winfo_rootx(), lambda w: w.winfo_rooty(), lambda w: w.winfo_width(),
                                          lambda w: w.winfo_height())) for w in (app.page, app.foot, app.foot_col))
    assert foot[1] == page[1] and foot[3] == page[3], "футер - екран на все вікно, а не хвіст під програмою"
    assert not app.main.winfo_viewable() or app.main.winfo_rooty() + app.main.winfo_height() <= page[1], "програми над ним не видно"
    top_gap, bottom_gap = col[1] - foot[1], (foot[1] + foot[3]) - (col[1] + col[3])
    assert abs(top_gap - bottom_gap) <= 2, "вміст рівно по центру екрана по вертикалі"
    assert top_gap >= M + 12, "над вмістом є місце для стрічки плівки"
    assert app._foot_logo is not None and app._foot_mascot is not None
    logo, mascot = app._foot_logo, app._foot_mascot
    assert logo.winfo_rooty() < app.guide_tabs["start"].winfo_rooty() - FOOT_GAP + 1, "котушка й логотип - над гайдом"
    mid = (mascot.winfo_rootx() + logo.winfo_rootx() + logo.winfo_width()) / 2
    assert abs(mid - (col[0] + col[2] / 2)) <= 2, "шапка футера по центру колонки"
    app.pstack.scroll_to(0)
    settle(app)


def test_help_button_and_f1_toggle_between_screens(app):
    app.pstack.scroll_to(0)
    settle(app)
    app.btn_help.cmd()
    _settled(app)
    assert app.pstack.cur == 1 and app.foot.winfo_rooty() == app.page.winfo_rooty()
    app._goto_guide()                                    # ще раз - назад
    _settled(app)
    assert app.pstack.cur == 0 and app.main.winfo_rooty() == app.page.winfo_rooty()


def test_up_button_returns_to_the_app(app):
    app.pstack.scroll_to(10 ** 6)
    settle(app)
    app.btn_top.cmd()
    _settled(app)
    assert app.pstack.cur == 0


def test_wheel_flips_over_blank_areas_but_not_over_the_picture_or_volume(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "_vol_step", lambda d, key=False: calls.append(("vol", d)))
    monkeypatch.setattr(app, "_zoom_step", lambda d, *a: calls.append(("zoom", d)))
    app.pstack.scroll_to(0)
    settle(app)
    _wheel_over(app, app.vol, -120)                      # над повзунком звуку - гучність, а не екрани
    assert calls == [("vol", -5)] and app.pstack.idle and app.pstack.cur == 0
    _wheel_over(app, app.screen, -120)                   # над картинкою - масштаб
    assert calls[-1] == ("zoom", -1) and app.pstack.idle and app.pstack.cur == 0
    n = len(calls)
    _wheel_over(app, app.seek, -120)                     # над рештою - перехід на футер
    _settled(app)
    assert app.pstack.cur == 1 and len(calls) == n
    time.sleep(app.pstack.COOL + 0.1)
    _wheel_over(app, app.guide_lbl, 120)                 # колесо вгору над гайдом - назад
    _settled(app)
    assert app.pstack.cur == 0


def test_page_keys(app):
    app.pstack.scroll_to(0)
    settle(app)
    app.pstack.step(1)
    _settled(app)
    assert app.pstack.cur == 1
    time.sleep(app.pstack.COOL + 0.1)
    app.pstack.step(-1)
    _settled(app)
    assert app.pstack.cur == 0


def test_quote_click_goes_back_up_then_runs_the_egg(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "_egg_defocus", lambda: calls.append(app.pstack.cur))
    app.pstack.scroll_to(10 ** 6)
    settle(app)
    app._quote_click()
    assert not calls, "поки їде нагору - ще не розфокусовується"
    _settled(app)
    assert calls == [0] and app.pstack.cur == 0


def test_footer_fits_the_smallest_window_in_both_languages(app):
    """на мінімальному вікні 1000 x 640 футер влазить без внутрішньої прокрутки (найвищий розділ гайда - в обох мовах)"""
    from benvid.gui import MIN_MAIN
    for lang in ("uk", "en"):
        app._set_lang(lang)
        for sec in app.guide_tabs:
            app._guide_pick(sec)
            settle(app, rounds=2)
            assert app.foot.winfo_reqheight() <= 640 - 4, (lang, sec, app.foot.winfo_reqheight())
    assert MIN_MAIN <= 640
    app._set_lang("uk")
    app._guide_pick("start")
    settle(app)


def test_every_tip_takes_one_line_in_the_footer_of_the_smallest_window(app):
    """«Порада» під філософією - один рядок для кожної поради в обох мовах: інакше висота футера залежить від того, яка порада випала (випадково), і на
    мінімальному вікні 640 px він міг би то влізти, то ні. Колонка футера - для мінімальної ширини вікна 1000 px"""
    from benvid import guide
    for lang in ("uk", "en"):
        app._set_lang(lang)
        app._guide_pick("start")
        settle(app, rounds=2)
        app._foot_key = None
        app._foot_fit(1000)
        heights = {}
        for i in range(len(guide.TIPS[lang])):
            app._tip_n = i
            app._fill_tip()
            app.update_idletasks()
            heights[i] = app.tip_lbl.winfo_reqheight()
        assert len(set(heights.values())) == 1, (lang, {i: h for i, h in heights.items() if h != min(heights.values())})
    app._set_lang("uk")
    app._foot_key = None
    settle(app)


def test_opening_a_file_brings_back_the_app_screen(app, tmp_path):
    """відкрили файл, поки видно футер, - програма сама повертається на головний екран: картинка й панелі там"""
    from PIL import Image
    p = str(tmp_path / "pic.png")
    Image.new("RGB", (160, 90), (90, 90, 90)).save(p)
    app.pstack.scroll_to(10 ** 6)
    settle(app)
    assert app.pstack.cur == 1
    app._open(p)
    assert pump(app, lambda: app.ready and app._last_mask is not None, 60), "файл не відкрився"
    _settled(app)
    assert app.pstack.cur == 0 and app.main.winfo_rooty() == app.page.winfo_rooty()
