"""Поведінка вікна, а не розкладка (її перевіряє test_layout): «навпаки» перефарбовує весь інтерфейс, ручка повзунка гучності не
вилазить за край, збільшену картинку можна тягти, кнопка відтворення ставить рендер на паузу, на екрані видно, що робить програма.

Потрібен екран (Linux: xvfb-run -a python -m pytest tests/test_gui_behaviour.py); налаштування користувача не чіпаються."""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pytest
from conftest import NO_SCREEN, pump, settle

pytestmark = pytest.mark.skipif(NO_SCREEN, reason="немає екрана (запускай під xvfb-run)")


def _rgb(app, c):
    return tuple(app.winfo_rgb(c))


def _widget_colours(app):
    """{(віджет, параметр): колір} для всіх звичайних віджетів вікна (ttk має свої стилі й тут не береться)"""
    import tkinter as tk
    out = {}

    def walk(w):
        for o in ("bg", "fg", "highlightbackground", "activebackground", "insertbackground"):
            try:
                v = str(w.cget(o))
            except (tk.TclError, AttributeError):
                continue
            out[(str(w), o)] = _rgb(app, v)
        for c in w.winfo_children():
            walk(c)
    walk(app)
    return out


def _canvas_colours(app):
    """усі кольори, якими намальовано предмети полотен (кнопки, повзунки, смужки прокрутки, прогрес, плашка)"""
    import tkinter as tk
    found = set()

    def walk(w):
        if isinstance(w, tk.Canvas):
            for i in w.find_all():
                for o in ("fill", "outline"):
                    try:
                        v = w.itemcget(i, o)
                    except tk.TclError:
                        continue
                    if v:
                        try:
                            found.add(_rgb(app, v))
                        except tk.TclError:
                            pass
        for c in w.winfo_children():
            walk(c)
    walk(app)
    return found


def test_inversion_recolours_the_whole_window(app):
    """кнопка «навпаки» діє не лише на картинку: тло й текст вікна міняються місцями («Класика~» - біле вікно з чорним текстом),
    і жоден віджет не лишається в старих кольорах"""
    from benvid import gui
    assert not app.pal_inv and app.pal_var.get() == "Класика"
    black, white, dim0, dim1 = (_rgb(app, c) for c in ("#000000", "#ffffff", "#9a9a9a", "#656565"))
    before = _widget_colours(app)
    assert black in {v for v in before.values()}
    app._toggle_inv()
    settle(app)
    try:
        assert app.pal_inv and (gui.BG, gui.FG) == ("#ffffff", "#000000")
        after = _widget_colours(app)
        swap = {black: white, white: black, dim0: dim1}
        wrong = [(k, v, after[k]) for k, v in before.items() if v in swap and k in after and after[k] != swap[v]]
        assert not wrong, wrong[:6]
        assert app.cget("bg") == "#ffffff" and app.main.cget("bg") == "#ffffff" and app.screen.cget("bg") == "#ffffff"
        used = _canvas_colours(app)
        assert dim0 not in used, "на полотнах лишилися старі кольори"
        assert used <= {black, white, dim1}, used
    finally:
        app._toggle_inv()
        settle(app)
    assert not app.pal_inv and (gui.BG, gui.FG) == ("#000000", "#ffffff")
    assert _widget_colours(app) == before, "після другого натискання вікно має повернутися до початкових кольорів"


def test_inversion_works_with_colour_palettes_and_keeps_text_readable(app):
    from benvid import gui, palette
    for name in ("Синька", "Бурштин", "Термінал"):
        for inv in (False, True):
            app.pal_var.set(name)
            app.pal_inv = inv
            app.btn_pinv.set_on(inv)
            app._set_palette()
            settle(app, rounds=2)
            h = lambda s: tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))
            assert palette.contrast(h(gui.FG), h(gui.BG)) >= 4.5, (name, inv)
            assert (gui.BG, gui.FG) == (palette.theme(palette.key(name, inv))["BG"], palette.theme(palette.key(name, inv))["FG"])
    app.pal_var.set("Класика")
    app.pal_inv = False
    app.btn_pinv.set_on(False)
    app._set_palette()
    settle(app, rounds=2)
    assert (gui.BG, gui.FG) == ("#000000", "#ffffff")


def test_volume_thumb_never_leaves_the_slider(app):
    """ручка повзунка гучності при максимумі (й мінімумі) лишається повністю на полотні: раніше її правий край «з'їдався»"""
    v = app.vol
    settle(app, rounds=2)
    W = v.winfo_width()
    assert W > 40
    for val in (0, 37, 100):
        v.set(val)
        v.draw()
        app.update_idletasks()
        thumb = v.find_all()[-1]                       # ручка малюється останньою - поверх усього
        x0, y0, x1, y1 = v.coords(thumb)
        assert x0 >= 0 and x1 <= W - 1, (val, x0, x1, W)       # контур прямокутника займає й стовпчик x1, тож x1 <= W-1
        assert y0 >= 0 and y1 <= v.winfo_height() - 1
    v.set(100)
    v.draw()
    assert v.coords(v.find_all()[-1])[2] == W - 1, "на максимумі ручка впритул до правого краю"
    # колесо й клавіші теж доходять до краю
    app._on_volume(100)
    assert int(v.get()) == 100


def _photo(path, w=320, h=240):
    from PIL import Image, ImageDraw
    im = Image.new("L", (w, h), 30)
    d = ImageDraw.Draw(im)
    for i in range(0, w, 16):
        d.rectangle([i, 0, i + 7, h], fill=60 + (i * 3) % 190)
    d.ellipse([w // 2 - 70, h // 2 - 70, w // 2 + 70, h // 2 + 70], fill=235)
    im.convert("RGB").save(path)


def _open_and_wait(app, path):
    app._open(path)
    assert pump(app, lambda: app.ready and app._last_mask is not None, 60), "файл не відкрився"
    settle(app, rounds=3)


def _mouse(app, kind, x, y, **kw):
    app.screen.event_generate(kind, x=x, y=y, **kw)
    app.update()


def test_drag_pans_the_zoomed_picture_and_a_click_keeps_the_zoom(app, tmp_path):
    """збільшену картинку можна рухати перетягуванням (раніше цифра «1» у прив'язці читалась як ЛІВА КНОПКА МИШІ, і кожен клік
    скидав масштаб); клавіші 1 / 0 - 100% і «вписати»"""
    p = str(tmp_path / "photo.png")
    _photo(p)
    _open_and_wait(app, p)
    app._zoom_to(8)
    settle(app, rounds=2)
    assert app._zoom == 8
    vc0 = app._vc
    assert app.screen.cget("cursor") == "hand2", "над збільшеною картинкою, що не влізла, - рука"
    _mouse(app, "<ButtonPress-1>", 300, 300)
    for i in range(1, 7):
        _mouse(app, "<B1-Motion>", 300 - i * 20, 300 - i * 10, state=0x100)
    assert app.screen.cget("cursor") == "fleur"
    _mouse(app, "<ButtonRelease-1>", 180, 240)
    assert app._zoom == 8, "перетягування не повинно міняти масштаб"
    assert app._vc[0] > vc0[0] and app._vc[1] > vc0[1], (vc0, app._vc)       # потягли вліво-вгору: видно правіше й нижче
    z, vc = app._zoom, app._vc
    _mouse(app, "<ButtonPress-1>", 200, 200)                                   # звичайний клік без руху
    _mouse(app, "<ButtonRelease-1>", 200, 200)
    assert app._zoom == z and app._vc == vc, "клік не скидає масштаб і не зсуває"
    # назад: потягли в протилежний бік - повернулись
    _mouse(app, "<ButtonPress-1>", 100, 100)
    _mouse(app, "<B1-Motion>", 160, 130, state=0x100)
    _mouse(app, "<B1-Motion>", 220, 160, state=0x100)
    _mouse(app, "<ButtonRelease-1>", 220, 160)
    assert app._vc[0] < vc[0] and app._vc[1] < vc[1]
    # клавіші
    app.focus_force()
    app.update()
    app.event_generate("<Key-1>")
    app.update()
    assert app._zoom == 1
    app.event_generate("<Key-0>")
    app.update()
    assert app._zoom is None, "0 - вписати"
    app._zoom_step(1)
    assert app._zoom is not None
    app.screen.event_generate("<Button-2>", x=50, y=50)
    app.update()
    assert app._zoom is None, "середня кнопка - вписати"


def test_dragging_a_picture_that_fits_does_nothing(app, tmp_path):
    p = str(tmp_path / "photo2.png")
    _photo(p, 64, 48)
    _open_and_wait(app, p)
    assert app._zoom is None
    _mouse(app, "<ButtonPress-1>", 300, 300)
    _mouse(app, "<B1-Motion>", 200, 250, state=0x100)
    _mouse(app, "<ButtonRelease-1>", 200, 250)
    assert app._zoom is None and tuple(app._vc) == (0.5, 0.5)
    assert app.screen.cget("cursor") == ""


def _clip(path, n=300, w=320, h=180, fps=24):
    import cv2
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        f = np.dstack([(xx * 255 // w + i * 2) % 256, (yy * 255 // h), (xx + yy + i * 3) % 256]).astype(np.uint8)
        cv2.circle(f, (30 + (i * 3) % (w - 60), h // 2), 40, (240, 240, 240), -1)
        vw.write(f)
    vw.release()


def test_play_button_pauses_and_resumes_the_render(app, tmp_path, monkeypatch):
    """під час рендеру кнопка відтворення (і Пробіл) ставить на паузу сам рендер, а не плеєр: прогрес стоїть, значок - «грати», на екрані
    «ПАУЗА»; ще раз - рендер іде далі й доходить до кінця. Годинник на паузі не йде"""
    from tkinter import filedialog
    from benvid import gui
    src, out = str(tmp_path / "v.mp4"), str(tmp_path / "out.mp4")
    _clip(src)
    app.size_var.set("360p")
    _open_and_wait(app, src)
    assert app.kind == "video" and app.btn_play.enabled
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **kw: out)
    app.fmt_var.set("MP4 (H.264)")
    app.q_var.set("Мала")
    app._export()
    assert app.exporting and app._pausable
    try:
        assert app._play_icon() == "pause" and app.btn_play.enabled
        assert pump(app, lambda: app._exp_f > 0.02, 120), "рендер не рушив"
        app._toggle()                                             # ← кнопка (та сама, що викликає Пробіл)
        assert app.pause_ev.is_set() and app._play_icon() == "play"
        text, spin = app._hud_info()
        assert text.startswith("ПАУЗА") and not spin
        assert "[Пробіл]" in app.prog_lbl.cget("text"), "у панелі підказка, як продовжити"
        for _ in range(40):                                        # дати шматкам, що вже в роботі, дорахуватись (на повільному ПК шматок - секунди)
            before = app._exp_f
            pump(app, lambda: False, 1.0)
            if app._exp_f == before:
                break
        f0, el0 = app._exp_f, app._clock.elapsed()
        pump(app, lambda: False, 1.2)
        assert app._exp_f == f0, "на паузі прогрес не рухається"
        assert abs(app._clock.elapsed() - el0) < 0.05, "час на паузі не рахується"
        assert app.exporting
        app.focus_force()
        app.update()
        app.event_generate("<space>")                              # Пробіл знімає паузу
        app.update()
        assert not app.pause_ev.is_set() and app._play_icon() == "pause"
        text, spin = app._hud_info()
        assert text.startswith("РЕНДЕР") and spin
        assert pump(app, lambda: not app.exporting, 240), "рендер не дійшов до кінця після паузи"
    finally:
        if app.exporting:
            app.cancel.set()
            pump(app, lambda: not app.exporting, 60)
    assert os.path.isfile(out) and os.path.getsize(out) > 1000
    assert not app.pause_ev.is_set() and app._play_icon() == "play" and app.btn_play.enabled
    assert app._hud_info() is None or not app._hud_info()[0].startswith(("РЕНДЕР", "ПАУЗА"))


def test_cancel_while_paused_stops_the_export(app, tmp_path, monkeypatch):
    from tkinter import filedialog
    src, out = str(tmp_path / "v2.mp4"), str(tmp_path / "out2.mp4")
    _clip(src)
    app.size_var.set("360p")
    _open_and_wait(app, src)
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **kw: out)
    app._export()
    assert pump(app, lambda: app._exp_f > 0.02, 120)
    app._toggle()
    assert app.pause_ev.is_set()
    app._stop_export()                                             # «Стоп» діє і на паузі
    assert pump(app, lambda: not app.exporting, 30), "скасування на паузі не спрацювало"
    assert not os.path.exists(out)
    assert not app.pause_ev.is_set() and app.btn_open.enabled


def test_screen_hud_tells_what_the_program_is_doing(app, tmp_path, monkeypatch):
    """плашка в куті екрана: прогрів «Гравюри», буферизація, обробка кадра - щоб не здавалось, що програма зависла"""
    from benvid.player import Player
    p = str(tmp_path / "photo3.png")
    _photo(p)
    _open_and_wait(app, p)
    assert app._hud_info() is None, "у спокої плашки нема"
    pl = app.player
    monkeypatch.setattr(Player, "warming", property(lambda s: True))
    text, spin = app._hud_info()
    assert text.startswith("ГОТУЮ ГРАВЮРУ") and spin
    app._hud_update()
    app._redraw()
    assert app.screen.find_withtag("hud"), "плашку намальовано"
    monkeypatch.setattr(Player, "warming", property(lambda s: False))
    monkeypatch.setattr(Player, "waiting_frame", lambda s: True)
    assert app._hud_info() == ("ОБРОБКА…", True)
    monkeypatch.setattr(Player, "waiting_frame", lambda s: False)
    monkeypatch.setattr(Player, "playing", property(lambda s: True))
    monkeypatch.setattr(Player, "state", property(lambda s: "buffering"))
    monkeypatch.setattr(Player, "buffer_progress", lambda s: (7, 24))
    assert app._hud_info() == ("БУФЕРИЗАЦІЯ · 7/24 кадрів", True)
    monkeypatch.setattr(Player, "stride", property(lambda s: 3))
    assert "кожен 3-й" in app._hud_info()[0]
    monkeypatch.undo()
    app._hud_update()
    settle(app, rounds=2)
    assert not app.screen.find_withtag("hud") or app._hud_info() is not None


def test_hud_texts_are_translated(app):
    """плашка англійською теж"""
    from benvid.i18n import tr
    app._set_lang("en")
    try:
        assert tr("ОБРОБКА…") == "PROCESSING…"
        assert tr("ПАУЗА · {pct:.0f}%", pct=42) == "PAUSED · 42%"
        assert tr("РЕНДЕР · кадр {cur}/{tot}", cur=3, tot=9) == "RENDERING · frame 3/9"
    finally:
        app._set_lang("uk")


def test_only_one_tip_at_a_time_inside_the_screen_and_it_expires(app):
    """нова підказка прибирає стару; вікно підказки лежить у межах екрана й побудоване прихованим (без блимання в куті);
    якщо подію «курсор пішов» не доставили - підказка зникає сама"""
    import re
    from benvid import gui
    a, b = gui.Tip(app.btn_open, "перша"), gui.Tip(app.btn_export, "друга")
    try:
        a._show()
        assert a._win is not None and gui.Tip._cur is a
        b._show()
        assert a._win is None and b._win is not None and gui.Tip._cur is b
        pump(app, lambda: False, 0.1)
        m = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", b._win.wm_geometry())
        w, h, x, y = (int(v) for v in m.groups())
        vx, vy, vw, vh = app.winfo_vrootx(), app.winfo_vrooty(), app.winfo_vrootwidth(), app.winfo_vrootheight()
        assert vx <= x and x + w <= vx + vw and vy <= y and y + h <= vy + vh
        gui.Tip.hide_all()
        assert b._win is None and gui.Tip._cur is None
        old = gui.Tip.HIDE
        gui.Tip.HIDE = 80
        try:
            a._show()
            assert a._win is not None
            assert pump(app, lambda: a._win is None, 2.0), "забута підказка мусить зникнути сама"
        finally:
            gui.Tip.HIDE = old
    finally:
        a._leave()
        b._leave()
    assert gui.Tip._cur is None


def test_popups_belong_to_the_window_and_never_cover_other_apps(app):
    """підказка й хмаринка котушки - вікна-«підлеглі» (transient), а не «поверх усіх»; повторний запуск піднімає вікно лише на мить"""
    from benvid import gui
    tip = gui.Tip(app.btn_open, "підказка")
    tip._show()
    try:
        assert tip._win is not None
        assert str(tip._win.attributes("-topmost")).lower() in ("0", "false")
        assert str(tip._win.wm_transient()) == str(app)
    finally:
        tip._leave()
    app._bubble(app.btn_open, "фраза")
    try:
        assert app._bub is not None
        assert str(app._bub.attributes("-topmost")).lower() in ("0", "false")
        assert str(app._bub.wm_transient()) == str(app)
    finally:
        app._bubble_close()
    app.bring_front()
    pump(app, lambda: False, 0.5)
    assert str(app.attributes("-topmost")).lower() in ("0", "false"), "вікно не повинно лишатись «поверх усіх»"
    app.attributes("-topmost", True)                               # навіть якщо щось його поставило - _untop знімає
    app._untop()
    assert str(app.attributes("-topmost")).lower() in ("0", "false")


def test_fps_list_follows_the_video_and_remembers_the_choice(app):
    """список «Кадри»: як в оригіналі + стандартні значення, нижчі за fps відео; вибір пам'ятається й діє лише там, де він нижчий за fps"""
    from benvid import i18n
    from benvid.video import fps_options
    pl = app.player
    old = (pl.fps, app.kind, app.fps_sel)
    try:
        app.kind, app.fps_sel = "video", 24
        pl.fps = 60.0
        app._sync_fps()
        assert app._fps_vals == [0] + fps_options(60.0) and app._fps_limit() == 24
        assert app.fps_cb.current() == app._fps_vals.index(24)
        assert len(app.fps_cb.cget("values")) == len(app._fps_vals) and "24" in app.fps_cb.get()
        pl.fps = 20.0                                         # відео вже повільніше за вибір: ліміт не діє, у списку - як в оригіналі
        app._sync_fps()
        assert app._fps_limit() is None and app.fps_cb.current() == 0 and app.fps_sel == 24, "вибір не губиться"
        app.fps_cb.current(app._fps_vals.index(15))
        app._on_fps_pick()
        assert app.fps_sel == 15 and app._fps_limit() == 15
        app.fps_cb.current(0)
        app._on_fps_pick()
        assert app.fps_sel == 0 and app._fps_limit() is None
        assert i18n.tr("Кадри: {f} к/с", f=30)
    finally:
        pl.fps, app.kind, app.fps_sel = old
        app._sync_fps()


def test_fps_field_accepts_a_typed_number(app):
    """у полі «Кадри» можна вписати своє число: воно стає лімітом, а список і далі працює; сміття й число вище за fps відео нічого не міняють"""
    pl = app.player
    old = (pl.fps, app.kind, app.fps_sel)
    try:
        app.kind, app.fps_sel = "video", 0.0
        pl.fps = 60.0
        app._sync_fps()
        assert str(app.fps_cb.cget("state")) == "normal" and app._fps_limit() is None
        app.fps_cb.set("17")
        app._on_fps_typed()
        assert app.fps_sel == 17.0 and app._fps_limit() == 17.0 and "17" in app.fps_cb.get()
        app.fps_cb.set("23,976")
        app._on_fps_typed()
        assert app._fps_limit() == 23.976
        app.fps_cb.set("abc")                                  # не число: лишається попереднє, поле повертається до нього
        app._on_fps_typed()
        assert app._fps_limit() == 23.976 and "23.976" in app.fps_cb.get()
        app.fps_cb.set("75")                                   # не нижче за fps відео: ліміту немає
        app._on_fps_typed()
        assert app._fps_limit() is None and app.fps_cb.current() == 0
        app.fps_cb.set("24")
        app._on_fps_typed()
        assert app._fps_limit() == 24 and app.fps_cb.current() == app._fps_vals.index(24), "число зі списку - це той самий пункт"
        pl.fps = 20.0                                          # інше відео: вибір пам'ятається, але не діє; пройти через поле його не стирає
        app._sync_fps()
        app._on_fps_typed()
        assert app._fps_limit() is None and app.fps_sel == 24
    finally:
        pl.fps, app.kind, app.fps_sel = old
        app._sync_fps()


def test_letter_hotkeys_work_after_choosing_in_a_pick_only_list_and_clicking_away_from_a_field(app, monkeypatch):
    """після вибору формату/якості (список «лише вибір» лишається у фокусі) F/M/L/I мають працювати; клік по картинці забирає фокус з поля вводу (fps)"""
    got = []
    monkeypatch.setattr(app, "_fullscreen", lambda *a: got.append("f"))
    app.focus_force()
    app.update()
    app.fmt_cb.focus_force()
    app.update()
    app.event_generate("<Key-f>", when="now")
    app.update()
    assert got == ["f"], "список лише для вибору не друкує літер"
    old = (app.player.fps, app.kind)                              # рядок «Кадри» видно лише для відео: показуємо його, щоб у нього можна було встати
    app.player.fps = 30.0
    app._apply_kind("video")
    app._sync_fps()
    settle(app, rounds=3)
    try:
        app.fps_cb.focus_force()
        assert pump(app, app._editing, 3.0), "фокус у полі вводу (оконний менеджер віддає його не одразу)"
        app.event_generate("<Key-f>", when="now")
        app.update()
        assert got == ["f"], "у полі вводу літери - для поля"
        _mouse(app, "<ButtonPress-1>", 50, 50)
        _mouse(app, "<ButtonRelease-1>", 50, 50)
        settle(app, rounds=2)
        assert not app._typing(), "клік по картинці забрав фокус з поля"
    finally:
        app.player.fps = old[0]
        app._apply_kind(old[1] or "video")
        app._sync_fps()


def test_leaving_fullscreen_returns_a_normal_window_not_a_full_screen_one(app, monkeypatch):
    """вихід з повного екрана: вікно було розгорнуте/на весь екран - стає звичайним (≈72% × 78% монітора, по центру), а не лишається розміром на весь екран"""
    monkeypatch.setattr(app, "winfo_screenwidth", lambda: 2000)
    monkeypatch.setattr(app, "winfo_screenheight", lambda: 1200)
    app._fs = False
    app._pre_fs = ("zoomed", 2000, 1160, 0, 0)
    app._leave_fullscreen_size()
    app.update()
    assert app.state() == "normal"
    w, h = (int(v) for v in app.geometry().split("+")[0].split("x"))
    assert (w, h) == (max(app.minsize()[0], 1440), max(app.minsize()[1], 936))
    app._pre_fs = ("normal", 1100, 700, 40, 30)                  # розмір, що був до входу й не займав весь екран, повертається як був
    app._leave_fullscreen_size()
    app.update()
    assert app.geometry().split("+")[0] == "1100x700"


def test_f11_edge_button_slides_the_right_panel_out(app):
    """у звичайному вікні панель завжди на місці й кнопки справа немає; в F11 панель схована, вузька кнопка на правому краю висуває її (картинка
    віддає місце) і ховає назад; вихід з F11 повертає панель"""
    settle(app, rounds=2)
    assert app.side.winfo_ismapped() and not app.edge.winfo_ismapped()
    app._window_fullscreen()
    settle(app, rounds=3)
    assert app._wfs and not app.side.winfo_ismapped() and app.edge.winfo_ismapped()
    assert app.edge.winfo_rootx() + app.edge.winfo_width() >= app.winfo_rootx() + app.winfo_width() - 30, "кнопка на правому краю"
    w0 = app.screen.winfo_width()
    app.edge.event_generate("<ButtonPress-1>", x=5, y=5)             # справжній клік по кнопці
    app.edge.event_generate("<ButtonRelease-1>", x=5, y=5)
    settle(app, rounds=3)
    assert app.side.winfo_ismapped() and app.edge.winfo_ismapped() and app.screen.winfo_width() < w0
    app._toggle_side()
    settle(app, rounds=3)
    assert not app.side.winfo_ismapped() and app.screen.winfo_width() == w0
    app._toggle_side()
    app._fullscreen()                                              # «кіно»: ні панелі, ні кнопки
    settle(app, rounds=2)
    assert not app.side.winfo_ismapped() and not app.edge.winfo_ismapped()
    app._fullscreen(False)
    settle(app, rounds=2)
    assert app.side.winfo_ismapped() and app.edge.winfo_ismapped(), "після кіно F11 лишається, як було"
    app._window_fullscreen(False)
    settle(app, rounds=3)
    assert app.side.winfo_ismapped() and not app.edge.winfo_ismapped()


def test_f11_stretches_the_whole_app_while_f_is_cinema(app):
    """F11 - просто вікно на весь екран з усім інтерфейсом; F (кіно) лишає лише картинку; Esc виходить; F11 із кіно виходить повністю"""
    settle(app, rounds=2)
    app._window_fullscreen()
    settle(app, rounds=2)
    assert app._wfs and not app._fs
    assert app.pstack.enabled and app.edge.winfo_ismapped(), "в режимі F11 інтерфейс на місці, панель - за кнопкою справа"
    app._on_escape()
    settle(app, rounds=2)
    assert not app._wfs and not app._fs
    app._fullscreen()
    settle(app, rounds=2)
    assert app._fs and not app.side.winfo_ismapped() and not app.pstack.enabled, "кіно: лише картинка"
    app._window_fullscreen()                                      # F11 з кіно - вийти зовсім
    settle(app, rounds=2)
    assert not app._fs and not app._wfs and app.side.winfo_ismapped() and app.pstack.enabled
    app._window_fullscreen()                                      # F11 -> кіно -> Esc: лишається вікно на весь екран
    app._fullscreen()
    app._on_escape()
    settle(app, rounds=2)
    assert app._wfs and not app._fs
    app._on_escape()
    settle(app, rounds=2)
    assert not app._wfs


def _ctrl_physical(app, widget, keysym, keycode):
    """Ctrl + фізична клавіша з кириличним keysym, як її шле Windows у кириличній розкладці. Tk не генерує подію для keysym, якого нема в активній розкладці
    (ПК з англійською розкладкою: TclError «no keycode for keysym "Cyrillic_em"») - тоді та сама подія йде в обробник напряму, логіку це перевіряє так само"""
    import tkinter as tk
    import types
    try:
        widget.event_generate("<Key>", keysym=keysym, keycode=keycode, state=0x4)
    except tk.TclError:
        app._ctrl_key(types.SimpleNamespace(keysym=keysym, keycode=keycode, state=0x4, widget=widget))
    app.update()


def test_copy_paste_work_in_the_size_field_in_any_keyboard_layout(app, monkeypatch):
    """Ctrl+C/V/X/A у полі розміру й списках: за фізичною клавішею (кирилична розкладка шле інший keysym); «1920x1080» з буфера лягає в обидва поля"""
    from benvid import gui
    monkeypatch.setattr(gui.sys, "platform", "win32")               # на Windows keycode клавіші = її віртуальний код A-Z
    e = app.ent_w
    app.w_var.set("")
    app.clipboard_clear()
    app.clipboard_append("1280")
    e.focus_force()
    app.update()
    _ctrl_physical(app, e, "Cyrillic_em", 86)                       # Ctrl + фізична «V» у кириличній розкладці
    assert app.w_var.get() == "1280", "Ctrl+V у кириличній розкладці"
    e.selection_range(0, "end")
    app.clipboard_clear()
    _ctrl_physical(app, e, "Cyrillic_es", 67)                       # Ctrl + фізична «C»
    assert app.clipboard_get() == "1280", "Ctrl+C у кириличній розкладці"
    _ctrl_physical(app, e, "Cyrillic_che", 88)                      # Ctrl + «X»: вирізати
    assert app.w_var.get() == "", "Ctrl+X у кириличній розкладці"
    app.clipboard_clear()
    app.clipboard_append("1920 x 1080")
    e.event_generate("<<Paste>>")
    app.update()
    assert (app.w_var.get(), app.h_var.get()) == ("1920", "1080"), "«ШxВ» розкладається по полях"
    app.w_var.set("")
    app.h_var.set("")
    app.clipboard_clear()
    app.clipboard_append("абв")
    e.event_generate("<<Paste>>")
    app.update()
    assert app.h_var.get() == "", "не число в поле розміру не потрапляє"
    app.w_var.set("640")                                          # вставка ЗАМІНЮЄ вміст поля, а не дописується
    app.clipboard_clear()
    app.clipboard_append("1280")
    e.event_generate("<<Paste>>")
    app.update()
    assert app.w_var.get() == "1280"
    app.fps_cb.set("24")
    app.clipboard_clear()
    app.clipboard_append("29.97")
    app.fps_cb.event_generate("<<Paste>>")
    app.update()
    assert app.fps_cb.get() == "29.97", "те саме в полі «кадри»"
    app.w_var.set("")


def test_player_hotkeys_do_nothing_on_the_footer_screen(app, monkeypatch):
    """на екрані футера (гайд, про програму) пробіл, стрілки, гучність, «[ ]», M/L і Home не керують плеєром; F, I - як і раніше"""
    from benvid import gui
    got = []
    for name in ("_toggle", "_stop", "_key_mute", "_key_loop"):
        monkeypatch.setattr(app, name, lambda *a, n=name: got.append(n))
    monkeypatch.setattr(app, "_step", lambda *a: got.append("step"))
    monkeypatch.setattr(app, "_jump", lambda *a: got.append("jump"))
    monkeypatch.setattr(app, "_vol_step", lambda *a, **k: got.append("vol"))
    monkeypatch.setattr(app, "_set_mark_here", lambda *a: got.append("mark"))
    keys = ["space", "Left", "Right", "Up", "Down", "bracketleft", "bracketright", "Home", "m", "l", "f", "plus", "minus", "0"]
    app.focus_force()
    settle(app, rounds=2)
    monkeypatch.setattr(app.pstack, "tcur", 1)
    assert app._on_footer()
    for k in keys:
        app.event_generate(f"<Key-{k}>", when="now")
        app.update()
    assert got == [], f"на футері плеєр не керується: {got}"
    monkeypatch.setattr(app.pstack, "tcur", 0)
    monkeypatch.setattr(app.pstack, "cur", 0)
    assert not app._on_footer()
    for k in keys:
        app.event_generate(f"<Key-{k}>", when="now")
        app.update()
    assert {"_toggle", "step", "vol", "mark", "_stop", "_key_mute", "_key_loop"} <= set(got), got


def test_fps_field_selects_only_the_value_and_selection_follows_the_theme(app):
    """«Кадри: 25 к/с»: клік виділяє лише значення після двокрапки, мишкою текст не тягнеться, при виході з поля виділення зникає; виділення в полях розміру - в кольорах теми; висока кнопка F11 не інвертується"""
    from benvid import gui
    settle(app, rounds=2)
    app.fps_cb.set("Кадри: 25 к/с")
    gui._select_value(app.fps_cb)                                  # те, що робить клік у текст поля (_fps_click)
    t = app.fps_cb.get()
    assert app.fps_cb.selection_present() and t[app.fps_cb.index("sel.first"):] == "25 к/с"
    app.fps_cb.event_generate("<FocusOut>")                        # фокус пішов з поля
    app.update()
    assert not app.fps_cb.selection_present(), "виділення зникає при виході з поля"
    assert str(app.ent_w.cget("selectbackground")).lower() == gui.FG.lower()
    assert app.edge.FLASH is False


def test_size_fields_keep_theme_colours_when_disabled(app):
    """у замку (готовий .ikr) поля «свій розмір» вимкнені: тло лишається кольором палітри, а не світлим системним, і при зміні палітри перефарбовується"""
    from benvid import gui
    settle(app, rounds=2)
    app.ent_w.config(state="disabled")
    assert str(app.ent_w.cget("disabledbackground")).lower() == gui.BG.lower()
    app.pal_var.set("Кармін")
    app._set_palette()
    app.update()
    assert str(app.ent_w.cget("disabledbackground")).lower() == gui.BG.lower()
    app.ent_w.config(state="normal")


# ---------------------------------------------------------------- пакет П-1: «Порада»
def test_tip_on_empty_screen_and_footer_hidden_while_exporting(app):
    from benvid import guide
    settle(app, rounds=2)
    saved, app._last_mask = app._last_mask, None            # порожній екран: файл не відкрито
    app._empty_shown = False
    app._redraw()
    app._last_mask = saved
    assert 0 <= app._tip_n < len(guide.TIPS["uk"])
    texts = [app.screen.itemcget(i, "text") for i in app.screen.find_all() if app.screen.type(i) == "text"]
    assert guide.tip_text(app._tip_n) in texts, "порада на порожньому екрані"
    app._fill_tip()
    assert app.tip_lbl.cget("text") == guide.tip_text(app._tip_n) and app.tip_lbl.winfo_manager()
    app.exporting = True
    try:
        app._fill_tip()
        assert not app.tip_lbl.winfo_manager(), "під час експорту поради нема"
    finally:
        app.exporting = False
    app._fill_tip()
    assert app.tip_lbl.winfo_manager()
