"""режим замка у вікні: «Порахувати все» зберігає .ikr і відкриває його (нічого не рахується, регулятори якості/розміру вимкнені, експорт і
кадр у фото працюють), «Звичайний режим» повертає оригінал, налаштування файлу й ту саму мить.

Потрібен екран (Linux: xvfb-run -a python -m pytest tests/test_lock_mode.py)"""
from __future__ import annotations

import os

import numpy as np
import pytest
from conftest import NO_SCREEN, pump, settle

from benvid import bake
from benvid.video import find_ffmpeg

pytestmark = [pytest.mark.skipif(NO_SCREEN, reason="немає екрана (запускай під xvfb-run)"),
              pytest.mark.skipif(find_ffmpeg() is None, reason="потрібен ffmpeg")]


def _clip(path, n=36, w=320, h=180, fps=24):
    import cv2
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        f = np.dstack([(xx * 255 // w + i * 2) % 256, (yy * 255 // h), (xx + yy + i * 3) % 256]).astype(np.uint8)
        cv2.circle(f, (30 + (i * 6) % (w - 60), h // 2), 40, (240, 240, 240), -1)
        vw.write(f)
    vw.release()


def test_freeze_lock_and_unfreeze(app, tmp_path, monkeypatch):
    from tkinter import filedialog
    src, ikr = str(tmp_path / "v.mp4"), str(tmp_path / "v.ikr")
    _clip(src)
    app.size_var.set("Як у джерела")
    app._apply_method(3)
    app.player.set_params(app.params)
    app._open(src)
    assert pump(app, lambda: app.ready and app._last_mask is not None, 60)
    assert not app.locked and app.btn_lock.text == "Порахувати все" and app.m_slider.enabled

    # --- порахувати все: зберегти .ikr і відкрити його в замку
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **kw: ikr)
    app._lock_toggle()
    assert app.exporting
    assert pump(app, lambda: not app.exporting and app.locked, 180), "заморожений файл не відкрився"
    settle(app, rounds=3)
    assert os.path.isfile(ikr) and bake.is_baked(ikr)
    assert app.player.baked and app.player.path == ikr and app.btn_lock.text == "Звичайний режим"
    assert pump(app, lambda: app.player.cached >= app.player.n, 60)
    assert app.player._ex is None                                   # пул не створювався: нічого не рахується
    # регулятори вимкнені й показують те, з чим порахований файл; зміни відхиляються
    assert not app.m_slider.enabled and not app.btn_ink.enabled and not app.btn_ok.enabled
    assert all(not b.enabled for b in app.size_btns.values())
    assert app.m_slider.v == 3 and "ЗАМОК" in app.q_title.cget("text")
    assert "замок" in app.size_lbl.cget("text")
    app._apply_method(1)
    app._set_glyph(True)
    app._set_size("144p")
    assert app.params.method == 3 and not app.params.glyph and app.size_var.get() == "Як у джерела"
    assert app.player.size == (320, 180)

    # --- кадр у фото з замка
    png = str(tmp_path / "f.png")
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **kw: png)
    app.player.seek(7)
    assert pump(app, lambda: int(app.player.position()) == 7 and app._last_mask is not None, 30)
    app._save_frame()
    assert os.path.isfile(png)
    import cv2
    im = cv2.imdecode(np.fromfile(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert im.shape == (180, 320) and set(np.unique(im)) <= {0, 255}

    # --- експорт діапазону з замка в інший формат
    out = str(tmp_path / "o.mp4")
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **kw: out)
    app.fmt_var.set("MP4 (H.264)")
    app.q_var.set("Мала")
    app._export()
    assert pump(app, lambda: not app.exporting, 180)
    assert os.path.isfile(out) and os.path.getsize(out) > 500

    # --- звичайний режим: оригінал, налаштування файлу, та сама мить
    app.params.method = 5                                           # «чужі» налаштування в вікні: повертаються ті, з якими порахований файл
    app.player.seek(10)
    assert pump(app, lambda: int(app.player.position()) == 10, 30)
    app._lock_toggle()
    assert pump(app, lambda: not app.locked and app.ready, 60)
    settle(app, rounds=3)
    assert app.player.path == src and not app.player.baked
    assert app.params.method == 3 and app.m_slider.v == 3 and app.m_slider.enabled and app.btn_ink.enabled
    assert all(b.enabled for b in app.size_btns.values()) and app.btn_lock.text == "Порахувати все"
    assert pump(app, lambda: int(app.player.position()) == 10, 30)
    # пораховані кадри не викинуто: розмір той самий, а буфер береться з файлу .ikr (не рахується з нуля)
    assert app.size_var.get() == "Як у джерела" and app.player.size == (320, 180)
    assert app.player._seed is not None and app.player._seed[1] == ikr
    assert pump(app, lambda: app.player.cached >= app.player.n, 60)
    rd = bake.open_baked(ikr)
    _, snap = app.player.snapshot()
    assert all(np.array_equal(snap[i], rd.packed(i)) for i in (0, 10, 20, 35))


def test_unfreeze_asks_for_original_when_it_moved(app, tmp_path, monkeypatch):
    from tkinter import filedialog
    src, ikr = str(tmp_path / "w.mp4"), str(tmp_path / "w.ikr")
    _clip(src, n=12)
    app.size_var.set("Як у джерела")
    app._open(src)
    assert pump(app, lambda: app.ready and app._last_mask is not None, 60)
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **kw: ikr)
    app._lock_toggle()
    assert pump(app, lambda: not app.exporting and app.locked, 180)
    moved = str(tmp_path / "moved.mp4")
    os.replace(src, moved)                                          # оригінал перенесли
    asked, said = [], []
    from tkinter import messagebox
    monkeypatch.setattr(messagebox, "askokcancel", lambda title, msg, **kw: said.append(msg) or True)     # пояснення: у файлі лише результат
    monkeypatch.setattr(filedialog, "askopenfilename", lambda **kw: asked.append(kw["title"]) or moved)
    app._lock_toggle()
    assert pump(app, lambda: not app.locked and app.ready and app.player.path == moved, 60)
    assert asked and "w.mp4" in asked[0]
    assert said and "w.mp4" in said[0] and ".ikr" in said[0]
