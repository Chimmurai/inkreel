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


def test_window_palette_language_and_level_controls(qapp, win):
    from benvid.qtui.theme import THEME
    win._step_palette(1)
    assert win.pal_name == palette.NAMES[1] and THEME.key == win._pal_key()
    win._toggle_inv()
    assert palette.INV in THEME.key or THEME.key.endswith(palette.INV)
    win._toggle_inv()
    win._set_level(4)
    assert win.params.method == 4 and win.qbar.level == 4
    win._set_glyph(True)
    assert win.params.glyph and win.btn_sym.isOn()
    win._set_lang("en")
    assert i18n.lang() == "en" and win.btn_open.text() == "Open"
    win._set_lang("uk")
    assert win.btn_open.text() == "Відкрити"


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
    assert spin(qapp, lambda: not win.export.running and out.exists(), timeout=60), win.prog_lbl.text()
    assert "Готово" in win.prog_lbl.text() and win.btn_export.isEnabled() and not win.btn_stop.isEnabled()
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
    assert win.export.running and not win.btn_export.isEnabled() and win.btn_stop.isEnabled()
    assert spin(qapp, lambda: not win.export.running, timeout=120), win.prog_lbl.text()
    assert out2.exists() and out2.stat().st_size > 1000 and "Готово" in win.prog_lbl.text()
    assert seen and max(seen) > 0.5 and win.prog.f == 1.0

    win._export()                                    # скасування: стоп під час роботи
    assert spin(qapp, lambda: win.export.running, timeout=5)
    win.export.stop()
    assert spin(qapp, lambda: not win.export.running, timeout=60)
    assert win.btn_export.isEnabled()


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
