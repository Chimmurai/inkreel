import os
import sys
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid import palette, view  # noqa: E402

SENTINEL = 7


def _mask(w, h, seed=0):
    return (np.random.default_rng(seed).random((h, w)) > 0.5).astype(np.uint8)


def _lut(name):
    ink, paper = palette.colors(name)
    t = np.arange(256, dtype=np.float32)[:, None] / 255.0
    return (np.array(ink, np.float32) * (1 - t) + np.array(paper, np.float32) * t).astype(np.uint8).reshape(256, 1, 3)


def _canvas(mask, lut, cw, ch, zoom, vc=view.CENTER):
    """те, що побачить користувач: полотно cw×ch із зображенням, покладеним як у GUI"""
    comp = view.compose(mask, lut, cw, ch, zoom, vc)
    shape = (ch, cw) if lut is None else (ch, cw, 3)
    cv = np.full(shape, SENTINEL, np.uint8)
    if comp is None:
        return cv
    img, k, x, y, crop = comp
    fin = view.finish(img, k, crop)
    cv[y:y + fin.shape[0], x:x + fin.shape[1]] = fin
    return cv


def _reference(mask, lut, cw, ch, zoom, vc=view.CENTER):
    """той самий результат, порахований «в лоб»: усе зображення цілком, потім вікно"""
    h, w = mask.shape
    s, W, H, ox, oy = view.geometry(cw, ch, w, h, zoom, vc)
    g8 = (mask * 255).astype(np.uint8)
    full = np.repeat(np.repeat(g8, int(s), 0), int(s), 1) if s >= 1 else cv2.resize(g8, (W, H), interpolation=cv2.INTER_AREA)
    if lut is not None:
        full = lut.reshape(256, 3)[full]
    shape = (ch, cw) if lut is None else (ch, cw, 3)
    cv = np.full(shape, SENTINEL, np.uint8)
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(W, ox + cw), min(H, oy + ch)
    cv[max(0, -oy):max(0, -oy) + (y1 - y0), max(0, -ox):max(0, -ox) + (x1 - x0)] = full[y0:y1, x0:x1]
    return cv


def test_fit_scale_rules():
    assert view.fit_scale(940, 640, 320, 240) == 2.0                 # менше за вікно - ЦІЛЕ збільшення (1.47 -> 1), тут 2.67 -> 2
    assert view.fit_scale(940, 640, 640, 360) == 1.0                 # 1.47 -> 1: не розмиваємо дробовим масштабом
    assert view.fit_scale(940, 640, 1880, 1280) == 0.5               # більше за вікно - плавне зменшення
    assert view.fit_scale(940, 640, 1920, 1080) == pytest.approx(940 / 1920)
    assert view.fit_scale(100, 100, 100, 100) == 1.0


@pytest.mark.parametrize("pal", [None, "Апельсин", "Класика~"])
@pytest.mark.parametrize("zoom", [None, 1 / 4, 1 / 2, 1, 2, 3, 5, 8, 16, 64])
def test_compose_matches_brute_force_for_many_views(zoom, pal):
    lut = _lut(pal) if pal and pal != "Класика" else None
    rng = np.random.default_rng(3)
    for (w, h) in ((97, 61), (300, 200), (5, 7)):
        if zoom and zoom >= 1 and w * h * zoom * zoom > 3e7:               # еталон будує все збільшене зображення - не для гігантів
            continue
        m = _mask(w, h, seed=w)
        for (cw, ch) in ((200, 120), (64, 64), (333, 101)):
            for vc in (view.CENTER, (0.0, 0.0), (1.0, 1.0), tuple(rng.random(2))):
                a, b = _canvas(m, lut, cw, ch, zoom, vc), _reference(m, lut, cw, ch, zoom, vc)
                assert a.shape == b.shape and (a == b).all(), (w, h, cw, ch, zoom, vc)


def test_enlargement_never_blurs():
    """будь-який цілий масштаб: на екрані лише кольори палітри - ні проміжних відтінків, ні «апскейльнутого» розмиття"""
    ink, paper = palette.colors("Бурштин")
    m = _mask(60, 40, seed=5)
    for z in (1, 2, 3, 4, 7, 16):
        c = _canvas(m, _lut("Бурштин"), 300, 200, z, (0.4, 0.6))
        cols = {tuple(px) for px in c.reshape(-1, 3)}
        assert cols <= {ink, paper, (SENTINEL,) * 3}, z
    g = _canvas(m, None, 300, 200, 4, (0.5, 0.5))                        # чорно-біле: лише 0 і 255
    assert set(np.unique(g)) <= {0, 255, SENTINEL}
    # кожен піксель результату = рівний квадрат z×z
    z = 5
    c = _canvas(m, None, 40 * z, 30 * z, z, (0.0, 0.0))
    blocks = c.reshape(30, z, 40, z).transpose(0, 2, 1, 3).reshape(30, 40, z * z)
    assert (blocks.min(-1) == blocks.max(-1)).all()
    assert (blocks[..., 0] == m[:30, :40] * 255).all()


def test_zoom_out_averages_instead_of_dropping_pixels():
    m = (np.indices((64, 64)).sum(0) % 2).astype(np.uint8)               # шахівниця 1×1: при 1/2 має дати рівний сірий
    c = _canvas(m, None, 32, 32, 1 / 2)
    assert abs(int(c.mean()) - 127) <= 1 and c.min() >= 126 and c.max() <= 129


def test_wheel_levels_and_limits():
    fit = 0.73
    ok, n = view.step(fit, fit, +1, False)
    assert ok and n == 1                                                 # з «вписати» - до 1:1
    chain = [1]
    while True:
        ok, n = view.step(chain[-1], fit, +1, True)
        if not ok:
            break
        chain.append(n)
    assert chain == [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]
    assert all(float(v).is_integer() for v in chain)                     # лише цілі - чіткі пікселі
    # униз до «вписати» і далі нікуди
    cur, seq = 64, []
    while True:
        ok, n = view.step(cur, fit, -1, True)
        assert ok
        if n is None:
            break
        seq.append(n)
        cur = n
    assert seq[-1] == 1 and seq == sorted(seq, reverse=True)
    assert view.step(fit, fit, -1, False) == (False, None)               # у режимі «вписати» зменшувати нікуди
    # зображення менше за вікно: «вписати» = 3, колесо вгору -> 4, униз з 4 -> назад у «вписати»
    assert view.step(3.0, 3.0, +1, False) == (True, 4)
    assert view.step(4, 3.0, -1, True) == (True, None)
    # дрібний масштаб між «вписати» і 1:1
    assert view.step(1, 0.2, -1, True) == (True, 0.5) or view.step(1, 0.2, -1, True) == (True, 0.25)


def test_point_under_cursor_stays_put():
    cw, ch, w, h = 600, 400, 1920, 1080
    zoom, vc = None, view.CENTER
    mx, my = 431, 118
    for new in (1, 2, 3, 4, 8, 16):
        s0, W0, H0, ox0, oy0 = view.geometry(cw, ch, w, h, zoom, vc)
        src = ((ox0 + mx) / s0, (oy0 + my) / s0)                          # що було під курсором (пікселі джерела)
        vc = view.anchored(cw, ch, w, h, s0, ox0, oy0, new, mx, my)
        zoom = new
        s1, W1, H1, ox1, oy1 = view.geometry(cw, ch, w, h, zoom, vc)
        got = ((ox1 + mx) / s1, (oy1 + my) / s1)
        assert abs(got[0] - src[0]) <= 1.0 and abs(got[1] - src[1]) <= 1.0, (new, src, got)


def test_pan_is_clamped_and_small_images_stay_centred():
    cw, ch = 300, 200
    vc = view.pan(cw, ch, 400, 300, 2, (0.5, 0.5), -10_000, -10_000)      # тягнемо картинку далеко вліво-вгору -> вікно упирається в правий нижній край
    s, W, H, ox, oy = view.geometry(cw, ch, 400, 300, 2, vc)
    assert (ox, oy) == (W - cw, H - ch)
    vc = view.pan(cw, ch, 400, 300, 2, vc, 10_000, 10_000)                # а тепер далеко вправо-вниз -> лівий верхній край
    s, W, H, ox, oy = view.geometry(cw, ch, 400, 300, 2, vc)
    assert (ox, oy) == (0, 0)
    s, W, H, ox, oy = view.geometry(cw, ch, 50, 40, 2, view.pan(cw, ch, 50, 40, 2, (0.5, 0.5), 77, 55))
    assert (W, H) == (100, 80) and ox == -100 and oy == -60              # менше за вікно - по центру, не їздить
    vc = view.pan(cw, ch, 600, 100, 2, (0.5, 0.5), 5, 5)                  # по одній осі зображення більше, по іншій менше
    assert vc[1] == 0.5


def test_zoom_cost_does_not_depend_on_zoom_level():
    m = _mask(3840, 2160, seed=1)
    lut = _lut("Синька")
    for z in (1, 4, 64):
        t = time.perf_counter()
        for _ in range(10):
            comp = view.compose(m, lut, 1280, 720, z, (0.3, 0.6))
            view.finish(*[comp[0], comp[1], comp[4]])
        assert (time.perf_counter() - t) / 10 < 0.25, z


def test_degenerate_windows_are_safe():
    m = _mask(10, 10)
    assert view.compose(m, None, 0, 0, None) is None
    assert _canvas(m, None, 1, 1, 64, (0.0, 0.0)).shape == (1, 1)
    assert view.label(4) == "400%" and view.label(0.5) == "50%" and view.label(1 / 16) == "6.25%"
