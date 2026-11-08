"""Шум і штрихи «Ультри» їдуть разом із вмістом (benvid/motion.py, ядро малювання в ink5.py): на нерухомій картинці нічого не змінилось
(побітово), на панорамі й рухомому об'єкті візерунок іде за зображенням, склейка скидає зсув."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid import core as C, gpu, ink5, motion  # noqa: E402

W, H = 640, 360
needs_nb = pytest.mark.skipif(not ink5.HAVE_NUMBA or C._nb is None, reason="numba немає: «Ультра» без нього не працює")


def _scene(w=1100, h=H):
    """текстурована картинка завширшки більше за кадр (для панорами): хвилі, дрібний візерунок, кілька яскравих плям"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(4)
    g = 110 + 60 * np.sin(xx / 29.0 + yy / 41.0) + 40 * np.sin(xx / 7.3) * np.cos(yy / 9.1) + 25 * np.sin(yy / 5.0 + xx / 13.0)
    g = cv2.GaussianBlur(g.astype(np.float32) + rng.normal(0, 3, (h, w)).astype(np.float32), (0, 0), 0.8)
    for cx, cy, r in ((200, 120, 40), (520, 220, 55), (800, 90, 35)):
        cv2.circle(g, (cx, cy), r, 235.0, -1)
    g = np.clip(g, 0, 255).astype(np.uint8)
    return np.dstack([g, np.roll(g, 3, 1), 255 - g])


def _pan(big, n, vx):
    return [cv2.warpAffine(big, np.float32([[1, 0, -vx * t], [0, 1, 0]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            for t in range(n)]


def _run(frames, **over):
    st = C.InkState()
    p = C.Params(method=5)
    ms = []
    for f in frames:
        ms.append(ink5.ultra(f, W, H, p, st, None, **over).copy())
    return ms, st


def _shift_mismatch(ms, vx):
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    e = []
    for t in range(2, len(ms)):
        mw = cv2.remap(ms[t - 1], X + np.float32(round(vx)), Y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        e.append(float((mw != ms[t])[20:-20, int(abs(vx)) * 2 + 20:-20].mean()))
    return float(np.mean(e))


@needs_nb
@pytest.mark.parametrize("cell", [1, 2, 3])
def test_tile_at_is_the_same_threshold_as_the_tiled_frame(cell):
    """поріг у довільній точці (також поза кадром) - те саме поле, що й масив на весь кадр, тож нульовий зсув нічого не міняє"""
    w, h = 300, 200
    full = C._tile(C._blue(), w, h, cell)
    rng = np.random.default_rng(1)
    ys, xs = rng.integers(0, h, 800), rng.integers(0, w, 800)
    got = np.array([ink5._tile_at(C._blue(), cell, int(y), int(x)) for y, x in zip(ys, xs)], np.float32)
    assert np.array_equal(got, full[ys, xs])
    far = [ink5._tile_at(C._blue(), cell, -3 * 128 * cell - 5, 7 * 128 * cell + 2), ink5._tile_at(C._blue(), cell, -1, -1)]
    assert all(0.0 <= v <= 1.0 for v in far)


@needs_nb
def test_zero_shift_grid_draws_exactly_like_the_static_path():
    """шлях зі зсувом при нульових зсувах дає побітово ту саму маску, що шлях із готовими масивами порогів"""
    w, h = 320, 200
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.clip(0.5 + 0.35 * np.sin(xx / 37.0) * np.cos(yy / 29.0) + rng.normal(0, 0.05, (h, w)), -0.02, 1.02).astype(np.float32)
    sh, sw = 60, 100
    Fk = np.stack([np.cos(xx[:sh, :sw] / 20), np.sin(xx[:sh, :sw] / 20), 6 + 4 * np.sin(yy[:sh, :sw] / 9),
                   0.7 * np.ones((sh, sw)), 0.2 * np.ones((sh, sw))], -1).astype(np.float32)
    T = ink5._TILE
    noise, sd = ink5._ring_tile(T, 3.6, 0.3, 5), ink5._sigma_tab(T, 3.6, 0.3, 5)
    Tg, Td, _ = ink5._static_fields(w, h, 1, 1)
    prev = (rng.random((h, w)) > 0.5).astype(np.uint8)
    a = ink5._draw_cpu(t, Fk, noise, T, sd, Tg, Td, 4, 3, 0.7, 1.0, prev, 0.05, [None])
    b = ink5._draw_cpu(t, Fk, noise, T, sd, Tg, Td, 4, 3, 0.7, 1.0, prev, 0.05, [None], np.zeros((sh, sw, 2), np.float32), 1, 1)
    assert np.array_equal(a, b)
    c = ink5._draw_cpu(t, Fk, noise, T, sd, Tg, Td, 4, 3, 0.7, 1.0, prev, 0.05, [None], np.full((sh, sw, 2), 5.0, np.float32), 1, 1)
    assert float((a != c).mean()) > 0.05, "зсув на 5 px справді міняє візерунок"


@needs_nb
def test_still_picture_is_bit_identical_with_and_without_motion_anchoring():
    f = _scene()[:, :W]
    frames = [f] * 5
    a, _ = _run(frames, ta=0)
    b, st = _run(frames, ta=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert st.ink5.D is None or float(np.abs(st.ink5.D).max()) < 0.5


@needs_nb
def test_noisy_still_picture_stays_put():
    """кадри з різним шумом датчика, але без руху: візерунок не повзе (зсув не набігає)"""
    base = _scene()[:, :W].astype(np.float32)
    rng = np.random.default_rng(9)
    frames = [np.clip(base + rng.normal(0, 2.0, base.shape), 0, 255).astype(np.uint8) for _ in range(8)]
    _, st = _run(frames, ta=1)
    D = st.ink5.D
    assert D is None or float(np.abs(D).max()) < 1.0


@needs_nb
@pytest.mark.parametrize("vx", [3.0, 12.0])
def test_panning_picture_carries_its_pattern(vx):
    """камера їде: штрихи й крапки їдуть із зображенням. Раніше візерунок стояв на екрані, і тон плив повз нього"""
    big = cv2.copyMakeBorder(_scene(), 0, 0, 0, int(vx * 10 + 40), cv2.BORDER_REPLICATE)
    frames = _pan(big, 10, vx)
    still, _ = _run(frames, ta=0)
    moved, st = _run(frames, ta=1)
    e0, e1 = _shift_mismatch(still, vx), _shift_mismatch(moved, vx)
    assert e1 < 0.7 * e0, (e0, e1)
    assert st.ink5.D is not None and float(np.abs(st.ink5.D[..., 0]).max()) > 2.0 * vx


@needs_nb
def test_unrelated_frame_is_a_cut_and_resets_the_shift():
    big = cv2.copyMakeBorder(_scene(), 0, 0, 0, 100, cv2.BORDER_REPLICATE)
    frames = _pan(big, 6, 6.0)
    other = np.ascontiguousarray(255 - _scene()[:, ::-1][:, :W])                     # зовсім інша картинка
    st = C.InkState()
    p = C.Params(method=5)
    for f in frames:
        ink5.ultra(f, W, H, p, st, None)
    assert st.ink5.D is not None
    ink5.ultra(other, W, H, p, st, None)
    assert st.ink5.D is None, "монтажна склейка: зсув скинуто"


@needs_nb
def test_static_overlay_keeps_its_own_pattern_over_a_moving_picture():
    """нерухомий напис (інтерфейс) поверх рухомої картинки: де кадри збігаються без зсуву, візерунок лишається на місці"""
    big = cv2.copyMakeBorder(_scene(), 0, 0, 0, 200, cv2.BORDER_REPLICATE)
    frames = _pan(big, 8, 8.0)
    for f in frames:
        cv2.rectangle(f, (40, 30), (220, 120), (20, 20, 20), -1)
        cv2.putText(f, "HUD 100", (52, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    still, _ = _run(frames, ta=0)
    moved, _ = _run(frames, ta=1)
    sl = (slice(45, 115), slice(50, 215))
    d0 = np.mean([(still[t][sl] != still[t - 1][sl]).mean() for t in range(3, 8)])
    d1 = np.mean([(moved[t][sl] != moved[t - 1][sl]).mean() for t in range(3, 8)])
    assert d1 <= d0 + 0.02, (d0, d1)


def test_motion_step_of_identical_frames_has_no_flow():
    f = cv2.resize(_scene()[:, :W, 0], (427, 240), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    b, bh, expl = motion.analyze(f, f.copy(), cv2.GaussianBlur(f, (0, 0), 1.0), 0.33)
    assert not b.any() and expl
    assert motion.advance(None, b, 1.5, 0.33) is None


def test_motion_follows_a_shift_of_the_tone_grid():
    big = cv2.resize(_scene()[:, :, 0], (1100 * 2 // 3, 240), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    a, bb = big[:, 20:20 + 427], big[:, 24:24 + 427]                               # кадр bb - це a, зсунуте на 4 px сітки вліво
    b, bh, expl = motion.analyze(bb, a, cv2.GaussianBlur(bb, (0, 0), 1.0), 0.33)
    # піксель поточного кадру p був у попередньому в p + 4 (бо вміст поїхав вліво): потік по x ~ +4
    assert 3.0 < float(np.median(b[..., 0][b[..., 0] != 0])) < 5.0
    D = motion.advance(None, b, 1.5, 0.33)
    assert D is not None and motion.active(D)


@pytest.mark.skipif(not (gpu.HAVE_CL and bool(gpu.list_devices(allow_cpu=True))), reason="pyopencl або OpenCL-пристрій відсутні")
@needs_nb
def test_gpu_draw_matches_cpu_with_shifted_pattern(monkeypatch, tmp_path):
    monkeypatch.setenv("INKREEL_GPU", "cpu")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    gpu._dev, gpu._state["use"] = None, None
    gpu.set_allowed(True)
    try:
        g = gpu.device()
        assert g is not None
        w, h = 400, 240
        rng = np.random.default_rng(3)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = np.clip(0.5 + 0.35 * np.sin(xx / 37.0) * np.cos(yy / 29.0) + rng.normal(0, 0.05, (h, w)), -0.02, 1.02).astype(np.float32)
        sh, sw = h // 3, w // 3
        Fk = np.stack([np.cos(xx[:sh, :sw] / 20), np.sin(xx[:sh, :sw] / 20), 6 + 4 * np.sin(yy[:sh, :sw] / 9),
                       0.7 * np.ones((sh, sw)), 0.2 * np.ones((sh, sw))], -1).astype(np.float32)
        Dk = np.stack([30 * np.sin(yy[:sh, :sw] / 17) - 3.3, -20 * np.cos(xx[:sh, :sw] / 23) + 1.7], -1).astype(np.float32)
        T = ink5._TILE
        noise, sd = ink5._ring_tile(T, 3.6, 0.3, 5), ink5._sigma_tab(T, 3.6, 0.3, 5)
        for cg, cb in ((1, 1), (2, 1)):
            Tg, Td, _ = ink5._static_fields(w, h, cg, cb)
            prev = (rng.random((h, w)) > 0.5).astype(np.uint8)
            args = (t, Fk, noise, T, sd, Tg, Td, 4, 3, 0.7, 1.0, prev, 0.05)
            for D in (None, Dk):
                a = ink5._draw_cpu(*args, [None], D, cg, cb)
                with g.lock:
                    b = ink5._draw_gpu(g, *args, [None], D, cg, cb)
                assert float((np.asarray(a) == np.asarray(b)).mean()) > 0.995
    finally:
        gpu._dev, gpu._state["use"] = None, None
