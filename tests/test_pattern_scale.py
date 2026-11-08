"""масштаб візерунка понад 720p (core.PAT): до 720p без змін, вище - тонший візерунок і зерно в один піксель"""
import cv2
import numpy as np
import pytest

from benvid import core as C

OLD = dict(pow=1.0, ink=1.0, grain=0)


def _scene(w, h):
    rng = np.random.default_rng(7)
    g = np.linspace(40, 215, w, dtype=np.float32)[None, :].repeat(h, 0)
    g += 12 * np.sin(np.arange(h, dtype=np.float32)[:, None] / 9.0)
    g += rng.normal(0, 2, (h, w)).astype(np.float32)
    g = np.clip(g, 0, 255).astype(np.uint8)
    cv2.circle(g, (w // 3, h // 2), h // 4, 230, -1)
    cv2.putText(g, "InkReel 123", (w // 2, h // 3), cv2.FONT_HERSHEY_SIMPLEX, h / 360.0, 20, 2)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _with(pat, fn):
    keep = dict(C.PAT)
    C.PAT.update(pat)
    try:
        return fn()
    finally:
        C.PAT.clear()
        C.PAT.update(keep)


def test_pat_k_is_k_up_to_720p_and_sublinear_above():
    for k in (0.2, 0.5, 1.0):
        assert C.pat_k(k) == k
    assert C.pat_k(1.5) == pytest.approx(1.5 ** 0.5)
    assert C.pat_k(2.0) < 2.0 and C.pat_k(2.0) > C.pat_k(1.5)
    assert C.pat_k(2.0, 1.0) == 2.0


@pytest.mark.parametrize("method", [3, 4, 5])
def test_up_to_720p_output_does_not_depend_on_pat(method):
    """до 720p включно налаштування PAT нічого не міняють (побітово)"""
    w, h = 320, 180 * 2                     # 360p: k = 0.5
    img = _scene(w, h)
    p = C.Params(method=method)
    a = C.binarize(img, w, h, p, {"state": C.InkState()} if method == 5 else None)
    b = _with(OLD, lambda: C.binarize(img, w, h, p, {"state": C.InkState()} if method == 5 else None))
    assert np.array_equal(a, b)


def test_halftone_grain_is_single_pixel_above_720p():
    """рівень 3 на рівному середньому тоні: на 1080p раніше блоки 2x2 були однорідні, тепер зерно - окремі пікселі"""
    w, h = 400, 1080
    t = np.full((h, w), 0.5, np.float32)
    new = (t > C._tile(C._blue(), w, h, C.grain_cell(h))).astype(np.uint8)
    old = _with(OLD, lambda: (t > C._tile(C._blue(), w, h, C.grain_cell(h))).astype(np.uint8))

    def mixed_blocks(m):
        b = m[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2)
        s = b.sum((1, 3))
        return float(((s > 0) & (s < 4)).mean())

    assert mixed_blocks(old) == 0.0
    assert mixed_blocks(new) > 0.4
    assert abs(float(new.mean()) - 0.5) < 0.05


@pytest.mark.skipif(C._nb is None, reason="numba нема")
def test_ultra_1080_finer_pattern_keeps_tone():
    """Ультра 1080p: з новим масштабом помилка дрібного тону не гірша за колишню, і візерунок справді інший"""
    w, h = 1440, 1080
    img = _scene(w, h)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def run():
        return C.binarize(img, w, h, C.Params(method=5), {"state": C.InkState()}).astype(np.float32)

    new = run()
    old = _with(OLD, run)
    assert not np.array_equal(new, old)
    err = lambda m: float(np.sqrt(np.mean((cv2.GaussianBlur(m, (0, 0), 1.5) - cv2.GaussianBlur(g, (0, 0), 1.5)) ** 2)))
    assert err(new) <= err(old) * 1.05
    assert abs(float(new.mean()) - float(old.mean())) < 0.03
