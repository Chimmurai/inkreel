import time

import cv2
import numpy as np

from benvid import autosize
from benvid.player import Player


def _frame(w=640, h=360):
    rng = np.random.default_rng(0)
    return cv2.GaussianBlur((rng.random((h, w, 3)) * 255).astype(np.uint8), (0, 0), 3)


def test_pick_is_level_aware():
    """«Ультра» коштує в рази дорожче за «Графіку»: за тих самих ресурсів розмір «Авто» має бути меншим"""
    fr = _frame(1920, 1080)
    cheap = (1e-7, {480: 1e-7})
    dear = (6e-7, {480: 6e-7, 720: 5e-7})
    k4 = autosize.pick(fr, 30, 8, src_h=1080, costs=cheap, method=4)
    k5 = autosize.pick(fr, 30, 1, src_h=1080, costs=dear, method=5, threads=4)
    h4, h5 = int(k4[:-1]), int(k5[:-1])
    assert h4 >= 720 and h5 < h4
    # слабкіший ПК - менший розмір
    assert int(autosize.pick(fr, 30, 2, src_h=1080, costs=cheap, method=4)[:-1]) < h4
    # не більше висоти джерела
    assert int(autosize.pick(_frame(), 30, 64, src_h=360, costs=cheap, method=4)[:-1]) <= 360


def test_frame_cost_uses_nearest_measured_below():
    big = {480: 3.0, 720: 2.0}
    w, h = 1920, 1080
    assert autosize._frame_cost(9.0, big, 1280, 720, 5) == 2.0 * 1280 * 720
    assert autosize._frame_cost(9.0, big, 854, 480, 5) == 3.0 * 854 * 480
    assert autosize._frame_cost(9.0, big, w, h, 5) == 2.0 * w * h          # вище виміряного - від найбільшого виміряного
    assert autosize._frame_cost(9.0, big, 256, 144, 5) == 9.0 * 256 * 144  # малий шлях
    # нижче найменшої виміряної висоти: «Ультра» не дешевшає (вартість найменшого виміряного кадру), решта - за пікселями
    assert abs(autosize._frame_cost(9.0, big, 640, 360, 5) - 3.0 * (640 / 360 * 480) * 480) < 1e-6
    assert autosize._frame_cost(9.0, {480: 3.0}, 640, 360, 4) == 3.0 * 640 * 360


def test_measure_runs_for_every_level_and_restores_threads():
    fr = _frame()
    import numba
    nb0, cv0 = numba.get_num_threads(), cv2.getNumThreads()
    for lv, thr in ((1, 1), (3, 1), (5, 2)):
        small, big = autosize.measure(fr, lv, thr)
        assert small > 0 and all(v > 0 for v in big.values())
    assert numba.get_num_threads() == nb0 and cv2.getNumThreads() == cv0


def test_auto_pick_depends_on_current_level_and_releases_hold(tmp_path):
    path = str(tmp_path / "c.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 180))
    for i in range(20):
        vw.write(np.roll(_frame(320, 180), i * 3, axis=1))
    vw.release()
    p = Player(workers=1)
    try:
        p.open(path, "144p")
        from benvid.core import Params
        for m in (2, 5):
            p.set_params(Params(method=m))
            key = p.auto_pick()
            assert key.endswith("p") and int(key[:-1]) in autosize.CANDIDATES
            assert not p._hold
    finally:
        p.close()


def test_hold_pauses_new_work_until_released(tmp_path):
    path = str(tmp_path / "d.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (160, 96))
    for i in range(30):
        vw.write(np.full((96, 160, 3), 40 + i, np.uint8))
    vw.release()
    p = Player(workers=1)
    try:
        p.hold(True)
        p.open(path, "144p")
        time.sleep(0.6)
        assert p.cached == 0
        p.hold(False)
        end = time.time() + 30
        while p.cached < 5 and time.time() < end:
            p.update()
            time.sleep(0.02)
        assert p.cached >= 5
    finally:
        p.close()
