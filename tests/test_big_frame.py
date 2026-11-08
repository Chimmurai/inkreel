"""економні за пам'яттю шляхи для кадрів від BIG_H (4K) мають збігатися зі звичайними"""
import cv2
import numpy as np
import pytest

from benvid import core as C
from benvid import ink5

needs_nb = pytest.mark.skipif(C._nb is None, reason="numba нема")


@needs_nb
def test_snap_blend_matches_numpy_tail():
    rng = np.random.default_rng(3)
    h, w, s = 270, 480, 3
    t = rng.random((h, w), dtype=np.float32)
    mid, hr, sp = (rng.random((h // s, w // s)).astype(np.float32) * a for a in (1.0, 0.5, 1.0))
    mid_f, hr_f, sp_f = (cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in (mid, hr, sp))
    d = np.maximum(0.45 * hr_f, 0.03)
    z = np.clip((t - (mid_f - d)) / (2 * d), 0.0, 1.0)
    ref = (t + sp_f * (z * z * (3 - 2 * z) - t)).astype(np.float32)
    out = C._snap_blend_nb(t, mid, hr, sp)
    assert float(np.abs(out - ref).max()) < 2e-4


def test_small_frames_do_not_use_big_paths():
    assert C.BIG_H > 1080            # 720p і 1080p лишаються побітово як були


def test_guard_cal_identity_up_to_720p():
    """запобіжник «штрихи чи зерно»: до 720p поправка рівно 1 (результат не змінюється), вище росте (зерно 1 px на великих кадрах
    інакше забирає площу в штрихів) і не перевищує 2.5"""
    from benvid import ink5
    for h in (144, 240, 360, 480, 540, 720):
        assert ink5._guard_cal(h / 720.0) == 1.0
    vals = [ink5._guard_cal(h / 720.0) for h in (720, 900, 1080, 1440, 2160, 4320)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert vals[2] > 1.3 and vals[3] >= 2.4 and vals[-1] <= 2.5


def test_grain_cell_single_pixel_by_default():
    """зерно - окремі пікселі на будь-якому розмірі (раніше від 1008p було 2x2); PAT['grain']=0 повертає колишнє правило"""
    for h in (144, 720, 1080, 1440, 2160):
        assert C.grain_cell(h) == 1
    old = dict(C.PAT)
    try:
        C.PAT["grain"] = 0
        assert [C.grain_cell(h) for h in (720, 1007, 1008, 1080, 2160)] == [1, 1, 2, 2, 2]
    finally:
        C.PAT.update(old)


def test_dot_cell_uses_pattern_scale():
    """клітинка крапок = round(kp): 720p, 1080p і 1440p - 1 (k**0.5 = 1.22 / 1.41), 4K - 2"""
    for h, want in ((720, 1), (1080, 1), (1440, 1), (2160, 2)):
        assert max(1, int(C.pat_k(h / 720.0) + 0.5)) == want


def test_zone_area_floor_keeps_small_frames():
    from benvid import core as C
    ref = 1280.0 * 720.0
    for w, h in ((256, 144), (640, 360), (1280, 720), (1920, 1080)):
        assert max(ref / (w * h), C._AREA_FLOOR) == ref / (w * h)
    assert max(ref / (3840 * 2160), C._AREA_FLOOR) == C._AREA_FLOOR


# ------------------------------------------------------------------ запобіжник півтону й мертва зона (понад 720p)
def _dark_texture(w, h):
    """темна ділянка з дрібною текстурою (тканина, шерсть): те, що штрихи й крапки не вміщали"""
    rng = np.random.default_rng(3)
    n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.6)
    n /= max(float(n.std()), 1e-6)
    g = np.clip(38 + 16 * n + 14 * np.sin(np.arange(w, dtype=np.float32)[None, :] / 23.0), 0, 255).astype(np.uint8)
    g[: h // 3, : w // 3] = 220
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _detail_corr(img, m):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hp = lambda a, s: cv2.GaussianBlur(a, (0, 0), 1.2 * s) - cv2.GaussianBlur(a, (0, 0), 3.0 * s)
    z = cv2.GaussianBlur(g, (0, 0), 4.0) < 0.3
    a, b = hp(g, 1.0)[z], hp(m.astype(np.float32), 1.0)[z]
    return float(np.corrcoef(a, b)[0, 1])


@needs_nb
def test_halftone_guard_and_deadzone_ramp_are_off_up_to_720p(monkeypatch):
    """до 720p кадр Ультри не залежить ні від запобіжника півтону (hg), ні від ослаблення мертвої зони (dzd): до біта"""
    img = _dark_texture(960, 540)
    a = ink5.ultra(img, 960, 540, C.Params(method=5), C.InkState())
    b = ink5.ultra(img, 960, 540, C.Params(method=5), C.InkState(), None, hg=0, dzd=0.0)
    assert np.array_equal(a, b)
    seen = []
    real = ink5._deadzone
    monkeypatch.setattr(ink5, "_deadzone", lambda t, k, pr: (seen.append(round(pr["dz"], 4)), real(t, k, pr))[1])
    ink5.ultra(img, 960, 540, C.Params(method=5), C.InkState())
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    ink5.ultra(big, 1920, 1080, C.Params(method=5), C.InkState())
    assert seen[0] == round(ink5.P["dz"], 4), "до 720p чорна точка піднімається як раніше"
    assert seen[1] == round(ink5.P["dz"] - ink5.P["dzd"], 4), "на 1080p - слабше"


def test_halftone_map_prefers_the_grain_where_detail_is_dark_and_stylised_pattern_misses_it():
    """_halftone_map: темна детальна округа й візерунок без деталі -> зерно (р > 0.5); світла округа -> стилізація лишається; до 720p (k=1) - майже ніде"""
    h, w = 540, 960
    rng = np.random.default_rng(5)
    n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2)
    t = np.clip(0.3 + 0.18 * n / float(n.std()), 0, 1).astype(np.float32)
    grain = (t > C._tile(C._blue(), w, h, 1)).astype(np.uint8)                       # чисте півтонове зерно
    yy, xx = np.mgrid[0:h, 0:w]
    dots = (((xx % 5) < 2) & ((yy % 5) < 2)).astype(np.uint8)                        # візерунок із кроком 5 px: дрібну деталь не вміщає
    dark, light = np.full((h, w), 30.0, np.float32), np.full((h, w), 200.0, np.float32)
    p_dark = ink5._halftone_map(t, dots, grain, 1.5, ink5.P, dark)
    p_light = ink5._halftone_map(t, dots, grain, 1.5, ink5.P, light)
    p_720 = ink5._halftone_map(t, dots, grain, 1.0, ink5.P, dark)
    assert float(p_dark.mean()) > 0.5, float(p_dark.mean())
    assert float(p_light.mean()) < float(p_dark.mean()) - 0.2, (float(p_light.mean()), float(p_dark.mean()))
    assert float(p_720.mean()) < float(p_dark.mean()) - 0.2, "біля 720p запобіжник слабкий (плавний вхід)"
    same = ink5._halftone_map(t, grain, grain, 1.5, ink5.P, dark)
    assert float(same.max()) == 0.0, "де кадр уже збігається з зерном - вибирати нічого"


@needs_nb
@pytest.mark.parametrize("level", [4, 5])
def test_halftone_guard_changes_1080p_but_not_the_tone(level):
    """на 1080p запобіжник (Ультра) справді щось міняє, а на обох рівнях середня яскравість лишається, деталь не гіршає"""
    w, h = 1920, 1080
    img = _dark_texture(w, h)
    if level == 5:
        run = lambda **kw: ink5.ultra(img, w, h, C.Params(method=5), C.InkState(), None, **kw)
        off, on = run(hg=0, dzd=0.0), run()
    else:
        keep = dict(ink5.P)
        try:
            ink5.P["hg"] = 0
            off = C.binarize(img, w, h, C.Params(method=4))
        finally:
            ink5.P.update(keep)
        on = C.binarize(img, w, h, C.Params(method=4))
    if level == 5:
        assert not np.array_equal(on, off)
    assert abs(float(on.mean()) - float(off.mean())) < 0.03, "середня яскравість та сама"
    assert _detail_corr(img, on) >= _detail_corr(img, off) - 0.005, "деталь не гіршає"
