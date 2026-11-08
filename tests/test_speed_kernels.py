"""швидкі ядра (злиті проходи) дають те саме, що прямі формули, за якими їх написано (відхилення - лише порядок округлення)"""
import cv2
import numpy as np
import pytest

from benvid import core as C
from benvid import glyph, ink5

needs_nb = pytest.mark.skipif(C._nb is None, reason="numba нема")


@needs_nb
@pytest.mark.parametrize("prev_on", [False, True])
@pytest.mark.parametrize("hw,f", [((120, 200), 4), ((121, 203), 4), ((90, 150), 2), ((60, 100), 3)])
def test_fused_feedback_pass_equals_mask_then_block_sums(hw, f, prev_on):
    h, w = hw
    rng = np.random.default_rng(1)
    tc = rng.random((h, w), dtype=np.float32)
    thr = rng.random((h, w), dtype=np.float32)
    cumf = (rng.random((h, w), dtype=np.float32) - 0.5) * 0.4
    prev = (rng.random((h, w)) > 0.5).astype(np.uint8) if prev_on else np.zeros((1, 1), np.uint8)
    hy = 0.05 if prev_on else 0.0
    sh, sw = -(-h // f), -(-w // f)
    out = np.empty((h, w), np.uint8)
    ink5._fb_nb(tc, thr, cumf, prev, hy, out)
    ref = np.zeros((sh, sw), np.float64)
    ink5._blk_u8(out, f, ref)
    got = np.zeros((sh, sw), np.float64)
    ink5._fb_acc_nb(tc, thr, cumf, prev, hy, f, got)
    assert np.array_equal(ref, got)


@needs_nb
@pytest.mark.parametrize("hw,g", [((120, 200), (30, 50)), ((97, 131), (24, 33)), ((64, 64), (2, 2))])
def test_select_matches_bilinear_pick(hw, g):
    h, w = hw
    rng = np.random.default_rng(2)
    mS = (rng.random((h, w)) > 0.5).astype(np.uint8)
    mG = (rng.random((h, w)) > 0.5).astype(np.uint8)
    pg = rng.random(g, dtype=np.float32)
    pg[: g[0] // 3] = 0.0
    Tr = rng.random((h, w), dtype=np.float32)
    p = cv2.resize(pg, (w, h), interpolation=cv2.INTER_LINEAR)
    ref = np.where(Tr < p, mG, mS)
    out = ink5._select_nb(mS, mG, pg, Tr)
    assert np.mean(out != ref) < 0.001                      # цілі відмінності - лише на межі порога через округлення


@needs_nb
def test_error_grid_matches_the_chain_it_replaces():
    h, w = 216, 384
    rng = np.random.default_rng(4)
    t = cv2.GaussianBlur(rng.random((h, w), dtype=np.float32), (0, 0), 2.0)
    m = (cv2.GaussianBlur(rng.random((h, w), dtype=np.float32), (0, 0), 1.0) > 0.5).astype(np.uint8)
    k = 1.5
    bt = ink5._bp(t, k)
    ref = cv2.resize((ink5._bp(m.astype(np.float32), k) - bt) ** 2, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    got = ink5._err_grid(m, bt, k, h // 4, w // 4)
    assert got.shape == ref.shape
    assert float(np.abs(got - ref).max()) < 0.02 * float(ref.max())      # uint8-гаус: похибка ≤ 0.2% тону, на клітинці це кілька відсотків від піку
    assert float(np.corrcoef(got.ravel(), ref.ravel())[0, 1]) > 0.999


@needs_nb
def test_error_grid_odd_size_uses_generic_path():
    h, w = 105, 171
    rng = np.random.default_rng(5)
    t = rng.random((h, w), dtype=np.float32)
    m = (rng.random((h, w)) > 0.5).astype(np.uint8)
    bt = ink5._bp(t, 1.0)
    got = ink5._err_grid(m, bt, 1.0, max(2, h // 4), max(2, w // 4))
    assert got.shape == (26, 42) and np.isfinite(got).all() and float(got.min()) >= 0.0


@needs_nb
def test_guard_and_halftone_share_the_tone_bandpass_per_frame():
    h, w = 108, 192
    rng = np.random.default_rng(6)
    t = cv2.GaussianBlur(rng.random((h, w), dtype=np.float32), (0, 0), 2.0)
    cache = {}
    a = ink5._bt_get(t, 1.5, False, cache)
    b = ink5._bt_get(t, 1.5, False, cache)
    assert a is b, "той самий кадр, k і масштаб: смуговий фільтр тону рахується один раз"
    c = ink5._bt_get(t, 1.5, True, cache)
    assert c is not a and c.shape == (h // 2, w // 2)


@needs_nb
def test_smoothstep_kernel_is_bit_exact_with_numpy():
    rng = np.random.default_rng(7)
    x = (rng.random((97, 131), dtype=np.float32) * 1.8 - 0.4).astype(np.float32)
    for a, b in ((0.45, 0.7), (4, 40), (-0.3, 0.9)):
        assert np.array_equal(C._sm(x, a, b), C._sm_py(x, a, b))
    xd = x.astype(np.float64)
    assert C._sm(xd, 0.1, 0.5).dtype == np.float64              # не float32 - попередній шлях


@needs_nb
def test_level_w_in_component_boxes_equals_whole_frame():
    h, w = 120, 200
    rng = np.random.default_rng(8)
    ts = (0.5 + 0.02 * rng.standard_normal((h, w))).astype(np.float32)
    ts[:, :60] = np.linspace(0.2, 0.6, 60, dtype=np.float32)          # плавний градієнт - не рівна ділянка
    ts[30:90, 100:170] = 0.9
    fm = np.zeros((h, w), np.uint8)
    fm[:, :60] = 1
    fm[30:90, 100:170] = 1
    fm[10:50, 70:90] = 1
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fm, connectivity=4)
    size = np.bincount(lab.ravel(), minlength=n)
    # ділянки менші за 1% кадру ігноруються: беремо кадр, де великі є
    full = C._level_w(ts, lab, size, 1.0, 1)
    boxed = C._level_w(ts, lab, size, 1.0, 1, stats)
    assert np.allclose(full, boxed, atol=1e-6)


@needs_nb
def test_equalize_kernel_equals_interp():
    rng = np.random.default_rng(9)
    t = np.clip(rng.beta(2.0, 5.0, (90, 160)), 0, 1).astype(np.float32)
    got = glyph._equalize(t, None)
    # еталон: той самий ланцюжок на np.interp
    sub = t[::3, ::3] if t.shape[0] > 96 else t
    q = np.percentile(sub, np.linspace(0, 100, 33)).astype(np.float32)
    m = glyph.EQ * float(np.clip((float(q[31] - q[1]) - 0.05) / 0.25, 0.0, 1.0))
    qq = np.maximum.accumulate(q + np.arange(33, dtype=np.float32) * 1e-5)
    y = np.interp(t, qq, np.linspace(0.0, 1.0, 33, dtype=np.float32)).astype(np.float32)
    h_ = np.bincount(np.clip((sub * glyph.EQ_BINS).astype(np.int32), 0, glyph.EQ_BINS - 1).ravel(), minlength=glyph.EQ_BINS).astype(np.float64)
    h_ /= max(h_.sum(), 1.0)
    cap = glyph.EQ_CLIP / glyph.EQ_BINS
    h_ = np.minimum(h_, cap) + np.maximum(h_ - cap, 0.0).sum() / glyph.EQ_BINS
    cdf = np.concatenate([[0.0], np.cumsum(h_)]).astype(np.float32)
    cdf /= cdf[-1]
    lo, hi = glyph.EQ_PLATEAU
    wgt = float(np.clip((float(np.max(np.bincount(np.clip((sub * glyph.EQ_BINS).astype(np.int32), 0, glyph.EQ_BINS - 1).ravel(), minlength=glyph.EQ_BINS))) / sub.size - lo) / (hi - lo), 0.0, 1.0))
    if wgt > 0.0:
        y2 = np.interp(t, np.linspace(0.0, 1.0, glyph.EQ_BINS + 1, dtype=np.float32), cdf).astype(np.float32)
        y = ((1.0 - wgt) * y + wgt * y2).astype(np.float32)
    ref = ((1.0 - m) * t + m * y).astype(np.float32)
    assert float(np.abs(got - ref).max()) < 2e-6


@needs_nb
def test_smooth_kernel_equals_numpy_and_keeps_memory():
    rng = np.random.default_rng(10)
    t0 = rng.random((40, 70), dtype=np.float32)
    t1 = np.clip(t0 + 0.05 * rng.standard_normal((40, 70)).astype(np.float32), 0, 1)
    mem = {}
    glyph._smooth(t0, mem)
    old = mem["T"].copy()
    got = glyph._smooth(t1, mem)
    a = np.clip(np.abs(t1 - old) / 0.2, 0.3, 1.0)
    ref = old + a * (t1 - old)
    assert np.allclose(got, ref, atol=1e-6) and np.allclose(mem["T"], ref, atol=1e-6)


@needs_nb
def test_cell_stats_kernel_equals_numpy():
    rng = np.random.default_rng(11)
    F = rng.random((500, 16 * 8), dtype=np.float32)
    fm, si = glyph._cell_stats_nb(F, 16, 8)
    assert np.allclose(fm, F.mean(axis=1), atol=1e-6) and np.allclose(si, glyph._coarse(F, 16, 8), atol=1e-6)


@needs_nb
def test_pick_with_bias_equals_bias_added_to_scores():
    rng = np.random.default_rng(12)
    n, G = 64, 40
    S = rng.random((n, G), dtype=np.float32) * 30
    bias = -rng.random(G, dtype=np.float32) * 10
    dens = np.sort(rng.random(G, dtype=np.float32))
    fm = rng.random(n, dtype=np.float32)
    prev = np.full(n, -1, np.int32)
    a = glyph._pick(S.copy(), dens, fm, 20.0, prev, 1.5, 0, bias=bias)
    b = glyph._pick((S + bias).astype(np.float32), dens, fm, 20.0, prev, 1.5, 0)
    assert np.array_equal(a, b)


def test_kill_pool_ok_when_called_outside_the_manager_thread():
    from benvid import video
    ex = video.make_pool(1)
    video.kill_pool(ex)
    video.kill_pool(None)


def _plateau_frame(h=360, w=640, seed=7):
    """кадр із білим і чорним плато, на яких є слабка структура (щоб _recover і _deadzone справді працювали)"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = 0.5 + 0.4 * np.sin(xx / 37.0) * np.cos(yy / 29.0)
    t[: h // 3, : w // 2] = 1.0                                   # біле плато
    t[2 * h // 3:, w // 2:] = 0.0                                 # чорне плато
    t[h // 12:h // 6, w // 12:w // 4] = 0.93 + 0.04 * rng.random((h // 6 - h // 12, w // 4 - w // 12), dtype=np.float32)   # деталь на білому
    t[3 * h // 4:11 * h // 12, 7 * w // 12:3 * w // 4] = 0.02 + 0.05 * rng.random((11 * h // 12 - 3 * h // 4, 3 * w // 4 - 7 * w // 12), dtype=np.float32)
    return np.clip(t, 0, 1).astype(np.float32)


@needs_nb
@pytest.mark.parametrize("hw,g", [((120, 200), (30, 50)), ((97, 131), (25, 33))])
def test_deadzone_kernel_matches_the_formula(hw, g):
    h, w = hw
    rng = np.random.default_rng(3)
    t = rng.random((h, w), dtype=np.float32)
    t[5:9, 5:40] = 0.0
    t[20:24, 10:50] = 1.0
    lo = (0.08 * rng.random(g, dtype=np.float32)).astype(np.float32)
    hi = (0.05 * rng.random(g, dtype=np.float32)).astype(np.float32)
    got = ink5._dz_apply_nb(t, lo, hi)
    lof = cv2.resize(lo, (w, h), interpolation=cv2.INTER_LINEAR)
    hif = cv2.resize(hi, (w, h), interpolation=cv2.INTER_LINEAR)
    ref = np.where((t >= 1.0) | (t <= 0.0), t, (t - lof) / (1.0 - lof - hif)).astype(np.float32)
    assert np.allclose(got, ref, atol=2e-5)
    assert np.array_equal(got[t >= 1.0], t[t >= 1.0]) and np.array_equal(got[t <= 0.0], t[t <= 0.0]), "суцільні 0 і 1 лишаються"


@needs_nb
@pytest.mark.parametrize("hw,g", [((120, 200), (30, 50)), ((97, 131), (25, 33))])
def test_clip_add_kernel_matches_resize_then_clip(hw, g):
    h, w = hw
    rng = np.random.default_rng(4)
    t = rng.random((h, w), dtype=np.float32)
    ds = ((rng.random(g, dtype=np.float32) - 0.5) * 0.6).astype(np.float32)
    ref = np.clip(t + cv2.resize(ds, (w, h), interpolation=cv2.INTER_LINEAR), 0, 1)
    assert np.allclose(ink5._clip_add_nb(t, ds), ref, atol=2e-5)


@needs_nb
@pytest.mark.parametrize("hi_side", [True, False])
def test_recover_accumulation_kernel_matches_numpy(hi_side):
    rng = np.random.default_rng(5)
    shp = (80, 140)
    delta = (rng.random(shp, dtype=np.float32) - 0.5) * 0.1
    pz = rng.random(shp, dtype=np.float32)
    pz[rng.random(shp) > 0.6] = 0.0                              # поза плато поправки нема
    gt = rng.random(shp, dtype=np.float32)
    r = (rng.random(shp, dtype=np.float32) - 0.5) * 2.4
    ref = delta.copy()
    if hi_side:
        ref -= pz * gt * 0.42 * np.clip(-r, 0, 1) ** 0.8
    else:
        ref += pz * gt * 0.42 * np.clip(r, 0, 1) ** 0.8
    got = delta.copy()
    ink5._rec_acc_nb(got, pz, gt, r, hi_side)
    assert np.allclose(got, ref, atol=1e-6)


@needs_nb
def test_recover_and_deadzone_give_the_same_frame_with_and_without_numba(monkeypatch):
    t = _plateau_frame()
    gf = (t * 255.0).astype(np.float32)
    k = t.shape[0] / C.REF_H
    fast_r = ink5._recover(t, gf, k)
    fast_d = ink5._deadzone(fast_r, k, ink5.P)
    monkeypatch.setattr(ink5, "HAVE_NUMBA", False)
    slow_r = ink5._recover(t, gf, k)
    slow_d = ink5._deadzone(slow_r, k, ink5.P)
    assert not np.array_equal(fast_r, t), "на плато поправка справді діє"
    assert np.allclose(fast_r, slow_r, atol=1e-4)
    assert np.allclose(fast_d, slow_d, atol=1e-4)


@needs_nb
def test_flow_fields_give_the_same_with_and_without_numba(monkeypatch):
    rng = np.random.default_rng(6)
    yy, xx = np.mgrid[0:240, 0:427].astype(np.float32)
    ts = (0.5 + 0.3 * np.sin((xx + 0.6 * yy) / 11.0) + 0.1 * np.sin(yy / 5.0)).astype(np.float32)
    ts += 0.02 * rng.random(ts.shape, dtype=np.float32)
    ts = np.ascontiguousarray(np.clip(ts, 0, 1))
    fast = ink5._flow(ts, 0.333, ink5.P)
    monkeypatch.setattr(ink5, "HAVE_NUMBA", False)
    slow = ink5._flow(ts, 0.333, ink5.P)
    for a, b in zip(fast, slow):
        assert np.allclose(a, b, atol=2e-4)
