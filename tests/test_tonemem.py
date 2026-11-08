"""Часова пам'ять нормування тону (core.ToneMem): у відео автоконтраст не «пульсує» і не стирає згасання/появу.

Кадр сам нормується за своїми 0.5-м і 99.5-м процентилями (lo, hi) і медіанною гамою. На відео це давало «пампінг» (у кадр зайшов
світлий предмет - тло одразу темнішає) і стирало згасання (титр лишався білим, поки не зникне різко). Пам'ять згладжує дрібні
коливання, обмежує швидкість руху меж і ТРИМАЄ діапазон там, де він був, поки яскравість плавно змінюється; склейка скидає пам'ять.
Фото (без пам'яті) й перший кадр відео не змінюються ні на біт."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid import core as C  # noqa: E402

LO = 2.0
GAM = 0.7


def _raw(his, lo_t=LO):
    """послідовність справжніх hi кадрів -> сирі параметри, які дав би tone_map: у майже рівному кадрі (діапазон < 24) - 0..255"""
    for h in his:
        if h - lo_t < 24:
            yield 0.0, 255.0, GAM, lo_t, float(h)
        else:
            yield lo_t, float(h), GAM, lo_t, float(h)


def _run(tm, his, cut_first=True):
    out = []
    for i, (lo, hi, gam, lo_t, hi_t) in enumerate(_raw(his)):
        out.append(tm.step(lo, hi, gam, cut_first and i == 0, lo_t, hi_t))
    return out


def _fade(a, b, n, dup=1):
    """плавна зміна hi від a до b за n значень; dup - повторення кожного (джерело з меншою частотою кадрів дублює кадри)"""
    return [float(x) for x in np.repeat(np.linspace(a, b, n), dup)]


# ------------------------------------------------------------------ сама пам'ять (чиста арифметика)
def test_calm_frames_pass_through_bit_exact():
    tm = C.ToneMem()
    assert all(r == (LO, 200.0, GAM) for r in _run(tm, [200.0] * 30))


def test_first_frame_and_cut_adopt_the_raw_values():
    tm = C.ToneMem()
    assert tm.step(5.0, 90.0, 1.2, False, 5.0, 90.0) == (5.0, 90.0, 1.2)                   # перший кадр: пам'яті ще немає
    _run(tm, [200.0] * 10, cut_first=False)
    assert tm.step(5.0, 90.0, 1.2, True, 5.0, 90.0) == (5.0, 90.0, 1.2)                    # склейка - пам'ять скинуто
    assert tm.hold == [None, None]


def test_jump_is_rate_limited_converges_and_is_not_held():
    tm = C.ToneMem()
    _run(tm, [120.0] * 12)
    his = [tm.step(LO, 250.0, GAM, False, LO, 250.0)[1] for _ in range(60)]
    assert his[0] <= 120.0 + TNORM_Q() * (120.0 - LO) + 1e-6             # за кадр - не далі, ніж q діапазону
    assert all(b >= a for a, b in zip(his, his[1:]))                       # рівно в бік нового значення
    assert his[-1] == pytest.approx(250.0, abs=0.5)                        # і наздоганяє
    assert tm.hold == [None, None]                                         # різкий стрибок - не «плавна зміна»


def TNORM_Q():
    return C.TNORM["q"]


def test_fade_out_is_held_even_with_duplicated_frames():
    """CTH: титр гасне ~36 кадрів, джерело дублює кадри парами. Діапазон тримається там, де був, - згасання видно"""
    tm = C.ToneMem()
    his = [248.0] * 12 + _fade(248.0, 20.0, 18, dup=2) + [10.0] * 8
    out = _run(tm, his)
    held = [i for i in range(len(his)) if i >= 12 and out[i][1] >= 0.95 * 248.0 and his[i] < 0.6 * 248.0]
    fade_part = [i for i in range(12, len(his)) if his[i] < 0.6 * 248.0]
    assert fade_part and len(held) >= 0.9 * len(fade_part)                 # коли кадр уже вдвічі темніший, нормування ще старе
    ratio = his[12 + 18] / out[12 + 18][1]
    assert ratio < 0.6                                                      # відношення кадру до нормування = видима яскравість (посередині згасання ≈ ½)
    assert tm.hold == [None, None]                                         # у кінці (чорний кадр) пам'ять відпущено


def test_fade_in_from_black_is_held_at_full_range():
    tm = C.ToneMem()
    his = [6.0] * 10 + _fade(6.0, 248.0, 30, dup=2) + [248.0] * 8
    out = _run(tm, his)
    ramp = range(10, 10 + 60)
    visible = [his[i] / out[i][1] for i in ramp if his[i] > 60.0]
    assert min(out[i][1] for i in ramp) > 200.0                            # нормування не «провалилось» услід за темним кадром
    assert all(b >= a - 0.02 for a, b in zip(visible, visible[1:]))        # яскравість росте плавно, без стрибків униз
    assert out[-1][1] == pytest.approx(248.0, abs=8.0)                     # у кінці все знов як зазвичай


def test_hold_is_released_when_the_change_reverses():
    tm = C.ToneMem()
    his = [248.0] * 12 + _fade(248.0, 120.0, 12) + _fade(120.0, 248.0, 12) + [248.0] * 12
    out = _run(tm, his)
    assert tm.hold == [None, None]
    assert out[-1][1] == pytest.approx(248.0, abs=3.0)


def test_slow_brightening_beyond_the_old_range_is_not_held():
    """світлішає понад старий діапазон: тримати нормування означало б «засвітити» світле, тож пам'ять стежить за кадром"""
    tm = C.ToneMem()
    his = [100.0] * 12 + _fade(100.0, 250.0, 24) + [250.0] * 8
    held = []
    for i, (lo, hi, gam, lo_t, hi_t) in enumerate(_raw(his)):
        tm.step(lo, hi, gam, i == 0, lo_t, hi_t)
        held.append(tm.hold[1] is not None)
    assert not any(held)


def test_noise_around_a_level_never_triggers_a_hold():
    rng = np.random.default_rng(3)
    tm = C.ToneMem()
    his = list(200.0 + rng.normal(0.0, 2.5, 200))
    for i, (lo, hi, gam, lo_t, hi_t) in enumerate(_raw(his)):
        tm.step(lo, hi, gam, i == 0, lo_t, hi_t)
        assert tm.hold == [None, None]


# ------------------------------------------------------------------ кадри
W, H = 640, 360


def _scene(level, w=W, h=H):
    """темне тло + титр (світлий прямокутник із літерами), яскравість титру level (0..1)"""
    img = np.full((h, w, 3), 10, np.uint8)
    c = int(round(255 * level))
    cv2.rectangle(img, (110, 110), (530, 250), (c, c, c), -1)
    cv2.putText(img, "THE CALL", (130, 215), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (10, 10, 10), 6, cv2.LINE_AA)
    return img


def test_photo_and_first_frame_are_untouched_by_the_memory():
    img = _scene(1.0)
    t0, L0 = C.tone_map(img, W, H, True)
    t1, L1 = C.tone_map(img, W, H, True, tmem=C.ToneMem())
    assert np.array_equal(t0, t1) and np.array_equal(L0, L1)


def test_repeated_frame_stays_bit_exact_with_memory():
    img = _scene(0.8)
    t0, _ = C.tone_map(img, W, H, True)
    tm = C.ToneMem()
    for _ in range(6):
        t1, _ = C.tone_map(img, W, H, True, tmem=tm)
        assert np.array_equal(t0, t1)


def _region_tone(t):
    return float(np.clip(t[130:230, 140:520], 0.0, 1.0).mean())


def test_fade_is_visible_in_the_tone_map_with_memory_and_cancelled_without():
    levels = [1.0] * 14 + list(np.linspace(1.0, 0.2, 26))
    tm = C.ToneMem()
    with_mem, without = [], []
    for lv in levels:
        img = _scene(lv)
        with_mem.append(_region_tone(C.tone_map(img, W, H, True, tmem=tm)[0]))
        without.append(_region_tone(C.tone_map(img, W, H, True)[0]))
    mid = 14 + 16                                                          # титр на ~45% яскравості
    assert without[mid] > 0.9 * without[0]                                 # поодинці кожен кадр нормується до білого - згасання стерто
    assert with_mem[mid] < 0.6 * with_mem[0]                               # з пам'яттю титр відчутно тьмяніє
    seq = with_mem[14:]
    assert all(b <= a + 0.05 for a, b in zip(seq, seq[1:]))                # і тьмяніє монотонно, без «стрибків» назад


def test_cut_resets_the_normalisation_between_scenes():
    tm = C.ToneMem()
    dark = np.full((H, W, 3), 25, np.uint8)
    cv2.circle(dark, (320, 180), 70, (60, 60, 60), -1)
    for _ in range(10):
        C.tone_map(dark, W, H, True, tmem=tm)
    bright = _scene(1.0)
    t_mem, _ = C.tone_map(bright, W, H, True, tmem=tm)
    t_free, _ = C.tone_map(bright, W, H, True)
    assert np.array_equal(t_mem, t_free)                                   # склейка на іншу сцену - нормування з нуля


def test_priming_with_thumbnails_only_needs_no_full_frame():
    """прогрів шматка відео кадрами без малювання (tone_prime) дає ті самі сирі параметри, що й повний прохід tone_map"""
    levels = [1.0] * 20
    full, prim = C.ToneMem(), C.ToneMem()
    for lv in levels:
        C.tone_map(_scene(lv), W, H, True, tmem=full)
        C.tone_prime(_scene(lv), W, H, True, prim)
    assert abs(full.hi - prim.hi) < 1.0 and abs(full.lo - prim.lo) < 1.0 and abs(full.gam - prim.gam) < 0.02


# ------------------------------------------------------------------ обидва режими: згасання титру видно в масці
def _fade_densities(glyph, with_state):
    p = C.Params(method=5, glyph=glyph)
    st = C.InkState() if with_state else None
    levels = [1.0] * 12 + list(np.linspace(1.0, 0.2, 24))
    out = []
    for lv in levels:
        m = C.binarize(_scene(lv), W, H, p, {"state": st} if st is not None else None)
        out.append(float(m[130:230, 140:520].mean()))
    return levels, out


@pytest.mark.parametrize("glyph", [False, True], ids=["ink", "symbols"])
def test_fade_shows_in_the_mask_in_both_modes(glyph):
    levels, dens = _fade_densities(glyph, True)
    mid = 12 + 16                                                          # титр на ~45% яскравості
    assert dens[mid] < 0.65 * dens[0], (dens[0], dens[mid])                # густина знаків/крапок помітно впала
    tail = dens[12:]
    assert all(b <= a + 0.06 for a, b in zip(tail, tail[1:]))              # без стрибків угору посеред згасання


@pytest.mark.parametrize("glyph", [False, True], ids=["ink", "symbols"])
def test_stateless_frames_ignore_the_memory(glyph):
    """окремий кадр (фото, скріншот) без стану нормується лише за собою: як було"""
    p = C.Params(method=5, glyph=glyph)
    a = C.binarize(_scene(0.5), W, H, p)
    b = C.binarize(_scene(0.5), W, H, p, {"state": None})
    assert np.array_equal(a, b)


def test_symbols_tone_size_matches_the_grid_and_priming_uses_it():
    from benvid import glyph
    gw, gh = glyph.tone_size(W, H)
    rows, cols = glyph.grid_for(W, H)
    assert gw % cols == 0 and gh % rows == 0 and gw >= cols and gh >= rows
    st = C.InkState()
    glyph.prime(st, _scene(1.0), W, H, True)
    assert C.tmem_of(st).lo is not None                                     # пам'ять прогріта кадром без малювання


# ------------------------------------------------------------------ знебарвлення (PCA) не «дихає» між кадрами
def _pca_scene(block):
    """кольорове тло з двох половин (зелено-сіре й пурпурове); block - у кадр заходить світлий предмет: сирий напрям проєкції стрибає"""
    img = np.zeros((180, 320, 3), np.uint8)
    img[:, :200] = (60, 103, 48)
    img[:, 200:] = (209, 67, 129)
    if block:
        cv2.rectangle(img, (10, 30), (58, 150), (214, 206, 203), -1)
    return img


needs_numba = pytest.mark.skipif(C._nb is None, reason="згладжування проєкції є лише в шляху з numba")


@needs_numba
def test_pca_first_frame_cut_and_static_frames_are_stateless_exact():
    A, B = _pca_scene(False), _pca_scene(True)
    assert np.array_equal(C._gray_pca(A, 0.5, None, C.ToneMem(), True), C._gray_pca(A, 0.5, None))
    tm = C.ToneMem()
    for i in range(5):                                                      # той самий кадр знову й знову - рівно як окремо
        assert np.array_equal(C._gray_pca(B, 0.5, None, tm, i == 0), C._gray_pca(B, 0.5, None))
    C._gray_pca(A, 0.5, None, tm, False)
    assert np.array_equal(C._gray_pca(B, 0.5, None, tm, True), C._gray_pca(B, 0.5, None))   # склейка скидає пам'ять


@needs_numba
@pytest.mark.parametrize("forward", [True, False], ids=["enters", "leaves"])
def test_pca_background_gray_moves_at_a_bounded_rate_and_converges(forward):
    """у кадр заходить (чи з нього йде) світлий предмет: сирий сірий пурпурового тла стрибає на ~50 рівнів за кадр.
    З пам'яттю - не більш ніж pstep за кадр, рівно в бік нового значення, без перегону, і в кінці збігається з сирим"""
    A, B = _pca_scene(False), _pca_scene(True)
    first, last = (A, B) if forward else (B, A)
    ra, rb = C._gray_pca(first, 0.5, None), C._gray_pca(last, 0.5, None)
    jump = int(rb[5, 310]) - int(ra[5, 310])
    assert abs(jump) > 40                                                   # сцена справді дає сильний стрибок
    tm = C.ToneMem()
    bg = [int(C._gray_pca(im, 0.5, None, tm, i == 0)[5, 310]) for i, im in enumerate([first] * 2 + [last] * 24)]
    steps = [b - a for a, b in zip(bg, bg[1:])]
    assert max(abs(x) for x in steps) <= C.TNORM["pstep"] + 2
    sg = 1 if jump > 0 else -1
    assert all(x * sg >= 0 for x in steps)                                 # рівно в бік нового значення (без перегону)
    assert abs(bg[-1] - int(rb[5, 310])) <= 1                              # і наздоганяє
    assert bg[2] != int(rb[5, 310])                                        # а не стрибає за кадр


# ------------------------------------------------------------------ точка зриву: інша картинка - контраст перестрибує
def _step_q(tm, lo, hi, q, cut=False, gam=GAM):
    tm.qnow = q
    return tm.step(lo, hi, gam, cut, lo, hi)


def test_body_jump_resets_the_memory_at_once():
    """змінилась уся картинка (25-й, 50-й, 75-й процентилі пішли разом): нормування одразу нове, а не повзе десятки кадрів"""
    tm = C.ToneMem()
    for _ in range(12):
        _step_q(tm, LO, 120.0, (20.0, 55.0, 90.0))
    assert _step_q(tm, 60.0, 250.0, (110.0, 160.0, 215.0)) == (60.0, 250.0, GAM)
    assert tm.hold == [None, None]
    assert tm.lo == 60.0 and tm.hi == 250.0


def test_title_does_not_break_the_memory():
    """з'явився титр: верхній край стрибнув, а «тіло» кадру стоїть - це не зрив, швидкість меж і далі обмежена"""
    tm = C.ToneMem()
    for _ in range(12):
        _step_q(tm, LO, 120.0, (20.0, 55.0, 90.0))
    hi = _step_q(tm, LO, 250.0, (21.0, 56.0, 92.0))[1]
    assert hi <= 120.0 + TNORM_Q() * (120.0 - LO) + 1e-6, hi


def test_far_and_still_target_breaks_after_a_few_frames_not_at_once():
    """нова картинка (тіло кадру відійшло від колишнього) стоїть далеко від стану й не рухається nfar кадрів - стрибок, але не раніше
    (один-два кадри - ще може бути спалах)"""
    tm = C.ToneMem()
    for _ in range(12):
        _step_q(tm, LO, 120.0, (20.0, 55.0, 90.0))
    n = C.TNORM["nfar"]
    body = [(20.0 + 12.0 * (i + 1), 55.0 + 12.0 * (i + 1), 90.0 + 12.0 * (i + 1)) for i in range(n + 2)]      # тіло відходить повільно: за кадр менше за brk
    seen = [_step_q(tm, LO, 250.0, q)[1] for q in body]
    assert all(h < 250.0 for h in seen[:n - 1]), seen                       # перші кадри - плавно (межа швидкості)
    assert 250.0 in seen[n - 1:], seen                                      # далі - стрибок
    assert tm.hi == 250.0


def test_persistent_title_does_not_break_the_memory():
    """титр, що висить: верхній край стрибнув і стоїть, але тіло кадру те саме - тло не смикається, нормування наздоганяє плавно"""
    tm = C.ToneMem()
    for _ in range(12):
        _step_q(tm, LO, 120.0, (20.0, 55.0, 90.0))
    his = [_step_q(tm, LO, 250.0, (21.0, 56.0, 91.0))[1] for _ in range(40)]
    assert all(b - a <= TNORM_Q() * 250.0 + 1e-6 for a, b in zip([120.0] + his, his)), his     # жодного стрибка
    assert his[-1] == pytest.approx(250.0, abs=1.0) and all(b >= a for a, b in zip(his, his[1:]))


def test_slow_fade_is_not_a_break():
    """плавна поява/згасання (край іде одним боком) точку зриву не вмикає: пам'ять тримає діапазон, як і раніше"""
    tm = C.ToneMem()
    his = [6.0] * 10 + _fade(6.0, 248.0, 30, dup=2) + [248.0] * 8
    for i, (lo, hi, gam, lo_t, hi_t) in enumerate(_raw(his)):
        tm.qnow = (2.0, 3.0 + 0.5 * min(hi_t, 100.0) / 10, 4.0 + hi_t / 8)
        tm.step(lo, hi, gam, i == 0, lo_t, hi_t)
        assert tm.lo is not None and tm.hold is not None
    assert tm.hi == pytest.approx(248.0, abs=8.0)


def test_break_point_does_not_touch_photos_or_the_first_frame():
    """без пам'яті (фото) і на першому кадрі нічого не змінилось"""
    tm = C.ToneMem()
    tm.qnow = (1.0, 2.0, 3.0)
    assert tm.step(5.0, 90.0, 1.2, False, 5.0, 90.0) == (5.0, 90.0, 1.2)
