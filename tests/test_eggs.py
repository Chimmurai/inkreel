"""Пасхалки: чиста логіка (лічильник кліків, фрази, розфокусування, оберт котушки, вечірка палітр, крихти)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from benvid import eggs, guide, i18n, palette


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_clicks_fire_on_nth_quick_click_and_reset():
    clk = FakeClock()
    c = eggs.Clicks(5, gap=1.0, clock=clk)
    res = []
    for _ in range(5):
        clk.t += 0.3
        res.append(c.hit())
    assert res == [False, False, False, False, True]
    assert c.count == 0                                   # після спрацювання лічильник чистий
    clk.t += 0.3
    assert c.hit() is False and c.count == 1


def test_clicks_slow_taps_do_not_count():
    clk = FakeClock()
    c = eggs.Clicks(3, gap=1.0, clock=clk)
    for _ in range(10):
        clk.t += 1.5                                      # завжди довша пауза за gap
        assert c.hit() is False
    assert c.count == 1


def test_clicks_count_exposed_for_cookie_progress():
    clk = FakeClock()
    c = eggs.Clicks(eggs.COOKIE_CLICKS, gap=1.6, clock=clk)
    seen = []
    for _ in range(eggs.COOKIE_CLICKS - 1):
        clk.t += 0.2
        c.hit()
        seen.append(c.count)
    assert seen == list(range(1, eggs.COOKIE_CLICKS))     # 1..6 відкусів
    clk.t += 0.2
    assert c.hit() is True


def test_quotes_never_repeat_back_to_back_and_cover_all():
    rng = random.Random(1)
    last, seen = -1, set()
    for _ in range(400):
        i = eggs.pick_quote(last, rng)
        assert 0 <= i < len(eggs.QUOTES) and i != last
        seen.add(i)
        last = i
    assert seen == set(range(len(eggs.QUOTES)))


def test_quotes_first_pick_can_be_any():
    rng = random.Random(7)
    assert {eggs.pick_quote(-1, rng) for _ in range(300)} == set(range(len(eggs.QUOTES)))


def test_quotes_are_short_and_translated():
    from benvid.lang_en import EN
    assert len(set(eggs.QUOTES)) == len(eggs.QUOTES)
    for q in eggs.QUOTES:
        assert q in EN and len(q) <= 60 and len(EN[q]) <= 60, q


def test_defocus_curve_shape():
    T = eggs.DEFOCUS_TOTAL
    assert eggs.defocus_curve(0.0) == 0.0 and eggs.defocus_curve(T) == 0.0 and eggs.defocus_curve(-1) == 0.0
    assert eggs.defocus_curve(T * 1.5) == 0.0
    vals = [eggs.defocus_curve(T * i / 200) for i in range(201)]
    assert max(vals) == pytest.approx(1.0)
    peak = vals.index(max(vals))
    assert all(a <= b + 1e-12 for a, b in zip(vals[:peak], vals[1:peak + 1]))        # плавно вгору
    top = [i for i, v in enumerate(vals) if v > 0.999]
    assert top[-1] - top[0] > 200 * 0.3                                              # «тримаємо» помітний час
    last_up = max(i for i, v in enumerate(vals) if v >= max(vals) - 1e-9)
    assert all(a >= b - 1e-12 for a, b in zip(vals[last_up:], vals[last_up + 1:]))   # плавно назад
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_defocus_sigma_bounds():
    assert eggs.defocus_sigma(0.0, 720, 1280) == 0.0
    s_small, s_mid, s_big = (eggs.defocus_sigma(1.0, h, w) for h, w in ((90, 160), (360, 640), (2160, 3840)))
    assert 1.8 <= s_small <= s_mid <= s_big <= 4.0
    assert eggs.defocus_sigma(0.5, 720, 1280) == pytest.approx(eggs.defocus_sigma(1.0, 720, 1280) / 2)


def _checker(h=96, w=128):
    rng = np.random.default_rng(3)
    return (rng.random((h, w)) < 0.5).astype(np.uint8)         # шумовий растр 50 %


def test_defocus_zero_returns_same_mask():
    m = _checker()
    assert eggs.defocus(m, 0.0) is m
    assert eggs.defocus(m, 0.003) is m


def test_defocus_turns_halftone_into_shades():
    m = _checker()
    d = eggs.defocus(m, 1.0)
    assert d.dtype == np.float32 and d.shape == m.shape
    assert 0.0 <= float(d.min()) and float(d.max()) <= 1.0
    assert float(d.std()) < 0.5 * float(m.astype(np.float32).std())            # растр злився в сірі відтінки
    assert abs(float(d.mean()) - float(m.mean())) < 0.02                       # загальна яскравість та сама
    assert len(np.unique(np.round(d * 255))) > 40                              # справді багато відтінків


def test_defocus_amount_is_monotonic():
    m = _checker()
    stds = [float(eggs.defocus(m, a).std()) for a in (0.2, 0.5, 1.0)]
    assert stds[0] > stds[1] > stds[2]


def test_defocus_keeps_large_shapes_and_accepts_float_input():
    big = np.zeros((1600, 1800), np.uint8)
    big[:, 900:] = 1                                             # >2,5 Мпкс: іде шлях із половинним розміром
    d = eggs.defocus(big, 1.0)
    assert d.shape == big.shape and d.dtype == np.float32
    assert float(d[:, :100].max()) < 0.01 and float(d[:, -100:].min()) > 0.99 and 0.2 < float(d[800, 900]) < 0.8
    f = eggs.defocus(_checker().astype(np.float32), 0.7)
    assert f.dtype == np.float32


def test_defocus_result_is_accepted_by_view_compose():
    from benvid import view
    m = _checker(60, 80)
    lut = np.arange(256, dtype=np.uint8).reshape(256, 1, 1).repeat(3, 2)
    out = view.compose(eggs.defocus(m, 1.0), lut, 400, 300, None)
    assert out is not None
    img = out[0]
    assert img.dtype == np.uint8 and img.shape[2] == 3


@pytest.mark.parametrize("code", ["uk", "en"])
def test_philosophy_parts_join_back(code):
    a, b = eggs.philosophy_parts(code)
    assert a.endswith(".") and b
    assert a + " " + b == guide.PHILOSOPHY[code]


def test_philosophy_parts_follow_current_language():
    old = i18n.lang()
    try:
        i18n.set_lang("en")
        assert eggs.philosophy_parts()[0].startswith("Our whole world")
        i18n.set_lang("uk")
        assert eggs.philosophy_parts()[0].startswith("Весь наш світ")
    finally:
        i18n.set_lang(old)


def test_caption_alphas_timeline():
    T = eggs.DEFOCUS_TOTAL
    assert eggs.caption_alphas(0.0) == (0.0, 0.0)
    assert eggs.caption_alphas(T) == (0.0, 0.0)
    a1, a2 = eggs.caption_alphas(T * 0.3)
    assert a1 > 0.9 and a2 < a1                                  # перший рядок уже видно, другий ще проявляється
    a1, a2 = eggs.caption_alphas(T * 0.6)
    assert a1 == pytest.approx(1.0) and a2 == pytest.approx(1.0)  # обидва повністю
    a1, a2 = eggs.caption_alphas(T * 0.9)
    assert 0.0 < a1 < 1.0 and 0.0 < a2 < 1.0                     # гаснуть
    for i in range(101):
        assert all(0.0 <= v <= 1.0 for v in eggs.caption_alphas(T * i / 100))


def test_blend_hex():
    assert eggs.blend_hex("#000000", "#ffffff", 0) == "#000000"
    assert eggs.blend_hex("#000000", "#ffffff", 1) == "#ffffff"
    assert eggs.blend_hex("#000000", "#ffffff", 0.5) == "#808080"
    assert eggs.blend_hex("#102030", "#304050", 5) == "#304050" and eggs.blend_hex("#102030", "#304050", -3) == "#102030"


def test_spin_angles():
    a = eggs.spin_angles(14)
    assert len(a) == 14 and a[0] == 0.0 and a[-1] == pytest.approx(360.0)
    assert all(x < y for x, y in zip(a, a[1:]))                  # завжди вперед
    steps = [y - x for x, y in zip(a, a[1:])]
    assert steps[0] > steps[-1]                                  # гальмує
    assert len(eggs.spin_angles(1)) == 2                         # менше двох кадрів не буває


def test_party_order_covers_each_palette_once_and_ends_at_current():
    names = list(palette.NAMES)
    for cur in (names[0], names[5], names[-1]):
        o = eggs.party_order(names, cur)
        assert sorted(o) == sorted(names) and len(o) == len(names)
        assert o[-1] == cur
        assert o[0] == names[(names.index(cur) + 1) % len(names)]
    assert sorted(eggs.party_order(names, "нема такої")) == sorted(names)


def test_crumbs_burst_goes_up_then_falls_out():
    rng = random.Random(5)
    ps = eggs.crumbs_burst(rng, n=18)
    assert len(ps) == 18
    assert all(p[3] < 0 and p[4] in (1.0, 2.0, 3.0) for p in ps)         # спершу летять угору
    t, steps = 0.0, 0
    while ps and steps < 400:
        ps = eggs.crumbs_step(ps, 0.03, floor=17)
        steps += 1
    assert not ps and steps < 100                                        # менше ніж за ~3 с усі впали за край


def test_crumbs_step_without_floor_keeps_everything():
    ps = eggs.crumbs_burst(random.Random(2), n=5)
    for _ in range(10):
        ps = eggs.crumbs_step(ps, 0.05)
    assert len(ps) == 5
    assert all(p[1] > -30 for p in ps)                                   # гравітація повертає крихти вниз


def test_cookie_bites_fit_the_cookie():
    assert eggs.COOKIE_CLICKS == len(eggs.BITES) + 1
    for dx, dy, r in eggs.BITES:
        assert (dx * dx + dy * dy) ** 0.5 <= 12 and 3 <= r <= 8          # відкус лишається біля печива (радіус 9)
    assert eggs.EATEN > len(eggs.BITES)
