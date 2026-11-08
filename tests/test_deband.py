"""«Ультра»: прибирання східців кодека (deband), обмеження розтягнення кольорів (PCA_CAP) і режим рівних ділянок у _text_snap.

Кодек (H.264/JPEG) на плавних ділянках лишає плато по ~16 px, що різняться на 1-6 рівнів. Тонмап їх підсилює, і градієнт виходив
мозаїкою прямокутників різної густини крапок. Усе нове діє лише від DEBAND['h0'] (540p) і на рівнях 4-5;
144p-480p лишаються як були (так просив користувач: чіткість тексту на малих розмірах не чіпати)."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid import core as C  # noqa: E402
from benvid.core import InkState, Params, binarize  # noqa: E402

W, H = 1280, 720


def _terraced(w=W, h=H, amp=1.0, seed=0, blk=16):
    """плавна темна «віньєтка» 14..40, яку кодек замінив плато блоку blk x blk із випадковим зсувом DC (±amp рівня)"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = 14.0 + 26.0 * np.clip(np.hypot((xx - w * 0.35) / (w * 0.7), (yy - h * 0.4) / (h * 0.8)), 0, 1)
    bm = g.reshape(h // blk, blk, w // blk, blk).mean((1, 3))
    bm = np.round(bm + amp * np.random.default_rng(seed).uniform(-1.0, 1.0, bm.shape))
    return np.repeat(np.repeat(bm, blk, 0), blk, 1).astype(np.float32)


def _step(L, blk=16):
    """середній модуль перепаду через межі блоків"""
    a, b = L[:, blk::blk], L[:, blk - 1:-1:blk][:, :L[:, blk::blk].shape[1]]
    c, d = L[blk::blk, :], L[blk - 1:-1:blk, :][:L[blk::blk, :].shape[0], :]
    return 0.5 * (np.abs(a - b).mean() + np.abs(c - d).mean())


def _block_score(m, P=16):
    """>1: у масці видно решітку блоків кодека (перепади густини крапок частіші на межах блоків), 1.0 - решітки немає"""
    d = cv2.blur(m.astype(np.float32), (5, 5))
    gx = np.abs(np.diff(d, axis=1)).mean(0)
    gy = np.abs(np.diff(d, axis=0)).mean(1)

    def sc(g):
        idx = np.arange(len(g))
        return g[(idx % P) == (P - 1)].mean() / max(g[(idx % P) != (P - 1)].mean(), 1e-9)
    return 0.5 * (sc(gx) + sc(gy))


def _render(img, level, deband_on=True, w=W, h=H):
    old = C.DEBAND["h0"]
    if not deband_on:
        C.DEBAND["h0"] = 10 ** 6            # «як було» - вимкнено і прибирання, і обмеження кольорів, і режим рівних ділянок
    try:
        p = Params()
        p.method = level
        return binarize(img, w, h, p, {"state": InkState()})
    finally:
        C.DEBAND["h0"] = old


# ------------------------------------------------------------------ де діє
def test_only_from_540p_and_small_sizes_untouched():
    """144p-480p лишаються байт-в-байт як були: знебарвлення без обмеження кольорів, жодного прибирання, _text_snap без режиму рівних"""
    assert C.DEBAND["h0"] >= 540, "144p-480p чіпати не можна"
    assert not C.deband_active(480) and not C.deband_active(539) and C.deband_active(540) and C.deband_active(720)
    rng = np.random.default_rng(1)
    img = (rng.random((270, 480, 3)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 7)
    cv2.putText(img, "Subtitle", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
    for w, h in ((256, 144), (426, 240), (640, 360), (854, 480)):
        shrink = img.shape[1] >= w and img.shape[0] >= h
        legacy = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
        ref = C._gray_pca(legacy, 0.5).astype(np.float32)
        assert np.array_equal(C.gray_stage(img, w, h, True, deband=True), ref), (w, h)
        assert np.array_equal(C.gray_stage(img, w, h, True, deband=False), ref), (w, h)
        a, ga = C.tone_map(img, w, h, True, deband=True)
        b, gb = C.tone_map(img, w, h, True, deband=False)
        assert np.array_equal(a, b) and np.array_equal(ga, gb), (w, h)
        for level in (3, 4, 5):
            m = _render(img, level, True, w, h)
            assert np.array_equal(m, _render(img, level, False, w, h)), (w, h, level)


# ------------------------------------------------------------------ _deband
def test_terraces_flattened_and_change_is_small():
    L = _terraced(amp=2.0)
    out = C._deband(L, H)
    assert out.shape == L.shape and out.dtype == np.float32 and np.isfinite(out).all()
    assert _step(L) > 0.8 and _step(out) < 0.25 * _step(L), (_step(L), _step(out))
    assert np.abs(out - L).max() < 2.6, "зміна яскравості обмежена кількома рівнями з 255"
    assert np.array_equal(C._deband(L, H), out), "детермінізм (відео не мерехтить)"


def test_real_texture_noise_text_and_edges_untouched():
    rng = np.random.default_rng(3)
    tex = (120 + rng.normal(0, 6, (H, W))).astype(np.float32)           # справжня текстура/шум: плато немає
    assert C._deband(tex, H) is tex

    fr = _terraced(amp=2.0)
    m = np.zeros((H, W), np.uint8)
    cv2.putText(m, "Subtitle line of text", (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 3, cv2.LINE_AA)
    tx = np.where(m > 0, 235.0, fr).astype(np.float32)
    out = C._deband(tx, H)
    assert np.abs(out - tx)[m > 200].max() < 0.05, "штрихи тексту не змінюються"
    assert _step(out) < 0.4 * _step(tx), "а тло навколо - рівніше"

    ed = fr.copy()
    ed[:, 640:] += 60.0                                                  # справжній край: не розмивається
    o = C._deband(ed, H)
    assert np.median(o[:, 641] - o[:, 638]) > 59.0
    ed2 = fr.copy()
    ed2[:, 640:] += 14.0                                                 # і невеликий, але вже не «східець»
    o2 = C._deband(ed2, H)
    assert np.median(o2[:, 641] - o2[:, 638]) > 12.0


@pytest.mark.skipif(C._nb is None, reason="numba недоступна - є лише numpy-версія")
def test_numba_kernels_match_numpy_and_nan_guard():
    rng = np.random.default_rng(5)
    L = (rng.random((64, 64)) * 255).astype(np.float32)
    hp2 = (rng.random((64, 64)) * 2.0).astype(np.float32)
    hp2[0, :8] = -1e-15            # розмиття дає від'ємні «нулі»; корінь із них був би NaN
    G1, LG1, G2, LG2 = (np.empty_like(L) for _ in range(4))
    C._pl_nb(L, hp2, 0.25, 0.7, G1, LG1)
    C._pl_py(L, hp2, 0.25, 0.7, G2, LG2)
    assert np.isfinite(G1).all() and np.isfinite(LG1).all()
    assert np.allclose(G1, G2, atol=1e-5) and np.allclose(LG1, LG2, atol=1e-3)
    T = L + rng.normal(0, 3, L.shape).astype(np.float32)
    F = rng.random(L.shape).astype(np.float32)
    a, b = np.empty_like(L), np.empty_like(L)
    C._fin_nb(L, T, F, 1.5, 4.5, a)
    C._fin_py(L, T, F, 1.5, 4.5, b)
    assert np.allclose(a, b, atol=1e-3)


# ------------------------------------------------------------------ обмеження кольорів
def _flat_with_sparse_text(seed=0):
    """майже рівний кадр (титри, заставка) з рідким кольоровим текстом (<1% кадру) і шумом кодека"""
    rng = np.random.default_rng(seed)
    g = 128 + rng.normal(0, 1.2, (360, 640))
    img = np.clip(np.dstack([g + rng.normal(0, 0.6, g.shape) for _ in range(3)]), 0, 255).astype(np.uint8)
    m = np.zeros(g.shape, np.uint8)
    cv2.putText(m, "Rerorero", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 1, cv2.LINE_AA)
    img[m > 0] = (0, 0, 255)
    return img, m


def test_pca_cap_stops_noise_blowup_on_flat_frames(monkeypatch):
    img, m = _flat_with_sparse_text()
    scales = []
    if C._nb is not None:
        orig = C._pca_nb

        def spy(lab, vL, va, vb, off, scale, Lo, mix):
            scales.append(float(scale))
            return orig(lab, vL, va, vb, off, scale, Lo, mix)
        monkeypatch.setattr(C, "_pca_nb", spy)
    free = C._gray_pca(img, 0.5)
    capped = C._gray_pca(img, 0.5, C.PCA_CAP)
    if scales:
        assert scales[0] > 3 * C.PCA_CAP, "сценарій має справді роздувати проєкцію"
        assert scales[1] <= C.PCA_CAP + 1e-6
    bg = cv2.dilate(m, np.ones((9, 9), np.uint8)) == 0
    assert capped[bg].std() < 0.9 * free[bg].std(), (capped[bg].std(), free[bg].std())


def test_pca_cap_is_noop_on_normal_frames():
    """законне розтягнення проєкції більше за cap неможливе (напрям найбільшого розкиду не бідніший за світлоту): звичайні кадри ті самі"""
    frames = []
    for sd in range(4):
        rng = np.random.default_rng(sd)
        im = cv2.GaussianBlur((rng.random((360, 640, 3)) * 255).astype(np.uint8), (0, 0), 8)
        cv2.rectangle(im, (50, 50), (200, 150), (255, 255, 255), -1)
        frames.append(im)
    ramp = np.tile(np.linspace(0, 255, 640, dtype=np.float32), (360, 1))
    frames.append(np.dstack([ramp] * 3).astype(np.uint8))
    bars = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 255, 255], [0, 0, 0]], np.uint8)
    frames.append(np.repeat(np.repeat(bars[None], 360, 0), 80, 1))
    for im in frames:
        assert np.array_equal(C._gray_pca(im, 0.5), C._gray_pca(im, 0.5, C.PCA_CAP))


# ------------------------------------------------------------------ _text_snap у режимі рівних ділянок
def _snap_map(ramp=(300, 500), lo=0.30, hi=0.85):
    x = np.arange(W, dtype=np.float32)
    row = np.where(x < ramp[0], lo, np.where(x > ramp[1], hi, lo + (hi - lo) * (x - ramp[0]) / (ramp[1] - ramp[0])))
    t = np.tile(row[None, :].astype(np.float32), (H, 1))
    m = np.zeros((H, W), np.uint8)
    for i in range(3):
        cv2.putText(m, "Rerorero %d" % i, (700, 500 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 2, cv2.LINE_AA)
    cv2.putText(m, "Subtitle line of text", (700, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
    return (t * (1.0 - m.astype(np.float32) / 255.0)).astype(np.float32), m


def test_text_snap_level_keeps_gradient_and_flat_panels():
    t, m = _snap_map()
    o = C._text_snap(t, W, H, 1.0, True)
    txt = m > 0
    ring = (cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0) & ~txt
    assert o[txt].mean() < 0.25 and o[ring].mean() > 0.85, "текст чистий: літери темні, тло навколо суцільне"
    assert o[100:250, 560:640].mean() > 0.99, "світла рівна панель - суцільна"
    assert o[100:250, 20:240].mean() < 0.45, "темна рівна ділянка далеко від тексту не білішає (раніше ~0.85)"
    ramp = o[100:250, 330:470]
    prof = ramp.mean(0)
    assert (((ramp > 0.05) & (ramp < 0.95)).mean() > 0.9), "плавний градієнт не заливається суцільним"
    assert -np.diff(prof).min() < 0.02 and prof[-1] - prof[0] > 0.35, "півтони зростають плавно (у вхідному градієнті розмах 0.385)"


def test_text_snap_without_text_is_identity():
    t = np.tile(np.linspace(0.2, 0.8, W, dtype=np.float32)[None], (H, 1))
    for level in (False, True):
        assert np.array_equal(C._text_snap(t, W, H, 1.0, level), t)


# ------------------------------------------------------------------ весь конвеєр
@pytest.mark.parametrize("level", [4, 5])
def test_pipeline_removes_block_lattice(level):
    L = _terraced(amp=2.0)
    img = np.dstack([L * 1.04, L, L * 0.96]).clip(0, 255).astype(np.uint8)
    old = _render(img, level, deband_on=False)
    new = _render(img, level, deband_on=True)
    assert new.shape == (H, W) and set(np.unique(new)) <= {0, 1}
    s_old, s_new = _block_score(old), _block_score(new)
    assert s_old > 1.25, "перевірка має справді бачити решітку в старому результаті: %.3f" % s_old
    assert s_new < 1.12 and s_new < s_old - 0.15, (s_old, s_new)
    assert (new == _render(img, level, deband_on=True)).all()          # детермінізм


def test_pipeline_keeps_clean_text_over_terraces():
    """субтитри на терасованому темному тлі: літери читаються, а навколо них немає розсипу крапок"""
    L = _terraced(amp=2.0)
    img = np.dstack([L * 1.04, L, L * 0.96]).clip(0, 255).astype(np.uint8)
    cv2.putText(img, "Subtitle line of text", (160, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (245, 245, 245), 3, cv2.LINE_AA)
    txt = img[:, :, 1] > 200
    ring = (cv2.dilate(txt.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~txt
    for level in (4, 5):
        m = _render(img, level)
        assert m[txt].mean() > 0.8, (level, "літери світлі")
        assert m[ring].mean() < 0.35, (level, "навколо літер тло темне, без розсипу")
