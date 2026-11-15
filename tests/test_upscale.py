"""Збільшення кадра перед обробкою: мале збільшення - чистий INTER_CUBIC, велике - спершу згладжене джерело (інакше пікселі джерела стають «кубиками»)."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from benvid import core as C  # noqa: E402


def _pixelated(h=60, w=80, seed=3):
    """кадр зі ступінчастими контурами в масштабі пікселя: плавний градієнт + різкі краї"""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur((rng.random((h, w, 3)) * 255).astype(np.uint8), (0, 0), 3)
    base[:, w // 2:] = np.clip(base[:, w // 2:].astype(np.int16) + 90, 0, 255).astype(np.uint8)
    return base


def test_sigma_is_zero_for_small_enlargements_and_grows_to_a_cap():
    assert C.up_blur_sigma(1.0) == 0.0 and C.up_blur_sigma(1.5) == 0.0
    vals = [C.up_blur_sigma(k) for k in (1.6, 2.0, 3.0, 4.0, 6.0, 12.0)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), vals
    assert vals[-1] == C.UP_BLUR_MAX and vals[-2] == C.UP_BLUR_MAX


def test_shrinking_and_equal_size_are_plain_area_resize():
    img = _pixelated(120, 160)
    for w, h in ((80, 60), (160, 120), (100, 120)):
        want = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        assert np.array_equal(C.fit_frame(img, w, h), want), (w, h)


@pytest.mark.parametrize("k", [1.2, 1.5, 2.0, 2.4])
def test_up_to_about_2x_is_bit_identical_to_plain_cubic(k):
    img = _pixelated()
    w, h = int(80 * k), int(60 * k)
    assert np.array_equal(C.fit_frame(img, w, h), cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC))


def test_large_enlargement_smooths_the_source_grid_and_keeps_the_tone():
    img = _pixelated()
    w, h = 480, 360                                               # x6
    got = C.fit_frame(img, w, h)
    plain = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    assert got.shape == plain.shape and got.dtype == np.uint8
    assert not np.array_equal(got, plain)

    def edge_energy(a):                                           # ступінчастість: середня вага різниць сусідніх пікселів
        g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())
    assert edge_energy(got) < edge_energy(plain)
    assert abs(float(got.mean()) - float(plain.mean())) < 1.0     # тон не зсувається
    assert np.array_equal(got, C.fit_frame(img, w, h))            # детерміновано


def test_gray_and_color_inputs_both_work():
    g = cv2.cvtColor(_pixelated(), cv2.COLOR_BGR2GRAY)
    out = C.fit_frame(g, 480, 360)
    assert out.shape == (360, 480) and out.dtype == np.uint8


@pytest.mark.parametrize("level", [1, 2, 3])
def test_binarize_uses_the_same_fit_for_the_source_it_is_given(level):
    """binarize(малий кадр, великий розмір) = binarize(той самий кадр, уже приведений fit_frame): рівні рахують на однаковому вході"""
    img = _pixelated(60, 80)
    w, h = 480, 360
    p = C.Params(method=level)
    a = C.binarize(img, w, h, p, None)
    b = C.binarize(C.fit_frame(img, w, h), w, h, p, None)
    assert a.shape == (h, w) and np.array_equal(a, b)
