"""Контрольний тест якості (benvid.qc) як регресійний захист: пороги взято з реальних вимірів із запасом."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid import qc  # noqa: E402
from benvid.video import find_ffmpeg  # noqa: E402

SIZE = (320, 180)


@pytest.fixture(scope="module")
def report():
    imgs = qc.synthetic_set(*SIZE)
    return imgs, qc.level_report(imgs, SIZE)


def test_tone_is_conveyed_by_density(report):
    imgs, res = report
    for name in ("градієнт", "портрет", "текстура"):
        r = res[name]
        for lv in (3, 4, 5):
            assert r[lv]["tone2"] < 0.26, (name, lv, r[lv])          # густина чорного/білого ≈ світлота джерела
        assert r[5]["tone2"] < r[1]["tone2"] - 0.10, name             # і ЗНАЧНО краще за голий поріг


def test_text_is_legible_in_both_polarities(report):
    imgs, res = report
    for name in ("текст (чорне на білому)", "текст (біле на чорному)"):
        for lv in qc.LEVELS:
            assert res[name][lv]["text"] >= 0.72, (name, lv, res[name][lv])


def test_no_repeating_tile_grid(report):
    imgs, res = report
    for name in ("градієнт", "портрет", "текстура"):
        for lv in (3, 4, 5):                                              # растр і графіка: дізер без плиток, що повторюються
            assert res[name][lv]["repeat"] < 0.06, (name, lv, res[name][lv]["repeat"])


def test_repeat_metric_catches_tiled_pattern():
    """сама метрика працює: плитка 64×64, що повторюється, дає близько 1; випадковий шум - близько 0"""
    rng = np.random.default_rng(1)
    tile = (rng.random((64, 64)) > 0.5).astype(np.uint8)
    tiled = np.tile(tile, (4, 6))
    assert qc.repeat_score(tiled) > 0.5
    assert qc.repeat_score((rng.random((256, 384)) > 0.5).astype(np.uint8)) < 0.05


def test_level5_flickers_less_than_level4():
    base = qc.synthetic_set(*SIZE)["портрет"]
    f4, f5 = qc.flicker(4, base, n=6, size=SIZE), qc.flicker(5, base, n=6, size=SIZE)
    assert f5 < 0.06 and f5 < f4 * 0.5, (f4, f5)


@pytest.mark.skipif(not find_ffmpeg(), reason="потрібен ffmpeg")
def test_export_formats_are_clean_and_complete():
    rows = qc.format_report(size=(256, 144), frames=16)
    by = {(r["fmt"], r["q"]): r for r in rows}
    assert not [r for r in rows if "error" in r], rows
    for (fmt, q), r in by.items():
        assert r["frames_ok"] and r["size"] == (256, 144), (fmt, q, r)
        if fmt in ("GIF", "PNG-послідовність", "JSON (стиснутий)", "InkReel (.ikr, без втрат)"):
            assert r["mae"] == 0.0, (fmt, r)                           # беззбиткові формати - біт у біт
        if fmt.split()[0] in ("MP4", "MKV", "MOV", "AVI", "WebM"):
            assert r["audio"] is True, (fmt, q)                         # звук переноситься
    hi, mid = by[("MP4 (H.264)", "Висока")], by[("MP4 (H.264)", "Середня")]
    assert hi["grey"] < 0.3 and hi["mae"] < 2.0, hi                     # «Висока» - майже без сірого після стиснення
    assert mid["grey"] < 1.0 and mid["mae"] < 5.0, mid
    assert hi["mae"] < mid["mae"] < by[("MP4 (H.264)", "Мала")]["mae"]  # шкала якості монотонна
