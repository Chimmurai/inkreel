"""Шостий рівень «Нейро»: ОКРЕМА ГІЛКА (навчена мережа), а не наступник «Ультри». Ядро, відео-шматки, символи, CLI й панель якості."""
import os
import sys

import cv2
import numpy as np
import pytest

pytest.importorskip("onnxruntime")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benvid import core, student  # noqa: E402
from benvid.core import BRANCH, MAIN_LEVELS, Params, binarize  # noqa: E402
from benvid.student import inputs as I  # noqa: E402
from benvid.student import runtime as R  # noqa: E402


def _img(h, w, seed=1):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur((rng.random((h, w, 3)) * 255).astype(np.uint8), (0, 0), 3)
    cv2.circle(base, (w // 2, h // 2), h // 3, (240, 240, 240), -1)
    return np.ascontiguousarray(base)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setenv("INKREEL_STUDENT", "cpu")
    monkeypatch.setattr("benvid.log.app_dir", lambda: str(tmp_path))
    R._inst = None
    R._state.update(backend=None, reason="", probing=False, ms={})
    I._noise_cache.clear()
    I._rings.clear()
    yield
    R._inst = None


# ---------------------------------------------------------------- таблиця рівнів
def test_branch_is_separate_from_the_five_level_scale():
    assert MAIN_LEVELS == 5 and BRANCH == 6
    assert sorted(core.METHODS) == [1, 2, 3, 4, 5, 6] and set(core.METHOD_INFO) == set(core.METHODS)
    assert set(core.GLYPH_INFO) == {1, 2, 3, 4, 5}, "у «Символах» лишається п'ять рівнів"
    assert "гілка" in core.METHOD_INFO[BRANCH] and "не вищий" in core.METHOD_INFO[BRANCH], "з опису ясно: це не наступник Ультри"


def test_symbols_never_get_the_branch():
    assert Params.from_dict({"method": 6, "glyph": True}).method == MAIN_LEVELS
    assert Params.from_dict({"method": 6, "glyph": False}).method == BRANCH
    img = _img(96, 160)
    a = binarize(img, 160, 96, Params(method=6, glyph=True))
    b = binarize(img, 160, 96, Params(method=5, glyph=True))
    assert np.array_equal(a, b), "символи на шостому рівні = п'ятий рівень символів"


# ---------------------------------------------------------------- ядро
def test_level6_is_the_student_mask_and_invert_flips_it():
    img = _img(120, 160)
    m = binarize(img, 160, 120, Params(method=6))
    assert m.shape == (120, 160) and m.dtype == np.uint8 and set(np.unique(m)) <= {0, 1}
    assert np.array_equal(m, student.binarize(img)), "рівень 6 = нейромережа"
    assert np.array_equal(binarize(img, 160, 120, Params(method=6, invert=True)), m ^ 1)
    assert np.array_equal(m, binarize(img, 160, 120, Params(method=6))), "детермінованість"


def test_level6_resizes_the_source_to_the_requested_size():
    img = _img(240, 320)
    m = binarize(img, 160, 96, Params(method=6))
    assert m.shape == (96, 160)


def test_film_state_travels_in_aux():
    img = _img(96, 160)
    aux = {"film": student.FilmState()}
    binarize(img, 160, 96, Params(method=6), aux)
    assert aux["film"].film is not None
    binarize(np.roll(img, 3, 1), 160, 96, Params(method=6), aux)       # другий кадр - згладжування вже працює без помилок


def test_falls_back_to_ultra_when_the_student_is_unavailable(monkeypatch):
    img = _img(120, 160)
    monkeypatch.setenv("INKREEL_STUDENT", "off")
    R._inst = None
    got = binarize(img, 160, 120, Params(method=6))
    assert np.array_equal(got, binarize(img, 160, 120, Params(method=5))), "без мережі рівень 6 рахується як Ультра"


def test_falls_back_when_the_frame_is_too_small():
    img = _img(40, 60)
    got = binarize(img, 60, 40, Params(method=6))
    assert got.shape == (40, 60)                                       # менше 48 px мережа не бере: рахує алгоритм, але картинка є


# ---------------------------------------------------------------- відео
def _clip(path, n=24, w=320, h=180, fps=24):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        f = np.dstack([(xx * 255 // w + i * 2) % 256, (yy * 255 // h), (xx + yy + i * 3) % 256]).astype(np.uint8)
        cv2.circle(f, (30 + (i * 6) % (w - 60), h // 2), 40, (240, 240, 240), -1)
        vw.write(f)
    vw.release()


def _masks(path, chunk, size=(320, 176)):
    from benvid import video
    video._FILM_CONT.clear()
    out = []
    for packed in video.packed_video(path, Params(method=6), size, chunk=chunk):
        out.append(np.unpackbits(packed, axis=-1, count=size[0]))
    return np.concatenate(out)


def test_video_chunks_continue_the_smoothing_exactly(tmp_path):
    """шматок поруч із попереднім продовжує стан згладжування, а не розігрівається заново: результат не залежить від нарізки"""
    path = str(tmp_path / "v.mp4")
    _clip(path)
    whole = _masks(path, chunk=100)
    assert whole.shape == (24, 176, 320)
    assert np.array_equal(_masks(path, chunk=5), whole), "нарізка на шматки змінила картинку"


def test_video_matches_a_direct_pass_with_one_film_state(tmp_path):
    path = str(tmp_path / "v.mp4")
    _clip(path, n=10)
    cap = cv2.VideoCapture(path)
    st, direct = student.FilmState(), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        direct.append(binarize(f, 320, 176, Params(method=6), {"film": st}))
    cap.release()
    assert np.array_equal(np.stack(direct), _masks(path, chunk=4)[:len(direct)])


def test_video_cold_chunk_warms_up_from_preceding_frames(tmp_path):
    """перемотка: шматок не з початку відео і без продовження прогрівається кадрами перед собою, тож близький до безперервного проходу"""
    from benvid import video
    path = str(tmp_path / "v.mp4")
    _clip(path, n=30)
    whole = _masks(path, chunk=100)
    video._FILM_CONT.clear()
    got = np.concatenate([np.unpackbits(p, axis=-1, count=320) for p in video.packed_video(path, Params(method=6), (320, 176), start=20, chunk=100)])
    assert got.shape[0] == 10
    diff = float((got != whole[20:]).mean())
    assert diff < 0.01, f"розігрів не зійшовся з безперервним проходом: {diff:.4f}"


def test_gif_export_runs_without_a_pool(tmp_path):
    from PIL import Image
    from benvid import media
    frames = [Image.fromarray(np.roll(_img(96, 128), i * 6, 1)) for i in range(4)]
    src, out = str(tmp_path / "a.gif"), str(tmp_path / "b.gif")
    frames[0].save(src, save_all=True, append_images=frames[1:], duration=80, loop=0)
    r = media.export_gif_source(src, out, Params(method=6), "GIF", "Як у джерела", workers=4)
    assert r["frames"] == 4 and os.path.isfile(out)


# ---------------------------------------------------------------- програвач
def test_player_uses_a_single_inprocess_lane_for_level6():
    from benvid.player import Player
    pl = Player(workers=4)
    try:
        pl.size = (1920, 1080)
        pl.params = Params(method=6)
        assert pl._eff() == 1 and not pl._ink_pool(), "нейро не йде в пул процесів: модель одна, на відеокарті"
        pl.params = Params(method=4)
        assert pl._eff() >= 1
    finally:
        pl.close()


# ---------------------------------------------------------------- CLI
def test_cli_level6_not_with_symbols(tmp_path, capsys):
    from benvid.cli import main
    a, o = str(tmp_path / "a.png"), str(tmp_path / "o.png")
    cv2.imwrite(a, _img(120, 200))
    with pytest.raises(SystemExit):
        main([a, o, "--method", "6", "--symbols"])
    assert "6" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([a, o, "--method", "7"])
    main([a, o, "--method", "6", "--size", "144p"])
    out = cv2.imread(o, 0)
    assert out.shape[0] == 144 and set(np.unique(out)) <= {0, 255}


# ---------------------------------------------------------------- панель якості
def test_quality_bar_has_a_branch_marker_off_the_scale(app):
    from conftest import settle
    from types import SimpleNamespace
    from benvid.core import METHOD_INFO
    from benvid.i18n import tr
    settle(app, rounds=3)
    bar = app.m_slider
    xs, bx = bar._xs(), bar._bx()
    assert bar.n == 5 and len(xs) == 5 and bar.branch == "IRNM1"
    assert bx > xs[-1] + 30, "ромб стоїть окремо від п'яти гуртків, з розривом"
    assert bar._at(xs[0]) == 1 and bar._at(xs[-1]) == 5 and bar._at(bx) == 6
    bar._click(SimpleNamespace(x=bx))
    assert app.params.method == BRANCH and bar.v == 6
    title = app.q_title.cget("text")
    assert "IRNM1" in title and "/5" not in title, "гілка не має місця «n із 5»"
    assert app.m_info.cget("text") == tr(METHOD_INFO[6])
    bar._click(SimpleNamespace(x=xs[3]))
    assert app.params.method == 4 and "4/5" in app.q_title.cget("text")


def test_symbols_switch_the_branch_off_and_bring_it_back(app):
    from conftest import settle
    from types import SimpleNamespace
    bar = app.m_slider
    settle(app, rounds=2)
    app._apply_method(BRANCH)
    assert app.params.method == BRANCH
    app._set_glyph(True)
    assert app.params.glyph and app.params.method == MAIN_LEVELS and not bar.branch_on
    assert bar._at(bar._bx()) <= 5, "у символах ромб не клікається"
    assert "5/5" in app.q_title.cget("text")
    app._set_glyph(False)
    assert not app.params.glyph and app.params.method == BRANCH and bar.branch_on and bar.v == BRANCH
    app._apply_method(4)
    app._set_glyph(True)
    app._set_glyph(False)
    assert app.params.method == 4, "обраний у символах рівень лишається, гілка не вигулькує сама"
    bar._click(SimpleNamespace(x=bar._xs()[3]))


def test_branch_unavailable_note_when_no_model(app, monkeypatch):
    from conftest import settle
    monkeypatch.setenv("INKREEL_STUDENT", "off")
    settle(app, rounds=2)
    app._apply_method(4)
    app._apply_method(BRANCH)
    assert "Гравюра" in app.prog_lbl.cget("text")
    app._apply_method(4)
