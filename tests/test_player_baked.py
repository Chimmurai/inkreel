"""плеєр у режимі із замочком (.ikr): кадри лише читаються з файлу - без пулу, без перерахунку; розмір і налаштування не діють"""
import os
import time

import numpy as np
import pytest

from benvid import bake
from benvid.core import Params
from benvid.player import Player
from benvid.video import export_video, find_ffmpeg

from test_bake import _clip
from test_player import _wait

FMT = "InkReel (.ikr, без втрат)"


def _bake(tmp_path, n=24, audio=False, method=3):
    src = str(tmp_path / "v.mp4")
    _clip(src, n=n, audio=audio)
    ikr = str(tmp_path / "v.ikr")
    export_video(src, ikr, Params(method=method), FMT, "Як у джерела", "Висока", workers=1, audio=audio)
    return src, ikr


@pytest.mark.skipif(find_ffmpeg() is None, reason="потрібен ffmpeg")
def test_baked_player_reads_frames_without_pool(tmp_path):
    src, ikr = _bake(tmp_path)
    rd = bake.open_baked(ikr)
    p = Player(workers=2)
    try:
        info = p.open(ikr, "144p")                      # розмір з аргументу ігнорується
        assert p.baked and info["kind"] == "video" and info["frames"] == 24
        assert p.size == (320, 180) and p.size_key == "Як у джерела"
        assert p.baked_meta["src"]["name"] == "v.mp4" and p.baked_meta["params"]["method"] == 3
        assert _wait(p, lambda: p.cached >= 24)
        assert p._ex is None                            # пул процесів не створювався
        psize, snap = p.snapshot()
        assert psize == (320, 180) and sorted(snap) == list(range(24))
        assert all(np.array_equal(snap[i], rd.packed(i)) for i in range(24))       # кадри в буфері - рівно ті, що у файлі
        for i in (0, 7, 23, 12):                                                   # перемотка віддає саме той кадр
            p.seek(i)
            got = None
            end = time.time() + 10
            while got is None and time.time() < end:
                got = p.update()
                time.sleep(0.01)
            assert got is not None and got[0] == i and np.array_equal(got[1] > 0, rd.mask(i) > 0)
    finally:
        p.close()


@pytest.mark.skipif(find_ffmpeg() is None, reason="потрібен ffmpeg")
def test_baked_player_ignores_size_and_params(tmp_path):
    src, ikr = _bake(tmp_path)
    p = Player(workers=2)
    try:
        p.open(ikr)
        assert _wait(p, lambda: p.cached >= 24)
        gen, have = p._gen, p.cached
        p.set_size("144p")
        p.set_params(Params(method=1))
        assert p.size == (320, 180) and p.size_key == "Як у джерела"
        assert p._gen == gen and p.cached == have       # буфер не скидався, перерахунку нема
        assert p.params.method == 1                     # налаштування лише запам'ятовуються (для повернення до звичайного режиму)
        assert p.auto_pick() == "Як у джерела"
    finally:
        p.close()


@pytest.mark.skipif(find_ffmpeg() is None, reason="потрібен ffmpeg")
def test_baked_player_loads_embedded_audio(tmp_path):
    src, ikr = _bake(tmp_path, audio=True)
    p = Player(workers=1)
    try:
        p.open(ikr)
        end = time.time() + 30
        while time.time() < end and p.audio.pcm is None:
            time.sleep(0.05)
        assert p.audio.pcm is not None and len(p.audio.pcm) > 1000
    finally:
        p.close()


@pytest.mark.skipif(find_ffmpeg() is None, reason="потрібен ffmpeg")
def test_open_normal_after_baked_restores_normal_mode(tmp_path):
    src, ikr = _bake(tmp_path)
    p = Player(workers=2)
    try:
        p.open(ikr)
        assert p.baked
        p.open(src, "144p")
        assert not p.baked and p.baked_meta == {} and p.size_key == "144p"
        assert _wait(p, lambda: p.buffered_end() >= 10)
    finally:
        p.close()


# ---------------------------------------------------------------- звичайний режим із готових кадрів (розморозка)
@pytest.mark.skipif(find_ffmpeg() is None, reason="потрібен ffmpeg")
def test_open_original_with_seed_takes_frames_from_the_ikr_and_starts_at_the_position(tmp_path):
    """розморозка: буфер стартує там, де зупинились, а кадри беруться з .ikr - нічого не рахується (пул не створюється)"""
    src, ikr = _bake(tmp_path, n=40)
    rd = bake.open_baked(ikr)
    p = Player(workers=2)
    try:
        p.set_params(Params(method=3))
        info = p.open(src, "Як у джерела", start=25, seed=dict(path=ikr, base=0))
        assert not p.baked and p._seed is not None and p._head == 25 and int(p.position()) == 25
        assert _wait(p, lambda: p.cached >= 40)
        assert p._ex is None, "кадри мали прийти з файлу, а не з пулу"
        _, snap = p.snapshot()
        assert sorted(snap) == list(range(40)) and all(np.array_equal(snap[i], rd.packed(i)) for i in range(40))
        assert info["frames"] == 40
    finally:
        p.close()


@pytest.mark.skipif(find_ffmpeg() is None, reason="потрібен ffmpeg")
def test_seed_is_ignored_when_size_or_fps_limit_differ(tmp_path):
    src, ikr = _bake(tmp_path, n=12)
    p = Player(workers=1)
    try:
        p.open(src, "144p", seed=dict(path=ikr, base=0))             # розмір інший, ніж у файлі
        assert p._seed is None
        p.set_fps_limit(6)
        p.open(src, "Як у джерела", seed=dict(path=ikr, base=0))     # ліміт к/с: у буфері мають бути лише потрібні кадри
        assert p._seed is None
        p.set_fps_limit(None)
        p.open(src, "Як у джерела", seed=dict(path=str(tmp_path / "нема.ikr"), base=0))
        assert p._seed is None
    finally:
        p.close()


@pytest.mark.skipif(find_ffmpeg() is None, reason="потрібен ffmpeg")
def test_broken_seed_falls_back_to_computing(tmp_path):
    src, ikr = _bake(tmp_path, n=12)
    p = Player(workers=1)
    try:
        p.set_params(Params(method=3))
        p.open(src, "Як у джерела", seed=dict(path=ikr, base=0))
        assert p._seed is not None
        os.remove(ikr)                                               # файл зник до того, як кадри прочитано
        assert _wait(p, lambda: p.cached >= 12, 60)                  # далі кадри рахуються звичайним шляхом
        assert p._seed is None
    finally:
        p.close()


@pytest.mark.skipif(find_ffmpeg() is None, reason="потрібен ffmpeg")
def test_changing_params_or_size_after_unfreeze_drops_the_seed_and_recomputes(tmp_path):
    """після «Звичайного режиму» зміна налаштувань чи розміру мусить рахувати наново, а не віддавати кадри зі старого .ikr"""
    src, ikr = _bake(tmp_path, n=24)
    p = Player(workers=2)
    try:
        p.set_params(Params(method=3))
        p.open(src, "Як у джерела", seed=dict(path=ikr, base=0))
        assert p._seed is not None
        p.set_params(Params(method=3))                               # ті самі налаштування: файл лишається джерелом кадрів
        assert p._seed is not None
        p.set_params(Params(method=1))
        assert p._seed is None
        assert _wait(p, lambda: p.cached >= 24)
        rd = bake.open_baked(ikr)
        _, snap = p.snapshot()
        assert any(not np.array_equal(snap[i], rd.packed(i)) for i in range(24))        # кадри справді інші, ніж у файлі
    finally:
        p.close()
    p = Player(workers=2)
    try:
        p.set_params(Params(method=3))
        p.open(src, "Як у джерела", seed=dict(path=ikr, base=0))
        assert p._seed is not None
        p.set_size("144p")
        assert p._seed is None and p.size != (320, 180)
    finally:
        p.close()


def test_ink_pool_is_released_when_switching_to_neuro(tmp_path, monkeypatch):
    """пул процесів Ink/Ultra не лишається простоювати (з пам'яттю й контекстами відеокарти), коли рахує «Нейро»"""
    from benvid import player as pl
    killed = []
    monkeypatch.setattr(pl, "kill_pool", lambda ex, wait=0.0: killed.append(ex))
    p = Player.__new__(Player)
    p.__dict__.update(_gen=0, _ex=object(), _ink_ready=True, _ink_warm=object(), baked=False, params=Params(method=6), n=3,
                      _head=0, _shown=0, _urgent=False, _stride=1, _frames={}, _have=None, _pend=None)
    p._cancel_inflight = lambda: None
    p._gap_of = lambda s: 0
    p._restart(0)
    assert len(killed) == 1 and p._ex is None and p._ink_ready is False
    p._ex = object()
    p.params = Params(method=3)
    p._restart(0)
    assert len(killed) == 1 and p._ex is not None                    # у режимі Ink пул лишається
