import os
import time

import cv2
import numpy as np

from benvid.player import Player


def _clip(path, n=30, fps=24, w=160, h=96):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), 40, np.uint8)
        cv2.circle(f, (10 + i * 4, h // 2), 18, (230, 230, 230), -1)
        vw.write(f)
    vw.release()


def _wait(p, cond, t=60):
    end = time.time() + t
    while time.time() < end:
        p.update()
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_player_basic(tmp_path):
    path = str(tmp_path / "a.mp4")
    _clip(path)
    p = Player(workers=2)
    try:
        info = p.open(path, "144p")
        assert info["frames"] >= 28
        assert _wait(p, lambda: p.buffered_end() >= 10)
        p.seek(5)
        assert _wait(p, lambda: p.buffered_end() > 6)
        key = p.auto_pick()
        assert key.endswith("p")
        psize, snap = p.snapshot()
        assert psize == p.size
        p.set_size("144p")
        assert p.stride == 1
    finally:
        p.close()


def test_suspend_releases_pool_and_resumes(tmp_path):
    path = str(tmp_path / "b.mp4")
    _clip(path, n=40)
    p = Player(workers=2)
    try:
        p.open(path, "144p")
        assert _wait(p, lambda: p.cached >= 10)
        p.suspend(True)
        assert p._ex is None
        p.suspend(False)
        p.seek(30)
        assert _wait(p, lambda: p.buffered_end() >= 35)
        assert p._ex is not None
    finally:
        p.close()


def test_animated_gif_survives_auto_pick(tmp_path):
    """auto_pick() клонує джерело й закриває клон: у GIF/APNG/WebP це не має спустошувати спільні кадри
    (раніше після відкриття анімації плеєр вічно «буферизував»)"""
    from PIL import Image
    fr = [Image.fromarray(np.full((60, 80, 3), 30 + i * 40, np.uint8)) for i in range(5)]
    path = str(tmp_path / "a.gif")
    fr[0].save(path, save_all=True, append_images=fr[1:], duration=100, loop=0)
    p = Player(workers=1)
    try:
        info = p.open(path, "144p")
        assert info["kind"] == "gif" and info["frames"] == 5
        p.auto_pick()
        assert p._src.read(2) is not None
        assert _wait(p, lambda: p.cached == 5, t=20)
        p.play()
        assert _wait(p, lambda: p.state == "playing", t=20)
        assert not p._fails
    finally:
        p.close()


def _big_clip(path, n=30, fps=24, w=480, h=320):
    """кадри від 290 px по висоті: «Ультра» малює їх основним (великим) шляхом"""
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        f = np.dstack([(xx * 255 // w + i * 3) % 256, (yy * 255 // h), (xx + yy + i * 5) % 256]).astype(np.uint8)
        cv2.circle(f, (40 + i * 6, h // 2), 50, (240, 240, 240), -1)
        vw.write(f)
    vw.release()


def test_job_step_and_skip_read_the_right_frames(tmp_path):
    """шматок із кроком (слабкий ПК рахує кожен N-й кадр) і близькі/далекі переходи між шматками беруть саме ті кадри"""
    from benvid.core import Params, binarize
    from benvid.video import _CAPS, _job
    path = str(tmp_path / "s.mp4")
    _clip(path, n=60)
    cap = cv2.VideoCapture(path)
    allf = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        allf.append(f)
    cap.release()
    w, h = 160, 96
    p = Params(method=1)
    pd = p.to_dict()
    pk = lambda i: np.packbits(binarize(allf[i], w, h, p), axis=-1)
    _CAPS.clear()
    for start, count, step in ((3, 4, 2), (11, 3, 2), (17, 5, 3), (40, 4, 1), (44, 3, 1), (2, 5, 1), (50, 4, 4)):
        got = _job((path, start, count, w, h, pd, step))
        idx = [start + k * step for k in range(count)]
        idx = [i for i in idx if i < len(allf)]
        assert len(got) == len(idx), (start, count, step)
        for g, i in zip(got, idx):
            assert np.array_equal(g, pk(i)), (start, step, i)
    for e in _CAPS.values():
        e[0].release()
    _CAPS.clear()


def test_ultra_video_runs_in_pool_and_matches_sequential_chunks(tmp_path):
    """«Ультра» відео рахується в процесах пулу (нижчий пріоритет: вікно не тормозить, працюють усі ядра), а кадри такі самі, як у
    послідовному розрахунку шматка з розігрівом; поки йде розрахунок, головний потік не підвисає"""
    from benvid.core import Params
    from benvid.video import _CAPS, _job
    path = str(tmp_path / "u.mp4")
    _big_clip(path, n=24)
    p = Player(workers=2)
    try:
        p.open(path, "Як у джерела")
        p.set_params(Params(method=5))
        assert p._ink_pool() and p._eff() > 1
        gaps, t0, last = [], time.time(), time.perf_counter()
        while time.time() - t0 < 240 and p.cached < p.n:
            time.sleep(0.001)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now
        assert p.cached == p.n, f"порахувано {p.cached} з {p.n}"
        assert p._ink_ready and p._inflight == 0 and not p._futs
        assert max(gaps) < 1.0, f"головний потік підвис на {max(gaps):.2f} с"
        w, h = p.size
        pd = Params(method=5).to_dict()
        ch = p._chunk5()
        ref = np.concatenate([_job((path, a, min(ch, p.n - a), w, h, pd)) for a in range(0, p.n, ch)])
        _CAPS.clear()
        for i in range(p.n):
            assert np.array_equal(p._frames[i], ref[i]), f"кадр {i} відрізняється від еталона"
    finally:
        p.close()


def test_ultra_seek_drops_stale_chunks_and_buffers_from_the_new_place(tmp_path):
    from benvid.core import Params
    path = str(tmp_path / "v.mp4")
    _big_clip(path, n=120)
    p = Player(workers=2)
    try:
        p.open(path, "Як у джерела")
        p.set_params(Params(method=5))
        assert _wait(p, lambda: p._inflight > 0, t=120)           # шматки пішли в пул
        p.seek(100)
        assert _wait(p, lambda: p.buffered_end() >= 105, t=180)
        assert bool(p._have[100])
        assert _wait(p, lambda: p._inflight == 0 or p.cached >= p.n, t=240)
        assert not any(f for f in p._futs if not f.done())
    finally:
        p.close()


def test_slots_are_returned_when_the_pool_refuses_a_frame(tmp_path):
    """пул закрили чи зламали, і кадр не вдалося подати: слот кільця не лишається зайнятим (раніше кілька таких збоїв вичерпували
    кільце, і плеєр навіки «буферизував»), кадр повертається в чергу й таки рахується"""
    path = str(tmp_path / "c.mp4")
    _clip(path, n=40)
    p = Player(workers=2)
    try:
        real, calls = p._pool, {"n": 0}

        class Dead:
            def submit(self, *a, **k):
                calls["n"] += 1
                raise RuntimeError("cannot schedule new futures after shutdown")

            def shutdown(self, *a, **k):
                pass

        p._pool = lambda: Dead() if calls["n"] < 8 else real()
        p.open(path, "144p")
        assert _wait(p, lambda: p.cached >= 30, t=120), f"порахувано лише {p.cached}"
        assert calls["n"] >= 8
        ring = p._ring
        assert _wait(p, lambda: ring.out == 0 and len(ring.free) == ring.n, t=30), (ring.out, len(ring.free), ring.n)
    finally:
        p.close()


def test_warm_does_not_block_the_caller(tmp_path):
    """warm() викликає інтерфейс при старті: запуск процесів іде у фоні й не тримає замок, потрібний інтерфейсу щомиті"""
    p = Player(workers=2)
    try:
        t = time.perf_counter()
        p.warm()
        assert time.perf_counter() - t < 0.25
        assert _wait(p, lambda: p._ex is not None, t=30)
        t = time.perf_counter()
        for _ in range(50):                                     # замок вільний: звичайні виклики інтерфейсу миттєві
            p.position()
            p.update()
        assert time.perf_counter() - t < 0.5
    finally:
        p.close()


def test_pool_ramps_up_gradually():
    """новий пул не вмикає всі процеси одразу: спершу RAMP0, далі +1 кожні RAMP_STEP с (старт 27 процесів разом душив систему)"""
    from benvid import player as pm
    p = Player(workers=2)
    try:
        p._ex_n = 15
        p._ex_t0 = time.perf_counter()
        assert p._conc() == pm.RAMP0
        p._ex_t0 -= pm.RAMP_STEP * 5 + 0.05
        assert p._conc() == pm.RAMP0 + 5
        p._ex_t0 -= 1000
        assert p._conc() == 15
        p.ramp = False
        p._ex_t0 = time.perf_counter()
        assert p._conc() == 15
        p.ramp, p._ex_n = True, 3
        assert p._conc() == 3
    finally:
        p.close()


def test_first_frame_after_open_and_seek_is_urgent(tmp_path):
    """«Ультра»: кадр у поточній позиції після відкриття/перемотки йде окремим завданням без розігріву (на 4K шматок віддається лише
    повністю - перший кадр чекав десятки секунд); далі - звичайні шматки"""
    from benvid.core import Params
    path = str(tmp_path / "u.mp4")
    _clip(path, n=60)
    p = Player(workers=2)
    try:
        p._alive = False                      # подавачі виходять: _claim викликаємо вручну, без гонки
        time.sleep(0.3)
        p.params = Params(method=5)
        p.open(path, "144p")
        p._ink_ready = True
        a = p._claim()
        assert a is not None and a[1] == [0] and a[-1] is True
        b = p._claim()
        assert b is not None and b[1][0] == 1 and len(b[1]) > 1 and b[-1] is False
        p.seek(40)
        c = p._claim()
        assert c is not None and c[1] == [40] and c[-1] is True
    finally:
        p.close()


def test_job_cold_skips_warmup(tmp_path):
    """cold-завдання не читає кадри розігріву й дає ті самі розміри"""
    from benvid import video
    from benvid.core import Params
    path = str(tmp_path / "c.mp4")
    _clip(path, n=30)
    pd = Params(method=5).to_dict()
    r = video._job((path, 15, 1, 160, 96, pd, 1, True))
    assert r.shape[0] == 1 and r.shape[1] == 96


def test_unsafe_numba_layer_falls_back_to_process_pool(monkeypatch):
    """господар (потоки) вимагає шару numba, що витримує паралельні виклики (omp/tbb); інакше - пул процесів"""
    from concurrent.futures import Future
    monkeypatch.delenv("INKREEL_HOST", raising=False)
    p = Player(workers=2)
    try:
        p._ex = type("H", (), {"is_host": True})()
        called = []
        monkeypatch.setattr(p, "reset_pool", lambda: called.append(1))
        fu = Future()
        fu.set_result("workqueue")
        p._ink_warmed(fu)
        assert _wait(p, lambda: called, 5) and os.environ.get("INKREEL_HOST") == "0" and not p._ink_ready
        monkeypatch.delenv("INKREEL_HOST", raising=False)
        fu2 = Future()
        fu2.set_result("omp")
        p._ink_warmed(fu2)
        assert p._ink_ready
    finally:
        os.environ.pop("INKREEL_HOST", None)
        p._ex = None
        p.close()


def test_sparse_mode_turns_off_when_buffer_is_full(tmp_path):
    """режим «кожен N-й»: коли буфер за цим кроком повний, плеєр докладає пропущені кадри й вимикає режим сам"""
    path = str(tmp_path / "s.mp4")
    _clip(path, n=40)
    p = Player(workers=2)
    try:
        p.open(path, "144p")
        assert _wait(p, lambda: p.cached >= 5)
        with p._lock:                                    # ніби ПК не встигав: рахуємо кожен 4-й кадр
            p._stride = p._gap = 4
            p._stalls, p._start_s = 3, 9.0
            p._restart(0)
        assert _wait(p, lambda: p.cached == p.n, t=60)   # спершу кожен 4-й, далі крок 2, далі 1 - до повного буфера
        assert _wait(p, lambda: p._gap == 1 and p.stride == 1, t=20)
        assert p._stride == 1 and p._stalls == 0
        with p._lock:
            assert all(p._have[:p.n]) and p._key(7) == 7
    finally:
        p.close()


def test_sparse_key_shows_nearest_previous_frame(tmp_path):
    path = str(tmp_path / "k.mp4")
    _clip(path, n=12)
    p = Player(workers=1)
    try:
        p.open(path, "144p")
        with p._lock:
            p._have[:] = False
            p._have[[0, 3, 6]] = True
            p._gap = 3
        assert [p._key(i) for i in range(8)] == [0, 0, 0, 3, 3, 3, 6, 6]
        assert list(p._keys(np.arange(8))) == [0, 0, 0, 3, 3, 3, 6, 6]
        assert p.stride == 3
        with p._lock:
            p._have[:] = True
        assert p.stride == 1
    finally:
        p.close()


def test_fps_limit_renders_only_the_frames_the_video_would_have(tmp_path):
    """ліміт кадрів на секунду діє й на буфер перегляду: рахуються лише кадри int(k*ratio + 0.5) (ті самі, що бере експорт), решта
    не рахується; на екрані тримається останній порахований, а позиція й тривалість лишаються у кадрах файлу"""
    path = str(tmp_path / "l.mp4")
    _clip(path, n=60, fps=60)
    p = Player(workers=2)
    try:
        p.open(path, "144p")
        n = p.n
        p.set_fps_limit(24)
        ratio = 60 / 24
        want = sorted({min(n - 1, int(k * ratio + 0.5)) for k in range(int(round(n / ratio)))})
        assert p._ratio == ratio and p._gb == 3 and list(np.flatnonzero(p._want)) == want
        assert _wait(p, lambda: p.cached_in(0, n) == len(want), t=60)
        with p._lock:
            have = list(np.flatnonzero(p._have[:n]))
            assert have == want, "рахуються лише потрібні кадри"
            assert p._gap == 3 and p.total_in(0, n) == len(want)
            assert p._key(1) == 0 and p._key(4) == 3 and p._key(5) == 5
        assert _wait(p, lambda: p.stride == 1)          # проміжки від ліміту - не «кожен N-й»
        assert _wait(p, lambda: p._gap == 3 and p._head >= n, t=20)
        p.set_fps_limit(None)                           # ліміт знято: докладаються решта кадрів
        assert p._want is None and p._gb == 1
        assert _wait(p, lambda: p.cached == n, t=60)
        assert _wait(p, lambda: p._gap == 1 and p.stride == 1, t=20)
    finally:
        p.close()


def test_fps_limit_ignored_when_not_lower_than_the_file(tmp_path):
    path = str(tmp_path / "m.mp4")
    _clip(path, n=20, fps=24)
    p = Player(workers=1)
    try:
        p.open(path, "144p")
        p.set_fps_limit(30)                             # вищий за початковий - нічого не міняє
        assert p._want is None and p._ratio == 1.0 and p._gb == 1
        p.set_fps_limit(24)
        assert p._want is None
    finally:
        p.close()


def test_fps_limit_claims_in_output_order_with_warmup_list(tmp_path):
    """шматки йдуть по потрібних кадрах; кадри розігріву перед шматком - попередні потрібні (проміжки нерівні), а не кожен N-й файлу"""
    from benvid.core import Params
    path = str(tmp_path / "c.mp4")
    _clip(path, n=60, fps=60)
    p = Player(workers=2)
    try:
        p._alive = False
        time.sleep(0.3)
        p.params = Params(method=5)
        p.open(path, "144p")
        p._ink_ready = True
        p.set_fps_limit(24)
        a = p._claim()
        assert a is not None and a[1] == [0] and a[-1] is True and a[-2] == []
        b = p._claim()
        want = list(np.flatnonzero(p._want))
        assert b[1] == want[1:1 + len(b[1])] and b[-2] == [0] and len(b[1]) > 1
        p.seek(31)
        c = p._claim()
        assert c[1] == [30] and c[-1] is True, "показується останній потрібний кадр не пізніше позиції"
        assert c[-2] == [i for i in want if i < 30][-40:]
        # крок «наперед» по вихідному ряду
        p.step(1)
        assert int(p.position()) == want[want.index(30) + 1]
    finally:
        p.close()


def test_job_with_explicit_frames_reads_exactly_those(tmp_path):
    """_job з явним списком кадрів (плеєр з лімітом к/с): кадри читаються саме ці, розігрів - явний список"""
    from benvid import video
    from benvid.core import Params
    path = str(tmp_path / "e.mp4")
    _clip(path, n=40, fps=60)
    pd = Params(method=5).to_dict()
    srcs = [3, 5, 8, 10]
    a = video._job((path, 3, len(srcs), 160, 96, pd, 1, False, 1.0, srcs, [0, 1, 2]))
    assert a.shape[0] == 4
    b = video._job((path, 3, len(srcs), 160, 96, pd, 1, True, 1.0, srcs, []))
    assert b.shape == a.shape


def test_ultra_memory_carries_over_to_the_next_chunk_in_a_thread(tmp_path, monkeypatch):
    """без пулу (слабкий ПК, один подавач) шматок «Ультри» й «Символів», що йде одразу за попереднім, продовжує його часову пам'ять,
    а не розігріває її наново (розігрів кожного шматка забирав третину роботи): розігрівається лише перший шматок після перемотки"""
    from benvid import ink5
    from benvid.core import Params
    path = str(tmp_path / "t.mp4")
    _clip(path, n=80)
    calls = {"warm": 0, "prime": 0}
    ow, op = ink5.warm, ink5.prime

    def warm(*a, **k):
        calls["warm"] += 1
        return ow(*a, **k)

    def prime(*a, **k):
        calls["prime"] += 1
        return op(*a, **k)
    monkeypatch.setattr(ink5, "warm", warm)
    monkeypatch.setattr(ink5, "prime", prime)
    p = Player(workers=1)
    try:
        p.params = Params(method=5)
        p.open(path, "144p")
        p.seek(30)
        assert _wait(p, lambda: p.cached_in(30, p.n) >= p.n - 30, t=240)
        total = calls["warm"] + calls["prime"]
        assert 0 < total <= ink5.WARM + 12, f"розігрів лише один раз ({total} викликів), а не для кожного з трьох шматків"
    finally:
        p.close()


def test_ink_chunk_frames_arrive_while_the_chunk_is_still_running(tmp_path):
    """пул, «Ультра»: готові кадри шматка лягають у буфер по одному (спільна пам'ять), а не всі разом у кінці шматка - на слабкому ПК
    перший кадр шматка з'являється одразу, а не за пів хвилини. Скасований шматок зупиняється між кадрами"""
    from benvid import video
    from benvid.core import Params
    from benvid.player import _Stream
    path = str(tmp_path / "s.mp4")
    _clip(path, n=40)
    w, h = 160, 96
    st = _Stream(1, list(range(10, 18)), w, h)
    try:
        pd = Params(method=5).to_dict()
        res = video._job((path, 10, 8, w, h, pd, 1, False, 1.0, None, None, st.name))
        assert len(res) == 8 and st.ready() == 8, "лічильник дорівнює числу кадрів"
        ref = video._job((path, 10, 8, w, h, pd, 1, False))
        assert all(np.array_equal(st.frame(k), ref[k]) for k in range(8)), "кадри з потоку ті самі, що й у звичайному результаті"
    finally:
        st.destroy()
    st = _Stream(1, list(range(10, 18)), w, h)
    try:
        st.cancel()
        res = video._job((path, 10, 8, w, h, Params(method=5).to_dict(), 1, False, 1.0, None, None, st.name))
        assert len(res) == 0 and st.ready() == 0, "скасований шматок не рахує нічого"
    finally:
        st.destroy()


def test_pool_chunks_stream_into_the_player_buffer(tmp_path):
    from benvid.core import Params
    path = str(tmp_path / "pl.mp4")
    _clip(path, n=50)
    p = Player(workers=2)
    try:
        p.params = Params(method=5)
        p.open(path, "144p")
        assert _wait(p, lambda: p.cached == p.n, t=180)
        with p._lock:
            assert all(p._have[:p.n]) and not p._streams, "усе забрано, спільна пам'ять звільнена"
        _, snap = p.snapshot()
        ref = None
        from benvid import video
        ref = video._job((path, 0, 6, p.size[0], p.size[1], p.params.to_dict(), 1, False))
        assert np.array_equal(snap[0], ref[0])
    finally:
        p.close()


def test_watchdog_logs_thread_stacks_when_playback_stalls(caplog):
    """відтворення стоїть, а в журналі пусто - не діагноз: сторож один раз пише, де застрягли потоки"""
    import logging
    p = Player(workers=1)
    try:
        caplog.set_level(logging.WARNING, logger="inkreel")
        p.info, p._src = {"kind": "video"}, object()
        p._want_play, p._state = True, "buffering"
        p._last_done -= 30
        p._stall_s = 1.0                                   # у справжньому плеєрі 8 с; тут швидше
        end = time.time() + 10
        while time.time() < end and not any("watchdog" in r.getMessage() for r in caplog.records):
            time.sleep(0.1)
        msgs = [r.getMessage() for r in caplog.records if "watchdog" in r.getMessage()]
        assert len(msgs) == 1 and "frames are not produced" in msgs[0] and "--- thread" in msgs[0]
        time.sleep(2.5)
        assert len([r for r in caplog.records if "watchdog" in r.getMessage()]) == 1, "той самий застій не повторюється"
    finally:
        p._src = None
        p.close()


def test_watchdog_does_not_count_a_long_pause_as_a_stall(caplog):
    """користувач постояв на паузі пів години й натиснув «грати»: «востаннє кадр був 1800 с тому» - не застій, відлік іде від початку очікування кадрів
    (у журналі так з'являлося «кадри не готуються 142 с» через 1 с після запуску пулу)"""
    import logging
    p = Player(workers=1)
    try:
        caplog.set_level(logging.WARNING, logger="inkreel")
        p.info, p._src = {"kind": "video"}, object()
        p._want_play, p._state = True, "buffering"
        p._last_done -= 1800
        p._stall_s = 3.0
        time.sleep(3.5)                                    # сторож зазирає раз на 2 с: за цей час він мав би одразу вдарити на тривогу, якби ліку від паузи не було
        assert not any("watchdog" in r.getMessage() for r in caplog.records)
        end = time.time() + 8
        while time.time() < end and not any("watchdog" in r.getMessage() for r in caplog.records):
            time.sleep(0.1)
        msgs = [r.getMessage() for r in caplog.records if "watchdog" in r.getMessage()]
        assert len(msgs) == 1 and "frames are not produced" in msgs[0], "справжній застій після початку очікування все одно ловиться"
    finally:
        p._src = None
        p.close()


def test_render_fps_is_one_counter_for_all_threads():
    """кілька потоків віддають кадри одночасно: лічильник показує їхню СУМУ, а не швидкість одного"""
    import threading
    p = Player(workers=1)
    try:
        p._have = np.zeros(2000, bool)
        p._pend = np.zeros(2000, bool)
        p._gen = 7
        stop = time.perf_counter() + 2.5

        def feed(base):
            i = 0
            while time.perf_counter() < stop:
                p._done(7, base + i, np.zeros(1, np.uint8))        # ≈ 25 кадрів/с на потік
                i += 1
                time.sleep(0.04)
        ts = [threading.Thread(target=feed, args=(k * 400,)) for k in range(4)]
        for t in ts:
            t.start()
        time.sleep(2.2)
        r = p.render_fps()
        for t in ts:
            t.join()
        assert 70 <= r <= 130, r                                   # 4 потоки по ≈25 = ≈100, а не 25
        time.sleep(1.2)
        assert p.render_fps() == 0.0                               # нових кадрів нема - 0
    finally:
        p.close()


def _vfr_clip(path):
    """відео зі змінною частотою кадрів: ffmpeg викидає кадри з рівного ряду 30 к/с, лишаючи їхні справжні мітки часу (як телефон/месенджер)"""
    import shutil
    import subprocess

    import pytest
    from benvid import video
    ff = video.find_ffmpeg()
    if not ff:
        pytest.skip("потрібен ffmpeg")
    src = path + ".cfr.mp4"
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x96:rate=30", "-t", "4", "-c:v", "mpeg4", src], capture_output=True)
    r2 = subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, "-vf", "select='gt(mod(n,10),0)*gt(mod(n,10),3)+eq(mod(n,10),1)'", "-vsync", "vfr", "-c:v", "mpeg4", path],
                        capture_output=True)
    if r.returncode or r2.returncode or not os.path.isfile(path):
        pytest.skip("ffmpeg не зібрав тестове VFR-відео")
    return src


def test_vfr_video_is_played_by_real_frame_times(tmp_path):
    """змінна частота кадрів: плеєр знає справжні моменти кадрів (starts), а сталий файл їх не вимагає й нічого зайвого не читає"""
    from benvid import video
    from benvid.media import VideoSource
    path = str(tmp_path / "v.mp4")
    cfr = _vfr_clip(path)
    info = video.video_info(path)
    t = video.vfr_times(path, info)
    assert t is not None and len(t) == info["frames"] and np.all(np.diff(t) >= 0) and abs(t[-1] - 3.93) < 0.1
    assert video.vfr_times(cfr, video.video_info(cfr)) is None            # рівний ряд: розбіжності fps/tbr нема - ffmpeg для міток не викликається
    src = VideoSource(path)
    try:
        assert src.info["starts"] is not None and abs(src.info["duration"] - 4.0) < 0.2
    finally:
        src.cap.release()
    p = Player(workers=2)
    try:
        p.open(path, "144p")
        assert p._starts is not None
        assert abs(p.time_of(p.n - 1) - t[-1]) < 1e-6 and abs(p.duration - src.info["duration"]) < 0.2
        p.seek(10)
        assert abs(p.time_of(p.position()) - t[10]) < 1e-6
    finally:
        p.close()


def test_start_of_video_is_filled_after_the_rest(tmp_path):
    """зміна налаштувань посеред відео: порахувавши все попереду, черга дораховує й початок позаду позиції (якщо відео вміщається в пам'ять) - без перемотки"""
    path = str(tmp_path / "w.mp4")
    _clip(path, n=40)
    p = Player(workers=2)
    try:
        p.open(path, "144p")
        assert _wait(p, lambda: p.buffered_end() >= 5)
        p.seek(25)
        assert _wait(p, lambda: p.buffered_end() >= p.n - 2)
        assert _wait(p, lambda: bool(p._have[:p.n].all()), t=60), "початок лишився непорахованим"
    finally:
        p.close()


def test_fps_limit_changes_what_is_shown_from_the_ready_buffer(tmp_path):
    """уже пораховані кадри не показуються поза рядом ліміту к/с: після зміни ліміту на готовому буфері кадр позиції береться з вихідного ряду"""
    path = str(tmp_path / "f.mp4")
    _clip(path, n=40, fps=24)
    p = Player(workers=2)
    try:
        p.open(path, "144p")
        assert _wait(p, lambda: bool(p._have[:p.n].all()), t=60)
        assert p._key(7) == 7
        p.set_fps_limit(12)
        assert p._key(7) == 6 and p._key(6) == 6 and p._key(8) == 8
    finally:
        p.close()
