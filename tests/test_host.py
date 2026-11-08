"""процес-господар (потоки в одному процесі) замість пулу процесів"""
import os
import time
from concurrent.futures import CancelledError

import cv2
import numpy as np
import pytest

from benvid import video
from benvid.core import Params
from benvid.host import HostBroken, RenderHost


def _add(a, b):
    return a + b


def _boom(_):
    raise ValueError("нема")


def _nap(t):
    time.sleep(t)
    return os.getpid()


def _clip(path, n=14, w=640, h=360):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[:] = (60 + (xx * 140 // w))[..., None]
        cv2.circle(f, (60 + i * 18, h // 2), 70, (235, 235, 235), -1)
        cv2.putText(f, "Ab", (200, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (10, 10, 10), 4)
        vw.write(f)
    vw.release()


@pytest.fixture
def host():
    h = RenderHost(3)
    yield h
    video.kill_pool(h)


def test_results_and_errors(host):
    fs = [host.submit(_add, i, 1) for i in range(8)]
    assert [f.result(60) for f in fs] == [i + 1 for i in range(8)]
    with pytest.raises(ValueError):
        host.submit(_boom, 0).result(60)
    assert len({host.submit(_nap, 0.05).result(60) for _ in range(3)}) == 1     # усе в одному процесі
    assert host._proc.pid not in (None, os.getpid())


def test_cancel_queued_jobs(host):
    """завдання, що ще в черзі, скасовуються (після перемотки застарілі шматки не мають займати потоки)"""
    busy = [host.submit(_nap, 1.5) for _ in range(3)]
    time.sleep(0.5)
    queued = [host.submit(_nap, 0.01) for _ in range(5)]
    assert all(f.cancel() for f in queued)
    assert all(f.cancelled() for f in queued)
    for f in queued:
        with pytest.raises(CancelledError):
            f.result(1)
    assert [f.result(60) > 0 for f in busy] == [True] * 3
    assert host.submit(_add, 2, 3).result(60) == 5                  # господар живий


def test_dead_host_fails_everything_at_once(host):
    """процес-господар аварійно помер: усе в польоті одразу падає з HostBroken (плеєр перестворює пул), нових завдань не беремо"""
    f = host.submit(_nap, 30)
    g = host.submit(_nap, 30)
    time.sleep(0.5)
    t = time.time()
    host._proc.terminate()
    for x in (f, g):
        with pytest.raises(HostBroken):
            x.result(15)
    assert time.time() - t < 15
    with pytest.raises(Exception):
        host.submit(_add, 1, 1)


def test_kill_pool_cancels_and_stops_fast(host):
    f = host.submit(_nap, 30)
    time.sleep(0.5)
    procs = list(host._processes.values())
    t = time.time()
    video.kill_pool(host)
    for p in procs:
        p.join(10)
    assert not any(p.is_alive() for p in procs) and time.time() - t < 10
    assert f.cancelled() or f.done()


def test_make_pool_picks_host_or_processes(monkeypatch):
    monkeypatch.delenv("INKREEL_HOST", raising=False)              # типово - пул процесів (він швидший за потоки господаря)
    ex = video.make_pool(2, (640, 360))
    try:
        assert not getattr(ex, "is_host", False) and ex.submit(_add, 1, 2).result(60) == 3
    finally:
        video.kill_pool(ex)
    monkeypatch.setenv("INKREEL_HOST", "1")
    ex = video.make_pool(2, (640, 360))
    try:
        assert getattr(ex, "is_host", False) and ex.submit(_add, 1, 2).result(60) == 3
    finally:
        video.kill_pool(ex)


def test_ultra_chunks_in_threads_equal_sequential(tmp_path, monkeypatch):
    """кадри «Ультри», пораховані кількома потоками одного процесу одночасно, збігаються з послідовним розрахунком до біта:
    спільні кеші й таблиці не псуються паралельними викликами, декодери не ділять потоки.
    Відеокарту вимкнено: її ядра не побітові з numba (на ПК із GPU послідовний прохід міг би піти на GPU, а потоковий - на CPU)"""
    from benvid.core import Params
    monkeypatch.setenv("INKREEL_GPU", "off")
    path = str(tmp_path / "u.mp4")
    _clip(path)
    pd = Params(method=5).to_dict()
    jobs = [(path, 0, 5, 640, 360, pd), (path, 5, 5, 640, 360, pd), (path, 10, 4, 640, 360, pd)]
    ref = [video._job(j) for j in jobs]
    h = RenderHost(3)
    try:
        got = [f.result(300) for f in [h.submit(video._job, j) for j in jobs]]
        again = [f.result(300) for f in [h.submit(video._job, j) for j in jobs]]      # другий прохід: кеші вже тепліші
    finally:
        video.kill_pool(h)
    for a, b, c in zip(ref, got, again):
        assert a.shape == b.shape and np.array_equal(a, b) and np.array_equal(a, c)


def _big(n):
    return np.zeros(n, np.uint8)


def test_results_do_not_stop_other_threads_and_stats_arrive(host):
    """відправка великого результату не блокує обчислювальні потоки (їх обслуговує окремий потік-відправник), а лічильники господаря
    приходять батькові"""
    t0 = time.perf_counter()
    big = host.submit(_big, 60 << 20)
    naps = [host.submit(_nap, 0.3) for _ in range(3)]
    for f in naps:
        f.result(30)
    assert time.perf_counter() - t0 < 5
    assert len(big.result(60)) == 60 << 20
    end = time.time() + 10
    while host.stats.get("sent", 0) < 1 and time.time() < end:
        time.sleep(0.1)
    assert host.stats.get("sent", 0) >= 1 and "send_busy" in host.stats


def test_memory_governor_logic():
    """страж пам'яті: понад бюджет нові завдання чекають, але хоч одне йде завжди (без зависань); нижче 85% бюджету - знову все йде"""
    import threading

    from benvid.host import _Gov
    g = _Gov(soft=1000, floor=100)
    g.enter()                                            # перше завдання стартує завжди
    g.update(rss=1500, free=10_000, dt=0.3)
    assert g.over and g.waits == 0
    started = []

    def second():
        g.enter()
        started.append(1)
        g.leave()
    t = threading.Thread(target=second, daemon=True)
    t.start()
    time.sleep(0.3)
    assert not started, "над бюджетом друге завдання чекає"
    g.update(rss=900, free=10_000, dt=0.3)
    assert g.over, "гістерезис: 900 - ще понад 85% бюджету"
    g.update(rss=800, free=10_000, dt=0.3)
    t.join(3)
    assert started and not g.over and g.waits == 1
    g.leave()
    g.update(rss=100, free=50, dt=0.3)                   # мало вільної пам'яті в системі - теж «понад»
    assert g.over
    g.enter()                                            # але порожній господар завдання приймає
    g.leave()
    assert g.peak == 1500


def test_governor_keeps_the_host_alive_over_budget(monkeypatch):
    """бюджет крихітний - усі завдання все одно доходять до кінця (по одному), а в лічильниках видно, що страж втручався"""
    monkeypatch.setenv("INKREEL_HOST_MEM_MB", "1")
    h = RenderHost(3)
    try:
        fs = [h.submit(_nap, 0.6 + 0.35 * i) for i in range(6)]       # різної тривалості: нові завдання стартують, коли вже відомо, що над бюджетом
        for f in fs:
            f.result(60)
        end = time.time() + 15
        while time.time() < end and not (h.stats.get("rss_peak", 0) > 0 and h.stats.get("mem_waits", 0) >= 1):
            time.sleep(0.2)                                           # лічильники приходять від господаря із затримкою
        assert h.stats.get("rss_peak", 0) > 0 and h.stats.get("mem_soft") == 1
        assert h.stats.get("mem_waits", 0) >= 1, "над бюджетом друга хвиля завдань чекала"
    finally:
        video.kill_pool(h)


def _thread_names(_):
    import threading
    return sorted(t.name for t in threading.enumerate())


def test_stack_sampler_is_off_by_default(host):
    """знімок стеків усіх потоків 10 разів на секунду (діагностика) тримав масиви кадрів живими й роздував господаря до десятків ГБ:
    типово його немає, є лише легкий сторож пам'яті"""
    names = host.submit(_thread_names, 0).result(30)
    assert "ink-host-sampler" not in names and "ink-host-memwatch" in names, names


def test_stage_accounting_is_collected(tmp_path):
    """облік часу по стадіях (читання, бінаризація, упакування) накопичується й потрапляє в лічильники господаря - щоб у журналі було видно, де йде час"""
    path = str(tmp_path / "a.mp4")
    _clip(path, n=8)
    before = video.take_acc()
    packed = video._job((path, 0, 6, 320, 180, Params(method=3).to_dict()))
    assert len(packed) == 6
    after = video.take_acc()
    assert after["frames"] - before["frames"] == 6
    assert after["proc"] > before["proc"] and after["dec"] >= before["dec"]
    from benvid import host
    st = host._stats({})
    assert st["acc"]["frames"] >= 6 and "gpu_busy" in st and "gpu_calls" in st
