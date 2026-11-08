"""після закриття вікна процес мусить справді зникнути: жодного пулу, слухача чи потоку, що тримали б його живим (інакше невидимий
InkReel у диспетчері завдань блокує повторний запуск, доки його не вб'ють вручну)"""
import os
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import pytest

from benvid import procs, single, video

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_kill_all_pools_stops_busy_workers():
    from multiprocessing.connection import wait as wait_exit
    ex = video.make_pool(2)
    try:
        assert ex in list(video._LIVE_POOLS)
        ex.submit(time.sleep, 60)
        ex.submit(time.sleep, 60)
        end = time.time() + 20
        while time.time() < end and len(getattr(ex, "_processes", {})) < 2:
            time.sleep(0.05)
        procs_ = list(ex._processes.values())
        assert len(procs_) == 2
        sentinels = [p.sentinel for p in procs_]
        video.kill_all_pools(wait=5.0)
        # «процес завершився» дивимось за вартовим дескриптором (ОС закриває його, коли процес зникає), а не за Process.is_alive(): пул тим часом сам збирає
        # свої процеси (waitpid), і два одночасні waitpid на тому самому процесі на мить роблять is_alive() хибно True, хоч процеса вже нема
        # (2 збої з 40 запусків, процеси при цьому мали код -15 і зникали з /proc за 0.0 с)
        assert len(wait_exit(sentinels, 10)) == len(sentinels)
    finally:
        video.kill_pool(ex)


def test_pool_registry_does_not_keep_pools_alive():
    import gc
    ex = video.make_pool(1)
    n = len(video._LIVE_POOLS)
    video.kill_pool(ex)
    del ex
    gc.collect()
    assert len(video._LIVE_POOLS) < n


def test_exit_watchdog_fires(monkeypatch):
    called = threading.Event()
    codes = []
    monkeypatch.setattr(os, "_exit", lambda c: (codes.append(c), called.set()))
    t = procs.arm_exit_watchdog(0.05)
    assert t.daemon                                        # сам таймер не тримає процес
    assert called.wait(3) and codes == [0]


def test_release_everything_closes_listener_and_pools(monkeypatch):
    monkeypatch.setattr(os, "_exit", lambda c: None)
    port, magic = 47917, "RELEASE1"
    lst = single.acquire("", port, magic)
    if not isinstance(lst, single.Listener):
        pytest.skip("порт зайнятий чимось стороннім")
    ex = video.make_pool(1)
    ex.submit(time.sleep, 30)
    end = time.time() + 20
    while time.time() < end and not getattr(ex, "_processes", {}):
        time.sleep(0.05)
    ps = list(ex._processes.values())
    assert ps and all(p.is_alive() for p in ps)
    procs.release_everything(None, lst)
    lst._t.join(3)
    assert not lst._t.is_alive() and single._handoff("x", port, magic) is False
    for p in ps:
        p.join(10)
    assert all(not p.is_alive() for p in ps)


def test_process_with_busy_pool_exits_after_release(tmp_path):
    """сам сценарій з репорту: головний потік скінчився, пул ще дораховує довгий шматок. Без release_everything інтерпретатор
    дочікується робітників і висить (перевірено вручну: таймаут); з ним - виходить за кілька секунд"""
    script = tmp_path / "bye.py"
    script.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {ROOT!r})
        from benvid.video import make_pool
        from benvid import procs
        if __name__ == "__main__":
            ex = make_pool(2)
            ex.submit(time.sleep, 120)
            ex.submit(time.sleep, 120)
            time.sleep(2.0)
            procs.release_everything(3.0)
            print("main done", flush=True)
    """), encoding="utf-8")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "main done" in r.stdout
    assert time.time() - t0 < 30


def test_deliver_confirms_only_when_the_window_handles_it(monkeypatch):
    """повторний запуск отримує «OK», лише коли головний цикл вікна справді показав файл (див. gui._deliver)"""
    from conftest import NO_SCREEN, pump
    if NO_SCREEN:
        pytest.skip("немає екрана (запускай під xvfb-run)")
    import queue
    from benvid import gui

    class FakeApp:
        def __init__(self):
            self.q, self.got = queue.Queue(), []

        def bring_front(self, p=""):
            self.got.append(p)
    app = FakeApp()
    out = []
    th = threading.Thread(target=lambda: out.append(gui._deliver(app, "a.mp4", 5.0)))
    th.start()
    end = time.time() + 5
    while th.is_alive() and time.time() < end:                 # «головний цикл»: виконує чергу
        try:
            app.q.get(timeout=0.05)()
        except queue.Empty:
            pass
    th.join(2)
    assert out == [True] and app.got == ["a.mp4"]
    assert gui._deliver(app, "b.mp4", 0.2) is False            # цикл не крутиться (вікно закрите/зависло) - підтвердження нема


def test_kill_pool_from_a_done_callback_of_a_broken_pool_does_not_hang_the_process():
    """воркер пулу вмирає -> колбек майбутнього (потік-менеджер, під замком shutdown) викликає kill_pool: раніше shutdown чекав на цей самий замок
    вічно, і процес не міг завершитись (висів у _python_exit) та блокував повторний запуск. Перевіряємо в окремому процесі, що він сам завершується"""
    code = (
        "import os, sys, threading\n"
        "sys.path.insert(0, %r)\n"
        "from benvid import video\n"
        "ex = video.make_pool(1)\n"
        "done = threading.Event()\n"
        "def cb(f):\n"
        "    video.kill_pool(ex)\n"
        "    done.set()\n"
        "f = ex.submit(os._exit, 3)\n"
        "f.add_done_callback(cb)\n"
        "assert done.wait(30), 'колбек не відпрацював'\n"
        "print('ok')\n"
    ) % ROOT
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert r.returncode == 0 and "ok" in r.stdout, (r.stdout, r.stderr[-500:])
