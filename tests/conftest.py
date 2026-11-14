"""Спільне для тестів: сторож зависання, мітки повільних тестів, акуратний вихід процесу pytest."""
from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    HANG_S = float(os.environ.get("INKREEL_TEST_HANG_S", "900"))        # 0 - без сторожа
except ValueError:
    HANG_S = 900.0

_ERR_FD = None                    # справжній stderr (pytest на час тесту перехоплює fd 2: усе, що туди пишеться, пропадає разом із процесом)


def _hang_exit(name: str) -> None:
    """тест не завершився за HANG_S: стеки всіх потоків у справжній stderr і вихід з кодом 3"""
    try:
        fd = _ERR_FD if _ERR_FD is not None else 2
        os.write(fd, f"\n!!! {name}: тест не завершився за {HANG_S:.0f} с (INKREEL_TEST_HANG_S); стеки потоків нижче, процес завершується\n".encode("utf-8", "replace"))
        faulthandler.dump_traceback(file=fd, all_threads=True)
    except Exception:  # noqa: BLE001
        pass
    os._exit(3)


@pytest.fixture(autouse=True)
def _hang_guard(request):
    """тест, що висить, не повинен тримати весь прогін і лишати в пам'яті python.exe (так було з дедлоком в student/runtime.py): якщо тест не завершився за
    HANG_S секунд, у справжній stderr падають стеки всіх потоків і процес завершується - за стеком видно, де саме він завис.
    Свій таймер, а не faulthandler.dump_traceback_later: той один на процес, і він скасував би `-o faulthandler_timeout=...` зі scripts\\tests.bat"""
    if HANG_S <= 0:
        yield
        return
    t = threading.Timer(HANG_S, _hang_exit, args=(request.node.nodeid,))
    t.daemon = True
    t.start()
    try:
        yield
    finally:
        t.cancel()


SLOW_FILES = {"test_video.py", "test_player.py", "test_host.py", "test_glyph.py", "test_train_ink.py", "test_make_dataset.py", "test_gpu.py", "test_autosize.py",
              "test_view.py", "test_qc.py", "test_msix.py", "test_motion.py", "test_formats.py", "test_bake.py", "test_big_frame.py"}


def pytest_configure(config):
    global _ERR_FD
    try:
        _ERR_FD = os.dup(2)                                  # тут захоплення ще не ввімкнене: це справжній stderr
    except OSError:
        _ERR_FD = None
    config.addinivalue_line("markers", "slow: експорти відео, пули процесів, навчання (хвилини); швидкий запуск: -m \"not slow\"")


def pytest_collection_modifyitems(config, items):
    """тести повільних файлів автоматично отримують мітку slow"""
    for it in items:
        if os.path.basename(str(it.fspath)) in SLOW_FILES:
            it.add_marker(pytest.mark.slow)


def pytest_sessionfinish(session, exitstatus):
    """після тестів із вікном у процесі лишаються пули/потоки (вікно закрите в обхід release_everything) - pytest тоді не виходить, а меню scripts\\tests.bat не повертається.
    Закриваємо пули й ставимо страховку: якщо процес не завершився за 15 с, виходимо примусово з тим самим кодом"""
    import threading

    def _force():
        time.sleep(15.0)
        try:
            import multiprocessing
            for p in multiprocessing.active_children():      # os._exit не чіпає дітей: обробники пулу лишались би осиротілими (по ~230 МБ кожен) після кожного такого виходу
                p.kill()
        except Exception:  # noqa: BLE001
            pass
        os._exit(int(exitstatus))
    threading.Thread(target=_force, daemon=True, name="pytest-exit-guard").start()      # СПЕРШУ страховка: зламаний пул може заблокувати й сам kill_all_pools
    def _kill():
        try:
            from benvid import video
            video.kill_all_pools()
        except Exception:  # noqa: BLE001
            pass
    threading.Thread(target=_kill, daemon=True, name="pytest-kill-pools").start()      # в потоці: зламаний пул не повинен блокувати друк підсумку pytest
