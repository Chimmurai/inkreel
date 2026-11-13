"""Спільне для тестів вікна (test_layout, test_gui_behaviour): справжнє вікно InkReel на тимчасових налаштуваннях.
Де немає екрана (Linux без DISPLAY), такі тести пропускаються:  xvfb-run -a python -m pytest tests"""
from __future__ import annotations

import faulthandler
import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NO_SCREEN = sys.platform.startswith("linux") and not os.environ.get("DISPLAY")

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


def settle(app, rounds=8, pause=0.05):
    """дати вікну домалюватись і розкластись (і прибрати чергу подій)"""
    for _ in range(rounds):
        app.update_idletasks()
        app.update()
        time.sleep(pause)
    app.sstack.layout()
    app.pstack.layout()
    app.update_idletasks()
    app.update()


def pump(app, cond, timeout=30.0, pause=0.02):
    """крутити цикл подій, доки cond() не стане істинним (або мине timeout); -> результат cond()"""
    end = time.time() + timeout
    while time.time() < end:
        app.update()
        if cond():
            return True
        time.sleep(pause)
    return bool(cond())


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    """вікно з білою «Класикою», мовою uk й тимчасовим файлом налаштувань (налаштування користувача не чіпаються)"""
    if NO_SCREEN:
        pytest.skip("немає екрана (запускай під xvfb-run)")
    import tkinter as tk
    from benvid import engine, gui

    tmp = tmp_path_factory.mktemp("settings")
    mp = pytest.MonkeyPatch()
    mp.setattr(engine, "SETTINGS_FILE", str(tmp / "settings.json"))
    mp.setattr(engine, "_OLD", str(tmp / "old.json"))
    with open(engine.SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump({"ui": {"lang": "uk", "palette": "Класика", "pal_inv": False, "size": "Авто"}}, f)
    try:
        a = gui.App()
    except tk.TclError as e:                          # немає працюючого екрана
        mp.undo()
        pytest.skip(f"Tk недоступний: {e}")
    a.geometry("1280x900")
    try:                                              # Windows: winfo_containing і клавіші працюють лише для вікна, що зверху й у фокусі (інакше його закриває консоль, з якої запущено тести)
        a.attributes("-topmost", True)
        a.lift()
        a.focus_force()
    except tk.TclError:
        pass
    settle(a)
    yield a
    try:
        a.player.close()
    except Exception:  # noqa: BLE001
        pass
    a.destroy()
    mp.undo()


SLOW_FILES = {"test_video.py", "test_player.py", "test_host.py", "test_glyph.py", "test_train_ink.py", "test_make_dataset.py", "test_gpu.py", "test_autosize.py",
              "test_view.py", "test_qc.py", "test_msix.py", "test_motion.py", "test_formats.py", "test_bake.py", "test_big_frame.py"}


def pytest_configure(config):
    global _ERR_FD
    try:
        _ERR_FD = os.dup(2)                                  # тут захоплення ще не ввімкнене: це справжній stderr
    except OSError:
        _ERR_FD = None
    config.addinivalue_line("markers", "slow: експорти відео, пули процесів, навчання (хвилини); швидкий запуск: -m \"not gui and not slow\"")
    config.addinivalue_line("markers", "gui: тест відкриває справжнє вікно InkReel (повільний; не торкайся миші й клавіатури). Швидкий запуск: -m \"not gui\"")


def pytest_collection_modifyitems(config, items):
    """кожен тест, що користується вікном (фікстура app), автоматично отримує мітку gui"""
    for it in items:
        if "app" in getattr(it, "fixturenames", ()):
            it.add_marker(pytest.mark.gui)
        if os.path.basename(str(it.fspath)) in SLOW_FILES:
            it.add_marker(pytest.mark.slow)


def pytest_sessionfinish(session, exitstatus):
    """після тестів із вікном у процесі лишаються пули/потоки (вікно закрите в обхід release_everything) - pytest тоді не виходить, а меню scripts\\tests.bat не повертається.
    Закриваємо пули й ставимо страховку: якщо процес не завершився за 15 с, виходимо примусово з тим самим кодом"""
    import threading

    def _force():
        time.sleep(15.0)
        os._exit(int(exitstatus))
    threading.Thread(target=_force, daemon=True, name="pytest-exit-guard").start()      # СПЕРШУ страховка: зламаний пул може заблокувати й сам kill_all_pools
    try:
        from benvid import video
        video.kill_all_pools()
    except Exception:  # noqa: BLE001
        pass
