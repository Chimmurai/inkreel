"""Порядок серед процесів (Windows). Обробники кадрів - це окремі процеси; щоб у диспетчері завдань вони не
виглядали як купа однакових «InkReel.exe», а головне - не лишалися сиротами після аварійного закриття:
  • усі вони складаються в один Job Object, який знищується разом з головним вікном;
  • у зібраному .exe обробники запускаються з InkReel-Worker.exe (поруч з InkReel.exe), тож у списку видно,
    що є що: один «InkReel» і кілька «InkReel-Worker»."""
from __future__ import annotations

import multiprocessing
import os
import sys
import threading

_JOB = None


def _kill_with_parent():
    global _JOB
    import ctypes
    from ctypes import wintypes as wt

    k = ctypes.windll.kernel32
    k.CreateJobObjectW.restype = wt.HANDLE
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
    k.GetCurrentProcess.restype = wt.HANDLE
    k.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    k.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("a", "b", "c", "d", "e", "f")]

    class EXT(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job = k.CreateJobObjectW(None, None)
    if not job:
        return
    info = EXT()
    info.Basic.LimitFlags = 0x2000                      # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        if k.AssignProcessToJobObject(job, k.GetCurrentProcess()):
            _JOB = job                                  # тримаємо дескриптор до кінця життя процесу


def _mutex():
    """іменований м'ютекс: за ним інсталятор (AppMutex) знає, що програма запущена"""
    global _MUTEX
    import ctypes
    _MUTEX = ctypes.windll.kernel32.CreateMutexW(None, False, "InkReelAppMutex")


_MUTEX = None


def setup():
    """викликати один раз у головному процесі (не в обробниках)"""
    if os.name != "nt" or multiprocessing.parent_process() is not None:
        return
    try:
        _mutex()
    except Exception:  # noqa: BLE001
        pass
    try:
        _kill_with_parent()
    except Exception:  # noqa: BLE001
        pass
    if getattr(sys, "frozen", False):
        w = os.path.join(os.path.dirname(sys.executable), "InkReel-Worker.exe")
        if os.path.isfile(w):
            try:
                multiprocessing.set_executable(w)
            except Exception:  # noqa: BLE001
                pass


def crisp_text():
    """Windows із масштабом екрана 125-200% розтягує вікно програми, що не знає про DPI, як картинку: текст на таких моніторах «напіврозмитий».
    Режим DPI_AWARENESS_CONTEXT_UNAWARE_GDISCALED (Windows 10 1809+): вікно й розкладка лишаються як були (жодних зсувів), а текст і лінії, що малює GDI (весь
    інтерфейс Tk), Windows малює одразу в потрібному розмірі - чітко. Викликати ДО створення першого вікна; на старих Windows і в разі відмови нічого не робить."""
    if os.name != "nt" or os.environ.get("INKREEL_DPI") == "0":          # INKREEL_DPI=0 - вимкнути (запасний вихід, якщо на чиємусь ПК стане гірше)
        return False
    try:
        import ctypes
        f = ctypes.windll.user32.SetProcessDpiAwarenessContext
        f.argtypes = [ctypes.c_void_p]
        f.restype = ctypes.c_int
        return bool(f(ctypes.c_void_p(-5)))
    except Exception:  # noqa: BLE001
        return False


def fine_timer():
    """Windows за замовчуванням будить таймери раз на ~15,6 мс: кадри відео показуються нерівно (смикання).
    timeBeginPeriod(1) дає точність 1 мс - відтворення стає плавним. Без побічних ефектів, діє лише поки програма жива."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:  # noqa: BLE001
        pass


def lower_thread_priority():
    """потік фонової роботи (подача кадрів, експорт) працює з нижчим за звичайний пріоритетом, ніж потік інтерфейсу: вікно й система
    відповідають, навіть коли всі ядра зайняті обробкою (Windows; деінде нічого не робить)"""
    if os.name != "nt":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.GetCurrentThread.restype = ctypes.c_void_p
        k.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        k.SetThreadPriority(k.GetCurrentThread(), -1)       # THREAD_PRIORITY_BELOW_NORMAL
    except Exception:  # noqa: BLE001
        pass


def set_thread_priority(level: int):
    """пріоритет поточного потоку Windows: -1 нижче звичайного, 0 звичайний, 1 вищий. Для «Нейро» потоки подачі звичайні: мережа на відеокарті
    ледь навантажує процесор, а простій черги коштує кадрів"""
    if os.name != "nt":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.GetCurrentThread.restype = ctypes.c_void_p
        k.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        k.SetThreadPriority(k.GetCurrentThread(), int(level))
    except Exception:  # noqa: BLE001
        pass


def arm_exit_watchdog(delay: float = 4.0, code: int = 0) -> threading.Timer:
    """аварійний вихід: вікно закрите, а процес не завершився за delay секунд (потік завис у драйвері відеокарти, у ffmpeg, на диску) -
    примусово завершити. Фоновий потік працює й тоді, коли інтерпретатор уже чекає на інші потоки, тож процес не лишається в
    диспетчері завдань і не блокує повторний запуск"""
    t = threading.Timer(delay, os._exit, (code,))
    t.daemon = True
    t.start()
    return t


def release_everything(watchdog: float | None = 4.0, listener=None):
    """вікно закрито (або закривається): слухач одиночного запуску, усі пули процесів, решта дочірніх процесів, аварійний вихід.
    Кожен крок окремо: збій одного не повинен лишити решту живими"""
    if listener is not None:
        try:
            listener.close()
        except Exception:  # noqa: BLE001
            pass
    try:
        from .video import kill_all_pools
        kill_all_pools()
    except Exception:  # noqa: BLE001
        pass
    try:
        for p in multiprocessing.active_children():
            p.terminate()
    except Exception:  # noqa: BLE001
        pass
    if watchdog is not None:
        arm_exit_watchdog(watchdog)
