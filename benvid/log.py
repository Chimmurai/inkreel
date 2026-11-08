"""Журнал помилок і службові теки. Програма без консолі (.exe) не має куди писати traceback, тому все йде у файл:
Windows: %LOCALAPPDATA%\\InkReel\\inkreel.log, інакше ~/.inkreel/inkreel.log."""
from __future__ import annotations

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler


def app_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else None
    d = os.path.join(base, "InkReel") if base else os.path.join(os.path.expanduser("~"), ".inkreel")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = os.path.expanduser("~")
    return d


def log_path() -> str:
    return os.path.join(app_dir(), "inkreel.log")


def shown_path() -> str:
    """шлях журналу для показу людині: у збірці зі Store (MSIX) Windows перенаправляє запис у теку пакета, і саме там файл видно в Провіднику"""
    from . import store
    return store.visible_path(log_path())


def prepare_env():
    """кеш скомпільованих ядер numba - у теці користувача, якщо тека програми недоступна для запису (Program Files).
    Ще: декодер ffmpeg усередині OpenCV не засмічує консоль попередженнями про биті опорні кадри (h264 «mmco: unref short failure» - типово для відео,
    пересланих месенджерами; картинку це не ламає)"""
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
    if "NUMBA_CACHE_DIR" in os.environ:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    if getattr(sys, "frozen", False) or not os.access(here, os.W_OK):
        d = os.path.join(app_dir(), "cache")
        try:
            os.makedirs(d, exist_ok=True)
            os.environ["NUMBA_CACHE_DIR"] = d
        except OSError:
            pass


_NOISE = (b"mmco: unref short failure",)


def quiet_console():
    """запуск з консолі: libav (всередині OpenCV і дочірніх процесів) пише в stderr «[h264 @ ...] mmco: unref short failure» - типове попередження для відео,
    пересланих месенджерами, картинку воно не ламає. Змінна OPENCV_FFMPEG_LOGLEVEL діє не в усіх збірках, тож додатково фільтруємо stderr: усе, крім цього
    одного рядка, проходить як було. У зібраному exe без консолі нічого не робимо. Повертає True, якщо фільтр увімкнено"""
    import atexit
    import threading
    try:
        if os.environ.get("INKREEL_NOISE") == "1" or not sys.stderr or not os.isatty(2):
            return False
        keep = os.dup(2)
        r, w = os.pipe()
        os.dup2(w, 2)
        os.close(w)
    except (OSError, ValueError, AttributeError):
        return False

    def pump():
        buf = b""
        try:
            while True:
                chunk = os.read(r, 4096)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = buf.split(b"\n")
                out = b"".join(ln + b"\n" for ln in lines if not any(k in ln for k in _NOISE))
                if out:
                    os.write(keep, out)
                if len(buf) > 65536:                          # довгий рядок без переносу: не копити
                    os.write(keep, buf)
                    buf = b""
        except OSError:
            pass
        if buf and not any(k in buf for k in _NOISE):
            try:
                os.write(keep, buf)
            except OSError:
                pass

    t = threading.Thread(target=pump, daemon=True, name="stderr-filter")
    t.start()

    def restore():                                            # при виході повернути справжній stderr і дописати хвіст (напр. traceback)
        try:
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            os.dup2(keep, 2)
            t.join(1.0)
        except OSError:
            pass
    atexit.register(restore)
    return True


_CRASH = None


def enable_crash_trace():
    """нативний збій (access violation у драйвері/onnxruntime) не лишає traceback у inkreel.log: процес просто зникає. faulthandler пише стеки всіх потоків
    у crash.log поруч із журналом; файл відкритий весь час роботи. Помилка тут ніколи не заважає запуску"""
    global _CRASH
    if _CRASH is not None:
        return
    try:
        import faulthandler
        _CRASH = open(os.path.join(app_dir(), "crash.log"), "a", buffering=1, encoding="utf-8")
        _CRASH.write(f"--- запуск {os.getpid()}\n")
        faulthandler.enable(file=_CRASH, all_threads=True)
    except Exception:  # noqa: BLE001
        _CRASH = None


_LOG = None


def get() -> logging.Logger:
    global _LOG
    if _LOG is None:
        _LOG = logging.getLogger("inkreel")
        _LOG.setLevel(logging.INFO)
        try:
            h = RotatingFileHandler(log_path(), maxBytes=256 * 1024, backupCount=2, encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            _LOG.addHandler(h)
        except OSError:
            _LOG.addHandler(logging.NullHandler())
    return _LOG


get_logger = get                 # ім'я, яким журнал викликають gpu/player/video (раніше його не існувало: ті записи тихо губились)


def log_environment():
    """один рядок на початку роботи: версія програми, система, процесор, пам'ять і версії ключових бібліотек. Коли користувач присилає журнал, одразу видно, на чому
    він працював. Без імен, шляхів і будь-чого особистого. Бібліотеки, яких ще не завантажено, не імпортуються заради цього рядка"""
    try:
        import platform

        from . import __version__
        parts = [f"InkReel {__version__}" + (" (exe)" if getattr(sys, "frozen", False) else ""), platform.platform(), f"Python {platform.python_version()}",
                 f"{os.cpu_count() or '?'} логічних ядер"]
        try:
            from .video import ram_bytes
            parts.append(f"{ram_bytes() / 2 ** 30:.0f} ГБ пам'яті")
        except Exception:  # noqa: BLE001
            pass
        libs = []
        for name in ("numpy", "cv2", "numba", "onnxruntime", "PIL"):
            m = sys.modules.get(name)
            v = getattr(m, "__version__", None) if m else None
            if v:
                libs.append(f"{name} {v}")
        if libs:
            parts.append(", ".join(libs))
        get().info("запуск: %s", " | ".join(parts))
    except Exception:  # noqa: BLE001 - рядок для зручності, ніколи не заважає запуску
        pass


def install_hooks():
    """необроблені винятки (головний і фонові потоки) -> у журнал"""
    lg = get()
    old = sys.excepthook

    def hook(t, v, tb):
        lg.error("необроблена помилка", exc_info=(t, v, tb))
        old(t, v, tb)
    sys.excepthook = hook

    def thook(a):
        lg.error("помилка в потоці %s", getattr(a.thread, "name", "?"), exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
    threading.excepthook = thook
