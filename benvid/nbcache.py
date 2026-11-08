"""Кеш скомпільованих ядер numba у зібраному exe.

Чому окремо. Стандартні «локатори» кешу numba вимагають, щоб існував .py-файл функції. У exe (PyInstaller) вихідних файлів немає, тож numba мовчки
йшла до єдиного запасного локатора: теки користувача за шляхом, що залежить від місця встановлення програми (у MSIX воно містить номер версії), а штамп
свіжості кешу бере з файлу, що ЗАПУЩЕНО (sys.executable). У нас два exe: InkReel.exe (вікно) та InkReel-Worker.exe (процеси обробки). Якби їхні байти чи
час змінення різнилися, вікно й обробники вважали б кеш одне одного застарілим і перекомпільовували ядра при кожному експорті (десятки секунд на
кожен процес). Тут: кеш завжди в теці програми (NUMBA_CACHE_DIR, див. log.prepare_env), підтека залежить лише від пакета, штамп - вміст ГОЛОВНОГО exe,
однаковий для обох процесів і новий для кожної збірки (після оновлення старі файли просто перезаписуються на тих самих іменах, сміття не лишається).

У звичайному запуску (python -m benvid) нічого не змінюється: цей модуль працює лише в зібраному exe. INKREEL_NBCACHE=0 - вимкнути."""
from __future__ import annotations

import hashlib
import os
import sys

_DONE = False
_STAMP: str | None = None


def stamp() -> str:
    """штамп свіжості кешу: SHA-256 головного exe (він змінюється з кожною збіркою й однаковий для вікна та обробників)"""
    global _STAMP
    if _STAMP is None:
        here = os.path.dirname(os.path.abspath(sys.executable))
        path = sys.executable
        for name in ("InkReel.exe", "InkReel"):
            p = os.path.join(here, name)
            if os.path.isfile(p):
                path = p
                break
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            _STAMP = h.hexdigest()
        except OSError:
            from . import __version__
            _STAMP = "v" + __version__
    return _STAMP


def cache_root() -> str:
    d = os.environ.get("NUMBA_CACHE_DIR")
    if d:
        return d
    from . import log
    return os.path.join(log.app_dir(), "cache")


def install() -> bool:
    """додає наш локатор першим у список numba. Викликати ПІСЛЯ import numba і ДО створення функцій із cache=True. True - локатор активний"""
    global _DONE
    if _DONE:
        return True
    if not getattr(sys, "frozen", False) or os.environ.get("INKREEL_NBCACHE") == "0":
        return False
    try:
        from numba.core import caching as C
        base = C._CacheLocator
        impl = getattr(C, "CacheImpl", None) or C._CacheImpl
        classes = impl._locator_classes
    except Exception:  # noqa: BLE001 - інша будова numba: лишаємо стандартну поведінку
        return False

    class FrozenLocator(base):
        def __init__(self, py_func, py_file):
            self._py_file = py_file
            self._lineno = py_func.__code__.co_firstlineno
            pkg = (getattr(py_func, "__module__", "") or "benvid").rpartition(".")[0] or "benvid"
            self._cache_path = os.path.join(cache_root(), pkg)

        def get_cache_path(self):
            return self._cache_path

        def get_source_stamp(self):
            return stamp()

        def get_disambiguator(self):
            return str(self._lineno)

        @classmethod
        def from_function(cls, py_func, py_file):
            if not (getattr(py_func, "__module__", "") or "").startswith("benvid"):
                return None                               # чужі функції - стандартним шляхом
            self = cls(py_func, py_file)
            try:
                self.ensure_cache_path()
            except OSError:
                return None                               # тека недоступна для запису: далі спробує запасний локатор numba
            return self

    if not any(getattr(c, "__name__", "") == "FrozenLocator" for c in classes):
        classes.insert(0, FrozenLocator)
    _DONE = True
    return True
