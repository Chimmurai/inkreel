"""Кеш ядер numba у зібраному exe (benvid/nbcache.py): працює без вихідних .py-файлів, лежить у теці програми, а штамп свіжості однаковий для вікна
(InkReel.exe) і процесів обробки (InkReel-Worker.exe) - інакше кожен експорт перекомпілював би ядра в усіх процесах."""
import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
numba = pytest.importorskip("numba")
from benvid import nbcache  # noqa: E402


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """вдає зібраний exe: поряд лежать два різні за вмістом файли; кеш - у тимчасовій теці. Після тесту наш локатор прибирається зі списку numba"""
    exe_dir = tmp_path / "app"
    exe_dir.mkdir()
    (exe_dir / "InkReel.exe").write_bytes(b"main-" * 100)
    (exe_dir / "InkReel-Worker.exe").write_bytes(b"worker-" * 100)
    cache = tmp_path / "cache"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "InkReel.exe"))
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(cache))
    monkeypatch.delenv("INKREEL_NBCACHE", raising=False)
    monkeypatch.setattr(nbcache, "_DONE", False)
    monkeypatch.setattr(nbcache, "_STAMP", None)
    from numba.core import caching
    impl = getattr(caching, "CacheImpl", None) or caching._CacheImpl
    before = list(impl._locator_classes)
    yield exe_dir, cache, impl
    impl._locator_classes[:] = before


def _compile_without_source(name="_nbtest"):
    """функція numba з cache=True, чий «файл» не існує - як у exe, де .py немає"""
    src = "import numba\n@numba.njit(cache=True)\ndef f(x):\n    return x + 1\n"
    ns = {"__name__": "benvid." + name}
    exec(compile(src, f"/nonexistent/_internal/benvid/{name}.py", "exec"), ns)
    return ns["f"]


def test_stamp_is_the_same_for_window_and_worker(frozen, monkeypatch):
    exe_dir, _, _ = frozen
    want = hashlib.sha256((exe_dir / "InkReel.exe").read_bytes()).hexdigest()
    assert nbcache.stamp() == want
    monkeypatch.setattr(sys, "executable", str(exe_dir / "InkReel-Worker.exe"))
    monkeypatch.setattr(nbcache, "_STAMP", None)
    assert nbcache.stamp() == want, "обробник мусить бачити той самий штамп, що й вікно"


def test_stamp_changes_with_a_new_build(frozen, monkeypatch):
    exe_dir, _, _ = frozen
    old = nbcache.stamp()
    (exe_dir / "InkReel.exe").write_bytes(b"new build")
    monkeypatch.setattr(nbcache, "_STAMP", None)
    assert nbcache.stamp() != old


def test_cache_is_written_in_the_app_folder_without_source_files(frozen):
    _, cache, _ = frozen
    assert nbcache.install() is True
    f = _compile_without_source()
    assert f(1) == 2
    assert list((cache / "benvid").glob("*.nbi")), "індекс кешу має лежати в теці програми, а не в ~/.cache/numba"
    assert list((cache / "benvid").glob("*.nbc"))


def test_cache_is_reused_by_another_process_with_the_same_stamp(frozen, monkeypatch):
    exe_dir, cache, impl = frozen
    nbcache.install()
    _compile_without_source("_nbreuse")(1)
    idx = next((cache / "benvid").glob("_nbreuse*.nbi"))
    first = idx.read_bytes()
    # «обробник»: інший exe, але той самий штамп -> індекс лишається чинним і не переписується
    monkeypatch.setattr(sys, "executable", str(exe_dir / "InkReel-Worker.exe"))
    monkeypatch.setattr(nbcache, "_STAMP", None)
    f2 = _compile_without_source("_nbreuse")
    assert f2(5) == 6
    assert idx.read_bytes() == first


def test_foreign_functions_are_left_to_numba(frozen):
    nbcache.install()
    from numba.core import caching
    impl = getattr(caching, "CacheImpl", None) or caching._CacheImpl
    loc = impl._locator_classes[0]
    assert loc.__name__ == "FrozenLocator"

    def other():
        return 1
    other.__module__ = "somebody_else"
    assert loc.from_function(other, "/x/y.py") is None


def test_not_frozen_changes_nothing(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(nbcache, "_DONE", False)
    from numba.core import caching
    impl = getattr(caching, "CacheImpl", None) or caching._CacheImpl
    before = list(impl._locator_classes)
    assert nbcache.install() is False
    assert impl._locator_classes == before


def test_can_be_switched_off(frozen, monkeypatch):
    monkeypatch.setenv("INKREEL_NBCACHE", "0")
    assert nbcache.install() is False


def test_install_twice_adds_one_locator(frozen):
    _, _, impl = frozen
    nbcache.install()
    nbcache._DONE = False
    nbcache.install()
    assert [c.__name__ for c in impl._locator_classes].count("FrozenLocator") == 1
