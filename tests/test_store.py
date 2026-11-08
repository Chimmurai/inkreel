"""Збірка для Microsoft Store: визначення пакета MSIX, шлях журналу, донати (політика 10.8.2). Перевірки самого пакета - нижче."""
from __future__ import annotations

import ctypes
import os

import pytest

from benvid import log, meta, store


# ---------------------------------------------------------------- визначення пакета
class FakeKernel:
    """підроблений kernel32.GetCurrentPackageFamilyName: перший виклик (без буфера) повертає потрібну довжину, другий - заповнює"""

    def __init__(self, name=None, first=None, second=0):
        self.name, self.first, self.second, self.calls = name, first, second, 0

    def GetCurrentPackageFamilyName(self, plen, buf):
        self.calls += 1
        n = plen._obj
        if buf is None:
            if self.name is None:
                return self.first if self.first is not None else store.APPMODEL_ERROR_NO_PACKAGE
            n.value = len(self.name) + 1
            return self.first if self.first is not None else store.ERROR_INSUFFICIENT_BUFFER
        if self.second == 0:
            buf.value = self.name
        return self.second


@pytest.fixture
def clean_cache():
    old = store._PFN
    store._PFN = False
    yield
    store._PFN = old


def test_not_packaged_returns_none():
    assert store.package_family_name(FakeKernel()) is None


def test_packaged_returns_family_name():
    k = FakeKernel("12345Author.InkReel_8wekyb3d8bbwe")
    assert store.package_family_name(k) == "12345Author.InkReel_8wekyb3d8bbwe" and k.calls == 2


@pytest.mark.parametrize("kernel", [FakeKernel("X_1", first=5), FakeKernel("X_1", second=87), FakeKernel(None, first=0),
                                    object()])
def test_unexpected_answers_mean_not_packaged(kernel):
    assert store.package_family_name(kernel) is None


@pytest.mark.skipif(os.name == "nt", reason="на Windows відповідь залежить від того, як запущено тест")
def test_outside_windows_is_never_packaged():
    assert store.package_family_name() is None


def test_pfn_is_asked_once(monkeypatch, clean_cache):
    calls = []
    monkeypatch.setattr(store, "package_family_name", lambda kernel=None: calls.append(1) or "A_b")
    assert store.pfn() == "A_b" and store.is_packaged() and store.pfn() == "A_b"
    assert len(calls) == 1


def test_pfn_none_is_cached_too(monkeypatch, clean_cache):
    calls = []
    monkeypatch.setattr(store, "package_family_name", lambda kernel=None: calls.append(1))
    assert store.pfn() is None and not store.is_packaged()
    assert len(calls) == 1


# ---------------------------------------------------------------- шлях журналу
def test_visible_path_redirects_localappdata_into_the_package(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    p = tmp_path / "InkReel" / "inkreel.log"
    assert store.visible_path(str(p), "Fam_1") == str(tmp_path / "Packages" / "Fam_1" / "LocalCache" / "Local" / "InkReel" / "inkreel.log")


def test_visible_path_leaves_everything_else_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    other = str(tmp_path / "Documents" / "a.png")
    assert store.visible_path(other, "Fam_1") == other
    assert store.visible_path(str(tmp_path / "localx" / "f"), "Fam_1") == str(tmp_path / "localx" / "f")      # лише справжня підтека
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(store, "_PFN", None)                                                      # не в пакеті
    inside = str(tmp_path / "InkReel" / "x")
    assert store.visible_path(inside) == inside                                                   # без пакета - без змін


def test_visible_path_without_localappdata_is_unchanged(monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert store.visible_path("/x/y", "Fam_1") == "/x/y"


def test_log_shown_path_uses_package_location(tmp_path, monkeypatch, clean_cache):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(log, "app_dir", lambda: str(tmp_path / "InkReel"))
    monkeypatch.setattr(store, "_PFN", "Fam_1")
    assert log.shown_path() == str(tmp_path / "Packages" / "Fam_1" / "LocalCache" / "Local" / "InkReel" / "inkreel.log")
    assert log.log_path() == str(tmp_path / "InkReel" / "inkreel.log")                           # а писати програма й далі «логічним» шляхом
    monkeypatch.setattr(store, "_PFN", None)
    assert log.shown_path() == log.log_path()


# ---------------------------------------------------------------- донати
@pytest.mark.parametrize("enabled,in_store,packaged,expected", [
    (True, False, False, True),          # звичайна збірка: кнопка є
    (True, False, True, False),          # Store за замовчуванням: сховано
    (True, True, True, True),            # автор свідомо дозволив у Store
    (False, True, True, False),          # вимкнено цілком - ніде немає
    (False, False, False, False),
])
def test_donate_button_rules(monkeypatch, clean_cache, enabled, in_store, packaged, expected):
    monkeypatch.setattr(meta, "DONATE_ENABLED", enabled)
    monkeypatch.setattr(meta, "DONATE_IN_STORE", in_store)
    monkeypatch.setattr(store, "_PFN", "Fam_1" if packaged else None)
    assert meta.donate_enabled() is expected


def test_shipped_defaults_hide_donate_in_the_store():
    assert meta.DONATE_IN_STORE is False and meta.DONATE_ENABLED is True
