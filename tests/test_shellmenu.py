"""Меню правої кнопки без інсталятора (benvid/shellmenu.py): запис і прибирання ключів реєстру перевіряємо на підробленому winreg."""
from __future__ import annotations

import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import sync_installer as si                                              # noqa: E402
from benvid import i18n, media, shellmenu                                # noqa: E402

EXE = "C:\\Apps\\InkReel\\InkReel.exe"
PNG = f"{shellmenu.BASE}\\.png\\shell"


class FakeReg:
    """мінімум winreg: ключі за шляхом без урахування регістру; DeleteKey, як і справжній, не видаляє ключ із підключами"""
    HKEY_CURRENT_USER = "HKCU"
    KEY_SET_VALUE = 2
    KEY_READ = 0x20019
    REG_SZ = 1

    class Key:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def __init__(self, fail_on_create=False):
        self.keys: dict[str, dict[str, str]] = {}
        self.fail_on_create = fail_on_create

    def CreateKeyEx(self, root, path, reserved=0, access=0):
        assert root == self.HKEY_CURRENT_USER
        if self.fail_on_create:
            raise PermissionError("реєстр лише для читання")
        parts = path.split("\\")
        for i in range(1, len(parts) + 1):
            self.keys.setdefault("\\".join(parts[:i]).lower(), {})
        return self.Key(path)

    def SetValueEx(self, key, name, reserved, typ, value):
        assert typ == self.REG_SZ and isinstance(value, str)
        self.keys[key.path.lower()][name] = value

    def OpenKeyEx(self, root, path, reserved=0, access=0):
        if path.lower() not in self.keys:
            raise FileNotFoundError(path)
        return self.Key(path)

    def EnumKey(self, key, i):
        pre = key.path.lower() + "\\"
        subs = sorted({k[len(pre):].split("\\")[0] for k in self.keys if k.startswith(pre)})
        if i >= len(subs):
            raise OSError("більше немає")
        return subs[i]

    def DeleteKey(self, root, path):
        p = path.lower()
        if p not in self.keys:
            raise FileNotFoundError(path)
        if any(k.startswith(p + "\\") for k in self.keys):
            raise PermissionError("ключ має підключі")
        del self.keys[p]

    def QueryValueEx(self, key, name):
        return self.keys[key.path.lower()][name], self.REG_SZ

    def value(self, path, name=""):
        return self.keys[path.lower()][name]


@pytest.fixture(autouse=True)
def _restore_lang():
    old = i18n.lang()
    i18n.set_lang(i18n.UK)
    yield
    i18n.set_lang(old)


def test_extensions_are_photos_and_gifs_without_icons():
    ex = shellmenu.extensions()
    assert len(ex) == len(set(ex))
    assert set(ex) == (set(media.IMAGE_EXT) | set(media.GIF_EXT)) - {".ico"}
    for e in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".tif"):
        assert e in ex


def test_entries_have_three_values_per_verb_and_extension():
    es = shellmenu.entries(EXE)
    assert len(es) == len(shellmenu.extensions()) * len(shellmenu.VERBS) * 3
    d = {(k.lower(), n): v for k, n, v in es}
    assert d[(PNG.lower() + "\\inkreel.open", "MUIVerb")] == "Відкрити в InkReel"
    assert d[(PNG.lower() + "\\inkreel.quick", "MUIVerb")] == "Зробити чорно-біле (InkReel)"
    assert d[(PNG.lower() + "\\inkreel.open\\command", "")] == f'"{EXE}" "%1"'
    assert d[(PNG.lower() + "\\inkreel.quick\\command", "")] == f'"{EXE}" --quick "%1"'
    assert d[(PNG.lower() + "\\inkreel.quick", "Icon")] == f'"{EXE}",0'


def test_paths_with_spaces_and_cyrillic_stay_quoted():
    exe = "D:\\Мої програми\\Ink Reel\\InkReel.exe"
    for k, n, v in shellmenu.entries(exe):
        if k.endswith("\\command"):
            assert v.startswith(f'"{exe}" ') and v.endswith(' "%1"')
        elif n == "Icon":
            assert v == f'"{exe}",0'


def test_register_writes_everything_and_twice_changes_nothing():
    reg = FakeReg()
    n = shellmenu.register(EXE, reg)
    assert n == len(shellmenu.entries(EXE))
    state = {k: dict(v) for k, v in reg.keys.items()}
    assert reg.value(PNG + "\\InkReel.Quick\\command") == f'"{EXE}" --quick "%1"'
    assert shellmenu.register(EXE, reg) == n
    assert reg.keys == state


def test_registering_again_from_a_new_folder_updates_the_path():
    reg = FakeReg()
    shellmenu.register(EXE, reg)
    assert shellmenu.registered(reg) == f'"{EXE}" --quick "%1"'
    new = "E:\\Portable\\InkReel.exe"
    shellmenu.register(new, reg)
    assert shellmenu.registered(reg) == f'"{new}" --quick "%1"'
    assert not any(EXE in v for vals in reg.keys.values() for v in vals.values())


def test_unregister_removes_only_our_verbs():
    reg = FakeReg()
    for path, name, val in [(PNG + "\\open\\command", "", "paint.exe"), (PNG + "\\OtherApp", "MUIVerb", "Інша програма"),
                            (f"{shellmenu.BASE}\\.png", "Foo", "bar"), ("Software\\Classes\\.png", "", "pngfile")]:
        with reg.CreateKeyEx(reg.HKEY_CURRENT_USER, path, 0, reg.KEY_SET_VALUE) as k:
            reg.SetValueEx(k, name, 0, reg.REG_SZ, val)
    shellmenu.register(EXE, reg)
    assert shellmenu.registered(reg)
    n = shellmenu.unregister(reg)
    assert n == len(shellmenu.extensions()) * len(shellmenu.VERBS)
    assert shellmenu.registered(reg) is None
    assert not [k for k in reg.keys if "inkreel" in k]
    assert reg.value(PNG + "\\open\\command") == "paint.exe"                    # чужі пункти цілі
    assert reg.value(PNG + "\\OtherApp", "MUIVerb") == "Інша програма"
    assert reg.value(f"{shellmenu.BASE}\\.png", "Foo") == "bar" and reg.value("Software\\Classes\\.png") == "pngfile"
    assert shellmenu.unregister(reg) == 0                                       # повторно - нічого й не було


def test_registered_is_none_on_a_clean_registry():
    assert shellmenu.registered(FakeReg()) is None


def test_captions_follow_language_and_match_the_installer():
    with open(si.ISS, encoding="utf-8-sig") as f:
        secs = si.sections(f.read())
    cm = {}
    for ln in secs["CustomMessages"]:
        key, _, val = ln.partition("=")
        cm[key] = val
    for lang, code in (("ukrainian", i18n.UK), ("english", i18n.EN)):
        i18n.set_lang(code)
        d = {(k.lower(), n): v for k, n, v in shellmenu.entries(EXE)}
        assert d[(PNG.lower() + "\\inkreel.open", "MUIVerb")] == cm[f"{lang}.CtxOpen"]
        assert d[(PNG.lower() + "\\inkreel.quick", "MUIVerb")] == cm[f"{lang}.CtxQuick"]


@pytest.mark.skipif(os.name == "nt", reason="поза Windows реєстру немає")
def test_main_outside_windows_explains_and_fails(capsys):
    assert shellmenu.main(["--register-menu"]) == 1
    assert "Windows" in capsys.readouterr().out


# ---------------------------------------------------------------- main() на підробленому Windows
@pytest.fixture
def fake_windows(monkeypatch):
    reg = FakeReg()
    said, notified = [], []
    monkeypatch.setattr(shellmenu, "os", types.SimpleNamespace(name="nt"))
    monkeypatch.setattr(shellmenu, "_winreg", lambda: reg)
    monkeypatch.setattr(shellmenu, "_notify", lambda: notified.append(1))
    monkeypatch.setattr(shellmenu, "_say", lambda text, error=False: said.append((text, error)))
    monkeypatch.setattr(sys, "executable", EXE)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    return reg, said, notified


def test_main_register_and_unregister(fake_windows):
    reg, said, notified = fake_windows
    assert shellmenu.main(["--register-menu"]) == 0
    assert shellmenu.registered(reg) == f'"{EXE}" --quick "%1"' and notified == [1]
    assert said[-1][1] is False and "InkReel" in said[-1][0]
    assert shellmenu.main(["--unregister-menu"]) == 0
    assert shellmenu.registered(reg) is None and notified == [1, 1] and said[-1][1] is False


def test_main_refuses_to_register_a_python_script(fake_windows, monkeypatch):
    reg, said, notified = fake_windows
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert shellmenu.main(["--register-menu"]) == 1
    assert not reg.keys and not notified and said[-1][1] is True and "register-menu" in said[-1][0]


def test_main_reports_registry_errors_instead_of_crashing(monkeypatch, fake_windows):
    reg, said, notified = fake_windows
    reg.fail_on_create = True
    assert shellmenu.main(["--register-menu"]) == 1
    assert said[-1][1] is True and "реєстр" in said[-1][0] and not notified


def test_main_refuses_inside_a_store_package(fake_windows, monkeypatch):
    """у пакеті MSIX запис у HKCU\\Software\\Classes потрапляє в приватний розділ - краще чесно відмовити, ніж написати «Готово»"""
    from benvid import store
    reg, said, notified = fake_windows
    monkeypatch.setattr(store, "_PFN", "Fam_1")
    for flag in ("--register-menu", "--unregister-menu"):
        assert shellmenu.main([flag]) == 1
    assert not reg.keys and not notified
    assert said[-1][1] is True and "Microsoft Store" in said[-1][0]


def test_cli_flags_are_wired_in_main_module():
    import re
    with open(os.path.join(ROOT, "benvid", "__main__.py"), encoding="utf-8") as f:
        src = f.read()
    assert '"--register-menu"' in src and '"--unregister-menu"' in src and re.search(r"from \.shellmenu import main", src)
    with open(os.path.join(ROOT, "inkreel.spec"), encoding="utf-8") as f:
        assert '"benvid.shellmenu"' in f.read()
