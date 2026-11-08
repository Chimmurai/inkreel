"""Правила для вікон (без екрана, розбір коду): програма не повинна ставати «поверх усіх» і не повинна ховати свої діалоги.
Користувач скаржився, що інші вікна відкриваються під InkReel, - тож усе, що може перекрити чужі вікна, має бути на виду й нечисленним."""
from __future__ import annotations

import ast
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benvid")


def _tree(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return ast.parse(f.read())


def _funcs(tree):
    """{id вузла: ім'я функції, у якій він лежить}"""
    owner = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(fn):
                owner[id(n)] = fn.name
    return owner


def _topmost_true(tree):
    out = []
    own = _funcs(tree)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("attributes", "wm_attributes"):
            a = n.args
            if len(a) >= 2 and isinstance(a[0], ast.Constant) and a[0].value == "-topmost" and isinstance(a[1], ast.Constant) and a[1].value is True:
                out.append((n.lineno, own.get(id(n), "<module>")))
    return out


def test_only_two_places_may_put_a_window_on_top_and_both_take_it_back():
    """«поверх усіх» - лише на мить: передній план при повторному запуску (bring_front, знімає _untop) і вікно швидкої дії
    (QuickWindow.__init__, знімає untop). Підказки й хмаринки - не topmost"""
    assert [fn for _, fn in _topmost_true(_tree("gui.py"))] == ["bring_front"]
    assert [fn for _, fn in _topmost_true(_tree("quick.py"))] == ["__init__"]
    gui = open(os.path.join(ROOT, "gui.py"), encoding="utf-8").read()
    assert "self._untop" in gui and "def _untop" in gui
    quick = open(os.path.join(ROOT, "quick.py"), encoding="utf-8").read()
    assert "def untop" in quick and "r.after(1200, untop)" in quick


def test_every_dialog_has_its_parent_window():
    """діалог без parent на Windows може відкритись під головним вікном; з parent він завжди над своїм власником"""
    tree = _tree("gui.py")
    bad, found = [], 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                and n.func.value.id in ("filedialog", "messagebox") and n.func.attr.startswith(("ask", "show")):
            found += 1
            if "parent" not in {k.arg for k in n.keywords}:
                bad.append((n.lineno, f"{n.func.value.id}.{n.func.attr}"))
    assert found >= 5 and not bad, bad


def test_background_threads_run_below_the_interface():
    """потоки подачі кадрів і експорту опускають свій пріоритет: інтерфейс не голодує, коли ядра зайняті"""
    pl = open(os.path.join(ROOT, "player.py"), encoding="utf-8").read()
    gui = open(os.path.join(ROOT, "gui.py"), encoding="utf-8").read()
    assert "lower_thread_priority" in pl and "lower_thread_priority" in gui
    from benvid import procs
    procs.lower_thread_priority()                              # на Linux - нічого не робить, але й не падає


def test_crisp_text_is_harmless_off_windows_and_can_be_switched_off(monkeypatch):
    """режим «GDI scaling» (чіткий текст на моніторах із масштабом) - лише Windows; не-Windows і INKREEL_DPI=0 нічого не чіпають і не падають"""
    from benvid import procs
    monkeypatch.setattr(procs.os, "name", "posix")
    assert procs.crisp_text() is False
    monkeypatch.setattr(procs.os, "name", "nt")
    monkeypatch.setenv("INKREEL_DPI", "0")
    assert procs.crisp_text() is False
    monkeypatch.delenv("INKREEL_DPI")
    assert procs.crisp_text() is False, "на ПК без Windows-бібліотек ctypes кидає виняток - функція його ковтає"
