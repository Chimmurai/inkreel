"""Попіксельне порівняння малювання Tk і Qt (вигляд 1:1).

    python tools/ui_diff.py primitives       -> калібрування: той самий набір примітивів на tk.Canvas і на qtui.tkcanvas.TkCanvas
    python tools/ui_diff.py widgets          -> піксельні віджети (кнопка, повзунок, панель якості, прогрес, смужка) у Tk і в Qt

Знімки й різниця: <тека результатів>/diff_*.png (ліворуч Tk, у центрі Qt, праворуч різниця; ×3). Вихід 0, якщо відмінних пікселів не більше порога."""
from __future__ import annotations

import ctypes
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.environ.get("UI_DIFF_OUT", os.path.join(ROOT, "reports"))

try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:  # noqa: BLE001
    pass

BG, FG, DIM = "#000000", "#ffffff", "#9a9a9a"
FONT = ("Consolas", 10, "bold")
FONT_S = ("Consolas", 9)


def sheet(c):
    """набір примітивів: усе, що малюють піксельні віджети"""
    R, L, O, P, T = c.create_rectangle, c.create_line, c.create_oval, c.create_polygon, c.create_text
    R(0, 0, 59, 29, fill=BG, outline=FG)                                   # кнопка: контур по x0 і x1 включно
    R(64, 0, 123, 29, fill=FG, outline=FG)                                 # залита
    R(2, 2, 25, 17, outline=FG, dash=(1, 2))                               # пунктир наведення
    R(130, 0, 145, 15, fill=FG, outline="")                                # заливка без контуру
    R(150, 0, 165, 15, fill=DIM, outline=DIM)
    L(0, 40, 60, 40, fill=FG)                                              # лінії
    L(0, 44, 60, 44, fill=FG, width=2)
    L(0, 50, 60, 50, fill=FG, width=2, dash=(3, 3))
    L(70, 36, 70, 70, fill=FG)
    L(76, 36, 110, 60, fill=FG)
    L(0, 56, 60, 56, fill=DIM, dash=(1, 1))
    O(0, 70, 18, 88, outline=FG, fill=FG, width=2)                         # гуртки
    O(24, 72, 38, 86, outline=FG, fill=BG, width=2)
    O(44, 76, 50, 82, outline=FG, fill=FG)
    P(60, 70, 60 + 14, 79, 60, 88, fill=FG, outline=FG)                    # трикутник і ромб
    P(90, 70, 100, 79, 90, 88, 80, 79, outline=FG, fill=BG, width=2)
    R(120, 40, 150, 60, fill=BG, outline=FG)
    R(122, 42, 140, 58, fill=FG, outline=FG)
    T(4, 100, text="Відкрити", fill=FG, font=FONT, anchor="w")             # текст
    T(80, 100, text="швидко", fill=DIM, font=FONT_S, anchor="sw")
    T(170, 118, text="ЯКІСНО", fill=DIM, font=FONT_S, anchor="se")
    T(100, 112, text="Abc 123", fill=FG, font=FONT, anchor="center")


def grab_tk(w, h, draw):
    import tkinter as tk
    from PIL import ImageGrab
    root = tk.Tk()
    root.geometry(f"{w}x{h}+40+40")
    root.attributes("-topmost", True)
    c = tk.Canvas(root, width=w, height=h, bg=BG, highlightthickness=0, bd=0)
    c.pack()
    draw(c)
    root.update()
    time.sleep(0.4)
    root.update()
    x, y = c.winfo_rootx(), c.winfo_rooty()
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h)).convert("RGB")
    root.destroy()
    return np.asarray(img)


def grab_qt(w, h, draw):
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from benvid.qtui.tkcanvas import TkCanvas
    c = TkCanvas(bg=BG)
    c.resize(w, h)
    draw(c)
    c.show()
    app.processEvents()
    im = c.grab().toImage().convertToFormat(__import__("PySide6.QtGui", fromlist=["QImage"]).QImage.Format_RGB888)
    ptr = im.constBits()
    arr = np.frombuffer(ptr, np.uint8).reshape(im.height(), im.bytesPerLine())[:, :im.width() * 3].reshape(im.height(), im.width(), 3).copy()
    c.close()
    return arr


def compare(name, a, b, threshold=0):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w].astype(int), b[:h, :w].astype(int)
    d = np.abs(a - b).sum(-1)
    n = int((d > 24).sum())
    from PIL import Image
    diff = np.zeros((h, w, 3), np.uint8)
    diff[d > 24] = (255, 0, 0)
    diff[(d > 0) & (d <= 24)] = (90, 90, 0)
    both = np.concatenate([a.astype(np.uint8), np.full((h, 2, 3), 128, np.uint8), b.astype(np.uint8), np.full((h, 2, 3), 128, np.uint8), diff], axis=1)
    os.makedirs(OUT, exist_ok=True)
    Image.fromarray(both).resize((both.shape[1] * 3, both.shape[0] * 3), Image.NEAREST).save(os.path.join(OUT, f"diff_{name}.png"))
    print(f"{name}: {n} відмінних пікселів із {h * w} ({100.0 * n / (h * w):.2f}%)")
    return n <= threshold


def widget_cases():
    """[(назва, ширина, висота, фабрика Tk(gui, parent), фабрика Qt(pix, parent))] - однакові стани віджетів з обох боків"""
    def btn(**kw):
        def tk_(g, p):
            im = kw.get("img")
            b = g.PixButton(p, text=kw.get("text", ""), width=kw["w"], height=kw.get("h", 30), icon=kw.get("icon"), image=g._img(im) if im else None)
            b.on, b.enabled, b.hover, b._pressed = kw.get("on", False), kw.get("enabled", True), kw.get("hover", False), kw.get("pressed", False)
            b.draw()
            return b

        def qt_(x, p, assets=None):
            from benvid.qtui import assets as A
            b = x.PixButton(p, text=kw.get("text", ""), width=kw["w"], height=kw.get("h", 30), icon=kw.get("icon"), image=A.img(im) if (im := kw.get("img")) else None)
            b.on, b.enabled, b.hover, b._pressed = kw.get("on", False), kw.get("enabled", True), kw.get("hover", False), kw.get("pressed", False)
            b.draw()
            return b
        return tk_, qt_

    cases = [
        ("button normal", 144, 30, *btn(text="Відкрити", w=144)),
        ("button hover", 144, 30, *btn(text="Зберегти фото", w=144, hover=True)),
        ("button pressed/on", 144, 30, *btn(text="Стоп", w=144, on=True)),
        ("button disabled", 144, 30, *btn(text="Стоп", w=144, enabled=False)),
        ("button + image", 144, 30, *btn(text="Відкрити", w=144, img="open")),
        ("button icon play", 38, 30, *btn(w=38, icon="play")),
        ("button icon pause", 38, 30, *btn(w=38, icon="pause")),
        ("button icon swap", 38, 30, *btn(w=38, icon="swap")),
        ("button icon snd", 38, 30, *btn(w=38, icon="snd")),
        ("button icon cookie+text", 288, 30, *btn(text="Пригостити автора печивом", w=288, icon="cookie")),
    ]

    def slider(marks=None, ranges=(), v=40, drag=False, h=34, hot=None):
        def make(mod, p):
            s = mod.PixSlider(p, 0, 100, height=h)
            s.v, s.drag = v, drag
            s.ranges = list(ranges)
            s.marks = list(marks) if marks else None
            if marks and hot is not None:
                s._hot = hot
            s.draw()
            return s
        return make, make

    cases += [
        ("slider plain", 300, 34, *slider()),
        ("slider ranges", 300, 34, *slider(ranges=[(0, 30), (45, 70)], v=20)),
        ("slider marks full", 300, 34, *slider(marks=[0, 100], ranges=[(0, 60)], v=30)),
        ("slider marks limited", 300, 34, *slider(marks=[20, 70], ranges=[(0, 90)], v=45, drag=True)),
    ]

    def qbar(v=4, hov=None, branch="IRNM1", on=True):
        def make(mod, p):
            q = mod.QualityBar(p, branch=branch)
            q.v, q.hov, q.branch_on = v, hov, on
            q.draw()
            return q
        return make, make

    cases += [
        ("quality bar 4", 290, 44, *qbar(4)),
        ("quality bar branch on", 290, 44, *qbar(6)),
        ("quality bar hover", 290, 44, *qbar(4, hov=2)),
        ("quality bar branch off", 290, 44, *qbar(5, on=False)),
    ]

    def prog(f):
        def make(mod, p):
            q = mod.PixProgress(p)
            q.f = f
            q.draw()
            return q
        return make, make

    def scroll(first, last, hot=False):
        def make(mod, p):
            s = mod.PixScroll(p, width=12)
            s.first, s.last, s._hot = first, last, hot
            s.draw()
            return s
        return make, make

    cases += [("progress 0.5", 290, 18, *prog(0.5)), ("progress full", 290, 18, *prog(1.0)),
              ("scroll", 12, 200, *scroll(0.1, 0.4)), ("scroll lit", 12, 200, *scroll(0.5, 0.9, True))]
    return cases


def grab_widgets(cases):
    import tkinter as tk
    from PIL import ImageGrab
    from benvid import gui
    gap = 8
    W = max(c[1] for c in cases) + 16
    H = sum(c[2] + gap for c in cases) + 8
    root = tk.Tk()
    root.geometry(f"{W}x{H}+20+20")
    root.attributes("-topmost", True)
    fr = tk.Frame(root, bg=BG)
    fr.pack(fill="both", expand=True)
    y = 4
    pos = []
    for name, w, h, tk_make, _ in cases:
        wd = tk_make(gui, fr)
        wd.place(x=8, y=y, width=w, height=h)
        pos.append((name, 8, y, w, h))
        y += h + gap
    root.update()
    time.sleep(0.5)
    root.update()
    for _, _, _, _, _ in []:
        pass
    x0, y0 = fr.winfo_rootx(), fr.winfo_rooty()
    img = np.asarray(ImageGrab.grab(bbox=(x0, y0, x0 + W, y0 + H)).convert("RGB"))
    root.destroy()
    return img, pos, (W, H)


def grab_qt_widgets(cases, size):
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication, QWidget
    app = QApplication.instance() or QApplication([])
    from benvid.qtui import pix
    W, H = size
    host = QWidget()
    host.setStyleSheet(f"background:{BG}")
    host.resize(W, H)
    y = 4
    keep = []
    for name, w, h, _, qt_make in cases:
        wd = qt_make(pix, host)
        wd.setGeometry(8, y, w, h)
        wd.draw()
        keep.append(wd)
        y += h + 8
    host.show()
    app.processEvents()
    im = host.grab().toImage().convertToFormat(QImage.Format_RGB888)
    arr = np.frombuffer(im.constBits(), np.uint8).reshape(im.height(), im.bytesPerLine())[:, :im.width() * 3].reshape(im.height(), im.width(), 3).copy()
    host.close()
    return arr


def _isolated_settings(ui):
    import json
    import tempfile
    from benvid import engine
    tmp = tempfile.mkdtemp()
    engine.SETTINGS_FILE = os.path.join(tmp, "settings.json")
    engine._OLD = os.path.join(tmp, "old.json")
    with open(engine.SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump({"ui": ui}, f)


def grab_screen_tk(w, h, ui, media=None, wait=6.0):
    import tkinter  # noqa: F401
    from PIL import ImageGrab
    _isolated_settings(ui)
    import random
    random.seed(7)                                       # однакова «Порада» на порожньому екрані з обох боків
    from benvid import guide
    guide.tip = lambda n=-1: 3
    from benvid import gui
    app = gui.App(media) if media else gui.App()
    app.geometry(f"{w}x{h}+30+30")
    app.attributes("-topmost", True)
    end = time.time() + wait
    while time.time() < end:
        app.update()
        time.sleep(0.02)
    x, y = app.winfo_rootx(), app.winfo_rooty()
    img = np.asarray(ImageGrab.grab(bbox=(x, y, x + app.winfo_width(), y + app.winfo_height())).convert("RGB"))
    app.destroy()
    return img


def grab_screen_qt(w, h, ui, media=None, wait=6.0):
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    _isolated_settings(ui)
    import random
    random.seed(7)
    from benvid import guide
    guide.tip = lambda n=-1: 3
    app = QApplication.instance() or QApplication([])
    from benvid.qtui.window import MainWindow
    win = MainWindow(media)
    win.resize(w, h)
    win.show()
    end = time.time() + wait
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    im = win.centralWidget().grab().toImage().convertToFormat(QImage.Format_RGB888)
    arr = np.frombuffer(im.constBits(), np.uint8).reshape(im.height(), im.bytesPerLine())[:, :im.width() * 3].reshape(im.height(), im.width(), 3).copy()
    win.close()
    return arr


def main(argv):
    what = argv[1] if len(argv) > 1 else "primitives"
    if what == "screen":
        ui = {"lang": "uk", "palette": "Класика", "size": "Авто", "method": 4}
        w, h = 1280, 860
        tk_img = grab_screen_tk(w, h, ui)
        qt_img = grab_screen_qt(w, h, ui)
        ok = compare("screen", tk_img, qt_img, threshold=0)
        return 0 if ok else 1
    if what == "primitives":
        w, h = 180, 124
        ok = compare("primitives", grab_tk(w, h, sheet), grab_qt(w, h, sheet))
        return 0 if ok else 1
    if what == "widgets":
        cases = widget_cases()
        tk_img, pos, size = grab_widgets(cases)
        qt_img = grab_qt_widgets(cases, size)
        total = 0
        for name, x, y, w, h in pos:
            a, b = tk_img[y:y + h, x:x + w].astype(int), qt_img[y:y + h, x:x + w].astype(int)
            n = int((np.abs(a - b).sum(-1) > 24).sum())
            total += n
            print(f"  {name:28s} {n:4d} відмінних пікселів із {w * h}")
        compare("widgets", tk_img, qt_img)
        return 0 if total == 0 else 1
    print("невідомий режим:", what)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
