"""Знімає з Tk (GDI) точні пікселі кіл для qtui.tkcanvas: GDI малює Ellipse «квадратнішими» за Qt, і щоб панель якості збігалась 1:1, форми беремо з Tk.

    python tools/gen_gdi_ovals.py      ->  benvid/qtui/gdi_ovals.py  (діаметри 1..40, товщини контуру 1 і 2; кожна форма: рядки бітових масок)

Для кожного (d, width): `all` - уся фігура із заливкою (контур включно), `ring` - лише контур. Малюється create_oval(x, y, x+d, y+d) (у Tk коло
займає d+1 піксель по кожній осі: x1 включно)."""
from __future__ import annotations

import ctypes
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:  # noqa: BLE001
    pass

DMAX = 40
CELL = 48


def main() -> int:
    import tkinter as tk
    from PIL import ImageGrab
    shapes = [(d, w) for d in range(1, DMAX + 1) for w in (1, 2)]
    cols = 10
    rows = -(-len(shapes) * 2 // cols)
    W, H = cols * CELL, rows * CELL
    root = tk.Tk()
    root.geometry(f"{W}x{H}+20+20")
    root.attributes("-topmost", True)
    c = tk.Canvas(root, width=W, height=H, bg="#000000", highlightthickness=0, bd=0)
    c.pack()
    pos = {}
    k = 0
    for d, w in shapes:
        for variant, fill in (("all", "#ffffff"), ("ring", "")):
            cx, cy = (k % cols) * CELL + 4, (k // cols) * CELL + 4
            c.create_oval(cx, cy, cx + d, cy + d, outline="#ffffff", fill=fill, width=w)
            pos[(d, w, variant)] = (cx, cy)
            k += 1
    root.update()
    time.sleep(0.6)
    root.update()
    x, y = c.winfo_rootx(), c.winfo_rooty()
    img = np.asarray(ImageGrab.grab(bbox=(x, y, x + W, y + H)).convert("L"))
    root.destroy()
    out = {}
    for (d, w, variant), (cx, cy) in pos.items():
        side = d + 3
        m = img[cy - 2:cy - 2 + side + 1, cx - 2:cx - 2 + side + 1] > 128         # поле навколо: форма може вийти за x1
        ys, xs = np.nonzero(m)
        if not len(ys):
            continue
        y0, x0 = ys.min(), xs.min()
        rows_ = []
        for yy in range(y0, ys.max() + 1):
            bits = 0
            for xx in range(x0, xs.max() + 1):
                if m[yy, xx]:
                    bits |= 1 << (xx - x0)
            rows_.append(bits)
        out.setdefault((d, w), {})[variant] = (int(x0) - 2, int(y0) - 2, int(xs.max() - x0 + 1), [int(b) for b in rows_])
    path = os.path.join(ROOT, "benvid", "qtui", "gdi_ovals.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write('"""Піксельні форми кіл GDI (Tk) для qtui.tkcanvas; створено tools/gen_gdi_ovals.py, руками не правити.\n'
                'OVALS[(діаметр, товщина)] = {"all": (dx, dy, ширина, [рядки-маски]), "ring": (...)}: dx, dy - зсув лівого верхнього піксельного кута форми від (x0, y0)."""\n\n')
        f.write("OVALS = {\n")
        for key in sorted(out):
            f.write(f"    {key!r}: {out[key]!r},\n")
        f.write("}\n")
    print("ok:", len(out), "форм ->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
