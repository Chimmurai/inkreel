"""Де малювати спливаючі вікна (підказки, хмаринка котушки): на моніторі, де зараз курсор, а не лише на першому екрані.

Раніше підказка обмежувала позицію розміром ПЕРШОГО екрана (winfo_screenwidth/-height). Коли вікно програми стоїть на другому
моніторі (праворуч, ліворуч чи вище за основний), точка курсора лежить за межами «першого екрана», і підказка притискалась до його
краю - тобто вилітала десь далеко від кнопки, «за вікно». Тут межі беруться з реального монітора під точкою (Windows: робоча
область без панелі завдань), а де цього не вміємо - з усього віртуального екрана.
"""
from __future__ import annotations

import os

Rect = tuple[int, int, int, int]            # (лівий, верхній, правий, нижній) у координатах Tk (віртуальний екран)


def _monitor_work_area(x: int, y: int) -> Rect | None:
    """робоча область монітора, якому належить (або найближчий до) точка (x, y); None - не Windows чи не вийшло"""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD)]
        u = ctypes.WinDLL("user32")                              # власний екземпляр: типи аргументів не чіпають чужих викликів
        u.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
        u.MonitorFromPoint.restype = ctypes.c_void_p
        u.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(MONITORINFO)]
        u.GetMonitorInfoW.restype = wintypes.BOOL
        hm = u.MonitorFromPoint(wintypes.POINT(int(x), int(y)), 2)           # MONITOR_DEFAULTTONEAREST
        if not hm:
            return None
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        if not u.GetMonitorInfoW(hm, ctypes.byref(mi)):
            return None
        r = mi.rcWork
        if r.right - r.left < 50 or r.bottom - r.top < 50:
            return None
        return int(r.left), int(r.top), int(r.right), int(r.bottom)
    except Exception:  # noqa: BLE001
        return None


def work_area(widget, x: int, y: int) -> Rect:
    """межі, у які має влізти спливаюче вікно, що з'являється біля точки (x, y) екрана: монітор під точкою; якщо Windows
    його не назвала - увесь віртуальний екран (усі монітори разом); у крайньому разі - перший екран"""
    a = _monitor_work_area(x, y)
    if a is not None:
        return a
    try:
        vx, vy = widget.winfo_vrootx(), widget.winfo_vrooty()
        vw, vh = widget.winfo_vrootwidth(), widget.winfo_vrootheight()
        if vw > 0 and vh > 0:
            return vx, vy, vx + vw, vy + vh
    except Exception:  # noqa: BLE001
        pass
    return 0, 0, int(widget.winfo_screenwidth()), int(widget.winfo_screenheight())


def _fit(v: int, lo: int, hi: int) -> int:
    """v у [lo, hi]; якщо вікно більше за область (hi < lo) - притиснути до лівого/верхнього краю, щоб було видно початок"""
    return lo if hi < lo else min(max(v, lo), hi)


def near_pointer(px: int, py: int, w: int, h: int, area: Rect, dx: int = 14, dy: int = 22, gap: int = 10, pad: int = 4) -> tuple[int, int]:
    """де поставити вікно w×h біля курсора (px, py): правіше й нижче за нього; не влізло праворуч - притиснути до правого краю
    області; не влізло знизу - поставити над курсором. Результат завжди в межах області (наскільки дозволяє розмір)"""
    l, t, r, b = area
    x = px + dx
    if x + w > r - pad:
        x = r - pad - w
    y = py + dy
    if y + h > b - pad:
        y = py - h - gap
    return _fit(x, l + pad, r - pad - w), _fit(y, t + pad, b - pad - h)


def under_widget(wx: int, wy: int, ww: int, wh: int, w: int, h: int, area: Rect, off: int = 18, gap: int = 6, pad: int = 4) -> tuple[int, int]:
    """де поставити вікно w×h під віджетом (його кут wx, wy й розміри ww, wh): майже по центру віджета; не влізло знизу - над ним"""
    l, t, r, b = area
    x = wx + ww // 2 - off
    y = wy + wh + gap
    if y + h > b - pad:
        y = wy - h - gap
    return _fit(x, l + pad, r - pad - w), _fit(y, t + pad, b - pad - h)
