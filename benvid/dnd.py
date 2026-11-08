"""Перетягування файлів у вікно (Windows) без сторонніх пакетів: підписуємось на WM_DROPFILES через ctypes.
Імена файлів приходять у Unicode (DragQueryFileW), тож кирилиця й емодзі в назвах працюють -
пакет windnd віддає байти в ANSI-кодуванні, і українські назви ним розпізнаються неправильно."""
from __future__ import annotations

import os

_KEEP: list = []                 # посилання, щоб збирач сміття не знищив зворотні виклики


def hook(widget, callback) -> bool:
    """callback(list[str]) - шляхи скинутих файлів (з головного потоку Tk). True, якщо підписатись вдалося"""
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes as wt

        user32, shell32 = ctypes.windll.user32, ctypes.windll.shell32
        LRESULT = ctypes.c_ssize_t
        WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)
        user32.GetWindowLongPtrW.restype = ctypes.c_void_p
        user32.GetWindowLongPtrW.argtypes = [wt.HWND, ctypes.c_int]
        user32.SetWindowLongPtrW.restype = ctypes.c_void_p
        user32.SetWindowLongPtrW.argtypes = [wt.HWND, ctypes.c_int, ctypes.c_void_p]
        user32.CallWindowProcW.restype = LRESULT
        user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
        user32.GetParent.restype = wt.HWND
        user32.GetParent.argtypes = [wt.HWND]
        shell32.DragQueryFileW.argtypes = [ctypes.c_void_p, wt.UINT, wt.LPWSTR, wt.UINT]
        shell32.DragQueryFileW.restype = wt.UINT
        shell32.DragFinish.argtypes = [ctypes.c_void_p]
        shell32.DragAcceptFiles.argtypes = [wt.HWND, wt.BOOL]
        WM_DROPFILES, WM_COPYGLOBALDATA, GWLP_WNDPROC = 0x233, 0x49, -4

        def sub(hwnd):
            old = user32.GetWindowLongPtrW(hwnd, GWLP_WNDPROC)

            def proc(h, msg, wp, lp):
                if msg == WM_DROPFILES:
                    paths = []
                    try:
                        n = shell32.DragQueryFileW(wp, 0xFFFFFFFF, None, 0)
                        for i in range(n):
                            ln = shell32.DragQueryFileW(wp, i, None, 0)
                            buf = ctypes.create_unicode_buffer(ln + 1)
                            shell32.DragQueryFileW(wp, i, buf, ln + 1)
                            paths.append(buf.value)
                    finally:
                        shell32.DragFinish(wp)
                    try:
                        callback(paths)
                    except Exception:  # noqa: BLE001
                        pass
                    return 0
                return user32.CallWindowProcW(old, h, msg, wp, lp)

            cb = WNDPROC(proc)
            _KEEP.append(cb)
            user32.SetWindowLongPtrW(hwnd, GWLP_WNDPROC, ctypes.cast(cb, ctypes.c_void_p).value)
            shell32.DragAcceptFiles(hwnd, True)
            for m in (WM_DROPFILES, WM_COPYGLOBALDATA, 0x233):      # дозволити й від імені адміністратора (UIPI)
                try:
                    user32.ChangeWindowMessageFilterEx(hwnd, m, 1, None)
                except Exception:  # noqa: BLE001
                    pass

        widget.update_idletasks()
        inner = int(widget.winfo_id())
        top = user32.GetParent(inner)
        for h in {inner, top}:
            if h:
                sub(h)
        return True
    except Exception:  # noqa: BLE001
        return False
