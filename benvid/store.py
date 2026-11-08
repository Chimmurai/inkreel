"""Збірка з Microsoft Store (пакет MSIX): як програма дізнається, що вона «в пакеті», і що з цього випливає.

У пакета є ідентичність (Package Family Name). Звичайна збірка (exe, інсталятор) її не має. Різниця для користувача мала:
  · теку програми Windows робить лише для читання, а запис у %LOCALAPPDATA% непомітно перенаправляє в
    %LOCALAPPDATA%\\Packages\\<PFN>\\LocalCache\\Local\\… - програма цього не відчуває, а Провідник бачить файл лише за справжнім
    шляхом, тому в повідомленнях (де лежить журнал) показуємо його, а не логічний;
  · правила Store (політика 10.8.2) щодо донатів - див. meta.donate_enabled().
Сам код перевірки - чиста функція над kernel32 (у тестах підміняється), поза Windows завжди «не в пакеті»."""
from __future__ import annotations

import os

ERROR_INSUFFICIENT_BUFFER = 122          # пакет є: Windows просить буфер потрібного розміру
APPMODEL_ERROR_NO_PACKAGE = 15700        # процес без ідентичності пакета

_PFN: str | None | bool = False          # False - ще не питали; None - не в пакеті


def package_family_name(kernel=None) -> str | None:
    """ім'я сімейства пакета (PFN) або None, якщо це не збірка зі Store. kernel - об'єкт з GetCurrentPackageFamilyName (у тестах підміна)"""
    if kernel is None:
        if os.name != "nt":
            return None
        try:
            import ctypes
            kernel = ctypes.windll.kernel32
        except Exception:  # noqa: BLE001
            return None
    try:
        import ctypes
        n = ctypes.c_uint32(0)
        rc = kernel.GetCurrentPackageFamilyName(ctypes.byref(n), None)
        if rc != ERROR_INSUFFICIENT_BUFFER or n.value <= 0:
            return None                                     # 15700 (немає пакета) або будь-яка інша несподіванка
        buf = ctypes.create_unicode_buffer(n.value)
        if kernel.GetCurrentPackageFamilyName(ctypes.byref(n), buf) != 0:
            return None
        return buf.value or None
    except Exception:  # noqa: BLE001
        return None


def pfn() -> str | None:
    """те саме, але питаємо Windows один раз за запуск"""
    global _PFN
    if _PFN is False:
        _PFN = package_family_name()
    return _PFN


def is_packaged() -> bool:
    return pfn() is not None


def visible_path(path: str, family: str | None = None) -> str:
    """шлях до файлу в %LOCALAPPDATA%, як його бачить Провідник: у пакеті запис туди перенаправлено в Packages\\<PFN>\\LocalCache\\Local.
    Поза пакетом (і для шляхів не з %LOCALAPPDATA%) повертає шлях без змін"""
    family = family or pfn()
    base = os.environ.get("LOCALAPPDATA")
    if not family or not base:
        return path
    ap, ab = os.path.abspath(path), os.path.abspath(base)
    if os.path.normcase(ap) == os.path.normcase(ab) or os.path.normcase(ap).startswith(os.path.normcase(ab) + os.sep):
        return os.path.join(ab, "Packages", family, "LocalCache", "Local", os.path.relpath(ap, ab))
    return path
