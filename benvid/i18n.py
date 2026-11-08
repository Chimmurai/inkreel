"""Локалізація інтерфейсу: українська (вихідна мова) й англійська.

Ключ перекладу - саме українське формулювання: у коді пишемо tr("Відкрити"), а англійське лежить у benvid/lang_en.py.
Нема перекладу - показується українське (нічого не ламається, лише лишається недоперекладене; тест tests/test_i18n.py
не дає такому дійти до релізу). Підстановки - звичайний str.format: tr("Готово: {n}", n=5).
Мова одна на весь процес: set_lang() міняє її миттєво, а вікно просто перемальовує свої підписи (gui.App._apply_lang).
Перша мова - за мовою системи (українська -> українська, решта -> англійська); вибір користувача зберігається."""
from __future__ import annotations

import os
import sys

UK, EN = "uk", "en"
LANGS = (UK, EN)
LABEL = {UK: "UA", EN: "EN"}          # що пишемо на кнопці перемикача

_lang = UK
_table: dict[str, str] | None = None


def _en() -> dict[str, str]:
    global _table
    if _table is None:
        from .lang_en import EN as t
        _table = t
    return _table


def lang() -> str:
    return _lang


def set_lang(code: str | None) -> str:
    global _lang
    _lang = code if code in LANGS else UK
    return _lang


def other(code: str | None = None) -> str:
    """друга мова (для кнопки-перемикача)"""
    return EN if (code or _lang) == UK else UK


def install_lang() -> str:
    """мова, яку користувач обрав в інсталяторі (Windows, ключ реєстру), або "" """
    if sys.platform != "win32":
        return ""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\InkReel") as k:
            v = str(winreg.QueryValueEx(k, "InstallLang")[0]).lower()
        return v if v in (UK, EN) else ""
    except Exception:  # noqa: BLE001
        return ""


def detect() -> str:
    """мова системи -> UK або EN. Windows: мова інтерфейсу користувача (LANGID); інакше - змінні середовища / locale"""
    code = ""
    if sys.platform == "win32":
        il = install_lang()
        if il:
            return il
        try:
            import ctypes
            return UK if ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF == 0x22 else EN     # 0x22 - українська
        except Exception:  # noqa: BLE001
            pass
    for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        v = os.environ.get(var, "")
        if v:
            code = v.lower().replace("-", "_").split("_")[0]
            break
    if not code:
        try:
            import locale
            code = (locale.getlocale()[0] or "").lower().split("_")[0]
        except Exception:  # noqa: BLE001
            code = ""
    return UK if code in ("uk", "ukrainian") else EN


def tr(text: str, **kw) -> str:
    """переклад поточною мовою (ключ - українське формулювання); kw - підстановки str.format"""
    s = _en().get(text, text) if _lang == EN else text
    return s.format(**kw) if kw else s


def _uk_form(n: int) -> int:
    """українські форми множини: 1 кадр, 2 кадри, 5 кадрів"""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return 0
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return 1
    return 2


def tn(n: int, forms: str, **kw) -> str:
    """число з іменником: tn(5, "{n} кадр|{n} кадри|{n} кадрів") -> «5 кадрів»; в англійській форм дві (1 frame | 2 frames)"""
    if _lang == EN:
        parts = _en().get(forms, forms).split("|")
        s = parts[0] if (int(n) == 1 or len(parts) < 2) else parts[1]
    else:
        parts = forms.split("|")
        s = parts[min(_uk_form(n), len(parts) - 1)]
    return s.format(n=n, **kw)
