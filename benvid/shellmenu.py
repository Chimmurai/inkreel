"""Пункти контекстного меню Провідника без інсталятора - для збірки «папкою» (dist\\InkReel) чи перенесеної програми:

    InkReel.exe --register-menu      додати «Відкрити в InkReel» і «Зробити чорно-біле (InkReel)» для фото та GIF
    InkReel.exe --unregister-menu    прибрати їх

Пишеться лише в HKCU\\Software\\Classes (права адміністратора не потрібні) - ті самі ключі, що й у installer.iss
(tests/test_installer.py стежить, щоб набори збігалися). Перенесли теку з програмою - запустіть --register-menu ще раз:
шлях до exe записується в команду. Правий клік по самому файлу в Windows 11 показує ці пункти під «Показати додаткові параметри».
У збірці з Microsoft Store (пакет MSIX) це не працює: Windows переспрямовує запис у реєстр у приватний розділ пакета, тож --register-menu там відмовляє."""
from __future__ import annotations

import os
import sys

from . import i18n, media, store
from .i18n import tr

BASE = "Software\\Classes\\SystemFileAssociations"
SKIP = frozenset({".ico"})            # значки: 1-бітне перетворення не має сенсу
# (ім'я ключа дієслова, підпис - це й ключ перекладу, що йде в командному рядку перед шляхом файлу)
VERBS = (
    ("InkReel.Open", "Відкрити в InkReel", ""),
    ("InkReel.Quick", "Зробити чорно-біле (InkReel)", "--quick "),
)
SHCNE_ASSOCCHANGED = 0x08000000


def extensions() -> list[str]:
    """розширення, для яких додаємо пункти: усі фото й GIF/APNG, крім значків; порядок - як у media.py"""
    return [e for e in list(media.IMAGE_EXT) + list(media.GIF_EXT) if e not in SKIP]


def command(exe: str, flags: str) -> str:
    return f'"{exe}" {flags}"%1"'


def entries(exe: str, caption=None) -> list[tuple[str, str, str]]:
    """усі значення реєстру: (ключ відносно HKCU, ім'я значення - '' це типове, текст) для кожного розширення й дієслова.
    caption(дієслово, підпис) -> текст пункту; за замовчуванням - переклад на поточну мову (інсталятор підставляє свої {cm:…})"""
    caption = caption or (lambda verb, label: tr(label))
    out = []
    for ext in extensions():
        for verb, label, flags in VERBS:
            key = f"{BASE}\\{ext}\\shell\\{verb}"
            out.append((key, "MUIVerb", caption(verb, label)))
            out.append((key, "Icon", f'"{exe}",0'))
            out.append((key + "\\command", "", command(exe, flags)))
    return out


def _winreg():
    import winreg
    return winreg


def register(exe: str, reg=None) -> int:
    """записати пункти меню для exe; повертає кількість записаних значень. reg - модуль winreg (у тестах підміна)"""
    reg = reg or _winreg()
    n = 0
    for key, name, value in entries(exe):
        with reg.CreateKeyEx(reg.HKEY_CURRENT_USER, key, 0, reg.KEY_SET_VALUE) as k:
            reg.SetValueEx(k, name, 0, reg.REG_SZ, value)
        n += 1
    return n


def _delete_tree(reg, path: str) -> int:
    """видалити ключ разом із підключами (winreg.DeleteKey цього не вміє); 1 - був і видалений, 0 - такого й не було"""
    try:
        with reg.OpenKeyEx(reg.HKEY_CURRENT_USER, path, 0, reg.KEY_READ) as k:
            subs, i = [], 0
            while True:
                try:
                    subs.append(reg.EnumKey(k, i))
                    i += 1
                except OSError:
                    break
    except FileNotFoundError:
        return 0
    for s in subs:
        _delete_tree(reg, path + "\\" + s)
    try:
        reg.DeleteKey(reg.HKEY_CURRENT_USER, path)
    except FileNotFoundError:
        return 0
    return 1


def unregister(reg=None) -> int:
    """прибрати лише наші дієслова (InkReel.Open / InkReel.Quick); чужі пункти й ключі розширень не чіпаємо. -> скільки ключів видалено"""
    reg = reg or _winreg()
    n = 0
    for ext in extensions():
        for verb, _label, _flags in VERBS:
            n += _delete_tree(reg, f"{BASE}\\{ext}\\shell\\{verb}")
    return n


def registered(reg=None) -> str | None:
    """команда швидкої дії, що зараз записана в реєстрі (для .png), або None, якщо пунктів немає"""
    reg = reg or _winreg()
    try:
        with reg.OpenKeyEx(reg.HKEY_CURRENT_USER, f"{BASE}\\.png\\shell\\InkReel.Quick\\command", 0, reg.KEY_READ) as k:
            return str(reg.QueryValueEx(k, "")[0])
    except OSError:
        return None


# ---------------------------------------------------------------- виклик із командного рядка
def _notify():
    """сказати Провіднику, що асоціації змінились: меню перечитається без перезапуску"""
    try:
        import ctypes
        ctypes.windll.shell32.SHChangeNotify(SHCNE_ASSOCCHANGED, 0, None, None)
    except Exception:  # noqa: BLE001
        pass


def _say(text: str, error: bool = False):
    """exe зібрано без консолі, тому результат - вікно Windows; поза Windows - звичайний друк"""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text, "InkReel", 0x10 if error else 0x40)
            return
        except Exception:  # noqa: BLE001
            pass
    print(text)


def main(argv: list[str]) -> int:
    try:
        from .quick import load_prefs
        i18n.set_lang(load_prefs().lang)
    except Exception:  # noqa: BLE001
        pass
    if os.name != "nt":
        _say(tr("Меню Провідника є лише у Windows."), True)
        return 1
    if store.is_packaged():                  # у пакеті запис у HKCU\\Software\\Classes Windows переспрямовує в приватний розділ - Провідник його не бачить
        _say(tr("Збірка з Microsoft Store не може додати пункти в меню Провідника: відкривайте файли через «Відкрити за допомогою»."), True)
        return 1
    if argv[:1] == ["--register-menu"]:
        if not getattr(sys, "frozen", False):
            _say(tr("Пункти меню додаються із зібраної програми: запустіть InkReel.exe --register-menu (збірка - scripts\\build_exe.bat)."), True)
            return 1
        try:
            register(sys.executable)
        except OSError as e:
            _say(tr("Не вдалося записати в реєстр: {e}", e=e), True)
            return 1
        _notify()
        _say(tr("Готово: у меню правої кнопки для фото й GIF з'явились «Відкрити в InkReel» і «Зробити чорно-біле (InkReel)». "
                "У Windows 11 вони під «Показати додаткові параметри»."))
        return 0
    try:
        unregister()
    except OSError as e:
        _say(tr("Не вдалося записати в реєстр: {e}", e=e), True)
        return 1
    _notify()
    _say(tr("Пункти InkReel прибрано з меню правої кнопки."))
    return 0
