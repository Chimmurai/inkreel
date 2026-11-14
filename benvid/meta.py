"""Відомості про програму для вікна «Про програму»: версія, автор, посилання, місце під донат.
Перед релізом достатньо заповнити цей файл - нічого більше міняти не треба: порожні поля просто не показуються."""
from __future__ import annotations

from . import BETA_LABEL as _BETA, __version__ as _VER

VERSION = _BETA or _VER     # у бета-каналі показуємо мітку бети, у стабільній - номер версії

APP = "InkReel"
AUTHOR = "Raimex"       # ім'я або нік автора
SITE = "https://inkreel.raimex.space"   # сторінка проєкту (https://…)
CONTACT = "raimex.crucible@gmail.com"   # пошта чи нік для зв'язку
# Офіційні джерела: лише ці три (сайт, код і релізи, сторінка в Store). Вони ж зазначені на сайті й у README, щоб клон легко відрізнити від оригіналу.
REPO = "https://github.com/Chimmurai/inkreel"
STORE = "https://apps.microsoft.com/detail/9NW62SPG91ZR"


def short(url: str) -> str:
    """адреса для показу в тексті: без схеми й кінцевого слеша"""
    return url.split("://", 1)[-1].rstrip("/")


def links() -> list[tuple[str, str]]:
    """[(підпис українською, адреса)] для меню в футері; порожні адреси пропускаємо"""
    return [(n, u) for n, u in (("Сайт InkReel", SITE), ("Код і релізи на GitHub", REPO), ("Сторінка в Microsoft Store", STORE)) if u]

# «Пригости автора печивом»: посилання на сервіси донатів - [(назва, адреса)], напр. [("Ko-fi", "https://ko-fi.com/…"),
# ("Monobank", "https://send.monobank.ua/jar/…")]. Одне посилання - кнопка відкриває його одразу, кілька - показує меню.
# Поки список порожній, кнопка в футері неактивна («скоро»), але місце під неї зарезервоване.
DONATE: list[tuple[str, str]] = []
DONATE_ENABLED = True   # False - кнопки немає зовсім (напр., якщо правила магазину забороняють посилання на донати)
# Microsoft Store (політика 10.8.2): донати користувачів треба приймати через платіжний API Microsoft або «безпечний сторонній» API;
# звичайне посилання на сторінку донату може не пройти сертифікацію. Тому в збірці зі Store кнопку за замовчуванням сховано
# (у звичайній збірці - exe чи інсталятор - вона працює). Змінюйте на True лише коли певні, що це дозволено.
DONATE_IN_STORE = False


def donate_enabled() -> bool:
    """чи показувати кнопку донату в цій збірці"""
    from . import store
    return DONATE_ENABLED and (DONATE_IN_STORE or not store.is_packaged())
