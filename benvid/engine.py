"""Налаштування інтерфейсу (запам'ятовуються між запусками): %LOCALAPPDATA%\\InkReel\\settings.json."""
from __future__ import annotations

import json
import os

from .log import app_dir

SETTINGS_FILE = os.path.join(app_dir(), "settings.json")
_OLD = os.path.join(os.path.expanduser("~"), ".benvid.json")


_STR_KEYS = ("size", "fmt", "quality", "palette", "guide", "lang", "install_lang")
_BOOL_KEYS = ("audio", "pal_inv", "glyph")
_INT_KEYS = {"method": (1, 6), "gdens3": (1, 3), "volume": (0, 100)}
_FLOAT_KEYS = {"fps": (0.0, 1000.0)}


def _num(v, lo, hi, as_int):
    """число з файлу налаштувань -> у межах lo..hi; не число, булеве, NaN та безкінечність -> None"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except (OverflowError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    f = min(float(hi), max(float(lo), f))
    return int(round(f)) if as_int else f


def clean_ui(ui) -> dict:
    """значення з settings.json -> типи й межі, які розуміє вікно. Усе, що не годиться (чужий тип, NaN, число поза межами, зайве довгий рядок), прибирається, і
    програма бере типове значення: файл могли відредагувати вручну, його міг лишити інший випуск програми, а після оновлення з Microsoft Store налаштування
    зберігаються, тож нова версія мусить пережити будь-що з нього. Невідомі ключі лишаються (їх могла дописати новіша версія)"""
    if not isinstance(ui, dict):
        return {}
    out = dict(ui)
    for k in _STR_KEYS:
        if k in out and not (isinstance(out[k], str) and len(out[k]) <= 200):
            del out[k]
    for k in _BOOL_KEYS:
        if k in out and not isinstance(out[k], bool):
            v = out[k]
            if isinstance(v, int) and v in (0, 1):
                out[k] = bool(v)
            else:
                del out[k]
    for k, (lo, hi) in _INT_KEYS.items():
        if k in out:
            v = _num(out[k], lo, hi, True)
            if v is None:
                del out[k]
            else:
                out[k] = v
    for k, (lo, hi) in _FLOAT_KEYS.items():
        if k in out:
            v = _num(out[k], lo, hi, False)
            if v is None:
                del out[k]
            else:
                out[k] = v
    return out


def load_settings() -> dict:
    for p in (SETTINGS_FILE, _OLD):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                d["ui"] = clean_ui(d.get("ui"))
                return d
        except Exception:  # noqa: BLE001
            continue
    return {"ui": {}}


def save_settings(d: dict):
    """запис через тимчасовий файл: збій посеред запису не зіпсує старі налаштування"""
    tmp = SETTINGS_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        os.replace(tmp, SETTINGS_FILE)
    except Exception:  # noqa: BLE001
        pass
