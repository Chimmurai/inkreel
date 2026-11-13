"""Локалізація UA/EN: жоден видимий рядок не лишився без англійської, підстановки збігаються, множина, гайд.

Ключ перекладу - українське формулювання з коду. Тест читає вихідники (ast), тож новий рядок у вікні без запису в
benvid/lang_en.py вилітає тут, а не в англомовного користувача."""
from __future__ import annotations

import ast
import re
import string
import sys
from pathlib import Path

import pytest

from benvid import guide, i18n, meta
from benvid.lang_en import EN

PKG = Path(__file__).resolve().parent.parent / "benvid"
CYR = re.compile("[А-ЩЬЮЯЄІЇҐа-щьюяєіїґ]")

# модулі, чиї рядки бачить користувач
UI_MODULES = ["gui", "core", "palette", "video", "media", "audio", "player", "autosize", "eggs", "quick", "shellmenu"]
# кирилиця, якій переклад не потрібен: журнал для автора, службові порівняння, старі назви з налаштувань
IGNORE = {
    "помилка інтерфейсу", "автовибір розміру", "не вдалося відкрити %s", "перетягування файлів недоступне",
    "помилка", "помилка експорту", "швидка дія: %s",   # лише в лог-файлі
    "встанови",                                      # префікс статусу audio, з яким порівнюємо
    "Ризограф",                                      # стара назва палітри (palette.LEGACY)
    "х",                                             # кирилична «х» у «320х180» -> латинська x
}
# у цих змінних guide.py тексти вже лежать у двох мовах поруч
GUIDE_BILINGUAL = {"PHILOSOPHY", "TAGLINE", "SLOGAN", "TIPS", "_TEXT", "KEYS"}


@pytest.fixture(autouse=True)
def _restore_lang():
    old = i18n.lang()
    yield
    i18n.set_lang(old)


def _parse(mod: str) -> ast.Module:
    return ast.parse((PKG / f"{mod}.py").read_text(encoding="utf-8"))


def _docstrings(tree: ast.AST) -> set[int]:
    ids = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = n.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                ids.add(id(b[0].value))
    return ids


def _fields(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f is not None}


def _all_constants() -> set[str]:
    out: set[str] = set()
    for p in PKG.glob("*.py"):
        if p.name == "lang_en.py":
            continue
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.add(n.value)
    return out


# ---------------------------------------------------------------- повнота
@pytest.mark.parametrize("mod", UI_MODULES + ["guide"])
def test_every_visible_string_has_english(mod):
    tree = _parse(mod)
    skip = _docstrings(tree)
    if mod == "guide":
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in GUIDE_BILINGUAL for t in n.targets):
                skip |= {id(x) for x in ast.walk(n)}
    parent = {id(c): n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    missing, fstr = [], []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Constant) and isinstance(n.value, str) and CYR.search(n.value)) or id(n) in skip:
            continue
        if n.value in IGNORE or n.value.startswith("показ:"):          # «показ:» - діагностика плавності відтворення в журналі
            continue
        if isinstance(parent.get(id(n)), ast.JoinedStr):
            fstr.append((n.lineno, n.value))
        elif n.value not in EN:
            missing.append((n.lineno, n.value))
    assert not fstr, (f"{mod}: кирилиця в f-рядку не перекладається - зроби tr('… {{x}} …', x=…): "
                      + "; ".join(f"{ln}: {s!r}" for ln, s in fstr))
    assert not missing, f"{mod}: немає англійського перекладу (додай у benvid/lang_en.py): " + "; ".join(
        f"{ln}: {s[:70]!r}" for ln, s in missing)


def test_no_stale_translations():
    used = _all_constants()
    stale = [k for k in EN if k not in used]
    assert not stale, "переклади без відповідного рядка в коді (змінився текст?): " + "; ".join(repr(k[:70]) for k in stale)


def test_english_values_are_english():
    same_ok = {"Українська", "English", "⚠ {note}"}
    bad = [k for k, v in EN.items() if not v.strip() or (CYR.search(v) and k not in same_ok)]
    assert not bad, "порожній або недоперекладений англійський текст: " + "; ".join(repr(k[:60]) for k in bad)


def test_names_are_covered():
    from benvid import core, media, palette, video
    names = list(palette.NAMES) + list(core.METHODS.values()) + list(core.METHOD_INFO.values()) + list(video.FORMATS) \
        + list(video.QUALITY) + list(media.GIF_FORMATS) + list(media.STILL_FORMATS)
    names = [n for n in names if CYR.search(n)]
    assert not [n for n in names if n not in EN], [n for n in names if n not in EN]


# ---------------------------------------------------------------- підстановки й множина
def test_placeholders_match():
    bad = []
    for k, v in EN.items():
        want = _fields(k)
        for form in v.split("|") if "{n}" in k else [v]:
            if _fields(form) != want:
                bad.append(k)
                break
    assert not bad, "підстановки {…} в англійському тексті не збігаються з ключем: " + "; ".join(repr(k[:60]) for k in bad)


def test_every_translation_formats():
    """жоден англійський текст не падає на .format() із підстановками з ключа"""
    for k, v in EN.items():
        kw = {f: 1 for f in _fields(k)}
        for form in v.split("|") if "{n}" in k else [v]:
            form.format(**kw)


def test_plural_forms_ukrainian():
    f = "{n} кадр|{n} кадри|{n} кадрів"
    i18n.set_lang("uk")
    got = {n: i18n.tn(n, f) for n in (0, 1, 2, 4, 5, 11, 12, 14, 21, 22, 25, 101, 111, 112, 121)}
    assert got[0] == "0 кадрів" and got[1] == "1 кадр" and got[2] == "2 кадри" and got[4] == "4 кадри"
    assert got[5] == "5 кадрів" and got[11] == "11 кадрів" and got[12] == "12 кадрів" and got[14] == "14 кадрів"
    assert got[21] == "21 кадр" and got[22] == "22 кадри" and got[25] == "25 кадрів"
    assert got[101] == "101 кадр" and got[111] == "111 кадрів" and got[112] == "112 кадрів" and got[121] == "121 кадр"


def test_plural_forms_english():
    f = "{n} кадр|{n} кадри|{n} кадрів"
    i18n.set_lang("en")
    assert [i18n.tn(n, f) for n in (0, 1, 2, 5)] == ["0 frames", "1 frame", "2 frames", "5 frames"]


def test_tr_fallback_and_formatting():
    i18n.set_lang("en")
    assert i18n.tr("цього рядка немає в словнику") == "цього рядка немає в словнику"      # недоперекладене - не падаємо
    assert i18n.tr("Відкрити") == "Open"
    assert i18n.tr("Помилка: {e}", e="x {y}") == "Error: x {y}"                           # фігурні дужки в значенні безпечні
    i18n.set_lang("uk")
    assert i18n.tr("Відкрити") == "Відкрити"
    assert i18n.set_lang("fr") == "uk"                                                    # невідома мова -> українська
    assert i18n.other("uk") == "en" and i18n.other("en") == "uk"


@pytest.mark.skipif(sys.platform == "win32", reason="на Windows мову дає ОС")
@pytest.mark.parametrize("env,want", [
    ({"LANG": "uk_UA.UTF-8"}, "uk"), ({"LANG": "en_US.UTF-8"}, "en"), ({"LC_ALL": "uk_UA", "LANG": "en_US"}, "uk"),
    ({"LANG": "de_DE.UTF-8"}, "en"), ({"LANGUAGE": "uk"}, "uk"),
])
def test_detect_from_environment(monkeypatch, env, want):
    for v in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        monkeypatch.delenv(v, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert i18n.detect() == want


# ---------------------------------------------------------------- гайд і «Про програму»
@pytest.mark.parametrize("code", ["uk", "en"])
def test_guide_sections_exist_in_both_languages(code):
    i18n.set_lang(code)
    for sec in guide.SECTIONS:
        t = guide.body(sec)
        assert t.strip(), (code, sec)
        assert guide.title(sec).strip()
    assert guide.philosophy().strip() and guide.TAGLINE[code].strip()
    if code == "en":
        for sec in guide.SECTIONS:                               # англійський гайд без українських абзаців
            assert not CYR.search(guide.title(sec)), sec
            body = guide.body(sec)
            assert not CYR.search(body.replace(str(meta.AUTHOR), "")), (sec, CYR.search(body))


def test_guide_bilingual_data_complete():
    for sec, texts in guide._TEXT.items():
        assert set(texts) == {i18n.UK, i18n.EN}, sec
    assert set(guide.PHILOSOPHY) == set(guide.TAGLINE) == {i18n.UK, i18n.EN}
    for row in guide.KEYS:
        assert len(row) == 3 and all(c.strip() for c in row), row
        assert not CYR.search(row[2]), row                       # третя колонка - англійською
        assert not CYR.search(row[0]) or row[0] in EN, row       # назва клавіші словом («Пробіл») - з перекладом
    assert 0 < guide.KEYS_LEFT < len(guide.KEYS)


def test_keys_table_alignment():
    for code in ("uk", "en"):
        i18n.set_lang(code)
        rows = guide.keys_table().split("\n\n")[0].split("\n")
        assert len(rows) == max(guide.KEYS_LEFT, len(guide.KEYS) - guide.KEYS_LEFT)
        for r in rows:
            assert len(r) <= guide.COL_W + guide.KEY_COL + 40, (code, r)       # ліва колонка не виїжджає на праву
        left = [k for k, _, _ in guide.KEYS[:guide.KEYS_LEFT]]
        assert all(len(k) < guide.KEY_COL for k, _, _ in guide.KEYS), "назва клавіші ширша за колонку"
        assert all(len(f"{k:<{guide.KEY_COL}}") == guide.KEY_COL for k in left)


def test_philosophy_is_the_agreed_text():
    assert guide.PHILOSOPHY[i18n.UK] == ("Весь наш світ ніби поділений на чорне й біле. Та варто на мить розфокусувати погляд — "
                                         "і між ними відкриваються незліченні відтінки.")


def test_about_mentions_author_fields_only_when_filled(monkeypatch):
    i18n.set_lang("uk")
    monkeypatch.setattr(meta, "AUTHOR", "")
    monkeypatch.setattr(meta, "SITE", "")
    monkeypatch.setattr(meta, "REPO", "")
    monkeypatch.setattr(meta, "STORE", "")
    monkeypatch.setattr(meta, "CONTACT", "")
    t = guide.about_text()
    assert "Автор:" not in t and "Сайт:" not in t and "Офіційно:" not in t and "Зв'язок:" not in t
    monkeypatch.setattr(meta, "AUTHOR", "Іван")
    monkeypatch.setattr(meta, "SITE", "example.org")
    monkeypatch.setattr(meta, "CONTACT", "a@b.c")
    t = guide.about_text()
    assert "Автор: Іван" in t and "Сайт: example.org" in t and "Зв'язок: a@b.c" in t        # без репозиторію й Store - лише сайт
    i18n.set_lang("en")
    t = guide.about_text()
    assert "Author: Іван" in t and "Website: example.org" in t and "Contact: a@b.c" in t


def test_about_lists_all_three_official_sources_in_both_languages():
    """сайт, GitHub і Store названо в «Про програму» й у меню футера: за цим відрізняють оригінал від клонів"""
    assert meta.SITE.startswith("https://inkreel.raimex.space") and meta.REPO == "https://github.com/Chimmurai/inkreel"
    assert meta.STORE == "https://apps.microsoft.com/detail/9NW62SPG91ZR"
    assert [u for _, u in meta.links()] == [meta.SITE, meta.REPO, meta.STORE]
    for lang, word in (("uk", "Офіційно:"), ("en", "Official:")):
        i18n.set_lang(lang)
        t = guide.about_text()
        assert word in t and "inkreel.raimex.space" in t and "github.com/Chimmurai/inkreel" in t and "Microsoft Store" in t
        assert all(i18n.tr(n) != n for n, _ in meta.links()) == (lang == "en")      # англійською кожен підпис має переклад
    i18n.set_lang("uk")


# ---------------------------------------------------------------- повідомлення про помилки з відео/фото
def test_open_errors_are_translated(tmp_path):
    from benvid import video
    missing = str(tmp_path / "nope.mp4")
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    i18n.set_lang("uk")
    assert video._open_error(missing, {}) == "файл не знайдено: nope.mp4"
    assert video._open_error(str(empty), {}) == "empty.mp4: файл порожній"
    i18n.set_lang("en")
    assert video._open_error(missing, {}) == "file not found: nope.mp4"
    assert video._open_error(str(empty), {}) == "empty.mp4: the file is empty"
    with pytest.raises(ValueError, match="cannot understand the size"):
        video.parse_size("12 by 34x")


# ---------------------------------------------------------------- виклики tr()/_say() дають усе, чого просить текст
def test_calls_supply_exactly_the_fields_their_text_needs():
    """кожне {поле} ключа в tr()/_say() передано, зайвих нема. Назва параметра самої функції не годиться за підстановку: у _say
    це done (прапорець «готово»), тож {done} у тексті дало б KeyError просто посеред рендеру - так і було"""
    bad = []
    for mod in UI_MODULES:
        for n in ast.walk(_parse(mod)):
            if not isinstance(n, ast.Call) or not n.args or not (isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)):
                continue
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if name not in ("tr", "_say") or any(k.arg is None for k in n.keywords):
                continue
            need = _fields(n.args[0].value) - {""}
            given = {k.arg for k in n.keywords}
            if name == "_say":
                given -= {"done"}
            if need != given:
                bad.append(f"{mod}:{n.lineno} {n.args[0].value[:40]!r}: треба {sorted(need)}, передано {sorted(given)}")
    assert not bad, "\n".join(bad)


# ---------------------------------------------------------------- рівні 1-5 та «Порада» (пакет П-1)
def test_level_names_ukrainian_and_english():
    from benvid import core
    assert [core.METHODS[n] for n in range(1, 7)] == ["Ескіз", "Контур", "Півтон", "Малюнок", "Гравюра", "IRNM1"]
    i18n.set_lang("en")
    assert [i18n.tr(core.METHODS[n]) for n in range(1, 7)] == ["Sketch", "Outline", "Halftone", "Drawing", "Engraving", "IRNM1"]
    i18n.set_lang("uk")
    assert not any("Ультра" in v for v in core.METHODS.values()), "стару назву рівня 5 прибрано"


def test_tips_bilingual_and_clean():
    uk, en = guide.TIPS[i18n.UK], guide.TIPS[i18n.EN]
    assert len(uk) == len(en) >= 10, "порад в обох мовах порівну"
    assert len(set(uk)) == len(uk) and len(set(en)) == len(en)
    assert not any(CYR.search(t) for t in en), "англійські поради без кирилиці"
    assert all(CYR.search(t) for t in uk)
    assert all(t.strip() and len(t) < 160 for t in uk + en)
    assert not any("у програмі" in t.lower() or "in the app" in t.lower() for t in uk + en), "це вже сама програма"
    assert not any("Ультр" in t or "Ultra" in t for t in uk + en)


def test_tip_picker_never_repeats_last():
    import random
    rng = random.Random(7)
    n = len(guide.TIPS[i18n.UK])
    last = guide.tip(-1, rng)
    for _ in range(200):
        nxt = guide.tip(last, rng)
        assert nxt != last and 0 <= nxt < n
        last = nxt
    for code in ("uk", "en"):
        i18n.set_lang(code)
        assert guide.tip_text(3) == guide.TIPS[code][3]


def test_slogans_in_about():
    for code in ("uk", "en"):
        i18n.set_lang(code)
        t = guide.about_text()
        for line in guide.SLOGAN[code] + (guide.TAGLINE[code],):
            assert line in t
    assert guide.SLOGAN[i18n.UK] == ("Світ складається з двох кольорів.", "Біле — пауза. Чорне — слово.")
    assert guide.SLOGAN[i18n.EN] == ("The world is made of two colors.", "White is a pause. Black is a word.")
    assert guide.TAGLINE[i18n.UK] == "Чорно-біле з характером." and guide.TAGLINE[i18n.EN] == "Black and white with character."
