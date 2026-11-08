"""Тексти футера «Гайд та інфо»: розділи, гарячі клавіші, «Про програму». Лише дані й форматування тексту, без tkinter -
тому перевіряються тестами. Довгі абзаци лежать у двох мовах поруч; короткі підписи - через i18n.tr."""
from __future__ import annotations

from . import i18n, meta
from .i18n import tr

SECTIONS = ("start", "quality", "palettes", "keys", "formats", "about")
TAB = {"start": "Старт", "quality": "Якість", "palettes": "Палітри", "keys": "Клавіші", "formats": "Формати",
       "about": "Про програму"}                                    # ключі для tr()

PHILOSOPHY = {
    i18n.UK: "Весь наш світ ніби поділений на чорне й біле. Та варто на мить розфокусувати погляд — "
             "і між ними відкриваються незліченні відтінки.",
    i18n.EN: "Our whole world seems divided into black and white. But defocus your gaze for a moment — "
             "and countless shades open up between them.",
}
TAGLINE = {i18n.UK: "Чорно-біле з характером.", i18n.EN: "Black and white with character."}
SLOGAN = {i18n.UK: ("Світ складається з двох кольорів.", "Біле — пауза. Чорне — слово."),
          i18n.EN: ("The world is made of two colors.", "White is a pause. Black is a word.")}

# «Порада»: короткі фрази для порожнього екрана й футера (спільні з сайтом; тут лише про саму програму). Українська й англійська
# лежать поруч і мають однакову довжину (tests/test_i18n.py). Випадкова - tip(); не показується під час відтворення й експорту.
TIPS = {
    i18n.UK: (
        "Чорне — не відсутність світла. Це рішення.",
        "Розфокусуйся на мить: між чорним і білим відкриються незліченні відтінки.",
        "Око ловить лише різницю: тут світліше, там темніше.",
        "Без тіні немає форми, без світла немає тіні.",
        "Гравюрники знали це п’ять століть тому. Ми лише навчили цьому кадри.",
        "Кожен піксель — одне питання: світло чи тінь?",
        "Півтон — чесний обман: око складає його з крапок.",
        "Чорно-біле — не бідність, а відбір.",
        "Між чорним і білим не порожнеча, а все решта.",
        "Усе зайве відпадає, лишається форма.",
        "П’ять рівнів: Ескіз, Контур, Півтон, Малюнок, Гравюра.",
        "IRNM1 — не шостий щабель, а окрема гілка: нейромережа, що вчилася в «Гравюри».",
        "Пробіл — пауза, ←/→ — кадр за кадром.",
        "Клавіша F або подвійний клік — режим «кіно»: лише кадр на весь екран.",
        "Колесо миші збільшує цілими кроками до 6400 %: піксель завжди лишається квадратом.",
        "Клавіша I міняє туш і папір місцями.",
        "«Порахувати все»: порахували раз — і граєте, масштабуєте, експортуєте без перерахунку.",
        "Розмір «Авто»: програма сама міряє швидкість вашого ПК і бере найбільший розмір, що йде в реальному часі.",
        "На слабкому ПК програма рахує кожен N-й кадр, а час і звук лишаються на місці.",
        "Експорт відео бере звук з оригіналу.",
        "Клавіші [ та ] позначають початок і кінець фрагмента для експорту.",
        "Файл можна відкрити через Ctrl+O або перетягнути у вікно.",
        "«Кадр → фото» зберігає без втрат: PNG, BMP, TIFF або WebP.",
        "Той самий кадр дає те саме чорнило: дрібниці не «пливуть» від запуску до запуску.",
        "У виді «Дві полярності» межа між позитивом і негативом іде за курсором.",
    ),
    i18n.EN: (
        "Black is not the absence of light. It is a decision.",
        "Unfocus for a moment: between black and white, countless shades open up.",
        "The eye catches only difference: lighter here, darker there.",
        "Without shadow there is no form, without light there is no shadow.",
        "Engravers knew this five centuries ago. We only taught it to frames.",
        "Every pixel is one question: light or shadow?",
        "A halftone is an honest trick: the eye assembles it from dots.",
        "Black and white is not poverty, it is selection.",
        "Between black and white there is not emptiness, but everything else.",
        "Everything extra falls away; form remains.",
        "Five levels: Sketch, Outline, Halftone, Drawing, Engraving.",
        "IRNM1 is not a sixth step but a separate branch: a neural network that learned from “Engraving”.",
        "Space pauses, ←/→ steps frame by frame.",
        "The F key or a double click is “cinema” mode: only the frame, full screen.",
        "The mouse wheel zooms in whole steps up to 6400%: a pixel always stays a square.",
        "The I key swaps ink and paper.",
        "“Render all”: compute once, then play, zoom and export without recomputing.",
        "Size “Auto”: the app measures your PC and picks the largest size that runs in real time.",
        "On a weak PC the app computes every Nth frame, while time and sound stay in place.",
        "Video export takes the sound from the original.",
        "The [ and ] keys mark the start and end of a fragment for export.",
        "Open a file with Ctrl+O or drag it into the window.",
        "“Frame → photo” saves losslessly: PNG, BMP, TIFF or WebP.",
        "The same frame gives the same ink: details do not drift from run to run.",
        "In the “Two polarities” view the line between positive and negative follows the cursor.",
    ),
}


def tip(last: int = -1, rng=None) -> int:
    """номер наступної поради: щоразу інша, не та сама, що щойно була"""
    import random
    rng = rng or random
    n = len(TIPS[i18n.UK])
    if last < 0 or last >= n:
        return rng.randrange(n)
    i = rng.randrange(n - 1)
    return i + 1 if i >= last else i


def tip_text(n: int) -> str:
    """порада під номером n поточною мовою"""
    t = TIPS[i18n.lang()]
    return t[n % len(t)]

_TEXT = {
    "start": {
        i18n.UK: (
            "Як це працює\n\n"
            "1.  Відкрий відео, GIF або фото: кнопка «Відкрити» або просто перетягни файл у вікно.\n"
            "2.  Усе рахується саме й одразу показується чорно-білим у реальному часі. Розмір «Авто» підбирає найбільший, "
            "який твій ПК тягне без підвисань.\n"
            "3.  Повзунок «Якість» — від найшвидшого порогу до найскладнішої графіки. Палітра міняє лише кольори: "
            "нічого не перераховується.\n"
            "4.  «Зберегти…» — обери формат і місце. У відео можна взяти лише фрагмент: прапорці [ ] над шкалою часу.\n"
            "5.  З Провідника: правий клік по фото чи GIF → «Зробити чорно-біле (InkReel)» — результат ляже поруч з оригіналом "
            "(пункти додає інсталятор або menu_on.bat).\n\n"
            "Окремий кадр відео чи GIF зберігається фото кнопкою «Кадр → фото»."),
        i18n.EN: (
            "How it works\n\n"
            "1.  Open a video, GIF or photo: the “Open” button, or simply drag a file into the window.\n"
            "2.  Everything is processed by itself and shown in black and white in real time. The “Auto” size picks the "
            "largest one your PC can play without stutter.\n"
            "3.  The “Quality” slider goes from the fastest threshold to the most elaborate graphics. The palette only "
            "changes colours: nothing is recomputed.\n"
            "4.  “Save…” — choose a format and a place. For videos you can take just a part: the [ ] flags above the "
            "timeline.\n"
            "5.  From Explorer: right-click a photo or GIF → “Make black-and-white (InkReel)” — the result lands next to the "
            "original (the installer or menu_on.bat adds the items).\n\n"
            "A single frame of a video or GIF can be saved as a photo with the “Frame → photo” button."),
    },
    "palettes": {
        i18n.UK: (
            "Результат завжди двоколірний: «туш» (темні місця) і «папір» (світлі).\n\n"
            "◀ ▶ — попередня й наступна палітра, ⟳ — навпаки: туш і папір міняються місцями (як негатив).\n"
            "Палітра застосовується в самому кінці — у перегляді й у всіх файлах, які зберігаєш, — тож перемикання "
            "миттєве, нічого не перераховується.\n"
            "«Класика» — чисте чорне на білому: лише вона дає справжні 1-бітні файли (PNG 1 біт, TIFF G4). Решта палітр "
            "зберігаються як зображення з двох кольорів.\n"
            "Кольори самого вікна теж підлаштовуються під обрану палітру."),
        i18n.EN: (
            "The result always has two colours: “ink” (the dark places) and “paper” (the light ones).\n\n"
            "◀ ▶ — previous and next palette, ⟳ — reversed: ink and paper swap places (like a negative).\n"
            "The palette is applied at the very end — in the preview and in every file you save — so switching is "
            "instant and nothing is recomputed.\n"
            "“Classic” is pure black on white: only it gives true 1-bit files (PNG 1-bit, TIFF G4). Other palettes are "
            "saved as two-colour images.\n"
            "The colours of the window itself follow the chosen palette too."),
    },
}

# (клавіші, українською, англійською); перші KEYS_LEFT рядків - ліва колонка таблиці, решта - права
KEYS = [
    ("Пробіл", "відтворення / пауза", "play / pause"),
    ("← →", "кадр назад / вперед", "previous / next frame"),
    ("Shift + ← →", "−10 с / +10 с", "−10 s / +10 s"),
    ("Home", "на початок", "back to the start"),
    ("↑ ↓", "гучність ±5 %", "volume ±5 %"),
    ("M", "без звуку", "mute"),
    ("L", "цикл (повтор)", "loop"),
    ("[   ]", "початок / кінець фрагмента тут", "range start / end here"),
    ("I", "палітра навпаки", "reverse the palette"),
    ("F", "кіно: лише картинка (Esc)", "cinema: picture only (Esc)"),
    ("F11", "програма на весь екран (Esc)", "whole app full screen (Esc)"),
    ("‹ ›", "в F11: права панель (кнопка справа)", "in F11: right panel (button on the right)"),
    ("Колесо · + −", "масштаб 1× 2× 3×… без розмиття", "zoom 1× 2× 3×… never blurred"),
    ("0", "вписати у вікно", "fit to window"),
    ("1", "один до одного (піксель = піксель)", "1:1 (pixel = pixel)"),
    ("Тягни мишею", "рух картинки при збільшенні", "pan when zoomed in"),
    ("Клік · 2× клік", "пауза · кіно", "play / pause · cinema"),
    ("Колесо над ♪", "гучність", "volume"),
    ("PgUp · PgDn", "екран вгору / вниз", "previous / next screen"),
    ("F1", "гайд (ще раз — нагору)", "this guide (again — back up)"),
]
KEYS_LEFT = 12
KEY_COL = 18
COL_W = 60


def keys_table() -> str:
    """гарячі клавіші двома колонками (шрифт моноширинний, тож вирівнюємо пробілами)"""
    en = i18n.lang() == i18n.EN
    rows = [(tr(k), e if en else u) for k, u, e in KEYS]       # назви клавіш теж перекладаємо («Пробіл» -> Space)
    left, right = rows[:KEYS_LEFT], rows[KEYS_LEFT:]
    fmt = lambda r: f"{r[0]:<{KEY_COL}}{r[1]}" if r else ""
    out = []
    for i in range(max(len(left), len(right))):
        a = fmt(left[i]) if i < len(left) else ""
        b = fmt(right[i]) if i < len(right) else ""
        out.append((f"{a:<{COL_W}}{b}").rstrip())
    note = tr("Літерні клавіші (M, L, F, I) працюють в будь-якій розкладці.")
    return "\n".join(out) + "\n\n" + note


def quality_text() -> str:
    from .core import METHOD_INFO, METHODS
    lines = [tr("П'ять рівнів — від швидкості до якості; шостий (ромб праворуч) — окрема гілка. У вікні показується лише кодова назва поточного.")]
    for n in sorted(METHODS):
        lines.append(f"{n}  {tr(METHODS[n]):<9} {tr(METHOD_INFO[n])}")
    lines += ["", tr("Чим вищий рівень, тим більше роботи на кадр. «Авто» підбирає розмір, який твій ПК встигає в реальному часі; "
                      "для збереження файлу поспіху нема — бери «Гравюру».")]
    lines += [tr("«Чорнило» (штрихи й крапки) та «Символи» (знаки шрифту, Consolas) — два погляди на той самий кадр; рівні, «Авто», палітри й експорт спільні.")]
    return "\n".join(lines)


def formats_text() -> str:
    from .media import GIF_EXT, GIF_FORMATS, IMAGE_EXT, STILL_FORMATS
    from .video import FORMATS, VIDEO_EXT
    ext = lambda xs: " ".join(x.lstrip(".") for x in xs)
    t = lambda ks: " · ".join(tr(k) for k in ks)
    return "\n".join([
        tr("Відкрити"),
        f"  {tr('Відео'):<7} {ext(VIDEO_EXT)}",
        f"  GIF     {ext(GIF_EXT)}",
        f"  {tr('Фото'):<7} {ext(IMAGE_EXT)}",
        "          " + tr("HEIC/HEIF — з пакетом pillow-heif; якщо щось не відкривається, глянь лог (див. «Про програму»)."),
        "",
        tr("Зберегти"),
        f"  {tr('Відео'):<7} {t(FORMATS)}",
        f"  GIF     {t(GIF_FORMATS)}",
        f"  {tr('Фото'):<7} {t(STILL_FORMATS)}",
        "          " + tr("У режимі «Символи» фото (й кадр) можна зберегти текстом .txt: світлі ділянки — густі знаки, "
                          "відкривай у моноширинному шрифті на темному тлі."),
    ])


def about_text() -> str:
    lines = [f"{meta.APP} {meta.VERSION}", *SLOGAN[i18n.lang()], TAGLINE[i18n.lang()], "",
             tr("Працює офлайн: файли не залишають твій комп'ютер. Без реклами й збору даних."),
             tr("Тут сховано кілька дрібниць — клікай не лише по кнопках."), ""]
    if meta.AUTHOR:
        lines.append(tr("Автор: {a}", a=meta.AUTHOR))
    if meta.SITE:
        lines.append(tr("Сайт: {s}", s=meta.SITE))
    if meta.CONTACT:
        lines.append(tr("Зв'язок: {c}", c=meta.CONTACT))
    if meta.AUTHOR or meta.SITE or meta.CONTACT:
        lines.append("")
    lines.append(tr("Використано: Python, Tk, NumPy, OpenCV, Numba, Pillow, FFmpeg, sounddevice."))
    lines.append(tr("Ліцензії компонентів: THIRD_PARTY_NOTICES.md у теці програми."))
    try:
        from . import log as _log
        lines.append(tr("Журнал помилок: {p}", p=_log.shown_path()))
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


def title(sec: str) -> str:
    return tr(TAB[sec])


def body(sec: str) -> str:
    if sec in _TEXT:
        return _TEXT[sec][i18n.lang()]
    return {"keys": keys_table, "quality": quality_text, "formats": formats_text, "about": about_text}[sec]()


def philosophy() -> str:
    return PHILOSOPHY[i18n.lang()]
