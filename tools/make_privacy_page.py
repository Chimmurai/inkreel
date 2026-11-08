"""Робить із store/PRIVACY.md готову веб-сторінку політики приватності - її адресу просить Partner Center:

    python tools/make_privacy_page.py                       ->  store/privacy.html  (контакт береться з benvid/meta.py: CONTACT)
    python tools/make_privacy_page.py --contact me@mail.ua  ->  те саме з указаним контактом

Файл самодостатній (без скриптів і зовнішніх ресурсів): викладіть його будь-де як index.html - GitHub Pages, свій сайт тощо (див. store/STORE.md).
Текст політики живе лише в PRIVACY.md; сторінку щоразу будуємо з нього, тож вони не розійдуться."""
from __future__ import annotations

import argparse
import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "store", "PRIVACY.md")
OUT = os.path.join(ROOT, "store", "privacy.html")
PLACEHOLDER = "__CONTACT__"
LANGS = {"english": "en", "українською": "uk"}

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>InkReel — Privacy Policy</title>
<style>
:root { color-scheme: light dark; --fg: #111; --bg: #fff; --mute: #555; --line: #ccc; }
@media (prefers-color-scheme: dark) { :root { --fg: #eee; --bg: #000; --mute: #aaa; --line: #444; } }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.6 system-ui, "Segoe UI", sans-serif; }
main { max-width: 42rem; margin: 0 auto; padding: 1.5rem 1rem 4rem; }
h1 { font-size: 1.8rem; margin: 0 0 .5rem; }
h2 { font-size: 1.3rem; margin: 1.5rem 0 .5rem; }
hr { border: 0; border-top: 1px solid var(--line); margin: 2rem 0; }
em { color: var(--mute); }
code { font: .92em ui-monospace, Consolas, monospace; }
a { color: inherit; }
</style>
</head>
<body>
<main>
{{BODY}}
</main>
</body>
</html>
"""


def inline(text: str) -> str:
    """**жирний**, *курсив*, `код` - усе інше екранується"""
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<em>\1</em>", t)
    return t


def md_to_html(md: str) -> str:
    out: list[str] = []
    para: list[str] = []
    section_open = False

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(x.strip() for x in para)) + "</p>")
            para.clear()

    def close_section():
        nonlocal section_open
        if section_open:
            out.append("</section>")
            section_open = False

    for line in md.splitlines():
        s = line.strip()
        if not s:
            flush()
        elif s == "---":
            flush()
            close_section()
            out.append("<hr>")
        elif s.startswith("## "):
            flush()
            close_section()
            title = s[3:].strip()
            lang = LANGS.get(title.lower())
            out.append(f'<section lang="{lang}">' if lang else "<section>")
            section_open = True
            out.append("<h2>" + inline(title) + "</h2>")
        elif s.startswith("# "):
            flush()
            out.append("<h1>" + inline(s[2:].strip()) + "</h1>")
        else:
            para.append(s)
    flush()
    close_section()
    return "\n".join(out)


def contact_html(contact: str) -> str:
    c = contact.strip()
    esc = html.escape(c)
    if re.fullmatch(r"[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+", c):
        return f'<a href="mailto:{esc}">{esc}</a>'
    if re.fullmatch(r"https?://\S+", c):
        return f'<a href="{html.escape(c, quote=True)}">{esc}</a>'
    return esc


def render(md: str, contact: str) -> str:
    if not contact.strip():
        raise ValueError("не вказано контакт: заповніть CONTACT у benvid/meta.py або передайте --contact")
    body = md_to_html(md.replace(PLACEHOLDER, "\x00"))
    body = body.replace("\x00", contact_html(contact))
    return PAGE.replace("{{BODY}}", body)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="store/PRIVACY.md -> store/privacy.html")
    ap.add_argument("--contact", help="пошта чи адреса сторінки підтримки; за замовчуванням CONTACT з benvid/meta.py")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    contact = a.contact
    if contact is None:
        sys.path.insert(0, ROOT)
        from benvid import meta
        contact = meta.CONTACT
    try:
        with open(SRC, encoding="utf-8") as f:
            page = render(f.read(), contact)
    except ValueError as e:
        print("ПОМИЛКА:", e)
        return 1
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    print("готово:", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
