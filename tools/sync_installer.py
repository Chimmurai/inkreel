"""Синхронізує згенеровані блоки в installer.iss зі списками форматів у коді (benvid/media.py, benvid/video.py), щоб
інсталятор не розходився з тим, що програма справді вміє відкривати:

    python tools/sync_installer.py          - переписати блоки в installer.iss
    python tools/sync_installer.py --check  - лише перевірити (код виходу 1, якщо розійшлося); те саме робить tests/test_installer.py

Блоки між рядками «; >>> Назва … >>>» і «; <<< Назва <<<» руками не правити:
  SupportedTypes - «Відкрити за допомогою» для всіх відео, GIF і фото;
  ContextMenu    - правий клік по фото й GIF: «Відкрити в InkReel» та «Зробити чорно-біле» (InkReel.exe --quick, benvid/quick.py);
                   набір пунктів - benvid/shellmenu.py (його ж пише `InkReel.exe --register-menu` для збірки без інсталятора)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISS = os.path.join(ROOT, "installer.iss")
APP_EXE = "{app}\\{#Exe}"                # так шлях до exe пишеться в installer.iss


def parse_params(line: str) -> dict[str, str]:
    """«Root: HKCU; Subkey: "…"; …» -> {'Root': 'HKCU', 'Subkey': '…', …}: розділювач «;» лише поза лапками, «""» усередині - одна лапка"""
    parts, cur, q, i = [], [], False, 0
    while i < len(line):
        c = line[i]
        if c == '"':
            if q and line[i + 1:i + 2] == '"':
                cur.append('""')
                i += 2
                continue
            q = not q
            cur.append(c)
        elif c == ";" and not q:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    parts.append("".join(cur))
    out = {}
    for p in parts:
        k, _, v = p.strip().partition(":")
        v = v.strip()
        if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
            v = v[1:-1].replace('""', '"')
        out[k.strip()] = v
    return out


def sections(text: str) -> dict[str, list[str]]:
    """секція -> її рядки без порожніх і коментарів"""
    import re
    cur, out = "", {}
    for ln in text.splitlines():
        s = ln.strip()
        m = re.fullmatch(r"\[(\w+)\]", s)
        if m:
            cur = m.group(1)
            out[cur] = []
        elif s and not s.startswith(";") and not s.startswith("//") and cur:
            out[cur].append(s)
    return out


def _exts():
    sys.path.insert(0, ROOT)
    from benvid import media, video
    return list(video.VIDEO_EXT), list(media.IMAGE_EXT), list(media.GIF_EXT)


def supported_types() -> list[str]:
    vid, img, gif = _exts()
    base = 'Root: HKCU; Subkey: "Software\\Classes\\Applications\\{#Exe}\\SupportedTypes"; ValueType: string; ValueName: "%s"; ValueData: ""'
    return [base % e for e in vid + gif + img]


def _q(value: str) -> str:
    """значення параметра Inno Setup: у лапках, а лапки всередині подвоєні"""
    return '"' + value.replace('"', '""') + '"'


def context_menu() -> list[str]:
    """рядки меню правої кнопки беремо з benvid/shellmenu.py - там та сама правда, що пише `InkReel.exe --register-menu`"""
    sys.path.insert(0, ROOT)
    from benvid import shellmenu
    cm = lambda verb, label: "{cm:Ctx" + verb.split(".", 1)[1] + "}"            # InkReel.Quick -> {cm:CtxQuick}
    out = []
    for key, name, value in shellmenu.entries(APP_EXE, caption=cm):
        line = f'Root: HKCU; Subkey: "{key}"; ValueType: string; '
        if name:
            line += f'ValueName: "{name}"; '
        line += f"ValueData: {_q(value)}; Tasks: ctxmenu"
        if name == "MUIVerb":
            line += "; Flags: uninsdeletekey"
        out.append(line)
    return out


BLOCKS = {"SupportedTypes": supported_types, "ContextMenu": context_menu}


def _markers(name: str) -> tuple[str, str]:
    return f"; >>> {name} (згенеровано: python tools/sync_installer.py - руками не правити) >>>", f"; <<< {name} <<<"


def render(text: str) -> str:
    """текст installer.iss із актуальними згенерованими блоками"""
    for name, fn in BLOCKS.items():
        a, b = _markers(name)
        i, j = text.find(a), text.find(b)
        if i < 0 or j < 0 or j < i:
            raise SystemExit(f"installer.iss: немає блоку «{name}» (рядки {a!r} … {b!r})")
        text = text[:i] + a + "\n" + "\n".join(fn()) + "\n" + text[j:]
    return text


def read() -> str:
    with open(ISS, encoding="utf-8-sig") as f:
        return f.read()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cur = read()
    new = render(cur)
    if "--check" in argv:
        if new != cur:
            print("installer.iss розійшовся зі списками форматів: запусти  python tools/sync_installer.py")
            return 1
        print("installer.iss актуальний")
        return 0
    if new != cur:
        with open(ISS, "w", encoding="utf-8-sig", newline="") as f:      # з BOM: Inno Setup тоді читає Unicode правильно
            f.write(new)
        print("installer.iss оновлено")
    else:
        print("installer.iss уже актуальний")
    return 0


if __name__ == "__main__":
    sys.exit(main())
