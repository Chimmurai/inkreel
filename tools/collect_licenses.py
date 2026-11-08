"""Складає тексти ліцензій у теку збірки (її викликає scripts\\build_exe.bat після PyInstaller), щоб вони їхали разом із програмою -
і в папці, і в інсталяторі, і в пакеті для Microsoft Store:

    python tools/collect_licenses.py dist\\InkReel

Що з'являється у <збірка>:
  THIRD_PARTY_NOTICES.md          перелік компонентів і ліцензій (пишеться вручну, лежить у корені проєкту)
  licenses\\GPL-3.0.txt ...        тексти GPL/LGPL, на які посилаються примітки (теки licenses\\ проєкту)
  licenses\\python\\<пакет>\\...    файли ліцензій саме тих версій пакетів, що стоять у середовищі збірки: беруться з dist-info
                                  (METADATA -> License-File) пакетів із requirements.txt і всіх їхніх залежностей
  licenses\\python\\INDEX.txt      «пакет версія ліцензія» для кожного з них"""
from __future__ import annotations

import importlib.metadata as md
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LICENSE_NAME = re.compile(r"(?i)^(licen[sc]e|licen[sc]es|copying|notice)([.\-_].*)?$")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def requirement_names(path: str) -> list[str]:
    """імена пакетів з requirements.txt (без версій, маркерів і коментарів)"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            m = re.match(r"[A-Za-z0-9][A-Za-z0-9._\-]*", line)
            if m:
                out.append(norm(m.group(0)))
    return out


def installed(search_path=None) -> dict[str, md.Distribution]:
    found: dict[str, md.Distribution] = {}
    for d in md.distributions(path=search_path) if search_path else md.distributions():
        name = d.metadata["Name"]
        if name:
            found.setdefault(norm(name), d)
    return found


def closure(names: list[str], dists: dict[str, md.Distribution]) -> list[str]:
    """names і всі їхні встановлені залежності (extras і пакети, яких немає в середовищі, пропускаємо)"""
    seen: list[str] = []
    todo = list(names)
    while todo:
        n = todo.pop(0)
        if n in seen or n not in dists:
            continue
        seen.append(n)
        for req in dists[n].requires or []:
            if "extra ==" in req.replace('"', "'").replace("'", ""):
                continue
            m = re.match(r"[A-Za-z0-9][A-Za-z0-9._\-]*", req.strip())
            if m:
                todo.append(norm(m.group(0)))
    return seen


def license_files(dist: md.Distribution) -> list[tuple[str, md.PackagePath]]:
    """(відносний шлях для збереження, файл) для ліцензійних файлів усередині *.dist-info"""
    out = []
    for f in dist.files or []:
        parts = f.parts
        di = next((i for i, p in enumerate(parts) if p.endswith(".dist-info")), None)
        if di is None or len(parts) <= di + 1 or f.suffix == ".py":
            continue
        inner = parts[di + 1:]
        if inner[0] == "licenses":
            inner = inner[1:]
        if not inner or not LICENSE_NAME.match(inner[-1]):
            continue
        out.append(("/".join(inner), f))
    return out


def license_label(dist: md.Distribution) -> str:
    m = dist.metadata
    expr = m.get("License-Expression") or m.get("License") or ""
    expr = " ".join(expr.split())
    if not expr or len(expr) > 80:
        cl = [c.split("::")[-1].strip() for c in (m.get_all("Classifier") or []) if c.startswith("License ::")]
        expr = ", ".join(cl) or "(див. файли)"
    return expr


def collect(dest: str, requirements: str | None = None, root: str = ROOT, search_path=None) -> dict:
    """-> {'written': [шляхи], 'packages': [(ім'я, версія, ліцензія)], 'missing': [імена без файлів ліцензій]}"""
    requirements = requirements or os.path.join(root, "requirements.txt")
    dists = installed(search_path)
    names = closure(requirement_names(requirements), dists)
    written, pkgs, missing = [], [], []
    base = os.path.join(dest, "licenses", "python")
    for n in names:
        d = dists[n]
        files = license_files(d)
        pkgs.append((d.metadata["Name"], d.version, license_label(d)))
        if not files:
            missing.append(n)
            continue
        for rel, f in files:
            target = os.path.join(base, n, *rel.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(str(d.locate_file(f)), target)
            written.append(target)
    os.makedirs(base, exist_ok=True)
    index = os.path.join(base, "INDEX.txt")
    with open(index, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("Пакети Python у збірці: назва, версія, ліцензія / Python packages in this build: name, version, licence\n\n")
        for name, ver, lic in pkgs:
            fh.write(f"{name} {ver} - {lic}\n")
        if missing:
            fh.write("\nБез файлу ліцензії в пакеті / no licence file shipped in the package: " + ", ".join(missing) + "\n")
    written.append(index)
    return {"written": written, "packages": pkgs, "missing": missing}


def copy_static(dest: str, root: str = ROOT) -> list[str]:
    """THIRD_PARTY_NOTICES.md і тексти GPL/LGPL з проєкту"""
    out = []
    notices = os.path.join(root, "THIRD_PARTY_NOTICES.md")
    if os.path.isfile(notices):
        shutil.copyfile(notices, os.path.join(dest, "THIRD_PARTY_NOTICES.md"))
        out.append(os.path.join(dest, "THIRD_PARTY_NOTICES.md"))
    # власна ліцензія InkReel (Apache-2.0) і NOTICE їдуть поруч: LICENSE.txt, NOTICE.txt
    for src_name, dst_name in (("LICENSE", "LICENSE.txt"), ("NOTICE", "NOTICE.txt")):
        own = os.path.join(root, src_name)
        if os.path.isfile(own):
            shutil.copyfile(own, os.path.join(dest, dst_name))
            out.append(os.path.join(dest, dst_name))
    src = os.path.join(root, "licenses")
    if os.path.isdir(src):
        os.makedirs(os.path.join(dest, "licenses"), exist_ok=True)
        for name in sorted(os.listdir(src)):
            if os.path.isfile(os.path.join(src, name)):
                shutil.copyfile(os.path.join(src, name), os.path.join(dest, "licenses", name))
                out.append(os.path.join(dest, "licenses", name))
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or not os.path.isdir(argv[0]):
        print("використання: python tools/collect_licenses.py <тека збірки, напр. dist\\InkReel>")
        return 2
    dest = argv[0]
    static = copy_static(dest)
    res = collect(dest)
    print(f"ліцензії: {len(static)} файлів із проєкту, {len(res['written'])} з пакетів Python ({len(res['packages'])} пакетів) -> {dest}")
    if res["missing"]:
        print("увага: без файлу ліцензії в пакеті:", ", ".join(res["missing"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
