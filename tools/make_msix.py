"""Збирає пакет MSIX для Microsoft Store з готової збірки dist\\InkReel (її робить scripts\\build_exe.bat).

    python tools/make_msix.py                      ->  dist\\InkReel.msix  (це й завантажують у Partner Center)
    python tools/make_msix.py --stage-only         ->  лише тека dist\\msix (маніфест + ресурси + програма), без makeappx
    python tools/make_msix.py --sign store\\InkReelDev.pfx --password inkreel
                                                    ->  ще й підписати тестовим сертифікатом: для перевірки на своєму ПК та WACK
                                                        (у Store підпис не потрібен - Microsoft підписує пакет сама)

Потрібен Windows SDK (makeappx.exe, makepri.exe, signtool.exe): https://developer.microsoft.com/windows/downloads/windows-sdk/
Ідентичність пакета (Name, Publisher) - з store/identity.json (див. store/STORE.md); без нього збирається тестовий пакет
(Name=InkReel.Dev), який у Store завантажити не вийде.

Маніфест - store/AppxManifest.xml.in; список форматів у ньому береться з benvid/media.py і benvid/video.py, версія - з benvid/__init__.py."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STORE = os.path.join(ROOT, "store")
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

TEMPLATE = os.path.join(STORE, "AppxManifest.xml.in")
IDENTITY = os.path.join(STORE, "identity.json")
DESCRIPTION = "Turns videos, GIFs and photos into high-quality black and white. Works offline."
NOT_IN_PACKAGE = ("menu_on.bat", "menu_off.bat")           # допоміжні файли збірки «папкою»: у пакеті Store вони ні до чого
DEFAULT_IDENTITY = {"name": "InkReel.Dev", "publisher": "CN=InkReelDev", "publisher_display_name": "InkReel (dev)",
                    "display_name": "InkReel"}


class BuildError(Exception):
    pass


# ---------------------------------------------------------------- ідентичність і версія
def load_identity(path: str | None = None) -> tuple[dict, bool]:
    """-> (ідентичність, справжня?). Немає файлу - тестова ідентичність і False"""
    path = path or IDENTITY
    if not os.path.isfile(path):
        return dict(DEFAULT_IDENTITY), False
    try:
        with open(path, encoding="utf-8-sig") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        raise BuildError(f"не вдалося прочитати {path}: {e}") from e
    if not isinstance(raw, dict):
        raise BuildError(f"{path}: очікував {{...}} з полями name, publisher, publisher_display_name")
    ident = dict(DEFAULT_IDENTITY)
    ident.update({k: str(v).strip() for k, v in raw.items() if k in ident and v is not None})
    validate_identity(ident)
    return ident, True


def validate_identity(i: dict):
    for k, v in i.items():
        if re.search(r"X{4,}|<[^>]*>|ВАШ|YOUR", v):
            raise BuildError(f"{k}: «{v}» - це зразок з identity.json.example; впиши значення з Partner Center (див. store/STORE.md)")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-]{2,49}", i["name"]):
        raise BuildError(f"Package/Identity/Name «{i['name']}»: 3-50 символів (літери, цифри, крапка, дефіс), як у Partner Center")
    if not re.fullmatch(r"CN=[^,;\"<>]+(,\s*[A-Z]+=[^,;\"<>]+)*", i["publisher"]):
        raise BuildError(f"Package/Identity/Publisher «{i['publisher']}»: має виглядати як CN=XXXXXXXX-XXXX-… (скопіюй із Partner Center)")
    for k in ("publisher_display_name", "display_name"):
        if not i[k] or len(i[k]) > 256:
            raise BuildError(f"{k}: порожнє або задовге")


def package_version(version: str, dev: bool = False) -> str:
    """0.2.0 -> 0.2.0.0. Четверте число Store резервує за собою: у пакеті воно завжди 0. Перше число не може бути 0 (правило Store):
    для справжнього пакета це помилка - постав у benvid/__init__.py версію 1.0.0 чи вище; тестовий (dev) пакет лише для перевірки
    на своєму ПК, тож там нуль замінюється одиницею"""
    parts = version.strip().split(".")
    if not 1 <= len(parts) <= 4 or not all(p.isdigit() for p in parts):
        raise BuildError(f"версія «{version}» не схожа на 1.2.3")
    nums = [int(p) for p in parts] + [0] * (4 - len(parts))
    if any(n > 65535 for n in nums):
        raise BuildError("кожне число версії має бути не більше 65535")
    if nums[3] != 0:
        raise BuildError("четверте число версії зарезервоване для Store й має бути 0")
    if nums[0] == 0:
        if not dev:
            raise BuildError(f"Store не приймає версію, що починається з 0 (зараз {version}): постав __version__ = \"1.0.0\" "
                             "(або вище) у benvid/__init__.py і збери програму знову")
        nums[0] = 1
    return ".".join(map(str, nums))


def app_version() -> str:
    with open(os.path.join(ROOT, "benvid", "__init__.py"), encoding="utf-8") as f:
        m = re.search(r'__version__\s*=\s*"([^"]+)"', f.read())
    if not m:
        raise BuildError("не знайшов __version__ у benvid/__init__.py")
    return m.group(1)


# ---------------------------------------------------------------- маніфест
def file_type_groups() -> list[tuple[str, list[str]]]:
    """групи форматів для «Відкрити за допомогою»: відео, GIF, фото (розширення не можуть повторюватись у різних групах).
    Name групи - 1-64 малі літери без пропусків; беремо лише a-z і 0-9, щоб не покладатися на те, які ще знаки дозволяє схема"""
    from benvid import media, video
    from benvid import bake
    return [("inkreelvideo", [e for e in video.VIDEO_EXT if e != bake.EXT]), ("inkreelbaked", [bake.EXT]),
            ("inkreelgif", list(media.GIF_EXT)), ("inkreelimage", list(media.IMAGE_EXT))]


def fta_xml(indent: str = "        ") -> str:
    out = []
    for name, exts in file_type_groups():
        out.append(f'{indent}<uap:Extension Category="windows.fileTypeAssociation">')
        out.append(f'{indent}  <uap:FileTypeAssociation Name="{name}">')
        out.append(f"{indent}    <uap:DisplayName>{'InkReel video (.ikr)' if name == 'inkreelbaked' else 'InkReel'}</uap:DisplayName>")
        if name == "inkreelbaked":                                   # власний значок файлу замість значка програми
            out.append(f"{indent}    <uap:Logo>Assets\\IkrFile.png</uap:Logo>")
        out.append(f"{indent}    <uap:SupportedFileTypes>")
        out += [f"{indent}      <uap:FileType>{escape(e)}</uap:FileType>" for e in exts]
        out.append(f"{indent}    </uap:SupportedFileTypes>")
        out.append(f"{indent}  </uap:FileTypeAssociation>")
        out.append(f"{indent}</uap:Extension>")
    return "\n".join(out)


def _attr(s: str) -> str:
    return escape(s, {'"': "&quot;"})


def render_manifest(identity: dict, version: str, template: str | None = None, dev: bool = False) -> str:
    if template is None:
        with open(TEMPLATE, encoding="utf-8") as f:
            template = f.read()
    values = {"IDENTITY_NAME": _attr(identity["name"]), "PUBLISHER": _attr(identity["publisher"]),
              "PUBLISHER_DISPLAY_NAME": escape(identity["publisher_display_name"]), "DISPLAY_NAME": _attr(identity["display_name"]),
              "VERSION": package_version(version, dev), "DESCRIPTION": _attr(DESCRIPTION), "FILE_TYPE_ASSOCIATIONS": fta_xml()}
    # DisplayName в елементі (не в атрибуті): лапки екранувати не треба, але & і < - так
    text = template
    text = text.replace("<DisplayName>{{DISPLAY_NAME}}</DisplayName>", f"<DisplayName>{escape(identity['display_name'])}</DisplayName>")
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", v)
    text = re.sub(r"<!--.*?-->\s*", "", text, count=1, flags=re.S)                   # шапка шаблона до готового файла не потрапляє
    text = text.replace("<Package", "<!-- Згенеровано tools/make_msix.py зі store/AppxManifest.xml.in; руками не правити -->\n<Package", 1)
    left = re.findall(r"\{\{(\w+)\}\}", text)
    if left:
        raise BuildError(f"у маніфесті лишились непідставлені поля: {sorted(set(left))}")
    return text


# ---------------------------------------------------------------- тека пакета
def stage(dist_dir: str, stage_dir: str, manifest_text: str) -> str:
    """тека, з якої makeappx робить пакет: програма + Assets + AppxManifest.xml. Повертає stage_dir"""
    if not os.path.isfile(os.path.join(dist_dir, "InkReel.exe")):
        raise BuildError(f"немає {os.path.join(dist_dir, 'InkReel.exe')}: спершу запусти scripts\\build_exe.bat")
    shutil.rmtree(stage_dir, ignore_errors=True)
    shutil.copytree(dist_dir, stage_dir, ignore=shutil.ignore_patterns(*NOT_IN_PACKAGE))
    import make_store_assets
    make_store_assets.write(os.path.join(stage_dir, "Assets"), make_store_assets.package_assets())
    with open(os.path.join(stage_dir, "AppxManifest.xml"), "w", encoding="utf-8", newline="\n") as f:
        f.write(manifest_text)
    return stage_dir


# ---------------------------------------------------------------- Windows SDK
def _sdk_version(path: str) -> tuple[int, ...]:
    folder = os.path.normpath(path).split(os.sep)[-3]                       # ...\bin\<версія>\x64\<інструмент>.exe
    return tuple(int(n) for n in re.findall(r"\d+", folder))


SDK_FALLBACK = "C:\\Program Files (x86)"


def find_sdk_tool(name: str) -> str | None:
    p = shutil.which(name)
    if p:
        return p
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), SDK_FALLBACK):
        if not base:
            continue
        hits = glob.glob(os.path.join(base, "Windows Kits", "10", "bin", "*", "x64", name + ".exe"))
        if hits:
            return max(hits, key=_sdk_version)                 # найновіша версія SDK (10.0.22621.0 > 10.0.9999.0: порівнюємо числа, а не рядки)
    return None


def _run(cmd: list[str]):
    print(">", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise BuildError(f"{os.path.basename(cmd[0])} завершився з кодом {r.returncode}")


def make_pri(stage_dir: str, work_dir: str, makepri: str):
    """resources.pri потрібен, щоб Windows вибирала логотипи за масштабом (Assets\\X.png -> X.scale-200.png). Індексуємо лише теку з
    Assets і маніфестом - а не всю програму з тисячами файлів PyInstaller; конфіг і результат лежать поза нею"""
    shutil.rmtree(work_dir, ignore_errors=True)
    root = os.path.join(work_dir, "root")
    os.makedirs(root)
    shutil.copytree(os.path.join(stage_dir, "Assets"), os.path.join(root, "Assets"))
    shutil.copy(os.path.join(stage_dir, "AppxManifest.xml"), root)
    cfg = os.path.join(work_dir, "priconfig.xml")
    pri = os.path.join(work_dir, "resources.pri")
    _run([makepri, "createconfig", "/cf", cfg, "/dq", "en-US", "/o"])
    _run([makepri, "new", "/pr", root, "/cf", cfg, "/mn", os.path.join(root, "AppxManifest.xml"), "/of", pri, "/o"])
    shutil.copy(pri, os.path.join(stage_dir, "resources.pri"))


def sign(pkg: str, pfx: str, password: str, signtool: str):
    _run([signtool, "sign", "/fd", "SHA256", "/f", pfx, "/p", password, pkg])


def build(dist_dir: str, out: str, identity: dict, version: str, stage_only=False, pfx=None, password="", dev=False) -> str:
    stage_dir = os.path.join(os.path.dirname(os.path.abspath(dist_dir)), "msix")
    stage(dist_dir, stage_dir, render_manifest(identity, version, dev=dev))
    print(f"тека пакета: {stage_dir}")
    if stage_only:
        return stage_dir
    makeappx, makepri = find_sdk_tool("makeappx"), find_sdk_tool("makepri")
    if not makeappx or not makepri:
        raise BuildError("не знайшов makeappx.exe / makepri.exe: встанови Windows SDK (https://developer.microsoft.com/windows/downloads/windows-sdk/)")
    make_pri(stage_dir, os.path.join(os.path.dirname(stage_dir), "msix_pri"), makepri)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if os.path.exists(out):
        os.remove(out)
    _run([makeappx, "pack", "/d", stage_dir, "/p", out, "/o"])
    if pfx:
        st = find_sdk_tool("signtool")
        if not st:
            raise BuildError("не знайшов signtool.exe (є у Windows SDK)")
        sign(out, pfx, password, st)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="збирає .msix для Microsoft Store")
    ap.add_argument("--dist", default=os.path.join(ROOT, "dist", "InkReel"), help="готова збірка PyInstaller (dist\\InkReel)")
    ap.add_argument("--out", default=os.path.join(ROOT, "dist", "InkReel.msix"))
    ap.add_argument("--identity", default=IDENTITY, help="store\\identity.json із даними Partner Center")
    ap.add_argument("--stage-only", action="store_true", help="лише скласти теку пакета, без makeappx")
    ap.add_argument("--sign", metavar="PFX", help="підписати тестовим сертифікатом (див. store\\make_test_cert.ps1)")
    ap.add_argument("--password", default="inkreel", help="пароль до .pfx")
    a = ap.parse_args(argv)
    try:
        ident, real = load_identity(a.identity)
        ver = app_version()
        print(f"InkReel {ver} -> пакет {package_version(ver, not real)}; Name={ident['name']}; Publisher={ident['publisher']}")
        if not real:
            print("УВАГА: немає store\\identity.json - це ТЕСТОВИЙ пакет (Name=InkReel.Dev); у Store його не завантажити. "
                  "Дані беруться з Partner Center, див. store\\STORE.md")
        res = build(a.dist, a.out, ident, ver, a.stage_only, a.sign, a.password, dev=not real)
        print("готово:", res)
        return 0
    except BuildError as e:
        print("ПОМИЛКА:", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
