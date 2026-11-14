"""Пакет для Microsoft Store: маніфест, логотипи, послідовність команд збирача, тексти сторінки, політика приватності, ліцензії.
Windows і Windows SDK тут немає, тож перевіряємо те, що можна: готові файли, їхню узгодженість із програмою й логіку збирача на підроблених командах."""
from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import collect_licenses as cl                                             # noqa: E402
import make_msix as mm                                                    # noqa: E402
import make_privacy_page as mpp                                           # noqa: E402
import make_store_assets as msa                                           # noqa: E402
import sync_installer as si                                               # noqa: E402
from benvid import __version__, media, meta, palette, video               # noqa: E402

NS = {"m": "http://schemas.microsoft.com/appx/manifest/foundation/windows10",
      "uap": "http://schemas.microsoft.com/appx/manifest/uap/windows10",
      "rescap": "http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"}
REAL = {"name": "12345Author.InkReel", "publisher": "CN=ABCDEF12-3456-7890-ABCD-EF1234567890",
        "publisher_display_name": "Автор", "display_name": "InkReel"}


def read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def assets() -> dict[str, Image.Image]:
    return msa.package_assets()


@pytest.fixture
def manifest() -> ET.Element:
    return ET.fromstring(mm.render_manifest(REAL, "1.0.0").encode("utf-8"))


# ---------------------------------------------------------------- версія
@pytest.mark.parametrize("given,expected", [("1.0.0", "1.0.0.0"), ("1.2", "1.2.0.0"), ("3", "3.0.0.0"), ("2.10.65535", "2.10.65535.0")])
def test_package_version_has_four_parts_with_store_revision_zero(given, expected):
    assert mm.package_version(given) == expected


@pytest.mark.parametrize("bad", ["", "a.b", "1.x", "1..2", "1.2.3.4", "1.2.3.4.5", "70000.1.1", "-1.0.0"])
def test_bad_versions_are_rejected(bad):
    with pytest.raises(mm.BuildError):
        mm.package_version(bad)


def test_store_does_not_accept_a_leading_zero_but_a_dev_package_may_have_it():
    with pytest.raises(mm.BuildError, match="1.0.0"):
        mm.package_version("0.2.0")
    assert mm.package_version("0.2.0", dev=True) == "1.2.0.0"


def test_app_version_is_read_from_the_package():
    assert mm.app_version() == __version__


# ---------------------------------------------------------------- ідентичність
def test_without_identity_file_the_package_is_a_dev_one(tmp_path):
    ident, real = mm.load_identity(str(tmp_path / "none.json"))
    assert real is False and ident["name"] == "InkReel.Dev"


def test_identity_file_values_are_used(tmp_path):
    import json
    f = tmp_path / "identity.json"
    f.write_text(json.dumps(REAL, ensure_ascii=False), encoding="utf-8")
    ident, real = mm.load_identity(str(f))
    assert real is True and ident == REAL


def test_identity_file_with_bom_is_read(tmp_path):
    import json
    f = tmp_path / "identity.json"
    f.write_bytes(b"\xef\xbb\xbf" + json.dumps(REAL).encode("utf-8"))             # Блокнот Windows любить додавати BOM
    assert mm.load_identity(str(f))[0]["name"] == REAL["name"]


def test_the_example_file_cannot_be_used_unchanged():
    """зразок ніхто не мусить встигнути завантажити в Store з XXXX замість справжніх значень"""
    with pytest.raises(mm.BuildError, match="зразок"):
        mm.load_identity(os.path.join(ROOT, "store", "identity.json.example"))


@pytest.mark.parametrize("field,value", [("name", "ab"), ("name", "has space.InkReel"), ("name", "x" * 60), ("publisher", "InkReelDev"),
                                         ("publisher", "CN="), ("publisher_display_name", ""), ("display_name", "")])
def test_bad_identity_values_are_rejected(tmp_path, field, value):
    import json
    f = tmp_path / "identity.json"
    f.write_text(json.dumps({**REAL, field: value}), encoding="utf-8")
    with pytest.raises(mm.BuildError):
        mm.load_identity(str(f))


def test_broken_identity_json_gives_a_readable_error(tmp_path):
    f = tmp_path / "identity.json"
    f.write_text("{нечитабельно", encoding="utf-8")
    with pytest.raises(mm.BuildError, match="identity.json"):
        mm.load_identity(str(f))


# ---------------------------------------------------------------- маніфест
def test_manifest_identity_and_capabilities(manifest):
    ident = manifest.find("m:Identity", NS)
    assert ident.get("Name") == REAL["name"] and ident.get("Publisher") == REAL["publisher"]
    assert ident.get("Version") == "1.0.0.0" and ident.get("ProcessorArchitecture") == "x64"
    assert manifest.find("m:Properties/m:PublisherDisplayName", NS).text == "Автор"
    caps = [c.get("Name") for c in manifest.findall("m:Capabilities/rescap:Capability", NS)]
    assert caps == ["runFullTrust"]
    app = manifest.find("m:Applications/m:Application", NS)
    assert app.get("EntryPoint") == "Windows.FullTrustApplication" and app.get("Executable") == "InkReel.exe"


def test_manifest_executable_is_what_the_spec_builds():
    spec = read("inkreel.spec")
    assert re.search(r'EXE\(pyz, a\.scripts.*?name="InkReel"', spec, re.S)


def test_manifest_declares_the_languages_of_the_interface(manifest):
    langs = [r.get("Language") for r in manifest.findall("m:Resources/m:Resource", NS)]
    assert langs == ["en-US", "uk-UA"]


def test_manifest_has_no_leftovers_and_no_template_header():
    text = mm.render_manifest(REAL, "1.0.0")
    assert "{{" not in text and "Шаблон маніфеста" not in text
    assert "Згенеровано tools/make_msix.py" in text and text.startswith('<?xml version="1.0" encoding="utf-8"?>')


def test_manifest_escapes_special_characters():
    ident = {**REAL, "publisher_display_name": 'A & B <"C">', "display_name": "Ink & Reel <1>"}
    root = ET.fromstring(mm.render_manifest(ident, "1.0.0").encode("utf-8"))
    assert root.find("m:Properties/m:PublisherDisplayName", NS).text == 'A & B <"C">'
    assert root.find("m:Properties/m:DisplayName", NS).text == "Ink & Reel <1>"
    assert root.find("m:Applications/m:Application/uap:VisualElements", NS).get("DisplayName") == "Ink & Reel <1>"


def test_unreplaced_placeholder_is_an_error():
    with pytest.raises(mm.BuildError, match="NEW_FIELD"):
        mm.render_manifest(REAL, "1.0.0", template="<Package>{{NEW_FIELD}}</Package>")


def file_types(root: ET.Element) -> dict[str, list[str]]:
    return {fta.get("Name"): [t.text for t in fta.findall("uap:SupportedFileTypes/uap:FileType", NS)]
            for fta in root.findall(".//uap:FileTypeAssociation", NS)}


def test_manifest_file_types_are_everything_the_program_opens(manifest):
    groups = file_types(manifest)
    every = [e for exts in groups.values() for e in exts]
    assert sorted(every) == sorted(set(video.VIDEO_EXT) | set(media.IMAGE_EXT) | set(media.GIF_EXT))
    assert len(every) == len(set(every))                                    # одне розширення - в одній групі
    assert all(e.startswith(".") and e == e.lower() for e in every)
    assert all(re.fullmatch(r"[a-z0-9]{1,50}", n) for n in groups)


def test_manifest_file_types_equal_the_installer_ones(manifest):
    inno = {m.group(1) for line in si.supported_types() for m in [re.search(r'ValueName: "(\.[^"]+)"', line)] if m}
    assert {e for exts in file_types(manifest).values() for e in exts} == inno


def test_file_type_association_children_are_in_schema_order(manifest):
    for fta in manifest.findall(".//uap:FileTypeAssociation", NS):
        tags = [c.tag.split("}")[1] for c in fta]
        assert tags in (["DisplayName", "SupportedFileTypes"], ["DisplayName", "Logo", "SupportedFileTypes"]), tags
        if fta.get("Name") == "inkreelbaked":
            assert tags[1] == "Logo" and fta.find("uap:Logo", NS).text == "Assets\\IkrFile.png"


def test_visual_elements_reference_existing_asset_families(manifest, assets):
    ve = manifest.find("m:Applications/m:Application/uap:VisualElements", NS)
    dt = ve.find("uap:DefaultTile", NS)
    refs = [manifest.find("m:Properties/m:Logo", NS).text, ve.get("Square150x150Logo"), ve.get("Square44x44Logo"),
            dt.get("Wide310x150Logo"), dt.get("Square71x71Logo"), dt.get("Square310x310Logo")]
    assert len(refs) == 6
    for ref in refs:
        assert re.fullmatch(r"Assets\\\w+\.png", ref), ref
        stem = ref.split("\\")[1][:-4]
        assert f"{stem}.png" in assets                                       # запасний файл без масштабу
        assert f"{stem}.scale-200.png" in assets and f"{stem}.scale-100.png" in assets


# ---------------------------------------------------------------- логотипи
def test_logo_sizes_follow_the_msix_scale_rules(assets):
    base = {"Square44x44Logo": (44, 44), "Square150x150Logo": (150, 150), "SmallTile": (71, 71), "StoreLogo": (50, 50),
            "Wide310x150Logo": (310, 150), "LargeTile": (310, 310), "IkrFile": (44, 44)}
    checked = 0
    for name, im in assets.items():
        m = re.fullmatch(r"(\w+?)\.scale-(\d+)\.png", name)
        if m:
            w, h = base[m.group(1)]
            k = int(m.group(2)) / 100
            assert im.size == (round(w * k), round(h * k)), name
            checked += 1
        m = re.fullmatch(r"Square44x44Logo\.targetsize-(\d+)(_altform-\w+)?\.png", name)
        if m:
            assert im.size == (int(m.group(1)),) * 2, name
            checked += 1
    assert checked >= 40


def test_unplated_logos_have_transparent_corners_and_plated_ones_do_not(assets):
    for t in msa.TARGETS:
        assert assets[f"Square44x44Logo.targetsize-{t}_altform-unplated.png"].getpixel((0, 0))[3] <= 16       # кут прозорий (з точністю до згладжування)
        assert assets[f"Square44x44Logo.targetsize-{t}.png"].getpixel((0, 0))[3] == 255


def test_unscaled_fallbacks_are_the_200_percent_images(assets):
    for name in msa.BASE_NAMES:
        assert assets[f"{name}.png"] is assets[f"{name}.scale-200.png"]


def test_committed_assets_are_up_to_date_with_the_generator(assets):
    """готові store/Assets не відстали від tools/make_store_assets.py: ті самі імена й розміри (піксельно не порівнюємо: залежить від версії Pillow)"""
    folder = os.path.join(ROOT, "store", "Assets")
    assert sorted(os.listdir(folder)) == sorted(assets)
    for name, im in assets.items():
        with Image.open(os.path.join(folder, name)) as saved:
            assert saved.size == im.size, name


def test_listing_assets_follow_the_store_rules():
    listing = msa.listing_assets()
    assert {n: im.size for n, im in listing.items()} == {"StoreAppTile_300x300.png": (300, 300)}
    folder = os.path.join(ROOT, "store", "listing_assets")
    # велика афіша лежить готовою (tools/make_listing_art.py), розміри за правилами Store
    sizes = {"StoreBoxArt_1080x1080.png": (1080, 1080), "StoreAppTile_300x300.png": (300, 300), "SuperHeroArt_1920x1080.png": (1920, 1080),
             "StorePosterArt_1440x2160.png": (1440, 2160), "SuperHeroArt_3840x2160.png": (3840, 2160)}
    for name, size in sizes.items():
        with Image.open(os.path.join(folder, name)) as saved:
            assert saved.size == size


def test_store_art_is_strictly_two_colour():
    """афіша Store - чорнило й папір без сірого й згладжування (її малює сама програма, tools/make_listing_art.py)"""
    import numpy as np
    folder = os.path.join(ROOT, "store", "listing_assets")
    for name in ("SuperHeroArt_1920x1080.png", "SuperHeroArt_3840x2160.png", "StoreBoxArt_1080x1080.png", "StorePosterArt_1440x2160.png"):
        with Image.open(os.path.join(folder, name)) as im:
            px = np.asarray(im.convert("L"))
        assert set(np.unique(px)) <= {0, 255}, name


def test_super_hero_has_no_text_baked_in_by_the_generator():
    """правило Store: на 16:9 «супергерої» немає ні назви, ні тексту - генератор не має малювати літер"""
    src = read("tools", "make_store_assets.py")
    hero = src[src.index("def hero("):src.index("def box_art(")]
    assert "_logo(" not in hero and "text(" not in hero and "ImageFont" not in src


# ---------------------------------------------------------------- тека пакета й збірка
def fake_dist(folder) -> str:
    os.makedirs(os.path.join(folder, "_internal"))
    for rel in ("InkReel.exe", "InkReel-Worker.exe", "menu_on.bat", "menu_off.bat", "THIRD_PARTY_NOTICES.md", os.path.join("_internal", "lib.dll")):
        with open(os.path.join(folder, rel), "wb") as f:
            f.write(b"x")
    return str(folder)


@pytest.fixture
def tiny_assets(monkeypatch):
    monkeypatch.setattr(msa, "package_assets", lambda: {"Probe.scale-100.png": Image.new("RGBA", (4, 4), (0, 0, 0, 255))})


def test_stage_copies_the_program_and_adds_manifest_and_assets(tmp_path, tiny_assets):
    dist = fake_dist(tmp_path / "dist" / "InkReel")
    out = mm.stage(dist, str(tmp_path / "dist" / "msix"), "<Package/>")
    names = {os.path.relpath(os.path.join(d, f), out).replace(os.sep, "/") for d, _, fs in os.walk(out) for f in fs}
    assert {"InkReel.exe", "InkReel-Worker.exe", "_internal/lib.dll", "THIRD_PARTY_NOTICES.md", "AppxManifest.xml", "Assets/Probe.scale-100.png"} <= names
    assert not {"menu_on.bat", "menu_off.bat"} & names                      # допоміжні файли збірки «папкою» у пакет не йдуть
    assert open(os.path.join(out, "AppxManifest.xml"), encoding="utf-8").read() == "<Package/>"


def test_stage_without_the_program_says_what_to_run_first(tmp_path):
    os.makedirs(tmp_path / "dist" / "InkReel")
    with pytest.raises(mm.BuildError, match="build_exe.bat"):
        mm.stage(str(tmp_path / "dist" / "InkReel"), str(tmp_path / "dist" / "msix"), "<Package/>")


def test_stage_replaces_a_previous_stage(tmp_path, tiny_assets):
    dist = fake_dist(tmp_path / "dist" / "InkReel")
    stage = str(tmp_path / "dist" / "msix")
    os.makedirs(stage)
    open(os.path.join(stage, "old.txt"), "w").close()
    mm.stage(dist, stage, "<Package/>")
    assert not os.path.exists(os.path.join(stage, "old.txt"))


class Sdk:
    """підроблений Windows SDK: записує команди й створює файли, які створили б справжні makepri/makeappx"""

    def __init__(self, monkeypatch, tools=("makeappx", "makepri", "signtool")):
        self.calls: list[list[str]] = []
        monkeypatch.setattr(mm, "_run", self.run)
        monkeypatch.setattr(mm, "find_sdk_tool", lambda n: f"C:\\sdk\\{n}.exe" if n in tools else None)

    def run(self, cmd):
        self.calls.append(list(cmd))
        tool = cmd[0].rsplit("\\", 1)[-1].replace(".exe", "")
        if tool == "makepri" and cmd[1] == "new":
            with open(cmd[cmd.index("/of") + 1], "wb") as f:
                f.write(b"PRI")
        if tool == "makeappx":
            with open(cmd[cmd.index("/p") + 1], "wb") as f:
                f.write(b"MSIX")

    def tools(self) -> list[str]:
        return [c[0].rsplit("\\", 1)[-1].replace(".exe", "") + ":" + c[1] for c in self.calls]


def test_build_runs_makepri_then_makeappx_and_ships_resources_pri(tmp_path, monkeypatch, tiny_assets):
    sdk = Sdk(monkeypatch)
    dist = fake_dist(tmp_path / "dist" / "InkReel")
    out = str(tmp_path / "dist" / "InkReel.msix")
    assert mm.build(dist, out, REAL, "1.0.0") == out
    assert sdk.tools() == ["makepri:createconfig", "makepri:new", "makeappx:pack"]
    new = sdk.calls[1]
    root = new[new.index("/pr") + 1]
    assert set(os.listdir(root)) == {"Assets", "AppxManifest.xml"}
    assert not os.path.exists(os.path.join(root, "InkReel.exe"))             # індексуємо лише ресурси, а не всю програму
    for flag in ("/cf", "/of"):
        assert not new[new.index(flag) + 1].startswith(root + os.sep)        # конфіг і результат лежать поза індексованою текою
    assert os.path.dirname(new[new.index("/mn") + 1]) == root
    pack = sdk.calls[2]
    stage = pack[pack.index("/d") + 1]
    assert os.path.isfile(os.path.join(stage, "resources.pri")) and pack[pack.index("/p") + 1] == out and "/o" in pack
    assert open(out, "rb").read() == b"MSIX"


def test_build_signs_only_when_asked(tmp_path, monkeypatch, tiny_assets):
    sdk = Sdk(monkeypatch)
    dist = fake_dist(tmp_path / "dist" / "InkReel")
    mm.build(dist, str(tmp_path / "dist" / "a.msix"), REAL, "1.0.0")
    assert not [c for c in sdk.calls if "signtool" in c[0]]
    mm.build(dist, str(tmp_path / "dist" / "b.msix"), REAL, "1.0.0", pfx="dev.pfx", password="pw")
    sign = [c for c in sdk.calls if "signtool" in c[0]][0]
    assert sign[1] == "sign" and "SHA256" in sign and sign[sign.index("/f") + 1] == "dev.pfx" and sign[sign.index("/p") + 1] == "pw"
    assert sign[-1].endswith("b.msix")


def test_build_replaces_an_old_package_and_creates_the_folder(tmp_path, monkeypatch, tiny_assets):
    Sdk(monkeypatch)
    dist = fake_dist(tmp_path / "dist" / "InkReel")
    out = tmp_path / "new" / "deeper" / "InkReel.msix"
    mm.build(dist, str(out), REAL, "1.0.0")
    assert out.read_bytes() == b"MSIX"


def test_build_without_the_sdk_explains_what_to_install(tmp_path, monkeypatch, tiny_assets):
    Sdk(monkeypatch, tools=())
    dist = fake_dist(tmp_path / "dist" / "InkReel")
    with pytest.raises(mm.BuildError, match="Windows SDK"):
        mm.build(dist, str(tmp_path / "dist" / "InkReel.msix"), REAL, "1.0.0")


def test_build_with_a_certificate_but_without_signtool_explains(tmp_path, monkeypatch, tiny_assets):
    Sdk(monkeypatch, tools=("makeappx", "makepri"))
    dist = fake_dist(tmp_path / "dist" / "InkReel")
    with pytest.raises(mm.BuildError, match="signtool"):
        mm.build(dist, str(tmp_path / "dist" / "InkReel.msix"), REAL, "1.0.0", pfx="x.pfx")


def test_stage_only_needs_no_sdk(tmp_path, monkeypatch, tiny_assets):
    sdk = Sdk(monkeypatch, tools=())
    dist = fake_dist(tmp_path / "dist" / "InkReel")
    stage = mm.build(dist, str(tmp_path / "dist" / "InkReel.msix"), REAL, "1.0.0", stage_only=True)
    assert os.path.isfile(os.path.join(stage, "AppxManifest.xml")) and not sdk.calls


def test_main_reports_problems_in_ukrainian_without_a_traceback(tmp_path, capsys):
    os.makedirs(tmp_path / "empty")
    assert mm.main(["--dist", str(tmp_path / "empty"), "--stage-only", "--identity", str(tmp_path / "none.json")]) == 1
    assert "build_exe.bat" in capsys.readouterr().out


def test_main_warns_that_a_dev_package_cannot_go_to_the_store(tmp_path, monkeypatch, capsys, tiny_assets):
    Sdk(monkeypatch, tools=())
    dist = fake_dist(tmp_path / "dist" / "InkReel")
    assert mm.main(["--dist", dist, "--stage-only", "--identity", str(tmp_path / "none.json")]) == 0
    out = capsys.readouterr().out
    assert "ТЕСТОВИЙ" in out and "InkReel.Dev" in out


def test_main_refuses_a_zero_version_for_a_real_identity(tmp_path, monkeypatch, capsys, tiny_assets):
    import json
    f = tmp_path / "identity.json"
    f.write_text(json.dumps(REAL), encoding="utf-8")
    monkeypatch.setattr(mm, "app_version", lambda: "0.9.0")
    dist = fake_dist(tmp_path / "dist" / "InkReel")
    assert mm.main(["--dist", dist, "--stage-only", "--identity", str(f)]) == 1
    assert "1.0.0" in capsys.readouterr().out


def test_find_sdk_tool_picks_the_newest_sdk_by_number_not_by_text(tmp_path, monkeypatch):
    for ver in ("10.0.19041.0", "10.0.22621.0", "10.0.9999.0"):
        d = tmp_path / "Windows Kits" / "10" / "bin" / ver / "x64"
        d.mkdir(parents=True)
        (d / "makeappx.exe").write_bytes(b"")
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path))
    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.setattr(mm.shutil, "which", lambda n: None)
    monkeypatch.setattr(mm, "SDK_FALLBACK", "")                    # на ПК з установленим Windows SDK запасний шлях знайшов би справжній signtool
    assert "10.0.22621.0" in mm.find_sdk_tool("makeappx")
    assert mm.find_sdk_tool("signtool") is None


def test_find_sdk_tool_prefers_the_one_on_path(monkeypatch):
    monkeypatch.setattr(mm.shutil, "which", lambda n: "D:\\tools\\" + n + ".exe")
    assert mm.find_sdk_tool("makepri") == "D:\\tools\\makepri.exe"


# ---------------------------------------------------------------- тексти сторінки
def sections_of(name: str) -> list[tuple[str, str]]:
    out, title, buf = [], None, []
    for line in read("store", name).splitlines():
        if line.startswith("## "):
            if title is not None:
                out.append((title, "\n".join(buf).strip()))
            title, buf = line[3:].strip(), []
        elif title is not None:
            buf.append(line)
    out.append((title, "\n".join(buf).strip()))
    return out


def listing(name: str) -> dict:
    s = sections_of(name)
    by = {t: b for t, b in s}
    return {"all": s, "by": by}


@pytest.mark.parametrize("name,lang", [("listing_en.md", "en"), ("listing_uk.md", "uk")])
def test_listing_fits_partner_center_limits(name, lang):
    titles = [t for t, _ in sections_of(name)]
    texts = [b for _, b in sections_of(name)]
    if lang == "en":
        assert titles[:5] == ["Product name", "Short description", "Description", "Product features", "Search terms"]
    else:
        assert titles[:5] == ["Назва", "Короткий опис", "Опис", "Ключові можливості", "Пошукові слова"]
    product, short, desc, features, terms = texts[:5]
    assert product == "InkReel"
    assert 0 < len(short) <= 270                                             # Microsoft: враховуються перші 270 символів
    assert 0 < len(desc) <= 10000
    assert "<" not in desc and ">" not in desc and "http" not in desc.lower()       # лише простий текст, без HTML і посилань
    feats = [ln[2:].strip() for ln in features.splitlines() if ln.startswith("- ")]
    assert 1 <= len(feats) <= 20 and all(0 < len(f) <= 200 for f in feats)
    words = [ln.strip() for ln in terms.splitlines() if ln.strip()]
    assert 1 <= len(words) <= 7 and all(len(w) <= 30 for w in words)


def test_both_listings_have_the_same_shape():
    en, uk = listing("listing_en.md")["all"], listing("listing_uk.md")["all"]
    count = lambda body: len([ln for ln in body.splitlines() if ln.startswith("- ")])
    assert count(en[3][1]) == count(uk[3][1])
    assert len(en[4][1].split()) > 0 and len(en[4][1].splitlines()) == len(uk[4][1].splitlines())


def test_listings_do_not_promise_what_the_program_cannot_do():
    en, uk = read("store", "listing_en.md"), read("store", "listing_uk.md")
    n = len(palette.PALETTES)
    assert re.search(rf"\b{n} two-colour palettes", en) and re.search(rf"\b{n} двокольорових палітр", uk)
    for fmt in ("MP4", "MKV", "MOV", "AVI", "WebM", "GIF"):
        assert fmt in en and fmt in uk
    assert {f.split()[0] for f in video.FORMATS} >= {"MP4", "MKV", "MOV", "AVI", "WebM", "GIF"}
    for ext in ("HEIC", "AVIF", "TIFF", "PSD"):
        assert ext in en and ext in uk and "." + ext.lower() in media.IMAGE_EXT
    assert "144p" in en and "144p" in uk and "144p" in video.SIZES


def test_certification_notes_describe_real_shortcuts():
    notes = dict(listing("listing_en.md")["all"])["Notes for certification"]
    gui = read("benvid", "gui.py")
    for key, bind in (("Ctrl+O", "<Control-o>"), ("Ctrl+S", "<Control-s>")):
        assert key in notes and bind in gui
    cap = dict(listing("listing_en.md")["all"])["Restricted capability: runFullTrust"]
    assert "runFullTrust" in cap and "does not use the network" in cap


# ---------------------------------------------------------------- політика приватності
def test_privacy_policy_is_bilingual_and_has_one_contact_placeholder_per_language():
    md = read("store", "PRIVACY.md")
    assert "## English" in md and "## Українською" in md and md.count(mpp.PLACEHOLDER) == 2


def test_privacy_claims_hold_for_the_code():
    """політика каже «не підключається до інтернету» - стежимо, щоб код цьому не суперечив"""
    banned = re.compile(r"^\s*(?:import|from)\s+(urllib|requests|http|httpx|ftplib|smtplib|xmlrpc|websocket|websockets|aiohttp|telnetlib)\b", re.M)
    package = os.path.join(ROOT, "benvid")
    for name in sorted(os.listdir(package)):
        if not name.endswith(".py"):
            continue
        src = read("benvid", name)
        assert not banned.search(src), f"{name}: мережевий імпорт - оновіть store/PRIVACY.md та сторінку Store"
        if name != "single.py":
            assert not re.search(r"^\s*(?:import|from)\s+socket\b", src, re.M), f"{name}: сокети лише в single.py (локальна адреса)"
    single = read("benvid", "single.py")
    assert "127.0.0.1" in single and "0.0.0.0" not in single                  # слухаємо лише локальну адресу
    assert read("benvid", "gui.py").count("webbrowser.open") == 2, "нове відкриття браузера: перегляньте store/PRIVACY.md"       # лише кнопка донату (її в Store сховано)


def test_privacy_page_replaces_the_contact_and_marks_languages():
    page = mpp.render(read("store", "PRIVACY.md"), "me@example.com")
    assert mpp.PLACEHOLDER not in page and page.count('href="mailto:me@example.com"') == 2
    assert '<section lang="en">' in page and '<section lang="uk">' in page
    assert page.startswith("<!doctype html>") and "<script" not in page.lower()


def test_privacy_page_escapes_the_contact():
    page = mpp.render("Зв'язок: __CONTACT__", '<script>alert("x")</script>')
    assert "<script" not in page.lower() and "&lt;script&gt;" in page
    assert 'href="https://example.com/help"' in mpp.render("__CONTACT__", "https://example.com/help")


def test_privacy_page_needs_a_contact():
    with pytest.raises(ValueError, match="контакт"):
        mpp.render("__CONTACT__", "   ")


def test_privacy_page_main_without_contact_fails_and_with_it_writes(tmp_path, monkeypatch, capsys):
    out = tmp_path / "privacy.html"
    monkeypatch.setattr(meta, "CONTACT", "")
    assert mpp.main(["--out", str(out)]) == 1 and not out.exists() and "CONTACT" in capsys.readouterr().out
    assert mpp.main(["--out", str(out), "--contact", "a@b.ua"]) == 0 and "a@b.ua" in out.read_text(encoding="utf-8")
    monkeypatch.setattr(meta, "CONTACT", "x@y.ua")
    assert mpp.main(["--out", str(out)]) == 0 and "x@y.ua" in out.read_text(encoding="utf-8")


def test_markdown_converter_basics():
    html = mpp.md_to_html("# T\n\n*i* **b** `c` <x>\nдругий рядок\n\n---\n\n## English\n\ntext")
    assert "<h1>T</h1>" in html and "<em>i</em>" in html and "<strong>b</strong>" in html and "<code>c</code>" in html
    assert "&lt;x&gt; другий рядок" in html                                  # рядки одного абзацу склеєні, HTML екрановано
    assert html.count("<hr>") == 1 and html.rstrip().endswith("</section>")


# ---------------------------------------------------------------- ліцензії
def make_dist(root, name, version, files: dict[str, str], requires=(), license_expr="MIT"):
    di = root / f"{name.replace('-', '_')}-{version}.dist-info"
    di.mkdir(parents=True)
    meta_lines = [f"Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}", f"License-Expression: {license_expr}"]
    meta_lines += [f"Requires-Dist: {r}" for r in requires]
    (di / "METADATA").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    record = [f"{di.name}/METADATA,,"]
    for rel, text in files.items():
        p = di / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        record.append(f"{di.name}/{rel},,")
    (di / "RECORD").write_text("\n".join(record) + "\n", encoding="utf-8")


def test_requirements_are_parsed_to_names(tmp_path):
    f = tmp_path / "requirements.txt"
    f.write_text("numpy\nPillow>=10 # зображення\nwindnd; sys_platform == 'win32'\n-r other.txt\n\npillow_heif\nopencv-python==5.0.0\n", encoding="utf-8")
    assert cl.requirement_names(str(f)) == ["numpy", "pillow", "windnd", "pillow-heif", "opencv-python"]


def test_collect_copies_license_files_of_requirements_and_their_dependencies(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    make_dist(site, "pkg-a", "1.0", {"licenses/LICENSE": "A licence", "licenses/sub/dir/COPYING.txt": "nested", "README.md": "no"},
              requires=["pkg-b>=2", 'extra-c; extra == "dev"'])
    make_dist(site, "pkg-b", "2.0", {"LICENSE.txt": "B licence", "NOTICE": "notice"}, license_expr="BSD-3-Clause")
    make_dist(site, "pkg-d", "3.0", {"AUTHORS": "nobody"})                    # без файлу ліцензії
    make_dist(site, "extra-c", "9.0", {"LICENSE": "never copied"})
    req = tmp_path / "requirements.txt"
    req.write_text("pkg-a\npkg-d\n", encoding="utf-8")
    dest = tmp_path / "dest"
    dest.mkdir()
    res = cl.collect(str(dest), str(req), search_path=[str(site)])
    base = dest / "licenses" / "python"
    assert (base / "pkg-a" / "LICENSE").read_text(encoding="utf-8") == "A licence"
    assert (base / "pkg-a" / "sub" / "dir" / "COPYING.txt").read_text(encoding="utf-8") == "nested"
    assert (base / "pkg-b" / "LICENSE.txt").exists() and (base / "pkg-b" / "NOTICE").exists()
    assert not (base / "pkg-a" / "README.md").exists() and not (base / "extra-c").exists()
    assert res["missing"] == ["pkg-d"]
    index = (base / "INDEX.txt").read_text(encoding="utf-8")
    assert "pkg-a 1.0 - MIT" in index and "pkg-b 2.0 - BSD-3-Clause" in index and "pkg-d" in index and "extra-c" not in index


def test_collect_survives_a_requirement_that_is_not_installed(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    req = tmp_path / "requirements.txt"
    req.write_text("windnd; sys_platform == 'win32'\n", encoding="utf-8")
    dest = tmp_path / "dest"
    dest.mkdir()
    res = cl.collect(str(dest), str(req), search_path=[str(site)])
    assert res["packages"] == [] and (dest / "licenses" / "python" / "INDEX.txt").exists()


def test_copy_static_brings_notices_and_license_texts(tmp_path):
    root = tmp_path / "proj"
    (root / "licenses").mkdir(parents=True)
    (root / "THIRD_PARTY_NOTICES.md").write_text("n", encoding="utf-8")
    (root / "licenses" / "GPL-3.0.txt").write_text("gpl", encoding="utf-8")
    dest = tmp_path / "dest"
    dest.mkdir()
    written = cl.copy_static(str(dest), str(root))
    assert (dest / "THIRD_PARTY_NOTICES.md").read_text() == "n" and (dest / "licenses" / "GPL-3.0.txt").read_text() == "gpl" and len(written) == 2


def test_collect_main_needs_an_existing_folder(tmp_path, capsys):
    assert cl.main([str(tmp_path / "missing")]) == 2 and "використання" in capsys.readouterr().out
    assert cl.main([]) == 2


def test_notices_mention_every_requirement_and_every_license_text():
    notices = read("THIRD_PARTY_NOTICES.md").lower()
    for name in cl.requirement_names(os.path.join(ROOT, "requirements.txt")):
        assert name in notices, f"{name} не згадано в THIRD_PARTY_NOTICES.md"
    for lic in ("GPL-3.0.txt", "GPL-2.0.txt", "LGPL-3.0.txt", "LGPL-2.1.txt"):
        assert lic.lower() in notices
        text = read("licenses", lic)
        assert "GNU" in text[:200] and len(text) > 5000


def test_build_scripts_ship_licenses_and_the_spec_knows_the_store_module():
    build = read("scripts", "build_exe.bat")
    assert build.index("pyinstaller inkreel.spec") < build.index("tools\\collect_licenses.py dist\\InkReel")
    assert '"benvid.store"' in read("inkreel.spec")
    bat = read("scripts", "build_msix.bat")
    assert "tools\\make_msix.py %*" in bat and "build_exe.bat" in bat


def test_about_tab_points_to_the_notices_in_both_languages():
    from benvid import guide, i18n
    try:
        for lang in (i18n.UK, i18n.EN):
            i18n.set_lang(lang)
            assert "THIRD_PARTY_NOTICES.md" in guide.about_text()
    finally:
        i18n.set_lang(i18n.UK)


# ---------------------------------------------------------------- документація не відстає від файлів
GENERATED = {"store\\identity.json", "store\\privacy.html", "store\\InkReelDev.pfx", "store\\InkReelDev.cer"}


def test_store_guide_refers_only_to_files_that_exist():
    text = read("store", "STORE.md")
    seen = 0
    for token in re.findall(r"`([^`\n]+)`", text):
        t = token.strip()
        if re.fullmatch(r"(?:store|tools|benvid|licenses)\\[\w.\-\\]+", t) or re.fullmatch(r"[\w\-]+\.(?:bat|md|iss|spec)", t):
            if t in GENERATED or t.startswith("dist\\"):
                continue
            seen += 1
            parts = t.split("\\")
            assert os.path.exists(os.path.join(ROOT, *parts)) or os.path.exists(os.path.join(ROOT, "store", *parts)), f"STORE.md згадує {t}, а такого файла немає"
    assert seen >= 15


def test_store_guide_tells_the_one_decision_about_the_version():
    text = read("store", "STORE.md")
    assert '__version__ = "1.0.0"' in text and "identity.json" in text and "make_privacy_page.py" in text
