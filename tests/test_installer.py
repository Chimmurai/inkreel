"""Інсталятор (installer.iss): компілятора Inno Setup тут немає, тож перевіряємо текст: кодування, синхронність зі списками форматів
програми, повноту пунктів контекстного меню Провідника, переклади {cm:…} та узгодженість з inkreel.spec."""
from __future__ import annotations

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import sync_installer as si                                              # noqa: E402
from benvid import media, shellmenu, video                              # noqa: E402

CMD_OPEN = '"{app}\\{#Exe}" "%1"'
CMD_QUICK = '"{app}\\{#Exe}" --quick "%1"'


# ---------------------------------------------------------------- розбір .iss (спільні з tools/sync_installer.py)
params = si.parse_params
sections = si.sections


@pytest.fixture(scope="module")
def iss() -> str:
    return si.read()


@pytest.fixture(scope="module")
def reg(iss) -> list[dict[str, str]]:
    return [params(ln) for ln in sections(iss)["Registry"]]


def ctx_lines(reg):
    """рядки контекстного меню: {розширення: {дієслово: [параметри…]}}"""
    out: dict[str, dict[str, list[dict[str, str]]]] = {}
    for p in reg:
        m = re.fullmatch(r"Software\\Classes\\SystemFileAssociations\\(\.\w+)\\shell\\(InkReel\.\w+)(\\command)?", p["Subkey"])
        if m:
            out.setdefault(m.group(1), {}).setdefault(m.group(2), []).append(p)
    return out


# ---------------------------------------------------------------- файл
def test_iss_is_utf8_with_bom():
    with open(si.ISS, "rb") as f:
        raw = f.read()
    assert raw[:3] == b"\xef\xbb\xbf", "без BOM Inno Setup читає кирилицю як ANSI й показує абракадабру"
    raw.decode("utf-8")


def test_generated_blocks_are_in_sync(iss):
    assert si.render(iss) == iss, "installer.iss розійшовся з форматами програми: python tools/sync_installer.py"


def test_sync_tool_check_mode_passes():
    assert si.main(["--check"]) == 0


def test_sync_tool_refuses_file_without_markers():
    with pytest.raises(SystemExit):
        si.render("[Registry]\nRoot: HKCU; Subkey: \"x\"\n")


# ---------------------------------------------------------------- «Відкрити за допомогою»
def test_supported_types_equal_program_formats(reg):
    got = [p["ValueName"] for p in reg if p["Subkey"].endswith("\\SupportedTypes")]
    want = set(video.VIDEO_EXT) | set(media.IMAGE_EXT) | set(media.GIF_EXT)
    assert len(got) == len(set(got)), "розширення в SupportedTypes повторюються"
    assert set(got) == want, f"немає: {sorted(want - set(got))}; зайве: {sorted(set(got) - want)}"


# ---------------------------------------------------------------- контекстне меню Провідника
def test_context_menu_covers_photos_and_gifs_not_video(reg):
    got = set(ctx_lines(reg))
    want = (set(media.IMAGE_EXT) | set(media.GIF_EXT)) - shellmenu.SKIP
    assert got == want, f"немає: {sorted(want - got)}; зайве: {sorted(got - want)}"
    assert not got & set(video.VIDEO_EXT)
    assert ".gif" in got and ".png" in got and ".jpg" in got and ".webp" in got and ".heic" in got


def test_every_extension_has_both_verbs_with_three_entries(reg):
    for ext, verbs in ctx_lines(reg).items():
        assert sorted(verbs) == ["InkReel.Open", "InkReel.Quick"], ext
        for verb, ps in verbs.items():
            assert len(ps) == 3, (ext, verb)
            names = sorted((p.get("ValueName", ""), p["Subkey"].endswith("\\command")) for p in ps)
            assert names == [("", True), ("Icon", False), ("MUIVerb", False)], (ext, verb)


def test_context_menu_entries_belong_to_the_task_and_are_removed_on_uninstall(reg):
    for ext, verbs in ctx_lines(reg).items():
        for verb, ps in verbs.items():
            assert all(p["Root"] == "HKCU" for p in ps), "меню лише для поточного користувача: права адміністратора не потрібні"
            assert all(p.get("Tasks") == "ctxmenu" for p in ps), (ext, verb)
            verb_line = [p for p in ps if p.get("ValueName") == "MUIVerb"][0]
            assert "uninsdeletekey" in verb_line.get("Flags", ""), (ext, verb)     # видалення ключа прибирає й command під ним


def test_context_menu_commands_and_captions(reg):
    for ext, verbs in ctx_lines(reg).items():
        for verb, ps in verbs.items():
            cmd = [p for p in ps if p["Subkey"].endswith("\\command")][0]
            cap = [p for p in ps if p.get("ValueName") == "MUIVerb"][0]
            ico = [p for p in ps if p.get("ValueName") == "Icon"][0]
            if verb == "InkReel.Open":
                assert cmd["ValueData"] == CMD_OPEN and cap["ValueData"] == "{cm:CtxOpen}", ext
            else:
                assert cmd["ValueData"] == CMD_QUICK and cap["ValueData"] == "{cm:CtxQuick}", ext
            assert ico["ValueData"] == '"{app}\\{#Exe}",0', ext


def test_quick_flag_is_handled_by_the_program():
    with open(os.path.join(ROOT, "benvid", "__main__.py"), encoding="utf-8") as f:
        src = f.read()
    assert re.search(r'==\s*\["--quick"\]', src) and re.search(r"from \.quick import main", src), \
        "команда меню `InkReel.exe --quick файл` має запускати benvid.quick.main"


# ---------------------------------------------------------------- переклади, завдання, лапки
def test_custom_messages_exist_in_both_languages(iss):
    secs = sections(iss)
    defined = {}
    for ln in secs["CustomMessages"]:
        key, _, val = ln.partition("=")
        lang, _, name = key.partition(".")
        assert val.strip(), f"порожній переклад {key}"
        defined.setdefault(name, set()).add(lang)
    langs = {params(ln)["Name"] for ln in secs["Languages"]}
    assert langs == {"ukrainian", "english"}
    used = set(re.findall(r"\{cm:(\w+)\}", "\n".join(v for k, v in secs.items() if k != "CustomMessages" for v in v)))
    assert used, "у файлі немає жодного {cm:…}"
    for name in used:
        assert defined.get(name) == langs, f"{{cm:{name}}} має бути і українською, і англійською: є {defined.get(name)}"
    for name, have in defined.items():
        assert have == langs, f"переклад {name} є лише для {have}"
    ua = [ln for ln in secs["CustomMessages"] if ln.startswith("ukrainian.")]
    assert any(re.search("[А-Яа-яІіЇїЄє]", ln) for ln in ua)


def test_tasks_used_are_declared(iss, reg):
    secs = sections(iss)
    declared = {params(ln)["Name"] for ln in secs["Tasks"]}
    used = {p["Tasks"] for sec in ("Registry", "Icons", "Files", "Run") for p in map(params, secs.get(sec, [])) if "Tasks" in p}
    assert used <= declared, f"невідомі завдання: {sorted(used - declared)}"
    assert "ctxmenu" in declared


def test_registry_lines_have_balanced_quotes(iss):
    for ln in sections(iss)["Registry"]:
        assert ln.count('"') % 2 == 0, ln
        p = params(ln)
        assert {"Root", "Subkey", "ValueType"} <= set(p), ln
        assert p["Root"] == "HKCU", ln


def test_no_duplicate_registry_values(reg):
    seen = set()
    for p in reg:
        k = (p["Subkey"].lower(), p.get("ValueName", ""))
        assert k not in seen, k
        seen.add(k)


# ---------------------------------------------------------------- узгодженість зі збіркою
def test_installer_matches_pyinstaller_spec(iss):
    with open(os.path.join(ROOT, "inkreel.spec"), encoding="utf-8") as f:
        spec = f.read()
    exe = re.search(r'#define Exe "([^"]+)"', iss).group(1)
    assert f'name="{exe[:-4]}"' in spec, "ім'я exe в інсталяторі й у inkreel.spec має збігатися"
    src = [params(ln)["Source"] for ln in sections(iss)["Files"]]
    assert src == ["dist\\InkReel\\*", "benvid\\assets\\ikr.ico"], "інсталятор бере те, що збирає COLLECT(name=\"InkReel\"), і значок файлів .ikr"
    assert re.search(r'COLLECT\([^)]*name="InkReel"', spec)
    assert os.path.exists(os.path.join(ROOT, "benvid", "assets", "inkreel.ico"))                 # SetupIconFile


def test_installer_mutex_matches_program():
    with open(si.ISS, encoding="utf-8-sig") as f:
        iss = f.read()
    with open(os.path.join(ROOT, "benvid", "procs.py"), encoding="utf-8") as f:
        procs = f.read()
    name = re.search(r"^AppMutex=(\S+)", iss, re.M).group(1)
    assert f'"{name}"' in procs, "AppMutex інсталятора має збігатися з м'ютексом, який створює програма"


def test_spec_bundles_quick_module():
    with open(os.path.join(ROOT, "inkreel.spec"), encoding="utf-8") as f:
        spec = f.read()
    assert '"benvid.quick"' in spec and '"benvid.eggs"' in spec


def test_params_parser_handles_inno_quoting():
    p = params('Root: HKCU; Subkey: "A\\B"; ValueData: """{app}\\x.exe"" --quick ""%1"""; Tasks: ctxmenu')
    assert p == {"Root": "HKCU", "Subkey": "A\\B", "ValueData": '"{app}\\x.exe" --quick "%1"', "Tasks": "ctxmenu"}
    assert params('Name: "a;b"; Flags: x')["Name"] == "a;b"


def test_bat_files_use_crlf_and_build_script_creates_menu_helpers():
    """cmd.exe із LF-кінцями рядків плутає мітки (goto) і дужки - усі .bat (корінь і scripts) мають бути з CRLF"""
    import glob
    bats = glob.glob(os.path.join(ROOT, "*.bat")) + glob.glob(os.path.join(ROOT, "scripts", "*.bat"))
    assert bats
    for f in bats:
        data = open(f, "rb").read()
        assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b""), os.path.basename(f)
    build = open(os.path.join(ROOT, "scripts", "build_exe.bat"), encoding="utf-8").read()
    assert "--register-menu" in build and "--unregister-menu" in build and "sync_installer.py" in build
    for lab in re.findall(r"goto (\w+)", build):
        assert re.search(rf"^:{lab}\s*$", build, re.M), f"немає мітки :{lab}"


def test_build_can_use_a_frozen_dependency_list():
    """випуск збирається з тих версій бібліотек, на яких його перевірено (requirements.lock.txt), якщо файл є; зафіксувати їх можна з меню збірки"""
    rd = lambda *p: open(os.path.join(ROOT, *p), encoding="utf-8").read()  # noqa: E731
    build, freeze, menu = rd("scripts", "build_exe.bat"), rd("scripts", "freeze_deps.bat"), rd("scripts", "build.bat")
    assert "if exist requirements.lock.txt" in build and "pip install -r requirements.lock.txt" in build and "pip install -r requirements.txt" in build
    assert "pip freeze" in freeze and "> requirements.lock.txt" in freeze
    assert "freeze_deps.bat" in menu and "choice /c 12340" in menu and "if %K%==5 exit /b 0" in menu


# ---------------------------------------------------------------- тип файлу .ikr
def test_ikr_file_type_has_a_name_an_icon_and_an_open_command(reg, iss):
    """без власного типу Провідник показує «Файл IKR» з безіменним значком: інсталятор реєструє тип, значок і відкриття в InkReel"""
    by = {(p["Subkey"], p.get("ValueName", "")): p for p in reg}
    assert by[("Software\\Classes\\.ikr", "")]["ValueData"] == "InkReel.Baked"
    assert ("Software\\Classes\\.ikr\\OpenWithProgids", "InkReel.Baked") in by
    assert by[("Software\\Classes\\InkReel.Baked", "")]["ValueData"] == "{cm:IkrType}"
    assert by[("Software\\Classes\\InkReel.Baked\\DefaultIcon", "")]["ValueData"] == "{app}\\ikr.ico"
    assert by[("Software\\Classes\\InkReel.Baked\\shell\\open\\command", "")]["ValueData"] == CMD_OPEN
    assert "uninsdeletekey" in by[("Software\\Classes\\.ikr", "")]["Flags"] and "uninsdeletekey" in by[("Software\\Classes\\InkReel.Baked", "")]["Flags"]
    assert "ChangesAssociations=yes" in iss, "після реєстрації Провідник має оновити значки"
    assert os.path.isfile(os.path.join(ROOT, "benvid", "assets", "ikr.ico"))
    from PIL import Image
    with Image.open(os.path.join(ROOT, "benvid", "assets", "ikr.ico")) as ico:
        assert {(16, 16), (32, 32), (48, 48), (256, 256)} <= set(ico.info["sizes"])
