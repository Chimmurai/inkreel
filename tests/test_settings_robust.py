"""Зіпсований або чужий файл налаштувань (settings.json) не повинен заважати запуску: ні вручну відредагований, ні лишений старішою чи новішою версією
(у Microsoft Store оновлення програми зберігає налаштування користувача, тож нова версія мусить пережити будь-що, що лежить у файлі)."""
import json
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid import engine  # noqa: E402

JUNK_UI = {"volume": "голосно", "method": "x", "gdens3": None, "size": 123, "fps": "abc", "palette": 5, "pal_inv": "ні", "fmt": 7, "quality": [],
           "audio": "може", "guide": {}, "lang": 3, "glyph": "так"}
CASES = {
    "not_json": "{це не json",
    "empty_file": "",
    "list_on_top": "[1, 2, 3]",
    "ui_is_a_list": json.dumps({"ui": [1, 2]}),
    "ui_is_null": json.dumps({"ui": None}),
    "wrong_types": json.dumps({"ui": JUNK_UI}),
    "huge_numbers": json.dumps({"ui": {"volume": 10 ** 12, "method": -5, "gdens3": 99, "fps": -1e9, "size": "99999x99999"}}),
    "unknown_keys": json.dumps({"ui": {"from_the_future": {"a": 1}, "lang": "uk"}, "other": 1}),
    "nan_and_inf": '{"ui": {"fps": NaN, "volume": Infinity}}',
}


def test_loader_always_returns_a_dict_with_ui(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    monkeypatch.setattr(engine, "SETTINGS_FILE", str(p))
    monkeypatch.setattr(engine, "_OLD", str(tmp_path / "old.json"))
    for text in CASES.values():
        p.write_text(text, encoding="utf-8")
        d = engine.load_settings()
        assert isinstance(d, dict) and isinstance(d["ui"], dict)


def test_saving_leaves_no_temp_file_and_is_readable(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    monkeypatch.setattr(engine, "SETTINGS_FILE", str(p))
    engine.save_settings({"ui": {"lang": "uk", "fps": 24.0}})
    assert json.loads(p.read_text(encoding="utf-8"))["ui"]["fps"] == 24.0
    assert not (tmp_path / "s.json.tmp").exists()


def test_clean_ui_drops_what_the_window_cannot_use_and_keeps_the_rest():
    ui = engine.clean_ui({"palette": 5, "lang": "uk", "volume": 250, "method": 9, "gdens3": 0, "fps": float("nan"), "audio": "ні", "pal_inv": 1,
                          "glyph": True, "size": "720p", "future_key": {"a": 1}, "fmt": "x" * 500})
    assert "palette" not in ui and "fmt" not in ui and "fps" not in ui and "audio" not in ui    # чужий тип, NaN, надто довгий рядок
    assert ui["volume"] == 100 and ui["method"] == 6 and ui["gdens3"] == 1                      # число поза межами -> найближча межа
    assert ui["pal_inv"] is True and ui["glyph"] is True and ui["lang"] == "uk" and ui["size"] == "720p"
    assert ui["future_key"] == {"a": 1}


def test_clean_ui_keeps_a_normal_saved_file_unchanged():
    ui = {"size": "Авто", "fmt": "MP4 (H.264)", "quality": "Висока", "audio": True, "palette": "Класика", "pal_inv": False, "method": 5, "glyph": False,
          "gdens3": 3, "volume": 80, "lang": "en", "guide": "start", "fps": 24.0}
    assert engine.clean_ui(ui) == ui
    assert engine.clean_ui(None) == {} and engine.clean_ui([1]) == {}
