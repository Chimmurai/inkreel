"""Рядок «запуск: …» у inkreel.log: версія, система й бібліотеки (щоб за журналом було видно, на чому працювала програма), але нічого особистого."""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid import __version__, log  # noqa: E402


class _Grab(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _run(monkeypatch):
    lg = logging.getLogger("inkreel.test_env")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    h = _Grab()
    lg.handlers[:] = [h]
    monkeypatch.setattr(log, "get", lambda: lg)
    log.log_environment()
    return h.lines


def test_start_line_has_version_system_and_libraries(monkeypatch):
    import numpy  # noqa: F401 - щоб бібліотека точно була серед завантажених
    lines = _run(monkeypatch)
    assert len(lines) == 1 and lines[0].startswith("запуск: InkReel " + __version__)
    assert "Python " in lines[0] and "numpy " in lines[0] and "логічних ядер" in lines[0]


def test_start_line_has_nothing_personal(monkeypatch):
    line = _run(monkeypatch)[0]
    home = os.path.expanduser("~")
    for secret in (home, os.environ.get("USERNAME", ""), os.environ.get("USER", ""), os.getcwd()):
        if len(secret) > 3:
            assert secret not in line, secret


def test_failure_inside_never_breaks_startup(monkeypatch):
    monkeypatch.setattr(log, "get", lambda: (_ for _ in ()).throw(RuntimeError("журнал недоступний")))
    log.log_environment()                    # не кидає
