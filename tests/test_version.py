"""Версія в одному місці: інсталятор і історія версій не відстають від benvid.__version__ (Store та 'Про програму' беруть її звідти)."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid import __version__  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def test_version_looks_like_major_minor_patch():
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__), __version__


def test_installer_has_the_same_version():
    assert re.search(r'#define AppVer "([^"]+)"', _read("installer.iss")).group(1) == __version__


def test_changelog_top_entry_is_the_current_version():
    first = re.search(r"^## (\d+\.\d+\.\d+)", _read("CHANGELOG.md"), re.M).group(1)
    assert first == __version__, f"у CHANGELOG.md зверху {first}, а версія програми {__version__}"
