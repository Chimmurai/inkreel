"""Публічний репозиторій не повинен містити робочих нотаток, файлів помічників, ключів і службових звітів. Тест дивиться на те, що відстежує git
(`git ls-files`), тож спрацьовує до публікації; пропускається поза git-копією."""
from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORBIDDEN = re.compile(
    r"(^|/)(CLAUDE|AGENTS|GEMINI)\.md$|(^|/)\.claude(/|$)|(^|/)\.cursor(/|$)|(^|/)memory(/|$)|(^|/)scratch(/|$)|(^|/)reports/"
    r"|docs/(PROGRAM_STATUS|HANDOFF_ARCHIVE|COMPLIANCE|QT_PORT)\.md$|HANDOFF|(^|/)NOTES[^/]*\.md$|\.local\.md$|identity\.json$"
    r"|\.pfx$|\.pem$|\.key$|(^|/)\.env|(^|/)WebSite/|(^|/)weights/[^/]+/last\.pt$",
    re.IGNORECASE)


def _tracked():
    if not shutil.which("git") or not os.path.isdir(os.path.join(ROOT, ".git")):
        pytest.skip("не git-копія")
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8", "replace")
    return [p for p in out.split("\0") if p]


def test_no_working_notes_or_private_files_are_tracked():
    bad = [p for p in _tracked() if FORBIDDEN.search(p)]
    assert not bad, "у публічному репозиторії не мусить бути: " + ", ".join(bad)


def test_tracked_text_files_do_not_mention_assistants():
    """тексти в репозиторії (код, документи, листинги) не згадують помічників і службові сесії; це перевіряємо лише за вмістом файлів,
    не за повідомленнями комітів"""
    pat = re.compile(r"claude|anthropic|chatgpt|copilot", re.IGNORECASE)
    hits = []
    for p in _tracked():
        if p == "tests/test_repo_clean.py" or not p.lower().endswith((".py", ".md", ".txt", ".iss", ".bat", ".spec", ".json", ".xml", ".ps1", ".toml", ".cfg", ".yml", ".yaml")):
            continue
        try:
            with open(os.path.join(ROOT, p), encoding="utf-8", errors="ignore") as f:
                if pat.search(f.read()):
                    hits.append(p)
        except OSError:
            continue
    assert not hits, "згадки помічників у відстежуваних файлах: " + ", ".join(hits)
