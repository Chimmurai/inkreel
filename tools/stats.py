"""Статистика проєкту: файли, рядки коду, розмір, тести за теками.   python tools/stats.py [--md]"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {"__pycache__", ".pytest_cache", ".git", "dist", "build", "reports", ".venv", ".venv_train", ".venv_dml"}
AREAS = ["benvid", "tests", "tools", "scripts", "store", "licenses", "installer", "weights", "docs"]


def walk(base):
    for d, dirs, files in os.walk(base):
        dirs[:] = [x for x in dirs if x not in SKIP]
        for f in files:
            yield os.path.join(d, f)


def lines(p):
    try:
        with open(p, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def main():
    rows, tot = [], [0, 0, 0, 0]
    for a in AREAS:
        base = os.path.join(ROOT, a)
        if not os.path.isdir(base):
            continue
        fs = list(walk(base))
        py = [f for f in fs if f.endswith(".py")]
        size = sum(os.path.getsize(f) for f in fs)
        loc = sum(lines(f) for f in py)
        rows.append((a, len(fs), len(py), loc, size))
        tot = [tot[0] + len(fs), tot[1] + len(py), tot[2] + loc, tot[3] + size]
    root = [f for f in os.listdir(ROOT) if os.path.isfile(os.path.join(ROOT, f))]
    rs = sum(os.path.getsize(os.path.join(ROOT, f)) for f in root)
    rows.append(("(корінь)", len(root), sum(f.endswith(".py") for f in root), sum(lines(os.path.join(ROOT, f)) for f in root if f.endswith(".py")), rs))
    tot = [tot[0] + len(root), tot[1] + rows[-1][2], tot[2] + rows[-1][3], tot[3] + rs]
    print(f"{'тека':<12}{'файлів':>8}{'.py':>6}{'рядків .py':>12}{'МБ':>8}")
    for r in rows:
        print(f"{r[0]:<12}{r[1]:>8}{r[2]:>6}{r[3]:>12}{r[4] / 1048576:>8.2f}")
    print(f"{'РАЗОМ':<12}{tot[0]:>8}{tot[1]:>6}{tot[2]:>12}{tot[3] / 1048576:>8.2f}")
    t = [f for f in walk(os.path.join(ROOT, "tests")) if os.path.basename(f).startswith("test_") and f.endswith(".py")]
    n = sum(open(f, encoding="utf-8", errors="ignore").read().count("\ndef test_") for f in t)
    print(f"тестових файлів {len(t)}, тестових функцій ≈{n}")


if __name__ == "__main__":
    main()
