"""Кільцеві плитки IRNM1 на диску (%LOCALAPPDATA%\\InkReel\\student, 2-8 МБ кожна): кожна нова висота кадру дає нову плитку, тож тека має межу,
а найдавніше використані плитки йдуть першими. Видалена плитка просто перерахується."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid import log  # noqa: E402
from benvid.student import inputs  # noqa: E402


def _fake(d, n, base=1_000_000_000):
    """n порожніх плиток із зростаючим часом змінення (перша - найдавніша)"""
    paths = []
    for i in range(n):
        p = os.path.join(d, f"ring_T1024_p{300 + i:04d}.npy")
        with open(p, "wb") as f:
            f.write(b"x")
        os.utime(p, (base + i, base + i))
        paths.append(p)
    return paths


def test_prune_keeps_the_newest_and_only_ring_files(tmp_path):
    paths = _fake(str(tmp_path), 30)
    other = tmp_path / "ring_T1024_p0999.npy.123.tmp.npy"        # чужа/тимчасова: не рахується й не видаляється
    other.write_bytes(b"x")
    note = tmp_path / "notes.txt"
    note.write_text("не плитка")
    inputs._prune_rings(str(tmp_path), keep=24)
    left = sorted(p for p in paths if os.path.exists(p))
    assert left == paths[6:], "лишаються 24 найсвіжіші"
    assert other.exists() and note.exists()


def test_prune_does_nothing_below_the_limit_and_survives_a_missing_folder(tmp_path):
    paths = _fake(str(tmp_path), 5)
    inputs._prune_rings(str(tmp_path), keep=24)
    assert all(os.path.exists(p) for p in paths)
    inputs._prune_rings(str(tmp_path / "нема_такої_теки"), keep=24)       # не падає


def test_ring_tile_writes_reads_and_stays_within_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(log, "app_dir", lambda: str(tmp_path))
    monkeypatch.setattr(inputs, "_rings", {})
    monkeypatch.setattr(inputs, "RING_KEEP", 3)
    d = tmp_path / "student"
    made = []
    for i, period in enumerate((3.0, 3.1, 3.2, 3.3, 3.4)):
        a = inputs.ring_tile(32, period)
        assert a.shape == (32, 32) and a.dtype == np.float16
        f = d / f"ring_T32_p{int(round(period * 100)):04d}.npy"
        assert f.exists()
        os.utime(f, (1_000_000_000 + i, 1_000_000_000 + i))    # детермінований порядок «давності»
        made.append(f)
        left = [p for p in d.iterdir() if p.name.startswith("ring_T")]
        assert len(left) <= 3
    assert [p.exists() for p in made] == [False, False, True, True, True]
    # та сама плитка вдруге читається з диска і збігається з перерахунком
    inputs._rings.clear()
    again = inputs.ring_tile(32, 3.4)
    assert np.array_equal(again, inputs.make_ring_tile(32, 3.4).astype(np.float16))


def test_reading_a_ring_marks_it_as_recently_used(tmp_path, monkeypatch):
    monkeypatch.setattr(log, "app_dir", lambda: str(tmp_path))
    monkeypatch.setattr(inputs, "_rings", {})
    inputs.ring_tile(32, 3.0)
    f = tmp_path / "student" / "ring_T32_p0300.npy"
    os.utime(f, (1_000_000_000, 1_000_000_000))
    inputs._rings.clear()
    before = time.time()
    inputs.ring_tile(32, 3.0)
    assert os.path.getmtime(f) >= before - 1, "після читання час змінення оновлюється"
