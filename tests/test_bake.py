"""власний формат .ikr (benvid.bake): запис/читання, довільний доступ, обірвані файли, експорт відео в .ikr і з .ikr у інші формати"""
import json
import os
import subprocess
import threading

import cv2
import numpy as np
import pytest

from benvid import bake
from benvid.core import Params
from benvid.video import export_video, find_ffmpeg, packed_video, video_info


def _masks(n=70, h=37, w=53, seed=1):
    rng = np.random.default_rng(seed)
    cur = (rng.random((h, w)) > 0.5).astype(np.uint8)
    out = []
    for _ in range(n):
        cur = cur.copy()
        cur[rng.integers(0, h, 20), rng.integers(0, w, 20)] ^= 1
        out.append(cur)
    return out


def _write(path, ms, **kw):
    h, w = ms[0].shape
    wr = bake.BakeWriter(path, w, h, 24000, 1001, kw.get("meta"), key=kw.get("key", bake.KEY))
    for m in ms:
        wr.add_mask(m)
    return wr.close()


def _clip(path, n=14, w=320, h=180, audio=False):
    raw = str(path) + ".raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[:] = (60 + (xx * 140 // w))[..., None]
        cv2.circle(f, (40 + i * 12, h // 2), 40, (235, 235, 235), -1)
        cv2.putText(f, "Ab", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (10, 10, 10), 3)
        vw.write(f)
    vw.release()
    if not audio:
        os.replace(raw, path)
        return
    ff = find_ffmpeg()
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", raw, "-f", "lavfi", "-i", "sine=frequency=440:duration=%.3f" % (n / 24), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(path)], check=True)
    os.remove(raw)


def test_roundtrip_and_random_access(tmp_path):
    ms = _masks()
    p = str(tmp_path / "a.ikr")
    r = _write(p, ms, meta=dict(params={"method": 5}))
    assert r["frames"] == len(ms) and os.path.getsize(p) == r["bytes"]
    assert bake.is_baked(p) and not bake.is_baked(str(tmp_path / "nope.ikr"))
    rd = bake.open_baked(p)
    assert (rd.w, rd.h, rd.frames) == (53, 37, 70) and rd.fps_frac == (24000, 1001)
    assert rd.meta["params"] == {"method": 5} and rd.info()["duration"] == pytest.approx(70 * 1001 / 24000)
    for i in [0, 1, 29, 30, 31, 59, 69, 5, 69, 3, 60, 61, 29, 0]:          # вперед, назад, через ключові кадри, повтор
        assert np.array_equal(rd.mask(i), ms[i]), i
    pk = rd.packed(7)
    pk[:] = 0                                                              # копія: кеш читача не псується
    assert np.array_equal(rd.mask(7), ms[7])
    with pytest.raises(IndexError):
        rd.packed(70)
    assert np.array_equal(rd.clone().mask(44), ms[44])


def test_key_interval_one_and_odd_sizes(tmp_path):
    ms = _masks(9, 5, 11, seed=4)
    p = str(tmp_path / "k1.ikr")
    _write(p, ms, key=1)
    rd = bake.open_baked(p)
    assert all(np.array_equal(rd.mask(i), ms[i]) for i in range(9))


def test_threads_read_concurrently(tmp_path):
    ms = _masks(90)
    p = str(tmp_path / "t.ikr")
    _write(p, ms)
    base = bake.open_baked(p)
    bad = []

    def work(seed):
        rd = base.clone()
        rng = np.random.default_rng(seed)
        for i in rng.integers(0, 90, 60):
            if not np.array_equal(rd.mask(int(i)), ms[int(i)]):
                bad.append(int(i))
    ts = [threading.Thread(target=work, args=(s,)) for s in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not bad


def test_broken_files_are_rejected(tmp_path):
    p = str(tmp_path / "b.ikr")
    _write(p, _masks(40))
    raw = open(p, "rb").read()
    cases = {"обрізаний": raw[:-25], "без кінця": raw[: len(raw) // 2], "сміття": b"hello world" * 50, "порожній": b"",
             "чужий підпис": b"X" * len(bake.MAGIC) + raw[len(bake.MAGIC):]}
    for name, data in cases.items():
        q = str(tmp_path / "c.ikr")
        open(q, "wb").write(data)
        with pytest.raises(bake.BakeError):
            bake.open_baked(q)
    bad = bytearray(raw)                                                   # зіпсований кадр: помилка при читанні, а не мовчки сміття
    rd = bake.open_baked(p)
    off = int(rd.index["off"][3]) + 2
    bad[off] ^= 0xFF
    bad[off + 1] ^= 0xFF
    q = str(tmp_path / "d.ikr")
    open(q, "wb").write(bytes(bad))
    rd2 = bake.open_baked(q)
    with pytest.raises(bake.BakeError):
        rd2.packed(3)


def test_abort_leaves_nothing(tmp_path):
    p = str(tmp_path / "x.ikr")
    wr = bake.BakeWriter(p, 16, 8, 24, 1)
    wr.add_mask(np.ones((8, 16), np.uint8))
    assert os.path.exists(p + ".part") and not os.path.exists(p)
    wr.abort()
    assert not os.path.exists(p + ".part") and not os.path.exists(p)


def test_wrong_frame_shape_is_refused(tmp_path):
    wr = bake.BakeWriter(str(tmp_path / "s.ikr"), 16, 8, 24, 1)
    with pytest.raises(ValueError):
        wr.add_packed(np.zeros((8, 3), np.uint8))
    wr.abort()


def test_picked_frames_matches_fps_rule():
    assert bake.picked_frames(10, 24.0, None) == list(range(10))
    assert bake.picked_frames(10, 24.0, None, 3, 7) == [3, 4, 5, 6]
    assert bake.picked_frames(10, 24.0, 24) == list(range(10))
    f = bake.picked_frames(48, 24.0, 12)
    assert len(f) == 24 and f[:4] == [0, 2, 4, 6] and f[-1] == 46
    g = bake.picked_frames(30, 30.0, 25, 0, 30)
    assert len(g) == 25 and g[0] == 0 and g[-1] <= 29


@pytest.mark.skipif(find_ffmpeg() is None, reason="потрібен ffmpeg")
def test_video_to_ikr_bits_equal_direct_render(tmp_path):
    src = str(tmp_path / "v.mp4")
    _clip(src)
    pr = Params(method=3)
    size = (320, 180)
    ref = np.concatenate(list(packed_video(src, pr, size, workers=1)))
    out = str(tmp_path / "v.ikr")
    r = export_video(src, out, pr, "InkReel (.ikr, без втрат)", "Як у джерела", "Висока", workers=1)
    assert r["frames"] == len(ref) and r["size"] == size and os.path.isfile(out)
    rd = bake.open_baked(out)
    assert rd.frames == len(ref) and (rd.w, rd.h) == size
    assert all(np.array_equal(rd.packed(i), ref[i]) for i in range(len(ref)))
    m = rd.meta
    assert m["params"]["method"] == 3 and m["src"]["name"] == "v.mp4" and m["src"]["frames"] == len(ref) and m["range"] == [0, len(ref)]
    assert not os.path.exists(out + ".part")


@pytest.mark.skipif(find_ffmpeg() is None, reason="потрібен ffmpeg")
def test_ikr_to_other_formats_and_range(tmp_path):
    src = str(tmp_path / "v.mp4")
    _clip(src, n=24)
    ikr = str(tmp_path / "v.ikr")
    export_video(src, ikr, Params(method=3), "InkReel (.ikr, без втрат)", "Як у джерела", "Висока", workers=1)
    rd = bake.open_baked(ikr)
    # PNG-послідовність: біти збігаються, діапазон і ліміт кадрів працюють
    d = str(tmp_path / "png")
    r = export_video(ikr, d, Params(), "PNG-послідовність", "720p", "Висока", workers=1, start=4, end=16, fps=12)
    idx = bake.picked_frames(24, 24.0, 12, 4, 16)
    files = sorted(os.listdir(d))
    assert r["frames"] == len(idx) == len(files)
    for f, i in zip(files, idx):
        im = cv2.imdecode(np.fromfile(os.path.join(d, f), np.uint8), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(im > 127, rd.mask(i) > 0)
    # MP4 і GIF із файлу .ikr (розмір - його власний, size_key ігнорується)
    r = export_video(ikr, str(tmp_path / "o.mp4"), Params(), "MP4 (H.264)", "144p", "Мала", workers=1)
    assert r["frames"] == 24 and r["size"] == (320, 180)
    info = video_info(str(tmp_path / "o.mp4"))
    assert (info["width"], info["height"]) == (320, 180) and info["frames"] == 24
    r = export_video(ikr, str(tmp_path / "o.gif"), Params(), "GIF", "144p", "Мала", workers=1, start=0, end=6)
    assert r["frames"] == 6
    # з .ikr у .ikr (вирізаний шматок): шлях до оригіналу й зсув діапазону зберігаються
    clip = str(tmp_path / "clip.ikr")
    r = export_video(ikr, clip, Params(), "InkReel (.ikr, без втрат)", "720p", "Висока", workers=1, start=5, end=11)
    rc = bake.open_baked(clip)
    assert rc.frames == 6 and all(np.array_equal(rc.packed(i), rd.packed(5 + i)) for i in range(6))
    assert rc.meta["src"]["name"] == "v.mp4" and rc.meta["range"] == [5, 11] and rc.meta["params"]["method"] == 3
    again = str(tmp_path / "again.ikr")
    export_video(clip, again, Params(), "InkReel (.ikr, без втрат)", "720p", "Висока", workers=1, start=2, end=4)
    ra = bake.open_baked(again)
    assert ra.meta["range"] == [7, 9] and ra.frames == 2 and np.array_equal(ra.packed(0), rd.packed(7))


@pytest.mark.skipif(find_ffmpeg() is None, reason="потрібен ffmpeg")
def test_audio_travels_with_ikr(tmp_path):
    src = str(tmp_path / "av.mp4")
    _clip(src, n=24, audio=True)
    ikr = str(tmp_path / "av.ikr")
    export_video(src, ikr, Params(method=3), "InkReel (.ikr, без втрат)", "Як у джерела", "Висока", workers=1, audio=True)
    rd = bake.open_baked(ikr)
    a = rd.audio()
    assert a is not None and len(a[0]) > 500 and a[1] == ".m4a"
    out = str(tmp_path / "av2.mp4")
    export_video(ikr, out, Params(), "MP4 (H.264)", "720p", "Мала", workers=1, audio=True)
    pr = subprocess.run([find_ffmpeg(), "-i", out], capture_output=True, text=True).stderr
    assert "Audio:" in pr
    noaud = str(tmp_path / "na.ikr")
    export_video(src, noaud, Params(method=3), "InkReel (.ikr, без втрат)", "Як у джерела", "Висока", workers=1, audio=False)
    assert bake.open_baked(noaud).audio() is None


def test_baked_source_for_player(tmp_path):
    ms = _masks(33, 20, 30)
    p = str(tmp_path / "s.ikr")
    _write(p, ms)
    s = bake.BakedSource(p)
    assert s.baked and s.info["kind"] == "video" and s.info["frames"] == 33 and s.info["locked"]
    assert np.array_equal(s.read(31), ms[31]) and s.packed(40) is None and s.read(-1) is None
    c = s.clone()
    assert np.array_equal(c.read(2), ms[2])
    s.close()
    c.close()


# ---------------------------------------------------------------- кодек ctx (версія 2) і сумісність зі старими файлами
def _scene(n=40, h=48, w=75, seed=3):
    """рухомі фігури на зернистому тлі, ширина не кратна 8: схоже на справжні кадри (є рух, є стале зерно)"""
    rng = np.random.default_rng(seed)
    grain = (rng.random((h, w)) > 0.55).astype(np.uint8)
    out = []
    for i in range(n):
        f = grain.copy()
        f[10:30, 5 + i:25 + i] = 1
        f[35:40, :] = 0
        f[rng.integers(0, h, 6), rng.integers(0, w, 6)] ^= 1
        out.append(f)
    return out


def test_new_files_use_the_ctx_codec_and_are_much_smaller_than_zlib(tmp_path):
    ms = _scene()
    h, w = ms[0].shape
    sizes = {}
    for codec in ("ctx", "zlib"):
        p = str(tmp_path / f"{codec}.ikr")
        wr = bake.BakeWriter(p, w, h, 24, 1, None, codec=codec)
        for m in ms:
            wr.add_mask(m)
        wr.close()
        sizes[codec] = os.path.getsize(p)
        rd = bake.open_baked(p)
        assert rd.codec == codec and rd.head["version"] == (2 if codec == "ctx" else 1)
        for i in (0, 1, 29, 30, 31, len(ms) - 1, 5, 0):                 # довільний доступ, у т. ч. назад і через межу проміжку
            assert np.array_equal(rd.mask(i), ms[i]), (codec, i)
            assert np.array_equal(rd.packed(i), np.packbits(ms[i], axis=-1)), (codec, i)
        rd.close()
    assert sizes["ctx"] < 0.8 * sizes["zlib"], sizes
    p = str(tmp_path / "auto.ikr")
    _write(p, ms)
    assert bake.open_baked(p).codec == "ctx", "типовий кодек - ctx"


def test_ctx_is_deterministic_and_reads_sequentially_and_by_clone(tmp_path, monkeypatch):
    monkeypatch.setattr(bake.time, "strftime", lambda *a, **k: "2026-01-01T00:00:00")
    ms = _scene(n=35)
    a, b = str(tmp_path / "a.ikr"), str(tmp_path / "b.ikr")
    _write(a, ms)
    _write(b, ms)
    assert open(a, "rb").read() == open(b, "rb").read(), "той самий вхід - ті самі байти"
    rd = bake.open_baked(a)
    for i in range(len(ms)):                                            # послідовно (кеш продовжує проміжок)
        assert np.array_equal(rd.mask(i), ms[i])
    c = rd.clone()
    assert np.array_equal(c.mask(33), ms[33]) and np.array_equal(rd.mask(2), ms[2]) and np.array_equal(c.mask(34), ms[34])


@pytest.mark.parametrize("w,h,key", ((2, 2, 30), (8, 8, 1), (9, 5, 3), (1, 7, 30), (200, 3, 4)))
def test_ctx_tiny_and_odd_sizes(tmp_path, w, h, key):
    rng = np.random.default_rng(w * 31 + h)
    ms = [(rng.random((h, w)) > 0.5).astype(np.uint8) for _ in range(9)]
    p = str(tmp_path / "t.ikr")
    wr = bake.BakeWriter(p, w, h, 10, 1, None, key=key, codec="ctx")
    for m in ms:
        wr.add_mask(m)
    wr.close()
    rd = bake.open_baked(p)
    for i in (8, 0, 4, 7, 3):
        assert np.array_equal(rd.mask(i), ms[i]), (w, h, key, i)


def test_ctx_all_black_all_white_and_noise_roundtrip(tmp_path):
    h, w = 40, 64
    rng = np.random.default_rng(0)
    ms = [np.zeros((h, w), np.uint8), np.ones((h, w), np.uint8), (rng.random((h, w)) > 0.5).astype(np.uint8), np.ones((h, w), np.uint8),
          np.zeros((h, w), np.uint8)] * 3
    p = str(tmp_path / "x.ikr")
    _write(p, ms)
    rd = bake.open_baked(p)
    assert all(np.array_equal(rd.mask(i), ms[i]) for i in range(len(ms)))


def test_ctx_damage_is_reported_not_silent(tmp_path):
    ms = _scene(n=12)
    p = str(tmp_path / "d.ikr")
    _write(p, ms)
    rd = bake.open_baked(p)
    off, ln = int(rd.index["off"][3]), int(rd.index["len"][3])
    rd.close()
    raw = bytearray(open(p, "rb").read())
    raw[off + ln // 2] ^= 0x55
    open(p, "wb").write(bytes(raw))
    rd = bake.open_baked(p)
    assert np.array_equal(rd.mask(2), ms[2])
    with pytest.raises(bake.BakeError):
        rd.mask(3)
    with pytest.raises(bake.BakeError):
        rd.mask(5)                                                      # далі по ланцюжку теж не мовчить
    assert np.array_equal(rd.mask(0), ms[0]), "після помилки читач лишається робочим"


def test_files_written_by_the_old_zlib_only_writer_still_open(tmp_path):
    """файл версії 1 без поля codec (так писала попередня версія): вручну збираємо той самий байтовий формат"""
    import struct
    import zlib
    ms = _masks(n=40, h=21, w=37)
    h, w = ms[0].shape
    p = str(tmp_path / "old.ikr")
    head = json.dumps(dict(format=bake.FORMAT, version=1, width=w, height=h, fps=[24, 1], key=30, created="2025-01-01T00:00:00"),
                      separators=(",", ":")).encode()
    with open(p, "wb") as f:
        f.write(bake.MAGIC + struct.pack("<I", len(head)) + head)
        idx, prev = [], None
        for i, m in enumerate(ms):
            pk = np.packbits(m, axis=-1)
            d = pk if i % 30 == 0 else np.bitwise_xor(pk, prev)
            z = zlib.compress(d.tobytes(), 4)
            idx.append((f.tell(), len(z)))
            f.write(z)
            prev = pk
        io = f.tell()
        arr = np.zeros(len(idx), bake._REC)
        arr["off"], arr["len"] = [o for o, _ in idx], [n for _, n in idx]
        f.write(arr.tobytes())
        tail = json.dumps(dict(frames=len(idx), index=[io, arr.nbytes], audio=None), separators=(",", ":")).encode()
        to = f.tell()
        f.write(tail)
        f.write(struct.pack("<4sQQ", bake.TRAILER, to, len(tail)))
    rd = bake.open_baked(p)
    assert rd.codec == "zlib" and rd.frames == len(ms)
    assert all(np.array_equal(rd.mask(i), ms[i]) for i in (0, 29, 30, 39, 12))


def test_unknown_codec_and_newer_version_are_refused_politely(tmp_path):
    import struct
    p = str(tmp_path / "n.ikr")
    head = json.dumps(dict(format=bake.FORMAT, version=2, codec="future", width=4, height=4, fps=[1, 1]), separators=(",", ":")).encode()
    open(p, "wb").write(bake.MAGIC + struct.pack("<I", len(head)) + head + b"\x00" * 64)
    with pytest.raises(bake.BakeError):
        bake.open_baked(p)


def test_without_numba_new_files_fall_back_to_zlib_and_ctx_files_are_refused(tmp_path, monkeypatch):
    ms = _scene(n=8)
    pc = str(tmp_path / "c.ikr")
    _write(pc, ms)                                                      # з numba - ctx
    monkeypatch.setattr(bake.ikrctx, "available", lambda: False)
    pz = str(tmp_path / "z.ikr")
    _write(pz, ms)
    rd = bake.open_baked(pz)
    assert rd.codec == "zlib" and np.array_equal(rd.mask(5), ms[5]), "нема numba - файл пишеться zlib і читається"
    with pytest.raises(bake.BakeError):
        bake.open_baked(pc)                                             # ctx без numba - зрозуміла відмова, а не збій
    with pytest.raises(ValueError):
        bake.BakeWriter(str(tmp_path / "x.ikr"), 4, 4, 1, 1, None, codec="ctx")
