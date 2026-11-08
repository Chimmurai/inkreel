"""генератор датасету для дистиляції (tools/make_dataset.py): розміри, поділ за джерелом, відтворюваність мішеней, продовження"""
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import make_dataset as md  # noqa: E402

from benvid import core as C  # noqa: E402

needs_nb = pytest.mark.skipif(C._nb is None, reason="numba нема")


def _scene(w, h, t):
    yy, xx = np.mgrid[0:h, 0:w]
    b = (120 + 80 * np.sin((xx + t * 5) / 37.0) * np.cos(yy / 29.0)).clip(0, 255).astype(np.uint8)
    img = np.dstack([b, np.roll(b, 11, 1), 255 - b])
    cv2.circle(img, (int(w * .4 + t * 3) % w, h // 2), h // 5, (240, 230, 60), -1)
    return img


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("src")
    for name, (w, h, n) in {"GameA/a.mp4": (640, 360, 60), "GameB/b.mp4": (480, 270, 60), "Viber/v.mp4": (320, 180, 60)}.items():
        p = root / "videos" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for t in range(n):
            vw.write(_scene(w, h, t))
        vw.release()
    (root / "shots").mkdir()
    for i in range(4):
        cv2.imwrite(str(root / "shots" / f"s{i}.png"), _scene(640, 400, i * 17))
    return str(root / "videos"), str(root / "shots")


def _args(out, sources, count, workers=0, **kw):
    v, s = sources
    a = ["--out", str(out), "--videos", v, "--shots", s, "--count", str(count), "--workers", str(workers), "--hhi", "240"]
    for k, val in kw.items():
        a += [f"--{k.replace('_', '-')}", str(val)]
    return a


def _manifest(out):
    return [json.loads(x) for x in open(os.path.join(out, "manifest.jsonl"), encoding="utf-8")]


def test_pick_size_stays_in_range_and_even():
    rng = np.random.default_rng(0)
    c = dict(hlo=48, hhi=2160, max_side=4096)
    hs = []
    for _ in range(3000):
        w, h = md.pick_size(rng, 1920, 1080, c)
        assert w % 2 == 0 and h % 2 == 0
        assert max(w, h) <= 4096 and min(w, h) >= md.MIN_SIDE
        hs.append(h)
    hs = np.array(hs)
    assert hs.min() <= 80 and hs.max() >= 2000                    # від дуже малих до 4K
    assert 0.35 < np.mean(hs < 360) < 0.7                         # log-розподіл: багато малих


def test_pick_crop_has_target_aspect_and_fits():
    rng = np.random.default_rng(1)
    for _ in range(500):
        w, h = int(rng.integers(40, 3000)), int(rng.integers(40, 2000))
        x, y, cw, ch = md.pick_crop(rng, 1920, 1080, w, h)
        assert 0 <= x and 0 <= y and x + cw <= 1920 and y + ch <= 1080
        assert abs(cw / ch - w / h) < 0.08 * max(1, w / h, h / w) + 0.1


def test_split_is_stable_per_source():
    assert md.split_of("GameA/a.mp4") == md.split_of("GameA/a.mp4")
    ks = [md.split_of(f"v{i}") for i in range(2000)]
    assert 0.86 < ks.count("train") / len(ks) < 0.94 and "val" in ks and "test" in ks


def test_index_skips_excluded_folders(tmp_path, sources):
    idx = md.build_index(sources[0], sources[1], md.EXCLUDE, log=lambda *a, **k: None)
    tops = {v["top"] for v in idx["videos"]}
    assert tops == {"GameA", "GameB"} and len(idx["shots"]) == 4
    w = md.source_weights(idx)
    assert abs(w.sum() - 1) < 1e-9 and (w > 0).all()


@needs_nb
def test_end_to_end_pairs_masks_and_reproducibility(tmp_path, sources):
    out = tmp_path / "ds"
    assert md.main(_args(out, sources, 10)) == 0
    recs = _manifest(out)
    assert len(recs) >= 10 and len({r["id"] for r in recs}) == len(recs)
    from PIL import Image
    for r in recs[:6]:
        pin, pk = os.path.join(out, r["inp"]), os.path.join(out, r["ink"])
        assert os.path.getsize(pin) == r["in_bytes"] and os.path.getsize(pk) == r["ink_bytes"]
        m = Image.open(pk)
        assert m.mode == "1" and m.size == (r["w"], r["h"])
        assert r["h"] <= 240 and r["split"] in ("train", "val", "test")
    assert {r["kind"] for r in recs} == {"video", "shot"} or len(recs) < 20
    assert all("viber" not in r["src"].lower() for r in recs)
    assert md.main(["check", "--out", str(out), "--n", "8"]) == 0       # мішень із файлу = мішень алгоритму, 0 відмінних пікселів


@needs_nb
def test_resume_adds_without_duplicates_and_keeps_old(tmp_path, sources):
    out = tmp_path / "ds"
    assert md.main(_args(out, sources, 12)) == 0
    first = _manifest(out)
    assert md.main(_args(out, sources, len(first) + 14)) == 0
    both = _manifest(out)
    ids = [r["id"] for r in both]
    assert len(ids) == len(set(ids)) and len(both) > len(first)
    assert [r["id"] for r in both[:len(first)]] == [r["id"] for r in first]
    n = len(both)
    assert md.main(_args(out, sources, n)) == 0                          # ціль уже досягнуто - нічого не додає
    assert len(_manifest(out)) == n


@needs_nb
def test_size_limit_stops_and_mixed_settings_are_refused(tmp_path, sources):
    out = tmp_path / "ds"
    assert md.main(_args(out, sources, 10 ** 6, max_gb=0.0001)) == 0
    n = len(_manifest(out))
    assert 0 < n < 200                                                   # зупинились за обсягом, а не за кількістю
    assert md.main(_args(out, sources, 10 ** 6, seed=5)) == 2            # інший seed у тій самій теці - відмова
    assert len(_manifest(out)) == n


@needs_nb
def test_pool_mode_works_like_inprocess(tmp_path, sources):
    out = tmp_path / "ds"
    assert md.main(_args(out, sources, 14, workers=2)) == 0
    recs = _manifest(out)
    assert len(recs) >= 14 and len({r["id"] for r in recs}) == len(recs)
    assert md.main(["check", "--out", str(out), "--n", "6"]) == 0


def _av1_encoder():
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if not ff:
        return None, None
    enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    for name in ("libsvtav1", "libaom-av1", "librav1e"):
        if name in enc:
            return ff, name
    return None, None


@needs_nb
def test_av1_video_is_read_through_ffmpeg_not_opencv(tmp_path, sources, monkeypatch):
    """AV1 (ShadowPlay на RTX 40): OpenCV після перемотки без заголовка послідовності засипає лог помилками - читаємо ffmpeg-ом"""
    import subprocess
    ff, enc = _av1_encoder()
    if not ff:
        pytest.skip("немає ffmpeg з AV1-кодувальником")
    root = tmp_path / "src"
    (root / "videos" / "Game").mkdir(parents=True)
    (root / "shots").mkdir()
    raw = str(root / "raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for t in range(90):
        vw.write(_scene(640, 360, t))
    vw.release()
    dst = str(root / "videos" / "Game" / "a.mp4")
    r = subprocess.run([ff, "-v", "error", "-y", "-i", raw, "-c:v", enc, "-g", "30", "-pix_fmt", "yuv420p", dst], capture_output=True)
    if r.returncode:
        pytest.skip("AV1-кодувальник не спрацював")
    import benvid.video as V
    monkeypatch.setattr(V, "_open_cv", lambda p: (_ for _ in ()).throw(AssertionError("OpenCV не мав чіпати AV1")))
    out = tmp_path / "ds"
    assert md.main(["--out", str(out), "--videos", str(root / "videos"), "--shots", str(root / "shots"), "--count", "8",
                    "--workers", "0", "--hhi", "200"]) == 0
    idx = json.load(open(out / "index.json", encoding="utf-8"))
    assert [v["codec"] for v in idx["videos"]] == ["av1"] and idx["videos"][0]["n"] in (89, 90, 91)
    recs = _manifest(out)
    assert len(recs) >= 8
    assert len({r["frame"] for r in recs}) >= 4                          # кадри з різних місць відео, а не один
    assert os.path.getsize(out / "errors.log") == 0
