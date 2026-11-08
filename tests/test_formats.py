"""Сумісність форматів: фото (EXIF-поворот, 16 біт, float, прозорість, незвичні формати), анімації (APNG/WebP/GIF), відео
(кодеки, поворот телефона, неквадратні пікселі, перемотка в .ts, хибний лічильник кадрів, зрозумілі помилки)."""
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid import media  # noqa: E402
from benvid.video import find_ffmpeg, open_capture, probe_stream, video_info  # noqa: E402

W, H = 240, 160
FF = find_ffmpeg()


def _base():
    """градієнт, біла «стрілка» в лівому верхньому куті, червоний квадрат праворуч-внизу: орієнтацію видно по ним"""
    a = np.zeros((H, W, 3), np.uint8)
    a[:] = np.linspace(20, 235, W, dtype=np.uint8)[None, :, None]
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    d.polygon([(10, 10), (70, 10), (10, 70)], fill=(255, 255, 255))
    d.rectangle([W - 60, H - 50, W - 20, H - 15], fill=(220, 30, 30))
    return im


def _where_red(img):
    b, g, r = [img[..., i].astype(int) for i in (0, 1, 2)]
    ys, xs = np.nonzero((r > 150) & (g < 90) & (b < 90))
    return xs.mean() / img.shape[1], ys.mean() / img.shape[0]


@pytest.fixture(scope="module")
def d(tmp_path_factory):
    return tmp_path_factory.mktemp("fmt")


def test_exif_orientation(d):
    base = _base()
    exp = {1: ((W, H), (0.83, 0.80)), 3: ((W, H), (0.17, 0.20)), 6: ((H, W), (0.20, 0.83)), 8: ((H, W), (0.80, 0.17))}
    for o, (dims, red) in exp.items():
        p = str(d / f"exif{o}.jpg")
        ex = Image.Exif()
        ex[0x0112] = o
        base.save(p, exif=ex, quality=95)
        img = media.imread_u(p)
        assert (img.shape[1], img.shape[0]) == dims, o
        x, y = _where_red(img)
        assert abs(x - red[0]) < 0.05 and abs(y - red[1]) < 0.05, (o, x, y)
    # справжнє фото відкривається вже повернутим і для Player-інформації
    ex = Image.Exif()
    ex[0x0112] = 6
    p = str(d / "portrait.jpg")
    base.save(p, exif=ex)
    src = media.open_source(p)
    assert (src.info["width"], src.info["height"]) == (H, W)


def test_png_exif_orientation(d):
    p = str(d / "o6.png")
    ex = Image.Exif()
    ex[0x0112] = 6
    _base().save(p, exif=ex)
    img = media.imread_u(p)
    assert (img.shape[1], img.shape[0]) == (H, W)


def test_bit_depths_alpha_float(d):
    gray = np.asarray(_base().convert("L"))
    # 16 біт: яскравість зберігається (а не обрізається до білого)
    Image.fromarray(gray.astype(np.uint16) * 257).save(str(d / "g16.png"))
    a = media.imread_u(str(d / "g16.png")).mean()
    assert abs(a - gray.mean()) < 2
    # дуже темний 16-бітний - лишається темним
    Image.fromarray((gray.astype(np.uint16) * 200 // 255)).save(str(d / "g16d.png"))
    assert media.imread_u(str(d / "g16d.png")).mean() < 2
    # float TIFF 0..1 і 0..4: не чорний
    f = gray.astype(np.float32) / 255
    Image.fromarray(f, mode="F").save(str(d / "f1.tif"))
    Image.fromarray(f * 4, mode="F").save(str(d / "f4.tif"))
    for n in ("f1.tif", "f4.tif"):
        assert abs(media.imread_u(str(d / n)).mean() - gray.mean()) < 6, n
    # прозорість лягає на білий
    rgba = _base().convert("RGBA")
    al = np.full((H, W), 255, np.uint8)
    al[:, W // 2:] = 0
    rgba.putalpha(Image.fromarray(al))
    rgba.save(str(d / "a.png"))
    img = media.imread_u(str(d / "a.png"))
    assert img[:, W // 2 + 2:].min() >= 250
    la = _base().convert("L").convert("LA")
    la.putalpha(Image.fromarray(al))
    la.save(str(d / "la.png"))
    assert media.imread_u(str(d / "la.png"))[:, W // 2 + 2:].min() >= 250


def test_cmyk_and_palette_and_misc_formats(d):
    base = _base()
    cases = {"cmyk.jpg": base.convert("CMYK"), "pal.png": base.convert("P", palette=Image.ADAPTIVE, colors=32),
             "bw.bmp": base.convert("L").convert("1"), "x.ppm": base, "x.pgm": base.convert("L"), "x.tga": base,
             "x.ico": base.resize((64, 64)), "x.jp2": base, "x.webp": base, "x.tif": base, "x.bmp": base}
    for n, im in cases.items():
        p = str(d / n)
        im.save(p)
        assert media.kind_of(p) == "image", n
        img = media.imread_u(p)
        assert img.ndim == 3 and img.shape[2] == 3 and img.shape[0] > 10, n
        assert abs(img.mean() - np.asarray(im.convert("L")).mean()) < 12 or n.startswith("bw"), n


def test_kind_by_content_not_extension(d):
    base = _base()
    p = str(d / "real.png")
    base.save(p)
    fake = {"fake.jpg": p, "noext": p, "pic.mp4": p}
    import shutil
    for n, src in fake.items():
        shutil.copy(src, str(d / n))
        assert media.kind_of(str(d / n)) == "image", n
        assert media.imread_u(str(d / n)).shape[:2] == (H, W)
    # неіснуючий або невпізнаний файл - за розширенням
    assert media.kind_of(str(d / "nope.mp4")) == "video"
    assert media.kind_of(str(d / "nope.heic")) == "image"
    assert media.kind_of("x.gif") == "gif"


def test_animations_apng_webp_gif(d):
    base = np.asarray(_base())
    frames = [Image.fromarray(np.roll(base, i * 20, axis=1)) for i in range(5)]
    for n, kw in (("a.png", {}), ("a.webp", {"lossless": True}), ("a.gif", {})):
        fr = frames if n != "a.gif" else [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
        fr[0].save(str(d / n), save_all=True, append_images=fr[1:], duration=100, loop=0, **kw)
        assert media.kind_of(str(d / n)) == "gif", n
        s = media.open_source(str(d / n))
        assert s.info["frames"] == 5 and s.info["kind"] == "gif" and s.info["width"] == W, n
        assert s.read(4) is not None and s.read(5) is None
    # статичний PNG/WebP - просто фото
    _base().save(str(d / "s.png"))
    _base().save(str(d / "s.webp"))
    assert media.kind_of(str(d / "s.png")) == "image" and media.kind_of(str(d / "s.webp")) == "image"


def test_image_errors_are_readable(d):
    (d / "empty.png").write_bytes(b"")
    (d / "junk.png").write_bytes(os.urandom(300))
    with pytest.raises(ValueError, match="порожній"):
        media.imread_u(str(d / "empty.png"))
    with pytest.raises(ValueError, match="не вдалося прочитати"):
        media.imread_u(str(d / "junk.png"))
    # обрізаний JPEG відновлюється із попередженням
    p = str(d / "full.jpg")
    _base().save(p, quality=90)
    b = open(p, "rb").read()
    (d / "cut.jpg").write_bytes(b[: len(b) * 2 // 3])
    notes = []
    img = media.imread_u(str(d / "cut.jpg"), notes.append)
    assert img.shape[:2] == (H, W) and notes


def test_still_export_formats(d):
    from benvid.core import Params
    src = str(d / "src.png")
    _base().save(src)
    for fmt, ext in media.STILL_FORMATS.items():
        out = str(d / ("o" + ext))
        r = media.export_still(src, out, Params(), "Як у джерела")
        assert os.path.getsize(out) > 0 and r["size"] == (W, H), fmt
        im = Image.open(out)
        assert im.size == (W, H)
        a = np.asarray(im.convert("L"))
        assert set(np.unique(a)) <= {0, 255}, fmt           # лише чорне й біле


# ---------------------------------------------------------------- відео
def _make(path, vargs, n=24, w=160, h=96, fps=30, extra=()):
    cmd = [FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           *extra, *vargs, path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(n):
        a = np.zeros((h, w, 3), np.uint8)
        a[:] = np.linspace(20, 235, w, dtype=np.uint8)[None, :, None]
        im = Image.fromarray(a)
        dr = ImageDraw.Draw(im)
        dr.polygon([(4, 4), (40, 4), (4, 40)], fill=(255, 255, 255))
        x = 10 + (i * 4) % (w - 40)
        dr.rectangle([x, h - 36, x + 24, h - 12], fill=(220, 30, 30))
        try:
            p.stdin.write(np.asarray(im).tobytes())
        except OSError:                                      # ffmpeg одразу вийшов (немає кодека, напр. libsvtav1 у збірці для Windows): це «нема кодека», а не помилка
            break
    try:
        p.stdin.close()
    except OSError:
        pass
    err = p.stderr.read().decode()
    return p.wait() == 0, err


needs_ff = pytest.mark.skipif(not FF, reason="потрібен ffmpeg")


@needs_ff
def test_video_frame_count_seek_and_rotation(d):
    X264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cases = {"a.mp4": X264, "a.avi": ["-c:v", "mpeg4", "-q:v", "3"], "a.mpg": ["-c:v", "mpeg2video", "-q:v", "3"],
             "a.ts": ["-c:v", "mpeg2video", "-q:v", "3"], "a.mkv": ["-c:v", "ffv1"], "a.webm": ["-c:v", "libvpx-vp9", "-crf", "35", "-b:v", "0"]}
    for n, v in cases.items():
        p = str(d / n)
        ok, err = _make(p, v)
        if not ok:
            continue                                         # цей кодек не зібрано в ffmpeg - пропускаємо
        info = video_info(p)
        assert info["frames"] == 24 and (info["width"], info["height"]) == (160, 96), (n, info)
        src = media.open_source(p)
        first = src.read(0)
        src.read(20)
        again = src.read(0)                                  # назад - після стрибка вперед
        assert first is not None and again is not None
        assert np.abs(first.astype(int) - again.astype(int)).mean() < 4, n   # той самий кадр (у .ts OpenCV промахується)
        assert src.read(23) is not None and src.read(24) is None, n         # останній кадр є, зайвого нема
        src.close()


@needs_ff
def test_video_phone_rotation_and_sar(d):
    X264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    base = str(d / "base.mp4")
    ok, _ = _make(base, X264)
    assert ok
    rot = str(d / "rot90.mp4")
    r = subprocess.run([FF, "-y", "-loglevel", "error", "-display_rotation", "90", "-i", base, "-c", "copy", rot])
    if r.returncode == 0 and probe_stream(rot).get("rotation"):
        info = video_info(rot)
        assert (info["width"], info["height"]) == (96, 160)            # портрет
        src = media.open_source(rot)
        f = src.read(3)
        assert f.shape[:2] == (160, 96)
        src.close()
    sar = str(d / "sar.mp4")
    ok, _ = _make(sar, X264, extra=["-vf", "setsar=2/1"])
    if ok:
        info = video_info(sar)
        assert (info["width"], info["height"]) == (320, 96)            # пропорції з урахуванням формату пікселя
        src = media.open_source(sar)
        assert src.read(0).shape[:2] == (96, 160)                      # кадр лишається як закодований; масштабує binarize
        src.close()


@needs_ff
def test_av1_and_interlaced_fall_back_to_ffmpeg(d):
    p = str(d / "av1.mp4")
    ok, _ = _make(p, ["-c:v", "libsvtav1", "-crf", "40", "-pix_fmt", "yuv420p"], n=12)
    if ok:
        src = media.open_source(p)
        assert src.read(0) is not None and src.read(11) is not None    # навіть якщо OpenCV AV1 не знає
        src.close()
    q = str(d / "il.mp4")
    ok, _ = _make(q, ["-c:v", "libx264", "-pix_fmt", "yuv420p"], extra=["-flags", "+ilme+ildct"])
    if ok:
        cap = open_capture(q)
        ok1, f = cap.read()
        cap.release()
        assert ok1 and f.shape[:2] == (96, 160)
        x, y = _where_red(f)
        assert 0.05 < x < 0.4 and 0.6 < y < 0.9                       # картинка не зіпсована (без цього - «розсипана»)


@needs_ff
def test_video_open_errors_are_readable(d):
    (d / "empty.mkv").write_bytes(b"")
    (d / "junk.mp4").write_bytes(os.urandom(3000))
    from benvid.video import VIDEO_EXT
    for name in ("empty.mkv", "junk.mp4"):
        with pytest.raises(ValueError) as e:
            video_info(str(d / name))
        assert name in str(e.value)
    ok, _ = subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=duration=1", str(d / "snd.mp3")]).returncode == 0, 0
    if ok:
        with pytest.raises(ValueError, match="лише звук"):
            video_info(str(d / "snd.mp3"))
    assert ".mts" in VIDEO_EXT and ".3gp" in VIDEO_EXT


def test_huge_animation_is_downscaled(d, monkeypatch):
    frames = [Image.fromarray(np.full((200, 300, 3), i * 40, np.uint8)) for i in range(5)]
    p = str(d / "big.gif")
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=50, loop=0)
    monkeypatch.setattr(media, "ANIM_MAX_BYTES", 5 * 200 * 300 * 3 // 4)      # «пам'яті» вдвічі менше, ніж треба
    src = media.GifSource(p)
    assert src.info["width"] < 300 and src.info["note"]
    assert src.read(0).shape[1] == src.info["width"]
