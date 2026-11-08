"""Датасет для дистиляції рівня «Ультра» у швидку модель.

Бере скріншоти й випадкові кадри відео, робить із них «фото» випадкових розмірів (висота 48…2160 px, до 4096 px по довгій
стороні, апскейл дозволений), форматів (PNG / JPEG / WebP / BMP) і якості, з природними спотвореннями, а мішень рахує справжній
алгоритм InkReel - Ultra, окремий кадр, лише CPU (INKREEL_GPU=off), щоб пара була відтворюваною.

    python tools/make_dataset.py                    # основний запуск (можна переривати Ctrl+C і запускати знову - продовжить)
    python tools/make_dataset.py --smoke            # 36 зразків у E:\\Inkreel_Dataset\\_smoke + preview.jpg: подивитись, перш ніж гнати все
    python tools/make_dataset.py check --n 40       # перерахувати мішені з файлів і порівняти (перевірка відтворюваності)
    python tools/make_dataset.py sheet --n 16       # контактний аркуш preview.jpg (вхід | чорнило)

Структура:  OUT/{train,val,test}/in/NNNN/ID.ext   OUT/{train,val,test}/ink/NNNN/ID.png (1-біт, 255 = світло)
            OUT/manifest.jsonl  OUT/config.json  OUT/index.json  OUT/chunks.done  OUT/errors.log
Розподіл на train/val/test - за ДЖЕРЕЛОМ (одне відео / один скріншот цілком в одній частині): кадри одного відео не просочуються
між навчанням і перевіркою. Усе детерміновано: зразок задається (seed, номер шматка, номер кадру), тож повторний запуск
дописує те саме, а не дублює."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")          # до імпорту cv2: без потоку повідомлень декодера
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEF_OUT = r"E:\Inkreel_Dataset"
DEF_VIDEOS = r"D:\alts"
DEF_SHOTS = r"C:\Users\Tima\Pictures\Screenshots"
VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".m4v", ".wmv", ".flv"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
# верхні теки відео, які за замовчуванням пропускаються: месенджер, віддалений стіл, проєкт монтажу (чужі люди й листування)
EXCLUDE = ("viber", "parsec", "parsecd", "clip champ")

CV_CODECS = {"h264", "mpeg4", "mjpeg"}   # решту (AV1, HEVC, VP9…) читаємо ffmpeg-ом напряму: OpenCV після перемотки в AV1 часто не має
                                         # заголовка послідовності, засипає лог помилками й дочитує файл до кінця, перш ніж здатись
INDEX_VER = 2
STD_H = (72, 90, 96, 120, 144, 180, 240, 288, 360, 480, 540, 576, 720, 900, 1080, 1200, 1440, 1600, 2160)
ASPECTS = ((16, 9), (16, 10), (4, 3), (3, 2), (5, 4), (1, 1), (21, 9), (32, 9), (9, 16), (3, 4), (2, 3), (9, 21))
ASP_P = np.array([34, 8, 10, 6, 3, 8, 7, 2, 10, 5, 4, 3], np.float64)
ASP_P /= ASP_P.sum()
MAX_SIDE = 4096
MIN_SIDE = 32
CHUNK_VIDEO = (8, 16)                 # зразків з одного відео за шматок (рівномірно розкинуті за часом)
CHUNK_SHOT = 24                       # зразків зі скріншотів за шматок
SHARD = 200                           # шматків у одній теці shard (≈ 3-5 тис. файлів)

# ---------------------------------------------------------------- допоміжне


def _log_uniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def split_of(key: str) -> str:
    v = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if v < 90 else "val" if v < 95 else "test"


def algo_id() -> str:
    """відбиток алгоритму: версія + хеш файлів, що визначають мішень. Датасет, зроблений іншою версією, зі старим не змішуємо"""
    h = hashlib.md5()
    for f in ("core.py", "ink5.py", "motion.py"):
        with open(os.path.join(ROOT, "benvid", f), "rb") as fh:
            h.update(fh.read())
    try:
        from benvid import __version__ as v
    except Exception:  # noqa: BLE001
        v = "?"
    return f"{v}-{h.hexdigest()[:10]}"


def _gb(n):
    return n / (1 << 30)


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- індекс джерел
def _walk(root, exts, exclude):
    out = []
    root = os.path.abspath(root)
    for dp, dn, fn in os.walk(root):
        rel_dir = os.path.relpath(dp, root)
        if rel_dir != "." and rel_dir.split(os.sep)[0].lower() in exclude:
            dn[:] = []
            continue
        dn.sort()
        for f in sorted(fn):
            if os.path.splitext(f)[1].lower() in exts:
                p = os.path.join(dp, f)
                out.append((p, os.path.relpath(p, root).replace("\\", "/")))
    return out


def _probe_video(item):
    path, rel = item
    try:
        from benvid.video import probe_stream
        pr = probe_stream(path)
        codec, n, fps, w, h = pr.get("codec", ""), 0, float(pr.get("fps") or 0), *(pr.get("size") or (0, 0))
        if pr.get("duration") and 1 <= fps <= 1000:
            n = int(pr["duration"] * fps)
        if n <= 0 or w <= 0:                                  # ffmpeg не допоміг - властивості контейнера через OpenCV
            from benvid.video import _open_cv
            cap = _open_cv(path)
            if not cap.isOpened():
                return None
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
        if n < 24 or w < 64 or h < 64 or not (1 <= fps <= 1000):
            return None
        return dict(path=path, rel=rel, top=rel.split("/")[0] if "/" in rel else "", n=n, fps=round(fps, 3), w=w, h=h,
                    dur=round(n / fps, 2), codec=codec)
    except Exception:  # noqa: BLE001
        return None


def build_index(videos, shots, exclude, log=print):
    from multiprocessing.pool import ThreadPool
    idx = dict(ver=INDEX_VER, videos_root=videos, shots_root=shots, exclude=list(exclude), videos=[], shots=[], bad=0)
    if videos and os.path.isdir(videos):
        items = _walk(videos, VIDEO_EXT, set(exclude))
        log(f"відео: знайдено {len(items)} файлів, читаю тривалість…")
        with ThreadPool(8) as tp:
            for i, r in enumerate(tp.imap(_probe_video, items), 1):
                if r:
                    idx["videos"].append(r)
                else:
                    idx["bad"] += 1
                if i % 100 == 0:
                    log(f"  прочитано {i} з {len(items)}")
    if shots and os.path.isdir(shots):
        idx["shots"] = [dict(path=p, rel=r) for p, r in _walk(shots, IMAGE_EXT, set())]
    dur = sum(v["dur"] for v in idx["videos"])
    cc = {}
    for v in idx["videos"]:
        cc[v["codec"] or "?"] = cc.get(v["codec"] or "?", 0) + 1
    log(f"джерела: відео {len(idx['videos'])} ({dur / 3600:.1f} год, пропущено {idx['bad']}), скріншотів {len(idx['shots'])}; "
        f"кодеки: {', '.join(f'{k} {n}' for k, n in sorted(cc.items()))}")
    return idx


def load_index(out, videos, shots, exclude, reindex=False, log=print):
    fp = os.path.join(out, "index.json")
    if not reindex and os.path.isfile(fp):
        try:
            with open(fp, encoding="utf-8") as f:
                idx = json.load(f)
            if idx.get("ver") == INDEX_VER and idx.get("videos_root") == videos and idx.get("shots_root") == shots and idx.get("exclude") == list(exclude):
                return idx
        except (OSError, ValueError):
            pass
    idx = build_index(videos, shots, exclude, log)
    _write_json(fp, idx)
    return idx


def source_weights(idx):
    """теки (ігри) беруться за ~sqrt(тривалості): CS:GO на 200 ГБ не заглушає решту; усередині теки - за тривалістю"""
    tops = {}
    for i, v in enumerate(idx["videos"]):
        tops.setdefault(v["top"], []).append(i)
    ft = {t: sum(idx["videos"][i]["dur"] for i in ii) for t, ii in tops.items()}
    fw = {t: math.sqrt(d) for t, d in ft.items()}
    z = sum(fw.values()) or 1.0
    w = np.zeros(len(idx["videos"]), np.float64)
    for t, ii in tops.items():
        for i in ii:
            w[i] = fw[t] / z * idx["videos"][i]["dur"] / ft[t]
    return w / (w.sum() or 1.0)


# ---------------------------------------------------------------- підготовка одного «фото»
def pick_size(rng, sw, sh, c):
    std = [x for x in STD_H if c["hlo"] <= x <= c["hhi"]]
    h = float(std[rng.integers(len(std))]) if (std and rng.random() < 0.30) else _log_uniform(rng, c["hlo"], c["hhi"])
    r = rng.random()
    if r < 0.45:
        asp = sw / sh
    elif r < 0.85:
        a, b = ASPECTS[rng.choice(len(ASPECTS), p=ASP_P)]
        asp = a / b
    else:
        asp = _log_uniform(rng, 0.5, 2.4)
    asp = min(max(asp, 0.25), 4.0)
    w = h * asp
    if max(w, h) > c["max_side"]:
        k = c["max_side"] / max(w, h)
        w, h = w * k, h * k
    if min(w, h) < MIN_SIDE:
        k = MIN_SIDE / min(w, h)
        w, h = w * k, h * k
    ev = lambda x: max(2, int(x / 2 + 0.5) * 2)        # парні сторони, як у застосунку
    return ev(w), ev(h)


def pick_crop(rng, sw, sh, w, h):
    asp = w / h
    cw0, ch0 = (sh * asp, float(sh)) if sw / sh >= asp else (float(sw), sw / asp)
    s = 1.0 if rng.random() < 0.5 else rng.uniform(0.3, 0.95)
    cw, ch = max(8, min(sw, int(cw0 * s))), max(8, min(sh, int(ch0 * s)))
    x, y = int(rng.integers(0, sw - cw + 1)), int(rng.integers(0, sh - ch + 1))
    return x, y, cw, ch


def resize_like_app(img, w, h):
    shrink = img.shape[1] >= w and img.shape[0] >= h
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)


def degrade(img, rng):
    """природні спотворення входу: м'яке/піксельне джерело, розмиття, різкість, колір, шум, чорні смуги"""
    aug = {}
    h, w = img.shape[:2]
    if rng.random() < 0.15 and min(h, w) >= 64:
        f = rng.uniform(1.5, 4.0)
        small = cv2.resize(img, (max(16, int(w / f)), max(16, int(h / f))), interpolation=cv2.INTER_AREA)
        near = rng.random() < 0.25
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST if near else cv2.INTER_CUBIC)
        aug["soft"] = [round(f, 2), int(near)]
    if rng.random() < 0.10:
        s = rng.uniform(0.4, 1.6)
        img = cv2.GaussianBlur(img, (0, 0), s)
        aug["blur"] = round(s, 2)
    elif rng.random() < 0.08:
        a, s = rng.uniform(0.4, 1.5), rng.uniform(0.6, 1.8)
        img = cv2.addWeighted(img, 1 + a, cv2.GaussianBlur(img, (0, 0), s), -a, 0)
        aug["sharp"] = [round(a, 2), round(s, 2)]
    if rng.random() < 0.40:                                           # яскравість / контраст / гама
        g, ct, br = _log_uniform(rng, 0.6, 1.7), rng.uniform(0.7, 1.3), rng.uniform(-25, 25)
        x = np.arange(256, dtype=np.float32) / 255.0
        lut = np.clip(((x ** g) - 0.5) * ct * 255.0 + 127.5 + br, 0, 255).astype(np.uint8)
        img = cv2.LUT(img, lut)
        aug["tone"] = [round(g, 2), round(ct, 2), round(br, 1)]
    gray = rng.random() < 0.06
    if gray:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        aug["gray"] = 1
    elif rng.random() < 0.35:                                         # насиченість і зсув відтінку
        sat, hue = rng.uniform(0.0, 1.7), (int(rng.integers(-40, 41)) if rng.random() < 0.3 else 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * sat, 0, 255).astype(np.uint8)
        if hue:
            hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue) % 180
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        aug["color"] = [round(sat, 2), hue]
    if rng.random() < 0.25:
        sg = _log_uniform(rng, 1.0, 8.0)
        n = rng.standard_normal(img.shape, dtype=np.float32) * np.float32(sg)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
        aug["noise"] = round(sg, 2)
    if rng.random() < 0.04:                                           # чорні смуги кіно / вертикальне відео
        b = rng.uniform(0.04, 0.14)
        img = img.copy()
        if rng.random() < 0.7:
            t = max(1, int(h * b))
            img[:t] = 0
            img[h - t:] = 0
        else:
            t = max(1, int(w * b))
            img[:, :t] = 0
            img[:, w - t:] = 0
        aug["bars"] = round(b, 3)
    return np.ascontiguousarray(img), aug, gray


def pick_format(rng, h):
    r = rng.random()
    if r < 0.40:
        return "png", 0, 0
    if r < 0.72:
        return "jpg", int(rng.uniform(40, 99)), 2 if rng.random() < 0.7 else 0       # 2 = 4:2:0, 0 = 4:4:4
    if r < 0.94:
        return ("webp", 100, 0) if rng.random() < 0.2 else ("webp", int(rng.uniform(40, 99)), 0)
    return ("bmp", 0, 0) if h <= 480 else ("png", 0, 0)


def encode(img, fmt, q, sub, gray, rng):
    """-> (bytes, розширення, опис формату)"""
    from PIL import Image
    import io
    arr = img[:, :, 0] if (gray and fmt in ("png", "jpg")) and rng.random() < 0.5 else img
    if fmt == "png":
        ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        return buf.tobytes(), ".png", "png"
    if fmt == "bmp":
        ok, buf = cv2.imencode(".bmp", arr)
        return buf.tobytes(), ".bmp", "bmp"
    im = Image.fromarray(arr if arr.ndim == 2 else arr[:, :, ::-1])
    bio = io.BytesIO()
    if fmt == "jpg":
        im.save(bio, "JPEG", quality=q, subsampling=sub, optimize=bool(rng.random() < 0.5), progressive=bool(rng.random() < 0.2))
        return bio.getvalue(), (".jpeg" if rng.random() < 0.2 else ".jpg"), f"jpg:q{q}:{'420' if sub == 2 else '444'}"
    if q >= 100:
        im.save(bio, "WEBP", lossless=True, quality=60, method=2)
        return bio.getvalue(), ".webp", "webp:lossless"
    im.save(bio, "WEBP", quality=q, method=2)
    return bio.getvalue(), ".webp", f"webp:q{q}"


def interesting(img, rng):
    """відсіює майже чорні/білі/плоскі кадри (завантаження, затемнення), лишаючи 5% для різноманіття"""
    g = cv2.cvtColor(cv2.resize(img, (96, max(2, int(96 * img.shape[0] / img.shape[1]))), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    ok = g.std() >= 5.0 and 6.0 < g.mean() < 249.0
    return ok or rng.random() < 0.05


# ---------------------------------------------------------------- воркер
_W: dict = {}


def _init_worker(wc, idx, inproc=False):
    os.environ["INKREEL_GPU"] = "off"
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    cv2.setNumThreads(1 if not inproc else 0)
    try:
        from benvid import gpu, ink5
        gpu.set_allowed(False)
        if not inproc:
            from benvid import video
            video._lower_priority()
        ink5.prewarm()
    except Exception:  # noqa: BLE001
        pass
    _W["wc"], _W["idx"] = wc, idx
    _W["vw"] = source_weights(idx) if idx["videos"] else None


def _paths(wc, split, c, sid, ext):
    shard = f"{c // SHARD:04d}"
    pin = os.path.join(wc["out"], split, "in", shard, sid + ext)
    pout = os.path.join(wc["out"], split, "ink", shard, sid + ".png")
    return pin, pout


def _save_mask(m, path):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(m.astype(bool)).save(path, "PNG", compress_level=6)


def _make_sample(src, rng, wc, c, sid, meta):
    """src - BGR кадр-джерело -> запис манифесту (файли записані) або None"""
    from benvid.core import Params, binarize
    from benvid.media import imread_u
    sh, sw = src.shape[:2]
    w, h = pick_size(rng, sw, sh, wc)
    x, y, cw, ch = pick_crop(rng, sw, sh, w, h)
    img = resize_like_app(src[y:y + ch, x:x + cw], w, h)
    img, aug, gray = degrade(img, rng)
    fmt, q, sub = pick_format(rng, h)
    data, ext, fdesc = encode(img, fmt, q, sub, gray, rng)
    pin, pout = _paths(wc, meta["split"], c, sid, ext)
    os.makedirs(os.path.dirname(pin), exist_ok=True)
    with open(pin, "wb") as f:
        f.write(data)
    cur = imread_u(pin)                                  # рівно те, що побачить застосунок (і модель) з цього файлу
    if cur.shape[0] != h or cur.shape[1] != w:
        raise ValueError(f"розмір після читання {cur.shape[1]}x{cur.shape[0]} замість {w}x{h}")
    t0 = time.time()
    m = binarize(cur, w, h, Params(method=5))
    ms = (time.time() - t0) * 1000
    _save_mask(m, pout)
    return dict(id=sid, chunk=c, split=meta["split"], kind=meta["kind"], src=meta["src"], frame=meta.get("frame"),
                t=meta.get("t"), crop=[x, y, cw, ch], src_wh=[sw, sh], w=w, h=h, fmt=fdesc, aug=aug,
                inp=os.path.relpath(pin, wc["out"]).replace("\\", "/"), ink=os.path.relpath(pout, wc["out"]).replace("\\", "/"),
                in_bytes=len(data), ink_bytes=os.path.getsize(pout), white=round(float(m.mean()), 4), ms=round(ms, 1))


def _open_video(v):
    """H.264 і подібні - через OpenCV (швидко); AV1/HEVC/VP9… - одразу ffmpeg, бо перемотка OpenCV-ом у них ненадійна"""
    from benvid.video import RobustCapture, find_ffmpeg, open_capture
    cap = None
    if (v.get("codec") or "") not in CV_CODECS and find_ffmpeg():
        cap = RobustCapture(v["path"], force_ff=True)
    if cap is None or not cap.isOpened():
        cap = open_capture(v["path"])
    if not cap.isOpened():
        raise OSError("не відкрилось")
    return cap


def _chunk_task(c):
    """один шматок: або кадри з одного відео, або пачка скріншотів -> (записи, помилки)"""
    from benvid.media import imread_u
    wc, idx = _W["wc"], _W["idx"]
    recs, errs = [], []
    rng = np.random.default_rng([wc["seed"], c, 0])
    vw = _W["vw"]
    shot = (not idx["videos"]) or (idx["shots"] and rng.random() < wc["p_shot"])
    if shot:
        picks = rng.integers(0, len(idx["shots"]), CHUNK_SHOT)
        for j, si in enumerate(picks):
            s = idx["shots"][int(si)]
            sid = f"{c:06d}_{j:02d}"
            try:
                src = imread_u(s["path"])
                r2 = np.random.default_rng([wc["seed"], c, j + 1, 0])
                if not interesting(src, r2):
                    continue
                recs.append(_make_sample(src, r2, wc, c, sid, dict(split=split_of(s["rel"]), kind="shot", src="shots/" + s["rel"])))
            except Exception as e:  # noqa: BLE001
                errs.append(f"{sid} {s['rel']}: {type(e).__name__}: {e}")
        return recs, errs
    v = idx["videos"][int(rng.choice(len(vw), p=vw))]
    n_s = int(rng.integers(CHUNK_VIDEO[0], CHUNK_VIDEO[1] + 1))
    split = split_of(v["rel"])
    try:
        cap = _open_video(v)
    except Exception as e:  # noqa: BLE001
        return recs, [f"{c:06d}: {v['rel']}: {type(e).__name__}: {e}"]
    k = 0
    try:
        for j in range(n_s):
            r2 = np.random.default_rng([wc["seed"], c, j + 1, 0])
            for _try in range(4):                               # у своїй смузі часу шукаємо не порожній кадр
                f = int((j + r2.random()) / n_s * (v["n"] - 2)) + 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, src = cap.read()
                if ok and src is not None and src.ndim == 3 and interesting(src, r2):
                    break
                ok = False
            if not ok:
                continue
            for var in range(1 + int(r2.random() < 0.3)):       # з одного кадру - 1-2 різні варіанти (розмір, обрізка, формат)
                sid = f"{c:06d}_{k:02d}"
                k += 1
                try:
                    r3 = np.random.default_rng([wc["seed"], c, j + 1, var + 1])
                    recs.append(_make_sample(src, r3, wc, c, sid, dict(split=split, kind="video", src="video/" + v["rel"],
                                                                      frame=f, t=round(f / v["fps"], 2))))
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{sid} {v['rel']} #{f}: {type(e).__name__}: {e}")
    finally:
        cap.release()
    return recs, errs


# ---------------------------------------------------------------- керування
def load_manifest(out):
    recs = {}
    fp = os.path.join(out, "manifest.jsonl")
    if os.path.isfile(fp):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    recs[r["id"]] = r
                except (ValueError, KeyError):
                    pass
    return recs


def load_done(out):
    fp = os.path.join(out, "chunks.done")
    if not os.path.isfile(fp):
        return set()
    with open(fp, encoding="utf-8") as f:
        return {int(x) for x in f.read().split() if x.isdigit()}


def default_workers():
    cpu = os.cpu_count() or 2
    try:
        from benvid.video import ram_bytes
        ram = ram_bytes() / (1 << 30)
    except Exception:  # noqa: BLE001
        ram = 16.0
    return max(1, min(cpu - 6, int(ram // 4), 24))


def run(a):
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    log = print
    aid = algo_id()
    cfg_fp = os.path.join(out, "config.json")
    key = dict(seed=a.seed, hlo=a.hlo, hhi=a.hhi, max_side=MAX_SIDE, p_shot=a.p_shot)
    if os.path.isfile(cfg_fp):
        with open(cfg_fp, encoding="utf-8") as f:
            old = json.load(f)
        diff = [k for k in key if old.get(k) != key[k]] + (["algo"] if old.get("algo") != aid else [])
        if diff and not a.force:
            log(f"УВАГА: у {out} датасет зроблено з іншими налаштуваннями ({', '.join(diff)}).\n"
                f"Змішувати не можна: або вкажи іншу теку (--out), або додай --force, якщо справді треба продовжити.")
            return 2
    else:
        _write_json(cfg_fp, dict(key, algo=aid, created=time.strftime("%Y-%m-%d %H:%M:%S"), reference="Ultra, одиночний кадр, CPU",
                                 videos=a.videos, shots=a.shots, exclude=list(a.exclude)))
    idx = load_index(out, a.videos, a.shots, a.exclude, a.reindex, log)
    if not idx["videos"] and not idx["shots"]:
        log("немає ні відео, ні скріншотів - перевір шляхи --videos / --shots")
        return 2
    recs = load_manifest(out)
    done = load_done(out)
    n_tot, b_tot = len(recs), sum(r["in_bytes"] + r["ink_bytes"] for r in recs.values())
    log(f"алгоритм {aid}; у теці вже {n_tot} зразків, {_gb(b_tot):.1f} ГБ, шматків {len(done)}")
    workers = a.workers if a.workers is not None else default_workers()
    wc = dict(out=out, seed=a.seed, hlo=a.hlo, hhi=a.hhi, max_side=MAX_SIDE, p_shot=a.p_shot)
    max_b, min_free = a.max_gb * (1 << 30), a.min_free_gb * (1 << 30)
    log(f"процесів: {workers}; ціль: {a.count} зразків або {a.max_gb:g} ГБ; Ctrl+C - зупинити (потім можна продовжити)")
    pool = None
    if workers > 0:
        os.environ["INKREEL_GPU"] = "off"
        pool = mp.get_context("spawn").Pool(workers, _init_worker, (wc, idx))
    else:
        _init_worker(wc, idx, inproc=True)
    pend, c_next, t0, t_log = {}, 0, time.time(), 0.0
    n0, why = n_tot, ""
    mf = open(os.path.join(out, "manifest.jsonl"), "a", encoding="utf-8")
    df = open(os.path.join(out, "chunks.done"), "a", encoding="utf-8")
    ef = open(os.path.join(out, "errors.log"), "a", encoding="utf-8")
    try:
        while True:
            if not why:
                if n_tot >= a.count:
                    why = "досягнуто ціль за кількістю"
                elif b_tot >= max_b:
                    why = "досягнуто ліміт місця"
                elif shutil.disk_usage(out).free < min_free:
                    why = f"на диску лишилось менше {a.min_free_gb:g} ГБ"
            while not why and len(pend) < max(1, workers * 2):
                while c_next in done:
                    c_next += 1
                if pool is None:
                    res = _chunk_task(c_next)
                    pend[c_next] = res
                else:
                    pend[c_next] = pool.apply_async(_chunk_task, (c_next,))
                c_next += 1
                if pool is None:
                    break
            if not pend:
                break
            got = False
            for c in list(pend):
                r = pend[c]
                if pool is not None:
                    if not r.ready():
                        continue
                    r = r.get()
                del pend[c]
                got = True
                rs, es = r
                for rec in rs:
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    recs[rec["id"]] = rec
                    n_tot += 1
                    b_tot += rec["in_bytes"] + rec["ink_bytes"]
                for e in es:
                    ef.write(e + "\n")
                mf.flush()
                ef.flush()
                df.write(f"{c}\n")
                df.flush()
                done.add(c)
            now = time.time()
            if now - t_log > 5:
                t_log = now
                dt = max(1e-6, now - t0)
                rate = (n_tot - n0) / dt
                eta_n = (a.count - n_tot) / rate if rate > 0 else float("inf")
                eta_b = (max_b - b_tot) / ((b_tot) / max(1, n_tot) * rate) if rate > 0 and n_tot else float("inf")
                eta = max(0.0, min(eta_n, eta_b))
                if why:
                    msg = f"[{why}] - дочекаюсь запущених шматків"
                elif math.isfinite(eta):
                    msg = f"({rate:.1f}/с)  шматків {len(done)}  лишилось ≈ {eta / 3600:.1f} год"
                else:
                    msg = "рахую швидкість…"
                print(f"  зразків {n_tot}  {_gb(b_tot):.1f} ГБ  {msg}", flush=True)
            if not got:
                time.sleep(0.2)
    except KeyboardInterrupt:
        why = "зупинено вручну"
        print("\nзупиняю…", flush=True)
        if pool is not None:
            pool.terminate()
    finally:
        mf.close()
        df.close()
        ef.close()
        if pool is not None:
            if not why.startswith("зупинено"):
                pool.close()
            pool.join()
    dt = time.time() - t0
    new = n_tot - n0
    log(f"\nготово ({why or 'кінець'}): зразків {n_tot} (+{new} за {dt / 60:.1f} хв, {new / max(dt, 1e-6):.2f}/с), "
        f"{_gb(b_tot):.1f} ГБ, у середньому {b_tot / max(1, n_tot) / (1 << 20):.2f} МБ на зразок")
    if os.path.getsize(os.path.join(out, "errors.log")):
        log(f"були помилки на окремих кадрах - дивись {os.path.join(out, 'errors.log')}")
    return 0


# ---------------------------------------------------------------- перевірка й прев'ю
def cmd_check(a):
    from PIL import Image
    from benvid.core import Params, binarize
    from benvid.media import imread_u
    os.environ["INKREEL_GPU"] = "off"
    out = os.path.abspath(a.out)
    recs = list(load_manifest(out).values())
    if not recs:
        print("манифест порожній")
        return 2
    rnd = random.Random(a.seed)
    pick = rnd.sample(recs, min(a.n, len(recs)))
    bad = 0
    for r in pick:
        cur = imread_u(os.path.join(out, r["inp"]))
        m = binarize(cur, r["w"], r["h"], Params(method=5))
        ref = np.array(Image.open(os.path.join(out, r["ink"]))).astype(bool)
        d = float((m.astype(bool) != ref).mean())
        bad += d > 0
        print(f"  {r['id']}  {r['w']}x{r['h']}  {r['fmt']:<14} відмінних пікселів {d * 100:.4f}%")
    print(f"перевірено {len(pick)}, з відмінностями {bad}")
    return 0 if not bad else 1


def cmd_sheet(a):
    from PIL import Image
    out = os.path.abspath(a.out)
    recs = list(load_manifest(out).values())
    if not recs:
        print("манифест порожній")
        return 2
    rnd = random.Random(a.seed)
    pick = rnd.sample(recs, min(a.n, len(recs)))
    TH = 240
    rows = []
    for r in pick:
        i = Image.open(os.path.join(out, r["inp"])).convert("RGB")
        k = Image.open(os.path.join(out, r["ink"])).convert("L")
        s = TH / i.height
        sz = (max(8, int(i.width * s)), TH)
        rows.append((i.resize(sz, Image.LANCZOS), k.resize(sz, Image.LANCZOS), r))
    cols = 2
    cw = max(p[0].width for p in rows)
    cells = [(r[0], r[1]) for r in rows]
    per_row = 2
    W = per_row * (2 * cw + 8) + 8
    H = ((len(cells) + per_row - 1) // per_row) * (TH + 8) + 8
    sheet = Image.new("RGB", (W, H), (40, 40, 40))
    for n, (i, k) in enumerate(cells):
        x = 8 + (n % per_row) * (2 * cw + 8)
        y = 8 + (n // per_row) * (TH + 8)
        sheet.paste(i, (x, y))
        sheet.paste(k.convert("RGB"), (x + cw, y))
    fp = os.path.join(out, "preview.jpg")
    sheet.save(fp, quality=88)
    print("прев'ю:", fp)
    return 0


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Датасет для дистиляції «Ультра»")
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "check", "sheet"])
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--videos", default=DEF_VIDEOS)
    ap.add_argument("--shots", default=DEF_SHOTS)
    ap.add_argument("--exclude", nargs="*", default=list(EXCLUDE), help="верхні теки відео, які пропустити (без урахування регістру)")
    ap.add_argument("--count", type=int, default=100000, help="скільки зразків усього")
    ap.add_argument("--max-gb", type=float, default=180.0, help="скільки ГБ датасет може зайняти")
    ap.add_argument("--min-free-gb", type=float, default=15.0, help="зупинитись, якщо на диску лишилось менше")
    ap.add_argument("--workers", type=int, default=None, help="процесів (0 - в одному процесі)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hlo", type=int, default=48)
    ap.add_argument("--hhi", type=int, default=2160)
    ap.add_argument("--p-shot", type=float, default=0.12, help="частка шматків зі скріншотів")
    ap.add_argument("--n", type=int, default=16, help="check / sheet: скільки зразків")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="пробний прогін: 36 зразків у ТЕКА\\_smoke й аркуш preview.jpg")
    a = ap.parse_args(argv)
    a.exclude = [x.lower() for x in a.exclude]
    if a.smoke:
        a.out = os.path.join(a.out, "_smoke")
        a.count, a.p_shot = 36, 0.25
    if a.cmd == "check":
        return cmd_check(a)
    if a.cmd == "sheet":
        return cmd_sheet(a)
    rc = run(a)
    if a.smoke and rc == 0:
        a.n = 16
        cmd_sheet(a)
    return rc


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
