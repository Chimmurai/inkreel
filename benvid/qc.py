"""Контрольний тест якості InkReel:   python -m benvid.qc [--quick] [--formats] [--speed] [--images ТЕКА] [--out ТЕКА]

Не потребує жодних зовнішніх файлів: картинки синтетичні (градієнт, «портрет», текст двох полярностей, тонкі лінії, шумова
текстура, ізолюмінантні кольори). Для кожного рівня якості (1-5) рахує ЧИСЛА, що можна порівнювати між версіями:

  тон     - наскільки густина чорного/білого, розмита як сприймає око (σ=2.5 і 5 px), збігається зі світлотою джерела
            (RMSE, менше = краще). Поріг (рівень 1) програє тут завжди - він не передає півтонів;
  контур  - F1 збігу контурів розмитої бінарної картинки з контурами джерела (більше = краще);
  текст   - IoU чорнила: чи читається дрібний текст (більше = краще), окремо чорне на білому й біле на чорному;
  повтор  - автокореляція «шуму» на довгих відстанях: ловить квадратну сітку з повторюваних плиток (менше = краще);
  мерехтіння - частка пікселів, що змінилися між сусідніми кадрами НЕРУХОМОЇ сцени із шумом камери (менше = краще);
  швидкість - мс на кадр 640×360 (один потік).

З --formats: кодує синтетичний ролик у всі формати експорту й міряє «чистоту» (скільки пікселів після стиснення не чисто чорні
й не чисто білі), розмір і наявність звуку. З --images: те саме для ваших картинок плюс контактна таблиця qc_sheet.png.
Звіт друкується й пишеться в qc_report.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

from .core import InkState, Params, binarize


def _imread_gray(path):
    """cv2.imread не читає шляхи з кирилицею на Windows (імена з якістю «Висока»): читаємо байти самі"""
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    except OSError:
        return None

LEVELS = (1, 2, 3, 4, 5)


# ------------------------------------------------------------------ синтетичні картинки
def _text(img, s, org, scale, fg, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def synthetic_set(w: int = 480, h: int = 270) -> dict:
    """-> {назва: BGR uint8}. Усі розміру w×h, щоб порівнювати рівні без перемасштабування."""
    rng = np.random.default_rng(7)
    out = {}
    # 1) плавний градієнт зі сходинками світлоти: чи не видно швів і банди
    g = np.linspace(0, 255, w, dtype=np.float32)[None, :].repeat(h, 0)
    g = 0.75 * g + 0.25 * np.linspace(255, 0, h, dtype=np.float32)[:, None]
    out["градієнт"] = cv2.cvtColor(g.clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # 2) «портрет»: м'які плями шкіри, тіні, фон
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    face = np.exp(-(((xx - w * 0.5) / (w * 0.22)) ** 2 + ((yy - h * 0.5) / (h * 0.38)) ** 2))
    eye = sum(np.exp(-(((xx - w * cx) / 14) ** 2 + ((yy - h * 0.42) / 7) ** 2)) for cx in (0.43, 0.57))
    lum = 70 + 130 * face - 90 * eye + 20 * np.sin(xx / 37.0)
    img = np.stack([lum * 0.85, lum * 0.95, lum * 1.05], -1).clip(0, 255).astype(np.uint8)
    out["портрет"] = cv2.GaussianBlur(img, (0, 0), 1.2)
    # 3) текст: чорне на світлому тлі, різні розміри (кегль 9-24 px)
    t = np.full((h, w, 3), 235, np.uint8)
    for i, (s, sc) in enumerate((("InkReel 1234", 0.55), ("The quick brown fox", 0.42), ("Ukrainian: Привіт, світе", 0.38), ("0123456789 abc", 0.3))):
        _text(t, s, (10, 38 + i * 56), sc, (20, 20, 20))
    out["текст (чорне на білому)"] = t
    # 4) текст: світле на темному тлі
    out["текст (біле на чорному)"] = (255 - t).copy()
    # 5) тонкі лінії й дрібні деталі: сітка, діагоналі, кола
    ln = np.full((h, w, 3), 210, np.uint8)
    for x in range(10, w, 24):
        cv2.line(ln, (x, 0), (x, h), (60, 60, 60), 1, cv2.LINE_AA)
    for y in range(8, h, 30):
        cv2.line(ln, (0, y), (w, y), (90, 90, 90), 1, cv2.LINE_AA)
    cv2.circle(ln, (w // 2, h // 2), h // 3, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.line(ln, (0, 0), (w, h), (20, 20, 20), 1, cv2.LINE_AA)
    out["тонкі лінії"] = ln
    # 6) текстура: розмитий шум (трава/камінь) + плавна віньєтка
    tex = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 3.0)
    tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-9)
    vig = np.exp(-(((xx - w / 2) / (w * 0.7)) ** 2 + ((yy - h / 2) / (h * 0.7)) ** 2))[..., None]
    out["текстура"] = (255 * (0.15 + 0.7 * tex * vig)).clip(0, 255).astype(np.uint8)
    # 7) кольорові плями однакової світлоти (червоне/зелене/синє/сіре): тон має братись зі світлоти, а не з одного каналу
    col = np.full((h, w, 3), 128, np.uint8)
    cells = ((0, 0, 255), (0, 150, 0), (190, 60, 0), (128, 128, 128))
    for i, c in enumerate(cells):
        col[:, i * w // 4:(i + 1) * w // 4] = c
    out["кольори"] = col
    return out


# ------------------------------------------------------------------ метрики
def _gray(img: np.ndarray, w: int, h: int) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if g.shape[:2] != (h, w):
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    return g.astype(np.float32) / 255.0


def tone_rmse(g: np.ndarray, b: np.ndarray, sigma: float) -> float:
    bl = cv2.GaussianBlur(b.astype(np.float32), (0, 0), sigma)
    gl = cv2.GaussianBlur(g, (0, 0), sigma)
    return float(np.sqrt(np.mean((bl - gl) ** 2)))


def _edges(x: np.ndarray, sigma: float) -> np.ndarray:
    u = (cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigma) * 255).astype(np.uint8)
    return cv2.Canny(u, 40, 100) > 0


def edge_f1(g: np.ndarray, b: np.ndarray, sigma: float = 1.5, tol: int = 2) -> float:
    eg, eb = _edges(g, sigma), _edges(b, sigma)
    if not eg.any() and not eb.any():
        return 1.0
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    dg, db = cv2.dilate(eg.astype(np.uint8), k) > 0, cv2.dilate(eb.astype(np.uint8), k) > 0
    p = (eb & dg).sum() / max(1, eb.sum())
    r = (eg & db).sum() / max(1, eg.sum())
    return float(2 * p * r / max(1e-9, p + r))


def text_iou(img: np.ndarray, b: np.ndarray, dark_ink: bool) -> float:
    """IoU «чорнила»: еталон - пороги по джерелу на тому ж розмірі, тло визначається як мода світлоти"""
    g = _gray(img, b.shape[1], b.shape[0])
    bg = float(np.median(g))
    ink_ref = (g < bg - 0.25) if dark_ink else (g > bg + 0.25)
    pred = (b == 0) if dark_ink else (b > 0)
    near = cv2.dilate(ink_ref.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0       # лише поблизу тексту
    u = (ink_ref | (pred & near)).sum()
    return float((ink_ref & pred).sum() / max(1, u))


def repeat_score(b: np.ndarray, lags=(16, 24, 32, 48, 64, 96, 128)) -> float:
    """максимум нормованої автокореляції високочастотної частини бінарної картинки на великих зсувах по x та y: плитки, що
    повторюються (квадратна сітка), дають значення, близькі до 1; випадковий шум - близько 0"""
    x = b.astype(np.float32)
    hp = x - cv2.GaussianBlur(x, (0, 0), 3.0)
    hp -= hp.mean()
    den = float((hp * hp).mean()) + 1e-9
    best = 0.0
    for L in lags:
        for ax in (1, 0):
            if hp.shape[ax] <= L * 2:
                continue
            a, c = (hp[:, :-L], hp[:, L:]) if ax == 1 else (hp[:-L], hp[L:])
            best = max(best, float((a * c).mean() / den))
    return best


def _aux(level: int):
    """пам'ять між кадрами для рівня: 5 - InkState, 6 (нейро) - згладжування опису кадру, решта - без пам'яті"""
    if level == 5:
        return {"state": InkState()}
    if level == 6:
        from .student import FilmState
        return {"film": FilmState()}
    return None


def flicker(level: int, base: np.ndarray, n: int = 8, noise: float = 3.0, size=(320, 180)) -> float:
    """нерухома сцена + незалежний шум камери в кожному кадрі: яка частка пікселів змінюється між сусідніми кадрами"""
    rng = np.random.default_rng(3)
    p = Params(method=level)
    aux = _aux(level)
    w, h = size
    prev, rates = None, []
    for _ in range(n):
        f = (base.astype(np.float32) + rng.normal(0, noise, base.shape)).clip(0, 255).astype(np.uint8)
        m = binarize(f, w, h, p, aux)
        if prev is not None:
            rates.append(float((m != prev).mean()))
        prev = m
    return float(np.mean(rates[1:] if len(rates) > 1 else rates))


def ms_per_frame(level: int, img: np.ndarray, w=640, h=360, reps=3) -> float:
    p = Params(method=level)
    aux = _aux(level)
    binarize(img, w, h, p, aux)       # прогрів (компіляція numba; для рівня 6 - вибір бекенду)
    t = time.perf_counter()
    for _ in range(reps):
        binarize(img, w, h, p, aux)
    return (time.perf_counter() - t) / reps * 1000


# ------------------------------------------------------------------ звіт по рівнях
def level_report(images: dict, size=(480, 270), levels=LEVELS) -> dict:
    """-> {назва картинки: {рівень: {метрика: значення}}} (+ контактні маски для таблиці в ключі '_masks')"""
    w, h = size
    res, masks = {}, {}
    for name, img in images.items():
        g = _gray(img, w, h)
        res[name], masks[name] = {}, {}
        for lv in levels:
            m = binarize(img, w, h, Params(method=lv))
            masks[name][lv] = m
            r = {"tone2": tone_rmse(g, m, 2.5), "tone5": tone_rmse(g, m, 5.0), "edge": edge_f1(g, m)}
            if name.startswith("текст"):
                r["text"] = text_iou(img, m, dark_ink=name.endswith("(чорне на білому)"))
            if name in ("градієнт", "текстура", "портрет"):
                r["repeat"] = repeat_score(m)
            res[name][lv] = r
    res["_masks"] = masks
    return res


def contact_sheet(images: dict, masks: dict, path: str, tile=(320, 180), levels=LEVELS):
    """рядок на картинку: джерело | рівні 1..5 (у масштабі tile)"""
    tw, th = tile
    rows = []
    for name, img in images.items():
        cells = [cv2.resize(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (tw, th), interpolation=cv2.INTER_AREA)]
        for lv in levels:
            m = (masks[name][lv] * 255).astype(np.uint8)
            m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST if m.shape[1] <= tw else cv2.INTER_AREA)
            cells.append(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
        row = np.hstack(cells)
        cv2.putText(row, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        rows.append(row)
    sheet = np.vstack(rows)
    ok, buf = cv2.imencode(".png", sheet)
    buf.tofile(path)


# ------------------------------------------------------------------ формати експорту
def format_report(size=(320, 180), frames=24, fps=24, qualities=("Висока", "Середня", "Мала")) -> list[dict]:
    """синтетичний ролик -> кожен формат експорту -> назад: кадри/розмір/fps/звук, «чистота» (частка не чисто чорно-білих
    пікселів після стиснення), середня похибка, розмір файлу. Беззбиткові (GIF, PNG, JSON) мають давати нуль."""
    import subprocess
    import tempfile

    from .video import FORMATS, RobustCapture, export_video, find_ffmpeg, probe_stream
    ff = find_ffmpeg()
    rows = []
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.mp4")
        w, h = size
        base = synthetic_set(w * 2, h * 2)["портрет"]
        if ff:
            cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w * 2}x{h * 2}", "-r", str(fps), "-i", "-",
                   "-f", "lavfi", "-i", f"sine=frequency=440:duration={frames / fps:.3f}", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-shortest", src]
            from .video import hidden_kw
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, **hidden_kw())
            for i in range(frames):
                M = np.float32([[1, 0, i * 3], [0, 1, 0]])
                p.stdin.write(cv2.warpAffine(base, M, (w * 2, h * 2), borderMode=cv2.BORDER_REFLECT).tobytes())
            p.stdin.close()
            p.wait()
        else:
            vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h * 2))
            for i in range(frames):
                M = np.float32([[1, 0, i * 3], [0, 1, 0]])
                vw.write(cv2.warpAffine(base, M, (w * 2, h * 2), borderMode=cv2.BORDER_REFLECT))
            vw.release()
        # еталон: ті самі маски, що піде в кодер (без стиснення)
        ref = os.path.join(d, "ref")
        export_video(src, ref, Params(method=4), "PNG-послідовність", f"{w}x{h}", "Висока", False, 1)
        refs = [_imread_gray(os.path.join(ref, f)) for f in sorted(os.listdir(ref))]
        for fmt, (ext, codec) in FORMATS.items():
            for q in (qualities if codec in ("mp4", "mkv", "mov", "webm", "avi") else qualities[:1]):
                out = os.path.join(d, f"o_{codec}_{q}{ext}")
                t = time.perf_counter()
                try:
                    r = export_video(src, out, Params(method=4), fmt, f"{w}x{h}", q, False, 1)
                except Exception as e:  # noqa: BLE001
                    rows.append(dict(fmt=fmt, q=q, error=str(e)[:80]))
                    continue
                el = time.perf_counter() - t
                row = dict(fmt=fmt, q=q, frames=r["frames"], size=r["size"], kb=r["bytes"] / 1024, sec=el, grey=None, mae=None, audio=None)
                got = []
                if codec == "png":
                    got = [_imread_gray(os.path.join(out, f)) for f in sorted(os.listdir(out))]
                elif codec == "json":
                    from .pack import read_json_video
                    m, fr = read_json_video(out)
                    got = [(x > 0).astype(np.uint8) * 255 for x in fr]
                elif codec == "ikr":
                    from .bake import open_baked
                    rd = open_baked(out)
                    got = [(rd.mask(i) > 0).astype(np.uint8) * 255 for i in range(rd.frames)]
                    rd.close()
                else:
                    cap = RobustCapture(out)
                    while True:
                        ok, f = cap.read()
                        if not ok:
                            break
                        got.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
                    cap.release()
                    row["audio"] = bool(probe_stream(out).get("has_audio")) if ff else None
                n = min(len(got), len(refs))
                if n:
                    a = np.stack(got[:n]).astype(np.int16)
                    b = np.stack(refs[:n]).astype(np.int16)
                    row["grey"] = float(((a > 40) & (a < 215)).mean() * 100)
                    row["mae"] = float(np.abs(a - b).mean())
                    row["frames_ok"] = abs(len(got) - len(refs)) <= 1
                rows.append(row)
    return rows


# ------------------------------------------------------------------ виведення
METRIC_TITLES = (("tone2", "тон σ2.5↓"), ("tone5", "тон σ5↓"), ("edge", "контур↑"), ("text", "текст↑"), ("repeat", "повтор↓"))


def _fmt_levels(res: dict, extra: dict) -> str:
    lines = []
    for name, per in res.items():
        if name.startswith("_"):
            continue
        keys = [k for k, _ in METRIC_TITLES if any(k in per[lv] for lv in per)]
        lines.append(f"\n{name}")
        lines.append("  рівень  " + "  ".join(f"{dict(METRIC_TITLES)[k]:>10s}" for k in keys))
        for lv, r in per.items():
            lines.append(f"    {lv}     " + "  ".join(f"{r[k]:10.3f}" if k in r else " " * 10 for k in keys))
    lines.append("\nМерехтіння нерухомої сцени з шумом (частка змінених пікселів, менше = краще) і швидкість (мс/кадр 640×360, 1 потік):")
    lines.append("  рівень  мерехтіння   мс/кадр")
    for lv in extra["flicker"]:
        lines.append(f"    {lv}      {extra['flicker'][lv]:8.4f}   {extra['ms'].get(lv, float('nan')):8.1f}")
    return "\n".join(lines)


def _fmt_formats(rows: list[dict]) -> str:
    lines = ["\nФормати експорту (кодування -> декодування): чистота й розмір", "  формат               якість    кадрів  розмір      КБ   сірих%    MAE  звук   с"]
    for r in rows:
        if "error" in r:
            lines.append(f"  {r['fmt']:20s} {r['q']:8s}  ПОМИЛКА: {r['error']}")
            continue
        au = "—" if r["audio"] is None else ("так" if r["audio"] else "НІ")
        grey = "—" if r["grey"] is None else f"{r['grey']:6.3f}"
        mae = "—" if r["mae"] is None else f"{r['mae']:6.2f}"
        lines.append(f"  {r['fmt']:20s} {r['q']:8s} {r['frames']:6d}  {r['size'][0]}x{r['size'][1]:<5d} {r['kb']:7.1f}  {grey:>7s} {mae:>6s}  {au:4s} {r['sec']:4.1f}")
    return "\n".join(lines)


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benvid.qc", description="Контрольний тест якості InkReel")
    ap.add_argument("--quick", action="store_true", help="менші картинки, без таблиці швидкості (≈10 с)")
    ap.add_argument("--formats", action="store_true", help="перевірити чистоту й сумісність форматів експорту (потрібен ffmpeg)")
    ap.add_argument("--speed", action="store_true", help="виміряти швидкість рівнів")
    ap.add_argument("--images", help="тека з вашими картинками: рахуємо ті самі метрики й робимо контактну таблицю")
    ap.add_argument("--out", default=".", help="куди писати qc_report.md та qc_sheet.png")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    size = (320, 180) if a.quick else (480, 270)
    imgs = synthetic_set(*size)
    if a.images:
        from .media import IMAGE_EXT, imread_u
        for f in sorted(os.listdir(a.images))[:12]:
            if f.lower().endswith(IMAGE_EXT):
                try:
                    im = imread_u(os.path.join(a.images, f))
                    imgs["▸ " + f] = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
                except Exception as e:  # noqa: BLE001
                    print("пропущено", f, e)
    t0 = time.time()
    res = level_report(imgs, size)
    base = imgs["портрет"]
    extra = {"flicker": {lv: flicker(lv, cv2.resize(base, (320, 180), interpolation=cv2.INTER_AREA), n=6 if a.quick else 8) for lv in LEVELS}, "ms": {}}
    if a.speed or not a.quick:
        big = cv2.resize(base, (640, 360), interpolation=cv2.INTER_CUBIC)
        extra["ms"] = {lv: ms_per_frame(lv, big, reps=2 if a.quick else 3) for lv in LEVELS}
    contact_sheet(imgs, res["_masks"], os.path.join(a.out, "qc_sheet.png"), tile=(260, 146) if a.quick else (320, 180))
    text = "InkReel — контрольний тест якості\n" + _fmt_levels(res, extra)
    if a.formats:
        text += _fmt_formats(format_report(size=(256, 144) if a.quick else (320, 180)))
    text += f"\n\nТаблицю з картинками збережено: {os.path.join(a.out, 'qc_sheet.png')}   (джерело | рівні 1..5)\nЧас: {time.time() - t0:.0f} с\n"
    print(text)
    with open(os.path.join(a.out, "qc_report.md"), "w", encoding="utf-8") as f:
        f.write("```\n" + text + "```\n")
    return 0


if __name__ == "__main__":
    sys.exit(run())
