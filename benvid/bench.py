"""Заміри швидкості InkReel на ЦЬОМУ ПК:   python -m benvid.bench [--quick] [--out ТЕКА]

Нічого не потребує: тестове відео 1080p створюється саме. Звіт друкується й пишеться в bench_report.md. Заміряє:
  1. вартість одного кадру кожного рівня в одному потоці (для порівняння між ПК і версіями);
  2. швидкість плеєра (кадрів/с, якими він накопичує буфер) на рівнях 3-4 при різній кількості процесів;
  3. «Ультру» (5) при різних налаштуваннях потоків: скільки кадрів рахується одночасно й скільки потоків numba дають кожному;
  4. декодування 1080p, зміну розміру й кодування (програмне x264 та апаратне, якщо є).
З цих чисел видно, чи вистачає процесора для реального часу (30 к/с) і де вузьке місце."""
from __future__ import annotations

import argparse
import multiprocessing
import os
import platform
import subprocess
import tempfile
import time

import cv2
import numpy as np

from .core import InkState, Params, binarize

CLIP_W, CLIP_H, CLIP_N, CLIP_FPS = 1920, 1080, 480, 24


def make_clip(path: str):
    """1080p із рухом і шумом камери: схоже на справжнє відео (панорама по складній картинці)"""
    from . import qc
    base = qc.synthetic_set(CLIP_W + 160, CLIP_H + 90)["портрет"]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (CLIP_W, CLIP_H))
    rng = np.random.default_rng(5)
    for i in range(CLIP_N):
        ph = (i * 1.5) % 320                              # панорама туди й назад: кадр завжди лишається в межах картинки
        x = int(ph if ph <= 160 else 320 - ph)
        y = int(x * 90 / 160)
        f = base[y:y + CLIP_H, x:x + CLIP_W].copy()
        f = np.clip(f.astype(np.int16) + rng.normal(0, 3, f.shape).astype(np.int16), 0, 255).astype(np.uint8)
        vw.write(f)
    vw.release()
    return path


def _sample(w, h):
    from . import qc
    img = qc.synthetic_set(w, h)["портрет"]
    rng = np.random.default_rng(1)
    return np.clip(img.astype(np.int16) + rng.normal(0, 4, img.shape).astype(np.int16), 0, 255).astype(np.uint8)


def kernel_ms(levels=(1, 2, 3, 4, 5), sizes=((640, 360), (1280, 720), (1920, 1080)), reps=2) -> dict:
    """мс на кадр, один потік (numba й cv2 - по одному потоку, як у воркері)"""
    old = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        import numba
        old_nb = numba.get_num_threads()
        numba.set_num_threads(1)
    except Exception:  # noqa: BLE001
        numba, old_nb = None, None
    out = {}
    try:
        for (w, h) in sizes:
            img = _sample(w, h)
            for lv in levels:
                p = Params(method=lv)
                aux = {"state": InkState()} if lv == 5 else None
                binarize(img, w, h, p, aux)
                t = time.perf_counter()
                for _ in range(reps):
                    binarize(img, w, h, p, aux)
                out[(lv, h)] = (time.perf_counter() - t) / reps * 1000
    finally:
        cv2.setNumThreads(old)
        if numba is not None and old_nb:
            numba.set_num_threads(old_nb)
    return out


def cpu_busy():
    """зайнятість усіх логічних ядер між двома викликами (0..1); перший виклик повертає None і запам'ятовує відлік"""
    global _CPU_PREV
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes as wt
            i, k, u = wt.FILETIME(), wt.FILETIME(), wt.FILETIME()
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
            q = lambda f: (f.dwHighDateTime << 32) | f.dwLowDateTime
            idle, tot = q(i), q(k) + q(u)               # kernel-час уже включає простій
        else:
            with open("/proc/stat") as f:
                v = [int(x) for x in f.readline().split()[1:]]
            idle, tot = v[3] + (v[4] if len(v) > 4 else 0), sum(v)
    except Exception:  # noqa: BLE001
        return None
    prev, _CPU_PREV = _CPU_PREV, (idle, tot)
    if prev is None or tot <= prev[1]:
        return None
    return 1.0 - (idle - prev[0]) / (tot - prev[1])


_CPU_PREV = None


def player_fps(path, size_key, method, frames=None, timeout=240, **kw):
    """-> (к/с, зайнятість процесора 0..1) у сталому режимі. Плеєр рахує наперед усе відео, тож міряємо швидкість наповнення буфера
    між 25 % і 85 % кліпу: перша чверть - розгін (старт процесів, завантаження ядер numba), хвіст - спад, коли роботи вже мало"""
    from .player import Player
    pl = Player(**kw)
    try:
        pl.params = Params(method=method)
        pl.open(path, size_key)
        pl.seek(0)
        n = pl.n
        lo, hi = int(n * 0.25), int(n * 0.85)
        end = time.time() + timeout
        t_lo = t_hi = None
        while time.time() < end:
            pl.update()
            c = pl.cached
            now = time.perf_counter()
            if t_lo is None and c >= lo:
                t_lo, c_lo = now, c
                cpu_busy()
            if t_lo is not None and c >= hi:
                t_hi, c_hi = now, c
                break
            time.sleep(0.01)
        if t_lo is None or t_hi is None:
            return 0.0, None
        return (c_hi - c_lo) / max(t_hi - t_lo, 1e-6), cpu_busy()
    finally:
        pl.close()


def decode_ms(path) -> tuple[float, float]:
    from .video import open_capture
    cap = open_capture(path)
    fr = []
    t = time.perf_counter()
    for _ in range(48):
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    dec = (time.perf_counter() - t) / max(1, len(fr)) * 1000
    t = time.perf_counter()
    for f in fr:
        cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA)
    rs = (time.perf_counter() - t) / max(1, len(fr)) * 1000
    cap.release() if hasattr(cap, "release") else None
    return dec, rs


def encode_fps(kind: str, w=1280, h=720, n=120) -> float | None:
    """к/с кодування 1-бітного кадру: kind = 'x264' або 'nvenc'/'amf'/'qsv'"""
    from .video import _hw_args, _x264_args, find_ffmpeg
    ff = find_ffmpeg()
    if not ff:
        return None
    img = _sample(w, h)
    mask = (binarize(img, w, h, Params(method=3)) * 255).astype(np.uint8).tobytes()
    v = ["-c:v", "libx264", *_x264_args(14)] if kind == "x264" else _hw_args(kind, 16)
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", "30", "-i", "-",
           *v, "-pix_fmt", "yuv420p", "-f", "null", "-"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    t = time.perf_counter()
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, creationflags=flags)
        for _ in range(n):
            p.stdin.write(mask)
        p.stdin.close()
        ok = p.wait() == 0
    except Exception:  # noqa: BLE001
        return None
    return n / (time.perf_counter() - t) if ok else None


def _row(cells, widths):
    return "| " + " | ".join(str(c).ljust(wd) for c, wd in zip(cells, widths)) + " |"


def table(head, rows):
    widths = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    out = [_row(head, widths), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [_row(r, widths) for r in rows]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m benvid.bench", description="Заміри швидкості InkReel")
    ap.add_argument("--quick", action="store_true", help="менше конфігурацій (≈1 хв)")
    ap.add_argument("--out", default=".", help="куди писати bench_report.md")
    a = ap.parse_args(argv)

    from .video import default_workers, detect_hw_encoder, find_ffmpeg
    cpu = os.cpu_count() or 2
    W = default_workers()
    lines = [f"# InkReel — заміри швидкості ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    try:
        import numba
        nbv = numba.__version__
    except Exception:  # noqa: BLE001
        nbv = "нема"
    ram = 0
    try:
        from .video import ram_bytes
        ram = ram_bytes() >> 30
    except Exception:  # noqa: BLE001
        pass
    lines += [f"- ПК: {platform.platform()}, логічних ядер {cpu}, ОЗП ≈ {ram} ГБ, воркерів за замовчуванням {W}",
              f"- python {platform.python_version()}, numba {nbv}, opencv {cv2.__version__}, ffmpeg: {find_ffmpeg() or 'не знайдено'}"]
    hw = detect_hw_encoder() or "немає"
    lines += [f"- апаратне кодування H.264: {hw}; OpenCL у OpenCV: {'так' if cv2.ocl.haveOpenCL() else 'ні'}", ""]
    print("\n".join(lines), flush=True)

    # 0. відеокарта (OpenCL)
    print("0/4 відеокарта…", flush=True)
    from . import gpu
    gi = gpu.list_devices(allow_cpu=gpu.mode() == "cpu")
    if not gi:
        lines += ["## 0. Відеокарта (OpenCL)", "", "OpenCL-відеокарти не знайдено" + ("" if gpu.HAVE_CL else " (pyopencl не встановлено)") + ": усе рахується на процесорі.", ""]
    else:
        r = gpu.run_probe()
        rows = [[k, f"{r['ms'][k][0]:.1f}", f"{r['ms'][k][1]:.1f}", "так" if r["use"].get(k) else ("ПОМИЛКА: " + r["err"][k] if k in r["err"] else "ні")]
                for k in gpu.KERNELS if k in r["ms"] or k in r["err"]]
        lines += [f"## 0. Відеокарта (OpenCL): {r['name'] or '—'}", "",
                  "Ядра на кадрі 720p, мс: numba в одному потоці проти відеокарти (разом із копіюванням даних). Вмикаються лише ті, що збігаються з numba й швидші.", "",
                  table(["ядро", "CPU, мс", "GPU, мс", "використовується"], rows), ""]
        frm = []
        for lv in (4, 5):
            res = {}
            for tag, md in (("CPU", "off"), ("GPU", "auto")):
                os.environ["INKREEL_GPU"] = md
                gpu._dev, gpu._state["use"] = None, None
                res[tag] = kernel_ms(levels=(lv,), sizes=((1280, 720),), reps=2)[(lv, 720)]
            frm.append([f"{lv}", f"{res['CPU']:.0f}", f"{res['GPU']:.0f}", f"×{res['CPU'] / max(res['GPU'], 1e-6):.2f}"])
        os.environ.pop("INKREEL_GPU", None)
        gpu._dev, gpu._state["use"] = None, None
        lines += ["Повний кадр 720p, один потік:", "", table(["рівень", "CPU, мс", "GPU+CPU, мс", "виграш"], frm), ""]
    print("\n".join(lines[-6:]), flush=True)

    tmp = tempfile.mkdtemp(prefix="inkreel_bench_")
    clip = make_clip(os.path.join(tmp, "bench1080.mp4"))

    # 1. один потік
    print("1/4 вартість кадру в одному потоці…", flush=True)
    sizes = ((640, 360), (1280, 720)) if a.quick else ((640, 360), (1280, 720), (1920, 1080))
    km = kernel_ms(sizes=sizes)
    rows = [[f"{h}p"] + [f"{km[(lv, h)]:.1f}" for lv in range(1, 6)] for (_, h) in sizes]
    lines += ["## 1. Один потік: мс на кадр", "", table(["розмір", "1 Ескіз", "2 Контур", "3 Півтон", "4 Малюнок", "5 Гравюра"], rows), ""]
    print(lines[-2], flush=True)

    # 2. пул процесів
    print("2/4 плеєр: накопичення буфера, рівні 3-5…", flush=True)
    counts = sorted({W} if a.quick else {1, max(1, W // 2), W})
    rows = []
    for lv in ((3, 5) if a.quick else (3, 4, 5)):
        for key in (("720p",) if a.quick else ("720p", "1080p")):
            cells = []
            os.environ["INKREEL_HOST"] = "0"                                  # пул процесів
            for n in counts:
                if n == 1 and (lv == 5 or key != "720p"):                 # один процес на довгому кліпі - хвилини, нічого нового не дає
                    cells.append("—")
                    continue
                fps, load = player_fps(clip, key, lv, workers=n)
                cells.append(f"{fps:.0f} к/с, ЦП {load * 100:.0f}%" if load is not None else f"{fps:.0f} к/с")
            os.environ["INKREEL_HOST"] = "1"                                  # господар: W потоків в одному процесі, одна відеокарта
            fps, load = player_fps(clip, key, lv, workers=W)
            cells.append(f"{fps:.0f} к/с, ЦП {load * 100:.0f}%" if load is not None else f"{fps:.0f} к/с")
            rows.append([f"{lv}", key] + cells)
    os.environ.pop("INKREEL_HOST", None)
    lines += ["## 2. Плеєр: к/с накопичення буфера і завантаження процесора (пул процесів проти потоків в одному процесі)", "",
              table(["рівень", "розмір"] + [f"{n} проц." for n in counts] + [f"{W} потоків, 1 процес (так працює програма)"], rows), ""]
    lines += ["Якщо при більшій кількості процесів к/с не росте, а «ЦП» нижче ~90 % — вузьке місце не в обчисленнях, а в подачі кадрів (декодування, копіювання, головний процес).", ""]
    print("\n".join(lines[-5:]), flush=True)

    # 3. чистий пул: масштабування обчислень без плеєра
    print("3/4 масштабування пулу (без декодування й плеєра)…", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    from .video import _bench_job, pool_kwargs
    rows = []
    for lv, hh in (((3, 720), (5, 720)) if a.quick else ((3, 720), (4, 720), (5, 720), (5, 1080))):
        ww = hh * 16 // 9
        solo = None
        for n in sorted({1, max(1, W // 2), W}):
            cnt = 4 if lv == 5 else 12
            with ProcessPoolExecutor(n, **pool_kwargs()) as ex:
                list(ex.map(_bench_job, [(ww, hh, lv, 2)] * n))                       # розгін усіх процесів
                cpu_busy()
                t0 = time.perf_counter()
                res = list(ex.map(_bench_job, [(ww, hh, lv, cnt)] * (n * 3)))
                dt = time.perf_counter() - t0
                load = cpu_busy()
            ms = sum(r[0] for r in res) / len(res)
            if n == 1:
                solo = ms
            rows.append([f"{lv}", f"{hh}p", n, f"{n * 3 * cnt / dt:.0f}", f"{ms:.0f}", f"×{ms / solo:.2f}" if solo else "—",
                         f"{load * 100:.0f}%" if load is not None else "—", f"{res[0][1]} / {res[0][2]}"])
    lines += ["## 3. Пул без плеєра: чисті обчислення", "",
              table(["рівень", "розмір", "процесів", "к/с", "мс на кадр у процесі", "повільніше за 1 проц.", "ЦП", "потоків numba / шар"], rows), "",
              "Якщо «мс на кадр» при багатьох процесах у кілька разів більше, ніж при одному, — процесори (або пам'ять) не вивозять паралельну роботу; "
              "якщо ні, а к/с у плеєрі (розділ 2) нижчі, — вузьке місце в подачі кадрів.", ""]
    print("\n".join(lines[-6:]), flush=True)

    # 3б. мікротести: що саме не масштабується
    print("3б/4 мікротести пулу…", flush=True)
    from .video import _bench_micro
    rows = []
    for kind, name, unit in (("cpu", "числа в кеші (чистий ЦП)", "млн ітер/с"), ("mem", "потік через ОЗП", "ГБ/с"),
                             ("alloc", "нові великі масиви (сторінки пам'яті)", "ГБ/с")):
        base, cells = None, []
        for n in sorted({1, max(1, W // 2), W}):
            with ProcessPoolExecutor(n, **pool_kwargs()) as ex:
                list(ex.map(_bench_micro, [(kind, 0.3)] * n))
                tot = sum(ex.map(_bench_micro, [(kind, 1.0)] * n))
            base = base or tot
            cells.append(f"{tot:.1f} {unit} (×{tot / base:.1f})")
        rows.append([name] + cells)
    lines += ["## 3б. Мікротести: сумарна швидкість усіх процесів (у дужках — у скільки разів більше, ніж один процес)", "",
              table(["тест"] + [f"{n} проц." for n in sorted({1, max(1, W // 2), W})], rows), "",
              "Ідеал для 27 процесів на 20 ядрах (8 потужних + 12 економних) — приблизно ×15–20. Якщо «числа в кеші» ростуть добре, а «потік через ОЗП» "
              "чи «нові масиви» застрягають на ×3–5 — вузьке місце пам'ять, а не процесор.", ""]
    print("\n".join(lines[-6:]), flush=True)

    # 4. декодування й кодування
    print("4/4 декодування й кодування…", flush=True)
    dec, rs = decode_ms(clip)
    rows = [["декодування 1080p (кадр)", f"{dec:.1f} мс"], ["зміна розміру 1080p → 720p", f"{rs:.1f} мс"]]
    x = encode_fps("x264")
    rows.append(["x264 (програмне, 720p)", f"{x:.0f} к/с" if x else "—"])
    if hw != "немає":
        h = encode_fps(hw)
        rows.append([f"{hw} (апаратне, 720p)", f"{h:.0f} к/с" if h else "—"])
    lines += ["## 4. Декодування й кодування", "", table(["що", "швидкість"], rows), ""]
    print(lines[-2], flush=True)

    path = os.path.join(a.out, "bench_report.md")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nЗвіт: {os.path.abspath(path)}")
    except OSError as e:
        print("не вдалося записати звіт:", e)
    try:
        os.remove(clip)
        os.rmdir(tmp)
    except OSError:
        pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
