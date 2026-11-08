"""Порівняння бекендів нейромоделі IRNM1 у ПОТОЧНОМУ середовищі: що є в onnxruntime (DirectML / CUDA / TensorRT / процесор), мс на кадр,
час першого кадру (TensorRT будує рушій) і збіг маски з процесором. Запускати в різних середовищах (.venv_dml, .venv_trt) і дивитись разом:
    python tools/bench_backends.py [--label dml]            -> reports/backends_<мітка>.json
    python tools/bench_backends.py --merge                   -> таблиця з усіх reports/backends_*.json"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SIZES = ((720, 1280), (1080, 1920), (1440, 3440))


def _dlls():
    """шляхи до бібліотек CUDA/cuDNN/TensorRT, поставлених pip-пакетами (nvidia-*, tensorrt_libs): без них onnxruntime-gpu не бачить CUDA"""
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:  # noqa: BLE001
        pass
    if os.name != "nt":
        return
    import site
    for base in site.getsitepackages():
        for sub in ("nvidia", "tensorrt_libs", "tensorrt"):
            d = os.path.join(base, sub)
            if os.path.isdir(d):
                for r, _, files in os.walk(d):
                    if any(f.lower().endswith(".dll") for f in files):
                        try:
                            os.add_dll_directory(r)
                            os.environ["PATH"] = r + os.pathsep + os.environ.get("PATH", "")
                        except OSError:
                            pass


def run(label: str) -> dict:
    _dlls()
    import numpy as np
    from benvid.student import runtime as R
    import onnxruntime as ort
    res = {"label": label, "ort": ort.__version__, "providers": R.providers_available(), "backends": {}}
    print(f"[{label}] onnxruntime {ort.__version__}, провайдери: {res['providers']}", flush=True)
    rng = np.random.default_rng(3)
    ref_img = np.clip(rng.random((200, 264, 1)) * 200 + rng.random((200, 264, 3)) * 55, 0, 255).astype(np.uint8)
    ref = R.Student("cpu").binarize(ref_img)
    for k in ("cpu", "dml", "cuda", "trt"):
        if k not in res["providers"]:
            continue
        r = res["backends"][k] = {}
        try:
            t0 = time.perf_counter()
            s = R.Student(k)
            r["load_s"] = round(time.perf_counter() - t0, 2)
            r["agree"] = round(float((s.binarize(ref_img) == ref).mean()), 4)
            for h, w in SIZES:
                img = (np.random.default_rng(1).random((h, w, 3)) * 255).astype(np.uint8)
                t0 = time.perf_counter()
                s.binarize(img)
                first = time.perf_counter() - t0
                ms = R._bench(s, w, h, reps=6 if k != "cpu" else 2, warm=2 if k != "cpu" else 1)
                r[f"{h}p"] = {"ms": round(ms, 1), "first_s": round(first, 2)}
                print(f"  {k:<5} {w}x{h}: {ms:7.1f} мс на кадр (перший кадр {first:.1f} с)", flush=True)
        except Exception as e:  # noqa: BLE001
            r["error"] = str(e)[:300]
            print(f"  {k}: помилка: {e}", flush=True)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", f"backends_{label}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def merge() -> str:
    rows, names = {}, []
    for p in sorted(glob.glob(os.path.join(ROOT, "reports", "backends_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        for k, v in d["backends"].items():
            name = f"{k} ({d['label']})"
            names.append(name)
            rows[name] = v
    lines = ["| бекенд | збіг з CPU | 720p, мс | 1080p, мс | 1440×3440, мс | перший кадр 1080p, с |", "|---|---|---|---|---|---|"]
    for n in names:
        v = rows[n]
        if "error" in v:
            lines.append(f"| {n} | помилка: {v['error'][:60]} | | | | |")
            continue
        g = lambda s: v.get(s, {}).get("ms", "")  # noqa: E731
        lines.append(f"| {n} | {v.get('agree', '')} | {g('720p')} | {g('1080p')} | {g('1440p')} | {v.get('1080p', {}).get('first_s', '')} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="env")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        t = merge()
        print(t)
        open(os.path.join(ROOT, "reports", "backends.md"), "w", encoding="utf-8").write(t + "\n")
    else:
        run(a.label)
