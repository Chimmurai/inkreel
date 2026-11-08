"""Швидкість ONNX-моделі на DirectML (будь-яка DX12-відеокарта): python tools/bench_dml.py model.onnx [--h 1072 --w 1920 --iters 40 --profile]
Міряє двічі: «з копіюванням» (s.run із numpy, як у простій інтеграції) і «на відеокарті» (входи вже у відеопам'яті, IOBinding - без PCIe).
--profile: топ вузлів за часом за даними самого onnxruntime. Потрібен onnxruntime-directml (середовище .venv_dml: InkReel.bat, пункт 2)."""
import argparse
import json
import sys
import time

import numpy as np
import onnxruntime as ort


def feeds(s, h, w):
    rng = np.random.default_rng(0)
    out = {}
    for i in s.get_inputs():
        dt = np.float16 if "float16" in i.type else np.float32
        shape = [h if n == "h" else w if n == "w" else h // int(n[1:]) if (isinstance(n, str) and n[0] == "h" and n[1:].isdigit()) else w // int(n[1:]) if (isinstance(n, str) and n[0] == "w" and n[1:].isdigit()) else 1 if (isinstance(n, str) or n is None) else n for n in i.shape]
        out[i.name] = np.zeros((1, 5), dt) if i.name == "meta" else rng.random(shape).astype(dt)
    return out


def timeit(fn, iters):
    for _ in range(5):
        fn()
    t0 = time.time()
    for _ in range(iters):
        fn()
    return (time.time() - t0) / iters * 1000


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--h", type=int, default=1072)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args(argv)
    so = ort.SessionOptions()
    so.enable_mem_pattern = False                          # рекомендація для DirectML
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_profiling = a.profile
    s = ort.InferenceSession(a.model, so, providers=[("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
    print("провайдери:", s.get_providers())
    if s.get_providers()[0] != "DmlExecutionProvider":
        print("УВАГА: DirectML недоступний - заміри йдуть на CPU й нічого не кажуть про відеокарту.")
    feed = feeds(s, a.h, a.w)
    try:
        ms = timeit(lambda: s.run(None, feed), a.iters)
    except Exception as e:  # noqa: BLE001
        print("ПОМИЛКА виконання:", str(e)[:600])
        return 1
    print(f"{a.model}: {a.w}x{a.h}  з копіюванням через PCIe: {ms:.1f} мс = {1000 / ms:.0f} к/с")
    try:
        b = s.io_binding()
        for k, v in feed.items():
            b.bind_ortvalue_input(k, ort.OrtValue.ortvalue_from_numpy(v, "dml", 0))
        for o in s.get_outputs():
            b.bind_output(o.name, "dml", 0)
        def go():
            s.run_with_iobinding(b)
            try:
                b.synchronize_outputs()                    # DirectML асинхронний: без цього міряється лише запуск, а не робота відеокарти
            except Exception:  # noqa: BLE001
                b.copy_outputs_to_cpu()
        ms2 = timeit(go, a.iters)
        print(f"{a.model}: {a.w}x{a.h}  лише відеокарта (без PCIe, з очікуванням завершення): {ms2:.1f} мс = {1000 / ms2:.0f} к/с")
    except Exception as e:  # noqa: BLE001
        print("замір без копіювання недоступний:", str(e)[:200])
    if a.profile:
        path = s.end_profiling()
        ev = json.load(open(path, encoding="utf-8"))
        agg = {}
        for e in ev:
            if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
                k = (e["args"].get("op_name", "?"), e["name"][:-12])
                agg[k] = agg.get(k, 0) + e.get("dur", 0)
        by_op = {}
        for (op, _), d in agg.items():
            by_op[op] = by_op.get(op, 0) + d
        tot = sum(by_op.values()) or 1
        n_runs = a.iters + 5 + 5
        print(f"--- за типами операцій (усього {tot / n_runs / 1000:.1f} мс на запуск за даними профайлера) ---")
        for op, d in sorted(by_op.items(), key=lambda x: -x[1])[:12]:
            print(f"  {op:<22} {d / n_runs / 1000:7.2f} мс  {100 * d / tot:5.1f}%")
        print("--- найдорожчі вузли ---")
        for (op, nm), d in sorted(agg.items(), key=lambda x: -x[1])[:10]:
            print(f"  {d / n_runs / 1000:7.2f} мс  {op:<14} {nm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
