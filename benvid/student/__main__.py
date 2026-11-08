"""python -m benvid.student [--probe] [--bench H W] [--stages H W] [--par H W] [--image файл] — перевірка студента й вибору бекенду"""
import argparse
import json
import os
import sys
import time

import numpy as np

from . import runtime as R


def stages(s, h, w):
    import cv2
    from . import inputs as I
    rng = np.random.default_rng(1)
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    acc = {}

    def T(name, fn, n=6):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            r = fn()
        acc[name] = (time.perf_counter() - t0) / n * 1000
        return r
    bgr = T("to_bgr", lambda: I.to_bgr(img))
    T("small_of", lambda: I.small_of(bgr))
    film, _ = s.film(bgr, h)
    T("film (global, cpu)", lambda: s.film(bgr, h))
    ph, pw = (-h) % 16, (-w) % 16
    T("pad", lambda: cv2.copyMakeBorder(bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT_101))
    T("mask (gpu+copies)", lambda: s.mask(bgr, film))
    T("packbits", lambda: np.packbits(s.mask(bgr, film), axis=-1), 1)
    for k, v in acc.items():
        print(f"  {k:<22}{v:7.1f} мс")
    print(f"  разом (послідовно)    {sum(v for k, v in acc.items() if k != 'packbits'):7.1f} мс")
    try:
        import onnxruntime as ort
        print("  провайдери сеансу тіла (процесор там завжди про запас, це не значить, що вузли на ньому):", s.body.get_providers())
    except Exception:
        pass


def parallel(backend, h, w, secs=6.0):
    """пропускна здатність при кількох потоках (як потоки подачі в застосунку): один сеанс під замком проти двох-трьох сеансів
    і проти «відеокарта + процесор разом»; у кожного рядка - к/с загалом, по бекендах і завантаження процесора"""
    import os
    import threading
    img = (np.random.default_rng(1).random((h, w, 3)) * 255).astype(np.uint8)
    ncpu = os.cpu_count() or 1

    def run(title, backends, threads):
        insts = [R.Student(b) for b in backends]
        for st in insts:
            st.binarize(img)
        done = {i: 0 for i in range(len(insts))}
        end = time.perf_counter() + secs
        lk = threading.Lock()
        busy = [0] * len(insts)

        def work(k):
            i = k % len(insts)                            # потік закріплений за сеансом
            while time.perf_counter() < end:
                insts[i].binarize(img)
                with lk:
                    done[i] += 1
        c0, t0 = time.process_time(), time.perf_counter()
        ts = [threading.Thread(target=work, args=(k,)) for k in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        dt = time.perf_counter() - t0
        cpu = (time.process_time() - c0) / dt / ncpu * 100
        tot = sum(done.values()) / dt
        per = ", ".join(f"{backends[i]}#{i} {done[i] / dt:.1f}" for i in range(len(insts)))
        print(f"  {title:<34}{tot:6.1f} к/с  [{per}]  процесор ≈{cpu:.0f}%")
        del insts

    print(f"{w}x{h}, по {secs:.0f} с на варіант:")
    run(f"{backend}×1, 1 потік", [backend], 1)
    run(f"{backend}×1, 4 потоки (зараз)", [backend], 4)
    run(f"{backend}×2 сеанси, 4 потоки", [backend] * 2, 4)
    run(f"{backend}×3 сеанси, 6 потоків", [backend] * 3, 6)
    if backend != "cpu":
        run(f"{backend}×1 + cpu×1, 4 потоки", [backend, "cpu"], 4)
        run(f"{backend}×2 + cpu×1, 6 потоків", [backend, backend, "cpu"], 6)


def profile(backend: str, h: int, w: int, n: int = 30):
    """час кожного вузла моделі на вибраному бекенді (профайлер ORT): куди саме йдуть мілісекунди, і чи є вузли поза відеокартою"""
    import collections
    rng = np.random.default_rng(1)
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    s = R.Student(backend, profile=True)
    for _ in range(4):
        s.binarize(img)
    t0 = time.perf_counter()
    for _ in range(n):
        s.binarize(img)
    wall = (time.perf_counter() - t0) / n * 1000
    path = s.body.end_profiling()
    ev = json.load(open(path, encoding="utf-8"))
    nodes = [e for e in ev if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")]
    runs = [e for e in ev if e.get("cat") == "Session" and e.get("name") == "model_run"]
    per_op, per_node, per_prov = collections.Counter(), collections.Counter(), collections.Counter()
    for e in nodes:
        a = e.get("args", {})
        per_op[a.get("op_name", "?")] += e["dur"]
        per_node[e["name"][:-12]] += e["dur"]
        per_prov[a.get("provider", "?")] += e["dur"]
    k = max(1, len(runs))                                      # у файлі і прогрів, і виміряні кадри: ділимо на всі запуски
    run_ms = sum(e["dur"] for e in runs) / k / 1000
    print(f"{w}x{h}, {n} кадрів на {backend}: кадр цілком {wall:.1f} мс, у сеансі моделі {run_ms:.1f} мс, сума вузлів {sum(per_op.values()) / k / 1000:.1f} мс")
    print("  за провайдером, мс на кадр:", {p: round(v / k / 1000, 2) for p, v in per_prov.items()})
    print("  за типом вузла, мс на кадр:")
    for op, v in per_op.most_common(8):
        print(f"    {op:<14}{v / k / 1000:7.2f}")
    print("  найдовші вузли, мс на кадр:")
    for nm, v in per_node.most_common(12):
        print(f"    {nm[:48]:<50}{v / k / 1000:7.2f}")
    try:
        os.remove(path)
    except OSError:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="заново вибрати бекенд (ігнорує кеш)")
    ap.add_argument("--bench", nargs=2, type=int, metavar=("H", "W"))
    ap.add_argument("--stages", nargs=2, type=int, metavar=("H", "W"), help="розкласти час кадра по стадіях на вибраному бекенді")
    ap.add_argument("--par", nargs=2, type=int, metavar=("H", "W"), help="к/с при кількох потоках і сеансах, у тому числі відеокарта + процесор разом")
    ap.add_argument("--prof", nargs=2, type=int, metavar=("H", "W"), help="час кожного вузла моделі (профайлер ORT)")
    ap.add_argument("--image")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")           # у файл (student_check.txt) без cp1251: «×» і «≈» падали з UnicodeEncodeError
    except Exception:  # noqa: BLE001
        pass
    print("провайдери ORT:", R.providers_available(), " режим:", R.mode(), " модель:", R.meta())
    if a.probe:
        d = R.probe()
        print("вибір бекенду:", json.dumps(d, ensure_ascii=False))
    s = R.get()
    if s is None:
        print("студент недоступний:", R.info()["reason"])
        return 1
    i = R.info()
    print("бекенд:", s.backend, f"({R.label(s.backend)})", i["ms"])
    for k, v in (i.get("tried") or {}).items():
        print(f"  сходинка {R.LABEL.get(k, k):<9} {v}")
    if a.bench:
        h, w = a.bench
        print(f"{w}x{h}: {R._bench(s, w, h):.1f} мс на кадр")
    if a.stages:
        stages(s, *a.stages)
    if a.par:
        parallel(s.backend, *a.par)
    if a.prof:
        profile(s.backend, *a.prof)
    if a.image:
        import cv2
        img = cv2.imdecode(np.fromfile(a.image, np.uint8), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        m = s.binarize(img)
        print(f"{img.shape[1]}x{img.shape[0]}: {(time.perf_counter() - t0) * 1000:.0f} мс")
        out = a.out or a.image.rsplit(".", 1)[0] + "_student.png"
        cv2.imwrite(out, m * 255)
        print("записано", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
