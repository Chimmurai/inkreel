"""Ставить навчений прогін у застосунок: експортує дві частини (fp32 і fp16, маска uint8 прямо з відеокарти), додає вхід uint8 BGR і кладе в
benvid/assets/student/ (global.onnx, bodyu8_fp32.onnx, bodyu8_fp16.onnx, meta.json). Застосунок сам заново виб'є бекенд (кеш прив'язаний до sha).

python tools/install_student.py ink2 [--data E:\\Inkreel_Dataset] [--note "текст"]
Перед заміною перевіряє: маска uint8-тіла (CPU) збігається з PyTorch на випадковому кадрі не гірше за 99,5 %."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ASSETS = os.path.join(os.path.dirname(HERE), "benvid", "assets", "student")


def sha_of(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def main(argv=None) -> int:
    import numpy as np
    import onnxruntime as ort
    import torch
    import train_ink as T
    from make_u8_body import add_u8_front

    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", default=T.DEF_DATA)
    ap.add_argument("--note", default="")
    ap.add_argument("--assets", default=ASSETS)
    ap.add_argument("--min-match", type=float, default=0.995)
    a = ap.parse_args(argv)
    run = a.run if (os.path.isdir(a.run) or os.path.isfile(a.run)) else os.path.join(a.data, "runs", a.run)
    model, ck = T.load_model(run, "ema", torch.device("cpu"))
    model.float()
    tmp = tempfile.mkdtemp(prefix="inkstudent_")
    out = {}
    for tag, extra in (("fp32", []), ("fp16", ["--fp16"])):
        base = os.path.join(tmp, f"body_{tag}.onnx")
        T.main(["export", "--run", run, "--parts", "--mask", "--out", base, *extra])
        out[tag] = os.path.join(tmp, f"bodyu8_{tag}.onnx")
        add_u8_front(base, out[tag])
    gpath = os.path.join(tmp, "body_fp32_global.onnx")
    # проста перевірка на випадковому кадрі: fp32 uint8-тіло на CPU проти PyTorch
    h, w = 128, 192
    rng = np.random.default_rng(3)
    bgr = (rng.random((1, h, w, 3)) * 255).astype(np.uint8)
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    ins = T.example_inputs(h, w, "cpu")
    x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))
    sg = ort.InferenceSession(gpath, providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(out["fp32"], providers=["CPUExecutionProvider"])
    fo = sg.run(None, dict(small=ins[2].numpy(), meta=ins[3].numpy()))
    names = [o.name for o in sg.get_outputs()]
    ofp = os.path.join(tmp, "body_fp32_of.onnx")
    extra = {}
    if os.path.isfile(ofp):                                           # ir1: внесок орієнтованого банка рахується окремо (один раз на розмір кадру)
        extra["ofeat"] = ort.InferenceSession(ofp, providers=["CPUExecutionProvider"]).run(None, {"noise": ins[1].numpy()})[0]
    r = sb.run(None, {"bgr": bgr, "noise": ins[1].numpy(), **extra, **dict(zip(names, fo))})[0]
    with torch.no_grad():
        ref = (model(x, ins[1], ins[2], ins[3]).numpy() > 0)
    match = float((r.astype(bool) == ref).mean())
    print(f"збіг маски uint8-тіла з PyTorch: {match * 100:.3f}%")
    if match < a.min_match:
        print("НЕ ВСТАНОВЛЕНО: збіг нижчий за допустимий")
        return 1
    os.makedirs(a.assets, exist_ok=True)
    prev = os.path.join(a.assets, "meta.json")
    if os.path.isfile(prev):                                          # попередню модель не губимо: копія в weights/student_backup (поза збіркою)
        try:
            om = json.load(open(prev, encoding="utf-8"))
            dst = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(a.assets)))), "weights", "student_backup",
                               f"{om.get('run', 'old')}_{om.get('preset', '')}_{om.get('step', '')}")
            if not os.path.isdir(dst):
                shutil.copytree(a.assets, dst)
                print("попередню модель збережено в", dst)
        except (OSError, ValueError):
            pass
    shutil.copy(gpath, os.path.join(a.assets, "global.onnx"))
    shutil.copy(out["fp32"], os.path.join(a.assets, "bodyu8_fp32.onnx"))
    shutil.copy(out["fp16"], os.path.join(a.assets, "bodyu8_fp16.onnx"))
    ofa = os.path.join(a.assets, "of_fp32.onnx")
    if os.path.isfile(ofp):
        shutil.copy(ofp, ofa)
    elif os.path.isfile(ofa):                                         # модель без орієнтованого банка: стару копію не лишаємо
        os.remove(ofa)
    files = [os.path.join(a.assets, n) for n in ("global.onnx", "bodyu8_fp32.onnx", "bodyu8_fp16.onnx", "of_fp32.onnx") if os.path.isfile(os.path.join(a.assets, n))]
    old = {}
    try:
        old = json.load(open(os.path.join(a.assets, "meta.json"), encoding="utf-8"))
    except (OSError, ValueError):
        pass
    meta = dict(preset=ck["args"].get("preset"), step=ck.get("step"), run=os.path.basename(os.path.dirname(os.path.abspath(run)) if os.path.isfile(run) else os.path.normpath(run)), sha=sha_of(files),
                teacher=old.get("teacher", "ink5.ultra (method=5)"), min_h=old.get("min_h", 48),
                note=a.note or "best.pt EMA; студент наближає Ультра, не дорівнює йому")
    json.dump(meta, open(os.path.join(a.assets, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print("встановлено:", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
