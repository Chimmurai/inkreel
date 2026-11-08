"""навчання студента для «Ультри» (tools/train_ink.py): шумові канали збігаються з алгоритмом, дані, модель, цикл навчання.
Потрібен PyTorch (без нього файл пропускається: у середовищі самого застосунку його нема)."""
import json
import os
import sys

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import train_ink as T  # noqa: E402

from benvid import core as C  # noqa: E402
from benvid import ink5  # noqa: E402


@pytest.mark.parametrize("w,h", [(333, 257), (97, 64), (1280, 720), (700, 1100)])
def test_noise_windows_equal_the_algorithms_static_fields(w, h):
    cg, cb = T.grain_cells(h)
    assert (cg, cb) == (C.grain_cell(h), max(1, int(C.pat_k(h / C.REF_H) + 0.5)))
    tg, td, tr = ink5._static_fields(w, h, cg, cb)
    y0, x0 = h // 3, w // 5
    ch, cw = min(90, h - y0), min(120, w - x0)
    n = T.noise_window(w, h, y0, x0, ch, cw, None)
    assert np.allclose(n[0], tg[y0:y0 + ch, x0:x0 + cw] * 2 - 1)
    assert np.allclose(n[1], td[y0:y0 + ch, x0:x0 + cw] * 2 - 1)
    assert np.allclose(n[2], tr[y0:y0 + ch, x0:x0 + cw] * 2 - 1)
    assert not n[3].any()                                            # без кільцевого кешу - нулі


def test_constants_and_ring_noise_match_ultra():
    assert (T.PERIOD, T.RING_Q, T.RING_SEED) == (ink5.P["period"], ink5.P["q"], 5)
    assert T.TILE == ink5._TILE and np.array_equal(T.blue(), C._blue()) and np.array_equal(T.blobs(), C._blobs())
    for h in (48, 500, 720, 1080, 2160):
        assert T.ring_period(h) == round(max(2.6, ink5.P["period"] * C.pat_k(h / C.REF_H)), 2)
    a = T.make_ring_tile(1024, 4.07)
    assert np.array_equal(a, ink5._ring_tile(1024, 4.07, ink5.P["q"], 5))


def test_ring_window_uses_the_kernels_offsets(tmp_path):
    ring = T.RingCache(str(tmp_path))
    w, h = 300, 200
    n = T.noise_window(w, h, 10, 20, 50, 60, ring)
    full = T.make_ring_tile(T.ring_T(w, h), T.ring_period(h))
    Tn = full.shape[0]
    assert np.allclose(n[3], full[(np.arange(10, 60) + T.RING_OY) % Tn][:, (np.arange(20, 80) + T.RING_OX) % Tn], atol=2e-3)


def test_noise_modes_are_explicit_per_dataset_not_only_global(tmp_path):
    w, h = 120, 90
    ring = T.RingCache(str(tmp_path / "r"))
    full = T.noise_window(w, h, 5, 7, 40, 50, ring, "all")
    tiles = T.noise_window(w, h, 5, 7, 40, 50, ring, "tiles")
    none = T.noise_window(w, h, 5, 7, 40, 50, ring, "none")
    assert np.array_equal(full[:3], tiles[:3]) and not tiles[3].any() and full[3].any()
    assert not none.any()


def test_prepare_rings_in_several_processes_matches_single(tmp_path):
    items = [("a", "b", 800, hh) for hh in range(560, 1000, 40)]                 # 11 різних кроків штрихів -> піде багатопроцесним шляхом
    need = {T.ring_period(hh) for _, _, _, hh in items}
    assert len(need) > 8
    ring = T.prepare_rings(str(tmp_path), items, log_=lambda *a, **k: None, procs=2)
    for per in sorted(need)[:2]:
        a = np.load(ring.path(T.ring_T(800, 600), per))
        assert a.dtype == np.float16 and np.array_equal(a, T.make_ring_tile(T.ring_T(800, 600), per).astype(np.float16))
    assert T.prepare_rings(str(tmp_path), items, log_=lambda *a, **k: (_ for _ in ()).throw(AssertionError("вдруге нічого генерувати")), procs=2)


def _pairs(tmp_path, n=6, sizes=((120, 200), (300, 500), (70, 90), (256, 256), (140, 140), (400, 300))):
    items = []
    rng = np.random.default_rng(0)
    from PIL import Image
    for i in range(n):
        h, w = sizes[i % len(sizes)]
        img = cv2.GaussianBlur((rng.random((h, w, 3)) * 255).astype(np.uint8), (0, 0), 3)
        m = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 120)
        (tmp_path / "in").mkdir(exist_ok=True)
        (tmp_path / "ink").mkdir(exist_ok=True)
        cv2.imwrite(str(tmp_path / "in" / f"{i}.png"), img)
        Image.fromarray(m).save(str(tmp_path / "ink" / f"{i}.png"))
        items.append((f"in/{i}.png", f"ink/{i}.png", w, h))
    return items


def test_dataset_item_shapes_masks_and_padding(tmp_path):
    items = _pairs(tmp_path)
    ds = T.InkData(str(tmp_path), items, ring_dir=str(tmp_path / "_c"))
    for i, (_, _, w, h) in enumerate(items):
        S = T.bucket_of(h, w)
        b = ds[(i, S)]
        K = T.CROPS_PER[S]
        assert b["x"].shape == (K, 3, S, S) and b["n"].shape == (K, 4, S, S) and b["t"].shape == (K, 1, S, S)
        assert b["small"].shape == (K, 3, 96, 96) and b["meta"].shape == (K, 5)
        assert set(np.unique(b["t"].numpy())) <= {0, 1}
        v = b["v"].numpy()
        assert v[:, :, :min(S, h), :min(S, w)].all() and v.sum() == K * min(S, h) * min(S, w)       # лише справжня частина вирізки важить
        assert not b["t"].numpy()[v == 0].any()


def test_dataset_noise_mode_travels_with_the_dataset(tmp_path):
    items = _pairs(tmp_path, 2, ((200, 200), (200, 200)))
    b_all = T.InkData(str(tmp_path), items, ring_dir=str(tmp_path / "_c"), mode="all")[(0, 128)]
    b_none = T.InkData(str(tmp_path), items, ring_dir=str(tmp_path / "_c"), mode="none")[(0, 128)]
    assert b_all["n"].float().abs().sum() > 0 and b_none["n"].float().abs().sum() == 0


def test_bucket_sampler_makes_uniform_batches_and_covers_everything(tmp_path):
    items = [(f"a{i}", f"b{i}", w, h) for i, (w, h) in enumerate([(400, 300)] * 20 + [(200, 140)] * 20 + [(90, 60)] * 20 + [(1000, 900)] * 20)]
    s = T.BucketSampler(items, shuffle=False)
    seen = []
    for b in s:
        assert len({S for _, S in b}) == 1
        seen += [i for i, _ in b]
    assert sorted(seen) == list(range(len(items)))
    assert len(s) == sum(1 for _ in s)


def test_model_forward_backward_and_initial_loss():
    m = T.build_model("tiny")
    x, n, sm, mt = T.example_inputs(64, 96, "cpu")
    x, n = x.repeat(2, 1, 1, 1), n.repeat(2, 1, 1, 1)
    sm, mt = sm.repeat(2, 1, 1, 1), mt.repeat(2, 1)
    lg = m(x, n, sm, mt)
    assert lg.shape == (2, 1, 64, 96) and torch.isfinite(lg).all()
    t = (torch.rand(2, 1, 64, 96) > 0.65).float()
    loss, bce, tone = T.losses(lg, t, torch.ones_like(t))
    assert 0.5 < float(bce) < 0.9                                    # старт близько до логіту середньої частки білого, а не з великих випадкових значень
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert T.count_macs(m) > 5000


@pytest.mark.parametrize("name", ["tinyq", "fastq"])
def test_quick_presets(name, tmp_path):
    m = T.build_model(name)
    x, n, sm, mt = T.example_inputs(64, 96, "cpu")
    lg = m(x, n, sm, mt)
    assert lg.shape == (1, 1, 64, 96) and torch.isfinite(lg).all()
    t = (torch.rand(1, 1, 64, 96) > 0.65).float()
    T.losses(lg, t, torch.ones_like(t))[0].backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert len(m.glob(sm, mt)) == 3                                   # лише FiLM на 1/16, 1/8, 1/4
    assert T.count_macs(m, 128, 128) < T.count_macs(T.build_model(name[:-1] + "l"), 128, 128)
    out = str(tmp_path / "q.onnx")                                    # експорт двох частин збігається з моделлю
    ort = pytest.importorskip("onnxruntime")
    T.main(["export", "--random", "--preset", name, "--parts", "--out", out])
    sg = ort.InferenceSession(out[:-5] + "_global.onnx", providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    fo = sg.run(None, dict(small=sm.numpy(), meta=mt.numpy()))
    r = sb.run(None, {"image": x.numpy(), "noise": n.numpy(), **dict(zip([o.name for o in sg.get_outputs()], fo))})[0]
    assert r.shape == (1, 1, 64, 96)


@pytest.mark.parametrize("name", ["tinyl", "fastl"])
def test_lean_presets_work_like_the_others(name):
    m = T.build_model(name)
    x, n, sm, mt = T.example_inputs(64, 96, "cpu")
    lg = m(x, n, sm, mt)
    assert lg.shape == (1, 1, 64, 96) and torch.isfinite(lg).all()
    t = (torch.rand(1, 1, 64, 96) > 0.65).float()
    T.losses(lg, t, torch.ones_like(t))[0].backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert 5000 < T.count_macs(m, 128, 128) < T.count_macs(T.build_model(name.replace("l", "", 1) if name != "tinyl" else "tiny"), 128, 128) + 1


def test_model_speed_budget_ordering():
    macs = {p: T.count_macs(T.build_model(p), 128, 128) for p in ("tiny", "fast", "base")}
    assert macs["tiny"] < macs["fast"] < macs["base"] and macs["fast"] < 25000     # fast лишається в бюджеті реального часу


def test_global_descriptor_responds_to_exposure():
    m = T.build_model("tiny")
    dark = torch.rand(1, 3, 96, 96) * 0.2
    bright = 0.6 + torch.rand(1, 3, 96, 96) * 0.4
    a, b = m.desc(dark), m.desc(bright)
    assert a.shape == (1, 32) and (a - b).abs().max() > 0.5


def test_overfit_one_batch_reduces_loss(tmp_path):
    items = _pairs(tmp_path, 4, ((96, 96),) * 4)
    ds = T.InkData(str(tmp_path), items, ring_dir=str(tmp_path / "_c"))
    b = T.collate([ds[(i, 96)] for i in range(4)])
    torch.manual_seed(0)
    m = T.build_model("tiny")
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for _ in range(60):
        lg = m(b["x"].float() / 255, b["n"].float(), b["small"].float(), b["meta"])
        loss, _, _ = T.losses(lg, b["t"], b["v"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
        last = float(loss)
    assert last < 0.6 * first


def _toy_root(tmp_path, n=10):
    items = _pairs(tmp_path, n)
    with open(tmp_path / "manifest.jsonl", "w", encoding="utf-8") as f:
        for i, (a, b, w, h) in enumerate(items):
            f.write(json.dumps(dict(id=str(i), split="train" if i < n - 2 else "val", inp=a, ink=b, w=w, h=h)) + "\n")
    return str(tmp_path)


def test_train_loop_runs_saves_resumes_and_evaluates(tmp_path):
    root = _toy_root(tmp_path)
    args = ["train", "--data", root, "--run", "t1", "--preset", "tiny", "--hours", "1", "--max-steps", "4", "--workers", "0",
            "--smax", "96", "--batch-scale", "0.1", "--val-n", "2", "--full-n", "1", "--eval-min", "100", "--save-min", "100"]
    assert (T.main(args) or 0) == 0
    run = os.path.join(root, "runs", "t1")
    for f in ("last.pt", "best.pt", "log.csv", "args.json"):
        assert os.path.isfile(os.path.join(run, f))
    ck = torch.load(os.path.join(run, "last.pt"), weights_only=False)
    assert ck["step"] == 4 and ck["best"] is not None                    # найкращий результат збережено і в last.pt (для продовження)
    # продовження: та сама модель, навіть якщо вказано auto; крок і «найкращий» не губляться
    assert (T.main(["train", "--data", root, "--run", "t1", "--preset", "auto", "--hours", "1", "--max-steps", "7", "--workers", "0", "--smax", "96",
                    "--batch-scale", "0.1", "--val-n", "2", "--full-n", "1", "--resume"]) or 0) == 0
    ck2 = torch.load(os.path.join(run, "last.pt"), weights_only=False)
    assert ck2["step"] == 7 and ck2["args"]["preset"] == "tiny" and ck2["best"] <= ck["best"]
    # дотренування: +хвилини до пройденого часу, крок не скидається
    assert (T.main(["train", "--data", root, "--run", "t1", "--preset", "auto", "--add-min", "1", "--max-steps", "9", "--workers", "0", "--smax", "96",
                    "--batch-scale", "0.1", "--val-n", "2", "--full-n", "1", "--resume"]) or 0) == 0
    assert torch.load(os.path.join(run, "last.pt"), weights_only=False)["step"] == 9
    with pytest.raises(SystemExit):
        T.main(["train", "--data", root, "--run", "t_new", "--add-min", "5", "--workers", "0", "--smax", "96", "--resume"])
    model, _ = T.load_model(run, "ema", torch.device("cpu"))
    img = cv2.imread(os.path.join(root, "in", "1.png"))
    pred = T.run_frame(model, img, T.RingCache(os.path.join(root, "_cache")), torch.device("cpu"))
    assert pred.shape == img.shape[:2] and set(np.unique(pred)) <= {0, 1}


def test_noise_command_and_cuda_check(tmp_path):
    out = str(tmp_path / "n.npz")
    assert (T.main(["noise", "--data", str(tmp_path), "--w", "100", "--h", "64", "--out", out]) or 0) == 0
    z = np.load(out)
    assert z["noise"].shape == (4, 64, 100) and z["noise"].dtype == np.float16 and z["meta"].shape == (5,)
    assert T.main(["cuda"]) == (0 if torch.cuda.is_available() else 1)


def test_onnx_export_matches_torch(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    root = _toy_root(tmp_path, 4)
    assert (T.main(["train", "--data", root, "--run", "t2", "--preset", "tiny", "--max-steps", "2", "--workers", "0", "--smax", "96",
                    "--batch-scale", "0.1", "--val-n", "1", "--full-n", "1", "--hours", "1"]) or 0) == 0
    run = os.path.join(root, "runs", "t2")
    assert (T.main(["export", "--data", root, "--run", run]) or 0) == 0
    s = ort.InferenceSession(os.path.join(run, "ink_student.onnx"), providers=["CPUExecutionProvider"])
    model, _ = T.load_model(run, "ema", torch.device("cpu"))
    x = T.example_inputs(160, 224, "cpu")                              # інший розмір, ніж при експорті: динамічні осі
    got = s.run(None, dict(image=x[0].numpy(), noise=x[1].numpy(), small=x[2].numpy(), meta=x[3].numpy()))[0]
    with torch.no_grad():
        assert np.abs(got - model(*x).numpy()).max() < 1e-3


def test_export_parts_matches_model(tmp_path):
    import pytest
    ort = pytest.importorskip("onnxruntime")
    import torch
    out = str(tmp_path / "p.onnx")
    T.main(["export", "--random", "--preset", "tinyl", "--parts", "--out", out])
    x = T.example_inputs(96, 128, "cpu")
    sg = ort.InferenceSession(out[:-5] + "_global.onnx", providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    fo = sg.run(None, dict(small=x[2].numpy(), meta=x[3].numpy()))
    names = [o.name for o in sg.get_outputs()]
    r = sb.run(None, {"image": x[0].numpy(), "noise": x[1].numpy(), **dict(zip(names, fo))})[0]
    assert r.shape == (1, 1, 96, 128) and len(names) == 10


def test_export_parts_mask_bakes_the_threshold_into_the_output(tmp_path):
    """--mask: частина «body» віддає вже маску uint8 (логіт > 0) - так її й споживає застосунок (benvid/student)"""
    import pytest
    ort = pytest.importorskip("onnxruntime")
    import numpy as np
    plain, masked = str(tmp_path / "a.onnx"), str(tmp_path / "m.onnx")
    T.main(["export", "--random", "--preset", "tinyl", "--parts", "--out", plain])
    T.main(["export", "--random", "--preset", "tinyl", "--parts", "--mask", "--out", masked])
    x = T.example_inputs(96, 128, "cpu")
    sg = ort.InferenceSession(plain[:-5] + "_global.onnx", providers=["CPUExecutionProvider"])
    names = [o.name for o in sg.get_outputs()]
    film = dict(zip(names, sg.run(None, dict(small=x[2].numpy(), meta=x[3].numpy()))))
    feeds = {"image": x[0].numpy(), "noise": x[1].numpy(), **film}
    y = ort.InferenceSession(plain, providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    m = ort.InferenceSession(masked, providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    assert m.dtype == np.uint8 and m.shape == y.shape and np.array_equal(m, (y > 0).astype(np.uint8))


def test_nan_loss_rolls_back_and_training_continues(tmp_path, monkeypatch):
    root = _toy_root(tmp_path)
    orig, calls = T.losses, {"n": 0}

    def bad(*a, **k):
        calls["n"] += 1
        loss, bce, tone = orig(*a, **k)
        if 6 <= calls["n"] <= 12:                                  # серія кроків із NaN
            loss = loss * float("nan")
        return loss, bce, tone

    monkeypatch.setattr(T, "losses", bad)
    assert (T.main(["train", "--data", root, "--run", "tn", "--preset", "tiny", "--hours", "1", "--max-steps", "20", "--workers", "0", "--smax", "96",
                    "--batch-scale", "0.1", "--val-n", "2", "--full-n", "1", "--eval-min", "100", "--save-min", "100"]) or 0) == 0
    ck = torch.load(os.path.join(root, "runs", "tn", "last.pt"), weights_only=False)
    assert ck["step"] == 20 and all(torch.isfinite(v).all() for v in ck["model"].values() if v.is_floating_point())


def test_descriptor_gradient_is_finite_on_flat_frames():
    m = T.build_model("tiny")
    small = torch.full((1, 3, 96, 96), 0.5, requires_grad=True)
    m.desc(small).sum().backward()
    assert torch.isfinite(small.grad).all()


# ================================================================ легші пресети fq_*, дотренування з ваг, дистиляція, uint8-вхід
LITE = ["fq_noh2", "fq_nor1", "fq_nb1", "fq_a", "fq_b", "fq_c", "fq_d"]


@pytest.mark.parametrize("name", LITE)
def test_lite_presets_are_lighter_and_trainable(name):
    m = T.build_model(name)
    x, n, sm, mt = T.example_inputs(64, 96, "cpu")
    lg = m(x, n, sm, mt)
    assert lg.shape == (1, 1, 64, 96) and torch.isfinite(lg).all()
    t = (torch.rand(1, 1, 64, 96) > 0.65).float()
    T.losses(lg, t, torch.ones_like(t))[0].backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert len(m.glob(sm, mt)) == 3                                     # ті самі 3 пари FiLM: вхідний інтерфейс тіла для застосунку не змінюється
    assert T.count_macs(m, 128, 128) < T.count_macs(T.build_model("fastq"), 128, 128)


def test_partial_load_moves_matching_weights_only():
    src = T.build_model("fastq")
    for p in src.parameters():
        torch.nn.init.normal_(p, 0, 0.1)
    same, skip0 = T.load_partial(T.build_model("fq_noh2"), src.state_dict())
    assert skip0 == []                                                  # все, що лишилось у моделі, є в джерелі з тією самою формою
    dst = T.build_model("fq_c")                                         # hh=8: голова інша, блоки на 1/8 і 1/16 прибрані
    ok, skip = T.load_partial(dst, src.state_dict())
    assert "h1.weight" in skip and "h3.weight" in skip and "stem.weight" in ok and "d3.weight" in ok and "r4a.a.weight" in ok
    assert torch.equal(dst.state_dict()["u3.weight"], src.state_dict()["u3.weight"])
    d2 = T.build_model("fq_d")                                          # вужчі канали 1/8 і 1/16: ці тензори нові
    ok2, skip2 = T.load_partial(d2, src.state_dict())
    assert "d2.weight" in skip2 and "stem.weight" in ok2


def test_kd_loss_is_smallest_when_student_matches_teacher():
    t = torch.randn(2, 1, 16, 16)
    v = torch.ones_like(t)
    same = float(T.kd_loss(t.clone(), t, v))
    other = float(T.kd_loss(torch.randn(2, 1, 16, 16), t, v))
    assert same < other
    s = t.clone().requires_grad_(True)
    T.kd_loss(s, t + 1.0, v).backward()
    assert torch.isfinite(s.grad).all() and float(s.grad.abs().sum()) > 0
    vm = torch.zeros_like(t)                                            # поза маскою валідних пікселів внеску немає
    assert float(T.kd_loss(torch.randn_like(t), t, vm)) == 0.0


def test_train_from_other_run_with_distillation(tmp_path):
    root = _toy_root(tmp_path)
    base = ["--data", root, "--workers", "0", "--smax", "96", "--batch-scale", "0.1", "--val-n", "2", "--full-n", "1", "--eval-min", "100", "--save-min", "100"]
    assert (T.main(["train", "--run", "teacher", "--preset", "tiny", "--hours", "1", "--max-steps", "3", *base]) or 0) == 0
    assert (T.main(["train", "--run", "stud", "--preset", "tiny", "--hours", "1", "--max-steps", "3", "--init-from", "teacher", "--kd", "teacher", "--kd-w", "0.5", *base]) or 0) == 0
    ck = torch.load(os.path.join(root, "runs", "stud", "last.pt"), weights_only=False)
    assert ck["step"] == 3 and ck["args"]["kd"] == "teacher" and ck["args"]["init_from"] == "teacher"
    t0 = torch.load(os.path.join(root, "runs", "teacher", "best.pt"), weights_only=False)["ema"]
    assert set(ck["ema"]) == set(t0)                                    # форми збігаються: перенос ваг не зламав збереження


def test_u8_front_end_matches_the_float_model(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from make_u8_body import add_u8_front
    out = str(tmp_path / "b.onnx")
    T.main(["export", "--random", "--preset", "fq_b", "--parts", "--mask", "--out", out])
    u8 = str(tmp_path / "bu8.onnx")
    add_u8_front(out, u8)
    h, w = 64, 96
    rng = np.random.default_rng(5)
    bgr = (rng.random((1, h, w, 3)) * 255).astype(np.uint8)
    ins = T.example_inputs(h, w, "cpu")
    sg = ort.InferenceSession(out[:-5] + "_global.onnx", providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(u8, providers=["CPUExecutionProvider"])
    assert [i.name for i in sb.get_inputs()][0] == "bgr" and sb.get_inputs()[0].type == "tensor(uint8)"
    fo = sg.run(None, dict(small=ins[2].numpy(), meta=ins[3].numpy()))
    r = sb.run(None, {"bgr": bgr, "noise": ins[1].numpy(), **dict(zip([o.name for o in sg.get_outputs()], fo))})[0]
    sf = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    x = np.ascontiguousarray((bgr[..., ::-1].astype(np.float32) / 255.0).transpose(0, 3, 1, 2))
    ref = sf.run(None, {"image": x, "noise": ins[1].numpy(), **dict(zip([o.name for o in sg.get_outputs()], fo))})[0]
    assert r.dtype == np.uint8 and r.shape == (1, 1, h, w) and float((r != ref).mean()) < 0.002


def test_install_student_writes_assets_with_u8_body_and_meta(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxconverter_common")
    root = _toy_root(tmp_path)
    assert (T.main(["train", "--data", root, "--run", "r", "--preset", "fq_b", "--hours", "1", "--max-steps", "2", "--workers", "0", "--smax", "96",
                    "--batch-scale", "0.1", "--val-n", "2", "--full-n", "1", "--eval-min", "100", "--save-min", "100"]) or 0) == 0
    import install_student as I
    assets = str(tmp_path / "assets")
    assert I.main(["r", "--data", root, "--assets", assets]) == 0
    assert sorted(os.listdir(assets)) == ["bodyu8_fp16.onnx", "bodyu8_fp32.onnx", "global.onnx", "meta.json"]   # fq_b: без орієнтованого банка, of_fp32.onnx нема
    meta = json.load(open(os.path.join(assets, "meta.json"), encoding="utf-8"))
    assert meta["preset"] == "fq_b" and meta["step"] == 2 and len(meta["sha"]) == 16 and meta["min_h"] == 48


# ---------------------------------------------------------------- штрихи (IRNM1 = ir1)
def _stripes(h=96, w=96, period=4.0, angle=0.6):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u = x * np.cos(angle) + y * np.sin(angle)
    return (np.sin(2 * np.pi * u / period) > 0).astype(np.float32)


def test_gabor_bank_shape_norm_and_orientation_selectivity():
    k = T.gabor_kernels(6, 9, 4.0)
    assert k.shape == (12, 1, 9, 9)
    assert np.allclose(np.sqrt((k.reshape(12, -1) ** 2).sum(1)), 1.0, atol=1e-5)
    q = torch.from_numpy(_stripes(angle=0.0) * 2 - 1)[None, None]       # вертикальні смуги -> енергія в напрямку 0
    e, _ = T.stroke_energy(q, torch.ones_like(q))
    m = e.mean((0, 2, 3))
    assert int(m.argmax()) == 0 and float(m[0]) > 3 * float(m[3])


def test_stroke_err_orders_strokes_dither_and_flat():
    t = torch.from_numpy(_stripes())[None, None]
    v = torch.ones_like(t)
    flat = torch.full_like(t, 0.5)
    rng = np.random.default_rng(0)
    dither = torch.from_numpy((rng.random(t.shape) > 0.5).astype(np.float32))
    same = float(T.stroke_err(t, t, v))
    assert same < 1e-6
    assert float(T.stroke_err(flat, t, v)) > 0.9 and float(T.stroke_err(dither, t, v)) > 0.3
    shifted = torch.from_numpy(_stripes(angle=0.6 + 1.0))[None, None]      # штрихи, але в іншому напрямку - теж гірше
    assert float(T.stroke_err(shifted, t, v)) > 0.5


def test_metrics_report_stroke_and_loss_has_gradient():
    t = torch.from_numpy(_stripes())[None, None]
    v = torch.ones_like(t)
    m = T.metrics((t * 2 - 1) * 5, t, v)
    assert m["stroke"] < 1e-3 and "bce" in m
    lg = (torch.randn(t.shape) * 0.5).requires_grad_(True)             # у точно плоскому 0.5 градієнт енергії нульовий (g² у нулі)
    T.stroke_err(torch.sigmoid(lg), t, v).backward()
    assert torch.isfinite(lg.grad).all() and float(lg.grad.abs().sum()) > 0


def test_ir1_adds_frozen_oriented_bank_on_top_of_fq_b():
    a, b = T.build_model("fq_b"), T.build_model("ir1")
    ok, skip = T.load_partial(b, a.state_dict())
    assert sorted(skip) == ["oc.weight", "op.bias", "op.weight", "os.weight"] and len(ok) == len(a.state_dict())
    assert not b.oc.weight.requires_grad and b.op.weight.requires_grad
    b.eval(), a.eval()
    ins = T.example_inputs(64, 96, "cpu")
    with torch.no_grad():
        assert torch.allclose(a(*ins), b(*ins), atol=1e-6)               # нульовий старт: поведінка fq_b збережена
        b.op.weight.normal_()
        ring_off = ins[1].clone()
        ring_off[:, 3] = 0
        assert not torch.allclose(b(*ins), b(ins[0], ring_off, ins[2], ins[3]), atol=1e-4)      # тепер кільцевий канал справді впливає
    assert T.count_macs(b) < 1.3 * T.count_macs(a)


def test_ir1_trains_and_exports_parts_with_bank(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    m = T.build_model("ir1").eval()
    with torch.no_grad():
        m.op.weight.normal_(0, 0.1)
    ck = tmp_path / "c.pt"
    torch.save(dict(model=m.state_dict(), ema=m.state_dict(), args=dict(preset="ir1"), step=1), ck)
    out = tmp_path / "ir1.onnx"
    ns = T.main.__globals__["argparse"].Namespace(mask=False, fp16=False, parts=True)
    T._export_parts(ns, m, torch.load(ck, weights_only=False), str(out))
    x = T.example_inputs(128, 192, "cpu")
    g = ort.InferenceSession(str(out).replace(".onnx", "_global.onnx"), providers=["CPUExecutionProvider"])
    bd = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    names = [o.name for o in g.get_outputs()]
    fo = g.run(None, dict(small=x[2].numpy(), meta=x[3].numpy()))
    assert "ofeat" in [i.name for i in bd.get_inputs()]                 # внесок банка - окремий вхід: застосунок рахує його один раз на розмір кадру
    of = ort.InferenceSession(str(out).replace(".onnx", "_of.onnx"), providers=["CPUExecutionProvider"]).run(None, {"noise": x[1].numpy()})[0]
    assert of.shape == (1, 16, 64, 96)
    r = bd.run(None, {"image": x[0].numpy(), "noise": x[1].numpy(), "ofeat": of, **dict(zip(names, fo))})[0]
    with torch.no_grad():
        assert abs(r - m(*x).numpy()).max() < 1e-3
        gs = m.glob(x[2], x[3])
        assert torch.equal(m.body(x[0], x[1], gs), m.body(x[0], x[1], gs, of=m.ofeat(x[1])))


def test_ir1_train_loop_with_stroke_loss_and_init(tmp_path):
    root = _toy_root(tmp_path)
    base = ["--data", root, "--workers", "0", "--smax", "96", "--batch-scale", "0.1", "--val-n", "2", "--full-n", "1"]
    assert (T.main(["train", *base, "--run", "src", "--preset", "fq_b", "--hours", "1", "--max-steps", "3", "--eval-min", "100"]) or 0) == 0
    out = str(tmp_path / "irnm1")
    assert (T.main(["train", *base, "--out", out, "--preset", "ir1", "--init-from", os.path.join(root, "runs", "src"), "--hours", "1",
                    "--max-steps", "4", "--eval-min", "100", "--stroke-w", "1.0", "--stroke-sel", "0.5"]) or 0) == 0
    ck = torch.load(os.path.join(out, "best.pt"), weights_only=False)
    assert ck["args"]["preset"] == "ir1" and "oc.weight" in ck["ema"] and ck["best"] is not None
    head = open(os.path.join(out, "log.csv"), encoding="utf-8").readline()
    assert head.strip().endswith("val_stroke,full_stroke")
