"""Студент рівня Ультра в застосунку: входи збігаються з тренувальними, маска коректна, вибір бекенду й запасний шлях."""
import os
import sys
import threading
import time

import cv2
import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import train_ink as T  # noqa: E402

from benvid.student import inputs as I  # noqa: E402
from benvid.student import runtime as R  # noqa: E402


def _img(h, w, seed=1):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur((rng.random((h, w, 3)) * 255).astype(np.uint8), (0, 0), 3)
    return np.ascontiguousarray(base)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setenv("INKREEL_STUDENT", "cpu")
    monkeypatch.setattr("benvid.log.app_dir", lambda: str(tmp_path))
    R._inst = None
    R._state.update(backend=None, reason="", probing=False, ms={})
    I._noise_cache.clear()
    I._rings.clear()
    yield
    R._inst = None


@pytest.mark.parametrize("w,h", [(97, 61), (320, 180), (642, 290)])
def test_inputs_equal_training_inputs(w, h, tmp_path):
    img = _img(h, w)
    x, n, sm, mt, _ = T.frame_inputs(img, T.RingCache(str(tmp_path / "ring")))
    assert np.array_equal(I.image_tensor(img, np.float32), x)
    assert np.array_equal(I.noise_for(w, h, np.float32), n)
    assert np.array_equal(I.small_of(img), sm[0]) and np.array_equal(I.meta_of(h), mt[0])


def test_ring_tile_matches_algorithm():
    from benvid import ink5
    a = I.make_ring_tile(1024, 3.6)
    b = ink5._ring_tile(1024, 3.6, I.RING_Q, I.RING_SEED)
    assert np.array_equal(a, b)


def test_binarize_shape_dtype_and_determinism():
    s = R.get()
    assert s is not None and s.backend == "cpu"
    img = _img(130, 211)
    m1, m2 = s.binarize(img), s.binarize(img)
    assert m1.shape == (130, 211) and m1.dtype == np.uint8 and set(np.unique(m1)) <= {0, 1}
    assert np.array_equal(m1, m2)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                           # GRAY і BGRA приймаються
    assert s.binarize(g).shape == g.shape
    assert s.binarize(np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])).shape == img.shape[:2]


def test_fp16_body_matches_fp32_body_closely():
    s = R.get()
    img = _img(144, 256, 4)
    bgr = I.to_bgr(img)
    film, _ = s.film(bgr, 144)
    m32 = s.mask(bgr, film)
    b16 = ort.InferenceSession(os.path.join(R.ASSETS, "bodyu8_fp16.onnx"), providers=["CPUExecutionProvider"])
    feeds = {"bgr": bgr[None], "noise": I.noise_for(256, 144, np.float16), **{n: f.astype(np.float16) for n, f in zip(s.film_names, film)}}
    if any(i.name == "ofeat" for i in b16.get_inputs()):                   # ir1: тіло просить готовий внесок орієнтованого банка (див. Student.mask)
        with s.lock:
            feeds["ofeat"] = s._ofeat(256, 144).astype(np.float16)
    m16 = b16.run(None, feeds)[0][0, 0, :144, :256]
    assert (m16 != m32).mean() < 0.01


def test_student_resembles_the_teacher_tone():
    """тон студента близький до тону вхідного кадру: яскравий кадр дає багато білого, темний - мало"""
    s = R.get()
    bright = np.full((120, 160, 3), 235, np.uint8)
    dark = np.full((120, 160, 3), 25, np.uint8)
    assert s.binarize(bright).mean() > 0.8 and s.binarize(dark).mean() < 0.2


def test_small_frames_and_off_mode_fall_back(monkeypatch):
    assert R.binarize(_img(40, 200)) is None                              # < 48 px - студент так не навчався
    monkeypatch.setenv("INKREEL_STUDENT", "off")
    R._inst = None
    assert R.binarize(_img(100, 100)) is None and R.get() is None


def test_runtime_error_on_gpu_switches_to_the_same_network_on_cpu(monkeypatch):
    s = R.get()
    img = _img(100, 100)
    want = s.binarize(img)
    s.backend = "dml"                                                       # імітуємо відеокарту, що відмовила під час роботи
    orig = R.Student.binarize
    calls = {"n": 0}

    def flaky(self, im, st=None):
        calls["n"] += 1
        if self is s:
            raise RuntimeError("boom")
        return orig(self, im, st)

    monkeypatch.setattr(R.Student, "binarize", flaky)
    got = R.binarize(img)
    assert np.array_equal(got, want), "кадр пораховано тією ж мережею на процесорі"
    assert R._inst is not s and R._inst.backend == "cpu" and R._state["backend"] == "cpu" and "boom" in R._state["reason"]
    assert np.array_equal(R.binarize(img), want)                            # далі йде новий сеанс без повторних помилок


def test_cpu_failure_returns_none_so_the_algorithm_takes_over(monkeypatch):
    R.get()
    monkeypatch.setattr(R.Student, "binarize", lambda self, img, st=None: (_ for _ in ()).throw(RuntimeError("boom")))
    assert R.binarize(_img(100, 100)) is None


def test_film_state_smooths_and_resets_on_scene_cut():
    st = R.FilmState(alpha=0.5, cut=0.1)
    a, b = [np.zeros((1, 3, 1, 1), np.float32)], [np.ones((1, 3, 1, 1), np.float32)]
    s0 = np.zeros((3, 4, 4), np.float32)
    assert np.array_equal(st.apply(a, s0)[0], a[0])
    assert np.allclose(st.apply(b, s0 + 0.01)[0], 0.5)                      # той самий план - згладжено
    assert np.array_equal(st.apply(b, s0 + 0.9)[0], b[0])                   # різка зміна сцени - скинуто


def test_probe_picks_cpu_when_no_gpu_provider_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "providers_available", lambda: ["cpu"])           # на ПК з відеокартою справжній список містив би dml
    d = R.probe(w=192, h=128)
    assert d["ok"] and d["backend"] == "cpu" and "cpu" in d["ms"]
    assert os.path.isfile(R._cache_file()) and R._cached()["backend"] == "cpu"


def test_recycle_gives_a_fresh_working_session_and_stats_are_reported(monkeypatch):
    """при відкритті нового відео сеанс піднімається заново (пам'ять відеокарти, плани під старі розміри); журнал знає, куди йде час"""
    from benvid.student import runtime as R
    monkeypatch.setenv("INKREEL_STUDENT", "cpu")
    R._inst = None
    s1 = R.get()
    img = (np.random.default_rng(3).random((96, 160, 3)) * 255).astype(np.uint8)
    R.stats_take()
    a = R.binarize(img)
    st = R.stats_take()
    assert st and st["n"] == 1 and st["size"] == (160, 96) and st["film"] > 0 and st["mask"] > 0 and R.stats_take() is None
    R.recycle()
    s2 = R.get()
    assert s2 is not None and s2 is not s1
    assert np.array_equal(R.binarize(img), a), "новий сеанс дає ту саму маску"


def test_gpu_body_session_is_per_frame_size_and_keeps_two(monkeypatch):
    """DirectML повільніший після зміни розміру в тому самому сеансі: кожен новий розмір - свій сеанс (перший використовується повторно), два останні в кеші"""
    monkeypatch.setenv("INKREEL_STUDENT", "cpu")
    s = R.Student("cpu")
    first = s.body
    a = s._body_for((1440, 3440))
    assert a is first and s.body is a                                   # перший сеанс не витрачено даремно
    b = s._body_for((1088, 1920))
    assert b is not a and s.body is b
    assert s._body_for((1440, 3440)) is a                               # повернення до попереднього розміру - без нового сеансу
    c = s._body_for((720, 1280))
    assert c is not a and c is not b and len(s._shapes) == 2 and (1088, 1920) not in s._shapes   # найдавніший (за використанням) вивільнено
    assert s._body_for((1440, 3440)) is a


def _noisy_static(n=10, h=192, w=320, sig=2.0, seed=1):
    import cv2
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.dstack([(xx * 255 // w), (yy * 255 // h), ((xx + yy) * 255 // (w + h))]).astype(np.uint8)
    cv2.putText(base, "IRNM1", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (230, 230, 230), 4)
    base = base.astype(np.float32)
    return base, [np.clip(base + rng.normal(0, sig, base.shape), 0, 255).astype(np.uint8) for _ in range(n)]


def test_input_stabiliser_averages_static_areas_and_keeps_motion(monkeypatch):
    monkeypatch.setenv("INKREEL_TSTAB", "on")
    base, fr = _noisy_static(6)
    fr[5][40:80, 40:90] = np.clip(fr[5][40:80, 40:90].astype(np.int16) + 100, 0, 255).astype(np.uint8)       # ділянка, що різко змінилась проти двох попередніх кадрів
    st = R.FilmState()
    outs = [st.stabilise(f) for f in fr]
    assert outs[0] is fr[0]                                     # без історії кадр лишається як є
    o, f = outs[5].astype(np.int16), fr[5].astype(np.int16)
    assert np.array_equal(outs[5][40:80, 40:90], fr[5][40:80, 40:90])          # там, де кадр змінився, нічого не змінено
    still = (slice(120, 190), slice(0, 320))
    assert np.abs(o[still] - base[still]).mean() < np.abs(f[still] - base[still]).mean()     # на нерухомому ближче до чистої картинки, ніж сирий кадр
    assert len(st.hist) == R.STAB_N and all(h[0].flags.owndata for h in st.hist)       # історія тримає власні копії: зовнішній буфер можна перезаписувати


def test_input_stabiliser_does_not_depend_on_chunking_and_can_be_switched_off(monkeypatch):
    _, fr = _noisy_static(7)
    a = R.FilmState()
    seq = [a.stabilise(f) for f in fr]
    b = R.FilmState()                                           # шматок, що почався з розігріву двома кадрами, дає той самий кадр
    for f in fr[3:5]:
        b.stabilise(f)
    assert np.array_equal(b.stabilise(fr[5]), seq[5])
    c = R.FilmState()
    c.stabilise(fr[0])
    assert np.array_equal(c.stabilise(np.zeros((10, 10, 3), np.uint8)), np.zeros((10, 10, 3), np.uint8))   # розмір змінився: історія скинута, не падає
    monkeypatch.setenv("INKREEL_TSTAB", "off")
    off = R.FilmState()
    assert all(off.stabilise(f) is f for f in fr) and off.hist == []


def test_stabiliser_cuts_flicker_of_a_static_noisy_scene(monkeypatch):
    monkeypatch.setenv("INKREEL_STUDENT", "cpu")
    _, fr = _noisy_static(9)

    def flips(stab):
        st = R.FilmState(stab=stab)
        ms = np.array([R.binarize(f, st) for f in fr][3:])
        return float((ms[1:] != ms[:-1]).mean())

    off, on = flips(False), flips(True)
    assert off > 0.005 and on < 0.7 * off                       # вимір на ПК-подібних кадрах давав ≈0.4


def test_recycle_releases_the_old_gpu_instance(monkeypatch):
    monkeypatch.setenv("INKREEL_STUDENT", "cpu")
    s = R.Student("cpu")
    s.backend = "dml"                                           # лише поведінка кешу сеансів: нічого GPU-шного не викликається
    s._body_for((64, 64))
    s._body_for((96, 96))
    s._noise_dev[(1, 1)] = object()
    s.release()
    assert not s._shapes and not s._noise_dev and s.body is None
    assert s._body_for((64, 64)) is not None                    # запізнілий кадр у старому екземплярі не падає
    c = R.Student("cpu")
    c.release()
    assert c.body is not None                                   # на процесорі нічого не чіпаємо



def _within(seconds, fn, *args):
    """виклик у потоці з обмеженням у часі: дедлок замка перетворюється на помилку тесту, а не на завислий прогін (і python.exe, що лишається в пам'яті)"""
    box = {}

    def run():
        try:
            box["r"] = fn(*args)
        except BaseException as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=run, daemon=True, name="test-within")
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"{getattr(fn, '__name__', fn)} не повернувся за {seconds} с: дедлок замка?"
    if "e" in box:
        raise box["e"]
    return box["r"]


def test_gpu_failure_falls_back_to_cpu_then_returns_to_gpu(monkeypatch):
    """збій відеокарти -> процесор; через RETRY_S нова спроба: нова відеокарта підміняє процесор лише після пробного кадру"""
    monkeypatch.setenv("INKREEL_STUDENT", "cpu")
    R.recycle()
    real = R.Student("cpu")

    class FakeGpu:
        backend = "dml"
        film_names = real.film_names

        def __init__(self, ok=True):
            self.ok = ok
            self.lock = threading.Lock()                # як у справжнього Student: звичайний (не вкладений) замок екземпляра

        def film(self, bgr, h):
            return real.film(bgr, h)

        def mask(self, bgr, film):
            with self.lock:                             # справжній mask бере замок сам (_run_bound): повторна спроба не має тримати його наперед
                if not self.ok:
                    raise RuntimeError("DML: out of memory")
                return real.mask(bgr, film)

        def binarize(self, img, st=None):
            if not self.ok:
                raise RuntimeError("DML: out of memory")
            return real.binarize(img, st)

    img = np.random.default_rng(0).integers(0, 255, (96, 160, 3), dtype=np.uint8)
    bad = FakeGpu(ok=False)
    monkeypatch.setattr(R, "_inst", bad)
    ref = _within(30, R.binarize, img)
    assert ref is not None and R._inst.backend == "cpu" and R._retry["backend"] == "dml"
    created = []
    monkeypatch.setattr(R, "Student", lambda be: created.append(be) or (FakeGpu(ok=False) if len(created) == 1 else FakeGpu(ok=True)))
    R._retry["at"] = 0.0
    assert np.array_equal(_within(30, R.binarize, img), ref) and R._inst.backend == "cpu" and created == ["dml"]       # перша спроба невдала - лишились на CPU
    R._retry["at"] = 0.0
    assert np.array_equal(_within(30, R.binarize, img), ref)
    assert R._inst.backend == "dml" and R._retry["backend"] is None and created == ["dml", "dml"]
    R._retry["at"] = 0.0
    _within(30, R.binarize, img)
    assert created == ["dml", "dml"]                                    # повернулись - зайвих спроб нема
    R.recycle()


def _fake_gpu_factory(real, broken=(), slow=()):
    class G:
        def __init__(self, be):
            if be in broken:
                raise RuntimeError(f"{be}: немає драйвера")
            self.backend, self.slow = be, be in slow
            self.lock, self.film_names = real.lock, real.film_names

        def binarize(self, img, st=None):
            if img.shape[:2] == (200, 264):
                return real.binarize(img)
            if self.slow:
                time.sleep(0.05)
            return np.zeros(img.shape[:2], np.uint8)
    return lambda be: real.__class__(be) if be == "cpu" else G(be)


def test_probe_goes_through_steps_trt_cuda_dml_and_stops_at_first_good(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_cache_file", lambda: str(tmp_path / "student.json"))
    monkeypatch.setattr(R, "providers_available", lambda: ["trt", "cuda", "dml", "cpu"])
    real = R.Student("cpu")
    monkeypatch.setattr(R, "Student", _fake_gpu_factory(real, broken=("trt",)))
    d = R.probe(w=192, h=128)
    assert d["backend"] == "cuda" and "dml" not in d["tried"]                   # DirectML навіть не пробували: CUDA вже підійшла
    assert d["tried"]["trt"].startswith("не запустився") and d["tried"]["cuda"].endswith("обрано")
    monkeypatch.setattr(R, "Student", _fake_gpu_factory(real, broken=("trt", "cuda")))
    assert R.probe(w=192, h=128, save=False)["backend"] == "dml"
    monkeypatch.setattr(R, "Student", _fake_gpu_factory(real, broken=("trt", "cuda", "dml")))
    assert R.probe(w=192, h=128, save=False)["backend"] == "cpu"
    monkeypatch.setattr(R, "Student", _fake_gpu_factory(real, slow=("trt", "cuda", "dml")))
    r = R.probe(w=192, h=128, save=False)
    assert r["backend"] in ("trt", "cuda", "dml", "cpu") and set(r["tried"]) <= {"trt", "cuda", "dml"}


def test_label_and_lower_steps(monkeypatch):
    monkeypatch.setattr(R, "providers_available", lambda: ["trt", "cuda", "dml", "cpu"])
    assert R.lower_steps("cuda") == ["dml"] and R.lower_steps("dml") == [] and R.lower_steps("trt") == ["cuda", "dml"]
    assert R.label("dml") == "DirectML · GPU" and R.label("cpu") == "CPU" and R.label("trt") == "TensorRT · GPU"


def test_runtime_failure_steps_down_one_level_before_cpu(monkeypatch):
    monkeypatch.setenv("INKREEL_STUDENT", "cpu")
    R.recycle()
    monkeypatch.setattr(R, "providers_available", lambda: ["cuda", "dml", "cpu"])
    real = R.Student("cpu")

    class Top:
        backend = "cuda"
        lock = real.lock

        def binarize(self, img, st=None):
            raise RuntimeError("CUDA: out of memory")

    class Mid:
        backend = "dml"

        def binarize(self, img, st=None):
            return real.binarize(img, st)

    monkeypatch.setattr(R, "Student", lambda be: Mid() if be == "dml" else real.__class__("cpu"))
    monkeypatch.setattr(R, "_inst", Top())
    img = np.random.default_rng(1).integers(0, 255, (96, 160, 3), dtype=np.uint8)
    assert R.binarize(img) is not None
    assert R._inst.backend == "dml" and R._retry["backend"] == "cuda"           # зупинились на DirectML, а не на процесорі; верх запам'ятовано для повернення
    R.recycle()


def test_ir1_oriented_bank_is_precomputed_once_per_size_and_matches_pytorch(tmp_path, monkeypatch):
    """ir1: внесок орієнтованого банка (залежить лише від шуму) рахується окремим малим сеансом один раз на розмір кадру й подається в тіло входом ofeat"""
    torch = pytest.importorskip("torch")                  # у звичайному .venv торча нема: тест для середовища навчання (.venv_train)
    pytest.importorskip("onnx")
    pytest.importorskip("onnxconverter_common")
    import install_student
    m = T.build_model("ir1").eval()
    with torch.no_grad():
        m.op.weight.normal_(0, 0.3)
        m.op.bias.normal_(0, 0.1)
    run = tmp_path / "run"
    run.mkdir()
    torch.save(dict(model=m.state_dict(), ema=m.state_dict(), args=dict(preset="ir1"), step=1), run / "best.pt")
    assets = str(tmp_path / "assets")
    assert install_student.main([str(run), "--assets", assets, "--min-match", "0.9"]) == 0
    assert "of_fp32.onnx" in os.listdir(assets)
    monkeypatch.setattr(R, "ASSETS", assets)
    s = R.Student("cpu")
    assert s._of is not None
    img = _img(80, 112)
    got = s.binarize(img)
    h, w = img.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0)
    with torch.no_grad():
        ref = m(x, torch.from_numpy(I.noise_for(w, h, np.float32)), torch.from_numpy(I.small_of(img))[None], torch.from_numpy(I.meta_of(h))[None]).numpy()[0, 0] > 0
    assert float((got.astype(bool) == ref).mean()) > 0.97
    s.binarize(img)
    assert list(s._of_np) == [(w, h)]                                   # другий кадр того самого розміру не рахує банк заново


def test_releasing_an_old_instance_waits_for_all_other_gpu_work():
    """звільнення старого сеансу (потік benvid-release) не йде паралельно зі створенням/запуском сеансів іншого екземпляра: спільний замок R._GPU"""
    import threading

    class _Old:
        backend = "dml"

        def __init__(self):
            self.lock, self.released = threading.Lock(), False

        def release(self):
            self.released = True

    Old = _Old()

    with R._GPU:                                   # «інший екземпляр» зараз працює з відеокартою
        t = threading.Thread(target=R._release_old, args=(Old,), daemon=True)
        t.start()
        t.join(0.3)
        assert t.is_alive() and not Old.released
    t.join(5)
    assert Old.released and not t.is_alive()
