"""Швидкий «Ультра»: навчений студент (ONNX Runtime). Сам вибирає, на чому рахувати: перебирає доступні провайдери
(CUDA, DirectML, CPU; TensorRT лише за INKREEL_STUDENT=trt), коротко міряє кожен на кадрі 720p, перевіряє, що маска збігається з CPU,
і запам'ятовує найшвидший (%LOCALAPPDATA%\\InkReel\\student.json). Налаштувань для користувача немає.
Змінна середовища INKREEL_STUDENT: auto (типово) | off | cpu | cuda | dml | trt - лише для розробки й перевірок."""
from __future__ import annotations

import collections
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

from . import inputs as I

_HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(_HERE), "assets", "student")
if getattr(sys, "frozen", False) and not os.path.isdir(ASSETS):
    ASSETS = os.path.join(getattr(sys, "_MEIPASS", ""), "benvid", "assets", "student")

PROV = {"trt": "TensorrtExecutionProvider", "cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider", "cpu": "CPUExecutionProvider"}
ORDER = ("trt", "cuda", "dml")                      # сходи запуску: від найшвидшого до найпростішого; процесор - останній (завжди є). Кожен сходинку перевіряють: запустився? маска збігається з CPU? швидший за процесор?
LABEL = {"trt": "TensorRT", "cuda": "CUDA", "dml": "DirectML", "cpu": "CPU"}
_lock = threading.RLock()
_acc_lock = threading.Lock()
_acc = dict(n=0, film=0.0, mask=0.0, wait=0.0, run=0.0, t0=time.perf_counter(), size=(0, 0))


def _note(**kw):
    with _acc_lock:
        for k, v in kw.items():
            if k == "size":
                _acc[k] = v
            else:
                _acc[k] += v


def stats_take() -> dict | None:
    """середні часи викликів студента з останнього разу (у мс на кадр) і швидкість; None, якщо викликів не було. Для журналу: куди йде час"""
    with _acc_lock:
        n, dt = _acc["n"], time.perf_counter() - _acc["t0"]
        out = None
        if n:
            out = dict(n=n, fps=n / max(dt, 1e-6), film=_acc["film"] / n * 1000, mask=_acc["mask"] / n * 1000, wait=_acc["wait"] / n * 1000,
                       run=_acc["run"] / n * 1000, size=_acc["size"])
        _acc.update(n=0, film=0.0, mask=0.0, wait=0.0, run=0.0, t0=time.perf_counter())
        return out


_GPU = threading.RLock()     # усі дії з відеокартою (створення/запуск/звільнення сеансів) - по одній, між усіма екземплярами Student: інакше звільнення старого сеансу
                             # (потік benvid-release, gc) йде паралельно зі створенням чи запуском нового, а DirectML такого не витримує (нативний збій без traceback)


def recycle():
    """відпустити завантажені сеанси (пам'ять відеокарти, плани DirectML під попередні розміри кадру) і підняти заново: викликається при відкритті
    нового відео, щоб друге відео не рахувалось на «обкладеному» першим сеансі. Вибір бекенду береться з кешу - це секунда"""
    global _inst
    old = None
    with _lock:
        if _inst:
            old, _inst = _inst, None
            _retry.update(backend=None, n=0, busy=False)
            get()
    if old is not None and old.backend != "cpu":
        threading.Thread(target=_release_old, args=(old,), daemon=True, name="benvid-release").start()


def _release_old(old) -> None:
    """старий Student: дочекатись кадрів, що ще біжать у ньому, і явно віддати відеопам'ять (сеанси DirectML, шум на пристрої); без цього пам'ять тримається до збирача сміття
    і за кілька відкриттів різних відео відеокарта впирається у пам'ять - кадр, що мав 20 мс, рахується 50-120 мс"""
    import gc
    try:
        with _GPU, old.lock:
            old.release()
            gc.collect()
    except Exception as e:  # noqa: BLE001
        _log(f"звільнення старого сеансу: {e}")


def _log(msg: str):
    try:
        from ..log import get
        get().info("student: %s", msg)
    except Exception:  # noqa: BLE001
        pass


def mode() -> str:
    m = (os.environ.get("INKREEL_STUDENT") or "auto").strip().lower()
    return m if m in ("auto", "off", "cpu", "cuda", "dml", "trt") else "auto"


def meta() -> dict:
    try:
        with open(os.path.join(ASSETS, "meta.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def files_ok() -> bool:
    return all(os.path.isfile(os.path.join(ASSETS, n)) for n in ("global.onnx", "bodyu8_fp32.onnx", "bodyu8_fp16.onnx"))


def _ort():
    try:
        import onnxruntime as ort
        return ort
    except Exception:  # noqa: BLE001
        return None


def providers_available() -> list[str]:
    ort = _ort()
    if ort is None:
        return []
    av = set(ort.get_available_providers())
    return [k for k in ("trt", "cuda", "dml", "cpu") if PROV[k] in av]


STAB_T = 6                                           # піксель «нерухомий», якщо його яскравість у двох попередніх кадрах відрізняється не більше ніж на стільки (з 255)
STAB_N = 2                                           # скільки попередніх кадрів усереднюється з поточним на нерухомих ділянках


def _stabilise(cur: np.ndarray, gcur: np.ndarray, hist: list) -> np.ndarray:
    """нерухомі ділянки кадру -> середнє з поточного й попередніх (шум стиснення зникає, а з ним і перемикання пікселів біля порогу); рух лишається як є"""
    g = None
    for _, gp in hist:
        d = cv2.absdiff(gcur, gp)
        g = d if g is None else cv2.max(g, d)
    m = cur
    for i, (f, _) in enumerate(reversed(hist), 1):
        m = cv2.addWeighted(m, i / (i + 1), f, 1 / (i + 1), 0)
    static = cv2.threshold(g, STAB_T, 255, cv2.THRESH_BINARY_INV)[1]
    out = cur.copy()
    cv2.copyTo(m, static, out)
    return out


class FilmState:
    """часова пам'ять рівня 6 у відео: (1) глобальні FiLM-параметри змінюються лише плавно, на різкій зміні сцени скидаються; (2) вхід стабілізується на
    нерухомих ділянках (середнє з поточного й двох попередніх кадрів) - без цього шум стиснення перемикає пікселі біля порогу, і статична картинка мерехтить.
    Результат кадра залежить лише від нього й двох попередніх кадрів, тож не залежить від нарізки відео на шматки (розігрів WARM кадрів це покриває)"""

    def __init__(self, alpha: float = 0.25, cut: float = 0.12, stab: bool | None = None):
        self.alpha, self.cut = alpha, cut
        self.film = None
        self.small = None
        self.hist: list = []                         # [(сирий BGR, сірий)] двох останніх кадрів, старіші першими
        self.stab = (os.environ.get("INKREEL_TSTAB", "on").lower() != "off") if stab is None else bool(stab)

    def stabilise(self, bgr: np.ndarray) -> np.ndarray:
        """-> кадр зі стабілізованими нерухомими ділянками (або сам bgr, якщо нема історії / вимкнено); кадр записується в історію"""
        if not self.stab:
            return bgr
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h = self.hist
        if h and h[-1][0].shape != bgr.shape:
            h.clear()                                # розмір змінився: історія не стосується цього кадру
        out = _stabilise(bgr, gray, h) if h else bgr
        h.append((bgr.copy(), gray))                 # сирий кадр, не стабілізований: інакше результат залежав би від довгої історії
        del h[:-STAB_N]
        return out

    def apply(self, film: list, small: np.ndarray) -> list:
        if self.film is not None and float(np.abs(small - self.small).mean()) <= self.cut:
            film = [a + self.alpha * (b - a) for a, b in zip(self.film, film)]
        self.film, self.small = film, small
        return film


def _trt_options() -> dict:
    """TensorRT будує рушій під форму входу (хвилини), тому: рушії кешуються в теці програми, а профіль форм охоплює всі розміри кадру (кратні 16,
    48..2160 × 48..3840), щоб будувати один раз, а не під кожен розмір. Параметри профілю - рядки «вхід:BxHxW...»; вхідні тензори - з моделі"""
    from ..log import app_dir
    cache = os.path.join(app_dir(), "trt_cache")
    try:
        os.makedirs(cache, exist_ok=True)
    except OSError:
        pass
    o = {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": cache, "trt_timing_cache_enable": True,
         "trt_timing_cache_path": cache}
    try:
        ort = _ort()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(os.path.join(ASSETS, "bodyu8_fp16.onnx"), so, providers=["CPUExecutionProvider"])
        sizes = {"h": (48, 1088, 2160), "w": (48, 1920, 3840), "h2": (24, 544, 1080), "w2": (24, 960, 1920)}              # мін / типовий / макс (усе кратне 16)
        lo, op, hi = [], [], []
        for inp in sess.get_inputs():
            for lst, k in ((lo, 0), (op, 1), (hi, 2)):
                dims = [str(d if isinstance(d, int) else sizes.get(str(d).lower(), (1, 1, 1))[k]) for d in inp.shape]
                lst.append(f"{inp.name}:" + "x".join(dims))
        o.update(trt_profile_min_shapes=",".join(lo), trt_profile_opt_shapes=",".join(op), trt_profile_max_shapes=",".join(hi))
    except Exception as e:  # noqa: BLE001
        _log(f"профіль форм TensorRT не складено ({e}): рушії будуватимуться під кожен розмір")
    return o


class Student:
    """одна завантажена модель на вибраному провайдері; потокобезпечна (виклики сходяться на одному сеансі)"""

    def __init__(self, backend: str, profile: bool = False):
        ort = _ort()
        if ort is None:
            raise RuntimeError("onnxruntime не встановлено")
        self.backend = backend
        self.fp16 = backend != "cpu"
        so = ort.SessionOptions()
        so.enable_mem_pattern = False                         # рекомендація для DirectML; на CPU різниці нема
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.log_severity_level = 3
        so.enable_profiling = bool(profile)                   # лише для `--prof`: ORT пише час кожного вузла в json
        so_g = ort.SessionOptions()                           # крихітна мережа на процесорі: один потік і без «крутіння» - кілька потоків подачі разом не душать процесор
        so_g.intra_op_num_threads = 1
        so_g.inter_op_num_threads = 1
        so_g.log_severity_level = 3
        so_g.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if backend != "cpu":
            so.intra_op_num_threads = 1                       # на відеокарті процесорні потоки сеансу не потрібні
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.glob = ort.InferenceSession(os.path.join(ASSETS, "global.onnx"), so_g, providers=["CPUExecutionProvider"])
        prov = {"cpu": ["CPUExecutionProvider"],
                "cuda": [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"],
                "dml": [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
                "trt": [("TensorrtExecutionProvider", _trt_options()), "CUDAExecutionProvider", "CPUExecutionProvider"]}[backend]
        self._mk_body = lambda: ort.InferenceSession(os.path.join(ASSETS, "bodyu8_fp16.onnx" if self.fp16 else "bodyu8_fp32.onnx"), so, providers=prov)
        with _GPU:
            self.body = self._mk_body()
        self._shapes = collections.OrderedDict()              # (H, W) кадру -> сеанс (лише відеокарта), див. _body_for
        self._unused = True                                   # перший сеанс ще не бачив жодного кадру
        got = self.body.get_providers()[0]
        if backend != "cpu" and got != PROV[backend]:
            raise RuntimeError(f"{backend}: сеанс узяв {got}")
        self.dt = np.float16 if self.fp16 else np.float32
        self.film_names = [o.name for o in self.glob.get_outputs()]
        self.lock = threading.Lock()
        self._noise_dev = {}                                  # (w, h) -> OrtValue на пристрої (шум залежить лише від розміру кадру)
        self._of = None                                       # ir1: тіло просить готовий внесок орієнтованого банка (вхід ofeat) - рахується на процесорі один раз на розмір кадру
        self._of_dev, self._of_np = {}, {}
        if any(i.name == "ofeat" for i in self.body.get_inputs()):
            self._of = ort.InferenceSession(os.path.join(ASSETS, "of_fp32.onnx"), so_g, providers=["CPUExecutionProvider"])

    # --- один кадр
    def film(self, bgr: np.ndarray, h: int) -> tuple[list, np.ndarray]:
        small = I.small_of(bgr)
        out = self.glob.run(None, {"small": small[None], "meta": I.meta_of(h)[None]})
        return out, small

    def mask(self, bgr: np.ndarray, film: list) -> np.ndarray:
        """кадр іде в модель як uint8 BGR (переведення в 0..1 RGB і транспонування - на відеокарті, у самому графі): на процесорі лише доповнення до кратного 16"""
        h, w = bgr.shape[:2]
        ph, pw = (-h) % 16, (-w) % 16
        pad = cv2.copyMakeBorder(bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT_101) if (ph or pw) else bgr
        feeds = {"bgr": np.ascontiguousarray(pad)[None], **{n: f.astype(self.dt, copy=False) for n, f in zip(self.film_names, film)}}
        if self.backend in ("dml", "cuda"):
            m = self._run_bound(feeds, w, h)
        else:
            feeds["noise"] = I.noise_for(w, h, self.dt)
            if self._of is not None:
                with self.lock:
                    feeds["ofeat"] = self._ofeat(w, h)
            m = self.body.run(None, feeds)[0]
        return np.ascontiguousarray(m[0, 0, :h, :w])

    def _ofeat(self, w: int, h: int) -> np.ndarray:
        """внесок орієнтованого банка для кадру w×h (1, c1, H/2, W/2) у типі моделі; кеш - останні 2 розміри (під self.lock)"""
        a = self._of_np.get((w, h))
        if a is None:
            if len(self._of_np) >= 2:
                self._of_np.pop(next(iter(self._of_np)))
            nz = I.noise_for(w, h, self.dt).astype(np.float32)             # fp32: процесорні ядра fp16-згортки є не скрізь
            a = self._of_np[(w, h)] = np.ascontiguousarray(self._of.run(None, {"noise": nz})[0].astype(self.dt))
        return a

    def release(self) -> None:
        """віддати сеанси й шум на пристрої (викликати під self.lock); якщо кадр ще прийде в цей екземпляр, сеанс просто створиться заново"""
        if self.backend == "cpu":
            return
        self._shapes.clear()
        self._noise_dev.clear()
        self._of_dev.clear()
        self._unused = False
        self.body = None

    def _body_for(self, key: tuple):
        """сеанс тіла під розмір кадру (викликати під self.lock). DirectML компілює ядра під ПЕРШИЙ побачений розмір, а після зміни розміру в тому самому
        сеансі працює в 2-3 рази повільніше (журнал: 2580x1080 давав 50 мс замість ~20 мс свіжого сеансу). Тому кожен новий розмір отримує свій сеанс;
        тримаємо два останніх (повернення до попереднього розміру не коштує нового компілювання), старіший вивільняється."""
        s_ = self._shapes.get(key)
        if s_ is not None:
            self._shapes.move_to_end(key)
            self.body = s_
            return s_
        t0 = time.perf_counter()
        if self._unused:
            s_, self._unused = self.body, False
        else:
            s_ = self._mk_body()
            _log(f"новий сеанс моделі під розмір {key[1]}x{key[0]}: {time.perf_counter() - t0:.2f} с")
        self._shapes[key] = s_
        dropped = False
        while len(self._shapes) > 2:
            self._shapes.popitem(last=False)
            dropped = True
        self.body = s_
        if dropped:
            import gc
            gc.collect()                                 # вивільнений сеанс одразу віддає відеопам'ять, а не «колись»
        return s_

    def _run_bound(self, feeds: dict, w: int, h: int) -> np.ndarray:
        ort = _ort()
        dev = "dml" if self.backend == "dml" else "cuda"
        t0 = time.perf_counter()
        with _GPU, self.lock:                                 # з відеокартою (шум на пристрої, прив'язка, запуск, копія результату) - по одному потоку: DirectML не любить паралельні виклики одного сеансу;
            t1 = time.perf_counter()                          # важка підготовка кадрів (опис, доповнення) у інших потоках іде поруч, поза замком
            body = self._body_for(tuple(feeds["bgr"].shape[1:3]))
            nz = self._noise_dev.get((w, h))
            if nz is None:
                if len(self._noise_dev) >= 2:
                    self._noise_dev.pop(next(iter(self._noise_dev)))
                nz = self._noise_dev[(w, h)] = ort.OrtValue.ortvalue_from_numpy(I.noise_for(w, h, self.dt), dev, 0)
            b = body.io_binding()
            b.bind_ortvalue_input("noise", nz)
            if self._of is not None:
                ov = self._of_dev.get((w, h))
                if ov is None:
                    if len(self._of_dev) >= 2:
                        self._of_dev.pop(next(iter(self._of_dev)))
                    ov = self._of_dev[(w, h)] = ort.OrtValue.ortvalue_from_numpy(self._ofeat(w, h), dev, 0)
                b.bind_ortvalue_input("ofeat", ov)
            for k, v in feeds.items():
                b.bind_cpu_input(k, v)
            b.bind_output(body.get_outputs()[0].name, "cpu")
            body.run_with_iobinding(b)
            out = b.copy_outputs_to_cpu()[0]
            t2 = time.perf_counter()
        _note(wait=t1 - t0, run=t2 - t1)
        if t2 - t1 > 1.0 or t1 - t0 > 3.0:                    # рідкість: відеокарта «задумалась» (інша програма, драйвер) - залишаємо слід у журналі
            _log(f"повільний виклик на {self.backend}: {w}×{h}, очікування замка {t1 - t0:.1f} с, запуск {t2 - t1:.1f} с")
        return out

    def binarize(self, img: np.ndarray, st: FilmState | None = None) -> np.ndarray:
        bgr = I.to_bgr(img)
        h, w = bgr.shape[:2]
        t0 = time.perf_counter()
        film, small = self.film(bgr, h)
        if st is not None:
            film = st.apply(film, small)
        t1 = time.perf_counter()
        m = self.mask(bgr, film)
        _note(n=1, film=t1 - t0, mask=time.perf_counter() - t1, size=(w, h))
        return m


# ---------------------------------------------------------------- вибір бекенду
_inst: Student | None = None
_state = {"backend": None, "reason": "", "probing": False, "ms": {}, "tried": {}, "stage": None}


def _cache_file() -> str:
    from ..log import app_dir
    return os.path.join(app_dir(), "student.json")


def _key() -> str:
    ort = _ort()
    m = meta()
    return f"v2|{getattr(ort, '__version__', '?')}|{','.join(providers_available())}|{m.get('sha', '')}"       # v2: вердикти старого способу заміру (3 повтори на 720p) відкинуто


def _bench(s: Student, w: int, h: int, reps: int = 6, warm: int = 3) -> float:
    """мс на кадр: НАЙКРАЩИЙ з reps повторів після прогріву. Середнє ламалося від чужого навантаження (інша програма на відеокарті, низькі
    частоти в першу секунду): одного такого заміру вистачало, щоб вибрати процесор, а відео потім рахувалось у 5-8 разів повільніше"""
    rng = np.random.default_rng(1)
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    for _ in range(warm):
        s.binarize(img)
    best = float("inf")
    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        s.binarize(img)
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


GPU_WINS = 0.85                                      # відеокарта береться, якщо її час менший за 85 % часу процесора (на 1080p реальний виграш - у 3-6 разів)


def probe(w: int = 1920, h: int = 1080, save: bool = True) -> dict:
    """заміряє кожного кандидата й перевіряє збіг маски з CPU-еталоном; повертає {"backend", "ms": {...}, "ok": bool}"""
    res = {"ok": False, "backend": "cpu", "ms": {}, "key": _key(), "ts": time.time()}
    av = providers_available()
    if "cpu" not in av:
        return res
    rng = np.random.default_rng(3)
    ref_img = np.clip(rng.random((200, 264, 1)) * 200 + rng.random((200, 264, 3)) * 55, 0, 255).astype(np.uint8)
    cpu = Student("cpu")
    ref = cpu.binarize(ref_img)
    res["ms"]["cpu"] = round(_bench(cpu, w, h, 3, 1), 1)
    best, best_ms = "cpu", res["ms"]["cpu"]
    m = mode()
    cands = [m] if m in ("cuda", "dml", "trt") else [k for k in ORDER if k in av]
    res["tried"] = {}
    for k in cands:                                      # сходинками: перша, що запустилась, збігається з CPU і швидша за нього, і є вибором
        if k not in av:
            continue
        _state["stage"] = k
        try:
            s = Student(k)
            agree = float((s.binarize(ref_img) == ref).mean())
            if agree < 0.97:
                _log(f"{k}: маска відрізняється від CPU ({agree:.3f}), іду на сходинку нижче")
                res["ms"][k] = None
                res["tried"][k] = f"маска відрізняється від CPU ({agree:.3f})"
                continue
            ms = round(_bench(s, w, h), 1)
            res["ms"][k] = ms
            if ms < best_ms * GPU_WINS:
                best, best_ms = k, ms
                res["tried"][k] = f"{ms} мс - обрано"
                break
            res["tried"][k] = f"{ms} мс - не швидший за процесор, іду нижче"
        except Exception as e:  # noqa: BLE001
            _log(f"{k}: не запустився: {e}")
            res["ms"][k] = None
            res["tried"][k] = f"не запустився: {str(e)[:120]}"
    _state["stage"] = None
    res.update(ok=True, backend=best)
    gpu_seen = any(v is not None for k, v in res["ms"].items() if k != "cpu")
    if save and not (best == "cpu" and gpu_seen):        # «процесор попри наявну відеокарту» не запам'ятовується: наступний запуск міряє знову (вердикт міг бути випадковим)
        try:
            with open(_cache_file(), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
        except OSError:
            pass
    return res


def _cached() -> dict | None:
    try:
        with open(_cache_file(), encoding="utf-8") as f:
            d = json.load(f)
        return d if d.get("ok") and d.get("key") == _key() else None
    except Exception:  # noqa: BLE001
        return None


def _build() -> Student | None:
    m = mode()
    if m == "off" or not files_ok() or _ort() is None:
        _state["reason"] = "вимкнено" if m == "off" else "немає моделі або onnxruntime"
        return None
    try:
        if m == "cpu":
            _state["backend"] = "cpu"
            return Student("cpu")
        d = _cached()
        if d is None:
            d = probe()
        _state.update(backend=d["backend"], ms=d.get("ms", {}), tried=d.get("tried", {}))
        _log(f"бекенд: {label(d['backend'])}; сходинки: {d.get('tried') or 'з кешу'}; мс: {d.get('ms')}")
        return Student(d["backend"])
    except Exception as e:  # noqa: BLE001
        _state["reason"] = f"не вдалося запустити: {e}"
        _log(_state["reason"])
        return None


def get() -> Student | None:
    """готовий студент або None (немає ORT/моделі/вимкнено). Перший виклик може тривати кілька секунд (вибір бекенду); далі - миттєво"""
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = _build() or False
    return _inst or None


def ready() -> bool:
    """студент уже вибрано й завантажено (без очікування); якщо ні - у фоні запускає вибір"""
    if _inst is not None:
        return bool(_inst)
    start_background()
    return False


def start_background():
    if _state["probing"] or _inst is not None:
        return
    _state["probing"] = True

    def run():
        try:
            get()
        finally:
            _state["probing"] = False
    threading.Thread(target=run, daemon=True, name="student-probe").start()


def is_fast() -> bool:
    """студент працює на відеокарті й швидший за CPU-алгоритм має сенс використовувати лише так"""
    s = get()
    return bool(s and s.backend in ("cuda", "dml", "trt"))


def label(backend: str | None = None) -> str:
    """що показати користувачу: «DirectML · GPU», «CPU» або порожньо, якщо студента ще нема"""
    b = backend if backend is not None else current()
    if not b:
        return ""
    return LABEL.get(b, b) + ("" if b == "cpu" else " · GPU")


def stage_label() -> str:
    """який варіант перевіряється просто зараз (під час вибору при першому запуску) або порожньо"""
    k = _state.get("stage")
    return LABEL.get(k, "") if k else ""


def info() -> dict:
    return dict(backend=_state.get("backend"), label=label(_state.get("backend")), reason=_state.get("reason"), ms=_state.get("ms"), tried=_state.get("tried"),
                providers=providers_available(), meta=meta(), mode=mode())


WARM = 10                                            # скільки кадрів перед шматком відео згладжування «пам'ятає» (0.75^10 ≈ 6% впливу)


def prepare(img: np.ndarray, st: FilmState | None) -> np.ndarray:
    """вхід моделі: BGR -> стабілізація нерухомих ділянок (історія кадрів)"""
    bgr = I.to_bgr(img)
    if st is not None:
        bgr = st.stabilise(bgr)
    return bgr


def warm(img: np.ndarray, st: FilmState | None) -> None:
    """лише опис кадру в згладжування (без маски): розігрів перед шматком відео. Мовчки нічого не робить, якщо студента нема"""
    s = get()
    if s is None or st is None or min(img.shape[:2]) < I.MIN_H:
        return
    try:
        bgr = prepare(img, st)                       # така сама історія, як при послідовному розрахунку
        film, small = s.film(bgr, bgr.shape[0])
        st.apply(film, small)
    except Exception as e:  # noqa: BLE001
        _log(f"розігрів: {e}")


def preparing() -> bool:
    """студента ще вибирають і завантажують (перший запуск - кілька секунд); не блокує"""
    return _inst is None and mode() != "off"


def current() -> str | None:
    """яким бекендом зараз рахує студент (cpu/cuda/dml/trt) або None, якщо ще не вибрано чи недоступний; не блокує"""
    return _state.get("backend") if _inst else None


def lower_steps(backend: str) -> list[str]:
    """GPU-сходинки нижче за backend, які є в onnxruntime (без процесора)"""
    av = providers_available()
    order = [k for k in ORDER if k in av]
    return order[order.index(backend) + 1:] if backend in order else []


_retry = {"backend": None, "at": 0.0, "n": 0, "busy": False}
RETRY_S, RETRY_MAX = 20.0, 5                       # перша повторна спроба через 20 с, далі вдвічі рідше; усього 5


def _gpu_retry() -> None:
    """після збою відеокарти (наприклад, DirectML-пам'ять на мить скінчилась) рахунок іде на процесорі ~4 к/с; тут раз на кілька секунд
    пробуємо повернути ту саму відеокарту: новий сеанс + пробний кадр, і лише тоді підміна. Зависла спроба потоки не блокує"""
    global _inst
    r = _retry
    be = r["backend"]
    if not be or r["busy"] or r["n"] >= RETRY_MAX or time.monotonic() < r["at"]:
        return
    with _lock:
        if r["busy"] or time.monotonic() < r["at"]:
            return
        r["busy"] = True
        r["at"] = time.monotonic() + RETRY_S * (2 ** r["n"])
        r["n"] += 1
    ok = None
    try:
        cand = Student(be)
        # кандидата ще ніхто не бачить, тож свій замок тут не береться: mask бере cand.lock сам (_run_bound, а з ofeat і процесорний шлях), а це
        # звичайний Lock - узятий тут наперед, він тримав би потік вічно (так і висів тест повернення на відеокарту, а в програмі - потік кадрів)
        f, _ = cand.film(np.zeros((96, 96, 3), np.uint8), 96)
        cand.mask(np.full((96, 96, 3), 127, np.uint8), f)
        ok = cand
    except Exception as e:  # noqa: BLE001
        _log(f"повернення на {be} не вдалося (спроба {r['n']}): {e}")
    with _lock:
        r["busy"] = False
        if ok is not None and _inst and _inst.backend != be:
            _inst = ok
            _state.update(backend=be, reason="")
            r["backend"] = None
            _log(f"відеокарта знову працює: повернулись на {be}")


def binarize(img: np.ndarray, st: FilmState | None = None) -> np.ndarray | None:
    """маска (h, w) uint8 {0,1} (1 = світло) або None, якщо студента нема / кадр менший за 48 px / помилка (тоді рахує алгоритм)"""
    s = get()
    if s is None or min(img.shape[:2]) < I.MIN_H:
        return None
    if _retry["backend"]:
        _gpu_retry()
        s = get()
    try:
        img = prepare(img, st)                       # один раз на кадр: повтор після збою відеокарти бере вже підготовлений
        return s.binarize(img, st)
    except Exception as e:  # noqa: BLE001
        _log(f"помилка виконання на {s.backend}: {e}")
        if s.backend == "cpu":
            return None
        global _inst
        with _lock:                                   # збій: на сходинку нижче (TensorRT -> CUDA -> DirectML), останньою - процесор (вигляд не стрибне: та сама мережа)
            if _inst is s:
                top = _retry["backend"] or s.backend
                nxt = None
                for k in lower_steps(s.backend):
                    try:
                        nxt = Student(k)
                        break
                    except Exception as e2:  # noqa: BLE001
                        _log(f"{k}: не запустився: {e2}")
                if nxt is None:
                    try:
                        nxt = Student("cpu")
                    except Exception as e2:  # noqa: BLE001
                        _inst = False
                        _state["reason"] = f"{s.backend}: {e}; cpu: {e2}"
                if nxt is not None:
                    _inst = nxt
                    _state.update(backend=nxt.backend, reason=f"{s.backend}: {e}")
                    _retry.update(backend=top, at=time.monotonic() + RETRY_S, n=0, busy=False)
                    _log(f"перейшли з {s.backend} на {nxt.backend}")
            cpu = _inst or None
        if cpu is None or cpu is s:
            return None
        try:
            return cpu.binarize(img, st)
        except Exception as e3:  # noqa: BLE001
            _log(f"помилка виконання на cpu: {e3}")
            return None
