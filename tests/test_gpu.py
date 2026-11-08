"""Відеокарта (OpenCL, benvid/gpu.py) і навантаження процесора. На CI/ПК без pyopencl тести GPU пропускаються;
перевірка йде на процесорному OpenCL-пристрої (pocl), бо ядра однакові з відеокарточними."""
import os

import numpy as np
import pytest

from benvid import core as C, gpu
from benvid.core import InkState, Params, binarize
from benvid.video import _make_plan

cl_ok = gpu.HAVE_CL and bool(gpu.list_devices(allow_cpu=True))
needs_cl = pytest.mark.skipif(not cl_ok, reason="pyopencl або OpenCL-пристрій відсутні")


@pytest.fixture()
def cpu_cl(monkeypatch, tmp_path):
    monkeypatch.setenv("INKREEL_GPU", "cpu")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    gpu._dev, gpu._state["use"] = None, None
    gpu.set_allowed(True)
    yield
    gpu._dev, gpu._state["use"] = None, None


def _img(w=640, h=360):
    yy, xx = np.mgrid[0:h, 0:w]
    base = (127 + 90 * np.sin(xx / 23.0) * np.cos(yy / 17.0)).clip(0, 255).astype(np.uint8)
    base[100:140, 50:300] = 255
    base[110:130, 60:90] = 0
    return np.dstack([base, np.roll(base, 5, 1), 255 - base])


def test_mode_parsing(monkeypatch):
    monkeypatch.setenv("INKREEL_GPU", "off")
    assert gpu.mode() == "off" and gpu.worker_slots() == 0
    monkeypatch.setenv("INKREEL_GPU", "дичина")
    assert gpu.mode() == "auto"


def test_off_means_no_device(monkeypatch):
    monkeypatch.setenv("INKREEL_GPU", "off")
    gpu._dev = None
    try:
        assert gpu.device() is None and not gpu.kernel_ok("draw")
        assert gpu.run("draw", lambda g: 1) is None
    finally:
        gpu._dev = None


@needs_cl
def test_every_kernel_matches_numba(cpu_cl):
    r = gpu.run_probe(w=640, h=360, save=False)
    assert r["err"] == {}, r["err"]
    assert set(gpu.KERNELS) <= set(r["use"])


@needs_cl
@pytest.mark.parametrize("level,limit", [(4, 0.004), (5, 0.006)])
def test_pipeline_gpu_equals_cpu(cpu_cl, level, limit):
    img = _img()
    p = Params(method=level)
    outs = {}
    for tag, on in (("cpu", False), ("gpu", True)):
        gpu._dev = None if on else False
        st = InkState()
        outs[tag] = [binarize(np.roll(img, k * 2, 1), 640, 360, p, {"state": st}) for k in range(3)]
    for a, b in zip(outs["cpu"], outs["gpu"]):
        assert float((a != b).mean()) < limit


@needs_cl
def test_gpu_error_falls_back_to_cpu(cpu_cl, monkeypatch):
    img = _img()
    p = Params(method=4)
    gpu._dev = False
    ref = binarize(img, 640, 360, p)
    gpu._dev = None
    g = gpu.device()
    assert g is not None

    def boom(*a, **k):
        raise RuntimeError("збій драйвера")
    monkeypatch.setattr(gpu.Gpu, "mix2", boom)
    out = binarize(img, 640, 360, p)             # помилка в ядрі не доходить до користувача
    assert out.shape == ref.shape and float((out != ref).mean()) < 0.004
    assert gpu.device() is None and "збій" in gpu.info()["reason"]


@needs_cl
def test_out_of_gpu_memory_is_not_fatal(cpu_cl):
    """відеокарту ділить багато процесів: нестача пам'яті в одному виклику - це CPU для цього виклику, а не вимкнений GPU"""
    gpu._dev = None
    g = gpu.device()
    assert g is not None

    def oom(gg):
        raise RuntimeError("clCreateBuffer failed: MEM_OBJECT_ALLOCATION_FAILURE")
    assert gpu.run("draw", oom) is None
    assert gpu.device() is not None and gpu._state.get("oom", 0) >= 1


@needs_cl
def test_big_buffers_are_released_after_call(cpu_cl):
    gpu._dev = None
    g = gpu.device()
    big = np.zeros(gpu.LEAN_BYTES // 4 + 1024, np.float32)
    assert gpu.run("draw", lambda gg: gg.up("x", big) and None) is None
    assert not any(ln._bufs for ln in list(g._free.queue))     # віддано пристрою
    small = np.zeros(1024, np.float32)
    gpu.run("draw", lambda gg: gg.up("y", small) and None)
    assert any("y" in ln._bufs for ln in list(g._free.queue))  # малі кадри тримають буфери


def test_slots_scale_with_frame_size(monkeypatch):
    class D:
        global_mem_size = 8 << 30
        vendor, name, driver_version = "v", "n", "1"
    monkeypatch.setattr(gpu, "HAVE_CL", True)
    monkeypatch.setattr(gpu, "list_devices", lambda allow_cpu=False: [(None, D())])
    monkeypatch.setattr(gpu, "mode", lambda: "auto")
    monkeypatch.setattr(gpu.os, "cpu_count", lambda: 28)
    monkeypatch.setattr(gpu, "_key_dev", lambda d: "k")
    import io, json as _j
    monkeypatch.setattr("builtins.open", lambda *a, **k: io.StringIO(_j.dumps({"key": "k", "ok": True, "use": {"draw": True}})))
    assert gpu.worker_slots(1280 * 720) == 8
    assert 8 <= gpu.worker_slots(3840 * 2160) <= 12            # 4K: більше процесів із відеокартою, але в межах відеопам'яті


@needs_cl
def test_not_allowed_process_uses_cpu(cpu_cl):
    gpu.set_allowed(False)
    try:
        assert gpu.device() is None
    finally:
        gpu.set_allowed(True)


def test_guided_plan_balances_workers():
    pd = Params(method=5)
    plan = _make_plan("x.mp4", pd, (1280, 720), 300, 96, None, 0, 27, 12)
    sizes = [p[1][2] for p in plan]
    assert sum(sizes) == 300 and min(sizes[:-1]) >= 12 and len(sizes) >= 20      # на коротких кліпах роботи вистачає всім ядрам
    plan = _make_plan("x.mp4", pd, (1280, 720), 6000, 96, None, 0, 27, 12)
    sizes = [p[1][2] for p in plan]
    assert sum(sizes) == 6000 and max(sizes) <= 96
    first = sizes[:27]
    assert first == sorted(first) and first[0] < first[-1], first                # перша хвиля - зростаючої довжини: процеси закінчують по черзі
    assert max(sizes[27:60]) > sizes[-2]                                         # далі великі шматки, наприкінці дрібні
    plan = _make_plan("x.mp4", pd, (1280, 720), 300, 96, None, 0)
    assert [p[1][2] for p in plan] == [96, 96, 96, 12]                           # без workers - як раніше


@needs_cl
def test_probe_works_in_auto_mode_without_verdict(monkeypatch, tmp_path):
    """перша перевірка (вердикту ще нема) мусить створити пристрій сама, а не вийти з порожнім звітом; далі вердикт діє"""
    monkeypatch.setenv("INKREEL_GPU", "auto")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    real = gpu.list_devices
    monkeypatch.setattr(gpu, "list_devices", lambda allow_cpu=False: real(allow_cpu=True))      # на CI відеокарти нема: підставляємо pocl
    gpu._dev, gpu._state["use"] = None, None
    assert gpu.device() is None                                    # без вердикту у звичайному режимі відеокарта не вмикається
    r = gpu.run_probe(w=640, h=360)
    assert r["name"] and r["err"] == {} and r["ok"] is not None
    assert os.path.isfile(gpu._cache_file())
    gpu._dev, gpu._state["use"], gpu._state["retry"] = None, None, 0
    assert (gpu.device() is not None) == bool(r["ok"])
    gpu._dev, gpu._state["use"] = None, None


def test_lane_count_follows_video_memory(monkeypatch):
    monkeypatch.delenv("INKREEL_GPU_LANES", raising=False)
    gb = 1 << 30
    assert [gpu.lanes_for(x * gb, "auto") for x in (1, 2, 4, 8, 16)] == [1, 2, 3, 4, 4]
    monkeypatch.setenv("INKREEL_GPU_LANES", "1")
    assert gpu.lanes_for(16 * gb, "auto") == 1


@needs_cl
def test_threads_use_separate_lanes_and_match_cpu(cpu_cl):
    """потоки господаря ходять у відеокарту через окремі смуги (власні черга, ядра й буфери): результат той самий, що на CPU, а
    смуг не більше за ліміт й усі повертаються у пул"""
    import threading
    img = _img(480, 270)
    p = Params(method=4)
    gpu._dev = False
    ref = binarize(img, 480, 270, p)
    gpu._dev = None
    g = gpu.device()
    assert g is not None
    outs, errs = [], []

    def work():
        try:
            for _ in range(3):
                outs.append(binarize(img, 480, 270, p))
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    th = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in th]
    [t.join(300) for t in th]
    assert not errs and len(outs) == 18
    assert all(float((o != ref).mean()) < 0.004 for o in outs)
    assert 1 <= g._made <= g.lanes_max and g._free.qsize() == g._made
    assert gpu.device() is g                                   # GPU не вимкнено
