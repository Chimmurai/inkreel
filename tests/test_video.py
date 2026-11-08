import os
import sys
import tempfile
import time
import threading

import cv2
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid.core import Params  # noqa: E402
from benvid.video import FORMATS, Cancelled, export_video, fps_fraction, out_size, video_info  # noqa: E402


def make_video(path, n=40, w=320, h=180, fps=23.976):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = np.linspace(0, 255, w, dtype=np.uint8)[None, :, None]
        cv2.circle(img, (20 + i * 6, 90), 30, (255, 255, 255), -1)
        vw.write(img)
    vw.release()


def test_helpers():
    assert str(fps_fraction(23.976023976)) == "24000/1001" and str(fps_fraction(25)) == "25"
    assert out_size(1280, 536, "720p") == (1720, 720)
    assert out_size(853, 480, "Як у джерела") == (854, 480)
    w, h = out_size(1920, 1080, "360p")
    assert (w, h) == (640, 360)


def test_out_size_keeps_the_aspect_at_the_side_limit():
    """21:9 на 2160p раніше давало 4096x2160 (ширину зрізало до стелі, висоту - ні): кадр розтягувався по вертикалі"""
    assert out_size(1280, 540, "2160p") == (4096, 1728)
    assert out_size(540, 1280, "x9000") == (1728, 4096)
    assert out_size(1280, 540, "5000x") == (4096, 1728)
    assert out_size(1920, 1080, "2160p") == (3840, 2160)
    assert out_size(1920, 1080, "9000x9000") == (4096, 4096), "явні ШxВ - як ввів користувач (кожна сторона окремо)"


def test_quality_ladder_shrinks_the_file_and_keeps_the_bits():
    """«Висока» > «Середня» > «Мала» за розміром файлу; після порогу 128 біти «Високої» збігаються з тим, що віддав рендер (1-бітне зображення)"""
    from benvid.video import QUALITY, _x264_args
    big = [float(_x264_args(q, 2560 * 1080)[3]) for q in QUALITY.values()]
    assert big == sorted(big) and len(set(big)) == 3 and big[0] >= 26, big
    small = [float(_x264_args(q, 640 * 360)[3]) for q in QUALITY.values()]
    assert small == sorted(small) and len(set(small)) == 3 and small[0] <= 16 and all(a < b for a, b in zip(small, big)), (small, big)
    assert "aq-mode=0" in _x264_args(16, 640 * 360) and "aq-mode=0" not in _x264_args(16, 2560 * 1080), "малий кадр - без adaptive-quant, великий - з ним"
    mid = [float(_x264_args(16, n)[3]) for n in (0.5e6, 1e6, 1.4e6, 2e6)]
    assert mid == sorted(mid), "між малим і великим crf росте плавно"
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.mp4")
        make_video(src, n=24)
        sizes = {}
        for q in QUALITY:
            out = os.path.join(d, q + ".mp4")
            export_video(src, out, Params(method=3), "MP4 (H.264)", "360p", q, workers=1)
            sizes[q] = os.path.getsize(out)
        assert sizes["Висока"] > sizes["Середня"] > sizes["Мала"], sizes
        cap = cv2.VideoCapture(os.path.join(d, "Висока.mp4"))
        ok, fr = cap.read()
        cap.release()
        g = fr[..., 1].astype(np.float32)
        assert ok and float(((g > 40) & (g < 215)).mean()) < 0.1, "«сірих» пікселів на «Високій» мало"


def test_export_all_formats():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.mp4")
        make_video(src)
        info = video_info(src)
        assert info["frames"] == 40
        for fmt in FORMATS:
            ext, codec = FORMATS[fmt]
            out = os.path.join(d, "o_" + codec + ext)
            prog = []
            r = export_video(src, out, Params(), fmt, "360p", "Середня", workers=2,
                             progress=lambda f, l: prog.append(f))
            assert r["frames"] == 40 and prog[-1] == 1.0 and r["size"] == (640, 360), (fmt, r)
            if codec == "png":
                assert len(os.listdir(out)) == 40
            elif codec == "json":
                from benvid.pack import read_json_video
                m, fr = read_json_video(out)
                assert m["frames"] == 40 and abs(m["fps"] - 23.976) < 0.01 and fr[0].shape == (360, 640)
            elif codec == "ikr":
                from benvid.bake import open_baked
                rd = open_baked(out)
                try:
                    assert (rd.w, rd.h, rd.frames) == (640, 360, 40) and abs(rd.fps - 23.976) < 0.01
                finally:
                    rd.close()
            elif codec != "gif":
                cap = cv2.VideoCapture(out)
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                assert abs(n - 40) <= 1 and abs(cap.get(cv2.CAP_PROP_FPS) - 23.976) < 0.05, (fmt, n)
                ok, fr = cap.read()
                cap.release()
                assert ok and fr.shape[:2] == (360, 640)


def test_parallel_matches_sequential():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.mp4")
        make_video(src, n=50)
        outs = []
        for wk in (1, 3):
            out = os.path.join(d, f"o{wk}")
            export_video(src, out, Params(), "PNG-послідовність", "360p", workers=wk)
            outs.append([cv2.imread(os.path.join(out, f), 0) for f in sorted(os.listdir(out))])
        assert len(outs[0]) == len(outs[1]) == 50
        assert all((a == b).all() for a, b in zip(*outs))


def test_cancel_removes_partial():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.mp4")
        make_video(src, n=200)
        ev = threading.Event()
        ev.set()
        out = os.path.join(d, "x.mp4")
        try:
            export_video(src, out, Params(), "MP4 (H.264)", "360p", workers=2, cancel=ev)
            raise AssertionError("мало бути Cancelled")
        except Cancelled:
            pass
        assert not os.path.exists(out)


if __name__ == "__main__":
    test_helpers(); test_export_all_formats(); test_parallel_matches_sequential(); test_cancel_removes_partial()
    print("ok")


def test_export_range():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.mp4")
        make_video(src)
        out = os.path.join(d, "r")
        r = export_video(src, out, Params(), "PNG-послідовність", "144p", workers=2, start=10, end=25)
        assert r["frames"] == 15 and len(os.listdir(out)) == 15
        out2 = os.path.join(d, "r.mp4")
        r = export_video(src, out2, Params(), "MP4 (H.264)", "144p", workers=1, start=5, end=20)
        assert r["frames"] == 15


def test_workers_for_scales_with_frame_size():
    from benvid.video import workers_for
    small, big = workers_for(320, 180, 16), workers_for(3840, 2160, 16)
    assert 1 <= big <= small <= 16
    assert workers_for(3840, 2160, 1) == 1


def test_export_unpacks_frame_by_frame_with_the_right_polarity(tmp_path):
    """експорт розпаковує упаковані шматки по одному кадру (без величезних тимчасових масивів) - результат той самий, що з масок"""
    from benvid import video
    src = str(tmp_path / "s.mp4")
    make_video(src, n=14, w=250, h=180)
    size = out_size(250, 180, "360p")
    assert size[0] % 8 != 0                                                   # рядок не кратний 8 пікселям: найкапризніший випадок
    ref = np.concatenate(list(video.binarize_video(src, Params(), size, workers=1)))
    pk = np.concatenate(list(video.packed_video(src, Params(), size, workers=1)))
    assert pk.shape[-1] == (size[0] + 7) // 8 and np.array_equal(np.unpackbits(pk, axis=-1, count=size[0]), ref)
    for pal, expect in ((None, ref * 255), ("Класика~", 255 - ref * 255)):
        out = str(tmp_path / ("o" + str(len(pal or ""))))
        export_video(src, out, Params(), "PNG-послідовність", "360p", workers=1, palette=pal)
        got = np.stack([cv2.imdecode(np.fromfile(os.path.join(out, f), np.uint8), cv2.IMREAD_GRAYSCALE) for f in sorted(os.listdir(out))])
        assert np.array_equal(got, expect), pal
    if video.find_ffmpeg():
        mp4 = str(tmp_path / "o.mp4")
        export_video(src, mp4, Params(), "MP4 (H.264)", "360p", "Висока", workers=1)
        cap = cv2.VideoCapture(mp4)
        ok, fr = cap.read()
        cap.release()
        assert ok and fr.shape[:2] == (size[1], size[0])
        agree = np.mean((fr[..., 1] > 127) == (ref[0] > 0))
        assert agree > 0.97, agree                                            # кодек не перевертає й не зсуває картинку


def test_palette_exports(tmp_path):
    import numpy as np
    import cv2
    from benvid import palette
    from benvid.media import save_still
    m = (np.indices((20, 30)).sum(0) % 2).astype(np.uint8)
    a = palette.paint(m, "Сепія")
    assert a.shape == (20, 30, 3) and len({tuple(x) for x in a.reshape(-1, 3)}) == 2
    out = str(tmp_path / "p.png")
    save_still(m, out, "Синька")
    img = cv2.imdecode(np.fromfile(out, np.uint8), cv2.IMREAD_COLOR)
    assert len({tuple(x) for x in img.reshape(-1, 3)}) == 2
    assert palette.is_classic(None) and palette.is_classic("Класика") and not palette.is_classic("Сепія")


def test_damaged_png_is_recovered(tmp_path):
    import numpy as np
    import cv2
    from benvid.media import imread_u
    a = (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", a, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    raw = bytearray(buf.tobytes())
    i = raw.index(b"IDAT") + 5000
    raw[i:i + 4] = b"\x00\x01\x02\x03"                      # псуємо дані всередині IDAT -> помилка CRC
    p = tmp_path / "Скриня.png"
    p.write_bytes(bytes(raw))
    notes = []
    img = imread_u(str(p), notes.append)
    assert img.shape == (200, 300, 3) and notes


def test_export_pause_holds_then_resumes_and_can_be_cancelled(tmp_path):
    """пауза рендеру: після поточного шматка нічого нового не береться (прогрес стоїть), зняття паузи - і експорт добігає кінця;
    скасування працює й на паузі"""
    from benvid.video import wait_pause
    src = str(tmp_path / "s.mp4")
    make_video(src, n=60)
    pause, prog, res = threading.Event(), [], {}
    pause.set()

    def run():
        try:
            res["r"] = export_video(src, str(tmp_path / "o"), Params(), "PNG-послідовність", "144p", workers=1, pause=pause,
                                    progress=lambda f, l: prog.append(f))
        except Exception as e:  # noqa: BLE001
            res["e"] = e

    t = threading.Thread(target=run)
    t.start()
    time.sleep(1.2)
    held = len(prog)
    assert t.is_alive() and 0 < prog[-1] < 1.0, prog[-3:]
    time.sleep(0.6)
    assert len(prog) == held and t.is_alive(), "на паузі прогрес не повинен рухатись"
    pause.clear()
    t.join(60)
    assert not t.is_alive() and "e" not in res and res["r"]["frames"] == 60 and prog[-1] == 1.0

    ev = threading.Event()
    pause.set()
    t0 = time.time()
    th = threading.Timer(0.3, ev.set)
    th.start()
    try:
        wait_pause(pause, ev)
        raise AssertionError("мало бути Cancelled")
    except Cancelled:
        assert time.time() - t0 < 2
    wait_pause(None)                                             # без події - просто повертається
    pause.clear()
    wait_pause(pause)


def test_export_progress_clock_excludes_paused_time(monkeypatch):
    from benvid import progress
    now = [1000.0]
    monkeypatch.setattr(progress.time, "time", lambda: now[0])
    c = progress.Clock()
    now[0] += 10
    assert c.elapsed() == 10
    c.hold(True)
    now[0] += 100                                                # година відпочинку не рахується
    assert c.elapsed() == 10
    c.hold(True)                                                 # повторна пауза нічого не ламає
    now[0] += 5
    c.hold(False)
    c.hold(False)
    assert c.elapsed() == 10
    now[0] += 10
    assert c.elapsed() == 20
    assert c.eta(0.5) == 20 and c.eta(0.01) is None


def _sleepy(_):
    import time
    time.sleep(60)


def test_kill_pool_ends_busy_workers_at_once():
    """старий пул не повинен доробляти шматок і тримати пам'ять, поки піднімається новий (MemoryError при старті рендеру на 4K)"""
    import time
    from concurrent.futures import ProcessPoolExecutor
    from benvid.video import kill_pool, pool_kwargs
    ex = ProcessPoolExecutor(2, **pool_kwargs())
    futs = [ex.submit(_sleepy, i) for i in range(2)]
    time.sleep(2.0)
    procs = list(ex._processes.values())
    assert procs and all(p.is_alive() for p in procs)
    t = time.time()
    kill_pool(ex)
    # join() тут ненадійний: менеджер пулу теж забирає код завершення (waitpid), і той, хто програв гонку, повертається одразу з is_alive()=True
    while time.time() - t < 10 and any(p.is_alive() for p in procs):
        time.sleep(0.02)
    assert time.time() - t < 10 and not any(p.is_alive() for p in procs)
    del futs


def test_default_workers_leaves_room_on_many_cores(monkeypatch):
    """заміри: на 28 ядрах 13 процесів = 27 за швидкістю, але без 100% навантаження; до 8 ядер - усі крім одного"""
    from benvid import video
    monkeypatch.setattr(video, "ram_bytes", lambda: 64 << 30)
    for cpu, want in ((1, 1), (2, 1), (4, 3), (8, 7), (12, 7), (16, 9), (28, 15), (32, 17)):
        monkeypatch.setattr(video.os, "cpu_count", lambda c=cpu: c)
        assert video.default_workers() == want, cpu


def test_ordered_ramps_up_and_keeps_order(monkeypatch):
    """експорт: у польоті спершу 3 завдання (процеси пулу не стартують разом), результати - у порядку плану"""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from benvid import video
    run = [0, 0, []]                                            # зараз іде, найбільше за перші 0.7 с, моменти запуску
    lk = threading.Lock()
    t0 = [0.0]

    def job(a):
        with lk:
            run[0] += 1
            run[2].append(time.perf_counter())
            if time.perf_counter() - t0[0] < 0.7:
                run[1] = max(run[1], run[0])
        time.sleep(0.3)
        with lk:
            run[0] -= 1
        return a
    monkeypatch.setattr(video, "_job", job)
    plan = [("j", i) for i in range(12)]
    with ThreadPoolExecutor(12) as ex:
        t0[0] = time.perf_counter()
        out = list(video._ordered(ex, plan, 10))
    assert out == list(range(12))
    assert run[1] <= 3, run[1]                                  # перші 0.7 с - не більше 3 завдань одночасно
    with ThreadPoolExecutor(12) as ex:
        assert list(video._ordered(ex, plan, 10, ramp=False)) == list(range(12))


def test_ordered_does_not_starve_when_head_is_slow(monkeypatch):
    """голова черги повільна, решта вже готові: нові завдання все одно додаються (готове-невіддане не займає місць вікна)"""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from benvid import video
    started = []

    def job(a):
        started.append(a)
        time.sleep(1.5 if a == 0 else 0.05)
        return a
    monkeypatch.setattr(video, "_job", job)
    plan = [("j", i) for i in range(20)]
    with ThreadPoolExecutor(4) as ex:
        g = video._ordered(ex, plan, 4, ramp=False)
        first = next(g)
        assert first == 0
        assert len(started) == 20                               # поки голова рахувалась, решту 19 уже запущено й дороблено
        assert list(g) == list(range(1, 20))


def test_ordered_cancel_works_while_waiting_for_slow_chunk(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from benvid import video
    monkeypatch.setattr(video, "_job", lambda a: (time.sleep(5), a)[1])
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    with ThreadPoolExecutor(2) as ex:
        t = time.perf_counter()
        with pytest.raises(video.Cancelled):
            list(video._ordered(ex, [("j", 0), ("j", 1)], 2, ramp=False, cancel=cancel))
        assert time.perf_counter() - t < 2.0


def test_clock_speed_follows_recent_progress(monkeypatch):
    """швидкість і ETA - за останні секунди: довгий старт без кадрів не тримає «0 к/с» й не псує ETA на хвилини"""
    from benvid import progress
    now = [1000.0]
    monkeypatch.setattr(progress.time, "time", lambda: now[0])
    c = progress.Clock()
    for _ in range(80):                                          # 20 с старту без жодного кадру
        now[0] += 0.25
        assert c.eta(0.0) is None
    assert c.speed() is None
    f = 0.0
    for _ in range(60):                                          # далі 1% за секунду
        now[0] += 1.0
        f += 0.01
        eta = c.eta(f)
    assert abs(c.speed() - 0.01) < 1e-3
    assert abs(eta - (1 - f) / 0.01) < 1.5, "ETA - за поточною швидкістю, а не за середньою з урахуванням старту"
    avg = c.elapsed() * (1 - f) / f
    assert avg > eta + 10


def test_binarize_video_reports_progress_between_chunks(tmp_path):
    """смуга прогресу рухається й між шматками (господар обробки присилає кадри всередині завдань), ніколи не назад і не до 100% раніше часу"""
    import threading
    from concurrent.futures import Future
    from benvid import video as V

    class FakeEx:
        is_host = True

        def __init__(self):
            self.part = 0
            self.futs = []

        def partial_frames(self):
            return self.part

        def submit(self, fn, job):
            f = Future()
            self.futs.append((f, job))
            return f

        def shutdown(self, *a, **k):
            pass

    ex = FakeEx()
    plan = [("j", ("p", i * 10, 10, 8, 8, {})) for i in range(3)]
    seen = []
    g = V._ordered(ex, plan, window=3, ramp=False, on_wait=seen.append)

    def feed():
        import time
        time.sleep(0.4)
        ex.part = 4
        time.sleep(0.4)
        ex.part = 7
        time.sleep(0.4)
        for f, job in ex.futs:
            f.set_result(np.zeros((job[2], 8, 1), np.uint8))

    th = threading.Thread(target=feed)
    th.start()
    got = [len(x) for x in g]
    th.join()
    assert got == [10, 10, 10]
    assert 4 in seen and 7 in seen, seen
    assert seen == sorted(seen)


def test_export_survives_ffmpeg_dying(tmp_path, monkeypatch):
    """ffmpeg помер посеред експорту (закрили вікно, антивірус): зрозуміла помилка замість падіння, недописаного файла нема"""
    import stat
    from benvid import video
    src = str(tmp_path / "s.mp4")
    make_video(src, n=60)
    fake = tmp_path / "fakeff.py"
    fake.write_text("import sys\nsys.stdin.buffer.read(1000)\nsys.stderr.write('boom')\nsys.exit(1)\n")
    if os.name == "nt":                                              # на Windows «скрипт-ffmpeg» - це .bat
        sh = tmp_path / "fakeff.bat"
        sh.write_text(f'@"{sys.executable}" "{fake}" %*\r\n')
    else:
        sh = tmp_path / "fakeff"
        sh.write_text(f"#!/bin/sh\nexec {sys.executable} {fake} \"$@\"\n")
        sh.chmod(sh.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(video, "find_ffmpeg", lambda: str(sh))
    out = str(tmp_path / "o.mp4")
    try:
        export_video(src, out, Params(), "MP4 (H.264)", "360p", "Середня", workers=2)
    except RuntimeError as e:
        assert "boom" in str(e) or "ffmpeg" in str(e)
    else:
        raise AssertionError("очікували помилку")
    assert not os.path.exists(out)


def test_hidden_kw_hides_console():
    from benvid.video import hidden_kw
    kw = hidden_kw()
    if os.name == "nt":
        assert kw["creationflags"] and kw["startupinfo"].wShowWindow == 0
    else:
        assert kw["creationflags"] == 0


# ---------------------------------------------------------------- ліміт кадрів на секунду
def test_fps_options_and_ratio():
    from benvid.video import fps_options, fps_ratio, out_fps
    assert fps_options(60) == [50, 30, 25, 24, 20, 15, 12, 10]
    assert fps_options(29.97) == [25, 24, 20, 15, 12, 10], "29.97 -> «30» нічого б не змінило, його не пропонуємо"
    assert fps_options(23.976) == [20, 15, 12, 10] and fps_options(10) == [] and fps_options(0) == []
    assert fps_ratio(60, 30) == 2.0 and fps_ratio(30, 60) == 1.0 and fps_ratio(30, None) == 1.0 and fps_ratio(30, 0) == 1.0
    assert fps_ratio(30, 30) == 1.0 and fps_ratio(29.97, 30) == 1.0 and fps_ratio(30, "x") == 1.0
    assert out_fps(60, 30) == 30 and out_fps(25, 60) == 25 and out_fps(25, None) == 25


def test_parse_fps_limit_takes_a_typed_number_and_ignores_the_rest():
    """поле «кадри на секунду» приймає своє число (кома теж), а також текст пункту списку; без ефекту (не нижче за fps відео) - як в оригіналі"""
    from benvid.video import parse_fps_limit as P
    assert P("17", 60) == 17.0 and P(" 23,976 ", 60) == 23.976 and P("12.5", 30) == 12.5
    assert P("Кадри: 24 к/с", 60) == 24.0 and P("24 fps", 60) == 24.0
    assert P("", 60) == 0.0 and P(None, 60) == 0.0 and P("0", 60) == 0.0
    assert P("60", 60) == 0.0 and P("100", 60) == 0.0 and P("30", 29.97) == 0.0, "не нижче за fps відео - нічого б не змінило"
    assert P("Кадри: як в оригіналі (29.97 к/с)", 29.97) == 0.0
    assert P("0.3", 60) == 1.0, "менше за кадр на секунду не буває"
    assert P("abc", 60) is None and P("Кадри: як в оригіналі", 60) is None
    assert P("24", 0) == 0.0, "без відео ліміту немає"


def _png_bytes(folder):
    return [open(os.path.join(folder, n), "rb").read() for n in sorted(os.listdir(folder))]


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("limit,src_fps,n", [(10, 30, 60), (12, 30, 60), (24, 60, 60)])
def test_fps_limit_picks_frames_by_time_and_keeps_the_duration(tmp_path, workers, limit, src_fps, n):
    """кадр k результату - кадр int(k*ratio+0.5) оригіналу (рівень без пам'яті кадрів: біти ті самі, що й у повному експорті); тривалість лишається"""
    src = str(tmp_path / "s.mp4")
    make_video(src, n=n, fps=src_fps)
    full = str(tmp_path / "full")
    export_video(src, full, Params(method=3), "PNG-послідовність", "144p", workers=1)
    cut = str(tmp_path / "cut")
    r = export_video(src, cut, Params(method=3), "PNG-послідовність", "144p", workers=workers, fps=limit)
    ratio = src_fps / limit
    want = int(round(n / ratio))
    assert r["frames"] == want and r["fps"] == limit, r
    a, b = _png_bytes(full), _png_bytes(cut)
    assert len(b) == want
    assert [a[min(n - 1, int(k * ratio + 0.5))] for k in range(want)] == b
    assert abs(r["frames"] / r["fps"] - n / src_fps) <= 1.0 / limit + 1e-9, "тривалість не змінилась (з точністю до кадру)"


def test_fps_limit_with_range_and_no_effect_above_source(tmp_path):
    src = str(tmp_path / "s.mp4")
    make_video(src, n=60, fps=30)
    full = str(tmp_path / "full")
    export_video(src, full, Params(method=3), "PNG-послідовність", "144p", workers=1)
    a = _png_bytes(full)
    cut = str(tmp_path / "cut")
    r = export_video(src, cut, Params(method=3), "PNG-послідовність", "144p", workers=2, start=10, end=40, fps=15)
    assert r["frames"] == 15 and _png_bytes(cut) == [a[10 + 2 * k] for k in range(15)]
    same = str(tmp_path / "same")
    r = export_video(src, same, Params(method=3), "PNG-послідовність", "144p", workers=1, fps=60)
    assert r["frames"] == 60 and r["fps"] == 30 and _png_bytes(same) == a, "вищий за початковий ліміт нічого не змінює"


def test_fps_limit_mp4_has_the_lower_rate_and_the_same_length(tmp_path):
    src = str(tmp_path / "s.mp4")
    make_video(src, n=60, fps=30)
    out = str(tmp_path / "o.mp4")
    r = export_video(src, out, Params(method=3), "MP4 (H.264)", "144p", workers=1, fps=10)
    info = video_info(out)
    assert abs(info["fps"] - 10) < 0.5 and abs(info["frames"] - 20) <= 1 and r["frames"] == 20, (info, r)
