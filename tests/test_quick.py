"""Швидка дія з Провідника (benvid/quick.py): імена результатів, налаштування, черга, справжнє перетворення фото й GIF."""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest
from PIL import Image

from benvid import i18n, quick, single


@pytest.fixture(autouse=True)
def _restore_lang():
    old = i18n.lang()
    yield
    i18n.set_lang(old)


# ---------------------------------------------------------------- імена файлів
def test_unique_path_next_to_source(tmp_path):
    src = tmp_path / "photo.jpg"
    src.write_bytes(b"x")
    p1 = quick.unique_path(str(src), ".png")
    assert p1 == str(tmp_path / "photo_inkreel.png")
    open(p1, "wb").close()
    p2 = quick.unique_path(str(src), ".png")
    assert p2 == str(tmp_path / "photo_inkreel (2).png")
    open(p2, "wb").close()
    assert quick.unique_path(str(src), ".png") == str(tmp_path / "photo_inkreel (3).png")
    assert quick.unique_path(str(src), ".gif") == str(tmp_path / "photo_inkreel.gif")          # інше розширення - вільне ім'я


def test_unique_path_handles_cyrillic_and_spaces(tmp_path):
    src = tmp_path / "Моє фото 1.png"
    src.write_bytes(b"x")
    assert os.path.basename(quick.unique_path(str(src), ".png")) == "Моє фото 1_inkreel.png"


def test_unreadable_folder_falls_back_to_pictures(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setattr(os, "access", lambda p, m: False)
    src = tmp_path / "ro" / "a.png"
    d = quick.out_dir_for(str(src))
    assert d.endswith(os.path.join("Pictures", "InkReel")) and os.path.isdir(d)


# ---------------------------------------------------------------- налаштування
def test_load_prefs_defaults_and_values():
    assert quick.load_prefs({"ui": {}}).method == 4
    p = quick.load_prefs({"ui": {"method": 5, "palette": "Сепія", "pal_inv": True, "lang": "en"}})
    assert (p.method, p.palette, p.lang) == (5, "Сепія~", "en")
    assert quick.load_prefs({"ui": {"method": 99}}).method == 6                                        # шостий - гілка «Нейро»
    assert quick.load_prefs({"ui": {"method": 6, "glyph": True}}).method == 5                           # у символах її нема
    assert quick.load_prefs({"ui": {"method": 0}}).method == 1
    assert quick.load_prefs({"ui": {"method": "сміття"}}).method == 4
    assert quick.load_prefs({"ui": {"palette": "Ризограф"}}).palette in ("Синька", "Синька~")        # стара назва
    assert quick.load_prefs({"ui": {"lang": "fr"}}).lang in ("uk", "en")                              # невідома мова -> за системою
    assert quick.load_prefs({"ui": "не словник"}).method == 4


# ---------------------------------------------------------------- черга
def _fake(log, fail=()):
    def conv(src, progress, cancel):
        log.append(os.path.basename(src))
        progress(0.5, "x")
        if os.path.basename(src) in fail:
            raise ValueError("боляче: " + os.path.basename(src))
        return src + ".out"
    return conv


def _drain(b):
    ev = []
    while not b.events.empty():
        ev.append(b.events.get())
    return ev


def test_batch_processes_in_order_and_reports():
    log = []
    b = quick.Batch(_fake(log), grace=0.05)
    for n in ("a.png", "b.png", "c.png"):
        b.add("/x/" + n)
    b.run()
    assert log == ["a.png", "b.png", "c.png"]
    assert [d[1] for d in b.done] == ["/x/a.png.out", "/x/b.png.out", "/x/c.png.out"]
    ev = _drain(b)
    assert [e[0] for e in ev if e[0] == "start"] == ["start"] * 3
    assert [e[1:] for e in ev if e[0] == "start"] == [("a.png", 1, 3), ("b.png", 2, 3), ("c.png", 3, 3)]
    assert ("progress", 0.5) in ev and sum(e[0] == "ok" for e in ev) == 3


def test_batch_error_does_not_stop_the_rest():
    log = []
    b = quick.Batch(_fake(log, fail=("b.png",)), grace=0.05)
    for n in ("a.png", "b.png", "c.png"):
        b.add("/x/" + n)
    b.run()
    assert log == ["a.png", "b.png", "c.png"]
    assert len(b.done) == 2 and [e[0] for e in b.errors] == ["/x/b.png"] and "боляче" in b.errors[0][1]
    assert any(e[0] == "err" for e in _drain(b))


def test_batch_cancel_stops_after_current():
    log = []
    b = quick.Batch(lambda s, p, c: (log.append(s), b.cancel.set(), s + ".out")[-1], grace=0.05)
    for n in "abc":
        b.add(n)
    b.run()
    assert log == ["a"] and len(b.done) == 1


def test_batch_stopped_exception_cancels():
    def conv(src, progress, cancel):
        raise quick.Stopped()
    b = quick.Batch(conv, grace=0.05)
    b.add("a")
    b.add("b")
    b.run()
    assert b.cancel.is_set() and not b.done and not b.errors


def test_batch_videos_are_set_aside_not_errors():
    def conv(src, progress, cancel):
        raise quick.VideoGiven(src)
    b = quick.Batch(conv, grace=0.05)
    b.add("clip.mp4")
    b.run()
    assert b.videos == ["clip.mp4"] and not b.errors and not b.done


def test_batch_ignores_empty_path_and_waits_for_latecomers():
    log = []
    b = quick.Batch(_fake(log), grace=0.6)
    b.add("")
    b.add("first.png")
    threading.Timer(0.25, lambda: b.add("late.png")).start()          # сусідній процес передав файл не одразу
    b.run()
    assert log == ["first.png", "late.png"] and b.total == 2


def test_work_posts_end_and_picks_up_last_second_files():
    log = []
    b = quick.Batch(_fake(log), grace=0.05)
    b.add("a")
    quick.work(b)
    assert log == ["a"] and _drain(b)[-1] == ("end",)


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_idle_wait_counts_from_last_arrival_and_start():
    clk = FakeClock()
    b = quick.Batch(lambda *a: "", grace=1.0, lead=3.0, clock=clk)
    assert b.idle_wait() == pytest.approx(3.0)                  # порожня черга: від старту lead
    clk.t += 2.0
    b.add("a")                                                  # файл на 2-й секунді: grace (до 3,0) не довший за lead (до 3,0)
    assert b.idle_wait() == pytest.approx(1.0)
    clk.t += 0.5
    b.add("b")                                                  # файл на 2,5-й: тепер grace від останнього файлу (до 3,5) довший
    assert b.idle_wait() == pytest.approx(1.0)
    clk.t += 0.7
    assert b.idle_wait() == pytest.approx(0.3)
    clk.t += 5.0
    assert b.idle_wait() == 0.0                                 # довге перетворення вже «відсиділо» очікування


def test_default_batch_has_no_lead_and_ignored_paths_do_not_extend_wait():
    clk = FakeClock()
    b = quick.Batch(lambda *a: "", grace=0.5, clock=clk)
    clk.t += 10
    assert b.idle_wait() == 0.0
    b.add("")
    assert b.idle_wait() == 0.0
    assert quick.LEAD > quick.GRACE >= 1.0 and quick.LEAD <= 3.0     # хвіст після готового файлу залишається коротким


def test_batch_lead_keeps_listening_for_neighbours_even_if_first_file_is_instant():
    log = []
    b = quick.Batch(_fake(log), grace=0.02, lead=0.4)
    b.add("first.png")
    threading.Timer(0.25, lambda: b.add("late.png")).start()          # сусід запустився вже після того, як перший файл готовий
    t0 = time.monotonic()
    b.run()
    assert log == ["first.png", "late.png"] and b.total == 2
    assert time.monotonic() - t0 >= 0.4 - 0.02


def test_batch_without_new_files_ends_right_after_a_long_conversion():
    def slow(src, progress, cancel):
        time.sleep(0.35)
        return src + ".out"
    b = quick.Batch(slow, grace=0.2, lead=0.1)
    b.add("big.png")
    t0 = time.monotonic()
    b.run()
    assert len(b.done) == 1
    assert time.monotonic() - t0 < 0.35 + 0.15              # без зайвих grace після перетворення


# ---------------------------------------------------------------- кілька процесів: другий передає файл першому
def test_second_quick_process_hands_its_file_to_the_queue_leader():
    got, ev = [], threading.Event()
    lst = single.acquire("", quick.PORT + 100, "TESTQ")
    if not isinstance(lst, single.Listener):
        pytest.skip("порт зайнятий чимось стороннім")
    lst.start(lambda p: (got.append(p), ev.set()))
    try:
        assert single.acquire("/tmp/second.png", quick.PORT + 100, "TESTQ") is False
        assert ev.wait(3) and got == ["/tmp/second.png"]
        assert single.acquire("/tmp/x", quick.PORT + 100, "ІНШИЙ_ПАРОЛЬ") is None       # чужий канал не відповідає
    finally:
        lst.srv.close()


# ---------------------------------------------------------------- справжнє перетворення
def _photo(path, w=96, h=64):
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([x * 255 // w, y * 255 // h, (x + y) * 255 // (w + h)], -1).astype(np.uint8)
    Image.fromarray(a).save(path)


def test_convert_photo_classic_is_one_bit_png(tmp_path):
    src = tmp_path / "pic.jpg"
    _photo(src)
    out = quick.convert(str(src), quick.Prefs(2, "Класика"))
    assert out == str(tmp_path / "pic_inkreel.png") and os.path.getsize(out) > 0
    im = Image.open(out)
    assert im.mode == "1" and im.size == (96, 64)
    assert src.exists()                                                      # оригінал не чіпаємо


def test_convert_photo_colour_palette_has_two_colours(tmp_path):
    src = tmp_path / "pic.png"
    _photo(src)
    out = quick.convert(str(src), quick.Prefs(1, "Сепія"))
    im = Image.open(out).convert("RGB")
    cols = {c for _, c in im.getcolors(10000)}
    assert 1 <= len(cols) <= 2
    from benvid import palette
    ink, paper = palette.colors("Сепія")
    assert cols <= {tuple(ink), tuple(paper)}


def test_convert_twice_never_overwrites(tmp_path):
    src = tmp_path / "pic.png"
    _photo(src)
    a = quick.convert(str(src), quick.Prefs(1))
    b = quick.convert(str(src), quick.Prefs(1))
    assert a != b and os.path.exists(a) and os.path.exists(b) and b.endswith("(2).png")


def test_convert_gif_keeps_frames_and_durations(tmp_path, monkeypatch):
    from benvid import video
    monkeypatch.setattr(video, "default_workers", lambda: 1)                 # без породження процесів
    frames = []
    for i in range(4):
        a = np.full((48, 64, 3), 30 + i * 60, np.uint8)
        a[10:30, 10 + i * 8:30 + i * 8] = 240
        frames.append(Image.fromarray(a))
    src = tmp_path / "anim.gif"
    frames[0].save(src, save_all=True, append_images=frames[1:], duration=[80, 120, 80, 120], loop=0)
    prog = []
    out = quick.convert(str(src), quick.Prefs(1, "Класика"), progress=lambda f, l=None: prog.append(f))
    assert out == str(tmp_path / "anim_inkreel.gif")
    g = Image.open(out)
    assert g.n_frames == 4 and g.size == (64, 48)
    durs = []
    for i in range(g.n_frames):
        g.seek(i)
        durs.append(g.info.get("duration"))
    assert durs == [80, 120, 80, 120]
    assert prog and prog[-1] >= 0.9


def test_convert_video_is_refused_for_the_normal_window(tmp_path):
    src = tmp_path / "x.mp4"
    src.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    with pytest.raises(quick.VideoGiven):
        quick.convert(str(src), quick.Prefs())


def test_convert_error_leaves_no_partial_file(tmp_path):
    src = tmp_path / "broken.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"nonsense" * 10)
    with pytest.raises(Exception):
        quick.convert(str(src), quick.Prefs())
    assert not [f for f in os.listdir(tmp_path) if f.startswith("broken_inkreel")]


def test_convert_cancel_cleans_up(tmp_path, monkeypatch):
    from benvid import video
    monkeypatch.setattr(video, "default_workers", lambda: 1)
    frames = [Image.fromarray(np.full((32, 32, 3), 40 * i, np.uint8)) for i in range(5)]
    src = tmp_path / "a.gif"
    frames[0].save(src, save_all=True, append_images=frames[1:], duration=100, loop=0)
    ev = threading.Event()
    ev.set()
    with pytest.raises(quick.Stopped):
        quick.convert(str(src), quick.Prefs(1), cancel=ev)
    assert not [f for f in os.listdir(tmp_path) if "_inkreel" in f]


def test_convert_cancel_during_still_leaves_no_result(tmp_path, monkeypatch):
    """фото рахується одним махом: якщо «Скасувати» натиснули посеред цього, готовий файл не лишаємо"""
    from benvid import media
    src = tmp_path / "p.png"
    _photo(src)
    ev = threading.Event()
    real = media.export_still

    def slow_then_cancel(*a, **k):
        real(*a, **k)
        ev.set()                                           # користувач натиснув «Скасувати», поки йшло збереження
    monkeypatch.setattr(media, "export_still", slow_then_cancel)
    with pytest.raises(quick.Stopped):
        quick.convert(str(src), quick.Prefs(1), cancel=ev)
    assert not [f for f in os.listdir(tmp_path) if "_inkreel" in f]


def test_convert_without_cancel_event_still_works(tmp_path):
    src = tmp_path / "p.png"
    _photo(src)
    out = quick.convert(str(src), quick.Prefs(1), cancel=threading.Event())
    assert os.path.isfile(out) and out.endswith("p_inkreel.png")
