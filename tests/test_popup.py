"""Де з'являються підказки: на моніторі під курсором, у межах його робочої області - і на другому екрані праворуч/ліворуч/вище за
основний (раніше підказка притискалась до краю ПЕРШОГО екрана й вилітала за вікно)."""
from __future__ import annotations

import pytest

from benvid import popup

# основний монітор 1920x1080 у (0,0); другий праворуч 2560x1440 у (1920,0); третій ліворуч 1280x1024 у (-1280, 56); четвертий над основним
PRIMARY = (0, 0, 1920, 1040)                  # без панелі завдань
RIGHT = (1920, 0, 4480, 1400)
LEFT = (-1280, 56, 0, 1080)
ABOVE = (0, -900, 1600, 0)


@pytest.mark.parametrize("area", [PRIMARY, RIGHT, LEFT, ABOVE])
def test_tip_follows_the_pointer_on_every_monitor(area):
    l, t, r, b = area
    px, py = (l + r) // 2, (t + b) // 2
    x, y = popup.near_pointer(px, py, 200, 60, area)
    assert (x, y) == (px + 14, py + 22)                                   # посередині монітора - праворуч-нижче за курсор
    assert l <= x and x + 200 <= r and t <= y and y + 60 <= b


@pytest.mark.parametrize("area", [PRIMARY, RIGHT, LEFT, ABOVE])
def test_tip_at_the_edges_stays_on_the_same_monitor(area):
    l, t, r, b = area
    w, h = 220, 80
    for px, py in ((r - 2, b - 2), (l + 1, t + 1), (r - 2, t + 1), (l + 1, b - 2), (r - 40, (t + b) // 2)):
        x, y = popup.near_pointer(px, py, w, h, area)
        assert l <= x <= r - w, (area, px, py, x)
        assert t <= y <= b - h, (area, px, py, y)
    x, y = popup.near_pointer(r - 5, b - 5, w, h, area)                    # у правому нижньому куті: притиснуто вліво й підняте над курсором
    assert x + w <= r and y + h <= b - 5 + 1


def test_tip_does_not_cover_the_pointer_when_flipped_up():
    x, y = popup.near_pointer(500, 1030, 200, 60, PRIMARY)
    assert y + 60 <= 1030                                                  # над курсором, а не під ним


def test_oversized_popup_is_pinned_to_the_top_left():
    x, y = popup.near_pointer(50, 50, 5000, 4000, PRIMARY)
    assert (x, y) == (4, 4)


def test_bubble_goes_under_the_widget_or_above_it():
    x, y = popup.under_widget(2000, 100, 40, 30, 200, 60, RIGHT)           # кнопка на другому моніторі
    assert RIGHT[0] <= x and x + 200 <= RIGHT[2] and y == 100 + 30 + 6
    x, y = popup.under_widget(2000, 1390, 40, 30, 200, 60, RIGHT)          # знизу немає місця - над кнопкою
    assert y + 60 <= 1390
    x, y = popup.under_widget(4470, 100, 40, 30, 300, 60, RIGHT)           # біля правого краю - притиснуто всередину
    assert x + 300 <= RIGHT[2]


def test_work_area_falls_back_to_the_virtual_screen_off_windows(monkeypatch):
    class W:
        def winfo_vrootx(self): return -1280
        def winfo_vrooty(self): return 0
        def winfo_vrootwidth(self): return 4480
        def winfo_vrootheight(self): return 1440
        def winfo_screenwidth(self): return 1920
        def winfo_screenheight(self): return 1080
    monkeypatch.setattr(popup, "_monitor_work_area", lambda x, y: None)
    assert popup.work_area(W(), 2500, 300) == (-1280, 0, 3200, 1440)        # увесь віртуальний екран, а не лише перший

    class Broken(W):
        def winfo_vrootwidth(self): raise RuntimeError
    assert popup.work_area(Broken(), 10, 10) == (0, 0, 1920, 1080)          # зовсім нічого - перший екран, як було


def test_work_area_prefers_the_monitor_under_the_point(monkeypatch):
    monkeypatch.setattr(popup, "_monitor_work_area", lambda x, y: RIGHT if x >= 1920 else PRIMARY)

    class W:
        def winfo_screenwidth(self): return 1920
        def winfo_screenheight(self): return 1080
    assert popup.work_area(W(), 3000, 200) == RIGHT
    assert popup.work_area(W(), 100, 200) == PRIMARY
