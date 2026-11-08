import os
import threading

import pytest

from benvid import single


def test_second_launch_hands_off_to_first():
    port, magic = 47912, "HANDOFF1"                # власний порт: справжній InkReel, що працює на машині, тесту не заважає
    got, ev = [], threading.Event()
    first = single.acquire("", port, magic)
    if not isinstance(first, single.Listener):     # порт зайнятий чимось стороннім - нічого перевіряти
        pytest.skip("порт зайнятий чимось стороннім")
    first.start(lambda p: (got.append(p), ev.set()))
    try:
        assert single.acquire("/tmp/x.mp4", port, magic) is False      # другий екземпляр: передав і виходить
        assert ev.wait(3) and got == ["/tmp/x.mp4"]
    finally:
        first.srv.close()


@pytest.mark.skipif(os.name == "nt", reason="TIME_WAIT блокує повторний bind лише на Linux/macOS (на Windows цього немає)")
def test_port_can_be_reused_right_after_the_previous_instance_closed():
    """сервер закриває з'єднання першим -> порт лишається в TIME_WAIT; новий екземпляр усе одно має стати слухачем,
    а не «працювати без черги» (так було б без SO_REUSEADDR на Linux/macOS)"""
    port, magic = 47913, "TIMEWAIT1"
    first = single.acquire("", port, magic)
    if not isinstance(first, single.Listener):
        pytest.skip("порт зайнятий чимось стороннім")
    got, ev = [], threading.Event()
    first.start(lambda p: (got.append(p), ev.set()))
    assert single.acquire("x", port, magic) is False
    assert ev.wait(3) and got == ["x"]
    first.close()
    again = single.acquire("", port, magic)
    try:
        assert isinstance(again, single.Listener)
    finally:
        if isinstance(again, single.Listener):
            again.close()


def test_live_listener_is_never_taken_over():
    port, magic = 47914, "LIVE1"
    first = single.acquire("", port, magic)
    if not isinstance(first, single.Listener):
        pytest.skip("порт зайнятий чимось стороннім")
    try:
        first.start(lambda p: None)
        assert single.acquire("y", port, magic) is False          # живий слухач не віддає порт: другий лише передає файл
    finally:
        first.srv.close()


def test_closed_listener_stops_answering():
    """після close() новий екземпляр не «передає файл мертвому слухачеві» (інакше файл загубиться), а стає слухачем сам"""
    port, magic = 47915, "CLOSED1"
    first = single.acquire("", port, magic)
    if not isinstance(first, single.Listener):
        pytest.skip("порт зайнятий чимось стороннім")
    got = []
    first.start(got.append)
    first.close()
    first._t.join(2)
    assert not first._t.is_alive(), "потік слухача має завершитись, а не висіти в accept()"
    assert single._handoff("late", port, magic) is False
    second = single.acquire("late", port, magic)
    try:
        assert isinstance(second, single.Listener) and got == []
    finally:
        if isinstance(second, single.Listener):
            second.close()


def test_handoff_needs_confirmation_from_the_window():
    """«OK» іде після того, як вікно обробило файл; вікно, що не відповідає (закрите, але процес ще живий), файл не «приймає» -
    новий запуск не зникає мовчки, а відкриває своє вікно"""
    port, magic = 47916, "CONFIRM1"
    first = single.acquire("", port, magic)
    if not isinstance(first, single.Listener):
        pytest.skip("порт зайнятий чимось стороннім")
    try:
        first.start(lambda p: False)                                 # вікно не змогло показати файл
        assert single._handoff("a", port, magic) is False
        assert single.acquire("a", port, magic) is None              # слухача немає, порт зайнятий -> працюємо самі
        got = []

        def slow(p):
            threading.Event().wait(0.4)                              # вікно відповідає не одразу, але відповідає
            got.append(p)
        first.start(slow)
        assert single._handoff("b", port, magic) is True and got == ["b"]
        first.start(lambda p: 1 / 0)                                 # збій у вікні - теж не підтвердження
        assert single._handoff("c", port, magic) is False
    finally:
        first.close()


def test_another_windows_user_does_not_hand_off_to_my_window(monkeypatch):
    """порт локальний для всього ПК: другий користувач (швидке перемикання) не повинен «передати» файл вікну першого, якого не бачить, - він запускає своє"""
    port, magic = 47915, "USERS1"
    monkeypatch.setattr(single, "_tag", lambda: "aaaaaa")
    first = single.acquire("", port, magic)
    if not isinstance(first, single.Listener):
        pytest.skip("порт зайнятий чимось стороннім")
    got = []
    first.start(lambda p: got.append(p))
    try:
        monkeypatch.setattr(single, "_tag", lambda: "bbbbbb")                  # інший користувач
        assert single.acquire("/tmp/theirs.mp4", port, magic) is None          # передати нікому: працює сам
        assert got == []
        monkeypatch.setattr(single, "_tag", lambda: "aaaaaa")                  # той самий користувач
        assert single.acquire("/tmp/mine.mp4", port, magic) is False
    finally:
        first.srv.close()


def test_user_tag_is_short_and_stable():
    t = single._tag()
    assert len(t) == 6 and t == single._tag() and all(c in "0123456789abcdef" for c in t)
