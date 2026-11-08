import os


def test_dnd_hook_is_noop_off_windows():
    from benvid import dnd
    if os.name != "nt":
        assert dnd.hook(object(), lambda p: None) is False
