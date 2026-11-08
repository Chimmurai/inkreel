import wave

import numpy as np

from benvid.audio import AudioOut, SR
from benvid.player import Player


def test_volume_curve_and_clamp():
    a = AudioOut()
    a.set_volume(1.0)
    assert a.volume == 1.0
    a.set_volume(0.5)
    assert abs(a.volume - 0.25) < 1e-9
    a.set_volume(-3)
    assert a.volume == 0.0
    a.set_volume(9)
    assert a.volume == 1.0


def test_sounddevice_callback_applies_volume_and_mute():
    a = AudioOut()
    a.pcm = np.full((4096, 2), 8000, np.int16)
    a.on = True
    out = np.zeros((1024, 2), np.int16)
    a.set_volume(0.5)
    a._cb2(out, 1024)
    assert out.max() == 2000                       # 8000 * 0.25
    a.pos = 0
    a.muted = True
    a._cb2(out, 1024)
    assert out.max() == 0


def test_winsound_fallback_scales_the_wave_data(tmp_path):
    played = {}

    class FakeWS:
        SND_FILENAME, SND_ASYNC, SND_PURGE = 1, 2, 4

        @staticmethod
        def PlaySound(path, flags):
            played["path"], played["flags"] = path, flags

    a = AudioOut()
    a._sd, a._ws = None, FakeWS
    a.pcm = np.full((SR * 2, 2), 10000, np.int16)
    a.set_volume(0.5)
    assert a.backend == "ws" and a.play_from(0.0)
    with wave.open(played["path"]) as w:
        data = np.frombuffer(w.readframes(1000), np.int16)
    assert abs(int(data.max()) - 2500) <= 1         # 10000 * 0.25
    a.close()


def test_player_volume_restarts_only_winsound_on_release():
    p = Player(workers=1)
    try:
        p._aud_on = True
        p.audio._sd, p.audio._ws = None, object()           # «winsound»
        p.set_volume(0.3)                                  # під час перетягування - без перезапуску
        assert p._aud_on and abs(p.audio.volume - 0.09) < 1e-9
        p.set_volume(0.3, restart=True)
        assert not p._aud_on
        p._aud_on = True
        p.audio._sd = object()                              # sounddevice: гучність міняється на льоту, перезапуск не потрібен
        p.set_volume(0.8, restart=True)
        assert p._aud_on
    finally:
        p.close()
