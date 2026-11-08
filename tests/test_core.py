import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid.core import Params, binarize, to_image  # noqa: E402
from benvid.video import out_size, parse_size  # noqa: E402

SIZES_TEST = ((16, 32), (32, 16), (2, 2), (64, 36), (144, 256), (256, 144), (426, 240), (640, 360),
              (1280, 720), (2048, 1152), (3000, 40))


def _img(seed=0, w=640, h=360):
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 8)
    cv2.rectangle(img, (50, 50), (200, 150), (255, 255, 255), -1)
    return img


def test_all_sizes_binary_deterministic():
    img = _img()
    for w, h in SIZES_TEST:
        m = binarize(img, w, h)
        assert m.shape == (h, w) and set(np.unique(m)) <= {0, 1}, (w, h)
        assert (m == binarize(img, w, h)).all(), (w, h)
        assert set(np.unique(to_image(m))) <= {0, 255}          # жодного сірого


def test_invert_and_gray_input():
    img = _img(1)
    a = binarize(img, 320, 180)
    assert (a ^ binarize(img, 320, 180, Params(invert=True))).all()
    assert binarize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 320, 180).shape == (180, 320)


def _ramp_frame(w=640, h=360):
    """чорне | плавний градієнт | біле"""
    img = np.zeros((h, w), np.uint8)
    img[:, 400:] = 255
    img[:, 200:400] = np.linspace(0, 255, 200, dtype=np.float32)[None, :].astype(np.uint8)
    return img


def test_gray_conveyed_by_density_at_all_scales():
    for w, h in ((640, 360), (256, 144), (1280, 720)):
        m = binarize(_ramp_frame(), w, h)
        x0, x1 = int(w * 200 / 640), int(w * 400 / 640)
        assert m[:, :x0 - 5].mean() < 0.03 and m[:, x1 + 5:].mean() > 0.97, (w, h, "чорне й біле суцільні")
        n = 5
        edges = np.linspace(x0, x1, n + 1).astype(int)
        dens = [m[:, edges[i]:edges[i + 1]].mean() for i in range(n)]
        assert dens[-1] > dens[0] + 0.3 and all(b >= a - 0.15 for a, b in zip(dens, dens[1:])), (w, h, dens)


def test_flat_frames():
    for v in (0, 255):
        assert len(np.unique(binarize(np.full((360, 640), v, np.uint8), 640, 360))) == 1
    m = binarize(np.full((360, 640), 128, np.uint8), 640, 360)   # рівний сірий = густина ~50%
    assert 0.3 < m.mean() < 0.7


def test_params_roundtrip():
    p = Params(period=5.0, invert=True)
    assert Params.from_dict(p.to_dict()) == p
    assert Params.from_dict({"period": "x", "foo": 1}) == Params()


def test_scale_invariance_of_look():
    """той самий кадр на 360p і 720p дає схожу світлоту (порівнюємо розмиті густини)"""
    img = _img(2, 1280, 720)
    a = binarize(img, 1280, 720).astype(np.float32)
    b = cv2.resize(binarize(img, 640, 360).astype(np.float32), (1280, 720), interpolation=cv2.INTER_LINEAR)
    da, db = (cv2.GaussianBlur(x, (0, 0), 10) for x in (a, b))
    assert np.abs(da - db).mean() < 0.12


def test_sizes():
    assert out_size(1920, 1080, "144p") == (256, 144)
    assert out_size(1920, 1080, "Як у джерела") == (1920, 1080)
    assert out_size(1920, 1080, "16x32") == (16, 32)
    assert out_size(1920, 1080, "320x") == (320, 180)
    assert out_size(1920, 1080, "x240") == (426, 240)
    assert out_size(1920, 1080, "9000x9000") == (4096, 4096)
    assert parse_size("720p") == (None, None) and parse_size("16×32") == (16, 32)


if __name__ == "__main__":
    test_all_sizes_binary_deterministic(); test_invert_and_gray_input(); test_gray_conveyed_by_density_at_all_scales()
    test_flat_frames(); test_params_roundtrip(); test_scale_invariance_of_look(); test_sizes()
    print("ok")


def test_text_on_dark_background_is_clean():
    import cv2
    img = np.full((720, 1280, 3), (70, 60, 50), np.uint8)
    cv2.putText(img, "Quick brown fox jumps", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (240, 240, 240), 3, cv2.LINE_AA)
    m = binarize(img, 854, 480, Params())
    bg = m[60:110, 500:800]                      # фон поруч із текстом
    assert bg.mean() < 0.03, bg.mean()           # без розсипу крапок
    assert m[120:160, 40:420].mean() > 0.12      # літери на місці


def test_threshold_tiles_do_not_repeat_exactly():
    """на великих кадрах плитка порогу не повинна давати точного повтору (квадратна сітка)"""
    import numpy as np
    from benvid.core import Params, binarize
    img = np.full((1080, 1920, 3), 60, np.uint8)
    m = binarize(img, 1920, 1080, Params()).astype(np.float32)
    for lag in (128, 256):
        a, b = m[:, :-lag] - m.mean(), m[:, lag:] - m.mean()
        assert abs(float((a * b).mean() / (a.std() * b.std() + 1e-9))) < 0.1


def test_methods_1_to_4_produce_distinct_binary_masks():
    import numpy as np
    from benvid.core import Params, binarize
    rng = np.random.default_rng(3)
    img = np.clip(np.linspace(30, 220, 320)[None, :, None] + rng.normal(0, 8, (180, 320, 3)), 0, 255).astype(np.uint8)
    outs = []
    for mth in (1, 2, 3, 4):
        m = binarize(img, 320, 180, Params(method=mth))
        assert m.shape == (180, 320) and set(np.unique(m)) <= {0, 1}
        outs.append(m)
    assert all((outs[i] != outs[j]).mean() > 0.01 for i in range(4) for j in range(i + 1, 4))


def test_method5_stable_and_inky():
    from benvid.core import InkState, Params, binarize
    rng = np.random.default_rng(1)
    base = (np.linspace(0, 255, 320)[None, :] * np.ones((200, 1))).astype(np.float32)
    fr = [np.dstack([np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)] * 3) for _ in range(8)]
    p = Params()
    p.method = 5
    st = InkState()
    ms = [binarize(f, 320, 200, p, {"state": st}).astype(int) for f in fr]
    flick = np.mean([np.abs(ms[i] - ms[i - 1]).mean() for i in range(1, 8)])
    assert flick < 0.05
    assert 0.1 < ms[0].mean() < 0.9


# ------------------------------------------------------------------ рівень 5 («Ультра», ink5.py)
def _l5(img, w, h, st=None, **kw):
    from benvid.core import InkState  # noqa: F401
    return binarize(img, w, h, Params(method=5, **kw), {"state": st} if st is not None else None)


def test_method5_all_sizes_binary_deterministic():
    img = _img()
    for w, h in SIZES_TEST:
        m = _l5(img, w, h)
        assert m.shape == (h, w) and set(np.unique(m)) <= {0, 1}, (w, h)
        assert (m == _l5(img, w, h)).all(), (w, h)


def test_method5_invert_gray_and_flat():
    img = _img(2)
    a = _l5(img, 640, 360)
    assert (a ^ _l5(img, 640, 360, invert=True)).all()
    assert _l5(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 640, 360).shape == (360, 640)
    for v in (0, 255):
        for w, h in ((640, 360), (256, 144)):
            assert len(np.unique(_l5(np.full((h, w), v, np.uint8), w, h))) == 1, (v, w, h)


def _noisy_seq(n=8, w=640, h=360, sigma=2.5, seed=4):
    rng = np.random.default_rng(seed)
    base = _img(seed, w, h).astype(np.float32)
    return [np.clip(base + rng.normal(0, sigma, base.shape), 0, 255).astype(np.uint8) for _ in range(n)]


def _flips(ms):
    return float(np.mean([(a != b).mean() for a, b in zip(ms[:-1], ms[1:])]))


def test_method5_static_noise_does_not_boil():
    """шум датчика на статичній сцені: рівень 5 мерехтить у рази менше за рівень 4, і на великих, і на малих кадрах"""
    from benvid.core import InkState
    for w, h in ((640, 360), (256, 144)):
        fr = _noisy_seq(8, w, h)
        st = InkState()
        m5 = [_l5(f, w, h, st) for f in fr]
        m4 = [binarize(f, w, h, Params(method=4)) for f in fr]
        assert _flips(m5) < 0.4 * _flips(m4), (w, h, _flips(m5), _flips(m4))
        assert _flips(m5) < 0.05, (w, h)


def test_method5_scene_cut_forgets_history():
    from benvid.core import InkState
    a = cv2.cvtColor(_ramp_frame(), cv2.COLOR_GRAY2BGR)                       # горизонтальний градієнт
    b = cv2.cvtColor(np.tile(np.linspace(255, 0, 360, dtype=np.float32)[:, None], (1, 640)).astype(np.uint8),
                     cv2.COLOR_GRAY2BGR)                                        # зовсім інша сцена: вертикальний
    cv2.circle(b, (320, 180), 70, (0, 0, 0), -1)
    st = InkState()
    for _ in range(3):
        _l5(a, 640, 360, st)
    assert (_l5(b, 640, 360, st) == _l5(b, 640, 360)).all()          # після склейки - як без пам'яті


def test_method5_chunk_warmup_matches_sequential():
    """шматок відео, що стартує з розігріву (як у воркері), майже не відрізняється від послідовної обробки"""
    from benvid import ink5
    from benvid.core import InkState
    w, h = 640, 360
    fr = _noisy_seq(14, w, h)
    st = InkState()
    seq = [_l5(f, w, h, st) for f in fr]
    st2 = InkState()
    p = Params(method=5)
    out = []
    for i in range(0, 14):
        if i < 8:
            if i < 7:
                ink5.warm(st2, fr[i], w, h, p)
            else:
                binarize(fr[i], w, h, p, {"state": st2})
            continue
        out.append(binarize(fr[i], w, h, p, {"state": st2}))
    assert all((a != b).mean() < 0.06 for a, b in zip(seq[8:], out))


def _detail_corr(m, t, k):
    g = lambda a, s: cv2.GaussianBlur(a, (0, 0), s * k)
    mm, tt = m.astype(np.float32), np.clip(t, 0, 1).astype(np.float32)
    bm, bt = g(mm, 0.7) - g(mm, 2.2), g(tt, 0.7) - g(tt, 2.2)
    return float(np.mean((bm - bm.mean()) * (bt - bt.mean())) / (bm.std() * bt.std() + 1e-9))


def _tone_rmse(m, t, k):
    g = lambda a: cv2.GaussianBlur(a, (0, 0), 2.5 * k)
    return float(np.sqrt(np.mean((g(m.astype(np.float32)) - g(np.clip(t, 0, 1).astype(np.float32))) ** 2)))


def _structured_frame(w=960, h=540):
    """плавний градієнт, текст, штрихова фактура вздовж кривих і зерниста ділянка - у одному кадрі"""
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = 40 + 120 * (0.5 + 0.5 * np.sin(xx / 170.0)) * (0.6 + 0.4 * np.cos(yy / 140.0))
    cx, cy = w * 0.28, h * 0.62                                  # «волосся»: смуги вздовж кіл
    r = np.hypot(xx - cx, yy - cy)
    hair = 110 + 90 * np.sin(r / 3.1 + 2.5 * rng.random((h, w)).astype(np.float32) * 0)
    mask = (r > 40) & (r < 190)
    img = np.where(mask, hair * (0.6 + 0.4 * np.exp(-((r - 115) / 70) ** 2)), img)
    rg = (xx > w * 0.62) & (yy > h * 0.55)
    img = np.where(rg, 120 + 70 * cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 2.0) * 3 - 70, img)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, "InkReel 5", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "small print 123", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def test_method5_not_worse_than_method4():
    """прямий ап четвертого: точність тону й дрібних деталей не гірші, а на фактурах - кращі"""
    from benvid.core import tone_map
    img = _structured_frame()
    for h in (540, 720):
        w = h * 16 // 9
        t, _ = tone_map(img, w, h, True)
        k = h / 720.0
        m4, m5 = binarize(img, w, h, Params(method=4)), _l5(img, w, h)
        assert _tone_rmse(m5, t, k) <= _tone_rmse(m4, t, k) * 1.1 + 1e-3, h
        assert _detail_corr(m5, t, k) >= _detail_corr(m4, t, k) - 0.02, h


def _snap_numpy(t, mid, hr, sp):
    d = np.maximum(0.45 * hr, 0.03)
    z = np.clip((t - (mid - d)) / (2 * d), 0.0, 1.0)
    return (t + sp * ((z * z * (3 - 2 * z)) - t)).astype(np.float32)


def test_snap_tail_is_bit_exact():
    """кінець _text_snap одним проходом (numba) дає ті самі біти, що ланцюжок numpy у float32: крайні випадки z<0, z>1, hr=0 теж"""
    from benvid import core
    if core._nb is None:
        return
    rng = np.random.default_rng(7)
    for h, w in ((540, 1280), (721, 1283), (144, 256), (3, 5), (1, 1)):
        for k in range(3):
            t = rng.random((h, w), dtype=np.float32)
            mid = (rng.random((h, w), dtype=np.float32) * 1.4 - 0.2).astype(np.float32)
            hr = (rng.random((h, w), dtype=np.float32) * (0.2 if k else 1.0)).astype(np.float32)
            hr[::3, ::2] = 0.0
            sp = rng.random((h, w), dtype=np.float32)
            assert np.array_equal(core._snap_tail_nb(t, mid, hr, sp), _snap_numpy(t, mid, hr, sp)), (h, w, k)


def test_text_snap_gives_the_same_frame_with_and_without_numba(monkeypatch):
    """весь _text_snap (з текстом у кадрі) - побітово той самий із ядром numba й без нього"""
    from benvid import core
    if core._nb is None:
        return
    w, h = 960, 540
    img = np.full((h, w, 3), 90, np.uint8)
    cv2.putText(img, "Hello, text 123", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (240, 240, 240), 6)
    cv2.putText(img, "second line", (80, 380), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 4)
    t = np.clip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0, 0, 1)
    a = core._text_snap(t.copy(), w, h, h / core.REF_H)
    assert not np.array_equal(a, t), "тексту в кадрі достатньо, щоб _text_snap спрацював"
    monkeypatch.setattr(core, "_nb", None)
    b = core._text_snap(t.copy(), w, h, h / core.REF_H)
    assert np.array_equal(a, b)
