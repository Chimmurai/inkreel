import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid import palette as P  # noqa: E402
from benvid.core import Params  # noqa: E402
from benvid.media import export_still, save_still  # noqa: E402


def _h(s):
    return tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))


def test_every_palette_is_distinct_readable_and_ink_is_darker():
    seen = set()
    for n in P.NAMES:
        ink, paper = P.colors(n)
        assert P._lum(ink) < P._lum(paper), n                  # правило: світле на фото лишається світлим
        assert P.contrast(ink, paper) >= 4.5, n                # палітру видно й читано
        for k in (n, P.key(n, True)):                          # і з «навпаки», коли інвертується весь інтерфейс
            t = P.theme(k)
            assert P.contrast(_h(t["FG"]), _h(t["BG"])) >= 4.5, k  # інтерфейс читабельний у будь-якій палітрі
            assert P.contrast(_h(t["DIM"]), _h(t["BG"])) >= 3.5, k
            assert P.contrast(_h(t["FG"]), _h(t["PANEL"])) >= 4.5, k
        assert (ink, paper) not in seen and (paper, ink) not in seen, f"{n}: дубль іншої палітри (інверсія - окрема кнопка)"
        seen.add((ink, paper))
    assert P.NAMES[0] == P.CLASSIC and len(P.NAMES) >= 15


def test_inversion_swaps_ink_and_paper_everywhere():
    for n in P.NAMES:
        k = P.key(n, True)
        assert P.colors(k) == P.colors(n)[::-1]
        assert P.pil_palette(k) == P.pil_palette(n)[3:] + P.pil_palette(n)[:3]
        m = (np.indices((8, 9)).sum(0) % 3 == 0).astype(np.uint8)
        assert (P.paint(m, k) == P.paint(1 - m, n)).all()      # інверсія = дзеркальна маска, але без перерахунку
        assert (P.lut(k, bgr=True) == P.lut(n, bgr=True)[::-1]).all()
        t0, t1 = P.theme(n), P.theme(k)                        # «навпаки» інвертує й саме вікно: тло і текст міняються місцями
        assert (t1["BG"], t1["FG"]) == (t0["FG"], t0["BG"]), n
        assert t1["PANEL"] != t0["PANEL"] and t1["DIM"] != t0["DIM"], n
    assert P.theme("Класика")["BG"] == "#000000" and P.theme("Класика~")["BG"] == "#ffffff"
    assert P.theme("Класика~")["FG"] == "#000000"              # біле вікно з чорним текстом
    assert P.split("Синька~") == ("Синька", True) and P.split("Синька") == ("Синька", False)
    assert P.split("немає такої~") == (P.CLASSIC, True) and P.split(None) == (P.CLASSIC, False)
    assert P.key("Бурштин", True) == "Бурштин~" and P.key("немає", False) == P.CLASSIC
    assert P.is_classic(None) and P.is_classic("Класика") and not P.is_classic("Класика~") and not P.is_classic("Сепія")
    assert P.is_negative("Класика~") and not P.is_negative("Класика") and not P.is_negative("Сепія~")


def test_old_settings_keep_their_look():
    assert P.migrate("Ризограф") == ("Синька", False)          # колишня синьо-біла пара - одна палітра
    assert P.migrate("Game Boy") == ("Ретро LCD", False)
    # колишня «Синька» (біла на синьому) = нинішня з «навпаки»; так само колишні «світні» палітри
    for n in ("Синька", "Бурштин", "Термінал", "Неон"):
        assert P.migrate(n) == (n, True), n
        assert P.migrate(n, False) == (n, False) and P.migrate(n, True) == (n, True)   # явний прапорець - головніший
    assert P.migrate("Сепія") == ("Сепія", False) and P.migrate("Сепія", True) == ("Сепія", True)
    assert P.migrate(None) == (P.CLASSIC, False) and P.migrate("щось не те", True) == (P.CLASSIC, True)


def _img(w=96, h=64):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = np.linspace(0, 255, w, dtype=np.uint8)[None, :, None]
    cv2.circle(img, (w // 2, h // 2), 18, (255, 255, 255), -1)
    return img


def _rd(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def test_still_exports_negative_classic_and_inverted_colours(tmp_path):
    m = (np.indices((20, 30)).sum(0) % 2).astype(np.uint8)
    a, b = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    save_still(m, a, "Класика")
    save_still(m, b, "Класика~")
    ia, ib = _rd(a), _rd(b)
    assert (ia == 255 - ib).all() and set(np.unique(ia)) == {0, 255}      # чисто чорно-біле з протилежною полярністю
    from PIL import Image
    assert Image.open(b).mode == "1"                                     # і лишається справжнім 1-бітним
    for ext in (".bmp", ".tif", ".webp"):
        c = str(tmp_path / ("c" + ext))
        save_still(m, c, "Класика~")
        assert (_rd(c) == ib).all(), ext
    s1, s2 = str(tmp_path / "s1.png"), str(tmp_path / "s2.png")
    save_still(m, s1, "Бурштин")
    save_still(m, s2, "Бурштин~")
    c1, c2 = _rd(s1), _rd(s2)
    ink, paper = P.colors("Бурштин")
    assert tuple(c1[0, 0][::-1]) == ink and tuple(c1[0, 1][::-1]) == paper
    assert tuple(c2[0, 0][::-1]) == paper and tuple(c2[0, 1][::-1]) == ink


def test_export_still_and_png_sequence_follow_inversion(tmp_path):
    src = str(tmp_path / "src.png")
    cv2.imwrite(src, _img())
    a, b = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    export_still(src, a, Params(method=3), "Як у джерела", palette="Туш і папір")
    export_still(src, b, Params(method=3), "Як у джерела", palette="Туш і папір~")
    ia, ib = _rd(a), _rd(b)
    ink, paper = P.colors("Туш і папір")
    assert {tuple(x[::-1]) for x in ia.reshape(-1, 3)} == {ink, paper}
    # те саме зображення, лише кольори міняються місцями
    swap = np.where((ia == np.array(ink[::-1], np.uint8)).all(-1, keepdims=True), np.array(paper[::-1], np.uint8),
                    np.array(ink[::-1], np.uint8))
    assert (swap == ib).all()


def _video(path, n=6):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (160, 96))
    for i in range(n):
        vw.write(np.roll(_img(160, 96), i * 4, axis=1))
    vw.release()


def test_video_negative_classic_stays_grey_and_is_exact_inverse(tmp_path):
    from benvid.video import export_video
    src = str(tmp_path / "v.mp4")
    _video(src)
    outs = {}
    for k in ("Класика", "Класика~", "Бурштин~"):
        d = str(tmp_path / ("f_" + str(len(outs))))
        r = export_video(src, d, Params(method=3), "PNG-послідовність", "144p", workers=1, palette=k)
        assert r["frames"] == 6
        outs[k] = [cv2.imdecode(np.fromfile(os.path.join(d, f), np.uint8), cv2.IMREAD_UNCHANGED) for f in sorted(os.listdir(d))]
    for a, b in zip(outs["Класика"], outs["Класика~"]):
        assert a.ndim == 2 and b.ndim == 2                     # лишається сірим каналом (швидкий шлях, без RGB)
        assert (a == 255 - b).all()
    ink, paper = P.colors("Бурштин~")                          # з «навпаки» ТУШ - світлий колір
    for f in outs["Бурштин~"]:
        cols = {tuple(x[::-1]) for x in f.reshape(-1, 3)}
        assert cols <= {ink, paper} and len(cols) == 2


def test_gif_export_inversion(tmp_path):
    from PIL import Image
    from benvid.media import export_gif_source
    frames = [Image.fromarray(np.roll(_img(80, 48), i * 3, axis=1)[..., ::-1]) for i in range(4)]
    g = str(tmp_path / "a.gif")
    frames[0].save(g, save_all=True, append_images=frames[1:], duration=80, loop=0)
    outs = {}
    for k in ("Кармін", "Кармін~"):
        o = str(tmp_path / (k + ".gif"))
        export_gif_source(g, o, Params(method=3), "GIF", "144p", workers=1, palette=k)
        im = Image.open(o)
        assert im.n_frames == 4
        outs[k] = np.array(im.convert("RGB"))
    ink, paper = P.colors("Кармін")
    assert {tuple(x) for x in outs["Кармін"].reshape(-1, 3)} <= {ink, paper}
    assert (outs["Кармін"][..., 0] != outs["Кармін~"][..., 0]).any()


def test_cli_palette_names_and_inversion(tmp_path):
    from benvid.cli import main
    src, o1, o2, o3 = (str(tmp_path / n) for n in ("a.png", "o1.png", "o2.png", "o3.png"))
    cv2.imwrite(src, _img())
    main([src, o1, "--size", "144p", "--palette", "Синька"])
    main([src, o2, "--size", "144p", "--palette", "Синька~"])
    main([src, o3, "--size", "144p", "--palette", "Ризограф"])            # стара назва = нинішня «Синька»
    a, b, c = _rd(o1), _rd(o2), _rd(o3)
    assert (a == c).all() and (a != b).any()
    with pytest.raises(SystemExit):
        main([src, o1, "--palette", "Не існує"])
