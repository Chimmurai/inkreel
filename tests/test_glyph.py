"""Режим «Символи»: кадр складається зі знаків шрифту. Перевіряємо алфавіт, форму результату, тон, пам'ять кадрів, текстовий вихід
і зв'язок із рештою програми (Params, CLI, експорт фото, вікно)."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid import core, glyph  # noqa: E402
from benvid.core import InkState, Params, binarize  # noqa: E402

SIZES = ((2, 2), (16, 32), (64, 36), (320, 180), (640, 352), (1280, 720), (3000, 40))


def _img(seed=0, w=640, h=360):
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 8)
    cv2.rectangle(img, (50, 50), (200, 150), (255, 255, 255), -1)
    cv2.circle(img, (420, 180), 70, (20, 20, 20), -1)
    return img


def _p(level, **kw):
    return Params(method=level, glyph=True, **kw)


# ---------------------------------------------------------------- банк знаків
def test_bank_is_real_ascii_in_consolas_cells():
    b = glyph.bank_for(1280, 720)
    assert (b.cw, b.ch) == (8, 16)
    assert b.ink.shape[1:] == (16, 8) and set(np.unique(b.ink)) <= {0, 1}
    for c in "@#%*+=-:.|/\\_`^~()[]{}<>!?;,\"'$&AaBbZz09":
        assert c in b.chars, c                                   # справжній ASCII, а не добірка: усі видимі знаки на місці
    assert b.ink[b.index[" "]].sum() == 0 and b.ink[b.index["█"]].all()
    assert b.group[b.index["█"]] == glyph.G_SOLID and all(b.group[b.index[c]] == glyph.G_PART for c in "▉▇" if c in b.index), \
        "повний █ - свій клас, неповні блоки (смугастість) - свій"
    assert b.group[b.index["@"]] != glyph.G_SOLID and b.group[b.index[" "]] == glyph.G_BLANK
    for c in "◘◙▣":                                               # «ящик із діркою»: у світлих полях давав ряди квадратів із крапкою
        if c in b.index:
            assert b.group[b.index[c]] == glyph.G_HOLE, c
    for c in "B8@DO0#MW":                                         # букви - не ящики
        assert b.group[b.index[c]] != glyph.G_HOLE, c
    assert len(b.chars) > 1000 and any(ord(c) > 0x370 for c in b.chars), "алфавіт - майже весь шрифт, не лише ASCII"
    keys = {k.tobytes() for k in b.ink}
    assert len(keys) == len(b.ink), "однакові картинки не дублюються"
    assert b.dens[b.index["@"]] > b.dens[b.index["."]] > b.dens[b.index[" "]]
    assert len(b.ramp) >= 8 and (np.diff(b.dens[b.ramp]) > 0).all()


def test_font_falls_back_to_the_bundled_one(monkeypatch):
    monkeypatch.setenv("WINDIR", "/nonexistent")
    monkeypatch.delenv("INKREEL_GLYPH_FONT", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    path, real = glyph.font_path()
    assert os.path.isfile(path) and not real and path.endswith(glyph.FALLBACK_FONT)
    monkeypatch.setenv("INKREEL_GLYPH_FONT", path)
    assert glyph.font_path()[0] == path


def test_cell_grid_follows_the_frame():
    assert glyph.cell_for(1280, 720) == (8, 16) and glyph.grid_for(1280, 720) == (45, 160), "типово - 8x16"
    cw, ch = glyph.cell_for(64, 36)
    assert cw == glyph.CW_MIN == 8 and ch == 16, "клітинка не вужча за 8 px: знак лишається знаком"
    for w, h in ((1280, 720), (1920, 1080), (3840, 2160), (7680, 4320)):
        assert [glyph.cell_for(w, h, d) for d in (1, 2, 3)] == [(32, 64), (16, 32), (8, 16)], "канонічні клітинки на будь-якому розмірі"


def test_small_frames_keep_the_chosen_coarseness_while_one_symbol_fits():
    """малі картинки: крупність міняється, доки в кадр уміщається хоч один знак (раніше клітинка сама меншала до 8x16 - позиції зливались)"""
    for w, h in ((256, 144), (320, 176), (426, 240), (640, 360)):
        assert [glyph.cell_for(w, h, d) for d in (1, 2, 3)] == [(32, 64), (16, 32), (8, 16)], (w, h)
    assert glyph.grid_for(256, 144, 1) == (3, 8), "256x144 при 32x64: 8 знаків по ширині, 3 рядки (останній неповний)"
    assert glyph.cell_for(32, 64, 1) == (32, 64) and glyph.grid_for(32, 64, 1) == (1, 1), "рівно один знак"
    assert glyph.cell_for(20, 40, 1) == (16, 32) and glyph.cell_for(10, 20, 1) == (8, 16), "кадр, менший за клітинку, що дійшов напряму (CLI): клітинка меншає вдвічі"


def test_symbols_result_is_never_smaller_than_one_symbol():
    """«Символи»: розмір результату не менший за одну клітинку (32x64, 16x32 або 8x16) - менший піднімається з пропорціями кадру"""
    from benvid.video import glyph_cell, out_size
    assert glyph_cell(Params(glyph=False)) is None and glyph_cell(None) is None
    assert [glyph_cell(Params(glyph=True, gdens=d)) for d in (1, 2, 3)] == [(32, 64), (16, 32), (8, 16)]
    assert out_size(1920, 1080, "144p") == (256, 144), "чорнило: без змін"
    assert out_size(1920, 1080, "144p", (32, 64)) == (256, 144), "уміщається - не чіпаємо"
    assert out_size(20, 20, "Як у джерела", (32, 64)) == (64, 64) and out_size(20, 20, "Як у джерела") == (20, 20), "малий кадр піднято: висота = клітинка, ширина за пропорціями"
    w, h = out_size(1000, 100, "50x", (32, 64))                          # свій розмір лише за шириною: пропорції зберігаються
    assert h >= 64 and abs(w / h - 10) < 0.3
    assert out_size(1920, 1080, "10x10", (32, 64)) == (32, 64), "явний ШxВ: кожна сторона окремо"
    assert out_size(1920, 1080, "100x30", (32, 64)) == (100, 64)
    assert out_size(1920, 1080, "16x32", (16, 32)) == (16, 32)


def test_player_lifts_the_size_to_one_symbol_when_coarseness_changes(tmp_path):
    from benvid.player import Player
    img = tmp_path / "tiny.png"
    cv2.imwrite(str(img), np.random.default_rng(2).integers(0, 255, (30, 40, 3), np.uint8))
    p = Player()
    p.params = Params(method=3, glyph=True, gdens=3)
    p.open(str(img))
    try:
        p.set_size("Як у джерела")
        assert p.size == (40, 30), "8x16 влізає у 40x30"
        pr = Params(method=3, glyph=True, gdens=1)
        p.set_params(pr)
        assert p.size[0] >= 32 and p.size[1] >= 64 and abs(p.size[0] / p.size[1] - 40 / 30) < 0.06, "крупніший знак піднімає розмір (32x64)"
        p.set_params(Params(method=3, glyph=False, gdens=1))
        assert p.size == (40, 30), "чорнило: розмір знову як є"
    finally:
        p.close()


@pytest.mark.parametrize("w,h,dens", ((4096, 2160, 1), (2560, 1080, 2), (3840, 2160, 1), (854, 480, 3), (643, 357, 3), (1720, 720, 2), (100, 70, 3), (1280, 720, 1),
                                      (640, 360, 3), (640, 360, 2), (1920, 1080, 3)))
def test_grid_is_uniform_on_any_frame_size(w, h, dens):
    """крок знаків однаковий по всьому кадру (рівна сітка): раніше на кадрі, не кратному клітинці, зайві пікселі розкидались по рядках -
    кожен ~3-й рядок стискався на 1 px, і візерунок ░▒▓ на їхніх межах зсувався (шви). Тепер неповним лишається лише останній рядок/стовпчик"""
    cw, ch = glyph.cell_for(w, h, dens)
    rows, cols = glyph.grid_for(w, h, dens)
    vh, vw = glyph.edge_sizes(w, h, cw, ch)
    assert len(vh) == rows and len(vw) == cols and vh.sum() == h and vw.sum() == w
    assert (vh[:-1] == ch).all() and (vw[:-1] == cw).all() and 0 < vh[-1] <= ch and 0 < vw[-1] <= cw
    b = glyph.bank_for(w, h, dens)
    idx = np.random.default_rng(3).integers(0, len(b.chars), (rows, cols))
    m = glyph.render_idx(idx, b, w, h)
    assert m.shape == (h, w)
    for r in (0, rows // 2, rows - 1):
        for c in (0, cols // 2, cols - 1):
            cell = m[r * ch:r * ch + vh[r], c * cw:c * cw + vw[c]]
            assert np.array_equal(cell, b.ink[idx[r, c]][:vh[r], :vw[c]]), (r, c)
    # один знак на всю сітку -> кадр суворо періодичний з кроком клітинки (без швів)
    ink = int(np.argmax([b.ink[i].mean() * (0.2 < b.ink[i].mean() < 0.8) for i in range(len(b.chars))]))
    mm = glyph.render_idx(np.full((rows, cols), ink), b, w, h)
    assert np.array_equal(mm[ch:], mm[:-ch]) and np.array_equal(mm[:, cw:], mm[:, :-cw])


def test_frames_divisible_by_the_cell_have_only_whole_cells():
    for w, h in ((1280, 720), (1920, 1088), (640, 352), (4096, 2160)):
        cw, ch = glyph.cell_for(w, h)
        rows, cols = glyph.grid_for(w, h)
        vh, vw = glyph.edge_sizes(w, h, cw, ch)
        assert (cols * cw, rows * ch) == (w, h) and (vh == ch).all() and (vw == cw).all()
        img = _img(1)
        assert glyph.canvas_img(img, w, h, cw, ch) is img


def test_canvas_image_pads_to_whole_cells_by_edge_replication():
    img = np.random.default_rng(5).integers(0, 255, (357, 643, 3), np.uint8)
    cw, ch = glyph.cell_for(643, 357, 3)
    out = glyph.canvas_img(img, 643, 357, cw, ch)
    rows, cols = glyph.grid_for(643, 357, 3)
    assert out.shape[:2] == (rows * ch, cols * cw) and np.array_equal(out[:357, :643], img)
    assert np.array_equal(out[-1, :643], img[-1]) and np.array_equal(out[:357, -1], img[:, -1])
    big = np.zeros((714, 1286, 3), np.uint8)                          # кадр у більшому розмірі, ніж вихід: добивка пропорційна
    o2 = glyph.canvas_img(big, 643, 357, cw, ch)
    assert abs(o2.shape[0] / big.shape[0] - rows * ch / 357) < 0.01 and abs(o2.shape[1] / big.shape[1] - cols * cw / 643) < 0.01


def test_text_from_mask_on_a_frame_not_divisible_by_the_cell():
    w, h = 643, 357
    img = _img(4)
    p = _p(4)
    idx, bank = glyph.plan(img, w, h, p)
    back = glyph.text_from_mask(glyph.render_idx(idx, bank, w, h), w, h).splitlines()
    for a, b in zip(glyph.to_text(idx, bank).splitlines(), back):
        assert len(a) == len(b) or abs(len(a) - len(b)) <= 3
    assert sum(x != y for a, b in zip(glyph.to_text(idx, bank).splitlines(), back) for x, y in zip(a, b)) < 0.01 * idx.size, \
        "різняться лише знаки, що після зрізу хвостового краю виглядають однаково"


# ---------------------------------------------------------------- вихід
@pytest.mark.parametrize("level", (1, 2, 3, 4, 5))
def test_every_level_every_size_binary_and_deterministic(level):
    img = _img()
    for w, h in SIZES:
        m = binarize(img, w, h, _p(level), {"state": InkState()} if level == 5 else None)
        assert m.shape == (h, w) and m.dtype == np.uint8 and set(np.unique(m)) <= {0, 1}, (level, w, h)
        m2 = binarize(img, w, h, _p(level), {"state": InkState()} if level == 5 else None)
        assert (m == m2).all(), (level, w, h)


def test_gray_and_invert_inputs():
    img = _img(1)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert binarize(g, 320, 180, _p(3)).shape == (180, 320)
    a = binarize(img, 320, 180, _p(3))
    assert (a ^ binarize(img, 320, 180, _p(3, invert=True))).all()


def test_frame_is_made_of_whole_glyphs():
    """кожна клітинка маски - точно біти одного знака банку (крайні, обрізані, - префікс його бітів)"""
    img = _img()
    w, h = 640, 352                                               # 640 / 160 = 4 px, 352 / 8 = 44 рядки: без обрізаних клітинок
    for level in (3, 4):
        m = binarize(img, w, h, _p(level))
        b = glyph.bank_for(w, h)
        table = {k.tobytes() for k in b.ink}
        cells = m.reshape(h // b.ch, b.ch, w // b.cw, b.cw).transpose(0, 2, 1, 3).reshape(-1, b.ch, b.cw)
        assert all(c.tobytes() in table for c in cells), level


def _allowed(level):
    return {chr(o) for lo, hi, t in glyph.RANGES if t <= level for o in range(lo, hi + 1)} | set(" \n")


def test_alphabet_grows_with_level():
    b = glyph.bank_for(1280, 720)
    n = [len(b.level(lv)[0]) for lv in (3, 4, 5)]
    assert n[0] < n[1] < n[2] and n[0] > 200, n
    img = _img(3, 1280, 720)
    for lv in (3, 4, 5):
        t = glyph.text(img, 1280, 720, _p(lv))
        assert set(t) <= _allowed(lv), (lv, set(t) - _allowed(lv))
        assert not set("▉▇▆▅▊▋◘◙") & set(t), "неповних блоків і «ящиків із дірою» немає: лише повний █ там, де клітинка справді світла"
    t5 = glyph.text(img, 1280, 720, _p(5))
    assert len(t5.splitlines()) == glyph.grid_for(1280, 720)[0]


def test_frame_has_a_zoo_of_glyphs_not_monotone_blocks():
    """головне: кадр не з однакових блоків - багато різних знаків, жоден не займає половини кадру"""
    img = _img(6, 1280, 720)
    for lv in (4, 5):
        idx, bank = glyph.plan(img, 1280, 720, _p(lv), None)
        vals, cnt = np.unique(idx, return_counts=True)
        assert len(vals) >= 40, (lv, len(vals))
        assert cnt.max() < 0.5 * idx.size, (lv, cnt.max() / idx.size)


def test_text_matches_the_mask():
    img = _img(2)
    w, h = 640, 352
    for level in (1, 2, 3, 4):
        p = _p(level)
        txt = glyph.text(img, w, h, p)
        back = glyph.text_from_mask(binarize(img, w, h, p), w, h)
        assert txt == back, level
    rows, cols = glyph.grid_for(w, h)
    lines = glyph.text(img, w, h, _p(4)).splitlines()
    assert len(lines) == rows and max(len(x) for x in lines) <= cols


def test_text_from_mask_handles_partial_edge_cells():
    img = _img(4)
    w, h = 643, 357                                               # не кратні клітинці: крайні знаки обрізано
    m = binarize(img, w, h, _p(4))
    rows, cols = glyph.grid_for(w, h)
    txt = glyph.text_from_mask(m, w, h)
    assert len(txt.splitlines()) == rows


# ---------------------------------------------------------------- тон
@pytest.mark.parametrize("level", (1, 2, 3, 4, 5))
def test_brighter_picture_gives_denser_glyphs(level):
    w, h = 640, 360
    ramp = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
    m = binarize(img, w, h, _p(level), {"state": InkState()} if level == 5 else None)
    q = [m[:, i * w // 4:(i + 1) * w // 4].mean() for i in range(4)]
    assert q[0] < q[1] < q[2] < q[3], (level, q)
    assert q[0] < 0.1 and q[3] > 0.6


def test_flat_black_and_white_frames():
    """чорний - порожньо; білий - суцільний █"""
    w, h = 320, 180
    for level in (1, 2, 3, 4, 5):
        st = {"state": InkState()} if level == 5 else None
        black = binarize(np.zeros((h, w, 3), np.uint8), w, h, _p(level), st)
        white = binarize(np.full((h, w, 3), 255, np.uint8), w, h, _p(level), {"state": InkState()} if level == 5 else None)
        assert black.mean() < 0.02, (level, black.mean())
        assert white.mean() > 0.97, (level, white.mean())


def test_many_grays_on_a_smooth_ramp():
    """плавний перехід - плавна густина: 16 смуг дають зростання без провалів і з великим розмахом (а не 4-6 сходинок)"""
    w, h = 1280, 160
    ramp = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
    for level in (3, 4, 5):
        m = binarize(img, w, h, _p(level), {"state": InkState()} if level == 5 else None)
        q = np.array([m[:, i * w // 16:(i + 1) * w // 16].mean() for i in range(16)])
        assert (np.diff(q) > -0.03).all() and q[-1] - q[0] > 0.4, (level, q.round(2))
        assert len(np.unique(q.round(2))) >= 10, (level, q.round(2))


def test_equalize_keeps_black_and_white_and_flat_frames():
    t = np.linspace(0, 1, 5000, dtype=np.float32).reshape(50, 100) ** 3
    e = glyph._equalize(t, None)
    assert abs(float(e.min())) < 1e-3 and abs(float(e.max()) - 1) < 1e-3
    assert (np.diff(np.sort(e.ravel())) >= -1e-6).all()
    flat = np.full((50, 100), 0.4, np.float32) + np.random.default_rng(0).normal(0, 0.002, (50, 100)).astype(np.float32)
    assert np.abs(glyph._equalize(flat, None) - flat).max() < 1e-6, "майже рівний кадр не вирівнюється: шум став би візерунком"


def test_solid_blocks_holes_and_half_blocks_are_told_apart():
    full = np.ones((16, 8), np.uint8)
    dotted = full.copy()
    dotted[6:10, 2:6] = 0                                          # квадрат із діркою, як ◘
    half = np.zeros((16, 8), np.uint8)
    half[8:] = 1
    seven = full.copy()
    seven[:2] = 0                                                  # ▇: майже повний, але не суцільний
    assert glyph._solid(full) and not glyph._solid(dotted) and not glyph._solid(half)
    assert glyph._class("█", full) == glyph.G_SOLID and glyph._class("▇", seven) == glyph.G_PART
    assert glyph._hole(dotted) and not glyph._hole(full) and not glyph._hole(half)
    dots = full.copy()
    dots[::2, ::2] = 0                                             # напівтінь з одиночними темними пікселями (▓) - не діра
    assert not glyph._hole(dots)


def test_holey_boxes_never_appear_and_white_field_is_a_clean_block():
    """◘-подібні знаки не вживаються ніде; суцільне біле поле - чисте █, а не ряди «квадратів із крапкою»"""
    w, h = 640, 352
    card = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(card, "Hello", (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6, cv2.LINE_AA)
    for level in (3, 4, 5):
        idx, bank = glyph.plan(card, w, h, _p(level), InkState() if level == 5 else None)
        used = set(np.unique(idx).tolist())
        assert not any(bank.group[i] in (glyph.G_HOLE, glyph.G_PART) for i in used), level
        top = idx[:4, :].ravel()                                    # поле над написом
        assert (bank.group[top] == glyph.G_SOLID).mean() > 0.95, level


def test_flat_field_takes_an_even_texture_not_random_letters():
    """рівне поле (як синя заставка: майже без структури, лише шум) - рівна текстура, а букви лишаються біля краю й тексту"""
    w, h = 640, 352
    rng = np.random.default_rng(5)
    img = np.clip(60 + rng.normal(0, 0.7, (h, w, 3)), 0, 255).astype(np.uint8)
    cv2.rectangle(img, (240, 150), (400, 200), (255, 255, 255), -1)
    cv2.putText(img, "Quest", (250, 188), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    idx, bank = glyph.plan(img, w, h, _p(5), InkState())
    flat = idx[:6, :]                                               # далеко від напису
    assert bank.coarse[flat].mean() < 0.06, "у рівному полі - рівні текстури"
    assert bank.coarse[idx[9:12, 28:42]].max() > 0.15, "біля напису - знаки з крупною структурою (штрихи, літери)"
    assert len(np.unique(flat)) >= 2, "але не один знак на все поле: текстура жива (два-три знаки чергуються за тоном)"


def test_equalize_keeps_a_big_flat_field_instead_of_scrambling_it():
    """70% кадру - біле поле: вирівнювання за процентилями віддавало йому випадковий тон посередині (біле ставало шумом), тепер лишається білим;
    у фото з великими насиченими плямами стиск білого - як раніше"""
    rng = np.random.default_rng(3)
    t = np.ones((100, 200), np.float32)
    t[70:] = np.tile(np.linspace(0, 1, 200, dtype=np.float32), (30, 1))
    e = glyph._equalize(t, None)
    assert e[:60].min() > 0.99, "біле поле лишилось білим"
    photo = np.clip(rng.normal(0.5, 0.3, (100, 200)), 0, 1).astype(np.float32)
    assert glyph._equalize(photo, None).std() > 0.2


# ---------------------------------------------------------------- пам'ять кадрів (рівень 5)
def test_level5_holds_glyphs_between_frames():
    base = cv2.resize(_img(5), (640, 352))
    rng = np.random.default_rng(7)

    def noisy():
        return np.clip(base.astype(np.int16) + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)

    w, h = 640, 352
    st = InkState()
    seq = [noisy() for _ in range(6)]
    with_mem = [glyph.plan(f, w, h, _p(5), st)[0] for f in seq]
    fresh = [glyph.plan(f, w, h, _p(5), InkState())[0] for f in seq]
    diff_mem = np.mean([(a != b).mean() for a, b in zip(with_mem[1:], with_mem[2:])])
    diff_fresh = np.mean([(a != b).mean() for a, b in zip(fresh[1:], fresh[2:])])
    assert diff_mem < diff_fresh, (diff_mem, diff_fresh)


def test_level5_memory_survives_a_size_change():
    st = InkState()
    img = _img()
    a = binarize(img, 640, 352, _p(5), {"state": st})
    b = binarize(img, 320, 176, _p(5), {"state": st})              # інша сітка: пам'ять тихо скидається, без помилки
    c = binarize(img, 640, 352, _p(5), {"state": st})
    assert a.shape == (352, 640) and b.shape == (176, 320) and c.shape == (352, 640)


# ---------------------------------------------------------------- прискорені частини = простий варіант
def test_numba_pick_equals_numpy(monkeypatch):
    rng = np.random.default_rng(1)
    n, g = 300, 120
    S = (rng.normal(size=(n, g)) * 5).astype(np.float32)
    dens = rng.random(g).astype(np.float32)
    fm = rng.random(n).astype(np.float32)
    prev = rng.integers(-1, g, n).astype(np.int32)
    sg = (rng.random(g) * 0.3).astype(np.float32)
    si = (rng.random(n) * 0.3).astype(np.float32)
    a = glyph._pick(S.copy(), dens, fm, 20.0, prev, 1.5, 40, sg, si, 90.0)
    monkeypatch.setattr(core, "_nb", None)
    b = glyph._pick(S.copy(), dens, fm, 20.0, prev, 1.5, 40, sg, si, 90.0)
    assert (a == b).mean() > 0.99


def test_lot_is_stable_and_breaks_ties_per_cell():
    """нічия між однаково добрими знаками розв'язується постійним «жеребом» клітинки: той самий кадр - той самий вибір,
    а сусідні клітинки беруть різні знаки (немає колон однакових)"""
    g = 64
    S = np.zeros((200, g), np.float32)
    dens = np.zeros(g, np.float32)
    fm = np.zeros(200, np.float32)
    prev = np.full(200, -1, np.int32)
    a = glyph._pick(S.copy(), dens, fm, 1.0, prev, 0.0, 0)
    assert (a == glyph._pick(S.copy(), dens, fm, 1.0, prev, 0.0, 0)).all()
    assert len(np.unique(a)) > g // 2


# ---------------------------------------------------------------- Params, експорт, CLI
def test_params_roundtrip_and_dispatch():
    p = Params(method=4, glyph=True)
    assert Params.from_dict(p.to_dict()).glyph is True and Params().glyph is False
    img = _img()
    assert (binarize(img, 320, 180, p) == glyph.binarize(img, 320, 180, p)).all()
    assert not (binarize(img, 320, 180, p) == binarize(img, 320, 180, Params(method=4))).all()


def test_still_formats_and_text_export(tmp_path):
    from benvid import media
    assert media.TEXT_FORMAT in media.still_formats(True) and media.TEXT_FORMAT not in media.still_formats(False)
    src = str(tmp_path / "a.png")
    cv2.imwrite(src, _img(0, 400, 300))
    out = str(tmp_path / "a.txt")
    r = media.export_still(src, out, Params(method=3, glyph=True), "Як у джерела")
    txt = open(out, encoding="utf-8").read()
    rows, cols = glyph.grid_for(*r["size"])
    assert len(txt.splitlines()) == rows and r["bytes"] == os.path.getsize(out) > 100
    out2 = str(tmp_path / "a.png.out.png")
    media.export_still(src, out2, Params(method=3, glyph=True), "Як у джерела")
    assert set(np.unique(cv2.imread(out2, 0))) <= {0, 255}


def test_cli_symbols(tmp_path):
    from benvid.cli import main
    a, o, t = str(tmp_path / "a.png"), str(tmp_path / "o.png"), str(tmp_path / "o.txt")
    cv2.imwrite(a, _img(0, 400, 300))
    main([a, o, "--symbols", "--method", "4", "--size", "240p"])
    assert set(np.unique(cv2.imread(o, 0))) <= {0, 255}
    main([a, t, "--symbols", "--method", "3", "--size", "240p"])
    assert open(t, encoding="utf-8").read().strip()
    with pytest.raises(SystemExit):
        main([a, t, "--method", "3"])                              # .txt без --symbols - помилка аргументів


def test_quick_prefs_carry_the_mode(tmp_path):
    from benvid import quick
    pr = quick.load_prefs({"ui": {"method": 3, "glyph": True}})
    assert pr.glyph is True and pr.method == 3
    assert quick.load_prefs({"ui": {}}).glyph is False
    a = str(tmp_path / "q.png")
    cv2.imwrite(a, _img(0, 300, 200))
    out = quick.convert(a, pr)
    assert os.path.isfile(out)


# ---------------------------------------------------------------- розмір Авто й відео
def test_autosize_measures_the_glyph_mode():
    from benvid import autosize
    frame = _img(0, 640, 360)
    small, big = autosize.measure(frame, 4, 1, True)
    assert small > 0 and all(v > 0 for v in big.values())
    key = autosize.pick(frame, 24.0, 2, src_h=360, method=4, glyph=True)
    assert key.endswith("p")


# ---------------------------------------------------------------- вікно
def _photo(path, w=320, h=240):
    from PIL import Image, ImageDraw
    im = Image.new("L", (w, h), 30)
    d = ImageDraw.Draw(im)
    for i in range(0, w, 16):
        d.rectangle([i, 0, i + 7, h], fill=60 + (i * 3) % 190)
    d.ellipse([w // 2 - 70, h // 2 - 70, w // 2 + 70, h // 2 + 70], fill=235)
    im.convert("RGB").save(path)


# ---------------------------------------------------------------- відео: програвач і експорт
def _clip(path, n=18, fps=24, w=320, h=176):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), 40, np.uint8)
        cv2.circle(f, (30 + i * 8, h // 2), 40, (230, 230, 230), -1)
        cv2.rectangle(f, (200, 20), (300, 70), (120, 120, 120), -1)
        vw.write(f)
    vw.release()


@pytest.mark.parametrize("level", (3, 5))
def test_player_renders_symbol_video(tmp_path, level):
    import time

    from benvid.player import Player
    path = str(tmp_path / "v.mp4")
    _clip(path)
    p = Player(workers=2)
    try:
        p.open(path, "Як у джерела")
        p.set_params(_p(level))
        assert p._ink_pool() == (level == 5), "рівень 5 символів рахується в окремому процесі, як Ультра (інакше вікно підфрізує); 3-4 - пул кадрів"
        t0 = time.time()
        while time.time() - t0 < 120 and p.cached < p.n:
            p.update()
            time.sleep(0.01)
        assert p.cached == p.n and p._ink_fail == 0
        w, h = p.size
        m = np.unpackbits(p._frames[5], axis=-1, count=w)
        assert m.shape == (h, w) and 0.02 < m.mean() < 0.9
        b = glyph.bank_for(w, h)
        table = {k.tobytes() for k in b.ink}
        cw, ch = b.cw, b.ch
        if w % cw == 0 and h % ch == 0:
            cells = m.reshape(h // ch, ch, w // cw, cw).transpose(0, 2, 1, 3).reshape(-1, ch, cw)
            assert all(c.tobytes() in table for c in cells)
    finally:
        p.close()


def test_export_symbol_video_every_level(tmp_path):
    from benvid.video import export_video
    path = str(tmp_path / "v.mp4")
    _clip(path, n=10)
    for level in (1, 3, 5):
        out = str(tmp_path / f"o{level}")
        r = export_video(path, out, _p(level), "PNG-послідовність", "Як у джерела", "Висока", False, 2)
        files = sorted(os.listdir(out))
        assert r["frames"] == len(files) == 10, (level, r["frames"], len(files))
        im = cv2.imread(os.path.join(out, files[4]), 0)
        assert set(np.unique(im)) <= {0, 255} and im.shape == (r["size"][1], r["size"][0])


# ---------------------------------------------------------------- щільність знаків
def test_density_positions_are_the_canonical_cells():
    assert glyph.DENS_N == 3 and glyph.DENS_DEFAULT == 3 and Params().gdens == 3, "типово - найдрібніша, 8x16"
    assert glyph.DENS_CW == (32, 16, 8)
    cols = [glyph.grid_for(4096, 2160, d)[1] for d in (1, 2, 3)]
    assert cols == [128, 256, 512] and glyph.grid_for(4096, 2160, 3)[0] == 135, "4096x2160: 512 x 135 знаків 8x16"
    assert (glyph.edge_sizes(4096, 2160, 8, 16)[0] == 16).all(), "на 4096x2160 клітинка 8x16 лягає без залишку"
    for w, h in ((1280, 720), (1920, 1080), (3840, 2160)):
        cs = [glyph.cell_for(w, h, d) for d in (1, 2, 3)]
        assert len(set(cs)) == 3 and all(c[1] == 2 * c[0] for c in cs), (w, h, cs)
    assert glyph.dens_of(0) == 1 and glyph.dens_of(9) == 3 and glyph.dens_of("x") == 3


@pytest.mark.parametrize("dens", (1, 2, 3))
@pytest.mark.parametrize("level", (3, 5))
def test_every_density_renders_a_whole_frame(dens, level):
    w, h = 640, 360
    img = _img(7)
    p = Params(method=level, glyph=True, gdens=dens)
    st = InkState() if level == 5 else None
    m = glyph.binarize(img, w, h, p, st)
    assert m.shape == (h, w) and set(np.unique(m)) <= {0, 1}
    rows, cols = glyph.grid_for(w, h, dens)
    assert len(glyph.text(img, w, h, p, InkState() if level == 5 else None).splitlines()) == rows
    assert glyph.tone_size(w, h, dens) == (cols * glyph.fdims(glyph.cell_for(w, h, dens)[0])[1], rows * glyph.fdims(glyph.cell_for(w, h, dens)[0])[0])


def test_finer_density_means_more_characters_and_more_detail():
    w, h = 960, 540
    yy, xx = np.mgrid[0:h, 0:w]
    base = (127 + 100 * np.sin(xx / 9.0) * np.cos(yy / 7.0)).clip(0, 255).astype(np.uint8)
    img = cv2.merge([base, base, base])
    rows = [glyph.grid_for(w, h, d)[0] * glyph.grid_for(w, h, d)[1] for d in (1, 2, 3)]
    assert rows[0] < rows[1] < rows[2]
    # дрібніші знаки точніше повторюють тон: середня помилка згладженого кадру менша
    err = []
    for d in (1, 2, 3):
        m = glyph.binarize(img, w, h, Params(method=3, glyph=True, gdens=d)).astype(np.float32)
        err.append(float(np.abs(cv2.GaussianBlur(m, (0, 0), 6) - cv2.GaussianBlur(base.astype(np.float32) / 255, (0, 0), 6)).mean()))
    assert err[2] <= err[0] + 0.05, err


def test_text_from_mask_follows_density():
    w, h = 640, 360
    img = _img(2)
    for d in (1, 2, 3):
        p = Params(method=4, glyph=True, gdens=d)
        a = glyph.text(img, w, h, p).splitlines()
        b = glyph.text_from_mask(glyph.binarize(img, w, h, p), w, h, d).splitlines()
        assert len(a) == len(b) == glyph.grid_for(w, h, d)[0]
        bad = sum(x != y for r1, r2 in zip(a, b) for x, y in zip(r1, r2))      # різняться лише знаки, що після зрізу хвостового краю однакові
        assert bad < 0.01 * glyph.grid_for(w, h, d)[0] * glyph.grid_for(w, h, d)[1], (d, bad)


# ---------------------------------------------------------------- нерухоме не дріботить
def _noisy_frames(n, seed=5, w=640, h=360, sigma=2.0, pan=0):
    """одна й та сама напівтемна картинка (_img, притемнена, зі світлою плямою) зі свіжим шумом відео на кожному кадрі"""
    rng = np.random.default_rng(seed)
    base = _img(seed=3, w=w, h=h).astype(np.float32).mean(axis=2) * 0.35 + 8
    base[200:300, 380:560] = 170
    for i in range(n):
        b = np.roll(base, i * pan, axis=1) if pan else base
        f = np.clip(b + rng.normal(0, sigma, (h, w)), 0, 255).astype(np.uint8)
        yield cv2.merge([f, f, f])


def _flicker(level_params, n=14):
    st = InkState()
    prev, rates = None, []
    for f in _noisy_frames(n):
        idx, _ = glyph.plan(f, 640, 360, level_params, st)
        if prev is not None:
            rates.append(float((idx != prev).mean()))
        prev = idx
    return float(np.mean(rates[4:]))                                       # перші кадри - розігрів пам'яті


def test_static_noisy_picture_does_not_flicker():
    """відео з зерном, де нічого не рухається: знаки не мають перемикатися щокадру (раніше змінювалась кожна десята-п'ята клітинка)"""
    still = _flicker(_p(5))
    assert still < 0.06, still            # раніше ≈0.43


def test_stronger_hold_only_where_still(monkeypatch):
    """утримання зростає лише де тон не змінився: без нього (HYST_STILL=0) дріботіння помітно більше, а справжня зміна кадру проходить одразу"""
    base = _flicker(_p(5))
    monkeypatch.setattr(glyph, "HYST_STILL", 0.0)
    from benvid import ink5
    monkeypatch.setitem(ink5.P, "tden", 0.0)                               # і без часового усереднення світлоти до підсилення тону
    assert _flicker(_p(5)) > 2 * base + 0.005
    monkeypatch.undo()
    st = InkState()
    it = _noisy_frames(8)
    for f in it:
        a, bank = glyph.plan(f, 640, 360, _p(5), st)
    f2 = f.copy()
    f2[:, :] = 235                                                         # кадр різко змінився: знаки мають змінитись одразу, а не «залипнути»
    b, _ = glyph.plan(f2, 640, 360, _p(5), st)
    assert float((a != b).mean()) > 0.8


def test_chunk_start_matches_the_sequential_render():
    """шматок відео стартує з розігрівом (прайм → лише тон → кілька кадрів повністю): його перші кадри мають збігатися з послідовним розрахунком
    (без шуму - до клітинки; на реальних відео з зерном розбіжність ≈5%, виміряно на 2 кліпах), інакше на стику шматків (кожні ~4 с) знаки «блимали» б"""
    fr = list(_noisy_frames(40, sigma=0.0))
    p = _p(5)
    st = InkState()
    seq = [glyph.plan(f, 640, 360, p, st)[0] for f in fr]
    k = 30
    st2 = InkState()
    n = glyph.WARM + glyph.WARM_CHEAP
    for f in fr[k - n - 12:k - n]:
        glyph.prime(st2, f, 640, 360, True, 3)
    for i, f in enumerate(fr[k - n:k]):
        glyph.plan(f, 640, 360, p, st2, warm_only=i < glyph.WARM_CHEAP)
    d = [float((glyph.plan(fr[k + i], 640, 360, p, st2)[0] != seq[k + i]).mean()) for i in range(3)]
    assert max(d) < 0.03, d


# ---------------------------------------------------------------- світлі тони: без сходинок, лише білі знаки на чорному
def _light_gradient(level, lo=0.5, hi=1.0, w=1280, h=720):
    """градієнт тону lo..hi через ту саму нарізку на рівні із синім шумом, що й у plan -> (знаки, бажана середня густина клітинок)"""
    bank = glyph.bank_for(w, h, 3)
    rows, cols = glyph.grid_for(w, h, 3)
    fh, fw = glyph.fdims(bank.cw)
    gw, gh = cols * fw, rows * fh
    grad = np.tile(np.linspace(lo, hi, gw, dtype=np.float32), (gh, 1))
    nl = glyph.LEVELS[level]
    lev = np.clip(np.floor(grad * (nl - 1) + core._tile(core._blue(), gw, gh, 1)), 0, nl - 1)
    idx = glyph._match((lev / (nl - 1)).astype(np.float32), bank, rows, cols, level, None).reshape(rows, cols)
    want = grad.reshape(rows, fh, cols, fw).mean(axis=(1, 3))
    return idx, bank, want


@pytest.mark.parametrize("level", [3, 4, 5])
def test_light_tones_are_not_stepped(level):
    """у наборі білих знаків густіше за ▒ (0.5) лише ▓ (0.83) й █: світлі тони лягали плато 0.5 -> 0.83 -> 1. Сусідні густини мають змішуватись
    між клітинками, тож середній тон по сусідніх клітинках точний (раніше помилка сягала 0.1 і більше)"""
    idx, bank, want = _light_gradient(level)
    got = cv2.blur(bank.dens[idx].astype(np.float32), (4, 4))
    err = float(np.sqrt(((got - cv2.blur(want.astype(np.float32), (4, 4))) ** 2).mean()))
    assert err < 0.03, err
    assert int(idx.max()) < len(bank.chars)                              # лише справжні знаки: білий на чорному, без інверсних


def test_gap_dither_mixes_neighbour_densities_in_flat_cells_only():
    rows, cols = 40, 50
    n = rows * cols
    ad = np.array([0.0, 0.5, 0.83, 1.0], np.float32)
    fm = np.full(n, 0.67, np.float32)
    flat, edge = np.full(n, 0.01, np.float32), np.full(n, 0.3, np.float32)
    out = glyph._gap_dither(fm, flat, ad, rows, cols)
    u = np.unique(out)
    assert len(u) == 2 and np.allclose(u, [0.5, 0.83], atol=1e-6), "рівна клітинка в прогалині - одна з двох сусідніх густин"
    assert abs(float(out.mean()) - 0.67) < 0.015, "а середнє по клітинках - точний тон"
    assert np.array_equal(glyph._gap_dither(fm, edge, ad, rows, cols), fm), "на краях і літерах тон не чіпаємо"
    close = np.linspace(0, 1, 60).astype(np.float32)
    assert np.array_equal(glyph._gap_dither(fm, flat, close, rows, cols), fm), "без прогалини - без змін"
    on = np.full(n, 0.5, np.float32)                                    # клітинки, що стояли на нижчому щаблі
    fm2 = np.full(n, 0.5 + 0.33 * 0.55, np.float32)
    free = float((glyph._gap_dither(fm2, flat, ad, rows, cols) > 0.6).mean())
    held = float((glyph._gap_dither(fm2, flat, ad, rows, cols, on) > 0.6).mean())
    assert held < free, "біля порога клітинка лишає минулий щабель"


def test_light_field_in_video_still_does_not_flicker():
    """нерухоме світле поле зі зерном: змішування густин не має мерехтіти (клітинки на порозі тримають щабель)"""
    rng = np.random.default_rng(9)
    base = np.tile(np.linspace(120, 250, 640, dtype=np.float32), (360, 1))
    st, prev, rates = InkState(), None, []
    for _ in range(14):
        f = np.clip(base + rng.normal(0, 2.0, base.shape), 0, 255).astype(np.uint8)
        idx, _ = glyph.plan(cv2.merge([f, f, f]), 640, 360, _p(5), st)
        if prev is not None:
            rates.append(float((idx != prev).mean()))
        prev = idx
    assert float(np.mean(rates[4:])) < 0.06, rates


def test_flat_cells_draw_from_many_glyphs_without_losing_tone(monkeypatch):
    """рівне поле середнього тону не складається з одного знака (смуги ▒▓): у пласких клітинках знаки майже однакової густини змагаються;
    середній тон при цьому той самий, а контури (непласкі клітинки) не змінюються"""
    rng = np.random.default_rng(3)
    base = np.tile(np.linspace(40, 200, 640, dtype=np.float32), (360, 1))
    f = np.clip(base + rng.normal(0, 1.0, base.shape), 0, 255).astype(np.uint8)
    img = cv2.merge([f, f, f])
    on, off = [], []
    for jit, lam, bag in ((glyph.FLAT_JIT, glyph.FLAT_LAM, on), (1.0, 1.0, off)):
        monkeypatch.setattr(glyph, "FLAT_JIT", jit)
        monkeypatch.setattr(glyph, "FLAT_LAM", lam)
        idx, bank = glyph.plan(img, 640, 360, _p(4), None)
        bag += [len(np.unique(idx)), float(glyph.render_idx(idx, bank, 640, 360).mean())]
    assert on[0] > off[0], (on, off)
    assert abs(on[1] - off[1]) < 0.01, "середня яскравість маски та сама"
    # краї: клітинки з великою крупною структурою (літери, контури) лишаються як були
    letters = np.full((360, 640, 3), 30, np.uint8)
    cv2.putText(letters, "InkReel", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (240, 240, 240), 5, cv2.LINE_AA)
    monkeypatch.setattr(glyph, "FLAT_JIT", 12.0)
    monkeypatch.setattr(glyph, "FLAT_LAM", 0.5)
    a, _ = glyph.plan(letters, 640, 360, _p(4), None)
    monkeypatch.setattr(glyph, "FLAT_JIT", 1.0)
    monkeypatch.setattr(glyph, "FLAT_LAM", 1.0)
    b, _ = glyph.plan(letters, 640, 360, _p(4), None)
    assert float((a != b).mean()) < 0.25, "літери майже не змінились"


def test_alphabet_has_the_rest_of_the_consolas_set():
    """набір користувача (1676 знаків Consolas) покритий діапазонами: окрім комбінівних знаків (ті ламають сітку тексту)"""
    for c in "฿⸗ꜗꜟﬓﬗ":
        assert any(lo <= ord(c) <= hi for lo, hi, _ in glyph.RANGES), c
