r"""Store screenshots: starts the real (Qt) app on a picture, waits for the result and grabs the window (PrintWindow) at 1920x1080, once per language.
No captions or text are added to the pictures: the captions go into the Store caption field (store/listing_en.md, store/listing_uk.md).

    python tools/make_listing_shots.py [name-prefix ...]   ->  store/listing_assets/screenshots/{en,uk}_<n>_<name>.png   (not in git)

Windows only, needs the scenes (tools/make_listing_scenes.py) and the project environment. Pictures are looked up in this order: the scenes folder,
then the folder named by the INKREEL_SHOT_SOURCES environment variable (any extra frames, searched recursively by file name).
It temporarily replaces %LOCALAPPDATA%\InkReel\settings.json (level, palette, language) and puts your own file back afterwards. Close InkReel first,
or the file is handed to the running copy.

Why the output size is a custom WxH and not 720p: the picture area of a 1920x1080 window is about 1570x890 px. The app shows a result at whole-pixel zoom
steps, and a 1280x720 picture does not fit twice, while a 1-bit picture shrunk by a fractional factor turns to mush. A size chosen so that 2x fills the area
(768x432 -> 1536x864) keeps every dot a crisp square, and the Store, which shows the screenshot much smaller, still reads it."""
import ctypes, json, os, re, shutil, subprocess, sys, tempfile, time
from ctypes import wintypes as wt
import numpy as np
from PIL import Image

ctypes.windll.shcore.SetProcessDpiAwareness(2)
U, G, D = ctypes.windll.user32, ctypes.windll.gdi32, ctypes.windll.dwmapi
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SC = os.path.join(ROOT, "store", "listing_assets", "scenes")
OUT = os.path.join(ROOT, "store", "listing_assets", "screenshots")
SET = os.path.join(os.environ["LOCALAPPDATA"], "InkReel", "settings.json")
EXTRA = os.environ.get("INKREEL_SHOT_SOURCES", "")
WIN_W, WIN_H = 1920, 1080

# (file stem, picture file name, method, glyph, palette key, output size)
SHOTS = [
    ("1_engraving_cow", "highland-cow__src.png", 5, False, "Класика", "768x432"),
    ("2_irnm1_pines", "foggy-pine-forest__src.png", 6, False, "Туш і папір", "768x432"),
    ("3_drawing_cyclist", "cyclist-at-sunset__src.png", 4, False, "Синька", "768x432"),
    ("4_halftone_wave", "ocean-wave__src.png", 3, False, "Сепія", "768x432"),
    ("5_symbols_lighthouse", "lighthouse-in-storm__src.png", 5, True, "Термінал", "1536x864"),     # symbols want finer cells: 192x54 characters
    ("6_outline_coast", "rocky-coast__src.png", 2, False, "Кармін", "768x432"),
    ("7_irnm1_peak", "peak-above-sea-of-clouds__src.png", 6, False, "Бурштин", "768x432"),
    ("8_engraving_crescent", "07_crescent_sea.png", 5, False, "Класика", "768x432"),
]


def find_source(name):
    """the picture's path; extra frames are copied under a plain name (no "__src" tail) so the title bar and the file label look clean"""
    d = os.path.join(tempfile.gettempdir(), "inkreel_shots")
    os.makedirs(d, exist_ok=True)
    plain = re.sub(r"^\d+_", "", name.replace("__src", "")).replace("_", "-")          # 07_crescent_sea.png -> crescent-sea.png
    q = os.path.join(d, plain)
    p = os.path.join(SC, name)
    if os.path.exists(p):
        shutil.copyfile(p, q)
        return q
    for base, _, files in os.walk(EXTRA or os.path.join(SC, "_none_")):
        if name in files:
            shutil.copyfile(os.path.join(base, name), q)
            return q
    raise FileNotFoundError(name)


def find_hwnd(pid):
    found = []
    cb = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)(lambda h, l: (found.append(h) if _own(h, pid) else None) or True)
    U.EnumWindows(cb, 0)
    return found[0] if found else None


def _own(h, pid):
    if not U.IsWindowVisible(h):
        return False
    c = ctypes.create_unicode_buffer(64)
    U.GetClassNameW(h, c, 64)
    if not c.value.startswith("Qt"):                 # the venv launcher spawns the real process, so match by class and title, not pid
        return False
    n = U.GetWindowTextLengthW(h)
    b = ctypes.create_unicode_buffer(n + 1)
    U.GetWindowTextW(h, b, n + 1)
    return b.value.startswith("InkReel")


def frame_rect(h):
    r = wt.RECT()
    D.DwmGetWindowAttribute(h, 9, ctypes.byref(r), ctypes.sizeof(r))
    return r


def win_rect(h):
    r = wt.RECT()
    U.GetWindowRect(h, ctypes.byref(r))
    return r


def grab(h):
    r = win_rect(h)
    w, hh = r.right - r.left, r.bottom - r.top
    dc = U.GetWindowDC(h)
    mdc = G.CreateCompatibleDC(dc)
    bmp = G.CreateCompatibleBitmap(dc, w, hh)
    G.SelectObject(mdc, bmp)
    U.PrintWindow(h, mdc, 2)

    class BI(ctypes.Structure):
        _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG), ("biPlanes", wt.WORD), ("biBitCount", wt.WORD),
                    ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD), ("a", wt.LONG), ("b", wt.LONG), ("c", wt.DWORD), ("d", wt.DWORD)]
    bi = BI(ctypes.sizeof(BI), w, -hh, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(w * hh * 4)
    G.GetDIBits(mdc, bmp, 0, hh, buf, ctypes.byref(bi), 0)
    G.DeleteObject(bmp); G.DeleteDC(mdc); U.ReleaseDC(h, dc)
    a = np.frombuffer(buf, np.uint8).reshape(hh, w, 4)[:, :, [2, 1, 0]]
    f = frame_rect(h)
    return a[f.top - r.top:f.bottom - r.top, f.left - r.left:f.right - r.left].copy()


def shoot(stem, src, method, glyph, pal, size, lang):
    ui = {"size": size, "fmt": "MP4 (H.264)", "quality": "Висока", "audio": True, "palette": pal, "pal_inv": False, "method": method,
          "glyph": glyph, "gdens3": 3, "volume": 69, "lang": lang, "guide": "quality", "fps": 0.0}
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\InkReel") as k:
            ui["install_lang"] = str(winreg.QueryValueEx(k, "InstallLang")[0]).lower()      # otherwise the app re-applies the installer's language
    except Exception:
        pass
    with open(SET, "w", encoding="utf-8") as f:
        json.dump({"ui": ui}, f, ensure_ascii=False, indent=1)
    p = subprocess.Popen([PY, "-m", "benvid", find_source(src)], cwd=ROOT, creationflags=0x08000000)
    h = None
    try:
        for _ in range(60):
            h = find_hwnd(p.pid)
            if h:
                break
            time.sleep(0.5)
        assert h, "no window"
        U.ShowWindow(h, 9)
        U.SetWindowPos(h, 0, 30, 10, WIN_W, WIN_H, 0x0004 | 0x0040)
        time.sleep(1.0)
        f, r = frame_rect(h), win_rect(h)                                  # make the visible frame exactly WIN_W x WIN_H
        dw, dh = (r.right - r.left) - (f.right - f.left), (r.bottom - r.top) - (f.bottom - f.top)
        U.SetWindowPos(h, 0, 30 - (f.left - r.left), 10, WIN_W + dw, WIN_H + dh, 0x0004 | 0x0040)
        time.sleep(1.0)
        prev, same, t0 = None, 0, time.time()
        img = None
        while time.time() - t0 < 120:
            time.sleep(2.0)
            img = grab(h)
            if prev is not None and prev.shape == img.shape and (prev == img).all():
                same += 1
                if same >= 2 and time.time() - t0 > 10:
                    break
            else:
                same = 0
            prev = img
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, "%s_%s.png" % (lang, stem))
        Image.fromarray(img).save(path)
        print(path, img.shape[1], img.shape[0], flush=True)
    finally:
        try:
            if h:
                U.PostMessageW(h, 0x0010, 0, 0)
        except Exception:
            pass
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()
        time.sleep(1.0)


if __name__ == "__main__":
    want = sys.argv[1:]
    bak = SET + ".bak_shots"
    shutil.copy(SET, bak)
    try:
        for stem, src, method, glyph, pal, size in SHOTS:
            if want and not any(stem.startswith(w) for w in want):
                continue
            for lang in ("en", "uk"):
                shoot(stem, src, method, glyph, pal, size, lang)
    finally:
        shutil.copy(bak, SET)
        os.remove(bak)
        print("settings restored", flush=True)

