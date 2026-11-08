"""Джерела медіа (відео, GIF, фото) і експорт фото/GIF.

Усі джерела мають однаковий вигляд: info (fps, width, height, frames, kind, starts) і read(idx) -> BGR або None.
starts - початок кожного кадру в секундах (GIF зі змінною тривалістю кадрів і відео зі змінною частотою кадрів; інакше None).
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .core import Params, binarize
from .i18n import tr

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".bmp", ".dib", ".webp", ".tif", ".tiff", ".avif", ".heic", ".heif",
             ".jp2", ".j2k", ".ico", ".tga", ".ppm", ".pgm", ".pbm", ".pnm", ".psd", ".hdr")
GIF_EXT = (".gif", ".apng")
STILL_FORMATS = {"PNG (1 біт)": ".png", "BMP": ".bmp", "TIFF (1 біт, G4)": ".tif", "WebP (без втрат)": ".webp"}
TEXT_FORMAT = "Текст (.txt, символи)"          # лише в режимі «Символи»: той самий кадр рядками знаків


def still_formats(glyph: bool = False) -> dict:
    """формати фото; для символів - ще текстовий файл"""
    return {**STILL_FORMATS, TEXT_FORMAT: ".txt"} if glyph else dict(STILL_FORMATS)
GIF_FORMATS = {"GIF": ".gif", "PNG-послідовність": ""}

_EXIF_EXT = (".jpg", ".jpeg", ".jpe", ".jfif", ".tif", ".tiff", ".webp", ".heic", ".heif", ".avif", ".png")
_HEIF_STILL = (b"avif", b"heic", b"heix", b"heim", b"heis", b"mif1", b"mif2")
_HEIF_ANIM = (b"avis", b"msf1", b"hevc", b"hevx")
_DIB_SIZES = (12, 40, 52, 56, 64, 108, 124)
ANIM_MAX_BYTES = 1_500_000_000            # скільки пам'яті дозволено на розпаковані кадри анімації (далі - зменшуємо)


def sniff(path: str) -> str:
    """тип файлу за ВМІСТОМ (розширення часто бреше або його немає): 'image', 'gif' (анімація: GIF/APNG/анімований WebP/AVIF)
    або '' - не впізнано (тоді вирішує розширення)"""
    try:
        with open(path, "rb") as f:
            h = f.read(32)
            if h[:4] == b"GIF8":
                return "gif"
            if h[:8] == b"\x89PNG\r\n\x1a\n":
                pos = 8
                for _ in range(400):                         # шматки до першого IDAT: acTL = APNG
                    f.seek(pos)
                    c = f.read(8)
                    if len(c) < 8:
                        break
                    ln, typ = int.from_bytes(c[:4], "big"), c[4:]
                    if typ == b"acTL":
                        return "gif"
                    if typ in (b"IDAT", b"IEND"):
                        break
                    pos += 12 + ln
                return "image"
            if h[:4] == b"RIFF" and h[8:12] == b"WEBP":
                return "gif" if h[12:16] == b"VP8X" and len(h) > 20 and h[20] & 2 else "image"
            if h[:3] == b"\xff\xd8\xff":
                return "image"
            if h[:2] == b"BM" and h[6:10] == b"\0\0\0\0" and int.from_bytes(h[14:18], "little") in _DIB_SIZES:
                return "image"
            if h[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
                return "image"
            if h[4:8] == b"ftyp":
                brand = h[8:12]
                if brand in _HEIF_ANIM:
                    return "gif"
                if brand in _HEIF_STILL:
                    return "image"
                return ""                                    # звичайне відео MP4/MOV
            if h[:4] == b"8BPS" or h[:4] == b"DDS " or h[:2] == b"#?" or h[:4] == b"\xff\x4f\xff\x51" \
                    or h[:12] == b"\x00\x00\x00\x0cjP  \r\n\x87\n":
                return "image"
    except OSError:
        pass
    return ""


def kind_of(path: str) -> str:
    """'image' | 'gif' | 'video': спершу вміст файлу, а якщо не впізнано - розширення (невідоме = відео)"""
    s = sniff(path)
    if s:
        return s
    e = os.path.splitext(path)[1].lower()
    return "image" if e in IMAGE_EXT else "gif" if e in GIF_EXT else "video"


_HEIF = None


def heif_ready() -> bool:
    """чи доступний декодер HEIC/HEIF (необов'язковий пакет pillow-heif); AVIF уміє й сам Pillow нових версій"""
    global _HEIF
    if _HEIF is None:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            try:
                pillow_heif.register_avif_opener()
            except Exception:  # noqa: BLE001
                pass
            _HEIF = True
        except Exception:  # noqa: BLE001
            _HEIF = False
    return _HEIF


def _float_u8(a: np.ndarray, ext: str = "") -> np.ndarray:
    """float-зображення (TIFF/HDR/EXR) -> uint8: автоматичні рівні за 99.9-м перцентилем (0..1 лишається 0..1),
    для сцен-лінійних .hdr/.exr додано гамму sRGB"""
    a = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    col = a[..., :3] if a.ndim == 3 and a.shape[2] == 4 else a
    hi = float(np.percentile(col, 99.9)) if col.size else 1.0
    hi = 1.0 if hi <= 1.0 else hi
    x = (col / hi).clip(0, 1)
    if ext in (".hdr", ".exr"):
        x = x ** (1 / 2.2)
    x = x * 255.0
    if a.ndim == 3 and a.shape[2] == 4:
        al = a[..., 3:4]
        m = float(al.max()) if al.size else 1.0
        al = (al / (m if m > 1.0 else 1.0)).clip(0, 1) * 255.0
        return np.concatenate([x, al], axis=2).round().astype(np.uint8)
    return x.round().astype(np.uint8)


def _flatten(img, ext: str = ""):
    """uint8/uint16/float (h,w[,c]) -> BGR uint8; прозорість кладемо на білий фон; 16 біт і float -> 8"""
    if img.dtype != np.uint8:
        if img.dtype == np.uint16:
            img = (img.astype(np.float32) / 257.0).round().clip(0, 255).astype(np.uint8)
        elif np.issubdtype(img.dtype, np.floating):
            img = _float_u8(img, ext)
        else:                                                    # цілі 32/64 біти (рідко): 8 біт як є, 16 біт - /257, більше - за максимумом
            mx = float(img.max()) if img.size else 0.0
            k = 1.0 if mx <= 255 else 1 / 257.0 if mx <= 65535 else 255.0 / mx
            img = (img.astype(np.float64) * k).round().clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 2:                                        # сірий + прозорість
        a = img[:, :, 1:2].astype(np.float32) / 255.0
        g = (img[:, :, :1] * a + 255.0 * (1 - a)).astype(np.uint8)
        return cv2.cvtColor(g[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        return (img[:, :, :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    return img


def _orient(a: np.ndarray, o: int) -> np.ndarray:
    """застосувати EXIF-орієнтацію (1..8): фото з телефона часто лежать «набік» і мають лише мітку повороту"""
    if o in (None, 0, 1):
        return a
    if o == 2:
        r = a[:, ::-1]
    elif o == 3:
        r = a[::-1, ::-1]
    elif o == 4:
        r = a[::-1]
    elif o == 5:
        r = a.transpose(1, 0, 2)
    elif o == 6:
        r = np.rot90(a, 3)                                       # 90° за годинниковою
    elif o == 7:
        r = np.rot90(a.transpose(1, 0, 2), 2)
    elif o == 8:
        r = np.rot90(a, 1)                                       # 90° проти годинникової
    else:
        return a
    return np.ascontiguousarray(r)


def exif_orientation(path: str) -> int:
    """значення EXIF Orientation (1 - без повороту). Читаються лише заголовки (без декодування картинки); помилки = 1"""
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "rb") as f:
            if f.read(8) == b"\x89PNG\r\n\x1a\n":                # PNG: шматок eXIf (зазвичай до IDAT)
                for _ in range(400):
                    c = f.read(8)
                    if len(c) < 8:
                        break
                    ln, typ = int.from_bytes(c[:4], "big"), c[4:]
                    if typ == b"eXIf":
                        from PIL import Image
                        ex = Image.Exif()
                        ex.load(f.read(ln))
                        return int(ex.get(0x0112, 1) or 1)
                    if typ in (b"IDAT", b"IEND"):
                        break
                    f.seek(ln + 4, 1)
                return 1
        if ext not in _EXIF_EXT and sniff(path) != "image":
            return 1
        from PIL import Image
        if ext in (".heic", ".heif", ".avif"):
            heif_ready()
        with Image.open(path) as im:
            return int(im.getexif().get(0x0112, 1) or 1)
    except Exception:  # noqa: BLE001
        return 1


def _pil_read(path: str, lenient: bool):
    """запасне читання через Pillow (дивні режими, 16 біт, CMYK, HEIC тощо; lenient - добрати пошкоджений файл, наскільки можна).
    -> (масив BGR(A)/сірий у рідному типі, орієнтація EXIF)"""
    from PIL import Image, ImageFile
    Image.MAX_IMAGE_PIXELS = None                                 # локальні файли: великі панорами - це нормально
    heif_ready()
    old = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = lenient
    try:
        im = Image.open(path)
        im.load()
        try:
            o = int(im.getexif().get(0x0112, 1) or 1)
        except Exception:  # noqa: BLE001
            o = 1
        mode = im.mode
        if mode in ("I;16", "I;16L", "I;16B", "I;16N"):
            a = np.array(im).astype(np.uint16)
        elif mode in ("I", "F"):
            a = np.array(im)
        else:
            if mode not in ("L", "RGB", "RGBA"):
                im = im.convert("RGBA" if ("A" in mode or "transparency" in im.info) else "RGB")
            a = np.array(im)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = old
    if a.ndim == 3:
        a = a[:, :, [2, 1, 0, 3][:a.shape[2]]]                    # RGB(A) -> BGR(A)
    return a, o


def imread_u(path: str, report=None):
    """читання зображення з будь-яким (юнікодним) шляхом; порядок спроб: OpenCV -> Pillow -> Pillow з відновленням
    пошкодженого файлу (тоді report(текст) отримує попередження). Повертає BGR uint8 вже з урахуванням EXIF-повороту"""
    ext = os.path.splitext(path)[1].lower()
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size and ext not in (".heic", ".heif") else None
    if img is not None:
        return _orient(_flatten(img, ext), exif_orientation(path))
    err = None
    for lenient in (False, True):
        try:
            a, o = _pil_read(path, lenient)
            out = _orient(_flatten(a, ext), o)
            if lenient:
                msg = "файл пошкоджений — показано те, що вдалося відновити"
                try:
                    from . import log as _log
                    _log.get().warning("%s: %s", path, msg)
                except Exception:  # noqa: BLE001
                    pass
                if report:
                    report(msg)
            return out
        except Exception as e:  # noqa: BLE001
            err = err or e
    name = os.path.basename(path)
    if ext in (".heic", ".heif") and not heif_ready():
        raise ValueError(tr("{name}: для HEIC/HEIF потрібен пакет pillow-heif (pip install pillow-heif)", name=name))
    if not data.size:
        raise ValueError(tr("{name}: файл порожній", name=name))
    raise ValueError(tr("не вдалося прочитати зображення {name}: формат не підтримується або файл пошкоджений ({err})", name=name, err=err))


class ImageSource:
    def __init__(self, path):
        notes = []
        self.img = imread_u(path, notes.append)
        h, w = self.img.shape[:2]
        self.info = dict(kind="image", fps=1.0, width=w, height=h, frames=1, duration=0.0, starts=None,
                         note=notes[0] if notes else "")

    def read(self, idx):
        return self.img

    def clone(self):
        return self

    def close(self):
        pass


class GifSource:
    """будь-яка анімація, яку вміє Pillow: GIF, APNG, анімований WebP/AVIF; усі кадри лежать у пам'яті (BGR) з білою підкладкою
    під прозорістю. Якщо розпакований обсяг завеликий - кадри зменшуються (з приміткою)"""
    def __init__(self, path):
        from PIL import Image, ImageSequence
        Image.MAX_IMAGE_PIXELS = None
        heif_ready()
        im = Image.open(path)
        n_est = max(1, int(getattr(im, "n_frames", 1) or 1))
        w0, h0 = im.size
        k = min(1.0, (ANIM_MAX_BYTES / max(1, n_est * w0 * h0 * 3)) ** 0.5)
        self.frames, durs = [], []
        for fr in ImageSequence.Iterator(im):
            d = fr.info.get("duration", 100) or 100
            rgba = fr.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            bg.alpha_composite(rgba)
            a = cv2.cvtColor(np.asarray(bg.convert("RGB")), cv2.COLOR_RGB2BGR)
            if k < 1.0:
                a = cv2.resize(a, (max(2, int(a.shape[1] * k)), max(2, int(a.shape[0] * k))), interpolation=cv2.INTER_AREA)
            self.frames.append(a)
            durs.append(max(20, int(d)) / 1000.0)         # браузери трактують <20 мс як 100 мс; беремо мінімум 20
        h, w = self.frames[0].shape[:2]
        starts = np.concatenate([[0.0], np.cumsum(durs)[:-1]])
        total = float(sum(durs))
        self.durs = durs
        self.info = dict(kind="gif", fps=len(durs) / total, width=w, height=h, frames=len(durs), duration=total,
                         starts=starts, note="кадри зменшено, щоб вмістилися в пам'ять" if k < 1.0 else "")

    def read(self, idx):
        return self.frames[idx] if 0 <= idx < len(self.frames) else None

    def clone(self):
        """вигляд на ті самі кадри для іншого потоку: закрити його можна без шкоди для оригіналу (раніше clone() повертав
        сам об'єкт, і c.close() в автопідборі розміру спустошував кадри - плеєр вічно «буферизував»)"""
        return _GifView(self)

    def close(self):
        self.frames = []            # прив'язує лише цей об'єкт до порожнього списку: вигляди ще тримають старий і дочитують його


class _GifView:
    def __init__(self, src):
        self.info, self.frames, self.durs = src.info, src.frames, src.durs

    def read(self, idx):
        return self.frames[idx] if 0 <= idx < len(self.frames) else None

    def clone(self):
        return _GifView(self)

    def close(self):
        self.frames = []


class VideoSource:
    def __init__(self, path):
        from .video import video_info, vfr_times
        info = video_info(path)
        starts = vfr_times(path, info)                         # змінна частота кадрів (телефон, месенджери): справжні моменти кадрів, див. video.vfr_times
        self.info = dict(info, kind="video", starts=starts)
        if starts is not None:
            self.info["duration"] = float(starts[-1] + (starts[-1] - starts[-2] if len(starts) > 1 else 0.0))
        self.path = path
        from .video import open_capture
        self.cap, self.pos = open_capture(path), -1

    def read(self, idx):
        cap = self.cap
        if cap is None:
            return None
        if self.pos < -1 or idx <= self.pos or idx - self.pos > 31:     # позиція невідома або далеко - seek
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        elif idx != self.pos + 1:                                        # трохи вперед - пропустити
            for _ in range(idx - self.pos - 1):
                cap.grab()
        ok, f = cap.read()
        self.pos = idx if ok else -2
        return f if ok else None

    def reset(self):
        self.pos = -2

    def clone(self):
        """окремий декодер для іншого потоку"""
        c = VideoSource.__new__(VideoSource)
        c.info, c.path = self.info, self.path
        from .video import open_capture
        c.cap, c.pos = open_capture(self.path), -1
        return c

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_source(path: str):
    from . import bake
    if bake.is_baked(path):                          # файл .ikr: кадри вже порахованi (режим із замочком), розширення не важить
        return bake.BakedSource(path)
    k = kind_of(path)
    return ImageSource(path) if k == "image" else GifSource(path) if k == "gif" else VideoSource(path)


# ---------------------------------------------------------------- збереження
def save_still(mask01: np.ndarray, path: str, palette=None):
    """маска 0/1 -> справжнє 2-кольорове зображення (PNG, BMP, TIFF або WebP без втрат - за розширенням); шлях може бути юнікодним.
    Класика - 1-бітне чорно-біле (TIFF - зі стисненням G4, як у факсів і сканерів); інша палітра - індексоване з двома кольорами"""
    from . import palette as pal
    ext = os.path.splitext(path)[1].lower() or ".png"
    fmt = {".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF", ".webp": "WEBP"}.get(ext, "PNG")
    try:
        from PIL import Image
        if pal.is_classic(palette) or pal.is_negative(palette):
            img = (mask01 > 0).astype(np.uint8) * 255
            if pal.is_negative(palette):                  # біле на чорному: теж справжнє 1-бітне, лише полярність протилежна
                img = 255 - img
            if fmt == "WEBP":
                Image.fromarray(img).convert("RGB").save(path, format="WEBP", lossless=True, quality=100, method=6)
            else:
                im = Image.fromarray(img).convert("1", dither=Image.NONE)
                im.save(path, format=fmt, **({"compression": "group4"} if fmt == "TIFF" else {}))
        else:
            im = Image.fromarray((mask01 > 0).astype(np.uint8), "P")
            im.putpalette(pal.pil_palette(palette))
            if fmt == "WEBP":
                im.convert("RGB").save(path, format="WEBP", lossless=True, quality=100, method=6)
            else:
                im.save(path, format=fmt, **({"compression": "tiff_lzw"} if fmt == "TIFF" else {}))
    except ImportError:
        img = (pal.paint(mask01, palette, bgr=True) if not (pal.is_classic(palette) or pal.is_negative(palette))
               else (mask01 > 0).astype(np.uint8) * 255 if pal.is_classic(palette) else (mask01 <= 0).astype(np.uint8) * 255)
        ok, buf = cv2.imencode(ext, img)
        buf.tofile(path)


def save_text(txt: str, path: str):
    """текст кадру -> файл UTF-8 (шлях може бути юнікодним)"""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)


def export_still(src_path, out, params: Params, size_key="Як у джерела", progress=None, palette=None):
    from .video import glyph_cell, out_size
    src = ImageSource(src_path)
    w, h = out_size(src.info["width"], src.info["height"], size_key, glyph_cell(params))
    if progress:
        progress(0.2, "Обробка")
    if os.path.splitext(out)[1].lower() == ".txt":                 # символи -> текст: світла клітинка - густий знак (як у консолі на чорному)
        from . import glyph
        save_text(glyph.text(src.img, w, h, params), out)
    else:
        save_still(binarize(src.img, w, h, params), out, palette)
    if progress:
        progress(1.0, "Обробка")
    return dict(frames=1, bytes=os.path.getsize(out), size=(w, h), fps=0.0, path=out)


def _resize(f, w, h):
    sh, sw = f.shape[:2]
    return f if (sw, sh) == (w, h) else cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA if w * h < sw * sh else cv2.INTER_CUBIC)


def export_gif_source(src_path, out, params: Params, fmt="GIF", size_key="240p", workers=None, progress=None,
                      cancel=None, palette=None):
    """GIF -> чорно-білий GIF (із збереженням тривалості кадрів) або PNG-послідовність"""
    from .video import Cancelled, _frame_job, default_workers, glyph_cell, kill_pool, make_pool, out_size
    src = GifSource(src_path)
    n, (w, h) = src.info["frames"], out_size(src.info["width"], src.info["height"], size_key, glyph_cell(params))
    workers = 1 if params.method == 6 else max(1, workers or default_workers())      # IRNM1: одна модель на відеокарті в цьому процесі
    pd = params.to_dict()
    jobs = [(_resize(f, w, h), w, h, pd) for f in src.frames]
    masks = []
    aux = None
    if params.method == 6:
        from .student import FilmState
        aux = {"film": FilmState()}                          # кадри GIF ідуть підряд: опис кадру плавно змінюється, як у відео
    if workers > 1 and n > 1:
        ex = make_pool(min(workers, n))
        try:
            futs = [ex.submit(_frame_job, j) for j in jobs]
            for i, fu in enumerate(futs):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                masks.append(np.unpackbits(fu.result(), axis=-1, count=w))
                if progress:
                    progress((i + 1) / n * 0.9, "Обробка")
        finally:
            kill_pool(ex)
    else:
        for i, j in enumerate(jobs):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            if aux is not None:
                from .core import binarize
                masks.append(binarize(j[0], w, h, params, aux))
            else:
                masks.append(np.unpackbits(_frame_job(j), axis=-1, count=w))
            if progress:
                progress((i + 1) / n * 0.9, "Обробка")
    if fmt == "PNG-послідовність":
        os.makedirs(out, exist_ok=True)
        for i, m in enumerate(masks):
            save_still(m, os.path.join(out, f"{i + 1:06d}.png"), palette)
        size_b = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    else:
        from PIL import Image
        from . import palette as pal
        ims = []
        for m in masks:
            im = Image.fromarray((m > 0).astype(np.uint8), "P")
            im.putpalette(pal.pil_palette(palette))
            ims.append(im)
        ims[0].save(out, save_all=True, append_images=ims[1:], duration=[int(round(d * 1000)) for d in src.durs],
                    loop=0, optimize=False, disposal=1)
        size_b = os.path.getsize(out)
    if progress:
        progress(1.0, "Обробка")
    return dict(frames=n, bytes=size_b, size=(w, h), fps=src.info["fps"], path=out)
