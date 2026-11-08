"""Відео: читання, паралельна бінаризація й експорт у вибраний формат (ffmpeg-пайп).

Кадри обробляються шматками у кількох процесах (відео ріжеться за номерами кадрів), результат у
порядку йде в ffmpeg. fps і тривалість лишаються як в оригіналі (точний дріб, 23.976 -> 24000/1001).
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import weakref
from collections import deque
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

import cv2
import numpy as np

from .core import Params, binarize
from .i18n import tr

VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wmv", ".mpg", ".mpeg", ".ts", ".m2ts", ".mts",
             ".3gp", ".3g2", ".ogv", ".vob", ".asf", ".f4v", ".divx", ".mxf", ".m2v", ".mpe", ".qt", ".rm", ".rmvb",
             ".ikr")                                  # «спечені» файли InkReel (benvid.bake): кадри вже пораховані

# назва в меню -> (розширення, ключ кодека)
IKR_FORMAT = "InkReel (.ikr, без втрат)"                   # ключ у FORMATS: власний формат (benvid.bake)
FORMATS = {
    "MP4 (H.264)": (".mp4", "mp4"),
    "MKV (H.264)": (".mkv", "mkv"),
    "MOV (H.264)": (".mov", "mov"),
    "AVI (MPEG-4)": (".avi", "avi"),
    "WebM (VP9)": (".webm", "webm"),
    "GIF": (".gif", "gif"),
    "JSON (стиснутий)": (".json.gz", "json"),
    "InkReel (.ikr, без втрат)": (".ikr", "ikr"),
    "PNG-послідовність": ("", "png"),
}
SIZES = {"128x64": 64, "320x240": 240, "144p": 144, "240p": 240, "360p": 360, "480p": 480, "720p": 720, "1080p": 1080,
         "1440p": 1440, "2160p": 2160, "Як у джерела": 0}     # висота результату (720p - межа реального часу; вище - для експорту/фото)
MAX_SIDE = 4096
QUALITY = {"Висока": 16, "Середня": 22, "Мала": 28}


class Cancelled(Exception):
    pass


def fps_fraction(fps: float) -> Fraction:
    """23.976 -> 24000/1001, 29.97 -> 30000/1001, 24 -> 24"""
    return Fraction(float(fps) or 24.0).limit_denominator(1001)


FPS_CHOICES = (60, 50, 30, 25, 24, 20, 15, 12, 10)       # стандартні значення «ліміту кадрів на секунду» (ще є «як в оригіналі»)


def fps_options(src_fps: float) -> list[int]:
    """які ліміти кадрів на секунду мають сенс для відео з частотою src_fps: лише нижчі за неї (0.5 к/с запасу: 29.97 -> 25 і нижче, не 30)"""
    return [v for v in FPS_CHOICES if v < float(src_fps or 0) - 0.5]


def parse_fps_limit(text, src_fps: float):
    """що вписав користувач у поле «кадри на секунду» (або що стоїть у списку): число к/с від 1 до частоти відео (з комою чи крапкою);
    0.0 - без ліміту (порожньо, 0, чи значення не нижче за частоту відео - воно б нічого не змінило); None - у тексті немає числа"""
    s = str(text if text is not None else "").strip()
    if not s:
        return 0.0
    m = re.search(r"\d+(?:[.,]\d+)?", s)
    if not m:
        return None
    v = float(m.group(0).replace(",", "."))
    if v <= 0 or v >= float(src_fps or 0) - 0.5:
        return 0.0
    return round(max(1.0, v), 3)


def fps_ratio(src_fps: float, target) -> float:
    """у скільки разів вихідний ряд кадрів рідший за початковий (1.0 - без змін). Вихідний кадр k береться з початкового int(k*ratio + 0.5):
    кадри вибираються за ЧАСОМ, тож тривалість і синхрон зі звуком лишаються, а змінюється лише частота"""
    try:
        t, f = float(target or 0), float(src_fps or 0)
    except (TypeError, ValueError):
        return 1.0
    return f / t if 0 < t < f - 0.01 else 1.0


def out_fps(src_fps: float, target) -> float:
    """частота результату: ліміт, якщо він нижчий за початкову, інакше початкова"""
    return float(target) if fps_ratio(src_fps, target) != 1.0 else float(src_fps)


def _src_index(k: int, ratio: float) -> int:
    return int(k * ratio + 0.5) if ratio != 1.0 else k


_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_PROBE: dict = {}


def hidden_kw() -> dict:
    """аргументи Popen/run, щоб консольна програма (ffmpeg) не відкривала власне чорне вікно: у зібраному exe без консолі Windows
    створює його для кожного запуску, а закриття цього вікна вбиває ffmpeg і валить експорт"""
    kw = dict(creationflags=_NOWIN)
    if os.name == "nt":
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0                       # SW_HIDE
            kw["startupinfo"] = si
        except Exception:  # noqa: BLE001
            pass
    return kw


def probe_stream(path: str) -> dict:
    """Розбір виводу `ffmpeg -i` (без ffprobe): size - кодований розмір, sar, fps, duration (с), rotation (градуси, 0 якщо
    немає), codec, hdr. Результат кешується; порожній dict, якщо ffmpeg недоступний."""
    key = path
    if key in _PROBE:
        return _PROBE[key]
    out: dict = {}
    ff = find_ffmpeg()
    if ff:
        try:
            r = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, timeout=30, creationflags=_NOWIN)
            txt = r.stderr.decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001
            txt = ""
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt)
        if m:
            out["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        out["has_audio"] = bool(re.search(r"Stream #\d+:\d+[^\n]*?:\s*Audio:", txt))
        out["error"] = txt.strip().splitlines()[-1][:200] if txt and "Input #" not in txt and txt.strip() else ""
        out["moov"] = "moov atom not found" in txt
        mv = re.search(r"Stream #\d+:\d+[^\n]*?:\s*Video:\s*([^\n]+)", txt)
        if mv:
            line = mv.group(1)
            out["codec"] = line.split(",")[0].split(" ")[0]
            m = re.search(r",\s*(\d{2,6})x(\d{2,6})(?=[\s,\[])", line)
            if m:
                out["size"] = (int(m.group(1)), int(m.group(2)))
            m = re.search(r"SAR\s+(\d+):(\d+)", line)
            if m and int(m.group(2)) > 0 and int(m.group(1)) > 0:
                out["sar"] = int(m.group(1)) / int(m.group(2))
            m = re.search(r",\s*([\d.]+)\s*fps", line) or re.search(r",\s*([\d.]+)\s*tbr", line)
            if m:
                out["fps"] = float(m.group(1))
            m = re.search(r",\s*([\d.]+)\s*tbr", line)
            if m:
                out["tbr"] = float(m.group(1))                  # «базовий» к/с контейнера; відрізняється від середнього fps у файлів зі змінною частотою кадрів
            out["hdr"] = bool(re.search(r"smpte2084|arib-std-b67", line))
            out["interlaced"] = bool(re.search(r"\((?:top|bottom) first\)", line))
        m = re.search(r"rotation of (-?[\d.]+) degrees", txt) or re.search(r"\brotate\s*:\s*(-?\d+)", txt)
        if m:
            out["rotation"] = int(round(float(m.group(1)))) % 360
    _PROBE[key] = out
    return out


class FFCapture:
    """Схожий на cv2.VideoCapture декодер на ffmpeg (кадри BGR із пайпа). Запасний варіант, коли OpenCV чогось не вміє:
    кодеки (AV1, HEVC, ProRes…), перемотка назад у .ts/.avi, поворот із метаданих телефона. Перемотка - перезапуск ffmpeg із
    потрібного часу (кадр = час/fps), короткі стрибки вперед - читанням."""

    def __init__(self, path: str, fps: float | None = None, frames: int = 0):
        self.path, self.ff = path, find_ffmpeg()
        pr = probe_stream(path) if self.ff else {}
        self.fps = float(fps or pr.get("fps") or 24.0)
        self.n = int(frames or round(pr.get("duration", 0) * self.fps) or 0)
        self.proc, self.pos, self.w, self.h, self.ok = None, 0, 0, 0, False
        self.deint = bool(pr.get("interlaced"))
        if not self.ff:
            return
        try:                                                      # розмір кадру - з першого кадру після поворотів (PPM-заголовок)
            r = subprocess.run(self._cmd(0.0, True), capture_output=True, timeout=120, creationflags=_NOWIN)
            m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+255\s", r.stdout)
            if m:
                self.w, self.h, self.ok = int(m.group(1)), int(m.group(2)), True
        except Exception:  # noqa: BLE001
            pass

    def _cmd(self, t: float, probe: bool = False) -> list[str]:
        cmd = [self.ff, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if t > 0:
            cmd += ["-ss", f"{t:.4f}"]
        cmd += ["-i", self.path, "-an", "-sn", "-dn"]
        if self.deint:                                           # чересрядкове відео (AVCHD, ТБ, DVD): прибрати «гребінець»
            cmd += ["-vf", "yadif=0:-1:1"]
        cmd += ["-frames:v", "1", "-pix_fmt", "rgb24", "-f", "image2pipe", "-c:v", "ppm", "-"] if probe \
            else ["-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        return cmd

    def isOpened(self) -> bool:
        return self.ok

    def _kill(self):
        p, self.proc = self.proc, None
        if p is not None:
            try:
                p.kill()
                p.stdout.close()
                p.wait(5)
            except Exception:  # noqa: BLE001
                pass

    def _spawn(self, idx: int):
        self._kill()
        t = max(0.0, (idx - 0.25) / self.fps) if idx > 0 else 0.0   # трохи раніше, щоб перший кадр був саме idx
        self.proc = subprocess.Popen(self._cmd(t), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                                     creationflags=_NOWIN)
        self.pos = idx

    def read(self):
        if not self.ok:
            return False, None
        if self.proc is None:
            self._spawn(self.pos)
        n = self.w * self.h * 3
        buf = bytearray(n)
        try:
            got = self.proc.stdout.readinto(buf)
        except Exception:  # noqa: BLE001
            got = 0
        if got != n:
            self._kill()
            return False, None
        self.pos += 1
        return True, np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)

    def grab(self) -> bool:
        if not self.ok:
            return False
        if self.proc is None:
            self._spawn(self.pos)
        n = self.w * self.h * 3
        try:
            ok = len(self.proc.stdout.read(n)) == n
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            self._kill()
            return False
        self.pos += 1
        return True

    def set(self, prop, value) -> bool:
        if prop == cv2.CAP_PROP_POS_MSEC:
            value, prop = value / 1000.0 * self.fps, cv2.CAP_PROP_POS_FRAMES
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return False
        idx = max(0, int(value))
        if idx == self.pos:
            return True
        if self.proc is not None and 0 < idx - self.pos <= 60:
            while self.pos < idx and self.grab():
                pass
            return True
        self._kill()
        self.pos = idx
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: self.n, cv2.CAP_PROP_FRAME_WIDTH: self.w,
                cv2.CAP_PROP_FRAME_HEIGHT: self.h, cv2.CAP_PROP_POS_FRAMES: self.pos}.get(prop, 0.0)

    def release(self):
        self._kill()

    def __del__(self):
        try:
            self._kill()
        except Exception:  # noqa: BLE001
            pass


def _open_cv(path: str):
    """cv2.VideoCapture, що відкриває й файли з кирилицею/юнікодом у шляху на Windows (через короткий шлях 8.3)"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened() and os.name == "nt" and not path.isascii():
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(32768)
            if ctypes.windll.kernel32.GetShortPathNameW(os.path.abspath(path), buf, 32768):
                cap.release()
                cap = cv2.VideoCapture(buf.value)
        except Exception:  # noqa: BLE001
            pass
    return cap


class RobustCapture:
    """VideoCapture, який сам переходить на ffmpeg-декодер (FFCapture), якщо OpenCV не відкрив файл, не віддав кадр чи не вміє
    перемотати (AV1, .ts, поворот телефона…). Інтерфейс - підмножина cv2.VideoCapture: read/grab/set/get/release."""

    def __init__(self, path: str, force_ff: bool = False, closed: bool = False):
        self.path, self.cv, self.ff, self.pos, self.n = path, None, None, 0, 0
        if closed:                                           # файл без відео (лише звук): не відкривати - OpenCV на Windows приймає mp3, а read() потім висить назавжди
            return
        if not force_ff:
            cap = _open_cv(path)
            if cap.isOpened():
                self.cv = cap
                self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            else:
                cap.release()
        if self.cv is None:
            self._to_ff(0)

    def _to_ff(self, pos: int) -> bool:
        if self.ff is None:
            ff = FFCapture(self.path, self.cv.get(cv2.CAP_PROP_FPS) if self.cv is not None else None, self.n)
            if not ff.ok:
                return False
            self.ff = ff
        if self.cv is not None:
            self.cv.release()
            self.cv = None
        self.ff.set(cv2.CAP_PROP_POS_FRAMES, pos)
        return True

    @property
    def backend(self) -> str:
        return "ffmpeg" if self.ff is not None else "opencv" if self.cv is not None else ""

    def isOpened(self) -> bool:
        return self.cv is not None or self.ff is not None

    def _failed(self) -> bool:
        """cv2 не віддав кадр: це кінець відео (тоді нічого) чи проблема декодера (тоді ffmpeg)"""
        return (self.pos == 0 or self.pos < self.n - 1) and self._to_ff(self.pos)

    def read(self):
        if self.ff is not None:
            return self.ff.read()
        if self.cv is None:
            return False, None
        ok, f = self.cv.read()
        if ok:
            self.pos += 1
            return True, f
        if self._failed():
            return self.ff.read()
        return False, None

    def grab(self) -> bool:
        if self.ff is not None:
            return self.ff.grab()
        if self.cv is None:
            return False
        if self.cv.grab():
            self.pos += 1
            return True
        return self._failed() and self.ff.grab()

    def set(self, prop, value) -> bool:
        if self.ff is not None:
            return self.ff.set(prop, value)
        if self.cv is None:
            return False
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = max(0, int(value))
        return self.cv.set(prop, value)

    def get(self, prop):
        if self.ff is not None:
            return self.ff.get(prop)
        return self.cv.get(prop) if self.cv is not None else 0.0

    def release(self):
        if self.cv is not None:
            self.cv.release()
            self.cv = None
        if self.ff is not None:
            self.ff.release()


_FF_FIRST = (".ts", ".m2ts", ".mts", ".mpg", ".mpeg", ".mpe", ".m2v", ".vob")       # OpenCV перемотує їх неточно


def open_capture(path: str):
    """захоплювач відео: OpenCV, а де він не годиться (поворот із метаданих телефона, чересрядковість, MPEG-потоки з
    неточною перемоткою, нерозпізнаний кодек) - ffmpeg"""
    pr = probe_stream(path) if find_ffmpeg() else {}
    force = bool(pr.get("rotation") or pr.get("interlaced")) or os.path.splitext(path)[1].lower() in _FF_FIRST
    if pr.get("has_audio") and "codec" not in pr:            # лише звук
        return RobustCapture(path, closed=True)
    return RobustCapture(path, force_ff=force and bool(pr))


def _probe_duration(path: str) -> float:
    """тривалість у секундах з виводу ffmpeg (коли контейнер не знає кількості кадрів, напр. webm/flv)"""
    return float(probe_stream(path).get("duration", 0.0))


_RELIABLE_N = (".mp4", ".m4v", ".mov", ".mkv", ".webm")


def _open_error(path: str, pr: dict) -> str:
    """зрозуміле пояснення, чому відео не відкрилося"""
    name = os.path.basename(path)
    if not os.path.exists(path):
        return tr("файл не знайдено: {name}", name=name)
    if os.path.getsize(path) == 0:
        return tr("{name}: файл порожній", name=name)
    if pr.get("moov"):
        return tr("{name}: файл недописаний або пошкоджений (запис було перервано)", name=name)
    if pr.get("has_audio") and "codec" not in pr:
        return tr("{name}: у файлі лише звук, відео немає", name=name)
    if not find_ffmpeg():
        return tr("{name}: не вдалося відкрити відео (немає ffmpeg для запасного декодера: pip install imageio-ffmpeg)", name=name)
    return tr("{name}: не вдалося відкрити — невідомий формат або файл пошкоджений", name=name)


def video_info(path: str) -> dict:
    if os.path.splitext(path)[1].lower() == ".ikr":            # власний формат: заголовок вже містить усе потрібне
        from . import bake
        try:
            rd = bake.open_baked(path)
        except bake.BakeError as e:
            raise ValueError(f"{os.path.basename(path)}: {e}") from None
        try:
            return rd.info()
        finally:
            rd.close()
    cap = open_capture(path)
    pr = probe_stream(path) if find_ffmpeg() else {}
    if not cap.isOpened():
        cap.release()
        raise ValueError(_open_error(path, pr))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1 or fps > 1000:       # 0, NaN або сміття
        fps = pr.get("fps") if 1 <= (pr.get("fps") or 0) <= 1000 else 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0 or n > 10 ** 8:                                # контейнер не знає кількості кадрів
        dur = _probe_duration(path)
        n = int(dur * fps) + 1 if dur > 0 else 0
        if n <= 1:                                           # крайній випадок: порахувати читанням
            n = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while n < 10 ** 7 and cap.grab():
                n += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, f = cap.read()                                       # розмір - із першого кадру (властивості контейнера бувають
    if not ok:                                               # до повороту/з іншими пропорціями)
        cap.release()
        raise ValueError(tr("{name}: не вдалося прочитати кадри (пошкоджений файл чи непідтримуваний кодек)", name=os.path.basename(path)))
    h, w = f.shape[:2]
    dur = pr.get("duration", 0.0)
    if n > 8 and (os.path.splitext(path)[1].lower() not in _RELIABLE_N or (dur and abs(n - dur * fps) > 1.5)):
        start = n - 8                                        # лічильник кадрів брешe на 1-2 (avi/mpg/ts): дочитуємо до справжнього кінця
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        cnt = 0
        while cnt < 24 and cap.grab():
            cnt += 1
        if cnt < 24:
            n = max(1, start + cnt)
    sar = pr.get("sar", 1.0)
    if sar and abs(sar - 1.0) > 0.02 and not pr.get("rotation"):    # неквадратні пікселі: показуємо з коректними пропорціями
        w = max(2, int(round(w * sar)))
    info = dict(fps=fps, width=int(w), height=int(h), frames=n, duration=n / fps if fps else 0.0)
    cap.release()
    if info["width"] <= 0 or info["height"] <= 0 or n <= 0:
        raise ValueError(tr("не вдалося прочитати відео (пошкоджений файл чи непідтримуваний кодек)"))
    return info


_VFR: dict = {}


def vfr_times(path: str, info: dict, timeout: float = 20.0):
    """Справжні моменти показу кадрів (с від першого кадру, масив довжини info["frames"]) для відео зі ЗМІННОЮ частотою кадрів (запис із телефона,
    переслане месенджером, екранні записи); None - якщо частота стала (звичайний випадок) чи міток не вдалось отримати. Дешево: у звичайного файлу
    без розбіжності «fps» і «tbr» у виводі ffmpeg нічого не читається; інакше `ffmpeg -c copy -f framecrc` проходить по пакетах без декодування (секунда на хвилину
    відео), рахується один раз на файл. Плеєр тоді показує кадр за годинником, а не за номером ÷ середній к/с, і ритм руху збігається зі справжнім."""
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    if key in _VFR:
        return _VFR[key]
    res = None
    try:
        pr = probe_stream(path)
        fps, tbr = pr.get("fps") or 0.0, pr.get("tbr") or 0.0
        ff = find_ffmpeg()
        if ff and info.get("frames", 0) > 8 and fps and tbr and abs(tbr - fps) / fps > 0.0008:
            r = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-i", path, "-map", "0:v:0", "-c", "copy", "-f", "framecrc", "-"],
                               capture_output=True, timeout=timeout, **hidden_kw())
            txt = r.stdout.decode("ascii", "ignore")
            m = re.search(r"#tb 0:\s*(\d+)/(\d+)", txt)
            pts = [int(p) for p in (ln.split(",")[2].strip() for ln in txt.splitlines() if ln and ln[0] not in "#" and ln.count(",") >= 4) if re.fullmatch(r"-?\d+", p)]
            if m and int(m.group(2)) > 0 and len(pts) > 8:
                t = np.sort(np.asarray(pts, np.float64)) * (int(m.group(1)) / int(m.group(2)))
                t -= t[0]
                n = int(info["frames"])
                if abs(len(t) - n) <= 2:                                  # лічильники кадрів збігаються (декодер міг віддати на 1-2 менше/більше)
                    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / fps
                    if len(t) < n:
                        t = np.concatenate([t, t[-1] + dt * np.arange(1, n - len(t) + 1)])
                    t = t[:n]
                    avg = (n - 1) / max(float(t[-1]), 1e-6)
                    dev = float(np.abs(t - np.arange(n) / avg).max())
                    if dev > 0.5 / avg and np.all(np.diff(t) >= 0):       # відхилення від рівної сітки понад пів кадру - справді змінна частота
                        res = t
    except Exception:  # noqa: BLE001
        res = None
    _VFR[key] = res
    return res


def find_ffmpeg() -> str | None:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def parse_size(key: str):
    """'720p' | 'Як у джерела' | '320x180' | '320x' | 'x180' -> (W|None, H|None)"""
    s = str(key).strip().lower().replace("×", "x").replace("х", "x").replace(" ", "")
    if "x" in s and not s.endswith("p"):
        a, _, b = s.partition("x")
        try:
            return (int(a) if a else None), (int(b) if b else None)
        except ValueError:
            raise ValueError(tr("не розумію розмір {key}: очікується 320x180, 320x або x180", key=repr(key)))
    return None, None


def glyph_cell(params) -> tuple[int, int] | None:
    """найменший розмір результату в «Символах»: одна клітинка знака при поточній крупності (32x64, 16x32 або 8x16); для чорнила - None"""
    if params is None or not getattr(params, "glyph", False):
        return None
    from . import glyph
    cw = glyph.DENS_CW[glyph.dens_of(getattr(params, "gdens", glyph.DENS_DEFAULT)) - 1]
    return cw, 2 * cw


def out_size(src_w: int, src_h: int, size_key: str = "720p", cell: tuple[int, int] | None = None) -> tuple[int, int]:
    """розмір результату. Пресет (за висотою) або свій «ШxВ» (одна з сторін
    може бути порожньою - друга береться за пропорціями). Сторони парні (для H.264), від 2 до MAX_SIDE.
    cell=(w, h) - найменший розмір (одна клітинка знака в «Символах»): менший результат збільшується до нього - з пропорціями кадру
    для пресета й однієї сторони, а в явному «ШxВ» кожна сторона окремо"""
    cw, ch = parse_size(size_key)
    if cw or ch:
        if cw and ch:
            w, h = cw, ch                                   # явні «ШxВ»: кожна сторона обмежується окремо
        elif cw:
            w, h = cw, cw * src_h / src_w
        else:
            w, h = ch * src_w / src_h, ch
    else:
        th = SIZES.get(size_key, 720)
        if th:
            w, h = src_w * th / src_h, th
        else:
            w, h = src_w, src_h
    if not (cw and ch) and max(w, h) > MAX_SIDE:            # пропорції джерела зберігаються й коли сторона впирається в стелю
        k = MAX_SIDE / max(w, h)                            # (раніше 2160p для 21:9 давало 4096x2160 - картинка розтягувалась по вертикалі)
        w, h = w * k, h * k
    if cell:
        if cw and ch:
            w, h = max(w, cell[0]), max(h, cell[1])
        else:
            k = max(1.0, cell[0] / max(w, 1e-6), cell[1] / max(h, 1e-6))
            w, h = w * k, h * k
    ev = lambda x: min(MAX_SIDE, max(2, int(x / 2 + 0.5) * 2))
    return ev(w), ev(h)


# ---------------------------------------------------------------- паралельна обробка
def _lower_priority():
    """воркери працюють із нижчим пріоритетом, щоб вікно й система не підвисали навіть на слабких ПК"""
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
        else:
            os.nice(5)
    except Exception:  # noqa: BLE001
        pass


def _init_worker(gpu_sem=None):
    """gpu_sem - семафор місць для відеокарти: перші воркери, що його взяли, рахують на GPU, решта - на CPU"""
    _lower_priority()
    try:
        from . import gpu
        gpu.set_allowed(gpu_sem is not None and gpu_sem.acquire(False))     # без місця - pyopencl навіть не імпортується
    except Exception:  # noqa: BLE001
        pass
    cv2.setNumThreads(1)
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


_TLS = threading.local()
_TICKER = threading.local()           # .fn(скільки кадрів шматка готово) - виставляє господар обробки (host.py): смуга прогресу рухається й між шматками


class _Caps:
    """відкриті декодери процесу-воркера: у господарі (потоки) у кожного потоку свої - один декодер не читають двоє одночасно"""
    _all: list = []                                   # словники всіх потоків: щоб після експорту закрити декодери й чужих потоків (Windows не дає видалити відкритий файл)
    _all_lock = threading.Lock()

    def _d(self):
        d = getattr(_TLS, "caps", None)
        if d is None:
            d = _TLS.caps = {}
            with self._all_lock:
                self._all.append(d)
        return d

    def release_path(self, path=None):
        """закрити декодери файлу path (усі, якщо None) у всіх потоках цього процесу; викликати, коли робота з файлом скінчилась"""
        with self._all_lock:
            dicts = list(self._all)
        for d in dicts:
            for k in [k for k in list(d) if path is None or k == path]:
                try:
                    d.pop(k)[0].release()
                except Exception:  # noqa: BLE001
                    pass

    def get(self, k):
        return self._d().get(k)

    def pop(self, k):
        return self._d().pop(k)

    def __setitem__(self, k, v):
        self._d()[k] = v

    def __len__(self):
        return len(self._d())

    def __iter__(self):
        return iter(list(self._d()))

    def clear(self):
        for e in self._d().values():
            try:
                e[0].release()
            except Exception:  # noqa: BLE001
                pass
        self._d().clear()

    def __getitem__(self, k):
        return self._d()[k]

    def values(self):
        return list(self._d().values())

    def items(self):
        return list(self._d().items())


_CAPS = _Caps()


def _read_at(ent, idx):
    """кадр idx з відкритого декодера ent=[захоплювач, номер наступного кадру]: близько попереду - пропуск через grab(), далі - seek.
    Невдале читання теж «з'їдає» номер (як і раніше: бите місце пропускається, а не перемотується знову й знову)"""
    cap, nxt = ent[0], ent[1]
    if idx != nxt:
        skipped = nxt >= 0 and 0 < idx - nxt <= 31
        if skipped:
            for _ in range(idx - nxt):
                if not cap.grab():
                    skipped = False
                    break
        if not skipped:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    ent[1] = idx + 1
    return f if ok else None


_FILM_CONT: dict = {}                               # IRNM1: ключ шматка -> (FilmState, номер останнього кадру відео), щоб наступний шматок поруч не розігрівався
_ACC = {"warm": 0.0, "dec": 0.0, "proc": 0.0, "pack": 0.0, "frames": 0}      # секунди потоків по стадіях (лише для журналу: де йде час)
_ACC_LOCK = threading.Lock()


def take_acc() -> dict:
    """накопичені секунди потоків: розігрів пам'яті перед шматком (warm), читання кадрів (dec), бінаризація (proc), упакування бітів (pack), кадрів"""
    with _ACC_LOCK:
        return {k: (v if k == "frames" else round(v, 2)) for k, v in _ACC.items()}


def _job(job):
    """шматок відео: кадри start, start+step, … (count штук; step > 1 - слабкий ПК рахує кожен N-й кадр). Відкритий декодер лишається
    в процесі: наступний шматок поруч - без seek. Приймає (path, start, count, w, h, params[, step[, cold[, ratio]]]); cold - без розігріву
    часової пам'яті (терміновий перший кадр після відкриття чи перемотки). ratio != 1 - ліміт кадрів на секунду (див. fps_ratio):
    тоді start - номер ВИХІДНОГО кадру, а з відео береться кадр int(k*ratio + 0.5) (step тоді не вживається).
    Десятий елемент - явний список кадрів відео (плеєр з лімітом к/с), одинадцятий - кадри розігріву перед ним"""
    path, start, count, w, h, pd = job[:6]
    step = max(1, int(job[6])) if len(job) > 6 else 1
    cold = bool(job[7]) if len(job) > 7 else False
    ratio = float(job[8]) if len(job) > 8 else 1.0
    srcs = job[9] if len(job) > 9 else None          # явний список кадрів відео (плеєр з лімітом к/с: проміжки нерівні, їх рахує плеєр)
    pre_x = job[10] if len(job) > 10 else None       # і кадри перед ним для розігріву (ті самі, що йшли б перед ним у відео з лімітом)
    stream = job[11] if len(job) > 11 else None      # ім'я спільної пам'яті: готові кадри лягають туди одразу, а не віддаються цілим шматком (плеєр)
    wb = (w + 7) // 8
    sh = hdr = None
    if stream:
        sh = _attach_shm(stream)
        hdr = np.ndarray((4,), np.int64, buffer=sh.buf)          # [0] скільки кадрів готово, [1] != 0 - плеєр скасував шматок
    p = Params.from_dict(pd)
    ent = _CAPS.get(path)
    if ent is None:
        if len(_CAPS) > 2:
            for k in list(_CAPS):
                _CAPS.pop(k)[0].release()
        ent = _CAPS[path] = [open_capture(path), 0]
    st = None
    aux = None
    t_warm = t_dec = t_proc = t_pack = 0.0
    t_job = time.perf_counter()
    last_src = -1
    ckey = None
    if p.method == 6:                                # IRNM1: пам'ять - лише згладжування опису кадру; шматок поруч із попереднім її продовжує
        from .core import neuro_warm
        from .student import FilmState, WARM as NEURO_WARM
        if srcs is not None:
            pre = [] if cold else [int(i) for i in (pre_x or ())][-NEURO_WARM:]
        elif ratio != 1.0:
            pre = [] if cold else [_src_index(k, ratio) for k in range(max(0, start - NEURO_WARM), start)]
        else:
            pre = [] if cold else [i for i in range(start - step * NEURO_WARM, start, step) if i >= 0]
        ckey = (path, w, h, tuple(sorted(pd.items())))
        hit = _FILM_CONT.get(ckey)
        if hit is not None and pre and hit[1] == pre[-1]:
            st = hit[0]                              # шматок іде одразу за попереднім: стан уже на місці, розігрів не потрібен
        else:
            st = FilmState()
            for wi in pre:
                if hdr is not None and hdr[1]:
                    break
                f = _read_at(ent, wi)
                if f is None:
                    break
                neuro_warm(f, w, h, st)
        _FILM_CONT.pop(ckey, None)                   # поки шматок іде, стан «в роботі»
        aux = {"film": st}
    if p.method == 5:                                # часова стійкість: розігрів пам'яті кадрами перед шматком
        from . import ink5, glyph
        from .core import PRIME, InkState
        from .glyph import WARM as GLYPH_WARM, WARM_CHEAP as GLYPH_CHEAP
        st = InkState()
        nwarm = GLYPH_WARM + GLYPH_CHEAP if p.glyph else ink5.WARM     # символи: пам'ять - тон і знаки; кілька кадрів лише тон, останні - повністю
        if srcs is not None:
            pre = [] if cold else [int(i) for i in (pre_x or ())][-(nwarm + PRIME):]
        elif ratio != 1.0:
            pre = [] if cold else [_src_index(k, ratio) for k in range(max(0, start - (nwarm + PRIME)), start)]
        else:
            pre = [] if cold else [i for i in range(start - step * (nwarm + PRIME), start, step) if i >= 0]
        nprime = max(0, len(pre) - nwarm)                # найдальші кадри: лише мініатюра для пам'яті нормування тону (згасання тягнеться довго)
        for j, wi in enumerate(pre):
            if hdr is not None and hdr[1]:
                break
            f = _read_at(ent, wi)
            if f is None:
                break
            if j < nprime:
                (glyph.prime(st, f, w, h, p.auto, p.gdens) if p.glyph else ink5.prime(st, f, w, h, p.auto))
            elif p.glyph and j < len(pre) - GLYPH_WARM:
                glyph.plan(f, w, h, p, st, warm_only=True)           # символи: далекі кадри - лише тон (вибір знаків не потрібен)
            elif j < len(pre) - 1 and not p.glyph:
                ink5.warm(st, f, w, h, p)            # далекі кадри - лише світлота (дешево)
            else:
                binarize(f, w, h, p, {"state": st})
    out, miss, n_out, last = [], 0, 0, None

    def idx_of(k):
        return int(srcs[k]) if srcs is not None else _src_index(start + k, ratio) if ratio != 1.0 else start + k * step
    pre_ex = nxt = None
    if p.method == 6 and count > 1:
        from concurrent.futures import ThreadPoolExecutor
        pre_ex = ThreadPoolExecutor(1)
        nxt = pre_ex.submit(_read_at, ent, idx_of(0))

    def emit(a):
        nonlocal n_out, last
        last = a
        if hdr is not None:                          # кадр одразу в спільну пам'ять (повністю записаний - тоді лічильник), плеєр забирає його не чекаючи кінця шматка
            np.ndarray((h, wb), np.uint8, buffer=sh.buf, offset=32 + n_out * h * wb)[:] = a
            n_out += 1
            hdr[0] = n_out
        else:
            out.append(a)
            n_out += 1
    t_warm = time.perf_counter() - t_job
    tick = getattr(_TICKER, "fn", None)
    for k in range(count):
        if hdr is not None and hdr[1]:               # плеєр скасував шматок (перемотка, інший розмір чи якість): решту не рахуємо
            break
        if tick is not None and n_out:
            tick(n_out)
        t0 = time.perf_counter()
        last_src = idx_of(k)
        if pre_ex is not None:                       # IRNM1: наступний кадр декодується, поки мережа рахує поточний
            f = nxt.result() if nxt is not None else _read_at(ent, last_src)
            nxt = pre_ex.submit(_read_at, ent, idx_of(k + 1)) if k + 1 < count else None
        else:
            f = _read_at(ent, last_src)
        t_dec += time.perf_counter() - t0
        if f is None:                                # битий кадр (або кінець відео): рішення відкладаємо
            miss += 1
            if miss > 3 or not n_out:
                break
            continue
        if miss:                                     # після битих кадрів пішли нормальні - дублюємо попередній, щоб не з'їхали час і звук
            for _ in range(miss):
                emit(last)
            miss = 0
        t0 = time.perf_counter()
        m = binarize(f, w, h, p, aux if aux is not None else {"state": st} if st is not None else None)
        t1 = time.perf_counter()
        emit(np.packbits(m, axis=-1))
        t_proc += t1 - t0
        t_pack += time.perf_counter() - t1
    if pre_ex is not None:
        if nxt is not None:
            nxt.result()                             # декодер спільний: дочекатись, перш ніж хтось ще з ним працюватиме
        pre_ex.shutdown()
    if ckey is not None and last_src >= 0 and n_out:
        if len(_FILM_CONT) > 4:
            _FILM_CONT.pop(next(iter(_FILM_CONT)))
        _FILM_CONT[ckey] = (st, last_src)            # стан відповідає останньому порахованому кадру: наступний шматок поруч його продовжить
    with _ACC_LOCK:
        _ACC["warm"] += t_warm
        _ACC["dec"] += t_dec
        _ACC["proc"] += t_proc
        _ACC["pack"] += t_pack
        _ACC["frames"] += n_out
    if sh is not None:
        hdr = None                                   # (представлення пам'яті не повинні жити, поки її закриваємо)
        try:
            sh.close()
        except Exception:  # noqa: BLE001
            pass
        return np.zeros((n_out, 0, 0), np.uint8)     # кадри вже в спільній пам'яті; довжина - скільки їх готово
    return np.stack(out) if out else np.zeros((0, h, wb), np.uint8)


def _attach_shm(name):
    """підключитись до спільної пам'яті плеєра (створює і знищує її він): без стеження, щоб воркер її не знищив при виході"""
    from multiprocessing import shared_memory as sm
    try:
        return sm.SharedMemory(name=name, track=False)
    except TypeError:
        return sm.SharedMemory(name=name)


def _ink_warm_job(glyph=False):
    """прогрів «Ультри» в одному процесі пулу: компіляція (або читання з кешу) ядер. Решта процесів потім лише підвантажує
    готове з кешу, а не компілює одне й те саме паралельно (це пам'ять і хвилина зайвого часу на першому запуску).
    glyph - режим «Символи»: ще й банк знаків (шрифт, ~2500 малюнків) та ядра вибору знака"""
    if glyph:
        from . import glyph as gl
        gl.prewarm()
    else:
        from . import ink5
        ink5.prewarm()
    try:
        import numba
        return str(numba.threading_layer())          # плеєр перевіряє, чи витримає цей шар паралельні виклики з потоків господаря
    except Exception:  # noqa: BLE001
        return "?"


def ram_bytes() -> int:
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                            ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
            m = MS()
            m.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.total)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:  # noqa: BLE001
        return 8 << 30


def ram_free_bytes() -> int:
    """вільна фізична пам'ять зараз (0 - невідомо)"""
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                            ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
            m = MS()
            m.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.avail)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except Exception:  # noqa: BLE001
        return 0


def proc_rss_bytes() -> int:
    """скільки пам'яті займає цей процес зараз (Windows: приватні сторінки, як «Пам'ять» у диспетчері завдань; 0 - невідомо)"""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]
            m = PMC()
            m.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.GetCurrentProcess.argtypes = []
            ps = ctypes.WinDLL("psapi")
            ps.GetProcessMemoryInfo.restype = wintypes.BOOL                     # без argtypes 64-бітний «псевдо-дескриптор» (-1) не вміщається в int
            ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            ok = ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(m), m.cb)
            return int(m.PrivateUsage or m.WorkingSetSize) if ok else 0
        with open("/proc/self/statm", "rb") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001
        return 0


def kill_pool(ex, wait: float = 0.0):
    """зупинити пул і одразу завершити його процеси. shutdown(wait=False) лишає воркерів доробляти поточний шматок - на 4K це десятки
    секунд, а пам'ять (≈1 ГБ на процес) не звільняється: новий пул експорту поверх старого вичерпував пам'ять (MemoryError при старті)"""
    if ex is None:
        return
    _WF_CACHE.clear()                                # пам'ять звільниться - кількість процесів порахуємо знову
    try:
        procs = list(getattr(ex, "_processes", {}).values())
    except Exception:  # noqa: BLE001
        procs = []
    def _stop():
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass

    if threading.current_thread() is getattr(ex, "_executor_manager_thread", None):
        # виклик із колбека майбутнього (його виконує потік-менеджер пулу; при зламаному пулі - ще й під замком shutdown):
        # shutdown звідси чекав би на замок, який тримає сам цей потік, - назавжди. Процес після закриття вікна не завершувався
        # (завис у concurrent.futures.process._python_exit) і блокував повторний запуск. Зупиняємо в окремому потоці
        threading.Thread(target=_stop, daemon=True, name="benvid-pool-stop").start()
    else:
        _stop()
    for p in procs:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    end = time.time() + wait                          # дочекатися, поки ОС забере пам'ять (потрібно, коли одразу піднімається новий пул)
    for p in procs:
        left = end - time.time()
        if left <= 0:
            break
        try:
            p.join(left)
        except Exception:  # noqa: BLE001
            pass


def pool_kwargs(size=None) -> dict:
    """аргументи ProcessPoolExecutor: spawn, ініціалізатор і місця для відеокарти"""
    try:
        from . import gpu
        n = gpu.worker_slots(int(size[0]) * int(size[1])) if size else gpu.worker_slots()
    except Exception:  # noqa: BLE001
        n = 0
    ctx = multiprocessing.get_context("spawn")
    try:
        from .log import get_logger
        from . import gpu
        d = gpu.list_devices(allow_cpu=gpu.mode() == "cpu") if n else []
        get_logger().info("GPU: %s, pool processes with GPU: %d (mode %s)", d[0][1].name.strip() if d else "—", n, gpu.mode())
    except Exception:  # noqa: BLE001
        pass
    return dict(mp_context=ctx, initializer=_init_worker, initargs=(ctx.Semaphore(n) if n else None,))


def host_enabled() -> bool:
    """потоки в одному процесі-господарі (benvid.host) замість пулу процесів. ТИПОВО ВИМКНЕНО: заміри на 28 ядрах (1080p, 600 кадрів,
    ab_speed/ab2/ab3) показали, що пул процесів швидший за господаря на 45-90% (Ультра 8.0 проти 5.2 к/с, рівень 4 21 проти 12.5,
    рівень 3 34 проти 23.5): у потоках одного процесу малі виклики cv2 і numpy сперечаються за GIL (мікротест: 15 потоків дають 28%
    швидкості 15 процесів на cv2 і 49% на numpy), а великі numba-ядра масштабуються однаково. INKREEL_HOST=1 вмикає господаря"""
    return os.environ.get("INKREEL_HOST", "0").strip().lower() in ("1", "on", "yes", "true")


_LIVE_POOLS = weakref.WeakSet()                    # усі пули, створені make_pool і ще не вбиті: щоб закриття вікна не лишало жодного живим


def kill_all_pools(wait: float = 0.0):
    """вікно закривається: вбити всі пули (плеєра, експорту, GIF), навіть якщо потік, що їх створив, ще не встиг цього зробити.
    Інакше інтерпретатор при виході дочікується робітників, які дораховують шматок (на 4K - хвилини), і процес висить у фоні"""
    for ex in list(_LIVE_POOLS):
        try:
            kill_pool(ex, wait)
        except Exception:  # noqa: BLE001
            pass


def make_pool(n: int, size=None):
    """виконавець завдань для плеєра, експорту й GIF: пул із n процесів (за замовчуванням) або процес-господар із n потоками (INKREEL_HOST=1).
    Обидва мають submit(fn, *args) -> Future, shutdown() і _processes, тож kill_pool працює з будь-яким"""
    ex = _make_pool(n, size)
    try:
        _LIVE_POOLS.add(ex)
    except TypeError:
        pass
    return ex


def _make_pool(n: int, size=None):
    if n > 1 and host_enabled():
        try:
            from . import gpu
            from .host import RenderHost
            slots = gpu.worker_slots(int(size[0]) * int(size[1])) if size else gpu.worker_slots()
            try:
                from .log import get_logger
                d = gpu.list_devices(allow_cpu=gpu.mode() == "cpu") if slots else []
                get_logger().info("GPU: %s, render host: %d threads, GPU %s (mode %s)", d[0][1].name.strip() if d else "-", n,
                                  "on" if slots else "off", gpu.mode())
            except Exception:  # noqa: BLE001
                pass
            return RenderHost(n, gpu_on=slots > 0)
        except Exception:  # noqa: BLE001
            try:
                from .log import get_logger
                get_logger().warning("render host failed to start, using a process pool", exc_info=True)
            except Exception:  # noqa: BLE001
                pass
    return ProcessPoolExecutor(n, **pool_kwargs(size))


def default_workers() -> int:
    """скільки процесів-обробників. До 8 логічних ядер - усі, крім одного для інтерфейсу. Далі - половина й ще один: заміри на 28 ядрах
    показали, що 13 процесів дають ту саму швидкість, що й 27 (Ультра: 30 к/с проти 30; рівень 4: 71 проти 80), а процесор у другому
    випадку завантажений на 100% - вікно й система тупляться, а кожен зайвий процес - ще сотні МБ пам'яті й секунди старту.
    Не більше, ніж вміщується в пам'ять (кожен процес ~150-300 МБ), і не більше 60 (ліміт ProcessPoolExecutor у Windows - 61).
    Процеси мають нижчий за звичайний пріоритет"""
    cpu = os.cpu_count() or 2
    n = 1 if cpu <= 2 else cpu - 1 if cpu <= 8 else cpu // 2 + 1
    by_ram = int((ram_bytes() * 0.4) // (280 << 20))
    return max(1, min(n, by_ram, 60))


_WF_CACHE: dict = {}


def workers_for(w: int, h: int, base: int | None = None) -> int:
    """скільки процесів потягне пам'ять при кадрі w×h: на 4K один обробник займає ~1 ГБ (проміжні масиви ~120 байт
    на піксель), тому 14 процесів × 4K = вильоти з нестачі пам'яті. Бюджет - 40% ОЗП, але не більше 70% того, що вільно зараз:
    інакше Windows роздуває файл підкачки на системному диску на десятки ГБ, а то й MemoryError при старті процесів.
    Число запам'ятовується до наступного kill_pool: пул не має перестворюватися через коливання вільної пам'яті."""
    base = base or default_workers()
    key = (int(w), int(h), base)
    hit = _WF_CACHE.get(key)
    if hit is not None:
        return hit
    per = (150 << 20) + 120 * max(1, int(w) * int(h))
    budget = ram_bytes() * 0.4
    free = ram_free_bytes()
    if free:
        budget = min(budget, free * 0.7)
    n = max(1, min(base, int(budget // per)))
    _WF_CACHE[key] = n
    return n


def _ordered(ex, plan, window, ramp=True, on_wait=None, cap_bytes=768 << 20, cancel=None, pause=None):
    """результати у порядку плану. window - скільки завдань одночасно ЙДЕ або чекає в черзі пулу; готові, але ще не віддані (чекають
    попередніх) у це число не входять. Раніше вони займали ті самі місця: коли голова черги затримувалась, решта доробляла й «забивала»
    вікно готовим, нові завдання не додавались, і всі потоки разом простоювали - на графіку навантаження виходила гребінка (100% -> 0).
    Пам'ять тепер обмежена не кількістю завдань, а розміром готового-невіддаваного (cap_bytes).
    Елемент плану: ("j", job) - порахувати в пулі; ("c", масив) - уже готово (з буфера плеєра).
    ramp: у польоті спершу 3 завдання, далі +1 кожні 0.8 с - процес пулу створюється, лише коли для нього є завдання, тож усі процеси
    не стартують разом (десятки процесів, що одночасно імпортують numpy/cv2/numba, душать диск і процесор - система підвисає).
    cancel/pause - Event-и експорту: скасування діє й тоді, коли чекаємо повільний шматок (раніше кнопка «Стоп» мовчала, доки він не
    дорахується), а на паузі нові завдання не додаються"""
    from concurrent.futures import Future
    it = iter(plan)
    t0 = time.perf_counter()
    wake = threading.Event()                   # будь-яке завдання завершилось: одразу додаємо нове, а не за таймером

    def start(item):
        if item[0] == "c":
            f = Future()
            f.set_result(item[1])
            return f
        f = ex.submit(_job, item[1])
        f.add_done_callback(lambda _f: wake.set())
        return f

    def lim():
        if not ramp or window <= 4:
            return window
        return min(window, 3 + int((time.perf_counter() - t0) / 0.8))

    q = deque()
    head = []                                  # завдання, яке зараз чекаємо (воно вже вийшло з черги)

    def ok(f) -> bool:
        return f.done() and not f.cancelled() and f.exception() is None

    def finished():
        for f in (*head, *q):
            if ok(f):
                try:
                    yield f.result()
                except Exception:  # noqa: BLE001
                    pass

    def pending() -> int:
        """кадрів, уже порахованих, але ще не віддалених у порядку: дораховані шматки, що чекають на попередніх, і частково дорахована частина тих, що йдуть"""
        n = sum(len(r) for r in finished())
        part = getattr(ex, "partial_frames", None)
        return n + (part() if part is not None else 0)

    def fill():
        if pause is not None and pause.is_set():
            return
        act = sum(1 for f in (*head, *q) if not f.done())
        while act < lim():
            if sum(getattr(r, "nbytes", 0) for r in finished()) >= cap_bytes:
                return                         # готового-невіддаваного забагато: чекаємо, доки голова звільнить місце
            nxt = next(it, None)
            if nxt is None:
                return
            f = start(nxt)
            q.append(f)
            act += 0 if f.done() else 1

    fill()
    while q:
        fut = q.popleft()
        head[:] = [fut]
        while not fut.done():
            fill()
            wake.wait(0.25 if on_wait is not None else 0.4)
            wake.clear()
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            if on_wait is not None and not fut.done():
                on_wait(pending())
        head.clear()
        fill()
        yield fut.result()


def _make_plan(path, params, size, n, chunk, cache, a=0, workers=None, lo=None, ratio=1.0, last=None):
    """поділ відео на шматки: уже пораховані кадри (cache: idx -> packed) беруться готовими, решта - у пул. n й a - у ВИХІДНИХ кадрах;
    ratio != 1 (ліміт кадрів на секунду) - вихідний кадр k це кадр int(k*ratio + 0.5) відео (last - останній кадр відео), у кеші плеєра шукаємо його"""
    w, h = size
    pd = params.to_dict()
    plan, s, nj = [], a, 0
    if ratio != 1.0:
        top = last if last is not None else int(n * ratio)

        def at(k):
            return min(top, _src_index(k, ratio))
    else:
        def at(k):
            return k
    while s < n:
        e = s + 1
        if cache is not None and at(s) in cache:
            while e < n and at(e) in cache and e - s < 512:
                e += 1
            plan.append(("c", np.stack([cache[at(i)] for i in range(s, e)])))
        else:
            # кероване нарізання: початок - великі шматки (менше витрат на розігрів), далі вони меншають до lo, тож в кінці
            # всі процеси закінчують разом, а на коротких кліпах роботи вистачає всім ядрам
            lim = chunk if not workers else max(lo or 1, min(chunk, -(-(n - s) // (workers * 3))))
            if workers and workers > 1 and nj < workers:
                # перша хвиля - шматки зростаючої довжини: процеси закінчують їх по черзі, а не всі разом (кожен далі тримає свою
                # фазу), тож готові кадри йдуть до кодувальника рівним струменем, а перший кадр з'являється за секунди, не за хвилину
                lim = max(lo or 1, min(lim, round(lim * (nj + 1) / workers)))
            nj += 1
            while e < n and (cache is None or at(e) not in cache) and e - s < lim:
                e += 1
            plan.append(("j", (path, s, e - s, w, h, pd) if ratio == 1.0 else (path, s, e - s, w, h, pd, 1, False, ratio)))
        s = e
    return plan


def wait_pause(pause, cancel=None):
    """експорт на паузі (pause - threading.Event): чекаємо, доки її знімуть; скасування перериває очікування"""
    while pause is not None and pause.is_set():
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        time.sleep(0.05)


def binarize_video(path, params: Params, size, *args, **kw):
    """Генератор шматків масок (n,h,w) uint8 {0,1} у порядку кадрів (аргументи - як у packed_video)"""
    gen = packed_video(path, params, size, *args, **kw)
    try:
        for packed in gen:
            yield np.unpackbits(packed, axis=-1, count=size[0])
    finally:
        gen.close()


def packed_video(path, params: Params, size, workers=None, chunk=None, max_frames=None, progress=None, cancel=None,
                 cache=None, start=0, end=None, pause=None, fps=None):
    """Генератор шматків масок у порядку кадрів, упакованих по 8 пікселів: (n,h,(w+7)//8) uint8. Розпаковувати по одному кадру
    (так дешевше за пам'яттю: шматок 96 кадрів 1440p у розпакованому вигляді - майже півгігабайта). cache - уже готові упаковані
    кадри плеєра. pause - Event: поки встановлена, нові шматки не беруться (ті, що вже в пулі, дорахуються й процесори вільні).
    fps - ліміт кадрів на секунду (див. fps_ratio): кадри беруться за часом, тривалість лишається; start, end, max_frames - у кадрах відео"""
    info = video_info(path)
    n = info["frames"] if not max_frames else min(info["frames"], max_frames)
    if end is not None:
        n = min(n, int(end))
    a = max(0, min(int(start), n))
    ratio = fps_ratio(info["fps"], fps)
    last = max(0, n - 1)
    if ratio != 1.0:                                 # далі n й a - у вихідних кадрах
        n, a = max(1 if n else 0, int(round(n / ratio))), int(round(a / ratio))
        a = min(a, n)
    w, h = size
    workers = 1 if params.method == 6 else workers_for(w, h, workers)      # IRNM1: модель одна, на відеокарті, у цьому процесі (пул дав би по копії в кожному воркері)
    if chunk is None:
        if params.method == 6:
            chunk = 48                                   # стан згладжування між шматками переходить сам (_FILM_CONT), тож довжина - лише про паузу й скасування
        elif params.method == 5:                           # «Ультра»: довші шматки - менше витрат на розігрів пам'яті
            chunk = 96 if workers > 1 else 12            # один процес: пауза й скасування перевіряються між шматками - вони короткі
        else:
            chunk = 32 if workers > 1 else 6
    plan = _make_plan(path, params, size, n, chunk, cache, a, workers if workers > 1 else None, 12 if params.method == 5 else 4, ratio, last)
    done = 0
    t_run = time.perf_counter()
    if progress:
        progress(0.0)
    todo = sum(1 for p in plan if p[0] == "j")
    total = max(1, n - a)
    shown = [0.0]

    lg = [time.perf_counter(), 0.0]                  # коли востаннє писали в журнал, кадри на той момент

    def trace(extra: int = 0, final: bool = False):
        """раз на ~15 с у журнал: швидкість і де стоять потоки господаря (чи не чекають усі одного місця) - щоб «провали» навантаження було видно"""
        now = time.perf_counter()
        if ex is None or (not final and now - lg[0] < float(os.environ.get("INKREEL_TRACE_EVERY", "15"))):
            return
        try:
            cur = done + extra
            fps = (cur - lg[1]) / max(1e-6, now - lg[0])
            pr = ex.take_profile() if hasattr(ex, "take_profile") else {}
            thr = max(1, pr.get("thr", 0))
            top = ", ".join(f"{k} {100 * v // thr}%" for k, v in sorted(pr.get("labels", {}).items(), key=lambda kv: -kv[1])[:6])
            conv = ", ".join(f"{k} {v}" for k, v in sorted(pr.get("conv_labels", {}).items(), key=lambda kv: -kv[1])[:3])
            from .log import get_logger
            get_logger().info("export: %d/%d frames, %.1f fps | idle %d%%, in-one-spot %d%% (%s) | top: %s | %s", cur, total, fps,
                              100 * pr.get("idle", 0) // thr, 100 * pr.get("convoy", 0) // max(1, pr.get("n", 0)), conv or "-", top or "-",
                              getattr(ex, "stats", None) or "-")
            lg[0], lg[1] = now, float(cur)
        except Exception:  # noqa: BLE001
            lg[0] = now

    def on_wait(extra: int):
        """між шматками: смуга рухається за кадрами, що вже порахувалися всередині завдань (ніколи не назад і не до 100% раніше часу)"""
        trace(extra)
        if progress is None:
            return
        f = min(0.999, (done + extra) / total)
        if f > shown[0]:
            shown[0] = f
            progress(f)

    if workers <= 1 or todo <= 1:
        results = (p[1] if p[0] == "c" else _job(p[1]) for p in plan)
        ex = None
    else:
        ex = make_pool(min(workers, todo), (w, h))
        results = _ordered(ex, plan, window=min(len(plan), workers * 2), ramp=not getattr(ex, "is_host", False),
                           on_wait=on_wait, cancel=cancel, pause=pause)
    try:
        for packed in results:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            done += len(packed)
            trace()
            yield packed
            if progress:
                f = min(1.0, done / total)
                if f > shown[0] or f >= 1.0:
                    shown[0] = f
                    progress(f)
            wait_pause(pause, cancel)
    finally:
        if ex:
            try:
                dt = time.perf_counter() - t_run
                if done and dt > 3:
                    trace(final=True)
                    from .log import get_logger
                    st = getattr(ex, "stats", None) or {}
                    get_logger().info("export: %d frames in %.1fs (%.1f fps), %dx%d, %d workers, host %s", done, dt, done / dt, w, h,
                                      workers, st or "-")
                    acc = st.get("acc")
                    if acc:                                    # середня кількість потоків господаря, зайнятих кожною стадією (з 'workers'), - де саме йде час
                        parts = ", ".join(f"{k} {acc.get(k, 0) / dt:.1f}" for k in ("warm", "dec", "proc", "pack"))
                        get_logger().info("export: busy threads of %d: %s | gpu calls %.1f (%s calls)", workers, parts,
                                          st.get("gpu_busy", 0) / dt, st.get("gpu_calls", 0))
            except Exception:  # noqa: BLE001
                pass
            kill_pool(ex)


# ---------------------------------------------------------------- експорт
def _size_ramp(pixels: int) -> float:
    """0 для малих кадрів (≤ 0.5 Мпікс, ~480p) -> 1 для великих (≥ 2 Мпікс, 1080p і далі), плавно за логарифмом площі"""
    if not pixels:
        return 0.0
    return min(1.0, max(0.0, math.log2(max(1, pixels) / 0.5e6) / 2.0))


def _x264_args(crf: int, pixels: int = 0) -> list[str]:
    """H.264 для 1-бітного (дізерованого) зображення. Дізер - це шум на всю площу кадру: він кодується в десятки разів дорожче за звичайне
    відео, а амплітуда в нього максимальна (0 і 255), тому чорне-біле читається вірно й при грубому квантуванні. Раніше «Висока» йшла з
    crf 14 і aq-mode=0 - практично без втрат, але ≈ 80-100 Мбіт/с на 2560x1080 (більше за «чисті» біти: 1 біт/піксель ≈ 66 Мбіт/с) - 3 хв 2К
    виходили майже 2 ГБ. Заміри на кліпі 2560x1080 (81 кадр; помилка бітів після порогу / частка «сірих» пікселів / Мбіт/с):
    crf 14 aq0: 0 / 0.06% / 101; crf 24 aq1: 0 / 0.2% / 52; crf 27: 0.002% / 0.5% / 45; crf 30: 0.04% / 2% / 37; crf 33: 0.24% / 4.9% / 28;
    crf 36: 0.9% / 9% / 20; crf 40: 3% / 16% / 11. Змінний aq-mode (за замовчуванням) на цих бітрейтах чистіший за aq0 при тому ж розмірі.
    Але чим менший кадр, тим помітніший кожен піксель, а файл і так малий: тому до ~480p лишаються колишні значення (контракт «Висока - майже
    без сірого» на 256x144 тримається), а від 1080p - нові; між ними плавно (`_size_ramp`).
    Швидкість. На шумі нічого шукати: рух, B-кадри, дрібні розбиття, mbtree й пресети повільніші за veryfast не дають ні байта виграшу -
    лише з'їдають процесор (кодування було 27-34% усього часу експорту 2К). Заміри (ядро, мс на кадр 2560x1080; вже при тому ж розмірі файлу):
    `fast` 198, `faster` 156, `veryfast` + trellis=1 106, + bframes=0 86, + partitions=none 73 - при цьому похибка бітів і «сіре» навіть
    трохи кращі (0.007% проти 0.015%). Те саме на кліпах 1920x740/804/1080 (Слайм, CYN, Макі): 69->28, 41->20, 189->84 мс. Бітрейт на 3-5%
    більший при тому ж crf, тож на великих кадрах crf +0.5 (вирівнює розмір і якість). Без B-кадрів нема й lookahead-залежності від рівня
    «aq»; trellis=1 лишаємо (без нього при crf 29 «сіре» росте утричі)."""
    lo, hi = (14, 29) if crf <= 16 else (20, 33) if crf <= 22 else (28, 38)
    f = _size_ramp(pixels)
    if crf <= 16 and f < 0.25:             # малі кадри, «Висока»: колишня майже беззбіткова схема (кодування там дешеве); без aq плоскі ділянки не «розмиваються»
        return ["-preset", "fast", "-crf", str(int(round(lo + f * (hi - lo)))), "-x264-params", "aq-mode=0"]
    c = lo + f * (hi - lo) + 0.5
    return ["-preset", "veryfast", "-crf", f"{round(c * 2) / 2:g}", "-x264-params", "trellis=1:bframes=0:partitions=none"]


def _hw_args(kind: str, crf: int) -> list[str]:
    """апаратний H.264: NVIDIA NVENC, AMD AMF, Intel Quick Sync (усі - через ffmpeg, без CUDA та залежностей від вендора).
    Шкала якості тут своя й м'якша, ніж у libx264: дізер на низьких cq/qp кодується в сотні Мбіт/с без жодної користі для 1-бітного зображення"""
    if kind == "amf":
        q = str(crf + 8)
        return ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", q, "-qp_p", q, "-qp_b", q]
    if kind == "qsv":
        return ["-c:v", "h264_qsv", "-preset", "slower", "-global_quality", str(crf + 10)]
    # NVENC: -b:v 0 обов'язково - інакше ffmpeg ставить стелю 2 Мбіт/с, і шумний дізер кодується з «мильними» блоками
    return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(crf + 9), "-b:v", "0"]


def _ffmpeg_args(codec: str, crf: int, hw=False, pixels: int = 0) -> list[str]:
    if codec in ("mp4", "mkv", "mov"):
        v = _hw_args("nvenc" if hw is True else hw, crf) if hw else ["-c:v", "libx264", *_x264_args(crf, pixels)]
        v += ["-pix_fmt", "yuv420p"]
        if codec == "mp4":
            v += ["-movflags", "+faststart"]
        return v
    if codec == "webm":
        return ["-c:v", "libvpx-vp9", "-crf", str(crf + 12), "-b:v", "0", "-row-mt", "1", "-cpu-used", "4",
                "-pix_fmt", "yuv420p"]
    if codec == "avi":
        return ["-c:v", "mpeg4", "-q:v", str({16: 2, 22: 5, 28: 9}.get(crf, 5)), "-pix_fmt", "yuv420p"]
    if codec == "gif":
        return ["-vf", "split[a][b];[a]palettegen=max_colors=2:reserve_transparent=0[pal];[b][pal]paletteuse=dither=none", "-loop", "0"]
    raise ValueError(codec)


def _audio_args(src, codec, audio, t0=0.0, dur=None) -> list[str]:
    """друге джерело - оригінал: беремо з нього звук (якщо він є)"""
    if not audio or codec in ("gif", "png", "json"):
        return []
    ac = {"webm": ["-c:a", "libopus", "-b:a", "128k"], "avi": ["-c:a", "libmp3lame", "-b:a", "160k"]}.get(
        codec, ["-c:a", "aac", "-b:a", "160k"])
    cut = (["-ss", f"{t0:.3f}"] if t0 > 0 else []) + (["-t", f"{dur:.3f}"] if dur else [])
    return [*cut, "-i", src, "-map", "0:v:0", "-map", "1:a:0?", *ac, "-shortest"]


def export_video(src, out, params: Params, fmt="MP4 (H.264)", size_key="720p", quality="Висока", nvenc=False,   # nvenc: False | True | "nvenc"/"amf"/"qsv" (апаратний H.264)
                
                 workers=None, max_frames=None, progress=None, cancel=None, audio=True, cache=None, start=0, end=None, palette=None,
                 pause=None, preview=None, fps=None) -> dict:
    """Бінаризує src і зберігає у вибраному форматі. progress(frac, label); cancel - threading.Event.
    fps - ліміт кадрів на секунду (None - як в оригіналі; вищий за початкову частоту нічого не змінює): кадри вибираються за часом,
    тож тривалість і звук не змінюються, а файл лишається легшим і швидшим у розрахунку.
    out для «PNG-послідовність» - тека. Повертає dict(frames, bytes, size, fps).
    pause - Event: експорт стоїть, поки він встановлений. preview(кількість_готових, маска01) - свіжий кадр результату
    (не частіше ~3 разів на секунду): інтерфейс показує, що саме зараз рахується"""
    ext, codec = FORMATS[fmt]
    from . import bake
    bk = bake.open_baked(src) if bake.is_baked(src) else None        # «спечений» файл InkReel: кадри вже готові, розмір - його власний
    a_tmp = None
    try:
        info = bk.info() if bk else video_info(src)
        size = (bk.w, bk.h) if bk else out_size(info["width"], info["height"], size_key, glyph_cell(params))
        a_src = src
        if bk:                                                       # звук із файлу .ikr: вийняти у тимчасову теку для ffmpeg
            a_tmp = tempfile.mkdtemp(prefix="ikr_")
            a_src = bk.audio_to_file(a_tmp)
            audio = bool(audio and a_src)
        return _export_video(src, out, params, fmt, ext, codec, info, size, quality, nvenc, workers, max_frames, progress, cancel, audio, cache,
                             start, end, palette, pause, preview, fps, bk, a_src)
    finally:
        if bk:
            bk.close()
        _CAPS.release_path(src)                                      # декодери джерела, відкриті потоками експорту: інакше файл лишається зайнятим (Windows)
        if a_tmp:
            shutil.rmtree(a_tmp, ignore_errors=True)


def _ikr_meta(src, bk, info, params, start, end, fps, palette, version) -> dict:
    """що вписати в заголовок .ikr: налаштування й вихідне відео - щоб із файлу можна було повернутись до звичайного режиму"""
    if bk is not None:                                    # з іншого .ikr (діапазон): шлях назад до оригіналу лишається тим самим
        srcm = dict(bk.meta.get("src") or {})
        pd = bk.meta.get("params") or params.to_dict()
        base = bk.meta.get("range", [0, None])[0] or 0
    else:
        st = None
        try:
            st = os.stat(src)
        except OSError:
            pass
        srcm = dict(path=os.path.abspath(src), name=os.path.basename(src), size=st.st_size if st else 0, mtime=st.st_mtime if st else 0,
                    fps=info["fps"], frames=info["frames"], width=info["width"], height=info["height"])
        pd = params.to_dict()
        base = 0
    n = min(end, info["frames"]) if end else info["frames"]
    return dict(params=pd, src=srcm, range=[base + int(start), base + int(n)], fps_limit=float(fps) if fps else 0.0, palette=palette or "",
                app=version, has_audio=False)


def _ikr_audio(src, bk, a_src, info, start, end, folder):
    """звук діапазону для .ikr: з відео - ffmpeg (AAC), з іншого .ikr - його потік (обрізаний, якщо взято частину); None - звуку нема"""
    ff = find_ffmpeg()
    if not ff or not a_src:
        return None
    t0 = start / (info["fps"] or 24.0)
    n = min(end, info["frames"]) if end else info["frames"]
    dur = (n - start) / (info["fps"] or 24.0) if (start or end) else None
    cut = (["-ss", f"{t0:.3f}"] if t0 > 0 else []) + (["-t", f"{dur:.3f}"] if dur else [])
    if bk is not None and not cut:
        return a_src
    outp = os.path.join(folder, "a.m4a")
    try:
        r = subprocess.run([ff, "-y", "-loglevel", "error", *cut, "-i", a_src, "-vn", "-map", "0:a:0", "-c:a", "aac", "-b:a", "128k", outp],
                           capture_output=True, timeout=600, **hidden_kw())
    except Exception:  # noqa: BLE001
        return None
    return outp if r.returncode == 0 and os.path.isfile(outp) and os.path.getsize(outp) > 0 else None


def _export_video(src, out, params, fmt, ext, codec, info, size, quality, nvenc, workers, max_frames, progress, cancel, audio, cache, start, end,
                  palette, pause, preview, fps, bk, a_src) -> dict:
    frac = fps_fraction(out_fps(info["fps"], fps))
    crf = QUALITY.get(quality, 22)
    ff = find_ffmpeg()
    from . import palette as pal
    neg = pal.is_negative(palette) and codec not in ("json", "ikr")     # біле на чорному лишається в сірому каналі (швидко), лише полярність інша
    color = not (pal.is_classic(palette) or pal.is_negative(palette)) and codec not in ("json", "ikr")
    label = "Обробка й кодування"
    start_label = "Запуск обробки"                     # поки жоден кадр не готовий: старт обробника, читання ядер з кешу, перший шматок
    if progress:
        progress(0.0, start_label)
    cb = (lambda f: progress(f, label if f > 0 else start_label)) if progress else None
    count, proc, vw, err, jw, ikw = 0, None, None, None, None, None
    if codec == "png":
        os.makedirs(out, exist_ok=True)
    elif codec == "json":
        from .pack import JsonWriter
        jw = JsonWriter(out, size[0], size[1], frac.numerator, frac.denominator)
    elif codec == "ikr":
        from . import bake
        from . import __version__
        ikw = bake.BakeWriter(out, size[0], size[1], frac.numerator, frac.denominator,
                              _ikr_meta(src, bk, info, params, start, end, fps, palette, __version__))
    elif ff:
        err = tempfile.TemporaryFile()
        cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24" if color else "gray", "-s", f"{size[0]}x{size[1]}",
               "-r", f"{frac.numerator}/{frac.denominator}", "-i", "-",
               *_audio_args(a_src, codec, audio, start / (info["fps"] or 24.0), ((min(end, info["frames"]) if end else info["frames"]) - start) / (info["fps"] or 24.0) if (start or end) else None), *_ffmpeg_args(codec, crf, nvenc, size[0] * size[1]), out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err, **hidden_kw())
    else:
        if codec not in ("mp4", "mov", "avi"):
            raise RuntimeError(tr("для цього формату потрібен ffmpeg: pip install imageio-ffmpeg"))
        fourcc = "MJPG" if codec == "avi" else "mp4v"
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*fourcc), float(frac), size, color)
        if not vw.isOpened():
            raise RuntimeError(tr("не вдалося відкрити VideoWriter"))
    t_wait = [0.0]                                     # скільки головний потік чекав готових кадрів (обробка - вузьке місце) і скільки віддавав їх кодувальнику
    t_feed = 0.0
    t_exp = time.perf_counter()

    def _timed(gen):
        try:
            while True:
                t0 = time.perf_counter()
                try:
                    item = next(gen)
                except StopIteration:
                    return
                t_wait[0] += time.perf_counter() - t0
                yield item
        finally:
            gen.close()
    try:
        t_prev = 0.0
        W = size[0]
        if bk is not None:                             # кадри вже порахованi: читаємо з файлу, нічого не рахуємо
            from . import bake
            n_end = end if end is not None else None
            if max_frames:
                n_end = min(n_end, max_frames) if n_end is not None else max_frames
            gen = bake.packed_chunks(bk, bake.picked_frames(bk.frames, bk.fps, fps, start, n_end), cancel=cancel, pause=pause,
                                     progress=(lambda f: cb(f)) if cb else None)
        else:
            gen = packed_video(src, params, size, workers, max_frames=max_frames, progress=cb, cancel=cancel, cache=cache, start=start,
                               end=end, pause=pause, fps=fps)
        for packed in _timed(gen):
            t_chunk = time.perf_counter()
            if preview is not None and len(packed) and time.perf_counter() - t_prev >= 0.35:
                t_prev = time.perf_counter()
                try:
                    preview(count + len(packed), np.unpackbits(packed[-1], axis=-1, count=W))
                except Exception:  # noqa: BLE001
                    pass
            for pf in packed:
                if ikw is not None:                                            # «спечений» файл: біти кадра кладуться як є
                    ikw.add_packed(pf)
                    count += 1
                    continue
                # розпаковка по ОДНОМУ кадру: раніше розпаковувався увесь шматок (96 кадрів 1440p - майже півгігабайта) і ще раз множився
                # на 255 - величезні тимчасові масиви й пауза в подачі кодувальнику на кожному шматку
                fr = np.unpackbits(pf, axis=-1, count=W)                     # (h, w) {0,1}
                if jw:
                    jw.add(fr)
                elif color:
                    if codec == "png":
                        ok, enc = cv2.imencode(".png", pal.paint(fr, palette, bgr=True), [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        enc.tofile(os.path.join(out, f"{count + 1:06d}.png"))
                    elif proc:
                        proc.stdin.write(pal.paint(fr, palette).data)
                    else:
                        vw.write(pal.paint(fr, palette, bgr=True))
                else:
                    if neg:
                        fr ^= 1                                              # біле на чорному: полярність міняється в самому масиві
                    fr *= 255
                    if codec == "png":
                        ok, enc = cv2.imencode(".png", fr, [cv2.IMWRITE_PNG_COMPRESSION, 1])     # imwrite ламається на юнікоді
                        enc.tofile(os.path.join(out, f"{count + 1:06d}.png"))
                    elif proc:
                        proc.stdin.write(fr.data)
                    else:
                        vw.write(fr)
                count += 1
            t_feed += time.perf_counter() - t_chunk
    except BaseException as e:
        broken = False
        if proc:
            # ffmpeg помер посеред запису (закрили його вікно, антивірус, кінець місця): на Windows це або BrokenPipe, або OSError EINVAL
            broken = isinstance(e, OSError) and (isinstance(e, BrokenPipeError) or e.errno in (22, 32) or proc.poll() is not None)
            if broken:
                proc.stdin = None
                try:
                    proc.wait(timeout=10)
                except Exception:  # noqa: BLE001
                    pass
            proc.kill()
            try:
                proc.wait(timeout=10)                       # Windows: файл звільняється, лише коли ffmpeg справді завершився
            except Exception:  # noqa: BLE001
                pass
        if vw:
            vw.release()
        if ikw is not None:
            ikw.abort()
        if codec != "png" and os.path.exists(out):
            for _ in range(10):                             # недописаний файл не лишаємо; антивірус/ffmpeg можуть тримати його ще мить
                try:
                    os.remove(out)
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    time.sleep(0.2)
        if broken:
            raise RuntimeError(_ff_error(err) or tr("ffmpeg завершився з помилкою") + (tr(" (спробуй без апаратного кодування)") if nvenc else "")) from None
        raise
    else:
        if jw:
            jw.close()
        if ikw is not None:
            tmpd = tempfile.mkdtemp(prefix="ikr_a_")
            try:
                if audio:
                    ap = _ikr_audio(src, bk, a_src, info, start, end, tmpd)
                    if ap:
                        ikw.set_audio_file(ap, ".m4a")
                ikw.close()
            except BaseException:
                ikw.abort()
                if os.path.exists(out):
                    os.remove(out)
                raise
            finally:
                shutil.rmtree(tmpd, ignore_errors=True)
        if proc:
            proc.stdin.close()
            if proc.wait() != 0:
                msg = _ff_error(err)
                raise RuntimeError(msg or tr("ffmpeg не зміг закодувати відео") + (tr(" (спробуй без апаратного кодування)") if nvenc else ""))
        if vw:
            vw.release()
    finally:
        if err:
            err.close()
    try:
        dt = time.perf_counter() - t_exp
        if count and dt > 3:                           # кодувальник - вузьке місце, якщо головний потік довго віддає кадри; обробка - якщо довго чекає їх
            from .log import get_logger
            get_logger().info("export: main thread over %.0fs: waiting for frames %.0f%%, feeding the encoder %.0f%%; %s", dt, 100 * t_wait[0] / dt,
                              100 * t_feed / dt, "encoder-bound" if t_feed > 0.5 * dt else "processing-bound" if t_wait[0] > 0.5 * dt else "balanced")
    except Exception:  # noqa: BLE001
        pass
    if progress:
        progress(1.0, label)
    size_b = (sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) if codec == "png"
              else os.path.getsize(out))
    return dict(frames=count, bytes=size_b, size=size, fps=float(frac), path=out)


def _ff_error(f) -> str:
    try:
        f.seek(0)
        return f.read().decode("utf-8", "replace").strip()[-400:]
    except Exception:  # noqa: BLE001
        return ""


_SHM = {}
_SHM_LOCK = threading.Lock()


def _frame_job_shm(args):
    """кадр лежить у спільній пам'яті (слот кільця): без пересилання через pickle/pipe, лише ім'я й номер слота"""
    name, slot, w, h, pd = args
    with _SHM_LOCK:
        shm = _SHM.get(name)
        if shm is None:
            from multiprocessing import shared_memory as sm
            try:
                shm = sm.SharedMemory(name=name, track=False)
            except TypeError:
                shm = sm.SharedMemory(name=name)
            if len(_SHM) > 3:                  # старі розміри вже не потрібні (якщо їх ще читає інший потік - close() відмовить, і це нормально)
                for k in list(_SHM)[:-2]:
                    try:
                        _SHM.pop(k).close()
                    except Exception:  # noqa: BLE001
                        pass
            _SHM[name] = shm
    f = np.ndarray((h, w, 3), np.uint8, buffer=shm.buf, offset=slot * w * h * 3)
    return np.packbits(binarize(f, w, h, Params.from_dict(pd)), axis=-1)


def _frame_job(args):
    """один кадр для плеєра: (bgr уже у розмірі результату, w, h, params-dict) -> упаковане 1-бітне зображення"""
    f, w, h, pd = args
    return np.packbits(binarize(f, w, h, Params.from_dict(pd)), axis=-1)


_BENCH_IMG = {}


def _bench_job(args):
    """лише для benvid.bench: рахує count кадрів синтетичного відео w×h в цьому процесі й повертає (мс на кадр, потоків numba, шар потоків)"""
    import time as _t
    w, h, method, count = args
    from . import bench
    from .core import InkState
    key = ("bj", w, h)
    img = _BENCH_IMG.get(key)
    if img is None:
        img = _BENCH_IMG[key] = bench._sample(w, h)
    p = Params(method=method)
    st = InkState() if method == 5 else None
    aux = {"state": st} if st is not None else None
    if method == 6:
        from .student import FilmState
        aux = {"film": FilmState()}
    binarize(img, w, h, p, aux)
    t0 = _t.perf_counter()
    for k in range(count):
        binarize(np.roll(img, k, 1), w, h, p, aux)
    ms = (_t.perf_counter() - t0) / count * 1000
    try:
        import numba
        return ms, numba.get_num_threads(), str(numba.threading_layer())
    except Exception:  # noqa: BLE001
        return ms, 0, "?"


_MICRO_NB = None


def _bench_micro(args):
    """лише для benvid.bench: -> швидкість (операцій/с) одного з трьох мікротестів за ~secs секунд у цьому процесі:
    cpu - числодробарка в кеші; mem - потокове читання/запис великого масиву (ГБ/с); alloc - свіжі великі масиви кожну ітерацію (ГБ/с)"""
    import time as _t
    kind, secs = args
    if kind == "cpu":
        global _MICRO_NB
        if _MICRO_NB is None:
            import numba

            @numba.njit(cache=False, fastmath=True)
            def f(n):
                s = 0.0
                for i in range(n):
                    s += np.sqrt(i * 1.0001 + 1.0) * 0.5
                return s
            f(10)
            _MICRO_NB = f
        f = _MICRO_NB
        n = 2_000_000
        t0, done = _t.perf_counter(), 0
        while _t.perf_counter() - t0 < secs:
            f(n)
            done += n
        return done / (_t.perf_counter() - t0) / 1e6         # млн ітерацій/с
    if kind == "mem":
        a = np.ones(24 << 20, np.float32)                   # 96 МБ - більше за кеш
        b = np.empty_like(a)
        t0, done = _t.perf_counter(), 0
        while _t.perf_counter() - t0 < secs:
            np.multiply(a, 1.0001, out=b)
            done += a.nbytes * 2
        return done / (_t.perf_counter() - t0) / 1e9
    t0, done = _t.perf_counter(), 0                         # alloc
    while _t.perf_counter() - t0 < secs:
        x = np.ones((720, 1280, 3), np.float32)
        x[0, 0, 0] = 2
        done += x.nbytes
    return done / (_t.perf_counter() - t0) / 1e9


_HW = None


def detect_hw_encoder() -> str:
    """який апаратний H.264-кодер реально працює на цьому ПК: 'nvenc' | 'amf' | 'qsv' | '' (пробне кодування одного кадру
    саме з тими параметрами, що підуть в експорт); результат запам'ятовується"""
    global _HW
    if _HW is None:
        _HW = ""
        ff = find_ffmpeg()
        for kind in (("nvenc", "amf", "qsv") if ff else ()):
            try:
                r = subprocess.run([ff, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1", "-frames:v", "1",
                                    *_hw_args(kind, 16), "-pix_fmt", "yuv420p", "-f", "null", "-"],
                                   capture_output=True, timeout=20, creationflags=_NOWIN)
                if r.returncode == 0:
                    _HW = kind
                    break
            except Exception:  # noqa: BLE001
                continue
    return _HW


def detect_nvenc() -> bool:
    """сумісність: чи є робоче апаратне кодування NVIDIA"""
    return detect_hw_encoder() == "nvenc"
