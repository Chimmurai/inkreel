"""Швидка дія з Провідника Windows: правий клік по фото чи GIF -> «Зробити чорно-біле (InkReel)».

Основне вікно не відкривається: з'являється маленьке вікно з прогресом, а результат лягає поруч з оригіналом
(<ім'я>_inkreel.png для фото, <ім'я>_inkreel.gif для GIF; якщо тека лише для читання - у «Зображення\\InkReel»). Коли все готово,
Провідник показує результат виділеним. Беруться палітра, «навпаки» й рівень якості, які ви востаннє обрали у вікні; розмір - як у
джерела.

Кілька виділених файлів Провідник запускає окремими процесами: перший стає «чергою» (локальний сокет, benvid/single.py зі своїм
портом), решта передають йому свої файли й завершуються - усе йде по черзі в одному вікні, а не десять програм одразу.
Відео сюди не потрапляє: його відкриває звичайне вікно InkReel.

Запуск: InkReel.exe --quick файл [файл …]   (python -m benvid --quick файл)"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from . import i18n, single
from .i18n import tr

PORT = 47614
MAGIC = "INKQUICK1"
SUFFIX = "_inkreel"
GRACE = 1.2            # скільки секунд після ОТРИМАННЯ останнього файлу чекаємо, чи не передадуть ще (процеси-сусіди стартують не миттєво)
LEAD = 1.8             # і в будь-якому разі не закінчуємо раніше за стільки секунд від старту: сусіди могли ще не встигнути завантажитись


class Stopped(Exception):
    """користувач скасував"""


class VideoGiven(Exception):
    """замість фото чи GIF передали відео: тут його не обробляємо, а відкриваємо у звичайному вікні"""


@dataclass
class Prefs:
    method: int = 4
    palette: str = "Класика"          # ключ палітри (з «~», якщо «навпаки»)
    lang: str = i18n.UK
    glyph: bool = False               # режим «Символи» (знаки шрифту замість чорнила)
    gdens: int = 3                    # їхня щільність: 1 (32×64 px) .. 3 (8×16 px)


# ---------------------------------------------------------------- налаштування, імена файлів
def load_prefs(settings: dict | None = None) -> Prefs:
    """останній вибір користувача з вікна: рівень якості, палітра (+ «навпаки»), мова"""
    from . import palette
    if settings is None:
        from .engine import load_settings
        settings = load_settings()
    ui = settings.get("ui") if isinstance(settings.get("ui"), dict) else {}
    try:
        method = min(6, max(1, int(ui.get("method", 4))))
    except (TypeError, ValueError):
        method = 4
    if ui.get("glyph") and method > 5:                              # шостий рівень (нейро) - лише для чорнила
        method = 5
    name, inv = palette.migrate(ui.get("palette"), ui.get("pal_inv"))
    lang = ui.get("lang") if ui.get("lang") in i18n.LANGS else i18n.detect()
    try:
        gdens = min(3, max(1, int(ui.get("gdens3", 3))))             # ключ «gdens3»: колишня шкала 1..5 (ключ «gdens») не збігається з нинішньою
    except (TypeError, ValueError):
        gdens = 3
    return Prefs(method, palette.key(name, inv), lang, bool(ui.get("glyph", False)), gdens)


def out_dir_for(src: str) -> str:
    """тека для результату: поруч з оригіналом, а якщо там не можна писати (диск лише для читання, системна тека) -
    «Зображення\\InkReel» у теці користувача"""
    d = os.path.dirname(os.path.abspath(src))
    if os.access(d, os.W_OK):
        return d
    alt = os.path.join(os.path.expanduser("~"), "Pictures", "InkReel")
    os.makedirs(alt, exist_ok=True)
    return alt


def unique_path(src: str, ext: str, outdir: str | None = None) -> str:
    """<ім'я>_inkreel.png; якщо вже є - <ім'я>_inkreel (2).png, (3) … ; чужі файли ніколи не перезаписуються"""
    stem = os.path.splitext(os.path.basename(src))[0]
    d = outdir or out_dir_for(src)
    p = os.path.join(d, stem + SUFFIX + ext)
    n = 2
    while os.path.exists(p):
        p = os.path.join(d, f"{stem}{SUFFIX} ({n}){ext}")
        n += 1
    return p


# ---------------------------------------------------------------- саме перетворення
def convert(src: str, prefs: Prefs, progress=None, cancel=None) -> str:
    """фото -> <ім'я>_inkreel.png, GIF -> <ім'я>_inkreel.gif; повертає шлях результату. progress(frac, label) - необов'язковий"""
    from . import media
    from .core import Params
    from .video import Cancelled
    kind = media.kind_of(src)
    if kind == "video":
        raise VideoGiven(src)
    params = Params()
    params.method = prefs.method
    params.glyph = bool(prefs.glyph)
    params.gdens = int(prefs.gdens)
    out = unique_path(src, ".gif" if kind == "gif" else ".png")
    try:
        if kind == "gif":
            media.export_gif_source(src, out, params, "GIF", "Як у джерела", progress=progress, cancel=cancel,
                                    palette=prefs.palette)
        else:
            media.export_still(src, out, params, "Як у джерела", progress=progress, palette=prefs.palette)
        if cancel is not None and cancel.is_set():          # скасували, поки йшло (фото рахується одним махом) - результат не потрібен
            raise Cancelled()
    except Cancelled:
        _drop(out)
        raise Stopped()
    except BaseException:
        _drop(out)                    # напівготовий файл не лишаємо
        raise
    return out


def _drop(path: str):
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------- черга (без tkinter - перевіряється тестами)
class Batch:
    """файли швидкої дії по черзі. Події для вікна лежать у self.events: ('start', ім'я, №, усього), ('progress', частка),
    ('ok', джерело, результат), ('err', джерело, текст), ('end',)"""

    def __init__(self, convert_fn, grace: float = GRACE, lead: float = 0.0, clock=time.monotonic):
        self.convert, self.grace, self.lead, self.clock = convert_fn, grace, lead, clock
        self.t0 = self.last_add = clock()
        self.q: queue.Queue[str] = queue.Queue()
        self.events: queue.Queue[tuple] = queue.Queue()
        self.cancel = threading.Event()
        self.total = 0
        self.index = 0
        self.done: list[tuple[str, str]] = []
        self.errors: list[tuple[str, str]] = []
        self.videos: list[str] = []
        self._lock = threading.Lock()

    def add(self, path: str):
        if not path:
            return
        with self._lock:
            self.total += 1
            self.last_add = self.clock()
        self.q.put(path)

    def idle_wait(self) -> float:
        """скільки ще чекати нових файлів, коли черга спорожніла: grace після останнього отриманого, але не менше ніж lead від старту.
        Час самого перетворення зараховується: після довгого перетворення чекати вже не треба - вікно не висить без діла"""
        with self._lock:
            end = max(self.last_add + self.grace, self.t0 + self.lead)
        return max(0.0, end - self.clock())

    def run(self):
        """обробляє чергу, доки вона не вичерпається й не мине час очікування (idle_wait) без нових файлів (або не скасують)"""
        while not self.cancel.is_set():
            try:
                src = self.q.get(timeout=self.idle_wait())
            except queue.Empty:
                return
            self.index += 1
            self.events.put(("start", os.path.basename(src), self.index, self.total))
            try:
                out = self.convert(src, lambda f, label=None: self.events.put(("progress", f)), self.cancel)
            except VideoGiven:
                self.videos.append(src)
            except Stopped:
                self.cancel.set()
            except Exception as e:  # noqa: BLE001
                self.errors.append((src, str(e)))
                try:
                    from . import log
                    log.get().exception("швидка дія: %s", src)
                except Exception:  # noqa: BLE001
                    pass
                self.events.put(("err", src, str(e)))
            else:
                self.done.append((src, out))
                self.events.put(("ok", src, out))


def work(batch: Batch, listener=None):
    """потік черги: обробляє, закриває «вхід» для нових файлів і перевіряє, чи не встигли щось передати; потім повідомляє про кінець"""
    while True:
        batch.run()
        if listener is not None:
            listener.close()                      # нові процеси-сусіди тепер самі стануть чергою
            time.sleep(0.25)                      # хто вже встиг підключитись - його файл уже в черзі
        if batch.cancel.is_set() or batch.q.empty():
            break
        listener = None                           # докручуємо те, що встигло прийти в останню мить
    batch.events.put(("end",))


# ---------------------------------------------------------------- Провідник
def reveal(path: str):
    """показати файл у Провіднику виділеним"""
    try:
        if os.name == "nt":
            subprocess.Popen(f'explorer /select,"{path}"')
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])
    except Exception:  # noqa: BLE001
        pass


def launch_main(path: str):
    """відкрити файл у звичайному вікні InkReel (окремим процесом)"""
    cmd = [sys.executable, path] if getattr(sys, "frozen", False) else [sys.executable, "-m", "benvid", path]
    try:
        subprocess.Popen(cmd, close_fds=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- вікно
class QuickWindow:
    """маленьке вікно прогресу у стилі програми (кольори - з вибраної палітри)"""
    W, H, CELLS = 460, 168, 36

    def __init__(self, batch: Batch, prefs: Prefs):
        import tkinter as tk
        from . import palette
        self.tk, self.batch, self.prefs = tk, batch, prefs
        t = palette.theme(prefs.palette)
        self.BG, self.FG, self.DIM = t["BG"], t["FG"], t["DIM"]
        try:
            from . import procs
            procs.crisp_text()                           # чіткий текст на моніторах із масштабом (до створення вікна)
        except Exception:  # noqa: BLE001
            pass
        r = self.root = tk.Tk()
        r.title("InkReel")
        r.configure(bg=self.BG)
        r.resizable(False, False)
        self._icon()
        try:                                                  # посередині того монітора, де зараз курсор (клацнули в Провіднику), а не першого
            from . import popup
            px, py = r.winfo_pointerxy()
            l, t, rr, bb = popup.work_area(r, px, py)
        except Exception:  # noqa: BLE001
            l, t, rr, bb = 0, 0, r.winfo_screenwidth(), r.winfo_screenheight()
        r.geometry(f"{self.W}x{self.H}+{l + (rr - l - self.W) // 2}+{t + (bb - t - self.H) // 3}")
        def untop():
            try:
                r.attributes("-topmost", False)           # лише на мить, щоб вікно з'явилось над Провідником; далі - як усі
            except tk.TclError:
                pass
        try:
            r.attributes("-topmost", True)
        except tk.TclError:
            pass
        r.after(1200, untop)
        font = ("Consolas", 10, "bold")
        self.head = tk.Label(r, text="InkReel", bg=self.BG, fg=self.FG, font=("Consolas", 14, "bold"), anchor="w")
        self.head.pack(fill="x", padx=16, pady=(12, 0))
        self.lbl = tk.Label(r, text=tr("Підготовка…"), bg=self.BG, fg=self.FG, font=font, anchor="w", justify="left",
                            wraplength=self.W - 32)
        self.lbl.pack(fill="x", padx=16, pady=(8, 0))
        self.cnt = tk.Label(r, text="", bg=self.BG, fg=self.DIM, font=("Consolas", 9), anchor="w")
        self.cnt.pack(fill="x", padx=16)
        self.bar = tk.Canvas(r, width=self.W - 32, height=16, bg=self.BG, highlightthickness=0, bd=0)
        self.bar.pack(padx=16, pady=(8, 0))
        self.btn = tk.Button(r, text=tr("Скасувати"), command=self._cancel, bg=self.BG, fg=self.FG, activebackground=self.FG,
                             activeforeground=self.BG, relief="flat", bd=0, highlightthickness=1, highlightbackground=self.FG,
                             font=("Consolas", 9, "bold"), padx=12, pady=2, cursor="hand2")
        self.btn.pack(side="right", padx=16, pady=(10, 0))
        self._frac, self._ended, self._errs = 0.0, False, []
        self._draw_bar()
        r.protocol("WM_DELETE_WINDOW", self._cancel)
        r.after(40, self._pump)

    def _icon(self):
        try:
            here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
            ico = os.path.join(here, "inkreel.ico")
            if os.name == "nt" and os.path.isfile(ico):
                self.root.iconbitmap(default=ico)
            self._img = self.tk.PhotoImage(file=os.path.join(here, "app_icon.png"))
            self.root.iconphoto(True, self._img)
        except Exception:  # noqa: BLE001
            pass

    def _draw_bar(self):
        c, n = self.bar, self.CELLS
        c.delete("all")
        w = (self.W - 32) / n
        full = int(self._frac * n + 1e-9)
        for i in range(n):
            x0 = int(i * w)
            c.create_rectangle(x0, 0, int((i + 1) * w) - 2, 15, fill=self.FG if i < full else self.BG, outline=self.FG)

    def _cancel(self):
        self.batch.cancel.set()
        self.root.destroy()

    def _pump(self):
        try:
            while True:
                ev = self.batch.events.get_nowait()
                getattr(self, "_ev_" + ev[0])(*ev[1:])
        except queue.Empty:
            pass
        except Exception:  # noqa: BLE001
            pass
        if not self._ended:
            self.root.after(40, self._pump)

    def _ev_start(self, name, i, n):
        self._frac = 0.0
        self.lbl.config(text=tr("Перетворюю: {name}", name=name))
        self.cnt.config(text=tr("{i} з {n}", i=i, n=n) if n > 1 else "")
        self._draw_bar()

    def _ev_progress(self, f):
        self._frac = min(1.0, max(0.0, f))
        self._draw_bar()

    def _ev_ok(self, src, out):
        self._frac = 1.0
        self._draw_bar()

    def _ev_err(self, src, msg):
        self._errs.append(tr("Не вдалося: {name}\n{e}", name=os.path.basename(src), e=msg))

    def _ev_end(self):
        self._ended = True
        b = self.batch
        if b.done:
            reveal(b.done[-1][1])
        for v in b.videos[:1]:
            launch_main(v)
        if self._errs:
            self.lbl.config(text="\n\n".join(self._errs), fg=self.FG)
            self.cnt.config(text="")
            self.btn.config(text=tr("Закрити"), command=self.root.destroy)
            self.root.geometry(f"{self.W}x{min(520, self.H + 40 * len(self._errs))}")
        else:
            self._frac = 1.0
            self._draw_bar()
            self.lbl.config(text=tr("Готово"))
            self.root.after(450, self.root.destroy)

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------- вхід
def main(argv: list[str]) -> int:
    paths = [os.path.abspath(a) for a in argv if os.path.isfile(a)]
    if not paths:
        return 1
    prefs = load_prefs()
    i18n.set_lang(prefs.lang)
    lst = single.acquire(paths[0], PORT, MAGIC)
    if lst is False:                                  # чергу вже веде інший процес: він отримав файл
        return 0
    batch = Batch(lambda src, progress, cancel: convert(src, prefs, progress, cancel), GRACE, LEAD)
    for p in paths:
        batch.add(p)
    if lst:
        lst.start(batch.add)
    th = threading.Thread(target=work, args=(batch, lst or None), daemon=True, name="quick")
    th.start()
    QuickWindow(batch, prefs).run()
    if batch.cancel.is_set():
        th.join(8)                                    # дати потоку прибрати напівготовий файл, а не обірвати його разом із процесом
    return 0
