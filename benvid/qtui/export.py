"""Експорт у фоновому потоці (відео, GIF, фото) на тих самих функціях, що й у Tk: `video.export_video`, `media.export_gif_source`, `media.export_still`.
Звіти йдуть сигналами Qt (з потоку в потік інтерфейсу вони стають чергою подій самі), без ручної черги `queue` і опитування `after(50)`."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal

from .. import log as _log
from ..core import Params
from ..media import export_gif_source, export_still
from ..progress import Clock, fmt_t
from ..video import Cancelled, default_workers, detect_hw_encoder, export_video, fps_ratio, out_size


class ExportController(QObject):
    progress = Signal(float, object, float, object)       # частка, підпис етапу, минуло с, лишилось с (або None)
    live = Signal(object)                                 # свіжий готовий кадр-маска, щоб було видно, що зараз рахується
    finished = Signal(str, float, bool, dict)             # ключ повідомлення, частка, чи успіх, підстановки

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cancel = threading.Event()
        self.pause_ev = threading.Event()
        self.running = False
        self.clock: Clock | None = None
        self.total = 0
        self.f = 0.0
        self.label = None
        self._thread: threading.Thread | None = None

    def start(self, *, kind: str, path: str, out: str, params: Params, fmt: str, size_key: str, quality: str, pal_name: str, audio: bool,
              cache=None, fps=None, src_fps: float = 24.0, n_frames: int = 0, a0: int = 0, b0=None):
        if self.running:
            return
        self.running = True
        self.cancel.clear()
        self.pause_ev.clear()
        self.f, self.label = 0.0, None
        self.total = max(1, int(round(((b0 or n_frames) - a0) / fps_ratio(src_fps, fps)))) if kind == "video" else 0
        clock = self.clock = Clock()
        params = Params(**params.to_dict())
        workers = default_workers()

        def prog(f, label=None):
            self.progress.emit(float(f), label, clock.elapsed(), clock.eta(f))

        def live(n, mask):
            self.live.emit(mask)

        def work():
            try:
                from .. import procs
                procs.lower_thread_priority()              # потік експорту нижче за інтерфейс: вікно відповідає під будь-яким навантаженням
            except Exception:  # noqa: BLE001
                pass
            try:
                if kind == "video":
                    nv = (detect_hw_encoder() or False) if quality != "Висока" else False      # «Висока» кодується програмним x264: на дізері він чистіший
                    try:
                        r = export_video(path, out, params, fmt, size_key, quality, nv, workers, None, prog, self.cancel, audio, cache, a0, b0, pal_name,
                                         pause=self.pause_ev, preview=live, fps=fps)
                    except Cancelled:
                        raise
                    except RuntimeError:
                        if not nv:
                            raise
                        r = export_video(path, out, params, fmt, size_key, quality, False, workers, None, prog, self.cancel, audio, cache, a0, b0, pal_name,
                                         pause=self.pause_ev, preview=live, fps=fps)
                elif kind == "gif":
                    r = export_gif_source(path, out, params, fmt, size_key, workers, prog, self.cancel, palette=pal_name)
                else:
                    r = export_still(path, out, params, size_key, prog, palette=pal_name)
                self._end("Готово: {frames} · {w}×{h} · {mb:.1f} МБ\n{p}", 1.0, True, frames=r["frames"], w=r["size"][0], h=r["size"][1], mb=r["bytes"] / 1e6, p=out)
            except Cancelled:
                self._end("Скасовано", 0.0, False)
            except Exception as e:  # noqa: BLE001
                _log.get().exception("помилка експорту")
                self._end("Помилка: {e}\n(деталі: {p})", 0.0, False, e=str(e), p=_log.shown_path())

        self._thread = threading.Thread(target=work, daemon=True, name="qt-export")
        self._thread.start()

    def _end(self, key, f, ok, **kw):
        self.running = False
        self.pause_ev.clear()
        self.finished.emit(key, f, ok, kw)

    def stop(self):
        self.cancel.set()

    def toggle_pause(self):
        if self.pause_ev.is_set():
            self.pause_ev.clear()
        else:
            self.pause_ev.set()
        if self.clock is not None:
            try:
                self.clock.hold(self.pause_ev.is_set())
            except Exception:  # noqa: BLE001
                pass

    def join(self, timeout=3.0):
        if self._thread is not None:
            self._thread.join(timeout)

    # текст прогресу тими самими формулюваннями, що в Tk (переклад уже є в lang_en)
    def status(self, f, label, el, eta, tr) -> str:
        lab = tr(label or "Обробка")
        if self.pause_ev.is_set():
            return tr("ПАУЗА  {pct:3.0f}%\nминуло {el}\n[Пробіл] — продовжити", pct=f * 100, el=fmt_t(el))
        if eta is not None and f < 1 and self.total > 1 and el >= 1.0:
            done = min(self.total, int(round(f * self.total)))
            sp = self.clock.speed() if self.clock is not None else None
            fps = sp * self.total if sp else done / max(el, 1e-6)
            return tr("{lab}  {pct:3.0f}%\nминуло {el}  залишилось ~{eta}\nкадр {cur}/{tot} · {fps:.1f} к/с", lab=lab, pct=f * 100, el=fmt_t(el), eta=fmt_t(eta),
                      cur=done, tot=self.total, fps=fps)
        if eta is not None and f < 1:
            return tr("{lab}  {pct:3.0f}%\nминуло {el}  залишилось ~{eta}", lab=lab, pct=f * 100, el=fmt_t(el), eta=fmt_t(eta))
        return tr("{lab}  {pct:3.0f}%\nминуло {el}", lab=lab, pct=f * 100, el=fmt_t(el))


__all__ = ["ExportController", "out_size"]
