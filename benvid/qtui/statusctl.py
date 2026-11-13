"""Такт інтерфейсу, рядок стану, оцінка часу експорту, HUD «що робить програма зараз», журнал плавності показу в Qt (`StatusMixin` для `MainWindow`).
Перенесено з `gui.App` (`_tick`, `_fit_status`, `_estimate`, `_hud_info`, `_hud_update`, `_meter`)."""
from __future__ import annotations

import time

from PySide6.QtGui import QFontMetrics

from .. import gpu
from .. import log as _log
from ..core import BRANCH
from ..i18n import tr
from ..progress import fmt_t
from .consts import FONT_S
from .theme import tk_font


class StatusMixin:
    # ------------------------------------------------------------------ рядок стану
    def _fit_status(self, segs, sep=" · "):
        """segs - [(важливість, текст)] у порядку показу (менше число = важливіше). Поки рядок не влазить у ширину підпису,
        прибираємо найменш важливу частину: час, прапорці й кнопки ніколи не затискаються довгим текстом стану"""
        fm = QFontMetrics(tk_font(FONT_S))
        avail = self.stat_lbl.width() - 8
        items = list(segs)
        self._stat_full = sep.join(t for _, t in items)
        while items:
            txt = sep.join(t for _, t in items)
            if fm.horizontalAdvance(txt) <= avail:
                return txt
            items.pop(max(range(len(items)), key=lambda i: items[i][0]))
        return ""

    def _estimate(self, pl):
        """скільки приблизно триватиме експорт: ще не пораховані кадри / поточна швидкість обробки"""
        txt = ""
        if not self.export.running and pl.kind != "image":
            r = pl.render_fps()
            a, b = self._range() or (0, pl.n)
            tot, got = pl.total_in(a, b), pl.cached_in(a, b)
            todo = max(0, tot - got)
            if todo == 0:
                txt = tr("Експорт: усе пораховано — миттєво")
            elif r > 1:
                txt = tr("Експорт ≈ {t}  (готово {p}%)", t=fmt_t(todo / r), p=100 * got // max(1, tot))
        self.est_lbl.config(text=txt)

    def _status_segments(self, pl, pos):
        fps = pl.fps
        ahead = (pl.buffered_end() - int(pos)) / fps
        st = {"playing": "відтворення", "buffering": "буферизація…", "paused": "пауза"}[pl.state]
        if pl.kind == "image":
            return [(0, f"{pl.size[0]}×{pl.size[1]}"), (1, tr("обробка…" if self.surface.mask is None else "готово"))]
        a = pl.audio.status
        snd = "♪" if a == "ок" else "♪ " + tr("без звуку" if a == "немає доріжки" else "немає sounddevice" if a.startswith("встанови") else a)
        segs = [(0, tr(st)), (1, tr("запас {s:.1f} с", s=ahead))]
        if pl.stride > 1:
            segs.append((2, tr("кожен {n}-й", n=pl.stride)))
        segs += [(3 if a != "ок" else 6, snd), (4, tr("обробка {r:.0f} к/с", r=pl.render_fps())),         # один лічильник: усі потоки разом
                 (7, f"{pl.workers}×"), (5, tr("файл {f:.2f} к/с", f=fps))]
        if self.params.method == BRANCH and not self.params.glyph:
            try:
                from ..student import label
                if label():
                    segs.insert(2, (6, "IRNM1: " + label()))              # чим рахує модель: TensorRT / CUDA / DirectML · GPU або CPU
            except Exception:  # noqa: BLE001
                pass
        return segs

    # ------------------------------------------------------------------ HUD
    def _hud_info(self):
        """-> (текст, чи крутиться лічильник) або None, коли все спокійно (іде відтворення чи стоїть готовий кадр). Будь-яке очікування -
        рендер, буферизація, прогрів «Гравюри», обробка кадра, пропуск кадрів - супроводжується написом і анімацією"""
        if self.export.running:
            f = self.export.f
            gtag = " · GPU" if self.params.method == 5 and not self.params.glyph and gpu.slots_cached() else ""        # відеокарта бере участь у рендері
            if self.params.method == BRANCH and not self.params.glyph:
                try:
                    from ..student import current, label
                    gtag = " · " + label() if current() in ("cuda", "dml", "trt") else ""
                except Exception:  # noqa: BLE001
                    gtag = ""
            if self.export.pause_ev.is_set():
                return tr("ПАУЗА · {pct:.0f}%", pct=f * 100), False
            tot = self.export.total
            if tot > 1 and f > 0:
                return tr("РЕНДЕР · кадр {cur}/{tot}", cur=min(tot, int(round(f * tot))), tot=tot) + gtag, True
            return tr("РЕНДЕР · {pct:.0f}%", pct=f * 100) + gtag, True
        if not self.ready:
            return None
        pl = self.player
        if pl.warming:
            return tr("ГОТУЮ СИМВОЛИ… перший запуск довший" if self.params.glyph else "ГОТУЮ ГРАВЮРУ… перший запуск довший"), True
        if self.params.method == BRANCH and not self.params.glyph:
            try:
                from ..student import preparing
                if preparing():
                    from ..student import stage_label
                    sl = stage_label()
                    return (tr("ГОТУЮ IRNM1… перевіряю {b}", b=sl) if sl else tr("ГОТУЮ IRNM1… перший запуск довший")), True
            except Exception:  # noqa: BLE001
                pass
        if pl.playing and pl.state == "buffering":
            have, need = pl.buffer_progress()
            t = tr("БУФЕРИЗАЦІЯ · {have}/{need} кадрів", have=have, need=need)
            if pl.stride > 1:
                t += " · " + tr("кожен {n}-й", n=pl.stride)
            return t, True
        if pl.waiting_frame():
            return tr("ОБРОБКА…"), True
        if pl.playing and pl.stride > 1:
            return tr("ПК не встигає · кожен {n}-й кадр", n=pl.stride), False
        return None

    def _hud_update(self):
        info = self._hud_info()
        phase = int(time.perf_counter() * 8) % 8 if info and info[1] else 0
        self.surface.set_hud(info, phase)

    # ------------------------------------------------------------------ такт
    def _tick(self):
        t0 = time.perf_counter()
        try:
            r = self.player.update() if self.ready else None
            if r:
                self._fr_idx = int(r[0])
                self.surface.set_mask(r[1])
            now = time.perf_counter()
            if now - self._last_lay_t > 0.3:                     # вміст міг змінити висоту (довший текст, інша мова): перерахувати прокрутки
                self._last_lay_t = now
                rq = (self.side_in.winfo_reqheight(), self.foot.winfo_reqheight(), self.main.winfo_reqheight())
                if rq != self._lay_req:
                    self._lay_req = rq
                    self._relayout()
            if now - self._hud_t > 0.1:
                self._hud_t = now
                self._hud_update()
            if self.ready:
                if self._seek_pending is not None and now - self._seek_t >= self.SEEK_GAP:
                    self._seek_t = now
                    self.player.seek(int(self._seek_pending))
                    self._seek_pending = None
                if now - self._last_info > 0.15:
                    self._last_info = now
                    pl = self.player
                    pos = pl.position()
                    if not self.seek.drag:
                        self.seek.set(pos)
                    if pl.kind == "image":
                        self.time_lbl.config(text=tr("ФОТО {w}×{h}", w=pl.info["width"], h=pl.info["height"]))
                    else:
                        self.time_lbl.config(text=f"{fmt_t(pl.time_of(pos))} / {fmt_t(pl.duration)}")
                    self.btn_play.set_icon(self._play_icon())
                    if now - self._last_rng_t > 0.5:             # важчі й рідше змінні підписи - не частіше, щоб не збивати ритм кадрів
                        self._last_rng_t = now
                        k = max(1.0, (pl.n - 1) / max(1, self.seek.width() - 16))       # злити відрізки в межах пікселя
                        rs = []
                        for a, b in pl.buffered_ranges():
                            if rs and a - rs[-1][1] <= k:
                                rs[-1][1] = b
                            else:
                                rs.append([a, b])
                        self.seek.set_ranges([tuple(r) for r in rs])
                        self.stat_lbl.config(text=self._fit_status(self._status_segments(pl, pos)))
                        self._estimate(pl)
        except Exception:  # noqa: BLE001
            _log.get().exception("помилка такту")
        try:
            self._meter(time.perf_counter() - t0)
        except Exception:  # noqa: BLE001
            self._pm = None

    # ------------------------------------------------------------------ журнал плавності показу
    def _meter(self, tick_s):
        """лічильник плавності показу: раз на 10 с відтворення пише в журнал, скільки кадрів показано, скільки коштує малювання кадра й яка найдовша пауза такту"""
        m = self._pm
        now = time.perf_counter()
        pl = self.player
        if m is None or not self.ready or pl.state != "playing":
            self._pm = None if pl.state != "playing" or not self.ready else dict(t0=now, n0=self.surface.stats["frames"], p0=self.surface.stats["paint_ms"], tkmax=0.0)
            return
        m["tkmax"] = max(m["tkmax"], tick_s)
        if now - m["t0"] >= 10.0:
            st = self.surface.stats
            n = st["frames"] - m["n0"]
            if n:
                _log.get().info("показ (Qt): %d кадрів за %.0f с (%.1f к/с); малювання кадра в середньому %.1f мс; найдовший такт інтерфейсу %.0f мс; картинка %s, поле %dx%d; "
                                "файл %.3f к/с, %dx%d", n, now - m["t0"], n / (now - m["t0"]), (st["paint_ms"] - m["p0"]) / n, 1000 * m["tkmax"],
                                "x".join(str(v) for v in ((self.surface._img.width(), self.surface._img.height()) if self.surface._img is not None else (0, 0))),
                                self.surface.width(), self.surface.height(), pl.fps, pl.size[0], pl.size[1])
            self._pm = dict(t0=now, n0=st["frames"], p0=st["paint_ms"], tkmax=0.0)
