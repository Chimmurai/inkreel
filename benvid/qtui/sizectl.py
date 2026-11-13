"""Розмір результату й ліміт кадрів на секунду в Qt (`SizeMixin` для `MainWindow`): кнопки розмірів, «свій розмір», «Авто» (підбір у фоні), підпис
розміру, мінімум для «Символів», щільність знаків, список «кадрів на секунду» з уписуваним числом, формат і якість кодування. Перенесено з `gui.App`
(`_set_size`, `_apply_custom`, `_auto_resolve/_auto_done`, `_update_size_label`, `_note_min_size`, `_show_dens`, `_apply_gdens`, `_reparam`, `_sync_fps`,
`_on_fps_pick/_on_fps_typed`, `_paste_replace`, `_on_fmt_pick`, `_on_q_pick`)."""
from __future__ import annotations

import re
import threading

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .. import glyph
from ..core import MAIN_LEVELS
from ..i18n import tr
from ..video import QUALITY, fps_options, out_size, parse_fps_limit
from .consts import AUTO


class SizeMixin:
    # ------------------------------------------------------------------ розмір результату
    def _size_key(self) -> str:
        """фактичний ключ розміру (для «Авто» - те, що підібрано)"""
        if self.size_key == AUTO:
            return self._auto_size or self.player.size_key
        return self.size_key

    def _locked(self) -> bool:
        return bool(getattr(self, "locked", False))

    def _update_size_label(self):
        info = self.player.info
        if not self.ready or not info:
            self.size_lbl.config(text="")
            return
        w, h = self.player.size
        tag = tr("авто") + " " if self.size_key == AUTO else ""
        src = f"{info['width']}×{info['height']}"
        if self.player.kind != "image" and info.get("fps"):
            src += tr(" · {f:.3f} к/с", f=info["fps"])
        if self._locked():                                    # замок: розмір сталий - такий, яким файл .ikr було порахований
            nm = (self.player.baked_meta.get("src") or {}).get("name", "")
            self.size_lbl.config(text=tr("→ {w}×{h} · замок: розмір сталий\nоригінал {n}", w=w, h=h, n=nm or src))
            return
        if self.params.glyph:                                 # символи: скільки знаків у рядку й скільки рядків
            self._show_dens()
            rows, cols = glyph.grid_for(w, h, self.params.gdens)
            cw, ch = glyph.cell_for(w, h, self.params.gdens)
            self.size_lbl.config(text=tr("→ {tag}{w}×{h} · {c}×{r} символів ({cw}×{ch} px)\nджерело {src}", tag=tag, w=w, h=h, c=cols, r=rows, cw=cw, ch=ch, src=src))
            return
        self.size_lbl.config(text=tr("→ {tag}{w}×{h}\nджерело {src}", tag=tag, w=w, h=h, src=src))

    def _show_dens(self):
        """підпис над повзунком щільності: скільки знаків і яка клітинка при поточному розмірі результату"""
        d = glyph.dens_of(self.params.gdens)
        info = self.player.info if self.ready else None
        txt = tr("Щільність знаків") + f"  {d}/{glyph.DENS_N}"             # два короткі рядки: довгий в одному не вміщається в панель
        if info:
            w, h = self.player.size
            rows, cols = glyph.grid_for(w, h, d)
            cw, ch = glyph.cell_for(w, h, d)
            txt += "\n" + tr("{c}×{r} · клітинка {cw}×{ch} px", c=cols, r=rows, cw=cw, ch=ch)
        self.dens_lbl.config(text=txt)

    def _set_size(self, key: str):
        self.size_key = key
        for k, b in self.size_btns.items():
            b.set_on(k == key)
        if key == AUTO:
            self._auto_resolve()
        else:
            self._auto_size = None
            if self.ready:
                self.player.set_size(key)
        self._update_size_label()
        self._note_min_size()
        self._save_later()

    def _note_min_size(self):
        """«Символи»: результат не буває меншим за один знак (при поточній крупності); якщо вибраний розмір менший - його піднято, і про це кажемо"""
        try:
            if not (self.params.glyph and self.ready and self.player.info and not self._locked() and not self.export.running):
                return
            info = self.player.info
            if out_size(info["width"], info["height"], self._size_key()) != tuple(self.player.size):
                w, h = self.player.size
                self._say("Розмір піднято до одного символа: {w}×{h}", w=w, h=h)
        except Exception:  # noqa: BLE001
            pass

    # --- «Авто»: підбір у фоні
    def _auto_resolve(self, keep=False):
        """автовибір розміру у фоні; поки рахується - тимчасово 240p (keep=True: лишається поточний розмір - так при зміні рівня якості картинка не блимає малою)"""
        if not keep:
            self._auto_size = None
        if not self.ready:
            return
        if not keep:
            self.player.set_size("240p")
        self._auto_tok += 1
        tok = self._auto_tok
        pl = self.player
        fallback = pl.size_key if keep else "240p"

        def work():
            try:
                k = pl.auto_pick()
            except Exception:  # noqa: BLE001
                from .. import log as _log
                _log.get().exception("автовибір розміру")
                k = fallback
            finally:
                pl.hold(False)                                    # про всяк випадок: буфер не повинен лишитись на паузі
            self.auto_done.emit(tok, k)                           # сигнал - у потік інтерфейсу
        threading.Thread(target=work, daemon=True, name="inkreel-autosize").start()

    def _auto_done(self, tok, key):
        if tok != self._auto_tok or self.size_key != AUTO or not self.ready:
            return
        self._auto_size = key
        self.player.set_size(key)
        self._update_size_label()

    # --- свій розмір
    @staticmethod
    def _valid_custom(w, h):
        if not (w or h):
            return False
        return all(v is None or 2 <= v <= 4096 for v in (w, h))

    def _apply_custom(self):
        def num(s):
            s = s.strip()
            return int(s) if s.isdigit() else None
        w, h = num(self.ent_w.text()), num(self.ent_h.text())
        if not self._valid_custom(w, h):
            self._say("свій розмір: ціле число 2…4096 (ширина, висота або обидві)")
            return
        self._set_size(f"{w or ''}x{h or ''}")

    def _paste_replace(self, w):
        """вставка в поле «ширина», «висота» чи «кадри»: вміст поля ЗАМІНЮЄТЬСЯ тим, що в буфері. У поля розміру «1920x1080» (х, ×, *) лягає одразу в обидва"""
        txt = (QApplication.clipboard().text() or "").strip().splitlines()
        txt = (txt[0] if txt else "").strip()
        if w in (self.ent_w, self.ent_h):
            m = re.match(r"^(\d{1,5})\s*[xXхХ×*]\s*(\d{1,5})$", txt)        # x, кирилична х, ×, *
            if m:
                self.ent_w.setText(m.group(1))
                self.ent_h.setText(m.group(2))
                return
            txt = re.sub(r"\D", "", txt)[:5]
        w.setText(txt)

    # ------------------------------------------------------------------ щільність, рівень, чорнило/символи: те, що тягне за собою розмір
    def _apply_gdens(self, n: int):
        n = min(3, max(1, int(n)))
        self.dens_bar.set(n)
        if n != self.params.gdens:
            self.params.gdens = n
            self._reparam()                                   # крупніший знак може підняти й розмір результату (мінімум - одна клітинка): плеєр уже знає новий
            self._show_dens()
            self._update_size_label()
            self._note_min_size()
            self._save_later()

    def _reparam(self):
        """параметри кадру змінились (рівень або чорнило/символи): плеєр рахує наново, розмір «Авто» підбирається під нову вартість"""
        if self.ready:
            auto = self.size_key == AUTO and self.player.kind != "image"
            if auto:
                self.player.hold(True)
            self.player.set_params(self.params)
            if auto:
                self._auto_resolve(keep=True)

    def _neuro_start(self):
        """IRNM1 вибрано: у фоні вибираємо, на чому рахувати (кілька секунд на першому разі, далі з пам'яті); немає моделі чи onnxruntime - кажемо одразу"""
        try:
            from .. import student
            from ..student import runtime as sr
            if sr.mode() == "off" or not sr.files_ok() or sr._ort() is None:
                self._say("IRNM1 тут недоступна (немає нейромережі чи onnxruntime): рахується як «Гравюра»")
            else:
                student.start_background()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ кадри на секунду
    def _src_fps(self) -> float:
        try:
            return float(self.player.fps or 0) if self.kind == "video" and self.ready else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def _fps_limit(self):
        """ліміт кадрів на секунду для експорту: число або None (як в оригіналі). Діє, лише якщо він нижчий за fps цього відео"""
        src = self._src_fps()
        return self.fps_sel if src and 1.0 <= self.fps_sel < src - 0.5 else None

    @staticmethod
    def _fps_name(v) -> str:
        return tr("Кадри: {f} к/с", f=f"{v:g}")

    def _sync_fps(self):
        """список «кадрів на секунду»: як в оригіналі й стандартні значення нижче за частоту відео; вибір пам'ятається й для інших відео"""
        src = self._src_fps()
        self._fps_vals = [0] + fps_options(src)
        self._fps_orig = tr("Кадри: як в оригіналі ({f:g} к/с)", f=round(src, 3)) if src else tr("Кадри: як в оригіналі")
        self.fps_cb.blockSignals(True)
        self.fps_cb.clear()
        self.fps_cb.addItems([self._fps_orig] + [self._fps_name(v) for v in self._fps_vals[1:]])
        self.fps_cb.blockSignals(False)
        self._show_fps()
        self._apply_fps_limit()

    def _apply_fps_limit(self):
        """ліміт кадрів на секунду діє й на перегляд: плеєр рахує в буфер лише кадри, що потраплять у відео (не весь потік джерела)"""
        if not self.ready:
            return
        try:
            self.player.set_fps_limit(self._fps_limit())
        except Exception:  # noqa: BLE001
            from .. import log as _log
            _log.get().exception("fps limit")

    def _show_fps(self):
        """у полі - ліміт, що діє зараз: «як в оригіналі», пункт списку чи вписане своє число"""
        lim = self._fps_limit()
        cb = self.fps_cb
        cb.blockSignals(True)
        if lim and lim in self._fps_vals[1:]:
            cb.setCurrentIndex(self._fps_vals.index(lim))
        elif lim:
            cb.setEditText(self._fps_name(lim))
        else:
            cb.setCurrentIndex(0)
        cb.blockSignals(False)
        self._fps_shown = cb.currentText()

    def _on_fps_pick(self, i: int):
        if 0 <= i < len(self._fps_vals):
            self.fps_sel = float(self._fps_vals[i])
            self._fps_shown = self.fps_cb.currentText()
            self._apply_fps_limit()
            self._save_later()

    def _on_fps_typed(self):
        """Enter чи вихід з поля: вписане число стає лімітом (не нижче 1; від fps відео й вище чи 0 - як в оригіналі; не число - лишається попереднє).
        Нічого не вписано (текст той самий, що ми показали) - вибір, який пам'ятається для інших відео, не чіпаємо"""
        src = self._src_fps()
        text = self.fps_cb.currentText()
        if src and text != self._fps_shown:
            v = parse_fps_limit(text, src)
            if v is not None and v != self.fps_sel:
                self.fps_sel = v
                self._apply_fps_limit()
                self._save_later()
        if src:
            self._show_fps()

    def _on_fps_enter(self):
        self._on_fps_typed()
        self.fps_cb.lineEdit().deselect()
        self.fps_cb.lineEdit().clearFocus()

    # ------------------------------------------------------------------ формат і якість кодування
    def _on_fmt_pick(self, i: int):
        self._save_later()

    def _on_q_pick(self, i: int):
        self._save_later()
