"""Файли в Qt (`FileMixin` для `MainWindow`): відкриття (`_open` з `start/size/seed`), режим замка (`.ikr`: «Порахувати все» й «Звичайний режим»),
тека результату, повідомлення в правій панелі. Перенесено з `gui.App` (`_open`, `_vp`, `_lock_guard`, `_sync_lock`, `_lock_toggle`, `_enter_lock`, `_frozen_size`,
`_same_original`, `_unfreeze`, `_show_method`, `_remember_dir`, `_src_dir`, `_open_out`, `_say`, `_short_path`)."""
from __future__ import annotations

import os

from PySide6.QtWidgets import QFileDialog, QMessageBox

from .. import log as _log
from ..core import GLYPH_INFO, MAIN_LEVELS, METHOD_INFO, METHODS, Params
from ..i18n import tr
from ..media import GIF_EXT, IMAGE_EXT
from ..video import VIDEO_EXT, out_size, parse_size
from .consts import AUTO


class FileMixin:
    # ------------------------------------------------------------------ повідомлення в правій панелі
    def _say(self, key: str = "", done: bool = False, **kw):
        """повідомлення в нижньому блоці правої панелі (прогрес, результат, помилки). key - ключ перекладу, kw - підстановки:
        повідомлення пам'ятається так, щоб мову можна було змінити на ходу"""
        self._exp_msg = (key, kw)
        self._prog_done = done
        self.prog_lbl.config(text=tr(key, **kw) if key else "")

    @staticmethod
    def _short_path(p, n=36):
        """«…\\тека\\файл.mp4» - щоб шлях влазив у вузьку панель, а ім'я файлу лишалось цілим"""
        p = p.rstrip("\\/")
        name, d = os.path.basename(p), os.path.basename(os.path.dirname(p))
        s = os.path.join(d, name) if d else name
        if os.path.dirname(os.path.dirname(p)) not in ("", os.path.dirname(p)):
            s = "…" + os.sep + s
        return s if len(s) <= n else "…" + s[-(n - 1):]

    def _remember_dir(self, path):
        self._last_dir = os.path.dirname(os.path.abspath(path))
        self._out_dir = None

    def _src_dir(self):
        return self._out_dir or os.path.dirname(self.player.path or "") or None

    def _open_out(self):
        """клік по підсумку експорту: відкрити теку з результатом"""
        out = getattr(self, "_last_out", None)
        if not out or not self._prog_done:
            return
        d = out if os.path.isdir(out) else os.path.dirname(out)
        self._out_dir = d
        try:
            if os.name == "nt":
                os.startfile(d)  # noqa: S606
            else:
                import subprocess
                subprocess.Popen(["xdg-open", d])
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ кадр -> фото
    def _save_frame(self):
        """поточний кадр (відео/GIF) -> окреме 1-бітне фото (PNG, BMP, TIFF, WebP; у «Символах» ще .txt зі знаками)"""
        mask = self.surface.mask
        if mask is None or not self.ready:
            return
        from .. import glyph
        from ..media import save_still, save_text
        base = os.path.splitext(os.path.basename(self.player.path))[0]
        types = [("PNG", "*.png", ".png"), ("BMP", "*.bmp", ".bmp"), ("TIFF", "*.tif", ".tif"), ("WebP", "*.webp", ".webp")]
        vp = self._vp()                                       # в замку - ті налаштування, з якими файл порахований
        if vp.glyph:
            types.append((tr("Текст (.txt, символи)"), "*.txt", ".txt"))
        flt = ";;".join(f"{n} ({pat})" for n, pat, _ in types)
        name = f"{base}_{int(self.player.position()) + 1:05d}.png"
        out, chosen = QFileDialog.getSaveFileName(self, tr("Зберегти кадр як фото"), os.path.join(self._src_dir() or "", name), flt)
        if not out:
            return
        if not os.path.splitext(out)[1]:                       # як defaultextension у Tk: без розширення - за вибраним типом (типово .png)
            ext = next((e for n, pat, e in types if chosen.startswith(n)), ".png")
            out += ext
        try:
            if os.path.splitext(out)[1].lower() == ".txt":
                mh, mw = mask.shape[:2]
                save_text(glyph.text_from_mask(mask, mw, mh, vp.gdens), out)
            else:
                save_still(mask, out, self._pal_key())
            self._last_out = out
            self._say("Кадр збережено:\n{p}\nклік — відкрити теку", done=True, p=self._short_path(out))
        except Exception as e:  # noqa: BLE001
            self._say("Помилка: {e}", e=e)

    # ------------------------------------------------------------------ рівень: підпис секції й пояснення
    def _show_method(self, n):
        """у смузі заголовка секції - лише кодова назва поточного рівня; пояснення під повзунком"""
        vp = self._vp()
        n = vp.method if self.locked else n
        if vp.glyph:
            n = min(MAIN_LEVELS, n)                          # у символах шостого рівня нема
        lock = "  ·  " + tr("ЗАМОК") if self.locked else ""
        pos = f"  {n}/{MAIN_LEVELS}" if n <= MAIN_LEVELS else ""      # IRNM1 - гілка: місця на шкалі «n із 5» не має
        self.sec_q.title.config(text=f"{tr('Якість').upper()}  ·  {tr(METHODS[n]).upper()}{pos}{lock}")
        self.m_info.config(text=tr((GLYPH_INFO if vp.glyph else METHOD_INFO)[n]))

    # ------------------------------------------------------------------ відкриття
    def _open(self, path: str | None = None, start: int = 0, size=None, seed=None):
        """start - кадр, з якого стартує буфер; size - готовий ключ розміру (тоді «Авто» не підбирається знову); seed - див. Player.open"""
        if not path:
            exts = tuple(VIDEO_EXT) + tuple(IMAGE_EXT) + tuple(GIF_EXT)
            path, _ = QFileDialog.getOpenFileName(
                self, tr("Відкрити відео, GIF або фото"), self._last_dir or "",
                tr("Усі медіа") + " (" + " ".join("*" + e for e in exts) + ");;" + tr("Відео") + " (" + " ".join("*" + e for e in VIDEO_EXT) + ");;GIF (*.gif);;"
                + tr("Фото") + " (" + " ".join("*" + e for e in IMAGE_EXT) + ");;" + tr("Усі файли") + " (*.*)")
        if not path:
            return
        try:
            want = self.size_key
            info = self.player.open(path, size or ("240p" if want == AUTO else want), start=start, seed=seed)
        except Exception as e:  # noqa: BLE001
            _log.get().exception("не вдалося відкрити %s", path)
            QMessageBox.critical(self, "InkReel", tr("Не вдалося відкрити файл:\n{e}\n\nДеталі: {p}", e=e, p=_log.shown_path()))
            return
        self.ready = True
        self._show_app()
        self.path = path
        self._name = os.path.basename(path)
        self._lock_p = None
        self.player.set_params(self.params)
        self.surface.reset_view()
        self.surface.set_mask(None)
        self._apply_kind(info["kind"])
        self.seek.lo, self.seek.hi = 0, max(1, info["frames"] - 1)
        self.seek.ranges = []
        self.seek.set(min(max(0, int(start)), max(0, info["frames"] - 1)))
        self._reset_range()
        if self.player.baked or size:
            self._auto_tok += 1                           # запізнілий автовибір попереднього файлу не повинен чіпати цей
        elif want == AUTO:
            self._auto_resolve()
        self._sync_lock()
        self.btn_play.set_icon("play")
        self._tr()                                        # підписи, назва файлу, підпис розміру й кадри на секунду під новий файл
        self._say("⚠ {note}", note=tr(info["note"])) if info.get("note") else self._say()
        self._remember_dir(path)
        self._relayout()

    def _relayout(self):
        """вміст міг змінити висоту (інший тип файлу, мова, довший текст): перерахувати обидві прокрутки"""
        self.side_in.layout().invalidate()
        self.sstack.layout()
        self.pstack.layout()

    # ------------------------------------------------------------------ режим замка (файл .ikr: кадри готові, нічого не рахується)
    @property
    def locked(self) -> bool:
        return bool(getattr(self, "ready", False) and getattr(self.player, "baked", False))

    def _vp(self):
        """налаштування, які ПОКАЗУЄ інтерфейс: у замку - ті, з якими порахований файл, інакше поточні"""
        if not self.locked:
            return self.params
        if getattr(self, "_lock_p", None) is None:
            self._lock_p = Params.from_dict(dict(self.player.baked_meta.get("params") or {}))
        return self._lock_p

    def _lock_guard(self) -> bool:
        """True - зараз замок, зміна відхилена (з підказкою)"""
        if self.locked:
            self._say("Режим замка: кадри вже пораховані. Щоб змінити налаштування, натисни «Звичайний режим»")
            return True
        return False

    def _sync_lock(self):
        """вигляд регуляторів під режим: в замку якість, чорнило/символи й розмір вимкнені й показують те, з чим порахований файл"""
        lk = self.locked
        vp = self._vp()
        for w in (self.m_slider, self.dens_bar, self.btn_ink, self.btn_sym, self.btn_ok, *self.size_btns.values()):
            w.set_enabled(not lk)
        for e in (self.ent_w, self.ent_h):
            e.setEnabled(not lk)
        self.m_slider.set(vp.method)
        self.dens_bar.set(vp.gdens)
        self.btn_ink.set_on(not vp.glyph)
        self.btn_sym.set_on(bool(vp.glyph))
        self.m_slider.set_branch(not vp.glyph)
        self.dens_box.setVisible(bool(vp.glyph))
        for k, b in self.size_btns.items():
            b.set_on(False if lk else k == self.size_key)
        self._show_method(vp.method)
        self._show_dens()
        self._export_labels()
        self._relayout()

    def _export_labels(self):
        t = tr({"video": "Зберегти…", "gif": "Зберегти GIF", "image": "Зберегти фото"}[self.kind])
        for b in (self.btn_export, self.btn_export2):
            b.set_text(t)
        self.btn_lock.set_text(tr("Звичайний режим") if self.locked else tr("Порахувати все"))

    def _lock_toggle(self):
        if not self.ready or self.export.running or self.kind != "video":
            return
        if self.locked:
            self._unfreeze()
        else:
            self._export(freeze=True)

    def _enter_lock(self, path):
        """експорт .ikr закінчено: відкрити його в режимі замка"""
        try:
            self._open(path)
        except Exception:  # noqa: BLE001
            _log.get().exception("не вдалося відкрити готовий файл .ikr")
            return
        if self.locked:
            self._say("Готово: усе відео пораховане, нічого не перераховується.\n«Звичайний режим» — повернутись до оригіналу")

    def _frozen_size(self, sm, size):
        """ключ розміру, що дає ТОЙ САМИЙ розмір, яким порахований файл: тоді розморожене відео не треба перераховувати й «Авто» не підбирається знов"""
        size = (int(size[0]), int(size[1]))
        sw, sh = int(sm.get("width") or 0), int(sm.get("height") or 0)
        cur = self.size_key
        cands = []
        if sw and sh:                                            # пресети рахуються від розміру оригіналу
            cands = [self._auto_size if cur == AUTO else cur] + [k for k in self.size_btns if k != AUTO]
        cands.append(f"{size[0]}x{size[1]}")                     # крайній випадок: цей самий розмір числами
        for k in cands:
            if not k:
                continue
            try:
                if out_size(sw or size[0], sh or size[1], k) == size:
                    return k
            except Exception:  # noqa: BLE001
                continue
        return None

    @staticmethod
    def _same_original(sm, path) -> bool:
        """оригінал той самий, з якого порахований файл (розмір і час зміни з заголовка .ikr): тільки тоді його кадри можна брати замість розрахунку"""
        try:
            st = os.stat(path)
        except OSError:
            return False
        return int(sm.get("size") or -1) == st.st_size and abs(float(sm.get("mtime") or 0) - st.st_mtime) < 2.0

    def _unfreeze(self):
        """назад до звичайного режиму: відкрити оригінал, повернути налаштування, з якими порахований файл, і ту саму мить.
        Пораховані кадри не викидаються: плеєр бере їх із самого файлу .ikr, тож буфер не рахується заново з нуля"""
        m = dict(self.player.baked_meta or {})
        sm = dict(m.get("src") or {})
        ikr = self.player.path
        path = sm.get("path") or ""
        if not (path and os.path.isfile(path)):
            ok = QMessageBox.question(
                self, "InkReel", tr("Оригіналу «{n}» поряд немає, а файл .ikr зберігає лише готовий результат, без вихідного відео.\n\n"
                                    "Щоб повернутись до звичайного режиму й міняти налаштування, потрібен оригінал: кадри доведеться порахувати знову.\n"
                                    "Вказати оригінал?", n=sm.get("name") or "?"), QMessageBox.Ok | QMessageBox.Cancel)
            if ok != QMessageBox.Ok:
                return
            exts = tuple(VIDEO_EXT)
            path, _ = QFileDialog.getOpenFileName(
                self, tr("Знайти оригінал: {n}", n=sm.get("name") or "?"), os.path.dirname(path) if path else self._last_dir or "",
                tr("Відео") + " (" + " ".join("*" + e for e in exts if e != ".ikr") + ");;" + tr("Усі файли") + " (*.*)")
            if not path:
                return
        pl = self.player
        i = int(pl.position())
        k = (float(sm.get("fps") or 0) / float(pl.fps)) if sm.get("fps") and pl.fps else 1.0
        base = int((m.get("range") or [0, 0])[0] or 0)
        idx = base + int(round(i * k))
        pd = dict(m.get("params") or {})
        for key in ("method", "glyph", "gdens"):                # налаштування файлу стають поточними (інакше «розморожене» виглядало б інакше)
            if key in pd:
                setattr(self.params, key, type(getattr(self.params, key))(pd[key]))
        self.m_slider.set(self.params.method)
        self.dens_bar.set(self.params.gdens)
        self.btn_ink.set_on(not self.params.glyph)
        self.btn_sym.set_on(bool(self.params.glyph))
        key = self._frozen_size(sm, pl.size)                     # розмір лишається таким, яким порахований файл
        if key and not (self.size_key == AUTO and self._auto_size == key):
            self.size_key = key
            self._auto_size = None
            if key not in self.size_btns:                        # свій розмір: показати його в полях
                cw, ch = parse_size(key)
                self.ent_w.setText(str(cw or ""))
                self.ent_h.setText(str(ch or ""))
            for kk, b in self.size_btns.items():
                b.set_on(kk == key)
        seed = None
        if key and abs(k - 1.0) < 1e-3 and self._same_original(sm, path) and Params.from_dict(pd).to_dict() == self.params.to_dict():
            seed = dict(path=ikr, base=base)                     # налаштування, розмір і оригінал ті самі - кадри з файлу точно такі, які порахував би плеєр
        self._open(path, start=idx, size=key, seed=seed)
        if self.locked:                                          # відкрилось знов .ikr (мало б не статись)
            return
        self._show_method(self.params.method)
        self._show_dens()
        self._update_size_label()
        self._save_later()
        self._say("Звичайний режим: оригінал відкрито, налаштування файлу повернуто")
