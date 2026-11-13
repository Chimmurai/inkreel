"""Керування плеєром у Qt (`PlayerMixin` для `MainWindow`): цикл, стоп, перемотка ±10 с, звук і гучність, фрагмент `[ ]`, вигляд нижнього ряду й правої
панелі під тип файлу, клавіші в будь-якій розкладці. Перенесено з `gui.App` (`_toggle_loop`, `_stop`, `_jump`, `_mute`, `_on_volume`, `_vol_step`, `_range`,
`_update_range`, `_set_mark_here`, `_layout_bottom`, `_apply_kind`, `_letter_of`, `_on_letter`)."""
from __future__ import annotations

import sys

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QAbstractItemView, QApplication, QComboBox, QLineEdit, QSizePolicy, QSpacerItem

from ..i18n import tn, tr
from ..progress import fmt_t
from .consts import G, GG, M


class Gap(QSpacerItem):
    """проміжок у розкладці, який можна «вимкнути» разом зі схованим сусідом (як padx/pady, що зникають з pack_forget)"""

    def __init__(self, w=0, h=0):
        super().__init__(w, h, QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.want = (w, h)

    def on(self, v: bool):
        self.changeSize(*(self.want if v else (0, 0)), QSizePolicy.Fixed, QSizePolicy.Fixed)


class PlayerMixin:
    # ------------------------------------------------------------------ тип файлу: що показувати
    def _apply_kind(self, kind):
        """інтерфейс під тип файлу: відео - повний, GIF - без параметрів кодека, фото - без часової шкали"""
        self.kind = kind
        self._fill_formats()
        vid = kind == "video"
        for w in (self.row2, self.row_fps, self.btn_lock):
            w.setVisible(vid)
        for g in (self.gap_row2, self.gap_fps, self.gap_lock):
            g.on(vid)
        self.side_in.layout().invalidate()
        self.seek.marks = [0, max(1, self.player.n - 1)] if vid and self.player.n else None
        self.seek.mark_cb = self._on_mark
        self.seek.draw()
        self.seek.update()
        self._update_range()
        self._layout_bottom(kind)
        self.sstack.layout()
        self.pstack.layout()

    def _layout_bottom(self, kind):
        """нижня панель: транспорт, звук, цикл, кадр->фото, час - у природну ширину; уся решта місця справа - рядок стану"""
        img, vid = kind == "image", kind == "video"
        for w in (self.tr, self.btn_loop, self.btn_photo, self.seek):
            w.setVisible(not img)
        for w in (self.btn_snd, self.vol):
            w.setVisible(vid)
        self.gap_seek.on(not img)
        self.gap_snd.on(vid)
        self.gap_vol.on(vid)
        self.gap_loop.on(not img)
        self.gap_photo.on(not img)
        self.gap_time.on(not img)                         # фото: ряд починається врівень з краєм поля
        self.bot_lay.invalidate()
        self.row_lay.invalidate()

    # ------------------------------------------------------------------ транспорт
    def _play_icon(self) -> str:
        """значок кнопки: під час рендеру - «пауза» (іде) або «грати» (стоїть), інакше - за станом плеєра"""
        if self.export.running:
            return "play" if self.export.pause_ev.is_set() else "pause"
        return "pause" if self.player.state == "playing" else "play"

    def _toggle_loop(self):
        self.player.loop = self.btn_loop.on

    def _key_loop(self):
        self.btn_loop.set_on(not self.btn_loop.on)
        self._toggle_loop()

    def _stop(self):
        if not self.ready or self.export.running:
            return
        self.player.pause()
        self.player.seek(0)
        self.btn_play.set_icon("play")

    def _jump(self, sec):
        if self.ready and not self.export.running:
            self.player.seek(self.player.position() + sec * self.player.fps)

    def _step(self, d: int):
        if not self.ready or self.export.running:
            return
        self.player.step(d)
        self.btn_play.set_icon("play")

    # ------------------------------------------------------------------ звук
    def _key_mute(self):
        if self.ready and self.kind == "video":
            self._mute()

    def _muted(self) -> bool:
        a = getattr(self.player, "audio", None)
        return bool(a is not None and a.muted)

    def _mute(self):
        m = not self._muted()
        if not m and self.vol.v < 1:                      # вмикаємо звук при нульовій гучності - піднімемо до половини
            self.vol.set(50)
            self.player.set_volume(0.5, restart=True)
        self.player.set_muted(m)
        self._sync_snd_icon()
        self._save_later()

    def _sync_snd_icon(self):
        self.btn_snd.set_icon("mute" if (self._muted() or self.vol.v < 1) else "snd")

    def _on_volume(self, v):
        """повзунок гучності тягнуть: звук змінюється на льоту; рух повзунка вимикає «без звуку»"""
        v = int(round(v))
        self.player.set_volume(v / 100)
        if v > 0 and self._muted():
            self.player.set_muted(False)
        self._sync_snd_icon()

    def _on_volume_end(self, v):
        self.player.set_volume(int(round(v)) / 100, restart=True)       # winsound: застосувати з перезапуском
        self._save_later()

    def _vol_step(self, d, key=False):
        if not self.ready or self.kind != "video" or (key and self._typing()):
            return
        v = min(100, max(0, int(self.vol.v) + d))
        self.vol.set(v)
        self._on_volume(v)
        self._on_volume_end(v)

    # ------------------------------------------------------------------ фрагмент [ ]
    def _range(self):
        """-> (початок, кінець-ексклюзивно) або None, якщо вибрано все відео"""
        m = self.seek.marks
        if not m or not self.player.n or (m[0] <= 0 and m[1] >= self.player.n - 1):
            return None
        return int(m[0]), int(m[1]) + 1

    def _update_range(self):
        r = self._range()
        fps = self.player.fps or 24.0
        if r is None:
            self.rng_lbl.config(text=tr("Фрагмент: усе відео\n[ ] — межі фрагмента"))
            self.btn_rng.set_enabled(False)
        else:
            a, b = r
            self.rng_lbl.config(text=tr("Фрагмент: {a} – {b}\n({d}, {n})", a=fmt_t(a / fps), b=fmt_t(b / fps), d=fmt_t((b - a) / fps),
                                        n=tn(b - a, "{n} кадр|{n} кадри|{n} кадрів")))
            self.btn_rng.set_enabled(True)

    def _on_mark(self, i, v):
        self._update_range()
        if self.ready:
            self.player.seek(int(v))                      # показати кадр під прапорцем

    def _reset_range(self):
        if self.seek.marks:
            self.seek.marks[:] = [0, max(1, self.player.n - 1)]
            self.seek.draw()
            self.seek.update()
            self._update_range()

    def _set_mark_here(self, i):
        m = self.seek.marks
        if not m or not self.ready:
            return
        pos = int(self.player.position())
        if i == 0:
            m[0] = min(pos, m[1] - 1)
        else:
            m[1] = max(pos, m[0] + 1)
        self.seek.draw()
        self.seek.update()
        self._update_range()

    # ------------------------------------------------------------------ фокус і клавіші
    def _editing(self) -> bool:
        """фокус у полі, де друкують (рядок вводу чи список, що дозволяє вписати): пробіл, стрілки, Home тоді не керують плеєром.
        Список лише для вибору (формат, якість) сюди не входить"""
        return isinstance(QApplication.focusWidget(), QLineEdit)

    def _typing(self) -> bool:
        """фокус у списку/полі вводу: стрілки ↑/↓ належать йому, а не гучності"""
        return isinstance(QApplication.focusWidget(), (QLineEdit, QComboBox))

    def _on_footer(self) -> bool:
        """зараз на екрані футера, а не на екрані з відео: клавіші плеєра (пауза, кадри, гучність, фрагмент) тут не діють"""
        try:
            return self.pstack.tcur != 0 or self.pstack.cur != 0
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _letter_of(e) -> str:
        """латинська літера натиснутої клавіші в будь-якій розкладці: за кодом клавіші, якщо це латинська літера; для кирилиці - за
        фізичною клавішею (на Windows віртуальна клавіша A-Z однакова в усіх розкладках) і за схожими літерами: «м» -> M, «і» -> I"""
        k = e.key()
        if Qt.Key_A <= k <= Qt.Key_Z:
            return chr(k).lower()
        alias = {"м": "m", "і": "i", "ї": "i"}.get((e.text() or "").lower())
        if alias:
            return alias
        if sys.platform == "win32" and 65 <= e.nativeVirtualKey() <= 90:
            return chr(e.nativeVirtualKey()).lower()
        return ""

    def keyPressEvent(self, e):
        mods, k = e.modifiers(), e.key()
        if mods & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
            return super().keyPressEvent(e)
        foot, edit = self._on_footer(), self._editing()
        shift = bool(mods & Qt.ShiftModifier)
        done = True
        if k == Qt.Key_Space and not (edit or foot):
            self._toggle()
        elif k in (Qt.Key_Left, Qt.Key_Right) and not (edit or foot):
            d = -1 if k == Qt.Key_Left else 1
            self._jump(10 * d) if shift else self._step(d)
        elif k in (Qt.Key_Up, Qt.Key_Down) and not foot and not self._typing():
            self._vol_step(5 if k == Qt.Key_Up else -5, key=True)
        elif k == Qt.Key_Home and not (edit or foot):
            self._stop()
        elif k in (Qt.Key_BracketLeft, Qt.Key_BracketRight) and not foot:
            self._set_mark_here(0 if k == Qt.Key_BracketLeft else 1)
        else:
            letter = self._letter_of(e)
            if letter in ("m", "l", "i", "f") and not edit and not (foot and letter in ("m", "l", "i", "f")):
                {"f": getattr(self, "toggle_cinema", lambda: None), "m": self._key_mute, "l": self._key_loop, "i": self._toggle_inv}[letter]()
            else:
                done = False
        if done:
            e.accept()
        else:
            super().keyPressEvent(e)

    def eventFilter(self, o, e):
        """клік по картинці чи кнопках забирає фокус із поля вводу: інакше пробіл, стрілки й літери лишаються «в полі»"""
        if e.type() == QEvent.MouseButtonPress and isinstance(QApplication.focusWidget(), QLineEdit):
            w = QApplication.widgetAt(e.globalPosition().toPoint()) if hasattr(e, "globalPosition") else None
            if w is not None and not isinstance(w, (QLineEdit, QComboBox, QAbstractItemView)) and not isinstance(w.parentWidget(), QComboBox):
                QApplication.focusWidget().clearFocus()
        return False
