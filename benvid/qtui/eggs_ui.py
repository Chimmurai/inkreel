"""Підказки наведенням і пасхалки на Qt (`EggsMixin` для `MainWindow`): перенесено з `gui.App` (`_tip`, `_bubble`, `_reel_click`, `_spin_reel`,
`_logo_click`, `_quote_click`, `_egg_defocus/_egg_tick`, `_swatch_click/_party`, `_cookie_click/_crumbs_tick`). Уся логіка й тексти - у `benvid/eggs.py`;
розфокусування й підпис малює `FrameSurface` (див. `set_defocus`)."""
from __future__ import annotations

import time

from PySide6.QtCore import QTimer

from .. import eggs, palette
from ..i18n import tr
from . import assets
from .theme import THEME
from .tips import Bubble, Tip

# тексти підказок (ключі перекладу): віджет -> текст; ті самі рядки, що в Tk-версії (`App._tip`)
TIPS = {
    "btn_rew": "−10 с   [Shift + ←]",
    "btn_prev": "Кадр назад   [←]",
    "btn_stopp": "Стоп: на початок   [Home]",
    "btn_next": "Кадр уперед   [→]",
    "btn_fwd": "+10 с   [Shift + →]",
    "btn_snd": "Звук: вимкнути / увімкнути   [M]",
    "vol": "Гучність   [↑ ↓]  або колесо миші",
    "btn_loop": "Повтор: після кінця починати спочатку   [L]",
    "btn_photo": "Зберегти поточний кадр окремим фото",
    "seek": "Перемотування. Прапорці [ ] над шкалою — фрагмент для збереження",
    "btn_open": "Відкрити відео, GIF або фото   [Ctrl + O]",
    "btn_export": "Зберегти результат   [Ctrl + S]",
    "btn_stop": "Зупинити збереження",
    "btn_help": "Гайд, клавіші, про програму   [F1]",
    "btn_ua": "Українська",
    "btn_en": "English",
    "btn_ink": "Чорнило: кадр малюється штрихами, крапками й зерном",
    "btn_sym": "Символи: кадр складається зі знаків шрифту (як текст у консолі); ті самі п'ять рівнів якості",
    "m_slider": ("Баланс «швидкість ↔ якість»: п'ять рівнів. Чим вище, тим більше роботи на кадр. Ромб справа - IRNM1 (InkreelNeuralModel v1): "
                 "не шостий щабель, а окрема гілка (навчена нейромережа на відеокарті); у «Символах» її нема"),
    "dens_bar": "Щільність знаків: розмір знака 32×64, 16×32 або 8×16 px. Чим дрібніше, тим більше знаків уміщається в той самий кадр і тим довше рахується",
    "btn_pprev": "Попередня палітра",
    "btn_pnext": "Наступна палітра",
    "btn_pinv": "Навпаки: туш і папір міняються місцями   [I]",
    "q_cb": ("Якість кодування. «Висока» - програмний x264 без поспіху; «Середня» й «Мала» - апаратне "
             "кодування відеокарти, швидше й менші файли"),
    "btn_aud": "Додати до файлу звук оригіналу",
    "btn_rng": "Скинути фрагмент: зберігати все відео",
    "fps_cb": ("Кадрів на секунду в результаті: виберіть зі списку або впишіть своє число (Enter). «Як в оригіналі» - без змін; "
               "нижче значення - легший файл і швидший розрахунок: кадри беруться за часом, тривалість і звук лишаються"),
    "edge": "Права панель: сховати / показати",
    "phil_lbl": "…розфокусуй погляд",
    "about_lbl": "Офіційні джерела: сайт, GitHub, Microsoft Store",
}
SIZE_TIPS = {"Авто": "Найбільший розмір, який ПК тягне в реальному часі на обраному рівні якості", "Як у джерела": "Розмір оригіналу"}
LOCK_TIP_ON = "Звичайний режим: повернутись до оригіналу цього відео з тими самими налаштуваннями. Уже пораховані кадри лишаються"
LOCK_TIP_OFF = ("Порахувати все: обробити все відео наперед і зберегти у файл .ikr (без втрат). Він відкриється в режимі замка - "
                "кадри готові, перемотка й експорт миттєві, нічого не перераховується; файл можна відкрити в InkReel будь-коли")
PLAY_TIPS = ("Відтворення / пауза   [Пробіл]", "Пауза / продовження рендеру   [Пробіл]")
DONATE_TIPS = ("Скоро тут з'являться посилання", "Подякувати автору печивом")


class EggsMixin:
    # ------------------------------------------------------------------ підказки
    def _tip(self, w, key):
        """підказка при наведенні: key - ключ перекладу або функція, що повертає готовий текст"""
        Tip(w, key if callable(key) else (lambda: tr(key)))
        return w

    def _install_tips(self):
        from .. import meta
        for attr, key in TIPS.items():
            w = getattr(self, attr, None)
            if w is None or (attr == "about_lbl" and not meta.links()):
                continue
            self._tip(w, key)
        self._tip(self.btn_play, lambda: tr(PLAY_TIPS[1] if self.export.running else PLAY_TIPS[0]))
        self._tip(self.stat_lbl, lambda: self._stat_full)
        for k, key in SIZE_TIPS.items():
            if k in self.size_btns:
                self._tip(self.size_btns[k], key)
        self._tip(self.btn_lock, lambda: tr(LOCK_TIP_ON if getattr(self, "locked", False) else LOCK_TIP_OFF))
        self._tip(self.prog_lbl, lambda: (self._last_out or "") if getattr(self, "_prog_done", False) else "")
        if self.btn_donate is not None:
            self._tip(self.btn_donate, lambda: tr(DONATE_TIPS[1] if meta.DONATE else DONATE_TIPS[0]))

    # ------------------------------------------------------------------ пасхалки: стан і підключення
    def _init_eggs(self):
        self._egg_clicks = {"logo": eggs.Clicks(5, 1.4), "swatch": eggs.Clicks(7, 1.6), "cookie": eggs.Clicks(eggs.COOKIE_CLICKS, 1.6)}
        self._egg_job = QTimer(self)
        self._egg_job.timeout.connect(self._egg_tick)
        self._egg_t0 = None
        self._spin_frames, self._spin_key, self._spin_i = None, None, None
        self._spin_job = QTimer(self)
        self._spin_job.timeout.connect(self._spin_step)
        self._bub: Bubble | None = None
        self._bub_job = QTimer(self)
        self._bub_job.setSingleShot(True)
        self._bub_job.timeout.connect(self._bubble_close)
        self._last_quote = -1
        self._party_job = QTimer(self)
        self._party_job.setSingleShot(True)
        self._party_job.timeout.connect(lambda: self._party(*self._party_seq) if self._party_seq else None)
        self._party_orig, self._party_seq = "", None
        self._cookie_job = QTimer(self)
        self._cookie_job.setSingleShot(True)
        self._cookie_job.timeout.connect(self._cookie_rebake)
        self._crumb_job = QTimer(self)
        self._crumb_job.setSingleShot(True)
        self._crumb_t = 0.0
        self._crumb_job.timeout.connect(self._crumbs_tick)
        self.lbl_mascot.clicked = self._reel_click
        self.lbl_logo.clicked = self._logo_click
        self.phil_lbl.clicked = self._quote_click
        if self.btn_donate is not None:
            self.btn_donate.on_press = self._cookie_click

    # ------------------------------------------------------------------ хмаринка
    def _bubble(self, widget, text, ms=3600):
        """хмаринка з фразою під віджетом; сама зникає"""
        self._bubble_close()
        b = Bubble(text, 300)
        b.place_under(widget)
        b.show()
        self._bub = b
        self._bub_job.start(ms)

    def _bubble_close(self):
        self._bub_job.stop()
        if self._bub is not None:
            self._bub.close()
            self._bub.deleteLater()
            self._bub = None

    # ------------------------------------------------------------------ котушка
    def _reel_click(self):
        """котушка в шапці: крутиться й каже фразу"""
        self._spin_reel()
        self._last_quote = eggs.pick_quote(self._last_quote)
        self._bubble(self.lbl_mascot, tr(eggs.QUOTES[self._last_quote]))

    def _spin_reel(self):
        """один оберт котушки (кадри повернутої картинки; будуються при першому разі й при зміні теми)"""
        if self._spin_job.isActive():
            return
        key = (THEME.bg.name(), THEME.fg.name())
        if self._spin_key != key:
            self._spin_frames = [assets.img("mascot_w", rot=a) for a in eggs.spin_angles()[:-1]]
            self._spin_key = key
        frames = self._spin_frames
        if not frames or any(f is None for f in frames):
            return
        self._spin_i = 0
        self._spin_job.start(max(10, int(eggs.SPIN_MS / len(frames))))
        self._spin_step()

    def _spin_step(self):
        frames, i = self._spin_frames, self._spin_i
        if i is None or not frames or i >= len(frames):
            self._spin_job.stop()
            self._spin_i = None
            self.lbl_mascot.show_image(None)             # назад до звичайної котушки
            return
        self.lbl_mascot.show_image(frames[i])
        self._spin_i = i + 1

    # ------------------------------------------------------------------ лого, цитата, розфокус
    def _logo_click(self):
        """лого ×5: картинка розфокусовується під слова філософії (без картинки - котушка каже фразу)"""
        if self._egg_clicks["logo"].hit():
            if self.surface.mask is not None:
                self._egg_defocus()
            else:
                self._reel_click()

    def _quote_click(self):
        """клік по цитаті в футері: нагору до картинки - і вона розфокусовується під ці слова (щойно перехід доїде)"""
        st = self.pstack
        if st.tcur > 0:
            st.go(0, done=self._egg_defocus)
        else:
            self._egg_defocus()

    def _egg_defocus(self):
        if self._egg_t0 is not None or self.surface.mask is None:
            return
        self._egg_t0 = time.perf_counter()
        self._egg_job.start(33)
        self._egg_tick()

    def _egg_tick(self):
        if self._egg_t0 is None:
            self._egg_job.stop()
            return
        t = time.perf_counter() - self._egg_t0
        if t >= eggs.DEFOCUS_TOTAL:
            self._egg_job.stop()
            self._egg_t0 = None
            self.surface.set_defocus(0.0, None)
            return
        self.surface.set_defocus(eggs.defocus_curve(t), self._egg_t0)

    # ------------------------------------------------------------------ вечірка палітр
    def _swatch_click(self):
        """квадратик палітри ×7: вечірка - вікно перебирає всі палітри й повертається до вашої"""
        if self._egg_clicks["swatch"].hit() and not self.export.running and not self._party_job.isActive() and self._party_seq is None:
            cur = self._party_orig = self.pal_name
            self._party(eggs.party_order(palette.NAMES, cur), cur, 0)

    def _party_show(self, name):
        self.pal_name = name
        self._apply_theme()
        self.pal_lbl.config(text=tr(name))

    def _party(self, order, orig, i):
        if i < len(order):
            self._party_seq = (order, orig, i + 1)
            self._party_show(order[i])
            self._party_job.start(170)
            return
        self._party_seq = None
        if self.pal_name != orig:
            self._party_show(orig)

    def _party_stop(self):
        """закрити вікно посеред вечірки - зберігається вибір користувача, а не випадкова палітра"""
        if self._party_seq is None:
            return
        self._party_job.stop()
        self._party_seq = None
        self.pal_name = self._party_orig

    # ------------------------------------------------------------------ печиво
    def _cookie_click(self):
        """печиво в футері (поки кнопка донату вимкнена): кліки відкушують його, сьомий - з'їдає й розсипає крихти"""
        b, ck = self.btn_donate, self._egg_clicks["cookie"]
        if b is None or b.enabled or b.bites >= eggs.EATEN:
            return
        self._cookie_job.stop()
        if ck.hit():
            b.bites, b.crumbs = eggs.EATEN, eggs.crumbs_burst()
            self._redraw_cookie()
            self._bubble(b, tr("Ням!"), 1500)
            self._crumb_t = time.perf_counter()
            self._crumbs_tick()
            self._cookie_job.start(2600)
        else:
            b.bites = ck.count
            self._redraw_cookie()
            self._cookie_job.start(int(ck.gap * 1000) + 120)

    def _redraw_cookie(self):
        self.btn_donate.draw()
        self.btn_donate.update()

    def _cookie_rebake(self):
        """кліки припинились (або печиво давно з'їдене): нове печиво"""
        self._egg_clicks["cookie"].reset()
        b = self.btn_donate
        if b is not None:
            b.bites, b.crumbs = 0, []
            self._redraw_cookie()

    def _crumbs_tick(self):
        b = self.btn_donate
        if b is None or not b.crumbs:
            return
        now = time.perf_counter()
        b.crumbs = eggs.crumbs_step(b.crumbs, min(0.05, now - self._crumb_t), floor=b._bh / 2)
        self._crumb_t = now
        self._redraw_cookie()
        if b.crumbs:
            self._crumb_job.start(30)
