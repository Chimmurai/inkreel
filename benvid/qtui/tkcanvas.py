"""Полотно в стилі Tk поверх QPainter: ті самі виклики `create_rectangle/line/oval/polygon/text/image/arc`, `delete`, `bbox`, `tag_lower`,
якими малювало колишнє вікно на Tk (видалене у 1.1.0). Мета - вигляд 1:1: код малювання кнопок, повзунків, смужок переноситься майже дослівно,
тож геометрія (контур по x0 і x1 включно, пунктири, товщини, центрування тексту за якорем) збігається з Tk до пікселя. Збіг перевірено попіксельно проти Tk.

Малюємо без згладжування (як Tk). Список елементів зберігається й перемальовується в paintEvent (один прохід), тому віджет можна
перемальовувати хоч щокадру без перебудови дерева віджетів."""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

try:
    from .gdi_ovals import OVALS
except Exception:  # noqa: BLE001
    OVALS = {}

_ANCHOR = {"nw": (0, 0), "n": (.5, 0), "ne": (1, 0), "w": (0, .5), "center": (.5, .5), "e": (1, .5), "sw": (0, 1), "s": (.5, 1), "se": (1, 1)}


def _col(c):
    if c in (None, "", "none"):
        return None
    return QColor(c)


@dataclass
class _Item:
    id: int
    kind: str
    coords: tuple
    opt: dict
    tags: set = field(default_factory=set)


class TkCanvas(QWidget):
    """віджет-полотно з API tk.Canvas (лише те, що використовує програма)"""

    def __init__(self, parent=None, bg="#000000"):
        super().__init__(parent)
        self.bg = bg
        self._items: list[_Item] = []
        self._next = 1
        self._cache, self._cache_key = None, None             # кеш-картинка намальованих елементів (див. paintEvent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    # ----------------------------------------------------------------------- API елементів
    def _add(self, kind, coords, opt, tags):
        it = _Item(self._next, kind, tuple(float(c) for c in coords), opt, set(tags.split() if isinstance(tags, str) else (tags or ())))
        self._next += 1
        self._items.append(it)
        self._cache = None
        return it.id

    def delete(self, *what):
        self._cache = None
        if not what or "all" in what:
            self._items = []
        else:
            keep = []
            for it in self._items:
                if it.id in what or (it.tags & {w for w in what if isinstance(w, str)}):
                    continue
                keep.append(it)
            self._items = keep

    def create_rectangle(self, x0, y0, x1, y1, fill="", outline="#000000", width=1, dash=None, tags=()):
        return self._add("rect", (x0, y0, x1, y1), dict(fill=fill, outline=outline, width=width, dash=dash), tags)

    def create_line(self, *coords, fill="#000000", width=1, dash=None, dashoffset=0, tags=()):
        return self._add("line", coords, dict(fill=fill, width=width, dash=dash, dashoffset=dashoffset), tags)

    def create_oval(self, x0, y0, x1, y1, fill="", outline="#000000", width=1, tags=()):
        return self._add("oval", (x0, y0, x1, y1), dict(fill=fill, outline=outline, width=width), tags)

    def create_polygon(self, *coords, fill="#000000", outline="", width=1, tags=()):
        return self._add("poly", coords, dict(fill=fill, outline=outline, width=width), tags)

    def create_arc(self, x0, y0, x1, y1, start=0, extent=90, style="pieslice", outline="#000000", fill="", width=1, tags=()):
        return self._add("arc", (x0, y0, x1, y1), dict(start=start, extent=extent, style=style, outline=outline, fill=fill, width=width), tags)

    def create_text(self, x, y, text="", fill="#000000", font=None, anchor="center", width=0, justify="left", tags=()):
        return self._add("text", (x, y), dict(text=text, fill=fill, font=font, anchor=anchor, width=width, justify=justify), tags)

    def create_image(self, x, y, image=None, anchor="center", tags=()):
        return self._add("image", (x, y), dict(image=image, anchor=anchor), tags)

    def tag_lower(self, item, below=None):
        its = [i for i in self._items if i.id == item]
        if not its:
            return
        self._items.remove(its[0])
        idx = next((k for k, i in enumerate(self._items) if i.id == below), 0)
        self._items.insert(idx, its[0])
        self._cache = None

    def bbox(self, item):
        for it in self._items:
            if it.id == item:
                if it.kind == "text":
                    r = self._text_rect(it)
                    return (int(r.left()), int(r.top()), int(r.right()) + 1, int(r.bottom()) + 1)
                xs, ys = it.coords[0::2], it.coords[1::2]
                return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        return None

    def winfo_width(self):
        return self.width()

    def winfo_height(self):
        return self.height()

    def place(self, x=0, y=0, width=None, height=None):
        self.setGeometry(int(x), int(y), int(width if width is not None else self.width()), int(height if height is not None else self.height()))
        self.show()

    def place_forget(self):
        self.hide()

    # ----------------------------------------------------------------------- малювання
    @staticmethod
    def _pen(color, width=1, dash=None, offset=0, phase=0.0):
        c = _col(color)
        if c is None:
            return QPen(Qt.NoPen)
        p = QPen(c, float(width))
        p.setCapStyle(Qt.FlatCap)
        p.setJoinStyle(Qt.MiterJoin)
        if dash:
            # Tk на Windows (GDI) малює будь-який dash однаково: 3 пікселі штриха й 3 проміжку при товщині 1 і 2+2 при товщині 2 (виміряно проти Tk);
            # одиниці пунктиру Qt - у товщинах пера, тож для товщини w потрібно (n/w, n/w)
            n = 3.0 if width <= 1 else 2.0
            p.setDashPattern([n / float(width), n / float(width)])
            p.setDashOffset(float(offset) + phase)
        return p

    def _font(self, spec) -> QFont:
        from .theme import tk_font
        return tk_font(spec)

    def _text_lines(self, it):
        from .labels import wrap_text
        fm = QFontMetrics(self._font(it.opt["font"]))
        wd = int(it.opt.get("width") or 0)
        return (wrap_text(it.opt["text"], wd, fm) if wd else str(it.opt["text"]).split("\n")), fm

    def _text_rect(self, it) -> QRectF:
        o = it.opt
        lines, fm = self._text_lines(it)
        w = max((fm.horizontalAdvance(s) for s in lines), default=0)
        h = fm.height() * len(lines)
        fx, fy = _ANCHOR.get(o["anchor"], (.5, .5))
        x, y = it.coords
        return QRectF(round(x) - int(fx * w), round(y) - int(fy * h), w, h)      # Tk центрує цілочисельно (ширина/2 з відкиданням дробу)

    def draw_items(self, p):
        """намалювати елементи на чужому QPainter (напис поверх картинки, без власного тла)"""
        for it in self._items:
            getattr(self, "_draw_" + it.kind)(p, it)

    def paintEvent(self, e):
        """елементи малюються в кеш-картинку один раз (після змін або зміни розміру), далі кожне перемальовування - один blit: коли вікно тягнуть за край, сорок
        віджетів лише зсуваються, і їхній Python-код малювання (текст, прямокутники) на кожен крок не виконується. Піксель у піксель те саме, що малювання напряму:
        картинка непрозора (RGB32), як і буфер самого вікна, тож згладжування тексту таке саме"""
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        dpr = self.devicePixelRatioF()
        key = (w, h, dpr, self.bg)
        if self._cache is None or self._cache_key != key:
            img = QImage(max(1, int(w * dpr)), max(1, int(h * dpr)), QImage.Format_RGB32)
            img.setDevicePixelRatio(dpr)
            q = QPainter(img)
            q.setRenderHint(QPainter.Antialiasing, False)
            q.setRenderHint(QPainter.TextAntialiasing, True)
            q.fillRect(QRect(0, 0, w, h), QColor(self.bg))
            self.draw_items(q)
            q.end()
            self._cache, self._cache_key = img, key
        p = QPainter(self)
        p.drawImage(0, 0, self._cache)

    def _draw_rect(self, p, it):
        x0, y0, x1, y1 = it.coords
        o = it.opt
        fill, outline, w = _col(o["fill"]), _col(o["outline"]), max(1, int(o["width"]))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        if fill is not None:
            # Tk: заливка займає x0..x1-1 (праве й нижнє ребро - лише контур); без контуру заливка така сама
            p.fillRect(QRect(int(x0), int(y0), int(x1 - x0) + (1 if outline is not None else 0), int(y1 - y0) + (1 if outline is not None else 0)), fill)
        if outline is not None:
            p.setBrush(Qt.NoBrush)
            p.setPen(self._pen(o["outline"], w, o["dash"]))
            h = w / 2.0
            if o["dash"]:
                # Tk (GDI) обводить пунктирний прямокутник суцільним контуром, що починається в ПРАВОМУ ВЕРХНЬОМУ куті й іде проти годинникової стрілки
                # (знайдено перебором усіх кутів і напрямків на трьох розмірах: збіг до пікселя); фаза штриха нульова
                a, b, c2, d2 = x0 + h, y0 + h, x1 + h, y1 + h
                p.drawPolyline(QPolygonF([QPointF(c2, b), QPointF(a, b), QPointF(a, d2), QPointF(c2, d2), QPointF(c2, b)]))
            else:
                p.drawRect(QRectF(x0 + h, y0 + h, x1 - x0 + 1 - w, y1 - y0 + 1 - w))

    def _draw_line(self, p, it):
        o = it.opt
        pts = it.coords
        if len(pts) < 4:
            return
        w = max(1, int(o["width"]))
        p.setPen(self._pen(o["fill"], w, o["dash"], o["dashoffset"]))
        s = .5 if w % 2 else 0.0                            # лінія непарної товщини центрується на пікселі, парної - на межі пікселів (як у Tk)
        if len(pts) == 4 and pts[1] == pts[3]:              # горизонтальна: уздовж лінії початок рівно на піщелі x0 (штрих Tk починається з першого пікселя)
            q = [QPointF(pts[0], pts[1] + s), QPointF(pts[2] + 1, pts[3] + s)]
        elif len(pts) == 4 and pts[0] == pts[2]:
            q = [QPointF(pts[0] + s, pts[1]), QPointF(pts[2] + s, pts[3] + 1)]
        else:
            q = [QPointF(pts[i] + s, pts[i + 1] + s) for i in range(0, len(pts) - 1, 2)]
        p.drawPolyline(QPolygonF(q))

    def _draw_oval(self, p, it):
        x0, y0, x1, y1 = it.coords
        o = it.opt
        w = max(1, int(o["width"]))
        fill = _col(o["fill"])
        x0, y0, x1, y1 = round(x0), round(y0), round(x1), round(y1)         # Tk округляє дробові координати
        d = int(x1 - x0)
        if d == int(y1 - y0) and (d, min(w, 2)) in OVALS:
            # коло точно як у GDI: форма знята з Tk (таблиця gdi_ovals.py); Qt малює Ellipse «круглішим», і діаметр у кілька пікселів виходить іншим
            shape = OVALS[(d, min(w, 2))]
            outline = _col(o["outline"])
            for variant, color in (("all", fill if fill is not None else None), ("ring", outline)):
                if color is None:
                    continue
                if variant == "all" and fill is None:
                    continue
                dx, dy, _w, rows = shape[variant]
                for r, bits in enumerate(rows):
                    c = 0
                    while bits >> c:
                        if bits >> c & 1:
                            e = c
                            while bits >> (e + 1) & 1:
                                e += 1
                            p.fillRect(QRect(int(x0) + dx + c, int(y0) + dy + r, e - c + 1, 1), color)
                            c = e + 1
                        else:
                            c += 1
            return
        p.setBrush(QBrush(fill) if fill is not None else Qt.NoBrush)
        p.setPen(self._pen(o["outline"], w))
        h = w / 2.0
        p.drawEllipse(QRectF(x0 + h, y0 + h, x1 - x0 + 1 - w, y1 - y0 + 1 - w))

    def _draw_poly(self, p, it):
        o = it.opt
        pts = it.coords
        fill, outline = _col(o["fill"]), _col(o["outline"])
        p.setBrush(QBrush(fill) if fill is not None else Qt.NoBrush)
        p.setPen(self._pen(o["outline"], max(1, int(o["width"]))) if outline is not None else QPen(Qt.NoPen))
        p.drawPolygon(QPolygonF([QPointF(pts[i] + .5, pts[i + 1] + .5) for i in range(0, len(pts) - 1, 2)]))

    def _draw_arc(self, p, it):
        x0, y0, x1, y1 = it.coords
        o = it.opt
        p.setBrush(Qt.NoBrush)
        p.setPen(self._pen(o["outline"], max(1, int(o["width"]))))
        p.setRenderHint(QPainter.Antialiasing, True)
        p.drawArc(QRectF(x0 + .5, y0 + .5, x1 - x0, y1 - y0), int(o["start"] * 16), int(o["extent"] * 16))
        p.setRenderHint(QPainter.Antialiasing, False)

    def _draw_text(self, p, it):
        o = it.opt
        c = _col(o["fill"])
        if c is None:
            return
        r = self._text_rect(it)
        p.setPen(c)
        p.setFont(self._font(o["font"]))
        lines, fm = self._text_lines(it)
        for k, s in enumerate(lines):
            lw = fm.horizontalAdvance(s)
            x = r.left() + ((r.width() - lw) // 2 if o["justify"] == "center" else (r.width() - lw) if o["justify"] == "right" else 0)
            p.drawText(QRectF(x, r.top() + k * fm.height(), lw + 2, fm.height()), Qt.AlignLeft | Qt.AlignTop | Qt.TextDontClip, s)

    def _draw_image(self, p, it):
        img = it.opt["image"]
        if img is None:
            return
        x, y = it.coords
        fx, fy = _ANCHOR.get(it.opt["anchor"], (.5, .5))
        p.drawImage(int(x - fx * img.width()), int(y - fy * img.height()), img)
