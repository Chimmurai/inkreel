"""Рух вмісту між кадрами для «Ультри»: шум і штрихи їдуть разом із зонами, а не стоять на екрані.

Нерухоме порогове поле (синій шум зерна, плямки крапок, шум штрихів) добре на нерухомій картинці, але коли об'єкт чи текстура
їдуть, візерунок лишається на місці й повз нього «пливе» тон: око зачіпається за сам візерунок, а не за зображення. Тут:

  1. ПОТІК. Оптичний потік (DIS) між тоном попереднього й поточного кадру на сітці аналізу: для кожного пікселя - де він був
     у минулому кадрі. Довіряємо йому лише там, де є текстура й де кадр, зсунутий потоком, збігається з поточним; дірки
     (рівні ділянки, відкриті ділянки) заповнюються сусідніми значеннями в малому околі, далі - потік 0 (візерунок лишається
     на екрані). Малий рух (нижче шуму потоку) - це 0: на нерухомій сцені зсуву нема взагалі.
  2. ЗСУВ ВІЗЕРУНКА D. Для кожного пікселя - на скільки пікселів зсунуто координату в полі шуму: D_t(p) = b(p) + D_{t-1}(p + b(p)),
     де b - потік. Візерунок тоді їде разом із вмістом (прив'язаний до зони), а новий вміст бере візерунок із нульовим зсувом.
  3. РОЗСЛАБЛЕННЯ. Різні частини картинки їдуть по-різному (обертання, наближення, вода), тож зсув з часом розтягнув би візерунок.
     Тому D щокадру трохи вирівнюється із розмитою копією: де зона рухається цілою (панорама, об'єкт) - D майже однаковий, і
     розслаблення нічого не міняє; де рух різний - візерунок повільно ковзає, а не розтягується.
Усе рахується на малій сітці аналізу; у ядро малювання йде лише ця сітка зсувів (білінійно розгортається, як поля штрихів)."""
from __future__ import annotations

import cv2
import numpy as np

TA = dict(
    v0=0.10,        # потік нижче цього (px сітки аналізу за кадр) - шум оцінки, а не рух; до 2*v0 - плавно
    e0=0.004, e1=0.02,   # довіра за текстурою: градієнт тону (на px сітки) від e0 до e1
    r0=0.04, r1=0.12,    # довіра за збігом: залишок після зсуву потоком від r0 (повна) до r1 (нуль)
    m0=0.25, m1=0.55,    # потік береться там, де його залишок менший за залишок «стоїть на місці» на частку m0 (повна довіра - на m1)
    n0=0.004, n1=0.012,   # і де кадр узагалі помітно змінився (вище за шум: залишок «стоїть на місці» від n0, повна довіра - на n1)
    s0=0.08, s1=0.25,    # «стоїть на місці» впевнене там, де різниця кадрів відповідає зсуву менше за s0 px сітки (при s1 px - нуль)
    fill=4.0,       # радіус (px сітки на 720p) заповнення дірок сусідніми значеннями потоку
    relax=0.04, rs=3.0,   # частка вирівнювання зсуву за кадр і радіус розмиття (px сітки на 720p)
    pers=0.5,       # вага нового кадру в згладженому в часі потоці: випадковий (шум, пил) гаситься, сталий рух лишається
    med=5,          # медіана потоку (px сітки): дрібні плями хибного руху зникають, зони (великі) лишаються
    decay=0.03,     # де руху нема, зсув повертається до нуля цією часткою за кадр (візерунок знову стає нерухомим)
    preset="MEDIUM",   # якість оптичного потоку DIS: ULTRAFAST / FAST / MEDIUM
    off=0.5,        # якщо найбільший зсув менший за це (px екрана) - візерунок лишається на екрані, шлях без зсувів
    x_mean=0.05, x_big=0.2, x_frac=0.12,   # «рух пояснює різницю кадрів»: середній залишок, і частка пікселів із залишком вище x_big
)


def _sm(x, a, b):
    z = np.clip((x - a) / (b - a), 0.0, 1.0)
    return z * z * (3 - 2 * z)


def backward_flow(cur: np.ndarray, prev: np.ndarray, preset: str = "MEDIUM") -> np.ndarray:
    """(lh, lw, 2) float32: де піксель поточного кадру був у минулому (prev(p + f(p)) ~ cur(p)); тон 0..1 -> 8 біт"""
    a = np.clip(cur * 255.0 + 0.5, 0, 255).astype(np.uint8)
    b = np.clip(prev * 255.0 + 0.5, 0, 255).astype(np.uint8)
    mk = getattr(cv2, "DISOpticalFlow_create", None)
    f = None
    if mk is not None:
        try:
            f = mk(getattr(cv2, "DISOPTICAL_FLOW_PRESET_" + preset, cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)).calc(a, b, None)
        except Exception:  # noqa: BLE001
            f = None
    if f is None:
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 15, 3, 5, 1.2, 0)
    return np.ascontiguousarray(f, np.float32)


_GRID: dict = {}


def _grid(lw: int, lh: int):
    g = _GRID.get((lw, lh))
    if g is None:
        yy, xx = np.mgrid[0:lh, 0:lw].astype(np.float32)
        if len(_GRID) > 4:
            _GRID.clear()
        g = _GRID[(lw, lh)] = (xx, yy)
    return g


def analyze(cur, prev, ts, kk, bh=None, pr=TA):
    """потік з довірою між кадрами сітки аналізу -> (b (lh, lw, 2) у px сітки, згладжений потік bh для наступного кадру, пояснює: bool).
    b = 0 там, де довіри нема або рух малий. «Пояснює» - зсув кадру потоком збігається з поточним кадром майже всюди: тоді кадри
    різні лише через рух (швидка панорама), а не через монтажну склейку, і історію скидати не треба"""
    lh, lw = cur.shape
    xx, yy = _grid(lw, lh)
    f = backward_flow(cur, prev, pr["preset"])
    warped = cv2.remap(prev, xx + f[..., 0], yy + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rr = np.abs(cur - warped)
    expl = bool(float(rr.mean()) < pr["x_mean"] and float((rr > pr["x_big"]).mean()) < pr["x_frac"])
    res = cv2.GaussianBlur(rr, (0, 0), 1.5)                          # залишок, якщо піксель поїхав за потоком
    res0 = cv2.GaussianBlur(np.abs(cur - prev), (0, 0), 1.5)         # залишок, якщо він стоїть на місці
    gx = cv2.Sobel(ts, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    gy = cv2.Sobel(ts, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    en = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.5)
    tex = _sm(en, pr["e0"], pr["e1"])
    rel = (res0 - res) / (res0 + 1e-4)
    cm = (tex * (1.0 - _sm(res, pr["r0"], pr["r1"])) * _sm(rel, pr["m0"], pr["m1"]) * _sm(res0, pr["n0"], pr["n1"])).astype(np.float32)   # рух пояснює кадр помітно краще, ніж спокій
    cs = (tex * (1.0 - _sm(res0 / (0.5 * en + 0.004), pr["s0"], pr["s1"])) * (1.0 - cm)).astype(np.float32)           # спокій пояснює кадр (інтерфейс, нерухомий фон)
    s = max(1.0, pr["fill"] * kk)
    num = cv2.GaussianBlur(f * cm[..., None], (0, 0), s)
    den = cv2.GaussianBlur(cm + cs, (0, 0), s)
    b = num / np.maximum(den, 1e-6)[..., None] * _sm(den, 0.02, 0.2)[..., None]
    if pr["med"] >= 3:
        b = cv2.medianBlur(np.ascontiguousarray(b, np.float32).reshape(lh, lw, 2), 5 if pr["med"] >= 5 else 3)
    bh_new = b if bh is None else (pr["pers"] * b + (1.0 - pr["pers"]) * warp_grid(bh, b))
    mag = np.sqrt(bh_new[..., 0] ** 2 + bh_new[..., 1] ** 2)
    b = b * _sm(mag, pr["v0"], 2.0 * pr["v0"])[..., None]
    return np.ascontiguousarray(b, np.float32), np.ascontiguousarray(bh_new, np.float32), expl


def advance(D, b, scale, kk, pr=TA):
    """D_t(p) = b(p)*scale + D_{t-1}(p + b(p)), потім розслаблення. D (lh, lw, 2) float32 або None (зсуву ще нема);
    scale = ширина кадру / ширина сітки (D - у пікселях кадру). -> D новий або None (рухів ще не було)"""
    moving = bool(np.any(b))
    if D is None and not moving:
        return None
    lh, lw = b.shape[:2]
    xx, yy = _grid(lw, lh)
    if D is None:
        D = np.zeros((lh, lw, 2), np.float32)
    if moving:
        D = cv2.remap(D, xx + b[..., 0], yy + b[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) + b * np.float32(scale)
    if pr["decay"] > 0:                                                    # де руху нема, візерунок поволі повертається на екранне місце
        mv = (np.abs(b[..., 0]) + np.abs(b[..., 1]) > 0).astype(np.float32)
        mv = cv2.GaussianBlur(mv, (0, 0), max(2.0, 2.0 * pr["rs"] * kk)) if moving else np.zeros_like(mv)
        D = D * (1.0 - np.float32(pr["decay"]) * (1.0 - np.minimum(mv * 4.0, 1.0)))[..., None]
    if pr["relax"] > 0:
        Db = cv2.GaussianBlur(D, (0, 0), max(1.0, pr["rs"] * kk))
        D = D + np.float32(pr["relax"]) * (Db - D)
    return np.ascontiguousarray(D, np.float32)


def warp_grid(a, b):
    """поле на сітці аналізу (lh, lw, c) -> те саме, зсунуте разом із вмістом: значення з точки p + b(p) минулого кадру"""
    lh, lw = a.shape[:2]
    xx, yy = _grid(lw, lh)
    mx, my = xx + b[..., 0], yy + b[..., 1]
    if a.ndim == 2 or a.shape[2] <= 4:
        return cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.concatenate([cv2.remap(np.ascontiguousarray(a[..., i:i + 2]), mx, my, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE).reshape(lh, lw, -1) for i in range(0, a.shape[2], 2)], axis=2)


def active(D, pr=TA) -> bool:
    """чи є хоч де-небудь зсув, що змінює піксель (інакше шлях без зсувів - побітово як без руху)"""
    return D is not None and float(np.abs(D).max()) >= pr["off"]
