"""Діагностика штрихів студента: збіг з Ultra при зануленні кільцевого каналу й усього шуму, когерентність напрямів і довжина серій.
    python tools/diag_strokes.py weights/ink1 кадр1.png кадр2.png      (картинки: Ultra | студент | без кільцевого шуму -> reports/strokes)"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools')); sys.path.insert(0, ROOT)
os.environ["INKREEL_GPU"] = "off"
import numpy as np, cv2, torch
import train_ink as T
dev = torch.device("cpu")
model, ck = T.load_model(sys.argv[1], "ema", dev)
T.NOISE_MODE = ck["args"].get("noise", "all")
OUT = os.path.join(ROOT, "reports", "strokes"); os.makedirs(OUT, exist_ok=True)
ring = T.RingCache(os.path.join(OUT, "_ring"))
from benvid.core import Params, binarize

def orient_energy(m):
    """частка енергії смугового сигналу в домінантному напрямі (структурний тензор маски після згладжування)"""
    g = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    Jxx = cv2.GaussianBlur(gx*gx, (0,0), 6); Jyy = cv2.GaussianBlur(gy*gy, (0,0), 6); Jxy = cv2.GaussianBlur(gx*gy, (0,0), 6)
    tr = Jxx + Jyy + 1e-9
    coh = np.sqrt((Jxx-Jyy)**2 + 4*Jxy**2) / tr
    return float(coh.mean())

def lenrun(m):
    """середня довжина горизонтальних/вертикальних/діагональних серій однакового кольору (px): штрихи дають довгі серії"""
    k = []
    for a in (m, m.T):
        d = np.diff(a.astype(np.int8), axis=1) != 0
        k.append(a.shape[1] / (d.sum(1).mean() + 1))
    return float(np.mean(k))

for path in sys.argv[2:]:
    img = cv2.imread(path)
    h0 = 720
    img = cv2.resize(img, (int(round(img.shape[1]*h0/img.shape[0]/2)*2), h0), interpolation=cv2.INTER_AREA)[:, :960]
    h, w = img.shape[:2]
    ref = binarize(img, w, h, Params(method=5)).astype(np.uint8)
    res = {}
    for mode in ("all", "noring", "none"):
        T.NOISE_MODE = "all"
        orig = T.noise_window
        if mode == "noring":
            def nw(*a, **k):
                o = orig(*a, **k); o[3] = 0; return o
            T.noise_window = nw
        elif mode == "none":
            def nw(*a, **k):
                o = orig(*a, **k); o[:] = 0; return o
            T.noise_window = nw
        pred = T.run_frame(model, img, ring, dev)
        T.noise_window = orig
        pred = (pred > 0.5).astype(np.uint8) if pred.dtype != np.uint8 else pred
        res[mode] = pred
    print(os.path.basename(path), (h, w), "білих: ultra %.3f" % ref.mean())
    print("  Ultra : coh %.3f  серія %.2f px" % (orient_energy(ref), lenrun(ref)))
    for k, p in res.items():
        _t, _p = (torch.from_numpy(z.astype(np.float32))[None, None] for z in (ref, p))
        print("  %-7s: збіг %.3f  coh %.3f  серія %.2f px  білих %.3f  штрихи(втрата) %.3f" % (k, (p == ref).mean(), orient_energy(p), lenrun(p), p.mean(),
                                                                                      float(T.stroke_err(_p, _t, torch.ones_like(_t)))))
    cv2.imwrite(os.path.join(OUT, os.path.basename(path) + "_cmp.png"), np.hstack([img[..., ::1], *[np.repeat((x*255)[..., None], 3, 2) for x in (ref, res['all'], res['noring'])]]))
