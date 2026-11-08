"""Діагностика швидкості: експорт окремих шматків моделі у fp16 ONNX, щоб зміряти на DirectML, де йде час.
python tools/bisect_export.py [--preset fastl] [--dir .]   ->   bis_<шматок>.onnx"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_ink as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fastl")
    ap.add_argument("--dir", default=".")
    ap.add_argument("--round", type=int, default=1, help="1 - великі шматки; 2 - дрібніше всередині trunk")
    a = ap.parse_args()
    m = T.build_model(a.preset).eval().float().requires_grad_(False)
    c1, cf, hh = m.cfg["c"][0], m.cfg["cf"], m.cfg["hh"]
    act = lambda t: F.leaky_relu(t, 0.05)  # noqa: E731
    H, W = 128, 192
    r = lambda *s: torch.rand(*s)  # noqa: E731
    x, nz, full, half = r(1, 3, H, W), r(1, 4, H, W), r(1, cf, H, W), r(1, c1, H // 2, W // 2)
    gg, bb = r(1, hh, 1, 1), r(1, hh, 1, 1)

    class Stem(nn.Module):
        def forward(s, x, nz):
            return act(m.stem(torch.cat([x * 2 - 1, nz], 1)))

    class ConvT(nn.Module):
        def forward(s, u):
            return m.o(u)

    class H1(nn.Module):
        def forward(s, full, x, nz):
            return act(m.h1(torch.cat([full, x * 2 - 1, nz], 1)))

    class HeadNoFilm(nn.Module):
        def forward(s, full, x, nz):
            h = act(m.h1(torch.cat([full, x * 2 - 1, nz], 1)))
            return m.h3(act(m.h2(h)))

    class Head(nn.Module):
        def forward(s, full, x, nz, g, b):
            h = act(m.h2(act(m.h1(torch.cat([full, x * 2 - 1, nz], 1)))))
            return m.h3(torch.addcmul(b, h, g))

    class Trunk(nn.Module):                                  # від виходу stem до входу ConvTranspose (усе на 1/2 і менше)
        def forward(s, a):
            fl = m.fl
            gs = [(torch.ones(1, c, 1, 1), torch.zeros(1, c, 1, 1)) for c in fl]
            f1 = m.r1(a)
            f2 = m.r2(act(m.d1(f1)))
            f3 = m.r3b(m.r3a(act(m.d2(f2))))
            f4 = m.r4b(m.r4a(act(m.d3(f3)), gs[0]), gs[0])
            u = m.q3(act(m.u3(F.interpolate(f4, scale_factor=2.0, mode="bilinear", align_corners=False))) + f3, gs[1])
            u = m.q2(act(m.u2(F.interpolate(u, scale_factor=2.0, mode="bilinear", align_corners=False))) + f2, gs[2])
            return m.q1(act(m.u1(F.interpolate(u, scale_factor=2.0, mode="bilinear", align_corners=False))) + f1, gs[3])

    class Pre(nn.Module):                                    # лише x*2-1 і cat на повному розмірі
        def forward(s, x, nz):
            return torch.cat([x * 2 - 1, nz], 1)

    fx, hx = {2: "h", 3: "w"}, {2: "h2", 3: "w2"}
    jobs = {
        "pre": (Pre(), (x, nz), ["image", "noise"], {"image": fx, "noise": fx}),
        "stem": (Stem(), (x, nz), ["image", "noise"], {"image": fx, "noise": fx}),
        "trunk": (Trunk(), (r(1, c1, H // 2, W // 2),), ["a"], {"a": hx}),
        "convt": (ConvT(), (half,), ["u"], {"u": hx}),
        "h1": (H1(), (full, x, nz), ["full", "image", "noise"], {"full": fx, "image": fx, "noise": fx}),
        "head_nofilm": (HeadNoFilm(), (full, x, nz), ["full", "image", "noise"], {"full": fx, "image": fx, "noise": fx}),
        "head": (Head(), (full, x, nz, gg, bb), ["full", "image", "noise", "g", "b"], {"full": fx, "image": fx, "noise": fx}),
    }
    if a.round == 2:
        up = lambda t: F.interpolate(t, scale_factor=2.0, mode="bilinear", align_corners=False)  # noqa: E731
        fl = m.fl
        one = lambda c: (torch.ones(1, c, 1, 1), torch.zeros(1, c, 1, 1))  # noqa: E731

        def mk(fn):
            class M(nn.Module):
                def forward(s, *t):
                    return fn(*t)
            return M()

        c = m.cfg["c"]
        d = lambda k: {2: f"h{k}", 3: f"w{k}"}  # noqa: E731
        z = lambda ch, k: r(1, ch, H // k, W // k)  # noqa: E731
        jobs = {
            "r1": (mk(lambda t: m.r1(t)), (z(c[0], 2),), ["a"], {"a": d(2)}),
            "d1": (mk(lambda t: act(m.d1(t))), (z(c[0], 2),), ["a"], {"a": d(2)}),
            "r2": (mk(lambda t: m.r2(t)), (z(c[1], 4),), ["a"], {"a": d(4)}),
            "d2": (mk(lambda t: act(m.d2(t))), (z(c[1], 4),), ["a"], {"a": d(4)}),
            "r3": (mk(lambda t: m.r3b(m.r3a(t))), (z(c[2], 8),), ["a"], {"a": d(8)}),
            "d3r4": (mk(lambda t: m.r4b(m.r4a(act(m.d3(t)), one(fl[0])), one(fl[0]))), (z(c[2], 8),), ["a"], {"a": d(8)}),
            "up3": (mk(lambda t, sk: m.q3(act(m.u3(up(t))) + sk, one(fl[1]))), (z(c[3], 16), z(c[2], 8)), ["a", "skip"], {"a": d(16), "skip": d(8)}),
            "up2": (mk(lambda t, sk: m.q2(act(m.u2(up(t))) + sk, one(fl[2]))), (z(c[2], 8), z(c[1], 4)), ["a", "skip"], {"a": d(8), "skip": d(4)}),
            "up1": (mk(lambda t, sk: m.q1(act(m.u1(up(t))) + sk, one(fl[3]))), (z(c[1], 4), z(c[0], 2)), ["a", "skip"], {"a": d(4), "skip": d(2)}),
            "resize1": (mk(lambda t: up(t)), (z(c[1], 4),), ["a"], {"a": d(4)}),
            "u1conv": (mk(lambda t: act(m.u1(t))), (z(c[1], 2),), ["a"], {"a": d(2)}),
        }
    for name, (mod, args, names, dyn) in jobs.items():
        path = os.path.join(a.dir, f"bis_{name}.onnx")
        with torch.no_grad():
            torch.onnx.export(mod, args, path, input_names=names, output_names=["y"], dynamic_axes=dyn, opset_version=17, dynamo=False)
        T._to_fp16(path)


if __name__ == "__main__":
    main()
