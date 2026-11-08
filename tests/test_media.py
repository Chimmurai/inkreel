import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image
from benvid.core import Params
from benvid.media import export_gif_source, export_still, kind_of, open_source


def main():
    d = tempfile.mkdtemp()
    rng = np.random.RandomState(1)
    fr = [Image.fromarray((rng.rand(60, 80, 3) * 255).astype("uint8")) for _ in range(5)]
    g = os.path.join(d, "a.gif")
    fr[0].save(g, save_all=True, append_images=fr[1:], duration=[100, 200, 100, 300, 100], loop=0)
    p = os.path.join(d, "a.png")
    fr[0].save(p)
    assert kind_of(g) == "gif" and kind_of(p) == "image" and kind_of("x.mp4") == "video"
    s = open_source(g)
    assert s.info["frames"] == 5 and abs(s.info["duration"] - 0.8) < 1e-6 and list(np.round(s.info["starts"], 2)) == [0, .1, .3, .4, .7]
    r = export_still(p, os.path.join(d, "o.png"), Params(), "64x")
    assert Image.open(os.path.join(d, "o.png")).mode == "1" and r["size"] == (64, 48)
    r = export_gif_source(g, os.path.join(d, "o.gif"), Params(), "GIF", "64x", workers=1)
    o = Image.open(os.path.join(d, "o.gif"))
    assert o.n_frames == 5 and o.size == (64, 48)
    print("ok")


if __name__ == "__main__":
    main()
