"""Заміряє швидкість обробки: python tools/bench.py [відео] — к/с на різних розмірах і кількості процесів."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    import cv2
    import numpy as np
    from benvid.core import Params
    from benvid.player import Player
    from benvid.video import default_workers

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        import tempfile
        path = os.path.join(tempfile.gettempdir(), "inkreel_bench.mp4")
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (1280, 720))
        rng = np.random.RandomState(0)
        base = cv2.GaussianBlur((rng.rand(720, 1280, 3) * 255).astype(np.uint8), (0, 0), 6)
        for i in range(240):
            w.write(np.roll(base, i * 3, axis=1))
        w.release()
    print(f"процесорів: {os.cpu_count()}, за замовчуванням воркерів: {default_workers()}")
    for key in ("144p", "360p", "720p", "1080p"):
        for wk in sorted({1, max(1, default_workers() // 2), default_workers()}):
            p = Player(wk)
            p.warm()
            time.sleep(3)
            p.open(path)
            p.set_size(key)
            t0 = time.time()
            while len(p.buffered_ranges()) == 0 or p.buffered_ranges()[0][1] < 100:
                time.sleep(0.05)
                if time.time() - t0 > 60:
                    break
            dt = time.time() - t0
            n = p.buffered_ranges()[0][1] + 1 if p.buffered_ranges() else 0
            print(f"  {key:>6}  воркерів {wk:>2}:  {n / dt:6.1f} к/с")
            p.close()


if __name__ == "__main__":
    main()
