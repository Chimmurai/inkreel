import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benvid.cli import main  # noqa: E402


def test_cli_photo_and_log():
    with tempfile.TemporaryDirectory() as d:
        img = (np.random.default_rng(0).random((120, 200, 3)) * 255).astype(np.uint8)
        cv2.circle(img, (100, 60), 40, (255, 255, 255), -1)
        a, o = os.path.join(d, "a.png"), os.path.join(d, "o.png")
        cv2.imwrite(a, img)
        main([a, o, "--size", "144p"])
        out = cv2.imread(o, 0)
        assert out.shape[0] == 144 and set(np.unique(out)) <= {0, 255}
    from benvid import log
    assert log.get() is log.get() and os.path.isdir(log.app_dir())


def test_cli_ikr_roundtrip():
    """відео -> .ikr (власний формат) -> відео: .ikr приймається і як вхід, і як вихід"""
    import pytest
    from benvid import bake
    from benvid.video import find_ffmpeg
    if find_ffmpeg() is None:
        pytest.skip("потрібен ffmpeg")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "v.mp4")
        vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 24, (160, 96))
        for i in range(8):
            f = np.full((96, 160, 3), 40, np.uint8)
            cv2.circle(f, (10 + i * 12, 48), 20, (230, 230, 230), -1)
            vw.write(f)
        vw.release()
        ikr, out = os.path.join(d, "v.ikr"), os.path.join(d, "o.mp4")
        main([src, ikr, "--method", "3", "--size", "Як у джерела", "--workers", "1"])
        rd = bake.open_baked(ikr)
        assert (rd.w, rd.h, rd.frames) == (160, 96, 8) and rd.meta["params"]["method"] == 3
        getattr(rd, "close", lambda: None)()                                  # Windows не дає видалити теку, поки файл відкритий
        main([ikr, out, "--quality", "Мала", "--workers", "1"])
        cap = cv2.VideoCapture(out)
        try:
            assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
        finally:
            cap.release()


def test_export_leaves_no_open_decoders_on_the_source(tmp_path):
    """Windows не дає видалити/перейменувати відкритий файл: після експорту жоден потік не тримає декодер джерела"""
    import pytest
    from benvid import video
    if video.find_ffmpeg() is None:
        pytest.skip("потрібен ffmpeg")
    src = str(tmp_path / "v.mp4")
    vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 24, (160, 96))
    for i in range(8):
        vw.write(np.full((96, 160, 3), 40 + i * 20, np.uint8))
    vw.release()
    main([src, str(tmp_path / "o.mp4"), "--method", "3", "--size", "Як у джерела", "--workers", "1"])
    assert all(src not in d for d in video._CAPS._all)
    os.remove(src)
