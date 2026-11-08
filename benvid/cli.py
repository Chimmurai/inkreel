"""CLI:  inkreel in.mp4 out.mp4 [параметри]  |  фото: inkreel in.jpg out.png  |  GIF: inkreel in.gif out.gif
(без аргументів - програма з вікном)"""
from __future__ import annotations

import argparse
import os
import sys

from .core import Params
from .media import IMAGE_EXT
from .progress import ConsoleBar
from .video import FORMATS, QUALITY, SIZES, default_workers, export_video, video_info

EXT2FMT = {ext: name for name, (ext, codec) in FORMATS.items() if ext}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="inkreel", description="InkReel — автоматична чорно-біла обробка відео")
    ap.add_argument("input", help="вхідне відео (або .ikr), GIF або фото")
    ap.add_argument("output", help="вихід: .mp4/.mkv/.mov/.avi/.webm/.gif, .ikr (власний формат без втрат: кадри готові, відкривається в InkReel у режимі замка) або тека (PNG-кадри)")
    ap.add_argument("--size", default="720p",
                    help="пресет (" + ", ".join(SIZES) + ") або свій розмір, напр. 320x180, 64x чи x240")
    ap.add_argument("--quality", default="Висока", choices=list(QUALITY))
    ap.add_argument("--hw", "--nvenc", dest="hw", action="store_true",
                    help="кодувати H.264 на відеокарті (NVIDIA NVENC / AMD AMF / Intel Quick Sync, якщо працює)")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--limit", type=int, help="обробити лише перші N кадрів")
    ap.add_argument("--fps", type=float, help="ліміт кадрів на секунду для відео (напр. 30): кадри беруться за часом, тривалість і звук лишаються; "
                                              "вищий за початкову частоту нічого не змінює")
    ap.add_argument("--period", type=float, help="крок штрихів, px на висоті 720 (4)")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--no-auto", action="store_true", help="вимкнути автоконтраст і авто-CLAHE")
    ap.add_argument("--palette", default="Класика", help="кольорова палітра: " + ", ".join(__import__("benvid.palette", fromlist=["x"]).NAMES)
                    + "; з суфіксом ~ (Синька~) - туш і папір навпаки")
    ap.add_argument("--method", type=int, choices=range(1, 7), default=4,
                    help="метод: 1 ескіз, 2 контур, 3 півтон, 4 малюнок (типово), 5 гравюра, 6 IRNM1 (InkreelNeuralModel v1, окрема гілка: навчена мережа на відеокарті, "
                         "швидка; без неї чи без моделі рахує як 5)")
    ap.add_argument("--symbols", action="store_true",
                    help="режим «Символи»: кадр складається зі знаків шрифту (ті самі п'ять рівнів); для фото вихід може бути .txt")
    ap.add_argument("--density", type=int, choices=range(1, 4), default=3,
                    help="щільність знаків у режимі --symbols: клітинка 1 - 32×64 px, 2 - 16×32, 3 - 8×16 (типово)")
    ap.add_argument("--version", action="version", version="InkReel " + __import__("benvid").__version__)
    a = ap.parse_args(argv)

    if not os.path.isfile(a.input):
        ap.error(f"файл не знайдено: {a.input}")
    if os.path.abspath(a.input) == os.path.abspath(a.output):
        ap.error("вихід збігається з входом")
    a.workers = a.workers if a.workers and a.workers > 0 else default_workers()
    from .palette import INV, LEGACY, NAMES
    from .palette import key as pal_key
    if a.palette in LEGACY:                                    # стара назва («Ризограф», «Game Boy») -> нова
        a.palette = pal_key(*LEGACY[a.palette])
    if (a.palette[:-len(INV)] if a.palette.endswith(INV) else a.palette) not in NAMES:
        ap.error(f"невідома палітра: {a.palette}. Є: " + ", ".join(NAMES) + f" (суфікс {INV} = туш і папір навпаки)")
    if a.symbols and a.method > 5:
        ap.error("метод 6 (IRNM1) не поєднується з --symbols: у символах п'ять рівнів")
    p = Params()
    for k in ("period",):
        if getattr(a, k) is not None:
            setattr(p, k, getattr(a, k))
    p.invert = a.invert
    p.method = a.method
    p.glyph = a.symbols
    p.gdens = a.density
    p.auto = not a.no_auto
    ext = os.path.splitext(a.output)[1].lower()
    if ext == ".txt" and not (a.symbols and os.path.splitext(a.input)[1].lower() in IMAGE_EXT):
        ap.error("вихід .txt - лише для фото в режимі --symbols")
    if a.output.lower().endswith((".json.gz", ".json")):
        fmt = "JSON (стиснутий)"
    else:
        fmt = EXT2FMT.get(ext, "PNG-послідовність")
    from .media import export_gif_source, export_still, kind_of
    kind = kind_of(a.input)
    if kind == "image":
        r = export_still(a.input, a.output, p, a.size if a.size != "720p" else "Як у джерела", palette=a.palette)
        print(f"{a.output}: {r['size'][0]}x{r['size'][1]}, {r['bytes']/1e6:.2f} МБ")
        return
    if kind == "gif":
        bar = ConsoleBar("Обробка")
        gf = "GIF" if ext == ".gif" else "PNG-послідовність"
        r = export_gif_source(a.input, a.output, p, gf, a.size, a.workers, lambda f, l=None: bar(f), palette=a.palette)
        bar.close()
        print(f"{a.output}: {r['frames']} кадрів, {r['size'][0]}x{r['size'][1]}, {r['bytes']/1e6:.2f} МБ")
        return
    info = video_info(a.input)
    print(f"{a.input}: {info['width']}x{info['height']}, {info['fps']:.3f} к/с, {info['frames']} кадрів "
          f"-> {fmt}, {a.size}", file=sys.stderr)
    bar = ConsoleBar("Обробка")
    from .video import detect_hw_encoder
    r = export_video(a.input, a.output, p, fmt, a.size, a.quality, (detect_hw_encoder() or False) if a.hw else False, a.workers, a.limit,
                     progress=lambda f, l=None: bar(f), palette=a.palette, fps=a.fps)
    bar.close()
    print(f"{a.output}: {r['frames']} кадрів, {r['size'][0]}x{r['size'][1]}, {r['bytes']/1e6:.2f} МБ")


if __name__ == "__main__":
    main()
