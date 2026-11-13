import multiprocessing
import os
import sys


def _run():
    multiprocessing.freeze_support()          # обов'язково для .exe: дочірні процеси (spawn) стартують звідси
    from . import log
    log.install_hooks()
    log.enable_crash_trace()
    from . import procs
    procs.setup()
    a = sys.argv[1:]
    if a[:1] == ["--bench"]:                     # заміри швидкості (звіт bench_report.md); те саме, що python -m benvid.bench
        from .bench import main as bench_main
        bench_main(a[1:])
    elif a[:1] == ["--gpu-probe"]:                # перевірка відеокарти (окремий процес із тайм-аутом, див. gpu.py)
        from .gpu import run_probe
        run_probe()
    elif a[:1] == ["--quick"]:                   # правий клік по фото/GIF у Провіднику: швидке перетворення без вікна програми
        from .quick import main as quick_main
        quick_main(a[1:])
    elif a[:1] in (["--register-menu"], ["--unregister-menu"]):   # пункти меню правої кнопки без інсталятора (benvid/shellmenu.py)
        from .shellmenu import main as menu_main
        sys.exit(menu_main(a))
    elif a[:1] == ["--qt"] or (os.environ.get("INKREEL_UI", "").lower() == "qt" and (not a or (len(a) == 1 and os.path.isfile(a[0])))):
        # нове вікно на Qt (1.1, поки робоча версія; типове лишається Tk): `InkReel --qt [файл]` чи змінна INKREEL_UI=qt
        from .qtui.window import main as qt_main
        sys.exit(qt_main([sys.argv[0]] + [x for x in a if x != "--qt"]))
    elif len(a) == 1 and os.path.isfile(a[0]):   # перетягнули файл на ярлик / "відкрити за допомогою"
        from .gui import main
        main(a[0])
    elif a:
        from .cli import main
        main()
    else:
        from .gui import main
        main()


if __name__ == "__main__":  # обов'язково: на Windows дочірні процеси перезавантажують модуль
    _run()
