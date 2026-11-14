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
    elif a[:1] == ["--qt"] or (_want_qt() and (not a or (len(a) == 1 and os.path.isfile(a[0])))):
        # вікно на Qt: `InkReel --qt [файл]`, змінна INKREEL_UI=qt, а в бета-каналі (BETA_LABEL) воно типове
        from .qtui.window import main as qt_main
        sys.exit(qt_main([sys.argv[0]] + [x for x in a if x != "--qt"]))
    elif a[:1] == ["--tk"] and (len(a) == 1 or (len(a) == 2 and os.path.isfile(a[1]))):   # запасне вікно на Tk у бета-каналі
        from .gui import main
        main(a[1] if len(a) == 2 else None)
    elif len(a) == 1 and os.path.isfile(a[0]):   # перетягнули файл на ярлик / "відкрити за допомогою"
        from .gui import main
        main(a[0])
    elif a:
        from .cli import main
        main()
    else:
        from .gui import main
        main()


def _want_qt() -> bool:
    """Qt є типовим лише в бета-каналі; INKREEL_UI=tk|qt перекриває вибір."""
    ui = os.environ.get("INKREEL_UI", "").lower()
    if ui in ("tk", "qt"):
        return ui == "qt"
    from . import BETA_LABEL
    return bool(BETA_LABEL)


if __name__ == "__main__":  # обов'язково: на Windows дочірні процеси перезавантажують модуль
    _run()
