# -*- mode: python ; coding: utf-8 -*-
# Збірка:  pyinstaller inkreel.spec --noconfirm        (або build_exe.bat)
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = [("benvid/assets", "benvid/assets"), ("benvid/bluenoise128.npy", "benvid")]
datas += collect_data_files("imageio_ffmpeg")          # ffmpeg.exe лежить у binaries/
hidden = collect_submodules("numba", filter=lambda n: ".tests" not in n) + ["sounddevice", "PIL.ImageGrab", "benvid.ink5",
                                         "benvid.lang_en", "benvid.guide", "benvid.view", "benvid.i18n", "benvid.meta",
                                         "benvid.eggs", "benvid.quick", "benvid.shellmenu", "benvid.store", "benvid.gpu", "benvid.host", "benvid.glyph", "benvid.bake", "benvid.ikrctx", "benvid.nbcache",
                                         # інтерфейс на Qt: PySide6 і всі модулі benvid.qtui (імпортуються зсередини функцій, тож статичний аналіз їх міг би не знайти)
                                         "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"] + collect_submodules("benvid.qtui")
binaries = []
try:                                                    # OpenCL-прискорення: необов'язково (pyopencl + її дані)
    from PyInstaller.utils.hooks import collect_all as _ca
    _d, _b, _h = _ca("pyopencl")
    datas += _d
    binaries += _b
    hidden += _h + ["pytools", "platformdirs"]
except Exception:
    pass
try:                                                    # рівень 6 «Нейро»: onnxruntime (DirectML.dll та провайдери лежать серед його бібліотек); без нього рівень рахується як «Ультра»
    from PyInstaller.utils.hooks import collect_all as _co
    _d, _b, _h = _co("onnxruntime")
    datas += _d
    binaries += _b
    hidden += _h + ["benvid.student", "benvid.student.runtime", "benvid.student.inputs"]
except Exception:
    pass
try:                                                    # HEIC/HEIF (фото з iPhone): необов'язковий пакет, якщо встановлено - вмикаємо
    from PyInstaller.utils.hooks import collect_all
    _d, _b, _h = collect_all("pillow_heif")
    datas += _d
    binaries += _b
    hidden += _h
except Exception:
    pass

a = Analysis(
    ["inkreel_app.py"], pathex=["."], datas=datas, binaries=binaries, hiddenimports=hidden,
    excludes=["matplotlib", "scipy", "pandas", "IPython", "pytest", "PyQt5", "PyQt6", "PySide2",
              # з PySide6 беремо лише QtCore/QtGui/QtWidgets (інтерфейс); решта модулів Qt у збірку не потрібна
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick", "PySide6.Qt3DCore", "PySide6.QtQuick",
              "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts", "PySide6.QtDataVisualization",
              "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtBluetooth", "PySide6.QtNetwork"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="InkReel", console=False,
          icon="benvid/assets/inkreel.ico", upx=False)
# другий exe з тим самим кодом - для процесів-обробників (у диспетчері завдань підписані «InkReel-Worker»)
exe_w = EXE(pyz, a.scripts, [], exclude_binaries=True, name="InkReel-Worker", console=False,
            icon="benvid/assets/inkreel.ico", upx=False)
coll = COLLECT(exe, exe_w, a.binaries, a.datas, name="InkReel", upx=False)
