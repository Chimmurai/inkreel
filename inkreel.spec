# -*- mode: python ; coding: utf-8 -*-
# Збірка:  pyinstaller inkreel.spec --noconfirm        (або build_exe.bat)
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = [("benvid/assets", "benvid/assets"), ("benvid/bluenoise128.npy", "benvid")]
datas += collect_data_files("imageio_ffmpeg")          # ffmpeg.exe лежить у binaries/
hidden = collect_submodules("numba", filter=lambda n: ".tests" not in n) + ["sounddevice", "windnd", "PIL.ImageGrab", "benvid.ink5",
                                         "benvid.lang_en", "benvid.guide", "benvid.view", "benvid.i18n", "benvid.meta",
                                         "benvid.eggs", "benvid.quick", "benvid.shellmenu", "benvid.store", "benvid.gpu", "benvid.host", "benvid.glyph", "benvid.bake", "benvid.ikrctx", "benvid.nbcache"]
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
    excludes=["matplotlib", "scipy", "pandas", "IPython", "pytest", "PyQt5", "PyQt6", "PySide2", "PySide6"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="InkReel", console=False,
          icon="benvid/assets/inkreel.ico", upx=False)
# другий exe з тим самим кодом - для процесів-обробників (у диспетчері завдань підписані «InkReel-Worker»)
exe_w = EXE(pyz, a.scripts, [], exclude_binaries=True, name="InkReel-Worker", console=False,
            icon="benvid/assets/inkreel.ico", upx=False)
coll = COLLECT(exe, exe_w, a.binaries, a.datas, name="InkReel", upx=False)
