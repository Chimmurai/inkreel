# InkReel — third-party software / Сторонні компоненти

InkReel works offline and sends nothing anywhere. To do its job it bundles the open-source components listed below. This file says what they are,
under which licences they come, and where their source code is. The full texts of the GPL and LGPL are in the `licenses` folder next to the program;
the licence files of every Python package in this exact build are in `licenses\python\<package>\`, and `licenses\python\INDEX.txt` lists their versions.

*Українською:* у збірку входять сторонні бібліотеки та програма ffmpeg. Нижче — що це, під якими ліцензіями вони поширюються й де їхній
вихідний код. Тексти GPL/LGPL лежать у теці `licenses` поруч із програмою, а ліцензії всіх пакетів Python саме цієї збірки — у `licenses\python`.

Versions below are those current when this file was written (October 2026); the versions in a particular build are listed in `licenses\python\INDEX.txt`.

## Components

| Component | Used for | Licence | Project |
|---|---|---|---|
| Python 3 with Tcl/Tk | runtime and the window | PSF License; Tcl/Tk licence (BSD-style) | https://www.python.org/ |
| PyInstaller (bootloader) | packs the program into `InkReel.exe` | GPL-2.0-or-later *with the bootloader exception*, which allows the programs it builds to be distributed under any licence | https://pyinstaller.org/ |
| NumPy 2.5 | arrays | BSD-3-Clause (the wheel also bundles OpenBLAS and other parts under BSD-3-Clause, 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org/ |
| OpenCV (`opencv-python` 5.0) | reading video and images, resizing | Apache-2.0; the FFmpeg library inside the wheel is LGPL-2.1-or-later | https://opencv.org/ |
| Numba 0.68 and llvmlite 0.50 | fast compiled kernels | BSD-2-Clause; llvmlite bundles LLVM under Apache-2.0 WITH LLVM-exception | https://numba.pydata.org/ |
| ONNX Runtime (`onnxruntime-directml` on Windows, `onnxruntime` elsewhere) | runs the trained IRNM1 (InkreelNeuralModel v1) network (quality level 6) on the graphics card or the processor | MIT; the DirectML build also contains Microsoft's DirectML library, a redistributable component of the DirectML package | https://onnxruntime.ai/ |
| Pillow 13 | image files | MIT-CMU (HPND); the wheel bundles brotli, FreeType (FTL), HarfBuzz, Little CMS, libavif, libjpeg-turbo, libpng, libwebp, OpenJPEG, libtiff, xz and zlib-ng under their own permissive licences (all in its `LICENSE`) | https://python-pillow.org/ |
| pillow-heif 1.8 | HEIC/HEIF photos | BSD-3-Clause for the Python code; the Windows wheel bundles **libheif (LGPL-3.0), libde265 (LGPL-3.0), x265 (GPL-2.0)** and the MinGW-w64 runtime (GPL-3.0 with GCC runtime exception; libwinpthread MIT/BSD). Its authors label the binary wheel as GPLv2 as a whole | https://github.com/bigcat88/pillow_heif |
| imageio-ffmpeg 0.6 | finds and starts `ffmpeg.exe` | BSD-2-Clause for the wrapper; the bundled **`ffmpeg.exe` is FFmpeg 7.1 built as GPL-3.0-or-later** (see below) | https://github.com/imageio/imageio-ffmpeg |
| sounddevice 0.5 and PortAudio | sound in the player | MIT; PortAudio is MIT-style | https://python-sounddevice.readthedocs.io/ |
| windnd 1.0 | drag-and-drop of files into the window | MIT | https://pypi.org/project/windnd/ |
| cffi, pycparser | needed by sounddevice | MIT-0, BSD-3-Clause | https://cffi.readthedocs.io/ |
| DejaVu Sans Mono 2.37 | fallback font of the “Symbols” mode on systems without Consolas | Bitstream Vera / DejaVu Fonts License (free, permits redistribution; `licenses\DejaVu-Fonts-License.txt`) | https://dejavu-fonts.github.io/ |

InkReel's own pixel-art interface, icon and noise tables are the author's own work.

The “Symbols” mode draws with **Consolas**, Microsoft's font that is part of Windows. InkReel does not contain or redistribute it: the font file is read from the
user's own Windows installation at run time, only to draw characters on the screen and into the user's own files. Where Consolas is missing, the bundled
DejaVu Sans Mono (above) is used instead.

## FFmpeg (GPL-3.0-or-later)

`ffmpeg.exe` is a separate program that InkReel starts to encode video; it is not linked into InkReel's code. The build that comes with `imageio-ffmpeg 0.6.0`
is **FFmpeg 7.1 (libavcodec 61.19) configured with `--enable-gpl --enable-version3`** and contains, among others, x264, x265, Xvid, libvpx, libaom, libopus,
libvorbis, libmp3lame, libwebp and libass. Because of the GPL parts the whole `ffmpeg.exe` is licensed under the **GNU GPL, version 3 or later**
(`licenses\GPL-3.0.txt`). InkReel does not modify FFmpeg or any of the libraries in it.

Source code of the FFmpeg version shipped with InkReel:
FFmpeg 7.1 — https://ffmpeg.org/releases/ffmpeg-7.1.tar.xz (git tag `n7.1`: https://github.com/FFmpeg/FFmpeg/tree/n7.1).
The libraries built into it are open source and are published by their own projects, for example
x264 — https://code.videolan.org/videolan/x264, x265 — https://bitbucket.org/multicoreware/x265_git,
libvpx — https://chromium.googlesource.com/webm/libvpx, libaom — https://aomedia.googlesource.com/aom/.
The full configuration of the build is printed by `ffmpeg.exe -version`. If you cannot get any of this source code,
ask the author (see the *About* section of InkReel) and you will be given it.

## libheif, libde265, x265 (inside pillow-heif)

InkReel only *reads* HEIC/HEIF photos; the libraries come as they are in the `pillow-heif` wheel. libheif and libde265 are under the
**GNU LGPL, version 3** (`licenses\LGPL-3.0.txt`), x265 is under the **GNU GPL, version 2** (`licenses\GPL-2.0.txt`).
Sources: https://github.com/strukturag/libheif, https://github.com/strukturag/libde265, https://github.com/Multicorewareinc/x265
(the exact tags are listed in `licenses\python\pillow-heif\LICENSES_bundled.txt`).
The LGPL libraries are loaded as separate DLL files that you can replace with your own builds.

## FFmpeg inside OpenCV

The `opencv-python` wheel carries its own FFmpeg library (`opencv_videoio_ffmpeg*.dll`) under the **GNU LGPL, version 2.1 or later**
(`licenses\LGPL-2.1.txt`; source: https://ffmpeg.org/ and https://github.com/opencv/opencv-python). It is a separate DLL that you can replace.

## PyOpenCL (optional GPU acceleration)

If installed, `pyopencl` (MIT licence, https://github.com/inducer/pyopencl) and its helpers `pytools` and `platformdirs` (MIT) let InkReel run its pixel kernels on any OpenCL graphics card. The OpenCL driver itself comes with your graphics driver; InkReel does not ship one.

## The IRNM1 (InkreelNeuralModel v1) network

The network weights in `benvid/assets/student` (quality level 6, IRNM1) are the author's own work: they were trained with PyTorch (BSD-3-Clause; used only for training, **not** part of the program) to imitate InkReel's own “Engraving” algorithm (level 5), on pairs produced by that algorithm. They are loaded and run with ONNX Runtime (above). If ONNX Runtime or the weights are missing, InkReel simply renders level 6 with “Engraving”.

## No warranty

All these components are provided "as is", without warranty of any kind, under the terms of their licences. Product names and trademarks belong to their owners.
