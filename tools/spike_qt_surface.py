"""Спайк 1.1: скільки коштує показ кадра на Qt проти Tk.

    python tools/spike_qt_surface.py            -> три варіанти по черзі, цифри в консоль

A  QWidget + QPainter.drawImage з Indexed8 (палітра в таблиці кольорів) - CPU-растр Qt
B  QOpenGLWidget + QPainter.drawImage - той самий малюнок, але растеризує OpenGL
C  QOpenGLWidget + шейдер: 8-бітна текстура маски (R8) і дві фарби палітри; масштаб і палітра на відеокарті

Кожен варіант малює MASK_W x MASK_H (типовий кадр 2580x1080) у вікно W x H, кожен кадр - інший вміст (4 різні маски по колу).
Без vsync, тож кадри/с = 1000 / (мс на кадр). Усе в одному процесі й з таймаутом: GUI-скрипти лише з main-guard."""
from __future__ import annotations

import sys
import time

import numpy as np
from PySide6.QtCore import QElapsedTimer, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QSurfaceFormat
from PySide6.QtWidgets import QApplication, QWidget

MASK_W, MASK_H = 2580, 1080
WIN_W, WIN_H = 1900, 1000
SECONDS = 4.0
INK, PAPER = QColor("#000000"), QColor("#ffffff")


def masks(n=4):
    rng = np.random.RandomState(3)
    out = []
    for _ in range(n):
        base = rng.rand(MASK_H // 6 + 1, MASK_W // 6 + 1) > 0.5
        m = np.kron(base, np.ones((6, 6), bool))[:MASK_H, :MASK_W]          # крупні «пікселі», як у справжньому кадрі
        out.append(np.ascontiguousarray(m.astype(np.uint8)))
    return out


class Meter:
    def __init__(self):
        self.t = QElapsedTimer()
        self.t.start()
        self.n = 0
        self.paint_ms = []

    def tick(self, ms):
        self.n += 1
        self.paint_ms.append(ms)

    def report(self, name):
        total = self.t.elapsed() / 1000.0
        a = sorted(self.paint_ms[5:]) or [0]
        print("%-34s %6.1f fps | paint median %.2f ms, p95 %.2f ms, max %.2f ms" %
              (name, self.n / total, a[len(a) // 2], a[int(len(a) * 0.95)], a[-1]))


class RasterView(QWidget):
    def __init__(self, ms):
        super().__init__()
        self.ms = masks()
        self.i = 0
        self.m = Meter()
        self.imgs = []

    def frame(self):
        im = QImage(self.ms[self.i % 4].data, MASK_W, MASK_H, MASK_W, QImage.Format_Indexed8)      # новий QImage щокадру: кешу текстури нема
        im.setColorTable([INK.rgb(), PAPER.rgb()])
        return im

    def paintEvent(self, e):
        t0 = time.perf_counter()
        p = QPainter(self)
        p.drawImage(QRectF(0, 0, self.width(), self.height()), self.frame())
        p.end()
        self.m.tick((time.perf_counter() - t0) * 1000)
        self.i += 1


def run(widget, name):
    app = QApplication.instance()
    widget.resize(WIN_W, WIN_H)
    widget.show()
    timer = QTimer()
    timer.timeout.connect(widget.update)
    timer.start(0)
    end = time.time() + SECONDS
    while time.time() < end:
        app.processEvents()
    timer.stop()
    widget.m.report(name)
    widget.close()


def main():
    fmt = QSurfaceFormat()
    fmt.setSwapInterval(0)
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    which = sys.argv[1] if len(sys.argv) > 1 else "ABC"
    if "A" in which:
        run(RasterView(0), "A raster QWidget + Indexed8")
    try:
        from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader, QOpenGLTexture
        from PySide6.QtOpenGLWidgets import QOpenGLWidget
        from PySide6.QtGui import QOpenGLFunctions
    except Exception as e:  # noqa: BLE001
        print("OpenGL widgets unavailable:", e)
        return

    class GLPainter(QOpenGLWidget):
        def __init__(self):
            super().__init__()
            self.ms = masks()
            self.i = 0
            self.m = Meter()
            self.imgs = []

        def frame(self):
            im = QImage(self.ms[self.i % 4].data, MASK_W, MASK_H, MASK_W, QImage.Format_Indexed8)
            im.setColorTable([INK.rgb(), PAPER.rgb()])
            return im

        def paintGL(self):
            t0 = time.perf_counter()
            p = QPainter(self)
            p.drawImage(QRectF(0, 0, self.width(), self.height()), self.frame())
            p.end()
            self.context().functions().glFinish()                 # дочекатись відеокарти: інакше міряємо лише постановку команд
            self.m.tick((time.perf_counter() - t0) * 1000)
            self.i += 1

    class GLShader(QOpenGLWidget):
        VS = """#version 330 core
        layout(location=0) in vec2 pos;
        out vec2 uv;
        void main(){ uv = vec2(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5); gl_Position = vec4(pos, 0.0, 1.0); }"""
        FS = """#version 330 core
        in vec2 uv; out vec4 col; uniform sampler2D tex; uniform vec3 ink; uniform vec3 paper;
        void main(){ float m = texture(tex, uv).r; col = vec4(mix(ink, paper, m), 1.0); }"""

        def __init__(self):
            super().__init__()
            self.ms = masks()
            self.i = 0
            self.m = Meter()
            self.tex = None

        def initializeGL(self):
            self.f = QOpenGLFunctions(self.context())
            self.f.initializeOpenGLFunctions()
            self.prog = QOpenGLShaderProgram(self)
            self.prog.addShaderFromSourceCode(QOpenGLShader.Vertex, self.VS)
            self.prog.addShaderFromSourceCode(QOpenGLShader.Fragment, self.FS)
            self.prog.link()
            self.tex = QOpenGLTexture(QOpenGLTexture.Target2D)
            self.tex.setFormat(QOpenGLTexture.R8_UNorm)
            self.tex.setSize(MASK_W, MASK_H)
            self.tex.setMipLevels(1)
            self.tex.allocateStorage(QOpenGLTexture.Red, QOpenGLTexture.UInt8)
            self.tex.setMinMagFilters(QOpenGLTexture.Linear, QOpenGLTexture.Linear)
            self.tex.setWrapMode(QOpenGLTexture.ClampToEdge)
            self.f.glPixelStorei(0x0CF5, 1)                         # GL_UNPACK_ALIGNMENT = 1: рядок 2580 байт не кратний 4
            from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLVertexArrayObject
            quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
            self.vao = QOpenGLVertexArrayObject(self)
            self.vao.create()
            self.vao.bind()
            self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            self.vbo.create()
            self.vbo.bind()
            self.vbo.allocate(quad.tobytes(), quad.nbytes)
            self.prog.bind()
            self.prog.enableAttributeArray(0)
            self.prog.setAttributeBuffer(0, 0x1406, 0, 2, 0)             # GL_FLOAT, 2 компоненти
            self.vao.release()

        def paintGL(self):
            t0 = time.perf_counter()
            a = self.ms[self.i % 4]
            self.tex.setData(0, QOpenGLTexture.Red, QOpenGLTexture.UInt8, a.tobytes())
            ratio = self.devicePixelRatio()
            self.f.glViewport(0, 0, int(self.width() * ratio), int(self.height() * ratio))
            self.prog.bind()
            self.tex.bind(0)
            self.prog.setUniformValue("tex", 0)
            self.prog.setUniformValue("ink", 0.0, 0.0, 0.0)
            self.prog.setUniformValue("paper", 1.0, 1.0, 1.0)
            self.vao.bind()
            self.f.glDrawArrays(5, 0, 4)                              # GL_TRIANGLE_STRIP
            self.vao.release()
            self.f.glFinish()
            self.m.tick((time.perf_counter() - t0) * 1000)
            self.i += 1
    if "B" in which:
        run(GLPainter(), "B QOpenGLWidget + QPainter")
    if "C" in which:
        try:
            run(GLShader(), "C OpenGL shader R8 (upload + draw)")
        except Exception as e:  # noqa: BLE001
            print("C failed:", type(e).__name__, e)


if __name__ == "__main__":
    main()
