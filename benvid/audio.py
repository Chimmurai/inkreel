"""Звук з оригінального відео: декодування через ffmpeg у PCM і відтворення (за наявності sounddevice)."""
from __future__ import annotations

import subprocess
import threading

import numpy as np

SR = 44100
MAX_SECONDS = 3 * 3600


def decode_pcm(path: str, ffmpeg: str | None):
    """-> int16 масив (n, 2) або None, якщо немає звуку/ffmpeg"""
    if not ffmpeg:
        return None
    try:
        r = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-vn", "-t", str(MAX_SECONDS), "-ac", "2", "-ar", str(SR),
                            "-f", "s16le", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=600,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:  # noqa: BLE001
        return None
    if r.returncode != 0 or len(r.stdout) < 4:
        return None
    a = np.frombuffer(r.stdout[: len(r.stdout) // 4 * 4], np.int16).reshape(-1, 2)
    return a


class AudioOut:
    """вивід звуку. Перший варіант - sounddevice (точна синхронізація), запасний на Windows - winsound (без залежностей:
    WAV із потрібного місця грається з пам'яті). Якщо нічого немає - status пояснює чому."""

    def __init__(self):
        self.pcm = None
        self.pos = 0
        self.on = False
        self.muted = False
        self.volume = 1.0
        self.status = "завантаження…"
        self._stream = None
        self._lock = threading.Lock()
        self._sd = self._ws = None
        self.cb_count, self.err, self.dev = 0, "", ""
        self._tmp = None
        try:
            import sounddevice as sd
            self._sd = sd
        except Exception:  # noqa: BLE001
            pass
        try:
            import winsound
            self._ws = winsound
        except Exception:  # noqa: BLE001
            pass

    @property
    def backend(self) -> str:
        return "sd" if self._sd else "ws" if self._ws else ""

    def set_volume(self, v: float):
        """v: положення повзунка 0..1 -> підсилення. Квадратична крива: на слух рівномірніше, ніж пряма (тихий кінець повзунка
        не «провалюється» миттєво)"""
        v = min(max(float(v), 0.0), 1.0)
        self.volume = v * v

    @property
    def info(self) -> str:
        """короткий опис для рядка стану: стан, пристрій, лічильник викликів потоку, остання помилка"""
        s = self.status
        if self.status == "ок":
            s += f" ({self.backend}{', ' + self.dev if self.dev else ''}{', ' + str(self.cb_count) if self._stream else ''})"
        return s + (f" [{self.err}]" if self.err else "")

    @property
    def available(self) -> bool:
        return bool(self.backend) and self.pcm is not None

    def load(self, path: str, ffmpeg: str | None):
        self.stop()
        self.pcm = None
        self.status = "завантаження…"
        pcm = decode_pcm(path, ffmpeg)
        with self._lock:
            self.pcm = pcm
        if pcm is None:
            self.status = "немає доріжки"
        elif not self.backend:
            self.status = "встанови sounddevice (pip install sounddevice)"
        else:
            self.status = "ок"

    def _cb(self, out, frames, ti, status):
        try:
            self._cb2(out, frames)
        except Exception as e:  # noqa: BLE001
            out[:] = 0
            self.err = f"cb:{e.__class__.__name__}"

    def _cb2(self, out, frames):
        self.cb_count += 1
        with self._lock:
            pcm, p = self.pcm, self.pos
            if pcm is None or not self.on:
                out[:] = 0
                return
            chunk = pcm[p:p + frames]
            n = len(chunk)
            out[:n] = chunk * (0.0 if self.muted else self.volume)
            out[n:] = 0
            self.pos = p + n

    def play_from(self, t: float) -> bool:
        """-> True, якщо звук справді запущено (звук ще міг не декодуватись - тоді викликаємо знову пізніше)"""
        if not self.available:
            return False
        if self._sd is not None:
            with self._lock:
                self.pos = max(0, int(t * SR))
                self.on = True
            if self._stream is None:
                try:
                    self._stream = self._sd.OutputStream(samplerate=SR, channels=2, dtype="int16", callback=self._cb,
                                                         blocksize=1024)
                    self._stream.start()
                    try:
                        self.dev = str(self._sd.query_devices(kind="output")["name"])[:24]
                    except Exception:  # noqa: BLE001
                        pass
                except Exception as e:  # noqa: BLE001
                    self._stream = None
                    self.err = f"sd:{e.__class__.__name__}"
                    self._sd = None
                    return self.play_from(t)        # спробувати запасний варіант
        elif self._ws is not None:
            self.on = True
            if self.muted:
                return True
            import io
            import wave
            p = max(0, int(t * SR))
            chunk = self.pcm[p:p + SR * 600]
            if len(chunk) < 2:
                return True
            if self.volume < 0.999:                       # winsound не має регулятора: гучність застосовуємо до самих даних
                chunk = (chunk.astype(np.float32) * self.volume).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(chunk.tobytes())
            try:                      # SND_MEMORY разом із SND_ASYNC не працює - граємо з тимчасового файлу
                import os
                import tempfile
                if self._tmp is None:
                    fd, self._tmp = tempfile.mkstemp(suffix=".wav", prefix="inkreel_")
                    os.close(fd)
                with open(self._tmp, "wb") as f:
                    f.write(buf.getvalue())
                self._ws.PlaySound(self._tmp, self._ws.SND_FILENAME | self._ws.SND_ASYNC)
                self.dev = "winsound"
            except Exception as e:  # noqa: BLE001
                self.err = f"ws:{e.__class__.__name__}"
        return True

    def sync(self, t: float):
        """підтягнути звук до відеогодинника (sounddevice): різниця > 60 мс - перестрибнути"""
        if self._sd is None or self._stream is None or not self.on:
            return
        try:
            lat = float(self._stream.latency)
        except Exception:  # noqa: BLE001
            lat = 0.0
        target = int((t + lat) * SR)
        with self._lock:
            if abs(self.pos - target) > 0.06 * SR:
                self.pos = max(0, target)

    def pause(self):
        with self._lock:
            self.on = False
        if self._ws is not None and self._sd is None and self.on is False:
            try:
                self._ws.PlaySound(None, self._ws.SND_PURGE)
            except Exception:  # noqa: BLE001
                pass

    def stop(self):
        self.pause()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None

    def close(self):
        self.stop()
        if self._tmp:
            try:
                import os
                os.remove(self._tmp)
            except OSError:
                pass
            self._tmp = None
