"""Один запущений екземпляр. Повторний запуск (ярлик, «відкрити за допомогою», подвійний клік) не створює ще одне
вікно з десятком обробників, а передає файл уже запущеному вікну, піднімає його й одразу завершується.
Зв'язок - локальний сокет 127.0.0.1 (лише цей комп'ютер) з простим рукостисканням."""
from __future__ import annotations

import hashlib
import os
import socket
import threading

PORT = 47613
MAGIC = "INKREEL1"


ANSWER_WAIT = 7.0          # скільки чекає той, хто передає файл: вікно може ще будуватись (до 5 с) + підтвердити (до 2 с)


def _tag() -> str:
    """короткий відбиток користувача Windows у «паролі»: порт локальний для всього комп'ютера, а не для одного входу в систему. Без відбитка другий
    користувач (швидке перемикання користувачів на спільному ПК) «передав» би файл вікну першого, якого не бачить, і сам лишився б без вікна: тепер чужий
    слухач йому не відповідає, і він працює як звичайно"""
    try:
        import getpass
        who = getpass.getuser()
    except Exception:  # noqa: BLE001
        who = ""
    return hashlib.sha1(who.encode("utf-8", "replace")).hexdigest()[:6]


def _handoff(arg: str, port: int = PORT, magic: str = MAGIC) -> bool:
    """віддати файл існуючому екземпляру -> True, лише якщо його вікно підтвердило, що отримало (мертвий чи завислий процес
    «OK» не скаже - і новий запуск відкриє своє вікно, а не зникне мовчки)"""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0) as s:
            s.settimeout(ANSWER_WAIT)
            s.sendall(f"{magic}{_tag()}\t{arg}\n".encode("utf-8"))
            return s.recv(8).startswith(b"OK")
    except OSError:
        return False


class Listener:
    def __init__(self, srv: socket.socket, magic: str = MAGIC):
        self.srv = srv
        self.magic = magic
        self.cb = None
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def start(self, cb):
        """cb(path_or_empty) викликається з потоку слухача. Відповідь «OK» іде ПІСЛЯ нього, і лише якщо cb не повернула False:
        так вікно, що вже закрилося або зависло, не «приймає» файл, якого ніхто не покаже"""
        self.cb = cb

    def close(self):
        """перестати слухати: нові підключення відхиляються. На Linux/macOS самого close() мало: потік, що завис в accept(), так і
        лишається слухачем, доки не прийде ще одне підключення, - тому спершу shutdown (на Windows він просто нічого не робить)"""
        try:
            self.srv.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.srv.close()
        except OSError:
            pass

    def _loop(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            try:
                c.settimeout(2.0)
                data = b""
                while not data.endswith(b"\n") and len(data) < 8192:
                    chunk = c.recv(1024)
                    if not chunk:
                        break
                    data += chunk
                head, _, arg = data.decode("utf-8", "replace").rstrip("\n").partition("\t")
                if head == self.magic:
                    ok = False
                    for _ in range(50):                      # вікно може ще будуватись
                        if self.cb:
                            try:
                                ok = self.cb(arg) is not False
                            except Exception:  # noqa: BLE001
                                ok = False
                            break
                        threading.Event().wait(0.1)
                    c.sendall(b"OK" if ok else b"NO")
            except Exception:  # noqa: BLE001
                pass
            finally:
                try:
                    c.close()
                except OSError:
                    pass


def acquire(arg: str = "", port: int = PORT, magic: str = MAGIC):
    """-> Listener (ми перші) | None (немає змоги слухати - просто працюємо) | False (передали існуючому, виходимо).
    port/magic - окремий «канал» (швидка дія з Провідника, benvid/quick.py, має свій, щоб не плутатись із головним вікном)"""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name != "nt":
            # Linux/macOS: дозволити зайняти порт, що ще тримається у TIME_WAIT після щойно закритого екземпляра (живого слухача це
            # не обходить). На Windows такого прапорця не треба, а SO_REUSEADDR там означав би «перехопити чужий порт»
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(16)
        return Listener(srv, magic + _tag())
    except OSError:
        srv.close()
    return False if _handoff(arg, port, magic) else None
