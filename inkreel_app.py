"""Точка входу для збірки .exe (PyInstaller). Звичайний запуск: python -m benvid"""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from benvid.__main__ import _run
    _run()
