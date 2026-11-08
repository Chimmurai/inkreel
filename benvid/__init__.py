"""InkReel: автоматичний чорно-білий обробник відео, GIF і фото."""
from . import log as _log

__version__ = "1.0.0"
import multiprocessing as _mp
import os as _os

if _mp.parent_process() is not None:        # процес-обробник пулу: один потік на процес (паралелізм дають самі процеси)
    for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ[_k] = "1"               # до імпорту numba: інакше кожен із десятків процесів створює пул потоків на всі ядра
    _os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
_log.prepare_env()          # до імпорту numba
