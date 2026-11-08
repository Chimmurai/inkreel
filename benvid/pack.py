"""Стиснутий JSON-формат відео (.json.gz): метадані + кадри.

{"format":"benvid-json","version":1,"width":W,"height":H,"fps":[num,den],"frames":N,
 "data":[base64(zlib(packbits(кадр) XOR packbits(попередній кадр))), ...]}
Усе ще раз загортається в gzip. Кадр - 1 біт на піксель, різниця між кадрами майже нульова -> дуже малий файл.
"""
from __future__ import annotations

import base64
import gzip
import json
import zlib

import numpy as np

FORMAT = "benvid-json"


class JsonWriter:
    def __init__(self, path, w, h, fps_num, fps_den):
        self.path, self.w, self.h = path, w, h
        self.fps = (int(fps_num), int(fps_den))
        self.data, self.prev = [], None

    def add(self, frame01: np.ndarray):
        pk = np.packbits(frame01.astype(np.uint8) > 0, axis=-1).ravel()
        d = pk if self.prev is None else np.bitwise_xor(pk, self.prev)
        self.prev = pk
        self.data.append(base64.b64encode(zlib.compress(d.tobytes(), 6)).decode("ascii"))

    def close(self):
        doc = dict(format=FORMAT, version=1, width=self.w, height=self.h, fps=list(self.fps),
                   frames=len(self.data), data=self.data)
        with gzip.open(self.path, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(doc, f, separators=(",", ":"))


def read_json_video(path):
    """-> (meta dict, список кадрів uint8 0/1 розміру h×w)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != FORMAT:
        raise ValueError("це не benvid-json")
    w, h = doc["width"], doc["height"]
    prev, frames = None, []
    for s in doc["data"]:
        d = np.frombuffer(zlib.decompress(base64.b64decode(s)), np.uint8)
        prev = d if prev is None else np.bitwise_xor(d, prev)
        frames.append(np.unpackbits(prev.reshape(h, -1), axis=-1, count=w))
    meta = dict(width=w, height=h, fps=doc["fps"][0] / doc["fps"][1], frames=len(frames))
    return meta, frames
