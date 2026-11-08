"""Додає до ONNX-тіла студента вхід uint8 BGR (1,H,W,3): Cast -> *1/255 -> BGR->RGB -> NHWC->NCHW прямо в графі (на відеокарті), замість float-тензора `image`.
Так кадр іде в модель байтами, а не float16 (удвічі-вчетверо менше даних через PCIe, без підготовки на процесорі).

python tools/make_u8_body.py тіло_з_--mask.onnx вихід.onnx   (тип float визначається за входом `image`: fp32 або fp16)"""
from __future__ import annotations

import sys


def add_u8_front(src: str, dst: str) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto as TP
    from onnx import helper as H
    from onnx import numpy_helper as NH

    m = onnx.load(src)
    g = m.graph
    img = next((i for i in g.input if i.name == "image"), None)
    if img is None:
        raise SystemExit("у графі нема входу `image` (це не тіло студента, або вже uint8)")
    ft = img.type.tensor_type.elem_type                           # FLOAT або FLOAT16
    if ft not in (TP.FLOAT, TP.FLOAT16):
        raise SystemExit(f"незвичний тип входу image: {ft}")
    g.input.remove(img)
    g.input.insert(0, H.make_tensor_value_info("bgr", TP.UINT8, [1, "H", "W", 3]))
    npdt = np.float32 if ft == TP.FLOAT else np.float16
    g.initializer.extend([
        NH.from_array(np.array(1.0 / 255.0, dtype=npdt), "u8_scale"),
        NH.from_array(np.array([-1], dtype=np.int64), "u8_st"),                    # канали навпаки: BGR -> RGB
        NH.from_array(np.array([-(2 ** 62)], dtype=np.int64), "u8_en"),
        NH.from_array(np.array([3], dtype=np.int64), "u8_ax"),
        NH.from_array(np.array([-1], dtype=np.int64), "u8_sp"),
    ])
    front = [
        H.make_node("Cast", ["bgr"], ["u8_f"], to=ft),
        H.make_node("Mul", ["u8_f", "u8_scale"], ["u8_n"]),
        H.make_node("Slice", ["u8_n", "u8_st", "u8_en", "u8_ax", "u8_sp"], ["u8_rgb"]),
        H.make_node("Transpose", ["u8_rgb"], ["image"], perm=[0, 3, 1, 2]),
    ]
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(front + nodes)
    onnx.checker.check_model(m)
    onnx.save(m, dst)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    add_u8_front(sys.argv[1], sys.argv[2])
    print("записано", sys.argv[2])
