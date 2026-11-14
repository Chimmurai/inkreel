"""Прискорення хвоста fp16-моделі IRNM1 для відеокарти (DirectML): python tools/opt_student_tail.py вхід.onnx вихід.onnx   (потрібен пакет onnx)

У тілі мережі останній крок на повній роздільності такий: ConvTranspose 2x2 крок 2 (16 -> 9 каналів) -> Concat з кадром (3) і шумом (4) = 16 каналів ->
Conv 1x1 (16 -> 16) -> LeakyReLU -> Conv 1x1 (16 -> 1) -> Greater 0 -> Cast uint8. Це чотири проходи по тензорах 16 x H x W (по 67 МБ на 1080p), і на відеокарті
вони коштують ~3,8 мс з 12,5. ConvTranspose 2x2/2 не має перекриття (це 1x1 згортка + DepthToSpace), тож увесь хвіст точно (в арифметиці дійсних чисел) зводиться до:
    SpaceToDepth(кадр+шум, 7 каналів) -> 28 каналів на H/2; Concat з входом хвоста (16) = 44 канали;
    Conv 1x1 (44 -> 64, ваги: свій блок 16 каналів на кожну з 4 фаз 2x2) -> LeakyReLU -> Conv 1x1 (64 -> 4, по блоку на фазу) -> DepthToSpace -> Greater -> Cast.
Тензорів на повній роздільності лишається один (1 канал). Ваги перераховуються в float64 з fp16-оригіналів і округлюються до fp16 (як було в оригіналі:
y9 округлювався до fp16 перед 1x1). На 6 кадрах 1080p маска відрізняється від оригіналу у 0,011 % пікселів (оригінал від fp32 CPU - 0,043 %), обчислення 12,5 -> 11,3 мс.
Моделі з іншою будовою голови (наприклад, fq_b) не змінюються: функція повертає False."""
from __future__ import annotations

import sys

import numpy as np

NEEDED = ("/o/ConvTranspose", "/Concat_1", "/h1/Conv", "/LeakyRelu_7", "/h3/Conv", "/Greater", "/Cast", "/Concat", "/Add_3", "/Constant_2")


def optimize(src: str, dst: str) -> bool:
    """зберегти в dst модель src з переробленим хвостом; False (нічого не записано), якщо будова голови інша"""
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(src)
    g = m.graph
    nodes = {n.name: n for n in g.node}
    if not all(k in nodes for k in NEEDED) or any(n.name.startswith("t/") for n in g.node):
        return False
    init = {i.name: i for i in g.initializer}
    convt = nodes["/o/ConvTranspose"]
    at = {a.name: helper.get_attribute_value(a) for a in convt.attribute}
    if list(at.get("kernel_shape", [])) != [2, 2] or list(at.get("strides", [])) != [2, 2] or any(at.get("pads", [0])) or at.get("group", 1) != 1:
        return False
    cat1 = nodes["/Concat_1"]
    if list(cat1.input) != ["/o/ConvTranspose_output_0", "/Sub_output_0", "noise"] or list(nodes["/Concat"].input) != ["/Sub_output_0", "noise"]:
        return False                                                  # порядок каналів 9 + 3 + 4 і 3 + 4 - саме він закладений у ваги нижче

    def arr(name):
        return numpy_helper.to_array(init[name]).astype(np.float64)

    wo, bo = arr(convt.input[1]), arr(convt.input[2])                 # (16, 9, 2, 2), (9)
    w1, b1 = arr(nodes["/h1/Conv"].input[1])[:, :, 0, 0], arr(nodes["/h1/Conv"].input[2])      # (16, 16), (16)
    w3, b3 = arr(nodes["/h3/Conv"].input[1])[:, :, 0, 0], arr(nodes["/h3/Conv"].input[2])      # (1, 16), (1)
    if wo.shape != (16, 9, 2, 2) or w1.shape != (16, 16) or w3.shape != (1, 16) or convt.input[0] != "/Add_3_output_0":
        return False
    alpha = [a.f for a in nodes["/LeakyRelu_7"].attribute if a.name == "alpha"][0]
    wn = np.zeros((64, 44), np.float64)
    bn = np.zeros(64, np.float64)
    w3n = np.zeros((4, 64), np.float64)
    for a in range(2):
        for b in range(2):
            p = a * 2 + b                                             # фаза (рядок, колонка) у блоці 2x2: так складає канали SpaceToDepth/DepthToSpace (DCR)
            wn[p * 16:(p + 1) * 16, 0:16] = w1[:, :9] @ wo[:, :, a, b].T
            wn[p * 16:(p + 1) * 16, 16 + p * 7:16 + p * 7 + 7] = w1[:, 9:16]
            bn[p * 16:(p + 1) * 16] = b1 + w1[:, :9] @ bo
            w3n[p, p * 16:(p + 1) * 16] = w3[0]
    b3n = np.full(4, float(b3[0]), np.float64)
    f16 = lambda x: x.astype(np.float16)
    new_init = [numpy_helper.from_array(f16(wn.reshape(64, 44, 1, 1)), "t.w1"), numpy_helper.from_array(f16(bn), "t.b1"),
                numpy_helper.from_array(f16(w3n.reshape(4, 64, 1, 1)), "t.w3"), numpy_helper.from_array(f16(b3n), "t.b3")]
    conv = dict(kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    new_nodes = [
        helper.make_node("SpaceToDepth", ["/Concat_output_0"], ["t.s2d"], name="t/s2d", blocksize=2),
        helper.make_node("Concat", ["/Add_3_output_0", "t.s2d"], ["t.cat"], name="t/cat", axis=1),
        helper.make_node("Conv", ["t.cat", "t.w1", "t.b1"], ["t.h1"], name="t/h1", **conv),
        helper.make_node("LeakyRelu", ["t.h1"], ["t.lr"], name="t/lr", alpha=alpha),
        helper.make_node("Conv", ["t.lr", "t.w3", "t.b3"], ["t.h3"], name="t/h3", **conv),
        helper.make_node("DepthToSpace", ["t.h3"], ["t.d2s"], name="t/d2s", blocksize=2, mode="DCR"),
        helper.make_node("Greater", ["t.d2s", "/Constant_2_output_0"], ["t.gt"], name="t/gt"),
        helper.make_node("Cast", ["t.gt"], [nodes["/Cast"].output[0]], name="t/cast", to=2),
    ]
    drop = {"/o/ConvTranspose", "/Concat_1", "/h1/Conv", "/LeakyRelu_7", "/h3/Conv", "/Greater", "/Cast"}
    kept = [n for n in g.node if n.name not in drop]
    del g.node[:]
    g.node.extend(kept)
    g.node.extend(new_nodes)
    for nm in (convt.input[1], convt.input[2], nodes["/h1/Conv"].input[1], nodes["/h1/Conv"].input[2], nodes["/h3/Conv"].input[1], nodes["/h3/Conv"].input[2]):
        g.initializer.remove(init[nm])
    g.initializer.extend(new_init)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ok = optimize(sys.argv[1], sys.argv[2])
    print("готово" if ok else "будова голови інша: модель не змінено")
    sys.exit(0 if ok else 1)
