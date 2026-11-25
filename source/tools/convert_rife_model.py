#!/usr/bin/env python3
"""Convert an upstream 11-channel FP32 RIFE ONNX model (vs-mlrt external-models)
into the 7-channel FP16 ONNX format used by vf_rife.

Usage: convert_rife_model.py SRC.onnx DST.onnx [FLOW_SCALE]

  * Supports both legacy [3, 3, 1, 2, 2] (e.g. v4.6) and modern [6, 1, 2, 2]
    (v4.15 - v4.26) first Split layouts.
  * Generates base_grid and multiplier inside the graph in FP32 and keeps the
    GridSample coordinate math in FP32 so warp coordinates keep sub-pixel precision.
  * Casts all remaining weights and operators to FP16.
"""

import os
import sys
import numpy as np
import onnx
from onnx import TensorProto as T, helper as H
from onnx.numpy_helper import from_array, to_array


def make_const(name, value, dtype):
    return H.make_node(
        "Constant",
        [],
        [name],
        name=name,
        value=from_array(np.array(value, dtype=dtype)),
    )


def convert(src: str, dst: str, scale: float = 1.0) -> None:
    m = onnx.load(src)
    g = m.graph
    in_name = g.input[0].name

    # 1. Optional internal flow scale rewrite (when supported by legacy models)
    if scale != 1.0:
        rc = mc = 0
        for n in g.node:
            if len(n.output) == 1 and n.op_type == "Constant":
                o = n.output[0]
                if o.startswith("onnx::Resize"):
                    rc += 1
                    a = to_array(n.attribute[0].t).copy()
                    a[2:4] = a[2:4] / scale if rc % 3 == 2 else a[2:4] * scale
                    n.attribute[0].t.raw_data = from_array(a).raw_data
                elif o.startswith("onnx::Mul"):
                    mc += 1
                    a = to_array(n.attribute[0].t).copy()
                    a = a / scale if mc % 2 == 1 else a * scale
                    n.attribute[0].t.raw_data = from_array(a).raw_data
        if (rc, mc) != (11, 7):
            raise RuntimeError(
                f"flow scale {scale} unsupported on this model (Resize={rc}, Mul={mc})"
            )

    # 2. Rewrite the first Split on the 11-channel input to take 7 channels
    split = next(
        (n for n in g.node if n.op_type == "Split" and n.input and n.input[0] == in_name),
        None,
    )
    if split is None:
        split = next(n for n in g.node if n.op_type == "Split")
    sizes_const = next(
        n for n in g.node if n.output and n.output[0] == split.input[1]
    )
    sizes = [int(x) for x in to_array(sizes_const.attribute[0].t)]
    if sum(sizes) != 11 or sizes[-2:] != [2, 2] or len(split.output) != len(sizes):
        raise RuntimeError(f"unexpected Split layout: sizes={sizes}, outputs={list(split.output)}")

    base_grid_name = split.output[-2]
    mult_name = split.output[-1]
    sizes_const.attribute[0].t.CopyFrom(
        from_array(np.array(sizes[:-2], dtype=np.int64))
    )
    del split.output[-2:]

    grid = [
        make_const("grid_all1", [1, 1, 1, 1], np.int64),
        make_const("grid_i0", 0, np.int64),
        make_const("grid_i1", 1, np.int64),
        make_const("grid_i2", [2], np.int64),
        make_const("grid_i3", [3], np.int64),
        make_const("grid_f1", 1.0, np.float32),
        make_const("grid_f2", 2.0, np.float32),
        make_const("grid_rsx", [1, 1, 1, -1], np.int64),
        make_const("grid_rsy", [1, 1, -1, 1], np.int64),
        make_const("grid_one", [1], np.int64),
        H.make_node("Shape", [in_name], ["grid_shape"], name="grid_shape"),
        H.make_node("Gather", ["grid_shape", "grid_i2"], ["grid_h1"], name="grid_h1", axis=0),
        H.make_node("Gather", ["grid_shape", "grid_i3"], ["grid_w1"], name="grid_w1", axis=0),
        H.make_node("Squeeze", ["grid_h1"], ["grid_h"], name="grid_h"),
        H.make_node("Squeeze", ["grid_w1"], ["grid_w"], name="grid_w"),
        H.make_node(
            "Concat",
            ["grid_one", "grid_one", "grid_h1", "grid_w1"],
            ["grid_full"],
            name="grid_full",
            axis=0,
        ),
    ]
    for axis_name, length_name, reshape_const in (
        ("x", "grid_w", "grid_rsx"),
        ("y", "grid_h", "grid_rsy"),
    ):
        grid += [
            H.make_node(
                "Range",
                ["grid_i0", length_name, "grid_i1"],
                [f"grid_r{axis_name}"],
                name=f"grid_r{axis_name}",
            ),
            H.make_node(
                "Cast",
                [f"grid_r{axis_name}"],
                [f"grid_rf{axis_name}"],
                name=f"grid_rf{axis_name}",
                to=T.FLOAT,
            ),
            H.make_node(
                "Cast",
                [length_name],
                [f"grid_nf{axis_name}"],
                name=f"grid_nf{axis_name}",
                to=T.FLOAT,
            ),
            H.make_node(
                "Sub",
                [f"grid_nf{axis_name}", "grid_f1"],
                [f"grid_n1{axis_name}"],
                name=f"grid_n1{axis_name}",
            ),
            H.make_node(
                "Div",
                ["grid_f2", f"grid_n1{axis_name}"],
                [f"grid_st{axis_name}"],
                name=f"grid_st{axis_name}",
            ),
            H.make_node(
                "Mul",
                [f"grid_rf{axis_name}", f"grid_st{axis_name}"],
                [f"grid_m{axis_name}"],
                name=f"grid_m{axis_name}",
            ),
            H.make_node(
                "Sub",
                [f"grid_m{axis_name}", "grid_f1"],
                [f"grid_s{axis_name}"],
                name=f"grid_s{axis_name}",
            ),
            H.make_node(
                "Reshape",
                [f"grid_s{axis_name}", reshape_const],
                [f"grid_rr{axis_name}"],
                name=f"grid_rr{axis_name}",
            ),
            H.make_node(
                "Expand",
                [f"grid_rr{axis_name}", "grid_full"],
                [f"grid_e{axis_name}"],
                name=f"grid_e{axis_name}",
            ),
            H.make_node(
                "Reshape",
                [f"grid_st{axis_name}", "grid_all1"],
                [f"grid_ms{axis_name}"],
                name=f"grid_ms{axis_name}",
            ),
        ]
    grid += [
        H.make_node("Concat", ["grid_ex", "grid_ey"], [base_grid_name], name="grid_base", axis=1),
        H.make_node("Concat", ["grid_msx", "grid_msy"], [mult_name], name="grid_mult", axis=1),
    ]

    int_out = {
        "grid_shape",
        "grid_h1",
        "grid_w1",
        "grid_h",
        "grid_w",
        "grid_full",
        "grid_rx",
        "grid_ry",
    }
    g.input[0].type.tensor_type.shape.dim[1].dim_value = 7
    g.input[0].type.tensor_type.elem_type = T.FLOAT16
    g.output[0].type.tensor_type.elem_type = T.FLOAT16

    # 3. Convert to FP16 everywhere except GridSample and its upstream grid arithmetic
    prod = {o: n for n in g.node for o in n.output}
    keep32 = set()
    for n in g.node:
        if n.op_type == "GridSample":
            keep32.add(n.name)
            stack = [n.input[1]]
            while stack:
                p = prod.get(stack.pop())
                if (
                    p is None
                    or p.name in keep32
                    or p.op_type
                    in ("Resize", "Conv", "ConvTranspose", "DepthToSpace", "Constant")
                ):
                    continue
                keep32.add(p.name)
                stack.extend(p.input)

    kind = {in_name: T.FLOAT16}
    for init in g.initializer:
        if init.data_type != T.FLOAT:
            kind[init.name] = None
            continue
        used32 = any(init.name in n.input and n.name in keep32 for n in g.node)
        used16 = any(init.name in n.input and n.name not in keep32 for n in g.node)
        if used32 and not used16:
            kind[init.name] = T.FLOAT
        elif used16 and not used32:
            init.CopyFrom(from_array(to_array(init).astype("float16"), init.name))
            kind[init.name] = T.FLOAT16
        else:
            kind[init.name] = T.FLOAT

    for n in grid:
        for o in n.output:
            if n.op_type == "Constant":
                kind[o] = T.FLOAT if n.attribute[0].t.data_type == T.FLOAT else None
            else:
                kind[o] = None if o in int_out else T.FLOAT

    out, casts = list(grid), {}

    def cast_tensor(t: str, to_type: int) -> str:
        key = (t, to_type)
        if key not in casts:
            casts[key] = f"{t}__cast{to_type}"
            out.append(H.make_node("Cast", [t], [casts[key]], name=casts[key], to=to_type))
        return casts[key]

    for n in g.node:
        if n.op_type == "Constant":
            kind[n.output[0]] = (
                T.FLOAT if n.attribute[0].t.data_type == T.FLOAT else None
            )
            out.append(n)
            continue
        want = T.FLOAT if n.name in keep32 else T.FLOAT16
        for ix, t in enumerate(n.input):
            if not t or kind.get(t) is None:
                continue
            if n.op_type == "Resize" and ix > 0:
                continue
            if kind[t] != want:
                n.input[ix] = cast_tensor(t, want)
        out.append(n)
        for o in n.output:
            if n.op_type == "Cast":
                to_attr = next((a.i for a in n.attribute if a.name == "to"), want)
                kind[o] = to_attr if to_attr in (T.FLOAT, T.FLOAT16) else None
            else:
                kind[o] = want

    oname = g.output[0].name
    if kind.get(oname) != T.FLOAT16:
        ren = oname + "_fp32"
        for n in out:
            for ix, o in enumerate(n.output):
                if o == oname:
                    n.output[ix] = ren
            for ix, i in enumerate(n.input):
                if i == oname:
                    n.input[ix] = ren
        out.append(H.make_node("Cast", [ren], [oname], name="output_cast", to=T.FLOAT16))

    del g.node[:]
    g.node.extend(out)
    del g.value_info[:]
    onnx.checker.check_model(m)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    onnx.save(m, dst)
    print(f"saved {dst} ({os.path.getsize(dst)} bytes, fp32_nodes={len(keep32)}, casts={len(casts)})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} SRC.onnx DST.onnx [FLOW_SCALE]", file=sys.stderr)
        sys.exit(2)
    convert(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
