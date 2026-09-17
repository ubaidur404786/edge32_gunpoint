"""Print what is really inside a .tflite file: ops, tensors, dtypes, quantization params.

Run from the project root:  python -m src.inspect_tflite models/model_fp32.tflite
"""

import sys                # the .tflite path comes from the command line
import numpy as np        # summarize per-channel scale arrays
import tensorflow as tf   # tf.lite.Interpreter (tensor details) and the model Analyzer (op list)


def describe_quant(q):
    """One-line summary of a tensor's quantization parameters (empty for float tensors)."""
    scales, zps, axis = q.get("scales", []), q.get("zero_points", []), q.get("quantized_dimension", 0)
    if len(scales) == 0:
        return "-"
    if len(scales) == 1:
        return f"scale={float(scales[0]):.8g} zero_point={int(zps[0])}"
    return (f"per-channel axis={axis}, {len(scales)} scales "
            f"[{float(np.min(scales)):.4g} .. {float(np.max(scales)):.4g}], zero_points all "
            f"{int(zps[0]) if np.all(np.asarray(zps) == zps[0]) else 'mixed'}")


def main(path):
    with open(path, "rb") as f:
        content = f.read()
    print(f"file: {path}  ({len(content)} bytes)\n")

    # Op list with tensor shapes, from TensorFlow's own analyzer (reads the flatbuffer).
    print("=== operators (tf.lite.experimental.Analyzer) ===")
    tf.lite.experimental.Analyzer.analyze(model_content=content)

    interp = tf.lite.Interpreter(model_content=content)
    interp.allocate_tensors()

    print("\n=== input / output ===")
    for kind, details in (("input", interp.get_input_details()), ("output", interp.get_output_details())):
        for d in details:
            print(f"  {kind:6s} #{d['index']:<3d} {d['name']:24s} shape={tuple(int(v) for v in d['shape'])} "
                  f"dtype={np.dtype(d['dtype']).name:8s} {describe_quant(d['quantization_parameters'])}")

    print("\n=== all tensors ===")
    for t in interp.get_tensor_details():
        print(f"  #{t['index']:<3d} {t['name'][:40]:40s} shape={tuple(int(v) for v in t['shape'])!s:22s} "
              f"dtype={np.dtype(t['dtype']).name:8s} {describe_quant(t['quantization_parameters'])}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.inspect_tflite <model.tflite>")
    main(sys.argv[1])
