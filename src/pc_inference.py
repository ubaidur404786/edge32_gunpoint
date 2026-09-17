"""Run the whole TEST split through PyTorch FP32, TFLite FP32 and TFLite INT8 on the PC.

Run from the project root:  python -m src.pc_inference
Writes results/pc_predictions.csv (per-sample outputs + predictions of all three) and
results/pc_summary.json. This CSV is the reference the board is compared against.
"""

import csv                # write the per-sample table
import json               # write the summary numbers
import numpy as np        # quantization arithmetic, argmax, comparisons
import torch              # PyTorch reference forward pass
import tensorflow as tf   # tf.lite.Interpreter for both .tflite files

import config                                                   # paths
from src.dataset import load_all, to_conv1d, to_channels_last   # test data in both layouts
from src.convert_tflite import load_pytorch_model, run_tflite_float, FP32_TFLITE, INT8_TFLITE


def quantize_input(x_float, scale, zero_point):
    """Float -> int8 exactly as the board will do it: q = rint(x / scale) + zp, clipped to int8.
    np.rint rounds half to even, which is what C's lrintf does in the default rounding mode."""
    q = np.rint(x_float / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def run_tflite_int8(tflite_bytes, X_nlc):
    """Run the int8 model one sample at a time. Returns raw int8 outputs (N, 2) and the
    scale/zero-point of input and output, read from the model, never hard-coded."""
     # BUILTIN_REF = TFLite's reference int8 kernels, no XNNPACK delegate. XNNPACK (the PC default)
    # differs from the reference by 1 LSB on requantization rounding boundaries (TEST sample 86);
    # the board reproduces the reference kernels, so the reference is the right PC baseline.
    interp = tf.lite.Interpreter(model_content=tflite_bytes,
                                 experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    if np.dtype(inp["dtype"]) != np.int8 or np.dtype(out["dtype"]) != np.int8:
        raise SystemExit("expected int8 input and output tensors")
    in_scale = float(inp["quantization_parameters"]["scales"][0])
    in_zp = int(inp["quantization_parameters"]["zero_points"][0])
    out_scale = float(out["quantization_parameters"]["scales"][0])
    out_zp = int(out["quantization_parameters"]["zero_points"][0])
    outs = []
    for i in range(len(X_nlc)):
        interp.set_tensor(inp["index"], quantize_input(X_nlc[i:i + 1], in_scale, in_zp))
        interp.invoke()
        outs.append(interp.get_tensor(out["index"])[0].copy())
    return np.stack(outs), (in_scale, in_zp, out_scale, out_zp)


def main():
    d = load_all()
    X, y = d["X_test"], d["y_test"]
    n = len(y)

    # 1. PyTorch FP32
    model, _ = load_pytorch_model()
    with torch.no_grad():
        pt = model(torch.from_numpy(to_conv1d(X))).numpy()

    # 2. TFLite FP32
    fp32 = run_tflite_float(FP32_TFLITE.read_bytes(), to_channels_last(X))

    # 3. TFLite INT8: raw int8 outputs; argmax is taken on the RAW int8 values.
    q8, (in_scale, in_zp, out_scale, out_zp) = run_tflite_int8(INT8_TFLITE.read_bytes(), to_channels_last(X))
    deq = (q8.astype(np.float32) - out_zp) * out_scale   # dequantized, for display only

    pred_pt, pred_fp32, pred_q8 = pt.argmax(1), fp32.argmax(1), q8.argmax(1)
    acc = lambda p: float((p == y).mean())
    agree = lambda a, b: int((a == b).sum())

    print(f"TEST split: {n} samples")
    print(f"input  quantization: scale={in_scale:.8g} zero_point={in_zp}")
    print(f"output quantization: scale={out_scale:.8g} zero_point={out_zp}")
    print(f"\naccuracy  PyTorch FP32 : {acc(pred_pt):.4f}  ({int((pred_pt == y).sum())}/{n})")
    print(f"accuracy  TFLite  FP32 : {acc(pred_fp32):.4f}  ({int((pred_fp32 == y).sum())}/{n})")
    print(f"accuracy  TFLite  INT8 : {acc(pred_q8):.4f}  ({int((pred_q8 == y).sum())}/{n})")
    print(f"\nagreement PyTorch vs TFLite FP32 : {agree(pred_pt, pred_fp32)}/{n}   max abs logit diff {np.abs(pt - fp32).max():.3e}")
    print(f"agreement TFLite FP32 vs INT8    : {agree(pred_fp32, pred_q8)}/{n}   max abs diff FP32 vs dequantized INT8 {np.abs(fp32 - deq).max():.4f}")
    print(f"agreement PyTorch vs INT8        : {agree(pred_pt, pred_q8)}/{n}")
    ties = int((q8[:, 0] == q8[:, 1]).sum())
    print(f"int8 output ties (out0 == out1)  : {ties}   (argmax picks index 0 on a tie; the board must do the same)")

    pred_pt, pred_fp32, pred_q8 = pt.argmax(1), fp32.argmax(1), q8.argmax(1)
    acc = lambda p: float((p == y).mean())
    agree = lambda a, b: int((a == b).sum())

    print(f"TEST split: {n} samples")
    print(f"input  quantization: scale={in_scale:.8g} zero_point={in_zp}")
    print(f"output quantization: scale={out_scale:.8g} zero_point={out_zp}")
    print(f"\naccuracy  PyTorch FP32 : {acc(pred_pt):.4f}  ({int((pred_pt == y).sum())}/{n})")
    print(f"accuracy  TFLite  FP32 : {acc(pred_fp32):.4f}  ({int((pred_fp32 == y).sum())}/{n})")
    print(f"accuracy  TFLite  INT8 : {acc(pred_q8):.4f}  ({int((pred_q8 == y).sum())}/{n})")
    print(f"\nagreement PyTorch vs TFLite FP32 : {agree(pred_pt, pred_fp32)}/{n}   max abs logit diff {np.abs(pt - fp32).max():.3e}")
    print(f"agreement TFLite FP32 vs INT8    : {agree(pred_fp32, pred_q8)}/{n}   max abs diff FP32 vs dequantized INT8 {np.abs(fp32 - deq).max():.4f}")
    print(f"agreement PyTorch vs INT8        : {agree(pred_pt, pred_q8)}/{n}")
    ties = int((q8[:, 0] == q8[:, 1]).sum())
    print(f"int8 output ties (out0 == out1)  : {ties}   (argmax picks index 0 on a tie; the board must do the same)")

    # Per-sample table: the reference for compare.py.
    config.RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = config.RESULTS_DIR / "pc_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true", "pt_out0", "pt_out1", "pt_pred", "fp32_out0", "fp32_out1", "fp32_pred",
                    "int8_out0", "int8_out1", "int8_pred", "int8_deq0", "int8_deq1"])
        for i in range(n):
            w.writerow([i, int(y[i]), f"{pt[i,0]:.6f}", f"{pt[i,1]:.6f}", int(pred_pt[i]),
                        f"{fp32[i,0]:.6f}", f"{fp32[i,1]:.6f}", int(pred_fp32[i]),
                        int(q8[i, 0]), int(q8[i, 1]), int(pred_q8[i]), f"{deq[i,0]:.4f}", f"{deq[i,1]:.4f}"])

    summary = {
        "n_test": n,
        "input_scale": in_scale, "input_zero_point": in_zp,
        "output_scale": out_scale, "output_zero_point": out_zp,
        "acc_pytorch_fp32": acc(pred_pt), "acc_tflite_fp32": acc(pred_fp32), "acc_tflite_int8": acc(pred_q8),
        "agree_pt_fp32": agree(pred_pt, pred_fp32), "agree_fp32_int8": agree(pred_fp32, pred_q8),
        "agree_pt_int8": agree(pred_pt, pred_q8),
        "max_abs_diff_pt_fp32": float(np.abs(pt - fp32).max()),
        "max_abs_diff_fp32_int8deq": float(np.abs(fp32 - deq).max()),
        "int8_output_ties": ties,
        "model_int8_bytes": INT8_TFLITE.stat().st_size,
    }
    with open(config.RESULTS_DIR / "pc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote", csv_path, "and", config.RESULTS_DIR / "pc_summary.json")


if __name__ == "__main__":
    main()