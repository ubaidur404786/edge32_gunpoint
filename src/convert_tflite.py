"""PyTorch checkpoint -> ONNX -> onnx2tf SavedModel -> TFLite FP32, then verify vs PyTorch.

Run from the project root:  python -m src.convert_tflite
Writes models/model.onnx, models/saved_model/, models/model_fp32.tflite.
"""

import numpy as np        # compare outputs, max abs diff
import torch              # load the checkpoint, run the reference forward pass, export ONNX
import onnx               # structural check of the exported .onnx
import onnx2tf            # ONNX -> TensorFlow SavedModel (the bridge chosen in Step 0.3)
import tensorflow as tf   # tf.lite.TFLiteConverter and tf.lite.Interpreter

import config                                                   # paths
from src.dataset import load_all, to_conv1d, to_channels_last   # test samples in both layouts
from src.model import GunPointCNN                               # rebuild the architecture, then load weights

ONNX_PATH = config.MODELS_DIR / "model.onnx"
SAVED_MODEL_DIR = config.MODELS_DIR / "saved_model"
FP32_TFLITE = config.MODELS_DIR / "model_fp32.tflite"
INT8_TFLITE = config.MODELS_DIR / "model_int8.tflite"
N_VERIFY = 10   # test samples used for the numerical check (not for any selection)


def load_pytorch_model():
    """Rebuild GunPointCNN with the series length stored in the checkpoint and load the weights."""
    ckpt = torch.load(config.MODELS_DIR / "model_fp32.pt", map_location="cpu")
    model = GunPointCNN(ckpt["series_length"]).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["series_length"]


def export_fp32(model, series_length):
    """PyTorch -> ONNX -> SavedModel -> model_fp32.tflite. Returns the tflite bytes."""
    # 1. ONNX. Fixed batch of 1: the board runs one sample at a time.
    dummy = torch.zeros(1, 1, series_length)
    torch.onnx.export(model, (dummy,), str(ONNX_PATH), input_names=["input"], output_names=["output"],
                      opset_version=18, dynamo=True)
    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
    print("1. ONNX      ->", ONNX_PATH)

    # 2. SavedModel via onnx2tf. Both flags are required (found in Step 0.3):
    #    tf_converter backend = classic ONNX->Keras->SavedModel path; output_signaturedefs = Keras 3 export.
    onnx2tf.convert(input_onnx_file_path=str(ONNX_PATH), output_folder_path=str(SAVED_MODEL_DIR),
                    tflite_backend="tf_converter", output_signaturedefs=True,
                    copy_onnx_input_output_names_to_tflite=True, verbosity="warn")
    print("2. SavedModel->", SAVED_MODEL_DIR)

    # 3. TFLite FP32: no optimizations, plain float graph.
    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
    tflite_bytes = converter.convert()
    FP32_TFLITE.write_bytes(tflite_bytes)
    print(f"3. TFLite    -> {FP32_TFLITE}  ({len(tflite_bytes)} bytes)")
    return tflite_bytes

def export_int8():
    """SavedModel -> fully int8 model_int8.tflite, calibrated on the TRAIN split only."""
    X_train = load_all()["X_train"]            # 40 rows of the train split; never val, never TEST
    X_rep = to_channels_last(X_train)          # TFLite layout (N, 150, 1), float32

    def representative_dataset():
        # The converter runs each sample through the float graph and records activation ranges.
        for i in range(len(X_rep)):
            yield [X_rep[i:i + 1]]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]              # enable quantization
    converter.representative_dataset = representative_dataset        # activation calibration data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # fail instead of falling back to float ops
    converter.inference_input_type = tf.int8                          # input tensor is int8 (no QUANTIZE op)
    converter.inference_output_type = tf.int8                         # output tensor is int8 (no DEQUANTIZE op)
    # Chirale_TensorFLowLite 2.0.0's FULLY_CONNECTED kernel supports ONE weight scale per layer
    # (it reads filter->params.scale = channel 0's scale). TF >= 2.17 quantizes dense layers per
    # channel by default, which made out1 wrong on the board in Step 8 run 1. Force per-tensor.
    converter._experimental_disable_per_channel_quantization_for_dense_layers = True
    tflite_bytes = converter.convert()
    INT8_TFLITE.write_bytes(tflite_bytes)
    print(f"4. TFLite int8 -> {INT8_TFLITE}  ({len(tflite_bytes)} bytes, calibrated on {len(X_rep)} TRAIN rows)")

    # Print the input/output quantization parameters right away; the board must use these exact values.
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    for kind, d in (("input", inp), ("output", out)):
        q = d["quantization_parameters"]
        print(f"   {kind}: dtype={np.dtype(d['dtype']).name} shape={tuple(int(v) for v in d['shape'])} "
              f"scale={float(q['scales'][0]):.8g} zero_point={int(q['zero_points'][0])}")
    return tflite_bytes

def run_tflite_float(tflite_bytes, X_nlc):
    """Run a float .tflite one sample at a time. X_nlc has the TFLite layout (N, L, 1)."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    expected = tuple(int(v) for v in inp["shape"])
    if expected != (1,) + X_nlc.shape[1:]:
        raise SystemExit(f"TFLite input shape is {expected}, data is {(1,) + X_nlc.shape[1:]} - layout assumption wrong")
    outs = []
    for i in range(len(X_nlc)):
        interp.set_tensor(inp["index"], X_nlc[i:i + 1])
        interp.invoke()
        outs.append(interp.get_tensor(out["index"])[0].copy())
    return np.stack(outs)


def verify_fp32(model, tflite_bytes):
    """Same N_VERIFY test samples through PyTorch and TFLite FP32: max abs diff and agreement."""
    d = load_all()
    X = d["X_test"][:N_VERIFY]
    with torch.no_grad():
        ref = model(torch.from_numpy(to_conv1d(X))).numpy()
    got = run_tflite_float(tflite_bytes, to_channels_last(X))
    diff = np.abs(ref - got).max()
    agree = int((ref.argmax(1) == got.argmax(1)).sum())
    print(f"\nverify on {N_VERIFY} test samples:")
    print(f"  max abs diff of raw logits : {diff:.3e}")
    print(f"  prediction agreement       : {agree}/{N_VERIFY}")
    print("  first 3 rows PyTorch:", np.array2string(ref[:3], precision=5))
    print("  first 3 rows TFLite :", np.array2string(got[:3], precision=5))
    return diff, agree


if __name__ == "__main__":
    config.MODELS_DIR.mkdir(exist_ok=True)
    model, L = load_pytorch_model()
    fp32_bytes = export_fp32(model, L)
    verify_fp32(model, fp32_bytes)
    export_int8()