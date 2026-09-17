# Tutorial: GunPoint on an ESP32-S3 with TensorFlow Lite Micro

One section per step. Every number in this file was measured on this machine; if a number
is missing it has not been measured yet.

## Step 0.1 - CLAUDE.md

CLAUDE.md is the project's instruction and status file. It holds the working rules, the
fixed facts about this machine and board, the plan, the decisions taken so far and a status
line per step with the key measured number. It is the first thing to read when returning
to the project.

## Step 0.2 - Python environment

A **venv** (virtual environment) is a private folder (`.venv/`) with its own Python and
packages, so nothing we install affects other projects. We need PyTorch (to train),
TensorFlow (for the `tf.lite` converter and interpreter), ONNX (the exchange format between
them) and NumPy, all in one place.

Installed and verified with a smoke test (Python 3.12.10, Windows 11):

| package    | version    | role                                        |
| ---------- | ---------- | ------------------------------------------- |
| torch      | 2.14.0+cpu | define, train and export the model          |
| tensorflow | 2.21.0     | TFLite converter + TFLite interpreter on PC |
| onnx       | 1.20.1     | model exchange format PyTorch -> onnx2tf    |
| numpy      | 2.2.6      | arrays everywhere                           |

Torch and TensorFlow coexist in one venv with no `numpy` / `protobuf` conflict, so a second
venv was not needed. `requirements.txt` pins exactly these versions; `pip install -r
requirements.txt` must print only "Requirement already satisfied".

## Step 0.3 - Choosing the PyTorch -> TFLite bridge

PyTorch cannot write `.tflite` files. Three routes were considered, in order of preference,
and tested rather than assumed:

**(a) ai-edge-torch (Google's direct converter) - ruled out on Windows.**
`pip install --dry-run ai-edge-torch` ends in `ResolutionImpossible`: its dependencies
`litert-converter` and `ai-edge-tensorflow` have no Windows distributions.

**(b) torch.onnx.export -> onnx2tf -> SavedModel -> tf.lite.TFLiteConverter - CHOSEN.**
Terms: **ONNX** is an open, framework-neutral file format for neural networks that PyTorch
can export. **onnx2tf** turns the ONNX file into a TensorFlow **SavedModel** (TensorFlow's
on-disk model folder). `tf.lite.TFLiteConverter` then reads the SavedModel and writes the
`.tflite` **flatbuffer** (a compact binary format that can be read in place, without
parsing, which is why microcontrollers use it).

Three things had to be discovered along the way:

1. onnx2tf 2.x pins its own dependency versions. Installing it downgraded numpy 2.5.3 ->
   2.2.6, onnx 1.22.0 -> 1.20.1, protobuf 7.36.1 -> 7.35.1 and ml_dtypes 0.6.0 -> 0.5.4.
   All four stay inside the ranges torch and tensorflow allow; torch and tensorflow
   themselves were untouched. The downgrade was accepted deliberately to keep one venv.
2. onnx2tf 2.x by default uses a new "flatbuffer_direct" backend that builds `.tflite`
   files straight from ONNX without TensorFlow and writes **no SavedModel**. We need the
   SavedModel because the INT8 conversion in Step 5 must go through
   `tf.lite.TFLiteConverter` with our own representative dataset. Fix:
   `tflite_backend="tf_converter"`, which also requires the package `tf-keras==2.21.0`.
3. On that backend, with Keras 3, onnx2tf's plain `tf.saved_model.save()` call fails and
   the failure is only logged at info level - the folder simply has no `saved_model.pb`.
   Fix: `output_signaturedefs=True`, which makes onnx2tf use Keras 3's `ExportArchive`
   and write a proper SavedModel with a `serving_default` signature.

Proof (`src/bridge_test.py`, a temporary Conv1d -> ReLU -> MaxPool1d -> Linear model with
input length 32, random weights):

TFLite input shape: (1, 32, 1)
PyTorch output: [[0.33820856 -0.12703004]]
TFLite output: [[0.33820862 -0.12703001]]
max abs diff : 5.96e-08
argmax agree : True

Two lessons from that output that matter later:

- **Layout change.** PyTorch is channels-first: `(batch, channels, length)` = `(1, 1, 32)`.
  TFLite is channels-last: `(batch, length, channels)` = `(1, 32, 1)`. onnx2tf does the
  swap for us; the PC test code and the board code must feed data in the TFLite order.
- **A difference of ~1e-7 is float rounding, not a bug.** A wrong conversion shows up as
  differences of 0.1 or more and disagreeing argmax.

**(c) Rebuild in Keras and copy weights by hand** - not needed; kept in CLAUDE.md as the
fallback description only.

`src/bridge_test.py` and `models/bridge_test/` are throwaway and will be deleted once the
real model converts in Step 4.

## Step 0.4 - Arduino toolchain and TensorFlow Lite Micro library

**arduino-cli.** Arduino IDE 2.x ships its own `arduino-cli.exe`
(`C:\Users\HP\AppData\Local\Programs\Arduino IDE\resources\app\lib\backend\resources`). That
folder was added to the user PATH, so the CLI and the IDE are the same binary and share the
same `Arduino15` package folder. Version: `arduino-cli 1.5.1`.

**Core.** `arduino-cli core list` -> `esp32:esp32 3.3.11`. A _core_ is the board support
package (compiler, libraries, board definitions) for one chip family.

**FQBN.** The board menu shows "PandaByte xS3 - ESP32S3"; `arduino-cli board listall` gives
its Fully Qualified Board Name `esp32:esp32:pandabyte_xs3`. Menu options are appended as
`key=value`. `arduino-cli board details` showed the USB CDC On Boot bug: two entries both
labelled "Enabled", `CDCOnBoot=default` (first, default) and `CDCOnBoot=cdc`. Only the first
one gives serial output on this board, so every command uses:

    esp32:esp32:pandabyte_xs3:CDCOnBoot=default

All other options stay at their defaults (240 MHz, 4 MB flash, default partition with a
1.2 MB app slot, PSRAM disabled). Port: COM11.

**TFLM library.** `Chirale_TensorFLowLite 2.0.0` (Library Manager; note the capital L in
"FLow"). Findings from reading its source, not its README:

- Modern TFLM API: `MicroInterpreter(model, resolver, arena, arena_size)`,
  `MicroMutableOpResolver<N>` with `AddConv2D`, `AddFullyConnected`, `AddMaxPool2D`,
  `AddReshape`, `AddExpandDims`, `AddTranspose`, `AddQuantize`, `AddDequantize`, ...;
  `arena_used_bytes()` is available.
- No ESP-NN (Espressif's optimized kernels) anywhere in the library.
- Conv, fully-connected and pooling kernels are shipped only in their CMSIS-NN form. On the
  Xtensa compiler the ARM DSP/MVE macros are undefined, so CMSIS-NN's portable plain-C
  branches are compiled: no assembly, but strictly speaking not the TFLite "reference"
  kernels either. All other ops are plain reference kernels. The sample-by-sample check
  against the PC in Step 8 is the safeguard; if it fails, this is suspect number one.

**Proof sketch** `firmware/tflm_smoke/` (temporary): loads the library's hello_world sine
model, registers 4 ops, allocates an 8 KB tensor arena, runs one `Invoke()`.

Terms: the **op resolver** is the table that maps each operator in the model to the C++
kernel that implements it; registering only the needed ops keeps flash small. The **tensor
arena** is one static byte buffer that TFLM carves up for input, output and every
intermediate tensor - TFLM never calls `malloc`, so a too-small arena makes
`AllocateTensors()` fail.

Commands (first build needs `--jobs 1`; with the default parallel jobs the compiler ran out
of memory: `cc1plus.exe: out of memory`. Later builds are cached and fast):

    arduino-cli compile --jobs 1 --fqbn esp32:esp32:pandabyte_xs3:CDCOnBoot=default firmware\tflm_smoke
    arduino-cli upload  --fqbn esp32:esp32:pandabyte_xs3:CDCOnBoot=default -p COM11 firmware\tflm_smoke
    arduino-cli monitor -p COM11 -c baudrate=115200

Measured:

    Sketch uses 337349 bytes (25%) of program storage space. Maximum is 1310720 bytes.
    Global variables use 31928 bytes (9%) of dynamic memory, leaving 295752 bytes.

    #BANNER,tflm_smoke,Chirale_TensorFLowLite 2.0.0
    #DEVICE,chip=ESP32-S3,cpu_mhz=240,heap_size=385184,free_heap=348568
    #INFO,model_len=2488,arena_used=748,arena_size=8192,output0_int8=5
    TFLM OK

So the dummy model needs 748 bytes of arena; the full TFLM runtime plus the ESP32 core
costs about 330 KB of flash before our own model is added. That leaves roughly 970 KB of
the app partition for the GunPoint model and its test data.

## Step 1 - Data

Files: `config.py` (every setting), `src/dataset.py` (loading, label mapping, split).
Scripts are run from the project root as modules: `python -m src.dataset`. That puts the
project root on the import path, so `import config` works everywhere without path tricks.

**Format.** Each `.tsv` row is: label, then the time series, tab separated. GunPoint labels
are 1 (Gun) and 2 (Point); we map them to 0 and 1 because PyTorch's `CrossEntropyLoss`
expects class indices starting at 0.

**Measured** (`results/data_stats.json`):

| split | rows | length | class 0 (Gun) | class 1 (Point) |
| ----- | ---- | ------ | ------------- | --------------- |
| TRAIN | 50   | 150    | 24            | 26              |
| TEST  | 150  | 150    | 76            | 74              |

Per-series mean: within +/-5e-8 of 0 in every row. Per-series standard deviation: exactly
0.996661 in every row of both files. That value is sqrt(149/150): the dataset authors
normalized each series to unit _sample_ standard deviation (divide by N-1 = 149) while
`np.std` divides by N = 150. Two hundred identical standard deviations can only mean that
every series was z-normalized individually before publication.

**Decision: no preprocessing.** Every preprocessing step (scaling, filtering, mean removal)
would have to be re-implemented in C on the board, bit for bit, and each is another place
for a silent mismatch between PC and board. Since the data is already normalized, the model
consumes the file values directly as float32. The only transform that will ever happen on
the board is the int8 quantization dictated by the model itself (Step 5).

**Shape.** PyTorch's `Conv1d` expects `(batch, channels, length)`. A time series with one
value per time step has one channel, so a batch of rows becomes `(batch, 1, 150)`.
`dataset.to_conv1d()` does exactly that: `X[:, None, :]`.

**Split.** The official TRAIN file is split once, stratified (same class ratio in both
parts), with `SEED = 42` and `VAL_FRACTION = 0.2`: 40 rows for fitting (19/21) and 10 rows
for validation (5/5). Ten validation rows means one sample is 10 percentage points, so the
validation accuracy is very noisy. Consequence: model selection is limited to early stopping
on validation loss; no hyperparameter search. The TEST file is never touched until the end
of Step 3, and then exactly once.

## Step 2 - Model

File: `src/model.py`. Architecture (all sizes from `config.py`):

    Conv1d(1->8, k=3) -> ReLU -> MaxPool1d(2) -> Conv1d(8->16, k=3) -> ReLU -> MaxPool1d(2)
    -> Flatten -> Linear(576 -> 2)

Shape after every layer, printed with forward hooks (batch of 1):

    input        (1, 1, 150)
    conv1        (1, 8, 148)     150 - 3 + 1 = 148   (no padding, so the series shrinks by k-1)
    relu         (1, 8, 148)
    pool1        (1, 8, 74)      148 // 2 = 74
    conv2        (1, 16, 72)     74 - 3 + 1 = 72
    relu         (1, 16, 72)
    pool2        (1, 16, 36)     72 // 2 = 36
    flatten      (1, 576)        16 channels x 36 = 576
    classifier   (1, 2)          two raw logits

Trainable parameters, measured: conv1 24 + 8, conv2 384 + 16, linear 1152 + 2 = **1586**.

Design choices, and why they matter for the board:

- **No Softmax.** `CrossEntropyLoss` applies log-softmax internally, so the model outputs
  raw logits. Softmax does not change the order of the values, so `argmax` over logits gives
  the same class. Leaving it out removes one quantized SOFTMAX op and one int8 rounding step
  on the board.
- **No BatchNorm.** For 1586 parameters it adds nothing; it behaves differently in train and
  eval mode and the converter would fold it into the conv weights anyway.
- **No Dropout** unless training shows clear overfitting (Step 3 decides).
- **No padding** ("valid" convolutions): no PAD op in the TFLite graph.
- **Flatten order.** PyTorch flattens (channels, length) channel-major. TFLite is
  channels-last, so the converter must insert a TRANSPOSE before the RESHAPE so that the
  576 Linear inputs keep their meaning. Step 4 checks that this happened.

## Step 3 - Training

File: `src/train.py`. Adam (lr 1e-3), `CrossEntropyLoss`, batch size 8, at most 200 epochs,
early stopping when the validation loss has not improved for 20 epochs, keeping the weights
of the best-validation-loss epoch. Seeds are fixed for the split, the weight initialization
and the batch order, so the run is reproducible.

Measured (`results/train_metrics.json`):

    early stop at epoch 97, best epoch 77: val_loss 0.2050, val_acc 0.900
    FINAL TEST (150 rows, evaluated once): loss 0.4534, accuracy 0.7733  (116 / 150)

**Why TEST must not be used for selection.** Every time you look at a test score and then
change something (epochs, seed, layer width, even "just one more run"), the test set stops
being unseen data and becomes part of training. The number you report afterwards is no
longer an estimate of how the model does on new data; it is an estimate of how well you
searched. With 150 test rows the temptation is large, because a lucky seed can easily move
the score by several points. So: model selection uses only the 10 validation rows; TEST is
evaluated once, at the end, and that single number is the result. Here it is 0.7733, and it
stays 0.7733 for the rest of the project.

**What the curve says.** Train loss was still 0.15 when training stopped and validation
loss never turned upward (it wobbled between 0.2 and 0.3 for sixty epochs), so this is not
overfitting and Dropout is not added. The gap between validation loss (0.205) and test loss
(0.453) is the 10-row validation set being too small to estimate anything precisely, exactly
as predicted in Step 1. A better model would be a separate experiment with a proper protocol
(for example cross-validation inside TRAIN); it is not part of this deployment path, whose
goal is that the board reproduces the PC model exactly, whatever its accuracy.

Outputs: `models/model_fp32.pt` (state_dict + series length) and `results/train_metrics.json`
(per-epoch history and the final numbers).

## Step 4 - TFLite FP32 conversion and verification

Files: `src/convert_tflite.py` (conversion + numerical check), `src/inspect_tflite.py`
(prints what is really inside any `.tflite`). Helper `to_channels_last()` added to
`src/dataset.py`.

Route: `model_fp32.pt` -> `torch.onnx.export` (dynamo, opset 18) -> `models/model.onnx` ->
onnx2tf (`tflite_backend="tf_converter"`, `output_signaturedefs=True`) ->
`models/saved_model/` -> `tf.lite.TFLiteConverter.from_saved_model` -> `models/model_fp32.tflite`.

**Layout change.** PyTorch is channels-first `(N, C, L)` = `(1, 1, 150)`. TFLite is
channels-last `(N, L, C)`, and the converted model's input is `(1, 150, 1)`. This was read
from the file with `inspect_tflite.py`, not assumed. On the PC and on the board the data
is fed in that order; for one channel it is the same 150 numbers, only the declared shape
differs.

**No CONV_1D in TFLite.** The 1-D convolutions became `CONV_2D` over tensors with height 1,
e.g. `(1, 1, 148, 8)`. The full operator list of `model_fp32.tflite` (9 ops, 5 types):

    Op#0 RESHAPE          (1,150,1) -> (1,1,150,1)       add the height-1 axis
    Op#1 CONV_2D          -> (1,1,148,8)                 conv1, ReLU fused into the op
    Op#2 MAX_POOL_2D      -> (1,1,74,8)
    Op#3 CONV_2D          -> (1,1,72,16)                 conv2, ReLU fused
    Op#4 MAX_POOL_2D      -> (1,1,36,16)
    Op#5 RESHAPE          -> (1,36,16)                   drop the height axis
    Op#6 TRANSPOSE [0,2,1]-> (1,16,36)                   back to channel-major ...
    Op#7 RESHAPE          -> (1,576)                     ... so this flatten matches PyTorch
    Op#8 FULLY_CONNECTED  -> (1,2)                       the Linear layer

ReLU does not appear as its own op: the converter fuses it into CONV_2D as the op's
"activation". The TRANSPOSE is the converter preserving PyTorch's channel-major flatten
order, so that the 576 Linear weights keep their meaning (see Step 2).

Measured: `model_fp32.tflite` is 10092 bytes (6504 bytes of weights, the rest is graph
structure); about 35 000 multiply-accumulates per inference.

**Verification** (10 test samples through PyTorch and through the TFLite interpreter):

    max abs diff of raw logits : 2.146e-06
    prediction agreement       : 10/10

A difference of 2e-6 on logits of magnitude 2-3 is float32 rounding from a different
summation order. A wrong conversion (bad layout, wrong flatten order) shows up as
differences of order 1 and disagreeing predictions.

## Step 5 - INT8 quantization and PC verification

Files: `src/convert_tflite.py` (function `export_int8`), `src/pc_inference.py`.

**Why int8.** Weights and activations stored as 8-bit integers make the model about 4x
smaller and let the interpreter use integer arithmetic only, which is what TFLite Micro's
kernels are optimized for. Accuracy can drop; it is measured, not assumed.

**Scale and zero-point.** A real value x is stored as q = round(x / scale) + zero_point and
recovered as x ~ (q - zero_point) \* scale. `scale` is the size of one integer step,
`zero_point` is the integer that stands for 0.0. Activations get one pair per tensor;
conv and FC weights get one scale per output channel ("per-channel"), zero-point 0.

**Representative dataset.** Weight ranges are known from the weights themselves, but the
range of every activation (output of conv1, pool1, ...) is only known by running real data
through the model. The converter ran the 40 rows of the train split (never validation,
never TEST) and recorded min/max of every tensor to choose the scales.

**Converter settings:** `optimizations=[DEFAULT]`, the representative dataset,
`supported_ops=[TFLITE_BUILTINS_INT8]` (fail instead of silently keeping float ops),
`inference_input_type = inference_output_type = int8` (so no QUANTIZE / DEQUANTIZE ops are
added at the ends; the caller quantizes the input and reads raw int8 output).

Measured, from the file (`inspect_tflite.py`):

    model_int8.tflite: 6512 bytes (1824 bytes weights, 4688 bytes graph structure)
    input  : int8 (1,150,1)  scale 0.017343521  zero_point 9
    output : int8 (1,2)      scale 0.041003779  zero_point 8
    conv1 weights (8,1,3,1)  per-channel, 8 scales;  conv2 (16,1,3,8) 16 scales;  FC (2,576) 2 scales
    ReLU outputs: zero_point -128 (the whole int8 range is used for non-negative values)
    ops: identical 9-op list as FP32, all tensors int8 (biases int32)

The FP32 file was 10092 bytes. Weights shrank about 3.6x; the graph structure did not
shrink at all, which is why the file is not one quarter of the size.

**Input quantization on the PC** (`pc_inference.quantize_input`): scale and zero-point are
read from the interpreter's input details, never typed in; `q = np.rint(x / scale) +
zero_point`, clipped to [-128, 127]. `np.rint` rounds half to even, the same rule as C's
`lrintf` in its default mode, which the board will use. `argmax` is taken on the raw int8
outputs; dequantized values are written to the CSV for reading only.

**Measured on the full TEST split** (`results/pc_predictions.csv`, `results/pc_summary.json`):

    accuracy  PyTorch FP32 : 0.7733 (116/150)
    accuracy  TFLite  FP32 : 0.7733 (116/150)
    accuracy  TFLite  INT8 : 0.7733 (116/150)      no drop from quantization
    agreement PyTorch vs TFLite FP32 : 150/150   max abs logit diff 2.9e-06
    agreement TFLite FP32 vs INT8    : 150/150
    int8 output ties                 : 0

**About the 1.378 max difference between FP32 and dequantized INT8.** It is saturation, not
arithmetic error. TEST logits reach -6.85 and +6.26, but the output scale was calibrated
on 40 training rows with smaller logits, so int8 can only express -5.58 .. +4.88. Four of
the 300 outputs are clipped at 127 or -128. Among the other 296 the max difference is
0.070 (about 1.7 int8 steps) and the median 0.0135. Clipping affects only the most
confident logits, so no prediction changes.

**Sensitivity for the board.** The smallest |out0 - out1| margin on TEST is exactly 1 int8
unit. The board therefore has to reproduce the PC's integer arithmetic bit for bit; a
kernel that rounds differently would flip at least that sample. This is what Steps 8 and 9
test, sample by sample.

## Step 6 - TFLite Micro operator support

TFLite Micro does not ship every operator to every board; the sketch registers the kernels
it needs in a `MicroMutableOpResolver`. If an op in the model has no registered kernel,
`AllocateTensors()` fails on the board. So each of the 5 op types found in
`model_int8.tflite` was checked against the kernel sources of Chirale_TensorFLowLite 2.0.0
(`src/tensorflow/lite/micro/kernels/`):

| op               | instances | kernel file                   | int8 evidence                                |
|------------------|-----------|-------------------------------|----------------------------------------------|
| RESHAPE          | 3         | reshape.cpp                   | memcpy of input bytes, any dtype             |
| CONV_2D (+ReLU)  | 2         | cmsis_nn/conv.cpp             | kTfLiteInt8 path, Register_CONV_2D()         |
| MAX_POOL_2D      | 2         | cmsis_nn/pooling.cpp          | kTfLiteInt8 path, Register_MAX_POOL_2D()     |
| TRANSPOSE        | 1         | transpose.cpp                 | `case kTfLiteInt8:` (line 100)               |
| FULLY_CONNECTED  | 1         | cmsis_nn/fully_connected.cpp  | kTfLiteInt8 path, Register_FULLY_CONNECTED() |

All supported; the model stays as it is. The resolver used by the firmware:

    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddReshape();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddTranspose();
    resolver.AddFullyConnected();

`<5>` is the number of distinct op types (capacity of the table), not the 9 op instances.
Nothing else is registered: there is no RELU op (fused into CONV_2D), no SOFTMAX (left out of
the model on purpose), no QUANTIZE / DEQUANTIZE (input and output are already int8).
(Step 8 later showed that this kernel supports only one weight scale per FC layer; see there.)

## Step 7 - Export for Arduino

File: `src/export_arduino.py`. The board has no file system access in this project, so the
model and the test samples are compiled into the firmware as C arrays. Windows has no
`xxd`, so Python writes the headers.

- `firmware/gunpoint_tflm/model_data.h`: `alignas(16) const unsigned char g_model[]`
  (6512 bytes; `alignas(16)` because TFLM reads the flatbuffer in place and wants it aligned),
  `g_model_len`, and `G_MODEL_SHA256`, the SHA-256 of those bytes:
  `780e693b2028a92eabaa6df9811857c8bcbdbaf07a1965ec8bb3e6167cc8b7a5`.
- `firmware/gunpoint_tflm/test_data.h`: `kSeriesLength = 150`, `kNumSamples = 5`,
  `kClassNames`, `kTestIndex` (position in the TEST split = the `idx` column of
  `pc_predictions.csv`), `kTestLabels` = {0, 1, 1, 0, 0}, and `kTestData[5][150]` as
  floats with 9 significant digits and an `f` suffix, which reproduces the exact float32
  values the PC used.
- `results/export_manifest.json`: byte count, SHA-256, sample count and indices, so that
  `compare.py` can check what the board reports against what was exported.

Why floats, not pre-quantized int8: the board must quantize the input itself with the
scale and zero-point it reads from the model, so that step is part of what gets tested.

Why a SHA-256: a 32-byte fingerprint that changes if a single byte differs. In Step 8 the
board hashes the model bytes it actually holds in flash and prints the result; if it
matches the manifest, the firmware runs exactly the file the PC verified in Step 5.

`python -m src.export_arduino --all` writes all 150 samples for Step 9.
## Step 8 - Firmware, first 5 samples

Files: `firmware/gunpoint_tflm/board_config.h` (baud, serial wait, boot delay, arena size,
warm-up count - everything board-specific), `firmware/gunpoint_tflm/gunpoint_tflm.ino`.

What the sketch does, in order: print `#META` with the SHA-256 it computes from the model
bytes in its own flash (mbedtls, part of the ESP32 core) next to the expected value from
`model_data.h`, and stop with `#ERROR` if they differ; print `#DEVICE` (chip, clock, heap,
sketch size, PSRAM - all read at run time); build a `MicroMutableOpResolver<5>` with exactly
the 5 op types of Step 6; create the `MicroInterpreter` over a 32 KB tensor arena, call
`AllocateTensors()` and print `#ARENA,used=..`; check that the input tensor is int8 with dims
(1, 150, 1) and the output int8 with dims (1, 2); read scale and zero-point from
`input->params` / `output->params` and print `#QUANT` (nothing is hard-coded); run 3 untimed
warm-up inferences; then for every sample quantize the floats with
`lrintf(x / scale) + zero_point` clipped to [-128, 127], time only `Invoke()` with
`micros()`, take argmax over the raw int8 outputs with a strict `>` (a tie goes to index 0,
like `np.argmax`), and print `#RES,idx,true,pred,latency_us,out0,out1`; finally `#SUMMARY`,
`#LATENCY`, `#MEMORY`, `#END`.

Build and flash:

    arduino-cli compile --jobs 1 --fqbn esp32:esp32:pandabyte_xs3:CDCOnBoot=default firmware\gunpoint_tflm
    arduino-cli upload  --fqbn esp32:esp32:pandabyte_xs3:CDCOnBoot=default -p COM11 firmware\gunpoint_tflm

Compile summary (5-sample build): 355717 bytes of flash (27 %), 56552 bytes of static RAM
(17 %), of which 32768 bytes are the arena.

### Run 1: the predictions matched and the outputs did not

    board:  (61,-52) (-34,27) (-71,91) (56,-50) (8,-17)     all 5 predictions correct
    PC:     (61,-46) (-34,25) (-71,84) (56,-45) (8,-14)

`out0` was identical on every sample and `out1` was wrong on every sample by a factor of
about 1.10 - and 1.10 is the ratio between the two per-channel scales of the FC weights
(0.002471 / 0.002248). Reading the library source confirmed it: this TFLM snapshot computes
one output multiplier per FULLY_CONNECTED layer from `filter->params.scale`, which for a
per-channel tensor is channel 0's scale (`fully_connected_common.cpp`, and
`cmsis_nn/fully_connected.cpp` copies that single value into every channel slot). CONV_2D
handles per-channel scales correctly, which is why everything up to `out0` was bit-exact.
TensorFlow >= 2.17 quantizes dense layers per channel by default, so this combination fails
silently. A prediction-only check would have passed it.

Fix (in `convert_tflite.py`): `converter._experimental_disable_per_channel_quantization_for_dense_layers = True`,
then Steps 5 -> 7 -> 8 again. New model: 6496 bytes, FC weights one scale (0.0024710435),
PC INT8 accuracy unchanged at 0.7733, 150/150 agreement with FP32. The failed capture is
kept as `logs/run_5samples_perchannel_fc_FAIL.txt`.

### Run 2: bit-exact

    #META   sha256 a0614175...5631 == expected, model_len 6496, series_length 150, n_samples 5
    #DEVICE chip ESP32-S3, 240 MHz, heap_size 360560, free_heap 323356, sketch_size 355872, psram 0
    #ARENA  used 3404 of 32768
    #QUANT  in_scale 0.0173435211, in_zp 9, out_scale 0.0410037786, out_zp 8   (== PC)
    #RES    (61,-46) (-34,25) (-71,84) (56,-44) (8,-14)   == PC, 5/5 predictions
    #LATENCY mean 2885 us, min 2819, max 2905

### Capturing a complete log on this board

COM11 is the ESP32-S3's native USB-Serial/JTAG port. Until the host has actually talked to
it, its driver discards what the sketch prints; `Serial.flush()` and a 2 s boot delay do not
change that, so a boot-time report always loses its first lines. The sketch therefore
re-runs the complete report whenever it receives a byte, and the capture command sends one:

    cmd /c "(ping -n 4 127.0.0.1 >nul & echo. & ping -n 11 127.0.0.1 >nul) | arduino-cli monitor -p COM11 -c baudrate=115200 > logs\run_5samples.txt 2>&1"

(connect, wait 3 s, send ENTER, record 10 s). Windows PowerShell 5.1 cannot stream a
pipeline into `arduino-cli monitor` - the output stays empty - while `cmd /c` and Git Bash
can. `Tee-Object` writes UTF-16 files; `compare.py` decodes either encoding and uses the last
complete `#META..#END` block, so a truncated boot block at the top of a log is harmless.

## Step 9 - Firmware, full TEST split

`python -m src.export_arduino --all` rewrites `test_data.h` with all 150 samples (the model
and its SHA-256 are unchanged), then compile, upload and capture to `logs/run_full.txt`.

Measured: flash 443169 bytes (33 %) - 87452 bytes more than the 5-sample build for 145 more
samples of 600 bytes each - and static RAM unchanged at 56552 bytes, because the test data
is `const` and stays in flash. Board: `#SUMMARY,correct=116,total=150,accuracy=0.7733`,
`#LATENCY,mean_us=2907.4,min_us=2812,max_us=2928`, arena used 3404 bytes, 150 `#RES` lines.

## Step 10 - Compare and report

`python -m src.compare logs/run_full.txt` checks the log against `export_manifest.json`
and joins it with `pc_predictions.csv`. First result: 7/7 integrity checks, 150/150
predictions, but 149/150 raw outputs - sample 86 had `out0` 61 on the board and 62 on the PC.

The PC interpreter had printed `Created TensorFlow Lite XNNPACK delegate for CPU`: the PC
was not running TFLite's reference int8 kernels but XNNPACK's optimized ones. Running the
same model with `experimental_op_resolver_type=BUILTIN_REF` (reference kernels, no delegate)
gives 61 on sample 86 and is identical to XNNPACK on the other 149. The board's kernels
(CMSIS-NN's portable C path) match the reference kernels exactly. One 8-bit step on a
requantization rounding boundary is the normal size of difference between two correct int8
implementations; it does not change a prediction here, but "bit-exact" has to name its
reference. `pc_inference.py` now builds the int8 interpreter with `BUILTIN_REF`.

Final `results/summary.txt`:

    integrity checks: 7/7 OK (sha256, model_len, n_samples, #RES count, series_length, input and output scale/zero-point)
    accuracy TEST: PyTorch FP32 0.7733, TFLite FP32 0.7733, TFLite INT8 PC 0.7733, TFLite INT8 board 0.7733 (116/150)
    board vs PC: predictions 150/150, raw int8 outputs 150/150, max |diff| 0
    latency (Invoke only, after warm-up): mean 2907.4 us, median 2908 us, min 2812 us, max 2928 us
    model_int8.tflite 6496 bytes; arena 3404 / 32768 bytes; heap 360560 (free 323356) bytes
    flash 443169 bytes (33 %) of 1310720; static RAM 56552 bytes (17 %) of 327680
    VERDICT: PASS

## What to change when reusing this template

The dependencies below are the ones that actually showed up in this project, not a wish list.

**New dataset (same board, same kind of model)**
- `config.py`: `DATA_DIR`, file names, `CLASS_NAMES`, `NUM_CLASSES`, possibly `VAL_FRACTION`.
- `src/dataset.py`: only if the file format differs (this one reads UCR-style TSV with the
  label in column 0). Re-run `python -m src.dataset` and re-decide preprocessing from the
  printed statistics; if preprocessing is needed it must be written three times identically
  (dataset.py, pc_inference.py, the .ino).
- Series length is read from the data everywhere, but it changes the Linear layer's input
  size (`model.flat_features`), the `.tflite` input shape, `test_data.h`, and the firmware's
  dims check - all automatic, but re-check the flash budget: samples cost 4 bytes per value.
- The representative dataset is whatever `X_train` is; INT8 scales change; the board reads
  them from the model, so no firmware edit, but re-run Steps 5, 7, 8, 9, 10.
- `export_arduino.py` writes labels as `int8_t` and indices as `uint16_t`: fine up to 65535
  samples and 127 classes; the sketch prints `out0,out1` only, so with more than 2 classes
  the `#RES` line and `compare.py` need one more column per class.

**New model**
- `src/model.py`. Then `python -m src.model` for shapes and parameter count.
- New op types appear in `inspect_tflite.py` (e.g. BatchNorm folds away, Softmax and
  Dropout(eval) add nothing, but Dense stacks add FULLY_CONNECTED, padding adds PAD, global
  pooling adds MEAN). Each must exist as an int8 kernel in the library (Step 6) and be added
  to the resolver list in the .ino with the right `<N>`.
- Arena size: check `#ARENA,used=` and keep a margin; bigger activations need a bigger arena.
- Per-channel quantized FULLY_CONNECTED is not supported by this library version: keep the
  converter flag from Step 8, and re-run the raw-output comparison - it is the only test that
  catches this class of bug.

**New board**
- `board_config.h` (baud, boot delay, arena) and the FQBN / port in the comment blocks,
  `config.py` (`SERIAL_PORT`), the compile/upload commands.
- Whether the board's serial port drops early output (Step 8) depends on the chip: a board
  with a USB-UART bridge chip does not have that problem; the ENTER-triggered report is
  harmless there.
- The TFLM library choice may change (a Cortex-M board gets real CMSIS-NN acceleration from
  the same library; other vendors' cores may not compile it). Re-run Step 0.4's smoke test,
  Step 6's kernel check, and Steps 8-10 in full. Optimized kernels are exactly what the
  raw-output comparison is for.
- `mbedtls_sha256` is ESP32-specific; other cores need another SHA-256 source or a
  different integrity check.

**New input length only** (same dataset family)
- Nothing to type: `series_length` flows from the data into the model, the export and the
  firmware check. What changes are the numbers: Linear input size, model bytes, arena, flash.
  Re-run Steps 2-10.

**Things that never need to change**
- The quantization formula on the board (`lrintf(x / scale) + zero_point`, clipped) and the
  argmax tie rule - as long as the PC side stays `np.rint` and `np.argmax`.
- The serial protocol and `compare.py`, unless the number of classes changes.