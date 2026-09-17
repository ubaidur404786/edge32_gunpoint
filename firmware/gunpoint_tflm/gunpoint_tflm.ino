/*
  gunpoint_tflm.ino - run the INT8 GunPoint model on the ESP32-S3 with TensorFlow Lite Micro
  and print results in a machine-readable form for src/compare.py.

  FQBN    : esp32:esp32:pandabyte_xs3:CDCOnBoot=default
  Compile : arduino-cli compile --jobs 1 --fqbn esp32:esp32:pandabyte_xs3:CDCOnBoot=default firmware\gunpoint_tflm
  Upload  : arduino-cli upload  --fqbn esp32:esp32:pandabyte_xs3:CDCOnBoot=default -p COM11 firmware\gunpoint_tflm
  Capture : (PowerShell terminal, project root; connects, waits 3 s, sends ENTER, records 10 s)
            cmd /c "(ping -n 4 127.0.0.1 >nul & echo. & ping -n 11 127.0.0.1 >nul) | arduino-cli monitor -p COM11 -c baudrate=115200 > logs\run_5samples.txt 2>&1"
            Interactive alternative: arduino-cli monitor -p COM11 -c baudrate=115200, press ENTER, Ctrl+C after #END.

  Why ENTER: COM11 is the chip's native USB-Serial/JTAG port. Until the host has actually talked
  to it, its driver discards what the sketch prints (flush() does not help), so a boot-time report
  always loses its first lines. The report therefore runs once at boot AND again on any received
  byte; compare.py uses the last complete #META..#END block in the log.

  Serial protocol (one record per line, fields comma separated):
    #META,sha256=..,expected=..,model_len=..,series_length=..,n_samples=..,n_classes=..
    #DEVICE,chip=..,cpu_mhz=..,heap_size=..,free_heap=..,sketch_size=..,free_sketch_space=..,psram_size=..
    #ARENA,used=..,size=..
    #QUANT,in_scale=..,in_zp=..,out_scale=..,out_zp=..
    #RES,idx,true,pred,latency_us,out0,out1        (out0/out1 = raw int8 logits)
    #SUMMARY,correct=..,total=..,accuracy=..
    #LATENCY,mean_us=..,min_us=..,max_us=..
    #MEMORY,arena_used=..,arena_size=..,free_heap=..
    #END
    #ERROR,<message>   on any failure; the sketch then stops.
*/

#include <math.h>                                              // lrintf(): round-to-nearest-even, same rule as np.rint on the PC
#include "mbedtls/sha256.h"                                    // mbedtls_sha256(): hash the model bytes in flash (mbedtls 3.6.6 ships with the core)
#include "board_config.h"                                      // baud, serial wait, arena size, warm-up count
#include <Chirale_TensorFlowLite.h>                            // umbrella header; makes the Arduino build compile the TFLM library
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"   // MicroMutableOpResolver: register exactly the 5 op types of the model
#include "tensorflow/lite/micro/micro_interpreter.h"           // MicroInterpreter: AllocateTensors(), Invoke(), input()/output()
#include "tensorflow/lite/schema/schema_generated.h"           // tflite::GetModel() and TFLITE_SCHEMA_VERSION
#include "model_data.h"                                        // g_model[], g_model_len, G_MODEL_SHA256 (generated)
#include "test_data.h"                                         // kTestData, kTestLabels, kTestIndex, kSeriesLength, kNumSamples (generated)

alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static tflite::MicroInterpreter* interpreter = nullptr;   // built once, reused by every report

// Print an #ERROR record and stop forever. compare.py treats any #ERROR as fatal.
static void halt(const char* msg) {
  Serial.print("#ERROR,");
  Serial.println(msg);
  Serial.flush();
  while (true) delay(1000);
}

// SHA-256 of the model bytes actually present in flash, as 64 hex characters.
static void model_sha256_hex(char out[65]) {
  unsigned char digest[32];
  mbedtls_sha256(g_model, g_model_len, digest, 0);   // 0 = SHA-256 (1 would be SHA-224)
  for (int i = 0; i < 32; i++) sprintf(out + 2 * i, "%02x", digest[i]);
  out[64] = '\0';
}

// Float -> int8 exactly like pc_inference.quantize_input(): q = rint(x / scale) + zp, clipped.
static inline int8_t quantize(float x, float scale, int32_t zero_point) {
  long q = lrintf(x / scale) + zero_point;
  if (q > 127) q = 127;
  if (q < -128) q = -128;
  return (int8_t)q;
}

// argmax over raw int8 logits with strict '>' : a tie goes to the lower index, like np.argmax.
static int argmax_int8(const int8_t* v, int n) {
  int best = 0;
  for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
  return best;
}

// Build resolver + interpreter once and allocate the arena. Later calls are no-ops.
static void init_interpreter() {
  if (interpreter != nullptr) return;

  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) halt("schema version mismatch");

  // Exactly the 5 op types found in model_int8.tflite (Step 6). Nothing else.
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddReshape();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddTranspose();
  resolver.AddFullyConnected();

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  if (static_interpreter.AllocateTensors() != kTfLiteOk) halt("AllocateTensors failed (arena too small or unsupported op)");
  interpreter = &static_interpreter;
}

// The complete report, #META .. #END. Runs at boot and again on every received byte.
static void run_report() {
  // ---- #META: prove the firmware holds exactly the exported model ---------------------
  char sha[65];
  model_sha256_hex(sha);
  Serial.printf("#META,sha256=%s,expected=%s,model_len=%u,series_length=%d,n_samples=%d,n_classes=%d\n",
                sha, G_MODEL_SHA256, g_model_len, kSeriesLength, kNumSamples, kNumClasses);
  Serial.flush();
  if (strcmp(sha, G_MODEL_SHA256) != 0) halt("model sha256 mismatch");

  // ---- #DEVICE: memory and chip facts, all read at runtime ----------------------------
  Serial.printf("#DEVICE,chip=%s,cpu_mhz=%u,heap_size=%u,free_heap=%u,sketch_size=%u,free_sketch_space=%u,psram_size=%u\n",
                ESP.getChipModel(), ESP.getCpuFreqMHz(), ESP.getHeapSize(), ESP.getFreeHeap(),
                ESP.getSketchSize(), ESP.getFreeSketchSpace(), ESP.getPsramSize());
  Serial.flush();

  // ---- interpreter (built once) and arena usage --------------------------------------------
  init_interpreter();
  Serial.printf("#ARENA,used=%u,size=%u\n", (unsigned)interpreter->arena_used_bytes(), (unsigned)TENSOR_ARENA_SIZE);
  Serial.flush();

  // ---- runtime checks: the model must match the header and be fully int8 -----------------
  TfLiteTensor* input = interpreter->input(0);
  TfLiteTensor* output = interpreter->output(0);
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) halt("input/output tensor is not int8");
  if (input->dims->size != 3 || input->dims->data[0] != 1 || input->dims->data[1] != kSeriesLength || input->dims->data[2] != 1)
    halt("input dims are not (1, series_length, 1)");
  if (output->dims->size != 2 || output->dims->data[1] != kNumClasses) halt("output dims are not (1, n_classes)");

  // Quantization parameters come from the model file, never from constants in this sketch.
  const float in_scale = input->params.scale;
  const int32_t in_zp = input->params.zero_point;
  const float out_scale = output->params.scale;
  const int32_t out_zp = output->params.zero_point;
  Serial.printf("#QUANT,in_scale=%.9g,in_zp=%d,out_scale=%.9g,out_zp=%d\n", in_scale, (int)in_zp, out_scale, (int)out_zp);
  Serial.flush();

  // ---- warm-up: first invokes can be slower (caches); not timed ----------------------------
  for (int w = 0; w < WARMUP_RUNS; w++) {
    for (int j = 0; j < kSeriesLength; j++) input->data.int8[j] = quantize(kTestData[0][j], in_scale, in_zp);
    if (interpreter->Invoke() != kTfLiteOk) halt("Invoke failed during warm-up");
  }

  // ---- the samples ---------------------------------------------------------------------------
  int correct = 0;
  uint64_t lat_sum = 0;
  uint32_t lat_min = UINT32_MAX, lat_max = 0;
  for (int i = 0; i < kNumSamples; i++) {
    for (int j = 0; j < kSeriesLength; j++) input->data.int8[j] = quantize(kTestData[i][j], in_scale, in_zp);

    const uint32_t t_start = micros();          // time ONLY Invoke()
    const TfLiteStatus st = interpreter->Invoke();
    const uint32_t dt = micros() - t_start;
    if (st != kTfLiteOk) halt("Invoke failed");

    const int8_t out0 = output->data.int8[0];
    const int8_t out1 = output->data.int8[1];
    const int pred = argmax_int8(output->data.int8, kNumClasses);
    if (pred == kTestLabels[i]) correct++;
    lat_sum += dt;
    if (dt < lat_min) lat_min = dt;
    if (dt > lat_max) lat_max = dt;

    Serial.printf("#RES,%d,%d,%d,%u,%d,%d\n", (int)kTestIndex[i], (int)kTestLabels[i], pred, dt, (int)out0, (int)out1);
  }

  // ---- summary -----------------------------------------------------------------------------
  Serial.printf("#SUMMARY,correct=%d,total=%d,accuracy=%.4f\n", correct, kNumSamples, (float)correct / kNumSamples);
  Serial.printf("#LATENCY,mean_us=%.1f,min_us=%u,max_us=%u\n", (double)lat_sum / kNumSamples, lat_min, lat_max);
  Serial.printf("#MEMORY,arena_used=%u,arena_size=%u,free_heap=%u\n",
                (unsigned)interpreter->arena_used_bytes(), (unsigned)TENSOR_ARENA_SIZE, ESP.getFreeHeap());
  Serial.println("#END");
  Serial.flush();
}

void setup() {
  // Wait up to SERIAL_WAIT_MS for a monitor; on the USB-Serial/JTAG port this is best effort only.
  Serial.begin(SERIAL_BAUD);
  const uint32_t t0 = millis();
  while (!Serial && millis() - t0 < SERIAL_WAIT_MS) delay(10);
  delay(SERIAL_BOOT_DELAY_MS);
  Serial.println();
  run_report();
}

void loop() {
  // Any received byte (e.g. ENTER in the monitor) triggers a fresh, complete report.
  if (Serial.available()) {
    delay(50);
    while (Serial.available()) Serial.read();   // swallow the whole line
    Serial.println();
    run_report();
  }
  delay(10);
}