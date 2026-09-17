// board_config.h - every board-specific setting in one place.
//
// Board : PandaByte xS3 (ESP32-S3), arduino-esp32 core 3.3.11
// FQBN  : esp32:esp32:pandabyte_xs3:CDCOnBoot=default
//         (USB CDC On Boot shows two "Enabled" entries; "default" = the FIRST one; the
//          other one gives no serial output on this board)
// Port  : COM11
// TFLM  : Chirale_TensorFLowLite 2.0.0
#pragma once

#define SERIAL_BAUD        115200
#define SERIAL_WAIT_MS     5000        // wait at most this long for a serial monitor before printing
#define SERIAL_BOOT_DELAY_MS 2000      // USB-Serial/JTAG drops output for a while after reset; wait before the first line

// Tensor arena: one static buffer for all input/output/intermediate tensors of the model.
// Generous on purpose; the sketch prints the measured arena_used_bytes() so it can be shrunk later.
#define TENSOR_ARENA_SIZE  (32 * 1024)

#define WARMUP_RUNS        3           // Invoke() calls before timing starts
