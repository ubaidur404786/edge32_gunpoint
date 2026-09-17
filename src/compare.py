"""Compare a board log (logs/*.txt) with the PC reference (results/pc_predictions.csv).

Run from the project root:
    python -m src.compare logs/run_full.txt
Options:
    --compile-summary results/compile_summary.txt   the two "Sketch uses / Global variables" lines
Writes results/summary.txt.
"""

import argparse   # log path and options from the command line
import csv        # read results/pc_predictions.csv
import json       # read export_manifest.json / pc_summary.json, write summary.json
import re         # find the # markers anywhere in a line, parse key=value fields
import statistics # mean / median of latencies

import config     # paths

MARKERS = ("META", "DEVICE", "ARENA", "QUANT", "RES", "SUMMARY", "LATENCY", "MEMORY", "END", "ERROR")
MARKER_RE = re.compile(r"#(" + "|".join(MARKERS) + r")\b(.*)$")


def read_log_text(path):
    """Read a serial capture. Tee-Object writes UTF-16 (BOM FF FE); cmd/bash redirects write UTF-8."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


def parse_blocks(text):
    """Split the log into #META..#END blocks. Markers may sit anywhere in a line (timestamp prefixes)."""
    blocks, current = [], None
    for line in text.splitlines():
        m = MARKER_RE.search(line)
        if not m:
            continue
        kind, rest = m.group(1), m.group(2).strip()
        if kind == "META":
            current = {"META": rest, "RES": [], "complete": False, "error": None}
            blocks.append(current)
        if current is None:
            continue                       # tail of a block whose #META was lost (boot-time truncation)
        if kind == "RES":
            current["RES"].append(rest)
        elif kind == "END":
            current["complete"] = True
        elif kind == "ERROR":
            current["error"] = rest
        else:
            current[kind] = rest
    return blocks


def kv(s):
    """',a=1,b=x' -> {'a': '1', 'b': 'x'}"""
    return dict(re.findall(r"(\w+)=([^,]*)", s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="board capture, e.g. logs/run_full.txt")
    ap.add_argument("--compile-summary", default=str(config.RESULTS_DIR / "compile_summary.txt"),
                    help="file holding the arduino-cli 'Sketch uses ... / Global variables ...' lines")
    args = ap.parse_args()

    log_path = config.ROOT / args.log
    manifest = json.loads((config.RESULTS_DIR / "export_manifest.json").read_text())
    pc_summary = json.loads((config.RESULTS_DIR / "pc_summary.json").read_text())
    with open(config.RESULTS_DIR / "pc_predictions.csv", newline="") as f:
        pc = {int(r["idx"]): r for r in csv.DictReader(f)}

    # ---- pick the last complete block -----------------------------------------------------
    blocks = parse_blocks(read_log_text(log_path))
    complete = [b for b in blocks if b["complete"]]
    if not complete:
        raise SystemExit(f"no complete #META..#END block in {log_path} ({len(blocks)} block starts found)")
    b = complete[-1]
    if b["error"]:
        raise SystemExit(f"board reported #ERROR,{b['error']}")

    meta, dev, arena, quant = kv(b["META"]), kv(b.get("DEVICE", "")), kv(b.get("ARENA", "")), kv(b.get("QUANT", ""))
    bsum, blat, bmem = kv(b.get("SUMMARY", "")), kv(b.get("LATENCY", "")), kv(b.get("MEMORY", ""))

    out = []
    p = out.append
    p(f"edge32_gunpoint - board vs PC comparison")
    p(f"log: {log_path.relative_to(config.ROOT)}   (block {len(complete)} of {len(complete)} complete, {len(blocks)} started)")
    p("")

    # ---- integrity checks ------------------------------------------------------------------
    checks = [
        ("sha256 board == export", meta.get("sha256") == manifest["model_sha256"] == meta.get("expected"),
         f"{meta.get('sha256')} vs {manifest['model_sha256']}"),
        ("model_len == export", int(meta.get("model_len", -1)) == manifest["model_bytes"],
         f"{meta.get('model_len')} vs {manifest['model_bytes']}"),
        ("n_samples == export", int(meta.get("n_samples", -1)) == manifest["n_samples"],
         f"{meta.get('n_samples')} vs {manifest['n_samples']}"),
        ("#RES count == n_samples", len(b["RES"]) == int(meta.get("n_samples", -1)),
         f"{len(b['RES'])} vs {meta.get('n_samples')}"),
        ("series_length == export", int(meta.get("series_length", -1)) == manifest["series_length"],
         f"{meta.get('series_length')} vs {manifest['series_length']}"),
        ("input scale/zp == PC", abs(float(quant.get("in_scale", 0)) - pc_summary["input_scale"]) < 1e-9
         and int(quant.get("in_zp", 999)) == pc_summary["input_zero_point"],
         f"{quant.get('in_scale')}/{quant.get('in_zp')} vs {pc_summary['input_scale']}/{pc_summary['input_zero_point']}"),
        ("output scale/zp == PC", abs(float(quant.get("out_scale", 0)) - pc_summary["output_scale"]) < 1e-9
         and int(quant.get("out_zp", 999)) == pc_summary["output_zero_point"],
         f"{quant.get('out_scale')}/{quant.get('out_zp')} vs {pc_summary['output_scale']}/{pc_summary['output_zero_point']}"),
    ]
    p("integrity checks")
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= bool(ok)
        p(f"  [{'OK' if ok else 'FAIL'}] {name:28s} {detail}")
    p("")

    # ---- sample-by-sample join -------------------------------------------------------------
    n = 0; pred_agree = 0; raw_agree = 0; board_correct = 0; label_mismatch = 0
    max_abs_raw_diff = 0; disagreements = []; lats = []
    for rest in b["RES"]:
        idx, true, pred, lat, o0, o1 = (int(v) for v in rest.lstrip(",").split(",")[:6])
        r = pc[idx]
        n += 1; lats.append(lat)
        if true != int(r["true"]): label_mismatch += 1
        if pred == true: board_correct += 1
        if pred == int(r["int8_pred"]): pred_agree += 1
        same_raw = (o0 == int(r["int8_out0"]) and o1 == int(r["int8_out1"]))
        raw_agree += same_raw
        d = max(abs(o0 - int(r["int8_out0"])), abs(o1 - int(r["int8_out1"])))
        max_abs_raw_diff = max(max_abs_raw_diff, d)
        if not same_raw or pred != int(r["int8_pred"]):
            disagreements.append((idx, true, int(r["int8_pred"]), pred, int(r["int8_out0"]), int(r["int8_out1"]), o0, o1))

    p("accuracy on the official TEST split")
    p(f"  PyTorch FP32 (PC)      : {pc_summary['acc_pytorch_fp32']:.4f}")
    p(f"  TFLite  FP32 (PC)      : {pc_summary['acc_tflite_fp32']:.4f}")
    p(f"  TFLite  INT8 (PC)      : {pc_summary['acc_tflite_int8']:.4f}")
    p(f"  TFLite  INT8 (board)   : {board_correct / n:.4f}  ({board_correct}/{n})   board's own #SUMMARY: {bsum.get('accuracy')} ({bsum.get('correct')}/{bsum.get('total')})")
    p("")
    p("board vs PC, sample by sample")
    p(f"  samples compared               : {n}")
    p(f"  prediction agreement           : {pred_agree}/{n}")
    p(f"  raw int8 output agreement      : {raw_agree}/{n}   (both logits identical)")
    p(f"  max |board - PC| raw int8 diff : {max_abs_raw_diff}")
    p(f"  true-label mismatches (export bug if > 0): {label_mismatch}")
    if disagreements:
        p("  disagreements (idx, true, pc_pred, board_pred, pc_out0, pc_out1, board_out0, board_out1):")
        for d in disagreements[:20]:
            p("    " + ", ".join(str(v) for v in d))
    p("")

    # ---- latency -------------------------------------------------------------------------------
    p("latency (Invoke() only, after warm-up, from #RES lines)")
    p(f"  mean {statistics.mean(lats):.1f} us, median {statistics.median(lats):.1f} us, min {min(lats)} us, max {max(lats)} us   "
      f"(board #LATENCY: mean {blat.get('mean_us')}, min {blat.get('min_us')}, max {blat.get('max_us')})")
    p("")

    # ---- memory / size -------------------------------------------------------------------------
    p("memory and size")
    p(f"  model_int8.tflite               : {manifest['model_bytes']} bytes")
    p(f"  tensor arena used / size        : {arena.get('used')} / {arena.get('size')} bytes")
    p(f"  heap size / free heap (board)   : {dev.get('heap_size')} / {bmem.get('free_heap')} bytes")
    p(f"  sketch size (board)             : {dev.get('sketch_size')} bytes, free sketch space {dev.get('free_sketch_space')} bytes")
    p(f"  chip / cpu / psram              : {dev.get('chip')} / {dev.get('cpu_mhz')} MHz / {dev.get('psram_size')} bytes")
    cs_path = config.ROOT / args.compile_summary
    if cs_path.exists():
        # cs = cs_path.read_text()
        cs = read_log_text(cs_path)   # Tee-Object writes UTF-16; reuse the BOM-aware reader
        flash = re.search(r"Sketch uses (\d+) bytes \((\d+)%\).*?Maximum is (\d+)", cs)
        ram = re.search(r"Global variables use (\d+) bytes \((\d+)%\).*?Maximum is (\d+)", cs)
        p(f"  flash (compile summary)         : {flash.group(1)} bytes ({flash.group(2)}%) of {flash.group(3)}" if flash else "  flash: not found in compile summary")
        p(f"  static RAM (compile summary)    : {ram.group(1)} bytes ({ram.group(2)}%) of {ram.group(3)}" if ram else "  static RAM: not found in compile summary")
    else:
        p(f"  flash / static RAM              : not measured yet (save the compile output to {cs_path.relative_to(config.ROOT)})")
    p("")
    verdict = all_ok and label_mismatch == 0 and raw_agree == n and pred_agree == n
    p(f"VERDICT: {'PASS - board reproduces the PC INT8 model bit for bit' if verdict else 'FAIL - see above'}")

    text = "\n".join(out)
    print(text)
    (config.RESULTS_DIR / "summary.txt").write_text(text + "\n")
    print("\nwrote", config.RESULTS_DIR / "summary.txt")


if __name__ == "__main__":
    main()