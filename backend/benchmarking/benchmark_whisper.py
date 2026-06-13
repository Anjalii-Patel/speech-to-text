"""
Whisper benchmark orchestrator. Spawns single_run.py as a fresh subprocess for
every (audio, model, device) combination, so each run gets clean RAM/VRAM with
zero cross-run contamination. Aggregates results into CSV + JSON.
"""

import os
import sys
import json
import csv
import subprocess
import platform
from datetime import datetime

import torch
import librosa
import glob
import re

# CONFIG
AUDIO_DIR = "./benchmarking/data/audio"
AUDIO_FILES = sorted(
    glob.glob(os.path.join(AUDIO_DIR, "english*.mp3")),
    key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group())
)

MODEL_CONFIGS = [
    ("openai-whisper", "large"),
    ("openai-whisper", "medium"),
    ("openai-whisper", "small"),
    ("faster-whisper", "large-v2"),
    ("faster-whisper", "large-v3"),
    ("faster-whisper", "medium"),
    ("faster-whisper", "small"),
]

DEVICES_TO_TEST = ["cpu", "cuda"]

OUTPUT_CSV = "benchmark_results.csv"
OUTPUT_JSON = "benchmark_results.json"
SINGLE_RUN_SCRIPT = os.path.join(os.path.dirname(__file__), "single_run.py")
SUBPROCESS_TIMEOUT_SEC = 1800

CSV_FIELDS = [
    "run_id", "timestamp",
    "audio_file", "audio_duration_sec",
    "library", "model_size", "device", "compute_type", "task",
    "detected_language", "language_probability",
    "model_load_time_sec", "transcription_time_sec", "total_time_sec",
    "realtime_factor",
    "ram_before_mb", "peak_ram_mb", "ram_delta_mb",
    "peak_vram_mb", "gpu_util_pct_avg", "gpu_util_pct_max", "nvml_vram_used_mb_max",
    "total_measured_gflops", "achieved_gflops", "flop_measurement_note",
    "cpu_threads_used",
    "transcript_char_len", "transcript_word_count", "transcript_text",
    "wer", "wer_substitutions", "wer_deletions", "wer_insertions",
    "wer_hits", "wer_ref_word_count", "wer_error",
    "status", "error_message",
]

def write_outputs(results):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in CSV_FIELDS})

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    cuda_available = torch.cuda.is_available()
    devices = [d for d in DEVICES_TO_TEST if d == "cpu" or cuda_available]
    if "cuda" in DEVICES_TO_TEST and not cuda_available:
        print("CUDA not available — skipping all GPU runs.")

    print(f"Host: {platform.platform()} | CPUs: {os.cpu_count()} | CUDA: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    audio_durations = {}
    valid_audio_files = []
    for path in AUDIO_FILES:
        if not os.path.isfile(path):
            print(f"SKIP (missing file): {path}")
            continue
        try:
            audio_durations[path] = librosa.get_duration(path=path)
            valid_audio_files.append(path)
        except Exception as e:
            print(f"SKIP (failed to read duration): {path} -> {e}")

    results = []
    run_id = 0
    total_runs = len(valid_audio_files) * len(MODEL_CONFIGS) * len(devices)
    print(f"Total planned runs: {total_runs}\n")

    for audio_path in valid_audio_files:
        duration = audio_durations[audio_path]

        for library, model_size in MODEL_CONFIGS:
            for device in devices:
                run_id += 1
                ts = datetime.now().isoformat()
                label = f"[{run_id}/{total_runs}] {library} | {model_size} | {device} | {os.path.basename(audio_path)}"
                print(label)

                cmd = [
                    sys.executable, SINGLE_RUN_SCRIPT,
                    "--library", library,
                    "--model-size", model_size,
                    "--device", device,
                    "--audio-path", audio_path,
                    "--audio-duration", str(duration),
                ]

                row = {
                    "run_id": run_id, "timestamp": ts,
                    "audio_file": audio_path, "audio_duration_sec": round(duration, 3),
                    "library": library, "model_size": model_size, "device": device,
                    "status": "error", "error_message": None,
                }

                try:
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True,
                        timeout=SUBPROCESS_TIMEOUT_SEC,
                    )
                    # last stdout line should be the JSON result
                    stdout_lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
                    if stdout_lines:
                        parsed = json.loads(stdout_lines[-1])
                        row.update(parsed)
                    if proc.returncode != 0 and row["status"] != "error":
                        row["status"] = "error"
                        row["error_message"] = f"subprocess exit code {proc.returncode}"
                    if proc.returncode != 0:
                        print(f"stderr: {proc.stderr[-1500:]}")
                    if row["status"] == "success" and proc.stderr.strip():
                        print(f"stderr (non-fatal):\n{proc.stderr[-2000:]}")

                except subprocess.TimeoutExpired:
                    row["error_message"] = f"timeout after {SUBPROCESS_TIMEOUT_SEC}s"
                except json.JSONDecodeError as e:
                    row["error_message"] = f"failed to parse subprocess output: {e}"
                except Exception as e:
                    row["error_message"] = f"{type(e).__name__}: {e}"

                status_line = "OK" if row["status"] == "success" else f"FAIL: {row['error_message']}"
                rtf = row.get("realtime_factor")
                if row["status"] == "success":
                    status_line += f"rtf={rtf}  load={row.get('model_load_time_sec')}s  gflops={row.get('achieved_gflops')}"
                print(status_line)

                results.append(row)
                write_outputs(results)

    print(f"\nDone. {len(results)} runs recorded -> {OUTPUT_CSV}, {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
