"""
Runs ONE (library, model_size, device, audio_file) benchmark in an isolated process.
Invoked by benchmark_whisper.py via subprocess. Prints a single JSON line to stdout
on success; non-zero exit + stderr traceback on failure (caught by the orchestrator).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import gc
import json
import time
import argparse
import threading
import re
import pynvml
import numpy as np
import psutil
import torch
import librosa
import jiwer
import whisper
from faster_whisper import WhisperModel
from thop import profile as thop_profile
import whisper as openai_whisper_pkg
import traceback

SAMPLE_RATE = 16000
BEAM_SIZE = 3
TASK = "translate"

def measure_openai_whisper_gflops(model, audio: np.ndarray, num_generated_tokens: int) -> dict | None:
    try:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        mel = openai_whisper_pkg.log_mel_spectrogram(audio).to(device=device, dtype=dtype)
        mel = openai_whisper_pkg.pad_or_trim(mel, openai_whisper_pkg.audio.N_FRAMES)
        mel_input = mel.unsqueeze(0)

        print(f"DEBUG: model dtype={dtype}, encoder param count={sum(p.numel() for p in model.encoder.parameters())}", file=sys.stderr)
        encoder_macs, _ = thop_profile(model.encoder, inputs=(mel_input,), verbose=False)

        # Decoder FLOPs: one forward pass at a representative single-token step
        with torch.no_grad():
            encoder_output = model.encoder(mel_input)

        dummy_tokens = torch.tensor([[model.decoder.token_embedding.num_embeddings - 1]],
                                     device=device, dtype=torch.long)
        decoder_macs_per_token, _ = thop_profile(
            model.decoder, inputs=(dummy_tokens, encoder_output), verbose=False
        )

        # MACs -> FLOPs (x2), encoder runs once, decoder runs ~num_generated_tokens times
        total_flops = (encoder_macs * 2) + (decoder_macs_per_token * 2 * max(num_generated_tokens, 1))
        return {
            "total_flops": total_flops,
            "encoder_gflops": round((encoder_macs * 2) / 1e9, 3),
            "decoder_gflops_per_token": round((decoder_macs_per_token * 2) / 1e9, 4),
        }
    except Exception as e:
        print(f"thop FLOP measurement failed: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None

def load_audio(path: str) -> np.ndarray:
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)

def load_reference_text(audio_path: str) -> str | None:
    base = os.path.splitext(os.path.basename(audio_path))[0]
    ref_path = os.path.join(os.path.dirname(audio_path), "..", "text", base + ".txt")
    ref_path = os.path.normpath(ref_path)
    if not os.path.isfile(ref_path):
        return None
    with open(ref_path, "r", encoding="utf-8") as f:
        return f.read()

_APOSTROPHE_RE = re.compile(r"['']")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

def normalize_for_wer(text: str) -> str:
    text = text.lower()
    text = _APOSTROPHE_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_wer(reference: str, hypothesis: str) -> dict:
    ref_norm = normalize_for_wer(reference)
    hyp_norm = normalize_for_wer(hypothesis)

    if not ref_norm.strip():
        return {
            "wer": None, "substitutions": None, "deletions": None,
            "insertions": None, "hits": None, "ref_word_count": 0,
            "wer_error": "empty reference after normalization",
        }

    output = jiwer.process_words(ref_norm, hyp_norm)
    ref_word_count = len(ref_norm.split())

    return {
        "wer": round(output.wer, 4),
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits,
        "ref_word_count": ref_word_count,
        "wer_error": None,
    }

def process_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

# NVML sampling thread
class NvmlSampler:
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.samples_util = []
        self.samples_mem_used = []
        self._stop = threading.Event()
        self._thread = None
        self._ok = False
        try:
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._ok = True
        except Exception as e:
            print(f"NVML unavailable: {e}", file=sys.stderr)

    def _loop(self):
        while not self._stop.is_set():
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.samples_util.append(util.gpu)
                self.samples_mem_used.append(mem.used / (1024 ** 2))
            except Exception:
                pass
            time.sleep(0.1)

    def start(self):
        if self._ok:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        if self._ok and self._thread:
            self._stop.set()
            self._thread.join(timeout=2)

    def summary(self):
        if not self.samples_util:
            return {"gpu_util_pct_avg": None, "gpu_util_pct_max": None, "nvml_vram_used_mb_max": None}
        return {
            "gpu_util_pct_avg": round(sum(self.samples_util) / len(self.samples_util), 1),
            "gpu_util_pct_max": max(self.samples_util),
            "nvml_vram_used_mb_max": round(max(self.samples_mem_used), 1),
        }

# Library runners
def run_openai_whisper(model_size, device, audio):
    t0 = time.perf_counter()
    model = whisper.load_model(model_size, device=device)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = model.transcribe(
        audio,
        task=TASK,
        language=None,
        beam_size=BEAM_SIZE,
        condition_on_previous_text=False,
        word_timestamps=True,
        fp16=(device == "cuda"),
    )
    transcribe_time = time.perf_counter() - t1

    return {
        "text": result.get("text", "").strip(),
        "detected_language": result.get("language"),
        "language_probability": None,
        "load_time": load_time,
        "transcribe_time": transcribe_time,
        "compute_type": "fp16" if device == "cuda" else "fp32",
        "cpu_threads": None,
        "_model_ref": model
    }

def run_faster_whisper(model_size, device, audio):
    compute_type = "float16" if device == "cuda" else "int8"
    cpu_threads = os.cpu_count() if device == "cpu" else 0
    kwargs = {"cpu_threads": cpu_threads} if device == "cpu" else {}

    t0 = time.perf_counter()
    model = WhisperModel(model_size, local_files_only=False, device=device,
                          compute_type=compute_type, **kwargs)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    segments, info = model.transcribe(
        audio,
        beam_size=BEAM_SIZE,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 700},
        task=TASK,
        language=None,
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    text = " ".join(seg.text.strip() for seg in segments)
    transcribe_time = time.perf_counter() - t1

    return {
        "text": text.strip(),
        "detected_language": info.language,
        "language_probability": round(info.language_probability, 4),
        "load_time": load_time,
        "transcribe_time": transcribe_time,
        "compute_type": compute_type,
        "cpu_threads": cpu_threads,
        "_model_ref": None,   # CTranslate2 backend — not traceable by thop
    }

RUNNERS = {"openai-whisper": run_openai_whisper, "faster-whisper": run_faster_whisper}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library", required=True)
    p.add_argument("--model-size", required=True)
    p.add_argument("--device", required=True, choices=["cpu", "cuda"])
    p.add_argument("--audio-path", required=True)
    p.add_argument("--audio-duration", required=True, type=float)
    args = p.parse_args()

    ram_before = process_ram_mb()

    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    sampler = NvmlSampler() if args.device == "cuda" else None
    if sampler:
        sampler.start()

    try:
        audio = load_audio(args.audio_path)
        runner = RUNNERS[args.library]

        t_total0 = time.perf_counter()
        out = runner(args.model_size, args.device, audio)
        total_time = time.perf_counter() - t_total0

        if sampler:
            sampler.stop()

        ram_after = process_ram_mb()
        vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if args.device == "cuda" else None

        achieved_gflops = None
        total_measured_gflops = None
        flop_measurement_note = None

        if args.library == "openai-whisper" and out.get("_model_ref") is not None:
            num_tokens = len(out["text"].split())  # proxy for generated token count
            flop_measurement = measure_openai_whisper_gflops(out["_model_ref"], audio, num_tokens)
            if flop_measurement is not None:
                total_measured_gflops = round(flop_measurement["total_flops"] / 1e9, 3)
                if out["transcribe_time"] > 0:
                    achieved_gflops = round(total_measured_gflops / out["transcribe_time"], 2)
                flop_measurement_note = (
                    f"encoder={flop_measurement['encoder_gflops']}GFLOPs (1 pass), "
                    f"decoder={flop_measurement['decoder_gflops_per_token']}GFLOPs/token "
                    f"x{num_tokens} tokens (approx, autoregressive cache not modeled)"
                )
            else:
                flop_measurement_note = "thop tracing failed — see stderr"
        elif args.library == "faster-whisper":
            flop_measurement_note = "not measurable: CTranslate2 backend is not a traceable PyTorch graph"

        nvml_summary = sampler.summary() if sampler else {
            "gpu_util_pct_avg": None, "gpu_util_pct_max": None, "nvml_vram_used_mb_max": None
        }

        reference_text = load_reference_text(args.audio_path)
        if reference_text is not None:
            wer_result = compute_wer(reference_text, out["text"])
        else:
            wer_result = {
                "wer": None, "substitutions": None, "deletions": None,
                "insertions": None, "hits": None, "ref_word_count": None,
                "wer_error": "reference .txt not found",
            }

        result = {
            "status": "success",
            "error_message": None,
            "audio_file": args.audio_path,
            "audio_duration_sec": round(args.audio_duration, 3),
            "library": args.library,
            "model_size": args.model_size,
            "device": args.device,
            "compute_type": out.get("compute_type"),
            "task": TASK,
            "detected_language": out.get("detected_language"),
            "language_probability": out.get("language_probability"),
            "model_load_time_sec": round(out["load_time"], 3),
            "transcription_time_sec": round(out["transcribe_time"], 3),
            "total_time_sec": round(total_time, 3),
            "realtime_factor": round(args.audio_duration / out["transcribe_time"], 3)
                if out["transcribe_time"] > 0 else None,
            "ram_before_mb": round(ram_before, 1),
            "peak_ram_mb": round(ram_after, 1),
            "ram_delta_mb": round(ram_after - ram_before, 1),
            "peak_vram_mb": round(vram_mb, 1) if vram_mb is not None else None,
            "gpu_util_pct_avg": nvml_summary["gpu_util_pct_avg"],
            "gpu_util_pct_max": nvml_summary["gpu_util_pct_max"],
            "nvml_vram_used_mb_max": nvml_summary["nvml_vram_used_mb_max"],
            "theoretical_gflops_per_sec_audio": None,
            "total_measured_gflops": total_measured_gflops,
            "achieved_gflops": achieved_gflops,
            "flop_measurement_note": flop_measurement_note,
            "cpu_threads_used": out.get("cpu_threads"),
            "transcript_char_len": len(out["text"]),
            "transcript_word_count": len(out["text"].split()),
            "transcript_text": out["text"],
            "wer": wer_result["wer"],
            "wer_substitutions": wer_result["substitutions"],
            "wer_deletions": wer_result["deletions"],
            "wer_insertions": wer_result["insertions"],
            "wer_hits": wer_result["hits"],
            "wer_ref_word_count": wer_result["ref_word_count"],
            "wer_error": wer_result["wer_error"],
        }
        print(json.dumps(result))

    except Exception as e:
        if sampler:
            sampler.stop()
        err_result = {
            "status": "error",
            "error_message": f"{type(e).__name__}: {e}",
            "audio_file": args.audio_path,
            "library": args.library,
            "model_size": args.model_size,
            "device": args.device,
        }
        print(json.dumps(err_result))
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
