#!/usr/bin/env python3
"""
WhisperX runner — invoked by the plugin as a subprocess when the main venv
is Python ≥3.14 (whisperX requires <3.14).  This runs inside its own
Python 3.12 venv.

Contract (stdout, JSONL):
  {"type": "status", "msg": "..."}
  {"type": "sub", "i": N, "start": S, "end": E, "text": "..."}
  {"type": "done", "segments": N, "srt_path": "..."}
  {"type": "error", "msg": "..."}

Args:  <media> <model> <language> <task> [mirror_file] [srt_path]
The SRT file is written ONLY when [srt_path] is given — the plugin's caller
(aisubs_whisper.py) owns SRT output, so the runner never creates side-effect
files next to the media (realtime-OSD mode, read-only media dirs).
Env:  VSCL_AISUBS_DEVICE=cpu|cuda forces the device; VSCL_AISUBS_COMPUTE
overrides the compute type; VSCL_AISUBS_MODEL_CACHE sets the HF cache dir.
"""
import json
import os
import sys


_CUDA_COMPUTE = ("int8", "int8_float16", "int8_float32", "float16", "float32")
_CPU_COMPUTE = ("int8", "int8_float32", "float32")


def resolve_device(cuda_available: bool) -> str:
    """Device selection: VSCL_AISUBS_DEVICE=cpu|cuda forces it; else auto."""
    env_device = os.environ.get("VSCL_AISUBS_DEVICE", "").strip().lower()
    if env_device == "cpu":
        return "cpu"
    if env_device == "cuda":
        return "cuda"
    return "cuda" if cuda_available else "cpu"


def resolve_compute(device: str) -> str:
    """Compute type: VSCL_AISUBS_COMPUTE overrides; sensible default per device.

    Validated per device — int8_float16/float16 are CUDA-only and would
    crash inside faster-whisper on CPU.
    """
    env_ct = os.environ.get("VSCL_AISUBS_COMPUTE", "").strip().lower()
    allowed = _CUDA_COMPUTE if device == "cuda" else _CPU_COMPUTE
    if env_ct in allowed:
        return env_ct
    return "int8_float16" if device == "cuda" else "float32"


def model_cache_dir() -> str | None:
    """VSCL_AISUBS_MODEL_CACHE → HF download_root (None = default cache)."""
    cache = os.environ.get("VSCL_AISUBS_MODEL_CACHE", "").strip()
    return cache or None


def hardened_asr_options() -> dict:
    """Research-backed decode options (see .research/final_report.md §2.2).

    WhisperX's defaults are beam_size=5 with temperature fallback ladder and
    no hallucination gate — fine for clean audio, but movie soundtracks get
    silence/music hallucinations. BoH mitigation: beam 1, fixed low
    temperature, no cross-window conditioning, explicit silence thresholds.
    """
    return {
        "beam_size": 1,
        "condition_on_previous_text": False,
        "temperatures": [0.0],
        "hallucination_silence_threshold": 2.0,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }


def format_srt_timestamp(seconds: float) -> str:
    # Mirror of core/srt.py — total-ms math with rounding, float-safe.
    total_ms = max(0, round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def emit(data: dict):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def write_srt_if_requested(srt_lines: list, srt_requested: str | None) -> str | None:
    """Write the SRT only when an explicit path was requested; else None.

    The plugin caller always writes the SRT itself — the runner must not
    create <media>.srt side effects (realtime-OSD mode, read-only dirs).
    Empty output is skipped (no 0-byte SRTs). Raises OSError on write
    failure so main() can emit a clean JSONL error.
    """
    if not srt_requested or not srt_lines:
        return None
    srt_path = srt_requested
    os.makedirs(os.path.dirname(srt_path) or ".", exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    return srt_path


def main():
    if len(sys.argv) < 5:
        emit({"type": "error", "msg": "Usage: runner <media> <model> <lang> <task> [mirror] [srt]"})
        sys.exit(1)

    import time
    _t0 = time.time()

    media_path = sys.argv[1]
    model_name = sys.argv[2]
    language = sys.argv[3] if sys.argv[3] != "auto" else None
    task = sys.argv[4]
    srt_requested = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6].strip() else None

    if not os.path.isfile(media_path):
        emit({"type": "error", "msg": f"File not found: {media_path}"})
        sys.exit(1)

    import whisperx

    # Resolve device: VSCL_AISUBS_DEVICE=cpu|cuda forces it; else auto-detect.
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    device = resolve_device(cuda_available)
    compute = resolve_compute(device)

    emit({"type": "status", "msg": f"WhisperX: loading {model_name} on {device} ({compute})..."})

    # 1. Transcribe — VAD-first (whisperx default gate) + hardened decode.
    model = whisperx.load_model(
        model_name, device, compute_type=compute,
        asr_options=hardened_asr_options(),
        vad_options={"vad_onset": 0.500, "vad_offset": 0.363},
        download_root=model_cache_dir(),
    )
    result = model.transcribe(
        media_path,
        language=language,
        task=task,
    )
    emit({
        "type": "status",
        "msg": f"WhisperX: transcription done (+{time.time() - _t0:.0f}s)",
    })

    # 2. Align (word-level timestamps)
    if device == "cuda" and result.get("segments"):
        try:
            lang_code = result.get("language") or language or "en"
            align_model, metadata = whisperx.load_align_model(
                language_code=lang_code, device=device,
            )
            result = whisperx.align(
                result["segments"], align_model, metadata,
                media_path, device, return_char_alignments=False,
            )
            emit({
                "type": "status",
                "msg": f"WhisperX: word alignment done (+{time.time() - _t0:.1f}s)",
            })
        except Exception:
            emit({"type": "status", "msg": "Alignment skipped (may need different language model)"})

    emit({"type": "status", "msg": "Transcribing..."})

    # 3. Yield segments + build SRT
    segments = result.get("segments", [])
    srt_lines = []
    count = 0

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        count += 1
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        emit({"type": "sub", "i": count, "start": round(start, 3), "end": round(end, 3), "text": text})
        srt_lines.append(
            f"{count}\n"
            f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n"
            f"{text}\n"
        )

    # 4. Write SRT — only when the caller explicitly requested a path (the
    # plugin caller owns SRT output; no <media>.srt side effects).
    try:
        srt_path = write_srt_if_requested(srt_lines, srt_requested)
    except OSError as exc:
        emit({"type": "error", "msg": f"Could not write SRT: {exc}"})
        sys.exit(1)

    emit({"type": "done", "segments": count, "srt_path": srt_path})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        emit({"type": "error", "msg": f"{exc}\n{traceback.format_exc()}"})
        sys.exit(1)