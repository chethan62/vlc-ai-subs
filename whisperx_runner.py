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
"""
import json
import os
import sys


def format_srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def emit(data: dict):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def main():
    if len(sys.argv) < 5:
        emit({"type": "error", "msg": "Usage: runner <media> <model> <lang> <task> [mirror] [srt]"})
        sys.exit(1)

    media_path = sys.argv[1]
    model_name = sys.argv[2]
    language = sys.argv[3] if sys.argv[3] != "auto" else None
    task = sys.argv[4]
    srt_requested = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6].strip() else None

    if not os.path.isfile(media_path):
        emit({"type": "error", "msg": f"File not found: {media_path}"})
        sys.exit(1)

    import whisperx

    # Resolve device
    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    compute = "int8_float16" if device == "cuda" else "float32"

    emit({"type": "status", "msg": f"WhisperX: loading {model_name} on {device}..."})

    # 1. Transcribe
    model = whisperx.load_model(model_name, device, compute_type=compute)
    result = model.transcribe(
        media_path,
        language=language,
        task=task,
    )

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

    # 4. Write SRT
    if srt_requested:
        srt_path = srt_requested
        os.makedirs(os.path.dirname(srt_path) or ".", exist_ok=True)
    else:
        base, _ = os.path.splitext(media_path)
        srt_path = base + ".srt"

    try:
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))
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