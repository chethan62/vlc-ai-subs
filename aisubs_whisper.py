#!/usr/bin/env python3
"""
vlc-ai-subs — Whisper transcription backend for VLC.

Architecture
────────────
  aisubs_whisper.py        CLI entry-point (you are here)
  core/
    emitter.py             JSONL output + file mirror for Lua polling
    device.py              CUDA lib preloading + device/compute detection
    srt.py                 SRT timestamp formatting and file writing
  backends/
    base.py                Abstract TranscriptionBackend
    whisper_cpp.py         whisper.cpp (Vulkan/CPU) — preferred
    faster_whisper.py      faster-whisper (CUDA/CPU) — fallback
    openai_whisper.py      openai-whisper (CPU) — last resort

Usage
─────
  python3 aisubs_whisper.py <media> <model> <language> <task> [out_file] [srt_path]

Output (stdout) — one JSON object per line
  {"type": "status", "msg": "..."}
  {"type": "sub", "i": N, "start": S, "end": E, "text": "..."}
  {"type": "done", "segments": N, "srt_path": "..."}
  {"type": "error", "msg": "..."}
"""

import sys
import os

from core.emitter import Emitter
from core.srt import write_srt
from backends import resolve_backend


def main():
    if len(sys.argv) < 5:
        sys.stderr.write(
            "Usage: aisubs_whisper.py <media> <model> <lang> <task> [out_file] [srt_path]\n"
        )
        sys.exit(1)

    media_path = sys.argv[1]
    model_name = sys.argv[2]
    language = sys.argv[3] if sys.argv[3] != "auto" else None
    task = sys.argv[4]
    mirror_file = sys.argv[5] if len(sys.argv) > 5 else None
    srt_requested = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6].strip() else None

    emitter = Emitter(mirror_file)

    if not os.path.isfile(media_path):
        emitter.emit({"type": "error", "msg": f"File not found: {media_path}"})
        emitter.close()
        sys.exit(1)

    try:
        backend = resolve_backend()
    except RuntimeError as exc:
        emitter.emit({"type": "error", "msg": str(exc)})
        emitter.close()
        sys.exit(1)

    # Resolve "recommended" → best model for this backend
    if model_name == "recommended":
        backend_name = backend.name().split()[0]  # e.g. "whisper.cpp", "whisperx", "faster-whisper"
        recommended_map = {
            "whisper.cpp":    "small",      # installer default model
            "whisperx":       "small",      # aligned, good balance
            "faster-whisper": "small",      # CUDA, fits 4GB VRAM
            "moonshine":      "base",       # fastest option is base
            "openai-whisper": "base",       # CPU-safe default
        }
        model_name = recommended_map.get(backend_name, "small")

    emitter.emit({
        "type": "status",
        "msg": f"Backend: {backend.name()} — {model_name} ({language or 'auto'}, {task})",
    })

    # Transcribe
    emitter.emit({"type": "status", "msg": "Transcribing..."})
    segments = []
    try:
        for seg in backend.transcribe(media_path, model_name, language, task):
            segment = {
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg["text"],
            }
            segments.append(segment)
            emitter.emit({
                "type": "sub",
                "i": len(segments),
                **segment,
            })
    except Exception as exc:
        import traceback
        emitter.emit({
            "type": "error",
            "msg": f"Transcription failed: {exc}\n{traceback.format_exc()}",
        })
        emitter.close()
        sys.exit(1)

    # Write SRT (skip if no segments — avoids empty .srt files)
    if not segments:
        emitter.emit({"type": "status", "msg": "No speech detected — skipping SRT."})
        emitter.emit({"type": "done", "segments": 0, "srt_path": None})
        emitter.close()
        sys.exit(0)

    try:
        srt_path = write_srt(segments, media_path, srt_requested)
    except OSError as exc:
        emitter.emit({"type": "error", "msg": f"Could not write SRT: {exc}"})
        emitter.close()
        sys.exit(1)

    emitter.emit({"type": "done", "segments": len(segments), "srt_path": srt_path})
    emitter.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        # Last-resort emit — no guarantee the Emitter is still functional
        try:
            print(
                '{"type": "error", "msg": "%s"}' % str(exc).replace('"', '\\"'),
                flush=True,
            )
        except OSError:
            pass
        sys.exit(1)