#!/usr/bin/env python3
"""
vlc-ai-subs — Whisper transcription backend.

Transcribes audio from a media file using faster-whisper (or openai-whisper
as fallback) and streams results as JSON lines to stdout. Also writes a
standard SRT subtitle file.

Improvements over upstream:
  * GPU/CUDA auto-detection — uses the fastest available device and the best
    compute type for it (int8_float16 on GPU for 4GB cards, float32 on CPU).
  * Device / compute-type / model-cache / output-path override via env vars:
        VSCL_AISUBS_DEVICE       cuda|cpu|auto        (default auto)
        VSCL_AISUBS_COMPUTE      int8_float16|int8_float32|float16|float32|...
                                 (default auto-picked)
        VSCL_AISUBS_MODEL_CACHE  directory for HF model downloads
                                 (default ~/.cache/huggingface)
  * Optionally controlled SRT output path as argv[6]; defaults to media dir.

Usage:
    python3 aisubs_whisper.py <media> <model> <language> <task> [out_file] [srt_path]

Arguments:
    media_path  Path to the video/audio file
    model       Whisper model size: tiny, base, small, medium, large
    language    Language code (e.g. en, es, hi) or "auto" for detection
    task        "transcribe" or "translate" (translate outputs English)
    out_file    Optional: JSONL output file that stdout is mirrored to
    srt_path    Optional: where to write the .srt (defaults to media_dir/<media>.srt)

Output (stdout):
    One JSON object per line:
      {"type": "status", "msg": "..."}           — progress updates
      {"type": "sub", "i": N, "start": S, "end": E, "text": "..."}  — subtitle
      {"type": "done", "segments": N, "srt_path": "..."}            — finished
      {"type": "error", "msg": "..."}            — fatal error
"""

import sys
import os
import json


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


_out_file = None  # optional output file set in main()


def emit(data: dict) -> None:
    """Write a JSON line to stdout and to the output file if set."""
    line = json.dumps(data, ensure_ascii=False)
    print(line, flush=True)
    if _out_file:
        try:
            _out_file.write(line + "\n")
            _out_file.flush()
        except Exception:
            pass


def _cuda_lib_search_dirs():
    """Return a list of directories likely to contain CUDA runtime libs.

    Cheap, dependency-free discovery used only to pre-load the shared libs
    ctranslate2 needs, so the plugin works even when VLC launches it without
    LD_LIBRARY_PATH set.
    """
    dirs = []
    # CUDA_HOME / CUDA_PATH (then lib, lib64)
    for var in ("CUDA_HOME", "CUDA_PATH"):
        base = os.environ.get(var)
        if base:
            dirs += [os.path.join(base, "lib"), os.path.join(base, "lib64")]
    # nice, well-known override
    cu12 = "/usr/local/cuda/lib64"
    # one-user-level fallback covering the CUDA-canonical /usr/local/cuda/lib*
    for sub in ("lib", "lib64"):
        dirs += [os.path.join("/usr/local/cuda", sub)]
    # user-level ~/.local/cuda12  (matches this box: ~/.local/cuda12/lib)
    for sub in ("lib", "lib64"):
        dirs += [os.path.expanduser(os.path.join("~/.local/cuda12", sub))]
    # also check `nvidia-ctk` style flat layout and ldconfig
    extra = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/wsl/lib",
        "/opt/cuda/lib64",
    ]
    for d in extra:
        if not os.path.isdir(d):
            continue
        # only include if it actually looks like CUDA (has cublas)
        if os.path.exists(os.path.join(d, "libcublas.so.12")) or \
           os.path.exists(os.path.join(d, "libcublas.so.13")):
            dirs.append(d)
    return dirs


def _preload_cuda_libs():
    """Load the CUDA runtime shared libs ctranslate2 needs (via ctypes)."""
    # If the env explicitly says CPU, skip CUDA entirely.
    if os.environ.get("VSCL_AISUBS_DEVICE", "").strip().lower() in ("cpu",):
        return

    # Sanity: only preload when a CUDA device is actually present.
    try:
        import ctypes
        from ctypes.util import find_library
    except Exception:
        return

    # We look for the exact sonames CTranslate2 dlopens at import time.
    wanted = [
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcudart.so.12",
        "libnvblas.so.12",
    ]

    def candidate_paths(name):
        """Absolute paths worth trying for a given soname."""
        for libdir in _cuda_lib_search_dirs():
            yield os.path.join(libdir, name)
        # fall back to what the system loader reports last
        p = find_library(name.split('.')[0])
        if p:
            yield p

    loaded_any = False
    for name in wanted:
        for p in candidate_paths(name):
            if os.path.isfile(p):
                try:
                    ctypes.CDLL(p)  # raises FileNotFoundError/OSError on failure
                    loaded_any = True
                    break
                except OSError:
                    continue
    return loaded_any


def detect_device():
    """Choose the best GPU and compute type for the current hardware.

    CUDA is preferred when ctranslate2 sees a device; falls back to CPU.
    On GPU, int8_float16 keeps a <4GB card under VRAM while staying fast.
    """
    # Try to preload CUDA so ctranslate2 can initialise without env vars.
    _preload_cuda_libs()

    env_dev = os.environ.get("VSCL_AISUBS_DEVICE", "").strip().lower()
    if env_dev:
        device = env_dev
    else:
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

    env_ct = os.environ.get("VSCL_AISUBS_COMPUTE", "").strip().lower()
    if env_ct:
        if env_ct == "compute":
            compute = "default"
        else:
            compute = env_ct
    elif device == "cuda":
        compute = "int8_float16"
    else:
        compute = "float32"

    return device, compute


def transcribe_faster_whisper(media_path, model_name, lang, task):
    """Transcribe using faster-whisper (CTranslate2 backend)."""
    from faster_whisper import WhisperModel

    device, compute = detect_device()
    emit({"type": "status", "msg": f"Loading {model_name} on {device} ({compute})..."})

    kw = dict(device=device, compute_type=compute)
    model_cache = os.environ.get("VSCL_AISUBS_MODEL_CACHE", "").strip()
    if model_cache:
        kw["download_root"] = model_cache

    model = WhisperModel(model_name, **kw)

    emit({"type": "status", "msg": "Transcribing..."})
    segments_gen, _info = model.transcribe(
        media_path,
        language=lang,
        task=task,
        beam_size=(5 if device == "cuda" else 1),
        vad_filter=True,
        vad_parameters={
            "threshold": 0.05,
            "min_silence_duration_ms": 200,
            "speech_pad_ms": 600,
            "min_speech_duration_ms": 50,
        },
    )

    for seg in segments_gen:
        text = seg.text.strip()
        if text:
            yield {"start": seg.start, "end": seg.end, "text": text}


def transcribe_openai_whisper(media_path, model_name, lang, task):
    """Transcribe using openai-whisper (fallback)."""
    import whisper

    emit({"type": "status", "msg": f"Loading {model_name} model..."})
    model = whisper.load_model(model_name)

    emit({"type": "status", "msg": "Transcribing (batch mode)..."})
    options = {"task": task}
    if lang:
        options["language"] = lang
    result = model.transcribe(media_path, **options)

    for seg in result["segments"]:
        text = seg["text"].strip()
        if text:
            yield {"start": seg["start"], "end": seg["end"], "text": text}


def main():
    global _out_file

    if len(sys.argv) < 5:
        emit({"type": "error", "msg": (
            "Usage: aisubs_whisper.py <media> <model> <lang> <task> [out_file] [srt_path]"
        )})
        sys.exit(1)

    media_path = sys.argv[1]
    model_name = sys.argv[2]
    language = sys.argv[3] if sys.argv[3] != "auto" else None
    task = sys.argv[4]

    # Optional output file — Lua passes this so output is captured without shell redirection
    if len(sys.argv) > 5:
        try:
            _out_file = open(sys.argv[5], "w", encoding="utf-8", buffering=1)
        except Exception as e:
            pass  # if we can't open it, stdout-only mode

    # Optional explicit SRT path
    srt_requested = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6].strip() else None

    if not os.path.isfile(media_path):
        emit({"type": "error", "msg": f"File not found: {media_path}"})
        sys.exit(1)

    # Detect backend
    backend = None
    try:
        import faster_whisper  # noqa: F401
        backend = "faster-whisper"
    except ImportError:
        pass

    if not backend:
        try:
            import whisper  # noqa: F401
            backend = "openai-whisper"
        except ImportError:
            emit({"type": "error", "msg": "No Whisper backend found. Run: pip install faster-whisper"})
            sys.exit(1)

    # Choose transcription function
    if backend == "faster-whisper":
        segments_iter = transcribe_faster_whisper(media_path, model_name, language, task)
    else:
        segments_iter = transcribe_openai_whisper(media_path, model_name, language, task)

    # Stream segments and build SRT
    srt_lines = []
    count = 0

    try:
        for seg in segments_iter:
            count += 1
            emit({
                "type": "sub",
                "i": count,
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg["text"],
            })
            srt_lines.append(
                f"{count}\n"
                f"{format_srt_timestamp(seg['start'])} --> {format_srt_timestamp(seg['end'])}\n"
                f"{seg['text']}\n"
            )
    except Exception as e:
        import traceback
        emit({"type": "error", "msg": f"Transcription failed: {e}\n{traceback.format_exc()}"})
        sys.exit(1)

    # Write SRT file (explicit path if given, else next to the media)
    if srt_requested:
        srt_path = srt_requested
        os.makedirs(os.path.dirname(srt_path) or ".", exist_ok=True)
    else:
        base, _ = os.path.splitext(media_path)
        srt_path = base + ".srt"

    try:
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))
    except Exception as e:
        emit({"type": "error", "msg": f"Could not write SRT: {e}"})
        sys.exit(1)

    emit({"type": "done", "segments": count, "srt_path": srt_path})

    if _out_file:
        _out_file.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        emit({"type": "error", "msg": str(e) + "\n" + traceback.format_exc()})
        if _out_file:
            _out_file.close()
        sys.exit(1)