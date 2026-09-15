#!/usr/bin/env python3
"""
whisper.cpp runner — the GPU path for machines without CUDA.

whisper.cpp is the only Whisper runtime with a Vulkan backend, and Vulkan is
vendor-neutral: the same binary drives NVIDIA, AMD and Intel GPUs, and falls
back to CPU when no device is present. This runner drives the ``whisper-cli``
binary, parses its JSON output, and emits the plugin's usual JSONL contract —
so the engine slots in beside WhisperX and Parakeet with no Lua changes.

It needs no Python ML packages at all: any Python 3 + ffmpeg + the binary.
Timing is segment-level (whisper.cpp has no wav2vec2 aligner), so cues come out
of the segment timestamps and the shared cue-quality pass in core/cues.py.

Contract (stdout, JSONL) — same as whisperx_runner:
  {"type": "status", "msg": "..."}
  {"type": "sub", "i": N, "start": S, "end": E, "text": "..."}
  {"type": "done", "segments": N, "srt_path": "..."}
  {"type": "error", "msg": "..."}

Args:  <media> <model> <language> <task> [mirror_file] [srt_path]
  <model> is a whisper size (tiny/base/small/medium/large/large-v3-turbo) or
  "recommended" (best installed model, largest first).
  <task> translate uses whisper.cpp's built-in `-tr` (Whisper translate, not
  the NLLB cascade the WhisperX engine runs).
Env:  VSCL_AISUBS_DEVICE=cpu passes `-ng` (no GPU); VSCL_AISUBS_WHISPERCPP_BIN
  and VSCL_AISUBS_WHISPERCPP_MODEL override the binary / model path.
The SRT file is written ONLY when [srt_path] is given — the plugin's caller
(aisubs_whisper.py) owns SRT output, so the runner never creates side-effect
files next to the media (realtime-OSD mode, read-only media dirs).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

from core.audio import decode_to_wav16k
from core.cues import apply_quality
from core.procs import run_captured
from core.srt import write_srt

# Dialog model choice → ggml file (matches install-whisper-cpp.sh).
MODEL_FILES = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large": "ggml-large-v3-turbo.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin",
}
# "recommended" → the best model that is actually installed, largest first.
MODEL_PREFERENCE = ("large-v3-turbo", "large", "medium", "small", "base", "tiny")

BINARY_CANDIDATES = (
    "~/.local/share/whisper-cpp/whisper-cli",
    "~/.local/bin/whisper-cli",
    "/usr/local/bin/whisper-cli",
    "/usr/bin/whisper-cli",
)
CONF_FILE = "~/.local/share/whisper-cpp/vlc-ai-subs.conf"
# whisper.cpp's JSON reports offsets in MILLISECONDS (not seconds).
_MS = 1000.0


def emit(data: dict):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def _conf_value(key: str) -> str | None:
    """Read one key from the installer's config file (whisper_bin=..., ...)."""
    path = os.path.expanduser(CONF_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                name, _, value = line.partition("=")
                if name.strip() == key and value.strip():
                    return os.path.expanduser(value.strip())
    except OSError:
        pass
    return None


def binary_path() -> str | None:
    """whisper-cli: env override → installer config → conventional locations.

    Every source is existence-checked (including the env override): a typo must
    surface as "not found — run install-whisper-cpp.sh", not as a confusing
    rc!=0 from a spawn attempt.
    """
    env_bin = os.environ.get("VSCL_AISUBS_WHISPERCPP_BIN", "").strip()
    if env_bin:
        path = os.path.expanduser(env_bin)
        return path if os.path.isfile(path) and os.access(path, os.X_OK) else None
    for candidate in (_conf_value("whisper_bin"), *BINARY_CANDIDATES):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)
    found = shutil.which("whisper-cli")
    return os.path.abspath(found) if found else None


def model_dirs(binary: str | None) -> list[str]:
    """Where to look for ggml-*.bin (installer puts them next to the binary)."""
    dirs = []
    env_model = os.environ.get("VSCL_AISUBS_WHISPERCPP_MODEL", "").strip()
    if env_model:
        env_model = os.path.expanduser(env_model)
        dirs.append(env_model if os.path.isdir(env_model) else os.path.dirname(env_model))
    if binary:
        dirs.append(os.path.dirname(binary))
    dirs.append(os.path.expanduser("~/.local/share/whisper-cpp"))
    dirs.append(os.path.expanduser("~/.local/share/whisper-cpp/models"))
    conf_model = _conf_value("whisper_model")
    if conf_model:
        dirs.append(os.path.dirname(conf_model))
    seen, unique = set(), []
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def installed_models(dirs: list[str]) -> dict[str, str]:
    """{model name: path} for every ggml file found, first dir wins."""
    found: dict[str, str] = {}
    for name, filename in MODEL_FILES.items():
        if name in found:
            continue
        for d in dirs:
            path = os.path.join(d, filename)
            if os.path.isfile(path):
                found[name] = path
                break
    return found


def resolve_model(requested: str | None, dirs: list[str]) -> tuple[str, str]:
    """Return (path, name) for the requested model, or raise with the fix."""
    available = installed_models(dirs)
    if not available:
        raise RuntimeError(
            "No whisper.cpp model found — run ./install-whisper-cpp.sh small "
            f"(looked in: {', '.join(dirs)})"
        )
    want = (requested or "").strip().lower()
    if want in ("", "recommended", "auto"):
        for name in MODEL_PREFERENCE:
            if name in available:
                return available[name], name
    if want in available:
        return available[want], want
    if want and want in MODEL_FILES:
        raise RuntimeError(
            f"whisper.cpp model '{want}' is not installed — "
            f"run ./install-whisper-cpp.sh {want} "
            f"(installed: {', '.join(sorted(available))})"
        )
    # Unknown name (e.g. a WhisperX-only pick): fall back to the best installed.
    for name in MODEL_PREFERENCE:
        if name in available:
            return available[name], name
    raise RuntimeError(  # pragma: no cover — available is non-empty here
        f"No usable whisper.cpp model in {', '.join(dirs)}"
    )


def build_args(
    binary: str,
    model_path: str,
    wav_path: str,
    out_prefix: str,
    language: str | None,
    task: str,
    device: str = "auto",
    threads: int | None = None,
) -> list[str]:
    """whisper-cli argv (kept separate so it is unit-tested without the binary)."""
    args = [
        binary, "-m", model_path, "-f", wav_path,
        "-oj", "-of", out_prefix,           # JSON next to out_prefix
        "-t", str(threads or min(8, os.cpu_count() or 2)),
        "-np",                              # no per-token prints
    ]
    if device == "cpu":
        args.append("-ng")                  # -ng: disable the GPU (Vulkan)
    if language and language != "auto":
        args += ["-l", language]
    if task == "translate":
        args.append("-tr")                  # whisper.cpp's built-in translate
    return args


def vulkan_status(stderr: str, gpu_used: bool = True) -> str | None:
    """GPU/CPU summary from whisper.cpp's ggml_vulkan log lines.

    whisper.cpp prints e.g. "ggml_vulkan: Found 1 Vulkan devices:" followed by
    "ggml_vulkan: 0 = NVIDIA GeForce GTX 1650 (NVIDIA) | ...". It enumerates
    devices even when the GPU is disabled with -ng, so *gpu_used* decides what
    we report — claiming a GPU for a CPU run would be a lie.
    """
    if not gpu_used:
        return "GPU disabled (VSCL_AISUBS_DEVICE=cpu) — running on CPU"
    if not stderr:
        return None
    for line in stderr.splitlines():
        if "ggml_vulkan:" in line and "Found" in line and "0 Vulkan devices" in line:
            return "Vulkan: no device — running on CPU"
    devices = [l.split("=", 1)[1].split("|")[0].strip()
               for l in stderr.splitlines()
               if "ggml_vulkan:" in l and " = " in l]
    device_lines = [d for d in devices if d]
    if device_lines:
        return "Vulkan GPU: " + device_lines[0]
    return None


def parse_transcription(payload: dict) -> list[dict]:
    """whisper.cpp JSON → [{start, end, text}] in seconds.

    Shape (v1.9): {"transcription": [{"offsets": {"from": ms, "to": ms},
    "timestamps": {...}, "text": " ..."}, ...]}
    """
    out = []
    for seg in (payload or {}).get("transcription") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        offsets = seg.get("offsets") or {}
        start = float(offsets.get("from", 0)) / _MS
        end = float(offsets.get("to", offsets.get("from", 0))) / _MS
        out.append({"start": start, "end": end, "text": text})
    return out


def transcribe(
    binary: str,
    model_path: str,
    wav_path: str,
    language: str | None,
    task: str,
    device: str = "auto",
):
    """Run whisper-cli, return (segments, vulkan_status, elapsed seconds)."""
    fd, out_prefix = tempfile.mkstemp(prefix="aisubs_whispercpp_")
    os.close(fd)
    os.unlink(out_prefix)
    json_path = out_prefix + ".json"
    t0 = time.time()
    try:
        proc = run_captured(
            build_args(binary, model_path, wav_path, out_prefix, language, task, device),
            timeout=None,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            raise RuntimeError(
                f"whisper-cli failed (rc={proc.returncode}): {' / '.join(tail)[:400]}"
            )
        try:
            with open(json_path, encoding="utf-8") as f:
                payload = json.load(f)
        except OSError as exc:
            raise RuntimeError(
                f"whisper-cli wrote no JSON output ({exc}); stderr: "
                f"{(proc.stderr or '').strip()[-300:]}"
            ) from exc
        return (
            parse_transcription(payload),
            vulkan_status(proc.stderr or "", gpu_used=device != "cpu"),
            time.time() - t0,
        )
    finally:
        for path in (out_prefix, json_path):
            try:
                os.remove(path)
            except OSError:
                pass


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

    binary = binary_path()
    if not binary:
        emit({
            "type": "error",
            "msg": "whisper-cli not found — run ./install-whisper-cpp.sh "
                   "(sets up the Vulkan build for AMD/Intel/NVIDIA GPUs)",
        })
        sys.exit(1)

    try:
        model_path, resolved_model = resolve_model(model_name, model_dirs(binary))
    except RuntimeError as exc:
        emit({"type": "error", "msg": str(exc)})
        sys.exit(1)

    device = os.environ.get("VSCL_AISUBS_DEVICE", "").strip().lower()
    device = "cpu" if device == "cpu" else "auto"
    emit({
        "type": "status",
        "msg": f"whisper.cpp: {resolved_model} ({'CPU' if device == 'cpu' else 'auto device'})...",
    })

    try:
        wav_path = decode_to_wav16k(media_path)
    except RuntimeError as exc:
        emit({"type": "error", "msg": str(exc)})
        sys.exit(1)

    try:
        segments, gpu_note, elapsed = transcribe(
            binary, model_path, wav_path, language, task, device
        )
    except RuntimeError as exc:
        emit({"type": "error", "msg": str(exc)})
        sys.exit(1)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    if gpu_note:
        emit({"type": "status", "msg": f"whisper.cpp: {gpu_note}"})
    emit({"type": "status", "msg": f"whisper.cpp: transcription done (+{elapsed:.0f}s)"})

    from core.blocklist import filter_segments
    segments = apply_quality(filter_segments(segments))

    if not segments:
        emit({"type": "status", "msg": "No speech detected."})
        emit({"type": "done", "segments": 0, "srt_path": None})
        return

    for i, seg in enumerate(segments, 1):
        emit({
            "type": "sub", "i": i,
            "start": round(seg["start"], 3), "end": round(seg["end"], 3),
            "text": seg["text"],
        })

    srt_path = None
    if srt_requested:
        try:
            srt_path = write_srt(segments, media_path, srt_requested)
        except OSError as exc:
            emit({"type": "error", "msg": f"Could not write SRT: {exc}"})
            sys.exit(1)

    emit({"type": "done", "segments": len(segments), "srt_path": srt_path})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        emit({"type": "error", "msg": f"{exc}\n{traceback.format_exc()}"})
        sys.exit(1)
