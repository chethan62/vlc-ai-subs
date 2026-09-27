"""Photon (Moondream) runner — Parakeet Redux behind the plugin's JSONL protocol.

Runs INSIDE `venv-photon` (PyTorch + `moondream`; see `core/photon_models.py` for
why this engine gets its own venv). Everything the other engines rely on is
reused rather than re-implemented: our own audio-track selection (ffmpeg's
default stream is the dub — a measured trap), the hallucination blocklist, the
published cue standards via `apply_quality`, and `core.srt.write_srt`.

What is specific to this engine:

* **The device is resolved here, not in the CLI.** `VSCL_AISUBS_DEVICE` is
  honoured, but `auto` prefers whichever device has a native packed-ternary
  kernel (Metal on Apple silicon, the x86/neon int8 paths on CPU) — CUDA has
  none and measured 2-2.4x slower than this laptop's CPU.
* **Segment timestamps only.** Photon can return word timings; subtitles need
  cue spans, and `apply_quality` re-times and re-wraps from the segment level.
* **No chunking.** The engine self-segments with its own VAD head and the model
  is 178 MB, so a film is one call. Photon's peak RSS on long media is
  unmeasured here — see the note in the README before trusting it on a feature
  film with a small machine.

Usage: photon_runner.py <media> <model> <lang> <task> [mirror_file] [srt_path]
"""

import json
import os
import sys
import time

from core.audio import (choose_audio_stream, cleanup_temp, decode_to_wav16k,
                        list_audio_streams)
from core.photon_models import (INSTALL_HINT, device_note, language_warning,
                                model_id, resolve_device, venv_python)
from core.procs import install_termination_handler
from core.srt import write_srt


def emit(data: dict):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def _debug_enabled() -> bool:
    """VSCL_AISUBS_DEBUG=1 — the same convention the CLI and the backends use."""
    return os.environ.get("VSCL_AISUBS_DEBUG") == "1"


def available_devices(torch_module) -> list:
    """Every device the runtime can actually use, CPU always included."""
    devices = ["cpu"]
    try:
        if torch_module.backends.mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    try:
        if torch_module.cuda.is_available():
            devices.append("cuda")
    except Exception:
        pass
    return devices


def native_kernels(ternary_module, devices) -> dict:
    """Which of these devices have a packed ternary kernel (engine's own API).

    `kestrel_kernels.ternary` is a public entry point of the installed engine; if
    it is missing or renamed, the facts are simply unknown and every device reads
    False — which sends `auto` down the non-CUDA path, the safe direction.
    """
    facts = {}
    if ternary_module is None:
        return {d: False for d in devices}
    for device in devices:
        base = device.split(":")[0]
        try:
            if base == "mps":
                facts[device] = bool(ternary_module.metal_gemm_ready())
            elif base == "cpu":
                facts[device] = bool(ternary_module.ternary_gemm_ready())
            else:
                facts[device] = False
        except Exception:
            facts[device] = False
    return facts


def to_segments(result: dict) -> list:
    """Photon's segments -> the plugin's {start, end, text} cue dicts.

    Blank segments are dropped here rather than left for the cue pass: an empty
    cue is not silence to be timed, it is a hole.
    """
    out = []
    for seg in result.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": text})
    return out


def main():
    t0 = time.time()
    if len(sys.argv) < 5:
        emit({"type": "error", "msg": "usage: photon_runner.py <media> <model> <lang> <task> [mirror] [srt]"})
        sys.exit(2)

    media_path = sys.argv[1]
    # argv[2] is the dialog's <model> dropdown (WhisperX's size names). This
    # engine's model is chosen by VSCL_AISUBS_PHOTON_MODEL (redux | ultra), so
    # the argument is accepted and deliberately ignored rather than mapped onto
    # a name that is not the one loading — the Parakeet/CrispASR rule.
    language = None if sys.argv[3] in ("", "auto") else sys.argv[3]
    task = sys.argv[4]
    # argv[5] is the mirror path: the CLI owns that file (it tees our stdout).
    srt_requested = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6].strip() else None

    install_termination_handler()

    if task == "translate":
        emit({"type": "error", "msg": (
            "Photon transcribes, it does not translate. Use the WhisperX backend "
            "(VSCL_AISUBS_BACKEND=whisperx) for the translate task.")})
        sys.exit(1)

    try:
        import torch
        import moondream as md
    except ImportError as exc:
        emit({"type": "error", "msg": (
            f"Photon is not installed in {venv_python() or 'its venv'} ({exc}). "
            f"Install it with:\n  {INSTALL_HINT}")})
        sys.exit(1)

    try:
        from kestrel_kernels import ternary
    except Exception:
        ternary = None

    devices = available_devices(torch)
    native = native_kernels(ternary, devices)
    try:
        device = resolve_device(os.environ.get("VSCL_AISUBS_DEVICE", ""), devices, native)
    except ValueError as exc:
        emit({"type": "error", "msg": f"Photon: {exc}"})
        sys.exit(1)

    repo = model_id()
    emit({"type": "status", "msg": (
        f"Photon {repo} ({language or 'auto'}, {task}, {device}"
        f"{', native ternary kernel' if native.get(device) else ', dense fallback'})")})
    if not native.get(device):
        note = device_note(device)
        if note:
            emit({"type": "status", "msg": f"Photon: {note}"})
    warning = language_warning(language)
    if warning:
        emit({"type": "status", "msg": f"Photon: {warning}"})

    streams = list_audio_streams(media_path)
    stream_index, why = choose_audio_stream(streams, language)
    if len(streams) > 1:
        emit({"type": "status", "msg": f"Audio track: {why}"})
    emit({"type": "status", "msg": f"Photon: decoding audio (+{time.time()-t0:.0f}s)"})
    wav_path = decode_to_wav16k(media_path, stream_index=stream_index)
    # 16 kHz mono 16-bit = 32 kB/s. Reported so a long run's cost is visible in
    # the status feed, not guessed at afterwards.
    seconds = os.path.getsize(wav_path) / 32000

    try:
        emit({"type": "status", "msg": f"Transcribing {seconds:.0f}s of audio..."})
        load0 = time.time()
        with md.photon(repo, device=device) as speech:
            emit({"type": "status", "msg": f"Photon: model loaded in {time.time()-load0:.1f}s"})
            # Timed from here, not from before the load: the one-time load is
            # reported on its own line, and mixing it into the throughput figure
            # is how a short clip reads slower than the engine really is.
            started = time.time()
            result = speech.transcribe(audio=str(wav_path), timestamps="segment")
        elapsed = time.time() - started
        segments = to_segments(result)
    except Exception as exc:
        emit({"type": "error", "msg": f"Photon failed: {exc}"})
        sys.exit(1)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        cleanup_temp()

    if not segments:
        emit({"type": "status", "msg": "No speech detected."})
        emit({"type": "done", "segments": 0, "srt_path": None})
        return

    # The same two steps every other engine gets: drop known hallucination
    # segments, then enforce line width / reading speed / cue length standards.
    from core.blocklist import filter_segments
    segments = filter_segments(segments)
    from core.cues import apply_quality
    segments = apply_quality(segments, language=language)

    for i, seg in enumerate(segments, 1):
        emit({"type": "sub", "i": i, "start": round(seg["start"], 3),
              "end": round(seg["end"], 3), "text": seg["text"]})

    srt_path = None
    if srt_requested:
        try:
            srt_path = write_srt(segments, media_path, srt_requested)
        except OSError as exc:
            emit({"type": "error", "msg": f"Could not write SRT: {exc}"})
            sys.exit(1)

    emit({"type": "status", "msg": (
        f"Photon: {len(segments)} cues, {elapsed:.1f}s for {seconds:.0f}s of audio "
        f"({seconds / elapsed:.2f}x realtime)")})
    emit({"type": "done", "segments": len(segments), "srt_path": srt_path})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        emit({"type": "error", "msg": f"{exc}\n{traceback.format_exc()}"})
        sys.exit(1)
