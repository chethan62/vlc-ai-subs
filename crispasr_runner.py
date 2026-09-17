"""CrispASR runner — one C++ ggml binary behind the plugin's JSONL protocol.

CrispASR (MIT, a whisper.cpp fork) runs Parakeet, Cohere Transcribe, Canary,
Granite, Qwen3-ASR and Voxtral from the same binary, with a **CTC forced
aligner** and no Python or PyTorch. Full survey and the measurements behind the
choices here: `.research/2026-09-16-asr-landscape-and-crispasr.md`.

What this runner adds on top of the binary is what the other engines also rely
on: our own audio-track selection (a multi-audio release is a measured problem —
`ffmpeg`'s default is the dub), the hallucination blocklist, and the published
cue standards via `apply_quality`. Measured on 90 s of the test film: CrispASR's
raw subtitle output had 8 lines over 42 characters (longest 76) and 2 cues over
7 s; after `apply_quality` it was 0 and 0, with all 179 words preserved.

Usage: crispasr_runner.py <media> <model> <lang> <task> [mirror_file] [srt_path]
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time

from core.audio import (choose_audio_stream, cleanup_temp, decode_to_wav16k,
                        list_audio_streams)
from core.crispasr_models import (INSTALL_HINT, align_enabled, binary,
                                  chunk_seconds, model_argument, model_for,
                                  model_tag)
from core.cues import apply_quality
from core.procs import install_termination_handler
from core.srt import write_srt


def emit(data: dict):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def _debug_enabled() -> bool:
    """VSCL_AISUBS_DEBUG=1 — the same convention the CLI and the backends use."""
    return os.environ.get("VSCL_AISUBS_DEBUG") == "1"


def _seconds(stamp: str) -> float:
    hh, mm, rest = stamp.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000


def parse_srt(path: str) -> list:
    """Read a .srt into [{start, end, text}] — the C++ binary's only cue output
    with per-cue timing (`-ojf` is word-level JSON for a different purpose)."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for block in fh.read().strip().split("\n\n"):
            lines = block.splitlines()
            if len(lines) >= 3 and " --> " in lines[1]:
                start, end = lines[1].split(" --> ")
                out.append({"start": _seconds(start.strip()), "end": _seconds(end.strip()),
                            "text": " ".join(lines[2:]).strip()})
    return out


def build_command(bin_path: str, wav_path: str, model_arg: str, backend: str,
                  language: str | None, out_base: str, align: bool,
                  device: str | None, chunk: int = 0) -> list:
    """The crispasr invocation, with the flags that decide quality and cost.

    * `-m` is `auto` for a backend's default model, or an explicit name/quant for
      a non-default variant (see `model_argument`).
    * `-sp` always: the binary's default output is ONE cue for the whole file
      (measured: an 86 s cue for 90 s of audio). Splitting at sentence
      punctuation is what makes it subtitle-shaped at all.
    * `-am auto -falign` only when the aligner is enabled: it turns decoder
      emission frames into real word timings, and costs +72% on CPU (measured:
      21.3 s -> 36.8 s for 90 s of audio, two runs each) but almost nothing on a
      GPU.
    * never force `--gpu-backend`: the CUDA tarball loads its backend at runtime
      and silently uses the CPU when the driver is absent, which is how one
      install scales from a CPU laptop to a GPU box. `device=cpu` asks for the
      whisper.cpp-compatible `-ng` instead.
    * `--chunk-seconds` unless the caller asked for 0: a 1200 s input with the
      aligner measured 8.3 GB RSS + 4.8 GB swap and an OOM kill, against ~300 MB
      for 90 s of the same material. Without it a feature film takes the machine
      down (see core/crispasr_models.chunk_seconds).
    """
    cmd = [bin_path, "--backend", backend, "-m", model_arg,
           "-f", wav_path, "-l", language or "auto",
           "-sp", "-osrt", "-of", out_base,
           "-t", str(min(8, os.cpu_count() or 2))]
    if chunk:
        cmd += ["--chunk-seconds", str(chunk)]
    if align:
        cmd += ["-am", "auto", "-falign"]
    if device == "cpu":
        cmd.append("-ng")
    return cmd


def failure_reason(returncode: int, stderr: str) -> str:
    """One actionable line for a non-zero exit from the binary.

    A crash is encoded as a negative return code (death by signal), and it writes
    to the kernel log rather than to stderr — so `rc=-11` with an empty tail is
    exactly the shape a segfault takes, and neither half of it tells the user
    anything. Measured 2026-09-17 on v0.8.33: a full-length file makes the binary
    request a 123.6 GB allocation, the kernel refuses it, and the NULL is
    dereferenced instead of handled (SIGSEGV in process_one_input).
    """
    tail = (stderr or "").strip()[-500:]
    if returncode < 0:
        try:
            why = f"killed by {signal.Signals(-returncode).name}"
        except ValueError:
            why = f"killed by signal {-returncode}"
        if not tail:
            tail = "no output on stderr (a crash reports to the kernel log)"
    else:
        why = f"rc={returncode}"
    return f"CrispASR failed ({why}): {tail}"


def main():
    t0 = time.time()
    if len(sys.argv) < 5:
        emit({"type": "error", "msg": "usage: crispasr_runner.py <media> <model> <lang> <task> [mirror] [srt]"})
        sys.exit(2)

    media_path = sys.argv[1]
    # argv[2] is the dialog's <model> (WhisperX's size names: tiny…large). This
    # engine has its own model menu, chosen by VRAM tier — VSCL_AISUBS_CRISPASR_MODEL
    # is how a user names one — so the argument is accepted and deliberately ignored
    # rather than mapped to a model that would not be the one the dialog asked for.
    language = None if sys.argv[3] in ("", "auto") else sys.argv[3]
    task = sys.argv[4]
    # argv[5] is the mirror path: the CLI owns that file (it tees our stdout), so
    # it is accepted for the shared calling convention and not used here.
    srt_requested = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6].strip() else None

    install_termination_handler()

    # CrispASR's --translate is whisper-only; the other backends transcribe.
    # Saying so beats transcribing a foreign film and labelling it a translation.
    if task == "translate":
        emit({"type": "error", "msg": (
            "CrispASR cannot translate — it transcribes. Use the WhisperX backend "
            "(VSCL_AISUBS_BACKEND=whisperx) for the translate task.")})
        sys.exit(1)

    bin_path = binary()
    if not bin_path:
        emit({"type": "error", "msg": f"CrispASR is not installed. Install it with:\n  {INSTALL_HINT}"})
        sys.exit(1)

    tag = model_tag(language)
    model = model_for(tag, language=language)
    align = align_enabled()
    device = (os.environ.get("VSCL_AISUBS_DEVICE", "").strip().lower() or None)

    emit({"type": "status", "msg": (
        f"CrispASR {model.label} ({model.backend}, {language or 'auto'}, {task}"
        f"{', CTC aligned' if align else ''}{', CPU' if device == 'cpu' else ''})")})

    streams = list_audio_streams(media_path)
    stream_index, why = choose_audio_stream(streams, language)
    if len(streams) > 1:
        emit({"type": "status", "msg": f"Audio track: {why}"})
    emit({"type": "status", "msg": f"CrispASR: decoding audio (+{time.time()-t0:.0f}s)"})
    wav_path = decode_to_wav16k(media_path, stream_index=stream_index)

    out_base = os.path.join(tempfile.gettempdir(), f"aisubs_crispasr_{os.getpid()}")
    try:
        cmd = build_command(bin_path, wav_path, model_argument(tag), model.backend,
                            language, out_base, align, device, chunk_seconds())
        if _debug_enabled():
            emit({"type": "status", "msg": f"CrispASR: {' '.join(cmd)}"})
        emit({"type": "status", "msg": "Transcribing..."})
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        if proc.returncode != 0:
            emit({"type": "error", "msg": failure_reason(proc.returncode, proc.stderr or "")})
            sys.exit(1)
        # The binary reports throughput itself; pass it through rather than
        # paraphrasing it (the earlier engines' lies all came from paraphrasing).
        for line in (proc.stderr or "").splitlines():
            if "transcribed" in line and _debug_enabled():
                emit({"type": "status", "msg": f"CrispASR: {line.strip()}"})

        srt_file = out_base + ".srt"
        if not os.path.isfile(srt_file):
            emit({"type": "status", "msg": "No speech detected."})
            emit({"type": "done", "segments": 0, "srt_path": None})
            return
        segments = parse_srt(srt_file)
        os.unlink(srt_file)
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

    # Same two steps every other engine gets: drop known hallucination segments,
    # then enforce the line width / reading speed / cue length standards.
    from core.blocklist import filter_segments
    segments = filter_segments(segments)
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

    if _debug_enabled():
        emit({"type": "status", "msg": f"CrispASR: {len(segments)} cues in {time.time()-t0:.0f}s"})
    emit({"type": "done", "segments": len(segments), "srt_path": srt_path})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        emit({"type": "error", "msg": f"{exc}\n{traceback.format_exc()}"})
        sys.exit(1)
