#!/usr/bin/env python3
"""
vlc-ai-subs — Whisper transcription backend for VLC.

Engines: WhisperX (default) · Parakeet-TDT v2/v3 via sherpa-onnx · whisper.cpp
(Vulkan). Plugin code is MIT (see LICENSE); models and runtimes are downloaded at
install time and keep their own licenses — the full list is in the README's
Credits section (Parakeet is CC-BY-4.0, © NVIDIA, ONNX conversion by k2-fsa;
sherpa-onnx is Apache-2.0).

Architecture
────────────
  aisubs_whisper.py        CLI entry-point (you are here)
  core/
    emitter.py             JSONL output + file mirror for Lua polling
    srt.py                 SRT timestamp formatting and file writing
    cues.py                cue line-wrapping + timing quality pass
    blocklist.py           hallucination-phrase filter
    procs.py               live-child registry (cancellation)
    timeouts.py            subprocess ceilings (VSCL_AISUBS_TIMEOUT)
  backends/
    base.py                Abstract TranscriptionBackend
    whisperx_backend.py    WhisperX (word-aligned, Python 3.12 subprocess) — default
    parakeet.py            Parakeet TDT via sherpa-onnx (English, CPU, ~10x faster)

Usage
─────
  python3 aisubs_whisper.py <media> <model> <language> <task> [out_file] [srt_path]

Output (stdout) — one JSON object per line
  {"type": "status", "msg": "..."}
  {"type": "sub", "i": N, "start": S, "end": E, "text": "..."}
  {"type": "done", "segments": N, "srt_path": "..."}
  {"type": "error", "msg": "..."}
"""

import atexit
import os
import signal
import sys
import time
import traceback

from core.audio import choose_audio_stream, list_audio_streams, sweep_stale_temp
from core.emitter import Emitter
from core.procs import terminate_all
from core.srt import write_srt
from core.blocklist import filter_segments
from backends import resolve_backend

# ── Debug logging ────────────────────────────────────────────────────
# Enable with a trailing `--debug` CLI arg or VSCL_AISUBS_DEBUG=1.
# Debug lines go to stderr AND /tmp/aisubs_debug.log (VLC itself shows
# stderr in its logs; the file survives terminal restarts).

DEBUG_FILE = "/tmp/aisubs_debug.log"


def _debug_enabled() -> bool:
    return os.environ.get("VSCL_AISUBS_DEBUG") == "1" or "--debug" in sys.argv


def _log_debug(msg: str) -> None:
    line = f"[debug {time.strftime('%H:%M:%S')}] {msg}"
    sys.stderr.write(line + "\n")
    try:
        with open(DEBUG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def friendly_error(exc: BaseException) -> str:
    """One readable, actionable line for the dialog; detail stays in the logs.

    The dialog renders this in a status label. A failed backend run embeds its
    whole stderr and stdout in the exception, so a real CUDA OOM reached the user
    as thousands of characters of pyannote warnings plus a traceback — measured
    in a live VLC run, where the dialog showed that wall of text.
    """
    raw = str(exc)
    first = " ".join(raw.split()).strip() or exc.__class__.__name__
    first = first.split("Traceback")[0].strip() or first
    if len(first) > 200:
        first = first[:200].rstrip() + "…"
    if "out of memory" in raw.lower():
        return (first + " | Out of GPU memory (VLC itself holds some): close other "
                       "video windows, choose a smaller model, or use Parakeet")
    return first


# ── Hardware-aware model recommendation ──────────────────────────────────

def _detect_vram_mb() -> int:
    """Return GPU VRAM in MiB via nvidia-smi, or 0 if detection fails."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
            text=True, timeout=5,
        )
        return int(out.strip().split()[0])
    except Exception:
        return 0


def _detect_ram_gb() -> int:
    """Return system RAM in GiB from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb // (1024 * 1024)  # KB → GiB
    except Exception:
        pass
    return 4  # conservative default


def _recommend_model(backend_name: str = "whisperx") -> str:
    """Pick the model based on GPU VRAM (system RAM on CPU).

    WhisperX transcribes via faster-whisper (CTranslate2, int8_float16 on
    CUDA) — roughly 2× the VRAM footprint of GGML models. When a GPU is
    detected, only VRAM tiering applies (never fall through to RAM sizing,
    which could over-recommend for a small GPU).

    Research (2026-08): large-v3-turbo (809M) is WhisperX's accuracy/speed
    sweet spot — near-large WER at ~4× the speed, ~1.5-1.8GB VRAM int8.
    """
    vram_mb = _detect_vram_mb()
    if vram_mb > 0:
        if vram_mb >= 8000:   return "large"
        elif vram_mb >= 4000: return "large-v3-turbo"
        elif vram_mb >= 2000: return "small"
        return "base"

    # No usable GPU — CPU path, bound by system RAM
    ram_gb = _detect_ram_gb()
    if ram_gb >= 8:   return "medium"
    elif ram_gb >= 4: return "small"
    return "base"


def resolve_model_name(
    model_name: str, backend_name: str, backend=None, language: str | None = None
) -> str:
    """Map the dialog's model choice to the model that will actually run.

    An engine that ignores the dialog's pick reports its own label through
    ``TranscriptionBackend.model_label()`` (Parakeet: an installed variant that
    depends on the language; whisper.cpp: an installed ggml file) — without
    that, status lines named models that were not running (e.g.
    "parakeet — large-v3-turbo"). Otherwise "recommended" is sized from
    VRAM/RAM and explicit picks pass through.
    """
    if backend is not None:
        label = backend.model_label(model_name, language)
        if label:
            return label
    if model_name == "recommended":
        return _recommend_model(backend_name)
    return model_name


# ── Cancellation ─────────────────────────────────────────────────────────

PID_SUFFIX = ".pid"


def _remove_pid_file(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _write_pid_file(mirror_file: str | None) -> str | None:
    """Record this process's PID next to the mirror file.

    The VLC extension has no process API: it cancels a run by killing the PID
    found in ``<mirror>.pid``, and the SIGTERM handler below then stops the ML
    subprocess — that child is not in a process group the extension can signal
    (VLC launches us via ``sh -c '... &'``, in VLC's own group).
    """
    if not mirror_file:
        return None
    path = mirror_file + PID_SUFFIX
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except OSError:
        return None
    return path


def _install_cancel_handler() -> None:
    """SIGTERM/SIGINT → stop the backend child, then exit non-zero."""

    def handler(signum, _frame):
        killed = terminate_all()
        sys.stderr.write(
            f"[aisubs] cancelled (signal {signum}); stopped {killed} subprocess(es)\n"
        )
        raise SystemExit(130)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):  # not the main thread / unsupported
            pass


# ── CLI entry-point ──────────────────────────────────────────────────────

def auto_engine_for(language: str | None, task: str = "transcribe") -> str:
    """Engine for ``VSCL_AISUBS_BACKEND`` unset/auto: "parakeet" or "auto".

    Auto prefers Parakeet for a **known** language an installed variant covers —
    English uses the v2 model, v3's 25 European languages use v3 — because it is
    ~10x faster with better English WER than WhisperX. Everything else falls
    through to the hardware policy in ``backends.resolve_backend("auto")``:
    NVIDIA → WhisperX, Vulkan-only → whisper.cpp, otherwise CPU.

    An *unspecified* language (`auto`) must not reach Parakeet: it has no
    language detection, and measured on the real v3 model, unhinted decoding
    garbles non-English audio (German came out as "Alas hat an ende, no divorce
    tatzwai", where Whisper/WhisperX's LID got it right). Pick Parakeet
    explicitly for English media left on `auto`. Same reason translate is out:
    Parakeet has no translation head.

    The dialog mirrors this rule in aisubs.lua's engine_for() so its engine
    preview is truthful before the run starts; core/parakeet_models.py is the
    single source for which languages are installed.
    """
    from core.parakeet_models import normalize_language, supported_languages

    if task == "translate":
        return "auto"
    installed = supported_languages()
    if not installed:
        return "auto"
    wanted = normalize_language(language)
    if wanted is not None and wanted in installed:
        return "parakeet"
    return "auto"


def main():
    _t0 = time.time()
    # --debug may appear anywhere; capture BEFORE stripping, then remove it
    # so positional parsing is unaffected.
    debug = _debug_enabled()
    if "--debug" in sys.argv:
        sys.argv.remove("--debug")
    if debug:
        # Propagate to the WhisperX subprocess (backend dumps runner output)
        os.environ["VSCL_AISUBS_DEBUG"] = "1"

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

    if debug:
        _log_debug(f"args: media={media_path!r} model={model_name!r} lang={language!r} task={task!r}")

    emitter = Emitter(mirror_file)

    # Cancel support (registered before any long work starts): the extension
    # kills this PID, we stop the ML child, and the pid file is cleaned up on
    # every exit path (atexit covers the sys.exit()/error branches below).
    _install_cancel_handler()
    pid_path = _write_pid_file(mirror_file)
    if pid_path:
        atexit.register(_remove_pid_file, pid_path)

    if not os.path.isfile(media_path):
        emitter.emit({"type": "error", "msg": f"File not found: {media_path}"})
        emitter.close()
        sys.exit(1)

    try:
        requested_backend = os.environ.get("VSCL_AISUBS_BACKEND", "").strip() or "auto"
        if requested_backend == "auto":
            pick = auto_engine_for(language, task)
            try:
                backend = resolve_backend(pick)
            except RuntimeError as exc:
                if pick == "auto":
                    raise
                # Parakeet model present but its runtime is unusable (sherpa-onnx
                # missing, venv gone) — the hardware policy still works, so a
                # broken optional engine must not fail the whole run.
                if debug:
                    _log_debug(f"auto: Parakeet unusable ({exc}); falling back to the hardware policy")
                backend = resolve_backend("auto")
        else:
            backend = resolve_backend(requested_backend)
    except RuntimeError as exc:
        emitter.emit({"type": "error", "msg": str(exc)})
        emitter.close()
        sys.exit(1)
    if debug:
        _log_debug(f"backend resolved: {backend.name()} ({time.time() - _t0:.1f}s)")

    # Resolve the dialog's pick → the model that will actually run
    resolved = resolve_model_name(model_name, backend.name(), backend, language)
    if debug and resolved != model_name:
        _log_debug(
            f"model: {model_name} -> {resolved} "
            f"(VRAM {_detect_vram_mb()} MiB, RAM {_detect_ram_gb()} GiB)"
        )
    model_name = resolved

    emitter.emit({
        "type": "status",
        "msg": f"Backend: {backend.name()} — {model_name} ({language or 'auto'}, {task})",
    })

    # A SIGKILLed run — or VLC crashing — gets no chance to clean up, so sweep what
    # is left before starting. Age-gated: another VLC window's decode is minutes old
    # at most, never hours.
    try:
        swept = sweep_stale_temp()
        if debug and swept:
            _log_debug(f"swept {swept} stale temp wav(s)")
    except Exception as exc:
        if debug:
            _log_debug(f"stale-temp sweep failed: {exc}")

    # Which audio track will be transcribed. Worth saying out loud on multi-track
    # releases: ffmpeg's own default is the first stream, and on a real DUAL/VFF
    # file that is the dub — an English-only model handed French audio answers with
    # confident nonsense rather than an error. The runners choose the same way
    # (core/audio.py), so this line cannot disagree with what they decode.
    try:
        streams = list_audio_streams(media_path)
        if len(streams) > 1:
            _, why = choose_audio_stream(streams, language)
            emitter.emit({"type": "status", "msg": f"Audio track: {why}"})
    except Exception as exc:              # a diagnostic must never break a run
        if debug:
            _log_debug(f"audio-track report failed: {exc}")

    # Transcribe
    emitter.emit({"type": "status", "msg": "Transcribing..."})
    segments = []
    _t1 = time.time()
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
        # Full detail goes to stderr (VLC logs it) and the debug log; the UI gets
        # one actionable line — it has only a status label to render into.
        detail = traceback.format_exc()
        sys.stderr.write(detail + "\n")
        _log_debug("transcription failed:\n" + detail)
        emitter.emit({
            "type": "error",
            "msg": f"Transcription failed: {friendly_error(exc)}",
        })
        emitter.close()
        sys.exit(1)
    if debug:
        _log_debug(f"transcription done: {len(segments)} segments in {time.time() - _t1:.1f}s")

    # Write SRT (skip if no segments — avoids empty .srt files)
    # Drop known hallucination segments (research §2.2) before writing.
    raw_empty = not segments
    segments = filter_segments(segments)
    if not segments:
        msg = ("No speech detected — skipping SRT." if raw_empty
               else "All segments filtered by the hallucination blocklist — skipping SRT.")
        emitter.emit({"type": "status", "msg": msg})
        emitter.emit({"type": "done", "segments": 0, "srt_path": None})
        emitter.close()
        sys.exit(0)

    try:
        srt_path = write_srt(segments, media_path, srt_requested)
    except OSError as exc:
        # Media dir may be read-only (mounted disc, network share) — fall
        # back to a writable temp path instead of aborting after a
        # successful run; the status line tells the user where it went.
        import tempfile
        fallback = os.path.join(
            tempfile.gettempdir(), f"aisubs_{int(time.time())}_{os.getpid()}.srt"
        )
        try:
            srt_path = write_srt(segments, media_path, fallback)
        except OSError as exc2:
            emitter.emit({"type": "error", "msg": f"Could not write SRT: {exc2}"})
            emitter.close()
            sys.exit(1)
        emitter.emit({
            "type": "status",
            "msg": f"Could not write SRT next to media ({exc}); wrote {srt_path} instead.",
        })

    emitter.emit({"type": "done", "segments": len(segments), "srt_path": srt_path})
    if debug:
        _log_debug(f"done: {len(segments)} segments -> {srt_path} (total {time.time() - _t0:.1f}s)")
    emitter.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            print(
                '{"type": "error", "msg": "%s"}' % str(exc).replace('"', '\\"'),
                flush=True,
            )
        except OSError:
            pass
        sys.exit(1)
