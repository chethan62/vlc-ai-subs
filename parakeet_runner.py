#!/usr/bin/env python3
"""
Parakeet runner — invoked by the plugin as a subprocess inside the Python
3.12 venv (sherpa-onnx). NVIDIA Parakeet-TDT-0.6B, int8 ONNX.

Two variants, same architecture (native word timestamps, transducer blanking =
no hallucination loops on music/silence), same four file names:
  v2  English only, WER 6.05% self-reported (whisper-large-v3 7.44% on a
      comparable English eval), ~0.7 GB.
  v3  25 European languages, ~0.64 GB, no language flag needed — k2-fsa's
      ONNX conversion handles the multilingual prompt internally.
Both CC-BY-4.0. English prefers v2 when it is installed (English-specialised);
anything else uses v3. See select_variant().

Contract (stdout, JSONL) — same as whisperx_runner:
  {"type": "status", "msg": "..."}
  {"type": "sub", "i": N, "start": S, "end": E, "text": "..."}
  {"type": "done", "segments": N, "srt_path": "..."}
  {"type": "error", "msg": "..."}

Args:  <media> <model> <language> <task> [mirror_file] [srt_path]
The <model> arg is ignored (the variant is picked from what is installed or
from VSCL_AISUBS_PARAKEET_VERSION / VSCL_AISUBS_PARAKEET_MODEL). translate, or
a language no installed variant covers → actionable error, callers fall back.
The SRT file is written ONLY when [srt_path] is given — the plugin's caller
(aisubs_whisper.py) owns SRT output, so the runner never creates side-effect
files next to the media (realtime-OSD mode, read-only media dirs).
"""
from __future__ import annotations

import json
import os
import sys
import time
import wave
from typing import TYPE_CHECKING

from core.audio import SAMPLE_RATE, choose_audio_stream, cleanup_temp, decode_to_wav16k, list_audio_streams
from core.cues import apply_quality
from core.procs import install_termination_handler
from core.srt import write_srt

if TYPE_CHECKING:
    import numpy as np

# Model variants, language sets and selection live in a leaf module shared with
# backends/parakeet.py (status-line labels) — see core/parakeet_models.py.
from core.parakeet_models import (  # noqa: E402
    INSTALL_HINT,
    KNOWN_LANGUAGES,
    MODEL_FILES,
    V3_LANGUAGES,
    normalize_language,
    select_variant,
    variant_dir,
)


def emit(data: dict):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def load_float32_16k(wav_path: str) -> "np.ndarray":
    """Read a 16 kHz mono wav into float32 samples in [-1, 1].

    numpy is imported lazily so the module stays importable in dev venvs
    that only install pytest (tests cover the pure logic, not the runtime).
    """
    import numpy as np

    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, "expected 16k mono"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def tokens_to_words(tokens, times) -> list:
    """Merge sherpa-onnx BPE tokens into words with (text, start, end)."""
    words = []
    cur, cur_start, last_t = "", 0.0, 0.0
    for tok, t in zip(tokens, times):
        if not cur:
            cur, cur_start = tok.strip(), t
        elif tok.startswith(" "):
            words.append((cur.strip(), cur_start, last_t))
            cur, cur_start = tok.strip(), t
        else:
            cur += tok
        last_t = max(last_t, t)
    if cur.strip():
        words.append((cur.strip(), cur_start, last_t))
    return words


def words_to_segments(words) -> list:
    """Group words into subtitle cues (sentence punctuation / length caps)."""
    segments, seg = [], []

    def flush():
        if not seg:
            return
        text = " ".join(w[0] for w in seg).strip()
        if text:
            segments.append({"start": seg[0][1], "end": seg[-1][2], "text": text})
        seg.clear()

    for w in words:
        seg.append(w)
        span = w[2] - seg[0][1]
        if w[0][-1:] in ".!?" or len(seg) >= 12 or span > 9.0:
            flush()
    flush()
    return segments


# Chunk length for long media. This is NOT a model semantic limit — measured
# here (2026-09-15, 15 GiB box), the int8 ONNX conversion has TWO hard limits:
#
#   audio    peak RSS   result
#   2 min    1961 MB    ok (17 segments)
#   5 min    3140 MB    ok (49 segments)
#   10 min   5070 MB    CRASH — onnxruntime: "Add node '/layers.0/self_attn/
#                       Add_2' … broadcast an axis by a dimension other than 1.
#                       2500 by 7500" (the encoder refuses this length)
#   20 min   ≈9 GB      → what OOM-killed the first chunk of a 47.5-min episode
#                       under the old cap (7.2 GB RSS + 5.1 GB swap)
#
# So memory grows ~1.17 GB fixed (model + onnxruntime) plus ~393 MB per minute of
# chunk audio, and length itself becomes fatal somewhere between 5 and 10 min.
# 30 s sits far inside both limits (≈1.4-1.5 GB peak — fine on a 4 GB laptop) and
# costs nothing in speed: the model stays loaded across chunks.
# Override with VSCL_AISUBS_PARAKEET_CHUNK (seconds).
CHUNK_SECONDS = 30

# Read each chunk with a little audio past its end so a word sitting on a seam
# is decoded with context on both sides; keep_nominal_window() then drops the
# duplicate from the following chunk.
CHUNK_OVERLAP_SECONDS = 2.0


def resolve_chunk_seconds() -> int:
    """Chunk length from VSCL_AISUBS_PARAKEET_CHUNK (5..600 s) or the default.

    Out-of-range or junk values keep the default rather than silently producing
    a chunk size that cannot work (a 0 would be an empty range, a huge one is
    what caused the OOM).
    """
    raw = os.environ.get("VSCL_AISUBS_PARAKEET_CHUNK", "").strip()
    if not raw:
        return CHUNK_SECONDS
    try:
        value = int(float(raw))
    except ValueError:
        return CHUNK_SECONDS
    return value if 5 <= value <= 600 else CHUNK_SECONDS


def keep_nominal_window(words: list, start_s: float, stop_s: float) -> list:
    """Words (text, start, end) whose *start* falls in this chunk's nominal window.

    Chunks overlap slightly (see CHUNK_OVERLAP_SECONDS), so without this rule a
    word in the overlap would be emitted by both neighbours, and a word split by
    the boundary would come out garbled at the seam. Keeping only words that
    start inside the window gives each word exactly one owner: the chunk that
    contains its opening.
    """
    return [w for w in words if start_s <= w[1] < stop_s]


def chunk_plan(n_samples: int, chunk_samples: int) -> list:
    """[(start, stop)] sample ranges covering n_samples in ≤chunk_samples pieces."""
    return [
        (start, min(start + chunk_samples, n_samples))
        for start in range(0, n_samples, chunk_samples)
    ]


def shift_words(words: list, dt: float) -> list:
    """Offset (text, start, end) words by dt seconds (chunk time alignment)."""
    return [(w[0], w[1] + dt, w[2] + dt) for w in words]


def main():
    # A cancelled run must stop this process's own children and remove the temp wav:
    # the CLI signals us (its registry cannot see our ffmpeg) and a SIGTERM leaves
    # no chance to clean up otherwise.
    install_termination_handler(cleanup_temp)

    if len(sys.argv) < 5:
        emit({"type": "error", "msg": "Usage: runner <media> <model> <lang> <task> [mirror] [srt]"})
        sys.exit(1)

    media_path = sys.argv[1]
    language = normalize_language(sys.argv[3])
    task = sys.argv[4]
    srt_requested = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6].strip() else None

    if task == "translate":
        emit({"type": "error", "msg": "Parakeet cannot translate (no translation head) — use WhisperX for translate."})
        sys.exit(1)

    variant = select_variant(language)
    if variant is None:
        forced = os.environ.get("VSCL_AISUBS_PARAKEET_VERSION", "").strip()
        if language and language not in KNOWN_LANGUAGES:
            emit({"type": "error", "msg": (
                f"Parakeet does not support '{language}'. The multilingual v3 model "
                f"({INSTALL_HINT}) adds: {', '.join(sorted(V3_LANGUAGES))} — otherwise use WhisperX."
            )})
        elif forced:
            emit({"type": "error", "msg": (
                f"VSCL_AISUBS_PARAKEET_VERSION={forced} but that model is not installed — run {INSTALL_HINT}"
            )})
        else:
            needed = "" if language in (None, "en") else f" (v3 is required for '{language}')"
            emit({"type": "error", "msg": f"Parakeet model not installed — run {INSTALL_HINT}{needed}"})
        sys.exit(1)

    if not os.path.isfile(media_path):
        emit({"type": "error", "msg": f"File not found: {media_path}"})
        sys.exit(1)

    enc, dec, joi, tok = (
        os.path.join(variant_dir(variant), name) for name in MODEL_FILES
    )

    import sherpa_onnx  # noqa: E402 — import lazily; heavy package

    t0 = time.time()
    num_threads = min(8, os.cpu_count() or 2)
    langs = "" if len(variant.languages) == 1 else f", {len(variant.languages)} languages"
    emit({"type": "status", "msg": f"Parakeet: loading {variant.label}{langs} (CPU, int8, {num_threads} threads)..."})
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=enc, decoder=dec, joiner=joi, tokens=tok,
        num_threads=num_threads, provider="cpu",
        model_type="nemo_transducer", modeling_unit="cjkchar",
    )

    streams = list_audio_streams(media_path)
    stream_index, why = choose_audio_stream(streams, language)
    if len(streams) > 1:
        emit({"type": "status", "msg": f"Audio track: {why}"})
    emit({"type": "status", "msg": f"Parakeet: decoding audio (+{time.time()-t0:.0f}s)"})
    wav_path = decode_to_wav16k(media_path, stream_index=stream_index)
    samples = load_float32_16k(wav_path)
    os.unlink(wav_path)

    # Long media: decode in memory-bounded chunks (see CHUNK_SECONDS) instead of
    # one giant stream — a full film used to take the machine down.
    ranges = chunk_plan(len(samples), resolve_chunk_seconds() * SAMPLE_RATE)
    overlap = int(CHUNK_OVERLAP_SECONDS * SAMPLE_RATE)
    words = []
    n_chunks = len(ranges)
    for ci, (start, stop) in enumerate(ranges, 1):
        if n_chunks > 1:
            emit({"type": "status", "msg": f"Parakeet: decoding chunk {ci}/{n_chunks} (+{time.time()-t0:.0f}s)"})
        stream = rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples[start:min(stop + overlap, len(samples))])
        rec.decode_stream(stream)
        result = stream.result
        tokens = result.tokens or []
        times = result.timestamps or []
        if tokens:
            chunk_words = shift_words(tokens_to_words(tokens, times), start / SAMPLE_RATE)
            # The last chunk owns everything from its start onwards: the model's
            # final timestamps can sit a hair past the audio length (padding),
            # and there is no following chunk to hand those words to.
            stop_s = stop / SAMPLE_RATE if ci < n_chunks else float("inf")
            words.extend(keep_nominal_window(chunk_words, start / SAMPLE_RATE, stop_s))

    if not words:
        emit({"type": "status", "msg": "No speech detected."})
        emit({"type": "done", "segments": 0, "srt_path": None})
        return

    segments = words_to_segments(words)

    # Drop known hallucination segments (research §2.2) before emitting, then
    # wrap the cue text (broadcast-style line breaks) + clean timing gaps.
    from core.blocklist import filter_segments
    segments = filter_segments(segments)
    segments = apply_quality(segments, language="en" if task == "translate" else language)

    # Emit each segment for the caller (status / OSD progress).
    for i, seg in enumerate(segments, 1):
        emit({
            "type": "sub", "i": i,
            "start": round(seg["start"], 3), "end": round(seg["end"], 3),
            "text": seg["text"],
        })

    # Write SRT — ONLY when the caller explicitly requested a path. The
    # plugin's caller (aisubs_whisper.py) owns SRT output; deriving
    # <media>.srt here would drop a side-effect file next to the media
    # (read-only media dirs, and realtime-OSD runs that deliberately write to
    # a temp path instead). Empty output → write_srt returns None.
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