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

from core.audio import SAMPLE_RATE, decode_to_wav16k
from core.cues import apply_quality
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


# 20 min — the 0.6B TDT model is designed for up to 24-min single-pass
# segments; long media is decoded in chunks instead of one giant stream.
CHUNK_SECONDS = 20 * 60


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

    emit({"type": "status", "msg": f"Parakeet: decoding audio (+{time.time()-t0:.0f}s)"})
    wav_path = decode_to_wav16k(media_path)
    samples = load_float32_16k(wav_path)
    os.unlink(wav_path)

    # Long media: decode in ≤20-min chunks (the 0.6B TDT model is designed
    # for up to 24-min single-pass segments) instead of one giant stream.
    ranges = chunk_plan(len(samples), CHUNK_SECONDS * SAMPLE_RATE)
    words = []
    n_chunks = len(ranges)
    for ci, (start, stop) in enumerate(ranges, 1):
        if n_chunks > 1:
            emit({"type": "status", "msg": f"Parakeet: decoding chunk {ci}/{n_chunks} (+{time.time()-t0:.0f}s)"})
        stream = rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples[start:stop])
        rec.decode_stream(stream)
        result = stream.result
        tokens = result.tokens or []
        times = result.timestamps or []
        if tokens:
            words.extend(shift_words(tokens_to_words(tokens, times), start / SAMPLE_RATE))

    if not words:
        emit({"type": "status", "msg": "No speech detected."})
        emit({"type": "done", "segments": 0, "srt_path": None})
        return

    segments = words_to_segments(words)

    # Drop known hallucination segments (research §2.2) before emitting, then
    # wrap the cue text (broadcast-style line breaks) + clean timing gaps.
    from core.blocklist import filter_segments
    segments = filter_segments(segments)
    segments = apply_quality(segments)

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