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
Translate task: NLLB-200 cascade by default (VSCL_AISUBS_NLLB=0 reverts to
Whisper's built-in translate; VSCL_AISUBS_NLLB_MODEL overrides the model dir;
VSCL_AISUBS_NLLB_FAMILY=nllb|m2m100 picks the cascade family — m2m100 is the
MIT-licensed commercial-use alternative).
"""
import json
import os
import sys

from core.audio import (
    choose_audio_stream,
    cleanup_temp,
    decode_to_wav16k,
    list_audio_streams,
    read_wav_pcm16,
)
from core.cues import apply_quality
from core.procs import install_termination_handler
from core.srt import write_srt


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


# whisperx.align() processes ONE segment per pass (its real signature has no
# batch knob — verified in the installed venv on 2026-09-15, after passing
# batch_size=1 raised TypeError and silently cost the alignment). No VRAM
# batching to tune here: 54 segments aligned in ~2 s on this box's 4 GB card.
# The default batch the research note worried about belongs to
# model.transcribe(), which runs fine at its default here.
#
# Its returned segments already carry word-tightened start/end (measured: cue
# times identical to a run with an extra refinement pass bolted on, 54/54 cues),
# so consume them directly — no second refinement step.


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


def emit(data: dict):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def _load_waveform(wav_path: str):
    """A 16 kHz mono PCM wav as float32 in [-1, 1] — what WhisperX expects.

    WhisperX's own load_audio() returns exactly this shape (and resamples the same
    way); doing it here instead means the model receives the track we chose rather
    than decoding the media a second time with ffmpeg's default.
    """
    import numpy as np

    return (np.frombuffer(read_wav_pcm16(wav_path), dtype=np.int16)
            .astype("float32") / 32768.0)


def main():
    # A cancelled run must stop this process's own children and remove the temp wav:
    # the CLI signals us, and its registry cannot see our children.
    install_termination_handler(cleanup_temp)

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
    try:
        import nllb_translate
        nllb_ok = True
    except ImportError:
        nllb_ok = False

    # NLLB cascade for translate: transcribe in the source language, then
    # NLLB-200 translates segments to English (research: ~44.7 BLEU CA→EN vs
    # Whisper's built-in translate). VSCL_AISUBS_NLLB=0 reverts to Whisper.
    # Falls back to Whisper translate — never silently source-language output.
    use_nllb = nllb_ok and nllb_translate.should_cascade(
        task, os.environ.get("VSCL_AISUBS_NLLB")
    )

    # Resolve device: VSCL_AISUBS_DEVICE=cpu|cuda forces it; else auto-detect.
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    device = resolve_device(cuda_available)
    compute = resolve_compute(device)

    # Load the NLLB/M2M translator up front so a missing/broken model switches
    # to Whisper translate before any audio is transcribed. Family: nllb
    # (CC-BY-NC-4.0, default) or m2m100 (MIT, commercial-use alternative).
    family = os.environ.get("VSCL_AISUBS_NLLB_FAMILY", "nllb").strip().lower()
    if family not in ("nllb", "m2m100"):
        emit({"type": "status", "msg": f"Unknown VSCL_AISUBS_NLLB_FAMILY '{family}' — using nllb (CC-BY-NC-4.0)"})
        family = "nllb"
    translator = None
    if use_nllb:
        translator = nllb_translate.try_load_translator(
            os.environ.get("VSCL_AISUBS_NLLB_MODEL") or nllb_translate.MODEL_DIR_DEFAULT,
            device="cuda" if device == "cuda" else "cpu",
            compute_type="int8_float16" if device == "cuda" else "int8",
            family=family,
        )
        if translator:
            emit({"type": "status", "msg": f"Translate: {family} cascade (transcribe → {family} → English)"})
        else:
            emit({"type": "status", "msg": "NLLB/M2M model not installed — using Whisper translate. Run ./install-nllb-model.sh (or install-m2m-model.sh)"})

    emit({"type": "status", "msg": f"WhisperX: loading {model_name} on {device} ({compute})..."})

    # 1. Transcribe — VAD-first (whisperx default gate) + hardened decode.
    model = whisperx.load_model(
        model_name, device, compute_type=compute,
        asr_options=hardened_asr_options(),
        vad_options={"vad_onset": 0.500, "vad_offset": 0.363},
        download_root=model_cache_dir(),
    )
    # Decode once, here, on the track we deliberately chose. Left to itself
    # WhisperX decodes the file itself — twice, with ffmpeg's default track, which
    # on a real dual-audio release is the dub (see core/audio.py): asking for
    # English subtitles transcribed French audio. One decode also costs less than
    # the two WhisperX would do.
    streams = list_audio_streams(media_path)
    stream_index, why = choose_audio_stream(streams, language)
    if len(streams) > 1:
        emit({"type": "status", "msg": f"Audio track: {why}"})
    emit({"type": "status", "msg": "Decoding audio..."})
    try:
        wav_path = decode_to_wav16k(media_path, stream_index=stream_index)
    except RuntimeError as exc:
        emit({"type": "error", "msg": str(exc)})
        sys.exit(1)
    waveform = _load_waveform(wav_path)
    try:
        os.unlink(wav_path)
    except OSError:
        pass

    emit({"type": "status", "msg": "Transcribing..."})
    result = model.transcribe(
        waveform,
        language=language,
        task="transcribe" if translator else task,
    )
    emit({
        "type": "status",
        "msg": f"WhisperX: transcription done (+{time.time() - _t0:.0f}s)",
    })

    # 2. Hallucination blocklist (research §2.2) — before the cascade and
    # alignment so garbage segments aren't translated or GPU-aligned.
    from core.blocklist import filter_segments
    result["segments"] = filter_segments(result.get("segments", []))

    # 2b. Cascade translation (translate task only, translator loaded)
    if translator and task == "translate":
        tgt_lang = nllb_translate.TARGET_M2M if family == "m2m100" else nllb_translate.TARGET
        src_code = (result.get("language") or language or "en").lower()
        src_flores = nllb_translate.lang_code(src_code, family)
        if not src_flores:
            # Unmapped language — re-transcribe with Whisper's translate so
            # output stays English (never silently source-language).
            emit({"type": "status", "msg": f"{family}: no mapping for '{src_code}' — re-running with Whisper translate"})
            result = model.transcribe(waveform, language=language, task="translate")
        elif src_flores != tgt_lang:
            emit({"type": "status", "msg": f"{family}: translating {src_flores} → {tgt_lang} (+{time.time() - _t0:.0f}s)"})
            before = [(s.get("text") or "").strip() for s in result.get("segments", [])]
            result["segments"] = nllb_translate.translate_segments(
                result.get("segments", []), src_flores, translator
            )
            after = [(s.get("text") or "").strip() for s in result["segments"]]
            if not nllb_translate.translation_viable(before, after):
                # Nothing came back translatable — fall back to Whisper.
                emit({"type": "status", "msg": f"{family} translation failed — re-running with Whisper translate"})
                result = model.transcribe(waveform, language=language, task="translate")
        # src == tgt: source is already English — pass through

    # Re-apply the blocklist: the fallback branches produced fresh unfiltered
    # segments, and NLLB-translated output may itself match an English phrase.
    result["segments"] = filter_segments(result.get("segments", []))

    # 3. Align (word-level timestamps) and fold them into the cue times.
    #
    # The align model maps SOUNDS to text, so it only means anything while the
    # segments still hold the transcript. After a translation they hold English
    # while the audio is still in the source language — the pass would cost a GPU
    # run plus a model download and yield nonsense timings. Skip it, and refine
    # otherwise.
    src_lang = (result.get("language") or language or "en").lower()
    have_transcript = task == "transcribe" or src_lang.startswith("en")
    if device == "cuda" and have_transcript and result.get("segments"):
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=src_lang, device=device,
            )
            aligned = whisperx.align(
                result["segments"], align_model, metadata,
                waveform, device, return_char_alignments=False,
            )
            result["segments"] = aligned.get("segments", [])
            emit({
                "type": "status",
                "msg": f"WhisperX: word alignment done, cue times refined (+{time.time() - _t0:.1f}s)",
            })
        except Exception as exc:
            # Say what actually went wrong: this used to blame the language
            # model even when the real cause was a CUDA OOM, and the alignment's
            # word timings are what tightens the cues below.
            reason = "GPU out of memory" if "out of memory" in str(exc).lower() else f"{type(exc).__name__}"
            emit({
                "type": "status",
                "msg": f"Word alignment skipped ({reason}) — cue times stay segment-level",
            })

    # 4. Wrap the cue text (broadcast-style line breaks) and clean the timing
    # gaps before emitting — the SRT and the OSD feed share this text, using the
    # reading-speed / line-length ceilings of the language the subtitles are
    # actually IN: the detected source language on a transcribe run, English on
    # every translate path of this runner (Whisper's own translate, or the
    # NLLB/M2M cascade — both emit English).
    text_language = "en" if (translator and task == "translate") else (result.get("language") or language)
    segments = apply_quality(result.get("segments", []), language=text_language)
    out_segments = []
    count = 0

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        count += 1
        item = {
            "start": round(seg.get("start", 0), 3),
            "end": round(seg.get("end", 0), 3),
            "text": text,
        }
        out_segments.append(item)
        emit({"type": "sub", "i": count, **item})

    # 5. Write SRT — ONLY when the caller explicitly requested a path. The
    # plugin's caller (aisubs_whisper.py) owns SRT output; deriving
    # <media>.srt here would drop a side-effect file next to the media
    # (read-only media dirs, and realtime-OSD runs that deliberately write to
    # a temp path instead). Empty output → write_srt returns None.
    srt_path = None
    if srt_requested:
        try:
            srt_path = write_srt(out_segments, media_path, srt_requested)
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