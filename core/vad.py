"""Speech spans from Silero VAD (via sherpa-onnx), for narrowing and skipping only.

Measured on this project's own material before being adopted
(``.research/2026-09-16-github-survey.md``): on a film's dialogue-dense minute the VAD
reports 53.7 s of speech out of 60 and **no speech across a known 5.6 s music gap**, and
on 60 s of music/credits it reports 0 s — agreeing with whisper.cpp's ``(eerie music)``
and Parakeet's own 0 tokens. Plain RMS energy cannot do this at all: that gap's median
frame level is -8.5 dB against a -1.0 dB peak, so a film mix has no quiet silence to
threshold, and every RMS threshold either spared the gap or marked 40 % of the dialogue
silent as well.

Used for exactly two things, both conservative:

* **skipping a chunk that contains no speech** — nothing is transcribed there, so nothing
  can be lost, and a film's music/silence chunks stop costing a decode each;
* **narrowing word boundaries against the non-speech it finds** — words move, words never
  disappear.

It is deliberately NOT used as a gate on transcribed audio: whisper.cpp's ``--vad`` was
measured dropping real dialogue that way, and a missing line is invisible.
"""

import os

# Fed in 100 ms pieces. Handing the whole waveform to one call silently processed only the
# tail (measured: 0.3 s of speech found in a minute of dialogue, with circular-buffer
# overflow warnings) — the detector expects to be driven a small block at a time.
FEED_CHUNK_SECONDS = 0.1
# 1.0 s of buffer is ample when driven in 100 ms blocks; the default is far larger.
BUFFER_SECONDS = 4

SAMPLE_RATE = 16000

# Speech probability at which Silero calls a frame speech. The library default is 0.5, and
# 0.5 was measured to be WRONG for this job: on the project's own film it scored the
# opening's second chunk — which holds the shouts the professional subtitle track
# transcribes at 47-53 s — as 0.6 s of speech in 89 s, and the gate would have skipped that
# chunk and lost the dialogue. The sweep, per 30 s chunk:
#
#   threshold   opening (47-73 s has dialogue)   credits (music, no speech)   dense dialogue
#   0.50        0.6 s, 2/3 chunks skip           0.0 s, 2/2 skip              53.7 s, 0/2 skip
#   0.40        1.5 s, 1/3 skip                  0.0 s, 2/2 skip              54.0 s, 0/2 skip
#   0.35        2.2 s, 1/3 skip                  0.7 s, 1/2 skip              54.3 s, 0/2 skip
#   0.25        9.2 s, 1/3 skip                  2.0 s, 1/2 skip              55.1 s, 0/2 skip
#
# 0.4 is the strongest threshold that still skips every music chunk while decoding every
# chunk holding dialogue: 1/3 == the genuinely silent opening chunk, which the reference
# track agrees has nothing before 47 s.
VAD_THRESHOLD = 0.4

# Values of VSCL_AISUBS_VAD_MODEL that switch the VAD off entirely, for A/B testing and for
# anyone who would rather not have a detector decide which chunks get decoded.
_DISABLED = frozenset({"none", "off", "0", "no", "disabled", "false"})


def resolve_vad_model() -> str | None:
    """Path to silero_vad.onnx, or None when it is not installed or switched off.

    Missing model is not an error: every caller treats None as "continue without VAD",
    which keeps the plugin working on a machine that never downloaded the 630 KB model.
    ``VSCL_AISUBS_VAD_MODEL=none`` (or off/0/no) disables it even when installed — a
    non-existent path is NOT one of those values: it falls back to the standard locations,
    which is why setting one to /nonexistent does not disable anything.
    """
    env = os.environ.get("VSCL_AISUBS_VAD_MODEL", "").strip()
    if env.lower() in _DISABLED:
        return None
    candidates = [
        env,
        os.path.expanduser("~/.local/share/sherpa-onnx/models/silero_vad.onnx"),
        os.path.expanduser("~/.local/share/vlc-ai-subs/models/silero_vad.onnx"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def speech_spans(samples, model_path: str, sample_rate: int = SAMPLE_RATE,
                 threshold: float = VAD_THRESHOLD) -> list:
    """[(start, end)] seconds the VAD scores as speech.

    Raises on a broken model — callers decide whether to continue without it, and the
    runner reports it rather than silently losing the narrowing.
    """
    import numpy as np  # noqa: PLC0415 — lazy: the test environment has no numpy
    import sherpa_onnx  # noqa: PLC0415 — heavy, and only needed when the model exists

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = model_path
    config.silero_vad.threshold = threshold
    config.silero_vad.min_silence_duration = 0.25
    config.silero_vad.min_speech_duration = 0.25
    config.sample_rate = sample_rate

    detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=BUFFER_SECONDS)
    audio = np.asarray(samples, dtype=np.float32)
    step = max(1, int(sample_rate * FEED_CHUNK_SECONDS))
    spans = []
    for begin in range(0, audio.size, step):
        detector.accept_waveform(audio[begin:begin + step])
        while not detector.empty():
            segment = detector.front
            spans.append((segment.start / sample_rate,
                          (segment.start + len(segment.samples)) / sample_rate))
            detector.pop()
    detector.flush()
    while not detector.empty():
        segment = detector.front
        spans.append((segment.start / sample_rate,
                      (segment.start + len(segment.samples)) / sample_rate))
        detector.pop()
    return spans


def holds_speech(speech: list, start: float, stop: float) -> bool:
    """True when any speech span overlaps [start, stop).

    The whole of the gating decision: a chunk with no speech in it needs no decode, and
    nothing transcribed in it could be lost, because there is nothing there.
    """
    return any(begin < stop and finish > start for begin, finish in speech)


def speech_extent(words: list) -> float:
    """Total seconds of audio the words occupy.

    For the reading-speed figure: words-per-minute over *speech* is comparable with the
    120-160 wpm people actually speak, whereas the same count over a cue's span is not —
    a short cue contains almost no silence, so its rate reads far too high.

    Note the asymmetry this measurement settled (2026-09-16): narrowing word boundaries
    against these spans changes *nothing* in the output, because the transducer already
    places words inside speech — its timestamps are compressed, not misplaced. That is why
    the VAD is used to skip chunks and not to re-time words.
    """
    if not words:
        return 0.0
    return sum(max(0.0, end - start) for _, start, end in words)
