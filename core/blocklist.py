"""Whisper hallucination blocklist — research-backed post-filter.

Reference: .research/final_report.md §2.2 — 40.3% of non-speech audio files
hallucinate (9.1% loop), and 67% of hallucinations come from ~1,270 recurring
phrases; a known-phrase blocklist "is cheap and strips most music/silence
garbage". This is a deliberately small, high-precision set — only distinctive
recurring phrases, never short generic ones that occur in real dialogue
("thank you", "you", …). Disable with VSCL_AISUBS_BLOCKLIST=0.
"""

import os
import re

# Lowercase, whitespace-normalized distinctive hallucination phrases.
HALLUCINATION_PHRASES = frozenset({
    "subtitles by the amara.org community",
    "subtitles by the amara.org community - youtube",
    "subtitles by the amara.org community and",
    "this video is sponsored by",
    "if you enjoyed this video, please subscribe",
    "please subscribe to my channel and like this video",
    "like, share and subscribe to my channel",
    "thank you for watching, please subscribe",
    "thanks for watching, please subscribe",
    "[music playing]",
    "[applause]",
    "[laughing]",
    "[background music]",
    "[no music]",
    "foreign music",
    "music playing",
    "♪ ♪ ♪",
    "♪♪♪",
})

_WS = re.compile(r"\s+")
_ANNOTATION = re.compile(r"[\[\](){}<>]")
_TRAILING_PUNCT = re.compile(r"[.!?,;:]+$")


def _normalize(text: str) -> str:
    """Lowercase, annotation- and punctuation-insensitive phrase key.

    The set spells sounds the way WhisperX does — "[music playing]" — while
    whisper.cpp emits the same hallucination bare ("music playing", measured on
    a real episode), and either engine may add a full stop. Stripping brackets
    and trailing punctuation lets one entry cover every engine's spelling.
    """
    text = _ANNOTATION.sub("", text or "")
    return _TRAILING_PUNCT.sub("", _WS.sub(" ", text).strip().lower())


# A phrase repeated this many times in a row is a decoder loop, not speech.
# Both sides measured on the same real episode:
#   loop   — whisper.cpp, 14 consecutive "I'm sorry." cues over 14 s
#   speech — Parakeet (no loops) heard "Go, go, go, go, go!", "Come on, come on,
#            come on, come on." and "Hey, come on, …" as *real dialogue*
# A threshold of 4 ate those three lines, so 7 sits in the measured gap with
# margin on both sides. Only collapse what is certainly a loop.
MAX_REPEATS = 7
MAX_LOOP_PHRASE = 5


def _key(token: str) -> str:
    """Comparison key for a token: punctuation-insensitive ("Sorry," == "sorry.")."""
    return re.sub(r"[^a-z0-9']", "", token.lower())


def deloop_text(text: str, max_repeats: int = MAX_REPEATS) -> str:
    """Collapse a phrase the decoder repeated far more often than anyone says it.

    Whisper's characteristic failure on music and silence is an immediate repeat
    loop; the blocklist cannot catch it, because the phrase is usually embedded
    in a longer segment rather than standing alone. Only runs of at least
    *max_repeats* collapse — the first copy is kept, with its original
    punctuation.

    Phrases are tried SHORTEST first on purpose. Trying them longest first looks
    tidier but is wrong: "I'm sorry," x14 also matches as seven repeats of the
    two-copy phrase, so a 4-word pass collapses it to two copies and the single
    copy is never reached (that bug shipped into this function's first draft).

    Repeats are collapsed within a line, never across one: a decoder loop does not
    span a deliberate line break, and flattening the breaks here destroyed the
    engine's wrapping (measured: 397 over-long lines on a real film).
    """
    return "\n".join(
        _deloop_repeats(line, max_repeats) for line in (text or "").split("\n")
    )


def _deloop_repeats(text: str, max_repeats: int) -> str:
    """Collapse a repeated phrase within one line — see :func:`deloop_text`."""
    tokens = text.split()
    keys = [_key(t) for t in tokens]
    for n in range(1, MAX_LOOP_PHRASE + 1):
        keep, i = [], 0
        while i < len(tokens):
            reps = 1
            while (i + (reps + 1) * n <= len(tokens)
                   and keys[i:i + n] == keys[i + reps * n:i + (reps + 1) * n]
                   and any(keys[i:i + n])):
                reps += 1
            if reps >= max_repeats:
                keep.extend(range(i, i + n))   # one copy survives
                i += reps * n
            else:
                keep.append(i)
                i += 1
        tokens = [tokens[k] for k in keep]
        keys = [keys[k] for k in keep]
    return " ".join(tokens)


def deloop_segments(segments: list, max_repeats: int = MAX_REPEATS) -> list:
    """Drop the copies when consecutive segments say exactly the same thing.

    Whisper's other loop shape: one short line emitted over and over as separate
    segments — measured, 14 consecutive "I'm sorry." cues spanning 14 s in a
    whisper.cpp run, where the blocklist and the in-text collapse both miss it.
    The first cue keeps its own timing, because the loop has no real duration.
    Runs shorter than *max_repeats* are left alone: "Okay? Okay?" is speech.
    """
    out: list[dict] = []
    i = 0
    while i < len(segments):
        key = _key(segments[i].get("text") or "")
        j = i + 1
        while j < len(segments) and key and _key(segments[j].get("text") or "") == key:
            j += 1
        if j - i >= max_repeats:
            out.append(segments[i])
        else:
            out.extend(segments[i:j])
        i = j
    return out


def is_blocklisted(text: str) -> bool:
    """True when the segment is a known hallucination (high-precision set)."""
    if os.environ.get("VSCL_AISUBS_BLOCKLIST", "1").strip().lower() in ("0", "false", "off"):
        return False
    return _normalize(text) in HALLUCINATION_PHRASES


def filter_segments(segments: list) -> list:
    """Drop hallucinated segments, collapse repeat loops in the survivors."""
    out = []
    for seg in segments:
        if is_blocklisted(seg.get("text") or ""):
            continue
        text = deloop_text(seg.get("text") or "")
        if not text.strip():
            continue
        out.append({**seg, "text": text})
    return deloop_segments(out)
