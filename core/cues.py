"""Cue quality: line breaking + timing cleanup, shared by both engines.

Raw ASR output is one long unwrapped line per segment, which is unreadable as
a subtitle. This pass applies broadcast-style presentation rules before the SRT
is written:

* at most ``MAX_LINES`` lines per cue, each at most ``MAX_LINE_CHARS`` wide
  (CJK scripts need far fewer characters per line — ``CJK_LINE_CHARS``)
* breaks at word boundaries for space-delimited scripts, anywhere in CJK
* cue timings clamped to at least ``MIN_DURATION``, spaced by ``MIN_GAP``
* a cue that would stay on screen longer than ``MAX_CUE_SECONDS`` is split at
  its sentence boundaries, its time shared out in proportion to each piece's
  length (see :func:`split_long_cue`)

Pure list-in/list-out, no I/O: the runners call :func:`apply_quality` and the
same behaviour is unit-testable in the dev venv.

Never loses text: a single word longer than the limit, or a cue that cannot be
balanced into ``MAX_LINES``, keeps its content (overlong) rather than being
truncated — subtitle text is the deliverable, wrapping is cosmetic.
"""

import os
import re

# Presentation defaults (Netflix/BBC-style simplified for one/two-liners).
MAX_LINE_CHARS = 42
CJK_LINE_CHARS = 20
MAX_LINES = 2
MIN_DURATION = 1.0
MIN_GAP = 0.08  # two frames at 25 fps — avoid back-to-back cue flicker
MIN_VISIBLE = 0.2
# One cue on screen longer than this is past what a viewer can hold in one
# glance (BBC/Netflix guidance is ~7 s), so it is split at sentence boundaries.
MAX_CUE_SECONDS = 7.0

_WS = re.compile(r"\s+")
# Sentence ends: ASCII + CJK terminators (CJK needs no trailing space).
_SENTENCE_BREAK = re.compile(r"(?<=[.!?\u2026])\s+|(?<=[\u3002\uff01\uff1f\uff0e])")
# CJK ideographs, kana, hangul + CJK punctuation: no spaces to break on.
_CJK = re.compile(
    "[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\uff00-\uffef\uac00-\ud7af]"
)


def is_cjk(text: str) -> bool:
    """True when the text is predominantly CJK (break anywhere, not on spaces)."""
    return bool(_CJK.search(text or ""))


def normalize(text: str | None) -> str:
    """Collapse whitespace runs and strip — keeps lyrics/spacing sane."""
    return _WS.sub(" ", text or "").strip()


def line_limit(max_chars: int | None = None) -> int:
    """Line width from the argument, else VSCL_AISUBS_MAX_LINE, else the default."""
    if max_chars is not None:
        return max(8, int(max_chars))
    raw = os.environ.get("VSCL_AISUBS_MAX_LINE", "").strip()
    if raw.isdigit():
        return max(8, int(raw))
    return MAX_LINE_CHARS


def _split_units(text: str, cjk: bool) -> list[str]:
    """Break text into break-safe units: words for spaced scripts, chars for CJK."""
    if cjk:
        return list(text)
    return text.split(" ")


def _join_units(units: list[str], cjk: bool) -> str:
    return "".join(units) if cjk else " ".join(units)


def _greedy_lines(units: list[str], cjk: bool, width: int, max_lines: int) -> list | None:
    """Fill lines greedily; None when the text needs more than *max_lines*."""
    lines: list[list[str]] = []
    current: list[str] = []
    for unit in units:
        candidate = current + [unit]
        if current and len(_join_units(candidate, cjk)) > width:
            lines.append(current)
            current = [unit]
            if len(lines) >= max_lines:
                return None  # a further line would be needed → caller balances
        else:
            current = candidate
    if current:
        lines.append(current)
    return None if len(lines) > max_lines else lines


def _best_two_way_split(units: list[str], cjk: bool) -> int:
    """Split index that minimises the longer of the two sides (readability).

    Used when the text cannot fit the width limit: a balanced pair of slightly
    overlong lines beats one full line plus a long tail.
    """
    best, best_score = 1, None
    for i in range(1, len(units)):
        head = len(_join_units(units[:i], cjk))
        tail = len(_join_units(units[i:], cjk))
        score = max(head, tail)
        if best_score is None or score < best_score:
            best, best_score = i, score
    return best


def wrap(text: str, max_chars: int | None = None, max_lines: int = MAX_LINES) -> str:
    """Wrap a cue into at most *max_lines* lines of at most *max_chars* columns.

    Greedy fill while it fits; when it does not, the text is split at the word
    boundary closest to balanced halves (a 2-line cue should not be "one very
    long line / one short word"). Text that cannot fit at all stays overlong —
    the content is never dropped.
    """
    text = normalize(text)
    if not text:
        return ""

    cjk = is_cjk(text)
    width = line_limit(max_chars)
    if cjk and max_chars is None and not os.environ.get("VSCL_AISUBS_MAX_LINE", "").strip():
        width = CJK_LINE_CHARS

    if len(text) <= width:
        return text

    units = _split_units(text, cjk)
    lines = _greedy_lines(units, cjk, width, max_lines)
    if lines is not None:
        return "\n".join(_join_units(line, cjk) for line in lines)
    if max_lines <= 1:
        return _join_units(units, cjk)

    split = _best_two_way_split(units, cjk)
    return _join_units(units[:split], cjk) + "\n" + _join_units(units[split:], cjk)


def split_long_cue(seg: dict, max_seconds: float = MAX_CUE_SECONDS) -> list[dict]:
    """Split one over-long cue at sentence boundaries; returns one or more cues.

    Without CUDA alignment to re-segment the transcript (CPU, AMD/Intel, or a
    translated run) Whisper hands back whole exchanges as single segments —
    measured on a real episode: 8 cues for 5 minutes, the longest showing 29 s
    of dialogue at once. Word timings are not needed for this: split where the
    text says a sentence ended and share the cue's span in proportion to each
    piece's length, which tracks speech duration closely enough to stay inside
    the cue's real time. A cue with nothing to split on (no terminator) is left
    exactly as it was, and a fragment too brief to read is folded back into its
    neighbour rather than flashing on and off.
    """
    duration = seg["end"] - seg["start"]
    if duration <= max_seconds:
        return [seg]

    pieces = [p.strip() for p in _SENTENCE_BREAK.split(normalize(seg.get("text", "")))]
    pieces = [p for p in pieces if p]
    if len(pieces) < 2:
        return [seg]

    # Fold in fragments that would be too brief to read on their own: "Mm-hmm."
    # at 0.2 s is a flicker, and its time cannot be extended into the next piece.
    total_chars = sum(len(p) for p in pieces) or 1
    merged: list[str] = []
    for piece in pieces:
        share = duration * len(piece) / total_chars
        if merged and share < MIN_DURATION:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    # A too-brief first piece has no predecessor to fold into — give it to the
    # piece that follows instead.
    if len(merged) > 1 and duration * len(merged[0]) / total_chars < MIN_DURATION:
        merged[1] = f"{merged[0]} {merged[1]}"
        merged.pop(0)
    pieces = merged

    weights = [len(p) for p in pieces]
    total = sum(weights) or 1
    out: list[dict] = []
    at = seg["start"]
    for i, (piece, weight) in enumerate(zip(pieces, weights)):
        end = seg["end"] if i == len(pieces) - 1 else at + duration * weight / total
        out.append({"start": at, "end": end, "text": piece})
        at = end
    return out


def apply_quality(
    segments: list[dict],
    max_chars: int | None = None,
    max_lines: int = MAX_LINES,
    min_duration: float = MIN_DURATION,
    min_gap: float = MIN_GAP,
    max_cue_seconds: float = MAX_CUE_SECONDS,
) -> list[dict]:
    """Split over-long cues, wrap cue text, clean up timings; returns new dicts.

    - cues longer than *max_cue_seconds* → split at sentence boundaries
    - text → :func:`wrap` (empty cues are dropped)
    - ``end - start`` extended up to *min_duration* (never into the next cue)
    - cues pushed apart by *min_gap*, keeping at least *min_visible* on screen
    A single sentence longer than *max_cue_seconds* stays long: it cannot be
    split without word timings, and chopping display time mid-sentence is worse
    than an over-long cue.
    """
    out: list[dict] = []
    for seg in segments:
        text = normalize(seg.get("text", ""))
        if not text:
            continue
        start = max(0.0, float(seg.get("start", 0.0)))
        end = max(start, float(seg.get("end", start)))
        for piece in split_long_cue({"start": start, "end": end, "text": text}, max_cue_seconds):
            wrapped = wrap(piece["text"], max_chars, max_lines)
            if wrapped:
                out.append({"start": piece["start"], "end": piece["end"], "text": wrapped})

    for i, seg in enumerate(out):
        nxt = out[i + 1] if i + 1 < len(out) else None
        ceiling = (nxt["start"] - min_gap) if nxt else None

        if seg["end"] - seg["start"] < min_duration:
            target = seg["start"] + min_duration
            seg["end"] = target if ceiling is None else min(target, ceiling)
        if ceiling is not None and seg["end"] > ceiling:
            seg["end"] = max(ceiling, seg["start"] + MIN_VISIBLE)

        if seg["end"] - seg["start"] < MIN_VISIBLE:
            seg["end"] = seg["start"] + MIN_VISIBLE

        seg["start"] = round(seg["start"], 3)
        seg["end"] = round(seg["end"], 3)
    return out
