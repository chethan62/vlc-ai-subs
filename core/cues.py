"""Cue quality: line breaking + timing cleanup, shared by both engines.

Raw ASR output is one long unwrapped line per segment, which is unreadable as
a subtitle. This pass applies broadcast-style presentation rules before the SRT
is written:

* at most ``MAX_LINES`` lines per cue, each at most ``MAX_LINE_CHARS`` wide
  (CJK scripts need far fewer characters per line — ``CJK_LINE_CHARS``)
* breaks at word boundaries for space-delimited scripts, anywhere in CJK
* cue timings clamped to at least ``MIN_DURATION``, spaced by ``MIN_GAP``

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

_WS = re.compile(r"\s+")
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


def apply_quality(
    segments: list[dict],
    max_chars: int | None = None,
    max_lines: int = MAX_LINES,
    min_duration: float = MIN_DURATION,
    min_gap: float = MIN_GAP,
) -> list[dict]:
    """Wrap cue text and clean up timings; returns new dicts.

    - text → :func:`wrap` (empty cues are dropped)
    - ``end - start`` extended up to *min_duration* (never into the next cue)
    - cues pushed apart by *min_gap*, keeping at least *min_visible* on screen
    Long cues are left long: splitting them needs per-word timings that only
    some engines provide, and chopping display time is worse than a 9 s cue.
    """
    out: list[dict] = []
    for seg in segments:
        text = wrap(seg.get("text", ""), max_chars, max_lines)
        if not text:
            continue
        start = max(0.0, float(seg.get("start", 0.0)))
        end = max(start, float(seg.get("end", start)))
        out.append({"start": start, "end": end, "text": text})

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
