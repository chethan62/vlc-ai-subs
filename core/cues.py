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
# glance (Netflix's maximum event duration is 7 s), so it is split at sentence
# boundaries.
MAX_CUE_SECONDS = 7.0
# Reading-speed and line-length ceilings, straight from the Netflix Timed Text
# Style Guides (the most widely cited public subtitle spec; the BBC's 160-180 wpm
# target lands in the same range for Latin scripts):
#
#   English 20 CPS / 42 chars per line      most other Latin+ 17 CPS / 42
#   Chinese  9 CPS / 16 full-width chars    Japanese 4 CPS / 13    Korean 12 / 16
#
# CJK characters are full-width and carry far more meaning each, which is why the
# caps are so much tighter. A cue whose text needs longer than its span gets more
# time up to the next cue — the guidelines say to add time before cutting words,
# and this plugin never rewrites dialogue.
# VSCL_AISUBS_MAX_CPS / VSCL_AISUBS_MAX_LINE override both globally.
LANGUAGE_LIMITS = {
    "en": (20.0, 42),
    "zh": (9.0, 16),
    "ja": (4.0, 13),
    "ko": (12.0, 16),
}
DEFAULT_LIMITS = (17.0, 42)   # everything else in the guides' Latin/other rows
ENGLISH_LIMITS = LANGUAGE_LIMITS["en"]

_KANA = re.compile("[\u3040-\u30ff]")
_HANGUL = re.compile("[\uac00-\ud7af]")

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


_WS = re.compile(r"[^\S\n]+")   # horizontal whitespace runs — newlines are kept


def normalize(text: str | None) -> str:
    """Collapse horizontal whitespace runs and strip; keep deliberate line breaks.

    Newlines must survive: the engines wrap their cues before this runs, and
    flattening the breaks pushed lines far past the reader's limit (measured on a
    145-minute film: 397 cues with lines up to 100 characters).
    """
    return re.sub(r" *\n *", "\n", _WS.sub(" ", text or "")).strip()


def line_limit(max_chars: int | None = None) -> int:
    """Line width from the argument, else VSCL_AISUBS_MAX_LINE, else the default."""
    if max_chars is not None:
        return max(8, int(max_chars))
    raw = os.environ.get("VSCL_AISUBS_MAX_LINE", "").strip()
    if raw.isdigit():
        return max(8, int(raw))
    return MAX_LINE_CHARS


def limits_for(language: str | None = None, text: str = "") -> tuple[float, int]:
    """Reading-speed (CPS) and line-length caps for this cue.

    The language decides when it is known; otherwise the text's own script does,
    because an `auto` run whose language was never detected still must not be
    wrapped as if it were English. Japanese kana win over Han characters, since a
    Japanese line may contain both.
    """
    code = (language or "").strip().lower().replace("_", "-").split("-")[0]
    if code and code not in ("en", ""):
        return LANGUAGE_LIMITS.get(code, DEFAULT_LIMITS)
    # English (or unknown): the script still decides. A run tagged English cannot
    # honestly produce Han text, so the text wins over a wrong tag.
    if text:
        if _KANA.search(text):
            return LANGUAGE_LIMITS["ja"]
        if _HANGUL.search(text):
            return LANGUAGE_LIMITS["ko"]
        if _CJK.search(text):
            return LANGUAGE_LIMITS["zh"]
    return ENGLISH_LIMITS


def cps_limit(max_cps: float | None = None) -> float:
    """Reading-speed ceiling: the argument, else VSCL_AISUBS_MAX_CPS, else 20.

    A value below 1 would make every cue need minutes of screen time, so junk or
    absurd settings fall back to the default rather than mangling the timings.
    """
    if max_cps is not None:
        return max(1.0, float(max_cps))
    raw = os.environ.get("VSCL_AISUBS_MAX_CPS", "").strip()
    try:
        value = float(raw)
    except ValueError:
        return ENGLISH_LIMITS[0]
    return value if 1.0 <= value <= 60.0 else ENGLISH_LIMITS[0]


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


def _cannot_fit(text: str, line_chars: int) -> bool:
    """True when *text* needs more than MAX_LINES lines at *line_chars*.

    The trigger for splitting a cue. Every script is subject to it (it used to be
    CJK-only): a cue that cannot be wrapped inside the reader's limit must become
    two cues — that is what the guides prescribe — instead of one cue whose line
    breaks the limit. Measured on a 145-minute film: 397 cues (21.6%) were written
    with lines up to 100 characters against a 42-character cap.
    """
    if not text or not line_chars:
        return False
    # Measure the text as it will be rendered: a line break is a space on screen,
    # and this must give the same answer for wrapped and unwrapped text.
    text = text.replace("\n", " ")
    cjk = is_cjk(text)
    return _greedy_lines(_split_units(text, cjk), cjk, int(line_chars), MAX_LINES) is None


def _word_runs(text: str, size: int) -> list[str]:
    """Group words into runs of about *size* characters, never splitting a word.

    The Latin counterpart of the CJK equal-character-run fallback: a long
    comma-separated sentence with no sentence terminator has no other break to
    offer, and leaving it whole is what produced the over-cap lines above.
    """
    runs: list[str] = []
    current = ""
    for word in text.split(" "):
        if current and len(current) + 1 + len(word) > size:
            runs.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        runs.append(current)
    return runs


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

    # Already-wrapped text (the engine wrapped it, and the CLI re-runs this pass):
    # wrap each line on its own so the pass is idempotent and never re-flows a
    # deliberate break into a longer line.
    if "\n" in text:
        return "\n".join(wrap(line, max_chars, max_lines) for line in text.split("\n"))

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


def split_long_cue(seg: dict, max_seconds: float = MAX_CUE_SECONDS,
                   line_chars: int | None = None) -> list[dict]:
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
    text = normalize(seg.get("text", ""))
    # Over-long means "cannot be shown within the script's own limits": either too
    # long on screen (7 s), or more text than the reader's line limit can hold —
    # for any script. A brief but dense cue of Latin text that needs a third line
    # is split for that reason alone, exactly as a dense CJK cue is.
    over_capacity = _cannot_fit(text, line_chars) if line_chars else False
    if duration <= max_seconds and not over_capacity:
        return [seg]

    pieces = [p.strip() for p in _SENTENCE_BREAK.split(text)]
    pieces = [p for p in pieces if p]
    if len(pieces) < 2:
        # No sentence break to split on. For CJK that is the normal case — the
        # script has no spaces and its ASR output often carries no terminators at
        # all (measured: 26 of the 27 over-long cues in a real Chinese file), so
        # cut it into equal character runs instead. That is standard practice for
        # CJK subtitles, and the alternative is a cue nobody can finish reading.
        text = normalize(seg.get("text", ""))
        if line_chars and text and (over_capacity or _CJK.search(text)):
            per_cue = max(1, int(line_chars) * MAX_LINES)
            chunks = max(2, int(-(-len(text) // per_cue)), int(-(-duration // max_seconds)))
            size = -(-len(text) // chunks)
            if _CJK.search(text):
                pieces = [text[i:i + size] for i in range(0, len(text), size)]
            else:
                # Latin with no terminator to split on: group whole words into
                # runs. A comma-separated 100-character sentence is a real
                # measured case, and leaving it whole broke the line limit.
                pieces = _word_runs(text, size)
        if len(pieces) < 2:
            return [seg]

    # Fold in fragments that would be too brief to read on their own: "Mm-hmm."
    # at 0.2 s is a flicker, and its time cannot be extended into the next piece.
    def _join(a: str, b: str) -> str:
        # CJK writes no spaces between words, so folding two CJK fragments with
        # one opens a gap that was never in the audio (measured: 108 stray spaces
        # in a real Chinese file before this).
        if a and b and _CJK.search(a[-1]) and _CJK.search(b[0]):
            return a + b
        return f"{a} {b}"

    total_chars = sum(len(p) for p in pieces) or 1
    merged: list[str] = []
    for piece in pieces:
        share = duration * len(piece) / total_chars
        if merged and share < MIN_DURATION:
            merged[-1] = _join(merged[-1], piece)
        else:
            merged.append(piece)
    # A too-brief first piece has no predecessor to fold into — give it to the
    # piece that follows instead.
    if len(merged) > 1 and duration * len(merged[0]) / total_chars < MIN_DURATION:
        merged[1] = _join(merged[0], merged[1])
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


def borrow_for_short_cues(segments: list[dict], min_duration: float = MIN_DURATION,
                          max_seconds: float = MAX_CUE_SECONDS,
                          borrow_gap: float = 0.0) -> list[dict]:
    """Give a too-short cue time so it stays readable, without moving any text.

    Measured against the professional subtitle track on this project's test film: 19.3 % of
    our cues ran under a second against 0.4 % of theirs (our median 0.72 s, their shortest
    0.88 s), and 299 of those 365 had no free time on either side, so the per-cue pass cannot
    lift them. Time comes, in this order, from:

    1. the silence after the cue — its end moves later, but never past the next cue's start
       (its speech is the next line's, and extending over it is what makes a subtitle appear
       before the words are spoken);
    2. the silence before the cue — its start moves earlier, touching no other cue;
    3. the previous cue's own spare tail, only when the two abut, and only down to that cue's
       own floor.

    *borrow_gap* is 0, not MIN_GAP: the professional track's median gap between cues is 0.00 s,
    so closing our 2-frame gap is free time that belongs to the cue that needs it. The gap is
    still kept by the main pass, which is what keeps cues apart when nothing needs lifting.

    A donor's end is never pushed later (a 1.84 s cue became 117.56 s that way, by handing over
    the silence that followed it) and no cue is ever merged: a cue that cannot be helped is
    left exactly as it was.
    """
    out = [dict(seg) for seg in segments]
    for i, cue in enumerate(out):
        deficit = min_duration - (cue["end"] - cue["start"])
        if deficit <= 0.001:
            continue

        following = out[i + 1] if i + 1 < len(out) else None
        previous = out[i - 1] if i else None

        # 1. Free time after, stopping short of the next cue's speech.
        if following is not None:
            room = (following["start"] - borrow_gap) - cue["end"]
            if room > 1e-9:
                taken = min(deficit, room)
                cue["end"] += taken
                deficit -= taken

        # 2. Free time before, which belongs to nobody.
        if deficit > 0.001 and previous is not None:
            room = (cue["start"] - previous["end"]) - borrow_gap
            if room > 1e-9:
                taken = min(deficit, room)
                cue["start"] -= taken
                deficit -= taken

        # 3. The earlier cue's spare tail — abutting neighbours only, and it never goes under
        # its own floor. Earlier first, so the next line still comes up with its own speech.
        if deficit > 0.001 and previous is not None and cue["start"] - previous["end"] <= borrow_gap + 1e-9:
            spare = (previous["end"] - previous["start"]) - min_duration
            taken = min(deficit, max(0.0, spare), cue["start"] - previous["start"] - min_duration)
            if taken > 1e-9:
                cue["start"] -= taken
                previous["end"] = cue["start"]

        if cue["end"] - cue["start"] > max_seconds:
            cue["end"] = cue["start"] + max_seconds
        cue["start"] = round(cue["start"], 3)
        cue["end"] = round(cue["end"], 3)
        if previous is not None:
            previous["end"] = round(previous["end"], 3)
        if following is not None:
            following["start"] = round(following["start"], 3)
    return out


def apply_quality(
    segments: list[dict],
    max_chars: int | None = None,
    max_lines: int = MAX_LINES,
    min_duration: float = MIN_DURATION,
    min_gap: float = MIN_GAP,
    max_cue_seconds: float = MAX_CUE_SECONDS,
    max_cps: float | None = None,
    language: str | None = None,
) -> list[dict]:
    """Split over-long cues, wrap cue text, clean up timings; returns new dicts.

    - cues longer than *max_cue_seconds* → split at sentence boundaries (for CJK,
      into equal character runs when the text has no punctuation to split on)
    - text → :func:`wrap`, at the language's line width from the Netflix timed-text
      guides (English 42, Chinese/Korean 16 full-width, Japanese 13); VSCL_AISUBS_MAX_LINE wins
    - a cue whose text exceeds the language's reading-speed ceiling → end extended
      toward the next cue (the guidelines prefer adding time to cutting words)
    - ``end - start`` extended up to *min_duration* (never into the next cue)
    - cues pushed apart by *min_gap*, keeping at least *min_visible* on screen
    A cue boxed in by its neighbour keeps its dense text rather than losing words.
    """
    env_line = os.environ.get("VSCL_AISUBS_MAX_LINE", "").strip()
    env_cps = os.environ.get("VSCL_AISUBS_MAX_CPS", "").strip()
    fixed_cps = cps_limit(max_cps) if (max_cps is not None or env_cps) else None
    fixed_line = max_chars if max_chars is not None else (line_limit(None) if env_line else None)
    out: list[dict] = []
    for seg in segments:
        text = normalize(seg.get("text", ""))
        if not text:
            continue
        _, auto_line = limits_for(language, text)
        line_cap = fixed_line or auto_line
        start = max(0.0, float(seg.get("start", 0.0)))
        end = max(start, float(seg.get("end", start)))
        for piece in split_long_cue({"start": start, "end": end, "text": text},
                                   max_cue_seconds, line_cap):
            wrapped = wrap(piece["text"], line_cap, max_lines)
            if wrapped:
                out.append({"start": piece["start"], "end": piece["end"], "text": wrapped})

    for i, seg in enumerate(out):
        nxt = out[i + 1] if i + 1 < len(out) else None
        ceiling = (nxt["start"] - min_gap) if nxt else None

        # Reading speed: a cue whose text cannot be read in its span gets more
        # time, up to (never past) the next cue. The ceiling follows the language
        # (Netflix: 20 CPS English, 17 most others, 9 Chinese, 4 Japanese, 12
        # Korean), counts every character including spaces, and our own line
        # break stands in for a space.
        needed = len(seg["text"].replace("\n", " ")) / (fixed_cps or limits_for(language, seg["text"])[0])
        if seg["end"] - seg["start"] < needed:
            target = seg["start"] + needed
            seg["end"] = target if ceiling is None else min(target, ceiling)

        if seg["end"] - seg["start"] < min_duration:
            target = seg["start"] + min_duration
            seg["end"] = target if ceiling is None else min(target, ceiling)
        if ceiling is not None and seg["end"] > ceiling:
            # Never run past the ceiling. A cue visible for less than MIN_VISIBLE
            # is a blemish; an overlap the renderer has to resolve is a fault.
            # Measured on the film: one pair 0.1s apart, where the MIN_VISIBLE
            # floor pushed a cue past its neighbour.
            seg["end"] = max(ceiling, seg["start"])

        if seg["end"] - seg["start"] < MIN_VISIBLE:
            target = seg["start"] + MIN_VISIBLE
            seg["end"] = min(target, ceiling) if ceiling is not None else target

        # Nothing may be shown longer than the maximum, even when the text offered
        # no break to split on. Measured on the film: a two-word cue stretched
        # across a 54s silence was displayed for 51.6s.
        if seg["end"] - seg["start"] > max_cue_seconds:
            seg["end"] = seg["start"] + max_cue_seconds

        seg["start"] = round(seg["start"], 3)
        seg["end"] = round(seg["end"], 3)

    # Last: lift the cues that are too brief to read. This moves boundaries only, and the
    # per-cue pass above cannot do it (measured: 299 of 365 short cues are boxed in).
    return borrow_for_short_cues(out)
