"""Film-scale cue regressions, measured on a real 145-minute feature.

The 47.5-minute episode this project had been testing with passed every check. The
film (1841 cues, 2h18m) failed five of them, and the causes were all in paths the
shorter media never reached:

- 397 cues (21.6%) came out with a line up to 100 characters against the 42-char cap.
  `split_long_cue` only split cues that were too long *in time*, or CJK text that was
  too long *in width*: a Latin cue whose text needs a third line was left whole, and
  `wrap`'s balanced fallback then emitted one 48/51-char pair or, once
  `filter_segments` had flattened the line breaks, a single 100-char line.
- `apply_quality` re-wraps, but the CLI ran `filter_segments` **after** the engine
  had wrapped the text — the filter normalises and flattens, so every line break was
  lost. The wrapping looked correct when tested in isolation.
- 43 cues exceeded the 7 s maximum, the worst showing a two-word cue ("Just You")
  stretched across a 54 s silence for 51.6 s: `split_long_cue` returns such a cue
  untouched when it has no break to split on, and nothing clamped it afterwards.
- one pair of cues overlapped, where MIN_VISIBLE pushed a cue past its neighbour.

Every figure below is from that film's own output.
"""

import re

from core.blocklist import deloop_text, filter_segments
from core.cues import MAX_CUE_SECONDS, MAX_LINE_CHARS, apply_quality, wrap

# The real cue: 100 characters, one comma-separated sentence with no terminator.
LONG = ("Further, in collaboration with the Department of Defense, Ordix has "
        "confiscated, reverse-engineered,")


def _lines(cue):
    return cue["text"].splitlines()


def test_latin_text_that_needs_a_third_line_is_split():
    """The 100-char cue: 1841 cues produced 397 with an over-cap line."""
    out = apply_quality([{"start": 0.0, "end": 6.0, "text": LONG}], language="en")
    assert len(out) >= 2, "a text that cannot fit two lines must become two cues"
    for cue in out:
        assert len(_lines(cue)) <= 2
        assert max(len(l) for l in _lines(cue)) <= MAX_LINE_CHARS


def test_splitting_loses_no_words():
    """The guarantee that makes splitting acceptable at all."""
    out = apply_quality([{"start": 0.0, "end": 6.0, "text": LONG}], language="en")
    before = re.findall(r"\S+", LONG)
    after = re.findall(r"\S+", " ".join(c["text"] for c in out))
    assert before == after


def test_splitting_keeps_the_cues_in_order_and_inside_the_span():
    out = apply_quality([{"start": 10.0, "end": 16.0, "text": LONG}], language="en")
    assert out[0]["start"] == 10.0
    assert out[-1]["end"] <= 16.0 + 1e-9
    for a, b in zip(out, out[1:]):
        assert a["end"] <= b["start"] + 1e-9, "split cues must not overlap"


def test_a_two_word_cue_in_a_long_silence_is_clamped():
    """The measured 51.6s cue: text with no break to split on, over a 54s gap."""
    out = apply_quality([{"start": 2846.0, "end": 2897.6, "text": "Just You"}], language="en")
    assert len(out) == 1
    assert out[0]["end"] - out[0]["start"] <= MAX_CUE_SECONDS, "no cue may be shown for 51.6s"


def test_nothing_is_shown_longer_than_the_maximum():
    for text in ("Just You", LONG, "Mm-hmm."):
        for span in (30.0, 51.6, 90.0):
            out = apply_quality([{"start": 0.0, "end": span, "text": text}], language="en")
            for cue in out:
                assert cue["end"] - cue["start"] <= MAX_CUE_SECONDS + 1e-9


def test_the_min_visible_floor_never_crosses_the_next_cue():
    """The one overlapping pair in the film: two cues 0.1s apart."""
    out = apply_quality(
        [{"start": 0.0, "end": 1.0, "text": "One."},
         {"start": 1.1, "end": 2.0, "text": "Two."}],
        language="en",
    )
    for a, b in zip(out, out[1:]):
        assert a["end"] <= b["start"] + 1e-9, "an overlap is a rendering fault"


def test_the_filter_preserves_the_line_breaks_it_is_given():
    """The step that destroyed the engine's wrapping — now fixed at the source."""
    wrapped = wrap(LONG, MAX_LINE_CHARS, 2)
    assert len(_lines({"text": wrapped})) == 2
    filtered = filter_segments([{"start": 0.0, "end": 6.0, "text": wrapped}])
    assert "\n" in filtered[0]["text"], (
        "deloop_text and normalize must not flatten a deliberate line break; "
        "flattening here is what pushed 397 lines past the reader's limit"
    )
    assert max(len(l) for l in filtered[0]["text"].splitlines()) <= 55, "no re-flow into a longer line"


def test_filter_then_quality_restores_the_line_limit():
    """The CLI's pipeline, in the CLI's order."""
    wrapped = wrap(LONG, MAX_LINE_CHARS, 2)
    segments = filter_segments([{"start": 0.0, "end": 6.0, "text": wrapped}])
    segments = apply_quality(segments, language="en")
    for cue in segments:
        assert max(len(l) for l in _lines(cue)) <= MAX_LINE_CHARS


def test_quality_is_idempotent():
    """The CLI re-runs it on text the engine already processed."""
    once = apply_quality([{"start": 0.0, "end": 6.0, "text": LONG}], language="en")
    twice = apply_quality(once, language="en")
    assert [(round(c["start"], 3), round(c["end"], 3), c["text"]) for c in once] == \
           [(round(c["start"], 3), round(c["end"], 3), c["text"]) for c in twice]


def test_scattered_repeats_are_dialogue_not_a_loop():
    """The film says "Okay." 16 times over 145 minutes — normal dialogue, kept.

    Measured on the film: the longest run of identical *consecutive* cues was 4.
    The deloop collapses adjacent identical cues (a decoder loop) and must leave
    dialogue alone — including dialogue that happens to repeat the same word many
    times, as long as other lines sit between the copies.
    """
    scattered = []
    for i in range(16):
        scattered.append({"start": i * 60.0, "end": i * 60.0 + 1.0, "text": "Okay."})
        scattered.append({"start": i * 60.0 + 2.0, "end": i * 60.0 + 3.0,
                          "text": f"Different line {i}."})
    kept = [s for s in filter_segments(scattered) if s["text"] == "Okay."]
    assert len(kept) == 16, "16 'Okay.' spread through a film is dialogue"

    loop = [{"start": i * 0.5, "end": i * 0.5 + 0.4, "text": "Okay."} for i in range(12)]
    assert len(filter_segments(loop)) < len(loop), "a real decoder loop must still collapse"


def test_deloop_leaves_a_two_line_cue_alone():
    """The deloop must not be the thing that flattens a wrapped cue's breaks."""
    assert "\n" in deloop_text("First line\nsecond line"), "deloop_text must not flatten"
