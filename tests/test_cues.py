"""Unit tests for core/cues.py — cue line-breaking and timing cleanup.

The wrapping rules are the difference between a wall of text and a watchable
subtitle, and they run on every engine's output, so they are pinned here:
word-boundary wrapping for spaced scripts, char wrapping for CJK, and timing
cleanup that never eats the next cue.
"""

import pytest

from core.cues import (
    CJK_LINE_CHARS,
    MAX_CUE_SECONDS,
    MAX_LINE_CHARS,
    apply_quality,
    is_cjk,
    normalize,
    split_long_cue,
    wrap,
)


def _widths(wrapped: str) -> list[int]:
    return [len(line) for line in wrapped.split("\n")]


def test_normalize_collapses_whitespace():
    assert normalize("  a \n b\t\tc ") == "a b c"
    assert normalize(None) == ""


def test_short_text_is_untouched():
    assert wrap("Hello there.") == "Hello there."


def test_long_text_wraps_within_two_lines_and_loses_nothing():
    text = "The quick brown fox jumps over the lazy dog while the camera keeps rolling"
    wrapped = wrap(text)
    lines = wrapped.split("\n")
    assert len(lines) == 2
    assert all(len(line) <= MAX_LINE_CHARS for line in lines)
    assert wrapped.replace("\n", " ") == text  # nothing dropped or reordered


def test_text_too_long_for_two_lines_is_split_in_balance():
    """A 2-line cue must not be one full line plus a long tail."""
    text = ("The quick brown fox jumps over the lazy dog while the camera keeps "
            "rolling on the empty street")
    wrapped = wrap(text)
    first, second = wrapped.split("\n")
    assert wrapped.replace("\n", " ") == text
    assert abs(len(first) - len(second)) <= 10
    assert min(len(first), len(second)) > MAX_LINE_CHARS // 2


def test_breaks_fall_on_word_boundaries():
    wrapped = wrap("one two three four five six seven eight nine ten eleven twelve")
    assert all(not line.startswith(" ") and not line.endswith(" ") for line in wrapped.split("\n"))
    assert "\n" in wrapped


def test_wrap_is_idempotent():
    text = "The quick brown fox jumps over the lazy dog and keeps on running"
    once = wrap(text)
    assert wrap(once) == once


def test_single_long_word_is_kept_intact():
    word = "A" * 60
    assert wrap(word) == word  # overlong, but never truncated


def test_cjk_is_detected_and_wrapped_by_characters():
    assert is_cjk("你好世界")
    assert not is_cjk("hello")
    text = "这是一段很长的中文字幕文本需要按照字数换行显示给观众阅读"
    wrapped = wrap(text)
    lines = wrapped.split("\n")
    assert len(lines) == 2
    assert all(len(line) <= CJK_LINE_CHARS for line in lines)
    assert wrapped.replace("\n", "") == text  # no text lost, no spaces added


def test_env_override_changes_line_width(monkeypatch):
    monkeypatch.setenv("VSCL_AISUBS_MAX_LINE", "20")
    wrapped = wrap("alpha beta gamma delta epsilon")
    assert max(_widths(wrapped)) <= 20
    assert "\n" in wrapped


def test_empty_text_wraps_to_empty():
    assert wrap("   ") == ""


# ── timing cleanup ─────────────────────────────────────────────────────

def test_short_cue_is_extended_to_min_duration():
    out = apply_quality([{"start": 0.0, "end": 0.3, "text": "Hi"}])
    assert out[0]["end"] == pytest.approx(1.0)


def test_extended_cue_never_overruns_the_next_cue():
    out = apply_quality([
        {"start": 0.0, "end": 0.2, "text": "one"},
        {"start": 0.9, "end": 2.5, "text": "two"},
    ])
    # Extended to the minimum duration, but stopping one MIN_GAP short of the
    # next cue (0.9 - 0.08) so the two do not sit back to back.
    assert out[0]["end"] == pytest.approx(0.82)
    assert out[0]["end"] < out[1]["start"]


def test_back_to_back_cues_get_a_gap():
    out = apply_quality([
        {"start": 0.0, "end": 2.0, "text": "one"},
        {"start": 2.0, "end": 4.0, "text": "two"},
    ])
    assert out[1]["start"] - out[0]["end"] >= 0.0
    assert out[0]["end"] < out[1]["start"]


def test_long_cue_is_left_long():
    """Splitting needs per-word timings; shortening display time is worse."""
    out = apply_quality([{"start": 0.0, "end": 12.0, "text": "a very long cue"}])
    assert out[0]["end"] == pytest.approx(12.0)


def test_empty_cues_are_dropped_and_starts_clamped():
    out = apply_quality([
        {"start": 0.0, "end": 1.0, "text": "   "},
        {"start": -3.0, "end": 1.5, "text": "kept"},
    ])
    assert len(out) == 1
    assert out[0]["start"] == 0.0


def test_timestamps_are_rounded_to_milliseconds():
    out = apply_quality([{"start": 1.234567, "end": 4.987654, "text": "x"}])
    assert out[0]["start"] == 1.235
    assert out[0]["end"] == 4.988


def test_quality_wraps_text_it_is_given():
    text = "This line is definitely longer than forty two characters, yes it is"
    out = apply_quality([{"start": 0.0, "end": 5.0, "text": text}])
    assert "\n" in out[0]["text"]
    assert out[0]["text"].replace("\n", " ") == text


# ── over-long cues: split at sentence boundaries ───────────────────────────
# Only the CUDA alignment pass re-segments a Whisper transcript; without it
# (CPU, AMD/Intel, a translated run, whisper.cpp) whole exchanges arrive as one
# segment. This is the real output of a 5-minute episode excerpt (large-v3-turbo
# on CUDA, alignment skipped because of a bug): 8 cues for 5 minutes, the
# longest 29 s — unreadable. Splitting needs no word timings if the split
# follows the text's own sentence endings and shares out the time by length.

REAL_GIANT_CUES = [
    {"start": 23.757, "end": 24.757, "text": "Lucky!"},
    {"start": 62.941, "end": 92.017, "text": (
        "You kept your dad's lighter? Yes. You okay? Ask me again tomorrow. Hey, "
        "come on. Come on, come on, come on. The hard part's over. We pulled it off."
    )},
    {"start": 92.658, "end": 121.278, "text": (
        "I don't know. Something feels off. Well, it's because you're hardwired to "
        "feel that way. Yes, but it doesn't mean I'm wrong. Hey, look. Everything is "
        "going according to plan. Until it isn't. You know what I think? You know "
        "what I'm gonna say? Mm-hmm. You gotta get them out of your head. Yeah. "
        "Maybe. What's he saying?"
    )},
    {"start": 156.395, "end": 174.468, "text": (
        "Okay. Besides, America's kind of over anyway. Well, it is for us. We're out "
        "of here, baby. Okay, let's do it. Let's finish getting dressed. Let's go "
        "downstairs. I am dressed. You're the one that needs to get dressed. You are "
        "dressed. You look amazing. What am I doing? Five minutes. Can you grab my "
        "watch? Yeah."
    )},
]


def test_split_long_cue_shares_the_span_in_proportion_to_length():
    seg = {"start": 0.0, "end": 20.0, "text": "Short one. " + "A much longer sentence here."}
    out = split_long_cue(seg)
    assert [p["text"] for p in out] == ["Short one.", "A much longer sentence here."]
    assert out[0]["start"] == 0.0 and out[-1]["end"] == 20.0
    assert out[0]["end"] == out[1]["start"], "pieces must stay contiguous"
    # the longer half gets the longer share of the 20 s
    assert (out[0]["end"] - out[0]["start"]) < (out[1]["end"] - out[1]["start"])


def test_split_long_cue_leaves_short_and_unsplittable_cues_alone():
    short = {"start": 0.0, "end": 4.0, "text": "One. Two. Three."}
    assert split_long_cue(short) == [short]
    one_sentence = {"start": 0.0, "end": 20.0, "text": "No terminator in this one at all"}
    assert split_long_cue(one_sentence) == [one_sentence]


def test_split_long_cue_folds_away_unreadable_fragments():
    """A 0.2 s flicker cannot be read — and its time cannot be extended, because
    the next piece starts where it ends. Fold it into a neighbour instead."""
    seg = {"start": 0.0, "end": 30.0,
           "text": "A reasonably long opening line here. Mm-hmm. And a closing line that is also long."}
    out = split_long_cue(seg)
    assert all(p["end"] - p["start"] >= 1.0 for p in out), [p["text"] for p in out]
    assert "Mm-hmm." in " ".join(p["text"] for p in out)   # merged, never dropped


def test_split_long_cue_handles_cjk_terminators():
    seg = {"start": 0.0, "end": 20.0, "text": "你好，很高兴见到你。我也是，很久不见了。今天天气很好。"}
    out = split_long_cue(seg)
    assert len(out) == 3
    assert "".join(p["text"] for p in out) == seg["text"]


def test_real_giant_cues_become_readable_without_losing_a_word():
    out = apply_quality([dict(c) for c in REAL_GIANT_CUES])
    durations = [c["end"] - c["start"] for c in out]
    assert len(out) >= 25, f"8 cues for 5 minutes should split into many: got {len(out)}"
    assert max(durations) <= 13.5, f"longest cue still {max(durations):.1f}s"
    assert sum(1 for d in durations if d > MAX_CUE_SECONDS) <= 3, "only unsplittable single sentences may stay long"
    assert min(durations) >= 0.9, f"no cue flashes past unreadably (min {min(durations):.3f}s)"
    # the deliverable is the text: every word survives, in order
    assert (" ".join(c["text"] for c in out).replace("\n", " ").split()
            == " ".join(c["text"] for c in REAL_GIANT_CUES).split())
    assert all(out[i]["end"] <= out[i + 1]["start"] + 1e-9 for i in range(len(out) - 1))
    assert out[0]["start"] >= REAL_GIANT_CUES[0]["start"]


def test_short_segments_are_untouched_by_the_splitter():
    """The engines whose segments are already short must not be re-cut."""
    segs = [{"start": 0.0, "end": 3.0, "text": "Hello there."},
            {"start": 3.2, "end": 6.0, "text": "How are you?"}]
    out = apply_quality([dict(s) for s in segs])
    assert [(c["start"], c["end"]) for c in out] == [(0.0, 3.0), (3.2, 6.0)]
