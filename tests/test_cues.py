"""Unit tests for core/cues.py — cue line-breaking and timing cleanup.

The wrapping rules are the difference between a wall of text and a watchable
subtitle, and they run on every engine's output, so they are pinned here:
word-boundary wrapping for spaced scripts, char wrapping for CJK, and timing
cleanup that never eats the next cue.
"""

import pytest

from core.cues import (
    CJK_LINE_CHARS,
    MAX_LINE_CHARS,
    apply_quality,
    is_cjk,
    normalize,
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
