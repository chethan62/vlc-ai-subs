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
    # Horizontal runs collapse; a deliberate line break survives. Newlines used to
    # be flattened here, which destroyed the engine's wrapping downstream.
    assert normalize("  a \n b\t\tc ") == "a\nb c"
    assert normalize("a   b   c") == "a b c"
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


def test_a_long_cue_is_clamped_to_the_maximum():
    """Reversed, on the film's evidence. This used to read "Splitting needs per-word
    timings; shortening display time is worse" and left the cue showing for 12 s.
    Seven seconds is the published maximum (Netflix timed-text guides), and the film
    showed the cost of ignoring it: a two-word cue displayed for 51.6 s. Clamping
    drops nothing — the text is short enough to read in a fraction of the span, and
    the missing time is silence, not speech."""
    out = apply_quality([{"start": 0.0, "end": 12.0, "text": "a very long cue"}])
    assert out[0]["end"] == pytest.approx(7.0)


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


# ── Reading speed and per-language limits (Netflix timed-text guides) ────────
# The plugin used to have one line width (42) and no reading-speed notion at all.
# Netflix's per-language guides — 20 CPS English, 17 most other languages, 9
# Chinese, 4 Japanese, 12 Korean, with 42- vs 13-16-character lines — are the
# most widely cited public spec, so they are what the cue pass now enforces.


def test_limits_follow_the_language_and_the_script():
    from core.cues import limits_for

    assert limits_for("en") == (20.0, 42)
    assert limits_for("zh") == (9.0, 16)
    assert limits_for("ja") == (4.0, 13)
    assert limits_for("ko") == (12.0, 16)
    assert limits_for("de")[0] == 17.0            # the guides' "all other languages"
    assert limits_for("zh-Hans") == (9.0, 16)     # region subtags do not matter
    assert limits_for(None, "ちょっと") == (4.0, 13)      # script decides when unknown
    assert limits_for(None, "안녕하세요") == (12.0, 16)
    assert limits_for(None, "测试一下") == (9.0, 16)
    assert limits_for(None, "Hello there") == (20.0, 42)
    assert limits_for("en", "测试一下") == (9.0, 16)      # a wrong tag loses to the text


def test_a_dense_cue_is_given_more_time_not_fewer_words():
    # 100 characters at 20 CPS needs 5 s; it starts with 1 s and the next cue is
    # far away, so the time is simply given to it.
    segs = [{"start": 0.0, "end": 1.0, "text": "x" * 100},
            {"start": 30.0, "end": 31.0, "text": "later"}]
    out = apply_quality(segs)
    assert out[0]["end"] == pytest.approx(5.0, abs=0.01)
    assert len(out[0]["text"]) == 100          # nothing was cut short to fit


def test_reading_speed_time_never_runs_into_the_next_cue():
    segs = [{"start": 0.0, "end": 1.0, "text": "x" * 200},
            {"start": 2.0, "end": 4.0, "text": "next"}]
    out = apply_quality(segs)
    assert out[0]["end"] <= out[1]["start"] - 0.08 + 1e-6
    assert out[0]["end"] == pytest.approx(1.92, abs=0.01)   # it took all the slack


def test_the_cjk_ceiling_gives_more_time_than_the_english_one():
    segs = [{"start": 0.0, "end": 1.0, "text": "x" * 30},
            {"start": 60.0, "end": 61.0, "text": "later"}]
    zh = apply_quality([dict(s) for s in segs], language="zh")[0]["end"]
    en = apply_quality([dict(s) for s in segs], language="en")[0]["end"]
    assert zh == pytest.approx(30 / 9, abs=0.01)     # 9 CPS: 3.33 s
    assert en == pytest.approx(30 / 20, abs=0.01)    # 20 CPS: 1.50 s
    assert zh > en


def test_max_cps_env_override(monkeypatch):
    segs = [{"start": 0.0, "end": 1.0, "text": "x" * 34}]
    assert apply_quality(segs)[0]["end"] == pytest.approx(1.7, abs=0.01)   # 34/20
    monkeypatch.setenv("VSCL_AISUBS_MAX_CPS", "17")
    assert apply_quality(segs)[0]["end"] == pytest.approx(2.0, abs=0.01)
    monkeypatch.setenv("VSCL_AISUBS_MAX_CPS", "junk")
    assert apply_quality(segs)[0]["end"] == pytest.approx(1.7, abs=0.01)   # falls back


def test_line_width_follows_the_language(monkeypatch):
    monkeypatch.delenv("VSCL_AISUBS_MAX_LINE", raising=False)
    out = apply_quality([{"start": 0.0, "end": 3.0, "text": "测" * 30}], language="zh")
    assert all(len(l) <= 16 for c in out for l in c["text"].split("\n"))
    assert "".join(c["text"] for c in out).replace("\n", "") == "测" * 30


def test_an_unpunctuated_cjk_cue_is_cut_into_readable_runs():
    # Real case: a 25 s Chinese cue with no terminator anywhere, so sentence
    # splitting has nothing to work with — it becomes equal character runs.
    cue = {"start": 0.0, "end": 25.0, "text": "说不定现在还在哪一个山沟里给人算命片饭吃"}
    out = apply_quality([dict(cue)], language="zh")
    assert len(out) > 1
    assert max(c["end"] - c["start"] for c in out) <= MAX_CUE_SECONDS + 0.5
    assert all(len(l) <= 16 for c in out for l in c["text"].split("\n"))
    assert "".join(c["text"] for c in out).replace("\n", "") == cue["text"]
    assert out[0]["start"] == 0.0 and out[-1]["end"] == 25.0   # span preserved


def test_an_unpunctuated_latin_cue_is_not_chopped_but_is_clamped():
    # "Lucky You" over a 44 s title card: chopping it up would be pointless and
    # padding it with invented text is not this program's business — still true.
    # Clamping the display to the 7 s maximum does neither: one cue, same text,
    # and the screen goes quiet for the silence that follows.
    out = apply_quality([{"start": 0.0, "end": 44.0, "text": "Lucky You"}], language="en")
    assert len(out) == 1, "nothing to split on, so it stays one cue"
    assert out[0]["text"] == "Lucky You", "no invented padding"
    assert out[0]["end"] == pytest.approx(7.0)


def test_a_brief_but_dense_cjk_cue_is_split_for_its_lines():
    # 40 characters is more than 2 x 16 full-width characters, however brief.
    out = apply_quality([{"start": 0.0, "end": 4.0, "text": "测" * 40}], language="zh")
    assert len(out) > 1
    assert all(len(l) <= 16 for c in out for l in c["text"].split("\n"))


def test_folding_cjk_fragments_adds_no_space():
    # A two-character opening ("好。") shares 0.95 s of a 20 s cue: below the 1 s
    # floor, so it folds back into its neighbour — with no space, because CJK
    # writes none there. (An earlier version of this test never folded at all and
    # passed with the bug still in place, which the negative control caught.)
    cue = {"start": 0.0, "end": 20.0, "text": "好。" + "测" * 40}
    out = split_long_cue(dict(cue), MAX_CUE_SECONDS, 16)
    assert len(out) == 1, "the stub should have folded back in"
    joined = out[0]["text"].replace("\n", "")
    assert joined.startswith("好。测")
    assert "。 " not in joined
