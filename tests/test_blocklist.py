"""Unit tests for core/blocklist.py — the hallucination phrase filter."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "core.blocklist", Path(__file__).resolve().parent.parent / "core" / "blocklist.py"
)
assert _SPEC is not None and _SPEC.loader is not None
bl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bl)


def test_is_blocklisted():
    assert bl.is_blocklisted("Subtitles by the Amara.org community")
    assert bl.is_blocklisted("[Music Playing]")
    assert bl.is_blocklisted("  ♪ ♪ ♪  ")
    assert bl.is_blocklisted("this video is sponsored by")


def test_not_blocklisted_real_dialogue():
    assert not bl.is_blocklisted("Thank you for watching this film with us.")
    assert not bl.is_blocklisted("You")
    assert not bl.is_blocklisted("Music was playing in the background.")
    assert not bl.is_blocklisted("Hello everyone, it's raining in the city.")


def test_blocklist_env_disable(monkeypatch):
    monkeypatch.setenv("VSCL_AISUBS_BLOCKLIST", "0")
    assert not bl.is_blocklisted("subtitles by the amara.org community")
    monkeypatch.delenv("VSCL_AISUBS_BLOCKLIST")
    assert bl.is_blocklisted("subtitles by the amara.org community")


def test_filter_segments():
    segs = [
        {"start": 0.0, "end": 1.0, "text": "Normal dialogue here."},
        {"start": 1.0, "end": 2.0, "text": "[Music playing]"},
        {"start": 2.0, "end": 3.0, "text": "subtitles by the amara.org community"},
    ]
    out = bl.filter_segments(segs)
    assert len(out) == 1
    assert out[0]["text"] == "Normal dialogue here."


def test_engine_spelling_of_the_same_hallucination_all_match():
    """WhisperX brackets sounds, whisper.cpp emits them bare and with a full
    stop — one entry has to cover every spelling, or the AMD/Intel engine's
    "music playing" slips through the filtered WhisperX form."""
    for spelling in ("[Music playing]", "music playing", "Music playing.",
                     "(music playing)", "Music  Playing!"):
        assert bl.is_blocklisted(spelling), spelling
    # …without swallowing real dialogue that merely mentions the words
    assert not bl.is_blocklisted("the music playing was lovely, wasn't it")
    assert not bl.is_blocklisted("thank you")


# ── repeat loops: the decoder's other hallucination shape ──────────────────

def test_deloop_text_collapses_a_repeated_phrase():
    """Measured in a real whisper.cpp run: "I'm sorry" 14 times in a row."""
    assert bl.deloop_text("I'm sorry, " * 14) == "I'm sorry,"


def test_deloop_text_keeps_repeats_people_actually_say():
    """Real emphatic dialogue, taken from this episode's Parakeet output — a
    4x/5x repeat is speech, and an earlier threshold of 4 ate all three."""
    for real in ("No, no, no.", "Come on, come on, come on.", "Yeah. Yeah. Yeah.",
                 "Go, go, go, go, go!",
                 "Come on, come on, come on, come on.",
                 "Hey, come on, come on, come on, come on."):
        assert bl.deloop_text(real) == real


def test_deloop_text_still_collapses_a_real_loop():
    """The measured loop was 14x, well clear of the 5x speech above."""
    assert bl.deloop_text("I'm sorry, " * 14) == "I'm sorry,"


def test_deloop_text_compares_ignoring_punctuation():
    text = ("I'm sorry, I'm sorry. I'm sorry! I'm sorry? I'm sorry, I'm sorry. "
            "I'm sorry! I'm sorry?")
    assert bl.deloop_text(text) == "I'm sorry,"


def test_deloop_text_leaves_ordinary_sentences_alone():
    text = "We pulled it off. The hard part's over."
    assert bl.deloop_text(text) == text


def test_deloop_segments_drops_the_copies_of_a_looping_cue():
    """Measured: 14 consecutive "I'm sorry." cues spanning 14 s, which neither
    the phrase blocklist nor the in-text collapse can see."""
    segs = [{"start": 75.0 + i, "end": 76.0 + i, "text": "I'm sorry."} for i in range(14)]
    out = bl.deloop_segments(segs)
    assert len(out) == 1
    assert out[0]["start"] == 75.0, "the survivor keeps its own timing"


def test_deloop_segments_keeps_short_runs_and_normal_dialogue():
    segs = [
        {"start": 0.0, "end": 1.0, "text": "Okay?"},
        {"start": 1.0, "end": 2.0, "text": "Okay?"},     # heard twice — that is speech
        {"start": 2.0, "end": 3.0, "text": "Let's go."},
    ]
    assert bl.deloop_segments(segs) == segs


def test_filter_segments_collapses_a_loop_end_to_end():
    segs = [{"start": float(i), "end": i + 1.0, "text": "I'm sorry."} for i in range(14)]
    assert [s["text"] for s in bl.filter_segments(segs)] == ["I'm sorry."]
