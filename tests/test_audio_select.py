"""Which audio track gets transcribed — core/audio.py stream selection.

The bug this pins: ffmpeg's default audio stream is the first one, and on real
dual-audio releases that is the dub, not the dialogue. Measured on this machine:

  * "Lanterns...MULTi.VFF...mkv"    track 1 fre (VFF) · track 2 eng · track 3 eng "Descriptive"
  * "Radioactive.Emergency...DUAL"  track 1 por · track 2 eng  (both flagged default=1)

so asking for English subtitles handed an English-only model French audio, which
answered with plausible-looking nonsense ("Tromaris troubles in sentence patrimony
genetic and young women") instead of failing. The JSON below is that real ffprobe
output, trimmed to the fields the parser reads.
"""

import array
import json
import shutil
import subprocess
import wave

import pytest

from core.audio import (
    ISO3_TO_1,
    choose_audio_stream,
    list_audio_streams,
    normalise_language,
)

# ffprobe -print_format json -show_streams -select_streams a  (real output, trimmed)
LANTERNS_JSON = json.dumps({"streams": [
    {"index": 1, "codec_name": "eac3", "channels": 6,
     "tags": {"language": "fre", "title": "VFF", "BPS": "256000"},
     "disposition": {"default": 1, "dub": 0, "visual_impaired": 0, "comment": 0}},
    {"index": 2, "codec_name": "eac3", "channels": 6,
     "tags": {"language": "eng", "BPS": "640000"}, "disposition": {}},
    {"index": 3, "codec_name": "eac3", "channels": 6,
     "tags": {"language": "eng", "title": "Descriptive", "BPS": "256000"},
     "disposition": {}},
]})

DUAL_JSON = json.dumps({"streams": [
    {"index": 1, "codec_name": "eac3", "channels": 6,
     "tags": {"language": "por"}, "disposition": {"default": 1}},
    {"index": 2, "codec_name": "eac3", "channels": 6,
     "tags": {"language": "eng"}, "disposition": {"default": 1}},
]})

MONO_JSON = json.dumps({"streams": [
    {"index": 1, "codec_name": "aac", "channels": 2, "tags": {}, "disposition": {"default": 1}},
]})

# a descriptive track that carries the *standard* flag rather than a title
FLAGGED_JSON = json.dumps({"streams": [
    {"index": 1, "codec_name": "aac", "channels": 2,
     "tags": {"language": "eng", "title": "English"},
     "disposition": {"visual_impaired": 1}},
    {"index": 2, "codec_name": "aac", "channels": 2,
     "tags": {"language": "eng"}, "disposition": {}},
]})


def _streams(monkeypatch, payload: str) -> list[dict]:
    """Parse a fixture through the real code path (ffprobe is stubbed, not bypassed)."""
    class _Done:
        returncode = 0
        stdout = payload
        stderr = ""

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Done())
    return list_audio_streams("/some/video.mkv")


def test_language_codes_are_normalised():
    assert normalise_language("eng") == "en"
    assert normalise_language("fr-FR") == "fr"
    assert normalise_language("FRE") == "fr"
    assert normalise_language("zh-Hans") == "zh"
    assert normalise_language("auto") == ""
    assert normalise_language(None) == ""
    assert ISO3_TO_1["por"] == "pt" and ISO3_TO_1["chi"] == "zh"


def test_streams_are_parsed_with_their_language_titles_and_flags(monkeypatch):
    streams = _streams(monkeypatch, LANTERNS_JSON)
    assert [s["index"] for s in streams] == [1, 2, 3]
    assert [s["language"] for s in streams] == ["fre", "eng", "eng"]
    assert [s["non_dialogue"] for s in streams] == [False, False, True]
    # the standard disposition flag is *also* honoured when a file sets it
    assert [s["non_dialogue"] for s in _streams(monkeypatch, FLAGGED_JSON)] == [True, False]


def test_the_requested_language_wins_over_the_first_track(monkeypatch):
    streams = _streams(monkeypatch, LANTERNS_JSON)
    index, why = choose_audio_stream(streams, "en")
    assert index == 2                      # not 1: that is the French VFF dub
    assert "eng" in why and "VFF" not in why

    index, why = choose_audio_stream(streams, "fr")
    assert index == 1 and "fre" in why

    streams = _streams(monkeypatch, DUAL_JSON)
    assert choose_audio_stream(streams, "en")[0] == 2
    assert choose_audio_stream(streams, "pt")[0] == 1


def test_a_descriptive_track_is_never_chosen(monkeypatch):
    streams = _streams(monkeypatch, LANTERNS_JSON)
    # English is requested; tracks 2 and 3 are both English, and 3 is the
    # audio-description narration — dialogue must win.
    assert choose_audio_stream(streams, "en")[0] == 2
    # And with no language requested, the French dialogue still beats narration.
    index, why = choose_audio_stream(streams, None)
    assert index == 1 and "descriptive track(s) skipped" in why


def test_without_a_language_it_takes_the_first_dialogue_track(monkeypatch):
    streams = _streams(monkeypatch, DUAL_JSON)
    index, why = choose_audio_stream(streams, "auto")
    assert index == 1 and "por" in why and "no en track" not in why


def test_a_missing_language_is_reported_not_hidden(monkeypatch):
    streams = _streams(monkeypatch, DUAL_JSON)
    index, why = choose_audio_stream(streams, "ja")
    assert index == 1 and "no ja track" in why


def test_single_track_files_are_unchanged(monkeypatch):
    streams = _streams(monkeypatch, MONO_JSON)
    index, why = choose_audio_stream(streams, "en")
    assert index == 1 and "track 1/1" in why


def test_no_stream_information_falls_back_to_ffmpeg(monkeypatch):
    monkeypatch.delenv("VSCL_AISUBS_AUDIO_TRACK", raising=False)
    assert choose_audio_stream([], "en") == (None, "ffmpeg's own default track (no stream info available)")


def test_the_env_override_accepts_index_position_or_language(monkeypatch):
    streams = _streams(monkeypatch, LANTERNS_JSON)
    for value, expected in (("3", 3), ("1", 1), ("eng", 2), ("fre", 1), ("pt", 1)):
        monkeypatch.setenv("VSCL_AISUBS_AUDIO_TRACK", value)
        index, why = choose_audio_stream(streams, "en")
        assert index == expected, f"{value} -> {index}, expected {expected}"
        assert "VSCL_AISUBS_AUDIO_TRACK" in why
    monkeypatch.setenv("VSCL_AISUBS_AUDIO_TRACK", "nonsense")
    index, why = choose_audio_stream(streams, "en")
    assert index == 1 and "matched nothing" in why   # falls back, says so


# ── the same thing against real ffmpeg: does -map actually select that track? ──

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not installed",
)


def _rms(path: str) -> float:
    with wave.open(path, "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    if not frames:
        return 0.0
    samples = array.array("h")
    samples.frombytes(frames)
    return (sum(abs(s) for s in samples) / len(samples)) / 32768.0


@needs_ffmpeg
def test_the_chosen_track_is_the_one_that_gets_decoded(tmp_path):
    """Track 1 is a loud tone, track 2 is tagged French, track 3 is silence.

    Track 3 standing in for "the dub nobody asked for": if the selection is ignored
    the decoded audio is the tone, so the silence proves -map was applied.
    """
    from core.audio import decode_to_wav16k

    media = tmp_path / "multi.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "2",
        "-map", "0:a", "-map", "1:a", "-map", "2:a",
        "-metadata:s:a:0", "language=tone",
        "-metadata:s:a:1", "language=fre",
        "-metadata:s:a:2", "language=eng",
        "-c:a", "flac", str(media),
    ], check=True, capture_output=True)

    streams = list_audio_streams(str(media))
    assert len(streams) == 3, streams

    index, why = choose_audio_stream(streams, "en")
    wav = decode_to_wav16k(str(media), stream_index=index)
    try:
        assert _rms(wav) < 0.01, f"expected the silent eng track, got audio ({why})"
    finally:
        import os
        os.unlink(wav)

    # Negative control: the previous behaviour, no -map at all, takes track 1.
    wav = decode_to_wav16k(str(media))
    try:
        assert _rms(wav) > 0.05, "ffmpeg's default should have taken the tone track"
    finally:
        import os
        os.unlink(wav)
