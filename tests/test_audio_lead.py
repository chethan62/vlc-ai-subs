"""The audio lead: reported, never silently corrected.

A container can start its audio stream after its video (measured on a real film: 1.008 s,
24 frames at 23.976 fps). Our decode begins the WAV at the first audio sample, so
everything we time is on the audio timeline — a uniform sync complaint waiting to happen.

Adding the lead back was measured, not assumed. Against that film's own professional
subtitle track, it moved every cue from 0.53 s early to 0.48 s late and cut cue-start
matches from 215 to 131. The lead and the model's emission lag partly cancel, so the
residual is smaller than either correction implies — hence a report, not a fix.
"""

import json

from core.audio import audio_lead_note, list_audio_streams


def _stream(index, lead):
    return {"index": index, "language": "eng", "title": "", "channels": 2,
            "start_time": lead, "non_dialogue": False}


def test_a_leading_audio_stream_is_reported_with_its_offset():
    note = audio_lead_note([_stream(1, 1.008)], 1)
    assert note is not None
    assert "1.008" in note
    assert "audio stream" in note


def test_no_lead_is_nothing_to_report():
    assert audio_lead_note([_stream(1, 0.0)], 1) is None


def test_a_negligible_lead_is_not_worth_reporting():
    """Below 50 ms is encoder slop, not a sync issue."""
    assert audio_lead_note([_stream(1, 0.02)], 1) is None


def test_only_the_chosen_stream_is_reported():
    """A dub may carry a lead; the track we actually transcribe is the one that matters."""
    streams = [_stream(1, 0.0), _stream(2, 1.5)]
    assert audio_lead_note(streams, 1) is None
    assert "1.500" in (audio_lead_note(streams, 2) or "")


def test_any_stream_when_none_was_chosen():
    assert audio_lead_note([_stream(1, 1.008)], None) is not None


def test_the_parser_records_start_time(monkeypatch):
    """ffprobe's own JSON carries start_time; the parser must keep it, not drop it."""
    payload = json.dumps({"streams": [
        {"index": 1, "tags": {"language": "eng"}, "channels": 2,
         "disposition": {}, "start_time": "1.008000"},
        {"index": 2, "tags": {"language": "eng"}, "channels": 2, "disposition": {}},
    ]})

    class _Proc:
        returncode = 0
        stdout = payload

    monkeypatch.setattr("core.audio.shutil.which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr("core.audio.subprocess.run", lambda *a, **k: _Proc())
    streams = list_audio_streams("/tmp/x.mkv")
    assert streams[0]["start_time"] == 1.008
    assert streams[1]["start_time"] == 0.0, "a missing start_time is zero, not an error"


def test_the_parser_survives_a_junk_start_time(monkeypatch):
    payload = json.dumps({"streams": [
        {"index": 1, "tags": {}, "channels": 1, "disposition": {}, "start_time": "N/A"},
    ]})

    class _Proc:
        returncode = 0
        stdout = payload

    monkeypatch.setattr("core.audio.shutil.which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr("core.audio.subprocess.run", lambda *a, **k: _Proc())
    assert list_audio_streams("/tmp/x.mkv")[0]["start_time"] == 0.0
