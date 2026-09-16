"""The pure half of the VAD module: gating arithmetic and model discovery.

The detector itself needs numpy, sherpa-onnx and the 630 KB model, so it is verified
against real media by hand. Those numbers are recorded in
`.research/2026-09-16-github-survey.md`: 53.7 s of speech in a 60 s dialogue minute, none
across a known 5.6 s music gap, none in 60 s of credits. What is covered here is
everything around it — a gating bug would silently skip a chunk full of dialogue, which is
the one failure mode this design must not have.
"""

import pytest

from core.vad import holds_speech, resolve_vad_model, speech_extent


def test_holds_speech_detects_overlap_not_containment():
    speech = [(2.0, 4.0)]
    assert holds_speech(speech, 1.0, 3.0), "a chunk overlapping speech counts"
    assert holds_speech(speech, 3.5, 9.0)
    assert not holds_speech(speech, 4.0, 6.0), "touching the end is not overlapping"
    assert not holds_speech(speech, 0.0, 2.0)


def test_a_chunk_with_no_speech_is_skippable():
    """This is what lets a music chunk avoid a decode entirely."""
    speech = [(10.0, 12.0), (40.0, 41.0)]
    assert not holds_speech(speech, 12.0, 40.0), "12s-40s holds no speech at all"
    assert holds_speech(speech, 9.5, 10.5)


def test_a_chunk_holding_any_speech_is_never_skipped():
    """The failure this gating must not have: skipping dialogue. A single overlapping
    speech span, however short, keeps the chunk."""
    speech = [(30.05, 30.20), (90.0, 95.0)]
    assert holds_speech(speech, 30.0, 60.0), "0.15s of speech in the chunk is enough"
    assert holds_speech(speech, 60.0, 90.02)


def test_no_vad_result_means_nothing_is_skipped():
    """With no spans (VAD absent or silent failure) every chunk is decoded, as before."""
    assert not holds_speech([], 0.0, 30.0)


def test_speech_extent_counts_only_voiced_time():
    words = [("a", 0.0, 0.5), ("b", 2.0, 2.5)]
    assert speech_extent(words) == pytest.approx(1.0), "the 1.5s gap is not speech"
    assert speech_extent([]) == 0.0


def test_model_discovery_prefers_the_environment(tmp_path, monkeypatch):
    model = tmp_path / "silero_vad.onnx"
    model.write_bytes(b"not really a model")
    monkeypatch.setenv("VSCL_AISUBS_VAD_MODEL", str(model))
    assert resolve_vad_model() == str(model)


def test_a_missing_model_is_not_an_error(monkeypatch):
    """Callers treat None as 'continue without VAD' — the plugin still works.

    Patched rather than relying on the machine's state: once the model is installed at a
    standard path this test would otherwise find it and pass for the wrong reason.
    """
    monkeypatch.setenv("VSCL_AISUBS_VAD_MODEL", "/nonexistent/silero_vad.onnx")
    monkeypatch.setattr("core.vad.os.path.isfile", lambda path: False)
    assert resolve_vad_model() is None
