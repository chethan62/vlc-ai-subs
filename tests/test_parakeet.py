"""Unit tests for parakeet_runner.py (loaded standalone via importlib).

Covers the pure logic (BPE-token → word merge, word → segment grouping,
SRT timestamps) and the CLI error contract — all without importing
sherpa-onnx, which the runner only loads inside main() after validation.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_RUNNER = str(Path(__file__).resolve().parent.parent / "parakeet_runner.py")


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("parakeet_runner_test", _RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (1.9999, "00:00:02,000"),      # rounds to nearest ms
        (61.5, "00:01:01,500"),
        (3661.789, "01:01:01,789"),
        (3599.999, "00:59:59,999"),    # float-drift safe
    ],
)
def test_format_srt_timestamp(runner, seconds: float, expected: str):
    assert runner.format_srt_timestamp(seconds) == expected


def test_tokens_to_words(runner):
    """BPE tokens with per-token timestamps merge into real words."""
    tokens = [" Well", ",", " I", " don", "'", "t", " w", "ish", " to", " go", "."]
    times = [1.0, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    words = runner.tokens_to_words(tokens, times)
    assert [w[0] for w in words] == ["Well,", "I", "don't", "wish", "to", "go."]
    assert words[0][1] == 1.0
    assert words[-1][2] == 1.8


def test_words_to_segments_sentence_split(runner):
    words = [
        ("Hello", 0.0, 0.4), ("world", 0.4, 0.9), ("this", 0.9, 1.2),
        ("is", 1.2, 1.5), ("a", 1.5, 1.7), ("test.", 1.7, 2.2),
        ("Next", 2.2, 2.5), ("sentence.", 2.5, 3.0),
    ]
    segs = runner.words_to_segments(words)
    assert len(segs) == 2
    assert segs[0]["text"] == "Hello world this is a test."
    assert segs[0]["end"] == 2.2
    assert segs[1]["start"] == 2.2
    assert segs[1]["text"] == "Next sentence."


def test_usage_error_emits_error_jsonl(runner, capsys, monkeypatch):
    # Deterministic argv — must not depend on how pytest was invoked (pytest's
    # own argv can be ≥5 entries, which would skip the usage branch).
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert '"type": "error"' in capsys.readouterr().out


def test_translate_task_rejected(runner, capsys, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["runner", "m.mp4", "tiny", "en", "translate", "mirror", "x.srt"]
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Parakeet is English-only" in out


def test_non_english_language_rejected(runner, capsys, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["runner", "m.mp4", "tiny", "fr", "transcribe"]
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert "supports English only" in capsys.readouterr().out


def test_missing_media_emits_error(runner, capsys, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["runner", "/nonexistent/file.mp4", "tiny", "en", "transcribe"]
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert "File not found" in capsys.readouterr().out


def test_missing_model_emits_install_hint(runner, capsys, monkeypatch, tmp_path):
    """Media exists but the model is not installed → actionable error."""
    media = tmp_path / "fake.mp4"
    media.write_bytes(b"junk")
    monkeypatch.setattr(runner, "MODEL_DIR", str(tmp_path / "no-model-dir"))
    monkeypatch.setattr(
        sys, "argv", ["runner", str(media), "tiny", "en", "transcribe"]
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert "install-parakeet-model.sh" in capsys.readouterr().out


# ── write_srt_if_requested ────────────────────────────────────────────

def test_write_srt_if_requested_writes(runner, tmp_path):
    srt = tmp_path / "out.srt"
    lines = ["1\n00:00:00,400 --> 00:00:03,520\nHi\n"]
    assert runner.write_srt_if_requested(lines, str(srt)) == str(srt)
    assert srt.read_text() == lines[0]


def test_write_srt_if_requested_skips_when_not_requested(runner, tmp_path):
    media = tmp_path / "m.mp4"
    media.write_bytes(b"x")
    assert runner.write_srt_if_requested(["1\nx\n"], None) is None
    assert not (tmp_path / "m.srt").exists()  # no side-effect file next to media


def test_write_srt_if_requested_skips_empty(runner, tmp_path):
    srt = tmp_path / "out.srt"
    assert runner.write_srt_if_requested([], str(srt)) is None
    assert not srt.exists()  # no 0-byte SRTs


# ── long-media chunking ───────────────────────────────────────────────

def test_chunk_plan_splits_by_chunk_size(runner):
    assert runner.chunk_plan(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert runner.chunk_plan(8, 4) == [(0, 4), (4, 8)]
    assert runner.chunk_plan(0, 4) == []
    # 21 minutes of 16k audio with 20-min chunks → two chunks
    assert len(runner.chunk_plan(21 * 60 * 16000, 20 * 60 * 16000)) == 2


def test_shift_words_offsets_timestamps(runner):
    words = [("hi", 0.5, 0.9), ("there", 1.0, 1.4)]
    assert runner.shift_words(words, 1200.0) == [
        ("hi", 1200.5, 1200.9), ("there", 1201.0, 1201.4),
    ]
    assert runner.shift_words([], 5.0) == []