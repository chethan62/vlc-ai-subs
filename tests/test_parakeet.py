"""Unit tests for parakeet_runner.py (loaded standalone via importlib).

Covers the pure logic (BPE-token → word merge, word → segment grouping,
SRT timestamps) and the CLI error contract — all without importing
sherpa-onnx, which the runner only loads inside main() after validation.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_RUNNER = str(Path(__file__).resolve().parent.parent / "parakeet_runner.py")


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("parakeet_runner_test", _RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_backend_model_name_tracks_the_runner(runner):
    """The CLI reports backends.parakeet.MODEL_NAME; it must match what the
    runner actually loads (single fixed model, no drift)."""
    from backends.parakeet import MODEL_NAME

    assert MODEL_NAME == runner.MODEL_NAME


# ── SRT side effects: write only when the caller asked for a path ──────────
# Same regression as the WhisperX runner: the backend never forwards
# argv[5]/argv[6], so a runner deriving <media>.srt would leave a file next to
# the media even when the caller asked for a temp path (realtime-OSD mode).

def _write_16k_wav(path, frames: int = 16000) -> str:
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * frames)
    return str(path)


class _FakeStream:
    def __init__(self) -> None:
        self.result = None

    def accept_waveform(self, rate, samples) -> None:
        self.samples = samples


class _FakeOfflineRecognizer:
    """Stands in for sherpa_onnx.OfflineRecognizer (fixed tokens, any input)."""

    @staticmethod
    def from_transducer(**kwargs):
        class _Rec:
            @staticmethod
            def create_stream():
                return _FakeStream()

            @staticmethod
            def decode_stream(stream):
                stream.result = SimpleNamespace(
                    tokens=_FAKE_TOKENS["tokens"], timestamps=_FAKE_TOKENS["times"]
                )

        return _Rec()


# Overwritten by the wiring tests that need a long sentence.
_FAKE_TOKENS = {
    "tokens": [" Hello", " world", "."],
    "times": [0.5, 1.0, 1.5],
}


def _model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (d / name).write_text("")
    return d


def _arm_runner(runner, monkeypatch, tmp_path):
    """Model files present; ffmpeg decode, numpy load and sherpa-onnx all faked.

    load_float32_16k() is the only numpy user in the runner (the dev test venv
    intentionally has no numpy), and the runner treats the samples as a plain
    sequence, so a list of floats stands in for the array.
    """
    monkeypatch.setattr(runner, "MODEL_DIR", str(_model_dir(tmp_path)))
    monkeypatch.setitem(
        sys.modules, "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=_FakeOfflineRecognizer),
    )
    monkeypatch.setattr(runner, "load_float32_16k", lambda _p: [0.0] * 16000)
    counter = [0]

    def fake_decode(_media):
        counter[0] += 1
        # A fresh file each call: the runner unlinks the decoded wav.
        return _write_16k_wav(tmp_path / f"decoded_{counter[0]}.wav")

    monkeypatch.setattr(runner, "decode_to_wav16k", fake_decode)


def test_runner_writes_no_srt_next_to_media(runner, monkeypatch, tmp_path, capsys):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"x")
    _arm_runner(runner, monkeypatch, tmp_path)

    monkeypatch.setattr(
        sys, "argv", ["runner", str(media), "ignored", "auto", "transcribe"]
    )
    runner.main()

    assert not (tmp_path / "clip.srt").exists()
    out = capsys.readouterr().out
    assert '"type": "done"' in out
    assert '"srt_path": null' in out


def test_runner_writes_srt_when_path_given(runner, monkeypatch, tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"x")
    wanted = tmp_path / "wanted.srt"
    _arm_runner(runner, monkeypatch, tmp_path)

    monkeypatch.setattr(
        sys, "argv",
        ["runner", str(media), "ignored", "auto", "transcribe", "mirror.txt", str(wanted)],
    )
    runner.main()

    assert wanted.is_file()
    assert "Hello world." in wanted.read_text(encoding="utf-8")
    assert not (tmp_path / "clip.srt").exists()


def test_runner_emits_wrapped_cues(runner, monkeypatch, tmp_path, capsys):
    """The cue-quality pass must run before emitting: SRT and OSD share the text
    (a refactor once dropped a runner's emission loop — pin the wiring)."""
    words = ("The quick brown fox jumps over the lazy dog while the camera keeps "
             "rolling on the empty street")
    tokens = [" " + w for w in words.split(" ")]
    times = [round(0.5 * i, 2) for i in range(1, len(tokens) + 1)]
    monkeypatch.setitem(_FAKE_TOKENS, "tokens", tokens)
    monkeypatch.setitem(_FAKE_TOKENS, "times", times)

    media = tmp_path / "clip.wav"
    media.write_bytes(b"x")
    wanted = tmp_path / "wrapped.srt"
    _arm_runner(runner, monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["runner", str(media), "ignored", "auto", "transcribe", "mirror.txt", str(wanted)],
    )
    runner.main()

    srt = wanted.read_text(encoding="utf-8")
    blocks = [b for b in srt.strip().split("\n\n") if b.strip()]
    # Block layout: index / timing / one or more text lines.
    texts = [" ".join(b.split("\n")[2:]) for b in blocks]
    assert " ".join(texts) == words                       # no text lost, in order
    assert any("\n" in b.split("\n", 2)[2].strip() for b in blocks)  # wrapped

    subs = [l for l in capsys.readouterr().out.splitlines() if '"type": "sub"' in l]
    assert subs and any("\\n" in s for s in subs)  # JSONL carries it too (OSD)
