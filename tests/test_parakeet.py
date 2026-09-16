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


@pytest.fixture(autouse=True)
def _no_audio_probe(runner, monkeypatch):
    """No ffprobe shell-out on fake media: a fixture file has no streams, and the
    track choice itself is covered in tests/test_audio_select.py."""
    monkeypatch.setattr(runner, "list_audio_streams", lambda *a, **k: [])


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
    assert "cannot translate" in out


def test_unsupported_language_rejected(runner, capsys, monkeypatch, tmp_path):
    """A language outside the variants' language sets (ja is not one of v3's 25)
    is refused with the list of languages v3 would add."""
    _arm_runner(runner, monkeypatch, tmp_path, variant="v2")  # English-only install
    monkeypatch.setattr(
        sys, "argv", ["runner", "m.mp4", "tiny", "ja", "transcribe"]
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "does not support 'ja'" in out
    assert "install-parakeet-model.sh" in out
    assert "es" in out  # v3's languages are listed


def test_v3_only_install_accepts_any_v3_language(runner, capsys, monkeypatch, tmp_path):
    """With v3 installed a non-English language runs; the status line names v3."""
    media = tmp_path / "clip.wav"
    media.write_bytes(b"x")
    _arm_runner(runner, monkeypatch, tmp_path, variant="v3")
    monkeypatch.setattr(sys, "argv", ["runner", str(media), "ignored", "fr", "transcribe"])
    runner.main()
    out = capsys.readouterr().out
    assert "does not support" not in out
    assert "parakeet-tdt-0.6b-v3" in out


def test_region_code_is_normalized(runner, capsys, monkeypatch, tmp_path):
    """'en-GB' used to be rejected (raw string compare against 'en')."""
    media = tmp_path / "clip.wav"
    media.write_bytes(b"x")
    _arm_runner(runner, monkeypatch, tmp_path, variant="v2")
    monkeypatch.setattr(sys, "argv", ["runner", str(media), "ignored", "en-GB", "transcribe"])
    runner.main()
    assert "does not support" not in capsys.readouterr().out


def test_missing_media_emits_error(runner, capsys, monkeypatch, tmp_path):
    _arm_runner(runner, monkeypatch, tmp_path, variant="v2")
    monkeypatch.setattr(
        sys, "argv", ["runner", "/nonexistent/file.mp4", "tiny", "en", "transcribe"]
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert "File not found" in capsys.readouterr().out


def test_missing_model_emits_install_hint(runner, capsys, monkeypatch, tmp_path):
    """Media exists but no model is installed → actionable error."""
    media = tmp_path / "fake.mp4"
    media.write_bytes(b"junk")
    monkeypatch.setenv("VSCL_AISUBS_PARAKEET_MODEL", str(tmp_path / "no-model-dir"))
    monkeypatch.setattr(
        sys, "argv", ["runner", str(media), "tiny", "en", "transcribe"]
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert "install-parakeet-model.sh" in capsys.readouterr().out


def test_forced_version_that_is_not_installed_says_so(runner, capsys, monkeypatch, tmp_path):
    _arm_runner(runner, monkeypatch, tmp_path, variant="v2")
    monkeypatch.setenv("VSCL_AISUBS_PARAKEET_VERSION", "v3")
    monkeypatch.setattr(
        sys, "argv", ["runner", "m.mp4", "tiny", "en", "transcribe"]
    )
    with pytest.raises(SystemExit):
        runner.main()
    out = capsys.readouterr().out
    assert "VSCL_AISUBS_PARAKEET_VERSION=v3" in out
    assert "install-parakeet-model.sh" in out


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


def test_backend_model_label_tracks_the_runner(runner, monkeypatch, tmp_path):
    """The CLI prints backends.parakeet.model_label(); the runner loads the same
    variant — a status line must never name a model that is not the one running."""
    from backends.parakeet import ParakeetBackend
    import core.parakeet_models as pm

    monkeypatch.delenv("VSCL_AISUBS_PARAKEET_VERSION", raising=False)
    for variant, expected in (("v2", "parakeet-tdt-0.6b-v2"),
                              ("v3", "parakeet-tdt-0.6b-v3")):
        monkeypatch.setenv("VSCL_AISUBS_PARAKEET_MODEL", str(_model_dir(tmp_path, variant)))
        assert pm.model_label() == expected
        assert ParakeetBackend().model_label("large") == expected


# ── chunking: memory-bounded size + seam handling ─────────────────────────

def test_chunk_seconds_default_and_override(runner, monkeypatch):
    """Default is the measured-safe 30 s; the env override is range-checked, so
    a junk or dangerous value keeps the default instead of OOMing again."""
    monkeypatch.delenv("VSCL_AISUBS_PARAKEET_CHUNK", raising=False)
    assert runner.CHUNK_SECONDS == 30
    assert runner.resolve_chunk_seconds() == 30
    for raw, want in (("60", 60), ("45.5", 45), ("5", 5), ("600", 600)):
        monkeypatch.setenv("VSCL_AISUBS_PARAKEET_CHUNK", raw)
        assert runner.resolve_chunk_seconds() == want, raw
    for bad in ("0", "4", "601", "-30", "hello", "", "  "):
        monkeypatch.setenv("VSCL_AISUBS_PARAKEET_CHUNK", bad)
        assert runner.resolve_chunk_seconds() == 30, bad


def test_keep_nominal_window_gives_each_word_one_owner(runner):
    """Words in the overlap belong to the chunk whose window starts them; a word
    straddling the seam stays with the chunk that opened it instead of being
    emitted twice (or garbled on both sides). Words are (text, start, end)."""
    words = [
        ("left", 9.0, 9.4), ("seam", 9.9, 10.3), ("right", 10.0, 10.4),
        ("tail", 29.9, 30.2), ("next", 30.1, 30.5),
    ]
    assert [w[0] for w in runner.keep_nominal_window(words, 10.0, 30.0)] == ["right", "tail"]
    assert [w[0] for w in runner.keep_nominal_window(words, 30.0, 60.0)] == ["next"]


def test_multi_chunk_run_keeps_every_chunk(runner, monkeypatch, tmp_path, capsys):
    """A 65 s clip at 30 s chunks = 3 chunks, and the words from each one must
    survive the seam filter (the fake model returns 3 words per chunk)."""
    media = tmp_path / "clip.wav"
    media.write_bytes(b"x")
    _arm_runner(runner, monkeypatch, tmp_path, variant="v2")
    monkeypatch.setattr(runner, "load_float32_16k", lambda _p: [0.0] * (65 * 16000))
    monkeypatch.setattr(sys, "argv", ["runner", str(media), "ignored", "en", "transcribe"])
    runner.main()
    out = capsys.readouterr().out
    assert "chunk 1/3" in out and "chunk 2/3" in out and "chunk 3/3" in out
    assert out.count('"type": "sub"') == 3        # one cue per chunk, none lost
    assert '"type": "done"' in out


def test_short_media_stays_single_chunk(runner, monkeypatch, tmp_path, capsys):
    """No chunk-per-chunk status noise for ordinary clips."""
    media = tmp_path / "clip.wav"
    media.write_bytes(b"x")
    _arm_runner(runner, monkeypatch, tmp_path, variant="v2")
    monkeypatch.setattr(sys, "argv", ["runner", str(media), "ignored", "en", "transcribe"])
    runner.main()
    assert "decoding chunk" not in capsys.readouterr().out


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


def _model_dir(tmp_path, variant="v2"):
    """A directory that looks like an installed Parakeet variant (v2/v3)."""
    name = {
        "v2": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        "v3": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    }[variant]
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    for f in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (d / f).write_text("")
    return d


def _arm_runner(runner, monkeypatch, tmp_path, variant="v2"):
    """Model files present; ffmpeg decode, numpy load and sherpa-onnx all faked.

    The model directory is handed over through VSCL_AISUBS_PARAKEET_MODEL, so
    the runner's real variant selection runs (no MODEL_* monkeypatching).

    load_float32_16k() is the only numpy user in the runner (the dev test venv
    intentionally has no numpy), and the runner treats the samples as a plain
    sequence, so a list of floats stands in for the array.
    """
    monkeypatch.setenv("VSCL_AISUBS_PARAKEET_MODEL", str(_model_dir(tmp_path, variant)))
    monkeypatch.delenv("VSCL_AISUBS_PARAKEET_VERSION", raising=False)
    monkeypatch.setitem(
        sys.modules, "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=_FakeOfflineRecognizer),
    )
    monkeypatch.setattr(runner, "load_float32_16k", lambda _p: [0.0] * 16000)
    counter = [0]

    def fake_decode(_media, timeout=None, stream_index=None):
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
