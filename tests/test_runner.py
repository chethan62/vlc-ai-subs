"""Unit tests for whisperx_runner.py (loaded as a standalone module).

The runner lives outside a package, so tests import it by path via
importlib.util — mirroring how the backend spawns it as a subprocess.
"""

import importlib.util
import os
import sys

import pytest

_RUNNER = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "whisperx_runner.py")


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("whisperx_runner_test", _RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_usage_error_emits_error_jsonl(runner, capsys, monkeypatch):
    # Deterministic argv — the test must not depend on how pytest was invoked
    # (pytest's own argv can be ≥5 entries, which would skip the usage branch).
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert '"type": "error"' in out
    assert '"msg": "Usage: runner' in out


def test_missing_media_emits_error_jsonl(runner, capsys, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["runner", "/nonexistent/file.mp4", "tiny", "en", "transcribe"]
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert '"type": "error"' in out
    assert "File not found" in out


# ── SRT writing via shared core.srt.write_srt (was runner.write_srt_if_requested) ──
# write_srt_if_requested lived in both runners and duplicated core/srt.py; it was
# removed in favor of core.srt.write_srt, which keeps the same explicit-path /
# empty-drop / symlink-refusal semantics. These tests pin that shared behavior.

def test_write_srt_writes_when_requested(tmp_path):
    from core.srt import write_srt
    srt = tmp_path / "out.srt"
    segs = [{"start": 0.0, "end": 1.0, "text": "Hi"}]
    assert write_srt(segs, "/unused/movie.mp4", str(srt)) == str(srt)
    assert "Hi" in srt.read_text()


def test_write_srt_skips_when_not_requested(tmp_path):
    from core.srt import write_srt
    media = tmp_path / "m.mp4"
    media.write_bytes(b"x")
    # No explicit path + no segments would derive <media>.srt; with no
    # segments nothing is written (no 0-byte SRTs next to media).
    assert write_srt([], str(media)) is None
    assert not (tmp_path / "m.srt").exists()


def test_write_srt_refuses_symlink(tmp_path):
    from core.srt import write_srt
    target = tmp_path / "victim.txt"
    target.write_text("do not clobber")
    link = tmp_path / "out.srt"
    link.symlink_to(target)
    path = write_srt([{"start": 0.0, "end": 1.0, "text": "line"}], "m.mp4", str(link))
    assert path is not None and path != str(link)
    assert target.read_text() == "do not clobber"  # symlink target untouched
    assert os.path.isfile(path)


# ── device / compute resolution (VSCL_AISUBS_DEVICE / VSCL_AISUBS_COMPUTE) ──

def test_resolve_device_defaults(runner, monkeypatch):
    monkeypatch.delenv("VSCL_AISUBS_DEVICE", raising=False)
    assert runner.resolve_device(True) == "cuda"
    assert runner.resolve_device(False) == "cpu"


def test_resolve_device_env_override(runner, monkeypatch):
    monkeypatch.setenv("VSCL_AISUBS_DEVICE", "cpu")
    assert runner.resolve_device(True) == "cpu"
    monkeypatch.setenv("VSCL_AISUBS_DEVICE", "cuda")
    assert runner.resolve_device(False) == "cuda"


def test_resolve_device_invalid_falls_back(runner, monkeypatch):
    monkeypatch.setenv("VSCL_AISUBS_DEVICE", "bogus")
    assert runner.resolve_device(False) == "cpu"


def test_resolve_compute_defaults_and_override(runner, monkeypatch):
    monkeypatch.delenv("VSCL_AISUBS_COMPUTE", raising=False)
    assert runner.resolve_compute("cuda") == "int8_float16"
    assert runner.resolve_compute("cpu") == "float32"
    monkeypatch.setenv("VSCL_AISUBS_COMPUTE", "int8")
    assert runner.resolve_compute("cuda") == "int8"
    assert runner.resolve_compute("cpu") == "int8"
    # CUDA-only compute types must be rejected on CPU (they crash faster-whisper)
    monkeypatch.setenv("VSCL_AISUBS_COMPUTE", "int8_float16")
    assert runner.resolve_compute("cpu") == "float32"
    assert runner.resolve_compute("cuda") == "int8_float16"
    monkeypatch.setenv("VSCL_AISUBS_COMPUTE", "bogus")
    assert runner.resolve_compute("cuda") == "int8_float16"


def test_hardened_asr_options(runner):
    """Research-backed decode hardening (§2.2) must stay in force."""
    opts = runner.hardened_asr_options()
    assert opts["beam_size"] == 1
    assert opts["condition_on_previous_text"] is False
    assert opts["temperatures"] == [0.0]
    assert opts["hallucination_silence_threshold"] == 2.0
    assert opts["no_speech_threshold"] >= 0.6


# ── SRT side effects: write only when the caller asked for a path ──────────
# Regression: the backends never forward argv[5]/argv[6], so a runner that
# derives <media>.srt drops a side-effect file next to the media — the same
# file realtime-OSD runs deliberately write to a temp path instead (and a
# read-only media dir would make it an error).

@pytest.fixture(autouse=True)
def _fake_audio_selection(runner, monkeypatch, tmp_path):
    """Keep these tests off the real media pipeline.

    The runner now asks ffprobe which audio track to transcribe (and decodes it
    itself, instead of letting WhisperX read the media twice). A test's fake media
    has no streams, so probe would shell out and return nothing on every run.
    Stub the seam with the REAL signature: a friendlier fake once hid a real
    TypeError for a whole release. test_audio_select.py owns this behaviour.
    """
    monkeypatch.setattr(runner, "list_audio_streams", lambda *a, **k: [])
    monkeypatch.setattr(
        runner, "decode_to_wav16k",
        lambda _p, timeout=None, stream_index=None: str(tmp_path / "decoded.wav"),
    )
    monkeypatch.setattr(runner, "_load_waveform", lambda _p: [0.0] * 16000)


class _FakeWxModel:
    def transcribe(self, path, language=None, task=None):
        return {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
        }


def _fake_whisperx(monkeypatch):
    import types

    # SimpleNamespace (not ModuleType) so attribute assignment is well-typed.
    mod = types.SimpleNamespace(load_model=lambda *a, **k: _FakeWxModel())
    monkeypatch.setitem(sys.modules, "whisperx", mod)


def test_runner_writes_no_srt_next_to_media(runner, monkeypatch, tmp_path, capsys):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    _fake_whisperx(monkeypatch)

    monkeypatch.setattr(sys, "argv", ["runner", str(media), "tiny", "en", "transcribe"])
    runner.main()

    assert not (tmp_path / "clip.srt").exists()
    out = capsys.readouterr().out
    assert '"type": "done"' in out
    assert '"srt_path": null' in out


def test_runner_writes_srt_when_path_given(runner, monkeypatch, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    wanted = tmp_path / "wanted.srt"
    _fake_whisperx(monkeypatch)

    monkeypatch.setattr(
        sys, "argv",
        ["runner", str(media), "tiny", "en", "transcribe", "mirror.txt", str(wanted)],
    )
    runner.main()

    assert wanted.is_file()
    assert "hello" in wanted.read_text(encoding="utf-8")
    assert not (tmp_path / "clip.srt").exists()


# ── alignment: the word timings must actually reach the cue times ──────────
# Regression: whisperx.align() ran on CUDA and its output was thrown away —
# only result["segments"] (segment-level times) was used, so the plugin paid a
# GPU pass plus a wav2vec2 download for a feature the README advertises as
# "word-level timing". The align branch was also unreachable in tests, because
# the dev venv has no torch and device therefore always resolved to "cpu".

def _fake_cuda(monkeypatch, calls=None, align_raises=None, lang="en"):
    """Fake whisperx + torch so the CUDA align path is exercised."""
    import types

    calls = calls if calls is not None else []

    def _load_model(*a, **k):
        class _M:
            def transcribe(self, path, language=None, task=None):
                return {"language": lang, "segments": [{"start": 0.0, "end": 3.0, "text": "hello there"}]}
        return _M()

    def _load_align_model(language_code=None, device=None):
        calls.append(("load_align_model", language_code))
        return ("align-model", {"lang": language_code})

    def _align(transcript, model, align_model_metadata, audio, device,
               interpolate_method="nearest", return_char_alignments=False,
               print_progress=False, combined_progress=False, progress_callback=None):
        # Signature mirrors whisperx.align() exactly (verified against the
        # installed venv). A stub that accepts **kwargs hides real breakage:
        # passing an unsupported batch_size=1 used to sail through this test
        # while the real runner silently skipped alignment and emitted
        # segment-level cues.
        calls.append(("align", return_char_alignments))
        if align_raises is not None:
            raise align_raises
        return {"segments": [{
            "start": 0.45, "end": 2.90, "text": "hello there",   # align's tightened span
            "words": [
                {"word": "hello", "start": 0.45, "end": 1.10},
                {"word": "there", "start": 1.15, "end": 2.30},
                {"word": ",", "start": None, "end": None},  # punctuation has no timing
            ],
        }]}

    monkeypatch.setitem(sys.modules, "whisperx", types.SimpleNamespace(
        load_model=_load_model, load_align_model=_load_align_model, align=_align,
    ))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
    ))
    return calls


def test_cuda_run_aligns_with_one_segment_per_pass_and_uses_the_words(
    runner, monkeypatch, tmp_path
):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    wanted = tmp_path / "out.srt"
    calls = _fake_cuda(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["runner", str(media), "turbo", "en", "transcribe", "m.txt", str(wanted)])

    runner.main()

    assert ("load_align_model", "en") in calls, "the align model must match the transcript language"
    assert any(c[0] == "align" for c in calls), "the alignment must actually run"
    srt = wanted.read_text(encoding="utf-8")
    # align's own span (0.45-2.90), NOT the transcribed segment's (0.0-3.0):
    # the alignment output is what the cue times come from.
    assert "00:00:00,450 --> 00:00:02,900" in srt, srt


def test_translated_run_skips_alignment(runner, monkeypatch, tmp_path, capsys):
    """Aligning translated text against foreign audio is meaningless (the align
    model maps sounds to text) — it must not load a model or burn GPU time."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    calls = _fake_cuda(monkeypatch, lang="de")
    # nllb_translate is imported inside main(), so it is a local name there — the
    # fake has to go into sys.modules for the import to pick it up.
    import types
    monkeypatch.setitem(sys.modules, "nllb_translate", types.SimpleNamespace(
        should_cascade=lambda task, env: False,
    ))
    monkeypatch.setattr(sys, "argv", ["runner", str(media), "turbo", "de", "translate", "m.txt", str(tmp_path / "o.srt")])

    runner.main()

    assert calls == [], f"no alignment for a translated run: {calls}"
    assert "Word alignment" not in capsys.readouterr().out


def test_alignment_oom_is_reported_as_such_and_the_run_still_finishes(
    runner, monkeypatch, tmp_path, capsys
):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    _fake_cuda(monkeypatch, align_raises=RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    monkeypatch.setattr(sys, "argv", ["runner", str(media), "turbo", "en", "transcribe"])

    runner.main()

    out = capsys.readouterr().out
    assert "GPU out of memory" in out
    assert "segment-level" in out          # says what the fallback means
    assert "language model" not in out     # the old, wrong diagnosis
    assert '"type": "done"' in out         # a skipped alignment is not a failure


def capsys_out(_monkeypatch):
    return ""
