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
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (1.0, "00:00:01,000"),
        (61.5, "00:01:01,500"),
        (3661.789, "01:01:01,789"),
        (3599.999, "00:59:59,999"),         # float-drift safe
        (86399.001, "23:59:59,001"),
    ],
)
def test_format_srt_timestamp(runner, seconds: float, expected: str):
    assert runner.format_srt_timestamp(seconds) == expected


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


# ── write_srt_if_requested ────────────────────────────────────────────

def test_write_srt_if_requested_writes(runner, tmp_path):
    srt = tmp_path / "out.srt"
    lines = ["1\n00:00:00,000 --> 00:00:01,000\nHi\n"]
    assert runner.write_srt_if_requested(lines, str(srt)) == str(srt)
    assert srt.read_text() == lines[0]


def test_write_srt_if_requested_skips_when_not_requested(runner, tmp_path):
    media = tmp_path / "m.mp4"
    media.write_bytes(b"x")
    assert runner.write_srt_if_requested(["1\nx\n"], None) is None
    assert not (tmp_path / "m.srt").exists()  # no side-effect file next to media


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


def test_write_srt_if_requested_skips_empty(runner, tmp_path):
    srt = tmp_path / "out.srt"
    assert runner.write_srt_if_requested([], str(srt)) is None
    assert not srt.exists()  # no 0-byte SRTs


def test_write_srt_if_requested_refuses_symlink(runner, tmp_path):
    target = tmp_path / "victim.txt"
    target.write_text("do not clobber")
    link = tmp_path / "out.srt"
    link.symlink_to(target)
    path = runner.write_srt_if_requested(["1\nline\n"], str(link))
    assert path is not None and path != str(link)
    assert target.read_text() == "do not clobber"  # symlink target untouched
    assert os.path.isfile(path)


def test_hardened_asr_options(runner):
    """Research-backed decode hardening (§2.2) must stay in force."""
    opts = runner.hardened_asr_options()
    assert opts["beam_size"] == 1
    assert opts["condition_on_previous_text"] is False
    assert opts["temperatures"] == [0.0]
    assert opts["hallucination_silence_threshold"] == 2.0
    assert opts["no_speech_threshold"] >= 0.6