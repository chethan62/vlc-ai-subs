"""Tests for backend resolution — WhisperX is the sole engine."""

import pytest

from backends import resolve_backend


def test_resolve_returns_whisperx_backend():
    be = resolve_backend()
    assert be is not None
    assert "whisperx" in be.name().lower()


def test_env_var_is_ignored(monkeypatch):
    """VSCL_AISUBS_BACKEND must have no effect — WhisperX only."""
    monkeypatch.setenv("VSCL_AISUBS_BACKEND", "whisper_cpp")
    be = resolve_backend()
    assert "whisperx" in be.name().lower()


def test_missing_whisperx_raises_helpful_error(monkeypatch):
    import backends.whisperx_backend as wx

    # Make detect() report "not installed" even though it may be on this box.
    monkeypatch.setattr(wx.WhisperXBackend, "detect", classmethod(lambda cls: None))
    with pytest.raises(RuntimeError) as exc:
        resolve_backend()
    msg = str(exc.value)
    assert "WhisperX is not available" in msg
    assert "venv-whisperx" in msg  # points at the fix


def test_detect_false_when_venv_missing(monkeypatch, tmp_path):
    """detect() returns None when the 3.12 venv/runner are not installed.

    Patch the module-level paths to a nonexistent location (isolated tmp).
    """
    import backends.whisperx_backend as wx

    monkeypatch.setattr(wx, "_RUNNER", str(tmp_path / "nope" / "whisperx_runner.py"))
    monkeypatch.setattr(wx, "_VENV", str(tmp_path / "nope" / "venv-whisperx"))
    monkeypatch.setattr(wx, "_PYTHON", None)
    assert wx.WhisperXBackend.detect() is None


def test_runner_and_venv_required_for_detect(monkeypatch, tmp_path):
    """detect() must require BOTH the runner script and the venv python."""
    import backends.whisperx_backend as wx

    # venv exists, runner missing
    venv = tmp_path / "venv-whisperx"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin" / "python3"
    py.write_text("#!/bin/sh\necho fake\n")  # any file is enough for isfile()

    monkeypatch.setattr(wx, "_VENV", str(venv))
    monkeypatch.setattr(wx, "_RUNNER", str(tmp_path / "missing" / "runner.py"))
    monkeypatch.setattr(wx, "_PYTHON", str(py))

    assert wx.WhisperXBackend.detect() is None