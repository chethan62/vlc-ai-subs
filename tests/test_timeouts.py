"""Subprocess timeout policy (core/timeouts.py) and its wiring into the backends.

Regression pin: a flat 20-minute ceiling (`1200`) failed every film longer than
~40 minutes at WhisperX's ~2x realtime — and this box's power-capped GPU is far
slower than that — i.e. exactly the long-media case the plugin exists for.
"""

import importlib
import subprocess

import pytest

from core.timeouts import DEFAULT_TIMEOUTS, resolve_timeout


def test_defaults_cover_feature_length_media(monkeypatch):
    monkeypatch.delenv("VSCL_AISUBS_TIMEOUT", raising=False)
    # A 2 h film at 2x realtime ≈ 1 h of compute, so the ceiling must clear
    # that comfortably; translate pays for the cascade on top of transcribing.
    assert resolve_timeout("transcribe") == DEFAULT_TIMEOUTS["transcribe"] == 4 * 3600
    assert DEFAULT_TIMEOUTS["translate"] > DEFAULT_TIMEOUTS["transcribe"]


def test_unknown_task_uses_conservative_default(monkeypatch):
    monkeypatch.delenv("VSCL_AISUBS_TIMEOUT", raising=False)
    assert resolve_timeout("a-task-invented-later") == DEFAULT_TIMEOUTS["transcribe"]


def test_env_override(monkeypatch):
    monkeypatch.setenv("VSCL_AISUBS_TIMEOUT", "90")
    assert resolve_timeout("transcribe") == 90.0


def test_zero_or_negative_means_no_limit(monkeypatch):
    monkeypatch.setenv("VSCL_AISUBS_TIMEOUT", "0")
    assert resolve_timeout("transcribe") is None
    monkeypatch.setenv("VSCL_AISUBS_TIMEOUT", "-5")
    assert resolve_timeout("transcribe") is None


def test_unparseable_value_keeps_the_ceiling(monkeypatch):
    """A typo must not silently disable the limit."""
    monkeypatch.setenv("VSCL_AISUBS_TIMEOUT", "half an hour")
    assert resolve_timeout("transcribe") == DEFAULT_TIMEOUTS["transcribe"]


def test_explicit_default_argument(monkeypatch):
    monkeypatch.delenv("VSCL_AISUBS_TIMEOUT", raising=False)
    assert resolve_timeout("transcribe", default=120) == 120
    monkeypatch.setenv("VSCL_AISUBS_TIMEOUT", "7")
    assert resolve_timeout("transcribe", default=120) == 7.0


def _timeout_passed_to_subprocess(monkeypatch, module):
    """Run a backend with its child runner stubbed; return the timeout it used.

    The backends call core.procs.run_captured (Popen + live-process registry so
    a cancel can stop the child) — stub that, not subprocess.run.
    """
    captured = {}

    def fake_run(cmd, timeout=None, env=None):
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(
            list(cmd), 1, '{"type": "error", "msg": "boom"}', "stderr tail"
        )

    monkeypatch.setattr(module, "run_captured", fake_run)
    backend = module.WhisperXBackend() if hasattr(module, "WhisperXBackend") else module.ParakeetBackend()
    with pytest.raises(RuntimeError):
        list(backend.transcribe("/tmp/whatever.mp4", "tiny", "en", "transcribe"))
    return captured["timeout"]


@pytest.mark.parametrize(
    "module_name",
    ["backends.whisperx_backend", "backends.parakeet"],
)
def test_backends_pass_the_resolved_timeout(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    monkeypatch.delenv("VSCL_AISUBS_TIMEOUT", raising=False)
    assert _timeout_passed_to_subprocess(monkeypatch, module) == DEFAULT_TIMEOUTS["transcribe"]
    monkeypatch.setenv("VSCL_AISUBS_TIMEOUT", "42")
    assert _timeout_passed_to_subprocess(monkeypatch, module) == 42.0
