"""Tests for backend resolution.

- WhisperX is the DEFAULT engine (VSCL_AISUBS_BACKEND unset or legacy value).
- `VSCL_AISUBS_BACKEND=parakeet` opts into the Parakeet backend (English).
- `VSCL_AISUBS_BACKEND=whispercpp` (alias: whisper_cpp) selects whisper.cpp,
  the Vulkan engine that accelerates AMD/Intel GPUs.
- `auto` (and unset) applies the hardware policy: NVIDIA → WhisperX, else a
  Vulkan whisper.cpp, else WhisperX on CPU.
"""

import pytest

import backends
from backends import resolve_backend


def _fake_whisperx_backend(monkeypatch, tmp_path):
    """Point the whisperx backend at fake-but-existing files so the resolution
    tests are hermetic (they must pass on a fresh machine with no runtime dir,
    e.g. CI)."""
    import backends.whisperx_backend as wx

    runner = tmp_path / "whisperx_runner.py"
    runner.write_text("")
    venv = tmp_path / "venv-whisperx"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin" / "python3"
    py.write_text("")
    monkeypatch.setattr(wx, "_RUNNER", str(runner))
    monkeypatch.setattr(wx, "_VENV", str(venv))
    monkeypatch.setattr(wx, "_PYTHON", str(py))
    return wx


def test_resolve_returns_whisperx_by_default(monkeypatch, tmp_path):
    _fake_whisperx_backend(monkeypatch, tmp_path)
    be = resolve_backend()
    assert be is not None
    assert "whisperx" in be.name().lower()


def test_legacy_env_values_are_ignored(monkeypatch, tmp_path):
    """moonshine/whisper_cpp-era values must fall through to WhisperX (the
    whisper_cpp alias is covered below)."""
    _fake_whisperx_backend(monkeypatch, tmp_path)
    monkeypatch.setattr(backends, "nvidia_gpu", lambda: None)
    monkeypatch.setenv("VSCL_AISUBS_BACKEND", "moonshine")
    be = resolve_backend()
    assert "whisperx" in be.name().lower()


# ── whisper.cpp (Vulkan: AMD/Intel/NVIDIA) ─────────────────────────────

def _fake_whispercpp(monkeypatch, vulkan=True):
    import backends.whispercpp as wc

    # Plain lambda (not classmethod): detect() is called on the class, so it
    # receives no args, and pyright can see the real constructor's signature.
    monkeypatch.setattr(
        wc.WhisperCppBackend, "detect",
        lambda: wc.WhisperCppBackend(binary="/bin/true", vulkan=vulkan),
    )
    return wc


def test_whispercpp_env_selects_the_vulkan_engine(monkeypatch):
    _fake_whispercpp(monkeypatch)
    monkeypatch.setenv("VSCL_AISUBS_BACKEND", "whispercpp")
    be = resolve_backend()
    assert "whisper.cpp" in be.name() and "vulkan" in be.name()


def test_whisper_cpp_alias_selects_the_vulkan_engine(monkeypatch):
    """The pre-fork env value should reach the new engine, not be ignored."""
    _fake_whispercpp(monkeypatch, vulkan=False)
    monkeypatch.setenv("VSCL_AISUBS_BACKEND", "whisper_cpp")
    be = resolve_backend()
    assert "whisper.cpp" in be.name() and "cpu" in be.name()


def test_whispercpp_missing_raises_install_hint(monkeypatch):
    import backends.whispercpp as wc

    monkeypatch.setenv("VSCL_AISUBS_BACKEND", "whispercpp")
    monkeypatch.setattr(wc.WhisperCppBackend, "detect", classmethod(lambda cls: None))
    with pytest.raises(RuntimeError) as exc:
        resolve_backend()
    msg = str(exc.value)
    assert "whisper.cpp backend is not available" in msg
    assert "install-whisper-cpp.sh" in msg


# ── hardware policy ("auto" / unset) ───────────────────────────────────

def test_auto_prefers_whisperx_on_nvidia(monkeypatch, tmp_path):
    """faster-whisper's CUDA path beats Vulkan for quality — NVIDIA wins."""
    _fake_whisperx_backend(monkeypatch, tmp_path)
    _fake_whispercpp(monkeypatch)
    monkeypatch.setattr(backends, "nvidia_gpu", lambda: "GeForce GTX 1650")
    monkeypatch.delenv("VSCL_AISUBS_BACKEND", raising=False)
    assert "whisperx" in resolve_backend().name().lower()


def test_auto_uses_whispercpp_on_a_vulkan_only_machine(monkeypatch):
    """No NVIDIA + a Vulkan driver (AMD/Intel) → the Vulkan engine."""
    _fake_whispercpp(monkeypatch)
    monkeypatch.setattr(backends, "nvidia_gpu", lambda: None)
    monkeypatch.setattr(backends, "vulkan_icds", lambda: ["/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"])
    monkeypatch.setattr(backends, "vulkan_gpu_driver", lambda: "AMD")
    monkeypatch.delenv("VSCL_AISUBS_BACKEND", raising=False)
    be = resolve_backend()
    assert "whisper.cpp" in be.name() and "vulkan" in be.name()


def test_auto_uses_whisperx_when_no_gpu_at_all(monkeypatch, tmp_path):
    _fake_whisperx_backend(monkeypatch, tmp_path)
    _fake_whispercpp(monkeypatch)
    monkeypatch.setattr(backends, "nvidia_gpu", lambda: None)
    monkeypatch.setattr(backends, "vulkan_icds", lambda: [])
    monkeypatch.delenv("VSCL_AISUBS_BACKEND", raising=False)
    assert "whisperx" in resolve_backend().name().lower()


def test_auto_falls_back_when_vulkan_exists_without_whispercpp(monkeypatch, tmp_path):
    """A Vulkan driver alone is not enough — no engine installed → WhisperX."""
    import backends.whispercpp as wc

    _fake_whisperx_backend(monkeypatch, tmp_path)
    monkeypatch.setattr(backends, "nvidia_gpu", lambda: None)
    monkeypatch.setattr(backends, "vulkan_icds", lambda: ["intel_icd.x86_64.json"])
    monkeypatch.setattr(wc.WhisperCppBackend, "detect", classmethod(lambda cls: None))
    monkeypatch.delenv("VSCL_AISUBS_BACKEND", raising=False)
    assert "whisperx" in resolve_backend().name().lower()


def test_parakeet_env_returns_parakeet_backend(monkeypatch):
    """The opt-in path; if sherpa-onnx is installed it resolves to Parakeet."""
    monkeypatch.setenv("VSCL_AISUBS_BACKEND", "parakeet")
    be = None
    try:
        be = resolve_backend()
    except RuntimeError:
        pytest.skip("Parakeet backend not installed on this machine")
    assert be is not None
    assert "parakeet" in be.name().lower()


def test_parakeet_missing_raises_helpful_error(monkeypatch):
    """VSCL_AISUBS_BACKEND=parakeet without the backend → actionable error."""
    import backends.parakeet as pk

    monkeypatch.setenv("VSCL_AISUBS_BACKEND", "parakeet")
    monkeypatch.setattr(pk.ParakeetBackend, "detect", classmethod(lambda cls: None))
    with pytest.raises(RuntimeError) as exc:
        resolve_backend()
    msg = str(exc.value)
    assert "Parakeet backend is not available" in msg
    assert "install-parakeet-model.sh" in msg  # points at the fix


def test_missing_whisperx_raises_helpful_error(monkeypatch):
    import backends.whisperx_backend as wx

    monkeypatch.setattr(wx.WhisperXBackend, "detect", classmethod(lambda cls: None))
    with pytest.raises(RuntimeError) as exc:
        resolve_backend()
    msg = str(exc.value)
    assert "WhisperX backend is not available" in msg
    assert "venv-whisperx" in msg  # points at the fix


def test_detect_false_when_venv_missing(monkeypatch, tmp_path):
    """detect() returns None when the 3.12 venv/runner are not installed."""
    import backends.whisperx_backend as wx

    monkeypatch.setattr(wx, "_RUNNER", str(tmp_path / "nope" / "whisperx_runner.py"))
    monkeypatch.setattr(wx, "_VENV", str(tmp_path / "nope" / "venv-whisperx"))
    monkeypatch.setattr(wx, "_PYTHON", None)
    assert wx.WhisperXBackend.detect() is None


def test_runner_and_venv_required_for_detect(monkeypatch, tmp_path):
    """detect() must require BOTH the runner script and the venv python."""
    import backends.whisperx_backend as wx

    venv = tmp_path / "venv-whisperx"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin" / "python3"
    py.write_text("#!/bin/sh\necho fake\n")  # any file is enough for isfile()

    monkeypatch.setattr(wx, "_VENV", str(venv))
    monkeypatch.setattr(wx, "_RUNNER", str(tmp_path / "missing" / "runner.py"))
    monkeypatch.setattr(wx, "_PYTHON", str(py))

    assert wx.WhisperXBackend.detect() is None