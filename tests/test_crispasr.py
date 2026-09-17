"""CrispASR engine: model tiering, command building, SRT parsing.

No binary and no model download here — the *decisions* are what can be silently
wrong, and they are what decides which model a given machine runs. The measured
facts behind them are in `.research/2026-09-16-asr-landscape-and-crispasr.md`.
"""

import pytest

from core.crispasr_models import (MODELS, align_enabled, binary, chunk_seconds,
                                  model_argument, model_label, model_tag, pick)
from crispasr_runner import build_command, failure_reason, parse_srt


def test_a_signal_death_is_named_not_numbered():
    """A crash arrives as a negative return code with nothing on stderr.

    Measured 2026-09-17: on a 47.5-minute file v0.8.33 asked the kernel for a
    123.6 GB allocation, was refused, and dereferenced the NULL (SIGSEGV). The
    message the user sees must say so — "rc=-11" and an empty tail is
    indistinguishable from silence, and silence reads as success.
    """
    msg = failure_reason(-11, "")
    assert "SIGSEGV" in msg
    assert "rc=-11" not in msg
    assert "kernel log" in msg          # an empty tail must be explained, not shown blank


def test_a_signal_with_a_stderr_tail_keeps_the_tail():
    msg = failure_reason(-6, "Aborted (core dumped)\n")
    assert "SIGABRT" in msg
    assert "core dumped" in msg


def test_an_ordinary_failure_reports_its_code_and_output():
    msg = failure_reason(1, "model file missing")
    assert "rc=1" in msg and "model file missing" in msg


def test_the_stderr_tail_is_bounded():
    """The status line has one label to render into; 500 chars is already plenty."""
    assert len(failure_reason(1, "x" * 4000)) < 600


def test_a_long_file_gets_a_warning_before_it_is_transcribed():
    """The engine segfaults on long files (v1.5.5). Someone who picks it for a film
    should be told before the run, not 30 minutes later when the fallback starts."""
    from core.crispasr_models import MAX_VERIFIED_SECONDS, long_run_warning

    assert long_run_warning(90) is None
    assert long_run_warning(MAX_VERIFIED_SECONDS) is None
    warning = long_run_warning(47 * 60)
    assert warning is not None
    assert "47-minute" in warning


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("VSCL_AISUBS_CRISPASR_MODEL", "VSCL_AISUBS_CRISPASR_ALIGN",
                "VSCL_AISUBS_CRISPASR_BIN", "VSCL_AISUBS_CRISPASR_CHUNK"):
        monkeypatch.delenv(var, raising=False)


def test_the_default_model_is_the_cpu_safe_one():
    """A VRAM-only rule would hand this laptop a 2B model: its GTX 1650 has 4 GB
    but is throttled to 300 MHz (measured, nvidia-smi throttle reason 0x4)."""
    assert model_tag("en") == "parakeet-0.6b"
    assert model_tag("de") == "parakeet-0.6b"
    assert model_tag(None) == "parakeet-0.6b"


def test_auto_tiers_by_vram(monkeypatch):
    monkeypatch.setenv("VSCL_AISUBS_CRISPASR_MODEL", "auto")
    assert model_tag("en", vram_mb=0) == "parakeet-0.6b"
    assert model_tag("en", vram_mb=4096) == "parakeet-1.1b"
    assert model_tag("en", vram_mb=6144) == "cohere"
    assert model_tag("en", vram_mb=8192) == "canary-qwen"


def test_a_tier_that_cannot_handle_the_language_is_not_chosen():
    """canary-qwen and voxtral do not cover Japanese; the pick must still be usable."""
    model = pick(8192, "ja")
    assert model.handles("ja")
    assert model.tag in ("cohere", "qwen3-1.7b")


def test_an_empty_language_set_means_wide_coverage():
    """Qwen3-ASR lists 30 languages + 22 dialects; enumerating it is pointless,
    and treating "not enumerated" as "handles nothing" made it unreachable."""
    qwen = next(m for m in MODELS if m.tag == "qwen3-1.7b")
    assert qwen.handles("ja") and qwen.handles("sw") and qwen.handles(None)


def test_a_named_model_wins(monkeypatch):
    monkeypatch.setenv("VSCL_AISUBS_CRISPASR_MODEL", "cohere")
    assert model_tag("en") == "cohere"
    assert model_label("en") == "cohere-transcribe-03-2026"


def test_a_gguf_path_is_passed_through(monkeypatch, tmp_path):
    gguf = tmp_path / "my-quant.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setenv("VSCL_AISUBS_CRISPASR_MODEL", str(gguf))
    assert model_tag("en") == str(gguf)


def test_the_model_argument_is_auto_only_for_backend_defaults():
    """`auto` for a non-default variant would run a different model than the
    status line names — the class of lie this project has been bitten by before."""
    assert model_argument("parakeet-0.6b") == "auto"
    assert model_argument("cohere") == "auto"
    assert model_argument("parakeet-1.1b") == "parakeet-1.1b"
    assert model_argument("qwen3-1.7b") == "qwen3-1.7b"
    assert model_argument("/tmp/x.gguf") == "/tmp/x.gguf"


def test_the_aligner_is_off_on_cpu_and_on_with_a_gpu():
    """Measured on 90 s of the test film: 21.1 s without the aligner, 88.8 s with it."""
    assert align_enabled(gpu=False) is False
    assert align_enabled(gpu=True) is True


def test_the_aligner_can_be_forced_either_way(monkeypatch):
    monkeypatch.setenv("VSCL_AISUBS_CRISPASR_ALIGN", "1")
    assert align_enabled(gpu=False) is True
    monkeypatch.setenv("VSCL_AISUBS_CRISPASR_ALIGN", "0")
    assert align_enabled(gpu=True) is False


def test_the_command_always_splits_on_punctuation():
    """Without -sp the binary emits ONE cue for the whole file (86 s measured)."""
    cmd = build_command("/bin/crispasr", "a.wav", "auto", "parakeet", "en", "/tmp/o", False, None)
    assert "-sp" in cmd and "-osrt" in cmd


def test_the_command_never_forces_a_gpu_backend():
    """The CUDA build selects its backend at runtime; forcing one would defeat the
    CPU fallback that lets a single install work on both machines."""
    cmd = build_command("/bin/crispasr", "a.wav", "auto", "parakeet", "en", "/tmp/o", False, None)
    assert "--gpu-backend" not in cmd


def test_cpu_device_adds_the_no_gpu_flag():
    cmd = build_command("/bin/crispasr", "a.wav", "auto", "parakeet", "en", "/tmp/o", False, "cpu")
    assert "-ng" in cmd


def test_the_aligner_flags_are_added_together():
    cmd = build_command("/bin/crispasr", "a.wav", "auto", "parakeet", "en", "/tmp/o", True, None)
    assert cmd[cmd.index("-am") + 1] == "auto"
    assert "-falign" in cmd


def test_the_chunk_is_sized_from_available_memory():
    """Measured ~4 MB of peak per second of audio; the budget is a quarter of
    MemAvailable, so a transcription cannot push the desktop into the OOM killer
    (which on this box has already killed Chromium). Unknown RAM = conservative."""
    assert chunk_seconds(8000) == 500
    assert chunk_seconds(2000) == 125
    assert chunk_seconds(400) == 30, "tiny machines clamp to the floor"
    assert chunk_seconds(0) == 60, "unreadable /proc/meminfo falls back conservatively"


def test_the_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("VSCL_AISUBS_CRISPASR_CHUNK", "600")
    assert chunk_seconds(8000) == 600
    monkeypatch.setenv("VSCL_AISUBS_CRISPASR_CHUNK", "5")
    assert chunk_seconds(8000) == 30, "clamped up"
    monkeypatch.setenv("VSCL_AISUBS_CRISPASR_CHUNK", "99999")
    assert chunk_seconds(8000) == 3600, "right: 3600, not the RAM value"
    monkeypatch.setenv("VSCL_AISUBS_CRISPASR_CHUNK", "0")
    assert chunk_seconds(8000) == 0, "0 is an explicit opt-out"
    monkeypatch.setenv("VSCL_AISUBS_CRISPASR_CHUNK", "nonsense")
    assert chunk_seconds(8000) == 500, "unparseable falls back to the RAM-derived value"


def test_chunking_is_on_by_default(monkeypatch):
    """Measured: a 1200 s input with the aligner peaked at 8.3 GB RSS + 4.8 GB swap
    (zram here, so real RAM) and was OOM-killed, against ~300 MB for 90 s."""
    monkeypatch.setattr("core.crispasr_models.available_ram_mb", lambda: 8000)
    cmd = build_command("/bin/crispasr", "a.wav", "auto", "parakeet", "en", "/tmp/o",
                        True, None, chunk_seconds())
    assert cmd[cmd.index("--chunk-seconds") + 1] == "500"


def test_no_chunk_flag_when_chunking_is_disabled():
    cmd = build_command("/bin/crispasr", "a.wav", "auto", "parakeet", "en", "/tmp/o",
                        False, None, 0)
    assert "--chunk-seconds" not in cmd


def test_parse_srt_reads_cues(tmp_path):
    path = tmp_path / "x.srt"
    path.write_text("1\n00:00:01,500 --> 00:00:03,250\nHello there.\n\n"
                    "2\n00:01:00,000 --> 00:01:02,000\nSecond\nline.\n", encoding="utf-8")
    cues = parse_srt(str(path))
    assert cues[0] == {"start": 1.5, "end": 3.25, "text": "Hello there."}
    assert cues[1]["text"] == "Second line."


def test_the_binary_honours_the_env_override(tmp_path, monkeypatch):
    fake = tmp_path / "crispasr"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("VSCL_AISUBS_CRISPASR_BIN", str(fake))
    assert binary() == str(fake)


def test_a_broken_env_path_reports_absent_rather_than_guessing(monkeypatch):
    """Pointing at a path that does not exist must not silently fall back to a
    different binary — the engine would then report a model it is not running."""
    monkeypatch.setenv("VSCL_AISUBS_CRISPASR_BIN", "/nonexistent/crispasr")
    assert binary() is None
