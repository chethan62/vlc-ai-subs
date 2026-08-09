"""Unit tests for whisperx_runner.py (loaded as a standalone module).

The runner lives outside a package, so tests import it by path via
importlib.util — mirroring how the backend spawns it as a subprocess.
"""

import importlib.util
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