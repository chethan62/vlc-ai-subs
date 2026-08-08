"""CLI contract tests — error semantics of aisubs_whisper.py.

Real transcription is out of scope for unit tests (needs WhisperX + model
download). These cover argument parsing, missing-file errors, JSONL error
emission, and exit codes.
"""

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(REPO, "aisubs_whisper.py")
PY = sys.executable  # this test runs under the project venv


def run_cli(args, timeout=60):
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ""}
    return subprocess.run(
        [PY, CLI, *args],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


def test_too_few_args_exits_1():
    proc = run_cli(["only-one-arg"])
    assert proc.returncode == 1
    assert "Usage" in (proc.stderr or "")


def test_missing_file_emits_error_on_stdout():
    proc = run_cli(["/nonexistent/movie.mp4", "tiny", "en", "transcribe"])
    assert proc.returncode == 1
    line = json.loads(proc.stdout.strip().splitlines()[-1])
    assert line["type"] == "error"
    assert "File not found" in line["msg"]


def test_missing_file_writes_error_to_mirror(tmp_path):
    mirror = tmp_path / "mirror.jsonl"
    proc = run_cli(
        ["/nonexistent/movie.mp4", "tiny", "en", "transcribe", str(mirror)]
    )
    assert proc.returncode == 1
    lines = mirror.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[-1])["type"] == "error"


def test_backend_failure_emits_json_error_instead_of_traceback(monkeypatch, capsys):
    """If resolve_backend() raises, the CLI must turn it into a JSONL error
    line with exit code 1 — never a raw Python traceback on stdout."""
    import aisubs_whisper

    def boom():
        raise RuntimeError(
            "WhisperX is not available. Install it with:\n  uv venv --python 3.12 ..."
        )

    monkeypatch.setattr(aisubs_whisper, "resolve_backend", boom)
    media = os.path.abspath(__file__)  # a file that exists
    monkeypatch.setattr(
        sys, "argv", ["aisubs_whisper.py", media, "tiny", "en", "transcribe"]
    )
    with pytest.raises(SystemExit) as exc:
        aisubs_whisper.main()
    assert exc.value.code == 1

    out = capsys.readouterr().out
    last = json.loads(out.strip().splitlines()[-1])
    assert last["type"] == "error"
    assert "WhisperX is not available" in last["msg"]
    assert "Traceback" not in out