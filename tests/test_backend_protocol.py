"""A runner that stops without its terminal event must be an error.

Every runner ends a run with {"type": "done"} or {"type": "error"} (the protocol in
aisubs_whisper.py). The backends used to ignore the done line entirely: they yielded
subs and raised on errors, and if the child died early they simply ran out of output.
Measured on a 145-minute film, a run died at 83 s and the plugin reported a clean
finish — the child exited 0 with an empty stdout, so "no subs" and "no speech detected"
were indistinguishable, and the child's stderr (the only clue) was thrown away.

The fakes below are the smallest thing that reproduces that: a completed process whose
stdout has no terminal line.
"""

import pytest

from backends import parakeet as parakeet_mod
from backends.base import check_runner_completed
from backends.parakeet import ParakeetBackend


class _Proc:
    """Stand-in for subprocess.CompletedProcess, as run_captured returns it."""

    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


SUB = '{"type": "sub", "start": 0.0, "end": 1.0, "text": "hello"}\n'
DONE = '{"type": "done", "segments": 1, "srt_path": "/tmp/x.srt"}\n'


def test_completed_run_passes():
    check_runner_completed(_Proc(DONE), "Test", saw_done=True)   # must not raise


def test_a_finished_run_still_reports_its_segments(monkeypatch):
    monkeypatch.setattr(parakeet_mod, "run_captured", lambda *a, **k: _Proc(SUB + DONE))
    out = list(ParakeetBackend().transcribe("/tmp/x.mkv", "recommended", "en", "transcribe"))
    assert [s["text"] for s in out] == ["hello"]


def test_a_run_that_stops_without_done_raises_with_the_child_stderr(monkeypatch):
    """The regression: exit code 0, one cue, no terminal line. Before this it looked
    like a completed run that simply produced no speech."""
    killed = "Killed\nsome diagnostic on stderr\n"
    monkeypatch.setattr(parakeet_mod, "run_captured", lambda *a, **k: _Proc(SUB, killed, 0))
    with pytest.raises(RuntimeError) as exc:
        list(ParakeetBackend().transcribe("/tmp/x.mkv", "recommended", "en", "transcribe"))
    message = str(exc.value)
    assert "without finishing" in message
    assert "rc=0" in message
    assert "diagnostic on stderr" in message, "the child's stderr is the only clue — keep it"


def test_an_empty_run_with_no_terminal_line_also_raises(monkeypatch):
    """Exit 0 and completely empty output — the shape that read as 'no speech'."""
    monkeypatch.setattr(parakeet_mod, "run_captured", lambda *a, **k: _Proc("", "", 0))
    with pytest.raises(RuntimeError) as exc:
        list(ParakeetBackend().transcribe("/tmp/x.mkv", "recommended", "en", "transcribe"))
    assert "without finishing" in str(exc.value)
