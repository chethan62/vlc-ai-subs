"""Unit tests for core/procs.py — cancellable subprocesses (real children).

Cancelling a run has to stop the ML subprocess, which the VLC extension cannot
signal directly (it is not in a process group VLC owns). These tests use real
child processes: the point of the module is exactly the OS-level behaviour.
"""

import subprocess
import sys
import time

import pytest

from core import procs


def _run(code: str) -> subprocess.CompletedProcess:
    return procs.run_captured([sys.executable, "-c", code])


def test_run_captured_returns_stdout_stderr_and_code():
    proc = _run("import sys; print('out'); print('err', file=sys.stderr)")
    assert proc.returncode == 0
    assert proc.stdout.strip() == "out"
    assert proc.stderr.strip() == "err"


def test_run_captured_reports_failure_code():
    proc = _run("raise SystemExit(3)")
    assert proc.returncode == 3


def test_live_process_is_unregistered_after_completion():
    assert procs.live() == []
    _run("pass")
    assert procs.live() == []


def test_timeout_kills_the_child_and_raises():
    started = time.time()
    with pytest.raises(subprocess.TimeoutExpired):
        procs.run_captured([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
    assert time.time() - started < 10  # killed, not waited out
    assert procs.live() == []


def test_terminate_all_stops_a_running_child(monkeypatch):
    """The CLI's cancel handler calls this while the child is mid-run."""
    import threading

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    procs.register(proc)
    try:
        assert procs.live() == [proc]

        # terminate_all blocks waiting for the child, so call it off-thread.
        result = {}
        thread = threading.Thread(target=lambda: result.update(n=procs.terminate_all(grace=5.0)))
        thread.start()
        thread.join(timeout=15)

        assert result["n"] == 1
        assert proc.poll() is not None  # actually dead
    finally:
        procs.unregister(proc)
        if proc.poll() is None:
            proc.kill()


def test_terminate_all_with_nothing_running_is_a_noop():
    assert procs.terminate_all() == 0
