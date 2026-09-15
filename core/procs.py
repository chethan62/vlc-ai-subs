"""Registry of live backend subprocesses, so a run can actually be cancelled.

The VLC extension cancels by signalling the CLI process (see the pid file the
CLI writes next to its mirror file). The CLI's cancel handler then has to stop
the ML subprocess it spawned — that child is NOT in a process group we can
signal from the extension (VLC launches us via `sh -c '... &'`), so the
backends register their child here and the handler calls :func:`terminate_all`.

A leaf module: stdlib only, no imports from core/, backends/ or the runners.
"""

import subprocess
import threading
from typing import Iterable

_LOCK = threading.Lock()
_LIVE: set[subprocess.Popen] = set()


def register(proc: subprocess.Popen) -> None:
    with _LOCK:
        _LIVE.add(proc)


def unregister(proc: subprocess.Popen) -> None:
    with _LOCK:
        _LIVE.discard(proc)


def live() -> list:
    """Snapshot of the running children (diagnostics/tests)."""
    with _LOCK:
        return list(_LIVE)


def terminate_all(grace: float = 5.0) -> int:
    """SIGTERM every live child, then SIGKILL the stragglers. Returns the count."""
    with _LOCK:
        procs = list(_LIVE)
    for proc in procs:
        try:
            proc.terminate()
        except OSError:
            pass
    for proc in procs:
        try:
            proc.wait(timeout=grace)
        except (subprocess.TimeoutExpired, OSError):
            try:
                proc.kill()
            except OSError:
                pass
    return len(procs)


def run_captured(
    cmd: Iterable[str],
    timeout: float | None = None,
    env: dict | None = None,
) -> "subprocess.CompletedProcess":
    """``subprocess.run(capture_output=True, text=True)`` + cancellability.

    Same contract as subprocess.run: on timeout the child is killed and
    ``TimeoutExpired`` raised. The only difference is that the live process is
    registered in this module while it runs, so terminate_all() can stop it.
    """
    cmd = list(cmd)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    register(proc)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(
                cmd, float(timeout or 0), output=stdout, stderr=stderr
            )
    finally:
        unregister(proc)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
