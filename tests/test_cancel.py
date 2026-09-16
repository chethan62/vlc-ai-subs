"""Cancelling a run must not leave processes or partial decodes behind.

Measured before this existed: SIGTERM to the CLI mid-decode stopped the processes
but left an **87 MB partial wav** in /tmp (a 145-minute film would leave ~280 MB),
and every cancelled run added another. Two separate holes caused it:

  * ``decode_to_wav16k`` used a bare ``subprocess.run``, so the ffmpeg child was
    never registered and outlived a cancelled run;
  * the CLI's cancel handler can only see *its own* children — the registry is
    per-process — so the runner's ffmpeg was invisible to it, and the runner had no
    handler to clean up its own decoded file.

These tests run the real seams in a real child process, because signal handling and
process lifetime are exactly what a mocked test would not exercise.
"""

import os
import signal
import subprocess
import sys
import time

import pytest

from core import audio as audio_mod


def test_cleanup_temp_removes_our_files_only(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    ours = tmp_path / "aisubs_keep.wav"
    ours.write_bytes(b"x")
    theirs = tmp_path / "unrelated.wav"
    theirs.write_bytes(b"x")
    audio_mod._LIVE_TEMP.clear()
    audio_mod._LIVE_TEMP.add(str(ours))

    assert audio_mod.cleanup_temp() == 1
    assert not ours.exists()
    assert theirs.exists()          # somebody else's file is not ours to delete
    assert audio_mod.cleanup_temp() == 0


def test_sweep_stale_temp_is_age_gated(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    fresh = tmp_path / "aisubs_fresh.wav"
    old = tmp_path / "aisubs_old.wav"
    fresh.write_bytes(b"x")
    old.write_bytes(b"x")
    os.utime(old, (time.time() - 3 * 3600, time.time() - 3 * 3600))

    assert audio_mod.sweep_stale_temp() == 1
    assert fresh.exists(), "a running job's decode must survive the sweep"
    assert not old.exists()


def test_sweep_covers_the_extension_temp_files_not_just_wavs(tmp_path, monkeypatch):
    """A killed VLC leaves the mirror/.pid/temp-srt behind too (measured: two mirrors
    and a temp srt). The debug log is not ours to delete."""
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    stale = time.time() - 3 * 3600
    for name in ("aisubs_1789564721_252943.txt", "aisubs_1789564721_252943.pid",
                 "aisubs_4uozf191.srt", "aisubs_99.wav"):
        path = tmp_path / name
        path.write_bytes(b"x")
        os.utime(path, (stale, stale))
    keep = tmp_path / "aisubs_debug.log"          # a log, not a temp artefact
    keep.write_bytes(b"x")
    os.utime(keep, (stale, stale))
    foreign = tmp_path / "something-else.srt"     # not our naming scheme
    foreign.write_bytes(b"x")
    os.utime(foreign, (stale, stale))

    assert audio_mod.sweep_stale_temp() == 4
    assert keep.exists()
    assert foreign.exists()


def test_decode_registers_its_temp_while_it_runs(monkeypatch):
    """The temp must be *in the registry* during the decode, or nothing can clean it."""
    seen = {}

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run_captured(cmd, timeout=None, env=None):
        seen["cmd"] = cmd
        seen["tmp"] = cmd[-1]
        seen["registered"] = cmd[-1] in audio_mod._LIVE_TEMP
        with open(cmd[-1], "wb") as handle:      # stand in for ffmpeg's output
            handle.write(b"\0" * 64)
        return _Done()

    monkeypatch.setattr(audio_mod, "run_captured", fake_run_captured)
    monkeypatch.setattr(audio_mod, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    path = audio_mod.decode_to_wav16k("/some/media.mkv", stream_index=2)
    try:
        assert seen["registered"], "the decode's temp was not registered while running"
        assert seen["cmd"][:2] == ["/usr/bin/ffmpeg", "-y"]
        assert ["-map", "0:2"] == seen["cmd"][seen["cmd"].index("-map"):seen["cmd"].index("-map") + 2]
    finally:
        audio_mod.discard_temp(path)
    assert seen["tmp"] not in audio_mod._LIVE_TEMP


# ── the handler, in a real child process ──────────────────────────────────────

_CHILD = """
import sys, time, tempfile, os
sys.path.insert(0, {repo!r})
from core.procs import run_captured, install_termination_handler

temp = os.path.join(tempfile.gettempdir(), "aisubs_child_test.wav")
open(temp, "wb").write(b"partial")

def cleanup():
    try:
        os.unlink(temp)
    except OSError:
        pass

import core.audio as audio
audio._LIVE_TEMP.add(temp)
install_termination_handler(cleanup)

# a real registered child: this is what a cancelled run must not leave running
run_captured(["sleep", "120"])
print("finished", flush=True)
"""


def test_sigterm_stops_the_child_and_removes_the_partial_decode(tmp_path):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp = "/tmp/aisubs_child_test.wav"
    try:
        os.unlink(temp)
    except OSError:
        pass

    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD.format(repo=repo)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    try:
        out = []
        deadline = time.time() + 15
        while time.time() < deadline:       # wait until the sleep is running
            out = subprocess.run(["pgrep", "-P", str(child.pid)],
                                 capture_output=True, text=True).stdout.split()
            if out:
                break
            time.sleep(0.2)
        assert out, "the child never started its registered subprocess"
        grandchild = int(out[0])

        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=15) == 130, "the handler must exit non-zero"

        assert not os.path.exists(temp), "the partial decode was left behind"
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild, 0)          # the registered child really is gone
    finally:
        if child.poll() is None:
            child.kill()
        try:
            os.unlink(temp)
        except OSError:
            pass
