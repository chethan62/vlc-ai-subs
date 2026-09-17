"""Engine failure handling: a diagnostic must survive, and a dead engine must not
leave the user with nothing.

Measured motivation: CrispASR v0.8.33 segfaults on a full-length file (it asks the
kernel for a 123.6 GB allocation, is refused, and dereferences the NULL instead of
handling it), so a user who explicitly chose that engine got no subtitles at all.
Two things had to be fixed to find that out and act on it:

1. the runner names the signal instead of reporting "rc=-11";
2. the backend reads the runner's error event *before* judging its exit code, so
   that message is not replaced by a bare "rc=1";
3. the CLI retries once with the policy's engine — loudly — when an engine dies
   before producing a single cue.
"""

import pytest

import aisubs_whisper as cli
from backends import crispasr as crispasr_mod
from backends.crispasr import CrispAsrBackend


class _Proc:
    """Stand-in for subprocess.CompletedProcess, as run_captured returns it."""

    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class FakeBackend:
    def __init__(self, name: str):
        self._name = name

    def name(self) -> str:
        return self._name


# ── the CLI's fallback decision ──────────────────────────────────────────

def test_a_failed_engine_falls_back_to_the_policy_pick(monkeypatch):
    monkeypatch.setattr(cli, "auto_engine_for", lambda language, task="transcribe": "parakeet")
    monkeypatch.setattr(cli, "resolve_backend", lambda name: FakeBackend(name))
    fallback = cli._fallback_engine(FakeBackend("crispasr"), "en", "transcribe")
    assert fallback is not None
    assert fallback.name() == "parakeet"


def test_no_fallback_when_the_failed_engine_is_the_policy_pick(monkeypatch):
    """Retrying the same engine would fail exactly the same way."""
    monkeypatch.setattr(cli, "auto_engine_for", lambda language, task="transcribe": "auto")
    monkeypatch.setattr(cli, "resolve_backend", lambda name: FakeBackend("whisperx"))
    assert cli._fallback_engine(FakeBackend("whisperx"), "en", "transcribe") is None


def test_no_fallback_when_nothing_else_resolves(monkeypatch):
    def boom(name):
        raise RuntimeError(f"{name} is not usable")

    monkeypatch.setattr(cli, "auto_engine_for", lambda language, task="transcribe": "parakeet")
    monkeypatch.setattr(cli, "resolve_backend", boom)
    assert cli._fallback_engine(FakeBackend("crispasr"), "en", "transcribe") is None


def test_the_policy_pick_is_still_tried_when_the_named_one_is_broken(monkeypatch):
    """A broken Parakeet must not stop the hardware policy from answering."""
    calls = []

    def resolve(name):
        calls.append(name)
        if name == "parakeet":
            raise RuntimeError("sherpa-onnx missing")
        return FakeBackend(name)

    monkeypatch.setattr(cli, "auto_engine_for", lambda language, task="transcribe": "parakeet")
    monkeypatch.setattr(cli, "resolve_backend", resolve)
    fallback = cli._fallback_engine(FakeBackend("crispasr"), "en", "transcribe")
    assert fallback is not None
    assert fallback.name() == "auto"
    assert calls == ["parakeet", "auto"]


def test_translate_never_falls_back_to_an_engine_without_a_translation_head(monkeypatch):
    """auto_engine_for already refuses Parakeet for translate; the fallback inherits it."""
    monkeypatch.setattr(cli, "resolve_backend", lambda name: FakeBackend(name))
    seen = {}

    def pick(language, task="transcribe"):
        seen["task"] = task
        return "auto"

    monkeypatch.setattr(cli, "auto_engine_for", pick)
    cli._fallback_engine(FakeBackend("crispasr"), "en", "translate")
    assert seen["task"] == "translate"


# ── the backend's message precedence ─────────────────────────────────────

CRASH_ERROR = ('{"type": "error", "msg": "CrispASR failed (killed by SIGSEGV): '
               'no output on stderr (a crash reports to the kernel log)"}\n')


def test_the_runners_own_error_message_survives_a_non_zero_exit(monkeypatch):
    """The regression: the runner exits 1 *after* emitting its error event, and
    judging the exit code first replaced the crash diagnosis with "rc=1"."""
    monkeypatch.setattr(crispasr_mod, "run_captured",
                        lambda *a, **k: _Proc(CRASH_ERROR, "", returncode=1))
    with pytest.raises(RuntimeError) as exc:
        list(CrispAsrBackend().transcribe("/tmp/x.mkv", "recommended", "en", "transcribe"))
    message = str(exc.value)
    assert "SIGSEGV" in message, message
    assert "rc=1" not in message, "the bare exit code must not win"


def test_a_runner_that_dies_silently_still_raises(monkeypatch):
    """No terminal event at all — the shared check must still catch it, with stderr."""
    monkeypatch.setattr(crispasr_mod, "run_captured",
                        lambda *a, **k: _Proc("", "Killed\n", returncode=137))
    with pytest.raises(RuntimeError) as exc:
        list(CrispAsrBackend().transcribe("/tmp/x.mkv", "recommended", "en", "transcribe"))
    message = str(exc.value)
    assert "without finishing" in message
    assert "Killed" in message, "the child's stderr is the only clue — keep it"
