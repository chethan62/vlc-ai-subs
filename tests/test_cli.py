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

    def boom(*_args, **_kwargs):
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


class _FakeBackend:
    def __init__(self, segments):
        self._segments = segments

    def name(self):
        return "fake"

    def model_label(self, requested, language=None):
        """No opinion → the CLI's VRAM/RAM picker decides."""
        return None

    def transcribe(self, media_path, model_name, language, task):
        yield from self._segments


def test_srt_write_failure_falls_back_to_temp(monkeypatch, capsys, tmp_path):
    """Media dir read-only: first SRT write fails → temp fallback + status
    warning + done with the fallback path (exit 0, never an abort)."""
    import aisubs_whisper

    monkeypatch.setattr(
        aisubs_whisper, "resolve_backend",
        lambda *_a, **_kw: _FakeBackend([{"start": 0.0, "end": 1.0, "text": "Hi"}]),
    )
    media = tmp_path / "ro.mp4"
    media.write_bytes(b"junk")

    calls = {"n": 0}

    def flaky_write(segments, media_path, srt_requested):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("Read-only file system")
        with open(srt_requested, "w", encoding="utf-8") as f:
            f.write("ok\n")
        return srt_requested

    monkeypatch.setattr(aisubs_whisper, "write_srt", flaky_write)
    monkeypatch.setattr(
        sys, "argv", ["aisubs_whisper.py", str(media), "tiny", "en", "transcribe"]
    )
    aisubs_whisper.main()  # no SystemExit → exit 0

    lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    last = lines[-1]
    assert last["type"] == "done"
    assert last["segments"] == 1
    assert last["srt_path"] and last["srt_path"] != str(media) + ".srt"
    assert os.path.isfile(last["srt_path"])
    assert any(l["type"] == "status" and "wrote" in l["msg"] for l in lines)
    os.remove(last["srt_path"])


def test_srt_write_failure_both_paths_errors(monkeypatch, capsys, tmp_path):
    """Media dir and temp dir both unwritable → clean JSONL error, exit 1."""
    import aisubs_whisper

    monkeypatch.setattr(
        aisubs_whisper, "resolve_backend",
        lambda *_a, **_kw: _FakeBackend([{"start": 0.0, "end": 1.0, "text": "Hi"}]),
    )
    media = tmp_path / "ro.mp4"
    media.write_bytes(b"junk")

    def always_fail(segments, media_path, srt_requested):
        raise OSError("Read-only file system")

    monkeypatch.setattr(aisubs_whisper, "write_srt", always_fail)
    monkeypatch.setattr(
        sys, "argv", ["aisubs_whisper.py", str(media), "tiny", "en", "transcribe"]
    )
    with pytest.raises(SystemExit) as exc:
        aisubs_whisper.main()
    assert exc.value.code == 1

    lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert lines[-1]["type"] == "error"
    assert "Could not write SRT" in lines[-1]["msg"]


# ── model resolution: dialog pick → the model that actually runs ──────────

def test_resolve_model_name_recommended_uses_hardware_pick(monkeypatch):
    import aisubs_whisper

    monkeypatch.setattr(aisubs_whisper, "_detect_vram_mb", lambda: 4096)
    monkeypatch.setattr(aisubs_whisper, "_detect_ram_gb", lambda: 16)
    assert aisubs_whisper.resolve_model_name("recommended", "whisperx (aligned)") == "large-v3-turbo"


def test_resolve_model_name_explicit_pick_is_passed_through():
    import aisubs_whisper

    assert aisubs_whisper.resolve_model_name("medium", "whisperx (aligned)") == "medium"
    assert aisubs_whisper.resolve_model_name("large-v3-turbo", "whisperx (aligned)") == "large-v3-turbo"


def test_resolve_model_name_uses_the_backend_label():
    """An engine that ignores the dialog's pick reports its own model — Parakeet
    runs an installed variant, so a WhisperX name must never be reported."""
    import aisubs_whisper
    from backends.parakeet import ParakeetBackend
    from core.parakeet_models import model_label

    assert "parakeet" in model_label()
    backend = ParakeetBackend()
    assert aisubs_whisper.resolve_model_name("recommended", backend.name(), backend) == model_label()
    assert aisubs_whisper.resolve_model_name("large", backend.name(), backend) == model_label()


def test_resolved_model_label_follows_the_language(monkeypatch, tmp_path):
    """Parakeet's label is language-dependent: English runs v2, the other 24
    languages run v3. A language-blind label printed "parakeet-tdt-0.6b-v2"
    while the French run was actually loading v3."""
    import aisubs_whisper
    from backends.parakeet import ParakeetBackend
    import core.parakeet_models as pm

    root = tmp_path / "models"
    for tag in ("v2", "v3"):
        variant = next(v for v in pm.VARIANTS if v.tag == tag)
        d = root / variant.dirname
        d.mkdir(parents=True)
        for name in pm.MODEL_FILES:
            (d / name).write_text("")
    monkeypatch.setattr(pm, "MODELS_ROOT", str(root))
    monkeypatch.delenv("VSCL_AISUBS_PARAKEET_MODEL", raising=False)
    monkeypatch.delenv("VSCL_AISUBS_PARAKEET_VERSION", raising=False)

    backend = ParakeetBackend()
    label = lambda lang: aisubs_whisper.resolve_model_name("recommended", backend.name(), backend, lang)
    assert label("en") == "parakeet-tdt-0.6b-v2"
    assert label("en-GB") == "parakeet-tdt-0.6b-v2"
    assert label(None) == "parakeet-tdt-0.6b-v2"
    assert label("fr") == "parakeet-tdt-0.6b-v3"
    assert label("uk") == "parakeet-tdt-0.6b-v3"


def test_auto_engine_rule_prefers_parakeet_for_covered_languages(monkeypatch):
    """Auto uses Parakeet when an installed variant covers the language, and the
    hardware policy otherwise (mirrored in aisubs.lua engine_for)."""
    import aisubs_whisper
    import core.parakeet_models as pm

    monkeypatch.delenv("VSCL_AISUBS_PARAKEET_MODEL", raising=False)
    monkeypatch.delenv("VSCL_AISUBS_PARAKEET_VERSION", raising=False)

    # English-only install: English yes, French no (that needs v3)
    monkeypatch.setattr(pm, "installed_variants", lambda: [pm.VARIANTS[0]])
    assert aisubs_whisper.auto_engine_for("en", "transcribe") == "parakeet"
    assert aisubs_whisper.auto_engine_for("en-GB", "transcribe") == "parakeet"
    assert aisubs_whisper.auto_engine_for(None, "transcribe") == "parakeet"
    assert aisubs_whisper.auto_engine_for("fr", "transcribe") == "auto"
    assert aisubs_whisper.auto_engine_for("ja", "transcribe") == "auto"

    # v3 installed: the 25 languages are covered, others still are not
    monkeypatch.setattr(pm, "installed_variants", lambda: [pm.VARIANTS[0], pm.VARIANTS[1]])
    assert aisubs_whisper.auto_engine_for("fr", "transcribe") == "parakeet"
    assert aisubs_whisper.auto_engine_for("uk", "transcribe") == "parakeet"
    assert aisubs_whisper.auto_engine_for("ja", "transcribe") == "auto"

    # no model installed: the hardware policy decides everything
    monkeypatch.setattr(pm, "installed_variants", lambda: [])
    assert aisubs_whisper.auto_engine_for("en", "transcribe") == "auto"

    # Parakeet has no translation head
    monkeypatch.setattr(pm, "installed_variants", lambda: [pm.VARIANTS[1]])
    assert aisubs_whisper.auto_engine_for("en", "translate") == "auto"


def test_resolve_model_name_without_a_label_uses_the_hardware_pick(monkeypatch):
    import aisubs_whisper

    class _NoOpinion:
        def model_label(self, requested, language=None):
            return None

    monkeypatch.setattr(aisubs_whisper, "_detect_vram_mb", lambda: 4096)
    monkeypatch.setattr(aisubs_whisper, "_detect_ram_gb", lambda: 16)
    assert aisubs_whisper.resolve_model_name(
        "recommended", "whisperx (aligned)", _NoOpinion()
    ) == "large-v3-turbo"
    assert aisubs_whisper.resolve_model_name(
        "tiny", "whisperx (aligned)", _NoOpinion()
    ) == "tiny"


# ── cancellation: pid file + SIGTERM handler ──────────────────────────────

def test_pid_file_round_trip(tmp_path):
    """The extension cancels by reading <mirror>.pid, so it must hold our PID."""
    import aisubs_whisper

    mirror = str(tmp_path / "aisubs_x.txt")
    path = aisubs_whisper._write_pid_file(mirror)
    assert path is not None
    assert path == mirror + aisubs_whisper.PID_SUFFIX
    assert open(path, encoding="utf-8").read().strip() == str(os.getpid())

    aisubs_whisper._remove_pid_file(path)
    assert not os.path.exists(path)


def test_pid_file_skipped_without_mirror():
    import aisubs_whisper

    assert aisubs_whisper._write_pid_file(None) is None


def test_sigterm_stops_children_and_exits(monkeypatch):
    """SIGTERM (what the extension sends) must stop the ML child, not leak it."""
    import signal

    import aisubs_whisper

    calls = []
    monkeypatch.setattr(aisubs_whisper, "terminate_all", lambda: calls.append("stop") or 1)
    aisubs_whisper._install_cancel_handler()

    with pytest.raises(SystemExit) as exc:
        os.kill(os.getpid(), signal.SIGTERM)

    assert exc.value.code == 130
    assert calls == ["stop"]
    # Restore the default so a later signal cannot confuse the test run.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
