"""Photon engine (Moondream Parakeet Redux): device rule, model choice, runner.

No torch and no moondream here — the *decisions* are what can be silently wrong,
and they are the reason this engine is opt-in: the device rule (CUDA must never
win `auto` — measured 2-2.4x slower than this laptop's CPU because the packed
ternary kernel does not exist there) and the honesty rule (nothing about the
non-English languages has been measured here, so the status line says so).

The runner is loaded standalone and driven with fake `torch` / `moondream` /
`kestrel_kernels` modules: that is how the JSONL contract and the "no stray
<media>.srt" rule get checked without a 3 GB download.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from core.photon_models import (MODELS, device_note, installed, language_warning,
                                model_id, model_label, resolve_device, venv_python)

_RUNNER = str(Path(__file__).resolve().parent.parent / "photon_runner.py")


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("photon_runner_test", _RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("VSCL_AISUBS_PHOTON_MODEL", "VSCL_AISUBS_PHOTON_VENV", "VSCL_AISUBS_DEVICE"):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------
# The device rule
# --------------------------------------------------------------------------

def test_auto_prefers_a_native_kernel_over_a_gpu():
    """The whole reason this engine is not GPU-first.

    Measured on this laptop: CUDA has no packed ternary kernel, so the codes are
    dequantized and four conformer kernels fall back to PyTorch — 2-2.4x slower
    than the same box's CPU. A rule that picked the GPU because it is a GPU would
    make the plugin slower by default.
    """
    assert resolve_device("auto", ["cpu", "cuda"], {"cpu": True, "cuda": False}) == "cpu"


def test_auto_takes_the_gpu_only_when_no_native_kernel_exists():
    """An old x86 with a scalar-only int8 path and a real GPU: use the GPU."""
    assert resolve_device("auto", ["cpu", "cuda"], {"cpu": False, "cuda": False}) == "cuda"


def test_apple_silicon_takes_metal():
    """mps is the native Apple GPU path and comes first on that machine."""
    assert resolve_device("auto", ["cpu", "mps"], {"cpu": True, "mps": True}) == "mps"


def test_the_cpu_is_used_when_it_is_the_only_device():
    assert resolve_device("auto", ["cpu"], {}) == "cpu"


def test_an_explicit_device_is_honoured_even_when_it_lacks_a_kernel():
    assert resolve_device("cuda", ["cpu", "cuda"], {"cpu": True}) == "cuda"


def test_an_unavailable_device_fails_loudly():
    """Silently running somewhere else is how a status line ends up naming a
    device the run never used — the class of lie this project keeps finding."""
    with pytest.raises(ValueError, match="not available"):
        resolve_device("cuda", ["cpu"], {"cpu": True})


def test_a_cuda_run_carries_its_caveat():
    assert "no ternary kernel" in device_note("cuda")
    assert device_note("cpu") is None


# --------------------------------------------------------------------------
# Model choice
# --------------------------------------------------------------------------

def test_the_default_variant_is_redux(monkeypatch):
    assert model_id() == MODELS["redux"][0]
    assert "178 MB" in model_label()


def test_the_variant_comes_from_the_env(monkeypatch):
    monkeypatch.setenv("VSCL_AISUBS_PHOTON_MODEL", "ultra")
    assert model_id() == MODELS["ultra"][0]
    # full precision, not ternary: the ultra card says "same 0.6B parameters in full
    # precision", and its repo is 1.26 GB of weights — the earlier "ternary, 385 MB"
    # label was an invented size for a model this engine can actually select
    assert "full precision" in model_label() and "1.3 GB" in model_label()


def test_an_arbitrary_hf_id_is_passed_through(monkeypatch):
    """A re-quantised or fine-tuned repo needs no code change."""
    monkeypatch.setenv("VSCL_AISUBS_PHOTON_MODEL", "someone/parakeet-redux-v2")
    assert model_id() == "someone/parakeet-redux-v2"
    assert model_label() == "parakeet-redux-v2"


def test_only_english_is_claimed():
    """Parakeet v3 covers 25 languages, but no run here has measured Redux on
    them — and an unverified claim is worse than none."""
    for known in (None, "", "auto", "en", "en-US"):
        assert language_warning(known) is None, known
    warning = language_warning("fr")
    assert warning is not None and "untested" in warning


# --------------------------------------------------------------------------
# Installation detection
# --------------------------------------------------------------------------

def _fake_venv(tmp_path, with_moondream=True):
    venv = tmp_path / "venv-photon"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("#!/bin/sh\n")
    if with_moondream:
        site = venv / "lib" / "python3.12" / "site-packages" / "moondream"
        site.mkdir(parents=True)
        (site / "__init__.py").write_text("")
    return venv


def test_a_missing_venv_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("VSCL_AISUBS_PHOTON_VENV", str(tmp_path / "nope"))
    assert venv_python() is None and installed() is False


def test_a_venv_without_the_package_is_not_installed(tmp_path, monkeypatch):
    """Half an install (interpreter, no package) must not resolve as available:
    the backend would then spawn a runner that dies on ImportError."""
    monkeypatch.setenv("VSCL_AISUBS_PHOTON_VENV", str(_fake_venv(tmp_path, False)))
    assert installed() is False


def test_a_complete_venv_is_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("VSCL_AISUBS_PHOTON_VENV", str(_fake_venv(tmp_path)))
    assert installed() is True


# --------------------------------------------------------------------------
# Runner internals (fake engine modules)
# --------------------------------------------------------------------------

def _fake_torch(cuda=False, mps=False):
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    module.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    return module


def _fake_ternary(metal=False, cpu=True):
    module = types.ModuleType("kestrel_kernels.ternary")
    module.metal_gemm_ready = lambda: metal
    module.ternary_gemm_ready = lambda: cpu
    return module


def test_available_devices_always_has_the_cpu(runner):
    assert runner.available_devices(_fake_torch()) == ["cpu"]
    assert runner.available_devices(_fake_torch(cuda=True, mps=True)) == ["cpu", "mps", "cuda"]


def test_native_kernels_are_read_from_the_engine(runner):
    facts = runner.native_kernels(_fake_ternary(), ["cpu", "cuda"])
    assert facts == {"cpu": True, "cuda": False}


def test_an_unreadable_kernel_module_reads_as_no_kernel(runner):
    """Facts unknown must not read as facts good: False sends `auto` to the
    non-CUDA path, which is the safe direction."""
    assert runner.native_kernels(None, ["cpu", "cuda"]) == {"cpu": False, "cuda": False}


def test_a_broken_kernel_probe_does_not_crash_the_run(runner):
    broken = types.ModuleType("ternary")
    broken.ternary_gemm_ready = lambda: (_ for _ in ()).throw(RuntimeError("nope"))
    assert runner.native_kernels(broken, ["cpu"]) == {"cpu": False}


def test_blank_segments_are_dropped(runner):
    result = {"segments": [
        {"start": 0.0, "end": 1.5, "text": " Hello there. "},
        {"start": 1.5, "end": 2.0, "text": "   "},
        {"start": 2.0, "end": 3.0, "text": "Again."},
    ]}
    assert runner.to_segments(result) == [
        {"start": 0.0, "end": 1.5, "text": "Hello there."},
        {"start": 2.0, "end": 3.0, "text": "Again."},
    ]


def test_a_missing_segments_key_is_not_an_exception(runner):
    """Photon returns {"text": ...} when no timestamps were asked for; a KeyError
    there would surface as a stack trace in the status label."""
    assert runner.to_segments({"text": "hi"}) == []


# --------------------------------------------------------------------------
# The runner's JSONL contract, end to end (fake engine)
# --------------------------------------------------------------------------

class _FakePhoton:
    """Stands in for `md.photon(...)`: a context manager with transcribe()."""

    def __init__(self, segments=None, fail=False):
        self.segments = segments or []
        self.fail = fail
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("engine exploded")
        return {"segments": self.segments}


def _wire(runner, monkeypatch, tmp_path, photon, *, torch_kwargs=None, ternary=True):
    """Point the runner at fakes and a real (empty) temp wav."""
    _FAKE.append(photon)
    wav = tmp_path / "audio.16k.wav"
    wav.write_bytes(b"\0" * 64000)          # 2 s at 32 kB/s
    monkeypatch.setattr(runner, "list_audio_streams", lambda *a, **k: [])
    monkeypatch.setattr(runner, "choose_audio_stream", lambda *a, **k: (0, "only track"))
    monkeypatch.setattr(runner, "decode_to_wav16k", lambda *a, **k: str(wav))
    monkeypatch.setattr(runner, "cleanup_temp", lambda: None)
    monkeypatch.setattr(runner, "install_termination_handler", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(**(torch_kwargs or {})))
    fake_md = types.ModuleType("moondream")
    fake_md.photon = lambda *a, **k: photon
    monkeypatch.setitem(sys.modules, "moondream", fake_md)
    if ternary:
        kernels = types.ModuleType("kestrel_kernels")
        kernels.ternary = _fake_ternary()
        monkeypatch.setitem(sys.modules, "kestrel_kernels", kernels)
        monkeypatch.setitem(sys.modules, "kestrel_kernels.ternary", kernels.ternary)
    else:
        monkeypatch.setitem(sys.modules, "kestrel_kernels", None)
    return wav


def _run(runner, argv, capsys=None):
    """Drive runner.main() with this argv and return (exit code, JSONL events)."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    code = 0
    saved = sys.argv
    sys.argv = list(argv)
    try:
        with redirect_stdout(buf):
            try:
                runner.main()
            except SystemExit as exc:
                code = exc.code or 0
    finally:
        sys.argv = saved
    events = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    return code, events


def test_a_run_emits_cues_and_a_done_event(runner, monkeypatch, tmp_path):
    media = tmp_path / "film.mkv"
    media.write_bytes(b"x")
    photon = _FakePhoton([{"start": 1.0, "end": 2.0, "text": " Hello there."},
                          {"start": 2.0, "end": 3.0, "text": "General Kenobi."}])
    _wire(runner, monkeypatch, tmp_path, photon)

    code, events = _run(runner, ["photon_runner.py", str(media), "recommended",
                                 "en", "transcribe"], None)
    kinds = [e["type"] for e in events]
    assert code == 0
    assert kinds[-1] == "done" and events[-1]["segments"] == 2
    cues = [e for e in events if e["type"] == "sub"]
    assert [c["text"] for c in cues] == ["Hello there.", "General Kenobi."]
    assert photon.calls[0]["timestamps"] == "segment", (
        "cue spans come from segment timestamps; word timings are for a different pass")
    assert photon.calls[0]["audio"].endswith(".wav")


def test_the_status_line_names_the_device_and_the_kernel(runner, monkeypatch, tmp_path):
    media = tmp_path / "film.mkv"
    media.write_bytes(b"x")
    _wire(runner, monkeypatch, tmp_path, _FakePhoton([]))

    _code, events = _run(runner, ["photon_runner.py", str(media), "recommended",
                                  "en", "transcribe"], None)
    first = next(e["msg"] for e in events if e["type"] == "status")
    assert "parakeet-redux" in first and "cpu" in first
    assert "native ternary kernel" in first


def test_a_cuda_run_warns_that_it_is_the_slow_choice(runner, monkeypatch, tmp_path):
    """VSCL_AISUBS_DEVICE=cuda must be honoured AND explained."""
    media = tmp_path / "film.mkv"
    media.write_bytes(b"x")
    monkeypatch.setenv("VSCL_AISUBS_DEVICE", "cuda")
    _wire(runner, monkeypatch, tmp_path, _FakePhoton([]), torch_kwargs={"cuda": True})

    _code, events = _run(runner, ["photon_runner.py", str(media), "recommended",
                                  "en", "transcribe"], None)
    messages = [e["msg"] for e in events if e["type"] == "status"]
    assert any("no ternary kernel" in m for m in messages)


def test_an_unavailable_forced_device_is_an_error_not_a_silent_substitution(runner, monkeypatch, tmp_path):
    media = tmp_path / "film.mkv"
    media.write_bytes(b"x")
    monkeypatch.setenv("VSCL_AISUBS_DEVICE", "cuda")
    _wire(runner, monkeypatch, tmp_path, _FakePhoton([]))

    code, events = _run(runner, ["photon_runner.py", str(media), "recommended",
                                 "en", "transcribe"], None)
    assert code == 1
    assert events[-1]["type"] == "error" and "not available" in events[-1]["msg"]


def test_translate_is_refused_with_the_engine_to_use_instead(runner, monkeypatch, tmp_path):
    media = tmp_path / "film.mkv"
    media.write_bytes(b"x")
    _wire(runner, monkeypatch, tmp_path, _FakePhoton([]))

    code, events = _run(runner, ["photon_runner.py", str(media), "recommended",
                                 "fr", "translate"], None)
    assert code == 1
    assert events[-1]["type"] == "error"
    assert "whisperx" in events[-1]["msg"]


def test_an_engine_failure_is_an_error_event_not_a_traceback(runner, monkeypatch, tmp_path):
    media = tmp_path / "film.mkv"
    media.write_bytes(b"x")
    _wire(runner, monkeypatch, tmp_path, _FakePhoton(fail=True))

    code, events = _run(runner, ["photon_runner.py", str(media), "recommended",
                                 "en", "transcribe"], None)
    assert code == 1
    assert "engine exploded" in events[-1]["msg"]


def test_no_srt_is_written_beside_the_media(runner, monkeypatch, tmp_path):
    """The stray-<media>.srt rule: the CLI is the only writer, and it passes the
    path explicitly. A runner that derives one drops files next to the media in
    every mode — including realtime OSD, which writes to a temp path on purpose."""
    media = tmp_path / "film.mkv"
    media.write_bytes(b"x")
    _wire(runner, monkeypatch, tmp_path,
          _FakePhoton([{"start": 1.0, "end": 2.0, "text": "Hello."}]))

    code, events = _run(runner, ["photon_runner.py", str(media), "recommended",
                                 "en", "transcribe"], None)
    assert code == 0
    assert events[-1]["srt_path"] is None
    assert not list(tmp_path.glob("*.srt"))


def test_an_explicit_srt_path_is_honoured(runner, monkeypatch, tmp_path):
    media = tmp_path / "film.mkv"
    media.write_bytes(b"x")
    out = tmp_path / "out.srt"
    _wire(runner, monkeypatch, tmp_path,
          _FakePhoton([{"start": 1.0, "end": 2.0, "text": "Hello."}]))

    code, events = _run(runner, ["photon_runner.py", str(media), "recommended",
                                 "en", "transcribe", str(tmp_path / "m.txt"), str(out)], None)
    assert code == 0
    assert events[-1]["srt_path"] == str(out) and out.is_file()
    assert "Hello." in out.read_text()


def test_no_speech_ends_cleanly(runner, monkeypatch, tmp_path):
    media = tmp_path / "film.mkv"
    media.write_bytes(b"x")
    _wire(runner, monkeypatch, tmp_path, _FakePhoton([]))

    code, events = _run(runner, ["photon_runner.py", str(media), "recommended",
                                 "en", "transcribe"], None)
    assert code == 0
    assert events[-1]["segments"] == 0 and events[-1]["srt_path"] is None


def test_a_missing_engine_names_the_install_command(runner, monkeypatch, tmp_path):
    media = tmp_path / "film.mkv"
    media.write_bytes(b"x")
    _wire(runner, monkeypatch, tmp_path, _FakePhoton([]))
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("moondream", "torch"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    code, events = _run(runner, ["photon_runner.py", str(media), "recommended",
                                 "en", "transcribe"], None)
    assert code == 1
    assert "install-photon-model.sh" in events[-1]["msg"]


def test_a_usage_error_exits_two(runner, monkeypatch):
    code, events = _run(runner, ["photon_runner.py", "only-two-args"], None)
    assert code == 2
    assert events[-1]["type"] == "error" and "usage" in events[-1]["msg"]


def test_the_host_can_ask_for_raw_segments(runner, monkeypatch, tmp_path):
    """A batch host driving this runner applies its own cue rules, so our
    blocklist/quality passes must not pre-wrap and pre-split the cues — measured
    on one long segment, whose fate is the only difference between the two runs."""
    media = tmp_path / "film.mkv"
    media.write_bytes(b"x")
    # 20 s in three sentences: past the 7 s cue ceiling, splittable at sentence
    # ends, and each sentence too long for one 42-char line
    long_text = ("This is a full sentence with enough words in it to need wrapping. " * 3).strip()
    photon = _FakePhoton([{"start": 0.0, "end": 20.0, "text": long_text}])
    _wire(runner, monkeypatch, tmp_path, photon)
    argv = ["photon_runner.py", str(media), "recommended", "en", "transcribe"]

    monkeypatch.setenv("VSCL_AISUBS_RAW_SEGMENTS", "1")
    code, events = _run(runner, argv, None)
    raw = [e for e in events if e["type"] == "sub"]
    assert code == 0
    assert len(raw) == 1 and raw[0]["end"] == 20.0, "raw keeps the engine's own span"
    assert "\n" not in raw[0]["text"], "and is not wrapped to this plugin's line width"

    monkeypatch.delenv("VSCL_AISUBS_RAW_SEGMENTS")
    _wire(runner, monkeypatch, tmp_path, _FakePhoton([{"start": 0.0, "end": 20.0, "text": long_text}]))
    code, events = _run(runner, argv, None)          # the first run unlinked its temp wav
    processed = [e for e in events if e["type"] == "sub"]
    assert len(processed) > 1, "without the flag the cue pass splits the 20 s cue"
    assert any("\n" in c["text"] for c in processed), "and wraps its lines"


def _wire_calls(runner, tmp_path):
    """The timestamps value the last wired fake engine was called with"""
    return [c["timestamps"] for c in _FAKE[-1].calls] if _FAKE else []


_FAKE = []


def _fake_photon_with_words():
    return _FakePhoton([{
        "start": 0.24, "end": 2.64, "text": "available?",
        "words": [{"start": 0.24, "end": 0.9, "word": "available?"},
                  {"start": 1.0, "end": 1.4, "word": "  "},        # blank token: dropped
                  {"start": 1.5, "end": 2.64, "word": "sir"}],
    }])


def test_word_timings_are_opt_in_and_ride_along(runner, monkeypatch, tmp_path):
    """A host that snaps or aligns to words asks for them; the runtime returns them
    in the same call, so this is a query parameter, not a second decode. The plugin
    never sets it, and its cue pass ignores whatever arrives."""
    media = tmp_path / "film.mkv"
    media.write_bytes(b"x")
    _wire(runner, monkeypatch, tmp_path, _fake_photon_with_words())
    argv = ["photon_runner.py", str(media), "recommended", "en", "transcribe"]

    code, events = _run(runner, argv, None)
    assert code == 0
    assert events[-1]["type"] == "done"
    sub = [e for e in events if e["type"] == "sub"][0]
    assert "words" not in sub, "words are not requested by default"
    assert _wire_calls(runner, tmp_path)[0] == "segment", "and the runtime is asked for segments"

    # A host that wants words is a host that applies its own cue rules: the plugin's
    # own pass rewrites the cue dicts and drops keys it has no use for, so the words
    # only survive for a host that asked for raw segments too (whisperer sets both).
    monkeypatch.setenv("VSCL_AISUBS_PHOTON_WORDS", "1")
    monkeypatch.setenv("VSCL_AISUBS_RAW_SEGMENTS", "1")
    photon = _fake_photon_with_words()
    _wire(runner, monkeypatch, tmp_path, photon)
    code, events = _run(runner, argv, None)
    sub = [e for e in events if e["type"] == "sub"][0]
    assert photon.calls[0]["timestamps"] == "word", "the flag reaches the runtime's own call"
    assert [w["word"] for w in sub["words"]] == ["available?", "sir"], "blank tokens are dropped"
    assert sub["words"][0]["start"] == 0.24 and sub["words"][-1]["end"] == 2.64
    assert any("word timings included" in e.get("msg", "") for e in events)
