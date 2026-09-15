"""Unit tests for whispercpp_runner.py — the Vulkan (AMD/Intel/NVIDIA) engine.

The binary is faked: a tiny shell script that mimics whisper-cli's contract
(writes <prefix>.json, logs ggml_vulkan device lines to stderr), so the whole
runner — argv construction, JSON parsing, SRT guard, JSONL emission — is
exercised hermetically, with an "AMD" device to keep the vendor-neutral path
honest. ffmpeg is stubbed by patching the runner's decode.
"""

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

_RUNNER = str(Path(__file__).resolve().parent.parent / "whispercpp_runner.py")

_FAKE_CLI = """#!/bin/sh
# Minimal whisper-cli stand-in: honours -of, logs a Vulkan device list.
out=""
while [ $# -gt 0 ]; do
    case "$1" in -of) out="$2"; shift ;; esac
    shift
done
echo "ggml_vulkan: Found 1 Vulkan devices:" >&2
echo "ggml_vulkan: 0 = Fake Radeon RX 7900 XTX (AMD) | uma: 0 | fp16: 1" >&2
cat > "$out.json" <<'JSON'
{"transcription": [
  {"offsets": {"from": 0, "to": 2880}, "text": " Testing the AMD/Intel path."},
  {"offsets": {"from": 2880, "to": 5000}, "text": "   "}
]}
JSON
exit 0
"""


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("whispercpp_runner_test", _RUNNER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_cli(tmp_path) -> str:
    path = tmp_path / "whisper-cli"
    path.write_text(_FAKE_CLI)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


# ── pure helpers ───────────────────────────────────────────────────────

def test_parse_transcription_converts_milliseconds_and_skips_blanks(runner):
    payload = {
        "transcription": [
            {"offsets": {"from": 1000, "to": 2500}, "text": " one "},
            {"offsets": {"from": 2500, "to": 3000}, "text": "   "},
            {"offsets": {"from": 3000, "to": 3500}, "text": "two"},
        ]
    }
    assert runner.parse_transcription(payload) == [
        {"start": 1.0, "end": 2.5, "text": "one"},
        {"start": 3.0, "end": 3.5, "text": "two"},
    ]


def test_parse_transcription_survives_missing_offsets(runner):
    assert runner.parse_transcription({"transcription": [{"text": "hi"}]}) == [
        {"start": 0.0, "end": 0.0, "text": "hi"}
    ]
    assert runner.parse_transcription({}) == []


def test_build_args_uses_vulkan_by_default_and_ng_for_cpu(runner):
    gpu = runner.build_args("/bin/whisper-cli", "m.bin", "a.wav", "/tmp/o",
                            "en", "transcribe")
    assert "-oj" in gpu and "-ng" not in gpu
    assert gpu[gpu.index("-l") + 1] == "en"
    assert "-tr" not in gpu

    cpu = runner.build_args("/bin/whisper-cli", "m.bin", "a.wav", "/tmp/o",
                            None, "transcribe", device="cpu")
    assert "-ng" in cpu and "-l" not in cpu

    translated = runner.build_args("/bin/whisper-cli", "m.bin", "a.wav", "/tmp/o",
                                   "fr", "translate")
    assert "-tr" in translated and translated[translated.index("-l") + 1] == "fr"


def test_vulkan_status_reports_the_device(runner):
    stderr = (
        "ggml_vulkan: Found 1 Vulkan devices:\n"
        "ggml_vulkan: 0 = AMD Radeon RX 7900 XTX (RADV NAVI31) | uma: 0 | fp16: 1\n"
    )
    assert runner.vulkan_status(stderr) == "Vulkan GPU: AMD Radeon RX 7900 XTX (RADV NAVI31)"
    assert runner.vulkan_status("", gpu_used=False).startswith("GPU disabled")
    assert "CPU" in runner.vulkan_status("ggml_vulkan: Found 0 Vulkan devices:\n")
    assert runner.vulkan_status("nothing here") is None


def test_model_dir_and_installed_models(runner, tmp_path):
    (tmp_path / "ggml-small.bin").write_text("")
    (tmp_path / "ggml-medium.bin").write_text("")
    dirs = [str(tmp_path)]
    assert runner.installed_models(dirs) == {
        "small": str(tmp_path / "ggml-small.bin"),
        "medium": str(tmp_path / "ggml-medium.bin"),
    }


def test_recommended_picks_the_largest_installed_model(runner, tmp_path):
    (tmp_path / "ggml-small.bin").write_text("")
    (tmp_path / "ggml-medium.bin").write_text("")
    path, name = runner.resolve_model("recommended", [str(tmp_path)])
    assert name == "medium" and path.endswith("ggml-medium.bin")


def test_explicit_model_requires_installation(runner, tmp_path):
    (tmp_path / "ggml-small.bin").write_text("")
    with pytest.raises(RuntimeError) as exc:
        runner.resolve_model("large", [str(tmp_path)])
    msg = str(exc.value)
    assert "install-whisper-cpp.sh large" in msg and "installed: small" in msg


def test_known_but_missing_size_is_actionable(runner, tmp_path):
    """An explicit whisper.cpp size that is not installed → installer hint."""
    (tmp_path / "ggml-small.bin").write_text("")
    with pytest.raises(RuntimeError) as exc:
        runner.resolve_model("large-v3-turbo", [str(tmp_path)])
    assert "install-whisper-cpp.sh large-v3-turbo" in str(exc.value)


def test_unknown_model_name_falls_back_to_installed(runner, tmp_path):
    """A name whisper.cpp has no file for (e.g. a WhisperX-only pick) must not
    break the engine — run the best installed model instead."""
    (tmp_path / "ggml-base.bin").write_text("")
    _, name = runner.resolve_model("large-v3", [str(tmp_path)])
    assert name == "base"


def test_no_models_is_actionable(runner, tmp_path):
    with pytest.raises(RuntimeError) as exc:
        runner.resolve_model("small", [str(tmp_path)])
    assert "install-whisper-cpp.sh" in str(exc.value)


# ── whole-runner contract (fake binary) ────────────────────────────────

def _run(runner, monkeypatch, tmp_path, argv):
    monkeypatch.setattr(runner, "decode_to_wav16k", lambda _p: str(tmp_path / "dec.wav"))
    (tmp_path / "ggml-small.bin").write_text("")
    monkeypatch.setenv("VSCL_AISUBS_WHISPERCPP_BIN", _fake_cli(tmp_path))
    monkeypatch.delenv("VSCL_AISUBS_WHISPERCPP_MODEL", raising=False)
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()


def test_main_emits_jsonl_and_honours_explicit_srt(runner, monkeypatch, tmp_path, capsys):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    wanted = tmp_path / "out.srt"
    _run(runner, monkeypatch, tmp_path,
         ["runner", str(media), "small", "en", "transcribe", "mirror.txt", str(wanted)])

    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    msgs = [l.get("msg", "") for l in lines if l["type"] == "status"]
    assert any("AMD" in m for m in msgs)          # device surfaced to the dialog
    assert any("GPU disabled" not in m for m in msgs)

    subs = [l for l in lines if l["type"] == "sub"]
    assert [s["text"] for s in subs] == ["Testing the AMD/Intel path."]
    assert subs[0]["start"] == 0.0 and subs[0]["end"] == 2.88

    done = lines[-1]
    assert done == {"type": "done", "segments": 1, "srt_path": str(wanted)}
    assert "Testing the AMD/Intel path." in wanted.read_text(encoding="utf-8")
    assert not (tmp_path / "clip.srt").exists()   # no side-effect SRT


def test_main_writes_no_srt_without_a_requested_path(runner, monkeypatch, tmp_path, capsys):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    _run(runner, monkeypatch, tmp_path,
         ["runner", str(media), "small", "en", "transcribe"])

    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert lines[-1] == {"type": "done", "segments": 1, "srt_path": None}
    assert not (tmp_path / "clip.srt").exists()


def test_main_reports_cpu_when_disabled(runner, monkeypatch, tmp_path, capsys):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    monkeypatch.setenv("VSCL_AISUBS_DEVICE", "cpu")
    try:
        _run(runner, monkeypatch, tmp_path,
             ["runner", str(media), "small", "en", "transcribe"])
        lines = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
        msgs = [l.get("msg", "") for l in lines if l["type"] == "status"]
        assert any("GPU disabled" in m for m in msgs)
        assert not any("Fake Radeon" in m for m in msgs)
    finally:
        os.environ.pop("VSCL_AISUBS_DEVICE", None)


def test_main_errors_without_the_binary(runner, monkeypatch, tmp_path, capsys):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    monkeypatch.setenv("VSCL_AISUBS_WHISPERCPP_BIN", str(tmp_path / "nope"))
    monkeypatch.setattr(runner, "BINARY_CANDIDATES", ())
    monkeypatch.setattr(runner, "_conf_value", lambda _k: None)
    # Patch the lookup inside the runner, not shutil.which itself: core.audio
    # (ffmpeg discovery) shares that function.
    monkeypatch.setattr(runner.shutil, "which", lambda name: None if name == "whisper-cli" else "/usr/bin/ffmpeg")
    monkeypatch.setattr(sys, "argv", ["runner", str(media), "small", "en", "transcribe"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert "install-whisper-cpp.sh" in capsys.readouterr().out
