"""whisper.cpp backend — the Vulkan GPU path (AMD / Intel / NVIDIA) plus CPU.

Why this engine exists: WhisperX (faster-whisper/CTranslate2) only accelerates
on NVIDIA CUDA, so **AMD and Intel GPUs get CPU speed** from it. whisper.cpp is
the only Whisper runtime with a Vulkan backend, and Vulkan is vendor-neutral —
the same binary drives Radeon, Arc and GeForce GPUs and silently falls back to
CPU when no device is present.

The runner is stdlib-only (it shells out to whisper-cli + ffmpeg), so this
backend needs no Python ML packages and runs under the CLI's own interpreter:
unlike WhisperX/Parakeet it does not require `venv-whisperx`.

Trade-off vs WhisperX: segment-level timestamps only (no wav2vec2 aligner), and
translate uses Whisper's built-in `-tr` rather than the NLLB cascade.
"""

import os
import sys
from typing import Iterable

from core.procs import run_captured
from core.timeouts import resolve_timeout

from .base import TranscriptionBackend

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNNER = os.path.join(_BASE, "whispercpp_runner.py")

if not os.path.isfile(_RUNNER):
    _RUNNER = os.path.expanduser("~/.local/share/vlc-ai-subs/whispercpp_runner.py")

# Installer config + conventional locations, mirroring whispercpp_runner.
_SHARE = os.path.expanduser("~/.local/share/whisper-cpp")
_BINARY_CANDIDATES = (
    os.path.join(_SHARE, "whisper-cli"),
    os.path.expanduser("~/.local/bin/whisper-cli"),
    "/usr/local/bin/whisper-cli",
    "/usr/bin/whisper-cli",
)


def runner_module():
    """Load whispercpp_runner by path (single source of truth for its tables)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("whispercpp_runner_shared", _RUNNER)
    if spec is None or spec.loader is None:  # pragma: no cover — path checked above
        raise RuntimeError(f"cannot load {_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _conf_binary() -> str | None:
    conf = os.path.join(_SHARE, "vlc-ai-subs.conf")
    try:
        with open(conf, encoding="utf-8") as f:
            for line in f:
                name, _, value = line.partition("=")
                if name.strip() == "whisper_bin" and value.strip():
                    return os.path.expanduser(value.strip())
    except OSError:
        pass
    return None


def binary_path() -> str | None:
    """whisper-cli path: env override → installer config → standard locations."""
    env_bin = os.environ.get("VSCL_AISUBS_WHISPERCPP_BIN", "").strip()
    if env_bin:
        path = os.path.expanduser(env_bin)
        return path if os.path.isfile(path) else None
    for candidate in (_conf_binary(), *_BINARY_CANDIDATES):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    import shutil

    found = shutil.which("whisper-cli")
    return found


def has_vulkan(binary: str | None = None) -> bool:
    """True when the whisper.cpp build carries its Vulkan backend.

    The installer keeps binaries and libs side by side, so the sibling
    libggml-vulkan(.so) is the cheap, reliable tell.
    """
    binary = binary or binary_path()
    if not binary:
        return False
    directory = os.path.dirname(binary)
    try:
        names = os.listdir(directory)
    except OSError:
        return False
    return any(n.startswith("libggml-vulkan") or n.startswith("ggml-vulkan.dll")
               for n in names)


class WhisperCppBackend(TranscriptionBackend):
    """Transcribe with whisper.cpp — Vulkan (AMD/Intel/NVIDIA) or CPU."""

    def __init__(self, binary: str | None = None, vulkan: bool | None = None) -> None:
        self._binary = binary or binary_path()
        self._vulkan = has_vulkan(self._binary) if vulkan is None else vulkan

    @classmethod
    def detect(cls) -> "WhisperCppBackend | None":
        """Available when the whisper-cli binary and the runner are installed."""
        binary = binary_path()
        if binary and os.path.isfile(binary) and os.path.isfile(_RUNNER):
            return cls(binary)
        return None

    def model_label(self, requested: str) -> str | None:
        """Model that will really run: whisper.cpp resolves to an installed
        ggml file (the dialog's WhisperX size names may not all be installed)."""
        if not self._binary:
            return None
        try:
            _, name = runner_module().resolve_model(
                requested, runner_module().model_dirs(self._binary)
            )
            return name
        except Exception:
            return None  # resolution failure surfaces as the runner's error

    def transcribe(
        self, media_path: str, model_name: str, language: str | None, task: str
    ) -> Iterable[dict]:
        # sys.executable: the runner is stdlib-only, so no ML venv is needed.
        cmd = [
            sys.executable, "-u", _RUNNER,
            media_path, model_name, language or "auto", task,
        ]
        env = {**os.environ, "PYTHONPATH": ""}
        debug = env.get("VSCL_AISUBS_DEBUG") == "1"
        proc = run_captured(cmd, timeout=resolve_timeout(task), env=env)
        if debug:
            import sys as _sys
            try:
                with open("/tmp/aisubs_whispercpp.log", "a", encoding="utf-8") as f:
                    f.write(f"--- run: {media_path} {model_name} ---\n")
                    f.write("STDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr + "\n")
            except OSError:
                pass
            for line in (proc.stdout + proc.stderr).splitlines():
                _sys.stderr.write(f"[whispercpp] {line}\n")

        if proc.returncode != 0:
            tail = (
                "stdout: " + proc.stdout.strip().splitlines()[-1][:300]
                if proc.stdout.strip() else ""
            )
            raise RuntimeError(
                "whisper.cpp failed (rc={}): {}{}".format(
                    proc.returncode, (proc.stderr or "").strip()[-500:],
                    "\n" + tail if tail else "",
                )
            )

        import json
        for line in proc.stdout.strip().splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("type") == "sub":
                yield {"start": obj["start"], "end": obj["end"], "text": obj["text"]}
            elif obj.get("type") == "error":
                raise RuntimeError(obj.get("msg", "whisper.cpp error"))
            elif obj.get("type") == "status" and debug:
                import sys as _sys
                _sys.stderr.write(f"[whispercpp] {obj.get('msg', '')}\n")

    def name(self) -> str:
        return "whisper.cpp (vulkan)" if self._vulkan else "whisper.cpp (cpu)"
