"""Parakeet backend — NVIDIA Parakeet-TDT-0.6B-v2 (English) via sherpa-onnx.

Runs inside the shared Python 3.12 venv (`venv-whisperx`), same subprocess
+ JSONL pattern as whisperx_backend. Selected explicitly via
VSCL_AISUBS_BACKEND=parakeet — WhisperX remains the default engine.

2026-08 research: WER 6.05 self-reported (whisper-large-v3 7.44 on a
comparable English eval), native word-level timestamps, ~0.7GB int8,
CC-BY-4.0, transducer = no hallucination loops.
"""

import os
from typing import Iterable

from core.parakeet_models import model_label as parakeet_model_label
from core.procs import run_captured
from core.timeouts import resolve_timeout

from .base import TranscriptionBackend

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNNER = os.path.join(_BASE, "parakeet_runner.py")
_VENV = os.path.join(_BASE, "venv-whisperx")

# The engines this backend can run (English v2 / 25-language v3). The runner
# picks the variant from what is installed; the label therefore comes from
# core/parakeet_models.py so it can never disagree with the runner's own
# status line (see parakeet_runner.py, which imports the same module).

if not (os.path.isfile(_RUNNER) and os.path.isdir(_VENV)):
    _SHARE = os.path.expanduser("~/.local/share/vlc-ai-subs")
    _RUNNER = os.path.join(_SHARE, "parakeet_runner.py")
    _VENV = os.path.join(_SHARE, "venv-whisperx")

_PYTHON = None
for _cand in (
    os.path.join(_VENV, "bin", "python3"),
    os.path.join(_VENV, "bin", "python"),
    os.path.join(_VENV, "Scripts", "python.exe"),
):
    if os.path.isfile(_cand):
        _PYTHON = _cand
        break


class ParakeetBackend(TranscriptionBackend):
    """Transcribe with Parakeet-TDT-0.6B-v2 — English, CPU, word timestamps."""

    def __init__(self) -> None:
        pass

    @classmethod
    def detect(cls) -> "ParakeetBackend | None":
        """Instance exists if runner + 3.12 venv are installed."""
        if _PYTHON and os.path.isfile(_PYTHON) and os.path.isfile(_RUNNER):
            return cls()
        return None

    def transcribe(
        self, media_path: str, model_name: str, language: str | None, task: str
    ) -> Iterable[dict]:
        cmd = [
            _PYTHON, "-u", _RUNNER,
            media_path, model_name, language or "auto", task,
        ]
        env = {**os.environ, "PYTHONPATH": ""}
        debug = env.get("VSCL_AISUBS_DEBUG") == "1"
        proc = run_captured(cmd, timeout=resolve_timeout(task), env=env)
        if debug:
            import sys as _sys
            try:
                with open("/tmp/aisubs_parakeet.log", "a", encoding="utf-8") as f:
                    f.write(f"--- run: {media_path} {model_name} ---\n")
                    f.write("STDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr + "\n")
            except OSError:
                pass
            for _l in (proc.stdout + proc.stderr).splitlines():
                _sys.stderr.write(f"[parakeet] {_l}\n")

        if proc.returncode != 0:
            tail = (
                "stdout: " + proc.stdout.strip().splitlines()[-1][:300]
                if proc.stdout.strip() else ""
            )
            raise RuntimeError(
                "Parakeet failed (rc={}): {}{}".format(
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
                raise RuntimeError(obj.get("msg", "Parakeet error"))
            elif obj.get("type") == "status" and debug:
                import sys as _sys
                _sys.stderr.write(f"[parakeet] {obj.get('msg', '')}\n")

    def name(self) -> str:
        return "parakeet (fast)"

    def model_label(self, requested: str, language: str | None = None) -> str | None:
        """Parakeet ignores the <model> arg — it runs an installed variant.

        The label comes from core/parakeet_models.py, shared with the runner, so
        the CLI's status line and the runner's own status line agree. `language`
        selects the variant: English → v2, other languages → v3.
        """
        return parakeet_model_label(language)