"""
WhisperX backend — word-level alignment via separate Python 3.12 venv.

WhisperX requires Python <3.14, so it runs in its own venv (created by
`uv venv --python 3.12 venv-whisperx && uv pip install whisperx`).
The plugin calls `whisperx_runner.py` as a subprocess — same contract
as the main JSONL interface.
"""

import os
import subprocess
from typing import Iterable

from .base import TranscriptionBackend

_RUNNER = os.path.expanduser("~/.local/share/vlc-ai-subs/whisperx_runner.py")
_VENV = os.path.expanduser("~/.local/share/vlc-ai-subs/venv-whisperx")
_PYTHON = os.path.join(_VENV, "bin", "python3") if os.path.isdir(_VENV) else os.path.join(_VENV, "bin", "python")


class WhisperXBackend(TranscriptionBackend):
    """Transcribe with WhisperX — word-aligned segments, optional speakers."""

    def __init__(self, align: bool = True):
        self._align = align

    @classmethod
    def detect(cls) -> "WhisperXBackend | None":
        """Return instance if the Python 3.12 venv + runner script exist."""
        if os.path.isfile(_PYTHON) and os.path.isfile(_RUNNER):
            return cls(align=True)
        return None

    def transcribe(
        self, media_path: str, model_name: str, language: str | None, task: str
    ) -> Iterable[dict]:
        cmd = [
            _PYTHON, "-u", _RUNNER,
            media_path, model_name, language or "auto", task,
        ]
        # Run with clean PYTHONPATH — the backends/ directory contains modules
        # (faster_whisper.py, moonshine.py) that shadow pip-installed packages.
        env = {**os.environ, "PYTHONPATH": ""}
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1200, env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "WhisperX failed (rc={}): {}".format(
                    proc.returncode, (proc.stderr or "").strip()[:500]
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
                raise RuntimeError(obj.get("msg", "WhisperX error"))

    @staticmethod
    def name() -> str:
        return "whisperx (aligned)"
