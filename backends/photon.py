"""Photon backend — Moondream's Parakeet Redux (ternary Parakeet v3, 178 MB).

Explicitly selected (`VSCL_AISUBS_BACKEND=photon`); never chosen by the hardware
policy, because what it needs is a *choice* about device and licence rather than
a hardware fact: the weights are CC-BY-4.0 but the engine that runs them is
proprietary, and it is the fastest engine here only on CPU (see
`core/photon_models.py` for the measured numbers and the licence note).

Unlike the other backends this one does NOT run under the CLI's interpreter: it
spawns `venv-photon/bin/python`, because Photon needs PyTorch — and it does not
borrow `venv-whisperx`, where a pip resolve could move the torch WhisperX
depends on.
"""

import os
import sys
from typing import Iterable

from core.photon_models import installed, model_label as photon_model_label, venv_python
from core.procs import run_captured
from core.timeouts import resolve_timeout

from .base import TranscriptionBackend, check_runner_completed

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNNER = os.path.join(_BASE, "photon_runner.py")
if not os.path.isfile(_RUNNER):
    _RUNNER = os.path.expanduser("~/.local/share/vlc-ai-subs/photon_runner.py")


class PhotonBackend(TranscriptionBackend):
    """Transcribe with Moondream's Parakeet Redux."""

    @classmethod
    def detect(cls) -> "PhotonBackend | None":
        """Available when the venv has moondream and this runner is reachable."""
        if installed() and os.path.isfile(_RUNNER):
            return cls()
        return None

    def transcribe(
        self, media_path: str, model_name: str, language: str | None, task: str
    ) -> Iterable[dict]:
        python = venv_python()
        if not python:
            raise RuntimeError(
                "Photon venv is missing. Install it with:\n  ./install-photon-model.sh"
            )
        cmd = [
            python, "-u", _RUNNER,
            media_path, model_name, language or "auto", task,
        ]
        env = {**os.environ, "PYTHONPATH": ""}
        proc = run_captured(cmd, timeout=resolve_timeout(task), env=env)
        debug = env.get("VSCL_AISUBS_DEBUG") == "1"

        # The runner's own error event is read before its exit code is judged: it
        # always emits one, and checking returncode first threw that message away
        # in favour of a bare "rc=1" (the same fix CrispASR needed).
        import json
        saw_done = False
        for line in proc.stdout.strip().splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            kind = obj.get("type")
            if kind == "sub":
                yield {"start": obj["start"], "end": obj["end"], "text": obj["text"]}
            elif kind == "error":
                raise RuntimeError(obj.get("msg", "Photon error"))
            elif kind == "done":
                saw_done = True
            elif kind == "status" and debug:
                sys.stderr.write(f"[photon] {obj.get('msg', '')}\n")
        check_runner_completed(proc, "Photon", saw_done)

    def name(self) -> str:
        return "photon (Parakeet Redux)"

    def model_label(self, requested: str, language: str | None = None) -> str | None:
        """The variant that will actually load (VSCL_AISUBS_PHOTON_MODEL)."""
        return photon_model_label()
