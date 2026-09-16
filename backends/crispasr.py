"""CrispASR backend — one C++ ggml binary, many models, optional CTC alignment.

Explicitly selected (`VSCL_AISUBS_BACKEND=crispasr`); it is never chosen by the
hardware policy, because which model to run there depends on VRAM and the user
should decide when to spend a GPU on it. See
`core/crispasr_models.py` for the tiering and
`.research/2026-09-16-asr-landscape-and-crispasr.md` for the measurements that
justify it (notably: on the test film it transcribes dialogue the sherpa-onnx
ONNX-int8 path silently drops, at 4.3x realtime on 8 CPU threads).
"""

import os
import sys
from typing import Iterable

from core.crispasr_models import installed, model_label as crispasr_model_label
from core.procs import run_captured
from core.timeouts import resolve_timeout

from .base import TranscriptionBackend, check_runner_completed

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNNER = os.path.join(_BASE, "crispasr_runner.py")
if not os.path.isfile(_RUNNER):
    _RUNNER = os.path.expanduser("~/.local/share/vlc-ai-subs/crispasr_runner.py")


class CrispAsrBackend(TranscriptionBackend):
    """Transcribe with the CrispASR binary (Parakeet/Cohere/Canary/… by VRAM)."""

    @classmethod
    def detect(cls) -> "CrispAsrBackend | None":
        """Available when the binary is installed and the runner is reachable.

        The runner needs no third-party Python package — it imports only this
        project's own modules — so the interpreter running the CLI is enough.
        """
        if installed() and os.path.isfile(_RUNNER):
            return cls()
        return None

    def transcribe(
        self, media_path: str, model_name: str, language: str | None, task: str
    ) -> Iterable[dict]:
        cmd = [
            sys.executable, "-u", _RUNNER,
            media_path, model_name, language or "auto", task,
        ]
        env = {**os.environ, "PYTHONPATH": ""}
        proc = run_captured(cmd, timeout=resolve_timeout(task), env=env)
        debug = env.get("VSCL_AISUBS_DEBUG") == "1"

        if proc.returncode != 0:
            raise RuntimeError(
                "CrispASR failed (rc={}): {}".format(
                    proc.returncode, (proc.stderr or "").strip()[-500:],
                )
            )

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
                # The runner reports actionable errors (not installed, translate
                # unsupported) as events, not exit codes — surface the message.
                raise RuntimeError(obj.get("msg", "CrispASR error"))
            elif kind == "done":
                saw_done = True
            elif kind == "status" and debug:
                sys.stderr.write(f"[crispasr] {obj.get('msg', '')}\n")
        check_runner_completed(proc, "CrispASR", saw_done)

    def name(self) -> str:
        return "crispasr (ggml)"

    def model_label(self, requested: str, language: str | None = None) -> str | None:
        """The model the runner will pick for this language and this machine's VRAM.

        Resolved from the same table the runner uses, so the CLI's status line
        cannot name a model that is not the one loading — the Parakeet backend's
        rule, applied to an engine with a real model menu.
        """
        return crispasr_model_label(language)
