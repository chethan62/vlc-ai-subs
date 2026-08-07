"""
whisper.cpp backend (Vulkan/CPU via subprocess).

Uses the ggml-org/whisper.cpp CLI binary (whisper-cli) installed by
``install-whisper-cpp.sh``.  Auto-detects Vulkan at the ggml level — the
binary loads libggml-vulkan.so when the shared library is in its directory.
"""

import json
import os
import subprocess
from typing import Iterable

from .base import TranscriptionBackend

_CONFIG_PATH = os.path.expanduser("~/.local/share/whisper-cpp/vlc-ai-subs.conf")


class WhisperCppBackend(TranscriptionBackend):
    """Transcribe via the whisper.cpp CLI — zero Python deps beyond stdlib."""

    def __init__(self, binary: str, model_path: str):
        self.binary = binary
        self.model_path = model_path

    @classmethod
    def detect(cls) -> "WhisperCppBackend | None":
        """Load config written by install-whisper-cpp.sh, return instance or None."""
        if not os.path.isfile(_CONFIG_PATH):
            return None
        cfg: dict[str, str] = {}
        try:
            with open(_CONFIG_PATH) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, _, v = line.partition("=")
                        cfg[k.strip()] = v.strip()
        except OSError:
            return None
        binary = cfg.get("whisper_bin", "")
        model = cfg.get("whisper_model", "")
        if not binary or not os.path.isfile(binary) or not os.path.isfile(model):
            return None
        return cls(binary, model)

    def transcribe(
        self, media_path: str, model_name: str, language: str | None, task: str
    ) -> Iterable[dict]:
        cmd = [self.binary, "-m", self.model_path, "-f", media_path, "-oj", "-of", "-"]
        if language:
            cmd += ["-l", language]
        if task == "translate":
            cmd += ["-tr"]

        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=None,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "whisper.cpp failed (rc={}): {}".format(
                    proc.returncode, (proc.stderr or "").strip()[:500]
                )
            )

        try:
            data = json.loads(proc.stdout)
        except ValueError:
            raise RuntimeError("whisper.cpp: could not parse JSON output")

        for seg in data.get("transcription", []):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            off = seg.get("offsets", {})
            yield {
                "start": (off.get("from") or 0) / 1000.0,
                "end": (off.get("to") or 0) / 1000.0,
                "text": text,
            }

    @staticmethod
    def name() -> str:
        return "whisper.cpp"
