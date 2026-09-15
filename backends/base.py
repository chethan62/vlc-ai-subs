"""
Abstract backend interface.

Every backend yields a sequence of ``{start: float, end: float, text: str}``
dictionaries. The caller handles SRT writing and JSONL emission.
"""

import abc
from typing import Iterable


class TranscriptionBackend(abc.ABC):
    """Transcribe a media file and yield timestamped text segments."""

    @abc.abstractmethod
    def name(self) -> str:
        """Short label for status lines, e.g. ``"whisperx (aligned)"``."""
        ...

    def model_label(self, requested: str, language: str | None = None) -> str | None:
        """Name of the model that will actually run, or None for "no opinion".

        Engines that ignore the dialog's pick (Parakeet: an installed variant
        whose coverage depends on the language; whisper.cpp: an installed ggml
        file) report their own label so a status line never names a model that
        is not being used. `language` matters for Parakeet: English runs v2
        while the other languages run v3. None means the caller falls back to
        the hardware-aware picker.
        """
        return None

    @abc.abstractmethod
    def transcribe(
        self,
        media_path: str,
        model_name: str,
        language: str | None,
        task: str,
    ) -> Iterable[dict]:
        """Yield ``{start, end, text}`` dicts.  Can be a generator."""
        ...
