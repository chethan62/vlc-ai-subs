"""
openai-whisper backend (CPU-only fallback).

Used only when neither whisper.cpp nor faster-whisper is available.
Batch-transcribes the whole file and returns segments at once.
"""

from typing import Iterable

from .base import TranscriptionBackend


class OpenAIWhisperBackend(TranscriptionBackend):
    """Transcribe via openai-whisper (the original PyTorch backend)."""

    def transcribe(
        self, media_path: str, model_name: str, language: str | None, task: str
    ) -> Iterable[dict]:
        import whisper

        model = whisper.load_model(model_name)

        options = {"task": task}
        if language:
            options["language"] = language
        result = model.transcribe(media_path, **options)

        for seg in result["segments"]:
            text = seg["text"].strip()
            if text:
                yield {"start": seg["start"], "end": seg["end"], "text": text}

    @staticmethod
    def name() -> str:
        return "openai-whisper"
