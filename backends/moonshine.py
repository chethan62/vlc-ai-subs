"""
Moonshine backend — ultra-fast ONNX-based STT for quick preview subtitles.

Moonshine (useful-inc / moonshine-ai) is a tiny speech-to-text model designed
for edge devices.  ~100× faster than Whisper base, good enough for quick
rough-draft subtitles or realtime captioning inside VLC.

Install:  pip install moonshine-voice

Trade-off: lower accuracy than Whisper, but near-instant transcription.
"""

from typing import Iterable

from .base import TranscriptionBackend


class MoonshineBackend(TranscriptionBackend):
    """Transcribe via Moonshine — fastest option, smallest footprint."""

    @classmethod
    def detect(cls) -> "MoonshineBackend | None":
        """Return an instance if moonshine_voice is importable, else None."""
        try:
            import moonshine_voice  # noqa: F401
            return cls()
        except ImportError:
            return None

    def transcribe(
        self, media_path: str, model_name: str, language: str | None, task: str
    ) -> Iterable[dict]:
        import moonshine_voice

        # Moonshine models: tiny, base. Map Whisper names down.
        ms_model = model_name
        if model_name in ("small", "medium", "large"):
            ms_model = "base"

        try:
            transcriber = moonshine_voice.Transcriber(model=ms_model)
            result = transcriber.transcribe(media_path)
        except Exception:
            # Fallback: try the lower-level API
            model = moonshine_voice.get_model_for_language(language or "en")
            result = model.transcribe(media_path)

        # Moonshine returns flat text with optional timestamps
        text = ""
        if isinstance(result, str):
            text = result.strip()
        elif isinstance(result, dict):
            text = (result.get("text") or "").strip()
        else:
            text = str(result).strip()

        if text:
            yield {"start": 0, "end": result.get("duration", 1) if isinstance(result, dict) else 1, "text": text}

    @staticmethod
    def name() -> str:
        return "moonshine (fast)"
