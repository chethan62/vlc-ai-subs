"""
Moonshine backend — ultra-fast ONNX-based STT for quick preview subtitles.

Moonshine (useful-inc / moonshine-ai) is a tiny streaming speech-to-text model
designed for edge devices.  ~100× faster than Whisper base, good enough for
quick rough-draft subtitles or realtime captioning inside VLC.

Install:  pip install moonshine-voice

Trade-off: lower accuracy than Whisper, but near-instant transcription.
"""

from typing import Iterable

from .base import TranscriptionBackend


class MoonshineBackend(TranscriptionBackend):
    """Transcribe via Moonshine-voice — fastest option, smallest footprint."""

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
        import soundfile as sf

        # Resolve model: map Whisper model names to Moonshine arch
        model_path, model_arch = moonshine_voice.get_model_for_language(
            language or "en"
        )

        transcriber = moonshine_voice.Transcriber(
            model_path=model_path,
            model_arch=model_arch,
        )

        # Load audio as float32 mono at 16 kHz (Moonshine's native rate)
        audio, sr = sf.read(media_path, dtype="float32", always_2d=True)
        audio = audio[:, 0].tolist()  # mono, Python list

        transcript = transcriber.transcribe_without_streaming(audio, sr)

        # transcript.lines is a list of TranscriptLine objects
        lines = getattr(transcript, "lines", [])
        if not lines:
            text = getattr(transcript, "text", "") or str(transcript)
            if text.strip():
                yield {"start": 0, "end": len(audio) / sr, "text": str(text).strip()}
            return

        for line in lines:
            text = (getattr(line, "text", "") or "").strip()
            if not text:
                continue
            start_ms = getattr(line, "start_ms", 0) or 0
            end_ms = getattr(line, "end_ms", 0) or start_ms + 3000
            yield {
                "start": start_ms / 1000.0,
                "end": end_ms / 1000.0,
                "text": text,
            }

    @staticmethod
    def name() -> str:
        return "moonshine (fast)"
