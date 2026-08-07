"""
faster-whisper backend (CUDA/CPU via ctranslate2).

Supports CUDA, int8_float16 for small VRAM, and CPU fallback.
CUDA libraries are preloaded by core.device so this works without
LD_LIBRARY_PATH (needed when VLC launches the backend).
"""

import os
from typing import Iterable

from core.device import detect_device
from .base import TranscriptionBackend


class FasterWhisperBackend(TranscriptionBackend):
    """Transcribe via faster-whisper (CTranslate2)."""

    def transcribe(
        self, media_path: str, model_name: str, language: str | None, task: str
    ) -> Iterable[dict]:
        from faster_whisper import WhisperModel

        device, compute = detect_device()

        kw: dict = dict(device=device, compute_type=compute)
        model_cache = os.environ.get("VSCL_AISUBS_MODEL_CACHE", "").strip()
        if model_cache:
            kw["download_root"] = model_cache

        model = WhisperModel(model_name, **kw)

        beam_size = 5 if device == "cuda" else 1
        segments_gen, _info = model.transcribe(
            media_path,
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.05,
                "min_silence_duration_ms": 200,
                "speech_pad_ms": 600,
                "min_speech_duration_ms": 50,
            },
        )

        for seg in segments_gen:
            text = seg.text.strip()
            if text:
                yield {"start": seg.start, "end": seg.end, "text": text}

    @staticmethod
    def name() -> str:
        return "faster-whisper"
