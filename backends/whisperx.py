"""
WhisperX backend — word-level alignment + speaker diarization.

Wraps m-bain/whisperX to get:
  • word-level timestamps via wav2vec2 forced alignment
  • speaker labels (optional, GPU-only) via pyannote + spectral clustering
  • tighter segment boundaries than raw whisper

The alignment step adds ~300 MB VRAM (wav2vec2 model) and runs on CUDA;
falls back to segment-level output on CPU.

Install:  pip install whisperx  (needs CUDA toolkit for GPU alignment)
"""

import os
from typing import Iterable

from .base import TranscriptionBackend


class WhisperXBackend(TranscriptionBackend):
    """Transcribe with WhisperX — word-aligned segments, optional speakers."""

    def __init__(self, align: bool = True, diarize: bool = False):
        self._align = align
        self._diarize = diarize

    @classmethod
    def detect(cls) -> "WhisperXBackend | None":
        """Return an instance if whisperx is importable, else None."""
        try:
            import whisperx  # noqa: F401
            return cls(align=True, diarize=False)
        except ImportError:
            return None

    def transcribe(
        self, media_path: str, model_name: str, language: str | None, task: str
    ) -> Iterable[dict]:
        import whisperx

        # Resolve device (reuse the core.device preload path)
        device = "cuda"
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() == 0:
                device = "cpu"
        except Exception:
            device = "cpu"

        compute = "int8_float16" if device == "cuda" else "float32"

        # 1. Transcribe with faster-whisper (bundled in whisperx)
        model = whisperx.load_model(
            model_name, device, compute_type=compute,
        )
        result = model.transcribe(
            media_path,
            language=language,
            task=task,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.05,
                "min_silence_duration_ms": 200,
                "speech_pad_ms": 600,
                "min_speech_duration_ms": 50,
            },
        )

        # 2. Align (word-level timestamps) if requested and audio is available
        if self._align and device == "cuda" and result.get("segments"):
            try:
                align_model, metadata = whisperx.load_align_model(
                    language_code=result.get("language", language or "en"),
                    device=device,
                )
                result = whisperx.align(
                    result["segments"],
                    align_model, metadata,
                    media_path, device,
                    return_char_alignments=False,
                )
            except Exception:
                # Alignment failed — fall back to segment-level output
                pass

        # 3. Diarization (speaker labels) — optional, GPU only
        if self._diarize and device == "cuda":
            try:
                import torch
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=False,
                    device=device if torch.cuda.is_available() else "cpu",
                )
                diarize_segments = diarize_model(media_path)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception:
                pass  # diarization is a best-effort extra

        # 4. Yield segments (word-aligned if possible)
        for seg in result.get("segments", []):
            text = (seg.get("text") or "").strip()
            if not text:
                continue

            # Speaker suffix if diarized
            speaker = seg.get("speaker")
            if speaker:
                text = f"[{speaker}] {text}"

            yield {
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": text,
            }

            # If word-level timestamps are present, emit individual words
            # as ultra-fine-grained subtitles (Kdenlive-compatible)
            # disabled by default — enable via env VSCL_AISUBS_WORDSUB=1
            if os.environ.get("VSCL_AISUBS_WORDSUB", "").strip() == "1":
                for w in seg.get("words", []):
                    wt = (w.get("word") or "").strip()
                    if wt:
                        yield {
                            "start": w.get("start", seg.get("start", 0)),
                            "end": w.get("end", seg.get("end", 0)),
                            "text": wt,
                        }

    @staticmethod
    def name() -> str:
        return "whisperx (aligned)"
