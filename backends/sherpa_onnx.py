"""
sherpa-onnx backend — ONNX-runtime Whisper (and friends).

sherpa-onnx (k2-fsa) runs Whisper ONNX models via onnxruntime with CPU/CUDA
providers.  It supports the widest range of model architectures of any STT
framework: Whisper, Moonshine, SenseVoice, Paraformer, Qwen3-ASR, and more.

Install:  pip install sherpa-onnx soundfile
Models:   Download from https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
          or run ./install-sherpa-onnx-model.sh <model>

Model types are detected from the file layout (encoder.onnx / decoder.onnx /
tokens.txt = Whisper; single model.onnx + tokens = Paraformer/SenseVoice).

Timestamps come from the recognizer's token-level output, grouped into
sentence-like subtitle segments.
"""

import os
from typing import Iterable

from .base import TranscriptionBackend

_MODEL_CACHE = os.path.expanduser("~/.local/share/sherpa-onnx/models")


class SherpaOnnxBackend(TranscriptionBackend):
    """Sherpa-onnx offline Whisper (and other) transcription."""

    def __init__(self, model_dir: str):
        self._model_dir = model_dir

    @classmethod
    def detect(cls) -> "SherpaOnnxBackend | None":
        """Return instance if sherpa-onnx is importable AND a model is cached."""
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            return None
        # Check for any cached model
        if not os.path.isdir(_MODEL_CACHE):
            return None
        for entry in os.scandir(_MODEL_CACHE):
            if entry.is_dir():
                return cls(entry.path)
        return None

    def transcribe(
        self, media_path: str, model_name: str, language: str | None, task: str
    ) -> Iterable[dict]:
        import sherpa_onnx
        import soundfile as sf

        model_dir = self._model_dir

        # Whisper (encoder+decoder+tokens)
        encoder = os.path.join(model_dir, "encoder.int8.onnx")
        if not os.path.isfile(encoder):
            encoder = os.path.join(model_dir, "encoder.onnx")
        decoder = os.path.join(model_dir, "decoder.int8.onnx")
        if not os.path.isfile(decoder):
            decoder = os.path.join(model_dir, "decoder.onnx")
        tokens = os.path.join(model_dir, "tokens.txt")

        # SenseVoice / Paraformer (single model.onnx + tokens.txt)
        single_model = os.path.join(model_dir, "model.onnx")
        single_tokens = os.path.join(model_dir, "tokens.txt")

        if os.path.isfile(encoder) and os.path.isfile(decoder) and os.path.isfile(tokens):
            recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=encoder,
                decoder=decoder,
                tokens=tokens,
                model_type=model_name,  # e.g. "small", "base"
                language=language or "",
                task=task,
            )
        elif os.path.isfile(single_model) and os.path.isfile(single_tokens):
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=single_model,
                tokens=single_tokens,
                use_itn=True,
            )
        else:
            raise RuntimeError(
                f"No recognisable model layout in {model_dir}. "
                f"Expected Whisper (encoder.onnx+decoder.onnx+tokens.txt) or "
                f"SenseVoice/Paraformer (model.onnx+tokens.txt)."
            )

        stream = recognizer.create_stream()
        audio, sample_rate = sf.read(media_path, dtype="float32", always_2d=True)
        audio = audio[:, 0]  # mono
        stream.accept_waveform(sample_rate, audio)
        recognizer.decode_stream(stream)

        # Build segments from token-level timestamps
        text = stream.result.text.strip()
        tokens = stream.result.tokens if hasattr(stream.result, 'tokens') else []

        if tokens:
            # Group tokens into sentence-like segments
            self._yield_from_tokens(tokens)
        elif text:
            # Fall back to a single segment
            yield {"start": 0, "end": len(audio) / sample_rate, "text": text}

    def _yield_from_tokens(self, tokens):
        """Group timestamped tokens into subtitle segments."""
        current_text = []
        current_start = None
        current_end = 0

        for tok in tokens:
            if not tok.text.strip():
                continue
            if current_start is None:
                current_start = tok.start
            current_end = tok.end
            current_text.append(tok.text)

            # Sentence boundary: period, question, exclamation, or max length
            joined = "".join(current_text)
            if tok.text.rstrip().endswith((".", "?", "!", "。", "？", "！")):
                yield {
                    "start": current_start,
                    "end": current_end,
                    "text": joined.strip(),
                }
                current_text = []
                current_start = None

        # Flush remaining
        if current_text:
            yield {
                "start": current_start or 0,
                "end": current_end,
                "text": "".join(current_text).strip(),
            }

    @staticmethod
    def name() -> str:
        return "sherpa-onnx"
