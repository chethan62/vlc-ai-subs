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
        encoder = os.path.join(model_dir, "encoder.onnx") or os.path.join(model_dir, "encoder.int8.onnx")
        decoder = os.path.join(model_dir, "decoder.onnx") or os.path.join(model_dir, "decoder.int8.onnx")
        tokens = os.path.join(model_dir, "tokens.txt")

        # Also support non-Whisper models (single model.onnx + tokens.txt)
        single_model = os.path.join(model_dir, "model.onnx")
        single_tokens = os.path.join(model_dir, "tokens.txt")

        conf = sherpa_onnx.OfflineRecognizerConfig()

        if os.path.isfile(encoder) and os.path.isfile(decoder) and os.path.isfile(tokens):
            # Whisper-style (encoder+decoder)
            conf.model_config.whisper = sherpa_onnx.OfflineWhisperModelConfig(
                encoder=encoder,
                decoder=decoder,
                tokens=tokens,
            )
        elif os.path.isfile(single_model) and os.path.isfile(single_tokens):
            # Generic: try SenseVoice / Paraformer / Qwen3
            # Fall back to sense_voice config (the most common single-model format)
            conf.model_config.sense_voice = sherpa_onnx.OfflineSenseVoiceModelConfig(
                model=single_model,
                tokens=single_tokens,
            )
        else:
            raise RuntimeError(
                f"No recognisable model layout in {model_dir}. "
                f"Expected Whisper (encoder.onnx+decoder.onnx+tokens.txt) or "
                f"generic (model.onnx+tokens.txt)."
            )

        # Language and task for Whisper
        if hasattr(conf.model_config, 'whisper') and conf.model_config.whisper is not None:
            if language:
                conf.model_config.whisper.language = language
            if task:
                conf.model_config.whisper.task = task

        recognizer = sherpa_onnx.OfflineRecognizer(conf)
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
