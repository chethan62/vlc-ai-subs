"""
Backend registry — auto-detects the best available backend.

Priority: whisper.cpp (Vulkan) → WhisperX (word-aligned) → faster-whisper
(CUDA) → Moonshine (fast preview) → openai-whisper (CPU).

Override with:  VSCL_AISUBS_BACKEND=whisper_cpp|whisperx|faster_whisper|moonshine
"""

import logging
import os

logger = logging.getLogger(__name__)


def resolve_backend() -> "TranscriptionBackend":
    """Return the best available TranscriptionBackend instance."""
    forced = os.environ.get("VSCL_AISUBS_BACKEND", "").strip().lower()

    # ── whisper.cpp (Vulkan) ──
    if not forced or forced == "whisper_cpp":
        try:
            from backends.whisper_cpp import WhisperCppBackend
            be = WhisperCppBackend.detect()
            if be:
                return be
        except Exception as exc:
            logger.debug("whisper.cpp: %s", exc)

    # ── WhisperX (word-aligned) ──
    if not forced or forced == "whisperx":
        try:
            from backends.whisperx import WhisperXBackend
            be = WhisperXBackend.detect()
            if be:
                return be
        except Exception as exc:
            logger.debug("whisperx: %s", exc)

    # ── faster-whisper (CUDA) ──
    if not forced or forced == "faster_whisper":
        try:
            import faster_whisper  # noqa: F401
            from backends.faster_whisper import FasterWhisperBackend
            return FasterWhisperBackend()
        except Exception as exc:
            logger.debug("faster-whisper: %s", exc)

    # ── Moonshine (ultra-fast preview) ──
    if not forced or forced == "moonshine":
        try:
            from backends.moonshine import MoonshineBackend
            be = MoonshineBackend.detect()
            if be:
                return be
        except Exception as exc:
            logger.debug("moonshine: %s", exc)

    # ── openai-whisper (CPU) ──
    if not forced or forced == "openai_whisper":
        try:
            import whisper  # noqa: F401
            from backends.openai_whisper import OpenAIWhisperBackend
            return OpenAIWhisperBackend()
        except Exception as exc:
            logger.debug("openai-whisper: %s", exc)

    if forced:
        raise RuntimeError(
            f"Requested backend '{forced}' is not available. "
            f"Install the required package and try again."
        )

    raise RuntimeError(
        "No Whisper backend found. Install one: ./install-whisper-cpp.sh (Vulkan), "
        "pip install whisperx (aligned), pip install faster-whisper (CUDA), "
        "pip install moonshine (fast preview), or pip install openai-whisper (CPU)."
    )
