"""
Backend registry — auto-detects the best available backend.

Priority: whisper.cpp (Vulkan) → faster-whisper (CUDA) → openai-whisper (CPU).
"""

import logging

logger = logging.getLogger(__name__)


def resolve_backend() -> "TranscriptionBackend":
    """Return the best available TranscriptionBackend instance.

    Raises RuntimeError if *no* backend can be loaded.
    """
    from backends.base import TranscriptionBackend  # noqa: F401 — re-export

    # 1. whisper.cpp (Vulkan) — preferred
    try:
        from backends.whisper_cpp import WhisperCppBackend
        be = WhisperCppBackend.detect()
        if be:
            return be
    except Exception as exc:
        logger.debug("whisper.cpp backend not available: %s", exc)

    # 2. faster-whisper (CUDA)
    try:
        import faster_whisper  # noqa: F401
        from backends.faster_whisper import FasterWhisperBackend
        return FasterWhisperBackend()
    except Exception as exc:
        logger.debug("faster-whisper backend not available: %s", exc)

    # 3. openai-whisper (CPU fallback)
    try:
        import whisper  # noqa: F401
        from backends.openai_whisper import OpenAIWhisperBackend
        return OpenAIWhisperBackend()
    except Exception as exc:
        logger.debug("openai-whisper backend not available: %s", exc)

    raise RuntimeError(
        "No Whisper backend found. Install one: pip install faster-whisper, "
        "or run ./install-whisper-cpp.sh for Vulkan acceleration."
    )
