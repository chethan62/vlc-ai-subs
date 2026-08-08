"""
Backend registry — WhisperX is the only transcription backend.

WhisperX (https://github.com/m-bain/whisperX) provides word-level aligned
timestamps, which are ideal for movie subtitle generation. It runs in its
own Python 3.12 venv (whisperX requires <3.14); see whisperx_backend.py.

No engine selection — VSCL_AISUBS_BACKEND is ignored. The plugin always
uses WhisperX.
"""

import logging

logger = logging.getLogger(__name__)


def resolve_backend() -> "TranscriptionBackend":
    """Return the WhisperX backend, or raise if it is not installed."""
    from backends.whisperx_backend import WhisperXBackend

    try:
        be = WhisperXBackend.detect()
    except Exception as exc:
        logger.debug("whisperx: %s", exc)
        be = None
    if be:
        return be

    raise RuntimeError(
        "WhisperX is not available. Install it with:\n"
        "  uv venv --python 3.12 ~/.local/share/vlc-ai-subs/venv-whisperx\n"
        "  uv pip install --python ~/.local/share/vlc-ai-subs/venv-whisperx/bin/python whisperx\n"
        "or re-run ./install.sh"
    )
