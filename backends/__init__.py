"""
Backend registry — picks the transcription engine for this machine.

Engines
  whisperx     WhisperX: multilingual, wav2vec2-aligned (NVIDIA CUDA only —
               faster-whisper/CTranslate2 has no ROCm backend, so AMD/Intel
               fall back to CPU here).
  parakeet     NVIDIA Parakeet-TDT-0.6B-v2 via sherpa-onnx (English, fastest).
  whispercpp   whisper.cpp: the Vulkan path that accelerates on AMD and Intel
               GPUs as well as NVIDIA, with a CPU fallback.

Selection
  VSCL_AISUBS_BACKEND unset / "auto" → hardware policy (see _auto_backend)
  VSCL_AISUBS_BACKEND=whisperx|parakeet|whispercpp → that engine (error if absent)
  legacy values (moonshine, ...) → WhisperX; "whisper_cpp" is an alias for the
  new whispercpp engine.
"""

import logging
import os

logger = logging.getLogger(__name__)

from core.gpu import nvidia_gpu, vulkan_gpu_driver, vulkan_icds

_ENGINES = {
    "whisperx": ("backends.whisperx_backend", "WhisperXBackend", "WhisperX"),
    "parakeet": ("backends.parakeet", "ParakeetBackend", "Parakeet"),
    "whispercpp": ("backends.whispercpp", "WhisperCppBackend", "whisper.cpp"),
}

_INSTALL_HINT = {
    "whisperx": (
        "uv venv --python 3.12 ~/.local/share/vlc-ai-subs/venv-whisperx && "
        "uv pip install --python ~/.local/share/vlc-ai-subs/venv-whisperx/bin/python whisperx"
    ),
    "parakeet": (
        "./install-parakeet-model.sh && uv pip install --python "
        "~/.local/share/vlc-ai-subs/venv-whisperx/bin/python sherpa-onnx"
    ),
    "whispercpp": "./install-whisper-cpp.sh",
}

# Pre-fork engine names still found in the wild / in old docs.
_ALIASES = {"whisper_cpp": "whispercpp"}


from backends.base import TranscriptionBackend


def _load_engine(name: str) -> TranscriptionBackend:
    """Import the engine and return an instance, or raise with its fix."""
    mod_name, cls_name, label = _ENGINES[name]
    try:
        import importlib

        module = importlib.import_module(mod_name)
        backend = getattr(module, cls_name).detect()
        if backend:
            return backend
    except Exception as exc:  # noqa: BLE001 — any import/detect failure = absent
        logger.debug("%s: %s", label, exc)
    raise RuntimeError(
        f"{label} backend is not available. Install it with:\n  {_INSTALL_HINT[name]}"
    )


def _auto_backend() -> TranscriptionBackend:
    """Hardware policy for VSCL_AISUBS_BACKEND unset/auto.

    NVIDIA → WhisperX: its CUDA path (int8_float16 + wav2vec2 alignment) beats
    the Vulkan alternative in quality and is the better-supported stack.
    No NVIDIA but a Vulkan driver → whisper.cpp, which is how AMD and Intel
    GPUs get GPU acceleration at all.
    Nothing usable → WhisperX (CPU), preserving the previous default.
    """
    if nvidia_gpu():
        return _load_engine("whisperx")
    driver = vulkan_gpu_driver()
    if vulkan_icds():
        try:
            backend = _load_engine("whispercpp")
            logger.debug("auto: Vulkan (%s) → whisper.cpp", driver or "unknown")
            return backend
        except RuntimeError as exc:
            logger.debug("auto: Vulkan present but whisper.cpp unusable (%s)", exc)
    return _load_engine("whisperx")


def resolve_backend(name: str | None = None) -> TranscriptionBackend:
    """Return the engine for `name`, or for VSCL_AISUBS_BACKEND when None.

    `name` lets the CLI pass the engine it already resolved (the dialog's
    language rule can pick "parakeet" before the hardware policy is consulted);
    omitting it keeps the env-var behaviour.
    """
    forced = (name if name is not None else os.environ.get("VSCL_AISUBS_BACKEND", ""))
    forced = forced.strip().lower()
    forced = _ALIASES.get(forced, forced)
    if forced in ("", "auto"):
        return _auto_backend()
    if forced in _ENGINES:
        return _load_engine(forced)

    # Unknown/legacy value → WhisperX, as before.
    return _load_engine("whisperx")
