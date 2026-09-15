"""Audio decoding shared by the engines that shell out to ffmpeg.

Parakeet (sherpa-onnx) and whisper.cpp both want a 16 kHz mono PCM wav, so the
decode + temp-file handling lives here once instead of in each runner.

Leaf module: stdlib only, no imports from core/, backends/ or the runners.
"""

import os
import shutil
import subprocess
import tempfile

SAMPLE_RATE = 16000
DECODE_TIMEOUT = 600


def ffmpeg_path() -> str | None:
    """Absolute path to ffmpeg, or None when it is not on PATH."""
    return shutil.which("ffmpeg")


def decode_to_wav16k(media_path: str, timeout: float = DECODE_TIMEOUT) -> str:
    """Decode arbitrary media to a fresh 16 kHz mono PCM wav.

    The caller owns the returned path (usually: ``os.unlink`` it).
    Raises RuntimeError with the ffmpeg stderr tail when decoding fails.
    """
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found — required by this backend")
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="aisubs_")
    os.close(fd)
    proc = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", media_path,
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", tmp],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0 or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise RuntimeError(f"ffmpeg decode failed: {(proc.stderr or '').strip()[:300]}")
    return tmp
