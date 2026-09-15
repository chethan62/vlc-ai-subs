"""Subprocess timeouts for the transcription backends.

Why this exists: a flat 20-minute cap on the WhisperX subprocess made the
default path fail on any film longer than ~40 minutes (WhisperX runs at
~2-4x realtime on a healthy GPU, and many times slower on a power-capped
one), i.e. exactly the long-media case the plugin is for. The ceilings are
"how long can a legitimate run take", not a target, and are overridable:

    VSCL_AISUBS_TIMEOUT=<seconds>    # 0 (or negative) = wait indefinitely

This is a leaf module: it imports nothing from core/, backends/ or the
runners, so any of them may depend on it.
"""

import os

# Per-task ceilings in seconds.
DEFAULT_TIMEOUTS = {
    "transcribe": 4 * 3600,
    "translate": 6 * 3600,  # transcription + the NLLB/M2M cascade on top
}

# Unknown task strings (future tasks, typos) get the conservative default.
_FALLBACK = 4 * 3600


def resolve_timeout(task: str, default: float | None = None) -> float | None:
    """Seconds to allow the backend subprocess, or None for no limit.

    *default* (when given) replaces the per-task table for this call.
    VSCL_AISUBS_TIMEOUT overrides everything; an unparseable value is ignored
    (falls back) rather than silently disabling the limit.
    """
    base: float | None = DEFAULT_TIMEOUTS.get(task, _FALLBACK) if default is None else default
    raw = os.environ.get("VSCL_AISUBS_TIMEOUT", "").strip()
    if not raw:
        return base
    try:
        secs = float(raw)
    except ValueError:
        return base
    return secs if secs > 0 else None
