"""
Core SRT module — timestamp formatting and file writing.

Produced SRT files follow the standard SubRip spec with millisecond-precision
timestamps and blank-line separators.
"""

import os
from typing import Iterable


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: Iterable[dict]) -> str:
    """Build a complete SRT file body from {start, end, text} segments."""
    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(
            f"{i}\n"
            f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n"
            f"{seg['text']}\n"
        )
    return "\n".join(blocks)


def write_srt(segments: list[dict], media_path: str, srt_path: str | None = None) -> str:
    """Write SRT to disk and return the path used.

    If *srt_path* is given it is used exactly; otherwise it is derived from
    the media filename (``<media>.srt``). Parent directories are created as
    needed when an explicit path is provided.
    """
    if srt_path:
        actual = srt_path
        os.makedirs(os.path.dirname(actual) or ".", exist_ok=True)
    else:
        base, _ = os.path.splitext(media_path)
        actual = base + ".srt"

    with open(actual, "w", encoding="utf-8") as f:
        f.write(segments_to_srt(segments))

    return actual
