#!/usr/bin/env python3
"""Measure the optimal global offset between our subtitles and a reference track.

    PYTHONPATH= venv/bin/python tools/sync_offset.py <ours.srt> <reference.srt>

Positive offset = our cues should move later.

Why this exists: "our subtitles look uniformly early, so shift them all" is a
tempting and wrong fix. On the two files measured here the optima were +0.55 s and
+0.00 s — a global correction that helps one and destroys the other. Anything a
single file appears to ask for must be cross-validated on a second file with a
*different audio start_time* before it can be called a property of the pipeline
rather than of that file's reference track. See README, "A track can also start
late", and .research/2026-09-17-sync-offset-cross-validation.md.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crispasr_runner import parse_srt

# ponytail: fixed sweep, no flags. Make STEP/SPAN arguments if a file ever needs a
# finer grain or a wider window than +/-1.2 s.
STEP = 0.05
SPAN = 1.2
TOLERANCE = 0.15


def cue_starts(path: str) -> list[float]:
    return sorted(cue["start"] for cue in parse_srt(path))


def count_matches(ours: list[float], reference: list[float], shift: float) -> int:
    """How many of our cues land within TOLERANCE of a reference cue, after `shift`."""
    return sum(1 for start in ours
               if any(abs(start + shift - ref) <= TOLERANCE for ref in reference))


def best_shift(ours: list[float], reference: list[float]) -> tuple[float, int]:
    """The CENTRE of the best plateau of offsets, and its match count.

    Within TOLERANCE every shift in the best plateau is equally good, so a plain
    argmax reports an arbitrary edge of it: a perfectly aligned file has a plateau
    from -0.15 to +0.15 and would be reported as -0.15. Sweep once and centre.
    """
    steps = int(SPAN / STEP)
    scores = {round(i * STEP, 6): count_matches(ours, reference, i * STEP)
              for i in range(-steps, steps + 1)}
    best = max(scores.values())
    winners = [shift for shift, matches in scores.items() if matches == best]
    return (min(winners) + max(winners)) / 2, best


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <ours.srt> <reference.srt>")
    ours, reference = cue_starts(sys.argv[1]), cue_starts(sys.argv[2])
    if not ours or not reference:
        raise SystemExit("one of the inputs contained no cues")

    shift, matches = best_shift(ours, reference)
    at_zero = count_matches(ours, reference, 0.0)
    print(f"  ours {len(ours)} cues · reference {len(reference)} cues")
    for i in range(-2, 5):                       # the shape, coarsely: dome vs plateau
        s = round(i * 0.2, 3)
        m = count_matches(ours, reference, s)
        print(f"  {s:+5.1f}s {m:6d} ({m / len(ours) * 100:4.1f}%)")
    print(f"  best {shift:+.2f}s -> {matches} ({matches / len(ours) * 100:.1f}%) · "
          f"at zero {at_zero} ({at_zero / len(ours) * 100:.1f}%) · "
          f"gain {matches / max(1, at_zero):.1f}x")


if __name__ == "__main__":
    main()
