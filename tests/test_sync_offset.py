"""Tests for tools/sync_offset.py — the instrument behind the sync decision.

It exists because "cues look uniformly early, shift them all" is a tempting wrong
fix: on the two files measured the optima were +0.55 s and +0.01 s. These tests pin
the arithmetic, so the numbers quoted in the README can be reproduced by anyone.
"""

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "sync_offset.py"
_spec = importlib.util.spec_from_file_location("sync_offset", _TOOL)
assert _spec is not None and _spec.loader is not None
so = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(so)


def srt(*starts: float) -> str:
    """An SRT whose cues all start at the given seconds."""
    blocks = []
    for i, start in enumerate(starts, 1):
        def stamp(t: float) -> str:
            h, rem = divmod(t, 3600)
            m, s = divmod(rem, 60)
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"
        blocks.append(f"{i}\n{stamp(start)} --> {stamp(start + 1.5)}\ntext {i}\n")
    return "\n".join(blocks)


def write(tmp_path, name: str, *starts: float) -> str:
    path = tmp_path / name
    path.write_text(srt(*starts), encoding="utf-8")
    return str(path)


def test_a_cue_is_a_match_within_the_tolerance():
    ours = [10.0]
    assert so.count_matches(ours, [10.1], shift=0.0) == 1
    assert so.count_matches(ours, [10.2], shift=0.0) == 0
    # and shifting us onto it counts again
    assert so.count_matches(ours, [10.2], shift=0.2) == 1


def test_a_shifted_file_reports_that_shift(tmp_path):
    """If the reference is our output moved 0.6 s later, 0.6 s is the answer."""
    ours = [10.0, 20.0, 30.0, 40.0]
    reference = [s + 0.6 for s in ours]
    shift, matches = so.best_shift(ours, reference)
    assert abs(shift - 0.6) <= 0.02, shift
    assert matches == len(ours)


def test_an_aligned_file_reports_no_shift(tmp_path):
    """The clean-file case: our output already aligned, so nothing should move."""
    ours = [10.0, 20.0, 30.0, 40.0]
    shift, matches = so.best_shift(ours, list(ours))
    assert abs(shift) <= 0.02, shift
    assert matches == len(ours)


def test_finds_the_optimum_from_srt_files(tmp_path):
    """End to end through the file readers."""
    ours_path = write(tmp_path, "ours.srt", 10.0, 20.0, 30.0, 40.0, 50.0)
    ref_path = write(tmp_path, "ref.srt", 10.5, 20.5, 30.5, 40.5, 50.5)
    assert so.cue_starts(ours_path) == [10.0, 20.0, 30.0, 40.0, 50.0]
    shift, matches = so.best_shift(so.cue_starts(ours_path), so.cue_starts(ref_path))
    assert abs(shift - 0.5) <= 0.02, shift
    assert matches == 5


def test_the_two_real_files_disagree(tmp_path):
    """The finding itself, as arithmetic: a shift that fixes one file ruins the other.

    Both are 5-cue files; the first needs +0.55 s, the second is already aligned.
    Applying the first's optimum to the second must lose matches, not gain them.
    """
    film = [10.0, 20.0, 30.0, 40.0, 50.0]
    film_ref = [s + 0.55 for s in film]
    film_shift, _ = so.best_shift(film, film_ref)
    assert abs(film_shift - 0.55) <= 0.02, film_shift

    lucky = [10.0, 20.0, 30.0, 40.0, 50.0]
    lucky_ref = list(lucky)                      # already aligned, lead 0.000 s
    assert so.count_matches(lucky, lucky_ref, 0.0) == 5
    assert so.count_matches(lucky, lucky_ref, film_shift) == 0   # the fix would wreck it
