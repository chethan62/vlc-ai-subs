# What a professional subtitle track says about our output

The test film `Disclosure.Day.2026…mkv` ships an embedded professional English subtitle track
(1988 cues). Timing was compared against it earlier (`2026-09-16-github-survey.md`). This is
what the same track says about the *quality* of the cues themselves — and it overturned one
of the project's standing claims and found one large real defect.

## Retired: the reading-speed "gap" is not a gap

The README treated our rate of cues over the Netflix 20 CPS ceiling as a known quality gap
(24 % of the film's cues). The professional track:

| | ours (Parakeet, 1891 cues) | professional (1988 cues) |
| --- | --- | --- |
| CPS median | 14.67 | 14.40 |
| CPS p90 | 25.00 | 24.96 |
| CPS p99 | 37.50 | 37.12 |
| CPS max | 62.50 | 52.63 |
| **cues over 20 CPS** | **24.0 %** | **23.7 %** |

The distributions are indistinguishable. The 20 CPS figure in the Netflix guide is a *target*
that professional subtitlers exceed at the same rate we do, not a rule that a good track
satisfies. Our earlier framing — that this was a defect to be fixed — was wrong, and no
redistribution work should be attempted against it. (This also explains why the dense-run
rebalancing experiment that was built and deleted during the cue-standards round could not
have helped: there was nothing to fix.)

## Found: cues too brief to read

| | ours | professional |
| --- | --- | --- |
| cues under 1.0 s | **365 (19.3 %)** | **8 (0.4 %)** |
| shortest cue | 0.08 s | 0.88 s |
| median duration | 1.60 s | 1.92 s |
| longest line | 42 chars | 33 chars |
| cues over 7 s | 0 | 0 |

This one is real. 299 of the 365 were boxed in on *both* sides by abutting neighbours, so the
existing per-cue pass could not lift them: its ceiling is the next cue's start minus the
2-frame gap, and the neighbours were already at the floor.

### The fix, and what it may not do

`borrow_for_short_cues` gives a brief cue time, in this order:

1. the silence after it — never past the next cue's *start*, because over that lie the next
   line's own words (a subtitle shown before its words are spoken is a sync fault, not a
   flicker);
2. the silence before it — which belongs to nobody;
3. the previous cue's spare tail, only when the two abut, and only down to *that* cue's own
   floor.

Two rules came from measuring rather than reasoning:

- **A donor's end may only move earlier, never later.** The first implementation handed over a
  donor's "tail" with `previous.end = new_start`; where the two cues did *not* abut — 116 s of
  silence separated them — that *stretched* a 1.84 s cue to **117.56 s**. Replaying the film's
  cues caught it: 13 cues over the 7 s maximum where the pass without the change had none.
- **Closing the 2-frame gap is legitimate when it is what readability costs.** Our
  `MIN_GAP = 0.08 s` was consuming the very time being borrowed. The professional track's
  median inter-cue gap is **0.00 s** — it abuts cues — so the borrow runs with a gap of 0 and
  the main pass keeps `MIN_GAP` for everything else.

### Result, measured by replaying the film's own 1891 cues

| | before | after | professional |
| --- | --- | --- | --- |
| cues under 1.0 s | 365 (19.3 %) | **157 (8.3 %)** | 8 (0.4 %) |
| shortest cue | 0.08 s | 0.16 s | 0.88 s |
| cues over 7 s | 0 | **0** | 0 |
| overlapping pairs | 0 | **0** | 0 |
| cues whose text changed | — | **0** | — |
| word sequence | — | **identical (10143 words)** | — |

Of the 157 left: 28 are under 0.5 s, 73 between 0.5 and 0.8 s, and 56 between 0.8 and 1.0 s.
The last band is professional-normal (their shortest cue is 0.88 s). The residual is
concentrated where there is genuinely nothing to take — neighbours abutting with no spare tail
— and the alternative remedies were rejected on principle: extending over the next cue's
speech, or merging two cues, both change *what the viewer reads* rather than when a boundary
moves.

## Method notes

- Compare **distributions, not aggregates**. "24 % of cues over 20 CPS" looked like a defect
  until the same statistic was computed for a professional track; the mean alone would not
  have shown it either. Percentile-by-percentile comparison settled it in one table.
- A professional subtitle track in the container is ground truth for *both* timing and
  quality. It costs one `ffmpeg -map 0:s:N` and answers questions no amount of self-review
  answers.
- Replaying a long run's own cues through a changed pass is a valid measurement of that pass
  and costs seconds instead of 20 minutes — but only the *pass*: it cannot catch anything the
  transcription itself did. The end-to-end re-run is still required before shipping.
