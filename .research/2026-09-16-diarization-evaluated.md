# Speaker diarization, evaluated against the professional track

Diarization was the plugin's biggest remaining feature gap: the professional English track on
the test film marks speaker changes with a leading `- ` in **201 of its 1988 cues (10.1 %)**,
and this plugin produces none. CrispASR ships diarization (`--diarize-speakers`: pyannote
segmentation + session-scoped clustering, no HF token, embeddings discarded per recording), so
it could be adopted. It was measured, and it does not meet the bar.

## Method — the professional track is speaker-change ground truth

Two properties make the reference track usable as ground truth, both verified:

| property | value |
| --- | --- |
| cues beginning with a dialogue dash (`- `) | **201 / 1988 = 10.1 %** |
| cues with an *inline* `- ` speaker marker | **0** |

So a dash is a leading marker on a line, never mid-sentence — which means any label change
that cuts a sentence in half is unambiguously wrong output, and that the dash *positions* are
where a professional subtitler decided a speaker change happened.

## Test 1 — the film's opening, 90 s, a two-character scene

Every configuration produced the same defect (so it is not a tuning problem):

| configuration | cues | label changes that cut a sentence in half |
| --- | --- | --- |
| `--diarize-speakers` | 29 | **4** |
| `--diarize-speakers --diarize-num-speakers 2` | 29 | **4** |
| `--diarize-speakers --diarize-num-speakers 3` | 29 | **4** |

Examples, verbatim:

```
(speaker 1) Hugo, how much do you remember about that
(speaker 0) night?
(speaker 2)  You
(speaker 1) have the gift, but no control.
```

`--diarize-num-speakers 2` fixes the speaker *count* (0 and 1 only) and changes nothing about
the boundary errors. The same class of finding as the chunk-size sweep in
`2026-09-16-asr-recall-gaps.md`: the knob is not the fix.

## Test 2 — the densest dialogue-dash window, scored against ground truth

Window `2026.57 → 2116.57` s: the professional track marks **16** speaker changes there, in 42
cues (38 % of cues — a rapid back-and-forth, which is worth remembering before calling any
switch rate implausible).

| | count |
| --- | --- |
| professional speaker changes (ground truth) | 16 |
| diarizer label changes | 15 |
| **ground-truth changes the diarizer also flags within 0.7 s** | **6 of 16 (38 %)** |
| random placement of 15 change points (2 000-trial Monte Carlo) | 2.9 of 16 (18 %) |
| diarizer above chance | **+3.1 cues of 16** |
| label changes that cut a sentence in half | 2 |

The counts matching (15 vs 16) is a coincidence: the changes are in the wrong places. Beating
chance by three cues out of sixteen is not a feature; it is noise with a shape.

## Cost, measured

Diarization adds ~30 % to a run on CPU: 27.4 s for that 90 s against a 21.3 s baseline
(8 threads, warm cache) — cheaper than the CTC aligner (+72 %). Cost was not the deciding
factor.

## Decision — not shipped, and nothing exposed

The rule this project applies to every candidate: **do not ship something that makes the
output worse than not having it.** A speaker label that changes in the middle of a sentence is
visible, wrong, and unfixable by the reader. So:

- No `- ` dashes, no `[Speaker N]` labels, no environment variable. Exposing a measured-broken
  feature is worse than not having it, because the user has no way to tell a diarizer error
  from a transcription one.
- The engine's diarization remains available to anyone who wants to look at it: it is a
  CrispASR feature, not something this plugin calls.

## What would change the verdict

- **A better diarizer**: 38 %/18 % is the number to beat, and the harness for it now exists —
  professional dashes for ground truth, a Monte Carlo baseline for the null hypothesis, and a
  scoring script that is twenty lines of stdlib Python.
- **Easier material**: this is a feature film with music, overlap and shouted dialogue. A clean
  two-speaker recording (interview, podcast, lecture) is the diarizer's home ground, and the
  numbers there would likely be far better. That is a different product decision from
  transcribing films.
- **A downstream correction**: trust a speaker label only at sentence boundaries, carrying the
  previous label across a mid-sentence split. That would remove the 2 visible errors here and 4
  in the opening — but it is our heuristic patching their model, and it would still leave the
  62 % miss rate, which is the part that actually matters.

## Reusable method

Two things here generalise and are worth keeping:

1. **A professional subtitle track is ground truth for speaker changes too**, not only for
   timing and cue quality — the dash convention carries the information, and its *position*
   gives a scoring target.
2. **Score against a random baseline, not against zero.** "38 % agreement" sounds like partial
   signal until you measure what random placement of the same number of change points scores
   (18 %). The gap is the only number that means anything.
