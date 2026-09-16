# Cue standards research — 2026-09-16

Round: make the emitted subtitles *readable* by the published standards, not merely
plausible. Prior rounds had already fixed what the engines say (hallucination
blocklist, decode hardening) and how long a cue may stay on screen (7 s cap,
sentence-boundary splitting, round 8). What was still missing was the second half
of every subtitle spec: **reading speed**, and the fact that line width and
reading speed are not the same number for every language.

## Sources

| Source | What it gave |
| --- | --- |
| [Netflix Timed Text Style Guide — General Requirements](https://partnerhelp.netflixstudios.com/hc/en-us/articles/215758617-Timed-Text-Style-Guide-General-Requirements) | min duration 5/6 s, max 7 s, 2 lines, 42 chars/line, minimum 2-frame gap, "add time before cutting text" |
| [Netflix English (USA) guide](https://partnerhelp.netflixstudios.com/hc/en-us/articles/217350977-English-USA-Timed-Text-Style-Guide) | 20 CPS adult, 17 children's (WPM retired in favour of CPS) |
| [Netflix per-language limits, tabulated](https://sublingo.cc/guides/subtitle-specs-by-language) | full per-language table: English 20 CPS/42, most languages 17/42, **Chinese 9/16, Japanese 4/13, Korean 12/16**, plus translation expansion factors |
| [IWSLT 2026 evaluation paper](https://aclanthology.org/2026.iwslt-1.8.pdf) | independent confirmation: "maximum subtitle reading speed 21 characters per second for Arabic, German and Spanish; 4 characters per second for Japanese" |

The plugin's existing constants already matched the spec on three of five axes —
7 s maximum, 42-character lines, 2-frame gap — which was worth confirming, since
they had been chosen from memory rather than from the guides.

## What was wrong

1. **No reading-speed notion at all.** A cue could show 100 characters in one
   second and nothing complained. Measured on a real 5-minute English run: 13 of
   49 cues (27%) exceeded 20 CPS, the worst at 39.4 CPS.
2. **One line width for every script.** CJK lines are capped at 13–16 full-width
   characters, not 42 — a full-width character is roughly two Latin widths. On a
   real 1,649-cue Chinese file: **90 lines over the cap, the worst 43 characters**.
3. **Unpunctuated CJK could not be split at all.** The round-8 splitter needs a
   sentence terminator; in that Chinese file only **1 of 27** over-long cues had
   one, so cues up to **25 s** stayed on screen intact.
4. **A latent fragment-join bug**: folding two CJK fragments inserted a literal
   space, opening a gap that was not in the audio.

## What changed (`core/cues.py`)

- `LANGUAGE_LIMITS` + `limits_for(language, text)` — the guides' CPS and
  line-width figures per language; when the language is unknown the *script*
  decides (kana → Japanese, Hangul → Korean, Han → Chinese), so an `auto` run whose
  language was never detected is not wrapped as if it were English.
- **Reading-speed pass**: a cue whose text needs longer than the language's CPS
  ceiling has its end extended toward the next cue (never past it, never
  overlapping). Words are never cut — per the guides, and per this plugin's rule
  that it does not rewrite dialogue.
- **CJK capacity split**: a cue holding more text than `max_lines × line_chars` is
  split even when it is brief, into equal character runs when there is no
  punctuation to split on (standard practice for CJK subtitles).
- Script-aware fragment join (no space between CJK characters).
- `VSCL_AISUBS_MAX_CPS` (1–60) and the existing `VSCL_AISUBS_MAX_LINE` override
  both ceilings globally.

## Measurements after the change

| Input | Before | After |
| --- | --- | --- |
| Chinese, 1,649 cues (real pipeline output) | longest cue 25.0 s; 90 lines over 16 chars; max line 43 | longest **7.1 s**; **0** lines over 16; max line **16**; every character preserved (no stray spaces added) |
| English, 49 cues, 5 min (Parakeet) | 13 cues over 20 CPS; max 39.4 CPS | 12 over 20 CPS; 8 cues given 1.5 s more screen time; **0 overlaps**; cue count and text unchanged |
| English, 418 cues, 47.5 min (Parakeet) | 418 cues | 418 cues, no overlaps, text unchanged |

The English gain is deliberately small and is reported as such: where speech is
continuous there is no slack between cues to take, and the plugin will not delete
words to manufacture it. The headroom only exists where the audio actually pauses.

## Verification

- 212 unit tests (10 new for these rules), all passing.
- Five negative controls — reverting each change in turn fails exactly the test
  named for it. One of the new tests (CJK fragment folding) turned out **vacuous**
  under its own negative control — it never folded anything — and was rewritten
  until the revert made it fail.
- The Python 5.4/Lua harnesses and the installer matrix were unaffected.
