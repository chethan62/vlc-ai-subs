# The decoder was skipping dialogue inside chunks

Found while auditing the transcription *against* the professional subtitle track rather than
against itself. This is the largest real defect found in this plugin so far, and it was
invisible in every output the plugin produces.

## How it surfaced

The film's professional track holds **11 549 words**; our transcript of the same film has
**10 089**. Auditing every professional cue for whether our overlapping cues contain its words:

| | count |
| --- | --- |
| professional cues we do not cover (< half their words present) | 302 |
| words in those cues | 1446 |

Most are explainable and not ours to fix: **♪ song lyrics** (ASR cannot transcribe singing —
whisper.cpp calls the same audio `(upbeat music)`), the **on-screen countdown** "One! Two!
Three!", numerals, and proper-noun lists. But one entry was plain dialogue:

> `6198.23` — **"You have the gift, but no control."**

We had produced nothing for it, or for the two sentences following it.

## The experiment that settled it

| how the audio was fed | result |
| --- | --- |
| that region alone, as a short file | **transcribed perfectly** |
| the film's own 30 s chunk containing it | **nothing at all** |

The audio and the model are identical; only the chunk it sits in differs. Then, sweeping the
chunk size over the *same* 90 s of audio:

| chunk | cue covering 47.6 s |
| --- | --- |
| 30 s (the default) | **missing** |
| 31 s | present |
| 45 s | present |
| 90 s (no chunking) | present |

So this is **not** a bug in our chunk merging, our window rule, or the cue grouping — all of
which were examined and are correct. The transducer's decode of a 30 s window **silently omits
roughly 13 s of speech** (a 42.20 s cue followed by one at 55.36 s), and the same audio decodes
correctly at every other boundary alignment tried. Tuning the chunk length would only move the
failure somewhere else.

The per-chunk word accounting that made this visible:
`chunk 2/3 [30.0,60.0) kept 71/78 words` — the chunk had *decoded* words, so nothing looked
wrong; the missing line simply was not among them.

## The fix: verify the output, then re-decode what it missed

`uncovered_speech()` asks the VAD a question it is good at — *where is the speech?* — and
compares it with what the chunk actually produced. Any speech span of ≥1 s that no decoded word
covers is decoded again on its own, and **whatever comes back is added**. Nothing already
transcribed is ever removed, which is what distinguishes this from the chunk-skipping gate
withdrawn in v1.4.3 (`2026-09-16-github-survey.md`, Addendum 3).

Two details came from measuring the first attempt:

- **Merge adjacent holes before decoding.** Re-decoding four neighbouring 1–3 s holes separately
  produced garbled fragments — `'No No under understanding of your purpose.'` — because a 1 s
  clip carries no context. Merged into one 7.7 s decode, the model returns the sentence cleanly.
- **Keep only words that start inside the hole.** The decode is padded by 0.5 s for context, so
  its first and last words belong to speech the chunk already transcribed. With a strict
  boundary, `"You have the gift"` lost its `"You"` to a hole starting at 47.69 s; a 0.35 s
  lead-in margin restores it without duplicating the neighbouring word.

On the reproduction case (30 s chunks, the exact configuration that lost the dialogue):
**23 words recovered**, producing

> `have the gift, but no control.` · `No understanding of your purpose.` · `Not a clue as to what your situation will demand of you`

The VAD is therefore used in **two** directions, and the asymmetry is the whole point: it may
*add* (decode again where it hears speech and we produced none — safe, bounded) but it may never
*subtract* (skip a chunk it calls speechless — measured deleting 97 words).

## What this says about verifying ASR at all

Every check this project had — cue counts, word counts, structural limits, the professional
track for *timing* — passed while three sentences of dialogue were missing. What found it was
asking a second question of the same ground truth: not "when" but "did we get it at all". A
detector that is unreliable as a *gate* is exactly the right instrument for that, because a
false positive costs one redundant decode and a false negative costs a line nobody ever sees.
