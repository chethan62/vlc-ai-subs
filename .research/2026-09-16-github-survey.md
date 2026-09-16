# GitHub survey: STT and subtitle generation — what the ecosystem does that we don't

Research round, 2026-09-16. Method: read the projects' own documentation, then **test any
candidate technique against this plugin's real data before adopting it**. Two of the four
ideas below were rejected on measurement; the survey is still worth having, because the
rejections are the expensive part.

## The projects surveyed

| project | what it is | what it brings |
| --- | --- | --- |
| [stable-ts](https://github.com/jianfch/stable-ts) | Whisper with stabilised timestamps | the closest analog to our cue work: `regroup`, `suppress_silence`, and a hallucination filter |
| [whisperX](https://github.com/m-bain/whisperX) | word-level timestamps + diarization | forced alignment (wav2vec2), pyannote diarization, VAD on by default |
| [ffsubsync](https://github.com/smacke/ffsubsync) | sync subtitles to video | VAD-based global alignment of an existing SRT |
| [subgen](https://github.com/McCloudS/subgen) | Whisper subtitles for Plex/Jellyfin | deployment patterns; notes that Whisper ignores silence padding |
| [subsai](https://github.com/absadiki/subsai) | subtitle generation CLI/WebUI | multi-backend UI, silence-based word timestamps |
| [silero-vad](https://github.com/snakers4/silero-vad) | the VAD everyone uses | fast, accurate speech/non-speech |

## Where this plugin is ahead

Worth stating, because the survey was one-directional otherwise:

- **Cue standards.** We enforce the Netflix limits per language (CPS, line width, max
  duration, gaps, sentence splitting). stable-ts has no such limits, and whisperX's own
  README still carries the TODO *"Add max-line etc. see (openai's whisper utils.py)"*.
- **No PyTorch for the default path.** Parakeet + whisper.cpp run offline on CPU/Vulkan;
  stable-ts, whisperX and subsai all require PyTorch.
- **In-player integration.** None of these live inside the player; this is a VLC extension.

## Idea 1 — `max_instant_words`: a hallucination filter with no VAD. **REJECTED, measured.**

stable-ts removes a segment when the fraction of instantaneous words exceeds 0.5
(`max_instant_words`, default 0.5). Attractive because it needs no VAD and no phrase list,
and this plugin's VAD options were all unsafe (see the VAD note).

Tested on real Parakeet word data. It would delete this line, which is in the film's own
transcript:

```
24.72->25.76  7w  instant 5  ( 71%)  'You were, but it was a partying.'   <-- would be removed
```

The premise is that word durations carry information. Here they often do not: consecutive
transducer tokens share a timestamp, so the derived span collapses to the 80 ms frame floor.
That is a fact about **our** word-duration derivation, not about real speech, and a filter
built on it cannot distinguish a hallucination from a fast line. **Not adopted.**

## Idea 2 — Parakeet already handles non-speech; whisper.cpp does not. **MEASURED.**

Run on 60 s of music/credits: Parakeet produced **0 words** (nothing to filter). whisper.cpp
with this plugin's flags produced `you you`. So hallucination filtering is a whisper.cpp
concern, and the existing blocklist + deloop are aimed at the right engine — just not at the
short-hallucination case, which is what whisper.cpp emits on music.

## Idea 3 — the ecosystem's safe VAD shape: adjust timestamps, don't gate

stable-ts turns out to be the reference for this, and it confirms the VAD verdict in
`.research/2026-09-16-vad-and-timing.md` from an independent direction. Its defaults are all
*adjustments*, never deletions:

- `suppress_silence=True` / `suppress_word_ts=True` — move word timestamps to the detected
  speech, keeping the words.
- `use_word_position=True` — "if it is the first word, keep end; else if it is the last
  word, keep the start", so suppression cannot drag a boundary across the utterance.
- `min_word_dur` — the shortest a word may shrink to during suppression.
- `nonspeech_skip` — skip non-speech *sections* longer than a threshold (a section-level
  gate, not a sub-segment trim, which is the difference that made whisper.cpp's `--vad`
  unsafe).
- `only_voice_freq` — restrict to 200–5000 Hz, where speech lives.
- `q_levels`/`k_size` — build a silence mask by quantising and pooling the waveform, i.e.
  **without any VAD model**.

That is the design for a future round: detect silence, then *tighten or extend* timings
around it, and never drop a transcribed word.

## Idea 4 — diarization: the real feature gap. **DEFERRED, needs a token and hardware.**

whisperX does speaker labels via pyannote + `assign_word_speakers`, which is what the Netflix
guides' leading-dash convention ("- " for two speakers in one event) and speaker IDs need.
Costs: a Hugging Face token and accepting the `speaker-diarization-community-1` agreement,
PyTorch, and GPU time — on this box a capped dGPU. Its own README says "Diarization is far
from perfect", and "overlapping speech is not handled particularly well". Deferred rather
than dismissed.

## Also noted

- whisperX sets **VAD on by default** and reports "no WER degradation", which is the
  strongest counter-argument to this project's "never gate on VAD" rule. The distinction
  that survives: its VAD runs over the whole file for batching, and pyannote's detector is
  not whisper.cpp's segment processor — which is what measurably dropped dialogue here.
- whisperX's alignment limitation is worth knowing for our own: *"Transcript words which do
  not contain characters in the alignment models dictionary e.g. '2014.' or '£13.60' cannot
  be aligned and therefore are not given a timing."* Numbers and currency are exactly the
  tokens a subtitle needs to place.
- `gap_padding` (stable-ts, default `' ...'`) prepends padding to reduce the chance of the
  model predicting a timestamp *earlier* than the first utterance — relevant to the
  "captions slightly early" behaviour noted in the timing note.

## Ranked for this plugin, after the measurements

1. ~~Silence **adjustment**~~ — **built, measured, deleted.** Narrowing word boundaries
   against the VAD's spans changed *nothing* in the output: identical cue count, identical
   words, identical cue spans, identical CPS violations (35 cues / 187 words / 1.12 s median
   span / 14 cues over 20 CPS, before and after). The reason is now clear from everything
   else measured here: the transducer's timestamps are **compressed, not misplaced**, so
   words already sit inside speech and there is nothing at the cue level to correct. The
   RMS-based silence detector it was going to consume was also deleted — unusable on film
   audio, see above.
2. **Shipped instead: skipping chunks with no speech** (`core/vad.py`,
   `install-vad-model.sh`). A chunk the VAD finds no speech in needs no decode: on a 60 s
   music/credits clip, 2 of 2 chunks skipped, 3 s instead of 10 s, no text invented. This is
   the one VAD use that cannot lose anything, because a chunk with no speech has nothing to
   lose.
3. Keep each word's real end; measure reading speed from speech extent (carried over).
4. Diarization for speaker labels (Idea 4) — the largest genuine feature gap.
5. Song/lyric handling with `<i>` (VLC support verified at source).
6. Unverified and labelled as such: AMD/Intel Vulkan, VLC 4.0 registration, Parakeet v3
   per-language accuracy.

## Addendum — the `hclivess/whisperer` claim, measured

A reader pointed at [hclivess/whisperer](https://github.com/hclivess/whisperer), a mature
batch subtitle GUI whose changelog contains: *"audio streams that start after the video lost
their lead during extraction, making every cue early by that amount."* We have the same
extraction shape, so this was worth checking rather than admiring.

**The lead is real.** On the 145-minute film: video `start_time = 0.000`, audio
`start_time = 1.008` (24 frames at 23.976 fps). Our decode begins the WAV at the first audio
sample — container 8738.563 s, decoded WAV 8732.960 s — so every timestamp we produce is on
the *audio* timeline, 1.008 s before its container position.

**The net error was measured against ground truth**, using the film's own embedded
professional English subtitle track (1988 cues):

| | cue-start matches (≤0.15 s) | median error | best global fit |
| --- | --- | --- | --- |
| as produced | 215 / 1891 | +0.367 s | +0.53 s (749 hits) |
| + the audio lead | **131** | −0.435 s | −0.48 s (749) |
| + VAD onset snapping | **268** | +0.294 s | +0.44 s (711) |
| + both | 161 | −0.419 s | −0.48 s (703) |

Random baseline: ~136 hits.

**Neither correction ships.** Adding the lead back moves every cue from 0.53 s early to
0.48 s late — worse on cue-start matches — because the lead and the model's own emission lag
(a TDT model emits a token after the audio it belongs to, ~0.5 s here) partly cancel. VAD
onset snapping — whisperer's own approach — improves the tight metric (215 → 268) while
slightly worsening the broad fit (749 → 711), which is not a clear enough win to justify
changing every timestamp in the file.

So the plugin now **reports** the lead when it is non-zero instead of silently correcting or
silently ignoring it: `audio starts 1.008s after the video; subtitles are timed to the audio
stream`. A uniform sync complaint is otherwise very hard to diagnose, and this is the number
that explains it.

**Reusable method, worth more than the finding:** a container's embedded professional
subtitle track is ground truth for sync. Extract it (`ffmpeg -map 0:s:N`), then fit a global
offset by cross-correlating cue starts — a nearest-cue comparison across two differently
segmented tracks is too noisy to conclude anything from (it gave +0.367 s with a ±3 s
spread where the fit gave a confident +0.53 s).
