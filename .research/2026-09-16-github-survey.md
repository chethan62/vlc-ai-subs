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

1. Silence **adjustment** (Idea 3): tighten word ends around detected silence; never gate.
2. Keep each word's real end; measure reading speed from speech extent (carried over).
3. Diarization for speaker labels (Idea 4) — the largest genuine feature gap.
4. Song/lyric handling with `<i>` (VLC support verified at source).
5. Unverified and labelled as such: AMD/Intel Vulkan, VLC 4.0 registration, Parakeet v3
   per-language accuracy.
