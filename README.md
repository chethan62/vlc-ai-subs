# vlc-ai-subs

VLC media player plugin that generates subtitles using local AI — works offline
with any video, any language. Transcribe and translate into clean `.srt` files
or real-time on-screen captions.

[Upstream](https://github.com/voidrlm/vlc-ai-subs) · [Install](#quick-start) · [Engine](#engine) · [Models](#models) · [Env vars](#environment-variables)

## Features

| | |
|---|---|
| **Zero-config** | "Recommended (auto)" model + Auto engine + Translate to English by default — open a video, click Generate, done |
| **Word-level timing** | WhisperX wav2vec2 alignment on CUDA, or Parakeet's native TDT word timestamps (CPU too) |
| **Three engines** | WhisperX (multilingual, word-aligned), Parakeet (English + 25 European languages, ~10× faster) or whisper.cpp (Vulkan) |
| **Auto engine** | Auto (default) picks the fastest engine that covers the language (Parakeet v2/v3), else WhisperX (NVIDIA) / whisper.cpp (AMD/Intel) / CPU |
| **GPU acceleration** | CUDA on NVIDIA; **Vulkan for AMD/Intel/NVIDIA** via whisper.cpp; CPU fallback |
| **Two modes** | Generate & Load SRT (default), or Real-time OSD — cues appear on the OSD when playback reaches them |
| **Right audio track** | Multi-audio releases (DUAL/MULTi) are common and ffmpeg's default is the *first* track — the dub. The plugin picks the one matching your language, skips audio-description/commentary tracks, and says which it used |
| **SRT output** | Standard `.srt` files written next to your video — compatible with Kdenlive, VLC, mpv, PotPlayer |
| **Readable cues** | Text wrapped to the *language's* line width from the Netflix timed-text guides (42 chars Latin, 16 Chinese/Korean, 13 Japanese, ≤2 lines); cues over 7 s split at sentence boundaries (cut into equal runs for unpunctuated CJK); a cue whose text needs longer than 20 CPS *(9 Chinese, 4 Japanese, 12 Korean, 17 most other languages)* is given more time up to the next cue — never fewer words; min-duration and gap cleanup |
| **Cancel + memory** | Cancel a running transcription — it stops the backend child and its ffmpeg decode, and removes the partial decoded audio (an 87 MB file, measured); a run killed outright leaves it for the next run's sweep; the dialog remembers engine/model/language/task/mode |
| **Any language** | Auto-detection or specify a language code (`en`, `es`, `fr`, `hi`, `ja`, `zh`…) |
| **Translation** | Translate any language to English subtitles |
| **VLC 3.x now, 4.x-ready** | Every Lua API it uses is present in VLC 4.0's source (checked); no 4.0 build has been run yet |
| **Cross-platform** | Linux, macOS, Windows (native, snap, flatpak) |

## Subtitle standards

Cue text is not free-form — the plugin applies the published limits so the output
is readable and passes a QC pass rather than merely looking plausible.

| Subtitle language | Reading speed | Line width | Max on screen | Min gap |
| --- | --- | --- | --- | --- |
| English | 20 CPS | 42 chars | 7 s | 2 frames |
| Spanish, French, German, Hindi, Russian, … | 17 CPS | 42 chars | 7 s | 2 frames |
| Chinese | 9 CPS | 16 full-width chars | 7 s | 2 frames |
| Korean | 12 CPS | 16 full-width chars | 7 s | 2 frames |
| Japanese | 4 CPS | 13 full-width chars | 7 s | 2 frames |

Source: the [Netflix Timed Text Style Guides](https://partnerhelp.netflixstudios.com/hc/en-us/articles/215758617-Timed-Text-Style-Guide-General-Requirements)
(the most widely cited public subtitle spec; the BBC's 160–180 wpm target lands in
the same range for Latin scripts). CPS counts every character, spaces and
punctuation included.

Where a cue carries more text than its span can show at that speed, the plugin
**extends the cue toward the next one** rather than shortening the dialogue — the
guidance is explicit that adding time is preferred to cutting words, and this
plugin never rewrites what was said. Where the speech is continuous and there is
no slack to take, the cue stays dense; the alternative would be dropping words.

Three rules keep those limits true in the output, and all three came from testing a
feature-length film after episode-length media had passed:

- **A cue whose text needs a third line is split**, whatever the script. The
  splitter only looked at cues that were too long in *time*, or CJK text too wide
  for its line cap, so a long comma-separated English sentence survived intact and
  was written as one line.
- **Nothing is displayed longer than the maximum.** A cue the splitter cannot break
  (two words over a 54 s silence) used to be left exactly as it was.
- **Line breaks survive the whole pipeline.** The engines wrap the text, and the
  CLI's hallucination filter then normalised the whitespace, flattening every
  wrapped cue back into a single line.
- **A cue never bridges a pause, and a word always has a span.** A transducer reports
  the frame a token was *emitted* on, not a span, so a word's own last-token
  timestamp is its start — taking that as the end gave every single-token word a
  zero-length span and made cue spans shorter than the speech. And the grouping guard
  ran *after* the offending word was added, so "Just" (26.96 s) and "You" (78.56 s) —
  52 seconds apart in the audio — became one 51.6 s cue that displayed "You" 52
  seconds early. Verified against the real model on a 120 s slice of the film.

Measured on a 145-minute feature (2 h 18 m, one English track, 1841 cues, 20 m 24 s
of Parakeet on CPU, 2.7 GB peak):

| | as produced | after |
| --- | --- | --- |
| cues with a line over 42 chars | 397 (worst 100) | **0** (worst 42) |
| cues longer than 7 s | 43 (worst 51.6 s) | **0** (worst 6.5 s) |
| overlapping cues | 1 | **0** |
| cues | 1841 | 1891 (the necessary splits) |
| words | 10143 | **10143, same order** |

The one standard the plugin cannot yet meet is reading speed: 453 cues (24.0 %) of
that film carry more text than 20 CPS allows in the span the engine gave them, the
worst needing 62 CPS. (The grouping fix above took that from 505 cues / 27.3 %: cue
spans used to end where a word's own timestamp sat, which was its *start*, so every
span was short.) This is a timing limitation, not a text-handling one — and the honest
version of the evidence is narrower than it first looked. A few cues imply physically
impossible speech (450–580 wpm, i.e. 8–10 words per second) and those are real
timestamp defects; but most of the apparent excess is the *metric*: a cue's span runs to
the last word's start, not the end of its audio, so three-word cues lose about a third
of their span and short function-word lines read far faster than they are. Speech itself
is 120–160 wpm. The time exists — the film averages ~6 CPS — the words are simply
reported closer together than they were spoken.

**WhisperX does not fix this, and the earlier claim that it did has been withdrawn.**
Measured on the same 60 s minute: Parakeet median 217 wpm / max 500, WhisperX (aligned)
median **244** / max **580**, with tighter cue spans (median 1.00 s). Its alignment is
real, but its cue timings are no more plausible. See
`.research/2026-09-16-vad-and-timing.md` for the full comparison, the VAD measurements,
and the ranked list of what still needs improving. The plugin never shortens the text to
hide any of this.

Measured on real files (see `tests/` for the fixtures):

- a 5-minute English episode — 8 of 49 cues were given a little more time, 1.5 s
  in total, none of them overlapping the next cue
- a 1,649-cue Chinese file — the longest cue 25.0 s → **7.1 s**, lines over the
  16-character cap **90 → 0**, every character preserved
- a 47.5-minute English episode — 418 cues, unchanged cue count, no overlaps

## Which audio track

A release can carry several audio tracks, and **ffmpeg's default is the first
one** — which on real releases is the dub rather than the dialogue:

| Real file | Tracks | ffmpeg's default |
| --- | --- | --- |
| `…MULTi.VFF….mkv` | 1 `fre` (VFF) · 2 `eng` · 3 `eng` *"Descriptive"* | track 1 — French |
| `…DUAL….mkv` | 1 `por` · 2 `eng` (both flagged `default=1`) | track 1 — Portuguese |

So asking for English subtitles transcribed the French audio, and an English-only
model answered with confident nonsense instead of an error:

```
Tromaris troubles in sentence patrimony genétic and young women.
```

The plugin now chooses deliberately: **the track matching your language** (container
tags are ISO 639-2 — `fre`, `eng`, `por` — while the UI speaks 639-1, so both are
understood), **skipping descriptive and commentary tracks** (measured: the
descriptive track above carries no standard flag at all, only `title=Descriptive`),
and otherwise taking the first track that is dialogue. It reports its choice in the
status line, and `VSCL_AISUBS_AUDIO_TRACK` overrides it (an index, a position, or a
language). The same file now yields actual dialogue:

```
Did you know that?
I did not.
```

## Loading the SRT into the playing video

`vlc.input.add_subtitle(path, autoselect)` takes a second argument that decides
whether the track is *shown*, and it defaults to **false** — from
`modules/lua/libs/input.c` at 3.0.23:

```c
bool b_autoselect = false;
if( lua_gettop( L ) >= 2 ) b_autoselect = lua_toboolean( L, 2 );
```

Called with one argument the file is added and nothing appears. Measured in a real
VLC: `spu-es` stayed **-1** and the caption never reached the screen, while the dialog
said "Subtitles loaded". With `true`, `spu-es` became **2** and OCR of a screenshot
read the caption back off the video.

The dialog now asks for the autoselect and reports what actually happened:
"Subtitles loaded." only when a subtitle track really is selected, otherwise
"SRT: <path> — choose it under Subtitles".

## Real-time OSD

OSD mode used to push each cue the moment the transcription produced it. Nothing
paces transcription to the video — Parakeet runs ~10× faster than real time and
WhisperX slower than it — so the captions sat minutes away from the picture
(measured in a real VLC run: the dialog's "current cue" ran far ahead of the
video). Cues now queue and appear when playback reaches their start, up to one poll
(1 s) early. A cue the video has already passed still shows — late beats never — for
at least 1.5 s, and cues passed over are dropped rather than replayed.

The position comes from the input's `time` variable: microseconds, the same unit
`vlc.osd.message` takes. Both were confirmed in a **real VLC 3.0.23 run**, not only
from the source: a Lua interface probe read `time` advancing 1.5 s per 1.5 s of
playback (150238 → 1650187 → 3150225 µs), and the same
`vlc.osd.message(text, chan, "bottom", µs)` call the extension makes put
"OSD PROOF AISUBS" on screen — read back by OCR of a screenshot of the video window.

## Cancelling a run

Cancel signals the CLI (the extension has no process API), which stops the backend
runner; the runner stops its own ffmpeg decode and deletes the half-written temp
wav. Measured before that existed: cancelling mid-decode stopped the processes but
left an **87 MB partial wav** in `/tmp` — a 145-minute film would leave ~280 MB, and
every cancelled run added another.

A run killed outright (SIGKILL, or VLC crashing) gets no chance to clean up, so each
new run sweeps what it left in `/tmp` — the decoded wav, the extension's mirror file,
its `.pid`, and realtime mode's temp `.srt` — but only files older than two hours, so a
second VLC window's live run is never touched. (The debug log has its own `.log` suffix
and is left alone.)

## When a run dies

Every runner ends a run with a terminal event — `done` or `error`. A run that stops
without one was killed or crashed, and it can exit **0** with an empty stdout: measured
on a 145-minute film, a run died 83 seconds in and the plugin reported a clean finish,
with the child's stderr — the only clue — discarded. The engines now require that
terminal event and fail with the child's exit code and stderr tail rather than quietly
producing nothing.

## Quick Start

### Linux / macOS

```bash
git clone https://github.com/chethan62/vlc-ai-subs.git
cd vlc-ai-subs
./install.sh
```

`install.sh` is the full installer (WhisperX + Parakeet + NLLB translate
models + whisper.cpp/Vulkan on machines without an NVIDIA GPU + ffmpeg check +
VLC extension sync; NLLB is skippable via
`VSCL_AISUBS_SKIP_NLLB=1`, whisper.cpp via `VSCL_AISUBS_SKIP_WHISPERCPP=1`,
and `VSCL_AISUBS_PARAKEET_V3=1` adds Parakeet's multilingual v3 model).
`setup.sh` is the minimal WhisperX-only
variant (no Parakeet, no NLLB, no whisper.cpp, no ffmpeg check).

The extension is installed into VLC's **user data dir**, so no root is needed; if
a system-wide VLC directory exists the installer also tries to copy there via
`sudo`, and declines gracefully when that isn't possible (no rights, a password
prompt you can't answer — e.g. `curl … | bash`). Every step reports its own
result, so a failed model download or build says so instead of printing a success
line.

### Windows

```powershell
git clone https://github.com/chethan62/vlc-ai-subs.git
cd vlc-ai-subs
setup.bat
```

Then:

1. **Restart VLC**
2. Open a video
3. **View → AI subtitle generator (WhisperX/Parakeet)**
4. Click **Generate**

> **Where you clone matters on Windows.** `setup.bat` installs the Lua extension
> and the WhisperX venv only — it does not copy the Python files. The extension
> looks for `aisubs_whisper.py` in `%USERPROFILE%\Documents\vlc-ai-subs`,
> `…\Desktop\vlc-ai-subs`, `…\Desktop\aisubs`, `…\vlc-ai-subs` or
> `%APPDATA%\vlc-ai-subs`, so clone into one of those (or move the files there).
> Parakeet and the NLLB/M2M translate cascade install via the Linux/macOS `bash`
> scripts, so a Windows translate task falls back to Whisper translate.

## Engines

| Engine | Languages | Word timing | Speed | License |
|---|---|---|---|---|
| **WhisperX** (default) | 99 (faster-whisper + wav2vec2 alignment) | wav2vec2 forced alignment (tightens cue times) | 2–4× realtime on a healthy GPU; **0.6× measured here** | BSD-2 + MIT |
| **Parakeet** (opt-in: `VSCL_AISUBS_BACKEND=parakeet`) | English (v2) or 25 European languages (v3) | **native TDT word timestamps** (no aligner) | **~10× faster, CPU-friendly** | CC-BY-4.0 |
| **whisper.cpp** (Vulkan: AMD/Intel too) | 99 (whisper.cpp / ggml) | segment-level (no aligner) | GPU via Vulkan, CPU fallback | MIT |

Pick **Parakeet** in the dialog for English films — self-reported mean WER
6.05% on the HF Open-ASR leaderboard (independent 2026 evals put
whisper-large-v3 at 7.44% on a comparable English eval), ~0.7 GB int8 model,
and its transducer decoder structurally avoids the hallucination loops
Whisper hits on music/silence.

Speeds measured on this box (EC-capped 300 MHz GTX 1650, 4 GB) rather than
quoted: **Parakeet 4m51s for a 47.5-min episode (~9.8× realtime, CPU)** versus
**WhisperX 8m11s for 5 min (~0.6× realtime, large-v3-turbo on GPU)** — i.e. on a
capped or small GPU Parakeet is roughly 16× faster per minute of audio, and the
whisper.cpp/Vulkan path measured 22.9s per 60s (~2.6× realtime).

Two variants, same 0.6B TDT architecture and the same native word timestamps:

| Variant | Languages | Size | Install |
|---|---|---|---|
| **v2** (default) | English — the English-specialised model | ~0.7 GB | `./install-parakeet-model.sh` |
| **v3** | 25 European languages — bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk | ~0.64 GB | `./install-parakeet-model.sh v3` |

Both can be installed side by side, and the language decides which one runs:
English keeps using v2, anything else uses v3 (no language flag is needed —
k2-fsa's ONNX conversion handles the multilingual prompt internally).
`VSCL_AISUBS_PARAKEET_V3=1 ./install.sh` installs v3 as well,
`VSCL_AISUBS_PARAKEET_VERSION=v2|v3` forces one variant, and
`VSCL_AISUBS_PARAKEET_MODEL=<dir>` points at a model directory of your own.

**Auto (the default)** picks the fastest engine that covers the language: an
installed Parakeet that supports it — English → v2, the other 24 → v3 — wins
over the hardware policy (NVIDIA → WhisperX, a Vulkan-only GPU (AMD/Intel) →
whisper.cpp, nothing usable → WhisperX on CPU).

Parakeet is only chosen for a **known** language: it has no language detection,
and unhinted decoding garbles non-English audio (measured on v3 — German came
out as "Alas hat an ende, no divorce tatzwai", while WhisperX's LID got it
right). So `auto` language, translate (no translation head) and any language no
installed variant covers all stay on the detecting engines — WhisperX or
whisper.cpp. Choosing Parakeet *explicitly* with an unsupported language falls
back to WhisperX with a note in the status line; choosing it explicitly with
`auto` language is honoured, so leave `auto` only for English-ish media.

### AMD / Intel GPUs (Vulkan)

WhisperX accelerates on NVIDIA CUDA only — faster-whisper/CTranslate2 has no
ROCm backend, so an AMD or Intel GPU gets *CPU speed* from it. The fix is
**whisper.cpp**, the one Whisper runtime with a Vulkan backend, and Vulkan is
vendor-neutral: the same binary drives Radeon, Arc and GeForce GPUs, falling
back to CPU when no device is present.

```bash
./install-whisper-cpp.sh small     # builds whisper.cpp with GGML_VULKAN=ON + ggml model
# then either pick "whisper.cpp (Vulkan …)" in the dialog, or leave Engine on Auto
```

- `install.sh` runs that automatically on machines **without** an NVIDIA GPU
  (set `VSCL_AISUBS_WHISPERCPP=1` to install it on an NVIDIA box too, or
  `VSCL_AISUBS_SKIP_WHISPERCPP=1` to skip it). Those branches are exercised by
  `tests/install_branches.sh` rather than left to inspection — including a failed
  Vulkan build, which warns and carries on so the rest of the install still
  completes.
- The engine needs no Python ML packages — only ffmpeg and the binary — so it
  works even when `venv-whisperx` is absent.
- Decode is hardened exactly as the WhisperX path is, in whisper.cpp's flags:
  `-mc 0` (no cross-window context — this build has no `--no-context`), `-nf`
  (no temperature fallback), `-bs 1` (beam 1) and `-sns`. With the defaults, a
  real 5-minute excerpt invented text over music and applause — `music playing`
  ×4, `applause`, `cheering`, `sighs`, and a **14-cue `I'm sorry.` repeat
  loop** — and took 231 s; hardened it takes **127 s (1.8× faster)** with none
  of those captions.
- Repeat loops that survive are collapsed in `core/blocklist.py`, which also
  folds a repeated phrase inside one cue. The threshold is calibrated against
  both real cases from this same episode: the loop ran to 14, while Parakeet —
  which does not loop — heard `Go, go, go, go, go!` and `Come on, come on, come
  on, come on.` as genuine dialogue. **Only runs of 7 or more collapse**, so real
  shouting is never eaten.
- Measured on this repo's dev box (a power-capped GTX 1650, so the gain here is
  a floor): **60 s clip, `small` → 22.9 s on Vulkan vs 32.8 s on CPU**.
  `VSCL_AISUBS_DEVICE=cpu` forces `-ng` (no GPU) for a comparison.
- Trade-offs vs WhisperX: segment-level timestamps only (no wav2vec2 word
  alignment) and `translate` uses Whisper's built-in `-tr`, not the NLLB
  cascade. Both are reported in the dialog's status line.
- Verified here on Vulkan/NVIDIA (`Vulkan GPU: NVIDIA GeForce GTX 1650`); AMD
  and Intel share this code path and binary flags but no such GPU was available
  in the development machine, so per-vendor driver behaviour is untested.

- WhisperX requires Python **< 3.14**, so it runs in its own Python 3.12 venv
  (`venv-whisperx`), launched as a subprocess with the same JSONL contract.
- The **"Recommended (auto)"** model is picked from GPU VRAM:
  large ≥ 8 GB, **large-v3-turbo ≥ 4 GB** (the 4 GB sweet spot — near-large
  accuracy at ~4× the speed), small ≥ 2 GB; CPU falls back to RAM-based sizing.
- To force CPU: `VSCL_AISUBS_DEVICE=cpu vlc`.
- WhisperX's wav2vec2 word alignment runs on **CUDA only** (one segment per pass
  — its API has no batch knob) and does two things: it re-segments a long
  transcript into cue-sized pieces, and its word timings **tighten each cue's
  start and end** to where speech actually begins and ends. On a translated run
  the words no longer match the audio, so the pass is skipped; if it fails, the
  status line names the real cause (e.g. a CUDA OOM) instead of blaming the
  language model.
- Without that pass — CPU, AMD/Intel, a translated run, whisper.cpp — Whisper
  hands back whole exchanges as a single segment: measured on a real episode,
  **8 cues for 5 minutes, the longest 29 s**. So `core/cues.py` splits any cue
  longer than 7 s at its own sentence boundaries, sharing the time in proportion
  to each piece's length (no word timings needed). On that same audio this gives
  40–52 cues, in line with the 54 the GPU alignment produced. No word is ever
  lost, a sentence with no terminator stays long, and a fragment too brief to
  read is folded into its neighbour.
- WhisperX decode is hardened for movies (beam 1, no cross-window
  conditioning, silence/hallucination gates) — research-backed, see
  `.research/final_report.md` §2.2.
- Translate uses the **NLLB-200 cascade** (WhisperX transcribes in the
  source language, NLLB-200 translates to English — research: ~44.7 BLEU
  CA→EN vs Whisper's built-in translate). The model is ~1.3 GB,
  **CC-BY-NC-4.0** (research/non-commercial); `VSCL_AISUBS_NLLB=0` reverts
  to Whisper translate, and unmapped languages fall back automatically.
- Parakeet decodes audio with `ffmpeg`, which must be on PATH — `install.sh`
  checks for it up front and aborts with a clear message otherwise (the
  runner also errors cleanly if ffmpeg is missing at runtime).
- Chunks holding no speech are **skipped**, via Silero VAD (`install-vad-model.sh`,
  optional, ~630 KB). Measured on a 60 s music/credits clip: 2 of 2 chunks skipped, 3 s
  instead of 10 s, and no text invented. The VAD is only ever asked *where the speech is* —
  it never removes a transcribed word, because whisper.cpp's own `--vad` was measured
  dropping real dialogue and a missing line is invisible.
- Long media is decoded in 30 s chunks (`CHUNK_SECONDS` in
  `parakeet_runner.py`; override with `VSCL_AISUBS_PARAKEET_CHUNK`). Chunk
  length is bounded by two *measured* limits of the int8 ONNX conversion, not by
  the model's design: peak memory grows ~1.17 GB fixed (model + onnxruntime)
  plus ~393 MB per minute of chunk audio, and length itself becomes fatal
  somewhere between 5 and 10 min (`onnxruntime` fails the encoder with
  `/layers.0/self_attn/Add_2 … broadcast an axis by a dimension other than 1.
  2500 by 7500`). The previous 20-min cap OOM-killed a 47.5-min episode at
  9.3 GB; at 30 s that same episode finishes in 4m51s at a 2.0 GB peak —
  ~9.8× realtime, with the model staying loaded across chunks.

## Models

| Model | Speed | Accuracy | RAM | Download |
|-------|-------|----------|-----|----------|
| `tiny` | Fastest | Basic | ~1 GB | ~75 MB |
| `base` | Fast | Good | ~1 GB | ~140 MB |
| `small` | Moderate | Better | ~2 GB | ~460 MB |
| `medium` | Slow | Great | ~5 GB | ~1.5 GB |
| `large` | Slowest | Best | ~10 GB | ~3 GB |
| `large-v3-turbo` | Fast | Near-large | ~2 GB | ~1.6 GB |

Models are downloaded from Hugging Face on first use (cached in `~/.cache/huggingface` by default).

## Environment Variables

| Variable | Values | Default | Applies to |
|----------|--------|---------|------------|
| `VSCL_AISUBS_BACKEND` | `whisperx` \| `parakeet` \| `whispercpp` (alias `whisper_cpp`) \| `auto` | `auto` | backend selection |
| `VSCL_AISUBS_DEVICE` | `cuda` \| `cpu` | auto | WhisperX (runner); `cpu` = `-ng` for whisper.cpp |
| `VSCL_AISUBS_PARAKEET_CHUNK` | seconds (5–600) | 30 | Parakeet chunk length — smaller = less RAM, larger = fewer seams |
| `VSCL_AISUBS_VAD_MODEL` | path to `silero_vad.onnx` | `~/.local/share/sherpa-onnx/models/` | where the VAD model lives; absent = no chunk skipping |
| `VSCL_AISUBS_COMPUTE` | `int8_float16` \\| `int8_float32` \\| `float16` \\| `float32` \\| `int8` | per device | WhisperX (runner) |
| `VSCL_AISUBS_MODEL_CACHE` | directory path | `~/.cache/huggingface` | WhisperX (runner) |
| `VSCL_AISUBS_NLLB` | `1` \| `0` | `1` | translate task (0 = Whisper translate) |
| `VSCL_AISUBS_NLLB_MODEL` | directory path | `~/.local/share/vlc-ai-subs/nllb-200-distilled-1.3B-int8` | translate task |
| `VSCL_AISUBS_NLLB_FAMILY` | `nllb` \| `m2m100` | `nllb` | cascade model family (m2m100 = MIT) |
| `VSCL_AISUBS_BLOCKLIST` | `1` \| `0` | `1` | hallucination-phrase filter (research §2.2) |
| `VSCL_AISUBS_DEBUG` | `1` \| unset | unset | debug logs to stderr + `/tmp` (main CLI, runners) |
| `VSCL_AISUBS_SKIP_NLLB` | `1` \| unset | unset | `install.sh` only: skip the NLLB model download |
| `VSCL_AISUBS_TIMEOUT` | seconds (`0` = no limit) | 4 h transcribe / 6 h translate | backend subprocess ceiling (long films) |
| `VSCL_AISUBS_MAX_LINE` | characters (min 8) | 42 (16 CJK, 13 Japanese) | cue line width |
| `VSCL_AISUBS_MAX_CPS` | characters/second (1–60) | 20 (9 CJK, 4 Japanese) | reading-speed ceiling |
| `VSCL_AISUBS_AUDIO_TRACK` | index, position or language | auto (match the requested language) | which audio stream to transcribe |
| `VSCL_AISUBS_WHISPERCPP_BIN` | path to `whisper-cli` | auto-detected | whisper.cpp engine |
| `VSCL_AISUBS_PARAKEET_V3` | `1` | `0` | `install.sh` only: also install the multilingual v3 model |
| `VSCL_AISUBS_PARAKEET_VERSION` | `v2` \| `v3` | auto (by language) | Parakeet: force one model |
| `VSCL_AISUBS_PARAKEET_MODEL` | directory | auto-detected | Parakeet: use this model directory |
| `VSCL_AISUBS_WHISPERCPP_MODEL` | path to a `ggml-*.bin` | best installed | whisper.cpp engine |
| `VSCL_AISUBS_WHISPERCPP` | `1` installs the Vulkan build on NVIDIA boxes too | unset | `install.sh` |
| `VSCL_AISUBS_SKIP_WHISPERCPP` | `1` skips the whisper.cpp build | unset | `install.sh` |

## Architecture

```
aisubs.lua                   VLC extension (dialog + timer polling)
aisubs_whisper.py            CLI entry-point (args → backend → JSONL → SRT)
whisperx_runner.py           WhisperX inside the Python 3.12 venv (subprocess)
parakeet_runner.py           Parakeet TDT via sherpa-onnx (same JSONL contract)
whispercpp_runner.py         whisper.cpp via whisper-cli — Vulkan (AMD/Intel/NVIDIA)
nllb_translate.py            NLLB-200 / M2M-100 translate cascade (ctranslate2)
core/
  emitter.py                 JSONL + Lua poll-mirror output
  srt.py                     SRT timestamp formatting + file writing
  cues.py                    cue line-wrapping + timing quality pass
  blocklist.py               hallucination-phrase filter (VSCL_AISUBS_BLOCKLIST)
  audio.py                   ffmpeg decode to 16 kHz mono wav
  gpu.py                     NVIDIA/Vulkan capability probes (engine choice)
  parakeet_models.py         Parakeet v2/v3 variants: installed models, language sets
  procs.py                   live-child registry (cancellation)
  timeouts.py                backend subprocess ceilings (VSCL_AISUBS_TIMEOUT)
backends/
  base.py                    TranscriptionBackend ABC
  whisperx_backend.py        WhisperX (Python 3.12 subprocess, PYTHONPATH-cleaned)
  parakeet.py                Parakeet (sherpa-onnx, v2 English / v3 multilingual)
  whispercpp.py              whisper.cpp (Vulkan: AMD/Intel/NVIDIA, or CPU)
```

**JSONL contract (stdout):** `{"type":"status","msg":...}`, `{"type":"sub","i":N,"start":S,"end":E,"text":...}`, `{"type":"done","segments":N,"srt_path":...}`, `{"type":"error","msg":...}`. Lua polls the mirror file (argv[5]) for progress.

## VLC 4.0 readiness

VLC 4.0 is still unreleased (VideoLAN's release page lists 3.0.x and older) and
its Linux nightlies are snap-only, so this plugin is **verified on VLC 3.x**.
What *can* be checked without a 4.0 binary is the Lua API it depends on — done
against VLC master (4.0-dev) on 2026-09-15:

| API the plugin uses | in VLC master |
|---|---|
| `vlc.input.item()`, `vlc.input.add_subtitle()` | present (`modules/lua/libs/input.c`) |
| `vlc.osd.message(text, chan?, pos?, dur?)`, `channel_register()` | present, all args optional |
| `vlc.config.userdatadir()` | present (`configuration.c`) |
| `vlc.dialog` and every widget used here | present (`dialog.c`) |
| `vlc.object.input()`, `vlc.var.set()` | present (`objects.c`, `variables.c`) |
| Lua extensions themselves | still supported (`extension.c`, `extension_thread.c`) |

There is **no `vlc.player` table** in 3.0.x or in master, though several plugins
try one. Those calls survive only as a last-ditch `pcall`; `vlc.input.*` is the
path that runs. (The test harness used to stub `vlc.player.*`, which is how the
fiction went unnoticed — it now models the real API and fails if the plugin
depends on anything else.)

So: 3.x works today, and 4.0 should load unchanged — but that is a source-level
claim, not a run against a 4.0 build, and it is described as such here.

Two details matter for *loading*, and both check out the same way:

- **The scan-time environment is still bare.** `ScanLuaCallback()` in 4.0
  creates a fresh `luaL_newstate()` with only a dummy `require` before running
  the file — no `io`, `os` or `math` — identical to 3.0's batch scan. This
  plugin's top level is deliberately conservative for that reason (all dialog
  work happens inside functions, `math.randomseed` is guarded); that is exactly
  what registering in VLC 3.0.23 exercises.
- **The user directory is unchanged.** 4.0 still resolves the user script dir
  through `config_GetUserDir(VLC_USERDATA_DIR)`, so
  `~/.local/share/vlc/lua/extensions/` remains correct. (4.0 adds zip-packaged
  `.vle` extensions as an extra format; the plain `.lua` file still works.)

**How far the 4.0 claim was pushed here:** no Linux 4.0 binary is obtainable
(VideoLAN's `master-daily` PPA reports *Failed to build* for `vlc`, and the Linux
nightlies are snap-only), so a real 4.0.0-dev **win64** build was run under Wine
instead — revision `4.0.0-dev-39018-g86363b2f28`, built 2026-09-15:

- it starts, loads 597 plugin modules including the Lua plugin (`--enable-lua` in
  the build's own configure line), and resolves the user script dir exactly as
  the source predicts (`C:\users\<user>\AppData\Roaming\vlc\lua\…`);
- its Lua machinery runs from that layout: the log shows the batch scan
  (`Trying Lua scripts in …`) executing the build's shipped `.luac` scripts —
  the same code path extensions go through;
- it ships an extension of its own (`lua/extensions/VLSub.luac`), so the
  extension folder and compiled-script support are exercised by the build.

What could **not** be triggered headlessly is this plugin's own registration:
4.0 creates the extensions manager **lazily from its Qt UI** — no scan happens at
startup, while playing media, or under `-I dummy`; it appears only when the GUI's
Extensions view is opened, which needs GUI automation. So the honest summary is:
in a real 4.0 build the APIs, the script directories, the scan environment and
the Lua runtime are all verified present and working; this plugin's own
registration in 4.0 is *inferred* from that (plus its registration in 3.0.23),
not observed.

**The check that paid off: 4.0 embeds a different Lua.** 3.0.23 links
`liblua5.1`; 4.0 bundles **Lua 5.4.4** (read from the plugin's `$LuaVersion`
string). Running the UI harness under `lua5.4` — the first time any test here
did — found three real 4.0-only breakages, all now fixed:

| symptom on 4.0 | why Lua 5.1 hid it |
|---|---|
| `math.randomseed` aborted the whole file at runtime | 5.4 rejects a float seed with no exact integer representation; 5.1 has only floats and accepts it |
| `string.format("%ds", eta)` raised, killing the progress tick | same integer rule for `%d`: a fractional ETA (7 s clip × 0.5 = 3.5) is rejected by 5.4, silently truncated by 5.1 |
| the Windows launch path reported *"failed to launch Python"* even on success | `os.execute` returns an exit code in 5.1 but `true/"exit"/code` in 5.4, so `ok ~= 0` inverted on 4.0 |

Both harnesses now run under **`lua5.1` and `lua5.4`** (103 + 25 checks, 0 failed
on each), and reverting any of the three fixes fails a check or aborts the load
under 5.4 while still passing under 5.1 — which is exactly the asymmetry that
had been hiding this class of bug.

## Testing

### Automated tests (dev)

```bash
cd vlc-ai-subs
python3 -m venv venv && venv/bin/pip install pytest              # one-time
PYTHONPATH= venv/bin/python -m pytest tests/ -v               # suite: 253 tests (model-free)
bash tests/install_branches.sh                               # installer branch matrix: 19 checks
```

The installer's engine-selection and failure-reporting branches — the ones that
never run on a CUDA box — are covered by `tests/install_branches.sh`. It runs the
real `install.sh`/`setup.sh` against a sandbox (private `HOME`, a curated `PATH`
so `nvidia-smi` can be present or absent, stubbed downloads, and a `sudo` that
always refuses) and asserts: AMD/Intel → the Vulkan engine installs itself,
NVIDIA → it stays opt-in, `VSCL_AISUBS_SKIP_WHISPERCPP=1` wins anywhere, an
existing build is not rebuilt, a failed Vulkan build does not abort the install,
a denied `sudo` still lands the extension in the user data dir, and a hard
failure is never reported as success.

Coverage: SRT formatting (float-drift-safe rounding, rollover, clamp),
cue wrapping + timing cleanup (word-boundary/balanced/CJK, min duration/gap),
JSONL emitter + mirror file, VRAM/RAM model recommendation (boundary cases),
backend resolution (WhisperX default, Parakeet opt-in, whisper.cpp + its
`whisper_cpp` alias, the auto hardware policy, missing-backend errors),
Parakeet v2/v3 variant selection (language sets, forced version, relocated
model directories) + the auto engine's language rule, runner CLI errors,
Parakeet token-merge + cue grouping, whisper.cpp argv/JSON
(ms→s) parsing + model resolution, NLLB/M2M batch
translation + language-map fallback, hallucination-blocklist matching,
backend subprocess timeouts + cancellable children (real processes), model-name
resolution, the runners'
SRT side-effect guard, and the
main CLI's JSONL error contract. No WhisperX
model download needed — transcription is out of scope for unit tests.

The `PYTHONPATH=` prefix neutralizes any foreign `PYTHONPATH` exported by
the calling shell (e.g. a desktop-agent terminal), which would otherwise
shadow packages with an unrelated interpreter's site-packages.

### End-to-end (installed plugin)

```bash
# 1. Run the backend directly (bypasses VLC). The CLI is stdlib-only, so any
#    python3 works — install.sh does not create a CLI venv of its own.
python3 ~/.local/share/vlc-ai-subs/aisubs_whisper.py \
  /path/to/video.mp4 recommended auto translate

# 2. Check the .srt file written next to the video
ls -la /path/to/video.srt

# 3. Confirm WhisperX CUDA is active (watch the status lines)
~/.local/share/vlc-ai-subs/venv-whisperx/bin/python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

In VLC: restart → open a video → **View → AI subtitle generator (WhisperX/Parakeet)** → click **Generate**.

## Debugging

Add `--debug` to the CLI args (VLC already appends it to every run it
launches) or set `VSCL_AISUBS_DEBUG=1`. This writes:

| File | Contents |
|------|----------|
| `/tmp/aisubs_debug.log` | main CLI phases + timings (args, backend, model pick, segment count, elapsed) |
| `/tmp/aisubs_whisperx.log` | full subprocess stdout+stderr dump (WhisperX runtime logs) |
| `/tmp/aisubs_parakeet.log` | full subprocess stdout+stderr dump (Parakeet runtime logs) |

Debug lines are also mirrored to stderr, so they appear in VLC's own logs
(`vlc -vvv`). Failed WhisperX runs additionally include the stderr tail and
stdout's last JSONL line in the emitted error — no more silent failures.

The CLI also writes its PID to `<mirror>.pid` while it runs (removed on exit):
that is how the extension's **Cancel** button stops a run — it signals the PID,
and the CLI's handler then terminates the model subprocess (`core/procs.py`).
The extension removes the mirror, the temp SRT and the pid file afterwards.

## Options

- **Engine** — Auto (default: the fastest engine that covers the language), WhisperX (multilingual, word-aligned), Parakeet (fastest — v2 for English, v3 for 25 European languages) or whisper.cpp (Vulkan — AMD/Intel GPUs).
- **Model** — `Recommended (auto)` (VRAM-aware) or `tiny` / `base` / `small` / `medium` / `large` / `large-v3-turbo`.
- **Language** — `auto` for detection, or a code like `en`, `es`, `fr`, `hi`, `ja`, `zh`, `en-US`, etc.
- **Task** — `Translate to English` (default) or `Transcribe (same language)`.
- **Mode** — `Generate & Load SRT` (default) or `Real-time OSD`.
- **Failures read as one line** — a backend error is shown as a short, actionable
  status (`… | Out of GPU memory (VLC itself holds some): close other video
  windows, choose a smaller model, or use Parakeet`), while the full traceback goes
  to VLC's log and `/tmp/aisubs_debug.log`. It used to paste the entire
  stderr/stdout — measured at ~4 KB of warnings — into the status label.
- **Closing the dialog mid-run is safe** — it cancels the progress timer (VLC
  keeps calling a live timer after the dialog is deleted, and the callback then
  indexed the removed progress bar: `attempt to index upvalue 'progress_bar' (a
  nil value)`, seen in a real run) and the transcription still finishes, writing
  its SRT.
- **Cancel** — stops the run (its process tree, including the model subprocess); starting a new run cancels the previous one.
- **Remembered settings** — engine/model/language/task/mode are stored in `<vlc user data dir>/vlc-ai-subs/settings.conf` and restored next session; the details pane shows the engine, model, elapsed, ETA and cue count, plus the latest transcribed cue.

## Manual Installation

If the setup script doesn't work for your system:

1. Install WhisperX (Python 3.12 venv):
   ```bash
   uv venv --python 3.12 venv-whisperx
   uv pip install --python venv-whisperx/bin/python whisperx
   ```
   The default engine and the SRT path are now complete. For the optional
   engines/models:
   ```bash
   uv pip install --python venv-whisperx/bin/python sherpa-onnx  # Parakeet runtime
   bash install-parakeet-model.sh                                # Parakeet model (~0.7 GB)
   bash install-vad-model.sh                                     # Silero VAD (~630 KB, optional)
   bash install-nllb-model.sh                                    # translate cascade (~1.3 GB)
   bash install-whisper-cpp.sh small                             # Vulkan engine (AMD/Intel/NVIDIA)
   ```
2. Copy `aisubs.lua` to your VLC extensions folder:
   - **Linux**: `~/.local/share/vlc/lua/extensions/`
   - **macOS**: `~/Library/Application Support/org.videolan.vlc/lua/extensions/`
   - **Windows**: `%APPDATA%\vlc\lua\extensions\`
3. Place the Python files (`aisubs_whisper.py`, `whisperx_runner.py`, `nllb_translate.py`, `parakeet_runner.py`, `core/`, `backends/`) next to `venv-whisperx` (the backend falls back to `~/.local/share/vlc-ai-subs/`).
4. Restart VLC.

## Credits

**Original plugin** — this fork extends someone else's work:

- [voidrlm/vlc-ai-subs](https://github.com/voidrlm/vlc-ai-subs) — the original VLC plugin (Lua extension)

**Host and audio:**

- [VideoLAN — VLC](https://www.videolan.org/vlc/) ([source](https://code.videolan.org/videolan/vlc)) — the player and its Lua extension API
- [FFmpeg](https://ffmpeg.org/) — audio decoding to 16 kHz mono

**Speech recognition:**

- [OpenAI Whisper](https://github.com/openai/whisper) — the model family every engine here runs
- [m-bain/whisperX](https://github.com/m-bain/whisperX) — word-level forced alignment
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) with [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) — the WhisperX inference stack
- [wav2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) (Meta AI) — the alignment models WhisperX aligns with
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — the Parakeet-TDT architecture behind `parakeet_runner.py`
- [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) · [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — the Parakeet models (CC-BY-4.0)
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — Parakeet inference, and the int8 ONNX conversions of both Parakeet models
- [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) — the Vulkan (AMD/Intel/NVIDIA) and CPU Whisper runtime

**Translation:**

- [NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb) (Meta AI, CC-BY-NC-4.0) — the default translate cascade
- [M2M-100](https://github.com/facebookresearch/fairseq/tree/main/examples/m2m_100) (Meta AI, MIT) — the commercial-friendly cascade

**Models and data:**

- [Hugging Face](https://huggingface.co/) — model hosting, and the [Open ASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) that the WER figures quoted in this README come from

Nothing third-party is vendored in this repo: models and runtimes are downloaded
or pip-installed at install time, which is why attribution lives here and in the
license table rather than in a NOTICE file. The two that actually *require* it —
Parakeet (CC-BY-4.0, © NVIDIA, converted by the k2-fsa team) and sherpa-onnx
(Apache-2.0, © Next-gen Kaldi) — are credited above.

## License

Plugin code: **MIT** (see LICENSE). Models are separate runtime downloads, each
with its own license — checked per model card:

| Model | License | Use in the plugin |
|---|---|---|
| faster-whisper (MIT) / WhisperX (BSD-2-Clause) | MIT + BSD-2 | default engine (multilingual, word-aligned) |
| OpenAI Whisper | MIT | the model family all engines run; the `translate` fallback |
| Parakeet-TDT-0.6B v2 (English) / v3 (25 European languages) | **CC-BY-4.0** (commercial OK, attribution required) | Parakeet engine — © NVIDIA, ONNX conversion by [k2-fsa](https://github.com/k2-fsa/sherpa-onnx) |
| whisper.cpp + ggml models | MIT | Vulkan (AMD/Intel/NVIDIA) + CPU engine |
| sherpa-onnx | Apache-2.0 | Parakeet inference runtime |
| NLLB-200 (translate cascade, default) | **CC-BY-NC-4.0** | personal / non-commercial |
| M2M-100 1.2B (translate cascade, optional) | **MIT** | commercial use |

**Translate license decision**: NLLB-200 is the default cascade model — best
quality (research: 44.7 BLEU CA→EN) and its CC-BY-NC-4.0 license permits
personal/non-commercial use, which covers this plugin's typical use. It is
flagged at install time and skippable (`VSCL_AISUBS_SKIP_NLLB=1`), and
`VSCL_AISUBS_NLLB=0` reverts to Whisper translate (MIT). For commercial
deployment, install the MIT-licensed M2M-100 cascade instead:

```bash
bash install-m2m-model.sh          # sentencepiece + M2M-100 1.2B (~2.5 GB, fp16 weights, int8 compute)
VSCL_AISUBS_NLLB_FAMILY=m2m100 VSCL_AISUBS_NLLB_MODEL=~/.local/share/vlc-ai-subs/m2m100_1.2B-int8 vlc
```

M2M-100 covers 100 languages (vs NLLB's 200) at somewhat lower quality — the
trade for a permissive license. Unmapped languages fall back to Whisper
translate in either family.