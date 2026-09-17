# The 2026 ASR landscape, and what CrispASR could change here

Reconnaissance for "what new software could improve this plugin", aimed at the defects we have
actually measured rather than at benchmark charts. Everything below is either a published claim
(attributed) or a measurement taken on this project's own material (marked).

## The constraint that decides everything

This laptop's dGPU is hard-capped (~300 MHz, ~15.5 W) and WhisperX here runs ~6× slow, so the
factory is **CPU**: Parakeet TDT 0.6b-v3 via sherpa-onnx int8, whisper.cpp, Silero VAD. Any
engine that needs a real GPU is not usable *here* — but the plugin ships to other machines, so
the design has to **scale up on a better box** rather than being tuned to this one. Both halves
of that matter and they pull in opposite directions.

## The field, 2026

| model | params | avg WER (claimed) | languages | CPU-viable | note |
| --- | --- | --- | --- | --- | --- |
| Canary-Qwen 2.5B | 2.5B | 5.63 | en | no (GPU, ~8× slower than Parakeet) | top of the Open ASR board for a while |
| Granite Speech 4.1-2B | 2B | 5.33 | en/fr/de/es/pt/ja | marginal | Apache-2.0 |
| Cohere Transcribe (03-2026) | 2B | 5.42 | 13-14 | marginal | claimed lowest WER on the board at launch |
| Qwen3-ASR | 1.7B / 0.6B | — | 52 | marginal / yes | timestamps, LID |
| Parakeet TDT | 0.6B / 1.1B | ~6.3 / ~8.0 | 25 EU / en | **yes** | what we already use |
| Whisper large-v3-turbo | 809M | 7.75 | 99 | yes (int8) | faster, not more accurate, than Parakeet |
| Moonshine | 27-331M | — | en + 6 | **yes** | speed/edge, not accuracy |
| MOSS-Music | 8B | — | — | no | **singing** ASR with word timestamps — the only thing that addresses the ♪ lines we cannot transcribe |

The thing that matters most for subtitles is not on that table at all: **word-level timing that
reflects speech rather than decoder emission**. Our measured defect — cue spans that end at the
last word's *start*, because a transducer reports the frame a token was emitted on — has
survived VAD-onset snapping (a measured no-op) and cue redistribution (made reading speed worse).

## CrispASR — one C++ ggml binary, no Python

`CrispStrobe/CrispASR` v0.8.33, MIT, a whisper.cpp fork extended into a multi-backend speech
engine. Relevant capabilities, all confirmed from its own CLI on this machine:

- ASR backends: **parakeet** (0.6b-v2/v3, 1.1b, ctc variants), **cohere**, **canary**,
  canary-qwen, **granite/granite-4.1**, **qwen3/qwen3-1.7b**, voxtral, kyutai, moonshine…
- `-am/--aligner-model` + `-falign/--force-aligner` — **CTC forced alignment** for word
  timestamps, plus a standalone `--align-only`
- `--diarize` + `--diarize-method` (`energy|xcorr|vad-turns|sherpa|pyannote|ecapa|foxnose`) and
  `--diarize-speakers` — pyannote segmentation + session-scoped clustering, **no HF token**
- `--vad`, `-sp/--split-on-punct`, `-osrt/-ovtt/-ojf` (JSON with word+token detail)
- Build matrix that matters for shipping to other machines:

| tarball | on a machine without that GPU |
| --- | --- |
| `…-cuda.tar.gz` / `…-cuda13.tar.gz` | **falls back to CPU** (CUDA is a `dlopen`'d module since v0.8.30) |
| `-hip` (AMD) / `-vulkan` | **no fallback** — link-time dependency |
| plain / `-cpu-legacy` / `-avx512` | CPU-only by construction |

So one CUDA tarball scales from this capped laptop to a 24 GB box; AMD needs a self-build with
`-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON` (or the vulkan binary with its no-fallback caveat).

## Measurements on our own material

90 s of the test film (`/var/tmp/aisubs-r26/window90.wav`), 8 threads, AVX2, no usable GPU.

**1. It recovers the dialogue our engine drops.** The passage our sherpa-onnx path silently
skipped — `"I told you no! You have the gift, but no control. No understanding of your purpose.
Not a clue as to what your situation will demand of you beyond today."` — comes back intact, in
one pass with no gap. Same Parakeet **0.6b-v3 weights**, different runtime (GGML q4_k vs ONNX
int8). That locates our skipping defect in the ONNX-int8/sherpa path, not in the model.

**2. Throughput, measured** (warm model cache, 2 runs each, 8 threads on this laptop):

| configuration | 90 s of audio | equivalent for the 145-min film |
| --- | --- | --- |
| parakeet, no aligner | **21.3 / 21.7 s** (≈4.2× realtime) | ~35 min |
| parakeet + CTC aligner | **36.9 / 36.8 s** (≈2.4× realtime) | ~61 min |
| our current engine (sherpa-onnx), for reference | — | 21 min (measured, `v1.4.4`) |

An earlier reading of 88.8 s for the aligner run was wrong: that first invocation
included the aligner model download. The real cost of the aligner on CPU is
**+72 %**, not +320 % — affordable enough that the choice between the two is a
quality decision rather than a budget one.

**3. The aligner fixes the thing we could not.** With `-am` + `-falign` + `-sp`: 25 cues for the
90 s, median span 2.72 s, **0 overlaps**, 2 cues over 7 s. The recovered line lands at
`47.60 → 50.96` — 3.36 s for 7 words, ≈125 wpm, which is plausible speech.

**3b. …and that was an assumption, not a measurement. Withdrawn.** Plausible-looking cue
structure is not sync. Scored against the reference track over six 90-second windows spread
across the film, at each window's best global offset, within 0.15 s:

| engine | cues | fit hits | rate |
| --- | --- | --- | --- |
| this plugin's existing engine (`film6`) | 159 | 68 | **42.8 %** |
| CrispASR, no aligner | 129 | 59 | **45.7 %** |
| CrispASR + CTC aligner | 129 | 58 | **45.0 %** |

The aligner does not improve sync — it is a fraction worse, inside noise — and neither engine
separates from the other. Per window the aligner won one, tied two and lost three. The claim
that alignment was "the one timing defect that survived VAD-onset snapping and cue
redistribution" was written *before* this test; the honest statement is that the aligner
produces timings from speech, that its cues look structurally sound, and that **no improvement
in alignment with professional timings has been demonstrated**. It stays available (free on a
GPU) and is not recommended on CPU, where it costs +72 % for no measured gain.

**4. It composes with our quality pass.** CrispASR's `-sp` splits at punctuation but does not
wrap, so raw output had **8 lines over 42 chars (longest 76)**. Running the existing
`apply_quality()` over it: **0 lines over 42 (longest 41)**, 0 cues over 7 s, 0 overlaps, and all
179 words preserved. The division of labour is clean — engine supplies text + timings, our pass
enforces the published cue standards.

## What I would do with this

1. **Add `crispasr` as a fourth backend**, device- and VRAM-tiered, reusing what exists:
   `core/gpu.py` for detection (extended past NVIDIA), `aisubs_whisper.py:_detect_vram_mb()` and
   the existing VRAM/RAM model tiering, and the `backends/_ENGINES` registry.
2. **Tier the model by VRAM**: CPU → Parakeet 0.6b (today's path, unchanged); ~4 GB → Parakeet
   1.1B / Qwen3-ASR 0.6B; ~6-8 GB → Cohere Transcribe / Granite 4.1-2B / Qwen3-ASR 1.7B; ~8-10 GB
   → Canary-Qwen 2.5B / Voxtral-Mini-3B.
3. **Make alignment the GPU default and a CPU opt-in.** On GPU it is nearly free and it is the
   only measured answer to our timestamp compression; on this laptop it costs ~1× realtime
   (~2.4 h for a feature film), which is too much to impose by default.
4. **Evaluate diarization next** — `--diarize-speakers` needs no HF token, and the professional
   track we compare against carries dialogue dashes in 201 of its 1988 cues (10 %) that we
   produce none of.

## Caveats, stated rather than implied

- Not yet measured: diarization quality, the full film through CrispASR on this box (≈34 min
  without the aligner, ≈2.4 h with it), and any GPU path — there is no usable GPU here.
- WER figures in the table are vendor/leaderboard claims, not measurements of ours; the numbers
  from our own material are the throughputs and cue statistics above.
- Model licences differ (Parakeet CC-BY-4.0, Cohere/Granite Apache-2.0, Canary-Qwen CC-BY-4.0) —
  relevant if this plugin is ever distributed with weights.
- CrispASR's own segmented output is *not* subtitle-compliant on its own (unwrapped lines); our
  pass is still required, which is the point of keeping the two layers separate.
