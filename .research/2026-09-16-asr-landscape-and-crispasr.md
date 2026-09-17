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

## Addendum — the recall measurement, which is the real reason to keep this engine

The aligner claim above did not survive scoring. The engine's actual strength showed up when the
test was aimed at the plugin's *worst* material instead of at a random window.

Six regions were selected because **our own engine fails them**: stretches where the
professional track has dialogue and our output has almost none (found by a coverage pass; lyrics
and on-screen text excluded). Each was extracted (60 s, memory-bounded) and transcribed by
CrispASR, then both engines were scored on how much of the reference's vocabulary they produced
in that stretch, in container time:

| region (s) | reference words | our engine | CrispASR |
| --- | --- | --- | --- |
| 959 | 91 | 27 % | **53 %** |
| 7749 | 112 | 71 % | 71 % |
| 5342 | 54 | **0 %** | **85 %** |
| 2293 | 66 | 27 % | 32 % |
| 5292 | 63 | 27 % | **68 %** |
| 3671 | 41 | **0 %** | **39 %** |
| **total** | **427** | **139 (33 %)** | **254 (59 %)** |

**How to read this.** The sample was chosen *because* our engine fails it, so it measures the
size of the gap, not overall accuracy — it is not a claim that CrispASR is 1.8× better in
general. Six regions and 427 words is a small sample, and word coverage is a bag-of-words proxy
that ignores order. What it establishes is narrow and useful: **when a line is missing from our
output, this engine is materially more likely to have it**, and two of these regions are cases
where we produced nothing at all and it produced 85 % and 39 %.

That is the reason to keep the backend, and it is a different reason from the one v1.5.0 gave.
The lesson generalises: an engine's advertised feature (here, forced alignment) may measure as
nothing while the same engine quietly fixes your worst failure mode (recall). Aim the test at
your own failures, measure both, and let the numbers choose the positioning.

## Addendum 2026-09-17 — it crashes on a feature-length file, and my earlier memory claim did not hold

Everything above rests on runs of 90 seconds or less. That was not stated as a limitation,
and it is one.

A full 47.5-minute episode (`Lucky.S01E01`) was run end to end through this backend to get a
clean-file comparison. **It never finished.** After ~31 minutes the binary died, and the kernel
log gives the mechanism:

```
__vm_enough_memory: pid: 76643, comm: crispasr, bytes: 123601297408 not enough memory for the allocation
crispasr[76643]: segfault at 0 ip ... error 4 in crispasr ... signal 11/SEGV
```

A request for a **123.6 GB** allocation — not gradual growth; resident memory was a healthy
1.0 GB at the time — refused by the kernel, whose NULL result the binary dereferenced instead of
handling (frame: `process_one_input`). That is an upstream bug of the unchecked-allocation kind,
and **`--chunk-seconds` does not prevent it**: this run was chunked from the start.

Two corrections follow, both to my own work:

1. **"Chunked at 300 s peaks at 1.2 GB" (v1.5.2) was not a supported claim.** The measurement
   that produced it came from a run that was *also* never verified to complete — the earlier
   20-minute run left no output file either. Memory behaviour was measured; completion was not,
   and I reported the former as though it settled the latter. **No CrispASR run longer than
   ~90 seconds has completed on this machine.** Chunking addresses the OOM path (a real, separate
   failure at 8.3 GB + 4.8 GB swap); it does not address this one.
2. **The engine's practical envelope here is short files**, which is what the plugin's opt-in
   status should be read as. The recall advantage on gap regions (59 % vs 33 %, measured on
   20–45 s clips) stands unchanged; it says nothing about long-form reliability.

Three fixes came out of it, and they generalise to every engine:

- **Name the signal.** `rc=-11` tells a user nothing; `killed by SIGSEGV`, with an explicit note
  that a crash writes to the kernel log rather than stderr, does.
- **Read the runner's error event before judging its exit code.** CrispASR's runner exits 1 *after*
  emitting a precise error, and the backend's early `returncode != 0` check replaced that message
  with a bare `rc=1`. The other three backends already checked last; this one did not.
- **Retry once, loudly, rather than returning nothing.** If an engine dies before producing a
  single cue, the CLI falls back to the engine the hardware policy would have chosen and says so
  in the status line. Only when zero cues were emitted — partial output must never be silently
  swapped for a differently-shaped cue list.

**Upgrade path, when upstream fixes the crash** (from its own docs, not measured): the binary has
a server mode — `crispasr --server` — that loads the model **once** and takes bounded requests via
`offset_t_ms` / `duration_ms`. That is the right shape for long media: our current chunking
restarts the process per chunk (~419 s each, measured 31 minutes to get nowhere on a 47.5-minute
file), while the server would chunk in-process with no reload. Not implemented deliberately —
the engine cannot complete a long file at all yet, so speeding that path up is premature. Its
troubleshooting page classifies our exit as `139` = "a genuine bug — please report it", which is
where the real fix has to come from.

Verified end to end with a deliberately crashing engine (a stub that raises SIGSEGV), before and
after — and the real run's own output matches the "before" exactly, with `free` showing 2.7 GB
free at the moment of failure, which is what rules memory out:

```
real:    Transcription failed: CrispASR failed (rc=1):          <- empty tail, no diagnosis
before:  crispasr (ggml) failed (CrispASR failed (rc=1):); retrying with parakeet (fast)
after:   crispasr (ggml) failed (CrispASR failed (killed by SIGSEGV): no output on stderr
         (a crash reports to the kernel log)); retrying with parakeet (fast)
```

## Caveats, stated rather than implied

- Not yet measured: diarization quality, and any GPU path — there is no usable GPU here.
- The full film through CrispASR was attempted on 2026-09-17 and **the binary crashed** (see the
  addendum); the only completed runs are 90 seconds or shorter, so nothing above should be read
  as a statement about long-form reliability.
- WER figures in the table are vendor/leaderboard claims, not measurements of ours; the numbers
  from our own material are the throughputs and cue statistics above.
- Model licences differ (Parakeet CC-BY-4.0, Cohere/Granite Apache-2.0, Canary-Qwen CC-BY-4.0) —
  relevant if this plugin is ever distributed with weights.
- CrispASR's own segmented output is *not* subtitle-compliant on its own (unwrapped lines); our
  pass is still required, which is the point of keeping the two layers separate.
