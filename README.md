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
| **Two modes** | Generate & Load SRT (default) or OSD captions (Real-time OSD — each cue is pushed to the OSD as it is produced) |
| **SRT output** | Standard `.srt` files written next to your video — compatible with Kdenlive, VLC, mpv, PotPlayer |
| **Readable cues** | Text is wrapped to ≤2 lines × 42 chars (CJK-aware) with min-duration and gap cleanup |
| **Cancel + memory** | Cancel a running transcription; the dialog remembers engine/model/language/task/mode |
| **Any language** | Auto-detection or specify a language code (`en`, `es`, `fr`, `hi`, `ja`, `zh`…) |
| **Translation** | Translate any language to English subtitles |
| **VLC 3.x now, 4.x-ready** | Every Lua API it uses is present in VLC 4.0's source (checked); no 4.0 build has been run yet |
| **Cross-platform** | Linux, macOS, Windows (native, snap, flatpak) |

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

### Windows

```powershell
git clone https://github.com/chethan62/vlc-ai-subs.git
cd vlc-ai-subs
setup.bat
```

Then:

1. **Restart VLC**
2. Open a video
3. **View → AI Subs Generator**
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
| **WhisperX** (default) | 99 (faster-whisper + wav2vec2 alignment) | wav2vec2 forced alignment | ~2–4× realtime (medium, GPU-dependent) | BSD-2 + MIT |
| **Parakeet** (opt-in: `VSCL_AISUBS_BACKEND=parakeet`) | English (v2) or 25 European languages (v3) | **native TDT word timestamps** (no aligner) | **~10× faster, CPU-friendly** | CC-BY-4.0 |
| **whisper.cpp** (Vulkan: AMD/Intel too) | 99 (whisper.cpp / ggml) | segment-level (no aligner) | GPU via Vulkan, CPU fallback | MIT |

Pick **Parakeet** in the dialog for English films — self-reported mean WER
6.05% on the HF Open-ASR leaderboard (independent 2026 evals put
whisper-large-v3 at 7.44% on a comparable English eval), ~0.7 GB int8 model,
and its transducer decoder structurally avoids the hallucination loops
Whisper hits on music/silence.

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
  `VSCL_AISUBS_SKIP_WHISPERCPP=1` to skip it).
- The engine needs no Python ML packages — only ffmpeg and the binary — so it
  works even when `venv-whisperx` is absent.
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
- WhisperX's wav2vec2 word alignment runs on **CUDA only**; on CPU the
  segments keep faster-whisper's timestamps (no alignment pass). Parakeet's
  native TDT word timestamps work on CPU as well.
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
- Long media is decoded in ≤20-min chunks (`CHUNK_SECONDS` in
  `parakeet_runner.py`) — the 0.6B TDT model ingests at most ~24 min in a
  single forward pass (`.research/asr_report.json`).

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
| `VSCL_AISUBS_COMPUTE` | `int8_float16` \\| `int8_float32` \\| `float16` \\| `float32` \\| `int8` | per device | WhisperX (runner) |
| `VSCL_AISUBS_MODEL_CACHE` | directory path | `~/.cache/huggingface` | WhisperX (runner) |
| `VSCL_AISUBS_NLLB` | `1` \| `0` | `1` | translate task (0 = Whisper translate) |
| `VSCL_AISUBS_NLLB_MODEL` | directory path | `~/.local/share/vlc-ai-subs/nllb-200-distilled-1.3B-int8` | translate task |
| `VSCL_AISUBS_NLLB_FAMILY` | `nllb` \| `m2m100` | `nllb` | cascade model family (m2m100 = MIT) |
| `VSCL_AISUBS_BLOCKLIST` | `1` \| `0` | `1` | hallucination-phrase filter (research §2.2) |
| `VSCL_AISUBS_DEBUG` | `1` \| unset | unset | debug logs to stderr + `/tmp` (main CLI, runners) |
| `VSCL_AISUBS_SKIP_NLLB` | `1` \| unset | unset | `install.sh` only: skip the NLLB model download |
| `VSCL_AISUBS_TIMEOUT` | seconds (`0` = no limit) | 4 h transcribe / 6 h translate | backend subprocess ceiling (long films) |
| `VSCL_AISUBS_MAX_LINE` | characters (min 8) | 42 (20 for CJK) | cue line width |
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

**Why no 4.0 run in this repo's verification:** VideoLAN's `master-daily` PPA
(their only 4.0 channel that isn't a snap) currently reports *Failed to build*
for `vlc` on amd64, and the Linux nightlies are snap-only. A container with the
distribution package installs 3.0.x instead, so it would prove nothing about 4.0.

## Testing

### Automated tests (dev)

```bash
cd vlc-ai-subs
python3 -m venv venv && venv/bin/pip install pytest              # one-time
PYTHONPATH= venv/bin/python -m pytest tests/ -v               # suite: 175 tests (model-free)
```

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

In VLC: restart → open a video → **View → AI Subs Generator** → click **Generate**.

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