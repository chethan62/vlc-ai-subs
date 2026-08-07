# vlc-ai-subs

VLC media player plugin that generates subtitles using local AI — works offline
with any video, any language. Transcribe and translate into clean `.srt` files
or real-time on-screen captions.

[Upstream](https://github.com/voidrlm/vlc-ai-subs) · [Install](#quick-start) · [Backends](#engines--backends) · [GPU](#gpu-acceleration) · [Env vars](#environment-variables)

## Features

| | |
|---|---|
| **Zero-config** | "Recommended (auto)" model + Translate to English by default — open a video, click Generate, done |
| **GPU acceleration** | Vulkan (whisper.cpp), CUDA (faster-whisper), auto-detected — works on NVIDIA, AMD, Intel |
| **Multi-engine** | 6 backends: whisper.cpp, WhisperX (word-aligned), faster-whisper, sherpa-onnx, Moonshine, openai-whisper |
| **Two modes** | Real-time OSD (subtitles appear as they're generated) or Generate & Load SRT |
| **SRT output** | Standard `.srt` files written next to your video — compatible with Kdenlive, VLC, mpv, PotPlayer |
| **Any language** | Auto-detection or specify a language code (`en`, `es`, `fr`, `hi`, `ja`, `zh`…) |
| **Translation** | Translate any language to English subtitles |
| **VLC 3.x & 4.x** | Works with current and next-gen VLC |
| **Cross-platform** | Linux, macOS, Windows (native, snap, flatpak) |
| **Pluggable** | Add a new STT engine by dropping a single Python file in `backends/` |

## Quick Start

### Linux / macOS

```bash
git clone https://github.com/voidrlm/vlc-ai-subs.git
cd vlc-ai-subs
./setup.sh
```

Or use this fork with GPU acceleration and more engines:

```bash
git clone https://github.com/chethan62/vlc-ai-subs.git
cd vlc-ai-subs
./setup.sh                         # basic faster-whisper install
./install-whisper-cpp.sh small     # Vulkan GPU acceleration (recommended)
./install-sherpa-onnx-model.sh small.en  # optional: ONNX Whisper model
```

### Windows

```powershell
git clone https://github.com/voidrlm/vlc-ai-subs.git
cd vlc-ai-subs
setup.bat
```

Then:

1. **Restart VLC**
2. Open a video
3. **View → AI Subs Generator**
4. Click **Generate**

## Engines / Backends

The plugin auto-picks the best available engine. Priority order:

| # | Backend | Runtime | GPU | Quality | Notes |
|---|---------|---------|-----|---------|-------|
| 1 | **whisper.cpp** | GGML | Vulkan | ★★★★ | Fastest GPU path; single-binary, no Python deps |
| 2 | **WhisperX** | faster-whisper | CUDA | ★★★★★ | Word-level alignment for perfect subtitle sync |
| 3 | **faster-whisper** | CTranslate2 | CUDA | ★★★★ | CUDA with int8_float16 for 4GB cards |
| 4 | **Moonshine** | ONNX | CPU | ★★☆ | Near-instant, good for quick preview |
| 5 | **sherpa-onnx** | ONNX Runtime | CPU/CUDA | ★★★ | Whisper + SenseVoice + Paraformer via ONNX |
| 6 | **openai-whisper** | PyTorch | CPU | ★★★ | Last-resort CPU fallback |

Force a specific backend: `VSCL_AISUBS_BACKEND=whisper_cpp vlc`

## GPU Acceleration

### Vulkan (recommended)

One command:

```bash
./install-whisper-cpp.sh small
```

Builds whisper.cpp v1.9.2 with GGML_VULKAN=ON into `~/.local/share/whisper-cpp/`,
downloads a ggml model, and wires it up. No CUDA toolkit needed — works on
NVIDIA (GTX 1650+), AMD (RDNA2+), and Intel Arc GPUs.

Model sizes: `tiny` (75 MB) · `base` (140 MB) · **`small` (466 MB, default)** · `medium` (1.5 GB) · `large` (3 GB).

### CUDA

faster-whisper auto-detects a CUDA device and runs with `int8_float16`
(VRAM-friendly on ≤4GB cards). The backend pre-loads CUDA runtime libs
via ctypes — no `LD_LIBRARY_PATH` needed.

To force CPU: `VSCL_AISUBS_DEVICE=cpu vlc`.

### sherpa-onnx (ONNX)

```bash
pip install sherpa-onnx soundfile
./install-sherpa-onnx-model.sh small.en
```

Downloads a Whisper ONNX model from [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models). Supports Whisper, SenseVoice, Paraformer, and Qwen3-ASR architectures.

## Environment Variables

| Variable | Values | Default | Applies to |
|----------|--------|---------|------------|
| `VSCL_AISUBS_BACKEND` | `whisper_cpp` \| `whisperx` \| `faster_whisper` \| `moonshine` \| `sherpa_onnx` \| `openai_whisper` | auto | All |
| `VSCL_AISUBS_DEVICE` | `cuda` \| `cpu` \| `auto` | auto | faster-whisper, WhisperX |
| `VSCL_AISUBS_COMPUTE` | `int8_float16` \| `int8_float32` \| `float16` \| `float32`… | auto | faster-whisper, WhisperX |
| `VSCL_AISUBS_MODEL_CACHE` | directory path | `~/.cache/huggingface` | faster-whisper |
| `VSCL_AISUBS_WORDSUB` | `1` to enable word-level subtitles | off | WhisperX |

## Architecture

```
aisubs_whisper.py          ← CLI entry-point (parses args, resolves backend)
├── core/
│   ├── emitter.py         ← JSONL + Lua poll-mirror output
│   ├── device.py          ← CUDA lib preload + device/compute detection
│   └── srt.py             ← SRT timestamp formatting + file writing
└── backends/
    ├── base.py            ← TranscriptionBackend ABC
    ├── whisper_cpp.py     ← whisper.cpp (GGML Vulkan/CPU)
    ├── whisperx.py        ← WhisperX (word-aligned, CUDA)
    ├── faster_whisper.py  ← faster-whisper (CTranslate2 CUDA/CPU)
    ├── moonshine.py       ← Moonshine (ONNX, ultra-fast)
    ├── sherpa_onnx.py     ← sherpa-onnx (Whisper + SenseVoice + Paraformer)
    └── openai_whisper.py  ← openai-whisper (PyTorch CPU)
```

## How It Works

| Mode | Description |
|------|-------------|
| **Real-time OSD** | Subtitles appear on screen as the engine transcribes. Great for first-time viewing. |
| **Generate & Load SRT** | Full transcription runs first, then the `.srt` file is loaded as a proper subtitle track. Perfect sync on replay. |

## Models (faster-whisper / openai-whisper)

| Model | Speed | Accuracy | RAM | Download |
|-------|-------|----------|-----|----------|
| `tiny` | Fastest | Basic | ~1 GB | ~75 MB |
| `base` | Fast | Good | ~1 GB | ~140 MB |
| `small` | Moderate | Better | ~2 GB | ~460 MB |
| `medium` | Slow | Great | ~5 GB | ~1.5 GB |
| `large` | Slowest | Best | ~10 GB | ~3 GB |

## Options

- **Language** — `auto` for detection, or a code like `en`, `es`, `fr`, `hi`, `ja`, `zh`, etc.
- **Task** — `Translate to English` (default) or `Transcribe (same language)`

## Manual Installation

If the setup script doesn't work for your system:

1. Install faster-whisper:
   ```bash
   python3 -m venv venv
   venv/bin/pip install faster-whisper        # Linux/macOS
   venv\Scripts\pip.exe install faster-whisper # Windows
   ```

2. Copy `aisubs.lua` to your VLC extensions folder:
   - **Linux**: `~/.local/share/vlc/lua/extensions/`
   - **macOS**: `~/Library/Application Support/org.videolan.vlc/lua/extensions/`
   - **Windows**: `%APPDATA%\vlc\lua\extensions\`

3. Restart VLC.

## Credits

- [voidrlm/vlc-ai-subs](https://github.com/voidrlm/vlc-ai-subs) — original VLC plugin (Lua extension, faster-whisper backend)
- [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) — GGML Whisper with Vulkan acceleration
- [m-bain/whisperX](https://github.com/m-bain/whisperX) — word-level forced alignment
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 Whisper
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — ONNX runtime STT ecosystem
- [usefulinc/moonshine](https://github.com/usefulinc/moonshine) — ultra-fast edge STT

Improvements in this fork: Vulkan GPU acceleration, WhisperX word-aligned subtitles, sherpa-onnx engine, Moonshine fast path, CUDA lib auto-preloading, pluggable backend architecture, zero-config "Recommended" mode, English-as-default translate task, idempotent model installers.

## License

MIT
