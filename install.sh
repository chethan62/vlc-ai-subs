#!/usr/bin/env bash
#
# vlc-ai-subs — full installer
#
# One command to set up the VLC subtitle plugin with WhisperX
# (word-level aligned timestamps — ideal for movie subtitles).
# Detects what's already installed and skips completed steps (idempotent).
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/chethan62/vlc-ai-subs/main/install.sh | bash
#   or
#   git clone https://github.com/chethan62/vlc-ai-subs && cd vlc-ai-subs && ./install.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/.local/share/vlc-ai-subs"
EXT_DIR="$HOME/.local/share/vlc/lua/extensions"

GREEN=$'\033[32m'; CYAN=$'\033[36m'; YELLOW=$'\033[33m'; NC=$'\033[0m'
log() { printf "  ${CYAN}→${NC} %s\n" "$*"; }
ok()  { printf "  ${GREEN}✓${NC} %s\n" "$*"; }
warn() { printf "  ${YELLOW}!${NC} %s\n" "$*"; }

echo ""
echo "  vlc-ai-subs installer"
echo "  ---------------------"
echo ""

# ── 0. Check prerequisites ──
log "Checking prerequisites..."
command -v python3 >/dev/null || { echo "python3 required"; exit 1; }
command -v vlc    >/dev/null || { echo "VLC not found — install it first"; exit 1; }
command -v uv     >/dev/null || { echo "uv required (for Python 3.12 venv). Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg required (audio decode for the Parakeet engine). Install via your package manager, e.g.: sudo apt install ffmpeg"; exit 1; }
ok "python3 $(python3 --version 2>&1), VLC $(vlc --version 2>/dev/null | head -1 | cut -d' ' -f3), uv $(uv --version 2>&1 | cut -d' ' -f2), ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f3)"

# ── 1. Basic plugin (Lua extension) ──
if [ -f "$EXT_DIR/aisubs.lua" ]; then
    ok "Plugin already installed — skipping setup.sh"
else
    log "Installing VLC extension..."
    if "$SCRIPT_DIR/setup.sh" --install; then
        ok "Plugin installed"
    else
        warn "VLC extension NOT installed — copy aisubs.lua into VLC's lua/extensions"
        warn "folder yourself (README → Manual install), then restart VLC"
    fi
fi

# ── 2. WhisperX backend (Python 3.12 venv — WhisperX needs <3.14) ──
if [ -x "$INSTALL_DIR/venv-whisperx/bin/python" ] && \
   "$INSTALL_DIR/venv-whisperx/bin/python" -c "import whisperx" 2>/dev/null; then
    ok "WhisperX already installed"
else
    log "Installing WhisperX (Python 3.12 venv, this takes a few minutes)..."
    if uv venv --python 3.12 "$INSTALL_DIR/venv-whisperx" && \
       uv pip install --python "$INSTALL_DIR/venv-whisperx/bin/python" whisperx; then
        ok "WhisperX ready"
    else
        warn "WhisperX install failed — the multi-language engine will be missing"
        warn "Retry with: uv venv --python 3.12 $INSTALL_DIR/venv-whisperx"
    fi
fi

# ── 3. Parakeet backend (recommended for English media) ──
# NVIDIA Parakeet-TDT-0.6B via sherpa-onnx: ~10x faster, native word
# timestamps, CC-BY-4.0. Runs alongside WhisperX (multilingual default).
# v2 = English, v3 = 25 European languages; set VSCL_AISUBS_PARAKEET_V3=1 to
# install both (English keeps using the v2 model, other languages use v3).
if [ -f "$HOME/.local/share/sherpa-onnx/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx" ]; then
    ok "Parakeet model already installed"
else
    log "Installing Parakeet model (~600MB download)..."
    if bash "$SCRIPT_DIR/install-parakeet-model.sh"; then
        ok "Parakeet model ready"
    else
        warn "Parakeet model NOT installed — retry: bash install-parakeet-model.sh"
    fi
fi
if [ "${VSCL_AISUBS_PARAKEET_V3:-0}" = "1" ]; then
    log "Installing multilingual Parakeet v3 (25 European languages, ~640MB)..."
    if bash "$SCRIPT_DIR/install-parakeet-model.sh" v3; then
        ok "Parakeet v3 ready"
    else
        warn "Parakeet v3 NOT installed — retry: bash install-parakeet-model.sh v3"
    fi
fi
if [ -x "$INSTALL_DIR/venv-whisperx/bin/python" ] && \
   "$INSTALL_DIR/venv-whisperx/bin/python" -c "import sherpa_onnx" 2>/dev/null; then
    ok "sherpa-onnx already installed"
else
    log "Installing sherpa-onnx runtime..."
    if uv pip install --python "$INSTALL_DIR/venv-whisperx/bin/python" sherpa-onnx; then
        ok "sherpa-onnx ready"
    else
        warn "sherpa-onnx install failed — the Parakeet engine will be missing"
    fi
fi

# NLLB-200 model (translate cascade) — ~1.3 GB, CC-BY-NC-4.0. Optional: a
# failed download must not kill the install (translate falls back to Whisper).
if [ "${VSCL_AISUBS_SKIP_NLLB:-0}" = "1" ]; then
    log "Skipping NLLB model (VSCL_AISUBS_SKIP_NLLB=1) — translate will use Whisper"
elif [ -f "$INSTALL_DIR/nllb-200-distilled-1.3B-int8/model.bin" ]; then
    ok "NLLB model already installed"
else
    log "Installing NLLB model (~1.3GB, CC-BY-NC-4.0)..."
    if bash "$SCRIPT_DIR/install-nllb-model.sh"; then
        ok "NLLB model ready"
    else
        log "NLLB download failed — translate falls back to Whisper (retry: bash install-nllb-model.sh)"
    fi
fi

# ── 3b. whisper.cpp (Vulkan) — the AMD/Intel GPU path ──
# Only WhisperX accelerates on NVIDIA CUDA (faster-whisper has no ROCm backend),
# so GPUs from AMD/Intel get acceleration through whisper.cpp's vendor-neutral
# Vulkan backend instead. Installed automatically when no NVIDIA GPU is present;
# opt in on an NVIDIA box with VSCL_AISUBS_WHISPERCPP=1, opt out anywhere with
# VSCL_AISUBS_SKIP_WHISPERCPP=1 (the build takes a few minutes).
if [ "${VSCL_AISUBS_SKIP_WHISPERCPP:-0}" = "1" ]; then
    log "Skipping whisper.cpp (VSCL_AISUBS_SKIP_WHISPERCPP=1)"
elif [ -x "$HOME/.local/share/whisper-cpp/whisper-cli" ]; then
    ok "whisper.cpp already installed"
elif [ "${VSCL_AISUBS_WHISPERCPP:-0}" = "1" ] || ! command -v nvidia-smi >/dev/null 2>&1; then
    log "Installing whisper.cpp with Vulkan (AMD/Intel GPU support)..."
    if bash "$SCRIPT_DIR/install-whisper-cpp.sh" small; then
        ok "whisper.cpp ready (Vulkan)"
    else
        log "whisper.cpp build failed — GPU-less machines still work on CPU (retry: bash install-whisper-cpp.sh)"
    fi
else
    log "NVIDIA GPU detected — skipping whisper.cpp (VSCL_AISUBS_WHISPERCPP=1 adds the Vulkan engine)"
fi

# ── 4. Sync plugin files ──
log "Syncing plugin files..."
mkdir -p "$INSTALL_DIR" "$EXT_DIR"
cp -r "$SCRIPT_DIR/core" "$INSTALL_DIR/"       2>/dev/null || true
cp -r "$SCRIPT_DIR/backends" "$INSTALL_DIR/"   2>/dev/null || true
# Never ship bytecode: a __pycache__ copied next to updated sources can shadow
# them (a stale core/__pycache__/blocklist.pyc made a fixed module look broken
# during development). Sources only.
find "$INSTALL_DIR" -maxdepth 3 -type d -name '__pycache__' \
    -not -path "$INSTALL_DIR/venv-whisperx/*" -prune -exec rm -rf {} + 2>/dev/null || true
cp "$SCRIPT_DIR/aisubs_whisper.py" "$INSTALL_DIR/"  2>/dev/null || true
cp "$SCRIPT_DIR/whisperx_runner.py" "$INSTALL_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR/nllb_translate.py" "$INSTALL_DIR/"   2>/dev/null || true
cp "$SCRIPT_DIR/parakeet_runner.py" "$INSTALL_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR/whispercpp_runner.py" "$INSTALL_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR/crispasr_runner.py" "$INSTALL_DIR/"  2>/dev/null || true
cp "$SCRIPT_DIR/install-parakeet-model.sh" "$INSTALL_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR/install-whisper-cpp.sh" "$INSTALL_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR/install-crispasr.sh"     "$INSTALL_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR/install-nllb-model.sh"      "$INSTALL_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR/install-m2m-model.sh"       "$INSTALL_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR/aisubs.lua" "$EXT_DIR/"             2>/dev/null || true
ok "Plugin files synced"

echo ""
echo "  ${GREEN}Install complete.${NC}"
echo ""
echo "  Restart VLC, then:  View → AI subtitle generator (WhisperX/Parakeet) → Generate"
echo "  'Recommended (auto)' picks large-v3-turbo on 4GB+ GPUs."
echo ""
echo "  Engines:"
echo "    - WhisperX (multilingual, word-aligned) — default"
echo "    - Parakeet (English, ~10x faster, native word timing)"
echo ""