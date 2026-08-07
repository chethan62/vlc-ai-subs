#!/usr/bin/env bash
#
# vlc-ai-subs — full installer
#
# One command to set up the VLC subtitle plugin with Vulkan GPU acceleration.
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

GREEN='\033[32m'; CYAN='\033[36m'; NC='\033[0m'
log() { printf "  ${CYAN}→${NC} %s\n" "$*"; }
ok()  { printf "  ${GREEN}✓${NC} %s\n" "$*"; }

echo ""
echo "  vlc-ai-subs installer"
echo "  ---------------------"
echo ""

# ── 0. Check prerequisites ──
log "Checking prerequisites..."
command -v python3 >/dev/null || { echo "python3 required"; exit 1; }
command -v vlc    >/dev/null || { echo "VLC not found — install it first"; exit 1; }
ok "python3 $(python3 --version 2>&1), VLC $(vlc --version 2>/dev/null | head -1 | cut -d' ' -f3)"

# ── 1. Basic plugin (Python deps + Lua extension) ──
if [ -f "$INSTALL_DIR/venv/bin/python" ] && [ -f "$EXT_DIR/aisubs.lua" ]; then
    ok "Plugin already installed — skipping setup.sh"
else
    log "Installing basic plugin (faster-whisper + Lua extension)..."
    "$SCRIPT_DIR/setup.sh"
    ok "Plugin installed"
fi

# ── 2. whisper.cpp Vulkan backend ──
if [ -x "$HOME/.local/share/whisper-cpp/whisper-cli" ] && \
   [ -f "$HOME/.local/share/whisper-cpp/ggml-small.bin" ]; then
    ok "whisper.cpp (Vulkan) already installed"
elif [ -f "$SCRIPT_DIR/install-whisper-cpp.sh" ]; then
    log "Installing whisper.cpp Vulkan backend (this takes a few minutes)..."
    "$SCRIPT_DIR/install-whisper-cpp.sh" small
    ok "whisper.cpp Vulkan ready"
fi

# ── 3. Moonshine backend (optional, fast) ──
if "$INSTALL_DIR/venv/bin/python" -c "import moonshine_voice" 2>/dev/null; then
    ok "Moonshine already installed"
else
    log "Installing Moonshine (optional fast backend)..."
    "$INSTALL_DIR/venv/bin/pip" install moonshine-voice soundfile --quiet 2>&1 | tail -1
    ok "Moonshine installed"
fi

# ── 4. Sync latest Python modules + Lua ──
log "Syncing plugin files..."
mkdir -p "$INSTALL_DIR" "$EXT_DIR"
cp -r "$SCRIPT_DIR/core" "$INSTALL_DIR/"       2>/dev/null || true
cp -r "$SCRIPT_DIR/backends" "$INSTALL_DIR/"   2>/dev/null || true
cp "$SCRIPT_DIR/aisubs_whisper.py" "$INSTALL_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR/aisubs.lua" "$EXT_DIR/"           2>/dev/null || true
ok "Plugin files synced"

echo ""
echo "  ${GREEN}Install complete.${NC}"
echo ""
echo "  Restart VLC, then:  View → AI Subs Generator → Generate"
echo "  The 'Recommended (auto)' model uses Vulkan GPU by default."
echo ""
echo "  Backends installed:"
echo "    - whisper.cpp (Vulkan) — default, fastest GPU"
echo "    - faster-whisper (CUDA) — fallback"
echo "    - Moonshine (CPU) — ultra-fast preview"
echo "    - openai-whisper (CPU) — last resort"
echo ""