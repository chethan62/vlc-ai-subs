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
command -v uv     >/dev/null || { echo "uv required (for Python 3.12 venv). Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
ok "python3 $(python3 --version 2>&1), VLC $(vlc --version 2>/dev/null | head -1 | cut -d' ' -f3), uv $(uv --version 2>&1 | cut -d' ' -f2)"

# ── 1. Basic plugin (Lua extension) ──
if [ -f "$EXT_DIR/aisubs.lua" ]; then
    ok "Plugin already installed — skipping setup.sh"
else
    log "Installing VLC extension..."
    "$SCRIPT_DIR/setup.sh" --install
    ok "Plugin installed"
fi

# ── 2. WhisperX backend (Python 3.12 venv — WhisperX needs <3.14) ──
if [ -x "$INSTALL_DIR/venv-whisperx/bin/python" ] && \
   "$INSTALL_DIR/venv-whisperx/bin/python" -c "import whisperx" 2>/dev/null; then
    ok "WhisperX already installed"
else
    log "Installing WhisperX (Python 3.12 venv, this takes a few minutes)..."
    uv venv --python 3.12 "$INSTALL_DIR/venv-whisperx"
    uv pip install --python "$INSTALL_DIR/venv-whisperx/bin/python" whisperx
    ok "WhisperX ready"
fi

# ── 3. Sync plugin files ──
log "Syncing plugin files..."
mkdir -p "$INSTALL_DIR" "$EXT_DIR"
cp -r "$SCRIPT_DIR/core" "$INSTALL_DIR/"       2>/dev/null || true
cp -r "$SCRIPT_DIR/backends" "$INSTALL_DIR/"   2>/dev/null || true
cp "$SCRIPT_DIR/aisubs_whisper.py" "$INSTALL_DIR/"  2>/dev/null || true
cp "$SCRIPT_DIR/whisperx_runner.py" "$INSTALL_DIR/" 2>/dev/null || true
cp "$SCRIPT_DIR/aisubs.lua" "$EXT_DIR/"             2>/dev/null || true
ok "Plugin files synced"

echo ""
echo "  ${GREEN}Install complete.${NC}"
echo ""
echo "  Restart VLC, then:  View → AI Subs Generator → Generate"
echo "  The 'Recommended (auto)' model is picked from your GPU VRAM."
echo ""
echo "  Engine: WhisperX (word-aligned subtitles) — the only backend."
echo ""