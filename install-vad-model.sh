#!/usr/bin/env bash
# Silero VAD for the Parakeet path (about 630 KB).
#
# What it buys: a Parakeet run decodes in 30 s chunks, and a chunk with no speech in it
# still costs a full decode and can produce a hallucinated line. The VAD is asked where the
# speech is, and chunks with none are skipped — measured on a 60 s music/credits clip, 2 of
# 2 chunks skipped, 3 s instead of 10 s, and no text invented.
#
# What it does NOT do: gate audio that was transcribed. whisper.cpp's own --vad was
# measured dropping real dialogue that way (present in the no-VAD run, missing from every
# VAD configuration tried — see .research/2026-09-16-github-survey.md), and a missing line
# is invisible while a hallucinated one can be caught. Words are never removed.
#
# Optional: without this model the plugin transcribes exactly as it did before.
set -euo pipefail

DEST="${XDG_DATA_HOME:-$HOME/.local/share}/sherpa-onnx/models"
MODEL="$DEST/silero_vad.onnx"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

mkdir -p "$DEST"
if [ -s "$MODEL" ]; then
    echo "already installed: $MODEL"
    exit 0
fi

echo "downloading silero_vad.onnx (~630 KB)…"
if ! curl -fL --retry 3 --connect-timeout 20 -o "$MODEL.tmp" "$URL"; then
    echo "download failed — the plugin runs fine without the VAD, just without chunk skipping." >&2
    rm -f "$MODEL.tmp"
    exit 1
fi
mv "$MODEL.tmp" "$MODEL"
echo "installed: $MODEL"
echo "override the location with VSCL_AISUBS_VAD_MODEL=/path/to/silero_vad.onnx"
