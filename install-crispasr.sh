#!/usr/bin/env bash
# Install the CrispASR binary for the vlc-ai-subs plugin.
#
# Which build to fetch is the whole point, so it is decided here rather than left
# to the user:
#
#   GPU present (NVIDIA)  → the CUDA tarball. Since v0.8.30 its CUDA backend is a
#                           dlopen'd module, so the SAME binary uses the GPU
#                           where there is one and the CPU where there isn't.
#                           One install, both machines.
#   no GPU                → the plain CPU tarball (AVX2: Intel 2013+ / AMD 2015+).
#
# The `-hip` (AMD) and `-vulkan` tarballs are deliberately NOT auto-selected:
# they are statically linked and do NOT fall back, so they would install a
# binary that fails on a machine whose driver is missing. Pass --vulkan or
# --hip only if you know the target has that runtime. For AMD with a fallback,
# build CrispASR yourself with -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON.
#
# Usage: ./install-crispasr.sh [--cpu|--cuda|--vulkan|--hip|--cuda13] [--version vX.Y.Z]
set -euo pipefail

FLAVOR=""
VERSION=""
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu|--cuda|--cuda13|--vulkan|--hip) FLAVOR="${1#--}" ;;
        --version) shift; VERSION="${1:-}" ;;
        -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

REPO="CrispStrobe/CrispASR"
DEST="$HOME/.local/share/crispasr"
LINK="$DEST/crispasr"          # core/crispasr_models.py looks for exactly this path

if [ -z "$FLAVOR" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        FLAVOR="cuda"
        echo "NVIDIA GPU detected — installing the CUDA build (falls back to CPU where absent)."
    else
        FLAVOR="cpu"
        echo "No NVIDIA GPU detected — installing the CPU build."
    fi
fi

case "$FLAVOR" in
    cpu)    ASSET="crispasr-linux-x86_64.tar.gz" ;;
    *)      ASSET="crispasr-linux-x86_64-${FLAVOR}.tar.gz" ;;
esac

if [ -z "$VERSION" ]; then
    VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
        | grep -m1 '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')
    [ -n "$VERSION" ] || { echo "Could not determine the latest release tag." >&2; exit 1; }
fi

DIR="$DEST/${VERSION}-${FLAVOR}"
URL="https://github.com/$REPO/releases/download/$VERSION/$ASSET"

mkdir -p "$DEST"
if [ -x "$DIR/crispasr-linux-x86_64/crispasr" ]; then
    echo "Already installed: $DIR"
else
    echo "Downloading $ASSET ($VERSION)…"
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    curl -fL --progress-bar -o "$TMP/crispasr.tar.gz" "$URL" || {
        echo "Download failed: $URL" >&2
        echo "Check the asset name for $VERSION (release assets differ per flavour)." >&2
        exit 1
    }
    rm -rf "$DIR"; mkdir -p "$DIR"
    tar xzf "$TMP/crispasr.tar.gz" -C "$DIR"
fi

BIN="$DIR/crispasr-linux-x86_64/crispasr"
[ -x "$BIN" ] || { echo "Archive did not contain crispasr-linux-x86_64/crispasr" >&2; exit 1; }

ln -sf "$BIN" "$LINK"
echo
"$BIN" --version | head -3
echo
echo "Installed: $LINK"
echo "The plugin finds it there automatically; override with VSCL_AISUBS_CRISPASR_BIN."
echo
echo "Try it (models download on first use, ~467 MB for the smallest):"
echo "  $LINK --backend parakeet -m auto -f some.wav -l en --vad -osrt -sp"
echo
echo "Throughput measured on this project's test film (8 CPU threads, AVX2):"
echo "  parakeet                  ~4.2x realtime  (~35 min for a 145-min film)"
echo "  parakeet + CTC aligner    ~2.4x realtime  (~61 min; word timings from speech, not emission frames)"
