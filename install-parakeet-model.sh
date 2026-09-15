#!/usr/bin/env bash
#
# install-parakeet-model.sh [v2|v3] — download an NVIDIA Parakeet-TDT-0.6B
# int8 ONNX model for the vlc-ai-subs Parakeet backend
# (VSCL_AISUBS_BACKEND=parakeet). CC-BY-4.0. Idempotent.
#
#   v2 (default)  English only, ~0.7GB. English-specialised: a run for "en"
#                 uses this one whenever it is installed.
#   v3            25 European languages, ~0.64GB. Needed for every other
#                 language; also handles English.
#
# Both are the same 0.6B TDT architecture with native word timestamps. Pass
# v3 (or VSCL_AISUBS_PARAKEET_VERSION=v3) to install the multilingual one;
# installing both is fine — the runner picks per language.
set -euo pipefail

VARIANT="${1:-${VSCL_AISUBS_PARAKEET_VERSION:-v2}}"
case "$VARIANT" in
    2|v2) NAME="sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"; DESC="English";;
    3|v3) NAME="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"; DESC="25 European languages";;
    *)
        echo "usage: $0 [v2|v3]   (v2 = English only, v3 = 25 European languages)" >&2
        exit 2
        ;;
esac

DEST="$HOME/.local/share/sherpa-onnx/models"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NAME}.tar.bz2"

mkdir -p "$DEST"
cd "$DEST"

if [ -f "$NAME/encoder.int8.onnx" ] && [ -f "$NAME/tokens.txt" ]; then
    echo "  ✓ Parakeet $VARIANT ($DESC) already installed ($DEST/$NAME)"
    exit 0
fi

echo "  → Downloading $NAME ($DESC) ..."
curl -L --fail --progress-bar -o "$NAME.tar.bz2" "$URL"

echo "  → Extracting..."
tar xf "$NAME.tar.bz2"
rm -f "$NAME.tar.bz2"

ls -lh "$NAME"/*.onnx "$NAME"/tokens.txt
echo "  ✓ Parakeet $VARIANT ready: $DEST/$NAME"
