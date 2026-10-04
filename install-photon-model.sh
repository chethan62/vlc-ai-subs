#!/usr/bin/env bash
# Install Moondream Photon + Parakeet Redux for the vlc-ai-subs plugin.
#
# Opt-in, not part of install.sh: it downloads a PROPRIETARY engine.
#   * weights: CC-BY-4.0 (Moondream, on NVIDIA parakeet-tdt-0.6b-v3)
#   * runtime: Moondream Photon + kestrel-kernels — proprietary (M87 Labs). Its
#     licence grants nothing without a separate written agreement with them ("if
#     you have not entered into such an Agreement, you have no license to use this
#     software") and forbids reverse engineering, unpacking or redistribution. So:
#     running this install is your call, and what you install here is not yours to
#     pass on. The other engines in this plugin are MIT/Apache-2.0 only.
#
# Its own venv on purpose. Photon needs PyTorch, so it cannot use the plugin's
# stdlib-only CLI venv — and it must not borrow venv-whisperx, where a pip
# resolve could move the torch build WhisperX depends on.
#
# Measured on this project's laptop (i5-10300H, AVX2, 8 threads, cool box),
# 125 s clip: 13.98x realtime, against 7.46x for the sherpa-onnx int8 Parakeet
# path the plugin already had and 4.64x for whisper.cpp small.en.
#
# Usage: ./install-photon-model.sh [--venv DIR]
set -euo pipefail

DEST="${HOME}/.local/share/vlc-ai-subs"
VENV="${DEST}/venv-photon"

while [ $# -gt 0 ]; do
    case "$1" in
        --venv) shift; VENV="${1:-}" ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

PY="${VENV}/bin/python"

if [ -f "$PY" ] && "$PY" -c "import moondream" >/dev/null 2>&1; then
    echo "Already installed: $VENV"
    "$PY" -c "import moondream, torch; print('moondream ok, torch', torch.__version__)"
    exit 0
fi

mkdir -p "$DEST"

# Python 3.12: the version venv-whisperx already uses, and the one with wheels
# for everything involved. Photon itself is version-tolerant; keeping the two ML
# venvs on the same interpreter avoids a second full toolchain.
if command -v uv >/dev/null 2>&1; then
    echo "Creating $VENV (uv, python 3.12)…"
    uv venv --python 3.12 "$VENV"
    # --torch-backend cpu is load-bearing: the default wheel set drags in ~3.2 GB
    # of nvidia-* packages for a GPU this engine should not be using (CUDA has no
    # ternary kernel and measured 2-2.4x slower here).
    echo "Installing moondream (CPU torch)…"
    uv pip install --python "$PY" --torch-backend cpu moondream
else
    echo "uv not found — using python3 -m venv + pip (CPU wheels)"
    python3 -m venv "$VENV"
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
    "$PY" -m pip install moondream
fi

if ! "$PY" -c "import moondream" >/dev/null 2>&1; then
    echo "Install failed: 'import moondream' does not work in $VENV" >&2
    echo "Retry with: $PY -m pip install moondream" >&2
    exit 1
fi

echo
"$PY" -c "import moondream, torch; print('moondream ok · torch', torch.__version__, '· cuda', torch.cuda.is_available())"
SIZE=$(du -sh "$VENV" 2>/dev/null | cut -f1)
echo "Installed: $VENV (${SIZE:-unknown})"
echo
echo "The plugin finds it there automatically; override with VSCL_AISUBS_PHOTON_VENV."
echo "Select it with the Engine dropdown, or:"
echo "  VSCL_AISUBS_BACKEND=photon <the plugin's CLI> <media> recommended en transcribe"
echo
echo "Model variants (VSCL_AISUBS_PHOTON_MODEL):"
echo "  redux  178 MB  default, measured 13.98x realtime here"
echo "  ultra  385 MB  larger; not measured on this box"
echo
echo "Device: 'auto' picks the first device with a NATIVE ternary kernel"
echo "(Metal on Apple silicon, the int8 paths on x86) — CUDA has none and"
echo "measured 2-2.4x slower than this laptop's CPU, so it is never auto-picked."
echo
echo "Caveat worth keeping: in noise this model degrades to similar-sounding"
echo "substitutions (measured 7.1% WER at 5 dB SNR, 10.7% at 0 dB, 0% clean)."
echo "It is a fast clean-speech/draft engine, not the film-accuracy engine."
echo
echo "License: weights CC-BY-4.0 (Moondream); the Photon runtime (kestrel-kernels)"
echo "+ moondream) is proprietary M87 Labs software, licensed to you only under their"
echo "own agreement — and its licence forbids reverse engineering and redistribution."
echo "Nothing is vendored into the plugin, and do not ship this venv inside a build."
