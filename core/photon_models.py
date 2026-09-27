"""Photon / Parakeet Redux: which model, which device, and where the venv lives.

Moondream's **Parakeet Redux** is NVIDIA's `parakeet-tdt-0.6b-v3` re-quantised to
ternary weights (1.58-bit, 178 MB instead of 1.2 GB) and run by Moondream's
`photon` engine. It is the fastest ASR measured on this project's own laptop —
and the measurement is the reason it is here at all:

    125 s clip, 8 CPU threads, cool box (load < 2, package 55-58 C)
      photon, parakeet-redux, CPU (avx2 int8)   8.96 s   13.98x realtime
      sherpa-onnx int8 parakeet (this plugin)  16.79 s    7.46x
      whisper.cpp small.en, CPU                26.97 s    4.64x

Two things that are NOT marketing claims and belong in the code, because they
decide when this engine may be used:

* **CUDA is the wrong device for it.** The packed ternary GEMM exists only for
  x86 int8 (avx2 / avxvnni / avx512vnni) and Apple Metal; on CUDA the codes are
  dequantized to a dense form and four conformer kernels fall back to plain
  PyTorch — measured 2-2.4x *slower* than this CPU. So "auto" never picks CUDA
  while a native-kernel device exists.
* **It is the weaker model in noise.** Measured WER on the same clip: 0.0 % at
  10 dB SNR, 7.1 % at 5 dB, 10.7 % at 0 dB, all of it similar-sounding
  substitutions (turnips -> turnets, fattened -> satin). Film audio is the hard
  case, so this is a clean-speech / draft engine, not a drop-in replacement for
  the sherpa path.

Licence: the *weights* are CC-BY-4.0 (Moondream, on NVIDIA's Parakeet v3), but
the engine that runs them (Moondream Photon + `kestrel-kernels`) is proprietary
and its licence forbids reverse engineering. Nothing third-party is vendored
here — `install-photon-model.sh` is opt-in and downloads it at install time.

Leaf module: stdlib only, imported by both the runner and the backend.
"""
from __future__ import annotations

import glob
import os

VENV_ENV = "VSCL_AISUBS_PHOTON_VENV"
MODEL_ENV = "VSCL_AISUBS_PHOTON_MODEL"
DEVICE_ENV = "VSCL_AISUBS_DEVICE"          # shared with the other engines
INSTALL_HINT = "./install-photon-model.sh"
INSTALL_DIR = os.path.expanduser("~/.local/share/vlc-ai-subs")
DEFAULT_VENV = os.path.join(INSTALL_DIR, "venv-photon")

# Parakeet v3 languages the *base* model covers. Redux is a re-quantisation of
# it, so these probably carry over — unverified here, and an unverified
# multilingual claim is worse than none: only English is measured.
VERIFIED_LANGUAGES = frozenset({"en"})

MODELS = {
    "redux": ("moondream/parakeet-redux", "parakeet-redux (ternary, 178 MB)"),
    "ultra": ("moondream/parakeet-ultra", "parakeet-ultra (ternary, 385 MB)"),
}
DEFAULT_MODEL = "redux"

# Devices with a NATIVE packed-ternary kernel, best first. CUDA is last on
# purpose: it has none, so it is only ever reached when nothing else exists.
NATIVE_ORDER = ("mps", "cpu", "cuda")


def venv_python() -> str | None:
    """The interpreter holding `moondream`, or None when the venv is absent.

    VSCL_AISUBS_PHOTON_VENV names a venv DIRECTORY (the install script's
    default is `~/.local/share/vlc-ai-subs/venv-photon`). Photon needs PyTorch,
    so it cannot share the stdlib-only CLI venv — and it deliberately does not
    share `venv-whisperx` either, where a pip resolve could move the torch that
    WhisperX depends on.
    """
    venv = os.environ.get(VENV_ENV, "").strip() or DEFAULT_VENV
    python = os.path.join(venv, "bin", "python")
    return python if os.path.isfile(python) else None


def installed() -> bool:
    """True when the venv exists AND `moondream` is importable-looking.

    A file check, not an import: this runs in the CLI process, which has no
    torch and no moondream, and spawning the runner's interpreter to ask would
    add seconds to every dialog open. The runner itself imports for real and
    fails loudly with INSTALL_HINT if the package is broken.
    """
    python = venv_python()
    if not python:
        return False
    venv = os.path.dirname(os.path.dirname(python))
    pattern = os.path.join(venv, "lib", "python*", "site-packages", "moondream")
    return bool(glob.glob(pattern))


def model_id(name: str | None = None) -> str:
    """The HF repo id to run: `redux` (default), `ultra`, or a literal id."""
    key = (name if name is not None else os.environ.get(MODEL_ENV, "")).strip()
    if not key:
        key = DEFAULT_MODEL
    known = MODELS.get(key.lower())
    return known[0] if known else key


def model_label(name: str | None = None) -> str:
    """Label for the CLI's "Backend: photon — <label>" status line."""
    key = (name if name is not None else os.environ.get(MODEL_ENV, "")).strip()
    known = MODELS.get(key.lower() or DEFAULT_MODEL)
    return known[1] if known else os.path.basename(model_id(key))


def resolve_device(pref: str | None, available, native=None) -> str:
    """Pick the device, preferring one with a native ternary kernel.

    `available` is every device the runtime reports; `native` maps a device to
    whether it has a packed-ternary kernel (both come from the runner, which is
    the side that can import torch and the engine's kernel module — this stays
    a pure function so the rule is testable without either).

    auto → the first of mps/cpu/cuda that is available AND native, else the last
    available one (CUDA on a GPU-only machine, CPU everywhere else). An explicit
    device that is not available raises: silently running somewhere else is how
    a status line ends up naming a device the run never used.
    """
    native = native or {}
    avail = list(available)
    pref = (pref or "auto").strip().lower()
    if pref and pref != "auto":
        if pref.split(":")[0] not in avail:
            raise ValueError(
                f"device '{pref}' is not available here (have: {', '.join(avail) or 'none'})"
            )
        return pref
    for device in NATIVE_ORDER:
        if device in avail and native.get(device):
            return device
    return avail[-1] if avail else "cpu"


def language_warning(language: str | None) -> str | None:
    """A status line for a language nobody has measured this engine on, or None.

    Parakeet Redux is a re-quantisation of the 25-language v3 model, so other
    languages may well work — but no run here has checked, and this project has
    already shipped one timing that was 2.3x wrong by trusting a plausible
    inference. Say "unverified" instead.
    """
    if not language or language == "auto":
        return None
    primary = language.lower().split("-")[0]
    if primary in VERIFIED_LANGUAGES:
        return None
    return (f"Photon Redux is only verified in English here — '{language}' is untested "
            f"(the base model covers 25 languages, this quantisation was not measured on them)")


def device_note(device: str) -> str | None:
    """The caveat that belongs beside a CUDA run, or None."""
    if device.split(":")[0] == "cuda":
        return ("CUDA runs these weights dense (no ternary kernel) and falls back to "
                "PyTorch for four conformer kernels — a CPU with a native int8 path is "
                "usually faster. Measured 2-2.4x slower than the same box's CPU.")
    return None
