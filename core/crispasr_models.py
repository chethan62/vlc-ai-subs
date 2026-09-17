"""CrispASR: which model runs on which machine, and with which flags.

CrispASR is one C++ ggml binary (MIT, a whisper.cpp fork) that runs many ASR
architectures — Parakeet, Cohere Transcribe, Canary, Granite, Qwen3-ASR,
Voxtral — plus a **CTC forced aligner**, VAD and pyannote-based diarization,
with no Python and no PyTorch. See
`.research/2026-09-16-asr-landscape-and-crispasr.md` for the survey and the
measurements.

Why this module exists
----------------------
The point of this engine is to **scale up on a better machine**. This project's
own laptop has a hard-capped dGPU (measured ~300 MHz / ~15.5 W) and runs the CPU
build; a machine with real VRAM should get a bigger model and the aligner
automatically, from the same code. So the model choice is a VRAM tier, and it is
built here — one table — so that `crispasr_runner.py` (which builds the command)
and `backends/crispasr.py` (which prints the status line) can never disagree
about which model is running, exactly as `core/parakeet_models.py` does for
Parakeet.

Alignment policy
----------------
The CTC aligner produces word timings from speech rather than from the decoder's
emission frames, which is the only measured answer to this project's remaining
timing defect (cue spans that end at the last word's *start*). It is NOT free:
measured on this laptop (two runs each, warm cache), 90 s of audio took 21.3 s
with the aligner off (4.2x realtime) and 36.8 s with it on (2.4x realtime) —
about 35 vs 61 minutes for a feature film. On a GPU that cost almost disappears.
So: align by default when a GPU is doing the work, opt-in on CPU
(`VSCL_AISUBS_CRISPASR_ALIGN=1`).

Leaf module: stdlib only, imported by both the runner and the backend.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from core.parakeet_models import V3_LANGUAGES

BINARY_ENV = "VSCL_AISUBS_CRISPASR_BIN"
INSTALL_HINT = "./install-crispasr.sh"

# Where install-crispasr.sh puts it, and the fallback if PATH has it.
DEFAULT_BINARY = os.path.expanduser("~/.local/share/crispasr/crispasr")


@dataclass(frozen=True)
class Model:
    """One CrispASR backend+model pair, with the VRAM it needs to be worth it."""

    tag: str
    backend: str
    label: str
    min_vram_mb: int
    languages: frozenset
    # True when this model IS the backend's default, so the binary can fetch it
    # with `-m auto`. A non-default variant has to be named explicitly — and
    # naming it is preferable to passing `auto`, which would quietly run a
    # different model than the status line advertises.
    default_for_backend: bool = True

    def handles(self, language: str | None) -> bool:
        """Whether this model covers *language*.

        An EMPTY language set means the model's coverage is wide and not worth
        enumerating (Qwen3-ASR: 30 languages plus 22 Chinese dialects) — treating
        it as "handles nothing" made those models unreachable.
        """
        if language is None or not self.languages:
            return True
        return language in self.languages


# Tiers, cheapest first. `min_vram_mb = 0` means "the CPU path" — the model that
# has to work everywhere, which is the same Parakeet v3 weights this plugin
# already runs through sherpa-onnx (v1.4.4 measured: 4.3x realtime on 8 CPU
# threads, and it transcribes a passage the ONNX-int8 path silently drops).
#
# The larger models are the payload for a machine with a real GPU. WER figures
# are vendor/leaderboard claims, not measurements of ours; licences differ
# (CC-BY-4.0 vs Apache-2.0), which matters if weights are ever redistributed.
MODELS: tuple[Model, ...] = (
    Model("parakeet-0.6b", "parakeet", "parakeet-tdt-0.6b-v3", 0, V3_LANGUAGES),
    # ~4 GB: a small wide-coverage LLM-ASR, then the bigger English Parakeet —
    # which is the preferred pick at this tier, hence its later position.
    Model("qwen3-0.6b", "qwen3", "Qwen3-ASR-0.6B", 4000, frozenset()),
    Model("parakeet-1.1b", "parakeet", "parakeet-tdt-1.1b", 4000, frozenset({"en"}),
          default_for_backend=False),
    # ~6 GB: 2B-class models that top the Open ASR Leaderboard (5.33-5.42 avg WER
    # claimed). Cohere transcribes 14 languages and is preferred here; Qwen3-1.7B
    # covers languages no other tier does.
    Model("qwen3-1.7b", "qwen3-1.7b", "Qwen3-ASR-1.7B", 6000, frozenset(),
          default_for_backend=False),
    Model("granite-2b", "granite-4.1", "granite-speech-4.1-2b", 6000,
          frozenset({"en", "fr", "de", "es", "pt", "ja"})),
    Model("cohere", "cohere", "cohere-transcribe-03-2026", 6000,
          frozenset({"en", "fr", "de", "it", "es", "pt", "el", "nl", "pl",
                     "zh", "ja", "ko", "vi", "ar"})),
    # ~8 GB: a 3B Mistral-ASR (8 languages) and the accuracy ceiling of the open
    # field (English-only) — the latter preferred for English.
    Model("voxtral-3b", "voxtral", "Voxtral-Mini-3B-2507", 8000,
          frozenset({"en", "fr", "de", "es", "it", "pt", "nl", "hi"})),
    Model("canary-qwen", "canary-qwen", "canary-qwen-2.5b", 8000, frozenset({"en"})),
)

BY_TAG = {m.tag: m for m in MODELS}
# Languages no tier above covers (Qwen3-ASR: 30 languages + Chinese dialects,
# gemma4/omniasr are wide) — a language outside every set must still resolve to
# *something*, and the smallest model is the only safe answer on a small box.
FALLBACK = MODELS[0]


def binary() -> str | None:
    """Path to the crispasr binary, or None when it is not installed.

    VSCL_AISUBS_CRISPASR_BIN wins (a build you made yourself, or the CUDA tarball
    in a non-standard place); then the install script's location; then PATH.
    """
    explicit = os.environ.get(BINARY_ENV, "").strip()
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    if os.path.isfile(DEFAULT_BINARY):
        return DEFAULT_BINARY
    return shutil.which("crispasr")


def installed() -> bool:
    return binary() is not None


def is_available() -> bool:
    """True when the engine can run at all — used by backends.crispasr.detect()."""
    return installed()


def pick(vram_mb: int, language: str | None = None,
         forced: str | None = None) -> Model:
    """The best model for this machine, preferring the largest the VRAM allows.

    A language the tier cannot handle pushes down to the next tier that can, so
    a machine with 8 GB asking for Japanese still gets a model instead of one
    that would transcribe it badly or refuse it. `forced` is the user override
    (a tag from BY_TAG, or a bare .gguf path handled by the caller).

    Among models that need the same VRAM the later entry in MODELS wins, which is
    why the table is ordered cheapest first and tuned within each tier.
    """
    if forced:
        chosen = BY_TAG.get(forced.strip().lower())
        if chosen is not None:
            return chosen
    usable = [m for m in MODELS if m.min_vram_mb <= max(0, vram_mb) and m.handles(language)]
    if not usable:
        # Nothing at this VRAM handles the language: fall back to the smallest
        # model (which for an unknown language is the multilingual one).
        return next((m for m in MODELS if m.handles(language)), FALLBACK)
    return max(usable, key=lambda m: (m.min_vram_mb, MODELS.index(m)))


def model_tag(language: str | None = None, vram_mb: int | None = None) -> str:
    """Resolve the model to run, honouring VSCL_AISUBS_CRISPASR_MODEL.

    Three forms:

    * a tier tag (`cohere`, `parakeet-1.1b`) — that model, on any machine
    * a `.gguf` path — passed straight through, so an experimental quant needs
      no code change
    * `auto` — tier by the detected VRAM, which is what a machine with a real
      GPU wants

    Unset means the smallest model, and that default is deliberate: this
    project's laptop has a GTX 1650 with 4 GB that advertises a 1785 MHz maximum
    while being throttled to 300 MHz, so a VRAM-only rule would hand it a 2B
    model and turn a 20-minute film into an afternoon. A machine that really can
    afford the bigger model asks for it — by name or with `auto`.
    """
    env = os.environ.get("VSCL_AISUBS_CRISPASR_MODEL", "").strip()
    if env and ("/" in env or env.endswith(".gguf")):
        return env
    if env.lower() == "auto":
        if vram_mb is None:
            from core.gpu import vram_mb as _detected

            vram_mb = _detected()
        return pick(vram_mb, language).tag
    return pick(0, language, forced=env or None).tag


def available_ram_mb() -> int:
    """MemAvailable from /proc/meminfo, in MiB, or 0 when it cannot be read.

    MemAvailable (not MemFree) because it is the kernel's own estimate of what a
    new workload can take without swapping. On this machine swap is zram — a
    compressed RAM device (~15 GB of it, 10 GB in use at the time of writing) —
    so "swapping" costs real memory and memory pressure here has already killed a
    browser, not just a job. Falling back to 0 means "assume nothing", which
    makes the callers choose the most conservative setting.
    """
    try:
        with open("/proc/meminfo", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def chunk_seconds(available_mb: int | None = None) -> int:
    """Seconds of audio the binary processes at a time.

    Measured on this laptop (15 GB, 8 threads): a 1200 s file with the CTC aligner
    peaked at **8.3 GB RSS plus 4.8 GB swap — which here is zram, i.e. real
    RAM — and was OOM-killed** after 7 minutes, while 90 s inputs run in ~300 MB.
    The binary's own banner offers the remedy: "use --chunk-seconds N if OOM".
    The plugin therefore always chunks, and sizes the chunk from what the machine
    can actually spare.

    The constant is measured: ~4 MB of peak per second of audio (300 s -> 1.2 GB
    observed), and the budget is a quarter of MemAvailable — enough headroom that
    a transcription cannot push the desktop into the OOM killer, which on this
    box it has already done to Chromium. Clamped to 30-600 s.

    VSCL_AISUBS_CRISPASR_CHUNK overrides all of it (seconds); 0 means "no
    chunking", which is only safe for short files on a roomy machine.
    """
    raw = os.environ.get("VSCL_AISUBS_CRISPASR_CHUNK", "").strip()
    if raw:
        try:
            value = int(float(raw))
        except ValueError:
            value = -1
        if value == 0:
            return 0
        if value > 0:
            return max(30, min(3600, value))
    if available_mb is None:
        available_mb = available_ram_mb()
    if available_mb <= 0:
        return 60                      # unknown machine: the conservative choice
    return max(30, min(600, int(available_mb * 0.25 / 4)))


def model_argument(tag: str) -> str:
    """The value to pass as the binary's `-m`.

    `auto` for a backend's own default model — the binary's registry then picks
    the quant, which is what keeps the download small and the choice current.
    Anything else is named explicitly: passing `auto` for a non-default variant
    would quietly run a different model than the status line advertises.
    """
    known = BY_TAG.get(tag.strip().lower())
    if known is not None:
        return "auto" if known.default_for_backend else tag
    return tag


def model_for(tag: str, vram_mb: int = 0, language: str | None = None) -> Model:
    """The Model a tag refers to, or the VRAM-appropriate pick when unknown."""
    known = BY_TAG.get(tag.strip().lower())
    return known if known is not None else pick(vram_mb, language)


def model_label(language: str | None = None, vram_mb: int | None = None) -> str:
    """Label for the CLI's "Backend: crispasr — <label>" status line."""
    tag = model_tag(language, vram_mb)
    known = BY_TAG.get(tag.lower())
    return known.label if known is not None else os.path.basename(tag)


def align_enabled(gpu: bool | None = None) -> bool:
    """Whether to run the CTC aligner.

    VSCL_AISUBS_CRISPASR_ALIGN=1/0 forces it. Otherwise: on when a GPU is doing
    the work (the cost is negligible there) and off on CPU, where the same pass
    measured ~1.0x realtime against 4.3x without it — 2.4 h for a feature film.
    `gpu=None` asks the hardware probe.
    """
    env = os.environ.get("VSCL_AISUBS_CRISPASR_ALIGN", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    if gpu is None:
        from core.gpu import nvidia_gpu

        gpu = nvidia_gpu() is not None
    return bool(gpu)


def supported_languages() -> set:
    """Every language any tier covers (for 'unsupported language' messages)."""
    langs: set = set()
    for model in MODELS:
        langs |= set(model.languages)
    return langs
