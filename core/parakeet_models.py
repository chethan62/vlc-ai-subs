"""Parakeet-TDT-0.6B variants: which models exist, which are installed, which
one runs a given language.

A leaf module (stdlib only, no imports from core/, backends/ or the runners) so
that both `parakeet_runner.py` (selects and loads the model) and
`backends/parakeet.py` (reports the model in status lines) share ONE table —
the CLI's "Backend: parakeet — <label>" line and the runner's status line can
never disagree about the model names.

The Lua dialog mirrors the language rule in aisubs.lua's engine_for() so it can
show the engine before the run starts; this module stays authoritative.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

MODELS_ROOT = os.path.expanduser("~/.local/share/sherpa-onnx/models")
MODEL_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")

INSTALL_HINT = "./install-parakeet-model.sh [v2|v3]"

# 25 European languages of parakeet-tdt-0.6b-v3 (k2-fsa model card).
V3_LANGUAGES = frozenset({
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
    "sl", "es", "sv", "ru", "uk",
})

# Every language any variant can handle (used to tell "unsupported language"
# apart from "supported but not installed" in error messages).
KNOWN_LANGUAGES = V3_LANGUAGES | {"en"}


@dataclass(frozen=True)
class Variant:
    tag: str
    label: str
    dirname: str
    languages: frozenset

    @property
    def is_multilingual(self) -> bool:
        return len(self.languages) > 1


VARIANTS = (
    Variant("v2", "parakeet-tdt-0.6b-v2", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
            frozenset({"en"})),
    Variant("v3", "parakeet-tdt-0.6b-v3", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            V3_LANGUAGES),
)


def normalize_language(language: str | None) -> str | None:
    """'en-GB'/'EN'/'en_GB' → 'en'; None/'auto'/'' → None.

    WhisperX accepts region codes, so the dialog can pass them through; the
    language sets here are primary subtags, and comparing the raw string
    rejected perfectly good codes like 'en-GB'.
    """
    if not language or language.strip().lower() == "auto":
        return None
    return language.strip().lower().replace("_", "-").split("-")[0]


def variant_dir(variant: Variant) -> str:
    """Absolute model directory (a custom variant carries its own path)."""
    if os.path.isabs(variant.dirname):
        return variant.dirname
    return os.path.join(MODELS_ROOT, variant.dirname)


def model_ok(path: str) -> bool:
    return all(os.path.isfile(os.path.join(path, name)) for name in MODEL_FILES)


def installed_variants() -> "list[Variant]":
    """Variants whose model files are present.

    VSCL_AISUBS_PARAKEET_MODEL=<dir> replaces the search entirely: an unknown
    directory with the four files is reported as a 'custom' variant that
    accepts v3's language set (we cannot introspect an arbitrary directory, and
    refusing every language would make the override useless).
    """
    explicit = os.environ.get("VSCL_AISUBS_PARAKEET_MODEL", "").strip()
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit).rstrip(os.sep))
        if not model_ok(path):
            return []
        # Match on the directory NAME, not the full path, so a model that was
        # moved/relocated still counts as the variant it actually is. Keep in
        # sync with aisubs.lua's parakeet_installed().
        base = os.path.basename(path)
        for variant in VARIANTS:
            if base == variant.dirname:
                return [variant]
        return [Variant("custom", base or "parakeet", path, V3_LANGUAGES)]
    return [v for v in VARIANTS if model_ok(variant_dir(v))]


def select_variant(language: str | None) -> "Variant | None":
    """The installed variant to run `language` (normalize it first), else None.

    English — and `auto`, which on a v2-only box can only mean English —
    prefers v2 (the English-specialised model). Anything else needs v3.
    VSCL_AISUBS_PARAKEET_VERSION=v2|v3 forces one variant; None if it is not
    installed, so callers report "not installed" rather than running another.
    """
    installed = installed_variants()
    if not installed:
        return None
    forced = os.environ.get("VSCL_AISUBS_PARAKEET_VERSION", "").strip().lower()
    if forced:
        return next((v for v in installed if v.tag == forced), None)
    if language is None or language == "en":
        english = next((v for v in installed if v.tag == "v2"), None)
        if english is not None:
            return english
    return next((v for v in installed if language is None or language in v.languages), None)


def model_label(language: str | None = None) -> str:
    """Label of the variant that runs `language` (English when None).

    Used by `backends.parakeet.model_label()` for the CLI's
    "Backend: parakeet — <label>" line: English runs v2 when it is installed,
    every other language runs v3 — the label has to follow the language or the
    status line names a model that is not the one loading.
    """
    variant = select_variant(normalize_language(language) or "en") or select_variant(None)
    return variant.label if variant else "parakeet-tdt-0.6b-v2"


def supported_languages() -> set:
    """Every language covered by an INSTALLED variant."""
    langs: set = set()
    for variant in installed_variants():
        langs |= set(variant.languages)
    return langs
