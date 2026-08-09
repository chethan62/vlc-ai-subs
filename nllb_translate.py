"""NLLB-200 cascade translator — upgrades the WhisperX `translate` task.

Pipeline: WhisperX transcribes in the source language (task="transcribe"),
then each segment's text is translated to English with NLLB-200 via
ctranslate2 (int8). Timestamps are untouched; only `text` changes.

Model: michaelfeil/ct2fast-nllb-200-distilled-1.3B (ctranslate2 int8, 200
languages), installed by install-nllb-model.sh.

LICENSE NOTE: NLLB-200 is CC-BY-NC-4.0 (research / non-commercial). The model
is a separate runtime download — the plugin code stays MIT. `VSCL_AISUBS_NLLB=0`
switches the translate task back to Whisper's built-in translation.

Heavy imports (ctranslate2, transformers) are deferred to NllbTranslator so
this module stays importable in dev venvs without them (unit-testable).
"""

import logging
import os

logger = logging.getLogger(__name__)

# Default install location (matches install-nllb-model.sh).
MODEL_DIR_DEFAULT = os.path.expanduser(
    "~/.local/share/vlc-ai-subs/nllb-200-distilled-1.3B-int8"
)

# Target language for the cascade: English.
TARGET = "eng_Latn"

# Whisper language code → NLLB FLORES-200 code. Covers the languages whisper
# users realistically translate; unmapped codes fall back to Whisper translate.
WHISPER_TO_FLORES = {
    "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "as": "asm_Beng",
    "az": "azj_Latn", "ba": "bak_Cyrl", "be": "bel_Cyrl", "bg": "bul_Cyrl",
    "bn": "ben_Beng", "bo": "bod_Tibt", "bs": "bos_Latn", "ca": "cat_Latn",
    "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "de": "deu_Latn",
    "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn", "et": "est_Latn",
    "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fo": "fao_Latn",
    "fr": "fra_Latn", "gl": "glg_Latn", "gu": "guj_Gujr", "ha": "hau_Latn",
    "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn", "ht": "hat_Latn",
    "hu": "hun_Latn", "hy": "hye_Armn", "id": "ind_Latn", "is": "isl_Latn",
    "it": "ita_Latn", "ja": "jpn_Jpan", "jw": "jav_Latn", "ka": "kat_Geor",
    "kk": "kaz_Cyrl", "km": "khm_Khmr", "kn": "kan_Knda", "ko": "kor_Hang",
    "lb": "ltz_Latn", "ln": "lin_Latn", "lo": "lao_Laoo", "lt": "lit_Latn",
    "lv": "lvs_Latn", "mg": "plt_Latn", "mi": "mri_Latn", "mk": "mkd_Cyrl",
    "ml": "mal_Mlym", "mn": "khk_Cyrl", "mr": "mar_Deva", "ms": "zsm_Latn",
    "mt": "mlt_Latn", "my": "mya_Mymr", "ne": "npi_Deva", "nl": "nld_Latn",
    "nn": "nno_Latn", "no": "nob_Latn", "oc": "oci_Latn", "pa": "pan_Guru",
    "pl": "pol_Latn", "ps": "pbt_Arab", "pt": "por_Latn", "ro": "ron_Latn",
    "ru": "rus_Cyrl", "sa": "san_Deva", "sd": "snd_Arab", "si": "sin_Sinh",
    "sk": "slk_Latn", "sl": "slv_Latn", "sn": "sna_Latn", "so": "som_Latn",
    "sq": "als_Latn", "sr": "srp_Cyrl", "su": "sun_Latn", "sv": "swe_Latn",
    "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu", "tg": "tgk_Cyrl",
    "th": "tha_Thai", "tk": "tuk_Latn", "tl": "tgl_Latn", "tr": "tur_Latn",
    "tt": "tat_Cyrl", "uk": "ukr_Cyrl", "ur": "urd_Arab", "uz": "uzn_Latn",
    "vi": "vie_Latn", "yi": "ydd_Hebr", "yo": "yor_Latn", "yue": "yue_Hant",
    "zh": "zho_Hans",
}


def flores_code(whisper_code: str | None) -> str | None:
    """Map a whisper language code to an NLLB FLORES-200 code, or None."""
    if not whisper_code:
        return None
    return WHISPER_TO_FLORES.get(whisper_code.strip().lower())


def should_cascade(task: str, env_value: str | None) -> bool:
    """Cascade is on for translate unless VSCL_AISUBS_NLLB=0."""
    if task != "translate":
        return False
    return (env_value or "1").strip().lower() not in ("0", "false", "off")


class NllbTranslator:
    """Thin wrapper over ctranslate2's NLLB model + HF tokenizer."""

    def __init__(self, model_dir: str, device: str = "cpu", compute_type: str = "int8"):
        import ctranslate2  # lazy — heavy deps only when translating
        from transformers import AutoTokenizer

        self._ct = ctranslate2.Translator(model_dir, device=device, compute_type=compute_type)
        self._tok = AutoTokenizer.from_pretrained(model_dir, src_lang=TARGET)

    def translate(self, text: str, src_lang: str, tgt_lang: str = TARGET) -> str:
        self._tok.src_lang = src_lang
        ids = self._tok(text, add_special_tokens=True)["input_ids"]
        tokens = self._tok.convert_ids_to_tokens(ids)
        # Keep the leading __src_lang__ token; drop the trailing EOS
        # (ctranslate2 adds its own).
        if tokens and tokens[-1] == "</s>":
            tokens = tokens[:-1]
        result = self._ct.translate_batch(
            [tokens],
            target_prefix=[["__" + tgt_lang + "__"]],
            beam_size=1,
        )
        out = result[0].tokens
        if out and out[0] == "__" + tgt_lang + "__":
            out = out[1:]
        return self._tok.convert_tokens_to_string(out).strip()


def try_load_translator(
    model_dir: str | None, device: str = "cpu", compute_type: str = "int8"
) -> NllbTranslator | None:
    """Load the translator, or None if the model is missing/broken.

    The caller treats None as 'fall back to Whisper translate' — the
    translate task must never hard-fail because the NLLB model is absent.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return None
    try:
        return NllbTranslator(model_dir, device=device, compute_type=compute_type)
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        logger.warning("NLLB model failed to load (%s) — using Whisper translate", exc)
        return None


def translation_viable(before_texts: list, after_texts: list) -> bool:
    """True when the cascade produced usable output.

    Falls back to Whisper translate when there WAS source text to translate
    but nothing came back non-empty (empty output) or nothing changed at all
    (every call raised — source text kept). Partial successes are kept
    (per-segment tolerance); an empty transcript is fine.
    """
    return (not any(before_texts)) or (any(after_texts) and after_texts != before_texts)


def translate_segments(segments: list, src_flores: str, translator: NllbTranslator) -> list:
    """Translate each segment's text to English; timestamps preserved.

    A single segment that fails keeps its original text (the run continues).
    """
    out = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if text:
            try:
                text = translator.translate(text, src_flores)
            except Exception as exc:  # noqa: BLE001 — one bad segment ≠ failed run
                logger.warning("NLLB segment translate failed (%s) — keeping source text", exc)
        out.append({**seg, "text": text})
    return out
