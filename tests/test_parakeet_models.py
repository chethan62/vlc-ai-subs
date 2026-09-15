"""Parakeet variant selection (core/parakeet_models.py).

This module is the authoritative source for "which model runs this language";
aisubs.lua mirrors it for the dialog's engine preview and parakeet_runner.py
loads it, so the rules pinned here are the ones both sides follow.
"""
import pytest

import core.parakeet_models as pm


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("VSCL_AISUBS_PARAKEET_MODEL", raising=False)
    monkeypatch.delenv("VSCL_AISUBS_PARAKEET_VERSION", raising=False)


def _install(tmp_path, *variants):
    """Create model directories for the named variants, return the root."""
    root = tmp_path / "models"
    for tag in variants:
        variant = next(v for v in pm.VARIANTS if v.tag == tag)
        d = root / variant.dirname
        d.mkdir(parents=True)
        for name in pm.MODEL_FILES:
            (d / name).write_text("")
    return root


def _use(monkeypatch, root):
    monkeypatch.setattr(pm, "MODELS_ROOT", str(root))


def _tag(language):
    """The tag of the variant selected for `language` (asserts it exists)."""
    variant = pm.select_variant(language)
    assert variant is not None
    return variant.tag


def _custom_tag(language):
    """Same, for the explicit-directory ("custom") variant."""
    variants = pm.installed_variants()
    assert [v.tag for v in variants] == ["custom"]
    variant = pm.select_variant(language)
    assert variant is not None
    return variant.tag


# ── normalization ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("en", "en"), ("EN", "en"), ("en-GB", "en"), ("en_US", "en"),
    ("zh-CN", "zh"), ("yue", "yue"), ("auto", None), (None, None), ("", None),
])
def test_normalize_language(raw, expected):
    assert pm.normalize_language(raw) == expected


# ── installed variants ───────────────────────────────────────────────────

def test_nothing_installed(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path / "empty")
    assert pm.installed_variants() == []
    assert pm.supported_languages() == set()
    assert pm.select_variant("en") is None
    assert pm.select_variant(None) is None


def test_v2_only(monkeypatch, tmp_path):
    _use(monkeypatch, _install(tmp_path, "v2"))
    assert [v.tag for v in pm.installed_variants()] == ["v2"]
    assert pm.supported_languages() == {"en"}
    assert _tag("en") == "v2"
    assert _tag(None) == "v2"      # auto on an English-only box
    assert pm.select_variant("fr") is None          # needs v3
    assert pm.model_label() == "parakeet-tdt-0.6b-v2"


def test_v3_only_but_still_preferred_for_english(monkeypatch, tmp_path):
    _use(monkeypatch, _install(tmp_path, "v3"))
    assert _tag("en") == "v3"
    assert _tag("fr") == "v3"
    assert pm.model_label() == "parakeet-tdt-0.6b-v3"


def test_both_installed_english_prefers_v2(monkeypatch, tmp_path):
    """v2 is the English-specialised model, so English keeps using it."""
    _use(monkeypatch, _install(tmp_path, "v2", "v3"))
    assert _tag("en") == "v2"
    assert _tag(None) == "v2"      # auto on an English-only box
    assert _tag("de") == "v3"
    assert pm.model_label() == "parakeet-tdt-0.6b-v2"
    assert "de" in pm.supported_languages() and "en" in pm.supported_languages()


def test_partial_model_files_do_not_count(monkeypatch, tmp_path):
    """A half-downloaded directory (missing joiner/tokens) is not installed."""
    root = _install(tmp_path, "v2")
    (root / pm.VARIANTS[0].dirname / "joiner.int8.onnx").unlink()
    _use(monkeypatch, root)
    assert pm.installed_variants() == []


def test_forced_version_selects_that_variant(monkeypatch, tmp_path):
    _use(monkeypatch, _install(tmp_path, "v2", "v3"))
    monkeypatch.setenv("VSCL_AISUBS_PARAKEET_VERSION", "v3")
    assert _tag("en") == "v3"
    assert pm.model_label() == "parakeet-tdt-0.6b-v3"


def test_forced_version_that_is_not_installed_is_none(monkeypatch, tmp_path):
    """Forcing v3 on a v2-only box must report "not installed", never silently
    run the other model (the runner turns this into an actionable error)."""
    _use(monkeypatch, _install(tmp_path, "v2"))
    monkeypatch.setenv("VSCL_AISUBS_PARAKEET_VERSION", "v3")
    assert pm.select_variant("en") is None
    assert pm.select_variant("fr") is None


# ── explicit directory override ──────────────────────────────────────────

def test_explicit_dir_is_recognised_by_name(monkeypatch, tmp_path):
    """A relocated model keeps the language set of the variant it actually is."""
    root = _install(tmp_path, "v2")
    source = root / pm.VARIANTS[0].dirname
    moved = tmp_path / "elsewhere" / pm.VARIANTS[0].dirname
    moved.parent.mkdir()
    source.rename(moved)
    monkeypatch.setenv("VSCL_AISUBS_PARAKEET_MODEL", str(moved))

    assert [v.tag for v in pm.installed_variants()] == ["v2"]
    assert pm.select_variant("fr") is None
    assert pm.model_label() == "parakeet-tdt-0.6b-v2"


def test_explicit_dir_trailing_slash_and_tilde(monkeypatch, tmp_path):
    root = _install(tmp_path, "v3")
    explicit = str(root / pm.VARIANTS[1].dirname) + "/"
    monkeypatch.setenv("VSCL_AISUBS_PARAKEET_MODEL", explicit)
    assert [v.tag for v in pm.installed_variants()] == ["v3"]


def test_unknown_explicit_dir_is_treated_as_the_wider_model(monkeypatch, tmp_path):
    """An arbitrary model directory cannot be introspected, so it is taken to
    cover v3's languages (refusing everything would make the override useless)."""
    d = tmp_path / "my-parakeet"
    d.mkdir()
    for name in pm.MODEL_FILES:
        (d / name).write_text("")
    monkeypatch.setenv("VSCL_AISUBS_PARAKEET_MODEL", str(d))

    variants = pm.installed_variants()
    assert [v.tag for v in variants] == ["custom"]
    assert _custom_tag("fr") == "custom"
    assert pm.select_variant("ja") is None          # outside v3's 25


def test_explicit_dir_missing_files_is_not_installed(monkeypatch, tmp_path):
    monkeypatch.setenv("VSCL_AISUBS_PARAKEET_MODEL", str(tmp_path / "nope"))
    assert pm.installed_variants() == []
    assert pm.select_variant("en") is None


# ── the v3 language table ────────────────────────────────────────────────

def test_v3_language_table_is_the_25_from_the_model_card():
    expected = {
        "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
        "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
        "sl", "es", "sv", "ru", "uk",
    }
    assert pm.V3_LANGUAGES == expected
    assert len(pm.V3_LANGUAGES) == 25
    assert pm.KNOWN_LANGUAGES == expected            # v2 adds nothing new
    assert "ja" not in pm.KNOWN_LANGUAGES and "zh" not in pm.KNOWN_LANGUAGES


def test_model_label_follows_the_language(monkeypatch, tmp_path):
    """English → v2, the other 24 languages → v3 (the label is what the status
    line prints, so it has to match the model the runner loads)."""
    _use(monkeypatch, _install(tmp_path, "v2", "v3"))
    assert pm.model_label("en") == "parakeet-tdt-0.6b-v2"
    assert pm.model_label(None) == "parakeet-tdt-0.6b-v2"
    assert pm.model_label("fr") == "parakeet-tdt-0.6b-v3"
    assert pm.model_label("uk") == "parakeet-tdt-0.6b-v3"
    # A language no variant covers: the run is refused before loading, so the
    # label just falls back to the default variant.
    assert pm.model_label("ja") == "parakeet-tdt-0.6b-v2"


def test_variant_labels_are_distinct():
    labels = [v.label for v in pm.VARIANTS]
    assert labels == ["parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v3"]
    assert len(set(labels)) == len(labels)