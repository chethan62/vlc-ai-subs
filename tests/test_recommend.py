"""Unit tests for the VRAM-aware "Recommended" model picker.

Boundary cases are asserted exactly — this caught a real bug: with a GPU
present but < 2 GB VRAM, the picker used to fall through to the RAM path
and could recommend `medium` for a card that can only run `base`.
"""

import pytest

import aisubs_whisper


@pytest.mark.parametrize(
    ("vram_mb", "ram_gb", "expected"),
    [
        # GPU path (CTranslate2 / int8_float16 ≈ 2x GGML footprint)
        (8192, 0, "large"),
        (8000, 0, "large"),
        (4096, 0, "medium"),   # GTX 1650 — this box
        (2049, 0, "small"),
        (2000, 0, "small"),
        (1999, 0, "base"),     # just under the small threshold
        (1024, 0, "base"),     # small GPU must NOT borrow the RAM path
        (4096, 16, "medium"),  # GPU present -> RAM is ignored
        # CPU path — no usable GPU (vram == 0), sized by system RAM
        (0, 16, "medium"),
        (0, 8, "medium"),
        (0, 4, "small"),
        (0, 1, "base"),
    ],
)
def test_recommend_model(monkeypatch, vram_mb: int, ram_gb: int, expected: str):
    monkeypatch.setattr(aisubs_whisper, "_detect_vram_mb", lambda: vram_mb)
    monkeypatch.setattr(aisubs_whisper, "_detect_ram_gb", lambda: ram_gb)
    assert aisubs_whisper._recommend_model() == expected


def test_recommend_ignores_backend_arg(monkeypatch):
    """The backend_name arg is vestigial — WhisperX is the only engine."""
    monkeypatch.setattr(aisubs_whisper, "_detect_vram_mb", lambda: 4096)
    monkeypatch.setattr(aisubs_whisper, "_detect_ram_gb", lambda: 16)
    assert aisubs_whisper._recommend_model("whisper.cpp") == "medium"
    assert aisubs_whisper._recommend_model("faster_whisper") == "medium"