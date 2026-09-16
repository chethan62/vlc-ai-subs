"""Audio decoding shared by the engines that shell out to ffmpeg.

Parakeet (sherpa-onnx) and whisper.cpp both want a 16 kHz mono PCM wav, so the
decode + temp-file handling lives here once instead of in each runner.

Which *track* gets decoded matters as much as the decoding. ffmpeg's own default
takes the first audio stream, and on real dual-audio releases that is the dub, not
the dialogue: measured here, a "MULTi VFF" file's first track is French and a
"DUAL" file's is Portuguese, so asking for English subtitles handed an English-only
model French audio — which produced plausible-looking nonsense rather than an
error. Stream selection lives here for that reason.

Leaf module: stdlib only, no imports from core/, backends/ or the runners.
"""

import json
import os
import shutil
import subprocess
import tempfile

SAMPLE_RATE = 16000
DECODE_TIMEOUT = 600
PROBE_TIMEOUT = 30

# Container language tags are ISO 639-2 ("fre", "eng", "por") while the plugin's
# UI and engines speak 639-1 ("fr", "en", "pt"), so a straight string compare
# silently never matches — which is how a French track reads as "no match, take the
# first one". Both the /T and /B spellings are listed because containers use both.
ISO3_TO_1 = {
    "eng": "en", "fre": "fr", "fra": "fr", "por": "pt", "spa": "es", "ger": "de",
    "deu": "de", "ita": "it", "rus": "ru", "hin": "hi", "ara": "ar", "jpn": "ja",
    "kor": "ko", "chi": "zh", "zho": "zh", "cmn": "zh", "yue": "zh", "tur": "tr",
    "pol": "pl", "dut": "nl", "nld": "nl", "swe": "sv", "nor": "no", "dan": "da",
    "fin": "fi", "cze": "cs", "ces": "cs", "gre": "el", "ell": "el", "heb": "he",
    "hun": "hu", "rum": "ro", "ron": "ro", "ukr": "uk", "vie": "vi", "tha": "th",
    "ind": "id", "tam": "ta", "mal": "ml", "ben": "bn", "mar": "mr", "tel": "te",
    "urd": "ur", "fas": "fa", "per": "fa", "swa": "sw", "fil": "tl", "tgl": "tl",
}

# A track whose title says one of these is not the dialogue. The standard
# dispositions are useless here: measured on a real release, the audio-description
# track carried visual_impaired=0 and no flag at all — only title="Descriptive".
_NON_DIALOGUE_MARKERS = ("descript", "comment", "narration", "visually impaired", "audio desc")


def ffmpeg_path() -> str | None:
    """Absolute path to ffmpeg, or None when it is not on PATH."""
    return shutil.which("ffmpeg")


def normalise_language(code: str | None) -> str:
    """eng / fr-FR / FR / auto → the 639-1 code used for comparisons ("" if none)."""
    raw = (code or "").strip().lower().replace("_", "-")
    if not raw or raw == "auto":
        return ""
    base = raw.split("-")[0]
    if len(base) == 3:
        return ISO3_TO_1.get(base, base)
    return base


def list_audio_streams(media_path: str, timeout: float = PROBE_TIMEOUT) -> list[dict]:
    """The container's audio streams as ffprobe sees them.

    Each entry: ``index`` (absolute stream index, what ``-map 0:<index>`` wants),
    ``language``, ``title``, ``channels``, ``non_dialogue``. An empty list means
    ffprobe is unavailable or the file could not be read — callers then leave the
    choice to ffmpeg, which is what earlier builds did.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return []
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_streams",
             "-select_streams", "a", media_path],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    try:
        streams = json.loads(proc.stdout or "{}").get("streams") or []
    except ValueError:
        return []
    out = []
    for stream in streams:
        tags = stream.get("tags") or {}
        disposition = stream.get("disposition") or {}
        title = (tags.get("title") or "").strip()
        low = title.lower()
        out.append({
            "index": int(stream.get("index") or 0),
            "language": (tags.get("language") or "").strip().lower(),
            "title": title,
            "channels": int(stream.get("channels") or 0),
            "non_dialogue": bool(disposition.get("visual_impaired") or disposition.get("comment"))
            or any(marker in low for marker in _NON_DIALOGUE_MARKERS),
        })
    return out


def _label(streams: list[dict], stream: dict) -> str:
    bits = [f"track {streams.index(stream) + 1}/{len(streams)}"]
    if stream["language"]:
        bits.append(stream["language"])
    if stream["title"]:
        bits.append(stream["title"])
    return " · ".join(bits)


def choose_audio_stream(streams: list[dict], language: str | None = None) -> tuple[int | None, str]:
    """Which audio stream to transcribe — and why, as a sentence for the user.

    Order: ``VSCL_AISUBS_AUDIO_TRACK`` if set (an absolute index, a 1-based
    position, or a language); drop descriptive/commentary tracks; prefer the
    requested language; otherwise the first one left.
    """
    if not streams:
        return None, "ffmpeg's own default track (no stream info available)"

    override = (os.environ.get("VSCL_AISUBS_AUDIO_TRACK") or "").strip()
    if override:
        wanted = normalise_language(override)
        for pos, stream in enumerate(streams, start=1):
            if override in (str(stream["index"]), str(pos)) or (wanted and _matches(stream, wanted)):
                return stream["index"], f"{_label(streams, stream)} (VSCL_AISUBS_AUDIO_TRACK)"
        return (streams[0]["index"],
                f"{_label(streams, streams[0])} (VSCL_AISUBS_AUDIO_TRACK='{override}' matched nothing)")

    dialogue = [s for s in streams if not s["non_dialogue"]] or streams
    wanted = normalise_language(language)
    if wanted:
        for stream in dialogue:
            if _matches(stream, wanted):
                return stream["index"], f"{_label(streams, stream)} — matches the requested {wanted}"

    chosen = dialogue[0]
    if len(streams) == 1:
        return chosen["index"], _label(streams, chosen)
    skipped = len(streams) - len(dialogue)
    skipped_note = f"{skipped} descriptive track(s) skipped" if skipped else "no descriptive track"
    wanted_note = f"no {wanted} track" if wanted else "auto-detect"
    return chosen["index"], f"{_label(streams, chosen)} ({wanted_note}, {skipped_note})"


def _matches(stream: dict, wanted: str) -> bool:
    tagged = normalise_language(stream.get("language"))
    return bool(tagged) and tagged == wanted


def decode_to_wav16k(media_path: str, timeout: float = DECODE_TIMEOUT,
                     stream_index: int | None = None) -> str:
    """Decode arbitrary media to a fresh 16 kHz mono PCM wav.

    *stream_index* is an absolute stream index from :func:`list_audio_streams`;
    when it is None ffmpeg picks the track, as older builds did.

    The caller owns the returned path (usually: ``os.unlink`` it).
    Raises RuntimeError with the ffmpeg stderr tail when decoding fails.
    """
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found — required by this backend")
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="aisubs_")
    os.close(fd)
    cmd = [ffmpeg, "-y", "-v", "error", "-i", media_path]
    if stream_index is not None:
        cmd += ["-map", f"0:{int(stream_index)}"]
    cmd += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", tmp]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise RuntimeError(f"ffmpeg decode failed: {(proc.stderr or '').strip()[:300]}")
    return tmp


def read_wav_pcm16(path: str) -> bytes:
    """Raw little-endian 16-bit PCM from a wav this module produced.

    Kept bytes-level and stdlib-only: the caller that needs float32 (WhisperX)
    already has numpy and does the scaling there.
    """
    import wave

    with wave.open(path, "rb") as handle:
        return handle.readframes(handle.getnframes())
