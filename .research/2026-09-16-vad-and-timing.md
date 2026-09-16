# What needs improvement: VAD, cue timing, and the standards we don't implement

Research round, 2026-09-16. Every claim below is marked with how it was established:
**[measured]** = run here, on this machine, on real media; **[cited]** = from a published
source, quoted; **[unverified]** = plausible but not yet checked. The plugin's rule is
that docs may only claim what was measured, so unverified items are labelled as such
rather than quietly promoted.

## 1. Voice activity detection

whisper.cpp 1.9.2 has full VAD support — `--vad`, `-vm`, `-vt`, `-vspd`, `-vsd`,
`-vmsd`, `-vp`, `-vo` [measured: `whisper-cli --help`]. No VAD model was installed here,
so the plugin has been fighting hallucination entirely downstream: a phrase blocklist
(`core/blocklist.py`) and a repetition deloop (`MAX_REPEATS = 7`).

**Silero VAD does remove hallucination [measured].** On 60 s of music/credits (a stretch
the film's own transcript never reached, i.e. no speech):

| | output | wall |
| --- | --- | --- |
| plugin's current flags (`-mc 0 -nf -bs 1 -sns`) | `you you` | 6.33 s |
| + `--vad -vm ggml-silero-v5.1.2.bin` | *(nothing)* | **0.70 s** |

So it is both a correctness fix and — on non-speech — a 9× speedup.

**But as a gate it loses real dialogue, and tuning does not fix that [measured].** On a
dialogue-dense minute (5040–5100 s, 34 cues), against the film's own transcript as ground
truth, the line *"It was dark. I was in bed and I dreamed that something was watching
me."* is present in the no-VAD run and **absent from every VAD configuration**:

| config | wall | words | dark | bed | dreamed | watching |
| --- | --- | --- | --- | --- | --- | --- |
| no VAD | 9.7 s | 108 | ✓ | ✓ | ✓ | ✓ |
| `--vad` (defaults) | 9.2 s | 81 | ✗ | ✗ | ✗ | ✗ |
| `-vt 0.35` | 8.8 s | 76 | ✗ | ✗ | ✗ | ✗ |
| `-vp 300` | 9.5 s | 101 | ✗ | ✗ | ✗ | ✓ |
| `-vt 0.35 -vp 300` | 9.5 s | 99 | ✗ | ✗ | ✗ | ✓ |
| `-vt 0.25 -vp 500 -vsd 500` | 9.5 s | 102 | ✗ | ✗ | ✗ | ✓ |

`--vad` processes only the segments the VAD considers speech, so anything it scores as
non-speech is gone. **Verdict: do not gate on VAD.** A hallucinated `you you` is visible
and the blocklist already covers that class; a missing line is invisible and
unrecoverable. The safe form is the inverse — keep every transcribed segment, but *drop*
one whose span contains no detected speech — which needs a callable VAD (sherpa-onnx's
`VoiceActivityDetector` is present in the `venv-whisperx` environment [measured]). That is
the next round's work, not this one.

## 2. Cue timing is approximate on every engine

The plugin's cue spans come from word timestamps. Read the words-per-minute implied by a
cue's own span and several cues are **physically impossible** — natural speech is
120–160 wpm and peaks around 200 [cited: standard subtitle practice; Netflix's 20 CPS
English ceiling implies the same order]:

- Parakeet, film: a 6-word cue in 0.80 s = **450 wpm**; that minute's median **217 wpm**,
  max **500 wpm**, 19/34 cues over 200 [measured].
- whisper.cpp, same minute, default segment times: `How long you been able to do this?`
  (8 words) in 1.40 s = **343 wpm** [measured].

**whisper.cpp's DTW token timestamps do not obviously fix it [measured].** With
`-dtw medium -ojf` the JSON exposes per-token `t_dtw` times; merging them gives coherent
per-word boundaries but the same order of rates (190 wpm across the clip, zero inter-word
gaps everywhere). DTW sharpens *where* each word sits; it does not change the underlying
scale.

**WhisperX's alignment: measured during this round** — see the note at the end. The
README recommends it for timing-sensitive work, and that claim had never been checked.

## 3. Standards the plugin does not implement

[cited: Netflix English (USA) Timed Text Style Guide, and the Subtitle Templates guide]

- **Songs.** "Subtitle all audible song lyrics that do not interfere with dialogue. Lyrics
  should be transcribed verbatim as per the audio. Use song title identifiers when
  applicable — song titles should be in quotation marks, for example
  `["Forever Your Girl" playing]`." The plugin transcribes lyrics if the model hears them
  but does no lyric handling and emits no song identifiers.
- **Italics.** The guides italicise song lyrics and certain off-screen/voice-over
  speech. SRT has no official styling, but `[unverified — reported by third-party
  tooling, not yet checked here]` VLC renders `<i>` in SRT as italics while ignoring
  `<font>`. Before implementing, verify in VLC's own source or a real render, per this
  project's rule about player behaviour.
- **Speaker dashes** (two speakers in one event, marked with a leading `-`) would need
  diarization. Out of scope for ASR-only subtitles.
- **SDH** (sound-effect descriptions, speaker IDs) is a different output mode, not a fix.

## 4. Ranked: what to improve next

| # | item | evidence | cost |
| --- | --- | --- | --- |
| 1 | Verify or drop the "WhisperX has the timing" claim | this round | done below |
| 2 | VAD-as-filter: drop segments over non-speech, keep everything else | §1, 4 tunings | medium (needs the sherpa-onnx VAD call) |
| 3 | Song/lyric handling per the Netflix guides: verbatim lyrics, identifiers, italics | §3 | medium, depends on verifying VLC's `<i>` |
| 4 | AMD/Intel Vulkan path: the README says it is unexercised on Radeon/Arc | this machine has neither | needs hardware |
| 5 | Parakeet v3 per-language accuracy: the plugin offers 25 European languages, unmeasured | no benchmark exists here | needs ground-truth corpora |
| 6 | VLC 4.0 extension registration is inferred, not observed | 4.0's manager is GUI-only | needs a 4.0 install |

## Measured during this round: does WhisperX actually time better?

**No — and the claim that it does is now withdrawn.** WhisperX `small` with its alignment
pass, over the same 60 s minute:

| | cues | median implied rate | max | over 200 wpm |
| --- | --- | --- | --- | --- |
| Parakeet (film run) | 34 | 217 wpm | 500 wpm | 19/34 |
| WhisperX (aligned) | 33 | **244 wpm** | **580 wpm** | 23/33 |

WhisperX's cue spans are *tighter* (median 1.00 s, min 0.26 s) and its implied rates are no
better. Its alignment is real — it computes word-level boundaries from a forced-alignment
model — but that does not make its cue timings more plausible than Parakeet's. The README
recommended WhisperX for timing-sensitive work on the strength of the alignment existing,
not on any measurement; that recommendation has been removed.

## The metric I used to condemn the timings was partly wrong

Worth recording because it nearly produced a false conclusion. The implied words-per-minute
above is computed from a cue's span, and that span ends at the last word's **start**, not
the end of its audio — a word takes ~0.3 s, so a three-word cue loses a third of its span
and its rate is inflated accordingly. Short cues made of function words ("I was 20.",
5 syllables in 0.76 s) read as 237 wpm where a clipped delivery is genuinely 4–6
syllables/s.

So the honest version of this finding:

- **The artifact is visible in the per-cue detail.** WhisperX's long cues are entirely
  plausible — "You've been able to do this since yesterday since the bird." over 4.38 s is
  137 wpm — while every implausible rate comes from its *short* cues: "I was 20." 0.76 s,
  "I was in college." 0.58 s, "Yeah." 0.58 s. Short cues are exactly where a span that ends
  at the last word's start loses the largest fraction of itself.
- **Truly impossible cases still exist** — 450 and 580 wpm are 8–10 words per second, which
  no speaker produces. Those are real timestamp defects, in both engines.
- **The bulk of the "impossible" rates are the metric**, not the engine. A cue-level
  words-per-minute test needs the words' *audio* spans, which the plugin does not currently
  keep (it keeps start times and derives ends).
- Fixing the metric is itself an improvement worth making: keep each word's real end from
  the aligner (WhisperX) or the derived one (Parakeet, already bounded by the next word's
  start) and measure the reading speed from the *speech* extent rather than the cue span.
  It also makes a caption linger over its final word's audio instead of vanishing a frame
  after its start.

## Song lyrics and italics: VLC's support verified from source

[measured: `videolan/vlc`, `modules/codec/subsdec.c`, 3.0.x] The SubRip decoder's tag
parser handles `<i>`/`</i>` by setting `STYLE_ITALIC` (line ~690), and `<b>`, `<u>`, `<s>`
alongside it. `{\i1}`-style override tags are handled too. So emitting `<i>` in SRT is
safe for VLC — the earlier third-party claim is now confirmed at the source, which is the
standard this project holds player behaviour to.
