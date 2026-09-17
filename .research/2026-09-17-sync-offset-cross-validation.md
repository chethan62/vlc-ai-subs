# The sync correction that looked obvious, and the second file that killed it

**Conclusion first: no sync correction ships.** The offset that repairs the test film by 3.4×
destroys a second film, because the bias belongs to the first film's *reference track*, not to
our transcription. Cross-validating on a file with a different audio start time is what
separated the two — a single file could not.

## The temptation

The film's own numbers had been on the books for several rounds: our cues sit ~0.53 s early
against its professional track, and every cue is off by roughly the same amount. That is the
signature of a systematic offset — the kind you fix once, globally, forever.

Sweeping every offset against that track made it look conclusive:

| shift | matches within 0.15 s | rate |
| --- | --- | --- |
| +0.00 s | 220 | 11.3 % |
| +0.30 s | 460 | 23.6 % |
| +0.50 s | 737 | 37.7 % |
| **+0.55 s** | **740** | **37.9 %** |
| +0.70 s | 509 | 26.1 % |
| +1.008 s (the audio lead) | 142 | 7.3 % |

A broad, smooth peak — 3.4× better than doing nothing — and it also explained the earlier
failed experiment: correcting by the *full* audio lead (1.008 s) overshoots the true optimum by
nearly a factor of two, which is why that correction measured as harmful.

The mechanism I proposed was principled: our timestamps live on the audio timeline
(container − lead), and a transducer emits tokens *after* the speech. So the correction should
be `lead − emission-lag`. With lead = 1.008 s and an optimum of +0.55 s, that implies an
emission lag of ≈0.46 s — a **model** constant, which would apply to every file.

## The cross-validation

A model constant makes a falsifiable prediction: on a file whose audio starts at 0.000 s, the
lead contributes nothing, so the predicted optimum is **−0.46 s** — the opposite sign.

`Lucky.S01E01.No.Shortcuts` (47.5 min, audio `start_time` **0.000**, three embedded English
subtitle tracks) is exactly that file. Our engine transcribed it (453 cues, 30 s chunks,
~1.6 GB peak, ~10 min), and the sweep against its own reference track:

| shift | matches within 0.15 s | rate |
| --- | --- | --- |
| −0.50 s | 27 | 6.0 % |
| −0.20 s | 85 | 18.8 % |
| −0.10 s | 160 | 35.3 % |
| **+0.00 s** | **236** | **52.1 %** |
| +0.10 s | 217 | 47.9 % |
| +0.30 s | 46 | 10.2 % |
| +0.55 s (the film's optimum) | 14 | 3.1 % |

**The prediction failed.** The optimum is +0.00 s — there is no shift to make — and the
predicted −0.46 s would have scored 6.6 %. Applying the film's "fix" here would take 52.1 %
down to 3.1 %.

So the 0.53 s bias is **not** a property of the pipeline: no emission-lag constant exists, and
the lead (already device-detected and reported) is not the correction either. What the numbers
actually describe is that **the film's professional track sits about half a second from the
speech it describes** — its own authoring quirk. The plugin's existing behaviour (report the
lead, correct nothing) is the correct behaviour, and this is now measured rather than assumed.

## The result worth keeping

On a clean reference track, **52.1 % of our cue starts land within 0.15 s of professional
timings with no correction at all** (241 with the plateau centre, 53.2 %). The film's 11.3 %
"sync problem", which has been listed as an open defect for several rounds, is largely an
artefact of that film's reference track rather than our timing error. Any future sync claim
should be measured on a file whose reference is not itself suspect — and cross-validated on a
second file before it is believed.

## A methodology footnote: an optimum is a plateau, not a point

`tools/sync_offset.py` was written to make this reproducible, and its first version reported an
arbitrary *edge* of the optimum: with a 0.15 s tolerance, a perfectly aligned file has a plateau
of equally-good shifts from −0.15 s to +0.15 s, and a plain argmax returns −0.15. Refining
around the argmax does not fix it — it explores one edge rather than finding the middle. The
tool now reports the **centre of the best plateau**, which is the honest estimate; four unit
tests pin it (a 0.6 s shift is reported as 0.60, an aligned file as 0.00). Those tests caught
the bug, and one of them encodes this note's finding as arithmetic: the film's optimum, applied
to the second file, loses every match.

## Reproducing this

The decoded audio these numbers were measured on has been deleted — 267 MB of regenerable `wav`.
Recreate it from the film, and score any two SRTs with the tool in the repo:

```bash
ffmpeg -i Disclosure.Day.2026.1080p.MA.WEBRip.10Bit.DDP5.1.x265-NeoNoir.mkv \
       -map 0:a:0 -ar 16000 -ac 1 /var/tmp/film_audio.wav
PYTHONPATH= venv/bin/python tools/sync_offset.py our-subtitles.srt its-professional-track.srt
```

Every result file behind the tables above — the generated SRTs, the reference track, the scoring
scripts — is kept; only the audio, which ffmpeg reproduces exactly, was removed.

## Loose ends

- The second film's clean track makes it the right place to compare **engines** on sync without
  the reference-offset confound: our engine scores 52.1 % at zero there. The same run through
  CrispASR (chunked, ~12 min) was launched to complete that comparison.
- Recall remains CrispASR's measured advantage (59 % vs 33 % on regions we fail, see
  `2026-09-16-asr-landscape-and-crispasr.md`); sync does not separate them yet on either file.
- The film's reference track being ~0.5 s off its own speech is worth remembering when reading
  earlier notes that treated its timing as ground truth for *offsets* (it remains sound as
  ground truth for cue count, cue duration, reading speed and speaker dashes).
