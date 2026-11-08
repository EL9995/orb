# Spec: music ears for vtube-cat (max accuracy)

Hand-off spec for a fresh session. Repos: `~/Documents/GitHub/Eric Portfolio/vtube-cat` (the orb, public)
and `~/Documents/GitHub/Eric Portfolio/fl-studio` (the pitch classifier and FL dataset; moved out of model-factory 2026-11-07).
Machine: MacBook M5, 16 GB, macOS, everything local. Written 2026-11-05.

## Goal

The orb hears music, not just speech: which notes and chords are playing, the key and tempo, and,
for released songs, what the song is. Rook can then comment on it. Today `ears.py` sends all mic audio
to Whisper, so lyrics get transcribed as if Eric said them, and anything else is ignored.

## Why not the existing classifier

`fl-studio/music/pitch_classifier.py` (a numpy MLP from scratch) was trained on 6 FL presets of
clean exports (4,537 clips: notes, scales, chords). It proved the point (100% held-out vs ~8% for the
audio LLMs), but it has never heard a real instrument, a full mix or the laptop mic. Keep it as a
baseline in the shootout and stop investing in it.

## Step 1: the shootout (decides everything else)

Score the candidates on audio **none of them trained on**. Our FL clips are home turf for our own
classifier, so they don't count as a fair test.

| Test set | What it checks | Size |
|---|---|---|
| NSynth test split | Single notes, ~1,000 instruments, pitch labels | 350 MB download, ~0.5 GB unpacked |
| GuitarSet (mono-mic audio + annotations) | Real guitar, chords and playing, note labels | 657 MB download |
| ~20 clips through the MacBook mic | FL notes/chords on the speakers, humming, a song | a few MB |

Candidates (note transcription):
- **Spotify Basic Pitch**: polyphonic, audio in, MIDI out.
- **MuScriptor** (audio.cpp): small first, medium if small is close.
- **SheetSage2** (audio.cpp): melody + harmony.
- **YourMT3+** (optional): the strongest multi-instrument model, heavier.
- **Our classifier**: the baseline.

Score with `mir_eval` (note F1 with onsets, pitch accuracy). Also measure speed (seconds per minute
of audio) and peak RAM. The winner is the most accurate model that fits the RAM budget below.

## Step 2: the stack (after the shootout)

```
mic → Silero VAD ─┬─ speech → Whisper (as today)
                  └─ not speech → CED tagger ("music 0.88", "dog bark") ─┬─ music → bars buffer
                                                                          └─ other sounds → later
bars buffer → BS-RoFormer stems → shootout winner (chords/keys stem) + CREPE (bass, vocal)
            → BTC chords, key, beat_this tempo + downbeat → "[hearing: Am - F - C - G, 95 BPM]" → brain
            → Chromaprint + AcoustID → "[song: title - artist]" for released songs
```

- Music goes to Rook the way screen notes do today: as a bracketed note in the next turn, or as an
  initiative trigger ("a song started"). Lyrics never reach Whisper as user speech.
- It works a few bars behind the music (separation + transcription isn't real time). Fine for
  commentary, not for live note readouts.
- AcoustID needs a free API key that Eric registers himself.

## Build time (estimate)

| Piece | Time |
|---|---|
| Shootout harness, downloads, scoring | ½–1 day |
| Stems (BS-RoFormer) on files | a few hours |
| Notes per stem + CREPE | ½–1 day |
| Chords, key, tempo, bar lines | ½ day |
| Song ID | 2–3 hours |
| Live in the orb (VAD + CED routing, bar buffering, tuning in Eric's room) | 1–2 days |
| **Total** | **~3–5 working days**, plus Eric's testing |

## Disk budget (sizes checked on Hugging Face / PyPI / Zenodo, 2026-11-05)

Free space at writing: **96 GB**.

| Item | Size |
|---|---|
| SheetSage2 GGUF | 2.7 GB |
| MuScriptor small / medium / large GGUF | 0.41 / 1.2 / 5.5 GB (start with small) |
| Basic Pitch env (tensorflow-macos + coremltools via uv) | ~1–1.5 GB (estimate) |
| YourMT3+ checkpoint + env (optional) | ~1 GB (estimate) |
| NSynth test + GuitarSet | ~1 GB downloads, ~1.2 GB unpacked |
| BS-RoFormer q8 | 0.17 GB (Mel-Band RoFormer q8: 0.25 GB) |
| CED small (base) | 0.09 GB (0.34 GB) |
| Smart Turn v3.2 | 0.03 GB |
| CREPE full, beat_this, BTC, Silero VAD, fpcalc | ~0.2 GB together (estimate) |
| **Shootout + full stack** | **~7–9 GB** (~15 GB with every large variant) |

- Delete the dataset archives after unpacking, and drop the losing shootout models afterwards.
- Stems are temporary: process them in memory or delete after use. No WAV files kept (Eric listens
  in FL; see the no-WAV-exports rule).
- Torch is already in the uv cache (5.4 GB total cache), so CREPE / beat_this add little.

## RAM budget

The running app must stay under ~8 GB. Today's voice loop is ~4.5 GB (cogito 3.5 GB on GPU, Whisper,
Kokoro). That leaves ~3.5 GB for the music ears: BS-RoFormer + a small transcriber + CED should fit,
SheetSage2 (2.7 GB of weights) may not alongside the LLM. Measure in the shootout.

## Risks

- `basic-pitch` 0.4.0 pins `tensorflow-macos<2.15.1` on macOS, which may not support Python 3.12;
  it may need its own Python 3.11 env, or its ONNX / CoreML model run directly.
- The mic: room echo, speaker coloration and Rook's own voice. Mute the music ears while Rook speaks
  (the same rule ears.py already uses for speech).
- Full mixes will have wrong notes in dense parts. Chords and key are more reliable than every note.

## Constraints

Same as ORB_SPEC.md: local only, Python via uv, minimal dependencies, nothing personal committed,
avatar mode untouched, keep the AI-built note in the README. Datasets and model weights stay out of git.
