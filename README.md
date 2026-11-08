# Orb

A fully local voice assistant for macOS with a character. You talk (or type), a small LLM answers
in character, a neural voice speaks it, and an animated orb in a browser page shows what it's
doing: listening, thinking, speaking. It can also watch your screen and react to it.

Nothing leaves the machine: the LLM, speech-to-text, text-to-speech and vision all run on a laptop.
It's the live engine behind NoRA (her persona and voice plug in through `ORB_PERSONA`).

Orb grew out of [vtube-cat](../vtube-cat), which drives a VTube Studio avatar with the same brain.
This repo keeps only the orb.

```
mic ──► ears.py (Whisper, MLX) ──┐
                                 ├──► brain.py ──► Ollama ──► streamed reply
screen ──► eyes.py (vision) ─────┘        │
                                          ├── [emotion:happy] tags ──► orb colour
                                          └── sentences ──► voice.py ──► speakers
                                                  (Kokoro, or Qwen3-TTS via audio.cpp,
                                                   optional Seed-VC / RVC conversion)
                                                                   └── loudness ──► orb pulse
                                     body port 127.0.0.1:47865 ──► orb_body.py ──► orb page at 127.0.0.1:47866
```

## What makes it work

- **Talks before it's done thinking.** Replies stream from the model, and each sentence is sent to
  TTS the moment it ends. A two-thread pipeline renders sentence 2 while sentence 1 plays, so the
  first words come out while the model is still writing the rest.
- **The orb pulses with the real audio.** `brain.py` builds a loudness curve from each rendered WAV
  (RMS per frame, a noise floor, syllable peaks above a local average) and streams it to the orb
  at 30 fps.
- **The body and the brain are separate processes.** `orb_body.py` serves the page and takes
  plain-text or JSON commands on a local TCP port. The LLM sets the mood with inline tags like
  `[emotion:annoyed]`, which are stripped before speaking.
- **Small-model guardrails.** A 3B model loops and parrots easily, so the brain caps sentences per
  reply, trims history, cuts generation when the model starts writing the user's lines, and drops
  sentences it already said in recent turns. Persona examples are fed as real past messages instead
  of a script, so the model copies the style without continuing the script.
- **Can be left on.** The model sees a sliding window of recent exchanges (about 1,200 of its
  4,096 tokens), so the prompt never grows. In a long silence only the newest two speak-ups stay
  in that window, so its own monologue can't push your conversation out of view. The orb shows a
  context gauge (Ollama's own token count), and when something you said is about to slide out of
  view the chat log says so and the character gets a plain note it may pass on, once.
- **Sleeps instead of nagging.** After 20 minutes with no word from you (the orb's "sleeps
  after" setting) it signs off, asks whether to end the session or stay on standby, and unloads
  the model. The mic stays on: talk or type to wake it, or say "end session". The orb shows a
  card with the same two choices.
- **Doesn't hear itself.** In voice mode the mic is ignored while the character speaks and for a
  moment after, and anything captured during a reply is thrown away.
- **Screen commentary.** `eyes.py` grabs the screen about once a second, skips frames that didn't
  change, and asks a small vision model for a one-line caption. The character comments on new
  things on its own, with a cooldown, and always answers you first.
- **Speaks up on its own.** After a randomized stretch of silence it may check in, or answer
  `PASS` and stay quiet. If you brush it off ("yeah" counts) or ignore it, it escalates instead
  of repeating itself: check in, give you space, a jab, then muttering to itself. Talking to it
  properly resets that. Say "shh" to stop it until you talk again. Capped at 8 an hour, and it
  never cuts in while you're mid-sentence. In orb mode the **Speaks up** fader sets how soon and
  how often (left = never, right = after ~15 s and no PASS), the chat log shows every nudge,
  PASS and brush-off, and a countdown under the orb says when it'll speak up next. Timings are
  in `initiative.py`; `--no-initiative` turns it off.
- **One-command setup.** The launchers check for missing packages and re-run themselves through
  `uv` on Python 3.12 with exactly what they need. Voice models download on first use.

## Run it

You need macOS on Apple Silicon, [Ollama](https://ollama.com) and [uv](https://docs.astral.sh/uv/).

```bash
ollama pull gemma3:1b-it-q8_0     # the default model; the orb page has a model picker
```

Double-click `Orb.command` (or run `python3 orb_body.py`). The orb page opens in your browser.
Click **Start**: the first time it downloads the voice and speech models, then it measures room
noise (stay quiet for a second) and starts listening. Just talk. **Stop** turns off the mic and
unloads the model; **Start** works again after it. Close the Terminal window to quit.

| Orb | Means |
|---|---|
| grey, still | off |
| grey, shimmering | starting up (loading models, measuring room noise) |
| soft breathing, swells as you talk | listening |
| lights circling inside it | thinking |
| pulsing with the voice | speaking |
| dimmed, mic icon | muted (the mute button, or press M): it ignores the mic |
| "· quiet mode" under the name | you said "shh": it won't speak up on its own until you talk to it |
| very dim, slow breathing | sleeping: model unloaded, mic on. Talk or type to wake it |

The colour follows the character's mood: warm when happy, red when annoyed, blue when sad.
Subtitles under the orb show what you said while it thinks, then each sentence as it's spoken.
The chat button (or C) opens a log of the conversation, with a box to type to it instead of
talking (mute plus typing makes it a text chat). It lives in memory only: nothing is
saved to disk, and Clear empties it.
`python3 brain_type.py` (with `orb_body.py` running) lets you type from a terminal instead.
`python3 orb_body.py --watch` makes Start turn on screen commentary as well (`--video` to heckle
what you're watching; run `ollama pull qwen2.5vl:3b-q8_0` first). The voice loop's output goes
to `orb.log`.

## Make it your own

- **Persona:** `persona.md` holds the default character. `personas/` has more. To use another
  file without copying it, start with `ORB_PERSONA=/path/to/persona.md ./Orb.command`.
- **Voice per persona:** put `<name>.voice.json` beside the persona file to pick its voice: a
  Kokoro voice, a Qwen3-TTS voice clone served by audio.cpp, or a fine-tuned Seed-VC model that
  re-voices every sentence. The format is at the top of `voice.py`.
- **Settings** sit at the top of each file: model, voice and speaking speed in `brain.py`, mic
  sensitivity in `ears.py`, the vision model in `eyes.py`, speak-up timing in `initiative.py`.
- **RVC (optional):** set `RVC_MODEL` in `brain.py` to a folder holding an RVC `.pth` voice you
  have the rights to use; sentences are then re-voiced through
  [Applio](https://github.com/IAHispano/Applio) in a background worker.

## Files

| File | What it does |
|---|---|
| `Orb.command`, `orb_body.py`, `orb/index.html` | The orb: launcher, command port, page, Start / Stop |
| `brain.py` | Prompting, streaming, tag parsing, the speech pipeline and the loudness curve |
| `brain_voice.py`, `brain_type.py`, `launcher.py` | Entry points (mic, typing) that set up dependencies |
| `ears.py` | Mic capture, voice activity detection, on-device Whisper |
| `initiative.py` | When to speak up unprompted: silence timer, escalation, brush-off grader, cap, "shh" |
| `eyes.py` | Screen capture, change detection, vision captions |
| `voice.py` | Kokoro, macOS and audio.cpp TTS, and the voice-conversion workers |
| `seedvc_worker.py`, `rvc_worker.py` | Background voice conversion over stdin/stdout JSON |
| `test_initiative.py` | Checks the speak-up logic with a fake clock |
| `docs/` | The hand-off specs Orb and its music ears were built from |

## How it was made

This project was **built with AI under my direction.** Claude (Anthropic's AI coding assistant)
wrote the code. I set the goals and the design, chose the models and the trade-offs, tested it
live, and directed every change.
