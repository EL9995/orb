# Spec: Voice Orb mode for vtube-cat (no VTube Studio)

Hand-off spec for a fresh session. Repo: `~/Documents/GitHub/Eric Portfolio/vtube-cat`
(public: `EL9995/vtube-cat`). Machine: MacBook M5, 16 GB, macOS, everything local.

## Goal

Run vtube-cat as a **voice assistant with an orb**, like ChatGPT Voice or Claude Voice: one
on/off button, an animated orb that shows its state, and no avatar or VTube Studio needed.
The brain, ears, voice and persona stay as they are. Only the "body" changes.

Done when Eric can open one page (or one window), click **Start**, talk to the character, and
watch the orb react: listening, thinking, speaking (pulsing with the voice), and off. No
terminals and no VTube Studio.

## How the code works today (read these first)

- `brain.py`: the main loop. Streams replies from Ollama, speaks them sentence by sentence,
  and drives the body through `BodyLink` (TCP `127.0.0.1:47865`, one plain-text command per line).
- `vts_body.py`: the current body. A daemon that takes these commands and drives VTube Studio:
  `emotion X`, `look X`, `anim X`, `energy X`, `mouth 0..1` (streamed at ~30 fps for lip sync),
  `talk N`, `status`, `quit`. Plain text or JSON `{"cmd": ..., "arg": ...}`.
- Lip sync: `brain.py` `mouth_envelope()` builds a 0..1 loudness curve from each rendered WAV and
  streams `mouth <v>` while the audio plays. **That curve is what pulses the orb.**
- State hooks already in `brain.py`: `speaking` (a threading.Event) is set while a reply is generated
  or spoken. It sends `look up` while thinking and `look center` when done. `emotion` tags come from
  the LLM as `[emotion:happy]` and are stripped before speaking.
- `ears.py` (mlx-whisper + sounddevice): records and transcribes. It already ignores the mic while
  the character speaks.
- `voice.py`: Kokoro TTS (optional RVC). `eyes.py`: optional screen watching (`--watch` / `--video`).
- `brain_voice.py` / `brain_type.py`: entry points for mic mode and typing mode.

## Fix first (stale model names: these models were removed or renamed on 2026-11-04)

- `brain.py` `MODEL = "hermes3:3b"`: **hermes3 is no longer installed.** Switch to an installed
  model. Ask Eric which. Good candidates: `cogito:3b-v1-preview-llama-q8_0`, `qwen3.5:4b`,
  `granite4.2:3b-q8_0`. Update the README's `ollama pull hermes3:3b` line to match.
- `eyes.py` `VISION_MODEL = "qwen2.5vl:3b"`: now installed as **`qwen2.5vl:3b-q8_0`**.
- Check with `ollama list` before relying on any name.

## Build

### 1. `orb_body.py`: a drop-in replacement for `vts_body.py`

- Listens on the **same TCP port (47865)** with the **same commands**, so `brain.py` needs no changes
  to drive it. Run it *instead of* `vts_body.py`, never alongside (same port).
- Serves the orb page (one HTML file) on `http://127.0.0.1:<port>` and pushes state to it over
  a WebSocket.
- Maps commands to orb state:
  - `mouth v` → the speaking level (drives the pulse); `mouth 0` for a while → back to listening
  - `look up` → **thinking**; `look center` → back to listening (or speaking if mouth > 0)
  - `emotion X` → orb tint (for example happy = warm, annoyed = red, sad = blue, neutral = default)
  - `energy X` → how lively the idle motion is
  - `anim`, `talk`: accept and ignore (or a small flourish), so nothing breaks
- Also accepts two new commands the page or brain can send: `state off|listening|thinking|speaking|muted`
  and `level v` (mic input level, so the orb swells when the *user* talks).

### 2. The orb page (`orb/index.html`, plain HTML + canvas or CSS, no build step)

States:

| State | Look |
|---|---|
| off | grey, still |
| listening | soft breathing; swells with mic level |
| thinking | slow shimmer or rotation |
| speaking | pulses with the `mouth` level |
| muted / shh | dimmed, small mute icon |

- One big **Start / Stop** button, plus a mute toggle. Show the character's name and the current
  state in small text under the orb.
- Respects light and dark mode. Works full-window and as a small window.
- Smooth the `mouth` value on the page (the brain sends ~30 fps; ease it so the orb doesn't jitter).

### 3. Start / Stop without terminals

- The page's Start button starts the voice loop (`brain_voice.py`'s logic) and Stop ends it.
  Simplest: `orb_body.py` launches and stops `brain_voice.py` as a child process. Cleaner, if it
  stays small: run the brain loop in a thread inside `orb_body.py`.
- One launcher command (or a double-clickable `.command` file, like `Audex Listen.command` on Eric's
  Desktop) that starts `orb_body.py` and opens the page in the browser.
- Mic level for the "listening" swell: have `ears.py` report RMS (it already computes `rms()`) as
  `level v` to the body, throttled to ~20 fps.

### 4. README

Add an "Orb mode (no VTube Studio)" section: how to start it, what the states mean, and that
`vts_body.py` still works for the avatar mode. Keep the existing architecture diagram accurate.

## Constraints

- Local only, no cloud APIs. Keep total RAM for the running app under ~8 GB.
- Don't break avatar mode: `vts_body.py` + VTube Studio must still work exactly as before.
- Python via `uv`, as the launchers already do (Python 3.12). Keep dependencies minimal
  (the stdlib plus what's already used; a tiny WebSocket lib is fine).
- The repo is public, so commit nothing personal: no tokens, logs or recordings. `vts_token.txt` and
  `*.log` are already git-ignored.
- The README notes the project was built with AI under Eric's direction. Keep that note.

## Verify

- Start orb mode, say something, and confirm the orb goes listening → thinking → speaking (pulsing
  in time with the voice) → listening.
- Type mode (`brain_type.py`) works with the orb too.
- Stop actually stops the mic and the model. Start works again after Stop.
- Avatar mode still works with VTube Studio (or at least `vts_body.py` starts and accepts commands).
- Screenshot the orb in each state for Eric.

## Later (not in this build)

This orb is the base for the **proactive voice harness**: the character speaks up on its own
after silences (a jittered timer, a scoreboard that escalates check-in → give space → jab →
self-talk, a grader that treats "yeah" as a brush-off, a `PASS` option, an hourly cap, a "shh"
mode), and reacts to room sounds (Silero VAD + an AudioSet tagger like AST or CED, via audio.cpp).
The orb's states map onto that directly. Add a "heard a sound" flicker state then. Don't build
any of it now; just keep the state system easy to extend.
