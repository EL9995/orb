"""
brain.py - Orb's brain.

You talk or type -> a small LLM (via Ollama) streams a reply -> the character starts speaking
the first sentence while the rest is still being written. The mouth level from the real audio
is streamed to the body (orb_body.py, on port 47865), which pulses the orb with it.

Orb.command (orb_body.py) starts this through brain_voice.py when you press Start. To run it
without the page: start python3 orb_body.py --no-open, then python3 brain_type.py to type.

persona.md (or ORB_PERSONA=path) sets who the character is.
"""

import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave
from array import array
from pathlib import Path

# ---------------------------------------------------------------- settings
MODEL = os.environ.get("ORB_MODEL") or "gemma3:1b-it-q8_0"   # Ollama model (`ollama list`); the orb page sets ORB_MODEL
TTS_ENGINE = "kokoro"     # "kokoro" (natural AI voice) or "mac" (built-in `say`, instant setup)
KOKORO_VOICE = "af_bella" # try af_bella, af_nicole, am_michael, bf_emma... (list in voice.py)
VOICE_SPEED = 1.0         # 0.8 = slower, 1.2 = faster
RVC_MODEL = None          # folder with an RVC .pth (+ .index), or None to turn RVC off
RVC_PITCH = -12           # semitones. Female voice -> male RVC: -12. Male -> female: +12. Same: 0
RVC_INDEX_RATE = 0.4      # 0-1: higher = more like the character, lower = clearer words
VOICE = None              # Mac voice for TTS_ENGINE="mac", e.g. "Zoe (Premium)". `say -v ?` lists them
OLLAMA_URL = "http://localhost:11434/api/chat"
BODY = ("127.0.0.1", 47865)
PERSONA_FILE = Path(os.environ.get("ORB_PERSONA") or Path(__file__).with_name("persona.md"))  # ORB_PERSONA=path overrides
WATCH_SCREEN = False      # True = watches your screen and commentates (or run with --watch). Needs eyes.py
COMMENT_COOLDOWN = 12     # min seconds of quiet before commenting on the screen unprompted
WATCH_MODE = "game"       # "game" = roast their gameplay, "video" = heckle whatever they're watching
WATCH_STYLES = {
    "game": ("You're watching the player play a video game, through [screen: ...] notes. "
             "Trash-talk their gameplay: ruthless but playful, in character, one short line. "
             "Don't describe the screen back to them. React to it like you're watching over their shoulder."),
    "video": ("You're watching a video with the user, through [screen: ...] notes. "
              "React like a friend on the couch: hot takes, heckling, jokes about what's on screen, "
              "in character, one short line. Don't summarize the video back to them."),
}
PROACTIVE = True          # speaks up on its own after silences (initiative.py). --no-initiative turns it off
MAX_SENTENCES = 3         # hard cap per reply; extra sentences are dropped
HISTORY_TURNS = 6         # back-and-forths remembered (small models parrot long histories)
NUM_CTX = 4096            # context window. Set it on every call, or Ollama reloads the model at its
                          # full default context (131k for cogito: ~19 GB instead of ~4 GB)
KEEP_WARM = "30m"         # keep the model loaded in memory between messages
NUM_PREDICT = int(os.environ.get("ORB_NUM_PREDICT") or 120)   # reply length cap in tokens; lower = less rambling
TEMPERATURE = os.environ.get("ORB_TEMP")                     # sampling temperature; unset = the model's default
LIPSYNC_FPS = 30
MOUTH_FLOOR = 0.9         # 0-1.2: raise if the mouth hangs open, lower if it barely opens
MOUTH_CONTRAST = 1.4      # 1 = soft, 2 = snappy open/close on each syllable
MOUTH_GAIN = 1.6          # how wide the mouth opens overall
AUDIO_DELAY = 0.04        # seconds the speakers lag behind; raise if mouth leads the voice

TAG_RE = re.compile(r"\[(emotion|look|anim|energy)\s*:\s*([^\]]+)\]", re.I)
LOOSE_TAG_RE = re.compile(                    # misspelled or unbracketed tags: "emtion:tired", "anim:meow"
    r"\[?\b(?:(?:emo|emt|anim)\w*\s*:\s*[a-z_\-]+|look\s*:\s*(?:center|left|right|up|down|chat)\b"
    r"|energy\s*:\s*[0-9.]+)\]?", re.I)
DEBRIS_RE = re.compile(                       # leftovers of broken tags at the start: "0-9 chat", "010"
    # (only 0-led numbers: a real line can start "100 percent" or "2-3 days")
    r"^\s*0[\d\-]*\.?(?:\s+(?:center|left|right|up|down|chat)\b)?\s+(?=\S)", re.I)
MOOD_WORDS = r"neutral|happy|annoyed|sad|surprised|smug|tired|angry|excited|calm"
MOOD_DEBRIS_RE = re.compile(                  # "neutral:0.2 Hmm", "Neutral I've", "1 Annoyed Seriously?"
    rf"(?:^|(?<=[.!?]\s))(?:[01](?:\.\d+)?\s+)?(?i:{MOOD_WORDS})(?::\s*[\d.]+)?\s+(?=[A-Z])"
    rf"|(?i:\b(?:{MOOD_WORDS}):\s*[\d.]+\s*)")
SENTENCE_RE = re.compile(r"^(.*?[.!?]+[\"')\]]*)\s+", re.S)   # a finished sentence + following space
TMP = Path(tempfile.gettempdir())


# ---------------------------------------------------------------- body link
class BodyLink:
    def __init__(self):
        self.lock = threading.Lock()
        try:
            self.sock = socket.create_connection(BODY, timeout=5)
            self.f = self.sock.makefile("rw")
        except OSError:
            self.sock = None

    def send(self, cmd):
        if not self.sock:
            return ""
        with self.lock:   # the speaker thread and main thread both talk to the body
            try:
                self.f.write(cmd + "\n")
                self.f.flush()
                return self.f.readline().strip()
            except OSError:
                self.sock = None
                return ""


def model_options(body):
    """Ask the body which expressions and animations this avatar has."""
    reply = body.send("list")
    emos, anims = [], []
    m = re.search(r"expressions:\s*(.*?)\s*(?:\|\s*hotkeys:\s*(.*))?$", reply)
    if m:
        if m.group(1) and m.group(1) != "(none)":
            emos = [e.strip() for e in m.group(1).split(",") if e.strip()]
        if m.group(2):
            anims = [h.split("[")[0].strip() for h in m.group(2).split(",")
                     if "TriggerAnimation" in h]
    return emos, anims


# ---------------------------------------------------------------- prompt
def split_examples(persona):
    """Pull 'User: ... / Name: ...' example pairs out of the persona text.
    They're fed to the model as real earlier messages instead, so it copies the style
    without continuing a 'User:/Name:' script on its own."""
    keep, examples, pending_user = [], [], None
    for line in persona.splitlines():
        m = re.match(r"^\s*(User|You|Me)\s*:\s*(.+)$", line)
        a = re.match(r"^\s*([A-Z][\w ]{0,20}?)\s*:\s*(.+)$", line)
        if m:
            pending_user = m.group(2).strip()
        elif a and pending_user:
            examples += [{"role": "user", "content": pending_user},
                         {"role": "assistant", "content": a.group(2).strip()}]
            pending_user = None
        elif line.strip().lower().rstrip(":") in ("example exchanges", "examples"):
            continue
        else:
            keep.append(line)
    return "\n".join(keep).strip(), examples


def build_system(emos, anims):
    persona = PERSONA_FILE.read_text().strip() if PERSONA_FILE.exists() else "You are Nova, a voice assistant."
    persona, examples = split_examples(persona)
    lines = [
        persona, "",
        "You control an animated avatar. Begin each reply with 0-3 tags, then your spoken line.",
        "Tags (only use these exact options):",
        f"[emotion:X] X = {', '.join(emos)}, or neutral" if emos else "[emotion:neutral]",
        "[look:X] X = center, left, right, up, down, chat",
        "[energy:N] N = 0.0 (calm) to 1.0 (hyped)",
    ]
    if anims:
        lines.append(f"[anim:X] X = {', '.join(anims)} (use rarely)")
    lines += [
        "Example: [emotion:happy][energy:0.8] Oh nice, that's actually sick.",
        "Never mention the tags. Everything after them is spoken aloud, so no emojis, "
        "asterisks, or stage directions.",
        f"Reply ONLY with your own next line. Never write lines for the user. Max {MAX_SENTENCES} sentences.",
    ]
    return "\n".join(lines), examples


STOP_AT = ["User:", "\nUser", "You:", "\nYou:", "Me:"]   # cut it off if it starts writing your lines


CTX = {"prompt": 0, "reply": 0, "chars_per_token": 3.8, "estimated": True}   # the context gauge


def count_tokens(messages, done=None):
    """Context used by this request. Exact when Ollama finished it (it reports the counts);
    estimated from the text when we cut the reply short (it only reports them at the end)."""
    chars = sum(len(m["content"]) for m in messages)
    if done and done.get("prompt_eval_count"):
        CTX.update(prompt=done["prompt_eval_count"], reply=done.get("eval_count", 0), estimated=False)
        CTX["chars_per_token"] = chars / max(1, done["prompt_eval_count"])
    else:
        CTX.update(prompt=round(chars / CTX["chars_per_token"]), estimated=True)


def ask_stream(messages, stops=()):
    """Yield the reply from Ollama a few words at a time."""
    data = json.dumps({
        "model": MODEL, "messages": messages, "stream": True, "keep_alive": KEEP_WARM,
        "think": False,           # thinking models (qwen3.5) would reason before every line: too slow to talk
        "options": {"num_predict": NUM_PREDICT, "num_ctx": NUM_CTX,
                    "repeat_penalty": 1.15, "repeat_last_n": 256,     # discourage repeating itself
                    "stop": STOP_AT + list(stops),
                    **({"temperature": float(TEMPERATURE)} if TEMPERATURE else {})},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        for line in r:
            if not line.strip():
                continue
            j = json.loads(line)
            if j.get("error"):
                raise RuntimeError(j["error"])
            if j.get("done"):
                count_tokens(messages, j)
            yield j.get("message", {}).get("content", "")
            if j.get("done"):
                break


def unload():
    """Drop the model from memory now (sleep mode); the next request loads it again."""
    data = json.dumps({"model": MODEL, "keep_alive": 0}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            OLLAMA_URL.replace("/api/chat", "/api/generate"), data,
            {"Content-Type": "application/json"}), timeout=10).read()
    except Exception:
        pass


def warm_up():
    """Load the model into memory now, so the first real reply isn't slow."""
    data = json.dumps({"model": MODEL, "messages": [], "keep_alive": KEEP_WARM,
                       "options": {"num_ctx": NUM_CTX}}).encode()
    req = urllib.request.Request(OLLAMA_URL, data, {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=180).read()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- voice + lip sync
SMART_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                              "\u2013": "-", "\u2014": ", "})


def clean_speech(text):
    text = TAG_RE.sub("", text)
    text = LOOSE_TAG_RE.sub("", text)
    text = DEBRIS_RE.sub("", text)
    text = MOOD_DEBRIS_RE.sub("", text)
    text = text.translate(SMART_QUOTES)             # ’ “ ” -> ' " " (or the filter below eats them)
    text = re.sub(r"\*[^*]*\*", "", text)          # *waves*
    text = re.sub(r"[(（][^()（）]*[)）]", "", text)    # (a slow pause) stage directions, never spoken
    text = re.sub(r"[^\w\s.,!?'\":;()\-。，！？、；：]", "", text)  # emojis etc. (keeps Chinese punctuation)
    text = re.sub(r"\s+([,.!?])", r"\1", " ".join(text.split()))   # "salt , it's" -> "salt, it's"
    return text.lstrip("-").strip()


def mouth_envelope(path):
    """Mouth openness (0-1) per animation frame, following syllables rather than raw loudness,
    so dense voices (RVC) still open and close instead of hanging open."""
    with wave.open(path) as w:
        rate, width = w.getframerate(), w.getsampwidth()
        samples = array("h", w.readframes(w.getnframes())) if width == 2 else array("h")
    step = max(1, rate // LIPSYNC_FPS)
    rms = []
    for i in range(0, len(samples), step):
        chunk = samples[i:i + step]
        rms.append((sum(x * x for x in chunk) / len(chunk)) ** 0.5 if chunk else 0.0)
    n = len(rms)
    if not n:
        return []
    ranked = sorted(rms)
    floor, peak = ranked[int(n * 0.2)], ranked[int(n * 0.95)]   # between-word noise / loud speech
    span = (peak - floor) or 1.0
    # local average over ~a quarter second: syllables are the bumps above it
    win = max(1, int(0.25 * LIPSYNC_FPS))
    env, prev = [], 0.0
    for i, r in enumerate(rms):
        local = sum(rms[max(0, i - win):i + win + 1]) / len(rms[max(0, i - win):i + win + 1])
        level = max(0.0, (r - floor) / span)                       # 0 at the noise floor
        bump = max(0.0, (r - local * MOUTH_FLOOR) / span)          # emphasis on syllable peaks
        v = min(1.0, MOUTH_GAIN * (0.4 * level + 0.6 * bump)) ** MOUTH_CONTRAST
        v = 0.0 if v < 0.1 else v                                  # snap shut between syllables
        prev = prev + (v - prev) * (0.75 if v > prev else 0.6)     # open fast, close fast-ish
        env.append(round(prev, 2))
    return env


class Speaker:
    """Two-stage pipeline: one thread renders sentences to audio, another plays them.
    So sentence 2 is being rendered while sentence 1 is being spoken."""

    def __init__(self, body, voice):
        self.body = body
        self.voice = voice
        self.enabled = bool(shutil.which("afplay"))   # macOS audio player
        self.text_q = queue.Queue()
        self.audio_q = queue.Queue()
        self.turn_done = threading.Event()
        self.n = 0
        self.orb = False      # orb_body.py: send each sentence as a subtitle when it starts playing
        threading.Thread(target=self._render_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def say(self, sentence):
        self.text_q.put(sentence)

    def finish_turn(self):
        """Block until everything queued this turn has been spoken."""
        self.turn_done.clear()
        self.text_q.put(None)
        self.turn_done.wait()

    def _voice(self):
        return ["-v", VOICE] if VOICE else []

    def _render_loop(self):
        while True:
            sentence = self.text_q.get()
            if sentence is None or not self.enabled:
                self.audio_q.put(None if sentence is None else (None, None, sentence))
                continue
            self.n += 1
            path = TMP / f"nova_line_{self.n % 8}.wav"   # recycle a few files
            try:
                self.voice.render(sentence, path)
                env = mouth_envelope(str(path))
            except Exception as e:
                print(f"\n(voice error: {e})", flush=True)
                path, env = None, None
            self.audio_q.put((path, env, sentence))

    def _play_loop(self):
        while True:
            item = self.audio_q.get()
            if item is None:
                self.body.send("mouth 0")
                self.turn_done.set()
                continue
            path, env, sentence = item
            if self.orb:
                self.body.send(f"subtitle {sentence}")
            if not self.enabled:
                continue
            if not env:                              # fallback: plain say + flapping mouth
                if not shutil.which("say"):
                    continue
                self.body.send("talk 120")
                subprocess.run(["say"] + self._voice() + [sentence])
                self.body.send("talk 0")
                continue
            player = subprocess.Popen(["afplay", str(path)])
            start = time.time() + AUDIO_DELAY
            while player.poll() is None:
                i = int((time.time() - start) * LIPSYNC_FPS)
                if 0 <= i < len(env):
                    self.body.send(f"mouth {env[i]}")
                time.sleep(1 / LIPSYNC_FPS)


# ---------------------------------------------------------------- one turn
def _norm(sentence):
    return re.sub(r"[^a-z0-9 ]", "", sentence.lower()).strip()


FORGET_NOTICE_EVERY = 10  # after the first notice, mention forgetting again only every 10 of your messages
HISTORY_KEEP = 200        # messages kept in memory at all (the model only ever sees a window of them)


def oldest_real(win):
    """The first thing you actually said that's still in the model's view."""
    return next((m for m in win if m["role"] == "user" and not m.get("nudge")), None)


def window(history, turns=HISTORY_TURNS, keep_nudges=2, raw=False):
    """The slice of history the model sees: the last few exchanges, but in any stretch of
    back-to-back speak-ups only the newest two are kept. A long silence then can't push your
    actual conversation out of view, or fill it with the character's own monologue (which a
    small model starts copying, and spirals)."""
    out, run = [], []

    def flush():
        units = []                                     # a speak-up prompt + what it said
        for m in run:
            if m["role"] == "user" or not units:
                units.append([m])
            else:
                units[-1].append(m)
        for unit in units[-keep_nudges:]:
            out.extend(unit)
        run.clear()

    for m in history:
        if m.get("nudge"):
            run.append(m)
        else:
            flush()
            out.append(m)
    flush()
    out = out[-turns * 2:]
    return out if raw else [{"role": m["role"], "content": m["content"]} for m in out]


def recent_lines(history, turns=3):
    """Sentences the character said in its last few replies (for the anti-repeat filter)."""
    said = set()
    for msg in [m for m in history if m["role"] == "assistant"][-turns:]:
        for sent in re.split(r"(?<=[.!?])\s+", clean_speech(msg["content"])):
            if len(_norm(sent).split()) >= 3:          # short reactions ("Don't.") may recur
                said.add(_norm(sent))
    return said


def respond(body, speaker, messages, recent=frozenset(), name="nova", allow_pass=False, on_start=None):
    """Stream the reply; fire tags immediately and speak each sentence as it lands.
    Sentences repeated from the last few replies are skipped, which breaks catchphrase loops.
    Returns what was actually said (tags + spoken text), to store in history, or None if
    allow_pass is on and the model chose to stay quiet (answered PASS)."""
    stops = [f"{name.title()}:", f"\n{name.title()}"]
    raw, buf, started, passed = "", "", False, False
    spoken, skipped, tags = [], [], []
    t0 = time.time()

    def emit(sentence):
        nonlocal started, passed
        sentence = clean_speech(sentence)
        if not sentence:
            return
        if _norm(sentence) in ("pass", "passing"):    # the PASS token is never spoken
            if allow_pass:
                passed = True                          # (after real lines: just drop it and stop)
            return
        if _norm(sentence) in recent:
            skipped.append(sentence)
            return
        if not started:
            started = True
            if on_start:
                on_start()
            print(f"{name}> ({time.time() - t0:.1f}s) ", end="", flush=True)
        print(sentence + " ", end="", flush=True)
        spoken.append(sentence)
        speaker.say(sentence)

    count_tokens(messages)                             # (replaced by Ollama's exact count if it finishes)
    for chunk in ask_stream(messages, stops):
        if len(spoken) >= MAX_SENTENCES or passed:
            break                                      # enough said (or nothing to say); stop generating
        raw += chunk
        buf += chunk
        for kind, val in TAG_RE.findall(buf):         # act on tags the moment they appear
            body.send(f"{kind.lower()} {val.strip()}")
            tags.append(f"[{kind.lower()}:{val.strip()}]")
        buf = TAG_RE.sub("", buf)
        if "[" in buf and "]" not in buf.split("[")[-1]:
            continue                                   # half-written tag; wait for the rest
        while (m := SENTENCE_RE.match(buf)) and len(spoken) < MAX_SENTENCES:
            emit(m.group(1))
            buf = buf[m.end():]
    if len(spoken) < MAX_SENTENCES and not passed:
        emit(buf)                                      # whatever's left at the end
    if CTX["estimated"]:
        CTX["reply"] = round(len(raw) / CTX["chars_per_token"])
    if passed and not spoken:
        speaker.finish_turn()
        return None
    if not spoken and skipped:                         # everything was a repeat: say it anyway
        recent = frozenset()
        for sentence in skipped:
            emit(sentence)
    print(flush=True)
    speaker.finish_turn()
    return "".join(tags) + " " + " ".join(spoken)


# ---------------------------------------------------------------- main
def required_packages(voice_mode=False):
    """Extra Python packages the current settings need (the launchers install them)."""
    pkgs = {}
    if TTS_ENGINE == "kokoro":
        pkgs.update({"kokoro_onnx": "kokoro-onnx", "numpy": "numpy"})
    if voice_mode:
        pkgs.update({"mlx_whisper": "mlx-whisper", "sounddevice": "sounddevice", "numpy": "numpy"})
    if WATCH_SCREEN or "--watch" in sys.argv or "--video" in sys.argv:
        pkgs.update({"mss": "mss", "PIL": "pillow"})
    return pkgs


def _similar(a, b):
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) > 0.6


def mic_level(rms_value):
    """Mic RMS -> 0..1 for the orb: about -50 dB is 0, -10 dB is 1."""
    if rms_value <= 0:
        return 0.0
    import math
    return round(max(0.0, min(1.0, (20 * math.log10(rms_value) + 50) / 40)), 2)


def main():
    body = BodyLink()
    if not body.sock:
        print("(body not running - Nova can still chat and talk, but the avatar won't move)")
    orb = body.send("status").startswith("orb")    # orb_body.py: also send it explicit states
    from voice import Voice
    voice = Voice(TTS_ENGINE, KOKORO_VOICE, VOICE_SPEED, VOICE,
                  RVC_MODEL, RVC_PITCH, RVC_INDEX_RATE)
    speaker = Speaker(body, voice)
    speaker.orb = orb
    emos, anims = model_options(body)
    system, examples = build_system(emos, anims)
    persona_text = PERSONA_FILE.read_text() if PERSONA_FILE.exists() else ""
    m = re.search(r"You are (\w+)", persona_text)
    name = m.group(1).lower() if m else "nova"
    print(f"warming up {MODEL}...", flush=True)
    if not warm_up():
        print(f"(couldn't reach Ollama with MODEL='{MODEL}'. Is Ollama running? Check `ollama list`.)")
    ears = None
    if "--voice" in sys.argv:
        try:
            from ears import Ears
            ears = Ears()
            print("loading Whisper...", flush=True)
            ears.warm_up()
            print("measuring room noise - stay quiet for a sec...", flush=True)
            ears.calibrate()
        except ImportError:
            print("(voice mode needs: python3 -m pip install mlx-whisper sounddevice numpy)")
            return
        except Exception as e:
            print(f"(couldn't open the mic: {e}. macOS blocks the mic for the app that launched this "
                  "(Terminal for Orb.command): System Settings > Privacy & Security > Microphone > "
                  "turn it on, then quit and reopen that app.)")
            return

    events = queue.Queue()          # everything that can make the character talk: ("user"|"screen"|"quit", ...)
    speaking = threading.Event()    # set while a reply is being generated/spoken
    muted = threading.Event()       # orb mode: the mute button is on, so ignore the mic
    turn = [0]                      # bumps every reply, so the mic can discard what it heard mid-reply

    eyes = None
    if WATCH_SCREEN or "--watch" in sys.argv or "--video" in sys.argv:
        try:
            from eyes import Eyes, VISION_MODEL
            watch_mode = "video" if "--video" in sys.argv else WATCH_MODE
            eyes = Eyes(on_caption=lambda cap, secs: events.put(("screen", cap, secs)), mode=watch_mode)
            print(f"loading vision model {VISION_MODEL}...", flush=True)
            secs = eyes.warm_up()
            print(f"eyes ready ({secs:.1f}s per look).", flush=True)
            system += "\n\n" + WATCH_STYLES.get(watch_mode, WATCH_STYLES["game"])
        except ImportError:
            print("(watch mode needs: mss pillow - use the launchers, they install it)")
            return
        except Exception as e:
            print(f"(eyes off: {e}. Did you `ollama pull` the vision model?)")
            eyes = None

    def listen_loop():
        while True:
            try:
                if ears:
                    while speaking.is_set() or muted.is_set():
                        time.sleep(0.05)
                    time.sleep(0.3)                    # let the room echo of its voice die down
                    ears.flush()                       # drop anything heard while it was talking
                    started = turn[0]
                    print("(listening...)", end="\r", flush=True)
                    text, stt_time = ears.hear()
                    if turn[0] != started or speaking.is_set() or muted.is_set():
                        continue                       # it talked (or got muted) while we listened
                    speaking.set()                     # set it now, not when the main loop picks this up,
                    events.put(("user", text, stt_time))   # or the mic starts recording the reply
                else:
                    events.put(("user", input("you> ").strip(), None))
            except (EOFError, KeyboardInterrupt):
                events.put(("quit",))
                return

    mode = "Just talk" if ears else "Type to chat"
    watching = " Watching your screen." if eyes else ""
    if PROACTIVE and "--no-initiative" not in sys.argv:
        watching += " Speaks up after silences (say 'shh' to stop that)."
    print(f"{name.title()} is live. Expressions: {', '.join(emos) or 'none'}. {mode}.{watching} "
          "Ctrl+C to quit.\n")
    def level_loop():
        """Orb mode: stream the mic level (~20 fps) and pick up the mute button from the reply."""
        while True:
            v = 0.0 if speaking.is_set() or muted.is_set() else mic_level(ears.level)
            reply = body.send(f"level {v}")
            if not body.sock:
                return
            (muted.set if "muted" in reply else muted.clear)()
            time.sleep(0.05)

    threading.Thread(target=listen_loop, daemon=True).start()
    if orb and ears:
        threading.Thread(target=level_loop, daemon=True).start()
    if orb:
        body.send("state listening")
    if eyes:
        eyes.start()

    init = None
    if PROACTIVE and "--no-initiative" not in sys.argv:
        from initiative import Initiative
        init = Initiative(name)

    def event(text):
        """A line for the orb's silence event log."""
        if orb:
            body.send(f"event {text}")

    sync = {"at": 0.0, "sent": None}

    def sync_orb():
        """About once a second: read the orb's fader and sleep setting, tell it when the next nudge is."""
        now = time.time()
        if now - sync["at"] < 1:
            return
        sync["at"] = now
        reply = body.send("settings")
        m = re.search(r"chattiness ([\d.]+)", reply)
        if m and abs(float(m.group(1)) - init.chattiness) > 1e-3:
            init.set_chattiness(float(m.group(1)))
        m = re.search(r"sleep ([\d.]+)", reply)
        if m:
            init.sleep_after_min = float(m.group(1))
        nxt = "0 off" if init.off(now) else f"{init.next_at:.1f} {init.next_stage()}"
        if nxt != sync["sent"]:
            body.send(f"nudge-at {nxt}")
            sync["sent"] = nxt

    if init:
        from initiative import is_end, is_standby
    forgot = {"at": None}          # how many of your messages had dropped out at the last notice
    history = []
    last_spoke = time.time()
    last_commented = ""
    while True:
        try:
            ev = events.get(timeout=0.5)
        except queue.Empty:
            if orb:
                reply = body.send("inbox")
                typed = reply[len("inbox:"):].strip() if reply.startswith("inbox:") else ""
                if typed:                              # from the orb page's text box
                    speaking.set()                     # (same as the mic: stop recording now)
                    events.put(("user", typed, None, "typed"))
                    continue
            if not init:
                continue
            if orb:
                sync_orb()
            if speaking.is_set() or (ears and ears.hearing):
                continue                               # (never cut in while you're mid-sentence)
            if init.should_sleep(time.time()):
                mins = (time.time() - init.last_user) / 60
                print(f"[initiative] going to sleep after {mins:.0f} min with no word from you", flush=True)
                event(f"{mins:.0f} min with no word from you: going to sleep")
                events.put(("sleep", init.sleep_prompt(time.time())))
                continue
            if muted.is_set():
                continue
            nudge = init.due(time.time())
            if not nudge:
                continue
            silent = time.time() - init.last
            label = init.stage.replace("_", " ")
            print(f"[initiative] {init.stage} after {silent:.0f}s of silence", flush=True)
            event(f"{silent:.0f}s of silence: {label}" + ("" if init.offered_pass else " (no PASS)"))
            ev = ("nudge", nudge)
        except KeyboardInterrupt:
            print()
            break
        # if several things piled up while it was talking: you come first, then the newest screen
        pending = [ev]
        while not events.empty():
            pending.append(events.get_nowait())
        if any(e[0] == "quit" for e in pending):
            break
        users = [e for e in pending if e[0] == "user" and e[1]]
        screens = [e for e in pending if e[0] == "screen"]
        nudges = [e for e in pending if e[0] == "nudge"]
        sleeps = [e for e in pending if e[0] == "sleep"]
        is_nudge = sleep_turn = False
        if users:
            for extra in screens:                      # keep the newest screen note for later
                events.put(extra)
            text, stt = users[0][1], users[0][2]
            if stt is not None:
                print(f"you> {text}   ({stt:.1f}s to transcribe)")
            elif len(users[0]) > 3:
                print(f"you> {text}   (typed)")
            if orb:
                body.send(f"heard {text}")
            if init and init.asleep:                   # its question was: end, or standby?
                if is_end(text, name):
                    print("(session ended)", flush=True)
                    event("Session ended")
                    return
                if is_standby(text, name):
                    event("Staying on standby: talk or type to wake it")
                    if orb:
                        body.send("sleep ack")
                    speaking.clear()
                    continue
                init.wake(time.time())                 # anything else: wake up and answer it
                print("(awake)", flush=True)
                event("Woke up (the model reloads, so this reply is a little slower)")
                if orb:
                    body.send("sleep off")
            change = init.on_user(text, time.time()) if init else None
            if init and init.last_grade == "brush-off":
                event(f"\u201c{text[:40]}\u201d = brush-off, next: {init.stage.replace('_', ' ')}")
            if change == "shh":                        # no reply, no check-ins until you talk again
                print("(quiet mode: it won't speak up until you talk to it)", flush=True)
                if orb:
                    body.send("shh on")
                speaking.clear()
                continue
            if change == "shh_off" and orb:
                body.send("shh off")
            screen = eyes.latest() if eyes else ""
            content = f"{text}\n[screen right now: {screen}]" if screen else text
        elif screens:
            cap = screens[-1][1]
            quiet_for = time.time() - last_spoke
            if quiet_for < COMMENT_COOLDOWN or _similar(cap, last_commented):
                continue                               # too soon, or nothing new to roast
            last_commented = cap
            print(f"\n(saw: {cap})", flush=True)
            content = f"[screen: {cap}]\n(The user didn't say anything. React in ONE short line.)"
        elif sleeps:
            is_nudge = sleep_turn = True
            content = sleeps[0][1]
        elif nudges:
            is_nudge = True
            content = nudges[0][1]
        else:
            continue

        history.append({"role": "user", "content": content, "nudge": is_nudge})
        del history[:-HISTORY_KEEP]
        speaking.set()
        turn[0] += 1

        def thinking():
            body.send("look up")                       # "thinking" glance hides model latency
            if orb:
                body.send("state thinking")

        if not is_nudge:
            thinking()                                 # unprompted: only look busy once it talks
        passed = False
        view = window(history)
        if not is_nudge:                               # a fair notice when your words leave its view
            seen = window(history, raw=True)
            first = oldest_real(seen)
            real = [m for m in history if m["role"] == "user" and not m.get("nudge")]
            dropped = sum(1 for m in real if not any(m is s for s in seen))
            if (first is not None and len(seen) >= HISTORY_TURNS * 2 - 1 and seen.index(first) <= 1
                    and (forgot["at"] is None or dropped - forgot["at"] >= FORGET_NOTICE_EVERY)):
                forgot["at"] = dropped
                gist = " ".join(first["content"].split("\n")[0].split()[:8]).rstrip(".!?,")
                view[-1]["content"] += (f"\n[Memory note: what they said earlier (\"{gist}...\") drops out "
                                        "of your view after this reply. If it matters, mention it once, plainly.]")
                event(f"Memory: \u201c{gist}...\u201d leaves the model's view after this reply")
        try:
            raw = respond(body, speaker,
                          [{"role": "system", "content": system}] + examples + view,
                          recent=recent_lines(history[:-1]), name=name,
                          allow_pass=is_nudge and not sleep_turn, on_start=thinking if is_nudge else None)
            if raw is None:                            # it chose to stay quiet
                passed = True
                history.pop()
                print("[initiative] PASS", flush=True)
                event("stayed quiet (PASS)")
            else:
                history.append({"role": "assistant", "content": raw, "nudge": is_nudge})
        except Exception as e:
            print(f"\n(couldn't reach Ollama: {e}. Is it running, and is MODEL='{MODEL}' right?)")
            history.pop()
        finally:
            body.send("look center")
            if orb:
                body.send("state listening")
            last_spoke = time.time()
            if init:
                (init.passed if passed else init.on_spoke)(time.time())
            if orb:
                body.send(f"ctx {CTX['prompt'] + CTX['reply']} {NUM_CTX} {int(CTX['estimated'])}")
            if sleep_turn:
                init.sleep()
                unload()                               # frees the model's memory until you talk
                print("(asleep: model unloaded, mic still on)", flush=True)
                event("Asleep: model unloaded. Talk or type to wake it, or say \u201cend session\u201d")
                if orb:
                    body.send("sleep on")
            speaking.clear()


if __name__ == "__main__":
    main()
