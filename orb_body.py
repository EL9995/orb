"""
orb_body.py - a voice-assistant orb as the character's body.

It listens on the brain port (47865) for the commands brain.py sends. Instead of moving an avatar it serves
one page (orb/index.html) with an animated orb, and pushes state to it over a WebSocket.

    python3 orb_body.py              then open http://127.0.0.1:47866 (it opens it for you)
    python3 orb_body.py --watch      Start also turns on screen commentary (--video works too)

The page's Start button launches the voice loop (brain_voice.py) as a child process and Stop
ends it (and unloads the LLM). Run one orb_body.py at a time: it owns the port.

Brain port commands (plain text, or JSON like {"cmd": "emotion", "arg": "happy"}):
    mouth 0.7        speaking level, streamed ~30 fps during speech; drives the pulse
    look up          thinking;  look center -> back to listening
    emotion happy    tints the orb (happy warm, annoyed red, sad blue, neutral default)
    energy 0.8       how lively the idle motion is
    talk 3           fake speaking pulse for 3 s;  anim X is accepted and ignored
    state X          off | listening | thinking | speaking | muted   (set the state directly)
    level 0.4        mic input level 0..1, so the orb swells when you talk; replies "muted" if muted
    heard X          what the user said (subtitle + chat log)
    subtitle X       a sentence the character is starting to speak (subtitle + chat log)
    mute on|off      the mute button
    shh on|off       quiet mode: the character won't speak up on its own (the brain sends this)
    chattiness [v]   the fader, 0..1: how soon and how often it speaks up (no value = read it)
    nudge-at T STAGE when the brain plans to speak up next (epoch seconds; 0 = not planned)
    event X          a line for the silence event log in the chat
    type X           a typed message from the page's text box (queued for the brain)
    sleep-after M    minutes with no word from you before it goes to sleep (0 = never; the page)
    settings         the brain reads the fader and sleep setting ("chattiness 0.60 sleep 20")
    sleep on|off|ack it went to sleep / woke up / you chose "keep the mic on"
    ctx USED MAX E   the context gauge: tokens in the last request (E=1 if estimated)
    inbox            the brain picks up the oldest typed message ("inbox: <text>", empty if none)
    start / stop     the Start / Stop button;  clear empties the chat log
    list / status / help / quit

Requires: Python 3.10+ and websockets 13+ (this relaunches itself through uv if missing).
"""

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launcher  # noqa: E402

try:
    from websockets.asyncio.server import serve
    from websockets.datastructures import Headers
    from websockets.http11 import Response
except ImportError:
    launcher.relaunch(["websockets>=13"], ["websockets>=13"])

import brain  # noqa: E402  (settings only: model name, persona file, Ollama URL)

# ---------------------------------------------------------------- settings
COMMAND_HOST, COMMAND_PORT = "127.0.0.1", 47865   # the brain port brain.py talks to
PAGE_PORT = 47866                                  # the orb page + WebSocket
PAGE_FILE = HERE / "orb" / "index.html"
LOG_FILE = HERE / "orb.log"                        # the voice loop's output (git-ignored)
PUSH_HZ = 30
CAPTION_LINGER = 4      # seconds a subtitle stays up after the reply ends
LOG_KEEP = 200          # chat log entries kept (in memory only, never written to disk)
SPEAK_HOLD = 1.2        # seconds after the last mouth value before "speaking" ends (bridges sentence gaps)
EMOTIONS = ["happy", "annoyed", "sad", "surprised", "smug", "tired"]   # offered to the LLM
STATES = ("off", "listening", "thinking", "speaking", "muted", "sleeping")
PASS_FLAGS = [a for a in sys.argv[1:] if a in ("--watch", "--video", "--no-initiative")]
CODE_FILE = Path(__file__).resolve()
CODE_MTIME = CODE_FILE.stat().st_mtime    # if this file changes while running, Start reloads the orb


def code_changed():
    try:
        return CODE_FILE.stat().st_mtime > CODE_MTIME
    except OSError:
        return False


_page_v = {"at": 0.0, "v": 0.0}


def page_version():
    """index.html's modified time (checked at most once a second); the page reloads when it changes."""
    now = time.time()
    if now - _page_v["at"] > 1:
        _page_v["at"] = now
        try:
            _page_v["v"] = PAGE_FILE.stat().st_mtime
        except OSError:
            pass
    return _page_v["v"]


def reload_self(autostart):
    """Re-run this file in place (same window, same process id) so new code takes effect.
    The page reconnects on its own."""
    args = [a for a in sys.argv[1:] if a not in ("--autostart", "--no-open")] + ["--no-open"]
    if autostart:
        args.append("--autostart")
    log("orb_body.py changed since this orb started - reloading it")
    os.execv(sys.executable, [sys.executable, str(CODE_FILE)] + args)


def log(msg):
    print(f"[orb {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def persona_name():
    text = brain.PERSONA_FILE.read_text() if brain.PERSONA_FILE.exists() else ""
    m = re.search(r"You are (\w+)", text)
    return m.group(1) if m else "Nova"


def unload_model(model=None):
    """Tell Ollama to drop the LLM from memory now instead of keeping it warm."""
    url = brain.OLLAMA_URL.replace("/api/chat", "/api/generate")
    data = json.dumps({"model": model or brain.MODEL, "keep_alive": 0}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            url, data, {"Content-Type": "application/json"}), timeout=10).read()
    except Exception:
        pass


SKIP_MODELS = ("embed", "audio", "omni")   # not chat models: hidden from the page's picker


def chat_models():
    """Installed Ollama chat models, smallest first ([] if Ollama isn't up)."""
    url = brain.OLLAMA_URL.replace("/api/chat", "/api/tags")
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            models = json.load(r).get("models", [])
    except Exception:
        return []
    models = [m for m in models if not any(k in m["name"] for k in SKIP_MODELS)]
    return [m["name"] for m in sorted(models, key=lambda m: m.get("size", 0))]


# ---------------------------------------------------------------- the orb's state
class Orb:
    def __init__(self):
        self.name = persona_name()
        self.model = brain.MODEL    # the page's model picker; passed to the voice loop as ORB_MODEL
        self.models = chat_models()
        self.emotion, self.energy = "neutral", 0.5
        self.mouth, self.mouth_at, self.loud_at = 0.0, 0.0, 0.0
        self.level, self.level_at = 0.0, 0.0
        self.talk_until = 0.0
        self.busy = False          # between "thinking" and the end of the reply
        self.forced = None         # a `state` that overrides the computed one (speaking / off)
        self.explicit = False      # this brain sends `state` itself, so `look` is just gaze
        self.ready = False         # the brain said it's listening
        self.muted = False
        self.quiet = False         # "shh" mode: no speaking up unprompted
        self.chattiness = 0.6      # the page's fader; the brain reads it about once a second
        self.nudge_at, self.nudge_stage = 0.0, ""
        self.inbox = []            # typed messages waiting for the brain
        self.sleep_after = 20      # minutes (the page's setting; the brain reads it)
        self.sleeping, self.sleep_ack = False, False
        self.ctx = None            # {"used", "max", "estimated"} from the brain
        self.clients = 0           # brain-port connections
        self.child = None          # the voice loop we started
        self.note = ""             # last status line from the voice loop
        self.stopping = False
        self.caption = None        # {"who": "you"|"char", "text": ...} shown under the orb
        self.caption_until = 0.0   # when it fades (0 = stays while the reply is going)
        self.chat = []             # [{"who", "text", "t"}], newest last
        self.chat_v = 0            # bumps on every change, so the page only gets it when needed
        self.reply_open = False    # the character's current reply is still being added to

    def add_chat(self, who, text):
        if who == "char" and self.reply_open and self.chat and self.chat[-1]["who"] == "char":
            self.chat[-1]["text"] += " " + text        # next sentence of the same reply
        else:
            self.chat.append({"who": who, "text": text, "t": time.strftime("%H:%M")})
            del self.chat[:-LOG_KEEP]
        self.reply_open = who == "char"
        self.chat_v += 1

    def end_reply(self):
        self.reply_open = False
        if self.caption and not self.caption_until:
            self.caption_until = time.time() + CAPTION_LINGER

    # ---- derived state
    def running(self):
        return self.clients > 0 or (self.child is not None and self.child.poll() is None)

    def state(self, now=None):
        now = now or time.time()
        if not self.running() or self.forced == "off":
            return "off"
        if self.forced == "speaking" or now < self.talk_until:
            return "speaking"
        recent = now - self.mouth_at < SPEAK_HOLD
        if recent and (self.busy or now - self.loud_at < SPEAK_HOLD):
            return "speaking"
        if self.busy:
            return "thinking"
        if not self.ready:
            return "starting"
        if self.sleeping:
            return "sleeping"
        return "muted" if self.muted else "listening"

    def snapshot(self):
        now = time.time()
        talking = now < self.talk_until
        mouth = self.mouth if now - self.mouth_at < SPEAK_HOLD else 0.0
        if talking:   # `talk N`: a synthetic syllable pulse (no real audio to follow)
            mouth = 0.35 + 0.35 * abs(((now * 4.5) % 2) - 1)
        caption = self.caption if (self.caption and (not self.caption_until
                                                     or now < self.caption_until)) else None
        return {"type": "state", "state": self.state(now), "mouth": round(mouth, 2),
                "caption": caption,
                "level": round(self.level, 2) if now - self.level_at < 0.5 else 0.0,
                "emotion": self.emotion, "energy": round(self.energy, 2),
                "muted": self.muted, "quiet": self.quiet, "running": self.running(),
                "chattiness": self.chattiness, "nudge_at": self.nudge_at, "nudge_stage": self.nudge_stage,
                "sleep_after": self.sleep_after, "sleeping": self.sleeping, "sleep_ack": self.sleep_ack,
                "ctx": self.ctx, "page_v": page_version(),
                "launched": self.child is not None and self.child.poll() is None,
                "model": self.model, "models": self.models,
                "name": self.name, "note": self.note}

    def reset(self):
        self.busy, self.forced, self.explicit, self.ready = False, None, False, False
        self.mouth, self.level, self.talk_until = 0.0, 0.0, 0.0
        self.emotion, self.energy = "neutral", 0.5
        self.caption, self.caption_until, self.reply_open = None, 0.0, False
        self.quiet = False
        self.nudge_at, self.nudge_stage = 0.0, ""
        self.inbox = []
        self.sleeping, self.sleep_ack, self.ctx = False, False, None

    # ---- the voice loop (Start / Stop)
    def start(self):
        if self.child is not None and self.child.poll() is None:
            return "already running"
        self.reset()
        self.note = "starting..."
        if self.chat and self.chat[-1]["who"] != "sys":
            self.add_chat("sys", "New session")   # the brain starts with no memory of earlier ones
        env = dict(os.environ, PYTHONUNBUFFERED="1", ORB_MODEL=self.model)
        env.pop("NOVA_VIA_UV", None)          # let brain_voice.py set up its own uv environment
        logf = open(LOG_FILE, "a")
        logf.write(f"\n==== start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.child = subprocess.Popen(
            [sys.executable, str(HERE / "brain_voice.py"), *PASS_FLAGS], cwd=HERE, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True)   # own process group: Stop kills it all
        threading.Thread(target=self._read_child, args=(self.child, logf), daemon=True).start()
        log(f"voice loop started (pid {self.child.pid})")
        return "starting"

    def _read_child(self, child, logf):
        for line in child.stdout:
            line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
            if not line:
                continue
            logf.write(line + "\n")
            logf.flush()
            if line.startswith("(listening"):
                continue
            if " is live." in line:
                self.note = ""
            elif not self.ready or line.startswith("("):   # setup progress, or an error/notice
                self.note = line.strip("()")
        code = child.wait()
        logf.write(f"==== exit {code}\n")
        logf.close()
        if child is self.child:
            if not self.stopping and code not in (0, -signal.SIGTERM, -signal.SIGKILL, 143, 137):
                self.note = f"voice loop stopped (exit {code}) - see orb.log"
            self.ready = False
        log(f"voice loop exited ({code})")

    def stop(self):
        """Blocking: end the voice loop (mic, Whisper, Kokoro) and unload the LLM."""
        child = self.child
        if child is None:
            return "not running"
        self.stopping = True
        try:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                except ProcessLookupError:
                    pass
            unload_model(self.model)
        finally:
            self.stopping = False
            self.child = None
        self.reset()
        self.note = ""
        return "stopped"

    # ---- commands
    async def handle(self, line):
        line = line.strip()
        if not line:
            return ""
        if line.startswith("{"):
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                return "bad JSON"
            cmd, arg = str(j.get("cmd", "")).lower(), str(j.get("arg", "")).strip()
        else:
            parts = line.split(maxsplit=1)
            cmd, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
        now = time.time()

        if cmd == "mouth":
            try:
                v = max(0.0, min(1.0, float(arg)))
            except ValueError:
                return "mouth needs 0-1"
            self.mouth, self.mouth_at = v, now
            if v > 0:
                self.loud_at = now
            return "ok"
        if cmd == "level":
            try:
                self.level, self.level_at = max(0.0, min(1.0, float(arg))), now
            except ValueError:
                return "level needs 0-1"
            return "muted" if self.muted else "ok"
        if cmd == "look":
            if not self.explicit:
                if arg.lower() == "up":
                    self.busy = True
                elif arg.lower() == "center":
                    self.busy, self.ready = False, True
                    self.end_reply()
            return f"look: {arg}"
        if cmd == "state":
            s = arg.lower()
            if s not in STATES:
                return f"state needs one of {', '.join(STATES)}"
            self.explicit, self.forced = True, None
            if s == "thinking":
                self.busy, self.ready = True, True
            elif s == "listening":
                self.busy, self.ready = False, True
                self.end_reply()
            elif s == "speaking":
                self.forced = "speaking"
            elif s == "muted":
                self.muted = True
            elif s == "off":
                self.forced = "off"
            return f"state: {self.state()}"
        if cmd in ("heard", "subtitle"):
            if not arg:
                return "needs text"
            who = "you" if cmd == "heard" else "char"
            if who == "you":
                self.reply_open = False
            self.add_chat(who, arg)
            self.caption, self.caption_until = {"who": who, "text": arg}, 0.0
            return "ok"
        if cmd == "clear":
            self.chat, self.reply_open = [], False
            self.chat_v += 1
            return "chat cleared"
        if cmd == "chattiness":
            if arg:
                try:
                    self.chattiness = max(0.0, min(1.0, float(arg)))
                except ValueError:
                    return "chattiness needs 0-1"
            return f"chattiness {self.chattiness:.2f}"
        if cmd == "nudge-at":
            parts = arg.split()
            try:
                self.nudge_at = float(parts[0]) if parts else 0.0
            except ValueError:
                return "nudge-at needs seconds"
            self.nudge_stage = parts[1] if len(parts) > 1 and self.nudge_at else ""
            return "ok"
        if cmd == "type":
            text = " ".join(arg.split())[:500]
            if not text:
                return "needs text"
            if not self.running():
                return "not running"
            self.inbox = (self.inbox + [text])[-5:]
            return "queued"
        if cmd == "sleep-after":
            try:
                self.sleep_after = max(0.0, float(arg))
            except ValueError:
                return "sleep-after needs minutes"
            return f"sleep after {self.sleep_after:g} min"
        if cmd == "models":
            self.models = chat_models() or self.models
            return ", ".join(self.models)
        if cmd == "model":
            if not arg or arg == self.model:
                return self.model
            if self.models and arg not in self.models:
                return f"not installed: {arg}"
            was_running = self.child is not None and self.child.poll() is None
            if was_running:
                await asyncio.to_thread(self.stop)      # unloads the old model
            self.model = arg
            log(f"brain model: {arg}")
            if was_running:
                self.add_chat("sys", f"Switched to {arg}")
                return self.start()
            return f"model {arg}"
        if cmd == "settings":
            return f"chattiness {self.chattiness:.2f} sleep {self.sleep_after:g}"
        if cmd == "sleep":
            a = arg.lower()
            if a == "ack":
                self.sleep_ack = True
            else:
                self.sleeping, self.sleep_ack = a != "off", False
            return f"sleeping: {self.sleeping}"
        if cmd == "ctx":
            try:
                used, mx, est = (arg.split() + ["0"])[:3]
                self.ctx = {"used": int(used), "max": int(mx), "estimated": est == "1"}
            except ValueError:
                return "ctx needs used max"
            return "ok"
        if cmd == "inbox":
            return "inbox: " + (self.inbox.pop(0) if self.inbox else "")
        if cmd == "event":
            if arg:
                self.add_chat("event", arg)
            return "ok"
        if cmd == "shh":
            on = arg.lower() not in ("off", "0", "false", "no")
            if on != self.quiet:
                self.quiet = on
                self.add_chat("sys", "Quiet mode: it won't speak up on its own" if on else "Quiet mode off")
                self.reply_open = False
                if not on:
                    self.note = ""
            return f"quiet: {self.quiet}"
        if cmd == "mute":
            self.muted = arg.lower() not in ("off", "0", "false", "no")
            return f"muted: {self.muted}"
        if cmd == "emotion":
            self.emotion = (arg or "neutral").lower()
            return f"emotion: {self.emotion}"
        if cmd == "energy":
            try:
                self.energy = max(0.0, min(1.0, float(arg)))
            except ValueError:
                return "energy needs a number 0-1"
            return f"energy: {self.energy:.2f}"
        if cmd == "talk":
            try:
                secs = float(arg or 2)
            except ValueError:
                return "talk needs seconds"
            self.talk_until = now + secs if secs > 0 else 0.0
            return f"talking {secs:.1f}s"
        if cmd == "anim":
            return f"anim {arg}: ignored (orb)"
        if cmd == "list":
            return f"expressions: {', '.join(EMOTIONS)}"
        if cmd == "status":
            return (f"orb state={self.state()} emotion={self.emotion} energy={self.energy:.2f} "
                    f"muted={self.muted} running={self.running()}")
        if cmd == "start":
            if code_changed() and not (self.child and self.child.poll() is None):
                # an old orb with a new brain don't understand each other: reload first
                asyncio.get_running_loop().call_later(0.3, reload_self, True)
                self.note = "reloading the orb (its code changed)..."
                return "reloading"
            return self.start()
        if cmd == "stop":
            return await asyncio.to_thread(self.stop)
        if cmd == "help":
            return __doc__.split("Brain port commands")[1].split("Requires:")[0].strip()
        if cmd == "quit":
            await asyncio.to_thread(self.stop)
            log("bye")
            os._exit(0)
        return f"unknown command '{cmd}' - try help"


# ---------------------------------------------------------------- servers
async def brain_port(orb):
    async def client(reader, writer):
        orb.clients += 1
        try:
            while line := await reader.readline():
                reply = await orb.handle(line.decode(errors="ignore"))
                writer.write((reply.replace("\n", " | ") + "\n").encode())
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            orb.clients -= 1
            writer.close()
            if not orb.running():
                orb.reset()

    try:
        server = await asyncio.start_server(client, COMMAND_HOST, COMMAND_PORT)
    except OSError:
        log(f"port {COMMAND_PORT} is busy - another orb_body.py (or vtube-cat's vts_body.py) is running.")
        log("fix: run  pkill -f orb_body.py; pkill -f vts_body.py  then start this again.")
        os._exit(1)
    log(f"brain port listening on {COMMAND_HOST}:{COMMAND_PORT}")
    async with server:
        await server.serve_forever()


async def page_server(orb):
    sockets = set()

    def http(connection, request):
        if request.path.startswith("/ws"):
            return None                                 # let websockets take it
        if request.path in ("/", "/index.html"):
            body, ctype, status = PAGE_FILE.read_bytes(), "text/html; charset=utf-8", 200
        else:
            body, ctype, status = b"not found", "text/plain", 404
        headers = Headers([("Content-Type", ctype), ("Content-Length", str(len(body))),
                           ("Cache-Control", "no-store")])
        return Response(status, "OK" if status == 200 else "Not Found", headers, body)

    def chat_msg():
        return json.dumps({"type": "chat", "chat": orb.chat})

    async def ws(sock):
        sockets.add(sock)
        try:
            await sock.send(json.dumps(orb.snapshot()))
            await sock.send(chat_msg())
            async for msg in sock:
                try:
                    j = json.loads(msg)
                except (json.JSONDecodeError, TypeError):
                    continue
                line = json.dumps({"cmd": j.get("cmd", ""), "arg": j.get("arg", "")})
                await orb.handle(line)
        except Exception:
            pass
        finally:
            sockets.discard(sock)

    async def push():
        last, last_chat = None, 0
        while True:
            snap = json.dumps(orb.snapshot())
            out = [snap] if snap != last else []
            if orb.chat_v != last_chat:
                out.append(chat_msg())
                last_chat = orb.chat_v
            if out and sockets:
                for s in list(sockets):
                    try:
                        for m in out:
                            await s.send(m)
                    except Exception:
                        sockets.discard(s)
            last = snap
            await asyncio.sleep(1 / PUSH_HZ)

    async with serve(ws, "127.0.0.1", PAGE_PORT, process_request=http):
        url = f"http://127.0.0.1:{PAGE_PORT}"
        log(f"orb page at {url}")
        if "--no-open" not in sys.argv:
            webbrowser.open(url)
        await push()


async def main(orb):
    log(f"{orb.name}'s orb. Brain model: {brain.MODEL}. Ctrl+C to quit.")
    if "--autostart" in sys.argv:                  # reloaded because Start was pressed
        asyncio.get_running_loop().call_later(0.5, orb.start)
    await asyncio.gather(brain_port(orb), page_server(orb))


def page_already_up():
    import socket
    try:
        socket.create_connection(("127.0.0.1", PAGE_PORT), timeout=0.5).close()
        return True
    except OSError:
        return False


if __name__ == "__main__":
    if page_already_up():          # double-clicked twice: just show the page again
        log("orb is already running - opening the page")
        webbrowser.open(f"http://127.0.0.1:{PAGE_PORT}")
        sys.exit(0)
    orb = Orb()

    def shutdown(*_):              # Ctrl+C, or the Terminal window closing
        orb.stop()                 # the voice loop has its own process group, so stop it here
        os._exit(0)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, shutdown)
    try:
        asyncio.run(main(orb))
    except KeyboardInterrupt:
        shutdown()
