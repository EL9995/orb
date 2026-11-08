"""
voice.py - text-to-speech engines for Nova. Each one turns a sentence into a .wav file;
brain.py plays it and drives the mouth from its loudness.

  "kokoro" - local neural voice, sounds close to human. 54 voices. Needs Python 3.12
             via uv (the launchers handle it). Downloads ~350 MB of model files on first use.
  "mac"    - the built-in macOS `say` voice. Instant, zero setup, sounds like a computer.

Kokoro voices worth trying (American): af_heart (best), af_bella, af_nicole, af_sarah,
am_michael, am_fenrir, am_puck.   British: bf_emma, bm_george, bm_fable.
Full list: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

Optional RVC voice conversion: after Kokoro speaks a sentence, an RVC model re-skins it
into a character's voice (see brain.py RVC_* settings). Runs Applio's engine in a
background worker (rvc_worker.py) with its own Python 3.12 environment, set up on first use.

Optional Seed-VC voice conversion, per persona: if the persona file (ORB_PERSONA) has a
<name>.voice.json beside it with a "seedvc" block, a fine-tuned Seed-VC model re-skins every
sentence instead (seedvc_worker.py, run in the Seed-VC repo's own Python). The same file can
set "kokoro_voice" for the voice Kokoro speaks in before conversion. Example:
    {"kokoro_voice": "af_nicole",
     "seedvc": {"python": "~/.venvs/seedvc/bin/python", "repo": "~/seed-vc",
                "checkpoint": "~/seed-vc/runs/me.pth", "config": "~/seed-vc/runs/me.yml",
                "reference": "~/voice/me_ref.wav", "steps": 10, "pitch": 0, "enabled": true}}

Or the persona speaks in its own cloned voice with a "tts" block: a voice-clone TTS model (Qwen3-TTS)
served by audio.cpp (audiocpp_server, started here if not already running on the port):
    {"tts": {"family": "qwen3_tts", "server": "~/audiocpp/audiocpp_server",
             "model": "~/audiocpp/models/Qwen3-TTS-12Hz-1.7B-Base-GGUF/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf",
             "reference": "~/voice/me.wav", "reference_text_file": "~/voice/me.txt", "seed": 7}}
"""

import atexit
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import wave
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KOKORO_DIR = HERE / "kokoro_models"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
KOKORO_FILES = ["kokoro-v1.0.onnx", "voices-v1.0.bin"]


def _download(name):
    dest = KOKORO_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    KOKORO_DIR.mkdir(exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"downloading {name} (one time)...", flush=True)
    with urllib.request.urlopen(KOKORO_URL + name, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"  {done * 100 // total}%", end="\r", flush=True)
    tmp.rename(dest)
    print(f"  {name} done.     ", flush=True)
    return dest


# ---------------------------------------------------------------- RVC voice conversion
APPLIO_COMMIT = "67a38f6abab3e90a55efa534ea3185a222c102f0"   # pinned, known-good Applio version
APPLIO_DIR = HERE / "applio"
RVC_REQUIREMENTS = HERE / "rvc_requirements.txt"


def _find_uv():
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if not Path(uv).exists():
        raise RuntimeError("RVC needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
    return uv


def _ensure_applio():
    if (APPLIO_DIR / "rvc" / "infer" / "infer.py").exists():
        return
    print("downloading Applio voice engine (one time)...", flush=True)
    url = f"https://github.com/IAHispano/Applio/archive/{APPLIO_COMMIT}.zip"
    zpath = HERE / "applio_download.zip"
    with urllib.request.urlopen(url, timeout=120) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(HERE)
    (HERE / f"Applio-{APPLIO_COMMIT}").rename(APPLIO_DIR)
    zpath.unlink()


class RVC:
    """Talks to rvc_worker.py, which keeps the voice model loaded between sentences."""

    def __init__(self, model_dir, pitch=0, index_rate=0.4, protect=0.33):
        folder = (HERE / model_dir) if not Path(model_dir).is_absolute() else Path(model_dir)
        pths = sorted(folder.glob("*.pth"))
        if not pths:
            raise FileNotFoundError(f"no .pth voice file in {folder}")
        idx = sorted(folder.glob("*.index"))
        self.job = {"model": str(pths[0]), "index": str(idx[0]) if idx else "",
                    "pitch": pitch, "index_rate": index_rate if idx else 0, "protect": protect}
        _ensure_applio()
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        cmd = [_find_uv(), "run", "--no-project", "--python", "3.12",
               "--with-requirements", str(RVC_REQUIREMENTS),
               "python", str(HERE / "rvc_worker.py"), str(APPLIO_DIR)]
        print(f"starting RVC voice '{folder.name}' (first time installs ~1-2 GB, be patient)...",
              flush=True)
        self.proc = subprocess.Popen(cmd, cwd=HERE, env=env, text=True, bufsize=1,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self.ready = threading.Event()
        self.log = open(HERE / "rvc_worker.log", "w")
        threading.Thread(target=self._pipe_stderr, daemon=True).start()
        hello = json.loads(self.proc.stdout.readline() or '{"ready": false, "error": "worker exited"}')
        self.ready.set()
        if not hello.get("ready"):
            raise RuntimeError(f"RVC worker failed: {hello.get('error')} (see rvc_worker.log)")
        print(f"RVC ready on {hello.get('device')}.", flush=True)
        self.lock = threading.Lock()
        self.last_secs = 0.0

    def _pipe_stderr(self):
        """Show setup progress in the terminal; after that, send the engine's chatter to a log."""
        for line in self.proc.stderr:
            self.log.write(line)
            self.log.flush()
            if not self.ready.is_set() and any(k in line for k in
                                               ("ownload", "%", "nstall", "repar", "Resolved")):
                print("  " + line.rstrip()[:110], flush=True)

    def convert(self, src, dst):
        with self.lock:
            self.proc.stdin.write(json.dumps({**self.job, "in": str(src), "out": str(dst)}) + "\n")
            self.proc.stdin.flush()
            res = json.loads(self.proc.stdout.readline() or '{"ok": false, "error": "worker died"}')
        if not res.get("ok"):
            raise RuntimeError(res.get("error"))
        self.last_secs = res.get("secs", 0.0)


# ---------------------------------------------------------------- Seed-VC voice conversion
def persona_voice():
    """The persona's <name>.voice.json settings ({} if none), found beside ORB_PERSONA."""
    persona = os.environ.get("ORB_PERSONA")
    f = Path(persona).with_suffix(".voice.json") if persona else None
    if f and not f.is_absolute():
        f = HERE / f
    return json.loads(f.read_text()) if f and f.exists() else {}


class SeedVC:
    """Talks to seedvc_worker.py, which keeps a fine-tuned Seed-VC model loaded between sentences.
    Same convert(src, dst) interface as RVC."""

    def __init__(self, cfg):
        p = {k: str(Path(os.path.expanduser(str(v)))) for k, v in cfg.items()
             if k in ("python", "repo", "checkpoint", "config", "reference")}
        for k in ("python", "checkpoint", "config", "reference"):
            if not Path(p[k]).exists():
                raise FileNotFoundError(f"Seed-VC {k} not found: {p[k]}")
        self.job = {"steps": int(cfg.get("steps", 10)), "pitch": float(cfg.get("pitch", 0)),
                    "cfg": float(cfg.get("cfg", 0.7))}
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        cmd = [p["python"], str(HERE / "seedvc_worker.py"), p["repo"], p["checkpoint"], p["config"], p["reference"]]
        print(f"starting Seed-VC voice '{Path(p['checkpoint']).stem}' (loads ~4 GB, ~30 s)...", flush=True)
        self.proc = subprocess.Popen(cmd, cwd=HERE, env=env, text=True, bufsize=1,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.log = open(HERE / "seedvc_worker.log", "w")
        threading.Thread(target=self._pipe_stderr, daemon=True).start()
        hello = json.loads(self.proc.stdout.readline() or '{"ready": false, "error": "worker exited"}')
        if not hello.get("ready"):
            raise RuntimeError(f"Seed-VC worker failed: {hello.get('error')} (see seedvc_worker.log)")
        print(f"Seed-VC ready on {hello.get('device')}.", flush=True)
        self.lock = threading.Lock()
        self.last_secs = 0.0

    def _pipe_stderr(self):
        for line in self.proc.stderr:
            self.log.write(line)
            self.log.flush()

    def convert(self, src, dst):
        with self.lock:
            self.proc.stdin.write(json.dumps({**self.job, "in": str(src), "out": str(dst)}) + "\n")
            self.proc.stdin.flush()
            res = json.loads(self.proc.stdout.readline() or '{"ok": false, "error": "worker died"}')
        if not res.get("ok"):
            raise RuntimeError(res.get("error"))
        self.last_secs = res.get("secs", 0.0)


# ---------------------------------------------------------------- audio.cpp voice-clone TTS
class AudiocppTTS:
    """A voice-clone TTS (e.g. Qwen3-TTS) served by audio.cpp's audiocpp_server, which keeps the model
    loaded. Speaks in the character's own voice from a reference clip, so no conversion step.
    Starts the server if nothing answers on the port, and stops it when Nova exits."""

    def __init__(self, cfg):
        x = lambda k: str(Path(os.path.expanduser(str(cfg[k]))))
        self.url = f"http://127.0.0.1:{int(cfg.get('port', 8790))}"
        self.ref = x("reference")
        text = cfg.get("reference_text") or Path(x("reference_text_file")).read_text()
        self.job = {"model": "tts", "voice_ref": self.ref, "reference_text": text.strip(),
                    **({"seed": int(cfg["seed"])} if "seed" in cfg else {})}
        self.proc = None
        if not self._up():
            conf = Path(tempfile.gettempdir()) / "nova_audiocpp_tts.json"
            conf.write_text(json.dumps({"host": "127.0.0.1", "port": int(cfg.get("port", 8790)),
                                        "threads": int(cfg.get("threads", 8)), "models": [
                {"id": "tts", "family": cfg.get("family", "qwen3_tts"), "task": "tts",
                 "mode": "offline", "path": x("model")}]}))
            print(f"starting {cfg.get('family', 'qwen3_tts')} voice server (loads ~2.5 GB)...", flush=True)
            self.log = open(HERE / "audiocpp_tts.log", "w")
            self.proc = subprocess.Popen([x("server"), "--config", str(conf), "--backend",
                                          cfg.get("backend", "metal"), "--no-ui"],
                                         stdout=self.log, stderr=subprocess.STDOUT)
            atexit.register(self.stop)
            for _ in range(120):
                if self._up() or self.proc.poll() is not None:
                    break
                time.sleep(0.5)
            if not self._up():
                self.stop()
                raise RuntimeError("voice server didn't start (see audiocpp_tts.log)")
        else:
            self.job["model"] = self._model_id()
        print("voice server ready.", flush=True)

    def _up(self):
        try:
            with urllib.request.urlopen(self.url + "/v1/models", timeout=2) as r:
                return any(m.get("loaded") for m in json.load(r).get("data", []))
        except Exception:
            return False

    def _model_id(self):                             # a server we didn't start: use its first TTS model
        with urllib.request.urlopen(self.url + "/v1/models", timeout=2) as r:
            return next(m["id"] for m in json.load(r)["data"] if m.get("task") == "tts")

    def speak(self, text, path):
        req = urllib.request.Request(self.url + "/v1/audio/speech", json.dumps({**self.job, "input": text}).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:   # a 16-bit mono WAV
            Path(path).write_bytes(r.read())

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


class Voice:
    def __init__(self, engine="kokoro", kokoro_voice="af_heart", speed=1.0, mac_voice=None,
                 rvc_model=None, rvc_pitch=0, rvc_index_rate=0.4):
        self.rvc = None                              # the converter: RVC or Seed-VC, same interface
        pv = persona_voice()
        kokoro_voice = pv.get("kokoro_voice", kokoro_voice)
        self.tts = None
        if pv.get("tts") and pv["tts"].get("enabled", True):
            try:
                self.tts = AudiocppTTS(pv["tts"])    # the persona speaks in its own cloned voice
                engine, rvc_model = "audiocpp", None
            except Exception as e:                   # fall back to Kokoro rather than go silent
                print(f"(cloned voice off: {e} - using Kokoro)", flush=True)
        if pv.get("seedvc") and pv["seedvc"].get("enabled", True):
            try:
                self.rvc = SeedVC(pv["seedvc"])
                rvc_model = None                     # the persona's own voice wins over brain.py's RVC
            except Exception as e:                   # don't let a bad voice model kill the character
                print(f"(Seed-VC off: {e})", flush=True)
        if rvc_model:
            try:
                self.rvc = RVC(rvc_model, rvc_pitch, rvc_index_rate)
            except Exception as e:                   # don't let a bad voice model kill Nova
                print(f"(RVC off: {e})", flush=True)
        self.engine = engine
        self.kokoro_voice = kokoro_voice
        self.speed = speed
        self.mac_voice = mac_voice
        self.kokoro = None
        if engine == "kokoro":
            from kokoro_onnx import Kokoro            # only needed for this engine
            model, voices = (_download(f) for f in KOKORO_FILES)
            print(f"loading Kokoro voice '{kokoro_voice}'...", flush=True)
            self.kokoro = Kokoro(str(model), str(voices))
            if kokoro_voice not in self.kokoro.get_voices():
                raise ValueError(f"unknown Kokoro voice '{kokoro_voice}'. Options: "
                                 + ", ".join(self.kokoro.get_voices()))
            self.render("Ready.", Path(tempfile.gettempdir()) / "nova_warmup.wav")   # first call is slow; do it now

    def render(self, text, path):
        """Write `text` as speech to `path` (.wav, 16-bit mono), through RVC if enabled."""
        if not self.rvc:
            return self._speak(text, path)
        raw = Path(path).with_name(Path(path).stem + "_raw.wav")
        self._speak(text, raw)
        try:
            self.rvc.convert(raw, path)
        except Exception as e:                       # never go silent: fall back to plain voice
            print(f"\n(voice conversion failed: {e} - using plain voice)", flush=True)
            shutil.copyfile(raw, path)
        return True

    def _speak(self, text, path):
        if self.engine == "audiocpp":
            self.tts.speak(text, path)
            return True
        if self.engine == "kokoro":
            import numpy as np
            lang = "en-gb" if self.kokoro_voice.startswith("b") else "en-us"
            samples, rate = self.kokoro.create(text, voice=self.kokoro_voice,
                                               speed=self.speed, lang=lang)
            pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(pcm)
            return True
        cmd = ["say"] + (["-v", self.mac_voice] if self.mac_voice else []) + \
              ["-o", str(path), "--data-format=LEI16@22050", text]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
