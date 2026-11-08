"""
seedvc_worker.py - background Seed-VC voice-conversion worker (started by voice.py).

Like rvc_worker.py, but for a fine-tuned Seed-VC model (the 44.1 kHz singing/f0 preset): keeps the
model, Whisper, RMVPE, CAMPPlus and BigVGAN loaded, precomputes the character's reference voice once,
then converts one sentence at a time. Reads JSON jobs on stdin, writes JSON replies on stdout.
Runs in the Seed-VC repo's own Python environment. You never run this directly.

    argv  : SEEDVC_REPO CHECKPOINT CONFIG REFERENCE_WAV
    stdin : {"in": "kokoro.wav", "out": "seedvc.wav", "steps": 10, "pitch": 0, "cfg": 0.7}
    stdout: {"ok": true, "secs": 0.9}   or   {"ok": false, "error": "..."}
"""

import json
import os
import sys
import time
import traceback
import types

REPO, CKPT, CONFIG, REF = (os.path.abspath(os.path.expanduser(a)) for a in sys.argv[1:5])
PROTO = sys.stdout                 # our replies go here...
sys.stdout = sys.stderr            # ...and the engine's chatter goes to stderr
os.chdir(REPO)                     # Seed-VC finds its checkpoints relative to its own folder
sys.path.insert(0, REPO)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def reply(**kw):
    PROTO.write(json.dumps(kw) + "\n")
    PROTO.flush()


try:
    import numpy as np
    import librosa
    import soundfile as sf
    import torch
    import torchaudio
    import inference as vc        # the repo's inference.py: load_models() and the device it picked

    SR = 44100                    # the f0 preset works at 44.1 kHz
    args = types.SimpleNamespace(f0_condition=True, checkpoint=CKPT, config=CONFIG, fp16=True)
    model, semantic_fn, f0_fn, vocoder_fn, campplus, to_mel, _ = vc.load_models(args)
    dev = vc.device

    def resample(x, a, b):        # long sinc kernels don't fit MPS conv1d: do it on the CPU
        return torchaudio.functional.resample(x.cpu(), a, b).to(dev)

    with torch.no_grad():         # the reference voice, once
        ref = torch.tensor(librosa.load(REF, sr=SR)[0][: SR * 25]).unsqueeze(0).float().to(dev)
        ref16 = resample(ref, SR, 16000)
        mel_ref = to_mel(ref)
        fb = torchaudio.compliance.kaldi.fbank(ref16.cpu(), num_mel_bins=80, dither=0,
                                               sample_frequency=16000).to(dev)
        style = campplus((fb - fb.mean(dim=0, keepdim=True)).unsqueeze(0))
        f0_ref = torch.from_numpy(f0_fn(ref16[0], thred=0.03).astype("float32")).to(dev)[None]
        prompt, *_ = model.length_regulator(semantic_fn(ref16), ylens=torch.LongTensor([mel_ref.size(2)]).to(dev),
                                            n_quantizers=3, f0=f0_ref)
    reply(ready=True, device=str(dev))
except Exception as e:
    traceback.print_exc()
    reply(ready=False, error=f"{type(e).__name__}: {e}")
    sys.exit(1)


@torch.no_grad()
def convert(job):
    src = torch.tensor(librosa.load(job["in"], sr=SR)[0]).unsqueeze(0).float().to(dev)
    src16 = resample(src, SR, 16000)
    S = semantic_fn(src16)
    mel = to_mel(src)
    f0 = torch.from_numpy(f0_fn(src16[0], thred=0.03).astype("float32")).to(dev)[None]
    if job.get("pitch"):
        f0[f0 > 1] = f0[f0 > 1] * 2 ** (job["pitch"] / 12)
    cond, *_ = model.length_regulator(S, ylens=torch.LongTensor([mel.size(2)]).to(dev), n_quantizers=3, f0=f0)
    cat = torch.cat([prompt, cond], dim=1)
    with torch.autocast(device_type=dev.type, dtype=torch.float16):
        out = model.cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(dev), mel_ref, style, None,
                                  int(job.get("steps", 10)), inference_cfg_rate=float(job.get("cfg", 0.7)))
    wav = vocoder_fn(out[:, :, mel_ref.size(-1):].float()).squeeze().cpu().numpy()
    sf.write(job["out"], np.clip(wav, -1, 1), SR, subtype="PCM_16")   # 16-bit, as brain.py's lip sync expects
    if dev.type == "mps":
        torch.mps.empty_cache()


for line in sys.stdin:
    try:
        t = time.time()
        convert(json.loads(line))
        reply(ok=True, secs=round(time.time() - t, 2))
    except Exception as e:
        traceback.print_exc()
        reply(ok=False, error=f"{type(e).__name__}: {e}")
