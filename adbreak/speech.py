"""Checkpoint 3: audio.wav -> speech.json (when people talk, when it is silent).

Uses Silero VAD (ONNX version: same results as the PyTorch version, about 2x faster on CPU),
run the standard way: one pass from the start of the audio. Batching was tried and rejected,
see plan/reports/checkpoint3.html.

Also gives the safety check used by later checkpoints:
    speaking_at(t) -> "speech" | "maybe" | "silence"
Only "silence" is safe for an ad break. "maybe" (the model is unsure) counts as NOT safe.

Usage: python -m adbreak.speech <video>
"""
import json
import sys
import time
import wave

import numpy as np

from adbreak.common import load_json, save_json, video_path, work_dir

SR = 16000
CHUNK = 512                 # Silero looks at 32 ms at a time
STEP = CHUNK / SR           # 0.032 s
SPEECH_ON = 0.5             # speech starts when the probability goes above this ...
SPEECH_OFF = 0.35           # ... and ends when it drops below this (Silero's defaults)
MAYBE = 0.2                 # outside speech, a probability above this means "unsure"
MIN_SPEECH_SEC = 0.25       # shorter bursts are ignored (clicks, bangs)
MIN_SILENCE_SEC = 0.10      # shorter pauses inside a sentence do not split it
PAD_SEC = 0.03              # small padding around speech


def load_audio(path):
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1:
            raise ValueError("expected 16 kHz mono wav (run checkpoint 1 first)")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def speech_probabilities(audio):
    import torch
    from silero_vad import load_silero_vad
    model = load_silero_vad(onnx=True)
    return model.audio_forward(torch.from_numpy(audio), SR).squeeze(0).numpy()


class _Replay:
    """Stands in for the Silero model: hands back the probabilities we already computed, one 32 ms chunk
    at a time. This lets us use Silero's own official get_speech_timestamps() without running the model twice."""

    def __init__(self, probs):
        self.probs, self.i = probs, 0

    def reset_states(self):
        self.i = 0

    def __call__(self, chunk, sr):
        import torch
        p = float(self.probs[self.i]) if self.i < len(self.probs) else 0.0
        self.i += 1
        return torch.tensor([[p]])


def to_segments(probs, audio):
    """Speech segments (seconds) using Silero's official post-processing on our probabilities."""
    import torch
    from silero_vad import get_speech_timestamps
    segs = get_speech_timestamps(torch.from_numpy(audio), _Replay(probs), sampling_rate=SR, return_seconds=True,
                                 threshold=SPEECH_ON, neg_threshold=SPEECH_OFF,
                                 min_speech_duration_ms=int(MIN_SPEECH_SEC * 1000),
                                 min_silence_duration_ms=int(MIN_SILENCE_SEC * 1000),
                                 speech_pad_ms=int(PAD_SEC * 1000))
    return [{"start": round(float(x["start"]), 3), "end": round(float(x["end"]), 3)} for x in segs]


def gaps_between(segs, probs, duration):
    """Silence gaps between speech, with the part where the model is sure it is silent."""
    edges = [0.0] + [x for s in segs for x in (s["start"], s["end"])] + [duration]
    gaps = []
    for s, e in zip(edges[0::2], edges[1::2]):
        if e - s <= 0.01:
            continue
        lo, hi = int(s / STEP), max(int(s / STEP) + 1, int(e / STEP))
        seg = probs[lo:hi]
        gaps.append({"start": round(s, 3), "end": round(e, 3), "duration": round(e - s, 3),
                     "max_prob": round(float(seg.max()), 3) if len(seg) else 0.0,
                     "unsure": bool(len(seg) and seg.max() >= MAYBE)})
    return gaps


def speaking_at(t, speech, margin=0.0):
    """Safety check. speech = the dict saved in speech.json.
    'speech' if someone is talking within `margin` seconds of t, 'maybe' if the model is unsure there,
    otherwise 'silence'."""
    for s in speech["segments"]:
        if s["start"] - margin <= t <= s["end"] + margin:
            return "speech"
    probs = speech["_probs"]
    lo, hi = max(0, int((t - margin) / STEP)), int((t + margin) / STEP) + 1
    if len(probs[lo:hi]) and max(probs[lo:hi]) >= MAYBE:
        return "maybe"
    return "silence"


def load(name):
    """Load speech.json plus the probability curve (needed by speaking_at)."""
    out = work_dir(video_path(name).name)
    speech = load_json(out / "speech.json")
    speech["_probs"] = np.load(out / "speech_probs.npy")
    return speech


def main(name: str) -> dict:
    video = video_path(name)
    out = work_dir(video.name)
    info = load_json(out / "info.json")
    duration = info["duration_sec"]

    t0 = time.time()
    audio = load_audio(out / "audio.wav")
    probs = speech_probabilities(audio)
    t_model = time.time() - t0
    segs = to_segments(probs, audio)
    gaps = gaps_between(segs, probs, duration)
    np.save(out / "speech_probs.npy", probs.astype(np.float32))

    speech_sec = sum(s["end"] - s["start"] for s in segs)
    clean = [g for g in gaps if not g["unsure"]]
    result = {
        "video": video.name,
        "method": "Silero VAD 6 (ONNX), one sequential pass, official get_speech_timestamps",
        "settings": {"speech_on": SPEECH_ON, "speech_off": SPEECH_OFF, "maybe": MAYBE,
                     "min_speech_sec": MIN_SPEECH_SEC, "min_silence_sec": MIN_SILENCE_SEC, "pad_sec": PAD_SEC,
                     "step_sec": STEP},
        "stats": {"speech_segments": len(segs), "speech_sec": round(speech_sec, 1),
                  "speech_pct": round(100 * speech_sec / duration, 1),
                  "silence_gaps": len(gaps), "clean_gaps": len(clean),
                  "gaps_over_0_4s": sum(g["duration"] >= 0.4 for g in gaps),
                  "gaps_over_1s": sum(g["duration"] >= 1.0 for g in gaps),
                  "longest_gap_sec": round(max((g["duration"] for g in gaps), default=0), 2)},
        "segments": segs,
        "gaps": gaps,
        "timing": {"model_sec": round(t_model, 1), "total_sec": round(time.time() - t0, 1)},
    }
    save_json(out / "speech.json", result)
    return result


if __name__ == "__main__":
    r = main(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k not in ("segments", "gaps")}, indent=2))
