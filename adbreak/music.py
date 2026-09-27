"""Song / music-sequence timeline for the WHERE step (checkpoint 7).

Why: checkpoints 3/4 look for SPEECH. A gap between two sung lines, or an instrumental passage, looks like
"silence" to them, and a camera cut during a song looks like a scene change - so a naive system puts an ad
break in the middle of a song. A cut during the same song must not create a break opportunity.

How: AST (Audio Spectrogram Transformer, trained on Google AudioSet) scores a 4 s window every second for
the AudioSet music classes in config/music.json and for "Speech".
  song strength  = music - 0.5 x speech        (songs: music high, speech low;  dialogue over background
                                                 music: speech high -> NOT a song)
  smoothed with a 5 s running median; a SONG SEQUENCE = song strength >= 0.30 for at least 6 s.
continuity(t): VETO if t lies inside a song sequence and the music does not clearly pause at t.
A pause = song strength below 0.10 within +/-1 s, OR the loudness (50 ms steps) drops below 20% of the
song's own level for >= 0.4 s within +/-1 s (short pauses are invisible to the 4 s classifier windows).
So a break is only allowed once the music has ended, clearly paused, or before it starts.
(How well it works on the sample episode is documented in plan/reports/checkpoint7.html.)

Usage: python -m adbreak.music <video>
"""
import json
import sys
import time

import numpy as np

from adbreak import speech
from adbreak.common import MODELS, ROOT, device, dtype, free_gpu, load_json, save_json, video_path, work_dir

MODEL_DIR = MODELS / "ast-audioset"
CONFIG = ROOT / "config" / "music.json"


def ready():
    w = MODEL_DIR / "model.safetensors"
    return w.exists() and w.stat().st_size > 340_000_000


class AST:
    def __init__(self):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        self.torch = torch
        self.fe = ASTFeatureExtractor.from_pretrained(MODEL_DIR)
        self.m = ASTForAudioClassification.from_pretrained(MODEL_DIR, dtype=dtype()).to(device()).eval()
        if device() == "cpu":  # 8-bit weights for the linear layers: ~2x faster on a small CPU, same decisions
            self.m = torch.ao.quantization.quantize_dynamic(self.m, {torch.nn.Linear}, dtype=torch.qint8)
        self.labels = self.m.config.id2label

    def run(self, clips, batch=16):
        """clips: list of 16 kHz float arrays -> probabilities [n x 527]."""
        torch = self.torch
        probs = []
        for k in range(0, len(clips), batch):
            x = self.fe(clips[k:k + batch], sampling_rate=16000, return_tensors="pt")["input_values"].to(device(), dtype())
            with torch.no_grad():
                o = self.m(input_values=x)
            probs.append(torch.sigmoid(o.logits).float().cpu().numpy())
        return np.concatenate(probs)

    def close(self):
        del self.m
        free_gpu()


def running_median(x, k):
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(len(x))])


def timeline(audio, cfg, ast=None, centres=None):
    """Per window: music probability, speech probability, song strength (+ smoothed)."""
    own = ast is None
    ast = ast or AST()
    hop, win = cfg["hop_sec"], cfg["window_sec"]
    if centres is None:
        centres = np.arange(0, len(audio) / speech.SR, hop)
    clips = [audio[max(0, int((c - win / 2) * speech.SR)):int((c + win / 2) * speech.SR)] for c in centres]
    probs = ast.run(clips)
    names = {v: int(k) for k, v in ast.labels.items()}
    cols = [names[n] for n in cfg["music_classes"] if n in names]
    mus = probs[:, cols].max(1)
    sp = probs[:, names["Speech"]]
    song = mus - cfg["song"]["speech_weight"] * sp
    smooth = running_median(song, cfg["song"]["smooth_windows"])
    if own:
        ast.close()
    # fine loudness (50 ms) - a short pause in a song is too short for the 4 s classifier windows to notice
    hop = int(0.05 * speech.SR)
    n = len(audio) // hop
    rms = np.sqrt(np.mean(audio[:n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1))
    return {"centres": np.asarray(centres, float), "music": mus, "speech": sp, "song": song, "smooth": smooth,
            "rms": rms.astype(np.float32), "rms_hop": np.float32(0.05)}


def sequences(tl, cfg):
    """Song / music sequences: smoothed song strength >= threshold for at least min_song_sec."""
    th, out, start = cfg["song"]["threshold"], [], None
    c, sm = tl["centres"], tl["smooth"]
    for t, v in zip(c, sm):
        if v >= th and start is None:
            start = t
        elif v < th and start is not None:
            if t - start >= cfg["song"]["min_song_sec"]:
                out.append([float(start), float(t)])
            start = None
    if start is not None and c[-1] - start >= cfg["song"]["min_song_sec"]:
        out.append([float(start), float(c[-1])])
    return out


def energy_pause(t, tl, cfg):
    """True if the sound clearly drops out near t: a stretch of >= pause_min_sec within +/- pause_sec whose
    loudness is below pause_energy x the song's typical loudness (median of the surrounding 6 s)."""
    rms, hop = tl["rms"], float(tl["rms_hop"])
    i = int(t / hop)
    ref = np.median(rms[max(0, i - int(3 / hop)):i + int(3 / hop)]) + 1e-9
    lo, hi = max(0, i - int(cfg["song"]["pause_sec"] / hop)), min(len(rms), i + int(cfg["song"]["pause_sec"] / hop) + 1)
    quiet = rms[lo:hi] < cfg["song"]["pause_energy"] * ref
    need, run = int(cfg["song"]["pause_min_sec"] / hop), 0
    for q in quiet:
        run = run + 1 if q else 0
        if run >= need:
            return True
    return False


def continuity(t, tl, seqs, cfg):
    """(veto, detail): veto if t is inside a song sequence and the music does not clearly pause at t."""
    inside = next((s for s in seqs if s[0] <= t < s[1]), None)
    near = (tl["centres"] >= t - cfg["song"]["pause_sec"]) & (tl["centres"] <= t + cfg["song"]["pause_sec"])
    lowest = float(tl["song"][near].min()) if near.any() else 1.0
    pause = lowest < cfg["song"]["pause_below"] or ("rms" in tl and energy_pause(t, tl, cfg))
    detail = {"inside_song": inside, "song_strength_min_near": round(lowest, 3), "music_pauses_here": bool(pause)}
    return bool(inside is not None and not pause), detail


def load(name):
    out = work_dir(video_path(name).name)
    d = np.load(out / "music.npz")
    tl = {k: d[k] for k in d.files}
    return tl, load_json(out / "music.json")["song_sequences"]


def near_cuts(audio, cfg, cut_times):
    """CPU mode (free hosting): instead of a window every second (1,500+ on a 26-min episode, ~1 hour on
    2 CPU cores) score only two 10 s windows per scene change - just before and just after the cut.
    If the song strength is high on BOTH sides, a song runs through that cut -> a song sequence around it.
    The loudness-pause check (50 ms) still applies, so a clear pause inside the song can still be used."""
    ast = AST()
    w = cfg["cpu_mode"]["side_sec"]
    centres = []
    for c in cut_times:
        centres += [c - w / 2, c + w / 2]
    sub = dict(cfg, window_sec=w)
    tl = timeline(audio, sub, ast=ast, centres=np.array(centres))
    ast.close()
    seqs = []
    for k, c in enumerate(cut_times):
        before, after = tl["song"][2 * k], tl["song"][2 * k + 1]
        if before >= cfg["song"]["threshold"] and after >= cfg["song"]["threshold"]:
            seqs.append([float(c - w), float(c + w)])
    tl["song"] = np.full(len(tl["centres"]), 1.0, np.float32)   # classifier pause unknown at this resolution
    return tl, seqs


def _could_get_a_break(name, cut, _cache={}):
    """True if some moment near this scene change passes the SPEECH rules of checkpoint 7 - only those
    scene changes need the (slow on CPU) music check."""
    from adbreak import breaks  # local import: breaks imports this module
    if name not in _cache:
        out = work_dir(video_path(name).name)
        _cache[name] = (speech.load(name), load_json(out / "transcript.json"), load_json(out / "info.json")["duration_sec"])
    sp, tr, dur = _cache[name]
    t = max(0.0, cut - breaks.BEFORE_SEC)
    while t <= min(dur, cut + breaks.AFTER_SEC):
        if breaks.check_moment(round(t, 2), sp, tr)[0]:
            return True
        t += 0.25
    return False


def main(name):
    video = video_path(name)
    out = work_dir(video.name)
    cfg = load_json(CONFIG)
    t0 = time.time()
    audio = speech.load_audio(out / "audio.wav")
    if device() == "cpu" and (out / "scenes.json").exists():
        cuts = [s["start"] for s in load_json(out / "scenes.json")["scenes"][1:]]
        cuts = [c for c in cuts if _could_get_a_break(name, c)]  # skip cuts that speech already rules out
        tl, seqs = near_cuts(audio, cfg, cuts)
        mode = "cpu: two 10 s windows per scene change"
    else:
        tl = timeline(audio, cfg)
        seqs = sequences(tl, cfg)
        mode = "full: a 4 s window every second"
    np.savez_compressed(out / "music.npz", **tl)
    result = {"video": video.name, "model": MODEL_DIR.name, "mode": mode, "config": cfg,
              "stats": {"song_sequences": len(seqs), "song_seconds": round(sum(b - a for a, b in seqs), 1)},
              "song_sequences": seqs, "seconds": round(time.time() - t0, 1)}
    save_json(out / "music.json", result)
    return result


if __name__ == "__main__":
    r = main(sys.argv[1])
    f = lambda t: f"{int(t // 60)}:{t % 60:04.1f}"
    print(json.dumps(r["stats"]), r["seconds"], "s")
    print("song sequences:", [(f(a), f(b)) for a, b in r["song_sequences"]])
