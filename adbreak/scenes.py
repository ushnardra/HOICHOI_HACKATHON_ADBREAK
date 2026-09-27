"""Checkpoint 5: shots -> scenes.

A camera cut is a SCENE change only if the story moves on (new place, time or people). For every cut:
  1. Picture link: the most similar pair of shots across the cut, looking 2 shots back and 2 forward.
     A conversation cuts back and forth between faces (1, 2, 1, 2...), so a scene keeps "linking" to itself.
     Two fingerprints per shot: a colour histogram and a SigLIP image embedding (3 keyframes averaged).
  2. Sound change: how different the background sound is 5 s before vs 5 s after the cut
     (rain, room tone, music usually change with the scene).
  3. Black / fade at the cut -> always a scene change.
Each signal is turned into a robust z-score *within the video* (median and spread), so the rule adapts
to each video instead of using fixed numbers tuned on one episode:
  scene_score = mean(-z(colour link), -z(SigLIP link)) + 0.5 * z(sound change)
A cut is a scene change if scene_score >= SCENE_THRESHOLD (or black). The score is kept, so later steps can
prefer the strongest scene changes for ad breaks.

Dialogue across the cut is recorded but NOT used: editors often start the next scene's sound a moment early
("J-cut"), so talking across a cut happened at 41% of real scene changes vs 54% of all cuts.

Usage: python -m adbreak.scenes <video>
"""
import argparse
import json
import time

import cv2
import numpy as np

from adbreak import speech
from adbreak.common import MODELS, ROOT, device, dtype, free_gpu, load_json, save_json, video_path, work_dir

SIGLIP_DIR = MODELS / "siglip-base-patch16-224"
SIGLIP_SIZE = 812672320

LINK_SHOTS = 2            # shots on each side compared for the picture link
SOUND_WINDOW_SEC = 5.0    # seconds of audio compared on each side of the cut
SOUND_WEIGHT = 0.5
SCENE_THRESHOLD = 1.6     # robust z-score; see plan/reports/checkpoint5.html for how it was chosen
STRONG_THRESHOLD = 2.5    # "clear" scene changes (preferred for ad breaks later)


# ---------- shot fingerprints ----------
def colour_features(out, frames):
    feats = []
    for sid in sorted(frames, key=int):
        hs = []
        for f in frames[sid]:
            hsv = cv2.cvtColor(cv2.imread(str(out / f)), cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256]).ravel()
            hs.append(h / (h.sum() + 1e-9))
        v = np.sqrt(np.mean(hs, axis=0))  # cosine of sqrt-histograms = Bhattacharyya coefficient
        feats.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(feats, np.float32)


def siglip_ready():
    w = SIGLIP_DIR / "model.safetensors"
    return w.exists() and w.stat().st_size == SIGLIP_SIZE


def siglip_features(out, frames):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(SIGLIP_DIR)
    model = AutoModel.from_pretrained(SIGLIP_DIR, dtype=dtype()).to(device()).eval()
    feats, ids, per_frame = [], sorted(frames, key=int), []
    with torch.no_grad():
        for k in range(0, len(ids), 32):
            batch = ids[k:k + 32]
            imgs = [Image.open(out / f).convert("RGB") for sid in batch for f in frames[sid]]
            x = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device(), dtype())
            e = model.get_image_features(pixel_values=x)
            if not torch.is_tensor(e):  # newer transformers return an output object
                e = e.pooler_output
            e = (e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy()
            per_frame.append(e)
            n = 0
            for sid in batch:
                m = len(frames[sid])
                v = e[n:n + m].mean(0)
                feats.append(v / np.linalg.norm(v))
                n += m
    del model
    free_gpu()
    # the same per-picture fingerprints are reused by checkpoint 6 (context) - SigLIP runs only once
    np.save(out / "siglip_frame_embeddings.npy", np.concatenate(per_frame).astype(np.float32))
    return np.array(feats, np.float32)


def shot_features(name, kind):
    out = work_dir(video_path(name).name)
    cache = out / f"shot_features_{kind}.npy"
    frames = load_json(out / "keyframes384.json")["frames"]
    if cache.exists() and cache.stat().st_mtime > (out / "keyframes384.json").stat().st_mtime:
        return np.load(cache)
    f = siglip_features(out, frames) if kind == "siglip" else colour_features(out, frames)
    np.save(cache, f)
    return f


# ---------- signals per cut ----------
def picture_links(feats, n_shots):
    sim = feats @ feats.T
    return np.array([max(sim[a, b] for a in range(max(0, i - LINK_SHOTS), i)
                         for b in range(i, min(n_shots, i + LINK_SHOTS))) for i in range(1, n_shots)])


def sound_change(audio, times):
    import torch
    import torchaudio
    mel = torchaudio.transforms.MelSpectrogram(speech.SR, n_fft=1024, hop_length=320, n_mels=64)(torch.from_numpy(audio))
    lm = torch.log(mel + 1e-6).numpy()
    fps = speech.SR / 320

    def window(t0, t1):
        x = lm[:, max(0, int(t0 * fps)):max(int(t0 * fps) + 1, int(t1 * fps))]
        return np.concatenate([x.mean(1), x.std(1)])

    out = []
    for t in times:
        a, b = window(t - SOUND_WINDOW_SEC, t - 0.1), window(t + 0.1, t + SOUND_WINDOW_SEC)
        out.append(1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    return np.array(out)


def robust_z(x):
    return (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25) + 1e-9)


def dialogue_across(t, segments, words):
    """Recorded for information only (see module docstring)."""
    if any(w["start"] < t - 0.1 and w["end"] > t + 0.1 for w in words):
        return True
    return any(s["start"] < t - 0.3 and s["end"] > t + 0.3 for s in segments)


def main(name):
    video = video_path(name)
    out = work_dir(video.name)
    t0 = time.time()
    info = load_json(out / "info.json")
    sh = load_json(out / "shots.json")
    shots, black = sh["shots"], sh["black_segments"]
    sp = speech.load(video.name)
    words = load_json(out / "transcript.json")["words"]
    audio = speech.load_audio(out / "audio.wav")
    times = np.array([s["start"] for s in shots[1:]])

    kinds = ["colour", "siglip"] if siglip_ready() else ["colour"]
    links = {k: picture_links(shot_features(video.name, k), len(shots)) for k in kinds}
    snd = sound_change(audio, times)
    picture = -np.mean([robust_z(links[k]) for k in kinds], axis=0)
    score = picture + SOUND_WEIGHT * robust_z(snd)

    cuts = []
    for i, t in enumerate(times):
        blk = any(b["end"] >= t - 0.6 and b["start"] <= t + 0.6 for b in black)
        cuts.append({"shot": i + 1, "t": round(float(t), 3), "score": round(float(score[i]), 3),
                     "picture_link": {k: round(float(links[k][i]), 3) for k in kinds},
                     "sound_change": round(float(snd[i]), 4), "black": blk,
                     "dialogue_across": dialogue_across(t, sp["segments"], words),
                     "scene_change": bool(blk or score[i] >= SCENE_THRESHOLD),
                     "strong": bool(blk or score[i] >= STRONG_THRESHOLD)})

    bounds = [c for c in cuts if c["scene_change"]]
    starts = [0.0] + [c["t"] for c in bounds]
    ends = starts[1:] + [info["duration_sec"]]
    first = [0] + [c["shot"] for c in bounds]
    last = [s - 1 for s in first[1:]] + [len(shots) - 1]
    scenes = [{"id": k, "start": round(a, 2), "end": round(b, 2), "duration": round(b - a, 2),
               "shots": [first[k], last[k]], "starts_with": None if k == 0 else
               ("black" if bounds[k - 1]["black"] else "strong" if bounds[k - 1]["strong"] else "normal"),
               "boundary_score": None if k == 0 else bounds[k - 1]["score"]}
              for k, (a, b) in enumerate(zip(starts, ends))]
    d = np.array([s["duration"] for s in scenes])
    result = {"video": video.name, "features": kinds + ["sound"],
              "settings": {"link_shots": LINK_SHOTS, "sound_window_sec": SOUND_WINDOW_SEC, "sound_weight": SOUND_WEIGHT,
                           "scene_threshold": SCENE_THRESHOLD, "strong_threshold": STRONG_THRESHOLD},
              "stats": {"scenes": len(scenes), "strong_changes": sum(c["strong"] for c in bounds),
                        "median_scene_sec": round(float(np.median(d)), 1), "shortest_sec": round(float(d.min()), 1),
                        "longest_sec": round(float(d.max()), 1)},
              "scenes": scenes, "cuts": cuts, "seconds": round(time.time() - t0, 1)}
    save_json(out / "scenes.json", result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="video name in assets/ (without .mp4)")
    r = main(ap.parse_args().name)
    print(json.dumps({k: v for k, v in r.items() if k not in ("scenes", "cuts")}, indent=2))
