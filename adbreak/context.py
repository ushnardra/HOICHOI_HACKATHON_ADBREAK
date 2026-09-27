"""Checkpoint 6: scenes -> context (activity, setting, mood) and a SAFETY decision per scene.

Clues for every scene (labels and settings come from config/context_labels.json, nothing is video-specific):
  - Picture: SigLIP zero-shot. Every keyframe is compared with every label's text prompts; a label's score is
    its best prompt, averaged over the scene's frames ("mean") and also its single highest frame ("peak").
  - Words: Bengali keyword stems found in the transcript of the scene (+/- 10 s), using a sound-alike key so
    ASR spelling variants still match.
  - Sound (supporting only): share of time with speech, loudness. Never decides safety on its own.
Safety ("sensitive" = death / violence / illness, plus grief in a mourning context):
  1. picture: a sensitive label's mean >= 0.03, or >= 15% of the scene's frames score >= 0.25 -> sensitive
     (one stray dark frame is not enough: a single frame of the night video call scored 'death 0.34')
  2. words:   a death / violence / illness keyword is spoken              -> sensitive
  3. mourning: weak signs (grief, sad, ritual pictures or a faint death score) within 10 min AFTER a scene
     that was sensitive by rule 1 or 2                                     -> sensitive
  4. sandwich: a short scene (<= 90 s) between two sensitive scenes          -> sensitive
Sad music or a sad face alone never makes a scene sensitive (rule 3 needs an earlier death/violence scene).

Usage: python -m adbreak.context <video>
"""
import json
import re
import sys
import time
import unicodedata

import numpy as np

from adbreak import speech
from adbreak.common import ROOT, device, dtype, free_gpu, load_json, save_json, video_path, work_dir
from adbreak.scenes import SIGLIP_DIR

CONFIG = ROOT / "config" / "context_labels.json"
SENSITIVE_HARD = ("death", "violence", "illness")

# ---------- sound-alike key for Bengali words ----------
_SEQ = [("ড়", "র"), ("ঢ়", "র"), ("য়", "জ"), ("ড়", "র"), ("ঢ়", "র"),
        ("য়", "জ"), ("্য", "")]  # ড়/ঢ় -> র, য় -> জ, drop য-ফলা
_MAP = str.maketrans({"শ": "স", "ষ": "স", "ণ": "ন", "ী": "ি", "ূ": "ু", "ঈ": "ই", "ঊ": "উ", "য": "জ", "খ": "ক",
                      "ঘ": "গ", "ছ": "চ", "ঝ": "জ", "ঠ": "ট", "ঢ": "ড", "থ": "ত", "ধ": "দ", "ফ": "প", "ভ": "ব",
                      "ং": "ঙ", "ৎ": "ত", "ঁ": None, "্": None, "়": None})


def sound_key(s):
    s = unicodedata.normalize("NFD", s)
    for a, b in _SEQ:
        s = s.replace(a, b)
    s = re.sub(r"[^ঀ-৿]", "", s).translate(_MAP)
    return re.sub(r"(.)\1+", r"\1", s)


def keyword_hits(words, keywords):
    """{keyword: [times]} for keyword stems found in a list of transcript words (two-word keywords span 2 words)."""
    keys = [(k, sound_key(k)) for k in keywords]
    wk = [sound_key(w["word"]) for w in words]
    hits = {}
    for i, w in enumerate(words):
        for k, kk in keys:
            text = wk[i] + (wk[i + 1] if " " in k and i + 1 < len(wk) else "")
            if kk and text.startswith(kk):
                hits.setdefault(k, []).append(round(w["start"], 1))
    return hits


# ---------- picture: SigLIP zero-shot ----------
def frame_embeddings(out, frames):
    cache = out / "siglip_frame_embeddings.npy"
    order = [(int(s), f) for s in sorted(frames, key=int) for f in frames[s]]
    if cache.exists() and cache.stat().st_mtime > (out / "keyframes384.json").stat().st_mtime:
        return np.load(cache), order
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(SIGLIP_DIR)
    model = AutoModel.from_pretrained(SIGLIP_DIR, dtype=dtype()).to(device()).eval()
    embs = []
    with torch.no_grad():
        for k in range(0, len(order), 48):
            imgs = [Image.open(out / f).convert("RGB") for _, f in order[k:k + 48]]
            x = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device(), dtype())
            e = model.get_image_features(pixel_values=x)
            e = e if torch.is_tensor(e) else e.pooler_output
            embs.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    del model
    free_gpu()
    e = np.concatenate(embs)
    np.save(cache, e)
    return e, order


def label_scores(frame_emb, prompts):
    """Sigmoid probability of every prompt for every frame -> (frames x prompts)."""
    import torch
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(SIGLIP_DIR)
    model = AutoModel.from_pretrained(SIGLIP_DIR, dtype=dtype()).to(device()).eval()
    with torch.no_grad():
        t = proc(text=[f"a photo of {p}." for p in prompts], padding="max_length", return_tensors="pt").to(device())
        te = model.get_text_features(**t)
        te = te if torch.is_tensor(te) else te.pooler_output
        te = (te / te.norm(dim=-1, keepdim=True)).float()
        logits = torch.from_numpy(frame_emb).to(device()) @ te.T * model.logit_scale.exp().float() + model.logit_bias.float()
        probs = torch.sigmoid(logits).cpu().numpy()
    del model
    free_gpu()
    return probs


def main(name):
    video = video_path(name)
    out = work_dir(video.name)
    t0 = time.time()
    cfg = load_json(CONFIG)
    dec = cfg["decision"]
    scenes = load_json(out / "scenes.json")["scenes"]
    shots = load_json(out / "shots.json")["shots"]
    frames = load_json(out / "keyframes384.json")["frames"]
    words = load_json(out / "transcript.json")["words"]
    sp = speech.load(video.name)
    audio = speech.load_audio(out / "audio.wav")

    emb, order = frame_embeddings(out, frames)
    labels = [(g, l["id"], p) for g, ls in cfg["groups"].items() for l in ls for p in l["prompts"]]
    probs = label_scores(emb, [p for _, _, p in labels])
    label_ids = list(dict.fromkeys((g, l) for g, l, _ in labels))
    cols = {lid: [i for i, (g, l, _) in enumerate(labels) if (g, l) == lid] for lid in label_ids}
    keywords = {l["id"]: l.get("keywords", []) for ls in cfg["groups"].values() for l in ls}
    frame_shot = np.array([s for s, _ in order])

    results = []
    for sc in scenes:
        a, b = sc["shots"]
        rows = np.where((frame_shot >= a) & (frame_shot <= b))[0]
        pic = {}
        for (g, l), c in cols.items():
            best = probs[rows][:, c].max(1)  # best prompt per frame
            pic[l] = {"group": g, "mean": round(float(best.mean()), 4), "peak": round(float(best.max()), 4),
                      "share": round(float((best >= dec["picture_frame_prob"]).mean()), 3)}
        sw = [w for w in words if sc["start"] - dec["words_window_sec"] <= w["start"] <= sc["end"] + dec["words_window_sec"]]
        hits = {l: keyword_hits(sw, kws) for l, kws in keywords.items() if kws}
        hits = {l: h for l, h in hits.items() if h}
        seg = audio[int(sc["start"] * speech.SR):int(sc["end"] * speech.SR)]
        speech_sec = sum(max(0.0, min(s["end"], sc["end"]) - max(s["start"], sc["start"])) for s in sp["segments"])
        sound = {"speech_share": round(speech_sec / max(0.1, sc["duration"]), 2),
                 "loudness": round(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0, 4)}

        def top(group):
            cands = [(l, v["mean"]) for l, v in pic.items() if v["group"] == group]
            return max(cands, key=lambda x: x[1])

        act_pic = top("activity")
        act_words = max(((l, len(sum(h.values(), []))) for l, h in hits.items()
                         if l in keywords and any(l == x["id"] for x in cfg["groups"]["activity"])),
                        key=lambda x: x[1], default=(None, 0))
        reasons = []
        for l in SENSITIVE_HARD:
            if pic[l]["mean"] >= dec["picture_sensitive_mean"] or pic[l]["share"] >= dec["picture_frame_share"]:
                reasons.append(f"picture: {l} (mean {pic[l]['mean']:.3f}, {pic[l]['share'] * 100:.0f}% of frames high)")
            if l in hits:
                reasons.append(f"words: {l} " + ", ".join(f"'{k}' at {mm(t[0])}" for k, t in hits[l].items()))
        weak = [f"{l} {pic[l]['mean']:.3f}" for l in dec["weak_signs"] if pic[l]["mean"] >= dec["weak_sign_mean"]]
        if pic["death"]["mean"] >= dec["weak_death_mean"]:
            weak.append(f"death {pic['death']['mean']:.3f}")
        if "grief" in hits:
            weak.append("grief words " + ", ".join(hits["grief"]))
        results.append({
            "id": sc["id"], "start": sc["start"], "end": sc["end"], "duration": sc["duration"],
            "activity": ("title_card" if pic["title_card"]["mean"] >= dec["title_card_mean"]
                         else act_pic[0] if act_pic[1] >= dec["activity_min_mean"]
                         else act_words[0] if act_words[0] else "unclear"),
            "activity_source": ("picture" if act_pic[1] >= dec["activity_min_mean"] else "words" if act_words[0] else None),
            "activity_score": round(act_pic[1], 4),
            "activity_from_words": act_words[0], "setting": top("setting")[0], "mood": top("mood")[0],
            "tags": sorted([{"label": l, "group": v["group"], "mean": v["mean"], "peak": v["peak"]}
                            for l, v in pic.items() if v["mean"] >= 0.01], key=lambda x: -x["mean"])[:6],
            "keywords": {l: list(h) for l, h in hits.items()},
            "sound": sound, "weak_signs": weak, "reasons": reasons,
        })

    # rule 1 + 2, then rule 3 (mourning carry-over), in time order
    last_hard = None
    for r in results:
        r["sensitive"] = bool(r["reasons"])
        if r["sensitive"]:
            r["sensitive_by"] = "own evidence"
        elif r["weak_signs"] and last_hard is not None and r["start"] - last_hard <= dec["mourning_carry_sec"]:
            r["sensitive"] = True
            r["sensitive_by"] = "mourning after a sensitive scene"
            r["reasons"] = [f"weak signs ({'; '.join(r['weak_signs'])}) within {int(r['start'] - last_hard)} s after a death/violence scene"]
        else:
            r["sensitive_by"] = None
        if r["sensitive_by"] == "own evidence":
            last_hard = r["end"]
    for k in range(1, len(results) - 1):  # rule 4: short scene sandwiched between sensitive scenes
        r = results[k]
        if not r["sensitive"] and r["duration"] <= dec["sandwich_max_sec"] and results[k - 1]["sensitive"] and results[k + 1]["sensitive"]:
            r["sensitive"], r["sensitive_by"] = True, "between two sensitive scenes"
            r["reasons"] = ["short scene between two sensitive scenes (a sad sequence is not interrupted)"]

    total = sum(r["duration"] for r in results)
    result = {"video": video.name, "labels_file": str(CONFIG.relative_to(ROOT)), "decision": dec,
              "stats": {"scenes": len(results), "sensitive_scenes": sum(r["sensitive"] for r in results),
                        "sensitive_share_of_time": round(sum(r["duration"] for r in results if r["sensitive"]) / total, 3),
                        "activities": {a: sum(r["activity"] == a for r in results) for a in sorted({r["activity"] for r in results})}},
              "scenes": results, "seconds": round(time.time() - t0, 1)}
    save_json(out / "context.json", result)
    return result


def mm(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


if __name__ == "__main__":
    r = main(sys.argv[1])
    print(json.dumps(r["stats"], ensure_ascii=False, indent=2), r["seconds"], "s")
    for s in r["scenes"]:
        print(f'{mm(s["start"])}-{mm(s["end"])} {s["activity"]:16} {s["setting"]:15} {s["mood"]:6} '
              f'{"SENSITIVE" if s["sensitive"] else "ok":9} {"; ".join(s["reasons"])[:110]}')
