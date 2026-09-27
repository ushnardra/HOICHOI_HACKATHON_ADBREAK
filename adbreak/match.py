"""Checkpoint 9 (WHAT): match every selected break to the best SAFE synthetic brand.

For each break we look at the scene BEFORE and the scene AFTER it (from checkpoint 6).
1. Hard block (negative_contexts). A brand is removed if ANY of its negative phrases matches either scene,
   using three independent checks (any one is enough to block):
     a. synonym list  (config/brand_matching.json) -> scene category, e.g. "funeral" -> death, grief
     b. text meaning  (all-MiniLM-L6-v2): phrase -> every scene category within 0.10 of its best match
     c. picture       (SigLIP): the phrase itself compared with the scene's keyframes
   A category is "present" in a scene if checkpoint 6 marked it (sensitive reasons; mourning / sandwich scenes
   count as death + grief), or its picture score / keywords / activity say so.
2. Rank the remaining brands: dominant scene activity (text meaning of target_contexts vs the scene's activity)
   0.6 + picture match of target_contexts 0.4; scene before 0.7, scene after 0.3.
3. Final safety assert: an independent re-check of every placement; any violation raises an error.
Brands are read from config/brands.json at run time, so a new brand needs no code change.

Usage: python -m adbreak.match <video> [--extra-brand tests/data/brand_09_unseen.json]
"""
import argparse
import json
import re
import sys
import time

import numpy as np

from adbreak.common import MODELS, ROOT, device, dtype, free_gpu, load_json, save_json, video_path, work_dir
from adbreak.scenes import SIGLIP_DIR

BRANDS = ROOT / "config" / "brands.json"
SETTINGS = ROOT / "config" / "brand_matching.json"
LABELS = ROOT / "config" / "context_labels.json"
SENSITIVE = ("death", "grief", "illness", "violence")


class TextModel:
    def __init__(self, path):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.tok, self.m = torch, AutoTokenizer.from_pretrained(path), AutoModel.from_pretrained(path).eval()

    def __call__(self, texts):
        key = tuple(texts)
        if not hasattr(self, "_cache"):
            self._cache = {}
        if key not in self._cache:
            self._cache[key] = self._embed(texts)
        return self._cache[key]

    def _embed(self, texts):
        x = self.tok(list(texts), padding=True, truncation=True, return_tensors="pt")
        with self.torch.no_grad():
            o = self.m(**x).last_hidden_state
        e = (o * x["attention_mask"][..., None]).sum(1) / x["attention_mask"].sum(1, keepdim=True)
        return (e / e.norm(dim=-1, keepdim=True)).numpy()


class Picture:
    """SigLIP text side, compared with the cached keyframe embeddings of checkpoint 6."""

    def __init__(self, frame_emb):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.emb = torch, torch.from_numpy(frame_emb)
        self.proc = AutoProcessor.from_pretrained(SIGLIP_DIR)
        self.m = AutoModel.from_pretrained(SIGLIP_DIR, dtype=dtype()).to(device()).eval()

    def probs(self, phrases):
        key = tuple(phrases)
        if not hasattr(self, "_cache"):
            self._cache = {}
        if key not in self._cache:  # the same brand phrases come back at every break
            self._cache[key] = self._probs(phrases)
        return self._cache[key]

    def _probs(self, phrases):
        torch = self.torch
        with torch.no_grad():
            t = self.proc(text=[f"a photo of {p}." for p in phrases], padding="max_length", return_tensors="pt").to(device())
            te = self.m.get_text_features(**t)
            te = te if torch.is_tensor(te) else te.pooler_output
            te = (te / te.norm(dim=-1, keepdim=True)).float().cpu()
            logits = self.emb @ te.T * self.m.logit_scale.exp().float().cpu() + self.m.logit_bias.float().cpu()
        return torch.sigmoid(logits).numpy()  # frames x phrases


def category_descriptions(labels_cfg):
    out = {}
    for group in labels_cfg["groups"].values():
        for l in group:
            out[l["id"]] = l["id"].replace("_", " ") + ": " + ", ".join(l["prompts"])
    return out


def present_categories(scene, cfg):
    """Which categories are present in a checkpoint-6 scene (for the hard block).
    Sensitive categories come ONLY from checkpoint 6's own decision (its reasons / mourning / sandwich rules);
    ordinary categories come from picture tags, keywords and the activity."""
    pres = {}
    if scene["sensitive"]:
        for r in scene["reasons"]:
            for c in SENSITIVE:
                if r.startswith(f"picture: {c} ") or r.startswith(f"words: {c} ") or f"; picture: {c} " in r:
                    pres[c] = r
        if scene.get("sensitive_by") in ("mourning after a sensitive scene", "between two sensitive scenes"):
            pres.setdefault("death", scene["sensitive_by"])
            pres.setdefault("grief", scene["sensitive_by"])
        if not pres:  # sensitive for a reason we did not parse -> be safe
            for c in SENSITIVE:
                pres[c] = "scene is sensitive"
    for t in scene["tags"]:
        if t["label"] not in SENSITIVE and t["mean"] >= cfg["scene_present_mean"]:
            pres.setdefault(t["label"], f"picture {t['label']} {t['mean']:.3f}")
    for lab in scene.get("keywords", {}):
        if lab not in SENSITIVE:
            pres.setdefault(lab, f"words: {lab}")
    if scene["activity"] not in ("unclear", "title_card"):
        pres.setdefault(scene["activity"], f"activity {scene['activity']}")
    return pres


def map_phrase(phrase, cfg, text_model, cat_names, cat_emb):
    """Scene categories a (brand) phrase refers to: synonym list + text meaning.
    Sensitive categories (death, grief, illness, violence) are matched very conservatively: any synonym word in
    the phrase, or any sensitive category with similarity >= 0.25 within 0.15 of the best text match. Ordinary categories (eating, party...)
    need the WHOLE phrase to mean them, so "funeral meal" -> death/grief, but not "eating"."""
    p = phrase.lower().strip()
    words = p.split()
    cats = {}
    for c, syns in cfg["negative_synonyms"].items():
        if c == "about":
            continue
        if c in SENSITIVE:
            if any(w == p or w in words or (" " in w and w in p) for w in syns):
                cats[c] = "synonym list"
        elif p in syns or all(w in syns for w in words):
            cats[c] = "synonym list"
    sim = (text_model([phrase]) @ cat_emb.T)[0]
    best = int(sim.argmax())
    for i, (c, s) in enumerate(zip(cat_names, sim)):
        near = s >= cfg["text_sensitive_min"] and s >= sim[best] - cfg["text_sensitive_margin"]
        if (c in SENSITIVE and near) or (i == best and s >= cfg["text_ordinary_min"]):
            cats.setdefault(c, f"text meaning {s:.2f}")
    return cats


def main(name, extra_brand=None, breaks=None, save=True, log=print):
    video = video_path(name)
    out = work_dir(video.name)
    t0 = time.time()
    cfg = load_json(SETTINGS)
    brands = load_json(BRANDS)["brands"]
    if extra_brand:
        brands = brands + [load_json(extra_brand)["brand"]]
    ctx = {s["id"]: s for s in load_json(out / "context.json")["scenes"]}
    pac = load_json(out / "pacing.json")
    frames = load_json(out / "keyframes384.json")["frames"]
    scenes5 = {s["id"]: s for s in load_json(out / "scenes.json")["scenes"]}
    frame_shot = np.array([int(s) for s in sorted(frames, key=int) for _ in frames[s]])

    text_model = TextModel(MODELS / cfg["text_model"])
    descs = category_descriptions(load_json(LABELS))
    cat_names = list(descs)
    cat_emb = text_model(descs.values())
    picture = Picture(np.load(out / "siglip_frame_embeddings.npy"))

    def scene_rows(sid):
        a, b = scenes5[sid]["shots"]
        return np.where((frame_shot >= a) & (frame_shot <= b))[0]

    placements = []
    for brk in (breaks if breaks is not None else pac["selected"]):
        sides = {"before": ctx[brk["scene_before"]["id"]], "after": ctx[brk["scene_after"]["id"]]}
        present = {k: present_categories(s, cfg) for k, s in sides.items()}
        rows = {k: scene_rows(s["id"]) for k, s in sides.items()}
        results = []
        for br in brands:
            blocks = []
            neg = br["negative_contexts"]
            neg_probs = picture.probs(neg) if neg else np.zeros((len(frame_shot), 0))
            for i, phrase in enumerate(neg):
                cats = map_phrase(phrase, cfg, text_model, cat_names, cat_emb)
                for side, pres in present.items():
                    for c, how in cats.items():
                        if c in pres:
                            blocks.append(f"'{phrase}' → {c} ({how}); scene {side} the break has {c} ({pres[c]})")
                    pr = neg_probs[rows[side], i] if len(rows[side]) else np.zeros(1)
                    if pr.mean() >= cfg["picture_mean"] or (pr >= cfg["picture_frame_prob"]).mean() >= cfg["picture_frame_share"]:
                        blocks.append(f"'{phrase}' seen in the pictures of the scene {side} the break (mean {pr.mean():.3f})")
            # ranking (only meaningful if not blocked, but computed for everyone for the debug output)
            t_emb = text_model(br["target_contexts"])
            tp = picture.probs(br["target_contexts"])
            side_scores = {}
            for side, s in sides.items():
                act = s["activity"] if s["activity"] in descs else None
                a_sim = float((t_emb @ text_model([descs[act]]).T).max()) if act else 0.0
                v = float(tp[rows[side]].max(1).mean()) if len(rows[side]) else 0.0
                side_scores[side] = {"activity": act, "activity_match": round(a_sim, 3), "picture_match": round(v, 4)}
            results.append({"brand": br["id"], "name": br["name"], "category": br["category"],
                            "blocked": bool(blocks), "blocked_because": sorted(set(blocks)), "sides": side_scores})
        # picture matches are small numbers: scale them against the best brand at this break
        w = cfg["rank_weights"]
        for side in ("before", "after"):
            top_v = max(r["sides"][side]["picture_match"] for r in results) or 1e-9
            for r in results:
                r["sides"][side]["picture_rel"] = round(r["sides"][side]["picture_match"] / top_v, 3)
        for r in results:
            r["score"] = round(sum(w[f"scene_{side}"] * (w["dominant_activity"] * r["sides"][side]["activity_match"] +
                                                          w["picture"] * r["sides"][side]["picture_rel"])
                                   for side in ("before", "after")), 4)
        allowed = sorted([r for r in results if not r["blocked"]], key=lambda r: -r["score"])
        winner = allowed[0] if allowed else None
        placements.append({
            "t": brk["t"], "cut": brk["cut"],
            "scene_before": {"id": sides["before"]["id"], "activity": sides["before"]["activity"], "sensitive": sides["before"]["sensitive"]},
            "scene_after": {"id": sides["after"]["id"], "activity": sides["after"]["activity"], "sensitive": sides["after"]["sensitive"]},
            "present_categories": {k: sorted(v) for k, v in present.items()},
            "brand": winner["brand"] if winner else cfg["no_safe_brand"],
            "brand_name": winner["name"] if winner else "House promo (no safe brand)",
            "why": (f"best match for the dominant activity '{sides['before']['activity']}' (score {winner['score']:.3f}); "
                    f"{sum(r['blocked'] for r in results)} brand(s) blocked by negative contexts") if winner else
                   "every brand was blocked by its negative contexts",
            "ranking": [{k: r[k] for k in ("brand", "name", "score")} for r in allowed],
            "blocked": [{k: r[k] for k in ("brand", "name", "blocked_because")} for r in results if r["blocked"]],
            "all": results,
        })

    final_safety_assert(placements, brands, ctx, cfg, text_model, cat_names, cat_emb, picture, scene_rows)
    result = {"video": video.name, "brands_file": str(BRANDS.relative_to(ROOT)), "extra_brand": str(extra_brand) if extra_brand else None,
              "brands": [b["id"] for b in brands], "placements": placements,
              "stats": {"breaks": len(placements), "house_promos": sum(p["brand"] == cfg["no_safe_brand"] for p in placements),
                        "blocked_pairs": sum(len(p["blocked"]) for p in placements)},
              "safety_assert": "PASS", "seconds": round(time.time() - t0, 1)}
    if save:
        save_json(out / ("match.json" if not extra_brand else "match_with_extra_brand.json"), result)
    return result


def final_safety_assert(placements, brands, ctx, cfg, text_model, cat_names, cat_emb, picture=None, rows_of=None):
    """Independent re-check of every placement with the same rules, written separately from the matcher:
    brand negative phrase -> sensitive categories (synonyms, or text similarity >= 0.25 within 0.15 of the best)
    must not meet a sensitive category that checkpoint 6 found in the scene before or after the break;
    and the phrase must not be clearly visible in either scene's pictures. Any violation raises an error."""
    by_id = {b["id"]: b for b in brands}
    idx = {c: i for i, c in enumerate(cat_names)}
    for p in placements:
        br = by_id.get(p["brand"])
        if br is None:
            continue  # house promo
        for side in ("scene_before", "scene_after"):
            s = ctx[p[side]["id"]]
            found = set()
            if s["sensitive"]:
                # only real evidence counts ("picture: violence ...", "words: death ..."), not words in a description
                found = {m for r in s["reasons"] for m in re.findall(r"(?:picture|words): (death|grief|illness|violence)\b", r)}
                if s.get("sensitive_by") in ("mourning after a sensitive scene", "between two sensitive scenes"):
                    found |= {"death", "grief"}
                if not found:
                    found = set(SENSITIVE)
            for phrase in br["negative_contexts"]:
                sim = (text_model([phrase]) @ cat_emb.T)[0]
                top = max(sim[idx[c]] for c in cat_names)
                cats = {c for c in SENSITIVE if c in idx and sim[idx[c]] >= 0.25 and sim[idx[c]] >= top - 0.15}
                low = phrase.lower()
                cats |= {c for c in SENSITIVE if any(w in low.split() or (" " in w and w in low)
                                                     for w in cfg["negative_synonyms"].get(c, []))}
                if cats & found:
                    raise AssertionError(f"SAFETY VIOLATION at {p['t']}: {br['name']} negative '{phrase}' "
                                         f"matches {cats & found} in {side}")
                if picture is not None and rows_of is not None:
                    r = rows_of(s["id"])
                    pr = picture.probs([phrase])[r, 0] if len(r) else None
                    if pr is not None and (pr.mean() >= cfg["picture_mean"] or
                                           (pr >= cfg["picture_frame_prob"]).mean() >= cfg["picture_frame_share"]):
                        raise AssertionError(f"SAFETY VIOLATION at {p['t']}: '{phrase}' visible in {side} for {br['name']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="video name in assets/ (without .mp4)")
    ap.add_argument("--extra-brand", default=None)
    a = ap.parse_args()
    r = main(a.name, a.extra_brand)
    f = lambda t: f"{int(t // 60)}:{t % 60:04.1f}"
    print(json.dumps(r["stats"]), "safety assert:", r["safety_assert"], r["seconds"], "s")
    for p in r["placements"]:
        print(f"\n{f(p['t'])}  before={p['scene_before']['activity']}{'(S)' if p['scene_before']['sensitive'] else ''} "
              f"after={p['scene_after']['activity']}{'(S)' if p['scene_after']['sensitive'] else ''}  ->  {p['brand_name']}")
        print("   ranking:", ", ".join(f"{x['name']} {x['score']:.3f}" for x in p["ranking"]))
        for b in p["blocked"]:
            print(f"   BLOCKED {b['name']}: {b['blocked_because'][0][:120]}")
