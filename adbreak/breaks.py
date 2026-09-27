"""Checkpoint 7 (WHERE): for every scene change, find the exact moment where an ad break feels natural,
or reject the scene change with the reason.

Search window: from 2.5 s before the scene change to 1.0 s after it, in 0.05 s steps. A moment t is VALID only if
  - checkpoint 3 says SILENCE at t (with 0.2 s margin)       -> no speech, and no "maybe" (unsure) either
  - checkpoint 4 says t is not inside a word (0.15 s margin)
  - the last word ended at least 0.4 s before t             -> the sentence is finished
  - the next word starts at least 0.25 s after t            -> the ad does not start as someone begins to talk
  - NO SONG / MUSIC runs through t (adbreak/music.py): music on both sides, no dip at t, same piece before
    and after -> vetoed. A camera cut during the same song is NOT a break opportunity; a break is only allowed
    once the music has ended, paused, or clearly changed. (Singing gaps and instrumental passages look like
    "silence" to the speech checks - this veto closes that hole.)
Among valid moments the best score wins (0-1):
  0.25 x silence length around t (2 s or more = full marks)
  0.20 x sentence finished (last word ends a sentence, or 1 s+ of quiet)
  0.20 x scene change strength (checkpoint 5; a black/fade = full marks)
  0.15 x quiet: loudness at t (+/-0.25 s) vs the video's typical speech loudness. "No talking" is not the
         same as quiet - loud background music at the break feels abrupt (found by an independent loudness check)
  0.10 x background sound changes across the cut (music/ambience not running through)
  0.10 x closeness to the camera cut (the picture changes right at the break)

Usage: python -m adbreak.breaks <video>
"""
import json
import sys
import time

import numpy as np

from adbreak import music, speech, transcribe
from adbreak.common import load_json, save_json, video_path, work_dir

BEFORE_SEC, AFTER_SEC, STEP = 2.5, 1.0, 0.05
VAD_MARGIN, WORD_MARGIN = 0.2, 0.15
MIN_SINCE_LAST_WORD, MIN_TO_NEXT_WORD = 0.4, 0.25
W = {"silence": 0.25, "sentence": 0.20, "scene": 0.20, "quiet": 0.15, "sound": 0.10, "cut": 0.10}
LOUD = 1.2   # loudness above 1.2 x typical speech at the break is flagged as "loud music / sound"


def silence_around(t, segments):
    before = max((s["end"] for s in segments if s["end"] <= t), default=0.0)
    after = min((s["start"] for s in segments if s["start"] >= t), default=t + 5)
    return after - before, before, after


def check_moment(t, sp, tr, mus=None):
    """(ok, reason) for a single moment. mus = (timeline, song_sequences, cfg) from adbreak.music, or None."""
    st = speech.speaking_at(t, sp, margin=VAD_MARGIN)
    if st != "silence":
        return False, "someone is speaking" if st == "speech" else "unsure if someone is speaking (maybe)"
    w = transcribe.in_word(t, tr, margin=WORD_MARGIN)
    if w:
        return False, f"inside the word '{w['word']}'"
    prev = [x for x in tr["words"] if x["end"] <= t]
    if prev and t - prev[-1]["end"] < MIN_SINCE_LAST_WORD:
        return False, f"only {t - prev[-1]['end']:.2f} s after the last word"
    nxt = [x for x in tr["words"] if x["start"] >= t]
    if nxt and nxt[0]["start"] - t < MIN_TO_NEXT_WORD:
        return False, f"a word starts {nxt[0]['start'] - t:.2f} s later"
    if mus is not None:
        veto, d = music.continuity(t, *mus)
        if veto:
            a, b = d["inside_song"]
            return False, (f"inside a continuing song / music sequence ({int(a // 60)}:{int(a % 60):02d}-"
                           f"{int(b // 60)}:{int(b % 60):02d}) with no pause here")
    return True, ""


def main(name):
    video = video_path(name)
    out = work_dir(video.name)
    t0 = time.time()
    info = load_json(out / "info.json")
    sc = load_json(out / "scenes.json")
    ctx = {s["id"]: s for s in load_json(out / "context.json")["scenes"]}
    sp = speech.load(video.name)
    tr = load_json(out / "transcript.json")
    cuts = {c["t"]: c for c in sc["cuts"]}
    audio = speech.load_audio(out / "audio.wav")
    rms = lambda a, b: float(np.sqrt(np.mean(audio[max(0, int(a * speech.SR)):int(b * speech.SR)] ** 2)))
    speech_rms = float(np.median([rms(x["start"], x["end"]) for x in sp["segments"]]))
    snd = np.array([c["sound_change"] for c in sc["cuts"]])
    snd_med, snd_iqr = np.median(snd), np.percentile(snd, 75) - np.percentile(snd, 25) + 1e-9
    mus = None
    if music.ready():  # song / music continuity veto
        if not (out / "music.npz").exists():
            music.main(video.name)
        mus = (*music.load(video.name), load_json(music.CONFIG))  # (timeline, song sequences, config)

    results = []
    for k in range(1, len(sc["scenes"])):
        s = sc["scenes"][k]
        cut = s["start"]
        c = cuts.get(round(cut, 3)) or min(sc["cuts"], key=lambda x: abs(x["t"] - cut))
        scene_strength = 1.0 if c["black"] else float(np.clip(c["score"] / 4.0, 0, 1))
        sound = float(np.clip(((c["sound_change"] - snd_med) / snd_iqr + 1) / 3, 0, 1))
        naive_ok, naive_why = check_moment(cut, sp, tr, mus)
        best, why_not = None, {}
        for t in np.arange(max(0.0, cut - BEFORE_SEC), min(info["duration_sec"], cut + AFTER_SEC) + 1e-9, STEP):
            t = round(float(t), 2)
            ok, why = check_moment(t, sp, tr, mus)
            if not ok:
                key = why.split(" (")[0].split(" '")[0].split(" 0")[0]
                why_not[key] = why_not.get(key, 0) + 1
                continue
            gap, g0, g1 = silence_around(t, sp["segments"])
            prev = [x for x in tr["words"] if x["end"] <= t]
            quiet = t - prev[-1]["end"] if prev else 99.0
            sentence = 1.0 if (quiet >= 1.0 or (prev and prev[-1]["sentence_end"])) else 0.5
            loud = rms(t - 0.25, t + 0.25) / speech_rms
            parts = {"silence": min(gap, 2.0) / 2.0, "sentence": sentence, "scene": scene_strength,
                     "quiet": float(np.clip(1 - loud / 1.5, 0, 1)), "sound": sound,
                     "cut": 1 - abs(t - cut) / max(BEFORE_SEC, AFTER_SEC)}
            score = sum(W[p] * v for p, v in parts.items())
            if best is None or score > best["score"] + 1e-9:
                best = {"t": t, "score": round(score, 3), "parts": {p: round(v, 2) for p, v in parts.items()},
                        "silence_sec": round(gap, 2), "since_last_word": round(min(quiet, 99.0), 2),
                        "offset_from_cut": round(t - cut, 2), "loudness_vs_speech": round(loud, 2)}
        reasons = []
        if best:
            p = best["parts"]
            reasons.append(f"silence {best['silence_sec']:.1f} s")
            reasons.append("sentence finished" if p["sentence"] == 1.0 else "pause after a word (sentence may go on)")
            reasons.append("black / fade" if c["black"] else f"scene change strength {c['score']:.1f}")
            if best["loudness_vs_speech"] >= LOUD:
                reasons.append(f"WARNING: loud music / sound at the break ({best['loudness_vs_speech'] * 100:.0f}% of speech loudness)")
            if abs(best["offset_from_cut"]) > 0.05:
                reasons.append(f"moved {best['offset_from_cut']:+.2f} s from the camera cut to reach silence")
            if mus is not None:
                _, d = music.continuity(best["t"], *mus)
                best["music"] = d
                if d["inside_song"] and d["music_pauses_here"]:
                    reasons.append("inside a song, but the music clearly pauses here")
                elif any(0 <= best["t"] - b < 3 for _, b in mus[1]):
                    reasons.append("a song has just ended")
        results.append({
            "scene_change": k, "cut": round(cut, 2), "strong": s["starts_with"] in ("strong", "black"),
            "naive_cut_ok": naive_ok, "naive_cut_problem": naive_why or None,
            "accepted": best is not None, "break": best,
            "reasons": reasons,
            "rejected_because": None if best else "no valid silent moment within 2.5 s before / 1 s after the scene change: "
                                                  + ", ".join(f"{k2} ({v})" for k2, v in sorted(why_not.items(), key=lambda x: -x[1])),
            "scene_before": {"id": k - 1, "activity": ctx[k - 1]["activity"], "sensitive": ctx[k - 1]["sensitive"]},
            "scene_after": {"id": k, "activity": ctx[k]["activity"], "sensitive": ctx[k]["sensitive"]},
        })
    acc = [r for r in results if r["accepted"]]
    result = {"video": video.name,
              "settings": {"window_before_sec": BEFORE_SEC, "window_after_sec": AFTER_SEC, "step_sec": STEP,
                           "vad_margin": VAD_MARGIN, "word_margin": WORD_MARGIN,
                           "min_since_last_word": MIN_SINCE_LAST_WORD, "min_to_next_word": MIN_TO_NEXT_WORD, "weights": W,
                           "loud_flag": LOUD},
              "stats": {"scene_changes": len(results), "accepted": len(acc),
                        "rejected": len(results) - len(acc),
                        "naive_cut_would_interrupt_speech": sum(not r["naive_cut_ok"] for r in results),
                        "moved_to_reach_silence": sum(abs(r["break"]["offset_from_cut"]) > 0.05 for r in acc),
                        "loud_at_break": sum(r["break"]["loudness_vs_speech"] >= LOUD for r in acc),
                        "music_veto_active": mus is not None,
                        "rejected_by_music": sum(1 for r in results if not r["accepted"] and "song" in (r["rejected_because"] or "")),
                        "song_sequences": mus[1] if mus is not None else [],
                        "naive_cut_inside_song": sum(1 for r in results if r["naive_cut_problem"] and "song" in r["naive_cut_problem"]),
                        "median_score": round(float(np.median([r["break"]["score"] for r in acc])), 3) if acc else None},
              "candidates": results, "seconds": round(time.time() - t0, 1)}
    save_json(out / "breaks.json", result)
    return result


if __name__ == "__main__":
    r = main(sys.argv[1])
    print(json.dumps(r["stats"], indent=2), r["seconds"], "s")
    f = lambda t: f"{int(t // 60)}:{t % 60:04.1f}"
    for c in r["candidates"]:
        if c["accepted"]:
            b = c["break"]
            print(f"{f(c['cut']):8} -> BREAK {f(b['t']):8} score {b['score']:.2f}  {'; '.join(c['reasons'])}")
        else:
            print(f"{f(c['cut']):8} -> REJECTED  {c['rejected_because'][:120]}")
